// Phase 4.5: loads a two-method (prefill/decode) .pte via ExecuTorch's Module facade, runs a
// fixed token sequence through prefill + N decode steps, timing each call and dumping the
// output hidden states -- so numeric correctness (vs. an eager PC-side reference) and per-step
// latency (windowed vs. dense) can both be checked from the same run.
//
// Input/output construction follows the same pattern as ExecuTorch's own
// extension/llm/runner/text_decoder_runner.cpp (TextDecoderRunner::step): build TensorPtrs via
// from_blob, wrap in a std::vector<EValue>, call module.execute(method_name, inputs).
//
// Token file format (binary, little-endian, written by make_test_tokens.py):
//   int64 n_tokens
//   int64 tokens[n_tokens]
//   int64 prefill_len
//
// Output file format (binary):
//   int64 hidden_size
//   int64 prefill_len
//   float prefill_out[prefill_len * hidden_size]        -- prefill's output hidden states
//   int64 n_decode
//   float decode_out[n_decode * hidden_size]             -- one row per decode step
//   double prefill_ms
//   double decode_ms[n_decode]
//
//   adb shell /data/local/tmp/prefill_decode_runner <pte_path> <tokens_path> <out_path>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <pte_path> <tokens_path> <out_path>\n";
    return 1;
  }
  const std::string pte_path = argv[1];
  const std::string tokens_path = argv[2];
  const std::string out_path = argv[3];

  std::ifstream tin(tokens_path, std::ios::binary);
  if (!tin) {
    std::cerr << "failed to open tokens file: " << tokens_path << "\n";
    return 1;
  }
  int64_t n_tokens = 0;
  tin.read(reinterpret_cast<char*>(&n_tokens), sizeof(int64_t));
  std::vector<int64_t> tokens(n_tokens);
  tin.read(reinterpret_cast<char*>(tokens.data()), n_tokens * sizeof(int64_t));
  int64_t prefill_len = 0;
  tin.read(reinterpret_cast<char*>(&prefill_len), sizeof(int64_t));
  std::cout << "loaded " << n_tokens << " tokens, prefill_len=" << prefill_len << "\n";

  std::cout << "loading module (mmap) from " << pte_path << " ...\n";
  // share_memory_arenas=true: required so prefill and decode -- independently exported
  // methods -- share the same underlying k_cache/v_cache buffers at runtime instead of each
  // getting its own zero-initialized copy (must match share_mutable_buffers=True on the export
  // side; see export_llama.py's to_executorch() call for the full explanation).
  Module module(pte_path, Module::LoadMode::Mmap, nullptr, nullptr, nullptr, /*share_memory_arenas=*/true);
  auto load_err = module.load();
  if (load_err != executorch::runtime::Error::Ok) {
    std::cerr << "module.load() failed: " << static_cast<int>(load_err) << "\n";
    return 1;
  }
  std::cout << "module loaded.\n";

  // --- Prefill ---
  std::vector<int64_t> prefill_ids(tokens.begin(), tokens.begin() + prefill_len);
  TensorPtr prefill_ids_t = from_blob(
      prefill_ids.data(),
      {1, static_cast<executorch::aten::SizesType>(prefill_len)},
      ScalarType::Long);
  int64_t start_pos0 = 0;
  TensorPtr start_pos0_t = from_blob(&start_pos0, {1}, ScalarType::Long);

  std::cout << "running prefill...\n";
  auto t0 = std::chrono::steady_clock::now();
  std::vector<EValue> prefill_inputs{prefill_ids_t, start_pos0_t};
  auto prefill_res = module.execute("prefill", prefill_inputs);
  auto t1 = std::chrono::steady_clock::now();
  if (!prefill_res.ok()) {
    std::cerr << "prefill execute() failed: " << static_cast<int>(prefill_res.error()) << "\n";
    return 1;
  }
  double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto prefill_out_tensor = prefill_res.get()[0].toTensor();
  int64_t hidden_size = prefill_out_tensor.size(prefill_out_tensor.dim() - 1);
  // Copy immediately -- with share_memory_arenas=true, this output aliases a shared activation
  // arena that the decode calls below will overwrite (Module's own docs: "outputs from one
  // method may be invalidated by executing another method... consume or copy outputs before
  // calling execute again"). Holding a live Tensor view across the decode loop and reading it
  // only at the end (as this used to do) reads stale/overwritten data.
  std::vector<float> prefill_out_flat(
      prefill_out_tensor.const_data_ptr<float>(),
      prefill_out_tensor.const_data_ptr<float>() + prefill_len * hidden_size);
  std::cout << "prefill OK, " << prefill_ms << " ms, hidden_size=" << hidden_size << "\n";

  // --- Decode loop: one step per remaining token ---
  int64_t n_decode = n_tokens - prefill_len;
  std::vector<float> decode_out_flat(static_cast<size_t>(n_decode * hidden_size));
  std::vector<double> decode_ms(n_decode);

  for (int64_t i = 0; i < n_decode; i++) {
    int64_t pos = prefill_len + i;
    int64_t tok = tokens[pos];
    TensorPtr decode_id_t = from_blob(&tok, {1, 1}, ScalarType::Long);
    TensorPtr pos_t = from_blob(&pos, {1}, ScalarType::Long);

    auto d0 = std::chrono::steady_clock::now();
    std::vector<EValue> decode_inputs{decode_id_t, pos_t};
    auto decode_res = module.execute("decode", decode_inputs);
    auto d1 = std::chrono::steady_clock::now();
    if (!decode_res.ok()) {
      std::cerr << "decode step " << i << " execute() failed: "
                << static_cast<int>(decode_res.error()) << "\n";
      return 1;
    }
    decode_ms[i] = std::chrono::duration<double, std::milli>(d1 - d0).count();
    auto out_tensor = decode_res.get()[0].toTensor();
    const float* out_data = out_tensor.const_data_ptr<float>();
    std::copy(out_data, out_data + hidden_size, decode_out_flat.begin() + i * hidden_size);
    std::cout << "decode step " << i << " (pos=" << pos << "): " << decode_ms[i] << " ms\n";
  }

  // --- Write results ---
  std::ofstream fout(out_path, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int64_t));
  fout.write(reinterpret_cast<const char*>(&prefill_len), sizeof(int64_t));
  fout.write(reinterpret_cast<const char*>(prefill_out_flat.data()),
             prefill_out_flat.size() * sizeof(float));
  fout.write(reinterpret_cast<const char*>(&n_decode), sizeof(int64_t));
  fout.write(reinterpret_cast<const char*>(decode_out_flat.data()),
             decode_out_flat.size() * sizeof(float));
  fout.write(reinterpret_cast<const char*>(&prefill_ms), sizeof(double));
  fout.write(reinterpret_cast<const char*>(decode_ms.data()), n_decode * sizeof(double));
  fout.close();
  std::cout << "wrote results to " << out_path << "\n";

  return 0;
}
