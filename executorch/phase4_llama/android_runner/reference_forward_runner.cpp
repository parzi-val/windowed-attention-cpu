// Phase 4.5 control experiment: times ExecuTorch's own official reference Llama-3.2-1B export
// (executorch-community/Llama-3.2-1B-ET on HuggingFace -- standard SDPA + KV-cache, XNNPACK
// delegate, no custom op) on the SAME phone, SAME token sequence, SAME timing methodology as
// prefill_decode_runner.cpp, to determine whether our windowed_sdpa_kv_cache op is the source
// of decode's ~25-30s/step slowness or whether that's generic to this hardware/pipeline.
//
// Their .pte exports a single "forward(tokens, input_pos)" method with dynamic shapes (not
// separate prefill/decode methods like ours) -- confirmed via method_meta inspection. Called
// once with the 64-token prompt batch (prefill-equivalent), then once per remaining token
// (decode-equivalent), reusing the exact same test_tokens.bin split as our own runner.
//
//   adb shell /data/local/tmp/reference_forward_runner <pte_path> <tokens_path> <out_path>

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
  Module module(pte_path, Module::LoadMode::Mmap);
  auto load_err = module.load();
  if (load_err != executorch::runtime::Error::Ok) {
    std::cerr << "module.load() failed: " << static_cast<int>(load_err) << "\n";
    return 1;
  }
  std::cout << "module loaded.\n";

  // --- Prefill-equivalent: forward() with the full prompt batch, start_pos=0 ---
  std::vector<int64_t> prefill_ids(tokens.begin(), tokens.begin() + prefill_len);
  TensorPtr prefill_ids_t = from_blob(
      prefill_ids.data(),
      {1, static_cast<executorch::aten::SizesType>(prefill_len)},
      ScalarType::Long);
  int64_t start_pos0 = 0;
  TensorPtr start_pos0_t = from_blob(&start_pos0, {1}, ScalarType::Long);

  std::cout << "running forward (prefill-equivalent)...\n";
  auto t0 = std::chrono::steady_clock::now();
  std::vector<EValue> prefill_inputs{prefill_ids_t, start_pos0_t};
  auto prefill_res = module.execute("forward", prefill_inputs);
  auto t1 = std::chrono::steady_clock::now();
  if (!prefill_res.ok()) {
    std::cerr << "forward (prefill) execute() failed: " << static_cast<int>(prefill_res.error()) << "\n";
    return 1;
  }
  double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "prefill-equivalent OK, " << prefill_ms << " ms\n";

  // --- Decode-equivalent: forward() once per remaining token, Tn=1 ---
  int64_t n_decode = n_tokens - prefill_len;
  std::vector<double> decode_ms(n_decode);

  for (int64_t i = 0; i < n_decode; i++) {
    int64_t pos = prefill_len + i;
    int64_t tok = tokens[pos];
    TensorPtr decode_id_t = from_blob(&tok, {1, 1}, ScalarType::Long);
    TensorPtr pos_t = from_blob(&pos, {1}, ScalarType::Long);

    auto d0 = std::chrono::steady_clock::now();
    std::vector<EValue> decode_inputs{decode_id_t, pos_t};
    auto decode_res = module.execute("forward", decode_inputs);
    auto d1 = std::chrono::steady_clock::now();
    if (!decode_res.ok()) {
      std::cerr << "forward (decode) step " << i << " execute() failed: "
                << static_cast<int>(decode_res.error()) << "\n";
      return 1;
    }
    decode_ms[i] = std::chrono::duration<double, std::milli>(d1 - d0).count();
    std::cout << "decode-equivalent step " << i << " (pos=" << pos << "): " << decode_ms[i] << " ms\n";
  }

  std::ofstream fout(out_path, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&prefill_ms), sizeof(double));
  fout.write(reinterpret_cast<const char*>(&n_decode), sizeof(int64_t));
  fout.write(reinterpret_cast<const char*>(decode_ms.data()), n_decode * sizeof(double));
  fout.close();
  std::cout << "wrote timing results to " << out_path << "\n";

  return 0;
}
