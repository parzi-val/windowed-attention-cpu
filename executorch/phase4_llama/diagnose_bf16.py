"""Phase 4.7 diagnostic: does the fp32-cast-around-the-op fix in real_attention_patch.py
actually work for a real bf16 model, both eager and through torch.export? Uses random-init
weights (fast) cast to bfloat16, matching export_llama.py's new random-init bf16 path.

    .venv-executorch/Scripts/python.exe diagnose_bf16.py
"""
import sys
import torch

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import register_kernel_impl  # noqa: E402
from export_llama import WindowedLlamaForExport  # noqa: E402

register_kernel_impl(kernel_module)

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

n_layers = 2
config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
)
base_model = LlamaModel(config).to(torch.bfloat16).eval()
MAX_T = 64 + 256
SINK, WINDOW = 4, 64
head_windowed_per_layer = [[i % 2 == 0 for i in range(32)] for _ in range(n_layers)]
model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, MAX_T).eval()

print("eager forward (bf16 model)...", flush=True)
prefill_inputs = (torch.randint(0, config.vocab_size, (1, 64)), torch.zeros(1, dtype=torch.long))
with torch.no_grad():
    out = model(*prefill_inputs)
print(f"  output dtype: {out.dtype}, shape: {tuple(out.shape)}")
assert out.dtype == torch.bfloat16, f"expected bf16 output, got {out.dtype}"
assert not torch.isnan(out).any(), "NaN in output -- dtype boundary bug"
print("  PASS: bf16 eager forward produces finite bf16 output", flush=True)

decode_inputs = (torch.randint(0, config.vocab_size, (1, 1)), torch.tensor([64], dtype=torch.long))
with torch.no_grad():
    out2 = model(*decode_inputs)
assert out2.dtype == torch.bfloat16 and not torch.isnan(out2).any()
print("  PASS: bf16 decode step also clean", flush=True)

print("\ntorch.export (bf16 model)...", flush=True)
exported = torch.export.export(model, decode_inputs)
print("  PASS: bf16 model exports cleanly", flush=True)
