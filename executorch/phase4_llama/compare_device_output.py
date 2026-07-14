"""Phase 4.5: eager PC-side reference for the same fixed token sequence in test_tokens.bin,
compared against the on-device runner's pulled-back results -- the real correctness gate (does
the ARM-compiled kernel + real weights produce numerically correct output, not just "does it
run"). Also reports the device's own prefill/decode timing.

    .venv-executorch/Scripts/python.exe compare_device_output.py --arm map50
"""
import argparse
import struct
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402

from real_attention_patch import RealWindowedLlamaAttention, register_kernel_impl  # noqa: E402
from export_llama import WindowedLlamaForExport  # noqa: E402

register_kernel_impl(kernel_module)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="map50")
parser.add_argument("--tokens", default="test_tokens.bin")
parser.add_argument("--device-out", default="device_out.bin")
parser.add_argument("--tol", type=float, default=1e-2)
args = parser.parse_args()

with open(args.tokens, "rb") as f:
    (n_tokens,) = struct.unpack("<q", f.read(8))
    tokens = list(struct.unpack(f"<{n_tokens}q", f.read(8 * n_tokens)))
    (prefill_len,) = struct.unpack("<q", f.read(8))
n_decode = n_tokens - prefill_len
print(f"tokens: {n_tokens}, prefill_len={prefill_len}, n_decode={n_decode}")

with open(args.device_out, "rb") as f:
    (hidden_size,) = struct.unpack("<q", f.read(8))
    (dev_prefill_len,) = struct.unpack("<q", f.read(8))
    assert dev_prefill_len == prefill_len
    n_pf = prefill_len * hidden_size
    dev_prefill_out = torch.tensor(struct.unpack(f"<{n_pf}f", f.read(4 * n_pf))).view(prefill_len, hidden_size)
    (dev_n_decode,) = struct.unpack("<q", f.read(8))
    assert dev_n_decode == n_decode
    n_dc = n_decode * hidden_size
    dev_decode_out = torch.tensor(struct.unpack(f"<{n_dc}f", f.read(4 * n_dc))).view(n_decode, hidden_size)
    (prefill_ms,) = struct.unpack("<d", f.read(8))
    decode_ms = list(struct.unpack(f"<{n_decode}d", f.read(8 * n_decode)))

print(f"device timing: prefill {prefill_ms:.2f} ms, decode {[f'{x:.2f}' for x in decode_ms]} ms")

print(f"\nLoading real Llama-3.2-1B for eager reference (arm={args.arm})...")
import json
arms_data = json.load(open("llama_3.2_1b_arms.json"))
head_windowed_per_layer = arms_data["arms"][args.arm]["head_windowed"]
SINK, WINDOW = arms_data["sink"], arms_data["window"]

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32).model.eval()
model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, prefill_len + n_decode + 64).eval()

with torch.no_grad():
    input_ids = torch.tensor([tokens[:prefill_len]], dtype=torch.long)
    ref_prefill_out = model(input_ids, torch.zeros(1, dtype=torch.long))[0]  # (prefill_len, hidden)

    ref_decode_out = []
    for i in range(n_decode):
        pos = prefill_len + i
        tok_id = torch.tensor([[tokens[pos]]], dtype=torch.long)
        pos_t = torch.tensor([pos], dtype=torch.long)
        out = model(tok_id, pos_t)[0, 0]  # (hidden,)
        ref_decode_out.append(out)
    ref_decode_out = torch.stack(ref_decode_out)

prefill_diff = (dev_prefill_out - ref_prefill_out).abs().max().item()
decode_diff = (dev_decode_out - ref_decode_out).abs().max().item()
print(f"\nprefill max abs diff (device vs. eager reference): {prefill_diff:.4e}")
print(f"decode  max abs diff (device vs. eager reference): {decode_diff:.4e}")

ok = prefill_diff < args.tol and decode_diff < args.tol
print(f"\n{'PASS' if ok else 'FAIL'}: device output {'matches' if ok else 'does NOT match'} "
      f"eager reference within tol={args.tol}")
sys.exit(0 if ok else 1)
