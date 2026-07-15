"""Phase 4.6 diagnostic: isolate the XNNPACK-partitioner segfault. Tests to_edge_transform_and_lower
with the XNNPACK partitioner on ONE method alone (not a {"prefill":..., "decode":...} dict), using
random-init weights (fast, no download) -- since the crash is structural (graph shape/ops), not
weight-value dependent. If this does NOT crash, the multi-method dict export is implicated. If it
DOES crash, the custom op (or general model structure) is implicated, independent of multi-method.

    .venv-executorch/Scripts/python.exe diagnose_xnnpack_segfault.py
"""
import json
import sys

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import register_kernel_impl  # noqa: E402
from export_llama import WindowedLlamaForExport  # noqa: E402

register_kernel_impl(kernel_module)

arms_data = json.load(open("llama_3.2_1b_arms.json"))
arm = arms_data["arms"]["map50"]
head_windowed_per_layer = arm["head_windowed"]
n_layers = arms_data["n_layers"]
SINK, WINDOW = arms_data["sink"], arms_data["window"]
MAX_T = 64 + 256

config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
)
base_model = LlamaModel(config).eval()
model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, MAX_T).eval()

decode_inputs = (
    torch.randint(0, config.vocab_size, (1, 1)),
    torch.tensor([64], dtype=torch.long),
)

print("torch.export decode (single method, not a dict)...", flush=True)
decode_exported = torch.export.export(model, decode_inputs)
print("export OK", flush=True)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

print("to_edge_transform_and_lower (SINGLE ExportedProgram, not dict)...", flush=True)
edge_program = to_edge_transform_and_lower(
    decode_exported,  # single program, NOT {"decode": decode_exported}
    partitioner=[XnnpackDynamicallyQuantizedPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
print("to_edge OK -- no crash with single-method export", flush=True)
