"""Phase 4.6 diagnostic, round 4: bisect by layer count. Round 3 showed ONE real layer survives
XNNPACK partitioning cleanly (once FLATC_EXECUTABLE is set); the full 16-layer model still
segfaults even with flatc fixed, with no exception at all -- suggesting the crash happens
during partitioning itself (before flatc is ever invoked), scaling with graph size/layer count
rather than being caused by any single op. This tests N in {2, 4, 8} to find where it breaks.

    .venv-executorch/Scripts/python.exe diagnose_xnnpack_nlayers.py <n_layers>
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import RealWindowedLlamaAttention, register_kernel_impl  # noqa: E402
from export_llama import WindowedLlamaForExport  # noqa: E402

register_kernel_impl(kernel_module)

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
print(f"testing n_layers={n_layers}", flush=True)

config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
)
base_model = LlamaModel(config).eval()
MAX_T = 64 + 256
SINK, WINDOW = 4, 64
head_windowed_per_layer = [[i % 2 == 0 for i in range(32)] for _ in range(n_layers)]
model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, MAX_T).eval()

decode_inputs = (torch.randint(0, config.vocab_size, (1, 1)), torch.tensor([64], dtype=torch.long))

print("torch.export...", flush=True)
exported = torch.export.export(model, decode_inputs)
print("export OK", flush=True)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

print(f"to_edge_transform_and_lower (n_layers={n_layers})...", flush=True)
edge_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackDynamicallyQuantizedPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
print(f"to_edge OK -- n_layers={n_layers} survives", flush=True)
