"""Phase 4.6 diagnostic, round 2: does XNNPACK partitioning crash on Phase 3's minimal
TinyKVCacheAttn (just windowed_sdpa_kv_cache + mutable buffers, no real Llama structure at
all)? Isolates whether the segfault is fundamentally about {custom mutating op + XNNPACK},
independent of everything else in the real model (RoPE, RMSNorm, embedding, MLP, ...).

    .venv-executorch/Scripts/python.exe diagnose_xnnpack_tiny.py
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import register_kernel_impl  # noqa: E402

register_kernel_impl(kernel_module)


class TinyKVCacheAttn(nn.Module):
    def __init__(self, B, H, D, sink, window, max_t):
        super().__init__()
        self.sink = sink
        self.window = window
        hw = torch.tensor([i % 2 == 0 for i in range(H)], dtype=torch.bool)
        self.register_buffer("head_windowed", hw)
        self.register_buffer("k_cache", torch.zeros(B, H, max_t, D))
        self.register_buffer("v_cache", torch.zeros(B, H, max_t, D))

    def forward(self, q_new, k_new, v_new):
        start_pos = 5
        return torch.ops.sg.windowed_sdpa_kv_cache(
            q_new, k_new, v_new, self.k_cache, self.v_cache,
            self.head_windowed, self.sink, self.window, start_pos,
        )


B, H, D, sink, window, max_t = 1, 4, 8, 4, 16, 64
model = TinyKVCacheAttn(B, H, D, sink, window, max_t).eval()
example_inputs = (torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), torch.randn(B, H, 1, D))

print("torch.export ...", flush=True)
exported = torch.export.export(model, example_inputs)
print("export OK", flush=True)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

print("to_edge_transform_and_lower (tiny model, XNNPACK DQ partitioner)...", flush=True)
edge_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackDynamicallyQuantizedPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
print("to_edge OK -- no crash on the tiny model", flush=True)
