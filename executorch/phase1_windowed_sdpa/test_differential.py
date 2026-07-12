"""Phase 1 differential test: the pure C++ kernel (windowed_sdpa_pybind, gather-based --
only ever touches the KV positions a query can actually see) against an independently
structured PyTorch reference (full T x T masked softmax -- computes every score then masks
out invalid positions). Same masking definition, deliberately different implementation
shape, so agreement is a real correctness signal rather than two copies of the same code.

    .venv-executorch/Scripts/python.exe test_differential.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "build")
import windowed_sdpa_pybind as kernel_module


def reference_windowed_attention(q, k, v, head_windowed, sink, window):
    """Full T x T masked-softmax reference. q,k,v: (B,H,T,D) float32. head_windowed: (H,) bool."""
    B, H, T, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,H,T,T)

    idx = torch.arange(T)
    causal = (idx[None, :] <= idx[:, None])  # (T,T): kv_idx <= q_idx
    win_valid = (idx[None, :] < sink) | (idx[None, :] > idx[:, None] - window)
    dense_mask = causal
    windowed_mask = causal & win_valid

    hw = head_windowed.view(1, H, 1, 1)
    mask = torch.where(hw, windowed_mask.view(1, 1, T, T), dense_mask.view(1, 1, T, T))
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def run_case(name, B, H, T, D, sink, window, frac_windowed, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)
    head_windowed = torch.rand(H) < frac_windowed

    ref = reference_windowed_attention(q, k, v, head_windowed, sink, window)

    kernel_out = kernel_module.windowed_sdpa(
        q.numpy(), k.numpy(), v.numpy(), head_windowed.numpy(), sink, window
    )
    kernel_out = torch.from_numpy(kernel_out)

    max_diff = (kernel_out - ref).abs().max().item()
    ok = max_diff < 1e-4
    status = "PASSED" if ok else "FAILED"
    print(
        f"[{status}] {name:32} B={B} H={H} T={T} D={D} sink={sink} window={window} "
        f"frac_windowed={frac_windowed:.2f}  max abs diff = {max_diff:.3e}"
    )
    return ok


cases = [
    ("all_dense",        1, 4, 40,  8, 4, 64, 0.0, 0),
    ("all_windowed",     1, 4, 40,  8, 4, 64, 1.0, 1),
    ("mixed_short_ctx",  1, 8, 40,  8, 4, 64, 0.5, 2),   # T < window: windowed == dense
    ("mixed_long_ctx",   1, 8, 300, 8, 4, 64, 0.5, 3),   # T > window: real windowing kicks in
    ("single_head",      1, 1, 200, 16, 4, 64, 1.0, 4),
    ("batch_gt_1",       3, 6, 150, 8, 4, 64, 0.5, 5),
    ("tiny_window",      1, 4, 100, 8, 2, 8, 0.5, 6),
    ("sink_gt_window_lo", 1, 4, 100, 8, 4, 64, 0.5, 7),  # exercises the sink/window overlap branch at small t
]

all_ok = all(run_case(*c) for c in cases)
print()
print("ALL PASSED" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
