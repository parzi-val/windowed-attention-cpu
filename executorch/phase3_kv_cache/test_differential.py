"""Phase 3 differential test: incremental KV-cache decode (windowed_sdpa_kv_cache_pybind)
against a monolithic full-sequence reference (independent PyTorch implementation, full T x T
masked softmax -- same reference shape used in Phase 1's test_differential.py).

The correctness claim: prefilling the cache with the first `prefill_len` tokens, then decoding
the remaining tokens one at a time (or in a second chunk) through the cache-based kernel, must
produce the same output as running the whole sequence through the monolithic reference at once
and slicing out the corresponding positions. This is the standard KV-cache correctness pattern
(same one used for the NIAH DynamicCache rewrite earlier in this project) -- it validates that
the cache's masking is evaluated at the right *absolute* position, not just that isolated calls
are internally consistent.

    .venv-executorch/Scripts/python.exe test_differential.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "build")
import windowed_sdpa_kv_cache_pybind as kernel_module


def reference_windowed_attention(q, k, v, head_windowed, sink, window):
    """Full T x T masked-softmax reference over the WHOLE sequence. q,k,v: (B,H,T,D)."""
    B, H, T, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    idx = torch.arange(T)
    causal = (idx[None, :] <= idx[:, None])
    win_valid = (idx[None, :] < sink) | (idx[None, :] > idx[:, None] - window)
    dense_mask = causal
    windowed_mask = causal & win_valid

    hw = head_windowed.view(1, H, 1, 1)
    mask = torch.where(hw, windowed_mask.view(1, 1, T, T), dense_mask.view(1, 1, T, T))
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def run_case(name, B, H, T, D, sink, window, frac_windowed, seed, prefill_len, decode_chunk, max_t):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)
    head_windowed = torch.rand(H) < frac_windowed

    ref = reference_windowed_attention(q, k, v, head_windowed, sink, window)

    k_cache = np.zeros((B, H, max_t, D), dtype=np.float32)
    v_cache = np.zeros((B, H, max_t, D), dtype=np.float32)

    outputs = []
    pos = 0
    # Prefill: one call covering [0, prefill_len).
    chunk = kernel_module.windowed_sdpa_kv_cache(
        q[:, :, pos:pos + prefill_len].numpy(),
        k[:, :, pos:pos + prefill_len].numpy(),
        v[:, :, pos:pos + prefill_len].numpy(),
        k_cache, v_cache,
        head_windowed.numpy(), sink, window, pos,
    )
    outputs.append(chunk)
    pos += prefill_len

    # Decode: remaining tokens in chunks of decode_chunk (1 == true token-at-a-time decode).
    while pos < T:
        n = min(decode_chunk, T - pos)
        chunk = kernel_module.windowed_sdpa_kv_cache(
            q[:, :, pos:pos + n].numpy(),
            k[:, :, pos:pos + n].numpy(),
            v[:, :, pos:pos + n].numpy(),
            k_cache, v_cache,
            head_windowed.numpy(), sink, window, pos,
        )
        outputs.append(chunk)
        pos += n

    kernel_out = torch.from_numpy(np.concatenate(outputs, axis=2))
    max_diff = (kernel_out - ref).abs().max().item()
    ok = max_diff < 1e-4
    status = "PASSED" if ok else "FAILED"
    print(
        f"[{status}] {name:32} B={B} H={H} T={T} D={D} sink={sink} window={window} "
        f"frac_windowed={frac_windowed:.2f} prefill={prefill_len} decode_chunk={decode_chunk} "
        f"max abs diff = {max_diff:.3e}"
    )
    return ok


cases = [
    # name, B, H, T, D, sink, window, frac_windowed, seed, prefill_len, decode_chunk, max_t
    ("prefill_then_single_decode", 1, 4, 40, 8, 4, 16, 0.5, 0, 20, 1, 64),
    ("all_dense_decode",           1, 4, 40, 8, 4, 16, 0.0, 1, 20, 1, 64),
    ("all_windowed_decode",        1, 4, 40, 8, 4, 16, 1.0, 2, 20, 1, 64),
    ("long_ctx_past_window",       1, 8, 200, 8, 4, 16, 0.5, 3, 50, 1, 256),
    ("chunked_decode",             1, 8, 200, 8, 4, 16, 0.5, 4, 50, 8, 256),
    ("batch_gt_1",                 3, 6, 100, 8, 4, 16, 0.5, 5, 30, 4, 128),
    ("zero_prefill_pure_decode",   1, 4, 60, 8, 4, 16, 0.5, 6, 0, 1, 64),
    ("tiny_window_boundary",       1, 4, 80, 8, 2, 8, 0.5, 7, 10, 1, 128),
    ("single_call_no_decode",      1, 4, 40, 8, 4, 16, 0.5, 8, 40, 1, 64),  # prefill == full T
]

all_ok = all(run_case(*c) for c in cases)
print()
print("ALL PASSED" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
