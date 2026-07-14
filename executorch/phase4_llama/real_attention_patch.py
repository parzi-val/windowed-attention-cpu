"""Phase 4.4a: wires the Phase 3 windowed_sdpa_kv_cache op into real Llama attention layers
(real q/k/v/o projections, real RoPE, real GQA via repeat_kv) instead of the toy synthetic
module Phase 3 used to prove export survives mutation. The op itself is already validated
against 9 synthetic prefill/decode cases (test_differential.py) -- what's new here is real
HF-shaped wiring, not kernel math.

K/V are expanded to n_heads (not n_kv_heads) via repeat_kv BEFORE hitting the cache/op, same
as the eager sanity-check notebook's WindowedLlamaAttention. This means the cache stores
n_rep redundant copies per KV group -- wasteful memory, but consistent with the earlier
decision to keep the Phase 3 kernel's H-uniform (no GQA-aware) signature rather than rewrite
it, since this project measures compute savings, not memory savings ([[project_gqa_extension_a4]]).

    .venv-executorch/Scripts/python.exe real_attention_patch.py
"""
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# --- Op registration (same low-level Library.define()/.impl() pattern as
# phase3_export_survival.py -- see that file's comment for why the torch.library.custom_op
# decorator can't be used: infer_schema()'s multi-char alias names break ExecuTorch's
# ToOutVarPass schema parsing). ---
_lib = torch.library.Library("sg", "FRAGMENT")
_lib.define(
    "windowed_sdpa_kv_cache(Tensor q_new, Tensor k_new, Tensor v_new, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, Tensor head_windowed, SymInt sink, SymInt window, SymInt start_pos) -> Tensor"
)
_lib.define(
    "windowed_sdpa_kv_cache.out(Tensor q_new, Tensor k_new, Tensor v_new, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, Tensor head_windowed, SymInt sink, SymInt window, SymInt start_pos, "
    "*, Tensor(c!) out) -> Tensor(c!)"
)


def register_kernel_impl(kernel_module):
    """kernel_module: the pybind11 module (built under phase3_kv_cache/build/). Deferred to a
    function rather than imported at module scope, since eager/pybind execution is only needed
    for the parity check below -- export itself only needs the Meta/fake impl."""

    def _impl(q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos):
        out = kernel_module.windowed_sdpa_kv_cache(
            q_new.detach().numpy(), k_new.detach().numpy(), v_new.detach().numpy(),
            k_cache.detach().numpy(), v_cache.detach().numpy(),
            head_windowed.numpy(), sink, window, start_pos,
        )
        return torch.from_numpy(out)

    def _out_impl(q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos, out):
        result = kernel_module.windowed_sdpa_kv_cache(
            q_new.detach().numpy(), k_new.detach().numpy(), v_new.detach().numpy(),
            k_cache.detach().numpy(), v_cache.detach().numpy(),
            head_windowed.numpy(), sink, window, start_pos,
        )
        out.copy_(torch.from_numpy(result))
        return out

    _lib.impl("windowed_sdpa_kv_cache", _impl, "CompositeExplicitAutograd")
    _lib.impl("windowed_sdpa_kv_cache.out", _out_impl, "CompositeExplicitAutograd")


def _fake(q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos):
    return q_new.new_empty(q_new.shape)


_lib.impl("windowed_sdpa_kv_cache", _fake, "Meta")


class RealWindowedLlamaAttention(nn.Module):
    """Wraps a real HuggingFace LlamaAttention. Reuses its q/k/v/o projections and RoPE
    (identical math to the sanity-check notebook's WindowedLlamaAttention), but routes the
    actual attention computation through our KV-cache op instead of a masked full-sequence
    softmax -- so this module is stateful (owns k_cache/v_cache buffers) and must be called
    once per prefill/decode step with an explicit start_pos, not once per full sequence.
    """

    def __init__(self, original_attn, head_windowed, sink, window, max_batch_size, max_t):
        super().__init__()
        self.sink = sink
        self.window = window
        self.original_attn = original_attn
        cfg = original_attn.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(original_attn, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scaling = getattr(original_attn, "scaling", self.head_dim ** -0.5)

        self.register_buffer("head_windowed", torch.as_tensor(head_windowed, dtype=torch.bool))
        self.register_buffer("k_cache", torch.zeros(max_batch_size, self.num_heads, max_t, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(max_batch_size, self.num_heads, max_t, self.head_dim))

    def forward(self, hidden_states, start_pos, position_ids=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        hd = self.head_dim
        hidden_shape = (bsz, q_len, -1, hd)

        q = self.original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_ids is None:
            position_ids = (torch.arange(q_len, device=hidden_states.device) + start_pos).unsqueeze(0)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            try:
                cos, sin = self.original_attn.rotary_emb(v, position_ids)
            except TypeError:
                cos, sin = self.original_attn.rotary_emb(v, seq_len=q_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # GQA: expand K/V to num_heads before the op -- the kernel has no group notion, same
        # tradeoff the eager sanity check already made.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = torch.ops.sg.windowed_sdpa_kv_cache(
            q, k, v, self.k_cache, self.v_cache, self.head_windowed,
            self.sink, self.window, start_pos,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return (self.original_attn.o_proj(out), None)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../phase3_kv_cache/build")
    import windowed_sdpa_kv_cache_pybind as kernel_module
    register_kernel_impl(kernel_module)

    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

    # Real Llama-3.2-1B shape, random weights -- no download/gate/GPU needed for this wiring
    # check. This validates real HF projection/RoPE/GQA plumbing against the op, independent
    # of what the weights actually are (kernel math is already covered by test_differential.py).
    config = LlamaConfig(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192,
        num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
    )
    model = LlamaModel(config).eval()
    layer = model.layers[0]

    SINK, WINDOW, MAX_T = 4, 64, 128
    head_windowed = [i % 2 == 0 for i in range(config.num_attention_heads)]  # alternate, for a mixed test
    patched = RealWindowedLlamaAttention(layer.self_attn, head_windowed, SINK, WINDOW, 1, MAX_T)

    hidden = torch.randn(1, config.hidden_size)
    rotary_emb = model.rotary_emb
    prompt_len = 10

    print("Prefill (10 tokens) + 3 decode steps, checking shapes and cache mutation...")
    prompt_hidden = torch.randn(1, prompt_len, config.hidden_size)
    position_ids = torch.arange(prompt_len).unsqueeze(0)
    cos, sin = rotary_emb(prompt_hidden, position_ids)
    out, _ = patched(prompt_hidden, start_pos=0, position_ids=position_ids, position_embeddings=(cos, sin))
    print(f"  prefill out shape: {tuple(out.shape)}  (expect (1, {prompt_len}, {config.hidden_size}))")
    assert out.shape == (1, prompt_len, config.hidden_size)
    assert patched.k_cache[:, :, :prompt_len, :].abs().sum().item() > 0, "prefill did not write cache"

    decode_steps = []
    pos = prompt_len
    for step in range(3):
        step_hidden = torch.randn(1, 1, config.hidden_size)
        position_ids = torch.tensor([[pos]])
        cos, sin = rotary_emb(step_hidden, position_ids)
        out, _ = patched(step_hidden, start_pos=pos, position_ids=position_ids, position_embeddings=(cos, sin))
        assert out.shape == (1, 1, config.hidden_size)
        assert patched.k_cache[:, :, pos, :].abs().sum().item() > 0, f"decode step {step} did not write cache"
        decode_steps.append(step_hidden)
        pos += 1
    print(f"  {3} decode steps OK, cache written through position {pos - 1}")

    # --- Numeric parity: incremental cache path vs. a full-sequence masked-softmax reference,
    # using the SAME real weights and SAME hidden states, over the SAME positions. The kernel's
    # own math is already validated (test_differential.py, synthetic q/k/v); this checks the
    # NEW part -- real RoPE/GQA/position_ids wiring composed with it -- actually matches.
    print("\nNumeric parity: incremental (cache/op) vs. full-sequence (masked-softmax) reference...")
    full_hidden = torch.cat([prompt_hidden] + decode_steps, dim=1)  # (1, prompt_len+3, hidden)
    T = full_hidden.shape[1]
    attn = layer.self_attn
    hd, nh, nkv, n_rep = patched.head_dim, patched.num_heads, patched.num_kv_heads, patched.n_rep
    scaling = patched.scaling

    q = attn.q_proj(full_hidden).view(1, T, nh, hd).transpose(1, 2)
    k = attn.k_proj(full_hidden).view(1, T, nkv, hd).transpose(1, 2)
    v = attn.v_proj(full_hidden).view(1, T, nkv, hd).transpose(1, 2)
    position_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rotary_emb(full_hidden, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
    idx = torch.arange(T)
    causal = idx[None, :] <= idx[:, None]
    win_valid = (idx[None, :] < SINK) | (idx[None, :] > idx[:, None] - WINDOW)
    hw = patched.head_windowed.view(1, nh, 1, 1)
    mask = torch.where(hw, (causal & win_valid).view(1, 1, T, T), causal.view(1, 1, T, T))
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(1, T, -1)
    ref_out = attn.o_proj(ref_out)

    # Re-run the incremental path fresh (prior calls already mutated patched's cache/state) and
    # collect outputs at every position to compare against the reference's full-sequence output.
    patched2 = RealWindowedLlamaAttention(attn, head_windowed, SINK, WINDOW, 1, MAX_T)
    cos_p, sin_p = rotary_emb(prompt_hidden, torch.arange(prompt_len).unsqueeze(0))
    inc_out, _ = patched2(prompt_hidden, start_pos=0, position_ids=torch.arange(prompt_len).unsqueeze(0),
                           position_embeddings=(cos_p, sin_p))
    inc_chunks = [inc_out]
    for i, step_hidden in enumerate(decode_steps):
        p = prompt_len + i
        cos_s, sin_s = rotary_emb(step_hidden, torch.tensor([[p]]))
        step_out, _ = patched2(step_hidden, start_pos=p, position_ids=torch.tensor([[p]]),
                                position_embeddings=(cos_s, sin_s))
        inc_chunks.append(step_out)
    inc_out_full = torch.cat(inc_chunks, dim=1)

    max_diff = (inc_out_full - ref_out).abs().max().item()
    print(f"  max abs diff (incremental vs. full-sequence reference) = {max_diff:.3e}")
    assert max_diff < 1e-4, "incremental cache path does not match full-sequence reference -- wiring bug"
    print("  PASS: incremental cache path is numerically identical to the full-sequence reference.")

    print("\nPASS: real HF-shaped q/k/v/o + RoPE + GQA wiring through the KV-cache op works end to end.")
