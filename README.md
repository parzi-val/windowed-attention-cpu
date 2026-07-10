# windowed-attention-cpu

A full end-to-end GPT-2 small forward pass, in C, comparing dense causal attention against
sink + local-window structured sparse attention, driven by a real per-head compressibility
map rather than a blanket "window everything" switch. CPU-only, no GPU/CUDA/Triton
dependency anywhere in the loop.

Built on top of [Andrej Karpathy's llm.c](https://github.com/karpathy/llm.c) (MIT licensed),
checked out here as a submodule -- `llm.c`'s reference CPU implementation already gets the
GPT-2 weight export (including the Conv1D-to-Linear transpose) and the exact forward-pass
numerics right, so that part is reused as-is rather than reimplemented, and the submodule
tracks upstream directly instead of a frozen fork.

## Why

FlexAttention's block-sparse kernel (used elsewhere for GPU inference) is Triton-compiled
and has no CPU backend. This is a from-scratch CPU kernel for the same idea: each query
attends to a fixed sink (`SINK=4` tokens) plus a local window (`WINDOW=64` tokens) instead
of the full causal history -- bounded KV set regardless of sequence length, so it's O(n)
instead of O(n^2).

The interesting part isn't "can you window everything" (that's easy, and degrades quality).
It's windowing *only the heads that don't need long-range attention* while keeping the
load-bearing heads dense -- `head_compressibility_gpt2.json` is a real per-(layer,head)
causal compressibility map (replace one head at a time with the sink+window mask, measure
the perplexity cost on WikiText-103; heads with `delta_ppl < 0.05` are "lazy" -- near-free to
window in isolation). `gpt2_forward_windowed.c` runs three arms with that map:

- **dense** -- every head full causal, the control.
- **map** -- only the lazy heads (per the map) windowed, load-bearing heads stay dense.
- **allwindow** -- every head windowed (the StreamingLLM-style reference point).

## Correctness

Four gates run automatically before any timing number gets printed:

1. Dense C forward pass vs. real HuggingFace `gpt2` logits, on a fixed prompt.
2. `attention_forward_perhead`'s all-dense-mask branch vs. the same reference (the prompt is
   short enough that windowing at any fraction is mathematically required to reduce to dense
   exactly, so this validates the per-head function's dense-head code path).
3. Its all-windowed-mask branch, same check (validates the windowed-head code path).
4. The real map mask, same check (validates that the map loads and threads through
   end-to-end correctly).

All four must pass or the program exits before running anything timed.

## Usage

```
git clone --recurse-submodules https://github.com/parzi-val/windowed-attention-cpu.git
cd windowed-attention-cpu
pip install torch transformers   # for the export step only
python export_gpt2_c.py          # downloads real gpt2 weights, writes 3 .bin files

gcc -O2 -std=c11 -o gpt2_forward_windowed gpt2_forward_windowed.c -lm
./gpt2_forward_windowed
```

On Windows with clang targeting MSVC, drop `-lm` (its math functions are already in the CRT):

```
clang -O2 -std=c11 -o gpt2_forward_windowed.exe gpt2_forward_windowed.c
```

If you didn't clone with `--recurse-submodules`: `git submodule update --init`.

## What this does and doesn't tell you

GPT-2 small's non-attention layers (QKV/attention-output/MLP matmuls) dominate total FLOPs
at short-to-moderate context lengths -- attention itself is a small slice of the total, and
`matmul_forward` here is a naive triple-loop implementation (no BLAS, no explicit SIMD),
deliberately kept simple and portable rather than fast, costing the same for all three arms.
So the speedup on the *full model* forward pass is much smaller than the speedup on the
attention kernel in isolation would be -- this answers "how much does windowing move the
needle end-to-end on real hardware," a more conservative and more honest question than "how
much faster is the attention math by itself."

## Provenance

The compressibility map and the sink+window technique are part of an ongoing research thread
on structured sparse attention (K.R. Balasubramanian). This repo is the CPU/edge-inference
angle of that work, standalone and citable independent of any paper it may end up supporting.
