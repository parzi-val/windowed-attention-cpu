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
the perplexity cost on WikiText-103). Arms are built from **fractions of that ranking**
(lowest `delta_ppl` -- cheapest to window -- first), matching the paper's actual reported
operating points rather than a fixed threshold:

- **dense** -- every head full causal, the control.
- **map25 / map50 / map75** -- the cheapest 25/50/75% of heads windowed, load-bearing heads
  stay dense.
- **allwindow** -- every head windowed (the StreamingLLM-style reference point).

## Correctness

Six gates run automatically before any timing number gets printed: dense C forward pass vs.
real HuggingFace `gpt2` logits, then all five arms (`dense`/`map25`/`map50`/`map75`/
`allwindow`) against that same reference on a fixed prompt short enough that windowing at
*any* fraction is mathematically required to reduce to dense exactly -- this validates
`attention_forward_perhead`'s dense and windowed code paths, and that the fraction-based
mask selection loads and threads through correctly end to end.

All six must pass or the program exits before running anything timed.

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

### Multi-core (OpenMP)

`gpt2_forward_windowed.c` is deliberately serial throughout (matches Karpathy's "clean,
minimal, readable" reference and avoids attributing any of the numbers above to threading
variance). `gpt2_forward_windowed_omp.c` is the same five-arm forward pass with `#pragma
omp parallel for` added to every layer -- not just the matmuls (the dominant cost), since
threading only the biggest piece just promotes the next-biggest to new-biggest (Amdahl's
law). Same six parity gates; if they fail under OpenMP but passed serially, that's a race
condition to chase before trusting any timing:

```
gcc -O2 -fopenmp -std=c11 -o gpt2_forward_windowed_omp gpt2_forward_windowed_omp.c -lm
OMP_NUM_THREADS=4 ./gpt2_forward_windowed_omp
```

### Realistic matmul (BLAS)

The naive triple-loop `matmul_forward` in the two variants above answers "does windowing
help on top of an artificially slow matmul" -- useful for isolating attention's own cost,
but not representative of any real deployment, which would never ship an unoptimized
matmul either. `gpt2_forward_windowed_blas.c` swaps `matmul_forward` for `cblas_sgemm`
(OpenBLAS/Accelerate/MKL, anything providing the standard BLAS C API) and leaves the
attention kernels -- the actual thing being tested -- untouched. This is the
closer-to-real-deployment comparison point: excluding PyTorch was about avoiding its
dispatch/generation overhead, not about avoiding optimized numerical libraries.

```
sudo apt-get install -y libopenblas-dev   # Linux/Colab/Lightning
gcc -O2 -std=c11 -o gpt2_forward_windowed_blas gpt2_forward_windowed_blas.c -lopenblas -lm
./gpt2_forward_windowed_blas
```

macOS needs no separate install (links against the system Accelerate framework instead):

```
clang -O2 -std=c11 -DACCELERATE_NEW_LAPACK -o gpt2_forward_windowed_blas \
    gpt2_forward_windowed_blas.c -framework Accelerate
```

OpenBLAS threads internally by default -- control it with `OPENBLAS_NUM_THREADS=N`,
independent of anything in this file.

Builds fine without `-fopenmp` too (falls back to serial, same file either way).

If you didn't clone with `--recurse-submodules`: `git submodule update --init`.

## Results (BLAS variant, gpt2 small, CPU)

Full end-to-end forward pass, all six parity gates passed (max abs diff vs. real HF logits:
2.3e-4; all five arms bit-exact vs. the dense reference on the short gate prompt).

**Speedup vs. dense**, by context length:

| T | dense (ms) | map25 | map50 | map75 | allwindow |
|---|---|---|---|---|---|
| 128 | 568.2 | 1.21x | 1.08x | 1.22x | 1.20x |
| 256 | 1213.4 | 1.15x | 1.24x | 1.32x | 1.33x |
| 512 | 3344.4 | 1.12x | 1.34x | 1.56x | 1.98x |
| 1024 | 12562.0 | 1.16x | **1.67x** | 2.36x | 3.64x |

**Joint quality cost** (WikiText-103 perplexity, dense baseline 29.883):

| arm | PPL | delta vs. dense |
|---|---|---|
| map25 | 30.029 | +0.146 |
| map50 | 30.062 | **+0.180** |
| map75 | 30.987 | +1.105 |
| allwindow | 51.781 | +21.898 |

**map50 is the standout operating point at long context**: 1.67x full-model speedup at
T=1024 for +0.180 PPL -- a small fraction of allwindow's +21.9 PPL for only ~2x more
speedup. The speedup curve widens with context length across every fraction (tight at
T=256, wide open by T=1024), the O(n) vs O(n^2) gap compounding on the full model exactly
as it does on the isolated attention kernel, just visible here with a realistic matmul
underneath it instead of buried in naive-matmul noise.

## What this does and doesn't tell you

GPT-2 small's non-attention layers (QKV/attention-output/MLP matmuls) dominate total FLOPs
at short-to-moderate context lengths -- attention itself is a small slice of the total. With
the naive triple-loop `matmul_forward` (no BLAS, no explicit SIMD, deliberately kept simple
and portable rather than fast), that dominance drowns out windowing's effect almost
entirely: full-model speedups there stay close to 1.0x-1.3x even at T=1024, because the
identical-cost-for-every-arm matmul is nearly the whole bill.

The BLAS results above are the more realistic picture. Fixing the matmul doesn't just make
everything faster uniformly -- it shrinks the portion of total time that's identical across
arms, so windowing's *relative* contribution grows substantially (map50 goes from
~1.0x-ish under naive matmul to 1.67x under BLAS, at the same T=1024). No real deployment
would ship an unoptimized matmul either, so the BLAS numbers -- not the naive ones -- are
the ones that answer "does this help in practice." The naive-matmul variants stay in this
repo because they isolate attention's own behavior cleanly (no BLAS threading or blocking
decisions in the way), which is useful for a different question than the deployment one.

## Provenance

The compressibility map and the sink+window technique are part of an ongoing research thread
on structured sparse attention (K.R. Balasubramanian). This repo is the CPU/edge-inference
angle of that work, standalone and citable independent of any paper it may end up supporting.
