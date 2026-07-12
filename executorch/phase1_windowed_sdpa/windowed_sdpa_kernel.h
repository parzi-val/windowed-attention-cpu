// windowed_sdpa_kernel.h -- pure, framework-free sink+window attention. No ExecuTorch or
// ATen dependency at all, deliberately: this is the one place the actual math lives, and
// it's wrapped by both the ExecuTorch out-variant kernel (windowed_sdpa_et.cpp) and a
// pybind11 binding (windowed_sdpa_pybind.cpp) used only for the differential test. Same
// masking convention as attention_forward_perhead() in gpt2_forward_windowed.c (repo root).
#pragma once
#include <cstdint>

namespace sg {

// q,k,v: (B,H,T,D) contiguous fp32. head_windowed: (H,) bool. out: (B,H,T,D), pre-allocated.
void windowed_sdpa_kernel(
    const float* q,
    const float* k,
    const float* v,
    const bool* head_windowed,
    int64_t sink,
    int64_t window,
    int64_t B,
    int64_t H,
    int64_t T,
    int64_t D,
    float* out);

}  // namespace sg
