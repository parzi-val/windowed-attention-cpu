// windowed_sdpa_kv_cache_kernel.h -- incremental-decode sink+window attention. Same masking
// convention as windowed_sdpa_kernel (phase1_windowed_sdpa/), generalized to write new K/V
// into a persistent cache and read from the cache instead of a fully-materialized sequence.
// Pure, framework-free: no ExecuTorch/ATen dependency, wrapped by both the ET out-variant
// kernel and a pybind11 binding for the differential test, same split as Phase 1.
#pragma once
#include <cstdint>

namespace sg {

// q_new,k_new,v_new: (B,H,Tn,D) contiguous fp32 -- the new chunk of tokens being decoded.
// k_cache,v_cache: (B,H,MaxT,D) contiguous fp32, mutated in place -- k_new/v_new are written
// into [start_pos, start_pos+Tn) before attention is computed.
// head_windowed: (H,) bool. out: (B,H,Tn,D), pre-allocated.
// Each new query at local index i attends over cache[0 : start_pos+i+1) under the sink+window
// mask evaluated at its absolute position (start_pos+i), same convention as Phase 1.
void windowed_sdpa_kv_cache_kernel(
    const float* q_new,
    const float* k_new,
    const float* v_new,
    float* k_cache,
    float* v_cache,
    const bool* head_windowed,
    int64_t sink,
    int64_t window,
    int64_t start_pos,
    int64_t B,
    int64_t H,
    int64_t Tn,
    int64_t D,
    int64_t MaxT,
    float* out);

}  // namespace sg
