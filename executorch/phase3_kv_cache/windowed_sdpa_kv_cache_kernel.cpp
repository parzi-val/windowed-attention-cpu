#include "windowed_sdpa_kv_cache_kernel.h"
#include <cmath>
#include <vector>

namespace sg {

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
    float* out) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // Write the new chunk into the cache first -- every query below, including i==0, is allowed
  // to attend to its own token (causal self-attention), so the cache must hold it up front.
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      for (int64_t i = 0; i < Tn; i++) {
        const float* k_src = k_new + ((b * H + h) * Tn + i) * D;
        const float* v_src = v_new + ((b * H + h) * Tn + i) * D;
        float* k_dst = k_cache + ((b * H + h) * MaxT + (start_pos + i)) * D;
        float* v_dst = v_cache + ((b * H + h) * MaxT + (start_pos + i)) * D;
        for (int64_t d = 0; d < D; d++) k_dst[d] = k_src[d];
        for (int64_t d = 0; d < D; d++) v_dst[d] = v_src[d];
      }
    }
  }

  const int64_t max_ctx = start_pos + Tn;  // upper bound on any query's attendable range
  std::vector<float> scores(static_cast<size_t>(max_ctx));
  std::vector<int64_t> idxs(static_cast<size_t>(max_ctx));

  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      const bool windowed = head_windowed[h];
      for (int64_t i = 0; i < Tn; i++) {
        const int64_t t = start_pos + i;  // absolute position of this query
        const float* q_t = q_new + ((b * H + h) * Tn + i) * D;
        float* out_t = out + ((b * H + h) * Tn + i) * D;

        int64_t win_lo = t - window + 1;
        if (win_lo < 0) win_lo = 0;

        int64_t n = 0;
        if (windowed) {
          for (int64_t s = 0; s < sink && s <= t; s++) {
            if (s < win_lo) idxs[n++] = s;
          }
          for (int64_t s = win_lo; s <= t; s++) idxs[n++] = s;
        } else {
          for (int64_t s = 0; s <= t; s++) idxs[n++] = s;
        }

        float maxval = -1e9f;
        for (int64_t j = 0; j < n; j++) {
          int64_t s = idxs[j];
          const float* k_s = k_cache + ((b * H + h) * MaxT + s) * D;
          float val = 0.0f;
          for (int64_t d = 0; d < D; d++) val += q_t[d] * k_s[d];
          val *= scale;
          scores[j] = val;
          if (val > maxval) maxval = val;
        }
        float expsum = 0.0f;
        for (int64_t j = 0; j < n; j++) {
          scores[j] = std::exp(scores[j] - maxval);
          expsum += scores[j];
        }
        float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
        for (int64_t j = 0; j < n; j++) scores[j] *= inv;

        for (int64_t d = 0; d < D; d++) out_t[d] = 0.0f;
        for (int64_t j = 0; j < n; j++) {
          int64_t s = idxs[j];
          const float* v_s = v_cache + ((b * H + h) * MaxT + s) * D;
          float w = scores[j];
          for (int64_t d = 0; d < D; d++) out_t[d] += w * v_s[d];
        }
      }
    }
  }
}

}  // namespace sg
