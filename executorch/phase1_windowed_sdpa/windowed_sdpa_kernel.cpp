#include "windowed_sdpa_kernel.h"
#include <cmath>
#include <vector>

namespace sg {

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
    float* out) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // Sized once to T (the max any single query could need), reused across every (b,h,t).
  std::vector<float> scores(static_cast<size_t>(T));
  std::vector<int64_t> idxs(static_cast<size_t>(T));

  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      const bool windowed = head_windowed[h];
      for (int64_t t = 0; t < T; t++) {
        const float* q_t = q + ((b * H + h) * T + t) * D;
        float* out_t = out + ((b * H + h) * T + t) * D;

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
        for (int64_t i = 0; i < n; i++) {
          int64_t s = idxs[i];
          const float* k_s = k + ((b * H + h) * T + s) * D;
          float val = 0.0f;
          for (int64_t d = 0; d < D; d++) val += q_t[d] * k_s[d];
          val *= scale;
          scores[i] = val;
          if (val > maxval) maxval = val;
        }
        float expsum = 0.0f;
        for (int64_t i = 0; i < n; i++) {
          scores[i] = std::exp(scores[i] - maxval);
          expsum += scores[i];
        }
        float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
        for (int64_t i = 0; i < n; i++) scores[i] *= inv;

        for (int64_t d = 0; d < D; d++) out_t[d] = 0.0f;
        for (int64_t i = 0; i < n; i++) {
          int64_t s = idxs[i];
          const float* v_s = v + ((b * H + h) * T + s) * D;
          float w = scores[i];
          for (int64_t d = 0; d < D; d++) out_t[d] += w * v_s[d];
        }
      }
    }
  }
}

}  // namespace sg
