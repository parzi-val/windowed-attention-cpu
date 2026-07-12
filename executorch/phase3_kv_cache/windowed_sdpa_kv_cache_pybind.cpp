// windowed_sdpa_kv_cache_pybind.cpp -- exposes the pure incremental-decode kernel
// (windowed_sdpa_kv_cache_kernel.cpp) to Python for the Phase 3 differential test.
// k_cache/v_cache are mutated in place, matching the real op's semantics (mutates_args in the
// AOT registration) -- forcecast is deliberately NOT used on those two so a copy is never
// silently substituted for the caller's array.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "windowed_sdpa_kv_cache_kernel.h"

namespace py = pybind11;

static py::array_t<float> windowed_sdpa_kv_cache_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> q_new,
    py::array_t<float, py::array::c_style | py::array::forcecast> k_new,
    py::array_t<float, py::array::c_style | py::array::forcecast> v_new,
    py::array_t<float, py::array::c_style> k_cache,
    py::array_t<float, py::array::c_style> v_cache,
    py::array_t<bool, py::array::c_style | py::array::forcecast> head_windowed,
    int64_t sink,
    int64_t window,
    int64_t start_pos) {
  auto qbuf = q_new.request();
  if (qbuf.ndim != 4) throw std::runtime_error("q_new must be (B,H,Tn,D)");
  int64_t B = qbuf.shape[0], H = qbuf.shape[1], Tn = qbuf.shape[2], D = qbuf.shape[3];

  auto kcbuf = k_cache.request();
  if (kcbuf.ndim != 4) throw std::runtime_error("k_cache must be (B,H,MaxT,D)");
  int64_t MaxT = kcbuf.shape[2];
  if (start_pos + Tn > MaxT) throw std::runtime_error("start_pos + Tn exceeds cache capacity");

  auto out = py::array_t<float>({B, H, Tn, D});
  auto obuf = out.request();

  sg::windowed_sdpa_kv_cache_kernel(
      static_cast<const float*>(q_new.request().ptr),
      static_cast<const float*>(k_new.request().ptr),
      static_cast<const float*>(v_new.request().ptr),
      static_cast<float*>(k_cache.request().ptr),
      static_cast<float*>(v_cache.request().ptr),
      static_cast<const bool*>(head_windowed.request().ptr),
      sink,
      window,
      start_pos,
      B, H, Tn, D, MaxT,
      static_cast<float*>(obuf.ptr));

  return out;
}

PYBIND11_MODULE(windowed_sdpa_kv_cache_pybind, m) {
  m.def(
      "windowed_sdpa_kv_cache",
      &windowed_sdpa_kv_cache_py,
      "Sink+window attention with incremental KV cache. Mutates k_cache/v_cache in place.");
}
