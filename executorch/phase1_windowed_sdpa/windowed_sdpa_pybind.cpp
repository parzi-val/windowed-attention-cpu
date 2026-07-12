// windowed_sdpa_pybind.cpp -- exposes the pure kernel (windowed_sdpa_kernel.cpp) to Python
// for the Phase 1 differential test. Deliberately independent of ExecuTorch/ATen tensor
// construction -- takes contiguous float32/bool numpy arrays directly. This validates the
// same arithmetic the EXECUTORCH_LIBRARY-registered kernel (windowed_sdpa_et.cpp) wraps;
// it is not itself the torch.library custom op used for export (that's Phase 2).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "windowed_sdpa_kernel.h"

namespace py = pybind11;

static py::array_t<float> windowed_sdpa_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> v,
    py::array_t<bool, py::array::c_style | py::array::forcecast> head_windowed,
    int64_t sink,
    int64_t window) {
  auto qbuf = q.request();
  if (qbuf.ndim != 4) throw std::runtime_error("q must be (B,H,T,D)");
  int64_t B = qbuf.shape[0], H = qbuf.shape[1], T = qbuf.shape[2], D = qbuf.shape[3];

  auto out = py::array_t<float>({B, H, T, D});
  auto obuf = out.request();

  sg::windowed_sdpa_kernel(
      static_cast<const float*>(q.request().ptr),
      static_cast<const float*>(k.request().ptr),
      static_cast<const float*>(v.request().ptr),
      static_cast<const bool*>(head_windowed.request().ptr),
      sink,
      window,
      B, H, T, D,
      static_cast<float*>(obuf.ptr));

  return out;
}

PYBIND11_MODULE(windowed_sdpa_pybind, m) {
  m.def(
      "windowed_sdpa",
      &windowed_sdpa_py,
      "Sink+window attention, pure C++ kernel. q,k,v: (B,H,T,D) fp32, head_windowed: (H,) bool.");
}
