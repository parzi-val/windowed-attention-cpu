// windowed_sdpa_et.cpp -- ExecuTorch out-variant kernel, registered via EXECUTORCH_LIBRARY.
// Thin: extracts raw pointers/sizes from the ExecuTorch Tensor API and calls the pure
// kernel in windowed_sdpa_kernel.cpp, which is where the actual math lives and is what the
// differential test in windowed_sdpa_pybind.cpp validates directly.
#include "windowed_sdpa_kernel.h"
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace sg {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

Tensor& windowed_sdpa_out(
    KernelRuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& head_windowed,
    int64_t sink,
    int64_t window,
    Tensor& out) {
  (void)ctx;
  sg::windowed_sdpa_kernel(
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      head_windowed.const_data_ptr<bool>(),
      sink,
      window,
      q.size(0),
      q.size(1),
      q.size(2),
      q.size(3),
      out.mutable_data_ptr<float>());
  return out;
}

}  // namespace native
}  // namespace sg

EXECUTORCH_LIBRARY(sg, "windowed_sdpa.out", sg::native::windowed_sdpa_out);
