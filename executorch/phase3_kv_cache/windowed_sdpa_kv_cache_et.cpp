// windowed_sdpa_kv_cache_et.cpp -- ExecuTorch out-variant kernel for incremental-decode
// sink+window attention, registered via EXECUTORCH_LIBRARY. Thin: extracts raw pointers from
// the ExecuTorch Tensor API and calls the pure kernel in windowed_sdpa_kv_cache_kernel.cpp,
// which is what the differential test in windowed_sdpa_kv_cache_pybind.cpp validates.
//
// Not wired as a CMake link target on this platform -- see CMakeLists.txt and the Phase 1
// notes (windowed_sdpa_et.cpp) for why: the pip-distributed _portable_lib.pyd exports nothing
// for external C++ to link against on Windows. Compile-checked standalone instead; gets a
// real link target in Phase 4 (Android, built from source).
#include "windowed_sdpa_kv_cache_kernel.h"
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace sg {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

Tensor& windowed_sdpa_kv_cache_out(
    KernelRuntimeContext& ctx,
    const Tensor& q_new,
    const Tensor& k_new,
    const Tensor& v_new,
    Tensor& k_cache,
    Tensor& v_cache,
    const Tensor& head_windowed,
    int64_t sink,
    int64_t window,
    int64_t start_pos,
    Tensor& out) {
  (void)ctx;
  sg::windowed_sdpa_kv_cache_kernel(
      q_new.const_data_ptr<float>(),
      k_new.const_data_ptr<float>(),
      v_new.const_data_ptr<float>(),
      k_cache.mutable_data_ptr<float>(),
      v_cache.mutable_data_ptr<float>(),
      head_windowed.const_data_ptr<bool>(),
      sink,
      window,
      start_pos,
      q_new.size(0),
      q_new.size(1),
      q_new.size(2),
      q_new.size(3),
      k_cache.size(2),
      out.mutable_data_ptr<float>());
  return out;
}

}  // namespace native
}  // namespace sg

EXECUTORCH_LIBRARY(sg, "windowed_sdpa_kv_cache.out", sg::native::windowed_sdpa_kv_cache_out);
