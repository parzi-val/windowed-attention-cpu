cmake_minimum_required(VERSION 3.19)
project(prefill_decode_runner CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Same op-registration split as phase1/phase3 (windowed_sdpa_kernel.cpp etc.) -- this runner
# needs its own compiled copy of the ET-registered KV-cache op (windowed_sdpa_kv_cache_et.cpp)
# WHOLE_ARCHIVE-linked in, since the .pte's decode/prefill methods call it, but nothing else
# references the translation unit directly (its EXECUTORCH_LIBRARY registration is a
# static-initializer side effect -- see phase1_windowed_sdpa/CMakeLists.txt for the full
# reasoning, unchanged here).
if(NOT DEFINED EXECUTORCH_SRC_DIR)
  set(EXECUTORCH_SRC_DIR "" CACHE PATH "Path to the ExecuTorch source checkout root")
endif()
if(NOT DEFINED TORCH_INCLUDE_DIR)
  set(TORCH_INCLUDE_DIR "" CACHE PATH "Path to torch's own include dir (site-packages/torch/include)")
endif()
if(NOT DEFINED EXECUTORCH_BUILD_SUBDIR)
  set(EXECUTORCH_BUILD_SUBDIR "cmake-out-android" CACHE STRING "cmake-out-desktop or cmake-out-android")
endif()

set(_kv_cache_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../phase3_kv_cache")
set(_et_out "${EXECUTORCH_SRC_DIR}/${EXECUTORCH_BUILD_SUBDIR}")
set(_et_inc "${_et_out}/include" "${EXECUTORCH_SRC_DIR}/.." "${TORCH_INCLUDE_DIR}")
set(_et_lib "${_et_out}/lib")
set(_p "${CMAKE_STATIC_LIBRARY_PREFIX}")
set(_s "${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_library(windowed_sdpa_kv_cache_kernel_rt STATIC "${_kv_cache_dir}/windowed_sdpa_kv_cache_kernel.cpp")
target_include_directories(windowed_sdpa_kv_cache_kernel_rt PUBLIC "${_kv_cache_dir}")

add_library(windowed_sdpa_kv_cache_et_rt STATIC "${_kv_cache_dir}/windowed_sdpa_kv_cache_et.cpp")
target_include_directories(windowed_sdpa_kv_cache_et_rt PRIVATE ${_et_inc})
target_link_libraries(windowed_sdpa_kv_cache_et_rt PRIVATE windowed_sdpa_kv_cache_kernel_rt)
target_compile_features(windowed_sdpa_kv_cache_et_rt PRIVATE cxx_std_20)

# Standard-op kernel set: optimized_native_cpu_ops_lib (registration) + optimized_kernels
# (SIMD implementations, backed by cpublas/eigen_blas) + portable_kernels as fallback for any
# op without an optimized implementation -- the exact combination ExecuTorch's own reference
# Llama CMakeLists.txt (examples/models/llama/CMakeLists.txt) uses for real inference, not the
# portable-only set Phase 1-3 used for toy-model export-survival testing (where speed never
# mattered). portable_kernels alone measured ~30s/decode-step on-device -- not remotely
# representative of a real deployment number.
add_executable(prefill_decode_runner prefill_decode_runner.cpp)
target_include_directories(prefill_decode_runner PRIVATE ${_et_inc})
target_compile_features(prefill_decode_runner PRIVATE cxx_std_20)
target_link_libraries(prefill_decode_runner PRIVATE
  $<LINK_LIBRARY:WHOLE_ARCHIVE,windowed_sdpa_kv_cache_et_rt>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}optimized_native_cpu_ops_lib${_s}>
  "${_et_lib}/${_p}extension_module_static${_s}"
  "${_et_lib}/${_p}extension_tensor${_s}"
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}executorch${_s}>
  "${_et_lib}/${_p}executorch_core${_s}"
  "${_et_lib}/${_p}extension_data_loader${_s}"
  "${_et_lib}/${_p}extension_flat_tensor${_s}"
  "${_et_lib}/${_p}extension_named_data_map${_s}"
  "${_et_lib}/${_p}optimized_kernels${_s}"
  "${_et_lib}/${_p}portable_kernels${_s}"
  "${_et_lib}/${_p}cpublas${_s}"
  "${_et_lib}/${_p}eigen_blas${_s}"
  "${_et_lib}/${_p}extension_threadpool${_s}"
  "${_et_lib}/${_p}pthreadpool${_s}"
  "${_et_lib}/${_p}cpuinfo${_s}"
  "${_et_lib}/${_p}kernels_util_all_deps${_s}"
  "${_et_lib}/${_p}flatccrt${_s}"
)
if(ANDROID)
  target_link_libraries(prefill_decode_runner PRIVATE log)
endif()

# Control experiment: runs ExecuTorch's own official reference Llama-3.2-1B export (standard
# SDPA + KV-cache, XNNPACK-delegated, no custom op) to determine whether our windowed_sdpa_kv_cache
# op is the source of decode's slowness or whether that's generic to this hardware/pipeline.
# xnnpack_backend WHOLE_ARCHIVE'd for the same reason as optimized_native_cpu_ops_lib -- its
# backend registration is a static-initializer side effect nothing else references directly.
# Library set matches examples/models/llama/CMakeLists.txt's own xnnpack_backend branch.
add_executable(reference_forward_runner reference_forward_runner.cpp)
target_include_directories(reference_forward_runner PRIVATE ${_et_inc})
target_compile_features(reference_forward_runner PRIVATE cxx_std_20)
target_link_libraries(reference_forward_runner PRIVATE
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}optimized_native_cpu_ops_lib${_s}>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}xnnpack_backend${_s}>
  # custom_ops (llama::update_cache / llama::custom_sdpa): built via a separate reconfigure with
  # EXECUTORCH_BUILD_KERNELS_LLM=ON, so it installs to extension/llm/custom_ops/, not the
  # aggregated lib/ directory other targets use -- confirmed via find, not guessed.
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_out}/extension/llm/custom_ops/${_p}custom_ops${_s}>
  "${_et_lib}/${_p}extension_module_static${_s}"
  "${_et_lib}/${_p}extension_tensor${_s}"
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}executorch${_s}>
  "${_et_lib}/${_p}executorch_core${_s}"
  "${_et_lib}/${_p}extension_data_loader${_s}"
  "${_et_lib}/${_p}extension_flat_tensor${_s}"
  "${_et_lib}/${_p}extension_named_data_map${_s}"
  "${_et_lib}/${_p}XNNPACK${_s}"
  "${_et_lib}/${_p}xnnpack-microkernels-prod${_s}"
  "${_et_lib}/${_p}kleidiai${_s}"
  "${_et_lib}/${_p}optimized_kernels${_s}"
  "${_et_lib}/${_p}portable_kernels${_s}"
  "${_et_lib}/${_p}cpublas${_s}"
  "${_et_lib}/${_p}eigen_blas${_s}"
  "${_et_lib}/${_p}extension_threadpool${_s}"
  "${_et_lib}/${_p}pthreadpool${_s}"
  "${_et_lib}/${_p}cpuinfo${_s}"
  "${_et_lib}/${_p}kernels_util_all_deps${_s}"
  "${_et_lib}/${_p}flatccrt${_s}"
)
if(ANDROID)
  target_link_libraries(reference_forward_runner PRIVATE log)
endif()
