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

add_executable(prefill_decode_runner prefill_decode_runner.cpp)
target_include_directories(prefill_decode_runner PRIVATE ${_et_inc})
target_compile_features(prefill_decode_runner PRIVATE cxx_std_20)
target_link_libraries(prefill_decode_runner PRIVATE
  $<LINK_LIBRARY:WHOLE_ARCHIVE,windowed_sdpa_kv_cache_et_rt>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}portable_ops_lib${_s}>
  "${_et_lib}/${_p}extension_module_static${_s}"
  "${_et_lib}/${_p}extension_tensor${_s}"
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}executorch${_s}>
  "${_et_lib}/${_p}executorch_core${_s}"
  "${_et_lib}/${_p}extension_data_loader${_s}"
  "${_et_lib}/${_p}extension_flat_tensor${_s}"
  "${_et_lib}/${_p}extension_named_data_map${_s}"
  "${_et_lib}/${_p}portable_kernels${_s}"
  "${_et_lib}/${_p}kernels_util_all_deps${_s}"
  "${_et_lib}/${_p}flatccrt${_s}"
)
