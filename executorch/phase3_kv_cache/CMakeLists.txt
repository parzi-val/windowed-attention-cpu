cmake_minimum_required(VERSION 3.19)
project(windowed_sdpa_phase3 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The pure kernel -- no ExecuTorch/ATen dependency, shared by the pybind and (eventually) ET
# targets, same split as Phase 1.
add_library(windowed_sdpa_kv_cache_kernel STATIC windowed_sdpa_kv_cache_kernel.cpp)
target_include_directories(windowed_sdpa_kv_cache_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# pybind11 module for the differential test. Skipped when cross-compiling (Android) -- no
# host Python to load an ARM .pyd into.
if(NOT CMAKE_CROSSCOMPILING)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(windowed_sdpa_kv_cache_pybind windowed_sdpa_kv_cache_pybind.cpp)
  target_link_libraries(windowed_sdpa_kv_cache_pybind PRIVATE windowed_sdpa_kv_cache_kernel)
endif()

# ---------------------------------------------------------------------------------------
# Target 2 (Phase 4.1 desktop / Phase 4.2 Android): the real ExecuTorch-registered out-variant
# kernel, linked for real against a from-source ExecuTorch build. Same pattern as
# phase1_windowed_sdpa's CMakeLists.txt (see that file's comments for the full reasoning).
# ---------------------------------------------------------------------------------------
if(NOT DEFINED EXECUTORCH_SRC_DIR)
  set(EXECUTORCH_SRC_DIR "" CACHE PATH "Path to the ExecuTorch source checkout root")
endif()
if(NOT DEFINED TORCH_INCLUDE_DIR)
  set(TORCH_INCLUDE_DIR "" CACHE PATH "Path to torch's own include dir (site-packages/torch/include)")
endif()
if(NOT DEFINED EXECUTORCH_BUILD_SUBDIR)
  set(EXECUTORCH_BUILD_SUBDIR "cmake-out-desktop" CACHE STRING "cmake-out-desktop or cmake-out-android")
endif()

if(EXECUTORCH_SRC_DIR)
  set(_et_out "${EXECUTORCH_SRC_DIR}/${EXECUTORCH_BUILD_SUBDIR}")
  set(_et_inc "${_et_out}/include" "${EXECUTORCH_SRC_DIR}/.." "${TORCH_INCLUDE_DIR}")
  set(_et_lib "${_et_out}/lib")
  set(_p "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(_s "${CMAKE_STATIC_LIBRARY_SUFFIX}")

  add_library(windowed_sdpa_kv_cache_et STATIC windowed_sdpa_kv_cache_et.cpp)
  target_include_directories(windowed_sdpa_kv_cache_et PRIVATE ${_et_inc})
  target_link_libraries(windowed_sdpa_kv_cache_et PRIVATE windowed_sdpa_kv_cache_kernel)
  target_compile_features(windowed_sdpa_kv_cache_et PRIVATE cxx_std_20)

  add_executable(phase3_runner "${EXECUTORCH_SRC_DIR}/examples/portable/executor_runner/executor_runner.cpp")
  target_include_directories(phase3_runner PRIVATE ${_et_inc} "${_et_out}/third-party/gflags/include")
  target_compile_features(phase3_runner PRIVATE cxx_std_17)
  target_link_libraries(phase3_runner PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,windowed_sdpa_kv_cache_et>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,${_et_lib}/${_p}portable_ops_lib${_s}>
    "${_et_lib}/${_p}executorch${_s}"
    "${_et_lib}/${_p}executorch_core${_s}"
    "${_et_lib}/${_p}extension_evalue_util${_s}"
    "${_et_lib}/${_p}extension_runner_util${_s}"
    "${_et_lib}/${_p}extension_flat_tensor${_s}"
    "${_et_lib}/${_p}extension_data_loader${_s}"
    "${_et_lib}/${_p}portable_kernels${_s}"
    "${_et_lib}/${_p}kernels_util_all_deps${_s}"
    "${_et_lib}/${_p}flatccrt${_s}"
  )
  if(WIN32)
    target_link_libraries(phase3_runner PRIVATE
      "${_et_out}/third-party/gflags/gflags_nothreads_static.lib"
      shlwapi.lib
    )
  else()
    target_link_libraries(phase3_runner PRIVATE
      "${_et_out}/third-party/gflags/libgflags_nothreads.a"
    )
  endif()
else()
  message(STATUS "EXECUTORCH_SRC_DIR not set -- skipping windowed_sdpa_kv_cache_et target (pybind11 differential test is unaffected)")
endif()
