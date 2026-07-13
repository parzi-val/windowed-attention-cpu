cmake_minimum_required(VERSION 3.19)
project(windowed_sdpa_phase1 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The pure kernel -- no ExecuTorch/ATen dependency, shared by both targets below.
add_library(windowed_sdpa_kernel STATIC windowed_sdpa_kernel.cpp)
target_include_directories(windowed_sdpa_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------------------
# Target 1 (required, desktop only): pybind11 module for the differential test. Skipped when
# cross-compiling (Android) -- there's no host Python to load an ARM .pyd into, and it isn't
# what this configure pass is for anyway.
# ---------------------------------------------------------------------------------------
if(NOT CMAKE_CROSSCOMPILING)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(windowed_sdpa_pybind windowed_sdpa_pybind.cpp)
  target_link_libraries(windowed_sdpa_pybind PRIVATE windowed_sdpa_kernel)
endif()

# ---------------------------------------------------------------------------------------
# Target 2 (Phase 4.1 desktop / Phase 4.2 Android): the real ExecuTorch-registered out-variant
# kernel (windowed_sdpa_et.cpp), linked for real against a from-source ExecuTorch build --
# EXECUTORCH_BUILD_SUBDIR selects which one (cmake-out-desktop or cmake-out-android).
# EXECUTORCH_SRC_DIR is the checkout root (must be named exactly "executorch", a separate
# constraint -- see git history); TORCH_INCLUDE_DIR a venv with torch installed (both builds
# are ATen-mode: exec_aten::Tensor is at::Tensor, confirmed by array_ref.h needing
# c10/util/irange.h). Library prefix/suffix (lib*.a vs *.lib) comes from CMake's own
# CMAKE_STATIC_LIBRARY_PREFIX/SUFFIX so the same file works for both the clang-cl/lld-link
# desktop toolchain and the NDK's clang/lld toolchain without hardcoding either.
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
  # ${EXECUTORCH_SRC_DIR}/.. (not EXECUTORCH_SRC_DIR itself) so that source-tree-only headers
  # like extension/flat_tensor/flat_tensor_data_map.h (not part of the installed set) resolve
  # via their #include <executorch/...> prefix.
  set(_et_inc "${_et_out}/include" "${EXECUTORCH_SRC_DIR}/.." "${TORCH_INCLUDE_DIR}")
  set(_et_lib "${_et_out}/lib")
  set(_p "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(_s "${CMAKE_STATIC_LIBRARY_SUFFIX}")

  add_library(windowed_sdpa_et STATIC windowed_sdpa_et.cpp)
  target_include_directories(windowed_sdpa_et PRIVATE ${_et_inc})
  target_link_libraries(windowed_sdpa_et PRIVATE windowed_sdpa_kernel)
  target_compile_features(windowed_sdpa_et PRIVATE cxx_std_20)

  # Runner: examples/portable/executor_runner/executor_runner.cpp is reused directly from the
  # vendored checkout rather than copied -- it's a well-tested, self-contained example binary
  # that already does exactly what we need (load a .pte, run "forward", print outputs). Our op
  # is linked WHOLE_ARCHIVE, same mechanism ExecuTorch's own CMakeLists.txt uses to force-link
  # `custom_ops` into executor_runner (nothing else calls into windowed_sdpa_et.cpp directly --
  # its EXECUTORCH_LIBRARY registration is a static-initializer side effect, and a normal link
  # would dead-strip the whole translation unit as "unreferenced").
  add_executable(phase1_runner "${EXECUTORCH_SRC_DIR}/examples/portable/executor_runner/executor_runner.cpp")
  target_include_directories(phase1_runner PRIVATE ${_et_inc} "${_et_out}/third-party/gflags/include")
  target_compile_features(phase1_runner PRIVATE cxx_std_17)
  target_link_libraries(phase1_runner PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,windowed_sdpa_et>
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
    target_link_libraries(phase1_runner PRIVATE
      "${_et_out}/third-party/gflags/gflags_nothreads_static.lib"
      shlwapi.lib
    )
  else()
    target_link_libraries(phase1_runner PRIVATE
      "${_et_out}/third-party/gflags/libgflags_nothreads.a"
    )
  endif()
else()
  message(STATUS "EXECUTORCH_SRC_DIR not set -- skipping windowed_sdpa_et target (pybind11 differential test is unaffected)")
endif()
