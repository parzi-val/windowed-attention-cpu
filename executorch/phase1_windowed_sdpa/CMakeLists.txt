cmake_minimum_required(VERSION 3.19)
project(windowed_sdpa_phase1 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The pure kernel -- no ExecuTorch/ATen dependency, shared by both targets below.
add_library(windowed_sdpa_kernel STATIC windowed_sdpa_kernel.cpp)
target_include_directories(windowed_sdpa_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------------------
# Target 1 (required): pybind11 module for the differential test. No ExecuTorch needed.
# ---------------------------------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(windowed_sdpa_pybind windowed_sdpa_pybind.cpp)
target_link_libraries(windowed_sdpa_pybind PRIVATE windowed_sdpa_kernel)

# ---------------------------------------------------------------------------------------
# Target 2 (informational only on this platform): the real ExecuTorch-registered out-variant
# kernel (windowed_sdpa_et.cpp). It compiles cleanly against the pip package's headers, but
# is not added as a CMake target here because it cannot be *linked* against the pip package
# on Windows -- see the note below for why. The file itself is what Phase 2/4 will build on.
#
# find_package(executorch) is skipped deliberately: the shipped executorch-config.cmake's
# find_library() call for _portable_lib fails to match the literal .pyd filename on Windows
# (its NAMES matching expects CMAKE_FIND_LIBRARY_SUFFIXES-style names, not an arbitrary
# extension) even though the file demonstrably exists at the exact path it searches.
# ---------------------------------------------------------------------------------------
if(NOT DEFINED EXECUTORCH_PACKAGE_DIR)
  set(EXECUTORCH_PACKAGE_DIR "" CACHE PATH "Path to the executorch package root (site-packages/executorch)")
endif()

if(EXECUTORCH_PACKAGE_DIR)
  file(GLOB _et_portable_lib "${EXECUTORCH_PACKAGE_DIR}/extension/pybindings/_portable_lib*")
  if(_et_portable_lib)
    message(STATUS "executorch portable lib found at ${_et_portable_lib}")
    message(STATUS "windowed_sdpa_et: compile-only on this platform -- see note below, not added as a link target")
    # NOT buildable as a linked shared library against the pip package on Windows, and this
    # is a hard limit, not a flag/tooling issue: `llvm-readobj --coff-exports` on
    # _portable_lib.pyd shows exactly one export, PyInit__portable_lib (the CPython module
    # entry point). Every ExecuTorch runtime symbol this kernel needs (kernel registry,
    # at::Tensor methods, etc.) is statically linked *inside* the pyd and never exposed --
    # there is no import lib to generate because there's nothing exported to import. Grepping
    # site-packages/executorch confirms zero .lib/.a files ship with the pip wheel either; the
    # package is Python-only by design, not meant to be linked against from external C++.
    #
    # windowed_sdpa_et.cpp itself is correct and compiles cleanly against ExecuTorch's headers
    # (confirmed separately) -- it's the EXECUTORCH_LIBRARY-registered kernel that Phase 2/4
    # will use. The real link-and-run verification of it will happen naturally in Phase 4,
    # which builds ExecuTorch from source for Android (the standard workflow for
    # XNNPACK/Vulkan/NDK targets), producing actual linkable static libraries. Desktop
    # correctness for Phase 1 is instead covered by windowed_sdpa_pybind + test_differential.py.
  else()
    message(STATUS "EXECUTORCH_PACKAGE_DIR set but _portable_lib not found under it -- skipping windowed_sdpa_et")
  endif()
else()
  message(STATUS "EXECUTORCH_PACKAGE_DIR not set -- skipping windowed_sdpa_et target (pybind11 differential test is unaffected)")
endif()
