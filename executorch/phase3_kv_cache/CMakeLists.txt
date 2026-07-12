cmake_minimum_required(VERSION 3.19)
project(windowed_sdpa_phase3 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The pure kernel -- no ExecuTorch/ATen dependency, shared by the pybind and (eventually) ET
# targets, same split as Phase 1.
add_library(windowed_sdpa_kv_cache_kernel STATIC windowed_sdpa_kv_cache_kernel.cpp)
target_include_directories(windowed_sdpa_kv_cache_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# pybind11 module for the differential test. No ExecuTorch needed.
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(windowed_sdpa_kv_cache_pybind windowed_sdpa_kv_cache_pybind.cpp)
target_link_libraries(windowed_sdpa_kv_cache_pybind PRIVATE windowed_sdpa_kv_cache_kernel)

# windowed_sdpa_kv_cache_et.cpp (the real EXECUTORCH_LIBRARY-registered kernel) is
# deliberately not a CMake target here -- Phase 1 established that the pip-distributed
# _portable_lib.pyd exports nothing to link against on Windows (llvm-readobj shows exactly
# one export, PyInit__portable_lib). It's compile-checked standalone instead (clang++ -c) and
# will get a real link target in Phase 4, which builds ExecuTorch from source for Android.
