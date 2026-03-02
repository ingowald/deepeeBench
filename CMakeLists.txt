# SPDX-FileCopyrightText: Copyright (c) Ingo Wald
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

option(DPRT_OMP "Use OpenMP" OFF)

project(dpBench LANGUAGES C CXX)

if (NOT DPRT_OMP)
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
endif()

add_subdirectory(submodules/miniDP)
#find_package(dprt REQUIRED)
add_subdirectory(submodules/deepeeRT)


add_executable(dpBench
  bench.cu
)
target_link_libraries(dpBench
  PUBLIC
  dprt::dprt
  miniSceneDP
  )
