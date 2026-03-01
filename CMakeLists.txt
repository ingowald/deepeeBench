# SPDX-FileCopyrightText: Copyright (c) Ingo Wald
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

option(DEEPEE_OMP "Use OpenMP" OFF)

project(dpBench LANGUAGES C CXX)

if (DEEPEE_OMP)
  set(DEEPEERT_OMP ON)
else()
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
endif()

add_subdirectory(submodules/miniDP)
#find_package(deepeeRT REQUIRED)
add_subdirectory(submodules/deepeeRT)


add_executable(dpBench
  bench.cu
)
target_link_libraries(dpBench
  PUBLIC
  deepeeRT::deepeeRT
  miniSceneDP
  )
