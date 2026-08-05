# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Google Benchmark Preparation - Same as google test ../tests/CMakeLists.txt
# Google Benchmark is provided under Apache-2.0 license
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "Disable google-benchmark tests")
set(BENCHMARK_ENABLE_INSTALL
    OFF
    CACHE INTERNAL "Disable google-benchmark install")
FetchContent_MakeAvailable(googlebenchmark)

add_custom_target(micro_benchmarks)

include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

function(add_benchmark test_name test_src)
  add_executable(${test_name} ${test_src})

  message("Linking with ${SHARED_TARGET}")
  target_link_libraries(${test_name} PUBLIC ${SHARED_TARGET}
                                            benchmark::benchmark)

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(
    ${test_name} PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}"
                        USING_OSS_CUTLASS_MOE_GEMM)

  add_dependencies(micro_benchmarks ${test_name})
endfunction()

add_benchmark(mixtureOfExpertsBackendBenchmark
              mixtureOfExpertsBackendBenchmarkLauncher.cu)

# Standalone NVFP4 boundary A/B. Compile the production boundary kernel and
# KVCM V2 copy implementation directly so this target exercises the real
# functions without linking the full TensorRT-LLM shared library. It remains
# EXCLUDE_FROM_ALL and is built only by naming this target (or the aggregate
# micro_benchmarks target).
set(NVFP4_BOUNDARY_BENCHMARK_SOURCES
    nvfp4BoundaryKernelsBenchmark.cu
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/kernels/nvfp4BoundaryKernels.cu
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/assert.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/cudaDriverWrapper.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/envUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/logger.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/stringUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/tllmException.cpp)

add_executable(nvfp4BoundaryKernelsBenchmark EXCLUDE_FROM_ALL
               ${NVFP4_BOUNDARY_BENCHMARK_SOURCES})
target_include_directories(
  nvfp4BoundaryKernelsBenchmark
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
target_compile_features(nvfp4BoundaryKernelsBenchmark PRIVATE cxx_std_17)
target_compile_definitions(nvfp4BoundaryKernelsBenchmark
                           PRIVATE TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
target_compile_options(
  nvfp4BoundaryKernelsBenchmark
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections;-fdata-sections>
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>)
target_link_options(nvfp4BoundaryKernelsBenchmark PRIVATE -Wl,--gc-sections)
target_link_libraries(
  nvfp4BoundaryKernelsBenchmark
  PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads ${CMAKE_DL_LIBS})

add_dependencies(micro_benchmarks nvfp4BoundaryKernelsBenchmark)
