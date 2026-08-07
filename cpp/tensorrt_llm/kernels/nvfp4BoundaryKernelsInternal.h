/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::detail
{

//! Benchmark-only boundary-transfer strategy.
//!
//! Production callers use the two public `kAuto` launchers. These forced modes
//! let the microbenchmark compare pipelines without changing Page layout,
//! quantization math, or output bytes; they are not user or KVCM configuration.
enum class Nvfp4BoundaryTransferPipeline : std::uint8_t
{
    kAuto,
    kWholePage,
    kCompressedOutputTiled,
    //! Two-tile producer/transfer warp pipeline. Benchmarks may force it for
    //! any supported path; production `kAuto` selects it only for the measured
    //! large-cohort FP16/BF16 onboard region.
    kDoubleBufferedTiled,
};

void invokeNvfp4BoundaryOffloadCompressWithPipeline(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType,
    Nvfp4BoundaryTransferPipeline pipeline, cudaStream_t stream);

void invokeNvfp4BoundaryOnboardDecompressWithPipeline(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType,
    Nvfp4BoundaryTransferPipeline pipeline, cudaStream_t stream);

} // namespace kernels::detail

TRTLLM_NAMESPACE_END
