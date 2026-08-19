/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Active GPU representation at the cold-page boundary.
enum class Nvfp4BoundaryRuntimeType : std::uint8_t
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

//! One Base Page selected for GPU-to-Host transformation.
struct Nvfp4BoundaryOffloadPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! One Base Page selected for Host-to-GPU transformation.
struct Nvfp4BoundaryOnboardPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! Per-layer geometry and scales for [K | V | K scales | V scales] records in HND order.
//! `headDim` is a multiple of 16; `*OrigQuant` encodes and `*QuantOrig` decodes K/V at indices 0/1.
struct Nvfp4BoundaryKernelParams
{
    std::int32_t numKvHeads;
    std::int32_t tokensPerPage;
    std::int32_t headDim;
    float nvfp4ScaleOrigQuant[2];
    float nvfp4ScaleQuantOrig[2];
    float fp8ScaleOrigQuant[2];
    float fp8ScaleQuantOrig[2];
};

//! Immutable transform plan for one Attention layer.
struct Nvfp4BoundaryLayerPlan
{
    std::uintptr_t rawKBase;
    std::uintptr_t rawVBase;
    std::size_t rawKSlotBytes;
    std::size_t rawVSlotBytes;
    std::size_t coldOffset;
    Nvfp4BoundaryKernelParams params;
};

inline constexpr std::uint32_t kNvfp4BoundaryMaxLayersPerLaunch = 128;

//! Configure-time launch plan for one Attention lifecycle.
struct Nvfp4BoundaryPreparedPlan
{
    std::array<Nvfp4BoundaryLayerPlan, kNvfp4BoundaryMaxLayersPerLaunch> layers{};
    std::uint32_t numLayers = 0;
    std::uint32_t maxTileHalfGroups = 0;
    std::size_t coldPageBytes = 0;
    Nvfp4BoundaryRuntimeType runtimeType = Nvfp4BoundaryRuntimeType::kFloat16;
};

//! Validate and freeze one lifecycle's boundary-transform plan.
[[nodiscard]] Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(
    std::vector<Nvfp4BoundaryLayerPlan> const& layers, std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType);

//! Compress GPU Pages into mapped-Host NVFP4 records.
void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream);

//! Restore mapped-Host NVFP4 records into GPU Pages.
void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
