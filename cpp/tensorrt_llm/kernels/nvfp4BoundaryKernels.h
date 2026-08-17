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

//! Active GPU representation restored at the Host boundary.
//!
//! This is an algorithm-side dispatch value, not Page metadata. KVCM owns the
//! runtime layout and supplies one homogeneous layer cohort per call.
enum class Nvfp4BoundaryRuntimeType : std::uint8_t
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

//! One complete KVCM Base Page selected for GPU-to-Host transformation.
//!
//! One descriptor represents all Attention layers and both K/V roles of a Base
//! Page. Up to 256 such descriptors are submitted in one launch, matching the
//! Page batching unit used by KVCM V2's batchedCopy. Layer addresses, geometry,
//! compact offsets, and calibrated scales are manager-lifetime state supplied
//! separately in Nvfp4BoundaryLayerPlan; they must never split this Page batch.
struct Nvfp4BoundaryOffloadPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! One complete KVCM Base Page selected for Host-to-GPU transformation.
//!
//! The index meanings are direction-independent: gpuPageIndex always selects
//! the hot GPU Slot and coldPageIndex always selects the compact lower-tier
//! Slot. Keeping that convention identical in encode and decode prevents a
//! source/destination swap from leaking into the kernel ABI.
struct Nvfp4BoundaryOnboardPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! Geometry and immutable per-layer scale convention shared by one Page batch.
//!
//! Each raw K/V pointer describes one HND Page with
//! `numKvHeads * tokensPerPage * headDim` elements. If that count is `N`, the
//! compact record contains `N/2`, `N/2`, `N/16`, and `N/16` bytes in the order
//! documented above, for `9N/8` bytes total. Packed K/V retain HND order and
//! both scale regions are linear `[H, P, D / 16]`. `headDim` must be divisible
//! by 16. One complete scale group contributes eight packed bytes; the CUDA
//! backend uses aligned 16-byte bodies plus an exact eight-byte tail. Scale
//! regions use vector transfers when aligned and retain a byte path otherwise.
//!
//! `*OrigQuant` multiplies an original-domain value before storing the named
//! quantized representation. `*QuantOrig` restores a stored quantized value to
//! the original domain. K and V are independent and indexed as [0] and [1].
//!
//! Pointer address spaces are a caller-side admission contract: raw buffers are
//! GPU allocations and compact buffers are CUDA-mapped Host allocations. The
//! hot path intentionally does not query every pointer with CUDA at launch.
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

//! Immutable transform plan for one Attention layer inside a complete Base
//! Page.
//!
//! `rawKBase` and `rawVBase` already include the BufferId offset within their
//! KVCM Pools. A Page address is therefore `raw*Base + gpuPageIndex *
//! raw*SlotBytes`. `coldOffset` selects this layer's self-contained
//! `[K packed | V packed | K block scales | V block scales]` record inside the
//! single compact cold Slot. Different layers may carry different geometry and
//! calibrated K/V scales; blockIdx.y selects the plan in device code so all
//! layers still execute in the same CUDA launch.
struct Nvfp4BoundaryLayerPlan
{
    std::uintptr_t rawKBase;
    std::uintptr_t rawVBase;
    std::size_t rawKSlotBytes;
    std::size_t rawVSlotBytes;
    std::size_t coldOffset;
    Nvfp4BoundaryKernelParams params;
};

//! Maximum number of local Attention layers carried by one complete-Page
//! launch. The fixed array is passed as a CUDA ``__grid_constant__`` argument;
//! only ``numLayers`` entries are observed by device code.
inline constexpr std::uint32_t kNvfp4BoundaryMaxLayersPerLaunch = 128;

//! Immutable, configure-time-validated launch plan for one Attention
//! lifecycle.
//!
//! KVCM Pool addresses, Slot strides, compact offsets, geometry, calibrated
//! scales, and the padded CUDA argument are fixed after StorageManager is
//! constructed. ``prepareNvfp4BoundaryPlan`` validates and derives them once
//! while the codec is configured. The offload/onboard hot path therefore only
//! validates dynamic Page indices and the cold base pointer before enqueueing
//! work; it never rescans every layer or rebuilds this 128-entry array.
struct Nvfp4BoundaryPreparedPlan
{
    std::array<Nvfp4BoundaryLayerPlan, kNvfp4BoundaryMaxLayersPerLaunch> layers{};
    std::uint32_t numLayers = 0;
    std::uint32_t maxHalfGroups = 0;
    std::uint32_t maxTileHalfGroups = 0;
    std::size_t coldPageBytes = 0;
    Nvfp4BoundaryRuntimeType runtimeType = Nvfp4BoundaryRuntimeType::kFloat16;
    bool allStandardGeometry = false;
};

//! Validate and freeze one lifecycle's immutable boundary-transform plan.
//!
//! This function is intentionally separate from the invoke functions so the
//! compression codec can call it exactly once from ``configure()``. It throws
//! synchronously when the model geometry, scales, Pool addresses, or compact
//! offsets cannot be represented by the SM100 NVFP4 kernels.
[[nodiscard]] Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(
    std::vector<Nvfp4BoundaryLayerPlan> const& layers, std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType);

//! Compress a homogeneous GPU Page cohort and transfer it directly to mapped
//! Host memory.
//!
//! The caller owns Page selection, source/destination lifetime, the CUDA
//! stream, completion fencing, publication, release, and synchronous launch
//! rollback. Asynchronous CUDA faults are fail-stop. This function submits
//! only the complete NVFP4 transform-plus-transfer payload.
void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream);

//! Transfer a homogeneous mapped-Host cohort and restore the active GPU
//! representation.
//!
//! For a non-empty batch, successful return means work was enqueued on
//! ``stream``; it does not publish the destination or synchronize the stream.
void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
