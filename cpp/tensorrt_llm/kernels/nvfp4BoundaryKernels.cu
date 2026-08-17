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

#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <exception>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

// BATCHEDCOPY REUSE: match KVCM V2's Host-copy CTA and four-stage pipeline.
// Original implementation:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
// Both boundary directions touch CUDA-mapped Host memory, so they intentionally
// keep batchedCopy's one-split low-bandwidth policy.
constexpr std::uint32_t kThreadsPerBlock = 128;
constexpr std::uint32_t kAsyncStages = 4;
// Mapped Host reads have much longer dependency latency than the GPU-resident
// raw inputs consumed by offload. Keep batchedCopy's four-stage ring for raw
// GPU loads, but let the tiled onboard loaders use all eight hardware cp.async
// groups before throttling. This changes only how many
// already-disjoint 16-byte Host grains are in flight; layout, arithmetic, and
// the number of final GPU stores remain unchanged.
constexpr std::uint32_t kHostLoadAsyncStages = 8;
constexpr std::uint32_t kHostMemorySplits = 1;
constexpr std::uint32_t kMaxTasksPerLaunch = 256;
// The layer plan is passed by value with the Page descriptors. This avoids a
// second metadata upload and gives one launch direct access to every layer's
// addresses and scales. Current TRT-LLM models remain well below 128 local
// Attention layers; reject a larger plan synchronously instead of silently
// returning to a per-layer host launch loop.
constexpr std::uint32_t kMaxLayersPerLaunch = kNvfp4BoundaryMaxLayersPerLaunch;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;
// A 1-KiB scale interval bounds the compact shared working set without
// changing compact bytes, arithmetic, or the public task contract.
constexpr std::uint32_t kTargetScaleTransferBytes = 1024;
constexpr std::uint32_t kHalfGroupsPerTransfer = 2U * kTargetScaleTransferBytes;
constexpr std::size_t kModernKernelParameterLimit = 32764;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(kTargetScaleTransferBytes > 0 && kTargetScaleTransferBytes % sizeof(uint4) == 0,
    "Compact scale transfer must remain a positive 16-byte multiple");
static_assert(kHalfGroupsPerTransfer % 2U == 0, "A transfer tile must not split an NVFP4 scale group");
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOffloadPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOnboardPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryLayerPlan>);
static_assert(sizeof(std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch>)
        + sizeof(std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch>) + 2U * sizeof(std::uintptr_t)
        + 3U * sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);
static_assert(sizeof(std::array<Nvfp4BoundaryOnboardPageTask, kMaxTasksPerLaunch>)
        + sizeof(std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch>) + 2U * sizeof(std::uintptr_t)
        + 3U * sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);

struct Sm100CapabilityEntry
{
    std::once_flag initialized;
    bool supported{false};
};

//! Query the current device on every call, but cache its immutable compute
//! capability by CUDA device ordinal. A single function-static `bool` would be
//! incorrect: `cudaSetDevice` is thread-local, so different host threads in one
//! process may launch on different GPU families. `call_once` makes the first
//! major/minor query for each ordinal race-free; later validation calls perform
//! only `cudaGetDevice` plus the cheap initialized check.
bool isCurrentDeviceSm100FamilyCached()
{
    int device{-1};
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    static std::vector<Sm100CapabilityEntry> entries = []
    {
        int deviceCount{0};
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        return std::vector<Sm100CapabilityEntry>(static_cast<std::size_t>(deviceCount));
    }();

    TLLM_CHECK_WITH_INFO(device >= 0 && static_cast<std::size_t>(device) < entries.size(),
        "Current CUDA device ordinal %d is outside the cached "
        "device range [0, %zu)",
        device, entries.size());
    auto& entry = entries[static_cast<std::size_t>(device)];
    std::call_once(entry.initialized,
        [device, &entry]
        {
            int major{0};
            int minor{0};
            TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            int const sm = major * 10 + minor;
            entry.supported = sm == 100 || sm == 103;
        });
    return entry.supported;
}

//! Reuse boundary for the fused implementation:
//!
//! * Quantization math is reused directly from `quantization.cuh`.
//! * The task-array batching, split/iteration indexing, four-stage shared ring,
//!   and commit/wait order are adapted from KVCM V2's `batchedCopy<N>`.
//! * `cp_async_commit_group()` and `cp_async_wait_group<N>()` are reused from
//!   TensorRT-LLM's shared `cudaAsyncOps.cuh` primitives.
//! * Raw and aligned packed bodies use batchedCopy's 16-byte `cp.async.cg`
//!   grain. A tight packed interval may have one leading or trailing 8-byte
//!   scale group; the bounded tile handles it in place.
//! * Every legal geometry uses the same bounded compressed-payload tile. The
//!   tile loop handles packed uint2 and byte-scale tails without introducing a
//!   geometry-specific kernel or dispatch rule.
//! * The transformed destination cannot call the original identity-copy
//!   kernel because source and destination layouts differ.
//!
//! No persistent GPU staging or second kernel is introduced. The compact tile
//! is CTA-local shared memory and disappears when the fused kernel completes.

//! Issue one global-to-shared asynchronous load.
//!
//! BATCHEDCOPY DIRECT REUSE: this is the same 16-byte `cp.async.cg` instruction
//! and predication contract used by `kvCacheManagerV2Utils.cu::batchedCopy<N>`.
template <typename T>
__device__ __forceinline__ void copyAsyncGlobalToShared(T* shared, T const* global, bool valid)
{
    static_assert(sizeof(T) == 16, "Boundary transfer grains must match batchedCopy's 16-byte width");
    if (valid)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "l"(__cvta_generic_to_shared(shared)), "l"(global)
                     : "memory");
    }
}

struct OffloadLayerTask
{
    void const* rawK;
    void const* rawV;
    std::uint8_t* compactPage;
};

struct OnboardLayerTask
{
    std::uint8_t const* compactPage;
    void* rawK;
    void* rawV;
};

__device__ OffloadLayerTask resolveTask(Nvfp4BoundaryOffloadPageTask const& page, Nvfp4BoundaryLayerPlan const& layer,
    std::uint8_t* coldBase, std::size_t coldPageBytes)
{
    std::size_t const gpuPage = static_cast<std::size_t>(page.gpuPageIndex);
    auto* compactPage = coldBase + static_cast<std::size_t>(page.coldPageIndex) * coldPageBytes + layer.coldOffset;
    return {reinterpret_cast<void const*>(layer.rawKBase + gpuPage * layer.rawKSlotBytes),
        reinterpret_cast<void const*>(layer.rawVBase + gpuPage * layer.rawVSlotBytes), compactPage};
}

__device__ OnboardLayerTask resolveTask(Nvfp4BoundaryOnboardPageTask const& page, Nvfp4BoundaryLayerPlan const& layer,
    std::uint8_t const* coldBase, std::size_t coldPageBytes)
{
    std::size_t const gpuPage = static_cast<std::size_t>(page.gpuPageIndex);
    auto const* compactPage
        = coldBase + static_cast<std::size_t>(page.coldPageIndex) * coldPageBytes + layer.coldOffset;
    return {compactPage, reinterpret_cast<void*>(layer.rawKBase + gpuPage * layer.rawKSlotBytes),
        reinterpret_cast<void*>(layer.rawVBase + gpuPage * layer.rawVSlotBytes)};
}

template <typename T>
__device__ T const* selectRawInput(OffloadLayerTask const& task, std::uint32_t role)
{
    return reinterpret_cast<T const*>(role == 0 ? task.rawK : task.rawV);
}

__host__ __device__ constexpr std::uint32_t packedBytesPerRole(std::uint32_t halfGroups)
{
    return halfGroups * sizeof(std::uint32_t);
}

__host__ __device__ constexpr std::uint32_t scaleBytesPerRole(std::uint32_t halfGroups)
{
    return halfGroups / 2U;
}

//! Resolve one logical BufferId inside the single compact lower-tier record.
//! Keeping this arithmetic in one helper makes the boundary task ABI one base
//! address per Page while preserving the existing packed and scale ordering.
__host__ __device__ constexpr std::uint32_t packedOffset(std::uint32_t role, std::uint32_t halfGroups)
{
    return role * packedBytesPerRole(halfGroups);
}

__host__ __device__ constexpr std::uint32_t scaleOffsetInCompactPage(std::uint32_t role, std::uint32_t halfGroups)
{
    return 2U * packedBytesPerRole(halfGroups) + role * scaleBytesPerRole(halfGroups);
}

__device__ std::uint8_t* selectPackedOutput(OffloadLayerTask const& task, std::uint32_t role, std::uint32_t halfGroups)
{
    return task.compactPage + packedOffset(role, halfGroups);
}

__device__ std::uint8_t* selectScaleOutput(OffloadLayerTask const& task, std::uint32_t role, std::uint32_t halfGroups)
{
    return task.compactPage + scaleOffsetInCompactPage(role, halfGroups);
}

__device__ std::uint8_t const* selectPackedInput(
    OnboardLayerTask const& task, std::uint32_t role, std::uint32_t halfGroups)
{
    return task.compactPage + packedOffset(role, halfGroups);
}

__device__ std::uint8_t const* selectScaleInput(
    OnboardLayerTask const& task, std::uint32_t role, std::uint32_t halfGroups)
{
    return task.compactPage + scaleOffsetInCompactPage(role, halfGroups);
}

template <typename T>
__device__ T* selectRawOutput(OnboardLayerTask const& task, std::uint32_t role)
{
    return reinterpret_cast<T*>(role == 0 ? task.rawK : task.rawV);
}

//! Shared staging may add internal alignment so the following scale interval
//! remains vector-addressable. This padding never appears in the cold record.
__host__ __device__ constexpr std::uint32_t packedStagingBytesPerRole(std::uint32_t halfGroups)
{
    return (packedBytesPerRole(halfGroups) + sizeof(uint4) - 1U) / sizeof(uint4) * sizeof(uint4);
}

__host__ __device__ constexpr std::uint32_t compactStagingBytesPerRole(std::uint32_t halfGroups)
{
    return packedStagingBytesPerRole(halfGroups) + scaleBytesPerRole(halfGroups);
}

__host__ __device__ constexpr std::uint32_t totalHalfGroupsPerRole(Nvfp4BoundaryKernelParams const& params)
{
    return static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage)
        * (static_cast<std::uint32_t>(params.headDim) / kElementsPerLane);
}

//! Select one bounded transfer tile from the linear compact payload.
//!
//! One tile contains at most 2,048 eight-value half-groups: 8 KiB of packed
//! values and 1 KiB of one-byte block scales. Both the capacity and every legal
//! Page contain a whole number of two-half-group scales because `headDim % 16
//! == 0`. A tile may cross a row boundary, but it cannot split a scale group;
//! packed values and scales are linear in the same HND order.
__host__ __device__ constexpr std::uint32_t compressedTransferHalfGroups(Nvfp4BoundaryKernelParams const& params)
{
    // Do not use std::min here. Its reference-returning overload ODR-uses the
    // namespace-scope constant in device code, which requires a device-side
    // definition even though the value is constexpr. The conditional keeps the
    // bound an immediate constant in both host and device compilation passes.
    auto const halfGroups = totalHalfGroupsPerRole(params);
    return halfGroups < kHalfGroupsPerTransfer ? halfGroups : kHalfGroupsPerTransfer;
}

//! Flush one compact NVFP4 Page/role range from CTA-local shared memory to its
//! concatenated region in the final mapped-Host compact Slot.
//!
//! BATCHEDCOPY ADAPTATION: the quantization loop produces one 32-bit packed
//! word and one shared scale byte per 16 values. Writing those results to Host
//! immediately made only one quarter / one half of the lanes issue stores and
//! coupled every quant warp to system-memory backpressure. This final phase
//! instead has all lanes stream aligned uint4 grains, just like batchedCopy.
//! A packed interval that is only 8-byte aligned uses its natural uint2 scale
//! groups; an arbitrary staging offset and every scale tail remain byte-exact.
//! No path writes cold padding.
__device__ void flushCompactRangeToHostForGroup(std::uint8_t const* compactStages, OffloadLayerTask const& task,
    std::uint32_t role, std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes,
    std::uint32_t packedDestinationOffset, std::uint32_t packedBytes, std::uint32_t scaleDestinationOffset,
    std::uint32_t scaleBytes, std::uint32_t groupThread, std::uint32_t groupThreads)
{
    auto const* packedSource = compactStages;
    auto* packedDestination = selectPackedOutput(task, role, halfGroupsPerRole) + packedDestinationOffset;
    bool const alignedPacked = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint4) == 0;
    bool const alignedPackedPair = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint2) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint2) == 0;
    std::uint32_t const packedVectorBytes = alignedPacked ? packedBytes - packedBytes % sizeof(uint4) : 0;
    std::uint32_t const packedPairBytes = alignedPackedPair ? packedBytes - packedBytes % sizeof(uint2) : 0;
    for (std::uint32_t grain = groupThread; grain < packedVectorBytes / sizeof(uint4); grain += groupThreads)
    {
        reinterpret_cast<uint4*>(packedDestination)[grain] = reinterpret_cast<uint4 const*>(packedSource)[grain];
    }
    for (std::uint32_t pair = packedVectorBytes / sizeof(uint2) + groupThread; pair < packedPairBytes / sizeof(uint2);
         pair += groupThreads)
    {
        reinterpret_cast<uint2*>(packedDestination)[pair] = reinterpret_cast<uint2 const*>(packedSource)[pair];
    }
    for (std::uint32_t byte = packedPairBytes + groupThread; byte < packedBytes; byte += groupThreads)
    {
        packedDestination[byte] = packedSource[byte];
    }

    auto const* scaleSource = compactStages + packedStageCapacityBytes;
    auto* scaleDestination = selectScaleOutput(task, role, halfGroupsPerRole) + scaleDestinationOffset;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(scaleDestination) % sizeof(uint4) == 0;
    std::uint32_t const scaleVectorBytes = alignedScale ? scaleBytes - scaleBytes % sizeof(uint4) : 0;
    for (std::uint32_t grain = groupThread; grain < scaleVectorBytes / sizeof(uint4); grain += groupThreads)
    {
        reinterpret_cast<uint4*>(scaleDestination)[grain] = reinterpret_cast<uint4 const*>(scaleSource)[grain];
    }
    for (std::uint32_t byte = scaleVectorBytes + groupThread; byte < scaleBytes; byte += groupThreads)
    {
        scaleDestination[byte] = scaleSource[byte];
    }
}

__device__ void flushCompactRangeToHost(std::uint8_t const* compactStages, OffloadLayerTask const& task,
    std::uint32_t role, std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes,
    std::uint32_t packedDestinationOffset, std::uint32_t packedBytes, std::uint32_t scaleDestinationOffset,
    std::uint32_t scaleBytes)
{
    flushCompactRangeToHostForGroup(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
        packedDestinationOffset, packedBytes, scaleDestinationOffset, scaleBytes, threadIdx.x, blockDim.x);
}

//! Join two consecutive eight-value FP8 words into one 16-byte GPU store.
__device__ __forceinline__ uint4 collectTwoFp8Words(std::uint64_t first, std::uint64_t second)
{
    return make_uint4(static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32U),
        static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32U));
}

// Unpack one lane's eight E2M1 values into four float2 values. "Eight" is the
// number of scalars, not the FP8 E4M3 dtype. The SM100 E2M1->FP16x2 PTX is
// adapted from ARCQuant/fused-MoE, then widened to float2 so the destination
// may independently be FP16, BF16, or FP8.
// Original dequant helper sources:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/arcquantFP4.cu
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
__device__ void unpackE2m1ToFloat(std::uint32_t packed, float2 (&values)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    std::uint32_t fp16Pairs[4];
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(fp16Pairs[0]), "=r"(fp16Pairs[1]), "=r"(fp16Pairs[2]), "=r"(fp16Pairs[3])
        : "r"(packed));

#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        values[i] = __half22float2(reinterpret_cast<__half2&>(fp16Pairs[i]));
    }
#endif
}

template <typename T>
__device__ void store16BitValues(T* output, std::uint32_t elementOffset, float2 const (&values)[4], float scale)
{
    // BATCHEDCOPY VECTOR-WIDTH ADAPTATION: construct the final raw values in
    // four 32-bit registers, then publish them through a uint4 pointer. A
    // PackedVec<T> assignment is numerically equivalent, but nvcc may scalarize
    // that store because PackedVec's type alignment is weaker than 16 bytes.
    // The explicit uint4 contract makes the final H2D-side GPU write STG.128
    // for both FP16 and BF16 specializations.
    std::uint32_t outputWords[4];
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        float2 const scaled = make_float2(values[i].x * scale, values[i].y * scale);
        if constexpr (std::is_same_v<T, half>)
        {
            half2 const pair = __float22half2_rn(scaled);
            outputWords[i] = reinterpret_cast<std::uint32_t const&>(pair);
        }
        else
        {
            __nv_bfloat162 const pair = __float22bfloat162_rn(scaled);
            outputWords[i] = reinterpret_cast<std::uint32_t const&>(pair);
        }
    }
    uint4 const outputGrain = make_uint4(outputWords[0], outputWords[1], outputWords[2], outputWords[3]);
    reinterpret_cast<uint4*>(output + elementOffset)[0] = outputGrain;
}

//! Quantize one complete 16-value FP8 grain in one lane while preserving the
//! accepted boundary scale contract exactly.
//!
//! The generic `cvt_warp_fp8_to_fp4` cannot be used here: its `SFScaleVal`
//! contract deliberately cancels the source FP8 scale before publishing the
//! block scale. Boundary compression instead restores the runtime FP8 domain
//! with `fp8ScaleQuantOrig`, then applies an independent calibrated
//! `nvfp4ScaleOrigQuant`. Restoring and reducing both eight-value halves
//! locally avoids the former adjacent-lane redistribution and warp
//! synchronization. The input uses the same `PackedVec<__nv_fp8_e4m3>` and
//! `__nv_fp8x2_e4m3 -> float2` conversion surface as the production
//! `cvt_warp_fp8_to_fp4` helper. Only that pair-conversion surface is reused:
//! the boundary path must retain its independent source-FP8 and
//! destination-NVFP4 global scales. The helper then rounds one E4M3 scale and
//! packs each half with the production `fp32_vec_to_e2m1` primitive.
__device__ uint2 quantizeFp8GrainToNvfp4(PackedVec<__nv_fp8_e4m3> const& grain, float fp8ScaleQuantOrig,
    float nvfp4ScaleOrigQuant, std::uint8_t* scaleOutput)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // NVFP4 REUSE: convert adjacent E4M3 values through the production packed
    // FP8x2 type instead of reconstructing and casting two scalar FP8 objects.
    // ALGORITHM INVARIANT: multiplication remains in FP32 and each restored
    // pair is rounded to FP16 before the unchanged 16-value absmax reduction.
    PackedVec<half> restored[2];
#pragma unroll
    for (std::uint32_t pair = 0; pair < 8; ++pair)
    {
        float2 values = static_cast<float2>(grain.elts[pair]);
        values.x *= fp8ScaleQuantOrig;
        values.y *= fp8ScaleQuantOrig;
        restored[pair / 4U].elts[pair % 4U] = __float22half2_rn(values);
    }

    auto firstHalfMax = cuda_abs(restored[0].elts[0]);
    auto secondHalfMax = cuda_abs(restored[1].elts[0]);
#pragma unroll
    for (std::uint32_t i = 1; i < 4; ++i)
    {
        firstHalfMax = cuda_max(firstHalfMax, cuda_abs(restored[0].elts[i]));
        secondHalfMax = cuda_max(secondHalfMax, cuda_abs(restored[1].elts[i]));
    }
    auto const localMax = cuda_max(firstHalfMax, secondHalfMax);
    float const vecMax = static_cast<float>(cuda_max(localMax.x, localMax.y));

    float scaleValue = nvfp4ScaleOrigQuant * (vecMax * reciprocal_approximate_ftz(6.0F));
    __nv_fp8_e4m3 const roundedScale(scaleValue);
    *scaleOutput = roundedScale.__x;
    scaleValue = static_cast<float>(roundedScale);
    float const outputScale = vecMax != 0.0F
        ? reciprocal_approximate_ftz(scaleValue * reciprocal_approximate_ftz(nvfp4ScaleOrigQuant))
        : 0.0F;

    std::uint32_t packed[2];
#pragma unroll
    for (std::uint32_t halfGroup = 0; halfGroup < 2; ++halfGroup)
    {
        float2 values[4];
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            values[i] = __half22float2(restored[halfGroup].elts[i]);
            values[i].x *= outputScale;
            values[i].y *= outputScale;
        }
        packed[halfGroup] = fp32_vec_to_e2m1(values);
    }
    return make_uint2(packed[0], packed[1]);
#else
    static_cast<void>(grain);
    static_cast<void>(fp8ScaleQuantOrig);
    static_cast<void>(nvfp4ScaleOrigQuant);
    static_cast<void>(scaleOutput);
    return make_uint2(0U, 0U);
#endif
}

//! Restore the natural 16-value NVFP4 scale group shared by every onboard
//! path. Transport may batch two groups in a uint4, but correctness is defined
//! by this exact uint2 unit.
template <typename T>
__device__ float onboardDequantScale(
    std::uint8_t encodedScale, Nvfp4BoundaryKernelParams const& params, std::uint32_t role)
{
    __nv_fp8_e4m3 blockScale;
    blockScale.__x = encodedScale;
    float scale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        scale *= params.fp8ScaleOrigQuant[role];
    }
    return scale;
}

template <typename T>
__device__ void restoreNvfp4Pair(uint2 packedPair, T* output, std::uint32_t firstHalfGroup, float dequantScale)
{
    std::uint32_t const packedWords[2] = {packedPair.x, packedPair.y};
    if constexpr (!std::is_same_v<T, __nv_fp8_e4m3>)
    {
#pragma unroll
        for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
        {
            float2 values[4];
            unpackE2m1ToFloat(packedWords[laneInScale], values);
            store16BitValues(output, (firstHalfGroup + laneInScale) * kElementsPerLane, values, dequantScale);
        }
    }
    else
    {
        std::uint64_t packedFp8[2];
#pragma unroll
        for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
        {
            float2 values[4];
            unpackE2m1ToFloat(packedWords[laneInScale], values);
#pragma unroll
            for (std::uint32_t i = 0; i < 4; ++i)
            {
                values[i].x *= dequantScale;
                values[i].y *= dequantScale;
            }
            packedFp8[laneInScale] = fp32_vec_to_e4m3(values);
        }
        reinterpret_cast<uint4*>(output)[firstHalfGroup / 2U] = collectTwoFp8Words(packedFp8[0], packedFp8[1]);
    }
}

//! FP16/BF16 GPU Page -> Host NVFP4 with bounded compact-output tiles.
//!
//! Each tile is sized from the compact scale interval, uses the production
//! FP16/BF16-to-NVFP4 leaf primitive, and flushes packed uint4/uint2 bodies plus
//! exact scale-byte tails to mapped Host memory. One kernel handles every
//! admitted geometry.
template <typename T>
__global__ void offloadFrom16BitTiledKernel(
    std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = totalHalfGroupsPerRole(params);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerRole(tileHalfGroups);

    // BATCHEDCOPY REUSE: the raw source retains the same four-stage 16-byte
    // cp.async ring. BOUNDARY ADAPTATION: compact shared storage is bounded by
    // one compressed-output tile instead of growing with the Page size.
    __shared__ __align__(16) PackedVec<T> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerRole;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerRole - firstHalfGroup);
        std::uint32_t const iterations = (halfGroups + kThreadsPerBlock - 1U) / kThreadsPerBlock;

        for (std::uint32_t iteration = 0; iteration < iterations + kAsyncStages; ++iteration)
        {
            std::uint32_t const stage = iteration % kAsyncStages;
            if (iteration >= kAsyncStages)
            {
                std::uint32_t const transformIteration = iteration - kAsyncStages;
                std::uint32_t const localHalfGroup = kThreadsPerBlock * transformIteration + threadIdx.x;
                cp_async_wait_group<kAsyncStages - 1>();
                if (localHalfGroup < halfGroups)
                {
                    std::uint32_t const globalHalfGroup = firstHalfGroup + localHalfGroup;
                    std::uint32_t const laneInScale = globalHalfGroup & 1U;

                    // NVFP4 DIRECT REUSE: unchanged production block quant.
                    // TILED ADAPTATION: every even tile boundary is also a
                    // block-scale boundary, so the local scale index is linear.
                    std::uint32_t const localScaleOffset = localHalfGroup >> 1U;
                    std::uint8_t* scale = laneInScale == 0 ? scaleStages + localScaleOffset : nullptr;
                    PackedVec<T> input = rawStages[stage][threadIdx.x];
                    packedStages[localHalfGroup] = cvt_warp_fp16_to_fp4<T, kElementsPerBlockScale, false>(
                        input, params.nvfp4ScaleOrigQuant[role], scale);
                }
            }

            std::uint32_t const localLoadHalfGroup = kThreadsPerBlock * iteration + threadIdx.x;
            bool const valid = localLoadHalfGroup < halfGroups;
            auto const* rawInput = reinterpret_cast<PackedVec<T> const*>(selectRawInput<T>(task, role));
            auto const* source = valid ? rawInput + firstHalfGroup + localLoadHalfGroup : rawInput;
            copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
            cp_async_commit_group();
        }

        // Drain the ring before reusing either shared region for the next
        // tile. The first barrier publishes quant results to all writer lanes;
        // the second prevents the next tile from overwriting an active flush.
        cp_async_wait_group<0>();
        __syncthreads();
        flushCompactRangeToHost(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
            packedBytesPerRole(firstHalfGroup), packedBytesPerRole(halfGroups), firstHalfGroup / 2U, halfGroups / 2U);
        __syncthreads();
    }
#endif
}

//! FP8 E4M3 GPU Page -> Host NVFP4 with bounded compact-output tiles.
//!
//! The output tile is identical to the 16-bit path; only its raw input is
//! smaller. One lane restores and quantizes one complete 16-value group while
//! preserving independent source-FP8 and destination-NVFP4 scales.
__global__ void offloadFromFp8TiledKernel(
    std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = totalHalfGroupsPerRole(params);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerRole(tileHalfGroups);

    // BATCHEDCOPY ADAPTATION: cp.async still moves one 16-byte grain. Naming
    // the shared object as production PackedVec exposes its eight FP8x2 pairs
    // directly to the quantizer without changing bytes, alignment, or layout.
    __shared__ __align__(16) PackedVec<__nv_fp8_e4m3> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerRole;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerRole - firstHalfGroup);
        std::uint32_t const firstGrain = firstHalfGroup / 2U;
        std::uint32_t const grains = halfGroups / 2U;
        std::uint32_t const iterations = (grains + kThreadsPerBlock - 1U) / kThreadsPerBlock;

        for (std::uint32_t iteration = 0; iteration < iterations + kAsyncStages; ++iteration)
        {
            std::uint32_t const stage = iteration % kAsyncStages;
            if (iteration >= kAsyncStages)
            {
                std::uint32_t const transformIteration = iteration - kAsyncStages;
                cp_async_wait_group<kAsyncStages - 1>();
                // The async load already assigns one complete 16-value scale
                // group to this lane. Consume it in place; no adjacent-lane
                // redistribution or warp-wide synchronization is required.
                std::uint32_t const localGrain = kThreadsPerBlock * transformIteration + threadIdx.x;
                if (localGrain < grains)
                {
                    uint2 const packed = quantizeFp8GrainToNvfp4(rawStages[stage][threadIdx.x],
                        params.fp8ScaleQuantOrig[role], params.nvfp4ScaleOrigQuant[role], scaleStages + localGrain);
                    reinterpret_cast<uint2*>(packedStages)[localGrain] = packed;
                }
            }

            std::uint32_t const localLoadGrain = kThreadsPerBlock * iteration + threadIdx.x;
            bool const valid = localLoadGrain < grains;
            auto const* rawInput
                = reinterpret_cast<PackedVec<__nv_fp8_e4m3> const*>(selectRawInput<__nv_fp8_e4m3>(task, role));
            auto const* source = valid ? rawInput + firstGrain + localLoadGrain : rawInput;
            copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
            cp_async_commit_group();
        }

        cp_async_wait_group<0>();
        __syncthreads();
        flushCompactRangeToHost(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
            packedBytesPerRole(firstHalfGroup), packedBytesPerRole(halfGroups), firstHalfGroup / 2U, halfGroups / 2U);
        __syncthreads();
    }
#endif
}

//! Densely load one compact packed+scale interval from mapped Host memory into
//! CTA-local shared memory.
//!
//! Packed bytes use aligned uint4 grains, natural uint2 scale-group tails, or a
//! byte path for arbitrary staging offsets. Scale bytes use the vector path
//! when both endpoints are aligned; the generic byte tail preserves the compact
//! format's one-byte scale contract. This is the H2D counterpart of
//! `flushCompactRangeToHost` and deliberately finishes with a CTA barrier before
//! any dequantization reads the tile.
__device__ void loadCompactRangeFromHostForGroup(std::uint8_t* compactStages, OnboardLayerTask const& task,
    std::uint32_t role, std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes,
    std::uint32_t packedSourceOffset, std::uint32_t packedBytes, std::uint32_t scaleSourceOffset,
    std::uint32_t scaleBytes, std::uint32_t groupThread, std::uint32_t groupThreads)
{
    auto const* packedSource = selectPackedInput(task, role, halfGroupsPerRole) + packedSourceOffset;
    auto* packedDestination = compactStages;
    auto const* scaleSource = selectScaleInput(task, role, halfGroupsPerRole) + scaleSourceOffset;
    auto* scaleDestination = compactStages + packedStageCapacityBytes;
    bool const alignedPacked = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint4) == 0;
    bool const alignedPackedPair = reinterpret_cast<std::uintptr_t>(packedSource) % sizeof(uint2) == 0
        && reinterpret_cast<std::uintptr_t>(packedDestination) % sizeof(uint2) == 0;
    std::uint32_t const packedVectorBytes = alignedPacked ? packedBytes - packedBytes % sizeof(uint4) : 0;
    std::uint32_t const packedPairBytes = alignedPackedPair ? packedBytes - packedBytes % sizeof(uint2) : 0;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleSource) % sizeof(uint4) == 0
        && reinterpret_cast<std::uintptr_t>(scaleDestination) % sizeof(uint4) == 0;
    std::uint32_t const scaleVectorBytes = alignedScale ? scaleBytes - scaleBytes % sizeof(uint4) : 0;
    std::uint32_t const packedGrains = packedVectorBytes / sizeof(uint4);
    std::uint32_t const scaleGrains = scaleVectorBytes / sizeof(uint4);
    std::uint32_t const totalGrains = packedGrains + scaleGrains;
    std::uint32_t const iterations = (totalGrains + groupThreads - 1U) / groupThreads;

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (iteration >= kHostLoadAsyncStages)
        {
            cp_async_wait_group<kHostLoadAsyncStages - 1>();
        }
        std::uint32_t const grain = groupThreads * iteration + groupThread;
        bool const valid = grain < totalGrains;
        auto const* source = reinterpret_cast<uint4 const*>(packedSource);
        auto* destination = reinterpret_cast<uint4*>(packedDestination);
        if (valid && grain < packedGrains)
        {
            source += grain;
            destination += grain;
        }
        else if (valid)
        {
            source = reinterpret_cast<uint4 const*>(scaleSource) + grain - packedGrains;
            destination = reinterpret_cast<uint4*>(scaleDestination) + grain - packedGrains;
        }
        copyAsyncGlobalToShared(destination, source, valid);
        cp_async_commit_group();
    }
    cp_async_wait_group<0>();

    // D%16 makes packed intervals an exact number of uint2 scale groups.
    // Aligned Slots use uint4/uint2; an arbitrary staging suballocation uses
    // the byte tail without changing the compact record.
    for (std::uint32_t pair = packedVectorBytes / sizeof(uint2) + groupThread; pair < packedPairBytes / sizeof(uint2);
         pair += groupThreads)
    {
        reinterpret_cast<uint2*>(packedDestination)[pair] = reinterpret_cast<uint2 const*>(packedSource)[pair];
    }
    for (std::uint32_t byte = packedPairBytes + groupThread; byte < packedBytes; byte += groupThreads)
    {
        packedDestination[byte] = packedSource[byte];
    }
    for (std::uint32_t byte = scaleVectorBytes + groupThread; byte < scaleBytes; byte += groupThreads)
    {
        scaleDestination[byte] = scaleSource[byte];
    }
}

__device__ void loadCompactRangeFromHost(std::uint8_t* compactStages, OnboardLayerTask const& task, std::uint32_t role,
    std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes, std::uint32_t packedSourceOffset,
    std::uint32_t packedBytes, std::uint32_t scaleSourceOffset, std::uint32_t scaleBytes)
{
    loadCompactRangeFromHostForGroup(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
        packedSourceOffset, packedBytes, scaleSourceOffset, scaleBytes, threadIdx.x, blockDim.x);
    __syncthreads();
}

//! Host NVFP4 -> runtime GPU Page with bounded transfer/dequant tiles.
//!
//! Packed uint4 bodies, natural uint2 tails, and byte-scale tails share this
//! path for FP16, BF16, and FP8 destinations.
template <typename T>
__global__ void onboardTiledKernel(
    std::array<Nvfp4BoundaryOnboardPageTask, kMaxTasksPerLaunch> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers,
    std::uint8_t const* coldBase, std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("griddepcontrol.launch_dependents;\n");

    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = totalHalfGroupsPerRole(params);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedStagingBytesPerRole(tileHalfGroups);
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* rawOutput = selectRawOutput<T>(task, role);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstHalfGroup = blockIdx.x * tileHalfGroups; firstHalfGroup < halfGroupsPerRole;
         firstHalfGroup += gridDim.x * tileHalfGroups)
    {
        std::uint32_t const halfGroups = std::min(tileHalfGroups, halfGroupsPerRole - firstHalfGroup);
        std::uint32_t const packedBytes = packedBytesPerRole(halfGroups);
        std::uint32_t const scaleBytes = halfGroups / 2U;

        // FUSED H2D PHASE: all packed data and block scales reach shared
        // through dense mapped-Host reads before dequantization starts.
        loadCompactRangeFromHost(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
            packedBytesPerRole(firstHalfGroup), packedBytes, firstHalfGroup / 2U, scaleBytes);

        auto const* packedStages = reinterpret_cast<uint4 const*>(compactStages);
        auto const* scaleStages = compactStages + packedStageCapacityBytes;
        std::uint32_t const packedGrains = packedBytes / sizeof(uint4);
        for (std::uint32_t localGrain = threadIdx.x; localGrain < packedGrains; localGrain += blockDim.x)
        {
            uint4 const packedGrain = packedStages[localGrain];
            std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
#pragma unroll
            for (std::uint32_t pair = 0; pair < 2; ++pair)
            {
                std::uint32_t const localScaleGroup = localGrain * 2U + pair;
                std::uint32_t const firstPairHalfGroup = firstHalfGroup + localScaleGroup * 2U;

                restoreNvfp4Pair(make_uint2(packedWords[pair * 2U], packedWords[pair * 2U + 1U]), rawOutput,
                    firstPairHalfGroup, onboardDequantScale<T>(scaleStages[localScaleGroup], params, role));
            }
        }
        if (packedBytes % sizeof(uint4) != 0U && threadIdx.x == 0)
        {
            std::uint32_t const localScaleGroup = packedGrains * 2U;
            std::uint32_t const firstPairHalfGroup = firstHalfGroup + localScaleGroup * 2U;
            restoreNvfp4Pair(reinterpret_cast<uint2 const*>(compactStages)[localScaleGroup], rawOutput,
                firstPairHalfGroup, onboardDequantScale<T>(scaleStages[localScaleGroup], params, role));
        }

        // All consumers must finish before the next compact tile overwrites shared
        // memory.
        __syncthreads();
    }
#endif
}

void validateParams(Nvfp4BoundaryKernelParams const& params, bool useFp8)
{
    TLLM_CHECK_WITH_INFO(params.numKvHeads > 0, "numKvHeads must be positive");
    TLLM_CHECK_WITH_INFO(params.tokensPerPage > 0, "tokensPerPage must be positive");
    TLLM_CHECK_WITH_INFO(params.headDim > 0 && params.headDim % kElementsPerBlockScale == 0,
        "headDim must be positive and divisible by 16, got %d", params.headDim);

    std::uint64_t const rows
        = static_cast<std::uint64_t>(params.numKvHeads) * static_cast<std::uint64_t>(params.tokensPerPage);
    // One half-group contributes four packed bytes per role; the complete
    // two-role record contributes eight packed bytes plus one scale byte.
    // Keep every derived compact offset representable by the uint32 device ABI.
    constexpr std::uint64_t maxHalfGroups = std::numeric_limits<std::uint32_t>::max() / 9U;
    std::uint64_t const halfGroupsPerRow = static_cast<std::uint64_t>(params.headDim / kElementsPerLane);
    TLLM_CHECK_WITH_INFO(rows <= maxHalfGroups / halfGroupsPerRow,
        "Page geometry exceeds the 32-bit compact-offset range: "
        "heads=%d, tokens=%d, headDim=%d",
        params.numKvHeads, params.tokensPerPage, params.headDim);

    for (std::uint32_t role = 0; role < 2; ++role)
    {
        TLLM_CHECK_WITH_INFO(std::isfinite(params.nvfp4ScaleOrigQuant[role]) && params.nvfp4ScaleOrigQuant[role] > 0.0F,
            "NVFP4 original-to-quantized scale must be finite and positive");
        TLLM_CHECK_WITH_INFO(std::isfinite(params.nvfp4ScaleQuantOrig[role]) && params.nvfp4ScaleQuantOrig[role] > 0.0F,
            "NVFP4 quantized-to-original scale must be finite and positive");
        if (useFp8)
        {
            TLLM_CHECK_WITH_INFO(std::isfinite(params.fp8ScaleOrigQuant[role]) && params.fp8ScaleOrigQuant[role] > 0.0F,
                "FP8 original-to-quantized scale must be finite and positive");
            TLLM_CHECK_WITH_INFO(std::isfinite(params.fp8ScaleQuantOrig[role]) && params.fp8ScaleQuantOrig[role] > 0.0F,
                "FP8 quantized-to-original scale must be finite and positive");
        }
    }
}

template <typename Pointer>
void validateAlignedPointer(Pointer pointer, std::uintptr_t alignment, char const* name)
{
    TLLM_CHECK_WITH_INFO(pointer != nullptr, "%s must not be null", name);
    TLLM_CHECK_WITH_INFO(
        reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0, "%s must be aligned to %zu bytes", name, alignment);
}

template <typename Task>
void validatePages(std::vector<Task> const& pages)
{
    for (auto const& page : pages)
    {
        TLLM_CHECK_WITH_INFO(
            page.gpuPageIndex >= 0 && page.coldPageIndex >= 0, "GPU and cold Base Page indices must be non-negative");
    }
}

void validateLayerPlan(Nvfp4BoundaryLayerPlan const& layer, std::size_t coldPageBytes, bool useFp8)
{
    validateParams(layer.params, useFp8);
    validateAlignedPointer(reinterpret_cast<void const*>(layer.rawKBase), alignof(uint4), "rawKBase");
    validateAlignedPointer(reinterpret_cast<void const*>(layer.rawVBase), alignof(uint4), "rawVBase");
    TLLM_CHECK_WITH_INFO(layer.rawKSlotBytes > 0 && layer.rawVSlotBytes > 0, "GPU K/V Slot strides must be positive");
    TLLM_CHECK_WITH_INFO(layer.rawKSlotBytes % alignof(uint4) == 0 && layer.rawVSlotBytes % alignof(uint4) == 0,
        "GPU K/V Slot strides must be aligned to %zu bytes", alignof(uint4));
    TLLM_CHECK_WITH_INFO(
        layer.coldOffset % alignof(uint4) == 0, "Layer record offset must be aligned to %zu bytes", alignof(uint4));
    std::size_t const halfGroups = static_cast<std::size_t>(layer.params.numKvHeads)
        * static_cast<std::size_t>(layer.params.tokensPerPage)
        * static_cast<std::size_t>(layer.params.headDim / kElementsPerLane);
    std::size_t const recordBytes = halfGroups * 9U;
    TLLM_CHECK_WITH_INFO(layer.coldOffset <= coldPageBytes && recordBytes <= coldPageBytes - layer.coldOffset,
        "Layer compact record exceeds the cold Page stride");
}

//! Launch the single 256-descriptor kernel ABI through the raw-argument runtime
//! API. A full chunk points the parameter copy directly at the caller's
//! contiguous vector. Only an incomplete chunk is zero-padded in a stack array;
//! grid.z ensures device code never observes the padding.
//!
//! `cudaLaunchKernelExC` keeps that raw-argument fast path while adding the PDL
//! attribute that current batchedCopy's plain `cuLaunchKernel` omits.
template <typename Task, typename Kernel, typename ColdPointer>
void launchBoundaryBatch(Kernel kernel, Task const* tasks, std::uint32_t count, dim3 grid, dim3 block,
    std::uint32_t dynamicSmemBytes, std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const& layers,
    ColdPointer coldBase, std::size_t coldPageBytes, std::uint32_t numLayers, cudaStream_t stream)
{
    static_assert(std::is_trivially_copyable_v<std::array<Task, kMaxTasksPerLaunch>>);
    static_assert(sizeof(std::array<Task, kMaxTasksPerLaunch>) == sizeof(Task) * kMaxTasksPerLaunch);
    TLLM_CHECK(count > 0 && count <= kMaxTasksPerLaunch);

    // BATCHEDCOPY DIRECT REUSE: complete descriptor batches are passed from
    // the vector without copying. Only a partial chunk needs a padded by-value
    // kernel argument; value initialization keeps its unused
    // object representation deterministic for sanitizers and launch tracing.
    std::array<Task, kMaxTasksPerLaunch> tail{};
    void const* taskArgument = tasks;
    if (count < kMaxTasksPerLaunch)
    {
        std::copy_n(tasks, count, tail.begin());
        taskArgument = tail.data();
    }

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL() ? 1 : 0;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = dynamicSmemBytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;

    // Opt in explicitly to the configure-time bounded compact-tile size so a
    // legal geometry is not silently limited by CUDA's default shared-memory
    // carveout.
    if (dynamicSmemBytes != 0)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSmemBytes));
    }

    void* arguments[] = {const_cast<void*>(taskArgument), const_cast<Nvfp4BoundaryLayerPlan*>(layers.data()), &coldBase,
        &coldPageBytes, &numLayers};
    TLLM_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), arguments));
}

//! Submit disjoint Page descriptors in fixed-capacity chunks. The final chunk
//! uses the same kernel ABI as every full chunk; its actual count remains in
//! grid.z.
template <typename Task, typename LaunchBatch>
void launchTaskBatches(std::vector<Task> const& tasks, LaunchBatch const& launchBatch)
{
    std::size_t offset = 0;
    while (offset < tasks.size())
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(tasks.size() - offset, kMaxTasksPerLaunch));
        launchBatch(tasks.data() + offset, count);
        offset += count;
    }
}

//! One general SM100 launch policy covers every admitted Page geometry:
//!
//!   FP16/BF16 -> NVFP4 + D2H: one bounded tiled CTA per Page/role
//!   FP8       -> NVFP4 + D2H: one bounded tiled CTA per Page/role
//!   H2D + NVFP4 -> FP16/BF16: one tiled CTA per Page/role
//!   H2D + NVFP4 -> FP8: one tiled CTA per Page/role
//!
//! grid.y expands all layers and K/V. Geometry changes only the number of
//! linear half-groups consumed by the bounded in-kernel tile loop; it never
//! selects another kernel family.

template <typename T>
void launchOffloadFrom16Bit(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            launchBoundaryBatch(offloadFrom16BitTiledKernel<T>, taskData, count, grid, block,
                compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                plan.numLayers, stream);
        });
}

void launchOffloadFromFp8(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            launchBoundaryBatch(offloadFromFp8TiledKernel, taskData, count, grid, block,
                compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                plan.numLayers, stream);
        });
}

template <typename T>
void launchOnboard(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t const* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            launchBoundaryBatch(onboardTiledKernel<T>, taskData, count, grid, block,
                compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                plan.numLayers, stream);
        });
}

//! Preserve StorageManager's rollback contract when a multi-chunk launch
//! fails synchronously. Validation happens before this helper. On success it
//! remains fully asynchronous; only the exceptional path drains work already
//! submitted to the owner stream before destination Slots may be recycled.
template <typename Launch>
void launchAndDrainOnFailure(cudaStream_t stream, Launch const& launch)
{
    try
    {
        launch();
    }
    catch (...)
    {
        cudaError_t const drainStatus = cudaStreamSynchronize(stream);
        if (drainStatus != cudaSuccess)
        {
            // A prior chunk reached the stream but failed asynchronously, so
            // neither source nor destination ownership can be established.
            // Returning to the codec would let KVCM recycle potentially live
            // Slots. Treat the CUDA context as poisoned and fail-stop instead.
            TLLM_LOG_ERROR("NVFP4 boundary rollback drain failed: %s", cudaGetErrorString(drainStatus));
            std::terminate();
        }
        throw;
    }
}

} // namespace

Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(
    std::vector<Nvfp4BoundaryLayerPlan> const& layers, std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType)
{
    TLLM_CHECK_WITH_INFO(isCurrentDeviceSm100FamilyCached(), "NVFP4 boundary kernels require an SM100-family GPU");
    TLLM_CHECK_WITH_INFO(!layers.empty(), "NVFP4 boundary launch requires at least one Attention layer");
    TLLM_CHECK_WITH_INFO(layers.size() <= kMaxLayersPerLaunch,
        "NVFP4 boundary launch supports at most %u local "
        "Attention layers, got %zu",
        kMaxLayersPerLaunch, layers.size());
    TLLM_CHECK_WITH_INFO(coldPageBytes > 0, "Cold Page stride must be positive");
    TLLM_CHECK_WITH_INFO(
        coldPageBytes % alignof(uint4) == 0, "Cold Page stride must be aligned to %zu bytes", alignof(uint4));

    bool useFp8 = false;
    switch (runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
    case Nvfp4BoundaryRuntimeType::kBfloat16: break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3: useFp8 = true; break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }

    Nvfp4BoundaryPreparedPlan plan;
    plan.numLayers = static_cast<std::uint32_t>(layers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    std::copy(layers.begin(), layers.end(), plan.layers.begin());
    for (auto const& layer : layers)
    {
        validateLayerPlan(layer, coldPageBytes, useFp8);
        plan.maxTileHalfGroups = std::max(plan.maxTileHalfGroups, compressedTransferHalfGroups(layer.params));
    }
    return plan;
}

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream)
{
    if (pages.empty())
    {
        return;
    }
    validatePages(pages);
    TLLM_CHECK_WITH_INFO(coldBase != nullptr, "coldBase must not be null");
    switch (plan.runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        launchAndDrainOnFailure(
            stream, [&] { launchOffloadFrom16Bit<half>(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        launchAndDrainOnFailure(stream,
            [&] { launchOffloadFrom16Bit<__nv_bfloat16>(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        launchAndDrainOnFailure(
            stream, [&] { launchOffloadFromFp8(pages, plan, static_cast<std::uint8_t*>(coldBase), stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream)
{
    if (pages.empty())
    {
        return;
    }
    validatePages(pages);
    TLLM_CHECK_WITH_INFO(coldBase != nullptr, "coldBase must not be null");
    switch (plan.runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        launchAndDrainOnFailure(
            stream, [&] { launchOnboard<half>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        launchAndDrainOnFailure(stream,
            [&] { launchOnboard<__nv_bfloat16>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        launchAndDrainOnFailure(stream,
            [&] { launchOnboard<__nv_fp8_e4m3>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
