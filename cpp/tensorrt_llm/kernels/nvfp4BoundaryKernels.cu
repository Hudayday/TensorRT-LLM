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
constexpr std::uint32_t kMinTasksPerLaunch = 32;
constexpr std::uint32_t kMaxTasksPerLaunch = 256;
// The layer plan is passed by value with the Page descriptors. This avoids a
// second metadata upload and gives one launch direct access to every layer's
// addresses and scales. Current TRT-LLM models remain well below 128 local
// Attention layers; reject a larger plan synchronously instead of silently
// returning to a per-layer host launch loop.
constexpr std::uint32_t kMaxLayersPerLaunch = kNvfp4BoundaryMaxLayersPerLaunch;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;
constexpr std::uint32_t kNativeVScaleRows = 4;
// A 1-KiB scale interval bounds the ping-pong shared working set without
// changing compact bytes, arithmetic, or the public task contract.
constexpr std::uint32_t kTargetScaleTransferBytes = 1024;
constexpr std::size_t kModernKernelParameterLimit = 32764;
constexpr std::int32_t kStandardNumKvHeads = 8;
constexpr std::int32_t kStandardTokensPerPage = 64;
constexpr std::int32_t kStandardHeadDim = 128;
constexpr std::uint32_t kMaxWholePageCompactStagingBytes = 36U * 1024U;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(kTargetScaleTransferBytes > 0 && kTargetScaleTransferBytes % sizeof(uint4) == 0,
    "Compact scale transfer must remain a positive 16-byte multiple");
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
//! * Every raw/packed main-payload source load uses batchedCopy's 16-byte
//!   `cp.async.cg` grain. The retained tiled FP8 offload path consumes one
//!   complete 16-value scale group per lane; onboard consumes four packed
//!   words per grain. The tiled whole-Page paths retain their original
//!   two-round mapping.
//! * Offload uses compressed-output tiles for the standard H=8/P=64/D=128
//!   geometry and for non-standard Pages that exceed the whole-Page staging
//!   bound. Other non-standard Pages use the direct mapped-Host path. Onboard
//!   uses tiles for the standard geometry and the direct path otherwise.
//!   Dispatch is independent of Page-batch size.
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
//! Keeping this arithmetic in one helper makes the native task ABI one base
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

__host__ __device__ constexpr std::uint32_t compactStagingBytesPerRole(std::uint32_t halfGroups)
{
    return packedBytesPerRole(halfGroups) + scaleBytesPerRole(halfGroups);
}

__host__ __device__ constexpr std::uint32_t greatestCommonDivisor(std::uint32_t lhs, std::uint32_t rhs)
{
    while (rhs != 0)
    {
        std::uint32_t const remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

__host__ __device__ constexpr std::uint32_t roundUpTo(std::uint32_t value, std::uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

//! Select one transfer tile from the compressed NVFP4 payload rather than the
//! raw source size.
//!
//! A row contributes `headDim / 2` packed bytes and `headDim / 16` scale
//! bytes. Block scales are the smaller region, so they determine the tile: one
//! tile contains at least one complete batchedCopy wave (128 lanes * 16 B) of
//! scales. Packed data is exactly eight times larger and therefore contributes
//! eight full waves. Rows are rounded so every tile starts on both a native
//! token-4 V-scale boundary and a 16-byte multiple inside the scale region.
//! Model-like region bases are aligned; tiny legal records use the scalar
//! scale-transfer path. For
//! H=8/P=64/D=128 this selects 256 rows: 64 KiB FP16/BF16 input -> 16 KiB
//! packed + 2 KiB scales.
__host__ __device__ constexpr std::uint32_t compressedTransferRows(Nvfp4BoundaryKernelParams const& params)
{
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const scaleAlignmentRows = sizeof(uint4) / greatestCommonDivisor(sizeof(uint4), scalesPerRow);
    std::uint32_t const rowAlignment = scaleAlignmentRows < kNativeVScaleRows ? kNativeVScaleRows : scaleAlignmentRows;
    std::uint32_t const targetRows = (kTargetScaleTransferBytes + scalesPerRow - 1U) / scalesPerRow;
    std::uint32_t const totalRows
        = static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage);
    return std::min(roundUpTo(targetRows, rowAlignment), totalRows);
}

__host__ __device__ constexpr std::uint32_t compressedTransferHalfGroups(Nvfp4BoundaryKernelParams const& params)
{
    return compressedTransferRows(params) * static_cast<std::uint32_t>(params.headDim) / kElementsPerLane;
}

//! Collect four adjacent lanes' packed NVFP4 words into one 16-byte store.
//!
//! This helper belongs to the direct FP16/BF16 offload path used for
//! non-standard geometries whose compact payload fits the whole-Page staging
//! bound. The tiled path below trades an extra shared-memory round trip for
//! dense mapped-Host stores.
__device__ __forceinline__ uint4 collectFourPackedWords(std::uint32_t packed)
{
    std::uint32_t const lane = threadIdx.x & 31U;
    std::uint32_t const firstLane = lane & ~3U;
    std::uint32_t const groupMask = 0xFU << firstLane;
    return make_uint4(__shfl_sync(groupMask, packed, firstLane), __shfl_sync(groupMask, packed, firstLane + 1U),
        __shfl_sync(groupMask, packed, firstLane + 2U), __shfl_sync(groupMask, packed, firstLane + 3U));
}

//! Flush one native NVFP4 Page/role range from CTA-local shared memory to its
//! concatenated region in the final mapped-Host compact Slot.
//!
//! BATCHEDCOPY ADAPTATION: the quantization loop produces one 32-bit packed
//! word and one shared scale byte per 16 values. Writing those results to Host
//! immediately made only one quarter / one half of the lanes issue stores and
//! coupled every quant warp to system-memory backpressure. This final phase
//! instead has all lanes stream aligned uint4 grains, just like batchedCopy.
//! The rare scale tail (only for tiny test geometries) remains byte-exact and
//! never writes beyond the logical scale allocation.
__device__ void flushCompactRangeToHostForGroup(std::uint8_t const* compactStages, OffloadLayerTask const& task,
    std::uint32_t role, std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes,
    std::uint32_t packedDestinationOffset, std::uint32_t packedBytes, std::uint32_t scaleDestinationOffset,
    std::uint32_t scaleBytes, std::uint32_t groupThread, std::uint32_t groupThreads)
{
    auto const* packedSource = reinterpret_cast<uint4 const*>(compactStages);
    auto* packedDestination
        = reinterpret_cast<uint4*>(selectPackedOutput(task, role, halfGroupsPerRole) + packedDestinationOffset);
    for (std::uint32_t grain = groupThread; grain < packedBytes / sizeof(uint4); grain += groupThreads)
    {
        packedDestination[grain] = packedSource[grain];
    }

    auto const* scaleSource = compactStages + packedStageCapacityBytes;
    auto* scaleDestination = selectScaleOutput(task, role, halfGroupsPerRole) + scaleDestinationOffset;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleDestination) % sizeof(uint4) == 0;
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

__device__ void flushCompactToHost(
    std::uint8_t const* compactStages, OffloadLayerTask const& task, std::uint32_t role, std::uint32_t halfGroups)
{
    std::uint32_t const packedBytes = packedBytesPerRole(halfGroups);
    flushCompactRangeToHost(
        compactStages, task, role, halfGroups, packedBytes, 0, packedBytes, 0, scaleBytesPerRole(halfGroups));
}

//! Join two consecutive eight-value FP8 words into one 16-byte GPU store.
__device__ __forceinline__ uint4 collectTwoFp8Words(std::uint64_t first, std::uint64_t second)
{
    return make_uint4(static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32U),
        static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32U));
}

// K scales are linear. V scales use the token-4 order consumed by the native
// TRTLLM-gen NVFP4 KV path. `row` is the flattened [head, token] index; a
// normal tokens-per-Page value divisible by four therefore never mixes heads.
// Native writer (`quantizeAndWriteFP4KVCache`) source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h
__device__ std::uint32_t scaleOffset(
    std::uint32_t role, std::uint32_t row, std::uint32_t scaleInRow, std::uint32_t scalesPerRow)
{
    if (role == 0)
    {
        return row * scalesPerRow + scaleInRow;
    }
    return (row / 4) * (4 * scalesPerRow) + scaleInRow * 4 + row % 4;
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

// FP16/BF16 GPU Page -> Host NVFP4: fused quantization plus D2H.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOffloadCompress                  [BOUNDARY API]
//     -> launchOffloadFrom16Bit                         [BOUNDARY BATCHING]
//     -> offloadFrom16BitKernel                         [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async GPU raw -> shared          [BATCHEDCOPY
// ADAPTATION]
//        -> cvt_warp_fp16_to_fp4                        [NVFP4 DIRECT REUSE]
//        -> native scale offset + mapped-Host stores    [BOUNDARY ADAPTATION]
//
// The generic quantization hierarchy is invokeFP4Quantization ->
// quantize_with_block_size -> cvt_warp_fp16_to_fp4. This Page-specialized
// kernel calls the proven leaf primitive. Its async ring and load/commit/wait
// order come from KVCM V2 batchedCopy<N>. Quantization changes one 16-byte raw
// input into a 4-byte packed output plus one scale shared by two lanes, so the
// final mapped-Host stores are boundary-specific and stay in this kernel.
// Quantization hierarchy and leaf sources:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cu
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
// D2H/H2D pipeline source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N, typename T>
__global__ void offloadFrom16BitKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: announce the dependent grid as soon as this
    // producer CTA starts, then delay global-memory work until its own stream
    // predecessor has reached the matching dependency point. These hints do
    // not change correctness; they reduce serialization between consecutive
    // migration launches on SM90+.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: grid.z selects one disjoint Page and grid.y selects
    // K or V. grid.x remains batchedCopy's split dimension; Host transfers use
    // its explicit one-split low-bandwidth policy.
    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);

    // BATCHEDCOPY ADAPTATION: this is the same split/iteration arithmetic and
    // four-stage ring used by batchedCopy<N>. PackedVec<T> is exactly 16 B, so
    // each issued load uses batchedCopy's cp.async.cg width.
    __shared__ __align__(16) PackedVec<T> rawStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalIterations = (halfGroupsPerRole + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstHalfGroup = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x + threadIdx.x;
    std::uint32_t const endHalfGroup
        = std::min(firstHalfGroup + kThreadsPerBlock * maxIterationsPerCta, halfGroupsPerRole);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const halfGroup = firstHalfGroup + kThreadsPerBlock * transformIteration;

            // BATCHEDCOPY REUSE: wait until the oldest of four async groups is
            // visible in shared memory before consuming that ring stage.
            cp_async_wait_group<kAsyncStages - 1>();
            bool const valid = halfGroup < endHalfGroup;

            // NVFP4 CORRECTNESS: geometry validation makes every tail end on a
            // complete 8-half-group boundary. Therefore all lanes named by the
            // primitive's two-lane mask and collectFourPackedWords' four-lane
            // mask take this branch together; invalid groups do no quant work.
            if (valid)
            {
                std::uint32_t const laneInScale = halfGroup & 1U;
                std::uint32_t const scaleGroup = halfGroup >> 1U;
                std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                // NVFP4 DIRECT REUSE: the production primitive performs the
                // two-lane amax reduction, E4M3 block-scale generation, and
                // E2M1 packing. BOUNDARY ADAPTATION: packed values and scales
                // land directly in their CUDA-mapped Host destinations.
                PackedVec<T> input = rawStages[stage][threadIdx.x];
                std::uint8_t* scale = laneInScale == 0 ? selectScaleOutput(task, role, halfGroupsPerRole)
                        + scaleOffset(role, row, scaleInRow, scalesPerRow)
                                                       : nullptr;
                std::uint32_t const packed = cvt_warp_fp16_to_fp4<T, kElementsPerBlockScale, false>(
                    input, params.nvfp4ScaleOrigQuant[role], scale);
                uint4 const packedGrain = collectFourPackedWords(packed);
                if ((threadIdx.x & 3U) == 0)
                {
                    reinterpret_cast<uint4*>(selectPackedOutput(task, role, halfGroupsPerRole))[halfGroup / 4U]
                        = packedGrain;
                }
            }
        }

        std::uint32_t const loadHalfGroup = firstHalfGroup + kThreadsPerBlock * iteration;
        bool const valid = loadHalfGroup < endHalfGroup;
        auto const* rawInput = reinterpret_cast<PackedVec<T> const*>(selectRawInput<T>(task, role));
        // Avoid out-of-range pointer arithmetic on inactive tail lanes; the
        // predicated async helper does not dereference this fallback address.
        auto const* source = valid ? rawInput + loadHalfGroup : rawInput;
        // FUSED D2H SOURCE PIPELINE: adapt batchedCopy's GPU-global -> shared
        // cp.async stage. The actual device-to-Host crossing happens after
        // quantization at the mapped-Host packed/scale stores above.
        copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }

#endif
}

// FP8 E4M3 GPU Page -> Host NVFP4: fused restore, quantization, and D2H.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOffloadCompress                  [BOUNDARY API]
//     -> launchOffloadFromFp8                           [BOUNDARY BATCHING]
//     -> offloadFromFp8Kernel                           [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async GPU FP8 -> shared          [BATCHEDCOPY
// ADAPTATION]
//        -> source FP8 restore in registers             [BOUNDARY ADAPTATION]
//        -> cvt_warp_fp16_to_fp4                        [NVFP4 DIRECT REUSE]
//        -> native scale offset + mapped-Host stores    [BOUNDARY ADAPTATION]
//
// The source-FP8 inverse scale and target-NVFP4 scale are independent, so this
// kernel does not call the differently shaped cvt_warp_fp8_to_fp4 helper.
// Every lane loads one 16-byte cp.async.cg grain. After the wait, each warp
// redistributes its grains into two quantization rounds; adjacent lanes still
// quantize one 16-value block and share exactly one scale.
// Quantization primitive source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
// D2H/H2D pipeline source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N>
__global__ void offloadFromFp8Kernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers, std::uint8_t* coldBase,
    std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: enable programmatic dependent-launch overlap
    // while preserving stream-ordered visibility before any Page bytes move.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);

    // BATCHEDCOPY DIRECT REUSE: identical 16-byte grain and four-stage ring.
    // One grain contains two logical eight-value half-groups.
    __shared__ __align__(16) uint4 rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedBytesPerRole(halfGroupsPerRole);
    std::uint32_t const totalGrains = halfGroupsPerRole / 2U;
    std::uint32_t const totalIterations = (totalGrains + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstGrainBase = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x;
    std::uint32_t const firstGrain = firstGrainBase + threadIdx.x;
    std::uint32_t const endGrain = std::min(firstGrainBase + kThreadsPerBlock * maxIterationsPerCta, totalGrains);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            cp_async_wait_group<kAsyncStages - 1>();

            // BATCHEDCOPY ADAPTATION: each lane waits for its own async grain;
            // the warp barrier then makes all 32 grains visible to the paired
            // lane that consumes the other 8-byte half. This keeps the native
            // quant primitive's two-adjacent-lane reduction unchanged.
            __syncwarp();
            std::uint32_t const lane = threadIdx.x & 31U;
            std::uint32_t const warpThreadBase = threadIdx.x - lane;
            std::uint32_t const warpGrainBase = firstGrain + kThreadsPerBlock * transformIteration - lane;
#pragma unroll
            for (std::uint32_t round = 0; round < 2; ++round)
            {
                std::uint32_t const sourceThread = warpThreadBase + round * 16U + (lane >> 1U);
                std::uint32_t const sourceGrain = warpGrainBase + round * 16U + (lane >> 1U);
                std::uint32_t const halfGroup = sourceGrain * 2U + (lane & 1U);
                bool const valid = sourceGrain < endGrain;
                if (valid)
                {
                    uint4 const grain = rawStages[stage][sourceThread];
                    std::uint64_t const fp8Bytes = (lane & 1U) == 0
                        ? static_cast<std::uint64_t>(grain.x) | (static_cast<std::uint64_t>(grain.y) << 32U)
                        : static_cast<std::uint64_t>(grain.z) | (static_cast<std::uint64_t>(grain.w) << 32U);

                    std::uint32_t const laneInScale = halfGroup & 1U;
                    std::uint32_t const scaleGroup = halfGroup >> 1U;
                    std::uint32_t const scalesPerRow
                        = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                    std::uint32_t const row = scaleGroup / scalesPerRow;
                    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                    // BOUNDARY ADAPTATION: restore eight source FP8 values with
                    // their independent inverse scale before invoking the
                    // existing FP16->FP4 leaf primitive.
                    PackedVec<half> restored;
#pragma unroll
                    for (std::uint32_t i = 0; i < 4; ++i)
                    {
                        __nv_fp8_e4m3 lo;
                        __nv_fp8_e4m3 hi;
                        lo.__x = static_cast<std::uint8_t>(fp8Bytes >> (16U * i));
                        hi.__x = static_cast<std::uint8_t>(fp8Bytes >> (16U * i + 8U));
                        float const loValue = static_cast<float>(lo) * params.fp8ScaleQuantOrig[role];
                        float const hiValue = static_cast<float>(hi) * params.fp8ScaleQuantOrig[role];
                        restored.elts[i] = __floats2half2_rn(loValue, hiValue);
                    }

                    // NVFP4 DIRECT REUSE: production two-lane block
                    // quantization with independent boundary scales.
                    std::uint8_t* scale
                        = laneInScale == 0 ? scaleStages + scaleOffset(role, row, scaleInRow, scalesPerRow) : nullptr;
                    std::uint32_t const packed = cvt_warp_fp16_to_fp4<half, kElementsPerBlockScale, false>(
                        restored, params.nvfp4ScaleOrigQuant[role], scale);
                    packedStages[halfGroup] = packed;
                }
            }
        }

        std::uint32_t const loadGrain = firstGrain + kThreadsPerBlock * iteration;
        bool const valid = loadGrain < endGrain;
        auto const* rawInput = reinterpret_cast<uint4 const*>(selectRawInput<__nv_fp8_e4m3>(task, role));
        auto const* source = valid ? rawInput + loadGrain : rawInput;
        // FUSED D2H SOURCE PIPELINE: one exact batchedCopy-width grain carries
        // sixteen runtime FP8 values into the CTA ring.
        copyAsyncGlobalToShared(&rawStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }

    // FUSED D2H FINISH: FP8 restoration/quantization also uses the same
    // CTA-local native-layout tile and all-lane uint4 Host flush.
    __syncthreads();
    flushCompactToHost(compactStages, task, role, halfGroupsPerRole);
#endif
}

//! FP16/BF16 GPU Page -> Host NVFP4, compressed-output-tiled alternative.
//!
//! Unlike the direct mapped-Host specialization, this kernel stages a bounded
//! Page/role output region in shared memory. Each tile is sized from the
//! NVFP4 scale region: one 2-KiB scale wave plus eight 2-KiB packed waves for
//! the model-like geometry. The corresponding raw input is 3.56x larger. This
//! is still one kernel and one Page/role CTA; the barrier-delimited tile loop
//! is intentionally simple and does not add writer-warp overlap.
template <std::uint32_t N, typename T>
__global__ void offloadFrom16BitTiledKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ pages,
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
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const halfGroupsPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerLane;
    std::uint32_t const totalRows
        = static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage);
    std::uint32_t const tileRows = compressedTransferRows(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedBytesPerRole(tileHalfGroups);

    // BATCHEDCOPY REUSE: the raw source retains the same four-stage 16-byte
    // cp.async ring. BOUNDARY ADAPTATION: compact shared storage is bounded by
    // one compressed-output tile instead of growing with the Page size.
    __shared__ __align__(16) PackedVec<T> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstRow = blockIdx.x * tileRows; firstRow < totalRows; firstRow += gridDim.x * tileRows)
    {
        std::uint32_t const rows = std::min(tileRows, totalRows - firstRow);
        std::uint32_t const firstHalfGroup = firstRow * halfGroupsPerRow;
        std::uint32_t const halfGroups = rows * halfGroupsPerRow;
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
                    std::uint32_t const scaleGroup = globalHalfGroup >> 1U;
                    std::uint32_t const row = scaleGroup / scalesPerRow;
                    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                    // NVFP4 DIRECT REUSE: unchanged production block quant.
                    // TILED ADAPTATION: translate the native global scale
                    // offset into this tile's compact shared interval.
                    std::uint32_t const localScaleOffset
                        = scaleOffset(role, row, scaleInRow, scalesPerRow) - firstRow * scalesPerRow;
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
            packedBytesPerRole(firstHalfGroup), packedBytesPerRole(halfGroups), firstRow * scalesPerRow,
            rows * scalesPerRow);
        __syncthreads();
    }
#endif
}

//! FP8 E4M3 GPU Page -> Host NVFP4, compressed-output-tiled alternative.
//!
//! The output tile is identical to the 16-bit path; only its raw input is
//! smaller (1.78x the NVFP4 payload). One lane restores and quantizes one
//! complete 16-value group, preserving the whole-Page path's scale and packed
//! byte contract without its adjacent-lane redistribution.
template <std::uint32_t N>
__global__ void offloadFromFp8TiledKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ pages,
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
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const halfGroupsPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerLane;
    std::uint32_t const totalRows
        = static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage);
    std::uint32_t const tileRows = compressedTransferRows(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedBytesPerRole(tileHalfGroups);

    // BATCHEDCOPY ADAPTATION: cp.async still moves one 16-byte grain. Naming
    // the shared object as production PackedVec exposes its eight FP8x2 pairs
    // directly to the quantizer without changing bytes, alignment, or layout.
    __shared__ __align__(16) PackedVec<__nv_fp8_e4m3> rawStages[kAsyncStages][kThreadsPerBlock];
    extern __shared__ __align__(16) std::uint8_t compactStages[];
    auto* packedStages = reinterpret_cast<std::uint32_t*>(compactStages);
    auto* scaleStages = compactStages + packedStageCapacityBytes;

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstRow = 0; firstRow < totalRows; firstRow += tileRows)
    {
        std::uint32_t const rows = std::min(tileRows, totalRows - firstRow);
        std::uint32_t const firstHalfGroup = firstRow * halfGroupsPerRow;
        std::uint32_t const halfGroups = rows * halfGroupsPerRow;
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
                    std::uint32_t const scaleGroup = firstGrain + localGrain;
                    std::uint32_t const row = scaleGroup / scalesPerRow;
                    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
                    std::uint32_t const localScaleOffset
                        = scaleOffset(role, row, scaleInRow, scalesPerRow) - firstRow * scalesPerRow;
                    uint2 const packed
                        = quantizeFp8GrainToNvfp4(rawStages[stage][threadIdx.x], params.fp8ScaleQuantOrig[role],
                            params.nvfp4ScaleOrigQuant[role], scaleStages + localScaleOffset);
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
            packedBytesPerRole(firstHalfGroup), packedBytesPerRole(halfGroups), firstRow * scalesPerRow,
            rows * scalesPerRow);
        __syncthreads();
    }
#endif
}

//! Densely load one native packed+scale interval from mapped Host memory into
//! CTA-local shared memory.
//!
//! Packed bytes are always a complete set of aligned uint4 grains. Scale bytes
//! use the same path when the compact-region address is aligned; the generic
//! byte tail preserves the pre-existing native one-byte scale contract. This
//! helper is the H2D counterpart of `flushCompactRangeToHost` and deliberately
//! finishes with a CTA barrier before any dequantization reads the tile.
__device__ void loadCompactRangeFromHostForGroup(std::uint8_t* compactStages, OnboardLayerTask const& task,
    std::uint32_t role, std::uint32_t halfGroupsPerRole, std::uint32_t packedStageCapacityBytes,
    std::uint32_t packedSourceOffset, std::uint32_t packedBytes, std::uint32_t scaleSourceOffset,
    std::uint32_t scaleBytes, std::uint32_t groupThread, std::uint32_t groupThreads)
{
    auto const* packedSource = selectPackedInput(task, role, halfGroupsPerRole) + packedSourceOffset;
    auto const* scaleSource = selectScaleInput(task, role, halfGroupsPerRole) + scaleSourceOffset;
    bool const alignedScale = reinterpret_cast<std::uintptr_t>(scaleSource) % sizeof(uint4) == 0;
    std::uint32_t const scaleVectorBytes = alignedScale ? scaleBytes - scaleBytes % sizeof(uint4) : 0;
    std::uint32_t const packedGrains = packedBytes / sizeof(uint4);
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
        auto* destination = reinterpret_cast<uint4*>(compactStages);
        if (valid && grain < packedGrains)
        {
            source += grain;
            destination += grain;
        }
        else if (valid)
        {
            source = reinterpret_cast<uint4 const*>(scaleSource) + grain - packedGrains;
            destination = reinterpret_cast<uint4*>(compactStages + packedStageCapacityBytes) + grain - packedGrains;
        }
        copyAsyncGlobalToShared(destination, source, valid);
        cp_async_commit_group();
    }
    cp_async_wait_group<0>();

    // Only a non-vector tail or a byte-aligned scale region reaches this loop.
    // Model-like concatenated KVCM Slots take the uint4 path.
    auto* scaleDestination = compactStages + packedStageCapacityBytes;
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

// Host NVFP4 -> FP16/BF16 GPU Page: fused H2D plus dequantization.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOnboardDecompress                [BOUNDARY API]
//     -> launchOnboardTo16Bit                           [BOUNDARY BATCHING]
//     -> onboardTo16BitKernel                           [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async Host packed -> shared      [BATCHEDCOPY
// ADAPTATION]
//        -> native K/V scale load                       [BOUNDARY ADAPTATION]
//        -> unpackE2m1ToFloat                           [ARCQUANT/MOE
// ADAPTATION]
//        -> scaled 16-byte FP16/BF16 GPU store          [BOUNDARY ADAPTATION]
//
// unpackE2m1ToFloat adapts the cvt.rn.f16x2.e2m1x2 sequence from
// ARCQuant/fused-MoE; it does not call a MoE kernel. Every lane pipelines one
// 16-byte mapped-Host grain and expands its four packed words. Each adjacent
// word pair shares one scalar scale load. Packed and scale segments coexist in
// one concatenated Host Slot; K and V scale segments keep their native order.
// Dequant PTX sources:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/arcquantFP4.cu
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
// H2D/D2H pipeline source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N, typename T>
__global__ void onboardTo16BitKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers,
    std::uint8_t const* coldBase, std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: let the next migration grid launch early, but
    // fence this grid's mapped-Host reads behind its stream predecessor.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);

    // BATCHEDCOPY DIRECT REUSE: one 16-byte grain contains four packed E2M1
    // words, and all four pass through the same four-stage ring.
    __shared__ __align__(16) uint4 packedStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalGrains = halfGroupsPerRole / 4U;
    std::uint32_t const totalIterations = (totalGrains + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstGrainBase = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x;
    std::uint32_t const firstGrain = firstGrainBase + threadIdx.x;
    std::uint32_t const endGrain = std::min(firstGrainBase + kThreadsPerBlock * maxIterationsPerCta, totalGrains);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const grain = firstGrain + kThreadsPerBlock * transformIteration;
            cp_async_wait_group<kAsyncStages - 1>();
            if (grain < endGrain)
            {
                uint4 const packedGrain = packedStages[stage][threadIdx.x];
                std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
#pragma unroll
                for (std::uint32_t pair = 0; pair < 2; ++pair)
                {
                    std::uint32_t const firstHalfGroup = grain * 4U + pair * 2U;
                    std::uint32_t const scaleGroup = firstHalfGroup >> 1U;
                    std::uint32_t const scalesPerRow
                        = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                    std::uint32_t const row = scaleGroup / scalesPerRow;
                    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                    // BOUNDARY ADAPTATION: two adjacent packed words represent
                    // the same 16-value block and therefore reuse one native
                    // K-linear/V-token-4 scale lookup.
                    __nv_fp8_e4m3 blockScale;
                    blockScale.__x = selectScaleInput(
                        task, role, halfGroupsPerRole)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
                    float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];

#pragma unroll
                    for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
                    {
                        std::uint32_t const halfGroup = firstHalfGroup + laneInScale;
                        std::uint32_t const elementOffset = halfGroup * kElementsPerLane;
                        float2 values[4];
                        // NVFP4 DEQUANT: expand one packed word to registers.
                        unpackE2m1ToFloat(packedWords[pair * 2U + laneInScale], values);
                        // FUSED H2D FINISH: apply the shared scale and store
                        // directly into the final FP16/BF16 GPU Slot.
                        store16BitValues(selectRawOutput<T>(task, role), elementOffset, values, dequantScale);
                    }
                }
            }
        }

        std::uint32_t const loadGrain = firstGrain + kThreadsPerBlock * iteration;
        bool const valid = loadGrain < endGrain;
        auto const* packedInput = reinterpret_cast<uint4 const*>(selectPackedInput(task, role, halfGroupsPerRole));
        auto const* source = valid ? packedInput + loadGrain : packedInput;
        // FUSED H2D START: move four compact E2M1 words directly from mapped
        // Host memory with batchedCopy's exact 16-byte async grain.
        copyAsyncGlobalToShared(&packedStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

// Host NVFP4 -> FP8 E4M3 GPU Page: fused H2D, dequantization, and
// requantization.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOnboardDecompress                [BOUNDARY API]
//     -> launchOnboardToFp8                             [BOUNDARY BATCHING]
//     -> onboardToFp8Kernel                             [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async Host packed -> shared      [BATCHEDCOPY
// ADAPTATION]
//        -> native K/V scale load                       [BOUNDARY ADAPTATION]
//        -> unpackE2m1ToFloat                           [ARCQUANT/MOE
// ADAPTATION]
//        -> fp32_vec_to_e4m3 + 16-byte GPU store        [NVFP4/BOUNDARY]
//
// The 16-byte mapped-Host source pipeline and paired scale reuse match the
// FP16/BF16 onboard path. The numerical tail composes the NVFP4 inverse scale
// with destination FP8 quantization, then reuses fp32_vec_to_e4m3.
// Dequant PTX and FP8 packer sources:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
// H2D/D2H pipeline source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N>
__global__ void onboardToFp8Kernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ pages,
    std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const __grid_constant__ layers,
    std::uint8_t const* coldBase, std::size_t coldPageBytes, std::uint32_t numLayers)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: use the same programmatic grid dependency
    // protocol for the FP8 destination specialization.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const layerIndex = blockIdx.y / 2U;
    assert(layerIndex < numLayers);
    std::uint32_t const role = blockIdx.y % 2U;
    auto const& layer = layers[layerIndex];
    auto const params = layer.params;
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);

    // BATCHEDCOPY DIRECT REUSE: four-stage ring over exact 16-byte grains.
    __shared__ __align__(16) uint4 packedStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalGrains = halfGroupsPerRole / 4U;
    std::uint32_t const totalIterations = (totalGrains + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstGrainBase = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x;
    std::uint32_t const firstGrain = firstGrainBase + threadIdx.x;
    std::uint32_t const endGrain = std::min(firstGrainBase + kThreadsPerBlock * maxIterationsPerCta, totalGrains);

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const grain = firstGrain + kThreadsPerBlock * transformIteration;
            cp_async_wait_group<kAsyncStages - 1>();
            if (grain < endGrain)
            {
                uint4 const packedGrain = packedStages[stage][threadIdx.x];
                std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
#pragma unroll
                for (std::uint32_t pair = 0; pair < 2; ++pair)
                {
                    std::uint32_t const firstHalfGroup = grain * 4U + pair * 2U;
                    std::uint32_t const scaleGroup = firstHalfGroup >> 1U;
                    std::uint32_t const scalesPerRow
                        = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                    std::uint32_t const row = scaleGroup / scalesPerRow;
                    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                    __nv_fp8_e4m3 blockScale;
                    blockScale.__x = selectScaleInput(
                        task, role, halfGroupsPerRole)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
                    float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role]
                        * params.fp8ScaleOrigQuant[role];

                    std::uint64_t packedFp8[2];
#pragma unroll
                    for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
                    {
                        float2 values[4];
                        unpackE2m1ToFloat(packedWords[pair * 2U + laneInScale], values);
#pragma unroll
                        for (std::uint32_t i = 0; i < 4; ++i)
                        {
                            values[i].x *= dequantScale;
                            values[i].y *= dequantScale;
                        }
                        // FP8 REQUANT: reuse TRT-LLM's E4M3 packer. The two
                        // consecutive results are joined below before the GPU
                        // destination is touched.
                        packedFp8[laneInScale] = fp32_vec_to_e4m3(values);
                    }
                    // FUSED H2D FINISH + BATCHEDCOPY VECTOR WIDTH: one aligned
                    // 16-byte store publishes sixteen restored FP8 values to
                    // the final runtime Slot.
                    reinterpret_cast<uint4*>(selectRawOutput<__nv_fp8_e4m3>(task, role))[firstHalfGroup / 2U]
                        = collectTwoFp8Words(packedFp8[0], packedFp8[1]);
                }
            }
        }

        std::uint32_t const loadGrain = firstGrain + kThreadsPerBlock * iteration;
        bool const valid = loadGrain < endGrain;
        auto const* packedInput = reinterpret_cast<uint4 const*>(selectPackedInput(task, role, halfGroupsPerRole));
        auto const* source = valid ? packedInput + loadGrain : packedInput;
        // FUSED H2D START: mapped Host NVFP4 data -> CTA shared ring with the
        // exact 16-byte batchedCopy async-load mechanism.
        copyAsyncGlobalToShared(&packedStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

//! Host NVFP4 -> FP16/BF16 GPU Page with tiled transfer/dequant phases.
template <std::uint32_t N, typename T>
__global__ void onboardTo16BitTiledKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ pages,
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
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const halfGroupsPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerLane;
    std::uint32_t const totalRows
        = static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage);
    std::uint32_t const tileRows = compressedTransferRows(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedBytesPerRole(tileHalfGroups);
    extern __shared__ __align__(16) std::uint8_t compactStages[];

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstRow = blockIdx.x * tileRows; firstRow < totalRows; firstRow += gridDim.x * tileRows)
    {
        std::uint32_t const rows = std::min(tileRows, totalRows - firstRow);
        std::uint32_t const firstHalfGroup = firstRow * halfGroupsPerRow;
        std::uint32_t const halfGroups = rows * halfGroupsPerRow;
        std::uint32_t const packedBytes = packedBytesPerRole(halfGroups);
        std::uint32_t const scaleBytes = rows * scalesPerRow;

        // FUSED H2D PHASE: all packed data and block scales reach shared
        // through dense mapped-Host reads before dequantization starts.
        loadCompactRangeFromHost(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
            packedBytesPerRole(firstHalfGroup), packedBytes, firstRow * scalesPerRow, scaleBytes);

        auto const* packedStages = reinterpret_cast<uint4 const*>(compactStages);
        auto const* scaleStages = compactStages + packedStageCapacityBytes;
        std::uint32_t const packedGrains = packedBytes / sizeof(uint4);
        for (std::uint32_t localGrain = threadIdx.x; localGrain < packedGrains; localGrain += blockDim.x)
        {
            uint4 const packedGrain = packedStages[localGrain];
            std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
            std::uint32_t const globalGrain = firstHalfGroup / 4U + localGrain;
#pragma unroll
            for (std::uint32_t pair = 0; pair < 2; ++pair)
            {
                std::uint32_t const firstPairHalfGroup = globalGrain * 4U + pair * 2U;
                std::uint32_t const scaleGroup = firstPairHalfGroup >> 1U;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
                std::uint32_t const localScaleOffset
                    = scaleOffset(role, row, scaleInRow, scalesPerRow) - firstRow * scalesPerRow;

                __nv_fp8_e4m3 blockScale;
                blockScale.__x = scaleStages[localScaleOffset];
                float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];
#pragma unroll
                for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
                {
                    std::uint32_t const halfGroup = firstPairHalfGroup + laneInScale;
                    float2 values[4];
                    unpackE2m1ToFloat(packedWords[pair * 2U + laneInScale], values);
                    store16BitValues(
                        selectRawOutput<T>(task, role), halfGroup * kElementsPerLane, values, dequantScale);
                }
            }
        }

        // All consumers must finish before the next compact tile overwrites shared
        // memory.
        __syncthreads();
    }
#endif
}

//! Host NVFP4 -> FP8 GPU Page with tiled transfer/dequant/requant phases.
template <std::uint32_t N>
__global__ void onboardToFp8TiledKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ pages,
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
    std::uint32_t const halfGroupsPerRole = static_cast<std::uint32_t>(params.numKvHeads)
        * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
    auto const task = resolveTask(pages[blockIdx.z], layer, coldBase, coldPageBytes);
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const halfGroupsPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerLane;
    std::uint32_t const totalRows
        = static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage);
    std::uint32_t const tileRows = compressedTransferRows(params);
    std::uint32_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint32_t const packedStageCapacityBytes = packedBytesPerRole(tileHalfGroups);
    extern __shared__ __align__(16) std::uint8_t compactStages[];

    asm volatile("griddepcontrol.wait;\n" : : : "memory");
    for (std::uint32_t firstRow = blockIdx.x * tileRows; firstRow < totalRows; firstRow += gridDim.x * tileRows)
    {
        std::uint32_t const rows = std::min(tileRows, totalRows - firstRow);
        std::uint32_t const firstHalfGroup = firstRow * halfGroupsPerRow;
        std::uint32_t const halfGroups = rows * halfGroupsPerRow;
        std::uint32_t const packedBytes = packedBytesPerRole(halfGroups);
        std::uint32_t const scaleBytes = rows * scalesPerRow;
        loadCompactRangeFromHost(compactStages, task, role, halfGroupsPerRole, packedStageCapacityBytes,
            packedBytesPerRole(firstHalfGroup), packedBytes, firstRow * scalesPerRow, scaleBytes);

        auto const* packedStages = reinterpret_cast<uint4 const*>(compactStages);
        auto const* scaleStages = compactStages + packedStageCapacityBytes;
        std::uint32_t const packedGrains = packedBytes / sizeof(uint4);
        for (std::uint32_t localGrain = threadIdx.x; localGrain < packedGrains; localGrain += blockDim.x)
        {
            uint4 const packedGrain = packedStages[localGrain];
            std::uint32_t const packedWords[4] = {packedGrain.x, packedGrain.y, packedGrain.z, packedGrain.w};
            std::uint32_t const globalGrain = firstHalfGroup / 4U + localGrain;
#pragma unroll
            for (std::uint32_t pair = 0; pair < 2; ++pair)
            {
                std::uint32_t const firstPairHalfGroup = globalGrain * 4U + pair * 2U;
                std::uint32_t const scaleGroup = firstPairHalfGroup >> 1U;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
                std::uint32_t const localScaleOffset
                    = scaleOffset(role, row, scaleInRow, scalesPerRow) - firstRow * scalesPerRow;

                __nv_fp8_e4m3 blockScale;
                blockScale.__x = scaleStages[localScaleOffset];
                float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role]
                    * params.fp8ScaleOrigQuant[role];
                std::uint64_t packedFp8[2];
#pragma unroll
                for (std::uint32_t laneInScale = 0; laneInScale < 2; ++laneInScale)
                {
                    float2 values[4];
                    unpackE2m1ToFloat(packedWords[pair * 2U + laneInScale], values);
#pragma unroll
                    for (std::uint32_t i = 0; i < 4; ++i)
                    {
                        values[i].x *= dequantScale;
                        values[i].y *= dequantScale;
                    }
                    packedFp8[laneInScale] = fp32_vec_to_e4m3(values);
                }
                reinterpret_cast<uint4*>(selectRawOutput<__nv_fp8_e4m3>(task, role))[firstPairHalfGroup / 2U]
                    = collectTwoFp8Words(packedFp8[0], packedFp8[1]);
            }
        }
        __syncthreads();
    }
#endif
}

void validateParams(Nvfp4BoundaryKernelParams const& params, bool useFp8)
{
    TLLM_CHECK_WITH_INFO(params.numKvHeads > 0, "numKvHeads must be positive");
    TLLM_CHECK_WITH_INFO(params.tokensPerPage > 0, "tokensPerPage must be positive");
    TLLM_CHECK_WITH_INFO(params.tokensPerPage % 4 == 0,
        "tokensPerPage must be divisible by 4 for the native "
        "V-scale layout, got %d",
        params.tokensPerPage);
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
    std::uint64_t const halfGroups = rows * halfGroupsPerRow;
    TLLM_CHECK_WITH_INFO(halfGroups % 8 == 0,
        "Page geometry must produce complete 16-byte FP8 and "
        "NVFP4 transfer grains, got %llu half-groups",
        static_cast<unsigned long long>(halfGroups));

    // A token-4 V-scale interval is the smallest legal compact tile. For
    // normal model head dimensions it stays far below this accepted bound;
    // reject an extreme geometry synchronously if even that minimum tile would
    // exceed the shared-memory contract used by every automatic fallback.
    std::uint64_t const tileHalfGroups = compressedTransferHalfGroups(params);
    std::uint64_t const tileBytes = tileHalfGroups * sizeof(std::uint32_t) + tileHalfGroups / 2U;
    TLLM_CHECK_WITH_INFO(tileBytes <= kMaxWholePageCompactStagingBytes,
        "Minimum NVFP4 boundary tile requires %llu "
        "shared-memory bytes; maximum supported is %u",
        static_cast<unsigned long long>(tileBytes), kMaxWholePageCompactStagingBytes);

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
void validatePointer(Pointer pointer, std::uintptr_t alignment, char const* name)
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
    validatePointer(reinterpret_cast<void const*>(layer.rawKBase), alignof(uint4), "rawKBase");
    validatePointer(reinterpret_cast<void const*>(layer.rawVBase), alignof(uint4), "rawVBase");
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

std::uint32_t halfGroupsPerRole(Nvfp4BoundaryKernelParams const& params)
{
    return static_cast<std::uint32_t>(params.numKvHeads) * static_cast<std::uint32_t>(params.tokensPerPage)
        * static_cast<std::uint32_t>(params.headDim / kElementsPerLane);
}

//! Launch one descriptor-capacity specialization through the raw-argument
//! runtime API. A full batch points the kernel-parameter copy directly at the
//! caller's contiguous vector, matching batchedCopy's no-intermediate-copy
//! fast path. Only an incomplete tail is copied into a stack array because the
//! kernel ABI still contains the complete compile-time descriptor capacity.
//!
//! `cudaLaunchKernelExC` keeps that raw-argument fast path while adding the PDL
//! attribute that current batchedCopy's plain `cuLaunchKernel` omits.
template <std::uint32_t N, typename Task, typename Kernel, typename ColdPointer>
void launchBoundaryBatch(Kernel kernel, Task const* tasks, std::uint32_t count, dim3 grid, dim3 block,
    std::uint32_t dynamicSmemBytes, std::array<Nvfp4BoundaryLayerPlan, kMaxLayersPerLaunch> const& layers,
    ColdPointer coldBase, std::size_t coldPageBytes, std::uint32_t numLayers, cudaStream_t stream)
{
    static_assert(std::is_trivially_copyable_v<std::array<Task, N>>);
    static_assert(sizeof(std::array<Task, N>) == sizeof(Task) * N);
    TLLM_CHECK(count > 0 && count <= N);

    // BATCHEDCOPY DIRECT REUSE: complete descriptor batches are passed from
    // the vector without copying. Only the final partial specialization needs
    // a padded by-value kernel argument; value initialization keeps its unused
    // object representation deterministic for sanitizers and launch tracing.
    std::array<Task, N> tail{};
    void const* taskArgument = tasks;
    if (count < N)
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

    // The model-like H=8/P=64/D=128 Page needs 36 KiB of compact shared
    // staging in addition to the 8 KiB input ring. Opt in explicitly so larger
    // valid Page geometries are not silently limited by CUDA's default shared-
    // memory carveout; an unsupported request fails synchronously here.
    if (dynamicSmemBytes != 0)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSmemBytes));
    }

    void* arguments[] = {const_cast<void*>(taskArgument), const_cast<Nvfp4BoundaryLayerPlan*>(layers.data()), &coldBase,
        &coldPageBytes, &numLayers};
    TLLM_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), arguments));
}

//! Submit disjoint Page descriptors with the same power-of-two specialization
//! scheme as KVCM V2 `launchBatchedCopy`. Production currently uses a 256-task
//! maximum. Every tail recursively selects the smallest 32-or-larger
// specialization that can
//! hold it. The actual count remains in grid.z, so padded descriptors are never
//! observed and a diagnostic maximum cannot silently drop an intermediate
//! specialization.
template <std::uint32_t N, typename Task, typename LaunchBatch>
void launchTaskTail(Task const* tasks, std::uint32_t count, LaunchBatch const& launchBatch)
{
    static_assert(N >= kMinTasksPerLaunch && (N & (N - 1U)) == 0U);
    TLLM_CHECK_DEBUG(count > 0 && count <= N);
    if constexpr (N == kMinTasksPerLaunch)
    {
        launchBatch(std::integral_constant<std::uint32_t, N>{}, tasks, count);
    }
    else if (count > N / 2U)
    {
        launchBatch(std::integral_constant<std::uint32_t, N>{}, tasks, count);
    }
    else
    {
        // Keep every power-of-two tail specialization down to 32.
        launchTaskTail<N / 2U>(tasks, count, launchBatch);
    }
}

template <typename Task, typename LaunchBatch>
void launchTaskBatches(std::vector<Task> const& tasks, LaunchBatch const& launchBatch)
{
    std::size_t offset = 0;
    while (tasks.size() - offset >= kMaxTasksPerLaunch)
    {
        launchBatch(
            std::integral_constant<std::uint32_t, kMaxTasksPerLaunch>{}, tasks.data() + offset, kMaxTasksPerLaunch);
        offset += kMaxTasksPerLaunch;
    }

    std::uint32_t const count = static_cast<std::uint32_t>(tasks.size() - offset);
    if (count == 0)
    {
        return;
    }
    launchTaskTail<kMaxTasksPerLaunch>(tasks.data() + offset, count, launchBatch);
}

bool useTiledStandardGeometry(Nvfp4BoundaryKernelParams const& params)
{
    // Do not extrapolate the standard-Page scheduling heuristic to other Page
    // geometries.
    return params.numKvHeads == kStandardNumKvHeads && params.tokensPerPage == kStandardTokensPerPage
        && params.headDim == kStandardHeadDim;
}

bool requiresBoundedTiledStaging(std::uint32_t halfGroups)
{
    // Whole-Page staging is a latency/performance choice, not a correctness
    // requirement. Keep it within the largest dynamic allocation exercised by
    // the supported H=8/P=64/D=128 geometry (32 KiB packed + 4 KiB scales).
    // Larger valid Pages use the existing bounded tile, whose compact staging
    // is selected from a roughly 2-KiB scale wave and does not grow with Page
    // size. This avoids an architecture-dependent launch failure without
    // extrapolating the standard-geometry policy to a new geometry.
    return compactStagingBytesPerRole(halfGroups) > kMaxWholePageCompactStagingBytes;
}

//! Conservative whole-Page product choices for SM100. Each row is still one
//! fused transform-plus-transfer launch:
//!
//!   FP16/BF16 -> NVFP4 + D2H: one bounded tiled CTA per Page/role
//!   FP8       -> NVFP4 + D2H: one compressed-output tiled CTA per Page/role
//!   H2D + NVFP4 -> FP16/BF16: one tiled CTA per Page/role
//!   H2D + NVFP4 -> FP8: one compressed-input tiled CTA per Page/role
//!
//! The older single-layer prototype selected extra CTA splits and the
//! double-buffered path from Page count alone. That evidence does not transfer
//! to this launch, where grid.y already expands all layers and K/V. Keep one
//! CTA per Page/role until same-shape whole-Page A/B data justifies a dispatch
//! based on total CTAs and total bytes. Non-standard geometries retain the
//! direct correctness fallback unless their Page size requires bounded tiles.

template <typename T>
void launchOffloadFrom16Bit(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](auto capacity, Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            // Whole-Page baseline: grid.y already exposes every layer and K/V,
            // so one mapped-Host CTA per Page/role is the evidence-neutral
            // analogue of batchedCopy's low-bandwidth path.
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            if (plan.allStandardGeometry || requiresBoundedTiledStaging(plan.maxHalfGroups))
            {
                launchBoundaryBatch<taskCapacity>(offloadFrom16BitTiledKernel<taskCapacity, T>, taskData, count, grid,
                    block, compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase,
                    plan.coldPageBytes, plan.numLayers, stream);
            }
            else
            {
                launchBoundaryBatch<taskCapacity>(offloadFrom16BitKernel<taskCapacity, T>, taskData, count, grid, block,
                    0, plan.layers, coldBase, plan.coldPageBytes, plan.numLayers, stream);
            }
        });
}

void launchOffloadFromFp8(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](auto capacity, Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            // FP8 uses compressed-output tiles for the standard Page geometry
            // without changing the quantization math or launch count.
            if (plan.allStandardGeometry || requiresBoundedTiledStaging(plan.maxHalfGroups))
            {
                launchBoundaryBatch<taskCapacity>(offloadFromFp8TiledKernel<taskCapacity>, taskData, count, grid, block,
                    compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                    plan.numLayers, stream);
            }
            else
            {
                launchBoundaryBatch<taskCapacity>(offloadFromFp8Kernel<taskCapacity>, taskData, count, grid, block,
                    compactStagingBytesPerRole(plan.maxHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                    plan.numLayers, stream);
            }
        });
}

template <typename T>
void launchOnboardTo16Bit(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t const* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](auto capacity, Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            if (plan.allStandardGeometry)
            {
                launchBoundaryBatch<taskCapacity>(onboardTo16BitTiledKernel<taskCapacity, T>, taskData, count, grid,
                    block, compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase,
                    plan.coldPageBytes, plan.numLayers, stream);
            }
            else
            {
                launchBoundaryBatch<taskCapacity>(onboardTo16BitKernel<taskCapacity, T>, taskData, count, grid, block,
                    0, plan.layers, coldBase, plan.coldPageBytes, plan.numLayers, stream);
            }
        });
}

void launchOnboardToFp8(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages, Nvfp4BoundaryPreparedPlan const& plan,
    std::uint8_t const* coldBase, cudaStream_t stream)
{
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(pages,
        [&](auto capacity, Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2U * plan.numLayers, count);
            if (plan.allStandardGeometry)
            {
                launchBoundaryBatch<taskCapacity>(onboardToFp8TiledKernel<taskCapacity>, taskData, count, grid, block,
                    compactStagingBytesPerRole(plan.maxTileHalfGroups), plan.layers, coldBase, plan.coldPageBytes,
                    plan.numLayers, stream);
            }
            else
            {
                launchBoundaryBatch<taskCapacity>(onboardToFp8Kernel<taskCapacity>, taskData, count, grid, block, 0,
                    plan.layers, coldBase, plan.coldPageBytes, plan.numLayers, stream);
            }
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
    plan.allStandardGeometry = true;
    std::copy(layers.begin(), layers.end(), plan.layers.begin());
    for (auto const& layer : layers)
    {
        validateLayerPlan(layer, coldPageBytes, useFp8);
        plan.maxHalfGroups = std::max(plan.maxHalfGroups, halfGroupsPerRole(layer.params));
        plan.maxTileHalfGroups = std::max(plan.maxTileHalfGroups, compressedTransferHalfGroups(layer.params));
        plan.allStandardGeometry = plan.allStandardGeometry && useTiledStandardGeometry(layer.params);
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
    validatePointer(coldBase, alignof(uint4), "coldBase");
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
    validatePointer(coldBase, alignof(uint4), "coldBase");
    switch (plan.runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        launchAndDrainOnFailure(stream,
            [&] { launchOnboardTo16Bit<half>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        launchAndDrainOnFailure(stream, [&]
            { launchOnboardTo16Bit<__nv_bfloat16>(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        launchAndDrainOnFailure(
            stream, [&] { launchOnboardToFp8(pages, plan, static_cast<std::uint8_t const*>(coldBase), stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
