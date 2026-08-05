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
#include <limits>
#include <type_traits>

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
constexpr std::uint32_t kHostMemorySplits = 1;
constexpr std::uint32_t kMinTasksPerLaunch = 32;
constexpr std::uint32_t kMaxTasksPerLaunch = 256;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;
constexpr std::size_t kModernKernelParameterLimit = 32764;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOffloadPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOnboardPageTask>);
static_assert(sizeof(std::array<Nvfp4BoundaryOffloadPageTask, kMaxTasksPerLaunch>) + sizeof(Nvfp4BoundaryKernelParams)
        + sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);
static_assert(sizeof(std::array<Nvfp4BoundaryOnboardPageTask, kMaxTasksPerLaunch>) + sizeof(Nvfp4BoundaryKernelParams)
        + sizeof(std::uint32_t)
    <= kModernKernelParameterLimit);

//! Reuse boundary for the fused implementation:
//!
//! * Quantization math is reused directly from `quantization.cuh`.
//! * The task-array batching, split/iteration indexing, four-stage shared ring,
//!   and commit/wait order are adapted from KVCM V2's `batchedCopy<N>`.
//! * `cp_async_commit_group()` and `cp_async_wait_group<N>()` are reused from
//!   TensorRT-LLM's shared `cudaAsyncOps.cuh` primitives.
//! * Every raw/packed main-payload source load uses batchedCopy's 16-byte
//!   `cp.async.cg` grain. FP8 offload redistributes a grain into two
//!   quantization rounds; onboard consumes four packed words per grain.
//! * Block-scale bytes retain the native NVFP4 scalar producer/consumer
//!   contract; they are transform metadata, not batchedCopy identity payload.
//! * The transformed destination cannot call the original identity-copy
//!   kernel because source and destination layouts differ.
//!
//! No compact GPU staging is introduced: shared memory is CTA-local pipeline
//! storage, then the kernel writes the final destination directly.

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

template <typename T>
__device__ T const* selectRawInput(Nvfp4BoundaryOffloadPageTask const& task, std::uint32_t role)
{
    return reinterpret_cast<T const*>(role == 0 ? task.rawK : task.rawV);
}

__device__ std::uint8_t* selectPackedOutput(Nvfp4BoundaryOffloadPageTask const& task, std::uint32_t role)
{
    return role == 0 ? task.packedK : task.packedV;
}

__device__ std::uint8_t* selectScaleOutput(Nvfp4BoundaryOffloadPageTask const& task, std::uint32_t role)
{
    return role == 0 ? task.blockScaleK : task.blockScaleV;
}

__device__ std::uint8_t const* selectPackedInput(Nvfp4BoundaryOnboardPageTask const& task, std::uint32_t role)
{
    return role == 0 ? task.packedK : task.packedV;
}

__device__ std::uint8_t const* selectScaleInput(Nvfp4BoundaryOnboardPageTask const& task, std::uint32_t role)
{
    return role == 0 ? task.blockScaleK : task.blockScaleV;
}

template <typename T>
__device__ T* selectRawOutput(Nvfp4BoundaryOnboardPageTask const& task, std::uint32_t role)
{
    return reinterpret_cast<T*>(role == 0 ? task.rawK : task.rawV);
}

//! Collect four adjacent lanes' packed NVFP4 words into one 16-byte store.
//!
//! BATCHEDCOPY ADAPTATION: identity migration naturally stores one uint4 per
//! lane. Boundary quantization produces one uint32 per lane, so four converged
//! lanes first assemble the same 16-byte transaction width in registers. The
//! caller lets only lane 0 of each four-lane group publish the uint4.
__device__ __forceinline__ uint4 collectFourPackedWords(std::uint32_t packed)
{
    std::uint32_t const lane = threadIdx.x & 31U;
    std::uint32_t const firstLane = lane & ~3U;
    std::uint32_t const groupMask = 0xFU << firstLane;
    return make_uint4(__shfl_sync(groupMask, packed, firstLane), __shfl_sync(groupMask, packed, firstLane + 1U),
        __shfl_sync(groupMask, packed, firstLane + 2U), __shfl_sync(groupMask, packed, firstLane + 3U));
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

// FP16/BF16 GPU Page -> Host NVFP4: fused quantization plus D2H.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOffloadCompress                  [BOUNDARY API]
//     -> launchOffloadFrom16Bit                         [BOUNDARY BATCHING]
//     -> offloadFrom16BitKernel                         [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async GPU raw -> shared          [BATCHEDCOPY ADAPTATION]
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
__global__ void offloadFrom16BitKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
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
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

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
                // E2M1 packing. BOUNDARY ADAPTATION: its scale pointer and the
                // packed output point directly into native-layout Host Pools.
                PackedVec<T> input = rawStages[stage][threadIdx.x];
                std::uint8_t* scale = laneInScale == 0
                    ? selectScaleOutput(task, role) + scaleOffset(role, row, scaleInRow, scalesPerRow)
                    : nullptr;
                std::uint32_t const packed = cvt_warp_fp16_to_fp4<T, kElementsPerBlockScale, false>(
                    input, params.nvfp4ScaleOrigQuant[role], scale);
                uint4 const packedGrain = collectFourPackedWords(packed);
                if ((threadIdx.x & 3U) == 0)
                {
                    // FUSED D2H FINISH + BATCHEDCOPY VECTOR WIDTH: four adjacent
                    // quantized words become one aligned uint4 transaction directly
                    // into the mapped-Host packed Pool. The separate native scale
                    // Pool retains the primitive's scalar E4M3 writes.
                    reinterpret_cast<uint4*>(selectPackedOutput(task, role))[halfGroup / 4U] = packedGrain;
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
//        -> 4-stage cp.async GPU FP8 -> shared          [BATCHEDCOPY ADAPTATION]
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
__global__ void offloadFromFp8Kernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: enable programmatic dependent-launch overlap
    // while preserving stream-ordered visibility before any Page bytes move.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

    // BATCHEDCOPY DIRECT REUSE: identical 16-byte grain and four-stage ring.
    // One grain contains two logical eight-value half-groups.
    __shared__ __align__(16) uint4 rawStages[kAsyncStages][kThreadsPerBlock];
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
                    std::uint8_t* scale = laneInScale == 0
                        ? selectScaleOutput(task, role) + scaleOffset(role, row, scaleInRow, scalesPerRow)
                        : nullptr;
                    std::uint32_t const packed = cvt_warp_fp16_to_fp4<half, kElementsPerBlockScale, false>(
                        restored, params.nvfp4ScaleOrigQuant[role], scale);
                    uint4 const packedGrain = collectFourPackedWords(packed);
                    if ((lane & 3U) == 0)
                    {
                        // FUSED D2H FINISH: preserve batchedCopy's 16-byte global
                        // transaction width after FP8 restoration and quantization.
                        reinterpret_cast<uint4*>(selectPackedOutput(task, role))[halfGroup / 4U] = packedGrain;
                    }
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
#endif
}

// Host NVFP4 -> FP16/BF16 GPU Page: fused H2D plus dequantization.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOnboardDecompress                [BOUNDARY API]
//     -> launchOnboardTo16Bit                           [BOUNDARY BATCHING]
//     -> onboardTo16BitKernel                           [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async Host packed -> shared      [BATCHEDCOPY ADAPTATION]
//        -> native K/V scale load                       [BOUNDARY ADAPTATION]
//        -> unpackE2m1ToFloat                           [ARCQUANT/MOE ADAPTATION]
//        -> scaled 16-byte FP16/BF16 GPU store          [BOUNDARY ADAPTATION]
//
// unpackE2m1ToFloat adapts the cvt.rn.f16x2.e2m1x2 sequence from
// ARCQuant/fused-MoE; it does not call a MoE kernel. Every lane pipelines one
// 16-byte mapped-Host grain and expands its four packed words. Each adjacent
// word pair shares one scalar scale load; the scale Pool stays separate
// because K and V use different native ordering.
// Dequant PTX sources:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/arcquantFP4.cu
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
// H2D/D2H pipeline source:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N, typename T>
__global__ void onboardTo16BitKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: let the next migration grid launch early, but
    // fence this grid's mapped-Host reads behind its stream predecessor.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

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
                    blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
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
        auto const* packedInput = reinterpret_cast<uint4 const*>(selectPackedInput(task, role));
        auto const* source = valid ? packedInput + loadGrain : packedInput;
        // FUSED H2D START: move four compact E2M1 words directly from mapped
        // Host memory with batchedCopy's exact 16-byte async grain.
        copyAsyncGlobalToShared(&packedStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

// Host NVFP4 -> FP8 E4M3 GPU Page: fused H2D, dequantization, and requantization.
//
// Exact call/data flow:
//   invokeNvfp4BoundaryOnboardDecompress                [BOUNDARY API]
//     -> launchOnboardToFp8                             [BOUNDARY BATCHING]
//     -> onboardToFp8Kernel                             [BOUNDARY FUSED KERNEL]
//        -> 4-stage cp.async Host packed -> shared      [BATCHEDCOPY ADAPTATION]
//        -> native K/V scale load                       [BOUNDARY ADAPTATION]
//        -> unpackE2m1ToFloat                           [ARCQUANT/MOE ADAPTATION]
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
__global__ void onboardToFp8Kernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BATCHEDCOPY DIRECT REUSE: use the same programmatic grid dependency
    // protocol for the FP8 destination specialization.
    asm volatile("griddepcontrol.launch_dependents;\n");

    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

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
                    blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
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
        auto const* packedInput = reinterpret_cast<uint4 const*>(selectPackedInput(task, role));
        auto const* source = valid ? packedInput + loadGrain : packedInput;
        // FUSED H2D START: mapped Host NVFP4 data -> CTA shared ring with the
        // exact 16-byte batchedCopy async-load mechanism.
        copyAsyncGlobalToShared(&packedStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

void validateParams(Nvfp4BoundaryKernelParams const& params, bool useFp8)
{
    TLLM_CHECK_WITH_INFO(common::isSM100Family(), "NVFP4 boundary kernels require an SM100-family GPU");
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
    constexpr std::uint64_t maxHalfGroups
        = std::numeric_limits<std::uint32_t>::max() / static_cast<std::uint64_t>(kElementsPerLane);
    std::uint64_t const halfGroupsPerRow = static_cast<std::uint64_t>(params.headDim / kElementsPerLane);
    TLLM_CHECK_WITH_INFO(rows <= maxHalfGroups / halfGroupsPerRow,
        "Page geometry exceeds the 32-bit element-offset range: heads=%d, tokens=%d, headDim=%d", params.numKvHeads,
        params.tokensPerPage, params.headDim);
    std::uint64_t const halfGroups = rows * halfGroupsPerRow;
    TLLM_CHECK_WITH_INFO(halfGroups % 8 == 0,
        "Page geometry must produce complete 16-byte FP8 and "
        "NVFP4 transfer grains, got %llu half-groups",
        static_cast<unsigned long long>(halfGroups));

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

void validateTasks(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks, std::uintptr_t rawAlignment)
{
    for (auto const& task : tasks)
    {
        validatePointer(task.rawK, rawAlignment, "rawK");
        validatePointer(task.rawV, rawAlignment, "rawV");
        validatePointer(task.packedK, alignof(uint4), "packedK");
        validatePointer(task.packedV, alignof(uint4), "packedV");
        validatePointer(task.blockScaleK, alignof(std::uint8_t), "blockScaleK");
        validatePointer(task.blockScaleV, alignof(std::uint8_t), "blockScaleV");
    }
}

void validateTasks(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks, std::uintptr_t rawAlignment)
{
    for (auto const& task : tasks)
    {
        validatePointer(task.packedK, alignof(uint4), "packedK");
        validatePointer(task.packedV, alignof(uint4), "packedV");
        validatePointer(task.blockScaleK, alignof(std::uint8_t), "blockScaleK");
        validatePointer(task.blockScaleV, alignof(std::uint8_t), "blockScaleV");
        validatePointer(task.rawK, rawAlignment, "rawK");
        validatePointer(task.rawV, rawAlignment, "rawV");
    }
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
template <std::uint32_t N, typename Task, typename Kernel>
void launchBoundaryBatch(Kernel kernel, Task const* tasks, std::uint32_t count, dim3 grid, dim3 block,
    Nvfp4BoundaryKernelParams const& params, std::uint32_t halfGroups, cudaStream_t stream)
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
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;

    void* arguments[] = {const_cast<void*>(taskArgument), const_cast<Nvfp4BoundaryKernelParams*>(&params), &halfGroups};
    TLLM_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), arguments));
}

//! Submit disjoint Page descriptors with the same capacity specializations as
//! KVCM V2 `launchBatchedCopy`: full batches use 256 descriptors and the tail
//! selects 256/128/64/32. The kernel receives the actual count through grid.z,
//! so zero-filled descriptor slots are never observed. Compared with a fixed
//! 32-Page chunk this preserves the small-batch parameter footprint while
//! avoiding up to seven redundant launches for a 256-Page migration.
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
    auto const* batch = tasks.data() + offset;
    if (count > kMaxTasksPerLaunch / 2)
    {
        launchBatch(std::integral_constant<std::uint32_t, kMaxTasksPerLaunch>{}, batch, count);
    }
    else if (count > kMaxTasksPerLaunch / 4)
    {
        launchBatch(std::integral_constant<std::uint32_t, kMaxTasksPerLaunch / 2>{}, batch, count);
    }
    else if (count > kMaxTasksPerLaunch / 8)
    {
        launchBatch(std::integral_constant<std::uint32_t, kMaxTasksPerLaunch / 4>{}, batch, count);
    }
    else
    {
        launchBatch(std::integral_constant<std::uint32_t, kMinTasksPerLaunch>{}, batch, count);
    }
}

template <typename T>
void launchOffloadFrom16Bit(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(tasks,
        [&](auto capacity, Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            // BATCHEDCOPY REUSE: Host transfers use one low-bandwidth split per
            // Page/role; the kernel itself pipelines all source tiles in that CTA.
            dim3 const grid(kHostMemorySplits, 2, count);
            launchBoundaryBatch<taskCapacity>(
                offloadFrom16BitKernel<taskCapacity, T>, taskData, count, grid, block, params, halfGroups, stream);
        });
}

void launchOffloadFromFp8(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(tasks,
        [&](auto capacity, Nvfp4BoundaryOffloadPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2, count);
            launchBoundaryBatch<taskCapacity>(
                offloadFromFp8Kernel<taskCapacity>, taskData, count, grid, block, params, halfGroups, stream);
        });
}

template <typename T>
void launchOnboardTo16Bit(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(tasks,
        [&](auto capacity, Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2, count);
            launchBoundaryBatch<taskCapacity>(
                onboardTo16BitKernel<taskCapacity, T>, taskData, count, grid, block, params, halfGroups, stream);
        });
}

void launchOnboardToFp8(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks, Nvfp4BoundaryKernelParams const& params,
    cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    launchTaskBatches(tasks,
        [&](auto capacity, Nvfp4BoundaryOnboardPageTask const* taskData, std::uint32_t count)
        {
            constexpr std::uint32_t taskCapacity = decltype(capacity)::value;
            dim3 const grid(kHostMemorySplits, 2, count);
            launchBoundaryBatch<taskCapacity>(
                onboardToFp8Kernel<taskCapacity>, taskData, count, grid, block, params, halfGroups, stream);
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
        static_cast<void>(cudaStreamSynchronize(stream));
        throw;
    }
}

} // namespace

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType, cudaStream_t stream)
{
    if (tasks.empty())
    {
        return;
    }
    switch (runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        validateParams(params, false);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOffloadFrom16Bit<half>(tasks, params, stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        validateParams(params, false);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOffloadFrom16Bit<__nv_bfloat16>(tasks, params, stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        validateParams(params, true);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOffloadFromFp8(tasks, params, stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType, cudaStream_t stream)
{
    if (tasks.empty())
    {
        return;
    }
    switch (runtimeType)
    {
    case Nvfp4BoundaryRuntimeType::kFloat16:
        validateParams(params, false);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOnboardTo16Bit<half>(tasks, params, stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kBfloat16:
        validateParams(params, false);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOnboardTo16Bit<__nv_bfloat16>(tasks, params, stream); });
        break;
    case Nvfp4BoundaryRuntimeType::kFp8E4m3:
        validateParams(params, true);
        validateTasks(tasks, 16);
        launchAndDrainOnFailure(stream, [&] { launchOnboardToFp8(tasks, params, stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
