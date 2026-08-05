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
// Both boundary directions touch CUDA-mapped Host memory, so the first async
// implementation also keeps batchedCopy's one-split low-bandwidth policy.
// Split-count tuning is a separate measured optimization, not part of the
// correctness port.
constexpr std::uint32_t kThreadsPerBlock = 128;
constexpr std::uint32_t kAsyncStages = 4;
constexpr std::uint32_t kHostMemorySplits = 1;
constexpr std::uint32_t kTasksPerLaunch = 32;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOffloadPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOnboardPageTask>);

//! Reuse boundary for the fused implementation:
//!
//! * Quantization math is reused directly from `quantization.cuh`.
//! * The task-array batching, split/iteration indexing, four-stage shared ring,
//!   and commit/wait order are adapted from KVCM V2's `batchedCopy<N>`.
//! * `cp_async_commit_group()` and `cp_async_wait_group<N>()` are reused from
//!   TensorRT-LLM's shared `cudaAsyncOps.cuh` primitives.
//! * The source load width is adapted to each numerical work unit: 16 bytes
//!   for FP16/BF16 offload, 8 bytes for FP8 offload, and 4 packed bytes for
//!   onboard. The transformed destination therefore cannot call the original
//!   identity-copy kernel, whose source and destination have equal layouts.
//!
//! No compact GPU staging is introduced: shared memory is CTA-local pipeline
//! storage, then the kernel writes the final destination directly.

//! Issue one global-to-shared asynchronous load.
//!
//! BATCHEDCOPY REUSE: the 16-byte branch is the same `cp.async.cg` instruction
//! used by `kvCacheManagerV2Utils.cu::batchedCopy<N>`. BOUNDARY ADAPTATION:
//! FP8 and packed E2M1 naturally use 8-byte and 4-byte work units, so those
//! branches reuse the existing generic `ldgsts<Bytes>` (`cp.async.ca`) helper
//! instead of changing the quant/dequant thread mapping merely to force 16 B.
//! Original async helpers:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cudaAsyncOps.cuh
template <std::uint32_t Bytes, typename T>
__device__ __forceinline__ void copyAsyncGlobalToShared(T* shared, T const* global, bool valid)
{
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async supports 4, 8, or 16-byte copies");
    static_assert(sizeof(T) == Bytes, "The async-copy width must match the staged value");
    if constexpr (Bytes == 16)
    {
        if (valid)
        {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :
                         : "l"(__cvta_generic_to_shared(shared)), "l"(global)
                         : "memory");
        }
    }
    else
    {
        ldgsts<Bytes>(reinterpret_cast<int*>(shared), reinterpret_cast<int const*>(global), valid);
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

//! K scales are linear. V scales use the token-4 order consumed by the native
//! TRTLLM-gen NVFP4 KV path. `row` is the flattened [head, token] index; a
//! normal tokens-per-Page value divisible by four therefore never mixes heads.
//! Original native KV writer (`quantizeAndWriteFP4KVCache`):
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h
__device__ std::uint32_t scaleOffset(
    std::uint32_t role, std::uint32_t row, std::uint32_t scaleInRow, std::uint32_t scalesPerRow)
{
    if (role == 0)
    {
        return row * scalesPerRow + scaleInRow;
    }
    return (row / 4) * (4 * scalesPerRow) + scaleInRow * 4 + row % 4;
}

//! Unpack one lane's eight E2M1 values into four float2 values.
//!
//! The word "eight" describes the number of scalar values; this helper does
//! not produce the FP8 E4M3 dtype. The SM100 E2M1->FP16x2 PTX sequence is
//! adapted from ARCQuant/fused-MoE, then FP16x2 is widened to float2 so the two
//! onboard kernels can independently target FP16/BF16 or FP8.
//! Original dequant helpers:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/arcquantFP4.cu
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
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
    PackedVec<T> packedOutput;
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        float2 const scaled = make_float2(values[i].x * scale, values[i].y * scale);
        if constexpr (std::is_same_v<T, half>)
        {
            packedOutput.elts[i] = __float22half2_rn(scaled);
        }
        else
        {
            packedOutput.elts[i] = __float22bfloat162_rn(scaled);
        }
    }
    reinterpret_cast<PackedVec<T>*>(output + elementOffset)[0] = packedOutput;
}

//! FP16/BF16 GPU Page -> Host NVFP4: fused quantization plus D2H.
//!
//! Exact fused call/data flow:
//!
//! `invokeNvfp4BoundaryOffloadCompress`
//!   -> `launchOffloadFrom16Bit`                         [BOUNDARY NEW]
//!   -> `offloadFrom16BitKernel`
//!      -> 4-stage `cp.async` GPU raw -> shared          [BATCHEDCOPY ADAPTATION]
//!      -> `cvt_warp_fp16_to_fp4`                        [NVFP4 DIRECT REUSE]
//!      -> native K/V scale offset + mapped-Host stores  [BOUNDARY ADAPTATION]
//!
//! The generic quantization hierarchy is `invokeFP4Quantization ->
//! quantize_with_block_size -> cvt_warp_fp16_to_fp4`. This Page-specialized
//! kernel calls only the proven innermost primitive. Its async ring and
//! load/commit/wait order come from KVCM V2 `batchedCopy<N>`; unlike the
//! identity copy, quantization changes one 16-byte raw input into a 4-byte
//! packed output plus one scale shared by two lanes, so the final mapped-Host
//! stores are boundary-specific and remain in this same kernel.
//! Original quant hierarchy and leaf primitive:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cu
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
//! Original D2H/H2D pipeline:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N, typename T>
__global__ void offloadFrom16BitKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BOUNDARY ADAPTATION: grid.z selects one disjoint Page and grid.y selects
    // K or V. grid.x remains a split dimension so the low-bandwidth policy can
    // later be tuned without changing the Page-task ABI.
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
            if (halfGroup < endHalfGroup)
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
                // FUSED D2H FINISH: this global store targets CUDA-mapped Host
                // memory, so there is no second copy kernel or GPU Page-sized
                // staging allocation after quantization.
                reinterpret_cast<std::uint32_t*>(selectPackedOutput(task, role))[halfGroup] = packed;
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
        copyAsyncGlobalToShared<sizeof(PackedVec<T>)>(&rawStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

//! FP8 E4M3 GPU Page -> Host NVFP4: fused source restoration, NVFP4
//! quantization, and D2H.
//!
//! Exact fused call/data flow:
//!
//! `invokeNvfp4BoundaryOffloadCompress`
//!   -> `launchOffloadFromFp8`                            [BOUNDARY NEW]
//!   -> `offloadFromFp8Kernel`
//!      -> 4-stage `cp.async` GPU FP8 -> shared           [BATCHEDCOPY ADAPTATION]
//!      -> FP8 restore in registers                       [BOUNDARY ADAPTATION]
//!      -> `cvt_warp_fp16_to_fp4`                         [NVFP4 DIRECT REUSE]
//!      -> native K/V scale offset + mapped-Host stores   [BOUNDARY ADAPTATION]
//!
//! The source-FP8 inverse scale and target-NVFP4 scale remain independent.
//! Therefore this kernel deliberately does not call the differently shaped
//! `cvt_warp_fp8_to_fp4` helper. BATCHEDCOPY ADAPTATION: the same four-stage
//! ring is used, but each lane loads its natural eight-FP8-value (8-byte)
//! work unit with `cp.async.ca`, not batchedCopy's 16-byte identity grain.
//! Original FP8 and FP16/BF16 NVFP4 primitives:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
//! Original D2H/H2D pipeline:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N>
__global__ void offloadFromFp8Kernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

    // BATCHEDCOPY ADAPTATION: identical split/iteration/ring sequencing, with
    // an 8-byte source grain selected by the FP8 numerical work unit.
    __shared__ __align__(8) std::uint64_t rawStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalIterations = (halfGroupsPerRole + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstHalfGroup = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x + threadIdx.x;
    std::uint32_t const endHalfGroup
        = std::min(firstHalfGroup + kThreadsPerBlock * maxIterationsPerCta, halfGroupsPerRole);

    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const halfGroup = firstHalfGroup + kThreadsPerBlock * transformIteration;
            cp_async_wait_group<kAsyncStages - 1>();
            if (halfGroup < endHalfGroup)
            {
                std::uint32_t const laneInScale = halfGroup & 1U;
                std::uint32_t const scaleGroup = halfGroup >> 1U;
                std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                // BOUNDARY ADAPTATION: restore eight source FP8 values with
                // their independent inverse scale. The staged uint64_t avoids
                // imposing the generic FP8->FP4 helper's different scale ABI.
                std::uint64_t const fp8Bytes = rawStages[stage][threadIdx.x];
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

                // NVFP4 DIRECT REUSE: production two-lane block quantization.
                // BOUNDARY ADAPTATION: native scale address and mapped-Host
                // packed destination remain separate logical Pool roles.
                std::uint8_t* scale = laneInScale == 0
                    ? selectScaleOutput(task, role) + scaleOffset(role, row, scaleInRow, scalesPerRow)
                    : nullptr;
                std::uint32_t const packed = cvt_warp_fp16_to_fp4<half, kElementsPerBlockScale, false>(
                    restored, params.nvfp4ScaleOrigQuant[role], scale);
                // FUSED D2H FINISH: write the packed result directly to the
                // CUDA-mapped Host NVFP4 data Pool. The scale primitive above
                // writes the separate mapped Host scale Pool at the same
                // boundary; neither output uses a compact GPU staging Page.
                reinterpret_cast<std::uint32_t*>(selectPackedOutput(task, role))[halfGroup] = packed;
            }
        }

        std::uint32_t const loadHalfGroup = firstHalfGroup + kThreadsPerBlock * iteration;
        bool const valid = loadHalfGroup < endHalfGroup;
        auto const* rawInput = reinterpret_cast<std::uint64_t const*>(selectRawInput<__nv_fp8_e4m3>(task, role));
        auto const* source = valid ? rawInput + loadHalfGroup : rawInput;
        // FUSED D2H SOURCE PIPELINE: asynchronously stage eight runtime FP8
        // values from GPU global memory. Their mapped-Host D2H store occurs
        // only after source-scale restoration and NVFP4 quantization above.
        copyAsyncGlobalToShared<sizeof(std::uint64_t)>(&rawStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

//! Host NVFP4 -> FP16/BF16 GPU Page: fused H2D plus dequantization.
//!
//! Exact fused call/data flow:
//!
//! `invokeNvfp4BoundaryOnboardDecompress`
//!   -> `launchOnboardTo16Bit`                           [BOUNDARY NEW]
//!   -> `onboardTo16BitKernel`
//!      -> 4-stage `cp.async` Host packed -> shared      [BATCHEDCOPY ADAPTATION]
//!      -> native K/V scale load                         [BOUNDARY ADAPTATION]
//!      -> `unpackE2m1ToFloat` conversion                 [ARCQUANT/MOE ADAPTATION]
//!      -> scale + direct FP16/BF16 GPU store            [BOUNDARY ADAPTATION]
//!
//! `unpackE2m1ToFloat` adapts the `cvt.rn.f16x2.e2m1x2` sequence from
//! ARCQuant/fused-MoE; it does not call a MoE kernel. BATCHEDCOPY ADAPTATION:
//! four packed bytes are each lane's natural input, so `cp.async.ca` pipelines
//! the mapped-Host packed payload. The much smaller one-byte scale payload
//! keeps its native scalar lookup because K and V use different ordering and
//! the public task contract guarantees only byte alignment for scales.
//! Original dequant PTX:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/arcquantFP4.cu
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
//! Original H2D/D2H pipeline:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N, typename T>
__global__ void onboardTo16BitKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

    // BATCHEDCOPY ADAPTATION: identical split/iteration/ring sequencing. Each
    // lane pipelines one 4-byte E2M1 word from CUDA-mapped Host memory.
    __shared__ __align__(4) std::uint32_t packedStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalIterations = (halfGroupsPerRole + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstHalfGroup = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x + threadIdx.x;
    std::uint32_t const endHalfGroup
        = std::min(firstHalfGroup + kThreadsPerBlock * maxIterationsPerCta, halfGroupsPerRole);

    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const halfGroup = firstHalfGroup + kThreadsPerBlock * transformIteration;
            cp_async_wait_group<kAsyncStages - 1>();
            if (halfGroup < endHalfGroup)
            {
                std::uint32_t const scaleGroup = halfGroup >> 1U;
                std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
                std::uint32_t const elementOffset = halfGroup * kElementsPerLane;

                // BATCHEDCOPY REUSE: packed data is ready in the shared ring.
                // BOUNDARY ADAPTATION: scaleOffset preserves linear K and
                // token-4 V layout; the scalar mapped-Host scale read is 1/9
                // of the compact payload and does not widen the scale ABI.
                std::uint32_t const packed = packedStages[stage][threadIdx.x];
                __nv_fp8_e4m3 blockScale;
                blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
                float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];

                // ARCQUANT/MOE ADAPTATION: convert E2M1 pairs with the proven
                // SM100 PTX sequence. BOUNDARY ADAPTATION: apply the native KV
                // scale and write the caller-owned raw GPU Slot directly.
                float2 values[4];
                // NVFP4 DEQUANT: E2M1 bytes already moved Host -> shared by
                // copyAsyncGlobalToShared below are expanded to FP32 registers.
                unpackE2m1ToFloat(packed, values);
                // FUSED H2D FINISH: apply the block/global scale and store the
                // restored FP16/BF16 values into the final GPU raw Slot. This
                // is not another copy: the H2D source movement was the
                // cp.async mapped-Host load; this store completes that same
                // one-kernel H2D+dequant path.
                store16BitValues(selectRawOutput<T>(task, role), elementOffset, values, dequantScale);
            }
        }

        std::uint32_t const loadHalfGroup = firstHalfGroup + kThreadsPerBlock * iteration;
        bool const valid = loadHalfGroup < endHalfGroup;
        auto const* packedInput = reinterpret_cast<std::uint32_t const*>(selectPackedInput(task, role));
        auto const* source = valid ? packedInput + loadHalfGroup : packedInput;
        // FUSED H2D START: adapt batchedCopy's async global -> shared load to
        // read one compact E2M1 word directly from CUDA-mapped Host memory.
        copyAsyncGlobalToShared<sizeof(std::uint32_t)>(&packedStages[stage][threadIdx.x], source, valid);
        cp_async_commit_group();
    }
#endif
}

//! Host NVFP4 -> FP8 E4M3 GPU Page: fused H2D, NVFP4 dequantization, and FP8
//! requantization.
//!
//! Exact fused call/data flow:
//!
//! `invokeNvfp4BoundaryOnboardDecompress`
//!   -> `launchOnboardToFp8`                            [BOUNDARY NEW]
//!   -> `onboardToFp8Kernel`
//!      -> 4-stage `cp.async` Host packed -> shared     [BATCHEDCOPY ADAPTATION]
//!      -> native K/V scale load                        [BOUNDARY ADAPTATION]
//!      -> `unpackE2m1ToFloat` conversion                [ARCQUANT/MOE ADAPTATION]
//!      -> scale + `fp32_vec_to_e4m3` + GPU store       [NVFP4/BOUNDARY ADAPTATION]
//!
//! The mapped-Host source-load pipeline is identical to the FP16/BF16 onboard
//! path. The numerical tail additionally composes the NVFP4 inverse scale with
//! the destination FP8 quantization scale, then directly reuses
//! `fp32_vec_to_e4m3` from `quantization.cuh`.
//! Original dequant PTX and FP8 output primitive:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
//! Original H2D/D2H pipeline:
//! https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cu
template <std::uint32_t N>
__global__ void onboardToFp8Kernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // BOUNDARY ADAPTATION: select one disjoint Page and one K/V role while
    // retaining batchedCopy's split dimension.
    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];

    // BATCHEDCOPY ADAPTATION: four-stage ring over 4-byte packed E2M1 words.
    __shared__ __align__(4) std::uint32_t packedStages[kAsyncStages][kThreadsPerBlock];
    std::uint32_t const totalIterations = (halfGroupsPerRole + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    std::uint32_t const maxIterationsPerCta = (totalIterations + gridDim.x - 1U) / gridDim.x;
    std::uint32_t const firstHalfGroup = kThreadsPerBlock * maxIterationsPerCta * blockIdx.x + threadIdx.x;
    std::uint32_t const endHalfGroup
        = std::min(firstHalfGroup + kThreadsPerBlock * maxIterationsPerCta, halfGroupsPerRole);

    for (std::uint32_t iteration = 0; iteration < maxIterationsPerCta + kAsyncStages; ++iteration)
    {
        std::uint32_t const stage = iteration % kAsyncStages;
        if (iteration >= kAsyncStages)
        {
            std::uint32_t const transformIteration = iteration - kAsyncStages;
            std::uint32_t const halfGroup = firstHalfGroup + kThreadsPerBlock * transformIteration;
            cp_async_wait_group<kAsyncStages - 1>();
            if (halfGroup < endHalfGroup)
            {
                std::uint32_t const scaleGroup = halfGroup >> 1U;
                std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
                std::uint32_t const row = scaleGroup / scalesPerRow;
                std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

                // BATCHEDCOPY REUSE: consume packed bytes from the ready ring.
                // BOUNDARY ADAPTATION: look up the separate native scale Pool
                // and compose source-NVFP4 with destination-FP8 scales.
                std::uint32_t const packed = packedStages[stage][threadIdx.x];
                __nv_fp8_e4m3 blockScale;
                blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
                float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role]
                    * params.fp8ScaleOrigQuant[role];

                // ARCQUANT/MOE ADAPTATION + NVFP4 DIRECT REUSE: expand E2M1,
                // apply the composed scale, then reuse the production E4M3
                // packer. The final 8-byte store targets the raw GPU Slot.
                float2 values[4];
                // NVFP4 DEQUANT: expand the Host E2M1 word, which the async
                // stage below already moved into shared memory.
                unpackE2m1ToFloat(packed, values);
#pragma unroll
                for (std::uint32_t i = 0; i < 4; ++i)
                {
                    values[i].x *= dequantScale;
                    values[i].y *= dequantScale;
                }
                // FP8 REQUANT + FUSED H2D FINISH: reuse TRT-LLM's E4M3 packer
                // and store directly into the final GPU FP8 Slot.
                reinterpret_cast<std::uint64_t*>(selectRawOutput<__nv_fp8_e4m3>(task, role))[halfGroup]
                    = fp32_vec_to_e4m3(values);
            }
        }

        std::uint32_t const loadHalfGroup = firstHalfGroup + kThreadsPerBlock * iteration;
        bool const valid = loadHalfGroup < endHalfGroup;
        auto const* packedInput = reinterpret_cast<std::uint32_t const*>(selectPackedInput(task, role));
        auto const* source = valid ? packedInput + loadHalfGroup : packedInput;
        // FUSED H2D START: mapped Host NVFP4 data -> CTA shared ring via the
        // adapted batchedCopy async-load mechanism.
        copyAsyncGlobalToShared<sizeof(std::uint32_t)>(&packedStages[stage][threadIdx.x], source, valid);
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
        "tokensPerPage must be divisible by 4 for the native V-scale layout, got %d", params.tokensPerPage);
    TLLM_CHECK_WITH_INFO(params.headDim > 0 && params.headDim % kElementsPerBlockScale == 0,
        "headDim must be positive and divisible by 16, got %d", params.headDim);

    std::uint64_t const rows
        = static_cast<std::uint64_t>(params.numKvHeads) * static_cast<std::uint64_t>(params.tokensPerPage);
    std::uint64_t const halfGroups = rows * static_cast<std::uint64_t>(params.headDim / kElementsPerLane);
    TLLM_CHECK_WITH_INFO(
        halfGroups <= std::numeric_limits<std::uint32_t>::max(), "Page geometry exceeds the 32-bit kernel index range");

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
        validatePointer(task.packedK, alignof(std::uint32_t), "packedK");
        validatePointer(task.packedV, alignof(std::uint32_t), "packedV");
        validatePointer(task.blockScaleK, alignof(std::uint8_t), "blockScaleK");
        validatePointer(task.blockScaleV, alignof(std::uint8_t), "blockScaleV");
    }
}

void validateTasks(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks, std::uintptr_t rawAlignment)
{
    for (auto const& task : tasks)
    {
        validatePointer(task.packedK, alignof(std::uint32_t), "packedK");
        validatePointer(task.packedV, alignof(std::uint32_t), "packedV");
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

template <typename T>
void launchOffloadFrom16Bit(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    for (std::size_t offset = 0; offset < tasks.size(); offset += kTasksPerLaunch)
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(kTasksPerLaunch, tasks.size() - offset));
        std::array<Nvfp4BoundaryOffloadPageTask, kTasksPerLaunch> batch{};
        std::copy_n(tasks.data() + offset, count, batch.begin());
        // BATCHEDCOPY REUSE: Host transfers use one low-bandwidth split per
        // Page/role; the kernel itself pipelines all source tiles in that CTA.
        dim3 const grid(kHostMemorySplits, 2, count);
        offloadFrom16BitKernel<kTasksPerLaunch, T><<<grid, block, 0, stream>>>(batch, params, halfGroups);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void launchOffloadFromFp8(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    for (std::size_t offset = 0; offset < tasks.size(); offset += kTasksPerLaunch)
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(kTasksPerLaunch, tasks.size() - offset));
        std::array<Nvfp4BoundaryOffloadPageTask, kTasksPerLaunch> batch{};
        std::copy_n(tasks.data() + offset, count, batch.begin());
        dim3 const grid(kHostMemorySplits, 2, count);
        offloadFromFp8Kernel<kTasksPerLaunch><<<grid, block, 0, stream>>>(batch, params, halfGroups);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
void launchOnboardTo16Bit(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    for (std::size_t offset = 0; offset < tasks.size(); offset += kTasksPerLaunch)
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(kTasksPerLaunch, tasks.size() - offset));
        std::array<Nvfp4BoundaryOnboardPageTask, kTasksPerLaunch> batch{};
        std::copy_n(tasks.data() + offset, count, batch.begin());
        dim3 const grid(kHostMemorySplits, 2, count);
        onboardTo16BitKernel<kTasksPerLaunch, T><<<grid, block, 0, stream>>>(batch, params, halfGroups);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void launchOnboardToFp8(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks, Nvfp4BoundaryKernelParams const& params,
    cudaStream_t stream)
{
    std::uint32_t const halfGroups = halfGroupsPerRole(params);
    dim3 const block(kThreadsPerBlock);
    for (std::size_t offset = 0; offset < tasks.size(); offset += kTasksPerLaunch)
    {
        std::uint32_t const count
            = static_cast<std::uint32_t>(std::min<std::size_t>(kTasksPerLaunch, tasks.size() - offset));
        std::array<Nvfp4BoundaryOnboardPageTask, kTasksPerLaunch> batch{};
        std::copy_n(tasks.data() + offset, count, batch.begin());
        dim3 const grid(kHostMemorySplits, 2, count);
        onboardToFp8Kernel<kTasksPerLaunch><<<grid, block, 0, stream>>>(batch, params, halfGroups);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
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
        validateTasks(tasks, 8);
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
        validateTasks(tasks, 8);
        launchAndDrainOnFailure(stream, [&] { launchOnboardToFp8(tasks, params, stream); });
        break;
    default: TLLM_THROW("Unsupported NVFP4 boundary runtime type");
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
