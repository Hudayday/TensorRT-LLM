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

constexpr std::uint32_t kThreadsPerBlock = 256;
constexpr std::uint32_t kTasksPerLaunch = 32;
constexpr std::uint32_t kElementsPerLane = 8;
constexpr std::uint32_t kElementsPerBlockScale = 16;

static_assert(kThreadsPerBlock % 2 == 0, "An NVFP4 scale group is shared by two lanes");
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOffloadPageTask>);
static_assert(std::is_trivially_copyable_v<Nvfp4BoundaryOnboardPageTask>);

//! Reuse boundary for this first prototype:
//!
//! * Quantization math is reused directly from `quantization.cuh`.
//! * The task-array idea mirrors KVCM V2's `batchedCopy<N>` so one launch can
//!   cover non-contiguous Pages.  The implementation below does not copy its
//!   four-stage `cp.async`/shared-memory pipeline yet.
//! * A future optimized variant can adapt that pipeline around the transform:
//!   async source load -> quant/dequant in registers -> direct destination
//!   store.  It cannot call `batchedCopy<N>` itself because source and
//!   destination layouts and byte counts differ.
//!
//! Keeping that distinction explicit prevents a functional prototype from
//! being mistaken for a measured port of the production copy kernel.

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
__device__ std::uint32_t scaleOffset(
    std::uint32_t role, std::uint32_t row, std::uint32_t scaleInRow, std::uint32_t scalesPerRow)
{
    if (role == 0)
    {
        return row * scalesPerRow + scaleInRow;
    }
    return (row / 4) * (4 * scalesPerRow) + scaleInRow * 4 + row % 4;
}

//! Convert one lane's eight packed E2M1 values to float. This is the same SM100
//! conversion sequence used by ARCQuant and fused MoE communication, kept local
//! here because those implementations expose no reusable public device helper.
__device__ void e2m1ToFloat8(std::uint32_t packed, float2 (&values)[4])
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
//! The generic contiguous path is
//! `invokeFP4Quantization -> quantize_with_block_size ->
//! cvt_warp_fp16_to_fp4`. This boundary kernel does not call or copy the first
//! two layers; it directly calls their existing innermost device primitive,
//! `cvt_warp_fp16_to_fp4`, from `quantization.cuh`. It also adapts the
//! K-linear/V-token-4 scale order from `quantizeAndWriteFP4KVCache`.
//!
//! The new Page shell batches disjoint Slots and routes K/V buffers. Its
//! packed-data store below and the scale store inside
//! `cvt_warp_fp16_to_fp4` target CUDA-mapped Host addresses, so those GPU
//! global stores are the D2H transfer. There is no call to KVCM V2
//! `batchedCopy<N>`, compact GPU staging buffer, or separate copy kernel in the
//! current implementation.
//!
//! Provenance: DIRECT REUSE = `PackedVec` + `cvt_warp_fp16_to_fp4`;
//! ADAPTED CONTRACT = native-KV K/V scale offsets; NEW = Page-task routing,
//! launch geometry, and mapped-Host stores.
template <std::uint32_t N, typename T>
__global__ void offloadFrom16BitKernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // NEW: grid.z selects one disjoint Page task; grid.y selects K or V.
    std::uint32_t const halfGroup = blockIdx.x * blockDim.x + threadIdx.x;
    if (halfGroup >= halfGroupsPerRole)
    {
        return;
    }

    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];
    std::uint32_t const laneInScale = halfGroup & 1U;
    std::uint32_t const scaleGroup = halfGroup >> 1U;
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const row = scaleGroup / scalesPerRow;
    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
    std::uint32_t const elementOffset = row * static_cast<std::uint32_t>(params.headDim)
        + scaleInRow * kElementsPerBlockScale + laneInScale * kElementsPerLane;

    // REUSED: native KV scale placement and the production FP16/BF16->E2M1
    // quantization primitive. NEW: the outputs are mapped Host Pool addresses.
    PackedVec<T> input = reinterpret_cast<PackedVec<T> const*>(selectRawInput<T>(task, role) + elementOffset)[0];
    std::uint8_t* scale
        = laneInScale == 0 ? selectScaleOutput(task, role) + scaleOffset(role, row, scaleInRow, scalesPerRow) : nullptr;
    std::uint32_t const packed
        = cvt_warp_fp16_to_fp4<T, kElementsPerBlockScale, false>(input, params.nvfp4ScaleOrigQuant[role], scale);
    reinterpret_cast<std::uint32_t*>(selectPackedOutput(task, role))[halfGroup] = packed;
#endif
}

//! FP8 E4M3 GPU Page -> Host NVFP4: fused source restoration, NVFP4
//! quantization, and D2H.
//!
//! This boundary kernel is new. It restores FP8 in registers, then directly
//! reuses the innermost `PackedVec`/`cvt_warp_fp16_to_fp4` device primitive
//! from `quantization.cuh`; it calls neither `invokeFP4Quantization` nor
//! `quantize_with_block_size`, and it does not call the differently shaped
//! `cvt_warp_fp8_to_fp4` helper. Its adapter keeps the source-FP8 and
//! target-NVFP4 scales explicit, batches disjoint Pages, routes K/V, and
//! performs D2H through mapped-Host global stores. It does not call or contain
//! KVCM V2's `batchedCopy<N>` pipeline.
//!
//! Provenance: DIRECT REUSE = `cvt_warp_fp16_to_fp4`; NEW = register-only FP8
//! restoration with an independent source inverse scale, Page/native-KV
//! routing, launch geometry, and mapped-Host stores.
template <std::uint32_t N>
__global__ void offloadFromFp8Kernel(std::array<Nvfp4BoundaryOffloadPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // NEW: grid.z selects one disjoint Page task; grid.y selects K or V.
    std::uint32_t const halfGroup = blockIdx.x * blockDim.x + threadIdx.x;
    if (halfGroup >= halfGroupsPerRole)
    {
        return;
    }

    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];
    std::uint32_t const laneInScale = halfGroup & 1U;
    std::uint32_t const scaleGroup = halfGroup >> 1U;
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const row = scaleGroup / scalesPerRow;
    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
    std::uint32_t const elementOffset = row * static_cast<std::uint32_t>(params.headDim)
        + scaleInRow * kElementsPerBlockScale + laneInScale * kElementsPerLane;

    // NEW: restore the source FP8 values with their own inverse scale before
    // applying the independent target-NVFP4 global scale in the reused core.
    auto const* input = selectRawInput<__nv_fp8_e4m3>(task, role) + elementOffset;
    PackedVec<half> restored;
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        float const lo = static_cast<float>(input[2 * i]) * params.fp8ScaleQuantOrig[role];
        float const hi = static_cast<float>(input[2 * i + 1]) * params.fp8ScaleQuantOrig[role];
        restored.elts[i] = __floats2half2_rn(lo, hi);
    }

    // REUSED: production FP16->E2M1 quantization and native KV scale order.
    // NEW: packed data and scales land directly in mapped Host Pool addresses.
    std::uint8_t* scale
        = laneInScale == 0 ? selectScaleOutput(task, role) + scaleOffset(role, row, scaleInRow, scalesPerRow) : nullptr;
    std::uint32_t const packed
        = cvt_warp_fp16_to_fp4<half, kElementsPerBlockScale, false>(restored, params.nvfp4ScaleOrigQuant[role], scale);
    reinterpret_cast<std::uint32_t*>(selectPackedOutput(task, role))[halfGroup] = packed;
#endif
}

//! Host NVFP4 -> FP16/BF16 GPU Page: fused H2D plus dequantization.
//!
//! This boundary kernel is new. Its local `e2m1ToFloat8` helper adapts the
//! `cvt.rn.f16x2.e2m1x2` sequence in ARCQuant's
//! `e2m1_uint32_to_float8` (also used by fused MoE); it is not a call to a MoE
//! kernel. The packed/scale pointer dereferences are GPU global loads whose
//! addresses map to pinned Host memory; those transactions perform H2D. Native
//! K/V scale math then restores directly into caller-owned FP16/BF16 GPU
//! Slots. The current kernel does not yet contain `batchedCopy<N>`'s
//! `cp.async`/shared-memory source-load pipeline.
//!
//! Provenance: ADAPTED DONOR = ARCQuant/FusedMoE E2M1 PTX and inverse-scale
//! flow; NEW = native-KV scale lookup, Page-task routing, mapped-Host loads,
//! launch geometry, and caller-owned raw-Slot stores.
template <std::uint32_t N, typename T>
__global__ void onboardTo16BitKernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // NEW: grid.z selects one disjoint Page task; grid.y selects K or V.
    std::uint32_t const halfGroup = blockIdx.x * blockDim.x + threadIdx.x;
    if (halfGroup >= halfGroupsPerRole)
    {
        return;
    }

    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];
    std::uint32_t const scaleGroup = halfGroup >> 1U;
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const row = scaleGroup / scalesPerRow;
    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;
    std::uint32_t const elementOffset = halfGroup * kElementsPerLane;

    // REUSED: native KV scale placement and the SM100 E2M1 conversion math.
    // NEW: read mapped Host Pools and write the KVCM-owned raw GPU Slot directly.
    std::uint32_t const packed = reinterpret_cast<std::uint32_t const*>(selectPackedInput(task, role))[halfGroup];
    __nv_fp8_e4m3 blockScale;
    blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
    float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];

    float2 values[4];
    e2m1ToFloat8(packed, values);
    store16BitValues(selectRawOutput<T>(task, role), elementOffset, values, dequantScale);
#endif
}

//! Host NVFP4 -> FP8 E4M3 GPU Page: fused H2D, NVFP4 dequantization, and FP8
//! requantization.
//!
//! This boundary kernel is new. It uses the same local SM100 E2M1 conversion
//! adapted from ARCQuant/fused MoE, then directly reuses
//! `fp32_vec_to_e4m3` from `quantization.cuh`. GPU global loads from the mapped
//! Host packed/scale addresses perform H2D; new scale composition and Page/K/V
//! routing write the caller-owned FP8 GPU Slot without a staging buffer or
//! separate copy kernel. The current kernel borrows the disjoint-task batch
//! shape, not `batchedCopy<N>`'s `cp.async` implementation.
//!
//! Provenance: DIRECT REUSE = `fp32_vec_to_e4m3`; ADAPTED DONOR =
//! ARCQuant/FusedMoE E2M1 PTX; NEW = NVFP4-to-FP8 scale composition,
//! Page/native-KV routing, mapped-Host loads, and launch geometry.
template <std::uint32_t N>
__global__ void onboardToFp8Kernel(std::array<Nvfp4BoundaryOnboardPageTask, N> const __grid_constant__ tasks,
    Nvfp4BoundaryKernelParams params, std::uint32_t halfGroupsPerRole)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // NEW: grid.z selects one disjoint Page task; grid.y selects K or V.
    std::uint32_t const halfGroup = blockIdx.x * blockDim.x + threadIdx.x;
    if (halfGroup >= halfGroupsPerRole)
    {
        return;
    }

    std::uint32_t const role = blockIdx.y;
    auto const& task = tasks[blockIdx.z];
    std::uint32_t const scaleGroup = halfGroup >> 1U;
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(params.headDim) / kElementsPerBlockScale;
    std::uint32_t const row = scaleGroup / scalesPerRow;
    std::uint32_t const scaleInRow = scaleGroup - row * scalesPerRow;

    // REUSED: native KV scale placement, SM100 E2M1 conversion, and the
    // production FP32->E4M3 packer. NEW: mapped-Host input and two-scale
    // composition write directly into the KVCM-owned FP8 GPU Slot.
    std::uint32_t const packed = reinterpret_cast<std::uint32_t const*>(selectPackedInput(task, role))[halfGroup];
    __nv_fp8_e4m3 blockScale;
    blockScale.__x = selectScaleInput(task, role)[scaleOffset(role, row, scaleInRow, scalesPerRow)];
    float const dequantScale
        = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role] * params.fp8ScaleOrigQuant[role];

    float2 values[4];
    e2m1ToFloat8(packed, values);
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        values[i].x *= dequantScale;
        values[i].y *= dequantScale;
    }
    reinterpret_cast<std::uint64_t*>(selectRawOutput<__nv_fp8_e4m3>(task, role))[halfGroup] = fp32_vec_to_e4m3(values);
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

std::uint32_t blocksForHalfGroups(std::uint32_t halfGroups)
{
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(halfGroups) + kThreadsPerBlock - 1) / kThreadsPerBlock);
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
        dim3 const grid(blocksForHalfGroups(halfGroups), 2, count);
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
        dim3 const grid(blocksForHalfGroups(halfGroups), 2, count);
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
        dim3 const grid(blocksForHalfGroups(halfGroups), 2, count);
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
        dim3 const grid(blocksForHalfGroups(halfGroups), 2, count);
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
