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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Active GPU representation restored at the Host boundary.
//
//! This is an algorithm-side dispatch value, not Page metadata. KVCM owns the
//! runtime layout and supplies one homogeneous layer cohort per call.
enum class Nvfp4BoundaryRuntimeType : std::uint8_t
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

//! One GPU-to-Host Page transform.
//!
//! The six addresses are intentionally independent: KVCM may coalesce logical
//! roles into physical Pools, but the kernel never assumes adjacency between
//! K, V, their scales, or different Pages. The packed and scale addresses must
//! be device-visible aliases of CUDA-registered mapped Host memory. The raw
//! inputs and mapped-Host outputs must remain valid and non-overwritable until
//! work submitted to the caller's stream completes.
struct Nvfp4BoundaryOffloadPageTask
{
    void const* rawK;
    void const* rawV;
    std::uint8_t* packedK;
    std::uint8_t* packedV;
    std::uint8_t* blockScaleK;
    std::uint8_t* blockScaleV;
};

//! One Host-to-GPU Page transform.
//!
//! The packed and scale inputs must remain mapped and immutable until the
//! caller-observed stream completion event fires. The raw GPU outputs must stay
//! reserved for the same interval.
struct Nvfp4BoundaryOnboardPageTask
{
    std::uint8_t const* packedK;
    std::uint8_t const* packedV;
    std::uint8_t const* blockScaleK;
    std::uint8_t const* blockScaleV;
    void* rawK;
    void* rawV;
};

//! Geometry and immutable per-layer scale convention shared by one Page batch.
//!
//! Each raw K/V pointer describes one HND Page with
//! `numKvHeads * tokensPerPage * headDim` elements. Each packed K/V pointer has
//! half that many bytes in the same HND order. Each block-scale pointer has one
//! E4M3 byte per 16 elements. K scales are linear `[H, P, D / 16]`; V scales
//! use the native token-4 order over flattened `(head, token)` rows. Therefore
//! `tokensPerPage` must be divisible by four and `headDim` by 16.
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

//! Compress a homogeneous GPU Page cohort and transfer it directly to mapped Host memory.
//
//! The caller owns Page selection, source/destination lifetime, the CUDA
//! stream, completion fencing, publication, release, and rollback. This
//! function submits only the complete NVFP4 transform-plus-transfer payload.
void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType, cudaStream_t stream);

//! Transfer a homogeneous mapped-Host cohort and restore the active GPU representation.
//
//! For a non-empty batch, successful return means work was enqueued on
//! ``stream``; it does not publish the destination or synchronize the stream.
void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const& params, Nvfp4BoundaryRuntimeType runtimeType, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
