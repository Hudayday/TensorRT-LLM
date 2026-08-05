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

#include "bindings.h"
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <vector>

namespace nb = nanobind;
namespace kernels = tensorrt_llm::kernels;

namespace tensorrt_llm::nanobind::kv_cache_compression
{
namespace
{

using BoundaryAddressTuple = std::array<std::uintptr_t, 6>;
using BoundaryScalePair = std::array<float, 2>;

//! Build the immutable parameters shared by one homogeneous Page/layer cohort.
//!
//! Keeping this conversion in the binding makes the Python prototype pass only
//! plain values and raw addresses. It does not create Tensor views over KVCM
//! memory, switch streams, or acquire ownership of any source/destination Slot.
kernels::Nvfp4BoundaryKernelParams makeBoundaryParams(std::int32_t numKvHeads, std::int32_t tokensPerPage,
    std::int32_t headDim, BoundaryScalePair const& nvfp4ScaleOrigQuant, BoundaryScalePair const& nvfp4ScaleQuantOrig,
    BoundaryScalePair const& fp8ScaleOrigQuant, BoundaryScalePair const& fp8ScaleQuantOrig)
{
    kernels::Nvfp4BoundaryKernelParams params{};
    params.numKvHeads = numKvHeads;
    params.tokensPerPage = tokensPerPage;
    params.headDim = headDim;
    for (std::size_t role = 0; role < 2; ++role)
    {
        params.nvfp4ScaleOrigQuant[role] = nvfp4ScaleOrigQuant[role];
        params.nvfp4ScaleQuantOrig[role] = nvfp4ScaleQuantOrig[role];
        params.fp8ScaleOrigQuant[role] = fp8ScaleOrigQuant[role];
        params.fp8ScaleQuantOrig[role] = fp8ScaleQuantOrig[role];
    }
    return params;
}

} // namespace

void initBindings(nb::module_& module)
{
    nb::enum_<kernels::Nvfp4BoundaryRuntimeType>(module, "Nvfp4BoundaryRuntimeType")
        .value("FLOAT16", kernels::Nvfp4BoundaryRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4BoundaryRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3);

    // Prototype/Python-parity bridge for boundary compression. The Python
    // Compression Manager lowers KVCM-owned addresses to plain tuples and
    // crosses into the native launchers here. This binding neither selects
    // the compression algorithm nor owns Page residency/migration.
    //
    // The product C++ StorageManager/Compression Manager path will call the
    // same native launchers directly; it must not route _batchedMigrate()
    // through Python or nanobind.
    //
    // Every tuple uses one canonical order in both directions:
    //   raw K, raw V, packed K, packed V, K block scales, V block scales.
    // A tuple is one (Page, layer) task. The native launcher batches all tasks
    // in this homogeneous cohort and internally chunks only very large batches.
    module.def(
        "nvfp4_boundary_offload_compress",
        [](std::vector<BoundaryAddressTuple> const& addresses, std::int32_t numKvHeads, std::int32_t tokensPerPage,
            std::int32_t headDim, BoundaryScalePair const& nvfp4ScaleOrigQuant,
            BoundaryScalePair const& nvfp4ScaleQuantOrig, BoundaryScalePair const& fp8ScaleOrigQuant,
            BoundaryScalePair const& fp8ScaleQuantOrig, kernels::Nvfp4BoundaryRuntimeType runtimeType,
            std::uintptr_t stream)
        {
            std::vector<kernels::Nvfp4BoundaryOffloadPageTask> tasks;
            tasks.reserve(addresses.size());
            for (auto const& address : addresses)
            {
                tasks.push_back({reinterpret_cast<void const*>(address[0]), reinterpret_cast<void const*>(address[1]),
                    reinterpret_cast<std::uint8_t*>(address[2]), reinterpret_cast<std::uint8_t*>(address[3]),
                    reinterpret_cast<std::uint8_t*>(address[4]), reinterpret_cast<std::uint8_t*>(address[5])});
            }
            auto const params = makeBoundaryParams(numKvHeads, tokensPerPage, headDim, nvfp4ScaleOrigQuant,
                nvfp4ScaleQuantOrig, fp8ScaleOrigQuant, fp8ScaleQuantOrig);
            kernels::invokeNvfp4BoundaryOffloadCompress(
                tasks, params, runtimeType, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("addresses"), nb::arg("num_kv_heads"), nb::arg("tokens_per_page"), nb::arg("head_dim"),
        nb::arg("nvfp4_scale_orig_quant"), nb::arg("nvfp4_scale_quant_orig"), nb::arg("fp8_scale_orig_quant"),
        nb::arg("fp8_scale_quant_orig"), nb::arg("runtime_type"), nb::arg("stream"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Compress a homogeneous batch of non-contiguous GPU KV Pages directly into mapped Host NVFP4 buffers");

    module.def(
        "nvfp4_boundary_onboard_decompress",
        [](std::vector<BoundaryAddressTuple> const& addresses, std::int32_t numKvHeads, std::int32_t tokensPerPage,
            std::int32_t headDim, BoundaryScalePair const& nvfp4ScaleOrigQuant,
            BoundaryScalePair const& nvfp4ScaleQuantOrig, BoundaryScalePair const& fp8ScaleOrigQuant,
            BoundaryScalePair const& fp8ScaleQuantOrig, kernels::Nvfp4BoundaryRuntimeType runtimeType,
            std::uintptr_t stream)
        {
            std::vector<kernels::Nvfp4BoundaryOnboardPageTask> tasks;
            tasks.reserve(addresses.size());
            for (auto const& address : addresses)
            {
                tasks.push_back({reinterpret_cast<std::uint8_t const*>(address[2]),
                    reinterpret_cast<std::uint8_t const*>(address[3]),
                    reinterpret_cast<std::uint8_t const*>(address[4]),
                    reinterpret_cast<std::uint8_t const*>(address[5]), reinterpret_cast<void*>(address[0]),
                    reinterpret_cast<void*>(address[1])});
            }
            auto const params = makeBoundaryParams(numKvHeads, tokensPerPage, headDim, nvfp4ScaleOrigQuant,
                nvfp4ScaleQuantOrig, fp8ScaleOrigQuant, fp8ScaleQuantOrig);
            kernels::invokeNvfp4BoundaryOnboardDecompress(
                tasks, params, runtimeType, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("addresses"), nb::arg("num_kv_heads"), nb::arg("tokens_per_page"), nb::arg("head_dim"),
        nb::arg("nvfp4_scale_orig_quant"), nb::arg("nvfp4_scale_quant_orig"), nb::arg("fp8_scale_orig_quant"),
        nb::arg("fp8_scale_quant_orig"), nb::arg("runtime_type"), nb::arg("stream"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Restore mapped Host NVFP4 buffers directly into a homogeneous batch of non-contiguous GPU KV Pages");
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
