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

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>

namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
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

std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

void KVCacheManagerV2UtilsBindings::initBindings(nb::module_& module)
{
    nb::enum_<kernels::Nvfp4BoundaryRuntimeType>(module, "Nvfp4BoundaryRuntimeType")
        .value("FLOAT16", kernels::Nvfp4BoundaryRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4BoundaryRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3);

    // Bind DiskAddress struct
    nb::class_<DiskAddress>(module, "DiskAddress")
        .def(nb::init<int, ssize_t>(), nb::arg("fd"), nb::arg("pos"))
        .def_rw("fd", &DiskAddress::fd)
        .def_rw("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    nb::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(nb::init<DiskAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_rw("src", &Task<DiskAddress, DiskAddress>::src);

    nb::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(nb::init<MemAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_rw("src", &Task<MemAddress, DiskAddress>::src);

    nb::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(nb::init<DiskAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_rw("src", &Task<DiskAddress, MemAddress>::src);

    nb::class_<Task<MemAddress, MemAddress>>(module, "MemToMemTask")
        .def(nb::init<MemAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, MemAddress>::dst)
        .def_rw("src", &Task<MemAddress, MemAddress>::src);

    nb::class_<IndexMapper>(module, "IndexMapper")
        .def(nb::init<SizeType32, SizeType32>(), nb::arg("max_batch_size"), nb::arg("max_beam_width"))
        .def("add_new_sequence", &IndexMapper::addNewSequence)
        .def("get_index", &IndexMapper::getIndex)
        .def("remove_sequence", &IndexMapper::removeSequence)
        .def("get_copy_index", &IndexMapper::getCopyIndex)
        .def("gather_k_block_offsets", &IndexMapper::gatherKBlockOffsets, nb::arg("source"), nb::arg("destination"),
            nb::arg("request_ids"), nb::arg("num_blocks"))
        .def("size", &IndexMapper::size)
        .def("num_free_slots", &IndexMapper::numFreeSlots);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to disk using CUDA host function");

    module.def(
        "copy_host_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to host using CUDA host function");

    module.def(
        "copy_host_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to device using CUDA kernels");

    module.def(
        "copy_device_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to host using CUDA kernels");

    module.def(
        "copy_device_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to device using CUDA kernels");

    // Prototype/Python-parity bridge for boundary compression.  The product
    // C++ StorageManager will call the same native functions directly; it must
    // not callback through Python from _batchedMigrate().
    //
    // Every tuple uses one canonical order in both directions:
    //   raw K, raw V, packed K, packed V, K block scales, V block scales.
    // A tuple is one (Page, layer) task.  The native launcher batches all tasks
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

    module.def(
        "copy_batch_block_offsets_to_device",
        [](at::Tensor input, at::Tensor output, at::Tensor copyIndex, at::Tensor indexScales, at::Tensor kvOffset,
            uintptr_t stream)
        {
            auto _input = from_torch(input);
            auto _output = from_torch(output);
            auto _copyIndex = from_torch(copyIndex);
            auto _indexScales = from_torch(indexScales);
            auto _kvOffset = from_torch(kvOffset);
            TLLM_CHECK_WITH_INFO(_input.has_value(), "Invalid input tensor.");
            TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
            TLLM_CHECK_WITH_INFO(_copyIndex.has_value(), "Invalid copy index tensor.");
            TLLM_CHECK_WITH_INFO(_indexScales.has_value(), "Invalid index scales tensor.");
            TLLM_CHECK_WITH_INFO(_kvOffset.has_value(), "Invalid kv offset tensor.");
            copyBatchBlockOffsetsToDevice(*(_input.value()), *(_output.value()), *(_copyIndex.value()),
                *(_indexScales.value()), *(_kvOffset.value()), reinterpret_cast<CUstream>(stream));
        },
        nb::arg("input"), nb::arg("output"), nb::arg("copy_index"), nb::arg("index_scales"), nb::arg("kv_offset"),
        nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(), "Copy batch block indices to device");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
