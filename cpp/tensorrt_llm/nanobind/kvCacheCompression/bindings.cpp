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
#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
namespace compression = tensorrt_llm::kv_cache_compression;
namespace kernels = tensorrt_llm::kernels;
namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace
{

//! Convert either Python KVCM V2's dataclass descriptor or C++ KVCM V2's
//! nanobind descriptor into the codec's native initialization contract.
//!
//! Both backends deliberately expose the same field names. Keeping this
//! one-time adapter beside the codec binding avoids a Python callback on the
//! migration path and does not duplicate runtime Page addresses or geometry.
kv::PoolGroupDesc toNativePoolGroupDesc(nb::handle value)
{
    auto const desc = nb::borrow<nb::object>(value);
    kv::PoolGroupDesc nativeDesc;
    nativeDesc.poolGroupIndex = kv::PoolGroupIndex{nb::cast<int>(desc.attr("pool_group_index"))};
    nativeDesc.numSlots = kv::SlotCount{nb::cast<int>(desc.attr("num_slots"))};

    auto const slotDesc = desc.attr("slot_desc");
    for (nb::handle variantValue : nb::cast<nb::iterable>(slotDesc.attr("variants")))
    {
        auto const variant = nb::borrow<nb::object>(variantValue);
        kv::SlotDescVariant nativeVariant;
        nativeVariant.lifeCycleId = kv::LifeCycleId{nb::cast<int>(variant.attr("layer_group_id"))};
        for (nb::handle coalescedValue : nb::cast<nb::iterable>(variant.attr("coalesced_buffers")))
        {
            auto const coalesced = nb::borrow<nb::object>(coalescedValue);
            kv::CoalescedBuffer nativeCoalesced;
            nativeCoalesced.singleBufferSize = nb::cast<std::size_t>(coalesced.attr("single_buffer_size"));
            for (nb::handle bufferValue : nb::cast<nb::iterable>(coalesced.attr("buffer_ids")))
            {
                auto const buffer = nb::borrow<nb::object>(bufferValue);
                nativeCoalesced.bufferIds.push_back(
                    {nb::cast<int>(buffer.attr("layer_id")), nb::cast<std::string>(buffer.attr("role"))});
            }
            nativeVariant.coalescedBuffers.push_back(std::move(nativeCoalesced));
        }
        nativeDesc.slotDesc.variants.push_back(std::move(nativeVariant));
    }

    int expectedPoolIndex = 0;
    for (nb::handle poolValue : nb::cast<nb::iterable>(desc.attr("pools")))
    {
        auto const pool = nb::borrow<nb::object>(poolValue);
        int const poolIndex = nb::cast<int>(pool.attr("pool_index"));
        if (poolIndex != expectedPoolIndex)
        {
            throw std::invalid_argument("PoolGroupDesc pools must be ordered by contiguous pool_index");
        }
        nativeDesc.pools.push_back({kv::PoolIndex{poolIndex}, nb::cast<kv::MemAddress>(pool.attr("base_address")),
            nb::cast<std::size_t>(pool.attr("slot_bytes"))});
        ++expectedPoolIndex;
    }
    return nativeDesc;
}

} // namespace

namespace tensorrt_llm::nanobind::kv_cache_compression
{

void initBindings(nb::module_& module)
{
    nb::enum_<kernels::Nvfp4BoundaryRuntimeType>(module, "Nvfp4BoundaryRuntimeType")
        .value("FLOAT16", kernels::Nvfp4BoundaryRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4BoundaryRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3);

    nb::enum_<kv::PageIndexLocation>(module, "PageIndexLocation")
        .value("HOST", kv::PageIndexLocation::kHost)
        .value("DEVICE", kv::PageIndexLocation::kDevice);

    nb::class_<compression::Nvfp4ColdPageLayerConfig>(module, "Nvfp4ColdPageLayerConfig")
        .def(nb::init<>())
        .def_prop_rw(
            "layer_group_id",
            [](compression::Nvfp4ColdPageLayerConfig const& self) { return self.layerGroupId.value(); },
            [](compression::Nvfp4ColdPageLayerConfig& self, int value) { self.layerGroupId = kv::LayerGroupId{value}; })
        .def_rw("layer_id", &compression::Nvfp4ColdPageLayerConfig::layerId)
        .def_rw("runtime_type", &compression::Nvfp4ColdPageLayerConfig::runtimeType)
        .def_rw("num_kv_heads", &compression::Nvfp4ColdPageLayerConfig::numKvHeads)
        .def_rw("tokens_per_page", &compression::Nvfp4ColdPageLayerConfig::tokensPerPage)
        .def_rw("head_dim", &compression::Nvfp4ColdPageLayerConfig::headDim)
        .def_rw("nvfp4_scale_orig_quant", &compression::Nvfp4ColdPageLayerConfig::nvfp4ScaleOrigQuant)
        .def_rw("nvfp4_scale_quant_orig", &compression::Nvfp4ColdPageLayerConfig::nvfp4ScaleQuantOrig)
        .def_rw("fp8_scale_orig_quant", &compression::Nvfp4ColdPageLayerConfig::fp8ScaleOrigQuant)
        .def_rw("fp8_scale_quant_orig", &compression::Nvfp4ColdPageLayerConfig::fp8ScaleQuantOrig);

    // Yao Yao's KVCM-facing hook type. It is intentionally data-plane only:
    // Python creates a concrete codec and hands the same native object to the
    // selected Python or C++ KVCM V2 implementation during initialization.
    nb::class_<kv::IKvCacheColdPageCodec>(module, "IKvCacheColdPageCodec");

    // QuantizationCompression constructs, configures, and retains this native
    // object. KVCM's future native integration receives the same object and
    // invokes the exact C++ methods directly, without a Python callback.
    nb::class_<compression::Nvfp4ColdPageCodec, kv::IKvCacheColdPageCodec>(module, "Nvfp4ColdPageCodec")
        .def(nb::init<std::vector<compression::Nvfp4ColdPageLayerConfig>>(), nb::arg("layer_configs"))
        .def("configure",
            [](compression::Nvfp4ColdPageCodec& self, nb::object const& gpuDesc)
            { return self.configure(toNativePoolGroupDesc(gpuDesc)); },
            nb::arg("gpu_desc"),
            "Configure one authoritative GPU PoolGroupDesc")
        .def(
            "query_cold_page_bytes", [](compression::Nvfp4ColdPageCodec const& self, int layerGroupId)
            { return self.queryColdPageBytes(kv::LayerGroupId{layerGroupId}); }, nb::arg("layer_group_id"),
            "Return the fixed byte stride of one compact Host Slot")
        .def(
            "get_batching_layer_group_id", [](compression::Nvfp4ColdPageCodec const& self, int layerGroupId)
            { return self.getBatchingLayerGroupId(kv::LayerGroupId{layerGroupId}).value(); }, nb::arg("layer_group_id"),
            "Return the representative layer group used for codec batching")
        .def(
            "query_page_index_location", [](compression::Nvfp4ColdPageCodec const& self, int layerGroupId)
            { return static_cast<int>(self.queryPageIndexLocation(kv::LayerGroupId{layerGroupId})); },
            nb::arg("layer_group_id"), "Return 0 for Host Page-index pairs or 1 for Device Page-index pairs")
        .def(
            "encode",
            [](compression::Nvfp4ColdPageCodec& self, int layerGroupId, std::uintptr_t dstBasePtr,
                std::vector<std::array<std::int32_t, 2>> const& pageIndexPairs, std::uintptr_t stream)
            {
                std::vector<kv::PageIndexPair> pageIndices;
                pageIndices.reserve(pageIndexPairs.size());
                for (auto const& pair : pageIndexPairs)
                {
                    pageIndices.push_back({pair[0], pair[1]});
                }
                return self.encode(kv::LayerGroupId{layerGroupId}, reinterpret_cast<void*>(dstBasePtr),
                    pageIndices.data(), pageIndices.size(), reinterpret_cast<cudaStream_t>(stream));
            },
            nb::arg("layer_group_id"), nb::arg("dst_base_ptr"), nb::arg("page_index_pairs"), nb::arg("stream"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Enqueue GPU runtime KV to mapped-Host NVFP4 for disjoint base Pages")
        .def(
            "decode",
            [](compression::Nvfp4ColdPageCodec& self, int layerGroupId, std::uintptr_t srcBasePtr,
                std::vector<std::array<std::int32_t, 2>> const& pageIndexPairs, std::uintptr_t stream)
            {
                std::vector<kv::PageIndexPair> pageIndices;
                pageIndices.reserve(pageIndexPairs.size());
                for (auto const& pair : pageIndexPairs)
                {
                    pageIndices.push_back({pair[0], pair[1]});
                }
                return self.decode(kv::LayerGroupId{layerGroupId}, reinterpret_cast<void const*>(srcBasePtr),
                    pageIndices.data(), pageIndices.size(), reinterpret_cast<cudaStream_t>(stream));
            },
            nb::arg("layer_group_id"), nb::arg("src_base_ptr"), nb::arg("page_index_pairs"), nb::arg("stream"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Enqueue mapped-Host NVFP4 to GPU runtime KV for disjoint base Pages");
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
