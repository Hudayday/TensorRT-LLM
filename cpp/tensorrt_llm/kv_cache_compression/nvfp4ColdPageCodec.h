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

#include "kv_cache_manager_v2/coldPageCodec.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! Algorithm-owned metadata for one KV layer.
//!
//! GPU addresses and Pool/Slot geometry intentionally do not appear here.
//! They are discovered from KVCM V2's authoritative PoolGroupDesc during
//! configure(). The model loader supplies only geometry, runtime dtype, and
//! immutable calibration owned by QuantizationCompression.
struct Nvfp4ColdPageLayerConfig
{
    kv::LayerGroupId layerGroupId{0};
    kv::LayerId layerId = 0;
    kernels::Nvfp4BoundaryRuntimeType runtimeType = kernels::Nvfp4BoundaryRuntimeType::kFloat16;
    std::int32_t numKvHeads = 0;
    std::int32_t tokensPerPage = 0;
    std::int32_t headDim = 0;
    std::array<float, 2> nvfp4ScaleOrigQuant{};
    std::array<float, 2> nvfp4ScaleQuantOrig{};
    std::array<float, 2> fp8ScaleOrigQuant{1.0F, 1.0F};
    std::array<float, 2> fp8ScaleQuantOrig{1.0F, 1.0F};
};

//! NVFP4 implementation behind QuantizationCompression's native data plane.
//!
//! QuantizationCompression owns this codec's configuration, calibrated scales,
//! registration, and lifetime. KVCM remains the sole Page-lifecycle authority:
//! it selects Pages, allocates Slots, invokes encode()/decode() from its
//! migration transaction, fences work, and publishes or rolls back mappings.
//! C++ never calls back into Python on the hot path.
//!
//! A compact Host Slot concatenates one 16-byte-aligned record per configured
//! layer. Each layer record is [K packed | V packed | K scales | V scales].
//! KVCM uses only the fixed Slot stride and does not interpret these offsets.
//!
//! The only type crossing the manager boundary is this native object: Python
//! owns and registers it but is absent from Page migration. Page selection,
//! admission, fencing, publication, rollback, and eviction remain KVCM
//! responsibilities.
class Nvfp4ColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs);

    //! Consume one authoritative GPU PoolGroup layout.
    //!
    //! KVCM may call this once for each PoolGroupDesc. Configuration is
    //! transactional: a rejected descriptor does not publish partial state.
    bool configure(kv::PoolGroupDesc const& gpuDesc) noexcept override;

    //! Return the fixed byte stride of one compact Host Slot for a supported
    //! layer group. Zero indicates failure or an unknown layer group.
    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Merge lifecycle calls only when their complete physical NVFP4 transform
    //! is identical. KVCM still verifies same-PoolGroup membership and stride.
    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Request Host Page-index pairs. encode()/decode() lower every pair into
    //! self-contained kernel tasks before returning, so CUDA work does not
    //! retain the PageIndexPair array.
    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Enqueue GPU runtime KV -> mapped-Host NVFP4 for disjoint base Pages.
    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

    //! Enqueue mapped-Host NVFP4 -> GPU runtime KV for disjoint base Pages.
    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

private:
    struct BufferLocation
    {
        kv::PoolIndex poolIndex{0};
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };

    struct LayerState
    {
        Nvfp4ColdPageLayerConfig config;
        BufferLocation gpuK;
        BufferLocation gpuV;
        std::size_t coldOffset = 0;
    };

    struct LayerGroupState
    {
        // GPU virtual bases and Slot strides are stable layout state. KVCM's
        // live Slot capacity is deliberately not cached here: dynamic Pool
        // resizing remains allocator-owned and may issue larger Slot IDs.
        kv::TypedVec<kv::PoolIndex, kv::PoolDesc> gpuPools;
        std::vector<LayerState> layers;
        std::size_t coldPageBytes = 0;
    };

    std::vector<Nvfp4ColdPageLayerConfig> mLayerConfigs;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::map<kv::LayerGroupId, kv::LayerGroupId> mBatchingLayerGroups;
};

} // namespace tensorrt_llm::kv_cache_compression
