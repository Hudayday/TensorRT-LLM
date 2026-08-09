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
//! QuantizationCompression is the lifecycle authority: it constructs and
//! configures this object, then injects the same native object into KVCM V2.
//! KVCM may retain shared ownership for asynchronous safety and invokes
//! encode()/decode() directly from its migration transaction; C++ never calls
//! back into Python on the hot path. The integration owner must detach the
//! codec before tearing down the compression manager/KVCM pair.
//!
//! This class intentionally matches IKvCacheColdPageCodec's configure/query/
//! encode/decode contract. Once the KVCM-side interface lands, the only type
//! integration required here is inheritance plus override specifiers; Page
//! selection, admission, fencing, publication, rollback, and eviction remain
//! KVCM responsibilities.
class Nvfp4ColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs);

    //! Consume one authoritative GPU PoolGroup layout.
    //!
    //! KVCM may call this once for each PoolGroupDesc. Configuration is
    //! transactional: a rejected descriptor does not publish partial state.
    bool configure(kv::PoolGroupDesc const& gpuDesc) noexcept override;

    //! Return the fixed byte stride of one compact Host Slot for a layer group.
    //! Zero means that the group is not configured or its layout was rejected.
    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Merge lifecycle calls only when their complete physical NVFP4 transform
    //! is identical. KVCM still verifies same-PoolGroup membership and stride.
    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Request Host Page-index pairs. encode()/decode() lower every pair into
    //! self-contained kernel tasks before returning, so CUDA work does not
    //! retain the PageIndexPair array.
    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(
        kv::LayerGroupId layerGroupId) const noexcept override;

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
        kv::PoolGroupDesc gpuDesc;
        std::vector<LayerState> layers;
        std::size_t coldPageBytes = 0;
    };

    std::vector<Nvfp4ColdPageLayerConfig> mLayerConfigs;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::map<kv::LayerGroupId, kv::LayerGroupId> mBatchingLayerGroups;
};

} // namespace tensorrt_llm::kv_cache_compression
