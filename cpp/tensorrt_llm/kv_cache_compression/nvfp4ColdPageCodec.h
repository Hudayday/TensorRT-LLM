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
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
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
//! QuantizationCompression owns the algorithm choice and calibrated
//! configuration. A native factory constructs this codec before KVCM;
//! nanobind then transfers its unique ownership into KVCM's constructor. Native
//! StorageManager owns the codec lifetime, selects Pages, allocates Slots,
//! invokes encode()/decode(), fences work, and publishes mappings. Synchronous
//! launch rejection is rolled back before publication; an asynchronous CUDA
//! fault remains worker-fatal. C++ never calls back into Python on the
//! migration path.
//!
//! A compact Host Slot concatenates one 16-byte-aligned record per configured
//! layer. Each layer record is [K packed | V packed | K scales | V scales].
//! KVCM uses only the fixed Slot stride and does not interpret these offsets.
//!
//! The only object crossing the manager boundary is the constructor-owned
//! native codec. There is no late setter or unregister path. Page selection,
//! admission, fencing, publication, rollback, and eviction remain KVCM
//! responsibilities.
class Nvfp4ColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs);

    //! Consume every authoritative GPU PoolGroup layout exactly once.
    //!
    //! Layer configs are keyed only by BufferId.layerId. This call discovers
    //! which lifecycle owns each Attention layer, builds the immutable kernel
    //! plan, and gives every other lifecycle a lossless concat plan.
    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept override;

    //! Return the fixed byte stride of one compact Host Slot for a supported
    //! layer group. Zero indicates failure or an unknown layer group.
    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    //! Keep batching lifecycle-keyed. KVCM may share physical cold PoolGroups
    //! without promising that two lifecycle transforms are interchangeable.
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
    enum class Transform
    {
        kNvfp4Attention,
        kLosslessConcat,
    };

    struct LayerGroupState
    {
        Transform transform = Transform::kLosslessConcat;
        kernels::Nvfp4BoundaryPreparedPlan preparedPlan;
        std::size_t coldPageBytes = 0;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::vector<Nvfp4ColdPageLayerConfig> mLayerConfigs;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
    bool mConfigured = false;
};

} // namespace tensorrt_llm::kv_cache_compression
