/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! One hot buffer resolved from KVCM's authoritative pool descriptors.
struct ResolvedHotBuffer
{
    std::uintptr_t rawBase = 0;
    std::size_t rawSlotBytes = 0;
    std::size_t rawBytes = 0;
};

using ResolvedHotLayer = std::map<kv::DataRole, ResolvedHotBuffer>;

//! One KVCM lifecycle resolved into its hot buffers.
struct ResolvedHotLifecycle
{
    kv::LifeCycleId lifeCycleId{-1};
    std::map<kv::LayerId, ResolvedHotLayer> layers;
};

//! Storage properties produced while an algorithm prepares one lifecycle.
struct ColdPageLifecycleProperties
{
    std::size_t coldPageBytes = 0;
    kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
    std::size_t maxColdPageBytes = 0;
    int tokensPerPage = 0;
};

//! Normalized selection shared by all cold-page quantizers.
struct ColdPageSelectionConfig
{
    int keepFirstTokens = 0;
    int keepLastTokens = 0;
    std::vector<kv::ColdPageTokenRange> keepTokenRanges;
};

//! Resolves KVCM layouts and routes lifecycles for one native compression method.
class NativeColdPageCodec : public kv::IKvCacheColdPageCodec
{
public:
    explicit NativeColdPageCodec(std::set<kv::LayerId> layerIds, ColdPageSelectionConfig selection = {});

    [[nodiscard]] std::vector<std::size_t> coldPageCapacities(kv::LayerGroupId layerGroupId) const final;
    [[nodiscard]] bool selectsTokens(kv::LayerGroupId layerGroupId) const noexcept final;
    std::vector<double> coldPageCapacityFractions(kv::LayerGroupId layerGroupId, int historyLength) const final;
    std::vector<kv::ColdPageTokenRange> protectedTokens(
        kv::LayerGroupId layerGroupId, kv::ColdPageContext const& context) const final;
    [[nodiscard]] std::shared_ptr<kv::ColdPageRepresentation const> prepareColdPage(
        kv::LayerGroupId layerGroupId, kv::ColdPageContext const& context) final;
    [[nodiscard]] int reusablePrefix(kv::LayerGroupId layerGroupId, int startToken, int sequenceLength, int validTokens,
        std::vector<kv::ColdPageTokenRange> const& losslessTokens) const final;

    bool encodeSelected(kv::LayerGroupId layerGroupId, kv::ColdPageRepresentation const* representation,
        void* dstBasePtr, kv::PageIndexPair const* pageIndices, std::size_t numBasePages,
        cudaStream_t stream) noexcept final;
    bool decodeSelected(kv::LayerGroupId layerGroupId, kv::ColdPageRepresentation const* representation,
        void const* srcBasePtr, kv::PageIndexPair const* pageIndices, std::size_t numBasePages,
        cudaStream_t stream) noexcept final;

    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept final;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept final;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

private:
    virtual std::size_t prepareProvider(std::size_t lifecycleIndex, int layoutId, int validTokens,
        std::vector<kv::ColdPageTokenRange> const& rawTokens, std::size_t capacityBytes);
    virtual void encodeSelectedProvider(std::size_t lifecycleIndex, int layoutId, void* coldBase,
        kv::PageIndexPair const* pageIndices, std::size_t numPages, cudaStream_t stream);
    virtual void decodeSelectedProvider(std::size_t lifecycleIndex, int layoutId, void const* coldBase,
        kv::PageIndexPair const* pageIndices, std::size_t numPages, cudaStream_t stream);

    [[nodiscard]] std::vector<kv::ColdPageTokenRange> selectTokens(kv::ColdPageContext const& context) const;
    virtual std::vector<ColdPageLifecycleProperties> configureProvider(
        std::vector<ResolvedHotLifecycle> const& lifecycles)
        = 0;

    //! Enqueue only on stream; this codec drains partial submissions after a throw.
    virtual void encodeProvider(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    virtual void decodeProvider(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    struct LayerGroupState
    {
        std::optional<std::size_t> lifecycleIndex;
        std::size_t coldPageBytes = 0;
        kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
        std::size_t maxColdPageBytes = 0;
        int tokensPerPage = 0;
        int nextLayoutId = 0;
        std::map<std::pair<int, std::vector<kv::ColdPageTokenRange>>, std::shared_ptr<kv::ColdPageRepresentation const>>
            representations;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::set<kv::LayerId> mLayerIds;
    ColdPageSelectionConfig mSelection;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
