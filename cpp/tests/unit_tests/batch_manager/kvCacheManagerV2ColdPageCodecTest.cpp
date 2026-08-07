/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

class RecordingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    struct Call
    {
        bool encode;
        LayerGroupId layerGroup;
        std::vector<std::int32_t> dstIndices;
        std::vector<std::int32_t> srcIndices;
        cudaStream_t stream;
    };

    bool configure(PoolGroupDesc const&) override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const override
    {
        return 1U << 20;
    }

    bool encode(LayerGroupId layerGroup, void*, std::int32_t const* dstIndices, std::int32_t const* srcIndices,
        size_t count, cudaStream_t stream) override
    {
        calls.push_back(
            Call{true, layerGroup, {dstIndices, dstIndices + count}, {srcIndices, srcIndices + count}, stream});
        return submit;
    }

    bool decode(LayerGroupId layerGroup, std::int32_t const* dstIndices, void const*, std::int32_t const* srcIndices,
        size_t count, cudaStream_t stream) override
    {
        calls.push_back(
            Call{false, layerGroup, {dstIndices, dstIndices + count}, {srcIndices, srcIndices + count}, stream});
        return submit;
    }

    bool submit = true;
    std::vector<Call> calls;
};

KVCacheManagerConfig makeTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 1 << 20, std::nullopt});
    layer.buffers.push_back(BufferConfig{"value", 1 << 20, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    return config;
}

TEST(KvCacheManagerV2ColdPageCodecTest, InstallsCompactHostLayoutAndRoutesBothDirections)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto codec = std::make_shared<RecordingColdPageCodec>();
    manager->setColdPageCodec(codec);

    auto& storage = manager->storage();
    PoolGroupIndex const poolGroup{0};
    LifeCycleId const lifeCycle{0};
    // Equal-size K/V buffers are coalesced into one existing GPU Pool; the
    // cold tier is independently one compact Pool with a smaller Slot stride.
    EXPECT_EQ(storage.numPools(poolGroup, kGpuLevel), PoolIndex{1});
    EXPECT_EQ(storage.numPools(poolGroup, CacheLevel{1}), PoolIndex{1});
    ASSERT_EQ(storage.slotSize(poolGroup, CacheLevel{1}).size(), PoolIndex{1});
    EXPECT_EQ(storage.slotSize(poolGroup, CacheLevel{1}).front(), 1U << 20);

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    int tokenBase = 0;
    auto makeCommittedPages = [&](std::vector<Slot> slots)
    {
        std::vector<SharedPtr<Page>> pages;
        NodeBase* previous = &root;
        for (auto& slot : slots)
        {
            std::vector<TokenIdExt> tokens;
            for (int i = 0; i < manager->tokensPerBlock(); ++i)
                tokens.emplace_back(TokenId{tokenBase++});
            auto block = addOrGetExistingBlock(previous, LifeCycleId{1}, std::move(tokens));
            auto page = makeShared<CommittedPage>(
                &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
            page->setSlot(slot);
            block->storage[lifeCycle] = page.get();
            storage.scheduleForEviction(*page);
            pages.push_back(page);
            previous = block.get();
        }
        return pages;
    };

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto initialSlots = storage.newGpuSlots(twoSlots);
    auto pages = makeCommittedPages(std::move(initialSlots[lifeCycle]));

    // A second full GPU allocation forces the two reusable Pages to Host.
    auto temporarySlots = storage.newGpuSlots(twoSlots);
    ASSERT_EQ(codec->calls.size(), 1);
    EXPECT_TRUE(codec->calls.front().encode);
    EXPECT_EQ(codec->calls.front().srcIndices.size(), 2);
    for (auto& slot : temporarySlots[lifeCycle])
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));

    auto cache = manager->createKvCache();
    std::vector<BatchedLockTarget> targets;
    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{2}; ++ordinal)
    {
        auto const& page = pages.at(toSizeT(ordinal));
        storage.excludeFromEviction(*page);
        targets.push_back({page, kDefaultBeamIndex, ordinal, lifeCycle});
    }
    storage.batchedMigrateToGpu(targets, *cache, {});

    ASSERT_EQ(codec->calls.size(), 2);
    EXPECT_FALSE(codec->calls.back().encode);
    EXPECT_EQ(codec->calls.back().dstIndices.size(), 2);
    EXPECT_TRUE(
        std::all_of(pages.begin(), pages.end(), [](auto const& page) { return page->cacheLevel == kGpuLevel; }));

    cache->close();
    manager->setColdPageCodec(nullptr);
}

} // namespace
