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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/pendingStats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/stats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

KVCacheManagerConfig makeConfig(bool enableStats = true)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    config.enableStats = enableStats;
    return config;
}

KVCacheManagerConfig makeDiskTieredConfig()
{
    auto config = makeConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    return config;
}

KVCacheManagerConfig makeTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 2 << 20, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    return config;
}

KVCacheManagerConfig makeGpuTieredConfig()
{
    auto config = makeTieredConfig();
    config.cacheTiers[1] = GpuCacheTierConfig{4 << 20};
    return config;
}

KVCacheManagerConfig makeSplitColdGroupingConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});

    AttentionLayerConfig first;
    first.layerId = 0;
    first.slidingWindowSize = 128;
    first.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(first));

    AttentionLayerConfig second;
    second.layerId = 1;
    second.slidingWindowSize = 256;
    second.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(second));
    return config;
}

class RejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return false;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

class SplitColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    explicit SplitColdPageCodec(bool batchTogether = false)
        : mBatchTogether(batchTogether)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept override
    {
        if (layerGroupId == LifeCycleId{0})
            return 1024;
        if (layerGroupId == LifeCycleId{1})
            return 2048;
        return 0;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept override
    {
        return mBatchTogether ? LifeCycleId{0} : layerGroupId;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

private:
    bool mBatchTogether;
};

class OversizedColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return std::numeric_limits<size_t>::max() / 3 + 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

class MixedIndexLocationColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1024;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId) const noexcept override
    {
        return LifeCycleId{0};
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept override
    {
        return layerGroupId == LifeCycleId{0} ? PageIndexLocation::kHost : PageIndexLocation::kDevice;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

TEST(KvCacheManagerV2StatsTest, ConstructionFailurePreservesCodecUniquePtr)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<RejectingColdPageCodec>();
    auto* codecPtr = codec.get();

    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        std::invalid_argument);

    EXPECT_EQ(codec.get(), codecPtr);
}

TEST(KvCacheManagerV2StatsTest, RejectsColdPageStagingSizeOverflow)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<OversizedColdPageCodec>();
    auto* codecPtr = codec.get();

    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeDiskTieredConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        std::overflow_error);

    EXPECT_EQ(codec.get(), codecPtr);
}

TEST(KvCacheManagerV2StatsTest, DoesNotSizePageStagingWithoutColdTier)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<OversizedColdPageCodec>();

    EXPECT_NO_THROW({
        auto manager = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::move(codec));
        (void) manager;
    });
    EXPECT_EQ(codec, nullptr);
}

TEST(KvCacheManagerV2StatsTest, ColdGpuTierSupportsSingleSlotRoundTrip)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeGpuTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hotSlots = storage.newSlots(kGpuLevel, oneSlot);
    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    ASSERT_EQ(hotSlots[lifeCycle].size(), 1);
    ASSERT_EQ(coldSlots[lifeCycle].size(), 1);

    Slot& hotSlot = hotSlots[lifeCycle].front();
    Slot& coldSlot = coldSlots[lifeCycle].front();
    PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kGpuLevel, lifeCycle);
    size_t const hotPageBytes = storage.slotSize(hotPoolGroup).at(PoolIndex{0});
    MemAddress const hotAddress
        = std::get<MemAddress>(storage.slotAddress(kGpuLevel, hotPoolGroup, hotSlot.slotId(), PoolIndex{0}));
    constexpr uint8_t kPattern = 0xA7;
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(hotAddress), kPattern, hotPageBytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    storage.copySlotData(lifeCycle, coldLevel, kGpuLevel, coldSlot.slotId(), hotSlot.slotId(), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(hotAddress), 0, hotPageBytes), cudaSuccess);
    storage.copySlotData(lifeCycle, kGpuLevel, coldLevel, hotSlot.slotId(), coldSlot.slotId(), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t firstByte = 0;
    uint8_t lastByte = 0;
    ASSERT_EQ(
        cudaMemcpy(&firstByte, reinterpret_cast<void const*>(hotAddress), 1, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(&lastByte, reinterpret_cast<void const*>(hotAddress + hotPageBytes - 1), 1, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(firstByte, kPattern);
    EXPECT_EQ(lastByte, kPattern);

    storage.releaseSlot(lifeCycle, coldLevel, std::move(coldSlot));
    storage.releaseSlot(lifeCycle, kGpuLevel, std::move(hotSlot));
}

TEST(KvCacheManagerV2StatsTest, RejectsBatchingClassWithDifferentColdPageSizes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<SplitColdPageCodec>(true);
    auto* codecPtr = codec.get();

    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeSplitColdGroupingConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        std::invalid_argument);

    EXPECT_EQ(codec.get(), codecPtr);
}

TEST(KvCacheManagerV2StatsTest, RejectsBatchingClassWithDifferentIndexLocations)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<MixedIndexLocationColdPageCodec>();
    auto* codecPtr = codec.get();

    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeSplitColdGroupingConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        std::invalid_argument);

    EXPECT_EQ(codec.get(), codecPtr);
}

TEST(KvCacheManagerV2StatsTest, ColdGroupingIsIndependentOfHotGrouping)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<SplitColdPageCodec>();
    auto manager = std::make_shared<KvCacheManager>(makeSplitColdGroupingConfig(), nullptr, std::move(codec));
    EXPECT_FALSE(codec);

    StorageManager const& storage = manager->storage();
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    EXPECT_EQ(storage.numPoolGroups(kGpuLevel), PoolGroupIndex{1});
    EXPECT_EQ(storage.numPoolGroups(CacheLevel{1}), PoolGroupIndex{2});

    PoolGroupIndex const hotGroup0 = storage.getPoolGroupIndex(kGpuLevel, LifeCycleId{0});
    PoolGroupIndex const hotGroup1 = storage.getPoolGroupIndex(kGpuLevel, LifeCycleId{1});
    EXPECT_EQ(hotGroup0, hotGroup1);

    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        PoolGroupIndex const coldGroup = storage.getPoolGroupIndex(CacheLevel{1}, lifeCycle);
        EXPECT_NE(coldGroup, storage.getPoolGroupIndex(CacheLevel{1}, LifeCycleId{1 - lifeCycle.value()}));
        EXPECT_EQ(storage.numPools(CacheLevel{1}, coldGroup), PoolIndex{1});
        auto const coldSlotSizes = storage.slotSize(CacheLevel{1}, coldGroup);
        ASSERT_EQ(coldSlotSizes.size(), PoolIndex{1});
        size_t const expectedBytes = lifeCycle == LifeCycleId{0} ? 1024 : 2048;
        EXPECT_EQ(coldSlotSizes.at(PoolIndex{0}), expectedBytes);
    }
}

TEST(KvCacheManagerV2StatsTest, StatsDeltaArithmetic)
{
    KVCacheStatsDelta stats{4, 3, 2, 1};
    KVCacheStatsDelta const delta{1, 2, 3, 4};
    stats.add(delta);
    EXPECT_EQ(stats.allocTotalBlocks, 5);
    EXPECT_EQ(stats.allocNewBlocks, 5);
    EXPECT_EQ(stats.reusedBlocks, 5);
    EXPECT_EQ(stats.missedBlocks, 5);

    KVCacheStatsDelta const copy = stats.copy();
    stats.subtract(delta);
    EXPECT_EQ(stats.allocTotalBlocks, 4);
    EXPECT_EQ(copy.allocTotalBlocks, 5);
    stats.clear();
    EXPECT_TRUE(stats.empty());
}

TEST(KvCacheManagerV2StatsTest, IterationStatsDeltaArithmeticAndHitRate)
{
    KVCacheIterationStatsDelta stats;
    stats.iterReusedBlocks = 3;
    stats.iterFullReusedBlocks = 2;
    stats.iterPartialReusedBlocks = 1;
    stats.iterMissedBlocks = 1;
    stats.iterOnboardBytes = 1024;
    EXPECT_DOUBLE_EQ(stats.iterCacheHitRate(), 0.75);

    KVCacheIterationStatsDelta delta = stats.copy();
    stats.add(delta);
    EXPECT_EQ(stats.iterReusedBlocks, 6);
    EXPECT_EQ(stats.iterOnboardBytes, 2048);
    stats.subtract(delta);
    EXPECT_EQ(stats.iterReusedBlocks, 3);
    stats.clear();
    EXPECT_TRUE(stats.empty());
    EXPECT_DOUBLE_EQ(stats.iterCacheHitRate(), 0.0);
}

TEST(KvCacheManagerV2StatsTest, PendingAllocationRangesAreReversibleAndScoped)
{
    PendingStats pending;
    EXPECT_TRUE(pending.recordAllocationRange(
        LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{3}, /*beamWidth=*/2, /*countAsMissed=*/true));
    EXPECT_TRUE(pending.recordAllocationRange(LifeCycleId{1}, BlockOrdinal{3}, BlockOrdinal{5},
        /*beamWidth=*/1, /*countAsMissed=*/false, /*countAsGeneration=*/true));

    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 8);
    EXPECT_EQ(pending.globalStats().allocNewBlocks, 8);
    EXPECT_EQ(pending.globalStats().missedBlocks, 6);
    EXPECT_EQ(pending.requestStats().allocTotalBlocks, 8);
    ASSERT_EQ(pending.iterationStatsByLifeCycle().size(), 2);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{0}).iterMissedBlocks, 6);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{1}).iterGenAllocBlocks, 2);

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{2}, BlockOrdinal{5}));
    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 4);
    EXPECT_EQ(pending.globalStats().missedBlocks, 4);
    ASSERT_EQ(pending.iterationStatsByLifeCycle().size(), 1);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{0}).iterAllocTotalBlocks, 4);

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{0}, BlockOrdinal{2}));
    EXPECT_TRUE(pending.empty());
}

TEST(KvCacheManagerV2StatsTest, PendingReuseSurvivesAllocationRollbackUntilClear)
{
    PendingStats pending;
    EXPECT_TRUE(pending.recordAllocationRange(
        LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{1}, /*beamWidth=*/1, /*countAsMissed=*/true));
    EXPECT_TRUE(pending.recordReuse(LifeCycleId{0}, /*fullReusedBlocks=*/2, /*partialReusedBlocks=*/1));

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{0}, BlockOrdinal{1}));
    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 0);
    EXPECT_EQ(pending.globalStats().reusedBlocks, 3);
    auto const& iteration = pending.iterationStatsByLifeCycle().at(LifeCycleId{0});
    EXPECT_EQ(iteration.iterReusedBlocks, 3);
    EXPECT_EQ(iteration.iterFullReusedBlocks, 2);
    EXPECT_EQ(iteration.iterPartialReusedBlocks, 1);

    pending.clear();
    EXPECT_TRUE(pending.empty());
}

TEST(KvCacheManagerV2StatsTest, ManagerCommitResetAndRequestIdTracking)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());

    KVCacheStatsDelta globalStats{4, 3, 2, 1};
    KVCacheIterationStatsDelta iterationStats;
    iterationStats.iterAllocTotalBlocks = 4;
    iterationStats.iterReusedBlocks = 2;
    manager->commitStats(globalStats, {{LifeCycleId{0}, iterationStats}});

    EXPECT_EQ(manager->getCommittedStats().allocTotalBlocks, 4);
    auto firstIteration = manager->getAndResetIterationStats();
    ASSERT_EQ(firstIteration.size(), 1);
    EXPECT_EQ(firstIteration.at(LifeCycleId{0}).iterReusedBlocks, 2);
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());

    manager->markStatsDirty(11);
    manager->markStatsDirty(std::nullopt);
    EXPECT_EQ(manager->getDirtyStatsKvCacheIds().count(11), 1);
    manager->markStatsExcluded(11);
    EXPECT_TRUE(manager->isStatsExcluded(11));
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    manager->clearStatsExcluded(11);
    EXPECT_FALSE(manager->isStatsExcluded(11));

    auto cache = manager->createKvCache({}, {}, 17, {}, 8);
    manager->markStatsDirty(17);
    EXPECT_TRUE(cache->commitPendingStats().empty());
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    cache->close();

    RequestIdType const cudaGraphDummyRequestId = std::numeric_limits<RequestIdType>::max();
    auto dummyCache = manager->createKvCache({}, {}, cudaGraphDummyRequestId);
    ASSERT_TRUE(dummyCache->id.has_value());
    EXPECT_EQ(*dummyCache->id, cudaGraphDummyRequestId);
    manager->markStatsDirty(cudaGraphDummyRequestId);
    EXPECT_EQ(manager->getDirtyStatsKvCacheIds(), std::unordered_set<RequestIdType>{cudaGraphDummyRequestId});
    manager->markStatsExcluded(cudaGraphDummyRequestId);
    EXPECT_TRUE(manager->isStatsExcluded(cudaGraphDummyRequestId));
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    dummyCache->close();
}

TEST(KvCacheManagerV2StatsTest, DisabledStatsSuppressManagerCommit)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig(false));
    manager->commitStats(KVCacheStatsDelta{4, 3, 2, 1});
    EXPECT_TRUE(manager->getCommittedStats().empty());
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());
}

TEST(KvCacheManagerV2StatsTest, PeakBlockStatsResetStartsNextIntervalFromCurrentSnapshot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
    manager->commitStats({});

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<SharedPtr<Page>> pages;
    NodeBase* previous = &root;
    int token = 0;
    for (auto& slot : gpuSlots[lifeCycle])
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < manager->tokensPerBlock(); ++i)
        {
            tokens.emplace_back(TokenId{token++});
        }
        auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }
    manager->commitStats({});

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hostSlots = storage.newSlots(CacheLevel{1}, oneSlot);
    manager->commitStats({});
    storage.releaseSlot(lifeCycle, CacheLevel{1}, std::move(hostSlots[lifeCycle].front()));
    manager->clearReusableBlocks();
    pages.clear();

    auto primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    auto secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    ASSERT_EQ(primaryPeak.size(), PoolGroupIndex{1});
    ASSERT_EQ(secondaryPeak.size(), PoolGroupIndex{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 1);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    auto nextIntervalSlots = storage.newSlots(kGpuLevel, oneSlot);
    manager->commitStats({});
    storage.releaseSlot(lifeCycle, kGpuLevel, std::move(nextIntervalSlots[lifeCycle].front()));
    primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 1);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 0);
}

TEST(KvCacheManagerV2StatsTest, MigrationAndLastTierDropRecordersReceiveExactPages)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    ASSERT_EQ(storage.getStatistics(kGpuLevel).total, 2);
    ASSERT_EQ(storage.getStatistics(CacheLevel{1}).total, 2);

    int offloaded = 0;
    int onboarded = 0;
    int dropped = 0;
    MigrationRecorder const migrationRecorder
        = [&](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
              CacheLevel dstLevel)
    {
        EXPECT_EQ(pages.size(), slots.size());
        if (srcLevel == kGpuLevel && dstLevel == CacheLevel{1})
        {
            offloaded += static_cast<int>(pages.size());
        }
        else if (srcLevel == CacheLevel{1} && dstLevel == kGpuLevel)
        {
            onboarded += static_cast<int>(pages.size());
        }
    };
    DropRecorder const dropRecorder = [&](std::vector<SharedPtr<Page>> const& pages, CacheLevel level)
    {
        EXPECT_EQ(level, CacheLevel{1});
        dropped += static_cast<int>(pages.size());
    };

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
            {
                tokens.emplace_back(TokenId{tokenBase++});
            }
            auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
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
    auto firstPages = makeCommittedPages(std::move(initialSlots[lifeCycle]));
    PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kGpuLevel, lifeCycle);
    size_t const hotPageBytes = storage.slotSize(hotPoolGroup).at(PoolIndex{0});
    std::array<uint8_t, 2> const pagePatterns{0x3C, 0xA7};
    for (size_t index = 0; index < firstPages.size(); ++index)
    {
        MemAddress const address = std::get<MemAddress>(
            storage.slotAddress(kGpuLevel, hotPoolGroup, firstPages[index]->slotId(), PoolIndex{0}));
        ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(address), pagePatterns[index], hotPageBytes), cudaSuccess);
    }

    auto temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 2);
    EXPECT_EQ(onboarded, 0);
    EXPECT_EQ(dropped, 0);
    for (auto& slot : temporarySlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }

    auto cache = manager->createKvCache();
    std::vector<BatchedLockTarget> targets;
    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{2}; ++ordinal)
    {
        auto const& page = firstPages[toSizeT(ordinal)];
        ASSERT_TRUE(page->scheduledForEviction());
        storage.excludeFromEviction(*page);
        targets.push_back({page, kDefaultBeamIndex, ordinal, lifeCycle});
    }
    storage.batchedMigrateToGpu(targets, *cache, migrationRecorder);
    EXPECT_EQ(onboarded, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (size_t index = 0; index < firstPages.size(); ++index)
    {
        MemAddress const address = std::get<MemAddress>(
            storage.slotAddress(kGpuLevel, hotPoolGroup, firstPages[index]->slotId(), PoolIndex{0}));
        uint8_t firstByte = 0;
        uint8_t lastByte = 0;
        ASSERT_EQ(
            cudaMemcpy(&firstByte, reinterpret_cast<void const*>(address), 1, cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(&lastByte, reinterpret_cast<void const*>(address + hotPageBytes - 1), 1, cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_EQ(firstByte, pagePatterns[index]);
        EXPECT_EQ(lastByte, pagePatterns[index]);
    }
    for (auto const& page : firstPages)
    {
        storage.scheduleForEviction(*page);
    }

    temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 4);
    auto secondPages = makeCommittedPages(std::move(temporarySlots[lifeCycle]));
    (void) secondPages;
    firstPages.clear();
    targets.clear();

    auto finalSlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 6);
    EXPECT_EQ(onboarded, 2);
    EXPECT_EQ(dropped, 2);
    for (auto& slot : finalSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
    cache->close();
}

} // namespace
