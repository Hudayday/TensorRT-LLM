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
#include <map>
#include <memory>
#include <stdexcept>
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
        std::vector<PageIndexPair> pageIndices;
        cudaStream_t stream;
    };

    bool configure(PoolGroupDesc const&) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId layerGroup) const noexcept override
    {
        auto const found = coldPageBytes.find(layerGroup);
        return found == coldPageBytes.end() ? defaultColdPageBytes : found->second;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroup) const noexcept override
    {
        auto const found = batchingLayerGroups.find(layerGroup);
        return found == batchingLayerGroups.end() ? layerGroup : found->second;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId layerGroup, void*, PageIndexPair const* pageIndices, size_t count,
        cudaStream_t stream) noexcept override
    {
        calls.push_back(Call{true, layerGroup, {pageIndices, pageIndices + count}, stream});
        return submit;
    }

    bool decode(LayerGroupId layerGroup, void const*, PageIndexPair const* pageIndices, size_t count,
        cudaStream_t stream) noexcept override
    {
        calls.push_back(Call{false, layerGroup, {pageIndices, pageIndices + count}, stream});
        return submit;
    }

    bool submit = true;
    size_t defaultColdPageBytes = 1U << 20;
    std::map<LayerGroupId, size_t> coldPageBytes;
    std::map<LayerGroupId, LayerGroupId> batchingLayerGroups;
    std::vector<Call> calls;
};

KVCacheManagerConfig makeTieredConfig(bool enableStats = false)
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
    config.enableStats = enableStats;
    return config;
}

KVCacheManagerConfig makeDiskTieredConfig()
{
    auto config = makeTieredConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    return config;
}

KVCacheManagerConfig makeHybridTieredConfig(bool matchingRawLayouts = false, bool includeDisk = false)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{8 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{8 << 20});
    if (includeDisk)
    {
        config.cacheTiers.emplace_back(DiskCacheTierConfig{8 << 20, "/tmp"});
    }
    config.commitMinSnapshot = true;

    AttentionLayerConfig attention;
    attention.layerId = 0;
    attention.buffers.push_back(BufferConfig{"key", 1 << 20, std::nullopt});
    if (!matchingRawLayouts)
        attention.buffers.push_back(BufferConfig{"value", 1 << 20, std::nullopt});
    config.layers.emplace_back(std::move(attention));

    SsmLayerConfig ssm;
    ssm.layerId = 1;
    ssm.buffers.push_back(BufferConfig{"ssm_state", 1 << 20, std::nullopt});
    if (!matchingRawLayouts)
        ssm.buffers.push_back(BufferConfig{"conv_state", 256 << 10, std::nullopt});
    config.layers.emplace_back(std::move(ssm));
    return config;
}

KVCacheManagerConfig makeTwoLifeCycleTieredConfig()
{
    auto config = makeTieredConfig();
    AttentionLayerConfig slidingWindowLayer;
    slidingWindowLayer.layerId = 1;
    slidingWindowLayer.slidingWindowSize = 4;
    slidingWindowLayer.buffers.push_back(BufferConfig{"key", 1 << 20, std::nullopt});
    slidingWindowLayer.buffers.push_back(BufferConfig{"value", 1 << 20, std::nullopt});
    config.layers.emplace_back(std::move(slidingWindowLayer));
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
    EXPECT_EQ(codec->calls.front().pageIndices.size(), 2);
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
    EXPECT_EQ(codec->calls.back().pageIndices.size(), 2);
    EXPECT_TRUE(
        std::all_of(pages.begin(), pages.end(), [](auto const& page) { return page->cacheLevel == kGpuLevel; }));

    cache->close();
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, IterationStatsCountCompactTransferredAndDroppedBytes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(/*enableStats=*/true));
    auto codec = std::make_shared<RecordingColdPageCodec>();
    manager->setColdPageCodec(codec);

    auto const& storage = manager->storage();
    PoolGroupIndex const poolGroup{0};
    LifeCycleId const lifeCycle{0};
    auto const rawPageBytes = storage.slotSize(poolGroup, kGpuLevel).front();
    auto const coldPageBytes = storage.slotSize(poolGroup, CacheLevel{1}).front();
    ASSERT_EQ(rawPageBytes, 2U << 20);
    ASSERT_EQ(coldPageBytes, 1U << 20);

    auto storePrefix = [&](int tokenBase)
    {
        std::vector<TokenIdExt> tokens;
        for (int offset = 0; offset < 2 * manager->tokensPerBlock(); ++offset)
        {
            tokens.emplace_back(TokenId{tokenBase + offset});
        }
        auto cache = manager->createKvCache();
        if (!cache->resume(CUstream{}) || !cache->resize(static_cast<int>(tokens.size())))
        {
            throw std::runtime_error("Failed to allocate a test prefix");
        }
        cache->commit(tokens, /*isEnd=*/true);
        cache->close();
        return tokens;
    };

    storePrefix(0);
    manager->getAndResetIterationStats();
    auto const prefixB = storePrefix(100);

    auto stats = manager->getAndResetIterationStats();
    ASSERT_EQ(stats.count(lifeCycle), 1U);
    EXPECT_EQ(stats.at(lifeCycle).iterOffloadBlocks, 2);
    EXPECT_EQ(stats.at(lifeCycle).iterOffloadBytes, 2 * static_cast<int64_t>(coldPageBytes));

    // Fill all four compact Host Slots, then force another offload. KVCM must
    // drop the two least-recently-used cold Pages before storing the next two.
    storePrefix(200);
    manager->getAndResetIterationStats();
    storePrefix(300);
    stats = manager->getAndResetIterationStats();
    ASSERT_EQ(stats.count(lifeCycle), 1U);
    EXPECT_EQ(stats.at(lifeCycle).iterOffloadBlocks, 2);
    EXPECT_EQ(stats.at(lifeCycle).iterOffloadBytes, 2 * static_cast<int64_t>(coldPageBytes));
    EXPECT_EQ(stats.at(lifeCycle).iterHostDroppedBlocks, 2);
    EXPECT_EQ(stats.at(lifeCycle).iterHostDroppedBytes, 2 * static_cast<int64_t>(coldPageBytes));

    // Prefix B is still Host-resident. A reuse hit restores it to raw GPU
    // Slots, but the onboard metric counts the compact bytes read from Host.
    auto reused = manager->createKvCache({}, prefixB);
    ASSERT_TRUE(reused->resume(CUstream{}));
    stats = manager->getAndResetIterationStats();
    ASSERT_EQ(stats.count(lifeCycle), 1U);
    EXPECT_EQ(stats.at(lifeCycle).iterOnboardBlocks, 2);
    EXPECT_EQ(stats.at(lifeCycle).iterOnboardBytes, 2 * static_cast<int64_t>(coldPageBytes));

    reused->close();
    manager->clearReusableBlocks();
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, RoutesDirectGpuDiskMigrationThroughCodecStaging)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeDiskTieredConfig());
    auto codec = std::make_shared<RecordingColdPageCodec>();
    manager->setColdPageCodec(codec);

    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto gpuSlots = storage.newGpuSlots(oneSlot);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    auto block = addOrGetExistingBlock(&root, LifeCycleId{1}, {TokenIdExt{TokenId{7}}});
    auto gpuPage
        = makeShared<CommittedPage>(&storage, block, lifeCycle, kGpuLevel, /*numTokensInBlock=*/1, kPriorityDefault);
    gpuPage->setSlot(gpuSlots[lifeCycle].front());

    auto diskSlot = storage.clonePageToLevel(gpuPage, CacheLevel{2});
    ASSERT_EQ(codec->calls.size(), 1U);
    EXPECT_TRUE(codec->calls.front().encode);
    EXPECT_EQ(codec->calls.front().pageIndices.size(), 1U);
    EXPECT_EQ(codec->calls.front().pageIndices.front().dst, 0);

    // Model the Page after GPU->Disk publication, then exercise the direct
    // Disk->GPU route. The Disk Slot's ready event orders the staging read
    // after the preceding encode+write without an intermediate Host Page.
    auto diskPage = makeShared<CommittedPage>(
        &storage, block, lifeCycle, CacheLevel{2}, /*numTokensInBlock=*/1, kPriorityDefault);
    diskPage->setSlot(diskSlot);
    block->storage[lifeCycle] = diskPage.get();
    auto cache = manager->createKvCache();
    storage.batchedMigrateToGpu(
        {BatchedLockTarget{diskPage, kDefaultBeamIndex, BlockOrdinal{0}, lifeCycle}}, *cache, MigrationRecorder{});

    ASSERT_EQ(codec->calls.size(), 2U);
    EXPECT_FALSE(codec->calls.back().encode);
    EXPECT_EQ(codec->calls.back().pageIndices.size(), 1U);
    EXPECT_EQ(codec->calls.back().pageIndices.front().src, 0);
    EXPECT_EQ(diskPage->cacheLevel, kGpuLevel);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cache->close();
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, FailedDiskDecodeKeepsSourceMappingAndReclaimsGpuSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeDiskTieredConfig());
    auto codec = std::make_shared<RecordingColdPageCodec>();
    manager->setColdPageCodec(codec);

    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto gpuSlots = storage.newGpuSlots(oneSlot);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    auto block = addOrGetExistingBlock(&root, LifeCycleId{1}, {TokenIdExt{TokenId{7}}});
    auto gpuPage
        = makeShared<CommittedPage>(&storage, block, lifeCycle, kGpuLevel, /*numTokensInBlock=*/1, kPriorityDefault);
    gpuPage->setSlot(gpuSlots[lifeCycle].front());
    auto diskSlot = storage.clonePageToLevel(gpuPage, CacheLevel{2});
    auto diskPage = makeShared<CommittedPage>(
        &storage, block, lifeCycle, CacheLevel{2}, /*numTokensInBlock=*/1, kPriorityDefault);
    diskPage->setSlot(diskSlot);
    block->storage[lifeCycle] = diskPage.get();

    SlotId const sourceSlotId = diskPage->slotId();
    auto const freeGpuSlots = storage.getStatistics(kGpuLevel).free;
    codec->submit = false;
    auto cache = manager->createKvCache();
    EXPECT_THROW(storage.batchedMigrateToGpu(
                     {BatchedLockTarget{diskPage, kDefaultBeamIndex, BlockOrdinal{0}, lifeCycle}}, *cache, {}),
        std::runtime_error);

    EXPECT_EQ(diskPage->cacheLevel, CacheLevel{2});
    EXPECT_EQ(diskPage->slotId(), sourceSlotId);
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, freeGpuSlots);
    ASSERT_EQ(codec->calls.size(), 2U);
    EXPECT_FALSE(codec->calls.back().encode);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cache->close();
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, BatchesCodecEquivalentLifeCyclesInOneCall)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTwoLifeCycleTieredConfig());
    auto codec = std::make_shared<RecordingColdPageCodec>();
    codec->batchingLayerGroups.emplace(LayerGroupId{1}, LayerGroupId{0});
    manager->setColdPageCodec(codec);

    auto& storage = manager->storage();
    ASSERT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    ASSERT_EQ(storage.getPoolGroupIndex(LifeCycleId{0}), storage.getPoolGroupIndex(LifeCycleId{1}));

    TypedVec<LifeCycleId, SlotCount> oneSlotEach(LifeCycleId{2}, 1);
    auto slots = storage.newGpuSlots(oneSlotEach);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
        tokens.emplace_back(TokenId{token});
    auto block = addOrGetExistingBlock(&root, LifeCycleId{2}, std::move(tokens));

    std::vector<SharedPtr<Page>> pages;
    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slots[lifeCycle].front());
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
    }

    // The 4 MiB GPU PoolGroup has four 1 MiB Slots. Two are occupied above;
    // requesting four more leaves a two-Slot deficit and therefore evicts both
    // reusable Pages in one PoolGroup migration. The representative API lets
    // KVCM concatenate both lifecycle index arrays into one codec call.
    TypedVec<LifeCycleId, SlotCount> twoSlotsEach(LifeCycleId{2}, 2);
    auto replacementSlots = storage.newGpuSlots(twoSlotsEach);
    ASSERT_EQ(codec->calls.size(), 1);
    EXPECT_TRUE(codec->calls.front().encode);
    EXPECT_EQ(codec->calls.front().layerGroup, LayerGroupId{0});
    EXPECT_EQ(codec->calls.front().pageIndices.size(), 2);

    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        for (auto& slot : replacementSlots[lifeCycle])
            storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, PartialSnapshotCloneUsesCodecWithoutMovingSourcePage)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto codec = std::make_shared<RecordingColdPageCodec>();
    manager->setColdPageCodec(codec);

    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto gpuSlots = storage.newGpuSlots(oneSlot);

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    auto block = addOrGetExistingBlock(&root, LifeCycleId{1}, {TokenIdExt{TokenId{7}}});
    auto source
        = makeShared<CommittedPage>(&storage, block, lifeCycle, kGpuLevel, /*numTokensInBlock=*/1, kPriorityDefault);
    source->setSlot(gpuSlots[lifeCycle].front());
    SlotId const sourceSlotId = source->slotId();

    auto coldSlot = storage.clonePageToLevel(source, CacheLevel{1});

    ASSERT_EQ(codec->calls.size(), 1);
    EXPECT_TRUE(codec->calls.front().encode);
    ASSERT_EQ(codec->calls.front().pageIndices.size(), 1);
    EXPECT_EQ(codec->calls.front().pageIndices.front().src, sourceSlotId.value());
    EXPECT_EQ(codec->calls.front().pageIndices.front().dst, coldSlot.slotId().value());
    EXPECT_EQ(source->cacheLevel, kGpuLevel);
    EXPECT_EQ(source->slotId(), sourceSlotId);

    storage.releaseSlot(lifeCycle, CacheLevel{1}, std::move(coldSlot));
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, KeepsCodecDeclinedSsmPoolGroupRaw)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHybridTieredConfig(/*matchingRawLayouts=*/false,
        /*includeDisk=*/true));
    auto codec = std::make_shared<RecordingColdPageCodec>();
    codec->defaultColdPageBytes = 0;

    auto& storage = manager->storage();
    auto const attentionLifeCycle = manager->getLayerGroupId(LayerId{0});
    auto const ssmLifeCycle = manager->getLayerGroupId(LayerId{1});
    auto const attentionPoolGroup = storage.getPoolGroupIndex(attentionLifeCycle);
    auto const ssmPoolGroup = storage.getPoolGroupIndex(ssmLifeCycle);
    ASSERT_NE(attentionPoolGroup, ssmPoolGroup);
    codec->coldPageBytes.emplace(attentionLifeCycle, 512U << 10);

    manager->setColdPageCodec(codec);

    for (CacheLevel coldLevel{1}; coldLevel < storage.numCacheLevels(); ++coldLevel)
    {
        EXPECT_EQ(storage.numPools(attentionPoolGroup, coldLevel), PoolIndex{1});
        EXPECT_EQ(storage.slotSize(attentionPoolGroup, coldLevel).front(), 512U << 10);
        EXPECT_EQ(storage.numPools(ssmPoolGroup, coldLevel), storage.numPools(ssmPoolGroup, kGpuLevel));
        EXPECT_EQ(storage.slotSize(ssmPoolGroup, coldLevel).raw(), storage.slotSize(ssmPoolGroup, kGpuLevel).raw());
    }

    // Target-ratio sampling uses the actual representation at each level.
    // Host and Disk share the cold layout; GPU keeps the raw runtime layout.
    auto const gpuRatio = storage.ratioFromLength(kGpuLevel, /*tokensPerBlock=*/4, /*historyLength=*/4, /*capacity=*/4);
    auto const hostRatio
        = storage.ratioFromLength(CacheLevel{1}, /*tokensPerBlock=*/4, /*historyLength=*/4, /*capacity=*/4);
    auto const diskRatio
        = storage.ratioFromLength(CacheLevel{2}, /*tokensPerBlock=*/4, /*historyLength=*/4, /*capacity=*/4);
    ASSERT_EQ(gpuRatio.size(), PoolGroupIndex{2});
    ASSERT_EQ(hostRatio.size(), PoolGroupIndex{2});
    ASSERT_EQ(diskRatio.size(), PoolGroupIndex{2});
    EXPECT_NE(gpuRatio.at(attentionPoolGroup), hostRatio.at(attentionPoolGroup));
    EXPECT_FLOAT_EQ(hostRatio.at(attentionPoolGroup), diskRatio.at(attentionPoolGroup));
    EXPECT_FLOAT_EQ(hostRatio.at(ssmPoolGroup), diskRatio.at(ssmPoolGroup));

    TypedVec<LifeCycleId, SlotCount> oneSlot(storage.numLifeCycles(), SlotCount{0});
    oneSlot[ssmLifeCycle] = SlotCount{1};
    auto gpuSlots = storage.newGpuSlots(oneSlot);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    auto block = addOrGetExistingBlock(&root, storage.numLifeCycles(), {TokenIdExt{TokenId{7}}});
    auto ssmPage = makeShared<CommittedPage>(&storage, block, ssmLifeCycle, kGpuLevel,
        /*numTokensInBlock=*/1, kPriorityDefault);
    ssmPage->setSlot(gpuSlots[ssmLifeCycle].front());
    block->storage[ssmLifeCycle] = ssmPage.get();

    auto rawHostSlot = storage.clonePageToLevel(ssmPage, CacheLevel{1});
    auto rawDiskSlot = storage.clonePageToLevel(ssmPage, CacheLevel{2});
    EXPECT_TRUE(codec->calls.empty());
    EXPECT_EQ(ssmPage->cacheLevel, kGpuLevel);
    storage.releaseSlot(ssmLifeCycle, CacheLevel{1}, std::move(rawHostSlot));
    storage.releaseSlot(ssmLifeCycle, CacheLevel{2}, std::move(rawDiskSlot));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    manager->setColdPageCodec(nullptr);
}

TEST(KvCacheManagerV2ColdPageCodecTest, RejectsMixedCodecCoverageInsideOnePoolGroup)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHybridTieredConfig(/*matchingRawLayouts=*/true));
    auto codec = std::make_shared<RecordingColdPageCodec>();
    codec->defaultColdPageBytes = 0;

    auto& storage = manager->storage();
    auto const attentionLifeCycle = manager->getLayerGroupId(LayerId{0});
    auto const ssmLifeCycle = manager->getLayerGroupId(LayerId{1});
    auto const poolGroup = storage.getPoolGroupIndex(attentionLifeCycle);
    ASSERT_EQ(poolGroup, storage.getPoolGroupIndex(ssmLifeCycle));
    auto const originalHostSlotSizes = storage.slotSize(poolGroup, CacheLevel{1}).raw();
    codec->coldPageBytes.emplace(attentionLifeCycle, 512U << 10);

    EXPECT_THROW(manager->setColdPageCodec(codec), std::runtime_error);
    EXPECT_EQ(storage.slotSize(poolGroup, CacheLevel{1}).raw(), originalHostSlotSizes);
}

} // namespace
