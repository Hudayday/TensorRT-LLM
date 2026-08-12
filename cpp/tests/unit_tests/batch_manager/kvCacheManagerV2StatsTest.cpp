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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

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

KVCacheManagerConfig makePartialSnapshotTieredConfig()
{
    auto config = makeTieredConfig();
    config.commitMinSnapshot = true;
    return config;
}

KVCacheManagerConfig makeHostDiskTieredConfig()
{
    auto config = makeTieredConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
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

KVCacheManagerConfig makeTwoLifeCycleRollbackConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});

    for (int layerId = 0; layerId < 2; ++layerId)
    {
        AttentionLayerConfig layer;
        layer.layerId = layerId;
        layer.slidingWindowSize = layerId == 0 ? 128 : 256;
        layer.buffers.push_back(BufferConfig{"key", 2 << 20, std::nullopt});
        config.layers.emplace_back(std::move(layer));
    }
    return config;
}

KVCacheManagerConfig makeRollbackOrderConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 1 << 20, std::nullopt});
    config.layers.emplace_back(std::move(layer));
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

class RuntimeRejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    enum class Direction
    {
        kEncode,
        kDecode,
    };

    explicit RuntimeRejectingColdPageCodec(Direction direction)
        : mDirection(direction)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return mDirection != Direction::kEncode;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return mDirection != Direction::kDecode;
    }

private:
    Direction mDirection;
};

class AsyncRejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    AsyncRejectingColdPageCodec()
    {
        if (cudaEventCreateWithFlags(&mCallbackDone, cudaEventDisableTiming) != cudaSuccess)
        {
            throw std::runtime_error("failed to create async-rejection test event");
        }
    }

    ~AsyncRejectingColdPageCodec() override
    {
        // A fatal test assertion may unwind before the test explicitly opens
        // the callback gate. Keep the fake itself failure-safe: release and
        // wait before destroying the atomics read by the Host callback.
        releaseCallback();
        if (mSubmitted.load(std::memory_order_acquire))
        {
            static_cast<void>(cudaEventSynchronize(mCallbackDone));
        }
        static_cast<void>(cudaEventDestroy(mCallbackDone));
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t stream) noexcept override
    {
        cudaError_t const status = cudaLaunchHostFunc(
            stream,
            [](void* opaque)
            {
                auto& self = *static_cast<AsyncRejectingColdPageCodec*>(opaque);
                while (!self.mReleaseCallback.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
            },
            this);
        mSubmitted.store(status == cudaSuccess, std::memory_order_release);
        if (status == cudaSuccess && cudaEventRecord(mCallbackDone, stream) != cudaSuccess)
        {
            // If the completion marker itself cannot be submitted, drain here
            // so the destructor never races the callback through an untracked
            // stream. The codec still rejects the migration below.
            releaseCallback();
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        // The interface permits rejecting after work was submitted. KVCM must
        // fence that work before recycling either source or destination Slot.
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

    bool submitted() const noexcept
    {
        return mSubmitted.load(std::memory_order_acquire);
    }

    void releaseCallback() noexcept
    {
        mReleaseCallback.store(true, std::memory_order_release);
    }

private:
    cudaEvent_t mCallbackDone = nullptr;
    std::atomic_bool mSubmitted{false};
    std::atomic_bool mReleaseCallback{false};
};

class ThrowingCacheLevelEventSink final : public EventSink
{
public:
    void addStoredBlock(Block const&) override {}

    void addStoredLifeCycle(Block const&, LifeCycleId) override {}

    void addRemovedBlock(Digest const&) override {}

    void addRemovedLifeCycle(Digest const&, LifeCycleId) override {}

    void addCacheLevelUpdated(Digest const&, CacheLevel, CacheLevel, LifeCycleId) override
    {
        ++mCalls;
        throw std::runtime_error("synthetic cache-level event failure");
    }

    int calls() const noexcept
    {
        return mCalls;
    }

private:
    int mCalls = 0;
};

class SameRepresentationGuardColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept override
    {
        mQueriedInvalidLayerGroup |= layerGroupId == LayerGroupId{-1};
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        ++mTransformCalls;
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        ++mTransformCalls;
        return false;
    }

    bool queriedInvalidLayerGroup() const noexcept
    {
        return mQueriedInvalidLayerGroup;
    }

    int transformCalls() const noexcept
    {
        return mTransformCalls;
    }

private:
    mutable bool mQueriedInvalidLayerGroup = false;
    int mTransformCalls = 0;
};

class RejectSecondEncodeColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return ++mEncodeCalls != 2;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

private:
    int mEncodeCalls = 0;
};

class RejectFirstEncodeColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return ++mEncodeCalls != 1;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

private:
    int mEncodeCalls = 0;
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

TEST(KvCacheManagerV2StatsTest, GpuStorageConstructorFailureCleansUpExistingPoolGroups)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    SlotDesc validDesc;
    SlotDescVariant validVariant;
    validVariant.lifeCycleId = LifeCycleId{0};
    validVariant.coalescedBuffers.push_back(CoalescedBuffer{4096, {BufferId{0, "key"}}});
    validDesc.variants.push_back(std::move(validVariant));

    // The second PoolGroup has no Pool. Its construction fails only after the
    // first PoolGroup has created a VirtMem mapping that borrows the physical
    // allocator owned by GpuCacheLevelStorage.
    SlotDesc invalidDesc;
    SlotDescVariant invalidVariant;
    invalidVariant.lifeCycleId = LifeCycleId{1};
    invalidDesc.variants.push_back(std::move(invalidVariant));

    TypedVec<PoolGroupIndex, SlotDesc> slotDescs;
    slotDescs.push_back(std::move(validDesc));
    slotDescs.push_back(std::move(invalidDesc));
    TypedVec<PoolGroupIndex, SlotCount> slotCounts;
    slotCounts.push_back(SlotCount{1});
    slotCounts.push_back(SlotCount{1});

    // Repeat the constructor-unwind path: the regression was a dangling
    // allocator access during cleanup, not merely the expected validation
    // exception from the invalid second PoolGroup.
    for (int attempt = 0; attempt < 4; ++attempt)
    {
        EXPECT_ANY_THROW({ GpuCacheLevelStorage storage(slotDescs, slotCounts, 4 << 20); });
    }
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

    // The cold-page codec rejects non-empty work without a stream. Exercise
    // the same asynchronous contract used by StorageManager migrations.
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    CUstream const driverStream = reinterpret_cast<CUstream>(stream);
    storage.copySlotData(lifeCycle, coldLevel, kGpuLevel, coldSlot.slotId(), hotSlot.slotId(), driverStream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(hotAddress), 0, hotPageBytes), cudaSuccess);
    storage.copySlotData(lifeCycle, kGpuLevel, coldLevel, hotSlot.slotId(), coldSlot.slotId(), driverStream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

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
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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

TEST(KvCacheManagerV2StatsTest, EncodeRejectionKeepsSourcePagesAndReleasesColdSlots)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<RuntimeRejectingColdPageCodec>(RuntimeRejectingColdPageCodec::Direction::kEncode);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<SharedPtr<Page>> pages;
    NodeBase* previous = &root;
    int token = 0;
    for (auto& slot : gpuSlots[lifeCycle])
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < manager->tokensPerBlock(); ++i)
            tokens.emplace_back(TokenId{token++});
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }

    std::array<SlotId, 2> const sourceSlots{pages[0]->slotId(), pages[1]->slotId()};
    auto const gpuBefore = storage.getStatistics(kGpuLevel);
    auto const hostBefore = storage.getStatistics(hostLevel);
    int migrations = 0;
    MigrationRecorder recorder = [&](auto const&, auto const&, CacheLevel, CacheLevel) { ++migrations; };

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    EXPECT_THROW(storage.newGpuSlots(oneSlot, recorder), LogicError);

    EXPECT_EQ(migrations, 0);
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, gpuBefore.free);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBefore.free);
    for (std::size_t i = 0; i < pages.size(); ++i)
    {
        EXPECT_EQ(pages[i]->cacheLevel, kGpuLevel);
        EXPECT_EQ(pages[i]->slotId(), sourceSlots[i]);
        EXPECT_TRUE(pages[i]->scheduledForEviction());
    }
}

TEST(KvCacheManagerV2StatsTest, AsyncEncodeRejectionFencesRecycledColdSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>();
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    // Occupy all but one Host Slot. The failed migration must therefore return
    // exactly its pending destination Slot on the next allocation.
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    SlotCount const hostBlockers = storage.getStatistics(hostLevel).total - 1;
    ASSERT_GE(hostBlockers, 0);
    SlotId const expectedRolledBackSlot{hostBlockers};
    TypedVec<LifeCycleId, SlotCount> hostBlockerCount(LifeCycleId{1}, hostBlockers);
    auto occupiedHostSlots = storage.newSlots(hostLevel, hostBlockerCount);

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
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
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }

    EXPECT_THROW(storage.newGpuSlots(oneSlot), LogicError);
    ASSERT_TRUE(codecPtr->submitted());
    for (auto const& page : pages)
    {
        EXPECT_EQ(page->cacheLevel, kGpuLevel);
        EXPECT_TRUE(page->scheduledForEviction());
    }

    // No inactive Host Slot remains, so this is the destination just rolled
    // back. Its event must still fence the callback enqueued by the rejecting
    // codec; otherwise the allocator could hand out memory still being used.
    auto recycledHostSlot = storage.newSlots(hostLevel, oneSlot);
    ASSERT_EQ(recycledHostSlot[lifeCycle].size(), 1);
    EXPECT_EQ(recycledHostSlot[lifeCycle].front().slotId(), expectedRolledBackSlot);
    EXPECT_FALSE(recycledHostSlot[lifeCycle].front().readyEvent.isClosed());
    codecPtr->releaseCallback();
    recycledHostSlot[lifeCycle].front().readyEvent.synchronize();

    storage.releaseSlot(lifeCycle, hostLevel, std::move(recycledHostSlot[lifeCycle].front()));
    for (auto& slot : occupiedHostSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, hostLevel, std::move(slot));
    }
}

TEST(KvCacheManagerV2StatsTest, PartialSnapshotCodecRejectionFencesAndReleasesColdCloneSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>();
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makePartialSnapshotTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(reinterpret_cast<CUstream>(stream)));
    cache->setCapacity(manager->tokensPerBlock());
    ASSERT_EQ(cache->blocks().size(), BlockOrdinal{1});
    auto sourcePage = blockPageGetPage(cache->blocks().front().pages[kDefaultBeamIndex][lifeCycle]);
    ASSERT_TRUE(sourcePage);
    SlotId const sourceSlot = sourcePage->slotId();

    // The partial snapshot first tries a same-level clone. Occupying every
    // remaining GPU Slot forces the only cold-codec path outside
    // _batchedMigrate: GPU source Page -> Host tree-snapshot Page.
    SlotCount const gpuBlockers = storage.getStatistics(kGpuLevel).total - 1;
    ASSERT_GE(gpuBlockers, 0);
    auto occupiedGpuSlots = storage.newSlotsForPoolGroup(kGpuLevel, PoolGroupIndex{0}, gpuBlockers);

    SlotCount const hostBlockers = storage.getStatistics(hostLevel).total - 1;
    ASSERT_GE(hostBlockers, 0);
    SlotId const expectedRolledBackSlot{hostBlockers};
    TypedVec<LifeCycleId, SlotCount> hostBlockerCount(LifeCycleId{1}, hostBlockers);
    auto occupiedHostSlots = storage.newSlots(hostLevel, hostBlockerCount);
    auto const hostBefore = storage.getStatistics(hostLevel);

    std::vector<TokenIdExt> partialTokens;
    partialTokens.emplace_back(TokenId{7});
    EXPECT_THROW(cache->commit(toSpan(partialTokens), /*isEnd=*/false), LogicError);
    ASSERT_TRUE(codecPtr->submitted());
    EXPECT_EQ(sourcePage->cacheLevel, kGpuLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlot);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBefore.free);

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto recycledHostSlot = storage.newSlots(hostLevel, oneSlot);
    ASSERT_EQ(recycledHostSlot[lifeCycle].size(), 1);
    EXPECT_EQ(recycledHostSlot[lifeCycle].front().slotId(), expectedRolledBackSlot);
    EXPECT_FALSE(recycledHostSlot[lifeCycle].front().readyEvent.isClosed());
    codecPtr->releaseCallback();
    recycledHostSlot[lifeCycle].front().readyEvent.synchronize();

    storage.releaseSlot(lifeCycle, hostLevel, std::move(recycledHostSlot[lifeCycle].front()));
    for (auto& slot : occupiedHostSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, hostLevel, std::move(slot));
    }
    for (auto& slot : occupiedGpuSlots)
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
    cache->close();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StatsTest, RecorderFailureFencesSubmittedMigrationBeforeRollback)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
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
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }

    SlotId const firstSourceSlot = pages.front()->slotId();
    auto const gpuBefore = storage.getStatistics(kGpuLevel);
    auto const hostBefore = storage.getStatistics(hostLevel);
    SharedPtr<Page> submittedPage;
    bool recorderSawBothFences = false;
    MigrationRecorder recorder = [&](auto const& migrated, auto const& destinations, CacheLevel, CacheLevel)
    {
        ASSERT_EQ(migrated.size(), 1);
        ASSERT_EQ(destinations.size(), 1);
        submittedPage = migrated.front();
        recorderSawBothFences = !submittedPage->readyEvent.isClosed() && !destinations.front().readyEvent.isClosed();
        throw std::runtime_error("synthetic recorder failure");
    };

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    EXPECT_THROW(storage.newGpuSlots(oneSlot, recorder), std::runtime_error);

    ASSERT_TRUE(submittedPage);
    EXPECT_EQ(submittedPage.get(), pages.front().get());
    EXPECT_EQ(submittedPage->cacheLevel, kGpuLevel);
    EXPECT_EQ(submittedPage->slotId(), firstSourceSlot);
    EXPECT_TRUE(recorderSawBothFences);
    // Releasing the rejected destination Slot may already have observed the
    // shared event complete and closed every copy. synchronize() is valid in
    // either state and proves rollback left no unfinished owner-visible work.
    submittedPage->readyEvent.synchronize();
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, gpuBefore.free);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBefore.free);
    for (auto const& page : pages)
    {
        EXPECT_TRUE(page->scheduledForEviction());
    }
}

TEST(KvCacheManagerV2StatsTest, CacheLevelEventFailureKeepsPublishedPageAndRemainingSlotsConsistent)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto eventSink = std::make_shared<ThrowingCacheLevelEventSink>();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), eventSink);
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
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
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }

    auto const gpuBefore = storage.getStatistics(kGpuLevel);
    auto const hostBefore = storage.getStatistics(hostLevel);
    // Requesting both GPU Slots migrates both source Pages in one batch. The
    // first event callback throws only after Page 0 has accepted its Host
    // Slot; Page 1's destination must remain locally owned and be rolled back.
    EXPECT_THROW(storage.newGpuSlots(twoSlots), std::runtime_error);

    EXPECT_EQ(eventSink->calls(), 1);
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, gpuBefore.free + 1);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBefore.free - 1);
    EXPECT_EQ(std::count_if(
                  pages.begin(), pages.end(), [hostLevel](auto const& page) { return page->cacheLevel == hostLevel; }),
        1);
    EXPECT_EQ(
        std::count_if(pages.begin(), pages.end(), [](auto const& page) { return page->cacheLevel == kGpuLevel; }), 1);
    for (auto const& page : pages)
    {
        EXPECT_TRUE(page->hasValidSlot());
        EXPECT_TRUE(page->scheduledForEviction());
    }
}

TEST(KvCacheManagerV2StatsTest, DecodeRejectionKeepsSourcePageAndReleasesGpuSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<RuntimeRejectingColdPageCodec>(RuntimeRejectingColdPageCodec::Direction::kDecode);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hostSlots = storage.newSlots(hostLevel, oneSlot);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < manager->tokensPerBlock(); ++i)
        tokens.emplace_back(TokenId{i});
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, hostLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
    page->setSlot(hostSlots[lifeCycle].front());
    block->storage[lifeCycle] = page.get();

    SlotId const sourceSlot = page->slotId();
    auto const gpuBefore = storage.getStatistics(kGpuLevel);
    auto const hostBefore = storage.getStatistics(hostLevel);
    auto cache = manager->createKvCache();
    std::vector<BatchedLockTarget> targets{{page, kDefaultBeamIndex, BlockOrdinal{0}, lifeCycle}};
    int migrations = 0;
    MigrationRecorder recorder = [&](auto const&, auto const&, CacheLevel, CacheLevel) { ++migrations; };

    EXPECT_THROW(storage.batchedMigrateToGpu(targets, *cache, recorder), LogicError);

    EXPECT_EQ(migrations, 0);
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, gpuBefore.free);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBefore.free);
    EXPECT_EQ(page->cacheLevel, hostLevel);
    EXPECT_EQ(page->slotId(), sourceSlot);
    cache->close();
}

TEST(KvCacheManagerV2StatsTest, HostToDiskMigrationBypassesColdPageCodec)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<SameRepresentationGuardColdPageCodec>();
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeHostDiskTieredConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};
    CacheLevel const diskLevel{2};

    StorageStatistics const hostStats = storage.getStatistics(hostLevel);
    TypedVec<LifeCycleId, SlotCount> allHostSlots(LifeCycleId{1}, hostStats.total);
    auto occupiedHostSlots = storage.newSlots(hostLevel, allHostSlots);
    ASSERT_EQ(occupiedHostSlots[lifeCycle].size(), slotCountToSizeT(hostStats.total));

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, hostLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
    page->setSlot(occupiedHostSlots[lifeCycle].front());
    block->storage[lifeCycle] = page.get();
    storage.scheduleForEviction(*page);

    int migrations = 0;
    MigrationRecorder recorder = [&](auto const& pages, auto const&, CacheLevel srcLevel, CacheLevel dstLevel)
    {
        EXPECT_EQ(pages.size(), 1);
        EXPECT_EQ(srcLevel, hostLevel);
        EXPECT_EQ(dstLevel, diskLevel);
        ++migrations;
    };
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto replacementHostSlot = storage.newSlots(hostLevel, oneSlot, recorder);
    page->readyEvent.synchronize();

    EXPECT_EQ(migrations, 1);
    EXPECT_EQ(page->cacheLevel, diskLevel);
    EXPECT_FALSE(codecPtr->queriedInvalidLayerGroup());
    EXPECT_EQ(codecPtr->transformCalls(), 0);

    storage.releaseSlot(lifeCycle, hostLevel, std::move(replacementHostSlot[lifeCycle].front()));
    for (auto& slot : occupiedHostSlots[lifeCycle])
    {
        if (slot.hasValidSlot())
        {
            storage.releaseSlot(lifeCycle, hostLevel, std::move(slot));
        }
    }
}

TEST(KvCacheManagerV2StatsTest, GpuDefragmentationBypassesColdPageCodec)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<SameRepresentationGuardColdPageCodec>();
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    PoolGroupIndex const poolGroup = storage.getPoolGroupIndex(kGpuLevel, lifeCycle);

    // Allocate slots 0, 1, and 2, then leave slot 0 free. Shrinking to two
    // slots must relocate the Page from overflow slot 2 into slot 0.
    TypedVec<LifeCycleId, SlotCount> threeSlots(LifeCycleId{1}, 3);
    auto gpuSlots = storage.newGpuSlots(threeSlots);
    ASSERT_EQ(gpuSlots[lifeCycle].size(), 3);
    ASSERT_EQ(gpuSlots[lifeCycle][0].slotId(), SlotId{0});
    ASSERT_EQ(gpuSlots[lifeCycle][1].slotId(), SlotId{1});
    ASSERT_EQ(gpuSlots[lifeCycle][2].slotId(), SlotId{2});
    storage.releaseSlot(lifeCycle, kGpuLevel, std::move(gpuSlots[lifeCycle][0]));

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
    page->setSlot(gpuSlots[lifeCycle][2]);
    block->storage[lifeCycle] = page.get();

    size_t const pageBytes = storage.slotSize(kGpuLevel, poolGroup).at(PoolIndex{0});
    MemAddress const sourceAddress
        = std::get<MemAddress>(storage.slotAddress(kGpuLevel, poolGroup, page->slotId(), PoolIndex{0}));
    constexpr uint8_t kPattern = 0xA7;
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(sourceAddress), kPattern, pageBytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<SharedPtr<Page>> persistentPages{page};
    storage.shrinkPoolGroup(kGpuLevel, poolGroup, SlotCount{2}, persistentPages);
    page->readyEvent.synchronize();

    EXPECT_EQ(page->cacheLevel, kGpuLevel);
    EXPECT_EQ(page->slotId(), SlotId{0});
    EXPECT_FALSE(codecPtr->queriedInvalidLayerGroup());
    EXPECT_EQ(codecPtr->transformCalls(), 0);

    MemAddress const destinationAddress
        = std::get<MemAddress>(storage.slotAddress(kGpuLevel, poolGroup, page->slotId(), PoolIndex{0}));
    uint8_t firstByte = 0;
    uint8_t lastByte = 0;
    ASSERT_EQ(cudaMemcpy(&firstByte, reinterpret_cast<void const*>(destinationAddress), 1, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&lastByte, reinterpret_cast<void const*>(destinationAddress + pageBytes - 1), 1,
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(firstByte, kPattern);
    EXPECT_EQ(lastByte, kPattern);

    storage.releaseSlot(lifeCycle, kGpuLevel, std::move(gpuSlots[lifeCycle][1]));
}

TEST(KvCacheManagerV2StatsTest, RecursiveCapacityFailureReschedulesGpuAndHostVictims)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHostDiskTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};
    CacheLevel const diskLevel{2};

    auto const gpuCapacity = storage.getStatistics(kGpuLevel);
    auto const hostCapacity = storage.getStatistics(hostLevel);
    auto const diskCapacity = storage.getStatistics(diskLevel);
    ASSERT_GT(gpuCapacity.total, 0);
    ASSERT_GT(hostCapacity.total, 0);
    ASSERT_GT(diskCapacity.total, 0);

    TypedVec<LifeCycleId, SlotCount> allGpuSlots(LifeCycleId{1}, gpuCapacity.total);
    TypedVec<LifeCycleId, SlotCount> allHostSlots(LifeCycleId{1}, hostCapacity.total);
    TypedVec<LifeCycleId, SlotCount> allDiskSlots(LifeCycleId{1}, diskCapacity.total);
    auto occupiedGpuSlots = storage.newSlots(kGpuLevel, allGpuSlots);
    auto occupiedHostSlots = storage.newSlots(hostLevel, allHostSlots);
    auto occupiedDiskSlots = storage.newSlots(diskLevel, allDiskSlots);

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    int token = 0;
    auto makePage = [&](CacheLevel level, Slot& slot)
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < manager->tokensPerBlock(); ++i)
        {
            tokens.emplace_back(TokenId{token++});
        }
        auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, level, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        return page;
    };

    auto gpuPage = makePage(kGpuLevel, occupiedGpuSlots[lifeCycle].front());
    storage.scheduleForEviction(*gpuPage);
    auto hostPage = makePage(hostLevel, occupiedHostSlots[lifeCycle].front());
    auto hostHolder = hostPage->hold();
    storage.scheduleForEviction(*hostPage);
    ASSERT_TRUE(hostHolder);
    ASSERT_EQ(gpuPage->status(), PageStatus::DROPPABLE);
    ASSERT_EQ(hostPage->status(), PageStatus::HELD);
    ASSERT_TRUE(gpuPage->scheduledForEviction());
    ASSERT_TRUE(hostPage->scheduledForEviction());

    SlotId const gpuSlot = gpuPage->slotId();
    SlotId const hostSlot = hostPage->slotId();
    auto const gpuBeforeFailure = storage.getStatistics(kGpuLevel);
    auto const hostBeforeFailure = storage.getStatistics(hostLevel);
    auto const diskBeforeFailure = storage.getStatistics(diskLevel);
    TypedVec<LifeCycleId, SlotCount> oneGpuSlot(LifeCycleId{1}, 1);
    int migrations = 0;
    MigrationRecorder recorder = [&](auto const&, auto const&, CacheLevel, CacheLevel) { ++migrations; };
    EXPECT_THROW(storage.newGpuSlots(oneGpuSlot, recorder), OutOfPagesError);

    EXPECT_EQ(migrations, 0);
    EXPECT_EQ(gpuPage->cacheLevel, kGpuLevel);
    EXPECT_EQ(gpuPage->slotId(), gpuSlot);
    EXPECT_EQ(gpuPage->status(), PageStatus::DROPPABLE);
    EXPECT_TRUE(gpuPage->scheduledForEviction());
    EXPECT_EQ(hostPage->cacheLevel, hostLevel);
    EXPECT_EQ(hostPage->slotId(), hostSlot);
    EXPECT_EQ(hostPage->status(), PageStatus::HELD);
    EXPECT_TRUE(hostPage->scheduledForEviction());
    EXPECT_EQ(storage.getStatistics(kGpuLevel).free, gpuBeforeFailure.free);
    EXPECT_EQ(storage.getStatistics(hostLevel).free, hostBeforeFailure.free);
    EXPECT_EQ(storage.getStatistics(diskLevel).free, diskBeforeFailure.free);

    for (auto& slot : occupiedGpuSlots[lifeCycle])
    {
        if (slot.hasValidSlot())
        {
            storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
        }
    }
    for (auto& slot : occupiedHostSlots[lifeCycle])
    {
        if (slot.hasValidSlot())
        {
            storage.releaseSlot(lifeCycle, hostLevel, std::move(slot));
        }
    }
    for (auto& slot : occupiedDiskSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, diskLevel, std::move(slot));
    }
}

TEST(KvCacheManagerV2StatsTest, CodecRejectionRestoresSelectedVictimOrder)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<RejectFirstEncodeColdPageCodec>();
    auto manager = std::make_shared<KvCacheManager>(makeRollbackOrderConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const hostLevel{1};

    TypedVec<LifeCycleId, SlotCount> threeSlots(LifeCycleId{1}, 3);
    auto gpuSlots = storage.newGpuSlots(threeSlots);
    SlotCount const remainingGpuSlots = storage.numSlots(PoolGroupIndex{0}, kGpuLevel) - 3;
    ASSERT_GE(remainingGpuSlots, 0);
    auto reservedGpuSlots = storage.newSlotsForPoolGroup(kGpuLevel, PoolGroupIndex{0}, remainingGpuSlots);

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
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }
    ASSERT_EQ(pages.size(), 3);

    // A, B, and C entered the same-priority LRU in that order. The first
    // two-slot request selects A and B, then the codec rejects before either
    // migration is published. Rollback must put A and B back ahead of the
    // unselected C without reversing A/B's order.
    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    EXPECT_THROW(storage.newGpuSlots(twoSlots), LogicError);
    for (auto const& page : pages)
    {
        EXPECT_EQ(page->cacheLevel, kGpuLevel);
        EXPECT_TRUE(page->scheduledForEviction());
    }

    std::vector<Page const*> migratedPages;
    MigrationRecorder recorder = [&](auto const& migrated, auto const&, CacheLevel srcLevel, CacheLevel dstLevel)
    {
        EXPECT_EQ(srcLevel, kGpuLevel);
        EXPECT_EQ(dstLevel, hostLevel);
        for (auto const& page : migrated)
        {
            migratedPages.push_back(page.get());
        }
    };
    auto replacementSlots = storage.newGpuSlots(twoSlots, recorder);

    ASSERT_EQ(migratedPages.size(), 2);
    EXPECT_EQ(migratedPages[0], pages[0].get());
    EXPECT_EQ(migratedPages[1], pages[1].get());
    EXPECT_EQ(pages[0]->cacheLevel, hostLevel);
    EXPECT_EQ(pages[1]->cacheLevel, hostLevel);
    EXPECT_EQ(pages[2]->cacheLevel, kGpuLevel);

    for (auto& slot : replacementSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
    for (auto& slot : reservedGpuSlots)
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
}

TEST(KvCacheManagerV2StatsTest, LaterEncodeRejectionReschedulesEveryAcceptedMigrationGroup)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<RejectSecondEncodeColdPageCodec>();
    auto manager = std::make_shared<KvCacheManager>(makeTwoLifeCycleRollbackConfig(), nullptr, std::move(codec));
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    ASSERT_EQ(storage.numPoolGroups(kGpuLevel), PoolGroupIndex{1});

    TypedVec<LifeCycleId, SlotCount> onePerLifeCycle(LifeCycleId{2}, 1);
    auto gpuSlots = storage.newGpuSlots(onePerLifeCycle);
    SlotCount const remainingGpuSlots = storage.numSlots(PoolGroupIndex{0}, kGpuLevel) - 2;
    ASSERT_GE(remainingGpuSlots, 0);
    auto reservedGpuSlots = storage.newSlotsForPoolGroup(kGpuLevel, PoolGroupIndex{0}, remainingGpuSlots);
    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<SharedPtr<Page>> pages;
    NodeBase* previous = &root;
    int token = 0;
    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < manager->tokensPerBlock(); ++i)
            tokens.emplace_back(TokenId{token++});
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(gpuSlots[lifeCycle].front());
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }

    CacheLevel const hostLevel{1};
    std::vector<Page const*> migratedPages;
    MigrationRecorder recorder = [&](auto const& migrated, auto const&, CacheLevel srcLevel, CacheLevel dstLevel)
    {
        EXPECT_EQ(migrated.size(), 1);
        EXPECT_EQ(srcLevel, kGpuLevel);
        EXPECT_EQ(dstLevel, hostLevel);
        migratedPages.push_back(migrated.front().get());
    };
    EXPECT_THROW(storage.newGpuSlots(onePerLifeCycle, recorder), LogicError);
    ASSERT_EQ(migratedPages.size(), 1);
    EXPECT_EQ(migratedPages.front(), pages[0].get());
    EXPECT_EQ(pages[0]->cacheLevel, hostLevel);
    EXPECT_EQ(pages[1]->cacheLevel, kGpuLevel);
    EXPECT_TRUE(pages[0]->scheduledForEviction());
    EXPECT_TRUE(pages[1]->scheduledForEviction());

    // The first group remains published at Host: _prepareFreeSlots rollback is
    // intentionally partial-commit. The rejected second group stays on GPU and
    // must be restored to that level's eviction controller.
    auto replacementSlots = storage.newGpuSlots(onePerLifeCycle, recorder);
    ASSERT_EQ(migratedPages.size(), 2);
    EXPECT_EQ(migratedPages[1], pages[1].get());
    EXPECT_EQ(pages[0]->cacheLevel, hostLevel);
    EXPECT_EQ(pages[1]->cacheLevel, hostLevel);
    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        EXPECT_EQ(replacementSlots[lifeCycle].size(), 1);
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(replacementSlots[lifeCycle].front()));
    }
    for (auto& slot : reservedGpuSlots)
    {
        storage.releaseSlot(LifeCycleId{0}, kGpuLevel, std::move(slot));
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
    EXPECT_TRUE(pending.recordAllocationRange(LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{3}, /*beamWidth=*/2,
        /*countAsMissed=*/true));
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
    EXPECT_TRUE(pending.recordAllocationRange(LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{1}, /*beamWidth=*/1,
        /*countAsMissed=*/true));
    EXPECT_TRUE(pending.recordReuse(LifeCycleId{0}, /*fullReusedBlocks=*/2,
        /*partialReusedBlocks=*/1));

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
        auto block = addOrGetExistingBlock(previous, std::move(tokens),
            /*knownNoDigest=*/true);
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
            auto block = addOrGetExistingBlock(previous, std::move(tokens),
                /*knownNoDigest=*/true);
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
