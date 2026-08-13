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

#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/logger.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// CacheLevelManager
// ---------------------------------------------------------------------------

CacheLevelManager::CacheLevelManager(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cl,
    CacheTierConfig const& tierConfig, TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList)
    : cacheLevel(cl)
    , cacheTier(CacheTier(tierConfig.index()))
    , controller(lifeCycleGrouping, cl)
{
    storage = createCacheLevelStorage(tierConfig, slotDescList, slotCountList);
}

size_t CacheLevelManager::cacheTierGranularity(CacheTier tier, size_t quota)
{
    switch (tier)
    {
    case CacheTier::GPU_MEM:
    {
        constexpr size_t kPageSize = 2ULL << 20;
        return kPageSize << std::min(4, std::max(0, static_cast<int>(std::log2(quota / (kPageSize * 512)))));
    }
    case CacheTier::HOST_MEM: return HostMem::kAlignment; // 4 KiB
    case CacheTier::DISK: return size_t{2} << 20;         // DiskCacheLevelStorage::POOL_SIZE_GRANULARITY
    default: throw std::invalid_argument("Invalid cache tier");
    }
}

// ---------------------------------------------------------------------------
// StorageManager constructor helpers
// ---------------------------------------------------------------------------

namespace
{

bool isGpuAccessibleMemory(CacheTier tier) noexcept
{
    return tier == CacheTier::GPU_MEM || tier == CacheTier::HOST_MEM;
}

// Compute the slot-to-page-indices scale factors.
// For each (lcId, poolIdx), scale = numBuffersInCoalescedSlot.
// Python: _slot_to_page_indices[lc_id][pool_idx] = numBuffers
TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> computeSlotToPageIndices(StorageConfig const& config)
{
    LifeCycleId numLc = config.numLifeCycles();
    TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> result(numLc);

    auto const& slotDescList = config.slotDescList;
    auto const& grouping = config.lifeCycleGrouping();

    for (LifeCycleId lcId{0}; lcId < result.size(); ++lcId)
    {
        PoolGroupIndex pgIdx = grouping[lcId];
        SlotDesc const& sd = slotDescList.at(pgIdx);
        // Find the variant that corresponds to this lifecycle.
        for (auto const& variant : sd.variants)
        {
            if (variant.lifeCycleId == lcId)
            {
                // Each coalesced buffer contributes its numBuffers as the scale.
                result[lcId].reserve(variant.coalescedBuffers.size());
                for (auto const& cb : variant.coalescedBuffers)
                    result[lcId].push_back(cb.numBuffers());
                break;
            }
        }
        if (result[lcId].empty())
            result[lcId].push_back(1); // fallback
    }
    return result;
}

} // namespace

template <typename Submit>
bool StorageManager::submitColdPageCodec(
    PageIndexLocation location, PageIndexPair const* pageIndices, size_t numPages, CUstream stream, Submit&& submit)
{
    if (location == PageIndexLocation::kHost || numPages == 0)
    {
        return submit(pageIndices, numPages, stream);
    }
    if (location != PageIndexLocation::kDevice || pageIndices == nullptr || !mIndexStagingManager)
    {
        return false;
    }
    if (numPages > std::numeric_limits<size_t>::max() / sizeof(PageIndexPair))
    {
        throw std::overflow_error("Cold-page codec index-array size overflow");
    }

    size_t offset = 0;
    while (offset < numPages)
    {
        size_t const remainingBytes = (numPages - offset) * sizeof(PageIndexPair);
        auto device = mIndexStagingManager->acquire(sizeof(PageIndexPair),
            std::min(remainingBytes, kMaxIndexBatchBytes), sizeof(PageIndexPair), alignof(PageIndexPair), stream);
        size_t const chunkPages = std::min(numPages - offset, device.size() / sizeof(PageIndexPair));
        TLLM_CHECK_DEBUG(chunkPages > 0);
        size_t chunkBytes = chunkPages * sizeof(PageIndexPair);
        // The index vector is ephemeral. CUDA consumes it before returning while the H2D transfer remains asynchronous.
        CUdeviceptr dst = static_cast<CUdeviceptr>(device.address());
        CUdeviceptr src = reinterpret_cast<CUdeviceptr>(pageIndices + offset);
        CUmemcpyAttributes attributes{};
        attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL;
        attributes.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
        attributes.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
        attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        size_t firstCopy = 0;
        cuCheck(cuMemcpyBatchAsync(&dst, &src, &chunkBytes, 1, &attributes, &firstCopy, 1, stream));

        if (!submit(reinterpret_cast<PageIndexPair const*>(device.address()), chunkPages, stream))
        {
            return false;
        }
        offset += chunkPages;
    }
    return true;
}

// ---------------------------------------------------------------------------
// StorageManager
// ---------------------------------------------------------------------------

StorageManager::StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
    IKvCacheColdPageCodec& codec, std::optional<SwaScratchReuseConfig> swaScratchReuse,
    std::optional<BatchDesc> const& typicalBatch, std::vector<BatchDesc> const& constraints,
    std::optional<std::vector<float>> const& initialPoolRatio, std::shared_ptr<EventSink> eventSink,
    float maxUtilForResume)
    : mLifeCycles(lifeCycles)
    , mEventSink(std::move(eventSink))
    , mStorageConfig(config)
    , mSwaScratchReuse(std::move(swaScratchReuse))
{
    mLifeCycleGroupings.resize(config.cacheTiers.size(), config.lifeCycleGrouping());
    mLayerToLifeCycleIds = config.layerToLifeCycleIds();
    mSlotToPageIndices = computeSlotToPageIndices(config);
    mBufferAttr = config.bufferAttributes();
    mSlotDescLists.resize(config.cacheTiers.size(), config.slotDescList);

    // Compute layer attributes and slot utilization fractions for scratch support.
    mLayerAttributes = config.layerAttributes();
    mSlotUtilFracMax.resize(lifeCycles.size(), Rational{0, 1});
    for (auto const& [layerId, layerAttr] : mLayerAttributes)
    {
        LifeCycleId const lcIdx = layerAttr.lifeCycleId;
        if (layerAttr.slotUtilFracMax > mSlotUtilFracMax[lcIdx])
        {
            mSlotUtilFracMax[lcIdx] = layerAttr.slotUtilFracMax;
        }
    }

    auto const& hotLifeCycleGrouping = lifeCycleGrouping(kGpuLevel);
    auto const& hotSlotDescList = slotDescList(kGpuLevel);
    TLLM_CHECK_DEBUG(std::all_of(hotLifeCycleGrouping.begin(), hotLifeCycleGrouping.end(),
        [this](PoolGroupIndex pg) { return pg < numPoolGroups(); }));
    TLLM_CHECK_DEBUG(numPoolGroups()
        == PoolGroupIndex{static_cast<int>(
            std::set<PoolGroupIndex>(hotLifeCycleGrouping.begin(), hotLifeCycleGrouping.end()).size())});

    // Build one CacheLevelManager per tier.
    TLLM_CHECK_DEBUG(!config.cacheTiers.empty());
    bool const needsPageStaging = config.cacheTiers.size() > CacheLevel{1}
        && std::any_of(config.cacheTiers.begin() + 1, config.cacheTiers.end(),
            [](CacheTierConfig const& tierConfig) { return !isGpuAccessibleMemory(cacheTierOf(tierConfig)); });
    TLLM_CHECK_DEBUG_WITH_INFO(
        std::holds_alternative<GpuCacheTierConfig>(config.cacheTiers[kGpuLevel]), "First cache tier must be GPU");

    // Compute slot size lists for all pool groups.
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> slotSizeLists;
    slotSizeLists.reserve(hotSlotDescList.size());
    for (auto const& sd : hotSlotDescList)
    {
        slotSizeLists.push_back(sd.slotSizeList());
    }

    size_t gpuQuota = cacheTierQuota(config.cacheTiers[kGpuLevel]);
    size_t gpuGranularity = CacheLevelManager::cacheTierGranularity(CacheTier::GPU_MEM, gpuQuota);

    // Constraints stay feasibility floors even under an explicit initial pool
    // ratio (a share below what a declared batch needs is clamped up), and the
    // floors are scaled by 1/maxUtilForResume because KvCache::resume rejects any
    // pool group above that utilization. Mirrors PR#16269 on the Python side.
    auto minSlotsByLifeCycle
        = computeMinSlotsByLifeCycleFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse, maxUtilForResume);
    TypedVec<PoolGroupIndex, SlotCount> hotMinSlots(numPoolGroups(kGpuLevel), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        hotMinSlots[getPoolGroupIndex(kGpuLevel, lifeCycle)] += minSlotsByLifeCycle[lifeCycle];
    }
    mMinSlotsByLevel.resize(config.cacheTiers.size(), hotMinSlots);

    // Compute init_ratio from explicit config, typical_batch, constraints, or fallback.
    TypedVec<PoolGroupIndex, float> initRatio;
    if (initialPoolRatio.has_value())
    {
        if (initialPoolRatio->size() != toSizeT(numPoolGroups()))
        {
            throw std::invalid_argument("initial_pool_ratio length must match number of pool groups ("
                + std::to_string(toSizeT(numPoolGroups())) + "), got " + std::to_string(initialPoolRatio->size()));
        }
        if (std::any_of(initialPoolRatio->begin(), initialPoolRatio->end(), [](float ratio) { return ratio <= 0.0F; }))
        {
            throw std::invalid_argument("initial_pool_ratio values must be positive");
        }

        constexpr double kExpectedRatioSum = 1.0;
        constexpr double kRatioSumTolerance = 1e-6;
        double const ratioSum = std::accumulate(initialPoolRatio->begin(), initialPoolRatio->end(), 0.0);
        if (!std::isfinite(ratioSum) || std::abs(ratioSum - kExpectedRatioSum) > kRatioSumTolerance)
        {
            throw std::invalid_argument("initial_pool_ratio values must sum to 1.0");
        }
        initRatio = TypedVec<PoolGroupIndex, float>(*initialPoolRatio);
    }
    else if (typicalBatch.has_value())
    {
        initRatio = ratioFromBatch(*typicalBatch, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
    }
    else if (!constraints.empty())
    {
        // Use the constraint slot counts as the ratio basis.
        auto minBytes = slotsToBytes(mMinSlotsByLevel[kGpuLevel], gpuGranularity);
        initRatio = normalizeToRatio(minBytes);
    }
    else
    {
        // Fallback: average history length 2048.
        BatchDesc fallback;
        fallback.kvCaches.push_back(KVCacheDesc{2049, 2048});
        initRatio = ratioFromBatch(fallback, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
    }

    mLevels.reserve(config.cacheTiers.size());

    auto gpuSlotCounts
        = computeSlotCountForLevel(config.cacheTiers[kGpuLevel], slotSizeLists, initRatio, mMinSlotsByLevel[kGpuLevel]);
    mLevels.emplace_back(
        lifeCycleGrouping(kGpuLevel), kGpuLevel, config.cacheTiers[kGpuLevel], slotDescList(kGpuLevel), gpuSlotCounts);

    auto& gpuStorage = *mLevels[kGpuLevel].storage;
    TypedVec<PoolGroupIndex, PoolGroupDesc> gpuDescs;
    gpuDescs.reserve(numPoolGroups(kGpuLevel));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(kGpuLevel); ++pgIdx)
    {
        TypedVec<PoolIndex, PoolDesc> pools;
        auto const poolSizes = slotSize(kGpuLevel, pgIdx);
        pools.reserve(poolSizes.size());
        for (PoolIndex poolIdx{0}; poolIdx < poolSizes.size(); ++poolIdx)
        {
            pools.push_back(
                PoolDesc{poolIdx, gpuStorage.getBaseAddress(pgIdx, poolIdx, SlotId{0}), poolSizes.at(poolIdx)});
        }
        gpuDescs.push_back(
            PoolGroupDesc{pgIdx, gpuStorage.numSlots(pgIdx), slotDescList(kGpuLevel).at(pgIdx), std::move(pools)});
    }
    if (!codec.configure(gpuDescs.raw().data(), gpuDescs.size()))
    {
        throw std::invalid_argument("Cold-page codec rejected GPU pool-group configuration");
    }

    TypedVec<LifeCycleId, size_t> coldPageBytesByLifeCycle(numLifeCycles());
    size_t maxColdPageBytes = 0;
    mBatchingLayerGroupIds.resize(numLifeCycles());
    mPageIndexLocations.resize(numLifeCycles());
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        size_t const coldPageBytes = codec.queryColdPageBytes(lifeCycle);
        if (coldPageBytes == 0)
        {
            throw std::invalid_argument(
                "Cold-page codec returned zero bytes for lifecycle " + std::to_string(lifeCycle.value()));
        }
        coldPageBytesByLifeCycle[lifeCycle] = coldPageBytes;
        maxColdPageBytes = std::max(maxColdPageBytes, coldPageBytes);

        LayerGroupId const batchingLayerGroupId = codec.getBatchingLayerGroupId(lifeCycle);
        if (batchingLayerGroupId.value() < 0 || batchingLayerGroupId >= numLifeCycles())
        {
            throw std::invalid_argument("Cold-page codec returned an invalid batching layer-group ID for lifecycle "
                + std::to_string(lifeCycle.value()));
        }
        if (batchingLayerGroupId > lifeCycle)
        {
            throw std::invalid_argument(
                "Cold-page codec batching layer-group ID must be the smallest lifecycle in its equivalence class");
        }
        if (getPoolGroupIndex(kGpuLevel, batchingLayerGroupId) != getPoolGroupIndex(kGpuLevel, lifeCycle))
        {
            throw std::invalid_argument("Cold-page codec cannot batch lifecycles from different GPU pool groups");
        }
        mBatchingLayerGroupIds[lifeCycle] = batchingLayerGroupId;

        PageIndexLocation const pageIndexLocation = codec.queryPageIndexLocation(lifeCycle);
        if (pageIndexLocation != PageIndexLocation::kHost && pageIndexLocation != PageIndexLocation::kDevice)
        {
            throw std::invalid_argument("Cold-page codec returned an invalid page-index location for lifecycle "
                + std::to_string(lifeCycle.value()));
        }
        mPageIndexLocations[lifeCycle] = pageIndexLocation;
    }

    size_t pageStagingBytes = 0;
    if (needsPageStaging)
    {
        if (maxColdPageBytes > std::numeric_limits<size_t>::max() / kPageStagingDepth)
        {
            throw std::overflow_error("Cold-page staging size overflow");
        }
        pageStagingBytes = std::max(kDefaultPageStagingBytes, kPageStagingDepth * maxColdPageBytes);
    }

    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        LayerGroupId const batchingLayerGroupId = mBatchingLayerGroupIds[lifeCycle];
        if (mBatchingLayerGroupIds[batchingLayerGroupId] != batchingLayerGroupId)
        {
            throw std::invalid_argument("Cold-page codec returned a non-canonical batching layer-group ID");
        }
        if (coldPageBytesByLifeCycle[batchingLayerGroupId] != coldPageBytesByLifeCycle[lifeCycle])
        {
            throw std::invalid_argument("Cold-page codec batching equivalence class has inconsistent cold-page sizes");
        }
        if (mPageIndexLocations[batchingLayerGroupId] != mPageIndexLocations[lifeCycle])
        {
            throw std::invalid_argument(
                "Cold-page codec batching equivalence class has inconsistent page-index locations");
        }
    }

    bool const needsIndexStaging = std::any_of(mPageIndexLocations.begin(), mPageIndexLocations.end(),
        [](PageIndexLocation location) { return location == PageIndexLocation::kDevice; });
    if (needsIndexStaging)
    {
        mIndexStagingManager = std::make_unique<StagingBufferManager>(kIndexStagingBytes, StagingBufferMemory::kDevice);
    }

    TypedVec<LifeCycleId, PoolGroupIndex> coldGrouping(numLifeCycles());
    TypedVec<PoolGroupIndex, SlotDesc> coldSlotDescList;
    std::map<size_t, PoolGroupIndex> coldGroupByPageBytes;
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        size_t const coldPageBytes = coldPageBytesByLifeCycle[lifeCycle];
        auto [it, inserted] = coldGroupByPageBytes.emplace(
            coldPageBytes, PoolGroupIndex{static_cast<int>(coldSlotDescList.size().value())});
        PoolGroupIndex const coldPgIdx = it->second;
        if (inserted)
        {
            coldSlotDescList.push_back(SlotDesc{});
        }
        coldGrouping[lifeCycle] = coldPgIdx;

        SlotDescVariant variant;
        variant.lifeCycleId = lifeCycle;
        variant.coalescedBuffers.push_back(
            CoalescedBuffer{coldPageBytes, std::vector<BufferId>{BufferId{-1, "__cold_page__"}}});
        coldSlotDescList[coldPgIdx].variants.push_back(std::move(variant));
    }

    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        LayerGroupId const batchingLayerGroupId = mBatchingLayerGroupIds[lifeCycle];
        if (coldGrouping[batchingLayerGroupId] != coldGrouping[lifeCycle])
        {
            throw std::invalid_argument("Cold-page codec cannot batch lifecycles from different cold pool groups");
        }
    }

    TypedVec<PoolGroupIndex, int> hotLifeCycleCounts(numPoolGroups(kGpuLevel), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        ++hotLifeCycleCounts[getPoolGroupIndex(kGpuLevel, lifeCycle)];
    }

    TypedVec<PoolGroupIndex, float> coldRatio(coldSlotDescList.size(), 0.0F);
    TypedVec<PoolGroupIndex, SlotCount> coldMinSlots(coldSlotDescList.size(), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        PoolGroupIndex const hotPgIdx = getPoolGroupIndex(kGpuLevel, lifeCycle);
        PoolGroupIndex const coldPgIdx = coldGrouping[lifeCycle];
        int const lifeCyclesInHotGroup = hotLifeCycleCounts[hotPgIdx];
        coldRatio[coldPgIdx] += initRatio[hotPgIdx] / static_cast<float>(lifeCyclesInHotGroup);
        coldMinSlots[coldPgIdx] += minSlotsByLifeCycle[lifeCycle];
    }
    coldRatio = normalizeToRatio(coldRatio);

    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> coldSlotSizeLists;
    coldSlotSizeLists.reserve(coldSlotDescList.size());
    for (auto const& desc : coldSlotDescList)
    {
        auto sizes = desc.slotSizeList();
        TLLM_CHECK_WITH_INFO(sizes.size() == PoolIndex{1}, "Cold pool groups must contain exactly one pool");
        coldSlotSizeLists.push_back(std::move(sizes));
    }

    for (CacheLevel level{1}; level < config.cacheTiers.size(); ++level)
    {
        mLifeCycleGroupings[level] = coldGrouping;
        mSlotDescLists[level] = coldSlotDescList;
        mMinSlotsByLevel[level] = coldMinSlots;
        auto slotCounts
            = computeSlotCountForLevel(config.cacheTiers[level], coldSlotSizeLists, coldRatio, mMinSlotsByLevel[level]);
        mLevels.emplace_back(
            lifeCycleGrouping(level), level, config.cacheTiers[level], slotDescList(level), slotCounts);
    }

    for (CacheLevel level{0}; level < mLevels.size(); ++level)
    {
        TLLM_CHECK_DEBUG(numPoolGroups(level) == mLevels[level].storage->numPoolGroups());
    }

    if (needsPageStaging)
    {
        mPageStagingManager
            = std::make_unique<StagingBufferManager>(pageStagingBytes, StagingBufferMemory::kPinnedHost);
    }
    mCopyEngine = std::make_unique<CopyEngine>(mPageStagingManager.get());
}

StorageManager::~StorageManager()
{
    destroy();
}

void StorageManager::destroy()
{
    for (auto& lvl : mLevels)
    {
        TLLM_CHECK_DEBUG(lvl.storage);
        lvl.storage->destroy();
    }
    mLevels.clear();

    mIndexStagingManager.reset();
    mCopyEngine.reset();
    mPageStagingManager.reset();
}

// ---------------------------------------------------------------------------
// newSlots
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newSlots(CacheLevel level,
    TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc, MigrationRecorder const& migrationRecorder,
    DropRecorder const& dropRecorder)
{
    auto const& grouping = lifeCycleGrouping(level);
    TLLM_CHECK_DEBUG(numSlotsPerLc.size() == numLifeCycles());
    auto& storage = *mLevels.at(level).storage;

    // Aggregate by pool group.
    TypedVec<PoolGroupIndex, SlotCount> pgNumSlots(numPoolGroups(level), 0);
    for (LifeCycleId lcId{0}; lcId < numSlotsPerLc.size(); ++lcId)
    {
        SlotCount const numSlots = numSlotsPerLc[lcId];
        if (numSlots < 0)
        {
            throw LogicError("StorageManager::newSlots: slot count must be non-negative");
        }
        pgNumSlots[grouping[lcId]] += numSlots;
    }

    // Prepare free slots if needed.
    bool needMore = false;
    for (PoolGroupIndex pgIdx{0}; pgIdx < pgNumSlots.size(); ++pgIdx)
    {
        if (pgNumSlots[pgIdx] > storage.numFreeSlots(pgIdx))
        {
            needMore = true;
            break;
        }
    }

    if (needMore)
    {
        prepareFreeSlots(level, pgNumSlots, migrationRecorder, dropRecorder);
    }

    // A14: post-condition — free-slot counts satisfy requirements.
    for (PoolGroupIndex pgIdx{0}; pgIdx < pgNumSlots.size(); ++pgIdx)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(pgNumSlots[pgIdx] <= storage.numFreeSlots(pgIdx),
            "Free slot count does not satisfy requirement after prepareFreeSlots");
    }

    // Allocate.
    TypedVec<LifeCycleId, std::vector<Slot>> ret(numLifeCycles());
    try
    {
        for (LifeCycleId lcId{0}; lcId < ret.size(); ++lcId)
        {
            PoolGroupIndex pg = grouping[lcId];
            ret[lcId] = storage.allocateMultiple(pg, numSlotsPerLc[lcId]);
        }
    }
    catch (...)
    {
        for (LifeCycleId lcId{0}; lcId < ret.size(); ++lcId)
        {
            PoolGroupIndex pg = grouping[lcId];
            for (auto& s : ret[lcId])
                storage.release(pg, std::move(s));
        }
        throw;
    }
    return ret;
}

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newGpuSlots(
    TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc, MigrationRecorder const& migrationRecorder,
    DropRecorder const& dropRecorder)
{
    return newSlots(kGpuLevel, numSlotsPerLc, migrationRecorder, dropRecorder);
}

std::vector<Slot> StorageManager::newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount numSlots,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    if (numSlots < 0)
    {
        throw LogicError("StorageManager::newSlotsForPoolGroup: numSlots must be non-negative");
    }
    auto& storage = *mLevels.at(level).storage;
    if (numSlots > storage.numFreeSlots(pgIdx))
    {
        TypedVec<PoolGroupIndex, SlotCount> requirements(numPoolGroups(level), 0);
        requirements.at(pgIdx) = numSlots;
        prepareFreeSlots(level, requirements, migrationRecorder, dropRecorder);
    }
    TLLM_CHECK_DEBUG(numSlots <= storage.numFreeSlots(pgIdx));
    return storage.allocateMultiple(pgIdx, numSlots);
}

Address StorageManager::slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, PoolIndex poolIdx) const
{
    return mLevels.at(level).storage->slotAddress(pgIdx, slotId).at(poolIdx);
}

void StorageManager::copySlotData(LifeCycleId lifeCycle, CacheLevel dstLevel, CacheLevel srcLevel, SlotId dstSlotId,
    SlotId srcSlotId, CUstream stream)
{
    PoolGroupIndex const srcPgIdx = getPoolGroupIndex(srcLevel, lifeCycle);
    PoolGroupIndex const dstPgIdx = getPoolGroupIndex(dstLevel, lifeCycle);
    CacheTier const srcTier = cacheTier(srcLevel);
    CacheTier const dstTier = cacheTier(dstLevel);
    bool const srcIsHot = srcLevel == kGpuLevel;
    bool const dstIsHot = dstLevel == kGpuLevel;

    if (srcIsHot && dstIsHot)
    {
        auto const srcSizes = slotSize(srcLevel, srcPgIdx);
        TLLM_CHECK_DEBUG(srcSizes == slotSize(dstLevel, dstPgIdx));
        for (PoolIndex poolIdx{0}; poolIdx < srcSizes.size(); ++poolIdx)
        {
            mCopyEngine->transfer(dstTier, srcTier, srcSizes.at(poolIdx),
                {{slotAddress(dstLevel, dstPgIdx, dstSlotId, poolIdx),
                    slotAddress(srcLevel, srcPgIdx, srcSlotId, poolIdx)}},
                stream);
        }
        return;
    }

    if (!srcIsHot && !dstIsHot)
    {
        size_t const coldPageBytes = slotSize(srcLevel, srcPgIdx).at(PoolIndex{0});
        TLLM_CHECK_DEBUG(coldPageBytes == slotSize(dstLevel, dstPgIdx).at(PoolIndex{0}));
        mCopyEngine->transfer(dstTier, srcTier, coldPageBytes,
            {{slotAddress(dstLevel, dstPgIdx, dstSlotId, PoolIndex{0}),
                slotAddress(srcLevel, srcPgIdx, srcSlotId, PoolIndex{0})}},
            stream);
        return;
    }

    CacheLevel const coldLevel = srcIsHot ? dstLevel : srcLevel;
    PoolGroupIndex const coldPgIdx = srcIsHot ? dstPgIdx : srcPgIdx;
    CacheTier const coldTier = srcIsHot ? dstTier : srcTier;
    size_t const coldPageBytes = slotSize(coldLevel, coldPgIdx).at(PoolIndex{0});
    int32_t const srcPageIndex = slotIdToPageIndexValue(srcSlotId);
    int32_t const dstPageIndex = slotIdToPageIndexValue(dstSlotId);
    LayerGroupId const batchingLayerGroupId = mBatchingLayerGroupIds.at(lifeCycle);
    PageIndexLocation const pageIndexLocation = mPageIndexLocations.at(lifeCycle);
    auto encodeOne = [this, batchingLayerGroupId, pageIndexLocation](
                         void* dstBasePtr, PageIndexPair const& pageIndex, CUstream stream)
    {
        return submitColdPageCodec(pageIndexLocation, &pageIndex, 1, stream,
            [this, batchingLayerGroupId, dstBasePtr](PageIndexPair const* indices, size_t numPages, CUstream stream)
            {
                return mColdPageCodec->encode(
                    batchingLayerGroupId, dstBasePtr, indices, numPages, reinterpret_cast<cudaStream_t>(stream));
            });
    };
    auto decodeOne = [this, batchingLayerGroupId, pageIndexLocation](
                         void const* srcBasePtr, PageIndexPair const& pageIndex, CUstream stream)
    {
        return submitColdPageCodec(pageIndexLocation, &pageIndex, 1, stream,
            [this, batchingLayerGroupId, srcBasePtr](PageIndexPair const* indices, size_t numPages, CUstream stream)
            {
                return mColdPageCodec->decode(
                    batchingLayerGroupId, srcBasePtr, indices, numPages, reinterpret_cast<cudaStream_t>(stream));
            });
    };

    if (isGpuAccessibleMemory(coldTier))
    {
        MemAddress const coldBase = mLevels.at(coldLevel).storage->getBaseAddress(coldPgIdx, PoolIndex{0}, SlotId{0});
        PageIndexPair const pageIndex{dstPageIndex, srcPageIndex};
        bool const submitted = srcIsHot ? encodeOne(reinterpret_cast<void*>(coldBase), pageIndex, stream)
                                        : decodeOne(reinterpret_cast<void const*>(coldBase), pageIndex, stream);
        if (!submitted)
        {
            throw LogicError("Cold-page codec rejected a single-slot GPU-accessible transfer");
        }
        return;
    }

    TLLM_CHECK_WITH_INFO(coldTier == CacheTier::DISK, "Unsupported cold cache tier");
    auto staging = mPageStagingManager->acquire(coldPageBytes, coldPageBytes, coldPageBytes, 1, stream);
    int32_t const stagingIndex = 0;
    Address const stagingAddress{std::in_place_type<MemAddress>, static_cast<MemAddress>(staging.address())};
    if (srcIsHot)
    {
        PageIndexPair const pageIndex{stagingIndex, srcPageIndex};
        if (!encodeOne(reinterpret_cast<void*>(staging.address()), pageIndex, stream))
        {
            throw LogicError("Cold-page codec rejected a single-slot GPU-to-disk transfer");
        }
        mCopyEngine->transfer(CacheTier::DISK, CacheTier::HOST_MEM, coldPageBytes,
            {{slotAddress(dstLevel, dstPgIdx, dstSlotId, PoolIndex{0}), stagingAddress}}, stream);
    }
    else
    {
        mCopyEngine->transfer(CacheTier::HOST_MEM, CacheTier::DISK, coldPageBytes,
            {{stagingAddress, slotAddress(srcLevel, srcPgIdx, srcSlotId, PoolIndex{0})}}, stream);
        PageIndexPair const pageIndex{dstPageIndex, stagingIndex};
        if (!decodeOne(reinterpret_cast<void const*>(staging.address()), pageIndex, stream))
        {
            throw LogicError("Cold-page codec rejected a single-slot disk-to-GPU transfer");
        }
    }
}

CacheTier StorageManager::cacheTier(CacheLevel level) const
{
    return mLevels.at(level).cacheTier;
}

void StorageManager::releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot)
{
    PoolGroupIndex pg = getPoolGroupIndex(level, lc);
    mLevels.at(level).storage->release(pg, std::move(slot));
}

// ---------------------------------------------------------------------------
// isEvictable
// ---------------------------------------------------------------------------

bool StorageManager::isEvictable(Page const& page, std::optional<CacheLevel> level) const noexcept
{
    PageStatus s = page.status();
    CacheLevel lvl = level.value_or(page.cacheLevel);
    return (s == PageStatus::DROPPABLE && page.isCommitted()) || (s == PageStatus::HELD && lvl < numCacheLevels() - 1);
}

// ---------------------------------------------------------------------------
// scheduleForEviction / excludeFromEviction
// ---------------------------------------------------------------------------

void StorageManager::scheduleForEviction(Page& page)
{
    if (isEvictable(page))
        mLevels.at(page.cacheLevel).controller.scheduleForEviction(page);
}

void StorageManager::excludeFromEviction(Page& page)
{
    TLLM_CHECK_DEBUG(page.nodeRef.has_value());
    mLevels.at(page.cacheLevel).controller.remove(*page.nodeRef);
}

// ---------------------------------------------------------------------------
// prepareFreeSlots
// ---------------------------------------------------------------------------

void StorageManager::prepareFreeSlots(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& requirements,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(lvl), 0);
    }
    for (PoolGroupIndex pgIdx{0}; pgIdx < requirements.size(); ++pgIdx)
    {
        goals.at(level).at(pgIdx) = requirements.at(pgIdx);
    }

    TypedVec<LifeCycleId, std::vector<SharedPtr<Page>>> fallenPages(numLifeCycles());
    _prepareFreeSlots(goals, level, fallenPages, migrationRecorder, dropRecorder);
}

void StorageManager::forceEvict(
    CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& minNumPages, DropRecorder const& dropRecorder)
{
    auto evicted = mLevels.at(level).controller.evict(minNumPages);

    if (isLastLevel(level))
    {
        // Last level: all evicted pages must be DROPPABLE (they get dropped, not migrated).
        for (auto const& pages : evicted)
        {
            for (auto const& page : pages)
            {
                TLLM_CHECK_DEBUG_WITH_INFO(page->status() == PageStatus::DROPPABLE, "Corrupted eviction controller");
            }
        }
        if (dropRecorder)
        {
            for (auto const& pages : evicted)
            {
                if (!pages.empty())
                {
                    dropRecorder(pages, level);
                }
            }
        }
        return;
    }

    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(lvl), 0);
    }
    CacheLevel nextLvl = level + 1;

    TypedVec<LifeCycleId, std::vector<SharedPtr<Page>>> fallen(numLifeCycles());
    for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
    {
        for (auto& sp : evicted.at(pgIdx))
            fallen.at(sp->lifeCycle).push_back(sp);
    }
    _prepareFreeSlots(goals, nextLvl, fallen, MigrationRecorder{}, dropRecorder);
}

// ---------------------------------------------------------------------------
// _prepareFreeSlots (recursive)
// ---------------------------------------------------------------------------

void StorageManager::_prepareFreeSlots(TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>>& goals,
    CacheLevel lvlId, TypedVec<LifeCycleId, std::vector<SharedPtr<Page>>>& fallenPages,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    if (TLLM_UNLIKELY(gDebug))
    {
        TLLM_CHECK_WITH_INFO(goals.size() == numCacheLevels(), "goals.rows must equal numCacheLevels");
        for (CacheLevel level{0}; level < goals.size(); ++level)
        {
            TLLM_CHECK_DEBUG_WITH_INFO(
                goals[level].size() == numPoolGroups(level), "goals row must match the level's pool groups");
        }
        TLLM_CHECK_DEBUG_WITH_INFO(fallenPages.size() == numLifeCycles(), "fallenPages must be lifecycle-keyed");
    }

    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(fallenPages.begin(), fallenPages.end(),
                                   [lvlId](auto const& pages)
                                   {
                                       return std::all_of(pages.begin(), pages.end(),
                                           [lvlId](auto const& p) { return p->cacheLevel < lvlId; });
                                   }),
        "Fallen pages must come from upper cache levels");

    auto& lvl = mLevels.at(lvlId);
    auto& storage = *lvl.storage;
    auto& ctrl = lvl.controller;
    bool const isLast = isLastLevel(lvlId);

    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> fallenByPoolGroup(numPoolGroups(lvlId));
    TypedVec<PoolGroupIndex, SlotCount> numToEvict(numPoolGroups(lvlId), 0);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> heldPages(numPoolGroups(lvlId));
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> evicted(numPoolGroups(lvlId));
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> acceptedPages(numPoolGroups(lvlId));

    bool completed = false;
    auto rollbackGuard = FuncGuard(
        [this, &completed, &fallenPages, &fallenByPoolGroup, &heldPages, &evicted, &acceptedPages]() noexcept
        {
            if (completed)
            {
                return;
            }

            try
            {
                // Eviction temporarily removes the controller's strong owner. Restore every still-resident evictable
                // Page that has not already been published and scheduled at its destination.
                auto restoreEvictionOwnership = [this](auto const& groupedPages)
                {
                    for (auto const& pages : groupedPages)
                    {
                        for (auto it = pages.rbegin(); it != pages.rend(); ++it)
                        {
                            auto const& page = *it;
                            if (page && !page->scheduledForEviction() && isEvictable(*page))
                            {
                                mLevels.at(page->cacheLevel).controller.scheduleForEviction(*page, /*evictFirst=*/true);
                            }
                        }
                    }
                };
                restoreEvictionOwnership(fallenPages);
                restoreEvictionOwnership(fallenByPoolGroup);
                restoreEvictionOwnership(heldPages);
                restoreEvictionOwnership(evicted);
                restoreEvictionOwnership(acceptedPages);
            }
            catch (...)
            {
                // Losing the controller's only strong Page owner would corrupt the cache. A guard destructor cannot
                // propagate, so fail closed if re-queuing itself cannot complete.
                std::terminate();
            }
        });

    for (LifeCycleId lifeCycle{0}; lifeCycle < fallenPages.size(); ++lifeCycle)
    {
        auto& pages = fallenPages[lifeCycle];
        TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(pages.begin(), pages.end(),
                                       [lifeCycle](auto const& page) { return page->lifeCycle == lifeCycle; }),
            "Fallen page stored under the wrong lifecycle");
        auto& poolGroupPages = fallenByPoolGroup[getPoolGroupIndex(lvlId, lifeCycle)];
        poolGroupPages.insert(poolGroupPages.end(), pages.begin(), pages.end());
        pages.clear();
    }

    for (PoolGroupIndex pgIdx{0}; pgIdx < numToEvict.size(); ++pgIdx)
    {
        SlotCount const goal = goals.at(lvlId).at(pgIdx);
        SlotCount const fallen = slotCountValueFromSize(fallenByPoolGroup.at(pgIdx).size());
        SlotCount const oldFree = storage.numFreeSlots(pgIdx);
        SlotCount const evictableCount = ctrl.numEvictablePages(pgIdx);
        SlotCount const required = goal + fallen;
        SlotCount const shortage = required > oldFree ? required - oldFree : 0;
        numToEvict.at(pgIdx) = std::min(shortage, evictableCount);

        SlotCount fallenHeld = 0;
        if (isLast)
        {
            auto& pages = fallenByPoolGroup.at(pgIdx);
            heldPages.at(pgIdx)
                = stealIf(pages, [](SharedPtr<Page> const& p) { return p->status() == PageStatus::HELD; });
            fallenHeld = slotCountValueFromSize(heldPages.at(pgIdx).size());

            if (fallenHeld > oldFree + evictableCount)
            {
                throw OutOfPagesError(
                    "Too many held pages falling to last-level cache for group " + std::to_string(pgIdx.value()));
            }
        }

        if (oldFree + evictableCount < fallenHeld + goal)
        {
            throw OutOfPagesError("Impossible to meet free-slot goal " + std::to_string(goal) + " for group "
                + std::to_string(pgIdx.value()));
        }
    }

    evicted = ctrl.evict(numToEvict);

    if (isLast)
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
        {
            auto& ev = evicted.at(pgIdx);
            SlotCount const oldFree = storage.numFreeSlots(pgIdx);
            SlotCount const numEvicted = slotCountValueFromSize(ev.size());
            TLLM_CHECK_DEBUG_WITH_INFO(
                std::all_of(ev.begin(), ev.end(), [](auto const& p) { return p->status() == PageStatus::DROPPABLE; }),
                "Evicted page at last level must be DROPPABLE");
            if (dropRecorder && !ev.empty())
            {
                dropRecorder(ev, lvlId);
            }
            ev.clear();

            SlotCount const newFree = storage.numFreeSlots(pgIdx);
            TLLM_CHECK_DEBUG(newFree >= numEvicted + oldFree);
            TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(heldPages.at(pgIdx).size()) <= newFree,
                "held_pages count exceeds new free slot count");

            auto& hp = heldPages.at(pgIdx);
            auto& fp = fallenByPoolGroup.at(pgIdx);
            fp.insert(fp.end(), hp.begin(), hp.end());
            hp.clear();

            SlotCount const goal = goals.at(lvlId).at(pgIdx);
            SlotCount const freeAfterGoal = newFree > goal ? newFree - goal : 0;
            SlotCount const numAccepted = std::min(freeAfterGoal, slotCountValueFromSize(fp.size()));
            if (numAccepted > 0)
            {
                acceptedPages.at(pgIdx).assign(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
            }
            fp.clear();
        }
    }
    else
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
        {
            auto& ev = evicted.at(pgIdx);
            SlotCount const oldFree = storage.numFreeSlots(pgIdx);
            SlotCount const numEvicted = slotCountValueFromSize(ev.size());
            auto& fp = fallenByPoolGroup.at(pgIdx);
            fp.insert(fp.begin(), ev.begin(), ev.end());
            ev.clear();

            SlotCount const goal = goals.at(lvlId).at(pgIdx);
            SlotCount const availableAfterGoal = oldFree + numEvicted > goal ? oldFree + numEvicted - goal : 0;
            SlotCount const numAccepted = std::min(availableAfterGoal, slotCountValueFromSize(fp.size()));
            if (numAccepted > 0)
            {
                acceptedPages.at(pgIdx).assign(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
                fp.erase(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
            }

            for (auto& page : fp)
            {
                fallenPages.at(page->lifeCycle).push_back(std::move(page));
            }
            fp.clear();
        }
    }

    if (!isLast)
    {
        _prepareFreeSlots(goals, lvlId + 1, fallenPages, migrationRecorder, dropRecorder);
    }

    TLLM_CHECK_DEBUG_WITH_INFO(
        std::all_of(fallenPages.begin(), fallenPages.end(), [](auto const& fp) { return fp.empty(); }),
        "All fallen pages must be consumed after level loop");

    for (auto& poolGroupPages : acceptedPages)
    {
        auto byMigrationPath = partition(poolGroupPages,
            [this, lvlId](SharedPtr<Page> const& page)
            {
                return std::tuple{page->cacheLevel, getPoolGroupIndex(page->cacheLevel, page->lifeCycle),
                    getPoolGroupIndex(lvlId, page->lifeCycle),
                    getMigrationBatchingLayerGroupId(lvlId, page->cacheLevel, page->lifeCycle)};
            });
        for (auto& [migrationPath, pages] : byMigrationPath)
        {
            CacheLevel const srcLevel = std::get<0>(migrationPath);
            _batchedMigrate(lvlId, srcLevel, pages, /*updateSrc=*/true, migrationRecorder);
            for (auto const& page : pages)
            {
                if (!isLast || page->status() != PageStatus::HELD)
                {
                    lvl.controller.scheduleForEviction(*page);
                }
            }
        }
    }
    completed = true;
}

// ---------------------------------------------------------------------------
// _batchedMigrate
// ---------------------------------------------------------------------------

LayerGroupId StorageManager::getMigrationBatchingLayerGroupId(
    CacheLevel dstLevel, CacheLevel srcLevel, LifeCycleId lifeCycle) const
{
    bool const srcIsHot = srcLevel == kGpuLevel;
    bool const dstIsHot = dstLevel == kGpuLevel;
    return srcIsHot != dstIsHot ? mBatchingLayerGroupIds.at(lifeCycle) : LayerGroupId{-1};
}

void StorageManager::_batchedMigrate(CacheLevel dstLevel, CacheLevel srcLevel,
    std::vector<SharedPtr<Page>> const& srcPages, bool updateSrc, MigrationRecorder const& migrationRecorder,
    bool defrag)
{
    TLLM_CHECK_DEBUG(defrag || dstLevel != srcLevel);
    if (srcPages.empty())
    {
        return;
    }

    SlotCount const numSlots = slotCountValueFromSize(srcPages.size());
    LifeCycleId const firstLifeCycle = srcPages.front()->lifeCycle;
    PoolGroupIndex const srcPgIdx = getPoolGroupIndex(srcLevel, firstLifeCycle);
    PoolGroupIndex const dstPgIdx = getPoolGroupIndex(dstLevel, firstLifeCycle);
    LayerGroupId const batchingLayerGroupId = getMigrationBatchingLayerGroupId(dstLevel, srcLevel, firstLifeCycle);
    TLLM_CHECK_DEBUG(std::all_of(srcPages.begin(), srcPages.end(),
        [this, srcLevel, dstLevel, srcPgIdx, dstPgIdx, batchingLayerGroupId](auto const& page)
        {
            return getPoolGroupIndex(srcLevel, page->lifeCycle) == srcPgIdx
                && getPoolGroupIndex(dstLevel, page->lifeCycle) == dstPgIdx
                && getMigrationBatchingLayerGroupId(dstLevel, srcLevel, page->lifeCycle) == batchingLayerGroupId;
        }));
    auto& srcPoolGroup = poolGroup(srcLevel, srcPgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, dstPgIdx);

    if (dstPoolGroup.numFreeSlots() < numSlots)
        throw OutOfPagesError("Not enough free slots for migration");

    auto dstSlots = dstPoolGroup.allocateMultiple(numSlots);
    // A15: allocated slot count must match the request.
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(dstSlots.size()) == numSlots, "dst_slots size mismatch");
    try
    {
        CacheTier const dstTier = mLevels.at(dstLevel).cacheTier;
        CacheTier const srcTier = mLevels.at(srcLevel).cacheTier;

        thread_local std::vector<PageIndexPair> pageIndices;
        thread_local std::vector<PageIndexPair> stagingPageIndices;
        pageIndices.clear();
        stagingPageIndices.clear();
        pageIndices.reserve(srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            TLLM_CHECK_DEBUG(defrag || !srcPages.at(i)->scheduledForEviction());
            pageIndices.push_back(PageIndexPair{
                slotIdToPageIndexValue(dstSlots.at(i).slotId()), slotIdToPageIndexValue(srcPages.at(i)->slotId())});
        }

        std::vector<CachedCudaEvent const*> priorEvents;
        priorEvents.reserve(2 * srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            priorEvents.push_back(&srcPages.at(i)->readyEvent);
            priorEvents.push_back(&dstSlots.at(i).readyEvent);
        }

        TemporaryCudaStream tempStream(priorEvents);
        {
            auto scope = tempStream.enter();
            CUstream const stream = tempStream.get();
            int const exceptionCount = std::uncaught_exceptions();
            auto fenceOnFailure = FuncGuard(
                [&srcPages, &dstSlots, stream, exceptionCount]() noexcept
                {
                    if (std::uncaught_exceptions() == exceptionCount)
                    {
                        return;
                    }
                    // A codec or copy engine may enqueue asynchronous work before reporting failure. Fence both
                    // owners before the outer catch returns destination Slots to their allocator.
                    CachedCudaEvent completion(reinterpret_cast<CudaStream>(stream));
                    for (std::size_t i = 0; i < srcPages.size(); ++i)
                    {
                        srcPages.at(i)->readyEvent = completion;
                        dstSlots.at(i).readyEvent = completion;
                    }
                });
            bool const srcIsHot = srcLevel == kGpuLevel;
            bool const dstIsHot = dstLevel == kGpuLevel;

            if (srcIsHot && dstIsHot)
            {
                PoolIndex const poolCount = numPools(srcLevel, srcPgIdx);
                TLLM_CHECK_DEBUG(poolCount == numPools(dstLevel, dstPgIdx));
                auto const slotSizes = slotSize(srcLevel, srcPgIdx);
                TLLM_CHECK_DEBUG(slotSizes == slotSize(dstLevel, dstPgIdx));
                for (PoolIndex poolIdx{0}; poolIdx < poolCount; ++poolIdx)
                {
                    std::vector<CopyTask> tasks;
                    tasks.reserve(srcPages.size());
                    for (std::size_t i = 0; i < srcPages.size(); ++i)
                    {
                        tasks.push_back({dstPoolGroup.slotAddress(dstSlots.at(i).slotId()).at(poolIdx),
                            srcPoolGroup.slotAddress(srcPages.at(i)->slotId()).at(poolIdx)});
                    }
                    mCopyEngine->transfer(dstTier, srcTier, slotSizes.at(poolIdx), tasks, stream);
                }
            }
            else if (!srcIsHot && !dstIsHot)
            {
                TLLM_CHECK_DEBUG(numPools(srcLevel, srcPgIdx) == PoolIndex{1});
                TLLM_CHECK_DEBUG(numPools(dstLevel, dstPgIdx) == PoolIndex{1});
                size_t const coldPageBytes = slotSize(srcLevel, srcPgIdx).at(PoolIndex{0});
                TLLM_CHECK_DEBUG(coldPageBytes == slotSize(dstLevel, dstPgIdx).at(PoolIndex{0}));
                std::vector<CopyTask> tasks;
                tasks.reserve(srcPages.size());
                for (std::size_t i = 0; i < srcPages.size(); ++i)
                {
                    tasks.push_back({dstPoolGroup.slotAddress(dstSlots.at(i).slotId()).at(PoolIndex{0}),
                        srcPoolGroup.slotAddress(srcPages.at(i)->slotId()).at(PoolIndex{0})});
                }
                mCopyEngine->transfer(dstTier, srcTier, coldPageBytes, tasks, stream);
            }
            else
            {
                // LayerGroupId{-1} is only a grouping sentinel for same-representation copies. Codec metadata exists
                // exclusively for hot-to-cold and cold-to-hot conversions, so query it only in this branch.
                PageIndexLocation const pageIndexLocation = mPageIndexLocations.at(batchingLayerGroupId);
                auto encodeBatch = [this, batchingLayerGroupId, pageIndexLocation](void* dstBasePtr,
                                       PageIndexPair const* indices, size_t numPages, CUstream codecStream)
                {
                    return submitColdPageCodec(pageIndexLocation, indices, numPages, codecStream,
                        [this, batchingLayerGroupId, dstBasePtr](
                            PageIndexPair const* submittedIndices, size_t submittedPages, CUstream submittedStream)
                        {
                            return mColdPageCodec->encode(batchingLayerGroupId, dstBasePtr, submittedIndices,
                                submittedPages, reinterpret_cast<cudaStream_t>(submittedStream));
                        });
                };
                auto decodeBatch = [this, batchingLayerGroupId, pageIndexLocation](void const* srcBasePtr,
                                       PageIndexPair const* indices, size_t numPages, CUstream codecStream)
                {
                    return submitColdPageCodec(pageIndexLocation, indices, numPages, codecStream,
                        [this, batchingLayerGroupId, srcBasePtr](
                            PageIndexPair const* submittedIndices, size_t submittedPages, CUstream submittedStream)
                        {
                            return mColdPageCodec->decode(batchingLayerGroupId, srcBasePtr, submittedIndices,
                                submittedPages, reinterpret_cast<cudaStream_t>(submittedStream));
                        });
                };

                CacheLevel const coldLevel = srcIsHot ? dstLevel : srcLevel;
                PoolGroupIndex const coldPgIdx = srcIsHot ? dstPgIdx : srcPgIdx;
                size_t const coldPageBytes = slotSize(coldLevel, coldPgIdx).at(PoolIndex{0});
                CacheTier const coldTier = srcIsHot ? dstTier : srcTier;

                if (isGpuAccessibleMemory(coldTier))
                {
                    MemAddress const coldBase
                        = mLevels.at(coldLevel).storage->getBaseAddress(coldPgIdx, PoolIndex{0}, SlotId{0});
                    bool const submitted = srcIsHot
                        ? encodeBatch(reinterpret_cast<void*>(coldBase), pageIndices.data(), pageIndices.size(), stream)
                        : decodeBatch(
                              reinterpret_cast<void const*>(coldBase), pageIndices.data(), pageIndices.size(), stream);
                    if (!submitted)
                    {
                        throw LogicError("Cold-page codec rejected a GPU-accessible migration batch");
                    }
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(coldTier == CacheTier::DISK, "Unsupported cold cache tier");
                    size_t remaining = srcPages.size();
                    size_t offset = 0;
                    while (remaining > 0)
                    {
                        size_t const maxStagingBytes = remaining > std::numeric_limits<size_t>::max() / coldPageBytes
                            ? std::numeric_limits<size_t>::max()
                            : coldPageBytes * remaining;
                        auto staging
                            = mPageStagingManager->acquire(coldPageBytes, maxStagingBytes, coldPageBytes, 1, stream);
                        size_t const batchSize = std::min(remaining, staging.size() / coldPageBytes);
                        stagingPageIndices.resize(batchSize);

                        if (srcIsHot)
                        {
                            for (size_t i = 0; i < batchSize; ++i)
                            {
                                stagingPageIndices[i]
                                    = PageIndexPair{static_cast<int32_t>(i), pageIndices[offset + i].src};
                            }
                            bool const submitted = encodeBatch(reinterpret_cast<void*>(staging.address()),
                                stagingPageIndices.data(), batchSize, stream);
                            if (!submitted)
                            {
                                throw LogicError("Cold-page codec rejected a GPU-to-disk migration batch");
                            }

                            std::vector<CopyTask> tasks;
                            tasks.reserve(batchSize);
                            for (size_t i = 0; i < batchSize; ++i)
                            {
                                tasks.push_back(
                                    {dstPoolGroup.slotAddress(dstSlots.at(offset + i).slotId()).at(PoolIndex{0}),
                                        Address{
                                            std::in_place_type<MemAddress>, staging.address() + i * coldPageBytes}});
                            }
                            mCopyEngine->transfer(CacheTier::DISK, CacheTier::HOST_MEM, coldPageBytes, tasks, stream);
                        }
                        else
                        {
                            std::vector<CopyTask> tasks;
                            tasks.reserve(batchSize);
                            for (size_t i = 0; i < batchSize; ++i)
                            {
                                tasks.push_back(
                                    {Address{std::in_place_type<MemAddress>, staging.address() + i * coldPageBytes},
                                        srcPoolGroup.slotAddress(srcPages.at(offset + i)->slotId()).at(PoolIndex{0})});
                            }
                            mCopyEngine->transfer(CacheTier::HOST_MEM, CacheTier::DISK, coldPageBytes, tasks, stream);
                            for (size_t i = 0; i < batchSize; ++i)
                            {
                                stagingPageIndices[i]
                                    = PageIndexPair{pageIndices[offset + i].dst, static_cast<int32_t>(i)};
                            }
                            bool const submitted = decodeBatch(reinterpret_cast<void const*>(staging.address()),
                                stagingPageIndices.data(), batchSize, stream);
                            if (!submitted)
                            {
                                throw LogicError("Cold-page codec rejected a disk-to-GPU migration batch");
                            }
                        }

                        offset += batchSize;
                        remaining -= batchSize;
                    }
                }
            }
        } // ~Scope records finish event

        CachedCudaEvent finishEvent = tempStream.takeFinishEvent();
        // From this point on, every owner carries the completion fence. This also protects rollback if a recorder or
        // later bookkeeping step throws after the asynchronous migration was successfully submitted.
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            srcPages.at(i)->readyEvent = finishEvent;
            dstSlots.at(i).readyEvent = finishEvent;
        }

        constexpr size_t kMaxRetainedPageIndexPairs = (1u << 20u) / sizeof(PageIndexPair);
        if (pageIndices.capacity() > kMaxRetainedPageIndexPairs)
        {
            std::vector<PageIndexPair>().swap(pageIndices);
        }
        if (stagingPageIndices.capacity() > kMaxRetainedPageIndexPairs)
        {
            std::vector<PageIndexPair>().swap(stagingPageIndices);
        }

        if (migrationRecorder && !defrag)
        {
            migrationRecorder(srcPages, dstSlots, srcLevel, dstLevel);
        }
        std::set<std::pair<std::string, int>> emittedCacheLevelUpdates;
        bool const emitCacheLevelUpdates
            = updateSrc && !defrag && srcLevel != dstLevel && static_cast<bool>(mEventSink);
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            if (updateSrc)
            {
                bool wasScheduled = srcPages.at(i)->scheduledForEviction();
                if (wasScheduled)
                    excludeFromEviction(*srcPages.at(i));
                // Extract source slot from the page and release it back to the pool.
                Slot srcSlot;
                srcSlot.setSlotId(srcPages.at(i)->slotId()); // asserts valid
                srcSlot.readyEvent = finishEvent;
                srcPages.at(i)->resetSlot();
                srcPoolGroup.release(std::move(srcSlot));
                // Transfer dst slot ownership to the page.
                srcPages.at(i)->setSlot(dstSlots.at(i));
                srcPages.at(i)->cacheLevel = dstLevel;
                if (emitCacheLevelUpdates && srcPages.at(i)->isCommitted())
                {
                    auto const& page = static_cast<CommittedPage const&>(*srcPages.at(i));
                    Block const* block = page.block;
                    std::string const blockKey = block
                        ? std::string(reinterpret_cast<char const*>(block->key.data()), block->key.size())
                        : std::string{};
                    if (block && !block->isOrphan()
                        && emittedCacheLevelUpdates.insert({blockKey, page.lifeCycle.value()}).second)
                    {
                        mEventSink->addCacheLevelUpdated(block->key, srcLevel, dstLevel, page.lifeCycle);
                    }
                }
                if (wasScheduled)
                    scheduleForEviction(*srcPages.at(i));
            }
        }
    }
    catch (...)
    {
        for (auto& s : dstSlots)
            dstPoolGroup.release(std::move(s));
        throw;
    }
}

// ---------------------------------------------------------------------------
// batchedMigrateToGpu
// ---------------------------------------------------------------------------

void StorageManager::batchedMigrateToGpu(
    std::vector<BatchedLockTarget> const& targets, KvCache& /*kvCache*/, MigrationRecorder const& migrationRecorder)
{
    using MigrationPath = std::tuple<CacheLevel, PoolGroupIndex, PoolGroupIndex, LayerGroupId>;
    std::map<MigrationPath, std::vector<SharedPtr<Page>>> groups;
    for (auto const& t : targets)
    {
        if (t.page->cacheLevel == kGpuLevel)
        {
            continue;
        }
        CacheLevel const srcLevel = t.page->cacheLevel;
        groups[{srcLevel, getPoolGroupIndex(srcLevel, t.lifeCycle), getPoolGroupIndex(kGpuLevel, t.lifeCycle),
                   getMigrationBatchingLayerGroupId(kGpuLevel, srcLevel, t.lifeCycle)}]
            .push_back(t.page);
    }
    for (auto& [key, pages] : groups)
    {
        _batchedMigrate(kGpuLevel, std::get<0>(key), pages, /*updateSrc=*/true, migrationRecorder);
    }
}

void StorageManager::prefetch(
    CacheLevel dstLevel, TypedVec<LifeCycleId, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> const& pages)
{
    TypedVec<PoolGroupIndex, SlotCount> numSlotsToMigrate(numPoolGroups(dstLevel), 0);
    std::vector<SharedPtr<Page>> scheduled;

    struct ReschedulePagesGuard
    {
        StorageManager& storageManager;
        std::vector<SharedPtr<Page>>& scheduled;

        ~ReschedulePagesGuard()
        {
            for (auto const& page : scheduled)
            {
                storageManager.scheduleForEviction(*page);
            }
            scheduled.clear();
        }
    } reschedulePagesGuard{*this, scheduled};

    for (LifeCycleId lifeCycle{0}; lifeCycle < pages.size(); ++lifeCycle)
    {
        auto const& lifeCyclePages = pages.at(lifeCycle);
        for (CacheLevel level{0}; level < lifeCyclePages.size(); ++level)
        {
            auto const& levelPages = lifeCyclePages.at(level);
            TLLM_CHECK_DEBUG(level >= dstLevel || levelPages.empty());
            TLLM_CHECK_DEBUG(std::all_of(levelPages.begin(), levelPages.end(),
                [lifeCycle](auto const& page) { return page->lifeCycle == lifeCycle; }));
            for (auto const& page : levelPages)
            {
                if (page->scheduledForEviction())
                {
                    excludeFromEviction(*page);
                    scheduled.push_back(page);
                }
                else if (isEvictable(*page, dstLevel))
                {
                    scheduled.push_back(page);
                }
                if (level != dstLevel)
                {
                    auto const dstPgIdx = getPoolGroupIndex(dstLevel, lifeCycle);
                    ++numSlotsToMigrate.at(dstPgIdx);
                }
            }
        }
    }

    prepareFreeSlots(dstLevel, numSlotsToMigrate);
    using MigrationPath = std::tuple<CacheLevel, PoolGroupIndex, PoolGroupIndex, LayerGroupId>;
    std::map<MigrationPath, std::vector<SharedPtr<Page>>> migrationGroups;
    for (LifeCycleId lifeCycle{0}; lifeCycle < pages.size(); ++lifeCycle)
    {
        auto const& lifeCyclePages = pages.at(lifeCycle);
        for (CacheLevel level = dstLevel + 1; level < numCacheLevels(); ++level)
        {
            auto const& levelPages = lifeCyclePages.at(level);
            auto& group = migrationGroups[{level, getPoolGroupIndex(level, lifeCycle),
                getPoolGroupIndex(dstLevel, lifeCycle), getMigrationBatchingLayerGroupId(dstLevel, level, lifeCycle)}];
            group.insert(group.end(), levelPages.begin(), levelPages.end());
        }
    }
    for (auto& [migrationPath, migrationPages] : migrationGroups)
    {
        _batchedMigrate(dstLevel, std::get<0>(migrationPath), migrationPages, /*updateSrc=*/true);
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

LifeCycle const& StorageManager::getLifeCycle(LifeCycleId lc) const
{
    return mLifeCycles[lc];
}

PoolGroupIndex StorageManager::getPoolGroupIndex(CacheLevel level, LifeCycleId lc) const
{
    return lifeCycleGrouping(level).at(lc);
}

PoolGroupIndex StorageManager::getPoolGroupIndex(LifeCycleId lc) const
{
    return getPoolGroupIndex(kGpuLevel, lc);
}

PoolIndex StorageManager::numPools(CacheLevel level, PoolGroupIndex pgIdx) const
{
    return mLevels.at(level).storage->numPools(pgIdx);
}

PoolIndex StorageManager::numPools(PoolGroupIndex pgIdx) const
{
    return numPools(kGpuLevel, pgIdx);
}

TypedVec<PoolIndex, size_t> StorageManager::slotSize(CacheLevel level, PoolGroupIndex pgIdx) const
{
    return slotDescList(level).at(pgIdx).slotSizeList();
}

TypedVec<PoolIndex, size_t> StorageManager::slotSize(PoolGroupIndex pgIdx) const
{
    return slotSize(kGpuLevel, pgIdx);
}

PoolGroupBase& StorageManager::poolGroup(CacheLevel lvl, PoolGroupIndex pgIdx)
{
    return mLevels.at(lvl).storage->poolGroup(pgIdx);
}

MemAddress StorageManager::getMemPoolBaseAddress(LayerId layerId, DataRole role) const
{
    auto it = mBufferAttr.find(BufferId{layerId, role});
    if (it == mBufferAttr.end())
        throw std::out_of_range("Unknown BufferId");
    auto const& attr = it->second;
    PoolGroupIndex pgIdx = getPoolGroupIndex(kGpuLevel, attr.lifeCycleId);
    return mLevels[kGpuLevel].storage->getBaseAddress(pgIdx, attr.poolIndex, SlotId{0}) + attr.offset;
}

MemAddress StorageManager::getMemPoolBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    return mLevels[kGpuLevel].storage->getBaseAddress(pgIdx, poolIdx, SlotId{0});
}

LayerAttr const& StorageManager::getLayerAttr(LayerId layerId) const
{
    auto it = mLayerAttributes.find(layerId);
    if (it == mLayerAttributes.end())
        throw std::out_of_range("Unknown LayerId for LayerAttr");
    return it->second;
}

SlotCount StorageManager::numSlots(PoolGroupIndex pgIdx, CacheLevel level) const
{
    return mLevels.at(level).storage->numSlots(pgIdx);
}

StorageStatistics StorageManager::getStatistics(CacheLevel level, PoolGroupIndex pgIdx) const
{
    auto const& lvl = mLevels.at(level);
    SlotCount freeSlots = lvl.storage->numFreeSlots(pgIdx);
    SlotCount totalSlots = lvl.storage->numSlots(pgIdx);
    SlotCount evictable = lvl.controller.numEvictablePages(pgIdx);
    auto sizes = lvl.storage->slotSize(pgIdx);
    return StorageStatistics{sizes, totalSlots, freeSlots, evictable};
}

TypedVec<PoolGroupIndex, float> StorageManager::getUtilization(CacheLevel level) const
{
    TypedVec<PoolGroupIndex, float> result;
    result.reserve(numPoolGroups(level));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(level); ++pgIdx)
    {
        auto const s = getStatistics(level, pgIdx);
        TLLM_CHECK_DEBUG(s.total > 0);
        result.push_back(static_cast<float>(s.unavailable()) / static_cast<float>(s.total));
    }
    return result;
}

float StorageManager::getOverallUtilization(CacheLevel level) const
{
    float num = 0.f, den = 0.f;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(level); ++pgIdx)
    {
        auto s = getStatistics(level, pgIdx);
        float sz = 0.f;
        for (auto v : s.slotSizes)
            sz += static_cast<float>(v);
        num += sz * static_cast<float>(s.unavailable());
        den += sz * static_cast<float>(s.total);
    }
    TLLM_CHECK_DEBUG(den > 0.f);
    return num / den;
}

// ---------------------------------------------------------------------------
// expandPoolGroup
// ---------------------------------------------------------------------------

void StorageManager::expandPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots)
{
    auto& pg = poolGroup(level, pgIdx);
    TLLM_CHECK_DEBUG(newNumSlots > pg.numSlots());
    pg.resizePools(newNumSlots);
    pg.slotAllocator().expand(newNumSlots);
}

// ---------------------------------------------------------------------------
// shrinkPoolGroup — mirrors Python _storage_manager.py::shrink_pool_group
// ---------------------------------------------------------------------------

void StorageManager::shrinkPoolGroup(
    CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots, std::vector<SharedPtr<Page>> const& persistentPages)
{
    auto& pg = poolGroup(level, pgIdx);
    auto& allocator = pg.slotAllocator();
    auto& ctrl = mLevels.at(level).controller;
    TLLM_CHECK_DEBUG(newNumSlots < pg.numSlots());

    // A16: persistent_pages preconditions.
    TLLM_CHECK_DEBUG_WITH_INFO(
        persistentPages.size() <= slotCountToSizeT(newNumSlots), "Not enough slots to hold all persistent pages");
    TLLM_CHECK_DEBUG_WITH_INFO(
        std::all_of(persistentPages.begin(), persistentPages.end(), [this, level, pgIdx](auto const& p)
            { return p->cacheLevel == level && getPoolGroupIndex(level, p->lifeCycle) == pgIdx; }),
        "Persistent page cache level or pool group mismatch");

    // Fast path: when no slot id has ever been issued in the to-be-removed
    // range [newNumSlots, capacity), there is nothing to migrate.
    // numActiveSlots() is a monotone high-water mark of issued ids.
    if (allocator.numActiveSlots() <= newNumSlots)
    {
        allocator.prepareForShrink(newNumSlots);
        allocator.finishShrink();
        pg.resizePools(newNumSlots);
        return;
    }

    // Find overflow pages: scheduled pages with slot_id >= newNumSlots.
    auto gen = ctrl.pageGenerator(pgIdx);
    std::deque<std::pair<SlotCount, SharedPtr<Page>>> overflowSlots;
    {
        SlotCount idx = 0;
        while (auto const* page = gen())
        {
            if ((*page)->slotId() >= newNumSlots)
                overflowSlots.emplace_back(idx, *page);
            ++idx;
        }
    }

    // Persistent pages in overflow range.
    std::vector<SharedPtr<Page>> overflowPersistent;
    for (auto const& p : persistentPages)
    {
        if (p->slotId() >= newNumSlots)
            overflowPersistent.push_back(p);
    }
    SlotCount numOverflowPersistent = slotCountValueFromSize(overflowPersistent.size());

    // A2: RUNTIME check — persistent overflow pages must fit in the new capacity.
    if (numOverflowPersistent > newNumSlots)
    {
        throw OutOfPagesError("Not enough slots to hold all persistent pages");
    }

    // Mark the allocator for shrink.
    allocator.prepareForShrink(newNumSlots);

    // Calculate minimum number of lowest-priority pages to evict.
    // Need numEvictedOverflowSlots because evicted overflow pages won't become free,
    // because only free non-overflow slots can be used for defragmentation.
    SlotCount minNumEvicted = 0;
    SlotCount numEvictedOverflowSlots = 0;
    while (!overflowSlots.empty()
        && slotCountValueFromSize(overflowSlots.size()) + numOverflowPersistent
            > std::min(newNumSlots, overflowSlots.front().first + allocator.numFreeSlots() - numEvictedOverflowSlots))
    {
        minNumEvicted = overflowSlots.front().first + 1;
        overflowSlots.pop_front();
        ++numEvictedOverflowSlots;
    }

    // Force-evict the required pages.
    TypedVec<PoolGroupIndex, SlotCount> evictReqs(numPoolGroups(level), 0);
    evictReqs[pgIdx] = minNumEvicted;
    forceEvict(level, evictReqs);

    // Remaining overflow pages to defragment.
    std::vector<SharedPtr<Page>> overflowPages;
    overflowPages.reserve(overflowSlots.size() + overflowPersistent.size());
    for (auto& [idx, p] : overflowSlots)
        overflowPages.push_back(p);
    for (auto& p : overflowPersistent)
        overflowPages.push_back(p);

    // Ensure free slots for the overflow pages.
    TypedVec<PoolGroupIndex, SlotCount> reqs(numPoolGroups(level), 0);
    reqs[pgIdx] = slotCountValueFromSize(overflowPages.size());
    prepareFreeSlots(level, reqs);

    // A17: all overflow pages must be at the expected cache level.
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(overflowPages.begin(), overflowPages.end(),
                                   [level](auto const& p) { return p->cacheLevel == level; }),
        "Overflow page cache level mismatch");

    // Defragment: migrate overflow pages to free slots within the same level.
    _batchedMigrate(level, level, overflowPages, /*updateSrc=*/true, MigrationRecorder{}, /*defrag=*/true);

    // A18: post-defrag overflow assertion — overflow slot count matches expectations.
    TLLM_CHECK_DEBUG_WITH_INFO(allocator.numOverflowSlots() == allocator.numActiveSlots() - allocator.targetCapacity(),
        "Post-defrag overflow slot count mismatch");

    // Finalize shrink and resize pools.
    allocator.finishShrink();
    pg.resizePools(newNumSlots);
}

// ---------------------------------------------------------------------------
// adjustCacheLevel — mirrors Python _storage_manager.py::adjust_cache_level
// ---------------------------------------------------------------------------

void StorageManager::adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota,
    TypedVec<PoolGroupIndex, float> const& ratioList,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages)
{
    auto& lvlStorage = *mLevels.at(level).storage;
    auto oldNumSlots = lvlStorage.slotCountList();
    size_t quota = newQuota.has_value()
        ? roundUp(newQuota.value(), static_cast<size_t>(lvlStorage.poolSizeGranularity()))
        : lvlStorage.totalQuota();
    auto const& minSlots = mMinSlotsByLevel.at(level);
    size_t minQuota = minQuotaForLevel(lvlStorage.slotSizeLists(), lvlStorage.poolSizeGranularity(), minSlots);
    if (quota < minQuota)
    {
        throw std::invalid_argument("Quota " + std::to_string(quota)
            + " is insufficient for min_slots constraints (requires at least " + std::to_string(minQuota) + ")");
    }
    auto newNumSlots = lvlStorage.computeSlotCountList(ratioList, minSlots, quota);

    if (!isLastLevel(level))
        TLLM_CHECK_DEBUG(persistentPages == nullptr);

    // Shrink first.
    for (PoolGroupIndex pgIdx{0}; pgIdx < newNumSlots.size(); ++pgIdx)
    {
        if (newNumSlots[pgIdx] >= oldNumSlots[pgIdx])
            continue;
        std::vector<SharedPtr<Page>> pages;
        if (persistentPages)
            pages = (*persistentPages)[pgIdx];
        shrinkPoolGroup(level, pgIdx, newNumSlots[pgIdx], pages);
    }
    // Then expand.
    for (PoolGroupIndex pgIdx{0}; pgIdx < newNumSlots.size(); ++pgIdx)
    {
        if (newNumSlots[pgIdx] <= oldNumSlots[pgIdx])
            continue;
        expandPoolGroup(level, pgIdx, newNumSlots[pgIdx]);
    }
    lvlStorage.postResize();
}

TypedVec<PoolGroupIndex, float> StorageManager::getRatioList(CacheLevel level) const
{
    return mLevels.at(level).storage->ratioList();
}

TypedVec<PoolGroupIndex, float> StorageManager::ratioFromLength(
    int tokensPerBlock, int historyLength, int capacity) const
{
    return ratioFromLength(kGpuLevel, tokensPerBlock, historyLength, capacity);
}

TypedVec<PoolGroupIndex, float> StorageManager::ratioFromLength(
    CacheLevel level, int tokensPerBlock, int historyLength, int capacity) const
{
    if (capacity < historyLength)
    {
        TLLM_LOG_WARNING("Bad sampling for capacity and history_length");
        capacity = historyLength;
    }
    int numBlocks = divUp(capacity, tokensPerBlock);
    TypedVec<PoolGroupIndex, size_t> numBytes(numPoolGroups(level), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    auto const& lifecycles = mLifeCycles.getAll();
    for (LifeCycleId lcId{0}; lcId < lifecycles.size(); ++lcId)
    {
        PoolGroupIndex pgIdx = getPoolGroupIndex(level, lcId);
        auto ss = slotSize(level, pgIdx);
        size_t slotSizeSum = 0;
        for (auto s : ss)
            slotSizeSum += s;
        int numRequiredBlocks;
        if (ssmLcId.has_value() && lcId == *ssmLcId)
        {
            numRequiredBlocks = 1;
        }
        else
        {
            auto stale = getStaleRange(lifecycles[lcId], historyLength, tokensPerBlock);
            numRequiredBlocks = std::max(numBlocks - stale.length(), 1);
        }
        numBytes[pgIdx] += static_cast<size_t>(numRequiredBlocks) * slotSizeSum;
    }
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// ratioFromBatch
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, float> StorageManager::ratioFromBatch(BatchDesc const& batch, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, size_t granularity) const
{
    auto numSlots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
    auto numBytes = slotsToBytes(numSlots, granularity);
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// computeMinSlotsByLifeCycleFromConstraints
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, SlotCount> StorageManager::computeMinSlotsByLifeCycleFromConstraints(
    std::vector<BatchDesc> const& constraints, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, float maxUtilForResume) const
{
    TLLM_CHECK_DEBUG(maxUtilForResume > 0.0f && maxUtilForResume <= 1.0f);
    // All returned elements are positive. Constraint-derived floors include headroom
    // for the utilization gate checked by KvCache::resume.
    TypedVec<LifeCycleId, SlotCount> maxSlots(numLifeCycles(), 0);

    auto swaFloorBlocks = [tokensPerBlock](AttnLifeCycle const& lc) -> int
    {
        int window = *lc.windowSize;
        // Handle oscillation of slot count required by SWA while the window slides.
        return lc.numSinkBlocks + (window + tokensPerBlock - 2) / tokensPerBlock + 1;
    };

    // Full-attention lifecycles share the largest SWA floor: all attention
    // lifecycles see the same seq_len, so this is a valid lower bound.
    int floorNumBlocks = 1;
    for (auto const& [lcId, attn] : mLifeCycles.attentionLifeCycles())
    {
        if (attn->windowSize.has_value())
            floorNumBlocks = std::max(floorNumBlocks, swaFloorBlocks(*attn));
    }
    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        auto const* attn = std::get_if<AttnLifeCycle>(&lc);
        if (attn == nullptr)
        {
            // SSM / non-attention: 1 slot floor per life cycle.
            maxSlots[lcIdx] = 1;
        }
        else if (attn->windowSize.has_value())
        {
            maxSlots[lcIdx] = swaFloorBlocks(*attn);
        }
        else
        {
            maxSlots[lcIdx] = floorNumBlocks;
        }
    }
    for (auto const& batch : constraints)
    {
        auto slots = computeSlotsForBatchByLifeCycle(batch, tokensPerBlock, swaScratchReuse);
        for (LifeCycleId lifeCycle{0}; lifeCycle < slots.size(); ++lifeCycle)
        {
            auto const scaledSlots = static_cast<SlotCount>(
                std::ceil(static_cast<double>(slots[lifeCycle]) / static_cast<double>(maxUtilForResume)));
            maxSlots[lifeCycle] = std::max(maxSlots[lifeCycle], scaledSlots);
        }
    }
    return maxSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsForBatchByLifeCycle
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, SlotCount> StorageManager::computeSlotsForBatchByLifeCycle(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    TypedVec<LifeCycleId, SlotCount> numSlots(numLifeCycles(), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    int sysBlocks = batch.systemPromptLength / tokensPerBlock;

    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            // SSM: always 1 dedicated block per request, never shared.
            numSlots[lcIdx] += slotCountValueFromSize(batch.kvCaches.size());
            continue;
        }
        // Shared sys blocks (counted once): union of non-stale sys blocks across all requests.
        HalfOpenRange<BlockOrdinal> sysRange{0, sysBlocks};
        HalfOpenRange<BlockOrdinal> staleIntersection = sysRange;
        for (auto const& kv : batch.kvCaches)
        {
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            staleIntersection = intersect(staleIntersection, stale);
        }
        numSlots[lcIdx] += sysBlocks - staleIntersection.length();

        // Per-request unique blocks (excluding shared sys blocks already counted above).
        for (auto const& kv : batch.kvCaches)
        {
            int totalBlocks = divUp(kv.capacity, tokensPerBlock);
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            int nonStale = totalBlocks - stale.length();
            int nonStaleSys = sysBlocks - intersect(stale, sysRange).length();
            int uniqueNonStale = std::max(0, nonStale - nonStaleSys);
            if (swaScratchReuse.has_value())
            {
                auto scratch = computeScratchRange(
                    lc, kv.historyLength, kv.capacity, tokensPerBlock, swaScratchReuse->maxRewindLen);
                int numScratch = scratch.length();
                // Scratch blocks share coalesced slots: actual slots = ceil(numScratch * fracMax).
                numSlots[lcIdx] += (uniqueNonStale - numScratch) + mSlotUtilFracMax[lcIdx].ceilMul(numScratch);
            }
            else
            {
                numSlots[lcIdx] += uniqueNonStale;
            }
        }
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsForBatch
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeSlotsForBatch(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    auto const slotsByLifeCycle = computeSlotsForBatchByLifeCycle(batch, tokensPerBlock, swaScratchReuse);
    TypedVec<PoolGroupIndex, SlotCount> numSlots(numPoolGroups(kGpuLevel), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < slotsByLifeCycle.size(); ++lifeCycle)
    {
        numSlots[getPoolGroupIndex(kGpuLevel, lifeCycle)] += slotsByLifeCycle[lifeCycle];
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// slotsToBytes
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, size_t> StorageManager::slotsToBytes(
    TypedVec<PoolGroupIndex, SlotCount> const& numSlots, size_t granularity) const
{
    TypedVec<PoolGroupIndex, size_t> numBytes(numPoolGroups(), 0);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numSlots.size(); ++pgIdx)
    {
        for (auto poolSize : slotSize(pgIdx))
        {
            numBytes[pgIdx] += roundUp(slotCountToSizeT(numSlots[pgIdx]) * poolSize, granularity);
        }
    }
    return numBytes;
}

// ---------------------------------------------------------------------------
// computeSlotCountForLevel
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeSlotCountForLevel(CacheTierConfig const& tierConfig,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    TypedVec<PoolGroupIndex, float> const& ratio, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const
{
    CacheTier tier = cacheTierOf(tierConfig);
    size_t quota = cacheTierQuota(tierConfig);
    size_t granularity = CacheLevelManager::cacheTierGranularity(tier, quota);
    quota = std::max(minQuotaForLevel(slotSizeLists, granularity, minSlots), roundUp(quota, granularity));
    return CacheLevelStorage::ratioToSlotCountList(quota, slotSizeLists, ratio, granularity, minSlots);
}

// ---------------------------------------------------------------------------
// minQuotaForLevel
// ---------------------------------------------------------------------------

size_t StorageManager::minQuotaForLevel(TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    size_t granularity, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const
{
    size_t total = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        for (auto slotSize : slotSizeLists[pgIdx])
        {
            total += roundUp(slotCountToSizeT(minSlots[pgIdx]) * slotSize, granularity);
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// constrainRatio
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, float> StorageManager::constrainRatio(TypedVec<PoolGroupIndex, float> const& ratio) const
{
    auto& gpuStorage = *mLevels[kGpuLevel].storage;
    size_t granularity = gpuStorage.poolSizeGranularity();
    auto slotCountList = gpuStorage.computeSlotCountList(ratio, mMinSlotsByLevel[kGpuLevel]);
    auto numBytes = slotsToBytes(slotCountList, granularity);
    return normalizeToRatio(numBytes);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
