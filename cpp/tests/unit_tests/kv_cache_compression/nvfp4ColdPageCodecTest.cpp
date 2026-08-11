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

#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

static_assert(std::is_base_of_v<kv::IKvCacheColdPageCodec, Nvfp4ColdPageCodec>);

namespace kv = batch_manager::kv_cache_manager_v2;

std::vector<kernels::Nvfp4BoundaryOffloadPageTask> gOffloadTasks;
std::vector<kernels::Nvfp4BoundaryOnboardPageTask> gOnboardTasks;
cudaStream_t gStream{};

constexpr std::uintptr_t kGpuKBase = 0x100000;
constexpr std::uintptr_t kGpuVBase = 0x200000;
constexpr std::uintptr_t kColdBase = 0x300000;
constexpr std::size_t kGpuSlotBytes = 512;
constexpr std::size_t kLayerRawBytes = 128;
constexpr std::size_t kLayerColdBytesAligned = 80;
constexpr std::size_t kColdSlotBytes = 160;
constexpr std::uintptr_t kStreamValue = 0x7000;

std::vector<Nvfp4ColdPageLayerConfig> makeLayers()
{
    std::vector<Nvfp4ColdPageLayerConfig> layers;
    for (int layerId : {0, 1})
    {
        Nvfp4ColdPageLayerConfig layer;
        layer.layerGroupId = kv::LayerGroupId{3};
        layer.layerId = layerId;
        layer.runtimeType = kernels::Nvfp4BoundaryRuntimeType::kFloat16;
        layer.numKvHeads = 1;
        layer.tokensPerPage = 4;
        layer.headDim = 16;
        layer.nvfp4ScaleOrigQuant = {2.0F, 4.0F};
        layer.nvfp4ScaleQuantOrig = {0.5F, 0.25F};
        layers.push_back(layer);
    }
    return layers;
}

kv::PoolGroupDesc makeGpuDesc()
{
    kv::SlotDescVariant variant;
    variant.lifeCycleId = kv::LayerGroupId{3};
    variant.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{kLayerRawBytes, {{0, "key"}, {1, "key"}}},
        kv::CoalescedBuffer{kLayerRawBytes, {{0, "value"}, {1, "value"}}},
    };

    kv::PoolGroupDesc gpuDesc;
    gpuDesc.poolGroupIndex = kv::PoolGroupIndex{0};
    gpuDesc.numSlots = kv::SlotCount{8};
    gpuDesc.slotDesc.variants = {variant};
    gpuDesc.pools = kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
        kv::PoolDesc{kv::PoolIndex{0}, kGpuKBase, kGpuSlotBytes},
        kv::PoolDesc{kv::PoolIndex{1}, kGpuVBase, kGpuSlotBytes},
    };
    return gpuDesc;
}

std::vector<Nvfp4ColdPageLayerConfig> makeEquivalentLifeCycleLayers(bool identicalParameters)
{
    auto layers = makeLayers();
    for (int layerId : {2, 3})
    {
        Nvfp4ColdPageLayerConfig layer = layers.at(static_cast<std::size_t>(layerId - 2));
        layer.layerGroupId = kv::LayerGroupId{4};
        layer.layerId = layerId;
        layers.push_back(layer);
    }
    if (!identicalParameters)
    {
        layers.back().nvfp4ScaleOrigQuant[0] = 8.0F;
    }
    return layers;
}

kv::PoolGroupDesc makeEquivalentLifeCycleGpuDesc()
{
    auto gpuDesc = makeGpuDesc();
    auto secondVariant = gpuDesc.slotDesc.variants.front();
    secondVariant.lifeCycleId = kv::LayerGroupId{4};
    secondVariant.coalescedBuffers[kv::PoolIndex{0}].bufferIds = {{2, "key"}, {3, "key"}};
    secondVariant.coalescedBuffers[kv::PoolIndex{1}].bufferIds = {{2, "value"}, {3, "value"}};
    gpuDesc.slotDesc.variants.push_back(std::move(secondVariant));
    return gpuDesc;
}

TEST(Nvfp4ColdPageCodecTest, ConfiguresLayoutAndLowersDisjointPages)
{
    gOffloadTasks.clear();
    gOnboardTasks.clear();
    gStream = nullptr;

    Nvfp4ColdPageCodec codec{makeLayers()};
    EXPECT_TRUE(codec.configure(makeGpuDesc()));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), kColdSlotBytes);
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{3}), kv::PageIndexLocation::kHost);

    kv::PageIndexPair const offloadIndices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    EXPECT_TRUE(codec.encode(
        kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), offloadIndices, std::size(offloadIndices), stream));
    EXPECT_EQ(gStream, stream);
    ASSERT_EQ(gOffloadTasks.size(), 4U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[0].rawK), kGpuKBase + kGpuSlotBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[1].rawK), kGpuKBase + kGpuSlotBytes + kLayerRawBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[2].rawV), kGpuVBase + 3 * kGpuSlotBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[3].rawV), kGpuVBase + 3 * kGpuSlotBytes + kLayerRawBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[0].compactPage), kColdBase + 2 * kColdSlotBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[1].compactPage),
        kColdBase + 2 * kColdSlotBytes + kLayerColdBytesAligned);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[2].compactPage), kColdBase + 5 * kColdSlotBytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks[3].compactPage),
        kColdBase + 5 * kColdSlotBytes + kLayerColdBytesAligned);

    kv::PageIndexPair const onboardIndices[]{{1, 2}, {3, 5}};
    EXPECT_TRUE(codec.decode(kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), onboardIndices,
        std::size(onboardIndices), stream));
    ASSERT_EQ(gOnboardTasks.size(), 4U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOnboardTasks[0].compactPage),
        reinterpret_cast<std::uintptr_t>(gOffloadTasks[0].compactPage));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOnboardTasks[3].rawV),
        reinterpret_cast<std::uintptr_t>(gOffloadTasks[3].rawV));
}

TEST(Nvfp4ColdPageCodecTest, RejectedConfigurePreservesPublishedLayout)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto const gpuDesc = makeGpuDesc();
    ASSERT_TRUE(codec.configure(gpuDesc));
    EXPECT_FALSE(codec.configure(gpuDesc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), kColdSlotBytes);
}

TEST(Nvfp4ColdPageCodecTest, UsesKvcmsCurrentSlotIndicesAfterPoolResize)
{
    gOffloadTasks.clear();
    gOnboardTasks.clear();

    Nvfp4ColdPageCodec codec{makeLayers()};
    ASSERT_TRUE(codec.configure(makeGpuDesc()));

    // GPU Pool virtual addresses and Slot strides remain stable when KVCM
    // expands a PoolGroup. KVCM owns the live allocator capacity, so the codec
    // must accept a newly allocated index without requiring reconfiguration.
    kv::PageIndexPair const offloadIndices[]{{0, 11}};
    EXPECT_TRUE(codec.encode(
        kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), offloadIndices, std::size(offloadIndices), nullptr));
    ASSERT_EQ(gOffloadTasks.size(), 2U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOffloadTasks.front().rawK), kGpuKBase + 11 * kGpuSlotBytes);

    kv::PageIndexPair const onboardIndices[]{{11, 0}};
    EXPECT_TRUE(codec.decode(kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), onboardIndices,
        std::size(onboardIndices), nullptr));
    ASSERT_EQ(gOnboardTasks.size(), 2U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(gOnboardTasks.front().rawK), kGpuKBase + 11 * kGpuSlotBytes);
}

TEST(Nvfp4ColdPageCodecTest, RejectsAttentionBufferWithMismatchedGeometry)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto gpuDesc = makeGpuDesc();
    gpuDesc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}].singleBufferSize += 16U;

    EXPECT_FALSE(codec.configure(gpuDesc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, DeclinesLayerGroupWithoutAttentionKvConfig)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto gpuDesc = makeGpuDesc();
    gpuDesc.slotDesc.variants.front().lifeCycleId = kv::LayerGroupId{4};

    // A descriptor with no matching K/V layer is valid but is not owned by
    // this codec. StorageManager treats its zero size query as raw fallback.
    ASSERT_TRUE(codec.configure(gpuDesc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{4}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, RejectsUnrepresentedAttentionSideBuffer)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto gpuDesc = makeGpuDesc();
    gpuDesc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}].bufferIds.push_back({0, "index_key"});

    EXPECT_FALSE(codec.configure(gpuDesc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, ReturnsSmallestRepresentativeForIdenticalPhysicalTransforms)
{
    Nvfp4ColdPageCodec codec{makeEquivalentLifeCycleLayers(
        /*identicalParameters=*/true)};
    ASSERT_TRUE(codec.configure(makeEquivalentLifeCycleGpuDesc()));

    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{3}), kv::LayerGroupId{3});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{4}), kv::LayerGroupId{3});
}

TEST(Nvfp4ColdPageCodecTest, KeepsDifferentKernelParametersInSeparateBatches)
{
    Nvfp4ColdPageCodec codec{makeEquivalentLifeCycleLayers(
        /*identicalParameters=*/false)};
    ASSERT_TRUE(codec.configure(makeEquivalentLifeCycleGpuDesc()));

    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{3}), kv::LayerGroupId{3});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{4}), kv::LayerGroupId{4});
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

namespace tensorrt_llm::kernels
{

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const&, Nvfp4BoundaryRuntimeType, cudaStream_t stream)
{
    kv_cache_compression::gOffloadTasks = tasks;
    kv_cache_compression::gStream = stream;
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& tasks,
    Nvfp4BoundaryKernelParams const&, Nvfp4BoundaryRuntimeType, cudaStream_t stream)
{
    kv_cache_compression::gOnboardTasks = tasks;
    kv_cache_compression::gStream = stream;
}

} // namespace tensorrt_llm::kernels
