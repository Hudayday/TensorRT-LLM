/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

namespace kv = batch_manager::kv_cache_manager_v2;

static_assert(std::is_base_of_v<kv::IKvCacheColdPageCodec, Nvfp4ColdPageCodec>);

struct RecordedLaunch
{
    int offloadCalls = 0;
    int onboardCalls = 0;
    std::vector<kernels::Nvfp4BoundaryOffloadPageTask> offloadPages;
    std::vector<kernels::Nvfp4BoundaryOnboardPageTask> onboardPages;
    kernels::Nvfp4BoundaryPreparedPlan plan;
    void const* coldBase = nullptr;
    cudaStream_t stream{};
};

RecordedLaunch gLaunch;

constexpr std::uintptr_t kGpuKBase = 0x100000;
constexpr std::uintptr_t kGpuVBase = 0x200000;
constexpr std::uintptr_t kColdBase = 0x300000;
constexpr std::size_t kLayerRawBytes = 320;
constexpr std::size_t kLayerColdBytesAligned = 192;
constexpr std::size_t kNumAttentionLayers = 8;
constexpr std::size_t kGpuSlotBytes = kNumAttentionLayers * kLayerRawBytes;
constexpr std::size_t kColdSlotBytes = kNumAttentionLayers * kLayerColdBytesAligned;
constexpr std::uintptr_t kStreamValue = 0x7000;

void resetLaunch()
{
    gLaunch = {};
}

std::vector<Nvfp4ColdPageLayerConfig> makeLayers(std::size_t count = kNumAttentionLayers, int firstLayer = 0)
{
    std::vector<Nvfp4ColdPageLayerConfig> layers;
    layers.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        Nvfp4ColdPageLayerConfig layer;
        layer.layerId = firstLayer + static_cast<int>(index);
        layer.runtimeType = kernels::Nvfp4BoundaryRuntimeType::kFloat16;
        layer.numKvHeads = 1;
        layer.tokensPerPage = 5;
        layer.headDim = 32;
        auto const scale = static_cast<float>(index + 2U);
        layer.nvfp4ScaleOrigQuant = {scale, scale + 0.5F};
        layer.nvfp4ScaleQuantOrig = {1.0F / scale, 1.0F / (scale + 0.5F)};
        layers.push_back(layer);
    }
    return layers;
}

kv::PoolGroupDesc makeAttentionDesc(kv::PoolGroupIndex poolGroupIndex = kv::PoolGroupIndex{0},
    kv::LayerGroupId lifeCycle = kv::LayerGroupId{3}, std::size_t count = kNumAttentionLayers, int firstLayer = 0,
    std::uintptr_t keyBase = kGpuKBase, std::uintptr_t valueBase = kGpuVBase)
{
    kv::CoalescedBuffer keys;
    keys.singleBufferSize = kLayerRawBytes;
    kv::CoalescedBuffer values;
    values.singleBufferSize = kLayerRawBytes;
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const layerId = firstLayer + static_cast<int>(index);
        keys.bufferIds.push_back({layerId, "key"});
        values.bufferIds.push_back({layerId, "value"});
    }

    kv::SlotDescVariant variant;
    variant.lifeCycleId = lifeCycle;
    variant.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{keys, values};

    auto const slotBytes = count * kLayerRawBytes;
    kv::PoolGroupDesc desc;
    desc.poolGroupIndex = poolGroupIndex;
    desc.numSlots = kv::SlotCount{512};
    desc.slotDesc.variants = {variant};
    desc.pools = kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
        kv::PoolDesc{kv::PoolIndex{0}, keyBase, slotBytes},
        kv::PoolDesc{kv::PoolIndex{1}, valueBase, slotBytes},
    };
    return desc;
}

bool configureOne(Nvfp4ColdPageCodec& codec, kv::PoolGroupDesc const& desc)
{
    return codec.configure(&desc, kv::PoolGroupIndex{1});
}

TEST(Nvfp4ColdPageCodecTest, OneCompletePageTaskCoversAllLayersWithDistinctScales)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto const desc = makeAttentionDesc();
    ASSERT_TRUE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), kColdSlotBytes);
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{3}), kv::PageIndexLocation::kHost);

    kv::PageIndexPair const indices[]{{2, 1}, {5, 3}};
    auto const stream = reinterpret_cast<cudaStream_t>(kStreamValue);
    ASSERT_TRUE(
        codec.encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices, std::size(indices), stream));

    // The codec submits the complete Page batch once. Eight layers with eight
    // distinct scale pairs are immutable launch metadata, never eight calls.
    EXPECT_EQ(gLaunch.offloadCalls, 1);
    ASSERT_EQ(gLaunch.offloadPages.size(), 2U);
    EXPECT_EQ(gLaunch.offloadPages[0].gpuPageIndex, 1);
    EXPECT_EQ(gLaunch.offloadPages[0].coldPageIndex, 2);
    EXPECT_EQ(gLaunch.offloadPages[1].gpuPageIndex, 3);
    EXPECT_EQ(gLaunch.offloadPages[1].coldPageIndex, 5);
    ASSERT_EQ(gLaunch.plan.numLayers, kNumAttentionLayers);
    for (std::size_t layer = 0; layer < kNumAttentionLayers; ++layer)
    {
        EXPECT_EQ(gLaunch.plan.layers[layer].rawKBase, kGpuKBase + layer * kLayerRawBytes);
        EXPECT_EQ(gLaunch.plan.layers[layer].rawVBase, kGpuVBase + layer * kLayerRawBytes);
        EXPECT_EQ(gLaunch.plan.layers[layer].rawKSlotBytes, kGpuSlotBytes);
        EXPECT_EQ(gLaunch.plan.layers[layer].rawVSlotBytes, kGpuSlotBytes);
        EXPECT_EQ(gLaunch.plan.layers[layer].coldOffset, layer * kLayerColdBytesAligned);
        EXPECT_EQ(gLaunch.plan.layers[layer].params.tokensPerPage, 5);
        EXPECT_EQ(gLaunch.plan.layers[layer].params.headDim, 32);
        EXPECT_FLOAT_EQ(gLaunch.plan.layers[layer].params.nvfp4ScaleOrigQuant[0], static_cast<float>(layer + 2U));
    }
    EXPECT_EQ(gLaunch.coldBase, reinterpret_cast<void*>(kColdBase));
    EXPECT_EQ(gLaunch.plan.coldPageBytes, kColdSlotBytes);
    EXPECT_EQ(gLaunch.stream, stream);

    ASSERT_TRUE(codec.decode(
        kv::LayerGroupId{3}, reinterpret_cast<void const*>(kColdBase), indices, std::size(indices), stream));
    EXPECT_EQ(gLaunch.onboardCalls, 1);
    ASSERT_EQ(gLaunch.onboardPages.size(), 2U);
    EXPECT_EQ(gLaunch.onboardPages[0].gpuPageIndex, 2);
    EXPECT_EQ(gLaunch.onboardPages[0].coldPageIndex, 1);
}

TEST(Nvfp4ColdPageCodecTest, PreservesOneCodecSubmissionAcrossThe256PageKernelBoundary)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto const desc = makeAttentionDesc();
    ASSERT_TRUE(configureOne(codec, desc));

    std::vector<kv::PageIndexPair> indices(257);
    for (std::size_t page = 0; page < indices.size(); ++page)
    {
        indices[page] = {static_cast<std::int32_t>(500U - page), static_cast<std::int32_t>(page * 2U)};
    }
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{3}, reinterpret_cast<void*>(kColdBase), indices.data(), indices.size(),
        reinterpret_cast<cudaStream_t>(kStreamValue)));
    EXPECT_EQ(gLaunch.offloadCalls, 1);
    EXPECT_EQ(gLaunch.offloadPages.size(), 257U);
    EXPECT_EQ(gLaunch.plan.numLayers, kNumAttentionLayers);
}

TEST(Nvfp4ColdPageCodecTest, EmptyAttentionBatchIsValidAndDoesNotLaunch)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto const desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);
    ASSERT_TRUE(configureOne(codec, desc));

    EXPECT_TRUE(codec.encode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_TRUE(codec.decode(kv::LayerGroupId{0}, nullptr, nullptr, 0U, nullptr));
    EXPECT_EQ(gLaunch.offloadCalls, 0);
    EXPECT_EQ(gLaunch.onboardCalls, 0);
}

TEST(Nvfp4ColdPageCodecTest, NonEmptyAttentionBatchRejectsNullStream)
{
    resetLaunch();
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto const desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);
    ASSERT_TRUE(configureOne(codec, desc));

    kv::PageIndexPair const indices[]{{0, 0}};
    EXPECT_FALSE(codec.encode(kv::LayerGroupId{0}, reinterpret_cast<void*>(kColdBase), indices, 1U, nullptr));
    EXPECT_FALSE(codec.decode(kv::LayerGroupId{0}, reinterpret_cast<void const*>(kColdBase), indices, 1U, nullptr));
    EXPECT_EQ(gLaunch.offloadCalls, 0);
    EXPECT_EQ(gLaunch.onboardCalls, 0);
}

TEST(Nvfp4ColdPageCodecTest, ConfigureConsumesAllPoolGroupsOnceAndDiscoversLifecycleMembership)
{
    auto layers = makeLayers(2);
    auto secondGroupLayers = makeLayers(2, 2);
    layers.insert(layers.end(), secondGroupLayers.begin(), secondGroupLayers.end());
    Nvfp4ColdPageCodec codec{layers};
    std::array descs{makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 2),
        makeAttentionDesc(kv::PoolGroupIndex{1}, kv::LayerGroupId{1}, 2, 2, 0x400000, 0x500000)};
    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 2U * kLayerColdBytesAligned);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kLayerColdBytesAligned);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{0}), kv::LayerGroupId{0});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{1}), kv::LayerGroupId{1});
    EXPECT_FALSE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
}

TEST(Nvfp4ColdPageCodecTest, RejectsAttentionBufferWithMismatchedGeometry)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto desc = makeAttentionDesc();
    desc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}].singleBufferSize += 16U;
    desc.pools[kv::PoolIndex{0}].slotBytes += 16U * kNumAttentionLayers;

    EXPECT_FALSE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{3}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, RejectsConfiguredAttentionLayerMissingFromAllGpuLayouts)
{
    Nvfp4ColdPageCodec codec{makeLayers(2)};
    auto const desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);

    EXPECT_FALSE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, RejectsConfiguredAttentionLayerOwnedByTwoLifecycles)
{
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);
    auto duplicate = desc.slotDesc.variants.front();
    duplicate.lifeCycleId = kv::LayerGroupId{1};
    desc.slotDesc.variants.push_back(std::move(duplicate));

    EXPECT_FALSE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), 0U);
}

TEST(Nvfp4ColdPageCodecTest, RejectsAttentionAndUnknownSideBufferInOneLifecycle)
{
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);
    auto& keys = desc.slotDesc.variants.front().coalescedBuffers[kv::PoolIndex{0}];
    keys.bufferIds.push_back({0, "index_key"});
    desc.pools[kv::PoolIndex{0}].slotBytes += keys.singleBufferSize;

    EXPECT_FALSE(configureOne(codec, desc));
}

TEST(Nvfp4ColdPageCodecTest, UnknownLifecycleUsesFailureSentinels)
{
    Nvfp4ColdPageCodec codec{makeLayers()};
    auto const desc = makeAttentionDesc();
    ASSERT_TRUE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{99}), 0U);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{99}), kv::LayerGroupId{-1});
    EXPECT_EQ(codec.queryPageIndexLocation(kv::LayerGroupId{99}), kv::PageIndexLocation::kBadLocation);
}

TEST(Nvfp4ColdPageCodecTest, NonAttentionLifecycleUsesLosslessSingleBlob)
{
    resetLaunch();
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "CUDA device is required for the lossless copy data-plane test";
    }

    constexpr std::size_t kPoolBytes = 64;
    constexpr std::size_t kSlots = 2;
    std::byte* statePool = nullptr;
    std::byte* convPool = nullptr;
    std::byte* coldPool = nullptr;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&statePool), kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&convPool), kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&coldPool), 2U * kPoolBytes * kSlots), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    kv::SlotDescVariant variant;
    variant.lifeCycleId = kv::LayerGroupId{1};
    variant.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{kPoolBytes, {{10, "ssm_state"}}},
        kv::CoalescedBuffer{kPoolBytes, {{10, "conv_state"}}},
    };
    kv::PoolGroupDesc desc;
    desc.poolGroupIndex = kv::PoolGroupIndex{1};
    desc.numSlots = kSlots;
    desc.slotDesc.variants = {variant};
    desc.pools = kv::TypedVec<kv::PoolIndex, kv::PoolDesc>{
        kv::PoolDesc{kv::PoolIndex{0}, reinterpret_cast<std::uintptr_t>(statePool), kPoolBytes},
        kv::PoolDesc{kv::PoolIndex{1}, reinterpret_cast<std::uintptr_t>(convPool), kPoolBytes},
    };

    Nvfp4ColdPageCodec codec{makeLayers(1)};
    std::array descs{makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1), desc};
    ASSERT_TRUE(codec.configure(descs.data(), kv::PoolGroupIndex{2}));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kPoolBytes);

    std::vector<std::byte> state(kPoolBytes);
    std::vector<std::byte> conv(kPoolBytes);
    for (std::size_t index = 0; index < kPoolBytes; ++index)
    {
        state[index] = static_cast<std::byte>(index + 1U);
        conv[index] = static_cast<std::byte>(index + 65U);
    }
    ASSERT_EQ(cudaMemcpy(statePool + kPoolBytes, state.data(), kPoolBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(convPool + kPoolBytes, conv.data(), kPoolBytes, cudaMemcpyHostToDevice), cudaSuccess);

    kv::PageIndexPair const encodePair{0, 1};
    ASSERT_TRUE(codec.encode(kv::LayerGroupId{1}, coldPool, &encodePair, 1U, stream));
    ASSERT_EQ(cudaMemsetAsync(statePool, 0, kPoolBytes, stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(convPool, 0, kPoolBytes, stream), cudaSuccess);
    kv::PageIndexPair const decodePair{0, 0};
    ASSERT_TRUE(codec.decode(kv::LayerGroupId{1}, coldPool, &decodePair, 1U, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<std::byte> restoredState(kPoolBytes);
    std::vector<std::byte> restoredConv(kPoolBytes);
    ASSERT_EQ(cudaMemcpy(restoredState.data(), statePool, kPoolBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(restoredConv.data(), convPool, kPoolBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(restoredState, state);
    EXPECT_EQ(restoredConv, conv);
    EXPECT_EQ(gLaunch.offloadCalls, 0);

    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(coldPool), cudaSuccess);
    EXPECT_EQ(cudaFree(convPool), cudaSuccess);
    EXPECT_EQ(cudaFree(statePool), cudaSuccess);
}

TEST(Nvfp4ColdPageCodecTest, AttentionAndSsmSharingOneHotPoolGroupUseDifferentTransforms)
{
    Nvfp4ColdPageCodec codec{makeLayers(1)};
    auto desc = makeAttentionDesc(kv::PoolGroupIndex{0}, kv::LayerGroupId{0}, 1);

    kv::SlotDescVariant ssm;
    ssm.lifeCycleId = kv::LayerGroupId{1};
    ssm.coalescedBuffers = kv::TypedVec<kv::PoolIndex, kv::CoalescedBuffer>{
        kv::CoalescedBuffer{kLayerRawBytes, {{10, "ssm_state"}}},
        kv::CoalescedBuffer{kLayerRawBytes, {{10, "conv_state"}}},
    };
    desc.slotDesc.variants.push_back(std::move(ssm));

    ASSERT_TRUE(configureOne(codec, desc));
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{0}), kLayerColdBytesAligned);
    EXPECT_EQ(codec.queryColdPageBytes(kv::LayerGroupId{1}), 2U * kLayerRawBytes);
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{0}), kv::LayerGroupId{0});
    EXPECT_EQ(codec.getBatchingLayerGroupId(kv::LayerGroupId{1}), kv::LayerGroupId{1});
}

} // namespace
} // namespace tensorrt_llm::kv_cache_compression

namespace tensorrt_llm::kernels
{

Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(
    std::vector<Nvfp4BoundaryLayerPlan> const& layers, std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType)
{
    if (layers.empty() || layers.size() > kNvfp4BoundaryMaxLayersPerLaunch || coldPageBytes == 0U)
    {
        throw std::invalid_argument("invalid test launch plan");
    }
    Nvfp4BoundaryPreparedPlan plan;
    std::copy(layers.begin(), layers.end(), plan.layers.begin());
    plan.numLayers = static_cast<std::uint32_t>(layers.size());
    plan.coldPageBytes = coldPageBytes;
    plan.runtimeType = runtimeType;
    return plan;
}

void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.offloadCalls;
    launch.offloadPages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream)
{
    auto& launch = kv_cache_compression::gLaunch;
    ++launch.onboardCalls;
    launch.onboardPages = pages;
    launch.plan = plan;
    launch.coldBase = coldBase;
    launch.stream = stream;
}

} // namespace tensorrt_llm::kernels
