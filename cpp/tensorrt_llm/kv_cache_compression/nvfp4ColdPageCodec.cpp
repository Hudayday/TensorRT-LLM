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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

constexpr std::size_t kCompactAlignment = 16U;
constexpr std::size_t kElementsPerBlockScale = 16U;
constexpr std::size_t kPackedElementsPerByte = 2U;
constexpr char const* kKeyRole = "key";
constexpr char const* kValueRole = "value";

std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U))
    {
        throw std::overflow_error("Cold Page size overflows size_t");
    }
    return (value + alignment - 1U) / alignment * alignment;
}

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs * rhs;
}

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

std::size_t scalarCount(Nvfp4ColdPageLayerConfig const& config)
{
    if (config.numKvHeads <= 0 || config.tokensPerPage <= 0 || config.headDim <= 0)
    {
        throw std::invalid_argument("NVFP4 cold Page geometry must be positive");
    }
    if (config.headDim % static_cast<std::int32_t>(kElementsPerBlockScale) != 0)
    {
        throw std::invalid_argument("NVFP4 cold Pages require headDim divisible by 16");
    }
    auto const headsTimesTokens = checkedMul(static_cast<std::size_t>(config.numKvHeads),
        static_cast<std::size_t>(config.tokensPerPage), "NVFP4 Page geometry overflows size_t");
    return checkedMul(
        headsTimesTokens, static_cast<std::size_t>(config.headDim), "NVFP4 Page geometry overflows size_t");
}

std::size_t compactLayerBytes(Nvfp4ColdPageLayerConfig const& config)
{
    auto const elements = scalarCount(config);
    auto const packedBytes = elements / kPackedElementsPerByte;
    auto const scaleBytes = elements / kElementsPerBlockScale;
    return checkedAdd(checkedMul(packedBytes, 2U, "NVFP4 packed Page size overflows size_t"),
        checkedMul(scaleBytes, 2U, "NVFP4 scale Page size overflows size_t"),
        "NVFP4 compact Page size overflows size_t");
}

kernels::Nvfp4BoundaryKernelParams makeKernelParams(Nvfp4ColdPageLayerConfig const& config)
{
    kernels::Nvfp4BoundaryKernelParams params{};
    params.numKvHeads = config.numKvHeads;
    params.tokensPerPage = config.tokensPerPage;
    params.headDim = config.headDim;
    for (std::size_t role = 0; role < 2U; ++role)
    {
        params.nvfp4ScaleOrigQuant[role] = config.nvfp4ScaleOrigQuant[role];
        params.nvfp4ScaleQuantOrig[role] = config.nvfp4ScaleQuantOrig[role];
        params.fp8ScaleOrigQuant[role] = config.fp8ScaleOrigQuant[role];
        params.fp8ScaleQuantOrig[role] = config.fp8ScaleQuantOrig[role];
    }
    return params;
}

} // namespace

Nvfp4ColdPageCodec::Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs)
    : mLayerConfigs(std::move(layerConfigs))
{
    std::set<kv::LayerId> identities;
    for (auto const& config : mLayerConfigs)
    {
        static_cast<void>(compactLayerBytes(config));
        if (!identities.emplace(config.layerId).second)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec layer IDs must be unique");
        }
        if (config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kBfloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec received an unsupported runtime type");
        }
        for (auto const& scales : {config.nvfp4ScaleOrigQuant, config.nvfp4ScaleQuantOrig, config.fp8ScaleOrigQuant,
                 config.fp8ScaleQuantOrig})
        {
            if (!std::all_of(
                    scales.begin(), scales.end(), [](float value) { return std::isfinite(value) && value > 0; }))
            {
                throw std::invalid_argument("Nvfp4ColdPageCodec scales must be finite and positive");
            }
        }
    }
}

bool Nvfp4ColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        if (mConfigured || gpuDescs == nullptr || numGpuDescs.value() <= 0)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec must be configured exactly once with all GPU layouts");
        }

        std::map<kv::LayerId, Nvfp4ColdPageLayerConfig const*> configsByLayer;
        for (auto const& config : mLayerConfigs)
        {
            configsByLayer.emplace(config.layerId, &config);
        }

        struct BufferLocation
        {
            kv::PoolIndex poolIndex{0};
            std::size_t offset = 0;
            std::size_t bytes = 0;
            bool found = false;
        };

        struct AttentionBuffers
        {
            BufferLocation key;
            BufferLocation value;
        };

        // Compose PR #17512's default concat codec instead of duplicating its
        // cuMemcpyBatchAsync and Host-registration-boundary handling. The
        // local object is published only after every NVFP4 plan validates, so
        // a failed configure leaves this codec externally unconfigured.
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        std::set<kv::LayerId> configuredAttentionLayers;
        std::map<kv::LayerGroupId, LayerGroupState> pending;
        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            if (gpuDesc.poolGroupIndex != poolGroupIndex || gpuDesc.numSlots <= 0 || gpuDesc.pools.empty()
                || gpuDesc.slotDesc.variants.empty())
            {
                throw std::invalid_argument("GPU PoolGroupDesc is incomplete or out of order");
            }
            for (kv::PoolIndex poolIndex{0}; poolIndex < gpuDesc.pools.size(); ++poolIndex)
            {
                auto const& pool = gpuDesc.pools.at(poolIndex);
                if (pool.poolIndex != poolIndex || pool.baseAddress == 0U || pool.slotBytes == 0U)
                {
                    throw std::invalid_argument("GPU PoolGroupDesc contains an invalid Pool");
                }
            }

            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                if (variant.lifeCycleId.value() < 0 || variant.coalescedBuffers.size() != gpuDesc.pools.size()
                    || pending.count(variant.lifeCycleId) != 0U)
                {
                    throw std::invalid_argument("GPU lifecycle layout is invalid or duplicated");
                }

                std::map<kv::LayerId, AttentionBuffers> attentionBuffers;
                bool allBuffersAreConfiguredKv = true;
                for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
                {
                    auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
                    if (coalesced.size() != gpuDesc.pools.at(poolIndex).slotBytes)
                    {
                        throw std::invalid_argument("Lifecycle buffers do not cover their complete GPU Pool Slot");
                    }
                    std::size_t offset = 0U;
                    for (auto const& bufferId : coalesced.bufferIds)
                    {
                        auto const config = configsByLayer.find(bufferId.layerId);
                        if (config == configsByLayer.end()
                            || (bufferId.role != kKeyRole && bufferId.role != kValueRole))
                        {
                            allBuffersAreConfiguredKv = false;
                        }
                        else
                        {
                            auto& buffers = attentionBuffers[bufferId.layerId];
                            auto& location = bufferId.role == kKeyRole ? buffers.key : buffers.value;
                            if (location.found)
                            {
                                throw std::invalid_argument("GPU lifecycle contains a duplicate K/V buffer");
                            }
                            location = BufferLocation{poolIndex, offset, coalesced.singleBufferSize, true};
                        }
                        if (coalesced.singleBufferSize > std::numeric_limits<std::size_t>::max() - offset)
                        {
                            throw std::overflow_error("GPU Slot buffer offsets overflow size_t");
                        }
                        offset += coalesced.singleBufferSize;
                    }
                }

                LayerGroupState state;
                if (!attentionBuffers.empty() && allBuffersAreConfiguredKv)
                {
                    state.transform = Transform::kNvfp4Attention;
                    kernels::Nvfp4BoundaryRuntimeType runtimeType{};
                    std::vector<kernels::Nvfp4BoundaryLayerPlan> layers;
                    layers.reserve(attentionBuffers.size());
                    bool firstLayer = true;
                    for (auto const& [layerId, buffers] : attentionBuffers)
                    {
                        if (!buffers.key.found || !buffers.value.found)
                        {
                            throw std::invalid_argument("Configured Attention layer is missing K or V");
                        }
                        auto const& config = *configsByLayer.at(layerId);
                        if (firstLayer)
                        {
                            runtimeType = config.runtimeType;
                            firstLayer = false;
                        }
                        else if (runtimeType != config.runtimeType)
                        {
                            throw std::invalid_argument(
                                "One Attention lifecycle must use one runtime dtype for whole-Page batching");
                        }

                        auto const rawElementBytes
                            = config.runtimeType == kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3 ? 1U : 2U;
                        auto const rawBytes
                            = checkedMul(scalarCount(config), rawElementBytes, "Runtime KV Page size overflows size_t");
                        if (buffers.key.bytes != rawBytes || buffers.value.bytes != rawBytes)
                        {
                            throw std::invalid_argument(
                                "GPU K/V buffer size does not match the configured Attention geometry");
                        }
                        auto const& keyPool = gpuDesc.pools.at(buffers.key.poolIndex);
                        auto const& valuePool = gpuDesc.pools.at(buffers.value.poolIndex);
                        if (buffers.key.offset > keyPool.slotBytes || rawBytes > keyPool.slotBytes - buffers.key.offset
                            || buffers.value.offset > valuePool.slotBytes
                            || rawBytes > valuePool.slotBytes - buffers.value.offset)
                        {
                            throw std::invalid_argument("GPU K/V buffer exceeds its Pool Slot");
                        }

                        auto const coldOffset = state.coldPageBytes;
                        layers.push_back(kernels::Nvfp4BoundaryLayerPlan{keyPool.baseAddress + buffers.key.offset,
                            valuePool.baseAddress + buffers.value.offset, keyPool.slotBytes, valuePool.slotBytes,
                            coldOffset, makeKernelParams(config)});
                        state.coldPageBytes = alignUp(
                            checkedAdd(coldOffset, compactLayerBytes(config), "NVFP4 cold Page size overflows size_t"),
                            kCompactAlignment);
                        if (!configuredAttentionLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("One Attention layer belongs to multiple lifecycles");
                        }
                    }
                    // Pool addresses, Slot strides, compact offsets, geometry,
                    // scales, and CUDA argument padding are immutable after
                    // construction. Validate and freeze them once here rather
                    // than rescanning every layer in encode()/decode().
                    state.preparedPlan = kernels::prepareNvfp4BoundaryPlan(layers, state.coldPageBytes, runtimeType);
                }
                else
                {
                    if (!attentionBuffers.empty())
                    {
                        throw std::invalid_argument(
                            "A lifecycle cannot mix configured Attention K/V with lossless side buffers");
                    }
                    // Any lifecycle without configured Attention K/V is an
                    // intentional generic byte-exact passthrough. This covers
                    // today's SSM and convolutional state without teaching the
                    // compressor their semantics; future layouts remain safe
                    // but gain no compression until explicitly supported.
                    state.transform = Transform::kLosslessConcat;
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                }
                if (state.coldPageBytes == 0U)
                {
                    throw std::invalid_argument("Lifecycle produced an empty cold Page");
                }
                pending.emplace(variant.lifeCycleId, std::move(state));
            }
        }

        if (configuredAttentionLayers.size() != mLayerConfigs.size())
        {
            throw std::invalid_argument("Not every configured Attention layer was found in KVCM GPU layouts");
        }

        mLayerGroups = std::move(pending);
        mLosslessCodec = std::move(losslessCodec);
        mConfigured = true;
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
}

std::size_t Nvfp4ColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? 0U : state->coldPageBytes;
}

kv::LayerGroupId Nvfp4ColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::LayerGroupId{-1} : layerGroupId;
}

kv::PageIndexLocation Nvfp4ColdPageCodec::queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::PageIndexLocation::kBadLocation : kv::PageIndexLocation::kHost;
}

Nvfp4ColdPageCodec::LayerGroupState const* Nvfp4ColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
}

bool Nvfp4ColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr
            || (numBasePages != 0U && (dstBasePtr == nullptr || pageIndices == nullptr || stream == nullptr)))
        {
            throw std::invalid_argument("encode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (state->transform == Transform::kLosslessConcat)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }

        // Keep the compression range exclusive to Attention/NVFP4 work.
        // Lossless SSM/conv migration above deliberately remains ordinary
        // KVCM traffic, so Nsight can distinguish the two representations.
        NVTX3_SCOPED_RANGE(KVCC_OFFLOAD_COMPRESS_D2H);
        thread_local std::vector<kernels::Nvfp4BoundaryOffloadPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].src, pageIndices[page].dst});
        }
        kernels::invokeNvfp4BoundaryOffloadCompress(pages, state->preparedPlan, dstBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
}

bool Nvfp4ColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr
            || (numBasePages != 0U && (srcBasePtr == nullptr || pageIndices == nullptr || stream == nullptr)))
        {
            throw std::invalid_argument("decode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (state->transform == Transform::kLosslessConcat)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }

        // This range means NVFP4->runtime conversion, not generic onboarding.
        NVTX3_SCOPED_RANGE(KVCC_ONBOARD_H2D_DECOMPRESS);
        thread_local std::vector<kernels::Nvfp4BoundaryOnboardPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].dst, pageIndices[page].src});
        }
        kernels::invokeNvfp4BoundaryOnboardDecompress(pages, state->preparedPlan, srcBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
