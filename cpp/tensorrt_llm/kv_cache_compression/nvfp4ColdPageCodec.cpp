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

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
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

std::size_t scalarCount(Nvfp4ColdPageLayerConfig const& config)
{
    if (config.numKvHeads <= 0 || config.tokensPerPage <= 0 || config.headDim <= 0)
    {
        throw std::invalid_argument("NVFP4 cold Page geometry must be positive");
    }
    if (config.tokensPerPage % 4 != 0 || config.headDim % static_cast<std::int32_t>(kElementsPerBlockScale) != 0)
    {
        throw std::invalid_argument(
            "NVFP4 cold Pages require tokensPerPage "
            "divisible by 4 and headDim by 16");
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
    return checkedMul(packedBytes, 2U, "NVFP4 packed Page size overflows size_t")
        + checkedMul(scaleBytes, 2U, "NVFP4 scale Page size overflows size_t");
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

using KernelCohortKey = std::tuple<kernels::Nvfp4BoundaryRuntimeType, std::int32_t, std::int32_t, std::int32_t,
    std::array<float, 2>, std::array<float, 2>, std::array<float, 2>, std::array<float, 2>>;

KernelCohortKey cohortKey(Nvfp4ColdPageLayerConfig const& config)
{
    return {config.runtimeType, config.numKvHeads, config.tokensPerPage, config.headDim, config.nvfp4ScaleOrigQuant,
        config.nvfp4ScaleQuantOrig, config.fp8ScaleOrigQuant, config.fp8ScaleQuantOrig};
}

template <typename Pointer>
Pointer addBytes(Pointer base, std::size_t offset)
{
    return reinterpret_cast<Pointer>(reinterpret_cast<std::uintptr_t>(base) + offset);
}

std::size_t checkedPageOffset(std::int32_t pageIndex, std::size_t stride)
{
    if (pageIndex < 0)
    {
        throw std::invalid_argument("Base Page indices must be non-negative");
    }
    return checkedMul(static_cast<std::size_t>(pageIndex), stride, "Base Page address overflows size_t");
}

} // namespace

Nvfp4ColdPageCodec::Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs)
    : mLayerConfigs(std::move(layerConfigs))
{
    if (mLayerConfigs.empty())
    {
        throw std::invalid_argument("Nvfp4ColdPageCodec requires at least one layer config");
    }
    std::set<std::pair<kv::LayerGroupId, kv::LayerId>> identities;
    for (auto const& config : mLayerConfigs)
    {
        static_cast<void>(compactLayerBytes(config));
        if (!identities.emplace(config.layerGroupId, config.layerId).second)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec layer configs must be unique");
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

bool Nvfp4ColdPageCodec::configure(kv::PoolGroupDesc const& gpuDesc) noexcept
{
    try
    {
        if (gpuDesc.numSlots <= 0 || gpuDesc.pools.empty())
        {
            throw std::invalid_argument("GPU PoolGroupDesc must contain Pools and Slots");
        }

        std::map<kv::LayerGroupId, LayerGroupState> pending;
        for (auto const& variant : gpuDesc.slotDesc.variants)
        {
            std::vector<Nvfp4ColdPageLayerConfig const*> configs;
            for (auto const& config : mLayerConfigs)
            {
                if (config.layerGroupId == variant.lifeCycleId)
                {
                    configs.push_back(&config);
                }
            }
            if (configs.empty())
            {
                continue;
            }
            if (mLayerGroups.count(variant.lifeCycleId) != 0U)
            {
                throw std::invalid_argument("A layer group was configured more than once");
            }

            LayerGroupState state;
            state.gpuDesc = gpuDesc;
            for (auto const* config : configs)
            {
                LayerState layer;
                layer.config = *config;
                bool foundK = false;
                bool foundV = false;
                for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
                {
                    auto const& coalesced = variant.coalescedBuffers[poolIndex];
                    std::size_t offset = 0U;
                    for (auto const& bufferId : coalesced.bufferIds)
                    {
                        if (bufferId.layerId == config->layerId && bufferId.role == kKeyRole)
                        {
                            layer.gpuK = BufferLocation{poolIndex, offset};
                            foundK = true;
                        }
                        if (bufferId.layerId == config->layerId && bufferId.role == kValueRole)
                        {
                            layer.gpuV = BufferLocation{poolIndex, offset};
                            foundV = true;
                        }
                        offset += coalesced.singleBufferSize;
                    }
                }
                if (!foundK || !foundV)
                {
                    throw std::invalid_argument(
                        "GPU PoolGroupDesc is missing a "
                        "configured layer's key/value buffer");
                }

                auto const rawElementBytes
                    = config->runtimeType == kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3 ? 1U : 2U;
                auto const rawBytes
                    = checkedMul(scalarCount(*config), rawElementBytes, "Runtime KV Page size overflows size_t");
                for (auto const& location : {layer.gpuK, layer.gpuV})
                {
                    if (location.poolIndex >= gpuDesc.pools.size())
                    {
                        throw std::invalid_argument("GPU buffer selects a missing Pool");
                    }
                    auto const& pool = gpuDesc.pools[location.poolIndex];
                    if (pool.baseAddress == 0U || location.offset > pool.slotBytes
                        || rawBytes > pool.slotBytes - location.offset)
                    {
                        throw std::invalid_argument("GPU key/value buffer exceeds its configured Slot");
                    }
                }

                layer.coldOffset = alignUp(state.coldPageBytes, kCompactAlignment);
                state.coldPageBytes = alignUp(layer.coldOffset + compactLayerBytes(*config), kCompactAlignment);
                state.layers.push_back(std::move(layer));
            }
            pending.emplace(variant.lifeCycleId, std::move(state));
        }

        // All variants in one PoolGroupDesc use the same physical Pools and
        // Slot strides, but their lifecycle rules and buffer meanings may
        // differ. Collapse calls only when the bytes seen by every kernel task
        // are identical. Semantic layer IDs are intentionally ignored: once
        // pool/offset placement and every kernel-visible parameter match, the
        // transform is the same operation over another Page-index array.
        auto const sameCodecBehavior = [](LayerGroupState const& lhs, LayerGroupState const& rhs)
        {
            if (lhs.coldPageBytes != rhs.coldPageBytes || lhs.layers.size() != rhs.layers.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < lhs.layers.size(); ++i)
            {
                auto const& left = lhs.layers[i];
                auto const& right = rhs.layers[i];
                if (left.gpuK.poolIndex != right.gpuK.poolIndex || left.gpuK.offset != right.gpuK.offset
                    || left.gpuV.poolIndex != right.gpuV.poolIndex || left.gpuV.offset != right.gpuV.offset
                    || left.coldOffset != right.coldOffset || cohortKey(left.config) != cohortKey(right.config))
                {
                    return false;
                }
            }
            return true;
        };

        std::map<kv::LayerGroupId, kv::LayerGroupId> pendingBatchingLayerGroups;
        for (auto const& [layerGroupId, state] : pending)
        {
            auto representative = layerGroupId;
            for (auto const& [candidateId, candidateState] : pending)
            {
                if (candidateId >= layerGroupId)
                {
                    break;
                }
                if (sameCodecBehavior(candidateState, state))
                {
                    representative = candidateId;
                    break;
                }
            }
            pendingBatchingLayerGroups.emplace(layerGroupId, representative);
        }

        // Publish the entire descriptor atomically. Configuration happens only
        // during manager construction, so copying the small immutable map is
        // preferable to leaving a partially configured codec if allocation
        // throws while inserting one of several layer groups.
        auto configured = mLayerGroups;
        auto batchingLayerGroups = mBatchingLayerGroups;
        for (auto& [layerGroupId, state] : pending)
        {
            configured.emplace(layerGroupId, std::move(state));
        }
        for (auto const& [layerGroupId, representative] : pendingBatchingLayerGroups)
        {
            batchingLayerGroups.emplace(layerGroupId, representative);
        }
        mLayerGroups.swap(configured);
        mBatchingLayerGroups.swap(batchingLayerGroups);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::configure rejected PoolGroupDesc: %s", error.what());
        return false;
    }
}

std::size_t Nvfp4ColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? 0U : found->second.coldPageBytes;
}

kv::LayerGroupId Nvfp4ColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mBatchingLayerGroups.find(layerGroupId);
    return found == mBatchingLayerGroups.end() ? layerGroupId : found->second;
}

kv::PageIndexLocation Nvfp4ColdPageCodec::queryPageIndexLocation(kv::LayerGroupId) const noexcept
{
    return kv::PageIndexLocation::kHost;
}

bool Nvfp4ColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        if (numBasePages == 0U)
        {
            return true;
        }
        if (dstBasePtr == nullptr || pageIndices == nullptr)
        {
            throw std::invalid_argument("Non-empty encode requires the cold base pointer and Page-index pairs");
        }
        auto const found = mLayerGroups.find(layerGroupId);
        if (found == mLayerGroups.end())
        {
            throw std::invalid_argument("encode received an unconfigured layer group");
        }
        auto const& state = found->second;
        using Cohort
            = std::pair<Nvfp4ColdPageLayerConfig const*, std::vector<kernels::Nvfp4BoundaryOffloadPageTask>>;
        std::map<KernelCohortKey, Cohort> cohorts;
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            auto const srcIndex = pageIndices[page].src;
            if (srcIndex < 0 || static_cast<kv::SlotCount>(srcIndex) >= state.gpuDesc.numSlots)
            {
                throw std::invalid_argument(
                    "encode source Page index exceeds the "
                    "configured GPU Slot capacity");
            }
            auto const coldPage = addBytes(static_cast<std::uint8_t*>(dstBasePtr),
                checkedPageOffset(pageIndices[page].dst, state.coldPageBytes));
            for (auto const& layer : state.layers)
            {
                auto const gpuAddress = [&](BufferLocation const& location)
                {
                    auto const& pool = state.gpuDesc.pools[location.poolIndex];
                    return pool.baseAddress + checkedPageOffset(srcIndex, pool.slotBytes) + location.offset;
                };
                auto& [config, tasks] = cohorts[cohortKey(layer.config)];
                config = &layer.config;
                tasks.push_back({reinterpret_cast<void const*>(gpuAddress(layer.gpuK)),
                    reinterpret_cast<void const*>(gpuAddress(layer.gpuV)), addBytes(coldPage, layer.coldOffset)});
            }
        }

        for (auto const& item : cohorts)
        {
            auto const& [config, tasks] = item.second;
            kernels::invokeNvfp4BoundaryOffloadCompress(
                tasks, makeKernelParams(*config), config->runtimeType, stream);
        }
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
        if (numBasePages == 0U)
        {
            return true;
        }
        if (srcBasePtr == nullptr || pageIndices == nullptr)
        {
            throw std::invalid_argument("Non-empty decode requires the cold base pointer and Page-index pairs");
        }
        auto const found = mLayerGroups.find(layerGroupId);
        if (found == mLayerGroups.end())
        {
            throw std::invalid_argument("decode received an unconfigured layer group");
        }
        auto const& state = found->second;
        using Cohort
            = std::pair<Nvfp4ColdPageLayerConfig const*, std::vector<kernels::Nvfp4BoundaryOnboardPageTask>>;
        std::map<KernelCohortKey, Cohort> cohorts;
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            auto const dstIndex = pageIndices[page].dst;
            if (dstIndex < 0 || static_cast<kv::SlotCount>(dstIndex) >= state.gpuDesc.numSlots)
            {
                throw std::invalid_argument(
                    "decode destination Page index exceeds "
                    "the configured GPU Slot capacity");
            }
            auto const coldPage = addBytes(static_cast<std::uint8_t const*>(srcBasePtr),
                checkedPageOffset(pageIndices[page].src, state.coldPageBytes));
            for (auto const& layer : state.layers)
            {
                auto const gpuAddress = [&](BufferLocation const& location)
                {
                    auto const& pool = state.gpuDesc.pools[location.poolIndex];
                    return pool.baseAddress + checkedPageOffset(dstIndex, pool.slotBytes) + location.offset;
                };
                auto& [config, tasks] = cohorts[cohortKey(layer.config)];
                config = &layer.config;
                tasks.push_back({addBytes(coldPage, layer.coldOffset), reinterpret_cast<void*>(gpuAddress(layer.gpuK)),
                    reinterpret_cast<void*>(gpuAddress(layer.gpuV))});
            }
        }

        for (auto const& item : cohorts)
        {
            auto const& [config, tasks] = item.second;
            kernels::invokeNvfp4BoundaryOnboardDecompress(
                tasks, makeKernelParams(*config), config->runtimeType, stream);
        }
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
