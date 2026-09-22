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

#include "tensorrt_llm/batch_manager/kv_cache_compression/nativeColdPageCodec.h"

#include "kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

ResolvedHotLifecycle resolveLifecycle(kv::PoolGroupDesc const& gpuDesc, kv::SlotDescVariant const& variant)
{
    ResolvedHotLifecycle result{variant.lifeCycleId, {}};
    for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
    {
        auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
        auto const& pool = gpuDesc.pools.at(poolIndex);
        std::size_t offset = 0;
        for (auto const& bufferId : coalesced.bufferIds)
        {
            auto& layer = result.layers[bufferId.layerId];
            if (!layer
                     .emplace(bufferId.role,
                         ResolvedHotBuffer{pool.baseAddress + offset, pool.slotBytes, coalesced.singleBufferSize})
                     .second)
            {
                throw std::invalid_argument("GPU lifecycle contains a duplicate buffer role");
            }
            offset += coalesced.singleBufferSize;
        }
    }
    return result;
}

void drainAfterProviderFailure(cudaStream_t stream) noexcept
{
    auto const status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        TLLM_LOG_ERROR("Cold-page provider rollback drain failed: %s", cudaGetErrorString(status));
        std::terminate();
    }
}

} // namespace

NativeColdPageCodec::NativeColdPageCodec(std::set<kv::LayerId> layerIds, ColdPageSelectionConfig selection)
    : mLayerIds(std::move(layerIds))
    , mSelection(std::move(selection))
{
    if (mSelection.keepFirstTokens < 0 || mSelection.keepLastTokens < 0)
        throw std::invalid_argument("Cold-page token counts must be non-negative");
    for (auto const& [begin, end] : mSelection.keepTokenRanges)
        if (begin < 0 || begin >= end)
            throw std::invalid_argument("Invalid cold-page token range");
}

bool NativeColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        std::map<kv::LayerGroupId, LayerGroupState> pendingGroups;
        std::vector<ResolvedHotLifecycle> providerLifecycles;
        std::set<kv::LayerId> consumedLayers;

        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto resolved = resolveLifecycle(gpuDesc, variant);
                auto const providerLayerCount = static_cast<std::size_t>(std::count_if(resolved.layers.begin(),
                    resolved.layers.end(), [this](auto const& layer) { return mLayerIds.count(layer.first) != 0U; }));

                LayerGroupState state;
                if (providerLayerCount == 0U)
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                    state.pageIndexLocation = losslessCodec->queryPageIndexLocation(variant.lifeCycleId);
                }
                else
                {
                    if (providerLayerCount != resolved.layers.size())
                    {
                        throw std::invalid_argument("A lifecycle cannot mix provider-owned and fallback layers");
                    }
                    for (auto const& [layerId, buffers] : resolved.layers)
                    {
                        static_cast<void>(buffers);
                        if (!consumedLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("A provider layer appears in multiple lifecycles");
                        }
                    }
                    state.lifecycleIndex = providerLifecycles.size();
                    providerLifecycles.push_back(std::move(resolved));
                }

                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (consumedLayers != mLayerIds)
        {
            throw std::invalid_argument("A provider layer is absent from all GPU descriptors");
        }

        // Fail closed until KVCM replaces the batched cuMemcpyBatchAsync copies with kernels: on host
        // kernels that need chunked pinned-memory registration (Linux 6.11-6.13), the embedded lossless
        // codec cannot split its copies at registration boundaries when wrapped by this codec.
        bool hasFallbackLifecycle = false;
        for (auto const& [lifeCycleId, state] : pendingGroups)
        {
            static_cast<void>(lifeCycleId);
            if (!state.lifecycleIndex)
            {
                hasFallbackLifecycle = true;
                break;
            }
        }
        if (hasFallbackLifecycle && kv::HostMem::shouldUseChunkedRegistration())
        {
            throw std::invalid_argument(
                "Cold-page compression is not supported for models with lossless-fallback lifecycles (SSM/GDN) on "
                "this host kernel: chunked pinned-memory registration (Linux 6.11-6.13) breaks the fallback codec's "
                "batched copies. Disable KV cache compression for this model or use a different host kernel.");
        }

        auto const properties = configureProvider(providerLifecycles);
        if (properties.size() != providerLifecycles.size())
        {
            throw std::invalid_argument("Cold-page provider returned an unexpected lifecycle count");
        }
        for (std::size_t index = 0; index < properties.size(); ++index)
        {
            auto const& lifecycle = properties[index];
            if (lifecycle.coldPageBytes == 0U || lifecycle.pageIndexLocation == kv::PageIndexLocation::kBadLocation
                || lifecycle.tokensPerPage < 0
                || (lifecycle.tokensPerPage > 0 && lifecycle.maxColdPageBytes < lifecycle.coldPageBytes))
            {
                throw std::invalid_argument("Cold-page provider returned invalid storage properties");
            }
            auto& state = pendingGroups.at(providerLifecycles[index].lifeCycleId);
            state.coldPageBytes = lifecycle.coldPageBytes;
            state.maxColdPageBytes = std::max(lifecycle.coldPageBytes, lifecycle.maxColdPageBytes);
            state.tokensPerPage = lifecycle.tokensPerPage;
            state.pageIndexLocation = lifecycle.pageIndexLocation;
        }

        mLayerGroups = std::move(pendingGroups);
        mLosslessCodec = std::move(losslessCodec);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::configure rejected GPU layouts: unknown error");
        return false;
    }
}

std::vector<std::size_t> NativeColdPageCodec::coldPageCapacities(kv::LayerGroupId layerGroupId) const
{
    auto const& state = mLayerGroups.at(layerGroupId);
    if (selectsTokens(layerGroupId) && state.maxColdPageBytes > state.coldPageBytes)
    {
        return {state.coldPageBytes, state.maxColdPageBytes};
    }
    return {state.coldPageBytes};
}

bool NativeColdPageCodec::selectsTokens(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state && state->tokensPerPage > 0
        && (mSelection.keepFirstTokens != 0 || mSelection.keepLastTokens != 0 || !mSelection.keepTokenRanges.empty());
}

std::vector<double> NativeColdPageCodec::coldPageCapacityFractions(kv::LayerGroupId layerGroupId, int length) const
{
    if (coldPageCapacities(layerGroupId).size() == 1)
        return {1.0};
    // Before any length observation, reserve both representations. The normal
    // KVCM ratio update uses observed reuse lengths thereafter.
    if (length <= 0)
        return {0.5, 0.5};
    int const pageTokens = mLayerGroups.at(layerGroupId).tokensPerPage;
    int64_t const pages = (int64_t{length} + pageTokens - 1) / pageTokens;
    int64_t rawPages = 0, lastPage = 0;
    for (auto const& [begin, end] : selectTokens(kv::ColdPageContext{0, length, {length}, {}}))
    {
        int64_t const first = begin / pageTokens;
        int64_t const last = (int64_t{end} + pageTokens - 1) / pageTokens;
        rawPages += std::max(int64_t{0}, last - std::max(lastPage, first));
        lastPage = std::max(lastPage, last);
    }
    double const rawFraction = static_cast<double>(rawPages) / pages;
    return {1.0 - rawFraction, rawFraction};
}

std::vector<kv::ColdPageTokenRange> NativeColdPageCodec::selectTokens(kv::ColdPageContext const& context) const
{
    auto ranges = context.retainedProtection;
    auto append = [&](int begin, int end)
    {
        begin = std::max(0, begin - context.startToken);
        end = std::min(context.validTokens, end - context.startToken);
        if (begin < end)
        {
            ranges.emplace_back(begin, end);
        }
    };
    append(0, mSelection.keepFirstTokens);
    for (auto const& [begin, end] : mSelection.keepTokenRanges)
    {
        append(begin, end);
    }
    for (int length : context.sequenceLengths)
    {
        append(std::max(0, length - mSelection.keepLastTokens), length);
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<kv::ColdPageTokenRange> merged;
    for (auto [begin, end] : ranges)
    {
        begin = std::max(begin, 0);
        end = std::min(end, context.validTokens);
        if (begin >= end)
        {
            continue;
        }
        if (!merged.empty() && begin <= merged.back().second)
        {
            merged.back().second = std::max(merged.back().second, end);
        }
        else
        {
            merged.emplace_back(begin, end);
        }
    }
    return merged;
}

std::vector<kv::ColdPageTokenRange> NativeColdPageCodec::protectedTokens(
    kv::LayerGroupId layerGroupId, kv::ColdPageContext const& context) const
{
    return selectsTokens(layerGroupId) ? selectTokens(context) : std::vector<kv::ColdPageTokenRange>{};
}

std::shared_ptr<kv::ColdPageRepresentation const> NativeColdPageCodec::prepareColdPage(
    kv::LayerGroupId layerGroupId, kv::ColdPageContext const& context)
{
    if (!selectsTokens(layerGroupId))
    {
        return nullptr;
    }
    auto& state = mLayerGroups.at(layerGroupId);
    if (context.startToken < 0 || context.validTokens < 0 || context.validTokens > state.tokensPerPage)
    {
        throw std::invalid_argument("Cold-page selection received invalid logical coverage");
    }
    auto rawTokens = selectTokens(context);
    auto key = std::make_pair(context.validTokens, rawTokens);
    auto const found = state.representations.find(key);
    if (found != state.representations.end())
    {
        return found->second;
    }
    auto representation = std::make_shared<kv::ColdPageRepresentation>();
    representation->validTokens = context.validTokens;
    representation->rawTokens = std::move(rawTokens);
    // Recycle unused layouts once the cache reaches 64 entries. Live pages hold shared ownership, so
    // their frozen decode tables cannot be recycled. Reusing an unreferenced
    // ID also replaces the provider's corresponding metadata allocation.
    auto reusable = state.representations.end();
    if (state.representations.size() >= 64)
        reusable = std::find_if(state.representations.begin(), state.representations.end(),
            [](auto const& item) { return item.second.use_count() == 1; });
    if (reusable != state.representations.end())
    {
        representation->layoutId = reusable->second->layoutId;
        state.representations.erase(reusable);
    }
    else
    {
        if (state.nextLayoutId == std::numeric_limits<int>::max())
            throw std::overflow_error("Cold-page layout ID exhausted");
        representation->layoutId = state.nextLayoutId++;
    }
    representation->capacityClass
        = !representation->rawTokens.empty() && state.maxColdPageBytes > state.coldPageBytes ? 1 : 0;
    auto const capacity = coldPageCapacities(layerGroupId).at(representation->capacityClass);
    representation->encodedBytes = prepareProvider(
        *state.lifecycleIndex, representation->layoutId, context.validTokens, representation->rawTokens, capacity);
    if (representation->encodedBytes > capacity)
    {
        throw std::invalid_argument("Selected cold-page payload exceeds its capacity class");
    }
    state.representations.emplace(std::move(key), representation);
    return representation;
}

int NativeColdPageCodec::reusablePrefix(kv::LayerGroupId layerGroupId, int startToken, int sequenceLength,
    int validTokens, std::vector<kv::ColdPageTokenRange> const& losslessTokens) const
{
    if (!selectsTokens(layerGroupId))
    {
        return validTokens;
    }
    kv::ColdPageContext const context{startToken, validTokens, {sequenceLength}, {}};
    for (auto const& [begin, end] : selectTokens(context))
    {
        int covered = begin;
        for (auto const& [rawBegin, rawEnd] : losslessTokens)
        {
            if (rawBegin > covered)
            {
                break;
            }
            if (rawEnd > covered)
            {
                covered = rawEnd;
            }
        }
        if (covered < end)
        {
            return covered;
        }
    }
    return validTokens;
}

std::size_t NativeColdPageCodec::prepareProvider(
    std::size_t, int, int, std::vector<kv::ColdPageTokenRange> const&, std::size_t)
{
    throw std::logic_error("Cold-page provider does not implement selective compression");
}

void NativeColdPageCodec::encodeSelectedProvider(
    std::size_t, int, void*, kv::PageIndexPair const*, std::size_t, cudaStream_t)
{
    throw std::logic_error("Cold-page provider does not implement selective encoding");
}

void NativeColdPageCodec::decodeSelectedProvider(
    std::size_t, int, void const*, kv::PageIndexPair const*, std::size_t, cudaStream_t)
{
    throw std::logic_error("Cold-page provider does not implement selective decoding");
}

NativeColdPageCodec::LayerGroupState const* NativeColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
}

std::size_t NativeColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? 0U : state->coldPageBytes;
}

kv::LayerGroupId NativeColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::LayerGroupId{-1} : layerGroupId;
}

kv::PageIndexLocation NativeColdPageCodec::queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? kv::PageIndexLocation::kBadLocation : state->pageIndexLocation;
}

bool NativeColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    bool providerStarted = false;
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && (dstBasePtr == nullptr || pageIndices == nullptr)))
        {
            throw std::invalid_argument("encode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (!state->lifecycleIndex)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }
        providerStarted = true;
        encodeProvider(*state->lifecycleIndex, dstBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    bool providerStarted = false;
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && (srcBasePtr == nullptr || pageIndices == nullptr)))
        {
            throw std::invalid_argument("decode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (!state->lifecycleIndex)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }
        providerStarted = true;
        decodeProvider(*state->lifecycleIndex, srcBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::encodeSelected(kv::LayerGroupId layerGroupId,
    kv::ColdPageRepresentation const* representation, void* coldBase, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    if (representation == nullptr)
    {
        return encode(layerGroupId, coldBase, pageIndices, numBasePages, stream);
    }
    try
    {
        auto const& state = mLayerGroups.at(layerGroupId);
        encodeSelectedProvider(
            *state.lifecycleIndex, representation->layoutId, coldBase, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        drainAfterProviderFailure(stream);
        TLLM_LOG_ERROR("Selective cold-page encode failed: %s", error.what());
        return false;
    }
    catch (...)
    {
        drainAfterProviderFailure(stream);
        TLLM_LOG_ERROR("Selective cold-page encode failed: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::decodeSelected(kv::LayerGroupId layerGroupId,
    kv::ColdPageRepresentation const* representation, void const* coldBase, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    if (representation == nullptr)
    {
        return decode(layerGroupId, coldBase, pageIndices, numBasePages, stream);
    }
    try
    {
        auto const& state = mLayerGroups.at(layerGroupId);
        decodeSelectedProvider(
            *state.lifecycleIndex, representation->layoutId, coldBase, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        drainAfterProviderFailure(stream);
        TLLM_LOG_ERROR("Selective cold-page decode failed: %s", error.what());
        return false;
    }
    catch (...)
    {
        drainAfterProviderFailure(stream);
        TLLM_LOG_ERROR("Selective cold-page decode failed: unknown error");
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
