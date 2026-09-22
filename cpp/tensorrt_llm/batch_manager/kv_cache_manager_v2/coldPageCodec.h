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

#pragma once

#include "kv_cache_manager_v2/storage/config.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Describes one physical pool in the hot storage representation.
struct PoolDesc
{
    PoolIndex poolIndex{0};
    MemAddress baseAddress = 0;
    size_t slotBytes = 0;
};

//! Describes one hot pool group and the lifecycles that share its physical layout.
struct PoolGroupDesc
{
    PoolGroupIndex poolGroupIndex{0};
    SlotCount numSlots = 0;
    SlotDesc slotDesc;
    TypedVec<PoolIndex, PoolDesc> pools;
};

//! Location of the page-index array passed to encode() and decode().
enum class PageIndexLocation : int
{
    //! Indicates failure or an unknown layer group; never a valid index-array location.
    kBadLocation = -1,
    //! PageIndexPair array resides in host memory.
    kHost,
    //! PageIndexPair array resides in device memory.
    kDevice,
};

//! Source and destination page indices for one logical page conversion.
//!
//! Caller contract: both indices are non-negative and in range for their pool. KVCM2 derives them
//! from concrete allocated pages, so they are valid by construction. Implementations are entitled
//! to use them unchecked -- validating costs an O(numBasePages) scan on the eviction critical path,
//! and on the device-array path the values are not host-readable at all. A negative index
//! sign-extends into a wild device address rather than faulting cleanly, so a violation is memory
//! corruption, not a caught error.
struct alignas(8) PageIndexPair
{
    int32_t dst;
    int32_t src;
};

static_assert(sizeof(PageIndexPair) == 8);
static_assert(std::is_trivially_copyable_v<PageIndexPair>);

//! Half-open token interval in a page, independent of its physical slot.
using ColdPageTokenRange = std::pair<int, int>;

//! A request's confirmed KV endpoint. Closed requests leave a frozen endpoint.
struct ColdPageSequence
{
    int length = 0;
    bool closed = false;
};

//! Frozen selection describing stored bytes. A larger dtype does not undo this history.
struct ColdPageRepresentation
{
    int validTokens = 0;
    std::vector<ColdPageTokenRange> rawTokens;
    int layoutId = 0;
    int capacityClass = 0;
    size_t encodedBytes = 0;
};

//! Logical inputs supplied before allocating a cold destination.
struct ColdPageContext
{
    int startToken = 0;
    int validTokens = 0;
    std::vector<int> sequenceLengths;
    std::vector<ColdPageTokenRange> retainedProtection;
};

//! Transforms KV pages between hot multi-pool and cold single-blob representations.
class IKvCacheColdPageCodec
{
public:
    IKvCacheColdPageCodec();
    virtual ~IKvCacheColdPageCodec();

    IKvCacheColdPageCodec(IKvCacheColdPageCodec const&) = delete;
    IKvCacheColdPageCodec& operator=(IKvCacheColdPageCodec const&) = delete;

    //! Configures all hot GPU pool groups in one call. KVCM calls this method exactly once.
    //!
    //! gpuDescs points to numGpuDescs contiguous descriptors ordered by PoolGroupIndex. Each descriptor must contain
    //! the matching poolGroupIndex. Returns false when the pointer, count, or any descriptor is invalid or unsupported.
    virtual bool configure(PoolGroupDesc const* gpuDescs, PoolGroupIndex numGpuDescs) noexcept = 0;

    //! Returns the fixed cold-page size for a layer group. Zero indicates failure or an unknown layer group.
    [[nodiscard]] virtual size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept = 0;

    //! Capacity classes, in increasing byte size; the default codec has one fixed class.
    [[nodiscard]] virtual std::vector<size_t> coldPageCapacities(LayerGroupId layerGroupId) const;

    //! Whether logical token selection is enabled for this lifecycle.
    [[nodiscard]] virtual bool selectsTokens(LayerGroupId layerGroupId) const noexcept;
    virtual std::vector<double> coldPageCapacityFractions(LayerGroupId layerGroupId, int historyLength) const;
    virtual std::vector<ColdPageTokenRange> protectedTokens(
        LayerGroupId layerGroupId, ColdPageContext const& context) const;

    //! Resolve and intern a representation before destination allocation. Null selects the fixed format.
    [[nodiscard]] virtual std::shared_ptr<ColdPageRepresentation const> prepareColdPage(
        LayerGroupId layerGroupId, ColdPageContext const& context);

    //! First reusable token prefix under the requested endpoint, accounting for prior lossy storage.
    [[nodiscard]] virtual int reusablePrefix(LayerGroupId layerGroupId, int startToken, int sequenceLength,
        int validTokens, std::vector<ColdPageTokenRange> const& losslessTokens) const;

    //! Execute a batch with one frozen representation. Existing codecs retain their fixed-format path.
    virtual bool encodeSelected(LayerGroupId layerGroupId, ColdPageRepresentation const* representation,
        void* dstBasePtr, PageIndexPair const* pageIndices, size_t numBasePages, cudaStream_t stream) noexcept;
    virtual bool decodeSelected(LayerGroupId layerGroupId, ColdPageRepresentation const* representation,
        void const* srcBasePtr, PageIndexPair const* pageIndices, size_t numBasePages, cudaStream_t stream) noexcept;

    //! Returns the representative layer-group ID used for cross-lifecycle batching.
    //!
    //! KVCM may concatenate page-index arrays for lifecycles that return the same ID and issue one encode or decode
    //! call using that ID. Equal IDs promise identical codec behavior, including the algorithm, parameters,
    //! cold-page size, encoded representation, and page-index location. The returned ID must be the smallest lifecycle
    //! ID in that codec-equivalence class, and all members must belong to the same configured GPU pool group.
    //!
    //! Returns a negative layer-group ID on failure or for an unknown layer group.
    //!
    //! The default implementation returns layerGroupId, disabling cross-lifecycle batching.
    [[nodiscard]] virtual LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept;

    //! Returns the memory location of the PageIndexPair array used by both encode and decode.
    //!
    //! Lifecycles for which getBatchingLayerGroupId() returns the same ID must return the same location. Returns
    //! PageIndexLocation::kBadLocation on failure or for an unknown layer group.
    [[nodiscard]] virtual PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept = 0;

    //! Encodes hot pages into cold pages.
    //!
    //! The cold base pointer is GPU-accessible. The index-array location is selected by queryPageIndexLocation(). Host
    //! arrays remain valid until this method returns; device arrays remain valid until work enqueued on stream
    //! completes. The pairs may have been concatenated from multiple codec-equivalent lifecycles.
    //!
    //! Every pageIndices entry must satisfy the PageIndexPair contract; passing an invalid index is
    //! undefined behaviour rather than a `false` return.
    virtual bool encode(LayerGroupId layerGroupId, void* dstBasePtr, PageIndexPair const* pageIndices,
        size_t numBasePages, cudaStream_t stream) noexcept
        = 0;

    //! Decodes cold pages into hot pages.
    //!
    //! The cold base pointer is GPU-accessible. The index-array location is selected by queryPageIndexLocation(). Host
    //! arrays remain valid until this method returns; device arrays remain valid until work enqueued on stream
    //! completes. The pairs may have been concatenated from multiple codec-equivalent lifecycles.
    //!
    //! Every pageIndices entry must satisfy the PageIndexPair contract; passing an invalid index is
    //! undefined behaviour rather than a `false` return.
    virtual bool decode(LayerGroupId layerGroupId, void const* srcBasePtr, PageIndexPair const* pageIndices,
        size_t numBasePages, cudaStream_t stream) noexcept
        = 0;
};

//! Creates the lossless default codec that concatenates hot pools into one cold-page blob.
[[nodiscard]] std::unique_ptr<IKvCacheColdPageCodec> createDefaultKvCacheColdPageCodec();

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
