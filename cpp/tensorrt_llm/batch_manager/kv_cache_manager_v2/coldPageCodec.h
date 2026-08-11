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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct PoolGroupDesc;

//! Location of the Page-index array passed to encode() and decode().
enum class PageIndexLocation
{
    kHost,
    kDevice,
};

//! Source and destination Page indices for one logical Page conversion.
struct alignas(8) PageIndexPair
{
    std::int32_t dst;
    std::int32_t src;
};

static_assert(sizeof(PageIndexPair) == 8);
static_assert(std::is_trivially_copyable_v<PageIndexPair>);

//! Transform hook invoked by KVCM when Pages cross a hot/cold layout boundary.
//!
//! KVCM owns Page selection, Slot allocation, stream/event ordering, publish,
//! rollback, and eviction. A compression manager owns one concrete codec,
//! configures it from KVCM's authoritative GPU Pool descriptors, and registers
//! that same native object with KVCM. Runtime calls stay entirely in C++.
class IKvCacheColdPageCodec
{
public:
    IKvCacheColdPageCodec();
    virtual ~IKvCacheColdPageCodec();

    IKvCacheColdPageCodec(IKvCacheColdPageCodec const&) = delete;
    IKvCacheColdPageCodec& operator=(IKvCacheColdPageCodec const&) = delete;

    virtual bool configure(PoolGroupDesc const& gpuDesc) noexcept = 0;

    //! Query the cold-page stride for one layer group. Zero indicates failure
    //! or an unknown layer group.
    [[nodiscard]] virtual std::size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept = 0;

    //! Return the representative layer-group ID used for cross-lifecycle batching.
    //!
    //! KVCM may concatenate Page-index arrays for lifecycles that return the
    //! same ID and issue one encode or decode call using that ID. Equal IDs
    //! promise identical codec behavior, including the algorithm, parameters,
    //! cold-Page size, encoded representation, and Page-index location. The
    //! returned ID must be the smallest lifecycle ID in that codec-equivalence
    //! class, and all members must belong to the same configured GPU PoolGroup.
    //!
    //! The default keeps each lifecycle in its own codec call.
    [[nodiscard]] virtual LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept;

    //! Return the required memory location for the PageIndexPair array.
    [[nodiscard]] virtual PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept = 0;

    //! Enqueue GPU hot Pages -> cold Pages on stream.
    virtual bool encode(LayerGroupId layerGroupId, void* dstBasePtr, PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept
        = 0;

    //! Enqueue cold Pages -> GPU hot Pages on stream.
    virtual bool decode(LayerGroupId layerGroupId, void const* srcBasePtr, PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept
        = 0;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
