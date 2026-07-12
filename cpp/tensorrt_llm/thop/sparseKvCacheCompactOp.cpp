/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/sparseKvCacheCompact.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <optional>
#include <vector>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Compact one uniform group of KVCacheManagerV2 layer pools with one CUDA
//! kernel launch. poolPointers and pageTablePointers are persistent device
//! arrays containing the addresses of the corresponding tensors. Keeping the
//! tensors in the argument lists makes ownership and mutation explicit while
//! the device pointer arrays avoid per-replay host staging.
void sparseKvCacheCompactLayers(std::vector<th::Tensor> const& pools, th::Tensor const& poolPointers,
    std::vector<th::Tensor> const& pageTables, th::Tensor const& pageTablePointers, th::Tensor const& sourceIndices,
    th::Tensor const& sourceOffsets, std::optional<th::Tensor> const& sourceLayerIndices,
    std::optional<th::Tensor> const& destinationIndices)
{
    TORCH_CHECK(!pools.empty(), "sparse_kv_cache_compact_layers: pools must be non-empty");
    TORCH_CHECK(
        pageTables.size() == pools.size(), "sparse_kv_cache_compact_layers: page_tables must have one entry per pool");

    auto const& firstPool = pools.front();
    auto const& firstPageTable = pageTables.front();
    TORCH_CHECK(firstPool.is_cuda(), "sparse_kv_cache_compact_layers: pools must be CUDA tensors");
    TORCH_CHECK(firstPool.dim() == 5 && firstPool.size(1) == 2,
        "sparse_kv_cache_compact_layers: every pool must be [pages, 2, kv_heads, tokens_per_block, head_dim]");
    TORCH_CHECK(firstPool.is_contiguous(), "sparse_kv_cache_compact_layers: pools must be contiguous");
    TORCH_CHECK(firstPageTable.is_cuda() && firstPageTable.dim() == 2 && firstPageTable.scalar_type() == th::kInt32
            && firstPageTable.is_contiguous(),
        "sparse_kv_cache_compact_layers: page tables must be contiguous CUDA [batch, max_pages] int32 tensors");

    auto const device = firstPool.get_device();
    auto const dtype = firstPool.scalar_type();
    auto const numLayers = static_cast<int32_t>(pools.size());
    auto const numKvHeads = static_cast<int32_t>(firstPool.size(2));
    auto const tokensPerBlock = static_cast<int32_t>(firstPool.size(3));
    auto const headDim = static_cast<int32_t>(firstPool.size(4));
    auto const batchSize = static_cast<int32_t>(firstPageTable.size(0));
    auto const maxPagesPerSeq = static_cast<int32_t>(firstPageTable.size(1));

    for (int32_t layer = 0; layer < numLayers; ++layer)
    {
        auto const& pool = pools[layer];
        auto const& pageTable = pageTables[layer];
        TORCH_CHECK(pool.is_cuda() && pool.get_device() == device && pool.scalar_type() == dtype && pool.dim() == 5
                && pool.size(1) == 2 && pool.is_contiguous(),
            "sparse_kv_cache_compact_layers: all pools must have one device, dtype, layout, and contiguous storage");
        TORCH_CHECK(pool.size(2) == numKvHeads && pool.size(3) == tokensPerBlock && pool.size(4) == headDim,
            "sparse_kv_cache_compact_layers: all pools must share KV-head, block, and head-dimension geometry");
        TORCH_CHECK(pageTable.is_cuda() && pageTable.get_device() == device && pageTable.scalar_type() == th::kInt32
                && pageTable.is_contiguous() && pageTable.sizes() == firstPageTable.sizes(),
            "sparse_kv_cache_compact_layers: all page tables must share one CUDA int32 geometry");
    }

    auto const checkPointerArray = [device, numLayers](th::Tensor const& pointers, char const* name)
    {
        TORCH_CHECK(pointers.is_cuda() && pointers.get_device() == device && pointers.scalar_type() == th::kInt64
                && pointers.dim() == 1 && pointers.size(0) == numLayers && pointers.is_contiguous(),
            "sparse_kv_cache_compact_layers: ", name, " must be contiguous CUDA int64 [num_layers]");
    };
    checkPointerArray(poolPointers, "pool_pointers");
    checkPointerArray(pageTablePointers, "page_table_pointers");

    TORCH_CHECK(sourceIndices.is_cuda() && sourceIndices.get_device() == device
            && sourceIndices.scalar_type() == th::kInt32 && sourceIndices.is_contiguous()
            && (sourceIndices.dim() == 2 || sourceIndices.dim() == 3),
        "sparse_kv_cache_compact_layers: source_indices must be contiguous CUDA int32 [kv_heads, total] or "
        "[layers, kv_heads, total]");
    int64_t sourceLayerStride = 0;
    int32_t const* sourceLayerPtr = nullptr;
    if (sourceIndices.dim() == 2)
    {
        TORCH_CHECK(sourceIndices.size(0) == numKvHeads,
            "sparse_kv_cache_compact_layers: shared source_indices KV-head dimension mismatch");
    }
    else
    {
        TORCH_CHECK(sourceIndices.size(0) > 0 && sourceIndices.size(1) == numKvHeads,
            "sparse_kv_cache_compact_layers: per-layer source_indices geometry mismatch");
        TORCH_CHECK(sourceLayerIndices.has_value(),
            "sparse_kv_cache_compact_layers: 3D source_indices require source_layer_indices");
        sourceLayerStride = sourceIndices.stride(0);
    }
    if (sourceLayerIndices.has_value())
    {
        auto const& layerIndices = *sourceLayerIndices;
        TORCH_CHECK(layerIndices.is_cuda() && layerIndices.get_device() == device
                && layerIndices.scalar_type() == th::kInt32 && layerIndices.is_contiguous() && layerIndices.dim() == 1
                && layerIndices.size(0) == numLayers,
            "sparse_kv_cache_compact_layers: source_layer_indices must be contiguous CUDA int32 [num_layers]");
        sourceLayerPtr = layerIndices.data_ptr<int32_t>();
    }
    auto const totalMoves = sourceIndices.size(-1);

    TORCH_CHECK(sourceOffsets.is_cuda() && sourceOffsets.get_device() == device
            && sourceOffsets.scalar_type() == th::kInt32 && sourceOffsets.is_contiguous() && sourceOffsets.dim() == 1
            && sourceOffsets.size(0) == batchSize + 1,
        "sparse_kv_cache_compact_layers: source_offsets must be contiguous CUDA int32 [batch + 1]");

    int32_t const* destinationPtr = nullptr;
    int64_t destinationLayerStride = 0;
    int64_t destinationHeadStride = 0;
    if (destinationIndices.has_value())
    {
        auto const& destination = *destinationIndices;
        TORCH_CHECK(destination.is_cuda() && destination.get_device() == device
                && destination.scalar_type() == th::kInt32 && destination.is_contiguous()
                && (destination.dim() == 1 || destination.dim() == 2 || destination.dim() == 3),
            "sparse_kv_cache_compact_layers: destination_indices must be contiguous CUDA int32 [total], "
            "[kv_heads, total], or [layers, kv_heads, total]");
        TORCH_CHECK(destination.size(-1) == totalMoves,
            "sparse_kv_cache_compact_layers: destination total-move dimension mismatch");
        if (destination.dim() >= 2)
        {
            TORCH_CHECK(destination.size(-2) == numKvHeads,
                "sparse_kv_cache_compact_layers: destination KV-head dimension mismatch");
            destinationHeadStride = destination.stride(-2);
        }
        if (destination.dim() == 3)
        {
            TORCH_CHECK(destination.size(0) == numLayers,
                "sparse_kv_cache_compact_layers: destination layer dimension mismatch");
            destinationLayerStride = destination.stride(0);
        }
        destinationPtr = destination.data_ptr<int32_t>();
    }

    auto stream = at::cuda::getCurrentCUDAStream(device);
    if (dtype == th::kBFloat16)
    {
        tk::invokeSparseKvCacheCompactV2Layers<__nv_bfloat16>(poolPointers.data_ptr<int64_t>(),
            pageTablePointers.data_ptr<int64_t>(), numLayers, maxPagesPerSeq, sourceIndices.data_ptr<int32_t>(),
            sourceLayerPtr, sourceLayerStride, sourceOffsets.data_ptr<int32_t>(), destinationPtr,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, headDim, stream);
    }
    else if (dtype == th::kHalf)
    {
        tk::invokeSparseKvCacheCompactV2Layers<half>(poolPointers.data_ptr<int64_t>(),
            pageTablePointers.data_ptr<int64_t>(), numLayers, maxPagesPerSeq, sourceIndices.data_ptr<int32_t>(),
            sourceLayerPtr, sourceLayerStride, sourceOffsets.data_ptr<int32_t>(), destinationPtr,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, headDim, stream);
    }
    else if (dtype == th::kFloat)
    {
        tk::invokeSparseKvCacheCompactV2Layers<float>(poolPointers.data_ptr<int64_t>(),
            pageTablePointers.data_ptr<int64_t>(), numLayers, maxPagesPerSeq, sourceIndices.data_ptr<int32_t>(),
            sourceLayerPtr, sourceLayerStride, sourceOffsets.data_ptr<int32_t>(), destinationPtr,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, headDim, stream);
    }
    else
    {
        TORCH_CHECK(false, "sparse_kv_cache_compact_layers: unsupported pool dtype ", dtype);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sparse_kv_cache_compact_layers(Tensor(a!)[] pools, Tensor pool_pointers, Tensor[] page_tables, Tensor "
        "page_table_pointers, Tensor source_indices, Tensor source_offsets, Tensor? source_layer_indices=None, Tensor? "
        "destination_indices=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_kv_cache_compact_layers", &tensorrt_llm::torch_ext::sparseKvCacheCompactLayers);
}
