/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *All rights reserved.
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

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! In-place sparse compaction of one KVCacheManagerV2 per-layer HND pool.
//! pool:                [pages, 2, kv_heads, tokens_per_block, head_dim]
//! pageTable:           [batch, max_pages_per_seq] int32
//! sourceIndices:       [kv_heads, total_moves] int32
//! sourceOffsets:       [batch + 1] int32
//! destinationIndices: optional [total_moves] or [kv_heads, total_moves] int32
//! Per request/head, source and destination indices must be strictly increasing
//! with destination[i] <= source[i].
void sparseKvCacheCompact(th::Tensor pool, th::Tensor const& pageTable, th::Tensor const& sourceIndices,
    th::Tensor const& sourceOffsets, std::optional<th::Tensor> const& destinationIndices)
{
    TORCH_CHECK(pool.is_cuda() && pageTable.is_cuda() && sourceIndices.is_cuda() && sourceOffsets.is_cuda(),
        "sparse_kv_cache_compact: all required tensors must be CUDA tensors");
    TORCH_CHECK(pageTable.get_device() == pool.get_device() && sourceIndices.get_device() == pool.get_device()
            && sourceOffsets.get_device() == pool.get_device(),
        "sparse_kv_cache_compact: all tensors must be on the pool device");
    TORCH_CHECK(pool.dim() == 5 && pool.size(1) == 2,
        "sparse_kv_cache_compact: pool must be [pages, 2, kv_heads, "
        "tokens_per_block, head_dim]");
    TORCH_CHECK(pool.is_contiguous(), "sparse_kv_cache_compact: pool must be contiguous");
    TORCH_CHECK(pageTable.dim() == 2 && pageTable.scalar_type() == th::kInt32 && pageTable.is_contiguous(),
        "sparse_kv_cache_compact: page_table must be contiguous [batch, "
        "max_pages] int32");
    TORCH_CHECK(sourceIndices.dim() == 2 && sourceIndices.scalar_type() == th::kInt32 && sourceIndices.is_contiguous(),
        "sparse_kv_cache_compact: source_indices must be contiguous "
        "[kv_heads, total] int32");
    TORCH_CHECK(sourceOffsets.dim() == 1 && sourceOffsets.scalar_type() == th::kInt32 && sourceOffsets.is_contiguous(),
        "sparse_kv_cache_compact: source_offsets must be contiguous "
        "[batch + 1] int32");

    auto const numKvHeads = static_cast<int32_t>(pool.size(2));
    auto const tokensPerBlock = static_cast<int32_t>(pool.size(3));
    auto const headDim = static_cast<int32_t>(pool.size(4));
    auto const batchSize = static_cast<int32_t>(pageTable.size(0));
    auto const maxPagesPerSeq = static_cast<int32_t>(pageTable.size(1));
    TORCH_CHECK(sourceIndices.size(0) == numKvHeads, "sparse_kv_cache_compact: source_indices head dimension mismatch");
    TORCH_CHECK(
        sourceOffsets.size(0) == batchSize + 1, "sparse_kv_cache_compact: source_offsets must have batch + 1 entries");

    int32_t const* destinationPtr = nullptr;
    bool destinationPerHead = false;
    if (destinationIndices.has_value())
    {
        auto const& destination = *destinationIndices;
        TORCH_CHECK(destination.is_cuda() && destination.get_device() == pool.get_device(),
            "sparse_kv_cache_compact: destination_indices must be on the "
            "pool CUDA device");
        TORCH_CHECK(destination.scalar_type() == th::kInt32 && destination.is_contiguous(),
            "sparse_kv_cache_compact: destination_indices must be "
            "contiguous int32");
        TORCH_CHECK(destination.dim() == 1 || destination.dim() == 2,
            "sparse_kv_cache_compact: destination_indices must be [total] "
            "or [kv_heads, total]");
        if (destination.dim() == 1)
        {
            TORCH_CHECK(destination.size(0) == sourceIndices.size(1),
                "sparse_kv_cache_compact: shared destination size mismatch");
        }
        else
        {
            TORCH_CHECK(destination.sizes() == sourceIndices.sizes(),
                "sparse_kv_cache_compact: per-head destination shape mismatch");
            destinationPerHead = true;
        }
        destinationPtr = destination.data_ptr<int32_t>();
    }

    auto stream = at::cuda::getCurrentCUDAStream(pool.get_device());
    auto const dtype = pool.scalar_type();
    if (dtype == th::kBFloat16)
    {
        tk::invokeSparseKvCacheCompactV2<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(pool.data_ptr()),
            pageTable.data_ptr<int32_t>(), maxPagesPerSeq, sourceIndices.data_ptr<int32_t>(),
            sourceOffsets.data_ptr<int32_t>(), destinationPtr, destinationPerHead, batchSize, numKvHeads,
            tokensPerBlock, headDim, stream);
    }
    else if (dtype == th::kHalf)
    {
        tk::invokeSparseKvCacheCompactV2<half>(reinterpret_cast<half*>(pool.data_ptr()), pageTable.data_ptr<int32_t>(),
            maxPagesPerSeq, sourceIndices.data_ptr<int32_t>(), sourceOffsets.data_ptr<int32_t>(), destinationPtr,
            destinationPerHead, batchSize, numKvHeads, tokensPerBlock, headDim, stream);
    }
    else if (dtype == th::kFloat)
    {
        tk::invokeSparseKvCacheCompactV2<float>(pool.data_ptr<float>(), pageTable.data_ptr<int32_t>(), maxPagesPerSeq,
            sourceIndices.data_ptr<int32_t>(), sourceOffsets.data_ptr<int32_t>(), destinationPtr, destinationPerHead,
            batchSize, numKvHeads, tokensPerBlock, headDim, stream);
    }
    else
    {
        TORCH_CHECK(false, "sparse_kv_cache_compact: unsupported pool dtype ", dtype);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sparse_kv_cache_compact(Tensor(a!) pool, Tensor page_table, Tensor "
        "source_indices, "
        "Tensor source_offsets, Tensor? destination_indices=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_kv_cache_compact", &tensorrt_llm::torch_ext::sparseKvCacheCompact);
}
