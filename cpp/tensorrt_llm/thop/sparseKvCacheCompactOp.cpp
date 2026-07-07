/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! In-place sparse compaction of one KVCacheManagerV2 per-layer HND pool.
//! pool:              [num_pages, 2, num_kv_heads, tokens_per_block, head_dim] (mutated)
//! page_table:        [batch, max_pages_per_seq] int32, logical block -> physical page
//! sparse_kv_indices: [num_kv_heads, total_kept] int32, per-head kept token lists
//! sparse_kv_offsets: [batch + 1] int32, per-request ranges into the kept lists
void sparse_kv_cache_compact(th::Tensor pool, th::Tensor const& page_table,
    th::Tensor const& sparse_kv_indices, th::Tensor const& sparse_kv_offsets)
{
    TORCH_CHECK(pool.is_cuda() && page_table.is_cuda() && sparse_kv_indices.is_cuda() && sparse_kv_offsets.is_cuda(),
        "sparse_kv_cache_compact: all tensors must be CUDA tensors");
    TORCH_CHECK(pool.dim() == 5 && pool.size(1) == 2,
        "sparse_kv_cache_compact: pool must be [pages, 2, kv_heads, tokens_per_block, head_dim]");
    TORCH_CHECK(pool.is_contiguous(), "sparse_kv_cache_compact: pool must be contiguous");
    TORCH_CHECK(page_table.dim() == 2 && page_table.scalar_type() == th::kInt32 && page_table.is_contiguous(),
        "sparse_kv_cache_compact: page_table must be contiguous [batch, max_pages] int32");
    TORCH_CHECK(sparse_kv_indices.dim() == 2 && sparse_kv_indices.scalar_type() == th::kInt32
            && sparse_kv_indices.is_contiguous(),
        "sparse_kv_cache_compact: sparse_kv_indices must be contiguous [kv_heads, total] int32");
    TORCH_CHECK(sparse_kv_offsets.dim() == 1 && sparse_kv_offsets.scalar_type() == th::kInt32
            && sparse_kv_offsets.is_contiguous(),
        "sparse_kv_cache_compact: sparse_kv_offsets must be contiguous [batch + 1] int32");

    auto const num_kv_heads = static_cast<int32_t>(pool.size(2));
    auto const tokens_per_block = static_cast<int32_t>(pool.size(3));
    auto const head_dim = static_cast<int32_t>(pool.size(4));
    auto const batch_size = static_cast<int32_t>(page_table.size(0));
    auto const max_pages_per_seq = static_cast<int32_t>(page_table.size(1));
    TORCH_CHECK(sparse_kv_indices.size(0) == num_kv_heads,
        "sparse_kv_cache_compact: sparse_kv_indices head dimension mismatch");
    TORCH_CHECK(sparse_kv_offsets.size(0) == batch_size + 1,
        "sparse_kv_cache_compact: sparse_kv_offsets must have batch + 1 entries");

    auto stream = at::cuda::getCurrentCUDAStream(pool.get_device());
    auto const dtype = pool.scalar_type();
    if (dtype == th::kBFloat16)
    {
        tk::invokeSparseKvCacheCompactV2<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(pool.data_ptr()), page_table.data_ptr<int32_t>(), max_pages_per_seq,
            sparse_kv_indices.data_ptr<int32_t>(), sparse_kv_offsets.data_ptr<int32_t>(), batch_size, num_kv_heads,
            tokens_per_block, head_dim, stream);
    }
    else if (dtype == th::kHalf)
    {
        tk::invokeSparseKvCacheCompactV2<half>(reinterpret_cast<half*>(pool.data_ptr()),
            page_table.data_ptr<int32_t>(), max_pages_per_seq, sparse_kv_indices.data_ptr<int32_t>(),
            sparse_kv_offsets.data_ptr<int32_t>(), batch_size, num_kv_heads, tokens_per_block, head_dim, stream);
    }
    else if (dtype == th::kFloat)
    {
        tk::invokeSparseKvCacheCompactV2<float>(pool.data_ptr<float>(), page_table.data_ptr<int32_t>(),
            max_pages_per_seq, sparse_kv_indices.data_ptr<int32_t>(), sparse_kv_offsets.data_ptr<int32_t>(),
            batch_size, num_kv_heads, tokens_per_block, head_dim, stream);
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
        "sparse_kv_cache_compact(Tensor(a!) pool, Tensor page_table, Tensor sparse_kv_indices, "
        "Tensor sparse_kv_offsets) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_kv_cache_compact", &tensorrt_llm::torch_ext::sparse_kv_cache_compact);
}
