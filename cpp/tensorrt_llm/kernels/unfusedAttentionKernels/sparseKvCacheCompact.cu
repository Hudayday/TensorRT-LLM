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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/sparseKvCacheCompact.h"
#include "unfusedAttentionKernels_2_template.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! KVCacheBuffer adapter over ONE KVCacheManagerV2 per-layer HND pool:
//! [num_pages, 2 (K/V), num_kv_heads, tokens_per_block, head_dim].
//! ``page_table`` maps (sequence, logical block) -> physical page. Only the
//! three accessors used by updateSparseKvCacheAfterFmha are provided.
struct KvCacheV2PoolBuffer
{
    uint8_t* pool;
    int32_t const* page_table; // [batch, max_pages_per_seq]
    int32_t max_pages_per_seq;
    int32_t num_kv_heads;
    int32_t tokens_per_block;
    size_t bytes_per_page;    // 2 * num_kv_heads * tokens_per_block * head_dim * sizeof(T)
    size_t bytes_per_kv_half; // bytes_per_page / 2

    __device__ __forceinline__ void* getKBlockPtr(int batch_idx, int token_idx) const
    {
        int32_t const page = page_table[batch_idx * max_pages_per_seq + token_idx / tokens_per_block];
        return pool + static_cast<size_t>(page) * bytes_per_page;
    }

    __device__ __forceinline__ void* getVBlockPtr(int batch_idx, int token_idx) const
    {
        int32_t const page = page_table[batch_idx * max_pages_per_seq + token_idx / tokens_per_block];
        return pool + static_cast<size_t>(page) * bytes_per_page + bytes_per_kv_half;
    }

    //! uint4 index within one K (or V) page half for (token slot, head, vec).
    __device__ __forceinline__ int getKVLocalIdx(
        int token_idx, int head_idx, int vecs_per_head, int head_vec_idx) const
    {
        return (head_idx * tokens_per_block + (token_idx % tokens_per_block)) * vecs_per_head + head_vec_idx;
    }
};

template <typename T>
void invokeSparseKvCacheCompactV2(T* pool, int32_t const* page_table, int32_t max_pages_per_seq,
    int32_t const* sparse_kv_indices, int32_t const* sparse_kv_offsets, int32_t batch_size, int32_t num_kv_heads,
    int32_t tokens_per_block, int32_t head_dim, cudaStream_t stream)
{
    KvCacheV2PoolBuffer buffer{};
    buffer.pool = reinterpret_cast<uint8_t*>(pool);
    buffer.page_table = page_table;
    buffer.max_pages_per_seq = max_pages_per_seq;
    buffer.num_kv_heads = num_kv_heads;
    buffer.tokens_per_block = tokens_per_block;
    buffer.bytes_per_kv_half
        = static_cast<size_t>(num_kv_heads) * tokens_per_block * head_dim * sizeof(T);
    buffer.bytes_per_page = 2 * buffer.bytes_per_kv_half;

    QKVPreprocessingParams<T, KvCacheV2PoolBuffer> params;
    memset(&params, 0, sizeof(params));
    params.kv_cache_buffer = buffer;
    params.sparse_kv_indices = sparse_kv_indices;
    params.sparse_kv_offsets = sparse_kv_offsets;
    params.batch_size = batch_size;
    params.kv_head_num = num_kv_heads;
    params.size_per_head = head_dim;

    invokeUpdateSparseKvCacheAfterFmha<T, T, KvCacheV2PoolBuffer>(params, stream);
}

template void invokeSparseKvCacheCompactV2<half>(half*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeSparseKvCacheCompactV2<float>(float*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t, int32_t, int32_t, int32_t, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeSparseKvCacheCompactV2<__nv_bfloat16>(__nv_bfloat16*, int32_t const*, int32_t, int32_t const*,
    int32_t const*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
