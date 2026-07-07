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

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! In-place sparse KV compaction for one KVCacheManagerV2 per-layer HND pool
//! (general: any eviction method that produces per-head kept-token lists)
//! ([num_pages, 2, num_kv_heads, tokens_per_block, head_dim], K/V interleaved
//! per page). Reuses the attention sparse-compaction kernel
//! (updateSparseKvCacheAfterFmha) with a V2-pool KVCacheBuffer adapter:
//! kept tokens (per KV head lists in ``sparse_kv_indices`` laid out
//! [num_kv_heads, total]) are gathered to each request's sequence prefix.
template <typename T>
void invokeSparseKvCacheCompactV2(T* pool, int32_t const* page_table, int32_t max_pages_per_seq,
    int32_t const* sparse_kv_indices, int32_t const* sparse_kv_offsets, int32_t batch_size, int32_t num_kv_heads,
    int32_t tokens_per_block, int32_t head_dim, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
