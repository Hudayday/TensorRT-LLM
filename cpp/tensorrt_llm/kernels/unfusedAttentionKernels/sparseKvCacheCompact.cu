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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/sparseKvCacheCompact.h"
#include "unfusedAttentionKernels_2_template.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! KVCacheBuffer adapter over one KVCacheManagerV2 per-layer HND pool:
//! [num_pages, 2 (K/V), num_kv_heads, tokens_per_block, head_dim].
struct KvCacheV2PoolBuffer
{
    uint8_t* pool;
    int32_t const* pageTable;
    int32_t maxPagesPerSeq;
    int32_t tokensPerBlock;
    size_t bytesPerPage;
    size_t bytesPerKvHalf;

    __device__ __forceinline__ void* getKBlockPtr(int batchIdx, int tokenIdx) const
    {
        int32_t const page = pageTable[batchIdx * maxPagesPerSeq + tokenIdx / tokensPerBlock];
        return pool + static_cast<size_t>(page) * bytesPerPage;
    }

    __device__ __forceinline__ void* getVBlockPtr(int batchIdx, int tokenIdx) const
    {
        int32_t const page = pageTable[batchIdx * maxPagesPerSeq + tokenIdx / tokensPerBlock];
        return pool + static_cast<size_t>(page) * bytesPerPage + bytesPerKvHalf;
    }

    __device__ __forceinline__ int getKVLocalIdx(int tokenIdx, int headIdx, int valuesPerHead, int headValueIdx) const
    {
        return (headIdx * tokensPerBlock + (tokenIdx % tokensPerBlock)) * valuesPerHead + headValueIdx;
    }
};

//! Runtime-head-dimension path. One block owns one request/head and visits
//! fixed-size token chunks in ascending destination order. TriAttention
//! supplies sorted sources and monotonically increasing destinations with
//! destination <= source. Loading a complete chunk before its stores makes
//! these overlapping left compactions safe without a global scratch allocation.
template <typename T>
__global__ void sparseKvCacheCompactV2Runtime(KvCacheV2PoolBuffer buffer, int32_t const* sparseKvIndices,
    int32_t const* sparseKvOffsets, int32_t const* destinationIndices, bool destinationPerHead, int32_t batchSize,
    int32_t headDim)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const kvHeadIdx = blockIdx.y;
    int32_t const totalMoves = sparseKvOffsets[batchSize];
    int32_t const moveBegin = sparseKvOffsets[batchIdx];
    int32_t const moveEnd = sparseKvOffsets[batchIdx + 1];

    extern __shared__ __align__(16) uint8_t sharedBytes[];
    T* const kShared = reinterpret_cast<T*>(sharedBytes);
    T* const vShared = kShared + blockDim.y * headDim;

    int32_t const movesPerChunk = blockDim.y;
    for (int32_t chunkBegin = moveBegin; chunkBegin < moveEnd; chunkBegin += movesPerChunk)
    {
        int32_t const globalMove = chunkBegin + threadIdx.y;
        bool const active = globalMove < moveEnd;
        int32_t sourceToken = 0;
        int32_t destinationToken = 0;
        if (active)
        {
            int32_t const sourceOffset = kvHeadIdx * totalMoves + globalMove;
            sourceToken = sparseKvIndices[sourceOffset];
            destinationToken = globalMove - moveBegin;
            if (destinationIndices != nullptr)
            {
                int32_t const destinationOffset = destinationPerHead ? kvHeadIdx * totalMoves + globalMove : globalMove;
                destinationToken = destinationIndices[destinationOffset];
            }

            auto const sourceK = reinterpret_cast<T const*>(buffer.getKBlockPtr(batchIdx, sourceToken));
            auto const sourceV = reinterpret_cast<T const*>(buffer.getVBlockPtr(batchIdx, sourceToken));
            int32_t const sourceBase = buffer.getKVLocalIdx(sourceToken, kvHeadIdx, headDim, 0);
            int32_t const sharedBase = threadIdx.y * headDim;
            for (int32_t dim = threadIdx.x; dim < headDim; dim += blockDim.x)
            {
                kShared[sharedBase + dim] = sourceK[sourceBase + dim];
                vShared[sharedBase + dim] = sourceV[sourceBase + dim];
            }
        }
        __syncthreads();

        if (active && sourceToken != destinationToken)
        {
            auto destinationK = reinterpret_cast<T*>(buffer.getKBlockPtr(batchIdx, destinationToken));
            auto destinationV = reinterpret_cast<T*>(buffer.getVBlockPtr(batchIdx, destinationToken));
            int32_t const destinationBase = buffer.getKVLocalIdx(destinationToken, kvHeadIdx, headDim, 0);
            int32_t const sharedBase = threadIdx.y * headDim;
            for (int32_t dim = threadIdx.x; dim < headDim; dim += blockDim.x)
            {
                destinationK[destinationBase + dim] = kShared[sharedBase + dim];
                destinationV[destinationBase + dim] = vShared[sharedBase + dim];
            }
        }
        __syncthreads();
    }
}

template <typename T>
void invokeSparseKvCacheCompactV2Runtime(KvCacheV2PoolBuffer const& buffer, int32_t const* sparseKvIndices,
    int32_t const* sparseKvOffsets, int32_t const* destinationIndices, bool destinationPerHead, int32_t batchSize,
    int32_t numKvHeads, int32_t headDim, cudaStream_t stream)
{
    constexpr int32_t kThreadsPerToken = 32;
    constexpr int32_t kTokensPerChunk = 8;
    dim3 const grid(batchSize, numKvHeads);
    dim3 const block(kThreadsPerToken, kTokensPerChunk);
    size_t const sharedBytes = 2 * static_cast<size_t>(kTokensPerChunk) * headDim * sizeof(T);
    sparseKvCacheCompactV2Runtime<T><<<grid, block, sharedBytes, stream>>>(
        buffer, sparseKvIndices, sparseKvOffsets, destinationIndices, destinationPerHead, batchSize, headDim);
}

template <typename T>
bool supportsOptimizedPrefixCompact(int32_t headDim)
{
    constexpr size_t kPortableDynamicSharedBytes = 48 * 1024;
    constexpr int32_t kTokensPerChunk = 32;
    size_t const sharedBytes = 2 * static_cast<size_t>(kTokensPerChunk) * headDim * sizeof(T);
    if (sharedBytes > kPortableDynamicSharedBytes)
    {
        return false;
    }
    switch (headDim)
    {
    case 16:
    case 32:
    case 64:
    case 128:
    case 256: return true;
    default: return false;
    }
}

template <typename T>
void invokeSparseKvCacheCompactV2(T* pool, int32_t const* pageTable, int32_t maxPagesPerSeq,
    int32_t const* sparseKvIndices, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    bool destinationPerHead, int32_t batchSize, int32_t numKvHeads, int32_t tokensPerBlock, int32_t headDim,
    cudaStream_t stream)
{
    KvCacheV2PoolBuffer buffer{};
    buffer.pool = reinterpret_cast<uint8_t*>(pool);
    buffer.pageTable = pageTable;
    buffer.maxPagesPerSeq = maxPagesPerSeq;
    buffer.tokensPerBlock = tokensPerBlock;
    buffer.bytesPerKvHalf = static_cast<size_t>(numKvHeads) * tokensPerBlock * headDim * sizeof(T);
    buffer.bytesPerPage = 2 * buffer.bytesPerKvHalf;

    if (destinationIndices != nullptr || !supportsOptimizedPrefixCompact<T>(headDim))
    {
        invokeSparseKvCacheCompactV2Runtime<T>(buffer, sparseKvIndices, sparseKvOffsets, destinationIndices,
            destinationPerHead, batchSize, numKvHeads, headDim, stream);
        return;
    }

    QKVPreprocessingParams<T, KvCacheV2PoolBuffer> params{};
    params.kv_cache_buffer = buffer;
    params.sparse_kv_indices = sparseKvIndices;
    params.sparse_kv_offsets = sparseKvOffsets;
    params.batch_size = batchSize;
    params.kv_head_num = numKvHeads;
    params.size_per_head = headDim;
    invokeUpdateSparseKvCacheAfterFmha<T, T, KvCacheV2PoolBuffer>(params, stream);
}

template void invokeSparseKvCacheCompactV2<half>(half*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeSparseKvCacheCompactV2<float>(float*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeSparseKvCacheCompactV2<__nv_bfloat16>(__nv_bfloat16*, int32_t const*, int32_t, int32_t const*,
    int32_t const*, int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
