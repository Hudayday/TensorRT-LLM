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
__global__ void sparseKvCacheCompactV2LayersRuntime(int64_t const* poolPointers, int64_t const* pageTablePointers,
    int32_t maxPagesPerSeq, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize, int32_t numKvHeads,
    int32_t tokensPerBlock, int32_t headDim)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const kvHeadIdx = blockIdx.y;
    int32_t const layerIdx = blockIdx.z;
    int32_t const totalMoves = sparseKvOffsets[batchSize];
    int32_t const moveBegin = sparseKvOffsets[batchIdx];
    int32_t const moveEnd = sparseKvOffsets[batchIdx + 1];

    KvCacheV2PoolBuffer buffer{};
    buffer.pool = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(poolPointers[layerIdx]));
    buffer.pageTable = reinterpret_cast<int32_t const*>(static_cast<uintptr_t>(pageTablePointers[layerIdx]));
    buffer.maxPagesPerSeq = maxPagesPerSeq;
    buffer.tokensPerBlock = tokensPerBlock;
    buffer.bytesPerKvHalf = static_cast<size_t>(numKvHeads) * tokensPerBlock * headDim * sizeof(T);
    buffer.bytesPerPage = 2 * buffer.bytesPerKvHalf;

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
            int32_t const sourceLayerIdx = sourceLayerIndices == nullptr ? layerIdx : sourceLayerIndices[layerIdx];
            int64_t const sourceOffset
                = static_cast<int64_t>(sourceLayerIdx) * sourceLayerStride + kvHeadIdx * totalMoves + globalMove;
            sourceToken = sparseKvIndices[sourceOffset];
            destinationToken = globalMove - moveBegin;
            if (destinationIndices != nullptr)
            {
                int64_t const destinationOffset = static_cast<int64_t>(layerIdx) * destinationLayerStride
                    + static_cast<int64_t>(kvHeadIdx) * destinationHeadStride + globalMove;
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

template <typename T, int32_t HeadDim>
__global__ __launch_bounds__(1024) void sparseKvCacheCompactV2LayersOptimized(int64_t const* poolPointers,
    int64_t const* pageTablePointers, int32_t maxPagesPerSeq, int32_t const* sparseKvIndices,
    int32_t const* sourceLayerIndices, int64_t sourceLayerStride, int32_t const* sparseKvOffsets,
    int32_t const* destinationIndices, int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize,
    int32_t numKvHeads, int32_t tokensPerBlock)
{
    constexpr int32_t kBytesPerVector = 16;
    constexpr int32_t kVectorsPerHead = HeadDim * sizeof(T) / kBytesPerVector;
    static_assert(HeadDim * sizeof(T) % kBytesPerVector == 0, "Head dimension must contain complete vectors");

    int32_t const batchIdx = blockIdx.x;
    int32_t const kvHeadIdx = blockIdx.y;
    int32_t const layerIdx = blockIdx.z;
    int32_t const totalMoves = sparseKvOffsets[batchSize];
    int32_t const moveBegin = sparseKvOffsets[batchIdx];
    int32_t const moveEnd = sparseKvOffsets[batchIdx + 1];
    int32_t const moveCount = moveEnd - moveBegin;

    KvCacheV2PoolBuffer buffer{};
    buffer.pool = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(poolPointers[layerIdx]));
    buffer.pageTable = reinterpret_cast<int32_t const*>(static_cast<uintptr_t>(pageTablePointers[layerIdx]));
    buffer.maxPagesPerSeq = maxPagesPerSeq;
    buffer.tokensPerBlock = tokensPerBlock;
    buffer.bytesPerKvHalf = static_cast<size_t>(numKvHeads) * tokensPerBlock * HeadDim * sizeof(T);
    buffer.bytesPerPage = 2 * buffer.bytesPerKvHalf;

    extern __shared__ uint4 shared[];
    uint4* const kShared = shared;
    uint4* const vShared = kShared + blockDim.y * kVectorsPerHead;

    int32_t const movesPerChunk = blockDim.y;
    for (int32_t chunkOffset = 0; chunkOffset < moveCount; chunkOffset += movesPerChunk)
    {
        int32_t const requestMove = chunkOffset + threadIdx.y;
        bool const active = requestMove < moveCount;
        int32_t sourceToken = 0;
        if (active)
        {
            int32_t const globalMove = moveBegin + requestMove;
            int32_t const sourceLayerIdx = sourceLayerIndices == nullptr ? layerIdx : sourceLayerIndices[layerIdx];
            int64_t const sourceOffset
                = static_cast<int64_t>(sourceLayerIdx) * sourceLayerStride + kvHeadIdx * totalMoves + globalMove;
            sourceToken = sparseKvIndices[sourceOffset];
            auto const sourceK = reinterpret_cast<uint4 const*>(buffer.getKBlockPtr(batchIdx, sourceToken));
            auto const sourceV = reinterpret_cast<uint4 const*>(buffer.getVBlockPtr(batchIdx, sourceToken));
            for (int32_t vector = threadIdx.x; vector < kVectorsPerHead; vector += blockDim.x)
            {
                int32_t const sourceLocal = buffer.getKVLocalIdx(sourceToken, kvHeadIdx, kVectorsPerHead, vector);
                int32_t const sharedLocal = threadIdx.y * kVectorsPerHead + vector;
                kShared[sharedLocal] = sourceK[sourceLocal];
                vShared[sharedLocal] = sourceV[sourceLocal];
            }
        }
        __syncthreads();

        int32_t destinationToken = requestMove;
        if (active && destinationIndices != nullptr)
        {
            int32_t const globalMove = moveBegin + requestMove;
            int64_t const destinationOffset = static_cast<int64_t>(layerIdx) * destinationLayerStride
                + static_cast<int64_t>(kvHeadIdx) * destinationHeadStride + globalMove;
            destinationToken = destinationIndices[destinationOffset];
        }
        if (active && sourceToken != destinationToken)
        {
            auto destinationK = reinterpret_cast<uint4*>(buffer.getKBlockPtr(batchIdx, destinationToken));
            auto destinationV = reinterpret_cast<uint4*>(buffer.getVBlockPtr(batchIdx, destinationToken));
            for (int32_t vector = threadIdx.x; vector < kVectorsPerHead; vector += blockDim.x)
            {
                int32_t const destinationLocal
                    = buffer.getKVLocalIdx(destinationToken, kvHeadIdx, kVectorsPerHead, vector);
                int32_t const sharedLocal = threadIdx.y * kVectorsPerHead + vector;
                destinationK[destinationLocal] = kShared[sharedLocal];
                destinationV[destinationLocal] = vShared[sharedLocal];
            }
        }
        __syncthreads();
    }
}

template <typename T, int32_t HeadDim>
void invokeSparseKvCacheCompactV2LayersOptimized(int64_t const* poolPointers, int64_t const* pageTablePointers,
    int32_t numLayers, int32_t maxPagesPerSeq, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize, int32_t numKvHeads,
    int32_t tokensPerBlock, cudaStream_t stream)
{
    constexpr int32_t kThreadsPerVector = 32;
    constexpr int32_t kTokensPerChunk = 32;
    constexpr int32_t kVectorsPerHead = HeadDim * sizeof(T) / 16;
    dim3 const grid(batchSize, numKvHeads, numLayers);
    dim3 const block(kThreadsPerVector, kTokensPerChunk);
    size_t const sharedBytes = 2 * static_cast<size_t>(kTokensPerChunk) * kVectorsPerHead * sizeof(uint4);
    sparseKvCacheCompactV2LayersOptimized<T, HeadDim><<<grid, block, sharedBytes, stream>>>(poolPointers,
        pageTablePointers, maxPagesPerSeq, sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets,
        destinationIndices, destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock);
}

template <typename T>
void dispatchSparseKvCacheCompactV2LayersOptimized(int64_t const* poolPointers, int64_t const* pageTablePointers,
    int32_t numLayers, int32_t maxPagesPerSeq, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize, int32_t numKvHeads,
    int32_t tokensPerBlock, int32_t headDim, cudaStream_t stream)
{
    switch (headDim)
    {
    case 16:
        invokeSparseKvCacheCompactV2LayersOptimized<T, 16>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, stream);
        break;
    case 32:
        invokeSparseKvCacheCompactV2LayersOptimized<T, 32>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, stream);
        break;
    case 64:
        invokeSparseKvCacheCompactV2LayersOptimized<T, 64>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, stream);
        break;
    case 128:
        invokeSparseKvCacheCompactV2LayersOptimized<T, 128>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, stream);
        break;
    case 256:
        invokeSparseKvCacheCompactV2LayersOptimized<T, 256>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, stream);
        break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported multi-layer sparse KV compact head dimension %d", headDim);
    }
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

template <typename T>
void invokeSparseKvCacheCompactV2Layers(int64_t const* poolPointers, int64_t const* pageTablePointers,
    int32_t numLayers, int32_t maxPagesPerSeq, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize, int32_t numKvHeads,
    int32_t tokensPerBlock, int32_t headDim, cudaStream_t stream)
{
    if (supportsOptimizedPrefixCompact<T>(headDim))
    {
        dispatchSparseKvCacheCompactV2LayersOptimized<T>(poolPointers, pageTablePointers, numLayers, maxPagesPerSeq,
            sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
            destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, headDim, stream);
        return;
    }

    constexpr int32_t kThreadsPerToken = 32;
    constexpr int32_t kTokensPerChunk = 8;
    dim3 const grid(batchSize, numKvHeads, numLayers);
    dim3 const block(kThreadsPerToken, kTokensPerChunk);
    size_t const sharedBytes = 2 * static_cast<size_t>(kTokensPerChunk) * headDim * sizeof(T);
    sparseKvCacheCompactV2LayersRuntime<T><<<grid, block, sharedBytes, stream>>>(poolPointers, pageTablePointers,
        maxPagesPerSeq, sparseKvIndices, sourceLayerIndices, sourceLayerStride, sparseKvOffsets, destinationIndices,
        destinationLayerStride, destinationHeadStride, batchSize, numKvHeads, tokensPerBlock, headDim);
}

template void invokeSparseKvCacheCompactV2<half>(half*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeSparseKvCacheCompactV2<float>(float*, int32_t const*, int32_t, int32_t const*, int32_t const*,
    int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeSparseKvCacheCompactV2Layers<half>(int64_t const*, int64_t const*, int32_t, int32_t, int32_t const*,
    int32_t const*, int64_t, int32_t const*, int32_t const*, int64_t, int64_t, int32_t, int32_t, int32_t, int32_t,
    cudaStream_t);
template void invokeSparseKvCacheCompactV2Layers<float>(int64_t const*, int64_t const*, int32_t, int32_t,
    int32_t const*, int32_t const*, int64_t, int32_t const*, int32_t const*, int64_t, int64_t, int32_t, int32_t,
    int32_t, int32_t, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeSparseKvCacheCompactV2<__nv_bfloat16>(__nv_bfloat16*, int32_t const*, int32_t, int32_t const*,
    int32_t const*, int32_t const*, bool, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeSparseKvCacheCompactV2Layers<__nv_bfloat16>(int64_t const*, int64_t const*, int32_t, int32_t,
    int32_t const*, int32_t const*, int64_t, int32_t const*, int32_t const*, int64_t, int64_t, int32_t, int32_t,
    int32_t, int32_t, cudaStream_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
