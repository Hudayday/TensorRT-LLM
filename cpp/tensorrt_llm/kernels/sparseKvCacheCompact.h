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

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! In-place sparse KV compaction for uniform KVCacheManagerV2 layer pools. All layers must share
//! the same batch, KV-head, block, and head-dimension geometry. For every request and head, source
//! and destination indices must be strictly increasing with destination[i] <= source[i]. This
//! monotonic-left contract makes the in-place operation safe without scratch. The pool and page-table
//! pointer arrays reside on the device so CUDA Graph replay launches one kernel for the complete
//! layer group without rebuilding host metadata.
//! Source indices are either shared by all layers (sourceLayerStride == 0) or
//! separated by sourceLayerStride elements. sourceLayerIndices maps each group
//! layer to its row in a larger per-layer source tensor. Destination strides use
//! the same convention and are ignored when destinationIndices is null.
template <typename T>
void invokeSparseKvCacheCompactV2Layers(int64_t const* poolPointers, int64_t const* pageTablePointers,
    int32_t numLayers, int32_t maxPagesPerSeq, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int32_t const* sparseKvOffsets, int32_t const* destinationIndices,
    int64_t destinationLayerStride, int64_t destinationHeadStride, int32_t batchSize, int32_t numKvHeads,
    int32_t tokensPerBlock, int32_t headDim, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
