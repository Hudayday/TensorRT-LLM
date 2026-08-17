/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace
{

using tensorrt_llm::batch_manager::kv_cache_manager_v2::HostMem;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::MemAddress;
using tensorrt_llm::kernels::Nvfp4BoundaryKernelParams;
using tensorrt_llm::kernels::Nvfp4BoundaryLayerPlan;
using tensorrt_llm::kernels::Nvfp4BoundaryOffloadPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryOnboardPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryPreparedPlan;
using tensorrt_llm::kernels::Nvfp4BoundaryRuntimeType;

constexpr std::size_t kGuardBytes = 64;
constexpr std::uint8_t kCanary = 0xA5;
constexpr std::size_t kDefaultNumPages = 3;
constexpr std::size_t kCrossLaunchNumPages = 257;

struct PageGeometry
{
    std::int32_t numHeads;
    std::int32_t tokensPerPage;
    std::int32_t headDim;
};

constexpr PageGeometry kDefaultGeometry{2, 8, 32};
constexpr PageGeometry kMinimumCompactGeometry{1, 1, 16};
constexpr PageGeometry kPackedBodyAndTailGeometry{1, 3, 16};
constexpr PageGeometry kSmallVectorGeometry{1, 4, 16};
constexpr PageGeometry kLinearScaleTailGeometry{1, 5, 32};
constexpr PageGeometry kTiledLinearScaleTailGeometry{1, 4097, 16};
constexpr PageGeometry kSecondCtaTailGeometry{2, 20, 64};
constexpr PageGeometry kModelLikeGeometry{8, 64, 128};
constexpr PageGeometry kLargeGeometry{64, 64, 128};

enum class RawKind
{
    kFloat16,
    kBfloat16,
    kFp8,
};

enum class InputPattern
{
    kDense,
    kAllZero,
    kSparseOutlier,
    kRoundingMargins,
};

std::size_t roundUp(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

class CudaStream
{
public:
    CudaStream()
    {
        TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    operator cudaStream_t() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{};
};

//! Device allocation with canaries around the Page payload. It catches a
//! descriptor-index or vector-tail bug independently of output comparisons.
class DeviceRegion
{
public:
    explicit DeviceRegion(std::size_t payloadBytes)
        : mPayloadBytes(payloadBytes)
        , mTotalBytes(payloadBytes + 2 * kGuardBytes)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mBase, mTotalBytes));
        TLLM_CUDA_CHECK(cudaMemset(mBase, kCanary, mTotalBytes));
    }

    ~DeviceRegion()
    {
        if (mBase != nullptr)
        {
            cudaFree(mBase);
        }
    }

    DeviceRegion(DeviceRegion const&) = delete;
    DeviceRegion& operator=(DeviceRegion const&) = delete;

    void* data() const
    {
        return static_cast<std::uint8_t*>(mBase) + kGuardBytes;
    }

    void copyFrom(std::vector<std::uint8_t> const& bytes)
    {
        ASSERT_EQ(bytes.size(), mPayloadBytes);
        ASSERT_EQ(cudaMemcpy(data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void copyFrom(std::size_t offset, std::vector<std::uint8_t> const& bytes)
    {
        ASSERT_LE(offset + bytes.size(), mPayloadBytes);
        ASSERT_EQ(
            cudaMemcpy(static_cast<std::uint8_t*>(data()) + offset, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
    }

    std::vector<std::uint8_t> copyToHost() const
    {
        std::vector<std::uint8_t> bytes(mPayloadBytes);
        EXPECT_EQ(cudaMemcpy(bytes.data(), data(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        return bytes;
    }

    std::vector<std::uint8_t> copyToHost(std::size_t offset, std::size_t bytes) const
    {
        EXPECT_LE(offset + bytes, mPayloadBytes);
        std::vector<std::uint8_t> result(bytes);
        EXPECT_EQ(
            cudaMemcpy(result.data(), static_cast<std::uint8_t const*>(data()) + offset, bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        return result;
    }

    void expectCanaries() const
    {
        std::vector<std::uint8_t> bytes(mTotalBytes);
        ASSERT_EQ(cudaMemcpy(bytes.data(), mBase, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::all_of(
            bytes.begin(), bytes.begin() + kGuardBytes, [](std::uint8_t value) { return value == kCanary; }));
        EXPECT_TRUE(
            std::all_of(bytes.end() - kGuardBytes, bytes.end(), [](std::uint8_t value) { return value == kCanary; }));
    }

private:
    void* mBase{};
    std::size_t mPayloadBytes{};
    std::size_t mTotalBytes{};
};

//! A real KVCM V2 Host carrier. HostMem uses
//! CU_MEMHOSTREGISTER_DEVICEMAP, so the tested kernel accesses the same kind of
//! pointer that StorageManager will eventually provide; no cudaMemcpy is used
//! for the boundary payload itself.
class MappedHostRegion
{
public:
    explicit MappedHostRegion(std::size_t payloadBytes)
        : mMemory(roundUp(kGuardBytes + payloadBytes + kGuardBytes, HostMem::kAlignment))
        , mPayloadBytes(payloadBytes)
    {
        TLLM_CHECK_WITH_INFO(kGuardBytes + payloadBytes + kGuardBytes <= mMemory.size(),
            "Mapped Host test allocation is too small for payload and canaries");
        std::memset(reinterpret_cast<void*>(mMemory.address()), kCanary, mMemory.size());
    }

    void* data() const
    {
        return reinterpret_cast<void*>(mMemory.address() + kGuardBytes);
    }

    std::uint8_t* bytes() const
    {
        return static_cast<std::uint8_t*>(data());
    }

    std::vector<std::uint8_t> payload() const
    {
        return {bytes(), bytes() + mPayloadBytes};
    }

    void expectCanaries() const
    {
        auto const* base = reinterpret_cast<std::uint8_t const*>(mMemory.address());
        EXPECT_TRUE(std::all_of(base, base + kGuardBytes, [](std::uint8_t value) { return value == kCanary; }));
        EXPECT_TRUE(std::all_of(base + kGuardBytes + mPayloadBytes, base + mMemory.size(),
            [](std::uint8_t value) { return value == kCanary; }));
    }

private:
    HostMem mMemory;
    std::size_t mPayloadBytes{};
};

struct PageBuffers
{
    PageBuffers(std::size_t rawBytes, std::size_t packedBytes, std::size_t scaleBytes)
        : rawInputK(rawBytes)
        , rawInputV(rawBytes)
        , rawOutputK(rawBytes)
        , rawOutputV(rawBytes)
        , compactPage(2 * (packedBytes + scaleBytes))
    {
    }

    std::vector<std::uint8_t> compactRegion(std::size_t offset, std::size_t bytes) const
    {
        auto const payload = compactPage.payload();
        return {payload.begin() + static_cast<std::ptrdiff_t>(offset),
            payload.begin() + static_cast<std::ptrdiff_t>(offset + bytes)};
    }

    DeviceRegion rawInputK;
    DeviceRegion rawInputV;
    DeviceRegion rawOutputK;
    DeviceRegion rawOutputV;
    MappedHostRegion compactPage;
    std::array<std::vector<std::uint8_t>, 2> rawHost;
};

std::size_t numElements(PageGeometry const& geometry)
{
    return static_cast<std::size_t>(geometry.numHeads) * geometry.tokensPerPage * geometry.headDim;
}

std::size_t rawBytes(RawKind kind, PageGeometry const& geometry)
{
    return numElements(geometry) * (kind == RawKind::kFp8 ? 1 : 2);
}

std::size_t packedBytes(PageGeometry const& geometry)
{
    return numElements(geometry) / 2;
}

std::size_t scaleBytes(PageGeometry const& geometry)
{
    return numElements(geometry) / 16;
}

Nvfp4BoundaryKernelParams makeParams(PageGeometry const& geometry = kDefaultGeometry)
{
    Nvfp4BoundaryKernelParams params{};
    params.numKvHeads = geometry.numHeads;
    params.tokensPerPage = geometry.tokensPerPage;
    params.headDim = geometry.headDim;
    params.nvfp4ScaleOrigQuant[0] = 1.0F;
    params.nvfp4ScaleOrigQuant[1] = 2.0F;
    params.nvfp4ScaleQuantOrig[0] = 1.0F;
    params.nvfp4ScaleQuantOrig[1] = 0.5F;
    params.fp8ScaleOrigQuant[0] = 2.0F;
    params.fp8ScaleOrigQuant[1] = 4.0F;
    params.fp8ScaleQuantOrig[0] = 0.5F;
    params.fp8ScaleQuantOrig[1] = 0.25F;
    return params;
}

template <typename T>
void storeScalar(std::vector<std::uint8_t>& bytes, std::size_t index, T value)
{
    std::memcpy(bytes.data() + index * sizeof(T), &value, sizeof(T));
}

template <typename T>
T loadScalar(std::vector<std::uint8_t> const& bytes, std::size_t index)
{
    T value;
    std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
    return value;
}

void storeRawValue(std::vector<std::uint8_t>& bytes, RawKind kind, std::size_t index, float value,
    Nvfp4BoundaryKernelParams const& params, std::uint32_t role)
{
    switch (kind)
    {
    case RawKind::kFloat16: storeScalar(bytes, index, __float2half(value)); break;
    case RawKind::kBfloat16: storeScalar(bytes, index, __float2bfloat16(value)); break;
    case RawKind::kFp8: storeScalar(bytes, index, __nv_fp8_e4m3(value * params.fp8ScaleOrigQuant[role])); break;
    }
}

float loadRawValue(std::vector<std::uint8_t> const& bytes, RawKind kind, std::size_t index,
    Nvfp4BoundaryKernelParams const& params, std::uint32_t role)
{
    switch (kind)
    {
    case RawKind::kFloat16: return __half2float(loadScalar<half>(bytes, index));
    case RawKind::kBfloat16: return __bfloat162float(loadScalar<__nv_bfloat16>(bytes, index));
    case RawKind::kFp8:
        return static_cast<float>(loadScalar<__nv_fp8_e4m3>(bytes, index)) * params.fp8ScaleQuantOrig[role];
    }
    return 0.0F;
}

std::uint32_t linearScaleOffset(std::uint32_t row, std::uint32_t scaleInRow, PageGeometry const& geometry)
{
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    return row * scalesPerRow + scaleInRow;
}

float e2m1Value(std::uint8_t nibble)
{
    constexpr std::array<float, 8> levels{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    float const value = levels[nibble & 0x7U];
    return (nibble & 0x8U) != 0 ? -value : value;
}

//! Independent nearest-level oracle. Test inputs avoid exact midpoints, so the
//! production instruction's tie rule is intentionally not duplicated here.
std::uint8_t quantizeE2m1(float value)
{
    constexpr std::array<float, 8> levels{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    bool const negative = std::signbit(value);
    float const magnitude = std::abs(value);
    std::uint8_t best = 0;
    float bestDistance = std::abs(magnitude - levels[0]);
    for (std::uint8_t index = 1; index < levels.size(); ++index)
    {
        float const distance = std::abs(magnitude - levels[index]);
        if (distance < bestDistance)
        {
            best = index;
            bestDistance = distance;
        }
    }
    return static_cast<std::uint8_t>(best | (negative ? 0x8U : 0U));
}

//! Dense and sparse values use exact E2M1 levels and exactly representable
//! E4M3 scales. The rounding fixture keeps the same exact amax/scale but places
//! other values safely away from E2M1 ties. This preserves deterministic byte
//! comparisons while varying Page, K/V role, row, and scale group.
std::vector<std::uint8_t> makeRawPage(RawKind kind, std::size_t page, std::uint32_t role,
    Nvfp4BoundaryKernelParams const& params, PageGeometry const& geometry, InputPattern inputPattern)
{
    constexpr std::array<float, 16> densePattern{
        0.0F, 0.5F, -1.0F, 1.5F, -2.0F, 3.0F, -4.0F, 6.0F, -0.5F, 1.0F, -1.5F, 2.0F, -3.0F, 4.0F, -6.0F, 0.5F};
    constexpr std::array<float, 16> firstLaneOutlierPattern{
        6.0F, -0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    constexpr std::array<float, 16> secondLaneOutlierPattern{
        0.0F, -0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -6.0F};
    constexpr std::array<float, 16> roundingMarginsPattern{
        6.0F, 0.20F, 0.30F, 0.65F, 0.85F, 1.10F, 1.40F, 1.60F, 1.90F, 2.30F, 2.70F, 3.20F, 3.80F, 4.50F, 5.20F, -0.30F};
    constexpr std::array<float, 4> blockScales{0.25F, 0.5F, 1.0F, 2.0F};

    std::vector<std::uint8_t> bytes(rawBytes(kind, geometry));
    for (std::size_t index = 0; index < numElements(geometry); ++index)
    {
        std::size_t const scaleGroup = index / 16;
        float const blockScale = blockScales[(scaleGroup + page + role) % blockScales.size()];
        float normalizedValue = 0.0F;
        if (inputPattern == InputPattern::kDense)
        {
            normalizedValue = densePattern[index % densePattern.size()];
        }
        else if (inputPattern == InputPattern::kSparseOutlier)
        {
            auto const& pattern = (scaleGroup & 1U) == 0 ? firstLaneOutlierPattern : secondLaneOutlierPattern;
            normalizedValue = pattern[index % pattern.size()];
        }
        else if (inputPattern == InputPattern::kRoundingMargins)
        {
            normalizedValue = roundingMarginsPattern[index % roundingMarginsPattern.size()];
        }
        if (((page / 32) & 1U) != 0)
        {
            normalizedValue = -normalizedValue;
        }
        float const value = normalizedValue * blockScale / params.nvfp4ScaleOrigQuant[role];
        storeRawValue(bytes, kind, index, value, params, role);
    }
    return bytes;
}

struct ReferenceNvfp4
{
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
};

ReferenceNvfp4 compressReference(std::vector<std::uint8_t> const& raw, RawKind kind, std::uint32_t role,
    Nvfp4BoundaryKernelParams const& params, PageGeometry const& geometry)
{
    ReferenceNvfp4 result{{}, {}};
    result.packed.resize(packedBytes(geometry));
    result.scales.resize(scaleBytes(geometry));
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    std::uint32_t const rows = static_cast<std::uint32_t>(geometry.numHeads * geometry.tokensPerPage);

    for (std::uint32_t row = 0; row < rows; ++row)
    {
        for (std::uint32_t scaleInRow = 0; scaleInRow < scalesPerRow; ++scaleInRow)
        {
            std::size_t const blockStart = static_cast<std::size_t>(row) * geometry.headDim + scaleInRow * 16;
            float amax = 0.0F;
            for (std::uint32_t i = 0; i < 16; ++i)
            {
                amax = std::max(amax, std::abs(loadRawValue(raw, kind, blockStart + i, params, role)));
            }

            __nv_fp8_e4m3 blockScale(params.nvfp4ScaleOrigQuant[role] * amax / 6.0F);
            result.scales[linearScaleOffset(row, scaleInRow, geometry)] = blockScale.__x;
            float const blockScaleFloat = static_cast<float>(blockScale);
            float const outputScale
                = blockScaleFloat == 0.0F ? 0.0F : params.nvfp4ScaleOrigQuant[role] / blockScaleFloat;
            for (std::uint32_t i = 0; i < 16; i += 2)
            {
                std::uint8_t const lo
                    = quantizeE2m1(loadRawValue(raw, kind, blockStart + i, params, role) * outputScale);
                std::uint8_t const hi
                    = quantizeE2m1(loadRawValue(raw, kind, blockStart + i + 1, params, role) * outputScale);
                result.packed[(blockStart + i) / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
            }
        }
    }
    return result;
}

std::vector<std::uint8_t> decompressReference(ReferenceNvfp4 const& compressed, RawKind kind, std::uint32_t role,
    Nvfp4BoundaryKernelParams const& params, PageGeometry const& geometry)
{
    std::vector<std::uint8_t> raw(rawBytes(kind, geometry));
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    std::uint32_t const rows = static_cast<std::uint32_t>(geometry.numHeads * geometry.tokensPerPage);
    for (std::uint32_t row = 0; row < rows; ++row)
    {
        for (std::uint32_t scaleInRow = 0; scaleInRow < scalesPerRow; ++scaleInRow)
        {
            __nv_fp8_e4m3 blockScale;
            blockScale.__x = compressed.scales[linearScaleOffset(row, scaleInRow, geometry)];
            float const dequantScale = static_cast<float>(blockScale) * params.nvfp4ScaleQuantOrig[role];
            std::size_t const blockStart = static_cast<std::size_t>(row) * geometry.headDim + scaleInRow * 16;
            for (std::uint32_t i = 0; i < 16; ++i)
            {
                std::uint8_t const byte = compressed.packed[(blockStart + i) / 2];
                std::uint8_t const nibble = (i & 1U) == 0 ? byte & 0xFU : byte >> 4;
                storeRawValue(raw, kind, blockStart + i, e2m1Value(nibble) * dequantScale, params, role);
            }
        }
    }
    return raw;
}

void runBoundaryRoundTrip(RawKind kind, PageGeometry const& geometry = kDefaultGeometry,
    std::size_t numPages = kDefaultNumPages, InputPattern inputPattern = InputPattern::kDense,
    bool synchronizeBetweenDirections = true, bool repeatRoundTrip = false)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }
    Nvfp4BoundaryKernelParams const params = makeParams(geometry);
    CudaStream stream;
    std::size_t const rawSlotBytes = rawBytes(kind, geometry);
    // Nvfp4ColdPageCodec preserves the exact compact payload but rounds each
    // cold Slot stride to the uint4 alignment required by the mapped-Host
    // vector path. Keep the standalone kernel fixture identical to that
    // production layout, including small records with alignment padding.
    std::size_t const compactSlotBytes = roundUp(2U * (packedBytes(geometry) + scaleBytes(geometry)), alignof(uint4));
    // Use every other Slot to prove that one launch handles non-contiguous KVCM
    // Page indices. The Layer plan owns bases/strides; Page tasks carry only the
    // complete Base Page index pair.
    std::size_t const slotCapacity = 2U * numPages;
    DeviceRegion rawInputK(slotCapacity * rawSlotBytes);
    DeviceRegion rawInputV(slotCapacity * rawSlotBytes);
    DeviceRegion rawOutputK(slotCapacity * rawSlotBytes);
    DeviceRegion rawOutputV(slotCapacity * rawSlotBytes);
    MappedHostRegion compactPages(slotCapacity * compactSlotBytes);
    std::vector<std::array<std::vector<std::uint8_t>, 2>> rawHost(numPages);
    std::vector<Nvfp4BoundaryOffloadPageTask> offloadTasks;
    offloadTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slot = 2U * page;
        rawHost[page][0] = makeRawPage(kind, page, 0, params, geometry, inputPattern);
        rawHost[page][1] = makeRawPage(kind, page, 1, params, geometry, inputPattern);
        rawInputK.copyFrom(slot * rawSlotBytes, rawHost[page][0]);
        rawInputV.copyFrom(slot * rawSlotBytes, rawHost[page][1]);
        offloadTasks.push_back({static_cast<std::int32_t>(slot), static_cast<std::int32_t>(slot)});
    }

    Nvfp4BoundaryLayerPlan const layer{reinterpret_cast<std::uintptr_t>(rawInputK.data()),
        reinterpret_cast<std::uintptr_t>(rawInputV.data()), rawSlotBytes, rawSlotBytes, 0U, params};
    std::vector<Nvfp4BoundaryLayerPlan> const layers{layer};

    auto runtimeType = Nvfp4BoundaryRuntimeType::kFp8E4m3;
    if (kind == RawKind::kFloat16)
    {
        runtimeType = Nvfp4BoundaryRuntimeType::kFloat16;
    }
    else if (kind == RawKind::kBfloat16)
    {
        runtimeType = Nvfp4BoundaryRuntimeType::kBfloat16;
    }
    auto const inputPlan = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(layers, compactSlotBytes, runtimeType);
    std::vector<std::array<ReferenceNvfp4, 2>> references(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        references[page][0] = compressReference(rawHost[page][0], kind, 0, params, geometry);
        references[page][1] = compressReference(rawHost[page][1], kind, 1, params, geometry);
    }

    auto const verifyCompressedPages = [&]
    {
        for (std::size_t page = 0; page < numPages; ++page)
        {
            std::size_t const packed = packedBytes(geometry);
            std::size_t const scale = scaleBytes(geometry);
            std::size_t const base = 2U * page * compactSlotBytes;
            auto const region = [&](std::size_t offset, std::size_t bytes)
            {
                auto const payload = compactPages.payload();
                return std::vector<std::uint8_t>(payload.begin() + static_cast<std::ptrdiff_t>(base + offset),
                    payload.begin() + static_cast<std::ptrdiff_t>(base + offset + bytes));
            };
            EXPECT_EQ(region(0, packed), references[page][0].packed);
            EXPECT_EQ(region(packed, packed), references[page][1].packed);
            EXPECT_EQ(region(2 * packed, scale), references[page][0].scales);
            EXPECT_EQ(region(2 * packed + scale, scale), references[page][1].scales);
            auto const padding = region(2U * (packed + scale), compactSlotBytes - 2U * (packed + scale));
            EXPECT_TRUE(
                std::all_of(padding.begin(), padding.end(), [](std::uint8_t value) { return value == kCanary; }));
        }
    };

    tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(offloadTasks, inputPlan, compactPages.data(), stream);

    if (synchronizeBetweenDirections)
    {
        // Model StorageManager's normal event-gated publish: the Host Slot is
        // inspected only after the offload completion event has fired.
        cudaEvent_t offloadComplete{};
        ASSERT_EQ(cudaEventCreateWithFlags(&offloadComplete, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(offloadComplete, stream), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(offloadComplete), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(offloadComplete), cudaSuccess);
        verifyCompressedPages();
    }

    std::vector<Nvfp4BoundaryOnboardPageTask> onboardTasks;
    onboardTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slot = 2U * page;
        onboardTasks.push_back({static_cast<std::int32_t>(slot), static_cast<std::int32_t>(slot)});
    }
    Nvfp4BoundaryLayerPlan const outputLayer{reinterpret_cast<std::uintptr_t>(rawOutputK.data()),
        reinterpret_cast<std::uintptr_t>(rawOutputV.data()), rawSlotBytes, rawSlotBytes, 0U, params};
    auto const outputPlan
        = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan({outputLayer}, compactSlotBytes, runtimeType);
    tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(onboardTasks, outputPlan, compactPages.data(), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    if (!synchronizeBetweenDirections)
    {
        // PDL stress path: offload and onboard were enqueued back-to-back on
        // one stream with no Host wait, event, or payload read between them.
        // The final synchronization is the first point where Host NVFP4 bytes
        // are observed.
        verifyCompressedPages();
    }

    if (repeatRoundTrip)
    {
        // Reuse the same logical Pages and compact Slots for a second lossy
        // lifecycle. This catches stale descriptors and proves that repeated
        // offload/onboard follows Q(D(Q(D(Q(x))))) rather than accidentally
        // retaining the first raw source. The second encode reads the first
        // decode's output; the second decode deliberately writes back into the
        // original raw Slots.
        for (std::size_t page = 0; page < numPages; ++page)
        {
            for (std::uint32_t role = 0; role < 2; ++role)
            {
                auto const restored = decompressReference(references[page][role], kind, role, params, geometry);
                references[page][role] = compressReference(restored, kind, role, params, geometry);
            }
        }
        tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
            offloadTasks, outputPlan, compactPages.data(), stream);
        tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
            onboardTasks, inputPlan, compactPages.data(), stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        verifyCompressedPages();
    }

    for (std::size_t page = 0; page < numPages; ++page)
    {
        std::size_t const slotOffset = 2U * page * rawSlotBytes;
        auto const& finalK = repeatRoundTrip ? rawInputK : rawOutputK;
        auto const& finalV = repeatRoundTrip ? rawInputV : rawOutputV;
        EXPECT_EQ(finalK.copyToHost(slotOffset, rawSlotBytes),
            decompressReference(references[page][0], kind, 0, params, geometry));
        EXPECT_EQ(finalV.copyToHost(slotOffset, rawSlotBytes),
            decompressReference(references[page][1], kind, 1, params, geometry));
    }
    rawInputK.expectCanaries();
    rawInputV.expectCanaries();
    rawOutputK.expectCanaries();
    rawOutputV.expectCanaries();
    compactPages.expectCanaries();
}

class Nvfp4Boundary16BitTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4Boundary16BitTest, BatchesDisjointPagesAndMatchesLinearCompactLayout)
{
    runBoundaryRoundTrip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RuntimeType, Nvfp4Boundary16BitTest, testing::Values(RawKind::kFloat16, RawKind::kBfloat16));

TEST(Nvfp4BoundaryFp8Test, BatchesDisjointPagesWithIndependentSourceAndTargetScales)
{
    runBoundaryRoundTrip(RawKind::kFp8);
}

TEST(Nvfp4BoundaryWholePageTest, DifferentLayerScalesRemainInOneCompletePageBatch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }

    constexpr std::size_t numLayers = 2;
    RawKind constexpr kind = RawKind::kBfloat16;
    std::size_t const rawSlotBytes = rawBytes(kind, kDefaultGeometry);
    std::size_t const layerRecordBytes = 2U * (packedBytes(kDefaultGeometry) + scaleBytes(kDefaultGeometry));
    std::size_t const coldPageBytes = numLayers * layerRecordBytes;

    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawInputK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawInputV;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawOutputK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawOutputV;
    std::array<Nvfp4BoundaryKernelParams, numLayers> params{makeParams(), makeParams()};
    // Deliberately give the second layer a different calibrated K/V convention.
    // A correct whole-Page launch selects these values through blockIdx.y; a
    // per-layer host loop is neither required nor permitted by the kernel ABI.
    params[1].nvfp4ScaleOrigQuant[0] = 0.5F;
    params[1].nvfp4ScaleQuantOrig[0] = 2.0F;
    params[1].nvfp4ScaleOrigQuant[1] = 4.0F;
    params[1].nvfp4ScaleQuantOrig[1] = 0.25F;

    std::array<std::array<std::vector<std::uint8_t>, 2>, numLayers> rawHost;
    std::array<std::array<ReferenceNvfp4, 2>, numLayers> references;
    std::vector<Nvfp4BoundaryLayerPlan> inputPlans;
    std::vector<Nvfp4BoundaryLayerPlan> outputPlans;
    inputPlans.reserve(numLayers);
    outputPlans.reserve(numLayers);
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        rawInputK[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawInputV[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawOutputK[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        rawOutputV[layer] = std::make_unique<DeviceRegion>(rawSlotBytes);
        for (std::uint32_t role = 0; role < 2; ++role)
        {
            rawHost[layer][role]
                = makeRawPage(kind, layer, role, params[layer], kDefaultGeometry, InputPattern::kDense);
            references[layer][role]
                = compressReference(rawHost[layer][role], kind, role, params[layer], kDefaultGeometry);
        }
        rawInputK[layer]->copyFrom(rawHost[layer][0]);
        rawInputV[layer]->copyFrom(rawHost[layer][1]);
        inputPlans.push_back({reinterpret_cast<std::uintptr_t>(rawInputK[layer]->data()),
            reinterpret_cast<std::uintptr_t>(rawInputV[layer]->data()), rawSlotBytes, rawSlotBytes,
            layer * layerRecordBytes, params[layer]});
        outputPlans.push_back({reinterpret_cast<std::uintptr_t>(rawOutputK[layer]->data()),
            reinterpret_cast<std::uintptr_t>(rawOutputV[layer]->data()), rawSlotBytes, rawSlotBytes,
            layer * layerRecordBytes, params[layer]});
    }

    MappedHostRegion compactPage(coldPageBytes);
    CudaStream stream;
    auto const inputPlan = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        inputPlans, coldPageBytes, Nvfp4BoundaryRuntimeType::kBfloat16);
    auto const outputPlan = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        outputPlans, coldPageBytes, Nvfp4BoundaryRuntimeType::kBfloat16);
    tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress({{0, 0}}, inputPlan, compactPage.data(), stream);
    tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress({{0, 0}}, outputPlan, compactPage.data(), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto const compact = compactPage.payload();
    std::size_t const packed = packedBytes(kDefaultGeometry);
    std::size_t const scale = scaleBytes(kDefaultGeometry);
    auto const compactRegion = [&](std::size_t offset, std::size_t bytes)
    {
        return std::vector<std::uint8_t>(compact.begin() + static_cast<std::ptrdiff_t>(offset),
            compact.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    };
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        std::size_t const base = layer * layerRecordBytes;
        EXPECT_EQ(compactRegion(base, packed), references[layer][0].packed);
        EXPECT_EQ(compactRegion(base + packed, packed), references[layer][1].packed);
        EXPECT_EQ(compactRegion(base + 2U * packed, scale), references[layer][0].scales);
        EXPECT_EQ(compactRegion(base + 2U * packed + scale, scale), references[layer][1].scales);
        EXPECT_EQ(rawOutputK[layer]->copyToHost(),
            decompressReference(references[layer][0], kind, 0, params[layer], kDefaultGeometry));
        EXPECT_EQ(rawOutputV[layer]->copyToHost(),
            decompressReference(references[layer][1], kind, 1, params[layer], kDefaultGeometry));
    }
    compactPage.expectCanaries();
}

void expectWholePageLaunchTopology(std::size_t numPages, std::vector<std::uint32_t> expectedGridZ)
{
    constexpr std::size_t numLayers = 2;
    RawKind constexpr kind = RawKind::kBfloat16;
    std::size_t const rawSlotBytes = rawBytes(kind, kSmallVectorGeometry);
    std::size_t const recordBytes = 2U * (packedBytes(kSmallVectorGeometry) + scaleBytes(kSmallVectorGeometry));
    std::size_t const recordStride = roundUp(recordBytes, alignof(uint4));
    std::size_t const coldPageBytes = numLayers * recordStride;

    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawK;
    std::array<std::unique_ptr<DeviceRegion>, numLayers> rawV;
    std::vector<Nvfp4BoundaryLayerPlan> layers;
    layers.reserve(numLayers);
    for (std::size_t layer = 0; layer < numLayers; ++layer)
    {
        rawK[layer] = std::make_unique<DeviceRegion>(numPages * rawSlotBytes);
        rawV[layer] = std::make_unique<DeviceRegion>(numPages * rawSlotBytes);
        auto params = makeParams(kSmallVectorGeometry);
        params.nvfp4ScaleOrigQuant[0] *= static_cast<float>(layer + 1U);
        params.nvfp4ScaleQuantOrig[0] /= static_cast<float>(layer + 1U);
        layers.push_back({reinterpret_cast<std::uintptr_t>(rawK[layer]->data()),
            reinterpret_cast<std::uintptr_t>(rawV[layer]->data()), rawSlotBytes, rawSlotBytes, layer * recordStride,
            params});
    }

    MappedHostRegion coldPages(numPages * coldPageBytes);
    std::vector<Nvfp4BoundaryOffloadPageTask> offloadPages;
    std::vector<Nvfp4BoundaryOnboardPageTask> onboardPages;
    offloadPages.reserve(numPages);
    onboardPages.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        auto const pageIndex = static_cast<std::int32_t>(page);
        offloadPages.push_back({pageIndex, pageIndex});
        onboardPages.push_back({pageIndex, pageIndex});
    }

    CudaStream stream;
    auto const plan
        = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(layers, coldPageBytes, Nvfp4BoundaryRuntimeType::kBfloat16);
    auto const expectWholePageKernels = [&](auto const& enqueue)
    {
        cudaGraph_t graph{};
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        enqueue();
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);

        std::size_t numNodes = 0;
        ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &numNodes), cudaSuccess);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &numNodes), cudaSuccess);
        std::size_t kernelNodes = 0;
        std::vector<std::uint32_t> actualGridZ;
        for (auto const node : nodes)
        {
            cudaGraphNodeType nodeType{};
            ASSERT_EQ(cudaGraphNodeGetType(node, &nodeType), cudaSuccess);
            if (nodeType != cudaGraphNodeTypeKernel)
            {
                continue;
            }
            ++kernelNodes;
            cudaKernelNodeParams nodeParams{};
            ASSERT_EQ(cudaGraphKernelNodeGetParams(node, &nodeParams), cudaSuccess);
            EXPECT_EQ(nodeParams.gridDim.y, 2U * numLayers);
            actualGridZ.push_back(nodeParams.gridDim.z);
        }
        std::sort(actualGridZ.begin(), actualGridZ.end());
        std::sort(expectedGridZ.begin(), expectedGridZ.end());
        EXPECT_EQ(kernelNodes, expectedGridZ.size());
        EXPECT_EQ(actualGridZ, expectedGridZ);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    };

    expectWholePageKernels([&]
        { tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(offloadPages, plan, coldPages.data(), stream); });
    expectWholePageKernels([&]
        { tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(onboardPages, plan, coldPages.data(), stream); });
}

TEST(Nvfp4BoundaryWholePageTest, TwoHundredFiftySixPagesAndAllLayersUseOneKernelPerDirection)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }
    expectWholePageLaunchTopology(256, {256});
}

TEST(Nvfp4BoundaryWholePageTest, TwoHundredFiftySevenPagesUseExactlyTwoWholePageKernelsPerDirection)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }
    expectWholePageLaunchTopology(257, {1, 256});
}

TEST(Nvfp4BoundaryGeometryTest, SupportsSmallVectorGeometryAndWarpTail)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kSmallVectorGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsMinimumTightCompactGeometry)
{
    EXPECT_EQ(packedBytes(kMinimumCompactGeometry), 8U);
    EXPECT_EQ(scaleBytes(kMinimumCompactGeometry), 1U);
    EXPECT_EQ(2U * (packedBytes(kMinimumCompactGeometry) + scaleBytes(kMinimumCompactGeometry)), 18U);
    EXPECT_EQ(roundUp(18U, alignof(uint4)), 32U);
    runBoundaryRoundTrip(RawKind::kFloat16, kMinimumCompactGeometry, 1);
    runBoundaryRoundTrip(RawKind::kBfloat16, kMinimumCompactGeometry, 1);
    runBoundaryRoundTrip(RawKind::kFp8, kMinimumCompactGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsPackedVectorBodyWithLeadingAndTrailingScaleGroups)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kPackedBodyAndTailGeometry, 1);
    runBoundaryRoundTrip(RawKind::kFp8, kPackedBodyAndTailGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsOddTokenCountAndNonVectorScaleTail)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kLinearScaleTailGeometry, 1);
    runBoundaryRoundTrip(RawKind::kBfloat16, kLinearScaleTailGeometry, 1);
    runBoundaryRoundTrip(RawKind::kFp8, kLinearScaleTailGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsTiledPackedAndScaleTails)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kTiledLinearScaleTailGeometry, 1);
    runBoundaryRoundTrip(RawKind::kFp8, kTiledLinearScaleTailGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsSecondCtaWithPartialThreadTail)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kSecondCtaTailGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsModelLikeBfloat16Page)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kModelLikeGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsModelLikeFp8Page)
{
    runBoundaryRoundTrip(RawKind::kFp8, kModelLikeGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, BoundsAutoStagingForLargeBfloat16Page)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kLargeGeometry, 1);
}

TEST(Nvfp4BoundaryGeometryTest, BoundsAutoStagingForLargeFp8Page)
{
    runBoundaryRoundTrip(RawKind::kFp8, kLargeGeometry, 1);
}

class Nvfp4BoundaryInputPatternTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4BoundaryInputPatternTest, HandlesAllZeroScaleGroups)
{
    runBoundaryRoundTrip(GetParam(), kDefaultGeometry, 1, InputPattern::kAllZero);
}

TEST_P(Nvfp4BoundaryInputPatternTest, ReducesAmaxAcrossBothWarpLanes)
{
    runBoundaryRoundTrip(GetParam(), kDefaultGeometry, 1, InputPattern::kSparseOutlier);
}

TEST_P(Nvfp4BoundaryInputPatternTest, ReusesTheSamePagesAcrossTwoLossyColdRounds)
{
    runBoundaryRoundTrip(GetParam(), kDefaultGeometry, 3, InputPattern::kDense, true, true);
}

TEST_P(Nvfp4BoundaryInputPatternTest, ReusesModelLikeTiledPagesAcrossTwoLossyColdRounds)
{
    runBoundaryRoundTrip(GetParam(), kModelLikeGeometry, 2, InputPattern::kDense, true, true);
}

INSTANTIATE_TEST_SUITE_P(AllRuntimeTypes, Nvfp4BoundaryInputPatternTest,
    testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

TEST(Nvfp4BoundaryRoundingTest, QuantizesValuesSafelyAwayFromE2m1Ties)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kDefaultGeometry, 1, InputPattern::kRoundingMargins);
}

TEST(Nvfp4BoundaryBatchingTest, AcceptsMinimumThirtyTwoPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kSmallVectorGeometry, 32);
}

TEST(Nvfp4BoundaryBatchingTest, SelectsSixtyFourPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kSmallVectorGeometry, 33);
}

TEST(Nvfp4BoundaryBatchingTest, Bfloat16SelectsOneHundredTwentyEightPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kSmallVectorGeometry, 65);
}

TEST(Nvfp4BoundaryBatchingTest, Fp8SelectsTwoHundredFiftySixPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFp8, kSmallVectorGeometry, 129);
}

TEST(Nvfp4BoundaryBatchingTest, Bfloat16CrossesTheTwoHundredFiftySixPageChunkBoundary)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kSmallVectorGeometry, kCrossLaunchNumPages);
}

TEST(Nvfp4BoundaryBatchingTest, Fp8CrossesTheTwoHundredFiftySixPageChunkBoundary)
{
    runBoundaryRoundTrip(RawKind::kFp8, kSmallVectorGeometry, kCrossLaunchNumPages);
}

TEST(Nvfp4BoundaryPdlTest, ChainsBfloat16OffloadAndOnboardWithoutIntermediateHostSync)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kSmallVectorGeometry, 65, InputPattern::kDense, false);
}

TEST(Nvfp4BoundaryPdlTest, ChainsFp8OffloadAndOnboardWithoutIntermediateHostSync)
{
    runBoundaryRoundTrip(RawKind::kFp8, kSmallVectorGeometry, 65, InputPattern::kDense, false);
}

TEST(Nvfp4BoundaryValidationTest, EmptyBatchIsAnAsyncNoOp)
{
    tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress({}, Nvfp4BoundaryPreparedPlan{}, nullptr, nullptr);
    tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress({}, Nvfp4BoundaryPreparedPlan{}, nullptr, nullptr);
}

TEST(Nvfp4BoundaryValidationTest, RejectsInvalidGeometryAndScalesBeforeLaunch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }

    PageBuffers buffers(
        rawBytes(RawKind::kFloat16, kDefaultGeometry), packedBytes(kDefaultGeometry), scaleBytes(kDefaultGeometry));
    std::size_t const coldPageBytes = 2U * (packedBytes(kDefaultGeometry) + scaleBytes(kDefaultGeometry));
    auto const prepare16Bit = [&](Nvfp4BoundaryKernelParams const& params)
    {
        Nvfp4BoundaryLayerPlan const layer{reinterpret_cast<std::uintptr_t>(buffers.rawInputK.data()),
            reinterpret_cast<std::uintptr_t>(buffers.rawInputV.data()), rawBytes(RawKind::kFloat16, kDefaultGeometry),
            rawBytes(RawKind::kFloat16, kDefaultGeometry), 0U, params};
        static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
            {layer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFloat16));
    };
    auto const prepareFp8 = [&](Nvfp4BoundaryKernelParams const& params)
    {
        Nvfp4BoundaryLayerPlan const layer{reinterpret_cast<std::uintptr_t>(buffers.rawInputK.data()),
            reinterpret_cast<std::uintptr_t>(buffers.rawInputV.data()), rawBytes(RawKind::kFloat16, kDefaultGeometry),
            rawBytes(RawKind::kFloat16, kDefaultGeometry), 0U, params};
        static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
            {layer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFp8E4m3));
    };

    Nvfp4BoundaryKernelParams invalid = makeParams();
    invalid.numKvHeads = 0;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.tokensPerPage = 0;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.tokensPerPage = 6;
    EXPECT_NO_THROW(prepare16Bit(invalid));
    invalid = makeParams(PageGeometry{1, 1, 16});
    EXPECT_NO_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 0;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 24;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 262144;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.numKvHeads = std::numeric_limits<std::int32_t>::max();
    EXPECT_ANY_THROW(prepare16Bit(invalid));

    invalid = makeParams();
    invalid.nvfp4ScaleOrigQuant[0] = 0.0F;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleQuantOrig[1] = -1.0F;
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleOrigQuant[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleQuantOrig[0] = std::numeric_limits<float>::infinity();
    EXPECT_ANY_THROW(prepare16Bit(invalid));
    invalid = makeParams();
    invalid.fp8ScaleOrigQuant[0] = 0.0F;
    EXPECT_ANY_THROW(prepareFp8(invalid));
    invalid = makeParams();
    invalid.fp8ScaleQuantOrig[1] = std::numeric_limits<float>::infinity();
    EXPECT_ANY_THROW(prepareFp8(invalid));
}

TEST(Nvfp4BoundaryValidationTest, RejectsNullMisalignedAndUnsupportedDescriptors)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }

    Nvfp4BoundaryKernelParams const params = makeParams();
    PageBuffers buffers(
        rawBytes(RawKind::kFloat16, kDefaultGeometry), packedBytes(kDefaultGeometry), scaleBytes(kDefaultGeometry));
    Nvfp4BoundaryOffloadPageTask const validOffload{0, 0};
    Nvfp4BoundaryOnboardPageTask const validOnboard{0, 0};
    std::size_t const coldPageBytes = 2U * (packedBytes(kDefaultGeometry) + scaleBytes(kDefaultGeometry));
    Nvfp4BoundaryLayerPlan const validLayer{reinterpret_cast<std::uintptr_t>(buffers.rawInputK.data()),
        reinterpret_cast<std::uintptr_t>(buffers.rawInputV.data()), rawBytes(RawKind::kFloat16, kDefaultGeometry),
        rawBytes(RawKind::kFloat16, kDefaultGeometry), 0U, params};
    auto const validPlan = tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {validLayer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFloat16);

    auto invalidOffload = validOffload;
    invalidOffload.gpuPageIndex = -1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, validPlan, buffers.compactPage.data(), nullptr));
    auto invalidLayer = validLayer;
    invalidLayer.rawKBase += 1U;
    EXPECT_ANY_THROW(static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {invalidLayer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFloat16)));
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {validOffload}, validPlan, buffers.compactPage.bytes() + 1, nullptr));

    invalidLayer = validLayer;
    invalidLayer.rawKSlotBytes += alignof(uint4) / 2U;
    EXPECT_ANY_THROW(static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {invalidLayer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFloat16)));
    invalidLayer = validLayer;
    invalidLayer.rawVSlotBytes += alignof(uint4) / 2U;
    EXPECT_ANY_THROW(static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {invalidLayer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFloat16)));
    EXPECT_ANY_THROW(static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {validLayer}, coldPageBytes + alignof(uint4) / 2U, Nvfp4BoundaryRuntimeType::kFloat16)));

    auto invalidOnboard = validOnboard;
    invalidOnboard.coldPageIndex = -1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {invalidOnboard}, validPlan, buffers.compactPage.data(), nullptr));
    invalidLayer = validLayer;
    invalidLayer.rawVBase += 1U;
    EXPECT_ANY_THROW(static_cast<void>(tensorrt_llm::kernels::prepareNvfp4BoundaryPlan(
        {invalidLayer}, coldPageBytes, Nvfp4BoundaryRuntimeType::kFp8E4m3)));

    auto const unsupportedType = static_cast<Nvfp4BoundaryRuntimeType>(255);
    EXPECT_ANY_THROW(static_cast<void>(
        tensorrt_llm::kernels::prepareNvfp4BoundaryPlan({validLayer}, coldPageBytes, unsupportedType)));
}

} // namespace
