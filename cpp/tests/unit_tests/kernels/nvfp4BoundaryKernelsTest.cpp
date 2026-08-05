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

#include "tensorrt_llm/kernels/nvfp4BoundaryKernelsInternal.h"
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
using tensorrt_llm::kernels::Nvfp4BoundaryOffloadPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryOnboardPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryRuntimeType;
using tensorrt_llm::kernels::detail::Nvfp4BoundaryTransferPipeline;

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
constexpr PageGeometry kMinimumNativeGeometry{1, 4, 16};
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

    std::vector<std::uint8_t> copyToHost() const
    {
        std::vector<std::uint8_t> bytes(mPayloadBytes);
        EXPECT_EQ(cudaMemcpy(bytes.data(), data(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        return bytes;
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
        , packedK(packedBytes)
        , packedV(packedBytes)
        , scaleK(scaleBytes)
        , scaleV(scaleBytes)
    {
    }

    DeviceRegion rawInputK;
    DeviceRegion rawInputV;
    DeviceRegion rawOutputK;
    DeviceRegion rawOutputV;
    MappedHostRegion packedK;
    MappedHostRegion packedV;
    MappedHostRegion scaleK;
    MappedHostRegion scaleV;
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

std::uint32_t scaleOffset(std::uint32_t role, std::uint32_t row, std::uint32_t scaleInRow, PageGeometry const& geometry)
{
    std::uint32_t const scalesPerRow = static_cast<std::uint32_t>(geometry.headDim) / 16;
    if (role == 0)
    {
        return row * scalesPerRow + scaleInRow;
    }
    return (row / 4) * (4 * scalesPerRow) + scaleInRow * 4 + row % 4;
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
            result.scales[scaleOffset(role, row, scaleInRow, geometry)] = blockScale.__x;
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
            blockScale.__x = compressed.scales[scaleOffset(role, row, scaleInRow, geometry)];
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
    bool synchronizeBetweenDirections = true,
    Nvfp4BoundaryTransferPipeline offloadPipeline = Nvfp4BoundaryTransferPipeline::kAuto,
    Nvfp4BoundaryTransferPipeline onboardPipeline = Nvfp4BoundaryTransferPipeline::kAuto)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!tensorrt_llm::common::isSM100Family())
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }
    Nvfp4BoundaryKernelParams const params = makeParams(geometry);
    CudaStream stream;

    std::vector<std::unique_ptr<PageBuffers>> pages;
    std::vector<Nvfp4BoundaryOffloadPageTask> offloadTasks;
    pages.reserve(numPages);
    offloadTasks.reserve(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        auto buffers
            = std::make_unique<PageBuffers>(rawBytes(kind, geometry), packedBytes(geometry), scaleBytes(geometry));
        buffers->rawHost[0] = makeRawPage(kind, page, 0, params, geometry, inputPattern);
        buffers->rawHost[1] = makeRawPage(kind, page, 1, params, geometry, inputPattern);
        buffers->rawInputK.copyFrom(buffers->rawHost[0]);
        buffers->rawInputV.copyFrom(buffers->rawHost[1]);
        offloadTasks.push_back({buffers->rawInputK.data(), buffers->rawInputV.data(), buffers->packedK.bytes(),
            buffers->packedV.bytes(), buffers->scaleK.bytes(), buffers->scaleV.bytes()});
        pages.push_back(std::move(buffers));
    }

    auto runtimeType = Nvfp4BoundaryRuntimeType::kFp8E4m3;
    if (kind == RawKind::kFloat16)
    {
        runtimeType = Nvfp4BoundaryRuntimeType::kFloat16;
    }
    else if (kind == RawKind::kBfloat16)
    {
        runtimeType = Nvfp4BoundaryRuntimeType::kBfloat16;
    }
    std::vector<std::array<ReferenceNvfp4, 2>> references(numPages);
    for (std::size_t page = 0; page < numPages; ++page)
    {
        references[page][0] = compressReference(pages[page]->rawHost[0], kind, 0, params, geometry);
        references[page][1] = compressReference(pages[page]->rawHost[1], kind, 1, params, geometry);
    }

    auto const verifyCompressedPages = [&]
    {
        for (std::size_t page = 0; page < numPages; ++page)
        {
            EXPECT_EQ(pages[page]->packedK.payload(), references[page][0].packed);
            EXPECT_EQ(pages[page]->packedV.payload(), references[page][1].packed);
            EXPECT_EQ(pages[page]->scaleK.payload(), references[page][0].scales);
            EXPECT_EQ(pages[page]->scaleV.payload(), references[page][1].scales);
        }
    };

    tensorrt_llm::kernels::detail::invokeNvfp4BoundaryOffloadCompressWithPipeline(
        offloadTasks, params, runtimeType, offloadPipeline, stream);

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
    for (auto const& page : pages)
    {
        onboardTasks.push_back({page->packedK.bytes(), page->packedV.bytes(), page->scaleK.bytes(),
            page->scaleV.bytes(), page->rawOutputK.data(), page->rawOutputV.data()});
    }
    tensorrt_llm::kernels::detail::invokeNvfp4BoundaryOnboardDecompressWithPipeline(
        onboardTasks, params, runtimeType, onboardPipeline, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    if (!synchronizeBetweenDirections)
    {
        // PDL stress path: offload and onboard were enqueued back-to-back on
        // one stream with no Host wait, event, or payload read between them.
        // The final synchronization is the first point where Host NVFP4 bytes
        // are observed.
        verifyCompressedPages();
    }

    for (std::size_t page = 0; page < numPages; ++page)
    {
        EXPECT_EQ(
            pages[page]->rawOutputK.copyToHost(), decompressReference(references[page][0], kind, 0, params, geometry));
        EXPECT_EQ(
            pages[page]->rawOutputV.copyToHost(), decompressReference(references[page][1], kind, 1, params, geometry));
        pages[page]->rawInputK.expectCanaries();
        pages[page]->rawInputV.expectCanaries();
        pages[page]->rawOutputK.expectCanaries();
        pages[page]->rawOutputV.expectCanaries();
        pages[page]->packedK.expectCanaries();
        pages[page]->packedV.expectCanaries();
        pages[page]->scaleK.expectCanaries();
        pages[page]->scaleV.expectCanaries();
    }
}

class Nvfp4Boundary16BitTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4Boundary16BitTest, BatchesDisjointPagesAndMatchesNativeLayout)
{
    runBoundaryRoundTrip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RuntimeType, Nvfp4Boundary16BitTest, testing::Values(RawKind::kFloat16, RawKind::kBfloat16));

TEST(Nvfp4BoundaryFp8Test, BatchesDisjointPagesWithIndependentSourceAndTargetScales)
{
    runBoundaryRoundTrip(RawKind::kFp8);
}

TEST(Nvfp4BoundaryGeometryTest, SupportsMinimumNativeGeometryAndWarpTail)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kMinimumNativeGeometry, 1);
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

class Nvfp4BoundaryTiledOffloadTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4BoundaryTiledOffloadTest, MatchesCpuOracleAndNativeLayout)
{
    runBoundaryRoundTrip(GetParam(), kModelLikeGeometry, 3, InputPattern::kDense, true,
        Nvfp4BoundaryTransferPipeline::kCompressedOutputTiled);
}

INSTANTIATE_TEST_SUITE_P(AllRuntimeTypes, Nvfp4BoundaryTiledOffloadTest,
    testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

class Nvfp4BoundaryOnboardPhaseTest : public testing::TestWithParam<RawKind>
{
};

TEST_P(Nvfp4BoundaryOnboardPhaseTest, WholePageMatchesCpuOracle)
{
    runBoundaryRoundTrip(GetParam(), kModelLikeGeometry, 3, InputPattern::kDense, true,
        Nvfp4BoundaryTransferPipeline::kAuto, Nvfp4BoundaryTransferPipeline::kWholePage);
}

TEST_P(Nvfp4BoundaryOnboardPhaseTest, CompressedOutputTilesMatchCpuOracle)
{
    runBoundaryRoundTrip(GetParam(), kModelLikeGeometry, 3, InputPattern::kDense, true,
        Nvfp4BoundaryTransferPipeline::kAuto, Nvfp4BoundaryTransferPipeline::kCompressedOutputTiled);
}

INSTANTIATE_TEST_SUITE_P(AllRuntimeTypes, Nvfp4BoundaryOnboardPhaseTest,
    testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

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

INSTANTIATE_TEST_SUITE_P(AllRuntimeTypes, Nvfp4BoundaryInputPatternTest,
    testing::Values(RawKind::kFloat16, RawKind::kBfloat16, RawKind::kFp8));

TEST(Nvfp4BoundaryRoundingTest, QuantizesValuesSafelyAwayFromE2m1Ties)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kDefaultGeometry, 1, InputPattern::kRoundingMargins);
}

TEST(Nvfp4BoundaryBatchingTest, AcceptsMinimumThirtyTwoPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kMinimumNativeGeometry, 32);
}

TEST(Nvfp4BoundaryBatchingTest, SelectsSixtyFourPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFloat16, kMinimumNativeGeometry, 33);
}

TEST(Nvfp4BoundaryBatchingTest, Bfloat16SelectsOneHundredTwentyEightPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kMinimumNativeGeometry, 65);
}

TEST(Nvfp4BoundaryBatchingTest, Fp8SelectsTwoHundredFiftySixPageDescriptorSpecialization)
{
    runBoundaryRoundTrip(RawKind::kFp8, kMinimumNativeGeometry, 129);
}

TEST(Nvfp4BoundaryBatchingTest, Bfloat16CrossesTheTwoHundredFiftySixPageChunkBoundary)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kMinimumNativeGeometry, kCrossLaunchNumPages);
}

TEST(Nvfp4BoundaryBatchingTest, Fp8CrossesTheTwoHundredFiftySixPageChunkBoundary)
{
    runBoundaryRoundTrip(RawKind::kFp8, kMinimumNativeGeometry, kCrossLaunchNumPages);
}

TEST(Nvfp4BoundaryPdlTest, ChainsBfloat16OffloadAndOnboardWithoutIntermediateHostSync)
{
    runBoundaryRoundTrip(RawKind::kBfloat16, kMinimumNativeGeometry, 65, InputPattern::kDense, false);
}

TEST(Nvfp4BoundaryPdlTest, ChainsFp8OffloadAndOnboardWithoutIntermediateHostSync)
{
    runBoundaryRoundTrip(RawKind::kFp8, kMinimumNativeGeometry, 65, InputPattern::kDense, false);
}

TEST(Nvfp4BoundaryValidationTest, EmptyBatchIsAnAsyncNoOp)
{
    Nvfp4BoundaryKernelParams params{};
    tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress({}, params, Nvfp4BoundaryRuntimeType::kFp8E4m3, nullptr);
    tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr);
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
    std::vector<Nvfp4BoundaryOffloadPageTask> const tasks{{buffers.rawInputK.data(), buffers.rawInputV.data(),
        buffers.packedK.bytes(), buffers.packedV.bytes(), buffers.scaleK.bytes(), buffers.scaleV.bytes()}};
    auto const invoke16Bit = [&](Nvfp4BoundaryKernelParams const& params)
    {
        tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
            tasks, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr);
    };
    auto const invokeFp8 = [&](Nvfp4BoundaryKernelParams const& params)
    {
        tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
            tasks, params, Nvfp4BoundaryRuntimeType::kFp8E4m3, nullptr);
    };

    Nvfp4BoundaryKernelParams invalid = makeParams();
    invalid.numKvHeads = 0;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.tokensPerPage = 0;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.tokensPerPage = 6;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 0;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 24;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.headDim = 262144;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.numKvHeads = std::numeric_limits<std::int32_t>::max();
    EXPECT_ANY_THROW(invoke16Bit(invalid));

    invalid = makeParams();
    invalid.nvfp4ScaleOrigQuant[0] = 0.0F;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleQuantOrig[1] = -1.0F;
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleOrigQuant[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.nvfp4ScaleQuantOrig[0] = std::numeric_limits<float>::infinity();
    EXPECT_ANY_THROW(invoke16Bit(invalid));
    invalid = makeParams();
    invalid.fp8ScaleOrigQuant[0] = 0.0F;
    EXPECT_ANY_THROW(invokeFp8(invalid));
    invalid = makeParams();
    invalid.fp8ScaleQuantOrig[1] = std::numeric_limits<float>::infinity();
    EXPECT_ANY_THROW(invokeFp8(invalid));
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
    Nvfp4BoundaryOffloadPageTask const validOffload{buffers.rawInputK.data(), buffers.rawInputV.data(),
        buffers.packedK.bytes(), buffers.packedV.bytes(), buffers.scaleK.bytes(), buffers.scaleV.bytes()};
    Nvfp4BoundaryOnboardPageTask const validOnboard{buffers.packedK.bytes(), buffers.packedV.bytes(),
        buffers.scaleK.bytes(), buffers.scaleV.bytes(), buffers.rawOutputK.data(), buffers.rawOutputV.data()};

    auto invalidOffload = validOffload;
    invalidOffload.rawK = nullptr;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));
    invalidOffload = validOffload;
    invalidOffload.rawK = static_cast<std::uint8_t const*>(validOffload.rawK) + 1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));
    invalidOffload = validOffload;
    invalidOffload.packedV += 1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, params, Nvfp4BoundaryRuntimeType::kFp8E4m3, nullptr));
    invalidOffload = validOffload;
    invalidOffload.blockScaleK = nullptr;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, params, Nvfp4BoundaryRuntimeType::kFp8E4m3, nullptr));
    invalidOffload = validOffload;
    invalidOffload.blockScaleK += 1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(
        {invalidOffload}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));

    auto invalidOnboard = validOnboard;
    invalidOnboard.packedK = nullptr;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {invalidOnboard}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));
    invalidOnboard = validOnboard;
    invalidOnboard.rawV = static_cast<std::uint8_t*>(validOnboard.rawV) + 1;
    EXPECT_ANY_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {invalidOnboard}, params, Nvfp4BoundaryRuntimeType::kFp8E4m3, nullptr));

    // Onboard accepts a byte-aligned external scale tail even though offload's
    // dense mapped-Host scale stores require 16-byte destination alignment.
    // Compare it against the aligned path as well as checking the launch.
    MappedHostRegion byteAlignedScale(scaleBytes(kDefaultGeometry) + 1);
    DeviceRegion byteAlignedRawK(rawBytes(RawKind::kFloat16, kDefaultGeometry));
    DeviceRegion byteAlignedRawV(rawBytes(RawKind::kFloat16, kDefaultGeometry));
    auto byteAlignedOnboard = validOnboard;
    byteAlignedOnboard.blockScaleK = byteAlignedScale.bytes() + 1;
    byteAlignedOnboard.rawK = byteAlignedRawK.data();
    byteAlignedOnboard.rawV = byteAlignedRawV.data();
    EXPECT_NO_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {validOnboard}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));
    EXPECT_NO_THROW(tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(
        {byteAlignedOnboard}, params, Nvfp4BoundaryRuntimeType::kFloat16, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(buffers.rawOutputK.copyToHost(), byteAlignedRawK.copyToHost());
    EXPECT_EQ(buffers.rawOutputV.copyToHost(), byteAlignedRawV.copyToHost());

    auto const unsupportedType = static_cast<Nvfp4BoundaryRuntimeType>(255);
    EXPECT_ANY_THROW(
        tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress({validOffload}, params, unsupportedType, nullptr));
    EXPECT_ANY_THROW(
        tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress({validOnboard}, params, unsupportedType, nullptr));
}

} // namespace
