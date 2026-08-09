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

#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using tensorrt_llm::kernels::Nvfp4BoundaryKernelParams;
using tensorrt_llm::kernels::Nvfp4BoundaryOffloadPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryOnboardPageTask;
using tensorrt_llm::kernels::Nvfp4BoundaryRuntimeType;

constexpr std::int32_t kNumHeads = 8;
constexpr std::int32_t kTokensPerPage = 64;
constexpr std::int32_t kHeadDim = 128;
constexpr std::array<std::int32_t, 5> kValidTokenCounts{1, 16, 17, 63, 64};

enum class RuntimeKind
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

std::size_t elementBytes(RuntimeKind kind)
{
    return kind == RuntimeKind::kFp8E4m3 ? 1U : 2U;
}

std::size_t scalarCount()
{
    return static_cast<std::size_t>(kNumHeads) * kTokensPerPage * kHeadDim;
}

std::size_t rawPageBytes(RuntimeKind kind)
{
    return scalarCount() * elementBytes(kind);
}

std::size_t compactPageBytes()
{
    // K and V each use a 4-bit payload plus one E4M3 scale per 16 values.
    return scalarCount() * 9U / 8U;
}

Nvfp4BoundaryRuntimeType runtimeType(RuntimeKind kind)
{
    switch (kind)
    {
    case RuntimeKind::kFloat16: return Nvfp4BoundaryRuntimeType::kFloat16;
    case RuntimeKind::kBfloat16: return Nvfp4BoundaryRuntimeType::kBfloat16;
    case RuntimeKind::kFp8E4m3: return Nvfp4BoundaryRuntimeType::kFp8E4m3;
    }
    return Nvfp4BoundaryRuntimeType::kFloat16;
}

Nvfp4BoundaryKernelParams makeParams()
{
    Nvfp4BoundaryKernelParams params{};
    params.numKvHeads = kNumHeads;
    params.tokensPerPage = kTokensPerPage;
    params.headDim = kHeadDim;
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

void checkCuda(cudaError_t error, char const* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t bytes)
        : mBytes(bytes)
    {
        checkCuda(cudaMalloc(&mPointer, bytes), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (mPointer != nullptr)
        {
            cudaFree(mPointer);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    void* data() const noexcept
    {
        return mPointer;
    }

    void copyFrom(std::vector<std::uint8_t> const& bytes)
    {
        if (bytes.size() != mBytes)
        {
            throw std::invalid_argument("DeviceBuffer source size does not match the allocation");
        }
        checkCuda(cudaMemcpy(mPointer, bytes.data(), mBytes, cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");
    }

    std::vector<std::uint8_t> copyToHost() const
    {
        std::vector<std::uint8_t> bytes(mBytes);
        EXPECT_EQ(cudaMemcpy(bytes.data(), mPointer, mBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        return bytes;
    }

private:
    void* mPointer{};
    std::size_t mBytes{};
};

class MappedHostBuffer
{
public:
    explicit MappedHostBuffer(std::size_t bytes)
    {
        checkCuda(cudaHostAlloc(&mHost, bytes, cudaHostAllocMapped), "cudaHostAlloc");
        checkCuda(cudaHostGetDevicePointer(&mDevice, mHost, 0), "cudaHostGetDevicePointer");
        std::memset(mHost, 0, bytes);
    }

    ~MappedHostBuffer()
    {
        if (mHost != nullptr)
        {
            cudaFreeHost(mHost);
        }
    }

    MappedHostBuffer(MappedHostBuffer const&) = delete;
    MappedHostBuffer& operator=(MappedHostBuffer const&) = delete;

    std::uint8_t* deviceBytes() const noexcept
    {
        return static_cast<std::uint8_t*>(mDevice);
    }

private:
    void* mHost{};
    void* mDevice{};
};

template <typename T>
void storeScalar(std::vector<std::uint8_t>& bytes, std::size_t index, T value)
{
    std::memcpy(bytes.data() + index * sizeof(T), &value, sizeof(T));
}

void storeRuntimeValue(std::vector<std::uint8_t>& bytes, RuntimeKind kind, std::size_t index, float value)
{
    switch (kind)
    {
    case RuntimeKind::kFloat16: storeScalar(bytes, index, __float2half(value)); break;
    case RuntimeKind::kBfloat16: storeScalar(bytes, index, __float2bfloat16(value)); break;
    case RuntimeKind::kFp8E4m3:
    {
        __nv_fp8_e4m3 const fp8(value);
        bytes[index] = fp8.__x;
        break;
    }
    }
}

std::vector<std::uint8_t> makeRawPage(
    RuntimeKind kind, std::int32_t validTokens, bool zeroTail, std::uint32_t role)
{
    std::vector<std::uint8_t> bytes(rawPageBytes(kind));
    for (std::int32_t head = 0; head < kNumHeads; ++head)
    {
        for (std::int32_t token = 0; token < kTokensPerPage; ++token)
        {
            for (std::int32_t dim = 0; dim < kHeadDim; ++dim)
            {
                std::size_t const index
                    = (static_cast<std::size_t>(head) * kTokensPerPage + token) * kHeadDim + dim;
                float value = 0.0F;
                if (token < validTokens)
                {
                    // Identical valid prefixes for zero-tail and stale-tail Pages.
                    value = static_cast<float>((dim % 13) - 6) * 0.125F + static_cast<float>(head) * 0.03125F
                        + static_cast<float>(token) * 0.0078125F + static_cast<float>(role) * 0.0625F;
                }
                else if (!zeroTail)
                {
                    // Model a recycled Slot whose inactive rows are arbitrary,
                    // including non-finite values. Each 16-value scale group is
                    // contained within one token row, so none of these values
                    // may affect the valid prefix.
                    if (dim % 31 == 0)
                    {
                        value = std::numeric_limits<float>::quiet_NaN();
                    }
                    else if (dim % 29 == 0)
                    {
                        value = std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        value = static_cast<float>((dim % 9) - 4) * 0.25F
                            + static_cast<float>(token) * 0.015625F
                            + static_cast<float>(head + 3 * role) * 0.046875F;
                    }
                }
                storeRuntimeValue(bytes, kind, index, value);
            }
        }
    }
    return bytes;
}

struct PageBuffers
{
    PageBuffers(RuntimeKind kind, std::int32_t validTokens_, bool zeroTail_)
        : validTokens(validTokens_)
        , zeroTail(zeroTail_)
        , rawInputK(rawPageBytes(kind))
        , rawInputV(rawPageBytes(kind))
        , rawOutputK(rawPageBytes(kind))
        , rawOutputV(rawPageBytes(kind))
        , compact(compactPageBytes())
    {
        rawInputK.copyFrom(makeRawPage(kind, validTokens, zeroTail, 0));
        rawInputV.copyFrom(makeRawPage(kind, validTokens, zeroTail, 1));
    }

    std::int32_t validTokens;
    bool zeroTail;
    DeviceBuffer rawInputK;
    DeviceBuffer rawInputV;
    DeviceBuffer rawOutputK;
    DeviceBuffer rawOutputV;
    MappedHostBuffer compact;
};

void expectSameValidPrefix(std::vector<std::uint8_t> const& lhs, std::vector<std::uint8_t> const& rhs,
    RuntimeKind kind, std::int32_t validTokens)
{
    std::size_t const rowBytes = static_cast<std::size_t>(kHeadDim) * elementBytes(kind);
    for (std::int32_t head = 0; head < kNumHeads; ++head)
    {
        for (std::int32_t token = 0; token < validTokens; ++token)
        {
            std::size_t const offset
                = (static_cast<std::size_t>(head) * kTokensPerPage + token) * rowBytes;
            EXPECT_EQ(std::memcmp(lhs.data() + offset, rhs.data() + offset, rowBytes), 0)
                << "valid prefix differs at head=" << head << " token=" << token;
        }
    }
}

void expectZeroTail(std::vector<std::uint8_t> const& bytes, RuntimeKind kind, std::int32_t validTokens)
{
    std::size_t const rowBytes = static_cast<std::size_t>(kHeadDim) * elementBytes(kind);
    std::vector<std::uint8_t> const zero(rowBytes, 0);
    for (std::int32_t head = 0; head < kNumHeads; ++head)
    {
        for (std::int32_t token = validTokens; token < kTokensPerPage; ++token)
        {
            std::size_t const offset
                = (static_cast<std::size_t>(head) * kTokensPerPage + token) * rowBytes;
            EXPECT_EQ(std::memcmp(bytes.data() + offset, zero.data(), rowBytes), 0)
                << "zero tail changed at head=" << head << " token=" << token;
        }
    }
}

void runPartialPageBatch(RuntimeKind kind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    if (properties.major < 10)
    {
        GTEST_SKIP() << "NVFP4 boundary kernels require an SM100-family GPU";
    }

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    std::vector<std::unique_ptr<PageBuffers>> pages;
    std::vector<Nvfp4BoundaryOffloadPageTask> offloadTasks;
    for (std::int32_t validTokens : kValidTokenCounts)
    {
        for (bool zeroTail : {true, false})
        {
            auto page = std::make_unique<PageBuffers>(kind, validTokens, zeroTail);
            offloadTasks.push_back(
                {page->rawInputK.data(), page->rawInputV.data(), page->compact.deviceBytes()});
            pages.push_back(std::move(page));
        }
    }

    Nvfp4BoundaryKernelParams const params = makeParams();
    tensorrt_llm::kernels::invokeNvfp4BoundaryOffloadCompress(offloadTasks, params, runtimeType(kind), stream);

    std::vector<Nvfp4BoundaryOnboardPageTask> onboardTasks;
    onboardTasks.reserve(pages.size());
    for (auto const& page : pages)
    {
        onboardTasks.push_back(
            {page->compact.deviceBytes(), page->rawOutputK.data(), page->rawOutputV.data()});
    }
    tensorrt_llm::kernels::invokeNvfp4BoundaryOnboardDecompress(onboardTasks, params, runtimeType(kind), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    for (std::size_t pair = 0; pair < kValidTokenCounts.size(); ++pair)
    {
        auto const& zeroTailPage = pages.at(2 * pair);
        auto const& staleTailPage = pages.at(2 * pair + 1);
        ASSERT_TRUE(zeroTailPage->zeroTail);
        ASSERT_FALSE(staleTailPage->zeroTail);
        for (bool key : {true, false})
        {
            auto const zeroOutput
                = (key ? zeroTailPage->rawOutputK : zeroTailPage->rawOutputV).copyToHost();
            auto const staleOutput
                = (key ? staleTailPage->rawOutputK : staleTailPage->rawOutputV).copyToHost();
            expectSameValidPrefix(zeroOutput, staleOutput, kind, zeroTailPage->validTokens);
            expectZeroTail(zeroOutput, kind, zeroTailPage->validTokens);
        }
    }
}

TEST(Nvfp4BoundaryPartialPageTest, Float16TailDoesNotAffectValidPrefix)
{
    runPartialPageBatch(RuntimeKind::kFloat16);
}

TEST(Nvfp4BoundaryPartialPageTest, Bfloat16TailDoesNotAffectValidPrefix)
{
    runPartialPageBatch(RuntimeKind::kBfloat16);
}

TEST(Nvfp4BoundaryPartialPageTest, Fp8TailDoesNotAffectValidPrefix)
{
    runPartialPageBatch(RuntimeKind::kFp8E4m3);
}

} // namespace
