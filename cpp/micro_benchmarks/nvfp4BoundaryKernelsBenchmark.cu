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

//! Transport-equivalence and whole-pipeline A/B for the NVFP4 boundary kernels.
//!
//! This benchmark deliberately keeps Page selection, allocation, publication,
//! and synchronization policy outside the timed region. It compares:
//!
//!   fused       : raw GPU <-> mapped-Host NVFP4 in one transform kernel;
//!   staged_sm   : the same transform through GPU compact staging, followed
//!                 or preceded by KVCM V2's real batchedCopy implementation;
//!   staged_dma  : the same staging transform with cudaMemcpyBatchAsync.
//!
//! Passing CUDA-device compact pointers to the boundary transform is a
//! benchmark-only ablation. It does not extend the product API contract.

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;
namespace kernels = tensorrt_llm::kernels;

using Clock = std::chrono::steady_clock;
using MMTask = kv::Task<kv::MemAddress, kv::MemAddress>;
using kernels::Nvfp4BoundaryKernelParams;
using kernels::Nvfp4BoundaryOffloadPageTask;
using kernels::Nvfp4BoundaryOnboardPageTask;
using kernels::Nvfp4BoundaryRuntimeType;

constexpr std::int32_t kNumKvHeads = 8;
constexpr std::int32_t kTokensPerPage = 64;
constexpr std::int32_t kHeadDim = 128;
constexpr std::size_t kCopyAlignment = sizeof(uint4);
constexpr std::uint8_t kUntouchedByte = 0xA5;

enum class RawKind
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

enum class Direction
{
    kOffload,
    kOnboard,
};

enum class AddressMode
{
    kContiguous,
    kPermuted,
};

enum class Variant
{
    kFused,
    kStagedSm,
    kStagedDma,
};

struct Options
{
    std::size_t pages{32};
    RawKind dtype{RawKind::kBfloat16};
    Direction direction{Direction::kOffload};
    AddressMode addressMode{AddressMode::kContiguous};
    std::string outputCsv{"-"};
    int warmup{10};
    int iterations{100};
    int samples{15};
    std::uint64_t seed{20260805};
};

struct PoolLayout
{
    std::size_t rawLogicalBytes{};
    std::size_t rawStride{};
    std::size_t packedLogicalBytes{};
    std::size_t packedStride{};
    std::size_t scaleLogicalBytes{};
    std::size_t scaleStride{};

    [[nodiscard]] std::size_t rawBoundaryBytesPerPage() const
    {
        return 2 * rawLogicalBytes;
    }

    [[nodiscard]] std::size_t compactBoundaryBytesPerPage() const
    {
        return 2 * (packedLogicalBytes + scaleLogicalBytes);
    }

    [[nodiscard]] std::size_t compactStagingBytesPerPage() const
    {
        return 2 * (packedStride + scaleStride);
    }
};

struct TimingSample
{
    Variant variant{};
    int sample{};
    double gpuUs{};
    double cpuEnqueueUs{};
    double speedupOverFused{1.0};
};

struct Summary
{
    double minimum{};
    double median{};
    double p95{};
};

[[noreturn]] void fail(std::string const& message)
{
    throw std::runtime_error(message);
}

void checkCuda(cudaError_t status, std::string_view operation)
{
    if (status != cudaSuccess)
    {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkDriver(CUresult status, std::string_view operation)
{
    if (status != CUDA_SUCCESS)
    {
        char const* name = nullptr;
        char const* description = nullptr;
        static_cast<void>(cuGetErrorName(status, &name));
        static_cast<void>(cuGetErrorString(status, &description));
        std::ostringstream message;
        message << operation << ": " << (name == nullptr ? "unknown" : name) << " ("
                << (description == nullptr ? "no description" : description) << ')';
        fail(message.str());
    }
}

std::size_t roundUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0 || value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        fail("invalid aligned allocation size");
    }
    return (value + alignment - 1) / alignment * alignment;
}

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs, std::string_view name)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        fail(std::string(name) + " overflows size_t");
    }
    return lhs * rhs;
}

std::string_view toString(RawKind value)
{
    switch (value)
    {
    case RawKind::kFloat16: return "fp16";
    case RawKind::kBfloat16: return "bf16";
    case RawKind::kFp8E4m3: return "fp8_e4m3";
    }
    fail("invalid dtype");
}

std::string_view toString(Direction value)
{
    return value == Direction::kOffload ? "offload" : "onboard";
}

std::string_view toString(AddressMode value)
{
    return value == AddressMode::kContiguous ? "contiguous" : "permuted";
}

std::string_view toString(Variant value)
{
    switch (value)
    {
    case Variant::kFused: return "fused";
    case Variant::kStagedSm: return "staged_sm";
    case Variant::kStagedDma: return "staged_dma";
    }
    fail("invalid variant");
}

Nvfp4BoundaryRuntimeType runtimeType(RawKind value)
{
    switch (value)
    {
    case RawKind::kFloat16: return Nvfp4BoundaryRuntimeType::kFloat16;
    case RawKind::kBfloat16: return Nvfp4BoundaryRuntimeType::kBfloat16;
    case RawKind::kFp8E4m3: return Nvfp4BoundaryRuntimeType::kFp8E4m3;
    }
    fail("invalid dtype");
}

std::size_t rawElementBytes(RawKind value)
{
    return value == RawKind::kFp8E4m3 ? 1 : 2;
}

std::uint64_t parseUnsigned(std::string const& text, std::string_view option)
{
    if (text.empty() || text.front() == '-')
    {
        fail(std::string(option) + " requires a non-negative integer");
    }
    std::size_t parsed = 0;
    std::uint64_t value = 0;
    try
    {
        value = std::stoull(text, &parsed);
    }
    catch (std::exception const&)
    {
        fail(std::string(option) + " requires a non-negative integer");
    }
    if (parsed != text.size())
    {
        fail(std::string(option) + " contains trailing characters");
    }
    return value;
}

int parseInt(std::string const& text, std::string_view option)
{
    std::uint64_t const value = parseUnsigned(text, option);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        fail(std::string(option) + " exceeds INT_MAX");
    }
    return static_cast<int>(value);
}

std::string optionValue(int& index, int argc, char** argv, std::string const& argument, std::string_view name)
{
    std::string const prefix = std::string(name) + '=';
    if (argument.rfind(prefix, 0) == 0)
    {
        return argument.substr(prefix.size());
    }
    if (argument == name)
    {
        if (++index >= argc)
        {
            fail(std::string(name) + " requires a value");
        }
        return argv[index];
    }
    return {};
}

void printHelp(char const* executable)
{
    std::cout << "Usage: " << executable << " [options]\n\n"
              << "One process runs one fixed H=8, P=64, D=128 benchmark cell.\n"
              << "  --pages N                       Page cohort size (default: 32)\n"
              << "  --dtype fp16|bf16|fp8_e4m3     Runtime GPU dtype (default: bf16)\n"
              << "  --direction offload|onboard    Boundary direction (default: offload)\n"
              << "  --address-mode contiguous|permuted\n"
              << "                                  Slot-address pattern (default: contiguous)\n"
              << "  --output-csv PATH              CSV output, or - for stdout (default: -)\n"
              << "  --warmup N                     Warm-up iterations per variant (default: 10)\n"
              << "  --iterations N                 Timed iterations per sample (default: 100)\n"
              << "  --samples N                    Repeated samples (default: 15)\n"
              << "  --seed N                       Input/permutation seed (default: 20260805)\n"
              << "  --help                         Show this text\n\n"
              << "Run PDL modes in separate processes with TRTLLM_ENABLE_PDL=0 or 1.\n";
}

Options parseOptions(int argc, char** argv)
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        std::string const argument = argv[index];
        if (argument == "--help" || argument == "-h")
        {
            printHelp(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (auto value = optionValue(index, argc, argv, argument, "--pages"); !value.empty())
        {
            result.pages = parseUnsigned(value, "--pages");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--dtype"); !value.empty())
        {
            if (value == "fp16")
                result.dtype = RawKind::kFloat16;
            else if (value == "bf16")
                result.dtype = RawKind::kBfloat16;
            else if (value == "fp8_e4m3")
                result.dtype = RawKind::kFp8E4m3;
            else
                fail("--dtype must be fp16, bf16, or fp8_e4m3");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--direction"); !value.empty())
        {
            if (value == "offload")
                result.direction = Direction::kOffload;
            else if (value == "onboard")
                result.direction = Direction::kOnboard;
            else
                fail("--direction must be offload or onboard");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--address-mode"); !value.empty())
        {
            if (value == "contiguous")
                result.addressMode = AddressMode::kContiguous;
            else if (value == "permuted")
                result.addressMode = AddressMode::kPermuted;
            else
                fail("--address-mode must be contiguous or permuted");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--output-csv"); !value.empty())
        {
            result.outputCsv = std::move(value);
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--warmup"); !value.empty())
        {
            result.warmup = parseInt(value, "--warmup");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--iterations"); !value.empty())
        {
            result.iterations = parseInt(value, "--iterations");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--samples"); !value.empty())
        {
            result.samples = parseInt(value, "--samples");
        }
        else if (auto value = optionValue(index, argc, argv, argument, "--seed"); !value.empty())
        {
            result.seed = parseUnsigned(value, "--seed");
        }
        else
        {
            fail("unknown option: " + argument);
        }
    }
    if (result.pages == 0 || result.pages > std::numeric_limits<std::uint32_t>::max())
    {
        fail("--pages must be in [1, UINT32_MAX]");
    }
    if (result.warmup < 0 || result.iterations <= 0 || result.samples <= 0)
    {
        fail("--warmup must be non-negative; --iterations and --samples must be positive");
    }
    return result;
}

class CudaStream
{
public:
    CudaStream()
    {
        checkCuda(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(mStream));
        }
    }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

    operator cudaStream_t() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{};
};

class CudaEvent
{
public:
    CudaEvent()
    {
        checkCuda(cudaEventCreate(&mEvent), "cudaEventCreate");
    }

    ~CudaEvent()
    {
        if (mEvent != nullptr)
        {
            static_cast<void>(cudaEventDestroy(mEvent));
        }
    }

    CudaEvent(CudaEvent const&) = delete;
    CudaEvent& operator=(CudaEvent const&) = delete;

    operator cudaEvent_t() const
    {
        return mEvent;
    }

private:
    cudaEvent_t mEvent{};
};

class DevicePool
{
public:
    explicit DevicePool(std::size_t bytes)
        : mBytes(bytes)
    {
        checkCuda(cudaMalloc(&mData, mBytes), "cudaMalloc");
    }

    ~DevicePool()
    {
        if (mData != nullptr)
        {
            static_cast<void>(cudaFree(mData));
        }
    }

    DevicePool(DevicePool const&) = delete;
    DevicePool& operator=(DevicePool const&) = delete;

    [[nodiscard]] std::uint8_t* bytes() const
    {
        return static_cast<std::uint8_t*>(mData);
    }

    void fill(std::uint8_t value, cudaStream_t stream)
    {
        checkCuda(cudaMemsetAsync(mData, value, mBytes, stream), "cudaMemsetAsync DevicePool");
    }

    void copyFrom(std::vector<std::uint8_t> const& source)
    {
        if (source.size() != mBytes)
        {
            fail("DevicePool::copyFrom size mismatch");
        }
        checkCuda(cudaMemcpy(mData, source.data(), mBytes, cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");
    }

    [[nodiscard]] std::vector<std::uint8_t> copyToHost() const
    {
        std::vector<std::uint8_t> result(mBytes);
        checkCuda(cudaMemcpy(result.data(), mData, mBytes, cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");
        return result;
    }

private:
    void* mData{};
    std::size_t mBytes{};
};

class MappedHostPool
{
public:
    explicit MappedHostPool(std::size_t bytes)
        : mLogicalBytes(bytes)
        , mMemory(roundUp(bytes, kv::HostMem::kAlignment))
    {
        std::memset(bytesPointer(), kUntouchedByte, mMemory.size());
    }

    MappedHostPool(MappedHostPool const&) = delete;
    MappedHostPool& operator=(MappedHostPool const&) = delete;

    [[nodiscard]] std::uint8_t* bytesPointer() const
    {
        return reinterpret_cast<std::uint8_t*>(mMemory.address());
    }

    void fill(std::uint8_t value)
    {
        std::memset(bytesPointer(), value, mMemory.size());
    }

    [[nodiscard]] std::vector<std::uint8_t> copyToVector() const
    {
        return {bytesPointer(), bytesPointer() + mLogicalBytes};
    }

private:
    std::size_t mLogicalBytes{};
    kv::HostMem mMemory;
};

struct CompactDevicePools
{
    CompactDevicePools(std::size_t pages, PoolLayout const& layout)
        : packedK(checkedProduct(pages, layout.packedStride, "packed K Pool"))
        , packedV(checkedProduct(pages, layout.packedStride, "packed V Pool"))
        , scaleK(checkedProduct(pages, layout.scaleStride, "scale K Pool"))
        , scaleV(checkedProduct(pages, layout.scaleStride, "scale V Pool"))
    {
    }

    void fill(std::uint8_t value, cudaStream_t stream)
    {
        packedK.fill(value, stream);
        packedV.fill(value, stream);
        scaleK.fill(value, stream);
        scaleV.fill(value, stream);
    }

    DevicePool packedK;
    DevicePool packedV;
    DevicePool scaleK;
    DevicePool scaleV;
};

struct CompactHostPools
{
    CompactHostPools(std::size_t pages, PoolLayout const& layout)
        : packedK(checkedProduct(pages, layout.packedStride, "Host packed K Pool"))
        , packedV(checkedProduct(pages, layout.packedStride, "Host packed V Pool"))
        , scaleK(checkedProduct(pages, layout.scaleStride, "Host scale K Pool"))
        , scaleV(checkedProduct(pages, layout.scaleStride, "Host scale V Pool"))
    {
    }

    void fill(std::uint8_t value)
    {
        packedK.fill(value);
        packedV.fill(value);
        scaleK.fill(value);
        scaleV.fill(value);
    }

    MappedHostPool packedK;
    MappedHostPool packedV;
    MappedHostPool scaleK;
    MappedHostPool scaleV;
};

struct RawDevicePools
{
    RawDevicePools(std::size_t pages, PoolLayout const& layout)
        : k(checkedProduct(pages, layout.rawStride, "raw K Pool"))
        , v(checkedProduct(pages, layout.rawStride, "raw V Pool"))
    {
    }

    void fill(std::uint8_t value, cudaStream_t stream)
    {
        k.fill(value, stream);
        v.fill(value, stream);
    }

    DevicePool k;
    DevicePool v;
};

std::vector<std::size_t> makeSlotOrder(std::size_t pages, AddressMode mode, std::uint64_t seed)
{
    std::vector<std::size_t> result(pages);
    std::iota(result.begin(), result.end(), 0);
    if (mode == AddressMode::kContiguous)
    {
        return result;
    }

    // Fixed xorshift64* Fisher-Yates: deterministic across standard-library
    // implementations, unlike std::shuffle's implementation-defined mapping.
    auto next = [&seed]()
    {
        seed ^= seed >> 12U;
        seed ^= seed << 25U;
        seed ^= seed >> 27U;
        return seed * 0x2545F4914F6CDD1DULL;
    };
    for (std::size_t remaining = pages; remaining > 1; --remaining)
    {
        std::size_t const selected = static_cast<std::size_t>(next() % remaining);
        std::swap(result[remaining - 1], result[selected]);
    }
    return result;
}

std::uint8_t* slot(DevicePool const& pool, std::size_t physicalSlot, std::size_t stride)
{
    return pool.bytes() + physicalSlot * stride;
}

std::uint8_t* slot(MappedHostPool const& pool, std::size_t physicalSlot, std::size_t stride)
{
    return pool.bytesPointer() + physicalSlot * stride;
}

kv::MemAddress address(void const* pointer)
{
    return reinterpret_cast<kv::MemAddress>(pointer);
}

Nvfp4BoundaryKernelParams makeParams()
{
    Nvfp4BoundaryKernelParams result{};
    result.numKvHeads = kNumKvHeads;
    result.tokensPerPage = kTokensPerPage;
    result.headDim = kHeadDim;
    result.nvfp4ScaleOrigQuant[0] = 1.0F;
    result.nvfp4ScaleOrigQuant[1] = 2.0F;
    result.nvfp4ScaleQuantOrig[0] = 1.0F;
    result.nvfp4ScaleQuantOrig[1] = 0.5F;
    result.fp8ScaleOrigQuant[0] = 2.0F;
    result.fp8ScaleOrigQuant[1] = 4.0F;
    result.fp8ScaleQuantOrig[0] = 0.5F;
    result.fp8ScaleQuantOrig[1] = 0.25F;
    return result;
}

template <typename T>
void storeScalar(std::uint8_t* bytes, std::size_t index, T value)
{
    std::memcpy(bytes + index * sizeof(T), &value, sizeof(T));
}

void storeInput(std::uint8_t* destination, RawKind kind, std::size_t index, float value,
    Nvfp4BoundaryKernelParams const& params, std::uint32_t role)
{
    switch (kind)
    {
    case RawKind::kFloat16: storeScalar(destination, index, __float2half(value)); break;
    case RawKind::kBfloat16: storeScalar(destination, index, __float2bfloat16(value)); break;
    case RawKind::kFp8E4m3:
        storeScalar(destination, index, __nv_fp8_e4m3(value * params.fp8ScaleOrigQuant[role]));
        break;
    }
}

std::vector<std::uint8_t> makeRawPool(Options const& options, PoolLayout const& layout,
    std::vector<std::size_t> const& slots, Nvfp4BoundaryKernelParams const& params, std::uint32_t role)
{
    constexpr std::array<float, 16> pattern{
        0.0F, 0.5F, -1.0F, 1.5F, -2.0F, 3.0F, -4.0F, 6.0F, -0.5F, 1.0F, -1.5F, 2.0F, -3.0F, 4.0F, -6.0F, 0.5F};
    constexpr std::array<float, 4> blockScales{0.25F, 0.5F, 1.0F, 2.0F};
    std::size_t const elements = static_cast<std::size_t>(kNumKvHeads) * kTokensPerPage * kHeadDim;
    std::vector<std::uint8_t> result(options.pages * layout.rawStride, kUntouchedByte);
    for (std::size_t page = 0; page < options.pages; ++page)
    {
        auto* output = result.data() + slots[page] * layout.rawStride;
        for (std::size_t index = 0; index < elements; ++index)
        {
            std::size_t const group = index / 16;
            std::size_t const selector = index + page * 3 + role * 7 + static_cast<std::size_t>(options.seed);
            float value = pattern[selector % pattern.size()]
                * blockScales[(group + page + role + options.seed) % blockScales.size()]
                / params.nvfp4ScaleOrigQuant[role];
            if (((page / 32) & 1U) != 0)
            {
                value = -value;
            }
            storeInput(output, options.dtype, index, value, params, role);
        }
    }
    return result;
}

PoolLayout makeLayout(RawKind dtype)
{
    std::size_t const elements = static_cast<std::size_t>(kNumKvHeads) * kTokensPerPage * kHeadDim;
    PoolLayout result{};
    result.rawLogicalBytes = elements * rawElementBytes(dtype);
    result.rawStride = roundUp(result.rawLogicalBytes, kCopyAlignment);
    result.packedLogicalBytes = elements / 2;
    result.packedStride = roundUp(result.packedLogicalBytes, kCopyAlignment);
    result.scaleLogicalBytes = elements / 16;
    // KVCM V2 batchedCopy copies uint4 grains and therefore requires every
    // per-Pool Slot stride to be a multiple of 16 bytes.
    result.scaleStride = roundUp(result.scaleLogicalBytes, kCopyAlignment);
    return result;
}

void reportMismatch(std::vector<std::uint8_t> const& expected, std::vector<std::uint8_t> const& actual,
    std::string_view expectedName, std::string_view actualName)
{
    if (expected.size() != actual.size())
    {
        fail(std::string(expectedName) + " and " + std::string(actualName) + " have different sizes");
    }
    auto const mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first != expected.end())
    {
        std::size_t const offset = static_cast<std::size_t>(mismatch.first - expected.begin());
        std::ostringstream message;
        message << "byte-exact correctness failure at byte " << offset << ": " << expectedName << "="
                << static_cast<unsigned>(*mismatch.first) << ", " << actualName << "="
                << static_cast<unsigned>(*mismatch.second);
        fail(message.str());
    }
}

class BenchmarkFixture
{
public:
    explicit BenchmarkFixture(Options const& options)
        : mOptions(options)
        , mLayout(makeLayout(options.dtype))
        , mParams(makeParams())
        , mSlots(makeSlotOrder(options.pages, options.addressMode, options.seed))
        , mRawInput(options.pages, mLayout)
        , mRawFused(options.pages, mLayout)
        , mRawSm(options.pages, mLayout)
        , mRawDma(options.pages, mLayout)
        , mHostFused(options.pages, mLayout)
        , mHostSm(options.pages, mLayout)
        , mHostDma(options.pages, mLayout)
        , mStaging(options.pages, mLayout)
    {
        mRawInput.k.copyFrom(makeRawPool(options, mLayout, mSlots, mParams, 0));
        mRawInput.v.copyFrom(makeRawPool(options, mLayout, mSlots, mParams, 1));
        // The input copies use the legacy default stream, while every measured
        // operation uses a non-blocking stream. Establish completion explicitly
        // once, outside correctness and timing, instead of relying on default-
        // stream ordering that a non-blocking stream deliberately does not have.
        checkCuda(cudaDeviceSynchronize(), "raw input initialization synchronization");
        buildTasks();
        prepareCanonicalInput();
    }

    ~BenchmarkFixture()
    {
        // Drain any work that still owns Pool pointers before member
        // destruction releases mapped Host or device storage.
        static_cast<void>(cudaStreamSynchronize(mStream));
    }

    [[nodiscard]] PoolLayout const& layout() const
    {
        return mLayout;
    }

    [[nodiscard]] bool dmaAvailable() const
    {
        return mDmaAvailable;
    }

    [[nodiscard]] std::string const& dmaUnavailableReason() const
    {
        return mDmaUnavailableReason;
    }

    void verify()
    {
        resetDestinations();
        enqueue(Variant::kFused);
        checkCuda(cudaStreamSynchronize(mStream), "fused correctness synchronization");
        enqueue(Variant::kStagedSm);
        checkCuda(cudaStreamSynchronize(mStream), "staged-SM correctness synchronization");

        probeAndRunDma();

        if (mOptions.direction == Direction::kOffload)
        {
            compareCompact(mHostFused, mHostSm, "fused", "staged_sm");
            if (mDmaAvailable)
            {
                compareCompact(mHostFused, mHostDma, "fused", "staged_dma");
            }
        }
        else
        {
            compareRaw(mRawFused, mRawSm, "fused", "staged_sm");
            if (mDmaAvailable)
            {
                compareRaw(mRawFused, mRawDma, "fused", "staged_dma");
            }
        }
        // This gate establishes that only the transport implementation changed:
        // fused and staged pipelines produced identical bytes. Independent
        // quantization/layout correctness is owned by nvfp4BoundaryKernelsTest's
        // CPU known-answer oracle and is intentionally not duplicated here.
    }

    void warmup()
    {
        std::array<Variant, 3> const variants{Variant::kFused, Variant::kStagedSm, Variant::kStagedDma};
        for (Variant variant : variants)
        {
            if (variant == Variant::kStagedDma && !mDmaAvailable)
            {
                continue;
            }
            for (int iteration = 0; iteration < mOptions.warmup; ++iteration)
            {
                enqueue(variant);
            }
            checkCuda(cudaStreamSynchronize(mStream), "warm-up synchronization");
        }
    }

    TimingSample measure(Variant variant, int sample)
    {
        checkCuda(cudaStreamSynchronize(mStream), "pre-sample synchronization");
        checkCuda(cudaEventRecord(mStart, mStream), "record sample start");
        auto const cpuStart = Clock::now();
        for (int iteration = 0; iteration < mOptions.iterations; ++iteration)
        {
            enqueue(variant);
        }
        auto const cpuStop = Clock::now();
        checkCuda(cudaEventRecord(mStop, mStream), "record sample stop");
        checkCuda(cudaEventSynchronize(mStop), "sample synchronization");

        float elapsedMs = 0.0F;
        checkCuda(cudaEventElapsedTime(&elapsedMs, mStart, mStop), "cudaEventElapsedTime");
        double const iterations = static_cast<double>(mOptions.iterations);
        double const cpuUs = std::chrono::duration<double, std::micro>(cpuStop - cpuStart).count() / iterations;
        return {variant, sample, static_cast<double>(elapsedMs) * 1000.0 / iterations, cpuUs, 1.0};
    }

private:
    void buildTasks()
    {
        mOffloadFused.reserve(mOptions.pages);
        mOffloadStaging.reserve(mOptions.pages);
        mOnboardFused.reserve(mOptions.pages);
        mOnboardSm.reserve(mOptions.pages);
        mOnboardDma.reserve(mOptions.pages);
        for (std::size_t page = 0; page < mOptions.pages; ++page)
        {
            std::size_t const physical = mSlots[page];
            auto* rawInputK = slot(mRawInput.k, physical, mLayout.rawStride);
            auto* rawInputV = slot(mRawInput.v, physical, mLayout.rawStride);
            mOffloadFused.push_back({rawInputK, rawInputV, slot(mHostFused.packedK, physical, mLayout.packedStride),
                slot(mHostFused.packedV, physical, mLayout.packedStride),
                slot(mHostFused.scaleK, physical, mLayout.scaleStride),
                slot(mHostFused.scaleV, physical, mLayout.scaleStride)});
            mOffloadStaging.push_back({rawInputK, rawInputV, slot(mStaging.packedK, physical, mLayout.packedStride),
                slot(mStaging.packedV, physical, mLayout.packedStride),
                slot(mStaging.scaleK, physical, mLayout.scaleStride),
                slot(mStaging.scaleV, physical, mLayout.scaleStride)});

            mOnboardFused.push_back({slot(mHostFused.packedK, physical, mLayout.packedStride),
                slot(mHostFused.packedV, physical, mLayout.packedStride),
                slot(mHostFused.scaleK, physical, mLayout.scaleStride),
                slot(mHostFused.scaleV, physical, mLayout.scaleStride), slot(mRawFused.k, physical, mLayout.rawStride),
                slot(mRawFused.v, physical, mLayout.rawStride)});
            mOnboardSm.push_back({slot(mStaging.packedK, physical, mLayout.packedStride),
                slot(mStaging.packedV, physical, mLayout.packedStride),
                slot(mStaging.scaleK, physical, mLayout.scaleStride),
                slot(mStaging.scaleV, physical, mLayout.scaleStride), slot(mRawSm.k, physical, mLayout.rawStride),
                slot(mRawSm.v, physical, mLayout.rawStride)});
            mOnboardDma.push_back({slot(mStaging.packedK, physical, mLayout.packedStride),
                slot(mStaging.packedV, physical, mLayout.packedStride),
                slot(mStaging.scaleK, physical, mLayout.scaleStride),
                slot(mStaging.scaleV, physical, mLayout.scaleStride), slot(mRawDma.k, physical, mLayout.rawStride),
                slot(mRawDma.v, physical, mLayout.rawStride)});
        }

        buildSmTasks(mHostSm, mSmOffload, Direction::kOffload);
        buildSmTasks(mHostFused, mSmOnboard, Direction::kOnboard);
        buildDmaTasks();
    }

    void buildSmTasks(CompactHostPools const& host, std::array<std::vector<MMTask>, 4>& tasks, Direction direction)
    {
        for (auto& poolTasks : tasks)
        {
            poolTasks.reserve(mOptions.pages);
        }
        for (std::size_t page = 0; page < mOptions.pages; ++page)
        {
            std::size_t const physical = mSlots[page];
            auto append = [&](std::size_t pool, void* hostPointer, void* devicePointer)
            {
                void* destination = direction == Direction::kOffload ? hostPointer : devicePointer;
                void* source = direction == Direction::kOffload ? devicePointer : hostPointer;
                tasks[pool].push_back({address(destination), address(source)});
            };
            append(0, slot(host.packedK, physical, mLayout.packedStride),
                slot(mStaging.packedK, physical, mLayout.packedStride));
            append(1, slot(host.packedV, physical, mLayout.packedStride),
                slot(mStaging.packedV, physical, mLayout.packedStride));
            append(2, slot(host.scaleK, physical, mLayout.scaleStride),
                slot(mStaging.scaleK, physical, mLayout.scaleStride));
            append(3, slot(host.scaleV, physical, mLayout.scaleStride),
                slot(mStaging.scaleV, physical, mLayout.scaleStride));
        }
    }

    void appendDmaPool(void* destination, void const* source, std::size_t stride)
    {
        mDmaDestinations.push_back(destination);
        mDmaSources.push_back(source);
        mDmaSizes.push_back(stride);
    }

    void buildDmaTasks()
    {
        mDmaDestinations.reserve(4 * mOptions.pages);
        mDmaSources.reserve(4 * mOptions.pages);
        mDmaSizes.reserve(4 * mOptions.pages);
        for (std::size_t page = 0; page < mOptions.pages; ++page)
        {
            std::size_t const physical = mSlots[page];
            if (mOptions.direction == Direction::kOffload)
            {
                appendDmaPool(slot(mHostDma.packedK, physical, mLayout.packedStride),
                    slot(mStaging.packedK, physical, mLayout.packedStride), mLayout.packedStride);
                appendDmaPool(slot(mHostDma.packedV, physical, mLayout.packedStride),
                    slot(mStaging.packedV, physical, mLayout.packedStride), mLayout.packedStride);
                appendDmaPool(slot(mHostDma.scaleK, physical, mLayout.scaleStride),
                    slot(mStaging.scaleK, physical, mLayout.scaleStride), mLayout.scaleStride);
                appendDmaPool(slot(mHostDma.scaleV, physical, mLayout.scaleStride),
                    slot(mStaging.scaleV, physical, mLayout.scaleStride), mLayout.scaleStride);
            }
            else
            {
                appendDmaPool(slot(mStaging.packedK, physical, mLayout.packedStride),
                    slot(mHostFused.packedK, physical, mLayout.packedStride), mLayout.packedStride);
                appendDmaPool(slot(mStaging.packedV, physical, mLayout.packedStride),
                    slot(mHostFused.packedV, physical, mLayout.packedStride), mLayout.packedStride);
                appendDmaPool(slot(mStaging.scaleK, physical, mLayout.scaleStride),
                    slot(mHostFused.scaleK, physical, mLayout.scaleStride), mLayout.scaleStride);
                appendDmaPool(slot(mStaging.scaleV, physical, mLayout.scaleStride),
                    slot(mHostFused.scaleV, physical, mLayout.scaleStride), mLayout.scaleStride);
            }
        }
    }

    void prepareCanonicalInput()
    {
        // Onboard needs one byte-stable native NVFP4 Host source. Produce it
        // once with the fused offload path before correctness or timing.
        if (mOptions.direction == Direction::kOnboard)
        {
            kernels::invokeNvfp4BoundaryOffloadCompress(mOffloadFused, mParams, runtimeType(mOptions.dtype), mStream);
            checkCuda(cudaStreamSynchronize(mStream), "canonical NVFP4 preparation");
        }
    }

    void resetDestinations()
    {
        // Queue device clears on the same stream as the verification kernels.
        // Host clears are safe here because verify() runs before any timed work.
        mStaging.fill(kUntouchedByte, mStream);
        if (mOptions.direction == Direction::kOffload)
        {
            mHostFused.fill(kUntouchedByte);
            mHostSm.fill(kUntouchedByte);
            mHostDma.fill(kUntouchedByte);
        }
        else
        {
            mRawFused.fill(kUntouchedByte, mStream);
            mRawSm.fill(kUntouchedByte, mStream);
            mRawDma.fill(kUntouchedByte, mStream);
        }
    }

    void copyCompactWithSm(Direction direction)
    {
        auto const& tasks = direction == Direction::kOffload ? mSmOffload : mSmOnboard;
        CUstream const stream = reinterpret_cast<CUstream>(static_cast<cudaStream_t>(mStream));
        auto copyPool = [&](std::vector<MMTask> const& poolTasks, std::size_t bytes, std::string_view name)
        {
            CUresult const status = direction == Direction::kOffload
                ? kv::copyDeviceToHost(poolTasks, static_cast<ssize_t>(bytes), stream)
                : kv::copyHostToDevice(poolTasks, static_cast<ssize_t>(bytes), stream);
            checkDriver(status, name);
        };
        // Four calls preserve current StorageManager's one-call-per-Pool
        // semantics. Combining K/V or scale Pools would be a separate
        // coalescing optimization, not the primary baseline.
        copyPool(tasks[0], mLayout.packedStride, "batchedCopy packed K");
        copyPool(tasks[1], mLayout.packedStride, "batchedCopy packed V");
        copyPool(tasks[2], mLayout.scaleStride, "batchedCopy scale K");
        copyPool(tasks[3], mLayout.scaleStride, "batchedCopy scale V");
    }

    cudaError_t copyCompactWithDma()
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        // Match the CUDA-13 call shape already used by TRT-LLM's
        // asyncUlyssesOp: every operation selects the one stream-ordered
        // attribute record. flags=1 requests overlap with compute.
        cudaMemcpyAttributes attributes[1]{};
        attributes[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attributes[0].flags = 1U;
        std::size_t attributeIndices[1]{0};
        return cudaMemcpyBatchAsync(mDmaDestinations.data(), mDmaSources.data(), mDmaSizes.data(), mDmaSizes.size(),
            attributes, attributeIndices, 1, mStream);
#else
        return cudaErrorNotSupported;
#endif
    }

    void enqueue(Variant variant)
    {
        if (mOptions.direction == Direction::kOffload)
        {
            if (variant == Variant::kFused)
            {
                kernels::invokeNvfp4BoundaryOffloadCompress(
                    mOffloadFused, mParams, runtimeType(mOptions.dtype), mStream);
            }
            else
            {
                kernels::invokeNvfp4BoundaryOffloadCompress(
                    mOffloadStaging, mParams, runtimeType(mOptions.dtype), mStream);
                if (variant == Variant::kStagedSm)
                {
                    copyCompactWithSm(Direction::kOffload);
                }
                else
                {
                    checkCuda(copyCompactWithDma(), "cudaMemcpyBatchAsync offload");
                }
            }
        }
        else
        {
            if (variant == Variant::kFused)
            {
                kernels::invokeNvfp4BoundaryOnboardDecompress(
                    mOnboardFused, mParams, runtimeType(mOptions.dtype), mStream);
            }
            else
            {
                if (variant == Variant::kStagedSm)
                {
                    copyCompactWithSm(Direction::kOnboard);
                    kernels::invokeNvfp4BoundaryOnboardDecompress(
                        mOnboardSm, mParams, runtimeType(mOptions.dtype), mStream);
                }
                else
                {
                    checkCuda(copyCompactWithDma(), "cudaMemcpyBatchAsync onboard");
                    kernels::invokeNvfp4BoundaryOnboardDecompress(
                        mOnboardDma, mParams, runtimeType(mOptions.dtype), mStream);
                }
            }
        }
    }

    void probeAndRunDma()
    {
#if !defined(CUDART_VERSION) || CUDART_VERSION < 13000
        mDmaAvailable = false;
        mDmaUnavailableReason = "compiled CUDA Runtime is older than 13.0; cudaMemcpyBatchAsync was not built";
#else
        cudaError_t const status = copyCompactWithDmaProbe();
        if (status == cudaErrorNotSupported)
        {
            mDmaAvailable = false;
            mDmaUnavailableReason = "cudaMemcpyBatchAsync returned cudaErrorNotSupported";
            static_cast<void>(cudaGetLastError());
            checkCuda(cudaStreamSynchronize(mStream), "staged-DMA unsupported-path synchronization");
            return;
        }
        checkCuda(status, "cudaMemcpyBatchAsync correctness probe");
        if (mOptions.direction == Direction::kOnboard)
        {
            kernels::invokeNvfp4BoundaryOnboardDecompress(mOnboardDma, mParams, runtimeType(mOptions.dtype), mStream);
        }
        cudaError_t const completion = cudaStreamSynchronize(mStream);
        if (completion == cudaErrorNotSupported)
        {
            fail(
                "cudaMemcpyBatchAsync reported cudaErrorNotSupported asynchronously; "
                "the CUDA stream is no longer safe to benchmark");
        }
        checkCuda(completion, "staged-DMA correctness synchronization");
#endif
    }

    cudaError_t copyCompactWithDmaProbe()
    {
        if (mOptions.direction == Direction::kOffload)
        {
            kernels::invokeNvfp4BoundaryOffloadCompress(mOffloadStaging, mParams, runtimeType(mOptions.dtype), mStream);
        }
        return copyCompactWithDma();
    }

    static void comparePool(DevicePool const& expected, DevicePool const& actual, std::string_view expectedName,
        std::string_view actualName, std::string_view poolName)
    {
        reportMismatch(expected.copyToHost(), actual.copyToHost(),
            std::string(expectedName) + " " + std::string(poolName),
            std::string(actualName) + " " + std::string(poolName));
    }

    static void comparePool(MappedHostPool const& expected, MappedHostPool const& actual, std::string_view expectedName,
        std::string_view actualName, std::string_view poolName)
    {
        reportMismatch(expected.copyToVector(), actual.copyToVector(),
            std::string(expectedName) + " " + std::string(poolName),
            std::string(actualName) + " " + std::string(poolName));
    }

    static void compareCompact(CompactHostPools const& expected, CompactHostPools const& actual,
        std::string_view expectedName, std::string_view actualName)
    {
        comparePool(expected.packedK, actual.packedK, expectedName, actualName, "packed K");
        comparePool(expected.packedV, actual.packedV, expectedName, actualName, "packed V");
        comparePool(expected.scaleK, actual.scaleK, expectedName, actualName, "scale K");
        comparePool(expected.scaleV, actual.scaleV, expectedName, actualName, "scale V");
    }

    static void compareRaw(RawDevicePools const& expected, RawDevicePools const& actual, std::string_view expectedName,
        std::string_view actualName)
    {
        comparePool(expected.k, actual.k, expectedName, actualName, "raw K");
        comparePool(expected.v, actual.v, expectedName, actualName, "raw V");
    }

    Options const& mOptions;
    PoolLayout mLayout;
    Nvfp4BoundaryKernelParams mParams;
    std::vector<std::size_t> mSlots;
    CudaStream mStream;
    CudaEvent mStart;
    CudaEvent mStop;

    RawDevicePools mRawInput;
    RawDevicePools mRawFused;
    RawDevicePools mRawSm;
    RawDevicePools mRawDma;
    CompactHostPools mHostFused;
    CompactHostPools mHostSm;
    CompactHostPools mHostDma;
    CompactDevicePools mStaging;

    std::vector<Nvfp4BoundaryOffloadPageTask> mOffloadFused;
    std::vector<Nvfp4BoundaryOffloadPageTask> mOffloadStaging;
    std::vector<Nvfp4BoundaryOnboardPageTask> mOnboardFused;
    std::vector<Nvfp4BoundaryOnboardPageTask> mOnboardSm;
    std::vector<Nvfp4BoundaryOnboardPageTask> mOnboardDma;
    std::array<std::vector<MMTask>, 4> mSmOffload;
    std::array<std::vector<MMTask>, 4> mSmOnboard;
    std::vector<void*> mDmaDestinations;
    std::vector<void const*> mDmaSources;
    std::vector<std::size_t> mDmaSizes;
    bool mDmaAvailable{true};
    std::string mDmaUnavailableReason;
};

Summary summarize(std::vector<double> values)
{
    if (values.empty())
    {
        fail("cannot summarize an empty sample set");
    }
    std::sort(values.begin(), values.end());
    std::size_t const middle = values.size() / 2;
    double const median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
    std::size_t const p95Index
        = std::min(values.size() - 1, static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1);
    return {values.front(), median, values[p95Index]};
}

double pagesPerSecond(std::size_t pages, double microseconds)
{
    return static_cast<double>(pages) * 1.0e6 / microseconds;
}

double gigabytesPerSecond(std::size_t bytesPerPage, std::size_t pages, double microseconds)
{
    return static_cast<double>(bytesPerPage) * static_cast<double>(pages) / (microseconds * 1.0e3);
}

std::string csvEscape(std::string const& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos)
    {
        return value;
    }
    std::string result{"\""};
    for (char character : value)
    {
        if (character == '\"')
        {
            result += "\"\"";
        }
        else
        {
            result += character;
        }
    }
    result += '\"';
    return result;
}

class CsvWriter
{
public:
    explicit CsvWriter(std::string const& path)
    {
        if (path == "-")
        {
            mOutput = &std::cout;
        }
        else
        {
            mFile.open(path, std::ios::out | std::ios::trunc);
            if (!mFile)
            {
                fail("cannot open CSV output: " + path);
            }
            mOutput = &mFile;
        }
        *mOutput << "row_kind,status,note,variant,sample,direction,dtype,address_mode,pages,num_kv_heads,"
                    "tokens_per_page,head_dim,pdl,warmup,iterations,seed,gpu_us,cpu_enqueue_us,min_gpu_us,"
                    "median_gpu_us,p95_gpu_us,pages_per_s,effective_raw_gbps,effective_compact_gbps,"
                    "staged_over_fused,"
                    "raw_bytes_per_page,compact_bytes_per_page,gpu_staging_bytes,cudart_version,driver_version,"
                    "device_name\n";
        mOutput->setf(std::ios::fixed);
        *mOutput << std::setprecision(6);
    }

    void sample(Options const& options, PoolLayout const& layout, TimingSample const& sample, bool pdl,
        int runtimeVersion, int driverVersion, std::string const& deviceName)
    {
        double const pps = pagesPerSecond(options.pages, sample.gpuUs);
        double const rawGbps = gigabytesPerSecond(layout.rawBoundaryBytesPerPage(), options.pages, sample.gpuUs);
        double const compactGbps
            = gigabytesPerSecond(layout.compactBoundaryBytesPerPage(), options.pages, sample.gpuUs);
        std::size_t const stagingBytes
            = sample.variant == Variant::kFused ? 0 : options.pages * layout.compactStagingBytesPerPage();
        *mOutput << "sample,ok,," << toString(sample.variant) << ',' << sample.sample << ','
                 << toString(options.direction) << ',' << toString(options.dtype) << ','
                 << toString(options.addressMode) << ',' << options.pages << ',' << kNumKvHeads << ',' << kTokensPerPage
                 << ',' << kHeadDim << ',' << (pdl ? 1 : 0) << ',' << options.warmup << ',' << options.iterations << ','
                 << options.seed << ',' << sample.gpuUs << ',' << sample.cpuEnqueueUs << ",,,," << pps << ',' << rawGbps
                 << ',' << compactGbps << ',' << sample.speedupOverFused << ',' << layout.rawBoundaryBytesPerPage()
                 << ',' << layout.compactBoundaryBytesPerPage() << ',' << stagingBytes << ',' << runtimeVersion << ','
                 << driverVersion << ',' << csvEscape(deviceName) << '\n';
    }

    void summary(Options const& options, PoolLayout const& layout, Variant variant, Summary const& gpu,
        Summary const& cpu, double speedup, bool pdl, int runtimeVersion, int driverVersion,
        std::string const& deviceName)
    {
        double const pps = pagesPerSecond(options.pages, gpu.median);
        double const rawGbps = gigabytesPerSecond(layout.rawBoundaryBytesPerPage(), options.pages, gpu.median);
        double const compactGbps = gigabytesPerSecond(layout.compactBoundaryBytesPerPage(), options.pages, gpu.median);
        std::size_t const stagingBytes
            = variant == Variant::kFused ? 0 : options.pages * layout.compactStagingBytesPerPage();
        *mOutput << "summary,ok,," << toString(variant) << ",," << toString(options.direction) << ','
                 << toString(options.dtype) << ',' << toString(options.addressMode) << ',' << options.pages << ','
                 << kNumKvHeads << ',' << kTokensPerPage << ',' << kHeadDim << ',' << (pdl ? 1 : 0) << ','
                 << options.warmup << ',' << options.iterations << ',' << options.seed << ",," << cpu.median << ','
                 << gpu.minimum << ',' << gpu.median << ',' << gpu.p95 << ',' << pps << ',' << rawGbps << ','
                 << compactGbps << ',' << speedup << ',' << layout.rawBoundaryBytesPerPage() << ','
                 << layout.compactBoundaryBytesPerPage() << ',' << stagingBytes << ',' << runtimeVersion << ','
                 << driverVersion << ',' << csvEscape(deviceName) << '\n';
    }

    void unsupported(Options const& options, PoolLayout const& layout, Variant variant, std::string const& reason,
        bool pdl, int runtimeVersion, int driverVersion, std::string const& deviceName)
    {
        *mOutput << "summary,unsupported," << csvEscape(reason) << ',' << toString(variant) << ",,"
                 << toString(options.direction) << ',' << toString(options.dtype) << ','
                 << toString(options.addressMode) << ',' << options.pages << ',' << kNumKvHeads << ',' << kTokensPerPage
                 << ',' << kHeadDim << ',' << (pdl ? 1 : 0) << ',' << options.warmup << ',' << options.iterations << ','
                 << options.seed << ",,,,,,,,,," << layout.rawBoundaryBytesPerPage() << ','
                 << layout.compactBoundaryBytesPerPage() << ',' << options.pages * layout.compactStagingBytesPerPage()
                 << ',' << runtimeVersion << ',' << driverVersion << ',' << csvEscape(deviceName) << '\n';
    }

    void finish()
    {
        mOutput->flush();
        if (!*mOutput)
        {
            fail("failed to write benchmark CSV");
        }
    }

private:
    std::ofstream mFile;
    std::ostream* mOutput{};
};

std::vector<Variant> variantOrder(int sample, bool dmaAvailable)
{
    std::vector<Variant> result{Variant::kFused, Variant::kStagedSm};
    if (dmaAvailable)
    {
        result.push_back(Variant::kStagedDma);
    }
    if (!result.empty())
    {
        // Rotate every variant through every measurement position. This avoids
        // pinning staged_sm to the middle when three variants are available.
        std::rotate(result.begin(), result.begin() + sample % result.size(), result.end());
    }
    return result;
}

int run(Options const& options)
{
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    if (!tensorrt_llm::common::isSM100Family())
    {
        fail("NVFP4 boundary benchmark requires an SM100-family GPU");
    }

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int runtimeVersion = 0;
    int driverVersion = 0;
    checkCuda(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion");
    checkCuda(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion");
    bool const pdl = tensorrt_llm::common::getEnvEnablePDL();

    BenchmarkFixture fixture(options);
    fixture.verify();
    if (!fixture.dmaAvailable())
    {
        std::cerr << "staged_dma unsupported: " << fixture.dmaUnavailableReason() << '\n';
    }
    fixture.warmup();

    std::vector<TimingSample> samples;
    samples.reserve(static_cast<std::size_t>(options.samples) * (fixture.dmaAvailable() ? 3 : 2));
    for (int sample = 0; sample < options.samples; ++sample)
    {
        std::size_t const begin = samples.size();
        for (Variant variant : variantOrder(sample, fixture.dmaAvailable()))
        {
            samples.push_back(fixture.measure(variant, sample));
        }
        auto const fused = std::find_if(samples.begin() + begin, samples.end(),
            [](TimingSample const& value) { return value.variant == Variant::kFused; });
        for (auto current = samples.begin() + begin; current != samples.end(); ++current)
        {
            current->speedupOverFused = current->gpuUs / fused->gpuUs;
        }
    }

    CsvWriter output(options.outputCsv);
    for (TimingSample const& sample : samples)
    {
        output.sample(options, fixture.layout(), sample, pdl, runtimeVersion, driverVersion, properties.name);
    }

    Summary fusedGpu{};
    for (Variant variant : {Variant::kFused, Variant::kStagedSm, Variant::kStagedDma})
    {
        if (variant == Variant::kStagedDma && !fixture.dmaAvailable())
        {
            output.unsupported(options, fixture.layout(), variant, fixture.dmaUnavailableReason(), pdl, runtimeVersion,
                driverVersion, properties.name);
            continue;
        }
        std::vector<double> gpuValues;
        std::vector<double> cpuValues;
        for (TimingSample const& sample : samples)
        {
            if (sample.variant == variant)
            {
                gpuValues.push_back(sample.gpuUs);
                cpuValues.push_back(sample.cpuEnqueueUs);
            }
        }
        Summary const gpu = summarize(gpuValues);
        Summary const cpu = summarize(cpuValues);
        if (variant == Variant::kFused)
        {
            fusedGpu = gpu;
        }
        output.summary(options, fixture.layout(), variant, gpu, cpu, gpu.median / fusedGpu.median, pdl, runtimeVersion,
            driverVersion, properties.name);
    }
    output.finish();
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return run(parseOptions(argc, argv));
    }
    catch (std::exception const& exception)
    {
        std::cerr << "nvfp4BoundaryKernelsBenchmark: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
