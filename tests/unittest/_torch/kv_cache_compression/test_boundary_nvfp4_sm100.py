# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-kernel checks for one-time NVFP4 GPU/Host boundary compression.

These tests intentionally do not use Attention or AttentionMetadata.  The
oracle decodes the packed nibbles and FP8 scale bytes with ordinary torch
operations, independently of the Triton decoder used by the implementation.
"""

from __future__ import annotations

import ctypes
import threading
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4 import dequant_nvfp4_2d_triton
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import HostMem

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 boundary compression requires SM100 or newer",
)


def _torch_nvfp4_decode_oracle(
    packed: torch.Tensor,
    scale_bytes: torch.Tensor,
    inverse_global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode linear NVFP4 without calling the production Triton decoder."""
    rows, packed_features = packed.shape
    feature_count = packed_features * 2

    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(rows, feature_count)
    codebook = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    values = codebook[codes].reshape(rows, feature_count // 16, 16)
    block_scales = scale_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    decoded = values * block_scales.unsqueeze(-1)
    decoded = decoded * inverse_global_scale.to(torch.float32)
    return decoded.reshape(rows, feature_count).to(dtype)


@requires_sm100
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 16), (3, 80), (32, 64), (129, 128)])
def test_linear_nvfp4_round_trip_matches_independent_oracle(dtype, shape):
    rows, feature_count = shape
    device = torch.device("cuda", torch.cuda.current_device())
    raw = (
        torch.linspace(
            -4.0,
            4.0,
            rows * feature_count,
            dtype=torch.float32,
            device=device,
        )
        .reshape(shape)
        .to(dtype)
    )

    packed, scale_bytes, inverse_global_scale = QuantizationForBoundaryCompression.compress_tensor(
        raw
    )

    assert packed.dtype == torch.uint8
    assert packed.shape == (rows, feature_count // 2)
    assert packed.is_contiguous()
    assert scale_bytes.dtype == torch.uint8
    assert scale_bytes.shape == (rows, feature_count // 16)
    assert scale_bytes.is_contiguous()
    assert inverse_global_scale.dtype == torch.float32
    assert inverse_global_scale.shape == (1,)
    assert torch.isfinite(inverse_global_scale).all()

    actual = dequant_nvfp4_2d_triton(
        packed,
        scale_bytes,
        inverse_global_scale,
        target_dtype=dtype,
    )
    expected = _torch_nvfp4_decode_oracle(
        packed,
        scale_bytes,
        inverse_global_scale,
        dtype,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("value_kind", ["zero", "tiny", "max"])
def test_bf16_extreme_global_scales_remain_finite_and_paired(value_kind):
    device = torch.device("cuda", torch.cuda.current_device())
    value = {
        "zero": 0.0,
        "tiny": torch.finfo(torch.bfloat16).tiny,
        "max": torch.finfo(torch.bfloat16).max,
    }[value_kind]
    raw = torch.full((2, 32), value, dtype=torch.bfloat16, device=device)

    packed, scales, inverse = QuantizationForBoundaryCompression.compress_tensor(raw)
    actual = dequant_nvfp4_2d_triton(
        packed,
        scales,
        inverse,
        target_dtype=torch.bfloat16,
    )
    expected = _torch_nvfp4_decode_oracle(
        packed,
        scales,
        inverse,
        torch.bfloat16,
    )

    assert torch.isfinite(inverse).all()
    assert torch.isfinite(torch.reciprocal(inverse)).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fresh_partial_physical_page_keeps_zero_tail_without_attention_metadata(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    tokens, heads, head_dim = 4, 2, 80
    physical = torch.randn(
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    physical[-1].zero_()
    raw = physical.reshape(tokens * heads, head_dim)

    packed, scales, inverse = QuantizationForBoundaryCompression.compress_tensor(raw)
    restored = dequant_nvfp4_2d_triton(
        packed,
        scales,
        inverse,
        target_dtype=dtype,
    ).reshape_as(physical)

    assert torch.count_nonzero(restored[-1]).item() == 0


def _one_buffer_manager(
    *,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_block: int,
    head_dim: int,
) -> QuantizationForBoundaryCompression:
    raw_size = tokens_per_block * head_dim * torch.empty((), dtype=dtype).element_size()
    record_size = QuantizationForBoundaryCompression._record_size(raw_size)
    buffer_id = SimpleNamespace(layer_id=0, role="key")
    coalesced = SimpleNamespace(
        single_buffer_size=raw_size,
        effective_host_single_buffer_size=record_size,
        size=raw_size,
        host_size=record_size,
        buffer_ids=(buffer_id,),
    )
    variant = SimpleNamespace(life_cycle_id=0, coalesced_buffers=(coalesced,))
    pool_group = SimpleNamespace(
        slot_desc=SimpleNamespace(variants=(variant,)),
    )
    kv_cache_manager = SimpleNamespace(
        num_kv_heads_per_layer=[1],
        head_dim_per_layer=[head_dim],
        tokens_per_block=tokens_per_block,
        impl=SimpleNamespace(pool_group_descs=(pool_group,)),
    )

    # Bypass the framework constructor: this test exercises only the two data
    # hooks with a minimal, source-faithful physical layout manifest.
    manager = object.__new__(QuantizationForBoundaryCompression)
    manager.kv_cache_manager = kv_cache_manager
    manager._torch_dtype = dtype
    manager._device = device
    manager._record_staging = torch.empty(record_size, dtype=torch.uint8, device=device)
    manager._record_staging_ready = torch.cuda.Event(blocking=False)
    manager._record_staging_in_flight = False
    manager._record_staging_lock = threading.Lock()
    return manager


@requires_sm100
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_host_record_alone_restores_page_with_unaligned_payload_sections(dtype):
    """Exercise the real 16-byte-grain KVCM copy path in both directions.

    Three rows by 80 features produce 120 packed bytes and 15 scale bytes;
    neither logical section is a multiple of the copy kernel's 16-byte grain.
    The implementation must therefore transfer one aligned record, not three
    independent sub-records.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    rows, head_dim = 3, 80
    manager = _one_buffer_manager(
        device=device,
        dtype=dtype,
        tokens_per_block=rows,
        head_dim=head_dim,
    )
    raw = (
        torch.linspace(
            -3.0,
            3.0,
            rows * head_dim,
            dtype=torch.float32,
            device=device,
        )
        .reshape(rows, head_dim)
        .to(dtype)
    )
    packed, scales, inverse = manager.compress_tensor(raw)
    expected = _torch_nvfp4_decode_oracle(packed, scales, inverse, dtype)
    # This focused hook test has no StorageManager/Page ready events around it,
    # so establish the source lease explicitly before using another stream.
    torch.cuda.synchronize(device)

    raw_size = raw.numel() * raw.element_size()
    record_size = manager._record_size(raw_size)
    assert record_size % 16 == 0
    host = HostMem(record_size)
    stream = torch.cuda.Stream(device=device)
    page = SimpleNamespace(life_cycle=0)
    common = dict(
        pool_group_index=0,
        src_pages=(page,),
        dst_slots=(SimpleNamespace(),),
        src_slot_sizes=(raw_size,),
        dst_slot_sizes=(record_size,),
        stream=stream.cuda_stream,
    )

    try:
        manager.on_offload_compress(
            **common,
            src_addresses=((raw.data_ptr(),),),
            dst_addresses=((host.address,),),
        )
        stream.synchronize()

        packed_size, scale_size, inverse_offset = manager._record_sections(
            raw_size,
            record_size,
        )
        host_bytes = ctypes.string_at(host.address, record_size)
        assert not any(host_bytes[packed_size + scale_size : inverse_offset])
        assert not any(host_bytes[inverse_offset + 4 :])

        # Destroy the only full-precision source before recall.  A successful
        # restore therefore proves that Host owns a complete compressed record,
        # rather than an auxiliary scale sidecar or a dual raw/compressed copy.
        raw.fill_(float("nan"))
        torch.cuda.synchronize(device)
        restored = torch.empty_like(raw)
        manager.on_onboard_decompress(
            pool_group_index=0,
            src_pages=(page,),
            dst_slots=(SimpleNamespace(),),
            src_addresses=((host.address,),),
            dst_addresses=((restored.data_ptr(),),),
            src_slot_sizes=(record_size,),
            dst_slot_sizes=(raw_size,),
            stream=stream.cuda_stream,
        )
        stream.synchronize()
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    finally:
        del host
