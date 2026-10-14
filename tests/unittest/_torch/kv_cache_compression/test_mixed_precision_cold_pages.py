# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU storage/Attention QA using synthetic K/V, not model accuracy."""

import pytest
import torch
from cuda.bindings import runtime as cudart
from transformers import PretrainedConfig

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_cache import (
    add_mixed_precision_windows,
    cold_page_quantizer_config,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("skip_rope_quantization", (False, True))
def test_mixed_precision_cold_pages_keep_high_values_through_host(skip_rope_quantization):
    """Both pools migrate; only history is quantized and each token is read once."""
    if not is_sm_100f():
        pytest.skip("NVFP4 requires an SM100-family GPU")
    block, heads, width, source_blocks = 128, 8, 256, 8
    page_bytes = heads * block * width * 2
    buffers = [kv.BufferConfig(role=role, size=page_bytes) for role in ("key", "value")]
    config, (pair,) = add_mixed_precision_windows(
        kv.KVCacheManagerConfig(
            tokens_per_block=block,
            cache_tiers=[
                kv.GpuCacheTierConfig(quota=32 << 20),
                kv.HostCacheTierConfig(quota=64 << 20),
            ],
            layers=[kv.AttentionLayerConfig(layer_id=0, buffers=buffers)],
            initial_pool_ratio=[0.75, 0.25],
        ),
        {0: buffers},
        first_n=256,
        last_n=256,
    )
    pretrained = PretrainedConfig(head_dim=width, partial_rotary_factor=0.25)
    pretrained.model_type = "qwen3_5"
    provider = Nvfp4ColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(skip_rope_quantization=skip_rope_quantization),
        pretrained_config=pretrained,
    )
    codec = provider.create_cold_page_codec(
        cold_page_quantizer_config(config, (pair,)),
        runtime_dtype=DataType.BF16,
        pp_layers=(0,),
        num_kv_heads_per_layer=(heads,),
        head_dim_per_layer=(width,),
    )
    manager = kv.KVCacheManager(config, cold_page_codec=codec)
    stream = torch.cuda.current_stream()
    caches = []
    groups = [manager.get_layer_group_id(layer) for layer in (pair.history_layer, pair.high_layer)]
    generator = torch.Generator(device="cuda").manual_seed(42)
    original = {
        role: torch.randn(
            source_blocks, heads, block, width, generator=generator, device="cuda"
        ).to(torch.bfloat16)
        for role in ("key", "value")
    }

    def copy_page(cache, high, role, ordinal, values, *, write):
        base, stride, converter = pair.buffer_addressing(manager, role, high=high)
        slots = cache.get_base_page_indices(groups[high])
        slot = converter([slots[ordinal]], kv.PageIndexMode.SHARED)[0]
        assert slot >= 0
        address = base + slot * stride
        source, destination = (
            (values.data_ptr(), address) if write else (address, values.data_ptr())
        )
        assert (
            cudart.cudaMemcpyAsync(
                destination,
                source,
                page_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )[0]
            == cudart.cudaError_t.cudaSuccess
        )

    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream) and source.resize(source_blocks * block)
        for high in (False, True):
            slots = source.get_base_page_indices(groups[high])
            for ordinal, slot in enumerate(slots):
                if slot >= 0:
                    for role in original:
                        copy_page(source, high, role, ordinal, original[role][ordinal], write=True)
        source.commit(list(range(source_blocks * block)))
        source.suspend()

        # Build pressure a block at a time so the native SWA allocator can recycle
        # obsolete working pages; the test does not manage a second window itself.
        pressure = manager.create_kv_cache(None, [])
        caches.append(pressure)
        assert pressure.resume(stream.cuda_stream)
        for ordinal in range(23):
            assert pressure.resize((ordinal + 1) * block)
            for high in (False, True):
                for role in original:
                    copy_page(pressure, high, role, ordinal, original[role][0], write=True)
            pressure.commit(list(range(10000 + ordinal * block, 10000 + (ordinal + 1) * block)))
        stream.synchronize()
        # Evidence of actual Host migration, not merely a configured Host tier.
        for group in groups:
            assert 1 in source.get_page_storage_snapshot(group).cache_levels
        pressure.close()
        caches.remove(pressure)
        assert source.resume(stream.cuda_stream)
        for group in groups:
            assert 1 not in source.get_page_storage_snapshot(group).cache_levels

        restored = {role: torch.empty_like(values) for role, values in original.items()}
        selected = {role: torch.empty_like(values) for role, values in original.items()}
        high_ordinals = [
            i for i, slot in enumerate(source.get_base_page_indices(groups[1])) if slot >= 0
        ]
        assert high_ordinals == [0, 1, 6, 7]
        for ordinal in range(source_blocks):
            for role in original:
                copy_page(source, False, role, ordinal, restored[role][ordinal], write=False)
                copy_page(
                    source,
                    ordinal in high_ordinals,
                    role,
                    ordinal,
                    selected[role][ordinal],
                    write=False,
                )
        stream.synchronize()
        for role in original:
            assert not torch.equal(restored[role], original[role])
            assert torch.equal(selected[role][high_ordinals], original[role][high_ordinals])
            assert torch.equal(selected[role][2:6], restored[role][2:6])
        if skip_rope_quantization:
            assert torch.equal(restored["key"][..., :64], original["key"][..., :64])

        # Consume the selected sources with ordinary full-history Attention.
        # This validates source order and duplicate removal, not a model backend.
        def attend(values):
            key, value = [
                values[role].transpose(0, 1).reshape(1, heads, -1, width) for role in original
            ]
            query = original["key"][-1:, :, -1:, :]
            return torch.nn.functional.scaled_dot_product_attention(query, key, value)

        expected = {role: values.clone() for role, values in restored.items()}
        for role in expected:
            expected[role][high_ordinals] = original[role][high_ordinals]
        torch.testing.assert_close(attend(selected), attend(expected), rtol=0, atol=0)
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()
