# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Python-selected layout through the real native cold-page kernels."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    PageIndexMode,
)


@pytest.mark.parametrize(
    "dtype,runtime_dtype",
    [
        (torch.float16, DataType.HALF),
        (torch.bfloat16, DataType.BF16),
        (torch.float8_e4m3fn, DataType.FP8),
    ],
)
@pytest.mark.parametrize("keep_layer", [False, True])
def test_selected_layout_round_trip_on_non_default_stream(dtype, runtime_dtype, keep_layer):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 cold-page kernels require Blackwell or newer")

    heads, tokens, width, valid = 2, 64, 256, 61
    element_bytes = torch.empty((), dtype=dtype).element_size()
    raw_bytes = heads * tokens * width * element_bytes
    slot_bytes = raw_bytes + 32
    model = PretrainedConfig(
        hidden_size=2048,
        num_attention_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=4096,
        head_dim=width,
        partial_rotary_factor=0.25,
    )
    model.model_type = "qwen3_5"
    provider = Nvfp4ColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(
            skip_rope_quantization=True,
            selective_compression=dict(
                keep_first_tokens=3,
                keep_last_tokens=2,
                keep_layers=[3] if keep_layer else [],
                keep_token_ranges=[(17, 20)],
            ),
        ),
        pretrained_config=model,
    )
    config = SimpleNamespace(
        tokens_per_block=tokens,
        layers=[
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role=role, size=raw_bytes) for role in ("key", "value")],
            )
        ],
    )
    state = provider.build_codec_state(
        config,
        runtime_dtype=runtime_dtype,
        pp_layers=[3],
        num_kv_heads_per_layer=[heads],
        head_dim_per_layer=[width],
    )
    pools = {
        role: torch.full((4, slot_bytes), 0xA5, dtype=torch.uint8, device="cuda")
        for role in ("key", "value")
    }
    lifecycle = SimpleNamespace(
        layers={
            0: {
                role: SimpleNamespace(
                    raw_base=pool.data_ptr(), raw_slot_bytes=slot_bytes, raw_bytes=raw_bytes
                )
                for role, pool in pools.items()
            }
        }
    )
    provider.configure(state, [lifecycle])
    # Exact request-local raw intervals resolved by the common native policy.
    raw_tokens = [(0, 3), (17, 20), (59, 61)]
    capacity = 2 * raw_bytes
    provider.prepare_selected_cold_page(state, 0, 0, valid, raw_tokens, capacity)
    cold = torch.full((4, capacity), 0xCD, dtype=torch.uint8, pin_memory=True)
    encode_indices = torch.tensor([[3, 2]], dtype=torch.int32)
    decode_indices = torch.tensor([[1, 3]], dtype=torch.int32)
    originals = {}
    generator = torch.Generator().manual_seed(19)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for role, pool in pools.items():
            values = torch.randn((heads, tokens, width), generator=generator).to(dtype)
            originals[role] = values
            pool[2, :raw_bytes].copy_(values.view(torch.uint8).reshape(-1))
        provider.encode_selected_cold_pages(
            state, 0, 0, cold.data_ptr(), encode_indices.data_ptr(), 1, stream.cuda_stream
        )
        provider.decode_selected_cold_pages(
            state, 0, 0, cold.data_ptr(), decode_indices.data_ptr(), 1, stream.cuda_stream
        )
    stream.synchronize()

    for role, pool in pools.items():
        restored = pool[1, :raw_bytes].cpu().view(dtype).reshape(heads, tokens, width)
        expected = originals[role]
        protected = torch.zeros((heads, tokens, width), dtype=torch.bool)
        if keep_layer:
            protected[:, :valid] = True
        else:
            for begin, end in raw_tokens:
                protected[:, begin:end] = True
            if role == "key":
                protected[:, :valid, :64] = True
        # Compare storage bytes, including BF16 and FP8 values, without tolerances.
        raw_mask = protected.unsqueeze(-1).expand(-1, -1, -1, element_bytes)
        shape = (heads, tokens, width, element_bytes)
        assert torch.equal(
            restored.view(torch.uint8).reshape(shape)[raw_mask],
            expected.view(torch.uint8).reshape(shape)[raw_mask],
        )
        assert not torch.count_nonzero(restored[:, valid:].float())
        if not keep_layer:
            quantized = ~protected[:, :valid]
            actual = restored[:, :valid].float()[quantized]
            original = expected[:, :valid].float()[quantized]
            assert torch.isfinite(actual).all()
            assert torch.count_nonzero(actual != original) > 0
            assert (actual - original).square().mean() < 0.05
        assert torch.all(pool[[0, 3]] == 0xA5)
        assert torch.all(pool[:, raw_bytes:] == 0xA5)
        assert torch.equal(pool[2, :raw_bytes].cpu(), expected.view(torch.uint8).reshape(-1))
    assert torch.all(cold[:3] == 0xCD)


def test_native_manager_pressure_round_trip_composes_all_controls():
    """Real allocation pressure exercises native policy -> Python codec -> CUDA -> reuse."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 cold-page kernels require Blackwell or newer")

    heads, page_tokens, width, length = 2, 64, 256, 189
    buffer_bytes = heads * page_tokens * width * 2
    model = PretrainedConfig(
        hidden_size=2048,
        num_attention_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=4096,
        head_dim=width,
        partial_rotary_factor=0.25,
    )
    model.model_type = "qwen3_5"
    provider = Nvfp4ColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(
            skip_rope_quantization=True,
            selective_compression=dict(
                keep_first_tokens=100,
                keep_last_tokens=5,
                keep_layers=[1],
                keep_token_ranges=[(140, 145)],
            ),
        ),
        pretrained_config=model,
    )
    config = KVCacheManagerConfig(
        tokens_per_block=page_tokens,
        cache_tiers=[
            GpuCacheTierConfig(quota=3 * 4 * buffer_bytes),
            HostCacheTierConfig(quota=16 * 4 * buffer_bytes),
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=layer,
                buffers=[BufferConfig(role=role, size=buffer_bytes) for role in ("key", "value")],
            )
            for layer in range(2)
        ],
        max_util_for_resume=1.0,
    )
    codec = provider.create_cold_page_codec(
        config,
        runtime_dtype=DataType.BF16,
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[heads, heads],
        head_dim_per_layer=[width, width],
    )
    # The native allocator expects the calling thread to have an active CUDA context.
    stream = torch.cuda.Stream()
    manager = KVCacheManager(config, cold_page_codec=codec)
    hot_stats = manager.get_storage_statistics()
    assert len(hot_stats) == 1
    assert hot_stats[0].total >= 3
    requests = []

    def pages(request, layer, role):
        group = manager.get_layer_group_id(layer)
        indices = manager.get_page_index_converter(layer, role)(
            request.get_base_page_indices(group), PageIndexMode.SHARED
        )
        base = manager.get_mem_pool_base_address(layer, role, PageIndexMode.SHARED)
        stride = manager.get_page_stride(layer, role)
        return [
            convert_to_torch_tensor(
                TensorWrapper(base + index * stride, torch.bfloat16, (heads, page_tokens, width))
            )
            for index in indices
        ]

    try:
        first = manager.create_kv_cache()
        requests.append(first)
        originals = {}
        with torch.cuda.stream(stream):
            assert first.resume(stream.cuda_stream)
            assert first.resize(length)
            generator = torch.Generator().manual_seed(31)
            for layer in range(2):
                for role in ("key", "value"):
                    values = torch.randn(
                        (heads, 3 * page_tokens, width), generator=generator
                    ).bfloat16()
                    originals[layer, role] = values
                    for index, page in enumerate(pages(first, layer, role)):
                        page.copy_(values[:, index * page_tokens : (index + 1) * page_tokens])
            first.commit(list(range(length)), is_end=True)
            first.suspend()
            manager.get_and_reset_iteration_stats()
            second = manager.create_kv_cache()
            requests.append(second)
            assert second.resume(stream.cuda_stream)
            # Physical allocation granularity can round the requested quota up.
            # Consume its actual capacity to force all three held pages off GPU.
            assert second.resize(hot_stats[0].total * page_tokens)
            offloaded = manager.get_and_reset_iteration_stats()
            assert sum(stats.iter_offload_blocks for stats in offloaded.values()) == 3
            assert sum(stats.iter_offload_bytes for stats in offloaded.values()) > 0
            second.close()
            assert first.resume(stream.cuda_stream)
            onloaded = manager.get_and_reset_iteration_stats()
            assert sum(stats.iter_onboard_blocks for stats in onloaded.values()) == 3
            restored = {
                (layer, role): torch.cat(pages(first, layer, role), dim=1).cpu()
                for layer in range(2)
                for role in ("key", "value")
            }
        stream.synchronize()
        for (layer, role), actual in restored.items():
            expected = originals[layer, role]
            protected = torch.zeros((heads, length, width), dtype=torch.bool)
            protected[:, :100] = True
            protected[:, -5:] = True
            protected[:, 140:145] = True
            if layer == 1:
                protected[:] = True
            elif role == "key":
                protected[:, :, :64] = True
            assert torch.equal(actual[:, :length][protected], expected[:, :length][protected])
            if layer == 0:
                delta = (
                    actual[:, :length].float()[~protected]
                    - expected[:, :length].float()[~protected]
                )
                assert torch.count_nonzero(delta) > 0
                assert delta.square().mean() < 0.05
        # The original request is reusable, but a new suffix inside the lossy middle is not.
        assert manager.probe_reuse(input_tokens=list(range(length))) == length
        assert manager.probe_reuse(input_tokens=list(range(110))) == 105
    finally:
        for request in requests:
            request.close()
        manager.clear_reusable_blocks()
        manager.shutdown()
