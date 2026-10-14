# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local functional QA for the eager two-pool Attention path, not model accuracy."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.vanilla import (
    VanillaAttention,
    VanillaAttentionMetadata,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv


@pytest.mark.parametrize(
    "history_dtype,high_dtype",
    ((DataType.FP8, DataType.BF16), (DataType.FP8, DataType.FP8), (DataType.BF16, DataType.BF16)),
)
def test_two_pool_vanilla_prefill_and_decode(history_dtype, high_dtype):
    """Use native SWA pages and the actual backend for both writes and reads."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native KVCM")
    block, heads, width, length = 4, 2, 16, 20
    high_torch_dtype = binding_to_torch_dtype(high_dtype)

    class WindowManager(KVCacheManagerV2):
        def _build_cache_config(self, config):
            high_buffers = [
                kv.BufferConfig(
                    role=role,
                    size=block
                    * heads
                    * width
                    * torch.empty((), dtype=high_torch_dtype).element_size(),
                )
                for role in ("key", "value")
            ]
            return self._add_mixed_precision_windows(
                config, {0: high_buffers}, first_n=4, last_n=8, high_dtype=high_dtype
            )

    manager = WindowManager(
        KvCacheConfig(max_tokens=256, enable_block_reuse=True, pool_ratio=[0.75, 0.25]),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=width,
        tokens_per_block=block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=history_dtype,
    )
    manager.configure_mixed_precision_reuse([1], first_n=4, last_n=8)
    generator = torch.Generator(device="cuda").manual_seed(123)
    query, key, value = [
        torch.randn(length + 1, heads * width, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        for _ in range(3)
    ]
    attention = VanillaAttention(
        layer_idx=0,
        num_heads=heads,
        head_dim=width,
        num_kv_heads=heads,
        quant_config=QuantConfig(
            kv_cache_quant_algo="FP8" if history_dtype == DataType.FP8 else None
        ),
    )

    def forward(begin, end, *, context):
        metadata = VanillaAttentionMetadata(
            seq_lens=torch.tensor([end - begin], dtype=torch.int32),
            num_contexts=int(context),
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[begin]),
            max_num_requests=1,
            max_num_tokens=length + 1,
            kv_cache_manager=manager,
            request_ids=[0],
        )
        metadata.prepare()
        return attention.forward(query[begin:end], key[begin:end], value[begin:end], metadata)

    try:
        manager.add_dummy_requests([0], [length], is_gen=False)
        forward(0, length, context=True)
        cache = manager.kv_cache_map[0]
        assert cache.resize(None, length)
        cache.suspend()
        assert cache.resume(torch.cuda.current_stream().cuda_stream, is_decoding=True)
        assert cache.resize(length + 1)
        manager._restore_page_index_bufs(0, cache)
        result = forward(length, length + 1, context=False)

        history = manager.get_buffers(0)
        history_ids = manager.get_batch_cache_indices([0], 0)[0]
        high, (high_ids,), _ = manager.get_mixed_precision_window(0, [0])
        assert high_ids[0] >= 0 and high_ids[1:3] == [kv.BAD_PAGE_INDEX] * 2
        assert high_ids[-1] >= 0
        reference = []
        for role, original in enumerate((key, value)):
            original = original.view(-1, heads, width)
            low_values = original.to(history.dtype).to(torch.bfloat16)
            high_values = original.to(high_torch_dtype).to(torch.bfloat16)
            expected = low_values.clone()
            for position in range(length + 1):
                ordinal, offset = divmod(position, block)
                torch.testing.assert_close(
                    history[history_ids[ordinal], role, offset].to(torch.bfloat16),
                    low_values[position],
                    rtol=0,
                    atol=0,
                )
                if high_ids[ordinal] >= 0:
                    expected[position] = high_values[position]
                    torch.testing.assert_close(
                        high[high_ids[ordinal], role, offset].to(torch.bfloat16),
                        high_values[position],
                        rtol=0,
                        atol=0,
                    )
            reference.append(expected.transpose(0, 1)[None])
        expected_output = torch.nn.functional.scaled_dot_product_attention(
            query[-1].view(1, heads, 1, width), *reference
        ).reshape_as(result)
        torch.testing.assert_close(result, expected_output)
    finally:
        manager.shutdown()


def test_high_gap_is_filled_before_attention_without_overwriting_saved_high():
    """A newly allocated high page starts invalid; filling it is not high-value reuse."""
    attention = VanillaAttention(layer_idx=0, num_heads=1, head_dim=2, num_kv_heads=1)
    low = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 1, 2)
    high = torch.full_like(low, float("nan"))
    high[0] = 40
    query = torch.ones(1, 2)
    new_key, new_value = torch.tensor([[5.0, 6.0]]), torch.tensor([[7.0, 8.0]])
    result = attention._single_request_forward(
        query,
        new_key,
        new_value,
        PredefinedAttentionMask.CAUSAL,
        low,
        3,
        [0, 1],
        0,
        None,
        high_cache=high,
        high_block_ids=[0, 1],
        missing_high_ranges=((2, 3),),
    )
    assert torch.isfinite(result).all()
    torch.testing.assert_close(high[0], torch.full_like(high[0], 40))
    torch.testing.assert_close(high[1, :, 0], low[1, :, 0])
    torch.testing.assert_close(high[1, 0, 1, 0], new_key[0])
    torch.testing.assert_close(high[1, 1, 1, 0], new_value[0])
