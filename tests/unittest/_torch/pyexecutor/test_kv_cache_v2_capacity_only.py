# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _manager(*, is_draft: bool = False) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager.enable_block_reuse = False
    manager.kv_cache_map = {}
    manager._stream = MagicMock()
    manager.generation_capacity_only = False
    return manager


def _request(request_id: int, *, rewind: int = 0, complete: bool = False) -> MagicMock:
    request = MagicMock()
    request.py_request_id = request_id
    request.py_rewind_len = rewind
    request.py_num_accepted_draft_tokens = 0
    request.py_num_accepted_draft_tokens_indices = []
    request.max_beam_num_tokens = 201
    request.state = (
        LlmRequestState.GENERATION_COMPLETE if complete else LlmRequestState.GENERATION_IN_PROGRESS
    )
    return request


def _cache(capacity: int = 256) -> MagicMock:
    cache = MagicMock()
    cache.is_active = True
    cache.capacity = capacity
    cache.resize.return_value = True
    return cache


@pytest.mark.parametrize(
    ("capacity_only", "rewind", "complete", "active", "expected_resize"),
    [
        (False, 3, False, True, (253, 200)),
        (True, 3, False, True, (253, None)),
        (True, 0, True, True, (None, None)),
        (True, 3, False, False, None),
    ],
)
def test_update_resize_policy(
    capacity_only: bool,
    rewind: int,
    complete: bool,
    active: bool,
    expected_resize: tuple[int | None, int | None] | None,
) -> None:
    manager = _manager()
    request = _request(1, rewind=rewind, complete=complete)
    cache = _cache()
    cache.is_active = active
    manager.kv_cache_map[1] = cache
    manager.generation_capacity_only = capacity_only

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    if expected_resize is None:
        cache.resize.assert_not_called()
    else:
        cache.resize.assert_called_once_with(*expected_resize)


def test_target_capacity_only_does_not_change_draft_manager() -> None:
    request = _request(1, rewind=3)
    target = _manager(is_draft=False)
    target.generation_capacity_only = True
    target_cache = _cache()
    target.kv_cache_map[1] = target_cache
    draft = _manager(is_draft=True)
    draft_cache = _cache()
    draft.kv_cache_map[1] = draft_cache

    batch = SimpleNamespace(generation_requests=[request])
    target.update_resources(batch)
    draft.update_resources(batch)

    target_cache.resize.assert_called_once_with(253, None)
    draft_cache.resize.assert_called_once_with(253, 200)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real V2 CUDA pool")
@pytest.mark.parametrize(
    ("manager_kwargs", "is_draft", "expected_max_seq_len"),
    [
        ({}, False, 9_280),
        ({"reuse_generation_kv_capacity": True}, False, 32_768),
        ({"reuse_generation_kv_capacity": True}, True, 9_280),
    ],
)
def test_real_v2_separates_logical_length_from_reused_physical_capacity(
    manager_kwargs: dict,
    is_draft: bool,
    expected_max_seq_len: int,
) -> None:
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=9_280,
            enable_block_reuse=False,
            max_util_for_resume=1.0,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        tokens_per_block=64,
        max_seq_len=32_768,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=128,
        is_draft=is_draft,
        **manager_kwargs,
    )
    try:
        assert manager.max_seq_len == expected_max_seq_len
        assert manager.max_blocks_per_seq == 148
        assert manager.host_kv_cache_block_offsets.shape[-1] == 148
        assert manager.get_num_available_tokens(token_num_upper_bound=32_768) == 9_280
    finally:
        manager.shutdown()
