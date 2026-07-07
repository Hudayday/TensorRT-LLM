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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import CacheTypeCpp, KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType


def _manager(
    *, is_draft: bool = False, kv_cache_type: CacheTypeCpp = CacheTypeCpp.SELF
) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager.kv_cache_type = kv_cache_type
    manager.enable_block_reuse = False
    manager.kv_cache_map = {}
    return manager


def _request(request_id: int, *, rewind: int = 0, complete: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        py_rewind_len=rewind,
        py_num_accepted_draft_tokens=0,
        py_num_accepted_draft_tokens_indices=[],
        max_beam_num_tokens=201,
        py_kv_cache_generation_capacity_only=False,
        state=(
            LlmRequestState.GENERATION_COMPLETE
            if complete
            else LlmRequestState.GENERATION_IN_PROGRESS
        ),
    )


def _cache(capacity: int = 256) -> MagicMock:
    cache = MagicMock()
    cache.is_active = True
    cache.capacity = capacity
    cache.resize.return_value = True
    return cache


def test_capacity_only_defaults_to_false() -> None:
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )

    assert request.py_kv_cache_generation_capacity_only is False


def test_default_generation_resize_updates_capacity_and_history() -> None:
    manager = _manager()
    request = _request(1, rewind=3)
    cache = _cache()
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_is_request_scoped() -> None:
    manager = _manager()
    regular = _request(1, rewind=3)
    capacity_only = _request(2, rewind=5)
    capacity_only.py_kv_cache_generation_capacity_only = True
    regular_cache = _cache()
    capacity_only_cache = _cache()
    manager.kv_cache_map = {1: regular_cache, 2: capacity_only_cache}

    manager.update_resources(SimpleNamespace(generation_requests=[regular, capacity_only]))

    regular_cache.resize.assert_called_once_with(253, 200)
    capacity_only_cache.resize.assert_called_once_with(251, None)


def test_capacity_only_policy_applies_only_to_target_kvcm() -> None:
    request = _request(1, rewind=3)
    request.py_kv_cache_generation_capacity_only = True
    scheduled_batch = SimpleNamespace(generation_requests=[request])
    draft_manager = _manager(is_draft=True)
    target_manager = _manager()
    draft_cache = _cache()
    target_cache = _cache()
    draft_manager.kv_cache_map[1] = draft_cache
    target_manager.kv_cache_map[1] = target_cache

    draft_manager.update_resources(scheduled_batch)
    target_manager.update_resources(scheduled_batch)

    draft_cache.resize.assert_called_once_with(253, 200)
    target_cache.resize.assert_called_once_with(253, None)
    assert request.py_kv_cache_generation_capacity_only is True


def test_capacity_only_policy_does_not_apply_to_cross_attention() -> None:
    manager = _manager(kv_cache_type=CacheTypeCpp.CROSS)
    request = _request(1, rewind=3)
    request.py_kv_cache_generation_capacity_only = True
    cache = _cache()
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_policy_does_not_apply_to_self_key_only() -> None:
    manager = _manager(kv_cache_type=CacheTypeCpp.SELFKONLY)
    request = _request(1, rewind=3)
    request.py_kv_cache_generation_capacity_only = True
    cache = _cache()
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_completion_preserves_target_history_only() -> None:
    request = _request(1, complete=True)
    request.py_kv_cache_generation_capacity_only = True
    scheduled_batch = SimpleNamespace(generation_requests=[request])
    draft_manager = _manager(is_draft=True)
    target_manager = _manager()
    draft_cache = _cache()
    target_cache = _cache()
    draft_manager.kv_cache_map[1] = draft_cache
    target_manager.kv_cache_map[1] = target_cache

    draft_manager.update_resources(scheduled_batch)
    target_manager.update_resources(scheduled_batch)

    draft_cache.resize.assert_called_once_with(None, 200)
    target_cache.resize.assert_called_once_with(None, None)


def test_failed_capacity_only_resize_raises() -> None:
    manager = _manager()
    request = _request(1)
    request.py_kv_cache_generation_capacity_only = True
    cache = _cache(capacity=256)
    cache.resize.return_value = False
    manager.kv_cache_map[1] = cache

    with pytest.raises(ValueError, match="Failed to resize KV cache"):
        manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(256, None)


def test_resource_manager_passes_attention_metadata_to_target_v2() -> None:
    manager = _manager()
    request = _request(1, rewind=3)
    request.py_num_accepted_draft_tokens = 1
    request.py_num_accepted_draft_tokens_indices = [0]
    cache = _cache()
    manager.kv_cache_map[1] = cache
    scheduled_batch = SimpleNamespace(generation_requests=[request])
    attn_metadata = object()
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})

    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2._update_kv_cache_draft_token_location"
    ) as update_draft_location:
        resource_manager.update_resources(scheduled_batch, attn_metadata, 2.0)

    update_draft_location.assert_called_once_with(manager, scheduled_batch, attn_metadata, 2.0)
    cache.resize.assert_called_once_with(253, 200)


def test_resource_manager_scopes_target_metadata_to_target_kvcm() -> None:
    scheduled_batch = object()
    attn_metadata = object()
    target_manager = MagicMock()
    draft_manager = MagicMock()
    compression_manager = MagicMock()
    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: target_manager,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft_manager,
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression_manager,
        }
    )

    resource_manager.update_resources(scheduled_batch, attn_metadata, 2.0)

    target_manager.update_resources.assert_called_once_with(scheduled_batch, attn_metadata, 2.0)
    draft_manager.update_resources.assert_called_once_with(scheduled_batch)
    compression_manager.update_resources.assert_called_once_with(scheduled_batch)
