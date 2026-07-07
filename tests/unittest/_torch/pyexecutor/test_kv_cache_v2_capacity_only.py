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
from unittest.mock import MagicMock

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig


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


def test_capacity_only_is_manager_scoped() -> None:
    manager = _manager()
    first = _request(1, rewind=3)
    second = _request(2, rewind=5)
    first_cache = _cache()
    second_cache = _cache()
    manager.kv_cache_map = {1: first_cache, 2: second_cache}
    manager.generation_capacity_only = True

    manager.update_resources(SimpleNamespace(generation_requests=[first, second]))

    first_cache.resize.assert_called_once_with(253, None)
    second_cache.resize.assert_called_once_with(251, None)


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


def test_llm_request_has_no_compression_marker() -> None:
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )

    assert "py_kv_cache_generation_capacity_only" not in request.__dict__
    assert "py_kv_cache_compaction" not in request.__dict__


def test_default_manager_preserves_native_history_update() -> None:
    manager = _manager()
    request = _request(1, rewind=3)
    cache = _cache()
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_completion_preserves_history() -> None:
    manager = _manager()
    request = _request(1, complete=True)
    cache = _cache()
    manager.kv_cache_map[1] = cache
    manager.generation_capacity_only = True

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(None, None)


def test_suspended_cache_keeps_native_v2_behavior() -> None:
    manager = _manager()
    manager.generation_capacity_only = True
    request = _request(1, rewind=3)
    cache = _cache()
    cache.is_active = False
    manager.kv_cache_map[1] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_not_called()
