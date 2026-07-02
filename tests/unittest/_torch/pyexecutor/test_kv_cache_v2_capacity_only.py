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
from unittest.mock import MagicMock, call

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, LayerId, SsmLayerConfig


def _full_attention_layer(layer_id: int = 0) -> AttentionLayerConfig:
    return AttentionLayerConfig(layer_id=LayerId(layer_id), buffers=[])


def _manager(
    *,
    enable_block_reuse: bool = False,
    layers: list[AttentionLayerConfig | SsmLayerConfig] | None = None,
) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_block_reuse = enable_block_reuse
    manager.max_attention_window_vec = [None]
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=list(layers if layers is not None else [_full_attention_layer()])
    )
    manager.kv_cache_map = {}
    manager._decode_capacity_only_requests = set()
    manager._pending_compacted_capacities = {}
    manager._allocated_draft_lens = {}
    manager._stream = MagicMock()
    manager.impl = MagicMock()
    manager._early_freed_index_requests = set()
    manager.index_mapper = MagicMock()
    return manager


def _request(
    request_id: int = 1,
    *,
    max_beam_tokens: int = 200,
    rewind: int = 0,
    state: LlmRequestState = LlmRequestState.GENERATION_IN_PROGRESS,
) -> MagicMock:
    request = MagicMock()
    request.py_request_id = request_id
    request.max_beam_num_tokens = max_beam_tokens
    request.py_rewind_len = rewind
    request.state = state
    return request


def _cache(capacity: int = 4096, history_length: int = 128) -> MagicMock:
    cache = MagicMock()
    cache.is_active = True
    cache.capacity = capacity
    cache.history_length = history_length

    def resize(new_capacity: int | None, _history_length: int | None) -> bool:
        if new_capacity is not None:
            cache.capacity = new_capacity
        return True

    cache.resize.side_effect = resize
    return cache


def _generation_batch(*requests: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(generation_requests=list(requests))


def test_default_generation_update_is_unchanged() -> None:
    manager = _manager()
    request = _request(rewind=3)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(_generation_batch(request))

    cache.resize.assert_called_once_with(4093, 199)


def test_capacity_only_mode_keeps_context_history_update() -> None:
    manager = _manager()
    request = _request()
    request.context_current_position = 128
    cache = _cache(capacity=128)
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)
    batch = SimpleNamespace(context_requests=[request])

    manager.update_context_resources(batch)

    cache.resize.assert_called_once_with(None, 128)


def test_capacity_only_generation_does_not_advance_history() -> None:
    manager = _manager()
    request = _request(rewind=3)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)

    manager.update_resources(_generation_batch(request))

    cache.resize.assert_called_once_with(4093, None)


def test_compacted_capacity_is_consumed_once() -> None:
    manager = _manager()
    request = _request(rewind=2)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150)

    manager.update_resources(_generation_batch(request))
    request.py_rewind_len = 0
    manager.update_resources(_generation_batch(request))

    assert cache.resize.call_args_list == [call(148, None), call(148, None)]
    assert request.py_request_id not in manager._pending_compacted_capacities


def test_compacted_capacity_waits_for_event() -> None:
    manager = _manager()
    request = _request()
    cache = _cache()
    event = MagicMock()
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150, event)

    manager.update_resources(_generation_batch(request))

    manager._stream.wait_event.assert_called_once_with(event)
    cache.resize.assert_called_once_with(150, None)


def test_capacity_only_completion_preserves_history() -> None:
    manager = _manager()
    request = _request(state=LlmRequestState.GENERATION_COMPLETE)
    cache = _cache()
    event = MagicMock()
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150, event)

    manager.update_resources(_generation_batch(request))

    cache.resize.assert_called_once_with(None, None)
    manager._stream.wait_event.assert_called_once_with(event)
    assert request.py_request_id not in manager._pending_compacted_capacities


def test_capacity_only_resize_failure_is_not_hidden() -> None:
    manager = _manager()
    request = _request()
    cache = _cache()
    cache.resize.side_effect = None
    cache.resize.return_value = False
    manager.kv_cache_map[request.py_request_id] = cache
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150)

    with pytest.raises(ValueError, match="preserving its finalized history"):
        manager.update_resources(_generation_batch(request))

    assert request.py_request_id in manager._pending_compacted_capacities


def test_free_resources_clears_capacity_only_state() -> None:
    manager = _manager()
    request = _request()
    event = MagicMock()
    manager.kv_cache_map[request.py_request_id] = _cache()
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150, event)

    manager.free_resources(request)

    manager._stream.wait_event.assert_called_once_with(event)
    assert request.py_request_id not in manager._decode_capacity_only_requests
    assert request.py_request_id not in manager._pending_compacted_capacities


def test_capacity_only_rejects_block_reuse() -> None:
    manager = _manager(enable_block_reuse=True)

    with pytest.raises(ValueError, match="block reuse"):
        manager.enable_decode_capacity_only(1)


def test_capacity_only_rejects_swa() -> None:
    swa_layer = AttentionLayerConfig(layer_id=LayerId(0), buffers=[], sliding_window_size=128)
    manager = _manager(layers=[swa_layer])

    with pytest.raises(ValueError, match="full-attention"):
        manager.enable_decode_capacity_only(1)


def test_capacity_only_rejects_global_vswa_config() -> None:
    manager = _manager()
    manager.max_attention_window_vec = [None, 128]

    with pytest.raises(ValueError, match="full-attention"):
        manager.enable_decode_capacity_only(1)


def test_capacity_only_rejects_ssm() -> None:
    manager = _manager(layers=[SsmLayerConfig(layer_id=LayerId(0), buffers=[])])

    with pytest.raises(ValueError, match="full-attention"):
        manager.enable_decode_capacity_only(1)


def test_compacted_capacity_requires_opt_in() -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="not enabled"):
        manager.set_compacted_capacity(1, 150)


def test_compacted_capacity_cannot_cross_finalized_history() -> None:
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(history_length=160)
    manager.enable_decode_capacity_only(request.py_request_id)

    with pytest.raises(ValueError, match="below finalized history"):
        manager.set_compacted_capacity(request.py_request_id, 150)


def test_compacted_capacity_rejects_pending_target_overwrite() -> None:
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache()
    manager.enable_decode_capacity_only(request.py_request_id)
    manager.set_compacted_capacity(request.py_request_id, 150)

    with pytest.raises(ValueError, match="already has a pending"):
        manager.set_compacted_capacity(request.py_request_id, 140)
