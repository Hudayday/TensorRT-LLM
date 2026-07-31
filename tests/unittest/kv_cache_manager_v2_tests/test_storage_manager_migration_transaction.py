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

"""Fault-injection coverage for StorageManager migration ownership.

The tests are intentionally import-light.  They model Slot ownership and CUDA
fences without allocating GPU memory, so transaction ordering is checked even
on CPU-only presubmit workers.
"""

from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CacheLevel
    from kv_cache_manager_v2 import _storage_manager as storage_manager_module
    from kv_cache_manager_v2._common import CacheTier
    from kv_cache_manager_v2._storage._core import PoolGroupIndex, PoolIndex
    from kv_cache_manager_v2._storage_manager import StorageManager
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _storage_manager as storage_manager_module
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import CacheTier
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import PoolGroupIndex, PoolIndex
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager


_SRC_LEVEL = CacheLevel(0)
_DST_LEVEL = CacheLevel(1)
_POOL_GROUP = PoolGroupIndex(0)
_NULL_EVENT = object()


class _InjectedFailure(RuntimeError):
    pass


class _FakeSlot:
    def __init__(self, slot_id: int, ready_event: object = _NULL_EVENT) -> None:
        self._slot_id = slot_id
        self.ready_event = ready_event

    @property
    def slot_id(self) -> int:
        assert self._slot_id is not None
        return self._slot_id

    @property
    def has_valid_slot(self) -> bool:
        return self._slot_id is not None

    def set_slot(self, slot: "_FakeSlot") -> None:
        assert not self.has_valid_slot
        self._slot_id = slot.slot_id
        self.ready_event = slot.ready_event
        slot._slot_id = None
        slot.ready_event = _NULL_EVENT


class _FakePage(_FakeSlot):
    def __init__(self, slot_id: int) -> None:
        super().__init__(slot_id)
        self.cache_level = _SRC_LEVEL
        self.node_ref = None

    @property
    def scheduled_for_eviction(self) -> bool:
        return self.node_ref is not None


class _FakePoolGroup:
    def __init__(self, first_slot_id: int, capacity: int) -> None:
        self._next_slot_id = first_slot_id
        self._capacity = capacity
        self._outstanding = 0
        self.released: list[tuple[int, object]] = []

    @property
    def num_free_slots(self) -> int:
        return self._capacity - self._outstanding

    def allocate_multiple(self, num_slots: int) -> list[_FakeSlot]:
        assert num_slots <= self.num_free_slots
        slots = [_FakeSlot(self._next_slot_id + offset) for offset in range(num_slots)]
        self._next_slot_id += num_slots
        self._outstanding += num_slots
        return slots

    def release(self, slot: _FakeSlot) -> None:
        # A second release of a Slot whose ownership already moved to a Page
        # is the regression these tests are meant to expose.
        assert slot.has_valid_slot, "double release"
        self.released.append((slot.slot_id, slot.ready_event))
        slot._slot_id = None
        slot.ready_event = _NULL_EVENT
        self._outstanding -= 1

    def slot_address(self, slot_id: int) -> tuple[int]:
        return (slot_id * 64,)


class _FakeTemporaryCudaStream:
    finish_event = object()

    def __init__(self, prior_events: set[object]) -> None:
        self.prior_events = prior_events

    def __enter__(self) -> "_FakeTemporaryCudaStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def get(self) -> int:
        return 0

    def take_finish_event(self) -> object:
        return self.finish_event


class _FakeStorageManager:
    def __init__(self, num_pages: int) -> None:
        self.src_pool = _FakePoolGroup(0, num_pages)
        self.dst_pool = _FakePoolGroup(100, num_pages)
        self._levels = [
            SimpleNamespace(cache_tier=CacheTier.GPU_MEM),
            SimpleNamespace(cache_tier=CacheTier.HOST_MEM),
        ]
        self._event_manager = None
        self._offload_compression_hook = None
        self._onboard_decompression_hook = None

    def num_pools(self, pool_group_index: PoolGroupIndex) -> PoolIndex:
        assert pool_group_index == _POOL_GROUP
        return PoolIndex(1)

    def _pool_group(
        self,
        cache_level: CacheLevel,
        pool_group_index: PoolGroupIndex,
    ) -> _FakePoolGroup:
        assert pool_group_index == _POOL_GROUP
        return self.src_pool if cache_level == _SRC_LEVEL else self.dst_pool

    def slot_size(
        self,
        pool_group_index: PoolGroupIndex,
        cache_level: CacheLevel | None = None,
    ) -> list[int]:
        assert pool_group_index == _POOL_GROUP
        assert cache_level in (_SRC_LEVEL, _DST_LEVEL)
        return [64]

    def exclude_from_eviction(self, page: _FakePage) -> None:
        raise AssertionError("test Pages are not in an eviction controller")

    def schedule_for_eviction(self, page: _FakePage) -> None:
        raise AssertionError("test Pages are not in an eviction controller")

    def _emit_cache_level_updated_event(self, *args, **kwargs) -> None:
        raise AssertionError("event emission is disabled by default")


def _migrate(
    manager: _FakeStorageManager,
    pages: list[_FakePage],
    recorder=None,
) -> None:
    StorageManager._batched_migrate(
        manager,
        _POOL_GROUP,
        _DST_LEVEL,
        _SRC_LEVEL,
        pages,
        update_src=True,
        migration_recorder=recorder,
    )


def test_recorder_failure_fences_source_and_destination_before_rollback() -> None:
    pages = [_FakePage(0), _FakePage(1)]
    manager = _FakeStorageManager(len(pages))
    finish_event = _FakeTemporaryCudaStream.finish_event

    def fail_after_copy(src_pages, dst_slots, src_level, dst_level) -> None:
        assert src_level == _SRC_LEVEL
        assert dst_level == _DST_LEVEL
        assert all(page.ready_event is finish_event for page in src_pages)
        assert all(slot.ready_event is finish_event for slot in dst_slots)
        raise _InjectedFailure("recorder fault")

    with (
        patch.object(
            storage_manager_module,
            "TemporaryCudaStream",
            _FakeTemporaryCudaStream,
        ),
        patch.object(storage_manager_module, "batched_copy"),
        pytest.raises(_InjectedFailure, match="recorder fault"),
    ):
        _migrate(manager, pages, fail_after_copy)

    assert [page.slot_id for page in pages] == [0, 1]
    assert all(page.cache_level == _SRC_LEVEL for page in pages)
    assert all(page.ready_event is finish_event for page in pages)
    assert manager.src_pool.released == []
    assert manager.dst_pool.released == [
        (100, finish_event),
        (101, finish_event),
    ]


def test_observer_failure_cannot_abort_a_committed_migration() -> None:
    pages = [_FakePage(0), _FakePage(1)]
    manager = _FakeStorageManager(len(pages))
    manager._event_manager = object()
    finish_event = _FakeTemporaryCudaStream.finish_event

    def fail_observer(*args, **kwargs) -> None:
        raise _InjectedFailure("observer fault")

    manager._emit_cache_level_updated_event = fail_observer
    with (
        patch.object(
            storage_manager_module,
            "TemporaryCudaStream",
            _FakeTemporaryCudaStream,
        ),
        patch.object(storage_manager_module, "batched_copy"),
        pytest.warns(RuntimeWarning, match="event failure after migration commit"),
    ):
        _migrate(manager, pages)

    # Observer failure is reported but deliberately not propagated.  The
    # caller can therefore finish its post-migration work, including attaching
    # these Pages to the destination eviction controller.
    assert [page.slot_id for page in pages] == [100, 101]
    assert all(page.cache_level == _DST_LEVEL for page in pages)
    assert all(page.ready_event is finish_event for page in pages)
    assert manager.src_pool.released == [
        (0, finish_event),
        (1, finish_event),
    ]
    assert manager.dst_pool.released == []


class _ForceEvictionController:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = list(pages)
        self.rescheduled: list[tuple[_FakePage, bool]] = []

    def evict(self, min_num_pages) -> list[list[_FakePage]]:
        assert min_num_pages == [len(self.pages)]
        victims = list(self.pages)
        self.pages.clear()
        return [victims]

    def schedule_for_eviction(self, page: _FakePage, *, evict_first: bool = False) -> None:
        page.node_ref = object()
        self.rescheduled.append((page, evict_first))


class _ForceEvictionManager:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.controller = _ForceEvictionController(pages)
        self._levels = [
            SimpleNamespace(controller=self.controller),
            SimpleNamespace(controller=object()),
        ]
        self.num_cache_levels = CacheLevel(2)
        self.num_pool_groups = PoolGroupIndex(1)

    @staticmethod
    def is_evictable(page: _FakePage) -> bool:
        return True

    @staticmethod
    def _prepare_free_slots(*args, **kwargs) -> None:
        raise _InjectedFailure("compression rejected")


def test_force_evict_restores_unmigrated_source_victims_on_failure() -> None:
    pages = [_FakePage(0), _FakePage(1)]
    manager = _ForceEvictionManager(pages)

    with pytest.raises(_InjectedFailure, match="compression rejected"):
        StorageManager.force_evict(manager, _SRC_LEVEL, [len(pages)])

    # Reverse insertion preserves the original oldest-first order when the
    # controller's evict-first operation prepends each Page.
    assert manager.controller.rescheduled == [
        (pages[1], True),
        (pages[0], True),
    ]
    assert all(page.cache_level == _SRC_LEVEL for page in pages)
    assert all(page.node_ref is not None for page in pages)
