# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused tests for KVCM V2's cold-page codec integration.

These tests intentionally substitute storage/event objects. They validate the
KVCM transaction contract without requiring a real NVFP4 kernel: per-level
layout installation, Page-index batching, transform-before-publish ordering,
and fail-closed submission. Native SM100 round-trip tests remain separate.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL, CacheLevel, CacheTier
from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager


class _Codec:
    def __init__(self, strides=None, *, submit=True, batching=None):
        self.strides = strides or {0: 72 << 10}
        self.submit = submit
        self.batching = batching or {}
        self.calls = []

    def query_cold_page_bytes(self, layer_group_id):
        return self.strides.get(layer_group_id, 0)

    def get_batching_layer_group_id(self, layer_group_id):
        return self.batching.get(layer_group_id, layer_group_id)

    def encode(self, *args):
        self.calls.append(("encode", args))
        return self.submit

    def decode(self, *args):
        self.calls.append(("decode", args))
        return self.submit


class _SequencedCodec(_Codec):
    def __init__(self, results):
        super().__init__({0: 4096, 1: 4096})
        self.results = iter(results)

    def encode(self, *args):
        self.calls.append(("encode", args))
        return next(self.results)


class _Controller:
    def num_evictable_pages(self, _pg_idx):
        return 0


class _InitStorage:
    def __init__(self, slot_sizes, slots=8):
        self.slot_sizes = slot_sizes
        self.slots = slots
        self.ratio_list = [1.0]
        self.destroyed = False

    def get_num_free_slots(self, _pg_idx):
        return self.slots

    def num_slots(self, _pg_idx):
        return self.slots

    def destroy(self):
        self.destroyed = True


def _empty_storage_manager(*, variants=(0,)):
    manager = StorageManager.__new__(StorageManager)
    manager.__rawref__ = SimpleNamespace(is_valid=False)
    manager._cold_page_codec = None
    manager._life_cycle_grouping = [0 for _ in variants]
    manager._slot_desc_list = [
        SimpleNamespace(variants=tuple(SimpleNamespace(life_cycle_id=lc) for lc in variants))
    ]
    manager._min_slots = [1]
    manager._cache_tier_configs = (
        SimpleNamespace(tier=CacheTier.GPU_MEM, quota=1 << 20),
        SimpleNamespace(tier=CacheTier.HOST_MEM, quota=8 << 20),
    )
    gpu = SimpleNamespace(cache_tier=CacheTier.GPU_MEM)
    host_storage = _InitStorage([[128 << 10, 128 << 10]])
    host = SimpleNamespace(
        cache_tier=CacheTier.HOST_MEM,
        storage=host_storage,
        controller=_Controller(),
    )
    manager._levels = [gpu, host]
    return manager, host_storage


def test_install_codec_replaces_empty_host_layout_with_one_compact_pool():
    manager, old_host_storage = _empty_storage_manager()
    codec = _Codec()
    replacement = SimpleNamespace(cache_tier=CacheTier.HOST_MEM)

    with (
        patch.object(
            StorageManager,
            "_compute_slot_count_for_level",
            autospec=True,
            return_value=[113],
        ) as compute_slots,
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.CacheLevelManager",
            return_value=replacement,
        ) as create_level,
    ):
        manager.set_cold_page_codec(codec)

    assert old_host_storage.destroyed
    assert manager._cold_page_codec is codec
    cold_slot_sizes = create_level.call_args.args[3]
    assert cold_slot_sizes == [[72 << 10]]
    assert create_level.call_args.args[4] == [113]
    assert compute_slots.call_args.args[2] == cold_slot_sizes


def test_install_codec_rejects_mixed_cold_strides_in_one_pool_group():
    manager, old_host_storage = _empty_storage_manager(variants=(0, 1))

    with pytest.raises(RuntimeError, match="identical"):
        manager.set_cold_page_codec(_Codec({0: 72 << 10, 1: 80 << 10}))

    assert not old_host_storage.destroyed
    assert manager._cold_page_codec is None


class _Slot:
    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.ready_event = object()


class _Page(_Slot):
    def __init__(self, slot_id, life_cycle, cache_level):
        super().__init__(slot_id)
        self.life_cycle = life_cycle
        self.cache_level = cache_level
        self.node_ref = None
        self.scheduled_for_eviction = False

    def set_slot(self, slot):
        self.slot_id = slot.slot_id


class _PoolGroup:
    def __init__(self, *, num_pools, base, dst_ids=()):
        self.num_pools = num_pools
        self.num_free_slots = 1024
        self.base = base
        self.dst_ids = list(dst_ids)
        self.released = []

    def allocate_multiple(self, count):
        assert count == len(self.dst_ids)
        return [_Slot(slot_id) for slot_id in self.dst_ids]

    def slot_address(self, slot_id):
        return tuple(
            self.base + int(slot_id) * 4096 + pool * 0x100000 for pool in range(self.num_pools)
        )

    def release(self, slot):
        self.released.append(slot)


class _TempStream:
    last_finish = None

    def __init__(self, _events):
        self.finish = object()
        _TempStream.last_finish = self.finish

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get(self):
        return 0xCAFE

    def take_finish_event(self):
        return self.finish


def _migration_manager(codec, *, src_level, dst_level, src_pages, dst_ids, host_num_pools=1):
    manager = StorageManager.__new__(StorageManager)
    manager.__rawref__ = SimpleNamespace(is_valid=False)
    manager._cold_page_codec = codec
    manager._event_manager = None
    manager._life_cycle_grouping = [0, 0]
    manager._slot_desc_list = [SimpleNamespace(slot_size_list=[4096, 4096])]
    gpu_pg = _PoolGroup(
        num_pools=2,
        base=0x100000,
        dst_ids=dst_ids if dst_level == GPU_LEVEL else (),
    )
    host_pg = _PoolGroup(
        num_pools=host_num_pools,
        base=0x900000,
        dst_ids=dst_ids if dst_level != GPU_LEVEL else (),
    )
    manager._levels = [
        SimpleNamespace(
            cache_tier=CacheTier.GPU_MEM,
            storage=SimpleNamespace(_pool_groups=[gpu_pg], slot_size=lambda _pg_idx: [4096, 4096]),
        ),
        SimpleNamespace(
            cache_tier=CacheTier.HOST_MEM,
            storage=SimpleNamespace(
                _pool_groups=[host_pg],
                slot_size=lambda _pg_idx: [4096] * host_num_pools,
            ),
        ),
    ]
    return manager, gpu_pg, host_pg


def test_no_codec_preserves_existing_same_layout_batched_copy_path():
    """Compression-disabled migration remains the original per-Pool copy."""
    src_level = GPU_LEVEL
    dst_level = CacheLevel(1)
    page = _Page(3, 0, src_level)
    manager, gpu_pg, _ = _migration_manager(
        None,
        src_level=src_level,
        dst_level=dst_level,
        src_pages=[page],
        dst_ids=(5,),
        host_num_pools=2,
    )

    with (
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
            _TempStream,
        ),
        patch("tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.batched_copy") as copy,
    ):
        manager._batched_migrate(0, dst_level, src_level, [page], update_src=True)

    assert copy.call_count == 2
    assert [call.args[2] for call in copy.call_args_list] == [4096, 4096]
    assert page.slot_id == 5
    assert page.cache_level == dst_level
    assert len(gpu_pg.released) == 1


@pytest.mark.parametrize(
    "src_level,dst_level,method",
    [
        (GPU_LEVEL, CacheLevel(1), "encode"),
        (CacheLevel(1), GPU_LEVEL, "decode"),
    ],
)
def test_boundary_migration_batches_noncontiguous_indices_before_page_publish(
    src_level, dst_level, method
):
    codec = _Codec({0: 4096, 1: 4096})
    pages = [_Page(3, 0, src_level), _Page(11, 1, src_level)]
    manager, gpu_pg, host_pg = _migration_manager(
        codec, src_level=src_level, dst_level=dst_level, src_pages=pages, dst_ids=(5, 17)
    )

    with patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
        _TempStream,
    ):
        manager._batched_migrate(0, dst_level, src_level, pages, update_src=True)

    assert [call[0] for call in codec.calls] == [method, method]
    assert [call[1][0] for call in codec.calls] == [0, 1]
    assert all(call[1][-1] == 0xCAFE for call in codec.calls)
    assert [page.slot_id for page in pages] == [5, 17]
    assert [page.cache_level for page in pages] == [dst_level, dst_level]
    assert len(gpu_pg.released if src_level == GPU_LEVEL else host_pg.released) == 2


def test_boundary_migration_batches_pages_of_one_lifecycle_in_one_codec_call():
    codec = _Codec({0: 4096})
    pages = [_Page(3, 0, GPU_LEVEL), _Page(11, 0, GPU_LEVEL)]
    manager, _, _ = _migration_manager(
        codec,
        src_level=GPU_LEVEL,
        dst_level=CacheLevel(1),
        src_pages=pages,
        dst_ids=(5, 17),
    )

    with patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
        _TempStream,
    ):
        manager._batched_migrate(0, CacheLevel(1), GPU_LEVEL, pages, update_src=True)

    assert len(codec.calls) == 1
    _, args = codec.calls[0]
    assert args[2] == [5, 17]
    assert args[3] == [3, 11]


@pytest.mark.parametrize(
    "src_level,dst_level,method",
    [
        (GPU_LEVEL, CacheLevel(1), "encode"),
        (CacheLevel(1), GPU_LEVEL, "decode"),
    ],
)
def test_boundary_migration_batches_codec_equivalent_lifecycles_together(
    src_level, dst_level, method
):
    codec = _Codec({0: 4096, 1: 4096}, batching={1: 0})
    pages = [
        _Page(3, 0, src_level),
        _Page(11, 1, src_level),
        _Page(19, 0, src_level),
    ]
    manager, _, _ = _migration_manager(
        codec,
        src_level=src_level,
        dst_level=dst_level,
        src_pages=pages,
        dst_ids=(5, 17, 23),
    )

    with patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
        _TempStream,
    ):
        manager._batched_migrate(0, dst_level, src_level, pages, update_src=True)

    assert len(codec.calls) == 1
    operation, args = codec.calls[0]
    assert operation == method
    assert args[0] == 0
    assert args[2 if method == "encode" else 1] == [5, 17, 23]
    assert args[3] == [3, 11, 19]


@pytest.mark.parametrize(
    "batching,match",
    [
        ({0: 1}, "invalid cross-lifecycle"),
        ({1: -1}, "invalid cross-lifecycle"),
    ],
)
def test_boundary_migration_rejects_invalid_batching_representative(batching, match):
    codec = _Codec({0: 4096, 1: 4096}, batching=batching)
    life_cycle = next(iter(batching))
    page = _Page(3, life_cycle, GPU_LEVEL)
    manager, _, _ = _migration_manager(
        codec,
        src_level=GPU_LEVEL,
        dst_level=CacheLevel(1),
        src_pages=[page],
        dst_ids=(5,),
    )

    with (
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
            _TempStream,
        ),
        pytest.raises(RuntimeError, match=match),
    ):
        manager._batched_migrate(0, CacheLevel(1), GPU_LEVEL, [page], update_src=True)


def test_codec_submission_failure_does_not_publish_destination_mapping():
    codec = _Codec(submit=False)
    src_level = GPU_LEVEL
    dst_level = CacheLevel(1)
    page = _Page(3, 0, src_level)
    manager, gpu_pg, host_pg = _migration_manager(
        codec, src_level=src_level, dst_level=dst_level, src_pages=[page], dst_ids=(5,)
    )

    with (
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
            _TempStream,
        ),
        pytest.raises(RuntimeError, match="submission failed"),
    ):
        manager._batched_migrate(0, dst_level, src_level, [page], update_src=True)

    assert page.slot_id == 3
    assert page.cache_level == src_level
    assert not gpu_pg.released
    assert [slot.slot_id for slot in host_pg.released] == [5]


def test_partial_codec_submission_is_fenced_before_destination_rollback():
    codec = _SequencedCodec([True, False])
    src_level = GPU_LEVEL
    dst_level = CacheLevel(1)
    pages = [_Page(3, 0, src_level), _Page(11, 1, src_level)]
    manager, gpu_pg, host_pg = _migration_manager(
        codec, src_level=src_level, dst_level=dst_level, src_pages=pages, dst_ids=(5, 17)
    )

    with (
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
            _TempStream,
        ),
        pytest.raises(RuntimeError, match="submission failed"),
    ):
        manager._batched_migrate(0, dst_level, src_level, pages, update_src=True)

    assert [page.slot_id for page in pages] == [3, 11]
    assert [page.cache_level for page in pages] == [src_level, src_level]
    assert not gpu_pg.released
    assert [slot.slot_id for slot in host_pg.released] == [5, 17]
    assert all(page.ready_event is _TempStream.last_finish for page in pages)
    assert all(slot.ready_event is _TempStream.last_finish for slot in host_pg.released)
