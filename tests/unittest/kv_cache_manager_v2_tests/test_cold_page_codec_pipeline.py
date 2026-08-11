# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused tests for KVCM V2's cold-page codec integration.

The tests construct ``StorageManager`` through its production constructor and
install codecs through ``set_cold_page_codec``. Only physical allocation,
CUDA-stream execution, and copy submission are replaced: those boundaries need
GPU or filesystem resources and are covered by native/E2E tests separately.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
    GPU_LEVEL,
    CacheLevel,
    CacheTier,
    LayerId,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._config import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManagerConfig,
    SsmLayerConfig,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
    LifeCycleId,
    LifeCycleRegistry,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import create_storage_config
from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import PoolGroupIndex, SlotId
from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import (
    CacheLevelManager,
    StorageManager,
)


class _Codec:
    def __init__(
        self,
        strides: dict[int, int],
        *,
        submit: bool = True,
        batching: dict[int, int] | None = None,
        results: tuple[bool, ...] | None = None,
        index_location: int = 0,
        timeline: list[str] | None = None,
    ) -> None:
        self.strides = strides
        self.submit = submit
        self.batching = batching or {}
        self.results = iter(results) if results is not None else None
        self.index_location = index_location
        self.timeline = timeline
        self.calls: list[tuple[str, int, int, list[tuple[int, int]], int]] = []

    def query_cold_page_bytes(self, layer_group_id: int) -> int:
        return self.strides.get(layer_group_id, 0)

    def get_batching_layer_group_id(self, layer_group_id: int) -> int:
        return self.batching.get(layer_group_id, layer_group_id)

    def query_page_index_location(self, _layer_group_id: int) -> int:
        return self.index_location

    def _submit(
        self,
        operation: str,
        layer_group_id: int,
        cold_base: int,
        page_indices: list[tuple[int, int]],
        stream: int,
    ) -> bool:
        self.calls.append((operation, layer_group_id, cold_base, list(page_indices), stream))
        if self.timeline is not None:
            self.timeline.append(operation)
        return self.submit if self.results is None else next(self.results)

    def encode(
        self,
        layer_group_id: int,
        cold_base: int,
        page_indices: list[tuple[int, int]],
        stream: int,
    ) -> bool:
        return self._submit("encode", layer_group_id, cold_base, page_indices, stream)

    def decode(
        self,
        layer_group_id: int,
        cold_base: int,
        page_indices: list[tuple[int, int]],
        stream: int,
    ) -> bool:
        return self._submit("decode", layer_group_id, cold_base, page_indices, stream)


class _Slot:
    def __init__(self, slot_id: int) -> None:
        self.slot_id = SlotId(slot_id)
        self.ready_event = object()


class _Page(_Slot):
    def __init__(self, slot_id: int, life_cycle: int, cache_level: CacheLevel) -> None:
        super().__init__(slot_id)
        self.life_cycle = LifeCycleId(life_cycle)
        self.cache_level = cache_level
        self.node_ref = None
        self.scheduled_for_eviction = False

    def set_slot(self, slot: _Slot) -> None:
        self.slot_id = slot.slot_id


class _PoolGroup:
    def __init__(self, slot_sizes: list[int], slot_count: int, base: int) -> None:
        self.slot_size = list(slot_sizes)
        self.num_pools = len(slot_sizes)
        self.num_slots = slot_count
        self.num_free_slots = slot_count
        self.base = base
        self.next_slot_ids: list[int] = []
        self.released: list[_Slot] = []

    def allocate_multiple(self, count: int) -> list[_Slot]:
        slot_ids = self.next_slot_ids or list(range(count))
        assert len(slot_ids) == count
        self.next_slot_ids = []
        self.num_free_slots -= count
        return [_Slot(slot_id) for slot_id in slot_ids]

    def slot_address(self, slot_id: SlotId) -> tuple[int, ...]:
        return tuple(
            self.base + int(slot_id) * 0x10000 + pool_index * 0x100000
            for pool_index in range(self.num_pools)
        )

    def release(self, slot: _Slot) -> None:
        self.released.append(slot)
        self.num_free_slots += 1


class _Storage:
    """Allocation-free implementation of the CacheLevelStorage test seam."""

    def __init__(
        self,
        cache_tier: CacheTier,
        slot_size_lists: list[list[int]],
        slot_count_list: list[int],
    ) -> None:
        self.cache_tier = cache_tier
        self.slot_size_lists = [list(sizes) for sizes in slot_size_lists]
        self.slot_count_list = list(slot_count_list)
        self.pool_size_granularity = 4096
        self.total_quota = sum(
            count * sum(sizes)
            for count, sizes in zip(self.slot_count_list, self.slot_size_lists)
        )
        total = self.total_quota or 1
        self.ratio_list = [
            count * sum(sizes) / total
            for count, sizes in zip(self.slot_count_list, self.slot_size_lists)
        ]
        tier_base = {
            CacheTier.GPU_MEM: 0x10000000,
            CacheTier.HOST_MEM: 0x50000000,
            CacheTier.DISK: 0x90000000,
        }[cache_tier]
        self._pool_groups = [
            _PoolGroup(sizes, count, tier_base + pg_index * 0x1000000)
            for pg_index, (sizes, count) in enumerate(
                zip(self.slot_size_lists, self.slot_count_list)
            )
        ]
        self.destroyed = False

    @property
    def num_pool_groups(self) -> int:
        return len(self._pool_groups)

    def get_num_free_slots(self, pg_index: PoolGroupIndex) -> int:
        return self._pool_groups[pg_index].num_free_slots

    def num_slots(self, pg_index: PoolGroupIndex) -> int:
        return self._pool_groups[pg_index].num_slots

    def slot_size(self, pg_index: PoolGroupIndex) -> list[int]:
        return self.slot_size_lists[pg_index]

    def destroy(self) -> None:
        self.destroyed = True


class _TemporaryCudaStream:
    last_finish: object | None = None

    def __init__(self, _prior_events: set[object]) -> None:
        self.finish = object()
        _TemporaryCudaStream.last_finish = self.finish

    def __enter__(self) -> "_TemporaryCudaStream":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def get(self) -> int:
        return 0xCAFE

    def take_finish_event(self) -> object:
        return self.finish


class _StagingBuffer:
    def __init__(self, address: int, size: int, stream: int) -> None:
        self.address = address
        self.size = size
        self.stream = stream

    def __enter__(self) -> "_StagingBuffer":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _StagingBufferManager:
    def __init__(self, address: int = 0xD0000000, size: int = 64 << 20) -> None:
        self.address = address
        self.size = size
        self.requests: list[tuple[int, int, int]] = []

    def new(self, min_size: int, max_size: int, stream: int) -> _StagingBuffer:
        self.requests.append((min_size, max_size, stream))
        return _StagingBuffer(self.address, min(max_size, self.size), stream)


@pytest.fixture(autouse=True)
def _use_test_cuda_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.TemporaryCudaStream",
        _TemporaryCudaStream,
    )


def _manager_config(*, topology: str, include_disk: bool) -> KVCacheManagerConfig:
    key = DataRole("key")
    value = DataRole("value")
    layers = [
        AttentionLayerConfig(
            layer_id=LayerId(0),
            buffers=[BufferConfig(key, 4096), BufferConfig(value, 2048)],
        )
    ]
    if topology == "hybrid":
        layers.append(
            SsmLayerConfig(
                layer_id=LayerId(1),
                buffers=[
                    BufferConfig(DataRole("ssm_state"), 3072),
                    BufferConfig(DataRole("conv_state"), 1024),
                ],
            )
        )
    elif topology == "two_attention_lifecycles":
        layers.append(
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(key, 4096), BufferConfig(value, 2048)],
                sliding_window_size=64,
            )
        )
    elif topology != "attention":
        raise ValueError(f"Unknown topology: {topology}")

    cache_tiers = [
        GpuCacheTierConfig(quota=64 << 20),
        HostCacheTierConfig(quota=64 << 20),
    ]
    if include_disk:
        cache_tiers.append(DiskCacheTierConfig(quota=64 << 20, path="/tmp"))
    return KVCacheManagerConfig(
        tokens_per_block=16,
        cache_tiers=cache_tiers,
        layers=layers,
        commit_min_snapshot=topology == "hybrid",
    )


def _make_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    topology: str = "attention",
    include_disk: bool = False,
) -> tuple[StorageManager, list[_Storage]]:
    storages: list[_Storage] = []

    def create_storage(config, slot_size_lists, slot_count_list):
        storage = _Storage(config.tier, slot_size_lists, slot_count_list)
        storages.append(storage)
        return storage

    monkeypatch.setattr(
        CacheLevelManager,
        "_create_cache_level_storage",
        staticmethod(create_storage),
    )
    config = _manager_config(topology=topology, include_disk=include_disk)
    manager = StorageManager(
        LifeCycleRegistry(config),
        create_storage_config(config),
        config.tokens_per_block,
        config.swa_scratch_reuse,
    )
    return manager, storages


def _pool_group(manager: StorageManager, level: CacheLevel, life_cycle: int) -> _PoolGroup:
    pg_index = manager.get_pool_group_index(LifeCycleId(life_cycle))
    return manager._pool_group(level, pg_index)


def _migrate(
    manager: StorageManager,
    src_level: CacheLevel,
    dst_level: CacheLevel,
    pages: list[_Page],
    dst_ids: tuple[int, ...],
    *,
    update_src: bool = True,
):
    pg_index = manager.get_pool_group_index(pages[0].life_cycle)
    dst_pool_group = manager._pool_group(dst_level, pg_index)
    dst_pool_group.next_slot_ids = list(dst_ids)
    return manager._batched_migrate(
        pg_index,
        dst_level,
        src_level,
        pages,
        update_src=update_src,
    )


def test_codec_install_uses_compact_attention_and_raw_ssm_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, storages = _make_manager(monkeypatch, topology="hybrid", include_disk=True)
    old_host, old_disk = storages[1:]

    manager.set_cold_page_codec(_Codec({0: 4096}))

    assert old_host.destroyed and old_disk.destroyed
    assert manager.slot_size(PoolGroupIndex(0), CacheLevel(1)) == [4096]
    assert manager.slot_size(PoolGroupIndex(0), CacheLevel(2)) == [4096]
    assert manager.slot_size(PoolGroupIndex(1), CacheLevel(1)) == [3072, 1024]
    assert manager.slot_size(PoolGroupIndex(1), CacheLevel(2)) == [3072, 1024]


@pytest.mark.parametrize(
    "strides,error",
    [
        ({0: 4096}, "cannot mix raw and compressed"),
        ({0: 4096, 1: 8192}, "identical cold Page stride"),
    ],
)
def test_codec_install_rejects_inconsistent_shared_pool_group(
    monkeypatch: pytest.MonkeyPatch,
    strides: dict[int, int],
    error: str,
) -> None:
    manager, storages = _make_manager(monkeypatch, topology="two_attention_lifecycles")
    old_host = storages[1]

    with pytest.raises(RuntimeError, match=error):
        manager.set_cold_page_codec(_Codec(strides))

    assert not old_host.destroyed


@pytest.mark.parametrize(
    "topology,codec,life_cycle",
    [
        ("attention", None, 0),
        ("hybrid", _Codec({0: 4096}), 1),
    ],
)
def test_raw_fallback_preserves_per_pool_copy(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    codec: _Codec | None,
    life_cycle: int,
) -> None:
    manager, _ = _make_manager(monkeypatch, topology=topology)
    if codec is not None:
        manager.set_cold_page_codec(codec)
    page = _Page(3, life_cycle, GPU_LEVEL)

    with patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.batched_copy"
    ) as copy:
        _migrate(manager, GPU_LEVEL, CacheLevel(1), [page], (5,))

    pg_index = manager.get_pool_group_index(LifeCycleId(life_cycle))
    assert copy.call_count == manager.num_pools(pg_index, GPU_LEVEL)
    assert codec is None or codec.calls == []
    assert (page.slot_id, page.cache_level) == (5, CacheLevel(1))


@pytest.mark.parametrize(
    "src_level,dst_level,operation",
    [
        (GPU_LEVEL, CacheLevel(1), "encode"),
        (CacheLevel(1), GPU_LEVEL, "decode"),
    ],
)
def test_host_boundary_batches_noncontiguous_pages_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    src_level: CacheLevel,
    dst_level: CacheLevel,
    operation: str,
) -> None:
    manager, _ = _make_manager(monkeypatch)
    codec = _Codec({0: 4096})
    manager.set_cold_page_codec(codec)
    pages = [_Page(3, 0, src_level), _Page(11, 0, src_level)]

    _migrate(manager, src_level, dst_level, pages, (5, 17))

    assert len(codec.calls) == 1
    call_operation, layer_group, _, indices, stream = codec.calls[0]
    assert (call_operation, layer_group, indices, stream) == (
        operation,
        0,
        [(5, 3), (17, 11)],
        0xCAFE,
    )
    assert [(page.slot_id, page.cache_level) for page in pages] == [
        (5, dst_level),
        (17, dst_level),
    ]
    assert len(_pool_group(manager, src_level, 0).released) == 2


def test_codec_equivalent_lifecycles_share_one_codec_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _make_manager(monkeypatch, topology="two_attention_lifecycles")
    codec = _Codec({0: 4096, 1: 4096}, batching={1: 0})
    manager.set_cold_page_codec(codec)
    pages = [
        _Page(3, 0, GPU_LEVEL),
        _Page(11, 1, GPU_LEVEL),
        _Page(19, 0, GPU_LEVEL),
    ]

    _migrate(manager, GPU_LEVEL, CacheLevel(1), pages, (5, 17, 23))

    assert len(codec.calls) == 1
    assert codec.calls[0][0:2] == ("encode", 0)
    assert codec.calls[0][3] == [(5, 3), (17, 11), (23, 19)]


@pytest.mark.parametrize(
    "src_level,dst_level,expected_dst_tier,expected_src_tier",
    [
        (CacheLevel(1), CacheLevel(2), CacheTier.DISK, CacheTier.HOST_MEM),
        (CacheLevel(2), CacheLevel(1), CacheTier.HOST_MEM, CacheTier.DISK),
    ],
)
def test_host_disk_copies_compact_pages_without_reencoding(
    monkeypatch: pytest.MonkeyPatch,
    src_level: CacheLevel,
    dst_level: CacheLevel,
    expected_dst_tier: CacheTier,
    expected_src_tier: CacheTier,
) -> None:
    manager, _ = _make_manager(monkeypatch, include_disk=True)
    codec = _Codec({0: 4096})
    manager.set_cold_page_codec(codec)
    page = _Page(3, 0, src_level)

    with patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.batched_copy"
    ) as copy:
        _migrate(manager, src_level, dst_level, [page], (5,))

    assert codec.calls == []
    copy.assert_called_once()
    assert copy.call_args.args[:3] == (expected_dst_tier, expected_src_tier, 4096)


@pytest.mark.parametrize(
    "src_level,dst_level,operation,timeline,indices",
    [
        (GPU_LEVEL, CacheLevel(2), "encode", ["encode", "copy"], [(0, 3), (1, 11)]),
        (CacheLevel(2), GPU_LEVEL, "decode", ["copy", "decode"], [(5, 0), (17, 1)]),
    ],
)
def test_gpu_disk_stages_compact_pages_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    src_level: CacheLevel,
    dst_level: CacheLevel,
    operation: str,
    timeline: list[str],
    indices: list[tuple[int, int]],
) -> None:
    observed: list[str] = []
    manager, _ = _make_manager(monkeypatch, include_disk=True)
    codec = _Codec({0: 4096}, timeline=observed)
    manager.set_cold_page_codec(codec)
    pages = [_Page(3, 0, src_level), _Page(11, 0, src_level)]
    staging = _StagingBufferManager()

    def record_copy(*_args) -> None:
        assert [(page.slot_id, page.cache_level) for page in pages] == [
            (3, src_level),
            (11, src_level),
        ]
        observed.append("copy")

    with (
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager._copy_engine",
            SimpleNamespace(staging_buffer_manager=staging),
        ),
        patch(
            "tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager.batched_copy",
            side_effect=record_copy,
        ),
    ):
        _migrate(manager, src_level, dst_level, pages, (5, 17))

    assert observed == timeline
    assert codec.calls[0][0] == operation
    assert codec.calls[0][3] == indices
    assert staging.requests == [(4096, 8192, 0xCAFE)]
    assert [(page.slot_id, page.cache_level) for page in pages] == [
        (5, dst_level),
        (17, dst_level),
    ]


def test_codec_submission_failure_fences_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _make_manager(monkeypatch)
    codec = _Codec({0: 4096}, submit=False)
    manager.set_cold_page_codec(codec)
    page = _Page(3, 0, GPU_LEVEL)
    src_pool_group = _pool_group(manager, GPU_LEVEL, 0)
    dst_pool_group = _pool_group(manager, CacheLevel(1), 0)

    with pytest.raises(RuntimeError, match="submission failed"):
        _migrate(manager, GPU_LEVEL, CacheLevel(1), [page], (5,))

    assert (page.slot_id, page.cache_level) == (3, GPU_LEVEL)
    assert page.ready_event is _TemporaryCudaStream.last_finish
    assert src_pool_group.released == []
    assert [slot.slot_id for slot in dst_pool_group.released] == [5]
    assert dst_pool_group.released[0].ready_event is _TemporaryCudaStream.last_finish


def test_python_path_rejects_device_page_indices_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _make_manager(monkeypatch)
    codec = _Codec({0: 4096}, index_location=1)
    manager.set_cold_page_codec(codec)
    page = _Page(3, 0, GPU_LEVEL)
    dst_pool_group = _pool_group(manager, CacheLevel(1), 0)

    with pytest.raises(RuntimeError, match="Host Page-index pairs only"):
        _migrate(manager, GPU_LEVEL, CacheLevel(1), [page], (5,))

    assert codec.calls == []
    assert (page.slot_id, page.cache_level) == (3, GPU_LEVEL)
    assert [slot.slot_id for slot in dst_pool_group.released] == [5]


def test_partial_submission_failure_fences_all_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _make_manager(monkeypatch, topology="two_attention_lifecycles")
    codec = _Codec({0: 4096, 1: 4096}, results=(True, False))
    manager.set_cold_page_codec(codec)
    pages = [_Page(3, 0, GPU_LEVEL), _Page(11, 1, GPU_LEVEL)]
    dst_pool_group = _pool_group(manager, CacheLevel(1), 0)

    with pytest.raises(RuntimeError, match="submission failed"):
        _migrate(manager, GPU_LEVEL, CacheLevel(1), pages, (5, 17))

    assert len(codec.calls) == 2
    assert [(page.slot_id, page.cache_level) for page in pages] == [
        (3, GPU_LEVEL),
        (11, GPU_LEVEL),
    ]
    assert all(page.ready_event is _TemporaryCudaStream.last_finish for page in pages)
    assert [slot.slot_id for slot in dst_pool_group.released] == [5, 17]
    assert all(
        slot.ready_event is _TemporaryCudaStream.last_finish
        for slot in dst_pool_group.released
    )


def test_snapshot_clone_keeps_source_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _make_manager(monkeypatch)
    codec = _Codec({0: 4096})
    manager.set_cold_page_codec(codec)
    page = _Page(3, 0, GPU_LEVEL)

    dst_slots = _migrate(
        manager,
        GPU_LEVEL,
        CacheLevel(1),
        [page],
        (5,),
        update_src=False,
    )

    assert dst_slots is not None
    assert [slot.slot_id for slot in dst_slots] == [5]
    assert codec.calls[0][0] == "encode"
    assert codec.calls[0][3] == [(5, 3)]
    assert (page.slot_id, page.cache_level) == (3, GPU_LEVEL)
    assert _pool_group(manager, GPU_LEVEL, 0).released == []
