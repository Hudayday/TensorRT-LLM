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

"""Import-light tests for tier-specific KVCM V2 physical slot geometry."""

from importlib.util import find_spec
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import patch

if find_spec("kv_cache_manager_v2") is not None:
    import kv_cache_manager_v2._core._kv_cache as kv_cache_module
    import kv_cache_manager_v2._storage_manager as storage_manager_module
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        BufferId,
        CoalescedBuffer,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
        _KVCache,
    )
    from kv_cache_manager_v2._common import GPU_LEVEL, MemAddress
    from kv_cache_manager_v2._copy_engine import zero_gpu_memory
    from kv_cache_manager_v2._exceptions import OutOfPagesError
    from kv_cache_manager_v2._storage._config import create_storage_config
    from kv_cache_manager_v2._storage_manager import StorageManager
else:
    import tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache as kv_cache_module
    import tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager as storage_manager_module
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        BufferId,
        CoalescedBuffer,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
        _KVCache,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL, MemAddress
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import zero_gpu_memory
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import create_storage_config
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager


def _config(layers):
    return KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[
            GpuCacheTierConfig(quota=1 << 20),
            HostCacheTierConfig(quota=1 << 20),
        ],
        layers=layers,
    )


def test_host_geometry_is_compact_without_changing_gpu_geometry() -> None:
    storage = create_storage_config(
        _config(
            [
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role="key", size=256, host_size=80),
                        BufferConfig(role="value", size=256, host_size=80),
                    ],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role="key", size=256, host_size=80),
                        BufferConfig(role="value", size=256, host_size=80),
                    ],
                ),
            ]
        )
    )

    assert len(storage.slot_desc_list) == 1
    slot = storage.slot_desc_list[0]
    assert list(slot.slot_size_list) == [4 * 256]
    assert list(slot.host_slot_size_list) == [4 * 80]
    assert slot.variants[0].coalesced_buffers[0].num_buffers == 4


def test_default_host_geometry_preserves_raw_representation() -> None:
    storage = create_storage_config(
        _config(
            [
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role="key", size=256)],
                )
            ]
        )
    )
    slot = storage.slot_desc_list[0]
    assert slot.host_slot_size_list == slot.slot_size_list


def test_zero_size_temporary_buffer_remains_valid_for_hybrid_config_builders() -> None:
    """Boundary metadata must not reject main's Mamba placeholder buffers.

    Hybrid cache managers construct zero-byte attention placeholders first and
    replace them with SSM/conv buffers before final storage construction.
    """
    placeholder = BufferConfig(role="key", size=0)

    assert placeholder.size == 0
    assert placeholder.host_size is None


def test_coalesced_buffer_preserves_legacy_positional_constructor() -> None:
    buffer_ids = (BufferId(LayerId(0), DataRole("key")),)

    legacy = CoalescedBuffer(256, buffer_ids)
    compact = CoalescedBuffer(
        256,
        buffer_ids,
        host_single_buffer_size=80,
    )

    assert legacy.effective_host_single_buffer_size == 256
    assert legacy.host_size == legacy.size == 256
    assert compact.effective_host_single_buffer_size == 80
    assert compact.host_size == 80


def test_life_cycles_merge_only_when_gpu_and_host_layouts_both_match() -> None:
    storage = create_storage_config(
        _config(
            [
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role="key", size=256, host_size=80)],
                    sliding_window_size=None,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role="key", size=256, host_size=96)],
                    sliding_window_size=64,
                ),
            ]
        )
    )

    assert len(storage.slot_desc_list) == 2
    assert {tuple(slot.slot_size_list) for slot in storage.slot_desc_list} == {(256,)}
    assert {tuple(slot.host_slot_size_list) for slot in storage.slot_desc_list} == {(80,), (96,)}


def test_equal_gpu_sizes_use_host_size_for_canonical_pool_order() -> None:
    storage = create_storage_config(
        _config(
            [
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role="key", size=256, host_size=80),
                        BufferConfig(role="value", size=256, host_size=96),
                    ],
                    sliding_window_size=None,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role="key", size=256, host_size=96),
                        BufferConfig(role="value", size=256, host_size=80),
                    ],
                    sliding_window_size=64,
                ),
            ]
        )
    )

    assert len(storage.slot_desc_list) == 1
    assert tuple(storage.slot_desc_list[0].slot_size_list) == (256, 256)
    assert tuple(storage.slot_desc_list[0].host_slot_size_list) == (96, 80)


def test_boundary_runtime_slot_initialization_clears_every_pool_and_sets_fence() -> None:
    old_events = (object(), object())
    slots = [
        [SimpleNamespace(slot_id=3, ready_event=old_events[0])],
        [SimpleNamespace(slot_id=5, ready_event=old_events[1])],
    ]
    finish_event = object()

    class FakeTemporaryCudaStream:
        prior_events = None

        def __init__(self, prior_events):
            type(self).prior_events = prior_events

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            assert exc_type is None

        def get(self):
            return 71

        def take_finish_event(self):
            return finish_event

    class StorageHarness(StorageManager):
        def __init__(self):
            self._offload_compression_hook = lambda **_: None

        def __del__(self):
            pass

        def get_pool_group_index(self, life_cycle):
            return life_cycle

        def slot_size(self, pool_group_index, cache_level=None):
            assert cache_level == GPU_LEVEL
            return [16, 32]

        def slot_address(self, level, pool_group_index, slot_id, pool_index):
            assert level == GPU_LEVEL
            return MemAddress(
                10_000 + int(pool_group_index) * 1_000 + int(slot_id) * 10 + int(pool_index)
            )

    with (
        patch.object(
            storage_manager_module,
            "TemporaryCudaStream",
            FakeTemporaryCudaStream,
        ),
        patch.object(storage_manager_module, "zero_gpu_memory") as zero_gpu_memory,
    ):
        StorageHarness()._initialize_runtime_gpu_slots(slots)

    assert FakeTemporaryCudaStream.prior_events == set(old_events)
    assert zero_gpu_memory.call_count == 4
    assert [call.args[1:] for call in zero_gpu_memory.call_args_list] == [
        (16, 71),
        (32, 71),
        (16, 71),
        (32, 71),
    ]
    assert all(slot.ready_event is finish_event for slot_list in slots for slot in slot_list)


def test_zero_gpu_memory_clears_recycled_payload() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required")

    payload = torch.full((256,), 0xA5, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    zero_gpu_memory(
        MemAddress(payload.data_ptr()),
        payload.numel(),
        CudaStream(stream.cuda_stream),
    )
    stream.synchronize()

    assert torch.count_nonzero(payload).item() == 0


def test_cross_tier_snapshot_rejoins_runtime_stream_on_success_and_failure() -> None:
    runtime_stream = object()

    for migration_error in (None, RuntimeError("transform failed")):
        wait_calls = []
        scheduled_pages = []

        class FinishEvent:
            def wait_in_stream(self, stream):
                wait_calls.append(stream)

        finish_event = FinishEvent()
        src_page = SimpleNamespace(
            cache_level=0,
            ready_event=SimpleNamespace(wait_in_stream=lambda _: None),
        )
        migrated_slot = object()

        class FakeCommittedPage:
            def __init__(self, storage, tree_block, life_cycle, level, slot, priority):
                del storage, tree_block, life_cycle, level, priority
                self.slot = slot

        class StorageHarness:
            num_cache_levels = 2
            num_pool_groups = 1

            def get_pool_group_index(self, life_cycle):
                return 0

            def slot_size(self, pool_group_index, level):
                del pool_group_index
                return [16] if level == 0 else [8]

            def new_slots_for_pool_group(self, level, pool_group_index, count):
                raise OutOfPagesError("force the cross-tier path")

            def prepare_free_slots(self, level, requirements):
                pass

            def _batched_migrate(self, *args, **kwargs):
                src_page.ready_event = finish_event
                if migration_error is not None:
                    raise migration_error
                return [migrated_slot]

            def schedule_for_eviction(self, page):
                scheduled_pages.append(page)

        class LifeCycles:
            ssm_life_cycle_id = None

            def __getitem__(self, life_cycle):
                return life_cycle

        fake_cache = SimpleNamespace(
            manager=SimpleNamespace(
                _life_cycles=LifeCycles(),
                _storage=StorageHarness(),
            ),
            cuda_stream=runtime_stream,
            _get_priority=lambda ordinal, life_cycle: 0,
            # A real _KVCache records this fence before taking a reusable
            # snapshot.  The harness supplies it explicitly so the test
            # exercises the cross-stream hand-off rather than failing on an
            # incomplete fake object.
            finish_event=object(),
        )
        tree_block = SimpleNamespace(storage=[None], ordinal=0)

        with (
            patch.object(kv_cache_module, "CommittedPage", FakeCommittedPage),
            patch.object(kv_cache_module.rawref, "ref", side_effect=lambda page: lambda: page),
        ):
            if migration_error is None:
                committed = _KVCache._copy_page_to_tree_block(
                    fake_cache,
                    tree_block,
                    0,
                    src_page,
                )
                assert committed is scheduled_pages[0]
                assert committed.slot is migrated_slot
                assert tree_block.storage[0]() is committed
            else:
                try:
                    _KVCache._copy_page_to_tree_block(
                        fake_cache,
                        tree_block,
                        0,
                        src_page,
                    )
                except RuntimeError as error:
                    assert error is migration_error
                else:
                    raise AssertionError("Expected the migration failure")
                assert not scheduled_pages
                assert tree_block.storage == [None]

        assert wait_calls == [runtime_stream]
