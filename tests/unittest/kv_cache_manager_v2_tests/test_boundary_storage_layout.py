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

    for migration_error, expected_error in (
        (None, AssertionError),
        (RuntimeError("transform failed"), RuntimeError),
    ):
        wait_calls = []

        class FinishEvent:
            def wait_in_stream(self, stream):
                wait_calls.append(stream)

        finish_event = FinishEvent()
        src_page = SimpleNamespace(
            cache_level=0,
            ready_event=SimpleNamespace(wait_in_stream=lambda _: None),
        )

        class StorageHarness:
            num_cache_levels = 2
            num_pool_groups = 1

            def get_pool_group_index(self, life_cycle):
                return 0

            def new_slots_for_pool_group(self, level, pool_group_index, count):
                raise OutOfPagesError("force the cross-tier path")

            def prepare_free_slots(self, level, requirements):
                pass

            def _batched_migrate(self, *args, **kwargs):
                src_page.ready_event = finish_event
                if migration_error is not None:
                    raise migration_error
                # Returning normally proves the finally block runs on success;
                # the invalid cardinality stops before Page publication.
                return []

        life_cycles = SimpleNamespace(ssm_life_cycle_id=None)
        fake_cache = SimpleNamespace(
            manager=SimpleNamespace(
                _life_cycles=life_cycles,
                _storage=StorageHarness(),
            ),
            cuda_stream=runtime_stream,
            # A real _KVCache records this fence before taking a reusable
            # snapshot.  The harness supplies it explicitly so the test
            # exercises the cross-stream hand-off rather than failing on an
            # incomplete fake object.
            finish_event=object(),
        )
        tree_block = SimpleNamespace(storage=[None])

        try:
            _KVCache._copy_page_to_tree_block(
                fake_cache,
                tree_block,
                0,
                src_page,
            )
        except expected_error:
            pass
        else:
            raise AssertionError(f"Expected {expected_error.__name__}")

        assert wait_calls == [runtime_stream]
