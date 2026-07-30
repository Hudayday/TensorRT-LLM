# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light tests for tier-specific KVCM V2 physical slot geometry."""

from importlib.util import find_spec

if find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
    )
    from kv_cache_manager_v2._storage._config import create_storage_config
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import (
        create_storage_config,
    )


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
    assert {tuple(slot.slot_size_list) for slot in storage.slot_desc_list} == {
        (256,)
    }
    assert {
        tuple(slot.host_slot_size_list) for slot in storage.slot_desc_list
    } == {(80,), (96,)}
