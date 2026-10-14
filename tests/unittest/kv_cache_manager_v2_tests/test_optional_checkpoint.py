# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for native OPTIONAL selection; storage contracts live in C++."""

import pytest
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_cache import (
    add_mixed_precision_windows,
)
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture(autouse=True)
def cuda_context():
    # The native manager expects its caller to establish the current context.
    torch.cuda.current_stream()


@pytest.mark.parametrize("high_bytes", [64, 128])
def test_mixed_precision_uses_native_swa_tables_and_pool_addresses(high_bytes: int) -> None:
    """Storage/selection QA, not a quantizer or model-accuracy test.

    Distinct byte patterns make accidental low/high aliasing visible. The
    two high-page sizes exercise equal and unequal physical page strides.
    """
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
        layers=[
            kv.AttentionLayerConfig(layer_id=0, buffers=[kv.BufferConfig(role="key", size=64)])
        ],
        initial_pool_ratio=[0.75, 0.25],
    )
    paired_config, (pair,) = add_mixed_precision_windows(
        config, {0: [kv.BufferConfig(role="key", size=high_bytes)]}, first_n=4, last_n=8
    )
    assert len(config.layers) == 1
    assert paired_config.layers[1].sliding_window_size == 9
    assert paired_config.layers[1].num_sink_tokens == 4
    assert paired_config.layers[1].reuse_policy == kv.AttentionReusePolicy.OPTIONAL
    manager = kv.KVCacheManager(paired_config)
    stream = torch.cuda.Stream()
    cache = manager.create_kv_cache(None, [])
    batch = kv.Batch(manager, max_rows=1, max_blocks=10)
    try:
        assert manager.get_layer_group_id(pair.history_layer) != manager.get_layer_group_id(
            pair.high_layer
        )
        assert cache.resume(stream.cuda_stream) and cache.resize(32)
        addressing = [pair.buffer_addressing(manager, "key", high=high) for high in (False, True)]
        assert addressing[0][0] != addressing[1][0]
        # Distinct payloads stand in for the two writers; no quantization is tested here.
        for high, layer in enumerate((pair.history_layer, pair.high_layer)):
            base, stride, converter = addressing[high]
            slots = cache.get_base_page_indices(manager.get_layer_group_id(layer))
            for ordinal, slot in enumerate(converter(slots, kv.PageIndexMode.SHARED)):
                if slot >= 0:
                    assert (
                        cudart.cudaMemsetAsync(
                            base + int(slot) * stride,
                            20 + 100 * high + ordinal,
                            high_bytes if high else 64,
                            stream.cuda_stream,
                        )[0]
                        == cudart.cudaError_t.cudaSuccess
                    )
        cache.commit(list(range(32)))
        batch.add(cache)
        batch.publish(stream.cuda_stream)
        with torch.cuda.stream(stream):
            batch.wait_ready(stream.cuda_stream)
            tables = [torch.from_dlpack(table) for table in pair.page_tables(manager, batch)]
            table_addresses = [table.data_ptr() for table in tables]
            slots = [table[0, 0, :8].cpu().tolist() for table in tables]
            assert all(slot >= 0 for slot in slots[0])
            assert [i for i, slot in enumerate(slots[1]) if slot >= 0] == [0, 6, 7]
            readback = torch.empty((8,), dtype=torch.uint8, device="cuda")
            for ordinal in range(8):
                high = int(slots[1][ordinal] >= 0)
                base, stride, converter = addressing[high]
                slot = converter([slots[high][ordinal]], kv.PageIndexMode.SHARED)[0]
                assert (
                    cudart.cudaMemcpyAsync(
                        readback.data_ptr() + ordinal,
                        base + int(slot) * stride,
                        1,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        stream.cuda_stream,
                    )[0]
                    == cudart.cudaError_t.cudaSuccess
                )
            batch.record_read(stream.cuda_stream)
            assert readback.cpu().tolist() == [120, 21, 22, 23, 24, 25, 126, 127]
        cache.suspend()
        assert cache.resume(stream.cuda_stream)
        batch.publish(stream.cuda_stream)
        with torch.cuda.stream(stream):
            batch.wait_ready(stream.cuda_stream)
            assert [
                torch.from_dlpack(table).data_ptr() for table in pair.page_tables(manager, batch)
            ] == table_addresses
            batch.record_read(stream.cuda_stream)
    finally:
        batch.close()
        cache.close()
        manager.shutdown()


def test_reuse_status_and_optional_selection_bindings():
    layers = [
        kv.AttentionLayerConfig(
            layer_id=i,
            buffers=[kv.BufferConfig(role="key", size=4096)],
            sliding_window_size=None if i == 0 else 4 if i == 1 else 2,
            reuse_policy=kv.AttentionReusePolicy.OPTIONAL
            if i
            else kv.AttentionReusePolicy.REQUIRED,
        )
        for i in range(3)
    ]
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4, cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)], layers=layers
    )
    stream = torch.cuda.Stream()
    manager = kv.KVCacheManager(config)
    caches = []
    tokens = list(range(8))
    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(8)
        source.commit(tokens)
        reader = manager.create_kv_cache(None, tokens)
        caches.append(reader)
        groups = [manager.get_layer_group_id(i) for i in range(3)]
        global_group, encoder, peer = groups
        for layer, group in enumerate(groups):
            status = reader.reuse_status[group]
            assert status.group_id == group
            assert status.policy == config.layers[layer].reuse_policy
            assert status.complete and status.endpoint == 8
            assert list(status.coverage) == ([(0, 8)] if layer == 0 else [(4, 8)])
        for invalid in ([global_group], [-1], [len(reader.reuse_status)]):
            with pytest.raises(ValueError):
                reader.resume(stream.cuda_stream, optional_reuse_groups=invalid)
            assert not reader.is_active
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[encoder])
        assert reader.get_base_page_indices(encoder)[1] == source.get_base_page_indices(encoder)[1]
        assert reader.get_base_page_indices(peer)[1] != source.get_base_page_indices(peer)[1]
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
        assert reader.get_base_page_indices(encoder)[1] == source.get_base_page_indices(encoder)[1]
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("missing_high", [False, True])
def test_optional_working_values_and_initialized_snapshot(missing_high: bool) -> None:
    """Recomputed high values stay local; publication never replaces old history."""
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=[
                kv.AttentionLayerConfig(
                    layer_id=0, buffers=[kv.BufferConfig(role="key", size=2048)]
                ),
                kv.AttentionLayerConfig(
                    layer_id=1,
                    buffers=[kv.BufferConfig(role="key", size=4096)],
                    sliding_window_size=5,
                    num_sink_tokens=4,
                    reuse_policy=kv.AttentionReusePolicy.OPTIONAL,
                ),
            ],
        )
    )
    stream = torch.cuda.Stream()
    groups = [manager.get_layer_group_id(i) for i in (0, 1)]
    bases = [manager.get_mem_pool_base_address(i, "key") for i in (0, 1)]
    strides = [manager.get_page_stride(i, "key") for i in (0, 1)]
    sizes = [2048, 4096]
    caches = []

    def fill(cache, layer, ordinal, value):
        slot = cache.get_base_page_indices(groups[layer])[ordinal]
        assert (
            cudart.cudaMemsetAsync(
                bases[layer] + int(slot) * strides[layer], value, sizes[layer], stream.cuda_stream
            )[0]
            == cudart.cudaError_t.cudaSuccess
        )

    def check(cache, layer, ordinal, value):
        slot = cache.get_base_page_indices(groups[layer])[ordinal]
        values = torch.empty((sizes[layer],), dtype=torch.uint8, device="cuda")
        assert (
            cudart.cudaMemcpyAsync(
                values.data_ptr(),
                bases[layer] + int(slot) * strides[layer],
                sizes[layer],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )[0]
            == cudart.cudaError_t.cudaSuccess
        )
        stream.synchronize()
        assert torch.all(values == value).item()

    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream) and source.resize(8)
        for layer, seed in ((0, 11), (1, 77)):
            for ordinal in (0, 1):
                fill(source, layer, ordinal, seed + ordinal)
        source.commit(list(range(8)))
        if missing_high:
            _introspection.drop_optional_pages(source, 8, [groups[1]])
        reader = manager.create_kv_cache(
            None, list(range(8 if missing_high else 4)), allow_partial_optional_reuse=True
        )
        caches.append(reader)
        reader.set_optional_working_groups([groups[1]])
        with pytest.raises(ValueError):
            reader.set_optional_working_groups([groups[0]])
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[groups[1]])
        assert reader.resize(8)
        fill(reader, 1, 1, 99)
        if not missing_high:
            fill(reader, 0, 1, 33)
            reader.commit(list(range(4, 8)))
        # Initialization precedes explicit publication. Resume itself never publishes holes.
        assert reader.publish_optional_pages([groups[1]]) == int(missing_high)
        assert reader.publish_optional_pages([groups[1]]) == 0
        check(reader, 1, 1, 99)
        check(source, 0, 1, 12)
        check(source, 1, 1, 78)
        assert (
            reader.get_base_page_indices(groups[1])[1] != source.get_base_page_indices(groups[1])[1]
        )
        next_reader = manager.create_kv_cache(
            None, list(range(8)), allow_partial_optional_reuse=True
        )
        caches.append(next_reader)
        assert next_reader.reuse_status[groups[1]].complete
        assert next_reader.resume(stream.cuda_stream, optional_reuse_groups=[groups[1]])
        check(next_reader, 0, 1, 12)
        check(next_reader, 1, 1, 99 if missing_high else 78)
        reader.suspend()
        with pytest.raises(RuntimeError):
            reader.publish_optional_pages([groups[1]])
        assert reader.resume(stream.cuda_stream)
        check(reader, 1, 1, 99)
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("endpoint", [6, 8])
def test_original_window_partial_source_preserves_shared_payload(endpoint: int) -> None:
    """Real native storage test, not quantization or Attention validation."""
    layers = [
        kv.AttentionLayerConfig(
            layer_id=0,
            buffers=[kv.BufferConfig(role="key", size=2048)],
            reuse_policy=kv.AttentionReusePolicy.REQUIRED,
        ),
        kv.AttentionLayerConfig(
            layer_id=1,
            buffers=[kv.BufferConfig(role="key", size=4096)],
            # Native windows include the next write: retain four written tokens.
            sliding_window_size=5,
            num_sink_tokens=4,
            reuse_policy=kv.AttentionReusePolicy.OPTIONAL,
        ),
    ]
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=layers,
        )
    )
    stream = torch.cuda.Stream()
    caches = []
    uploads = []
    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(8)
        groups = [manager.get_layer_group_id(i) for i in (0, 1)]
        bases = [manager.get_mem_pool_base_address(i, "key") for i in (0, 1)]
        strides = [manager.get_page_stride(i, "key") for i in (0, 1)]
        source_slots = [list(source.get_base_page_indices(group)) for group in groups]
        buffer_bytes = [2048, 4096]
        for layer, seed in ((0, 11), (1, 77)):
            for ordinal, slot in enumerate(source_slots[layer]):
                payload = torch.full(
                    (buffer_bytes[layer],), seed + ordinal, dtype=torch.uint8, device="cuda"
                )
                uploads.append(payload)
                assert (
                    cudart.cudaMemcpyAsync(
                        bases[layer] + int(slot) * strides[layer],
                        payload.data_ptr(),
                        buffer_bytes[layer],
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        stream.cuda_stream,
                    )[0]
                    == cudart.cudaError_t.cudaSuccess
                )
        stream.synchronize()
        source.commit(list(range(8)))
        reader = manager.create_kv_cache(
            None, list(range(endpoint)), allow_partial_optional_reuse=True
        )
        caches.append(reader)
        high = groups[1]
        assert reader.num_committed_tokens == endpoint
        assert reader.reuse_status[high].complete
        assert list(reader.reuse_status[high].coverage) == [(0, endpoint)]
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[high])
        assert reader.get_base_page_indices(high)[0] == source_slots[1][0]
        if endpoint % 4:
            assert reader.get_base_page_indices(high)[1] != source_slots[1][1]
        # High payload originates from independent source writes, never from low.
        for layer, seed in ((0, 11), (1, 77)):
            for ordinal, slot in enumerate(reader.get_base_page_indices(groups[layer])):
                count = min(4, endpoint - ordinal * 4) * (buffer_bytes[layer] // 4)
                readback = torch.empty((count,), dtype=torch.uint8, device="cuda")
                assert (
                    cudart.cudaMemcpyAsync(
                        readback.data_ptr(),
                        bases[layer] + int(slot) * strides[layer],
                        count,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        stream.cuda_stream,
                    )[0]
                    == cudart.cudaError_t.cudaSuccess
                )
                stream.synchronize()
                assert torch.all(readback == seed + ordinal).item()
        assert list(source.get_base_page_indices(groups[0])) == source_slots[0]
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


def test_missing_tail_publication_excludes_low_fallback_head() -> None:
    """A missing head does not block fresh tails in the same OPTIONAL group."""
    sizes = (2048, 4096)
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=[
                kv.AttentionLayerConfig(
                    layer_id=0, buffers=[kv.BufferConfig(role="key", size=sizes[0])]
                ),
                kv.AttentionLayerConfig(
                    layer_id=1,
                    buffers=[kv.BufferConfig(role="key", size=sizes[1])],
                    sliding_window_size=5,
                    num_sink_tokens=4,
                    reuse_policy=kv.AttentionReusePolicy.OPTIONAL,
                ),
            ],
        )
    )
    stream = torch.cuda.Stream()
    groups = [manager.get_layer_group_id(i) for i in (0, 1)]
    high = groups[1]
    bases = [manager.get_mem_pool_base_address(i, "key") for i in (0, 1)]
    strides = [manager.get_page_stride(i, "key") for i in (0, 1)]
    caches = []

    def fill(cache, layer, ordinal, value):
        slot = cache.get_base_page_indices(groups[layer])[ordinal]
        assert slot >= 0
        assert (
            cudart.cudaMemsetAsync(
                bases[layer] + int(slot) * strides[layer], value, sizes[layer], stream.cuda_stream
            )[0]
            == cudart.cudaError_t.cudaSuccess
        )

    def check(cache, layer, ordinal, value):
        slot = cache.get_base_page_indices(groups[layer])[ordinal]
        values = torch.empty((sizes[layer],), dtype=torch.uint8, device="cuda")
        assert (
            cudart.cudaMemcpyAsync(
                values.data_ptr(),
                bases[layer] + int(slot) * strides[layer],
                sizes[layer],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )[0]
            == cudart.cudaError_t.cudaSuccess
        )
        stream.synchronize()
        assert torch.all(values == value).item()

    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream) and source.resize(16)
        for layer, seed in ((0, 11), (1, 77)):
            for ordinal, slot in enumerate(source.get_base_page_indices(groups[layer])):
                if slot >= 0:
                    fill(source, layer, ordinal, seed + ordinal)
        source.commit(list(range(16)))
        for endpoint in (4, 8, 12):
            _introspection.drop_optional_pages(source, endpoint, [high])
        reader = manager.create_kv_cache(None, list(range(8)), allow_partial_optional_reuse=True)
        caches.append(reader)
        assert not reader.reuse_status[high].complete
        reader.set_optional_working_groups([high])
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[high])
        # These bytes stand for converted low-history values, not original high.
        fill(reader, 1, 0, 11)
        fill(reader, 1, 1, 99)
        # Excluding even one token skips the whole page, not the entire group.
        excluded = [(high, 1, 2)]
        assert reader.publish_optional_pages([high], excluded_ranges=excluded) == 1
        assert reader.publish_optional_pages([high], excluded_ranges=excluded) == 0
        probe = manager.create_kv_cache(None, list(range(8)), allow_partial_optional_reuse=True)
        caches.append(probe)
        assert list(probe.reuse_status[high].coverage) == [(4, 8)]
        assert probe.resume(stream.cuda_stream, optional_reuse_groups=[high])
        check(probe, 1, 1, 99)
        assert reader.resize(12)
        fill(reader, 0, 2, 33)
        fill(reader, 1, 2, 100)
        reader.commit(list(range(8, 12)))
        before_publish = manager.create_kv_cache(
            None, list(range(12)), allow_partial_optional_reuse=True
        )
        caches.append(before_publish)
        assert not before_publish.reuse_status[high].complete
        assert reader.publish_optional_pages([high], excluded_ranges=excluded) == 1
        next_reader = manager.create_kv_cache(
            None, list(range(12)), allow_partial_optional_reuse=True
        )
        caches.append(next_reader)
        assert list(next_reader.reuse_status[high].coverage) == [(8, 12)]
        assert next_reader.resume(stream.cuda_stream, optional_reuse_groups=[high])
        check(next_reader, 1, 2, 100)
        check(source, 0, 2, 13)
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


def test_optional_host_restore_oom_retry_preserves_payload() -> None:
    """A few MiB of real native storage exercise migration and failed admission."""
    page_bytes = 128 << 10
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[
                kv.GpuCacheTierConfig(quota=4 << 20),
                kv.HostCacheTierConfig(quota=4 << 20),
            ],
            initial_pool_ratio=[0.5, 0.5],
            max_util_for_resume=1.0,
            layers=[
                kv.AttentionLayerConfig(
                    layer_id=0, buffers=[kv.BufferConfig(role="key", size=page_bytes)]
                ),
                kv.AttentionLayerConfig(
                    layer_id=1,
                    buffers=[kv.BufferConfig(role="key", size=page_bytes)],
                    sliding_window_size=5,
                    num_sink_tokens=4,
                    reuse_policy=kv.AttentionReusePolicy.OPTIONAL,
                ),
            ],
        )
    )
    stream = torch.cuda.Stream()
    high = manager.get_layer_group_id(1)
    base = manager.get_mem_pool_base_address(1, "key")
    stride = manager.get_page_stride(1, "key")
    caches, pressure = [], []
    try:
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream) and source.resize(8)
        for slot in source.get_base_page_indices(high):
            assert (
                cudart.cudaMemsetAsync(
                    base + int(slot) * stride, 45, page_bytes, stream.cuda_stream
                )[0]
                == cudart.cudaError_t.cudaSuccess
            )
        source.commit(list(range(8)))
        source.suspend()
        for _ in range(64):
            if all(level == 1 for level in source.get_page_storage_snapshot(high).cache_levels):
                break
            blocker = manager.create_kv_cache(None, [])
            pressure.append(blocker)
            assert blocker.resume(stream.cuda_stream) and blocker.resize(4)
        else:
            pytest.fail("Storage pressure did not move the protected source to Host")
        reader = manager.create_kv_cache(None, list(range(8)), allow_partial_optional_reuse=True)
        caches.append(reader)
        assert not reader.resume(stream.cuda_stream, optional_reuse_groups=[high])
        assert not reader.is_active and reader.num_committed_tokens == 8
        for blocker in pressure:
            blocker.close()
        pressure.clear()
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[high])
        for slot in reader.get_base_page_indices(high):
            values = torch.empty((page_bytes,), dtype=torch.uint8, device="cuda")
            assert (
                cudart.cudaMemcpyAsync(
                    values.data_ptr(),
                    base + int(slot) * stride,
                    page_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream.cuda_stream,
                )[0]
                == cudart.cudaError_t.cudaSuccess
            )
            stream.synchronize()
            assert torch.all(values == 45).item()
    finally:
        for cache in reversed(pressure + caches):
            cache.close()
        manager.shutdown()
