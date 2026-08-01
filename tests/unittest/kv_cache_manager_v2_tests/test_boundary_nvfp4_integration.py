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
"""Real KVCM V2 StorageManager integration test for P0 NVFP4 Host offload.

This intentionally stops below the scheduler and Attention.  It exercises the
real Python ``KVCacheManagerV2`` storage layout, real ``StorageManager`` slot
transaction, and real NVFP4 compression/decompression hooks.  No attention
metadata is created or inspected.
"""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression import (
    quantization_for_boundary as boundary_compression_module,
)
from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4 import dequant_nvfp4_2d_triton
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv_cache_manager_v2_runtime
from tensorrt_llm.runtime.kv_cache_manager_v2 import rawref
from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
    CACHE_LEVEL1,
    GPU_LEVEL,
    PRIORITY_DEFAULT,
    PageStatus,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleId
from tensorrt_llm.runtime.kv_cache_manager_v2._page import Page
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import CachedCudaEvent, init_cuda_once

_TOKENS_PER_BLOCK = 4
_HEAD_DIM = 32
_VALID_TOKENS = 3
_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_SCALE_DENOMINATOR = 448.0 * 6.0


class _UncommittedTestPage(Page):
    """Minimal suspended-request Page for exercising P0 Host offload."""

    def is_committed(self) -> bool:
        return False


def _require_p0_runtime() -> None:
    """Skip only when the process cannot execute the real P0 data path."""
    if kv_cache_manager_v2_runtime._BACKEND != "python":
        pytest.skip("set TLLM_KV_CACHE_MANAGER_V2_BACKEND=python before importing TRT-LLM")
    if not torch.cuda.is_available():
        pytest.skip("the real StorageManager migration requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("the NVFP4 boundary kernel requires SM100 or newer")
    try:
        fp4_quantize_op = torch.ops.trtllm.fp4_quantize
    except AttributeError:
        pytest.skip("the installed TensorRT-LLM library has no fp4_quantize op")
    if fp4_quantize_op is None:
        pytest.skip("the installed TensorRT-LLM fp4_quantize op is unavailable")


def _direct_nvfp4_oracle(raw: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize without using the boundary manager helpers.

    This is deliberately independent of Host-record serialization.  Therefore
    a bad offset, scale placement, tier address, or migration direction cannot
    make both the implementation and its oracle pass in the same way.
    """
    matrix = raw.reshape(-1, raw.shape[-1]).contiguous()
    amax = torch.linalg.vector_norm(
        matrix,
        ord=float("inf"),
        dtype=torch.float32,
    )
    inverse_global_scale = amax / _NVFP4_GLOBAL_SCALE_DENOMINATOR
    global_scale = torch.reciprocal(inverse_global_scale)
    packed, block_scales = torch.ops.trtllm.fp4_quantize(
        matrix,
        global_scale,
        _NVFP4_BLOCK_SIZE,
        False,
        False,
    )
    block_scales = block_scales.view(matrix.shape[0], matrix.shape[1] // _NVFP4_BLOCK_SIZE)
    restored = torch.empty_like(matrix)
    dequant_nvfp4_2d_triton(
        packed,
        block_scales,
        inverse_global_scale.reshape(1),
        out=restored,
        target_dtype=matrix.dtype,
    )
    return restored.view_as(raw)


@pytest.fixture(params=(DataType.HALF, DataType.BF16), ids=("fp16", "bf16"))
def p0_manager(request):
    """Build the smallest real two-tier manager supported by the P0 proof."""
    _require_p0_runtime()
    init_cuda_once()

    # KVCacheManagerV2 owns the layout declaration.  Runtime GPU Pages remain
    # FP16/BF16 while the Host tier receives the compact NVFP4 record geometry.
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_tokens=_TOKENS_PER_BLOCK,
            host_cache_size=1 << 20,
            max_util_for_resume=1.0,
            block_reuse_policy="all_reusable",
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_TOKENS_PER_BLOCK,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=request.param,
        vocab_size=128,
        max_num_tokens=_TOKENS_PER_BLOCK,
        enable_stats=False,
        boundary_compression_quant="nvfp4",
    )

    # The compression manager binds only the two representation-transform
    # callbacks.  StorageManager still owns admission, events, publication,
    # source release, and rollback.
    boundary = QuantizationForBoundaryCompression(manager, quant="nvfp4")
    try:
        yield manager
    finally:
        boundary.shutdown()
        manager.shutdown()


def test_real_storage_manager_compact_offload_round_trip(p0_manager) -> None:
    manager = p0_manager
    storage = manager.impl._storage
    life_cycle = LifeCycleId(0)
    pool_group = storage.get_pool_group_index(life_cycle)
    page = None
    recycled = None

    # One FP16/BF16 K Page and one V Page are coalesced into the same physical
    # Pool slot: 2 * (4 tokens * 1 head * 32 elements * 2 bytes) = 512 bytes.
    # Each independent NVFP4 Host record is 80 bytes including its block
    # scales, inverse global scale, and alignment, so the Host slot is 160 B.
    assert tuple(storage.slot_size(pool_group, GPU_LEVEL)) == (512,)
    assert tuple(storage.slot_size(pool_group, CACHE_LEVEL1)) == (160,)

    gpu_free_initial = storage.get_statistics(GPU_LEVEL)[pool_group].free
    host_free_initial = storage.get_statistics(CACHE_LEVEL1)[pool_group].free

    try:
        raw_slot = storage.new_gpu_slots([1])[life_cycle][0]
        page = Page(
            _slot_id=None,
            ready_event=CachedCudaEvent.NULL,
            _manager=rawref.ref(storage),
            life_cycle=life_cycle,
            cache_level=GPU_LEVEL,
            _priority=PRIORITY_DEFAULT,
            _holder=None,
            node_ref=None,
        )
        page.set_slot(raw_slot)
        page.ready_event.synchronize()
        source_slot_id = page.slot_id

        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial - 1

        # ``new_gpu_slots`` clears the whole fresh physical Page before a
        # partial write. This proves the fresh-allocation behavior only; a
        # later partial-reuse copy-on-write can still copy an old suffix.
        buffers = manager.get_buffers(0, kv_layout="NHD")
        assert buffers is not None
        raw_page = buffers[source_slot_id]
        assert raw_page.shape == (2, _TOKENS_PER_BLOCK, 1, _HEAD_DIM)
        assert torch.count_nonzero(raw_page).item() == 0

        values = torch.arange(
            _VALID_TOKENS * _HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        ).view(_VALID_TOKENS, 1, _HEAD_DIM)
        raw_page[0, :_VALID_TOKENS].copy_((((values % 37) - 18) / 8).to(raw_page.dtype))
        raw_page[1, :_VALID_TOKENS].copy_(((((values * 3) % 41) - 20) / 7).to(raw_page.dtype))
        torch.cuda.current_stream().synchronize()
        source = raw_page.clone()
        torch.cuda.current_stream().synchronize()

        # The final token row is intentionally never written.  It must still
        # be deterministic physical input to NVFP4 and remain zero on recall.
        torch.testing.assert_close(
            source[:, _VALID_TOKENS:],
            torch.zeros_like(source[:, _VALID_TOKENS:]),
            rtol=0,
            atol=0,
        )
        oracle = torch.stack((_direct_nvfp4_oracle(source[0]), _direct_nvfp4_oracle(source[1])))
        torch.cuda.current_stream().synchronize()

        # P0 offload: StorageManager allocates the compact Host Slot, invokes
        # compression, publishes the Host backing, then releases the raw GPU
        # Slot.  The Page itself records only its new level and slot id.
        storage._batched_migrate(
            pool_group,
            CACHE_LEVEL1,
            GPU_LEVEL,
            [page],
            update_src=True,
        )
        page.ready_event.synchronize()
        assert page.cache_level == CACHE_LEVEL1
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial - 1

        # Prove that release was not merely a logical counter update: the next
        # allocation reuses the same physical GPU slot.  Boundary allocation
        # also clears its stale bytes before returning it to a runtime writer.
        # This deliberately destroys the original raw KV, so the later recall
        # can succeed only from the compact Host backing.
        recycled = storage.new_gpu_slots([1])[life_cycle][0]
        recycled.ready_event.synchronize()
        assert recycled.slot_id == source_slot_id
        assert torch.count_nonzero(buffers[recycled.slot_id]).item() == 0
        storage.release_slot(life_cycle, GPU_LEVEL, recycled)
        recycled = None

        # P0 onboard: StorageManager first admits a normal raw GPU Slot, asks
        # the manager to decompress directly into it, and only then publishes
        # the GPU mapping and releases the compact Host Slot.
        storage._batched_migrate(
            pool_group,
            GPU_LEVEL,
            CACHE_LEVEL1,
            [page],
            update_src=True,
        )
        page.ready_event.synchronize()
        assert page.cache_level == GPU_LEVEL
        assert page.slot_id == source_slot_id
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial - 1
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial

        restored = buffers[page.slot_id].clone()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(restored, oracle, rtol=0, atol=0)
        torch.testing.assert_close(
            restored[:, _VALID_TOKENS:],
            torch.zeros_like(restored[:, _VALID_TOKENS:]),
            rtol=0,
            atol=0,
        )

        # Repeated cold/hot cycles re-quantize the previously restored Page.
        # Compare every cycle with a fresh independent oracle so layout or
        # lifetime bugs cannot hide behind the first successful round trip.
        for _ in range(9):
            expected = torch.stack(
                (
                    _direct_nvfp4_oracle(restored[0]),
                    _direct_nvfp4_oracle(restored[1]),
                )
            )
            torch.cuda.current_stream().synchronize()
            storage._batched_migrate(
                pool_group,
                CACHE_LEVEL1,
                GPU_LEVEL,
                [page],
                update_src=True,
            )
            page.ready_event.synchronize()
            assert page.cache_level == CACHE_LEVEL1
            storage._batched_migrate(
                pool_group,
                GPU_LEVEL,
                CACHE_LEVEL1,
                [page],
                update_src=True,
            )
            page.ready_event.synchronize()
            restored = buffers[page.slot_id].clone()
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    finally:
        # Release every directly allocated Slot before StorageManager teardown;
        # this also keeps its allocator invariants meaningful when an assertion
        # above exposes a partially completed migration.
        if recycled is not None and recycled.has_valid_slot:
            recycled.ready_event.synchronize()
            storage.release_slot(life_cycle, GPU_LEVEL, recycled)
        if page is not None and page.has_valid_slot:
            page.ready_event.synchronize()
            storage.release_slot(life_cycle, page.cache_level, page)

    assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
    assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial


def test_real_pressure_path_requeues_failed_victim_and_prefetches(p0_manager) -> None:
    """Exercise admission pressure, eviction, compression, and direct recall."""
    manager = p0_manager
    storage = manager.impl._storage
    boundary = manager._boundary_compression_manager
    assert boundary is not None
    life_cycle = LifeCycleId(0)
    pool_group = storage.get_pool_group_index(life_cycle)
    page = None
    holder = None
    replacement = None

    # GPU virtual-memory pools grow in a 2 MiB physical granule, so the tiny
    # logical fixture still starts with thousands of 512-byte Slots. Shrink it
    # before any allocation to create a deterministic one-Slot pressure case.
    initial_stats = storage.get_statistics(GPU_LEVEL)[pool_group]
    if initial_stats.total > 1:
        storage.shrink_pool_group(GPU_LEVEL, pool_group, 1, [])
    gpu_free_initial = storage.get_statistics(GPU_LEVEL)[pool_group].free
    host_free_initial = storage.get_statistics(CACHE_LEVEL1)[pool_group].free
    assert gpu_free_initial == 1

    try:
        raw_slot = storage.new_gpu_slots([1])[life_cycle][0]
        page = _UncommittedTestPage(
            _slot_id=None,
            ready_event=CachedCudaEvent.NULL,
            _manager=rawref.ref(storage),
            life_cycle=life_cycle,
            cache_level=GPU_LEVEL,
            _priority=PRIORITY_DEFAULT,
            _holder=None,
            node_ref=None,
        )
        page.set_slot(raw_slot)
        holder = page.hold()
        assert holder.page is page
        assert page.status == PageStatus.HELD
        page.ready_event.synchronize()
        source_slot_id = page.slot_id
        buffers = manager.get_buffers(0, kv_layout="NHD")
        assert buffers is not None
        buffers[source_slot_id].normal_()
        torch.cuda.current_stream().synchronize()

        # The only GPU slot is occupied by a HELD Page from a suspended
        # request. Requesting another slot executes the real controller path:
        # prepare_free_slots -> evict -> GPU-to-Host compression.
        storage.schedule_for_eviction(page)
        with patch.object(
            boundary,
            "compress_tensor",
            side_effect=torch.OutOfMemoryError("injected pressure-path OOM"),
        ):
            with pytest.raises(OutOfPagesError, match="workspace is unavailable"):
                storage.new_gpu_slots([1])

        # A failed transform neither moves the Page nor loses it from the LRU.
        assert page.cache_level == GPU_LEVEL
        assert page.slot_id == source_slot_id
        assert page.scheduled_for_eviction
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == 0
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial

        # Retrying admission compresses the victim to Host, releases its raw
        # GPU slot, and returns that physical slot to the requester.
        replacement = storage.new_gpu_slots([1])[life_cycle][0]
        replacement.ready_event.synchronize()
        assert page.cache_level == CACHE_LEVEL1
        assert page.status == PageStatus.HELD
        # HELD backing at the last level cannot be dropped, so it is not put
        # into the Host LRU after a successful offload.
        assert not page.scheduled_for_eviction
        assert replacement.slot_id == source_slot_id
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial - 1

        storage.release_slot(life_cycle, GPU_LEVEL, replacement)
        replacement = None

        # A hit may skip directly from Host to runtime GPU. Prefetch allocates
        # the raw destination, decompresses, commits cache_level plus slot_id
        # before exposing the Page, and releases the compressed Host backing.
        storage.prefetch(
            GPU_LEVEL,
            [[[], [page]]],
        )
        page.ready_event.synchronize()
        assert page.cache_level == GPU_LEVEL
        assert page.slot_id == source_slot_id
        assert page.status == PageStatus.HELD
        assert page.scheduled_for_eviction
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial
    finally:
        if replacement is not None and replacement.has_valid_slot:
            replacement.ready_event.synchronize()
            storage.release_slot(life_cycle, GPU_LEVEL, replacement)
        if page is not None:
            if page.scheduled_for_eviction:
                storage.exclude_from_eviction(page)
            holder = None
            if page.has_valid_slot:
                page.ready_event.synchronize()
                storage.release_slot(life_cycle, page.cache_level, page)

    assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
    assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial


def test_storage_manager_workspace_oom_rolls_back_and_retries(p0_manager) -> None:
    """Synchronous transform OOM becomes cache pressure and is retryable."""
    manager = p0_manager
    storage = manager.impl._storage
    boundary = manager._boundary_compression_manager
    assert boundary is not None
    life_cycle = LifeCycleId(0)
    pool_group = storage.get_pool_group_index(life_cycle)
    page = None
    gpu_probe = None

    gpu_free_initial = storage.get_statistics(GPU_LEVEL)[pool_group].free
    host_free_initial = storage.get_statistics(CACHE_LEVEL1)[pool_group].free

    try:
        raw_slot = storage.new_gpu_slots([1])[life_cycle][0]
        page = Page(
            _slot_id=None,
            ready_event=CachedCudaEvent.NULL,
            _manager=rawref.ref(storage),
            life_cycle=life_cycle,
            cache_level=GPU_LEVEL,
            _priority=PRIORITY_DEFAULT,
            _holder=None,
            node_ref=None,
        )
        page.set_slot(raw_slot)
        page.ready_event.synchronize()
        source_gpu_slot_id = page.slot_id

        buffers = manager.get_buffers(0, kv_layout="NHD")
        assert buffers is not None
        raw_page = buffers[source_gpu_slot_id]
        values = torch.arange(raw_page.numel(), dtype=torch.float32, device="cuda").view_as(
            raw_page
        )
        raw_page.copy_((((values * 5) % 53) - 26).to(raw_page.dtype) / 9)
        torch.cuda.current_stream().synchronize()
        source = raw_page.clone()
        oracle = torch.stack((_direct_nvfp4_oracle(source[0]), _direct_nvfp4_oracle(source[1])))
        torch.cuda.current_stream().synchronize()

        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial - 1
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial

        # The existing fp4 op still owns dynamic output tensors. Simulate an
        # OOM at that boundary: the manager translates it to KVCM pressure,
        # and StorageManager retains the authoritative raw source.
        with patch.object(
            boundary,
            "compress_tensor",
            side_effect=torch.OutOfMemoryError("injected fp4 output OOM"),
        ) as compress:
            with pytest.raises(OutOfPagesError, match="workspace is unavailable"):
                storage._batched_migrate(
                    pool_group,
                    CACHE_LEVEL1,
                    GPU_LEVEL,
                    [page],
                    update_src=True,
                )

        compress.assert_called_once()
        assert page.cache_level == GPU_LEVEL
        assert page.slot_id == source_gpu_slot_id
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial - 1
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial
        page.ready_event.synchronize()
        torch.testing.assert_close(buffers[source_gpu_slot_id], source, rtol=0, atol=0)

        # Removing the fault must make the same Page retryable without any
        # ownership repair by the caller.
        storage._batched_migrate(
            pool_group,
            CACHE_LEVEL1,
            GPU_LEVEL,
            [page],
            update_src=True,
        )
        page.ready_event.synchronize()
        assert page.cache_level == CACHE_LEVEL1
        host_source_slot_id = page.slot_id
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial - 1

        # StorageManager admits a raw GPU destination before calling the
        # decompression hook. A decoder OOM must return that destination
        # while retaining the compressed Host source as the Page's backing.
        with patch.object(
            boundary_compression_module,
            "dequant_nvfp4_2d_triton",
            side_effect=torch.OutOfMemoryError("injected decoder OOM"),
        ) as decompress:
            with pytest.raises(OutOfPagesError, match="workspace is unavailable"):
                storage._batched_migrate(
                    pool_group,
                    GPU_LEVEL,
                    CACHE_LEVEL1,
                    [page],
                    update_src=True,
                )

        decompress.assert_called_once()
        assert page.cache_level == CACHE_LEVEL1
        assert page.slot_id == host_source_slot_id
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial - 1
        page.ready_event.synchronize()

        # Allocate the just-rolled-back GPU destination to prove it returned
        # to the allocator, then release it before retrying the real onboard.
        gpu_probe = storage.new_gpu_slots([1])[life_cycle][0]
        gpu_probe.ready_event.synchronize()
        assert gpu_probe.slot_id == source_gpu_slot_id
        storage.release_slot(life_cycle, GPU_LEVEL, gpu_probe)
        gpu_probe = None
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial

        storage._batched_migrate(
            pool_group,
            GPU_LEVEL,
            CACHE_LEVEL1,
            [page],
            update_src=True,
        )
        page.ready_event.synchronize()
        assert page.cache_level == GPU_LEVEL
        assert page.slot_id == source_gpu_slot_id
        assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial - 1
        assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial

        restored = buffers[page.slot_id].clone()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(restored, oracle, rtol=0, atol=0)
    finally:
        if gpu_probe is not None and gpu_probe.has_valid_slot:
            gpu_probe.ready_event.synchronize()
            storage.release_slot(life_cycle, GPU_LEVEL, gpu_probe)
        if page is not None and page.has_valid_slot:
            page.ready_event.synchronize()
            storage.release_slot(life_cycle, page.cache_level, page)

    assert storage.get_statistics(GPU_LEVEL)[pool_group].free == gpu_free_initial
    assert storage.get_statistics(CACHE_LEVEL1)[pool_group].free == host_free_initial
