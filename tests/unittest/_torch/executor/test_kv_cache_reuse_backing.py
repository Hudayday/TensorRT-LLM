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

from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache_reuse_backing import ReuseBackingState
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import Slot, SlotAllocator

_PAGE_KEY = (0, 17)


class _FakeEvent:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class _FailingEvent:
    def query(self) -> bool:
        raise RuntimeError("event query failed")


class _SourceReaderGate:
    def __init__(self, drained: bool = True) -> None:
        self.drained = drained

    def __call__(self) -> bool:
        return self.drained


class _KVCMReuseContractHarness(KVCacheManagerV2):
    """CPU harness that executes the production V2 wrapper contract.

    It intentionally bypasses the full CUDA-backed constructor, but remains an
    actual ``KVCacheManagerV2`` instance so the compression manager's ownership
    type guard is exercised.
    """

    def __init__(self, compressed_capacity_bytes: int) -> None:
        self.enable_block_reuse = True
        self.kv_compression_manages_history = False
        self.is_draft = False
        self.is_disagg = False
        self.dtype = DataType.BF16
        self.tokens_per_block = 4
        self._reuse_compression_manager = None
        self._reuse_backing_store = None

        self.manager = QuantizationForBoundaryCompression(
            self,
            quant="nvfp4",
            compressed_capacity_bytes=compressed_capacity_bytes,
        )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _payload_nbytes(payload: tuple[torch.Tensor, ...]) -> int:
    return sum(_tensor_nbytes(tensor) for tensor in payload)


def _raw_payloads() -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(1, 129, dtype=torch.bfloat16).reshape(4, 32),
        torch.arange(129, 257, dtype=torch.bfloat16).reshape(4, 32),
    )


def _compressed_payload(raw_payload: torch.Tensor) -> tuple[torch.Tensor, ...]:
    feature_count = raw_payload.shape[-1]
    row_count = raw_payload.numel() // feature_count
    return (
        torch.zeros(
            (*raw_payload.shape[:-1], feature_count // 2),
            dtype=torch.uint8,
        ),
        torch.zeros(
            (row_count, feature_count // 16),
            dtype=torch.uint8,
        ),
        torch.ones(1, dtype=torch.float32),
    )


def _new_source_slot() -> tuple[SlotAllocator, Slot]:
    allocator = SlotAllocator(1)
    slot = allocator.allocate()
    assert allocator.num_free_slots == 0
    return allocator, slot


def _destroy_allocator(allocator: SlotAllocator) -> None:
    assert allocator.num_occupied_slots == 0
    allocator.prepare_for_shrink(0)
    allocator.finish_shrink()


def _store_compressed_page(
    target: _KVCMReuseContractHarness,
    manager: QuantizationForBoundaryCompression,
    *,
    source_gate: _SourceReaderGate | None = None,
    compression_event: _FakeEvent | None = None,
) -> tuple[int, SlotAllocator, Slot]:
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    compressed_size_bytes = sum(_payload_nbytes(payload) for payload in compressed_payloads)
    manager.on_reuse_store = MagicMock(side_effect=compressed_payloads)
    source_allocator, source_slot = _new_source_slot()
    if source_gate is None:
        source_gate = _SourceReaderGate()

    assert target.try_store_compressed_reuse_backing(
        _PAGE_KEY,
        raw_payloads,
        valid_token_count=4,
        can_publish_compressed=source_gate,
        publish_compressed_and_release_source=lambda: source_allocator.release(
            source_slot
        ),
        drop_reuse_candidate=MagicMock(),
        completion_event_factory=(
            None if compression_event is None else lambda: compression_event
        ),
    )
    return compressed_size_bytes, source_allocator, source_slot


def test_compression_retires_source_slot_contract_and_has_no_stable_raw_state() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    manager = target.manager
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    compressed_size_bytes = sum(_payload_nbytes(payload) for payload in compressed_payloads)
    manager.on_reuse_store = MagicMock(side_effect=compressed_payloads)
    compression_event = _FakeEvent()
    source_gate = _SourceReaderGate(drained=False)
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock()

    assert target.try_store_compressed_reuse_backing(
        _PAGE_KEY,
        raw_payloads,
        valid_token_count=4,
        can_publish_compressed=source_gate,
        publish_compressed_and_release_source=lambda: source_allocator.release(
            source_slot
        ),
        drop_reuse_candidate=drop_reuse_candidate,
        completion_event_factory=lambda: compression_event,
    )

    compressing = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert compressing.state is ReuseBackingState.COMPRESSING
    assert compressing.compressed_resident_bytes == compressed_size_bytes
    assert compressing.compression_source_leases == 1
    assert not hasattr(compressing, "raw_resident_bytes")
    assert source_allocator.num_free_slots == 0
    assert not target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())

    compression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    assert source_allocator.num_free_slots == 0

    source_gate.drained = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    compressed = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert compressed.state is ReuseBackingState.COMPRESSED
    assert compressed.compression_source_leases == 0
    assert source_allocator.num_free_slots == 1
    drop_reuse_candidate.assert_not_called()

    unpublish = MagicMock()
    assert target.evict_compressed_reuse_backing(_PAGE_KEY, unpublish)
    unpublish.assert_called_once_with()
    with pytest.raises(KeyError, match="Unknown reuse backing page"):
        target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    assert target._reuse_backing_store.page_count == 0
    _destroy_allocator(source_allocator)


def test_capacity_miss_drops_reuse_candidate_without_creating_raw_backing() -> None:
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    compressed_size_bytes = sum(_payload_nbytes(payload) for payload in compressed_payloads)
    target = _KVCMReuseContractHarness(
        compressed_capacity_bytes=compressed_size_bytes - 1
    )
    target.manager.on_reuse_store = MagicMock(side_effect=compressed_payloads)
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock()

    assert not target.try_store_compressed_reuse_backing(
        _PAGE_KEY,
        raw_payloads,
        valid_token_count=4,
        can_publish_compressed=_SourceReaderGate(),
        publish_compressed_and_release_source=lambda: source_allocator.release(
            source_slot
        ),
        drop_reuse_candidate=drop_reuse_candidate,
    )

    drop_reuse_candidate.assert_called_once_with()
    with pytest.raises(KeyError, match="Unknown reuse backing page"):
        target.get_reuse_backing_snapshot(_PAGE_KEY)
    pool = target.get_reuse_backing_pool_snapshot()
    assert pool.used_bytes == 0
    assert pool.allocation_count == 0
    # The active request still owns its raw source; it is not a cold backing.
    assert source_allocator.num_free_slots == 0
    source_allocator.release(source_slot)
    _destroy_allocator(source_allocator)


def test_capacity_miss_defers_source_drop_until_compression_event() -> None:
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    compressed_size_bytes = sum(_payload_nbytes(payload) for payload in compressed_payloads)
    target = _KVCMReuseContractHarness(
        compressed_capacity_bytes=compressed_size_bytes - 1
    )
    target.manager.on_reuse_store = MagicMock(side_effect=compressed_payloads)
    compression_event = _FakeEvent()
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock(
        side_effect=lambda: source_allocator.release(source_slot)
    )

    assert not target.try_store_compressed_reuse_backing(
        _PAGE_KEY,
        raw_payloads,
        valid_token_count=4,
        can_publish_compressed=_SourceReaderGate(),
        publish_compressed_and_release_source=MagicMock(),
        drop_reuse_candidate=drop_reuse_candidate,
        completion_event_factory=lambda: compression_event,
    )

    cancelling = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert cancelling.state is ReuseBackingState.CANCELLING
    assert cancelling.compression_source_leases == 1
    # The already-created transform outputs are transient workspace and may
    # temporarily exceed stable compressed admission capacity.
    assert target.get_reuse_backing_pool_snapshot().used_bytes == compressed_size_bytes
    drop_reuse_candidate.assert_not_called()
    assert source_allocator.num_free_slots == 0

    compression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    drop_reuse_candidate.assert_called_once_with()
    assert source_allocator.num_free_slots == 1
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_transform_rejection_drops_partial_page_instead_of_retaining_raw() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    target.manager.on_reuse_store = MagicMock(return_value=None)
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock()

    assert not target.try_store_compressed_reuse_backing(
        _PAGE_KEY,
        _raw_payloads(),
        valid_token_count=3,
        can_publish_compressed=_SourceReaderGate(),
        publish_compressed_and_release_source=lambda: source_allocator.release(
            source_slot
        ),
        drop_reuse_candidate=drop_reuse_candidate,
    )

    drop_reuse_candidate.assert_called_once_with()
    with pytest.raises(KeyError, match="Unknown reuse backing page"):
        target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert source_allocator.num_free_slots == 0
    source_allocator.release(source_slot)
    _destroy_allocator(source_allocator)


def test_transform_failure_defers_source_drop_until_prior_launches_finish() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    target.manager.on_reuse_store = MagicMock(
        side_effect=[
            _compressed_payload(raw_payloads[0]),
            RuntimeError("second compression failed"),
        ]
    )
    compression_event = _FakeEvent()
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock(
        side_effect=lambda: source_allocator.release(source_slot)
    )

    with pytest.raises(RuntimeError, match="second compression failed"):
        target.try_store_compressed_reuse_backing(
            _PAGE_KEY,
            raw_payloads,
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=MagicMock(),
            drop_reuse_candidate=drop_reuse_candidate,
            completion_event_factory=lambda: compression_event,
        )

    failed_transform = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert failed_transform.state is ReuseBackingState.CANCELLING
    assert failed_transform.compression_source_leases == 1
    drop_reuse_candidate.assert_not_called()
    assert source_allocator.num_free_slots == 0

    compression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    drop_reuse_candidate.assert_called_once_with()
    assert source_allocator.num_free_slots == 1
    _destroy_allocator(source_allocator)


def test_duplicate_generation_fails_before_any_compression_hook() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    _, source_allocator, _ = _store_compressed_page(target, target.manager)
    target.manager.on_reuse_store = MagicMock()
    drop_reuse_candidate = MagicMock()

    with pytest.raises(ValueError, match="already exists"):
        target.try_store_compressed_reuse_backing(
            _PAGE_KEY,
            _raw_payloads(),
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=MagicMock(),
            drop_reuse_candidate=drop_reuse_candidate,
        )

    target.manager.on_reuse_store.assert_not_called()
    drop_reuse_candidate.assert_not_called()
    assert target.get_reuse_backing_snapshot(_PAGE_KEY).state is ReuseBackingState.COMPRESSED
    assert target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())
    _destroy_allocator(source_allocator)


def test_empty_compression_result_uses_deferred_abort_transaction() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    target.manager.on_reuse_store = MagicMock(
        return_value=(torch.empty(0, dtype=torch.uint8),)
    )
    compression_event = _FakeEvent()
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock(
        side_effect=lambda: source_allocator.release(source_slot)
    )

    with pytest.raises(ValueError, match="empty payload"):
        target.try_store_compressed_reuse_backing(
            _PAGE_KEY,
            _raw_payloads(),
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=MagicMock(),
            drop_reuse_candidate=drop_reuse_candidate,
            completion_event_factory=lambda: compression_event,
        )

    cancelling = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert cancelling.state is ReuseBackingState.CANCELLING
    assert cancelling.compressed_resident_bytes == 0
    assert cancelling.compression_source_leases == 1
    drop_reuse_candidate.assert_not_called()

    compression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    drop_reuse_candidate.assert_called_once_with()
    assert source_allocator.num_free_slots == 1
    _destroy_allocator(source_allocator)


def test_invalid_source_contract_drops_candidate_once() -> None:
    bad_inputs = (
        (),
        (torch.ones((3, 32), dtype=torch.bfloat16),),
        (torch.ones((32, 4), dtype=torch.bfloat16).transpose(0, 1),),
    )
    for generation, raw_payloads in enumerate(bad_inputs, start=100):
        target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
        drop_reuse_candidate = MagicMock()
        with pytest.raises(ValueError):
            target.try_store_compressed_reuse_backing(
                (0, generation),
                raw_payloads,
                valid_token_count=4,
                can_publish_compressed=_SourceReaderGate(),
                publish_compressed_and_release_source=MagicMock(),
                drop_reuse_candidate=drop_reuse_candidate,
            )
        drop_reuse_candidate.assert_called_once_with()
        assert target._reuse_backing_store.page_count == 0
        assert target.get_reuse_backing_pool_snapshot().used_bytes == 0


def test_completion_event_factory_runs_after_every_compression_hook() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    hook_calls = 0

    def compress(raw_payload, **kwargs):
        nonlocal hook_calls
        del raw_payload, kwargs
        result = compressed_payloads[hook_calls]
        hook_calls += 1
        return result

    completion_event = _FakeEvent()

    def record_after_launches():
        assert hook_calls == len(raw_payloads)
        return completion_event

    target.manager.on_reuse_store = MagicMock(side_effect=compress)
    assert target.try_store_compressed_reuse_backing(
        (0, 120),
        raw_payloads,
        valid_token_count=4,
        can_publish_compressed=_SourceReaderGate(),
        publish_compressed_and_release_source=MagicMock(),
        drop_reuse_candidate=MagicMock(),
        completion_event_factory=record_after_launches,
    )
    assert hook_calls == len(raw_payloads)
    assert target.get_reuse_backing_snapshot((0, 120)).state is ReuseBackingState.COMPRESSING


def test_event_query_failure_quarantines_transaction_until_owner_reconciles() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    compressed_payloads = tuple(_compressed_payload(payload) for payload in raw_payloads)
    compressed_size_bytes = sum(_payload_nbytes(payload) for payload in compressed_payloads)
    target.manager.on_reuse_store = MagicMock(side_effect=compressed_payloads)
    source_allocator, source_slot = _new_source_slot()
    publish = MagicMock()
    drop_reuse_candidate = MagicMock()

    with pytest.raises(RuntimeError, match="event query failed"):
        target.try_store_compressed_reuse_backing(
            (0, 119),
            raw_payloads,
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=publish,
            drop_reuse_candidate=drop_reuse_candidate,
            completion_event_factory=_FailingEvent,
        )

    failed = target.get_reuse_backing_snapshot((0, 119))
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compressed_resident_bytes == compressed_size_bytes
    assert failed.compression_source_leases == 1
    assert target.poll_reuse_backing((0, 119)) == 0
    publish.assert_not_called()
    drop_reuse_candidate.assert_not_called()
    assert not target.reconcile_failed_reuse_backing((0, 119), lambda: False)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == compressed_size_bytes

    # Native V2 owns cleanup; only a read-only confirmation may release the
    # quarantined compact allocation.
    source_allocator.release(source_slot)
    assert target.reconcile_failed_reuse_backing((0, 119), lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_atomic_publish_callback_is_at_most_once_on_failure() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    target.manager.on_reuse_store = MagicMock(
        side_effect=tuple(_compressed_payload(payload) for payload in raw_payloads)
    )
    source_allocator, source_slot = _new_source_slot()
    publish_count = 0

    def publish_then_fail() -> None:
        nonlocal publish_count
        publish_count += 1
        assert target.poll_reuse_backing((0, 121)) == 0
        source_allocator.release(source_slot)
        raise RuntimeError("native atomic publish failed")

    drop_reuse_candidate = MagicMock()
    with pytest.raises(RuntimeError, match="atomic publish failed"):
        target.try_store_compressed_reuse_backing(
            (0, 121),
            raw_payloads,
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=publish_then_fail,
            drop_reuse_candidate=drop_reuse_candidate,
        )

    assert publish_count == 1
    assert source_allocator.num_free_slots == 1
    drop_reuse_candidate.assert_not_called()
    assert target.poll_reuse_backing((0, 121)) == 0
    failed = target.get_reuse_backing_snapshot((0, 121))
    assert failed.state is ReuseBackingState.FAILED
    assert publish_count == 1
    assert target.get_reuse_backing_pool_snapshot().used_bytes != 0
    assert target.reconcile_failed_reuse_backing((0, 121), lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_compression_event_record_double_failure_quarantines_source_ownership() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    target.manager.on_reuse_store = MagicMock(
        side_effect=[
            _compressed_payload(raw_payloads[0]),
            RuntimeError("compression failed"),
        ]
    )
    source_allocator, source_slot = _new_source_slot()
    drop_reuse_candidate = MagicMock()

    with (
        patch.object(
            KVCacheManagerV2,
            "_record_reuse_completion_event",
            side_effect=RuntimeError("event record failed"),
        ),
        pytest.raises(RuntimeError, match="event record failed"),
    ):
        target.try_store_compressed_reuse_backing(
            (0, 122),
            raw_payloads,
            valid_token_count=4,
            can_publish_compressed=_SourceReaderGate(),
            publish_compressed_and_release_source=MagicMock(),
            drop_reuse_candidate=drop_reuse_candidate,
        )

    failed = target.get_reuse_backing_snapshot((0, 122))
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compression_source_leases == 1
    drop_reuse_candidate.assert_not_called()
    assert source_allocator.num_free_slots == 0

    source_allocator.release(source_slot)
    assert target.reconcile_failed_reuse_backing((0, 122), lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_cancelled_compression_waits_for_gpu_work_then_reclaims_capacity() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compression_event = _FakeEvent()
    compressed_size_bytes, source_allocator, source_slot = _store_compressed_page(
        target,
        target.manager,
        compression_event=compression_event,
    )
    assert source_allocator.num_free_slots == 0
    cancel_native_transaction = MagicMock(
        side_effect=lambda: source_allocator.release(source_slot)
    )

    assert not target.cancel_compressed_reuse_store(
        _PAGE_KEY,
        cancel_native_transaction,
    )
    cancelling = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert cancelling.state is ReuseBackingState.CANCELLING
    assert cancelling.compressed_resident_bytes == compressed_size_bytes
    cancel_native_transaction.assert_not_called()

    compression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    cancel_native_transaction.assert_called_once_with()
    with pytest.raises(KeyError, match="Unknown reuse backing page"):
        target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    # The native cancellation callback, not the compressed store, decides
    # when the full-precision source slot can be released.
    assert source_allocator.num_free_slots == 1
    _destroy_allocator(source_allocator)


def test_cancel_failure_is_quarantined_and_never_retried() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compression_event = _FakeEvent()
    compressed_size_bytes, source_allocator, source_slot = _store_compressed_page(
        target,
        target.manager,
        compression_event=compression_event,
    )
    cancel_count = 0

    def cancel_then_fail() -> None:
        nonlocal cancel_count
        cancel_count += 1
        assert target.poll_reuse_backing(_PAGE_KEY) == 0
        raise RuntimeError("native cancel failed")

    assert not target.cancel_compressed_reuse_store(_PAGE_KEY, cancel_then_fail)
    compression_event.ready = True
    with pytest.raises(RuntimeError, match="native cancel failed"):
        target.poll_reuse_backing(_PAGE_KEY)

    failed = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compressed_resident_bytes == compressed_size_bytes
    assert failed.compression_source_leases == 1
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    assert cancel_count == 1

    source_allocator.release(source_slot)
    assert target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_decompression_failure_rolls_back_raw_admission_and_keeps_compressed() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    assert source_allocator.num_free_slots == 1

    raw_destinations = tuple(torch.empty_like(payload) for payload in _raw_payloads())
    target.manager.on_reuse_materialize = MagicMock(
        side_effect=[None, RuntimeError("decompression failed")]
    )
    rollback_raw_slot = MagicMock()
    rollback_event = _FakeEvent()

    with pytest.raises(RuntimeError, match="decompression failed"):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            raw_destinations,
            publish_raw_slot=MagicMock(),
            rollback_raw_slot=rollback_raw_slot,
            completion_event_factory=lambda: rollback_event,
        )

    rollback_raw_slot.assert_not_called()
    pending = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert pending.state is ReuseBackingState.COMPRESSED
    assert pending.compressed_resident_bytes == compressed_size_bytes
    assert pending.compressed_read_leases == 1
    assert pending.pending_decompressions == 1
    assert not target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())

    rollback_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    rollback_raw_slot.assert_called_once_with(rollback_event)
    complete = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert complete.compressed_read_leases == 0
    assert complete.pending_decompressions == 0
    _destroy_allocator(source_allocator)


def test_mixed_destination_devices_fail_before_any_decompression_launch() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    rollback_raw_slot = MagicMock()
    destinations = (
        torch.empty_like(_raw_payloads()[0]),
        torch.empty(_raw_payloads()[1].shape, device="meta", dtype=torch.bfloat16),
    )

    with pytest.raises(ValueError, match="share a device"):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            destinations,
            publish_raw_slot=MagicMock(),
            rollback_raw_slot=rollback_raw_slot,
        )

    target.manager.on_reuse_materialize.assert_not_called()
    rollback_raw_slot.assert_called_once_with(None)
    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.COMPRESSED
    assert snapshot.compressed_resident_bytes == compressed_size_bytes
    assert snapshot.compressed_read_leases == 0
    _destroy_allocator(source_allocator)


def test_decompression_publish_waits_for_event_and_blocks_compressed_eviction() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    decompression_event = _FakeEvent()
    publish_raw_slot = MagicMock()
    rollback_raw_slot = MagicMock()

    decompression_id = target.decompress_reuse_backing(
        _PAGE_KEY,
        tuple(torch.empty_like(payload) for payload in _raw_payloads()),
        publish_raw_slot=publish_raw_slot,
        rollback_raw_slot=rollback_raw_slot,
        completion_event_factory=lambda: decompression_event,
    )
    assert decompression_id == 0

    pending = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert pending.state is ReuseBackingState.COMPRESSED
    assert pending.compressed_read_leases == 1
    assert pending.pending_decompressions == 1
    assert not target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())
    publish_raw_slot.assert_not_called()

    decompression_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    publish_raw_slot.assert_called_once_with()
    rollback_raw_slot.assert_not_called()
    ready = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert ready.compressed_resident_bytes == compressed_size_bytes
    assert ready.compressed_read_leases == 0
    assert ready.pending_decompressions == 0

    assert target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())
    _destroy_allocator(source_allocator)


def test_decompression_publish_is_consumed_before_reentrant_poll() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    _, source_allocator, _ = _store_compressed_page(target, target.manager)
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    publish_count = 0

    def publish_with_reentrant_poll() -> None:
        nonlocal publish_count
        publish_count += 1
        assert target.poll_reuse_backing(_PAGE_KEY) == 0
        assert not target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())

    target.decompress_reuse_backing(
        _PAGE_KEY,
        tuple(torch.empty_like(payload) for payload in _raw_payloads()),
        publish_raw_slot=publish_with_reentrant_poll,
        rollback_raw_slot=MagicMock(),
    )

    assert publish_count == 1
    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.compressed_read_leases == 0
    assert snapshot.pending_decompressions == 0
    _destroy_allocator(source_allocator)


def test_decompression_event_query_failure_quarantines_pending_raw_reservation() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    failing_event = _FailingEvent()
    rollback_raw_slot = MagicMock()

    with pytest.raises(RuntimeError, match="event query failed"):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            tuple(torch.empty_like(payload) for payload in _raw_payloads()),
            publish_raw_slot=MagicMock(),
            rollback_raw_slot=rollback_raw_slot,
            completion_event_factory=lambda: failing_event,
        )

    failed = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compressed_resident_bytes == compressed_size_bytes
    assert failed.compressed_read_leases == 1
    assert failed.pending_decompressions == 1
    rollback_raw_slot.assert_not_called()

    # The owner first cleans the raw reservation out-of-band, then supplies a
    # read-only confirmation that both GPU and native state are reconciled.
    rollback_raw_slot(failing_event)
    assert target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_decompression_event_record_double_failure_quarantines_both_reservations() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(
        side_effect=RuntimeError("decompression failed")
    )
    rollback_raw_slot = MagicMock()

    with (
        patch.object(
            KVCacheManagerV2,
            "_record_reuse_completion_event",
            side_effect=RuntimeError("event record failed"),
        ),
        pytest.raises(RuntimeError, match="event record failed"),
    ):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            tuple(torch.empty_like(payload) for payload in _raw_payloads()),
            publish_raw_slot=MagicMock(),
            rollback_raw_slot=rollback_raw_slot,
        )

    failed = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compressed_resident_bytes == compressed_size_bytes
    assert failed.compressed_read_leases == 1
    assert failed.pending_decompressions == 1
    rollback_raw_slot.assert_not_called()

    rollback_raw_slot(None)
    assert target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_raw_publish_failure_rolls_back_once_and_preserves_compressed_source() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    rollback_raw_slot = MagicMock()

    with pytest.raises(RuntimeError, match="publish failed"):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            tuple(torch.empty_like(payload) for payload in _raw_payloads()),
            publish_raw_slot=MagicMock(side_effect=RuntimeError("publish failed")),
            rollback_raw_slot=rollback_raw_slot,
        )

    rollback_raw_slot.assert_called_once_with(None)
    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.COMPRESSED
    assert snapshot.compressed_resident_bytes == compressed_size_bytes
    assert snapshot.compressed_read_leases == 0
    assert snapshot.pending_decompressions == 0
    _destroy_allocator(source_allocator)


def test_raw_publish_and_rollback_double_failure_is_quarantined() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    target.manager.on_reuse_materialize = MagicMock(return_value=None)
    rollback_raw_slot = MagicMock(side_effect=RuntimeError("rollback failed"))

    with pytest.raises(RuntimeError, match="rollback failed"):
        target.decompress_reuse_backing(
            _PAGE_KEY,
            tuple(torch.empty_like(payload) for payload in _raw_payloads()),
            publish_raw_slot=MagicMock(side_effect=RuntimeError("publish failed")),
            rollback_raw_slot=rollback_raw_slot,
        )

    rollback_raw_slot.assert_called_once_with(None)
    failed = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert failed.state is ReuseBackingState.FAILED
    assert failed.compressed_resident_bytes == compressed_size_bytes
    assert failed.compressed_read_leases == 1
    assert failed.pending_decompressions == 1
    assert target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: True)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)


def test_eviction_failure_is_quarantined_after_unknown_native_side_effect() -> None:
    target = _KVCMReuseContractHarness(compressed_capacity_bytes=1024)
    compressed_size_bytes, source_allocator, _ = _store_compressed_page(
        target, target.manager
    )
    unpublish_count = 0

    def unpublish_then_fail() -> None:
        nonlocal unpublish_count
        unpublish_count += 1
        raise RuntimeError("unpublish failed")

    with pytest.raises(RuntimeError, match="unpublish failed"):
        target.evict_compressed_reuse_backing(
            _PAGE_KEY,
            unpublish_then_fail,
        )

    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.FAILED
    assert snapshot.compressed_resident_bytes == compressed_size_bytes
    assert target.get_reuse_backing_pool_snapshot().used_bytes == compressed_size_bytes
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    with pytest.raises(RuntimeError, match="requires COMPRESSED"):
        target.evict_compressed_reuse_backing(_PAGE_KEY, MagicMock())
    assert unpublish_count == 1

    assert not target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: False)
    assert target.reconcile_failed_reuse_backing(_PAGE_KEY, lambda: True)
    assert target._reuse_backing_store.page_count == 0
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    _destroy_allocator(source_allocator)
