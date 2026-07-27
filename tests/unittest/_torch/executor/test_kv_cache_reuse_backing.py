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

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache_reuse_backing import ReuseBackingState
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType

_PAGE_KEY = (0, 17)


class _FakeEvent:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class _ByteLedger:
    def __init__(self) -> None:
        self.used_bytes = 0
        self.release_count = 0

    def reserve(self, size_bytes: int) -> None:
        self.used_bytes += size_bytes

    def release(self, size_bytes: int) -> None:
        if size_bytes > self.used_bytes:
            raise RuntimeError("Byte ledger underflow")
        self.used_bytes -= size_bytes
        self.release_count += 1


class _RawReaderGate:
    def __init__(self, drained: bool = True) -> None:
        self.drained = drained

    def __call__(self) -> bool:
        return self.drained


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _payload_nbytes(payload: tuple[torch.Tensor, ...]) -> int:
    return sum(_tensor_nbytes(tensor) for tensor in payload)


def _raw_payloads() -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(1, 129, dtype=torch.bfloat16).reshape(4, 32),
        torch.arange(129, 257, dtype=torch.bfloat16).reshape(4, 32),
    )


def _encode_payload(raw_payload: torch.Tensor) -> tuple[torch.Tensor, ...]:
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


class _KVCMReuseContractHarness:
    """CPU harness that executes the production KVCM reuse methods.

    It models only the explicit reuse-boundary contract and does not pretend to
    be a partially constructed ``KVCacheManagerV2``.
    """

    bind_reuse_compression_hooks = KVCacheManagerV2.bind_reuse_compression_hooks
    configure_reuse_encoded_backing = KVCacheManagerV2.configure_reuse_encoded_backing
    register_reuse_raw_backing = KVCacheManagerV2.register_reuse_raw_backing
    try_encode_reuse_backing = KVCacheManagerV2.try_encode_reuse_backing
    materialize_reuse_backing = KVCacheManagerV2.materialize_reuse_backing
    poll_reuse_backing = KVCacheManagerV2.poll_reuse_backing
    evict_reuse_encoded_backing = KVCacheManagerV2.evict_reuse_encoded_backing
    get_reuse_backing_snapshot = KVCacheManagerV2.get_reuse_backing_snapshot
    get_reuse_backing_pool_snapshot = KVCacheManagerV2.get_reuse_backing_pool_snapshot
    _require_reuse_compression_manager = KVCacheManagerV2._require_reuse_compression_manager
    _require_reuse_backing_store = KVCacheManagerV2._require_reuse_backing_store
    _record_reuse_completion_event = staticmethod(KVCacheManagerV2._record_reuse_completion_event)

    def __init__(self, encoded_capacity_bytes: int) -> None:
        self.enable_block_reuse = True
        self.kv_compression_manages_history = False
        self.is_draft = False
        self.is_disagg = False
        self.dtype = DataType.BF16
        self.tokens_per_block = 4
        self._reuse_compression_manager = None
        self._reuse_backing_store = None

        self.manager = QuantizationForBoundaryCompression(self, quant="nvfp4")
        self.configure_reuse_encoded_backing(encoded_capacity_bytes)


def _make_manager(
    encoded_capacity_bytes: int,
) -> tuple[_KVCMReuseContractHarness, QuantizationForBoundaryCompression]:
    target = _KVCMReuseContractHarness(encoded_capacity_bytes)
    return target, target.manager


def _register_raw_page(
    target: _KVCMReuseContractHarness,
    raw_payloads: tuple[torch.Tensor, ...],
    raw_ledger: _ByteLedger,
    raw_reader_gate: _RawReaderGate | None = None,
) -> int:
    if raw_reader_gate is None:
        raw_reader_gate = _RawReaderGate()
    raw_size_bytes = sum(_tensor_nbytes(payload) for payload in raw_payloads)
    raw_ledger.reserve(raw_size_bytes)
    target.register_reuse_raw_backing(
        _PAGE_KEY,
        raw_payloads,
        can_release_raw=raw_reader_gate,
        release_raw_slot=lambda: raw_ledger.release(raw_size_bytes),
    )
    return raw_size_bytes


def _publish_encoded_page(
    target: _KVCMReuseContractHarness,
    manager: QuantizationForBoundaryCompression,
    raw_payloads: tuple[torch.Tensor, ...],
    raw_ledger: _ByteLedger,
) -> int:
    _register_raw_page(target, raw_payloads, raw_ledger)
    encoded_payloads = tuple(_encode_payload(payload) for payload in raw_payloads)
    manager.on_reuse_store = MagicMock(side_effect=encoded_payloads)
    assert target.try_encode_reuse_backing(
        _PAGE_KEY,
        valid_token_count=4,
    )
    return sum(_payload_nbytes(payload) for payload in encoded_payloads)


def test_raw_dual_encoded_releases_raw_slot_and_evicts_compact_capacity() -> None:
    target, manager = _make_manager(encoded_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    raw_ledger = _ByteLedger()
    raw_reader_gate = _RawReaderGate(drained=False)
    raw_size_bytes = _register_raw_page(
        target,
        raw_payloads,
        raw_ledger,
        raw_reader_gate,
    )
    encoded_payloads = tuple(_encode_payload(payload) for payload in raw_payloads)
    encoded_size_bytes = sum(_payload_nbytes(payload) for payload in encoded_payloads)
    manager.on_reuse_store = MagicMock(side_effect=encoded_payloads)
    encode_event = _FakeEvent()

    raw_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert raw_snapshot.state is ReuseBackingState.RAW
    assert raw_snapshot.raw_resident_bytes == raw_size_bytes
    assert raw_snapshot.encoded_resident_bytes == 0

    assert target.try_encode_reuse_backing(
        _PAGE_KEY,
        valid_token_count=4,
        completion_event=encode_event,
    )
    dual_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert dual_snapshot.state is ReuseBackingState.DUAL
    assert dual_snapshot.raw_resident_bytes == raw_size_bytes
    assert dual_snapshot.encoded_resident_bytes == encoded_size_bytes
    assert dual_snapshot.encoding_read_leases == 1
    assert raw_ledger.used_bytes == raw_size_bytes
    assert target.get_reuse_backing_pool_snapshot().used_bytes == encoded_size_bytes
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    assert not target.evict_reuse_encoded_backing(_PAGE_KEY)
    assert raw_ledger.used_bytes == raw_size_bytes

    encode_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    assert target.get_reuse_backing_snapshot(_PAGE_KEY).state is ReuseBackingState.DUAL
    assert raw_ledger.used_bytes == raw_size_bytes
    assert not target.evict_reuse_encoded_backing(_PAGE_KEY)

    raw_reader_gate.drained = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    encoded_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert encoded_snapshot.state is ReuseBackingState.ENCODED
    assert encoded_snapshot.raw_resident_bytes == 0
    assert encoded_snapshot.encoded_resident_bytes == encoded_size_bytes
    assert encoded_snapshot.encoding_read_leases == 0
    assert raw_ledger.used_bytes == 0
    assert raw_ledger.release_count == 1
    assert target.poll_reuse_backing(_PAGE_KEY) == 0
    assert raw_ledger.release_count == 1

    assert target.evict_reuse_encoded_backing(_PAGE_KEY)
    evicted_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert evicted_snapshot.state is ReuseBackingState.EVICTED
    assert evicted_snapshot.encoded_resident_bytes == 0
    pool_snapshot = target.get_reuse_backing_pool_snapshot()
    assert pool_snapshot.used_bytes == 0
    assert pool_snapshot.allocation_count == 0


def test_encoded_pool_admission_failure_keeps_raw_backing() -> None:
    raw_payloads = _raw_payloads()
    encoded_payloads = tuple(_encode_payload(payload) for payload in raw_payloads)
    encoded_size_bytes = sum(_payload_nbytes(payload) for payload in encoded_payloads)
    target, manager = _make_manager(encoded_capacity_bytes=encoded_size_bytes - 1)
    raw_ledger = _ByteLedger()
    raw_size_bytes = _register_raw_page(target, raw_payloads, raw_ledger)
    manager.on_reuse_store = MagicMock(side_effect=encoded_payloads)

    assert not target.try_encode_reuse_backing(
        _PAGE_KEY,
        valid_token_count=4,
    )
    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.RAW
    assert snapshot.raw_resident_bytes == raw_size_bytes
    assert snapshot.encoded_resident_bytes == 0
    assert snapshot.encoding_read_leases == 0
    assert raw_ledger.used_bytes == raw_size_bytes
    assert raw_ledger.release_count == 0
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0


def test_materialize_failure_rolls_back_raw_admission_and_keeps_encoded() -> None:
    target, manager = _make_manager(encoded_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    raw_ledger = _ByteLedger()
    encoded_size_bytes = _publish_encoded_page(
        target,
        manager,
        raw_payloads,
        raw_ledger,
    )
    assert raw_ledger.used_bytes == 0

    raw_destinations = tuple(torch.empty_like(payload) for payload in raw_payloads)
    destination_size_bytes = sum(_tensor_nbytes(destination) for destination in raw_destinations)
    raw_ledger.reserve(destination_size_bytes)
    manager.on_reuse_materialize = MagicMock(side_effect=[None, RuntimeError("decode failed")])
    publish_raw_slot = MagicMock()
    rollback_raw_slot = MagicMock(side_effect=lambda: raw_ledger.release(destination_size_bytes))

    with pytest.raises(RuntimeError, match="decode failed"):
        target.materialize_reuse_backing(
            _PAGE_KEY,
            raw_destinations,
            publish_raw_slot=publish_raw_slot,
            rollback_raw_slot=rollback_raw_slot,
        )

    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.ENCODED
    assert snapshot.encoded_resident_bytes == encoded_size_bytes
    assert snapshot.encoded_read_leases == 0
    assert snapshot.pending_materializations == 0
    assert target.get_reuse_backing_pool_snapshot().used_bytes == encoded_size_bytes
    assert raw_ledger.used_bytes == 0
    publish_raw_slot.assert_not_called()
    rollback_raw_slot.assert_called_once_with()


def test_materialized_active_raw_survives_cold_encoded_eviction() -> None:
    target, manager = _make_manager(encoded_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    raw_ledger = _ByteLedger()
    encoded_size_bytes = _publish_encoded_page(
        target,
        manager,
        raw_payloads,
        raw_ledger,
    )
    raw_destinations = tuple(torch.empty_like(payload) for payload in raw_payloads)
    destination_size_bytes = sum(_tensor_nbytes(destination) for destination in raw_destinations)
    raw_ledger.reserve(destination_size_bytes)
    manager.on_reuse_materialize = MagicMock(return_value=None)
    publish_raw_slot = MagicMock()
    rollback_raw_slot = MagicMock(side_effect=lambda: raw_ledger.release(destination_size_bytes))
    decode_event = _FakeEvent()

    materialization_id = target.materialize_reuse_backing(
        _PAGE_KEY,
        raw_destinations,
        publish_raw_slot=publish_raw_slot,
        rollback_raw_slot=rollback_raw_slot,
        completion_event=decode_event,
    )
    assert materialization_id >= 0
    pending_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert pending_snapshot.state is ReuseBackingState.ENCODED
    assert pending_snapshot.encoded_read_leases == 1
    assert pending_snapshot.pending_materializations == 1
    assert not target.evict_reuse_encoded_backing(_PAGE_KEY)
    publish_raw_slot.assert_not_called()
    rollback_raw_slot.assert_not_called()

    decode_event.ready = True
    assert target.poll_reuse_backing(_PAGE_KEY) == 1
    ready_snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert ready_snapshot.encoded_read_leases == 0
    assert ready_snapshot.pending_materializations == 0
    publish_raw_slot.assert_called_once_with()
    rollback_raw_slot.assert_not_called()
    assert target.get_reuse_backing_pool_snapshot().used_bytes == encoded_size_bytes

    assert target.evict_reuse_encoded_backing(_PAGE_KEY)
    assert target.get_reuse_backing_pool_snapshot().used_bytes == 0
    assert raw_ledger.used_bytes == destination_size_bytes


def test_materialize_publish_failure_rolls_back_raw_admission() -> None:
    target, manager = _make_manager(encoded_capacity_bytes=1024)
    raw_payloads = _raw_payloads()
    raw_ledger = _ByteLedger()
    encoded_size_bytes = _publish_encoded_page(
        target,
        manager,
        raw_payloads,
        raw_ledger,
    )
    raw_destinations = tuple(torch.empty_like(payload) for payload in raw_payloads)
    destination_size_bytes = sum(_tensor_nbytes(payload) for payload in raw_destinations)
    raw_ledger.reserve(destination_size_bytes)
    manager.on_reuse_materialize = MagicMock(return_value=None)
    publish_raw_slot = MagicMock(side_effect=RuntimeError("raw publish failed"))
    rollback_raw_slot = MagicMock(side_effect=lambda: raw_ledger.release(destination_size_bytes))

    with pytest.raises(RuntimeError, match="raw publish failed"):
        target.materialize_reuse_backing(
            _PAGE_KEY,
            raw_destinations,
            publish_raw_slot=publish_raw_slot,
            rollback_raw_slot=rollback_raw_slot,
        )

    snapshot = target.get_reuse_backing_snapshot(_PAGE_KEY)
    assert snapshot.state is ReuseBackingState.ENCODED
    assert snapshot.encoded_read_leases == 0
    assert snapshot.pending_materializations == 0
    assert target.get_reuse_backing_pool_snapshot().used_bytes == encoded_size_bytes
    assert raw_ledger.used_bytes == 0
    publish_raw_slot.assert_called_once_with()
    rollback_raw_slot.assert_called_once_with()
