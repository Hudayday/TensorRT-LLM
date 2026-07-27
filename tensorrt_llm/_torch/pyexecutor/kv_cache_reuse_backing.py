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

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

ReusePageKey = tuple[int, int]
"""Process-local ``(life_cycle_id, page_generation)`` key.

The generation must be monotonic for the lifetime owner. A recyclable Python
``id(page)`` is not a valid generation.
"""


class ReuseBackingEvent(Protocol):
    """Completion event owned and polled by KVCM V2."""

    def query(self) -> bool:
        """Return whether all work preceding the event has completed."""


class ReuseBackingState(Enum):
    """Canonical cold-reuse representation state of one KVCM V2 page.

    Transient raw materializations already owned by active requests are
    intentionally outside this state.
    """

    RAW = "raw"
    DUAL = "dual"
    ENCODED = "encoded"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class ReuseBackingSnapshot:
    """Read-only cold-backing state and capacity ledger for one reusable page.

    ``encoding_read_leases`` counts only the compression transaction. Native
    V2 page readers are represented by the separate external retire gate.
    """

    state: ReuseBackingState
    raw_resident_bytes: int
    encoded_resident_bytes: int
    encoding_read_leases: int
    encoded_read_leases: int
    pending_materializations: int


@dataclass(frozen=True, slots=True)
class ReuseBackingPoolSnapshot:
    """Read-only logical compact-capacity ledger."""

    capacity_bytes: int
    used_bytes: int
    allocation_count: int


@dataclass(slots=True)
class _EncodedAllocation:
    payloads: tuple[object, ...]
    size_bytes: int


@dataclass(slots=True)
class _PendingMaterialization:
    completion_event: ReuseBackingEvent | None = None
    publish_raw: Callable[[], None] | None = None
    rollback_raw: Callable[[], None] | None = None


@dataclass(slots=True)
class _PageBacking:
    state: ReuseBackingState
    raw_payloads: tuple[object, ...]
    raw_size_bytes: int
    can_release_raw: Callable[[], bool] | None
    release_raw: Callable[[], None] | None
    encoding_read_leases: int = 0
    encoded_read_leases: int = 0
    encoding_in_progress: bool = False
    encoded_allocation_id: int | None = None
    encode_completion_event: ReuseBackingEvent | None = None
    pending_materializations: dict[int, _PendingMaterialization] = field(default_factory=dict)


class KVCacheReuseBackingStore:
    """KVCM-owned prototype state machine and compact-capacity ledger.

    The store deliberately knows nothing about a compression algorithm or
    attention layout. KVCM supplies stable physical payloads, completion
    events, and raw-slot publish/release callbacks at its ownership boundary.

    Encoded payloads are still allocator-owned tensors in this prototype. The
    byte ledger proves transactional admission and reclamation semantics; it is
    not a native V2 cache-tier slot arena.

    A successful materialization publishes transient active raw residency
    outside this cold-backing record. Evicting the encoded allocation therefore
    removes only future reuse eligibility; it never owns or releases that
    active raw reservation.
    """

    def __init__(self, encoded_capacity_bytes: int) -> None:
        if encoded_capacity_bytes < 0:
            raise ValueError("encoded_capacity_bytes must be non-negative")
        self._encoded_capacity_bytes = encoded_capacity_bytes
        self._encoded_used_bytes = 0
        self._next_allocation_id = 0
        self._next_materialization_id = 0
        self._allocations: dict[int, _EncodedAllocation] = {}
        self._pages: dict[ReusePageKey, _PageBacking] = {}

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def pool_snapshot(self) -> ReuseBackingPoolSnapshot:
        return ReuseBackingPoolSnapshot(
            capacity_bytes=self._encoded_capacity_bytes,
            used_bytes=self._encoded_used_bytes,
            allocation_count=len(self._allocations),
        )

    def page_snapshot(self, page_key: ReusePageKey) -> ReuseBackingSnapshot:
        page = self._page(page_key)
        encoded_size = 0
        if page.encoded_allocation_id is not None:
            encoded_size = self._allocations[page.encoded_allocation_id].size_bytes
        return ReuseBackingSnapshot(
            state=page.state,
            raw_resident_bytes=page.raw_size_bytes if page.raw_payloads else 0,
            encoded_resident_bytes=encoded_size,
            encoding_read_leases=page.encoding_read_leases,
            encoded_read_leases=page.encoded_read_leases,
            pending_materializations=len(page.pending_materializations),
        )

    def register_raw(
        self,
        page_key: ReusePageKey,
        raw_payloads: tuple[object, ...],
        raw_size_bytes: int,
        can_release_raw: Callable[[], bool],
        release_raw: Callable[[], None],
    ) -> None:
        if page_key in self._pages:
            raise ValueError(f"Reuse backing already exists for page {page_key}")
        if not raw_payloads:
            raise ValueError("A reusable page must contain at least one raw payload")
        if raw_size_bytes <= 0:
            raise ValueError("raw_size_bytes must be positive")
        self._pages[page_key] = _PageBacking(
            state=ReuseBackingState.RAW,
            raw_payloads=raw_payloads,
            raw_size_bytes=raw_size_bytes,
            can_release_raw=can_release_raw,
            release_raw=release_raw,
        )

    def begin_encoding(self, page_key: ReusePageKey) -> tuple[object, ...]:
        page = self._page(page_key)
        if page.state is not ReuseBackingState.RAW:
            raise RuntimeError(f"Encoding requires RAW backing, got {page.state.value}")
        if page.encoding_in_progress:
            raise RuntimeError("Encoding is already in progress")
        page.encoding_in_progress = True
        page.encoding_read_leases += 1
        return page.raw_payloads

    def cancel_encoding(self, page_key: ReusePageKey) -> None:
        page = self._page(page_key)
        if not page.encoding_in_progress:
            raise RuntimeError("No encoding transaction is in progress")
        page.encoding_in_progress = False
        page.encoding_read_leases -= 1

    def try_publish_encoded(
        self,
        page_key: ReusePageKey,
        encoded_payloads: tuple[object, ...],
        encoded_size_bytes: int,
        completion_event: ReuseBackingEvent | None,
    ) -> bool:
        page = self._page(page_key)
        if not page.encoding_in_progress:
            raise RuntimeError("No encoding transaction is in progress")
        if not encoded_payloads:
            self.cancel_encoding(page_key)
            raise ValueError("Encoded backing must contain at least one payload")
        if encoded_size_bytes <= 0:
            self.cancel_encoding(page_key)
            raise ValueError("encoded_size_bytes must be positive")
        if self._encoded_used_bytes + encoded_size_bytes > self._encoded_capacity_bytes:
            self.cancel_encoding(page_key)
            return False

        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        self._allocations[allocation_id] = _EncodedAllocation(
            payloads=encoded_payloads,
            size_bytes=encoded_size_bytes,
        )
        self._encoded_used_bytes += encoded_size_bytes

        page.encoding_in_progress = False
        page.encoded_allocation_id = allocation_id
        page.encode_completion_event = completion_event
        page.state = ReuseBackingState.DUAL
        return True

    def begin_materialization(self, page_key: ReusePageKey) -> tuple[int, tuple[object, ...]]:
        page = self._page(page_key)
        if page.state is not ReuseBackingState.ENCODED:
            raise RuntimeError(f"Materialization requires ENCODED backing, got {page.state.value}")
        allocation_id = page.encoded_allocation_id
        if allocation_id is None:
            raise RuntimeError("ENCODED page has no compact allocation")

        materialization_id = self._next_materialization_id
        self._next_materialization_id += 1
        page.pending_materializations[materialization_id] = _PendingMaterialization()
        page.encoded_read_leases += 1
        return materialization_id, self._allocations[allocation_id].payloads

    def commit_materialization(
        self,
        page_key: ReusePageKey,
        materialization_id: int,
        completion_event: ReuseBackingEvent | None,
        publish_raw: Callable[[], None],
        rollback_raw: Callable[[], None],
    ) -> None:
        pending = self._pending(page_key, materialization_id)
        if pending.publish_raw is not None:
            raise RuntimeError("Materialization is already committed")
        pending.completion_event = completion_event
        pending.publish_raw = publish_raw
        pending.rollback_raw = rollback_raw

    def cancel_materialization(self, page_key: ReusePageKey, materialization_id: int) -> None:
        page = self._page(page_key)
        self._pending(page_key, materialization_id)
        del page.pending_materializations[materialization_id]
        page.encoded_read_leases -= 1

    def poll(self, page_key: ReusePageKey) -> int:
        """Advance every completed transaction for ``page_key``.

        Returns:
            Number of state transitions or raw materialization publications.
        """

        page = self._page(page_key)
        transition_count = 0
        if page.state is ReuseBackingState.DUAL and self._is_ready(page.encode_completion_event):
            can_release_raw = page.can_release_raw
            if can_release_raw is None:
                raise RuntimeError("DUAL backing has no external raw-retire gate")
            if not can_release_raw():
                return transition_count
            release_raw = page.release_raw
            if release_raw is None:
                raise RuntimeError("DUAL backing has no raw-slot release callback")
            release_raw()
            page.raw_payloads = ()
            page.can_release_raw = None
            page.release_raw = None
            page.encoding_read_leases -= 1
            page.encode_completion_event = None
            page.state = ReuseBackingState.ENCODED
            transition_count += 1

        for materialization_id, pending in list(page.pending_materializations.items()):
            if pending.publish_raw is None or not self._is_ready(pending.completion_event):
                continue
            publish_raw = pending.publish_raw
            rollback_raw = pending.rollback_raw
            if rollback_raw is None:
                raise RuntimeError("Committed materialization has no rollback callback")
            try:
                publish_raw()
            except Exception:
                try:
                    rollback_raw()
                finally:
                    del page.pending_materializations[materialization_id]
                    page.encoded_read_leases -= 1
                raise
            del page.pending_materializations[materialization_id]
            page.encoded_read_leases -= 1
            transition_count += 1
        return transition_count

    def evict_encoded(self, page_key: ReusePageKey) -> bool:
        page = self._page(page_key)
        if page.state is ReuseBackingState.DUAL:
            return False
        if page.state is not ReuseBackingState.ENCODED:
            raise RuntimeError(f"Encoded eviction requires ENCODED backing, got {page.state.value}")
        if page.encoded_read_leases != 0:
            return False
        allocation_id = page.encoded_allocation_id
        if allocation_id is None:
            raise RuntimeError("ENCODED page has no compact allocation")

        allocation = self._allocations.pop(allocation_id)
        self._encoded_used_bytes -= allocation.size_bytes
        page.encoded_allocation_id = None
        page.state = ReuseBackingState.EVICTED
        return True

    def _page(self, page_key: ReusePageKey) -> _PageBacking:
        try:
            return self._pages[page_key]
        except KeyError as error:
            raise KeyError(f"Unknown reuse backing page {page_key}") from error

    def _pending(self, page_key: ReusePageKey, materialization_id: int) -> _PendingMaterialization:
        page = self._page(page_key)
        try:
            return page.pending_materializations[materialization_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown materialization {materialization_id} for page {page_key}"
            ) from error

    @staticmethod
    def _is_ready(completion_event: ReuseBackingEvent | None) -> bool:
        return completion_event is None or completion_event.query()
