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
    """State of the canonical cold-reuse backing for one V2 page.

    ``COMPRESSING``, ``CANCELLING``, and ``EVICTING`` are transaction states.
    In particular, compressed tensors may be admitted while the KVCM-owned
    source slot is still needed by an in-flight compression.
    ``COMPRESSED`` is the only stable reusable state. Runtime raw slots created
    by reuse hits are owned by active requests and never become part of this
    record. ``FAILED`` is quarantined and never hit-visible; KVCM may reclaim
    it only after independently confirming that native cleanup has completed.
    """

    COMPRESSING = "compressing"
    CANCELLING = "cancelling"
    COMPRESSED = "compressed"
    EVICTING = "evicting"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReuseBackingSnapshot:
    """Read-only compressed-backing state for one reusable page."""

    state: ReuseBackingState
    compressed_resident_bytes: int
    compression_source_leases: int
    compressed_read_leases: int
    pending_decompressions: int


@dataclass(frozen=True, slots=True)
class ReuseBackingPoolSnapshot:
    """Read-only compact-capacity ledger."""

    capacity_bytes: int
    used_bytes: int
    allocation_count: int


@dataclass(slots=True)
class _CompressedAllocation:
    payloads: tuple[object, ...]
    size_bytes: int


@dataclass(slots=True)
class _PendingDecompression:
    completion_event: ReuseBackingEvent | None = None
    publish_raw: Callable[[], None] | None = None
    rollback_raw: Callable[[ReuseBackingEvent | None], None] | None = None
    aborting: bool = False


@dataclass(slots=True)
class _PageBacking:
    state: ReuseBackingState
    can_publish_compressed: Callable[[], bool] | None
    publish_compressed_and_release_source: Callable[[], None] | None
    cancel_native_transaction: Callable[[], None] | None
    compressed_allocation_id: int
    compression_completion_event: ReuseBackingEvent | None
    compression_source_leases: int = 1
    compressed_read_leases: int = 0
    pending_decompressions: dict[int, _PendingDecompression] = field(default_factory=dict)
    native_callback_in_progress: bool = False


class KVCacheReuseBackingStore:
    """KVCM-owned compressed-only cold-backing transaction store.

    The store never retains a raw tensor or models raw as a cold backing.
    KVCM passes a source-retirement gate and one atomic publication callback
    after the compressed payloads have been produced. Once the compression
    event and native V2 reader gate are both complete, the callback swaps the
    Page to compressed backing and releases its full-precision source slot.

    Compressed payloads are allocator-owned tensors in this focused prototype.
    The byte ledger enforces admission and reclamation semantics, but is not
    yet a native V2 compact slot arena. Runtime integration must reserve this
    capacity from the same total GPU budget as the raw V2 pool.
    """

    def __init__(self, compressed_capacity_bytes: int) -> None:
        if compressed_capacity_bytes < 0:
            raise ValueError("compressed_capacity_bytes must be non-negative")
        self._compressed_capacity_bytes = compressed_capacity_bytes
        self._compressed_used_bytes = 0
        self._next_allocation_id = 0
        self._next_decompression_id = 0
        self._allocations: dict[int, _CompressedAllocation] = {}
        self._pages: dict[ReusePageKey, _PageBacking] = {}

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def pool_snapshot(self) -> ReuseBackingPoolSnapshot:
        return ReuseBackingPoolSnapshot(
            capacity_bytes=self._compressed_capacity_bytes,
            used_bytes=self._compressed_used_bytes,
            allocation_count=len(self._allocations),
        )

    def page_snapshot(self, page_key: ReusePageKey) -> ReuseBackingSnapshot:
        page = self._page(page_key)
        return ReuseBackingSnapshot(
            state=page.state,
            compressed_resident_bytes=self._allocations[
                page.compressed_allocation_id
            ].size_bytes,
            compression_source_leases=page.compression_source_leases,
            compressed_read_leases=page.compressed_read_leases,
            pending_decompressions=len(page.pending_decompressions),
        )

    def validate_new_page_key(self, page_key: ReusePageKey) -> None:
        """Fail before any transform launch when a generation is already live."""

        if page_key in self._pages:
            raise ValueError(f"Reuse backing already exists for page {page_key}")

    def try_admit_compression(
        self,
        page_key: ReusePageKey,
        compressed_payloads: tuple[object, ...],
        compressed_size_bytes: int,
        completion_event: ReuseBackingEvent | None,
        can_publish_compressed: Callable[[], bool],
        publish_compressed_and_release_source: Callable[[], None],
    ) -> bool:
        """Admit compressed payloads and start the publication transaction.

        A capacity miss creates no page record. The caller must remove that
        Page from cold reuse; retaining raw as a fallback is deliberately not
        part of this contract.
        """

        self.validate_new_page_key(page_key)
        if not compressed_payloads:
            raise ValueError("Compressed backing must contain at least one payload")
        if compressed_size_bytes <= 0:
            raise ValueError("compressed_size_bytes must be positive")
        if self._compressed_used_bytes + compressed_size_bytes > self._compressed_capacity_bytes:
            return False

        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        self._allocations[allocation_id] = _CompressedAllocation(
            payloads=compressed_payloads,
            size_bytes=compressed_size_bytes,
        )
        self._compressed_used_bytes += compressed_size_bytes
        self._pages[page_key] = _PageBacking(
            state=ReuseBackingState.COMPRESSING,
            can_publish_compressed=can_publish_compressed,
            publish_compressed_and_release_source=publish_compressed_and_release_source,
            cancel_native_transaction=None,
            compressed_allocation_id=allocation_id,
            compression_completion_event=completion_event,
        )
        return True

    def defer_compression_abort(
        self,
        page_key: ReusePageKey,
        transient_payloads: tuple[object, ...],
        transient_size_bytes: int,
        completion_event: ReuseBackingEvent | None,
        cancel_native_transaction: Callable[[], None],
    ) -> bool:
        """Retain source ownership until rejected compression work drains.

        This path deliberately bypasses stable compressed-capacity admission:
        the transform may already have allocated outputs before rejection or a
        capacity miss is known. These tensors are transient workspace, never a
        reusable backing, and are reclaimed only after the post-launch event
        and native cancellation callback complete.

        Returns whether cancellation completed immediately.
        """

        self.validate_new_page_key(page_key)
        if transient_size_bytes < 0:
            raise ValueError("transient_size_bytes must be non-negative")

        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        self._allocations[allocation_id] = _CompressedAllocation(
            payloads=transient_payloads,
            size_bytes=transient_size_bytes,
        )
        self._compressed_used_bytes += transient_size_bytes
        self._pages[page_key] = _PageBacking(
            state=ReuseBackingState.CANCELLING,
            can_publish_compressed=None,
            publish_compressed_and_release_source=None,
            cancel_native_transaction=cancel_native_transaction,
            compressed_allocation_id=allocation_id,
            compression_completion_event=completion_event,
        )
        return self.poll(page_key) == 1

    def quarantine_compression_abort(
        self,
        page_key: ReusePageKey,
        transient_payloads: tuple[object, ...],
        transient_size_bytes: int,
    ) -> None:
        """Retain ownership when even a post-launch event cannot be recorded."""

        self.validate_new_page_key(page_key)
        if transient_size_bytes < 0:
            raise ValueError("transient_size_bytes must be non-negative")

        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        self._allocations[allocation_id] = _CompressedAllocation(
            payloads=transient_payloads,
            size_bytes=transient_size_bytes,
        )
        self._compressed_used_bytes += transient_size_bytes
        self._pages[page_key] = _PageBacking(
            state=ReuseBackingState.FAILED,
            can_publish_compressed=None,
            publish_compressed_and_release_source=None,
            cancel_native_transaction=None,
            compressed_allocation_id=allocation_id,
            compression_completion_event=None,
        )

    def cancel_compression(
        self,
        page_key: ReusePageKey,
        cancel_native_transaction: Callable[[], None],
    ) -> bool:
        """Cancel an admitted compression before compressed publication.

        The native callback owns removal of the reuse candidate and release of
        the compression source lease. Compact tensors remain resident until
        the already-launched compression event completes.

        Returns whether cancellation completed immediately. Otherwise the
        caller must keep polling this page.
        """

        page = self._page(page_key)
        if page.state is not ReuseBackingState.COMPRESSING:
            raise RuntimeError(
                f"Compression cancellation requires COMPRESSING backing, "
                f"got {page.state.value}"
            )
        page.state = ReuseBackingState.CANCELLING
        page.can_publish_compressed = None
        page.publish_compressed_and_release_source = None
        page.cancel_native_transaction = cancel_native_transaction
        return self.poll(page_key) == 1

    def begin_decompression(self, page_key: ReusePageKey) -> tuple[int, tuple[object, ...]]:
        page = self._page(page_key)
        if page.state is not ReuseBackingState.COMPRESSED:
            raise RuntimeError(
                f"Reuse decompression requires COMPRESSED backing, got {page.state.value}"
            )

        decompression_id = self._next_decompression_id
        self._next_decompression_id += 1
        page.pending_decompressions[decompression_id] = _PendingDecompression()
        page.compressed_read_leases += 1
        return decompression_id, self._allocations[page.compressed_allocation_id].payloads

    def commit_decompression(
        self,
        page_key: ReusePageKey,
        decompression_id: int,
        completion_event: ReuseBackingEvent | None,
        publish_raw: Callable[[], None],
        rollback_raw: Callable[[ReuseBackingEvent | None], None],
    ) -> None:
        pending = self._pending(page_key, decompression_id)
        if pending.publish_raw is not None:
            raise RuntimeError("Decompression is already committed")
        pending.completion_event = completion_event
        pending.publish_raw = publish_raw
        pending.rollback_raw = rollback_raw

    def abort_decompression(
        self,
        page_key: ReusePageKey,
        decompression_id: int,
        completion_event: ReuseBackingEvent | None,
        rollback_raw: Callable[[ReuseBackingEvent | None], None],
    ) -> bool:
        """Keep the compressed read lease until failed GPU work drains."""

        pending = self._pending(page_key, decompression_id)
        if pending.publish_raw is not None or pending.aborting:
            raise RuntimeError("Decompression transaction is already finalized")
        pending.completion_event = completion_event
        pending.rollback_raw = rollback_raw
        pending.aborting = True
        return self.poll(page_key) == 1

    def quarantine_decompression_abort(
        self,
        page_key: ReusePageKey,
        decompression_id: int,
        rollback_raw: Callable[[ReuseBackingEvent | None], None],
    ) -> None:
        """Preserve both reservations when no trustworthy event is available."""

        page = self._page(page_key)
        pending = self._pending(page_key, decompression_id)
        if pending.publish_raw is not None or pending.aborting:
            raise RuntimeError("Decompression transaction is already finalized")
        pending.completion_event = None
        pending.rollback_raw = rollback_raw
        pending.aborting = True
        self._quarantine(page)

    def poll(self, page_key: ReusePageKey) -> int:
        """Advance completed compression/decompression transactions."""

        page = self._page(page_key)
        transition_count = 0
        if page.state is ReuseBackingState.FAILED:
            return 0
        if page.native_callback_in_progress:
            return 0

        try:
            compression_ready = self._is_ready(page.compression_completion_event)
        except Exception:
            if page.state in (
                ReuseBackingState.COMPRESSING,
                ReuseBackingState.CANCELLING,
            ):
                self._quarantine(page)
            raise

        if page.state is ReuseBackingState.CANCELLING and compression_ready:
            cancel = page.cancel_native_transaction
            if cancel is None:
                self._quarantine(page)
                raise RuntimeError("CANCELLING backing has no native cancel callback")
            # Consume before calling so a partially successful native callback
            # is never retried.
            page.cancel_native_transaction = None
            page.compression_completion_event = None
            page.native_callback_in_progress = True
            try:
                cancel()
            except Exception:
                self._quarantine(page)
                raise
            page.native_callback_in_progress = False
            page.compression_source_leases = 0
            self._remove_page(page_key)
            return 1

        if page.state is ReuseBackingState.COMPRESSING and compression_ready:
            can_publish_compressed = page.can_publish_compressed
            if can_publish_compressed is None:
                self._quarantine(page)
                raise RuntimeError("COMPRESSING backing has no native publish gate")
            try:
                can_publish = can_publish_compressed()
            except Exception:
                self._quarantine(page)
                raise
            if not can_publish:
                return transition_count
            publish = page.publish_compressed_and_release_source
            if publish is None:
                self._quarantine(page)
                raise RuntimeError("COMPRESSING backing has no native atomic-publish callback")
            # Consume the callback before invoking it. A callback that performs
            # its native swap and then raises must never be retried.
            page.can_publish_compressed = None
            page.publish_compressed_and_release_source = None
            page.compression_completion_event = None
            page.native_callback_in_progress = True
            try:
                # The callback owns one KVCM lifecycle critical section:
                # attach compressed backing, detach the full-precision source,
                # and return its slot. No hit can observe a half-swapped Page.
                publish()
            except Exception:
                self._quarantine(page)
                raise
            page.native_callback_in_progress = False
            page.compression_source_leases = 0
            page.state = ReuseBackingState.COMPRESSED
            transition_count += 1

        for decompression_id, pending in list(page.pending_decompressions.items()):
            if pending.publish_raw is None and not pending.aborting:
                continue
            try:
                decompression_ready = self._is_ready(pending.completion_event)
            except Exception:
                # The completion status and raw-reservation ownership are now
                # unknown. Preserve the allocation, pending callback, and read
                # lease for explicit owner reconciliation.
                self._quarantine(page)
                raise
            if not decompression_ready:
                continue

            if pending.aborting:
                rollback_raw = pending.rollback_raw
                if rollback_raw is None:
                    self._quarantine(page)
                    raise RuntimeError("Aborting decompression has no rollback callback")
                page.native_callback_in_progress = True
                try:
                    rollback_raw(pending.completion_event)
                except Exception:
                    self._quarantine(page)
                    raise
                page.native_callback_in_progress = False
                del page.pending_decompressions[decompression_id]
                page.compressed_read_leases -= 1
                transition_count += 1
                continue

            publish_raw = pending.publish_raw
            rollback_raw = pending.rollback_raw
            if rollback_raw is None:
                self._quarantine(page)
                raise RuntimeError("Committed decompression has no rollback callback")
            # Guard rather than delete the transaction while native callbacks
            # run. Reentrant poll/evict sees the read lease and cannot publish
            # twice or reclaim the compressed source.
            pending.publish_raw = None
            pending.aborting = True
            page.native_callback_in_progress = True
            try:
                publish_raw()
            except Exception:
                try:
                    rollback_raw(pending.completion_event)
                except Exception:
                    self._quarantine(page)
                    raise
                page.native_callback_in_progress = False
                del page.pending_decompressions[decompression_id]
                page.compressed_read_leases -= 1
                raise
            page.native_callback_in_progress = False
            del page.pending_decompressions[decompression_id]
            page.compressed_read_leases -= 1
            transition_count += 1
        return transition_count

    def evict_compressed(
        self,
        page_key: ReusePageKey,
        unpublish_reuse_candidate: Callable[[], None],
    ) -> bool:
        """Atomically unpublish and reclaim one compressed cold backing."""
        page = self._page(page_key)
        if page.state in (ReuseBackingState.COMPRESSING, ReuseBackingState.CANCELLING):
            return False
        if page.state is not ReuseBackingState.COMPRESSED:
            raise RuntimeError(
                f"Compressed eviction requires COMPRESSED backing, got {page.state.value}"
            )
        if page.compressed_read_leases != 0:
            return False

        # Block begin_decompression before touching the radix/Block.storage
        # owner. The owner callback must make the Page unreachable to new hits
        # before this method releases its compact allocation.
        page.state = ReuseBackingState.EVICTING
        try:
            unpublish_reuse_candidate()
        except Exception:
            # The callback may have completed the native unpublish before
            # raising. Never make an outcome-unknown page hit-visible again.
            self._quarantine(page)
            raise
        self._remove_page(page_key)
        return True

    def reconcile_failed(
        self,
        page_key: ReusePageKey,
        native_cleanup_is_complete: Callable[[], bool],
    ) -> bool:
        """Reclaim a quarantined record after owner-side reconciliation.

        The callback is a read-only confirmation, not a cleanup action. KVCM
        must first drain any GPU work and make the native Page unreachable.
        Returning ``False`` or raising keeps the compressed allocation
        quarantined and prevents unsafe reuse or reclamation.
        """

        page = self._page(page_key)
        if page.state is not ReuseBackingState.FAILED:
            raise RuntimeError(
                f"Failure reconciliation requires FAILED backing, got {page.state.value}"
            )
        if not native_cleanup_is_complete():
            return False
        if page.native_callback_in_progress:
            raise RuntimeError("Cannot reclaim FAILED backing during a native callback")
        # The owner confirmation covers every raw reservation represented by
        # these pending callbacks as well as the compressed/source leases.
        page.pending_decompressions.clear()
        page.compressed_read_leases = 0
        page.compression_source_leases = 0
        self._remove_page(page_key)
        return True

    def _remove_page(self, page_key: ReusePageKey) -> None:
        page = self._pages.pop(page_key)
        allocation = self._allocations.pop(page.compressed_allocation_id)
        self._compressed_used_bytes -= allocation.size_bytes

    @staticmethod
    def _quarantine(page: _PageBacking) -> None:
        page.state = ReuseBackingState.FAILED
        page.can_publish_compressed = None
        page.publish_compressed_and_release_source = None
        page.cancel_native_transaction = None
        page.compression_completion_event = None
        page.native_callback_in_progress = False

    def _page(self, page_key: ReusePageKey) -> _PageBacking:
        try:
            return self._pages[page_key]
        except KeyError as error:
            raise KeyError(f"Unknown reuse backing page {page_key}") from error

    def _pending(
        self, page_key: ReusePageKey, decompression_id: int
    ) -> _PendingDecompression:
        page = self._page(page_key)
        try:
            return page.pending_decompressions[decompression_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown decompression {decompression_id} for page {page_key}"
            ) from error

    @staticmethod
    def _is_ready(completion_event: ReuseBackingEvent | None) -> bool:
        return completion_event is None or completion_event.query()
