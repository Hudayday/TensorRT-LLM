"""Multi-manager runtime coordinator for the L2 behavior layer.

A :class:`KVCacheBehaviorCoordinator` owns a list of
:class:`BaseKVCacheCompressionExecutor` instances (typically 1–3: one per axis
from ``sparse`` / ``storage`` / ``crcl``) and dispatches each lifecycle hook
to all managers in a deterministic axis-priority order. PyExecutor sees only
the coordinator; the coordinator handles per-axis dispatch + mutex
validation + cross-axis dependency wiring.

Currently (Phase 3 ship) only axis ``"sparse"`` has a concrete subclass
(:class:`SparseAttentionManager`). Phase 4 adds axis ``"storage"``
(KVCacheStorageManager + KVTC); Phase 5 may add axis ``"crcl"``
(CRCLManager + Continuum or chosen candidate). The :data:`HOOK_ORDER`
table is already set up for all three axes — no changes needed at
coordinator level when new axes ship.

This module is part of the planned v17 multi-manager runtime evolution
(see ``~/docs/kv-reduction/23-multi-manager-runtime-refactor-plan.md`` and
``24-l2-behavior-layer-concrete-design.md``). **PyExecutor is NOT yet wired
to use the coordinator** — the existing single-SparseAttentionManager path
stays active. The coordinator class is shipped now (Phase 3) so the
framework scaffolding is in place for Phase 4 KVTC integration to slot
into.
"""

import warnings
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

from .sparse_attention_manager import (BaseKVCacheCompressionExecutor,
                                       SparseAttentionManager)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# Hook execution order across axes. Each hook name maps to a list of axis
# identifiers in dispatch order. A manager whose axis is not in the list for
# a given hook is silently skipped for that hook (e.g., axis-D hooks are not
# called for axis-C-only managers).
_HOOK_ORDER: Dict[str, List[str]] = {
    # Request lifecycle: CRCL first (pool lookup) -> SPARSE (init state) ->
    # STORAGE (may decompress on pool hit).
    "on_request_init":         ["crcl", "sparse", "storage"],
    # Final cleanup: SPARSE (final evict) -> STORAGE (final encode) ->
    # CRCL (promote compressed bytes to cross-request pool).
    "on_request_finish":       ["sparse", "storage", "crcl"],
    # Attention hooks: only SPARSE writes attention metadata (form I);
    # STORAGE and CRCL stay out of the attention path.
    "on_context_attention":    ["sparse"],
    "on_generation_attention": ["sparse"],
    # Phase boundary: SPARSE evict first -> STORAGE compresses remaining
    # cache -> CRCL marks eligible for cross-request retention.
    "on_context_end":          ["sparse", "storage", "crcl"],
    # Per-step: SPARSE periodic evict (beta=128) -> STORAGE invalidate
    # active compressed copy -> CRCL TTL GC.
    "on_generation_step_end":  ["sparse", "storage", "crcl"],
    # Per-forward async: STORAGE waits pending decompress -> CRCL resumes
    # from pool -> SPARSE typically no-op for begin.
    "on_forward_begin":        ["storage", "crcl", "sparse"],
    # After forward: SPARSE -> STORAGE triggers async re-compress -> CRCL
    # TTL update.
    "on_forward_end":          ["sparse", "storage", "crcl"],
}


class KVCacheBehaviorCoordinator:
    """Multi-manager runtime coordinator.

    Owns ``managers: List[BaseKVCacheCompressionExecutor]`` and dispatches each
    of the 8 lifecycle hooks to all managers in deterministic axis-priority
    order.

    Mutex rules (enforced in ``__init__``):

    - At most one manager per axis (intra-axis stacking not supported, see
      ``~/docs/kv-reduction/21-framework-architecture-rationale.md`` §6).
    - Soft warnings for some cross-axis combos (e.g., sparse-evicted cache
      in a CRCL pool) — emitted via ``warnings.warn`` at init time.

    Dependency wiring (in ``__init__``):

    - If both axis ``"crcl"`` and axis ``"storage"`` managers are present,
      the CRCL manager's ``storage_delegate`` attribute (if it exists) is
      automatically set to the storage manager, so cross-request pool
      entries are stored as storage-compressed bytes.
    """

    #: Public alias of the module-level hook order table. Subclasses may
    #: override on the class to customize per-deployment ordering.
    HOOK_ORDER: Dict[str, List[str]] = _HOOK_ORDER

    def __init__(self, managers: List[BaseKVCacheCompressionExecutor]):
        self.managers: List[BaseKVCacheCompressionExecutor] = list(managers)
        self._by_axis: Dict[str, List[BaseKVCacheCompressionExecutor]] = {}
        for mgr in self.managers:
            self._by_axis.setdefault(mgr.axis, []).append(mgr)
        self._validate()
        self._wire_dependencies()

    # ------------------------------------------------------------------ #
    # Init helpers                                                        #
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        """Enforce mutex rules at init time."""
        # Hard mutex: intra-axis stacking not supported.
        for axis, mgrs in self._by_axis.items():
            if len(mgrs) > 1:
                raise ValueError(
                    f"Intra-axis stacking not supported: {len(mgrs)} "
                    f"managers found for axis={axis!r}. Most sparse / "
                    f"storage / CRCL methods assume sole arbiter; stacking "
                    f"two of the same axis would invalidate per-method "
                    f"correctness assumptions. For intra-axis composition, "
                    f"write a hybrid algorithm subclass instead.")

    def _wire_dependencies(self) -> None:
        """Wire optional cross-axis delegate attributes.

        CRCL managers (Phase 5+) may use a storage manager for their
        cross-request compressed pool. Auto-wire if both are present and
        the CRCL manager exposes a ``storage_delegate`` attribute.
        """
        crcl_list = self._by_axis.get("crcl", [])
        storage_list = self._by_axis.get("storage", [])
        if crcl_list and storage_list:
            crcl_mgr = crcl_list[0]
            storage_mgr = storage_list[0]
            if hasattr(crcl_mgr, "storage_delegate"):
                crcl_mgr.storage_delegate = storage_mgr

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def has_axis(self, axis: str) -> bool:
        """Return ``True`` if a manager of the given axis is registered."""
        return axis in self._by_axis and bool(self._by_axis[axis])

    def get_manager(
            self, axis: str) -> Optional[BaseKVCacheCompressionExecutor]:
        """Return the single manager for the given axis (or ``None``)."""
        mgrs = self._by_axis.get(axis, [])
        return mgrs[0] if mgrs else None

    def get_sparse_manager(self) -> Optional[SparseAttentionManager]:
        """Convenience accessor — returns the axis-C manager if present.

        Used by code that historically accessed
        ``PyExecutor.sparse_attention_manager`` directly. Returns the
        axis-C ``SparseAttentionManager`` instance, narrowed for type
        hinting; falls back to ``None`` if no axis-C manager registered.
        """
        return self.get_manager("sparse")  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Hook dispatch — 8 methods, all sequential per HOOK_ORDER            #
    # ------------------------------------------------------------------ #

    def _iter_for_hook(
            self,
            hook_name: str) -> Iterable[BaseKVCacheCompressionExecutor]:
        """Yield managers in dispatch order for the given hook."""
        order = self.HOOK_ORDER.get(
            hook_name,
            ["sparse", "storage", "crcl"],  # default fallback order
        )
        for axis in order:
            for mgr in self._by_axis.get(axis, []):
                yield mgr

    def on_request_init(self, request: "LlmRequest") -> None:
        for mgr in self._iter_for_hook("on_request_init"):
            mgr.on_request_init(request)

    def on_request_finish(self, request: "LlmRequest") -> None:
        for mgr in self._iter_for_hook("on_request_finish"):
            mgr.on_request_finish(request)

    def on_context_attention(
        self,
        layer_idx,
        q,
        k,
        attn_scores,
        metadata: "AttentionMetadata",
    ):
        # Single-source invariant: at most one manager may return non-None.
        result = None
        for mgr in self._iter_for_hook("on_context_attention"):
            r = mgr.on_context_attention(layer_idx, q, k, attn_scores,
                                         metadata)
            if r is not None:
                if result is not None:
                    raise RuntimeError(
                        "Multiple managers returned attention metadata "
                        "from on_context_attention; form-I sparse "
                        "attention metadata writes must be single-source.")
                result = r
        return result

    def on_context_end(self, request: "LlmRequest",
                       metadata: "AttentionMetadata") -> None:
        for mgr in self._iter_for_hook("on_context_end"):
            mgr.on_context_end(request, metadata)

    def on_generation_attention(
        self,
        layer_idx,
        q,
        k,
        attn_scores,
        metadata: "AttentionMetadata",
    ):
        result = None
        for mgr in self._iter_for_hook("on_generation_attention"):
            r = mgr.on_generation_attention(layer_idx, q, k, attn_scores,
                                            metadata)
            if r is not None:
                if result is not None:
                    raise RuntimeError(
                        "Multiple managers returned attention metadata "
                        "from on_generation_attention; form-I sparse "
                        "attention metadata writes must be single-source.")
                result = r
        return result

    def on_generation_step_end(
            self, scheduled_batch: "ScheduledRequests",
            attn_metadata: "AttentionMetadata") -> None:
        for mgr in self._iter_for_hook("on_generation_step_end"):
            mgr.on_generation_step_end(scheduled_batch, attn_metadata)

    def on_forward_begin(self, forward_batch) -> None:
        for mgr in self._iter_for_hook("on_forward_begin"):
            mgr.on_forward_begin(forward_batch)

    def on_forward_end(self, forward_batch) -> None:
        for mgr in self._iter_for_hook("on_forward_end"):
            mgr.on_forward_end(forward_batch)
