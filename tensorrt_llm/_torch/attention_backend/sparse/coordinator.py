"""Multi-manager runtime coordinator — Path A (inherits BaseResourceManager).

A :class:`KVCacheBehaviorCoordinator` owns N
:class:`BaseKVCacheCompressionExecutor` instances (typically 1–3: one per axis
``sparse`` / ``storage`` / ``cross_request``) and is registered with PyExecutor as a
:class:`BaseResourceManager`. PyExecutor's main loop already invokes
``prepare_resources / update_resources / free_resources`` on every registered
resource manager — the Coordinator's overrides translate those 3 callbacks
into the 6 semantic hooks on the executors.

Two-tier API:

- **Low-level direct dispatch** (`on_*` methods) — fan out to executors in
  ``HOOK_ORDER`` for a given hook. Used by tests + future direct callers.
- **PyExecutor auto-invoke** (`prepare_resources` / `update_resources` /
  `free_resources` from BaseResourceManager) — internal filter/dedupe
  logic + call the low-level dispatch.

Architecture (Path A, 2026-05-27 design + V1 RocketKV evidence):

- **HOOK 1** ``on_request_init`` ← fan-out from ``prepare_resources`` (first-
  seen request via internal ``_seen_req_ids`` dedupe).
- **HOOK 2** ``on_context_attention`` ← direct fire from
  ``TrtllmAttention.forward`` (prefill path), via ``metadata.coordinator``.
- **HOOK 3** ``on_context_end`` ← fan-out from ``update_resources`` (detected
  via ``CONTEXT_INIT → GENERATION_IN_PROGRESS`` state transition).
- **HOOK 4** ``on_generation_attention`` ← direct fire from
  ``TrtllmAttention.forward`` (decode path).
- **HOOK 5** ``on_generation_step_end`` ← fan-out from ``update_resources``.
- **HOOK 6** ``on_request_finish`` ← fan-out from ``free_resources``.

Why Path A:

- Zero PyExecutor changes — uses ``BaseResourceManager`` interface already in
  trunk (V1 ``RocketKVCacheManager`` is the precedent: it inherits
  ``KVCacheManager`` -> ``BaseResourceManager`` and its Stage I-b physical
  evict via ``rewind_kv_cache`` runs in ``update_resources``).
- Aligns with Fanrong PR #12733 (per-method ``cache_manager.py`` is also a
  ``BaseResourceManager`` subclass — no conflict between his pattern and ours).
- Multi-axis composition still supported by Coordinator (V1's flat resource-
  manager registration doesn't enforce axis order; Coordinator does).

Currently (Phase 3 ship) only axis ``"sparse"`` has a concrete subclass
(:class:`SparseAttentionExecutor`). Phase 4 adds axis ``"storage"`` (KVTC);
Phase 5 may add axis ``"cross_request"`` (Continuum or chosen candidate).

PyExecutor wiring (Phase 3 NOT YET wired — coordinator class is shipped so
the framework scaffolding is in place; LLM init factory + attention metadata
builder integration are Phase 4+):

- Step 1: ``py_executor.resource_manager.add(coordinator)`` at LLM init
  → PyExecutor automatically calls ``prepare/update/free_resources`` each
  iteration without further PyExecutor code changes.
- Step 2: ``attention_metadata_builder.coordinator = coordinator``
  → ``TrtllmAttention.forward`` calls ``metadata.coordinator.on_*_attention``
  for HOOK 2/4 in the attention forward path.

See ``~/docs/kv-reduction/30-hook-analysis-v1-rocketkv-grounded.md`` §5 for
the full Path A design discussion + V1 evidence + migration plan.
"""

from itertools import chain
from typing import Dict, Iterable, List, Optional, Set, TYPE_CHECKING

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager

from .kv_cache_compression_executor import (BaseKVCacheCompressionExecutor,
                                            SparseAttentionExecutor)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# Hook execution order across axes. Each hook name maps to a list of axis
# identifiers in dispatch order. An executor whose axis is not in the list
# for a given hook is silently skipped for that hook.
_HOOK_ORDER: Dict[str, List[str]] = {
    # Request lifecycle: cross-request first (pool lookup) -> sparse (init
    # state) -> storage (may decompress on pool hit).
    "on_request_init":         ["cross_request", "sparse", "storage"],
    # Final cleanup: sparse (final evict) -> storage (final encode) ->
    # cross-request (promote compressed bytes to pool).
    "on_request_finish":       ["sparse", "storage", "cross_request"],
    # Attention hooks: only sparse-attention executors write attention
    # metadata; storage and cross-request executors stay out of the
    # attention path.
    "on_context_attention":    ["sparse"],
    "on_generation_attention": ["sparse"],
    # Phase boundary: sparse evict first -> storage compresses remaining
    # cache -> cross-request marks eligible for retention.
    "on_context_end":          ["sparse", "storage", "cross_request"],
    # Per-step: sparse periodic evict -> storage invalidate active
    # compressed copy -> cross-request TTL GC.
    "on_generation_step_end":  ["sparse", "storage", "cross_request"],
}


class KVCacheBehaviorCoordinator(BaseResourceManager):
    """Path A coordinator — inherits :class:`BaseResourceManager` so PyExecutor
    automatically invokes lifecycle callbacks each iteration without any
    PyExecutor code changes.
    """

    #: Public alias of the module-level hook order table. Subclasses may
    #: override on the class to customize per-deployment ordering.
    HOOK_ORDER: Dict[str, List[str]] = _HOOK_ORDER

    def __init__(self,
                 executors: List[BaseKVCacheCompressionExecutor]) -> None:
        self.executors: List[BaseKVCacheCompressionExecutor] = list(executors)
        self._by_axis: Dict[str, List[BaseKVCacheCompressionExecutor]] = {}
        for e in self.executors:
            self._by_axis.setdefault(e.axis, []).append(e)
        import os as _os
        if _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1":
            print(f"[Coordinator.__init__] n_executors={len(self.executors)} "
                  f"executor_types={[type(e).__name__ for e in self.executors]} "
                  f"axes={list(self._by_axis.keys())}",
                  flush=True)
        # Lifecycle state needed for fan-out logic:
        # - `_seen_req_ids` dedupes on_request_init across iterations.
        # - `_prev_req_state` detects CONTEXT_INIT->GENERATION_IN_PROGRESS
        #   transition for on_context_end.
        self._seen_req_ids: Set[int] = set()
        self._prev_req_state: Dict[int, LlmRequestState] = {}
        self._validate()
        self._wire_dependencies()

    # ================================================================== #
    # Tier 1 — low-level direct dispatch (6 hooks).                       #
    # Tests + future direct callers use these.                            #
    # ================================================================== #

    def on_request_init(self, request: "LlmRequest") -> None:
        """HOOK 1 direct fan-out. The production path goes through
        :meth:`prepare_resources` which calls this after dedupe."""
        for e in self._iter_for_hook("on_request_init"):
            e.on_request_init(request)

    def on_request_finish(self, request: "LlmRequest") -> None:
        """HOOK 6 direct fan-out. The production path goes through
        :meth:`free_resources` which calls this."""
        for e in self._iter_for_hook("on_request_finish"):
            e.on_request_finish(request)

    def on_context_end(self, request: "LlmRequest",
                       metadata: Optional["AttentionMetadata"]) -> None:
        """HOOK 3 direct fan-out. The production path goes through
        :meth:`update_resources` which calls this after detecting the
        ``CONTEXT_INIT → GENERATION_IN_PROGRESS`` state transition."""
        for e in self._iter_for_hook("on_context_end"):
            e.on_context_end(request, metadata)

    def on_generation_step_end(
            self, scheduled_batch: "ScheduledRequests",
            attn_metadata: Optional["AttentionMetadata"]) -> None:
        """HOOK 5 direct fan-out. The production path goes through
        :meth:`update_resources` which calls this once per iteration."""
        for e in self._iter_for_hook("on_generation_step_end"):
            e.on_generation_step_end(scheduled_batch, attn_metadata)

    def on_context_attention(
        self,
        layer_idx: int,
        q,
        k,
        attn_scores,
        metadata: "AttentionMetadata",
    ):
        """HOOK 2 direct fan-out (called from ``TrtllmAttention.forward``
        prefill path via ``metadata.coordinator``).

        Single-source attention metadata invariant: at most one executor may
        return non-None per call."""
        import os as _os
        if _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1":
            print(f"[Coordinator.on_context_attention] layer={layer_idx} "
                  f"n_executors_for_hook={len(list(self._iter_for_hook('on_context_attention')))}",
                  flush=True)
        result = None
        for e in self._iter_for_hook("on_context_attention"):
            r = e.on_context_attention(layer_idx, q, k, attn_scores, metadata)
            if r is not None:
                if result is not None:
                    raise RuntimeError(
                        "Multiple executors returned attention metadata "
                        "from on_context_attention; sparse-attention "
                        "metadata writes must be single-source.")
                result = r
        return result

    def on_generation_attention(
        self,
        layer_idx: int,
        q,
        k,
        attn_scores,
        metadata: "AttentionMetadata",
    ):
        """HOOK 4 direct fan-out (called from ``TrtllmAttention.forward``
        decode path via ``metadata.coordinator``).

        Same single-source invariant as :meth:`on_context_attention`."""
        result = None
        for e in self._iter_for_hook("on_generation_attention"):
            r = e.on_generation_attention(layer_idx, q, k, attn_scores,
                                          metadata)
            if r is not None:
                if result is not None:
                    raise RuntimeError(
                        "Multiple executors returned attention metadata "
                        "from on_generation_attention; sparse-attention "
                        "metadata writes must be single-source.")
                result = r
        return result

    # ================================================================== #
    # Tier 2 — BaseResourceManager required interface.                    #
    # PyExecutor auto-invokes these each iteration.                       #
    # Internally calls Tier 1 direct dispatch.                            #
    # ================================================================== #

    def get_max_resource_count(self) -> int:
        """Coordinator does not own physical resources (the V2 cache manager
        does). Returns 0 so PyExecutor's scheduler does not gate on us."""
        return 0

    def get_needed_resource_to_completion(self,
                                          request: "LlmRequest") -> int:
        """Coordinator does not own physical resources (the V2 cache manager
        does). Returns 0 so PyExecutor's scheduler does not block on us."""
        return 0

    def prepare_resources(self,
                          scheduled_batch: "ScheduledRequests") -> None:
        """Fan-out to HOOK 1 (``on_request_init``) for newly-seen requests.

        Dedupes via :attr:`_seen_req_ids` so init fires exactly once per
        request, regardless of how many iterations the request stays in
        ``scheduled_batch``.
        """
        for req in chain(scheduled_batch.context_requests,
                         scheduled_batch.generation_requests):
            rid = req.py_request_id
            if rid not in self._seen_req_ids:
                self.on_request_init(req)
                self._seen_req_ids.add(rid)

    def update_resources(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ) -> None:
        """Fan-out:
        - HOOK 3 (``on_context_end``) for each request that just transitioned
          from ``CONTEXT_INIT`` to ``GENERATION_IN_PROGRESS``.
        - HOOK 5 (``on_generation_step_end``) once per iteration.

        Signature matches V1 ``RocketKVCacheManager.update_resources`` so
        PyExecutor's call site (``py_executor.py:2128`` etc.) passes the
        optional ``attn_metadata`` argument through transparently.
        """
        # HOOK 3 — detect prefill→decode transition.
        #
        # 2026-05-28 fix: PyExecutor updates the request state to
        # GENERATION_IN_PROGRESS *before* dispatching update_resources
        # iteration on the registered resource managers, so by the time
        # the Coordinator first sees a request the state is already
        # GENERATION_IN_PROGRESS — the strict ``CONTEXT_INIT →
        # GENERATION_IN_PROGRESS`` check would miss every transition.
        # Detect "first time seen in GEN_IN_PROGRESS" (prev is None) OR
        # explicit CONTEXT_INIT → GEN_IN_PROGRESS transition (in case
        # we get an earlier callback path).
        import os as _os
        _debug = _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1"
        for req in chain(scheduled_batch.context_requests,
                         scheduled_batch.generation_requests):
            rid = req.py_request_id
            prev = self._prev_req_state.get(rid)
            curr = req.state
            if _debug:
                print(f"[Coordinator.update_resources] rid={rid} "
                      f"prev={prev} curr={curr}", flush=True)
            transition_to_gen = (
                curr == LlmRequestState.GENERATION_IN_PROGRESS
                and (prev is None
                     or prev == LlmRequestState.CONTEXT_INIT))
            if transition_to_gen:
                self.on_context_end(req, attn_metadata)
            self._prev_req_state[rid] = curr
        # HOOK 5 — once per iteration.
        self.on_generation_step_end(scheduled_batch, attn_metadata)

    def free_resources(self, request: "LlmRequest") -> None:
        """Fan-out to HOOK 6 (``on_request_finish``). Same-iteration with
        PyExecutor's ``_collect_finished_or_aborted`` — no abort race."""
        rid = request.py_request_id
        self._seen_req_ids.discard(rid)
        self._prev_req_state.pop(rid, None)
        self.on_request_finish(request)

    # ================================================================== #
    # Init helpers                                                        #
    # ================================================================== #

    def _validate(self) -> None:
        """Enforce mutex rules at init time."""
        for axis, exs in self._by_axis.items():
            if len(exs) > 1:
                raise ValueError(
                    f"Intra-axis stacking not supported: {len(exs)} "
                    f"executors found for axis={axis!r}. Most sparse / "
                    f"storage / cross-request methods assume sole arbiter; "
                    f"stacking two of the same axis would invalidate "
                    f"per-method correctness assumptions. For intra-axis "
                    f"composition, write a hybrid algorithm subclass "
                    f"instead.")

    def _wire_dependencies(self) -> None:
        """Wire optional cross-axis delegate attributes.

        Cross-request executors (Phase 5+) may use a storage executor for
        their cross-request compressed pool. Auto-wire if both are present
        and the cross-request executor exposes a ``storage_delegate`` attr.
        """
        cross_request_list = self._by_axis.get("cross_request", [])
        storage_list = self._by_axis.get("storage", [])
        if cross_request_list and storage_list:
            cross_request_exec = cross_request_list[0]
            storage_exec = storage_list[0]
            if hasattr(cross_request_exec, "storage_delegate"):
                cross_request_exec.storage_delegate = storage_exec

    # ================================================================== #
    # Introspection                                                       #
    # ================================================================== #

    def has_axis(self, axis: str) -> bool:
        """Return ``True`` if an executor of the given axis is registered."""
        return axis in self._by_axis and bool(self._by_axis[axis])

    def get_executor(
            self, axis: str) -> Optional[BaseKVCacheCompressionExecutor]:
        """Return the single executor for the given axis (or ``None``)."""
        exs = self._by_axis.get(axis, [])
        return exs[0] if exs else None

    def get_sparse_executor(self) -> Optional[SparseAttentionExecutor]:
        """Convenience accessor — returns the sparse-attention executor if
        present."""
        return self.get_executor("sparse")  # type: ignore[return-value]

    # ================================================================== #
    # Dispatch order helper                                               #
    # ================================================================== #

    def _iter_for_hook(
            self,
            hook_name: str) -> Iterable[BaseKVCacheCompressionExecutor]:
        """Yield executors in dispatch order for the given hook."""
        order = self.HOOK_ORDER.get(
            hook_name,
            ["sparse", "storage", "cross_request"],  # default fallback order
        )
        for axis in order:
            for e in self._by_axis.get(axis, []):
                yield e
