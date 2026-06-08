"""Standalone KV-cache behavior coordinator.

A :class:`KVCacheBehaviorCoordinator` owns N
:class:`BaseKVCacheCompressionExecutor` instances (typically 1-2: one per axis
``sparse`` / ``storage``) and fans the runtime's lifecycle + attention events
out to them in deterministic axis order. It is a pure *compute-behavior* object:
it owns no physical KV memory (the V2 cache manager does) and is
framework-agnostic.

**Independent registration.** The coordinator is held as a first-class member on
PyExecutor (``py_executor.kv_behavior_coordinator``) and driven by the main loop
directly. It is deliberately NOT a ``BaseResourceManager`` and is NOT placed in
the resource-manager registry -- its lifecycle is its own concern, not mixed in
with physical-resource management.

Eight semantic events, fired from two places:

* **Lifecycle events** -- driven by PyExecutor's main loop through the three
  entry points :meth:`on_batch_scheduled` / :meth:`on_iteration_end` /
  :meth:`on_request_finished`. These derive the four per-request lifecycle
  events and fan them out:

  ===========================  ====================================================
  event                        meaning
  ===========================  ====================================================
  ``on_request_init``          request admitted (first time scheduled)
  ``on_context_end``           request finished its context (prefill) phase
  ``on_generation_step_end``   once per generation (decode) iteration
  ``on_request_finish``        request completed / aborted
  ===========================  ====================================================

* **Attention events** -- fired directly from ``TrtllmAttention.forward`` via
  ``metadata.coordinator``, per layer:

  - ``on_context_attention`` / ``on_generation_attention`` may return
    sparse-attention metadata (single-source: at most one executor may return
    non-None per call).
  - ``on_context_attention_end`` / ``on_generation_attention_end`` are
    side-effect only (e.g. stash q/k/output for unified eviction).

The coordinator enforces a single executor per axis (intra-axis stacking is
rejected at init) and a deterministic cross-axis dispatch order. Today only the
``"sparse"`` axis has a concrete executor (:class:`SparseAttentionExecutor`).
"""

from itertools import chain
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

from .kv_cache_compression_executor import BaseKVCacheCompressionExecutor, SparseAttentionExecutor

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Cross-axis dispatch order per event. Each event name maps to a list of axis
# identifiers in dispatch order; an executor whose axis is absent for a given
# event is silently skipped for that event.
_EVENT_AXIS_ORDER: Dict[str, List[str]] = {
    "on_request_init": ["sparse", "storage"],
    "on_context_attention": ["sparse"],
    "on_context_attention_end": ["sparse", "storage"],
    "on_context_end": ["sparse", "storage"],
    "on_generation_attention": ["sparse"],
    "on_generation_attention_end": ["sparse", "storage"],
    "on_generation_step_end": ["sparse", "storage"],
    "on_request_finish": ["sparse", "storage"],
}


class KVCacheBehaviorCoordinator:
    """Owns the per-axis executors and fans events out to them.

    Two layers of API:

    * **Event fan-out** (``on_*`` methods) -- dispatch one event to the
      executors registered for it, in :attr:`EVENT_AXIS_ORDER`. The attention
      path calls these directly; tests call them directly too.
    * **Lifecycle driving** (:meth:`on_batch_scheduled` /
      :meth:`on_iteration_end` / :meth:`on_request_finished`) -- the three
      entry points PyExecutor's main loop invokes each iteration. They derive
      the four per-request lifecycle events (init / context-end / step-end /
      finish) from the per-iteration batch view and fan them out.
    """

    #: Public alias of the module-level dispatch-order table. Subclasses may
    #: override on the class to customize per-deployment ordering.
    EVENT_AXIS_ORDER: Dict[str, List[str]] = _EVENT_AXIS_ORDER

    def __init__(self, executors: List[BaseKVCacheCompressionExecutor]) -> None:
        self.executors: List[BaseKVCacheCompressionExecutor] = list(executors)
        self._by_axis: Dict[str, List[BaseKVCacheCompressionExecutor]] = {}
        for e in self.executors:
            self._by_axis.setdefault(e.axis, []).append(e)
        # Bookkeeping to derive per-request lifecycle events from the coarse
        # per-iteration batch views PyExecutor hands us:
        #  - `_seen_req_ids` dedupes on_request_init across iterations.
        #  - `_prev_req_state` detects the context->generation transition that
        #    drives on_context_end.
        self._seen_req_ids: Set[int] = set()
        self._prev_req_state: Dict[int, LlmRequestState] = {}
        self._validate()

    # ================================================================== #
    # Event fan-out -- one dispatch per event.                           #
    # ================================================================== #

    def on_request_init(self, request: "LlmRequest") -> None:
        for e in self._iter_for_event("on_request_init"):
            e.on_request_init(request)

    def on_context_attention(
        self, layer_idx: int, q, k, attn_scores, metadata: "AttentionMetadata"
    ):
        """Prefill-attention event (fired from ``TrtllmAttention.forward`` via
        ``metadata.coordinator``). Single-source: at most one executor may
        return non-None per call."""
        return self._dispatch_single_source(
            "on_context_attention", layer_idx, q, k, attn_scores, metadata
        )

    def on_context_attention_end(
        self, layer_idx: int, q, k, attn_output, metadata: "AttentionMetadata"
    ) -> None:
        """Post-prefill-attention event (after the attention output is
        computed). Side-effect only; no single-source constraint."""
        for e in self._iter_for_event("on_context_attention_end"):
            e.on_context_attention_end(layer_idx, q, k, attn_output, metadata)

    def on_context_end(
        self, request: "LlmRequest", metadata: Optional["AttentionMetadata"]
    ) -> None:
        for e in self._iter_for_event("on_context_end"):
            e.on_context_end(request, metadata)

    def on_generation_attention(
        self, layer_idx: int, q, k, attn_scores, metadata: "AttentionMetadata"
    ):
        """Decode-attention event. Same single-source invariant as
        :meth:`on_context_attention`."""
        return self._dispatch_single_source(
            "on_generation_attention", layer_idx, q, k, attn_scores, metadata
        )

    def on_generation_attention_end(
        self, layer_idx: int, q, k, attn_output, metadata: "AttentionMetadata"
    ) -> None:
        """Post-decode-attention event (side-effect only)."""
        for e in self._iter_for_event("on_generation_attention_end"):
            e.on_generation_attention_end(layer_idx, q, k, attn_output, metadata)

    def on_generation_step_end(
        self, scheduled_batch: "ScheduledRequests", attn_metadata: Optional["AttentionMetadata"]
    ) -> None:
        for e in self._iter_for_event("on_generation_step_end"):
            e.on_generation_step_end(scheduled_batch, attn_metadata)

    def on_request_finish(self, request: "LlmRequest") -> None:
        for e in self._iter_for_event("on_request_finish"):
            e.on_request_finish(request)

    def _dispatch_single_source(self, event: str, layer_idx, q, k, attn_scores, metadata):
        result = None
        for e in self._iter_for_event(event):
            r = getattr(e, event)(layer_idx, q, k, attn_scores, metadata)
            if r is not None:
                if result is not None:
                    raise RuntimeError(
                        f"Multiple executors returned attention metadata from "
                        f"{event}; sparse-attention metadata writes must be "
                        f"single-source."
                    )
                result = r
        return result

    # ================================================================== #
    # Lifecycle driving -- the three entry points PyExecutor calls each   #
    # iteration on ``py_executor.kv_behavior_coordinator``.               #
    # ================================================================== #

    def on_batch_scheduled(self, scheduled_batch: "ScheduledRequests") -> None:
        """Called when a batch is scheduled, before it runs. Fires
        ``on_request_init`` exactly once per request (deduped via
        :attr:`_seen_req_ids`), regardless of how many iterations the request
        stays scheduled."""
        for req in chain(scheduled_batch.context_requests, scheduled_batch.generation_requests):
            rid = req.py_request_id
            if rid not in self._seen_req_ids:
                self.on_request_init(req)
                self._seen_req_ids.add(rid)

    def on_iteration_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> None:
        """Called once after each executor iteration. Fires:
        * ``on_context_end`` for each request that just transitioned out of its
          context (prefill) phase, and
        * ``on_generation_step_end`` once for the iteration.

        PyExecutor flips the request state to ``GENERATION_IN_PROGRESS`` before
        this runs, so the transition is detected as "first time seen in
        GENERATION_IN_PROGRESS" (``prev is None``) OR an explicit
        ``CONTEXT_INIT -> GENERATION_IN_PROGRESS`` edge.
        """
        for req in chain(scheduled_batch.context_requests, scheduled_batch.generation_requests):
            rid = req.py_request_id
            prev = self._prev_req_state.get(rid)
            curr = req.state
            transition_to_gen = curr == LlmRequestState.GENERATION_IN_PROGRESS and (
                prev is None or prev == LlmRequestState.CONTEXT_INIT
            )
            if transition_to_gen:
                self.on_context_end(req, attn_metadata)
            self._prev_req_state[rid] = curr
        self.on_generation_step_end(scheduled_batch, attn_metadata)

    def on_request_finished(self, request: "LlmRequest") -> None:
        """Called when a request completes or is aborted. Drops its
        bookkeeping and fires ``on_request_finish``."""
        rid = request.py_request_id
        self._seen_req_ids.discard(rid)
        self._prev_req_state.pop(rid, None)
        self.on_request_finish(request)

    # ================================================================== #
    # Init / introspection                                                #
    # ================================================================== #

    def _validate(self) -> None:
        for axis, exs in self._by_axis.items():
            if len(exs) > 1:
                raise ValueError(
                    f"Intra-axis stacking not supported: {len(exs)} executors "
                    f"found for axis={axis!r}. Most sparse / storage methods "
                    f"assume sole arbiter; stacking two of the same axis would "
                    f"invalidate per-method correctness assumptions. For "
                    f"intra-axis composition, write a hybrid algorithm subclass."
                )

    def has_axis(self, axis: str) -> bool:
        return axis in self._by_axis and bool(self._by_axis[axis])

    def get_executor(self, axis: str) -> Optional[BaseKVCacheCompressionExecutor]:
        exs = self._by_axis.get(axis, [])
        return exs[0] if exs else None

    def get_sparse_executor(self) -> Optional[SparseAttentionExecutor]:
        return self.get_executor("sparse")  # type: ignore[return-value]

    def _iter_for_event(self, event: str) -> Iterable[BaseKVCacheCompressionExecutor]:
        order = self.EVENT_AXIS_ORDER.get(event, ["sparse", "storage"])
        for axis in order:
            for e in self._by_axis.get(axis, []):
                yield e
