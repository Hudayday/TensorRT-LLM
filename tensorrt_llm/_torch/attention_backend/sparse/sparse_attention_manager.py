"""L2 behavior layer for KV cache compression methods.

This module defines:

- :class:`BaseKVCacheBehaviorManager`: framework-level abstract base for all
  L2 behavior managers (sparse / storage transform / cross-request lifecycle).
  Subclasses must set ``axis`` ClassVar and may override any subset of the 8
  lifecycle hooks (default no-op).

- :class:`SparseAttentionManager`: axis-C convenience subclass for sparse /
  per-token eviction methods (RocketKV-V2-migrated, TriAttention, H2O,
  SnapKV, ...). Currently the only axis subclass shipped; future
  ``KVCacheStorageManager`` (axis B for KVTC etc.) and ``CRCLManager``
  (cross-request lifecycle for Continuum etc.) will be added as sibling
  subclasses of ``BaseKVCacheBehaviorManager`` in Phase 4 / Phase 5.

Architecture (3-layer stack):

- L0 attention kernel — attention math, optional kernel-fused quant decode.
- L1 ``KVCacheManagerV2`` — physical page management, dtype storage, 3-tier.
- L2 behavior (this module) — algorithm orchestration, lifecycle hooks.

Behavior managers hold the underlying ``KVCacheManagerV2`` as a tool and
never inherit from it. The behavior/memory split mirrors how speculative
decoding wires Eagle3 / MTPHiddenStatesManager into PyExecutor.

Hooks (8 total):

  - ``on_request_init`` / ``on_request_finish``
        Request lifecycle (entry / exit).
  - ``on_context_attention`` / ``on_context_end``
        Prefill phase, per attention layer and per phase-boundary.
  - ``on_generation_attention`` / ``on_generation_step_end``
        Decode phase, per attention layer and per generation step.
  - ``on_forward_begin`` / ``on_forward_end``
        Per-forward async coordination (wait pending / trigger async).
        Used primarily by future axis-B (storage) and axis-D (CRCL)
        managers; most axis-C sparse managers leave these as no-op.

A :class:`KVCacheBehaviorCoordinator` (see ``coordinator.py``) owns a list of
``BaseKVCacheBehaviorManager`` instances and dispatches each hook to them in
a deterministic axis-priority order (see ``coordinator.HOOK_ORDER``).
"""

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# ``(indices, offsets)`` tuple consumed by the attention kernel as an input-
# side sparse mask; ``None`` falls back to dense attention.
SparseAttentionIndices = Tuple[torch.Tensor, torch.Tensor]


class BaseKVCacheBehaviorManager:
    """Framework-level base class for all L2 KV-cache behavior managers.

    Subclasses must set ``axis`` ClassVar to one of:

    - ``"sparse"`` — sparse / per-token eviction
      (:class:`SparseAttentionManager`, shipped now).
    - ``"storage"`` — storage / transform-coding
      (:class:`KVCacheStorageManager`, planned Phase 4 for KVTC).
    - ``"crcl"`` — cross-request cache lifecycle
      (:class:`CRCLManager`, planned Phase 5 if α candidate selected).

    All 8 hooks default to no-op; subclasses override what they need. A
    :class:`KVCacheBehaviorCoordinator` instance dispatches each hook to all
    registered managers in deterministic axis-priority order.

    The behavior layer never inherits from any cache / resource manager
    because this layer decides *how* the physical KV is used, not *what*
    physical KV exists. Subclasses hold ``KVCacheManagerV2`` as a tool.
    """

    # ------------------------------------------------------------------ #
    # Class-level metadata                                                #
    # ------------------------------------------------------------------ #

    # Axis identifier — subclass MUST override.
    axis: ClassVar[str] = ""

    # Whether this manager class is compatible with cross-request KV cache
    # reuse (radix-tree / APC block reuse). Default conservative ``False``;
    # subclasses opt in explicitly. The LLM init factory may enforce a mutex:
    # combining a manager with ``supports_kv_cache_reuse=False`` and
    # ``KvCacheConfig.enable_block_reuse=True`` raises a config error at init.
    supports_kv_cache_reuse: ClassVar[bool] = False

    def __init__(self, kv_cache_manager: "KVCacheManagerV2"):
        if not self.axis:
            raise NotImplementedError(
                f"{type(self).__name__} must set the 'axis' ClassVar "
                f"to one of: 'sparse', 'storage', 'crcl'.")
        self.kv_cache_manager = kv_cache_manager

    # ------------------------------------------------------------------ #
    # Request lifecycle hooks                                            #
    # ------------------------------------------------------------------ #

    def on_request_init(self, request: "LlmRequest") -> None:
        """Per-request init hook.

        Override to allocate per-request accumulators (e.g., H2O cumulative
        attention sum buffers indexed by (layer, head, token)). CRCL
        managers (future) check a cross-request pool for hits here.
        """
        pass

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Per-request finish / abort hook.

        Override to release per-request state allocated in
        ``on_request_init``. Underlying KV blocks are still freed by the
        ``KVCacheManagerV2``; subclasses must not free them here. Storage
        managers (future) emit final compressed bytes here; CRCL managers
        (future) promote those to a cross-request pool here.
        """
        pass

    # ------------------------------------------------------------------ #
    # Context (prefill) phase hooks                                      #
    # ------------------------------------------------------------------ #

    def on_context_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Per-layer hook after every context-phase attention forward.

        Fires once per chunk per layer under chunked prefill; once per
        layer otherwise. ``attn_scores`` is populated only when the
        attention kernel instantiation exposes scores (compile-time
        template flag); ``None`` when scores are not materialized.

        Form-I sparse managers return an ``(indices, offsets)`` tuple as an
        input-side sparse mask; form-III (eviction-after-compute) managers,
        storage managers, and CRCL managers return ``None``. The coordinator
        enforces single-source: at most one manager may return non-None per
        attention call.
        """
        return None

    def on_context_end(
        self,
        request: "LlmRequest",
        metadata: "AttentionMetadata",
    ) -> None:
        """Per-request hook fired once after the whole prompt has been
        consumed across all chunks and all layers (phase boundary).

        Override for one-shot prefill-end eviction (SnapKV, RocketKV Stage
        I), to commit retrieval-head decisions (DuoAttention), or to encode
        the prefilled cache for storage compression (KVTC).
        """
        pass

    # ------------------------------------------------------------------ #
    # Generation (decode) phase hooks                                    #
    # ------------------------------------------------------------------ #

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Per-layer hook after every generation-phase attention forward.

        Same single-source invariant as ``on_context_attention``: at most
        one manager may return non-None metadata. Used by form-I sparse
        managers (RocketKV Stage II HSA, Quest) to return query-aware
        masks; form-III managers (TriAttention) and non-sparse managers
        return ``None``.
        """
        return None

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Cross-layer, cross-batch hook fired once per generation step
        after every layer's forward completes.

        Override for periodic eviction (TriAttention ``beta=128`` trigger),
        budget-triggered eviction (H2O, Scissorhands), or runtime cleanup
        (RocketKV rewind). Storage managers may invalidate active
        compressed copies here if the cache shape changed (e.g., after a
        sparse evict).
        """
        pass

    # ------------------------------------------------------------------ #
    # Forward-pass async coordination hooks                              #
    # (added for async storage / CRCL operations)                        #
    # ------------------------------------------------------------------ #

    def on_forward_begin(self, forward_batch: Any) -> None:
        """Per-forward hook BEFORE the forward pass starts.

        Storage managers (future) wait for pending async decompress
        streams here; CRCL managers (future) may resume cache from
        cross-request pool here. Most axis-C sparse managers leave this
        as no-op.
        """
        pass

    def on_forward_end(self, forward_batch: Any) -> None:
        """Per-forward hook AFTER the forward pass completes.

        Storage managers (future) trigger async re-compress here; CRCL
        managers (future) update TTL counters and GC expired entries
        here. Most axis-C sparse managers leave this as no-op.
        """
        pass

    # ------------------------------------------------------------------ #
    # Capability introspection                                            #
    # ------------------------------------------------------------------ #

    def implements(self, hook_name: str) -> bool:
        """Return ``True`` if this subclass actually overrides
        ``hook_name`` (treating the default no-op inherited from
        :class:`BaseKVCacheBehaviorManager` as not implementing).

        Used by the coordinator to optionally skip-iterate managers that
        don't implement a particular hook (perf micro-optimization, off by
        default).
        """
        own_method = getattr(type(self), hook_name, None)
        if own_method is None:
            return False
        base_method = getattr(BaseKVCacheBehaviorManager, hook_name, None)
        if base_method is None:
            return False
        # In Python 3, accessing a method via class returns the function directly
        # (no .__func__ needed). MRO lookup means an inherited (non-overridden) hook
        # returns the SAME function object as the base, so identity check suffices.
        return own_method is not base_method


class SparseAttentionManager(BaseKVCacheBehaviorManager):
    """Axis-C subclass for sparse / per-token eviction methods.

    Subclasses: :class:`TriAttention` (Phase 3 first instance), and future
    H2O / SnapKV / RocketKV-V2-migrated. The legacy RocketKV / DSA /
    skip_softmax follow the older 5-class plugin pattern (separate
    cache-manager + attention shim classes) and do NOT inherit from this
    base.

    The framework base is :class:`BaseKVCacheBehaviorManager`;
    ``SparseAttentionManager`` is the axis-specific convenience subclass
    that sparse algorithms inherit from. Existing call sites that import
    ``SparseAttentionManager`` continue to work unchanged — TriAttention
    still inherits from it.

    Future axis-C-specific helpers (e.g., ``_read_req_k_cache`` for K-cache
    pool readback, ``_compact_req`` for physical eviction through the V2
    ``compact_request_cache`` wrapper API) will live on this class so all
    sparse subclasses share them without duplication.
    """

    axis: ClassVar[str] = "sparse"

    # ------------------------------------------------------------------ #
    # Axis-C-specific capability declarations (subclass overrides)        #
    # ------------------------------------------------------------------ #

    # ``True`` if this method physically compacts the cache (TriAttention,
    # H2O); ``False`` if it operates form-I (RocketKV Stage II HSA, DSA,
    # Quest) — returns a sparse mask but does not mutate cache contents.
    is_form_iii_evict: ClassVar[bool] = False
