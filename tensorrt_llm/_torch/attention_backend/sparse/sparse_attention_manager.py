"""Sparse attention behavior manager abstract base.

A ``SparseAttentionManager`` subclass owns the lifecycle of one sparse-attention
/ KV-compression method (RocketKV, TriAttention, H2O, SnapKV, ...). It decides
when to evict KV blocks, which tokens to keep, and what sparse mask (if any) to
feed back to the attention kernel. Physical KV storage stays owned by the
underlying ``KVCacheManagerV2``, which the subclass holds as a tool and never
inherits from. This separates the *behavior* layer (where / when / what to
sparsify) from the *memory* layer (how blocks are allocated and freed).

Hooks split across the two LLM-inference phases plus request lifecycle. All
hooks default to no-op; subclasses override only the hooks they need. Wiring:

  - on_context_attention      attention forward path, context (prefill) phase
  - on_context_end            after the whole prompt finishes (all chunks /
                              all layers), once per request
  - on_generation_attention   attention forward path, generation (decode) phase
  - on_generation_step_end    PyExecutor iteration end (after all layers, across
                              batch)
  - on_request_init / on_request_finish
                              PyExecutor request lifecycle (entry / exit)

This module intentionally does not inherit from ``BaseResourceManager``: a
sparse attention manager is a *behavior* layer, not a physical resource owner.
Framework wiring is done with explicit hook call sites in PyExecutor and the
attention forward path, mirroring how speculative decoding wires Eagle3 /
MTPHiddenStatesManager.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# (indices, offsets) tuple consumed by the attention kernel as an input-side
# sparse mask; ``None`` falls back to dense attention.
SparseAttentionIndices = Tuple[torch.Tensor, torch.Tensor]


class SparseAttentionManager:
    """Behavior layer for sparse-attention / KV-compression methods.

    See module docstring for the hook lifecycle. Subclasses hold the underlying
    ``KVCacheManagerV2`` as a tool; they do not inherit from any cache /
    resource manager because this layer decides *how* the physical KV is used,
    not *what* physical KV exists.
    """

    def __init__(self, kv_cache_manager: "KVCacheManagerV2"):
        self.kv_cache_manager = kv_cache_manager

    # ------------------------------------------------------------------ #
    # Context (prefill) phase                                            #
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

        Fires once per chunk per layer under chunked prefill; once per layer
        otherwise. ``attn_scores`` is populated only when the attention kernel
        instantiation exposes scores (compile-time template flag); ``None``
        when scores are not materialized. Override to accumulate per-token
        importance state (H2O, Scissorhands), build a per-page summary index
        (Quest), or return an input-side ``(indices, offsets)`` sparse mask.
        Return ``None`` to fall back to dense attention.
        """
        return None

    def on_context_end(
        self,
        request: "LlmRequest",
        metadata: "AttentionMetadata",
    ) -> None:
        """Per-request hook fired once after the whole prompt has been
        consumed across all chunks and all layers.

        Override for one-shot prefill-end eviction (SnapKV, RocketKV Stage I)
        or to commit retrieval-head decisions (DuoAttention). Physical
        eviction goes through ``self.kv_cache_manager``.
        """
        pass

    # ------------------------------------------------------------------ #
    # Generation (decode) phase                                          #
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

        ``attn_scores`` semantics match ``on_context_attention``. Override to
        return a query-aware input-side sparse mask (RocketKV Stage II HSA,
        Quest) or to accumulate score state (H2O). Return ``None`` to fall
        back to dense attention on the (possibly already compacted) cache.
        """
        return None

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Cross-layer, cross-batch hook fired once per generation step after
        every layer's forward completes.

        Override for periodic eviction (TriAttention, ``beta=128``), budget-
        triggered eviction (H2O, Scissorhands), or runtime cleanup
        (RocketKV rewind).
        """
        pass

    # ------------------------------------------------------------------ #
    # Request lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def on_request_init(self, request: "LlmRequest") -> None:
        """Per-request init hook.

        Override to allocate per-request accumulators (e.g., H2O cumulative
        attention sum buffers indexed by (layer, head, token)).
        """
        pass

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Per-request finish / abort hook.

        Override to release per-request state allocated in ``on_request_init``.
        Underlying KV blocks are still freed by the ``KVCacheManagerV2``;
        subclasses must not free them here.
        """
        pass
