"""RocketKV-V2 — sparse attention via 2-stage HSA (Hybrid Sparse Attention).

This is the v17 V2-migrated RocketKV skeleton (Phase 7 target). The legacy
V1 path lives in :mod:`sparse.rocket` (5-class plugin pattern with a
:class:`RocketKVCacheManager` subclass of V1 ``KVCacheManager`` — see commit
history). This file is named ``rocketkv.py`` (vs the legacy ``rocket.py``) so
both can coexist during migration.

Paper: arXiv:2502.14837. See ``~/docs/kv-reduction/07-rocketkv-deep-dive.md``
for paper-code cross-reference and ``~/docs/kv-reduction/27-framework-walkthrough-triattention-rocketkv.md``
§3 for V2 migration architectural design.

Key design choices (per 5/27 architectural discussion, doc 27 §3):

- **Form-I sparse** — Stage II returns ``(indices, offsets)`` sparse mask via
  :meth:`on_generation_attention`; cache contents are NOT modified.
- **KT_CACHE auxiliary pool — Pattern 2 declarative BufferConfig** — V2 stays
  unchanged. PyExecutor factory adds a ``KT_CACHE`` ``BufferConfig`` per layer
  at V2 instantiation, with ``tokens_per_block_override=kt_tokens_per_block``.
  Page IDs share with KEY (per-layer multi-pool design, same mechanism as
  NVFP4 ``KEY_BLOCK_SCALE``). No V2 subclass needed (``kv_cache_manager_class``
  ClassVar stays ``None``).
- **Stage I** — :meth:`on_context_attention` computes per-page KT summary
  from K and writes to ``KT_CACHE`` pool via the V2 generic
  ``write_kt_cache(req, layer, data)`` wrapper API (added by factory).
- **Stage II** — :meth:`on_generation_attention` reads ``KT_CACHE`` via V2
  generic ``get_buffers(layer, data_role=KT_CACHE)`` API, computes
  query-aware HSA mask within ``prompt_budget``, returns mask tuple.

**Status: SKELETON ONLY** — :meth:`on_context_attention` and
:meth:`on_generation_attention` are stubs (return ``None``). Algorithm body
is Phase 7 work (parallel to TriAttention M3.1 Phase 3 work).

Existing 5-class V1 RocketKV (``sparse/rocket.py``) stays in place. The
``rocketkv`` algorithm flag is reserved for this V2-migrated version when it
ships; until then, ``method="rocket"`` keeps routing to the V1 path via
the legacy ``is_behavior_layer_method=False`` config branch.
"""

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
    SparseAttentionIndices,
    SparseAttentionExecutor,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2


class RocketKV(SparseAttentionExecutor):
    """V2-migrated RocketKV form-I HSA + KT_CACHE auxiliary pool (skeleton).

    See module docstring for design choices. Algorithm body待写 (Phase 7
    parallel to TriAttention M3.1).

    User-facing API (planned):

    .. code-block:: python

        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import RocketSparseAttentionConfig, KvCacheConfig

        llm = LLM(
            model="meta-llama/Llama-3.1-8B",
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
            sparse_attention_config=RocketSparseAttentionConfig(
                method="rocketkv",            # routes to THIS class
                page_size=16,
                prompt_budget=2048,
                kt_cache_dtype="bfloat16",
                kt_tokens_per_block=2,
            ),
        )
    """

    # ------------------------------------------------------------------ #
    # Capability declarations                                             #
    # ------------------------------------------------------------------ #

    axis: ClassVar[str] = "sparse"

    # Form-I sparse: returns mask via on_*_attention, does NOT mutate cache.
    is_form_iii_evict: ClassVar[bool] = False

    # KT cache is request-specific and built per-context; cannot be reused
    # across requests (cross-request KT would have wrong K-source for current
    # query). Enforced via factory mutex at LLM init.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Pattern 1 + Pattern 2: default plain V2 (None ClassVar). KT_CACHE pool
    # is added by PyExecutor factory at V2 instantiation via declarative
    # BufferConfig — no V2 subclass needed.
    kv_cache_manager_class: ClassVar[Optional[type]] = None

    # ------------------------------------------------------------------ #
    # Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        page_size: int = 16,
        prompt_budget: int = 2048,
        kt_cache_dtype: str = "bfloat16",
        kt_tokens_per_block: Optional[int] = None,
    ):
        super().__init__(kv_cache_manager)
        self.page_size = page_size
        self.prompt_budget = prompt_budget
        self.kt_cache_dtype = kt_cache_dtype
        # kt_tokens_per_block normally computed in PyExecutor factory and
        # passed in alongside the matching BufferConfig allocation.
        self.kt_tokens_per_block = kt_tokens_per_block

        # Per-request state placeholder (Stage I/II algorithm待:
        # e.g. self._kt_built_per_req: dict[req_id, set[layer_idx]] = {}).

    # ------------------------------------------------------------------ #
    # Stage I — context-phase hook (build KT summary per page)            #
    # ------------------------------------------------------------------ #

    def on_context_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Stage I — compute per-page KT summary from K, write to KT_CACHE.

        Currently a stub. Returns ``None`` (context phase form-I does not
        return a sparse mask; full prompt attended dense).

        TODO (Phase 7 algorithm body):

        1. Get per-page indices for current req via
           ``self.kv_cache_manager.get_batch_cache_indices(...)``.
        2. Compute per-page KT summary (e.g., concat max/min of K per page).
        3. Write to ``KT_CACHE`` pool via
           ``self.kv_cache_manager.write_kt_cache(req, layer_idx, kt_data)``
           (factory-added V2 generic API).
        """
        return None

    # ------------------------------------------------------------------ #
    # Stage II — generation-phase hook (query-aware HSA mask)             #
    # ------------------------------------------------------------------ #

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Stage II — read KT_CACHE, compute query-aware HSA mask.

        Currently a stub. Returns ``None`` (falls back to dense attention).

        TODO (Phase 7 algorithm body):

        1. Read ``KT_CACHE`` pool view via
           ``self.kv_cache_manager.get_buffers(layer_idx, data_role=Role.KT_CACHE)``.
        2. Get per-req KT indices via
           ``self.kv_cache_manager.get_batch_cache_indices(req_ids, layer_idx, data_role=Role.KT_CACHE)``.
        3. Compute per-page score: ``page_score = q · kt_summary``.
        4. Select top-K pages within ``self.prompt_budget``.
        5. Build ``(indices, offsets)`` sparse mask tuple for kernel
           consumption.
        """
        return None
