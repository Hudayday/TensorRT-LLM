"""RocketKV-V2 — sparse attention via 2-stage HSA (Hybrid Sparse Attention).

This is the v17 V2-migrated RocketKV skeleton (Phase 7 target). The legacy
V1 path lives in :mod:`sparse.rocket` (5-class plugin pattern with a
:class:`RocketKVCacheManager` subclass of V1 ``KVCacheManager`` — see commit
history). This file is named ``rocketkv.py`` (vs the legacy ``rocket.py``) so
both can coexist during migration.

Paper: arXiv:2502.14837. See ``~/docs/kv-reduction/07-rocketkv-deep-dive.md``
for paper-code cross-reference and ``~/docs/kv-reduction/27-framework-walkthrough-triattention-rocketkv.md``
§3 for V2 migration architectural design.

This file defines BOTH halves of the RocketKV plug-in:

- :class:`RocketKV` — L2 behavior executor (subclass of
  :class:`SparseAttentionExecutor`). Drives Stage I + Stage II algorithm
  work via lifecycle hooks (``on_context_attention`` / ``on_generation_attention``).
- :class:`RocketKVTrtllmAttention` + :class:`RocketKVVanillaAttention` —
  L0 attention shims (subclasses of ``TrtllmAttention`` / ``VanillaAttention``).
  Carry RocketKV-specific metadata (KT cache block offsets, prompt budget)
  and consume the ``(indices, offsets)`` sparse mask produced by the
  executor. The attention factory in ``sparse/utils.py`` routes to them
  when ``algorithm="rocketkv"``.

Key design choices (per 5/27 architectural discussion, doc 27 §3):

- **Sparse-mask method** — Stage II returns an ``(indices, offsets)`` sparse
  mask via :meth:`on_generation_attention`; cache contents are NOT modified.
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
:meth:`on_generation_attention` are stubs (return ``None``). The attention
shim classes carry the class hierarchy and metadata fields but inherit
forward() from their base classes; method-specific kernels (paged KT bmm,
triton scoring, etc.) are Phase 7 ports from ``rocket.py``.

Existing 5-class V1 RocketKV (``sparse/rocket.py``) stays in place. The
``rocketkv`` algorithm flag is reserved for this V2-migrated version when it
ships; until then, ``method="rocket"`` keeps routing to the V1 path via
the legacy ``is_behavior_layer_method=False`` config branch.
"""

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.attention_backend.vanilla import (
    VanillaAttention, VanillaAttentionMetadata)

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


# ====================================================================== #
# L0 attention shims — RocketKV-specific attention classes                #
#                                                                        #
# These subclass the framework attention bases and add RocketKV-specific  #
# metadata (KT cache offsets, prompt budget). The attention forward()    #
# bodies in Phase 7 will consume the ``(indices, offsets)`` sparse mask  #
# the executor returns from ``on_generation_attention`` and skip pages   #
# outside the HSA top-K budget. Until Phase 7 they fall through to the   #
# base-class dense forward.                                              #
#                                                                        #
# Routed from ``sparse/utils.py:get_*_sparse_attn_attention_backend``    #
# when ``sparse_attn_config.algorithm == "rocketkv"``.                   #
# ====================================================================== #


class RocketKVTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Sparse-mask-aware metadata for RocketKV under the TRT-LLM backend.

    Skeleton: structural placeholder so the factory dispatch + executor
    pipeline can be exercised end-to-end. Real metadata population is
    Phase 7 work — port from ``rocket.py:RocketTrtllmAttentionMetadata``
    (line 38).

    Differences from V1 metadata (when ported):

    - V1 pre-allocates ``kt_cache_block_offsets`` tensors via a V1-cache-
      manager API. In V17, these come from the KT_CACHE auxiliary pool
      via the V2 generic ``get_buffers(layer, data_role=KT_CACHE)`` API.
    - The sparse mask itself (``indices``/``offsets``) is provided by the
      executor's ``on_generation_attention`` return value rather than
      computed inline here.
    """

    def __post_init__(self):
        super().__post_init__()
        # TODO (Phase 7): populate RocketKV-specific fields here.
        # Reference: rocket.py:RocketTrtllmAttentionMetadata (line 38).


class RocketKVTrtllmAttention(TrtllmAttention):
    """Sparse attention forward for RocketKV under the TRT-LLM backend.

    Skeleton: subclasses :class:`TrtllmAttention` and sets
    :attr:`Metadata` to :class:`RocketKVTrtllmAttentionMetadata`. In
    Phase 7 the ``forward_context`` / ``forward_generation`` methods
    will be overridden to consume the executor-provided sparse mask;
    until then it inherits the dense forward from
    :class:`TrtllmAttention` (so the pipeline runs end-to-end but
    behaves like dense attention until algorithm bodies land).

    Port reference: ``rocket.py:RocketTrtllmAttention`` (line 318).
    The legacy V1 class is ~270 lines including triton kernel calls
    for paged KT bmm scoring (Stage II). In V17 those scoring kernels
    live inside :meth:`RocketKV.on_generation_attention`; the
    attention class only needs to consume the resulting mask.
    """

    Metadata = RocketKVTrtllmAttentionMetadata

    # TODO (Phase 7): override forward_context() / forward_generation() to
    # consume the (indices, offsets) sparse mask from the executor and
    # skip non-selected pages. See rocket.py:RocketTrtllmAttention.forward
    # (line 487+) for the V1 paged-attention path.


class RocketKVVanillaAttentionMetadata(VanillaAttentionMetadata):
    """Sparse-mask-aware metadata for RocketKV under the vanilla backend.

    Skeleton parallel to :class:`RocketKVTrtllmAttentionMetadata` but for
    the eager vanilla attention backend (used for unit-test parity).
    Port reference: ``rocket.py:RocketVanillaAttentionMetadata`` (line 580).
    """

    def __post_init__(self):
        super().__post_init__()
        # TODO (Phase 7): populate prompt_budget / kt cache offsets here.
        # Reference: rocket.py:RocketVanillaAttentionMetadata (line 580).


class RocketKVVanillaAttention(VanillaAttention):
    """Sparse attention forward for RocketKV under the vanilla backend.

    Skeleton. Port reference: ``rocket.py:RocketVanillaAttention`` (line 624).
    """

    Metadata = RocketKVVanillaAttentionMetadata

    # TODO (Phase 7): override forward() to consume executor-provided sparse
    # mask. See rocket.py:RocketVanillaAttention (line 624+).


# ====================================================================== #
# L2 behavior executor — Stage I + Stage II algorithm orchestration       #
# ====================================================================== #


class RocketKV(SparseAttentionExecutor):
    """V2-migrated RocketKV: sparse-mask HSA + KT_CACHE auxiliary pool
    (skeleton).

    See module docstring for design choices. Algorithm body待写 (Phase 7
    parallel to TriAttention M3.1).

    User-facing API (planned):

    .. code-block:: python

        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import RocketKVSparseAttentionConfig, KvCacheConfig

        llm = LLM(
            model="meta-llama/Llama-3.1-8B",
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
            sparse_attention_config=RocketKVSparseAttentionConfig(
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

    # Returns a sparse mask via on_*_attention; does NOT mutate cache
    # contents. (Compare TriAttention which sets physically_evicts_kv=True.)
    physically_evicts_kv: ClassVar[bool] = False

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

        Currently a stub. Returns ``None`` (context phase here does not
        emit a sparse mask; the full prompt is attended dense).

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
