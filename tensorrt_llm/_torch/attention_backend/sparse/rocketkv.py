"""RocketKV-V17 — V2-migrated port of V1 RocketKV (see sparse/rocket.py).

This file ports V1 ``RocketTrtllmAttention`` / ``RocketTrtllmAttentionMetadata``
/ ``RocketKVCacheManager`` algorithm body into the v17 Path A framework
(``BaseKVCacheCompressionExecutor`` + ``KVCacheBehaviorCoordinator``,
inheriting ``BaseResourceManager``). Side-by-side comparison with V1 is
the explicit goal — kernel calls, metadata field names, and algorithm
sequence are kept identical.

Paper: arxiv 2502.14837 — 2-stage hybrid:
- Stage I (prefill): SnapKV-style top-pB physical eviction + per-page KT
  summary build streaming through every attention layer. Cache shrinks
  to ``prompt_budget`` tokens by end of prefill.
- Stage II (decode): query-aware HSA mask over the shrunk cache, using
  KT summaries as the page-level lookup table.

Mapping V1 → V17:

| V1 location                                            | V17 location                                 |
|--------------------------------------------------------|----------------------------------------------|
| ``RocketTrtllmAttention.sparse_kv_predict``            | ``RocketKV.on_context_attention`` (HOOK 2)   |
| ``RocketTrtllmAttention.sparse_attn_predict``          | ``RocketKV.on_generation_attention`` (HOOK 4)|
| ``RocketKVCacheManager.update_resources`` rewind logic | ``RocketKV.on_context_end`` (HOOK 3)         |
| ``RocketKVCacheManager.prepare_resources``             | ``RocketKV.on_request_init`` (HOOK 1)        |
| ``RocketKVCacheManager.__init__`` (KT pool alloc)      | ``RocketKV.__init__`` (Pattern 1: executor   |
|                                                        | owns ``kt_cache_pool_per_layer`` + BlockManager) |
| ``RocketTrtllmAttentionMetadata.__post_init__/prepare``| ``RocketKVTrtllmAttentionMetadata`` (direct  |
|                                                        | port, KT-related fields access executor via  |
|                                                        | ``self.coordinator.get_executor("sparse")``) |

KT_CACHE storage decision: **Pattern 1 (executor-owned)** for now. V2
declarative BufferConfig (Pattern 2) integration is a separate Phase 4+
deliverable — when V2 supports KT_CACHE BufferConfig, this file's
``__init__`` allocation moves to V2 factory and the executor reaches into
V2 via ``write_kt_cache(req, layer, data)``. No algorithm-body change at
that point — kernel calls stay identical.

Legacy V1 (``sparse/rocket.py``) stays in tree as the reference
implementation + comparison baseline. Both coexist via the
algorithm-discriminator (V1 = ``algorithm="rocket"``, V17 =
``algorithm="rocketkv"``).
"""

import math
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import torch
from triton import next_power_of_2

from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.attention_backend.vanilla import (
    VanillaAttention, VanillaAttentionMetadata)
# Pattern 2: V2 KVCacheManagerV2 owns the KT_CACHE pool via Role.KT_CACHE
# BufferConfig (resource_manager.py:Role.KT_CACHE). Executor delegates
# allocation + lifecycle to V2 instead of holding its own BlockManager
# (which was the Pattern 1 shortcut before 2026-05-28).
from tensorrt_llm._utils import prefer_pinned

from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
    SparseAttentionExecutor,
    SparseAttentionIndices,
)
from .kernel import (triton_bmm, triton_flatten_to_batch,
                     triton_rocket_batch_to_flatten,
                     triton_rocket_paged_kt_cache_bmm, triton_rocket_qk_split,
                     triton_rocket_reduce_scores,
                     triton_rocket_update_kt_cache_ctx,
                     triton_rocket_update_kt_cache_gen, triton_softmax,
                     triton_topk)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# =========================================================================
# L0 attention shim — minimal subclasses for backend routing                #
#                                                                          #
# These are intentionally thin: the framework HOOK 2/4 callbacks fire from #
# the base ``TrtllmAttention.forward`` / ``VanillaAttention.forward`` via  #
# ``metadata.coordinator.on_*_attention(...)``. The shim's only job is to  #
# carry the RocketKV-specific Metadata class (which holds prompt_budget,   #
# kt_cache_block_offsets, etc.) so backend routing picks them up.          #
# =========================================================================


class RocketKVTrtllmAttention(TrtllmAttention):
    """RocketKV V17 attention shim — TRT-LLM backend.

    Identical structure to V1 ``RocketTrtllmAttention`` except:
    - No ``sparse_kv_predict`` / ``sparse_attn_predict`` method overrides;
      those algorithm bodies live in :class:`RocketKV` executor's HOOK 2/4
      callbacks (fired by base ``TrtllmAttention.forward`` via
      ``metadata.coordinator``).
    - Holds :class:`RocketKVTrtllmAttentionMetadata` for backend routing.
    """

    Metadata: ClassVar[type] = None  # set below after metadata class defined


class RocketKVVanillaAttention(VanillaAttention):
    """RocketKV V17 attention shim — vanilla backend (used in tests).

    Algorithm body (HOOK 2/4 in :class:`RocketKV`) is the single source
    of truth; vanilla path uses Python/torch kernels instead of triton.
    """

    Metadata: ClassVar[type] = None  # set below


# =========================================================================
# L0 metadata — direct port of V1 RocketTrtllmAttentionMetadata             #
#                                                                          #
# All buffer fields kept verbatim for side-by-side comparison with V1.     #
# Access to KT cache pool / max_kt_blocks_per_seq is rerouted via          #
# ``self._rocket_executor`` (resolved through metadata.coordinator) instead#
# of the V1 path through ``self.kv_cache_manager.<...>``.                  #
# =========================================================================


class RocketKVTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Direct port of V1 ``RocketTrtllmAttentionMetadata`` (rocket.py:38).

    Field names + buffer layout match V1 1:1. KT-cache-related access
    routes through the coordinator-resolved executor instead of the V1
    cache-manager subclass.
    """

    @property
    def _rocket_executor(self) -> Optional["RocketKV"]:
        """Resolve the RocketKV executor instance through the coordinator.
        Returns ``None`` if no behavior-layer sparse method is configured
        (e.g., dummy attn_metadata init path); the metadata then skips
        KT-cache-related buffer setup."""
        if getattr(self, "coordinator", None) is None:
            return None
        return self.coordinator.get_executor("sparse")  # type: ignore[return-value]

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        # V17 ``RocketKVSparseAttentionConfig`` doesn't carry window_size /
        # topk / kernel_size — they're carried by the executor instance.
        # Resolve through coordinator if available; fall back to defaults
        # matching V1 RocketSparseAttentionConfig.
        e = self._rocket_executor
        self.window_size = e.window_size if e else 32
        self.page_size = self.sparse_attention_config.page_size
        self.topk = e.topk if e else 256

        assert self.page_size == next_power_of_2(
            self.page_size), "Page size must be a power of 2"

        capture_graph = self.is_cuda_graph

        # ---- Cumulative valid sequence lengths for query and key (V1 line 53)
        self.q_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="q_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.q_cu_seqlens = torch.zeros_like(self.q_cu_seqlens_cuda,
                                             device='cpu',
                                             dtype=torch.int32)

        self.k_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="k_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.k_cu_seqlens = torch.zeros_like(self.k_cu_seqlens_cuda,
                                             device='cpu',
                                             dtype=torch.int32)

        # ---- Context length of RocketKV key for each valid sequence (V1 line 77)
        self.k_context_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="k_context_lens_cuda",
            capture_graph=capture_graph,
        )
        self.k_context_lens = torch.zeros_like(self.k_context_lens_cuda,
                                               device='cpu',
                                               dtype=torch.int32)

        # ---- Start index of RocketKV key for each valid sequence (V1 line 90)
        self.k_context_start_cuda = self.get_empty(
            None,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="k_context_start_cuda",
            capture_graph=capture_graph,
        )

        # ---- Cumulative context lengths (V1 line 99)
        self.context_cumsum_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="context_cumsum_cuda",
            capture_graph=capture_graph,
        )
        self.context_cumsum = torch.zeros_like(self.context_cumsum_cuda,
                                               device='cpu',
                                               dtype=torch.int32)

        # ---- Sparse kv indices offsets for context phase (V1 line 111)
        self.sparse_offsets_ctx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="sparse_offsets_ctx_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_ctx = torch.zeros_like(self.sparse_offsets_ctx_cuda,
                                                   device='cpu',
                                                   dtype=torch.int32)

        # ---- Valid sequence indices (V1 line 123)
        self.valid_seq_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, ),
            dtype=torch.int32,
            cache_name="valid_seq_indices_cuda",
            capture_graph=capture_graph,
        )

        # ---- KT cache block offsets (V1 line 132 — accesses executor for max_kt_blocks_per_seq)
        max_kt_blocks = e.max_kt_blocks_per_seq if e else 0
        if max_kt_blocks > 0:
            self.kt_cache_block_offsets = self.get_empty(
                self.cuda_graph_buffers,
                [self.max_num_sequences, max_kt_blocks],
                dtype=torch.int32,
                cache_name="kt_cache_block_offsets",
                capture_graph=capture_graph,
            )
            self.host_kt_cache_block_offsets = torch.zeros_like(
                self.kt_cache_block_offsets,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
        else:
            # No executor wired (dummy/init path) — leave as None; algorithm
            # body will early-return on access.
            self.kt_cache_block_offsets = None
            self.host_kt_cache_block_offsets = None

        # ---- Number of KT tokens (V1 line 150)
        self.num_kt_tokens = torch.empty(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )

        # ---- Cumulative KT lengths (V1 line 157)
        self.cum_kt_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="cum_kt_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cum_kt_lens = torch.zeros_like(self.cum_kt_lens_cuda,
                                            device='cpu',
                                            dtype=torch.int32)

        # ---- Sparse attn indices offsets for generation phase (V1 line 169)
        self.sparse_offsets_gen_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            dtype=torch.int32,
            cache_name="sparse_offsets_gen_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_gen = torch.zeros_like(self.sparse_offsets_gen_cuda,
                                                   device='cpu',
                                                   dtype=torch.int32)

        # ---- Maximum number of KT tokens (V1 line 181)
        self.max_kt_tokens = (self.max_seq_len + self.page_size -
                              1) // self.page_size

    @property
    def kt_tokens_per_block(self) -> Optional[int]:
        """V1 line 185 — proxy to executor's kt_tokens_per_block. Used by
        triton kernels (passed as a kernel arg)."""
        e = self._rocket_executor
        return e.kt_tokens_per_block if e else None

    def prepare(self):
        """Direct port of V1 ``RocketTrtllmAttentionMetadata.prepare`` (line 192).

        Per-iteration setup: rewind ``num_cached_tokens_per_seq`` for
        sequences whose prompt exceeded ``prompt_budget``, clamp prompt_lens
        to prompt_budget for generation requests, and build the various
        cu-seqlen / sparse-offset / valid-mask CUDA buffers used by the
        triton kernels in HOOK 2/4.
        """
        if self.kv_cache_manager is not None:
            num_contexts = self.num_contexts
            num_generations = self.num_generations
            num_requests = num_contexts + num_generations

            # V1 line 198: rewind num_cached_tokens_per_seq. This block
            # assumes the cache has been physically rewound to
            # prompt_budget+1 tokens (V1's update_resources path). Under
            # V2 we may not have done that physical evict yet — skip the
            # adjustment when ROCKETKV_DISABLE_CACHED_TOKENS_ADJ=1.
            import os as _osm0
            skip_adj = _osm0.environ.get(
                "ROCKETKV_DISABLE_CACHED_TOKENS_ADJ") == "1"
            if not skip_adj:
                for i in range(num_requests):
                    if i < num_contexts:
                        self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
                    else:
                        if self.prompt_lens[i] > self.prompt_budget:
                            self.kv_cache_params.num_cached_tokens_per_seq[
                                i] += (self.prompt_budget -
                                       self.prompt_lens[i])

        super().prepare()

        if self.kv_cache_manager is not None:
            # V1 line 209: clamp prompt_lens for gen requests.
            #
            # 2026-05-28 ISOLATION: skip the clamp when
            # ROCKETKV_DISABLE_PROMPT_LENS_CLAMP=1. V1 clamp is paired
            # with V1's update_resources cache rewind to prompt_budget+1;
            # under V2 we can't easily do the same physical evict, so
            # toggling this lets us test acc parity without the clamp.
            import os as _osm
            skip_clamp = _osm.environ.get(
                "ROCKETKV_DISABLE_PROMPT_LENS_CLAMP") == "1"
            if not skip_clamp:
                _prompt_lens = self.prompt_lens.copy()
                for i in range(num_requests):
                    if i >= num_contexts:
                        _prompt_lens[i] = min(_prompt_lens[i],
                                              self.prompt_budget)
                _prompt_lens = torch.tensor(_prompt_lens,
                                            dtype=torch.int,
                                            device='cpu')
                self.prompt_lens_cpu[:self.num_seqs].copy_(_prompt_lens)
                self.prompt_lens_cuda[:self.num_seqs].copy_(
                    self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)
            self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.
                                                                  num_seqs]
            self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]

            # V1 line 224: copy KT block offsets from executor (was V1 cache mgr)
            e = self._rocket_executor
            if e is not None and self.host_kt_cache_block_offsets is not None:
                e.copy_kt_block_offsets(self.request_ids,
                                        self.host_kt_cache_block_offsets)
                self.kt_cache_block_offsets[:self.num_seqs].copy_(
                    self.host_kt_cache_block_offsets[:self.num_seqs],
                    non_blocking=True)

        # ---- Context phase setup (V1 line 231)
        self.context_cumsum[1:self.num_contexts + 1] = torch.cumsum(
            self.prompt_lens_cpu[:self.num_contexts], dim=0)
        self.context_cumsum_cuda[:self.num_contexts + 1].copy_(
            self.context_cumsum[:self.num_contexts + 1], non_blocking=True)

        # V1 line 237: filter sequences too short for sparse kv prediction
        valid_mask = self.prompt_lens_cpu[:self.
                                          num_contexts] >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices,
                                                             non_blocking=True)

        # V1 line 246: k_context_lens for valid sequences
        self.k_context_lens[:valid_batch_size] = self.prompt_lens_cpu[
            valid_seq_indices] - self.window_size
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True)

        sparse_counts_ctx = torch.zeros(self.num_contexts,
                                        dtype=torch.int32,
                                        device='cpu')
        sparse_counts_ctx[valid_seq_indices] = self.prompt_budget
        sparse_counts_ctx[invalid_seq_indices] = self.prompt_lens_cpu[
            invalid_seq_indices]

        self.sparse_offsets_ctx[1:self.num_contexts + 1] = torch.cumsum(
            sparse_counts_ctx, dim=0)
        self.sparse_offsets_ctx_cuda[:self.num_contexts + 1].copy_(
            self.sparse_offsets_ctx[:self.num_contexts + 1], non_blocking=True)

        # V1 line 264: q_cu_seqlens
        self.q_cu_seqlens[:valid_batch_size + 1] = torch.arange(
            valid_batch_size + 1, device='cpu',
            dtype=torch.int32) * self.window_size
        self.q_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.q_cu_seqlens[:valid_batch_size + 1], non_blocking=True)

        self.k_cu_seqlens[1:valid_batch_size + 1] = torch.cumsum(
            self.k_context_lens[:valid_batch_size], dim=0)
        self.k_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.k_cu_seqlens[:valid_batch_size + 1], non_blocking=True)

        if valid_batch_size > 0:
            self.max_rocket_k_ctx_len = self.k_context_lens[:
                                                            valid_batch_size].max(
                                                            ).item()
            self.total_rocket_k_ctx_tokens = self.k_cu_seqlens[
                valid_batch_size].item()
        else:
            self.max_rocket_k_ctx_len = 0
            self.total_rocket_k_ctx_tokens = 0

        self.valid_batch_size = valid_batch_size
        self.total_sparse_ctx_indices = self.sparse_offsets_ctx[
            self.num_contexts].item()

        # ---- Generation phase setup (V1 line 290)
        self.num_kt_tokens[:self.num_generations] = (
            self.kv_lens[self.num_contexts:self.num_seqs] + self.page_size -
            1) // self.page_size

        self.cum_kt_lens[1:self.num_generations + 1] = torch.cumsum(
            self.num_kt_tokens[:self.num_generations], dim=0)
        self.cum_kt_lens_cuda[:self.num_generations + 1].copy_(
            self.cum_kt_lens[:self.num_generations + 1], non_blocking=True)

        self.total_kt_tokens = self.num_generations * self.max_kt_tokens

        topk_tensor = torch.tensor(self.topk, dtype=torch.int32)

        sparse_counts_gen = torch.minimum(
            topk_tensor, self.num_kt_tokens[:self.num_generations])

        self.sparse_offsets_gen[1:self.num_generations + 1] = torch.cumsum(
            sparse_counts_gen[:self.num_generations], dim=0)
        self.sparse_offsets_gen_cuda[:self.num_generations + 1].copy_(
            self.sparse_offsets_gen[:self.num_generations + 1],
            non_blocking=True)

        self.total_sparse_gen_indices = self.topk * self.num_generations
        # V1 also exposes total_sparse_gen_indices on metadata via __post_init__
        # subclass extension; we set it here directly.


class RocketKVVanillaAttentionMetadata(VanillaAttentionMetadata):
    """Port of V1 ``RocketVanillaAttentionMetadata`` (rocket.py:580).
    Algorithm body for vanilla decode path is `_rocketkv_selection` in
    V1 — porting to V17 is Phase 7 work; this class is the structural
    placeholder so backend dispatch resolves correctly.
    """

    @property
    def _rocket_executor(self) -> Optional["RocketKV"]:
        if getattr(self, "coordinator", None) is None:
            return None
        return self.coordinator.get_executor("sparse")  # type: ignore[return-value]

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        e = self._rocket_executor
        max_kt_blocks = e.max_kt_blocks_per_seq if e else 0
        if max_kt_blocks > 0:
            self.kt_cache_block_offsets = torch.empty(
                [self.max_num_sequences, max_kt_blocks],
                dtype=torch.int32,
                device='cuda',
            )
            self.host_kt_cache_block_offsets = torch.zeros_like(
                self.kt_cache_block_offsets,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
        else:
            self.kt_cache_block_offsets = None
            self.host_kt_cache_block_offsets = None

    def prepare(self) -> None:
        """Port of V1 ``RocketVanillaAttentionMetadata.prepare`` (line 601)."""
        super().prepare()
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        num_requests = num_contexts + num_generations

        for i in range(num_requests):
            if i < num_contexts:
                self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
            else:
                if self.prompt_lens[i] > self.prompt_budget:
                    self.kv_cache_params.num_cached_tokens_per_seq[
                        i] += self.prompt_budget - self.prompt_lens[i]

        if self.kv_cache_manager is not None:
            e = self._rocket_executor
            if e is not None and self.host_kt_cache_block_offsets is not None:
                e.copy_kt_block_offsets(self.request_ids,
                                        self.host_kt_cache_block_offsets)
                self.kt_cache_block_offsets[:self.num_seqs].copy_(
                    self.host_kt_cache_block_offsets[:self.num_seqs],
                    non_blocking=True)


# Wire Metadata class refs (set after Metadata classes are defined).
RocketKVTrtllmAttention.Metadata = RocketKVTrtllmAttentionMetadata
RocketKVVanillaAttention.Metadata = RocketKVVanillaAttentionMetadata


# =========================================================================
# L1 cache manager — V2 subclass with rewind_kv_cache (V1 line 1022 parity) #
#                                                                          #
# V2's _KVCache.resize enforces ``history_length`` monotone-grow, blocking #
# RocketKV's Stage I-b physical evict.  Subclassing V2 (Pattern 3) lets    #
# us:                                                                      #
#   1. Mutate V2's private ``_history_length`` to allow shrink             #
#   2. Override ``update_resources`` so the standard gen-branch resize     #
#      doesn't try ``capacity >= max_beam_num_tokens-1`` (max_beam_num is  #
#      C++-read-only, doesn't shrink with our cache trim).                 #
#                                                                          #
# This was the user's original fallback ("能不subclass最好但...") when    #
# Pattern 2 (declarative BufferConfig) hit V2's invariants.                #
# =========================================================================


from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManagerV2 as _KVCacheManagerV2Base)


class RocketKVCacheManagerV2(_KVCacheManagerV2Base):
    """V17 RocketKV cache manager: V2 subclass that adds physical
    Stage I-b rewind support.

    V1 parity: see V1 ``RocketKVCacheManager.rewind_kv_cache`` →
    ``self.impl.rewind_kv_cache(req_id, rewind_len)`` (C++ binding).
    We replicate the semantics in pure Python because V2's _kv_cache.py
    asserts ``history_length`` monotone — but we own the subclass so
    we can mutate the private state.
    """

    def rewind_kv_cache(self, request: "LlmRequest", rewind_len: int) -> None:
        """V1 line 1022 byte-exact API: shrink request's cache by
        ``rewind_len`` tokens from the tail.

        After this call:
        - kv_cache.capacity → aligned-up to remaining tokens
        - kv_cache._history_length → reduced by rewind_len
        - request.py_kt_target_history → set, signal for update_resources
          override to keep history pinned to the truncated value across
          subsequent gen iters (max_beam_num_tokens grows, but cache
          history follows our target + gen-since-rewind).
        """
        import os as _os
        _dbg = _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1"
        if _dbg:
            print(f"[V2sub.rewind_kv_cache] req={request.py_request_id} "
                  f"rewind_len={rewind_len}", flush=True)
        if rewind_len <= 0:
            return
        kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is None or not kv_cache.is_active:
            if _dbg:
                print(f"[V2sub.rewind_kv_cache] SKIP no kv_cache", flush=True)
            return
        new_history = max(0, kv_cache._history_length - rewind_len)
        if _dbg:
            print(f"[V2sub.rewind_kv_cache] FIRE old_hist="
                  f"{kv_cache._history_length} new_hist={new_history} "
                  f"old_cap={kv_cache.capacity}", flush=True)
        tokens_per_block = self.tokens_per_block
        new_capacity = ((new_history + tokens_per_block - 1) //
                        tokens_per_block) * tokens_per_block
        # Mutate the monotone-history guard variable first; V2 resize
        # then sees history_length == self._history_length and the
        # "decrease" check passes.
        kv_cache._history_length = new_history
        kv_cache.resize(new_capacity, new_history)
        # Sticky flag: future update_resources iters use this to
        # compute effective history (target + new gen tokens since
        # rewind) instead of max_beam_num_tokens-1 (pre-rewind).
        request.py_kt_target_history = new_history
        # Snapshot max_beam_num_tokens at rewind moment — used to
        # compute "tokens generated since rewind".
        request.py_kt_max_beam_at_rewind = request.max_beam_num_tokens

    def update_resources(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata=None,
        kv_cache_dtype_byte_size=None,
    ) -> None:
        """Override V2's gen-branch resize for rocketkv-truncated requests.

        Standard V2 path: ``resize(capacity - py_rewind_len,
        max_beam_num_tokens - 1)``.  Fails for rocketkv-truncated requests
        because max_beam_num_tokens still reflects pre-rewind seq_len,
        which is > our shrunken capacity.

        Override: for requests where ``py_kt_target_history`` is set,
        compute history as ``target + (current_max_beam_num - snapshot)``
        and capacity to fit that.  All other requests fall through to
        standard V2 logic via super().
        """
        # Borrow LlmRequestState lazily — same import the base class does
        # at runtime.
        from tensorrt_llm._torch.pyexecutor.llm_request import (  # noqa
            LlmRequestState)
        import os as _os
        _dbg = _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1"
        # Handle truncated gen requests ourselves; defer the rest to base.
        handled = set()
        for req in scheduled_batch.generation_requests:
            if not hasattr(req, "py_kt_target_history"):
                continue
            if _dbg:
                print(f"[V2sub.update_resources] req={req.py_request_id} "
                      f"target_hist={req.py_kt_target_history} "
                      f"max_beam={req.max_beam_num_tokens} "
                      f"snap={getattr(req, 'py_kt_max_beam_at_rewind', None)}",
                      flush=True)
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue
            gen_since_rewind = (req.max_beam_num_tokens -
                                req.py_kt_max_beam_at_rewind)
            effective_history = req.py_kt_target_history + gen_since_rewind
            tpb = self.tokens_per_block
            new_capacity = ((effective_history + tpb - 1) // tpb) * tpb
            # Bypass monotone-history guard.
            kv_cache._history_length = effective_history
            kv_cache.resize(new_capacity, effective_history)
            handled.add(req.py_request_id)
        # Drop handled gen requests from the batch for super's call to
        # avoid double-processing. We mutate scheduled_batch's list
        # in-place (it's owned by py_executor and only read in this call).
        original_gen = list(scheduled_batch.generation_requests)
        try:
            scheduled_batch.generation_requests = [
                r for r in original_gen if r.py_request_id not in handled
            ]
            super().update_resources(scheduled_batch, attn_metadata,
                                     kv_cache_dtype_byte_size)
        finally:
            scheduled_batch.generation_requests = original_gen


# =========================================================================
# L2 executor — algorithm body                                              #
#                                                                          #
# All HOOK bodies port V1 logic verbatim, with kernel calls / arg ordering #
# preserved. The only structural change from V1: KT cache pool is owned by #
# this executor instance (Pattern 1) instead of by a V1 cache-manager      #
# subclass.                                                                #
# =========================================================================


class RocketKV(SparseAttentionExecutor):
    """V17 RocketKV — direct port of V1 RocketTrtllmAttention +
    RocketKVCacheManager algorithm bodies. See module docstring for V1→V17
    mapping table.

    RocketKV is a 2-stage hybrid sparse attention method:
    - Stage I (prefill): SnapKV-style top-pB physical eviction (HOOK 3)
      + per-page KT summary build streaming through every attention layer
      (HOOK 2 side-effect).
    - Stage II (decode): query-aware HSA mask over the shrunk cache,
      using KT summaries as the page-level lookup table (HOOK 4).
    """

    axis: ClassVar[str] = "sparse"

    # RocketKV physically deletes tokens at prefill end (Stage I-b SnapKV
    # top-pB keep + ``compact_request_cache``).
    physically_evicts_kv: ClassVar[bool] = True

    # Stage I keep-set depends on this prompt's last-window attn scores +
    # KT_CACHE is request-specific → cross-request reuse breaks.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Pattern 3 (V2 subclass) standby — RocketKVCacheManagerV2 above adds
    # rewind_kv_cache, but verifying the C++ kernel's selective-KV-write
    # path first (via sparse_attn_indices_block_size always-set fix in
    # trtllm.py forward).  If that alone gives V1-acc parity without
    # rewind, Pattern 3 isn't strictly needed.
    kv_cache_manager_class: ClassVar[Optional[type]] = None

    # Access type for different dtype sizes (V1 rocket.py:322).
    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        page_size: int = 16,
        prompt_budget: int = 2048,
        kt_cache_dtype: str = "bfloat16",
        kt_tokens_per_block: Optional[int] = None,
        # Extra V1 params not yet in V17 config (defaults match V1
        # RocketSparseAttentionConfig); these will be moved into
        # RocketKVSparseAttentionConfig in a follow-up.
        window_size: int = 32,
        kernel_size: int = 5,
        topk: int = 256,
        topr: int = 32,
    ):
        super().__init__(kv_cache_manager)
        self.page_size = page_size
        self.prompt_budget = prompt_budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.topk = topk
        self.topr = topr

        # Pattern 2 (V2 declarative BufferConfig, 2026-05-28): KT cache pool
        # is owned by KVCacheManagerV2 via Role.KT_CACHE BufferConfig (added
        # in resource_manager.py:_build_cache_config when
        # sparse_attn_config.algorithm == "rocketkv"). The executor delegates
        # to V2's get_kt_buffers / blocks_in_primary_pool / kt_tokens_per_block
        # for sizing + access. This replaces the earlier Pattern 1
        # (executor-owned pool) which didn't sync with V2 page lifecycle.
        #
        # When the V2 manager doesn't expose KT_CACHE (e.g., tests with
        # MagicMock cache mgr or non-rocketkv configs), executor methods
        # degrade gracefully (return None) — algorithm body skips KT-related
        # work in that case.
        # MagicMock-truthy gate trap [[feedback_mock_truthy_gate_trap]]:
        # ``hasattr(MagicMock, "_kt_cache_enabled")`` is True (auto-attr) and
        # ``getattr(MagicMock, "_kt_cache_enabled", False)`` returns a Mock
        # which is truthy. Use ``is True`` (boolean identity) to short-circuit
        # MagicMock test fixtures down the no-op path.
        self._kt_supported = (
            hasattr(kv_cache_manager, "_kt_cache_enabled") and getattr(
                kv_cache_manager, "_kt_cache_enabled", False) is True)
        if self._kt_supported:
            # V2 derives kt_tokens_per_block + dtype from sparse_attn_config
            # at construction. We mirror them here for the algorithm body
            # to access without an extra V2 round-trip.
            self.kt_tokens_per_block = kv_cache_manager.kt_tokens_per_block
            self.kt_cache_dtype = kv_cache_manager._kt_torch_dtype
            self.max_kt_blocks_per_seq = kv_cache_manager.max_kt_blocks_per_seq
        else:
            self.kt_tokens_per_block = (kt_tokens_per_block
                                        or next_power_of_2(
                                            math.ceil(page_size / page_size)))
            self.kt_cache_dtype = (torch.bfloat16
                                   if kt_cache_dtype == "bfloat16" else
                                   torch.float8_e5m2)
            self.max_kt_blocks_per_seq = 0

        import os as _os
        if _os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1":
            print(f"[RocketKV.__init__] _kt_supported={self._kt_supported} "
                  f"kt_tokens_per_block={self.kt_tokens_per_block} "
                  f"kt_cache_dtype={self.kt_cache_dtype} "
                  f"max_kt_blocks_per_seq={self.max_kt_blocks_per_seq} "
                  f"kv_cache_manager_type={type(kv_cache_manager).__name__}",
                  flush=True)

    # ===================================================================== #
    # Pool access helpers — delegate to V2's KT_CACHE BufferConfig pool      #
    # (Pattern 2). V2 inner manages allocation + lifecycle; executor just    #
    # exposes the V1-compatible API surface for algorithm body to call.     #
    # ===================================================================== #

    def get_kt_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return KT pool tensor for the given attention layer. Delegates
        to V2 ``get_kt_buffers(layer_idx)`` when V2 has KT_CACHE BufferConfig
        wired; returns ``None`` otherwise (mocked tests / non-rocketkv config).
        V1 RocketKVCacheManager.get_kt_buffers parity."""
        if not self._kt_supported:
            return None
        return self.kv_cache_manager.get_kt_buffers(layer_idx)

    def copy_kt_block_offsets(self, request_ids: List[int],
                              block_offsets: torch.Tensor) -> torch.Tensor:
        """Fill ``block_offsets[i, :]`` with the per-request block IDs for
        request ``request_ids[i]``. Pattern 2: KT shares block IDs with
        KEY/VALUE (same logical block in V2 multi-pool layout), so we
        read block IDs once from V2's ``get_block_ids_per_seq`` and copy
        into the KT-side tensor.

        V1 line 1004 parity: V1's separate ``BlockManager.copy_block_offsets``
        wrote KT block IDs into the destination tensor.
        """
        if not self._kt_supported:
            return block_offsets
        # V2 returns padded (B, max_blocks_for_any_req) tensor of block IDs.
        block_ids = self.kv_cache_manager.get_block_ids_per_seq(request_ids)
        B, blocks_in_req = block_ids.shape
        # block_offsets has shape (max_num_sequences, max_kt_blocks_per_seq);
        # write the same block ID into KT slot (KT shares block IDs with KEY
        # — V2 multi-pool layout puts both into the same physical block).
        target = block_offsets[:B, :blocks_in_req]
        target.copy_(block_ids)
        return block_offsets

    # ===================================================================== #
    # HOOK 1 — on_request_init (V1 RocketKVCacheManager.prepare_resources    #
    # context_requests branch port).                                         #
    # ===================================================================== #

    def on_request_init(self, request: "LlmRequest") -> None:
        """V1 line 1008-1020 parity. V1 RocketKVCacheManager.prepare_resources:
            for ctx req:  self.kt_cache_manager.add_tokens(req_id, ceil(prompt_len/page_size))
            for gen req:  if (max_beam_num_tokens + 1) % page_size == 1: add 1 KT slot

        Pattern 2 (Path A v17): KT lives in the same V2 multi-pool block
        as KEY/VALUE (shared block IDs via BufferConfig). When V2 allocates
        a new physical block (because KV crossed a tokens_per_block
        boundary), the corresponding KT slot inside that block is part
        of the same allocation — no separate KT bookkeeping needed.

        V1 add_tokens grew V1's *separate* BlockManager state; Pattern 2's
        KT block-IDs are read from V2 (``get_block_ids_per_seq``) at
        prepare-time. Init is therefore a no-op.
        """
        return

    # ===================================================================== #
    # HOOK 2 — on_context_attention (V1 sparse_kv_predict port,              #
    # rocket.py:354)                                                          #
    # ===================================================================== #

    def on_context_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Port of V1 ``RocketTrtllmAttention.sparse_kv_predict`` (line 354).

        Computes SnapKV sparse kv indices via:
          1. Split observation window Q from prefix K
          2. BMM(Q_window, K_prefix) → scores
          3. Softmax + reduce per-head → per-token importance scores
          4. Max-pool + topk → selected prefix indices
          5. Combine with window indices, flatten across batch
          6. Update KT cache pool (Stage I-a side effect)

        Returns ``(sparse_kv_indices, sparse_kv_offsets)`` for the kernel
        to consume as input-side sparse mask; ``None`` if no valid context
        sequences this iter.
        """
        import os
        debug = os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1"
        if not isinstance(metadata, RocketKVTrtllmAttentionMetadata):
            if debug:
                print(f"[HOOK2 SKIP] layer={layer_idx} metadata_type="
                      f"{type(metadata).__name__} (expected "
                      f"RocketKVTrtllmAttentionMetadata)", flush=True)
            return None
        if not self._kt_supported:
            if debug:
                print(f"[HOOK2 SKIP] layer={layer_idx} _kt_supported=False",
                      flush=True)
            return None
        num_ctx_tokens = metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            if debug:
                print(f"[HOOK2 SKIP] layer={layer_idx} num_ctx_tokens=0",
                      flush=True)
            return None
        # Cache num_heads_per_kv on first call so non-metadata-scope
        # helpers (algorithm-body internals) can recover num_heads.
        self._cached_num_heads_per_kv = int(
            getattr(metadata, "num_heads_per_kv", 1) or 1)
        if debug:
            print(f"[HOOK2 FIRE] layer={layer_idx} "
                  f"num_ctx_tokens={num_ctx_tokens} "
                  f"valid_batch_size={metadata.valid_batch_size} "
                  f"num_heads_per_kv={self._cached_num_heads_per_kv}",
                  flush=True)

        # V1 line 368: prepare qkv input
        if k is None:
            qkv_input = q[:num_ctx_tokens]
        else:
            qkv_input = torch.cat([q, k], dim=1)

        if metadata.valid_batch_size > 0:
            # V1 line 378: split observation window Q from prefix K
            q_window, k_context = triton_rocket_qk_split(
                qkv_input,
                metadata.prompt_lens_cuda,
                metadata.context_cumsum_cuda,
                metadata.valid_seq_indices_cuda,
                metadata.k_cu_seqlens_cuda,
                metadata.total_rocket_k_ctx_tokens,
                # num_heads / num_kv_heads / head_dim from base TrtllmAttention;
                # the attention class instance isn't directly available here, so
                # derive from kv_cache_manager.
                self._num_heads_from_kv_cache_manager(),
                self._num_kv_heads_from_kv_cache_manager(),
                self._head_dim_from_kv_cache_manager(),
                self.window_size,
                metadata.valid_batch_size,
            )

            # V1 line 392: BMM scores
            scores = triton_bmm(q_window,
                                k_context,
                                metadata.q_cu_seqlens_cuda,
                                metadata.k_cu_seqlens_cuda,
                                metadata.valid_batch_size,
                                causal=False)

            # V1 line 399: softmax
            scores = triton_softmax(scores, metadata.k_cu_seqlens_cuda,
                                    metadata.valid_batch_size)

            # V1 line 402: reduce over (heads_per_kv, window)
            num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
            num_heads = self._num_heads_from_kv_cache_manager()
            scores = scores.view(num_kv_heads, num_heads // num_kv_heads,
                                 self.window_size, -1).sum(dim=(1, 2))

            # V1 line 408: flatten variable-length batch
            scores = triton_flatten_to_batch(scores, metadata.k_cu_seqlens_cuda,
                                             metadata.valid_batch_size,
                                             metadata.max_rocket_k_ctx_len)

            # V1 line 413: max-pool smoothing
            scores = torch.nn.functional.max_pool1d(
                scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)

            # V1 line 419: indexer topk prefill
            total_tasks = metadata.valid_batch_size * num_kv_heads
            selected_prefix_indices = torch.empty(
                (total_tasks, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32)
            scores = scores.view(total_tasks, -1)

            row_starts = metadata.k_context_start_cuda[:metadata.
                                                       valid_batch_size].repeat_interleave(
                                                           num_kv_heads)
            row_ends = metadata.k_context_lens_cuda[:metadata.
                                                    valid_batch_size].repeat_interleave(
                                                        num_kv_heads)
            torch.ops.trtllm.indexer_topk_prefill(
                scores, row_starts, row_ends, selected_prefix_indices,
                self.prompt_budget - self.window_size)

            # V1 line 440: sort selected indices
            selected_prefix_indices = torch.sort(selected_prefix_indices,
                                                 dim=-1).values
        else:
            selected_prefix_indices = torch.empty(
                (0, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32)

        # V1 line 448: build sparse_kv_offsets + sparse_kv_indices
        sparse_kv_offsets = metadata.sparse_offsets_ctx_cuda[:metadata.
                                                             num_contexts + 1]
        sparse_kv_indices = triton_rocket_batch_to_flatten(
            selected_prefix_indices, metadata.prompt_lens_cuda,
            metadata.valid_seq_indices_cuda, sparse_kv_offsets,
            metadata.num_contexts, metadata.total_sparse_ctx_indices,
            self.window_size, self.prompt_budget,
            self._num_kv_heads_from_kv_cache_manager())

        # ISOLATION: return None to skip sparse mask + KT side effect.
        if os.environ.get("ROCKETKV_HOOK2_RETURN_NONE") == "1":
            return None
        # V1 line 458: Stage I-a side-effect — update KT cache pool
        kt_cache_tensor = self.get_kt_buffers(layer_idx)
        if kt_cache_tensor is not None:
            triton_rocket_update_kt_cache_ctx(
                qkv_input.contiguous(),
                kt_cache_tensor,
                metadata.kt_cache_block_offsets[:metadata.num_contexts],
                metadata.context_cumsum_cuda[:metadata.num_contexts + 1],
                sparse_kv_indices,
                sparse_kv_offsets,
                self._num_heads_from_kv_cache_manager(),
                self._num_kv_heads_from_kv_cache_manager(),
                self._head_dim_from_kv_cache_manager(),
                self.page_size,
                self.prompt_budget,
                metadata.kt_tokens_per_block,
                self.max_kt_blocks_per_seq,
            )

        # ---- DUMP for V1/V17 diff (2026-05-28)
        try:
            from .dump_util import rocketkv_dump
            kt_layer_view = kt_cache_tensor[0, 0, 0, :] if kt_cache_tensor is not None else None
            rocketkv_dump(
                "hook2_post",
                layer=layer_idx,
                sparse_kv_indices=sparse_kv_indices,
                sparse_kv_offsets=sparse_kv_offsets,
                selected_prefix_indices=selected_prefix_indices
                    if metadata.valid_batch_size > 0 else None,
                kt_block0_slot0=kt_layer_view,
                num_ctx_tokens=qkv_input.shape[0] if qkv_input is not None else 0,
            )
        except Exception:
            pass

        # V1 line 478: reduce post-processing
        if metadata.valid_batch_size == 0:
            return None
        return sparse_kv_indices, sparse_kv_offsets

    # ===================================================================== #
    # HOOK 3 — on_context_end (V1 RocketKVCacheManager.update_resources      #
    # rewind logic port, rocket.py:1022)                                     #
    # ===================================================================== #

    def on_context_end(self, request: "LlmRequest",
                       metadata: "AttentionMetadata") -> None:
        """Stage I-b SnapKV physical eviction.  V1 line 1022 byte-exact:
            seq_len = request.get_num_tokens(0)
            rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
            self.rewind_kv_cache(request, rewind_len)
        Fires once per request at prefill→decode state transition.

        Pattern 3 (V2 subclass): ``kv_cache_manager`` is
        :class:`RocketKVCacheManagerV2` which adds back the V1-style
        rewind_kv_cache by mutating V2's private _history_length +
        overriding update_resources so subsequent gen iters honor the
        truncated history (instead of max_beam_num_tokens-1 which is
        C++-read-only and doesn't shrink).
        """
        # V1 filter: skip terminated mid-prefill
        try:
            from tensorrt_llm._torch.pyexecutor.llm_request import (
                LlmRequestState)
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                return
        except Exception:
            pass

        seq_len = request.get_num_tokens(0) if hasattr(
            request, "get_num_tokens") else 0
        rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
        if rewind_len <= 0:
            return

        # NOTE: V2 base doesn't expose rewind_kv_cache.  Pattern 3
        # subclass is standby; for now rely on selective KV write in
        # the attention kernel (driven by sparse_kv_indices), which
        # compacts selected K/V to the front of the cache during the
        # prefill forward pass.  No physical evict needed when the
        # kernel-side selective write is firing.
        rewind_fn = getattr(self.kv_cache_manager, "rewind_kv_cache", None)
        if rewind_fn is not None:
            rewind_fn(request, rewind_len)

    # ===================================================================== #
    # HOOK 4 — on_generation_attention (V1 sparse_attn_predict port,         #
    # rocket.py:512)                                                          #
    # ===================================================================== #

    @torch.compile(dynamic=True, disable=True)  # disable during framework dev
    def _preprocess_for_gen(self, q, k, metadata):
        """V1 line 485 port."""
        num_heads = self._num_heads_from_kv_cache_manager()
        num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
        head_dim = self._head_dim_from_kv_cache_manager()
        if k is None:
            qkv_input = q[metadata.num_ctx_tokens:]
            q_hidden_size = num_heads * head_dim
            k_hidden_size = num_kv_heads * head_dim
            q = qkv_input[:, :q_hidden_size]
            k = qkv_input[:, q_hidden_size:q_hidden_size + k_hidden_size]
        else:
            q = q[metadata.num_ctx_tokens:]
            k = k[metadata.num_ctx_tokens:]
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim)
        return q, k

    @torch.compile(dynamic=True, disable=True)
    def _topr_filter(self, q):
        """V1 line 505 port."""
        head_dim = self._head_dim_from_kv_cache_manager()
        i1 = torch.topk(q.abs().sum(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        q_mask = torch.zeros_like(q)
        q_mask.scatter_(-1, i1.expand_as(q[..., :self.topr]), 1)
        return q * q_mask

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Port of V1 ``RocketTrtllmAttention.sparse_attn_predict`` (line 512).

        Stage II HSA: per-decode-step, per-attention-layer, build sparse
        attention mask over the (already-shrunk by Stage I-b) cache using
        KT page summaries as a coarse-grained lookup.
        """
        import os
        debug = os.environ.get("ROCKETKV_DEBUG_HOOKS") == "1"
        if not isinstance(metadata, RocketKVTrtllmAttentionMetadata):
            if debug:
                print(f"[HOOK4 SKIP] layer={layer_idx} metadata_type="
                      f"{type(metadata).__name__}", flush=True)
            return None
        if not self._kt_supported:
            if debug:
                print(f"[HOOK4 SKIP] layer={layer_idx} _kt_supported=False",
                      flush=True)
            return None
        if metadata.num_generations == 0:
            if debug:
                print(f"[HOOK4 SKIP] layer={layer_idx} num_generations=0",
                      flush=True)
            return None
        self._cached_num_heads_per_kv = int(
            getattr(metadata, "num_heads_per_kv", 1) or 1)
        if debug:
            print(f"[HOOK4 FIRE] layer={layer_idx} "
                  f"num_generations={metadata.num_generations}", flush=True)

        # ISOLATION: return None to skip Stage II HSA + KT_gen update.
        if os.environ.get("ROCKETKV_HOOK4_RETURN_NONE") == "1":
            return None

        q, k = self._preprocess_for_gen(q, k, metadata)

        head_dim = self._head_dim_from_kv_cache_manager()
        if self.topr < head_dim:
            q = self._topr_filter(q)

        kt_cache_tensor = self.get_kt_buffers(layer_idx)
        if kt_cache_tensor is None:
            return None

        num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
        num_heads = self._num_heads_from_kv_cache_manager()

        # V1 line 531: update KT cache for new gen step
        triton_rocket_update_kt_cache_gen(
            k,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts:],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts:],
            metadata.page_size,
            metadata.kt_tokens_per_block,
            self.max_kt_blocks_per_seq,
            num_kv_heads,
            head_dim,
        )

        # V1 line 544: BMM Q · KT
        scores = triton_rocket_paged_kt_cache_bmm(
            q,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts:],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts:],
            metadata.cum_kt_lens_cuda,
            metadata.page_size,
            metadata.kt_tokens_per_block,
            self.max_kt_blocks_per_seq,
            metadata.total_kt_tokens,
        )

        scores = triton_softmax(scores, metadata.cum_kt_lens_cuda,
                                metadata.num_generations)

        scores = triton_rocket_reduce_scores(
            scores,
            metadata.cum_kt_lens_cuda,
            metadata.num_generations,
            num_kv_heads,
            num_heads // num_kv_heads,
        )

        sparse_attn_offsets = metadata.sparse_offsets_gen_cuda[:metadata.
                                                               num_generations +
                                                               1]
        selected_indices = triton_topk(scores, metadata.cum_kt_lens_cuda,
                                       sparse_attn_offsets,
                                       metadata.total_sparse_gen_indices,
                                       metadata.topk)

        # ---- DUMP for V1/V17 diff (2026-05-28)
        try:
            from .dump_util import rocketkv_dump
            rocketkv_dump(
                "hook4_post",
                layer=layer_idx,
                sparse_attn_indices=selected_indices,
                sparse_attn_offsets=sparse_attn_offsets,
                scores=scores,
                kt_block0_slot0_gen=kt_cache_tensor[0, 0, 0, :]
                    if kt_cache_tensor is not None else None,
                q_first=q.flatten()[:64] if q is not None else None,
            )
        except Exception:
            pass

        return selected_indices, sparse_attn_offsets

    # ===================================================================== #
    # HOOK 6 — on_request_finish (V1 free_resources port, line 1038)         #
    # ===================================================================== #

    def on_request_finish(self, request: "LlmRequest") -> None:
        """V1 line 1038-1040 parity. V1 RocketKVCacheManager.free_resources:
            super().free_resources(request)
            self.kt_cache_manager.free_resources(request)

        Pattern 2 (Path A v17): KT shares block IDs with KEY/VALUE in V2's
        multi-pool BufferConfig, so when V2's ``free_resources``
        releases the request's blocks the KT slots inside those blocks
        are released alongside. V1's separate ``kt_cache_manager``
        existed because V1 had an *independent* BlockManager for KT;
        Pattern 2 unifies KT bookkeeping into V2's per-block tracking,
        so this hook is a no-op.
        """
        return

    # ===================================================================== #
    # Helpers — proxy attention layer params via kv_cache_manager.            #
    # ===================================================================== #

    def _num_kv_heads_from_kv_cache_manager(self) -> int:
        # V1 KVCacheManager: scalar self.num_kv_heads.
        # V2 KVCacheManagerV2: list self.num_kv_heads_per_layer (uniform
        # across layers for Llama-class models). Fall back to per-layer
        # list when scalar is absent.
        v1 = getattr(self.kv_cache_manager, "num_kv_heads", None)
        if v1:
            return int(v1)
        per_layer = getattr(self.kv_cache_manager,
                            "num_kv_heads_per_layer", None)
        if per_layer:
            return int(per_layer[0])
        return 0

    def _head_dim_from_kv_cache_manager(self) -> int:
        v1 = getattr(self.kv_cache_manager, "head_dim", None)
        if v1:
            return int(v1)
        per_layer = getattr(self.kv_cache_manager, "head_dim_per_layer",
                            None)
        if per_layer:
            return int(per_layer[0])
        return 0

    def _num_heads_from_metadata(self, metadata) -> int:
        """V1 stored num_heads on the attention class instance; V2's cache
        manager doesn't carry it. We read it via metadata.num_heads_per_kv
        (set by model_engine at attn_metadata construction): num_heads =
        num_kv_heads * num_heads_per_kv."""
        nhpkv = getattr(metadata, "num_heads_per_kv", 1) or 1
        return self._num_kv_heads_from_kv_cache_manager() * int(nhpkv)

    # Backward-compat alias for sites that don't have metadata in scope.
    def _num_heads_from_kv_cache_manager(self) -> int:
        # V1 scalar shortcut; V2 fallback uses cached num_heads_per_kv
        # set at first HOOK 2/4 call.
        v1 = getattr(self.kv_cache_manager, "num_heads", None)
        if v1:
            return int(v1)
        return self._num_kv_heads_from_kv_cache_manager() * int(
            getattr(self, "_cached_num_heads_per_kv", 1))
