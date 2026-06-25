"""TriAttention KV-cache compression: periodic physical KV eviction.

Every ``beta`` generation steps TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the eviction runs in one pre-forward ``on_generation_step_begin`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager` and nothing more -- it
has no attention backend of its own; decode runs the model's standard dense
kernel over the compacted cache. The model engine derives each request's cached
length from its logical length (``max_beam - 1``), unaware of the eviction, so
the compacted-length reconcile runs in ``adjust_attention_metadata`` -- the
compression-framework hook the executor calls just before
``attn_metadata.prepare()``. It uses the standard ``KVCacheManagerV2``: each
eviction sets a generic per-request marker (``py_kv_evicted_tokens``) that the
manager's ``update_resources`` reads to shrink history and return the
eviction-freed blocks to the pool for the capacity gain.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache (a self-consistent NHD round-trip passes an integrity probe, but the
kernel reads garbage). See ``_read_request_k`` / ``_evict_layer_perhead``.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The decode query rotates at its true absolute position
(``model_engine`` keeps the query position at ``max_beam - 1`` while decoupling
``num_cached = max_beam - 1 - evicted``), so a query at its true position
against a kept key at its original rotation still yields the correct relative
distance.

Calibration is NOT computed here: the user calibrates with the official tool
(github.com/WeianMao/triattention) and passes that .pt via ``calibration_path``;
the manager converts it to our runtime schema at load (see _resolve_calibration).
The scoring math follows the same upstream reference (``methods/pruning_utils.py``).
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})


def _build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Upstream pruning_utils.build_geometric_offsets: [1, 2, 4, ... <=max]."""
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_end``: every ``beta`` generation steps it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    top-B keep set. Each layer scores its own keys and keeps the same count
    (top-B) with its own kept set, so the per-request num_cached stays
    consistent across layers.
    """

    # The TRT-LLM IndexerTopK op sizes the single-block kernel's dynamic shared
    # memory as k * 4 bytes with no opt-in past the 48 KB per-block cap, so its
    # launch fails ("invalid argument") once k * 4 plus the kernel's static
    # cub-sort shared memory exceeds 48 KB. k = 4096 (16 KB) is verified-safe;
    # k = 8192 (32 KB) crashes. Above this the select falls back to torch.topk.
    _INDEXER_TOPK_MAX_K = 4096

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        top_B: int,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        window_size: int = 128,
        eviction_mode: str = "union",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
    ):
        super().__init__(kv_cache_manager)
        self.top_B = top_B
        self.beta = beta
        # Which token set each eviction round keeps (all reproduce the upstream
        # selection: z-normalize scores, pin the prompt tokens, no recency window):
        #   union              -- union of every KV head's top-B, re-ranked by the
        #                         per-token max score. Default; matches the official
        #                         base setting (per-head and per-layer-per-head
        #                         pruning both off).
        #   per_head           -- each KV head keeps its own set, shared across
        #                         layers (mean of per-layer max).
        #   per_layer_perhead  -- each (layer, KV head) keeps its own set, fully
        #                         independent per layer.
        self.eviction_mode = eviction_mode
        self.normalize_scores = bool(normalize_scores)
        self.pin_prefill = bool(pin_prefill)
        # Recency-window knob, kept for API compatibility. The upstream
        # calibration-based selection does not apply a recency window on the AIME
        # protocol and none of the implemented modes read this value, so it is
        # currently inert; the prompt is preserved via pin_prefill instead.
        self.window_size = int(window_size)
        self.score_aggregation = score_aggregation
        # Production runs the fused-Triton eviction path unconditionally:
        # all requests evicting this step are processed in ONE fused pass
        # (score x req x layer, select, per-layer compact), decoupling
        # launch count from request count for large-batch serving.

        # Calibration is the OFFICIAL TriAttention .pt (passed via
        # calibration_path), resolved + converted on the first request
        # (on_request_init). TRT-LLM does NOT compute calibration; model_path is
        # used only to derive the model's RoPE tables (omega / freq_scale_sq).
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        # Calibration-derived dims + stats, filled in on_request_init.
        self._L: Optional[int] = None
        self._H: Optional[int] = None
        self._F: Optional[int] = None
        self._freq_scale_sq: Optional[torch.Tensor] = None
        self._attention_scale = 1.0

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Per-request generation-step counter; eviction fires when it hits
        # ``beta``. Cleared on request finish.
        self._gen_steps: Dict[int, int] = {}
        # Cumulative physically-evicted token count per request, consumed by
        # the per-step history reconcile and the metadata shim.
        self._evicted: Dict[int, int] = {}

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Resolve calibration on the first request, then no-op.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._L = int(self.calibration["E_q"].shape[0])
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        self._attention_scale = float(self.calibration.get("attention_scale", 1.0))
        # Pre-split query stats + MLR coefficient for the Triton score kernel so
        # it doesn't recompute (E_q_norm - |E_q|) per call. Shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._triattn_q_real = _Eq.real.to(torch.float32).contiguous()
        self._triattn_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._triattn_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    # The framework drives all lifecycle hooks; TriAttention overrides three:
    # on_request_init (resolve calibration once), on_generation_step_begin
    # (periodic eviction), and on_request_finish (per-request cleanup). It scores
    # from offline calibration, not from live queries or attention scores, so it
    # needs no per-layer attention hook: the whole eviction runs once per period
    # in on_generation_step_begin, which loops the layers and reads each layer's
    # keys straight from the KV pool.

    def on_generation_step_begin(
        self, scheduled_batch: "ScheduledRequests", attn_metadata=None, **kwargs
    ) -> None:
        """Periodic physical eviction, PRE-forward (framework hook, fired from the
        base prepare_resources before _forward_step). Every beta steps over budget:
        score the full cache, select top-B, physically compact, and record
        per-request ``py_kv_evicted_tokens`` -- which the forward's attention
        metadata and the V2 manager's block reclaim both read.

        Uses the pre-forward hook (NOT post-forward on_generation_step_end) because
        the latter races the overlap scheduler: the next iteration's forward is
        enqueued before the post-forward hook mutates the KV, so it reads a racy /
        stale-length cache (det bs=32 overlap-ON: 32/32 divergent; a GPU sync did
        NOT fix it -- the forward metadata was already computed). Mirrors RocketKV.
        """
        self._periodic_evict(scheduled_batch)

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Bump a per-request step counter; every ``beta`` steps score the cache
        and physically evict down to top-B (per-layer, layer-uniform count)."""
        if not self._calibrated:
            return
        gen_requests = getattr(scheduled_batch, "generation_requests", None)
        if not gen_requests:
            return
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        num_layers = self._num_layers_from_manager()

        # (1) bump per-request step counters; collect who evicts THIS step.
        evict_now = []
        for request in gen_requests:
            rid = request.py_request_id
            step = self._gen_steps.get(rid, 0) + 1
            self._gen_steps[rid] = step
            if step % self.beta == 0:
                evict_now.append((request, rid))

        # (2) evict. The path processes all evicting requests in ONE pass
        # of eviction kernels (score x req x layer, select, per-layer compact).
        if evict_now:
            self._evict_requests(evict_now, num_layers)
        # Block-reclaim ordering: record a CUDA event AFTER the compaction kernels
        # (same stream they ran on). KVCacheManagerV2.update_resources waits this
        # on the manager stream before the capacity-shrink, so the page-reuse-gating
        # finish_event recorded inside resize() dominates the compaction -> no
        # read-after-free when a freed page is reallocated.
        if evict_now:
            _cev = torch.cuda.Event()
            _cev.record()
            for request, rid in evict_now:
                request.py_kv_compaction_event = _cev

        # History + capacity reconcile for evicting requests is owned by the generic
        # KVCacheManagerV2.update_resources reclaim path: every step it sets each
        # evicting request's history to the compacted length (max_beam - evicted)
        # and frees the trailing blocks via _KVCache.fork(). It reads the generic
        # ``py_kv_evicted_tokens`` marker, which _evict_requests sets and which
        # persists on the request between eviction steps. The kernel's view is
        # reconciled separately in adjust_attention_metadata.

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request step + evicted counters."""
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)

    # ------------------------------------------------------------------ #
    # Attention-metadata reconcile (compression-framework hook)          #
    # ------------------------------------------------------------------ #

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id``."""
        return self._evicted.get(request_id, 0)

    def adjust_attention_metadata(self, attn_metadata) -> None:
        """Reconcile the attention metadata for this iteration's eviction.

        The framework calls this (BaseKVCacheCompressionManager hook) right before
        ``attn_metadata.prepare()``. The model engine derives
        ``num_cached_tokens_per_seq`` from each request's logical length
        (``max_beam_num_tokens - 1``), unaware of eviction; once TriAttention has
        physically compacted the cache, the attention kernel must instead read the
        compacted length. Both fixes run BEFORE prepare(), so prepare() builds
        ``kv_lens`` and ``prompt_lens_cpu`` from the corrected values:
          1. num_cached -> (max_beam-1) + 1 - evicted = max_beam - evicted, matching
             the cache manager's compacted history.
          2. clamp each evicted request's prompt_len down to num_cached -- a
             prompt_len longer than the whole compacted cache desyncs the
             prompt/gen split and garbles the output.
        Requests with 0 evicted are untouched, so dense / no-evict steps stay
        byte-identical (and this is a no-op for any model that never evicts).
        """
        kvp = getattr(attn_metadata, "kv_cache_params", None)
        if kvp is None or getattr(kvp, "num_cached_tokens_per_seq", None) is None:
            return
        num_contexts = attn_metadata.num_contexts
        num_requests = num_contexts + attn_metadata.num_generations
        req_ids = attn_metadata.request_ids
        prompt_lens = getattr(attn_metadata, "prompt_lens", None)
        pl = list(prompt_lens) if prompt_lens is not None else None
        pl_changed = False
        for i in range(num_contexts, num_requests):
            ev = self.evicted_count(req_ids[i])
            if not ev:
                continue
            nc = int(kvp.num_cached_tokens_per_seq[i]) + 1 - ev
            kvp.num_cached_tokens_per_seq[i] = nc
            if pl is not None and int(pl[i]) > nc:
                pl[i] = nc
                pl_changed = True
        if pl_changed:
            attn_metadata.prompt_lens = pl

    # ================================================================== #
    # Helpers (eviction / scoring / V2 cache access / calibration)       #
    # ================================================================== #

    # --- Upstream-faithful eviction modes (per_head / per_layer_perhead / union) ---
    #
    # These reproduce github.com/WeianMao/triattention's selection: scores are NOT
    # averaged over heads (each KV head keeps its own token set), they are
    # z-normalized per head over the decode region, the prompt (prefill) tokens are
    # pinned, and there is no recency window. The kept COUNT stays uniform (= top_B)
    # so paged attention
    # and the num_cached bookkeeping are unchanged; only the kept SET differs per
    # head. Kept K keeps its original RoPE rotation (scored post-RoPE), so a head
    # holding a different token set still scores the correct relative distance
    # and no per-head position tracking is needed.

    def _zscore_decode(self, head_scores: "torch.Tensor") -> "torch.Tensor":
        """Z-normalize each head's scores over the token axis: ``(x - mean) /
        std`` with ``std`` clamped to 1e-6 (upstream). Row-wise on ``[H, seq]``;
        a no-op when ``normalize_scores`` is off."""
        if not self.normalize_scores or head_scores.numel() == 0:
            return head_scores
        mean = head_scores.mean(dim=1, keepdim=True)
        std = head_scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (head_scores - mean) / std

    def _group_heads_to_kv_max(
        self, head_scores: "torch.Tensor", num_kv_heads: int
    ) -> "torch.Tensor":
        """Reduce per-query-head scores ``[H, seq]`` to per-KV-head ``[nkv, seq]``
        by MAX over the query heads sharing each KV head (upstream within-group
        aggregation). Query heads group contiguously: head ``q`` -> KV head
        ``q // (H // nkv)`` (matches the ``q*nkv//H`` GQA map in scoring and
        upstream's ``head // num_key_value_groups``)."""
        num_q_heads, seq = head_scores.shape
        group = max(1, num_q_heads // num_kv_heads)
        return head_scores.view(num_kv_heads, group, seq).max(dim=1).values

    def _decode_topk(
        self, scores: "torch.Tensor", decode_start: int, decode_budget: int
    ) -> "torch.Tensor":
        """Pick each row's top-``decode_budget`` decode tokens by ``scores``
        (``[rows, decode_count]``); return ABSOLUTE slot indices prefixed with the
        pinned prompt slots, sorted ascending per row (``[rows, decode_start+k]``).
        One IndexerTopK launch over all rows -- the SAME select primitive as the
        union mode, so large-BS/seq decouples the cost from the row count. The
        kept SET is what matters (the result is sorted by slot)."""
        rows, decode_count = scores.shape
        k = min(decode_budget, decode_count)
        decode_idx = self._indexer_topk_idx(scores, k) + decode_start  # [rows, k]
        prefill_idx = torch.arange(decode_start, device=scores.device, dtype=torch.long).expand(
            rows, decode_start
        )
        return torch.sort(torch.cat([prefill_idx, decode_idx], dim=1), dim=1).values

    def _evict_modes(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        precomputed: List["torch.Tensor"],
    ) -> Optional[Union[int, "torch.Tensor"]]:
        """Select + physically compact per ``self.eviction_mode`` from the
        precomputed per-layer scores. Returns the uniform kept count (= ``top_B``,
        or ``seq_len`` if smaller), or None if nothing could be scored.

        ``precomputed``: per-layer ``[H, seq_len]`` scores (from the fused
        score kernel ``triton_tri_score_perhead``); used verbatim for the
        selection + compaction.

        ``compact``: when False AND ``self.eviction_mode == "union"``, the union
        keep set is computed but NOT physically compacted -- the 1-D sorted keep
        tensor (slot indices in ``[0, seq_len)``) is returned instead of the kept
        count, so a caller can batch the per-layer compaction itself. For
        per_head / per_layer_perhead this flag is ignored (they still compact and
        return the kept count); only the layer-uniform union keep can be hoisted
        out this way. The selection math is identical to the compacting path, so
        the kept-slot set is byte-identical regardless of ``compact``."""
        budget = min(self.top_B, seq_len)
        # Prompt pin: only decode tokens [decode_start, seq_len) compete; the
        # first decode_start (prompt) tokens are always kept.
        decode_start = 0
        if self.pin_prefill:
            decode_start = min(int(getattr(request, "py_prompt_len", 0) or 0), seq_len)
        decode_count = seq_len - decode_start
        decode_budget = budget - decode_start
        # Budget exhausted by the pinned prompt (or no decode tokens): keep the
        # first `budget` contiguous slots everywhere; nothing per-head to do.
        if decode_budget <= 0 or decode_count <= 0:
            return budget

        # Select from the precomputed per-layer K scores. For per_head /
        # per_layer_perhead reduce to per-KV-head [nkv, decode_count] (group-max),
        # z-normalized over the decode region. For union keep the raw per-head
        # rows to stack.
        # num KV heads from the pool (HND dim 2); needed to group query-head scores.
        get_buffers = getattr(self.kv_cache_manager, "get_buffers", None)
        p0 = get_buffers(0, kv_layout="HND") if get_buffers is not None else None
        if p0 is None:
            return None
        num_kv_heads = int(p0.shape[2])
        per_layer_kv_scores: List[torch.Tensor] = []
        union_rows: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            head_scores = precomputed[layer_idx]
            if head_scores is None:
                return None
            decode_scores = self._zscore_decode(head_scores[:, decode_start:seq_len])
            if self.eviction_mode == "union":
                union_rows.append(decode_scores)
            else:
                per_layer_kv_scores.append(self._group_heads_to_kv_max(decode_scores, num_kv_heads))

        if self.eviction_mode == "union":
            return self._evict_union(
                request,
                num_layers,
                seq_len,
                decode_start,
                decode_budget,
                torch.cat(union_rows, dim=0),
            )
        if self.eviction_mode == "per_head":
            return self._evict_per_head(
                request, num_layers, seq_len, decode_start, decode_budget, per_layer_kv_scores
            )
        if self.eviction_mode == "per_layer_perhead":
            return self._evict_per_layer_perhead(
                request, seq_len, decode_start, decode_budget, per_layer_kv_scores
            )
        raise ValueError(
            f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
            "'union', 'per_head', 'per_layer_perhead'"
        )

    def _evict_per_head(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        per_layer_kv_scores: List["torch.Tensor"],
    ) -> int:
        """per_head: each KV head's score = MEAN over layers of the per-layer MAX
        (upstream ``_select_per_head_independent``). One keep set per KV head,
        applied to EVERY layer."""
        agg = torch.stack(per_layer_kv_scores, dim=0).mean(dim=0)  # [nkv, dc]
        keep_2d = self._decode_topk(agg, decode_start, decode_budget)  # [nkv, keep_count]
        for layer_idx in range(num_layers):
            self._evict_layer_perhead(request, layer_idx, keep_2d, seq_len)
        return int(keep_2d.shape[1])

    def _evict_per_layer_perhead(
        self,
        request: "LlmRequest",
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        per_layer_kv_scores: List["torch.Tensor"],
    ) -> Optional[int]:
        """per_layer_perhead: each (layer, KV head) selects independently from
        that layer's per-KV-head max (upstream
        ``_select_per_layer_perhead_independent``)."""
        keep_count = None
        for layer_idx, kv_scores in enumerate(per_layer_kv_scores):
            keep_2d = self._decode_topk(kv_scores, decode_start, decode_budget)
            self._evict_layer_perhead(request, layer_idx, keep_2d, seq_len)
            keep_count = int(keep_2d.shape[1])
        return keep_count

    def _evict_union(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        head_matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        """union: union of every head's top-k, re-ranked by the per-token max
        (upstream ``_select_union_based``). One 1-D keep set shared by every
        layer; returns the sorted kept slot indices in ``[0, seq_len)`` so the
        caller compacts all layers in one pass (``triton_tri_compact``)."""
        combined = head_matrix.max(dim=0).values  # [decode_count]
        keep_1d = self._select_union(head_matrix, combined, decode_budget)
        prefill_idx = torch.arange(decode_start, device=combined.device, dtype=torch.long)
        keep = torch.sort(torch.cat([prefill_idx, keep_1d + decode_start])).values
        return keep

    def _indexer_topk_idx(self, scores: "torch.Tensor", k: int) -> "torch.Tensor":
        """Top-k indices per row via the TRT-LLM IndexerTopK op (AIR-TopK) — the
        fastest available top-k for the dense ``[rows, seq]`` eviction select
        (~4-6x faster than torch.topk at scale, same kept SET for distinct
        scores). Returns ``[rows, k]`` int64 slot indices in ``[0, seq)``,
        unsorted (the caller scatters into a mask / re-sorts by slot). For ``k``
        beyond what the op can launch (``_INDEXER_TOPK_MAX_K``) it falls back to
        ``torch.topk``, which is correct for any ``k``."""
        rows, seq = scores.shape
        k = int(k)
        if k > self._INDEXER_TOPK_MAX_K:
            return torch.topk(scores, k, dim=1, sorted=False).indices.to(torch.long)
        out = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
        seq_lens = torch.full((rows,), seq, dtype=torch.int32, device=scores.device)
        torch.ops.trtllm.indexer_topk_decode(
            scores.contiguous().to(torch.float32), seq_lens, out, 1, k
        )
        return out.to(torch.long)

    def _select_union(
        self, per_head_scores: "torch.Tensor", combined: "torch.Tensor", keep_count: int
    ) -> "torch.Tensor":
        """Each head picks its top-``keep_count``; take the union; from the union
        keep the top-``keep_count`` by ``combined``; if the union is smaller,
        fill from the highest-scoring remaining tokens. Decode-relative
        indices."""
        n = int(combined.shape[0])
        if n <= keep_count:
            return torch.arange(n, device=combined.device, dtype=torch.long)
        union_mask = torch.zeros(n, device=combined.device, dtype=torch.bool)
        quota = min(keep_count, n)
        # Top-k over all (layer x head) rows at once via the TRT-LLM IndexerTopK
        # op: each row's top-quota is computed independently, collapsing H per-row
        # launches into one (H = num_layers * num_q_heads = 1152 for Qwen3-8B --
        # this was the dominant high-BS eviction cost). The union_mask scatter
        # below is order-independent, so the (unsorted) indices are fine.
        top_idx = self._indexer_topk_idx(per_head_scores, quota)
        union_mask[top_idx.reshape(-1)] = True
        union_idx = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_idx.numel() >= keep_count:
            subset = combined.index_select(0, union_idx)
            top_subset = self._indexer_topk_idx(subset.unsqueeze(0), keep_count).squeeze(0)
            return union_idx.index_select(0, torch.sort(top_subset).values)
        remaining = keep_count - int(union_idx.numel())
        if remaining > 0:
            residual = combined.clone()
            residual[union_mask] = float("-inf")
            extra = self._indexer_topk_idx(
                residual.unsqueeze(0), min(remaining, n - int(union_idx.numel()))
            ).squeeze(0)
            union_idx = torch.cat([union_idx, extra])
        return torch.sort(union_idx).values

    def _evict_layer_perhead(
        self, request: "LlmRequest", layer_idx: int, keep_2d: "torch.Tensor", seq_len: int
    ) -> None:
        """Physically compact ONE layer's cache, keeping a DIFFERENT token set per
        KV head. ``keep_2d`` is ``[num_kv_heads, keep_count]`` slot indices in
        ``[0, seq_len)``; each KV head's token axis is reordered independently
        (``[kept..., dropped...]``) so every slot in ``[0, seq_len)`` still holds
        a real key/value. Same HND layout + no-re-RoPE reasoning as
        the union compaction -- only the reorder is now per-head (a gather on the
        ``num_kv_heads`` axis) rather than one shared permutation."""
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        keep_2d = keep_2d.to(dtype=torch.long)
        keep_count = int(keep_2d.shape[1])
        if keep_count >= seq_len:
            return
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return
        pool = get_buffers(layer_idx, kv_layout="HND")
        if pool is None:
            return
        tokens_per_block = pool.shape[3]
        page_ids_t = torch.as_tensor(page_ids, device=pool.device, dtype=torch.long)
        request_pages = pool[page_ids_t]
        num_pages, kv_factor, num_kv_heads, _, head_dim = request_pages.shape
        keep_2d = keep_2d.to(request_pages.device)
        # [kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim]
        kv_by_token = (
            request_pages.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim)
        )
        # Per-head full token permutation [num_kv_heads, seq_len]: kept slots
        # first (in their given order), then the dropped slots -- so no slot in
        # [0, seq_len) is left holding stale data.
        all_token_ids = torch.arange(seq_len, device=kv_by_token.device, dtype=torch.long)
        new_orders: List[torch.Tensor] = []
        for h in range(num_kv_heads):
            is_dropped = torch.ones(seq_len, device=kv_by_token.device, dtype=torch.bool)
            is_dropped[keep_2d[h]] = False
            new_orders.append(torch.cat([keep_2d[h], all_token_ids[is_dropped]]))
        new_order = torch.stack(new_orders, dim=0)  # [num_kv_heads, seq_len]
        # Gather the token axis (dim 2) with a DIFFERENT order per KV head.
        region = kv_by_token[:, :, :seq_len]
        idx = new_order.view(1, num_kv_heads, seq_len, 1).expand(
            kv_factor, num_kv_heads, seq_len, head_dim
        )
        reordered = torch.gather(region, 2, idx).clone()
        kv_by_token[:, :, :seq_len] = reordered
        num_touched_pages = (seq_len + tokens_per_block - 1) // tokens_per_block
        repaged = (
            kv_by_token.reshape(kv_factor, num_kv_heads, num_pages, tokens_per_block, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        pool[page_ids_t[:num_touched_pages]] = repaged[:num_touched_pages]

    def _evict_requests(self, evict_reqs, num_layers: int) -> None:
        """Batched eviction over ALL requests evicting this step (per_head /
        per_layer_perhead / union): ONE per-head score launch over
        (request x layer) via ``triton_tri_score_perhead``, then
        then selects + compacts each request via ``_evict_modes(precomputed=)``.
        Updates ``self._evicted`` / ``request.py_kv_evicted_tokens``."""
        from .triattention_kernels import flat_perhead_to_list, triton_tri_score_perhead

        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        layer_pools = [get_buffers(layer, kv_layout="HND") for layer in range(num_layers)]
        if any(p is None for p in layer_pools):
            return
        device = layer_pools[0].device
        # Per-request committed-trim + page ids.
        kept = []  # (request, rid, page_ids_t, seq_len, round_start)
        for request, rid in evict_reqs:
            round_start = request.max_beam_num_tokens - 1
            seq_len = request.max_beam_num_tokens - self._evicted.get(rid, 0)
            if seq_len <= self.top_B:
                continue
            _k0 = self._read_request_k(request, 0, seq_len)
            if _k0 is not None:
                _nz = _k0.abs().sum(dim=(0, 2)) > 0
                if bool(_nz.any()):
                    _committed = int(_nz.nonzero().max()) + 1
                    if _committed < seq_len:
                        seq_len = _committed
            if seq_len <= self.top_B:
                continue
            page_ids = self._resolve_page_ids(request, 0)
            if not page_ids:
                continue
            kept.append(
                (
                    request,
                    rid,
                    torch.as_tensor(page_ids, device=device, dtype=torch.int64),
                    int(seq_len),
                    float(round_start),
                )
            )
        if not kept:
            return
        page_ids_list = [k[2] for k in kept]
        seq_lens = [k[3] for k in kept]
        round_starts = [k[4] for k in kept]
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        # Batched per-head score, grouped by storage (VSWA multi-pool safe),
        # emitting [H, seq] per (layer, request).
        from collections import defaultdict as _defaultdict

        storage_groups = _defaultdict(list)
        for layer in range(num_layers):
            storage_groups[layer_pools[layer].untyped_storage().data_ptr()].append(layer)
        req_layer_scores = [dict() for _ in kept]  # [req] -> {layer_idx: [H,seq]}
        for lids in storage_groups.values():
            ph, so, sm = triton_tri_score_perhead(
                layer_pools,
                page_ids_list,
                seq_lens,
                round_starts,
                self._triattn_q_real,
                self._triattn_q_imag,
                self._triattn_mlr_coef,
                self._freq_scale_sq,
                self.calibration["omega"],
                self._offsets,
                self._H,
                score_aggregation=self.score_aggregation,
                layer_indices=lids,
            )
            seg_list = flat_perhead_to_list(ph, so)
            for s, meta in enumerate(sm):
                req_layer_scores[meta.request_index][meta.layer_index] = seg_list[s]
        # Per request: select from the precomputed scores. All scores were taken
        # BEFORE any compaction; requests touch disjoint pages, so compacting one
        # does not disturb another's (already-read) scores.
        #
        # For union (1-D layer-uniform keep) we compute keep WITHOUT compacting,
        # accumulate per layer, then run ONE compaction per layer over all
        # requests -> K*L*2 compact launches collapse to L*2 (bit-exact;
        # kernel-equivalence validated by test_compact_equiv.py).
        is_union = self.eviction_mode == "union"
        union_by_layer = {} if is_union else None
        for r, (request, rid, _pi, seq_len, round_start) in enumerate(kept):
            precomputed = [req_layer_scores[r].get(layer) for layer in range(num_layers)]
            if any(p is None for p in precomputed):
                continue
            keep_count = self._evict_modes(request, num_layers, seq_len, precomputed=precomputed)
            if keep_count is None:
                continue
            if is_union and isinstance(keep_count, torch.Tensor):
                keep = keep_count
                for lid in range(num_layers):
                    grp = union_by_layer.setdefault(lid, ([], [], []))
                    grp[0].append(page_ids_list[r])
                    grp[1].append(keep)
                    grp[2].append(seq_len)
                keep_count = int(keep.numel())
            evicted = seq_len - keep_count
            if evicted > 0:
                self._evicted[rid] = self._evicted.get(rid, 0) + evicted
                request.py_kv_evicted_tokens = self._evicted[rid]
        if union_by_layer:
            from .triattention_kernels import triton_tri_compact

            for lid, (pl, kl, sl) in union_by_layer.items():
                triton_tri_compact(layer_pools[lid], pl, kl, sl)

    # ------------------------------------------------------------------ #
    # V2-manager cache access + physical eviction (HND physical layout)  #
    # ------------------------------------------------------------------ #

    def _resolve_page_ids(self, request: "LlmRequest", layer_idx: int) -> Optional[List[int]]:
        """Return the page (block) ids that hold THIS request's KV for one layer.

        The V2 KV cache is PAGED: a request's tokens live in several (possibly
        non-contiguous) fixed-size pages inside one big shared pool. Before we
        can read or compact a request's cache we must know WHICH pages are its
        own. V2's ``get_batch_cache_indices([ids], layer_idx)`` returns one list
        of block ids per requested id; we pass a single id and take ``[0]``.

        These ids index the PAGE axis (dim 0) of the tensor ``get_buffers``
        returns; the key/value split is a SEPARATE axis (kv_factor, dim 1) that
        callers index on their own. We do NOT divide or rescale the ids here.

        Falls back to the V1 ``get_cache_indices(request)`` signature, and returns
        ``None`` when neither API exists (e.g. mocked unit tests). Negative ids
        (unallocated slots) are filtered out.
        """
        mgr = self.kv_cache_manager
        get_batch = getattr(mgr, "get_batch_cache_indices", None)  # V2 API
        if get_batch is not None:
            try:
                batch = get_batch([request.py_request_id], layer_idx)
            except Exception:
                batch = None
            if batch:
                page_ids = [int(p) for p in batch[0] if int(p) >= 0]
                return page_ids or None
        get_single = getattr(mgr, "get_cache_indices", None)  # V1 fallback
        if get_single is not None:
            try:
                page_ids = get_single(request)
            except Exception:
                page_ids = None
            if page_ids:
                return [int(p) for p in page_ids if int(p) >= 0]
        return None

    def _read_request_k(
        self, request: "LlmRequest", layer_idx: int, seq_len: int
    ) -> Optional[torch.Tensor]:
        """Read this request's KEY tensor for one layer out of the paged pool.

        Steps: (1) get a VIEW of the layer's pool in HND layout, (2) slice out
        this request's pages, (3) take the KEY half, (4) merge the (page, slot)
        axes into one token axis and trim padding past ``seq_len``.

        Returns ``[num_kv_heads, seq_len, head_dim]`` (keys only), or ``None``
        when the manager exposes no readable pool (mocked tests).

        WHY HND: ``get_buffers`` reinterprets the SAME raw bytes under a chosen
        layout. The trtllm-gen / XQA attention kernel stores keys in HND
        ``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``; we
        MUST read with that same layout. ``get_buffers`` defaults to NHD (head
        and token axes swapped) -- reading NHD here would silently transpose
        heads and tokens and return scrambled keys.
        """
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return None
        # HND view: [num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]
        pool = get_buffers(layer_idx, kv_layout="HND")
        if pool is None:
            return None
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return None
        tokens_per_block = pool.shape[3]  # HND: dim 3 = slots per page
        KEY = 0  # kv_factor index: 0 = key, 1 = value
        # this request's pages, keys only:
        #   [num_pages, num_kv_heads, tokens_per_block, head_dim]
        pages = pool[page_ids][:, KEY]
        num_pages, num_kv_heads = pages.shape[0], pages.shape[1]
        # The logical token axis is (page, slot). Move num_kv_heads to the front,
        # then merge (page, slot) into one contiguous token axis:
        #   [num_kv_heads, num_pages, tokens_per_block, head_dim]
        #     -> [num_kv_heads, num_pages * tokens_per_block, head_dim]
        keys = pages.permute(1, 0, 2, 3).reshape(
            num_kv_heads, num_pages * tokens_per_block, pages.shape[3]
        )
        keys = keys[:, :seq_len, :]  # drop padding slots beyond seq_len
        return keys.contiguous()  # [num_kv_heads, seq_len, head_dim]

    def _num_layers_from_manager(self) -> int:
        mgr = self.kv_cache_manager
        layer_offsets = getattr(mgr, "layer_offsets", None)
        if layer_offsets:
            return len(layer_offsets)
        return self._L  # fall back to the calibrated layer count

    # ------------------------------------------------------------------ #
    # Helpers: calibration loading                                       #
    # ------------------------------------------------------------------ #

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the user-supplied calibration .pt and return our runtime schema.

        TriAttention does NOT compute calibration -- the user calibrates with the
        official tool (github.com/WeianMao/triattention) and passes that file via
        ``calibration_path``; we only run inference. Both the official R-KV layout
        (``{metadata, stats{"layerLL_headHH": {q_mean_real, q_mean_imag,
        q_abs_mean}}}``) and our already-converted flat layout are accepted -- the
        official one is converted here."""
        if self.calibration_path is None:
            raise ValueError(
                "TriAttention requires `calibration_path`: a calibration .pt from "
                "the official tool (github.com/WeianMao/triattention). TRT-LLM does "
                "not compute calibration -- see examples/ for the Qwen3-8B file and "
                "the official calibration instructions."
            )
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            calib = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in raw.items()}
            self._validate_calibration(calib)
            return calib
        if isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            return self._convert_official_calibration(raw)
        got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"Unrecognized calibration at {self.calibration_path}: expected the "
            f"official {{metadata, stats}} layout or "
            f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
        )

    def _convert_official_calibration(self, raw) -> Dict[str, torch.Tensor]:
        """Convert the official per-(layer, head) stats to our flat runtime schema.

        ``E_q[l,h] = q_mean_real + i*q_mean_imag`` and ``E_q_norm[l,h] =
        q_abs_mean`` are the same statistic, just restacked into ``[L, H, F]``.
        ``omega`` / ``freq_scale_sq`` are not in the official file (its runtime
        recomputes them from the model rotary), so we derive them from the model
        config -- model-intrinsic and corpus-independent."""
        stats = raw["stats"]
        meta = raw.get("metadata", {})
        if "sampled_heads" in meta:
            heads = [(int(a), int(b)) for a, b in meta["sampled_heads"]]
        else:
            heads = [
                (int(k[len("layer") : k.index("_head")]), int(k[k.index("_head") + len("_head") :]))
                for k in stats
            ]
        num_layers = max(layer for layer, _ in heads) + 1
        num_heads = max(h for _, h in heads) + 1
        freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
        E_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
        E_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
        for layer, h in heads:
            s = stats[f"layer{layer:02d}_head{h:02d}"]
            E_q[layer, h] = torch.complex(s["q_mean_real"].float(), s["q_mean_imag"].float())
            E_q_norm[layer, h] = s["q_abs_mean"].float()
        omega, freq_scale_sq = self._rope_tables(freq_count)
        calib = {
            "E_q": E_q.to("cuda"),
            "E_q_norm": E_q_norm.to("cuda"),
            "omega": omega.to("cuda"),
            "freq_scale_sq": freq_scale_sq.to("cuda"),
        }
        self._validate_calibration(calib)
        logger.info(
            f"TriAttention: converted official calibration {self.calibration_path}"
            f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
        )
        return calib

    def _rope_tables(self, freq_count: int):
        """RoPE ``omega`` (inv_freq) + ``freq_scale_sq`` (squared position-0
        amplitude) from the model config -- model-intrinsic, corpus-independent
        (the official file does not store them). transformers' rope-init handles
        plain and scaled RoPE; plain RoPE has attention_factor 1 so freq_scale_sq
        is all ones. Falls back to the analytic inv_freq if rope-init is absent."""
        if self.model_path is None:
            raise ValueError(
                "TriAttention needs `model_path` to derive the RoPE tables "
                "(omega / freq_scale_sq) when converting the official calibration."
            )
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        cfg = getattr(cfg, "text_config", cfg)
        head_dim = freq_count * 2
        base = float(getattr(cfg, "rope_theta", 10000.0))
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            scaling = getattr(cfg, "rope_scaling", None) or {}
            rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](cfg, device="cpu")
            omega = inv_freq.to(torch.float32)[:freq_count].clone()
            scale_sq = float(attention_factor) ** 2
        except Exception:
            idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
            omega = (1.0 / (base ** (idx / head_dim)))[:freq_count].clone()
            scale_sq = 1.0
        return omega, torch.full((freq_count,), scale_sq, dtype=torch.float32)

    def _validate_calibration(self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys."""
        missing = _REQUIRED_CALIBRATION_KEYS - set(calibration.keys())
        if missing:
            raise ValueError(
                f"TriAttention calibration is missing keys: {sorted(missing)}; "
                f"got {sorted(calibration.keys())}."
            )
