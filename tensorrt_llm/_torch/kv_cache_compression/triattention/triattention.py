"""TriAttention KV-cache compression: periodic physical KV eviction.

Every ``beta`` generation steps TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the eviction runs in one pre-forward ``on_generation_step_begin`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager` and nothing more -- it
has no attention backend of its own; decode runs the model's standard dense
kernel over the compacted cache. TriAttention derives each request's effective
pre-forward physical length from V2 capacity and any unconsumed compaction;
it writes that value through ``adjust_attention_metadata`` just before
``attn_metadata.prepare()``. Physical reclaim uses V2's existing resize path:
TriAttention publishes a request-scoped capacity-only target and trailing pages
return to the pool after compaction finishes.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache. See ``_evict_layer_perhead``.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The model engine keeps the decode query at its true absolute
position while the attention metadata uses the compacted physical length, so a
query against a kept key at its original rotation still yields the correct
relative distance.

Calibration is NOT computed here: the user calibrates with the official tool
(github.com/WeianMao/triattention) and passes that .pt via ``calibration_path``;
the manager converts it to our runtime schema at load (see _resolve_calibration).
The scoring math follows the same upstream reference (``methods/pruning_utils.py``).
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

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


def _validate_swa_rebase(seq_len: int, keep_count: int, window_size: int) -> None:
    if window_size <= 0:
        raise ValueError(f"SWA window size must be positive, got {window_size}")
    if keep_count < window_size:
        raise ValueError(
            f"TriAttention compacted length {keep_count} must be at least the "
            f"SWA window size {window_size}"
        )
    if keep_count > seq_len:
        raise ValueError(
            f"TriAttention compacted length {keep_count} exceeds cache length {seq_len}"
        )


def _build_swa_rebase_keep(
    seq_len: int,
    keep_count: int,
    window_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return source slots that place the latest SWA window at the new tail."""
    _validate_swa_rebase(seq_len, keep_count, window_size)
    prefix = torch.arange(keep_count - window_size, device=device, dtype=torch.long)
    recent = torch.arange(seq_len - window_size, seq_len, device=device, dtype=torch.long)
    return torch.cat([prefix, recent])


def _build_swa_rebase_copy(
    seq_len: int,
    keep_count: int,
    window_size: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the source and destination ranges needed for an SWA rebase."""
    _validate_swa_rebase(seq_len, keep_count, window_size)
    source = torch.arange(seq_len - window_size, seq_len, device=device, dtype=torch.long)
    destination = torch.arange(
        keep_count - window_size, keep_count, device=device, dtype=torch.long
    )
    return source, destination


def _request_draft_length(request: "LlmRequest") -> int:
    """Return the active speculative-draft length exposed by a request."""
    draft_tokens = getattr(request, "py_draft_tokens", None)
    if draft_tokens is not None and len(draft_tokens) > 0:
        return len(draft_tokens)
    for field in ("num_draft_tokens", "py_num_draft_tokens"):
        value = getattr(request, field, 0)
        if isinstance(value, int) and value > 0:
            return value
    if (
        getattr(request, "is_disagg_generation_transmission_complete", False)
        and (context_params := getattr(request, "context_phase_params", None)) is not None
        and (context_drafts := getattr(context_params, "draft_tokens", None)) is not None
        and len(context_drafts) > 0
    ):
        return len(context_drafts)
    return 0


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_begin``: every ``beta`` generation steps it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    keep set. Full-attention layers are scored; kernel-masked SWA layers preserve
    their latest window in the same compacted prefix. Every layer ends with the
    same request-wide cached length.
    """

    # The TRT-LLM IndexerTopK op's single-block kernel sizes dynamic shared memory
    # as k * 4 bytes with no opt-in past the 48 KB per-block cap; its heuristic
    # only supports k in {512, 1024, 2048}, so k above 2048 must fall back.
    # Independently of k, the op splits a row into multiple 2048-column sub-blocks
    # once the score width reaches 4096. That path requires caller-owned radix
    # scratch, which this caller does not provide. torch.topk is exact for both
    # unsupported cases, so either boundary can fall back without changing the
    # retained set.
    _INDEXER_TOPK_MAX_K = 2048
    _INDEXER_TOPK_SUBBLOCK = 2048

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
        count_prompt_tokens: bool = False,
        skip_swa: bool = True,
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
        # cpt=False (default): budget counts DECODE tokens only (pinned prompt is
        # extra). cpt=True: budget INCLUDES the pinned prompt.
        self.count_prompt_tokens = bool(count_prompt_tokens)
        if not self.pin_prefill or self.count_prompt_tokens:
            raise ValueError(
                "TriAttention physical KV reclaim requires pin_prefill=True and "
                "count_prompt_tokens=False so finalized prompt KV is preserved"
            )
        self.skip_swa = bool(skip_swa)
        # per_head / per_layer_perhead compaction backend:
        #   "torch" (default) -- the vectorized PyTorch reorder (_build_new_order +
        #     _evict_layer_perhead). MEASURED FASTEST at BS=32 (per_head -13% vs
        #     dense; per_layer -23%), beating both the old per-head loop (-35%) and
        #     the fused Triton kernel below (-23%): once the per-head loop is
        #     vectorized, the Triton per-layer scratch + 2-phase overhead dominates.
        #   "triton"           -- one fused batched Triton launch per layer
        #     (triton_tri_compact_perhead); may win at high-BS serving (the
        #     launch-amortization regime) -- re-benchmark there. Both backends
        #     produce a byte-identical compacted cache.
        self._compact_backend = os.environ.get("TRIATTN_COMPACT_BACKEND", "torch")
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
        # used for RoPE tables and local layer_types/sliding_window metadata.
        self.model_path = model_path
        if self.skip_swa and self.model_path is None:
            raise ValueError(
                "TriAttention skip_swa=True requires model_path so kernel-masked "
                "sliding-attention layers can be classified safely"
            )
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
        # Cumulative physically-evicted token count per request, consumed by the
        # public introspection API.
        self._evicted: Dict[int, int] = {}
        # Authoritative written-KV length for the forward being prepared.
        self._pre_forward_kv_lengths: Dict[int, int] = {}
        # Context requests initialize through the framework hook; generation-only
        # disaggregated requests initialize lazily in the pre-forward hook.
        self._capacity_only_request_ids: set[int] = set()

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Mark capacity-only decode and resolve calibration once.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        request_id = request.py_request_id
        if request_id not in self._capacity_only_request_ids:
            self._validate_v2_compatibility()
            num_layers = self._num_layers_from_manager()
            if num_layers is not None:
                self._attention_layer_partition(num_layers)
            request.py_kv_cache_decode_capacity_only = True
            self._capacity_only_request_ids.add(request_id)
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

    def _validate_v2_compatibility(self) -> None:
        """Reject runtime modes that do not follow single-token full-attention V2."""
        manager = self.kv_cache_manager
        mapping = getattr(manager, "mapping", None)
        if getattr(mapping, "enable_attention_dp", False):
            raise ValueError("TriAttention does not support attention DP")
        if getattr(manager, "is_disagg", False):
            raise ValueError("TriAttention does not support disaggregated serving")
        if (
            getattr(manager, "max_beam_width", 1) != 1
            or getattr(manager, "num_extra_kv_tokens", 0)
            or getattr(manager, "max_total_draft_tokens", 0)
            or getattr(manager, "_kv_reserve_draft_tokens", 0)
        ):
            raise ValueError("TriAttention requires single-token, beam-width-one decoding")
        windows = getattr(manager, "max_attention_window_vec", ())
        config = getattr(manager, "kv_cache_manager_py_config", None)
        layers = getattr(config, "layers", ())
        if any(window is not None for window in windows) or any(
            not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
            for layer in layers
        ):
            raise ValueError(
                "TriAttention requires full-attention V2 lifecycles; native SWA, "
                "VSWA, and SSM pools are not supported"
            )

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
        score the full cache, select top-B, physically compact, reconcile the
        forward's attention metadata, and resize V2 to the compacted capacity.

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
        and physically evict to the pinned prompt plus top-B decode tokens."""
        gen_requests = getattr(scheduled_batch, "generation_requests", None)
        if not gen_requests:
            return
        active_requests = []
        for request in gen_requests:
            if bool(getattr(request, "is_dummy", False)):
                continue
            if request.py_request_id not in self._capacity_only_request_ids:
                self.on_request_init(request)
            active_requests.append(request)
        if not active_requests or not self._calibrated:
            return
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        num_layers = self._num_layers_from_manager()

        # (1) bump per-request step counters; collect who evicts THIS step.
        evict_now = []
        for request in active_requests:
            rid = request.py_request_id
            draft_len = _request_draft_length(request)
            if draft_len:
                raise ValueError(
                    "TriAttention physical eviction does not support speculative "
                    f"decoding; request {rid} has {draft_len} draft tokens"
                )
            kv_cache = mgr.kv_cache_map.get(rid)
            if kv_cache is None or not kv_cache.is_active:
                raise RuntimeError(f"Request {rid} has no active V2 KV cache")
            raw_capacity = int(kv_cache.capacity)
            compaction = getattr(request, "py_kv_cache_compaction", None)
            if compaction is not None:
                target_capacity, published_capacity, _ = compaction
                capacity_growth = raw_capacity - published_capacity
                if capacity_growth < 0:
                    raise RuntimeError(
                        f"Request {rid} capacity {raw_capacity} fell below "
                        f"published capacity {published_capacity}"
                    )
                effective_capacity = target_capacity + capacity_growth
            else:
                effective_capacity = raw_capacity
            # The scheduler has reserved one unwritten slot for this forward.
            seq_len = effective_capacity - 1
            if seq_len < kv_cache.history_length:
                raise RuntimeError(
                    f"Request {rid} KV length {seq_len} is below finalized "
                    f"history {kv_cache.history_length}"
                )
            self._pre_forward_kv_lengths[rid] = seq_len
            step = self._gen_steps.get(rid, 0) + 1
            self._gen_steps[rid] = step
            if step % self.beta == 0:
                if compaction is not None:
                    self._gen_steps[rid] = step - 1
                    continue
                if seq_len > self._minimum_evictable_length(request, seq_len):
                    evict_now.append((request, rid))

        # (2) Compact all affected dense and kernel-masked SWA layers, then publish
        # one target per request. V2 consumes it at its existing post-enqueue
        # update boundary, where host block-table mutation is overlap-safe.
        capacity_targets = self._evict_requests(evict_now, num_layers) if evict_now else []
        if capacity_targets:
            compaction_event = torch.cuda.Event()
            compaction_event.record()
            requests_by_id = {request.py_request_id: request for request in active_requests}
            for rid, target_capacity in capacity_targets:
                kv_cache = mgr.kv_cache_map.get(rid)
                if kv_cache is None or not kv_cache.is_active:
                    raise RuntimeError(f"Request {rid} has no active V2 KV cache")
                if target_capacity > kv_cache.capacity:
                    raise RuntimeError(
                        f"Request {rid} compacted capacity {target_capacity} exceeds "
                        f"current capacity {kv_cache.capacity}"
                    )
                request = requests_by_id[rid]
                if getattr(request, "py_kv_cache_compaction", None) is not None:
                    raise RuntimeError(f"Request {rid} already has an unconsumed KV compaction")
                request.py_kv_cache_compaction = (
                    target_capacity,
                    int(kv_cache.capacity),
                    compaction_event,
                )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity.

        With a decode-only budget, pinned prompt tokens do not consume ``top_B``.
        Selection therefore keeps every token until the cache exceeds
        ``prompt_len + top_B``.
        """
        if self.pin_prefill and not self.count_prompt_tokens:
            prompt_len = min(int(getattr(request, "py_prompt_len", 0) or 0), seq_len)
            return prompt_len + self.top_B
        return self.top_B

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request length and eviction state."""
        compaction = getattr(request, "py_kv_cache_compaction", None)
        if compaction is not None and compaction[2] is not None:
            self.kv_cache_manager._stream.wait_event(compaction[2])
        request.py_kv_cache_decode_capacity_only = False
        request.py_kv_cache_compaction = None
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)
        self._pre_forward_kv_lengths.pop(request.py_request_id, None)
        self._capacity_only_request_ids.discard(request.py_request_id)

    # ------------------------------------------------------------------ #
    # Attention-metadata reconcile (compression-framework hook)          #
    # ------------------------------------------------------------------ #

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id``."""
        return self._evicted.get(request_id, 0)

    def adjust_attention_metadata(self, attn_metadata) -> None:
        """Reconcile the attention metadata for this iteration's eviction.

        The framework calls this immediately before ``attn_metadata.prepare()``.
        V2 capacity after scheduling gives the written KV length for this
        forward. Use that physical value instead of reconstructing it from
        logical request length.
        Prompt length is clamped when necessary so the prompt/generation split
        cannot extend beyond the compacted prefix.
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
            nc = self._pre_forward_kv_lengths.get(req_ids[i])
            if nc is None:
                continue
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
        """Select and compact from precomputed dense-layer scores.

        Returns either the uniform kept count or, for union mode, the sorted
        shared keep tensor whose compaction is batched by the caller. Returns
        ``None`` if no dense layer could be scored.

        ``precomputed``: per-layer ``[H, seq_len]`` scores (from the fused
        score kernel ``triton_tri_score_perhead``); used verbatim for the
        selection and compaction. Kernel-masked SWA entries are ``None`` and are
        excluded by the dense-layer partition.
        """
        budget = min(self.top_B, seq_len)
        # Prompt pin: only decode tokens [decode_start, seq_len) compete; the
        # first decode_start (prompt) tokens are always kept.
        decode_start = 0
        if self.pin_prefill:
            decode_start = min(int(getattr(request, "py_prompt_len", 0) or 0), seq_len)
        decode_count = seq_len - decode_start
        decode_budget = (budget - decode_start) if self.count_prompt_tokens else budget
        # Budget exhausted by the pinned prompt (or no decode tokens): keep the
        # first `budget` contiguous slots everywhere; nothing per-head to do.
        if decode_budget <= 0 or decode_count <= 0:
            return budget

        # Select from the precomputed per-layer K scores. For per_head /
        # per_layer_perhead reduce to per-KV-head [nkv, decode_count] (group-max),
        # z-normalized over the decode region. For union keep the raw per-head
        # rows to stack.
        dense_layers = self._dense_layers(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        # Number of KV heads from the first scored layer (HND dim 2).
        get_buffers = getattr(self.kv_cache_manager, "get_buffers", None)
        first_global_layer = self._global_layer_id(dense_layers[0], num_layers)
        p0 = get_buffers(first_global_layer, kv_layout="HND") if get_buffers is not None else None
        if p0 is None:
            return None
        num_kv_heads = int(p0.shape[2])
        per_layer_kv_scores: List[torch.Tensor] = []
        union_rows: List[torch.Tensor] = []
        dense_set = set(dense_layers)
        for layer_idx in range(num_layers):
            if layer_idx not in dense_set:
                continue
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
                request, num_layers, seq_len, decode_start, decode_budget, per_layer_kv_scores
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
        # One keep set shared by every (dense) layer.
        self._compact_perhead_layers(request, self._dense_layers(num_layers), keep_2d, seq_len)
        return int(keep_2d.shape[1])

    def _compact_perhead_layers(
        self, request: "LlmRequest", layer_indices: List[int], keep_2d: "torch.Tensor", seq_len: int
    ) -> None:
        """Physically compact ``layer_indices`` keeping a per-KV-head token set
        (``keep_2d`` = ``[num_kv_heads, keep_count]``). Two interchangeable backends
        (``self._compact_backend``) producing a byte-identical compacted cache:
        ``triton`` runs one fused ``triton_tri_compact_perhead`` launch per layer;
        ``torch`` runs the vectorized PyTorch reorder. SWA layers are excluded by
        the caller (``layer_indices`` is already dense-only)."""
        if int(keep_2d.shape[1]) >= seq_len:
            return  # nothing to drop
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            raise RuntimeError("TriAttention requires KVCacheManagerV2.get_buffers()")
        num_layers = self._num_layers_from_manager()
        if num_layers is None:
            raise RuntimeError("TriAttention could not resolve the local attention layer count")
        prepared_layers = []
        for layer_idx in layer_indices:
            global_layer = self._global_layer_id(layer_idx, num_layers)
            pool = get_buffers(global_layer, kv_layout="HND")
            page_ids = self._resolve_page_ids(request, layer_idx)
            if pool is None:
                raise RuntimeError(f"Missing KV pool for attention layer {global_layer}")
            if not page_ids:
                raise RuntimeError(
                    f"Missing KV page ids for attention layer {global_layer} "
                    f"of request {request.py_request_id}"
                )
            prepared_layers.append((layer_idx, pool, page_ids))
        if self._compact_backend == "torch":
            new_order = self._build_new_order(keep_2d, seq_len)
            for layer_idx, _, _ in prepared_layers:
                self._evict_layer_perhead(request, layer_idx, new_order, seq_len)
            return
        from .triattention_kernels import triton_tri_compact_perhead

        for _, pool, page_ids in prepared_layers:
            page_ids_t = torch.as_tensor(page_ids, device=pool.device, dtype=torch.long)
            triton_tri_compact_perhead(pool, [page_ids_t], [keep_2d.to(pool.device)], [seq_len])

    def _evict_per_layer_perhead(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        per_layer_kv_scores: List["torch.Tensor"],
    ) -> Optional[int]:
        """per_layer_perhead: each (layer, KV head) selects independently from
        that layer's per-KV-head max (upstream
        ``_select_per_layer_perhead_independent``). ``per_layer_kv_scores`` is
        DENSE-ONLY (sliding-window layers skipped in ``_evict_modes``), so pair it
        with the real dense layer indices rather than ``enumerate`` (whose 0..N-1
        running index would no longer match the physical layer)."""
        keep_count = None
        for layer_idx, kv_scores in zip(self._dense_layers(num_layers), per_layer_kv_scores):
            keep_2d = self._decode_topk(kv_scores, decode_start, decode_budget)
            self._compact_perhead_layers(request, [layer_idx], keep_2d, seq_len)
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
        unsorted (the caller scatters into a mask / re-sorts by slot). Falls back
        to ``torch.topk`` when either ``k`` exceeds the launch cap or the score
        width requires the unsupported multi-block radix path."""
        rows, seq = scores.shape
        k = int(k)
        if k > self._INDEXER_TOPK_MAX_K or seq >= 2 * self._INDEXER_TOPK_SUBBLOCK:
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

    def _build_new_order(self, keep_2d: "torch.Tensor", seq_len: int) -> "torch.Tensor":
        """Per-KV-head full token permutation ``[num_kv_heads, seq_len]``: each row =
        kept slots (in their given order) then the dropped slots (ascending), so no
        slot in ``[0, seq_len)`` is left holding stale data. Vectorized over heads
        (one scatter + one nonzero + one cat) -- replaces the old per-head Python
        loop, which issued ~num_kv_heads x more tiny ones/index_put/cat launches (the
        dominant per_head eviction cost in the nsys kernel-launch profile). The
        result is identical to the loop: kept-then-dropped, dropped ascending."""
        keep_2d = keep_2d.to(dtype=torch.long)
        nkv, keep_count = keep_2d.shape
        if keep_count >= seq_len:
            return keep_2d[:, :seq_len].contiguous()
        kept_mask = torch.zeros(nkv, seq_len, device=keep_2d.device, dtype=torch.bool)
        kept_mask.scatter_(1, keep_2d, True)
        # nonzero is row-major sorted -> per-head dropped cols ascending; every row
        # drops exactly seq_len-keep_count (keep_2d rows are unique), so reshape groups.
        dropped = (~kept_mask).nonzero(as_tuple=True)[1].reshape(nkv, seq_len - keep_count)
        return torch.cat([keep_2d, dropped], dim=1)

    def _local_to_global_layers(self, num_layers: int) -> List[int]:
        """Return V2's global layer id for every local TriAttention layer slot."""
        cached = getattr(self, "_local_to_global_layers_cache", None)
        if cached is not None:
            if len(cached) != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {len(cached)} to {num_layers}"
                )
            return cached

        mgr = self.kv_cache_manager
        pp_layers = getattr(mgr, "pp_layers", None)
        if pp_layers is not None:
            global_layers = [int(layer) for layer in pp_layers]
            if len(global_layers) != num_layers:
                raise ValueError(
                    f"KVCacheManagerV2 exposes {len(global_layers)} PP layers, "
                    f"but TriAttention received {num_layers} local layers"
                )
        else:
            total_layers = int(getattr(mgr, "num_layers", num_layers))
            if total_layers != num_layers:
                raise ValueError(
                    "TriAttention requires KVCacheManagerV2.pp_layers for pipeline-parallel "
                    "local-to-global layer mapping"
                )
            global_layers = list(range(num_layers))
        self._local_to_global_layers_cache = global_layers
        return global_layers

    def _global_layer_id(self, local_layer: int, num_layers: int) -> int:
        return self._local_to_global_layers(num_layers)[local_layer]

    @staticmethod
    def _has_sliding_window_signal(config) -> bool:
        """Return whether config metadata hints at sliding attention."""
        use_sliding_window = getattr(config, "use_sliding_window", None)
        if isinstance(use_sliding_window, bool):
            return use_sliding_window
        for field in (
            "sliding_window",
            "sliding_window_size",
            "sliding_window_pattern",
            "max_window_layers",
        ):
            value = getattr(config, field, None)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value > 0:
                    return True
            elif value:
                return True
        return False

    def _attention_layer_partition(
        self, num_layers: int
    ) -> Tuple[List[int], List[int], Optional[int]]:
        """Return dense layers, kernel-masked SWA layers, and the SWA window.

        TriAttention initialization has already rejected real V2 windowed
        lifecycles. A sliding layer found here is therefore stored at full length
        and applies its window only in the attention kernel.
        """
        if not self.skip_swa:
            return list(range(num_layers)), [], None
        cached = getattr(self, "_attention_layer_partition_cache", None)
        if cached is not None:
            return cached

        model_path = getattr(self, "model_path", None)
        if model_path is None:
            raise ValueError("TriAttention skip_swa=True requires model_path")

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise ValueError(
                f"TriAttention could not load the local model config from {model_path!r}"
            ) from exc
        config = getattr(config, "text_config", config)
        layer_types = getattr(config, "layer_types", None)
        if not layer_types:
            if self._has_sliding_window_signal(config):
                raise ValueError(
                    "Model config exposes sliding-window metadata but no layer_types; "
                    "TriAttention cannot classify kernel-masked SWA layers safely"
                )
            result = (list(range(num_layers)), [], None)
            self._attention_layer_partition_cache = result
            return result
        global_layers = self._local_to_global_layers(num_layers)
        if global_layers and max(global_layers) >= len(layer_types):
            raise ValueError(
                f"Model config has {len(layer_types)} layer_types entries, "
                f"but this PP rank references global layer {max(global_layers)}"
            )

        swa_layers = [
            local_layer
            for local_layer, global_layer in enumerate(global_layers)
            if "sliding" in str(layer_types[global_layer]).lower()
        ]
        swa_set = set(swa_layers)
        dense_layers = [layer for layer in range(num_layers) if layer not in swa_set]
        window_size = None
        if swa_layers:
            raw_window = getattr(config, "sliding_window", None)
            if not isinstance(raw_window, int) or raw_window <= 0:
                raise ValueError(
                    "TriAttention requires a positive integer model sliding_window "
                    "when layer_types contains sliding attention"
                )
            if self.top_B < raw_window:
                raise ValueError(
                    f"TriAttention decode budget top_B={self.top_B} must be at least "
                    f"the kernel-masked SWA window size {raw_window}"
                )
            window_size = raw_window
        result = (dense_layers, swa_layers, window_size)
        self._attention_layer_partition_cache = result
        return result

    def _dense_layers(self, num_layers: int) -> List[int]:
        """Return full-attention layers used for TriAttention scoring."""
        return self._attention_layer_partition(num_layers)[0]

    def _evict_layer_perhead(
        self, request: "LlmRequest", layer_idx: int, new_order: "torch.Tensor", seq_len: int
    ) -> None:
        """Physically compact ONE layer's cache, keeping a DIFFERENT token set per
        KV head. ``new_order`` is the ``[num_kv_heads, seq_len]`` per-head token
        permutation from ``_build_new_order`` (kept-then-dropped); each KV head's
        token axis is reordered independently (a gather on the ``num_kv_heads`` axis)
        so every slot in ``[0, seq_len)`` still holds a real key/value. Same HND
        layout + no-re-RoPE reasoning as the union compaction. ``new_order`` is built
        ONCE by the caller (it is identical across layers in per_head mode) rather
        than rebuilt per layer."""
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            raise RuntimeError("TriAttention requires KVCacheManagerV2.get_buffers()")
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            raise RuntimeError(
                f"Missing KV page ids for local attention layer {layer_idx} "
                f"of request {request.py_request_id}"
            )
        num_layers = self._num_layers_from_manager()
        if num_layers is None:
            raise RuntimeError("TriAttention could not resolve the local attention layer count")
        global_layer = self._global_layer_id(layer_idx, num_layers)
        pool = get_buffers(global_layer, kv_layout="HND")
        if pool is None:
            raise RuntimeError(f"Missing KV pool for attention layer {global_layer}")
        tokens_per_block = pool.shape[3]
        page_ids_t = torch.as_tensor(page_ids, device=pool.device, dtype=torch.long)
        request_pages = pool[page_ids_t]
        num_pages, kv_factor, num_kv_heads, _, head_dim = request_pages.shape
        new_order = new_order.to(request_pages.device)
        # [kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim]
        kv_by_token = (
            request_pages.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim)
        )
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

    def _evict_requests(self, evict_reqs, num_layers: int) -> List[Tuple[int, int]]:
        """Score and compact requests, returning ``(request_id, capacity)`` targets.

        Only full-attention layers participate in scoring. For kernel-masked SWA
        layers, the latest model window is rebased to the tail of the common
        compacted prefix before the request-wide capacity is reduced.
        """
        from .triattention_kernels import flat_perhead_to_list, triton_tri_score_perhead

        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            raise RuntimeError("TriAttention requires KVCacheManagerV2.get_buffers()")
        global_layers = self._local_to_global_layers(num_layers)
        layer_pools = [get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(p is None for p in layer_pools):
            missing = [layer for layer, pool in zip(global_layers, layer_pools) if pool is None]
            raise RuntimeError(f"Missing KV pools for attention layers {missing}")
        dense_layers, swa_layers, swa_window = self._attention_layer_partition(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        first_dense_layer = dense_layers[0]
        device = layer_pools[first_dense_layer].device

        # Dense pools may have distinct block tables. Resolve one representative
        # layer per storage group and reuse its page ids only within that group.
        from collections import defaultdict as _defaultdict

        storage_groups = _defaultdict(list)
        for layer in dense_layers:
            storage_groups[layer_pools[layer].untyped_storage().data_ptr()].append(layer)
        dense_group_representatives = [layers[0] for layers in storage_groups.values()]
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }

        # Resolve request length and page metadata before mutating any layer.
        prepared = []
        for request, rid in evict_reqs:
            draft_len = _request_draft_length(request)
            if draft_len:
                raise ValueError(
                    "TriAttention physical eviction does not support speculative "
                    f"decoding; request {rid} has {draft_len} draft tokens"
                )
            seq_len = self._pre_forward_kv_lengths.get(rid)
            if seq_len is None:
                raise RuntimeError(f"Missing authoritative pre-forward KV length for request {rid}")
            # Restore the uncompressed logical position from the authoritative
            # physical prefix. This is max_beam-1 without overlap and max_beam
            # for the previous-tensor overlap path, without branching on executor mode.
            round_start = seq_len + self._evicted.get(rid, 0)
            if seq_len <= self._minimum_evictable_length(request, seq_len):
                continue
            dense_page_ids_by_group = {}
            for representative in dense_group_representatives:
                group_page_ids = self._resolve_page_ids(request, representative)
                if not group_page_ids:
                    global_layer = global_layers[representative]
                    raise RuntimeError(
                        f"Missing KV page ids for attention layer {global_layer} of request {rid}"
                    )
                dense_page_ids_by_group[representative] = torch.as_tensor(
                    group_page_ids,
                    device=layer_pools[representative].device,
                    dtype=torch.int64,
                )
            expected_keep_count = self._minimum_evictable_length(request, seq_len)
            swa_source = None
            swa_destination = None
            swa_page_ids = {}
            if swa_layers:
                assert swa_window is not None
                swa_source, swa_destination = _build_swa_rebase_copy(
                    seq_len, expected_keep_count, swa_window, device=device
                )
                for layer in swa_layers:
                    layer_page_ids = self._resolve_page_ids(request, layer)
                    if not layer_page_ids:
                        raise RuntimeError(
                            f"Missing KV page ids for kernel-masked SWA layer {layer} "
                            f"of request {rid}"
                        )
                    swa_page_ids[layer] = torch.as_tensor(
                        layer_page_ids,
                        device=layer_pools[layer].device,
                        dtype=torch.int64,
                    )
            prepared.append(
                {
                    "request": request,
                    "request_id": rid,
                    "dense_page_ids_by_group": dense_page_ids_by_group,
                    "seq_len": int(seq_len),
                    "round_start": float(round_start),
                    "expected_keep_count": expected_keep_count,
                    "swa_source": swa_source,
                    "swa_destination": swa_destination,
                    "swa_page_ids": swa_page_ids,
                }
            )
        if not prepared:
            return []
        seq_lens = [item["seq_len"] for item in prepared]
        round_starts = [item["round_start"] for item in prepared]
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        if global_layers and max(global_layers) >= self._triattn_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._triattn_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        if global_layers == list(range(global_layers[0], global_layers[0] + num_layers)):
            layer_slice = slice(global_layers[0], global_layers[0] + num_layers)
            q_real = self._triattn_q_real[layer_slice]
            q_imag = self._triattn_q_imag[layer_slice]
            mlr_coef = self._triattn_mlr_coef[layer_slice]
        else:
            layer_ids = torch.as_tensor(
                global_layers, device=self._triattn_q_real.device, dtype=torch.long
            )
            q_real = self._triattn_q_real.index_select(0, layer_ids)
            q_imag = self._triattn_q_imag.index_select(0, layer_ids)
            mlr_coef = self._triattn_mlr_coef.index_select(0, layer_ids)
        # Score only dense layers, grouped by their backing storage.
        req_layer_scores = [dict() for _ in prepared]  # [req] -> {layer_idx: [H,seq]}
        for lids in storage_groups.values():
            representative = lids[0]
            group_page_ids = [item["dense_page_ids_by_group"][representative] for item in prepared]
            ph, so, sm = triton_tri_score_perhead(
                layer_pools,
                group_page_ids,
                seq_lens,
                round_starts,
                q_real,
                q_imag,
                mlr_coef,
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
        # before any compaction; requests touch disjoint pages, so compacting one
        # does not disturb another's (already-read) scores.
        #
        # For union (1-D layer-uniform keep) we compute keep WITHOUT compacting,
        # accumulate per layer, then run ONE compaction per layer over all
        # requests -> K*L*2 compact launches collapse to L*2 (bit-exact;
        # kernel-equivalence validated by test_compact_equiv.py).
        is_union = self.eviction_mode == "union"
        union_by_layer = {} if is_union else None
        swa_by_layer = {}
        pending_updates = []
        for r, item in enumerate(prepared):
            request = item["request"]
            rid = item["request_id"]
            seq_len = item["seq_len"]
            precomputed = [req_layer_scores[r].get(layer) for layer in range(num_layers)]
            if any(precomputed[layer] is None for layer in dense_layers):
                continue
            keep_count = self._evict_modes(request, num_layers, seq_len, precomputed=precomputed)
            if keep_count is None:
                continue
            if is_union and isinstance(keep_count, torch.Tensor):
                keep = keep_count
                keep_count = int(keep.numel())
                for lid in dense_layers:
                    grp = union_by_layer.setdefault(lid, ([], [], []))
                    representative = layer_group_representative[lid]
                    grp[0].append(item["dense_page_ids_by_group"][representative])
                    grp[1].append(keep)
                    grp[2].append(seq_len)
            if keep_count != item["expected_keep_count"]:
                raise RuntimeError(
                    f"TriAttention selected {keep_count} tokens for request {rid}, "
                    f"expected {item['expected_keep_count']}"
                )
            evicted = seq_len - keep_count
            if evicted > 0:
                for lid in swa_layers:
                    grp = swa_by_layer.setdefault(lid, ([], [], [], []))
                    grp[0].append(item["swa_page_ids"][lid])
                    grp[1].append(item["swa_source"])
                    grp[2].append(seq_len)
                    grp[3].append(item["swa_destination"])
                pending_updates.append((rid, evicted, keep_count))

        if union_by_layer or swa_by_layer:
            from .triattention_kernels import triton_tri_compact

        if union_by_layer:
            for lid, (pl, kl, sl) in union_by_layer.items():
                triton_tri_compact(layer_pools[lid], pl, kl, sl)
        for lid, (pl, sources, seq_lens, destinations) in swa_by_layer.items():
            triton_tri_compact(
                layer_pools[lid],
                pl,
                sources,
                seq_lens,
                dest_list=destinations,
            )

        capacity_targets = []
        for rid, evicted, keep_count in pending_updates:
            self._evicted[rid] = self._evicted.get(rid, 0) + evicted
            self._pre_forward_kv_lengths[rid] = keep_count
            # The current forward appends one token immediately after the kept prefix.
            capacity_targets.append((rid, keep_count + 1))
        return capacity_targets

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

        Returns ``None`` when V2 has no page metadata for the request. Negative
        ids (unallocated slots) are filtered out.
        """
        mgr = self.kv_cache_manager
        get_batch = getattr(mgr, "get_batch_cache_indices", None)
        if get_batch is not None:
            try:
                num_layers = self._num_layers_from_manager()
                if num_layers is None:
                    return None
                global_layer = self._global_layer_id(layer_idx, num_layers)
                batch = get_batch([request.py_request_id], global_layer)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resolve KV pages for local layer {layer_idx} "
                    f"of request {request.py_request_id}"
                ) from exc
            if batch:
                page_ids = [int(p) for p in batch[0] if int(p) >= 0]
                return page_ids or None
        return None

    def _num_layers_from_manager(self) -> Optional[int]:
        mgr = self.kv_cache_manager
        pp_layers = getattr(mgr, "pp_layers", None)
        if pp_layers is not None:
            return len(pp_layers)
        layer_offsets = getattr(mgr, "layer_offsets", None)
        if layer_offsets:
            return len(layer_offsets)
        return getattr(self, "_L", None)  # fall back to the calibrated layer count

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
