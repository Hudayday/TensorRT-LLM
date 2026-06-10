"""TriAttention sparse attention: periodic physical KV eviction.

Every ``beta`` generation steps TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the whole algorithm runs in one ``on_generation_step_end`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager`. It uses the standard
``KVCacheManagerV2`` and does not subclass it. The cache manager resets each
request's ``history_length`` to ``max_beam - 1`` every step, unaware of the
eviction, so the compacted-history reconcile runs in ``on_generation_step_end``
on every step. The compression manager is registered after the cache manager,
so this reconcile is the last word on the request's length.

It ships a small attention backend (``TriAttentionTrtllmAttention``) only to
reconcile ``num_cached`` after compaction; decode then runs the standard dense
kernel over the surviving tokens.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache (a self-consistent NHD round-trip passes an integrity probe, but the
kernel reads garbage). See ``_read_request_k`` / ``_evict_layer``.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The decode query rotates at its true absolute position
(``model_engine`` keeps the query position at ``max_beam - 1`` while decoupling
``num_cached = max_beam - 1 - evicted``), so a query at its true position
against a kept key at its original rotation still yields the correct relative
distance.

Calibration is computed offline by
``triattention_calibration.compute_triattention_calibration`` and loaded once at
init time. The scoring math follows the upstream reference
(github.com/WeianMao/triattention, ``methods/pruning_utils.py``).
"""

import os
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import torch

from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_manager import (
    BaseKVCacheCompressionManager,
)
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
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

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        top_B: int,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        calibration_cache_dir: Optional[str] = None,
        calib_dataset: str = "cnn_dailymail",
        calib_batches: int = 64,
        calib_max_seq_length: int = 2048,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        window_size: int = 128,
    ):
        super().__init__(kv_cache_manager)
        self.top_B = top_B
        self.beta = beta
        # Recency window: the most recent ``window_size`` tokens are ALWAYS kept
        # (upstream TRIATTN_RUNTIME_WINDOW_SIZE). Without it the trig scorer
        # evicts the model's freshly-generated tokens (they score low) and the
        # model degenerates on repeated eviction -- the single most important
        # correctness knob for multi-round eviction.
        self.window_size = int(os.environ.get("TRTLLM_TRI_WINDOW", str(window_size)))
        self.score_aggregation = score_aggregation

        # Calibration is resolved on the first request (on_request_init), not
        # here: it is model-intrinsic, so it is computed once and cached to a
        # config-keyed file, then reused for any later run with the same config.
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.calibration_cache_dir = calibration_cache_dir
        self.calib_dataset = calib_dataset
        self.calib_batches = calib_batches
        self.calib_max_seq_length = calib_max_seq_length
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

        Calibration is model-intrinsic, so it is computed once and cached: a
        config-keyed file is loaded if present, otherwise computed here and
        saved for later runs with the same config.
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
        self._calibrated = True

    # The framework drives all 8 lifecycle hooks; TriAttention overrides only
    # on_generation_step_end (periodic eviction) and on_request_finish (per-
    # request cleanup). It scores from offline calibration, not from live
    # queries or attention scores, so it needs no per-layer attention hook: the
    # whole eviction runs once per period in on_generation_step_end, which loops
    # the layers and reads each layer's keys straight from the KV pool.

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """After every generation step's full forward across all layers, bump a
        per-request step counter; every ``beta`` steps score the cache and
        physically evict down to top-B (per-layer, layer-uniform count)."""
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

        for request in gen_requests:
            rid = request.py_request_id
            step = self._gen_steps.get(rid, 0) + 1
            self._gen_steps[rid] = step
            if step % self.beta == 0:
                self._maybe_evict(request, rid, num_layers)
            # The cache manager's update_resources already reset this request's
            # history_length to max_beam-1 this iteration (it does not know about
            # eviction). Reconcile to the compacted length here -- every step for
            # any request with cumulative evictions, not only on the eviction
            # step, or the in-between steps leave history at the full length
            # while the cache content is compacted and the kernel reads a stale
            # tail. The compression manager runs after the cache manager, so this
            # reconcile is the last word.
            ev = self._evicted.get(rid, 0)
            if ev > 0:
                request.py_tri_evicted = ev
                kv_cache_map = getattr(mgr, "kv_cache_map", None)
                kv_cache = kv_cache_map.get(rid) if kv_cache_map is not None else None
                if kv_cache is not None and getattr(kv_cache, "is_active", False):
                    # history = max_beam - cum_evicted = num_cached + 1. Bypass the
                    # monotonic-decrease guard (kept tokens already gather-compacted
                    # to the front); capacity kept (block-free is a follow-up).
                    target_hist = request.max_beam_num_tokens - ev
                    if target_hist < kv_cache.history_length:
                        kv_cache._history_length = target_hist
                    kv_cache.resize(None, target_hist)

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request step + evicted counters."""
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)

    # ------------------------------------------------------------------ #
    # Public introspection (read by TriAttentionTrtllmAttentionMetadata) #
    # ------------------------------------------------------------------ #

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id`` (read by the
        metadata shim to reconcile num_cached after compaction)."""
        return self._evicted.get(request_id, 0)

    # ================================================================== #
    # Helpers (eviction / scoring / V2 cache access / calibration)       #
    # ================================================================== #

    def _maybe_evict(self, request: "LlmRequest", rid: int, num_layers: int) -> None:
        """Score + physically evict this request down to top-B (per-layer,
        layer-uniform); update the cumulative ``self._evicted[rid]`` and
        ``request.py_tri_evicted``. Called every ``beta`` steps."""
        # round_start = current query absolute position. max_beam_num_tokens
        # counts the just-generated (uncommitted) token; the cache holds
        # max_beam-1 committed tokens, so the latest committed position is
        # max_beam-1 (matches num_cached / position_ids derivation).
        round_start = request.max_beam_num_tokens - 1
        # FULL committed count (includes the just-generated token at slot
        # round_start). Using round_start (=count-1) stranded+lost that token.
        seq_len = request.max_beam_num_tokens - self._evicted.get(rid, 0)
        if seq_len <= self.top_B:
            return
        # Clamp to the COMMITTED extent: a just-written token's K may not be
        # flushed to the paged pool yet (reads all-zeros). Trim trailing
        # all-zero (uncommitted) slots so attention never sees a zero-K row.
        _k0 = self._read_request_k(request, 0, seq_len)
        if _k0 is not None:
            _nz = _k0.abs().sum(dim=(0, 2)) > 0
            if bool(_nz.any()):
                _committed = int(_nz.nonzero().max()) + 1
                if _committed < seq_len:
                    seq_len = _committed
        if seq_len <= self.top_B:
            return
        # PER-LAYER eviction: each layer scores its OWN cached K_rot, aggregates
        # heads by mean, keeps its own budget + recency window, compacted
        # INDEPENDENTLY. Every layer keeps the SAME COUNT (top_B), so the
        # per-request num_cached is consistent across layers; only the kept SET
        # differs. Kept K retains its original RoPE rotation.
        keep_count = None
        for layer_idx in range(num_layers):
            k_rot = self._read_request_k(request, layer_idx, seq_len)
            if k_rot is None:
                continue
            head_scores = self._score_layer(k_rot, None, layer_idx, round_start)
            if head_scores is None:
                continue
            layer_score = head_scores.mean(dim=0)  # [seq] mean over heads
            keep = self._select_with_recency(layer_score, seq_len)
            self._evict_layer(request, layer_idx, keep, seq_len)
            keep_count = int(keep.numel())
        if keep_count is None:
            return
        evicted = seq_len - keep_count
        if evicted > 0:
            # cumulative count consumed by the every-step reconcile (history
            # shrink) above + the metadata shim (num_cached clamp). seq_len is
            # the full committed length, so evicted_cum is exact.
            self._evicted[rid] = self._evicted.get(rid, 0) + evicted
            request.py_tri_evicted = self._evicted[rid]

    # ------------------------------------------------------------------ #
    # Selection + scoring                                                #
    # ------------------------------------------------------------------ #

    def _select_with_recency(self, scores: "torch.Tensor", seq_len: int) -> "torch.Tensor":
        """Decide which token slots THIS layer keeps, given per-token scores.

        The keep budget is ``top_B`` tokens, split into two parts:

          1. RECENCY window -- the most-recent ``window_size`` slots are ALWAYS
             kept, regardless of score. The trigonometric importance score
             systematically UNDER-rates freshly generated tokens, so without
             this guarantee the model would evict its own recent output and
             degenerate over repeated eviction rounds. (This is the single most
             important correctness knob for multi-round eviction.)

          2. TOP-K of the rest -- the remaining budget (``top_B - window_size``)
             is spent on the highest-scoring tokens in the OLDER region. We use
             top-k (not a threshold) because the budget is a fixed token count:
             we keep exactly the K most "important" older tokens and drop the
             rest. ``torch.topk`` returns those K indices.

        Args:
            scores:  per-token importance for one layer, shape ``[seq_len]``
                     (already aggregated over heads in ``on_generation_step_end``).
            seq_len: number of valid tokens currently cached for this request.

        Returns:
            kept slot indices in ``[0, seq_len)``, SORTED ascending (the kernel
            expects ascending order). Length == ``min(top_B, seq_len)``.
        """
        device = scores.device
        keep_count = min(self.top_B, seq_len)

        # Budget covers everything -> keep all tokens (this layer evicts nothing).
        if keep_count >= seq_len:
            return torch.arange(seq_len, device=device, dtype=torch.long)

        recency_window = min(self.window_size, seq_len)

        # Budget is no bigger than the recency window -> no room to score older
        # tokens; just keep the most-recent ``keep_count`` slots.
        if keep_count <= recency_window:
            return torch.arange(seq_len - keep_count, seq_len, device=device, dtype=torch.long)

        # (1) the last ``recency_window`` slots, always kept.
        recent_indices = torch.arange(
            seq_len - recency_window, seq_len, device=device, dtype=torch.long
        )
        # (2) top-K highest-scoring tokens from the OLDER region (everything
        #     before the recency window), using the leftover budget.
        older_budget = keep_count - recency_window
        older_scores = scores[: seq_len - recency_window]
        older_keep = torch.topk(older_scores, older_budget).indices.to(torch.long)

        # Merge the two kept sets and return ascending (kernel requirement).
        return torch.sort(torch.cat([older_keep, recent_indices])).values

    def _score_layer(
        self,
        cached_k: torch.Tensor,
        key_positions: Optional[torch.Tensor],  # unused -- see note below
        layer_idx: int,
        round_start: int,
    ) -> Optional[torch.Tensor]:
        """Compute a per-token IMPORTANCE score for one layer's cached keys.

        We don't have the upcoming query vectors at eviction time, so instead of
        the usual ``q . kᵀ`` we approximate the expected attention each key will
        receive using OFFLINE-CALIBRATED statistics of the query distribution:
          * ``E_q``      -- mean of the query's complex per-frequency form  [H, F]
          * ``E_q_norm`` -- mean of the query's magnitude                   [H, F]
        (port of the upstream vLLM ``compute_scores_pytorch``.)

        K is used EXACTLY as stored in the cache (post-RoPE). The token's
        absolute position is already baked into K's RoPE rotation, so there is
        NO RoPE inversion and NO per-token position input. ``round_start`` is the
        single current query position; ``key_positions`` is unused (kept only for
        signature symmetry with other scorers).

        Shapes use: H = num query heads, nkv = num KV heads, F = head_dim/2
        (RoPE pairs the head dim into F complex frequencies). Vectorized over heads.

        Args:
            cached_k:  this layer's cached keys, ``[num_kv_heads, seq_len, head_dim]``.
            layer_idx: selects this layer's calibration stats.
            round_start: current query's absolute position.
        Returns:
            per-(query-head, token) scores, ``[num_q_heads, seq_len]``.
        """
        device = self.calibration["E_q"].device
        k = cached_k.to(device=device, dtype=torch.float32)  # [num_kv_heads, seq, head_dim]
        num_kv_heads, seq_len, head_dim = k.shape
        num_freqs = head_dim // 2

        # "half" RoPE layout: first half of head_dim = REAL part, second = IMAG part.
        k_real, k_imag = (
            k[..., :num_freqs],
            k[..., num_freqs:],
        )  # each [num_kv_heads, seq, num_freqs]

        # Per-layer calibration stats (precomputed offline over a corpus):
        q_mean_complex = self.calibration["E_q"][layer_idx].to(device)  # [H, F] complex: mean query
        q_mean_norm = self.calibration["E_q_norm"][layer_idx].to(
            device, torch.float32
        )  # [H, F]: mean |query|
        rope_inv_freq = self.calibration["omega"].to(
            device, torch.float32
        )  # [F]: RoPE inverse frequencies
        freq_scale_sq = self._freq_scale_sq.to(
            device, torch.float32
        )  # [F]: per-freq RoPE amplitude^2
        num_q_heads = self._H

        # GQA: several query heads share one KV head. Map each query head to its
        # KV head, then gather that KV head's keys so everything below is
        # per-QUERY-head.
        q_head_ids = torch.arange(num_q_heads, device=device)
        qhead_to_kvhead = torch.clamp(
            q_head_ids * num_kv_heads // max(1, num_q_heads), max=num_kv_heads - 1
        )  # [H]
        k_real_q = k_real[qhead_to_kvhead]  # [H, seq, F]
        k_imag_q = k_imag[qhead_to_kvhead]
        q_real = q_mean_complex.real.unsqueeze(1)  # [H, 1, F]
        q_imag = q_mean_complex.imag.unsqueeze(1)

        # Complex product  Q . conj(K)  per (query-head, token, freq):
        #   real = q_re*k_re + q_im*k_im ;  imag = q_im*k_re - q_re*k_im
        prod_real = q_real * k_real_q + q_imag * k_imag_q  # [H, seq, F]
        prod_imag = q_imag * k_real_q - q_real * k_imag_q

        # ---- position-dependent term ----
        # Rotate the product by the (query - key) relative distance. We don't
        # know the exact future query position, so we average over a GEOMETRIC
        # set of look-ahead offsets [1,2,4,...] -- a cheap proxy for "the next
        # several decode steps". O = num offsets.
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        offsets = self._offsets.to(device, torch.float32)  # [O]
        query_positions = (float(round_start) + offsets).view(-1, 1, 1, 1)  # [O,1,1,1]
        phase = query_positions * rope_inv_freq.view(1, 1, 1, -1)  # [O,1,1,F]
        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
        # real part of (prod * e^{i*phase}), scaled per frequency:
        position_term = freq_scale_sq.view(1, 1, 1, -1) * (
            prod_real.unsqueeze(0) * cos_phase - prod_imag.unsqueeze(0) * sin_phase
        )  # [O, H, seq, F]
        score_per_offset = position_term.sum(dim=-1)  # sum over freqs -> [O, H, seq]
        if self.score_aggregation == "max":
            scores = score_per_offset.max(dim=0).values  # [H, seq]
        else:
            scores = score_per_offset.mean(dim=0)  # average the look-ahead offsets

        # ---- position-INDEPENDENT term (the MLR correction) ----
        # (mean|q| - |mean q|) * |k| * freq_scale, summed over freqs. Captures the
        # part of the expected attention that doesn't depend on relative position.
        k_magnitude = torch.sqrt(k_real_q**2 + k_imag_q**2)  # [H, seq, F]
        q_mean_magnitude = q_mean_complex.abs()  # [H, F]
        mlr_coef = (q_mean_norm - q_mean_magnitude).unsqueeze(1)  # [H, 1, F]
        mlr_term = (k_magnitude * mlr_coef * freq_scale_sq.view(1, 1, -1)).sum(dim=-1)  # [H, seq]

        return scores + mlr_term  # [H, seq]

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

    def _evict_layer(
        self,
        request: "LlmRequest",
        layer_idx: int,
        keep: "torch.Tensor",
        seq_len: int,
    ) -> None:
        """Physically compact ONE layer's cache in place: move the kept tokens
        to the front contiguous slots of this request's pages.

        ``keep`` is a SORTED list of slot indices in ``[0, seq_len)`` to retain.
        We build the FULL permutation ``[kept..., dropped...]`` and apply it to
        the (page, slot) token axis, so every slot in ``[0, seq_len)`` still holds
        a real key/value afterwards (the kept ones up front, the dropped ones
        after -- nothing is left stale). We do NOT re-apply RoPE: the kept keys
        keep their original rotation. The decode query rotates at its true
        absolute position, which gives the correct relative distance against
        those kept keys, so no re-rotation is needed.

        WHY HND (and why this was subtle): ``get_buffers`` is a VIEW over the raw
        pool bytes. The kernel stores HND
        ``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``, so
        the reorder MUST be done on the HND axes. An earlier version reordered on
        the NHD view, which transposed the token and head axes and silently
        scrambled the cache (every probe looked fine; the kernel read garbage).
        Both K and V are moved together (we reorder the whole kv_factor axis).
        """
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        keep = keep.to(dtype=torch.long)
        keep_count = int(keep.numel())
        if keep_count >= seq_len:
            return  # nothing to drop
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return
        # HND view: [num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]
        pool = get_buffers(layer_idx, kv_layout="HND")
        if pool is None:
            return
        tokens_per_block = pool.shape[3]  # HND: dim 3 = slots per page
        page_ids_t = torch.as_tensor(page_ids, device=pool.device, dtype=torch.long)
        # Advanced indexing returns a COPY of this request's pages:
        #   [num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]
        request_pages = pool[page_ids_t]
        num_pages, kv_factor, num_kv_heads, _, head_dim = request_pages.shape
        keep = keep.to(request_pages.device)
        # Bring the two NON-token axes (kv_factor, num_kv_heads) to the front and
        # merge (page, slot) into a single token axis we can reorder (dim 2):
        #   [kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim]
        kv_by_token = (
            request_pages.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim)
        )
        # Full reorder of the token axis: kept slots first (in their given
        # order), then the dropped slots. Writing BOTH halves means no slot in
        # [0, seq_len) is left holding stale data.
        all_token_ids = torch.arange(seq_len, device=kv_by_token.device, dtype=torch.long)
        is_dropped = torch.ones(seq_len, device=kv_by_token.device, dtype=torch.bool)
        is_dropped[keep] = False
        new_order = torch.cat([keep, all_token_ids[is_dropped]])  # [seq_len]
        reordered = kv_by_token[:, :, :seq_len].index_select(2, new_order).clone()
        kv_by_token[:, :, :seq_len] = reordered
        # Reshape back to paged HND and write the touched pages into the live pool.
        num_touched_pages = (seq_len + tokens_per_block - 1) // tokens_per_block
        repaged = (
            kv_by_token.reshape(kv_factor, num_kv_heads, num_pages, tokens_per_block, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        pool[page_ids_t[:num_touched_pages]] = repaged[:num_touched_pages]

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
        """Return the calibration stats: an explicit file if given, else the
        config-keyed cache file (computed on a miss)."""
        if self.calibration_path is not None:
            return self._load_calibration(self.calibration_path)
        cache_file = self._cache_file()
        if not os.path.exists(cache_file):
            # Model-intrinsic stats: computed once per (model, calib config),
            # then reused by every later run.
            self._compute_calibration(cache_file)
        return self._load_calibration(cache_file)

    def _cache_file(self) -> str:
        """Config-keyed calibration cache path (model + calib settings)."""
        if self.model_path is None:
            raise ValueError(
                "TriAttention needs model_path to compute calibration on first "
                "use; pass calibration_path to use a precomputed file instead."
            )
        cache_dir = self.calibration_cache_dir or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
            "triattn_calib",
        )
        os.makedirs(cache_dir, exist_ok=True)
        model_tag = os.path.basename(os.path.normpath(self.model_path)) or "model"
        key = f"{self.calib_dataset}_{self.calib_batches}_{self.calib_max_seq_length}"
        return os.path.join(cache_dir, f"triattn_{model_tag}_{key}.pt")

    def _compute_calibration(self, out_path: str) -> None:
        """Compute calibration stats and save them to ``out_path``.

        Loads a throwaway HF copy of the model and runs the q_proj
        forward-hook harness over the calibration corpus. The serving model is
        already resident, so this transiently holds a second model copy.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tensorrt_llm._torch.attention_backend.sparse.triattention_calibration import (
            compute_triattention_calibration,
        )
        from tensorrt_llm.llmapi.llm_args import CalibConfig

        logger.info(f"TriAttention: computing calibration -> {out_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        try:
            calib_config = CalibConfig(
                calib_dataset=self.calib_dataset,
                calib_batches=self.calib_batches,
                calib_max_seq_length=self.calib_max_seq_length,
            )
            compute_triattention_calibration(model, tokenizer, calib_config, out_path)
        finally:
            del model
            torch.cuda.empty_cache()

    def _load_calibration(self, path: str) -> Dict[str, torch.Tensor]:
        """Load calibration ``.pt`` onto GPU."""
        calibration = torch.load(path, map_location="cuda")
        self._validate_calibration(calibration)
        return calibration

    def _validate_calibration(self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys."""
        missing = _REQUIRED_CALIBRATION_KEYS - set(calibration.keys())
        if missing:
            raise ValueError(
                f"TriAttention calibration is missing keys: {sorted(missing)}; "
                f"got {sorted(calibration.keys())}."
            )


# TRT-LLM attention shim for TriAttention. TriAttention runs the standard dense
# attention kernel over the compacted cache; the only shim work is reconciling
# ``num_cached_tokens_per_seq`` after physical eviction.
from tensorrt_llm._torch.attention_backend.trtllm import (  # noqa: E402
    TrtllmAttention,
    TrtllmAttentionMetadata,
)


class TriAttentionTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Metadata shim: reconcile num_cached after TriAttention compaction.

    The model engine derives ``num_cached_tokens_per_seq`` from the request's
    logical length (``max_beam_num_tokens - 1 - py_tri_evicted``), and history =
    num_cached + 1 (the current token's slot). ``prepare`` bumps num_cached by
    +1 to match the cache manager's compacted history, then clamps each evicted
    gen request's ``prompt_lens`` down to num_cached: a prompt_len longer than
    the whole compacted cache desyncs the prompt/gen split. ``position_ids`` are
    already baked from the full logical length, so the query rotates at its true
    absolute position (kept keys keep their original rotation)."""

    @property
    def _tri_manager(self) -> Optional["TriAttention"]:
        # The framework sets ``compression_manager`` on the attention metadata
        # (interface.py); read the TriAttention manager from there.
        cm = getattr(self, "compression_manager", None)
        return cm if isinstance(cm, TriAttention) else None

    def prepare(self) -> None:
        e = self._tri_manager
        kvp = getattr(self, "kv_cache_params", None)
        if (
            e is not None
            and kvp is not None
            and getattr(kvp, "num_cached_tokens_per_seq", None) is not None
        ):
            num_contexts = self.num_contexts
            num_requests = num_contexts + self.num_generations
            req_ids = self.request_ids
            for i in range(num_contexts, num_requests):
                ev = e.evicted_count(req_ids[i])
                if ev:
                    # history = num_cached + 1 (current token slot). model_engine
                    # decoupled num_cached = max_beam-1 - py_tri_evicted; bump +1
                    # to match the cache manager history (max_beam - py_tri_evicted)
                    # so the new token (written at slot num_cached) is not
                    # overwritten / skipped.
                    kvp.num_cached_tokens_per_seq[i] = int(kvp.num_cached_tokens_per_seq[i]) + 1
        super().prepare()
        # Clamp gen-request prompt_lens to the compacted cache length. After
        # eviction num_cached is compressed but prompt_lens still holds the
        # original prompt length; prompt_lens > num_cached makes the kernel's
        # prompt/gen split and cache offsets inconsistent and the output garbled.
        if (
            e is not None
            and kvp is not None
            and getattr(kvp, "num_cached_tokens_per_seq", None) is not None
            and hasattr(self, "prompt_lens")
            and self.prompt_lens is not None
        ):
            _pl = list(self.prompt_lens)
            _changed = False
            for i in range(num_contexts, num_requests):
                ev = e.evicted_count(req_ids[i])
                if ev:
                    nc = int(kvp.num_cached_tokens_per_seq[i])
                    if int(_pl[i]) > nc:
                        _pl[i] = nc
                        _changed = True
            if _changed:
                _t = torch.tensor(_pl, dtype=torch.int, device="cpu")
                self.prompt_lens_cpu[: self.num_seqs].copy_(_t[: self.num_seqs])
                self.prompt_lens_cuda[: self.num_seqs].copy_(
                    self.prompt_lens_cpu[: self.num_seqs], non_blocking=True
                )


class TriAttentionTrtllmAttention(TrtllmAttention):
    """Base TRT-LLM attention carrying the TriAttention reconciliation Metadata.

    TriAttention physically evicts tokens (the cache manager gather-compacts the
    kept tokens and shrinks history; the metadata clamps num_cached). Decode then
    runs dense attention over exactly the surviving num_cached tokens -- there is
    no decode-time sparse mask, so both sparse predictors return (None, None)."""

    Metadata: ClassVar[type] = None  # set below

    def sparse_kv_predict(self, q, k, metadata, forward_args):
        return None, None

    def sparse_attn_predict(self, q, k, metadata, forward_args):
        return None, None


TriAttentionTrtllmAttention.Metadata = TriAttentionTrtllmAttentionMetadata
