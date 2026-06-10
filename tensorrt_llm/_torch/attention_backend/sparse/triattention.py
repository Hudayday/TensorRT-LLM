"""TriAttention sparse-attention method (periodic physical KV eviction).

TriAttention performs periodic generation-phase KV eviction guided by a
trigonometric importance score computed from offline-calibrated statistics of
the model's Q-pre-RoPE vectors (paper §4.1, §4.3). It physically deletes
tokens from the cache: no work in context phase, no per-step attention-time
mask, and a single ``on_generation_step_end`` hook that fires every ``beta``
steps to evict tokens below the top-B keep set.

Behavior-layer integration (kv-cache-compression framework):
  - ``TriAttention`` is a :class:`SparseAttentionManager` registered as the
    ``KV_CACHE_COMPRESSION_MANAGER`` resource manager; PyExecutor's main loop
    drives its ``on_generation_step_end`` each iteration. Eviction sets
    ``request.py_tri_evicted``.
  - Pattern 1 (no V2 subclass): uses the plain ``KVCacheManagerV2``. The
    compacted-history reconcile lives in ``on_generation_step_end`` itself --
    every step, for any request with cumulative evictions, it shrinks
    history_length to ``max_beam - py_tri_evicted`` directly on the plain
    manager's ``kv_cache_map`` entry. KV_CACHE_MANAGER is ordered BEFORE this
    compression manager (framework default), so the plain manager's per-step
    reset of history to max_beam-1 happens first and our reconcile is the last
    word. (``kv_cache_manager_class = KVCacheManagerV2`` is only an isinstance
    sanity check.)
  - ``TriAttentionTrtllmAttention`` (custom backend, kept because physical
    eviction needs ``num_cached`` reconciliation) carries
    ``TriAttentionTrtllmAttentionMetadata``, which clamps
    ``num_cached_tokens_per_seq`` (+ ``prompt_lens``) after compaction.

Root-cause note (decode-garble bug, fixed): the trtllm-gen / XQA decode kernel
reads the physical KV pool in **HND** layout
(``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``). The
Python gather/score MUST reinterpret ``get_buffers`` with ``kv_layout="HND"``;
an earlier ``"NHD"`` reorder silently transposed the tokens<->heads axes and
scrambled the cache (a self-consistent NHD round-trip passed an integrity probe
but the kernel saw garbage). See ``_evict_layer`` / ``_read_request_k``.

Position handling (design choice A): kept keys retain their ORIGINAL RoPE
rotation (no re-RoPE on compaction); the decode query rotates at its TRUE
absolute position (``model_engine`` keeps ``position_id = max_beam-1`` while
decoupling ``num_cached = max_beam-1 - py_tri_evicted``), so query@true-pos vs
kept-key@original-rotation yields the correct relative distance.

Calibration is computed offline via
``triattention_calibration.compute_triattention_calibration`` and loaded once
at LLM init time.

Upstream-GitHub provenance (authoritative over paper §4.3 text):
  - scoring math: ``methods/pruning_utils.py`` (Q*conj(K_rot), geometric
    offsets, MLR additive term).
  - per-(layer) mean-over-heads top-B + recency window: the validated default.
"""

import os
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import torch

from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_manager import (
    SparseAttentionManager,
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


class TriAttention(SparseAttentionManager):
    """Periodic physical KV eviction driven by trigonometric importance
    scoring (behavior-layer compression manager).

    Overrides ``on_generation_step_end``: every ``beta`` generation steps,
    reads the current K cache through the underlying ``KVCacheManagerV2``,
    computes a per-token importance score using offline-calibrated stats, and
    physically evicts tokens below the top-B keep set. Each layer scores its
    OWN cached K and keeps the SAME COUNT (top_B) but its own kept SET
    (per-layer compaction), so the kernel's per-request num_cached is
    consistent across layers.
    """

    # Per-request decode-time eviction => the compacted cache is not safe to
    # reuse across requests.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Physically deletes tokens from cache (vs RocketKV-style sparse mask).
    physically_evicts_kv: ClassVar[bool] = True

    # TriAttention ships its own attention shim + Metadata (num_cached
    # reconciliation after physical eviction), so the framework must NOT null
    # its sparse config in get_attention_backend (see get_attention_backend's
    # ``ships_attention_backend`` check).
    ships_attention_backend: ClassVar[bool] = True

    # Pattern 3: the resize-only V2 subclass that shrinks history_length to the
    # compacted length. Set below the class body (forward reference).
    kv_cache_manager_class: ClassVar[Optional[type]] = None

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        top_B: int,
        beta: int = 128,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        window_size: int = 128,
    ):
        # Validate own config BEFORE delegating to the base __init__ (which
        # asserts the Pattern-3 cache-manager subclass via isinstance).
        if calibration_path is None:
            raise ValueError(
                "TriAttention requires calibration_path; compute offline via "
                "triattention_calibration.compute_triattention_calibration "
                "before LLM init."
            )
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
        self.calibration: Dict[str, torch.Tensor] = self._load_calibration(calibration_path)

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Per-request generation-step counter; eviction fires when it hits
        # ``beta``. Cleared on request finish.
        self._gen_steps: Dict[int, int] = {}
        # Cumulative physically-evicted token count per request, consumed by
        # the metadata shim + the V2 subclass resize.
        self._evicted: Dict[int, int] = {}

        # Calibration-derived dims + stats.
        self._L = int(self.calibration["E_q"].shape[0])
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        self._attention_scale = float(self.calibration.get("attention_scale", 1.0))

    # ------------------------------------------------------------------ #
    # Override: per-step end                                             #
    # ------------------------------------------------------------------ #

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """After every generation step's full forward across all layers, bump a
        per-request step counter; every ``beta`` steps score the cache and
        physically evict down to top-B (per-layer, layer-uniform count)."""
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
            # Pattern 1 (no V2 subclass): the plain KVCacheManagerV2's
            # update_resources already reset this request's history_length to
            # max_beam-1 THIS iteration (it doesn't know about eviction). We own
            # the compacted history, so reconcile it here -- EVERY step for any
            # request with cumulative evictions, not only on the eviction step:
            # the in-between steps would otherwise leave history at the full
            # length while the cache content is compacted -> kernel reads a stale
            # tail. The compression manager is ordered AFTER KV_CACHE_MANAGER
            # (framework default), so this reconcile is the last word.
            ev = self._evicted.get(rid, 0)
            if ev > 0:
                request.py_tri_evicted = ev
                kv_cache_map = getattr(mgr, "kv_cache_map", None)
                kv_cache = (kv_cache_map.get(rid)
                            if kv_cache_map is not None else None)
                if kv_cache is not None and getattr(kv_cache, "is_active", False):
                    # history = max_beam - cum_evicted = num_cached + 1. Bypass the
                    # monotonic-decrease guard (kept tokens already gather-compacted
                    # to the front); capacity kept (block-free is a follow-up).
                    target_hist = request.max_beam_num_tokens - ev
                    if target_hist < kv_cache.history_length:
                        kv_cache._history_length = target_hist
                    kv_cache.resize(None, target_hist)

    def _maybe_evict(self, request: "LlmRequest", rid: int,
                     num_layers: int) -> None:
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
            _nz = (_k0.abs().sum(dim=(0, 2)) > 0)
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

    def _select_with_recency(self, score: "torch.Tensor", seq_len: int) -> "torch.Tensor":
        """Select keep indices for ONE layer: always retain the most recent
        ``window_size`` slots (the just-generated tokens the trig scorer alone
        would wrongly evict) and fill the remaining budget with the
        highest-scoring OLDER tokens (topk). Returns SORTED long indices."""
        device = score.device
        keep_count = min(self.top_B, seq_len)
        if keep_count >= seq_len:
            return torch.arange(seq_len, device=device, dtype=torch.long)
        window = min(self.window_size, seq_len)
        if keep_count <= window:
            return torch.arange(
                seq_len - keep_count, seq_len, device=device, dtype=torch.long
            )
        recent = torch.arange(seq_len - window, seq_len, device=device, dtype=torch.long)
        older_budget = keep_count - window
        older_scores = score[: seq_len - window]
        older_keep = torch.topk(older_scores, older_budget).indices.to(torch.long)
        return torch.sort(torch.cat([older_keep, recent])).values

    def _score_layer(
        self,
        k_layer: torch.Tensor,
        key_positions: Optional[torch.Tensor],
        layer_idx: int,
        round_start: int,
    ) -> Optional[torch.Tensor]:
        """Official DIRECT Q*conj(K_rot) scoring (port of vllm
        compute_scores_pytorch). K is used AS CACHED (post-RoPE): the token's
        position lives in K's RoPE rotation, so there is NO inversion and NO
        per-token position input (``round_start`` is the scalar current query
        position; ``key_positions`` is ignored). Vectorized over heads.
        Returns ``[num_q_heads, seq_len]`` per-head scores."""
        device = self.calibration["E_q"].device
        k = k_layer.to(device=device, dtype=torch.float32)  # [nkv, seq, hd]
        nkv, seq, hd = k.shape
        F = hd // 2
        # "half" RoPE layout: first F = real, last F = imag.
        k_real, k_imag = k[..., :F], k[..., F:]  # [nkv, seq, F]
        E_q = self.calibration["E_q"][layer_idx].to(device)  # [H, F] complex
        E_q_norm = self.calibration["E_q_norm"][layer_idx].to(
            device=device, dtype=torch.float32)  # [H, F]
        omega = self.calibration["omega"].to(device=device, dtype=torch.float32)  # [F]
        fss = self._freq_scale_sq.to(device=device, dtype=torch.float32)  # [F]
        H = self._H
        # GQA: map each (calibrated) Q head onto its KV head, expand K per Q head.
        idx = torch.arange(H, device=device)
        q2kv = torch.clamp(idx * nkv // max(1, H), max=nkv - 1)  # [H]
        kr = k_real[q2kv]  # [H, seq, F]
        ki = k_imag[q2kv]
        qr = E_q.real.unsqueeze(1)  # [H, 1, F]
        qi = E_q.imag.unsqueeze(1)
        # prod = Q * conj(K_rot)
        prod_real = qr * kr + qi * ki  # [H, seq, F]
        prod_imag = qi * kr - qr * ki
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        offs = self._offsets.to(device=device, dtype=torch.float32)  # [O]
        t = (float(round_start) + offs).view(-1, 1, 1, 1)  # [O,1,1,1]
        phase = t * omega.view(1, 1, 1, -1)  # [O,1,1,F]
        cos_v, sin_v = torch.cos(phase), torch.sin(phase)
        # position_term = freq_scale * (prod_real*cos - prod_imag*sin)
        pt = fss.view(1, 1, 1, -1) * (
            prod_real.unsqueeze(0) * cos_v - prod_imag.unsqueeze(0) * sin_v)  # [O,H,seq,F]
        score_off = pt.sum(dim=-1)  # [O, H, seq]
        if self.score_aggregation == "max":
            scores = score_off.max(dim=0).values  # [H, seq]
        else:
            scores = score_off.mean(dim=0)
        # position-independent MLR term: (q_abs_mean - |q_mean|) * |K| * freq_scale
        k_abs = torch.sqrt(kr ** 2 + ki ** 2)  # [H, seq, F]
        q_mean_abs = E_q.abs()  # [H, F]
        extra_coef = (E_q_norm - q_mean_abs).unsqueeze(1)  # [H,1,F]
        extra = (k_abs * extra_coef * fss.view(1, 1, -1)).sum(dim=-1)  # [H, seq]
        return scores + extra  # [H, seq]

    # ------------------------------------------------------------------ #
    # V2-manager cache access + physical eviction (HND physical layout)  #
    # ------------------------------------------------------------------ #

    def _resolve_page_ids(self, request: "LlmRequest", layer_idx: int) -> Optional[List[int]]:
        """Resolve this request's paged-cache block indices via V2's public
        ``get_batch_cache_indices(ids, layer_idx) -> List[List[int]]``, falling
        back to V1 ``get_cache_indices(request)``; ``None`` when neither exists
        (mocked tests). Indices already divide by kv_factor."""
        mgr = self.kv_cache_manager
        get_batch = getattr(mgr, "get_batch_cache_indices", None)
        if get_batch is not None:
            try:
                batch = get_batch([request.py_request_id], layer_idx)
            except Exception:
                batch = None
            if batch:
                ids = [int(p) for p in batch[0] if int(p) >= 0]
                return ids or None
        get_single = getattr(mgr, "get_cache_indices", None)
        if get_single is not None:
            try:
                ids = get_single(request)
            except Exception:
                ids = None
            if ids:
                return [int(p) for p in ids if int(p) >= 0]
        return None

    def _read_request_k(
        self, request: "LlmRequest", layer_idx: int, seq_len: int
    ) -> Optional[torch.Tensor]:
        """Read this request's K tokens for ``layer_idx`` out of the V2 paged
        pool in HND layout. Returns ``[num_kv_heads, seq_len, head_dim]`` (K
        only) or ``None`` when the manager doesn't expose a readable pool."""
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return None
        pool = get_buffers(layer_idx, kv_layout="HND")  # [np, kv, nkv, tpb, hd]
        if pool is None:
            return None
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return None
        tpb = pool.shape[3]
        k_index = 0  # KEY is index 0 along kv_factor; VALUE is 1.
        pages = pool[page_ids][:, k_index]  # [num_pages, num_kv_heads, tpb, hd] (HND)
        num_pages = pages.shape[0]
        nkv = pages.shape[1]
        # HND token axis = (np, tpb) with nkv first -> [nkv, np, tpb, hd] -> [nkv, np*tpb, hd]
        flat = pages.permute(1, 0, 2, 3).reshape(nkv, num_pages * tpb, pages.shape[3])
        flat = flat[:, :seq_len, :]
        return flat.contiguous()  # [num_kv_heads, seq_len, head_dim]

    def _evict_layer(
        self,
        request: "LlmRequest",
        layer_idx: int,
        keep: "torch.Tensor",
        seq_len: int,
    ) -> None:
        """Compact ONE layer's cache in HND layout: gather kept tokens to the
        front contiguous slots, NO re-RoPE (cached post-RoPE K used as-is;
        design-A). ``keep`` is SORTED long indices in [0, seq_len).

        HND physical layout is ``[num_pages, kv_factor, num_kv_heads,
        tokens_per_block, head_dim]`` (confirmed: trtllm-gen/XQA reads HND). The
        reorder MUST happen on the HND axes; an earlier NHD reorder transposed
        tokens<->heads and silently scrambled the cache."""
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        keep = keep.to(dtype=torch.long)
        keep_count = int(keep.numel())
        if keep_count >= seq_len:
            return
        layer_page_ids = self._resolve_page_ids(request, layer_idx)
        if not layer_page_ids:
            return
        pool = get_buffers(layer_idx, kv_layout="HND")  # [np, kv, num_kv_heads, tpb, hd]
        if pool is None:
            return
        tpb = pool.shape[3]  # HND: tokens_per_block is dim 3 (nkv is dim 2)
        page_idx_t = torch.as_tensor(layer_page_ids, device=pool.device, dtype=torch.long)
        # COPY of this request's pages (HND): [num_pages, kv_factor, nkv, tpb, hd].
        req_pages = pool[page_idx_t]
        num_pages, kv_factor, nkv, _, hd = req_pages.shape
        keep_dev = keep.to(req_pages.device)
        # Permute to [kv, nkv, np, tpb, hd] then flatten (np, tpb) so the token
        # axis (np*tpb) is one contiguous dim (dim 2).
        perm = (
            req_pages.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(kv_factor, nkv, num_pages * tpb, hd)
        )
        # vLLM compact_request_kv_in_place: FULL permutation [kept..., dropped...],
        # NO re-RoPE, NO stale tail. Reorder the HND token axis (dim 2).
        all_tok = torch.arange(seq_len, device=perm.device, dtype=torch.long)
        drop_mask = torch.ones(seq_len, device=perm.device, dtype=torch.bool)
        drop_mask[keep_dev] = False
        perm_order = torch.cat([keep_dev, all_tok[drop_mask]])  # [seq_len]
        reordered = perm[:, :, :seq_len].index_select(2, perm_order).clone()
        perm[:, :, :seq_len] = reordered
        num_seq_pages = (seq_len + tpb - 1) // tpb
        back = (
            perm.reshape(kv_factor, nkv, num_pages, tpb, hd).permute(2, 0, 1, 3, 4).contiguous()
        )
        pool[page_idx_t[:num_seq_pages]] = back[:num_seq_pages]

    def _num_layers_from_manager(self) -> int:
        mgr = self.kv_cache_manager
        layer_offsets = getattr(mgr, "layer_offsets", None)
        if layer_offsets:
            return len(layer_offsets)
        return self._L  # fall back to the calibrated layer count

    # ------------------------------------------------------------------ #
    # Request lifecycle + introspection                                  #
    # ------------------------------------------------------------------ #

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Drop this request's per-request step + evicted counters."""
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id`` (read by the
        metadata shim to reconcile num_cached after compaction)."""
        return self._evicted.get(request_id, 0)

    # ------------------------------------------------------------------ #
    # Calibration loading                                                #
    # ------------------------------------------------------------------ #

    def _load_calibration(self, path: str) -> Dict[str, torch.Tensor]:
        """Load offline-computed calibration ``.pt`` onto GPU."""
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


# ====================================================================== #
# TRT-LLM attention shim for TriAttention                                #
#                                                                        #
# TriAttention uses the base dense attention kernel over the COMPACTED   #
# cache (unlike RocketKV which emits a sparse mask). The only shim work  #
# is reconciling ``num_cached_tokens_per_seq`` after physical eviction.  #
# ====================================================================== #
from tensorrt_llm._torch.attention_backend.trtllm import (  # noqa: E402
    TrtllmAttention,
    TrtllmAttentionMetadata,
)


class TriAttentionTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Metadata shim: reconcile num_cached after TriAttention compaction.

    The model engine derives ``num_cached_tokens_per_seq`` from the request's
    logical length (``max_beam_num_tokens - 1 - py_tri_evicted``), and history =
    num_cached + 1 (the current token's slot). ``prepare`` bumps num_cached by
    +1 to match the V2 manager's compacted history, then clamps each evicted
    gen request's ``prompt_lens`` down to num_cached (mirrors RocketKV: a
    prompt_len longer than the whole compacted cache desyncs the prompt/gen
    split). ``position_ids`` are already baked from the full logical length, so
    the query rotates at its true absolute position (design choice A)."""

    @property
    def _tri_executor(self) -> Optional["TriAttention"]:
        # Behavior-layer: the framework sets ``compression_manager`` on the
        # attention metadata (interface.py); read the TriAttention from there.
        cm = getattr(self, "compression_manager", None)
        return cm if isinstance(cm, TriAttention) else None

    def prepare(self) -> None:
        e = self._tri_executor
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
                    # to match the V2 manager history (max_beam - py_tri_evicted)
                    # so the new token (written at slot num_cached) is not
                    # overwritten / skipped.
                    kvp.num_cached_tokens_per_seq[i] = int(kvp.num_cached_tokens_per_seq[i]) + 1
        super().prepare()
        # ALIGN RocketKV: clamp gen-request prompt_lens to the compacted cache
        # length. After eviction num_cached is compressed but prompt_lens still
        # holds the ORIGINAL prompt length; prompt_lens > num_cached makes the
        # kernel's prompt/gen split + cache offsets inconsistent -> garbled.
        if (e is not None and kvp is not None
                and getattr(kvp, "num_cached_tokens_per_seq", None) is not None
                and hasattr(self, "prompt_lens") and self.prompt_lens is not None):
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
                self.prompt_lens_cpu[:self.num_seqs].copy_(_t[:self.num_seqs])
                self.prompt_lens_cuda[:self.num_seqs].copy_(
                    self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)


class TriAttentionTrtllmAttention(TrtllmAttention):
    """Base TRT-LLM attention carrying the TriAttention reconciliation Metadata.

    TriAttention physically EVICTS (the V2 manager gather-compacts kept tokens
    + shrinks history; the metadata clamps num_cached). Decode then does DENSE
    attention over exactly the surviving num_cached tokens -- there is NO
    decode-time sparse mask. So both sparse predictors return (None, None)."""

    Metadata: ClassVar[type] = None  # set below

    def sparse_kv_predict(self, q, k, metadata, forward_args):
        return None, None

    def sparse_attn_predict(self, q, k, metadata, forward_args):
        return None, None


TriAttentionTrtllmAttention.Metadata = TriAttentionTrtllmAttentionMetadata


# ====================================================================== #
# Pattern 1: TriAttention uses the plain KVCacheManagerV2 (NO subclass).   #
#                                                                        #
# The compacted-history reconcile lives in on_generation_step_end (above): #
# the plain V2 manager's update_resources resets each request's history to #
# max_beam-1 every step (it is unaware of eviction); the compression       #
# manager is ordered AFTER KV_CACHE_MANAGER (framework default) and        #
# shrinks history back to the compacted length there. kv_cache_manager_class#
# is the BASE V2 type purely as an isinstance sanity check that the         #
# injected manager is a V2 manager (the base __init__ asserts it).          #
# ====================================================================== #
from tensorrt_llm._torch.pyexecutor.resource_manager import (  # noqa: E402
    KVCacheManagerV2 as _KVCacheManagerV2,
)

TriAttention.kv_cache_manager_class = _KVCacheManagerV2
