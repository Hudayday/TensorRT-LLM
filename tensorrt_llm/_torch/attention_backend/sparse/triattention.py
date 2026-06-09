"""TriAttention sparse-attention method (periodic physical KV eviction).

TriAttention performs periodic generation-phase KV eviction guided by a
trigonometric importance score computed from offline-calibrated statistics of
the model's Q-pre-RoPE vectors (paper §4.1, §4.3). It physically deletes
tokens from the cache: no work in context phase, no per-step attention-time
mask, and a single ``on_generation_step_end`` hook that fires every ``beta``
steps to evict tokens below the top-B keep set.

Calibration is computed offline via
``triattention_calibration.compute_triattention_calibration`` and loaded once
at LLM init time. See the calibration module for the statistic schema.

Upstream-GitHub provenance (authoritative over paper §4.3 text):
  - scoring math: ``methods/pruning_utils.py::score_keys_for_round`` +
    ``compute_frequency_statistics_from_means`` + ``invert_rope`` /
    ``to_complex_pairs``.
  - per-(layer, head) loop + RoPE inversion:
    ``methods/triattention.py::_compute_layer_head_scores`` (~line 566).
  - global / layer-uniform top-B selection (DEFAULT mode,
    ``per_layer_perhead_pruning=False``): ``compute_keep_indices`` (~line 168)
    -> ``head_matrix.max(dim=0)`` combined score -> ``_select_union_based``
    (~line 306). ``score_aggregation="mean"``, ``normalize_scores=False``,
    ``disable_trig=False``.
"""

import os
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

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
# See triattention_calibration.compute_triattention_calibration for shapes.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})


# ====================================================================== #
# RoPE-inversion / complex-pairing helpers — direct ports of upstream    #
# methods/pruning_utils.py. The "half" (front/back pairing) RoPE style   #
# is the default for Llama/Qwen-class models (determine_rope_style       #
# returns "half" for all model_types in the reference).                  #
# ====================================================================== #


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Front/back pairing rotate_half (upstream pruning_utils.rotate_half,
    style="half")."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Undo RoPE on rotated keys (upstream pruning_utils.invert_rope,
    style="half"):  base = rotated/scale; cos_u = cos/scale; sin_u = sin/scale;
    restored = base*cos_u - rotate_half(base)*sin_u."""
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    return base * cos_unit - _rotate_half(base) * sin_unit


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Forward RoPE, exact inverse of ``_invert_rope`` (multiplies by
    ``scale**2``). Used only by the optional re-RoPE eviction path."""
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_sq = torch.tensor(scale * scale, device=x.device, dtype=x.dtype)
    return (x * cos + _rotate_half(x) * sin) * scale_sq


def _to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Pair the front/back halves of the last dim into a complex tensor
    (upstream pruning_utils.to_complex_pairs, style="half").  Input
    ``[..., head_dim]`` -> output ``[..., head_dim//2]`` complex."""
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    t = tensor.to(dtype=real_dtype)
    freq = t.shape[-1] // 2
    real = t[..., :freq].contiguous()
    imag = t[..., freq:].contiguous()
    return torch.complex(real, imag)


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
    scoring.

    Overrides ``on_generation_step_end``: every ``beta`` generation steps,
    reads the current K cache through the underlying ``KVCacheManagerV2``,
    computes a per-token importance score using offline-calibrated stats, and
    physically evicts tokens below the top-B keep set (layer-uniform: every
    layer keeps the same token set).  All other hooks remain no-op (this
    method needs no context-phase or per-attention work).

    Default mode (matches the upstream GitHub config, authoritative over the
    paper §4.3 text): ``per_layer_perhead_pruning=False`` (global /
    layer-uniform top-B), ``score_aggregation="mean"``,
    ``normalize_scores=False``, ``disable_trig=False``.
    """

    # TriAttention physically evicts tokens during decode based on per-request
    # query / step state, so the resulting cache is not safe to reuse across
    # requests (a different request would have evicted a different token set).
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Physically deletes tokens from cache (vs. RocketKV-style sparse mask).
    physically_evicts_kv: ClassVar[bool] = True

    # Pattern 3 (V2 subclass): reassigned to TriAttentionKVCacheManagerV2 at the
    # end of this module. The base manager __init__ asserts the injected cache
    # manager is that subclass; physical eviction compacts kept tokens to the
    # front pages, then the subclass update_resources resizes history_length to
    # the compacted length.
    kv_cache_manager_class: ClassVar[Optional[type]] = None

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        top_B: int,
        beta: int = 128,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
    ):
        # Validate own config BEFORE delegating to the base __init__ (which
        # asserts the Pattern-3 cache-manager subclass via isinstance). A
        # missing calibration is a user-config error, so surface it first and
        # independently of the cache-manager type.
        if calibration_path is None:
            raise ValueError(
                "TriAttention requires calibration_path; compute offline via "
                "triattention_calibration.compute_triattention_calibration "
                "before LLM init."
            )
        super().__init__(kv_cache_manager)
        self.top_B = top_B
        self.beta = beta
        # Upstream defaults (methods/triattention.py TriAttentionConfig):
        # global/layer-uniform top-B, mean aggregation, trig enabled.
        self.score_aggregation = score_aggregation
        self.calibration: Dict[str, torch.Tensor] = self._load_calibration(calibration_path)

        # Geometric integration offsets for score_keys_for_round (upstream
        # build_geometric_offsets); built lazily on first eviction so the
        # device matches the cache pool.
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Per-request generation-step counter; eviction fires when it hits
        # ``beta``.  Cleared on request finish.
        self._gen_steps: Dict[int, int] = {}

        # Cumulative physically-evicted token count per request, consumed
        # by TriAttentionTrtllmAttentionMetadata.prepare to clamp
        # num_cached_tokens_per_seq to the compacted cache length.
        self._evicted: Dict[int, int] = {}
        # Absolute positions of the physically-cached tokens per request,
        # preserved across compaction rounds (kept-token ORIGINAL positions,
        # NOT compacted slot indices). Required for the scoring RoPE
        # inversion (stored K is rotated for its absolute position) and the
        # trig recency delta. None until the first eviction.
        self._abs_positions: Dict[int, torch.Tensor] = {}
        # Optional per-eviction probe logging (TRTLLM_TRIATTENTION_PROBE=1):
        # keep/evicted counts for empirical validation vs dense.
        self._probe: bool = os.environ.get("TRTLLM_TRIATTENTION_PROBE", "0") == "1"
        # Experiment: re-RoPE kept K to contiguous positions on compaction
        # (design choice B). Default off = option A (cached K as-is).
        self._rerope: bool = os.environ.get("TRI_REROPE", "0") == "1"

        # Derived from calibration: number of layers / heads / freqs and the
        # reconstructed upstream-style stats (q_mean_complex == E_q,
        # q_abs_mean == E_q_norm).
        self._L = int(self.calibration["E_q"].shape[0])
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # ``freq_scale_sq`` is the squared per-frequency RoPE scaling factor
        # (upstream compute_frequency_scaling**2).  For "default"/un-scaled RoPE
        # this is all-ones; if the calibration carries it, honor it.
        # freq_scale_sq is now a REQUIRED calibration key (model-specific
        # RoPE amplitude; see triattention_calibration._freq_scale_sq).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        # attention_scaling (upstream rotary.attention_scaling, default 1.0).
        self._attention_scale = float(self.calibration.get("attention_scale", 1.0))

    # ------------------------------------------------------------------ #
    # Override: per-step end                                             #
    # ------------------------------------------------------------------ #

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Triggered after every generation step's full forward across all
        layers.  Per in-flight generation request, bump a step counter; every
        ``beta`` steps score the cache and physically evict down to top-B.
        """

        gen_requests = getattr(scheduled_batch, "generation_requests", None)
        if not gen_requests:
            return

        for request in gen_requests:
            rid = request.py_request_id
            step = self._gen_steps.get(rid, 0) + 1
            self._gen_steps[rid] = step
            if step % self.beta != 0:
                continue
            # Physical cache length = LOGICAL token count minus everything
            # already physically evicted for this request. get_num_tokens is
            # unaware of physical eviction and keeps growing, so reading /
            # scoring `seq_len` logical tokens out of a cache that physically
            # holds only `logical - cum_evicted` would index freed pages ->
            # CUDA IMA. The kept tokens occupy contiguous front slots
            # [0, physical_len); scoring positions are taken in this compacted
            # space (recency-consistent across rounds).
            logical_len = self._request_seq_len(request)
            seq_len = logical_len - self._evicted.get(rid, 0)
            if seq_len <= self.top_B:
                continue
            # Per-token positions for scoring (RoPE inversion + trig delta).
            device = self.calibration["E_q"].device
            prev_abs = self._abs_positions.get(rid)
            if self._rerope:
                # Option B: kept K are re-RoPE'd to contiguous positions every
                # round, so the physical cache is ALWAYS contiguous [0, seq_len).
                cur_abs = torch.arange(seq_len, device=device, dtype=torch.long)
                round_start = seq_len
            elif prev_abs is None:
                # Option A, first round: contiguous prefix [0, seq_len).
                cur_abs = torch.arange(seq_len, device=device, dtype=torch.long)
                round_start = logical_len
            else:
                # Option A, later rounds: kept tokens keep ORIGINAL absolute
                # positions; new decode tokens take the most recent logical
                # positions.
                num_new = seq_len - int(prev_abs.numel())
                new_abs = torch.arange(
                    logical_len - num_new, logical_len, device=device, dtype=torch.long
                )
                cur_abs = torch.cat([prev_abs.to(device), new_abs])
                round_start = logical_len
            keep_indices = self._compute_keep_indices(
                request, seq_len, key_positions=cur_abs, round_start=round_start
            )
            if keep_indices is None:
                continue
            kept_abs = cur_abs.index_select(0, keep_indices.to(cur_abs.device))
            self._evict(request, keep_indices, seq_len, kept_abs=kept_abs)
            if self._rerope:
                # Re-RoPE'd to contiguous; next round they live at [0, keep).
                self._abs_positions[rid] = torch.arange(
                    int(kept_abs.numel()), device=device, dtype=torch.long
                )
            else:
                self._abs_positions[rid] = kept_abs

    # ------------------------------------------------------------------ #
    # Scoring (upstream port)                                            #
    # ------------------------------------------------------------------ #

    def _compute_keep_indices(
        self,
        request: "LlmRequest",
        seq_len: int,
        key_positions: Optional[torch.Tensor] = None,
        round_start: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Global / layer-uniform top-B keep-set (upstream
        compute_keep_indices, per_layer_perhead_pruning=False path).

        1. score every token with each sampled (layer, head) -> head_matrix
           ``[num_sampled_heads, seq_len]``.
        2. combined = head_matrix.max(dim=0).values  (upstream line 292).
        3. union-based top-B selection (upstream _select_union_based).
        Returns sorted ascending keep indices (LongTensor) or ``None`` when no
        head could be scored.
        """
        device = self.calibration["E_q"].device
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        # Absolute token positions of the keys currently in cache (0..seq_len-1)
        # — these are the per-key indices for score_keys_for_round and the
        # RoPE positions for invert_rope.  TriAttention keeps the full prefix
        # (no per-head divergence in the default mode), so positions are dense.
        if key_positions is None:
            key_positions = torch.arange(seq_len, device=device, dtype=torch.long)
        else:
            key_positions = key_positions.to(device=device, dtype=torch.long)
        if round_start is None:
            round_start = seq_len

        all_head_scores: List[torch.Tensor] = []
        for layer_idx in range(self._L):
            k_layer = self._read_request_k(request, layer_idx, seq_len)
            if k_layer is None:
                continue
            layer_scores = self._score_layer(k_layer, key_positions, layer_idx, round_start)
            if layer_scores is not None:
                all_head_scores.append(layer_scores)

        if not all_head_scores:
            return None

        # [num_sampled_heads, seq_len]
        head_matrix = torch.cat(all_head_scores, dim=0)
        # Default mode: normalize_scores=False, no tie-break noise (deterministic
        # greedy decode).  combined = per-token max across heads (upstream 292).
        combined = head_matrix.max(dim=0).values

        keep_count = min(self.top_B, seq_len)
        return self._select_union_based(head_matrix, combined, keep_count)

    def _score_layer(
        self,
        k_layer: torch.Tensor,
        key_positions: torch.Tensor,
        layer_idx: int,
        round_start: int,
    ) -> Optional[torch.Tensor]:
        """Port of upstream _compute_layer_head_scores for a single layer.

        ``k_layer`` is ``[num_kv_heads, seq_len, head_dim]`` (RoPE-rotated, as
        stored in the cache).  Returns ``[num_heads_in_layer, seq_len]`` scores
        or ``None``.
        """
        device = self.calibration["E_q"].device
        head_dim = k_layer.shape[-1]
        num_kv_heads = k_layer.shape[0]

        # Build a single cos/sin table for the key positions (shared across
        # heads — default mode, positions_per_kv_head is None upstream).
        cos, sin = self._rope_tables(key_positions, head_dim, device)

        E_q = self.calibration["E_q"][layer_idx]  # [H, F] complex
        E_q_norm = self.calibration["E_q_norm"][layer_idx]  # [H, F] float
        omega = self.calibration["omega"].to(device=device, dtype=torch.float32)

        per_head_scores: List[torch.Tensor] = []
        for head in range(self._H):
            # GQA: map the (calibrated) Q head onto its KV head.
            kv_head = min(num_kv_heads - 1, head * num_kv_heads // max(1, self._H))
            k_values = k_layer[kv_head].to(device=device, dtype=cos.dtype)
            # Undo RoPE -> pre-RoPE keys, then complex-pair.
            k_unrot = _invert_rope(k_values, cos, sin, self._attention_scale)
            amp, phi, extra = self._frequency_statistics(E_q[head], E_q_norm[head], k_unrot)
            head_scores = self._score_keys_for_round(
                key_indices=key_positions,
                round_start=round_start,
                amp=amp,
                phi=phi,
                omega=omega,
                extra=extra,
            )
            per_head_scores.append(head_scores)

        if not per_head_scores:
            return None
        return torch.stack(per_head_scores, dim=0)  # [H, seq_len]

    def _frequency_statistics(
        self,
        q_mean_complex: torch.Tensor,
        q_abs_mean: torch.Tensor,
        k_unrot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Port of upstream compute_frequency_statistics_from_means
        (disable_mlr=False):

            k_complex = to_complex_pairs(k_unrot)              # [seq, F]
            relative  = q_mean_complex * conj(k_complex)       # broadcast
            phi  = atan2(relative.imag, relative.real)         # [seq, F]
            amp  = |q_mean_complex| * |k_complex|              # [seq, F]
            extra = (q_abs_mean - |q_mean_complex|) * |k_complex|

        ``q_mean_complex`` == calib E_q[layer, head]; ``q_abs_mean`` ==
        calib E_q_norm[layer, head].  Both are ``[F]``; k_complex is
        ``[seq, F]`` so the result is ``[seq, F]`` (upstream uses unsqueeze(0)
        over a single-element batch — here the per-token seq axis plays that
        role directly).
        """
        k_complex = _to_complex_pairs(k_unrot)  # [seq, F]
        q_mean_complex = q_mean_complex.to(k_complex.dtype)
        q_mean_abs = torch.abs(q_mean_complex)  # [F]
        k_abs = torch.abs(k_complex)  # [seq, F]
        relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
        phi = torch.atan2(relative.imag, relative.real)  # [seq, F]
        amp = q_mean_abs.unsqueeze(0) * k_abs  # [seq, F]
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
        return amp, phi, extra

    def _score_keys_for_round(
        self,
        key_indices: torch.Tensor,
        round_start: int,
        amp: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """Port of upstream pruning_utils.score_keys_for_round (mean
        aggregation, disable_trig=False).

            delta      = round_start - key_index                       # [N]
            delta_grid = delta[:, None] + offsets[None, :]             # [N, O]
            phase      = delta_grid[:, :, None]*omega[None,None,:] + phi[:,None,:]
            base       = (amp[:, None, :] * freq_scale_sq * cos(phase)).sum(-1) # [N, O]
            additive   = (extra * freq_scale_sq).sum(-1, keepdim=True)          # [N, 1]
            combined   = base + additive
            score      = combined.mean(dim=1)   (or .max for aggregation="max")
        """
        if key_indices.numel() == 0:
            return torch.empty(0, device=amp.device, dtype=torch.float32)
        offsets = self._offsets
        fss = self._freq_scale_sq.to(device=amp.device, dtype=torch.float32)

        base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
        delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)  # [N, O]
        phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)  # [N, O, F]
        cos_phase = torch.cos(phase)
        scale = fss.view(1, 1, -1)
        base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)  # [N, O]
        additive = (extra * fss.view(1, -1)).sum(dim=1, keepdim=True)  # [N, 1]
        combined = base_scores + additive  # [N, O]
        if self.score_aggregation == "mean":
            return combined.mean(dim=1)
        return combined.max(dim=1).values

    def _select_union_based(
        self,
        head_matrix: torch.Tensor,
        combined: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        """Port of upstream TriAttention._select_union_based (~line 306).

        1. each head selects its own top-keep_count tokens -> union mask.
        2. union indices.
        3. if |union| >= keep_count, take top-keep_count of union by combined.
        4. else fill the remainder from the best non-union tokens by combined.
        Returns sorted ascending LongTensor of kept indices.
        """
        candidate_count = combined.shape[0]
        device = combined.device
        if keep_count >= candidate_count:
            return torch.arange(candidate_count, device=device, dtype=torch.long)

        union_mask = torch.zeros(candidate_count, device=device, dtype=torch.bool)
        head_k = min(keep_count, candidate_count)
        for h in range(head_matrix.shape[0]):
            top_idx = torch.topk(head_matrix[h], k=head_k, largest=True).indices
            union_mask[top_idx] = True

        union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_indices.numel() >= keep_count:
            subset_scores = combined.index_select(0, union_indices)
            top_subset = torch.topk(subset_scores, k=keep_count, largest=True).indices
            return union_indices.index_select(0, torch.sort(top_subset).values)

        remaining = keep_count - union_indices.numel()
        available = candidate_count - union_indices.numel()
        if remaining > 0 and available > 0:
            extra_k = min(remaining, available)
            residual_scores = combined.clone()
            residual_scores[union_mask] = float("-inf")
            extra_indices = torch.topk(residual_scores, k=extra_k, largest=True).indices
            union_indices = torch.cat([union_indices, extra_indices])
        return torch.sort(union_indices).values

    # ------------------------------------------------------------------ #
    # V2-manager cache access + physical eviction                        #
    # ------------------------------------------------------------------ #

    def _rope_tables(self, positions: torch.Tensor, head_dim: int, device: torch.device):
        """Build cos/sin RoPE tables for the given absolute positions from the
        calibrated ``omega`` (inv_freq).  Mirrors the upstream rotary(base,
        positions) call but reconstructs cos/sin from omega directly so no live
        HF rotary module is needed in the executor.

        For "half" RoPE, the cos/sin tables tile the per-frequency angle across
        both halves of the head dim: angle = position * omega[f]; the full-dim
        table is ``cat([angle, angle], dim=-1)`` then cos / sin.
        """
        omega = self.calibration["omega"].to(device=device, dtype=torch.float32)
        pos = positions.to(device=device, dtype=torch.float32)  # [N]
        angles = pos.unsqueeze(1) * omega.unsqueeze(0)  # [N, F]
        full = torch.cat([angles, angles], dim=-1)  # [N, 2F]
        return torch.cos(full), torch.sin(full)

    def _resolve_page_ids(self, request: "LlmRequest", layer_idx: int) -> Optional[List[int]]:
        """Resolve this request's paged-cache block indices via V2's public
        API, falling back to the V1 signature.

        V2 (``KVCacheManagerV2``) exposes ``get_batch_cache_indices(ids,
        layer_idx) -> List[List[int]]`` (resource_manager.py:3348); V1
        (``KVCacheManager``) exposes ``get_cache_indices(request) ->
        List[int]`` (resource_manager.py:1469).  Prefer the V2 batch API (the
        manager TriAttention is wired to), fall back to V1, return ``None``
        when neither is present (mocked tests).  The returned indices already
        divide by ``kv_factor`` so they index the aggregate
        ``[num_pages, kv_factor, tpb, ...]`` pool returned by ``get_buffers``.
        """
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
        pool.  Mirrors how rocketkv reads pool tensors via the V2 manager
        (``get_buffers`` / ``get_cache_indices``).

        Returns ``[num_kv_heads, seq_len, head_dim]`` (K only) or ``None`` when
        the manager doesn't expose a readable pool (e.g. mocked tests).
        """
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return None
        # [max_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim] (NHD)
        pool = get_buffers(layer_idx, kv_layout="NHD")
        if pool is None:
            return None
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return None
        tpb = pool.shape[2]
        k_index = 0  # KEY is index 0 along kv_factor; VALUE is 1.
        pages = pool[page_ids][:, k_index]  # [num_pages, tpb, num_kv_heads, hd]
        num_pages = pages.shape[0]
        # [num_pages*tpb, num_kv_heads, hd] -> trim to seq_len
        flat = pages.reshape(num_pages * tpb, pages.shape[2], pages.shape[3])
        flat = flat[:seq_len]
        # -> [num_kv_heads, seq_len, head_dim]
        return flat.permute(1, 0, 2).contiguous()

    def _evict(
        self,
        request: "LlmRequest",
        keep_indices: torch.Tensor,
        seq_len: int,
        kept_abs: Optional[torch.Tensor] = None,
    ) -> None:
        """Physically evict tokens NOT in ``keep_indices`` (design choice A,
        doc 48 §6): gather-compact the kept tokens to the front contiguous
        slots of the request's pages and shrink the cache, **without
        re-rotating** the kept keys.

        K is stored POST-RoPE (rope_fusion disabled for sparse methods, see
        modules/attention.py).  Each kept key keeps its ORIGINAL rotation and
        the cached value is used directly (user directive: 计算时直接用
        kvcache 的值，不加).  The decode query's RoPE position comes from the
        request's FULL logical length (model_engine ``past_seen_token_num =
        max_beam_num_tokens - 1``) and is baked into ``position_ids`` BEFORE
        ``TriAttentionTrtllmAttentionMetadata.prepare`` clamps
        ``num_cached_tokens_per_seq``.  So the query still rotates at its true
        absolute position while the kernel reads only the compacted tokens:
        query@true-pos vs kept-key@original-rotation yields the CORRECT
        relative distance with NO re-RoPE (mirrors vLLM's preserve-positions
        compaction; re-RoPE = choice B was rejected).  The probe (NIAH/MMLU vs
        dense) is the empirical gate on this position-decoupling.

        Layer-uniform: the same keep-set applies to every layer.
        """
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return

        keep = keep_indices.to(dtype=torch.long)
        keep_count = int(keep.numel())
        if keep_count >= seq_len:
            return  # nothing to evict

        num_layers = self._num_layers_from_manager()
        for layer_idx in range(num_layers):
            layer_page_ids = self._resolve_page_ids(request, layer_idx)
            if not layer_page_ids:
                continue
            pool = get_buffers(layer_idx, kv_layout="NHD")
            if pool is None:
                continue
            tpb = pool.shape[2]
            page_idx_t = torch.as_tensor(layer_page_ids, device=pool.device, dtype=torch.long)
            # Advanced-indexed read returns a COPY of this request's pages:
            #   [num_pages, kv_factor, tpb, num_kv_heads, head_dim] (NHD).
            req_pages = pool[page_idx_t]
            num_pages, kv_factor, _, nkv, hd = req_pages.shape
            keep_dev = keep.to(req_pages.device)
            # The token axis is (page, slot) = dims 0 and 2, but kv_factor
            # (dim 1) sits BETWEEN them, so a direct reshape to
            # [num_pages*tpb, ...] would scramble memory (cf. _read_request_k,
            # which selects the kv_factor index out first). Permute kv_factor
            # to the front so the (num_pages, tpb) token axis is contiguous:
            #   [num_pages, kv_factor, tpb, nkv, hd]
            #     -> [kv_factor, num_pages, tpb, nkv, hd]
            #     -> [kv_factor, num_pages*tpb, nkv, hd]
            perm = (
                req_pages.permute(1, 0, 2, 3, 4)
                .contiguous()
                .reshape(kv_factor, num_pages * tpb, nkv, hd)
            )
            # Gather kept tokens (K at index 0 and V at index 1 together) to
            # the front slots.  No re-RoPE: cached post-RoPE K used as-is; V is
            # position-independent.
            kept = perm.index_select(1, keep_dev)  # [kv_factor, keep, nkv, hd]
            if self._rerope and kept_abs is not None:
                # Design choice B: re-rotate kept K from its current position
                # to the NEW contiguous position [0, keep_count) so a collapsed
                # (contiguous) query position stays self-consistent.
                kf = kept[0].to(torch.float32)  # [keep, nkv, hd]
                orig = kept_abs.to(device=kf.device, dtype=torch.long)
                new = torch.arange(keep_count, device=kf.device, dtype=torch.long)
                cos_o, sin_o = self._rope_tables(orig, hd, kf.device)
                cos_n, sin_n = self._rope_tables(new, hd, kf.device)
                kf = _invert_rope(kf, cos_o.unsqueeze(1), sin_o.unsqueeze(1), self._attention_scale)
                kf = _apply_rope(kf, cos_n.unsqueeze(1), sin_n.unsqueeze(1), self._attention_scale)
                kept[0] = kf.to(kept.dtype)
            perm[:, :keep_count].copy_(kept)
            # Reshape / permute back and scatter the touched front pages into
            # the live pool.
            num_front_pages = (keep_count + tpb - 1) // tpb
            back = (
                perm.reshape(kv_factor, num_pages, tpb, nkv, hd).permute(1, 0, 2, 3, 4).contiguous()
            )
            pool[page_idx_t[:num_front_pages]] = back[:num_front_pages]

        # Shrink the physical cache (V2 public rewind_kv_cache, the same
        # binding RocketKV.on_context_end uses) and record the cumulative
        # evicted count so the metadata shim reconciles num_cached next iter.
        evicted = seq_len - keep_count
        rid = request.py_request_id
        self._evicted[rid] = self._evicted.get(rid, 0) + evicted
        # Hand the cumulative physical-eviction count to the V2 manager.
        # TriAttentionKVCacheManagerV2.update_resources resizes this request's
        # history_length to (max_beam_num_tokens - 1 - py_tri_evicted) so the
        # manager's slot mapping matches the front-compacted data written by
        # the gather loop above (get_buffers view).  No rewind_kv_cache (no-op
        # mid-decode); the manager owns the resize and runs after this hook
        # (KV_CACHE_MANAGER is invoked last among resource managers).
        request.py_tri_evicted = self._evicted[rid]
        if self._probe:
            logger.info(
                "[TriAttention] evict rid=%d seq_len=%d keep=%d evicted=%d cum=%d",
                rid,
                seq_len,
                keep_count,
                evicted,
                self._evicted[rid],
            )

    def _request_seq_len(self, request: "LlmRequest") -> int:
        get_num_tokens = getattr(request, "get_num_tokens", None)
        if get_num_tokens is not None:
            try:
                return int(get_num_tokens(0))
            except Exception:
                pass
        return 0

    def _num_layers_from_manager(self) -> int:
        mgr = self.kv_cache_manager
        layer_offsets = getattr(mgr, "layer_offsets", None)
        if layer_offsets:
            return len(layer_offsets)
        # Fall back to the calibrated layer count.
        return self._L

    # ------------------------------------------------------------------ #
    # Request lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Drop this request's per-request step + evicted counters."""
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)
        self._abs_positions.pop(request.py_request_id, None)

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id`` (read by
        TriAttentionTrtllmAttentionMetadata.prepare to reconcile
        num_cached_tokens_per_seq after compaction)."""
        return self._evicted.get(request_id, 0)

    # ------------------------------------------------------------------ #
    # Calibration loading                                                #
    # ------------------------------------------------------------------ #

    def _load_calibration(self, path: str) -> Dict[str, torch.Tensor]:
        """Load offline-computed calibration ``.pt`` onto GPU.

        Expected schema (produced by ``compute_triattention_calibration``):
          E_q       [L, H, D/2] complex   per-(layer, head, freq) Q center
          E_q_norm  [L, H, D/2] float     E[||q_f||]
          R         [L, H, D/2] float     MRL (Mean Resultant Length)
          omega     [D/2]       float     RoPE freqs (model-dependent)
          phi       [L, H, D/2] float     phase offset, arg(E_q)
        """
        calibration = torch.load(path, map_location="cuda")
        self._validate_calibration(calibration)
        return calibration

    def _validate_calibration(self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys.

        Shape consistency checks against the running model's L / H / D are
        deferred to the eviction kernel (paper §4.3 algorithm); the skeleton
        only enforces key presence.
        """
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
# cache (unlike RocketKV which emits a sparse mask).  The only shim work #
# is reconciling ``num_cached_tokens_per_seq`` after physical eviction;  #
# everything else is the base TrtllmAttention.                           #
# ====================================================================== #
from tensorrt_llm._torch.attention_backend.trtllm import (  # noqa: E402
    TrtllmAttention,
    TrtllmAttentionMetadata,
)


class TriAttentionTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Metadata shim: reconcile num_cached after TriAttention compaction.

    The model engine derives ``num_cached_tokens_per_seq`` from the request's
    FULL logical length (``max_beam_num_tokens - 1``), unaware that
    ``TriAttention._evict`` has physically compacted the cache.  Left
    unadjusted, the dense kernel would read more tokens than physically
    exist -> OOB.  ``prepare`` subtracts each request's cumulative evicted
    count (mirrors RocketKV's prepare num_cached rewind) so the kernel reads
    exactly the compacted tokens.  ``position_ids`` (query RoPE rotation) are
    already baked from the full logical length, so the query rotates at its
    true absolute position -> correct relative distance vs the kept keys'
    original rotations (design choice A, doc 48 §6).
    """

    @property
    def _tri_executor(self) -> Optional["TriAttention"]:
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
                    kvp.num_cached_tokens_per_seq[i] -= ev
        super().prepare()


class TriAttentionTrtllmAttention(TrtllmAttention):
    """Base TRT-LLM attention carrying the TriAttention reconciliation
    Metadata (num_cached clamp after physical eviction)."""

    Metadata: ClassVar[type] = None  # set below


TriAttentionTrtllmAttention.Metadata = TriAttentionTrtllmAttentionMetadata


# ====================================================================== #
# V2 KV cache manager subclass — reconcile logical length after physical  #
# eviction (F6). V2 _core is pure-Python so the subclass can shrink a      #
# request mid-decode, which V1 (C++ KVCacheManagerCpp) cannot.             #
# ====================================================================== #
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState  # noqa: E402
from tensorrt_llm._torch.pyexecutor.resource_manager import (  # noqa: E402
    DEFAULT_BEAM_INDEX,
    KVCacheManagerV2,
    _update_kv_cache_draft_token_location,
)


class TriAttentionKVCacheManagerV2(KVCacheManagerV2):
    """V2 manager that reconciles a request's logical length after TriAttention
    physically compacts its KV mid-decode.

    The base gen ``update_resources`` resizes every gen request to
    ``history_length = max_beam_num_tokens - 1`` each iteration (full logical
    length, unaware of physical eviction).  After ``TriAttention._evict`` moves
    the kept K/V to the front pages and sets ``request.py_tri_evicted``, this
    would resize ``history_length`` back to the full length — placing new
    decode tokens at the wrong (full-length) slot and (if capacity were
    shrunk) violating ``history_length <= capacity``.  We override the gen
    resize to use the COMPACTED length ``max_beam_num_tokens - 1 -
    py_tri_evicted`` so the manager's slot mapping matches the physical
    front-compacted data.  ``position_ids`` are derived from
    ``max_beam_num_tokens - 1`` in the model engine (independent of this), so
    the decode query still rotates at its TRUE absolute position while the
    kernel reads/writes the compacted cache.

    NOTE: the gen+context loop is replicated from
    ``KVCacheManagerV2.update_resources`` (resource_manager.py) with only the
    gen ``history_length`` adjusted — keep in sync if the base changes.
    """

    def update_resources(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
        if not self.is_draft:
            _update_kv_cache_draft_token_location(
                self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
            )

        # Context requests — identical to base.
        for req in scheduled_batch.context_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            if not kv_cache.is_active:
                continue
            if self.enable_block_reuse and not self.is_draft and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    tokens = self._augment_tokens_for_block_reuse(
                        req.get_tokens(DEFAULT_BEAM_INDEX),
                        req,
                        start=kv_cache.num_committed_tokens,
                        end=req.context_current_position,
                    )
                    kv_cache.commit(tokens)
                if req.context_remaining_length == 0:
                    kv_cache.stop_committing()
            else:
                if not kv_cache.resize(None, req.context_current_position):
                    raise ValueError(f"ctx resize failed for req {req.py_request_id}")

        # Generation requests — history_length reduced by cumulative eviction.
        for req in scheduled_batch.generation_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            if not kv_cache.is_active:
                continue
            if req.state in (LlmRequestState.GENERATION_COMPLETE, LlmRequestState.CONTEXT_INIT):
                new_capacity = None
                target_hist = req.max_beam_num_tokens - 1
            else:
                ev = int(getattr(req, "py_tri_evicted", 0) or 0)
                target_hist = req.max_beam_num_tokens - 1 - ev
                new_capacity = kv_cache.capacity - req.py_rewind_len
                if new_capacity < target_hist:
                    new_capacity = target_hist
            # Bypass the monotonic-history_length guard on the eviction-step
            # decrease (the guard lives in resize(); _history_length is a plain
            # _core attr).
            if target_hist < kv_cache.history_length:
                kv_cache._history_length = target_hist
            if not kv_cache.resize(new_capacity, target_hist):
                raise ValueError(
                    f"gen resize failed for req {req.py_request_id} "
                    f"cap={new_capacity} hist={target_hist}"
                )


# Wire Pattern-3: TriAttention executor requires this V2 subclass.
TriAttention.kv_cache_manager_class = TriAttentionKVCacheManagerV2
