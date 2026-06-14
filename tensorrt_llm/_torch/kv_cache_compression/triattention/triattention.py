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
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
)
from tensorrt_llm.logger import logger

try:
    from .triattention_kernels import (
        triton_tri_score,
        triton_tri_reduce_heads,
        triton_tri_select,
        triton_tri_compact,
    )
    _TRI_TRITON_AVAILABLE = True
except Exception:  # triton not importable in some envs (CPU-only / mocked tests)
    _TRI_TRITON_AVAILABLE = False

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

# Per-component eviction profiling (temporary; remove before PR). Gated by a
# sentinel FILE (not env -- env does not reach the IPC executor worker). When
# present, each batched eviction round appends a JSONL line of per-stage timings
# to a per-pid file. Zero overhead when the sentinel is absent.
_TRI_PROFILE = os.path.exists("/scratch/triattn_e2e/.tri_profile")
_TRI_PROF_FILE = f"/scratch/triattn_e2e/prof/round_{os.getpid()}.jsonl"

# Determinism trace (temporary; remove before PR). Logs the eviction LENGTH
# bookkeeping per round per request -- all host ints, NO cuda sync, so it does
# not perturb the overlap pipeline. Two cross-process runs of an identical batch
# should produce identical traces; a diff localizes an overlap length-desync.
_TRI_DETTRACE = os.path.exists("/scratch/triattn_e2e/.tri_dettrace")
_TRI_DETTRACE_FILE = f"/scratch/triattn_e2e/dettrace/dt_{os.getpid()}.jsonl"

# Overlap-safety fix-test (temporary). Brackets the periodic eviction with a
# full device sync so the eviction's KV read/write cannot race the overlapped
# next-iteration forward. Heavy (kills overlap during the eviction step); a
# confirmation tool -- if it makes the batched-overlap run deterministic, the
# real fix is a lighter stream/event ordering, not a global sync.
_TRI_EVICT_SYNC = os.path.exists("/scratch/triattn_e2e/.tri_evict_sync")
_TRI_ROOTPROBE = os.path.exists("/scratch/triattn_e2e/.tri_rootprobe")
_TRI_EVCOUNT = os.path.exists("/scratch/triattn_e2e/.tri_evcount")
_TRI_FWDPROBE = os.path.exists("/scratch/triattn_e2e/.tri_fwdprobe")
# Keep-set probe (remove before PR): logs the first 2 evictions' layer-0 kept
# token set per request, so different eviction_modes can be compared directly
# (do per_layer / per_head / per_layer_perhead actually select different
# tokens? do heads within a per-head mode differ?). Sentinel file, per-pid out.
_TRI_KEEPPROBE = os.path.exists("/scratch/triattn_e2e/.tri_keepprobe")
_TRI_KEEPPROBE_FILE = f"/scratch/triattn_e2e/keepprobe/kp_{os.getpid()}.jsonl"

# Block-reclaim experiment (paper capacity gain). Gated; OFF = compact-only
# (byte-identical, validated). ON = after compaction, mark each evicted request
# with a pending capacity-shrink + a CUDA event recorded on the eviction stream;
# the TriAttentionKVCacheManagerV2 subclass executes the DEFERRED free one
# iteration later (event.query()-guarded), so the page-reuse-gating finish_event
# transitively orders behind the compaction kernel (fixes the read-after-free).
_TRI_FREE_BLOCKS = os.path.exists("/scratch/triattn_e2e/.tri_free_blocks")


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


# The active TriAttention manager, registered in ``TriAttention.__init__`` so the
# attention-metadata shim can find it WITHOUT any framework wiring. PR-15106 does
# not set ``metadata.compression_manager`` and its hooks fire post-forward, so the
# num_cached-after-compaction reconcile (which must run at metadata.prepare time)
# reads the manager from here. Keeps the whole integration inside this package.
_ACTIVE_TRI_MANAGER = None


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
        use_triton: bool = False,
        use_batched: bool = False,
        eviction_mode: str = "per_layer",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
    ):
        super().__init__(kv_cache_manager)
        # Register as the active manager so the attention-metadata shim can find
        # us without framework wiring (see _ACTIVE_TRI_MANAGER above).
        global _ACTIVE_TRI_MANAGER
        _ACTIVE_TRI_MANAGER = self
        self.top_B = top_B
        self.beta = beta
        # Which token set each eviction keeps:
        #   per_layer          -- average heads, one set per layer (the simple
        #                         variant; uses the recency window below).
        #   per_head           -- each KV head keeps its own set, shared across
        #                         layers (mean of per-layer max). Upstream AIME
        #                         default.
        #   per_layer_perhead  -- each (layer, KV head) keeps its own set,
        #                         fully independent per layer.
        #   union              -- union of every head's top-k, re-ranked by the
        #                         per-token max score.
        # The non-per_layer modes reproduce the upstream selection: z-normalize
        # the scores, pin the prompt (prefill) tokens, and use NO recency window.
        self.eviction_mode = eviction_mode
        self.normalize_scores = bool(normalize_scores)
        self.pin_prefill = bool(pin_prefill)
        # Recency window: the most recent ``window_size`` tokens are ALWAYS kept
        # (upstream TRIATTN_RUNTIME_WINDOW_SIZE). Without it the trig scorer
        # evicts the model's freshly-generated tokens (they score low) and the
        # model degenerates on repeated eviction -- the single most important
        # correctness knob for multi-round eviction.
        self.window_size = int(window_size)
        self.score_aggregation = score_aggregation
        # Use the vendored Triton eviction kernels, else the PyTorch reference
        # path. Driven by the config (reaches the executor worker through the
        # serialized config); default OFF -- PyTorch stays the reference.
        self.use_triton = bool(use_triton) and _TRI_TRITON_AVAILABLE
        # Batched eviction: process ALL requests evicting this step in ONE pass
        # of batched kernels (score x req x layer, select, per-layer compact),
        # decoupling launch count from request count (large-batch serving). Needs
        # the Triton kernels; default OFF (per-request path is the reference).
        self.use_batched = bool(use_batched) and self.use_triton

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
        # Keep-set probe bookkeeping (remove before PR): evictions logged per rid.
        self._keepprobe_n: Dict[int, int] = {}

    def _log_keepprobe(self, mode: str, rid: int, step: int, seq_len: int,
                       decode_start: int, keep: "torch.Tensor") -> None:
        """Append a compact summary of this eviction's layer-0 kept token set so
        modes can be compared offline. ``keep`` is 1-D (per_layer/union) or 2-D
        ``[nkv, k]`` (per_head/per_layer_perhead). Remove before PR."""
        if self._keepprobe_n.get(rid, 0) >= 2:
            return
        self._keepprobe_n[rid] = self._keepprobe_n.get(rid, 0) + 1
        import json as _json
        import hashlib as _hl

        def _h(t):
            return _hl.md5(t.detach().cpu().to(torch.long).numpy().tobytes()
                           ).hexdigest()[:10]
        rec = {"mode": mode, "rid": rid, "step": step, "seq_len": seq_len,
               "decode_start": decode_start, "ndim": int(keep.dim())}
        if keep.dim() == 1:
            ks = torch.sort(keep).values
            rec.update(k=int(ks.numel()), union_hash=_h(ks),
                       first8=ks[:8].tolist(), last8=ks[-8:].tolist())
        else:
            nkv = int(keep.shape[0])
            uni = torch.sort(torch.unique(keep.flatten())).values
            h0 = torch.sort(keep[0]).values
            h1 = torch.sort(keep[min(1, nkv - 1)]).values
            rec.update(nkv=nkv, k_per_head=int(keep.shape[1]),
                       union_size=int(uni.numel()), union_hash=_h(uni),
                       head0_hash=_h(h0), head1_hash=_h(h1),
                       heads_differ=bool(not torch.equal(h0, h1)))
        try:
            os.makedirs(os.path.dirname(_TRI_KEEPPROBE_FILE), exist_ok=True)
            with open(_TRI_KEEPPROBE_FILE, "a") as _f:
                _f.write(_json.dumps(rec) + "\n")
        except OSError:
            pass

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
        # Pre-split query stats + MLR coefficient for the Triton score kernel so
        # it doesn't recompute (E_q_norm - |E_q|) per call. Shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._tri_q_real = _Eq.real.to(torch.float32).contiguous()
        self._tri_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._tri_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    # The framework drives all 8 lifecycle hooks; TriAttention overrides only
    # on_generation_step_end (periodic eviction) and on_request_finish (per-
    # request cleanup). It scores from offline calibration, not from live
    # queries or attention scores, so it needs no per-layer attention hook: the
    # whole eviction runs once per period in on_generation_step_end, which loops
    # the layers and reads each layer's keys straight from the KV pool.

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Run the periodic eviction at the START of the iteration -- BEFORE this
        iteration's forward -- so the forward (and the attention metadata) read
        the post-eviction, post-reconcile cache.

        Doing the eviction in on_generation_step_end (AFTER the forward, via
        update_resources) races the overlap scheduler: the next iteration's
        forward is launched before the eviction mutates the KV, so it reads a
        racy / stale-length cache (det bs=32 overlap-ON: 32/32 divergent; a GPU
        sync around the eviction did NOT fix it because the forward's metadata
        was already computed). Mirrors RocketKV, which evicts before the forward
        (at the prefill->generation boundary).
        """
        super().prepare_resources(scheduled_batch)
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

        # (2) evict. Batched path processes all evicting requests in ONE pass of
        # batched kernels (score x req x layer, select, per-layer compact); the
        # per-request path stays the byte-identical reference.
        if _TRI_PROFILE:
            import time as _tm2; _ev_t0 = _tm2.perf_counter()
        if evict_now:
            if _TRI_EVICT_SYNC:
                # fix-test: the overlapped next-iter forward reads/writes the KV;
                # finish it before the eviction reads K / overwrites slots.
                torch.cuda.synchronize()
            if self.use_batched:
                self._evict_batch(evict_now, num_layers)
            else:
                for request, rid in evict_now:
                    self._maybe_evict(request, rid, num_layers)
            if _TRI_EVICT_SYNC:
                # fix-test: the compaction must be fully applied before the next
                # forward (enqueued in the following loop iteration) reads the KV.
                torch.cuda.synchronize()
        # Block-reclaim ordering: record a CUDA event AFTER the compaction kernels
        # (same stream they ran on). TriAttentionKVCacheManagerV2.update_resources
        # waits this on the manager stream before the capacity-shrink, so the
        # page-reuse-gating finish_event recorded inside resize() dominates the
        # compaction -> no read-after-free when a freed page is reallocated.
        if _TRI_FREE_BLOCKS and evict_now:
            _cev = torch.cuda.Event()
            _cev.record()
            for request, rid in evict_now:
                request.py_tri_compaction_event = _cev
        if _TRI_PROFILE:
            self._t_evict = getattr(self, "_t_evict", 0.0) + (_tm2.perf_counter() - _ev_t0)
            _rc_t0 = _tm2.perf_counter()

        # (3) reconcile history_length for EVERY request with cumulative evictions.
        for request in gen_requests:
            rid = request.py_request_id
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
                    # to the front).
                    target_hist = request.max_beam_num_tokens - ev
                    if target_hist < kv_cache.history_length:
                        kv_cache._history_length = target_hist
                    kv_cache.resize(None, target_hist)
        if _TRI_PROFILE:
            self._t_reconcile = getattr(self, "_t_reconcile", 0.0) + (_tm2.perf_counter() - _rc_t0)
            if evict_now:
                try:
                    with open(f"/scratch/triattn_e2e/prof/timers_{os.getpid()}.json", "w") as _f:
                        import json as _js
                        _js.dump({"t_evict": self._t_evict, "t_reconcile": self._t_reconcile,
                                  "t_trim": getattr(self, "_t_trim", 0.0),
                                  "t_score": getattr(self, "_t_score", 0.0),
                                  "t_select": getattr(self, "_t_select", 0.0),
                                  "t_compact": getattr(self, "_t_compact", 0.0),
                                  "step": self._gen_steps.get(gen_requests[0].py_request_id, 0)}, _f)
                except Exception:
                    pass

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request step + evicted counters."""
        if _TRI_EVCOUNT:
            # trigger-count probe (remove before PR): gen_steps = #times
            # _periodic_evict saw this request (= #generation iterations it was
            # scheduled in). Should equal the real decode-step count. Compare
            # overlap vs disable for the same prompt: gen_steps_overlap >
            # gen_steps_disable => the eviction trigger is OVER-COUNTED under
            # overlap (prepare_resources runs more than once per real token).
            rid = request.py_request_id
            jid = os.environ.get("SLURM_JOB_ID", "x")
            with open(f"/scratch/triattn_e2e/evcount_{jid}_{os.getpid()}.jsonl", "a") as _f:
                _f.write(f"{rid}\t{self._gen_steps.get(rid, -1)}\t"
                         f"{self._evicted.get(rid, 0)}\t{request.max_beam_num_tokens}\n")
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
        # Upstream-faithful modes (per_head / per_layer_perhead / union): a
        # separate selection that keeps a DIFFERENT token set per KV head (or a
        # single union set), z-normalizes the scores, and pins the prompt. The
        # kept COUNT is still uniform (= top_B) across heads and layers, so the
        # per-request num_cached bookkeeping below is unchanged.
        if self.eviction_mode != "per_layer":
            keep_count = self._evict_modes(request, num_layers, seq_len,
                                           round_start)
            if keep_count is None:
                return
            evicted = seq_len - keep_count
            if evicted > 0:
                self._evicted[rid] = self._evicted.get(rid, 0) + evicted
                request.py_tri_evicted = self._evicted[rid]
            return
        # PER-LAYER eviction: each layer scores its OWN cached K_rot, aggregates
        # heads by mean, keeps its own budget + recency window, compacted
        # INDEPENDENTLY. Every layer keeps the SAME COUNT (top_B), so the
        # per-request num_cached is consistent across layers; only the kept SET
        # differs. Kept K retains its original RoPE rotation.
        # root-cause probe (remove before PR): capture eviction INPUT (kh_in,
        # round_start, seq_len) here; kept-set + post-compact K (layer 0) after the
        # loop. Run twice under overlap + diff: round_start/seq_len differ =>
        # scheduler race; kh_in differs => input-K race; kept differs (same kh_in)
        # => SELECT non-determinism; kh_out differs (same kept) => COMPACT write race.
        _probe_kh_in = None
        _probe_keep0 = None
        if _TRI_ROOTPROBE:
            _k0 = self._read_request_k(request, 0, seq_len)
            _probe_kh_in = float(_k0.float().sum().item()) if _k0 is not None else -1.0
        keep_count = None
        for layer_idx in range(num_layers):
            if self.use_triton:
                # Triton path: one helper fuses paged-read + score + reduce +
                # select + compact for this layer.
                keep = self._maybe_evict_layer_triton(
                    request, layer_idx, seq_len, round_start
                )
                if keep is not None:
                    keep_count = int(keep.numel())
            else:
                # PyTorch reference path (the A/B baseline; verified token-identical
                # to the Triton path).
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
            if _TRI_ROOTPROBE and layer_idx == 0:
                _probe_keep0 = keep
            if _TRI_KEEPPROBE and layer_idx == 0 and keep is not None:
                self._log_keepprobe("per_layer", rid,
                                    self._gen_steps.get(rid, -1), seq_len, 0,
                                    keep)
        if keep_count is None:
            return
        if _TRI_ROOTPROBE and _probe_keep0 is not None:
            _kp = self._read_request_k(request, 0, int(_probe_keep0.numel()))
            _kh_out = float(_kp.float().sum().item()) if _kp is not None else -1.0
            with open("/scratch/triattn_e2e/rootprobe.jsonl", "a") as _f:
                _f.write(f"{rid}\t{self._gen_steps.get(rid, -1)}\t{round_start}\t{seq_len}\t"
                         f"{(_probe_kh_in if _probe_kh_in is not None else -1.0):.5f}\t"
                         f"{int(_probe_keep0.numel())}\t{float(_probe_keep0.float().sum().item()):.1f}\t"
                         f"{_kh_out:.5f}\n")
        evicted = seq_len - keep_count
        if evicted > 0:
            # cumulative count consumed by the every-step reconcile (history
            # shrink) above + the metadata shim (num_cached clamp). seq_len is
            # the full committed length, so evicted_cum is exact.
            self._evicted[rid] = self._evicted.get(rid, 0) + evicted
            request.py_tri_evicted = self._evicted[rid]

    # --- Upstream-faithful eviction modes (per_head / per_layer_perhead / union) ---
    #
    # These reproduce github.com/WeianMao/triattention's selection. The key
    # differences from the per_layer path: scores are NOT averaged over heads
    # (each KV head keeps its own token set), they are z-normalized per head over
    # the decode region, the prompt (prefill) tokens are pinned, and there is no
    # recency window. The kept COUNT stays uniform (= top_B) so paged attention
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
        std = head_scores.std(dim=1, unbiased=False,
                              keepdim=True).clamp_min(1e-6)
        return (head_scores - mean) / std

    def _group_heads_to_kv_max(self, head_scores: "torch.Tensor",
                               num_kv_heads: int) -> "torch.Tensor":
        """Reduce per-query-head scores ``[H, seq]`` to per-KV-head ``[nkv, seq]``
        by MAX over the query heads sharing each KV head (upstream within-group
        aggregation). Query heads group contiguously: head ``q`` -> KV head
        ``q // (H // nkv)`` (matches the ``q*nkv//H`` GQA map in scoring and
        upstream's ``head // num_key_value_groups``)."""
        num_q_heads, seq = head_scores.shape
        group = max(1, num_q_heads // num_kv_heads)
        return head_scores.view(num_kv_heads, group, seq).max(dim=1).values

    def _decode_topk(self, scores: "torch.Tensor", decode_start: int,
                     decode_budget: int) -> "torch.Tensor":
        """Pick the top-``decode_budget`` decode tokens by ``scores``
        (``[decode_count]``); return ABSOLUTE slot indices (``decode_start``
        added) prefixed with the pinned prompt slots ``[0, decode_start)``,
        sorted ascending (compaction reorders by these)."""
        k = min(decode_budget, int(scores.numel()))
        decode_idx = torch.topk(scores, k, largest=True).indices + decode_start
        prefill_idx = torch.arange(decode_start, device=scores.device,
                                   dtype=torch.long)
        return torch.sort(torch.cat([prefill_idx, decode_idx])).values

    def _layer_head_scores(self, request: "LlmRequest", layer_idx: int,
                           seq_len: int,
                           round_start: int) -> Optional["torch.Tensor"]:
        """One layer's per-query-head scores ``[H, seq]``. With ``use_triton`` the
        heavy paged-K read + score runs on the Triton score kernel (the SAME
        kernel the per_layer Triton path uses, before its head-mean); else the
        PyTorch ``_score_layer``. The new modes' downstream selection +
        compaction are shared torch code, so the only Triton-vs-PyTorch
        difference is this scoring source (bit-exact-verified). None if
        unreadable."""
        if self.use_triton:
            get_buffers = getattr(self.kv_cache_manager, "get_buffers", None)
            if get_buffers is None:
                return None
            pool = get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                return None
            page_ids = self._resolve_page_ids(request, layer_idx)
            if not page_ids:
                return None
            device = pool.device
            page_ids_t = torch.as_tensor(page_ids, device=device,
                                         dtype=torch.int64)
            if self._offsets is None:
                self._offsets = _build_geometric_offsets(
                    self._offset_max_length, device)
            return triton_tri_score(
                pool, page_ids_t,
                self._tri_q_real[layer_idx], self._tri_q_imag[layer_idx],
                self._tri_mlr_coef[layer_idx],
                self._freq_scale_sq, self.calibration["omega"], self._offsets,
                float(round_start), seq_len, self._H,
                score_aggregation=self.score_aggregation)
        k_rot = self._read_request_k(request, layer_idx, seq_len)
        if k_rot is None:
            return None
        return self._score_layer(k_rot, None, layer_idx, round_start)

    def _evict_modes(self, request: "LlmRequest", num_layers: int,
                     seq_len: int, round_start: int,
                     precomputed: Optional[List["torch.Tensor"]] = None,
                     compact: bool = True
                     ) -> Optional[Union[int, "torch.Tensor"]]:
        """Score every layer, then select + physically compact per
        ``self.eviction_mode``. Returns the uniform kept count (= ``top_B``, or
        ``seq_len`` if smaller), or None if nothing could be scored.

        ``precomputed``: optional per-layer ``[H, seq_len]`` scores (from the
        batched score kernel); when given, scoring is skipped and these are used
        verbatim, so the batched path is byte-identical to the per-request one.

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
            decode_start = min(int(getattr(request, "py_prompt_len", 0) or 0),
                               seq_len)
        decode_count = seq_len - decode_start
        decode_budget = budget - decode_start
        # Budget exhausted by the pinned prompt (or no decode tokens): keep the
        # first `budget` contiguous slots everywhere; nothing per-head to do.
        if decode_budget <= 0 or decode_count <= 0:
            return budget

        # Score every layer's cached K. For per_head / per_layer_perhead reduce
        # to per-KV-head [nkv, decode_count] (group-max), z-normalized over the
        # decode region. For union keep the raw per-head rows to stack.
        # num KV heads from the pool (HND dim 2); needed to group query-head
        # scores. Works for both the torch and Triton scoring paths.
        get_buffers = getattr(self.kv_cache_manager, "get_buffers", None)
        p0 = get_buffers(0, kv_layout="HND") if get_buffers is not None else None
        if p0 is None:
            return None
        num_kv_heads = int(p0.shape[2])
        per_layer_kv_scores: List[torch.Tensor] = []
        union_rows: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            if precomputed is not None:
                head_scores = precomputed[layer_idx]
            else:
                head_scores = self._layer_head_scores(request, layer_idx,
                                                      seq_len, round_start)
            if head_scores is None:
                return None
            decode_scores = self._zscore_decode(
                head_scores[:, decode_start:seq_len])
            if self.eviction_mode == "union":
                union_rows.append(decode_scores)
            else:
                per_layer_kv_scores.append(
                    self._group_heads_to_kv_max(decode_scores, num_kv_heads))

        if self.eviction_mode == "union":
            return self._evict_union(request, num_layers, seq_len, decode_start,
                                     decode_budget, torch.cat(union_rows, dim=0),
                                     compact=compact)
        if self.eviction_mode == "per_head":
            return self._evict_per_head(request, num_layers, seq_len,
                                        decode_start, decode_budget,
                                        per_layer_kv_scores, num_kv_heads)
        if self.eviction_mode == "per_layer_perhead":
            return self._evict_per_layer_perhead(request, seq_len, decode_start,
                                                 decode_budget,
                                                 per_layer_kv_scores,
                                                 num_kv_heads)
        return None  # defensive; the config Literal should prevent this

    def _evict_per_head(self, request: "LlmRequest", num_layers: int,
                        seq_len: int, decode_start: int, decode_budget: int,
                        per_layer_kv_scores: List["torch.Tensor"],
                        num_kv_heads: int) -> int:
        """per_head: each KV head's score = MEAN over layers of the per-layer MAX
        (upstream ``_select_per_head_independent``). One keep set per KV head,
        applied to EVERY layer."""
        agg = torch.stack(per_layer_kv_scores, dim=0).mean(dim=0)  # [nkv, dc]
        keep_2d = torch.stack(
            [self._decode_topk(agg[h], decode_start, decode_budget)
             for h in range(num_kv_heads)], dim=0)  # [nkv, keep_count]
        if _TRI_KEEPPROBE:
            rid = request.py_request_id
            self._log_keepprobe("per_head", rid,
                                self._gen_steps.get(rid, -1), seq_len,
                                decode_start, keep_2d)
        for layer_idx in range(num_layers):
            self._evict_layer_perhead(request, layer_idx, keep_2d, seq_len)
        return int(keep_2d.shape[1])

    def _evict_per_layer_perhead(self, request: "LlmRequest", seq_len: int,
                                 decode_start: int, decode_budget: int,
                                 per_layer_kv_scores: List["torch.Tensor"],
                                 num_kv_heads: int) -> Optional[int]:
        """per_layer_perhead: each (layer, KV head) selects independently from
        that layer's per-KV-head max (upstream
        ``_select_per_layer_perhead_independent``)."""
        keep_count = None
        for layer_idx, kv_scores in enumerate(per_layer_kv_scores):
            keep_2d = torch.stack(
                [self._decode_topk(kv_scores[h], decode_start, decode_budget)
                 for h in range(num_kv_heads)], dim=0)
            if _TRI_KEEPPROBE and layer_idx == 0:
                rid = request.py_request_id
                self._log_keepprobe("per_layer_perhead", rid,
                                    self._gen_steps.get(rid, -1), seq_len,
                                    decode_start, keep_2d)
            self._evict_layer_perhead(request, layer_idx, keep_2d, seq_len)
            keep_count = int(keep_2d.shape[1])
        return keep_count

    def _evict_union(self, request: "LlmRequest", num_layers: int, seq_len: int,
                     decode_start: int, decode_budget: int,
                     head_matrix: "torch.Tensor",
                     compact: bool = True) -> Union[int, "torch.Tensor"]:
        """union: union of every head's top-k, re-ranked by the per-token max
        (upstream ``_select_union_based``). One 1-D keep set for every layer.

        ``compact=True`` (default): physically compact every layer in place and
        return the kept count -- unchanged behavior. ``compact=False``: skip the
        per-layer compaction and return the 1-D sorted keep tensor (slot indices
        in ``[0, seq_len)``) so the caller can batch the compaction itself. The
        keep tensor is computed by exactly the same code in both cases, so the
        kept-slot set is byte-identical regardless of ``compact``."""
        combined = head_matrix.max(dim=0).values  # [decode_count]
        keep_1d = self._select_union(head_matrix, combined, decode_budget)
        prefill_idx = torch.arange(decode_start, device=combined.device,
                                   dtype=torch.long)
        keep = torch.sort(
            torch.cat([prefill_idx, keep_1d + decode_start])).values
        if _TRI_KEEPPROBE:
            rid = request.py_request_id
            self._log_keepprobe("union", rid, self._gen_steps.get(rid, -1),
                                seq_len, decode_start, keep)
        if not compact:
            # Keep-only: caller compacts (batched per-layer over all requests).
            return keep
        for layer_idx in range(num_layers):
            self._evict_layer(request, layer_idx, keep, seq_len)
        return int(keep.numel())

    def _select_union(self, per_head_scores: "torch.Tensor",
                      combined: "torch.Tensor",
                      keep_count: int) -> "torch.Tensor":
        """Each head picks its top-``keep_count``; take the union; from the union
        keep the top-``keep_count`` by ``combined``; if the union is smaller,
        fill from the highest-scoring remaining tokens. Decode-relative
        indices."""
        n = int(combined.shape[0])
        if n <= keep_count:
            return torch.arange(n, device=combined.device, dtype=torch.long)
        union_mask = torch.zeros(n, device=combined.device, dtype=torch.bool)
        quota = min(keep_count, n)
        # Batched top-k over all (layer x head) rows at once. torch.topk along
        # dim=1 computes each row's top-quota independently -- byte-identical to
        # the per-row Python loop (top-k is row-independent and the union_mask OR
        # is order-independent), but collapses H iterations + H kernel launches
        # into one. H = num_layers * num_q_heads (1152 for Qwen3-8B), so per
        # request this loop was the dominant high-BS eviction cost (85% of the
        # eviction dispatch: 4.6M topk launches at BS64). Validated byte-identical.
        top_idx = torch.topk(per_head_scores, quota, dim=1, largest=True).indices
        union_mask[top_idx.reshape(-1)] = True
        union_idx = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_idx.numel() >= keep_count:
            subset = combined.index_select(0, union_idx)
            top_subset = torch.topk(subset, keep_count, largest=True).indices
            return union_idx.index_select(0, torch.sort(top_subset).values)
        remaining = keep_count - int(union_idx.numel())
        if remaining > 0:
            residual = combined.clone()
            residual[union_mask] = float("-inf")
            extra = torch.topk(residual,
                               min(remaining, n - int(union_idx.numel())),
                               largest=True).indices
            union_idx = torch.cat([union_idx, extra])
        return torch.sort(union_idx).values

    def _evict_layer_perhead(self, request: "LlmRequest", layer_idx: int,
                             keep_2d: "torch.Tensor", seq_len: int) -> None:
        """Physically compact ONE layer's cache, keeping a DIFFERENT token set per
        KV head. ``keep_2d`` is ``[num_kv_heads, keep_count]`` slot indices in
        ``[0, seq_len)``; each KV head's token axis is reordered independently
        (``[kept..., dropped...]``) so every slot in ``[0, seq_len)`` still holds
        a real key/value. Same HND layout + no-re-RoPE reasoning as
        ``_evict_layer`` -- only the reorder is now per-head (a gather on the
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
        page_ids_t = torch.as_tensor(page_ids, device=pool.device,
                                     dtype=torch.long)
        request_pages = pool[page_ids_t]
        num_pages, kv_factor, num_kv_heads, _, head_dim = request_pages.shape
        keep_2d = keep_2d.to(request_pages.device)
        # [kv_factor, num_kv_heads, num_pages * tokens_per_block, head_dim]
        kv_by_token = (
            request_pages.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(kv_factor, num_kv_heads, num_pages * tokens_per_block,
                     head_dim)
        )
        # Per-head full token permutation [num_kv_heads, seq_len]: kept slots
        # first (in their given order), then the dropped slots -- so no slot in
        # [0, seq_len) is left holding stale data.
        all_token_ids = torch.arange(seq_len, device=kv_by_token.device,
                                     dtype=torch.long)
        new_orders: List[torch.Tensor] = []
        for h in range(num_kv_heads):
            is_dropped = torch.ones(seq_len, device=kv_by_token.device,
                                    dtype=torch.bool)
            is_dropped[keep_2d[h]] = False
            new_orders.append(
                torch.cat([keep_2d[h], all_token_ids[is_dropped]]))
        new_order = torch.stack(new_orders, dim=0)  # [num_kv_heads, seq_len]
        # Gather the token axis (dim 2) with a DIFFERENT order per KV head.
        region = kv_by_token[:, :, :seq_len]
        idx = new_order.view(1, num_kv_heads, seq_len, 1).expand(
            kv_factor, num_kv_heads, seq_len, head_dim)
        reordered = torch.gather(region, 2, idx).clone()
        kv_by_token[:, :, :seq_len] = reordered
        num_touched_pages = (seq_len + tokens_per_block - 1) // tokens_per_block
        repaged = (
            kv_by_token.reshape(kv_factor, num_kv_heads, num_pages,
                                tokens_per_block, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        pool[page_ids_t[:num_touched_pages]] = repaged[:num_touched_pages]

    def _evict_batch_modes(self, evict_reqs, num_layers: int) -> None:
        """Batched form of ``_evict_modes`` (per_head / per_layer_perhead /
        union): ONE batched per-head score launch over (request x layer) via
        ``triton_tri_score_batched_perhead``, then per-request selection +
        compaction reusing ``_evict_modes(precomputed=...)`` -- so the result is
        byte-identical to the per-request Triton path (only the score LAUNCH is
        fused across requests; selection/compaction are the same torch code).
        Falls back to per-request ``_maybe_evict`` if pools are unreadable."""
        from .triattention_kernels import (triton_tri_score_batched_perhead,
                                           flat_perhead_to_list)
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        layer_pools = [get_buffers(l, kv_layout="HND") for l in range(num_layers)]
        if any(p is None for p in layer_pools):
            for request, rid in evict_reqs:
                self._maybe_evict(request, rid, num_layers)
            return
        device = layer_pools[0].device
        if _TRI_PROFILE:
            import time as _tm3; _pt = _tm3.perf_counter()
        # Per-request committed-trim + page ids (identical to _evict_batch).
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
            kept.append((request, rid,
                         torch.as_tensor(page_ids, device=device,
                                         dtype=torch.int64),
                         int(seq_len), float(round_start)))
        if not kept:
            return
        page_ids_list = [k[2] for k in kept]
        seq_lens = [k[3] for k in kept]
        round_starts = [k[4] for k in kept]
        if _TRI_PROFILE:
            self._t_trim = getattr(self, "_t_trim", 0.0) + (_tm3.perf_counter() - _pt)
            _pt = _tm3.perf_counter()
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length,
                                                     device)
        # Batched per-head score, grouped by storage (VSWA multi-pool safe),
        # exactly like _evict_batch but emitting [H, seq] per (layer, request).
        from collections import defaultdict as _defaultdict
        storage_groups = _defaultdict(list)
        for l in range(num_layers):
            storage_groups[layer_pools[l].untyped_storage().data_ptr()].append(l)
        req_layer_scores = [dict() for _ in kept]  # [req] -> {layer_idx: [H,seq]}
        for lids in storage_groups.values():
            ph, so, sm = triton_tri_score_batched_perhead(
                layer_pools, page_ids_list, seq_lens, round_starts,
                self._tri_q_real, self._tri_q_imag, self._tri_mlr_coef,
                self._freq_scale_sq, self.calibration["omega"], self._offsets,
                self._H, score_aggregation=self.score_aggregation,
                layer_indices=lids)
            seg_list = flat_perhead_to_list(ph, so)
            for s, meta in enumerate(sm):
                req_layer_scores[meta.request_index][meta.layer_index] = \
                    seg_list[s]
        if _TRI_PROFILE:
            self._t_score = getattr(self, "_t_score", 0.0) + (_tm3.perf_counter() - _pt)
            _pt = _tm3.perf_counter()
        # Per request: select from the precomputed scores. All scores were taken
        # BEFORE any compaction; requests touch disjoint pages, so compacting one
        # does not disturb another's (already-read) scores.
        #
        # P1 (re-instated 2026-06-13): a timer measurement showed the per-request
        # per-layer compaction loop (K*L*rounds Python iterations) is the DOMINANT
        # high-BS eviction cost (239s of tri's 263s BS64 overhead; per-step reconcile
        # was only 0.6s). For union (1-D layer-uniform keep) we compute keep WITHOUT
        # compacting, accumulate per layer, then run ONE batched compaction per layer
        # over all requests -> K*L*2 compact launches collapse to L*2. Bit-exact
        # (kernel-equivalence validated: batched compact == per-request, byte-identical).
        is_union = self.eviction_mode == "union"
        union_by_layer = {} if is_union else None
        for r, (request, rid, _pi, seq_len, round_start) in enumerate(kept):
            precomputed = [req_layer_scores[r].get(l)
                           for l in range(num_layers)]
            if any(p is None for p in precomputed):
                self._maybe_evict(request, rid, num_layers)
                continue
            keep_count = self._evict_modes(request, num_layers, seq_len,
                                           round_start, precomputed=precomputed,
                                           compact=not is_union)
            if keep_count is None:
                continue
            if is_union and isinstance(keep_count, torch.Tensor):
                keep = keep_count
                for lid in range(num_layers):
                    grp = union_by_layer.setdefault(lid, ([], [], []))
                    grp[0].append(page_ids_list[r]); grp[1].append(keep); grp[2].append(seq_len)
                keep_count = int(keep.numel())
            evicted = seq_len - keep_count
            if evicted > 0:
                self._evicted[rid] = self._evicted.get(rid, 0) + evicted
                request.py_tri_evicted = self._evicted[rid]
        if _TRI_PROFILE:
            self._t_select = getattr(self, "_t_select", 0.0) + (_tm3.perf_counter() - _pt)
            _pt = _tm3.perf_counter()
        if union_by_layer:
            from .triattention_kernels import triton_tri_compact_batched
            for lid, (pl, kl, sl) in union_by_layer.items():
                triton_tri_compact_batched(layer_pools[lid], pl, kl, sl)
        if _TRI_PROFILE:
            self._t_compact = getattr(self, "_t_compact", 0.0) + (_tm3.perf_counter() - _pt)

    def _evict_batch(self, evict_reqs, num_layers: int) -> None:
        """Batched form of ``_maybe_evict`` over ALL requests evicting this step:
        one fused score launch over (request x layer), one histogram-topk select,
        per-layer compaction (batched over requests). Keep count is layer-uniform
        per request, exactly like the per-request path; updates ``self._evicted``
        / ``request.py_tri_evicted`` identically. Falls back to per-request if the
        layer pools are unreadable."""
        # The non-per_layer modes keep a different token set per KV head; route
        # them to the per-head batched path.
        if self.eviction_mode != "per_layer":
            return self._evict_batch_modes(evict_reqs, num_layers)
        from .triattention_kernels import (triton_tri_score_batched,
                                           flat_scores_to_list,
                                           triton_tri_select_batched,
                                           triton_tri_compact_batched)
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return
        layer_pools = [get_buffers(l, kv_layout="HND") for l in range(num_layers)]
        if any(p is None for p in layer_pools):
            for request, rid in evict_reqs:
                self._maybe_evict(request, rid, num_layers)
            return
        device = layer_pools[0].device
        _prof = _TRI_PROFILE
        if _prof:
            import time as _tm
            _ph0 = _tm.perf_counter_ns()

        # Per-request metadata. page_ids are shared across layers in V2 (one block
        # table per request; layers share the physical page). seq_len is committed-
        # trimmed exactly as _maybe_evict; round_start = max_beam-1.
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
            kept.append((request, rid,
                         torch.as_tensor(page_ids, device=device, dtype=torch.int64),
                         int(seq_len), float(round_start)))
        if not kept:
            return

        page_ids_list = [k[2] for k in kept]
        seq_lens = [k[3] for k in kept]
        round_starts = [k[4] for k in kept]

        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)

        if _prof:
            _phost = (_tm.perf_counter_ns() - _ph0) / 1e6
            _pe = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            _pe[0].record()

        # (1) batched fused score over (request x layer). V2 allocates per-layer
        # / per-pool-group KV pools that do NOT all share one storage buffer, and
        # the fused score addresses layers via offsets off ONE storage base, so
        # group layers by their storage and score each group in one launch (still
        # batched over ALL requests -- the launch count is per-group, not per-req).
        from collections import defaultdict as _defaultdict
        storage_groups = _defaultdict(list)
        for l in range(num_layers):
            storage_groups[layer_pools[l].untyped_storage().data_ptr()].append(l)
        score_list = []          # per (layer, request) score
        owner = []               # (layer_index, request_index) for each score
        for lids in storage_groups.values():
            fs, so, sm = triton_tri_score_batched(
                layer_pools, page_ids_list, seq_lens, round_starts,
                self._tri_q_real, self._tri_q_imag, self._tri_mlr_coef,
                self._freq_scale_sq, self.calibration["omega"], self._offsets,
                self._H, score_aggregation=self.score_aggregation,
                layer_indices=lids)
            sl = flat_scores_to_list(fs, so)
            for s, meta in enumerate(sm):
                score_list.append(sl[s])
                owner.append((meta.layer_index, meta.request_index))

        # (2) ONE batched recency + histogram top-k select over all (layer x req).
        if _prof:
            _pe[1].record()
        keeps = triton_tri_select_batched(score_list, self.top_B, self.window_size)

        # (3) regroup by layer; per-layer batched compaction. keep_count is
        # layer-uniform, so any segment of a request gives its kept count.
        if _prof:
            _pe[2].record()
        by_layer = {}
        keep_count = {}  # request_index -> kept token count
        for idx, (lid, r) in enumerate(owner):
            grp = by_layer.setdefault(lid, ([], [], []))
            grp[0].append(page_ids_list[r])
            grp[1].append(keeps[idx])
            grp[2].append(seq_lens[r])
            keep_count[r] = int(keeps[idx].numel())
        for lid, (pl, kl, sl) in by_layer.items():
            triton_tri_compact_batched(layer_pools[lid], pl, kl, sl)
        if _prof:
            _pe[3].record()

        # (4) bookkeeping (same semantics as _maybe_evict): cumulative evicted,
        # consumed by the every-step history reconcile + the num_cached clamp.
        for r, (request, rid, _pi, seq_len, _rs) in enumerate(kept):
            kc = keep_count.get(r)
            if kc is None:
                continue
            evicted = seq_len - kc
            if evicted > 0:
                self._evicted[rid] = self._evicted.get(rid, 0) + evicted
                request.py_tri_evicted = self._evicted[rid]

        if _prof:
            torch.cuda.synchronize()
            import json as _js
            os.makedirs(os.path.dirname(_TRI_PROF_FILE), exist_ok=True)
            with open(_TRI_PROF_FILE, "a") as _f:
                _f.write(_js.dumps(dict(
                    nseg=len(score_list), K=len(kept), seq_max=max(seq_lens),
                    host_ms=_phost,
                    score_ms=_pe[0].elapsed_time(_pe[1]),
                    select_ms=_pe[1].elapsed_time(_pe[2]),
                    compact_ms=_pe[2].elapsed_time(_pe[3]))) + "\n")

        if _TRI_DETTRACE:
            import json as _jd
            os.makedirs(os.path.dirname(_TRI_DETTRACE_FILE), exist_ok=True)
            with open(_TRI_DETTRACE_FILE, "a") as _df:
                for _r, (_rq, _rid, _pi, _sl, _rs) in enumerate(kept):
                    _kc = keep_count.get(_r)
                    _df.write(_jd.dumps(dict(
                        step=int(_rq.max_beam_num_tokens), rid=int(_rid),
                        seq_len=int(_sl),
                        kept=(int(_kc) if _kc is not None else -1),
                        evicted=int(self._evicted.get(_rid, 0)))) + "\n")

    def _maybe_evict_layer_triton(self, request, layer_idx, seq_len, round_start):
        """Triton A/B path for one layer of _maybe_evict: fused paged-read+score,
        head-mean, recency+topk select, in-place compaction. Returns the kept
        slot indices (sorted ascending), or None if the pool is unreadable.

        Mirrors the PyTorch branch exactly; the per-block kernels are verified
        allclose / bit-exact against _score_layer / _select_with_recency /
        _evict_layer by the standalone GPU equivalence test.
        """
        mgr = self.kv_cache_manager
        get_buffers = getattr(mgr, "get_buffers", None)
        if get_buffers is None:
            return None
        pool = get_buffers(layer_idx, kv_layout="HND")
        if pool is None:
            return None
        page_ids = self._resolve_page_ids(request, layer_idx)
        if not page_ids:
            return None
        device = pool.device
        page_ids_t = torch.as_tensor(page_ids, device=device, dtype=torch.int64)
        # Build the geometric offsets once, on the pool's device (mirrors the
        # set-before-use ordering in _score_layer).
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        if not getattr(self, "_tri_path_logged", False):
            logger.info("TriAttention: Triton eviction path ACTIVE "
                        "(score + reduce + compact on Triton; select on torch.topk)")
            self._tri_path_logged = True
        # (1)+(2): fused paged-K read + score, then mean over heads -> [seq].
        head_scores = triton_tri_score(
            pool, page_ids_t,
            self._tri_q_real[layer_idx], self._tri_q_imag[layer_idx],
            self._tri_mlr_coef[layer_idx],
            self._freq_scale_sq, self.calibration["omega"],
            self._offsets,
            float(round_start), seq_len, self._H,
            score_aggregation=self.score_aggregation,
        )
        layer_score = triton_tri_reduce_heads(head_scores)
        # (3) recency window + top-k via the vendored RocketKV histogram top-k
        # (all-Triton). (4) physical compaction in place on Triton.
        keep = triton_tri_select(layer_score, seq_len, self.top_B, self.window_size)
        triton_tri_compact(pool, page_ids_t, keep, seq_len)
        return keep

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

        from .triattention_calibration import (
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
        # PR-15106 does NOT set ``metadata.compression_manager`` (and its hooks
        # fire post-forward), so read the active manager from the module global
        # set in ``TriAttention.__init__``. Fall back to the metadata attribute
        # if a future framework ever wires it. Keeps the reconcile self-contained
        # in this package -- no pyexecutor / framework edits.
        cm = _ACTIVE_TRI_MANAGER or getattr(self, "compression_manager", None)
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
                    # model_engine is NOT eviction-aware on PR-15106 (we don't
                    # touch it), so it leaves num_cached = max_beam-1. Do the FULL
                    # reconcile here: num_cached = (max_beam-1) + 1 - ev =
                    # max_beam - ev, matching the cache manager's compacted
                    # history. (ev==0 path is untouched, so dense/no-evict steps
                    # keep the stock max_beam-1 -> byte-identical.)
                    kvp.num_cached_tokens_per_seq[i] = int(kvp.num_cached_tokens_per_seq[i]) + 1 - ev
            if _TRI_FWDPROBE:
                # forward-side probe (remove before PR): log the num_cached the
                # attention kernel actually uses, per (gen step, request), in the
                # divergence window. Run overlap vs disable + diff: if num_cached
                # varies run-to-run (or across the identical-prompt requests within
                # a run) at the divergence step => the post-eviction forward's
                # cache view is racy under overlap (the real root cause).
                _fp = f"/scratch/triattn_e2e/fwdprobe_{os.environ.get('SLURM_JOB_ID','x')}_{os.getpid()}.jsonl"
                with open(_fp, "a") as _f:
                    for _i in range(num_contexts, num_requests):
                        _rid = req_ids[_i]
                        _st = e._gen_steps.get(_rid, -1)
                        if 120 <= _st <= 230:
                            _f.write(f"{_st}\t{_rid}\t{e.evicted_count(_rid)}\t"
                                     f"{int(kvp.num_cached_tokens_per_seq[_i])}\n")
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
