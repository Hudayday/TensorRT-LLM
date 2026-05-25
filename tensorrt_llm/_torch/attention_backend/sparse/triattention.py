"""TriAttention sparse-attention method (form-III periodic eviction).

TriAttention performs periodic generation-phase KV eviction guided by a
trigonometric importance score computed from offline-calibrated statistics of
the model's Q-pre-RoPE vectors (paper §4.1, §4.3). It is a form-III method:
no work in context phase, no per-step attention-time mask, and a single
``on_generation_step_end`` hook that fires every ``beta`` steps to physically
evict blocks below the top-B keep set.

Calibration is computed offline via
``triattention_calibration.compute_triattention_calibration`` and loaded once
at LLM init time. See the calibration module for the statistic schema.
"""

from typing import TYPE_CHECKING, ClassVar, Dict, Optional

import torch

from tensorrt_llm._torch.attention_backend.sparse.sparse_attention_manager import \
    SparseAttentionManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
        ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
# See triattention_calibration.compute_triattention_calibration for shapes.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "R", "omega", "phi"})


class TriAttention(SparseAttentionManager):
    """Form-III periodic KV eviction driven by trigonometric importance scoring.

    Overrides only ``on_generation_step_end``: every ``beta`` generation steps,
    reads the current K cache through the underlying ``KVCacheManagerV2``,
    computes a per-token importance score using offline-calibrated stats, and
    physically evicts blocks below the top-B keep set. All other hooks remain
    no-op (form-III does not need context-phase or per-attention work).
    """

    # TriAttention physically evicts tokens during decode based on per-request
    # query / step state, so the resulting cache is not safe to reuse across
    # requests (a different request would have evicted a different token set).
    supports_kv_cache_reuse: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        top_B: int,
        beta: int = 128,
        calibration_path: Optional[str] = None,
    ):
        super().__init__(kv_cache_manager)
        self.top_B = top_B
        self.beta = beta
        if calibration_path is None:
            raise ValueError(
                "TriAttention requires calibration_path; compute offline via "
                "triattention_calibration.compute_triattention_calibration "
                "before LLM init.")
        self.calibration: Dict[str, torch.Tensor] = self._load_calibration(
            calibration_path)

    # ------------------------------------------------------------------ #
    # Override: per-step end                                             #
    # ------------------------------------------------------------------ #

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Triggered after every generation step's full forward across all
        layers. Fires the actual eviction every ``self.beta`` steps; other
        steps return immediately.

        Per in-flight generation request:
          1. Read the current K cache through ``self.kv_cache_manager``.
          2. Score each token with the trigonometric importance function
             (paper §4.3) using ``self.calibration``.
          3. Select top-B tokens to keep; physically free the rest via
             ``self.kv_cache_manager``.
        """
        # TODO(Phase 3 M3.1): implement
        #   - step counter (per-request? or global per-batch?)
        #   - if step % self.beta != 0: return
        #   - for req in scheduled_batch.generation_requests:
        #       k = self.kv_cache_manager.read_k_block(req)
        #       scores = self._trigonometric_score(k, self.calibration, ...)
        #       keep = torch.topk(scores, self.top_B).indices
        #       evict = self._invert(keep, total=len(k))
        #       self.kv_cache_manager.free_blocks(req, evict)
        raise NotImplementedError(
            "TriAttention.on_generation_step_end algorithm pending; "
            "see paper §4.3 and design doc 15 §3.4.5.")

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

    def _validate_calibration(
            self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys.

        Shape consistency checks against the running model's L / H / D are
        deferred to the eviction kernel (paper §4.3 algorithm); the skeleton
        only enforces key presence.
        """
        missing = _REQUIRED_CALIBRATION_KEYS - set(calibration.keys())
        if missing:
            raise ValueError(
                f"TriAttention calibration is missing keys: {sorted(missing)}; "
                f"got {sorted(calibration.keys())}.")
