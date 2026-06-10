"""Unit tests for the TriAttention SparseAttentionManager pipeline.

Covers the wiring landed in commits e0a29e3b50, 79cf1adcd9, 6b9719bc10:

- Pydantic discriminator dispatch via ``SparseAttentionConfig`` Annotated Union.
- ``BaseSparseAttentionConfig.is_behavior_layer_method`` property semantics.
- ``SparseAttentionManager`` base-class default hook contract (no-op return
  values, ``supports_kv_cache_reuse`` capability).
- ``create_sparse_attention_manager`` factory dispatch + ``KVCacheManagerV2``
  isinstance assertion.
- ``TriAttention`` construction + calibration loading + missing-key validation.

The tests intentionally do *not* exercise actual eviction / attention forward
behavior; those require model weights and are out of scope for this
configuration-layer unit test.
"""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from tensorrt_llm._torch.attention_backend.sparse import (
    SparseAttentionManager,
    create_sparse_attention_manager,
)
from tensorrt_llm._torch.attention_backend.sparse.triattention import TriAttention
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    RocketSparseAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
    TriAttentionConfig,
)

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TriAttention._load_calibration uses map_location='cuda'",
)


@pytest.fixture
def dummy_calibration_pt(tmp_path):
    """Build a minimal valid calibration ``.pt`` with the required key set."""
    path = tmp_path / "tri_calib.pt"
    # Corrected runtime schema (doc 48 §2): only E_q, E_q_norm, omega and
    # freq_scale_sq are consumed at runtime (R was never read; phi is
    # recomputed; freq_scale_sq is the per-frequency RoPE amplitude scale).
    calibration = {
        "E_q": torch.zeros(2, 2, 4, dtype=torch.complex64),
        "E_q_norm": torch.ones(2, 2, 4, dtype=torch.float32),
        "omega": torch.arange(4, dtype=torch.float32),
        "freq_scale_sq": torch.ones(4, dtype=torch.float32),
    }
    torch.save(calibration, path)
    return str(path)


# ---------------------------------------------------------------------------
# is_behavior_layer_method property semantics
# ---------------------------------------------------------------------------


class TestIsBehaviorLayerMethod:
    """The property routes dispatch in _util.py and trtllm.py; semantics
    must stay frozen so legacy methods cannot accidentally opt in."""

    def test_rocket_is_legacy(self):
        assert RocketSparseAttentionConfig().is_behavior_layer_method is False

    def test_dsa_is_legacy(self):
        assert DeepSeekSparseAttentionConfig().is_behavior_layer_method is False

    def test_skip_softmax_is_legacy(self):
        assert SkipSoftmaxAttentionConfig().is_behavior_layer_method is False

    def test_triattention_is_memory_layer(self):
        # ships its own attn shim -> classified memory-layer (flag False); the
        # eviction still runs in a SparseAttentionManager compression manager.
        cfg = TriAttentionConfig(calibration_path="/tmp/dummy.pt")
        assert cfg.is_behavior_layer_method is False


# ---------------------------------------------------------------------------
# TriAttentionConfig Pydantic validation
# ---------------------------------------------------------------------------


class TestTriAttentionConfig:
    def test_default_algorithm(self):
        cfg = TriAttentionConfig(calibration_path="/tmp/x.pt")
        assert cfg.algorithm == "triattention"

    def test_calibration_path_required(self):
        with pytest.raises(ValidationError):
            TriAttentionConfig()  # calibration_path has no default

    def test_field_defaults(self):
        cfg = TriAttentionConfig(calibration_path="/tmp/x.pt")
        assert cfg.top_B == 1024
        assert cfg.beta == 128

    def test_field_overrides(self):
        cfg = TriAttentionConfig(top_B=256, beta=64, calibration_path="/tmp/x.pt")
        assert cfg.top_B == 256
        assert cfg.beta == 64

    def test_supports_only_pytorch_backend(self):
        cfg = TriAttentionConfig(calibration_path="/tmp/x.pt")
        assert cfg.supports_backend("pytorch") is True
        assert cfg.supports_backend("tensorrt") is False


# ---------------------------------------------------------------------------
# SparseAttentionConfig Annotated Union discriminator dispatch
# ---------------------------------------------------------------------------


class TestUnionDiscriminator:
    """Pydantic ``Field(discriminator="algorithm")`` is the enum-equivalent
    that selects the concrete config subclass from a yaml / Python dict."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return TypeAdapter(SparseAttentionConfig)

    def test_dict_with_algorithm_triattention(self, adapter):
        obj = adapter.validate_python(
            {
                "algorithm": "triattention",
                "calibration_path": "/tmp/x.pt",
                "top_B": 32,
                "beta": 64,
            }
        )
        assert isinstance(obj, TriAttentionConfig)
        assert obj.top_B == 32
        assert obj.beta == 64

    def test_dict_with_algorithm_rocket(self, adapter):
        obj = adapter.validate_python({"algorithm": "rocket"})
        assert isinstance(obj, RocketSparseAttentionConfig)

    def test_dict_with_algorithm_dsa(self, adapter):
        obj = adapter.validate_python({"algorithm": "dsa"})
        assert isinstance(obj, DeepSeekSparseAttentionConfig)

    def test_dict_with_algorithm_skip_softmax(self, adapter):
        obj = adapter.validate_python({"algorithm": "skip_softmax"})
        assert isinstance(obj, SkipSoftmaxAttentionConfig)

    def test_unknown_algorithm_rejected(self, adapter):
        with pytest.raises(ValidationError):
            adapter.validate_python({"algorithm": "made_up_algorithm"})


# ---------------------------------------------------------------------------
# SparseAttentionManager base class default hooks
# ---------------------------------------------------------------------------


class TestSparseAttentionManagerBase:
    """Base hooks must be no-op so subclasses can override only what they need
    without breaking the dispatch wired in PyExecutor / trtllm.py."""

    def test_supports_kv_cache_reuse_default_false(self):
        assert SparseAttentionManager.supports_kv_cache_reuse is False

    def test_default_hooks_no_op(self):
        mgr = SparseAttentionManager.__new__(SparseAttentionManager)
        mgr.kv_cache_manager = None
        # Phase hooks return None (no input-side sparse mask)
        assert mgr.on_context_attention(0, None, None, None, None) is None
        assert mgr.on_generation_attention(0, None, None, None, None) is None
        # Side-effect hooks return None and must not raise
        assert mgr.on_context_end(None, None) is None
        assert mgr.on_generation_step_end(None, None) is None
        assert mgr.on_request_init(None) is None
        assert mgr.on_request_finish(None) is None


# ---------------------------------------------------------------------------
# TriAttention subclass: capability declaration + calibration loading
# ---------------------------------------------------------------------------


class TestTriAttentionClass:
    def test_physically_evicts_kv_true(self):
        assert TriAttention.physically_evicts_kv is True

    def test_pattern1_uses_base_v2(self):
        # Pattern 1: no V2 subclass; kv_cache_manager_class is the BASE V2 type
        # (isinstance sanity check only). The compacted-history reconcile lives
        # in on_generation_step_end.
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            KVCacheManagerV2,
        )
        assert TriAttention.kv_cache_manager_class is KVCacheManagerV2

    def test_supports_kv_cache_reuse_false(self):
        # Per-request periodic eviction is not safe to reuse across requests.
        assert TriAttention.supports_kv_cache_reuse is False

    def test_init_requires_calibration_path(self):
        # Passing kv_cache_manager=None is fine for this specific check
        # because the calibration_path None guard fires before any V2 check
        # (no V2 check inside __init__; V2 isinstance is in the factory).
        with pytest.raises(ValueError, match="requires calibration_path"):
            TriAttention(kv_cache_manager=None, top_B=32, calibration_path=None)

    @CUDA_REQUIRED
    def test_load_calibration_accepts_valid_pt(self, dummy_calibration_pt):
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = None
        loaded = mgr._load_calibration(dummy_calibration_pt)
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded

    @CUDA_REQUIRED
    def test_load_calibration_rejects_missing_keys(self, tmp_path):
        path = tmp_path / "incomplete.pt"
        torch.save({"E_q": torch.zeros(1)}, path)  # missing 3 required keys
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = None
        with pytest.raises(ValueError, match="missing keys"):
            mgr._load_calibration(str(path))


# ---------------------------------------------------------------------------
# create_sparse_attention_manager factory
# ---------------------------------------------------------------------------


class TestFactory:
    """The factory mirrors get_sparse_attn_kv_cache_manager but returns an
    instance (not a class) and only handles behavior-layer methods."""

    def test_returns_none_for_rocket(self):
        assert (
            create_sparse_attention_manager(RocketSparseAttentionConfig(), kv_cache_manager=None)
            is None
        )

    def test_returns_none_for_dsa(self):
        assert (
            create_sparse_attention_manager(DeepSeekSparseAttentionConfig(), kv_cache_manager=None)
            is None
        )

    def test_returns_none_for_skip_softmax(self):
        assert (
            create_sparse_attention_manager(SkipSoftmaxAttentionConfig(), kv_cache_manager=None)
            is None
        )

    def test_raises_for_wrong_cache_mgr_type(self, dummy_calibration_pt):
        cfg = TriAttentionConfig(top_B=32, beta=16, calibration_path=dummy_calibration_pt)
        # Pattern 1: BaseKVCacheCompressionManager.__init__ asserts the injected
        # manager isinstance of the declared kv_cache_manager_class
        # (KVCacheManagerV2, the base type). A non-matching object -> AssertionError.
        with pytest.raises(AssertionError, match="kv_cache_manager_class"):
            create_sparse_attention_manager(cfg, kv_cache_manager="not_a_v2")

    @CUDA_REQUIRED
    def test_returns_triattention_instance_with_v2(self, dummy_calibration_pt):
        # Pattern 1: TriAttention uses the plain KVCacheManagerV2 (no subclass);
        # kv_cache_manager_class = KVCacheManagerV2 (base) is only an isinstance
        # check, so a plain V2 instance satisfies the factory. Stub one via
        # __new__ to avoid real allocation.
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            KVCacheManagerV2,
        )

        fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
        cfg = TriAttentionConfig(top_B=32, beta=16, calibration_path=dummy_calibration_pt)
        mgr = create_sparse_attention_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.kv_cache_manager is fake_v2
