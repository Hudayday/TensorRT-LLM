"""Unit tests for the TriAttention compression-manager pipeline.

Covers the configuration and construction layer:
  - TriAttentionConfig field defaults / validation.
  - SparseAttentionConfig discriminated-union dispatch to TriAttentionConfig.
  - TriAttention construction, calibration loading, and capability flags.
  - create_kv_cache_compression_manager factory dispatch.

These tests do not run real eviction or attention; that needs model weights and
is covered by the NIAH end-to-end run.
"""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from tensorrt_llm._torch.attention_backend.sparse import (
    BaseKVCacheCompressionManager,
    create_kv_cache_compression_manager,
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
    calibration = {
        "E_q": torch.zeros(2, 2, 4, dtype=torch.complex64),
        "E_q_norm": torch.ones(2, 2, 4, dtype=torch.float32),
        "omega": torch.arange(4, dtype=torch.float32),
        "freq_scale_sq": torch.ones(4, dtype=torch.float32),
    }
    torch.save(calibration, path)
    return str(path)


# ---------------------------------------------------------------------------
# TriAttentionConfig Pydantic validation
# ---------------------------------------------------------------------------


class TestTriAttentionConfig:
    def test_default_algorithm(self):
        cfg = TriAttentionConfig(calibration_path="/tmp/x.pt")
        assert cfg.algorithm == "triattention"

    def test_calibration_path_optional(self):
        # Calibration is auto-computed and cached on first use, so an explicit
        # path is optional.
        cfg = TriAttentionConfig()
        assert cfg.calibration_path is None
        assert cfg.calib_dataset == "cnn_dailymail"
        assert cfg.calib_batches == 64
        assert cfg.calib_max_seq_length == 2048

    def test_field_defaults(self):
        cfg = TriAttentionConfig()
        assert cfg.top_B == 1024
        assert cfg.beta == 128
        assert cfg.window_size == 128

    def test_field_overrides(self):
        cfg = TriAttentionConfig(top_B=256, beta=64, calibration_path="/tmp/x.pt")
        assert cfg.top_B == 256
        assert cfg.beta == 64

    def test_supports_only_pytorch_backend(self):
        cfg = TriAttentionConfig(calibration_path="/tmp/x.pt")
        assert cfg.supports_backend("pytorch") is True
        assert cfg.supports_backend("tensorrt") is False


# ---------------------------------------------------------------------------
# SparseAttentionConfig discriminated-union dispatch
# ---------------------------------------------------------------------------


class TestUnionDiscriminator:
    """``Field(discriminator="algorithm")`` selects the concrete config subclass
    from a yaml / Python dict."""

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
# TriAttention manager: capability flags + calibration loading
# ---------------------------------------------------------------------------


class TestTriAttentionClass:
    def test_is_compression_manager(self):
        assert issubclass(TriAttention, BaseKVCacheCompressionManager)

    def test_cache_file_is_config_keyed(self, tmp_path):
        # The auto-calibration cache filename encodes model + calib config, so
        # the same config reuses the same file.
        mgr = TriAttention.__new__(TriAttention)
        mgr.model_path = "/models/Qwen3-8B"
        mgr.calibration_cache_dir = str(tmp_path)
        mgr.calib_dataset = "cnn_dailymail"
        mgr.calib_batches = 64
        mgr.calib_max_seq_length = 2048
        path = mgr._cache_file()
        assert "Qwen3-8B" in path
        assert "cnn_dailymail_64_2048" in path
        assert path.endswith(".pt")

    def test_cache_file_requires_model_path(self):
        mgr = TriAttention.__new__(TriAttention)
        mgr.model_path = None
        mgr.calibration_cache_dir = None
        with pytest.raises(ValueError, match="model_path"):
            mgr._cache_file()

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
# create_kv_cache_compression_manager factory
# ---------------------------------------------------------------------------


class TestFactory:
    """The factory returns a constructed compression manager for a framework
    method, or None for a method that owns its own cache manager."""

    def test_returns_none_for_rocket(self):
        assert (
            create_kv_cache_compression_manager(
                RocketSparseAttentionConfig(), kv_cache_manager=None
            )
            is None
        )

    def test_returns_none_for_dsa(self):
        assert (
            create_kv_cache_compression_manager(
                DeepSeekSparseAttentionConfig(), kv_cache_manager=None
            )
            is None
        )

    def test_returns_none_for_skip_softmax(self):
        assert (
            create_kv_cache_compression_manager(SkipSoftmaxAttentionConfig(), kv_cache_manager=None)
            is None
        )

    def test_block_reuse_rejected(self, dummy_calibration_pt):
        # TriAttention rewrites stored keys; the base guard rejects a cache
        # manager that has block reuse enabled.
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

        fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
        fake_v2.enable_block_reuse = True
        cfg = TriAttentionConfig(top_B=32, beta=16, calibration_path=dummy_calibration_pt)
        with pytest.raises(ValueError, match="block reuse"):
            create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)

    def test_returns_triattention_instance_with_v2(self):
        # A plain V2 manager (block reuse off) yields a TriAttention instance.
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

        fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
        fake_v2.enable_block_reuse = False
        cfg = TriAttentionConfig(top_B=32, beta=16)
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.kv_cache_manager is fake_v2
