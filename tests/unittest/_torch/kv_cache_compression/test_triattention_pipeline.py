"""Unit tests for the TriAttention compression-manager pipeline.

TriAttention is a pure KV-cache compression method on the PR-15106 framework: it
has NO sparse-attention config and NO attention backend of its own. Decode runs
the model's standard attention over the compacted cache; the manager reconciles
the cached-token count via the framework hook ``adjust_attention_metadata``,
which the model engine calls just before ``attn_metadata.prepare()``. These tests
cover the config + construction + reconcile layer:

  - ``TriAttentionKvCacheCompressionConfig`` (the eviction-manager knobs) and that
    "triattention" is NOT in the ``SparseAttentionConfig`` union.
  - ``TriAttention`` construction, calibration loading, and the
    ``adjust_attention_metadata`` reconcile.
  - ``create_kv_cache_compression_manager`` factory dispatch.
  - The block-free ``TriAttentionKVCacheManagerV2`` subclass (defaults OFF).

These tests do not run real eviction or attention; that needs model weights and
is covered by the NIAH end-to-end run.
"""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

# TriAttention lives in the kv_cache_compression package. It exposes only the
# manager + the optional block-free KV-manager subclass -- no attention classes.
from tensorrt_llm._torch.kv_cache_compression.triattention import (
    TriAttention,
    TriAttentionKVCacheManagerV2,
)

# The framework base class + factory live in pyexecutor.resource_manager.
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
    create_kv_cache_compression_manager,
)
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheCompressionConfig,
    RocketSparseAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
    TriAttentionKvCacheCompressionConfig,
)

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_resolve_calibration moves the loaded tensors to cuda",
)


@pytest.fixture
def flat_calibration_pt(tmp_path):
    """Build a minimal valid calibration ``.pt`` in our flat runtime schema."""
    path = tmp_path / "tri_calib.pt"
    calibration = {
        "E_q": torch.zeros(2, 2, 4, dtype=torch.complex64),
        "E_q_norm": torch.ones(2, 2, 4, dtype=torch.float32),
        "omega": torch.arange(4, dtype=torch.float32),
        "freq_scale_sq": torch.ones(4, dtype=torch.float32),
    }
    torch.save(calibration, path)
    return str(path)


def _make_fake_v2(enable_block_reuse=False):
    """A KVCacheManagerV2 with only the attribute the base guard reads."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    return fake_v2


# ---------------------------------------------------------------------------
# Public package surface (compression manager + block-free subclass only).
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_public_symbols_importable_from_package(self):
        import tensorrt_llm._torch.kv_cache_compression.triattention as pkg

        expected = {"TriAttention", "TriAttentionKVCacheManagerV2"}
        assert expected.issubset(set(pkg.__all__))
        for name in expected:
            assert isinstance(getattr(pkg, name), type), name

    def test_no_attention_classes_exported(self):
        # TriAttention has no attention backend of its own anymore; the old shim
        # classes must be gone from the package surface.
        import tensorrt_llm._torch.kv_cache_compression.triattention as pkg

        for gone in ("TriAttentionTrtllmAttention", "TriAttentionTrtllmAttentionMetadata"):
            assert gone not in pkg.__all__
            assert not hasattr(pkg, gone)

    def test_factory_and_base_live_in_resource_manager(self):
        from tensorrt_llm._torch.pyexecutor import resource_manager as rm

        assert rm.create_kv_cache_compression_manager is (create_kv_cache_compression_manager)
        assert rm.BaseKVCacheCompressionManager is BaseKVCacheCompressionManager


# ---------------------------------------------------------------------------
# TriAttentionKvCacheCompressionConfig: the manager-knob config.
# ---------------------------------------------------------------------------


class TestKvCacheCompressionConfig:
    def test_default_algorithm(self):
        cfg = TriAttentionKvCacheCompressionConfig(calibration_path="/tmp/x.pt")
        assert cfg.algorithm == "triattention"

    def test_calibration_path_defaults_none(self):
        # The config field defaults to None; the manager requires it at resolve
        # time -- TRT-LLM loads the official calibration, it does not compute it.
        cfg = TriAttentionKvCacheCompressionConfig()
        assert cfg.calibration_path is None

    def test_field_defaults(self):
        cfg = TriAttentionKvCacheCompressionConfig()
        assert cfg.top_B == 1024
        assert cfg.beta == 128
        assert cfg.window_size == 128
        assert cfg.eviction_mode == "per_layer"
        assert cfg.normalize_scores is True
        assert cfg.pin_prefill is True

    def test_field_overrides(self):
        cfg = TriAttentionKvCacheCompressionConfig(top_B=256, beta=64, calibration_path="/tmp/x.pt")
        assert cfg.top_B == 256
        assert cfg.beta == 64

    def test_eviction_mode_validated(self):
        with pytest.raises(ValidationError):
            TriAttentionKvCacheCompressionConfig(eviction_mode="made_up_mode")

    def test_is_subclass_of_base_compression_config(self):
        assert issubclass(TriAttentionKvCacheCompressionConfig, KvCacheCompressionConfig)


# ---------------------------------------------------------------------------
# SparseAttentionConfig union: "triattention" is NO LONGER a member (TriAttention
# is a compression method, not a sparse-attention method).
# ---------------------------------------------------------------------------


class TestUnionDiscriminator:
    @pytest.fixture(scope="class")
    def adapter(self):
        return TypeAdapter(SparseAttentionConfig)

    def test_triattention_rejected_by_sparse_union(self, adapter):
        # The whole point of the standalone design: TriAttention does not appear
        # in the sparse-attention discriminated union.
        with pytest.raises(ValidationError):
            adapter.validate_python({"algorithm": "triattention"})

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
# TriAttention manager: capability flags + calibration loading.
# The user supplies the official calibration .pt (github.com/WeianMao/triattention);
# the manager loads + validates it (and converts the official layout). TRT-LLM
# never computes calibration.
# ---------------------------------------------------------------------------


class TestTriAttentionClass:
    def test_is_compression_manager(self):
        assert issubclass(TriAttention, BaseKVCacheCompressionManager)

    def test_resolve_requires_calibration_path(self):
        mgr = TriAttention.__new__(TriAttention)
        mgr.calibration_path = None
        with pytest.raises(ValueError, match="calibration_path"):
            mgr._resolve_calibration()

    def test_validate_rejects_missing_keys(self):
        mgr = TriAttention.__new__(TriAttention)
        with pytest.raises(ValueError, match="missing keys"):
            mgr._validate_calibration({"E_q": torch.zeros(1)})

    def test_resolve_rejects_unrecognized_pt(self, tmp_path):
        path = tmp_path / "junk.pt"
        torch.save({"E_q": torch.zeros(1)}, path)
        mgr = TriAttention.__new__(TriAttention)
        mgr.calibration_path = str(path)
        with pytest.raises(ValueError, match="Unrecognized calibration"):
            mgr._resolve_calibration()

    @CUDA_REQUIRED
    def test_resolve_accepts_flat_pt(self, flat_calibration_pt):
        mgr = TriAttention.__new__(TriAttention)
        mgr.calibration_path = flat_calibration_pt
        mgr.model_path = None
        loaded = mgr._resolve_calibration()
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded


# ---------------------------------------------------------------------------
# adjust_attention_metadata: the cached-token reconcile (replaces the old
# attention-metadata shim). The model engine derives num_cached from the logical
# length; this hook subtracts the evicted count and clamps prompt_lens.
# ---------------------------------------------------------------------------


class _FakeKvCacheParams:
    def __init__(self, num_cached):
        self.num_cached_tokens_per_seq = list(num_cached)


class _FakeMetadata:
    def __init__(self, num_cached, prompt_lens, request_ids, num_contexts=0):
        self.kv_cache_params = _FakeKvCacheParams(num_cached)
        self.prompt_lens = list(prompt_lens)
        self.request_ids = list(request_ids)
        self.num_contexts = num_contexts
        self.num_generations = len(request_ids) - num_contexts


class TestAttentionMetadataReconcile:
    def test_base_hook_is_noop(self):
        # The base compression manager exposes a no-op adjust_attention_metadata
        # so every model that registers no eviction manager is unaffected.
        assert hasattr(BaseKVCacheCompressionManager, "adjust_attention_metadata")
        mgr = BaseKVCacheCompressionManager.__new__(BaseKVCacheCompressionManager)
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)  # must not raise / mutate
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100]

    def test_evicted_request_num_cached_reconciled(self):
        # num_cached -> (max_beam-1) + 1 - evicted. logical=100, evicted=10 -> 91.
        mgr = TriAttention.__new__(TriAttention)
        mgr._evicted = {7: 10}
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq[0] == 91
        assert meta.prompt_lens == [50]  # 50 <= 91, not clamped

    def test_prompt_len_clamped_to_compacted_cache(self):
        # A prompt longer than the whole compacted cache is clamped to num_cached.
        mgr = TriAttention.__new__(TriAttention)
        mgr._evicted = {7: 10}
        meta = _FakeMetadata([100], [200], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq[0] == 91
        assert meta.prompt_lens == [91]

    def test_unevicted_request_untouched(self):
        # ev==0 -> byte-identical (dense / no-evict steps and other models).
        mgr = TriAttention.__new__(TriAttention)
        mgr._evicted = {}
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100]
        assert meta.prompt_lens == [50]

    def test_context_requests_skipped(self):
        # Only generation requests (index >= num_contexts) are reconciled.
        mgr = TriAttention.__new__(TriAttention)
        mgr._evicted = {1: 10, 2: 10}
        meta = _FakeMetadata([100, 100], [50, 50], [1, 2], num_contexts=1)
        mgr.adjust_attention_metadata(meta)
        # request 1 is a context request -> untouched; request 2 is generation.
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100, 91]


class TestStepBeginHookRefactor:
    """Periodic eviction runs via the framework's pre-forward
    ``on_generation_step_begin`` hook, not a ``prepare_resources`` override."""

    def test_base_has_pre_forward_hook(self):
        assert hasattr(BaseKVCacheCompressionManager, "on_generation_step_begin")

    def test_triattention_uses_hook_not_prepare_resources_override(self):
        assert "prepare_resources" not in TriAttention.__dict__, (
            "prepare_resources override should be gone (use the base + the hook)"
        )
        assert "on_generation_step_begin" in TriAttention.__dict__

    def test_hook_runs_periodic_evict(self):
        import unittest.mock as mock

        mgr = TriAttention.__new__(TriAttention)
        with mock.patch.object(TriAttention, "_periodic_evict") as pe:
            mgr.on_generation_step_begin("BATCH")
            pe.assert_called_once_with("BATCH")

    def test_base_prepare_resources_fires_hook_pre_forward(self):
        import unittest.mock as mock

        mgr = BaseKVCacheCompressionManager.__new__(BaseKVCacheCompressionManager)
        batch = mock.MagicMock()
        batch.context_requests = []
        with mock.patch.object(BaseKVCacheCompressionManager, "on_generation_step_begin") as sb:
            mgr.prepare_resources(batch)
            sb.assert_called_once_with(batch)

    def test_evicted_count_default_zero(self):
        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4)
        assert mgr.evicted_count(12345) == 0


# ---------------------------------------------------------------------------
# Block-free KV-manager subclass: subclasses V2; block-free reclaim is gated OFF
# by default (pure pass-through, byte-identical).
# ---------------------------------------------------------------------------


class TestBlockFreeManager:
    def test_is_v2_subclass(self):
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

        assert issubclass(TriAttentionKVCacheManagerV2, KVCacheManagerV2)

    def test_block_free_defaults_off(self):
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kv_manager as km

        assert km._RECLAIM_BLOCKS is False

    def test_passthrough_delegates_to_super_when_off(self, monkeypatch):
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kv_manager as km
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

        monkeypatch.setattr(km, "_RECLAIM_BLOCKS", False)
        called = {}

        def fake_super(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
            called["args"] = (scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

        monkeypatch.setattr(KVCacheManagerV2, "update_resources", fake_super)
        mgr = TriAttentionKVCacheManagerV2.__new__(TriAttentionKVCacheManagerV2)
        sentinel = object()
        mgr.update_resources(sentinel, None, None)
        assert called["args"][0] is sentinel


# ---------------------------------------------------------------------------
# create_kv_cache_compression_manager factory.
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_none_for_unregistered_algorithm(self):
        cfg = KvCacheCompressionConfig(algorithm="made_up_algorithm")
        assert create_kv_cache_compression_manager(cfg, kv_cache_manager=None) is None

    def test_block_reuse_rejected(self, flat_calibration_pt):
        # TriAttention rewrites stored keys; the base guard rejects a cache
        # manager that has block reuse enabled (the guard fires in __init__).
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=32, beta=16, calibration_path=flat_calibration_pt
        )
        with pytest.raises(ValueError, match="block reuse"):
            create_kv_cache_compression_manager(
                cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=True)
            )

    def test_returns_triattention_instance_with_v2(self):
        # A plain V2 manager (block reuse off) yields a TriAttention instance.
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = TriAttentionKvCacheCompressionConfig(top_B=32, beta=16)
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.kv_cache_manager is fake_v2

    def test_factory_propagates_eviction_mode(self):
        cfg = TriAttentionKvCacheCompressionConfig(top_B=64, beta=8, eviction_mode="per_head")
        mgr = create_kv_cache_compression_manager(
            cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=False)
        )
        assert isinstance(mgr, TriAttention)
        assert mgr.eviction_mode == "per_head"
