"""Unit tests for the TriAttention compression-manager pipeline.

Ported from ``tests/unittest/_torch/attention/sparse/test_triattention_pipeline.py``
after TriAttention was relocated to the ``tensorrt_llm._torch.kv_cache_compression``
package (PR-15106 framework). Covers the configuration + construction layer:

  - The PR-15106 config SPLIT: ``TriAttentionSparseAttentionConfig`` (attention-
    backend / KV-manager-class selector, member of the ``SparseAttentionConfig``
    discriminated union) vs. ``TriAttentionKvCacheCompressionConfig`` (the
    eviction-manager tuning knobs: top_B / beta / window / calib / eviction_mode).
  - ``SparseAttentionConfig`` discriminated-union dispatch (to the sparse config).
  - ``TriAttention`` construction, calibration loading, and capability flags.
  - ``create_kv_cache_compression_manager`` factory dispatch -- now takes a
    ``KvCacheCompressionConfig`` (not the sparse config).
  - The relocation itself: every public symbol imports from the new package; the
    ``_ACTIVE_TRI_MANAGER`` num_cached shim resolves without framework wiring; the
    block-free ``TriAttentionKVCacheManagerV2`` subclass defaults OFF.

These tests do not run real eviction or attention; that needs model weights and
is covered by the NIAH end-to-end run.
"""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

# Relocated package: the manager + attention shim + block-free KV-manager
# subclass now live in tensorrt_llm._torch.kv_cache_compression.triattention.
from tensorrt_llm._torch.kv_cache_compression.triattention import (
    TriAttention,
    TriAttentionKVCacheManagerV2,
    TriAttentionTrtllmAttention,
    TriAttentionTrtllmAttentionMetadata,
)
# The framework base class + factory moved from attention_backend.sparse to
# pyexecutor.resource_manager.
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
    create_kv_cache_compression_manager,
)
# The single old TriAttentionConfig is now TWO configs, both in llm_args.
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheCompressionConfig,
    RocketSparseAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
    TriAttentionKvCacheCompressionConfig,
    TriAttentionSparseAttentionConfig,
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
# Relocation: every public symbol imports from the new package path.
# ---------------------------------------------------------------------------


class TestRelocationImports:
    """PR-15106 moved TriAttention out of attention_backend.sparse into the new
    kv_cache_compression package. Pin the public surface so a future move that
    silently drops a symbol fails here."""

    def test_public_symbols_importable_from_package(self):
        # Import the package fresh and assert __all__ exposes the 4 public names,
        # each bound to a class (not a stub / None).
        import tensorrt_llm._torch.kv_cache_compression.triattention as pkg

        expected = {
            "TriAttention",
            "TriAttentionTrtllmAttention",
            "TriAttentionTrtllmAttentionMetadata",
            "TriAttentionKVCacheManagerV2",
        }
        assert expected.issubset(set(pkg.__all__))
        for name in expected:
            assert isinstance(getattr(pkg, name), type), name

    def test_symbols_are_the_same_objects(self):
        # The names re-exported by the package __init__ are the very classes
        # imported at module top (no shadowing duplicate definition).
        import tensorrt_llm._torch.kv_cache_compression.triattention as pkg

        assert pkg.TriAttention is TriAttention
        assert pkg.TriAttentionTrtllmAttention is TriAttentionTrtllmAttention
        assert (pkg.TriAttentionTrtllmAttentionMetadata
                is TriAttentionTrtllmAttentionMetadata)
        assert pkg.TriAttentionKVCacheManagerV2 is TriAttentionKVCacheManagerV2

    def test_factory_and_base_live_in_resource_manager(self):
        # The factory + base class moved to pyexecutor.resource_manager (they
        # used to be in attention_backend.sparse).
        from tensorrt_llm._torch.pyexecutor import resource_manager as rm

        assert rm.create_kv_cache_compression_manager is (
            create_kv_cache_compression_manager)
        assert rm.BaseKVCacheCompressionManager is BaseKVCacheCompressionManager

    def test_attention_backend_class_relationships(self):
        # The attention shim subclasses the TRT-LLM backend and carries the
        # reconciliation Metadata; the metadata subclasses TrtllmAttentionMetadata.
        from tensorrt_llm._torch.attention_backend.trtllm import (
            TrtllmAttention,
            TrtllmAttentionMetadata,
        )

        assert issubclass(TriAttentionTrtllmAttention, TrtllmAttention)
        assert issubclass(TriAttentionTrtllmAttentionMetadata,
                          TrtllmAttentionMetadata)
        # The backend wires its Metadata to the reconciling subclass.
        assert (TriAttentionTrtllmAttention.Metadata
                is TriAttentionTrtllmAttentionMetadata)


# ---------------------------------------------------------------------------
# TriAttentionSparseAttentionConfig: backend-selector config (algorithm only).
# In the PR-15106 split this config no longer carries the eviction knobs; those
# moved to TriAttentionKvCacheCompressionConfig (see TestKvCacheCompressionConfig).
# ---------------------------------------------------------------------------


class TestSparseAttentionConfig:
    def test_default_algorithm(self):
        cfg = TriAttentionSparseAttentionConfig()
        assert cfg.algorithm == "triattention"

    def test_supports_only_pytorch_backend(self):
        cfg = TriAttentionSparseAttentionConfig()
        assert cfg.supports_backend("pytorch") is True
        assert cfg.supports_backend("tensorrt") is False

    def test_eviction_knobs_not_on_sparse_config(self):
        # The tuning knobs moved to the compression config; StrictBaseModel
        # forbids extra fields, so passing them to the sparse config is rejected.
        with pytest.raises(ValidationError):
            TriAttentionSparseAttentionConfig(top_B=256)


# ---------------------------------------------------------------------------
# TriAttentionKvCacheCompressionConfig: the manager-knob config. This is where
# the old TriAttentionConfig field defaults / overrides now live.
# ---------------------------------------------------------------------------


class TestKvCacheCompressionConfig:
    def test_default_algorithm(self):
        cfg = TriAttentionKvCacheCompressionConfig(calibration_path="/tmp/x.pt")
        assert cfg.algorithm == "triattention"

    def test_calibration_path_optional(self):
        # Calibration is auto-computed and cached on first use, so an explicit
        # path is optional.
        cfg = TriAttentionKvCacheCompressionConfig()
        assert cfg.calibration_path is None
        assert cfg.calib_dataset == "cnn_dailymail"
        assert cfg.calib_batches == 64
        assert cfg.calib_max_seq_length == 2048

    def test_field_defaults(self):
        cfg = TriAttentionKvCacheCompressionConfig()
        assert cfg.top_B == 1024
        assert cfg.beta == 128
        assert cfg.window_size == 128
        # Defaults newly surfaced on the compression config.
        assert cfg.eviction_mode == "per_layer"
        assert cfg.normalize_scores is True
        assert cfg.pin_prefill is True
        assert cfg.use_triton is False
        assert cfg.use_batched is False

    def test_field_overrides(self):
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=256, beta=64, calibration_path="/tmp/x.pt")
        assert cfg.top_B == 256
        assert cfg.beta == 64

    def test_eviction_mode_validated(self):
        # eviction_mode is a Literal; an unknown value is rejected at validation.
        with pytest.raises(ValidationError):
            TriAttentionKvCacheCompressionConfig(eviction_mode="made_up_mode")

    def test_is_subclass_of_base_compression_config(self):
        assert issubclass(TriAttentionKvCacheCompressionConfig,
                          KvCacheCompressionConfig)


# ---------------------------------------------------------------------------
# SparseAttentionConfig discriminated-union dispatch.
# The union still dispatches on "algorithm"; the triattention branch resolves to
# the SPARSE config (the union has no compression configs in it).
# ---------------------------------------------------------------------------


class TestUnionDiscriminator:
    """``Field(discriminator="algorithm")`` selects the concrete sparse-config
    subclass from a yaml / Python dict."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return TypeAdapter(SparseAttentionConfig)

    def test_dict_with_algorithm_triattention(self, adapter):
        # The sparse union resolves "triattention" to the SPARSE config (the
        # backend selector), not the compression config. It carries no knobs.
        obj = adapter.validate_python({"algorithm": "triattention"})
        assert isinstance(obj, TriAttentionSparseAttentionConfig)
        assert obj.algorithm == "triattention"

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


class TestStepBeginHookRefactor:
    """Q1 refactor: periodic eviction runs via the framework's pre-forward
    ``on_generation_step_begin`` hook (fired from BaseKVCacheCompressionManager.
    prepare_resources, which PyExecutor calls before _forward_step) -- NOT via a
    TriAttention ``prepare_resources`` override."""

    def test_base_has_pre_forward_hook(self):
        # framework gained the pre-forward hook (default no-op).
        assert hasattr(BaseKVCacheCompressionManager, "on_generation_step_begin")

    def test_triattention_uses_hook_not_prepare_resources_override(self):
        # the override was removed; the eviction lives in the hook instead.
        assert "prepare_resources" not in TriAttention.__dict__, \
            "prepare_resources override should be gone (use the base + the hook)"
        assert "on_generation_step_begin" in TriAttention.__dict__

    def test_hook_runs_periodic_evict(self):
        import unittest.mock as mock
        mgr = TriAttention.__new__(TriAttention)
        with mock.patch.object(TriAttention, "_periodic_evict") as pe:
            mgr.on_generation_step_begin("BATCH")
            pe.assert_called_once_with("BATCH")

    def test_base_prepare_resources_fires_hook_pre_forward(self):
        # the base prepare_resources (pre-forward) must fan out the step-begin hook
        # so eviction mutates KV before this iteration's forward reads it.
        import unittest.mock as mock
        mgr = BaseKVCacheCompressionManager.__new__(BaseKVCacheCompressionManager)
        batch = mock.MagicMock()
        batch.context_requests = []
        with mock.patch.object(BaseKVCacheCompressionManager,
                               "on_generation_step_begin") as sb:
            mgr.prepare_resources(batch)
            sb.assert_called_once_with(batch)


# ---------------------------------------------------------------------------
# num_cached shim: the metadata reconcile finds its manager via the module
# global _ACTIVE_TRI_MANAGER set in TriAttention.__init__ (PR-15106 does NOT
# wire metadata.compression_manager), so the shim works without framework glue.
# ---------------------------------------------------------------------------


def _make_fake_v2(enable_block_reuse=False):
    """A KVCacheManagerV2 with only the attribute the base guard reads."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    return fake_v2


class TestActiveManagerShim:
    """The attention-metadata shim resolves the active manager from a module
    global rather than ``metadata.compression_manager``."""

    def test_init_registers_active_manager(self):
        # Constructing a TriAttention should register it as the active manager.
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4)
        try:
            assert tri._ACTIVE_TRI_MANAGER is mgr
        finally:
            tri._ACTIVE_TRI_MANAGER = None

    def test_metadata_resolves_active_manager(self):
        # The metadata shim's _tri_manager property returns the global manager
        # without metadata.compression_manager being set. We avoid running the
        # heavy TrtllmAttentionMetadata.__init__ by constructing a bare instance
        # and exercising only the resolution property (pure Python).
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4)
        meta = TriAttentionTrtllmAttentionMetadata.__new__(
            TriAttentionTrtllmAttentionMetadata)
        try:
            # No metadata.compression_manager attribute set: must fall through
            # to the global and resolve to our manager.
            assert meta._tri_manager is mgr
        finally:
            tri._ACTIVE_TRI_MANAGER = None

    def test_metadata_resolution_returns_none_when_unset(self):
        # With no active manager and no metadata.compression_manager, the shim
        # resolves to None (so prepare() leaves num_cached untouched).
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        tri._ACTIVE_TRI_MANAGER = None
        meta = TriAttentionTrtllmAttentionMetadata.__new__(
            TriAttentionTrtllmAttentionMetadata)
        assert meta._tri_manager is None

    def test_metadata_resolution_type_guarded(self):
        # If the module global is something that is not a TriAttention (a stale /
        # wrong object), the shim must not return it: the property type-guards on
        # isinstance(cm, TriAttention).
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        tri._ACTIVE_TRI_MANAGER = object()
        try:
            meta = TriAttentionTrtllmAttentionMetadata.__new__(
                TriAttentionTrtllmAttentionMetadata)
            assert meta._tri_manager is None
        finally:
            tri._ACTIVE_TRI_MANAGER = None

    def test_evicted_count_default_zero(self):
        # evicted_count (read by the shim's reconcile) defaults to 0 for an
        # unseen request, so dense / no-evict steps keep stock num_cached.
        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4)
        try:
            assert mgr.evicted_count(12345) == 0
        finally:
            import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package
            tri._ACTIVE_TRI_MANAGER = None


# ---------------------------------------------------------------------------
# Block-free KV-manager subclass: subclasses V2; the block-free reclaim is gated
# OFF by default (pure pass-through, byte-identical).
# ---------------------------------------------------------------------------


class TestBlockFreeManager:
    def test_is_v2_subclass(self):
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            KVCacheManagerV2)

        assert issubclass(TriAttentionKVCacheManagerV2, KVCacheManagerV2)

    def test_block_free_defaults_off(self):
        # The reclaim is gated on a sentinel file under /scratch; in a unit-test
        # environment it is absent, so the gate constant is False and the manager
        # is a pure pass-through to V2.update_resources.
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kv_manager as km

        assert km._TRI_FREE_BLOCKS is False

    def test_passthrough_delegates_to_super_when_off(self, monkeypatch):
        # When the gate is OFF, update_resources delegates straight to
        # KVCacheManagerV2.update_resources (no eviction-aware path). Verify the
        # delegation without a real cache by patching the gate False + the super
        # method, on a bare (un-__init__'d) subclass instance.
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kv_manager as km
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            KVCacheManagerV2)

        monkeypatch.setattr(km, "_TRI_FREE_BLOCKS", False)
        called = {}

        def fake_super(self, scheduled_batch, attn_metadata=None,
                       kv_cache_dtype_byte_size=None):
            called["args"] = (scheduled_batch, attn_metadata,
                              kv_cache_dtype_byte_size)

        monkeypatch.setattr(KVCacheManagerV2, "update_resources", fake_super)
        mgr = TriAttentionKVCacheManagerV2.__new__(TriAttentionKVCacheManagerV2)
        sentinel = object()
        mgr.update_resources(sentinel, None, None)
        assert called["args"][0] is sentinel


# ---------------------------------------------------------------------------
# create_kv_cache_compression_manager factory.
# Now takes a KvCacheCompressionConfig (not the sparse config): it returns a
# TriAttention for algorithm="triattention", and None (with a warning) for any
# unregistered algorithm.
# ---------------------------------------------------------------------------


class TestFactory:
    """The factory builds the compression manager for a registered algorithm, or
    returns None (warns) for an unregistered one."""

    def test_returns_none_for_unregistered_algorithm(self):
        # A base compression config with an algorithm the factory doesn't know
        # about: warn + return None (no manager runs).
        cfg = KvCacheCompressionConfig(algorithm="made_up_algorithm")
        assert (
            create_kv_cache_compression_manager(cfg, kv_cache_manager=None)
            is None
        )

    def test_block_reuse_rejected(self, dummy_calibration_pt):
        # TriAttention rewrites stored keys; the base guard rejects a cache
        # manager that has block reuse enabled. (Construction is enough -- no
        # calibration / CUDA needed; the guard fires in __init__.)
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=32, beta=16, calibration_path=dummy_calibration_pt)
        with pytest.raises(ValueError, match="block reuse"):
            create_kv_cache_compression_manager(
                cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=True))

    def test_returns_triattention_instance_with_v2(self):
        # A plain V2 manager (block reuse off) yields a TriAttention instance.
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = TriAttentionKvCacheCompressionConfig(top_B=32, beta=16)
        try:
            mgr = create_kv_cache_compression_manager(
                cfg, kv_cache_manager=fake_v2)
            assert isinstance(mgr, TriAttention)
            assert mgr.top_B == 32
            assert mgr.beta == 16
            assert mgr.kv_cache_manager is fake_v2
        finally:
            tri._ACTIVE_TRI_MANAGER = None

    def test_factory_propagates_eviction_mode(self):
        # The eviction-mode knob (compression-config-only in the split) reaches
        # the constructed manager through the factory.
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri  # module (holds _ACTIVE_TRI_MANAGER), not the package

        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=64, beta=8, eviction_mode="per_head")
        try:
            mgr = create_kv_cache_compression_manager(
                cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=False))
            assert isinstance(mgr, TriAttention)
            assert mgr.eviction_mode == "per_head"
        finally:
            tri._ACTIVE_TRI_MANAGER = None
