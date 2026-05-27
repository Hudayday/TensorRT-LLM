"""Unit tests for the v17 V2-migrated RocketKV pipeline (parallel to
``test_triattention_pipeline.py``).

Covers the wiring landed in commit c92690ef09 + this commit's additions:

- :class:`RocketKVSparseAttentionConfig` Pydantic discriminator dispatch via
  the ``SparseAttentionConfig`` Annotated Union (``algorithm="rocketkv"``).
- Coexistence with the legacy :class:`RocketSparseAttentionConfig`
  (``algorithm="rocket"``) which routes to the V1 ``RocketKVCacheManager``
  cache-manager subclass — both must remain selectable.
- :attr:`BaseSparseAttentionConfig.is_behavior_layer_method` property
  semantics: ``True`` for ``rocketkv``, ``False`` for legacy ``rocket``.
- :func:`create_sparse_attention_manager` factory dispatch returning a
  :class:`RocketKV` instance for ``rocketkv``, ``None`` for legacy
  ``rocket`` (which goes through ``get_sparse_attn_kv_cache_manager``).
- :class:`RocketKV` skeleton hook contract (Stage I / Stage II stubs return
  ``None``; capability ClassVars correct).
- :class:`KVCacheManagerV2` isinstance assertion at factory level.

The tests intentionally do *not* exercise actual KT-summary computation or
HSA-mask construction; those are Phase 7 algorithm-body work. This is a
**pipeline-level wire test** mirroring the TriAttention pipeline test —
ensures the framework can adapt both sparse-attention methods (TriAttention
physical-evict + RocketKV sparse-mask).
"""

import pytest
from pydantic import TypeAdapter, ValidationError
from unittest.mock import MagicMock

from tensorrt_llm._torch.attention_backend.sparse import (
    SparseAttentionExecutor,
    create_sparse_attention_manager,
)
from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
    RocketKV,
)
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    RocketKVSparseAttentionConfig,
    RocketSparseAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
    TriAttentionConfig,
)


# ---------------------------------------------------------------------------
# 1. Pydantic discriminator: ``algorithm="rocketkv"`` resolves to the new
#    config class (NOT the legacy ``RocketSparseAttentionConfig``).


class TestPydanticDiscriminator:

    def test_rocketkv_string_resolves_to_v17_config(self):
        adapter = TypeAdapter(SparseAttentionConfig)
        cfg = adapter.validate_python({"algorithm": "rocketkv"})
        assert isinstance(cfg, RocketKVSparseAttentionConfig)
        assert cfg.algorithm == "rocketkv"

    def test_legacy_rocket_string_still_resolves_to_v1_config(self):
        """Coexistence check: legacy ``algorithm="rocket"`` MUST keep routing
        to the V1 ``RocketSparseAttentionConfig`` after the rocketkv addition."""
        adapter = TypeAdapter(SparseAttentionConfig)
        cfg = adapter.validate_python({"algorithm": "rocket"})
        assert isinstance(cfg, RocketSparseAttentionConfig)
        assert cfg.algorithm == "rocket"

    def test_unknown_algorithm_rejected(self):
        adapter = TypeAdapter(SparseAttentionConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({"algorithm": "rocketkv-mistyped"})

    def test_default_field_values(self):
        cfg = RocketKVSparseAttentionConfig()
        assert cfg.algorithm == "rocketkv"
        assert cfg.page_size == 16
        assert cfg.prompt_budget == 2048
        assert cfg.kt_cache_dtype == "bfloat16"
        assert cfg.kt_tokens_per_block is None

    def test_custom_field_values(self):
        cfg = RocketKVSparseAttentionConfig(
            page_size=8,
            prompt_budget=4096,
            kt_cache_dtype="float8_e5m2",
            kt_tokens_per_block=4,
        )
        assert cfg.page_size == 8
        assert cfg.prompt_budget == 4096
        assert cfg.kt_cache_dtype == "float8_e5m2"
        assert cfg.kt_tokens_per_block == 4


# ---------------------------------------------------------------------------
# 2. is_behavior_layer_method property semantics


class TestBehaviorLayerMethodProperty:

    def test_rocketkv_is_behavior_layer(self):
        cfg = RocketKVSparseAttentionConfig()
        assert cfg.is_behavior_layer_method is True

    def test_legacy_rocket_is_NOT_behavior_layer(self):
        """Coexistence: legacy ``rocket`` config keeps routing to the
        memory-layer V1 path (cache-manager subclass)."""
        cfg = RocketSparseAttentionConfig()
        assert cfg.is_behavior_layer_method is False

    def test_triattention_is_behavior_layer(self):
        """Sanity: TriAttention still routes to the behavior layer."""
        cfg = TriAttentionConfig(calibration_path="/tmp/dummy.pt")
        assert cfg.is_behavior_layer_method is True

    def test_dsa_and_skipsoftmax_are_NOT_behavior_layer(self):
        for cfg in (DeepSeekSparseAttentionConfig(),
                    SkipSoftmaxAttentionConfig()):
            assert cfg.is_behavior_layer_method is False


# ---------------------------------------------------------------------------
# 3. create_sparse_attention_manager factory dispatch


class TestFactoryDispatch:

    def _fake_v2(self):
        """Build a MagicMock that passes ``isinstance(_, KVCacheManagerV2)``."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import \
            KVCacheManagerV2
        m = MagicMock(spec=KVCacheManagerV2)
        return m

    def test_rocketkv_returns_rocketkv_instance_with_v2(self):
        cfg = RocketKVSparseAttentionConfig(page_size=16,
                                            prompt_budget=2048)
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert isinstance(mgr, RocketKV)
        assert isinstance(mgr, SparseAttentionExecutor)
        # Constructor params forwarded
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048

    def test_rocketkv_custom_values_forwarded(self):
        cfg = RocketKVSparseAttentionConfig(
            page_size=8,
            prompt_budget=4096,
            kt_cache_dtype="float8_e5m2",
            kt_tokens_per_block=4,
        )
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert isinstance(mgr, RocketKV)
        assert mgr.page_size == 8
        assert mgr.prompt_budget == 4096
        assert mgr.kt_cache_dtype == "float8_e5m2"
        assert mgr.kt_tokens_per_block == 4

    def test_legacy_rocket_returns_none(self):
        """Coexistence: legacy ``rocket`` config returns None from this
        factory (it goes through ``get_sparse_attn_kv_cache_manager`` which
        returns the V1 cache-manager subclass instead)."""
        cfg = RocketSparseAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_dsa_returns_none(self):
        cfg = DeepSeekSparseAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_skip_softmax_returns_none(self):
        cfg = SkipSoftmaxAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_rocketkv_raises_type_error_for_non_v2_cache_mgr(self):
        cfg = RocketKVSparseAttentionConfig()
        # Plain MagicMock without spec → fails isinstance(KVCacheManagerV2)
        not_v2 = MagicMock(name="not_v2_manager")
        with pytest.raises(TypeError, match="KVCacheManagerV2"):
            create_sparse_attention_manager(cfg, not_v2)


# ---------------------------------------------------------------------------
# 4. RocketKV class skeleton — hook + capability contract


class TestRocketKVClass:

    def test_subclass_chain(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheCompressionExecutor, )
        assert issubclass(RocketKV, SparseAttentionExecutor)
        assert issubclass(RocketKV, BaseKVCacheCompressionExecutor)

    def test_axis_classvar(self):
        assert RocketKV.axis == "sparse"

    def test_capability_declarations(self):
        # RocketKV is a 2-stage hybrid: Stage I-b at prefill end PHYSICALLY
        # evicts (SnapKV top-pB keep), then Stage II does sparse mask over
        # the shrunk cache. So physically_evicts_kv MUST be True.
        # (Earlier False was a hallucination — see README v16.0.16.)
        assert RocketKV.physically_evicts_kv is True
        # KT cache + Stage I-b keep-set are both request-specific.
        assert RocketKV.supports_kv_cache_reuse is False
        # Pattern 1+2: uses default plain V2 (no Pattern 3 subclass)
        assert RocketKV.kv_cache_manager_class is None

    def test_construct_with_minimal_args(self):
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048
        assert mgr.kt_cache_dtype == "bfloat16"
        assert mgr.kt_tokens_per_block is None

    def test_stage_i_stub_returns_none(self):
        """Stage I (on_context_attention) is a stub — algorithm body 待
        Phase 7. Must return None so kernel falls back to dense attention."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        result = mgr.on_context_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_stage_ii_stub_returns_none(self):
        """Stage II (on_generation_attention) is a stub — algorithm body 待
        Phase 7. Must return None so kernel falls back to dense attention."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        result = mgr.on_generation_attention(0, None, None, None,
                                             MagicMock())
        assert result is None

    def test_other_hooks_default_noop(self):
        """The 3 hooks RocketKV does NOT override inherit base no-op
        (request_init / generation_step_end / request_finish).

        The 3 hooks it DOES override are tested separately:
        - on_context_attention (Stage I-a) — stub returns None
        - on_context_end       (Stage I-b) — stub returns None (will do
          SnapKV physical evict in Phase 7)
        - on_generation_attention (Stage II) — stub returns None
        """
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.on_request_init(MagicMock()) is None
        assert mgr.on_generation_step_end(MagicMock(), MagicMock()) is None
        assert mgr.on_request_finish(MagicMock()) is None

    def test_stage_i_b_overrides_context_end(self):
        """Stage I-b lives in on_context_end. Even though the body is a
        stub right now, the method MUST be overridden on RocketKV (not
        inherited from base) — that's how the framework dispatches the
        physical evict at prefill end."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.implements("on_context_end") is True
        # Stub returns None for now; Phase 7 will do compact_request_cache.
        assert mgr.on_context_end(MagicMock(), MagicMock()) is None

    def test_attention_shims_defined_in_same_module(self):
        """RocketKV ships its own attention shim classes alongside the
        executor (per 2026-05-27 design: structure must be finalized so the
        executor pipeline routes to the correct KV manager AND the correct
        attention class)."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention, RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention, RocketKVVanillaAttentionMetadata)
        assert RocketKVTrtllmAttention.Metadata is \
            RocketKVTrtllmAttentionMetadata
        assert RocketKVVanillaAttention.Metadata is \
            RocketKVVanillaAttentionMetadata

    def test_attention_factory_routes_rocketkv_to_its_shim(self):
        """``get_trtllm_sparse_attn_attention_backend`` must return the
        RocketKV-specific attention class for algorithm="rocketkv", NOT
        the default ``TrtllmAttention`` short-circuit used by other
        behavior-layer methods (e.g., TriAttention)."""
        from tensorrt_llm._torch.attention_backend.sparse.utils import (
            get_trtllm_sparse_attn_attention_backend,
            get_vanilla_sparse_attn_attention_backend)
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention, RocketKVVanillaAttention)
        cfg = RocketKVSparseAttentionConfig()
        assert get_trtllm_sparse_attn_attention_backend(cfg) is \
            RocketKVTrtllmAttention
        assert get_vanilla_sparse_attn_attention_backend(cfg) is \
            RocketKVVanillaAttention


# ---------------------------------------------------------------------------
# 5. Coexistence — both configs can be selected without conflict


class TestCoexistence:
    """Critical: both algorithm="rocket" (legacy V1) and algorithm="rocketkv"
    (v17 behavior-layer) must remain selectable. Verifies no Pydantic
    discriminator collision and no factory cross-talk."""

    def test_both_configs_distinct_classes(self):
        cfg_v1 = RocketSparseAttentionConfig()
        cfg_v17 = RocketKVSparseAttentionConfig()
        assert type(cfg_v1) is not type(cfg_v17)
        assert cfg_v1.algorithm != cfg_v17.algorithm

    def test_v1_does_not_get_factory_routed_to_v17(self):
        """Legacy ``algorithm="rocket"`` must NOT return a RocketKV (v17)
        executor — must route through cache-manager subclass instead."""
        cfg = RocketSparseAttentionConfig()
        from tensorrt_llm._torch.pyexecutor.resource_manager import \
            KVCacheManagerV2
        fake_v2 = MagicMock(spec=KVCacheManagerV2)
        mgr = create_sparse_attention_manager(cfg, fake_v2)
        assert mgr is None  # not routed here; legacy path picks up via
        # ``get_sparse_attn_kv_cache_manager`` → ``RocketKVCacheManager``

    def test_v17_does_not_collide_with_v1_in_discriminator(self):
        """Discriminator must round-trip both algorithm values cleanly."""
        adapter = TypeAdapter(SparseAttentionConfig)
        # V1
        v1 = adapter.validate_python({"algorithm": "rocket"})
        assert v1.algorithm == "rocket"
        # V17
        v17 = adapter.validate_python({"algorithm": "rocketkv"})
        assert v17.algorithm == "rocketkv"
        # Cross-check: classes are distinct
        assert type(v1).__name__ == "RocketSparseAttentionConfig"
        assert type(v17).__name__ == "RocketKVSparseAttentionConfig"
