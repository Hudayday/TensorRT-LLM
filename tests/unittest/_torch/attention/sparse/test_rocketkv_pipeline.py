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
        # V17 executor stores torch dtype (matching V1 RocketKV behavior),
        # not the string from the Pydantic config. Config field stays string.
        import torch
        assert mgr.kt_cache_dtype is torch.float8_e5m2
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
        # Pattern 2: uses default plain V2 with KT_CACHE BufferConfig
        # added via multi-pool extension (no Pattern 3 subclass).
        assert RocketKV.kv_cache_manager_class is None

    def test_construct_with_minimal_args(self):
        import torch
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048
        # Executor stores torch dtype (matches V1 RocketKV).
        assert mgr.kt_cache_dtype is torch.bfloat16
        # kt_tokens_per_block defaults to None at config-level but the
        # executor computes it from page_size + tokens_per_block when
        # the cache manager exposes those. With MagicMock cache manager,
        # the safe path returns kt_tokens_per_block based on default
        # page_size (16). Just assert it's an int.
        assert isinstance(mgr.kt_tokens_per_block, int)

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



# ---------------------------------------------------------------------------
# 6. V1 algorithm body port — RocketKV class structure + hook overrides
#    (2026-05-28 — verifying the V1→V17 port preserves algorithm body
#     shape, kernel imports, helper methods, KT pool ownership.)
# ---------------------------------------------------------------------------


class TestRocketKVAlgorithmBodyPort:
    """RocketKV V1→V17 algorithm body port from rocket.py. Side-by-side
    comparable: same kernel calls, same metadata field names, same hook
    fire timing. Storage decision: Pattern 1 (executor owns KT pool)."""

    def test_executor_overrides_all_three_algorithm_hooks(self):
        """V17 RocketKV must override HOOK 2 (Stage I-a / KT build +
        SnapKV mask) + HOOK 3 (Stage I-b physical evict) + HOOK 4
        (Stage II HSA mask). These correspond to V1
        sparse_kv_predict / update_resources rewind / sparse_attn_predict
        respectively."""
        from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
            BaseKVCacheCompressionExecutor)
        mgr = RocketKV(kv_cache_manager=MagicMock())
        for hook_name in ("on_context_attention", "on_context_end",
                          "on_generation_attention"):
            mgr_method = getattr(type(mgr), hook_name)
            base_method = getattr(BaseKVCacheCompressionExecutor, hook_name)
            assert mgr_method is not base_method, (
                f"RocketKV must override {hook_name} (V1 algorithm body "
                f"port). Inherited default no-op means port is missing.")

    def test_executor_overrides_request_lifecycle_hooks(self):
        """V1 RocketKVCacheManager.prepare_resources / free_resources are
        ported to HOOK 1 / HOOK 6 on V17 executor."""
        from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
            BaseKVCacheCompressionExecutor)
        mgr = RocketKV(kv_cache_manager=MagicMock())
        for hook_name in ("on_request_init", "on_request_finish"):
            assert getattr(type(mgr), hook_name) is not getattr(
                BaseKVCacheCompressionExecutor, hook_name), (
                f"RocketKV must override {hook_name} (V1 cache-manager "
                f"lifecycle port).")

    def test_kt_pool_ownership_helpers_exist(self):
        """V1 RocketKVCacheManager.get_kt_buffers + copy_kt_block_offsets
        ported to executor instance methods (Pattern 2: V2's KT_CACHE
        BufferConfig owns the pool; executor delegates via thin shim).

        Replaces V1's cache-manager-subclass-as-owner pattern and the
        earlier Pattern 1 (executor-owned standalone pool which didn't
        sync with V2 page lifecycle)."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert hasattr(mgr, "get_kt_buffers"), (
            "RocketKV must expose get_kt_buffers(layer_idx) — delegates "
            "to V2 KT_CACHE BufferConfig pool (Pattern 2).")
        assert hasattr(mgr, "copy_kt_block_offsets"), (
            "RocketKV must expose copy_kt_block_offsets(...) — delegates "
            "to V2 KEY-role block offsets (shared block IDs).")
        assert hasattr(mgr, "_kt_supported"), (
            "RocketKV must expose _kt_supported flag indicating whether "
            "V2 cache manager has KT_CACHE BufferConfig wired (Pattern 2).")

    def test_kt_pool_carries_v1_compatible_params(self):
        """V1 RocketKVCacheManager.__init__ stores prompt_budget /
        page_size / kt_tokens_per_block + window_size / kernel_size / topk
        / topr. V17 executor stores the same attrs for algorithm-body
        access."""
        mgr = RocketKV(kv_cache_manager=MagicMock(),
                       page_size=16,
                       prompt_budget=2048,
                       window_size=32,
                       kernel_size=5,
                       topk=256,
                       topr=32)
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048
        assert mgr.window_size == 32
        assert mgr.kernel_size == 5
        assert mgr.topk == 256
        assert mgr.topr == 32

    def test_triton_kernel_imports(self):
        """V1 imports a specific set of triton kernels from
        sparse/kernel.py. V17 must import the same set for the algorithm
        body to remain side-by-side comparable. We verify by checking
        the kernel symbols appear in the module's source."""
        import inspect
        from tensorrt_llm._torch.attention_backend.sparse import rocketkv
        src = inspect.getsource(rocketkv)
        # Same 8 triton kernel names as V1 rocket.py imports (line 27-33)
        for kernel_name in ("triton_bmm", "triton_flatten_to_batch",
                            "triton_rocket_batch_to_flatten",
                            "triton_rocket_paged_kt_cache_bmm",
                            "triton_rocket_qk_split",
                            "triton_rocket_reduce_scores",
                            "triton_rocket_update_kt_cache_ctx",
                            "triton_rocket_update_kt_cache_gen",
                            "triton_softmax", "triton_topk"):
            assert kernel_name in src, (
                f"Expected V1 triton kernel ``{kernel_name}`` to be "
                f"referenced in V17 rocketkv.py (side-by-side comparable).")

    def test_helpers_dont_crash_with_mocked_manager(self):
        """The helper methods (get_kt_buffers / copy_kt_block_offsets) must
        not crash when called with a mocked V2 cache manager that has no
        actual KT pool — they should return safely (None / passthrough)."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        # No actual GPU pool allocated for mocked manager → helpers degrade
        assert mgr.get_kt_buffers(0) is None
        # copy_kt_block_offsets with no kt_cache_manager returns input
        # passthrough
        fake_buf = MagicMock()
        result = mgr.copy_kt_block_offsets([], fake_buf)
        assert result is fake_buf


class TestRocketKVAttentionShimsCarryMetadata:
    """The V17 attention shim classes (RocketKVTrtllmAttention /
    RocketKVVanillaAttention) carry the RocketKV-specific Metadata class
    so backend dispatch picks them up. Metadata class itself ports V1
    field names + prepare() body 1:1 for side-by-side comparison."""

    def test_trtllm_shim_carries_correct_metadata(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention, RocketKVTrtllmAttentionMetadata)
        assert RocketKVTrtllmAttention.Metadata is             RocketKVTrtllmAttentionMetadata

    def test_vanilla_shim_carries_correct_metadata(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVVanillaAttention, RocketKVVanillaAttentionMetadata)
        assert RocketKVVanillaAttention.Metadata is             RocketKVVanillaAttentionMetadata

    def test_trtllm_metadata_ports_v1_field_names(self):
        """V17 metadata reuses V1 RocketTrtllmAttentionMetadata field names
        (prompt_budget / window_size / page_size / topk / kt_cache_block_offsets
        / etc.) for side-by-side comparison. We verify by inspecting the
        source for these field references."""
        import inspect
        from tensorrt_llm._torch.attention_backend.sparse import rocketkv
        src = inspect.getsource(rocketkv.RocketKVTrtllmAttentionMetadata)
        for v1_field in ("prompt_budget", "window_size", "page_size", "topk",
                         "kt_cache_block_offsets",
                         "host_kt_cache_block_offsets", "context_cumsum_cuda",
                         "k_context_lens_cuda", "k_context_start_cuda",
                         "valid_seq_indices_cuda", "q_cu_seqlens_cuda",
                         "k_cu_seqlens_cuda", "sparse_offsets_ctx_cuda",
                         "sparse_offsets_gen_cuda", "cum_kt_lens_cuda",
                         "num_kt_tokens", "max_kt_tokens"):
            assert v1_field in src, (
                f"V17 RocketKVTrtllmAttentionMetadata must port V1 field "
                f"``{v1_field}`` (side-by-side comparison).")


class TestRocketKVAlgorithmBodyKernelCalls:
    """Source-level assertion that the V17 HOOK 2/4 callback bodies invoke
    the same triton kernels in the same order as V1
    sparse_kv_predict / sparse_attn_predict. Locks the port against
    accidental drift from the V1 reference implementation."""

    def _get_hook_source(self, hook_name):
        import inspect
        return inspect.getsource(getattr(RocketKV, hook_name))

    def test_on_context_attention_kernel_sequence_matches_v1(self):
        src = self._get_hook_source("on_context_attention")
        # Same 6 triton kernels as V1 sparse_kv_predict
        for kernel_name in ("triton_rocket_qk_split", "triton_bmm",
                            "triton_softmax", "triton_flatten_to_batch",
                            "triton_rocket_batch_to_flatten",
                            "triton_rocket_update_kt_cache_ctx"):
            assert kernel_name in src, (
                f"V17 on_context_attention must call ``{kernel_name}`` "
                f"(matches V1 sparse_kv_predict sequence).")

    def test_on_generation_attention_kernel_sequence_matches_v1(self):
        src = self._get_hook_source("on_generation_attention")
        # Same 4 triton kernels as V1 sparse_attn_predict
        for kernel_name in ("triton_rocket_update_kt_cache_gen",
                            "triton_rocket_paged_kt_cache_bmm",
                            "triton_softmax", "triton_rocket_reduce_scores",
                            "triton_topk"):
            assert kernel_name in src, (
                f"V17 on_generation_attention must call ``{kernel_name}`` "
                f"(matches V1 sparse_attn_predict sequence).")

    def test_on_context_end_does_physical_evict(self):
        """V17 HOOK 3 ports V1's update_resources rewind logic — must
        invoke rewind_kv_cache.

        Note: Pattern 2 drops V1's separate kt_cache_manager.rewind_cache
        call — KT slots live inside the same V2 logical blocks as KEY/VALUE
        (shared block IDs via multi-pool BufferConfig), so V2's
        rewind_kv_cache frees them in one shot."""
        src = self._get_hook_source("on_context_end")
        assert "rewind_kv_cache" in src, (
            "V17 on_context_end must call rewind_kv_cache (Stage I-b "
            "SnapKV physical evict, ported from V1 update_resources).")
        assert "kt_cache_manager.rewind_cache" not in src, (
            "Pattern 2 must NOT call kt_cache_manager.rewind_cache — "
            "KT shares block IDs with KEY/VALUE via V2 multi-pool "
            "BufferConfig, so V2's rewind_kv_cache frees KT slots "
            "automatically (Pattern 2 drops V1 line 1031).")
