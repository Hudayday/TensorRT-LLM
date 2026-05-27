"""Unit tests for the v17 multi-manager runtime framework skeleton.

Covers the framework scaffolding shipped on top of the existing
``SparseAttentionExecutor`` baseline:

- :class:`BaseKVCacheCompressionExecutor` ABC contract (axis ClassVar enforced,
  6 hook defaults, ``implements()`` introspection).
- :class:`SparseAttentionExecutor` is the convenience subclass of
  ``BaseKVCacheCompressionExecutor`` for sparse-attention methods (existing
  ``TriAttention`` inherits via this layer unchanged).
- :class:`KVCacheBehaviorCoordinator` mutex (intra-axis stacking raises),
  HOOK_ORDER deterministic dispatch, single-source attention metadata
  enforcement, introspection helpers (``has_axis`` / ``get_manager`` /
  ``get_sparse_manager``).
- :func:`create_behavior_coordinator` factory dispatch contract.

These tests intentionally use lightweight in-memory mock managers and do not
require model weights or CUDA. They exercise the framework layer only — the
existing ``test_triattention_pipeline.py`` covers TriAttention construction
+ legacy single-manager path.

PyExecutor wiring is NOT yet active for the coordinator; this test file
locks the framework contract so the Phase 4 wire-up cannot regress these
invariants.
"""

from typing import ClassVar, List
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.attention_backend.sparse import (
    BaseKVCacheCompressionExecutor,
    KVCacheBehaviorCoordinator,
    SparseAttentionExecutor,
    SparseAttentionManager,
    create_behavior_coordinator,
)


# ---------------------------------------------------------------------- #
# Test fixtures: in-memory mock managers (avoid touching V2 / model code)  #
# ---------------------------------------------------------------------- #


class _RecordingMixin:
    """Mixin that records every hook invocation on the instance.

    Lets tests assert dispatch order across managers without depending on
    real algorithm side-effects.
    """

    def __init__(self, kv_cache_manager, record_list: List[str], name: str):
        super().__init__(kv_cache_manager)
        self._record_list = record_list
        self._name = name

    def _record(self, hook_name: str):
        self._record_list.append(f"{self._name}:{hook_name}")


class _MockSparseManager(_RecordingMixin, SparseAttentionExecutor):
    """Mock sparse-attention manager (axis ``sparse``)."""

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_context_end(self, request, metadata):
        self._record("on_context_end")

    def on_generation_step_end(self, scheduled_batch, attn_metadata):
        self._record("on_generation_step_end")

    def on_request_finish(self, request):
        self._record("on_request_finish")


class _MockStorageManager(_RecordingMixin, BaseKVCacheCompressionExecutor):
    """Mock KV-storage manager (axis ``storage``)."""

    axis: ClassVar[str] = "storage"

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_context_end(self, request, metadata):
        self._record("on_context_end")

    def on_generation_step_end(self, scheduled_batch, attn_metadata):
        self._record("on_generation_step_end")

    def on_request_finish(self, request):
        self._record("on_request_finish")


class _MockCRCLManager(_RecordingMixin, BaseKVCacheCompressionExecutor):
    """Mock cross-request lifecycle manager (axis ``crcl``)."""

    axis: ClassVar[str] = "crcl"
    # Storage delegate slot used by coordinator's _wire_dependencies.
    storage_delegate = None

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_request_finish(self, request):
        self._record("on_request_finish")


@pytest.fixture
def fake_kv_cache_manager():
    """The framework layer never inspects this — it's just held as a tool."""
    return MagicMock(name="fake_KVCacheManagerV2")


# ---------------------------------------------------------------------- #
# 1. BaseKVCacheCompressionExecutor ABC contract                              #
# ---------------------------------------------------------------------- #


class TestBaseABC:

    def test_axis_classvar_required(self, fake_kv_cache_manager):
        """Direct instantiation of base (or any subclass missing ``axis``)
        must raise NotImplementedError."""

        class BadManager(BaseKVCacheCompressionExecutor):
            # axis intentionally not set
            pass

        with pytest.raises(NotImplementedError, match="must set the 'axis'"):
            BadManager(fake_kv_cache_manager)

    def test_6_hooks_default_noop(self, fake_kv_cache_manager):
        """All 6 hooks return None / no-op by default."""

        class TrivialManager(BaseKVCacheCompressionExecutor):
            axis = "sparse"

        mgr = TrivialManager(fake_kv_cache_manager)
        assert mgr.on_request_init(MagicMock()) is None
        assert mgr.on_context_attention(0, None, None, None,
                                        MagicMock()) is None
        assert mgr.on_context_end(MagicMock(), MagicMock()) is None
        assert mgr.on_generation_attention(0, None, None, None,
                                           MagicMock()) is None
        assert mgr.on_generation_step_end(MagicMock(), MagicMock()) is None
        assert mgr.on_request_finish(MagicMock()) is None

    def test_implements_introspection(self, fake_kv_cache_manager):
        """``implements()`` reports True only for actually-overridden hooks."""

        class PartialManager(BaseKVCacheCompressionExecutor):
            axis = "sparse"

            def on_generation_step_end(self, scheduled_batch,
                                       attn_metadata):
                pass  # override (even with pass-body is treated as override)

        mgr = PartialManager(fake_kv_cache_manager)
        assert mgr.implements("on_generation_step_end") is True
        assert mgr.implements("on_request_init") is False
        assert mgr.implements("on_context_attention") is False
        # Non-existent hook name returns False.
        assert mgr.implements("nonexistent_hook") is False


# ---------------------------------------------------------------------- #
# 2. SparseAttentionExecutor subclass                                     #
# ---------------------------------------------------------------------- #


class TestSparseAttentionExecutorSubclass:

    def test_is_subclass_of_base(self):
        assert issubclass(SparseAttentionExecutor,
                          BaseKVCacheCompressionExecutor)

    def test_axis_is_sparse(self):
        assert SparseAttentionExecutor.axis == "sparse"

    def test_supports_kv_cache_reuse_default_false(self):
        # Inherited from base, sparse default conservative.
        assert SparseAttentionExecutor.supports_kv_cache_reuse is False

    def test_physically_evicts_kv_default_false(self):
        # Sparse-specific declaration; subclass overrides (TriAttention
        # sets True, RocketKV stays False).
        assert SparseAttentionExecutor.physically_evicts_kv is False

    def test_triattention_inheritance_unchanged(self):
        """TriAttention must remain a SparseAttentionExecutor subclass
        (and therefore a BaseKVCacheCompressionExecutor subclass too)."""
        from tensorrt_llm._torch.attention_backend.sparse.triattention import (
            TriAttention, )
        assert issubclass(TriAttention, SparseAttentionExecutor)
        assert issubclass(TriAttention, BaseKVCacheCompressionExecutor)
        # axis ClassVar inherited
        assert TriAttention.axis == "sparse"

    def test_triattention_physically_evicts_true(self):
        from tensorrt_llm._torch.attention_backend.sparse.triattention import (
            TriAttention, )
        assert TriAttention.physically_evicts_kv is True


# ---------------------------------------------------------------------- #
# 3. KVCacheBehaviorCoordinator construction + mutex                     #
# ---------------------------------------------------------------------- #


class TestCoordinatorConstruction:

    def test_empty_coordinator(self):
        coord = KVCacheBehaviorCoordinator(managers=[])
        assert coord.managers == []
        assert coord.has_axis("sparse") is False
        assert coord.get_manager("sparse") is None
        assert coord.get_sparse_manager() is None

    def test_single_sparse_manager(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sparse1")
        coord = KVCacheBehaviorCoordinator(managers=[sparse])
        assert coord.has_axis("sparse") is True
        assert coord.has_axis("storage") is False
        assert coord.get_manager("sparse") is sparse
        assert coord.get_sparse_manager() is sparse

    def test_three_axes_coexist(self, fake_kv_cache_manager):
        """sparse + storage + crcl managers can coexist (Phase 5 scenario)."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        crcl = _MockCRCLManager(fake_kv_cache_manager, record, "cr")
        coord = KVCacheBehaviorCoordinator(
            managers=[sparse, storage, crcl])
        assert coord.has_axis("sparse") is True
        assert coord.has_axis("storage") is True
        assert coord.has_axis("crcl") is True

    def test_intra_axis_stacking_raises(self, fake_kv_cache_manager):
        """Two managers of the same axis must raise at coordinator init."""
        record = []
        m1 = _MockSparseManager(fake_kv_cache_manager, record, "sp1")
        m2 = _MockSparseManager(fake_kv_cache_manager, record, "sp2")
        with pytest.raises(ValueError,
                           match="Intra-axis stacking not supported"):
            KVCacheBehaviorCoordinator(managers=[m1, m2])

    def test_crcl_storage_dependency_auto_wired(self,
                                                 fake_kv_cache_manager):
        """If both cross-request and storage managers present,
        cross-request manager's ``storage_delegate`` is auto-set to the
        storage manager."""
        record = []
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        crcl = _MockCRCLManager(fake_kv_cache_manager, record, "cr")
        assert crcl.storage_delegate is None
        KVCacheBehaviorCoordinator(managers=[storage, crcl])
        assert crcl.storage_delegate is storage


# ---------------------------------------------------------------------- #
# 4. Hook dispatch order (HOOK_ORDER table)                              #
# ---------------------------------------------------------------------- #


class TestHookDispatchOrder:

    def _build_three_axis_coord(self, fake_kv_cache_manager):
        record: List[str] = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        crcl = _MockCRCLManager(fake_kv_cache_manager, record, "cr")
        coord = KVCacheBehaviorCoordinator(
            managers=[sparse, storage, crcl])
        return coord, record

    def test_on_request_init_order_crcl_sparse_storage(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_request_init(MagicMock())
        # storage manager does not implement on_request_init, but per
        # HOOK_ORDER it is dispatched anyway (default no-op).
        # Expected hooks recorded by the manager subclasses that implement
        # the method: cr (crcl) first, then sp (sparse), then st (storage).
        assert record == [
            "cr:on_request_init",
            "sp:on_request_init",
            "st:on_request_init",
        ]

    def test_on_context_end_order_sparse_storage_crcl(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_context_end(MagicMock(), MagicMock())
        # crcl mock does not implement on_context_end, so only sparse +
        # storage are recorded.
        assert record == [
            "sp:on_context_end",
            "st:on_context_end",
        ]

    def test_on_generation_step_end_order_sparse_storage_crcl(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_generation_step_end(MagicMock(), MagicMock())
        assert record == [
            "sp:on_generation_step_end",
            "st:on_generation_step_end",
        ]

    def test_on_request_finish_order_sparse_storage_crcl(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_request_finish(MagicMock())
        # All three mocks implement on_request_finish, expect full chain.
        assert record == [
            "sp:on_request_finish",
            "st:on_request_finish",
            "cr:on_request_finish",
        ]


# ---------------------------------------------------------------------- #
# 5. Single-source attention metadata invariant                          #
# ---------------------------------------------------------------------- #


class TestAttentionMetadataSingleSource:

    def test_only_sparse_dispatched_for_attention(self,
                                                   fake_kv_cache_manager):
        """``on_context_attention`` / ``on_generation_attention`` only
        dispatch to sparse-attention managers per HOOK_ORDER."""
        called_axes: List[str] = []

        class TrackingSparse(SparseAttentionExecutor):

            def on_context_attention(self, layer_idx, q, k, attn_scores,
                                     metadata):
                called_axes.append("sparse")
                return None

        class TrackingStorage(BaseKVCacheCompressionExecutor):
            axis = "storage"

            def on_context_attention(self, layer_idx, q, k, attn_scores,
                                     metadata):
                called_axes.append("storage")  # must NOT be called
                return None

        sparse = TrackingSparse(fake_kv_cache_manager)
        storage = TrackingStorage(fake_kv_cache_manager)
        coord = KVCacheBehaviorCoordinator(managers=[sparse, storage])
        coord.on_context_attention(0, None, None, None, MagicMock())
        assert called_axes == ["sparse"]

    def test_multiple_sparse_writers_raises(self, fake_kv_cache_manager):
        """If two sparse-attention managers existed and both returned
        non-None attention metadata, the coordinator must raise. We trigger
        this by bypassing the intra-axis mutex (constructing _by_axis
        manually); normally the mutex prevents two sparse managers, but the
        single-source guard is a runtime correctness backstop."""
        record: List[str] = []

        class WritingSparse(SparseAttentionExecutor):

            def on_context_attention(self, layer_idx, q, k, attn_scores,
                                     metadata):
                # Return any sentinel object as if it were sparse mask.
                return ("indices", "offsets")

        sparse = WritingSparse(fake_kv_cache_manager)
        coord = KVCacheBehaviorCoordinator(managers=[sparse])
        # Manually inject a second writing manager bypassing __init__'s
        # mutex check, to exercise the single-source runtime backstop.
        sparse2 = WritingSparse(fake_kv_cache_manager)
        coord._by_axis["sparse"].append(sparse2)
        with pytest.raises(
                RuntimeError,
                match=
                "Multiple managers returned attention metadata"):
            coord.on_context_attention(0, None, None, None, MagicMock())


# ---------------------------------------------------------------------- #
# 6. create_behavior_coordinator factory                                  #
# ---------------------------------------------------------------------- #


class TestBehaviorCoordinatorFactory:

    def test_factory_returns_none_for_none_config(self,
                                                   fake_kv_cache_manager):
        """No sparse config => no coordinator."""
        coord = create_behavior_coordinator(None, fake_kv_cache_manager)
        assert coord is None

    def test_factory_returns_none_for_legacy_method(self,
                                                     fake_kv_cache_manager):
        """Legacy memory-layer methods (rocket/dsa/skip_softmax) are not
        behavior-layer methods — factory returns None so the legacy path
        stays active."""
        legacy_cfg = MagicMock()
        legacy_cfg.is_behavior_layer_method = False
        coord = create_behavior_coordinator(legacy_cfg,
                                             fake_kv_cache_manager)
        assert coord is None


# ---------------------------------------------------------------------- #
# 7. HOOK_ORDER table integrity                                          #
# ---------------------------------------------------------------------- #


class TestHookOrderTable:

    def test_all_6_hooks_have_order_entry(self):
        expected_hooks = {
            "on_request_init",
            "on_request_finish",
            "on_context_attention",
            "on_generation_attention",
            "on_context_end",
            "on_generation_step_end",
        }
        actual_hooks = set(KVCacheBehaviorCoordinator.HOOK_ORDER.keys())
        assert actual_hooks == expected_hooks, (
            f"HOOK_ORDER missing: {expected_hooks - actual_hooks}, "
            f"extra: {actual_hooks - expected_hooks}")

    def test_attention_hooks_only_sparse(self):
        assert KVCacheBehaviorCoordinator.HOOK_ORDER[
            "on_context_attention"] == ["sparse"]
        assert KVCacheBehaviorCoordinator.HOOK_ORDER[
            "on_generation_attention"] == ["sparse"]

    def test_phase_boundary_hooks_have_three_axes(self):
        """Phase-boundary hooks must dispatch in sparse → storage →
        cross-request order."""
        for hook_name in ("on_context_end", "on_generation_step_end",
                          "on_request_finish"):
            order = KVCacheBehaviorCoordinator.HOOK_ORDER[hook_name]
            assert order == ["sparse", "storage", "crcl"], (
                f"{hook_name} order is {order}, expected "
                f"['sparse', 'storage', 'crcl']")


# ---------------------------------------------------------------------- #
# 8. Naming flip — backward-compat alias + new canonical name             #
# ---------------------------------------------------------------------- #


class TestNamingFlip:
    """v17 (2026-05-27) two-step rename:
    - Base: BaseKVCacheBehaviorManager → BaseKVCacheCompressionExecutor
    - Subclass: SparseAttentionManager → SparseAttentionExecutor
    - File: sparse_attention_manager.py → kv_cache_compression_executor.py
    Both aliases preserved for backward compat."""

    def test_canonical_base_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheCompressionExecutor, )
        assert BaseKVCacheCompressionExecutor is not None

    def test_canonical_sparse_executor_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            SparseAttentionExecutor, )
        assert SparseAttentionExecutor is not None

    def test_base_alias_points_to_same_class(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheBehaviorManager, BaseKVCacheCompressionExecutor)
        assert BaseKVCacheBehaviorManager is BaseKVCacheCompressionExecutor

    def test_sparse_alias_points_to_same_class(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            SparseAttentionExecutor, SparseAttentionManager)
        assert SparseAttentionManager is SparseAttentionExecutor

    def test_sparse_is_subclass_of_base(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheCompressionExecutor, SparseAttentionExecutor)
        assert issubclass(SparseAttentionExecutor,
                          BaseKVCacheCompressionExecutor)

    def test_old_module_path_still_importable_via_new_file(self):
        """File renamed sparse_attention_manager.py → kv_cache_compression_executor.py.
        New file path must export the same classes."""
        from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
            BaseKVCacheCompressionExecutor,
            SparseAttentionExecutor,
        )
        assert BaseKVCacheCompressionExecutor is not None
        assert SparseAttentionExecutor is not None


# ---------------------------------------------------------------------- #
# 9. Pattern 3 escape hatch — kv_cache_manager_class ClassVar              #
# ---------------------------------------------------------------------- #


class TestKVCacheManagerClassClassVar:
    """Design γ — Pattern 3 escape hatch (per doc 27 §2 + 5/27 discussion).

    Subclass can declare a specialized V2 type via ``kv_cache_manager_class``
    ClassVar. ``None`` (default) means use plain ``KVCacheManagerV2``.
    PyExecutor factory consults this to pick the right V2 instance type.
    The constructor enforces the type assertion when ClassVar is non-None.
    """

    def test_default_classvar_is_none(self):
        assert BaseKVCacheCompressionExecutor.kv_cache_manager_class is None
        assert SparseAttentionExecutor.kv_cache_manager_class is None

    def test_default_no_type_assert(self, fake_kv_cache_manager):
        """When ClassVar is None, constructor accepts any V2-shaped object."""

        class PlainV2User(SparseAttentionExecutor):
            pass

        # Should not raise; ClassVar is None, type check skipped.
        mgr = PlainV2User(fake_kv_cache_manager)
        assert mgr.kv_cache_manager is fake_kv_cache_manager

    def test_classvar_enforces_type_assert(self, fake_kv_cache_manager):
        """When ClassVar declares a specific V2 type, constructor asserts."""

        class FakeV2Subclass:
            """Imaginary specialized V2 subclass."""

        class StrictMethod(SparseAttentionExecutor):
            kv_cache_manager_class = FakeV2Subclass

        # MagicMock is not an instance of FakeV2Subclass → must raise.
        with pytest.raises(AssertionError,
                           match="kv_cache_manager_class"):
            StrictMethod(fake_kv_cache_manager)

    def test_rocketkv_uses_plain_v2(self, fake_kv_cache_manager):
        """RocketKV V2 skeleton uses Pattern 2 (BufferConfig declarative),
        not Pattern 3 — so its kv_cache_manager_class is None."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        assert RocketKV.kv_cache_manager_class is None
        # Construction should succeed with plain V2 (here a MagicMock).
        mgr = RocketKV(fake_kv_cache_manager)
        assert mgr.kv_cache_manager is fake_kv_cache_manager
        # 2-stage hybrid: Stage I-b at prefill end PHYSICALLY evicts;
        # KT cache + Stage I-b keep-set are request-specific → block
        # reuse incompatible.
        assert RocketKV.physically_evicts_kv is True
        assert RocketKV.supports_kv_cache_reuse is False


# ---------------------------------------------------------------------- #
# 10. RocketKV skeleton — executor + attention shims                       #
# ---------------------------------------------------------------------- #


class TestRocketKVSkeleton:
    """v17 RocketKV V2-migrated skeleton in ``sparse/rocketkv.py``.

    The module ships BOTH halves of the plug-in:
    - executor (RocketKV) — L2 behavior, orchestrates Stage I/II via hooks
    - attention shims (RocketKV*Attention + Metadata) — L0, consume the
      sparse mask. These are skeleton subclasses of the framework bases
      (Phase 7 will port forward bodies from rocket.py).
    """

    def test_subclass_of_sparse_attention_executor(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        assert issubclass(RocketKV, SparseAttentionExecutor)
        assert issubclass(RocketKV, BaseKVCacheCompressionExecutor)

    def test_axis_classvar(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        assert RocketKV.axis == "sparse"

    def test_capability_declarations(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        # 2-stage hybrid: Stage I-b at prefill end PHYSICALLY evicts.
        # (Earlier False was a hallucination — see README v16.0.16.)
        assert RocketKV.physically_evicts_kv is True
        # KT cache + Stage I-b keep-set are request-specific
        assert RocketKV.supports_kv_cache_reuse is False
        # Pattern 2 (BufferConfig declarative), plain V2
        assert RocketKV.kv_cache_manager_class is None

    def test_stage_i_stub_returns_none(self, fake_kv_cache_manager):
        """Stage I (on_context_attention) is stub; returns None."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        mgr = RocketKV(fake_kv_cache_manager)
        result = mgr.on_context_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_stage_ii_stub_returns_none(self, fake_kv_cache_manager):
        """Stage II (on_generation_attention) is stub; returns None
        (kernel falls back to dense)."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKV, )
        mgr = RocketKV(fake_kv_cache_manager)
        result = mgr.on_generation_attention(0, None, None, None,
                                             MagicMock())
        assert result is None

    def test_trtllm_attention_shim_classes_exist(self):
        """Skeleton attention shims are defined alongside the executor."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention, RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention, RocketKVVanillaAttentionMetadata)
        # Class identity + Metadata wiring is what the factory routes on.
        assert RocketKVTrtllmAttention.Metadata is \
            RocketKVTrtllmAttentionMetadata
        assert RocketKVVanillaAttention.Metadata is \
            RocketKVVanillaAttentionMetadata

    def test_attention_shim_inherits_framework_bases(self):
        """The shims must subclass the framework attention bases so they
        plug into the existing attention-class factory and forward paths."""
        from tensorrt_llm._torch.attention_backend.trtllm import (
            TrtllmAttention, TrtllmAttentionMetadata)
        from tensorrt_llm._torch.attention_backend.vanilla import (
            VanillaAttention, VanillaAttentionMetadata)
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention, RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention, RocketKVVanillaAttentionMetadata)
        assert issubclass(RocketKVTrtllmAttention, TrtllmAttention)
        assert issubclass(RocketKVTrtllmAttentionMetadata,
                          TrtllmAttentionMetadata)
        assert issubclass(RocketKVVanillaAttention, VanillaAttention)
        assert issubclass(RocketKVVanillaAttentionMetadata,
                          VanillaAttentionMetadata)
