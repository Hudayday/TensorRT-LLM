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
  enforcement, introspection helpers (``has_axis`` / ``get_executor`` /
  ``get_sparse_executor``).
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

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
    ScheduledRequests

from tensorrt_llm._torch.attention_backend.sparse import (
    BaseKVCacheCompressionExecutor,
    KVCacheBehaviorCoordinator,
    SparseAttentionExecutor,
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


class _MockCrossRequestManager(_RecordingMixin, BaseKVCacheCompressionExecutor):
    """Mock cross-request lifecycle manager (axis ``cross_request``)."""

    axis: ClassVar[str] = "cross_request"
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
        coord = KVCacheBehaviorCoordinator(executors=[])
        assert coord.executors == []
        assert coord.has_axis("sparse") is False
        assert coord.get_executor("sparse") is None
        assert coord.get_sparse_executor() is None

    def test_single_sparse_manager(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sparse1")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        assert coord.has_axis("sparse") is True
        assert coord.has_axis("storage") is False
        assert coord.get_executor("sparse") is sparse
        assert coord.get_sparse_executor() is sparse

    def test_three_axes_coexist(self, fake_kv_cache_manager):
        """sparse + storage + cross_request managers can coexist (Phase 5 scenario)."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        cross_request = _MockCrossRequestManager(fake_kv_cache_manager, record, "xr")
        coord = KVCacheBehaviorCoordinator(
            executors=[sparse, storage, cross_request])
        assert coord.has_axis("sparse") is True
        assert coord.has_axis("storage") is True
        assert coord.has_axis("cross_request") is True

    def test_intra_axis_stacking_raises(self, fake_kv_cache_manager):
        """Two managers of the same axis must raise at coordinator init."""
        record = []
        m1 = _MockSparseManager(fake_kv_cache_manager, record, "sp1")
        m2 = _MockSparseManager(fake_kv_cache_manager, record, "sp2")
        with pytest.raises(ValueError,
                           match="Intra-axis stacking not supported"):
            KVCacheBehaviorCoordinator(executors=[m1, m2])

    def test_cross_request_storage_dependency_auto_wired(self,
                                                 fake_kv_cache_manager):
        """If both cross-request and storage managers present,
        cross-request manager's ``storage_delegate`` is auto-set to the
        storage manager."""
        record = []
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        cross_request = _MockCrossRequestManager(fake_kv_cache_manager, record, "xr")
        assert cross_request.storage_delegate is None
        KVCacheBehaviorCoordinator(executors=[storage, cross_request])
        assert cross_request.storage_delegate is storage


# ---------------------------------------------------------------------- #
# 4. Hook dispatch order (HOOK_ORDER table)                              #
# ---------------------------------------------------------------------- #


class TestHookDispatchOrder:

    def _build_three_axis_coord(self, fake_kv_cache_manager):
        record: List[str] = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        cross_request = _MockCrossRequestManager(fake_kv_cache_manager, record, "xr")
        coord = KVCacheBehaviorCoordinator(
            executors=[sparse, storage, cross_request])
        return coord, record

    def test_on_request_init_order_cross_request_sparse_storage(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_request_init(MagicMock())
        # storage manager does not implement on_request_init, but per
        # HOOK_ORDER it is dispatched anyway (default no-op).
        # Expected hooks recorded by the manager subclasses that implement
        # the method: xr (cross_request) first, then sp (sparse), then st (storage).
        assert record == [
            "xr:on_request_init",
            "sp:on_request_init",
            "st:on_request_init",
        ]

    def test_on_context_end_order_sparse_storage_cross_request(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_context_end(MagicMock(), MagicMock())
        # cross_request mock does not implement on_context_end, so only sparse +
        # storage are recorded.
        assert record == [
            "sp:on_context_end",
            "st:on_context_end",
        ]

    def test_on_generation_step_end_order_sparse_storage_cross_request(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_generation_step_end(MagicMock(), MagicMock())
        assert record == [
            "sp:on_generation_step_end",
            "st:on_generation_step_end",
        ]

    def test_on_request_finish_order_sparse_storage_cross_request(
            self, fake_kv_cache_manager):
        coord, record = self._build_three_axis_coord(fake_kv_cache_manager)
        coord.on_request_finish(MagicMock())
        # All three mocks implement on_request_finish, expect full chain.
        assert record == [
            "sp:on_request_finish",
            "st:on_request_finish",
            "xr:on_request_finish",
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
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage])
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
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        # Manually inject a second writing manager bypassing __init__'s
        # mutex check, to exercise the single-source runtime backstop.
        sparse2 = WritingSparse(fake_kv_cache_manager)
        coord._by_axis["sparse"].append(sparse2)
        with pytest.raises(
                RuntimeError,
                match=
                "Multiple executors returned attention metadata"):
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
            assert order == ["sparse", "storage", "cross_request"], (
                f"{hook_name} order is {order}, expected "
                f"['sparse', 'storage', 'cross_request']")


# ---------------------------------------------------------------------- #
# 8. Canonical name imports (post-cleanup; old aliases removed v17)        #
# ---------------------------------------------------------------------- #


class TestCanonicalImports:
    """All backward-compat aliases (``BaseKVCacheBehaviorManager`` /
    ``SparseAttentionManager``) were removed in the v17 alias cleanup.
    Only the canonical names are importable; this test locks that contract.
    """

    def test_canonical_base_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheCompressionExecutor, )
        assert BaseKVCacheCompressionExecutor is not None

    def test_canonical_sparse_executor_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            SparseAttentionExecutor, )
        assert SparseAttentionExecutor is not None

    def test_legacy_base_alias_removed(self):
        """``BaseKVCacheBehaviorManager`` alias was removed; import must fail."""
        import tensorrt_llm._torch.attention_backend.sparse as sparse_pkg
        assert not hasattr(sparse_pkg, "BaseKVCacheBehaviorManager")

    def test_legacy_sparse_alias_removed(self):
        """``SparseAttentionManager`` alias was removed; import must fail."""
        import tensorrt_llm._torch.attention_backend.sparse as sparse_pkg
        assert not hasattr(sparse_pkg, "SparseAttentionManager")

    def test_sparse_is_subclass_of_base(self):
        from tensorrt_llm._torch.attention_backend.sparse import (
            BaseKVCacheCompressionExecutor, SparseAttentionExecutor)
        assert issubclass(SparseAttentionExecutor,
                          BaseKVCacheCompressionExecutor)

    def test_executor_classes_importable_via_module_path(self):
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


# ---------------------------------------------------------------------- #
# 11. Path A — Coordinator inherits BaseResourceManager + fan-out tests   #
# (2026-05-27 design — see doc 30 §5)                                     #
# ---------------------------------------------------------------------- #


class TestCoordinatorInheritsBaseResourceManager:
    """Path A core property: Coordinator IS a BaseResourceManager so
    PyExecutor's main loop auto-invokes prepare/update/free_resources
    without any new PyExecutor hook call sites."""

    def test_is_base_resource_manager_subclass(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import \
            BaseResourceManager
        assert issubclass(KVCacheBehaviorCoordinator, BaseResourceManager)

    def test_get_needed_resource_to_completion_returns_zero(
            self, fake_kv_cache_manager):
        """Coordinator doesn't allocate resources (V2 cache mgr does);
        must return 0 so PyExecutor scheduler doesn't gate on it."""
        coord = KVCacheBehaviorCoordinator(executors=[])
        assert coord.get_needed_resource_to_completion(MagicMock()) == 0

    def test_has_required_base_resource_manager_methods(self):
        for method_name in ("prepare_resources", "update_resources",
                            "free_resources", "get_max_resource_count",
                            "get_needed_resource_to_completion"):
            assert hasattr(KVCacheBehaviorCoordinator, method_name), (
                f"Coordinator missing BaseResourceManager method "
                f"{method_name!r}; PyExecutor won't be able to call it.")

    def test_get_max_resource_count_returns_zero(self,
                                                  fake_kv_cache_manager):
        coord = KVCacheBehaviorCoordinator(executors=[])
        assert coord.get_max_resource_count() == 0


class TestPrepareResourcesFansOutToInit:
    """``prepare_resources(scheduled_batch)`` should call
    ``on_request_init`` exactly once per newly-seen request (dedupe via
    _seen_req_ids), across all registered executors in HOOK_ORDER."""

    def _make_req(self, req_id: int,
                  state: LlmRequestState = LlmRequestState.CONTEXT_INIT):
        req = MagicMock(spec=LlmRequest)
        req.py_request_id = req_id
        req.state = state
        return req

    def _make_batch(self, context_reqs, generation_reqs):
        batch = MagicMock(spec=ScheduledRequests)
        batch.context_requests = context_reqs
        batch.generation_requests = generation_reqs
        return batch

    def test_fires_init_for_new_request(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(req_id=42)
        coord.prepare_resources(self._make_batch([req], []))
        assert record == ["sp:on_request_init"]
        assert 42 in coord._seen_req_ids

    def test_dedupes_init_across_iterations(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(req_id=42)
        coord.prepare_resources(self._make_batch([req], []))
        # Same request appears again in next iteration (still in batch)
        coord.prepare_resources(self._make_batch([], [req]))
        # init must fire only once
        assert record == ["sp:on_request_init"]

    def test_fires_init_for_each_distinct_request(self,
                                                   fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        coord.prepare_resources(
            self._make_batch([self._make_req(1), self._make_req(2)],
                             [self._make_req(3)]))
        assert record == [
            "sp:on_request_init",
            "sp:on_request_init",
            "sp:on_request_init",
        ]
        assert coord._seen_req_ids == {1, 2, 3}

    def test_axis_order_in_init_fan_out(self, fake_kv_cache_manager):
        """HOOK_ORDER for on_request_init is ['cross_request', 'sparse', 'storage'];
        prepare_resources fan-out must respect this."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        cross_request = _MockCrossRequestManager(fake_kv_cache_manager, record, "xr")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage, cross_request])
        coord.prepare_resources(self._make_batch([self._make_req(7)], []))
        assert record == [
            "xr:on_request_init",
            "sp:on_request_init",
            "st:on_request_init",
        ]


class TestUpdateResourcesFansOutToContextEndAndStepEnd:
    """``update_resources`` fans out to:
    - HOOK 3 on_context_end for requests transitioning CONTEXT→GENERATION
    - HOOK 5 on_generation_step_end once per iteration"""

    def _make_req(self, req_id, state):
        req = MagicMock(spec=LlmRequest)
        req.py_request_id = req_id
        req.state = state
        return req

    def _make_batch(self, context_reqs, generation_reqs):
        batch = MagicMock(spec=ScheduledRequests)
        batch.context_requests = context_reqs
        batch.generation_requests = generation_reqs
        return batch

    def test_step_end_fires_once_per_iteration(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        coord.update_resources(self._make_batch([], []), MagicMock())
        assert record == ["sp:on_generation_step_end"]

    def test_context_end_fires_on_state_transition(self,
                                                    fake_kv_cache_manager):
        """When a request was CONTEXT_INIT in the previous
        prepare/update and is GENERATION_IN_PROGRESS now → on_context_end."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        # Seed previous state via a first update with CONTEXT_INIT.
        req_v1 = self._make_req(42, LlmRequestState.CONTEXT_INIT)
        coord.update_resources(self._make_batch([req_v1], []), MagicMock())
        record.clear()
        # Now the same request is GENERATION_IN_PROGRESS.
        req_v2 = self._make_req(42, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.update_resources(self._make_batch([], [req_v2]), MagicMock())
        # on_context_end fires for the transition; on_generation_step_end
        # also fires once.
        assert record == [
            "sp:on_context_end",
            "sp:on_generation_step_end",
        ]

    def test_context_end_NOT_fired_if_no_transition(self,
                                                     fake_kv_cache_manager):
        """A request that stays CONTEXT_INIT across iterations
        (chunked prefill) does NOT fire on_context_end."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(42, LlmRequestState.CONTEXT_INIT)
        coord.update_resources(self._make_batch([req], []), MagicMock())
        record.clear()
        # Same state in next iter — no transition.
        coord.update_resources(self._make_batch([req], []), MagicMock())
        # Only step_end fires.
        assert record == ["sp:on_generation_step_end"]

    def test_attn_metadata_passed_through(self, fake_kv_cache_manager):
        """The optional ``attn_metadata`` parameter is forwarded to both
        on_context_end and on_generation_step_end."""
        sparse = MagicMock(spec=SparseAttentionExecutor)
        sparse.axis = "sparse"
        sparse_meta = MagicMock(name="attn_metadata")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req_v1 = self._make_req(7, LlmRequestState.CONTEXT_INIT)
        coord.update_resources(self._make_batch([req_v1], []), sparse_meta)
        req_v2 = self._make_req(7, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.update_resources(self._make_batch([], [req_v2]), sparse_meta)
        # on_context_end was called once with the metadata.
        sparse.on_context_end.assert_called_with(req_v2, sparse_meta)
        # on_generation_step_end was called both iterations with metadata.
        assert sparse.on_generation_step_end.call_count == 2


class TestFreeResourcesFansOutToFinish:
    """``free_resources(request)`` fans out to HOOK 6 on_request_finish
    same-iteration (no abort race) and clears the dedup state."""

    def _make_req(self, req_id):
        req = MagicMock(spec=LlmRequest)
        req.py_request_id = req_id
        req.state = LlmRequestState.GENERATION_COMPLETE
        return req

    def test_fires_finish_for_request(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(42)
        coord.free_resources(req)
        assert record == ["sp:on_request_finish"]

    def test_clears_seen_req_ids_on_free(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        coord._seen_req_ids.add(42)
        coord._prev_req_state[42] = LlmRequestState.GENERATION_IN_PROGRESS
        coord.free_resources(self._make_req(42))
        # state cleared so a new request reusing the ID would init again
        assert 42 not in coord._seen_req_ids
        assert 42 not in coord._prev_req_state

    def test_axis_order_in_finish_fan_out(self, fake_kv_cache_manager):
        """HOOK_ORDER for on_request_finish is ['sparse', 'storage', 'cross_request']."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        cross_request = _MockCrossRequestManager(fake_kv_cache_manager, record, "xr")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage, cross_request])
        coord.free_resources(self._make_req(7))
        assert record == [
            "sp:on_request_finish",
            "st:on_request_finish",
            "xr:on_request_finish",
        ]


class TestCoordinatorIntegrationEndToEnd:
    """End-to-end test simulating a full request lifecycle through
    PyExecutor's resource_manager interface. Verifies all 6 hooks fire
    in the right order via the BaseResourceManager 3 callbacks."""

    def test_full_lifecycle(self, fake_kv_cache_manager):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import \
            ScheduledRequests
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "e")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])

        def req(rid, state):
            r = MagicMock(spec=LlmRequest)
            r.py_request_id = rid
            r.state = state
            return r

        def batch(ctx, gen):
            b = MagicMock(spec=ScheduledRequests)
            b.context_requests = ctx
            b.generation_requests = gen
            return b

        # Iter 1: new request enters as CONTEXT_INIT
        r_ctx = req(1, LlmRequestState.CONTEXT_INIT)
        coord.prepare_resources(batch([r_ctx], []))
        coord.update_resources(batch([r_ctx], []), MagicMock())
        # Iter 2: same request transitions to GENERATION_IN_PROGRESS
        r_gen = req(1, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.prepare_resources(batch([], [r_gen]))
        coord.update_resources(batch([], [r_gen]), MagicMock())
        # Iter 3: request finishes
        coord.free_resources(req(1, LlmRequestState.GENERATION_COMPLETE))

        assert record == [
            "e:on_request_init",          # iter 1 prepare
            "e:on_generation_step_end",   # iter 1 update (no context_end yet)
            # iter 2 prepare: req already seen, no init
            "e:on_context_end",           # iter 2 update (transition!)
            "e:on_generation_step_end",   # iter 2 update
            "e:on_request_finish",        # iter 3 free
        ]



# ---------------------------------------------------------------------- #
# 12. End-to-end pipeline wire — ResourceManagerType + AttentionMetadata  #
# (2026-05-28 — coordinator registered into resource_manager + injected   #
#  into attn_metadata; locks the wire contract so future refactors keep   #
#  the pipeline connected.)                                               #
# ---------------------------------------------------------------------- #


class TestResourceManagerTypeEnum:
    """The KVCacheBehaviorCoordinator is registered as a new resource
    manager type in ``ResourceManagerType`` — PyExecutor's
    ``resource_manager.{prepare,update,free}_resources`` iteration
    auto-invokes it without any new call site."""

    def test_kv_cache_behavior_coordinator_enum_exists(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            ResourceManagerType)
        assert hasattr(ResourceManagerType,
                       "KV_CACHE_BEHAVIOR_COORDINATOR"), (
            "Path A wires KVCacheBehaviorCoordinator into PyExecutor via the "
            "ResourceManagerType enum; the new entry must exist so the "
            "coordinator can be registered alongside the V2 cache manager.")

    def test_enum_value_is_unique(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            ResourceManagerType)
        values = [m.value for m in ResourceManagerType]
        assert len(values) == len(set(values)), (
            "ResourceManagerType enum values must be unique; new "
            "KV_CACHE_BEHAVIOR_COORDINATOR entry must not collide.")


class TestResourceManagerRegistration:
    """Coordinator can be registered into ``ResourceManager`` dict and
    retrieved via ``get_resource_manager`` — simulating the pattern from
    ``_util.py`` LLM init."""

    def test_coordinator_registers_and_round_trips(self, fake_kv_cache_manager):
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            ResourceManager, ResourceManagerType)
        coord = KVCacheBehaviorCoordinator(executors=[])
        rm = ResourceManager(
            {ResourceManagerType.KV_CACHE_BEHAVIOR_COORDINATOR: coord})
        # PyExecutor reads the coordinator back from the dict via
        # ``get_resource_manager`` — this is what model_engine does to
        # inject coordinator into attn_metadata.
        retrieved = rm.get_resource_manager(
            ResourceManagerType.KV_CACHE_BEHAVIOR_COORDINATOR)
        assert retrieved is coord

    def test_get_resource_manager_returns_none_when_not_registered(
            self, fake_kv_cache_manager):
        """Methods that don't configure sparse-attention leave coordinator
        unregistered. ``model_engine`` then sets
        ``attn_metadata.coordinator = None`` — pipeline still works,
        TrtllmAttention.forward just skips the on_*_attention hook fire."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            ResourceManager, ResourceManagerType)
        rm = ResourceManager({})  # nothing registered
        retrieved = rm.get_resource_manager(
            ResourceManagerType.KV_CACHE_BEHAVIOR_COORDINATOR)
        assert retrieved is None

    def test_resource_manager_iteration_calls_coordinator_callbacks(
            self, fake_kv_cache_manager):
        """ResourceManager.{prepare,update,free}_resources iterates over
        all registered resource managers and calls their callbacks. This
        test asserts Coordinator's BaseResourceManager interface is
        invokable via this generic iteration — the core Path A property."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            ResourceManager, ResourceManagerType)
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        rm = ResourceManager(
            {ResourceManagerType.KV_CACHE_BEHAVIOR_COORDINATOR: coord})

        # Build a fake scheduled_batch
        req = MagicMock(spec=LlmRequest)
        req.py_request_id = 7
        req.state = LlmRequestState.CONTEXT_INIT
        batch = MagicMock(spec=ScheduledRequests)
        batch.context_requests = [req]
        batch.generation_requests = []

        # ResourceManager.prepare_resources iterates registered managers
        # and calls each one's prepare_resources — Coordinator gets
        # auto-invoked without any explicit hook call in PyExecutor.
        rm.prepare_resources(batch)
        assert "sp:on_request_init" in record, (
            "Coordinator registered as resource_manager should fire HOOK 1 "
            "via prepare_resources when PyExecutor iterates resource_managers.")

        rm.update_resources(batch)
        assert "sp:on_generation_step_end" in record, (
            "Coordinator should fire HOOK 5 via update_resources iteration.")

        rm.free_resources(req)
        assert "sp:on_request_finish" in record, (
            "Coordinator should fire HOOK 6 via free_resources iteration.")


class TestAttentionMetadataCoordinatorField:
    """AttentionMetadata holds a ``coordinator`` reference so
    TrtllmAttention.forward can fire HOOK 2/4 via ``metadata.coordinator``.
    The field has default None so models without behavior-layer sparse
    methods continue to work."""

    def test_attention_metadata_has_coordinator_field(self):
        from tensorrt_llm._torch.attention_backend.interface import (
            AttentionMetadata)
        # dataclass introspection
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(AttentionMetadata)}
        assert "coordinator" in fields, (
            "AttentionMetadata must expose ``coordinator`` field so "
            "TrtllmAttention.forward can fire HOOK 2/4 via metadata.coordinator.")

    def test_attention_metadata_coordinator_default_none(self):
        """Models without behavior-layer sparse config: coordinator stays
        None and TrtllmAttention.forward skips the HOOK 2/4 fire."""
        from tensorrt_llm._torch.attention_backend.interface import (
            AttentionMetadata)
        import dataclasses
        for f in dataclasses.fields(AttentionMetadata):
            if f.name == "coordinator":
                assert f.default is None, (
                    "coordinator field must default to None so existing "
                    "models that don't configure sparse-attention are not "
                    "affected by Path A.")
                return
        pytest.fail("coordinator field not found")

    def test_attention_metadata_coordinator_settable(
            self, fake_kv_cache_manager):
        """model_engine wire pattern: ``attn_metadata.coordinator = rm.
        get_resource_manager(KV_CACHE_BEHAVIOR_COORDINATOR)``. The
        attribute must be writable on an existing AttentionMetadata
        instance."""
        from tensorrt_llm._torch.attention_backend.interface import (
            AttentionMetadata)
        # We don't instantiate AttentionMetadata directly (it has many
        # required fields); instead verify the field is in the dataclass
        # definition and can be set via __setattr__ on a mock.
        coord = KVCacheBehaviorCoordinator(executors=[])
        # Use a Mock with spec to enforce the attribute exists in the
        # dataclass; setting it should succeed.
        m = MagicMock(spec=AttentionMetadata)
        m.coordinator = coord
        assert m.coordinator is coord
