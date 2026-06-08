"""Unit tests for the multi-manager runtime framework.

Covers:

- :class:`BaseKVCacheCompressionExecutor` ABC contract (axis ClassVar enforced,
  6 hook defaults, ``implements()`` introspection).
- :class:`SparseAttentionExecutor`, the convenience subclass of
  ``BaseKVCacheCompressionExecutor`` for sparse-attention methods.
- :class:`KVCacheBehaviorCoordinator` mutex (intra-axis stacking raises),
  EVENT_AXIS_ORDER deterministic dispatch, single-source attention metadata
  enforcement, introspection helpers (``has_axis`` / ``get_executor`` /
  ``get_sparse_executor``).
- :func:`create_behavior_coordinator` factory dispatch contract.

These tests use lightweight in-memory mock managers and do not require model
weights or CUDA; they exercise the framework layer only.
"""

from typing import ClassVar, List
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.attention_backend.sparse import (
    BaseKVCacheCompressionExecutor,
    KVCacheBehaviorCoordinator,
    SparseAttentionExecutor,
    create_behavior_coordinator,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

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
        assert mgr.on_context_attention(0, None, None, None, MagicMock()) is None
        assert mgr.on_context_end(MagicMock(), MagicMock()) is None
        assert mgr.on_generation_attention(0, None, None, None, MagicMock()) is None
        assert mgr.on_generation_step_end(MagicMock(), MagicMock()) is None
        assert mgr.on_request_finish(MagicMock()) is None

    def test_implements_introspection(self, fake_kv_cache_manager):
        """``implements()`` reports True only for actually-overridden hooks."""

        class PartialManager(BaseKVCacheCompressionExecutor):
            axis = "sparse"

            def on_generation_step_end(self, scheduled_batch, attn_metadata):
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
        assert issubclass(SparseAttentionExecutor, BaseKVCacheCompressionExecutor)

    def test_axis_is_sparse(self):
        assert SparseAttentionExecutor.axis == "sparse"

    def test_supports_kv_cache_reuse_default_false(self):
        # Inherited from base, sparse default conservative.
        assert SparseAttentionExecutor.supports_kv_cache_reuse is False

    def test_physically_evicts_kv_default_false(self):
        # Sparse-specific declaration; physical-evict subclasses override
        # to True (RocketKV stays False).
        assert SparseAttentionExecutor.physically_evicts_kv is False


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

    def test_two_axes_coexist(self, fake_kv_cache_manager):
        """sparse + storage managers can coexist."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage])
        assert coord.has_axis("sparse") is True
        assert coord.has_axis("storage") is True

    def test_intra_axis_stacking_raises(self, fake_kv_cache_manager):
        """Two managers of the same axis must raise at coordinator init."""
        record = []
        m1 = _MockSparseManager(fake_kv_cache_manager, record, "sp1")
        m2 = _MockSparseManager(fake_kv_cache_manager, record, "sp2")
        with pytest.raises(ValueError, match="Intra-axis stacking not supported"):
            KVCacheBehaviorCoordinator(executors=[m1, m2])


# ---------------------------------------------------------------------- #
# 4. Hook dispatch order (EVENT_AXIS_ORDER table)                              #
# ---------------------------------------------------------------------- #


class TestHookDispatchOrder:
    def _build_two_axis_coord(self, fake_kv_cache_manager):
        record: List[str] = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage])
        return coord, record

    def test_on_request_init_order_sparse_storage(self, fake_kv_cache_manager):
        coord, record = self._build_two_axis_coord(fake_kv_cache_manager)
        coord.on_request_init(MagicMock())
        # Dispatched in EVENT_AXIS_ORDER: sp (sparse) then st (storage).
        assert record == [
            "sp:on_request_init",
            "st:on_request_init",
        ]

    def test_on_context_end_order_sparse_storage(self, fake_kv_cache_manager):
        coord, record = self._build_two_axis_coord(fake_kv_cache_manager)
        coord.on_context_end(MagicMock(), MagicMock())
        assert record == [
            "sp:on_context_end",
            "st:on_context_end",
        ]

    def test_on_generation_step_end_order_sparse_storage(self, fake_kv_cache_manager):
        coord, record = self._build_two_axis_coord(fake_kv_cache_manager)
        coord.on_generation_step_end(MagicMock(), MagicMock())
        assert record == [
            "sp:on_generation_step_end",
            "st:on_generation_step_end",
        ]

    def test_on_request_finish_order_sparse_storage(self, fake_kv_cache_manager):
        coord, record = self._build_two_axis_coord(fake_kv_cache_manager)
        coord.on_request_finish(MagicMock())
        # Both mocks implement on_request_finish, expect full chain.
        assert record == [
            "sp:on_request_finish",
            "st:on_request_finish",
        ]


# ---------------------------------------------------------------------- #
# 5. Single-source attention metadata invariant                          #
# ---------------------------------------------------------------------- #


class TestAttentionMetadataSingleSource:
    def test_only_sparse_dispatched_for_attention(self, fake_kv_cache_manager):
        """``on_context_attention`` / ``on_generation_attention`` only
        dispatch to sparse-attention managers per EVENT_AXIS_ORDER."""
        called_axes: List[str] = []

        class TrackingSparse(SparseAttentionExecutor):
            def on_context_attention(self, layer_idx, q, k, attn_scores, metadata):
                called_axes.append("sparse")
                return None

        class TrackingStorage(BaseKVCacheCompressionExecutor):
            axis = "storage"

            def on_context_attention(self, layer_idx, q, k, attn_scores, metadata):
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

        class WritingSparse(SparseAttentionExecutor):
            def on_context_attention(self, layer_idx, q, k, attn_scores, metadata):
                # Return any sentinel object as if it were sparse mask.
                return ("indices", "offsets")

        sparse = WritingSparse(fake_kv_cache_manager)
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        # Manually inject a second writing manager bypassing __init__'s
        # mutex check, to exercise the single-source runtime backstop.
        sparse2 = WritingSparse(fake_kv_cache_manager)
        coord._by_axis["sparse"].append(sparse2)
        with pytest.raises(RuntimeError, match="Multiple executors returned attention metadata"):
            coord.on_context_attention(0, None, None, None, MagicMock())


# ---------------------------------------------------------------------- #
# 6. create_behavior_coordinator factory                                  #
# ---------------------------------------------------------------------- #


class TestBehaviorCoordinatorFactory:
    def test_factory_returns_none_for_none_config(self, fake_kv_cache_manager):
        """No sparse config => no coordinator."""
        coord = create_behavior_coordinator(None, fake_kv_cache_manager)
        assert coord is None

    def test_factory_returns_none_for_legacy_method(self, fake_kv_cache_manager):
        """Legacy memory-layer methods (rocket/dsa/skip_softmax) are not
        behavior-layer methods — factory returns None so the legacy path
        stays active."""
        legacy_cfg = MagicMock()
        legacy_cfg.is_behavior_layer_method = False
        coord = create_behavior_coordinator(legacy_cfg, fake_kv_cache_manager)
        assert coord is None


# ---------------------------------------------------------------------- #
# 7. EVENT_AXIS_ORDER table integrity                                          #
# ---------------------------------------------------------------------- #


class TestHookOrderTable:
    def test_all_8_hooks_have_order_entry(self):
        expected_hooks = {
            "on_request_init",
            "on_request_finish",
            "on_context_attention",
            "on_context_attention_end",
            "on_generation_attention",
            "on_generation_attention_end",
            "on_context_end",
            "on_generation_step_end",
        }
        actual_hooks = set(KVCacheBehaviorCoordinator.EVENT_AXIS_ORDER.keys())
        assert actual_hooks == expected_hooks, (
            f"EVENT_AXIS_ORDER missing: {expected_hooks - actual_hooks}, "
            f"extra: {actual_hooks - expected_hooks}"
        )

    def test_attention_hooks_only_sparse(self):
        assert KVCacheBehaviorCoordinator.EVENT_AXIS_ORDER["on_context_attention"] == ["sparse"]
        assert KVCacheBehaviorCoordinator.EVENT_AXIS_ORDER["on_generation_attention"] == ["sparse"]

    def test_phase_boundary_hooks_have_two_axes(self):
        """Phase-boundary hooks must dispatch in sparse → storage order."""
        for hook_name in ("on_context_end", "on_generation_step_end", "on_request_finish"):
            order = KVCacheBehaviorCoordinator.EVENT_AXIS_ORDER[hook_name]
            assert order == ["sparse", "storage"], (
                f"{hook_name} order is {order}, expected ['sparse', 'storage']"
            )


# ---------------------------------------------------------------------- #
# 8. Canonical name imports (post-cleanup; old aliases removed)            #
# ---------------------------------------------------------------------- #


class TestCanonicalImports:
    """All backward-compat aliases (``BaseKVCacheBehaviorManager`` /
    ``SparseAttentionManager``) were removed; only the canonical names are
    importable. This test locks that contract.
    """

    def test_canonical_base_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import BaseKVCacheCompressionExecutor

        assert BaseKVCacheCompressionExecutor is not None

    def test_canonical_sparse_executor_name_importable(self):
        from tensorrt_llm._torch.attention_backend.sparse import SparseAttentionExecutor

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
            BaseKVCacheCompressionExecutor,
            SparseAttentionExecutor,
        )

        assert issubclass(SparseAttentionExecutor, BaseKVCacheCompressionExecutor)

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
    """Pattern 3 escape hatch.

    Subclass can declare a specialized KVCacheManagerV2 type via ``kv_cache_manager_class``
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
        with pytest.raises(AssertionError, match="kv_cache_manager_class"):
            StrictMethod(fake_kv_cache_manager)

    def test_rocketkv_uses_plain_v2(self, fake_kv_cache_manager):
        """RocketKV uses Pattern 2 (BufferConfig declarative), not Pattern 3
        — so its kv_cache_manager_class is None."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

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
# 10. RocketKV — executor + attention shims                                #
# ---------------------------------------------------------------------- #


class TestRocketKV:
    """RocketKV in ``sparse/rocketkv.py``.

    The module ships BOTH halves of the plug-in:
    - executor (RocketKV) — L2 behavior, orchestrates Stage I/II via hooks
    - attention shims (RocketKV*Attention + Metadata) — L0, consume the
      sparse mask. These are subclasses of the framework bases.
    """

    def test_subclass_of_sparse_attention_executor(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

        assert issubclass(RocketKV, SparseAttentionExecutor)
        assert issubclass(RocketKV, BaseKVCacheCompressionExecutor)

    def test_axis_classvar(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

        assert RocketKV.axis == "sparse"

    def test_capability_declarations(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

        # 2-stage hybrid: Stage I-b at prefill end PHYSICALLY evicts.
        assert RocketKV.physically_evicts_kv is True
        # KT cache + Stage I-b keep-set are request-specific
        assert RocketKV.supports_kv_cache_reuse is False
        # Pattern 2 (BufferConfig declarative), plain KVCacheManagerV2
        assert RocketKV.kv_cache_manager_class is None

    def test_stage_i_returns_none_without_executor(self, fake_kv_cache_manager):
        """Stage I (on_context_attention) returns None when no sparse
        executor is wired (MagicMock fixture has no KT_CACHE pool)."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

        mgr = RocketKV(fake_kv_cache_manager)
        result = mgr.on_context_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_stage_ii_returns_none_without_executor(self, fake_kv_cache_manager):
        """Stage II (on_generation_attention) returns None when no sparse
        executor is wired (kernel falls back to dense)."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV

        mgr = RocketKV(fake_kv_cache_manager)
        result = mgr.on_generation_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_trtllm_attention_shim_classes_exist(self):
        """Attention shims are defined alongside the executor."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention,
            RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention,
            RocketKVVanillaAttentionMetadata,
        )

        # Class identity + Metadata wiring is what the factory routes on.
        assert RocketKVTrtllmAttention.Metadata is RocketKVTrtllmAttentionMetadata
        assert RocketKVVanillaAttention.Metadata is RocketKVVanillaAttentionMetadata

    def test_attention_shim_inherits_framework_bases(self):
        """The shims must subclass the framework attention bases so they
        plug into the existing attention-class factory and forward paths."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention,
            RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention,
            RocketKVVanillaAttentionMetadata,
        )
        from tensorrt_llm._torch.attention_backend.trtllm import (
            TrtllmAttention,
            TrtllmAttentionMetadata,
        )
        from tensorrt_llm._torch.attention_backend.vanilla import (
            VanillaAttention,
            VanillaAttentionMetadata,
        )

        assert issubclass(RocketKVTrtllmAttention, TrtllmAttention)
        assert issubclass(RocketKVTrtllmAttentionMetadata, TrtllmAttentionMetadata)
        assert issubclass(RocketKVVanillaAttention, VanillaAttention)
        assert issubclass(RocketKVVanillaAttentionMetadata, VanillaAttentionMetadata)


# ---------------------------------------------------------------------- #
# 11. Path A — Coordinator inherits BaseResourceManager + fan-out tests   #
# ---------------------------------------------------------------------- #


class TestPrepareResourcesFansOutToInit:
    """``prepare_resources(scheduled_batch)`` should call
    ``on_request_init`` exactly once per newly-seen request (dedupe via
    _seen_req_ids), across all registered executors in EVENT_AXIS_ORDER."""

    def _make_req(self, req_id: int, state: LlmRequestState = LlmRequestState.CONTEXT_INIT):
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
        coord.on_batch_scheduled(self._make_batch([req], []))
        assert record == ["sp:on_request_init"]
        assert 42 in coord._seen_req_ids

    def test_dedupes_init_across_iterations(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(req_id=42)
        coord.on_batch_scheduled(self._make_batch([req], []))
        # Same request appears again in next iteration (still in batch)
        coord.on_batch_scheduled(self._make_batch([], [req]))
        # init must fire only once
        assert record == ["sp:on_request_init"]

    def test_fires_init_for_each_distinct_request(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        coord.on_batch_scheduled(
            self._make_batch([self._make_req(1), self._make_req(2)], [self._make_req(3)])
        )
        assert record == [
            "sp:on_request_init",
            "sp:on_request_init",
            "sp:on_request_init",
        ]
        assert coord._seen_req_ids == {1, 2, 3}

    def test_axis_order_in_init_fan_out(self, fake_kv_cache_manager):
        """EVENT_AXIS_ORDER for on_request_init is ['sparse', 'storage'];
        prepare_resources fan-out must respect this."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage])
        coord.on_batch_scheduled(self._make_batch([self._make_req(7)], []))
        assert record == [
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
        coord.on_iteration_end(self._make_batch([], []), MagicMock())
        assert record == ["sp:on_generation_step_end"]

    def test_context_end_fires_on_state_transition(self, fake_kv_cache_manager):
        """When a request was CONTEXT_INIT in the previous
        prepare/update and is GENERATION_IN_PROGRESS now → on_context_end."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        # Seed previous state via a first update with CONTEXT_INIT.
        req_v1 = self._make_req(42, LlmRequestState.CONTEXT_INIT)
        coord.on_iteration_end(self._make_batch([req_v1], []), MagicMock())
        record.clear()
        # Now the same request is GENERATION_IN_PROGRESS.
        req_v2 = self._make_req(42, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.on_iteration_end(self._make_batch([], [req_v2]), MagicMock())
        # on_context_end fires for the transition; on_generation_step_end
        # also fires once.
        assert record == [
            "sp:on_context_end",
            "sp:on_generation_step_end",
        ]

    def test_context_end_NOT_fired_if_no_transition(self, fake_kv_cache_manager):
        """A request that stays CONTEXT_INIT across iterations
        (chunked prefill) does NOT fire on_context_end."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        req = self._make_req(42, LlmRequestState.CONTEXT_INIT)
        coord.on_iteration_end(self._make_batch([req], []), MagicMock())
        record.clear()
        # Same state in next iter — no transition.
        coord.on_iteration_end(self._make_batch([req], []), MagicMock())
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
        coord.on_iteration_end(self._make_batch([req_v1], []), sparse_meta)
        req_v2 = self._make_req(7, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.on_iteration_end(self._make_batch([], [req_v2]), sparse_meta)
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
        coord.on_request_finished(req)
        assert record == ["sp:on_request_finish"]

    def test_clears_seen_req_ids_on_free(self, fake_kv_cache_manager):
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        coord = KVCacheBehaviorCoordinator(executors=[sparse])
        coord._seen_req_ids.add(42)
        coord._prev_req_state[42] = LlmRequestState.GENERATION_IN_PROGRESS
        coord.on_request_finished(self._make_req(42))
        # state cleared so a new request reusing the ID would init again
        assert 42 not in coord._seen_req_ids
        assert 42 not in coord._prev_req_state

    def test_axis_order_in_finish_fan_out(self, fake_kv_cache_manager):
        """EVENT_AXIS_ORDER for on_request_finish is ['sparse', 'storage']."""
        record = []
        sparse = _MockSparseManager(fake_kv_cache_manager, record, "sp")
        storage = _MockStorageManager(fake_kv_cache_manager, record, "st")
        coord = KVCacheBehaviorCoordinator(executors=[sparse, storage])
        coord.on_request_finished(self._make_req(7))
        assert record == [
            "sp:on_request_finish",
            "st:on_request_finish",
        ]


class TestCoordinatorIntegrationEndToEnd:
    """End-to-end test simulating a full request lifecycle through
    PyExecutor's resource_manager interface. Verifies all 6 hooks fire
    in the right order via the BaseResourceManager 3 callbacks."""

    def test_full_lifecycle(self, fake_kv_cache_manager):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

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
        coord.on_batch_scheduled(batch([r_ctx], []))
        coord.on_iteration_end(batch([r_ctx], []), MagicMock())
        # Iter 2: same request transitions to GENERATION_IN_PROGRESS
        r_gen = req(1, LlmRequestState.GENERATION_IN_PROGRESS)
        coord.on_batch_scheduled(batch([], [r_gen]))
        coord.on_iteration_end(batch([], [r_gen]), MagicMock())
        # Iter 3: request finishes
        coord.on_request_finished(req(1, LlmRequestState.GENERATION_COMPLETE))

        assert record == [
            "e:on_request_init",  # iter 1 prepare
            "e:on_generation_step_end",  # iter 1 update (no context_end yet)
            # iter 2 prepare: req already seen, no init
            "e:on_context_end",  # iter 2 update (transition!)
            "e:on_generation_step_end",  # iter 2 update
            "e:on_request_finish",  # iter 3 free
        ]


# ---------------------------------------------------------------------- #
# 12. End-to-end pipeline wire — ResourceManagerType + AttentionMetadata  #
# (coordinator registered into resource_manager + injected into           #
#  attn_metadata; locks the wire contract so future refactors keep the    #
#  pipeline connected.)                                                   #
# ---------------------------------------------------------------------- #


class TestModelEngineCoordinatorSetup:
    """Production wire pattern (Path A): the coordinator is passed to
    AttentionMetadata **at construction time**
    inside ``model_engine._set_up_attn_metadata``, not via
    post-construction mutation. This avoids the per-iter mutate-after-
    cached-build pattern and keeps the setup single-step.

    These source-level assertions lock the wire pattern.
    """

    def _get_set_up_attn_metadata_source(self):
        import inspect

        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

        return inspect.getsource(PyTorchModelEngine._set_up_attn_metadata)

    def test_no_post_construction_coordinator_mutation_at_callsite(self):
        """The model_engine call site that builds attn_metadata MUST NOT
        do ``attn_metadata.coordinator = ...`` after construction —
        coordinator is passed at construction time via the new
        resource_manager kwarg."""
        import inspect

        # Get the full model_engine source to scan for stale wire pattern.
        # We do this on the module since the callsite isn't in
        # _set_up_attn_metadata itself.
        import tensorrt_llm._torch.pyexecutor.model_engine as me_module

        full_src = inspect.getsource(me_module)
        # The bad pattern would be: ``attn_metadata.coordinator =``
        # (post-construction mutation). It should NOT appear anywhere.
        assert "attn_metadata.coordinator =" not in full_src, (
            "model_engine.py must NOT do post-construction "
            "``attn_metadata.coordinator = ...`` mutation. Pass "
            "``coordinator=`` via resource_manager kwarg to "
            "_set_up_attn_metadata instead — single-step setup."
        )


# ---------------------------------------------------------------------- #
# 13. TrtllmAttention.forward HOOK 2/4 callsite wiring                    #
# (TrtllmAttention reads metadata.coordinator and fires on_*_attention    #
#  for behavior-layer sparse methods. Source-level assertions lock the    #
#  wire pattern.)                                                         #
# ---------------------------------------------------------------------- #


class TestTrtllmAttentionForwardCallsiteWiring:
    """The TrtllmAttention.forward path now invokes
    ``metadata.coordinator.on_context_attention`` and
    ``metadata.coordinator.on_generation_attention`` for behavior-layer
    methods. These source-level assertions lock the wire pattern so future
    refactors of TrtllmAttention don't silently drop the coordinator path.
    """

    def _get_forward_source(self):
        import inspect

        from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

        return inspect.getsource(TrtllmAttention.forward)

    def test_callsite_invokes_on_context_attention(self):
        src = self._get_forward_source()
        assert "metadata.coordinator.on_context_attention(" in src, (
            "TrtllmAttention.forward must invoke "
            "``metadata.coordinator.on_context_attention(...)`` for "
            "behavior-layer methods — this is HOOK 2 of the 6-hook "
            "framework. Without this callsite the Coordinator dispatch "
            "method exists but never gets called from production code."
        )

    def test_callsite_invokes_on_generation_attention(self):
        src = self._get_forward_source()
        assert "metadata.coordinator.on_generation_attention(" in src, (
            "TrtllmAttention.forward must invoke "
            "``metadata.coordinator.on_generation_attention(...)`` — HOOK 4."
        )

    def test_callsite_guards_for_none_coordinator(self):
        """When no behavior-layer sparse method is configured,
        metadata.coordinator is None; the callsite must skip dispatch
        instead of raising AttributeError."""
        src = self._get_forward_source()
        assert "metadata.coordinator is not None" in src, (
            "TrtllmAttention.forward must guard the coordinator dispatch "
            "with ``if metadata.coordinator is not None:`` — models without "
            "behavior-layer sparse method leave coordinator at None and "
            "must keep working unchanged."
        )

    def test_callsite_inside_behavior_layer_branch(self):
        """The coordinator dispatch should live inside the
        ``if metadata.coordinator is not None:`` branch — methods without
        a coordinator keep using ``sparse_kv_predict`` /
        ``sparse_attn_predict``."""
        src = self._get_forward_source()
        # Coordinator call must live after the ``coordinator is not None``
        # guard and before the legacy ``else`` (sparse_kv_predict).
        guard_idx = src.find("metadata.coordinator is not None")
        legacy_idx = src.find("self.sparse_kv_predict(")
        coord_idx = src.find("metadata.coordinator.on_context_attention(")
        assert guard_idx >= 0 and legacy_idx >= 0 and coord_idx >= 0, (
            "Expected branches not found in TrtllmAttention.forward source"
        )
        assert guard_idx < coord_idx < legacy_idx, (
            "Coordinator dispatch must be inside the coordinator-present "
            "branch, not in the legacy memory-layer path."
        )


# ---------------------------------------------------------------------- #
# Standalone coordinator + independent (first-class) registration         #
# ---------------------------------------------------------------------- #


class TestCoordinatorIsStandalone:
    """The coordinator is a plain object, NOT a BaseResourceManager, and is
    NOT placed in the resource-manager registry (independent registration)."""

    def test_not_a_base_resource_manager(self, fake_kv_cache_manager):
        from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager

        assert not issubclass(KVCacheBehaviorCoordinator, BaseResourceManager)

    def test_no_resource_manager_protocol_methods(self, fake_kv_cache_manager):
        coord = KVCacheBehaviorCoordinator(executors=[])
        # RM-protocol cruft must be gone from the standalone coordinator.
        for attr in (
            "prepare_resources",
            "update_resources",
            "free_resources",
            "get_max_resource_count",
            "get_needed_resource_to_completion",
        ):
            assert not hasattr(coord, attr), (
                f"{attr} should not exist on the standalone coordinator"
            )

    def test_lifecycle_driving_methods_present(self, fake_kv_cache_manager):
        coord = KVCacheBehaviorCoordinator(executors=[])
        for attr in ("on_batch_scheduled", "on_iteration_end", "on_request_finished"):
            assert callable(getattr(coord, attr))

    def test_enum_removed(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

        assert not hasattr(ResourceManagerType, "KV_CACHE_BEHAVIOR_COORDINATOR"), (
            "coordinator is no longer a resource manager; the enum must be gone"
        )


class TestFirstClassRegistrationWiring:
    """model_engine holds the coordinator as a first-class member
    (self.kv_behavior_coordinator) and PyExecutor drives it from there."""

    def test_model_engine_reads_first_class_member(self):
        import inspect

        from tensorrt_llm._torch.pyexecutor import model_engine as me

        src = inspect.getsource(me)
        assert "self.kv_behavior_coordinator" in src, (
            "model_engine must read the first-class self.kv_behavior_coordinator"
        )
        assert "ResourceManagerType.KV_CACHE_BEHAVIOR_COORDINATOR" not in src, (
            "must NOT fetch the coordinator from the resource-manager registry"
        )

    def test_py_executor_drives_coordinator_lifecycle(self):
        import inspect

        from tensorrt_llm._torch.pyexecutor import py_executor as pe

        src = inspect.getsource(pe)
        for entry in ("on_batch_scheduled", "on_iteration_end", "on_request_finished"):
            assert f"kv_behavior_coordinator.{entry}" in src, (
                f"PyExecutor must drive coordinator.{entry} explicitly"
            )
