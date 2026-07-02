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
  - Capacity-only V2 registration and immediate compacted-capacity resize.

These tests do not run real eviction or attention; that needs model weights and
is covered by the NIAH end-to-end run.
"""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention import TriAttention
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _build_swa_rebase_copy,
    _build_swa_rebase_keep,
)

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
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

        expected = {"TriAttention"}
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

    def test_factory_in_util_base_in_resource_manager(self):
        from tensorrt_llm._torch.pyexecutor import _util
        from tensorrt_llm._torch.pyexecutor import resource_manager as rm

        assert _util.create_kv_cache_compression_manager is create_kv_cache_compression_manager
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
        assert cfg.top_B == 2048
        assert cfg.beta == 128
        assert cfg.window_size == 128
        assert cfg.eviction_mode == "union"
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

    @pytest.mark.parametrize(
        ("pin_prefill", "count_prompt_tokens"),
        [(False, False), (True, True)],
    )
    def test_physical_reclaim_requires_pinned_decode_only_budget(
        self, pin_prefill, count_prompt_tokens
    ):
        with pytest.raises(ValueError, match="pin_prefill=True"):
            TriAttention(
                _make_fake_v2(),
                top_B=8,
                pin_prefill=pin_prefill,
                count_prompt_tokens=count_prompt_tokens,
            )

    def test_request_init_registers_capacity_only_for_every_request(self):
        from types import SimpleNamespace

        manager = _make_fake_v2()
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True
        first = SimpleNamespace(py_request_id=11)
        second = SimpleNamespace(py_request_id=12)

        triattention.on_request_init(first)
        triattention.on_request_init(second)

        assert first.py_kv_cache_decode_capacity_only
        assert second.py_kv_cache_decode_capacity_only

    def test_request_init_rejects_native_v2_swa(self):
        from types import SimpleNamespace

        manager = _make_fake_v2()
        manager.max_attention_window_vec = [128]
        triattention = TriAttention(manager, top_B=128, skip_swa=False)
        triattention._calibrated = True
        request = SimpleNamespace(py_request_id=11)

        with pytest.raises(ValueError, match="full-attention V2 lifecycles"):
            triattention.on_request_init(request)
        assert not hasattr(request, "py_kv_cache_decode_capacity_only")

    def test_request_init_rejects_attention_dp(self):
        from types import SimpleNamespace

        manager = _make_fake_v2()
        manager.mapping = SimpleNamespace(enable_attention_dp=True)
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True

        with pytest.raises(ValueError, match="attention DP"):
            triattention.on_request_init(SimpleNamespace(py_request_id=11))

    def test_request_init_rejects_speculative_capacity(self):
        from types import SimpleNamespace

        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 4
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True

        with pytest.raises(ValueError, match="single-token"):
            triattention.on_request_init(SimpleNamespace(py_request_id=11))

    def test_skip_swa_requires_model_path(self):
        with pytest.raises(ValueError, match="skip_swa=True requires model_path"):
            TriAttention(_make_fake_v2(), top_B=8)

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
# adjust_attention_metadata: use V2's authoritative pre-forward physical length
# and clamp prompt_lens to that compacted prefix.
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

    def test_authoritative_pre_forward_length_replaces_logical_length(self):
        mgr = TriAttention.__new__(TriAttention)
        mgr._pre_forward_kv_lengths = {7: 91}
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq[0] == 91
        assert meta.prompt_lens == [50]  # 50 <= 91, not clamped

    def test_prompt_len_clamped_to_compacted_cache(self):
        # A prompt longer than the whole compacted cache is clamped to num_cached.
        mgr = TriAttention.__new__(TriAttention)
        mgr._pre_forward_kv_lengths = {7: 91}
        meta = _FakeMetadata([100], [200], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq[0] == 91
        assert meta.prompt_lens == [91]

    def test_request_without_authoritative_length_is_untouched(self):
        mgr = TriAttention.__new__(TriAttention)
        mgr._pre_forward_kv_lengths = {}
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100]
        assert meta.prompt_lens == [50]

    def test_context_requests_skipped(self):
        # Only generation requests (index >= num_contexts) are reconciled.
        mgr = TriAttention.__new__(TriAttention)
        mgr._pre_forward_kv_lengths = {1: 80, 2: 91}
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
        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4, skip_swa=False)
        assert mgr.evicted_count(12345) == 0

    @staticmethod
    def _make_due_decode_request(seq_len):
        from types import SimpleNamespace
        from unittest import mock

        request = SimpleNamespace(
            py_request_id=7,
            py_prompt_len=1024,
            max_beam_num_tokens=seq_len + 1,
        )
        batch = SimpleNamespace(generation_requests=[request])
        mgr = TriAttention.__new__(TriAttention)
        mgr._calibrated = True
        cache = SimpleNamespace(
            capacity=seq_len + 1,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        mgr.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda *args, **kwargs: None,
            kv_cache_map={7: cache},
            _stream=mock.Mock(),
        )
        mgr._L = 2
        mgr._gen_steps = {7: 127}
        mgr._evicted = {}
        mgr._pre_forward_kv_lengths = {}
        mgr._capacity_only_request_ids = {7}
        mgr.beta = 128
        mgr.top_B = 4096
        mgr.pin_prefill = True
        mgr.count_prompt_tokens = False
        return mgr, request, batch

    def test_identity_gate_skips_exact_decode_only_budget(self):
        from unittest import mock

        mgr, _, batch = self._make_due_decode_request(seq_len=1024 + 4096)

        with mock.patch.object(mgr, "_evict_requests") as evict:
            mgr._periodic_evict(batch)

        evict.assert_not_called()
        assert mgr._gen_steps[7] == 128

    def test_identity_gate_preserves_real_eviction_round(self):
        from unittest import mock

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        timeline = []
        event = mock.Mock()
        event.record.side_effect = lambda: timeline.append("event")
        cache = mgr.kv_cache_manager.kv_cache_map[7]

        def compact(*args):
            timeline.append("compact")
            return [(7, 1024 + 4096 + 1)]

        with (
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            mgr._periodic_evict(batch)

        evict.assert_called_once_with([(request, 7)], 2)
        event.record.assert_called_once_with()
        cache.resize.assert_not_called()
        assert request.py_kv_cache_compaction == (
            1024 + 4096 + 1,
            1024 + 4096 + 2,
            event,
        )
        assert timeline == ["compact", "event"]

    def test_pending_compaction_uses_effective_length_and_defers_eviction(self):
        from unittest import mock

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        cache = mgr.kv_cache_manager.kv_cache_map[7]
        published_capacity = cache.capacity
        request.py_kv_cache_compaction = (
            published_capacity,
            published_capacity,
            mock.Mock(),
        )
        cache.capacity += 1

        with mock.patch.object(mgr, "_evict_requests") as evict:
            mgr._periodic_evict(batch)

        evict.assert_not_called()
        assert mgr._gen_steps[7] == 127
        assert mgr._pre_forward_kv_lengths[7] == 1024 + 4096 + 2

    def test_generation_only_request_is_initialized_once(self):
        from types import SimpleNamespace

        manager = SimpleNamespace(
            enable_block_reuse=False,
            get_buffers=lambda *args, **kwargs: None,
            kv_cache_map={7: SimpleNamespace(capacity=11, history_length=2, is_active=True)},
        )
        request = SimpleNamespace(
            py_request_id=7,
            py_prompt_len=2,
            max_beam_num_tokens=999,
            is_dummy=False,
        )
        mgr = TriAttention(manager, top_B=8, beta=4, skip_swa=False)
        mgr._calibrated = True
        mgr._L = 2

        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))
        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert request.py_kv_cache_decode_capacity_only
        assert mgr._pre_forward_kv_lengths == {7: 10}
        assert mgr._capacity_only_request_ids == {7}

    def test_generation_dummy_is_skipped(self):
        from types import SimpleNamespace

        manager = SimpleNamespace(enable_block_reuse=False)
        request = SimpleNamespace(py_request_id=7, is_dummy=True)
        mgr = TriAttention(manager, top_B=8, beta=4, skip_swa=False)
        mgr._calibrated = True

        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert not hasattr(request, "py_kv_cache_decode_capacity_only")
        assert mgr._capacity_only_request_ids == set()

    def test_request_finish_orders_unconsumed_compaction_before_free(self):
        from types import SimpleNamespace
        from unittest import mock

        event = mock.Mock()
        manager = SimpleNamespace(_stream=mock.Mock())
        request = SimpleNamespace(
            py_request_id=7,
            py_kv_cache_decode_capacity_only=True,
            py_kv_cache_compaction=(129, 256, event),
        )
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = manager
        mgr._gen_steps = {7: 1}
        mgr._evicted = {7: 127}
        mgr._pre_forward_kv_lengths = {7: 128}
        mgr._capacity_only_request_ids = {7}

        mgr.on_request_finish(request)

        manager._stream.wait_event.assert_called_once_with(event)
        assert not request.py_kv_cache_decode_capacity_only
        assert request.py_kv_cache_compaction is None
        assert mgr._capacity_only_request_ids == set()

    def test_evict_requests_returns_exact_post_forward_capacity(self):
        from types import SimpleNamespace
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        # The reserved-but-unwritten slot may contain nonzero data from a reused
        # page. Length accounting must use request metadata, not pool contents.
        pools = [torch.ones(2, 2, 1, 8, 2), torch.ones(2, 2, 1, 8, 2)]
        manager = SimpleNamespace(get_buffers=lambda layer, **kwargs: pools[layer])
        request = SimpleNamespace(py_request_id=7, py_prompt_len=2, max_beam_num_tokens=999)
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = manager
        mgr._evicted = {7: 5}
        mgr._pre_forward_kv_lengths = {7: 8}
        mgr.top_B = 4
        mgr.pin_prefill = True
        mgr.count_prompt_tokens = False
        mgr.eviction_mode = "union"
        mgr._offsets = torch.ones(1)
        mgr._H = 1
        mgr._triattn_q_real = torch.ones(2, 1, 1)
        mgr._triattn_q_imag = torch.ones(2, 1, 1)
        mgr._triattn_mlr_coef = torch.ones(2, 1, 1)
        mgr._freq_scale_sq = torch.ones(1)
        mgr.calibration = {"omega": torch.ones(1)}
        mgr.score_aggregation = "mean"
        mgr._attention_layer_partition = mock.Mock(return_value=([1], [0], 2))
        mgr._resolve_page_ids = mock.Mock(
            side_effect=lambda request, layer: [10, 11] if layer == 1 else [20, 21]
        )
        mgr._evict_modes = mock.Mock(return_value=torch.arange(6))
        segment = SimpleNamespace(request_index=0, layer_index=1)

        with (
            mock.patch.object(
                kernels,
                "triton_tri_score_perhead",
                return_value=(torch.empty(0), torch.empty(0), [segment]),
            ) as score,
            mock.patch.object(kernels, "flat_perhead_to_list", return_value=[torch.zeros(1, 8)]),
            mock.patch.object(kernels, "triton_tri_compact") as compact,
        ):
            targets = mgr._evict_requests([(request, 7)], num_layers=2)

        assert targets == [(7, 7)]
        assert mgr._evicted == {7: 7}
        assert mgr._pre_forward_kv_lengths == {7: 6}
        assert score.call_args.kwargs["layer_indices"] == [1]
        assert score.call_args.args[1][0].tolist() == [10, 11]
        assert score.call_args.args[2] == [8]
        assert score.call_args.args[3] == [13.0]
        assert mgr._resolve_page_ids.call_args_list == [
            mock.call(request, 1),
            mock.call(request, 0),
        ]
        assert compact.call_count == 2
        dense_call, swa_call = compact.call_args_list
        assert dense_call.args[0] is pools[1]
        assert dense_call.args[1][0].tolist() == [10, 11]
        assert swa_call.args[0] is pools[0]
        assert swa_call.args[1][0].tolist() == [20, 21]
        assert swa_call.args[2][0].tolist() == [6, 7]
        assert swa_call.kwargs["dest_list"][0].tolist() == [4, 5]
        assert swa_call.args[2][0].numel() == 2

    @pytest.mark.parametrize(
        "draft_fields",
        [
            {"py_draft_tokens": [101, 102]},
            {"py_draft_tokens": [], "num_draft_tokens": 2},
        ],
    )
    def test_evict_requests_rejects_speculative_decode(self, draft_fields):
        from types import SimpleNamespace
        from unittest import mock

        pool = torch.zeros(2, 2, 1, 8, 2)
        request = SimpleNamespace(
            py_request_id=7,
            py_prompt_len=2,
            max_beam_num_tokens=9,
            **draft_fields,
        )
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = SimpleNamespace(get_buffers=lambda *args, **kwargs: pool)
        mgr._evicted = {}
        mgr._attention_layer_partition = mock.Mock(return_value=([0], [], None))

        with pytest.raises(ValueError, match="does not support speculative decoding"):
            mgr._evict_requests([(request, 7)], num_layers=1)

    def test_dense_storage_groups_use_their_own_page_ids(self):
        from types import SimpleNamespace
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        pools = [torch.zeros(2, 2, 1, 8, 2), torch.zeros(2, 2, 1, 8, 2)]
        request = SimpleNamespace(py_request_id=7, py_prompt_len=2, max_beam_num_tokens=9)
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = SimpleNamespace(get_buffers=lambda layer, **kwargs: pools[layer])
        mgr._evicted = {}
        mgr._pre_forward_kv_lengths = {7: 8}
        mgr.top_B = 4
        mgr.pin_prefill = True
        mgr.count_prompt_tokens = False
        mgr.eviction_mode = "union"
        mgr._offsets = torch.ones(1)
        mgr._H = 1
        mgr._triattn_q_real = torch.ones(2, 1, 1)
        mgr._triattn_q_imag = torch.ones(2, 1, 1)
        mgr._triattn_mlr_coef = torch.ones(2, 1, 1)
        mgr._freq_scale_sq = torch.ones(1)
        mgr.calibration = {"omega": torch.ones(1)}
        mgr.score_aggregation = "mean"
        mgr._attention_layer_partition = mock.Mock(return_value=([0, 1], [], None))
        mgr._resolve_page_ids = mock.Mock(
            side_effect=lambda request, layer: [10, 11] if layer == 0 else [20, 21]
        )
        mgr._evict_modes = mock.Mock(return_value=torch.arange(6))

        def score_group(*args, **kwargs):
            layer = kwargs["layer_indices"][0]
            segment = SimpleNamespace(request_index=0, layer_index=layer)
            return torch.empty(0), torch.empty(0), [segment]

        with (
            mock.patch.object(
                kernels, "triton_tri_score_perhead", side_effect=score_group
            ) as score,
            mock.patch.object(kernels, "flat_perhead_to_list", return_value=[torch.zeros(1, 8)]),
            mock.patch.object(kernels, "triton_tri_compact") as compact,
        ):
            targets = mgr._evict_requests([(request, 7)], num_layers=2)

        assert targets == [(7, 7)]
        assert score.call_count == 2
        first_score, second_score = score.call_args_list
        assert first_score.kwargs["layer_indices"] == [0]
        assert first_score.args[1][0].tolist() == [10, 11]
        assert second_score.kwargs["layer_indices"] == [1]
        assert second_score.args[1][0].tolist() == [20, 21]
        assert compact.call_count == 2
        first_compact, second_compact = compact.call_args_list
        assert first_compact.args[0] is pools[0]
        assert first_compact.args[1][0].tolist() == [10, 11]
        assert second_compact.args[0] is pools[1]
        assert second_compact.args[1][0].tolist() == [20, 21]


class TestTopKRouting:
    def test_score_width_4096_routes_to_torch_topk(self):
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        scores = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
        real_topk = torch.topk

        with mock.patch.object(torch, "topk", wraps=real_topk) as topk:
            result = mgr._indexer_topk_idx(scores, 8)

        topk.assert_called_once()
        args, kwargs = topk.call_args
        assert args[0] is scores
        assert args[1:] == (8,)
        assert kwargs == {"dim": 1, "sorted": False}
        assert result.shape == (2, 8)
        assert result.dtype == torch.long


class TestKeptOnlyCompaction:
    def test_retained_prefix_matches_legacy_full_reorder(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            _build_compaction_indices,
        )

        seq_len = 17
        keep = torch.tensor([0, 2, 5, 9, 16], dtype=torch.long)
        source = torch.arange(2 * 3 * seq_len * 4).reshape(2, 3, seq_len, 4)

        kept_src, kept_dst = _build_compaction_indices(keep, seq_len, kept_only=True)
        legacy_src, legacy_dst = _build_compaction_indices(keep, seq_len, kept_only=False)
        kept_result = torch.full_like(source, -1)
        legacy_result = torch.full_like(source, -1)
        kept_result[:, :, kept_dst] = source[:, :, kept_src]
        legacy_result[:, :, legacy_dst] = source[:, :, legacy_src]

        assert torch.equal(kept_result[:, :, : keep.numel()], legacy_result[:, :, : keep.numel()])
        assert kept_src.numel() == keep.numel()
        assert legacy_src.numel() == seq_len


class TestKernelMaskedSwa:
    def test_rebase_boundary_keeps_exact_latest_window(self):
        keep = _build_swa_rebase_keep(seq_len=10, keep_count=4, window_size=4)

        assert keep.tolist() == [6, 7, 8, 9]
        assert keep.numel() == 4

    def test_rebase_preserves_prefix_and_places_latest_window_at_tail(self):
        keep = _build_swa_rebase_keep(seq_len=12, keep_count=7, window_size=3)

        assert keep.tolist() == [0, 1, 2, 3, 9, 10, 11]
        assert keep[-3:].tolist() == [9, 10, 11]

    def test_rebase_copy_moves_only_window_to_compacted_tail(self):
        source, destination = _build_swa_rebase_copy(seq_len=12, keep_count=7, window_size=3)

        assert source.tolist() == [9, 10, 11]
        assert destination.tolist() == [4, 5, 6]
        assert source.numel() == destination.numel() == 3

    def test_rebase_rejects_budget_smaller_than_window(self):
        with pytest.raises(ValueError, match="at least the SWA window size"):
            _build_swa_rebase_keep(seq_len=10, keep_count=3, window_size=4)

    def test_layer_partition_uses_local_model_config(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(num_layers=4)
        config = SimpleNamespace(
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=128,
        )

        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config) as load:
            dense, sliding, window = mgr._attention_layer_partition(4)

        load.assert_called_once_with(
            "/models/gpt-oss", trust_remote_code=True, local_files_only=True
        )
        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128

    def test_layer_partition_rejects_decode_budget_smaller_than_window(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 127
        mgr.kv_cache_manager = SimpleNamespace(num_layers=2)
        config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=128,
        )

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            pytest.raises(ValueError, match="decode budget top_B=127"),
        ):
            mgr._attention_layer_partition(2)

    def test_layer_partition_uses_pp_local_to_global_mapping(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[1, 2], num_layers=4)
        config = SimpleNamespace(
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=128,
        )

        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
            dense, sliding, window = mgr._attention_layer_partition(2)

        assert dense == [0]
        assert sliding == [1]
        assert window == 128

    def test_layer_partition_rejects_ambiguous_window_metadata(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        mgr.skip_swa = True
        mgr.model_path = "/models/ambiguous"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(num_layers=2)
        config = SimpleNamespace(layer_types=None, sliding_window=128)

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            pytest.raises(ValueError, match="cannot classify"),
        ):
            mgr._attention_layer_partition(2)

    def test_layer_partition_honors_explicit_disabled_sliding_window(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        mgr.skip_swa = True
        mgr.model_path = "/models/qwen3"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(num_layers=2)
        config = SimpleNamespace(
            layer_types=None,
            sliding_window=None,
            max_window_layers=36,
            use_sliding_window=False,
        )

        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
            dense, sliding, window = mgr._attention_layer_partition(2)

        assert dense == [0, 1]
        assert sliding == []
        assert window is None

    def test_pp_page_lookup_uses_global_layer_id(self):
        from types import SimpleNamespace
        from unittest import mock

        get_batch = mock.Mock(return_value=[[10, 11]])
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = SimpleNamespace(
            pp_layers=[9, 10],
            num_layers=36,
            get_batch_cache_indices=get_batch,
        )
        request = SimpleNamespace(py_request_id=7)

        assert mgr._resolve_page_ids(request, 0) == [10, 11]
        get_batch.assert_called_once_with([7], 9)

    def test_page_lookup_failure_is_not_hidden(self):
        from types import SimpleNamespace
        from unittest import mock

        get_batch = mock.Mock(side_effect=KeyError("missing pool"))
        mgr = TriAttention.__new__(TriAttention)
        mgr.kv_cache_manager = SimpleNamespace(
            pp_layers=[9],
            num_layers=36,
            get_batch_cache_indices=get_batch,
        )
        request = SimpleNamespace(py_request_id=7)

        with pytest.raises(RuntimeError, match="Failed to resolve KV pages"):
            mgr._resolve_page_ids(request, 0)


# ---------------------------------------------------------------------------
# Block reclaim uses V2's existing resize operation; no manager subclass is needed.
# ---------------------------------------------------------------------------


class TestNoBlockFreeSubclass:
    def test_subclass_module_removed(self):
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(
                "tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kv_manager"
            )

    def test_subclass_not_exported(self):
        import tensorrt_llm._torch.kv_cache_compression.triattention as pkg

        assert "TriAttentionKVCacheManagerV2" not in pkg.__all__
        assert not hasattr(pkg, "TriAttentionKVCacheManagerV2")

    def test_v2_has_no_triattention_specific_capacity_api(self):
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

        assert not hasattr(KVCacheManagerV2, "enable_decode_capacity_only")


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
            top_B=32,
            beta=16,
            calibration_path=flat_calibration_pt,
            skip_swa=False,
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
        cfg = TriAttentionKvCacheCompressionConfig(top_B=32, beta=16, skip_swa=False)
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.kv_cache_manager is fake_v2

    def test_factory_propagates_eviction_mode(self):
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=64,
            beta=8,
            eviction_mode="per_head",
            skip_swa=False,
        )
        mgr = create_kv_cache_compression_manager(
            cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=False)
        )
        assert isinstance(mgr, TriAttention)
        assert mgr.eviction_mode == "per_head"
