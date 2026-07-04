# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    _FixedUnionWorkspace,
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

        assert first.py_kv_cache_generation_capacity_only
        assert second.py_kv_cache_generation_capacity_only

    def test_request_init_rejects_native_v2_swa(self):
        from types import SimpleNamespace

        manager = _make_fake_v2()
        manager.max_attention_window_vec = [128]
        triattention = TriAttention(manager, top_B=128, skip_swa=False)
        triattention._calibrated = True
        request = SimpleNamespace(py_request_id=11)

        with pytest.raises(ValueError, match="full-attention V2 lifecycles"):
            triattention.on_request_init(request)
        assert not hasattr(request, "py_kv_cache_generation_capacity_only")

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

    def test_step_begin_hook_documents_deferred_v2_resize_boundary(self):
        doc = TriAttention.on_generation_step_begin.__doc__

        assert "publish a K+1 target" in doc
        assert "post-forward update boundary" in doc
        assert "resize V2 to the compacted capacity" not in doc

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
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        timeline = []
        event = mock.Mock()
        event.record.side_effect = lambda: timeline.append("event")
        cache = mgr.kv_cache_manager.kv_cache_map[7]

        def compact(*args):
            timeline.append("compact")
            return [(7, 1024 + 4096 + 1)]

        @contextlib.contextmanager
        def track_range(name, **kwargs):
            timeline.append(f"enter:{name}")
            yield
            timeline.append(f"exit:{name}")

        with (
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
            mock.patch.object(tri_module, "nvtx_range", side_effect=track_range),
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
        assert timeline == [
            "compact",
            "enter:triattention.publish",
            "event",
            "exit:triattention.publish",
        ]

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

        assert request.py_kv_cache_generation_capacity_only
        assert mgr._pre_forward_kv_lengths == {7: 10}
        assert mgr._capacity_only_request_ids == {7}

    def test_generation_dummy_is_skipped(self):
        from types import SimpleNamespace

        manager = SimpleNamespace(enable_block_reuse=False)
        request = SimpleNamespace(py_request_id=7, is_dummy=True)
        mgr = TriAttention(manager, top_B=8, beta=4, skip_swa=False)
        mgr._calibrated = True

        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert not hasattr(request, "py_kv_cache_generation_capacity_only")
        assert mgr._capacity_only_request_ids == set()

    def test_request_finish_orders_unconsumed_compaction_before_free(self):
        from types import SimpleNamespace
        from unittest import mock

        event = mock.Mock()
        manager = SimpleNamespace(_stream=mock.Mock())
        request = SimpleNamespace(
            py_request_id=7,
            py_kv_cache_generation_capacity_only=True,
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
        assert not request.py_kv_cache_generation_capacity_only
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
        import contextlib
        from types import SimpleNamespace
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
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
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ) as nvtx,
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
        assert [call.args[0] for call in nvtx.call_args_list] == [
            "triattention.metadata",
            "triattention.score",
            "triattention.score",
            "triattention.select",
            "triattention.compact",
            "triattention.compact",
        ]


class TestTopKRouting:
    def test_indexer_route_predicate_preserves_prior_dispatch_domain(self):
        mgr = TriAttention.__new__(TriAttention)

        for width in (1, 4095, 4096, 8191):
            for k in (1, 2048, 2049, 4096):
                prior_route = not (
                    k > mgr._INDEXER_TOPK_MAX_K or width >= 2 * mgr._INDEXER_TOPK_SUBBLOCK
                )
                assert mgr._indexer_topk_supported(width, k) is prior_route

    def test_score_width_4095_k2048_routes_to_native_indexer_topk(self):
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        scores = torch.arange(2 * 4095, dtype=torch.float32).reshape(2, 4095)

        def fake_indexer(values, seq_lens, output, next_n, k):
            assert values.shape == scores.shape
            assert values.dtype == scores.dtype
            assert values.device == scores.device
            assert seq_lens.tolist() == [4095, 4095]
            assert next_n == 1
            assert k == 2048
            output.copy_(torch.arange(k, dtype=torch.int32, device=output.device).expand(2, -1))

        with (
            mock.patch.object(
                torch.ops.trtllm,
                "indexer_topk_decode",
                side_effect=fake_indexer,
            ) as indexer,
            mock.patch.object(
                torch,
                "topk",
                side_effect=AssertionError("4095 must stay on native IndexerTopK"),
            ),
        ):
            result = mgr._indexer_topk_idx(scores, 2048)

        indexer.assert_called_once()
        assert result.shape == (2, 2048)
        assert result.dtype == torch.long
        assert torch.equal(result[0], torch.arange(2048))

    def test_score_width_4096_k2048_routes_to_torch_topk(self):
        from unittest import mock

        mgr = TriAttention.__new__(TriAttention)
        scores = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
        real_topk = torch.topk

        with mock.patch.object(torch, "topk", wraps=real_topk) as topk:
            result = mgr._indexer_topk_idx(scores, 2048)

        topk.assert_called_once()
        args, kwargs = topk.call_args
        assert args[0] is scores
        assert args[1:] == (2048,)
        assert kwargs == {"dim": 1, "sorted": False}
        assert result.shape == (2, 2048)
        assert result.dtype == torch.long


def _torch_tri_score_oracle(
    layer_pools,
    page_ids,
    seq_lens,
    round_starts,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    layer_indices,
    aggregation,
):
    """Independent Torch implementation of the paged TriAttention score."""
    scores = []
    num_q_heads = int(q_real.shape[1])
    for request, seq_len in enumerate(seq_lens):
        phase = (round_starts[request] + offsets[:, None]) * omega[None, :]
        mean_cos = torch.cos(phase).mean(dim=0)
        mean_sin = torch.sin(phase).mean(dim=0)
        for layer in layer_indices:
            pool = layer_pools[layer]
            request_page_ids = (
                page_ids[layer][request] if isinstance(page_ids, dict) else page_ids[request]
            )
            keys = (
                pool.index_select(0, request_page_ids)[:, 0]
                .permute(1, 0, 2, 3)
                .reshape(pool.shape[2], -1, pool.shape[4])[:, :seq_len]
                .float()
            )
            num_kv_heads = int(keys.shape[0])
            group_size = num_q_heads // num_kv_heads
            head_scores = []
            for head in range(num_q_heads):
                key = keys[head // group_size]
                num_freqs = int(key.shape[-1]) // 2
                key_real = key[:, :num_freqs]
                key_imag = key[:, num_freqs:]
                product_real = q_real[layer, head] * key_real + q_imag[layer, head] * key_imag
                product_imag = q_imag[layer, head] * key_real - q_real[layer, head] * key_imag
                if aggregation == "mean":
                    position = (
                        freq_scale_sq * (product_real * mean_cos - product_imag * mean_sin)
                    ).sum(dim=-1)
                else:
                    position = (
                        (
                            freq_scale_sq[None, None, :]
                            * (
                                product_real[None] * torch.cos(phase)[:, None, :]
                                - product_imag[None] * torch.sin(phase)[:, None, :]
                            )
                        )
                        .sum(dim=-1)
                        .max(dim=0)
                        .values
                    )
                mlr = (
                    torch.sqrt(key_real.square() + key_imag.square())
                    * mlr_coef[layer, head]
                    * freq_scale_sq
                ).sum(dim=-1)
                head_scores.append(position + mlr)
            scores.append(torch.stack(head_scores))
    return scores


def _logical_kv(pool, page_ids, seq_len):
    return (
        pool.index_select(0, page_ids)
        .permute(1, 2, 0, 3, 4)
        .reshape(pool.shape[1], pool.shape[2], -1, pool.shape[4])[:, :, :seq_len]
    )


def _torch_union_keep(head_scores, prompt_len, budget):
    decode = head_scores[:, prompt_len:]
    decode = (decode - decode.mean(dim=1, keepdim=True)) / decode.std(
        dim=1, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    combined = decode.max(dim=0).values
    row_top = torch.topk(decode, budget, dim=1, sorted=False).indices
    union = torch.unique(row_top, sorted=True)
    assert union.numel() >= budget
    selected = union.index_select(
        0,
        torch.topk(combined.index_select(0, union), budget, sorted=False).indices,
    )
    prompt = torch.arange(prompt_len, dtype=torch.long, device=head_scores.device)
    return torch.sort(torch.cat([prompt, selected + prompt_len])).values


class TestFixedScoreMetadata:
    def test_stage2_gate_requires_stage1_prewarm(self):
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "1",
                "TRIATTN_FIXED_PREWARM": "0",
                "TRIATTN_FIXED_SCORE_METADATA": "1",
            },
        ):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert not manager._fixed_score_metadata_enabled

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "1",
                "TRIATTN_FIXED_PREWARM": "1",
                "TRIATTN_FIXED_SCORE_METADATA": "1",
            },
        ):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert manager._fixed_score_metadata_enabled

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_ready_workspace_requires_uniform_exact_runtime_geometry(self, request_count):
        from types import SimpleNamespace
        from unittest import mock

        manager = TriAttention.__new__(TriAttention)
        manager._fixed_score_metadata_enabled = True
        manager._fixed_union_prewarm_key = mock.Mock(return_value=("bucket",))
        workspace = SimpleNamespace(
            max_requests=8,
            matches=mock.Mock(return_value=True),
            prewarm_key=("bucket",),
        )
        manager._fixed_score_workspaces = {("bucket",): workspace}
        manager._fixed_score_prewarm_states = {("bucket",): "ready"}
        manager._fixed_score_runtime_counts = {}
        pools = [torch.empty(1), torch.empty(1)]
        prepared = [
            {"request": SimpleNamespace(py_prompt_len=2), "seq_len": 8}
            for _ in range(request_count)
        ]

        result = manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared)

        assert result is workspace
        workspace.matches.assert_called_once_with(pools, [[0], [1]], [0, 1])
        workspace.matches.return_value = False
        assert (
            manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared) is None
        )
        assert manager._fixed_score_runtime_counts[("bucket",)]["fallback"] == 1
        workspace.matches.return_value = True
        manager._fixed_score_prewarm_states[("bucket",)] = "failed"
        assert (
            manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared) is None
        )
        assert manager._fixed_score_runtime_counts[("bucket",)]["fallback"] == 2

    def test_busy_pinned_staging_falls_back_without_querying_page_tables(self):
        from types import SimpleNamespace
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        workspace = _FixedScoreMetadataWorkspace.__new__(_FixedScoreMetadataWorkspace)
        workspace.device = torch.device("cuda")
        workspace.max_requests = 8
        stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=4)
        workspace.stream = stream
        workspace.copy_pending = True
        workspace.copy_done = SimpleNamespace(query=mock.Mock(return_value=False))
        get_batch = mock.Mock(side_effect=AssertionError("busy workspace queried V2"))

        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            assert not workspace.stage(get_batch, [1], [8.0])
        workspace.copy_done.query.assert_called_once_with()
        get_batch.assert_not_called()

    def test_cross_stream_staging_is_rejected_before_page_table_query(self):
        from types import SimpleNamespace
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
            _FixedScoreStreamMismatch,
        )

        workspace = _FixedScoreMetadataWorkspace.__new__(_FixedScoreMetadataWorkspace)
        workspace.device = torch.device("cuda")
        workspace.max_requests = 8
        workspace.stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=4)
        workspace.copy_pending = False
        workspace.copy_done = SimpleNamespace(query=mock.Mock())
        get_batch = mock.Mock(side_effect=AssertionError("cross-stream workspace queried V2"))

        other_stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=5)
        with mock.patch.object(torch.cuda, "current_stream", return_value=other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
                workspace.stage(get_batch, [1], [8.0])
        workspace.copy_done.query.assert_not_called()
        get_batch.assert_not_called()

    def test_attach_page_ids_does_not_swallow_cross_stream_rejection(self):
        from types import SimpleNamespace
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStreamMismatch,
        )

        manager = TriAttention.__new__(TriAttention)
        get_batch = mock.Mock()
        manager.kv_cache_manager = SimpleNamespace(get_batch_cache_indices=get_batch)
        manager._fixed_score_runtime_counts = {}
        manager._resolve_page_ids = mock.Mock(side_effect=AssertionError("eager fallback used"))
        workspace = SimpleNamespace(
            prewarm_key=("bucket",),
            stage=mock.Mock(
                side_effect=_FixedScoreStreamMismatch(
                    "TriAttention fixed score metadata is bound to its first CUDA stream"
                )
            ),
        )
        prepared = [
            {
                "request": SimpleNamespace(),
                "request_id": 7,
                "round_start": 8.0,
            }
        ]

        with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
            manager._attach_page_ids(
                prepared,
                dense_representatives=[0],
                swa_layers=[],
                layer_pools=[torch.empty(1)],
                global_layers=[0],
                workspace=workspace,
            )
        workspace.stage.assert_called_once_with(get_batch, [7], [8.0])
        manager._resolve_page_ids.assert_not_called()
        assert manager._fixed_score_runtime_counts[("bucket",)]["rejected"] == 1

    def test_first_runtime_stream_is_retained_and_records_copy_event(self):
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        workspace = _FixedScoreMetadataWorkspace.__new__(_FixedScoreMetadataWorkspace)
        workspace.device = torch.device("cuda")
        workspace.max_requests = 1
        workspace.stream = None
        workspace.copy_pending = False
        workspace.copy_done = mock.Mock()
        workspace.global_representatives = (0,)
        workspace.page_count = 1
        buffer = mock.MagicMock()
        buffer.__getitem__.return_value = buffer
        buffer.copy_.return_value = buffer
        buffer.view.return_value = buffer
        workspace.page_ids_host = buffer
        workspace.round_starts_host = buffer
        workspace.page_ids_device = buffer
        workspace.round_starts_device = buffer
        workspace.offsets = buffer
        workspace.omega = buffer
        workspace.phase_base = buffer
        workspace.phase = buffer
        workspace.cos_phase = buffer
        workspace.sin_phase = buffer
        workspace.mean_cos = buffer
        workspace.mean_sin = buffer
        stream = object()

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(torch, "as_tensor", return_value=buffer),
            mock.patch.object(torch, "add"),
            mock.patch.object(torch, "mul"),
            mock.patch.object(torch, "cos"),
            mock.patch.object(torch, "sin"),
            mock.patch.object(torch, "mean"),
        ):
            assert workspace.stage(lambda request_ids, layer: [[3]], [7], [8.0])
        assert workspace.stream is stream
        workspace.copy_done.record.assert_called_once_with(stream)

    def test_staged_page_tables_bypass_per_request_cuda_materialization(self):
        from types import SimpleNamespace
        from unittest import mock

        manager = TriAttention.__new__(TriAttention)
        get_batch = mock.Mock()
        manager.kv_cache_manager = SimpleNamespace(get_batch_cache_indices=get_batch)
        manager._fixed_score_runtime_counts = {}
        manager._resolve_page_ids = mock.Mock(side_effect=AssertionError("eager fallback used"))
        page_ids = torch.tensor([[[10, 11], [12, 13]], [[20, 21], [22, 23]]])
        workspace = SimpleNamespace(
            stage=mock.Mock(return_value=True),
            page_ids_device=page_ids,
            representative_slots={0: 0, 1: 1},
            prewarm_key=("bucket",),
        )
        prepared = [
            {"request": SimpleNamespace(), "request_id": 7, "round_start": 8.0},
            {"request": SimpleNamespace(), "request_id": 8, "round_start": 9.0},
        ]

        result = manager._attach_page_ids(
            prepared,
            dense_representatives=[0],
            swa_layers=[1],
            layer_pools=[torch.empty(1), torch.empty(1)],
            global_layers=[0, 1],
            workspace=workspace,
        )

        assert result
        workspace.stage.assert_called_once_with(get_batch, [7, 8], [8.0, 9.0])
        manager._resolve_page_ids.assert_not_called()
        assert manager._fixed_score_runtime_counts[("bucket",)]["hit"] == 1
        assert prepared[1]["page_ids"][0].tolist() == [12, 13]
        assert prepared[0]["page_ids"][1].tolist() == [20, 21]

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    @CUDA_REQUIRED
    def test_workspace_stages_dense_and_swa_tables_and_rejects_lifetime_changes(
        self, request_count
    ):
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
            _FixedScoreStreamMismatch,
        )

        device = torch.device("cuda")
        max_requests = 8
        page_count = 2
        seq_len = 7
        layer_elements = max_requests * page_count * 2 * 1 * 4 * 4
        shared = torch.randn(2 * layer_elements, device=device)
        pools = [
            shared[:layer_elements].view(max_requests * page_count, 2, 1, 4, 4),
            shared[layer_elements:].view(max_requests * page_count, 2, 1, 4, 4),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
        ]
        dense_groups = [[0, 1], [2]]
        representatives = [0, 2, 3]
        q_real = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        q_imag = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        mlr = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        freq = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64, device=device)[::2]
        omega = torch.tensor([0.01, 0.0, 0.03, 0.0], dtype=torch.float64, device=device)[::2]
        offsets = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float64, device=device)[::2]
        assert not q_real.is_contiguous()
        assert not freq.is_contiguous()
        assert not omega.is_contiguous()
        assert not offsets.is_contiguous()
        workspace = _FixedScoreMetadataWorkspace(
            pools,
            dense_groups,
            representatives,
            [10, 11, 12, 13],
            max_requests,
            seq_len,
            2,
            2,
            q_real,
            q_imag,
            mlr,
            freq,
            offsets,
            omega,
        )
        assert workspace.offsets.dtype == torch.float32
        assert workspace.offsets.is_contiguous()
        assert workspace.omega.dtype == torch.float32
        assert workspace.omega.is_contiguous()
        for group in workspace.groups.values():
            for calibration in (*group.pointer_middle[2:], *group.pointer_tail):
                assert calibration.dtype == torch.float32
                assert calibration.is_contiguous()
        tables = {
            10: [[2 * request, 2 * request + 1] for request in range(request_count)],
            12: [[2 * request + 1, 2 * request] for request in range(request_count)],
            13: [[15 - 2 * request, 14 - 2 * request] for request in range(request_count)],
        }
        get_batch = mock.Mock(side_effect=lambda request_ids, layer: tables[layer])
        request_ids = list(range(request_count))
        round_starts = [float(9 + request) for request in request_ids]

        assert workspace.stage(get_batch, request_ids, round_starts)
        torch.cuda.current_stream(device).synchronize()
        for slot, global_layer in enumerate((10, 12, 13)):
            expected = torch.tensor(tables[global_layer], dtype=torch.int64, device=device)
            assert torch.equal(workspace.page_ids_device[slot, :request_count], expected)
        assert workspace.matches(pools, dense_groups, representatives)
        changed_dense = list(pools)
        changed_dense[1] = pools[1].clone()
        assert not workspace.matches(changed_dense, dense_groups, representatives)
        changed_shape = list(pools)
        changed_shape[1] = pools[1].view(max_requests * page_count, 2, 1, 2, 8)
        assert not workspace.matches(changed_shape, dense_groups, representatives)
        changed_stride = list(pools)
        changed_stride[1] = pools[1].as_strided(pools[1].shape, (32, 16, 16, 1, 4))
        assert not workspace.matches(changed_stride, dense_groups, representatives)
        changed_dtype = list(pools)
        changed_dtype[1] = pools[1].view(torch.int32)
        assert not workspace.matches(changed_dtype, dense_groups, representatives)
        changed_swa = list(pools)
        changed_swa[3] = pools[3].clone()
        assert not workspace.matches(changed_swa, dense_groups, representatives)
        assert not workspace.matches(pools, [[0], [1], [2]], [0, 1, 2, 3])

        calls = get_batch.call_count
        other_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
                workspace.stage(get_batch, request_ids, round_starts)
        assert get_batch.call_count == calls

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    @CUDA_REQUIRED
    def test_fixed_score_matches_eager_and_torch_oracle_across_two_groups(
        self, request_count, aggregation
    ):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            _FixedScoreGroup,
            triton_tri_score_perhead,
        )

        device = torch.device("cuda")
        torch.manual_seed(20260703 + request_count)
        max_requests = 8
        page_count = 2
        seq_len = 7
        page_ids = torch.arange(max_requests * page_count, dtype=torch.int64, device=device).view(
            max_requests, page_count
        )
        layer_elements = max_requests * page_count * 2 * 1 * 4 * 4
        shared = torch.randn(2 * layer_elements, device=device)
        pools = [
            shared[:layer_elements].view(max_requests * page_count, 2, 1, 4, 4),
            shared[layer_elements:].view(max_requests * page_count, 2, 1, 4, 4),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
        ]
        storage_groups = [[0, 1], [2]]
        q_real = torch.randn(3, 2, 4, device=device)[..., ::2]
        q_imag = torch.randn(3, 2, 4, device=device)[..., ::2]
        mlr = torch.randn(3, 2, 4, device=device)[..., ::2]
        freq = torch.tensor([0.7, 0.0, 1.3, 0.0], device=device)[::2]
        omega = torch.tensor([0.013, 0.0, 0.071, 0.0], device=device)[::2]
        offsets = torch.tensor([1.0, 0.0, 2.0, 0.0, 4.0, 0.0], device=device)[::2]
        assert not q_real.is_contiguous()
        assert not q_imag.is_contiguous()
        assert not mlr.is_contiguous()
        assert not freq.is_contiguous()
        assert not omega.is_contiguous()
        assert not offsets.is_contiguous()
        round_device = torch.arange(max_requests, dtype=torch.float32, device=device) + 9.0
        round_starts = round_device[:request_count].tolist()
        seq_lens = [seq_len] * request_count
        phase = (round_device[:, None, None] + offsets[None, :, None]) * omega[None, None]
        oracle = _torch_tri_score_oracle(
            pools,
            page_ids[:request_count],
            seq_lens,
            round_starts,
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            [0, 1, 2],
            aggregation,
        )
        for layers in storage_groups:
            group = _FixedScoreGroup(
                pools,
                layers,
                max_requests,
                page_count,
                seq_len,
                2,
                page_ids,
                q_real,
                q_imag,
                mlr,
                freq,
                omega,
                offsets,
            )
            fixed, fixed_offsets = group.launch(
                request_count,
                round_device,
                torch.cos(phase).mean(dim=1),
                torch.sin(phase).mean(dim=1),
                aggregation,
            )
            eager, eager_offsets, _ = triton_tri_score_perhead(
                pools,
                list(page_ids[:request_count]),
                seq_lens,
                round_starts,
                q_real,
                q_imag,
                mlr,
                freq,
                omega,
                offsets,
                2,
                score_aggregation=aggregation,
                layer_indices=layers,
            )
            assert torch.equal(fixed, eager)
            assert torch.equal(fixed_offsets, eager_offsets)
            for request in range(request_count):
                for layer_slot, layer in enumerate(layers):
                    segment_index = request * len(layers) + layer_slot
                    segment = fixed[:, segment_index * seq_len : (segment_index + 1) * seq_len]
                    expected = oracle[request * len(pools) + layer]
                    torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)
                    selected = torch.topk(segment.max(dim=0).values, 3).indices.sort().values
                    expected_selected = (
                        torch.topk(expected.max(dim=0).values, 3).indices.sort().values
                    )
                    assert torch.equal(selected, expected_selected)

    @CUDA_REQUIRED
    def test_fixed_score_metadata_alternates_r7_r1_without_stale_rows(self):
        from types import SimpleNamespace

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
            _FixedUnionWorkspace,
        )
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            fixed_perhead_segment_views,
            triton_tri_score_perhead,
        )

        def tensor_pointers(owner):
            pointers = {}
            for name, value in vars(owner).items():
                values = value if isinstance(value, (tuple, list)) else (value,)
                for index, tensor in enumerate(values):
                    if isinstance(tensor, torch.Tensor):
                        pointers[(name, index)] = (
                            tensor.data_ptr(),
                            tensor.untyped_storage().data_ptr(),
                        )
            return pointers

        device = torch.device("cuda")
        torch.manual_seed(20260703)
        max_requests = 8
        page_count = 2
        seq_len = 7
        num_q_heads = 2
        total_pages = max_requests * page_count
        layer_elements = total_pages * 2 * 1 * 4 * 4
        shared = torch.randn(2 * layer_elements, device=device)
        pools = [
            shared[:layer_elements].view(total_pages, 2, 1, 4, 4),
            shared[layer_elements:].view(total_pages, 2, 1, 4, 4),
        ]
        q_real = torch.randn(2, num_q_heads, 2, device=device)
        q_imag = torch.randn_like(q_real)
        mlr = torch.randn_like(q_real)
        freq = torch.tensor([0.7, 1.3], device=device)
        omega = torch.tensor([0.013, 0.071], device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
        workspace = _FixedScoreMetadataWorkspace(
            pools,
            [[0, 1]],
            [0],
            [10, 11],
            max_requests,
            seq_len,
            num_q_heads,
            2,
            q_real,
            q_imag,
            mlr,
            freq,
            offsets,
            omega,
        )
        selection_key = ("fixed-score-to-selection",)
        workspace.prewarm_key = selection_key
        selection_base = _FixedUnionWorkspace(
            len(pools) * num_q_heads,
            seq_len - 1,
            3,
            1,
            dtype=torch.float32,
            device=device,
        )
        selection_bank = TriAttention._build_fixed_shape_selection_workspaces(
            selection_base,
            max_requests,
        )
        for item in selection_bank:
            item.prewarm_attempted = True
            item.prewarmed = True

        def build_selection_manager(fixed_shape):
            manager = TriAttention.__new__(TriAttention)
            manager.top_B = 3
            manager.pin_prefill = True
            manager.count_prompt_tokens = False
            manager.eviction_mode = "union"
            manager.normalize_scores = True
            manager.kv_cache_manager = SimpleNamespace(
                get_buffers=lambda layer, kv_layout: pools[layer]
            )
            manager._dense_layers = lambda num_layers: [0, 1]
            manager._global_layer_id = lambda layer, num_layers: layer
            manager._indexer_topk_supported = lambda width, k: False
            manager._fixed_union_enabled = False
            manager._fixed_union_active = {}
            manager._fixed_shape_selection_enabled = fixed_shape
            manager._fixed_shape_selection_workspaces = (
                {selection_key: selection_bank} if fixed_shape else {}
            )
            manager._fixed_shape_selection_prewarm_states = (
                {selection_key: "ready"} if fixed_shape else {}
            )
            manager._fixed_shape_selection_runtime_counts = {}
            return manager

        selection_manager = build_selection_manager(True)
        eager_selection_manager = build_selection_manager(False)
        group = workspace.groups[0]
        stable_workspace_pointers = tensor_pointers(workspace)
        stable_group_pointers = tensor_pointers(group)
        stable_selection_pointers = [
            tensor_pointers(item) for item in selection_bank
        ]
        previous_by_request_count = {}

        # Ninety-five full 7->1 cohort transitions reproduce the lifetime pattern
        # from the failing Stage2 run while changing both staged inputs every call.
        for iteration, request_count in enumerate((7, 1) * 95):
            torch.cuda.current_stream(device).synchronize()
            workspace.page_ids_host.fill_(-1)
            workspace.round_starts_host.fill_(float("nan"))
            shift = iteration % total_pages
            request_ids = [
                (iteration * 5 + slot) % max_requests
                for slot in range(request_count)
            ]
            tables = [
                [
                    (2 * request_id + shift + slot) % total_pages
                    for slot in range(page_count)
                ]
                for request_id in request_ids
            ]
            round_starts = [
                float(5000 + iteration * 128 + request_id)
                for request_id in request_ids
            ]

            previous = previous_by_request_count.get(request_count)
            if previous is not None:
                previous_tables, previous_round_starts = previous
                assert tables != previous_tables
                assert round_starts != previous_round_starts
            previous_by_request_count[request_count] = (tables, round_starts)

            def get_batch_cache_indices(observed_ids, global_layer):
                assert observed_ids == request_ids
                assert global_layer == 10
                return tables

            assert workspace.stage(get_batch_cache_indices, request_ids, round_starts)
            fixed, fixed_offsets = group.launch(
                request_count,
                workspace.round_starts_device,
                workspace.mean_cos,
                workspace.mean_sin,
                "mean",
            )
            eager_page_ids = [
                torch.tensor(row, dtype=torch.int64, device=device) for row in tables
            ]
            eager, eager_offsets, eager_meta = triton_tri_score_perhead(
                pools,
                eager_page_ids,
                [seq_len] * request_count,
                round_starts,
                q_real,
                q_imag,
                mlr,
                freq,
                omega,
                offsets,
                num_q_heads,
                score_aggregation="mean",
                layer_indices=[0, 1],
            )
            torch.cuda.current_stream(device).synchronize()

            num_segments = request_count * len(pools)
            assert fixed.shape == (num_q_heads, num_segments * seq_len)
            assert fixed_offsets.numel() == num_segments + 1
            assert len(eager_meta) == num_segments
            assert torch.equal(fixed, eager)
            assert torch.equal(fixed_offsets, eager_offsets)
            assert torch.isfinite(fixed).all()
            fixed_views = fixed_perhead_segment_views(
                fixed,
                request_count,
                len(pools),
                seq_len,
            )
            for item in selection_bank:
                item.input_scores.fill_(float("nan"))
                item.keep[1:].fill_(-1)
            active_selection = selection_manager._fixed_shape_selection_for(
                workspace,
                request_count,
            )
            assert active_selection == selection_bank[:request_count]
            for request in range(request_count):
                fixed_rows = []
                for layer in range(len(pools)):
                    segment = request * len(pools) + layer
                    meta = eager_meta[segment]
                    assert meta.seg_index == segment
                    assert meta.request_index == request
                    assert meta.layer_index == layer
                    assert meta.seq_len == seq_len
                    assert meta.round_start == round_starts[request]
                    begin = segment * seq_len
                    end = begin + seq_len
                    fixed_segment = fixed[:, begin:end]
                    eager_segment = eager[:, begin:end]
                    assert torch.equal(fixed_segment, eager_segment)
                    fixed_rows.append(fixed_segment)
                request_state = SimpleNamespace(
                    py_request_id=request_ids[request],
                    py_prompt_len=1,
                )
                precomputed = [
                    fixed_views[:, request, layer] for layer in range(len(pools))
                ]
                fixed_keep = selection_manager._evict_modes(
                    request_state,
                    len(pools),
                    seq_len,
                    precomputed,
                    fixed_union_workspace=active_selection[request],
                ).clone()
                eager_keep = eager_selection_manager._evict_modes(
                    request_state,
                    len(pools),
                    seq_len,
                    precomputed,
                ).clone()
                oracle_keep = _torch_union_keep(
                    torch.cat(fixed_rows, dim=0), prompt_len=1, budget=3
                )
                assert torch.equal(fixed_keep, eager_keep)
                assert torch.equal(fixed_keep, oracle_keep)

            for item in selection_bank[request_count:]:
                assert torch.isnan(item.input_scores).all()
                assert torch.equal(item.keep[1:], torch.full_like(item.keep[1:], -1))

            expected_tables = torch.tensor(tables, dtype=torch.int64, device=device)
            assert torch.equal(workspace.page_ids_device[0, :request_count], expected_tables)
            assert torch.equal(
                workspace.round_starts_device[:request_count],
                torch.tensor(round_starts, dtype=torch.float32, device=device),
            )
            if request_count < max_requests:
                assert torch.equal(
                    workspace.page_ids_device[0, request_count:],
                    torch.full_like(workspace.page_ids_device[0, request_count:], -1),
                )
                assert torch.isnan(workspace.round_starts_device[request_count:]).all()
            assert tensor_pointers(workspace) == stable_workspace_pointers
            assert tensor_pointers(group) == stable_group_pointers
            assert [tensor_pointers(item) for item in selection_bank] == (
                stable_selection_pointers
            )

        assert selection_manager._fixed_shape_selection_runtime_counts[
            selection_key
        ] == {"hit": 190, "fallback": 0}

    @CUDA_REQUIRED
    def test_gptoss_dense_swa_rebase_and_capacity_match_staging_fallback(self):
        from types import SimpleNamespace

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        device = torch.device("cuda")
        torch.manual_seed(731)
        seq_len = 8
        keep_count = 6
        page_lists = {0: [1, 2, 0], 1: [0, 1, 2], 2: [2, 0, 1]}
        page_ids = {
            layer: torch.tensor(pages, dtype=torch.int64, device=device)
            for layer, pages in page_lists.items()
        }
        base_pools = [
            torch.arange(3 * 2 * 1 * 4 * 4, dtype=torch.float32, device=device).view(3, 2, 1, 4, 4),
            torch.arange(3 * 2 * 1 * 4 * 4, dtype=torch.float32, device=device).view(3, 2, 1, 4, 4)
            + 500.0,
            torch.arange(3 * 2 * 1 * 4 * 4, dtype=torch.float32, device=device).view(3, 2, 1, 4, 4)
            + 1000.0,
        ]
        q_real = torch.ones(3, 2, 2, device=device)
        q_imag = torch.full_like(q_real, 0.25)
        mlr = torch.full_like(q_real, 0.1)
        freq = torch.tensor([0.8, 1.2], device=device)
        omega = torch.tensor([0.017, 0.043], device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)

        def build(force_fallback):
            pools = [pool.clone() for pool in base_pools]
            cache = _make_fake_v2()
            cache.max_batch_size = 8
            cache.pp_layers = [0, 1, 2]
            cache.num_layers = 3
            cache.layer_offsets = {0: 0, 1: 1, 2: 2}
            cache.batch_calls = []
            cache.get_buffers = lambda layer, **kwargs: pools[layer]

            def get_batch(request_ids, layer):
                cache.batch_calls.append((tuple(request_ids), layer))
                return [list(page_lists[layer]) for _ in request_ids]

            cache.get_batch_cache_indices = get_batch
            manager = TriAttention(
                cache,
                top_B=4,
                beta=4,
                score_aggregation="mean",
                eviction_mode="union",
                skip_swa=False,
            )
            manager._calibrated = True
            manager._L = 3
            manager._H = manager._F = 2
            manager._triattn_q_real = q_real
            manager._triattn_q_imag = q_imag
            manager._triattn_mlr_coef = mlr
            manager._freq_scale_sq = freq
            manager.calibration = {"omega": omega}
            manager._offsets = offsets
            manager._attention_layer_partition = lambda num_layers: ([1, 2], [0], 2)
            manager._local_to_global_layers_cache = [0, 1, 2]
            manager._indexer_topk_supported = lambda width, k: False
            manager._fixed_union_enabled = False
            manager._fixed_union_prewarm_enabled = False
            manager._fixed_union_compaction_enabled = False
            manager._fixed_score_metadata_enabled = True
            manager._fixed_union_prewarm_key = lambda *args: ("gptoss",)
            workspace = _FixedScoreMetadataWorkspace(
                pools,
                [[1], [2]],
                [1, 2, 0],
                [0, 1, 2],
                8,
                seq_len,
                2,
                2,
                q_real,
                q_imag,
                mlr,
                freq,
                offsets,
                omega,
            )
            if force_fallback:
                workspace.stage = lambda *args: False
            workspace.prewarm_key = ("gptoss",)
            manager._fixed_score_workspaces = {("gptoss",): workspace}
            manager._fixed_score_prewarm_states = {("gptoss",): "ready"}
            manager._fixed_score_runtime_counts = {}
            manager._pre_forward_kv_lengths = {7: seq_len}
            manager._evicted = {}
            original_evict_modes = manager._evict_modes

            def capture_keep(*args, **kwargs):
                keep = original_evict_modes(*args, **kwargs)
                manager.test_keep = keep.clone()
                return keep

            manager._evict_modes = capture_keep
            request = SimpleNamespace(
                py_request_id=7,
                py_prompt_len=2,
                py_draft_tokens=[],
                max_beam_num_tokens=seq_len + 1,
            )
            return manager, request, pools, cache

        fixed_manager, fixed_request, fixed_pools, fixed_cache = build(False)
        eager_manager, eager_request, eager_pools, _ = build(True)
        fixed_targets = fixed_manager._evict_requests([(fixed_request, 7)], num_layers=3)
        eager_targets = eager_manager._evict_requests([(eager_request, 7)], num_layers=3)
        torch.cuda.current_stream(device).synchronize()

        assert fixed_targets == eager_targets == [(7, keep_count + 1)]
        assert torch.equal(fixed_manager.test_keep, eager_manager.test_keep)
        oracle_scores = _torch_tri_score_oracle(
            base_pools,
            {layer: ids.unsqueeze(0) for layer, ids in page_ids.items()},
            [seq_len],
            [float(seq_len)],
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            [1, 2],
            "mean",
        )
        assert torch.equal(
            fixed_manager.test_keep,
            _torch_union_keep(torch.cat(oracle_scores), prompt_len=2, budget=4),
        )
        assert fixed_manager._fixed_score_runtime_counts[("gptoss",)]["hit"] == 1
        assert eager_manager._fixed_score_runtime_counts[("gptoss",)]["fallback"] == 1
        assert fixed_manager._evicted == eager_manager._evicted == {7: 2}
        assert fixed_manager._pre_forward_kv_lengths == {7: keep_count}
        for layer in (1, 2):
            original_dense = _logical_kv(base_pools[layer], page_ids[layer], seq_len)
            expected_dense = original_dense.index_select(2, fixed_manager.test_keep)
            fixed_dense = _logical_kv(fixed_pools[layer], page_ids[layer], keep_count)
            eager_dense = _logical_kv(eager_pools[layer], page_ids[layer], keep_count)
            assert torch.equal(fixed_dense, expected_dense)
            assert torch.equal(fixed_dense, eager_dense)
        original_swa = _logical_kv(base_pools[0], page_ids[0], seq_len)
        swa_keep = _build_swa_rebase_keep(seq_len, keep_count, 2, device=device)
        expected_swa = original_swa.index_select(2, swa_keep)
        fixed_swa = _logical_kv(fixed_pools[0], page_ids[0], keep_count)
        eager_swa = _logical_kv(eager_pools[0], page_ids[0], keep_count)
        assert torch.equal(fixed_swa, expected_swa)
        assert torch.equal(fixed_swa, eager_swa)
        assert fixed_cache.batch_calls[:3] == [((7,), 1), ((7,), 2), ((7,), 0)]


class TestFixedScoreSegmentViews:
    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_exact_geometry_returns_aliasing_request_layer_views(self, device):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            fixed_perhead_segment_views,
        )

        heads, requests, layers, seq_len = 2, 3, 4, 5
        scores = torch.arange(
            heads * requests * layers * seq_len,
            dtype=torch.float32,
            device=device,
        ).view(heads, -1)

        views = fixed_perhead_segment_views(scores, requests, layers, seq_len)

        assert views.shape == (heads, requests, layers, seq_len)
        assert views.data_ptr() == scores.data_ptr()
        for request in range(requests):
            for layer in range(layers):
                segment = request * layers + layer
                expected = scores[:, segment * seq_len : (segment + 1) * seq_len]
                assert torch.equal(views[:, request, layer], expected)

    def test_geometry_mismatch_fails_without_reading_device_offsets(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            fixed_perhead_segment_views,
        )

        scores = torch.zeros(2, 23)

        with pytest.raises(ValueError, match="output width"):
            fixed_perhead_segment_views(scores, request_count=2, layer_count=3, seq_len=4)
        with pytest.raises(ValueError, match="positive"):
            fixed_perhead_segment_views(scores, request_count=0, layer_count=3, seq_len=4)


class TestFixedUnionWorkspace:
    @staticmethod
    def _reference(scores, keep_count):
        combined = scores.max(dim=0).values
        row_top = torch.topk(scores, keep_count, dim=1, sorted=False).indices
        union_mask = torch.zeros(combined.numel(), dtype=torch.bool, device=scores.device)
        union_mask.scatter_(0, row_top.reshape(-1), True)
        union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
        subset = combined.index_select(0, union_indices)
        selected = torch.topk(subset, keep_count, sorted=False).indices
        return union_indices.index_select(0, torch.sort(selected).values)

    def test_gptoss_bucket_canonicalizes_unindexed_cuda_device(self):
        from types import SimpleNamespace
        from unittest import mock

        meta_scores = torch.empty((12 * 64, 4225), dtype=torch.float32, device="meta")
        meta_workspace = _FixedUnionWorkspace(
            12 * 64,
            4225,
            4096,
            1024,
            dtype=meta_scores.dtype,
            device=meta_scores.device,
        )
        assert meta_workspace.select(meta_scores).shape == (1024 + 4096,)

        with mock.patch.object(torch.cuda, "current_device", return_value=0) as current_device:
            device = _FixedUnionWorkspace._canonical_device(torch.device("cuda"))
        current_device.assert_called_once_with()

        workspace = _FixedUnionWorkspace.__new__(_FixedUnionWorkspace)
        workspace.rows = 12 * 64
        workspace.width = 4225
        workspace.dtype = torch.float32
        workspace.device = device
        scores = SimpleNamespace(
            shape=(12 * 64, 4225),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
        )

        assert workspace._matches_input(scores)

    def test_prewarm_has_distinct_nested_environment_gate(self):
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "0",
                "TRIATTN_FIXED_PREWARM": "1",
            },
        ):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert not manager._fixed_union_prewarm_enabled

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "1",
                "TRIATTN_FIXED_PREWARM": "0",
            },
        ):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert manager._fixed_union_enabled
            assert not manager._fixed_union_prewarm_enabled

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "1",
                "TRIATTN_FIXED_PREWARM": "1",
            },
        ):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert manager._fixed_union_prewarm_enabled

    def test_fixed_shape_selection_requires_every_cumulative_gate(self):
        import os
        from unittest import mock

        env = {
            "TRIATTN_FIXED_BUFFER_UNION": "1",
            "TRIATTN_FIXED_PREWARM": "1",
            "TRIATTN_FIXED_SCORE_METADATA": "1",
            "TRIATTN_FIXED_SHAPE_SELECTION": "0",
        }
        with mock.patch.dict(os.environ, env):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert not manager._fixed_shape_selection_enabled

        env["TRIATTN_FIXED_SHAPE_SELECTION"] = "1"
        with mock.patch.dict(os.environ, env):
            manager = TriAttention(_make_fake_v2(), top_B=4096, skip_swa=False)
            assert manager._fixed_shape_selection_enabled

    def test_prewarm_shape_parser_is_exact_and_fail_closed(self):
        assert TriAttention._parse_fixed_prewarm_shapes("1024:8192, 0:4097,1024:8192") == [
            (1024, 8192),
            (0, 4097),
        ]
        assert TriAttention._parse_fixed_prewarm_shapes("") == []
        with pytest.raises(ValueError, match="prompt_len:decode_width"):
            TriAttention._parse_fixed_prewarm_shapes("1024")
        with pytest.raises(ValueError, match="contain integers"):
            TriAttention._parse_fixed_prewarm_shapes("prompt:4225")
        with pytest.raises(ValueError, match="prompt_len >= 0"):
            TriAttention._parse_fixed_prewarm_shapes("-1:4225")

    def test_startup_prewarm_dispatches_first_and_steady_buckets(self):
        import os
        from unittest import mock

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        manager._INDEXER_TOPK_MAX_K = 2048
        manager._INDEXER_TOPK_SUBBLOCK = 2048
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        with (
            mock.patch.dict(
                os.environ,
                {"TRIATTN_FIXED_PREWARM_SHAPES": "1024:4095,1024:4096"},
            ),
            mock.patch.object(manager, "_validate_v2_compatibility"),
            mock.patch.object(manager, "_ensure_calibrated"),
            mock.patch.object(manager, "_num_layers_from_manager", return_value=2),
            mock.patch.object(
                manager,
                "_fixed_union_live_geometry",
                return_value=(layer_pools, dense_layers, storage_groups),
            ),
            mock.patch.object(manager, "_prewarm_fixed_union_bucket") as prewarm_bucket,
        ):
            manager.prewarm()

        assert [call.args[-2:] for call in prewarm_bucket.call_args_list] == [
            (1024, 4095),
            (1024, 4096),
        ]

    def test_dummy_pool_preserves_geometry_without_aliasing_live_storage(self):
        live = torch.arange(3 * 2 * 2 * 4 * 3, dtype=torch.float32).reshape(3, 2, 2, 4, 3)
        before = live.clone()

        dummy = TriAttention._dummy_pool_like(live, num_pages=2, zero=True)

        assert dummy.shape == (2, 2, 2, 4, 3)
        assert dummy.stride() == live.stride()
        assert dummy.dtype == live.dtype
        assert dummy.device == live.device
        assert dummy.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
        assert torch.equal(live, before)
        assert torch.count_nonzero(dummy) == 0

    def test_prewarm_key_uses_geometry_not_storage_pointer(self):
        manager, pools = self._make_mocked_prewarm_manager()
        clone_storage = torch.empty(64, dtype=pools[0].dtype)
        clone_pools = [
            clone_storage.as_strided(pools[0].shape, pools[0].stride(), storage_offset=0),
            clone_storage.as_strided(pools[1].shape, pools[1].stride(), storage_offset=32),
        ]

        first = manager._fixed_union_prewarm_key(
            pools,
            [0, 1],
            [[0, 1]],
            num_layers=2,
            future_seq_len=7,
            prompt_len=2,
        )
        second = manager._fixed_union_prewarm_key(
            clone_pools,
            [0, 1],
            [[0, 1]],
            num_layers=2,
            future_seq_len=7,
            prompt_len=2,
        )

        assert first == second

    def test_prewarm_key_records_exact_4095_indexer_boundary(self):
        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        manager._INDEXER_TOPK_MAX_K = 2048
        manager._INDEXER_TOPK_SUBBLOCK = 2048

        native = manager._fixed_union_prewarm_key(
            pools,
            [0, 1],
            [[0, 1]],
            num_layers=2,
            future_seq_len=1024 + 4095,
            prompt_len=1024,
        )
        eager = manager._fixed_union_prewarm_key(
            pools,
            [0, 1],
            [[0, 1]],
            num_layers=2,
            future_seq_len=1024 + 4096,
            prompt_len=1024,
        )
        manager.top_B = 4096
        fixed = manager._fixed_union_prewarm_key(
            pools,
            [0, 1],
            [[0, 1]],
            num_layers=2,
            future_seq_len=1024 + 8191,
            prompt_len=1024,
        )

        assert native[0] == "triattention.fixed-prewarm.v2"
        assert native[9] == "eager_union.indexer_topk"
        assert eager[9] == "eager_union.torch_topk"
        assert fixed[9] == "fixed_union.torch_topk"
        assert native[12:17] == (1024 + 4095, 1024, 2048, 4, 4095)
        assert native != eager

    @staticmethod
    def _make_mocked_prewarm_manager():
        from types import SimpleNamespace

        shape = (2, 2, 1, 4, 2)
        stride = (16, 8, 8, 2, 1)
        storage = torch.arange(64, dtype=torch.float32)
        pools = [
            storage.as_strided(shape, stride, storage_offset=0),
            storage.as_strided(shape, stride, storage_offset=32),
        ]
        manager = TriAttention.__new__(TriAttention)
        manager.kv_cache_manager = SimpleNamespace(get_buffers=lambda layer, **kwargs: pools[layer])
        manager.top_B = 4
        manager._INDEXER_TOPK_MAX_K = 2
        manager._H = 2
        manager._F = 1
        manager._offset_max_length = 1
        manager._offsets = torch.ones(1)
        manager._triattn_q_real = torch.ones(2, 2, 1)
        manager._triattn_q_imag = torch.ones(2, 2, 1)
        manager._triattn_mlr_coef = torch.ones(2, 2, 1)
        manager._freq_scale_sq = torch.ones(1)
        manager.calibration = {"omega": torch.ones(1)}
        manager.score_aggregation = "mean"
        manager.normalize_scores = True
        manager.eviction_mode = "union"
        manager.skip_swa = False
        manager._compact_backend = "torch"
        manager._fixed_union_compaction_enabled = True
        manager._fixed_union_prewarm_enabled = True
        manager._fixed_union_prewarm_states = {}
        manager._fixed_union_prewarmed_workspaces = {}
        manager._fixed_union_enabled = True
        manager._fixed_union_workspaces = {}
        manager._fixed_union_active = {}
        manager._gen_steps = {}
        manager._evicted = {}
        manager._pre_forward_kv_lengths = {}
        manager._capacity_only_request_ids = set()
        manager._local_to_global_layers_cache = [0, 1]
        manager._attention_layer_partition_cache = ([0, 1], [], None)
        return manager, pools

    def test_startup_prewarm_uses_dummy_score_select_and_compact_once(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        live_before = [pool.clone() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)
        seen_dummy_pools = []
        seen_page_ids = []
        request_state_before = (
            manager._gen_steps.copy(),
            manager._evicted.copy(),
            manager._pre_forward_kv_lengths.copy(),
            manager._capacity_only_request_ids.copy(),
        )

        def fake_score(dummy_pools, page_ids_list, seq_lens, *args, **kwargs):
            assert seq_lens == [7]
            assert kwargs["layer_indices"] == [0, 1]
            dummy_pool = dummy_pools[0]
            assert dummy_pools[1] is dummy_pool
            assert dummy_pool.shape == (2, 2, 1, 4, 2)
            assert dummy_pool.stride() == pools[0].stride()
            assert dummy_pool.untyped_storage().data_ptr() != pools[0].untyped_storage().data_ptr()
            seen_dummy_pools.append(dummy_pool)
            seen_page_ids.append(page_ids_list[0].clone())
            values = torch.arange(2 * 2 * 7, dtype=torch.float32).reshape(2, 2 * 7)
            return values, torch.tensor([0, 7, 14], dtype=torch.int32), []

        def fake_compact(workspace, dummy_pool, page_ids, seq_len, **kwargs):
            assert seq_len == 7
            assert dummy_pool is seen_dummy_pools[0]
            assert dummy_pool.untyped_storage().data_ptr() != pools[0].untyped_storage().data_ptr()
            assert page_ids.tolist() == [0, 1]

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score) as score,
            mock.patch.object(
                _FixedUnionWorkspace, "compact", autospec=True, side_effect=fake_compact
            ) as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                return_value=contextlib.nullcontext(),
            ) as nvtx,
            mock.patch.object(torch, "topk", wraps=torch.topk) as topk,
            mock.patch.object(torch.cuda, "Event") as event,
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=2,
                decode_width=5,
            )
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=2,
                decode_width=5,
            )

        score.assert_called_once()
        compact.assert_called_once()
        assert [call.args[0] for call in nvtx.call_args_list] == [
            "triattention.prewarm",
            "triattention.prewarm.score",
            "triattention.prewarm.select",
            "triattention.prewarm.compact",
        ]
        assert topk.call_count == 2
        event.assert_not_called()
        synchronize.assert_not_called()
        assert seen_page_ids[0].tolist() == [0, 1]
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        assert len(manager._fixed_union_prewarmed_workspaces) == 1
        prewarm_key = next(iter(manager._fixed_union_prewarm_states))
        runtime_scores = torch.zeros((4, 5), dtype=torch.float32)
        workspace = manager._get_fixed_union_workspace(
            99,
            runtime_scores,
            4,
            2,
            prewarm_key=prewarm_key,
        )
        assert workspace is next(iter(manager._fixed_union_prewarmed_workspaces.values()))
        assert workspace.prewarmed
        mismatched_key = (*prewarm_key[:-1], ("different-pool-geometry",))
        fallback = manager._get_fixed_union_workspace(
            100,
            runtime_scores,
            4,
            2,
            prewarm_key=mismatched_key,
        )
        assert fallback is not workspace
        assert not fallback.prewarmed
        assert request_state_before == (
            manager._gen_steps,
            manager._evicted,
            manager._pre_forward_kv_lengths,
            manager._capacity_only_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_startup_prewarm_executes_exact_4095_native_indexer_bucket(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        manager._INDEXER_TOPK_MAX_K = 2048
        manager._INDEXER_TOPK_SUBBLOCK = 2048
        live_before = [pool.clone() for pool in pools]
        live_ptrs = [pool.untyped_storage().data_ptr() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)
        seen_indexer_calls = []

        def fake_score(*args, **kwargs):
            seq_len = 1024 + 4095
            values = torch.zeros((2, 2 * seq_len), dtype=torch.float32)
            return values, torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32), []

        def fake_indexer(scores, k):
            rows, width = scores.shape
            seen_indexer_calls.append((rows, width, k))
            assert width == 4095
            assert k == 2048
            if rows == 1:
                return torch.arange(width - k, width, dtype=torch.long, device=scores.device)
            low = torch.arange(k, dtype=torch.long, device=scores.device)
            high = torch.arange(width - k, width, dtype=torch.long, device=scores.device)
            return torch.stack([high if row % 2 == 0 else low for row in range(rows)])

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(kernels, "triton_tri_compact") as compact,
            mock.patch.object(manager, "_indexer_topk_idx", side_effect=fake_indexer),
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
            mock.patch.object(
                torch,
                "topk",
                side_effect=AssertionError("native prewarm must not call torch.topk"),
            ),
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=1024,
                decode_width=4095,
            )

        assert seen_indexer_calls == [(4, 4095, 2048), (1, 4095, 2048)]
        compact.assert_called_once()
        assert compact.call_args.args[3] == [1024 + 4095]
        keep = compact.call_args.args[2][0]
        assert keep.shape == (1024 + 2048,)
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        assert not manager._fixed_union_prewarmed_workspaces
        assert live_ptrs == [pool.untyped_storage().data_ptr() for pool in pools]
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_startup_prewarm_executes_exact_4096_torch_topk_bucket(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        manager._INDEXER_TOPK_MAX_K = 2048
        manager._INDEXER_TOPK_SUBBLOCK = 2048
        live_before = [pool.clone() for pool in pools]
        live_ptrs = [pool.untyped_storage().data_ptr() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        def fake_score(*args, **kwargs):
            seq_len = 1024 + 4096
            values = torch.zeros((2, 2 * seq_len), dtype=torch.float32)
            return values, torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32), []

        real_topk = torch.topk
        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(kernels, "triton_tri_compact") as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
            mock.patch.object(torch, "topk", wraps=real_topk) as topk,
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=1024,
                decode_width=4096,
            )

        assert [tuple(call.args[0].shape) for call in topk.call_args_list] == [
            (4, 4096),
            (1, 4096),
        ]
        assert all(call.args[1] == 2048 for call in topk.call_args_list)
        assert all(call.kwargs == {"dim": 1, "sorted": False} for call in topk.call_args_list)
        compact.assert_called_once()
        assert compact.call_args.args[3] == [1024 + 4096]
        keep = compact.call_args.args[2][0]
        assert keep.shape == (1024 + 2048,)
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        key = next(iter(manager._fixed_union_prewarm_states))
        assert key[9] == "eager_union.torch_topk"
        assert not manager._fixed_union_prewarmed_workspaces
        assert live_ptrs == [pool.untyped_storage().data_ptr() for pool in pools]
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_prewarm_failure_marks_bucket_and_runtime_falls_back(self):
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        live_before = [pool.clone() for pool in pools]
        request_state_before = (
            manager._gen_steps.copy(),
            manager._evicted.copy(),
            manager._pre_forward_kv_lengths.copy(),
            manager._capacity_only_request_ids.copy(),
        )
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)
        with mock.patch.object(
            kernels,
            "triton_tri_score_perhead",
            side_effect=RuntimeError("compile failed"),
        ):
            with pytest.raises(RuntimeError, match="compile failed"):
                manager._prewarm_fixed_union_bucket(
                    layer_pools,
                    dense_layers,
                    storage_groups,
                    num_layers=2,
                    prompt_len=2,
                    decode_width=5,
                )

        assert list(manager._fixed_union_prewarm_states.values()) == ["failed"]
        assert not manager._fixed_union_prewarmed_workspaces
        fallback = manager._get_fixed_union_workspace(
            99,
            torch.zeros((4, 5), dtype=torch.float32),
            4,
            2,
        )
        assert fallback is not None
        assert not fallback.prewarmed
        assert request_state_before == (
            manager._gen_steps,
            manager._evicted,
            manager._pre_forward_kv_lengths,
            manager._capacity_only_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_width_fallback_prewarm_keeps_runtime_on_eager_route(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2
        manager._INDEXER_TOPK_MAX_K = 2
        manager._INDEXER_TOPK_SUBBLOCK = 2
        live_before = [pool.clone() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        def fake_score(*args, **kwargs):
            values = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2 * 6)
            return values, torch.tensor([0, 6, 12], dtype=torch.int32), []

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(kernels, "triton_tri_compact") as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ) as nvtx,
            mock.patch.object(torch, "topk", wraps=torch.topk) as topk,
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=2,
                decode_width=4,
            )

        assert [call.args[0] for call in nvtx.call_args_list] == [
            "triattention.prewarm",
            "triattention.prewarm.score",
            "triattention.prewarm.select",
            "triattention.prewarm.compact",
        ]
        assert topk.call_count == 2
        compact.assert_called_once()
        assert compact.call_args.args[2][0].tolist() == [0, 1, 4, 5]
        assert compact.call_args.args[3] == [6]
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        assert not manager._fixed_union_prewarmed_workspaces
        assert (
            manager._get_fixed_union_workspace(
                99,
                torch.zeros((4, 4), dtype=torch.float32),
                2,
                2,
            )
            is None
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_workspace_allocation_failure_marks_bucket_failed(self):
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager, pools = self._make_mocked_prewarm_manager()
        live_before = [pool.clone() for pool in pools]
        request_state_before = (
            manager._gen_steps.copy(),
            manager._evicted.copy(),
            manager._pre_forward_kv_lengths.copy(),
            manager._capacity_only_request_ids.copy(),
        )
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        with mock.patch.object(
            tri_module,
            "_FixedUnionWorkspace",
            side_effect=MemoryError("workspace OOM"),
        ):
            with pytest.raises(MemoryError, match="workspace OOM"):
                manager._prewarm_fixed_union_bucket(
                    layer_pools,
                    dense_layers,
                    storage_groups,
                    num_layers=2,
                    prompt_len=2,
                    decode_width=5,
                )

        assert list(manager._fixed_union_prewarm_states.values()) == ["failed"]
        assert not manager._fixed_union_prewarmed_workspaces
        assert request_state_before == (
            manager._gen_steps,
            manager._evicted,
            manager._pre_forward_kv_lengths,
            manager._capacity_only_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_missing_or_mismatched_prewarm_bucket_preserves_fixed_fallback(self):
        manager, _ = self._make_mocked_prewarm_manager()

        missing = manager._get_fixed_union_workspace(
            11,
            torch.zeros((4, 5), dtype=torch.float32),
            4,
            2,
        )
        mismatched = manager._get_fixed_union_workspace(
            12,
            torch.zeros((4, 6), dtype=torch.float32),
            4,
            2,
        )

        assert missing is not None and not missing.prewarmed
        assert mismatched is not None and not mismatched.prewarmed
        assert missing is not mismatched

    def test_finite_distinct_matches_eager_and_reuses_buffers(self):
        rows = 3
        width = 257
        keep_count = 32
        prompt_len = 11
        token = torch.arange(width, dtype=torch.float32)
        scores = torch.stack(
            [torch.sin(token * scale) + token * 1e-4 for scale in (0.07, 0.11, 0.17)]
        )
        workspace = _FixedUnionWorkspace(
            rows,
            width,
            keep_count,
            prompt_len,
            dtype=scores.dtype,
            device=scores.device,
        )
        pointers = {
            name: value.data_ptr()
            for name, value in vars(workspace).items()
            if isinstance(value, torch.Tensor)
        }

        first = workspace.select(scores).clone()
        second = workspace.select(scores).clone()
        reference = self._reference(scores, keep_count)

        assert torch.equal(first, second)
        assert torch.equal(first[:prompt_len], torch.arange(prompt_len))
        assert torch.equal(first[prompt_len:] - prompt_len, reference)
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(workspace).items()
            if isinstance(value, torch.Tensor)
        }

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    @pytest.mark.parametrize("normalize_scores", [False, True])
    def test_fixed_segments_match_normalized_reference_without_dynamic_nonzero(
        self, device, normalize_scores
    ):
        from unittest import mock

        rows = 4
        width = 67
        keep_count = 16
        prompt_len = 5
        token = torch.arange(width, dtype=torch.float32, device=device)
        segments = [
            torch.stack(
                [
                    torch.sin(token * 0.071) + token * 1e-3,
                    torch.cos(token * 0.113) + token * 2e-3,
                ]
            ),
            torch.stack(
                [
                    torch.sin(token * 0.173) + token * 3e-3,
                    torch.cos(token * 0.197) + token * 4e-3,
                ]
            ),
        ]
        raw = torch.cat(segments, dim=0)
        if normalize_scores:
            # Match the Stage2 route exactly: normalize each layer segment
            # independently, then concatenate its rows for union selection.
            normalized = torch.cat(
                [
                    (segment - segment.mean(dim=1, keepdim=True))
                    / segment.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                    for segment in segments
                ],
                dim=0,
            )
        else:
            normalized = raw
        reference = self._reference(normalized, keep_count)
        workspace = _FixedUnionWorkspace(
            rows,
            width,
            keep_count,
            prompt_len,
            dtype=raw.dtype,
            device=raw.device,
            allocate_segment_buffers=True,
        )
        pointers = {
            name: value.data_ptr()
            for name, value in vars(workspace).items()
            if isinstance(value, torch.Tensor)
        }

        with (
            mock.patch.object(
                torch,
                "nonzero",
                side_effect=AssertionError("fixed selection must not call nonzero"),
            ),
            mock.patch.object(torch, "cat", wraps=torch.cat) as cat,
        ):
            first = workspace.select_segments(
                segments, normalize_scores=normalize_scores
            ).clone()
            second = workspace.select_segments(
                segments, normalize_scores=normalize_scores
            ).clone()

        assert torch.equal(first, second)
        assert torch.equal(first[:prompt_len], torch.arange(prompt_len, device=device))
        assert torch.equal(first[prompt_len:] - prompt_len, reference)
        assert torch.equal(workspace.input_scores, normalized)
        assert cat.call_count == 2
        assert all(call.kwargs["out"] is workspace.input_scores for call in cat.call_args_list)
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(workspace).items()
            if isinstance(value, torch.Tensor)
        }

    def test_fixed_shape_workspace_bank_has_independent_stable_slots(self):
        from types import SimpleNamespace

        base = _FixedUnionWorkspace(
            3,
            33,
            16,
            4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert not base._segment_buffers_prepared
        assert not hasattr(base, "input_scores")
        workspaces = TriAttention._build_fixed_shape_selection_workspaces(base, 3)
        assert all(item._segment_buffers_prepared for item in workspaces)
        manager = TriAttention.__new__(TriAttention)
        manager._fixed_shape_selection_enabled = True
        manager._fixed_shape_selection_workspaces = {("bucket",): workspaces}
        manager._fixed_shape_selection_prewarm_states = {("bucket",): "ready"}
        manager._fixed_shape_selection_runtime_counts = {}
        score_workspace = SimpleNamespace(prewarm_key=("bucket",))

        assert not any(item.prewarmed for item in workspaces)
        assert manager._fixed_shape_selection_for(score_workspace, 2) is None
        for item in workspaces:
            item.prewarm_attempted = True
            item.prewarmed = True

        selected = manager._fixed_shape_selection_for(score_workspace, 2)

        assert selected == workspaces[:2]
        assert len({item.keep.data_ptr() for item in workspaces}) == 3
        assert all(item.prewarmed for item in workspaces)
        assert manager._fixed_shape_selection_runtime_counts == {
            ("bucket",): {"hit": 1, "fallback": 1}
        }
        assert manager._fixed_shape_selection_for(score_workspace, 4) is None
        assert manager._fixed_shape_selection_runtime_counts == {
            ("bucket",): {"hit": 1, "fallback": 2}
        }

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_fixed_shape_prewarm_executes_new_route_and_isolates_failure(
        self, device
    ):
        from types import SimpleNamespace
        from unittest import mock

        device = torch.device(device)
        key = ("bucket",)
        base = _FixedUnionWorkspace(
            4,
            17,
            8,
            2,
            dtype=torch.float32,
            device=device,
        )
        scores_by_layer = {
            0: torch.arange(2 * 19, dtype=torch.float32, device=device).view(2, 19),
            1: torch.arange(2 * 19, dtype=torch.float32, device=device)
            .view(2, 19)
            .flip(1),
        }
        score_workspace = SimpleNamespace(max_requests=3)
        manager = TriAttention.__new__(TriAttention)
        manager.normalize_scores = True
        manager._fixed_shape_selection_enabled = True
        manager._fixed_shape_selection_prewarm_states = {}
        manager._fixed_shape_selection_workspaces = {}
        manager._fixed_shape_selection_bank_bytes = {}

        with mock.patch.object(
            base,
            "select_segments",
            wraps=base.select_segments,
        ) as select_segments:
            manager._prewarm_fixed_shape_selection_bucket(
                key,
                base,
                score_workspace,
                scores_by_layer,
                [0, 1],
                2,
                19,
            )

        bank = manager._fixed_shape_selection_workspaces[key]
        assert select_segments.call_count == 1
        assert len(bank) == score_workspace.max_requests
        assert all(item.prewarm_attempted and item.prewarmed for item in bank)
        assert manager._fixed_shape_selection_prewarm_states[key] == "ready"
        assert manager._fixed_shape_selection_bank_bytes[key] == (
            base.selection_buffer_nbytes() * score_workspace.max_requests
        )

        stage2_workspace = object()
        failed_base = _FixedUnionWorkspace(
            4,
            17,
            8,
            2,
            dtype=torch.float32,
            device=device,
        )
        failed = TriAttention.__new__(TriAttention)
        failed.normalize_scores = True
        failed._fixed_shape_selection_enabled = True
        failed._fixed_shape_selection_prewarm_states = {}
        failed._fixed_shape_selection_workspaces = {}
        failed._fixed_shape_selection_bank_bytes = {}
        failed._fixed_score_workspaces = {key: stage2_workspace}
        failed._fixed_score_prewarm_states = {key: "ready"}
        failed._build_fixed_shape_selection_workspaces = mock.Mock(
            side_effect=torch.cuda.OutOfMemoryError("sealed Stage3 bank")
        )

        failed._prewarm_fixed_shape_selection_bucket(
            key,
            failed_base,
            score_workspace,
            scores_by_layer,
            [0, 1],
            2,
            19,
        )

        assert failed._fixed_shape_selection_prewarm_states[key] == "failed"
        assert key not in failed._fixed_shape_selection_workspaces
        assert key not in failed._fixed_shape_selection_bank_bytes
        assert failed._fixed_score_workspaces[key] is stage2_workspace
        assert failed._fixed_score_prewarm_states[key] == "ready"
        assert not failed_base._segment_buffers_prepared
        assert not hasattr(failed_base, "input_scores")

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_fixed_shape_selection_alternates_r7_r1_without_stale_slots(
        self, device
    ):
        from types import SimpleNamespace

        device = torch.device(device)
        key = ("bucket",)
        max_requests = 8
        prompt_len = 1
        seq_len = 8
        decode_width = seq_len - prompt_len
        keep_count = 3
        base = _FixedUnionWorkspace(
            4,
            decode_width,
            keep_count,
            prompt_len,
            dtype=torch.float32,
            device=device,
        )
        bank = TriAttention._build_fixed_shape_selection_workspaces(
            base,
            max_requests,
        )
        for item in bank:
            item.prewarm_attempted = True
            item.prewarmed = True

        pool = torch.empty(1, 2, 1, 1, 1, device=device)

        def build_manager(fixed):
            manager = TriAttention.__new__(TriAttention)
            manager.top_B = keep_count
            manager.pin_prefill = True
            manager.count_prompt_tokens = False
            manager.eviction_mode = "union"
            manager.normalize_scores = True
            manager.kv_cache_manager = SimpleNamespace(
                get_buffers=lambda layer, kv_layout: pool
            )
            manager._dense_layers = lambda num_layers: [0, 1]
            manager._global_layer_id = lambda layer, num_layers: layer
            manager._indexer_topk_supported = lambda width, k: False
            manager._fixed_union_enabled = False
            manager._fixed_union_active = {}
            manager._fixed_shape_selection_enabled = fixed
            manager._fixed_shape_selection_workspaces = {key: bank} if fixed else {}
            manager._fixed_shape_selection_prewarm_states = (
                {key: "ready"} if fixed else {}
            )
            manager._fixed_shape_selection_runtime_counts = {}
            return manager

        fixed_manager = build_manager(True)
        eager_manager = build_manager(False)
        score_workspace = SimpleNamespace(prewarm_key=key)
        tensor_pointers = [
            {
                name: value.data_ptr()
                for name, value in vars(workspace).items()
                if isinstance(value, torch.Tensor)
            }
            for workspace in bank
        ]
        previous_metadata = None

        for iteration, request_count in enumerate((7, 1) * 95):
            request_order = [
                (iteration * 5 + slot) % max_requests
                for slot in range(request_count)
            ]
            page_tables = [
                ((request_id + iteration) % 23, (request_id + iteration + 7) % 23)
                for request_id in request_order
            ]
            round_starts = [
                5000.0 + iteration * 128.0 + request_id
                for request_id in request_order
            ]
            metadata = (tuple(request_order), tuple(page_tables), tuple(round_starts))
            if previous_metadata is not None:
                assert metadata != previous_metadata
            previous_metadata = metadata

            for workspace in bank:
                workspace.input_scores.fill_(float("nan"))
                workspace.keep[prompt_len:].fill_(-1)
            fixed_workspaces = fixed_manager._fixed_shape_selection_for(
                score_workspace,
                request_count,
            )
            assert fixed_workspaces == bank[:request_count]

            for slot, (request_id, pages, round_start) in enumerate(
                zip(request_order, page_tables, round_starts)
            ):
                token = torch.arange(seq_len, dtype=torch.float32, device=device)
                phase = round_start * 1e-4 + sum(pages) * 1e-3
                precomputed = []
                for layer in range(2):
                    rows = []
                    for head in range(2):
                        scale = 0.071 + layer * 0.037 + head * 0.019
                        rows.append(
                            torch.sin(token * scale + phase)
                            + token * (1e-3 + layer * 2e-4 + head * 3e-4)
                        )
                    precomputed.append(torch.stack(rows))
                request = SimpleNamespace(
                    py_request_id=request_id,
                    py_prompt_len=prompt_len,
                )

                fixed_keep = fixed_manager._evict_modes(
                    request,
                    2,
                    seq_len,
                    precomputed,
                    fixed_union_workspace=fixed_workspaces[slot],
                ).clone()
                eager_keep = eager_manager._evict_modes(
                    request,
                    2,
                    seq_len,
                    precomputed,
                ).clone()

                assert torch.equal(fixed_keep, eager_keep)
                assert torch.isfinite(fixed_workspaces[slot].input_scores).all()

            for workspace in bank[request_count:]:
                assert torch.isnan(workspace.input_scores).all()
                assert torch.equal(
                    workspace.keep[prompt_len:],
                    torch.full_like(workspace.keep[prompt_len:], -1),
                )
            assert tensor_pointers == [
                {
                    name: value.data_ptr()
                    for name, value in vars(workspace).items()
                    if isinstance(value, torch.Tensor)
                }
                for workspace in bank
            ]

        assert fixed_manager._fixed_shape_selection_runtime_counts[key] == {
            "hit": 190,
            "fallback": 0,
        }
        assert fixed_manager._fixed_shape_selection_for(
            score_workspace,
            max_requests + 1,
        ) is None
        assert fixed_manager._fixed_shape_selection_runtime_counts[key] == {
            "hit": 190,
            "fallback": 1,
        }
        fixed_manager._fixed_shape_selection_prewarm_states[key] = "failed"
        assert fixed_manager._fixed_shape_selection_for(score_workspace, 1) is None
        assert fixed_manager._fixed_shape_selection_runtime_counts[key] == {
            "hit": 190,
            "fallback": 2,
        }

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_fixed_segment_ties_preserve_union_and_selected_value_multiset(
        self, device
    ):
        device = torch.device(device)
        width = 41
        keep_count = 8
        prompt_len = 3
        token = torch.arange(width, device=device)
        segments = [
            torch.stack([(token % 7).float().roll(shift) for shift in (0, 2)]),
            torch.stack([(token % 7).float().roll(shift) for shift in (4, 6)]),
        ]
        scores = torch.cat(segments, dim=0)
        workspace = _FixedUnionWorkspace(
            4,
            width,
            keep_count,
            prompt_len,
            dtype=scores.dtype,
            device=scores.device,
            allocate_segment_buffers=True,
        )

        fixed = workspace.select_segments(
            segments,
            normalize_scores=False,
        ).clone()
        reference = self._reference(scores, keep_count)
        combined = scores.max(dim=0).values
        row_top = torch.topk(scores, keep_count, dim=1, sorted=False).indices
        union_mask = torch.zeros(width, dtype=torch.bool, device=device)
        union_mask.scatter_(0, row_top.reshape(-1), True)
        fixed_decode = fixed[prompt_len:] - prompt_len

        assert union_mask.index_select(0, fixed_decode).all()
        assert torch.equal(
            combined.index_select(0, fixed_decode).sort().values,
            combined.index_select(0, reference).sort().values,
        )

    def test_finite_ties_are_repeatable_and_preserve_selected_values(self):
        width = 257
        keep_count = 32
        token = torch.arange(width)
        scores = torch.stack([(token % 11).float().roll(shift) for shift in (0, 3, 7)])
        workspace = _FixedUnionWorkspace(
            3,
            width,
            keep_count,
            0,
            dtype=scores.dtype,
            device=scores.device,
        )

        first = workspace.select(scores).clone()
        second = workspace.select(scores).clone()
        reference = self._reference(scores, keep_count)
        combined = scores.max(dim=0).values

        assert torch.equal(first, second)
        assert torch.equal(
            combined.index_select(0, first).sort().values,
            combined.index_select(0, reference).sort().values,
        )

    def test_manager_routes_only_large_torch_topk_bucket(self):
        from types import SimpleNamespace

        mgr = TriAttention.__new__(TriAttention)
        mgr._fixed_union_enabled = True
        mgr._fixed_union_workspaces = {}
        mgr._fixed_union_active = {}
        request = SimpleNamespace(py_request_id=7)
        large = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
        keep_count = 3072
        fallback = mgr._evict_union(request, 1, 4100, 4, keep_count, large)

        assert not mgr._fixed_union_workspaces
        fixed = mgr._evict_union(
            request,
            1,
            4100,
            4,
            keep_count,
            large,
            use_fixed_workspace=True,
        )
        workspace = mgr._get_fixed_union_workspace(7, large, keep_count, 4)

        assert workspace is not None
        assert torch.equal(fixed, fallback)
        assert mgr._get_fixed_union_workspace(7, large, keep_count, 4) is workspace
        assert mgr._get_fixed_union_workspace(7, large[:, :128], keep_count, 4) is None

    def test_request_cleanup_does_not_drop_other_request_bucket(self):
        mgr = TriAttention.__new__(TriAttention)
        mgr._fixed_union_workspaces = {
            (7, "shape-a"): object(),
            (7, "shape-b"): object(),
            (8, "shape-a"): object(),
        }
        mgr._fixed_union_active = {7: object(), 8: object()}

        mgr._clear_fixed_union_workspaces(7)

        assert set(mgr._fixed_union_workspaces) == {(8, "shape-a")}
        assert 7 not in mgr._fixed_union_active
        assert 8 in mgr._fixed_union_active


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
