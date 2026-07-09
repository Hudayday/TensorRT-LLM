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

These tests cover fixed-buffer selection, page-table preparation, graph
bookkeeping, and request lifecycle behavior. Model-level correctness is covered
by separate end-to-end tests.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention import TriAttention
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _BatchedFixedUnionWorkspace,
    _build_swa_rebase_copy,
    _build_swa_rebase_keep,
    _FixedShapeSelectionPlan,
    _FixedUnionWorkspace,
)

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
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

_TORCH_TOPK_ORACLE = torch.topk


def _fake_cute_dsl_topk(
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    next_n: int,
) -> None:
    """CPU oracle for the CUDA-only CuTE-DSL selector custom op."""
    assert next_n == 1
    for row in range(int(values.shape[0])):
        width = int(seq_lens[row])
        selected = _TORCH_TOPK_ORACLE(
            values[row, :width],
            top_k,
            sorted=False,
        ).indices
        output[row].copy_(selected.to(torch.int32))


@contextmanager
def _mock_cute_topk_without_fallbacks():
    """Provide the CuTE op while making both retired fallbacks fatal."""
    with (
        mock.patch.object(
            torch.ops.trtllm,
            "cute_dsl_indexer_topk_decode",
            side_effect=_fake_cute_dsl_topk,
            create=True,
        ) as cute_topk,
        mock.patch.object(
            torch.ops.trtllm,
            "indexer_topk_decode",
            side_effect=AssertionError("native IndexerTopK fallback is forbidden"),
            create=True,
        ),
        mock.patch.object(
            torch,
            "topk",
            side_effect=AssertionError("torch.topk production fallback is forbidden"),
        ),
    ):
        yield cute_topk


def _topk_oracle(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Return sorted per-row indices for an independent expected result."""
    return torch.sort(
        _TORCH_TOPK_ORACLE(scores, keep_count, dim=1, sorted=False).indices,
        dim=1,
    ).values


def _union_oracle(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Independent expected-result implementation of union selection."""
    combined = scores.max(dim=0).values
    row_top = _TORCH_TOPK_ORACLE(
        scores,
        keep_count,
        dim=1,
        sorted=False,
    ).indices
    union_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    union_mask.scatter_(0, row_top.reshape(-1), True)
    union_indices = torch.nonzero(union_mask, as_tuple=False).flatten()
    if union_indices.numel() >= keep_count:
        candidates = combined.index_select(0, union_indices)
        relative = _TORCH_TOPK_ORACLE(
            candidates,
            keep_count,
            sorted=False,
        ).indices
        return torch.sort(union_indices.index_select(0, relative)).values

    remaining = keep_count - int(union_indices.numel())
    residual = combined.clone()
    residual[union_mask] = float("-inf")
    extra = _TORCH_TOPK_ORACLE(
        residual,
        remaining,
        sorted=False,
    ).indices
    return torch.sort(torch.cat((union_indices, extra))).values


def _distinct_topk_scores(width: int, rows: int = 2) -> torch.Tensor:
    """Create deterministic finite rows without top-k boundary ties."""
    token = torch.arange(width, dtype=torch.float32)
    return torch.stack(
        [
            torch.sin(token * (0.0017 + row * 0.0003))
            + token * (0.00011 + row * 0.000013)
            + row * 0.000001
            for row in range(rows)
        ]
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


def _make_fake_v2(enable_block_reuse=False, *, is_draft=False):
    """Build an unallocated V2 double with TriAttention's production contract."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    fake_v2.is_draft = is_draft
    fake_v2.kv_factor = 2
    fake_v2.mapping = SimpleNamespace(enable_attention_dp=False)
    fake_v2.is_disagg = False
    fake_v2.max_beam_width = 1
    fake_v2.num_extra_kv_tokens = 0
    fake_v2.max_total_draft_tokens = 0
    fake_v2._kv_reserve_draft_tokens = 0
    fake_v2.max_seq_len = 65536
    fake_v2.max_attention_window_vec = []
    fake_v2.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    fake_v2.impl = object()
    fake_v2.kv_cache_map = {}
    fake_v2.host_kv_cache_block_offsets = torch.empty(1, dtype=torch.int64)
    fake_v2.pp_layers = []
    fake_v2.layer_offsets = {}
    fake_v2.layer_to_pool_mapping_dict = {}
    return fake_v2


def _make_triattention(**overrides):
    """Construct a fully initialized manager for method-level unit tests."""
    options = {"top_B": 8, "skip_swa": False}
    options.update(overrides)
    return TriAttention(_make_fake_v2(), **options)


def _make_request(request_id, **overrides):
    """Build the explicit request fields consumed by TriAttention."""
    fields = {
        "py_request_id": request_id,
        "py_prompt_len": 0,
        "py_draft_tokens": [],
        "py_num_accepted_draft_tokens": 0,
        "is_dummy": False,
        "state": LlmRequestState.GENERATION_IN_PROGRESS,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_hf_config(**values):
    """Expose the normalized Hugging Face text-config contract."""
    text_config = SimpleNamespace(to_dict=lambda: dict(values))
    return SimpleNamespace(get_text_config=lambda: text_config)


# ---------------------------------------------------------------------------
# Public package surface (compression manager + block-free subclass only).
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_implementation_has_no_dynamic_attribute_probes(self):
        repo_root = Path(__file__).resolve().parents[4]
        package_root = (
            repo_root / "tensorrt_llm" / "_torch" / "kv_cache_compression" / "triattention"
        )
        violations = []
        for source_path in package_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"getattr", "hasattr"}
                ):
                    violations.append(
                        f"{source_path.relative_to(repo_root)}:{node.lineno}:{node.func.id}"
                    )
        assert not violations, "dynamic attribute probes:\n" + "\n".join(violations)

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

    def test_llm_args_round_trip_preserves_concrete_config_fields(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

        args = TorchLlmArgs(
            model="dummy",
            kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
                top_B=16384,
                beta=16384,
                skip_swa=False,
            ),
        )

        serialized = args.model_dump()
        assert serialized["kv_cache_compression_config"]["top_B"] == 16384
        assert serialized["kv_cache_compression_config"]["beta"] == 16384
        explicit = args.model_dump(exclude_unset=True)
        assert explicit["kv_cache_compression_config"]["top_B"] == 16384
        assert explicit["kv_cache_compression_config"]["beta"] == 16384
        restored = TorchLlmArgs.model_validate(serialized)
        apply_model_defaults_to_llm_args(
            restored,
            {"kv_cache_config": {"enable_block_reuse": False}},
        )
        assert isinstance(
            restored.kv_cache_compression_config,
            TriAttentionKvCacheCompressionConfig,
        )
        assert restored.kv_cache_compression_config.top_B == 16384
        assert restored.kv_cache_compression_config.beta == 16384

    def test_llm_args_dispatches_concrete_and_unknown_algorithms(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        tri_args = TorchLlmArgs(
            model="dummy",
            kv_cache_compression_config={"algorithm": "triattention"},
        )
        assert isinstance(
            tri_args.kv_cache_compression_config,
            TriAttentionKvCacheCompressionConfig,
        )
        assert tri_args.kv_cache_compression_config.top_B == 2048
        assert tri_args.kv_cache_compression_config.beta == 128

        unknown_args = TorchLlmArgs(
            model="dummy",
            kv_cache_compression_config={"algorithm": "future_method"},
        )
        assert type(unknown_args.kv_cache_compression_config) is KvCacheCompressionConfig

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

    def test_constructor_initializes_optional_runtime_state(self):
        manager = _make_triattention()

        assert manager._local_to_global_layers_cache is None
        assert manager._attention_layer_partition_cache is None
        assert manager._fixed_union_active == {}
        assert manager._fixed_score_runtime_counts == {}
        assert manager._standalone_graph_cache is None

    def test_resolve_requires_calibration_path(self):
        mgr = _make_triattention()
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

    def test_triattention_enables_capacity_only_on_target_manager(self):
        manager = _make_fake_v2()
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True
        first = _make_request(11)
        second = _make_request(12)

        triattention.on_request_init(first)
        triattention.on_request_init(second)

        assert manager.generation_capacity_only
        assert triattention._initialized_request_ids == {11, 12}

    def test_request_init_rejects_native_v2_swa(self):
        manager = _make_fake_v2()
        manager.max_attention_window_vec = [128]
        triattention = TriAttention(manager, top_B=128, skip_swa=False)
        triattention._calibrated = True
        request = _make_request(11)

        with pytest.raises(ValueError, match="full-attention V2 lifecycles"):
            triattention.on_request_init(request)
        assert triattention._initialized_request_ids == set()

    def test_request_init_rejects_attention_dp(self):
        manager = _make_fake_v2()
        manager.mapping = SimpleNamespace(enable_attention_dp=True)
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True

        with pytest.raises(ValueError, match="attention DP"):
            triattention.on_request_init(_make_request(11))

    def test_request_init_accepts_speculative_capacity(self):
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 4
        manager.max_total_draft_tokens = 4
        manager._kv_reserve_draft_tokens = 4
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        triattention._calibrated = True
        request = _make_request(11)

        triattention.on_request_init(request)

        assert manager.generation_capacity_only
        assert triattention._initialized_request_ids == {11}

    def test_skip_swa_requires_model_path(self):
        with pytest.raises(ValueError, match="skip_swa=True requires model_path"):
            TriAttention(_make_fake_v2(), top_B=8)

    def test_validate_rejects_missing_keys(self):
        mgr = _make_triattention()
        with pytest.raises(ValueError, match="missing keys"):
            mgr._validate_calibration({"E_q": torch.zeros(1)})

    def test_resolve_rejects_unrecognized_pt(self, tmp_path):
        path = tmp_path / "junk.pt"
        torch.save({"E_q": torch.zeros(1)}, path)
        mgr = _make_triattention()
        mgr.calibration_path = str(path)
        with pytest.raises(ValueError, match="Unrecognized calibration"):
            mgr._resolve_calibration()

    @CUDA_REQUIRED
    def test_resolve_accepts_flat_pt(self, flat_calibration_pt):
        mgr = _make_triattention()
        mgr.calibration_path = flat_calibration_pt
        mgr.model_path = None
        loaded = mgr._resolve_calibration()
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded


# ---------------------------------------------------------------------------
# adjust_attention_metadata: preserve native scheduling semantics and subtract
# only the cumulative physically evicted length.
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

    def set_draft_kv_length_delta(self, delta):
        self.draft_kv_length_delta = list(delta)


class TestAttentionMetadataReconcile:
    def test_base_hook_is_noop(self):
        # The base compression manager exposes a no-op adjust_attention_metadata
        # so every model that registers no eviction manager is unaffected.
        assert hasattr(BaseKVCacheCompressionManager, "adjust_attention_metadata")
        mgr = BaseKVCacheCompressionManager.__new__(BaseKVCacheCompressionManager)
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)  # must not raise / mutate
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100]

    def test_no_eviction_preserves_native_first_draft_length(self):
        mgr = _make_triattention()
        meta = _FakeMetadata([63], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [63]
        assert meta.prompt_lens == [50]

    def test_cumulative_eviction_is_subtracted_from_native_length(self):
        mgr = _make_triattention()
        mgr._evicted = {7: 9}
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [91]
        assert meta.prompt_lens == [50]

    def test_prompt_len_clamped_to_compacted_cache(self):
        # A prompt longer than the whole compacted cache is clamped to num_cached.
        mgr = _make_triattention()
        mgr._evicted = {7: 9}
        meta = _FakeMetadata([100], [200], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq[0] == 91
        assert meta.prompt_lens == [91]

    def test_request_without_eviction_is_untouched(self):
        mgr = _make_triattention()
        meta = _FakeMetadata([100], [50], [7])
        mgr.adjust_attention_metadata(meta)
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100]
        assert meta.prompt_lens == [50]

    def test_none_kv_cache_params_is_a_noop(self):
        mgr = _make_triattention()
        meta = _FakeMetadata([100], [50], [7])
        meta.kv_cache_params = None

        mgr.adjust_attention_metadata(meta)

        assert meta.prompt_lens == [50]

    def test_none_prompt_lens_still_reconciles_cache_length(self):
        mgr = _make_triattention()
        mgr._evicted = {7: 9}
        meta = _FakeMetadata([100], [50], [7])
        meta.prompt_lens = None

        mgr.adjust_attention_metadata(meta)

        assert meta.kv_cache_params.num_cached_tokens_per_seq == [91]
        assert meta.prompt_lens is None

    def test_context_requests_skipped(self):
        # Only generation requests (index >= num_contexts) are reconciled.
        mgr = _make_triattention()
        mgr._evicted = {1: 20, 2: 9}
        meta = _FakeMetadata([100, 100], [50, 50], [1, 2], num_contexts=1)
        mgr.adjust_attention_metadata(meta)
        # request 1 is a context request -> untouched; request 2 is generation.
        assert meta.kv_cache_params.num_cached_tokens_per_seq == [100, 91]

    def test_vanilla_mtp_publishes_dense_draft_length_delta(self):
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        mgr = _make_triattention(
            spec_config=MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )
        mgr._evicted = {2: 37}
        meta = _FakeMetadata([40, 100], [40, 50], [1, 2], num_contexts=1)

        mgr.adjust_attention_metadata(meta)

        assert meta.kv_cache_params.num_cached_tokens_per_seq == [40, 63]
        assert meta.draft_kv_length_delta == [0, 37]

    def test_eagle3_paged_draft_attention_publishes_dense_length_delta(self):
        from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

        spec_config = Eagle3DecodingConfig(
            max_draft_len=4,
            speculative_model="/tmp/qwen3-eagle3-draft",
        )
        mgr = _make_triattention(
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )
        mgr._evicted = {2: 37}
        meta = _FakeMetadata([40, 100], [40, 50], [1, 2], num_contexts=1)

        mgr.adjust_attention_metadata(meta)

        assert meta.kv_cache_params.num_cached_tokens_per_seq == [40, 63]
        assert meta.draft_kv_length_delta == [0, 37]

    def test_dflash_private_context_does_not_publish_paged_draft_length_delta(self):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        mgr = _make_triattention(
            spec_config=DFlashDecodingConfig(max_draft_len=4),
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )
        mgr._evicted = {2: 37}
        meta = _FakeMetadata([40, 100], [40, 50], [1, 2], num_contexts=1)

        mgr.adjust_attention_metadata(meta)

        assert meta.kv_cache_params.num_cached_tokens_per_seq == [40, 63]
        assert not hasattr(meta, "draft_kv_length_delta")

    def test_eviction_cannot_exceed_native_cached_length(self):
        mgr = _make_triattention()
        mgr._evicted = {7: 101}
        meta = _FakeMetadata([100], [50], [7])

        with pytest.raises(RuntimeError, match="below its cumulative eviction count"):
            mgr.adjust_attention_metadata(meta)


class TestStepEndHookRefactor:
    """Periodic eviction runs through the framework's final update hook."""

    def test_triattention_prepare_only_snapshots_and_update_uses_final_hook(self):
        assert "prepare_resources" in TriAttention.__dict__
        assert "update_resources" not in TriAttention.__dict__
        assert "on_generation_step_end" in TriAttention.__dict__

    @pytest.mark.parametrize("top_B", [511, 512])
    def test_non_v2_manager_is_always_rejected(self, top_B):
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            TriAttention(SimpleNamespace(), top_B=top_B)

    def test_non_v2_draft_manager_is_rejected(self):
        with pytest.raises(TypeError, match="requires draft KVCacheManagerV2"):
            TriAttention(
                _make_fake_v2(),
                top_B=8,
                draft_kv_cache_manager=SimpleNamespace(),
            )

    def test_hook_runs_periodic_evict(self):
        import unittest.mock as mock

        mgr = _make_triattention()
        with mock.patch.object(TriAttention, "_periodic_evict") as pe:
            mgr.on_generation_step_end("BATCH")
            pe.assert_called_once_with("BATCH")

    def test_dummy_and_arbitrary_non_due_steps_do_not_materialize(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr, _, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        mgr._gen_steps = {7: 0}
        dummy = _make_request(99, is_dummy=True)
        allocation_error = AssertionError("non-eviction generation must not allocate a tensor")

        with (
            mock.patch.object(mgr, "_materialize_fixed_shape_selection_banks") as stage3,
            mock.patch.object(mgr, "_materialize_cross_request_selection_banks") as stage4,
            mock.patch.object(mgr, "_evict_requests") as evict,
            mock.patch.object(torch, "empty", side_effect=allocation_error),
            mock.patch.object(torch, "empty_like", side_effect=allocation_error),
            mock.patch.object(torch, "arange", side_effect=allocation_error),
            mock.patch.object(torch, "full", side_effect=allocation_error),
        ):
            mgr._periodic_evict(SimpleNamespace(generation_requests=[dummy]))
            for _ in range(mgr.beta - 1):
                mgr._periodic_evict(batch)

        stage3.assert_not_called()
        stage4.assert_not_called()
        evict.assert_not_called()
        assert mgr._gen_steps[7] == mgr.beta - 1
        assert mgr._standalone_graph_cache is None
        assert mgr._standalone_graph_arena_generation == 0

    def test_base_update_resources_fires_hook_after_native_managers(self):
        import unittest.mock as mock

        mgr = BaseKVCacheCompressionManager.__new__(BaseKVCacheCompressionManager)
        batch = mock.MagicMock()
        batch.context_requests_last_chunk = []
        with mock.patch.object(BaseKVCacheCompressionManager, "on_generation_step_end") as se:
            mgr.update_resources(batch)
            se.assert_called_once_with(batch, None)

    def test_step_end_hook_documents_direct_resize_boundary(self):
        doc = TriAttention.on_generation_step_end.__doc__

        assert "after native KV-cache updates" in doc
        assert "resize happens only after compaction" in doc

    def test_evicted_count_default_zero(self):
        mgr = TriAttention(_make_fake_v2(), top_B=8, beta=4, skip_swa=False)
        assert mgr.evicted_count(12345) == 0

    @staticmethod
    def _make_due_decode_request(seq_len):
        from types import SimpleNamespace
        from unittest import mock

        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=seq_len + 1,
        )
        batch = SimpleNamespace(generation_requests=[request])
        mgr = _make_triattention()
        mgr._calibrated = True
        cache = SimpleNamespace(
            capacity=seq_len,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        mgr.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda *args, **kwargs: None,
            kv_cache_map={7: cache},
            pp_layers=[0, 1],
            _stream=mock.Mock(),
            num_extra_kv_tokens=0,
        )
        mgr._L = 2
        mgr._gen_steps = {7: 127}
        mgr._evicted = {}
        mgr._confirmed_kv_lengths = {}
        mgr._initialized_request_ids = {7}
        mgr.beta = 128
        mgr.top_B = 4096
        mgr.pin_prefill = True
        mgr.count_prompt_tokens = False
        mgr._standalone_graph_cache = None
        mgr._standalone_graph_arena_generation = 0
        return mgr, request, batch

    def test_identity_gate_skips_exact_decode_only_budget(self):
        from unittest import mock

        mgr, _, batch = self._make_due_decode_request(seq_len=1024 + 4096)

        with (
            mock.patch.object(mgr, "_materialize_fixed_shape_selection_banks") as stage3,
            mock.patch.object(mgr, "_materialize_cross_request_selection_banks") as stage4,
            mock.patch.object(mgr, "_evict_requests") as evict,
        ):
            mgr._periodic_evict(batch)

        stage3.assert_not_called()
        stage4.assert_not_called()
        evict.assert_not_called()
        assert mgr._gen_steps[7] == 128
        assert mgr._standalone_graph_cache is None
        assert mgr._standalone_graph_arena_generation == 0

    def test_identity_gate_preserves_real_eviction_round(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        timeline = []
        event = mock.Mock()
        event.record.side_effect = lambda: timeline.append("event")
        event.synchronize.side_effect = lambda: timeline.append("sync")
        cache = mgr.kv_cache_manager.kv_cache_map[7]

        def compact(*args):
            timeline.append("stage5_dispatch")
            return [(7, 1024 + 4096)]

        def ensure_graph_buckets(*_args):
            timeline.append("ensure_graph_buckets")

        @contextlib.contextmanager
        def track_range(name, **kwargs):
            timeline.append(f"enter:{name}")
            yield
            timeline.append(f"exit:{name}")

        with (
            mock.patch.object(
                mgr,
                "_ensure_configured_graph_buckets",
                side_effect=ensure_graph_buckets,
            ) as ensure_buckets,
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
            mock.patch.object(tri_module, "nvtx_range", side_effect=track_range),
        ):
            mgr._periodic_evict(batch)

        ensure_buckets.assert_called_once_with([(request, 7)], 2)
        evict.assert_called_once_with([(request, 7)], 2, protected_tail_lengths={7: 0})
        event.record.assert_called_once_with()
        event.synchronize.assert_called_once_with()
        cache.resize.assert_called_once_with(1024 + 4096, None)
        assert timeline == [
            "ensure_graph_buckets",
            "stage5_dispatch",
            "enter:triattention.resize",
            "event",
            "sync",
            "exit:triattention.resize",
        ]

    def test_standalone_graph_coalesces_different_lengths_in_one_upper_bucket(self):
        from types import SimpleNamespace
        from unittest import mock

        mgr, request_a, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        request_b = _make_request(
            8,
            py_prompt_len=1024,
            max_beam_num_tokens=1024 + 4096 + 3,
        )
        request_c = _make_request(
            9,
            py_prompt_len=1024,
            max_beam_num_tokens=1024 + 4096 + 3,
        )
        batch = SimpleNamespace(generation_requests=[request_a, request_b, request_c])
        cache_b = SimpleNamespace(
            capacity=1024 + 4096 + 2,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        cache_c = SimpleNamespace(
            capacity=1024 + 4096 + 2,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        mgr.kv_cache_manager.kv_cache_map[8] = cache_b
        mgr.kv_cache_manager.kv_cache_map[9] = cache_c
        mgr._initialized_request_ids.update((8, 9))
        mgr._gen_steps[8] = 127
        mgr._gen_steps[9] = 127
        mgr._standalone_cuda_graph_enabled = True
        event = mock.Mock()

        def compact(group, _num_layers):
            return [
                (rid, mgr._minimum_evictable_length(request, mgr._confirmed_kv_lengths[rid]))
                for request, rid in group
            ]

        with (
            mock.patch.object(mgr, "_ensure_configured_graph_buckets"),
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request_a, 7), (request_b, 8), (request_c, 9)],
            2,
            protected_tail_lengths={7: 0, 8: 0, 9: 0},
        )
        mgr.kv_cache_manager.kv_cache_map[7].resize.assert_called_once_with(1024 + 4096, None)
        cache_b.resize.assert_called_once_with(1024 + 4096, None)
        cache_c.resize.assert_called_once_with(1024 + 4096, None)

    def test_cross_request_selection_coalesces_legacy_backend_boundary(self):
        from types import SimpleNamespace
        from unittest import mock

        prompt_len = 1024
        mgr, request_a, _ = self._make_due_decode_request(seq_len=prompt_len + 4095)
        request_b = _make_request(
            8,
            py_prompt_len=prompt_len,
            max_beam_num_tokens=prompt_len + 4096 + 1,
        )
        batch = SimpleNamespace(generation_requests=[request_a, request_b])
        cache_b = SimpleNamespace(
            capacity=prompt_len + 4096,
            history_length=prompt_len,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        mgr.kv_cache_manager.kv_cache_map[8] = cache_b
        mgr._initialized_request_ids.add(8)
        mgr._gen_steps[8] = 127
        mgr.top_B = 2048
        mgr._cross_request_selection_enabled = True
        event = mock.Mock()

        def compact(group, _num_layers):
            return [(rid, prompt_len + mgr.top_B) for _, rid in group]

        with (
            mock.patch.object(mgr, "_ensure_configured_graph_buckets"),
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request_a, 7), (request_b, 8)],
            2,
            protected_tail_lengths={7: 0, 8: 0},
        )

    def test_resize_failure_is_reported(self):
        from unittest import mock

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        cache = mgr.kv_cache_manager.kv_cache_map[7]
        cache.resize.return_value = False
        event = mock.Mock()

        with (
            mock.patch.object(mgr, "_ensure_configured_graph_buckets"),
            mock.patch.object(mgr, "_evict_requests", return_value=[(7, 1024 + 4096)]),
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            with pytest.raises(RuntimeError, match="Failed to resize compacted KV cache"):
                mgr._periodic_evict(batch)

        event.synchronize.assert_called_once_with()

    @pytest.mark.parametrize("accepted", [0, 1, 2, 3])
    def test_overlap_tail_is_excluded_from_selection_and_passed_to_graph(self, accepted):
        from unittest import mock

        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        confirmed = 1024 + 4096 + 1 + accepted
        reserve = 2
        current_growth = 4
        tail = reserve + current_growth
        retained = 1024 + 4096
        mgr, request, batch = self._make_due_decode_request(seq_len=confirmed)
        request.py_num_accepted_draft_tokens = accepted
        cache = mgr.kv_cache_manager.kv_cache_map[7]
        cache.capacity = confirmed + tail
        mgr.kv_cache_manager.num_extra_kv_tokens = reserve
        mgr._prepared_generation_growth = {7: current_growth}
        mgr._prepared_batch = SimpleNamespace(generation_requests=[request])
        mgr.spec_config = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
        mgr.draft_kv_cache_manager = _make_fake_v2(is_draft=True)
        event = mock.Mock()

        with (
            mock.patch.object(mgr, "_ensure_configured_graph_buckets"),
            mock.patch.object(mgr, "_evict_requests", return_value=[(7, retained)]) as evict,
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            mgr._periodic_evict(batch)

        evict.assert_called_once_with([(request, 7)], 2, protected_tail_lengths={7: tail})
        assert mgr._confirmed_kv_lengths[7] == retained
        cache.resize.assert_called_once_with(retained + tail, None)

    def test_confirmed_length_comes_from_capacity_ledger_not_logical_length(self):
        from unittest import mock

        physical_confirmed = 6100
        manager = _make_triattention(beta=128)
        manager._calibrated = True
        manager._L = 2
        manager._gen_steps = {7: 0}
        manager._evicted = {7: 100}
        manager._initialized_request_ids = {7}
        cache = SimpleNamespace(
            capacity=physical_confirmed,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}
        manager.kv_cache_manager.pp_layers = [0, 1]
        manager.kv_cache_manager.num_extra_kv_tokens = 0
        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=999999,
            py_draft_tokens=[1, 2, 3, 4],
        )

        manager._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert manager._confirmed_kv_lengths[7] == physical_confirmed
        cache.resize.assert_not_called()

    def test_mla_selfkonly_cache_is_rejected(self):
        manager = _make_triattention()
        manager.kv_cache_manager.kv_factor = 1

        with pytest.raises(ValueError, match="standard key/value KV cache"):
            manager._validate_v2_compatibility()

    def test_one_model_mtp_target_only_contract_is_accepted(self):
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        spec_config = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        manager._validate_v2_compatibility()
        assert manager.kv_cache_manager.generation_capacity_only is True

    def test_mtp_eagle_paged_draft_length_contract_is_accepted(self):
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        spec_config = MTPDecodingConfig(max_draft_len=1)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        manager._validate_v2_compatibility()

    @pytest.mark.parametrize("mode", ["draft_target", "pard"])
    def test_unvalidated_paged_draft_tail_contracts_remain_fail_closed(self, mode):
        from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, PARDDecodingConfig

        if mode == "draft_target":
            spec_config = DraftTargetDecodingConfig(
                max_draft_len=3,
                speculative_model="/tmp/draft-target-model",
            )
        else:
            spec_config = PARDDecodingConfig(max_draft_len=3)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        with pytest.raises(ValueError, match="has not validated.*target-tail"):
            manager._validate_v2_compatibility()

    def test_linear_eagle3_one_model_target_only_contract_is_accepted(self):
        from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

        spec_config = Eagle3DecodingConfig(
            max_draft_len=4,
            speculative_model="/tmp/qwen3-eagle3-draft",
        )
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        manager._validate_v2_compatibility()

    @pytest.mark.parametrize(
        "config_overrides,error",
        [
            (
                {"eagle3_one_model": False},
                "has not validated.*target-tail",
            ),
            (
                {"use_dynamic_tree": True, "dynamic_tree_max_topK": 2},
                "requires linear drafting",
            ),
        ],
    )
    def test_eagle3_separate_model_and_tree_modes_remain_fail_closed(self, config_overrides, error):
        from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

        spec_config = Eagle3DecodingConfig(
            max_draft_len=4,
            speculative_model="/tmp/qwen3-eagle3-draft",
            **config_overrides,
        )
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        with pytest.raises(ValueError, match=error):
            manager._validate_v2_compatibility()

    def test_dflash_target_only_contract_is_accepted(self):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        spec_config = DFlashDecodingConfig(max_draft_len=3)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        manager._validate_v2_compatibility()

    def test_dflash_requires_actual_separate_draft_manager(self):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=DFlashDecodingConfig(max_draft_len=3),
        )

        with pytest.raises(ValueError, match="requires a separate draft KV cache"):
            manager._validate_v2_compatibility()

    def test_dflash_shared_target_draft_cache_is_rejected(self):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        spec_config = DFlashDecodingConfig(max_draft_len=3)
        spec_config._allow_separate_draft_kv_cache = False
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
        )

        with pytest.raises(ValueError, match="requires a separate draft KV cache"):
            manager._validate_v2_compatibility()

    def test_dflash_runtime_acceptance_gate_is_rejected(self):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        spec_config = DFlashDecodingConfig(
            max_draft_len=3,
            acceptance_window=16,
            acceptance_length_threshold=1.0,
        )
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            skip_swa=False,
            spec_config=spec_config,
        )

        with pytest.raises(ValueError, match="runtime speculative acceptance gating"):
            manager._validate_v2_compatibility()

    def test_prepare_snapshots_fixed_linear_generation_growth(self):
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 2
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=[1, 2, 3])],
        )

        triattention.prepare_resources(batch)

        assert triattention._prepared_batch is batch
        assert triattention._prepared_generation_growth == {7: 4}

    def test_prepare_protects_reserved_draft_width(self):
        manager = _make_fake_v2()
        manager._kv_reserve_draft_tokens = 6
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(manager, top_B=8, skip_swa=False)
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=[1, 2])],
        )

        triattention.prepare_resources(batch)

        assert triattention._prepared_generation_growth == {7: 7}

    @pytest.mark.parametrize("native_cached", [63, 69, 104])
    def test_overlap_metadata_preserves_native_length_without_eviction(self, native_cached):
        triattention = TriAttention(_make_fake_v2(), top_B=8, skip_swa=False)
        metadata = _FakeMetadata([native_cached], [10], [7])

        triattention.adjust_attention_metadata(metadata)

        assert metadata.kv_cache_params.num_cached_tokens_per_seq == [native_cached]

    def test_terminal_request_is_ignored(self):
        manager = _make_fake_v2()
        terminal = _make_request(7, state=LlmRequestState.GENERATION_COMPLETE)
        triattention = TriAttention(manager, top_B=8, beta=1, skip_swa=False)
        triattention._calibrated = True

        triattention._periodic_evict(SimpleNamespace(generation_requests=[terminal]))

        assert triattention._initialized_request_ids == set()
        assert triattention._gen_steps == {}
        assert triattention._confirmed_kv_lengths == {}

    def test_suspended_target_cache_fails_closed(self):
        manager = _make_fake_v2()
        request = _make_request(7)
        cache = SimpleNamespace(capacity=20, history_length=0, is_active=False)
        manager.kv_cache_map = {7: cache}
        triattention = TriAttention(manager, top_B=100, beta=8, skip_swa=False)
        triattention._calibrated = True

        with pytest.raises(RuntimeError, match="suspended target KV cache"):
            triattention._periodic_evict(SimpleNamespace(generation_requests=[request]))

    def test_generation_only_request_is_initialized_once(self):
        manager = _make_fake_v2()
        manager.get_buffers = lambda *args, **kwargs: None
        manager.kv_cache_map = {7: SimpleNamespace(capacity=11, history_length=2, is_active=True)}
        request = _make_request(
            7,
            py_prompt_len=2,
            max_beam_num_tokens=12,
        )
        mgr = TriAttention(manager, top_B=8, beta=4, skip_swa=False)
        mgr._calibrated = True
        mgr._L = 2

        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))
        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert manager.generation_capacity_only
        assert mgr._confirmed_kv_lengths == {7: 11}
        assert mgr._initialized_request_ids == {7}

    def test_generation_dummy_is_skipped(self):
        manager = _make_fake_v2()
        request = _make_request(7, is_dummy=True)
        mgr = TriAttention(manager, top_B=8, beta=4, skip_swa=False)
        mgr._calibrated = True

        mgr._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert mgr._initialized_request_ids == set()
        assert mgr._gen_steps == {}
        assert mgr._evicted == {}
        assert mgr._confirmed_kv_lengths == {}

    def test_request_finish_clears_compression_state(self):
        from types import SimpleNamespace

        request = SimpleNamespace(py_request_id=7)
        mgr = _make_triattention()
        mgr._gen_steps = {7: 1}
        mgr._evicted = {7: 127}
        mgr._confirmed_kv_lengths = {7: 128}
        mgr._prepared_generation_growth = {7: 1}
        mgr._initialized_request_ids = {7}

        mgr.on_request_finish(request)

        assert mgr._initialized_request_ids == set()
        assert mgr._gen_steps == {}
        assert mgr._evicted == {}
        assert mgr._confirmed_kv_lengths == {}
        assert mgr._prepared_generation_growth == {}

    def test_evict_requests_builds_exact_graph_metadata(self):
        # Length accounting must use request state, not unused pool contents.
        pools = [torch.ones(2, 2, 1, 8, 2), torch.ones(2, 2, 1, 8, 2)]
        manager = SimpleNamespace(
            get_buffers=lambda layer, **kwargs: pools[layer],
            pp_layers=[0, 1],
            layer_offsets={0: 0, 1: 1},
            layer_to_pool_mapping_dict={0: 0, 1: 1},
            num_extra_kv_tokens=0,
            _kv_reserve_draft_tokens=0,
        )
        request = _make_request(7, py_prompt_len=2, max_beam_num_tokens=999)
        mgr = _make_triattention()
        mgr.kv_cache_manager = manager
        mgr._evicted = {7: 5}
        mgr._confirmed_kv_lengths = {7: 8}
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
        mgr._local_score_calibration = mock.Mock(
            return_value=(torch.ones(1), torch.ones(1), torch.ones(1))
        )
        score_workspace = object()
        selection_workspace = object()
        mgr._fixed_score_workspace_for = mock.Mock(return_value=score_workspace)
        mgr._attach_page_ids = mock.Mock(return_value=True)
        mgr._cross_request_selection_for = mock.Mock(return_value=selection_workspace)
        captured = {}

        def run_graph(**kwargs):
            captured.update(kwargs)
            return [(7, 6)]

        mgr._try_standalone_cuda_graph = mock.Mock(side_effect=run_graph)

        targets = mgr._evict_requests([(request, 7)], num_layers=2)

        assert targets == [(7, 6)]
        assert captured["score_workspace"] is score_workspace
        assert captured["selection_workspace"] is selection_workspace
        assert captured["dense_layers"] == [1]
        assert captured["swa_layers"] == [0]
        assert captured["dense_groups"] == [[1]]
        item = captured["prepared"][0]
        assert item["request"] is request
        assert item["request_id"] == 7
        assert item["seq_len"] == 8
        assert item["round_start"] == 13.0
        assert item["expected_keep_count"] == 6
        assert item["protected_tail"] == 0
        assert item["swa_source"].tolist() == [6, 7]
        assert item["swa_destination"].tolist() == [4, 5]
        mgr._attach_page_ids.assert_called_once()
        mgr._try_standalone_cuda_graph.assert_called_once()

    def test_distinct_v2_pools_are_passed_separately_to_the_graph(self):
        pools = [torch.zeros(2, 2, 1, 8, 2), torch.zeros(2, 2, 1, 8, 2)]
        request = _make_request(7, py_prompt_len=2, max_beam_num_tokens=9)
        mgr = _make_triattention()
        mgr.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda layer, **kwargs: pools[layer],
            pp_layers=[0, 1],
            layer_offsets={0: 0, 1: 1},
            layer_to_pool_mapping_dict={0: 10, 1: 20},
            num_extra_kv_tokens=0,
            _kv_reserve_draft_tokens=0,
        )
        mgr._evicted = {}
        mgr._confirmed_kv_lengths = {7: 8}
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
        mgr._local_score_calibration = mock.Mock(
            return_value=(torch.ones(1), torch.ones(1), torch.ones(1))
        )
        score_workspace = object()
        selection_workspace = object()
        mgr._fixed_score_workspace_for = mock.Mock(return_value=score_workspace)
        mgr._attach_page_ids = mock.Mock(return_value=True)
        mgr._cross_request_selection_for = mock.Mock(return_value=selection_workspace)
        captured = {}

        def run_graph(**kwargs):
            captured.update(kwargs)
            return [(7, 6)]

        mgr._try_standalone_cuda_graph = mock.Mock(side_effect=run_graph)

        targets = mgr._evict_requests([(request, 7)], num_layers=2)

        assert targets == [(7, 6)]
        assert captured["dense_groups"] == [[0], [1]]
        assert captured["layer_group_representative"] == {0: 0, 1: 1}
        attach_call = mgr._attach_page_ids.call_args
        assert attach_call.args[1] == [0, 1]
        assert attach_call.args[4] == [0, 1]


class TestTopKRouting:
    @pytest.mark.parametrize("keep_count", [4096, 8192])
    def test_eager_topk_uses_cute_without_fallback(self, keep_count):
        width = keep_count + 64
        scores = _distinct_topk_scores(width, rows=3)
        expected = _topk_oracle(scores, keep_count)
        manager = _make_triattention(top_B=keep_count)

        with _mock_cute_topk_without_fallbacks() as cute_topk:
            actual = manager._cute_dsl_topk_idx(scores, keep_count)

        cute_topk.assert_called_once()
        assert actual.dtype == torch.long
        assert torch.equal(torch.sort(actual, dim=1).values, expected)

    @pytest.mark.parametrize("keep_count", [4096, 8192])
    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    def test_per_head_modes_use_cute_without_fallback(self, keep_count, eviction_mode):
        width = keep_count + 64
        first = _distinct_topk_scores(width)
        second = _distinct_topk_scores(width).flip(1) + 0.000003
        per_layer_scores = [first, second]
        manager = _make_triattention(top_B=keep_count, eviction_mode=eviction_mode)
        manager._dense_layers = mock.Mock(return_value=[0, 1])
        manager._compact_perhead_layers = mock.Mock()
        request = SimpleNamespace(py_request_id=7)

        with _mock_cute_topk_without_fallbacks() as cute_topk:
            if eviction_mode == "per_head":
                kept_count = manager._evict_per_head(
                    request,
                    2,
                    width,
                    0,
                    keep_count,
                    per_layer_scores,
                )
                expected = [_topk_oracle(torch.stack(per_layer_scores).mean(dim=0), keep_count)]
            else:
                kept_count = manager._evict_per_layer_perhead(
                    request,
                    2,
                    width,
                    0,
                    keep_count,
                    per_layer_scores,
                )
                expected = [_topk_oracle(scores, keep_count) for scores in per_layer_scores]

        assert kept_count == keep_count
        assert cute_topk.call_count == len(expected)
        assert manager._compact_perhead_layers.call_count == len(expected)
        for call, expected_keep in zip(
            manager._compact_perhead_layers.call_args_list,
            expected,
        ):
            assert torch.equal(call.args[2], expected_keep)

    @pytest.mark.parametrize("keep_count", [4096, 8192])
    def test_fixed_union_uses_cute_without_fallback(self, keep_count):
        width = keep_count + 64
        scores = _distinct_topk_scores(width)
        expected = _union_oracle(scores, keep_count)
        workspace = _FixedUnionWorkspace(
            scores.shape[0],
            width,
            keep_count,
            0,
            dtype=scores.dtype,
            device=scores.device,
            selection_backend="cute_dsl_topk",
        )

        with _mock_cute_topk_without_fallbacks() as cute_topk:
            actual = workspace.select(scores)

        assert cute_topk.call_count == 2
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("keep_count", [4096, 8192])
    def test_cross_request_union_uses_cute_without_fallback(self, keep_count):
        width = keep_count + 64
        request_scores = [
            _distinct_topk_scores(width),
            _distinct_topk_scores(width).roll(17, dims=1) + 0.000007,
        ]
        expected = [_union_oracle(scores, keep_count) for scores in request_scores]
        workspace = _BatchedFixedUnionWorkspace(
            request_scores[0].shape[0],
            width,
            keep_count,
            0,
            dtype=request_scores[0].dtype,
            device=request_scores[0].device,
            selection_backend="cute_dsl_topk",
            max_requests=len(request_scores),
        )

        with _mock_cute_topk_without_fallbacks() as cute_topk:
            selected = workspace.select_requests(
                [[scores] for scores in request_scores],
                normalize_scores=False,
            )

        assert cute_topk.call_count == 2
        for actual, expected_keep in zip(selected, expected):
            assert torch.equal(actual.keep, expected_keep)


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
    def test_live_geometry_groups_by_v2_pool_not_tensor_storage(self):
        base = torch.empty(4, 2, 1, 8, 2)
        shared_storage = [base[:2], base[2:]]
        manager = _make_triattention()
        manager._dense_layers = mock.Mock(return_value=[0, 1])
        manager._local_to_global_layers = mock.Mock(return_value=[10, 11])
        manager.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda layer, **kwargs: shared_storage[layer - 10],
            layer_offsets={10: 100, 11: 101},
            layer_to_pool_mapping_dict={100: 3, 101: 4},
        )

        _, _, groups = manager._fixed_union_live_geometry(2)

        assert groups == [[0], [1]]

        separate_storage = [torch.empty_like(base[:2]), torch.empty_like(base[:2])]
        manager.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda layer, **kwargs: separate_storage[layer - 10],
            layer_offsets={10: 100, 11: 101},
            layer_to_pool_mapping_dict={100: 3, 101: 3},
        )

        _, _, groups = manager._fixed_union_live_geometry(2)

        assert groups == [[0, 1]]

    def test_page_table_pool_keys_and_slots_deduplicate_only_identical_v2_pools(self):
        from types import SimpleNamespace

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        manager = _make_triattention()
        manager.kv_cache_manager = SimpleNamespace(
            layer_offsets={10: 100, 11: 101, 12: 102},
            layer_to_pool_mapping_dict={100: 3, 101: 3, 102: 4},
        )
        representatives = [0, 1, 2]
        global_layers = [10, 11, 12]

        keys = manager._page_table_pool_keys(representatives, global_layers)
        unique_global, slots = _FixedScoreMetadataWorkspace._page_table_slot_layout(
            representatives,
            global_layers,
            keys,
        )

        assert keys == [("pool", 3), ("pool", 3), ("pool", 4)]
        assert unique_global == (10, 12)
        assert slots == {0: 0, 1: 0, 2: 1}

    def test_page_table_pool_keys_reject_missing_v2_mapping(self):
        manager = _make_triattention()
        manager.kv_cache_manager = _make_fake_v2()

        with pytest.raises(RuntimeError, match="invalid layer-to-pool mapping"):
            manager._page_table_pool_keys([0, 2], [10, 11, 12])

    def test_v2_batch_indices_only_convert_live_or_requested_blocks(self):
        from types import SimpleNamespace

        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
        from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

        manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
        manager.kv_factor = 2
        manager.index_scales = [4]
        manager.kv_cache_map = {
            7: SimpleNamespace(
                num_blocks=3,
                get_base_page_indices=lambda pool_id: [4, BAD_PAGE_INDEX, 8, 99, 100],
            )
        }

        assert manager._get_batch_cache_indices_by_pool_id([7]) == [[8, BAD_PAGE_INDEX, 16]]
        assert manager._get_batch_cache_indices_by_pool_id([7], num_blocks_per_seq=[2]) == [
            [8, BAD_PAGE_INDEX]
        ]

    def test_fixed_stage_rejects_bad_page_in_active_prefix_without_compacting_ordinals(self):
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
        workspace.global_representatives = (10,)
        workspace.page_count = 2
        workspace.bucket_seq_len = 8
        workspace.page_table_token_capacity = 8
        workspace.tokens_per_block = 4
        stream = object()
        get_batch = mock.Mock(return_value=[[7, -1]])

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(torch, "as_tensor") as as_tensor,
        ):
            assert not workspace.stage(get_batch, [42], [8.0])

        get_batch.assert_called_once_with([42], 10, num_blocks_per_seq=[workspace.page_count])
        as_tensor.assert_not_called()
        workspace.copy_done.record.assert_not_called()

    def test_fixed_stage_pads_only_after_each_requests_live_page_prefix(self):
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        workspace = _FixedScoreMetadataWorkspace.__new__(_FixedScoreMetadataWorkspace)
        workspace.device = torch.device("cuda")
        workspace.max_requests = 2
        workspace.stream = None
        workspace.copy_pending = False
        workspace.copy_done = mock.Mock()
        workspace.global_representatives = (10,)
        workspace.page_count = 3
        workspace.bucket_seq_len = 12
        workspace.page_table_token_capacity = 12
        workspace.tokens_per_block = 4
        workspace.page_ids_host = torch.empty((1, 2, 3), dtype=torch.int64)
        workspace.round_starts_host = torch.empty(2, dtype=torch.float32)
        workspace.valid_seq_lens_host = torch.empty(2, dtype=torch.int32)
        workspace.page_ids_device = mock.Mock()
        workspace.round_starts_device = mock.Mock()
        workspace.valid_seq_lens_device = mock.Mock()
        group = mock.Mock()
        workspace.fused_group = group
        stream = object()
        get_batch = mock.Mock(return_value=[[10, 11], [20, 21, 22]])

        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            assert workspace.stage(
                get_batch,
                [42, 43],
                [8.0, 9.0],
                [5, 9],
                [7, 11],
            )

        get_batch.assert_called_once_with([42, 43], 10, num_blocks_per_seq=[2, 3])
        assert workspace.page_ids_host.tolist() == [[[10, 11, 11], [20, 21, 22]]]
        assert workspace.valid_seq_lens_host.tolist() == [5, 9]
        group.stage_lengths.assert_called_once_with(workspace.valid_seq_lens_device, 2)
        workspace.copy_done.record.assert_called_once_with(stream)

    def test_bulk_page_table_copy_waits_for_the_manager_stream(self):
        from types import SimpleNamespace
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        workspace = _FixedScoreMetadataWorkspace.__new__(_FixedScoreMetadataWorkspace)
        workspace.device = torch.device("cuda")
        workspace.max_requests = 2
        workspace.page_count = 2
        workspace.global_representatives = (10,)
        workspace._bulk_stage_logged = True
        workspace.bulk_allocation_done = mock.Mock()
        workspace.bulk_copy_done = mock.Mock()
        bulk = mock.MagicMock()
        bulk.shape = (1, 2, 2, 4)
        converted = mock.Mock()
        bulk.__getitem__.return_value.__floordiv__.return_value = converted
        workspace._bulk_offsets_dst = None
        page_destination = mock.Mock()
        workspace.page_ids_device = mock.MagicMock()
        workspace.page_ids_device.__getitem__.return_value = page_destination

        manager_stream = mock.Mock()
        manager = SimpleNamespace(
            host_kv_cache_block_offsets=SimpleNamespace(shape=(1, 8, 2, 4), dtype=torch.int32),
            kv_factor=2,
            layer_offsets={10: 0},
            layer_to_pool_mapping_dict={0: 0},
            copy_batch_block_offsets=mock.Mock(),
            _stream=manager_stream,
        )
        current_stream = mock.Mock()
        calls = mock.Mock()
        workspace.bulk_allocation_done.record.side_effect = lambda *args: calls.allocation_record()
        manager_stream.wait_event.side_effect = lambda *args: calls.manager_wait()
        manager.copy_batch_block_offsets.side_effect = lambda *args: calls.copy()
        workspace.bulk_copy_done.record.side_effect = lambda *args: calls.record()
        current_stream.wait_event.side_effect = lambda *args: calls.wait()
        page_destination.copy_.side_effect = lambda *args: calls.read()

        with mock.patch.object(torch, "empty", return_value=bulk):
            assert workspace._stage_page_tables_bulk(manager, [7], current_stream)

        assert calls.mock_calls == [
            mock.call.allocation_record(),
            mock.call.manager_wait(),
            mock.call.copy(),
            mock.call.record(),
            mock.call.wait(),
            mock.call.read(),
        ]
        workspace.bulk_allocation_done.record.assert_called_once_with(current_stream)
        manager_stream.wait_event.assert_called_once_with(workspace.bulk_allocation_done)
        workspace.bulk_copy_done.record.assert_called_once_with(manager_stream)
        current_stream.wait_event.assert_called_once_with(workspace.bulk_copy_done)

    @pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
    def test_config_enables_one_graph_pipeline_without_environment_gates(self, eviction_mode):
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ,
            {
                "TRIATTN_FIXED_BUFFER_UNION": "0",
                "TRIATTN_FIXED_PREWARM": "0",
                "TRIATTN_FIXED_SCORE_METADATA": "0",
                "TRIATTN_FIXED_SHAPE_SELECTION": "0",
                "TRIATTN_CROSS_REQUEST_SELECTION": "0",
                "TRIATTN_STANDALONE_CUDA_GRAPH": "0",
            },
        ):
            manager = TriAttention(
                _make_fake_v2(),
                top_B=4096,
                eviction_mode=eviction_mode,
                skip_swa=False,
            )
            assert manager._fixed_union_enabled == (eviction_mode == "union")
            assert manager._fixed_union_prewarm_enabled
            assert manager._fixed_score_metadata_enabled
            assert not manager._fixed_shape_selection_enabled
            assert manager._cross_request_selection_enabled
            assert manager._standalone_cuda_graph_enabled

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_ready_workspace_selects_smallest_covering_upper_bucket(self, request_count):
        from types import SimpleNamespace
        from unittest import mock

        manager = _make_triattention()
        manager.top_B = 4
        manager._fixed_score_metadata_enabled = True
        manager._cross_request_selection_enabled = False
        workspace = SimpleNamespace(
            max_requests=8,
            prompt_len=2,
            bucket_seq_len=10,
            page_table_token_capacity=11,
            matches=mock.Mock(return_value=True),
            prewarm_key=("bucket",),
        )
        manager._fixed_score_workspaces = {("bucket",): workspace}
        manager._fixed_score_prewarm_states = {("bucket",): "ready"}
        manager._fixed_score_runtime_counts = {}
        pools = [torch.empty(1), torch.empty(1)]
        prepared = [
            {
                "request": SimpleNamespace(py_prompt_len=2),
                "seq_len": 8 + request_index % 3,
                "protected_tail": 0,
            }
            for request_index in range(request_count)
        ]

        result = manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared)

        assert result is workspace
        workspace.matches.assert_called_once_with(pools, [[0], [1]], [0, 1])
        workspace.matches.return_value = False
        assert (
            manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared) is None
        )
        workspace.matches.return_value = True
        manager._fixed_score_prewarm_states[("bucket",)] = "failed"
        assert (
            manager._fixed_score_workspace_for(pools, [0, 1], [[0], [1]], [], 2, prepared) is None
        )

    def test_busy_pinned_staging_waits_then_reuses_the_same_graph_buffers(self):
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
        workspace.copy_done = mock.Mock()
        workspace.copy_done.query.return_value = False
        workspace.global_representatives = (10,)
        workspace.page_count = 2
        workspace.bucket_seq_len = 8
        workspace.page_table_token_capacity = 8
        workspace.tokens_per_block = 4
        workspace.page_ids_host = torch.empty((1, 8, 2), dtype=torch.int64)
        workspace.round_starts_host = torch.empty(8, dtype=torch.float32)
        workspace.valid_seq_lens_host = torch.empty(8, dtype=torch.int32)
        workspace.page_ids_device = mock.Mock()
        workspace.round_starts_device = mock.Mock()
        workspace.valid_seq_lens_device = mock.Mock()
        workspace.fused_group = mock.Mock()
        get_batch = mock.Mock(return_value=[[4, 5]])

        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            assert workspace.stage(get_batch, [1], [8.0])
        workspace.copy_done.query.assert_called_once_with()
        workspace.copy_done.synchronize.assert_called_once_with()
        get_batch.assert_called_once_with([1], 10, num_blocks_per_seq=[2])

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
        workspace.copy_done = SimpleNamespace(query=mock.Mock(), synchronize=mock.Mock())
        get_batch = mock.Mock(side_effect=AssertionError("cross-stream workspace queried V2"))

        other_stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=5)
        with mock.patch.object(torch.cuda, "current_stream", return_value=other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
                workspace.stage(get_batch, [1], [8.0])
        workspace.copy_done.query.assert_not_called()
        workspace.copy_done.synchronize.assert_not_called()
        get_batch.assert_not_called()

    def test_attach_page_ids_does_not_swallow_cross_stream_rejection(self):
        from types import SimpleNamespace
        from unittest import mock

        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStreamMismatch,
        )

        manager = _make_triattention()
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
                "seq_len": 8,
                "protected_tail": 2,
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
        workspace.stage.assert_called_once_with(manager.kv_cache_manager, [7], [8.0], [8], [10])
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
        workspace.bucket_seq_len = 4
        workspace.page_table_token_capacity = 4
        workspace.tokens_per_block = 4
        buffer = mock.MagicMock()
        buffer.__getitem__.return_value = buffer
        buffer.copy_.return_value = buffer
        buffer.view.return_value = buffer
        workspace.page_ids_host = buffer
        workspace.round_starts_host = buffer
        workspace.valid_seq_lens_host = buffer
        workspace.page_ids_device = buffer
        workspace.round_starts_device = buffer
        workspace.valid_seq_lens_device = buffer
        workspace.fused_group = mock.Mock()
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
            mock.patch.object(torch, "add") as add,
            mock.patch.object(torch, "mul") as mul,
            mock.patch.object(torch, "cos") as cos,
            mock.patch.object(torch, "sin") as sin,
            mock.patch.object(torch, "mean") as mean,
        ):
            assert workspace.stage(
                lambda request_ids, layer, num_blocks_per_seq=None: [[3]],
                [7],
                [8.0],
            )
            add.assert_not_called()
            mul.assert_not_called()
            cos.assert_not_called()
            sin.assert_not_called()
            mean.assert_not_called()
            workspace.prepare_phase(1)
            add.assert_called_once()
            mul.assert_called_once()
            cos.assert_called_once()
            sin.assert_called_once()
            assert mean.call_count == 2
        assert workspace.stream is stream
        workspace.copy_done.record.assert_called_once_with(stream)

    def test_staged_page_tables_bypass_per_request_cuda_materialization(self):
        from types import SimpleNamespace
        from unittest import mock

        manager = _make_triattention()
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
            {
                "request": SimpleNamespace(),
                "request_id": 7,
                "round_start": 8.0,
                "seq_len": 8,
                "protected_tail": 2,
            },
            {
                "request": SimpleNamespace(),
                "request_id": 8,
                "round_start": 9.0,
                "seq_len": 9,
                "protected_tail": 3,
            },
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
        workspace.stage.assert_called_once_with(
            manager.kv_cache_manager,
            [7, 8],
            [8.0, 9.0],
            [8, 9],
            [10, 12],
        )
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
        page_count = 3
        seq_len = 7
        page_table_token_capacity = 11
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
            page_table_token_capacity=page_table_token_capacity,
        )
        assert workspace.bucket_seq_len == seq_len
        assert workspace.page_table_token_capacity == page_table_token_capacity
        assert workspace.page_count == page_count
        assert workspace.offsets.dtype == torch.float32
        assert workspace.offsets.is_contiguous()
        assert workspace.omega.dtype == torch.float32
        assert workspace.omega.is_contiguous()
        fused = workspace.fused_group
        for calibration in (*fused.pointer_middle[2:], *fused.pointer_tail):
            assert calibration.dtype == torch.float32
            assert calibration.is_contiguous()
        tables = {
            10: [
                [3 * request, 3 * request + 1, 3 * request + 2] for request in range(request_count)
            ],
            12: [
                [3 * request + 2, 3 * request + 1, 3 * request] for request in range(request_count)
            ],
            13: [
                [23 - 3 * request, 22 - 3 * request, 21 - 3 * request]
                for request in range(request_count)
            ],
        }

        def get_batch_side_effect(request_ids, layer, num_blocks_per_seq=None):
            assert num_blocks_per_seq == [page_count] * request_count
            return tables[layer]

        get_batch = mock.Mock(side_effect=get_batch_side_effect)
        request_ids = list(range(request_count))
        round_starts = [float(9 + request) for request in request_ids]

        assert workspace.stage(
            get_batch,
            request_ids,
            round_starts,
            [seq_len] * request_count,
            [10] * request_count,
        )
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
                page_ids.unsqueeze(0).contiguous(),
                [0] * len(layers),
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

    @pytest.mark.parametrize("request_count", [1, 7])
    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    @CUDA_REQUIRED
    def test_fused_score_spans_distinct_storages_and_block_tables(self, request_count, aggregation):
        """ONE launch over layers in DISTINCT storages with DISTINCT block tables.

        This is the production V2 shape: get_buffers wraps every layer as its
        own TensorWrapper storage and every layer allocates its own pages, so
        the fused path must not assume a shared storage anchor or a shared
        per-request block table.
        """
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            _FixedScoreGroup,
            flat_perhead_to_list,
            triton_tri_score_perhead,
        )

        device = torch.device("cuda")
        torch.manual_seed(20260707 + request_count)
        max_requests = 8
        page_count = 2
        seq_len = 7
        num_layers = 3
        # Three SEPARATE allocations (distinct storages, like V2 TensorWrapper).
        pools = [
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device)
            for _ in range(num_layers)
        ]
        assert len({pool.untyped_storage().data_ptr() for pool in pools}) == num_layers
        # A DIFFERENT block table per layer (per-layer page allocation).
        generator = torch.Generator(device="cpu").manual_seed(7 + request_count)
        page_ids_3d = torch.stack(
            [
                torch.randperm(max_requests * page_count, generator=generator)[
                    : max_requests * page_count
                ]
                .view(max_requests, page_count)
                .to(device=device, dtype=torch.int64)
                for _ in range(num_layers)
            ]
        ).contiguous()
        q_real = torch.randn(num_layers, 2, 2, device=device)
        q_imag = torch.randn(num_layers, 2, 2, device=device)
        mlr = torch.randn(num_layers, 2, 2, device=device)
        freq = torch.tensor([0.7, 1.3], device=device)
        omega = torch.tensor([0.013, 0.071], device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
        round_device = torch.arange(max_requests, dtype=torch.float32, device=device) + 9.0
        round_starts = round_device[:request_count].tolist()
        seq_lens = [seq_len] * request_count
        phase = (round_device[:, None, None] + offsets[None, :, None]) * omega[None, None]

        layer_order = list(range(num_layers))
        group = _FixedScoreGroup(
            pools,
            layer_order,
            max_requests,
            page_count,
            seq_len,
            2,
            page_ids_3d,
            layer_order,  # slot i holds layer i's tables
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
        )
        fixed, _ = group.launch(
            request_count,
            round_device,
            torch.cos(phase).mean(dim=1),
            torch.sin(phase).mean(dim=1),
            aggregation,
        )

        page_ids_per_layer = [
            [page_ids_3d[layer, request] for layer in layer_order]
            for request in range(request_count)
        ]
        eager, eager_offsets, _ = triton_tri_score_perhead(
            pools,
            [per_layer[0] for per_layer in page_ids_per_layer],
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
            layer_indices=layer_order,
            page_ids_per_layer=page_ids_per_layer,
        )
        # Fixed and eager fused launches must agree bit-for-bit.
        assert torch.equal(fixed, eager)

        # The fused launch must agree bit-for-bit with per-layer launches.
        segments = flat_perhead_to_list(eager, eager_offsets)
        for layer_slot, layer in enumerate(layer_order):
            single, single_offsets, _ = triton_tri_score_perhead(
                pools,
                [page_ids_3d[layer, request] for request in range(request_count)],
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
                layer_indices=[layer],
            )
            single_segments = flat_perhead_to_list(single, single_offsets)
            for request in range(request_count):
                assert torch.equal(
                    segments[request * num_layers + layer_slot], single_segments[request]
                )

        # And with the independent Torch oracle (per-layer block tables).
        oracle = _torch_tri_score_oracle(
            pools,
            {layer: page_ids_3d[layer, :request_count] for layer in layer_order},
            seq_lens,
            round_starts,
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            layer_order,
            aggregation,
        )
        for request in range(request_count):
            for layer_slot, layer in enumerate(layer_order):
                segment_index = request * num_layers + layer_slot
                segment = fixed[:, segment_index * seq_len : (segment_index + 1) * seq_len]
                expected = oracle[request * num_layers + layer]
                torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)

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
            manager = _make_triattention()
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
        group = workspace.fused_group
        stable_workspace_pointers = tensor_pointers(workspace)
        stable_group_pointers = tensor_pointers(group)
        stable_selection_pointers = [tensor_pointers(item) for item in selection_bank]
        previous_by_request_count = {}

        # Ninety-five full 7->1 cohort transitions reproduce the lifetime pattern
        # from the failing Stage2 run while changing both staged inputs every call.
        for iteration, request_count in enumerate((7, 1) * 95):
            torch.cuda.current_stream(device).synchronize()
            workspace.page_ids_host.fill_(-1)
            workspace.round_starts_host.fill_(float("nan"))
            shift = iteration % total_pages
            request_ids = [(iteration * 5 + slot) % max_requests for slot in range(request_count)]
            tables = [
                [(2 * request_id + shift + slot) % total_pages for slot in range(page_count)]
                for request_id in request_ids
            ]
            round_starts = [
                float(5000 + iteration * 128 + request_id) for request_id in request_ids
            ]

            previous = previous_by_request_count.get(request_count)
            if previous is not None:
                previous_tables, previous_round_starts = previous
                assert tables != previous_tables
                assert round_starts != previous_round_starts
            previous_by_request_count[request_count] = (tables, round_starts)

            def get_batch_cache_indices(observed_ids, global_layer, num_blocks_per_seq=None):
                assert observed_ids == request_ids
                assert global_layer == 10
                assert num_blocks_per_seq == [page_count] * request_count
                return tables

            assert workspace.stage(get_batch_cache_indices, request_ids, round_starts)
            workspace.prepare_phase(request_count)
            fixed, fixed_offsets = group.launch(
                request_count,
                workspace.round_starts_device,
                workspace.mean_cos,
                workspace.mean_sin,
                "mean",
            )
            eager_page_ids = [torch.tensor(row, dtype=torch.int64, device=device) for row in tables]
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
            selection_bank[0].input_scores.fill_(float("nan"))
            for item in selection_bank:
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
                precomputed = [fixed_views[:, request, layer] for layer in range(len(pools))]
                fixed_keep = selection_manager._evict_modes(
                    request_state,
                    len(pools),
                    seq_len,
                    precomputed,
                    fixed_union_workspace=active_selection[request],
                ).clone()
                oracle_keep = _torch_union_keep(
                    torch.cat(fixed_rows, dim=0), prompt_len=1, budget=3
                )
                assert torch.equal(fixed_keep, oracle_keep)

            for item in selection_bank[request_count:]:
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
            assert [tensor_pointers(item) for item in selection_bank] == (stable_selection_pointers)

        assert selection_manager._fixed_shape_selection_runtime_counts[selection_key] == {
            "hit": 190,
            "fallback": 0,
        }

    @CUDA_REQUIRED
    def test_gptoss_page_tables_stage_each_dense_and_swa_pool(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreMetadataWorkspace,
        )

        device = torch.device("cuda")
        seq_len = 8
        page_lists = {0: [1, 2], 1: [0, 1], 2: [2, 0]}
        pools = [torch.empty(3, 2, 1, 4, 4, dtype=torch.float32, device=device) for _ in range(3)]
        q_real = torch.ones(3, 2, 2, device=device)
        q_imag = torch.full_like(q_real, 0.25)
        mlr = torch.full_like(q_real, 0.1)
        freq = torch.tensor([0.8, 1.2], device=device)
        omega = torch.tensor([0.017, 0.043], device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
        workspace = _FixedScoreMetadataWorkspace(
            pools,
            [[1], [2]],
            [1, 2, 0],
            [0, 1, 2],
            2,
            seq_len,
            2,
            2,
            q_real,
            q_imag,
            mlr,
            freq,
            offsets,
            omega,
            page_table_keys=[("pool", 1), ("pool", 2), ("pool", 0)],
            prompt_len=2,
        )
        layer_offsets = {0: 0, 1: 1, 2: 2}
        layer_to_pool = {0: 0, 1: 1, 2: 2}
        bulk_calls = []

        def copy_batch_block_offsets(dst, request_ids, beam_width, num_contexts, num_seqs):
            bulk_calls.append((tuple(request_ids), beam_width, num_contexts, num_seqs))
            dst.zero_()
            for layer, pages in page_lists.items():
                pool_id = layer_to_pool[layer_offsets[layer]]
                page_offsets = torch.tensor(pages, dtype=dst.dtype, device=dst.device) * 2
                dst[pool_id, :num_seqs, 0, : len(pages)] = page_offsets

        cache = SimpleNamespace(
            host_kv_cache_block_offsets=torch.empty(3, 8, 2, 2, dtype=torch.int32, pin_memory=True),
            kv_factor=2,
            layer_offsets=layer_offsets,
            layer_to_pool_mapping_dict=layer_to_pool,
            copy_batch_block_offsets=copy_batch_block_offsets,
            _stream=torch.cuda.current_stream(device),
            get_batch_cache_indices=mock.Mock(
                side_effect=AssertionError("bulk V2 staging must not use the host path")
            ),
        )

        assert workspace.stage(cache, [7, 8], [8.0, 9.0], [8, 7])
        torch.cuda.current_stream(device).synchronize()

        expected = torch.tensor(
            [
                [page_lists[1], page_lists[1]],
                [page_lists[2], page_lists[2]],
                [page_lists[0], page_lists[0]],
            ],
            dtype=torch.int64,
            device=device,
        )
        assert torch.equal(workspace.page_ids_device[:, :2], expected)
        assert workspace.round_starts_device[:2].tolist() == [8.0, 9.0]
        assert workspace.valid_seq_lens_device[:2].tolist() == [8, 7]
        assert bulk_calls == [((7, 8), 1, 0, 2)]
        cache.get_batch_cache_indices.assert_not_called()


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

    def test_prewarm_shape_parser_is_explicit_and_fail_closed(self):
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

    def test_prewarm_upper_buckets_split_and_coalesce_selection_backend_bands(self):
        manager = _make_triattention()
        manager.top_B = 2048

        assert manager._upper_prewarm_shapes_by_backend(
            [(1024, 3072), (1024, 4096), (1024, 4224), (0, 8192)]
        ) == [
            (0, 8192),
            (1024, 4224),
        ]

    @pytest.mark.parametrize(
        "budget,beta,maximum_width,expected_widths",
        [
            (32, 4, 36, [36]),
            (2048, 2048, 4100, [4100]),
            (4096, 128, 4228, [4228]),
            (4096, 4096, 8196, [8196]),
        ],
    )
    def test_configured_budget_beta_k4_emits_backend_safe_upper_buckets(
        self, budget, beta, maximum_width, expected_widths
    ):
        manager = _make_triattention()
        manager.top_B = budget
        manager.beta = beta

        buckets = manager._upper_prewarm_shapes_by_backend([(1024, maximum_width)])

        assert buckets == [(1024, width) for width in expected_widths]

    def test_startup_prewarm_dispatches_first_and_steady_buckets(self):
        import os
        from unittest import mock

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        with (
            mock.patch.dict(
                os.environ,
                {"TRIATTN_FIXED_PREWARM_SHAPES": "1024:4096"},
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

        assert [call.args[-2:] for call in prewarm_bucket.call_args_list] == [(1024, 4096)]

    def test_runtime_graph_buckets_use_smallest_configured_upper_shape(self):
        import os
        from unittest import mock

        manager = _make_triattention(top_B=4096, beta=128)
        manager._confirmed_kv_lengths = {
            7: 1024 + 4223,
            8: 1024 + 4300,
            9: 512 + 4100,
        }
        layer_pools = [object()]
        dense_layers = [0]
        storage_groups = [[0]]
        manager._fixed_union_live_geometry = mock.Mock(
            return_value=(layer_pools, dense_layers, storage_groups)
        )
        manager._fixed_union_prewarm_key = mock.Mock(
            side_effect=lambda *args: (args[-1], args[-2] - args[-1])
        )

        def mark_ready(*args):
            key = (args[-2], args[-1])
            manager._fixed_union_prewarm_states[key] = "ready"
            manager._fixed_score_prewarm_states[key] = "ready"
            manager._cross_request_selection_prewarm_states[key] = "ready"

        manager._prewarm_fixed_union_bucket = mock.Mock(side_effect=mark_ready)
        manager._materialize_cross_request_selection_banks = mock.Mock()
        evict_now = [
            (SimpleNamespace(py_prompt_len=1024), 7),
            (SimpleNamespace(py_prompt_len=1024), 8),
            (SimpleNamespace(py_prompt_len=512), 9),
        ]

        with mock.patch.dict(
            os.environ,
            {"TRIATTN_FIXED_PREWARM_SHAPES": "512:4224,1024:4352"},
        ):
            manager._ensure_configured_graph_buckets(evict_now, num_layers=1)

        assert [call.args[-2:] for call in manager._prewarm_fixed_union_bucket.call_args_list] == [
            (512, 4224),
            (1024, 4352),
        ]
        manager._materialize_cross_request_selection_banks.assert_called_once_with()

    def test_runtime_graph_reuses_one_upper_shape_for_ragged_lengths_and_batches(self):
        import os
        from unittest import mock

        manager = _make_triattention(top_B=128, beta=4)
        layer_pools = [object()]
        dense_layers = [0]
        storage_groups = [[0]]
        upper_key = (64, 136)
        manager._fixed_union_live_geometry = mock.Mock(
            return_value=(layer_pools, dense_layers, storage_groups)
        )
        manager._fixed_union_prewarm_key = mock.Mock(
            side_effect=lambda *args: (args[-1], args[-2] - args[-1])
        )

        score_workspace = SimpleNamespace(
            prewarm_key=upper_key,
            prompt_len=64,
            bucket_seq_len=200,
            max_requests=2,
            matches=mock.Mock(return_value=True),
        )
        selection_workspace = SimpleNamespace(
            eviction_mode="union",
            selection_backend="cute_dsl_topk",
            max_requests=2,
            prompt_len=64,
            width=136,
            keep_count=128,
            dense_layers=(0,),
            num_query_heads=1,
            num_kv_heads=1,
        )
        selection_plan = SimpleNamespace(
            eviction_mode="union",
            dense_layers=(0,),
            selection_backend="cute_dsl_topk",
        )

        def mark_ready(*args):
            assert args[-2:] == (64, 136)
            manager._fixed_union_prewarm_states[upper_key] = "ready"
            manager._fixed_score_prewarm_states[upper_key] = "ready"
            manager._fixed_score_workspaces[upper_key] = score_workspace
            manager._cross_request_selection_prewarm_states[upper_key] = "ready"
            manager._cross_request_selection_workspaces[upper_key] = selection_workspace
            manager._cross_request_selection_plans[upper_key] = selection_plan

        manager._prewarm_fixed_union_bucket = mock.Mock(side_effect=mark_ready)
        manager._materialize_cross_request_selection_banks = mock.Mock()
        graph_keys = set()
        request_id = 0

        with mock.patch.dict(
            os.environ,
            {"TRIATTN_FIXED_PREWARM_SHAPES": "64:136"},
        ):
            for decode_width in (132, 133, 134, 136):
                for request_count in (1, 2):
                    evict_now = []
                    prepared = []
                    for _ in range(request_count):
                        request = SimpleNamespace(py_prompt_len=64)
                        manager._confirmed_kv_lengths[request_id] = 64 + decode_width
                        evict_now.append((request, request_id))
                        prepared.append(
                            {
                                "seq_len": 64 + decode_width,
                                "request": request,
                                "expected_keep_count": 192,
                                "protected_tail": 0,
                            }
                        )
                        request_id += 1

                    manager._ensure_configured_graph_buckets(evict_now, num_layers=1)
                    selected_score_workspace = manager._fixed_score_workspace_for(
                        layer_pools,
                        dense_layers,
                        storage_groups,
                        [],
                        1,
                        prepared,
                    )
                    assert selected_score_workspace is score_workspace
                    graph_keys.add(
                        manager._standalone_graph_bucket_for(
                            prepared,
                            selected_score_workspace,
                            selection_workspace,
                        )
                    )

        manager._prewarm_fixed_union_bucket.assert_called_once()
        assert manager._prewarm_fixed_union_bucket.call_args.args[-2:] == (64, 136)
        assert len(graph_keys) == 2
        assert {(key[3], key[4]) for key in graph_keys} == {(1, 200), (2, 200)}

    @pytest.mark.parametrize("configured_width", [128, 136])
    def test_runtime_graph_bucket_fails_when_configured_upper_shape_is_too_small(
        self, configured_width
    ):
        import os
        from unittest import mock

        manager = _make_triattention(top_B=128, beta=4)
        manager._confirmed_kv_lengths = {7: 64 + 137}
        manager._fixed_union_live_geometry = mock.Mock(return_value=([object()], [0], [[0]]))
        manager._prewarm_fixed_union_bucket = mock.Mock()
        evict_now = [(SimpleNamespace(py_prompt_len=64), 7)]

        with (
            mock.patch.dict(
                os.environ,
                {"TRIATTN_FIXED_PREWARM_SHAPES": f"64:{configured_width}"},
            ),
            pytest.raises(RuntimeError, match="no configured upper bucket covering"),
        ):
            manager._ensure_configured_graph_buckets(evict_now, num_layers=1)

        manager._prewarm_fixed_union_bucket.assert_not_called()

    def test_runtime_graph_bucket_uses_exact_width_without_startup_shapes(self):
        import os
        from unittest import mock

        manager = _make_triattention(top_B=4096, beta=4096)
        manager._confirmed_kv_lengths = {7: 163 + 8192}
        layer_pools = [object()]
        dense_layers = [0]
        storage_groups = [[0]]
        runtime_key = (163, 8192)
        manager._fixed_union_live_geometry = mock.Mock(
            return_value=(layer_pools, dense_layers, storage_groups)
        )
        manager._fixed_union_prewarm_key = mock.Mock(return_value=runtime_key)

        def mark_ready(*args):
            assert args[-2:] == runtime_key
            manager._fixed_union_prewarm_states[runtime_key] = "ready"
            manager._fixed_score_prewarm_states[runtime_key] = "ready"
            manager._cross_request_selection_prewarm_states[runtime_key] = "ready"

        manager._prewarm_fixed_union_bucket = mock.Mock(side_effect=mark_ready)
        manager._materialize_cross_request_selection_banks = mock.Mock()

        with mock.patch.dict(
            os.environ,
            {"TRIATTN_FIXED_PREWARM_SHAPES": ""},
        ):
            manager._ensure_configured_graph_buckets(
                [(SimpleNamespace(py_prompt_len=163), 7)],
                num_layers=1,
            )

        manager._prewarm_fixed_union_bucket.assert_called_once()
        manager._materialize_cross_request_selection_banks.assert_called_once_with()

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

    def test_prewarm_key_records_one_fixed_cute_backend_for_all_shapes(self):
        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048

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

        assert native[0] == "triattention.fixed-prewarm.v6"
        assert native[9] == "fixed_union.cute_dsl_topk"
        assert eager[9] == "fixed_union.cute_dsl_topk"
        assert fixed[9] == "fixed_union.cute_dsl_topk"
        assert native[11:18] == (
            1024 + 4095,
            1,
            1024,
            2048,
            4,
            4095,
            "union",
        )
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
        manager = _make_triattention()
        manager.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda layer, **kwargs: pools[layer],
            layer_offsets={0: 0, 1: 1},
            layer_to_pool_mapping_dict={0: 0, 1: 0},
            num_extra_kv_tokens=0,
            _kv_reserve_draft_tokens=0,
        )
        manager.top_B = 4
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
        manager._fixed_union_compaction_enabled = True
        manager._fixed_union_prewarm_enabled = True
        manager._fixed_union_prewarm_states = {}
        manager._fixed_union_prewarmed_workspaces = {}
        manager._fixed_union_enabled = True
        manager._fixed_union_workspaces = {}
        manager._fixed_union_active = {}
        manager._gen_steps = {}
        manager._evicted = {}
        manager._confirmed_kv_lengths = {}
        manager._initialized_request_ids = set()
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
            manager._confirmed_kv_lengths.copy(),
            manager._initialized_request_ids.copy(),
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
            _mock_cute_topk_without_fallbacks() as cute_topk,
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
        assert cute_topk.call_count == 2
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
            manager._confirmed_kv_lengths,
            manager._initialized_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_startup_prewarm_executes_exact_4095_cute_bucket(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        live_before = [pool.clone() for pool in pools]
        live_ptrs = [pool.untyped_storage().data_ptr() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        def fake_score(*args, **kwargs):
            seq_len = 1024 + 4095
            values = torch.zeros((2, 2 * seq_len), dtype=torch.float32)
            return values, torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32), []

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(_FixedUnionWorkspace, "compact", autospec=True) as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
            _mock_cute_topk_without_fallbacks() as cute_topk,
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=1024,
                decode_width=4095,
            )

        assert [tuple(call.args[0].shape) for call in cute_topk.call_args_list] == [
            (4, 4095),
            (1, 4095),
        ]
        assert all(call.args[3] == 2048 for call in cute_topk.call_args_list)
        compact.assert_called_once()
        assert compact.call_args.args[3] == 1024 + 4095
        workspace = next(iter(manager._fixed_union_prewarmed_workspaces.values()))
        assert workspace.keep.shape == (1024 + 2048,)
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        assert len(manager._fixed_union_prewarmed_workspaces) == 1
        assert live_ptrs == [pool.untyped_storage().data_ptr() for pool in pools]
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_startup_prewarm_executes_exact_4096_cute_bucket(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2048
        live_before = [pool.clone() for pool in pools]
        live_ptrs = [pool.untyped_storage().data_ptr() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        def fake_score(*args, **kwargs):
            seq_len = 1024 + 4096
            values = torch.zeros((2, 2 * seq_len), dtype=torch.float32)
            return values, torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32), []

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(_FixedUnionWorkspace, "compact", autospec=True) as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
            _mock_cute_topk_without_fallbacks() as cute_topk,
        ):
            manager._prewarm_fixed_union_bucket(
                layer_pools,
                dense_layers,
                storage_groups,
                num_layers=2,
                prompt_len=1024,
                decode_width=4096,
            )

        assert [tuple(call.args[0].shape) for call in cute_topk.call_args_list] == [
            (4, 4096),
            (1, 4096),
        ]
        assert all(call.args[3] == 2048 for call in cute_topk.call_args_list)
        compact.assert_called_once()
        assert compact.call_args.args[3] == 1024 + 4096
        workspace = next(iter(manager._fixed_union_prewarmed_workspaces.values()))
        assert workspace.keep.shape == (1024 + 2048,)
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        key = next(iter(manager._fixed_union_prewarm_states))
        assert key[9] == "fixed_union.cute_dsl_topk"
        assert len(manager._fixed_union_prewarmed_workspaces) == 1
        assert live_ptrs == [pool.untyped_storage().data_ptr() for pool in pools]
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_prewarm_failure_marks_bucket_and_runtime_uses_new_fixed_workspace(self):
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        live_before = [pool.clone() for pool in pools]
        request_state_before = (
            manager._gen_steps.copy(),
            manager._evicted.copy(),
            manager._confirmed_kv_lengths.copy(),
            manager._initialized_request_ids.copy(),
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
        runtime = manager._get_fixed_union_workspace(
            99,
            torch.zeros((4, 5), dtype=torch.float32),
            4,
            2,
        )
        assert runtime is not None
        assert not runtime.prewarmed
        assert request_state_before == (
            manager._gen_steps,
            manager._evicted,
            manager._confirmed_kv_lengths,
            manager._initialized_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_small_budget_prewarm_uses_the_same_fixed_route(self):
        import contextlib
        from unittest import mock

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module
        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels as kernels

        manager, pools = self._make_mocked_prewarm_manager()
        manager.top_B = 2
        live_before = [pool.clone() for pool in pools]
        layer_pools, dense_layers, storage_groups = manager._fixed_union_live_geometry(2)

        def fake_score(*args, **kwargs):
            values = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2 * 6)
            return values, torch.tensor([0, 6, 12], dtype=torch.int32), []

        with (
            mock.patch.object(kernels, "triton_tri_score_perhead", side_effect=fake_score),
            mock.patch.object(_FixedUnionWorkspace, "compact", autospec=True) as compact,
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ) as nvtx,
            _mock_cute_topk_without_fallbacks() as cute_topk,
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
        assert cute_topk.call_count == 2
        compact.assert_called_once()
        assert compact.call_args.args[3] == 6
        workspace = next(iter(manager._fixed_union_prewarmed_workspaces.values()))
        assert workspace.keep.tolist() == [0, 1, 4, 5]
        assert list(manager._fixed_union_prewarm_states.values()) == ["ready"]
        assert len(manager._fixed_union_prewarmed_workspaces) == 1
        assert (
            manager._get_fixed_union_workspace(
                99,
                torch.zeros((4, 4), dtype=torch.float32),
                2,
                2,
            )
            is not None
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
            manager._confirmed_kv_lengths.copy(),
            manager._initialized_request_ids.copy(),
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
            manager._confirmed_kv_lengths,
            manager._initialized_request_ids,
        )
        for pool, before in zip(pools, live_before):
            assert torch.equal(pool, before)

    def test_missing_or_mismatched_prewarm_bucket_creates_runtime_fixed_workspace(self):
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

        with _mock_cute_topk_without_fallbacks():
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
            _mock_cute_topk_without_fallbacks()
            if torch.device(device).type == "cpu"
            else nullcontext(),
            mock.patch.object(
                torch,
                "nonzero",
                side_effect=AssertionError("fixed selection must not call nonzero"),
            ),
            mock.patch.object(torch, "cat", wraps=torch.cat) as cat,
        ):
            first = workspace.select_segments(segments, normalize_scores=normalize_scores).clone()
            second = workspace.select_segments(segments, normalize_scores=normalize_scores).clone()

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

    def test_fixed_shape_workspace_bank_shares_scratch_and_keeps_stable_outputs(self):
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
        manager = _make_triattention()
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
        for tensor_index in range(len(base._selection_scratch_tensors())):
            assert (
                len(
                    {
                        item._selection_scratch_tensors()[tensor_index].data_ptr()
                        for item in workspaces
                    }
                )
                == 1
            )
        keep_bytes = base.keep.numel() * base.keep.element_size()
        shared_bank_bytes = base.selection_buffer_nbytes() + keep_bytes * (len(workspaces) - 1)
        legacy_bank_bytes = base.selection_buffer_nbytes() * len(workspaces)
        assert shared_bank_bytes == _FixedUnionWorkspace.planned_selection_bank_nbytes(
            base.rows,
            base.width,
            base.keep_count,
            base.prompt_len,
            base.dtype,
            base.selection_backend,
            len(workspaces),
        )
        assert shared_bank_bytes < legacy_bank_bytes
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
    def test_fixed_shape_prewarm_records_plan_and_defers_exactly_once(self, device):
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
            1: torch.arange(2 * 19, dtype=torch.float32, device=device).view(2, 19).flip(1),
        }
        score_workspace = SimpleNamespace(max_requests=3)
        manager = _make_triattention()
        manager.top_B = 8
        manager.normalize_scores = True
        manager._fixed_shape_selection_enabled = True
        manager._fixed_shape_selection_prewarm_states = {}
        manager._fixed_shape_selection_workspaces = {}
        manager._fixed_shape_selection_plans = {}
        manager._fixed_shape_selection_bank_bytes = {}
        manager._fixed_union_prewarmed_workspaces = {key: base}
        manager._fixed_shape_selection_materialization_state = "pending"

        allocation_error = AssertionError("Stage3 prewarm must not allocate a tensor")
        with (
            mock.patch.object(
                _FixedUnionWorkspace,
                "__init__",
                side_effect=AssertionError("Stage3 prewarm must not allocate a workspace"),
            ) as workspace_init,
            mock.patch.object(torch, "empty", side_effect=allocation_error),
            mock.patch.object(torch, "empty_like", side_effect=allocation_error),
            mock.patch.object(torch, "arange", side_effect=allocation_error),
            mock.patch.object(torch, "full", side_effect=allocation_error),
        ):
            manager._prewarm_fixed_shape_selection_bucket(
                key,
                base,
                score_workspace,
                scores_by_layer,
                [0, 1],
                2,
                2,
                19,
            )
        workspace_init.assert_not_called()

        assert manager._fixed_shape_selection_prewarm_states[key] == "planned"
        plan = manager._fixed_shape_selection_plans[key]
        assert not any(isinstance(field, torch.Tensor) for field in plan)
        assert key not in manager._fixed_shape_selection_workspaces
        assert key not in manager._fixed_shape_selection_bank_bytes
        assert not base._segment_buffers_prepared
        assert not hasattr(base, "input_scores")

        with (
            _mock_cute_topk_without_fallbacks() if device.type == "cpu" else nullcontext(),
            mock.patch.object(
                manager,
                "_build_fixed_shape_selection_workspaces",
                wraps=manager._build_fixed_shape_selection_workspaces,
            ) as build_bank,
            mock.patch.object(
                manager,
                "_warm_fixed_shape_selection_workspace",
                wraps=manager._warm_fixed_shape_selection_workspace,
            ) as warm_bank,
        ):
            manager._materialize_fixed_shape_selection_banks()
            manager._materialize_fixed_shape_selection_banks()

        bank = manager._fixed_shape_selection_workspaces[key]
        assert build_bank.call_count == 1
        assert warm_bank.call_count == 1
        assert len(bank) == score_workspace.max_requests
        assert bank[0] is base
        assert all(item.prewarm_attempted and item.prewarmed for item in bank)
        assert manager._fixed_shape_selection_prewarm_states[key] == "ready"
        assert manager._fixed_shape_selection_materialization_state == "done"
        keep_bytes = base.keep.numel() * base.keep.element_size()
        assert manager._fixed_shape_selection_bank_bytes[key] == (
            base.selection_buffer_nbytes() + keep_bytes * (score_workspace.max_requests - 1)
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
        failed = _make_triattention()
        failed.top_B = 8
        failed.normalize_scores = True
        failed._fixed_shape_selection_enabled = True
        failed._fixed_shape_selection_prewarm_states = {}
        failed._fixed_shape_selection_workspaces = {}
        failed._fixed_shape_selection_plans = {}
        failed._fixed_shape_selection_bank_bytes = {}
        failed._fixed_score_workspaces = {key: stage2_workspace}
        failed._fixed_score_prewarm_states = {key: "ready"}
        failed._fixed_union_prewarmed_workspaces = {key: failed_base}
        failed._fixed_shape_selection_materialization_state = "pending"

        failed._prewarm_fixed_shape_selection_bucket(
            key,
            failed_base,
            score_workspace,
            scores_by_layer,
            [0, 1],
            2,
            2,
            19,
        )
        failed._build_fixed_shape_selection_workspaces = mock.Mock(
            side_effect=torch.cuda.OutOfMemoryError("sealed Stage3 bank")
        )
        failed._materialize_fixed_shape_selection_banks()

        assert failed._fixed_shape_selection_prewarm_states[key] == "failed"
        assert key not in failed._fixed_shape_selection_workspaces
        assert key not in failed._fixed_shape_selection_bank_bytes
        assert failed._fixed_score_workspaces[key] is stage2_workspace
        assert failed._fixed_score_prewarm_states[key] == "ready"
        assert not failed_base._segment_buffers_prepared
        assert not hasattr(failed_base, "input_scores")

    def test_cute_bucket_prewarm_builds_exact_shared_workspace(self):
        from types import SimpleNamespace
        from unittest import mock

        key = ("cute-bucket",)
        scores_by_layer = {
            0: torch.arange(2 * 19, dtype=torch.float32).view(2, 19),
            1: torch.arange(2 * 19, dtype=torch.float32).view(2, 19).flip(1),
        }
        manager = _make_triattention()
        manager.top_B = 8
        manager.count_prompt_tokens = False
        manager.normalize_scores = False
        manager._fixed_shape_selection_enabled = True
        manager._fixed_shape_selection_prewarm_states = {}
        manager._fixed_shape_selection_workspaces = {}
        manager._fixed_shape_selection_plans = {}
        manager._fixed_shape_selection_bank_bytes = {}
        manager._fixed_union_prewarmed_workspaces = {}
        manager._fixed_shape_selection_materialization_state = "pending"

        original_init = _FixedUnionWorkspace.__init__
        with (
            mock.patch.object(
                _FixedUnionWorkspace,
                "__init__",
                autospec=True,
                side_effect=original_init,
            ) as workspace_init,
            _mock_cute_topk_without_fallbacks() as cute_topk,
            mock.patch.object(
                manager,
                "_warm_fixed_shape_selection_workspace",
                wraps=manager._warm_fixed_shape_selection_workspace,
            ) as warm_bank,
        ):
            manager._prewarm_fixed_shape_selection_bucket(
                key,
                None,
                SimpleNamespace(max_requests=3),
                scores_by_layer,
                [0, 1],
                2,
                2,
                19,
            )
            assert workspace_init.call_count == 0
            assert manager._fixed_shape_selection_prewarm_states[key] == "planned"
            assert key in manager._fixed_shape_selection_plans
            assert key not in manager._fixed_shape_selection_workspaces
            assert key not in manager._fixed_shape_selection_bank_bytes
            manager._materialize_fixed_shape_selection_banks()
            manager._materialize_fixed_shape_selection_banks()

        assert workspace_init.call_count == 1
        assert warm_bank.call_count == 1
        assert cute_topk.call_count == 2
        bank = manager._fixed_shape_selection_workspaces[key]
        assert manager._fixed_shape_selection_prewarm_states[key] == "ready"
        assert len(bank) == 3
        assert all(item.selection_backend == "cute_dsl_topk" for item in bank)
        assert len({item.keep.data_ptr() for item in bank}) == 3
        keep_bytes = bank[0].keep.numel() * bank[0].keep.element_size()
        assert manager._fixed_shape_selection_bank_bytes[key] == (
            bank[0].selection_buffer_nbytes() + keep_bytes * (len(bank) - 1)
        )
        for tensor_index in range(len(bank[0]._selection_scratch_tensors())):
            assert (
                len({item._selection_scratch_tensors()[tensor_index].data_ptr() for item in bank})
                == 1
            )

    def test_singleton_cute_selection_cannot_allocate_fixed_compaction(self):
        from types import SimpleNamespace

        workspace = _FixedUnionWorkspace(
            4,
            31,
            8,
            2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
        )
        workspace.selection_only = True
        pool = SimpleNamespace(is_cuda=True, device=workspace.device, ndim=5)
        page_ids = SimpleNamespace(
            ndim=1,
            device=workspace.device,
            dtype=torch.int64,
        )

        assert not workspace.can_compact(pool, page_ids, seq_len=33)
        with pytest.raises(ValueError, match="no longer matches"):
            workspace.prepare_compaction(pool, page_ids, seq_len=33)
        assert not workspace._compaction_buffers

    def test_cute_fixed_selection_matches_eager_subset_without_nonzero(self):
        width = 31
        keep_count = 8
        prompt_len = 2
        token = torch.arange(width, dtype=torch.float32)
        segments = [
            torch.stack((torch.sin(token * 0.07), torch.cos(token * 0.11))),
            torch.stack((torch.sin(token * 0.17), torch.cos(token * 0.19))),
        ]
        scores = torch.cat(segments, dim=0)
        workspace = _FixedUnionWorkspace(
            4,
            width,
            keep_count,
            prompt_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            allocate_segment_buffers=True,
            selection_backend="cute_dsl_topk",
        )

        with (
            _mock_cute_topk_without_fallbacks() as cute_topk,
            mock.patch.object(
                torch,
                "nonzero",
                side_effect=AssertionError("fixed selection must not call nonzero"),
            ),
        ):
            selected = workspace.select_segments(segments, normalize_scores=False).clone()

        assert cute_topk.call_count == 2
        combined = scores.max(dim=0).values
        row_top = _TORCH_TOPK_ORACLE(scores, keep_count, dim=1, sorted=False).indices
        union_mask = torch.zeros(width, dtype=torch.bool)
        union_mask.scatter_(0, row_top.reshape(-1), True)
        union_indices = torch.arange(width)[union_mask]
        relative = _TORCH_TOPK_ORACLE(
            combined.index_select(0, union_indices), keep_count, sorted=False
        ).indices
        expected_decode = torch.sort(union_indices.index_select(0, relative)).values
        expected = torch.cat((torch.arange(prompt_len), expected_decode + prompt_len))
        assert torch.equal(selected, expected)

    def test_gptoss_stage3_shared_bank_removes_gib_scale_lane_duplication(self):
        workspace = _FixedUnionWorkspace(
            1152,
            8192,
            4096,
            1024,
            dtype=torch.float32,
            device=torch.device("meta"),
            allocate_segment_buffers=True,
        )
        max_requests = 8
        bucket_count = 2
        slot_bytes = workspace.keep.numel() * workspace.keep.element_size()
        legacy_bytes = workspace.selection_buffer_nbytes() * max_requests * bucket_count
        shared_bytes = (
            workspace.selection_buffer_nbytes() + slot_bytes * (max_requests - 1)
        ) * bucket_count

        assert legacy_bytes > 1 << 30
        assert shared_bytes * 4 < legacy_bytes

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_fixed_shape_selection_alternates_r7_r1_r8_without_stale_slots(self, device):
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
            manager = _make_triattention()
            manager.top_B = keep_count
            manager.pin_prefill = True
            manager.count_prompt_tokens = False
            manager.eviction_mode = "union"
            manager.normalize_scores = True
            manager.kv_cache_manager = SimpleNamespace(get_buffers=lambda layer, kv_layout: pool)
            manager._dense_layers = lambda num_layers: [0, 1]
            manager._global_layer_id = lambda layer, num_layers: layer
            manager._fixed_union_enabled = False
            manager._fixed_union_active = {}
            manager._fixed_shape_selection_enabled = fixed
            manager._fixed_shape_selection_workspaces = {key: bank} if fixed else {}
            manager._fixed_shape_selection_prewarm_states = {key: "ready"} if fixed else {}
            manager._fixed_shape_selection_runtime_counts = {}
            return manager

        fixed_manager = build_manager(True)
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

        for iteration, request_count in enumerate((7, 1, 8) * 64):
            request_order = [(iteration * 5 + slot) % max_requests for slot in range(request_count)]
            page_tables = [
                ((request_id + iteration) % 23, (request_id + iteration + 7) % 23)
                for request_id in request_order
            ]
            round_starts = [5000.0 + iteration * 128.0 + request_id for request_id in request_order]
            metadata = (tuple(request_order), tuple(page_tables), tuple(round_starts))
            if previous_metadata is not None:
                assert metadata != previous_metadata
            previous_metadata = metadata

            base.input_scores.fill_(float("nan"))
            for workspace in bank:
                workspace.keep[prompt_len:].fill_(-1)
            fixed_workspaces = fixed_manager._fixed_shape_selection_for(
                score_workspace,
                request_count,
            )
            assert fixed_workspaces == bank[:request_count]

            pending_keeps = []
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

                with _mock_cute_topk_without_fallbacks() if device.type == "cpu" else nullcontext():
                    fixed_keep = fixed_manager._evict_modes(
                        request,
                        2,
                        seq_len,
                        precomputed,
                        fixed_union_workspace=fixed_workspaces[slot],
                    )
                expected_keep = _torch_union_keep(
                    torch.cat(precomputed, dim=0),
                    prompt_len=prompt_len,
                    budget=keep_count,
                )
                pending_keeps.append((fixed_keep, expected_keep))

            for fixed_keep, expected_keep in pending_keeps:
                assert torch.equal(fixed_keep, expected_keep)
            assert torch.isfinite(base.input_scores).all()

            for workspace in bank[request_count:]:
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
            "hit": 192,
            "fallback": 0,
        }
        assert (
            fixed_manager._fixed_shape_selection_for(
                score_workspace,
                max_requests + 1,
            )
            is None
        )
        assert fixed_manager._fixed_shape_selection_runtime_counts[key] == {
            "hit": 192,
            "fallback": 1,
        }
        fixed_manager._fixed_shape_selection_prewarm_states[key] = "failed"
        assert fixed_manager._fixed_shape_selection_for(score_workspace, 1) is None
        assert fixed_manager._fixed_shape_selection_runtime_counts[key] == {
            "hit": 192,
            "fallback": 2,
        }

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)],
    )
    def test_fixed_segment_ties_preserve_union_and_selected_value_multiset(self, device):
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

        with _mock_cute_topk_without_fallbacks() if device.type == "cpu" else nullcontext():
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

        with _mock_cute_topk_without_fallbacks():
            first = workspace.select(scores).clone()
            second = workspace.select(scores).clone()
        reference = self._reference(scores, keep_count)
        combined = scores.max(dim=0).values

        assert torch.equal(first, second)
        assert torch.equal(
            combined.index_select(0, first).sort().values,
            combined.index_select(0, reference).sort().values,
        )

    def test_manager_always_uses_one_fixed_workspace_for_union(self):
        from types import SimpleNamespace

        mgr = _make_triattention()
        mgr._fixed_union_enabled = True
        mgr._fixed_union_workspaces = {}
        mgr._fixed_union_active = {}
        request = SimpleNamespace(py_request_id=7)
        large = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
        keep_count = 3072
        with _mock_cute_topk_without_fallbacks():
            fixed = mgr._evict_union(
                request,
                1,
                4100,
                4,
                keep_count,
                large,
            )
        workspace = mgr._get_fixed_union_workspace(7, large, keep_count, 4)

        assert workspace is not None
        assert fixed is workspace.keep
        assert mgr._fixed_union_active[7] is workspace
        assert mgr._get_fixed_union_workspace(7, large, keep_count, 4) is workspace
        assert mgr._get_fixed_union_workspace(7, large[:, :128], keep_count, 4) is None

    def test_request_cleanup_does_not_drop_other_request_bucket(self):
        mgr = _make_triattention()
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


class TestCrossRequestFixedUnionWorkspace:
    @staticmethod
    def _stage3_plan(width, selection_backend, *, max_requests=8):
        rows = 4
        keep_count = 2048 if width >= 4095 else 8
        prompt_len = 1024 if width >= 4095 else 2
        dtype = torch.float32
        return _FixedShapeSelectionPlan(
            rows=rows,
            width=width,
            keep_count=keep_count,
            prompt_len=prompt_len,
            dtype=dtype,
            device=torch.device("cpu"),
            selection_backend=selection_backend,
            max_requests=max_requests,
            materialized_nbytes=_FixedUnionWorkspace.planned_selection_bank_nbytes(
                rows,
                width,
                keep_count,
                prompt_len,
                dtype,
                selection_backend,
                max_requests,
            ),
        )

    @staticmethod
    def _manager():
        manager = _make_triattention()
        manager.eviction_mode = "union"
        manager.normalize_scores = True
        manager._cross_request_selection_enabled = True
        manager._cross_request_selection_prewarm_states = {}
        manager._cross_request_selection_plans = {}
        manager._cross_request_selection_workspaces = {}
        manager._cross_request_selection_bank_bytes = {}
        manager._cross_request_selection_runtime_counts = {}
        manager._cross_request_selection_materialization_state = "pending"
        return manager

    @staticmethod
    def _segments(request_id, width):
        token = torch.arange(width, dtype=torch.float32)
        segments = []
        for layer in range(2):
            rows = []
            for head in range(2):
                scale = 0.071 + layer * 0.037 + head * 0.019
                rows.append(
                    torch.sin(token * scale + request_id * 0.113)
                    + token * (1e-3 + request_id * 1e-5)
                    + request_id * 3e-4
                )
            segments.append(torch.stack(rows))
        return segments

    @pytest.mark.parametrize(
        ("width", "selection_backend"),
        [(4095, "cute_dsl_topk"), (4096, "cute_dsl_topk")],
    )
    def test_prewarm_records_tensor_free_exact_scenario_b_plan(self, width, selection_backend):
        from unittest import mock

        key = ("scenario-b", width)
        stage3_plan = self._stage3_plan(width, selection_backend)
        manager = self._manager()
        allocation_error = AssertionError("Stage4 prewarm must not allocate a tensor")

        with (
            mock.patch.object(
                _BatchedFixedUnionWorkspace,
                "__init__",
                side_effect=AssertionError("Stage4 prewarm must not allocate a workspace"),
            ) as workspace_init,
            mock.patch.object(torch, "empty", side_effect=allocation_error),
            mock.patch.object(torch, "empty_like", side_effect=allocation_error),
            mock.patch.object(torch, "arange", side_effect=allocation_error),
            mock.patch.object(torch, "full", side_effect=allocation_error),
        ):
            manager._prewarm_cross_request_selection_bucket(
                key, stage3_plan, (0,), stage3_plan.rows, 1
            )

        workspace_init.assert_not_called()
        plan = manager._cross_request_selection_plans[key]
        assert manager._cross_request_selection_prewarm_states[key] == "planned"
        assert plan.width == width
        assert plan.keep_count == 2048
        assert plan.prompt_len == 1024
        assert plan.selection_backend == selection_backend
        assert selection_backend == "cute_dsl_topk"
        assert not any(isinstance(field, torch.Tensor) for field in plan)
        assert plan.materialized_nbytes == (
            _BatchedFixedUnionWorkspace.planned_selection_bank_nbytes(
                plan.rows,
                plan.width,
                plan.keep_count,
                plan.prompt_len,
                plan.dtype,
                plan.selection_backend,
                plan.max_requests,
            )
        )
        assert key not in manager._cross_request_selection_workspaces
        assert key not in manager._cross_request_selection_bank_bytes

    def test_materialization_runs_once_after_stage3_and_marks_ready(self):
        from unittest import mock

        key = ("bucket",)
        stage3_plan = self._stage3_plan(17, "cute_dsl_topk", max_requests=3)
        manager = self._manager()
        manager._fixed_shape_selection_prewarm_states = {key: "ready"}
        manager._prewarm_cross_request_selection_bucket(key, stage3_plan, (0,), stage3_plan.rows, 1)

        with (
            _mock_cute_topk_without_fallbacks(),
            mock.patch.object(
                manager,
                "_build_cross_request_selection_workspace",
                wraps=manager._build_cross_request_selection_workspace,
            ) as build_workspace,
        ):
            manager._materialize_cross_request_selection_banks()
            manager._materialize_cross_request_selection_banks()

        owner = manager._cross_request_selection_workspaces[key]
        assert build_workspace.call_count == 1
        assert manager._cross_request_selection_prewarm_states[key] == "ready"
        assert manager._cross_request_selection_materialization_state == "done"
        assert manager._cross_request_selection_bank_bytes[key] == (owner.selection_buffer_nbytes())
        assert len(owner.request_workspaces) == stage3_plan.max_requests
        assert all(item.prewarm_attempted and item.prewarmed for item in owner.request_workspaces)

    def test_indexer_request_views_preserve_row_and_shared_token_geometry(self):
        rows = 3
        width = 17
        max_requests = 4
        owner = _BatchedFixedUnionWorkspace(
            rows,
            width,
            8,
            2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=max_requests,
        )

        assert owner.row_seq_lens.shape == (max_requests * rows,)
        assert owner.union_counts.shape == (max_requests,)
        for request_index, workspace in enumerate(owner.request_workspaces):
            row_start = request_index * rows
            assert workspace.row_seq_lens.shape == (rows,)
            assert workspace.row_seq_lens.tolist() == [width] * rows
            assert workspace.row_seq_lens.data_ptr() == owner.row_seq_lens[row_start:].data_ptr()
            assert workspace.token_indices is owner.token_indices
            assert workspace.token_indices.shape == (width,)
            assert workspace.union_count.shape == (1,)
            assert (
                workspace.union_count.data_ptr()
                == owner.union_counts[request_index : request_index + 1].data_ptr()
            )

    def test_materialization_failure_is_key_sticky_and_preserves_stage3_fallback(self):
        from types import SimpleNamespace
        from unittest import mock

        key = ("bucket",)
        stage3_plan = self._stage3_plan(17, "cute_dsl_topk", max_requests=3)
        stage3_bank = tuple(SimpleNamespace(prewarmed=True) for _ in range(3))
        manager = self._manager()
        manager._fixed_shape_selection_enabled = True
        manager._fixed_shape_selection_prewarm_states = {key: "ready"}
        manager._fixed_shape_selection_workspaces = {key: stage3_bank}
        manager._fixed_shape_selection_runtime_counts = {}
        manager._prewarm_cross_request_selection_bucket(key, stage3_plan, (0,), stage3_plan.rows, 1)
        manager._build_cross_request_selection_workspace = mock.Mock(
            side_effect=torch.cuda.OutOfMemoryError("sealed Stage4 bank")
        )

        manager._materialize_cross_request_selection_banks()

        assert manager._cross_request_selection_prewarm_states[key] == "failed"
        assert key not in manager._cross_request_selection_workspaces
        assert key not in manager._cross_request_selection_bank_bytes
        assert manager._fixed_shape_selection_prewarm_states[key] == "ready"
        assert manager._fixed_shape_selection_workspaces[key] is stage3_bank
        assert (
            manager._fixed_shape_selection_for(SimpleNamespace(prewarm_key=key), 2)
            == (stage3_bank[:2])
        )

        manager._cross_request_selection_materialization_state = "pending"
        manager._materialize_cross_request_selection_banks()
        manager._prewarm_cross_request_selection_bucket(key, stage3_plan, (0,), stage3_plan.rows, 1)

        assert manager._build_cross_request_selection_workspace.call_count == 1
        assert manager._cross_request_selection_prewarm_states[key] == "failed"
        assert manager._fixed_shape_selection_prewarm_states[key] == "ready"

    def test_r1_r7_r8_match_stage3_with_stable_distinct_keep_addresses(self):
        from unittest import mock

        width = 37
        keep_count = 8
        prompt_len = 3
        max_requests = 8
        workspace = _BatchedFixedUnionWorkspace(
            4,
            width,
            keep_count,
            prompt_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=max_requests,
        )
        pointers = workspace.pointer_snapshot()
        keep_addresses = tuple(item.keep.data_ptr() for item in workspace.request_workspaces)
        assert len(set(keep_addresses)) == max_requests

        with (
            _mock_cute_topk_without_fallbacks(),
            mock.patch.object(
                torch,
                "nonzero",
                side_effect=AssertionError("cross-request selection must not call nonzero"),
            ),
        ):
            for iteration, request_count in enumerate((1, 7, 8, 1, 7, 8)):
                workspace.input_scores.fill_(float("nan"))
                workspace.keep[:, prompt_len:].fill_(-1)
                request_ids = [iteration * 11 + request for request in range(request_count)]
                segments_by_request = [
                    self._segments(request_id, width) for request_id in request_ids
                ]

                selected = workspace.select_requests(
                    segments_by_request,
                    normalize_scores=True,
                )

                assert len(selected) == request_count
                for request_index, segments in enumerate(segments_by_request):
                    reference = _FixedUnionWorkspace(
                        4,
                        width,
                        keep_count,
                        prompt_len,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                        allocate_segment_buffers=True,
                    ).select_segments(segments, normalize_scores=True)
                    assert torch.equal(selected[request_index].keep, reference)

                if request_count < max_requests:
                    assert torch.isnan(workspace.input_scores[request_count:]).all()
                    assert torch.equal(
                        workspace.keep[request_count:, prompt_len:],
                        torch.full_like(workspace.keep[request_count:, prompt_len:], -1),
                    )
                assert workspace.pointer_snapshot() == pointers
                assert (
                    tuple(item.keep.data_ptr() for item in workspace.request_workspaces)
                    == keep_addresses
                )

    def test_upper_bucket_masks_padded_scores_and_matches_per_request_reference(self):
        rows = 4
        width = 17
        keep_count = 8
        prompt_len = 3
        valid_widths = [9, 13]
        workspace = _BatchedFixedUnionWorkspace(
            rows,
            width,
            keep_count,
            prompt_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=2,
        )
        pointer_snapshot = workspace.pointer_snapshot()
        token = torch.arange(width, dtype=torch.float32)
        segments_by_request = []
        for request_index, valid_width in enumerate(valid_widths):
            scores = torch.stack(
                [
                    (row + 1.0) * token
                    + (request_index + 1.0) * torch.remainder(token * token + row, 7)
                    for row in range(rows)
                ]
            )
            scores[:, valid_width:] = 1.0e9 + token[valid_width:]
            segments_by_request.append([scores])

        workspace.stage_valid_widths_from_seq_lens(
            torch.tensor(
                [prompt_len + valid_width for valid_width in valid_widths],
                dtype=torch.int32,
            ),
            len(valid_widths),
        )
        with _mock_cute_topk_without_fallbacks():
            selected = workspace.select_requests(
                segments_by_request,
                normalize_scores=True,
            )

        for request_index, valid_width in enumerate(valid_widths):
            with _mock_cute_topk_without_fallbacks():
                reference = _FixedUnionWorkspace(
                    rows,
                    valid_width,
                    keep_count,
                    prompt_len,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    allocate_segment_buffers=True,
                ).select_segments(
                    [segments_by_request[request_index][0][:, :valid_width]],
                    normalize_scores=True,
                )
            assert torch.equal(selected[request_index].keep, reference)
            assert int(selected[request_index].keep.max()) < prompt_len + valid_width
        assert workspace.pointer_snapshot() == pointer_snapshot

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize(
        ("width", "selection_backend"),
        [(4095, "cute_dsl_topk"), (4096, "cute_dsl_topk")],
    )
    def test_exact_scenario_b_cuda_r1_r7_r8_matches_stage3(self, width, selection_backend):
        rows = 4
        keep_count = 2048
        prompt_len = 1024
        max_requests = 8
        dtype = torch.float32
        device = torch.device("cuda")
        workspace = _BatchedFixedUnionWorkspace(
            rows,
            width,
            keep_count,
            prompt_len,
            dtype=dtype,
            device=device,
            selection_backend=selection_backend,
            max_requests=max_requests,
        )
        planned_bytes = _BatchedFixedUnionWorkspace.planned_selection_bank_nbytes(
            rows,
            width,
            keep_count,
            prompt_len,
            dtype,
            selection_backend,
            max_requests,
        )
        assert workspace.selection_buffer_nbytes() == planned_bytes
        pointers = workspace.pointer_snapshot()
        keep_addresses = tuple(item.keep.data_ptr() for item in workspace.request_workspaces)
        assert len(set(keep_addresses)) == max_requests

        for iteration, request_count in enumerate((1, 7, 8)):
            segments_by_request = [
                [segment.to(device) for segment in self._segments(iteration * 11 + request, width)]
                for request in range(request_count)
            ]
            selected = workspace.select_requests(
                segments_by_request,
                normalize_scores=True,
            )
            for request_index, segments in enumerate(segments_by_request):
                reference_workspace = _FixedUnionWorkspace(
                    rows,
                    width,
                    keep_count,
                    prompt_len,
                    dtype=dtype,
                    device=device,
                    allocate_segment_buffers=True,
                    selection_backend=selection_backend,
                )
                reference = reference_workspace.select_segments(
                    segments,
                    normalize_scores=True,
                )
                assert torch.equal(selected[request_index].keep, reference)

            assert workspace.pointer_snapshot() == pointers
            assert (
                tuple(item.keep.data_ptr() for item in workspace.request_workspaces)
                == keep_addresses
            )

    def test_runtime_dispatch_requires_exact_ready_bucket(self):
        from types import SimpleNamespace

        key = ("bucket",)
        workspace = _BatchedFixedUnionWorkspace(
            4,
            17,
            8,
            2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=3,
        )
        manager = self._manager()
        manager._fixed_shape_selection_prewarm_states = {key: "ready"}
        manager._cross_request_selection_prewarm_states = {key: "ready"}
        manager._cross_request_selection_workspaces = {key: workspace}

        assert manager._cross_request_selection_for(SimpleNamespace(prewarm_key=key), 3) is (
            workspace
        )
        assert (
            manager._cross_request_selection_for(
                SimpleNamespace(prewarm_key=("other-bucket",)),
                1,
            )
            is None
        )
        assert manager._cross_request_selection_for(SimpleNamespace(prewarm_key=key), 4) is None
        manager._cross_request_selection_prewarm_states[key] = "failed"
        assert manager._cross_request_selection_for(SimpleNamespace(prewarm_key=key), 1) is None

        assert manager._cross_request_selection_runtime_counts[key] == {
            "hit": 1,
            "rejected": 2,
        }
        assert manager._cross_request_selection_runtime_counts[("other-bucket",)] == {
            "hit": 0,
            "rejected": 1,
        }

    def test_selection_runs_once_inside_the_request_batched_graph(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        requests = [_make_request(request_id, py_prompt_len=2) for request_id in (7, 8)]
        key = ("cross-request",)
        selection = SimpleNamespace(
            eviction_mode="union",
            dense_layers=(0, 1),
            num_query_heads=1,
            num_kv_heads=1,
            prompt_len=2,
            width=6,
            keep_count=4,
            max_requests=2,
            selection_backend="cute_dsl_topk",
            stage_valid_widths_from_seq_lens=mock.Mock(),
            select_requests=mock.Mock(),
        )
        stream = SimpleNamespace(
            device=torch.device("cpu"),
            cuda_stream=9,
        )
        score_output = torch.arange(2 * 2 * 8, dtype=torch.float32).view(1, -1)
        score_group = SimpleNamespace(launch=mock.Mock(return_value=(score_output, None)))
        score_workspace = SimpleNamespace(
            prewarm_key=key,
            bucket_seq_len=8,
            prompt_len=2,
            device=torch.device("cpu"),
            stream=stream,
            fused_group=score_group,
            dense_layer_order=[0, 1],
            round_starts_device=torch.tensor([8.0, 8.0]),
            valid_seq_lens_device=torch.tensor([8, 8], dtype=torch.int32),
            mean_cos=torch.empty(0),
            mean_sin=torch.empty(0),
            prepare_phase=mock.Mock(),
        )
        manager = _make_triattention()
        manager._evicted = {}
        manager._confirmed_kv_lengths = {7: 8, 8: 8}
        manager.top_B = 4
        manager.eviction_mode = "union"
        manager.normalize_scores = False
        manager.score_aggregation = "mean"
        manager._evict_modes = mock.Mock(
            side_effect=AssertionError("the graph must not select each request separately")
        )
        manager._standalone_graph_bucket_for = mock.Mock(return_value=key)
        graph_workspace = SimpleNamespace(
            pointer_fingerprint=mock.Mock(return_value=("stable",)),
            launch=mock.Mock(),
        )
        manager._standalone_graph_workspace_for = mock.Mock(return_value=graph_workspace)

        def execute_graph(**kwargs):
            kwargs["capture_body"]()
            return "capture"

        cache = SimpleNamespace(
            is_disabled=mock.Mock(return_value=False),
            classify=mock.Mock(return_value="capture"),
            execute=mock.Mock(side_effect=execute_graph),
        )
        manager._standalone_graph_cache_for = mock.Mock(return_value=cache)
        prepared = [
            {
                "request": request,
                "request_id": request.py_request_id,
                "seq_len": 8,
                "expected_keep_count": 6,
                "protected_tail": 0,
            }
            for request in requests
        ]
        fixed_views = torch.arange(1 * 2 * 2 * 8, dtype=torch.float32).view(1, 2, 2, 8)

        with (
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
        ):
            targets = manager._try_standalone_cuda_graph(
                prepared=prepared,
                layer_pools=[torch.empty(0), torch.empty(0)],
                dense_layers=[0, 1],
                dense_groups=[[0], [1]],
                swa_layers=[],
                swa_window=None,
                layer_group_representative={0: 0, 1: 1},
                global_layers=[0, 1],
                score_workspace=score_workspace,
                selection_workspace=selection,
                fixed_perhead_segment_views=lambda *_args: fixed_views,
            )

        assert targets == [(7, 6), (8, 6)]
        assert manager._evicted == {7: 2, 8: 2}
        assert manager._confirmed_kv_lengths == {7: 6, 8: 6}
        selection.stage_valid_widths_from_seq_lens.assert_called_once_with(
            score_workspace.valid_seq_lens_device,
            2,
        )
        selection.select_requests.assert_called_once()
        score_workspace.prepare_phase.assert_called_once_with(2)
        manager._evict_modes.assert_not_called()
        graph_workspace.launch.assert_called_once_with()
        cache.execute.assert_called_once()
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 1,
            "attempt_requests": 2,
            "success": 1,
            "success_requests": 2,
        }


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
        from unittest import mock

        mgr = _make_triattention()
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1, 2, 3])
        config = _make_hf_config(
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
        from unittest import mock

        mgr = _make_triattention()
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 127
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1])
        config = _make_hf_config(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=128,
        )

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            pytest.raises(ValueError, match="decode budget top_B=127"),
        ):
            mgr._attention_layer_partition(2)

    def test_layer_partition_uses_pp_local_to_global_mapping(self):
        from unittest import mock

        mgr = _make_triattention()
        mgr.skip_swa = True
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[1, 2])
        config = _make_hf_config(
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
        from unittest import mock

        mgr = _make_triattention()
        mgr.skip_swa = True
        mgr.model_path = "/models/ambiguous"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1])
        config = _make_hf_config(layer_types=None, sliding_window=128)

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            pytest.raises(ValueError, match="cannot classify"),
        ):
            mgr._attention_layer_partition(2)

    def test_layer_partition_honors_explicit_disabled_sliding_window(self):
        from unittest import mock

        mgr = _make_triattention()
        mgr.skip_swa = True
        mgr.model_path = "/models/qwen3"
        mgr.top_B = 128
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1])
        config = _make_hf_config(
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
        mgr = _make_triattention()
        mgr.kv_cache_manager = SimpleNamespace(
            pp_layers=[9, 10],
            num_layers=36,
            get_batch_cache_indices=get_batch,
        )
        request = SimpleNamespace(py_request_id=7)

        assert mgr._resolve_page_ids(request, 0, page_count=2) == [10, 11]
        get_batch.assert_called_once_with([7], 9, num_blocks_per_seq=[2])

    def test_page_lookup_rejects_bad_page_inside_required_prefix(self):
        from types import SimpleNamespace
        from unittest import mock

        get_batch = mock.Mock(return_value=[[10, -1]])
        mgr = _make_triattention()
        mgr.kv_cache_manager = SimpleNamespace(
            pp_layers=[9],
            num_layers=36,
            get_batch_cache_indices=get_batch,
        )
        request = SimpleNamespace(py_request_id=7)

        assert mgr._resolve_page_ids(request, 0, page_count=2) is None
        get_batch.assert_called_once_with([7], 9, num_blocks_per_seq=[2])

    def test_page_lookup_failure_is_not_hidden(self):
        from types import SimpleNamespace
        from unittest import mock

        get_batch = mock.Mock(side_effect=KeyError("missing pool"))
        mgr = _make_triattention()
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

    def test_triattention_algorithm_normalizes_base_config(self):
        from unittest.mock import patch

        cfg = KvCacheCompressionConfig(algorithm="triattention")
        kv_cache_manager = _make_fake_v2()
        expected = object()
        with patch(
            "tensorrt_llm._torch.kv_cache_compression.triattention.TriAttention",
            return_value=expected,
        ) as triattention_cls:
            manager = create_kv_cache_compression_manager(cfg, kv_cache_manager)

        assert manager is expected
        triattention_cls.assert_called_once_with(
            kv_cache_manager,
            draft_kv_cache_manager=None,
            spec_config=None,
            top_B=2048,
            beta=128,
            model_path=None,
            calibration_path=None,
            window_size=128,
            eviction_mode="union",
            normalize_scores=True,
            pin_prefill=True,
            skip_swa=True,
            count_prompt_tokens=False,
        )

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

    def test_factory_rejects_non_v2_manager(self):
        target = SimpleNamespace()
        cfg = TriAttentionKvCacheCompressionConfig(top_B=512, beta=128)

        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            create_kv_cache_compression_manager(cfg, kv_cache_manager=target)

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
