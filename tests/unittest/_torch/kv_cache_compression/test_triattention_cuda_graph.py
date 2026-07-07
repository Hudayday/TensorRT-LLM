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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.triattention.cuda_graph import (
    FixedBatchedCompactionWorkspace,
    StandaloneEvictionGraphCache,
    _FixedCompactLaunch,
    _tensor_fingerprint,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttention,
    _BatchedFixedUnionWorkspace,
    _FixedScoreMetadataWorkspace,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    fixed_perhead_segment_views,
    triton_tri_compact,
)

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _FakeEvent:
    def __init__(self, complete=True):
        self.complete = complete

    def query(self):
        return self.complete


class _FakeGraph:
    def __init__(self, *, fail_replay=False):
        self.fail_replay = fail_replay
        self.replays = 0
        self.resets = 0

    def replay(self):
        self.replays += 1
        if self.fail_replay:
            raise RuntimeError("replay failed")

    def reset(self):
        self.resets += 1


class _FakeStorage:
    def __init__(self, *, cdata, data_ptr=4096, nbytes=64):
        self._cdata = cdata
        self._data_ptr = data_ptr
        self._nbytes = nbytes

    def data_ptr(self):
        return self._data_ptr

    def nbytes(self):
        return self._nbytes


class _FakeTensor:
    def __init__(
        self,
        *,
        storage_cdata,
        storage_data_ptr=4096,
        storage_nbytes=64,
        data_ptr=4096,
        storage_offset=0,
        shape=(8,),
        stride=(1,),
    ):
        self._storage = _FakeStorage(
            cdata=storage_cdata,
            data_ptr=storage_data_ptr,
            nbytes=storage_nbytes,
        )
        self._data_ptr = data_ptr
        self._storage_offset = storage_offset
        self.shape = shape
        self._stride = stride
        self.dtype = torch.float16
        self.device = torch.device("cuda")

    def untyped_storage(self):
        return self._storage

    def data_ptr(self):
        return self._data_ptr

    def storage_offset(self):
        return self._storage_offset

    def stride(self):
        return self._stride


def _cache_workspace(nbytes=64):
    return SimpleNamespace(nbytes=nbytes, device=torch.device("cpu"))


def _make_triattention():
    """Construct a fully initialized manager for graph-orchestration tests."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    kv_cache_manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    kv_cache_manager.enable_block_reuse = False
    kv_cache_manager.mapping = SimpleNamespace(enable_attention_dp=False)
    kv_cache_manager.is_disagg = False
    kv_cache_manager.max_beam_width = 1
    kv_cache_manager.num_extra_kv_tokens = 0
    kv_cache_manager.max_total_draft_tokens = 0
    kv_cache_manager._kv_reserve_draft_tokens = 0
    kv_cache_manager.max_attention_window_vec = []
    kv_cache_manager.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    kv_cache_manager.pp_layers = []
    kv_cache_manager.layer_offsets = {}
    kv_cache_manager.layer_to_pool_mapping_dict = {}
    return TriAttention(kv_cache_manager, top_B=8, skip_swa=False)


class TestStandaloneEvictionGraphCache:
    def test_capture_body_can_update_inference_tensors(self):
        with torch.inference_mode():
            fixed_buffer = torch.zeros(1)
        assert fixed_buffer.is_inference()
        assert not torch.is_inference_mode_enabled()
        with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
            fixed_buffer.fill_(1)

        current_stream = mock.Mock()
        capture_stream = mock.Mock()
        graph = mock.Mock()
        workspace = SimpleNamespace(device=torch.device("cpu"))

        class GraphContext:
            def __enter__(self):
                assert torch.is_inference_mode_enabled()

            def __exit__(self, exc_type, exc_value, traceback):
                assert torch.is_inference_mode_enabled()
                fixed_buffer.fill_(2)

        def capture_body():
            assert torch.is_inference_mode_enabled()
            fixed_buffer.fill_(1)

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
            mock.patch.object(torch.cuda, "Stream", return_value=capture_stream),
            mock.patch.object(torch.cuda, "CUDAGraph", return_value=graph),
            mock.patch.object(torch.cuda, "memory_allocated", side_effect=(0, 0)),
            mock.patch.object(torch.cuda, "stream", return_value=nullcontext()),
            mock.patch.object(torch.cuda, "graph", return_value=GraphContext()),
        ):
            captured_graph, captured_stream, graph_bytes = (
                StandaloneEvictionGraphCache._capture_graph(workspace, capture_body)
            )

        assert captured_graph is graph
        assert captured_stream is capture_stream
        assert graph_bytes == 0
        assert fixed_buffer.item() == 2
        assert not torch.is_inference_mode_enabled()

    @CUDA_REQUIRED
    def test_capture_graph_can_update_cuda_inference_tensor(self):
        with torch.inference_mode():
            fixed_buffer = torch.zeros(1, device="cuda")
        assert fixed_buffer.is_inference()
        assert not torch.is_inference_mode_enabled()

        workspace = SimpleNamespace(device=fixed_buffer.device)
        graph, _capture_stream, _graph_bytes = StandaloneEvictionGraphCache._capture_graph(
            workspace, lambda: fixed_buffer.fill_(1)
        )
        graph.replay()
        torch.cuda.synchronize(fixed_buffer.device)

        assert fixed_buffer.item() == 1
        assert not torch.is_inference_mode_enabled()

    def test_capture_then_hit_replays_and_records_each_use(self):
        cache = StandaloneEvictionGraphCache(max_entries=2, max_bytes=1024)
        graph = _FakeGraph()
        events = [_FakeEvent(), _FakeEvent()]
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 0))
        cache._record_last_use = mock.Mock(side_effect=events)
        body = mock.Mock()
        workspace = _cache_workspace()

        assert (
            cache.execute(
                key=("bucket",),
                request_count=4,
                fingerprint=("pointers",),
                workspace=workspace,
                capture_body=body,
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("bucket",),
                request_count=4,
                fingerprint=("pointers",),
                workspace=workspace,
                capture_body=body,
            )
            == "replay"
        )

        cache._capture_graph.assert_called_once_with(workspace, body)
        assert graph.replays == 2
        assert cache.counts["capture"] == 1
        assert cache.counts["replay"] == 2
        assert cache.counts["launch"] == 2
        assert cache.counts["cache_hit"] == 1
        assert cache.counts["covered_requests"] == 8
        assert cache.snapshot()["buckets"] == [
            {
                "key": ("bucket",),
                "request_count": 4,
                "attempt": 2,
                "attempted_requests": 8,
                "capture": 1,
                "launch": 2,
                "cache_hit": 1,
                "covered_requests": 8,
                "fallback": 0,
                "invalidated": 0,
                "failure": 0,
                "capture_failure": 0,
                "replay_failure": 0,
            }
        ]

    def test_capture_failure_performs_one_caller_fallback(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        cache._capture_graph = mock.Mock(side_effect=RuntimeError("capture failed"))
        eager = mock.Mock()
        outcome = cache.execute(
            key=("bucket",),
            request_count=3,
            fingerprint=("pointers",),
            workspace=_cache_workspace(),
            capture_body=mock.Mock(),
        )
        if outcome == "fallback":
            eager()

        assert outcome == "fallback"
        eager.assert_called_once_with()
        assert cache.counts["capture_failure"] == 1
        assert cache.counts["replay"] == 0
        assert cache.snapshot()["last_error"] == {
            "phase": "capture",
            "type": "RuntimeError",
            "message": "capture failed",
        }
        assert (
            cache.execute(
                key=("bucket",),
                request_count=3,
                fingerprint=("new-pointers",),
                workspace=_cache_workspace(),
                capture_body=mock.Mock(),
            )
            == "fallback"
        )
        cache._capture_graph.assert_called_once()
        assert cache.snapshot()["disabled_buckets"] == 1
        assert cache.snapshot()["failure"] == 1
        assert cache.snapshot()["buckets"][0]["fallback"] == 2
        assert cache.snapshot()["buckets"][0]["covered_requests"] == 0

    def test_replay_failure_is_poisoned_without_eager_fallback(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        graph = _FakeGraph(fail_replay=True)
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 0))
        cache._record_last_use = mock.Mock()
        eager = mock.Mock()

        with pytest.raises(RuntimeError, match="replay failed"):
            outcome = cache.execute(
                key=("bucket",),
                request_count=2,
                fingerprint=("pointers",),
                workspace=_cache_workspace(),
                capture_body=mock.Mock(),
            )
            if outcome == "fallback":
                eager()

        eager.assert_not_called()
        assert cache.counts["replay_failure"] == 1
        assert cache.counts["failure"] == 1
        assert cache.counts["covered_requests"] == 0
        assert cache.counts["fallback"] == 0

    def test_in_flight_entry_cannot_be_freed_to_make_room(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=128)
        first_graph = _FakeGraph()
        second_graph = _FakeGraph()
        event = _FakeEvent(complete=False)
        cache._capture_graph = mock.Mock(
            side_effect=[(first_graph, object(), 0), (second_graph, object(), 0)]
        )
        cache._record_last_use = mock.Mock(side_effect=[event, _FakeEvent()])
        workspace = _cache_workspace()

        assert (
            cache.execute(
                key=("first",),
                request_count=1,
                fingerprint=("one",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("second",),
                request_count=1,
                fingerprint=("two",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "fallback"
        )
        assert first_graph.resets == 0

        event.complete = True
        assert (
            cache.execute(
                key=("second",),
                request_count=1,
                fingerprint=("two",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "capture"
        )
        assert first_graph.resets == 1

    def test_pointer_change_invalidates_completed_entry(self):
        cache = StandaloneEvictionGraphCache(max_entries=2, max_bytes=1024)
        first_graph = _FakeGraph()
        second_graph = _FakeGraph()
        cache._capture_graph = mock.Mock(
            side_effect=[(first_graph, object(), 0), (second_graph, object(), 0)]
        )
        cache._record_last_use = mock.Mock(side_effect=[_FakeEvent(), _FakeEvent()])
        workspace = _cache_workspace()

        cache.execute(
            key=("bucket",),
            request_count=7,
            fingerprint=("old",),
            workspace=workspace,
            capture_body=mock.Mock(),
        )
        cache.execute(
            key=("bucket",),
            request_count=7,
            fingerprint=("new",),
            workspace=workspace,
            capture_body=mock.Mock(),
        )

        assert first_graph.resets == 1
        assert second_graph.replays == 1
        assert cache.counts["invalidated"] == 1
        assert cache.counts["capture"] == 2
        assert cache.counts["launch"] == 2
        assert cache.counts["covered_requests"] == 14
        assert cache.snapshot()["buckets"][0]["invalidated"] == 1

    def test_two_exact_shapes_capture_once_then_reuse_first(self):
        cache = StandaloneEvictionGraphCache(max_entries=2, max_bytes=1024)
        first_graph = _FakeGraph()
        second_graph = _FakeGraph()
        cache._capture_graph = mock.Mock(
            side_effect=[(first_graph, object(), 0), (second_graph, object(), 0)]
        )
        cache._record_last_use = mock.Mock(side_effect=[_FakeEvent(), _FakeEvent(), _FakeEvent()])
        workspace = _cache_workspace()

        assert (
            cache.execute(
                key=("shape-a", 4),
                request_count=4,
                fingerprint=("a",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("shape-b", 2),
                request_count=2,
                fingerprint=("b",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("shape-a", 4),
                request_count=4,
                fingerprint=("a",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "replay"
        )

        assert cache.counts["capture"] == 2
        assert cache.counts["launch"] == 3
        assert cache.counts["cache_hit"] == 1
        assert cache.counts["covered_requests"] == 10
        assert [bucket["capture"] for bucket in cache.snapshot()["buckets"]] == [1, 1]
        assert [bucket["cache_hit"] for bucket in cache.snapshot()["buckets"]] == [1, 0]

    def test_key_cannot_change_request_count_semantics(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        cache.record_fallback(key=("bucket",), request_count=1)

        with pytest.raises(ValueError, match="request-count semantics"):
            cache.record_fallback(key=("bucket",), request_count=2)

    def test_capture_allocation_over_byte_cap_is_reset_before_replay(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=128)
        graph = _FakeGraph()
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 96))

        outcome = cache.execute(
            key=("bucket",),
            request_count=1,
            fingerprint=("pointers",),
            workspace=_cache_workspace(nbytes=64),
            capture_body=mock.Mock(),
        )

        assert outcome == "fallback"
        assert graph.replays == 0
        assert graph.resets == 1
        assert cache.snapshot()["active_entries"] == 0


class TestStandaloneGraphBuckets:
    @staticmethod
    def _mark_ready(manager, score, selection):
        key = score.prewarm_key
        manager._fixed_score_prewarm_states = {key: "ready"}
        manager._fixed_score_workspaces = {key: score}
        manager._cross_request_selection_prewarm_states = {key: "ready"}
        manager._cross_request_selection_workspaces = {key: selection}

    @pytest.mark.parametrize(
        "prompt_len,width,budget,backend,request_count",
        [
            (0, 2175, 2048, "indexer_topk", 1),
            (256, 3072, 2048, "indexer_topk", 2),
            (1024, 4095, 2048, "indexer_topk", 4),
            (1024, 4096, 2048, "torch_topk", 7),
            (1536, 4223, 4096, "torch_topk", 8),
            (1024, 4224, 4096, "torch_topk", 8),
            (1024, 5119, 4096, "torch_topk", 16),
            (1024, 8191, 4096, "torch_topk", 1),
            (2048, 8192, 4096, "torch_topk", 31),
            (1024, 9216, 8192, "torch_topk", 32),
        ],
    )
    def test_ready_dynamic_exact_bucket_is_accepted(
        self, prompt_len, width, budget, backend, request_count
    ):
        manager = _make_triattention()
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            {
                "seq_len": prompt_len + width,
                "request": SimpleNamespace(py_prompt_len=prompt_len),
                "expected_keep_count": prompt_len + budget,
                "protected_tail": 0,
            }
            for _ in range(request_count)
        ]
        score = SimpleNamespace(
            prewarm_key=("exact", prompt_len, width, budget, backend),
            max_requests=32,
        )
        selection = SimpleNamespace(
            selection_backend=backend,
            max_requests=32,
            prompt_len=prompt_len,
            width=width,
            keep_count=budget,
        )
        self._mark_ready(manager, score, selection)

        key = manager._standalone_graph_bucket_for(prepared, score, selection)

        assert key is not None
        assert key[1] == score.prewarm_key
        assert key[2:7] == (
            request_count,
            prompt_len + width,
            prompt_len,
            budget,
            backend,
        )

    def test_protected_tail_does_not_change_stable_graph_bucket(self):
        manager = _make_triattention()
        manager.top_B = 2048
        manager._standalone_cuda_graph_enabled = True
        score = SimpleNamespace(prewarm_key=(3072, 2048), max_requests=32)
        selection = SimpleNamespace(
            selection_backend="indexer_topk",
            max_requests=32,
            prompt_len=256,
            width=3072,
            keep_count=2048,
        )
        self._mark_ready(manager, score, selection)

        def prepared(protected_tail):
            return [
                {
                    "seq_len": 3328,
                    "request": SimpleNamespace(py_prompt_len=256),
                    "expected_keep_count": 2304,
                    "protected_tail": protected_tail,
                }
            ]

        assert manager._standalone_graph_bucket_for(
            prepared(0), score, selection
        ) == manager._standalone_graph_bucket_for(prepared(2), score, selection)

    @pytest.mark.parametrize(
        "request_count,prompt_len,width,budget,backend,ready",
        [
            (0, 1024, 4095, 2048, "indexer_topk", True),
            (33, 1024, 4095, 2048, "indexer_topk", True),
            (1, 1024, 4095, 2048, "torch_topk", True),
            (1, 1024, 4095, 2048, "indexer_topk", False),
        ],
    )
    def test_unready_capacity_or_backend_mismatch_falls_back(
        self, request_count, prompt_len, width, budget, backend, ready
    ):
        manager = _make_triattention()
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            {
                "seq_len": prompt_len + width,
                "request": SimpleNamespace(py_prompt_len=prompt_len),
                "expected_keep_count": prompt_len + budget,
                "protected_tail": 0,
            }
            for _ in range(request_count)
        ]
        score = SimpleNamespace(prewarm_key=(width, budget), max_requests=32)
        selection = SimpleNamespace(
            selection_backend=backend,
            max_requests=32,
            prompt_len=prompt_len,
            width=width,
            keep_count=budget,
        )
        if ready:
            self._mark_ready(manager, score, selection)

        assert manager._standalone_graph_bucket_for(prepared, score, selection) is None

    def test_mixed_geometry_or_stale_workspace_falls_back(self):
        manager = _make_triattention()
        manager.top_B = 2048
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            {
                "seq_len": 5119,
                "request": SimpleNamespace(py_prompt_len=1024),
                "expected_keep_count": 3072,
                "protected_tail": 0,
            },
            {
                "seq_len": 5375,
                "request": SimpleNamespace(py_prompt_len=1280),
                "expected_keep_count": 3328,
                "protected_tail": 0,
            },
        ]
        score = SimpleNamespace(prewarm_key=("mixed",), max_requests=32)
        selection = SimpleNamespace(
            selection_backend="indexer_topk",
            max_requests=32,
            prompt_len=1024,
            width=4095,
            keep_count=2048,
        )
        self._mark_ready(manager, score, selection)

        assert manager._standalone_graph_bucket_for(prepared, score, selection) is None
        manager._cross_request_selection_workspaces[score.prewarm_key] = SimpleNamespace()
        prepared[1] = dict(prepared[0])
        assert manager._standalone_graph_bucket_for(prepared, score, selection) is None

    def test_stats_are_zero_filled_before_first_eviction(self, monkeypatch):
        monkeypatch.setenv("TRIATTN_CUDA_GRAPH_MAX_ENTRIES", "2")
        monkeypatch.setenv("TRIATTN_CUDA_GRAPH_MAX_BYTES", "1024")
        manager = _make_triattention()
        manager._standalone_cuda_graph_enabled = True
        manager._standalone_graph_cache = None
        manager._standalone_graph_runtime_counts = {}

        stats = manager._standalone_cuda_graph_stats()

        assert stats["enabled"]
        assert stats["attempt"] == 0
        assert stats["capture"] == 0
        assert stats["launch"] == 0
        assert stats["cache_hit"] == 0
        assert stats["covered_requests"] == 0
        assert stats["buckets"] == []
        assert stats["max_entries"] == 2
        assert stats["max_bytes"] == 1024
        assert stats["runtime"] == {}

    def test_unready_eviction_records_admission_rejection_without_cache(self):
        manager = _make_triattention()
        manager._standalone_cuda_graph_enabled = True
        manager._standalone_graph_runtime_counts = {}
        prepared = [
            {
                "seq_len": 5119,
                "request": SimpleNamespace(py_prompt_len=1024),
                "expected_keep_count": 3072,
                "protected_tail": 0,
            }
        ]

        result = manager._try_standalone_cuda_graph(
            prepared=prepared,
            layer_pools=[torch.empty(1)],
            dense_layers=[0],
            dense_groups=[[0]],
            swa_layers=[],
            swa_window=None,
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=None,
            selection_workspace=None,
            fixed_perhead_segment_views=mock.Mock(),
        )

        assert result is None
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 1,
            "attempt_requests": 1,
            "admission_rejected": 1,
            "admission_rejected_requests": 1,
        }

    @pytest.mark.parametrize("protected_tail", [0, 2])
    def test_capture_body_covers_phase_score_select_compact_then_publishes(self, protected_tail):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager = _make_triattention()
        manager.top_B = 2048
        manager.normalize_scores = True
        manager.score_aggregation = "mean"
        manager._standalone_cuda_graph_enabled = True
        manager._evicted = {}
        manager._pre_forward_kv_lengths = {7: 3328}
        prepared = [
            {
                "request": SimpleNamespace(py_prompt_len=256),
                "request_id": 7,
                "seq_len": 3328,
                "expected_keep_count": 2304,
                "protected_tail": protected_tail,
            }
        ]
        score_output = torch.zeros(1, 3328)
        group = SimpleNamespace(launch=mock.Mock(return_value=(score_output, None)))
        score = SimpleNamespace(
            prewarm_key=("dynamic-indexer",),
            max_requests=32,
            prepare_phase=mock.Mock(),
            groups={0: group},
            round_starts_device=torch.zeros(8),
            mean_cos=torch.zeros(8, 1),
            mean_sin=torch.zeros(8, 1),
        )
        selection = SimpleNamespace(
            selection_backend="indexer_topk",
            max_requests=32,
            prompt_len=256,
            width=3072,
            keep_count=2048,
            select_requests=mock.Mock(),
        )
        self._mark_ready(manager, score, selection)
        workspace = SimpleNamespace(
            device=torch.device("cpu"),
            pointer_fingerprint=mock.Mock(return_value=("fingerprint",)),
            launch=mock.Mock(),
        )

        class ExecutingCache:
            @staticmethod
            def is_disabled(_key):
                return False

            @staticmethod
            def classify(_key, _fingerprint):
                return "capture"

            def execute(self, **kwargs):
                kwargs["capture_body"]()
                return "capture"

            @staticmethod
            def snapshot():
                return {"capture": 1, "replay": 1}

        manager._standalone_graph_workspace_for = mock.Mock(return_value=workspace)
        manager._standalone_graph_cache_for = mock.Mock(return_value=ExecutingCache())
        manager._copy_protected_suffixes = mock.Mock()
        views = torch.zeros(1, 1, 1, 3328)
        fixed_views = mock.Mock(return_value=views)
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score.device = torch.device("cpu")
        score.stream = stream

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
        ):
            targets = manager._try_standalone_cuda_graph(
                prepared=prepared,
                layer_pools=[torch.empty(1)],
                dense_layers=[0],
                dense_groups=[[0]],
                swa_layers=[],
                swa_window=None,
                layer_group_representative={0: 0},
                global_layers=[0],
                score_workspace=score,
                selection_workspace=selection,
                fixed_perhead_segment_views=fixed_views,
            )

        assert targets == [(7, 2304 + protected_tail)]
        score.prepare_phase.assert_called_once_with(1)
        group.launch.assert_called_once()
        selected_segments = selection.select_requests.call_args.args[0]
        assert selected_segments[0][0].shape == (1, 3072)
        workspace.launch.assert_called_once_with()
        assert manager._evicted == {7: 1024}
        assert manager._pre_forward_kv_lengths == {7: 2304 + protected_tail}
        if protected_tail:
            manager._copy_protected_suffixes.assert_called_once()
        else:
            manager._copy_protected_suffixes.assert_not_called()
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 1,
            "attempt_requests": 1,
            "success": 1,
            "success_requests": 1,
        }

    def test_graph_fallback_does_not_publish_or_execute_eager_inside_helper(self):
        manager = _make_triattention()
        manager.top_B = 4096
        manager._standalone_cuda_graph_enabled = True
        manager._evicted = {}
        manager._pre_forward_kv_lengths = {7: 9215}
        prepared = [
            {
                "request": SimpleNamespace(py_prompt_len=1024),
                "request_id": 7,
                "seq_len": 9215,
                "expected_keep_count": 5120,
                "protected_tail": 0,
            }
        ]
        score = SimpleNamespace(prewarm_key=("formal-a-first",), max_requests=8)
        selection = SimpleNamespace(
            selection_backend="torch_topk",
            max_requests=8,
            prompt_len=1024,
            width=8191,
            keep_count=4096,
        )
        self._mark_ready(manager, score, selection)
        workspace = SimpleNamespace(
            device=torch.device("cpu"),
            pointer_fingerprint=mock.Mock(return_value=("fingerprint",)),
        )
        cache = SimpleNamespace(
            is_disabled=mock.Mock(return_value=False),
            classify=mock.Mock(return_value="capture"),
            execute=mock.Mock(return_value="fallback"),
            snapshot=mock.Mock(return_value={"fallback": 1}),
        )
        manager._standalone_graph_workspace_for = mock.Mock(return_value=workspace)
        manager._standalone_graph_cache_for = mock.Mock(return_value=cache)
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score.device = torch.device("cpu")
        score.stream = stream

        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            result = manager._try_standalone_cuda_graph(
                prepared=prepared,
                layer_pools=[torch.empty(1)],
                dense_layers=[0],
                dense_groups=[[0]],
                swa_layers=[],
                swa_window=None,
                layer_group_representative={0: 0},
                global_layers=[0],
                score_workspace=score,
                selection_workspace=selection,
                fixed_perhead_segment_views=mock.Mock(),
            )

        assert result is None
        assert manager._evicted == {}
        assert manager._pre_forward_kv_lengths == {7: 9215}

    def test_capture_failure_disables_semantic_bucket_before_new_arena(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager = _make_triattention()
        manager.top_B = 4096
        manager._standalone_cuda_graph_enabled = True
        manager._evicted = {}
        manager._pre_forward_kv_lengths = {7: 9215}
        prepared = [
            {
                "request": SimpleNamespace(py_prompt_len=1024),
                "request_id": 7,
                "seq_len": 9215,
                "expected_keep_count": 5120,
                "protected_tail": 0,
            }
        ]
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score = SimpleNamespace(
            prewarm_key=("formal-a-first",),
            max_requests=8,
            device=torch.device("cpu"),
            stream=stream,
        )
        selection = SimpleNamespace(
            selection_backend="torch_topk",
            max_requests=8,
            prompt_len=1024,
            width=8191,
            keep_count=4096,
        )
        self._mark_ready(manager, score, selection)
        workspace = SimpleNamespace(
            device=torch.device("cpu"),
            nbytes=64,
            pointer_fingerprint=mock.Mock(return_value=("fingerprint",)),
        )
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        cache._capture_graph = mock.Mock(side_effect=RuntimeError("unsupported capture"))
        manager._standalone_graph_cache_for = mock.Mock(return_value=cache)
        manager._standalone_graph_workspace_for = mock.Mock(return_value=workspace)
        kwargs = dict(
            prepared=prepared,
            layer_pools=[torch.empty(1)],
            dense_layers=[0],
            dense_groups=[[0]],
            swa_layers=[],
            swa_window=None,
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score,
            selection_workspace=selection,
            fixed_perhead_segment_views=mock.Mock(),
        )

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(
                tri_module,
                "nvtx_range",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
        ):
            assert manager._try_standalone_cuda_graph(**kwargs) is None
            assert manager._try_standalone_cuda_graph(**kwargs) is None

        manager._standalone_graph_workspace_for.assert_called_once()
        cache._capture_graph.assert_called_once()
        assert cache.snapshot()["capture_failure"] == 1
        assert cache.snapshot()["attempt"] == 2
        assert cache.snapshot()["fallback"] == 2
        assert cache.snapshot()["covered_requests"] == 0
        assert cache.snapshot()["disabled_buckets"] == 1
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 2,
            "attempt_requests": 2,
            "fallback": 2,
            "fallback_requests": 2,
        }


class TestFixedBatchedCompactionWorkspace:
    def test_dense_copy_excludes_prompt_and_swa_uses_independent_table(self):
        pools = [
            torch.zeros(4, 2, 1, 4, 2),
            torch.zeros(4, 2, 1, 4, 2),
        ]
        score = SimpleNamespace(
            page_count=2,
            representative_slots={0: 0, 1: 1},
            page_ids_device=torch.tensor([[[3, 1]], [[2, 0]]], dtype=torch.int64),
        )
        selection = SimpleNamespace(
            max_requests=1,
            prompt_len=2,
            keep_count=4,
            width=6,
            keep=torch.tensor([[0, 1, 2, 4, 5, 7]], dtype=torch.int64),
        )
        workspace = FixedBatchedCompactionWorkspace(
            layer_pools=pools,
            dense_layers=[0],
            swa_layers=[1],
            layer_group_representative={0: 0},
            global_layers=[0, 1],
            score_workspace=score,
            selection_workspace=selection,
            request_count=1,
            seq_len=8,
            prompt_len=2,
            decode_keep_count=4,
            swa_window=2,
            arena_generation=1,
        )

        with mock.patch.object(_FixedCompactLaunch, "run") as run:
            workspace.launch()

        assert run.call_count == 2
        assert workspace.dense_source.tolist() == [2, 4, 5, 7]
        assert workspace.dense_destination.tolist() == [2, 3, 4, 5]
        assert 0 not in workspace.dense_destination.tolist()
        assert 1 not in workspace.dense_destination.tolist()
        assert workspace.dense_launches[0].page_ids.tolist() == [3, 1]
        assert workspace.swa_launches[0].page_ids.tolist() == [2, 0]
        assert workspace.swa_source.tolist() == [6, 7]
        assert workspace.swa_destination.tolist() == [4, 5]

    def test_tensor_fingerprint_changes_for_view_offset_and_stride(self):
        base = torch.arange(16)
        mutations = (
            base.clone(),
            base[1:],
            base.view(4, 4),
            base.view(4, 4).t(),
            base.to(torch.float32),
            torch.arange(32)[:16],
        )

        original = _tensor_fingerprint(base)
        assert all(original != _tensor_fingerprint(tensor) for tensor in mutations)

    def test_tensor_fingerprint_ignores_storage_wrapper_identity(self):
        captured = _FakeTensor(storage_cdata=1)
        rebound = _FakeTensor(storage_cdata=2)

        assert captured.untyped_storage()._cdata != rebound.untyped_storage()._cdata
        assert _tensor_fingerprint(captured) == _tensor_fingerprint(rebound)

    def test_graph_fingerprint_detects_named_selection_tensor_rebind(self):
        captured = _FakeTensor(storage_cdata=1)
        rebound = _FakeTensor(
            storage_cdata=2,
            storage_data_ptr=8192,
            data_ptr=captured.data_ptr(),
        )

        def weak_snapshot(tensor):
            return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride())

        assert weak_snapshot(captured) == weak_snapshot(rebound)
        assert _tensor_fingerprint(captured) != _tensor_fingerprint(rebound)

        selection = SimpleNamespace(selection_backend="torch_topk", keep=captured)

        def named_tensors():
            return (("keep", selection.keep),)

        selection.named_tensors = named_tensors
        score_tensor = _FakeTensor(
            storage_cdata=3,
            storage_data_ptr=12288,
            data_ptr=12288,
        )
        score = SimpleNamespace(
            page_ids_device=score_tensor,
            round_starts_device=score_tensor,
            phase_base=score_tensor,
            phase=score_tensor,
            cos_phase=score_tensor,
            sin_phase=score_tensor,
            mean_cos=score_tensor,
            mean_sin=score_tensor,
            offsets=score_tensor,
            omega=score_tensor,
            groups={},
        )
        workspace = object.__new__(FixedBatchedCompactionWorkspace)
        workspace.selection_workspace = selection
        workspace.score_workspace = score
        workspace.request_count = 1
        workspace.seq_len = 8
        workspace.prompt_len = 2
        workspace.decode_keep_count = 4
        workspace.dense_layers = ()
        workspace.swa_layers = ()
        workspace.global_layers = ()
        workspace.storage_groups = ()
        workspace.arena_generation = 1
        workspace.layer_pools = ()
        workspace._tensor_refs = ()
        stream = SimpleNamespace(device=torch.device("cuda"), cuda_stream=9)

        with mock.patch.object(torch.cuda, "current_blas_handle", return_value=17):
            before = workspace.pointer_fingerprint(stream)
            selection.keep = rebound
            after = workspace.pointer_fingerprint(stream)

        assert before != after

    def test_runtime_match_uses_allocation_pointer_not_storage_wrapper(self):
        captured = _FakeTensor(storage_cdata=1)
        rebound = _FakeTensor(storage_cdata=2)
        moved = _FakeTensor(
            storage_cdata=3,
            storage_data_ptr=8192,
            data_ptr=8192,
        )
        score_workspace = object()
        selection_workspace = object()
        workspace = object.__new__(FixedBatchedCompactionWorkspace)
        workspace.score_workspace = score_workspace
        workspace.selection_workspace = selection_workspace
        workspace.layer_pools = (captured,)
        workspace.dense_layers = (0,)
        workspace.swa_layers = ()
        workspace.global_layers = (0,)
        workspace.storage_groups = ((0, 0),)

        runtime = dict(
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score_workspace,
            selection_workspace=selection_workspace,
        )
        assert workspace.matches_runtime(layer_pools=[rebound], **runtime)
        assert not workspace.matches_runtime(layer_pools=[moved], **runtime)

    @CUDA_REQUIRED
    def test_cuda_alias_with_new_storage_wrapper_keeps_fingerprint(self):
        captured = torch.arange(16, device="cuda")
        rebound = torch.from_dlpack(captured)

        assert captured.data_ptr() == rebound.data_ptr()
        assert captured.untyped_storage()._cdata != rebound.untyped_storage()._cdata
        assert _tensor_fingerprint(captured) == _tensor_fingerprint(rebound)


class TestStandaloneGraphCuda:
    @staticmethod
    def _build_formal_path(width, budget, backend, request_count, initial_pool, *, prompt_len):
        device = initial_pool.device
        seq_len = prompt_len + width
        tokens_per_block = int(initial_pool.shape[3])
        page_count = (seq_len + tokens_per_block - 1) // tokens_per_block
        pool = initial_pool.clone()
        q_real = torch.tensor([[[0.75]]], dtype=torch.float32, device=device)
        q_imag = torch.tensor([[[0.25]]], dtype=torch.float32, device=device)
        mlr = torch.tensor([[[0.125]]], dtype=torch.float32, device=device)
        freq = torch.tensor([1.0], dtype=torch.float32, device=device)
        omega = torch.tensor([0.013], dtype=torch.float32, device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=device)
        score = _FixedScoreMetadataWorkspace(
            [pool],
            [[0]],
            [0],
            [0],
            request_count,
            seq_len,
            1,
            1,
            q_real,
            q_imag,
            mlr,
            freq,
            offsets,
            omega,
        )
        selection = _BatchedFixedUnionWorkspace(
            1,
            width,
            budget,
            prompt_len,
            dtype=torch.float32,
            device=device,
            selection_backend=backend,
            max_requests=request_count,
        )
        compaction = FixedBatchedCompactionWorkspace(
            layer_pools=[pool],
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score,
            selection_workspace=selection,
            request_count=request_count,
            seq_len=seq_len,
            prompt_len=prompt_len,
            decode_keep_count=budget,
            swa_window=None,
            arena_generation=1,
        )
        request_ids = list(range(request_count))
        tables = [
            list(range(request * page_count, (request + 1) * page_count)) for request in request_ids
        ]

        def stage(round_shift=0, reverse_pages=False):
            runtime_tables = [list(reversed(row)) if reverse_pages else list(row) for row in tables]
            round_starts = [float(seq_len + round_shift + request * 17) for request in request_ids]
            assert score.stage(
                lambda _ids, _layer, num_blocks_per_seq=None: runtime_tables,
                request_ids,
                round_starts,
            )
            return runtime_tables

        def score_and_select():
            score.prepare_phase(request_count)
            per_head, _ = score.groups[0].launch(
                request_count,
                score.round_starts_device,
                score.mean_cos,
                score.mean_sin,
                "mean",
            )
            views = fixed_perhead_segment_views(per_head, request_count, 1, seq_len)
            segments = [
                [views[:, request, 0, prompt_len:seq_len]] for request in range(request_count)
            ]
            selection.select_requests(segments, normalize_scores=True)

        def body():
            score_and_select()
            compaction.launch()

        def stage4_body():
            score_and_select()
            triton_tri_compact(
                pool,
                [score.page_ids_device[0, request] for request in range(request_count)],
                [selection.keep[request] for request in range(request_count)],
                [seq_len] * request_count,
            )

        return pool, score, selection, compaction, stage, body, stage4_body

    @pytest.mark.parametrize(
        "prompt_len,width,budget,backend,request_count",
        [
            (0, 2175, 2048, "indexer_topk", 1),
            (256, 3072, 2048, "indexer_topk", 2),
            (1024, 4095, 2048, "indexer_topk", 4),
            (1024, 4096, 2048, "torch_topk", 7),
            (1536, 4223, 4096, "torch_topk", 8),
            (1024, 4224, 4096, "torch_topk", 8),
            (1024, 5119, 4096, "torch_topk", 16),
            (1024, 8191, 4096, "torch_topk", 1),
            (2048, 8192, 4096, "torch_topk", 31),
            (1024, 9216, 8192, "torch_topk", 32),
        ],
    )
    @CUDA_REQUIRED
    def test_dynamic_exact_graph_matches_stage4_eager(
        self, prompt_len, width, budget, backend, request_count
    ):
        device = torch.device("cuda")
        seq_len = prompt_len + width
        tokens_per_block = 128
        page_count = (seq_len + tokens_per_block - 1) // tokens_per_block
        total_pages = request_count * page_count
        initial = torch.arange(
            total_pages * 2 * tokens_per_block * 2,
            dtype=torch.float32,
            device=device,
        ).view(total_pages, 2, 1, tokens_per_block, 2)
        eager = self._build_formal_path(
            width,
            budget,
            backend,
            request_count,
            initial,
            prompt_len=prompt_len,
        )
        graphed = self._build_formal_path(
            width,
            budget,
            backend,
            request_count,
            initial,
            prompt_len=prompt_len,
        )
        eager_pool, _, eager_selection, _, eager_stage, _, eager_body = eager
        graph_pool, _, graph_selection, graph_compaction, graph_stage, graph_body, _ = graphed

        # Warm the exact operators and Triton kernels outside capture, then reset.
        eager_stage()
        eager_body()
        torch.cuda.synchronize(device)
        eager_pool.copy_(initial)
        graph_pool.copy_(initial)
        tables = eager_stage()
        assert graph_stage() == tables
        eager_body()
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=2 * 1024**3)
        stream = torch.cuda.current_stream(device)
        fingerprint = graph_compaction.pointer_fingerprint(stream)
        outcome = cache.execute(
            key=(width, budget, backend, request_count),
            request_count=request_count,
            fingerprint=fingerprint,
            workspace=graph_compaction,
            capture_body=graph_body,
        )
        torch.cuda.synchronize(device)

        assert outcome == "capture"
        assert torch.equal(
            eager_selection.keep[:request_count],
            graph_selection.keep[:request_count],
        )
        assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
        eager_pool.copy_(initial)
        graph_pool.copy_(initial)
        torch.cuda.synchronize(device)
        tables = eager_stage(round_shift=37, reverse_pages=True)
        assert graph_stage(round_shift=37, reverse_pages=True) == tables
        eager_body()
        assert (
            cache.execute(
                key=(width, budget, backend, request_count),
                request_count=request_count,
                fingerprint=fingerprint,
                workspace=graph_compaction,
                capture_body=graph_body,
            )
            == "replay"
        )
        torch.cuda.synchronize(device)
        assert torch.equal(
            eager_selection.keep[:request_count],
            graph_selection.keep[:request_count],
        )
        assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
        for request, page_ids in enumerate(tables):
            initial_tokens = initial[page_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            graph_tokens = graph_pool[page_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            assert torch.equal(
                graph_tokens[:, :, :prompt_len],
                initial_tokens[:, :, :prompt_len],
            ), f"request {request} prompt was modified"
        assert cache.snapshot()["capture"] == 1
        assert cache.snapshot()["replay"] == 2
        assert cache.snapshot()["launch"] == 2
        assert cache.snapshot()["cache_hit"] == 1
        assert cache.snapshot()["covered_requests"] == 2 * request_count
        assert cache.snapshot()["fallback"] == 0
        assert cache.snapshot()["capture_failure"] == 0
        assert cache.snapshot()["replay_failure"] == 0

    @CUDA_REQUIRED
    def test_graph_compaction_preserves_prompt_and_rebases_swa_latest_window(self):
        device = torch.device("cuda")
        request_count = 2
        page_count = 3
        dense_tables = torch.tensor([[2, 0, 1], [5, 3, 4]], device=device)
        swa_tables = torch.tensor([[1, 2, 0], [4, 5, 3]], device=device)
        base_pools = [
            torch.arange(6 * 2 * 1 * 4 * 2, dtype=torch.float32, device=device).view(6, 2, 1, 4, 2),
            torch.arange(6 * 2 * 1 * 4 * 2, dtype=torch.float32, device=device).view(6, 2, 1, 4, 2)
            + 1000.0,
        ]
        keep = torch.tensor(
            [[0, 1, 2, 4, 5, 7], [0, 1, 3, 4, 6, 7]],
            dtype=torch.int64,
            device=device,
        )

        def build(pools):
            score = SimpleNamespace(
                page_count=page_count,
                representative_slots={0: 0, 1: 1},
                page_ids_device=torch.stack((dense_tables, swa_tables)),
            )
            selection = SimpleNamespace(
                max_requests=request_count,
                prompt_len=2,
                keep_count=4,
                width=6,
                keep=keep.clone(),
                selection_backend="torch_topk",
                named_tensors=lambda: (),
            )
            return FixedBatchedCompactionWorkspace(
                layer_pools=pools,
                dense_layers=[0],
                swa_layers=[1],
                layer_group_representative={0: 0},
                global_layers=[0, 1],
                score_workspace=score,
                selection_workspace=selection,
                request_count=request_count,
                seq_len=8,
                prompt_len=2,
                decode_keep_count=4,
                swa_window=2,
                arena_generation=1,
            )

        eager_pools = [pool.clone() for pool in base_pools]
        graph_pools = [pool.clone() for pool in base_pools]
        graphed = build(graph_pools)
        swa_source = torch.tensor([6, 7], dtype=torch.int64, device=device)
        swa_destination = torch.tensor([4, 5], dtype=torch.int64, device=device)
        prepared = [
            {"protected_tail": 0},
            {
                "protected_tail": 2,
                "page_ids": {0: dense_tables[1], 1: swa_tables[1]},
                "tail_source": torch.tensor([8, 9], dtype=torch.int64, device=device),
                "tail_destination": torch.tensor([6, 7], dtype=torch.int64, device=device),
                "compaction_length": 10,
            },
        ]

        def run_eager(pools):
            triton_tri_compact(
                pools[0],
                [dense_tables[request] for request in range(request_count)],
                [keep[request] for request in range(request_count)],
                [8] * request_count,
            )
            triton_tri_compact(
                pools[1],
                [swa_tables[request] for request in range(request_count)],
                [swa_source] * request_count,
                [8] * request_count,
                dest_list=[swa_destination] * request_count,
            )
            TriAttention._copy_protected_suffixes(prepared, pools, [0], [1], {0: 0})

        run_eager(eager_pools)
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024**3)
        assert (
            cache.execute(
                key=("swa",),
                request_count=request_count,
                fingerprint=("swa",),
                workspace=graphed,
                capture_body=graphed.launch,
            )
            == "capture"
        )
        TriAttention._copy_protected_suffixes(prepared, graph_pools, [0], [1], {0: 0})
        torch.cuda.synchronize(device)

        for eager_pool, graph_pool in zip(eager_pools, graph_pools):
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
        for request in range(request_count):
            dense_ids = dense_tables[request]
            swa_ids = swa_tables[request]
            dense_before = base_pools[0][dense_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            dense_after = graph_pools[0][dense_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            swa_before = base_pools[1][swa_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            swa_after = graph_pools[1][swa_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
            assert torch.equal(dense_after[:, :, :2], dense_before[:, :, :2])
            assert torch.equal(swa_after[:, :, :2], swa_before[:, :, :2])
            assert torch.equal(swa_after[:, :, 4:6], swa_before[:, :, 6:8])
        dense_after = graph_pools[0][dense_tables[1]].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
        dense_before = base_pools[0][dense_tables[1]].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
        swa_after = graph_pools[1][swa_tables[1]].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
        swa_before = base_pools[1][swa_tables[1]].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
        assert torch.equal(dense_after[:, :, 6:8], dense_before[:, :, 8:10])
        assert torch.equal(swa_after[:, :, 6:8], swa_before[:, :, 8:10])

        for pool, base in zip(eager_pools, base_pools):
            pool.copy_(base)
        for pool, base in zip(graph_pools, base_pools):
            pool.copy_(base)
        torch.cuda.synchronize(device)
        run_eager(eager_pools)
        assert (
            cache.execute(
                key=("swa",),
                request_count=request_count,
                fingerprint=("swa",),
                workspace=graphed,
                capture_body=graphed.launch,
            )
            == "replay"
        )
        TriAttention._copy_protected_suffixes(prepared, graph_pools, [0], [1], {0: 0})
        torch.cuda.synchronize(device)
        for eager_pool, graph_pool in zip(eager_pools, graph_pools):
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
        assert cache.snapshot()["capture"] == 1
        assert cache.snapshot()["cache_hit"] == 1

    @pytest.mark.parametrize(
        "stable_len,prompt_len,keep,protected_tail",
        [
            (7, 1, [0, 1, 3, 5, 6], 1),
            (8, 2, [0, 1, 2, 4, 5, 7], 2),
            (8, 2, [0, 1, 2, 4, 5, 7], 4),
        ],
    )
    @CUDA_REQUIRED
    def test_graph_prefix_then_suffix_copy_capture_and_replay_are_byte_exact(
        self, stable_len, prompt_len, keep, protected_tail
    ):
        device = torch.device("cuda")
        tokens_per_block = 4
        stable_pages = (stable_len + tokens_per_block - 1) // tokens_per_block
        total_pages = (stable_len + protected_tail + tokens_per_block - 1) // tokens_per_block
        page_ids = torch.tensor([2, 0, 3][:total_pages], dtype=torch.int64, device=device)
        base_pool = torch.arange(
            4 * 2 * 1 * tokens_per_block * 2,
            dtype=torch.float32,
            device=device,
        ).view(4, 2, 1, tokens_per_block, 2)
        eager_pool = base_pool.clone()
        graph_pool = base_pool.clone()
        keep_tensor = torch.tensor([keep], dtype=torch.int64, device=device)
        score = SimpleNamespace(
            page_count=stable_pages,
            representative_slots={0: 0},
            page_ids_device=page_ids[:stable_pages].view(1, 1, stable_pages),
        )
        selection = SimpleNamespace(
            max_requests=1,
            prompt_len=prompt_len,
            keep_count=len(keep) - prompt_len,
            width=stable_len - prompt_len,
            keep=keep_tensor,
            selection_backend="torch_topk",
            named_tensors=lambda: (),
        )
        workspace = FixedBatchedCompactionWorkspace(
            layer_pools=[graph_pool],
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score,
            selection_workspace=selection,
            request_count=1,
            seq_len=stable_len,
            prompt_len=prompt_len,
            decode_keep_count=len(keep) - prompt_len,
            swa_window=None,
            arena_generation=1,
        )
        tail_source = torch.arange(
            stable_len,
            stable_len + protected_tail,
            dtype=torch.int64,
            device=device,
        )
        tail_destination = torch.arange(
            len(keep),
            len(keep) + protected_tail,
            dtype=torch.int64,
            device=device,
        )
        prepared = [
            {
                "protected_tail": protected_tail,
                "page_ids": {0: page_ids},
                "tail_source": tail_source,
                "tail_destination": tail_destination,
                "compaction_length": stable_len + protected_tail,
            }
        ]
        stream = torch.cuda.Stream(device=device)

        # Compile both graph-owned and graph-external kernels before capture.
        with torch.cuda.stream(stream):
            workspace.launch()
            TriAttention._copy_protected_suffixes(prepared, [graph_pool], [0], [], {0: 0})
        stream.synchronize()
        graph_pool.copy_(base_pool)
        torch.cuda.synchronize(device)

        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024**3)
        for expected_outcome in ("capture", "replay"):
            eager_pool.copy_(base_pool)
            graph_pool.copy_(base_pool)
            torch.cuda.synchronize(device)

            triton_tri_compact(
                eager_pool,
                [page_ids[:stable_pages]],
                [keep_tensor[0]],
                [stable_len],
            )
            TriAttention._copy_protected_suffixes(prepared, [eager_pool], [0], [], {0: 0})
            with torch.cuda.stream(stream):
                outcome = cache.execute(
                    key=(stable_len, prompt_len, len(keep)),
                    request_count=1,
                    fingerprint=("stable-prefix", stable_len, prompt_len, len(keep)),
                    workspace=workspace,
                    capture_body=workspace.launch,
                )
                TriAttention._copy_protected_suffixes(
                    prepared,
                    [graph_pool],
                    [0],
                    [],
                    {0: 0},
                )
            stream.synchronize()
            torch.cuda.synchronize(device)

            assert outcome == expected_outcome
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))

        stats = cache.snapshot()
        assert stats["capture"] == 1
        assert stats["cache_hit"] == 1
        assert stats["launch"] == 2
