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
                fingerprint=("pointers",),
                workspace=workspace,
                capture_body=body,
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("bucket",),
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

    def test_capture_failure_performs_one_caller_fallback(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        cache._capture_graph = mock.Mock(side_effect=RuntimeError("capture failed"))
        eager = mock.Mock()
        outcome = cache.execute(
            key=("bucket",),
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
                fingerprint=("new-pointers",),
                workspace=_cache_workspace(),
                capture_body=mock.Mock(),
            )
            == "fallback"
        )
        cache._capture_graph.assert_called_once()
        assert cache.snapshot()["disabled_buckets"] == 1

    def test_replay_failure_is_poisoned_without_eager_fallback(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        graph = _FakeGraph(fail_replay=True)
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 0))
        cache._record_last_use = mock.Mock()
        eager = mock.Mock()

        with pytest.raises(RuntimeError, match="replay failed"):
            outcome = cache.execute(
                key=("bucket",),
                fingerprint=("pointers",),
                workspace=_cache_workspace(),
                capture_body=mock.Mock(),
            )
            if outcome == "fallback":
                eager()

        eager.assert_not_called()
        assert cache.counts["replay_failure"] == 1
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
                fingerprint=("one",),
                workspace=workspace,
                capture_body=mock.Mock(),
            )
            == "capture"
        )
        assert (
            cache.execute(
                key=("second",),
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
            fingerprint=("old",),
            workspace=workspace,
            capture_body=mock.Mock(),
        )
        cache.execute(
            key=("bucket",),
            fingerprint=("new",),
            workspace=workspace,
            capture_body=mock.Mock(),
        )

        assert first_graph.resets == 1
        assert second_graph.replays == 1
        assert cache.counts["invalidated"] == 1

    def test_capture_allocation_over_byte_cap_is_reset_before_replay(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=128)
        graph = _FakeGraph()
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 96))

        outcome = cache.execute(
            key=("bucket",),
            fingerprint=("pointers",),
            workspace=_cache_workspace(nbytes=64),
            capture_body=mock.Mock(),
        )

        assert outcome == "fallback"
        assert graph.replays == 0
        assert graph.resets == 1
        assert cache.snapshot()["active_entries"] == 0


class TestStandaloneGraphBuckets:
    @pytest.mark.parametrize(
        "width,budget,backend",
        [
            (8191, 4096, "torch_topk"),
            (8192, 4096, "torch_topk"),
            (4095, 2048, "indexer_topk"),
            (4096, 2048, "torch_topk"),
            (4223, 4096, "torch_topk"),
            (4224, 4096, "torch_topk"),
        ],
    )
    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_exact_formal_matrix_is_accepted(self, width, budget, backend, request_count):
        manager = TriAttention.__new__(TriAttention)
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            {
                "seq_len": 1024 + width,
                "request": SimpleNamespace(py_prompt_len=1024),
                "expected_keep_count": 1024 + budget,
            }
            for _ in range(request_count)
        ]
        score = SimpleNamespace(prewarm_key=(width, budget))
        selection = SimpleNamespace(
            selection_backend=backend,
            max_requests=8,
            prompt_len=1024,
            width=width,
            keep_count=budget,
        )

        key = manager._standalone_graph_bucket_for(prepared, score, selection)

        assert key is not None
        assert key[2:7] == (
            request_count,
            1024 + width,
            1024,
            budget,
            backend,
        )

    @pytest.mark.parametrize(
        "request_count,prompt_len,width,budget,backend",
        [
            (2, 1024, 8192, 4096, "torch_topk"),
            (1, 0, 8192, 4096, "torch_topk"),
            (1, 1024, 8193, 4096, "torch_topk"),
            (1, 1024, 4095, 2048, "torch_topk"),
        ],
    )
    def test_unsealed_shape_or_backend_falls_back(
        self, request_count, prompt_len, width, budget, backend
    ):
        manager = TriAttention.__new__(TriAttention)
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            {
                "seq_len": prompt_len + width,
                "request": SimpleNamespace(py_prompt_len=prompt_len),
                "expected_keep_count": prompt_len + budget,
            }
            for _ in range(request_count)
        ]
        score = SimpleNamespace(prewarm_key=(width, budget))
        selection = SimpleNamespace(
            selection_backend=backend,
            max_requests=8,
            prompt_len=prompt_len,
            width=width,
            keep_count=budget,
        )

        assert manager._standalone_graph_bucket_for(prepared, score, selection) is None

    def test_capture_body_covers_phase_score_select_compact_then_publishes(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager = TriAttention.__new__(TriAttention)
        manager.top_B = 4096
        manager.normalize_scores = True
        manager.score_aggregation = "mean"
        manager._standalone_cuda_graph_enabled = True
        manager._evicted = {}
        manager._pre_forward_kv_lengths = {7: 9215}
        prepared = [
            {
                "request": SimpleNamespace(py_prompt_len=1024),
                "request_id": 7,
                "seq_len": 9215,
                "expected_keep_count": 5120,
            }
        ]
        score_output = torch.zeros(1, 9215)
        group = SimpleNamespace(launch=mock.Mock(return_value=(score_output, None)))
        score = SimpleNamespace(
            prewarm_key=("formal-a-first",),
            prepare_phase=mock.Mock(),
            groups={0: group},
            round_starts_device=torch.zeros(8),
            mean_cos=torch.zeros(8, 1),
            mean_sin=torch.zeros(8, 1),
        )
        selection = SimpleNamespace(
            selection_backend="torch_topk",
            max_requests=8,
            prompt_len=1024,
            width=8191,
            keep_count=4096,
            select_requests=mock.Mock(),
        )
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
        views = torch.zeros(1, 1, 1, 9215)
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

        assert targets == [(7, 5121)]
        score.prepare_phase.assert_called_once_with(1)
        group.launch.assert_called_once()
        selected_segments = selection.select_requests.call_args.args[0]
        assert selected_segments[0][0].shape == (1, 8191)
        workspace.launch.assert_called_once_with()
        assert manager._evicted == {7: 4095}
        assert manager._pre_forward_kv_lengths == {7: 5120}

    def test_graph_fallback_does_not_publish_or_execute_eager_inside_helper(self):
        manager = TriAttention.__new__(TriAttention)
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
            }
        ]
        score = SimpleNamespace(prewarm_key=("formal-a-first",))
        selection = SimpleNamespace(
            selection_backend="torch_topk",
            max_requests=8,
            prompt_len=1024,
            width=8191,
            keep_count=4096,
        )
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

        manager = TriAttention.__new__(TriAttention)
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
            }
        ]
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score = SimpleNamespace(
            prewarm_key=("formal-a-first",),
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
        assert cache.snapshot()["disabled_buckets"] == 1


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
    def _build_formal_path(width, budget, backend, request_count, initial_pool):
        device = initial_pool.device
        prompt_len = 1024
        seq_len = prompt_len + width
        tokens_per_block = int(initial_pool.shape[3])
        page_count = (seq_len + tokens_per_block) // tokens_per_block
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
            8,
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
            max_requests=8,
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
                lambda _ids, _layer: runtime_tables,
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
        "width,budget,backend",
        [
            (8191, 4096, "torch_topk"),
            (8192, 4096, "torch_topk"),
            (4095, 2048, "indexer_topk"),
            (4096, 2048, "torch_topk"),
            (4223, 4096, "torch_topk"),
            (4224, 4096, "torch_topk"),
        ],
    )
    @pytest.mark.parametrize("request_count", [1, 7, 8])
    @CUDA_REQUIRED
    def test_formal_bucket_graph_matches_stage4_eager(self, width, budget, backend, request_count):
        device = torch.device("cuda")
        prompt_len = 1024
        seq_len = prompt_len + width
        tokens_per_block = 128
        page_count = (seq_len + tokens_per_block) // tokens_per_block
        total_pages = 8 * page_count
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
        )
        graphed = self._build_formal_path(
            width,
            budget,
            backend,
            request_count,
            initial,
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
                _tensor_names=(),
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
        triton_tri_compact(
            eager_pools[0],
            [dense_tables[request] for request in range(request_count)],
            [keep[request] for request in range(request_count)],
            [8] * request_count,
        )
        swa_source = torch.tensor([6, 7], dtype=torch.int64, device=device)
        swa_destination = torch.tensor([4, 5], dtype=torch.int64, device=device)
        triton_tri_compact(
            eager_pools[1],
            [swa_tables[request] for request in range(request_count)],
            [swa_source] * request_count,
            [8] * request_count,
            dest_list=[swa_destination] * request_count,
        )
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024**3)
        assert (
            cache.execute(
                key=("swa",),
                fingerprint=("swa",),
                workspace=graphed,
                capture_body=graphed.launch,
            )
            == "capture"
        )
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
