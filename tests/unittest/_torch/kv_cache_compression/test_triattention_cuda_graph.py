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
    _tensor_fingerprint,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttention,
    _BatchedFixedPerHeadWorkspace,
    _BatchedFixedUnionWorkspace,
    _EvictionBucketResources,
    _FixedScoreMetadataWorkspace,
    _PreparedEviction,
    _RequestCompressionState,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    fixed_perhead_segment_views,
)

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
_TORCH_TOPK_ORACLE = torch.topk


def _fake_cute_dsl_topk_decode(scores, seq_lens, output, top_k, next_n):
    """Write CuTE-op-shaped results while retaining torch.topk only as an oracle."""
    assert next_n == 1
    for row in range(scores.shape[0]):
        width = int(seq_lens[row].item())
        indices = _TORCH_TOPK_ORACLE(
            scores[row, :width],
            int(top_k),
            sorted=False,
        ).indices
        output[row].copy_(indices.to(dtype=torch.int32))


def _cpp_sparse_compact_reference(
    pool,
    page_ids_list,
    source_list,
    seq_len_list,
    *,
    dest_list=None,
):
    """Build the C++ compact-op inputs for eager graph test references."""
    num_kv_heads = int(pool.shape[2])
    tables = []
    sources = []
    destinations = []
    per_head_source = None
    per_head_destination = None
    for request_index, (page_ids, source, seq_len) in enumerate(
        zip(page_ids_list, source_list, seq_len_list)
    ):
        source = source.to(device=pool.device, dtype=torch.int32)
        move_count = int(source.shape[-1])
        if (dest_list is None and move_count >= int(seq_len)) or move_count == 0:
            continue
        current_per_head = source.ndim == 2
        if per_head_source is None:
            per_head_source = current_per_head
        else:
            assert per_head_source == current_per_head
        if not current_per_head:
            source = source.reshape(1, -1).expand(num_kv_heads, -1)
        sources.append(source)
        tables.append(page_ids.to(device=pool.device, dtype=torch.int32).reshape(-1))

        if dest_list is not None:
            destination = dest_list[request_index].to(device=pool.device, dtype=torch.int32)
            current_per_head = destination.ndim == 2
            if per_head_destination is None:
                per_head_destination = current_per_head
            else:
                assert per_head_destination == current_per_head
            destinations.append(destination)

    if not sources:
        return
    max_pages = max(int(table.numel()) for table in tables)
    page_table = torch.zeros(len(tables), max_pages, dtype=torch.int32, device=pool.device)
    for request_index, table in enumerate(tables):
        page_table[request_index, : table.numel()] = table
    offsets_host = [0]
    for source in sources:
        offsets_host.append(offsets_host[-1] + int(source.shape[1]))
    offsets = torch.tensor(offsets_host, dtype=torch.int32, device=pool.device)
    indices = torch.cat(sources, dim=1).contiguous()
    destination_indices = None
    if destinations:
        destination_indices = torch.cat(
            destinations,
            dim=1 if per_head_destination else 0,
        ).contiguous()
    torch.ops.trtllm.sparse_kv_cache_compact(
        pool,
        page_table,
        indices,
        offsets,
        destination_indices,
    )


class _AlternatingBoundaryTieTopK:
    """Emulate a raw TopK op with unspecified exact-tie membership."""

    def __init__(self):
        self._tie_calls = {}

    def __call__(self, scores, seq_lens, output, top_k, next_n):
        assert next_n == 1
        for row in range(scores.shape[0]):
            width = int(seq_lens[row].item())
            values = scores[row, :width]
            threshold = torch.sort(values, descending=True).values[int(top_k) - 1]
            higher = torch.nonzero(values > threshold, as_tuple=False).flatten()
            tied = torch.nonzero(values == threshold, as_tuple=False).flatten()
            remaining = int(top_k) - int(higher.numel())
            if tied.numel() > remaining:
                key = (scores.data_ptr(), row, width, int(top_k))
                tie_call = self._tie_calls.get(key, 0)
                self._tie_calls[key] = tie_call + 1
                tied = tied if tie_call % 2 == 0 else tied.flip(0)
            output[row].copy_(torch.cat((higher, tied[:remaining])).to(torch.int32))


def _boundary_row_scores(*, exact_tie, device):
    scores = torch.tensor(
        [8.0, 7.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0],
        dtype=torch.float32,
        device=device,
    )
    if not exact_tie:
        scores[4] = torch.nextafter(
            scores[3],
            torch.tensor(float("inf"), dtype=torch.float32, device=device),
        )
    return scores


def _boundary_union_scores(*, exact_tie, device):
    scores = torch.tensor(
        [
            [12.0, -10.0, -11.0, 9.0, 8.0, -12.0, 7.0, -13.0],
            [-10.0, 11.0, 10.0, -11.0, -12.0, 9.0, -13.0, 7.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    if not exact_tie:
        scores[1, 5] = torch.nextafter(
            scores[0, 3],
            torch.tensor(float("inf"), dtype=torch.float32, device=device),
        )
    return scores


def _union_keep_oracle(scores, keep_count):
    """Return the fixed-union contract without invoking a production selector."""
    combined = scores.max(dim=0).values
    row_indices = _TORCH_TOPK_ORACLE(
        scores,
        keep_count,
        dim=1,
        sorted=False,
    ).indices
    union_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    union_mask[row_indices.reshape(-1)] = True
    union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
    if union_indices.numel() >= keep_count:
        relative = _TORCH_TOPK_ORACLE(
            combined.index_select(0, union_indices),
            keep_count,
            sorted=False,
        ).indices
        union_indices = union_indices.index_select(0, relative)
    else:
        remaining = keep_count - int(union_indices.numel())
        residual = combined.clone()
        residual[union_mask] = float("-inf")
        extra = _TORCH_TOPK_ORACLE(
            residual,
            remaining,
            sorted=False,
        ).indices
        union_indices = torch.cat((union_indices, extra))
    return torch.sort(union_indices).values


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


def _make_triattention(eviction_mode="union"):
    """Construct a fully initialized manager for graph-orchestration tests."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    kv_cache_manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    kv_cache_manager.enable_block_reuse = False
    kv_cache_manager.is_draft = False
    kv_cache_manager.mapping = SimpleNamespace(enable_attention_dp=False)
    kv_cache_manager.is_disagg = False
    kv_cache_manager.max_beam_width = 1
    kv_cache_manager.max_batch_size = 32
    kv_cache_manager.num_extra_kv_tokens = 0
    kv_cache_manager.max_total_draft_tokens = 0
    kv_cache_manager._kv_reserve_draft_tokens = 0
    kv_cache_manager.max_seq_len = 65536
    kv_cache_manager.max_attention_window_vec = []
    kv_cache_manager.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    kv_cache_manager.pp_layers = []
    kv_cache_manager.layer_offsets = {}
    kv_cache_manager.layer_to_pool_mapping_dict = {}
    return TriAttention(
        kv_cache_manager,
        top_B=8,
        eviction_mode=eviction_mode,
        skip_swa=False,
    )


def _prepared_eviction(
    *,
    seq_len,
    prompt_len,
    expected_keep_count,
    protected_tail=0,
    request_id=0,
    round_start=0.0,
):
    return _PreparedEviction(
        request=SimpleNamespace(py_prompt_len=prompt_len),
        request_id=request_id,
        seq_len=seq_len,
        round_start=round_start,
        expected_keep_count=expected_keep_count,
        protected_tail=protected_tail,
    )


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
                "rejected": 0,
                "invalidated": 0,
                "failure": 0,
                "capture_failure": 0,
                "replay_failure": 0,
            }
        ]

    def test_capture_failure_is_rejected_without_running_eager_work(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        cache._capture_graph = mock.Mock(side_effect=RuntimeError("capture failed"))
        outcome = cache.execute(
            key=("bucket",),
            request_count=3,
            fingerprint=("pointers",),
            workspace=_cache_workspace(),
            capture_body=mock.Mock(),
        )
        assert outcome == "rejected"
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
            == "rejected"
        )
        cache._capture_graph.assert_called_once()
        assert cache.snapshot()["disabled_buckets"] == 1
        assert cache.snapshot()["failure"] == 1
        assert cache.snapshot()["buckets"][0]["rejected"] == 2
        assert cache.snapshot()["buckets"][0]["covered_requests"] == 0

    def test_replay_failure_is_poisoned_without_eager_work(self):
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024)
        graph = _FakeGraph(fail_replay=True)
        cache._capture_graph = mock.Mock(return_value=(graph, object(), 0))
        cache._record_last_use = mock.Mock()
        with pytest.raises(RuntimeError, match="replay failed"):
            cache.execute(
                key=("bucket",),
                request_count=2,
                fingerprint=("pointers",),
                workspace=_cache_workspace(),
                capture_body=mock.Mock(),
            )
        assert cache.counts["replay_failure"] == 1
        assert cache.counts["failure"] == 1
        assert cache.counts["covered_requests"] == 0
        assert cache.counts["rejected"] == 0

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
            == "rejected"
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
        cache.record_rejection(key=("bucket",), request_count=1)

        with pytest.raises(ValueError, match="request-count semantics"):
            cache.record_rejection(key=("bucket",), request_count=2)

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

        assert outcome == "rejected"
        assert graph.replays == 0
        assert graph.resets == 1
        assert cache.snapshot()["active_entries"] == 0


class TestCuTEDSLGraphSelection:
    @staticmethod
    def _scores(width, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.rand((2, width), generator=generator, dtype=torch.float32)

    @staticmethod
    def _boundary_selection(kind, *, exact_tie, device):
        if kind == "row":
            scores = _boundary_row_scores(exact_tie=exact_tie, device=device)
            workspace = _BatchedFixedPerHeadWorkspace(
                eviction_mode="per_head",
                dense_layers=(0,),
                num_query_heads=1,
                num_kv_heads=1,
                width=8,
                keep_count=4,
                prompt_len=0,
                dtype=torch.float32,
                device=device,
                selection_backend="cute_dsl_topk",
                max_requests=1,
            )
            segments = [[scores.view(1, -1)]]

            def select():
                workspace.select_requests(segments, normalize_scores=False)

            expected = torch.tensor(
                [0, 1, 2, 3 if exact_tie else 4],
                dtype=torch.int32,
                device=device,
            )

            def result():
                return workspace.keep[0, 0]

            return workspace, select, result, expected

        if kind != "final_union":
            raise ValueError(f"unsupported boundary selection kind: {kind}")
        scores = _boundary_union_scores(exact_tie=exact_tie, device=device)
        workspace = _BatchedFixedUnionWorkspace(
            rows=2,
            width=8,
            keep_count=4,
            prompt_len=0,
            dtype=torch.float32,
            device=device,
            selection_backend="cute_dsl_topk",
            max_requests=1,
        )
        segments = [[scores]]

        def select():
            workspace.select_requests(segments, normalize_scores=False)

        expected = torch.tensor(
            [0, 1, 2, 3 if exact_tie else 5],
            dtype=torch.long,
            device=device,
        )

        def result():
            return workspace.keep[0]

        return workspace, select, result, expected

    @pytest.mark.parametrize("kind", ["row", "final_union"])
    @pytest.mark.parametrize("exact_tie", [True, False], ids=("exact_tie", "one_ulp"))
    def test_boundary_selection_has_stable_identity_with_adversarial_raw_topk(
        self, kind, exact_tie
    ):
        _, select, result, expected = self._boundary_selection(
            kind,
            exact_tie=exact_tie,
            device=torch.device("cpu"),
        )
        with mock.patch.object(
            torch.ops.trtllm,
            "cute_dsl_indexer_topk_decode",
            side_effect=_AlternatingBoundaryTieTopK(),
            create=True,
        ):
            for _ in range(6):
                select()
                assert torch.equal(result(), expected)

    @pytest.mark.parametrize("kind", ["row", "final_union"])
    @pytest.mark.parametrize("exact_tie", [True, False], ids=("exact_tie", "one_ulp"))
    @CUDA_REQUIRED
    def test_boundary_selection_is_stable_in_eager_and_cuda_graph(self, kind, exact_tie):
        device = torch.device("cuda")
        _, eager_select, eager_result, expected = self._boundary_selection(
            kind,
            exact_tie=exact_tie,
            device=device,
        )
        _, graph_select, graph_result, _ = self._boundary_selection(
            kind,
            exact_tie=exact_tie,
            device=device,
        )

        eager_select()
        graph_select()
        torch.cuda.synchronize(device)
        for _ in range(6):
            eager_select()
            torch.cuda.synchronize(device)
            assert torch.equal(eager_result(), expected)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_select()
        torch.cuda.synchronize(device)
        assert torch.equal(graph_result(), expected)
        for _ in range(6):
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(graph_result(), expected)

    @pytest.mark.parametrize(
        "keep_count,width",
        [(4096, 4224), (8192, 9216)],
        ids=("k4096", "k8192"),
    )
    def test_cross_request_graph_workspace_has_no_legacy_topk_fallback(self, keep_count, width):
        prompt_len = 17
        request_scores = [
            self._scores(width, seed=keep_count + request_index + 1) for request_index in range(2)
        ]
        expected = [
            torch.cat(
                (
                    torch.arange(prompt_len, dtype=torch.long),
                    _union_keep_oracle(scores, keep_count) + prompt_len,
                )
            )
            for scores in request_scores
        ]
        workspace = _BatchedFixedUnionWorkspace(
            rows=2,
            width=width,
            keep_count=keep_count,
            prompt_len=prompt_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=2,
        )

        with (
            mock.patch.object(
                torch.ops.trtllm,
                "cute_dsl_indexer_topk_decode",
                side_effect=_fake_cute_dsl_topk_decode,
                create=True,
            ) as cute_topk,
            mock.patch.object(
                torch.ops.trtllm,
                "indexer_topk_decode",
                side_effect=AssertionError("legacy IndexerTopK fallback was called"),
                create=True,
            ),
            mock.patch.object(
                torch,
                "topk",
                side_effect=AssertionError("torch.topk fallback was called"),
            ),
        ):
            workspace.select_requests(
                [[scores] for scores in request_scores],
                normalize_scores=False,
            )
            actual = workspace.keep[: len(request_scores)].clone()

        assert cute_topk.call_count == 4
        assert all(torch.equal(result, oracle) for result, oracle in zip(actual, expected))

    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    def test_fixed_per_head_selection_matches_independent_oracle(self, eviction_mode):
        request_count = 2
        dense_layers = (3, 1)
        num_query_heads = 4
        num_kv_heads = 2
        group_size = num_query_heads // num_kv_heads
        prompt_len = 2
        width = 9
        keep_count = 3
        valid_widths = (9, 7)
        generator = torch.Generator().manual_seed(1234)
        segments_by_request = [
            [torch.randn(num_query_heads, width, generator=generator) for _ in dense_layers]
            for _ in range(request_count)
        ]
        workspace = _BatchedFixedPerHeadWorkspace(
            eviction_mode=eviction_mode,
            dense_layers=dense_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            width=width,
            keep_count=keep_count,
            prompt_len=prompt_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=request_count,
        )
        workspace.stage_valid_widths_from_seq_lens(
            torch.tensor(
                [prompt_len + valid_width for valid_width in valid_widths],
                dtype=torch.int32,
            ),
            request_count,
        )

        expected = []
        for request_index, segments in enumerate(segments_by_request):
            layer_scores = []
            for scores in segments:
                scores = scores[:, : valid_widths[request_index]]
                scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
                    dim=1,
                    unbiased=False,
                    keepdim=True,
                ).clamp_min(1e-6)
                layer_scores.append(scores.view(num_kv_heads, group_size, -1).max(dim=1).values)
            selection_scores = torch.stack(layer_scores)
            if eviction_mode == "per_head":
                selection_scores = selection_scores.mean(dim=0)
            else:
                selection_scores = selection_scores.reshape(-1, valid_widths[request_index])
            selected = torch.topk(selection_scores, keep_count, dim=1).indices.sort(dim=1).values
            prompt = torch.arange(prompt_len).view(1, -1).expand(selected.shape[0], -1)
            expected.append(torch.cat((prompt, selected + prompt_len), dim=1).to(torch.int32))

        pointers = tuple(tensor.data_ptr() for _, tensor in workspace.named_tensors())
        with mock.patch.object(
            torch.ops.trtllm,
            "cute_dsl_indexer_topk_decode",
            side_effect=_fake_cute_dsl_topk_decode,
            create=True,
        ):
            workspace.select_requests(
                segments_by_request,
                normalize_scores=True,
            )

        assert pointers == tuple(tensor.data_ptr() for _, tensor in workspace.named_tensors())
        actual = workspace.keep[:request_count]
        if eviction_mode == "per_layer_perhead":
            actual = actual.view(request_count, len(dense_layers), num_kv_heads, -1)
            expected = [item.view(len(dense_layers), num_kv_heads, -1) for item in expected]
        for result, oracle in zip(actual, expected):
            assert torch.equal(result, oracle)


class TestStandaloneGraphBuckets:
    @staticmethod
    def _mark_ready(manager, score, selection):
        key = score.prewarm_key
        selection.eviction_mode = manager.eviction_mode
        selection.dense_layers = (0,)
        selection.num_query_heads = 1
        selection.num_kv_heads = 1
        score.bucket_seq_len = selection.prompt_len + selection.width
        score.page_table_token_capacity = (
            score.bucket_seq_len + manager._configured_protected_tail_capacity()
        )
        score.valid_seq_lens_device = torch.full(
            (score.max_requests,),
            score.bucket_seq_len,
            dtype=torch.int32,
        )
        selection.stage_valid_widths_from_seq_lens = mock.Mock()
        manager._eviction_buckets[key] = _EvictionBucketResources(
            score_state="ready",
            selection_state="ready",
            score_workspace=score,
            selection_workspace=selection,
        )

    @pytest.mark.parametrize(
        "prompt_len,width,budget,backend,request_count",
        [
            (0, 2175, 2048, "cute_dsl_topk", 1),
            (256, 3072, 2048, "cute_dsl_topk", 2),
            (1024, 4095, 2048, "cute_dsl_topk", 4),
            (1024, 4096, 2048, "cute_dsl_topk", 7),
            (1536, 4223, 4096, "cute_dsl_topk", 8),
            (1024, 4224, 4096, "cute_dsl_topk", 8),
            (1024, 5119, 4096, "cute_dsl_topk", 16),
            (1024, 8191, 4096, "cute_dsl_topk", 1),
            (2048, 8192, 4096, "cute_dsl_topk", 31),
            (1024, 9216, 8192, "cute_dsl_topk", 32),
        ],
    )
    def test_ready_upper_bucket_is_accepted(
        self, prompt_len, width, budget, backend, request_count
    ):
        manager = _make_triattention()
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            _prepared_eviction(
                seq_len=prompt_len + width,
                prompt_len=prompt_len,
                expected_keep_count=prompt_len + budget,
            )
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
        assert key[1] == "union"
        assert key[2] == score.prewarm_key
        assert key[3:8] == (
            request_count,
            prompt_len + width,
            prompt_len,
            budget,
            backend,
        )

    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    def test_per_head_modes_publish_distinct_fixed_graph_keys(self, eviction_mode):
        manager = _make_triattention(eviction_mode)
        selection = _BatchedFixedPerHeadWorkspace(
            eviction_mode=eviction_mode,
            dense_layers=(2,),
            num_query_heads=2,
            num_kv_heads=1,
            width=12,
            keep_count=8,
            prompt_len=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            selection_backend="cute_dsl_topk",
            max_requests=2,
        )
        score = SimpleNamespace(
            prewarm_key=("per-head", eviction_mode),
            max_requests=2,
            bucket_seq_len=14,
            page_table_token_capacity=15,
        )
        manager._eviction_buckets[score.prewarm_key] = _EvictionBucketResources(
            score_state="ready",
            selection_state="ready",
            score_workspace=score,
            selection_workspace=selection,
        )
        prepared = [
            _prepared_eviction(
                seq_len=14,
                prompt_len=2,
                expected_keep_count=10,
            )
            for _ in range(2)
        ]

        key = manager._standalone_graph_bucket_for(prepared, score, selection)

        assert key is not None
        assert key[1] == eviction_mode
        assert key[8:11] == ((2,), 2, 1)

    def test_different_valid_lengths_share_one_upper_bucket(self):
        manager = _make_triattention()
        manager.top_B = 8
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            _prepared_eviction(
                seq_len=seq_len,
                prompt_len=2,
                expected_keep_count=10,
            )
            for seq_len in (11, 13)
        ]
        score = SimpleNamespace(prewarm_key=("upper",), max_requests=8)
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
            max_requests=8,
            prompt_len=2,
            width=12,
            keep_count=8,
        )
        self._mark_ready(manager, score, selection)

        key = manager._standalone_graph_bucket_for(prepared, score, selection)

        assert key is not None
        assert key[3:8] == (2, 14, 2, 8, "cute_dsl_topk")

    def test_protected_tail_geometry_is_explicit_in_graph_key(self):
        manager = _make_triattention()
        manager.top_B = 8
        manager.kv_cache_manager._kv_reserve_draft_tokens = 3
        manager._standalone_cuda_graph_enabled = True
        score = SimpleNamespace(prewarm_key=("tail",), max_requests=8)
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
            max_requests=8,
            prompt_len=2,
            width=12,
            keep_count=8,
        )
        self._mark_ready(manager, score, selection)

        def prepared(tails):
            return [
                _prepared_eviction(
                    seq_len=14,
                    prompt_len=2,
                    expected_keep_count=10,
                    protected_tail=tail,
                )
                for tail in tails
            ]

        first = manager._standalone_graph_bucket_for(prepared([1, 3]), score, selection)
        same = manager._standalone_graph_bucket_for(prepared([1, 3]), score, selection)
        different = manager._standalone_graph_bucket_for(prepared([2, 2]), score, selection)

        assert first == same
        assert first is not None
        assert first != different
        assert first[-2:] == ((1, 3), 18)
        assert manager._standalone_graph_bucket_for(prepared([1, 5]), score, selection) is None

    def test_old_backend_boundary_shares_one_cute_upper_bucket(self):
        manager = _make_triattention()
        manager.top_B = 2048
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            _prepared_eviction(
                seq_len=width,
                prompt_len=0,
                expected_keep_count=2048,
            )
            for width in (4095, 4096)
        ]
        score = SimpleNamespace(prewarm_key=("cute-upper",), max_requests=8)
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
            max_requests=8,
            prompt_len=0,
            width=4096,
            keep_count=2048,
        )
        self._mark_ready(manager, score, selection)

        key = manager._standalone_graph_bucket_for(prepared, score, selection)

        assert key is not None
        assert key[3:8] == (2, 4096, 0, 2048, "cute_dsl_topk")

    @pytest.mark.parametrize(
        "budget,beta,bucket_width,valid_widths,backend",
        [
            (32, 4, 36, [34, 36], "cute_dsl_topk"),
            (4096, 128, 4228, [4225, 4228], "cute_dsl_topk"),
            (4096, 4096, 8196, [8193, 8196], "cute_dsl_topk"),
        ],
    )
    def test_budget_beta_k4_graph_key_is_shape_stable_and_overflow_rejected(
        self, budget, beta, bucket_width, valid_widths, backend
    ):
        prompt_len = 1024
        manager = _make_triattention()
        manager.top_B = budget
        manager.beta = beta
        manager._standalone_cuda_graph_enabled = True
        score = SimpleNamespace(
            prewarm_key=("configured-upper", budget, beta, bucket_width),
            max_requests=8,
        )
        selection = SimpleNamespace(
            selection_backend=backend,
            max_requests=8,
            prompt_len=prompt_len,
            width=bucket_width,
            keep_count=budget,
        )
        self._mark_ready(manager, score, selection)

        def prepared(widths):
            return [
                _prepared_eviction(
                    seq_len=prompt_len + width,
                    prompt_len=prompt_len,
                    expected_keep_count=prompt_len + budget,
                )
                for width in widths
            ]

        first_key = manager._standalone_graph_bucket_for(prepared(valid_widths), score, selection)
        second_key = manager._standalone_graph_bucket_for(
            prepared([width - 1 for width in valid_widths]), score, selection
        )

        assert first_key == second_key
        assert first_key[4:8] == (
            prompt_len + bucket_width,
            prompt_len,
            budget,
            backend,
        )
        assert (
            manager._standalone_graph_bucket_for(prepared([bucket_width + 1]), score, selection)
            is None
        )

    def test_budget2048_uses_one_backend_with_distinct_shape_keys(self):
        prompt_len = 1024
        budget = 2048
        manager = _make_triattention()
        manager.top_B = budget
        manager.beta = 2048
        manager._standalone_cuda_graph_enabled = True

        def key_for(bucket_width, valid_widths, backend):
            score = SimpleNamespace(
                prewarm_key=("configured-upper", budget, backend, bucket_width),
                max_requests=8,
            )
            selection = SimpleNamespace(
                selection_backend=backend,
                max_requests=8,
                prompt_len=prompt_len,
                width=bucket_width,
                keep_count=budget,
            )
            self._mark_ready(manager, score, selection)
            prepared = [
                _prepared_eviction(
                    seq_len=prompt_len + width,
                    prompt_len=prompt_len,
                    expected_keep_count=prompt_len + budget,
                )
                for width in valid_widths
            ]
            return manager._standalone_graph_bucket_for(prepared, score, selection)

        lower_key = key_for(4095, [4092, 4095], "cute_dsl_topk")
        upper_key = key_for(4100, [4096, 4100], "cute_dsl_topk")

        assert lower_key is not None
        assert upper_key is not None
        assert lower_key != upper_key
        assert lower_key[4:8] == (
            prompt_len + 4095,
            prompt_len,
            budget,
            "cute_dsl_topk",
        )
        assert upper_key[4:8] == (
            prompt_len + 4100,
            prompt_len,
            budget,
            "cute_dsl_topk",
        )

    @pytest.mark.parametrize(
        "request_count,prompt_len,width,budget,backend,ready",
        [
            (0, 1024, 4095, 2048, "cute_dsl_topk", True),
            (33, 1024, 4095, 2048, "cute_dsl_topk", True),
            (1, 1024, 4095, 2048, "torch_topk", True),
            (1, 1024, 4095, 2048, "indexer_topk", True),
            (1, 1024, 4095, 2048, "cute_dsl_topk", False),
        ],
    )
    def test_unready_capacity_or_backend_mismatch_falls_back(
        self, request_count, prompt_len, width, budget, backend, ready
    ):
        manager = _make_triattention()
        manager.top_B = budget
        manager._standalone_cuda_graph_enabled = True
        prepared = [
            _prepared_eviction(
                seq_len=prompt_len + width,
                prompt_len=prompt_len,
                expected_keep_count=prompt_len + budget,
            )
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
            _prepared_eviction(
                seq_len=5119,
                prompt_len=1024,
                expected_keep_count=3072,
            ),
            _prepared_eviction(
                seq_len=5375,
                prompt_len=1280,
                expected_keep_count=3328,
            ),
        ]
        score = SimpleNamespace(prewarm_key=("mixed",), max_requests=32)
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
            max_requests=32,
            prompt_len=1024,
            width=4095,
            keep_count=2048,
        )
        self._mark_ready(manager, score, selection)

        assert manager._standalone_graph_bucket_for(prepared, score, selection) is None
        manager._eviction_buckets[score.prewarm_key].selection_workspace = SimpleNamespace()
        prepared[1] = prepared[0]
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

    def test_unready_eviction_fails_closed_and_records_admission_rejection(self):
        manager = _make_triattention()
        manager._standalone_cuda_graph_enabled = True
        manager._standalone_graph_runtime_counts = {}
        prepared = [
            _prepared_eviction(
                seq_len=5119,
                prompt_len=1024,
                expected_keep_count=3072,
            )
        ]

        with pytest.raises(RuntimeError, match="rejected its configured runtime bucket"):
            manager._try_standalone_cuda_graph(
                prepared=prepared,
                layer_pools=[torch.empty(1)],
                dense_layers=[0],
                swa_layers=[],
                swa_window=None,
                layer_group_representative={0: 0},
                layer_pool_keys=[("pool", 0)],
                global_layers=[0],
                score_workspace=None,
                selection_workspace=None,
                fixed_perhead_segment_views=mock.Mock(),
            )
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 1,
            "attempt_requests": 1,
            "admission_rejected": 1,
            "admission_rejected_requests": 1,
        }

    def test_capture_body_covers_phase_score_select_compact_then_publishes(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager = _make_triattention()
        manager.top_B = 2048
        manager.normalize_scores = True
        manager.score_aggregation = "mean"
        manager._standalone_cuda_graph_enabled = True
        manager._request_states = {7: _RequestCompressionState(confirmed_kv_length=3328)}
        prepared = [
            _prepared_eviction(
                request_id=7,
                seq_len=3328,
                prompt_len=256,
                expected_keep_count=2304,
            )
        ]
        score_output = torch.zeros(1, 3328)
        group = SimpleNamespace(launch=mock.Mock(return_value=(score_output, None)))
        score = SimpleNamespace(
            prewarm_key=("dynamic-indexer",),
            max_requests=32,
            prepare_phase=mock.Mock(),
            fused_group=group,
            dense_layer_order=[0],
            round_starts_device=torch.zeros(8),
            mean_cos=torch.zeros(8, 1),
            mean_sin=torch.zeros(8, 1),
        )
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
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
                swa_layers=[],
                swa_window=None,
                layer_group_representative={0: 0},
                layer_pool_keys=[("pool", 0)],
                global_layers=[0],
                score_workspace=score,
                selection_workspace=selection,
                fixed_perhead_segment_views=fixed_views,
            )

        assert targets == [(7, 2304)]
        score.prepare_phase.assert_called_once_with(1)
        group.launch.assert_called_once()
        selection.stage_valid_widths_from_seq_lens.assert_called_once_with(
            score.valid_seq_lens_device,
            1,
        )
        selected_segments = selection.select_requests.call_args.args[0]
        assert selected_segments[0][0].shape == (1, 3072)
        workspace.launch.assert_called_once_with()
        assert manager._request_states[7].evicted_tokens == 1024
        assert manager._request_states[7].confirmed_kv_length == 2304
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 1,
            "attempt_requests": 1,
            "success": 1,
            "success_requests": 1,
        }

    def test_graph_rejection_fails_closed_without_publishing(self):
        manager = _make_triattention()
        manager.top_B = 4096
        manager._standalone_cuda_graph_enabled = True
        manager._request_states = {7: _RequestCompressionState(confirmed_kv_length=9215)}
        prepared = [
            _prepared_eviction(
                request_id=7,
                seq_len=9215,
                prompt_len=1024,
                expected_keep_count=5120,
            )
        ]
        score = SimpleNamespace(prewarm_key=("formal-a-first",), max_requests=8)
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
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
            execute=mock.Mock(return_value="rejected"),
            snapshot=mock.Mock(return_value={"rejected": 1}),
        )
        manager._standalone_graph_workspace_for = mock.Mock(return_value=workspace)
        manager._standalone_graph_cache_for = mock.Mock(return_value=cache)
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score.device = torch.device("cpu")
        score.stream = stream

        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            with pytest.raises(RuntimeError, match="execution was rejected"):
                manager._try_standalone_cuda_graph(
                    prepared=prepared,
                    layer_pools=[torch.empty(1)],
                    dense_layers=[0],
                    swa_layers=[],
                    swa_window=None,
                    layer_group_representative={0: 0},
                    layer_pool_keys=[("pool", 0)],
                    global_layers=[0],
                    score_workspace=score,
                    selection_workspace=selection,
                    fixed_perhead_segment_views=mock.Mock(),
                )
        assert manager._request_states[7].evicted_tokens == 0
        assert manager._request_states[7].confirmed_kv_length == 9215

    def test_capture_failure_disables_semantic_bucket_before_new_arena(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        manager = _make_triattention()
        manager.top_B = 4096
        manager._standalone_cuda_graph_enabled = True
        manager._request_states = {7: _RequestCompressionState(confirmed_kv_length=9215)}
        prepared = [
            _prepared_eviction(
                request_id=7,
                seq_len=9215,
                prompt_len=1024,
                expected_keep_count=5120,
            )
        ]
        stream = SimpleNamespace(device=torch.device("cpu"), cuda_stream=9)
        score = SimpleNamespace(
            prewarm_key=("formal-a-first",),
            max_requests=8,
            device=torch.device("cpu"),
            stream=stream,
        )
        selection = SimpleNamespace(
            selection_backend="cute_dsl_topk",
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
            swa_layers=[],
            swa_window=None,
            layer_group_representative={0: 0},
            layer_pool_keys=[("pool", 0)],
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
            with pytest.raises(RuntimeError, match="execution was rejected"):
                manager._try_standalone_cuda_graph(**kwargs)
            with pytest.raises(RuntimeError, match="bucket is disabled"):
                manager._try_standalone_cuda_graph(**kwargs)

        manager._standalone_graph_workspace_for.assert_called_once()
        cache._capture_graph.assert_called_once()
        assert cache.snapshot()["capture_failure"] == 1
        assert cache.snapshot()["attempt"] == 2
        assert cache.snapshot()["rejected"] == 2
        assert cache.snapshot()["covered_requests"] == 0
        assert cache.snapshot()["disabled_buckets"] == 1
        assert manager._standalone_graph_runtime_counts == {
            "attempt": 2,
            "attempt_requests": 2,
            "rejected": 2,
            "rejected_requests": 2,
        }


class TestFixedBatchedCompactionWorkspace:
    def test_dense_copy_excludes_prompt_and_swa_uses_independent_table(self):
        pools = [
            torch.zeros(4, 2, 1, 4, 2),
            torch.zeros(4, 2, 1, 4, 2),
        ]
        score = SimpleNamespace(
            page_count=3,
            representative_slots={0: 0, 1: 1},
            page_ids_device=torch.tensor([[[3, 1, 0]], [[2, 0, 1]]], dtype=torch.int64),
            valid_seq_lens_device=torch.tensor([8], dtype=torch.int32),
        )
        selection = SimpleNamespace(
            eviction_mode="union",
            dense_layers=(0,),
            num_kv_heads=1,
            max_requests=1,
            prompt_len=2,
            keep_count=4,
            width=6,
            keep=torch.tensor([[0, 1, 2, 4, 5, 7]], dtype=torch.int64),
        )
        workspace = FixedBatchedCompactionWorkspace(
            eviction_mode="union",
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
            protected_tail_lengths=[2],
        )

        with mock.patch(
            "tensorrt_llm._torch.kv_cache_compression.triattention.cuda_graph._run_cpp_compact_layers"
        ) as run:
            workspace.launch()

        assert run.call_count == 2
        assert workspace.cpp_indices.tolist() == [[0, 1, 2, 4, 5, 7, 8, 9]]
        assert workspace.cpp_offsets.tolist() == [0, 8]
        assert workspace.cpp_page_tables[0][1].tolist() == [[3, 1, 0]]
        assert workspace.cpp_page_tables[1][1].tolist() == [[2, 0, 1]]
        assert workspace.swa_source.tolist() == [6, 7, 8, 9]
        assert workspace.swa_indices.tolist() == [[6, 7, 8, 9]]
        assert workspace.swa_destination.tolist() == [4, 5, 6, 7]
        assert run.call_args_list[0].args[-1] is None

    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    def test_mode_specific_keep_sets_are_packed_per_layer_and_head(self, eviction_mode):
        request_count = 2
        dense_layers = [0, 1]
        num_kv_heads = 2
        prompt_len = 1
        decode_keep_count = 2
        keep_count = prompt_len + decode_keep_count
        pools = [torch.zeros(4, 2, num_kv_heads, 4, 2) for _ in dense_layers]
        score = SimpleNamespace(
            representative_slots={0: 0, 1: 1},
            page_ids_device=torch.tensor([[[0, 1], [2, 3]], [[1, 0], [3, 2]]], dtype=torch.int64),
            valid_seq_lens_device=torch.tensor([5, 5], dtype=torch.int32),
        )
        if eviction_mode == "per_head":
            keep = torch.tensor(
                [
                    [[0, 1, 3], [0, 2, 4]],
                    [[0, 1, 4], [0, 2, 3]],
                ],
                dtype=torch.int32,
            )
        else:
            keep = torch.tensor(
                [
                    [[[0, 1, 3], [0, 2, 4]], [[0, 1, 4], [0, 2, 3]]],
                    [[[0, 2, 3], [0, 1, 4]], [[0, 2, 4], [0, 1, 3]]],
                ],
                dtype=torch.int32,
            ).view(request_count, len(dense_layers) * num_kv_heads, keep_count)
        selection = SimpleNamespace(
            eviction_mode=eviction_mode,
            dense_layers=tuple(dense_layers),
            num_kv_heads=num_kv_heads,
            max_requests=request_count,
            prompt_len=prompt_len,
            keep_count=decode_keep_count,
            width=4,
            keep=keep,
        )
        workspace = FixedBatchedCompactionWorkspace(
            eviction_mode=eviction_mode,
            layer_pools=pools,
            dense_layers=dense_layers,
            swa_layers=[],
            layer_group_representative={0: 0, 1: 1},
            global_layers=dense_layers,
            score_workspace=score,
            selection_workspace=selection,
            request_count=request_count,
            seq_len=5,
            prompt_len=prompt_len,
            decode_keep_count=decode_keep_count,
            swa_window=None,
            arena_generation=1,
            layer_pool_keys=[("pool", 0), ("pool", 0)],
        )
        launched_sources = []

        def record_source(group, source, _offsets, _destination):
            if source.ndim == 2:
                launched_sources.extend(source.clone() for _ in group.layers)
                return
            assert group.source_layer_indices is not None
            launched_sources.extend(
                source[layer_slot].clone() for layer_slot in group.source_layer_indices.tolist()
            )

        with mock.patch(
            "tensorrt_llm._torch.kv_cache_compression.triattention.cuda_graph._run_cpp_compact_layers",
            side_effect=record_source,
        ) as run:
            workspace.launch()

        assert run.call_count == 1
        assert len(launched_sources) == len(dense_layers)
        for layer_slot, actual in enumerate(launched_sources):
            if eviction_mode == "per_head":
                expected = keep.permute(1, 0, 2).reshape(num_kv_heads, -1)
            else:
                expected = (
                    keep.view(request_count, len(dense_layers), num_kv_heads, keep_count)[
                        :, layer_slot
                    ]
                    .permute(1, 0, 2)
                    .reshape(num_kv_heads, -1)
                )
            assert torch.equal(actual, expected)

    def test_ragged_protected_tails_are_appended_to_each_request_segment(self):
        request_count = 2
        pool = torch.zeros(8, 2, 1, 4, 2)
        score = SimpleNamespace(
            representative_slots={0: 0},
            page_ids_device=torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.int64),
            valid_seq_lens_device=torch.tensor([8, 9], dtype=torch.int32),
        )
        selection = SimpleNamespace(
            eviction_mode="union",
            dense_layers=(0,),
            num_kv_heads=1,
            max_requests=request_count,
            prompt_len=1,
            keep_count=2,
            width=8,
            keep=torch.tensor([[0, 2, 7], [0, 3, 8]], dtype=torch.int32),
        )
        workspace = FixedBatchedCompactionWorkspace(
            eviction_mode="union",
            layer_pools=[pool],
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score,
            selection_workspace=selection,
            request_count=request_count,
            seq_len=9,
            prompt_len=1,
            decode_keep_count=2,
            swa_window=None,
            arena_generation=1,
            protected_tail_lengths=[1, 3],
        )

        with mock.patch(
            "tensorrt_llm._torch.kv_cache_compression.triattention.cuda_graph._run_cpp_compact_layers"
        ) as run:
            workspace.launch()

        assert run.call_count == 1
        assert workspace.cpp_offsets.tolist() == [0, 4, 10]
        assert workspace.cpp_indices.tolist() == [[0, 2, 7, 8, 0, 3, 8, 9, 10, 11]]
        assert run.call_args.args[-1] is None

    @CUDA_REQUIRED
    def test_graph_compaction_preserves_ragged_protected_tail_bytes(self):
        device = torch.device("cuda")
        request_count = 2
        num_heads = 2
        tokens_per_block = 4
        head_dim = 16
        page_table = torch.tensor(
            [[2, 0, 3, 1], [6, 4, 7, 5]],
            dtype=torch.int64,
            device=device,
        )
        values = torch.arange(
            8 * 2 * num_heads * tokens_per_block * head_dim,
            dtype=torch.float32,
            device=device,
        )
        pool = values.view(8, 2, num_heads, tokens_per_block, head_dim)
        before = self._logical_tokens(pool.clone(), page_table)
        seq_lens = [10, 11]
        tail_lengths = [2, 3]
        keep = torch.tensor(
            [[0, 1, 3, 6, 9], [0, 1, 4, 8, 10]],
            dtype=torch.int32,
            device=device,
        )
        score = SimpleNamespace(
            representative_slots={0: 0},
            page_ids_device=page_table.view(1, request_count, -1),
            valid_seq_lens_device=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        )
        selection = SimpleNamespace(
            eviction_mode="union",
            dense_layers=(0,),
            num_kv_heads=num_heads,
            max_requests=request_count,
            prompt_len=2,
            keep_count=3,
            width=9,
            keep=keep,
        )
        workspace = FixedBatchedCompactionWorkspace(
            eviction_mode="union",
            layer_pools=[pool],
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            global_layers=[0],
            score_workspace=score,
            selection_workspace=selection,
            request_count=request_count,
            seq_len=11,
            prompt_len=2,
            decode_keep_count=3,
            swa_window=None,
            arena_generation=1,
            protected_tail_lengths=tail_lengths,
        )

        workspace.launch()
        torch.cuda.synchronize()
        after = self._logical_tokens(pool, page_table)

        for request_index, (seq_len, tail_length) in enumerate(zip(seq_lens, tail_lengths)):
            source = torch.cat(
                (
                    keep[request_index].to(torch.long),
                    torch.arange(
                        seq_len,
                        seq_len + tail_length,
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )
            expected = before[request_index].index_select(2, source)
            torch.testing.assert_close(
                after[request_index, :, :, : source.numel()],
                expected,
                rtol=0,
                atol=0,
            )

    @staticmethod
    def _logical_tokens(pool, page_table):
        return (
            pool[page_table]
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(page_table.shape[0], 2, pool.shape[2], -1, pool.shape[4])
        )

    @pytest.mark.parametrize("head_dim", [80, 96, 128, 192])
    @CUDA_REQUIRED
    def test_cpp_only_compact_multilayer_perhead_destination_and_reuse(self, head_dim):
        """Exercise every TriAttention move shape against a clone oracle."""
        device = torch.device("cuda")
        request_count = 2
        num_heads = 3
        tokens_per_block = 4
        page_tables = (
            torch.tensor([[2, 0, 1], [5, 3, 4]], dtype=torch.int64, device=device),
            torch.tensor([[1, 2, 0], [4, 5, 3]], dtype=torch.int64, device=device),
        )
        pools = []
        for layer in range(2):
            values = torch.arange(
                6 * 2 * num_heads * tokens_per_block * head_dim,
                dtype=torch.float32,
                device=device,
            )
            pools.append(
                values.view(6, 2, num_heads, tokens_per_block, head_dim) + layer * 1_000_000
            )

        # Each layer, request, and head has a distinct sorted keep set. The
        # wrapper must preserve this 2-D source layout rather than replicating
        # a union row.
        layer_sources = (
            [
                torch.tensor(
                    [[0, 2, 4, 6, 8], [0, 1, 5, 7, 8], [1, 3, 4, 7, 8]],
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor(
                    [[0, 2, 4, 6, 7], [0, 1, 3, 5, 7], [1, 2, 4, 6, 7]],
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            [
                torch.tensor(
                    [[0, 1, 3, 5, 8], [1, 2, 4, 6, 8], [0, 3, 5, 7, 8]],
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor(
                    [[0, 1, 4, 6, 7], [1, 2, 3, 6, 7], [0, 2, 5, 6, 7]],
                    dtype=torch.int64,
                    device=device,
                ),
            ],
        )
        seq_lens = [9, 8]
        for layer, (pool, page_table, sources) in enumerate(zip(pools, page_tables, layer_sources)):
            before = self._logical_tokens(pool, page_table).clone()
            _cpp_sparse_compact_reference(
                pool,
                [page_table[request] for request in range(request_count)],
                sources,
                seq_lens,
            )
            after = self._logical_tokens(pool, page_table)
            for request, source in enumerate(sources):
                for head in range(num_heads):
                    assert torch.equal(
                        after[request, :, head, : source.shape[1]],
                        before[request, :, head].index_select(1, source[head]),
                    ), f"layer={layer}, request={request}, head={head}"

        # Refill reused physical pages and move a protected/SWA-like suffix to
        # nonzero destinations. Reversing each page table ensures the second
        # round does not accidentally rely on first-round physical ordering.
        destination = torch.tensor([3, 4], dtype=torch.int64, device=device)
        reused_tables = tuple(table.flip(1).contiguous() for table in page_tables)
        suffix_sources = [
            torch.tensor([6, 7], dtype=torch.int64, device=device),
            torch.tensor([5, 6], dtype=torch.int64, device=device),
        ]
        for layer, (pool, page_table) in enumerate(zip(pools, reused_tables)):
            pool.copy_(
                torch.arange(pool.numel(), dtype=pool.dtype, device=device).view_as(pool)
                + (layer + 2) * 1_000_000
            )
            before = self._logical_tokens(pool, page_table).clone()
            _cpp_sparse_compact_reference(
                pool,
                [page_table[request] for request in range(request_count)],
                suffix_sources,
                [8, 7],
                dest_list=[destination, destination],
            )
            after = self._logical_tokens(pool, page_table)
            for request, source in enumerate(suffix_sources):
                assert torch.equal(
                    after[request].index_select(2, destination),
                    before[request].index_select(2, source),
                ), f"reuse layer={layer}, request={request}"

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

        selection = SimpleNamespace(selection_backend="cute_dsl_topk", keep=captured)

        def named_tensors():
            return (("keep", selection.keep),)

        selection.named_tensors = named_tensors
        score_tensor = _FakeTensor(
            storage_cdata=3,
            storage_data_ptr=12288,
            data_ptr=12288,
        )
        fused_group = SimpleNamespace(
            pointer_prefix=(score_tensor,),
            pointer_middle=(score_tensor,),
            pointer_tail=(score_tensor,),
            output=score_tensor,
            seg_offsets=score_tensor,
        )
        score = SimpleNamespace(
            page_ids_device=score_tensor,
            round_starts_device=score_tensor,
            valid_seq_lens_device=score_tensor,
            phase_base=score_tensor,
            phase=score_tensor,
            cos_phase=score_tensor,
            sin_phase=score_tensor,
            mean_cos=score_tensor,
            mean_sin=score_tensor,
            offsets=score_tensor,
            omega=score_tensor,
            fused_group=fused_group,
        )
        workspace = object.__new__(FixedBatchedCompactionWorkspace)
        workspace.eviction_mode = "union"
        workspace.selection_workspace = selection
        workspace.score_workspace = score
        workspace.request_count = 1
        workspace.seq_len = 8
        workspace.prompt_len = 2
        workspace.decode_keep_count = 4
        workspace.protected_tail_lengths = (0,)
        workspace.dense_layers = ()
        workspace.swa_layers = ()
        workspace.global_layers = ()
        workspace.layer_pool_keys = ()
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
        selection_workspace = SimpleNamespace(eviction_mode="union")
        workspace = object.__new__(FixedBatchedCompactionWorkspace)
        workspace.eviction_mode = "union"
        workspace.score_workspace = score_workspace
        workspace.selection_workspace = selection_workspace
        workspace.layer_pools = (captured,)
        workspace.dense_layers = (0,)
        workspace.swa_layers = ()
        workspace.global_layers = (0,)
        workspace.layer_pool_keys = (("layer", 0),)
        workspace.storage_groups = ((0, 0),)

        runtime = dict(
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            layer_pool_keys=[("layer", 0)],
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
            dense_layers=(0,),
            num_query_heads=1,
            num_kv_heads=1,
        )
        compaction = FixedBatchedCompactionWorkspace(
            eviction_mode="union",
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

        active_seq_lens = [seq_len] * request_count

        def stage(round_shift=0, reverse_pages=False, valid_seq_lens=None):
            if valid_seq_lens is None:
                valid_seq_lens = [seq_len] * request_count
            active_seq_lens[:] = valid_seq_lens
            full_tables = [list(reversed(row)) if reverse_pages else list(row) for row in tables]
            live_page_counts = [
                (valid_seq_len + tokens_per_block - 1) // tokens_per_block
                for valid_seq_len in valid_seq_lens
            ]
            runtime_tables = [
                row[:live_page_count] for row, live_page_count in zip(full_tables, live_page_counts)
            ]
            round_starts = [float(seq_len + round_shift + request * 17) for request in request_ids]
            assert score.stage(
                lambda _ids, _layer, num_blocks_per_seq=None: runtime_tables,
                request_ids,
                round_starts,
                valid_seq_lens,
            )
            return runtime_tables

        def score_and_select():
            score.prepare_phase(request_count)
            per_head, _ = score.fused_group.launch(
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
            selection.stage_valid_widths_from_seq_lens(
                score.valid_seq_lens_device,
                request_count,
            )
            selection.select_requests(segments, normalize_scores=True)

        def body():
            score_and_select()
            compaction.launch()

        def stage4_body():
            score_and_select()
            _cpp_sparse_compact_reference(
                pool,
                [score.page_ids_device[0, request] for request in range(request_count)],
                [selection.keep[request] for request in range(request_count)],
                active_seq_lens,
            )

        return pool, score, selection, compaction, stage, body, stage4_body

    @pytest.mark.parametrize(
        "prompt_len,width,budget,backend,request_count",
        [
            (0, 2175, 2048, "cute_dsl_topk", 1),
            (256, 3072, 2048, "cute_dsl_topk", 2),
            (1024, 4095, 2048, "cute_dsl_topk", 4),
            (1024, 4096, 2048, "cute_dsl_topk", 7),
            (1536, 4223, 4096, "cute_dsl_topk", 8),
            (1024, 4224, 4096, "cute_dsl_topk", 8),
            (1024, 5119, 4096, "cute_dsl_topk", 16),
            (1024, 8191, 4096, "cute_dsl_topk", 1),
            (2048, 8192, 4096, "cute_dsl_topk", 31),
            (1024, 9216, 8192, "cute_dsl_topk", 32),
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
        assert cache.snapshot()["rejected"] == 0
        assert cache.snapshot()["capture_failure"] == 0
        assert cache.snapshot()["replay_failure"] == 0

    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    @pytest.mark.parametrize("request_count", [1, 2])
    @CUDA_REQUIRED
    def test_per_head_graph_matches_eager_after_changed_input(self, eviction_mode, request_count):
        device = torch.device("cuda")
        prompt_len = 2
        width = 8
        budget = 4
        seq_len = prompt_len + width
        tokens_per_block = 4
        page_count = (seq_len + tokens_per_block - 1) // tokens_per_block
        total_pages = request_count * page_count
        initial_pools = [
            (
                torch.arange(
                    total_pages * 2 * 2 * tokens_per_block * 2,
                    dtype=torch.float32,
                    device=device,
                ).view(total_pages, 2, 2, tokens_per_block, 2)
                + layer * 10000.0
            )
            for layer in range(2)
        ]

        def build():
            pools = [pool.clone() for pool in initial_pools]
            q_real = (torch.arange(8, dtype=torch.float32, device=device).view(2, 4, 1) + 1.0) / 8.0
            q_imag = q_real.flip(1) * 0.25
            mlr = torch.full_like(q_real, 0.125)
            score = _FixedScoreMetadataWorkspace(
                pools,
                [[0], [1]],
                [0, 1],
                [0, 1],
                request_count,
                seq_len,
                4,
                1,
                q_real,
                q_imag,
                mlr,
                torch.ones(1, dtype=torch.float32, device=device),
                torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=device),
                torch.tensor([0.013], dtype=torch.float32, device=device),
                page_table_keys=[("pool", 0), ("pool", 1)],
                prompt_len=prompt_len,
            )
            selection = _BatchedFixedPerHeadWorkspace(
                eviction_mode=eviction_mode,
                dense_layers=(0, 1),
                num_query_heads=4,
                num_kv_heads=2,
                width=width,
                keep_count=budget,
                prompt_len=prompt_len,
                dtype=torch.float32,
                device=device,
                selection_backend="cute_dsl_topk",
                max_requests=request_count,
            )
            compaction = FixedBatchedCompactionWorkspace(
                eviction_mode=eviction_mode,
                layer_pools=pools,
                dense_layers=[0, 1],
                swa_layers=[],
                layer_group_representative={0: 0, 1: 1},
                global_layers=[0, 1],
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
            base_tables = [
                list(range(request * page_count, (request + 1) * page_count))
                for request in request_ids
            ]

            def stage(round_shift=0, reverse_pages=False):
                tables = [
                    list(reversed(row)) if reverse_pages else list(row) for row in base_tables
                ]
                round_starts = [
                    float(seq_len + round_shift + request * 11) for request in request_ids
                ]
                assert score.stage(
                    lambda _ids, _layer, num_blocks_per_seq=None: tables,
                    request_ids,
                    round_starts,
                    [seq_len] * request_count,
                )
                return tables

            def body():
                score.prepare_phase(request_count)
                per_head, _ = score.fused_group.launch(
                    request_count,
                    score.round_starts_device,
                    score.mean_cos,
                    score.mean_sin,
                    "mean",
                )
                views = fixed_perhead_segment_views(per_head, request_count, 2, seq_len)
                segments = [
                    [views[:, request, layer, prompt_len:seq_len] for layer in range(2)]
                    for request in range(request_count)
                ]
                selection.stage_valid_widths_from_seq_lens(
                    score.valid_seq_lens_device,
                    request_count,
                )
                selection.select_requests(segments, normalize_scores=True)
                compaction.launch()

            return pools, score, selection, compaction, stage, body

        eager = build()
        graphed = build()
        eager_pools, _, eager_selection, _, eager_stage, eager_body = eager
        graph_pools, _, graph_selection, graph_compaction, graph_stage, graph_body = graphed

        eager_stage()
        graph_stage()
        eager_body()
        graph_body()
        torch.cuda.synchronize(device)
        for eager_pool, graph_pool, initial in zip(eager_pools, graph_pools, initial_pools):
            eager_pool.copy_(initial)
            graph_pool.copy_(initial)

        assert eager_stage() == graph_stage()
        eager_body()
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=256 * 1024**2)
        fingerprint = graph_compaction.pointer_fingerprint(torch.cuda.current_stream(device))
        assert (
            cache.execute(
                key=(eviction_mode, request_count),
                request_count=request_count,
                fingerprint=fingerprint,
                workspace=graph_compaction,
                capture_body=graph_body,
            )
            == "capture"
        )
        torch.cuda.synchronize(device)
        assert torch.equal(eager_selection.keep, graph_selection.keep)
        for eager_pool, graph_pool in zip(eager_pools, graph_pools):
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))

        for eager_pool, graph_pool, initial in zip(eager_pools, graph_pools, initial_pools):
            eager_pool.copy_(initial)
            graph_pool.copy_(initial)
        assert eager_stage(round_shift=37, reverse_pages=True) == graph_stage(
            round_shift=37,
            reverse_pages=True,
        )
        eager_body()
        assert (
            cache.execute(
                key=(eviction_mode, request_count),
                request_count=request_count,
                fingerprint=fingerprint,
                workspace=graph_compaction,
                capture_body=graph_body,
            )
            == "replay"
        )
        torch.cuda.synchronize(device)
        assert torch.equal(eager_selection.keep, graph_selection.keep)
        for eager_pool, graph_pool in zip(eager_pools, graph_pools):
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
        snapshot = cache.snapshot()
        assert snapshot["capture"] == 1
        assert snapshot["cache_hit"] == 1
        assert snapshot["rejected"] == 0

    @CUDA_REQUIRED
    def test_ragged_upper_graph_matches_stage4_eager_across_replay(self):
        device = torch.device("cuda")
        prompt_len = 16
        width = 4224
        budget = 4096
        request_count = 2
        tokens_per_block = 128
        seq_len = prompt_len + width
        page_count = (seq_len + tokens_per_block - 1) // tokens_per_block
        initial = torch.arange(
            request_count * page_count * 2 * tokens_per_block * 2,
            dtype=torch.float32,
            device=device,
        ).view(request_count * page_count, 2, 1, tokens_per_block, 2)
        eager = self._build_formal_path(
            width,
            budget,
            "cute_dsl_topk",
            request_count,
            initial,
            prompt_len=prompt_len,
        )
        graphed = self._build_formal_path(
            width,
            budget,
            "cute_dsl_topk",
            request_count,
            initial,
            prompt_len=prompt_len,
        )
        eager_pool, _, eager_selection, _, eager_stage, _, eager_body = eager
        graph_pool, _, graph_selection, graph_compaction, graph_stage, graph_body, _ = graphed
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024**3)
        stream = torch.cuda.current_stream(device)

        for replay, valid_widths in enumerate(((4097, 4160), (4110, 4200))):
            valid_seq_lens = [prompt_len + valid_width for valid_width in valid_widths]
            eager_pool.copy_(initial)
            graph_pool.copy_(initial)
            eager_tables = eager_stage(
                round_shift=37 * replay,
                reverse_pages=bool(replay),
                valid_seq_lens=valid_seq_lens,
            )
            assert (
                graph_stage(
                    round_shift=37 * replay,
                    reverse_pages=bool(replay),
                    valid_seq_lens=valid_seq_lens,
                )
                == eager_tables
            )
            eager_body()
            fingerprint = graph_compaction.pointer_fingerprint(stream)
            assert cache.execute(
                key=("ragged-upper",),
                request_count=request_count,
                fingerprint=fingerprint,
                workspace=graph_compaction,
                capture_body=graph_body,
            ) == ("capture" if replay == 0 else "replay")
            torch.cuda.synchronize(device)

            assert torch.equal(
                eager_selection.keep[:request_count],
                graph_selection.keep[:request_count],
            )
            assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
            for request_index, valid_seq_len in enumerate(valid_seq_lens):
                assert int(graph_selection.keep[request_index].max()) < valid_seq_len

        assert cache.snapshot()["capture"] == 1
        assert cache.snapshot()["cache_hit"] == 1
        assert cache.snapshot()["rejected"] == 0

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
            [[0, 1, 2, 4, 5, 7], [0, 1, 2, 3, 5, 6]],
            dtype=torch.int64,
            device=device,
        )
        valid_seq_lens = torch.tensor([8, 7], dtype=torch.int32, device=device)
        protected_tail_lengths = [2, 1]
        page_ids_device = torch.stack((dense_tables, swa_tables))

        def build(pools):
            score = SimpleNamespace(
                page_count=page_count,
                representative_slots={0: 0, 1: 1},
                page_ids_device=page_ids_device,
                valid_seq_lens_device=valid_seq_lens,
            )
            selection = SimpleNamespace(
                eviction_mode="union",
                dense_layers=(0,),
                num_kv_heads=1,
                max_requests=request_count,
                prompt_len=2,
                keep_count=4,
                width=7,
                keep=keep.clone(),
                selection_backend="cute_dsl_topk",
                named_tensors=lambda: (),
            )
            return FixedBatchedCompactionWorkspace(
                eviction_mode="union",
                layer_pools=pools,
                dense_layers=[0],
                swa_layers=[1],
                layer_group_representative={0: 0},
                global_layers=[0, 1],
                score_workspace=score,
                selection_workspace=selection,
                request_count=request_count,
                seq_len=9,
                prompt_len=2,
                decode_keep_count=4,
                swa_window=2,
                arena_generation=1,
                protected_tail_lengths=protected_tail_lengths,
            )

        eager_pools = [pool.clone() for pool in base_pools]
        graph_pools = [pool.clone() for pool in base_pools]
        graphed = build(graph_pools)
        cache = StandaloneEvictionGraphCache(max_entries=1, max_bytes=1024**3)

        for replay, round_seq_lens in enumerate(((8, 7), (9, 8))):
            round_dense_tables = dense_tables.flip(1) if replay else dense_tables
            round_swa_tables = swa_tables.flip(1) if replay else swa_tables
            round_pools = [pool + replay * 10_000.0 for pool in base_pools]
            for eager_pool, graph_pool, round_pool in zip(eager_pools, graph_pools, round_pools):
                eager_pool.copy_(round_pool)
                graph_pool.copy_(round_pool)
            page_ids_device.copy_(torch.stack((round_dense_tables, round_swa_tables)))
            valid_seq_lens.copy_(torch.tensor(round_seq_lens, dtype=torch.int32, device=device))

            dense_sources = []
            swa_sources = []
            swa_destinations = []
            physical_seq_lens = []
            for request, (seq_len, tail_length) in enumerate(
                zip(round_seq_lens, protected_tail_lengths)
            ):
                protected_tail = torch.arange(
                    seq_len,
                    seq_len + tail_length,
                    dtype=torch.int64,
                    device=device,
                )
                dense_sources.append(torch.cat((keep[request], protected_tail)))
                swa_sources.append(
                    torch.arange(
                        seq_len - 2,
                        seq_len + tail_length,
                        dtype=torch.int64,
                        device=device,
                    )
                )
                swa_destinations.append(
                    torch.arange(
                        4,
                        6 + tail_length,
                        dtype=torch.int64,
                        device=device,
                    )
                )
                physical_seq_lens.append(seq_len + tail_length)

            _cpp_sparse_compact_reference(
                eager_pools[0],
                [round_dense_tables[request] for request in range(request_count)],
                dense_sources,
                physical_seq_lens,
            )
            _cpp_sparse_compact_reference(
                eager_pools[1],
                [round_swa_tables[request] for request in range(request_count)],
                swa_sources,
                physical_seq_lens,
                dest_list=swa_destinations,
            )
            assert cache.execute(
                key=("swa-with-protected-tail",),
                request_count=request_count,
                fingerprint=("swa-with-protected-tail",),
                workspace=graphed,
                capture_body=graphed.launch,
            ) == ("capture" if replay == 0 else "replay")
            torch.cuda.synchronize(device)

            for eager_pool, graph_pool in zip(eager_pools, graph_pools):
                assert torch.equal(eager_pool.view(torch.uint8), graph_pool.view(torch.uint8))
            for request in range(request_count):
                dense_ids = round_dense_tables[request]
                swa_ids = round_swa_tables[request]
                dense_before = round_pools[0][dense_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
                dense_after = graph_pools[0][dense_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
                swa_before = round_pools[1][swa_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
                swa_after = graph_pools[1][swa_ids].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 2)
                assert torch.equal(dense_after[:, :, :2], dense_before[:, :, :2])
                assert torch.equal(swa_after[:, :, :2], swa_before[:, :, :2])
                assert torch.equal(
                    dense_after[:, :, : dense_sources[request].numel()],
                    dense_before.index_select(2, dense_sources[request]),
                )
                assert torch.equal(
                    swa_after.index_select(2, swa_destinations[request]),
                    swa_before.index_select(2, swa_sources[request]),
                )

        assert cache.snapshot()["capture"] == 1
        assert cache.snapshot()["cache_hit"] == 1
        assert cache.snapshot()["rejected"] == 0
