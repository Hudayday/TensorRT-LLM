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
    _BatchedFixedPerHeadWorkspace,
    _BatchedFixedUnionWorkspace,
    _FixedScoreMetadataWorkspace,
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


def _cpp_multilayer_compact_reference(
    pool,
    page_ids_list,
    source_list,
    seq_len_list,
    *,
    dest_list=None,
):
    """Run the multi-layer compact operation eagerly for graph references."""
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
    torch.ops.trtllm.sparse_kv_cache_compact_layers(
        [pool],
        torch.tensor([pool.data_ptr()], dtype=torch.int64, device=pool.device),
        [page_table],
        torch.tensor([page_table.data_ptr()], dtype=torch.int64, device=pool.device),
        indices,
        offsets,
        None,
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


def _cache_workspace(nbytes=64):
    return SimpleNamespace(nbytes=nbytes, device=torch.device("cpu"))


class TestStandaloneEvictionGraphCache:
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
        bucket = cache.snapshot()["buckets"][0]
        assert (bucket["capture"], bucket["cache_hit"], bucket["covered_requests"]) == (1, 1, 8)

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


class TestGraphFingerprint:
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
            _cpp_multilayer_compact_reference(
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
            dense_destinations = []
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
                dense_sources.append(torch.cat((keep[request, 2:], protected_tail)))
                dense_destinations.append(
                    torch.arange(
                        2,
                        6 + tail_length,
                        dtype=torch.int64,
                        device=device,
                    )
                )
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

            _cpp_multilayer_compact_reference(
                eager_pools[0],
                [round_dense_tables[request] for request in range(request_count)],
                dense_sources,
                physical_seq_lens,
                dest_list=dense_destinations,
            )
            _cpp_multilayer_compact_reference(
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
                    dense_after.index_select(2, dense_destinations[request]),
                    dense_before.index_select(2, dense_sources[request]),
                )
                assert torch.equal(
                    swa_after.index_select(2, swa_destinations[request]),
                    swa_before.index_select(2, swa_sources[request]),
                )

        assert cache.snapshot()["capture"] == 1
        assert cache.snapshot()["cache_hit"] == 1
        assert cache.snapshot()["rejected"] == 0
