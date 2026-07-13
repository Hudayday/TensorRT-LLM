# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.triattention.compaction import (
    BatchedCompactionWorkspace,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _BatchedFixedPerHeadWorkspace,
    _BatchedFixedUnionWorkspace,
)

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _stable_topk(row: torch.Tensor, width: int, keep_count: int) -> torch.Tensor:
    values = row[:width].tolist()
    selected = sorted(range(width), key=lambda index: (-values[index], index))
    return torch.tensor(selected[:keep_count], dtype=torch.int32, device=row.device)


def _fake_cute_topk(scores, seq_lens, output, top_k, next_n):
    assert next_n == 1
    for row_index, row in enumerate(scores):
        output[row_index].copy_(_stable_topk(row, int(seq_lens[row_index]), int(top_k)))


class _AlternatingTieTopK:
    """Return alternating raw tie members; canonicalization must remove this choice."""

    def __init__(self):
        self.calls = 0

    def __call__(self, scores, seq_lens, output, top_k, next_n):
        assert next_n == 1
        self.calls += 1
        for row_index, row in enumerate(scores):
            width = int(seq_lens[row_index])
            values = row[:width]
            threshold = torch.sort(values, descending=True).values[int(top_k) - 1]
            higher = torch.nonzero(values > threshold, as_tuple=False).flatten()
            tied = torch.nonzero(values == threshold, as_tuple=False).flatten()
            if self.calls % 2 == 0:
                tied = tied.flip(0)
            remaining = int(top_k) - int(higher.numel())
            output[row_index].copy_(torch.cat((higher, tied[:remaining])).to(torch.int32))


def _legacy_union(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    row_top = {
        int(index)
        for row in scores
        for index in _stable_topk(row, row.numel(), keep_count).tolist()
    }
    combined = scores.max(dim=0).values
    ordered = sorted(
        row_top,
        key=lambda index: (-float(combined[index]), index),
    )
    return torch.tensor(ordered[:keep_count], dtype=torch.long)


@pytest.mark.parametrize("rows,width,keep_count", [(2, 8, 4), (5, 17, 7)])
def test_direct_union_topk_matches_legacy_union_with_heavy_ties(rows, width, keep_count):
    for seed in range(25):
        generator = torch.Generator().manual_seed(seed)
        scores = torch.randint(
            -2,
            3,
            (rows, width),
            generator=generator,
            dtype=torch.int32,
        ).to(torch.float32)
        combined = scores.max(dim=0).values
        direct = _stable_topk(combined, width, keep_count).to(torch.long)
        assert torch.equal(direct, _legacy_union(scores, keep_count))


@pytest.mark.parametrize("keep_count,width", [(4, 8), (4096, 4224), (8192, 9216)])
def test_union_eager_uses_one_deterministic_cute_selection(keep_count, width):
    prompt_len = 17
    generator = torch.Generator().manual_seed(keep_count)
    scores = torch.randint(
        -8,
        9,
        (2, width),
        generator=generator,
        dtype=torch.int32,
    ).to(torch.float32)
    workspace = _BatchedFixedUnionWorkspace(
        rows=2,
        width=width,
        keep_count=keep_count,
        prompt_len=prompt_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_requests=1,
    )
    raw_topk = _AlternatingTieTopK()
    with (
        mock.patch.object(
            torch.ops.trtllm,
            "cute_dsl_indexer_topk_decode",
            side_effect=raw_topk,
            create=True,
        ),
        mock.patch.object(
            torch.ops.trtllm,
            "indexer_topk_decode",
            side_effect=AssertionError("legacy selector was called"),
            create=True,
        ),
    ):
        workspace.select_requests(scores.unsqueeze(0), normalize_scores=False)

    expected = _stable_topk(scores.max(dim=0).values, width, keep_count)
    assert torch.equal(
        workspace.keep[0, prompt_len:],
        torch.sort(expected + prompt_len).values,
    )
    # One logical deterministic selection is two CuTE calls: provisional and
    # canonicalized. The removed row-union stage previously doubled this count.
    assert raw_topk.calls == 2


@pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
def test_per_head_eager_keeps_stable_indices(eviction_mode):
    workspace = _BatchedFixedPerHeadWorkspace(
        eviction_mode=eviction_mode,
        dense_layers=(0, 1),
        num_query_heads=4,
        num_kv_heads=2,
        width=16,
        keep_count=5,
        prompt_len=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_requests=1,
    )
    scores = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
    with mock.patch.object(
        torch.ops.trtllm,
        "cute_dsl_indexer_topk_decode",
        side_effect=_fake_cute_topk,
        create=True,
    ):
        workspace.select_requests(scores.unsqueeze(0), normalize_scores=False)
    assert tuple(workspace.keep.shape) == (
        1,
        workspace.selection_rows,
        workspace.total_keep,
    )
    assert torch.all(workspace.keep[..., 1:] >= workspace.keep[..., :-1])


@CUDA_REQUIRED
def test_union_eager_runs_the_registered_cute_op():
    if not hasattr(torch.ops.trtllm, "cute_dsl_indexer_topk_decode"):
        pytest.skip("CuTE TopK operation is not loaded")
    device = torch.device("cuda")
    workspace = _BatchedFixedUnionWorkspace(
        rows=4,
        width=96,
        keep_count=64,
        prompt_len=0,
        dtype=torch.float32,
        device=device,
        max_requests=2,
    )
    scores = torch.randn(2, 4, 96, dtype=torch.float32, device=device)
    workspace.select_requests(scores, normalize_scores=True)
    torch.cuda.synchronize(device)
    assert torch.all(workspace.keep[:, 1:] >= workspace.keep[:, :-1])


@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
@CUDA_REQUIRED
def test_eager_compaction_preserves_exact_selected_bytes_and_tail(eviction_mode):
    device = torch.device("cuda")
    request_count = 2
    num_layers = 2
    num_kv_heads = 2
    prompt_len = 2
    decode_keep_count = 4
    seq_len = 10
    protected_tails = [2, 1]
    tokens_per_block = 4
    pages_per_request = 3
    head_dim = 16
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    initial_pools = [
        (
            torch.arange(
                6 * 2 * num_kv_heads * tokens_per_block * head_dim,
                dtype=torch.float32,
                device=device,
            ).view(6, 2, num_kv_heads, tokens_per_block, head_dim)
            + layer * 100_000.0
        )
        for layer in range(num_layers)
    ]
    pools = [pool.clone() for pool in initial_pools]

    prompt = torch.tensor([0, 1], dtype=torch.int64, device=device)
    union_decode = torch.tensor([[2, 4, 7, 9], [3, 5, 6, 8]], dtype=torch.int64, device=device)
    if eviction_mode == "union":
        keep = torch.cat((prompt.view(1, -1).expand(request_count, -1), union_decode), dim=1)
        selection_rows = 1
    else:
        selection_rows = num_kv_heads if eviction_mode == "per_head" else num_layers * num_kv_heads
        decode = torch.empty(
            request_count,
            selection_rows,
            decode_keep_count,
            dtype=torch.int64,
            device=device,
        )
        for request in range(request_count):
            for row in range(selection_rows):
                decode[request, row] = torch.tensor(
                    sorted(
                        {
                            2 + ((request + row + offset * 2) % 8)
                            for offset in range(decode_keep_count)
                        }
                    ),
                    dtype=torch.int64,
                    device=device,
                )
        keep = torch.cat(
            (
                prompt.view(1, 1, -1).expand(request_count, selection_rows, -1),
                decode,
            ),
            dim=2,
        )

    score = SimpleNamespace(
        representative_slots={0: 0, 1: 1},
        page_ids_device=torch.stack((page_tables, page_tables)),
        valid_seq_lens_device=torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device),
    )
    selection = SimpleNamespace(
        eviction_mode=eviction_mode,
        dense_layers=(0, 1),
        num_kv_heads=num_kv_heads,
        max_requests=request_count,
        prompt_len=prompt_len,
        keep_count=decode_keep_count,
        width=seq_len - prompt_len,
        keep=keep,
    )
    workspace = BatchedCompactionWorkspace(
        eviction_mode=eviction_mode,
        layer_pools=pools,
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_keys=[("dense", 0), ("dense", 0)],
        score_workspace=score,
        selection_workspace=selection,
        request_count=request_count,
        seq_len=seq_len,
        prompt_len=prompt_len,
        decode_keep_count=decode_keep_count,
        swa_window=None,
        protected_tail_lengths=protected_tails,
    )
    workspace.launch()
    torch.cuda.synchronize(device)

    for layer, (before_pool, after_pool) in enumerate(zip(initial_pools, pools)):
        for request in range(request_count):
            pages = page_tables[request].to(torch.long)
            before = (
                before_pool[pages]
                .permute(1, 2, 0, 3, 4)
                .reshape(2, num_kv_heads, pages_per_request * tokens_per_block, head_dim)
            )
            after = after_pool[pages].permute(1, 2, 0, 3, 4).reshape_as(before)
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            for head in range(num_kv_heads):
                if eviction_mode == "union":
                    selected = keep[request, prompt_len:]
                elif eviction_mode == "per_head":
                    selected = keep[request, head, prompt_len:]
                else:
                    selected = keep[request, layer * num_kv_heads + head, prompt_len:]
                tail = torch.arange(
                    seq_len,
                    seq_len + protected_tails[request],
                    dtype=torch.int64,
                    device=device,
                )
                source = torch.cat((selected, tail))
                destination = torch.arange(
                    prompt_len,
                    prompt_len + source.numel(),
                    dtype=torch.int64,
                    device=device,
                )
                assert torch.equal(
                    after[:, head].index_select(1, destination),
                    before[:, head].index_select(1, source),
                )


@CUDA_REQUIRED
def test_eager_compaction_rebases_masked_swa_window_and_tail():
    device = torch.device("cuda")
    dense_tables = torch.tensor([[2, 0, 1], [5, 3, 4]], device=device)
    swa_tables = torch.tensor([[1, 2, 0], [4, 5, 3]], device=device)
    initial_pools = [
        torch.arange(6 * 2 * 1 * 4 * 16, dtype=torch.float32, device=device).view(6, 2, 1, 4, 16),
        torch.arange(6 * 2 * 1 * 4 * 16, dtype=torch.float32, device=device).view(6, 2, 1, 4, 16)
        + 1000.0,
    ]
    pools = [pool.clone() for pool in initial_pools]
    keep = torch.tensor(
        [[0, 1, 2, 4, 5, 7], [0, 1, 2, 3, 5, 6]],
        dtype=torch.int64,
        device=device,
    )
    valid_seq_lens = torch.tensor([8, 7], dtype=torch.int32, device=device)
    protected_tails = [2, 1]
    score = SimpleNamespace(
        representative_slots={0: 0, 1: 1},
        page_ids_device=torch.stack((dense_tables, swa_tables)),
        valid_seq_lens_device=valid_seq_lens,
    )
    selection = SimpleNamespace(
        eviction_mode="union",
        dense_layers=(0,),
        num_kv_heads=1,
        max_requests=2,
        prompt_len=2,
        keep_count=4,
        width=7,
        keep=keep,
    )
    workspace = BatchedCompactionWorkspace(
        eviction_mode="union",
        layer_pools=pools,
        dense_layers=[0],
        swa_layers=[1],
        layer_group_representative={0: 0},
        layer_pool_keys=[("dense", 0), ("swa", 0)],
        score_workspace=score,
        selection_workspace=selection,
        request_count=2,
        seq_len=9,
        prompt_len=2,
        decode_keep_count=4,
        swa_window=2,
        protected_tail_lengths=protected_tails,
    )
    workspace.launch()
    torch.cuda.synchronize(device)

    for request, (valid_seq_len, tail_length) in enumerate(
        zip(valid_seq_lens.tolist(), protected_tails)
    ):
        dense_pages = dense_tables[request].to(torch.long)
        swa_pages = swa_tables[request].to(torch.long)
        dense_before = initial_pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 16)
        dense_after = pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape_as(dense_before)
        swa_before = initial_pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 16)
        swa_after = pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape_as(swa_before)
        tail = torch.arange(
            valid_seq_len,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        dense_source = torch.cat((keep[request, 2:], tail))
        dense_destination = torch.arange(
            2, 2 + dense_source.numel(), dtype=torch.int64, device=device
        )
        swa_source = torch.arange(
            valid_seq_len - 2,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        swa_destination = torch.arange(4, 4 + swa_source.numel(), dtype=torch.int64, device=device)
        assert torch.equal(dense_after[:, :, :2], dense_before[:, :, :2])
        assert torch.equal(swa_after[:, :, :2], swa_before[:, :, :2])
        assert torch.equal(
            dense_after.index_select(2, dense_destination),
            dense_before.index_select(2, dense_source),
        )
        assert torch.equal(
            swa_after.index_select(2, swa_destination),
            swa_before.index_select(2, swa_source),
        )
