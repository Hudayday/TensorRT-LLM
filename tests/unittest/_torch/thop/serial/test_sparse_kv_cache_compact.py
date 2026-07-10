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

"""Tests for the V2 sparse KV-cache compact custom operators."""

from typing import NamedTuple, Optional

import pytest
import torch

import tensorrt_llm  # noqa: F401  # Register torch.ops.trtllm operators.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_TOKENS_PER_BLOCK = 4
_NUM_KV_HEADS = 2
_BATCH_SIZE = 2
_MAX_PAGES_PER_SEQUENCE = 3
_NUM_PAGES = _BATCH_SIZE * _MAX_PAGES_PER_SEQUENCE


class _DeviceArguments(NamedTuple):
    pool_pointers: torch.Tensor
    page_table_pointers: torch.Tensor
    source_indices: torch.Tensor
    source_offsets: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]
    destination_indices: Optional[torch.Tensor]


def _make_pools(
    num_layers: int,
    dtype: torch.dtype,
    head_dim: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    shape = (
        _NUM_PAGES,
        2,
        _NUM_KV_HEADS,
        _TOKENS_PER_BLOCK,
        head_dim,
    )
    numel = torch.Size(shape).numel()
    pools_cpu = [
        ((torch.arange(numel, dtype=torch.int32) + layer * 37) % 251).reshape(shape).to(dtype)
        for layer in range(num_layers)
    ]
    pools = [pool.cuda() for pool in pools_cpu]

    page_table_cpu = torch.tensor(
        [[4, 1, 5], [2, 0, 3]],
        dtype=torch.int32,
    )
    page_tables_cpu = [page_table_cpu.clone() for _ in range(num_layers)]
    page_tables = [page_table.cuda() for page_table in page_tables_cpu]
    return pools_cpu, pools, page_tables


def _device_arguments(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    source_indices: torch.Tensor,
    source_offsets: torch.Tensor,
    source_layer_indices: Optional[torch.Tensor] = None,
    destination_indices: Optional[torch.Tensor] = None,
) -> _DeviceArguments:
    device = pools[0].device
    pool_pointers = torch.tensor(
        [pool.data_ptr() for pool in pools],
        dtype=torch.int64,
        device=device,
    )
    page_table_pointers = torch.tensor(
        [page_table.data_ptr() for page_table in page_tables],
        dtype=torch.int64,
        device=device,
    )
    source_indices_cuda = source_indices.to(device)
    source_offsets_cuda = source_offsets.to(device)
    source_layer_indices_cuda = (
        None if source_layer_indices is None else source_layer_indices.to(device)
    )
    destination_indices_cuda = (
        None if destination_indices is None else destination_indices.to(device)
    )
    return _DeviceArguments(
        pool_pointers=pool_pointers,
        page_table_pointers=page_table_pointers,
        source_indices=source_indices_cuda,
        source_offsets=source_offsets_cuda,
        source_layer_indices=source_layer_indices_cuda,
        destination_indices=destination_indices_cuda,
    )


def _reference_compact(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    source_indices: torch.Tensor,
    source_offsets: torch.Tensor,
    source_layer_indices: Optional[torch.Tensor] = None,
    destination_indices: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    original = [pool.clone() for pool in pools]
    expected = [pool.clone() for pool in pools]
    total_moves = int(source_indices.shape[-1])

    for group_layer, (source_pool, destination_pool, page_table) in enumerate(
        zip(original, expected, page_tables)
    ):
        if source_indices.ndim == 2:
            layer_sources = source_indices
        else:
            assert source_layer_indices is not None
            layer_sources = source_indices[int(source_layer_indices[group_layer])]

        for request in range(_BATCH_SIZE):
            begin = int(source_offsets[request])
            end = int(source_offsets[request + 1])
            for head in range(_NUM_KV_HEADS):
                for move in range(begin, end):
                    source_token = int(layer_sources[head, move])
                    destination_token = move - begin
                    if destination_indices is not None:
                        if destination_indices.ndim == 1:
                            destination_token = int(destination_indices[move])
                        elif destination_indices.ndim == 2:
                            destination_token = int(destination_indices[head, move])
                        else:
                            destination_token = int(destination_indices[group_layer, head, move])

                    source_page = int(page_table[request, source_token // _TOKENS_PER_BLOCK])
                    destination_page = int(
                        page_table[request, destination_token // _TOKENS_PER_BLOCK]
                    )
                    source_slot = source_token % _TOKENS_PER_BLOCK
                    destination_slot = destination_token % _TOKENS_PER_BLOCK
                    destination_pool[destination_page, :, head, destination_slot, :] = source_pool[
                        source_page, :, head, source_slot, :
                    ]

    assert int(source_offsets[-1]) == total_moves
    return expected


def _run_multi_layer(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    arguments: _DeviceArguments,
) -> None:
    torch.ops.trtllm.sparse_kv_cache_compact_layers(
        pools,
        arguments.pool_pointers,
        page_tables,
        arguments.page_table_pointers,
        arguments.source_indices,
        arguments.source_offsets,
        arguments.source_layer_indices,
        arguments.destination_indices,
    )


@pytest.mark.parametrize(
    "dtype,head_dim",
    [
        (torch.float16, 16),
        (torch.bfloat16, 32),
        (torch.float32, 64),
        (torch.float16, 128),
        (torch.bfloat16, 256),
        (torch.float32, 256),
    ],
)
def test_sparse_kv_cache_compact_layers_union(dtype, head_dim):
    pools_cpu, pools, page_tables = _make_pools(3, dtype, head_dim)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    source_row = torch.tensor([0, 2, 5, 9, 1, 3, 6, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
    )
    arguments = _device_arguments(
        pools,
        page_tables,
        source_indices,
        source_offsets,
    )

    _run_multi_layer(pools, page_tables, arguments)
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sparse_kv_cache_compact_layers_explicit_destination(dtype):
    pools_cpu, pools, page_tables = _make_pools(2, dtype, head_dim=64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor([4, 7, 10, 3, 8, 11], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    destination_indices = torch.tensor([2, 5, 8, 1, 4, 7], dtype=torch.int32)
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
        destination_indices=destination_indices,
    )
    arguments = _device_arguments(
        pools,
        page_tables,
        source_indices,
        source_offsets,
        destination_indices=destination_indices,
    )

    _run_multi_layer(pools, page_tables, arguments)
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_layers_noncontiguous_source_layers():
    pools_cpu, pools, page_tables = _make_pools(3, torch.bfloat16, head_dim=64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    source_layer_indices = torch.tensor([4, 1, 3], dtype=torch.int32)
    variants = torch.tensor(
        [
            [0, 3, 6, 9, 1, 4, 7, 10],
            [1, 4, 7, 10, 0, 3, 6, 9],
            [2, 5, 8, 11, 2, 5, 8, 11],
        ],
        dtype=torch.int32,
    )
    source_indices = torch.empty(5, _NUM_KV_HEADS, 8, dtype=torch.int32)
    for source_layer in range(source_indices.shape[0]):
        for head in range(_NUM_KV_HEADS):
            source_indices[source_layer, head] = variants[(source_layer + head) % 3]
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
        source_layer_indices=source_layer_indices,
    )
    arguments = _device_arguments(
        pools,
        page_tables,
        source_indices,
        source_offsets,
        source_layer_indices=source_layer_indices,
    )

    _run_multi_layer(pools, page_tables, arguments)
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_layers_cuda_graph_replay():
    pools_cpu, pools, page_tables = _make_pools(3, torch.bfloat16, head_dim=64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    source_row = torch.tensor([0, 2, 5, 9, 1, 3, 6, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    replay_source_row = torch.tensor([1, 4, 7, 10, 0, 3, 8, 11], dtype=torch.int32)
    replay_source_indices = replay_source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    replay_expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        replay_source_indices,
        source_offsets,
    )
    arguments = _device_arguments(
        pools,
        page_tables,
        source_indices,
        source_offsets,
    )

    _run_multi_layer(pools, page_tables, arguments)
    torch.cuda.synchronize()
    for pool, initial in zip(pools, pools_cpu):
        pool.copy_(initial)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_multi_layer(pools, page_tables, arguments)

    for pool, initial in zip(pools, pools_cpu):
        pool.copy_(initial)
    arguments.source_indices.copy_(replay_source_indices)
    graph.replay()
    torch.cuda.synchronize()

    for actual, reference in zip(pools, replay_expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_single_layer_abi():
    pools_cpu, pools, page_tables = _make_pools(1, torch.float16, head_dim=32)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    source_row = torch.tensor([0, 2, 5, 9, 1, 3, 6, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
    )[0]

    torch.ops.trtllm.sparse_kv_cache_compact(
        pools[0],
        page_tables[0],
        source_indices.cuda(),
        source_offsets.cuda(),
        None,
    )
    torch.cuda.synchronize()

    assert torch.equal(pools[0].cpu(), expected)
