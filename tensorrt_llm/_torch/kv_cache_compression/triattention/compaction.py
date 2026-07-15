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

"""Batched physical KV-cache compaction for TriAttention eviction."""

from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch


class _CppCompactGroup(NamedTuple):
    """Tensors for one layered sparse-KV updater launch."""

    pools: Tuple[torch.Tensor, ...]
    page_table: torch.Tensor
    pool_pointers: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]


def _run_cpp_compact_layers(
    group: _CppCompactGroup,
    source: torch.Tensor,
    offsets: torch.Tensor,
    destination_base: int,
) -> None:
    """Compact one uniform V2 layer group with the sparse-KV updater."""
    torch.ops.trtllm.sparse_kv_cache_compact_layers(
        list(group.pools),
        group.pool_pointers,
        group.page_table,
        source,
        offsets,
        group.source_layer_indices,
        destination_base,
    )


class _PreparedPackCompactionSources:
    """Launch one fixed compaction-source packing kernel."""

    def __init__(
        self,
        selected_indices: torch.Tensor,
        valid_seq_lens: torch.Tensor,
        dense_offsets: torch.Tensor,
        dense_indices: torch.Tensor,
        *,
        eviction_mode: str,
        prompt_len: int,
        keep_count: int,
        num_dense_layers: int,
        num_kv_heads: int,
        max_protected_tail: int,
        swa_window: int,
        swa_offsets: Optional[torch.Tensor],
        swa_indices: Optional[torch.Tensor],
    ) -> None:
        if eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(f"unsupported compaction mode: {eviction_mode}")
        request_count = int(selected_indices.shape[0]) if selected_indices.ndim else 0
        per_layer = eviction_mode == "per_layer_perhead"
        union = eviction_mode == "union"
        if union:
            selection_rows = 1
        elif per_layer:
            selection_rows = num_dense_layers * num_kv_heads
        else:
            selection_rows = num_kv_heads
        selection_prefix = (request_count,) if union else (request_count, selection_rows)
        dense_prefix = (num_dense_layers, num_kv_heads) if per_layer else (num_kv_heads,)
        if (
            request_count <= 0
            or min(keep_count, num_dense_layers, num_kv_heads) <= 0
            or min(prompt_len, max_protected_tail, swa_window) < 0
            or tuple(selected_indices.shape) != (*selection_prefix, prompt_len + keep_count)
            or valid_seq_lens.shape != (request_count,)
            or dense_offsets.shape != (request_count + 1,)
            or dense_indices.ndim != len(dense_prefix) + 1
            or tuple(dense_indices.shape[:-1]) != dense_prefix
        ):
            raise ValueError("prepared compaction packing requires one valid fixed geometry")

        device = selected_indices.device
        fixed_tensors = (selected_indices, valid_seq_lens, dense_offsets, dense_indices)
        if any(
            not tensor.is_cuda
            or tensor.dtype != torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
            for tensor in fixed_tensors
        ):
            raise ValueError("prepared compaction packing requires contiguous CUDA int32 tensors")

        has_swa = swa_offsets is not None or swa_indices is not None
        if has_swa:
            if (
                swa_offsets is None
                or swa_indices is None
                or swa_window <= 0
                or swa_offsets.shape != (request_count + 1,)
                or swa_indices.ndim != 2
                or tuple(swa_indices.shape[:-1]) != (num_kv_heads,)
                or any(
                    not tensor.is_cuda
                    or tensor.dtype != torch.int32
                    or tensor.device != device
                    or not tensor.is_contiguous()
                    for tensor in (swa_offsets, swa_indices)
                )
            ):
                raise ValueError("prepared SWA packing buffers do not match the fixed geometry")
            swa_offsets_arg = swa_offsets
            swa_indices_arg = swa_indices
            swa_total = int(swa_indices.shape[-1])
        else:
            if swa_window != 0:
                raise ValueError("prepared SWA packing requires source buffers")
            # HAS_SWA specializes all corresponding loads and stores away.
            swa_offsets_arg = dense_offsets
            swa_indices_arg = dense_indices
            swa_total = 0

        from .triattention_kernels import _pack_compaction_sources_kernel

        block = 256
        max_move = keep_count + max_protected_tail
        if has_swa:
            max_move = max(max_move, swa_window + max_protected_tail)
        domain_count = num_dense_layers * num_kv_heads if per_layer else num_kv_heads
        grid = (request_count, domain_count, (max_move + block - 1) // block)
        self.device = device
        self.tensors = (
            selected_indices,
            valid_seq_lens,
            dense_offsets,
            dense_indices,
            swa_offsets_arg,
            swa_indices_arg,
        )
        self.constants = (
            int(dense_indices.shape[-1]),
            swa_total,
            selection_rows,
            prompt_len + keep_count,
            keep_count,
            prompt_len,
            num_kv_heads,
            swa_window,
            union,
            per_layer,
            has_swa,
            block,
        )
        with torch.cuda.device(device):
            self.stream = torch.cuda.current_stream(device)
            compiled = _pack_compaction_sources_kernel.warmup(
                *self.tensors,
                DENSE_TOTAL=self.constants[0],
                SWA_TOTAL=self.constants[1],
                SELECTION_ROWS=self.constants[2],
                SELECTION_STRIDE=self.constants[3],
                KEEP_COUNT=self.constants[4],
                PROMPT_LEN=self.constants[5],
                NUM_KV_HEADS=self.constants[6],
                SWA_WINDOW=self.constants[7],
                UNION=self.constants[8],
                PER_LAYER=self.constants[9],
                HAS_SWA=self.constants[10],
                BLOCK=self.constants[11],
                num_warps=4,
                grid=grid,
            )
            self.runner = compiled[grid]

    def __call__(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        if (current_stream.device, current_stream.cuda_stream) != (
            self.stream.device,
            self.stream.cuda_stream,
        ):
            raise RuntimeError("prepared compaction packing must run on its workspace stream")
        self.runner(
            *self.tensors,
            *self.constants,
            stream=self.stream.cuda_stream,
        )


class BatchedCompactionWorkspace:
    """Buffers for one batched physical KV-cache compaction geometry.

    Dense layers leave the prompt in place and compact selected decode tokens
    plus any target KV reserved for the next overlapped forward. Kernel-masked
    SWA layers compact the latest window plus the same protected target tail.
    """

    def __init__(
        self,
        *,
        eviction_mode: str,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        swa_layers: List[int],
        layer_group_representative: Dict[int, int],
        score_workspace,
        selection_workspace,
        request_count: int,
        seq_len: int,
        prompt_len: int,
        decode_keep_count: int,
        swa_window: Optional[int],
        protected_tail_lengths: Optional[List[int]] = None,
        layer_pool_keys: Optional[List[object]] = None,
    ) -> None:
        if eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(f"unsupported compaction mode: {eviction_mode}")
        if request_count <= 0 or decode_keep_count <= 0:
            raise ValueError("batched compaction requires requests and retained tokens")
        if not dense_layers:
            raise ValueError("batched compaction requires at least one dense layer")
        if request_count > selection_workspace.max_requests:
            raise ValueError("batched compaction exceeds the selection workspace")
        if (
            selection_workspace.prompt_len != prompt_len
            or selection_workspace.keep_count != decode_keep_count
            or selection_workspace.width != seq_len - prompt_len
        ):
            raise ValueError("batched compaction geometry does not match selection")

        if selection_workspace.eviction_mode != eviction_mode or tuple(
            selection_workspace.dense_layers
        ) != tuple(dense_layers):
            raise ValueError("selection mode or dense-layer order does not match compaction")

        self.eviction_mode = eviction_mode
        self.device = layer_pools[dense_layers[0]].device
        self.request_count = int(request_count)
        self.prompt_len = int(prompt_len)
        self.decode_keep_count = int(decode_keep_count)
        self.keep_count = self.prompt_len + self.decode_keep_count
        if protected_tail_lengths is None:
            protected_tail_lengths = [0] * self.request_count
        if len(protected_tail_lengths) != self.request_count or any(
            length < 0 for length in protected_tail_lengths
        ):
            raise ValueError("protected-tail lengths must match the request count")
        self.protected_tail_lengths = tuple(int(length) for length in protected_tail_lengths)
        self.max_protected_tail = max(self.protected_tail_lengths, default=0)
        self.dense_layers = tuple(int(layer) for layer in dense_layers)
        self.swa_layers = tuple(int(layer) for layer in swa_layers)
        if layer_pool_keys is None:
            layer_pool_keys = [("layer", layer) for layer in range(len(layer_pools))]
        if len(layer_pool_keys) != len(layer_pools):
            raise ValueError("pool keys must match the layer-pool count")
        self.layer_pool_keys = tuple(layer_pool_keys)
        self.selection_workspace = selection_workspace
        self.score_workspace = score_workspace

        # Each uniform V2 group uses one layered sparse-KV updater launch.
        first_dense_pool = layer_pools[self.dense_layers[0]]
        cpp_num_kv_heads = int(first_dense_pool.shape[2]) if first_dense_pool.ndim == 5 else -1
        supported_pools = all(
            layer_pools[layer].ndim == 5
            and layer_pools[layer].shape[1] == 2
            and layer_pools[layer].device == self.device
            and int(layer_pools[layer].shape[2]) == cpp_num_kv_heads
            and layer_pools[layer].is_contiguous()
            and layer_pools[layer].dtype in (torch.bfloat16, torch.float16, torch.float32)
            for layer in (*self.dense_layers, *self.swa_layers)
        )
        if not supported_pools:
            raise ValueError(
                "batched compaction requires contiguous interleaved BF16/FP16/FP32 pools "
                "with one common KV-head count"
            )
        self.cpp_num_kv_heads = cpp_num_kv_heads
        if selection_workspace.num_kv_heads != cpp_num_kv_heads:
            raise ValueError("selection KV-head count does not match the pool")
        move_counts = [self.decode_keep_count + length for length in self.protected_tail_lengths]
        move_offsets = [0]
        for count in move_counts:
            move_offsets.append(move_offsets[-1] + count)
        cpp_index_shape = (
            (len(self.dense_layers), cpp_num_kv_heads, move_offsets[-1])
            if self.eviction_mode == "per_layer_perhead"
            else (cpp_num_kv_heads, move_offsets[-1])
        )
        self.cpp_indices = torch.empty(
            cpp_index_shape,
            dtype=torch.int32,
            device=self.device,
        )
        self.cpp_offsets = torch.tensor(
            move_offsets,
            dtype=torch.int32,
            device=self.device,
        )
        self.dense_destination_base = self.prompt_len
        cpp_page_tables = {}

        def page_table_for(representative: int) -> torch.Tensor:
            slot = score_workspace.representative_slots[representative]
            if slot not in cpp_page_tables:
                block_offsets = score_workspace.block_offsets_device[slot, : self.request_count, 0]
                if block_offsets.device != self.device or block_offsets.dtype != torch.int32:
                    raise ValueError("block offsets must be int32 tensors on the pool device")
                if block_offsets.ndim != 2 or block_offsets.stride(1) != 1:
                    raise ValueError("K block offsets must have a contiguous block dimension")
                cpp_page_tables[slot] = block_offsets
            return cpp_page_tables[slot]

        dense_entries = [
            (
                layer,
                layer_pools[layer],
                page_table_for(layer_group_representative[layer]),
            )
            for layer in self.dense_layers
        ]

        self.swa_indices = None
        self.swa_destination_base = None
        self.swa_offsets = None
        self.swa_window = 0
        swa_entries = []
        if self.swa_layers:
            if swa_window is None or swa_window <= 0 or self.keep_count < swa_window:
                raise ValueError("SWA compaction requires a valid retained window")
            self.swa_window = int(swa_window)
            swa_move_counts = [int(swa_window) + length for length in self.protected_tail_lengths]
            swa_move_offsets = [0]
            for count in swa_move_counts:
                swa_move_offsets.append(swa_move_offsets[-1] + count)
            total_swa = swa_move_offsets[-1]
            self.swa_indices = torch.empty(
                cpp_num_kv_heads,
                total_swa,
                dtype=torch.int32,
                device=self.device,
            )
            self.swa_destination_base = self.keep_count - int(swa_window)
            self.swa_offsets = torch.tensor(
                swa_move_offsets,
                dtype=torch.int32,
                device=self.device,
            )
            for layer in self.swa_layers:
                swa_entries.append((layer, layer_pools[layer], page_table_for(layer)))

        dense_slot_by_layer = {layer: slot for slot, layer in enumerate(self.dense_layers)}

        def compact_groups(entries, mode: str) -> Tuple[_CppCompactGroup, ...]:
            grouped = OrderedDict()
            for layer, pool, page_table in entries:
                key = (
                    self.layer_pool_keys[layer],
                    mode,
                    str(pool.dtype),
                    str(pool.device),
                    tuple(int(value) for value in pool.shape[1:]),
                    tuple(int(value) for value in page_table.shape),
                )
                grouped.setdefault(key, []).append((layer, pool, page_table))

            result = []
            for group_entries in grouped.values():
                layers = tuple(entry[0] for entry in group_entries)
                pools = tuple(entry[1] for entry in group_entries)
                page_tables = tuple(entry[2] for entry in group_entries)
                if len({int(pool.data_ptr()) for pool in pools}) != len(pools):
                    raise ValueError(
                        "layered compaction requires a distinct pool view for every layer"
                    )
                if len({int(page_table.data_ptr()) for page_table in page_tables}) != 1:
                    raise ValueError("layers in one V2 pool must share one block-offset table")
                source_layer_indices = None
                if mode == "dense" and self.eviction_mode == "per_layer_perhead":
                    source_layer_indices = torch.tensor(
                        [dense_slot_by_layer[layer] for layer in layers],
                        dtype=torch.int32,
                        device=self.device,
                    )
                result.append(
                    _CppCompactGroup(
                        pools=pools,
                        page_table=page_tables[0],
                        pool_pointers=torch.tensor(
                            [pool.data_ptr() for pool in pools],
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        source_layer_indices=source_layer_indices,
                    )
                )
            return tuple(result)

        self.cpp_dense_groups = compact_groups(dense_entries, "dense")
        self.cpp_swa_groups = compact_groups(swa_entries, "swa")
        self._pack_compaction_sources = _PreparedPackCompactionSources(
            self.selection_workspace.keep[: self.request_count],
            self.score_workspace.valid_seq_lens_device[: self.request_count],
            self.cpp_offsets,
            self.cpp_indices,
            eviction_mode=self.eviction_mode,
            prompt_len=self.prompt_len,
            keep_count=self.decode_keep_count,
            num_dense_layers=len(self.dense_layers),
            num_kv_heads=self.cpp_num_kv_heads,
            max_protected_tail=self.max_protected_tail,
            swa_window=self.swa_window,
            swa_offsets=self.swa_offsets,
            swa_indices=self.swa_indices,
        )

    def launch(self) -> None:
        """Stage dynamic metadata and run C++ dense/SWA compact launches."""
        self._pack_compaction_sources()
        for group in self.cpp_dense_groups:
            _run_cpp_compact_layers(
                group,
                self.cpp_indices,
                self.cpp_offsets,
                self.dense_destination_base,
            )
        if self.swa_indices is not None:
            assert self.swa_offsets is not None
            assert self.swa_destination_base is not None
            for group in self.cpp_swa_groups:
                _run_cpp_compact_layers(
                    group,
                    self.swa_indices,
                    self.swa_offsets,
                    self.swa_destination_base,
                )
