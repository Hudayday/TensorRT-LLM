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
    page_tables: Tuple[torch.Tensor, ...]
    pool_pointers: torch.Tensor
    page_table_pointers: torch.Tensor
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
        list(group.page_tables),
        group.page_table_pointers,
        source,
        offsets,
        group.source_layer_indices,
        destination_base,
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
        self._uniform_protected_tail_length = (
            self.protected_tail_lengths[0] if len(set(self.protected_tail_lengths)) == 1 else None
        )
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
        self._move_offsets = tuple(move_offsets)
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
        max_protected_tail = max(self.protected_tail_lengths, default=0)
        self.protected_tail_offsets = torch.arange(
            max_protected_tail,
            dtype=torch.int32,
            device=self.device,
        )
        self.protected_tail_source = torch.empty(
            self.request_count,
            max_protected_tail,
            dtype=torch.int32,
            device=self.device,
        )
        self.cpp_page_tables = {}

        def page_table_for(representative: int) -> torch.Tensor:
            if representative not in self.cpp_page_tables:
                source_pages = score_workspace.page_ids_device[
                    score_workspace.representative_slots[representative],
                    : self.request_count,
                ]
                if source_pages.device != self.device or source_pages.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("page tables must be integer tensors on the pool device")
                self.cpp_page_tables[representative] = (
                    source_pages,
                    torch.empty(
                        source_pages.shape,
                        dtype=torch.int32,
                        device=self.device,
                    ).contiguous(),
                )
            return self.cpp_page_tables[representative][1]

        dense_entries = [
            (
                layer,
                layer_pools[layer],
                page_table_for(layer_group_representative[layer]),
            )
            for layer in self.dense_layers
        ]

        self.swa_source = None
        self.swa_indices = None
        self.swa_destination_base = None
        self.swa_offsets = None
        self.swa_source_offsets = None
        swa_entries = []
        if self.swa_layers:
            if swa_window is None or swa_window <= 0 or self.keep_count < swa_window:
                raise ValueError("SWA compaction requires a valid retained window")
            swa_move_counts = [int(swa_window) + length for length in self.protected_tail_lengths]
            swa_move_offsets = [0]
            for count in swa_move_counts:
                swa_move_offsets.append(swa_move_offsets[-1] + count)
            self._swa_move_offsets = tuple(swa_move_offsets)
            total_swa = swa_move_offsets[-1]
            self.swa_source_offsets = tuple(
                torch.arange(
                    -int(swa_window),
                    length,
                    dtype=torch.int32,
                    device=self.device,
                )
                for length in self.protected_tail_lengths
            )
            self.swa_source = torch.empty(total_swa, dtype=torch.int32, device=self.device)
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
                        page_tables=page_tables,
                        pool_pointers=torch.tensor(
                            [pool.data_ptr() for pool in pools],
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        page_table_pointers=torch.tensor(
                            [page_table.data_ptr() for page_table in page_tables],
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        source_layer_indices=source_layer_indices,
                    )
                )
            return tuple(result)

        self.cpp_dense_groups = compact_groups(dense_entries, "dense")
        self.cpp_swa_groups = compact_groups(swa_entries, "swa")

    def launch(self) -> None:
        """Stage dynamic metadata and run C++ dense/SWA compact launches."""
        if self.swa_source is not None:
            assert self.swa_source_offsets is not None
            assert self.swa_indices is not None
            uniform_tail = self._uniform_protected_tail_length
            if uniform_tail is not None:
                source_offsets = self.swa_source_offsets[0]
                torch.add(
                    self.score_workspace.valid_seq_lens_device[: self.request_count].view(
                        self.request_count, 1
                    ),
                    source_offsets.view(1, -1),
                    out=self.swa_source.view(self.request_count, -1),
                )
            else:
                for request_index, source_offsets in enumerate(self.swa_source_offsets):
                    begin = self._swa_move_offsets[request_index]
                    end = self._swa_move_offsets[request_index + 1]
                    torch.add(
                        self.score_workspace.valid_seq_lens_device[request_index],
                        source_offsets,
                        out=self.swa_source[begin:end],
                    )
            self.swa_indices.copy_(self.swa_source.reshape(1, -1).expand(self.cpp_num_kv_heads, -1))
        for source_pages, staged_i32 in self.cpp_page_tables.values():
            staged_i32.copy_(source_pages)
        self._stage_dense_indices()
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

    def _stage_dense_indices(self) -> None:
        """Pack selected decode tokens and opaque protected tail."""
        keep = self.selection_workspace.keep[: self.request_count]
        decode_keep = keep[..., self.prompt_len :]
        per_layer = None
        if self.eviction_mode == "per_layer_perhead":
            per_layer = decode_keep.view(
                self.request_count,
                len(self.dense_layers),
                self.cpp_num_kv_heads,
                self.decode_keep_count,
            )
        uniform_tail = self._uniform_protected_tail_length
        if uniform_tail is not None:
            move_count = self.decode_keep_count + uniform_tail
            if self.eviction_mode == "per_layer_perhead":
                assert per_layer is not None
                target = self.cpp_indices.view(
                    len(self.dense_layers),
                    self.cpp_num_kv_heads,
                    self.request_count,
                    move_count,
                )
                target_keep = target[:, :, :, : self.decode_keep_count]
                target_keep.copy_(per_layer.permute(1, 2, 0, 3))
            else:
                target = self.cpp_indices.view(
                    self.cpp_num_kv_heads,
                    self.request_count,
                    move_count,
                )
                target_keep = target[:, :, : self.decode_keep_count]
                if self.eviction_mode == "union":
                    target_keep.copy_(
                        decode_keep.view(
                            1,
                            self.request_count,
                            self.decode_keep_count,
                        )
                    )
                else:
                    target_keep.copy_(decode_keep.permute(1, 0, 2))
            if uniform_tail:
                torch.add(
                    self.score_workspace.valid_seq_lens_device[: self.request_count].view(
                        self.request_count, 1
                    ),
                    self.protected_tail_offsets[:uniform_tail].view(1, uniform_tail),
                    out=self.protected_tail_source[:, :uniform_tail],
                )
                tail_source = self.protected_tail_source[:, :uniform_tail]
                if self.eviction_mode == "per_layer_perhead":
                    target[:, :, :, self.decode_keep_count :].copy_(
                        tail_source.view(1, 1, self.request_count, uniform_tail).expand(
                            len(self.dense_layers),
                            self.cpp_num_kv_heads,
                            self.request_count,
                            uniform_tail,
                        )
                    )
                else:
                    target[:, :, self.decode_keep_count :].copy_(
                        tail_source.view(1, self.request_count, uniform_tail).expand(
                            self.cpp_num_kv_heads,
                            self.request_count,
                            uniform_tail,
                        )
                    )
            return
        for request_index, tail_length in enumerate(self.protected_tail_lengths):
            begin = self._move_offsets[request_index]
            keep_end = begin + self.decode_keep_count
            if self.eviction_mode == "per_layer_perhead":
                assert per_layer is not None
                target_keep = self.cpp_indices[:, :, begin:keep_end]
                target_keep.copy_(per_layer[request_index])
            else:
                target_keep = self.cpp_indices[:, begin:keep_end]
                if self.eviction_mode == "union":
                    target_keep.copy_(
                        decode_keep[request_index]
                        .view(1, self.decode_keep_count)
                        .expand_as(target_keep)
                    )
                else:
                    target_keep.copy_(decode_keep[request_index])
            if tail_length:
                torch.add(
                    self.score_workspace.valid_seq_lens_device[request_index],
                    self.protected_tail_offsets[:tail_length],
                    out=self.protected_tail_source[request_index, :tail_length],
                )
                tail_source = self.protected_tail_source[request_index, :tail_length]
                if self.eviction_mode == "per_layer_perhead":
                    self.cpp_indices[:, :, keep_end : keep_end + tail_length].copy_(
                        tail_source.view(1, 1, tail_length).expand(
                            len(self.dense_layers), self.cpp_num_kv_heads, tail_length
                        )
                    )
                else:
                    self.cpp_indices[:, keep_end : keep_end + tail_length].copy_(
                        tail_source.view(1, tail_length).expand(self.cpp_num_kv_heads, tail_length)
                    )
