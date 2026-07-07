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

"""Fixed-buffer CUDA Graph support for TriAttention eviction.

The graph owns only the CUDA work from phase preparation through score,
selection, and physical compaction. Scheduler decisions, page-table uploads,
request bookkeeping, and KV-cache resize remain on the host path.
"""

import os

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


def _tensor_fingerprint(tensor: torch.Tensor) -> tuple:
    """Return allocation-address and tensor-layout fields used by a graph."""
    storage = tensor.untyped_storage()
    return (
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _unique_tensor_nbytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: Dict[Tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), int(storage.data_ptr()))] = int(storage.nbytes())
    return sum(storages.values())


@dataclass(frozen=True)
class _FixedCompactLaunch:
    pool: torch.Tensor
    page_ids: torch.Tensor
    source: torch.Tensor
    destination: torch.Tensor
    page_base: torch.Tensor
    scratch: torch.Tensor

    def run(self) -> None:
        from .triattention_kernels import triton_tri_compact_fixed

        triton_tri_compact_fixed(
            self.pool,
            self.page_ids,
            self.source,
            self.destination,
            self.page_base,
            self.scratch,
        )


class FixedBatchedCompactionWorkspace:
    """Caller-owned upper-bucket buffers for graph-captured compaction.

    Dense layers move only selected decode slots. The finalized prompt is never
    a destination and therefore remains byte-untouched. Kernel-masked SWA
    layers use their own page table and copy only the latest window to the new
    compacted tail.
    """

    def __init__(
        self,
        *,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        swa_layers: List[int],
        layer_group_representative: Dict[int, int],
        global_layers: List[int],
        score_workspace,
        selection_workspace,
        request_count: int,
        seq_len: int,
        prompt_len: int,
        decode_keep_count: int,
        swa_window: Optional[int],
        arena_generation: int,
    ) -> None:
        if request_count <= 0 or decode_keep_count <= 0:
            raise ValueError("fixed graph compaction requires a non-empty cohort and keep set")
        if not dense_layers:
            raise ValueError("fixed graph compaction requires at least one dense layer")
        if request_count > selection_workspace.max_requests:
            raise ValueError("fixed graph compaction exceeds the selection workspace")
        if (
            selection_workspace.prompt_len != prompt_len
            or selection_workspace.keep_count != decode_keep_count
            or selection_workspace.width != seq_len - prompt_len
        ):
            raise ValueError("fixed graph compaction geometry does not match selection")

        self.device = layer_pools[dense_layers[0]].device
        self.request_count = int(request_count)
        self.seq_len = int(seq_len)
        self.prompt_len = int(prompt_len)
        self.decode_keep_count = int(decode_keep_count)
        self.keep_count = self.prompt_len + self.decode_keep_count
        self.arena_generation = int(arena_generation)
        self.dense_layers = tuple(int(layer) for layer in dense_layers)
        self.swa_layers = tuple(int(layer) for layer in swa_layers)
        self.global_layers = tuple(int(layer) for layer in global_layers)
        self.storage_groups = tuple(
            (int(layer), int(layer_group_representative[layer])) for layer in dense_layers
        )
        self.selection_workspace = selection_workspace
        self.score_workspace = score_workspace
        self.layer_pools = tuple(layer_pools)

        total_decode = self.request_count * self.decode_keep_count
        self.dense_source = torch.empty(total_decode, dtype=torch.int64, device=self.device)
        decode_destination = torch.arange(
            self.prompt_len,
            self.keep_count,
            dtype=torch.int64,
            device=self.device,
        )
        self.dense_destination = decode_destination.repeat(self.request_count)
        request_page_base = (
            torch.arange(self.request_count, dtype=torch.int64, device=self.device)
            * score_workspace.page_count
        )
        self.dense_page_base = request_page_base.repeat_interleave(self.decode_keep_count)

        scratch_by_geometry = {}
        self.dense_launches = []
        for layer in self.dense_layers:
            representative = layer_group_representative[layer]
            page_ids = self._page_ids_for(score_workspace, representative)
            scratch = self._scratch_for(layer_pools[layer], total_decode, scratch_by_geometry)
            self.dense_launches.append(
                _FixedCompactLaunch(
                    layer_pools[layer],
                    page_ids,
                    self.dense_source,
                    self.dense_destination,
                    self.dense_page_base,
                    scratch,
                )
            )

        # Optional cpp backend: ONE sparse_kv_cache_compact op launch per dense
        # layer replaces the two-phase Triton pair. Uses the FULL keep list
        # (prompt tokens have src == dst and the upstream kernel skips them),
        # so no destination remap is needed. All buffers are fixed-shape and
        # owned here; the launch path only issues copies + op calls (capture-safe).
        first_dense_pool = layer_pools[self.dense_layers[0]]
        cpp_num_kv_heads = int(first_dense_pool.shape[2]) if first_dense_pool.ndim == 5 else -1
        self.cpp_backend = (
            os.environ.get("TRIATTN_COMPACT_BACKEND") == "cpp"
            and hasattr(torch.ops.trtllm, "sparse_kv_cache_compact")
            and all(
                layer_pools[layer].ndim == 5
                and layer_pools[layer].shape[1] == 2
                and int(layer_pools[layer].shape[2]) == cpp_num_kv_heads
                and layer_pools[layer].is_contiguous()
                for layer in self.dense_layers
            )
        )
        if self.cpp_backend:
            self.cpp_num_kv_heads = cpp_num_kv_heads
            self.cpp_indices = torch.empty(
                cpp_num_kv_heads,
                self.request_count * self.keep_count,
                dtype=torch.int32,
                device=self.device,
            )
            self.cpp_offsets = (
                torch.arange(self.request_count + 1, dtype=torch.int32, device=self.device)
                * self.keep_count
            )
            self.cpp_page_tables = {}
            self.cpp_dense_ops = []
            for layer in self.dense_layers:
                representative = layer_group_representative[layer]
                if representative not in self.cpp_page_tables:
                    # 2-D [request_count, page_count] (the op's page-table shape;
                    # the Triton fixed path flattens this same plane to 1-D).
                    source_pages = score_workspace.page_ids_device[
                        score_workspace.representative_slots[representative],
                        : self.request_count,
                    ]
                    self.cpp_page_tables[representative] = (
                        source_pages,
                        torch.empty(
                            source_pages.shape, dtype=torch.int32, device=self.device
                        ).contiguous(),
                    )
                self.cpp_dense_ops.append(
                    (layer_pools[layer], self.cpp_page_tables[representative][1])
                )

        self.swa_launches = []
        self.swa_source = None
        self.swa_destination = None
        self.swa_page_base = None
        self.swa_source_offsets = None
        if self.swa_layers:
            if swa_window is None or swa_window <= 0 or self.keep_count < swa_window:
                raise ValueError("fixed graph SWA compaction requires a valid retained window")
            total_swa = self.request_count * int(swa_window)
            self.swa_source_offsets = torch.arange(
                -int(swa_window),
                0,
                dtype=torch.int64,
                device=self.device,
            )
            destination = torch.arange(
                self.keep_count - int(swa_window),
                self.keep_count,
                dtype=torch.int64,
                device=self.device,
            )
            self.swa_source = torch.empty(total_swa, dtype=torch.int64, device=self.device)
            self.swa_destination = destination.repeat(self.request_count)
            self.swa_page_base = request_page_base.repeat_interleave(int(swa_window))
            for layer in self.swa_layers:
                page_ids = self._page_ids_for(score_workspace, layer)
                scratch = self._scratch_for(layer_pools[layer], total_swa, scratch_by_geometry)
                self.swa_launches.append(
                    _FixedCompactLaunch(
                        layer_pools[layer],
                        page_ids,
                        self.swa_source,
                        self.swa_destination,
                        self.swa_page_base,
                        scratch,
                    )
                )

        owned_tensors = [
            self.dense_source,
            self.dense_destination,
            self.dense_page_base,
        ]
        if self.swa_source_offsets is not None:
            owned_tensors.append(self.swa_source_offsets)
        strong_refs = [*owned_tensors, *layer_pools]
        if self.cpp_backend:
            cpp_page_sources = [source for source, _ in self.cpp_page_tables.values()]
            cpp_page_staging = [staging for _, staging in self.cpp_page_tables.values()]
            owned_tensors.extend(
                (self.cpp_indices, self.cpp_offsets, *cpp_page_staging)
            )
            strong_refs.extend(
                (
                    self.cpp_indices,
                    self.cpp_offsets,
                    *cpp_page_sources,
                    *cpp_page_staging,
                )
            )
        for launch in (*self.dense_launches, *self.swa_launches):
            strong_refs.extend(
                (
                    launch.page_ids,
                    launch.source,
                    launch.destination,
                    launch.page_base,
                    launch.scratch,
                )
            )
            owned_tensors.extend(
                (
                    launch.source,
                    launch.destination,
                    launch.page_base,
                    launch.scratch,
                )
            )
        self._tensor_refs = tuple(dict.fromkeys(strong_refs))
        self._owned_tensor_refs = tuple(dict.fromkeys(owned_tensors))
        self.nbytes = _unique_tensor_nbytes(self._owned_tensor_refs)

    def _page_ids_for(self, score_workspace, representative: int) -> torch.Tensor:
        slot = score_workspace.representative_slots[representative]
        page_ids = score_workspace.page_ids_device[slot, : self.request_count]
        if not page_ids.is_contiguous():
            raise ValueError("fixed graph page-id view must be contiguous")
        return page_ids.view(-1)

    @staticmethod
    def _scratch_for(
        pool: torch.Tensor,
        total_tokens: int,
        scratch_by_geometry: dict,
    ) -> torch.Tensor:
        _, kv_factor, num_kv_heads, _, head_dim = pool.shape
        key = (
            str(pool.device),
            pool.dtype,
            int(kv_factor),
            int(num_kv_heads),
            int(total_tokens),
            int(head_dim),
        )
        scratch = scratch_by_geometry.get(key)
        if scratch is None:
            scratch = torch.empty(
                kv_factor * num_kv_heads * total_tokens * head_dim,
                dtype=pool.dtype,
                device=pool.device,
            )
            scratch_by_geometry[key] = scratch
        return scratch

    def launch(self) -> None:
        """Copy dynamic keep indices, then run fixed dense and SWA launches."""
        if self.swa_source is not None:
            assert self.swa_source_offsets is not None
            torch.add(
                self.score_workspace.valid_seq_lens_device[: self.request_count].view(
                    self.request_count, 1
                ),
                self.swa_source_offsets.view(1, -1),
                out=self.swa_source.view(self.request_count, -1),
            )
        if self.cpp_backend:
            # FULL keep list (prompt + selected decode), replicated per KV head
            # with an int64 -> int32 cast in one captured copy.
            keep_full = self.selection_workspace.keep[: self.request_count]
            self.cpp_indices.copy_(
                keep_full.reshape(1, -1).expand(self.cpp_num_kv_heads, -1)
            )
            for source_pages, staged_i32 in self.cpp_page_tables.values():
                staged_i32.copy_(source_pages)
            for pool, page_table in self.cpp_dense_ops:
                torch.ops.trtllm.sparse_kv_cache_compact(
                    pool, page_table, self.cpp_indices, self.cpp_offsets
                )
        else:
            selected_decode = self.selection_workspace.keep[
                : self.request_count, self.prompt_len :
            ]
            self.dense_source.view(self.request_count, self.decode_keep_count).copy_(
                selected_decode
            )
            for launch in self.dense_launches:
                launch.run()
        for launch in self.swa_launches:
            launch.run()

    def matches_runtime(
        self,
        *,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        swa_layers: List[int],
        layer_group_representative: Dict[int, int],
        global_layers: List[int],
        score_workspace,
        selection_workspace,
    ) -> bool:
        if (
            score_workspace is not self.score_workspace
            or selection_workspace is not self.selection_workspace
            or len(layer_pools) != len(self.layer_pools)
            or tuple(dense_layers) != self.dense_layers
            or tuple(swa_layers) != self.swa_layers
            or tuple(global_layers) != self.global_layers
            or tuple((int(layer), int(layer_group_representative[layer])) for layer in dense_layers)
            != self.storage_groups
        ):
            return False
        return all(
            _tensor_fingerprint(current) == _tensor_fingerprint(captured)
            for current, captured in zip(layer_pools, self.layer_pools)
        )

    def pointer_fingerprint(self, stream: torch.cuda.Stream) -> tuple:
        selection_tensors = tuple(
            (name, _tensor_fingerprint(tensor))
            for name, tensor in self.selection_workspace.named_tensors()
        )
        score_tensors = [
            ("page_ids_device", _tensor_fingerprint(self.score_workspace.page_ids_device)),
            (
                "round_starts_device",
                _tensor_fingerprint(self.score_workspace.round_starts_device),
            ),
            (
                "valid_seq_lens_device",
                _tensor_fingerprint(self.score_workspace.valid_seq_lens_device),
            ),
            ("phase_base", _tensor_fingerprint(self.score_workspace.phase_base)),
            ("phase", _tensor_fingerprint(self.score_workspace.phase)),
            ("cos_phase", _tensor_fingerprint(self.score_workspace.cos_phase)),
            ("sin_phase", _tensor_fingerprint(self.score_workspace.sin_phase)),
            ("mean_cos", _tensor_fingerprint(self.score_workspace.mean_cos)),
            ("mean_sin", _tensor_fingerprint(self.score_workspace.mean_sin)),
            ("offsets", _tensor_fingerprint(self.score_workspace.offsets)),
            ("omega", _tensor_fingerprint(self.score_workspace.omega)),
        ]
        fused_group = self.score_workspace.fused_group
        group_tensors = (
            *fused_group.pointer_prefix,
            *fused_group.pointer_middle,
            *fused_group.pointer_tail,
            fused_group.output,
            fused_group.seg_offsets,
        )
        score_tensors.extend(
            (f"group.fused.{index}", _tensor_fingerprint(tensor))
            for index, tensor in enumerate(group_tensors)
        )
        context_token = int(torch.cuda.current_blas_handle())
        return (
            "triattention.standalone-eviction-graph.v2",
            self.request_count,
            self.seq_len,
            self.prompt_len,
            self.decode_keep_count,
            self.dense_layers,
            self.swa_layers,
            self.global_layers,
            self.storage_groups,
            self.selection_workspace.selection_backend,
            self.arena_generation,
            (str(stream.device), int(stream.cuda_stream), context_token),
            tuple(_tensor_fingerprint(pool) for pool in self.layer_pools),
            tuple(score_tensors),
            selection_tensors,
            tuple(_tensor_fingerprint(tensor) for tensor in self._tensor_refs),
        )


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    capture_stream: object
    fingerprint: tuple
    workspace: FixedBatchedCompactionWorkspace
    nbytes: int
    last_use_event: Optional[object] = None


class _NeverCompleteEvent:
    """Conservatively retain an entry after an indeterminate replay failure."""

    @staticmethod
    def query() -> bool:
        return False


class StandaloneEvictionGraphCache:
    """Bounded graph cache with event-retired pointer lifetime management.

    ``max_bytes`` accounts for graph-owned compaction buffers and the measured
    allocator growth during capture. Score/selection buffers and live KV pools
    pre-exist the graph cache, but each entry retains strong references to their
    owners until its last-use event.
    """

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("standalone graph cache bounds must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.retired = []
        self.disabled = set()
        self.counts = {
            "attempt": 0,
            "attempted_requests": 0,
            "capture": 0,
            "launch": 0,
            "cache_hit": 0,
            "covered_requests": 0,
            # Historical observers treat replay as every successful
            # CUDAGraph.replay(), including the first launch after capture.
            "replay": 0,
            "fallback": 0,
            "invalidated": 0,
            "failure": 0,
            "capture_failure": 0,
            "replay_failure": 0,
            "retired": 0,
        }
        self.bucket_counts = OrderedDict()
        self.last_error: Optional[dict[str, str]] = None

    def _bucket_for(self, key: tuple, request_count: int) -> dict:
        if request_count <= 0:
            raise ValueError("standalone graph request count must be positive")
        bucket = self.bucket_counts.get(key)
        if bucket is None:
            bucket = {
                "request_count": int(request_count),
                "attempt": 0,
                "attempted_requests": 0,
                "capture": 0,
                "launch": 0,
                "cache_hit": 0,
                "covered_requests": 0,
                "fallback": 0,
                "invalidated": 0,
                "failure": 0,
                "capture_failure": 0,
                "replay_failure": 0,
            }
            self.bucket_counts[key] = bucket
        elif bucket["request_count"] != request_count:
            raise ValueError("standalone graph key changed request-count semantics")
        return bucket

    def _record_attempt(self, key: tuple, request_count: int) -> dict:
        bucket = self._bucket_for(key, request_count)
        self.counts["attempt"] += 1
        self.counts["attempted_requests"] += request_count
        bucket["attempt"] += 1
        bucket["attempted_requests"] += request_count
        return bucket

    def _record_fallback(self, bucket: dict) -> None:
        self.counts["fallback"] += 1
        bucket["fallback"] += 1

    def record_fallback(self, *, key: tuple, request_count: int) -> None:
        """Record a fail-safe fallback rejected before workspace mutation."""
        self._record_fallback(self._record_attempt(key, request_count))

    @staticmethod
    def _event_complete(event: Optional[object]) -> bool:
        return event is None or bool(event.query())

    @staticmethod
    def _reset_graph(entry: _GraphEntry) -> None:
        entry.graph.reset()

    def _collect_retired(self) -> None:
        pending = []
        for entry in self.retired:
            if self._event_complete(entry.last_use_event):
                self._reset_graph(entry)
            else:
                pending.append(entry)
        self.retired = pending

    def _retire(self, entry: _GraphEntry) -> None:
        if self._event_complete(entry.last_use_event):
            self._reset_graph(entry)
        else:
            self.retired.append(entry)
            self.counts["retired"] += 1

    def _resident_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries.values()) + sum(
            entry.nbytes for entry in self.retired
        )

    def workspace_for(self, key: tuple) -> Optional[FixedBatchedCompactionWorkspace]:
        entry = self.entries.get(key)
        return None if entry is None else entry.workspace

    def classify(self, key: tuple, fingerprint: tuple) -> str:
        if key in self.disabled:
            return "fallback"
        entry = self.entries.get(key)
        return "replay" if entry is not None and entry.fingerprint == fingerprint else "capture"

    def is_disabled(self, key: tuple) -> bool:
        return key in self.disabled

    def snapshot(self) -> dict:
        return {
            **self.counts,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "active_entries": len(self.entries),
            "retired_entries": len(self.retired),
            "disabled_buckets": len(self.disabled),
            "owned_bytes": self._resident_bytes(),
            "buckets": [
                {
                    "key": key,
                    **counts,
                }
                for key, counts in self.bucket_counts.items()
            ],
            "last_error": self.last_error,
        }

    def _make_room(self, nbytes: int) -> bool:
        self._collect_retired()
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self._resident_bytes() + nbytes > self.max_bytes
        ):
            victim_key = next(
                (
                    key
                    for key, entry in self.entries.items()
                    if self._event_complete(entry.last_use_event)
                ),
                None,
            )
            if victim_key is None:
                return False
            self._retire(self.entries.pop(victim_key))
        return (
            len(self.entries) < self.max_entries
            and self._resident_bytes() + nbytes <= self.max_bytes
        )

    @staticmethod
    def _capture_graph(
        workspace, capture_body: Callable[[], None]
    ) -> Tuple[torch.cuda.CUDAGraph, torch.cuda.Stream, int]:
        current_stream = torch.cuda.current_stream(workspace.device)
        capture_stream = torch.cuda.Stream(device=workspace.device)
        capture_stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        allocated_before = int(torch.cuda.memory_allocated(workspace.device))
        try:
            with torch.cuda.stream(capture_stream):
                # Fixed buffers may be inference tensors, while resource
                # preparation runs outside ModelEngine.forward's scope. Keep
                # graph setup and capture_end in the same inference context as
                # the captured body because both may update inference tensors.
                with torch.inference_mode():
                    with torch.cuda.graph(
                        graph,
                        stream=capture_stream,
                        capture_error_mode="thread_local",
                    ):
                        capture_body()
        except RuntimeError:
            graph.reset()
            raise
        current_stream.wait_stream(capture_stream)
        graph_bytes = max(
            0,
            int(torch.cuda.memory_allocated(workspace.device)) - allocated_before,
        )
        return graph, capture_stream, graph_bytes

    @staticmethod
    def _record_last_use(workspace) -> object:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(workspace.device))
        return event

    def execute(
        self,
        *,
        key: tuple,
        request_count: int,
        fingerprint: tuple,
        workspace: FixedBatchedCompactionWorkspace,
        capture_body: Callable[[], None],
    ) -> str:
        """Replay/capture one exact graph or return ``fallback`` before mutation.

        A capture failure is safe to fall back because stream capture records
        operations without executing them. Once replay is attempted, errors are
        propagated and the caller must not issue eager compaction for that step.
        """
        bucket = self._record_attempt(key, request_count)
        self._collect_retired()
        if key in self.disabled:
            self._record_fallback(bucket)
            return "fallback"

        entry = self.entries.get(key)
        if entry is not None and entry.fingerprint != fingerprint:
            self.entries.pop(key)
            self._retire(entry)
            self.counts["invalidated"] += 1
            bucket["invalidated"] += 1
            entry = None

        if entry is None:
            if not self._make_room(workspace.nbytes):
                self._record_fallback(bucket)
                return "fallback"
            try:
                graph, capture_stream, graph_bytes = self._capture_graph(workspace, capture_body)
            except RuntimeError as exc:
                self.disabled.add(key)
                self.counts["failure"] += 1
                self.counts["capture_failure"] += 1
                bucket["failure"] += 1
                bucket["capture_failure"] += 1
                self._record_fallback(bucket)
                self.last_error = {
                    "phase": "capture",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                return "fallback"
            entry_bytes = workspace.nbytes + graph_bytes
            if not self._make_room(entry_bytes):
                graph.reset()
                self._record_fallback(bucket)
                return "fallback"
            entry = _GraphEntry(
                graph=graph,
                capture_stream=capture_stream,
                fingerprint=fingerprint,
                workspace=workspace,
                nbytes=entry_bytes,
            )
            self.entries[key] = entry
            self.counts["capture"] += 1
            bucket["capture"] += 1
            outcome = "capture"
        else:
            self.entries.move_to_end(key)
            outcome = "replay"

        try:
            entry.graph.replay()
        except RuntimeError as exc:
            self.entries.pop(key, None)
            self.disabled.add(key)
            # A replay error does not prove that no graph nodes were submitted.
            # Keep every graph-visible reference alive for the failed process.
            entry.last_use_event = _NeverCompleteEvent()
            self.retired.append(entry)
            self.counts["retired"] += 1
            self.counts["failure"] += 1
            self.counts["replay_failure"] += 1
            bucket["failure"] += 1
            bucket["replay_failure"] += 1
            self.last_error = {
                "phase": "replay",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        entry.last_use_event = self._record_last_use(workspace)
        self.counts["launch"] += 1
        self.counts["covered_requests"] += request_count
        self.counts["replay"] += 1
        bucket["launch"] += 1
        bucket["covered_requests"] += request_count
        if outcome == "replay":
            self.counts["cache_hit"] += 1
            bucket["cache_hit"] += 1
        return outcome
