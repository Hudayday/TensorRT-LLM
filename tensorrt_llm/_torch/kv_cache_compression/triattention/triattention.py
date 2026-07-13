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

"""TriAttention KV-cache compression: periodic physical KV eviction.

Every ``beta`` confirmed generation tokens TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the eviction runs in the compression manager's final
``on_generation_step_end`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager` and nothing more -- it
has no attention backend of its own; decode runs the model's standard dense
kernel over the compacted cache. TriAttention derives each request's effective
confirmed physical length after V2's native update/rewind and writes that value
through ``adjust_attention_metadata`` just before ``attn_metadata.prepare()``.
Physical reclaim uses V2's existing resize path directly after compaction. An
already-enqueued speculative suffix is excluded from scoring and appended
unchanged to the retained prefix by the same per-layer compact operation.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The model engine keeps the decode query at its true absolute
position while the attention metadata uses the compacted physical length, so a
query against a kept key at its original rotation still yields the correct
relative distance.

Calibration is NOT computed here: the user calibrates with the official tool
(github.com/WeianMao/triattention) and passes that .pt via ``calibration_path``;
the manager converts it to our runtime schema at load (see _resolve_calibration).
The scoring math follows the same upstream reference (``methods/pruning_utils.py``).
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from tensorrt_llm._torch.kv_cache_compression.attention_metadata import (
    requires_paged_draft_kv_length_domain,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
    from tensorrt_llm.llmapi.llm_args import SpeculativeConfig


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

# Bound eager workspace memory independently of CUDA Graph support. A large
# due cohort is processed as consecutive request chunks with identical results.
_EAGER_REQUEST_CHUNK_SIZE = 32
_EAGER_RESOURCE_CACHE_LIMIT = 3
_EAGER_COMPACTION_CACHE_LIMIT = 6


def _build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Upstream pruning_utils.build_geometric_offsets: [1, 2, 4, ... <=max]."""
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _canonical_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _topk_indices_into(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    indices_i32: torch.Tensor,
    keep_count: int,
) -> None:
    """Write per-row top-k indices with the CuTE-DSL selector."""
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(scores, seq_lens, indices_i32, keep_count, 1)


def _deterministic_topk_indices_into(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    indices_i32: torch.Tensor,
    keep_count: int,
) -> None:
    """Write TopK indices with deterministic exact-boundary tie membership."""
    _topk_indices_into(scores, seq_lens, indices_i32, keep_count)
    if scores.is_cuda:
        from .triattention_kernels import canonicalize_topk_scores

        canonicalize_topk_scores(scores, seq_lens, indices_i32, keep_count)
    else:
        # CPU workspaces only exercise the selector contract in unit tests.
        # Keep this reference path independent of any production TopK fallback.
        selected_scores = torch.gather(scores, 1, indices_i32.to(torch.long))
        threshold = torch.amin(selected_scores, dim=1, keepdim=True)
        token_indices = torch.arange(
            scores.shape[1], dtype=torch.float32, device=scores.device
        ).view(1, -1)
        valid = token_indices < seq_lens.view(-1, 1)
        canonical_scores = torch.where(
            scores > threshold,
            1.0,
            torch.where(scores == threshold, -token_indices, float("-inf")),
        )
        canonical_scores = torch.where(valid, canonical_scores, float("-inf"))
        _topk_indices_into(canonical_scores, seq_lens, indices_i32, keep_count)
        return
    _topk_indices_into(scores, seq_lens, indices_i32, keep_count)


class _CrossRequestSelectionPlan(NamedTuple):
    """Selection dimensions used to allocate reusable eager buffers."""

    eviction_mode: str
    dense_layers: Tuple[int, ...]
    num_query_heads: int
    num_kv_heads: int
    rows: int
    width: int
    keep_count: int
    prompt_len: int
    dtype: torch.dtype
    device: torch.device
    selection_backend: str
    max_requests: int


class _RuntimeKVLayout(NamedTuple):
    """Manager-lifetime layer and pool views used by every eviction."""

    manager: object
    num_layers: int
    global_layers: List[int]
    layer_pools: List[torch.Tensor]
    dense_layers: List[int]
    swa_layers: List[int]
    swa_window: Optional[int]
    storage_groups: Dict[object, List[int]]
    layer_group_representative: Dict[int, int]
    layer_pool_keys: Tuple[object, ...]
    pool_representatives: Tuple[int, ...]
    pool_view_fingerprint: Tuple[tuple, ...]


class _BatchedFixedUnionWorkspace:
    """Persistent ``[request, ...]`` buffers for union selection."""

    def __init__(
        self,
        rows: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        selection_backend: str,
        max_requests: int,
        dense_layers: Tuple[int, ...] = (),
        num_query_heads: int = 0,
        num_kv_heads: int = 0,
    ) -> None:
        if rows <= 0 or width <= keep_count or keep_count <= 0:
            raise ValueError("cross-request selection requires rows > 0 and width > keep_count > 0")
        if max_requests <= 0:
            raise ValueError("cross-request selection requires a positive request capacity")
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported cross-request selection backend: {selection_backend}")
        self.max_requests = max_requests
        self.eviction_mode = "union"
        self.dense_layers = tuple(dense_layers)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.selection_backend = selection_backend
        self.rows = rows
        self.width = width
        self.keep_count = keep_count
        self.prompt_len = prompt_len
        self.total_keep = prompt_len + keep_count
        self.dtype = dtype
        self.device = _canonical_device(device)

        shape = (max_requests, rows, width)
        self.input_scores = torch.empty(shape, dtype=dtype, device=self.device)
        self.row_mean = torch.empty((max_requests, rows, 1), dtype=dtype, device=self.device)
        self.row_std = torch.empty_like(self.row_mean)
        self.combined = torch.empty((max_requests, width), dtype=dtype, device=self.device)
        self.combined_argmax = torch.empty(
            (max_requests, width), dtype=torch.long, device=self.device
        )
        self.final_indices = torch.empty(
            (max_requests, keep_count), dtype=torch.long, device=self.device
        )
        self.sorted_indices = torch.empty_like(self.final_indices)
        self.sort_order = torch.empty_like(self.final_indices)
        self.keep = torch.empty(
            (max_requests, self.total_keep), dtype=torch.long, device=self.device
        )
        self.valid_widths = torch.full(
            (max_requests,), width, dtype=torch.int32, device=self.device
        )
        self.valid_scale = torch.empty((max_requests, 1, 1), dtype=dtype, device=self.device)
        self.token_indices = torch.arange(width, dtype=torch.long, device=self.device)
        self.invalid_mask = torch.empty(
            (max_requests, 1, width), dtype=torch.bool, device=self.device
        )
        if prompt_len:
            prompt = torch.arange(prompt_len, dtype=torch.long, device=self.device)
            self.keep[:, :prompt_len].copy_(prompt.expand(max_requests, -1))
        self.final_indices_i32 = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=self.device
        )

    def named_tensors(self) -> Tuple[Tuple[str, torch.Tensor], ...]:
        """Return the fixed tensor inventory owned by this selection bucket."""
        tensors = (
            ("input_scores", self.input_scores),
            ("row_mean", self.row_mean),
            ("row_std", self.row_std),
            ("combined", self.combined),
            ("combined_argmax", self.combined_argmax),
            ("final_indices", self.final_indices),
            ("sorted_indices", self.sorted_indices),
            ("sort_order", self.sort_order),
            ("keep", self.keep),
            ("valid_widths", self.valid_widths),
            ("valid_scale", self.valid_scale),
            ("token_indices", self.token_indices),
            ("invalid_mask", self.invalid_mask),
        )
        return tensors + (("final_indices_i32", self.final_indices_i32),)

    def stage_valid_widths_from_seq_lens(
        self,
        valid_seq_lens: torch.Tensor,
        request_count: int,
    ) -> None:
        """Derive each request's valid decode width on the active device."""
        if (
            request_count <= 0
            or request_count > self.max_requests
            or valid_seq_lens.ndim != 1
            or valid_seq_lens.numel() < request_count
            or valid_seq_lens.dtype != torch.int32
            or valid_seq_lens.device != self.device
        ):
            raise ValueError("valid sequence lengths do not fit the selection bucket")
        torch.sub(
            valid_seq_lens[:request_count],
            self.prompt_len,
            out=self.valid_widths[:request_count],
        )

    def _select_input_scores(self, request_count: int, *, normalize_scores: bool) -> None:
        input_scores = self.input_scores[:request_count]
        valid_widths = self.valid_widths[:request_count]
        invalid_mask = self.invalid_mask[:request_count]
        torch.ge(
            self.token_indices.view(1, 1, self.width),
            valid_widths.view(request_count, 1, 1),
            out=invalid_mask,
        )
        if normalize_scores:
            row_mean = self.row_mean[:request_count]
            row_std = self.row_std[:request_count]
            input_scores.masked_fill_(invalid_mask, 0.0)
            torch.sum(input_scores, dim=2, keepdim=True, out=row_mean)
            self.valid_scale[:request_count].view(request_count).copy_(valid_widths)
            row_mean.div_(self.valid_scale[:request_count])
            torch.sub(input_scores, row_mean, out=input_scores)
            input_scores.masked_fill_(invalid_mask, 0.0)
            torch.linalg.vector_norm(
                input_scores,
                dim=2,
                keepdim=True,
                out=row_std,
            )
            self.valid_scale[:request_count].sqrt_()
            row_std.div_(self.valid_scale[:request_count])
            row_std.clamp_min_(1e-6)
            torch.div(input_scores, row_std, out=input_scores)
        input_scores.masked_fill_(invalid_mask, float("-inf"))

        combined = self.combined[:request_count]
        combined_argmax = self.combined_argmax[:request_count]
        final_indices = self.final_indices[:request_count]
        sorted_indices = self.sorted_indices[:request_count]
        sort_order = self.sort_order[:request_count]

        torch.max(input_scores, dim=1, out=(combined, combined_argmax))
        final_indices_i32 = self.final_indices_i32[:request_count]
        _deterministic_topk_indices_into(
            combined,
            valid_widths,
            final_indices_i32,
            self.keep_count,
        )
        final_indices.copy_(final_indices_i32)
        torch.sort(final_indices, dim=1, out=(sorted_indices, sort_order))
        torch.add(
            sorted_indices,
            self.prompt_len,
            out=self.keep[:request_count, self.prompt_len :],
        )

    def select_requests(
        self,
        segments_by_request: List[List[torch.Tensor]],
        *,
        normalize_scores: bool,
    ) -> None:
        """Pack and select every request with one fixed-shape operation sequence."""
        request_count = len(segments_by_request)
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds the cross-request selection capacity")

        flat_segments = []
        for segments in segments_by_request:
            if not segments or sum(int(segment.shape[0]) for segment in segments) != self.rows:
                raise ValueError("cross-request segments do not match the workspace row count")
            if any(
                segment.ndim != 2
                or int(segment.shape[1]) != self.width
                or segment.dtype != self.dtype
                or segment.device != self.device
                for segment in segments
            ):
                raise ValueError("cross-request segment geometry no longer matches its bucket")
            flat_segments.extend(segments)

        torch.cat(
            flat_segments,
            dim=0,
            out=self.input_scores[:request_count].view(request_count * self.rows, self.width),
        )
        self._select_input_scores(request_count, normalize_scores=normalize_scores)


class _BatchedFixedPerHeadWorkspace:
    """Fixed ``[request, ...]`` selector for both per-head modes."""

    def __init__(
        self,
        *,
        eviction_mode: str,
        dense_layers: Tuple[int, ...],
        num_query_heads: int,
        num_kv_heads: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        dtype: torch.dtype,
        device: torch.device,
        selection_backend: str,
        max_requests: int,
    ) -> None:
        if eviction_mode not in ("per_head", "per_layer_perhead"):
            raise ValueError(f"unsupported per-head eviction mode: {eviction_mode}")
        if not dense_layers or min(num_query_heads, num_kv_heads, max_requests) <= 0:
            raise ValueError("per-head selection requires positive layer, head, and request counts")
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if width <= keep_count or keep_count <= 0:
            raise ValueError("per-head selection requires width > keep_count > 0")
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported per-head selection backend: {selection_backend}")

        self.eviction_mode = eviction_mode
        self.dense_layers = tuple(int(layer) for layer in dense_layers)
        self.num_layers = len(self.dense_layers)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.query_group_size = self.num_query_heads // self.num_kv_heads
        self.rows = self.num_layers * self.num_query_heads
        self.selection_rows = (
            self.num_kv_heads
            if eviction_mode == "per_head"
            else self.num_layers * self.num_kv_heads
        )
        self.width = int(width)
        self.keep_count = int(keep_count)
        self.prompt_len = int(prompt_len)
        self.total_keep = self.prompt_len + self.keep_count
        self.dtype = dtype
        self.device = _canonical_device(device)
        self.selection_backend = selection_backend
        self.max_requests = int(max_requests)

        score_shape = (self.max_requests, self.num_layers, self.num_query_heads, self.width)
        grouped_shape = (self.max_requests, self.num_layers, self.num_kv_heads, self.width)
        self.input_scores = torch.empty(score_shape, dtype=dtype, device=self.device)
        self.row_mean = torch.empty(score_shape[:-1] + (1,), dtype=dtype, device=self.device)
        self.row_std = torch.empty_like(self.row_mean)
        self.valid_widths = torch.full(
            (self.max_requests,), self.width, dtype=torch.int32, device=self.device
        )
        self.valid_scale = torch.empty(
            (self.max_requests, 1, 1, 1), dtype=dtype, device=self.device
        )
        self.token_indices = torch.arange(self.width, dtype=torch.long, device=self.device)
        self.invalid_mask = torch.empty(
            (self.max_requests, 1, 1, self.width), dtype=torch.bool, device=self.device
        )
        self.grouped_scores = torch.empty(grouped_shape, dtype=dtype, device=self.device)
        self.grouped_argmax = torch.empty(grouped_shape, dtype=torch.long, device=self.device)
        if self.eviction_mode == "per_head":
            self.selection_scores = torch.empty(
                (self.max_requests, self.num_kv_heads, self.width),
                dtype=dtype,
                device=self.device,
            )
        else:
            self.selection_scores = self.grouped_scores.view(
                self.max_requests, self.selection_rows, self.width
            )
        self.row_seq_lens = torch.full(
            (self.max_requests, self.selection_rows),
            self.width,
            dtype=torch.int32,
            device=self.device,
        )
        selection_shape = (self.max_requests, self.selection_rows, self.keep_count)
        self.top_indices_i32 = torch.empty(selection_shape, dtype=torch.int32, device=self.device)
        self.sorted_indices_i32 = torch.empty_like(self.top_indices_i32)
        self.sort_order = torch.empty(selection_shape, dtype=torch.long, device=self.device)
        self.keep = torch.empty(
            (self.max_requests, self.selection_rows, self.total_keep),
            dtype=torch.int32,
            device=self.device,
        )
        if self.prompt_len:
            prompt = torch.arange(self.prompt_len, dtype=torch.int32, device=self.device)
            self.keep[:, :, : self.prompt_len].copy_(
                prompt.view(1, 1, -1).expand(self.max_requests, self.selection_rows, -1)
            )

    def named_tensors(self) -> Tuple[Tuple[str, torch.Tensor], ...]:
        return (
            ("input_scores", self.input_scores),
            ("row_mean", self.row_mean),
            ("row_std", self.row_std),
            ("valid_widths", self.valid_widths),
            ("valid_scale", self.valid_scale),
            ("token_indices", self.token_indices),
            ("invalid_mask", self.invalid_mask),
            ("grouped_scores", self.grouped_scores),
            ("grouped_argmax", self.grouped_argmax),
            ("selection_scores", self.selection_scores),
            ("row_seq_lens", self.row_seq_lens),
            ("top_indices_i32", self.top_indices_i32),
            ("sorted_indices_i32", self.sorted_indices_i32),
            ("sort_order", self.sort_order),
            ("keep", self.keep),
        )

    def stage_valid_widths_from_seq_lens(
        self, valid_seq_lens: torch.Tensor, request_count: int
    ) -> None:
        if (
            request_count <= 0
            or request_count > self.max_requests
            or valid_seq_lens.ndim != 1
            or valid_seq_lens.numel() < request_count
            or valid_seq_lens.dtype != torch.int32
            or valid_seq_lens.device != self.device
        ):
            raise ValueError("valid sequence lengths do not fit the per-head selection bucket")
        torch.sub(
            valid_seq_lens[:request_count],
            self.prompt_len,
            out=self.valid_widths[:request_count],
        )

    def _select_input_scores(self, request_count: int, *, normalize_scores: bool) -> None:
        input_scores = self.input_scores[:request_count]
        valid_widths = self.valid_widths[:request_count]
        invalid_mask = self.invalid_mask[:request_count]
        torch.ge(
            self.token_indices.view(1, 1, 1, self.width),
            valid_widths.view(request_count, 1, 1, 1),
            out=invalid_mask,
        )
        if normalize_scores:
            row_mean = self.row_mean[:request_count]
            row_std = self.row_std[:request_count]
            input_scores.masked_fill_(invalid_mask, 0.0)
            torch.sum(input_scores, dim=3, keepdim=True, out=row_mean)
            self.valid_scale[:request_count].view(request_count).copy_(valid_widths)
            row_mean.div_(self.valid_scale[:request_count])
            torch.sub(input_scores, row_mean, out=input_scores)
            input_scores.masked_fill_(invalid_mask, 0.0)
            torch.linalg.vector_norm(input_scores, dim=3, keepdim=True, out=row_std)
            self.valid_scale[:request_count].sqrt_()
            row_std.div_(self.valid_scale[:request_count])
            row_std.clamp_min_(1e-6)
            torch.div(input_scores, row_std, out=input_scores)
        input_scores.masked_fill_(invalid_mask, float("-inf"))

        grouped_scores = self.grouped_scores[:request_count]
        grouped_argmax = self.grouped_argmax[:request_count]
        torch.max(
            input_scores.view(
                request_count,
                self.num_layers,
                self.num_kv_heads,
                self.query_group_size,
                self.width,
            ),
            dim=3,
            out=(grouped_scores, grouped_argmax),
        )
        if self.eviction_mode == "per_head":
            torch.mean(grouped_scores, dim=1, out=self.selection_scores[:request_count])
        selection_scores = self.selection_scores[:request_count]
        row_seq_lens = self.row_seq_lens[:request_count]
        row_seq_lens.copy_(valid_widths.view(request_count, 1).expand(-1, self.selection_rows))
        _deterministic_topk_indices_into(
            selection_scores.reshape(request_count * self.selection_rows, self.width),
            row_seq_lens.reshape(-1),
            self.top_indices_i32[:request_count].reshape(-1, self.keep_count),
            self.keep_count,
        )
        torch.sort(
            self.top_indices_i32[:request_count],
            dim=2,
            out=(self.sorted_indices_i32[:request_count], self.sort_order[:request_count]),
        )
        torch.add(
            self.sorted_indices_i32[:request_count],
            self.prompt_len,
            out=self.keep[:request_count, :, self.prompt_len :],
        )

    def select_requests(
        self,
        segments_by_request: List[List[torch.Tensor]],
        *,
        normalize_scores: bool,
    ) -> None:
        request_count = len(segments_by_request)
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds the per-head selection capacity")
        flat_segments = []
        for segments in segments_by_request:
            if len(segments) != self.num_layers:
                raise ValueError("per-head segments do not match the dense-layer count")
            if any(
                segment.shape != (self.num_query_heads, self.width)
                or segment.dtype != self.dtype
                or segment.device != self.device
                for segment in segments
            ):
                raise ValueError("per-head segment geometry no longer matches its bucket")
            flat_segments.extend(segments)
        torch.cat(
            flat_segments,
            dim=0,
            out=self.input_scores[:request_count].view(request_count * self.rows, self.width),
        )
        self._select_input_scores(request_count, normalize_scores=normalize_scores)


class _FixedScoreStreamMismatch(RuntimeError):
    """Raised when a fixed score workspace is used from another CUDA stream."""


class _FixedScoreMetadataWorkspace:
    """Pool-bound fixed score metadata with one nonblocking page-table upload."""

    @staticmethod
    def _page_table_slot_layout(
        page_representatives: List[int],
        global_layers: List[int],
        page_table_keys: List[object],
    ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
        if len(page_table_keys) != len(page_representatives):
            raise ValueError("page-table keys must match the representative count")
        unique_global_representatives = []
        key_to_slot = {}
        representative_slots = {}
        for representative, key in zip(page_representatives, page_table_keys):
            slot = key_to_slot.get(key)
            if slot is None:
                slot = len(key_to_slot)
                key_to_slot[key] = slot
                unique_global_representatives.append(global_layers[representative])
            representative_slots[representative] = slot
        return tuple(unique_global_representatives), representative_slots

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        dense_groups: List[List[int]],
        page_representatives: List[int],
        global_layers: List[int],
        max_requests: int,
        seq_len: int,
        num_q_heads: int,
        num_freqs: int,
        q_real: torch.Tensor,
        q_imag: torch.Tensor,
        mlr_coef: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        offsets: torch.Tensor,
        omega: torch.Tensor,
        page_table_keys: Optional[List[object]] = None,
        prompt_len: int = 0,
        page_table_token_capacity: Optional[int] = None,
        staging_capacity: Optional[int] = None,
    ) -> None:
        from .triattention_kernels import _FixedScoreGroup

        if not dense_groups or not page_representatives or max_requests <= 0:
            raise ValueError("fixed score metadata requires non-empty positive geometry")
        self.device = _canonical_device(layer_pools[page_representatives[0]].device)
        if self.device.type != "cuda":
            raise ValueError("fixed score metadata is CUDA-only")
        self.max_requests = max_requests
        if staging_capacity is None:
            staging_capacity = max_requests
        if staging_capacity < max_requests:
            raise ValueError(
                "fixed score staging capacity cannot be smaller than its request capacity"
            )
        self.staging_capacity = int(staging_capacity)
        self.bucket_seq_len = seq_len
        if page_table_token_capacity is None:
            page_table_token_capacity = seq_len
        if page_table_token_capacity < seq_len:
            raise ValueError("page-table capacity cannot be smaller than the score bucket")
        self.page_table_token_capacity = int(page_table_token_capacity)
        if prompt_len < 0 or prompt_len > seq_len:
            raise ValueError("fixed score metadata prompt length is outside its bucket")
        self.prompt_len = prompt_len
        q_real = q_real.to(device=self.device, dtype=torch.float32).contiguous()
        q_imag = q_imag.to(device=self.device, dtype=torch.float32).contiguous()
        mlr_coef = mlr_coef.to(device=self.device, dtype=torch.float32).contiguous()
        freq_scale_sq = freq_scale_sq.to(device=self.device, dtype=torch.float32).contiguous()
        offsets = offsets.to(device=self.device, dtype=torch.float32).contiguous()
        omega = omega.to(device=self.device, dtype=torch.float32).contiguous()
        if page_table_keys is None:
            page_table_keys = list(range(len(page_representatives)))
        self.global_representatives, self.representative_slots = self._page_table_slot_layout(
            page_representatives,
            global_layers,
            page_table_keys,
        )
        self.page_table_keys = tuple(page_table_keys)
        self.signature = (
            self.page_table_keys,
            self._signature(layer_pools, dense_groups, page_representatives),
        )
        tokens_per_block = int(layer_pools[page_representatives[0]].shape[3])
        self.tokens_per_block = tokens_per_block
        self.page_count = (
            self.page_table_token_capacity + tokens_per_block - 1
        ) // tokens_per_block
        if any(
            (self.page_table_token_capacity + int(layer_pools[layer].shape[3]) - 1)
            // int(layer_pools[layer].shape[3])
            != self.page_count
            for layer in page_representatives
        ):
            raise ValueError("fixed score metadata requires a uniform page count")
        host_page_shape = (
            len(self.global_representatives),
            self.staging_capacity,
            self.page_count,
        )
        device_page_shape = (len(self.global_representatives), max_requests, self.page_count)
        self.page_ids_host = torch.empty(
            host_page_shape,
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.round_starts_host = torch.empty(
            self.staging_capacity,
            dtype=torch.float32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.valid_seq_lens_host = torch.empty(
            self.staging_capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._bulk_copy_idx_src = torch.empty(
            self.staging_capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.page_ids_device = torch.empty(
            device_page_shape,
            dtype=torch.int64,
            device=self.device,
        )
        self.round_starts_device = torch.empty(
            max_requests, dtype=torch.float32, device=self.device
        )
        self.valid_seq_lens_device = torch.empty(
            max_requests, dtype=torch.int32, device=self.device
        )
        num_offsets = int(offsets.numel())
        self.phase_base = torch.empty(
            (max_requests, num_offsets, 1), dtype=torch.float32, device=self.device
        )
        self.phase = torch.empty(
            (max_requests, num_offsets, num_freqs), dtype=torch.float32, device=self.device
        )
        self.cos_phase = torch.empty_like(self.phase)
        self.sin_phase = torch.empty_like(self.phase)
        self.mean_cos = torch.empty(
            (max_requests, num_freqs), dtype=torch.float32, device=self.device
        )
        self.mean_sin = torch.empty_like(self.mean_cos)
        self.offsets = offsets
        self.omega = omega
        # ONE fused group across ALL dense layers: segments carry their own
        # layer base address and page-table slot, so distinct per-layer
        # storages/block tables no longer force one launch per storage group.
        self.dense_layer_order = [layer for layers in dense_groups for layer in layers]
        _rep_of = {layer: layers[0] for layers in dense_groups for layer in layers}
        _page_table_slots = [
            self.representative_slots[_rep_of[layer]] for layer in self.dense_layer_order
        ]
        self.fused_group = _FixedScoreGroup(
            layer_pools,
            self.dense_layer_order,
            max_requests,
            self.page_count,
            seq_len,
            num_q_heads,
            self.page_ids_device,
            _page_table_slots,
            q_real,
            q_imag,
            mlr_coef,
            freq_scale_sq,
            omega,
            offsets,
        )
        self.copy_done = torch.cuda.Event()
        self.bulk_allocation_done = torch.cuda.Event()
        self.bulk_copy_done = torch.cuda.Event()
        self.bulk_consume_done = torch.cuda.Event()
        self.copy_pending = False
        self.bulk_consume_pending = False
        self.stream = None
        self._bulk_offsets_src: Optional[torch.Tensor] = None
        self._bulk_offsets_dst: Optional[torch.Tensor] = None
        self._bulk_stage_logged = False

    def _signature(
        self,
        layer_pools: List[torch.Tensor],
        dense_groups: List[List[int]],
        representatives: List[int],
    ) -> tuple:
        layers = dict.fromkeys(
            [layer for group in dense_groups for layer in group] + list(representatives)
        )
        tensors = tuple(
            (
                layer,
                int(layer_pools[layer].untyped_storage().data_ptr()),
                int(layer_pools[layer].data_ptr()),
                tuple(layer_pools[layer].shape),
                tuple(layer_pools[layer].stride()),
                str(layer_pools[layer].device),
                str(layer_pools[layer].dtype),
            )
            for layer in layers
        )
        return (
            tuple(tuple(group) for group in dense_groups),
            tuple(representatives),
            tensors,
        )

    def matches(
        self,
        layer_pools: List[torch.Tensor],
        dense_groups: List[List[int]],
        representatives: List[int],
    ) -> bool:
        try:
            return self.signature == (
                self.page_table_keys,
                self._signature(layer_pools, dense_groups, representatives),
            )
        except (AttributeError, IndexError, KeyError):
            return False

    def stage(
        self,
        cache_source,
        request_ids: List[int],
        round_starts: List[float],
        seq_lens: Optional[List[int]] = None,
        page_table_seq_lens: Optional[List[int]] = None,
    ) -> bool:
        """Copy one eager eviction cohort into reusable device buffers."""
        request_count = len(request_ids)
        if (
            request_count == 0
            or request_count > self.max_requests
            or len(round_starts) != request_count
        ):
            return False
        stream = torch.cuda.current_stream(self.device)
        if self.stream is None:
            self.stream = stream
        elif (stream.device, stream.cuda_stream) != (
            self.stream.device,
            self.stream.cuda_stream,
        ):
            raise _FixedScoreStreamMismatch(
                "TriAttention fixed score metadata is bound to its first CUDA stream"
            )
        if seq_lens is None:
            seq_lens = [self.bucket_seq_len] * request_count
        if len(seq_lens) != request_count or any(
            seq_len <= 0 or seq_len > self.bucket_seq_len for seq_len in seq_lens
        ):
            return False
        if page_table_seq_lens is None:
            page_table_seq_lens = seq_lens
        if len(page_table_seq_lens) != request_count or any(
            page_seq_len < seq_len or page_seq_len > self.page_table_token_capacity
            for seq_len, page_seq_len in zip(seq_lens, page_table_seq_lens)
        ):
            return False
        num_blocks_per_seq = [
            (seq_len + self.tokens_per_block - 1) // self.tokens_per_block
            for seq_len in page_table_seq_lens
        ]
        if callable(cache_source):
            manager = None
            get_batch_cache_indices = cache_source
        else:
            manager = cache_source
            get_batch_cache_indices = manager.get_batch_cache_indices
        staged_bulk = False
        if manager is not None:
            staged_bulk = self._stage_page_tables_bulk(
                manager,
                request_ids,
                stream,
            )
        elif self.copy_pending and not self.copy_done.query():
            # Callable test sources do not pass through the bulk staging guard.
            self.copy_done.synchronize()
        host_slice = slice(0, request_count)
        if not staged_bulk:
            rows_by_group = []
            for global_layer in self.global_representatives:
                rows = get_batch_cache_indices(
                    request_ids,
                    global_layer,
                    num_blocks_per_seq=num_blocks_per_seq,
                )
                if len(rows) != request_count:
                    return False
                padded_rows = []
                for row, live_page_count in zip(rows, num_blocks_per_seq):
                    pages = [int(page) for page in row]
                    if (
                        len(pages) != live_page_count
                        or not pages
                        or any(page < 0 for page in pages)
                    ):
                        return False
                    padded_rows.append(pages + [pages[-1]] * (self.page_count - live_page_count))
                rows_by_group.append(padded_rows)
            self.page_ids_host[:, host_slice].copy_(
                torch.as_tensor(rows_by_group, dtype=torch.int64)
            )
        self.round_starts_host[host_slice].copy_(torch.as_tensor(round_starts, dtype=torch.float32))
        self.valid_seq_lens_host[host_slice].copy_(torch.as_tensor(seq_lens, dtype=torch.int32))
        try:
            if not staged_bulk:
                self.page_ids_device[:, :request_count].copy_(
                    self.page_ids_host[:, host_slice],
                    non_blocking=True,
                )
            self.round_starts_device[:request_count].copy_(
                self.round_starts_host[host_slice],
                non_blocking=True,
            )
            self.valid_seq_lens_device[:request_count].copy_(
                self.valid_seq_lens_host[host_slice],
                non_blocking=True,
            )
            self.fused_group.stage_lengths(self.valid_seq_lens_device, request_count)
        finally:
            # The event guards pinned-source reuse. Requiring the same stream also
            # orders the next device-buffer overwrite after every score consumer.
            self.copy_done.record(stream)
            self.copy_pending = True
        return True

    def _stage_page_tables_bulk(
        self,
        manager: KVCacheManagerV2,
        request_ids: List[int],
        current_stream: torch.cuda.Stream,
    ) -> bool:
        """Copy one request group's V2 block offsets before live compaction.

        Uses the V2 block-offset kernel with immutable pinned snapshots of the
        manager's host table and index map. The snapshots are required because
        this method enqueues asynchronous host-memory reads; TriAttention later
        resizes the same cache, which mutates the manager's table in place, and
        the next metadata preparation reuses the manager's index-map buffer.
        ``dst[pool, r, 0(K), :]`` holds ``base_page * index_scales``; our HND page
        index is that value divided by ``kv_factor``.
        """
        if not request_ids or len(request_ids) > self.staging_capacity:
            return False

        host_table = manager.host_kv_cache_block_offsets
        kv_factor = int(manager.kv_factor)
        layer_offsets = manager.layer_offsets
        pool_of = manager.layer_to_pool_mapping_dict
        num_pools, _, kv_planes, max_blocks = host_table.shape
        # The native copy reads four int32 block offsets per access.
        copy_blocks = (self.page_count + 3) // 4 * 4
        if kv_planes != 2 or copy_blocks > max_blocks:
            return False
        request_count = len(request_ids)
        staging_count = request_count
        source_shape = (num_pools, host_table.shape[1], 2, copy_blocks)
        source = self._bulk_offsets_src
        if source is None or source.shape != source_shape or source.dtype != host_table.dtype:
            source = torch.empty_like(
                host_table[..., :copy_blocks],
                device="cpu",
                pin_memory=prefer_pinned(),
                memory_format=torch.contiguous_format,
            )
            self._bulk_offsets_src = source
        bulk = self._bulk_offsets_dst
        allocated = False
        bulk_shape = (num_pools, self.max_requests, 2, copy_blocks)
        if bulk is None or bulk.shape != bulk_shape or bulk.dtype != host_table.dtype:
            bulk = torch.empty(
                bulk_shape,
                dtype=host_table.dtype,
                device=self.device,
            )
            self._bulk_offsets_dst = bulk
            allocated = True
        submitted = False
        try:
            if self.copy_pending and not self.copy_done.query():
                self.copy_done.synchronize()
            source.copy_(host_table[..., :copy_blocks])
            copy_idx = manager.index_mapper.get_copy_index(request_ids, 0, 1)
            if copy_idx.shape[0] != staging_count:
                return False
            self._bulk_copy_idx_src[:staging_count].copy_(copy_idx)
            if allocated:
                self.bulk_allocation_done.record(current_stream)
                manager._stream.wait_event(self.bulk_allocation_done)
            if self.bulk_consume_pending:
                manager._stream.wait_event(self.bulk_consume_done)
            copy_idx_source = self._bulk_copy_idx_src[:request_count]
            copy_batch_block_offsets_to_device(
                source,
                bulk,
                copy_idx_source,
                manager.index_scales,
                manager.kv_offset,
                manager._stream.cuda_stream,
            )
            submitted = True
            self.bulk_copy_done.record(manager._stream)
            current_stream.wait_event(self.bulk_copy_done)
            for slot, global_layer in enumerate(self.global_representatives):
                pool_id = pool_of[layer_offsets[global_layer]]
                self.page_ids_device[slot, :request_count].copy_(
                    bulk[
                        pool_id,
                        :request_count,
                        0,
                        : self.page_count,
                    ]
                    // kv_factor
                )
            self.bulk_consume_done.record(current_stream)
            self.bulk_consume_pending = True
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            if submitted:
                raise RuntimeError(
                    "TriAttention bulk page-table copy failed after GPU submission"
                ) from exc
            logger.warning(
                f"TriAttention bulk page-table staging failed; using the host path: {exc}"
            )
            return False
        if not self._bulk_stage_logged:
            self._bulk_stage_logged = True
            logger.info("TriAttention bulk page-table staging engaged (copy_batch_block_offsets)")
        return True

    def prepare_phase(self, request_count: int) -> None:
        """Prepare round-dependent score tensors on the active execution stream."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("phase preparation exceeds the fixed score workspace")
        torch.add(
            self.round_starts_device[:request_count].view(request_count, 1, 1),
            self.offsets.view(1, -1, 1),
            out=self.phase_base[:request_count],
        )
        phase = self.phase[:request_count]
        cos_phase = self.cos_phase[:request_count]
        sin_phase = self.sin_phase[:request_count]
        torch.mul(
            self.phase_base[:request_count],
            self.omega.view(1, 1, -1),
            out=phase,
        )
        torch.cos(phase, out=cos_phase)
        torch.sin(phase, out=sin_phase)
        torch.mean(cos_phase, dim=1, out=self.mean_cos[:request_count])
        torch.mean(sin_phase, dim=1, out=self.mean_sin[:request_count])


@dataclass(frozen=True, kw_only=True, slots=True)
class _PreparedEviction:
    """Request metadata validated before eager score, select, and compact."""

    request: "LlmRequest"
    request_id: int
    seq_len: int
    round_start: float
    expected_keep_count: int
    protected_tail: int


@dataclass(kw_only=True, slots=True)
class _RequestCompressionState:
    """Mutable compression state owned by one live request."""

    generation_steps: int = 0
    evicted_tokens: int = 0
    confirmed_kv_length: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class _PreparedGenerationBatch:
    """Target growth reserved by the most recently prepared generation batch."""

    batch: "ScheduledRequests"
    growth_by_request: Dict[int, int]


@dataclass(kw_only=True, slots=True)
class _EvictionBucketResources:
    """Reusable eager score and selection buffers for one runtime shape."""

    score_workspace: _FixedScoreMetadataWorkspace
    selection_workspace: Union[_BatchedFixedUnionWorkspace, _BatchedFixedPerHeadWorkspace]


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_end``: every ``beta`` confirmed generation tokens it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    keep set. Full-attention layers are scored; kernel-masked SWA layers preserve
    their latest window in the same compacted prefix. Every layer ends with the
    same request-wide cached length.
    """

    adjusts_generation_kv_length = True

    def _selection_backend_for(self, width: int, keep_count: int) -> str:
        """Require the Blackwell CuTE-DSL selector for every eviction mode."""
        if keep_count <= 0:
            raise RuntimeError("TriAttention top-k keep_count must be positive")
        if width < keep_count:
            raise RuntimeError(
                f"TriAttention top-k width={width} is smaller than keep_count={keep_count}"
            )
        return "cute_dsl_topk"

    def __init__(
        self,
        kv_cache_manager: KVCacheManagerV2,
        top_B: int,
        draft_kv_cache_manager: Optional[KVCacheManagerV2] = None,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        eviction_mode: str = "union",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
        count_prompt_tokens: bool = False,
        spec_config: Optional["SpeculativeConfig"] = None,
    ):
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        self.spec_config = spec_config
        self._publish_draft_kv_length_delta = (
            self.has_independent_draft_kv_cache
            and requires_paged_draft_kv_length_domain(spec_config)
        )
        self.top_B = top_B
        self.beta = beta
        if self.top_B <= 0 or self.beta <= 0:
            raise ValueError("TriAttention top_B and beta must both be positive")
        # Which token set each eviction round keeps (all reproduce the upstream
        # selection: z-normalize scores, pin the prompt tokens, no recency window):
        #   union              -- union of every KV head's top-B, re-ranked by the
        #                         per-token max score. Default; matches the official
        #                         base setting (per-head and per-layer-per-head
        #                         pruning both off).
        #   per_head           -- each KV head keeps its own set, shared across
        #                         layers (mean of per-layer max).
        #   per_layer_perhead  -- each (layer, KV head) keeps its own set, fully
        #                         independent per layer.
        self.eviction_mode = eviction_mode
        if self.eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(
                f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
                "'union', 'per_head', 'per_layer_perhead'"
            )
        self.normalize_scores = bool(normalize_scores)
        self.pin_prefill = bool(pin_prefill)
        # cpt=False (default): budget counts DECODE tokens only (pinned prompt is
        # extra). cpt=True: budget INCLUDES the pinned prompt.
        self.count_prompt_tokens = bool(count_prompt_tokens)
        if not self.pin_prefill or self.count_prompt_tokens:
            raise ValueError(
                "TriAttention physical KV reclaim requires pin_prefill=True and "
                "count_prompt_tokens=False so finalized prompt KV is preserved"
            )
        # All physical moves use the C++ V2 compaction operation.
        # No other compaction path exists.
        self.score_aggregation = score_aggregation
        # Calibration is the OFFICIAL TriAttention .pt (passed via
        # calibration_path), resolved + converted on the first request
        # (on_request_init). TRT-LLM does NOT compute calibration; model_path is
        # used for RoPE tables and local layer_types/sliding_window metadata.
        self.model_path = model_path
        if self.model_path is None:
            raise ValueError(
                "TriAttention requires model_path so kernel-masked "
                "sliding-attention layers can be classified safely"
            )
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        # Calibration-derived dims + stats, filled in on_request_init.
        self._H: Optional[int] = None
        self._F: Optional[int] = None
        self._freq_scale_sq: Optional[torch.Tensor] = None

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Request presence records successful initialization. The record also
        # owns the counters and physical length cleared at request finish.
        self._request_states: Dict[int, _RequestCompressionState] = {}
        # The overlap executor prepares B(n) before finalizing B(n-1). Keep the
        # exact fixed-linear generation width for that currently in-flight batch;
        # the final hook treats those slots as an opaque suffix.
        self._prepared_generation_batch: Optional[_PreparedGenerationBatch] = None
        # Eager score/selection buffers are built from the first live cohort and
        # reused for subsequent evictions with the same runtime geometry.
        self._eviction_buckets = OrderedDict()
        self._compaction_workspaces = OrderedDict()
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None
        self._runtime_kv_layout_cache: Optional[_RuntimeKVLayout] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Mark capacity-only decode and resolve calibration once.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        request_id = request.py_request_id
        if request_id not in self._request_states:
            self._validate_v2_compatibility()
            self._validate_request_capacity(request)
            num_layers = self._num_layers_from_manager()
            self._attention_layer_partition(num_layers)
            self._request_states[request_id] = _RequestCompressionState()
        self._ensure_calibrated()

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        """Require enough target page-table capacity to reach first eviction."""
        manager = self.kv_cache_manager
        speculative_overshoot = (
            0 if self.spec_config is None else int(self.spec_config.max_draft_len)
        )
        first_eviction_decode_length = (
            self.top_B // self.beta + 1
        ) * self.beta + speculative_overshoot
        decode_capacity = min(int(request.py_max_new_tokens), first_eviction_decode_length)
        confirmed_capacity = int(request.py_prompt_len) + decode_capacity
        protected_tail_capacity = self._configured_protected_tail_capacity()
        required_capacity = confirmed_capacity + protected_tail_capacity
        pool_confirmed_capacity = manager.get_num_available_tokens(
            token_num_upper_bound=confirmed_capacity,
            max_num_draft_tokens=int(manager._kv_reserve_draft_tokens) + 1,
        )
        table_capacity = manager.max_blocks_per_seq * manager.tokens_per_block
        if confirmed_capacity > pool_confirmed_capacity or required_capacity > table_capacity:
            raise ValueError(
                "TriAttention target KV capacity is too small to reach its first "
                f"eviction: request requires {required_capacity} tokens "
                f"(prompt={request.py_prompt_len}, budget={self.top_B}, "
                f"beta={self.beta}, decode before eviction or completion="
                f"{decode_capacity}, speculative overshoot="
                f"{speculative_overshoot}, protected tail="
                f"{protected_tail_capacity}), "
                f"but the V2 pool covers {pool_confirmed_capacity + protected_tail_capacity} "
                f"tokens and its page table covers {table_capacity} tokens"
            )

    def _ensure_calibrated(self) -> None:
        """Resolve calibration once for the first request."""
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        # Pre-split query stats + MLR coefficient for the Triton score kernel so
        # it doesn't recompute (E_q_norm - |E_q|) per call. Shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._triattn_q_real = _Eq.real.to(torch.float32).contiguous()
        self._triattn_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._triattn_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    def _validate_v2_compatibility(self) -> None:
        """Reject runtime modes outside the V2 physical-compaction contract."""
        manager = self.kv_cache_manager
        if not isinstance(manager, KVCacheManagerV2):
            raise ValueError("TriAttention physical eviction requires KVCacheManagerV2")
        if manager.kv_factor != 2:
            raise ValueError(
                "TriAttention requires a standard key/value KV cache; "
                "MLA/SELFKONLY caches are not supported"
            )
        if manager.mapping.enable_attention_dp:
            raise ValueError("TriAttention does not support attention DP")
        if manager.is_disagg:
            raise ValueError("TriAttention does not support disaggregated serving")
        if manager.max_beam_width != 1:
            raise ValueError("TriAttention requires beam-width-one decoding")
        if self.spec_config is not None:
            if self.spec_config.max_draft_len is None:
                raise ValueError(
                    "TriAttention speculative compatibility requires a resolved max_draft_len"
                )
            if not self.spec_config.is_linear_tree:
                raise ValueError("TriAttention speculative compatibility requires linear drafting")
            if self.spec_config.draft_len_schedule is not None:
                raise ValueError(
                    "TriAttention does not yet support dynamic speculative draft lengths"
                )
            mode = self.spec_config.spec_dec_mode
            if not (mode.is_dflash() or mode.is_mtp_one_model() or mode.is_eagle3_one_model()):
                raise ValueError(
                    "TriAttention has not validated this speculative mode's target-tail "
                    "compaction lifecycle"
                )
            if (
                self.spec_config.acceptance_window is not None
                or self.spec_config.acceptance_length_threshold is not None
            ):
                raise ValueError(
                    "TriAttention does not support runtime speculative acceptance gating"
                )
            if not self.has_independent_draft_kv_cache:
                raise ValueError(
                    "TriAttention speculative compatibility requires a separate draft "
                    "KV cache; shared target/draft pools cannot be compacted safely"
                )
            draft_manager = self.draft_kv_cache_manager
            if draft_manager is None or not draft_manager.is_draft:
                raise ValueError(
                    "TriAttention speculative compatibility requires the actual "
                    "separate draft KV cache manager"
                )
            if draft_manager.max_seq_len < manager.max_seq_len:
                raise ValueError(
                    "TriAttention compacts only the target KV cache, so the dense "
                    "draft KV cache must cover the target logical max sequence "
                    f"length ({draft_manager.max_seq_len} < {manager.max_seq_len})"
                )
        if any(window is not None for window in manager.max_attention_window_vec) or any(
            not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
            for layer in manager.kv_cache_manager_py_config.layers
        ):
            raise ValueError(
                "TriAttention requires full-attention V2 lifecycles; native SWA, "
                "VSWA, and SSM pools are not supported"
            )

    # The framework drives all request-lifecycle hooks. TriAttention resolves
    # calibration on request init, evicts periodically at generation-step end,
    # and removes per-request state at finish. It scores from offline
    # calibration, not from live queries or attention scores, so it needs no
    # per-layer attention hook: the whole eviction runs once per period in
    # on_generation_step_end, which loops the layers and reads each layer's keys
    # straight from the KV pool.

    def on_generation_step_end(
        self, scheduled_batch: "ScheduledRequests", attn_metadata=None, **kwargs
    ) -> None:
        """Compact after native KV-cache updates have finalized this iteration.

        The compression manager is ordered after KVCacheManagerV2, so capacity
        already reflects the written token and any rewind. The overlap scheduler
        may already have enqueued the next forward; CUDA stream ordering keeps
        compaction after that reader. The resize happens only after compaction;
        it detaches the compacted tail without blocking the host, while V2's
        per-slot finish events prevent early page reuse.
        """
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._periodic_evict(scheduled_batch)

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Snapshot fixed-linear target growth; mutation remains in final update."""
        super().prepare_resources(scheduled_batch)
        generation_growth = {}
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            growth = 1 + max(
                get_draft_token_length(request),
                self.kv_cache_manager._kv_reserve_draft_tokens,
            )
            generation_growth[request_id] = growth
        self._prepared_generation_batch = _PreparedGenerationBatch(
            batch=scheduled_batch,
            growth_by_request=generation_growth,
        )

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        """Return exact newer target allocation width under overlap scheduling."""
        prepared = self._prepared_generation_batch
        if prepared is None or scheduled_batch is prepared.batch:
            return 0
        return prepared.growth_by_request.get(request_id, 0)

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Count confirmed tokens; every ``beta`` tokens score the cache
        and physically evict to the pinned prompt plus top-B decode tokens."""
        gen_requests = scheduled_batch.generation_requests
        if not gen_requests:
            return
        active_requests = []
        for request in gen_requests:
            if request.is_dummy or request.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT,
            ):
                continue
            kv_cache = self.kv_cache_manager.kv_cache_map.get(request.py_request_id)
            if kv_cache is None:
                continue
            if not kv_cache.is_active:
                raise RuntimeError(
                    "TriAttention cannot finalize a suspended target KV cache; "
                    f"request {request.py_request_id} must be resumed before "
                    "the final update hook"
                )
            if request.py_request_id not in self._request_states:
                self.on_request_init(request)
            active_requests.append(request)
        if not active_requests or not self._calibrated:
            return
        mgr = self.kv_cache_manager
        num_layers = self._num_layers_from_manager()
        protected_tails: Dict[int, int] = {}

        # (1) bump per-request step counters; collect who evicts THIS step.
        evict_now = []
        for request in active_requests:
            rid = request.py_request_id
            kv_cache = mgr.kv_cache_map.get(rid)
            if kv_cache is None or not kv_cache.is_active:
                continue
            raw_capacity = int(kv_cache.capacity)
            # One-engine speculative decoding keeps a fixed reserve E. Under
            # overlap, B(n) is allocated/enqueued before finalizing B(n-1), so
            # its exact scheduler growth Q is also opaque. Both spans are
            # contiguous after the stable target prefix and move byte-for-byte.
            protected_tail = int(mgr.num_extra_kv_tokens) + self._inflight_generation_growth(
                scheduled_batch, rid
            )
            seq_len = raw_capacity - protected_tail
            if seq_len < 0 or protected_tail < 0:
                raise RuntimeError(
                    f"Request {rid} has an inconsistent protected target tail: "
                    f"confirmed={seq_len}, capacity={raw_capacity}, "
                    f"protected_tail={protected_tail}"
                )
            if seq_len < kv_cache.history_length:
                raise RuntimeError(
                    f"Request {rid} KV length {seq_len} is below finalized "
                    f"history {kv_cache.history_length}"
                )
            request_state = self._request_states[rid]
            request_state.confirmed_kv_length = seq_len
            protected_tails[rid] = protected_tail
            previous_step = request_state.generation_steps
            confirmed_delta = 1 + int(request.py_num_accepted_draft_tokens)
            step = previous_step + confirmed_delta
            request_state.generation_steps = step
            if previous_step // self.beta < step // self.beta:
                if seq_len > self._minimum_evictable_length(request, seq_len):
                    evict_now.append((request, rid))

        # (2) Compact all affected dense and kernel-masked SWA layers, then release
        # the unreachable tail directly through V2's public resize primitive.
        if not evict_now:
            return
        protected_tail_lengths = {rid: protected_tails[rid] for _, rid in evict_now}
        # Prompt and retained geometry define selection and destination layout.
        # Requests with the same geometry execute eagerly in bounded chunks.
        # Chunking limits workspace memory; it is not a CUDA Graph constraint.
        eviction_groups = {}
        for request, rid in evict_now:
            seq_len = self._request_states[rid].confirmed_kv_length
            if seq_len is None:
                raise RuntimeError(f"Missing confirmed KV length for request {rid}")
            prompt_len = min(int(request.py_prompt_len), seq_len)
            keep_count = self._minimum_evictable_length(request, seq_len)
            selection_backend = self._selection_backend_for(
                seq_len - prompt_len,
                self.top_B,
            )
            key = (prompt_len, keep_count, selection_backend)
            eviction_groups.setdefault(key, []).append((request, rid))

        for group in eviction_groups.values():
            for begin in range(0, len(group), _EAGER_REQUEST_CHUNK_SIZE):
                chunk = group[begin : begin + _EAGER_REQUEST_CHUNK_SIZE]
                chunk_tails = {rid: protected_tail_lengths[rid] for _, rid in chunk}
                with nvtx_range_debug("triattention.evict_request_group", color="purple"):
                    capacity_targets = self._evict_requests(
                        chunk,
                        num_layers,
                        protected_tail_lengths=chunk_tails,
                    )
                self._resize_compacted_requests(capacity_targets, protected_tails)

    def _resize_compacted_requests(self, capacity_targets, protected_tails) -> None:
        if not capacity_targets:
            return
        mgr = self.kv_cache_manager
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.compaction_release_order", color="yellow"):
                # V2 records a finish event when resize detaches tail pages.
                # Reallocated pages wait on that event in their consumer stream.
                compaction_event = torch.cuda.Event()
                compaction_event.record()
                mgr._stream.wait_event(compaction_event)
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                for rid, target_capacity in capacity_targets:
                    kv_cache = mgr.kv_cache_map.get(rid)
                    if kv_cache is None or not kv_cache.is_active:
                        continue
                    if target_capacity > kv_cache.capacity:
                        raise RuntimeError(
                            f"Request {rid} compacted capacity {target_capacity} exceeds "
                            f"current capacity {kv_cache.capacity}"
                        )
                    protected_tail = protected_tails[rid]
                    resized_capacity = target_capacity + protected_tail
                    if not kv_cache.resize(resized_capacity, None):
                        raise RuntimeError(
                            f"Failed to resize compacted KV cache for request {rid} "
                            f"to {resized_capacity} tokens"
                        )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity.

        With a decode-only budget, pinned prompt tokens do not consume ``top_B``.
        Selection therefore keeps every token until the cache exceeds
        ``prompt_len + top_B``.
        """
        if self.pin_prefill and not self.count_prompt_tokens:
            prompt_len = min(int(request.py_prompt_len), seq_len)
            return prompt_len + self.top_B
        return self.top_B

    def _local_score_calibration(
        self,
        num_layers: int,
        global_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return calibration tensors indexed in this PP rank's local layer order."""
        if global_layers and max(global_layers) >= self._triattn_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._triattn_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        if global_layers == list(range(global_layers[0], global_layers[0] + num_layers)):
            layer_slice = slice(global_layers[0], global_layers[0] + num_layers)
            return (
                self._triattn_q_real[layer_slice],
                self._triattn_q_imag[layer_slice],
                self._triattn_mlr_coef[layer_slice],
            )
        layer_ids = torch.as_tensor(
            global_layers,
            device=self._triattn_q_real.device,
            dtype=torch.long,
        )
        return (
            self._triattn_q_real.index_select(0, layer_ids),
            self._triattn_q_imag.index_select(0, layer_ids),
            self._triattn_mlr_coef.index_select(0, layer_ids),
        )

    def _configured_protected_tail_capacity(self) -> int:
        """Return the largest target tail reserved by the native V2 lifecycle."""
        capacity = (
            int(self.kv_cache_manager.num_extra_kv_tokens)
            + int(self.kv_cache_manager._kv_reserve_draft_tokens)
            + 1
        )
        if capacity <= 0:
            raise RuntimeError("KVCacheManagerV2 exposes an invalid protected-tail capacity")
        return capacity

    @staticmethod
    def _build_cross_request_selection_workspace(
        plan: _CrossRequestSelectionPlan,
    ) -> Union[_BatchedFixedUnionWorkspace, _BatchedFixedPerHeadWorkspace]:
        """Allocate one fixed ``[request, ...]`` selection workspace."""
        if plan.eviction_mode == "union":
            return _BatchedFixedUnionWorkspace(
                plan.rows,
                plan.width,
                plan.keep_count,
                plan.prompt_len,
                dtype=plan.dtype,
                device=plan.device,
                selection_backend=plan.selection_backend,
                max_requests=plan.max_requests,
                dense_layers=plan.dense_layers,
                num_query_heads=plan.num_query_heads,
                num_kv_heads=plan.num_kv_heads,
            )
        return _BatchedFixedPerHeadWorkspace(
            eviction_mode=plan.eviction_mode,
            dense_layers=plan.dense_layers,
            num_query_heads=plan.num_query_heads,
            num_kv_heads=plan.num_kv_heads,
            width=plan.width,
            keep_count=plan.keep_count,
            prompt_len=plan.prompt_len,
            dtype=plan.dtype,
            device=plan.device,
            selection_backend=plan.selection_backend,
            max_requests=plan.max_requests,
        )

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request length and eviction state."""
        request_id = request.py_request_id
        self._request_states.pop(request_id, None)
        prepared = self._prepared_generation_batch
        if prepared is not None:
            prepared.growth_by_request.pop(request_id, None)
        if not self._request_states:
            self._eviction_buckets.clear()
            self._compaction_workspaces.clear()

    # ------------------------------------------------------------------ #
    # Attention-metadata reconcile (compression-framework hook)          #
    # ------------------------------------------------------------------ #

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id``."""
        state = self._request_states.get(request_id)
        return 0 if state is None else state.evicted_tokens

    def adjust_attention_metadata(self, attn_metadata) -> None:
        """Reconcile the attention metadata for this iteration's eviction.

        The framework calls this immediately before ``attn_metadata.prepare()``.
        Preserve the model engine's native first-draft versus previous-tensor
        semantics, subtracting only KV tokens that TriAttention physically
        removed. Physical capacity cannot reconstruct this value: the first
        generation step has one allocated query slot that is not cached yet,
        while later overlap steps include the previous speculative span.
        """
        kvp = attn_metadata.kv_cache_params
        if kvp is None or kvp.num_cached_tokens_per_seq is None:
            return
        num_contexts = attn_metadata.num_contexts
        num_requests = num_contexts + attn_metadata.num_generations
        req_ids = attn_metadata.request_ids
        prompt_lens = attn_metadata.prompt_lens
        pl = list(prompt_lens) if prompt_lens is not None else None
        pl_changed = False
        for i in range(num_contexts, num_requests):
            evicted = self.evicted_count(req_ids[i])
            if evicted == 0:
                continue
            native_cached = int(kvp.num_cached_tokens_per_seq[i])
            if native_cached < evicted:
                raise RuntimeError(
                    f"Request {req_ids[i]} native cached length {native_cached} "
                    f"is below its cumulative eviction count {evicted}"
                )
            nc = native_cached - evicted
            kvp.num_cached_tokens_per_seq[i] = nc
            if pl is not None and int(pl[i]) > nc:
                pl[i] = nc
                pl_changed = True
        if pl_changed:
            attn_metadata.prompt_lens = pl
        if self._publish_draft_kv_length_delta:
            self.publish_draft_kv_length_delta(
                attn_metadata,
                [
                    0 if i < num_contexts else self.evicted_count(req_ids[i])
                    for i in range(num_requests)
                ],
            )

    # ================================================================== #
    # Helpers (eviction / scoring / V2 cache access / calibration)       #
    # ================================================================== #

    # --- Upstream-faithful eviction modes (per_head / per_layer_perhead / union) ---
    #
    # These reproduce github.com/WeianMao/triattention's selection: scores are NOT
    # averaged over heads (each KV head keeps its own token set), they are
    # z-normalized per head over the decode region, the prompt (prefill) tokens are
    # pinned, and there is no recency window. The kept COUNT stays uniform (= top_B)
    # so paged attention
    # and the num_cached bookkeeping are unchanged; only the kept SET differs per
    # head. Kept K keeps its original RoPE rotation (scored post-RoPE), so a head
    # holding a different token set still scores the correct relative distance
    # and no per-head position tracking is needed.

    def _local_to_global_layers(self, num_layers: int) -> List[int]:
        """Return V2's global layer id for every local TriAttention layer slot."""
        cached = self._local_to_global_layers_cache
        if cached is not None:
            if len(cached) != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {len(cached)} to {num_layers}"
                )
            return cached

        global_layers = [int(layer) for layer in self.kv_cache_manager.pp_layers]
        if len(global_layers) != num_layers:
            raise ValueError(
                f"KVCacheManagerV2 exposes {len(global_layers)} PP layers, "
                f"but TriAttention received {num_layers} local layers"
            )
        self._local_to_global_layers_cache = global_layers
        return global_layers

    @staticmethod
    def _has_sliding_window_signal(config: Dict[str, object]) -> bool:
        """Return whether config metadata hints at sliding attention."""
        use_sliding_window = config.get("use_sliding_window")
        if isinstance(use_sliding_window, bool):
            return use_sliding_window
        for field in (
            "sliding_window",
            "sliding_window_size",
            "sliding_window_pattern",
            "max_window_layers",
        ):
            value = config.get(field)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value > 0:
                    return True
            elif value:
                return True
        return False

    def _attention_layer_partition(
        self, num_layers: int
    ) -> Tuple[List[int], List[int], Optional[int]]:
        """Return dense layers, kernel-masked SWA layers, and the SWA window.

        TriAttention initialization has already rejected real V2 windowed
        lifecycles. A sliding layer found here is therefore stored at full length
        and applies its window only in the attention kernel.
        """
        cached = self._attention_layer_partition_cache
        if cached is not None:
            return cached

        model_path = self.model_path
        if model_path is None:
            raise ValueError("TriAttention requires model_path")

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise ValueError(
                f"TriAttention could not load the local model config from {model_path!r}"
            ) from exc
        config_values = config.get_text_config().to_dict()
        layer_types = config_values.get("layer_types")
        if not layer_types:
            if self._has_sliding_window_signal(config_values):
                raise ValueError(
                    "Model config exposes sliding-window metadata but no layer_types; "
                    "TriAttention cannot classify kernel-masked SWA layers safely"
                )
            result = (list(range(num_layers)), [], None)
            self._attention_layer_partition_cache = result
            return result
        global_layers = self._local_to_global_layers(num_layers)
        if global_layers and max(global_layers) >= len(layer_types):
            raise ValueError(
                f"Model config has {len(layer_types)} layer_types entries, "
                f"but this PP rank references global layer {max(global_layers)}"
            )

        swa_layers = [
            local_layer
            for local_layer, global_layer in enumerate(global_layers)
            if "sliding" in str(layer_types[global_layer]).lower()
        ]
        swa_set = set(swa_layers)
        dense_layers = [layer for layer in range(num_layers) if layer not in swa_set]
        window_size = None
        if swa_layers:
            raw_window = config_values.get("sliding_window")
            if not isinstance(raw_window, int) or raw_window <= 0:
                raise ValueError(
                    "TriAttention requires a positive integer model sliding_window "
                    "when layer_types contains sliding attention"
                )
            if self.top_B < raw_window:
                raise ValueError(
                    f"TriAttention decode budget top_B={self.top_B} must be at least "
                    f"the kernel-masked SWA window size {raw_window}"
                )
            window_size = raw_window
        result = (dense_layers, swa_layers, window_size)
        self._attention_layer_partition_cache = result
        return result

    def _runtime_kv_layout(self, num_layers: int) -> _RuntimeKVLayout:
        """Return stable V2 pool views and layer groups for eager eviction.

        KVCacheManagerV2 keeps GPU virtual addresses and layer grouping stable,
        while opt-in pool rebalance can change the page dimension. Cache all
        layer views, then validate one fresh representative per physical pool
        before reuse.
        """
        cached = self._runtime_kv_layout_cache
        manager = self.kv_cache_manager
        if cached is not None:
            if cached.num_layers != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {cached.num_layers} to {num_layers}"
                )
            if cached.manager is not manager:
                raise RuntimeError("TriAttention target KV cache manager changed at runtime")
            representative_pools = [
                manager.get_buffers(
                    cached.global_layers[layer],
                    kv_layout="HND",
                )
                for layer in cached.pool_representatives
            ]
            if any(pool is None for pool in representative_pools):
                raise RuntimeError("TriAttention could not validate every cached V2 KV pool")
            current_fingerprint = self._pool_view_fingerprint(
                [pool for pool in representative_pools if pool is not None]
            )
            if current_fingerprint != cached.pool_view_fingerprint:
                raise RuntimeError(
                    "TriAttention V2 pool layout changed after workspace initialization; "
                    "KV pool rebalance is not supported"
                )
            return cached

        global_layers = self._local_to_global_layers(num_layers)
        maybe_layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(pool is None for pool in maybe_layer_pools):
            missing = [
                layer for layer, pool in zip(global_layers, maybe_layer_pools) if pool is None
            ]
            raise RuntimeError(f"Missing KV pools for attention layers {missing}")
        layer_pools = [pool for pool in maybe_layer_pools if pool is not None]

        dense_layers, swa_layers, swa_window = self._attention_layer_partition(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        storage_groups = self._dense_layer_pool_groups(dense_layers, global_layers)
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }
        all_layers = list(range(num_layers))
        layer_pool_keys = tuple(self._page_table_pool_keys(all_layers, global_layers))
        all_storage_groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(all_layers, layer_pool_keys):
            all_storage_groups.setdefault(pool_key, []).append(layer)
        pool_representatives = tuple(layers[0] for layers in all_storage_groups.values())
        layout = _RuntimeKVLayout(
            manager=manager,
            num_layers=num_layers,
            global_layers=global_layers,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            storage_groups=storage_groups,
            layer_group_representative=layer_group_representative,
            layer_pool_keys=layer_pool_keys,
            pool_representatives=pool_representatives,
            pool_view_fingerprint=self._pool_view_fingerprint(
                [layer_pools[layer] for layer in pool_representatives]
            ),
        )
        self._runtime_kv_layout_cache = layout
        return layout

    @staticmethod
    def _pool_view_fingerprint(pools: List[torch.Tensor]) -> Tuple[tuple, ...]:
        """Identify the V2 pool properties consumed by score and compact kernels."""
        return tuple(
            (
                pool.data_ptr(),
                tuple(int(value) for value in pool.shape),
                tuple(int(value) for value in pool.stride()),
                pool.dtype,
                pool.device,
            )
            for pool in pools
        )

    def _eager_resources_for(
        self,
        layout: _RuntimeKVLayout,
        prepared: Sequence[_PreparedEviction],
    ) -> _EvictionBucketResources:
        """Build or reuse eager score and selection buffers for one cohort."""
        if not prepared:
            raise ValueError("TriAttention eviction requires at least one request")
        prompt_lens = {min(int(item.request.py_prompt_len), item.seq_len) for item in prepared}
        if len(prompt_lens) != 1:
            raise ValueError("TriAttention batches require one common prompt length")
        prompt_len = next(iter(prompt_lens))
        seq_len = max(item.seq_len for item in prepared)
        page_table_token_capacity = max(item.seq_len + item.protected_tail for item in prepared)
        request_count = len(prepared)
        selection_backend = self._selection_backend_for(seq_len - prompt_len, self.top_B)
        dense_groups = list(layout.storage_groups.values())
        representatives = [group[0] for group in dense_groups]
        representatives.extend(layer for layer in layout.swa_layers if layer not in representatives)
        key = (
            "triattention.eager.v1",
            self.eviction_mode,
            request_count,
            seq_len,
            prompt_len,
            page_table_token_capacity,
            self.top_B,
            selection_backend,
            tuple(layout.dense_layers),
            layout.pool_view_fingerprint,
        )
        resources = self._eviction_buckets.get(key)
        if resources is not None:
            self._eviction_buckets.move_to_end(key)
            return resources

        first_pool = layout.layer_pools[layout.dense_layers[0]]
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, first_pool.device)
        q_real, q_imag, mlr_coef = self._local_score_calibration(
            layout.num_layers, layout.global_layers
        )
        score_workspace = _FixedScoreMetadataWorkspace(
            layout.layer_pools,
            dense_groups=dense_groups,
            page_representatives=representatives,
            global_layers=layout.global_layers,
            max_requests=request_count,
            seq_len=seq_len,
            num_q_heads=int(self._H),
            num_freqs=int(self._F),
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=self._freq_scale_sq,
            offsets=self._offsets,
            omega=self.calibration["omega"],
            page_table_keys=self._page_table_pool_keys(representatives, layout.global_layers),
            prompt_len=prompt_len,
            page_table_token_capacity=page_table_token_capacity,
            staging_capacity=request_count,
        )
        selection_workspace = self._build_cross_request_selection_workspace(
            _CrossRequestSelectionPlan(
                eviction_mode=self.eviction_mode,
                dense_layers=tuple(layout.dense_layers),
                num_query_heads=int(self._H),
                num_kv_heads=int(first_pool.shape[2]),
                rows=len(layout.dense_layers) * int(self._H),
                width=seq_len - prompt_len,
                keep_count=self.top_B,
                prompt_len=prompt_len,
                dtype=torch.float32,
                device=first_pool.device,
                selection_backend=selection_backend,
                max_requests=request_count,
            )
        )
        resources = _EvictionBucketResources(
            score_workspace=score_workspace,
            selection_workspace=selection_workspace,
        )
        self._eviction_buckets[key] = resources
        while len(self._eviction_buckets) > _EAGER_RESOURCE_CACHE_LIMIT:
            _, stale = self._eviction_buckets.popitem(last=False)
            stale_ids = (id(stale.score_workspace), id(stale.selection_workspace))
            for workspace_key in tuple(self._compaction_workspaces):
                if workspace_key[:2] == stale_ids:
                    del self._compaction_workspaces[workspace_key]
        return resources

    def _compaction_workspace_for(
        self,
        *,
        layout: _RuntimeKVLayout,
        prepared: Sequence[_PreparedEviction],
        score_workspace: _FixedScoreMetadataWorkspace,
        selection_workspace: Union[_BatchedFixedUnionWorkspace, _BatchedFixedPerHeadWorkspace],
    ):
        """Build or reuse the eager C++ compaction buffers for one cohort."""
        from .compaction import BatchedCompactionWorkspace

        protected_tail_lengths = tuple(item.protected_tail for item in prepared)
        key = (
            id(score_workspace),
            id(selection_workspace),
            protected_tail_lengths,
        )
        workspace = self._compaction_workspaces.get(key)
        if workspace is not None:
            self._compaction_workspaces.move_to_end(key)
        else:
            workspace = BatchedCompactionWorkspace(
                eviction_mode=self.eviction_mode,
                layer_pools=layout.layer_pools,
                dense_layers=layout.dense_layers,
                swa_layers=layout.swa_layers,
                layer_group_representative=layout.layer_group_representative,
                layer_pool_keys=list(layout.layer_pool_keys),
                score_workspace=score_workspace,
                selection_workspace=selection_workspace,
                request_count=len(prepared),
                seq_len=score_workspace.bucket_seq_len,
                prompt_len=selection_workspace.prompt_len,
                decode_keep_count=self.top_B,
                swa_window=layout.swa_window,
                protected_tail_lengths=list(protected_tail_lengths),
            )
            self._compaction_workspaces[key] = workspace
            while len(self._compaction_workspaces) > _EAGER_COMPACTION_CACHE_LIMIT:
                self._compaction_workspaces.popitem(last=False)
        return workspace

    def _page_table_pool_keys(
        self,
        representatives: List[int],
        global_layers: List[int],
    ) -> List[object]:
        """Return stable V2-pool keys for the representative layers."""
        manager = self.kv_cache_manager
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        try:
            return [
                ("pool", int(layer_to_pool[layer_offsets[global_layers[layer]]]))
                for layer in representatives
            ]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("KVCacheManagerV2 exposes an invalid layer-to-pool mapping") from exc

    def _dense_layer_pool_groups(
        self,
        dense_layers: List[int],
        global_layers: List[int],
    ) -> Dict[object, List[int]]:
        """Group layers that use the same V2 page table."""
        groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(
            dense_layers,
            self._page_table_pool_keys(dense_layers, global_layers),
        ):
            groups.setdefault(pool_key, []).append(layer)
        return groups

    def _attach_page_ids(
        self,
        prepared: Sequence[_PreparedEviction],
        workspace: _FixedScoreMetadataWorkspace,
    ) -> None:
        try:
            staged = workspace.stage(
                self.kv_cache_manager,
                [item.request_id for item in prepared],
                [item.round_start for item in prepared],
                [item.seq_len for item in prepared],
                [item.seq_len + item.protected_tail for item in prepared],
            )
        except _FixedScoreStreamMismatch:
            raise
        except Exception as exc:
            raise RuntimeError("TriAttention score staging failed") from exc
        if not staged:
            raise RuntimeError("TriAttention page-table staging rejected the cohort")

    def _evict_requests(
        self,
        evict_reqs,
        num_layers: int,
        protected_tail_lengths: Optional[Dict[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """Score and compact requests, returning ``(request_id, capacity)`` targets.

        Only full-attention layers participate in scoring. For kernel-masked SWA
        layers, the latest model window is rebased to the tail of the common
        compacted prefix before the request-wide capacity is reduced.
        """
        from .triattention_kernels import fixed_perhead_segment_views

        if protected_tail_lengths is None:
            protected_tail_lengths = {}
        protected_tail_capacity = self._configured_protected_tail_capacity()
        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            layout = self._runtime_kv_layout(num_layers)

        # Resolve request length and page metadata before mutating any layer.
        prepared: List[_PreparedEviction] = []
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, rid in evict_reqs:
                request_state = self._request_states.get(rid)
                seq_len = None if request_state is None else request_state.confirmed_kv_length
                if seq_len is None:
                    raise RuntimeError(f"Missing confirmed KV length for request {rid}")
                # Restore the uncompressed confirmed logical position from the
                # physical prefix and cumulative eviction count.
                round_start = seq_len + request_state.evicted_tokens
                if seq_len <= self._minimum_evictable_length(request, seq_len):
                    continue
                expected_keep_count = self._minimum_evictable_length(request, seq_len)
                protected_tail = int(protected_tail_lengths.get(rid, 0))
                if protected_tail < 0 or protected_tail > protected_tail_capacity:
                    raise RuntimeError(
                        f"Request {rid} protected tail {protected_tail} exceeds "
                        f"configured capacity {protected_tail_capacity}"
                    )
                prepared.append(
                    _PreparedEviction(
                        request=request,
                        request_id=rid,
                        seq_len=int(seq_len),
                        round_start=float(round_start),
                        expected_keep_count=expected_keep_count,
                        protected_tail=protected_tail,
                    )
                )
        if not prepared:
            return []
        with nvtx_range_debug("triattention.workspace_lookup", color="blue"):
            resources = self._eager_resources_for(layout, prepared)
            score_workspace = resources.score_workspace
            selection_workspace = resources.selection_workspace
            compaction_workspace = self._compaction_workspace_for(
                layout=layout,
                prepared=prepared,
                score_workspace=score_workspace,
                selection_workspace=selection_workspace,
            )
        with nvtx_range_debug("triattention.page_table_stage", color="orange"):
            self._attach_page_ids(prepared, score_workspace)

        request_count = len(prepared)
        seq_len = score_workspace.bucket_seq_len
        prompt_len = selection_workspace.prompt_len
        with nvtx_range("triattention.score", color="blue"):
            score_workspace.prepare_phase(request_count)
            per_head, _ = score_workspace.fused_group.launch(
                request_count,
                score_workspace.round_starts_device,
                score_workspace.mean_cos,
                score_workspace.mean_sin,
                self.score_aggregation,
            )
            layer_order = score_workspace.dense_layer_order
            views = fixed_perhead_segment_views(
                per_head,
                request_count,
                len(layer_order),
                seq_len,
            )
        with nvtx_range("triattention.select", color="yellow"):
            selection_workspace.stage_valid_widths_from_seq_lens(
                score_workspace.valid_seq_lens_device,
                request_count,
            )
            selection_workspace.select_requests(
                [
                    [
                        views[:, request_index, layer_slot, prompt_len:seq_len]
                        for layer_slot in range(len(layer_order))
                    ]
                    for request_index in range(request_count)
                ],
                normalize_scores=self.normalize_scores,
            )
        with nvtx_range("triattention.compact", color="purple"):
            compaction_workspace.launch()

        capacity_targets = []
        for item in prepared:
            keep_count = item.expected_keep_count
            evicted = item.seq_len - keep_count
            if evicted <= 0:
                raise RuntimeError("TriAttention attempted an identity compaction")
            request_state = self._request_states[item.request_id]
            request_state.evicted_tokens += evicted
            request_state.confirmed_kv_length = keep_count
            capacity_targets.append((item.request_id, keep_count))
        return capacity_targets

    def _num_layers_from_manager(self) -> int:
        return len(self.kv_cache_manager.pp_layers)

    # ------------------------------------------------------------------ #
    # Helpers: calibration loading                                       #
    # ------------------------------------------------------------------ #

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the user-supplied calibration .pt and return our runtime schema.

        TriAttention does NOT compute calibration -- the user calibrates with the
        official tool (github.com/WeianMao/triattention) and passes that file via
        ``calibration_path``; we only run inference. Both the official R-KV layout
        (``{metadata, stats{"layerLL_headHH": {q_mean_real, q_mean_imag,
        q_abs_mean}}}``) and our already-converted flat layout are accepted -- the
        official one is converted here."""
        if self.calibration_path is None:
            raise ValueError(
                "TriAttention requires `calibration_path`: a calibration .pt from "
                "the official tool (github.com/WeianMao/triattention). TRT-LLM does "
                "not compute calibration -- see examples/ for the Qwen3-8B file and "
                "the official calibration instructions."
            )
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            calib = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in raw.items()}
            self._validate_calibration(calib)
            return calib
        if isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            return self._convert_official_calibration(raw)
        got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"Unrecognized calibration at {self.calibration_path}: expected the "
            f"official {{metadata, stats}} layout or "
            f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
        )

    def _convert_official_calibration(self, raw) -> Dict[str, torch.Tensor]:
        """Convert the official per-(layer, head) stats to our flat runtime schema.

        ``E_q[l,h] = q_mean_real + i*q_mean_imag`` and ``E_q_norm[l,h] =
        q_abs_mean`` are the same statistic, just restacked into ``[L, H, F]``.
        ``omega`` / ``freq_scale_sq`` are not in the official file (its runtime
        recomputes them from the model rotary), so we derive them from the model
        config -- model-intrinsic and corpus-independent."""
        stats = raw["stats"]
        meta = raw.get("metadata", {})
        if "sampled_heads" in meta:
            heads = [(int(a), int(b)) for a, b in meta["sampled_heads"]]
        else:
            heads = [
                (int(k[len("layer") : k.index("_head")]), int(k[k.index("_head") + len("_head") :]))
                for k in stats
            ]
        num_layers = max(layer for layer, _ in heads) + 1
        num_heads = max(h for _, h in heads) + 1
        freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
        E_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
        E_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
        for layer, h in heads:
            s = stats[f"layer{layer:02d}_head{h:02d}"]
            E_q[layer, h] = torch.complex(s["q_mean_real"].float(), s["q_mean_imag"].float())
            E_q_norm[layer, h] = s["q_abs_mean"].float()
        omega, freq_scale_sq = self._rope_tables(freq_count)
        calib = {
            "E_q": E_q.to("cuda"),
            "E_q_norm": E_q_norm.to("cuda"),
            "omega": omega.to("cuda"),
            "freq_scale_sq": freq_scale_sq.to("cuda"),
        }
        self._validate_calibration(calib)
        logger.info(
            f"TriAttention: converted official calibration {self.calibration_path}"
            f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
        )
        return calib

    def _rope_tables(self, freq_count: int):
        """RoPE ``omega`` (inv_freq) + ``freq_scale_sq`` (squared position-0
        amplitude) from the model config -- model-intrinsic, corpus-independent
        (the official file does not store them). transformers' rope-init handles
        plain and scaled RoPE; plain RoPE has attention_factor 1 so freq_scale_sq
        is all ones. Falls back to the analytic inv_freq if rope-init is absent."""
        if self.model_path is None:
            raise ValueError(
                "TriAttention needs `model_path` to derive the RoPE tables "
                "(omega / freq_scale_sq) when converting the official calibration."
            )
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True).get_text_config()
        config_values = cfg.to_dict()
        head_dim = freq_count * 2
        base = float(config_values.get("rope_theta", 10000.0))
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            scaling = config_values.get("rope_scaling") or {}
            rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](cfg, device="cpu")
            omega = inv_freq.to(torch.float32)[:freq_count].clone()
            scale_sq = float(attention_factor) ** 2
        except Exception:
            idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
            omega = (1.0 / (base ** (idx / head_dim)))[:freq_count].clone()
            scale_sq = 1.0
        return omega, torch.full((freq_count,), scale_sq, dtype=torch.float32)

    def _validate_calibration(self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys."""
        missing = _REQUIRED_CALIBRATION_KEYS - set(calibration.keys())
        if missing:
            raise ValueError(
                f"TriAttention calibration is missing keys: {sorted(missing)}; "
                f"got {sorted(calibration.keys())}."
            )
