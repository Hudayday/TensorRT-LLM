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
already-enqueued speculative suffix is excluded from scoring, rebased unchanged,
and retained after the compressed prefix.

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

import os
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.kv_cache_compression.attention import requires_paged_draft_kv_length_domain
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
    KVCacheCompressionCacheOwner,
    KVCacheManager,
)
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
    from tensorrt_llm.llmapi.llm_args import SpeculativeConfig


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})


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


def _validate_swa_rebase(seq_len: int, keep_count: int, window_size: int) -> None:
    if window_size <= 0:
        raise ValueError(f"SWA window size must be positive, got {window_size}")
    if keep_count < window_size:
        raise ValueError(
            f"TriAttention compacted length {keep_count} must be at least the "
            f"SWA window size {window_size}"
        )
    if keep_count > seq_len:
        raise ValueError(
            f"TriAttention compacted length {keep_count} exceeds cache length {seq_len}"
        )


def _build_swa_rebase_keep(
    seq_len: int,
    keep_count: int,
    window_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return source slots that place the latest SWA window at the new tail."""
    _validate_swa_rebase(seq_len, keep_count, window_size)
    prefix = torch.arange(keep_count - window_size, device=device, dtype=torch.long)
    recent = torch.arange(seq_len - window_size, seq_len, device=device, dtype=torch.long)
    return torch.cat([prefix, recent])


def _build_swa_rebase_copy(
    seq_len: int,
    keep_count: int,
    window_size: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the source and destination ranges needed for an SWA rebase."""
    _validate_swa_rebase(seq_len, keep_count, window_size)
    source = torch.arange(seq_len - window_size, seq_len, device=device, dtype=torch.long)
    destination = torch.arange(
        keep_count - window_size, keep_count, device=device, dtype=torch.long
    )
    return source, destination


def _topk_indices_into(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    indices_i32: torch.Tensor,
    keep_count: int,
) -> None:
    """Write per-row top-k indices with the CuTE-DSL selector."""
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(scores, seq_lens, indices_i32, keep_count, 1)


class _FixedUnionWorkspace:
    """Persistent buffers for one request's fixed-shape union eviction.

    This workspace is intentionally internal and shape-specific. It removes the
    dynamic ``nonzero`` and temporary allocation chain from the capturable
    CuTE-DSL route. The caller owns this object for the request lifetime. Finite ties are
    repeatable for a fixed Torch runtime and preserve the selected-value
    multiset; cross-backend token identity remains the existing unspecified
    tie contract. Non-finite scores remain unsupported.
    """

    @staticmethod
    def _canonical_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def _matches_input(self, per_head_scores: torch.Tensor) -> bool:
        return (
            tuple(per_head_scores.shape) == (self.rows, self.width)
            and per_head_scores.dtype == self.dtype
            and per_head_scores.device == self.device
        )

    def __init__(
        self,
        rows: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        allocate_segment_buffers: bool = False,
        selection_backend: str = "cute_dsl_topk",
    ) -> None:
        if rows <= 0 or width <= keep_count or keep_count <= 0:
            raise ValueError("fixed union workspace requires rows > 0 and width > keep_count > 0")
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported fixed union selection backend: {selection_backend}")
        self.rows = rows
        self.width = width
        self.keep_count = keep_count
        self.prompt_len = prompt_len
        self.total_keep = prompt_len + keep_count
        self.dtype = dtype
        device = self._canonical_device(device)
        self.device = device
        self.selection_backend = selection_backend
        self.selection_only = False

        self.combined = torch.empty(width, dtype=dtype, device=device)
        self.combined_argmax = torch.empty(width, dtype=torch.long, device=device)
        self.row_top_indices = torch.empty((rows, keep_count), dtype=torch.long, device=device)
        self.union_mask = torch.empty(width, dtype=torch.bool, device=device)
        self.candidates = torch.empty(width, dtype=dtype, device=device)
        self.final_indices = torch.empty(keep_count, dtype=torch.long, device=device)
        self.sorted_indices = torch.empty_like(self.final_indices)
        self.sort_order = torch.empty_like(self.final_indices)
        self.keep = torch.empty(self.total_keep, dtype=torch.long, device=device)
        self.keep[:prompt_len].copy_(torch.arange(prompt_len, dtype=torch.long, device=device))
        self.row_seq_lens = torch.full((rows,), width, dtype=torch.int32, device=device)
        self.row_top_indices_i32 = torch.empty((rows, keep_count), dtype=torch.int32, device=device)
        self.token_indices = torch.arange(width, dtype=torch.long, device=device)
        self.width_sentinel = torch.full((), width, dtype=torch.long, device=device)
        self.union_physical_indices = torch.empty(width, dtype=torch.long, device=device)
        self.union_indices_sorted = torch.empty_like(self.union_physical_indices)
        self.union_sort_order = torch.empty_like(self.union_physical_indices)
        self.candidate_gather_indices = torch.empty_like(self.union_physical_indices)
        self.union_count = torch.empty(1, dtype=torch.int32, device=device)
        self.final_indices_i32 = torch.empty(keep_count, dtype=torch.int32, device=device)
        self.final_relative_indices = torch.empty_like(self.final_indices)
        self._compaction_buffers = {}
        self.prewarm_attempted = False
        self.prewarmed = False
        self.prewarm_failed = False
        self._segment_buffers_prepared = False
        if allocate_segment_buffers:
            self.prepare_segment_buffers()

    def prepare_segment_buffers(self) -> None:
        """Preallocate Stage3 packing buffers without leaving partial state."""
        if self._segment_buffers_prepared:
            return
        input_scores = torch.empty(
            (self.rows, self.width),
            dtype=self.dtype,
            device=self.device,
        )
        row_mean = torch.empty(
            (self.rows, 1),
            dtype=self.dtype,
            device=self.device,
        )
        row_std = torch.empty_like(row_mean)
        self.input_scores = input_scores
        self.row_mean = row_mean
        self.row_std = row_std
        self._segment_buffers_prepared = True

    def release_segment_buffers(self) -> None:
        """Drop optional Stage3 buffers after a fail-soft bank build."""
        if not self._segment_buffers_prepared:
            return
        del self.input_scores
        del self.row_mean
        del self.row_std
        self._segment_buffers_prepared = False

    def select_segments(
        self,
        segments: List[torch.Tensor],
        *,
        normalize_scores: bool,
    ) -> torch.Tensor:
        """Pack fixed segment views and select without data-dependent outputs."""
        if not self._segment_buffers_prepared:
            raise RuntimeError("fixed union segment buffers were not preallocated")
        if not segments or sum(int(segment.shape[0]) for segment in segments) != self.rows:
            raise ValueError("fixed union segments do not match the workspace row count")
        if any(
            segment.ndim != 2
            or int(segment.shape[1]) != self.width
            or segment.dtype != self.dtype
            or segment.device != self.device
            for segment in segments
        ):
            raise ValueError("fixed union segment geometry no longer matches its bucket")
        torch.cat(segments, dim=0, out=self.input_scores)
        if normalize_scores:
            torch.mean(self.input_scores, dim=1, keepdim=True, out=self.row_mean)
            torch.std(
                self.input_scores,
                dim=1,
                unbiased=False,
                keepdim=True,
                out=self.row_std,
            )
            self.row_std.clamp_min_(1e-6)
            torch.sub(self.input_scores, self.row_mean, out=self.input_scores)
            torch.div(self.input_scores, self.row_std, out=self.input_scores)
        return self.select(self.input_scores)

    def selection_buffer_nbytes(self) -> int:
        """Return bytes owned by the fixed selection path, excluding compaction."""
        if not self._segment_buffers_prepared:
            raise RuntimeError("fixed union segment buffers were not preallocated")
        tensors = self._selection_scratch_tensors() + (self.keep,)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _selection_scratch_tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.input_scores,
            self.row_mean,
            self.row_std,
            self.combined,
            self.combined_argmax,
            self.row_top_indices,
            self.union_mask,
            self.candidates,
            self.final_indices,
            self.sorted_indices,
            self.sort_order,
            self.row_seq_lens,
            self.row_top_indices_i32,
            self.token_indices,
            self.width_sentinel,
            self.union_physical_indices,
            self.union_indices_sorted,
            self.union_sort_order,
            self.candidate_gather_indices,
            self.union_count,
            self.final_indices_i32,
            self.final_relative_indices,
        )

    @staticmethod
    def planned_selection_bank_nbytes(
        rows: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        dtype: torch.dtype,
        selection_backend: str,
        max_requests: int,
    ) -> int:
        """Compute the deferred bank size without allocating a tensor."""
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported fixed union selection backend: {selection_backend}")
        dtype_bytes = torch.finfo(dtype).bits // 8
        long_bytes = 8
        int_bytes = 4
        total_keep = prompt_len + keep_count
        scratch_bytes = (
            rows * width * dtype_bytes
            + 2 * rows * dtype_bytes
            + width * dtype_bytes
            + width * long_bytes
            + rows * keep_count * long_bytes
            + width
            + width * dtype_bytes
            + 3 * keep_count * long_bytes
            + rows * int_bytes
            + rows * keep_count * int_bytes
            + width * long_bytes
            + long_bytes
            + 4 * width * long_bytes
            + int_bytes
            + keep_count * int_bytes
            + keep_count * long_bytes
        )
        return scratch_bytes + max_requests * total_keep * long_bytes

    def shared_selection_slot(self) -> "_FixedUnionWorkspace":
        """Allocate one stable keep output that aliases this bucket's scratch."""
        if not self._segment_buffers_prepared:
            raise RuntimeError("fixed union segment buffers were not preallocated")
        slot = _FixedUnionWorkspace.__new__(_FixedUnionWorkspace)
        slot.rows = self.rows
        slot.width = self.width
        slot.keep_count = self.keep_count
        slot.prompt_len = self.prompt_len
        slot.total_keep = self.total_keep
        slot.dtype = self.dtype
        slot.device = self.device
        slot.selection_backend = self.selection_backend
        slot.selection_only = self.selection_only
        slot.input_scores = self.input_scores
        slot.row_mean = self.row_mean
        slot.row_std = self.row_std
        slot.combined = self.combined
        slot.combined_argmax = self.combined_argmax
        slot.row_top_indices = self.row_top_indices
        slot.union_mask = self.union_mask
        slot.candidates = self.candidates
        slot.final_indices = self.final_indices
        slot.sorted_indices = self.sorted_indices
        slot.sort_order = self.sort_order
        slot.row_seq_lens = self.row_seq_lens
        slot.row_top_indices_i32 = self.row_top_indices_i32
        slot.token_indices = self.token_indices
        slot.width_sentinel = self.width_sentinel
        slot.union_physical_indices = self.union_physical_indices
        slot.union_indices_sorted = self.union_indices_sorted
        slot.union_sort_order = self.union_sort_order
        slot.candidate_gather_indices = self.candidate_gather_indices
        slot.union_count = self.union_count
        slot.final_indices_i32 = self.final_indices_i32
        slot.final_relative_indices = self.final_relative_indices
        slot.keep = torch.empty_like(self.keep)
        if self.prompt_len:
            slot.keep[: self.prompt_len].copy_(self.keep[: self.prompt_len])
        slot._compaction_buffers = {}
        slot.prewarm_attempted = False
        slot.prewarmed = False
        slot.prewarm_failed = False
        slot._segment_buffers_prepared = True
        return slot

    def select(self, per_head_scores: torch.Tensor) -> torch.Tensor:
        """Return a persistent sorted keep tensor without dynamic output shapes."""
        if not self._matches_input(per_head_scores):
            raise ValueError("fixed union input no longer matches its workspace bucket")

        torch.max(
            per_head_scores,
            dim=0,
            out=(self.combined, self.combined_argmax),
        )
        _topk_indices_into(
            per_head_scores,
            self.row_seq_lens,
            self.row_top_indices_i32,
            self.keep_count,
        )
        self.row_top_indices.copy_(self.row_top_indices_i32)
        self.union_mask.zero_()
        self.union_mask.scatter_(0, self.row_top_indices.reshape(-1), True)
        torch.where(
            self.union_mask,
            self.token_indices,
            self.width_sentinel,
            out=self.union_physical_indices,
        )
        torch.sort(
            self.union_physical_indices,
            out=(self.union_indices_sorted, self.union_sort_order),
        )
        torch.clamp(
            self.union_indices_sorted,
            max=self.width - 1,
            out=self.candidate_gather_indices,
        )
        torch.gather(
            self.combined,
            0,
            self.candidate_gather_indices,
            out=self.candidates,
        )
        torch.sum(
            self.union_mask,
            dim=0,
            dtype=torch.int32,
            out=self.union_count[0],
        )
        _topk_indices_into(
            self.candidates.view(1, self.width),
            self.union_count,
            self.final_indices_i32.view(1, self.keep_count),
            self.keep_count,
        )
        self.final_relative_indices.copy_(self.final_indices_i32)
        torch.gather(
            self.union_indices_sorted,
            0,
            self.final_relative_indices,
            out=self.final_indices,
        )
        torch.sort(
            self.final_indices,
            out=(self.sorted_indices, self.sort_order),
        )
        torch.add(
            self.sorted_indices,
            self.prompt_len,
            out=self.keep[self.prompt_len :],
        )
        return self.keep

    def can_compact(self, pool: torch.Tensor, page_ids: torch.Tensor, seq_len: int) -> bool:
        """Whether the fixed one-request compact launcher supports this pool."""
        return (
            not self.selection_only
            and pool.is_cuda
            and pool.device == self.device
            and pool.ndim == 5
            and page_ids.ndim == 1
            and page_ids.device == self.device
            and page_ids.dtype == torch.int64
            and pool.shape[1] == 2
            and pool.is_contiguous()
            and pool.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and self.total_keep < seq_len
        )

    @staticmethod
    def _compaction_key(pool: torch.Tensor, page_ids: torch.Tensor) -> tuple:
        _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape
        return (
            str(pool.device),
            pool.dtype,
            int(page_ids.numel()),
            int(kv_factor),
            int(num_kv_heads),
            int(tokens_per_block),
            int(head_dim),
        )

    def prepare_compaction(
        self,
        pool: torch.Tensor,
        page_ids: torch.Tensor,
        seq_len: int,
    ) -> tuple:
        """Materialize fixed compaction buffers without launching a kernel."""
        if not self.can_compact(pool, page_ids, seq_len):
            raise ValueError("pool no longer matches the fixed union compaction bucket")
        _, _, num_kv_heads, _, _ = pool.shape
        key = self._compaction_key(pool, page_ids)
        buffers = self._compaction_buffers.get(key)
        if buffers is None:
            page_table = torch.empty(
                1,
                page_ids.numel(),
                dtype=torch.int32,
                device=self.device,
            )
            indices = torch.empty(
                num_kv_heads,
                self.total_keep,
                dtype=torch.int32,
                device=self.device,
            )
            offsets = torch.tensor(
                [0, self.total_keep],
                dtype=torch.int32,
                device=self.device,
            )
            buffers = (page_table, indices, offsets)
            self._compaction_buffers[key] = buffers
        return buffers

    def compact(
        self,
        pool: torch.Tensor,
        page_ids: torch.Tensor,
        seq_len: int,
        *,
        copy_page_ids: bool = True,
    ) -> None:
        """Compact one HND layer using persistent C++ op inputs."""
        buffers = self.prepare_compaction(pool, page_ids, seq_len)
        page_table, indices, offsets = buffers
        if copy_page_ids:
            page_table[0].copy_(page_ids)
        indices.copy_(self.keep.reshape(1, -1).expand(indices.shape[0], -1))
        torch.ops.trtllm.sparse_kv_cache_compact(
            pool,
            page_table,
            indices,
            offsets,
            None,
        )


class _FixedShapeSelectionPlan(NamedTuple):
    """Tensor-free exact-bucket plan retained across model graph capture."""

    rows: int
    width: int
    keep_count: int
    prompt_len: int
    dtype: torch.dtype
    device: torch.device
    selection_backend: str
    max_requests: int
    materialized_nbytes: int


class _CrossRequestSelectionPlan(NamedTuple):
    """Tensor-free request-major selection plan retained across graph capture."""

    rows: int
    width: int
    keep_count: int
    prompt_len: int
    dtype: torch.dtype
    device: torch.device
    selection_backend: str
    max_requests: int
    materialized_nbytes: int


class _BatchedFixedUnionWorkspace:
    """Persistent request-major buffers for one upper-bound selection bucket."""

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
    ) -> None:
        if rows <= 0 or width <= keep_count or keep_count <= 0:
            raise ValueError("cross-request selection requires rows > 0 and width > keep_count > 0")
        if max_requests <= 0:
            raise ValueError("cross-request selection requires a positive request capacity")
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported cross-request selection backend: {selection_backend}")
        self.max_requests = max_requests
        self.selection_backend = selection_backend
        self.rows = rows
        self.width = width
        self.keep_count = keep_count
        self.prompt_len = prompt_len
        self.total_keep = prompt_len + keep_count
        self.dtype = dtype
        self.device = _FixedUnionWorkspace._canonical_device(device)

        shape = (max_requests, rows, width)
        self.input_scores = torch.empty(shape, dtype=dtype, device=self.device)
        self.row_mean = torch.empty((max_requests, rows, 1), dtype=dtype, device=self.device)
        self.row_std = torch.empty_like(self.row_mean)
        self.combined = torch.empty((max_requests, width), dtype=dtype, device=self.device)
        self.combined_argmax = torch.empty(
            (max_requests, width), dtype=torch.long, device=self.device
        )
        self.row_top_indices = torch.empty(
            (max_requests, rows, keep_count), dtype=torch.long, device=self.device
        )
        self.union_mask = torch.empty((max_requests, width), dtype=torch.bool, device=self.device)
        self.candidates = torch.empty_like(self.combined)
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
        self.row_seq_lens = torch.full(
            (max_requests * rows,), width, dtype=torch.int32, device=self.device
        )
        self.row_top_indices_i32 = torch.empty(
            (max_requests, rows, keep_count), dtype=torch.int32, device=self.device
        )
        self.width_sentinel = torch.full((), width, dtype=torch.long, device=self.device)
        self.union_physical_indices = torch.empty(
            (max_requests, width), dtype=torch.long, device=self.device
        )
        self.union_indices_sorted = torch.empty_like(self.union_physical_indices)
        self.union_sort_order = torch.empty_like(self.union_physical_indices)
        self.candidate_gather_indices = torch.empty_like(self.union_physical_indices)
        self.union_counts = torch.empty(max_requests, dtype=torch.int32, device=self.device)
        self.final_indices_i32 = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=self.device
        )
        self.final_relative_indices = torch.empty_like(self.final_indices)
        self.request_workspaces = tuple(
            self._request_workspace(request_index) for request_index in range(max_requests)
        )

    @staticmethod
    def planned_selection_bank_nbytes(
        rows: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        dtype: torch.dtype,
        selection_backend: str,
        max_requests: int,
    ) -> int:
        """Compute request-major workspace bytes without allocating a tensor."""
        if selection_backend != "cute_dsl_topk":
            raise ValueError(f"unsupported cross-request selection backend: {selection_backend}")
        dtype_bytes = torch.finfo(dtype).bits // 8
        long_bytes = 8
        int_bytes = 4
        total_keep = prompt_len + keep_count
        per_request = (
            rows * width * dtype_bytes
            + 2 * rows * dtype_bytes
            + width * dtype_bytes
            + width * long_bytes
            + rows * keep_count * long_bytes
            + width
            + width * dtype_bytes
            + 3 * keep_count * long_bytes
            + total_keep * long_bytes
            + int_bytes
            + dtype_bytes
            + width
        )
        total = max_requests * per_request + width * long_bytes
        total += max_requests * (
            rows * int_bytes
            + rows * keep_count * int_bytes
            + 4 * width * long_bytes
            + int_bytes
            + keep_count * int_bytes
            + keep_count * long_bytes
        )
        total += long_bytes
        return total

    def _request_workspace(self, request_index: int) -> _FixedUnionWorkspace:
        """Build a Stage3-compatible keep view over the request-major owner."""
        workspace = _FixedUnionWorkspace.__new__(_FixedUnionWorkspace)
        workspace.rows = self.rows
        workspace.width = self.width
        workspace.keep_count = self.keep_count
        workspace.prompt_len = self.prompt_len
        workspace.total_keep = self.total_keep
        workspace.dtype = self.dtype
        workspace.device = self.device
        workspace.selection_backend = self.selection_backend
        workspace.input_scores = self.input_scores[request_index]
        workspace.row_mean = self.row_mean[request_index]
        workspace.row_std = self.row_std[request_index]
        workspace.combined = self.combined[request_index]
        workspace.combined_argmax = self.combined_argmax[request_index]
        workspace.row_top_indices = self.row_top_indices[request_index]
        workspace.union_mask = self.union_mask[request_index]
        workspace.candidates = self.candidates[request_index]
        workspace.final_indices = self.final_indices[request_index]
        workspace.sorted_indices = self.sorted_indices[request_index]
        workspace.sort_order = self.sort_order[request_index]
        workspace.keep = self.keep[request_index]
        row_start = request_index * self.rows
        workspace.row_seq_lens = self.row_seq_lens[row_start : row_start + self.rows]
        workspace.row_top_indices_i32 = self.row_top_indices_i32[request_index]
        workspace.token_indices = self.token_indices
        workspace.width_sentinel = self.width_sentinel
        workspace.union_physical_indices = self.union_physical_indices[request_index]
        workspace.union_indices_sorted = self.union_indices_sorted[request_index]
        workspace.union_sort_order = self.union_sort_order[request_index]
        workspace.candidate_gather_indices = self.candidate_gather_indices[request_index]
        workspace.union_count = self.union_counts[request_index : request_index + 1]
        workspace.final_indices_i32 = self.final_indices_i32[request_index]
        workspace.final_relative_indices = self.final_relative_indices[request_index]
        workspace.selection_only = True
        workspace._compaction_buffers = {}
        workspace.prewarm_attempted = True
        workspace.prewarmed = True
        workspace.prewarm_failed = False
        workspace._segment_buffers_prepared = True
        return workspace

    def selection_buffer_nbytes(self) -> int:
        """Return unique bytes owned by the request-major selection workspace."""
        return sum(tensor.numel() * tensor.element_size() for _, tensor in self.named_tensors())

    def named_tensors(self) -> Tuple[Tuple[str, torch.Tensor], ...]:
        """Return the fixed tensor inventory owned by this selection bucket."""
        tensors = (
            ("input_scores", self.input_scores),
            ("row_mean", self.row_mean),
            ("row_std", self.row_std),
            ("combined", self.combined),
            ("combined_argmax", self.combined_argmax),
            ("row_top_indices", self.row_top_indices),
            ("union_mask", self.union_mask),
            ("candidates", self.candidates),
            ("final_indices", self.final_indices),
            ("sorted_indices", self.sorted_indices),
            ("sort_order", self.sort_order),
            ("keep", self.keep),
            ("valid_widths", self.valid_widths),
            ("valid_scale", self.valid_scale),
            ("token_indices", self.token_indices),
            ("invalid_mask", self.invalid_mask),
        )
        return tensors + (
            ("row_seq_lens", self.row_seq_lens),
            ("row_top_indices_i32", self.row_top_indices_i32),
            ("width_sentinel", self.width_sentinel),
            ("union_physical_indices", self.union_physical_indices),
            ("union_indices_sorted", self.union_indices_sorted),
            ("union_sort_order", self.union_sort_order),
            ("candidate_gather_indices", self.candidate_gather_indices),
            ("union_counts", self.union_counts),
            ("final_indices_i32", self.final_indices_i32),
            ("final_relative_indices", self.final_relative_indices),
        )

    def pointer_snapshot(self) -> Dict[str, tuple]:
        """Return stable addresses and geometry for runtime validation."""
        return {
            name: (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            for name, tensor in self.named_tensors()
        }

    def stage_valid_widths_from_seq_lens(
        self,
        valid_seq_lens: torch.Tensor,
        request_count: int,
    ) -> None:
        """Derive dynamic decode widths on device for graph capture and replay."""
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

    def _select_input_scores(
        self, request_count: int, *, normalize_scores: bool
    ) -> Tuple[_FixedUnionWorkspace, ...]:
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
        row_top_indices = self.row_top_indices[:request_count]
        union_mask = self.union_mask[:request_count]
        candidates = self.candidates[:request_count]
        final_indices = self.final_indices[:request_count]
        sorted_indices = self.sorted_indices[:request_count]
        sort_order = self.sort_order[:request_count]

        torch.max(input_scores, dim=1, out=(combined, combined_argmax))
        row_count = request_count * self.rows
        row_indices_i32 = self.row_top_indices_i32[:request_count]
        self.row_seq_lens[:row_count].view(request_count, self.rows).copy_(
            valid_widths.view(request_count, 1).expand(-1, self.rows)
        )
        _topk_indices_into(
            input_scores.view(row_count, self.width),
            self.row_seq_lens[:row_count],
            row_indices_i32.view(row_count, self.keep_count),
            self.keep_count,
        )
        row_top_indices.copy_(row_indices_i32)
        union_mask.zero_()
        union_mask.scatter_(1, row_top_indices.reshape(request_count, -1), True)
        union_physical_indices = self.union_physical_indices[:request_count]
        union_indices_sorted = self.union_indices_sorted[:request_count]
        union_sort_order = self.union_sort_order[:request_count]
        candidate_gather_indices = self.candidate_gather_indices[:request_count]
        union_counts = self.union_counts[:request_count]
        final_indices_i32 = self.final_indices_i32[:request_count]
        final_relative_indices = self.final_relative_indices[:request_count]
        torch.where(
            union_mask,
            self.token_indices,
            self.width_sentinel,
            out=union_physical_indices,
        )
        torch.sort(
            union_physical_indices,
            dim=1,
            out=(union_indices_sorted, union_sort_order),
        )
        torch.clamp(
            union_indices_sorted,
            max=self.width - 1,
            out=candidate_gather_indices,
        )
        torch.gather(combined, 1, candidate_gather_indices, out=candidates)
        torch.sum(union_mask, dim=1, dtype=torch.int32, out=union_counts)
        _topk_indices_into(
            candidates,
            union_counts,
            final_indices_i32,
            self.keep_count,
        )
        final_relative_indices.copy_(final_indices_i32)
        torch.gather(
            union_indices_sorted,
            1,
            final_relative_indices,
            out=final_indices,
        )
        torch.sort(final_indices, dim=1, out=(sorted_indices, sort_order))
        torch.add(
            sorted_indices,
            self.prompt_len,
            out=self.keep[:request_count, self.prompt_len :],
        )
        return self.request_workspaces[:request_count]

    def warm(self, *, normalize_scores: bool) -> None:
        """Warm the fixed backend with one full-width request."""
        self.input_scores[0].zero_()
        self.valid_widths[0].fill_(self.width)
        self._select_input_scores(1, normalize_scores=normalize_scores)

    def select_requests(
        self,
        segments_by_request: List[List[torch.Tensor]],
        *,
        normalize_scores: bool,
    ) -> Tuple[_FixedUnionWorkspace, ...]:
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
        return self._select_input_scores(request_count, normalize_scores=normalize_scores)


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
    ) -> None:
        from .triattention_kernels import _FixedScoreGroup

        if not dense_groups or not page_representatives or max_requests <= 0:
            raise ValueError("fixed score metadata requires non-empty positive geometry")
        self.device = _FixedUnionWorkspace._canonical_device(
            layer_pools[page_representatives[0]].device
        )
        if self.device.type != "cuda":
            raise ValueError("fixed score metadata is CUDA-only")
        self.max_requests = max_requests
        self.bucket_seq_len = seq_len
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
        self.page_count = (seq_len + tokens_per_block - 1) // tokens_per_block
        if any(
            (seq_len + int(layer_pools[layer].shape[3]) - 1) // int(layer_pools[layer].shape[3])
            != self.page_count
            for layer in page_representatives
        ):
            raise ValueError("fixed score metadata requires a uniform page count")
        page_shape = (len(self.global_representatives), max_requests, self.page_count)
        self.page_ids_host = torch.empty(
            page_shape, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.round_starts_host = torch.empty(
            max_requests, dtype=torch.float32, device="cpu", pin_memory=prefer_pinned()
        )
        self.valid_seq_lens_host = torch.empty(
            max_requests, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.page_ids_device = torch.empty(page_shape, dtype=torch.int64, device=self.device)
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
        self.copy_pending = False
        self.stream = None
        self._bulk_offsets_dst: Optional[torch.Tensor] = None
        self._bulk_stage_logged = False
        self.prewarm_key: Optional[tuple] = None

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
    ) -> bool:
        """Stage one cohort into an upper-bound bucket without waiting."""
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
        if self.copy_pending and not self.copy_done.query():
            return False
        if seq_lens is None:
            seq_lens = [self.bucket_seq_len] * request_count
        if len(seq_lens) != request_count or any(
            seq_len <= 0 or seq_len > self.bucket_seq_len for seq_len in seq_lens
        ):
            return False
        num_blocks_per_seq = [
            (seq_len + self.tokens_per_block - 1) // self.tokens_per_block for seq_len in seq_lens
        ]
        if callable(cache_source):
            manager = None
            get_batch_cache_indices = cache_source
        else:
            manager = cache_source
            get_batch_cache_indices = manager.get_batch_cache_indices
        staged_bulk = False
        if manager is not None:
            staged_bulk = self._stage_page_tables_bulk(manager, request_ids, stream)
            if staged_bulk and os.environ.get("TRIATTN_PAGE_TABLE_CHECK") == "1":
                self._assert_bulk_matches_legacy(manager, request_ids, num_blocks_per_seq)
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
            self.page_ids_host[:, :request_count].copy_(
                torch.as_tensor(rows_by_group, dtype=torch.int64)
            )
        self.round_starts_host[:request_count].copy_(
            torch.as_tensor(round_starts, dtype=torch.float32)
        )
        self.valid_seq_lens_host[:request_count].copy_(torch.as_tensor(seq_lens, dtype=torch.int32))
        try:
            if not staged_bulk:
                self.page_ids_device.copy_(self.page_ids_host, non_blocking=True)
            self.round_starts_device.copy_(self.round_starts_host, non_blocking=True)
            self.valid_seq_lens_device.copy_(self.valid_seq_lens_host, non_blocking=True)
            self.fused_group.stage_lengths(self.valid_seq_lens_device, request_count)
        finally:
            # The event guards pinned-source reuse. Requiring the same stream also
            # orders the next device-buffer overwrite after every score consumer.
            self.copy_done.record(stream)
            self.copy_pending = True
        return True

    def _stage_page_tables_bulk(
        self,
        manager,
        request_ids: List[int],
        current_stream: torch.cuda.Stream,
    ) -> bool:
        """ONE bulk block-offset copy replaces 36 x R host round-trips.

        Reuses the exact attention-backend path (copy_batch_block_offsets over
        the manager's persistent pinned host table): dst[pool, r, 0(K), :] holds
        base_page * index_scales; our HND page index is that value // kv_factor
        (the same formula _get_batch_cache_indices_by_pool_id applies on host).
        """
        try:
            host_table = manager.host_kv_cache_block_offsets
            kv_factor = int(manager.kv_factor)
            layer_offsets = manager.layer_offsets
            pool_of = manager.layer_to_pool_mapping_dict
        except AttributeError:
            return False
        num_pools, _, _, max_blocks = host_table.shape
        if max_blocks < self.page_count:
            return False
        request_count = len(request_ids)
        bulk = self._bulk_offsets_dst
        allocated = False
        if bulk is None or bulk.shape[1] < self.max_requests:
            bulk = torch.empty(
                num_pools,
                self.max_requests,
                2,
                max_blocks,
                dtype=host_table.dtype,
                device=self.device,
            )
            self._bulk_offsets_dst = bulk
            allocated = True
        try:
            if allocated:
                self.bulk_allocation_done.record(current_stream)
                manager._stream.wait_event(self.bulk_allocation_done)
            manager.copy_batch_block_offsets(bulk, request_ids, 1, 0, request_count)
            self.bulk_copy_done.record(manager._stream)
            current_stream.wait_event(self.bulk_copy_done)
            for slot, global_layer in enumerate(self.global_representatives):
                pool_id = pool_of[layer_offsets[global_layer]]
                self.page_ids_device[slot, :request_count].copy_(
                    bulk[pool_id, :request_count, 0, : self.page_count] // kv_factor
                )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                f"TriAttention bulk page-table staging failed; using the host path: {exc}"
            )
            return False
        if not self._bulk_stage_logged:
            self._bulk_stage_logged = True
            logger.info("TriAttention bulk page-table staging engaged (copy_batch_block_offsets)")
        return True

    def _assert_bulk_matches_legacy(
        self,
        manager,
        request_ids: List[int],
        num_blocks_per_seq: List[int],
    ) -> None:
        """TRIATTN_PAGE_TABLE_CHECK=1: prove the bulk formula against the host
        path on live data (used by validation runs, not production)."""
        request_count = len(request_ids)
        for slot, global_layer in enumerate(self.global_representatives):
            rows = manager.get_batch_cache_indices(
                request_ids,
                global_layer,
                num_blocks_per_seq=num_blocks_per_seq,
            )
            if len(rows) != request_count:
                raise RuntimeError("legacy page-table staging returned the wrong request count")
            for request_index, (row, live_page_count) in enumerate(zip(rows, num_blocks_per_seq)):
                pages = [int(page) for page in row]
                if len(pages) != live_page_count or not pages or any(page < 0 for page in pages):
                    raise RuntimeError("legacy page table contains BAD in its live prefix")
                expected = torch.as_tensor(pages, dtype=torch.int64, device=self.device)
                got = self.page_ids_device[slot, request_index, :live_page_count]
                if not torch.equal(got, expected):
                    raise RuntimeError(
                        f"bulk page-table staging mismatch at representative {global_layer}"
                    )

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


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_end``: every ``beta`` confirmed generation tokens it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    keep set. Full-attention layers are scored; kernel-masked SWA layers preserve
    their latest window in the same compacted prefix. Every layer ends with the
    same request-wide cached length.
    """

    # Fixed union workspaces remain the optimized/capturable route for large K.
    # Smaller or non-union shapes use the same CuTE-DSL op through an eager
    # scratch allocation; this threshold never selects another top-k backend.
    _FIXED_UNION_MIN_KEEP = 2048

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
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        top_B: int,
        draft_kv_cache_manager: Optional[KVCacheCompressionCacheOwner] = None,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        window_size: int = 128,
        eviction_mode: str = "union",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
        count_prompt_tokens: bool = False,
        skip_swa: bool = True,
        spec_config: Optional["SpeculativeConfig"] = None,
    ):
        self._inert_no_eviction = int(top_B) >= int(kv_cache_manager.max_seq_len)
        super().__init__(
            kv_cache_manager,
            draft_kv_cache_manager,
            mutates_kv_cache=not self._inert_no_eviction,
        )
        if self._inert_no_eviction:
            logger.info(
                "TriAttention is running in inert no-eviction mode because "
                f"top_B={top_B} is at least max_seq_len={kv_cache_manager.max_seq_len}"
            )
        elif not isinstance(kv_cache_manager, KVCacheManagerV2):
            raise ValueError(
                "TriAttention physical eviction requires KVCacheManagerV2; "
                "legacy and hybrid KV cache managers are supported only when "
                "top_B >= max_seq_len, where eviction is impossible"
            )
        else:
            kv_cache_manager.generation_capacity_only = True
        self.spec_config = spec_config
        self._publish_draft_kv_length_delta = (
            not self._inert_no_eviction
            and self.has_independent_draft_kv_cache
            and requires_paged_draft_kv_length_domain(spec_config)
        )
        self.top_B = top_B
        self.beta = beta
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
        self.skip_swa = bool(skip_swa)
        # All physical moves use the capture-safe C++ V2 compact op. Scoring
        # remains Triton; there is deliberately no runtime compact fallback.
        # Recency-window knob, kept for API compatibility. The upstream
        # calibration-based selection does not apply a recency window on the AIME
        # protocol and none of the implemented modes read this value, so it is
        # currently inert; the prompt is preserved via pin_prefill instead.
        self.window_size = int(window_size)
        self.score_aggregation = score_aggregation
        # Production runs the fused-Triton eviction path unconditionally:
        # all requests evicting this step are processed in ONE fused pass
        # (score x req x layer, select, per-layer compact), decoupling
        # launch count from request count for large-batch serving.

        # Calibration is the OFFICIAL TriAttention .pt (passed via
        # calibration_path), resolved + converted on the first request
        # (on_request_init). TRT-LLM does NOT compute calibration; model_path is
        # used for RoPE tables and local layer_types/sliding_window metadata.
        self.model_path = model_path
        if not self._inert_no_eviction and self.skip_swa and self.model_path is None:
            raise ValueError(
                "TriAttention skip_swa=True requires model_path so kernel-masked "
                "sliding-attention layers can be classified safely"
            )
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        # Calibration-derived dims + stats, filled in on_request_init.
        self._L: Optional[int] = None
        self._H: Optional[int] = None
        self._F: Optional[int] = None
        self._freq_scale_sq: Optional[torch.Tensor] = None
        self._attention_scale = 1.0

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Per-request confirmed-token counter; eviction fires when it crosses
        # ``beta``. Cleared on request finish.
        self._gen_steps: Dict[int, int] = {}
        # Cumulative physically-evicted token count per request, consumed by the
        # public introspection API.
        self._evicted: Dict[int, int] = {}
        # Authoritative confirmed physical prefix exposed to attention metadata.
        self._confirmed_kv_lengths: Dict[int, int] = {}
        # The overlap executor prepares B(n) before finalizing B(n-1). Keep the
        # exact fixed-linear generation width for that currently in-flight batch;
        # the final hook treats those slots as an opaque suffix.
        self._prepared_batch = None
        self._prepared_generation_growth: Dict[int, int] = {}
        # Context requests initialize through the framework hook; generation-only
        # requests initialize lazily in the final update hook.
        self._initialized_request_ids: set[int] = set()
        # Experimental production path promoted by the sealed real-shape P0.
        # It remains opt-in until same-shape serving A/B closes the e2e gate.
        self._fixed_union_enabled = os.environ.get("TRIATTN_FIXED_BUFFER_UNION", "0") == "1"
        self._fixed_union_compaction_enabled = self._fixed_union_enabled
        # Keep cold-path prewarm independently selectable so its correctness and
        # performance evidence cannot be confused with the fixed-buffer change.
        self._fixed_union_prewarm_enabled = (
            self._fixed_union_enabled and os.environ.get("TRIATTN_FIXED_PREWARM", "0") == "1"
        )
        self._fixed_score_metadata_enabled = (
            self._fixed_union_prewarm_enabled
            and os.environ.get("TRIATTN_FIXED_SCORE_METADATA", "0") == "1"
        )
        self._fixed_shape_selection_enabled = (
            self._fixed_score_metadata_enabled
            and os.environ.get("TRIATTN_FIXED_SHAPE_SELECTION", "0") == "1"
        )
        self._cross_request_selection_enabled = (
            self._fixed_shape_selection_enabled
            and self.eviction_mode == "union"
            and os.environ.get("TRIATTN_CROSS_REQUEST_SELECTION", "0") == "1"
        )
        self._standalone_cuda_graph_enabled = (
            self._cross_request_selection_enabled
            and self._fixed_union_compaction_enabled
            and os.environ.get("TRIATTN_STANDALONE_CUDA_GRAPH", "0") == "1"
        )
        self._fixed_union_workspaces = {}
        self._fixed_union_active = {}
        self._fixed_union_prewarm_states = {}
        self._fixed_union_prewarmed_workspaces = {}
        self._fixed_score_workspaces = {}
        self._fixed_score_prewarm_states = {}
        self._fixed_score_runtime_counts = {}
        self._fixed_shape_selection_workspaces = {}
        self._fixed_shape_selection_prewarm_states = {}
        self._fixed_shape_selection_plans = {}
        self._fixed_shape_selection_bank_bytes = {}
        self._fixed_shape_selection_runtime_counts = {}
        self._fixed_shape_selection_materialization_state = (
            "pending" if self._fixed_shape_selection_enabled else "disabled"
        )
        self._cross_request_selection_workspaces = {}
        self._cross_request_selection_prewarm_states = {}
        self._cross_request_selection_plans = {}
        self._cross_request_selection_bank_bytes = {}
        self._cross_request_selection_runtime_counts = {}
        self._cross_request_selection_materialization_state = (
            "pending" if self._cross_request_selection_enabled else "disabled"
        )
        # Graph arenas are created only after a real due cohort materializes
        # the Stage3 and Stage4 banks.
        self._standalone_graph_cache = None
        self._standalone_graph_arena_generation = 0
        self._standalone_graph_runtime_counts = {}
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Mark capacity-only decode and resolve calibration once.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        request_id = request.py_request_id
        if self._inert_no_eviction:
            self._initialized_request_ids.add(request_id)
            return
        if request_id not in self._initialized_request_ids:
            self._validate_v2_compatibility()
            num_layers = self._num_layers_from_manager()
            self._attention_layer_partition(num_layers)
            self._initialized_request_ids.add(request_id)
        self._ensure_calibrated()

    def _ensure_calibrated(self) -> None:
        """Resolve calibration once for startup prewarm or the first request."""
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._L = int(self.calibration["E_q"].shape[0])
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        self._attention_scale = float(self.calibration.get("attention_scale", 1.0))
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
        if any(window is not None for window in manager.max_attention_window_vec) or any(
            not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
            for layer in manager.kv_cache_manager_py_config.layers
        ):
            raise ValueError(
                "TriAttention requires full-attention V2 lifecycles; native SWA, "
                "VSWA, and SSM pools are not supported"
            )

    # The framework drives startup prewarm and all request-lifecycle hooks.
    # TriAttention overrides prewarm plus three lifecycle hooks: on_request_init
    # (resolve calibration once), on_generation_step_end (periodic eviction),
    # and on_request_finish (per-request cleanup). It scores from offline
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
        compaction after that reader, and resize happens only after compaction is
        complete so released pages cannot be reused early.
        """
        if not self._inert_no_eviction:
            self._periodic_evict(scheduled_batch)

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Snapshot fixed-linear target growth; mutation remains in final update."""
        super().prepare_resources(scheduled_batch)
        if self._inert_no_eviction:
            return
        generation_growth = {}
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            growth = 1 + get_draft_token_length(request)
            generation_growth[request_id] = growth
        self._prepared_batch = scheduled_batch
        self._prepared_generation_growth = generation_growth

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        """Return exact newer target allocation width under overlap scheduling."""
        if scheduled_batch is self._prepared_batch:
            return 0
        return self._prepared_generation_growth.get(request_id, 0)

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
            if request.py_request_id not in self._initialized_request_ids:
                self.on_request_init(request)
            active_requests.append(request)
        if not active_requests or not self._calibrated:
            return
        mgr = self.kv_cache_manager
        num_layers = self._num_layers_from_manager()
        protected_tails = {}

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
            self._confirmed_kv_lengths[rid] = seq_len
            protected_tails[rid] = (request, seq_len, protected_tail)
            previous_step = self._gen_steps.get(rid, 0)
            confirmed_delta = 1 + int(request.py_num_accepted_draft_tokens)
            step = previous_step + confirmed_delta
            self._gen_steps[rid] = step
            if previous_step // self.beta < step // self.beta:
                if seq_len > self._minimum_evictable_length(request, seq_len):
                    evict_now.append((request, rid))

        # (2) Compact all affected dense and kernel-masked SWA layers, then release
        # the unreachable tail directly through V2's public resize primitive.
        if evict_now:
            self._materialize_fixed_shape_selection_banks()
            self._materialize_cross_request_selection_banks()
            evicted_before = self._evicted.copy()
            confirmed_before = self._confirmed_kv_lengths.copy()
            try:
                if self._cross_request_selection_enabled:
                    # Length is dynamic inside an upper-bound graph bucket. Prompt
                    # and retained geometry remain exact because they define
                    # selection and destination layout. Every bucket uses CuTE-DSL.
                    graph_groups = {}
                    for request, rid in evict_now:
                        seq_len = self._confirmed_kv_lengths[rid]
                        prompt_len = min(int(request.py_prompt_len), seq_len)
                        keep_count = self._minimum_evictable_length(request, seq_len)
                        selection_backend = self._selection_backend_for(
                            seq_len - prompt_len,
                            self.top_B,
                        )
                        key = (prompt_len, keep_count, selection_backend)
                        graph_groups.setdefault(key, []).append((request, rid))
                    capacity_targets = []
                    for group in graph_groups.values():
                        capacity_targets.extend(self._evict_requests(group, num_layers))
                else:
                    capacity_targets = self._evict_requests(evict_now, num_layers)
            finally:
                # _evict_requests is also a directly testable execution primitive
                # and publishes its result. The lifecycle hook keeps that result
                # provisional until tail rebase, synchronization, and resize pass.
                self._evicted.clear()
                self._evicted.update(evicted_before)
                self._confirmed_kv_lengths.clear()
                self._confirmed_kv_lengths.update(confirmed_before)
        else:
            capacity_targets = []
        if capacity_targets:
            with nvtx_range("triattention.resize", color="red"):
                for rid, target_capacity in capacity_targets:
                    request, source_start, protected_tail = protected_tails[rid]
                    if protected_tail:
                        self._rebase_protected_tail(
                            request,
                            source_start=source_start,
                            destination_start=target_capacity,
                            token_count=protected_tail,
                        )
                compaction_event = torch.cuda.Event()
                compaction_event.record()
                compaction_event.synchronize()
                for rid, target_capacity in capacity_targets:
                    kv_cache = mgr.kv_cache_map.get(rid)
                    if kv_cache is None or not kv_cache.is_active:
                        continue
                    if target_capacity > kv_cache.capacity:
                        raise RuntimeError(
                            f"Request {rid} compacted capacity {target_capacity} exceeds "
                            f"current capacity {kv_cache.capacity}"
                        )
                    protected_tail = protected_tails[rid][2]
                    resized_capacity = target_capacity + protected_tail
                    if not kv_cache.resize(resized_capacity, None):
                        raise RuntimeError(
                            f"Failed to resize compacted KV cache for request {rid} "
                            f"to {resized_capacity} tokens"
                        )
                    source_capacity = protected_tails[rid][1]
                    evicted = source_capacity - target_capacity
                    self._evicted[rid] = self._evicted.get(rid, 0) + evicted
                    self._confirmed_kv_lengths[rid] = target_capacity

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

    @staticmethod
    def _dummy_pool_like(pool: torch.Tensor, num_pages: int, *, zero: bool) -> torch.Tensor:
        """Create an independent HND view with the live pool's exact strides."""
        if pool.ndim != 5 or num_pages <= 0:
            raise ValueError("fixed prewarm requires a five-dimensional HND pool")
        shape = (num_pages, *(int(value) for value in pool.shape[1:]))
        strides = tuple(int(value) for value in pool.stride())
        dummy = torch.empty_strided(
            shape,
            strides,
            dtype=pool.dtype,
            device=pool.device,
        )
        if dummy.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr():
            raise RuntimeError("fixed prewarm dummy unexpectedly aliases the live KV pool")
        if zero:
            dummy.zero_()
        return dummy

    def _fixed_union_prewarm_key(
        self,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        storage_groups: List[List[int]],
        num_layers: int,
        future_seq_len: int,
        prompt_len: int,
    ) -> tuple:
        """Return a provenance key containing geometry but no request identity."""
        decode_width = future_seq_len - prompt_len
        use_fixed_workspace = self.top_B > self._FIXED_UNION_MIN_KEEP
        selection_backend = (
            f"{'fixed_union' if use_fixed_workspace else 'eager_union'}."
            f"{self._selection_backend_for(decode_width, self.top_B)}"
        )
        if use_fixed_workspace:
            compaction_backend = (
                "cpp_sparse_kv_cache_compact"
                if self._fixed_union_compaction_enabled
                else "disabled"
            )
        else:
            compaction_backend = "cpp_sparse_kv_cache_compact"
        pool_geometry = []
        for lids in storage_groups:
            pool = layer_pools[lids[0]]
            tokens_per_block = int(pool.shape[3])
            num_pages = (future_seq_len + tokens_per_block - 1) // tokens_per_block
            device = _FixedUnionWorkspace._canonical_device(pool.device)
            pool_geometry.append(
                (
                    len(lids),
                    str(device),
                    str(pool.dtype),
                    tuple(int(value) for value in pool.shape[1:]),
                    tuple(int(value) for value in pool.stride()),
                    num_pages,
                )
            )
        return (
            "triattention.fixed-prewarm.v3",
            num_layers,
            tuple(dense_layers),
            int(self._H),
            int(self._F),
            int(self._offset_max_length),
            self.score_aggregation,
            bool(self.normalize_scores),
            "triton_tri_score_perhead",
            selection_backend,
            compaction_backend,
            future_seq_len,
            prompt_len,
            self.top_B,
            len(dense_layers) * int(self._H),
            decode_width,
            tuple(pool_geometry),
        )

    def _fixed_union_live_geometry(
        self,
        num_layers: int,
    ) -> Tuple[List[torch.Tensor], List[int], List[List[int]]]:
        """Read live pool metadata without exposing its storage to prewarm kernels."""
        get_buffers = self.kv_cache_manager.get_buffers
        global_layers = self._local_to_global_layers(num_layers)
        layer_pools = [get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(pool is None for pool in layer_pools):
            raise RuntimeError("TriAttention fixed prewarm could not resolve every KV pool")
        dense_layers = self._dense_layers(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        groups = {}
        for layer in dense_layers:
            pointer = layer_pools[layer].untyped_storage().data_ptr()
            groups.setdefault(pointer, []).append(layer)
        return layer_pools, dense_layers, list(groups.values())

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

    @staticmethod
    def _parse_fixed_prewarm_shapes(raw_shapes: str) -> List[Tuple[int, int]]:
        """Parse comma-separated ``prompt_len:maximum_decode_width`` buckets."""
        shapes = []
        for raw_shape in raw_shapes.split(","):
            raw_shape = raw_shape.strip()
            if not raw_shape:
                continue
            fields = raw_shape.split(":")
            if len(fields) != 2:
                raise ValueError(
                    "TRIATTN_FIXED_PREWARM_SHAPES entries must use prompt_len:decode_width"
                )
            try:
                prompt_len, decode_width = (int(field) for field in fields)
            except ValueError as exc:
                raise ValueError(
                    "TRIATTN_FIXED_PREWARM_SHAPES entries must contain integers"
                ) from exc
            if prompt_len < 0 or decode_width <= 0:
                raise ValueError(
                    "TRIATTN_FIXED_PREWARM_SHAPES requires prompt_len >= 0 and decode_width > 0"
                )
            shapes.append((prompt_len, decode_width))
        return list(dict.fromkeys(shapes))

    def _upper_prewarm_shapes_by_backend(
        self,
        shapes: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Emit one CuTE-DSL upper bucket per prompt length."""
        upper_by_prompt = {}
        for prompt_len, maximum_width in shapes:
            if maximum_width > self.top_B:
                upper_by_prompt[prompt_len] = max(
                    maximum_width,
                    upper_by_prompt.get(prompt_len, 0),
                )
        return sorted(upper_by_prompt.items())

    def prewarm(self) -> None:
        """Warm explicitly configured fixed-buffer buckets before graph capture.

        Startup has no real request from which to infer prompt length or the
        overlap-sensitive maximum eviction width. Upper buckets are therefore an
        explicit provenance input. Workloads must supply their externally
        observed prompt length and maximum decode width rather than infer one
        from beta.
        """
        if self._inert_no_eviction or not self._fixed_union_prewarm_enabled:
            return
        raw_shapes = os.environ.get("TRIATTN_FIXED_PREWARM_SHAPES", "")
        try:
            shapes = self._upper_prewarm_shapes_by_backend(
                self._parse_fixed_prewarm_shapes(raw_shapes)
            )
        except ValueError as exc:
            logger.warning(f"TriAttention fixed-buffer prewarm is disabled: {exc}")
            return
        if not shapes:
            logger.warning(
                "TriAttention fixed-buffer prewarm is enabled but "
                "TRIATTN_FIXED_PREWARM_SHAPES is empty; using the runtime path without prewarm"
            )
            return
        if self.eviction_mode != "union":
            return
        try:
            self._validate_v2_compatibility()
            self._ensure_calibrated()
            num_layers = self._num_layers_from_manager()
            layer_pools, dense_layers, storage_groups = self._fixed_union_live_geometry(num_layers)
        except Exception as exc:
            # This is an optional startup optimization. Backend/JIT exception
            # classes vary across Torch and Triton versions, so fail closed at
            # this boundary and leave runtime on the established fixed path.
            logger.warning(
                f"TriAttention prewarm failed; using the established runtime path: {exc}"
            )
            return
        for prompt_len, decode_width in shapes:
            try:
                self._prewarm_fixed_union_bucket(
                    layer_pools,
                    dense_layers,
                    storage_groups,
                    num_layers,
                    prompt_len,
                    decode_width,
                )
            except Exception as exc:
                logger.warning(
                    "TriAttention prewarm bucket "
                    f"{prompt_len}:{decode_width} failed; using the established runtime path: {exc}"
                )

    def _prewarm_fixed_union_bucket(
        self,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        storage_groups: List[List[int]],
        num_layers: int,
        prompt_len: int,
        decode_width: int,
    ) -> None:
        """Warm one upper-bound score/select/compact bucket using private dummy KV."""
        if decode_width <= self.top_B:
            logger.warning(
                "TriAttention prewarm bucket has no eviction work "
                f"({prompt_len}:{decode_width}); skipping prewarm"
            )
            return
        use_fixed_workspace = self.top_B > self._FIXED_UNION_MIN_KEEP
        seq_len = prompt_len + decode_width
        first_pool = layer_pools[dense_layers[0]]
        rows = len(dense_layers) * int(self._H)
        key = self._fixed_union_prewarm_key(
            layer_pools,
            dense_layers,
            storage_groups,
            num_layers,
            seq_len,
            prompt_len,
        )
        states = self._fixed_union_prewarm_states
        if states.get(key) in ("running", "ready", "failed"):
            return
        states[key] = "running"
        workspace = None
        try:
            with nvtx_range("triattention.prewarm", color="green"):
                if use_fixed_workspace:
                    workspace = _FixedUnionWorkspace(
                        rows,
                        decode_width,
                        self.top_B,
                        prompt_len,
                        dtype=torch.float32,
                        device=first_pool.device,
                    )
                    workspace.prewarm_attempted = True
                dummy_pools: List[Optional[torch.Tensor]] = [None] * num_layers
                group_inputs = []
                for lids in storage_groups:
                    live_pool = layer_pools[lids[0]]
                    tokens_per_block = int(live_pool.shape[3])
                    num_pages = (seq_len + tokens_per_block - 1) // tokens_per_block
                    dummy_pool = self._dummy_pool_like(live_pool, num_pages, zero=True)
                    for layer in lids:
                        # Aliasing only occurs among private dummy layer views.
                        # The exact strides and launch geometry match production.
                        dummy_pools[layer] = dummy_pool
                    page_ids = torch.arange(
                        num_pages,
                        dtype=torch.int64,
                        device=dummy_pool.device,
                    )
                    group_inputs.append((lids, dummy_pool, page_ids))

                device = first_pool.device
                if self._offsets is None:
                    self._offsets = _build_geometric_offsets(self._offset_max_length, device)
                global_layers = self._local_to_global_layers(num_layers)
                q_real, q_imag, mlr_coef = self._local_score_calibration(
                    num_layers,
                    global_layers,
                )
                from .triattention_kernels import triton_tri_score_perhead

                scores_by_layer = {}
                with nvtx_range("triattention.prewarm.score", color="blue"):
                    for lids, _, page_ids in group_inputs:
                        per_head, _, _ = triton_tri_score_perhead(
                            dummy_pools,
                            [page_ids],
                            [seq_len],
                            [float(seq_len)],
                            q_real,
                            q_imag,
                            mlr_coef,
                            self._freq_scale_sq,
                            self.calibration["omega"],
                            self._offsets,
                            self._H,
                            score_aggregation=self.score_aggregation,
                            layer_indices=lids,
                        )
                        for slot, layer in enumerate(lids):
                            scores_by_layer[layer] = per_head.narrow(
                                1,
                                slot * seq_len,
                                seq_len,
                            )
                with nvtx_range("triattention.prewarm.select", color="yellow"):
                    head_matrix = torch.cat(
                        [
                            self._zscore_decode(scores_by_layer[layer][:, prompt_len:seq_len])
                            for layer in dense_layers
                        ],
                        dim=0,
                    )
                    if workspace is not None:
                        keep = workspace.select(head_matrix)
                    else:
                        # Use alternating monotonic rows so the per-row union
                        # covers the full decode domain and both eager top-k
                        # stages warm the real width instead of a tie-collapsed
                        # dummy subset.
                        token = torch.arange(
                            decode_width,
                            dtype=head_matrix.dtype,
                            device=head_matrix.device,
                        )
                        direction = torch.where(
                            torch.arange(rows, device=head_matrix.device) % 2 == 0,
                            1.0,
                            -1.0,
                        ).to(head_matrix.dtype)
                        head_matrix.copy_(direction[:, None] * token[None, :])
                        combined = head_matrix.max(dim=0).values
                        decode_keep = self._select_union(
                            head_matrix,
                            combined,
                            self.top_B,
                        )
                        prompt = torch.arange(
                            prompt_len,
                            dtype=torch.long,
                            device=head_matrix.device,
                        )
                        keep = torch.sort(torch.cat([prompt, decode_keep + prompt_len])).values
                if workspace is None or self._fixed_union_compaction_enabled:
                    with nvtx_range("triattention.prewarm.compact", color="purple"):
                        if workspace is not None:
                            for _, dummy_pool, page_ids in group_inputs:
                                workspace.compact(dummy_pool, page_ids, seq_len)
                        else:
                            from .triattention_kernels import cpp_sparse_compact

                            for _, dummy_pool, page_ids in group_inputs:
                                cpp_sparse_compact(
                                    dummy_pool,
                                    [page_ids],
                                    [keep],
                                    [seq_len],
                                )
        except Exception:
            states[key] = "failed"
            if workspace is not None:
                workspace.prewarm_failed = True
            raise
        states[key] = "ready"
        if workspace is not None:
            workspace.prewarmed = True
            self._fixed_union_prewarmed_workspaces[key] = workspace
        if self._fixed_score_metadata_enabled:
            score_states = self._fixed_score_prewarm_states
            score_states[key] = "running"
            try:
                _, swa_layers, _ = self._attention_layer_partition(num_layers)
                representatives = [layers[0] for layers in storage_groups]
                representatives.extend(
                    layer for layer in swa_layers if layer not in representatives
                )
                global_layers = self._local_to_global_layers(num_layers)
                score_workspace = _FixedScoreMetadataWorkspace(
                    layer_pools,
                    storage_groups,
                    representatives,
                    global_layers,
                    int(self.kv_cache_manager.max_batch_size),
                    seq_len,
                    int(self._H),
                    int(self._F),
                    q_real,
                    q_imag,
                    mlr_coef,
                    self._freq_scale_sq,
                    self._offsets,
                    self.calibration["omega"],
                    page_table_keys=self._page_table_pool_keys(
                        representatives,
                        global_layers,
                    ),
                    prompt_len=prompt_len,
                )
                score_workspace.prewarm_key = key
                self._fixed_score_workspaces[key] = score_workspace
            except Exception as exc:
                score_states[key] = "failed"
                self._fixed_score_workspaces.pop(key, None)
                logger.warning(
                    "TriAttention fixed score metadata prewarm failed; "
                    f"using the Stage1 path: {exc}"
                )
            else:
                score_states[key] = "ready"
                self._prewarm_fixed_shape_selection_bucket(
                    key,
                    workspace,
                    score_workspace,
                    scores_by_layer,
                    dense_layers,
                    prompt_len,
                    seq_len,
                )

    def _prewarm_fixed_shape_selection_bucket(
        self,
        key: tuple,
        workspace: Optional[_FixedUnionWorkspace],
        score_workspace: _FixedScoreMetadataWorkspace,
        scores_by_layer: dict,
        dense_layers: List[int],
        prompt_len: int,
        seq_len: int,
    ) -> None:
        """Record a tensor-free Stage3 plan before model graph capture."""
        if not self._fixed_shape_selection_enabled:
            return
        states = self._fixed_shape_selection_prewarm_states
        if states.get(key) in ("running", "planned", "ready", "failed"):
            return

        states[key] = "running"
        max_requests = None
        bank_bytes = None
        try:
            max_requests = int(score_workspace.max_requests)
            decode_width = seq_len - prompt_len
            configured_budget = int(self.top_B)
            decode_budget = (
                configured_budget - prompt_len if self.count_prompt_tokens else configured_budget
            )
            if decode_budget <= 0 or decode_width <= decode_budget:
                raise ValueError("fixed shape selection bucket has no eviction work")
            selection_backend = self._selection_backend_for(decode_width, decode_budget)
            rows = sum(int(scores_by_layer[layer].shape[0]) for layer in dense_layers)
            first_scores = scores_by_layer[dense_layers[0]]
            bank_bytes = _FixedUnionWorkspace.planned_selection_bank_nbytes(
                rows,
                decode_width,
                decode_budget,
                prompt_len,
                first_scores.dtype,
                selection_backend,
                max_requests,
            )
            logger.info(
                "TriAttention is recording a tensor-free fixed-shape selection plan: "
                f"requests={max_requests}, backend={selection_backend}, "
                f"estimated_bytes={bank_bytes}"
            )
            plan = _FixedShapeSelectionPlan(
                rows=rows,
                width=decode_width,
                keep_count=decode_budget,
                prompt_len=prompt_len,
                dtype=first_scores.dtype,
                device=first_scores.device,
                selection_backend=selection_backend,
                max_requests=max_requests,
                materialized_nbytes=bank_bytes,
            )
        except Exception as exc:
            states[key] = "failed"
            self._fixed_shape_selection_plans.pop(key, None)
            self._fixed_shape_selection_workspaces.pop(key, None)
            self._fixed_shape_selection_bank_bytes.pop(key, None)
            logger.warning(
                "TriAttention fixed-shape selection prewarm failed; "
                "using the Stage2 fixed-score path "
                f"(requests={max_requests}, estimated_bytes={bank_bytes}): {exc}"
            )
            return

        plans = self._fixed_shape_selection_plans
        plans[key] = plan
        self._fixed_shape_selection_materialization_state = "pending"
        states[key] = "planned"
        self._prewarm_cross_request_selection_bucket(key, plan)
        logger.info(
            "TriAttention fixed-shape selection plan is ready; persistent buffers "
            f"are deferred until the first real generation prepare: requests={max_requests}, "
            f"bytes={bank_bytes}"
        )

    def _prewarm_cross_request_selection_bucket(
        self,
        key: tuple,
        stage3_plan: _FixedShapeSelectionPlan,
    ) -> None:
        """Record a tensor-free Stage4 plan for one fixed upper bucket."""
        if not self._cross_request_selection_enabled or self.eviction_mode != "union":
            return
        states = self._cross_request_selection_prewarm_states
        if states.get(key) in ("running", "planned", "ready", "failed"):
            return

        states[key] = "running"
        bank_bytes = None
        try:
            bank_bytes = _BatchedFixedUnionWorkspace.planned_selection_bank_nbytes(
                stage3_plan.rows,
                stage3_plan.width,
                stage3_plan.keep_count,
                stage3_plan.prompt_len,
                stage3_plan.dtype,
                stage3_plan.selection_backend,
                stage3_plan.max_requests,
            )
            plan = _CrossRequestSelectionPlan(
                rows=stage3_plan.rows,
                width=stage3_plan.width,
                keep_count=stage3_plan.keep_count,
                prompt_len=stage3_plan.prompt_len,
                dtype=stage3_plan.dtype,
                device=stage3_plan.device,
                selection_backend=stage3_plan.selection_backend,
                max_requests=stage3_plan.max_requests,
                materialized_nbytes=bank_bytes,
            )
        except Exception as exc:
            states[key] = "failed"
            self._cross_request_selection_plans.pop(key, None)
            logger.warning(
                "TriAttention cross-request selection planning failed; "
                f"using the Stage3 path (estimated_bytes={bank_bytes}): {exc}"
            )
            return

        self._cross_request_selection_plans[key] = plan
        self._cross_request_selection_materialization_state = "pending"
        states[key] = "planned"
        logger.info(
            "TriAttention cross-request selection plan is ready; persistent buffers "
            "are deferred until the first real generation prepare: "
            f"requests={plan.max_requests}, backend={plan.selection_backend}, "
            f"bytes={plan.materialized_nbytes}"
        )

    def _warm_fixed_shape_selection_workspace(self, workspace: _FixedUnionWorkspace) -> None:
        """Warm the materialized exact backend without allocating an output."""
        workspace.input_scores.zero_()
        if self.normalize_scores:
            torch.mean(workspace.input_scores, dim=1, keepdim=True, out=workspace.row_mean)
            torch.std(
                workspace.input_scores,
                dim=1,
                unbiased=False,
                keepdim=True,
                out=workspace.row_std,
            )
            workspace.row_std.clamp_min_(1e-6)
            torch.sub(workspace.input_scores, workspace.row_mean, out=workspace.input_scores)
            torch.div(workspace.input_scores, workspace.row_std, out=workspace.input_scores)
        workspace.select(workspace.input_scores)

    def _materialize_fixed_shape_selection_banks(self) -> None:
        """Allocate planned banks once, after model graph capture and before eviction."""
        state = self._fixed_shape_selection_materialization_state
        if state in ("disabled", "running", "done"):
            return
        self._fixed_shape_selection_materialization_state = "running"
        plans = self._fixed_shape_selection_plans
        states = self._fixed_shape_selection_prewarm_states
        for key, plan in plans.items():
            if states.get(key) != "planned":
                continue
            workspace = None
            try:
                candidate = self._fixed_union_prewarmed_workspaces.get(key)
                if (
                    candidate is not None
                    and candidate.rows == plan.rows
                    and candidate.width == plan.width
                    and candidate.keep_count == plan.keep_count
                    and candidate.prompt_len == plan.prompt_len
                    and candidate.dtype == plan.dtype
                    and candidate.device == plan.device
                    and candidate.selection_backend == plan.selection_backend
                ):
                    workspace = candidate
                else:
                    workspace = _FixedUnionWorkspace(
                        plan.rows,
                        plan.width,
                        plan.keep_count,
                        plan.prompt_len,
                        dtype=plan.dtype,
                        device=plan.device,
                        selection_backend=plan.selection_backend,
                    )
                    workspace.selection_only = True
                workspaces = self._build_fixed_shape_selection_workspaces(
                    workspace,
                    plan.max_requests,
                )
                self._warm_fixed_shape_selection_workspace(workspace)
            except Exception as exc:
                states[key] = "failed"
                self._fixed_shape_selection_workspaces.pop(key, None)
                self._fixed_shape_selection_bank_bytes.pop(key, None)
                if workspace is not None:
                    workspace.release_segment_buffers()
                logger.warning(
                    "TriAttention deferred fixed-shape selection materialization failed; "
                    f"using the Stage2 fixed-score path: {exc}"
                )
                continue

            for item in workspaces:
                item.prewarm_attempted = True
                item.prewarmed = True
            self._fixed_shape_selection_workspaces[key] = workspaces
            self._fixed_shape_selection_bank_bytes[key] = plan.materialized_nbytes
            states[key] = "ready"
        self._fixed_shape_selection_materialization_state = "done"

    @staticmethod
    def _build_cross_request_selection_workspace(
        plan: _CrossRequestSelectionPlan,
    ) -> _BatchedFixedUnionWorkspace:
        """Allocate one request-major owner from a deferred upper-bucket plan."""
        return _BatchedFixedUnionWorkspace(
            plan.rows,
            plan.width,
            plan.keep_count,
            plan.prompt_len,
            dtype=plan.dtype,
            device=plan.device,
            selection_backend=plan.selection_backend,
            max_requests=plan.max_requests,
        )

    def _materialize_cross_request_selection_banks(self) -> None:
        """Allocate Stage4 banks once after Stage3 and model graph capture."""
        state = self._cross_request_selection_materialization_state
        if state in ("disabled", "running", "done"):
            return
        self._cross_request_selection_materialization_state = "running"
        plans = self._cross_request_selection_plans
        states = self._cross_request_selection_prewarm_states
        stage3_states = self._fixed_shape_selection_prewarm_states
        for key, plan in plans.items():
            if states.get(key) != "planned":
                continue
            workspace = None
            try:
                if stage3_states.get(key) != "ready":
                    raise RuntimeError("the corresponding Stage3 bank is not ready")
                workspace = self._build_cross_request_selection_workspace(plan)
                if workspace.selection_buffer_nbytes() != plan.materialized_nbytes:
                    raise RuntimeError("materialized Stage4 workspace size differs from its plan")
                workspace.warm(normalize_scores=self.normalize_scores)
            except Exception as exc:
                states[key] = "failed"
                self._cross_request_selection_workspaces.pop(key, None)
                self._cross_request_selection_bank_bytes.pop(key, None)
                logger.warning(
                    "TriAttention deferred cross-request selection materialization failed; "
                    f"using the Stage3 path: {exc}"
                )
                continue

            self._cross_request_selection_workspaces[key] = workspace
            self._cross_request_selection_bank_bytes[key] = plan.materialized_nbytes
            states[key] = "ready"
        self._cross_request_selection_materialization_state = "done"

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request length and eviction state."""
        self._gen_steps.pop(request.py_request_id, None)
        self._evicted.pop(request.py_request_id, None)
        self._confirmed_kv_lengths.pop(request.py_request_id, None)
        self._prepared_generation_growth.pop(request.py_request_id, None)
        self._initialized_request_ids.discard(request.py_request_id)
        self._clear_fixed_union_workspaces(request.py_request_id)

    def _clear_fixed_union_workspaces(self, request_id: int) -> None:
        """Release fixed buffers only after this request's compaction is ordered."""
        for key in [key for key in self._fixed_union_workspaces if key[0] == request_id]:
            del self._fixed_union_workspaces[key]
        self._fixed_union_active.pop(request_id, None)

    # ------------------------------------------------------------------ #
    # Attention-metadata reconcile (compression-framework hook)          #
    # ------------------------------------------------------------------ #

    def evicted_count(self, request_id: int) -> int:
        """Cumulative tokens physically evicted for ``request_id``."""
        return self._evicted.get(request_id, 0)

    def adjust_attention_metadata(self, attn_metadata) -> None:
        """Reconcile the attention metadata for this iteration's eviction.

        The framework calls this immediately before ``attn_metadata.prepare()``.
        Preserve the model engine's native first-draft versus previous-tensor
        semantics, subtracting only KV tokens that TriAttention physically
        removed. Physical capacity cannot reconstruct this value: the first
        generation step has one allocated query slot that is not cached yet,
        while later overlap steps include the previous speculative span.
        """
        if self._inert_no_eviction:
            return
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
            evicted = self._evicted.get(req_ids[i], 0)
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
                    0 if i < num_contexts else self._evicted.get(req_ids[i], 0)
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

    def _zscore_decode(self, head_scores: "torch.Tensor") -> "torch.Tensor":
        """Z-normalize each head's scores over the token axis: ``(x - mean) /
        std`` with ``std`` clamped to 1e-6 (upstream). Row-wise on ``[H, seq]``;
        a no-op when ``normalize_scores`` is off."""
        if not self.normalize_scores or head_scores.numel() == 0:
            return head_scores
        mean = head_scores.mean(dim=1, keepdim=True)
        std = head_scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (head_scores - mean) / std

    def _group_heads_to_kv_max(
        self, head_scores: "torch.Tensor", num_kv_heads: int
    ) -> "torch.Tensor":
        """Reduce per-query-head scores ``[H, seq]`` to per-KV-head ``[nkv, seq]``
        by MAX over the query heads sharing each KV head (upstream within-group
        aggregation). Query heads group contiguously: head ``q`` -> KV head
        ``q // (H // nkv)`` (matches the ``q*nkv//H`` GQA map in scoring and
        upstream's ``head // num_key_value_groups``)."""
        num_q_heads, seq = head_scores.shape
        group = max(1, num_q_heads // num_kv_heads)
        return head_scores.view(num_kv_heads, group, seq).max(dim=1).values

    def _decode_topk(
        self, scores: "torch.Tensor", decode_start: int, decode_budget: int
    ) -> "torch.Tensor":
        """Pick each row's top-``decode_budget`` decode tokens by ``scores``
        (``[rows, decode_count]``); return ABSOLUTE slot indices prefixed with the
        pinned prompt slots, sorted ascending per row (``[rows, decode_start+k]``).
        One IndexerTopK launch over all rows -- the SAME select primitive as the
        union mode, so large-BS/seq decouples the cost from the row count. The
        kept SET is what matters (the result is sorted by slot)."""
        rows, decode_count = scores.shape
        k = min(decode_budget, decode_count)
        decode_idx = self._cute_dsl_topk_idx(scores, k) + decode_start  # [rows, k]
        prefill_idx = torch.arange(decode_start, device=scores.device, dtype=torch.long).expand(
            rows, decode_start
        )
        return torch.sort(torch.cat([prefill_idx, decode_idx], dim=1), dim=1).values

    def _evict_modes(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        precomputed: List["torch.Tensor"],
        use_fixed_union: bool = False,
        fixed_union_prewarm_key: Optional[tuple] = None,
        fixed_union_workspace: Optional[_FixedUnionWorkspace] = None,
    ) -> Optional[Union[int, "torch.Tensor"]]:
        """Select and compact from precomputed dense-layer scores.

        Returns either the uniform kept count or, for union mode, the sorted
        shared keep tensor whose compaction is batched by the caller. Returns
        ``None`` if no dense layer could be scored.

        ``precomputed``: per-layer ``[H, seq_len]`` scores (from the fused
        score kernel ``triton_tri_score_perhead``); used verbatim for the
        selection and compaction. Kernel-masked SWA entries are ``None`` and are
        excluded by the dense-layer partition.
        """
        budget = min(self.top_B, seq_len)
        # Prompt pin: only decode tokens [decode_start, seq_len) compete; the
        # first decode_start (prompt) tokens are always kept.
        decode_start = 0
        if self.pin_prefill:
            decode_start = min(request.py_prompt_len, seq_len)
        decode_count = seq_len - decode_start
        decode_budget = (budget - decode_start) if self.count_prompt_tokens else budget
        # Budget exhausted by the pinned prompt (or no decode tokens): keep the
        # first `budget` contiguous slots everywhere; nothing per-head to do.
        if decode_budget <= 0 or decode_count <= 0:
            return budget

        # Select from the precomputed per-layer K scores. For per_head /
        # per_layer_perhead reduce to per-KV-head [nkv, decode_count] (group-max),
        # z-normalized over the decode region. For union keep the raw per-head
        # rows to stack.
        dense_layers = self._dense_layers(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        # Number of KV heads from the first scored layer (HND dim 2).
        get_buffers = self.kv_cache_manager.get_buffers
        first_global_layer = self._global_layer_id(dense_layers[0], num_layers)
        p0 = get_buffers(first_global_layer, kv_layout="HND")
        if p0 is None:
            return None
        num_kv_heads = int(p0.shape[2])
        per_layer_kv_scores: List[torch.Tensor] = []
        union_rows: List[torch.Tensor] = []
        dense_set = set(dense_layers)
        for layer_idx in range(num_layers):
            if layer_idx not in dense_set:
                continue
            head_scores = precomputed[layer_idx]
            if head_scores is None:
                return None
            decode_scores = head_scores[:, decode_start:seq_len]
            if self.eviction_mode == "union":
                if fixed_union_workspace is None:
                    decode_scores = self._zscore_decode(decode_scores)
                union_rows.append(decode_scores)
            else:
                decode_scores = self._zscore_decode(decode_scores)
                per_layer_kv_scores.append(self._group_heads_to_kv_max(decode_scores, num_kv_heads))

        if self.eviction_mode == "union":
            if fixed_union_workspace is not None:
                return self._evict_union_fixed(
                    request,
                    decode_start,
                    decode_budget,
                    union_rows,
                    fixed_union_workspace,
                )
            return self._evict_union(
                request,
                num_layers,
                seq_len,
                decode_start,
                decode_budget,
                torch.cat(union_rows, dim=0),
                use_fixed_workspace=use_fixed_union,
                prewarm_key=fixed_union_prewarm_key,
            )
        if self.eviction_mode == "per_head":
            return self._evict_per_head(
                request, num_layers, seq_len, decode_start, decode_budget, per_layer_kv_scores
            )
        if self.eviction_mode == "per_layer_perhead":
            return self._evict_per_layer_perhead(
                request, num_layers, seq_len, decode_start, decode_budget, per_layer_kv_scores
            )
        raise ValueError(
            f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
            "'union', 'per_head', 'per_layer_perhead'"
        )

    def _evict_union_fixed(
        self,
        request: "LlmRequest",
        decode_start: int,
        decode_budget: int,
        score_segments: List["torch.Tensor"],
        workspace: _FixedUnionWorkspace,
    ) -> "torch.Tensor":
        """Select one exact-bucket request from caller-owned fixed buffers."""
        if workspace.keep_count != decode_budget or workspace.prompt_len != decode_start:
            raise ValueError("fixed union workspace no longer matches the request budget")
        self._fixed_union_active[request.py_request_id] = workspace
        return workspace.select_segments(
            score_segments,
            normalize_scores=self.normalize_scores,
        )

    def _evict_per_head(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        per_layer_kv_scores: List["torch.Tensor"],
    ) -> int:
        """per_head: each KV head's score = MEAN over layers of the per-layer MAX
        (upstream ``_select_per_head_independent``). One keep set per KV head,
        applied to EVERY layer."""
        agg = torch.stack(per_layer_kv_scores, dim=0).mean(dim=0)  # [nkv, dc]
        keep_2d = self._decode_topk(agg, decode_start, decode_budget)  # [nkv, keep_count]
        # One keep set shared by every (dense) layer.
        self._compact_perhead_layers(request, self._dense_layers(num_layers), keep_2d, seq_len)
        return int(keep_2d.shape[1])

    def _compact_perhead_layers(
        self, request: "LlmRequest", layer_indices: List[int], keep_2d: "torch.Tensor", seq_len: int
    ) -> None:
        """Compact dense layers with per-head source sets through the C++ op."""
        if int(keep_2d.shape[1]) >= seq_len:
            return  # nothing to drop
        mgr = self.kv_cache_manager
        get_buffers = mgr.get_buffers
        num_layers = self._num_layers_from_manager()
        prepared_layers = []
        for layer_idx in layer_indices:
            global_layer = self._global_layer_id(layer_idx, num_layers)
            pool = get_buffers(global_layer, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"Missing KV pool for attention layer {global_layer}")
            tokens_per_block = int(pool.shape[3])
            page_ids = self._resolve_page_ids(
                request,
                layer_idx,
                (seq_len + tokens_per_block - 1) // tokens_per_block,
            )
            if not page_ids:
                raise RuntimeError(
                    f"Missing KV page ids for attention layer {global_layer} "
                    f"of request {request.py_request_id}"
                )
            prepared_layers.append((layer_idx, pool, page_ids))
        from .triattention_kernels import cpp_sparse_compact

        for _, pool, page_ids in prepared_layers:
            page_ids_t = torch.as_tensor(page_ids, device=pool.device, dtype=torch.long)
            cpp_sparse_compact(
                pool,
                [page_ids_t],
                [keep_2d.to(pool.device)],
                [seq_len],
            )

    def _evict_per_layer_perhead(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        per_layer_kv_scores: List["torch.Tensor"],
    ) -> Optional[int]:
        """per_layer_perhead: each (layer, KV head) selects independently from
        that layer's per-KV-head max (upstream
        ``_select_per_layer_perhead_independent``). ``per_layer_kv_scores`` is
        DENSE-ONLY (sliding-window layers skipped in ``_evict_modes``), so pair it
        with the real dense layer indices rather than ``enumerate`` (whose 0..N-1
        running index would no longer match the physical layer)."""
        keep_count = None
        for layer_idx, kv_scores in zip(self._dense_layers(num_layers), per_layer_kv_scores):
            keep_2d = self._decode_topk(kv_scores, decode_start, decode_budget)
            self._compact_perhead_layers(request, [layer_idx], keep_2d, seq_len)
            keep_count = int(keep_2d.shape[1])
        return keep_count

    def _evict_union(
        self,
        request: "LlmRequest",
        num_layers: int,
        seq_len: int,
        decode_start: int,
        decode_budget: int,
        head_matrix: "torch.Tensor",
        use_fixed_workspace: bool = False,
        prewarm_key: Optional[tuple] = None,
    ) -> "torch.Tensor":
        """union: union of every head's top-k, re-ranked by the per-token max
        (upstream ``_select_union_based``). One 1-D keep set shared by every
        layer; returns the sorted kept slot indices in ``[0, seq_len)`` so the
        caller compacts all layers with the C++ V2 op."""
        request_id = request.py_request_id
        workspace = None
        if use_fixed_workspace:
            workspace = self._get_fixed_union_workspace(
                request_id,
                head_matrix,
                decode_budget,
                decode_start,
                prewarm_key=prewarm_key,
            )
        if workspace is not None:
            self._fixed_union_active[request_id] = workspace
            return workspace.select(head_matrix)

        self._fixed_union_active.pop(request_id, None)
        combined = head_matrix.max(dim=0).values  # [decode_count]
        keep_1d = self._select_union(head_matrix, combined, decode_budget)
        prefill_idx = torch.arange(decode_start, device=combined.device, dtype=torch.long)
        keep = torch.sort(torch.cat([prefill_idx, keep_1d + decode_start])).values
        return keep

    @staticmethod
    def _fixed_union_workspace_shape_key(
        rows: int,
        width: int,
        keep_count: int,
        prompt_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple:
        device = _FixedUnionWorkspace._canonical_device(device)
        return (rows, width, keep_count, prompt_len, dtype, str(device))

    @staticmethod
    def _build_fixed_shape_selection_workspaces(
        workspace: _FixedUnionWorkspace,
        max_requests: int,
    ) -> Tuple[_FixedUnionWorkspace, ...]:
        """Share one exact-bucket scratch while keeping request outputs stable."""
        if max_requests <= 0:
            raise ValueError("fixed shape selection requires a positive request capacity")
        workspace.prepare_segment_buffers()
        workspaces = [workspace]
        for _ in range(1, max_requests):
            workspaces.append(workspace.shared_selection_slot())
        return tuple(workspaces)

    def _fixed_shape_selection_for(
        self,
        score_workspace: Optional[_FixedScoreMetadataWorkspace],
        request_count: int,
    ) -> Optional[Tuple[_FixedUnionWorkspace, ...]]:
        """Return caller-owned slots only for a fully prewarmed exact bucket."""
        if not self._fixed_shape_selection_enabled or score_workspace is None or request_count <= 0:
            return None
        key = score_workspace.prewarm_key
        workspaces = self._fixed_shape_selection_workspaces.get(key)
        states = self._fixed_shape_selection_prewarm_states
        if (
            states.get(key) != "ready"
            or workspaces is None
            or len(workspaces) < request_count
            or not all(item.prewarmed for item in workspaces[:request_count])
        ):
            self._fixed_shape_selection_runtime_counts.setdefault(key, {"hit": 0, "fallback": 0})[
                "fallback"
            ] += 1
            return None
        self._fixed_shape_selection_runtime_counts.setdefault(key, {"hit": 0, "fallback": 0})[
            "hit"
        ] += 1
        return workspaces[:request_count]

    def _cross_request_selection_for(
        self,
        score_workspace: Optional[_FixedScoreMetadataWorkspace],
        request_count: int,
    ) -> Optional[_BatchedFixedUnionWorkspace]:
        """Return a ready upper-bucket Stage4 owner or preserve Stage3 fallback."""
        if (
            not self._cross_request_selection_enabled
            or self.eviction_mode != "union"
            or score_workspace is None
            or request_count <= 0
        ):
            return None
        key = score_workspace.prewarm_key
        workspace = self._cross_request_selection_workspaces.get(key)
        states = self._cross_request_selection_prewarm_states
        stage3_states = self._fixed_shape_selection_prewarm_states
        if (
            states.get(key) != "ready"
            or stage3_states.get(key) != "ready"
            or workspace is None
            or request_count > workspace.max_requests
        ):
            self._cross_request_selection_runtime_counts.setdefault(key, {"hit": 0, "fallback": 0})[
                "fallback"
            ] += 1
            return None
        self._cross_request_selection_runtime_counts.setdefault(key, {"hit": 0, "fallback": 0})[
            "hit"
        ] += 1
        return workspace

    def _select_cross_request_union(
        self,
        workspace: _BatchedFixedUnionWorkspace,
        score_workspace: _FixedScoreMetadataWorkspace,
        prepared: List[dict],
        req_layer_scores: List[dict],
        dense_layers: List[int],
    ) -> Tuple[_FixedUnionWorkspace, ...]:
        """Select every request from one backend-compatible upper bucket."""
        if self.eviction_mode != "union":
            raise RuntimeError("cross-request selection requires union eviction mode")
        segments_by_request = []
        valid_widths = []
        for request_index, item in enumerate(prepared):
            seq_len = item["seq_len"]
            prompt_len = min(item["request"].py_prompt_len, seq_len)
            valid_width = seq_len - prompt_len
            if (
                prompt_len != workspace.prompt_len
                or valid_width <= workspace.keep_count
                or valid_width > workspace.width
                or item["expected_keep_count"] != workspace.total_keep
            ):
                raise ValueError("cross-request selection geometry no longer matches its bucket")
            segments = []
            for layer in dense_layers:
                scores = req_layer_scores[request_index].get(layer)
                if scores is None:
                    raise RuntimeError(
                        "cross-request selection is missing a completed dense-layer score"
                    )
                segments.append(scores[:, prompt_len : prompt_len + workspace.width])
            segments_by_request.append(segments)
            valid_widths.append(valid_width)

        actual_backends = {self._selection_backend_for(width, self.top_B) for width in valid_widths}
        if actual_backends != {workspace.selection_backend}:
            raise ValueError("cross-request requests crossed a selection-backend band")
        workspace.stage_valid_widths_from_seq_lens(
            score_workspace.valid_seq_lens_device,
            len(prepared),
        )

        return workspace.select_requests(
            segments_by_request,
            normalize_scores=self.normalize_scores,
        )

    def _get_fixed_union_workspace(
        self,
        request_id: int,
        scores: torch.Tensor,
        keep_count: int,
        prompt_len: int,
        *,
        prewarm_key: Optional[tuple] = None,
    ) -> Optional[_FixedUnionWorkspace]:
        """Get a shape bucket when fixed selection preserves the eager contract."""
        if not self._fixed_union_enabled or scores.ndim != 2:
            return None
        rows, width = (int(value) for value in scores.shape)
        uses_fixed_route = keep_count > self._FIXED_UNION_MIN_KEEP
        if rows <= 0 or keep_count <= 0 or width <= keep_count or not uses_fixed_route:
            return None
        shape_key = self._fixed_union_workspace_shape_key(
            rows,
            width,
            keep_count,
            prompt_len,
            scores.dtype,
            scores.device,
        )
        if self._fixed_union_prewarm_enabled and prewarm_key is not None:
            states = self._fixed_union_prewarm_states
            workspace = self._fixed_union_prewarmed_workspaces.get(prewarm_key)
            if (
                states.get(prewarm_key) == "ready"
                and workspace is not None
                and workspace.prewarmed
                and workspace._matches_input(scores)
                and workspace.keep_count == keep_count
                and workspace.prompt_len == prompt_len
            ):
                return workspace

        key = (request_id, *shape_key)
        workspaces = self._fixed_union_workspaces
        workspace = workspaces.get(key)
        if workspace is None:
            workspace = _FixedUnionWorkspace(
                rows,
                width,
                keep_count,
                prompt_len,
                dtype=scores.dtype,
                device=scores.device,
            )
            workspaces[key] = workspace
        return workspace

    def _cute_dsl_topk_idx(self, scores: "torch.Tensor", k: int) -> "torch.Tensor":
        """Return unsorted per-row indices from the required CuTE-DSL top-k op."""
        if scores.ndim != 2:
            raise ValueError("TriAttention top-k scores must be two-dimensional")
        rows, width = scores.shape
        k = int(k)
        if rows <= 0:
            raise ValueError("TriAttention top-k requires at least one score row")
        self._selection_backend_for(int(width), k)
        if width == k:
            return torch.arange(width, device=scores.device, dtype=torch.long).expand(rows, -1)
        out = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
        seq_lens = torch.full((rows,), width, dtype=torch.int32, device=scores.device)
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(scores.contiguous(), seq_lens, out, k, 1)
        return out.to(torch.long)

    def _select_union(
        self, per_head_scores: "torch.Tensor", combined: "torch.Tensor", keep_count: int
    ) -> "torch.Tensor":
        """Each head picks its top-``keep_count``; take the union; from the union
        keep the top-``keep_count`` by ``combined``; if the union is smaller,
        fill from the highest-scoring remaining tokens. Decode-relative
        indices."""
        n = int(combined.shape[0])
        if n <= keep_count:
            return torch.arange(n, device=combined.device, dtype=torch.long)
        union_mask = torch.zeros(n, device=combined.device, dtype=torch.bool)
        quota = min(keep_count, n)
        # Top-k over all (layer x head) rows at once via the TRT-LLM CuTE-DSL
        # op: each row's top-quota is computed independently, collapsing H per-row
        # launches into one (H = num_layers * num_q_heads = 1152 for Qwen3-8B --
        # this was the dominant high-BS eviction cost). The union_mask scatter
        # below is order-independent, so the (unsorted) indices are fine.
        top_idx = self._cute_dsl_topk_idx(per_head_scores, quota)
        union_mask[top_idx.reshape(-1)] = True
        union_idx = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_idx.numel() >= keep_count:
            subset = combined.index_select(0, union_idx)
            top_subset = self._cute_dsl_topk_idx(subset.unsqueeze(0), keep_count).squeeze(0)
            return union_idx.index_select(0, torch.sort(top_subset).values)
        remaining = keep_count - int(union_idx.numel())
        if remaining > 0:
            residual = combined.clone()
            residual[union_mask] = float("-inf")
            extra = self._cute_dsl_topk_idx(
                residual.unsqueeze(0), min(remaining, n - int(union_idx.numel()))
            ).squeeze(0)
            union_idx = torch.cat([union_idx, extra])
        return torch.sort(union_idx).values

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

    def _global_layer_id(self, local_layer: int, num_layers: int) -> int:
        return self._local_to_global_layers(num_layers)[local_layer]

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
        if not self.skip_swa:
            return list(range(num_layers)), [], None
        cached = self._attention_layer_partition_cache
        if cached is not None:
            return cached

        model_path = self.model_path
        if model_path is None:
            raise ValueError("TriAttention skip_swa=True requires model_path")

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

    def _dense_layers(self, num_layers: int) -> List[int]:
        """Return full-attention layers used for TriAttention scoring."""
        return self._attention_layer_partition(num_layers)[0]

    def _record_fixed_score_runtime(self, key: Optional[tuple], outcome: str) -> None:
        if key is None:
            return
        bucket = self._fixed_score_runtime_counts.setdefault(
            key, {"hit": 0, "fallback": 0, "rejected": 0}
        )
        bucket[outcome] += 1

    def _fixed_score_workspace_for(
        self,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        dense_groups: List[List[int]],
        swa_layers: List[int],
        num_layers: int,
        prepared: List[dict],
    ) -> Optional[_FixedScoreMetadataWorkspace]:
        if not self._fixed_score_metadata_enabled or not prepared:
            return None
        seq_lens = [item["seq_len"] for item in prepared]
        prompt_lens = {min(item["request"].py_prompt_len, item["seq_len"]) for item in prepared}
        if len(prompt_lens) != 1:
            return None
        prompt_len = next(iter(prompt_lens))
        max_seq_len = max(seq_lens)
        actual_backends = {
            self._selection_backend_for(seq_len - prompt_len, self.top_B) for seq_len in seq_lens
        }
        if len(actual_backends) != 1:
            return None
        actual_backend = next(iter(actual_backends))
        representatives = [group[0] for group in dense_groups]
        representatives.extend(layer for layer in swa_layers if layer not in representatives)
        candidates = []
        for key, workspace in self._fixed_score_workspaces.items():
            selection_plan = self._cross_request_selection_plans.get(key)
            if (
                self._fixed_score_prewarm_states.get(key) == "ready"
                and workspace.prompt_len == prompt_len
                and workspace.bucket_seq_len >= max_seq_len
                and len(prepared) <= workspace.max_requests
                and (
                    not self._cross_request_selection_enabled
                    or (
                        selection_plan is not None
                        and selection_plan.selection_backend == actual_backend
                    )
                )
                and workspace.matches(layer_pools, dense_groups, representatives)
            ):
                candidates.append((workspace.bucket_seq_len, key, workspace))
        if not candidates:
            return None
        _, _, workspace = min(candidates, key=lambda candidate: candidate[0])
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

    def _attach_page_ids(
        self,
        prepared: List[dict],
        dense_representatives: List[int],
        swa_layers: List[int],
        layer_pools: List[torch.Tensor],
        global_layers: List[int],
        workspace: Optional[_FixedScoreMetadataWorkspace],
    ) -> bool:
        fixed = False
        if workspace is not None:
            try:
                fixed = workspace.stage(
                    self.kv_cache_manager,
                    [item["request_id"] for item in prepared],
                    [item["round_start"] for item in prepared],
                    [item["seq_len"] for item in prepared],
                )
            except _FixedScoreStreamMismatch:
                self._record_fixed_score_runtime(workspace.prewarm_key, "rejected")
                raise
            except Exception as exc:
                logger.warning(f"TriAttention fixed score staging failed; using eager path: {exc}")
        required_layers = [*dense_representatives, *swa_layers]
        if fixed:
            for request_index, item in enumerate(prepared):
                item["page_ids"] = {
                    layer: workspace.page_ids_device[
                        workspace.representative_slots[layer], request_index
                    ]
                    for layer in required_layers
                }
            self._record_fixed_score_runtime(workspace.prewarm_key, "hit")
            return True

        if workspace is not None:
            self._record_fixed_score_runtime(workspace.prewarm_key, "fallback")
        for item in prepared:
            request = item["request"]
            request_id = item["request_id"]
            item["page_ids"] = {}
            for layer in required_layers:
                tokens_per_block = int(layer_pools[layer].shape[3])
                page_ids = self._resolve_page_ids(
                    request,
                    layer,
                    (item["seq_len"] + tokens_per_block - 1) // tokens_per_block,
                )
                if not page_ids:
                    raise RuntimeError(
                        f"Missing KV page ids for attention layer {global_layers[layer]} "
                        f"of request {request_id}"
                    )
                item["page_ids"][layer] = torch.as_tensor(
                    page_ids, device=layer_pools[layer].device, dtype=torch.int64
                )
        return False

    def _standalone_graph_bucket_for(
        self,
        prepared: List[dict],
        score_workspace: Optional[_FixedScoreMetadataWorkspace],
        selection_workspace: Optional[_BatchedFixedUnionWorkspace],
    ) -> Optional[tuple]:
        """Return one ready upper-workspace graph bucket, or reject to Stage4."""
        request_count = len(prepared)
        if (
            not self._standalone_cuda_graph_enabled
            or score_workspace is None
            or selection_workspace is None
            or request_count <= 0
        ):
            return None
        prewarm_key = score_workspace.prewarm_key
        if (
            prewarm_key is None
            or self._fixed_score_prewarm_states.get(prewarm_key) != "ready"
            or self._cross_request_selection_prewarm_states.get(prewarm_key) != "ready"
            or self._fixed_score_workspaces.get(prewarm_key) is not score_workspace
            or self._cross_request_selection_workspaces.get(prewarm_key) is not selection_workspace
        ):
            return None
        seq_lens = [item["seq_len"] for item in prepared]
        prompt_lens = {min(item["request"].py_prompt_len, item["seq_len"]) for item in prepared}
        if len(prompt_lens) != 1:
            return None
        prompt_len = next(iter(prompt_lens))
        bucket_seq_len = score_workspace.bucket_seq_len
        width = bucket_seq_len - prompt_len
        actual_backends = {
            self._selection_backend_for(seq_len - prompt_len, self.top_B) for seq_len in seq_lens
        }
        if len(actual_backends) != 1:
            return None
        expected_backend = next(iter(actual_backends))
        if (
            max(seq_lens) > bucket_seq_len
            or min(seq_len - prompt_len for seq_len in seq_lens) <= self.top_B
            or selection_workspace.selection_backend != expected_backend
            or selection_workspace.prompt_len != prompt_len
            or selection_workspace.width != width
            or selection_workspace.keep_count != self.top_B
            or any(item.get("expected_keep_count") != prompt_len + self.top_B for item in prepared)
        ):
            return None
        if min(score_workspace.max_requests, selection_workspace.max_requests) < request_count:
            return None
        return (
            "triattention.standalone-eviction-graph.bucket.v2",
            prewarm_key,
            request_count,
            bucket_seq_len,
            prompt_len,
            self.top_B,
            expected_backend,
        )

    def _record_standalone_graph_runtime(self, outcome: str, request_count: int) -> None:
        counts = self._standalone_graph_runtime_counts
        counts[outcome] = counts.get(outcome, 0) + 1
        request_key = f"{outcome}_requests"
        counts[request_key] = counts.get(request_key, 0) + request_count

    def _standalone_graph_cache_for(self):
        cache = self._standalone_graph_cache
        if cache is None:
            from .cuda_graph import StandaloneEvictionGraphCache

            cache = StandaloneEvictionGraphCache(
                max_entries=int(os.environ.get("TRIATTN_CUDA_GRAPH_MAX_ENTRIES", "8")),
                max_bytes=int(os.environ.get("TRIATTN_CUDA_GRAPH_MAX_BYTES", str(4 * 1024**3))),
            )
            self._standalone_graph_cache = cache
        return cache

    def _standalone_graph_workspace_for(
        self,
        *,
        key: tuple,
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        swa_layers: List[int],
        layer_group_representative: Dict[int, int],
        global_layers: List[int],
        score_workspace: _FixedScoreMetadataWorkspace,
        selection_workspace: _BatchedFixedUnionWorkspace,
        request_count: int,
        seq_len: int,
        prompt_len: int,
        swa_window: Optional[int],
    ):
        cache = self._standalone_graph_cache_for()
        workspace = cache.workspace_for(key)
        if workspace is not None and workspace.matches_runtime(
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            layer_group_representative=layer_group_representative,
            global_layers=global_layers,
            score_workspace=score_workspace,
            selection_workspace=selection_workspace,
        ):
            return workspace

        from .cuda_graph import FixedBatchedCompactionWorkspace

        self._standalone_graph_arena_generation += 1
        workspace = FixedBatchedCompactionWorkspace(
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            layer_group_representative=layer_group_representative,
            global_layers=global_layers,
            score_workspace=score_workspace,
            selection_workspace=selection_workspace,
            request_count=request_count,
            seq_len=seq_len,
            prompt_len=prompt_len,
            decode_keep_count=self.top_B,
            swa_window=swa_window,
            arena_generation=self._standalone_graph_arena_generation,
        )
        return workspace

    def _try_standalone_cuda_graph(
        self,
        *,
        prepared: List[dict],
        layer_pools: List[torch.Tensor],
        dense_layers: List[int],
        dense_groups: List[List[int]],
        swa_layers: List[int],
        swa_window: Optional[int],
        layer_group_representative: Dict[int, int],
        global_layers: List[int],
        score_workspace: Optional[_FixedScoreMetadataWorkspace],
        selection_workspace: Optional[_BatchedFixedUnionWorkspace],
        fixed_perhead_segment_views,
    ) -> Optional[List[Tuple[int, int]]]:
        """Run an upper-bucket standalone eviction graph, or leave Stage4 untouched."""
        request_count = len(prepared)
        self._record_standalone_graph_runtime("attempt", request_count)
        key = self._standalone_graph_bucket_for(
            prepared,
            score_workspace,
            selection_workspace,
        )
        if key is None:
            self._record_standalone_graph_runtime("admission_rejected", request_count)
            return None
        assert score_workspace is not None
        assert selection_workspace is not None
        seq_len = key[3]
        prompt_len = key[4]
        cache = self._standalone_graph_cache_for()
        if cache.is_disabled(key):
            cache.record_fallback(key=key, request_count=request_count)
            self._record_standalone_graph_runtime("fallback", request_count)
            return None
        stream = torch.cuda.current_stream(score_workspace.device)
        staged_stream = score_workspace.stream
        if staged_stream is None or (stream.device, stream.cuda_stream) != (
            staged_stream.device,
            staged_stream.cuda_stream,
        ):
            raise _FixedScoreStreamMismatch(
                "TriAttention graph replay must use the fixed staging CUDA stream"
            )
        try:
            workspace = self._standalone_graph_workspace_for(
                key=key,
                layer_pools=layer_pools,
                dense_layers=dense_layers,
                swa_layers=swa_layers,
                layer_group_representative=layer_group_representative,
                global_layers=global_layers,
                score_workspace=score_workspace,
                selection_workspace=selection_workspace,
                request_count=request_count,
                seq_len=seq_len,
                prompt_len=prompt_len,
                swa_window=swa_window,
            )
            fingerprint = workspace.pointer_fingerprint(stream)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "TriAttention standalone CUDA Graph workspace rejected; "
                f"using Stage4 eager execution: {exc}"
            )
            cache.record_fallback(key=key, request_count=request_count)
            self._record_standalone_graph_runtime("workspace_rejected", request_count)
            self._record_standalone_graph_runtime("fallback", request_count)
            return None

        def graph_body() -> None:
            score_workspace.prepare_phase(request_count)
            req_layer_scores = [dict() for _ in prepared]
            # ONE fused launch over (request x dense layer) segments.
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
            for request_index in range(request_count):
                for layer_slot, layer in enumerate(layer_order):
                    req_layer_scores[request_index][layer] = views[:, request_index, layer_slot]
            segments_by_request = [
                [
                    req_layer_scores[request_index][layer][:, prompt_len:seq_len]
                    for layer in dense_layers
                ]
                for request_index in range(request_count)
            ]
            selection_workspace.stage_valid_widths_from_seq_lens(
                score_workspace.valid_seq_lens_device,
                request_count,
            )
            selection_workspace.select_requests(
                segments_by_request,
                normalize_scores=self.normalize_scores,
            )
            workspace.launch()

        expected_outcome = cache.classify(key, fingerprint)
        try:
            with nvtx_range(f"triattention.cuda_graph.{expected_outcome}", color="green"):
                outcome = cache.execute(
                    key=key,
                    request_count=request_count,
                    fingerprint=fingerprint,
                    workspace=workspace,
                    capture_body=graph_body,
                )
        except RuntimeError:
            self._record_standalone_graph_runtime("failure", request_count)
            raise
        logger.debug(
            "TriAttention standalone CUDA Graph outcome: "
            f"outcome={outcome}, requests={request_count}, "
            f"width={seq_len - prompt_len}, budget={self.top_B}, "
            f"backend={selection_workspace.selection_backend}"
        )
        if outcome == "fallback":
            self._record_standalone_graph_runtime("fallback", request_count)
            return None
        self._record_standalone_graph_runtime("success", request_count)

        capacity_targets = []
        for request_index, item in enumerate(prepared):
            keep_count = item["expected_keep_count"]
            evicted = item["seq_len"] - keep_count
            if evicted <= 0:
                raise RuntimeError("standalone eviction graph captured an identity compaction")
            request_id = item["request_id"]
            self._evicted[request_id] = self._evicted.get(request_id, 0) + evicted
            self._confirmed_kv_lengths[request_id] = keep_count
            capacity_targets.append((request_id, keep_count))
        return capacity_targets

    def _standalone_cuda_graph_stats(self) -> dict:
        """Return host-side counters used to seal graph-hit profiling windows."""
        cache = self._standalone_graph_cache
        enabled = self._standalone_cuda_graph_enabled
        if cache is None:
            if not enabled:
                return {}
            cache = self._standalone_graph_cache_for()
        return {
            **cache.snapshot(),
            "enabled": enabled,
            "runtime": dict(self._standalone_graph_runtime_counts),
        }

    def _evict_requests(self, evict_reqs, num_layers: int) -> List[Tuple[int, int]]:
        """Score and compact requests, returning ``(request_id, capacity)`` targets.

        Only full-attention layers participate in scoring. For kernel-masked SWA
        layers, the latest model window is rebased to the tail of the common
        compacted prefix before the request-wide capacity is reduced.
        """
        from .triattention_kernels import (
            fixed_perhead_segment_views,
            flat_perhead_to_list,
            triton_tri_score_perhead,
        )

        mgr = self.kv_cache_manager
        get_buffers = mgr.get_buffers
        global_layers = self._local_to_global_layers(num_layers)
        layer_pools = [get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(p is None for p in layer_pools):
            missing = [layer for layer, pool in zip(global_layers, layer_pools) if pool is None]
            raise RuntimeError(f"Missing KV pools for attention layers {missing}")
        dense_layers, swa_layers, swa_window = self._attention_layer_partition(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        first_dense_layer = dense_layers[0]
        device = layer_pools[first_dense_layer].device

        # Dense pools may have distinct block tables. Resolve one representative
        # layer per storage group and reuse its page ids only within that group.
        from collections import defaultdict as _defaultdict

        storage_groups = _defaultdict(list)
        for layer in dense_layers:
            storage_groups[layer_pools[layer].untyped_storage().data_ptr()].append(layer)
        dense_group_representatives = [layers[0] for layers in storage_groups.values()]
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }

        # Resolve request length and page metadata before mutating any layer.
        prepared = []
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, rid in evict_reqs:
                seq_len = self._confirmed_kv_lengths.get(rid)
                if seq_len is None:
                    raise RuntimeError(f"Missing confirmed KV length for request {rid}")
                # Restore the uncompressed confirmed logical position from the
                # physical prefix and cumulative eviction count.
                round_start = seq_len + self._evicted.get(rid, 0)
                if seq_len <= self._minimum_evictable_length(request, seq_len):
                    continue
                expected_keep_count = self._minimum_evictable_length(request, seq_len)
                swa_source = None
                swa_destination = None
                if swa_layers:
                    assert swa_window is not None
                    swa_source, swa_destination = _build_swa_rebase_copy(
                        seq_len, expected_keep_count, swa_window, device=device
                    )
                prepared.append(
                    {
                        "request": request,
                        "request_id": rid,
                        "seq_len": int(seq_len),
                        "round_start": float(round_start),
                        "expected_keep_count": expected_keep_count,
                        "swa_source": swa_source,
                        "swa_destination": swa_destination,
                    }
                )
        if not prepared:
            return []
        dense_groups = list(storage_groups.values())
        fixed_score_workspace = self._fixed_score_workspace_for(
            layer_pools,
            dense_layers,
            dense_groups,
            swa_layers,
            num_layers,
            prepared,
        )
        fixed_score_active = self._attach_page_ids(
            prepared,
            dense_group_representatives,
            swa_layers,
            layer_pools,
            global_layers,
            fixed_score_workspace,
        )
        cross_request_selection = (
            self._cross_request_selection_for(fixed_score_workspace, len(prepared))
            if fixed_score_active
            else None
        )
        fixed_selection_workspaces = None
        if (
            fixed_score_active
            and cross_request_selection is None
            and all(item["seq_len"] == fixed_score_workspace.bucket_seq_len for item in prepared)
        ):
            fixed_selection_workspaces = self._fixed_shape_selection_for(
                fixed_score_workspace,
                len(prepared),
            )
        seq_lens = [item["seq_len"] for item in prepared]
        round_starts = [item["round_start"] for item in prepared]
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, device)
        q_real, q_imag, mlr_coef = self._local_score_calibration(
            num_layers,
            global_layers,
        )
        graph_capacity_targets = self._try_standalone_cuda_graph(
            prepared=prepared,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            dense_groups=dense_groups,
            swa_layers=swa_layers,
            swa_window=swa_window,
            layer_group_representative=layer_group_representative,
            global_layers=global_layers,
            score_workspace=fixed_score_workspace if fixed_score_active else None,
            selection_workspace=cross_request_selection,
            fixed_perhead_segment_views=fixed_perhead_segment_views,
        )
        if graph_capacity_targets is not None:
            return graph_capacity_targets
        if fixed_score_active:
            fixed_score_workspace.prepare_phase(len(prepared))
        # Score ALL dense layers in ONE launch. Segments carry per-layer base
        # addresses and per-(request, layer) page tables, so distinct per-layer
        # storages/block tables no longer force one launch per storage group.
        req_layer_scores = [dict() for _ in prepared]  # [req] -> {layer_idx: [H,seq]}
        with nvtx_range("triattention.score", color="blue"):
            if fixed_score_active:
                ph, _ = fixed_score_workspace.fused_group.launch(
                    len(prepared),
                    fixed_score_workspace.round_starts_device,
                    fixed_score_workspace.mean_cos,
                    fixed_score_workspace.mean_sin,
                    self.score_aggregation,
                )
                layer_order = fixed_score_workspace.dense_layer_order
                fixed_views = fixed_perhead_segment_views(
                    ph,
                    len(prepared),
                    len(layer_order),
                    fixed_score_workspace.bucket_seq_len,
                )
                for request in range(len(prepared)):
                    for layer_slot, layer in enumerate(layer_order):
                        req_layer_scores[request][layer] = fixed_views[:, request, layer_slot]
            else:
                # Each layer keeps its own block table (V2 allocates pages per
                # layer); the staged page_ids dict is keyed by the layer's
                # storage-group representative.
                page_ids_per_layer = [
                    [item["page_ids"][layer_group_representative[layer]] for layer in dense_layers]
                    for item in prepared
                ]
                ph, so, sm = triton_tri_score_perhead(
                    layer_pools,
                    [per_layer[0] for per_layer in page_ids_per_layer],
                    seq_lens,
                    round_starts,
                    q_real,
                    q_imag,
                    mlr_coef,
                    self._freq_scale_sq,
                    self.calibration["omega"],
                    self._offsets,
                    self._H,
                    score_aggregation=self.score_aggregation,
                    layer_indices=dense_layers,
                    page_ids_per_layer=page_ids_per_layer,
                )
                seg_list = flat_perhead_to_list(ph, so)
                for scores, meta in zip(seg_list, sm):
                    req_layer_scores[meta.request_index][meta.layer_index] = scores
        cross_request_workspaces = None
        if cross_request_selection is not None:
            with nvtx_range("triattention.select", color="yellow"):
                with nvtx_range("triattention.select.cross_request", color="orange"):
                    cross_request_workspaces = self._select_cross_request_union(
                        cross_request_selection,
                        fixed_score_workspace,
                        prepared,
                        req_layer_scores,
                        dense_layers,
                    )
        # Per request: select from the precomputed scores. All scores were taken
        # before any compaction; requests touch disjoint pages, so compacting one
        # does not disturb another's (already-read) scores.
        #
        # For union (1-D layer-uniform keep) we compute keep WITHOUT compacting,
        # accumulate per layer, then run ONE compaction per layer over all
        # requests -> K*L*2 compact launches collapse to L*2 (bit-exact;
        # kernel-equivalence validated by test_compact_equiv.py).
        is_union = self.eviction_mode == "union"
        union_by_layer = {} if is_union else None
        swa_by_layer = {}
        pending_updates = []
        fixed_union_compaction = None
        for r, item in enumerate(prepared):
            request = item["request"]
            rid = item["request_id"]
            seq_len = item["seq_len"]
            precomputed = [req_layer_scores[r].get(layer) for layer in range(num_layers)]
            if any(precomputed[layer] is None for layer in dense_layers):
                continue
            if cross_request_workspaces is not None:
                keep_count = cross_request_workspaces[r].keep
            else:
                fixed_union_prewarm_key = None
                fixed_union_workspace = (
                    fixed_selection_workspaces[r]
                    if fixed_selection_workspaces is not None
                    else None
                )
                if (
                    is_union
                    and len(prepared) == 1
                    and fixed_union_workspace is None
                    and self._fixed_union_prewarm_enabled
                ):
                    prompt_len = min(request.py_prompt_len, seq_len)
                    fixed_union_prewarm_key = self._fixed_union_prewarm_key(
                        layer_pools,
                        dense_layers,
                        list(storage_groups.values()),
                        num_layers,
                        seq_len,
                        prompt_len,
                    )
                with nvtx_range("triattention.select", color="yellow"):
                    keep_count = self._evict_modes(
                        request,
                        num_layers,
                        seq_len,
                        precomputed=precomputed,
                        use_fixed_union=(
                            is_union and len(prepared) == 1 and fixed_union_workspace is None
                        ),
                        fixed_union_prewarm_key=fixed_union_prewarm_key,
                        fixed_union_workspace=fixed_union_workspace,
                    )
            if keep_count is None:
                continue
            if is_union and isinstance(keep_count, torch.Tensor):
                keep = keep_count
                keep_count = int(keep.numel())
                workspace = (
                    cross_request_workspaces[r]
                    if cross_request_workspaces is not None
                    else self._fixed_union_active.get(rid)
                )
                can_use_fixed_compaction = (
                    len(prepared) == 1
                    and self._fixed_union_compaction_enabled
                    and workspace is not None
                    and not workspace.selection_only
                    and workspace.keep is keep
                    and all(
                        workspace.can_compact(
                            layer_pools[lid],
                            item["page_ids"][layer_group_representative[lid]],
                            seq_len,
                        )
                        for lid in dense_layers
                    )
                )
                if can_use_fixed_compaction:
                    fixed_union_compaction = (workspace, item, seq_len)
                else:
                    for lid in dense_layers:
                        grp = union_by_layer.setdefault(lid, ([], [], []))
                        representative = layer_group_representative[lid]
                        grp[0].append(item["page_ids"][representative])
                        grp[1].append(keep)
                        grp[2].append(seq_len)
            if keep_count != item["expected_keep_count"]:
                raise RuntimeError(
                    f"TriAttention selected {keep_count} tokens for request {rid}, "
                    f"expected {item['expected_keep_count']}"
                )
            evicted = seq_len - keep_count
            if evicted > 0:
                for lid in swa_layers:
                    grp = swa_by_layer.setdefault(lid, ([], [], [], []))
                    grp[0].append(item["page_ids"][lid])
                    grp[1].append(item["swa_source"])
                    grp[2].append(seq_len)
                    grp[3].append(item["swa_destination"])
                pending_updates.append((rid, evicted, keep_count))

        if union_by_layer or swa_by_layer:
            from .triattention_kernels import cpp_sparse_compact

        if union_by_layer:
            for lid, (pl, kl, sl) in union_by_layer.items():
                with nvtx_range("triattention.compact", color="purple"):
                    cpp_sparse_compact(layer_pools[lid], pl, kl, sl)
        if fixed_union_compaction is not None:
            workspace, item, seq_len = fixed_union_compaction
            for lids in storage_groups.values():
                representative = lids[0]
                page_ids = item["page_ids"][representative]
                copied_buffer_keys = set()
                for lid in lids:
                    buffer_key = workspace._compaction_key(layer_pools[lid], page_ids)
                    with nvtx_range("triattention.compact", color="purple"):
                        workspace.compact(
                            layer_pools[lid],
                            page_ids,
                            seq_len,
                            copy_page_ids=buffer_key not in copied_buffer_keys,
                        )
                    copied_buffer_keys.add(buffer_key)
        for lid, (pl, sources, seq_lens, destinations) in swa_by_layer.items():
            with nvtx_range("triattention.compact", color="purple"):
                cpp_sparse_compact(
                    layer_pools[lid],
                    pl,
                    sources,
                    seq_lens,
                    dest_list=destinations,
                )

        capacity_targets = []
        for rid, evicted, keep_count in pending_updates:
            self._evicted[rid] = self._evicted.get(rid, 0) + evicted
            self._confirmed_kv_lengths[rid] = keep_count
            capacity_targets.append((rid, keep_count))
        return capacity_targets

    # ------------------------------------------------------------------ #
    # V2-manager cache access + physical eviction (HND physical layout)  #
    # ------------------------------------------------------------------ #

    def _rebase_protected_tail(
        self,
        request: "LlmRequest",
        *,
        source_start: int,
        destination_start: int,
        token_count: int,
    ) -> None:
        """Move an in-flight speculative suffix without inspecting its tokens.

        Native speculative acceptance and V2 rewind run before this hook. The
        remaining suffix belongs to the next forward, which overlap scheduling
        has already enqueued. It is outside TriAttention's selection domain, but
        shrinking the confirmed prefix changes its ordinal. Rebase the suffix
        identically for every target layer before releasing tail pages.
        """
        if token_count <= 0 or source_start == destination_start:
            return
        if min(source_start, destination_start) < 0:
            raise ValueError("protected-tail offsets must be non-negative")

        from .triattention_kernels import cpp_sparse_compact

        num_layers = self._num_layers_from_manager()
        source = None
        destination = None
        source_end = source_start + token_count
        for local_layer in range(num_layers):
            global_layer = self._global_layer_id(local_layer, num_layers)
            pool = self.kv_cache_manager.get_buffers(global_layer, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"Missing KV pool for protected-tail layer {global_layer}")
            tokens_per_block = int(pool.shape[3])
            page_count = (source_end + tokens_per_block - 1) // tokens_per_block
            page_ids = self._resolve_page_ids(request, local_layer, page_count)
            if not page_ids:
                raise RuntimeError(
                    f"Missing KV page ids for protected tail of request "
                    f"{request.py_request_id}, layer {global_layer}"
                )
            if source is None:
                source = torch.arange(
                    source_start,
                    source_end,
                    dtype=torch.long,
                    device=pool.device,
                )
                destination = torch.arange(
                    destination_start,
                    destination_start + token_count,
                    dtype=torch.long,
                    device=pool.device,
                )
            cpp_sparse_compact(
                pool,
                [torch.as_tensor(page_ids, dtype=torch.long, device=pool.device)],
                [source],
                [source_end],
                dest_list=[destination],
            )

    def _resolve_page_ids(
        self,
        request: "LlmRequest",
        layer_idx: int,
        page_count: Optional[int] = None,
    ) -> Optional[List[int]]:
        """Return the page (block) ids that hold THIS request's KV for one layer.

        The V2 KV cache is PAGED: a request's tokens live in several (possibly
        non-contiguous) fixed-size pages inside one big shared pool. Before we
        can read or compact a request's cache we must know WHICH pages are its
        own. V2's ``get_batch_cache_indices([ids], layer_idx)`` returns one list
        of block ids per requested id; we pass a single id and take ``[0]``.

        These ids index the PAGE axis (dim 0) of the tensor ``get_buffers``
        returns; the key/value split is a SEPARATE axis (kv_factor, dim 1) that
        callers index on their own. We do NOT divide or rescale the ids here.

        Returns ``None`` when V2 has no valid page metadata for the request.
        Padded tail slots are excluded by V2's live/requested-block bound. A
        negative id inside the required prefix is an invariant violation; it
        must not be removed because that would shift every later block ordinal.
        """
        try:
            num_layers = self._num_layers_from_manager()
            global_layer = self._global_layer_id(layer_idx, num_layers)
            kwargs = {"num_blocks_per_seq": [page_count]} if page_count is not None else {}
            batch = self.kv_cache_manager.get_batch_cache_indices(
                [request.py_request_id], global_layer, **kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve KV pages for local layer {layer_idx} "
                f"of request {request.py_request_id}"
            ) from exc
        if batch:
            page_ids = batch[0]
            if page_count is not None and len(page_ids) != page_count:
                return None
            if any(page < 0 for page in page_ids):
                return None
            return page_ids or None
        return None

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
