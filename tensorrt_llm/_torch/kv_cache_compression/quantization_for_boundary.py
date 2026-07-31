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

import math
import threading
from collections.abc import Sequence

import torch

from tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4 import dequant_nvfp4_2d_triton
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.runtime.kv_cache_manager_v2._common import CacheTier, MemAddress
from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
from tensorrt_llm.runtime.kv_cache_manager_v2._page import Page
from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import Slot

from ..pyexecutor.resource_manager import DataType, KVCacheCompressionManager
from .interface import (
    NVFP4_BOUNDARY_BLOCK_SIZE,
    NVFP4_BOUNDARY_RECORD_ALIGNMENT,
    nvfp4_boundary_record_layout,
)

_NVFP4_GLOBAL_SCALE_DENOMINATOR = 448.0 * 6.0
_SUPPORTED_ROLES = {"key", "value"}


class QuantizationForBoundaryCompression(KVCacheCompressionManager):
    """Compress GPU KV while KVCM V2 offloads it to the Host tier.

    The active GPU cache keeps its configured runtime representation (FP16 or
    BF16 in the initial proof). The stable Host copy is compressed-only. On
    recall, KVCM V2 allocates the normal GPU Page first and this same manager
    restores the Host record into that Page before KVCM publishes it.

    ``StorageManager`` remains responsible for page selection, destination
    admission, CUDA-event ordering, publication, source release, and rollback.
    This manager owns only the two representation transforms. It never reads
    attention state or attention metadata.

    This proof uses the existing NVFP4 quantization op and the existing linear
    Triton dequantization algorithm. The latter writes directly into the raw
    GPU Page pre-admitted by KVCM V2. Temporary packed tensors are released
    buffer-by-buffer and recorded on KVCM's migration stream; they are not
    stable cache backing. One aligned record staging allocation is kept for
    the manager lifetime and serialized across migration streams to bound that
    part of peak workspace.
    """

    supports_block_reuse = True

    def __init__(
        self,
        kv_cache_manager,
        draft_kv_cache_manager=None,
        *,
        quant: str,
    ) -> None:
        if draft_kv_cache_manager is not None:
            raise ValueError("NVFP4 Host offload does not support a draft KV cache")
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        if quant != "nvfp4":
            raise ValueError("QuantizationForBoundaryCompression only supports quant='nvfp4'")
        if kv_cache_manager.is_draft:
            raise ValueError("NVFP4 Host offload must bind to the target KV cache")
        if kv_cache_manager.dtype not in (DataType.HALF, DataType.BF16):
            raise ValueError("Active runtime KV must remain FP16 or BF16")
        if kv_cache_manager.boundary_compression_quant != quant:
            raise ValueError(
                "KVCM V2 must receive the boundary quantization before storage construction"
            )

        self.quant = quant
        self._torch_dtype = binding_to_torch_dtype(kv_cache_manager.dtype)
        # A production KVCacheManagerV2 always owns the stream that also owns
        # its Page addresses.  Import-light unit harnesses intentionally omit
        # it and exercise only pure layout/config behavior.
        execution_stream = getattr(kv_cache_manager, "_stream", None)
        self._device = (
            torch.device(execution_stream.device) if execution_stream is not None else None
        )
        self._validate_execution_device()
        self._validate_storage_layout()
        self._record_staging = self._allocate_record_staging()
        self._record_staging_ready = None
        if self._device is not None:
            # CUDA events initialize lazily on their first record. Do that at
            # construction, before cache pressure, so a later completion
            # record cannot fail while leaving shared staging unprotected.
            self._record_staging_ready = torch.cuda.Event(blocking=False)
            with torch.cuda.device(self._device):
                self._record_staging_ready.record(execution_stream)
                self._record_staging_ready.synchronize()
        self._record_staging_in_flight = False
        self._record_staging_lock = threading.Lock()
        kv_cache_manager.bind_boundary_compression_hooks(self)

    def shutdown(self) -> None:
        """Release persistent staging only after its last stream use finishes."""
        with self._record_staging_lock:
            if self._record_staging_in_flight:
                assert self._record_staging_ready is not None
                self._record_staging_ready.synchronize()
            self._record_staging = None
            self._record_staging_in_flight = False

    def _migration_device(self) -> torch.device:
        """Return the CUDA device that owns KVCM's addresses and stream.

        Production managers always expose their execution stream.  The lazy
        fallback keeps import-light unit harnesses usable and is resolved only
        when a real data hook executes.
        """
        return self._device or torch.device("cuda", torch.cuda.current_device())

    def _validate_execution_device(self) -> None:
        """Fail at manager binding instead of at the first pressure event.

        Real KVCM V2 managers already own a CUDA stream, so their device is
        known before Host pools become usable.  Import-light unit harnesses do
        not expose a stream and intentionally defer this check to the first
        data hook.
        """
        if self._device is None:
            return
        if self._device.type != "cuda":
            raise RuntimeError("NVFP4 boundary compression requires a CUDA device")
        major, _ = torch.cuda.get_device_capability(self._device)
        if major < 10:
            raise RuntimeError("NVFP4 boundary compression requires SM100 or newer")
        if not hasattr(torch.ops.trtllm, "fp4_quantize"):
            raise RuntimeError("TensorRT-LLM fp4_quantize operator is unavailable")

    @staticmethod
    def _record_size(raw_size: int) -> int:
        return nvfp4_boundary_record_layout(raw_size)[3]

    @staticmethod
    def _record_sections(raw_size: int, host_size: int) -> tuple[int, int, int]:
        packed_size, scale_size, inverse_scale_offset, expected = nvfp4_boundary_record_layout(
            raw_size
        )
        if host_size != expected:
            raise ValueError(
                f"NVFP4 Host record size mismatch: expected {expected}, got {host_size}"
            )
        return packed_size, scale_size, inverse_scale_offset

    def _allocate_record_staging(self) -> torch.Tensor | None:
        """Allocate the one reusable GPU record used by both boundaries.

        The existing NVFP4 operator still allocates its packed value and scale
        outputs.  Reusing one aligned record removes the second per-buffer
        allocation and bounds record staging independently of Page batch size.
        Cross-call ordering is protected by ``_record_staging_ready``.
        """
        if self._device is None:
            return None
        max_record_size = max(
            coalesced.effective_host_single_buffer_size
            for pool_group in self.kv_cache_manager.impl.pool_group_descs
            for variant in pool_group.slot_desc.variants
            for coalesced in variant.coalesced_buffers
        )
        return torch.empty(max_record_size, dtype=torch.uint8, device=self._device)

    def _record_view(self, record_size: int) -> torch.Tensor:
        if self._record_staging is None:
            # Only import-light CPU tests bypass construction and call pure
            # tensor helpers.  A real data hook must have CUDA staging.
            raise RuntimeError("NVFP4 boundary record staging is not initialized")
        if record_size > self._record_staging.numel():
            raise ValueError(
                "NVFP4 record exceeds the manager-lifetime staging capacity: "
                f"{record_size} > {self._record_staging.numel()}"
            )
        return self._record_staging[:record_size]

    def _validate_storage_layout(self) -> None:
        for pool_group in self.kv_cache_manager.impl.pool_group_descs:
            for variant in pool_group.slot_desc.variants:
                for coalesced in variant.coalesced_buffers:
                    self._record_sections(
                        coalesced.single_buffer_size,
                        coalesced.effective_host_single_buffer_size,
                    )
                    for buffer_id in coalesced.buffer_ids:
                        if str(buffer_id.role) not in _SUPPORTED_ROLES:
                            raise ValueError(
                                "NVFP4 Host boundary compression P0 supports "
                                f"only K/V buffers, got role={buffer_id.role!r}"
                            )

    def _variant(self, pool_group_index: int, life_cycle: int):
        variants = self.kv_cache_manager.impl.pool_group_descs[pool_group_index].slot_desc.variants
        for variant in variants:
            if int(variant.life_cycle_id) == int(life_cycle):
                return variant
        raise KeyError(f"No slot layout for pool_group={pool_group_index}, life_cycle={life_cycle}")

    def _buffer_matrix_shape(self, buffer_id, raw_size: int) -> tuple[int, int]:
        """Return a layout-neutral ``[physical rows, head_dim]`` view.

        NHD and HND both keep ``head_dim`` contiguous. Flattening every
        preceding dimension therefore preserves the exact physical bytes
        without asking which attention backend owns the Page layout.
        """
        layer_id = int(buffer_id.layer_id)
        num_heads = self.kv_cache_manager.num_kv_heads_per_layer[layer_id]
        head_dim = self.kv_cache_manager.head_dim_per_layer[layer_id]
        shape = (
            num_heads * self.kv_cache_manager.tokens_per_block,
            head_dim,
        )
        expected_size = math.prod(shape) * torch.tensor([], dtype=self._torch_dtype).element_size()
        if expected_size != raw_size:
            raise ValueError(
                "KVCM V2 Page geometry does not match its buffer size: "
                f"shape={shape}, expected_bytes={expected_size}, raw_size={raw_size}"
            )
        return shape

    def _raw_buffer_view(self, address, offset: int, shape) -> torch.Tensor:
        if not isinstance(address, int):
            raise TypeError("NVFP4 boundary transform requires a GPU memory address")
        raw = convert_to_torch_tensor(
            TensorWrapper(
                int(address) + offset,
                self._torch_dtype,
                shape,
            )
        )
        if raw.device != self._migration_device():
            raise RuntimeError(
                "KVCM V2 raw Page address is on a different CUDA device "
                f"({raw.device} != {self._migration_device()})"
            )
        return raw

    @staticmethod
    def compress_tensor(
        raw_payload: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse the existing NVFP4 quantization op for one physical Page.

        ``raw_payload`` is a layout-neutral ``[physical rows, head_dim]`` view.
        The first proof compresses every physical byte. Fresh runtime slots are
        cleared by KVCM V2, but a partial-reuse copy-on-write operation may
        copy stale, semantically invalid suffix bytes from an older Page. The
        runtime length still prevents Attention from reading that suffix, but
        it can enlarge this Page's global quantization scale. P0 deliberately
        records that accuracy risk instead of reading Attention metadata here.
        """
        if raw_payload.dim() != 2:
            raise ValueError("NVFP4 boundary compression expects [physical rows, head_dim]")
        if raw_payload.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 boundary compression expects FP16 or BF16 input")
        if not raw_payload.is_contiguous():
            raise ValueError("KVCM V2 must provide a contiguous stable payload lease")

        feature_count = raw_payload.shape[-1]
        if feature_count % NVFP4_BOUNDARY_BLOCK_SIZE != 0:
            raise ValueError(
                "NVFP4 feature width must be divisible by "
                f"{NVFP4_BOUNDARY_BLOCK_SIZE}, got {feature_count}"
            )
        if raw_payload.is_cuda:
            major, _ = torch.cuda.get_device_capability(raw_payload.device)
            if major < 10:
                raise RuntimeError("NVFP4 boundary compression requires SM100 or newer")

        quant_matrix = raw_payload
        # ``vector_norm(inf)`` performs the absolute-maximum reduction without
        # materializing a full-size ``abs()`` tensor.
        amax = torch.linalg.vector_norm(
            quant_matrix,
            ord=float("inf"),
            dtype=torch.float32,
        )
        # This scalar read is an intentional POC correctness gate. A failed
        # transform must be reported synchronously so StorageManager can roll
        # back the destination slot instead of publishing an invalid record.
        amax_value = float(amax.item())
        if not math.isfinite(amax_value):
            raise ValueError("NVFP4 boundary compression rejects non-finite KV")
        if amax_value > 0:
            ideal_inverse_scale = amax / _NVFP4_GLOBAL_SCALE_DENOMINATOR
            # BF16 can represent finite values whose ideal quantization
            # multiplier exceeds FP32. Clamp the multiplier by clamping its
            # reciprocal upward, and persist that *actual* reciprocal in the
            # record so decompression remains mathematically paired.
            # Do not clamp exactly to FLT_MAX.  Its rounded FP32 reciprocal is
            # slightly smaller than 1 / FLT_MAX, so taking the reciprocal again
            # can overflow to ``inf`` during validation or a future reader.
            # Half of FLT_MAX leaves one bit of exponent headroom while still
            # mapping BF16's smallest normal value into NVFP4's useful range.
            max_round_trip_safe_scale = torch.finfo(torch.float32).max / 2
            global_scale = torch.clamp_max(
                torch.reciprocal(ideal_inverse_scale),
                max_round_trip_safe_scale,
            )
            inverse_global_scale = torch.reciprocal(global_scale)
        else:
            inverse_global_scale = torch.ones_like(amax)
            global_scale = torch.ones_like(amax)

        # Keep the per-buffer scale contract explicitly one-dimensional.  The
        # existing operator accepts a scalar too, but the persisted inverse
        # scale is one FP32 element; matching those shapes prevents callers
        # from accidentally treating either value as a per-row scale.
        global_scale = global_scale.reshape(1)
        inverse_global_scale = inverse_global_scale.reshape(1)
        packed, block_scales = torch.ops.trtllm.fp4_quantize(
            quant_matrix,
            global_scale,
            NVFP4_BOUNDARY_BLOCK_SIZE,
            False,
            False,
        )
        row_count = quant_matrix.shape[0]
        expected_packed_shape = (row_count, feature_count // 2)
        expected_scale_elements = row_count * feature_count // NVFP4_BOUNDARY_BLOCK_SIZE
        if (
            tuple(packed.shape) != expected_packed_shape
            or packed.element_size() != 1
            or not packed.is_contiguous()
        ):
            raise RuntimeError(
                "fp4_quantize returned an incompatible packed layout: "
                f"shape={tuple(packed.shape)}, element_size={packed.element_size()}"
            )
        if (
            block_scales.numel() != expected_scale_elements
            or block_scales.element_size() != 1
            or not block_scales.is_contiguous()
        ):
            raise RuntimeError(
                "fp4_quantize returned an incompatible linear scale layout: "
                f"shape={tuple(block_scales.shape)}, "
                f"element_size={block_scales.element_size()}"
            )
        block_scales = block_scales.view(row_count, feature_count // NVFP4_BOUNDARY_BLOCK_SIZE)
        return packed, block_scales, inverse_global_scale

    @staticmethod
    def _copy_record_to_host(
        record: torch.Tensor,
        dst_address: int,
        num_bytes: int,
        stream: int,
    ) -> None:
        if record.dtype != torch.uint8 or not record.is_contiguous():
            raise TypeError("NVFP4 boundary record must be contiguous uint8")
        if record.numel() != num_bytes:
            raise ValueError(f"Record byte count mismatch: {record.numel()} != {num_bytes}")
        if num_bytes % NVFP4_BOUNDARY_RECORD_ALIGNMENT != 0:
            raise ValueError("KVCM V2 batched copy requires a 16-byte record")
        batched_copy(
            CacheTier.HOST_MEM,
            CacheTier.GPU_MEM,
            num_bytes,
            [
                CopyTask(
                    MemAddress(dst_address),
                    MemAddress(record.data_ptr()),
                )
            ],
            stream,
        )

    @staticmethod
    def _copy_host_to_record(
        src_address: int,
        record: torch.Tensor,
        num_bytes: int,
        stream: int,
    ) -> None:
        if record.dtype != torch.uint8 or not record.is_contiguous():
            raise TypeError("NVFP4 boundary record must be contiguous uint8")
        if record.numel() != num_bytes:
            raise ValueError(f"Record byte count mismatch: {record.numel()} != {num_bytes}")
        if num_bytes % NVFP4_BOUNDARY_RECORD_ALIGNMENT != 0:
            raise ValueError("KVCM V2 batched copy requires a 16-byte record")
        batched_copy(
            CacheTier.GPU_MEM,
            CacheTier.HOST_MEM,
            num_bytes,
            [
                CopyTask(
                    MemAddress(record.data_ptr()),
                    MemAddress(src_address),
                )
            ],
            stream,
        )

    def on_offload_compress(
        self,
        *,
        pool_group_index: int,
        src_pages: Sequence[Page],
        dst_slots: Sequence[Slot],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        src_slot_sizes: Sequence[int],
        dst_slot_sizes: Sequence[int],
        stream: int,
    ) -> None:
        """GPU→Host migration hook invoked by KVCM V2 ``StorageManager``."""
        if not (len(src_pages) == len(dst_slots) == len(src_addresses) == len(dst_addresses)):
            raise ValueError("Boundary migration batch fields have inconsistent lengths")

        try:
            # One manager-lifetime record buffer is shared by offload and
            # onboard. The lock serializes host submission; the CUDA event
            # serializes actual use when successive calls use different
            # migration streams. Page batches therefore do not multiply this
            # staging allocation under memory pressure.
            with self._record_staging_lock:
                device = self._migration_device()
                external_stream = torch.cuda.ExternalStream(stream, device=device)
                with torch.cuda.device(device), torch.cuda.stream(external_stream):
                    if self._record_staging_in_flight:
                        assert self._record_staging_ready is not None
                        external_stream.wait_event(self._record_staging_ready)
                    try:
                        for page, page_src, page_dst in zip(
                            src_pages,
                            src_addresses,
                            dst_addresses,
                        ):
                            variant = self._variant(pool_group_index, page.life_cycle)
                            if len(variant.coalesced_buffers) != len(page_src):
                                raise ValueError(
                                    "Pool count does not match the selected slot variant"
                                )
                            for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                                if src_slot_sizes[pool_index] != coalesced.size:
                                    raise ValueError("GPU slot size does not match its manifest")
                                if dst_slot_sizes[pool_index] != coalesced.host_size:
                                    raise ValueError("Host slot size does not match its manifest")
                                raw_offset = 0
                                host_offset = 0
                                for buffer_id in coalesced.buffer_ids:
                                    shape = self._buffer_matrix_shape(
                                        buffer_id,
                                        coalesced.single_buffer_size,
                                    )
                                    raw = self._raw_buffer_view(
                                        page_src[pool_index],
                                        raw_offset,
                                        shape,
                                    )
                                    packed, block_scales, inverse_scale = self.compress_tensor(raw)
                                    (
                                        packed_size,
                                        scale_size,
                                        inverse_scale_offset,
                                    ) = self._record_sections(
                                        coalesced.single_buffer_size,
                                        coalesced.effective_host_single_buffer_size,
                                    )
                                    dst_base = int(page_dst[pool_index]) + host_offset
                                    record_size = coalesced.effective_host_single_buffer_size
                                    # KVCM's copy engine accepts only 16-byte
                                    # grains. Build one self-contained record
                                    # containing packed values, per-block
                                    # scales, the per-buffer global scale, and
                                    # deterministic padding. Host never owns a
                                    # full-precision fallback.
                                    record = self._record_view(record_size)
                                    record.zero_()
                                    record[:packed_size].copy_(packed.view(torch.uint8).reshape(-1))
                                    record[packed_size : packed_size + scale_size].copy_(
                                        block_scales.view(torch.uint8).reshape(-1)
                                    )
                                    record[inverse_scale_offset : inverse_scale_offset + 4].view(
                                        torch.float32
                                    ).copy_(inverse_scale)
                                    self._copy_record_to_host(
                                        record,
                                        dst_base,
                                        record_size,
                                        stream,
                                    )
                                    # These tensors are dynamically allocated
                                    # by the existing fp4 op. Keep their
                                    # allocator ownership on this stream until
                                    # its record copy has consumed them.
                                    packed.record_stream(external_stream)
                                    block_scales.record_stream(external_stream)
                                    inverse_scale.record_stream(external_stream)
                                    del packed, block_scales, inverse_scale, record
                                    raw_offset += coalesced.single_buffer_size
                                    host_offset += coalesced.effective_host_single_buffer_size
                    finally:
                        assert self._record_staging_ready is not None
                        self._record_staging_ready.record(external_stream)
                        self._record_staging_in_flight = True
        except torch.OutOfMemoryError as error:
            # Scheduler admission handles cache pressure through this KVCM
            # exception. A raw torch OOM would bypass its retry/rollback path.
            raise OutOfPagesError("NVFP4 boundary compression workspace is unavailable") from error

    def on_onboard_decompress(
        self,
        *,
        pool_group_index: int,
        src_pages: Sequence[Page],
        dst_slots: Sequence[Slot],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        src_slot_sizes: Sequence[int],
        dst_slot_sizes: Sequence[int],
        stream: int,
    ) -> None:
        """Host→GPU migration hook invoked before KVCM V2 publishes the Page."""
        if not (len(src_pages) == len(dst_slots) == len(src_addresses) == len(dst_addresses)):
            raise ValueError("Boundary migration batch fields have inconsistent lengths")

        try:
            with self._record_staging_lock:
                device = self._migration_device()
                external_stream = torch.cuda.ExternalStream(stream, device=device)
                with torch.cuda.device(device), torch.cuda.stream(external_stream):
                    if self._record_staging_in_flight:
                        assert self._record_staging_ready is not None
                        external_stream.wait_event(self._record_staging_ready)
                    try:
                        for page, page_src, page_dst in zip(
                            src_pages,
                            src_addresses,
                            dst_addresses,
                        ):
                            variant = self._variant(pool_group_index, page.life_cycle)
                            if len(variant.coalesced_buffers) != len(page_src):
                                raise ValueError(
                                    "Pool count does not match the selected slot variant"
                                )
                            for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                                if src_slot_sizes[pool_index] != coalesced.host_size:
                                    raise ValueError("Host slot size does not match its manifest")
                                if dst_slot_sizes[pool_index] != coalesced.size:
                                    raise ValueError("GPU slot size does not match its manifest")
                                raw_offset = 0
                                host_offset = 0
                                for buffer_id in coalesced.buffer_ids:
                                    shape = self._buffer_matrix_shape(
                                        buffer_id,
                                        coalesced.single_buffer_size,
                                    )
                                    rows, feature_count = shape
                                    (
                                        packed_size,
                                        scale_size,
                                        inverse_scale_offset,
                                    ) = self._record_sections(
                                        coalesced.single_buffer_size,
                                        coalesced.effective_host_single_buffer_size,
                                    )
                                    record_size = coalesced.effective_host_single_buffer_size
                                    record = self._record_view(record_size)
                                    src_base = int(page_src[pool_index]) + host_offset
                                    self._copy_host_to_record(
                                        src_base,
                                        record,
                                        record_size,
                                        stream,
                                    )
                                    packed = record[:packed_size].view(
                                        rows,
                                        feature_count // 2,
                                    )
                                    block_scales = record[
                                        packed_size : packed_size + scale_size
                                    ].view(
                                        rows,
                                        feature_count // NVFP4_BOUNDARY_BLOCK_SIZE,
                                    )
                                    inverse_scale = record[
                                        inverse_scale_offset : inverse_scale_offset + 4
                                    ].view(torch.float32)
                                    raw = self._raw_buffer_view(
                                        page_dst[pool_index],
                                        raw_offset,
                                        shape,
                                    )
                                    dequant_nvfp4_2d_triton(
                                        packed,
                                        block_scales,
                                        inverse_scale,
                                        out=raw,
                                        target_dtype=self._torch_dtype,
                                    )
                                    del packed, block_scales, inverse_scale, record
                                    raw_offset += coalesced.single_buffer_size
                                    host_offset += coalesced.effective_host_single_buffer_size
                    finally:
                        assert self._record_staging_ready is not None
                        self._record_staging_ready.record(external_stream)
                        self._record_staging_in_flight = True
        except torch.OutOfMemoryError as error:
            raise OutOfPagesError(
                "NVFP4 boundary decompression workspace is unavailable"
            ) from error
