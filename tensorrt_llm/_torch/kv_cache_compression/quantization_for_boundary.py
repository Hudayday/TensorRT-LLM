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

import torch

from tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4 import (
    dequant_nvfp4_2d_triton,
)
from tensorrt_llm._utils import (
    TensorWrapper,
    binding_to_torch_dtype,
    convert_to_torch_tensor,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
    CacheTier,
    MemAddress,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import (
    CopyTask,
    batched_copy,
)

from ..pyexecutor.resource_manager import DataType, KVCacheCompressionManager

_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_SCALE_DENOMINATOR = 448.0 * 6.0
_RECORD_ALIGNMENT = 16
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
    GPU Page pre-admitted by KVCM V2. Temporary packed tensors are bounded by
    the migration batch and recorded on KVCM's migration stream.
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
        self._validate_storage_layout()
        kv_cache_manager.bind_boundary_compression_hooks(self)

    @staticmethod
    def _record_size(raw_size: int) -> int:
        if raw_size % 2 != 0:
            raise ValueError("FP16/BF16 raw buffer size must be divisible by 2")
        num_elements = raw_size // 2
        if num_elements % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 element count must be divisible by {_NVFP4_BLOCK_SIZE}"
            )
        unaligned = num_elements // 2 + num_elements // _NVFP4_BLOCK_SIZE + 4
        return ((unaligned + _RECORD_ALIGNMENT - 1) // _RECORD_ALIGNMENT) * (
            _RECORD_ALIGNMENT
        )

    @staticmethod
    def _record_sections(raw_size: int, host_size: int) -> tuple[int, int, int]:
        expected = QuantizationForBoundaryCompression._record_size(raw_size)
        if host_size != expected:
            raise ValueError(
                f"NVFP4 Host record size mismatch: expected {expected}, got {host_size}"
            )
        num_elements = raw_size // 2
        packed_size = num_elements // 2
        scale_size = num_elements // _NVFP4_BLOCK_SIZE
        inverse_scale_offset = packed_size + scale_size
        return packed_size, scale_size, inverse_scale_offset

    def _validate_storage_layout(self) -> None:
        for pool_group in self.kv_cache_manager.impl.pool_group_descs:
            for variant in pool_group.slot_desc.variants:
                for coalesced in variant.coalesced_buffers:
                    self._record_sections(
                        coalesced.single_buffer_size,
                        coalesced.host_single_buffer_size,
                    )
                    for buffer_id in coalesced.buffer_ids:
                        if str(buffer_id.role) not in _SUPPORTED_ROLES:
                            raise ValueError(
                                "NVFP4 Host boundary compression P0 supports "
                                f"only K/V buffers, got role={buffer_id.role!r}"
                            )

    def _variant(self, pool_group_index: int, life_cycle: int):
        variants = self.kv_cache_manager.impl.pool_group_descs[
            pool_group_index
        ].slot_desc.variants
        for variant in variants:
            if int(variant.life_cycle_id) == int(life_cycle):
                return variant
        raise KeyError(
            f"No slot layout for pool_group={pool_group_index}, life_cycle={life_cycle}"
        )

    def _buffer_shape(self, buffer_id, raw_size: int) -> tuple[int, int, int]:
        layer_id = int(buffer_id.layer_id)
        num_heads = self.kv_cache_manager.num_kv_heads_per_layer[layer_id]
        head_dim = self.kv_cache_manager.head_dim_per_layer[layer_id]
        shape = (num_heads, self.kv_cache_manager.tokens_per_block, head_dim)
        expected_size = math.prod(shape) * torch.tensor(
            [], dtype=self._torch_dtype
        ).element_size()
        if expected_size != raw_size:
            raise ValueError(
                "KVCM V2 HND Page geometry does not match its buffer size: "
                f"shape={shape}, expected_bytes={expected_size}, raw_size={raw_size}"
            )
        return shape

    def _raw_buffer_view(self, address, offset: int, shape) -> torch.Tensor:
        if not isinstance(address, int):
            raise TypeError("NVFP4 boundary transform requires a GPU memory address")
        return convert_to_torch_tensor(
            TensorWrapper(
                int(address) + offset,
                self._torch_dtype,
                shape,
            )
        )

    @staticmethod
    def compress_tensor(
        raw_payload: torch.Tensor,
        *,
        valid_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse the existing NVFP4 quantization op for one normalized Page.

        The storage adapter may expose NHD ``[tokens, features]`` or HND
        ``[heads, tokens, head_dim]``. Its logical valid-token count comes from
        KVCM-owned Page metadata, never Attention or AttentionMetadata.
        A partial Page's unused token region is zeroed before scale calculation
        so stale slot contents cannot change the scale or leak into Host.
        """
        if raw_payload.dim() not in (2, 3):
            raise ValueError(
                "NVFP4 boundary compression expects NHD [tokens, features] "
                "or HND [heads, tokens, head_dim]"
            )
        if raw_payload.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 boundary compression expects FP16 or BF16 input")
        if not raw_payload.is_contiguous():
            raise ValueError("KVCM V2 must provide a contiguous stable payload lease")
        token_axis = 0 if raw_payload.dim() == 2 else 1
        physical_token_count = raw_payload.shape[token_axis]
        if valid_token_count <= 0 or valid_token_count > physical_token_count:
            raise ValueError(
                "valid_token_count must be in [1, physical token rows], "
                f"got {valid_token_count} for {physical_token_count} rows"
            )

        feature_count = raw_payload.shape[-1]
        if feature_count % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 feature width must be divisible by {_NVFP4_BLOCK_SIZE}, got {feature_count}"
            )
        if raw_payload.is_cuda:
            major, _ = torch.cuda.get_device_capability(raw_payload.device)
            if major < 10:
                raise RuntimeError("NVFP4 boundary compression requires SM100 or newer")

        quant_input = raw_payload
        if valid_token_count < physical_token_count:
            quant_input = raw_payload.clone()
            if quant_input.dim() == 2:
                quant_input[valid_token_count:].zero_()
            else:
                quant_input[:, valid_token_count:, :].zero_()

        quant_matrix = quant_input.view(-1, feature_count)
        amax = quant_matrix.abs().amax().to(torch.float32)
        usable_amax = torch.isfinite(amax) & (amax > 0)
        inverse_global_scale = torch.where(
            usable_amax,
            amax / _NVFP4_GLOBAL_SCALE_DENOMINATOR,
            torch.ones_like(amax),
        )
        global_scale = torch.reciprocal(inverse_global_scale)

        packed, block_scales = torch.ops.trtllm.fp4_quantize(
            quant_matrix,
            global_scale,
            _NVFP4_BLOCK_SIZE,
            False,
            False,
        )
        row_count = quant_matrix.shape[0]
        block_scales = block_scales.view(
            row_count, feature_count // _NVFP4_BLOCK_SIZE
        )
        return packed, block_scales, inverse_global_scale.reshape(1)

    @staticmethod
    def _copy_tensor_to_host(
        tensor: torch.Tensor,
        dst_address: int,
        num_bytes: int,
        stream: int,
    ) -> None:
        if tensor.numel() * tensor.element_size() != num_bytes:
            raise ValueError(
                f"Tensor byte count mismatch: {tensor.numel() * tensor.element_size()} "
                f"!= {num_bytes}"
            )
        batched_copy(
            CacheTier.HOST_MEM,
            CacheTier.GPU_MEM,
            num_bytes,
            [
                CopyTask(
                    MemAddress(dst_address),
                    MemAddress(tensor.data_ptr()),
                )
            ],
            stream,
        )

    @staticmethod
    def _copy_host_to_tensor(
        src_address: int,
        tensor: torch.Tensor,
        num_bytes: int,
        stream: int,
    ) -> None:
        if tensor.numel() * tensor.element_size() != num_bytes:
            raise ValueError(
                f"Tensor byte count mismatch: {tensor.numel() * tensor.element_size()} "
                f"!= {num_bytes}"
            )
        batched_copy(
            CacheTier.GPU_MEM,
            CacheTier.HOST_MEM,
            num_bytes,
            [
                CopyTask(
                    MemAddress(tensor.data_ptr()),
                    MemAddress(src_address),
                )
            ],
            stream,
        )

    def on_offload_compress(self, **kwargs) -> None:
        """GPU→Host migration hook invoked by KVCM V2 ``StorageManager``."""
        pool_group_index = int(kwargs["pool_group_index"])
        src_pages = kwargs["src_pages"]
        src_addresses = kwargs["src_addresses"]
        dst_addresses = kwargs["dst_addresses"]
        src_slot_sizes = kwargs["src_slot_sizes"]
        dst_slot_sizes = kwargs["dst_slot_sizes"]
        valid_token_counts = kwargs["valid_token_counts"]
        stream = int(kwargs["stream"])

        if not (
            len(src_pages)
            == len(src_addresses)
            == len(dst_addresses)
            == len(valid_token_counts)
        ):
            raise ValueError("Boundary migration batch fields have inconsistent lengths")

        external_stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(external_stream):
            for page, page_src, page_dst, valid_token_count in zip(
                src_pages,
                src_addresses,
                dst_addresses,
                valid_token_counts,
            ):
                variant = self._variant(pool_group_index, page.life_cycle)
                if len(variant.coalesced_buffers) != len(page_src):
                    raise ValueError("Pool count does not match the selected slot variant")
                for pool_index, coalesced in enumerate(
                    variant.coalesced_buffers
                ):
                    if src_slot_sizes[pool_index] != coalesced.size:
                        raise ValueError("GPU slot size does not match its manifest")
                    if dst_slot_sizes[pool_index] != coalesced.host_size:
                        raise ValueError("Host slot size does not match its manifest")
                    raw_offset = 0
                    host_offset = 0
                    for buffer_id in coalesced.buffer_ids:
                        shape = self._buffer_shape(
                            buffer_id, coalesced.single_buffer_size
                        )
                        raw = self._raw_buffer_view(
                            page_src[pool_index], raw_offset, shape
                        )
                        packed, block_scales, inverse_scale = self.compress_tensor(
                            raw,
                            valid_token_count=int(valid_token_count),
                        )
                        (
                            packed_size,
                            scale_size,
                            inverse_scale_offset,
                        ) = self._record_sections(
                            coalesced.single_buffer_size,
                            coalesced.host_single_buffer_size,
                        )
                        dst_base = int(page_dst[pool_index]) + host_offset
                        self._copy_tensor_to_host(
                            packed, dst_base, packed_size, stream
                        )
                        self._copy_tensor_to_host(
                            block_scales,
                            dst_base + packed_size,
                            scale_size,
                            stream,
                        )
                        self._copy_tensor_to_host(
                            inverse_scale,
                            dst_base + inverse_scale_offset,
                            4,
                            stream,
                        )
                        packed.record_stream(external_stream)
                        block_scales.record_stream(external_stream)
                        inverse_scale.record_stream(external_stream)
                        raw_offset += coalesced.single_buffer_size
                        host_offset += coalesced.host_single_buffer_size

    def on_onboard_decompress(self, **kwargs) -> None:
        """Host→GPU migration hook invoked before KVCM V2 publishes the Page."""
        pool_group_index = int(kwargs["pool_group_index"])
        src_pages = kwargs["src_pages"]
        src_addresses = kwargs["src_addresses"]
        dst_addresses = kwargs["dst_addresses"]
        src_slot_sizes = kwargs["src_slot_sizes"]
        dst_slot_sizes = kwargs["dst_slot_sizes"]
        stream = int(kwargs["stream"])

        if not (
            len(src_pages) == len(src_addresses) == len(dst_addresses)
        ):
            raise ValueError("Boundary migration batch fields have inconsistent lengths")

        device = torch.device("cuda", torch.cuda.current_device())
        external_stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(external_stream):
            for page, page_src, page_dst in zip(
                src_pages, src_addresses, dst_addresses
            ):
                variant = self._variant(pool_group_index, page.life_cycle)
                if len(variant.coalesced_buffers) != len(page_src):
                    raise ValueError("Pool count does not match the selected slot variant")
                for pool_index, coalesced in enumerate(
                    variant.coalesced_buffers
                ):
                    if src_slot_sizes[pool_index] != coalesced.host_size:
                        raise ValueError("Host slot size does not match its manifest")
                    if dst_slot_sizes[pool_index] != coalesced.size:
                        raise ValueError("GPU slot size does not match its manifest")
                    raw_offset = 0
                    host_offset = 0
                    for buffer_id in coalesced.buffer_ids:
                        shape = self._buffer_shape(
                            buffer_id, coalesced.single_buffer_size
                        )
                        rows = shape[0] * shape[1]
                        feature_count = shape[2]
                        (
                            packed_size,
                            scale_size,
                            inverse_scale_offset,
                        ) = self._record_sections(
                            coalesced.single_buffer_size,
                            coalesced.host_single_buffer_size,
                        )
                        packed = torch.empty(
                            (rows, feature_count // 2),
                            dtype=torch.uint8,
                            device=device,
                        )
                        block_scales = torch.empty(
                            (rows, feature_count // _NVFP4_BLOCK_SIZE),
                            dtype=torch.uint8,
                            device=device,
                        )
                        inverse_scale = torch.empty(
                            (1,), dtype=torch.float32, device=device
                        )
                        src_base = int(page_src[pool_index]) + host_offset
                        self._copy_host_to_tensor(
                            src_base, packed, packed_size, stream
                        )
                        self._copy_host_to_tensor(
                            src_base + packed_size,
                            block_scales,
                            scale_size,
                            stream,
                        )
                        self._copy_host_to_tensor(
                            src_base + inverse_scale_offset,
                            inverse_scale,
                            4,
                            stream,
                        )
                        raw = self._raw_buffer_view(
                            page_dst[pool_index], raw_offset, shape
                        )
                        dequant_nvfp4_2d_triton(
                            packed,
                            block_scales,
                            inverse_scale,
                            out=raw.view(rows, feature_count),
                            target_dtype=self._torch_dtype,
                        )
                        packed.record_stream(external_stream)
                        block_scales.record_stream(external_stream)
                        inverse_scale.record_stream(external_stream)
                        raw_offset += coalesced.single_buffer_size
                        host_offset += coalesced.host_single_buffer_size
