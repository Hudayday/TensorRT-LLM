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

import torch

from ..pyexecutor.resource_manager import DataType, KVCacheCompressionManager

_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_SCALE_DENOMINATOR = 448.0 * 6.0


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

    The lifecycle callbacks are intentionally fail-closed in this scaffold.
    KVCM V2 still has one slot schema shared by GPU and Host, so a compact Host
    record cannot yet be allocated. TensorRT-LLM main also has no production
    standalone NVFP4 Page-to-FP16/BF16 restore operation that writes directly
    into a pre-admitted KVCM Page. Both gaps must be closed before enabling
    this algorithm in serving.
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

        self.quant = quant
        kv_cache_manager.bind_boundary_compression_hooks(self)

    @staticmethod
    def compress_tensor(
        raw_payload: torch.Tensor,
        *,
        valid_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse the existing NVFP4 quantization op for one normalized Page.

        The storage adapter must expose the Page as ``[tokens, features]`` and
        provide its logical valid-token count without consulting Attention or
        AttentionMetadata. The fixed-size record includes the whole physical
        Page, but a partial Page's unused rows are zeroed before scale
        calculation so stale slot contents cannot change the scale or leak
        into the compressed backing.
        """
        if raw_payload.dim() != 2:
            raise ValueError("NVFP4 boundary compression expects [tokens, features]")
        if raw_payload.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 boundary compression expects FP16 or BF16 input")
        if not raw_payload.is_contiguous():
            raise ValueError("KVCM V2 must provide a contiguous stable payload lease")
        if valid_token_count <= 0 or valid_token_count > raw_payload.shape[0]:
            raise ValueError(
                "valid_token_count must be in [1, physical token rows], "
                f"got {valid_token_count} for {raw_payload.shape[0]} rows"
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
        if valid_token_count < raw_payload.shape[0]:
            quant_input = raw_payload.clone()
            quant_input[valid_token_count:].zero_()

        amax = quant_input.float().abs().max()
        if not bool(torch.isfinite(amax).item()):
            raise ValueError("NVFP4 boundary compression requires finite input")
        if float(amax.item()) == 0.0:
            global_scale = torch.ones((), dtype=torch.float32, device=raw_payload.device)
            inverse_global_scale = global_scale.clone()
        else:
            global_scale = _NVFP4_GLOBAL_SCALE_DENOMINATOR / amax
            inverse_global_scale = (amax / _NVFP4_GLOBAL_SCALE_DENOMINATOR).to(torch.float32)

        packed, block_scales = torch.ops.trtllm.fp4_quantize(
            quant_input,
            global_scale,
            _NVFP4_BLOCK_SIZE,
            False,
            False,
        )
        row_count = raw_payload.numel() // feature_count
        block_scales = block_scales.view(row_count, feature_count // _NVFP4_BLOCK_SIZE)
        return packed, block_scales, inverse_global_scale.reshape(1)

    def on_offload_compress(self, **kwargs) -> None:
        """GPU→Host migration hook invoked by KVCM V2 ``StorageManager``."""
        del kwargs
        raise NotImplementedError(
            "NVFP4 Host offload needs tier-specific compact Host slots and a "
            "GPU-Page-to-Host-record adapter before it can replace the raw copy"
        )

    def on_onboard_decompress(self, **kwargs) -> None:
        """Host→GPU migration hook invoked before KVCM V2 publishes the Page."""
        del kwargs
        raise NotImplementedError(
            "NVFP4 Host recall needs a standalone SM100 Page restore operation "
            "that writes into the pre-admitted FP16/BF16 KVCM V2 destination"
        )
