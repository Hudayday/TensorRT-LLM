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

from typing import Optional

import torch

from ..pyexecutor.resource_manager import BaseKVCacheCompressionManager, DataType

_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_SCALE_DENOMINATOR = 448.0 * 6.0


class QuantizationForBoundaryCompression(BaseKVCacheCompressionManager):
    """Quantize cold reusable pages while keeping active runtime KV raw.

    This proof supports one boundary and one quantization:

    ``full committed raw reuse page -> NVFP4 backing -> raw reuse hit``.

    KVCM V2 remains the lifecycle and storage owner. Production wiring must
    supply each stable contiguous physical payload (one layer and K/V role) to
    :meth:`on_reuse_store`, aggregate all returned tensors for a committed page
    into one atomic backing, and supply pre-admitted raw destinations to
    :meth:`on_reuse_materialize`. This manager neither traverses KVCM internals
    nor owns eviction, tier migration, or page publication.

    The proof deliberately falls back to raw for partial pages. Production
    wiring still requires a compact representation pool and atomic
    RAW/DUAL/ENCODED transitions in KVCM V2; storing these tensors in a raw
    slot would not increase reuse capacity.
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
            raise ValueError("NVFP4 boundary reuse does not support a draft KV cache")
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        if quant != "nvfp4":
            raise ValueError("QuantizationForBoundaryCompression only supports quant='nvfp4'")
        if not kv_cache_manager.enable_block_reuse:
            raise ValueError("NVFP4 boundary reuse requires KV-cache block reuse")
        if kv_cache_manager.is_draft:
            raise ValueError("NVFP4 boundary reuse must bind to the target KV cache")
        if kv_cache_manager.is_disagg:
            raise ValueError("The reuse-only NVFP4 proof does not support disaggregation")
        if kv_cache_manager.dtype not in (DataType.HALF, DataType.BF16):
            raise ValueError("Active runtime KV must remain FP16 or BF16 for boundary reuse")

        self.quant = quant
        kv_cache_manager.bind_reuse_compression_hooks(self)

    def on_reuse_store(
        self,
        raw_payload: torch.Tensor,
        *,
        valid_token_count: Optional[int] = None,
        **kwargs,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode one full, stable ``[..., features]`` raw payload.

        The tuple is packed E2M1 data, linear per-16-value FP8 scales, and one
        inverse global FP32 scale. Every invocation is independently decodable.
        The global scale is local to one committed page x layer x K/V role;
        it is never shared across an eviction unit, layer, or role.

        ``valid_token_count`` is lifecycle information supplied by KVCM V2;
        the manager never infers a token axis or reads attention layout
        metadata. ``None`` requests KVCM V2's raw fallback. The proof uses that
        path for partial pages and feature widths that do not satisfy NVFP4
        alignment.
        """
        del kwargs
        if raw_payload.dim() < 2:
            raise ValueError("NVFP4 boundary reuse expects [..., features]")
        if raw_payload.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 boundary reuse expects FP16 or BF16 input")
        if not raw_payload.is_contiguous():
            raise ValueError("KVCM V2 must provide a contiguous stable raw payload lease")

        feature_count = raw_payload.shape[-1]
        if valid_token_count is None:
            raise ValueError("KVCM V2 must provide valid_token_count for reuse storage")
        if valid_token_count != self.kv_cache_manager.tokens_per_block:
            return None
        if feature_count % _NVFP4_BLOCK_SIZE != 0:
            return None
        if raw_payload.is_cuda:
            major, _ = torch.cuda.get_device_capability(raw_payload.device)
            if major < 10:
                raise RuntimeError("The NVFP4 boundary reuse proof requires SM100 or newer")

        amax = raw_payload.float().abs().max()
        if not bool(torch.isfinite(amax).item()):
            return None
        if float(amax.item()) == 0.0:
            global_scale = torch.ones((), dtype=torch.float32, device=raw_payload.device)
            inverse_global_scale = global_scale.clone()
        else:
            global_scale = _NVFP4_GLOBAL_SCALE_DENOMINATOR / amax
            inverse_global_scale = (amax / _NVFP4_GLOBAL_SCALE_DENOMINATOR).to(torch.float32)

        packed, block_scales = torch.ops.trtllm.fp4_quantize(
            raw_payload,
            global_scale,
            _NVFP4_BLOCK_SIZE,
            False,
            False,
        )
        row_count = raw_payload.numel() // feature_count
        block_scales = block_scales.view(row_count, feature_count // _NVFP4_BLOCK_SIZE)
        return packed, block_scales, inverse_global_scale.reshape(1)

    def on_reuse_materialize(
        self,
        encoded_payload: object,
        raw_destination: torch.Tensor,
        **kwargs,
    ) -> None:
        """Decode one NVFP4 backing into a KVCM-owned raw runtime slot."""
        del kwargs
        if not isinstance(encoded_payload, tuple) or len(encoded_payload) != 3:
            raise TypeError("NVFP4 reuse backing must be (packed, block_scales, global_scale)")
        packed, block_scales, inverse_global_scale = encoded_payload
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (packed, block_scales, inverse_global_scale)
        ):
            raise TypeError("Every NVFP4 reuse backing component must be a tensor")
        expected_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
        if tuple(raw_destination.shape) != expected_shape:
            raise ValueError(
                f"raw destination shape {tuple(raw_destination.shape)} does "
                f"not match encoded page shape {expected_shape}"
            )
        if raw_destination.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 materialization requires an FP16/BF16 slot")
        if not raw_destination.is_contiguous():
            raise ValueError("KVCM V2 must provide a contiguous raw materialization slot")

        from tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4 import (
            dequant_nvfp4_2d_triton,
        )

        feature_count = expected_shape[-1]
        dequant_nvfp4_2d_triton(
            packed.view(-1, feature_count // 2),
            block_scales,
            inverse_global_scale,
            target_dtype=raw_destination.dtype,
            sf_vec_size=_NVFP4_BLOCK_SIZE,
            out=raw_destination.view(-1, feature_count),
        )
