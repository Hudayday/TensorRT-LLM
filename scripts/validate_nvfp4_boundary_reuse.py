# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the reuse-boundary NVFP4 transform on an SM100-or-newer GPU.

This is deliberately a transform-only probe. It instantiates the real
``QuantizationForBoundaryCompression`` against a minimal ``KVCacheManagerV2``
harness, but it does not exercise the V2 page allocator, a compact backing
pool, reuse commit/resume callers, or physical raw-slot release.
"""

import argparse
import json
from pathlib import Path

import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType

DEFAULT_SHAPE = (2, 32, 128)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class _KVCacheManagerV2Harness(KVCacheManagerV2):
    """Only the explicit manager-binding contract needed by this GPU probe.

    The harness is a real V2 subtype so the production manager's type guard is
    exercised, but it deliberately bypasses the allocator-heavy V2
    constructor. KVCM transaction methods have separate CPU contract tests.
    """

    def __init__(self, tokens_per_block: int) -> None:
        self.enable_block_reuse = True
        self.kv_compression_manages_history = False
        self.is_draft = False
        self.is_disagg = False
        self.dtype = DataType.BF16
        self.tokens_per_block = tokens_per_block
        self._reuse_compression_manager = None
        self._reuse_backing_store = None

    def bind_reuse_compression_hooks(
        self,
        manager: QuantizationForBoundaryCompression,
    ) -> None:
        if self._reuse_compression_manager is not None:
            raise RuntimeError("A reuse compression manager is already bound")
        self._reuse_compression_manager = manager


def _make_kvcm_v2_harness(tokens_per_block: int) -> _KVCacheManagerV2Harness:
    return _KVCacheManagerV2Harness(tokens_per_block)


def _make_input(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Create a deterministic, finite BF16 tensor without RNG state."""
    element_count = shape[0] * shape[1] * shape[2]
    raw = torch.linspace(
        -4.0,
        4.0,
        steps=element_count,
        dtype=torch.float32,
        device=device,
    )
    return raw.to(torch.bfloat16).reshape(shape).contiguous()


def _encode(
    manager: QuantizationForBoundaryCompression,
    raw: torch.Tensor,
    valid_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compressed = manager.on_reuse_store(raw, valid_token_count=valid_token_count)
    if compressed is None:
        raise RuntimeError("A full, feature-aligned Page was not admitted to compressed reuse")
    return compressed


def _decompress(
    manager: QuantizationForBoundaryCompression,
    compressed: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    destination: torch.Tensor,
) -> None:
    result = manager.on_reuse_materialize(compressed, destination)
    if result is not None:
        raise RuntimeError("on_reuse_materialize must write in place and return None")


def _validate(
    shape: tuple[int, int, int],
    device: torch.device,
) -> dict[str, object]:
    if device.type != "cuda":
        raise ValueError(f"NVFP4 validation requires a CUDA device, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        raise RuntimeError(
            f"NVFP4 boundary validation requires SM100 or newer, got sm_{major}{minor}"
        )

    kv_cache_manager = _make_kvcm_v2_harness(tokens_per_block=shape[-2])
    manager = QuantizationForBoundaryCompression(
        kv_cache_manager,
        quant="nvfp4",
        compressed_capacity_bytes=1 << 20,
    )
    raw = _make_input(shape, device)

    # One invocation is one independently evictable physical role payload.
    # Do not share the FP32 global scale across K/V roles.
    role_payloads = tuple(component.contiguous() for component in raw.unbind(0))
    compressed_records = tuple(
        _encode(manager, payload, valid_token_count=shape[-2]) for payload in role_payloads
    )

    destination = torch.full_like(raw, torch.nan)
    destination_pointer = destination.data_ptr()
    for compressed, role_destination in zip(
        compressed_records, destination.unbind(0), strict=True
    ):
        _decompress(manager, compressed, role_destination)
    torch.cuda.synchronize(device)

    decode_into_destination = destination.data_ptr() == destination_pointer and not bool(
        torch.isnan(destination).any().item()
    )
    if not decode_into_destination:
        raise RuntimeError("NVFP4 decode did not populate the supplied raw destination")

    partial_not_admitted = all(
        manager.on_reuse_store(
            payload,
            valid_token_count=shape[-2] - 1,
        )
        is None
        for payload in role_payloads
    )
    if not partial_not_admitted:
        raise RuntimeError("A partial reuse Page was admitted to compressed cold reuse")

    zero_raw = torch.zeros_like(raw)
    zero_records = tuple(
        _encode(manager, payload.contiguous(), valid_token_count=shape[-2])
        for payload in zero_raw.unbind(0)
    )
    zero_destination = torch.full_like(zero_raw, 1.0)
    for compressed, role_destination in zip(
        zero_records, zero_destination.unbind(0), strict=True
    ):
        _decompress(manager, compressed, role_destination)
    torch.cuda.synchronize(device)
    zero_roundtrip = bool(torch.count_nonzero(zero_destination).item() == 0)
    if not zero_roundtrip:
        raise RuntimeError("The all-zero NVFP4 round trip was not exactly zero")

    difference = destination.float() - raw.float()
    relative_l2 = (
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(raw.float())
    ).item()
    mean_absolute_error = difference.abs().mean().item()

    component_bytes = {
        "block_scales": sum(
            _tensor_bytes(block_scales) for _, block_scales, _ in compressed_records
        ),
        "inverse_global_scale": sum(
            _tensor_bytes(inverse_global_scale)
            for _, _, inverse_global_scale in compressed_records
        ),
        "packed": sum(_tensor_bytes(packed) for packed, _, _ in compressed_records),
    }
    raw_bytes = _tensor_bytes(raw)
    compressed_bytes = sum(component_bytes.values())

    return {
        "status": "PASS",
        "proof_scope": "nvfp4_transform_with_minimal_kvcm_v2_harness_only",
        "device": {
            "capability": f"sm_{major}{minor}",
            "index": device.index if device.index is not None else torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(device),
        },
        "input": {
            "dtype": str(raw.dtype),
            "shape": list(shape),
        },
        "payload": {
            "compressed_bytes": compressed_bytes,
            "compressed_component_bytes": component_bytes,
            "compressed_record_count": len(compressed_records),
            "global_scale_scope": "one_fp32_scalar_per_role_payload",
            "raw_bytes": raw_bytes,
            "raw_to_compressed_ratio": round(raw_bytes / compressed_bytes, 12),
        },
        "accuracy": {
            "mean_absolute_error": round(mean_absolute_error, 12),
            "relative_l2": round(relative_l2, 12),
        },
        "checks": {
            "decode_writes_into_supplied_destination": decode_into_destination,
            "partial_page_not_admitted_to_cold_reuse": partial_not_admitted,
            "zero_roundtrip": zero_roundtrip,
        },
        "lifecycle_wired": False,
        "physical_kvcm_slot_release": False,
        "unified_pool_capacity": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for the SM100+ probe (default: cuda:0).",
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=DEFAULT_SHAPE,
        metavar=("KV", "TOKENS", "FEATURES"),
        help="Raw BF16 payload shape (default: 2 32 128).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the deterministic JSON result.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = tuple(args.shape)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Every shape dimension must be positive, got {shape}")

    result = _validate(shape, torch.device(args.device))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
