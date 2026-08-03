# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from typing import Optional

from tensorrt_llm.bindings import DataType

NVFP4_BOUNDARY_BLOCK_SIZE = 16
NVFP4_BOUNDARY_RECORD_ALIGNMENT = 16
NVFP4_BOUNDARY_DIRECT_RUNTIME_DTYPES = frozenset((DataType.HALF, DataType.BF16, DataType.FP8))


def nvfp4_boundary_supports_runtime_dtype(dtype: DataType) -> bool:
    """Whether this build contract has a direct two-way NVFP4 conversion.

    This is an admission capability, not a Page-format declaration.  The
    runtime GPU Slot keeps the dtype selected by KVCM; boundary compression
    merely chooses a direct transform for that existing representation.
    """
    return dtype in NVFP4_BOUNDARY_DIRECT_RUNTIME_DTYPES


def nvfp4_boundary_record_layout(num_elements: int) -> tuple[int, int, int, int]:
    """Return the fixed Host-record layout for one logical KV buffer.

    The returned values are ``(packed_size, scale_size,
    inverse_scale_offset, record_size)`` in bytes.  Keeping this arithmetic in
    one import-light helper makes the KVCM V2 Host allocator and the NVFP4
    transform consume exactly the same physical layout.  The input is an
    element count rather than source bytes because FP16, BF16, and FP8 runtime
    Slots have different byte widths but produce the same NVFP4 record for the
    same physical Page shape.
    """
    if num_elements <= 0:
        raise ValueError("NVFP4 element count must be positive")
    if num_elements % NVFP4_BOUNDARY_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 element count must be divisible by {NVFP4_BOUNDARY_BLOCK_SIZE}")

    packed_size = num_elements // 2
    scale_size = num_elements // NVFP4_BOUNDARY_BLOCK_SIZE
    inverse_scale_offset = (packed_size + scale_size + 3) // 4 * 4
    record_size = (
        (inverse_scale_offset + 4 + NVFP4_BOUNDARY_RECORD_ALIGNMENT - 1)
        // NVFP4_BOUNDARY_RECORD_ALIGNMENT
        * NVFP4_BOUNDARY_RECORD_ALIGNMENT
    )
    return packed_size, scale_size, inverse_scale_offset, record_size


class KvCacheCompressionMode(IntEnum):
    """Algorithm-level traits of a KV-cache compression method.

    Configs map their ``algorithm`` string to a member here; callers read the
    ``is_*`` predicates instead of comparing strings.
    """

    NONE = auto()
    QUANTIZATION_FOR_BOUNDARY = auto()

    def is_eviction_method(self):
        """Whether this method physically evicts cached tokens. Evicting
        algorithms add their member and extend this predicate."""
        return False

    @staticmethod
    def from_string(name: Optional[str]) -> "KvCacheCompressionMode":
        if name is None:
            return KvCacheCompressionMode.NONE
        try:
            return KvCacheCompressionMode[name.upper()]
        except KeyError:
            return KvCacheCompressionMode.NONE
