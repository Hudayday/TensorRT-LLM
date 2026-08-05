# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization compression at the two storage migration hooks.

This module intentionally contains one concrete compression manager and two
small immutable layout records.  It does not own Pages, Slots, streams,
eviction, publication, or rollback; those remain StorageManager duties.

The current KVCM V2 source does not yet expose different layouts per cache
level.  Consequently the prototype receives the authoritative layout records
explicitly.  Yao Yao's KVCM change can later provide the same records at
manager construction without changing the two Hook implementations below.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence, Tuple

from ..pyexecutor.resource_manager import KVCacheCompressionManager


@dataclass(frozen=True)
class BoundaryBufferLayout:
    """Location of one logical buffer inside a physical PoolGroup Slot.

    ``pool_index`` selects one base address in the Slot address row supplied by
    KVCM. ``offset`` selects the logical K, V, or scale buffer coalesced into
    that physical Pool.  This is the same information represented by KVCM V2's
    ``BufferAttr``; the manager never recomputes Pool coalescing.
    """

    pool_index: int
    offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.pool_index, int) or self.pool_index < 0:
            raise ValueError("pool_index must be non-negative")
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("buffer offset must be non-negative")

    def resolve(self, slot_pool_addresses: Sequence[int]) -> int:
        """Return ``Pool base + BufferAttr offset`` for one allocated Slot."""

        if self.pool_index >= len(slot_pool_addresses):
            raise ValueError(
                f"Pool row has {len(slot_pool_addresses)} addresses but "
                f"layout selects pool {self.pool_index}")
        base = slot_pool_addresses[self.pool_index]
        if not isinstance(base, int) or base <= 0:
            raise ValueError("Pool base addresses must be positive integers")
        return base + self.offset


ScalePair = Tuple[float, float]
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINTPTR_MAX = (1 << 64) - 1
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_FLOAT32_MIN_SUBNORMAL = float.fromhex("0x1p-149")


@dataclass(frozen=True)
class Nvfp4BoundaryLayerLayout:
    """Immutable layout, geometry, and calibration for one local KV layer.

    One KVCM Page may contain several layers.  Therefore the manager produces
    one native task for each ``(Page, layer)`` pair, not one task for the whole
    Page.  ``runtime_*`` locations belong to the active GPU layout;
    ``packed_*`` and ``block_scale_*`` belong to the compact Host layout.

    Scale pairs are ordered ``(K, V)``.  ``*_orig_quant`` converts values from
    the model/original domain to a quantized domain; ``*_quant_orig`` performs
    the inverse conversion.  They are manager-lifetime calibration, not bytes
    stored in a Page.
    """

    pool_group_index: int
    life_cycle_id: int
    layer_id: int

    runtime_k: BoundaryBufferLayout
    runtime_v: BoundaryBufferLayout
    packed_k: BoundaryBufferLayout
    packed_v: BoundaryBufferLayout
    block_scale_k: BoundaryBufferLayout
    block_scale_v: BoundaryBufferLayout

    num_kv_heads: int
    tokens_per_page: int
    head_dim: int
    runtime_dtype: str

    nvfp4_scale_orig_quant: ScalePair
    nvfp4_scale_quant_orig: ScalePair
    fp8_scale_orig_quant: Optional[ScalePair] = None
    fp8_scale_quant_orig: Optional[ScalePair] = None

    def __post_init__(self) -> None:
        ids = (self.pool_group_index, self.life_cycle_id, self.layer_id)
        if any(not isinstance(value, int) or value < 0 for value in ids):
            raise ValueError(
                "Pool-group, life-cycle, and layer ids must be non-negative")
        geometry = (self.num_kv_heads, self.tokens_per_page, self.head_dim)
        if any(not isinstance(value, int) or not 0 < value <= _INT32_MAX
               for value in geometry):
            raise ValueError("Page geometry must contain positive int32 values")
        if self.tokens_per_page % 4 != 0:
            raise ValueError("tokens_per_page must be divisible by 4")
        if self.head_dim % 16 != 0:
            raise ValueError("head_dim must be divisible by 16")
        half_groups = self.num_kv_heads * self.tokens_per_page * (
            self.head_dim // 8)
        if half_groups > _UINT32_MAX // 8:
            raise ValueError(
                "Page geometry exceeds the native uint32 element-offset range")
        if self.runtime_dtype not in {"float16", "bfloat16", "fp8_e4m3"}:
            raise ValueError(
                "runtime_dtype must be float16, bfloat16, or fp8_e4m3")

        if self.runtime_dtype == "fp8_e4m3" and (
                self.fp8_scale_orig_quant is None
                or self.fp8_scale_quant_orig is None):
            raise ValueError(
                "FP8 runtime Pages require explicit K/V FP8 scales")
        # The 16-bit kernels do not read FP8 scales. Fill neutral values only
        # to keep one compact native parameter struct for all dtype branches.
        if self.fp8_scale_orig_quant is None:
            object.__setattr__(self, "fp8_scale_orig_quant", (1.0, 1.0))
        if self.fp8_scale_quant_orig is None:
            object.__setattr__(self, "fp8_scale_quant_orig", (1.0, 1.0))

        scale_fields = (
            "nvfp4_scale_orig_quant",
            "nvfp4_scale_quant_orig",
            "fp8_scale_orig_quant",
            "fp8_scale_quant_orig",
        )
        for field_name in scale_fields:
            try:
                pair = tuple(
                    float(value) for value in getattr(self, field_name))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Every K/V scale pair must be numeric") from error
            if len(pair) != 2 or any(
                    not isfinite(value) or value < _FLOAT32_MIN_SUBNORMAL
                    or value > _FLOAT32_MAX for value in pair):
                raise ValueError(
                    "Every K/V scale pair must contain two finite positive values"
                )
            # Normalize list-like config input once so cohort keys remain
            # immutable and hashable on the migration hot path.
            object.__setattr__(self, field_name, pair)

    @property
    def cohort_key(self) -> tuple:
        """Static values that one native batched launch must share."""

        return (
            self.runtime_dtype,
            self.num_kv_heads,
            self.tokens_per_page,
            self.head_dim,
            self.nvfp4_scale_orig_quant,
            self.nvfp4_scale_quant_orig,
            self.fp8_scale_orig_quant,
            self.fp8_scale_quant_orig,
        )


def _load_native_bindings():
    """Load the raw-address binding lazily.

    Importing the Python package must remain possible in documentation and
    CPU-only environments where the native extension is intentionally absent.
    A selected boundary-compression route still fails immediately when it
    tries to submit work without the binding.
    """

    from tensorrt_llm.bindings.internal import \
        kv_cache_compression  # type: ignore

    return kv_cache_compression


class QuantizationCompression(KVCacheCompressionManager):
    """Compress storage-tier KV using the format selected by ``config.quant``.

    The manager owns the format dispatch while each format-specific lowering
    owns its layout and native kernels. NVFP4 is the first supported format.
    Request/step lifecycle behavior remains the base no-op because P0 is
    driven entirely by KVCM's existing migration transaction.

    ``layer_layouts`` is explicit in this prototype because current main has no
    per-level ``BufferAttr`` handoff.  It must eventually come from KVCM/model
    initialization, never from Attention or Attention metadata.
    """

    def __init__(
        self,
        config,
        kv_cache_manager,
        *,
        layer_layouts: Sequence[object],
        draft_kv_cache_manager=None,
    ) -> None:
        quant = getattr(config, "quant", None)
        if not isinstance(quant, str) or not quant:
            raise ValueError("QuantizationCompression requires config.quant")
        if getattr(config, "target_cache_tier", None) != "host":
            raise ValueError(
                "QuantizationCompression P0 supports the Host tier only")
        if draft_kv_cache_manager is not None:
            raise ValueError(
                "Boundary-compression prototype does not support an independent draft KV cache"
            )
        super().__init__(config, kv_cache_manager, draft_kv_cache_manager=None)
        self._quant = quant

        self._initialize_quantization_layouts(layer_layouts)

    def _initialize_quantization_layouts(
            self, layer_layouts: Sequence[object]) -> None:
        """Build only the immutable state owned by the selected format.

        Keeping this dispatch beside ``_run_quantization`` is what makes the
        public manager generic: a future format adds its own initialization and
        lowering branch without changing either StorageManager hook.
        """

        if self._quant == "nvfp4":
            self._init_nvfp4_layouts(layer_layouts)
            return
        raise RuntimeError(
            f"No quantization-compression layout for {self._quant!r}")

    def _init_nvfp4_layouts(self, layer_layouts: Sequence[object]) -> None:
        """Index the format-specific native NVFP4 layout handoff."""

        if not layer_layouts:
            raise ValueError(
                "At least one NVFP4 boundary layer layout is required")

        # KVCM may put several life-cycle variants in one PoolGroup.  Index by
        # both ids so the life_cycle accompanying each Page chooses the exact
        # authoritative BufferAttr variant.
        by_owner: Dict[Tuple[int, int],
                       List[Nvfp4BoundaryLayerLayout]] = defaultdict(list)
        seen_layers = set()
        nvfp4_layouts = []
        for layout in layer_layouts:
            if not isinstance(layout, Nvfp4BoundaryLayerLayout):
                raise TypeError(
                    "quant='nvfp4' requires Nvfp4BoundaryLayerLayout records")
            nvfp4_layouts.append(layout)
        for layout in sorted(
                nvfp4_layouts,
                key=lambda item:
            (item.pool_group_index, item.life_cycle_id, item.layer_id),
        ):
            identity = (layout.pool_group_index, layout.life_cycle_id,
                        layout.layer_id)
            if identity in seen_layers:
                raise ValueError(
                    f"Duplicate boundary layout for pool-group/life-cycle/layer {identity}"
                )
            seen_layers.add(identity)
            by_owner[(layout.pool_group_index,
                      layout.life_cycle_id)].append(layout)
        self._nvfp4_layouts_by_owner = {
            owner: tuple(layouts)
            for owner, layouts in by_owner.items()
        }

    @staticmethod
    def _validate_batch(
        src_life_cycles: Sequence[int],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        num_pages = len(src_life_cycles)
        if len(src_addresses) != num_pages or len(dst_addresses) != num_pages:
            raise ValueError(
                "src_life_cycles, src_addresses, and dst_addresses must have "
                "one entry for the same Page batch")
        if not isinstance(stream, int) or stream < 0 or stream > _UINTPTR_MAX:
            raise ValueError("stream must be a valid CUDA stream pointer")
        for life_cycle in src_life_cycles:
            if not isinstance(life_cycle, int) or life_cycle < 0:
                raise ValueError("life-cycle ids must be non-negative integers")

    def _run_quantization(
        self,
        *,
        offload: bool,
        pool_group_index: int,
        src_life_cycles: Sequence[int],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        """Dispatch one migration batch to the implementation selected by ``quant``."""

        if self._quant == "nvfp4":
            self._lower_nvfp4(
                offload=offload,
                pool_group_index=pool_group_index,
                src_life_cycles=src_life_cycles,
                src_addresses=src_addresses,
                dst_addresses=dst_addresses,
                stream=stream,
            )
            return
        raise RuntimeError(
            f"No quantization-compression implementation for {self._quant!r}")

    def _lower_nvfp4(
        self,
        *,
        offload: bool,
        pool_group_index: int,
        src_life_cycles: Sequence[int],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        """Prevalidate, lower, group, then enqueue one call per cohort.

        Address order passed to native code is always:
        ``raw K, raw V, packed K, packed V, K scale, V scale``.
        Direction only decides which Slot row supplies the raw or compact side.
        """

        if not isinstance(pool_group_index, int) or pool_group_index < 0:
            raise ValueError("pool_group_index must be a non-negative integer")
        self._validate_batch(src_life_cycles, src_addresses, dst_addresses,
                             stream)
        if not src_life_cycles:
            return

        # Finish every deterministic validation before the first kernel launch.
        # KVCM still owns fencing/rollback for a later asynchronous CUDA fault.
        cohorts: Dict[tuple, List[Tuple[int, int, int, int, int,
                                        int]]] = defaultdict(list)
        for page_index, life_cycle in enumerate(src_life_cycles):
            layouts = self._nvfp4_layouts_by_owner.get(
                (pool_group_index, life_cycle))
            if not layouts:
                raise ValueError(
                    f"No boundary layout for PoolGroup {pool_group_index}, life cycle {life_cycle}"
                )

            src_row = src_addresses[page_index]
            dst_row = dst_addresses[page_index]
            raw_row, compact_row = (src_row, dst_row) if offload else (dst_row,
                                                                       src_row)
            for layout in layouts:
                task = (
                    layout.runtime_k.resolve(raw_row),
                    layout.runtime_v.resolve(raw_row),
                    layout.packed_k.resolve(compact_row),
                    layout.packed_v.resolve(compact_row),
                    layout.block_scale_k.resolve(compact_row),
                    layout.block_scale_v.resolve(compact_row),
                )
                # Mirror the native ABI checks for the complete migration
                # batch before submitting its first cohort. This matters when
                # per-layer calibration produces several launches: a bad later
                # cohort must not be discovered after an earlier one started
                # writing into KVCM-owned destination Slots.
                # All four native paths reuse batchedCopy's 16-byte
                # cp.async.cg grain. KVCM must therefore hand off 16-byte
                # aligned raw and packed Pool addresses for every runtime dtype.
                raw_alignment = 16
                for name, address, alignment in (
                    ("raw K", task[0], raw_alignment),
                    ("raw V", task[1], raw_alignment),
                    ("packed K", task[2], 16),
                    ("packed V", task[3], 16),
                    ("K block scale", task[4], 1),
                    ("V block scale", task[5], 1),
                ):
                    if address > _UINTPTR_MAX or address % alignment:
                        raise ValueError(
                            f"{name} address must fit uintptr and be {alignment}-byte aligned"
                        )
                cohorts[layout.cohort_key].append(task)

        native = _load_native_bindings()
        runtime_types = {
            "float16": native.Nvfp4BoundaryRuntimeType.FLOAT16,
            "bfloat16": native.Nvfp4BoundaryRuntimeType.BFLOAT16,
            "fp8_e4m3": native.Nvfp4BoundaryRuntimeType.FP8_E4M3,
        }
        launch = (native.nvfp4_boundary_offload_compress
                  if offload else native.nvfp4_boundary_onboard_decompress)
        for key, tasks in cohorts.items():
            (
                runtime_dtype,
                num_kv_heads,
                tokens_per_page,
                head_dim,
                nvfp4_scale_orig_quant,
                nvfp4_scale_quant_orig,
                fp8_scale_orig_quant,
                fp8_scale_quant_orig,
            ) = key
            launch(
                tasks,
                num_kv_heads,
                tokens_per_page,
                head_dim,
                nvfp4_scale_orig_quant,
                nvfp4_scale_quant_orig,
                fp8_scale_orig_quant,
                fp8_scale_quant_orig,
                runtime_types[runtime_dtype],
                stream,
            )

    def on_offload_compress(
        self,
        *,
        pool_group_index: int,
        src_life_cycles: Sequence[int],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        """Enqueue GPU runtime KV -> mapped-Host NVFP4 for a Page batch."""

        self._run_quantization(
            offload=True,
            pool_group_index=pool_group_index,
            src_life_cycles=src_life_cycles,
            src_addresses=src_addresses,
            dst_addresses=dst_addresses,
            stream=stream,
        )

    def on_onboard_decompress(
        self,
        *,
        pool_group_index: int,
        src_life_cycles: Sequence[int],
        src_addresses: Sequence[Sequence[int]],
        dst_addresses: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        """Enqueue mapped-Host NVFP4 -> GPU runtime KV for a Page batch."""

        self._run_quantization(
            offload=False,
            pool_group_index=pool_group_index,
            src_life_cycles=src_life_cycles,
            src_addresses=src_addresses,
            dst_addresses=dst_addresses,
            stream=stream,
        )
