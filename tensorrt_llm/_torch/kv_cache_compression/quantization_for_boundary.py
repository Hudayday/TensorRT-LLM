# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization compression at KVCM V2's cold-page migration boundary.

``QuantizationCompression`` remains the lifecycle and configuration authority.
It creates one native codec selected by ``config.quant``, configures it, and
injects the same object into KVCM V2. KVCM may retain shared ownership for
asynchronous safety and calls the object directly; C++ never calls back into
Python from ``_batchedMigrate``.

The codec consumes KVCM's authoritative GPU ``PoolGroupDesc`` once, reports
the compact Host Slot size, and later receives only a Host Pool base pointer,
possibly non-contiguous base-Page indices, and the migration CUDA stream. Page
selection, Slot admission, events, publication, rollback, and eviction remain
KVCM responsibilities. No Attention object or Attention metadata is used.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..pyexecutor.resource_manager import DataType, KVCacheCompressionManager

ScalePair = Tuple[float, float]


def _load_native_bindings():
    """Load the compression-owned C++ codec and fused kernels lazily."""

    from tensorrt_llm.bindings.internal import kv_cache_compression  # type: ignore

    return kv_cache_compression


@dataclass(frozen=True)
class Nvfp4BoundaryLayerConfig:
    """Algorithm-owned geometry and calibration for one local KV layer.

    GPU Pool indices, Slot strides, buffer offsets, and base pointers are not
    duplicated here. The native codec discovers them from ``PoolGroupDesc``
    during ``configure``. Scale pairs are ordered ``(K, V)``.
    """

    layer_group_id: int
    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    head_dim: int
    runtime_dtype: str
    nvfp4_scale_orig_quant: ScalePair
    nvfp4_scale_quant_orig: ScalePair
    fp8_scale_orig_quant: Optional[ScalePair] = None
    fp8_scale_quant_orig: Optional[ScalePair] = None


def _create_nvfp4_codec(layer_configs: Sequence[Nvfp4BoundaryLayerConfig]):
    """Lower manager-owned layer metadata into the native NVFP4 codec."""

    native = _load_native_bindings()
    runtime_types = {
        "float16": native.Nvfp4BoundaryRuntimeType.FLOAT16,
        "bfloat16": native.Nvfp4BoundaryRuntimeType.BFLOAT16,
        "fp8_e4m3": native.Nvfp4BoundaryRuntimeType.FP8_E4M3,
    }
    native_configs = []
    for config in sorted(layer_configs,
                         key=lambda item: (item.layer_group_id, item.layer_id)):
        native_config = native.Nvfp4ColdPageLayerConfig()
        native_config.layer_group_id = config.layer_group_id
        native_config.layer_id = config.layer_id
        native_config.runtime_type = runtime_types[config.runtime_dtype]
        native_config.num_kv_heads = config.num_kv_heads
        native_config.tokens_per_page = config.tokens_per_page
        native_config.head_dim = config.head_dim
        native_config.nvfp4_scale_orig_quant = config.nvfp4_scale_orig_quant
        native_config.nvfp4_scale_quant_orig = config.nvfp4_scale_quant_orig
        if config.runtime_dtype == "fp8_e4m3" and (
                config.fp8_scale_orig_quant is None
                or config.fp8_scale_quant_orig is None):
            raise ValueError(
                "FP8 runtime Pages require explicit K/V FP8 scales")
        native_config.fp8_scale_orig_quant = (config.fp8_scale_orig_quant
                                              or (1.0, 1.0))
        native_config.fp8_scale_quant_orig = (config.fp8_scale_quant_orig
                                              or (1.0, 1.0))
        native_configs.append(native_config)
    return native.Nvfp4ColdPageCodec(native_configs)


def build_uncalibrated_nvfp4_layer_configs_for_testing(
    kv_cache_manager, ) -> tuple[Nvfp4BoundaryLayerConfig, ...]:
    """Build the geometry-only handoff used by the mechanism E2E test.

    KVCM V2 already owns the local layer geometry and lifecycle mapping needed
    by the codec.  Production must additionally obtain calibrated K/V global
    scales from a model-loader-owned provider.  Until that provider exists,
    this helper uses explicit unit scales and is reachable only through the
    fail-closed test gate in ``create_kv_cache_compression_manager``.

    No Attention object or Attention metadata is inspected here.
    """

    runtime_dtype = {
        DataType.HALF: "float16",
        DataType.BF16: "bfloat16",
        DataType.FP8: "fp8_e4m3",
    }.get(kv_cache_manager.dtype)
    if runtime_dtype is None:
        raise RuntimeError(
            "The uncalibrated NVFP4 mechanism test supports FP16, BF16, or "
            f"FP8 runtime KV, not {kv_cache_manager.dtype}")

    unit_scale = (1.0, 1.0)
    return tuple(
        Nvfp4BoundaryLayerConfig(
            layer_group_id=int(
                kv_cache_manager.impl.get_layer_group_id(layer_id)),
            layer_id=layer_id,
            num_kv_heads=int(kv_cache_manager.num_kv_heads_per_layer[layer_id]),
            tokens_per_page=int(kv_cache_manager.tokens_per_block),
            head_dim=int(kv_cache_manager.head_dim_per_layer[layer_id]),
            runtime_dtype=runtime_dtype,
            nvfp4_scale_orig_quant=unit_scale,
            nvfp4_scale_quant_orig=unit_scale,
            fp8_scale_orig_quant=unit_scale,
            fp8_scale_quant_orig=unit_scale,
        ) for layer_id in range(kv_cache_manager.num_local_layers))


class QuantizationCompression(KVCacheCompressionManager):
    """Own a quantization codec used only at cold storage boundaries.

    This object owns codec selection, configuration, registration, and
    teardown. Its native C++ codec is the actual pair of migration hooks:
    KVCM V2 invokes ``encode`` for GPU-to-Host offload and ``decode`` for
    Host-to-GPU onboard. Keeping those hooks native avoids a C++-to-Python
    callback in ``_batchedMigrate`` while leaving algorithm ownership here.

    Page selection, destination admission, ready events, publication, rollback,
    and eviction remain KVCM responsibilities. The manager therefore exposes
    no second Python forwarding API for the same data-plane operations.
    """

    def __init__(
        self,
        config,
        kv_cache_manager,
        *,
        layer_configs: Sequence[Nvfp4BoundaryLayerConfig],
    ) -> None:
        super().__init__(config, kv_cache_manager, draft_kv_cache_manager=None)
        self._native_codec = _create_nvfp4_codec(layer_configs)
        self._registered_backend = None

    def register_with_kv_cache_manager(
            self, *, gpu_pool_group_descs: Sequence[object]) -> None:
        """Configure and register one native codec with KVCM V2.

        Registration is deliberately one initialization transaction. KVCM's
        authoritative GPU descriptors are consumed first; only after every
        descriptor is accepted is the codec published to the backend. This
        avoids a manager-side ``configured`` state whose only purpose was to
        police the ordering of two setup calls made next to each other.

        Both Python and C++ KVCM V2 backends expose ``set_cold_page_codec``.
        The Python backend invokes the nanobind object directly; the C++
        backend retains the same ``IKvCacheColdPageCodec`` instance. Neither
        path creates a second codec or calls back into this Python manager.
        """

        for gpu_desc in gpu_pool_group_descs:
            if not self._native_codec.configure(gpu_desc):
                raise RuntimeError(
                    "NVFP4 cold-page codec rejected a GPU PoolGroupDesc")
        backend = self.kv_cache_manager.impl
        backend.set_cold_page_codec(self._native_codec)
        self._registered_backend = backend

    def shutdown(self) -> None:
        """Detach the native hook before releasing the manager-owned codec."""

        backend = self._registered_backend
        if backend is None:
            return
        backend.set_cold_page_codec(None)
        self._registered_backend = None
