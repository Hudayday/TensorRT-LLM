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
from math import isfinite
from typing import Optional, Sequence, Tuple

from ..pyexecutor.resource_manager import KVCacheCompressionManager

ScalePair = Tuple[float, float]
_INT32_MAX = (1 << 31) - 1
_UINTPTR_MAX = (1 << 64) - 1
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_FLOAT32_MIN_SUBNORMAL = float.fromhex("0x1p-149")


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

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0
               for value in (self.layer_group_id, self.layer_id)):
            raise ValueError("layer-group and layer ids must be non-negative")
        if any(not isinstance(value, int) or not 0 < value <= _INT32_MAX
               for value in (self.num_kv_heads, self.tokens_per_page,
                             self.head_dim)):
            raise ValueError("Page geometry must contain positive int32 values")
        if self.tokens_per_page % 4 != 0:
            raise ValueError("tokens_per_page must be divisible by 4")
        if self.head_dim % 16 != 0:
            raise ValueError("head_dim must be divisible by 16")
        if self.runtime_dtype not in {"float16", "bfloat16", "fp8_e4m3"}:
            raise ValueError(
                "runtime_dtype must be float16, bfloat16, or fp8_e4m3")
        if self.runtime_dtype == "fp8_e4m3" and (
                self.fp8_scale_orig_quant is None
                or self.fp8_scale_quant_orig is None):
            raise ValueError(
                "FP8 runtime Pages require explicit K/V FP8 scales")

        if self.fp8_scale_orig_quant is None:
            object.__setattr__(self, "fp8_scale_orig_quant", (1.0, 1.0))
        if self.fp8_scale_quant_orig is None:
            object.__setattr__(self, "fp8_scale_quant_orig", (1.0, 1.0))
        for field_name in (
                "nvfp4_scale_orig_quant",
                "nvfp4_scale_quant_orig",
                "fp8_scale_orig_quant",
                "fp8_scale_quant_orig",
        ):
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
            object.__setattr__(self, field_name, pair)


class QuantizationCompression(KVCacheCompressionManager):
    """Own a quantization codec used only at cold storage boundaries.

    This object owns codec selection and configuration. ``native_codec`` is the
    C++ data-plane object that KVCM V2 should retain and invoke from migration.
    The Python encode/decode wrappers below provide backend parity and focused
    prototype testing; the product C++ hot path does not cross Python. KVCM's
    integration must detach its codec reference when this manager is removed.
    """

    def __init__(
        self,
        config,
        kv_cache_manager,
        *,
        layer_configs: Sequence[object],
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
                "Boundary compression does not support an independent draft KV cache"
            )
        super().__init__(config, kv_cache_manager, draft_kv_cache_manager=None)
        self._quant = quant
        self._layer_group_ids = {
            item.layer_group_id
            for item in layer_configs
            if isinstance(item, Nvfp4BoundaryLayerConfig)
        }
        self._native_codec = self._create_native_codec(layer_configs)
        self._configured = False
        self._registered_backend = None

    def _create_native_codec(self, layer_configs: Sequence[object]):
        """Create the single algorithm object owned by this manager."""

        if self._quant != "nvfp4":
            raise RuntimeError(
                f"No quantization-compression codec for {self._quant!r}")
        if not layer_configs:
            raise ValueError(
                "At least one Nvfp4BoundaryLayerConfig is required")

        native = _load_native_bindings()
        runtime_types = {
            "float16": native.Nvfp4BoundaryRuntimeType.FLOAT16,
            "bfloat16": native.Nvfp4BoundaryRuntimeType.BFLOAT16,
            "fp8_e4m3": native.Nvfp4BoundaryRuntimeType.FP8_E4M3,
        }
        native_configs = []
        seen = set()
        for config in sorted(layer_configs,
                             key=lambda item:
                             (item.layer_group_id, item.layer_id)):
            if not isinstance(config, Nvfp4BoundaryLayerConfig):
                raise TypeError(
                    "quant='nvfp4' requires Nvfp4BoundaryLayerConfig records")
            identity = (config.layer_group_id, config.layer_id)
            if identity in seen:
                raise ValueError(f"Duplicate boundary layer config {identity}")
            seen.add(identity)

            native_config = native.Nvfp4ColdPageLayerConfig()
            native_config.layer_group_id = config.layer_group_id
            native_config.layer_id = config.layer_id
            native_config.runtime_type = runtime_types[config.runtime_dtype]
            native_config.num_kv_heads = config.num_kv_heads
            native_config.tokens_per_page = config.tokens_per_page
            native_config.head_dim = config.head_dim
            native_config.nvfp4_scale_orig_quant = (
                config.nvfp4_scale_orig_quant)
            native_config.nvfp4_scale_quant_orig = (
                config.nvfp4_scale_quant_orig)
            native_config.fp8_scale_orig_quant = config.fp8_scale_orig_quant
            native_config.fp8_scale_quant_orig = config.fp8_scale_quant_orig
            native_configs.append(native_config)
        return native.Nvfp4ColdPageCodec(native_configs)

    @property
    def native_codec(self):
        """Return the manager-owned C++ object that KVCM V2 should retain."""

        return self._native_codec

    def configure(self, *, gpu_pool_group_descs: Sequence[object]) -> None:
        """Bind KVCM's authoritative GPU layouts before codec injection.

        This is a concrete-manager initialization method, not a method added to
        the generic compression-manager base class.
        """

        if self._configured:
            raise RuntimeError("QuantizationCompression is already configured")
        if not gpu_pool_group_descs:
            raise ValueError("At least one GPU PoolGroupDesc is required")
        for gpu_desc in gpu_pool_group_descs:
            if not self._native_codec.configure(gpu_desc):
                raise RuntimeError(
                    "NVFP4 cold-page codec rejected a GPU PoolGroupDesc")
        missing = [
            layer_group_id for layer_group_id in self._layer_group_ids if
            int(self._native_codec.query_cold_page_bytes(layer_group_id)) <= 0
        ]
        if missing:
            raise RuntimeError(
                f"GPU PoolGroupDesc handoff omitted layer groups {missing}")
        self._configured = True

    def register_with_kv_cache_manager(self) -> None:
        """Register the manager-owned native hook with the selected V2 backend.

        Both the Python and C++ KVCM V2 implementations expose the same
        ``set_cold_page_codec`` initialization contract. The Python backend
        later invokes the nanobind codec directly; the C++ backend retains its
        ``IKvCacheColdPageCodec`` pointer. Neither path creates a second codec.
        """

        if not self._configured:
            raise RuntimeError("configure() must run before codec registration")
        if self._registered_backend is not None:
            raise RuntimeError("QuantizationCompression is already registered")
        backend = getattr(self.kv_cache_manager, "impl", None)
        setter = getattr(backend, "set_cold_page_codec", None)
        if setter is None:
            raise RuntimeError(
                "The selected KVCM V2 backend does not implement "
                "set_cold_page_codec; Yao Yao's hook integration is required")
        setter(self._native_codec)
        self._registered_backend = backend

    def shutdown(self) -> None:
        """Detach the native hook before releasing the manager-owned codec."""

        backend = self._registered_backend
        if backend is None:
            return
        backend.set_cold_page_codec(None)
        self._registered_backend = None

    def query_cold_page_bytes(self, layer_group_id: int) -> int:
        """Return the compact Host Slot stride requested from KVCM."""

        if not self._configured:
            raise RuntimeError(
                "configure() must run before querying cold layout")
        result = int(self._native_codec.query_cold_page_bytes(layer_group_id))
        if result <= 0:
            raise ValueError(
                f"No compact layout for layer group {layer_group_id}")
        return result

    @staticmethod
    def _validate_call(layer_group_id: int, base_ptr: int,
                       dst_indices: Sequence[int], src_indices: Sequence[int],
                       stream: int) -> None:
        if not isinstance(layer_group_id, int) or layer_group_id < 0:
            raise ValueError("layer_group_id must be non-negative")
        if len(dst_indices) != len(src_indices):
            raise ValueError("source and destination Page batches must match")
        if not isinstance(base_ptr, int) or not 0 <= base_ptr <= _UINTPTR_MAX:
            raise ValueError("cold Pool base must be a valid pointer")
        if not isinstance(stream, int) or not 0 <= stream <= _UINTPTR_MAX:
            raise ValueError("stream must be a valid CUDA stream pointer")
        for index in (*dst_indices, *src_indices):
            if not isinstance(index, int) or not 0 <= index <= _INT32_MAX:
                raise ValueError(
                    "base Page indices must be non-negative int32 values")

    def on_offload_compress(
        self,
        *,
        layer_group_id: int,
        dst_base_ptr: int,
        dst_base_page_indices: Sequence[int],
        src_base_page_indices: Sequence[int],
        stream: int,
    ) -> None:
        """Enqueue GPU runtime KV -> mapped-Host compact KV."""

        if not self._configured:
            raise RuntimeError("configure() must run before migration")
        self._validate_call(layer_group_id, dst_base_ptr, dst_base_page_indices,
                            src_base_page_indices, stream)
        if not self._native_codec.encode(layer_group_id, dst_base_ptr,
                                         list(dst_base_page_indices),
                                         list(src_base_page_indices), stream):
            raise RuntimeError("NVFP4 cold-page encode submission failed")

    def on_onboard_decompress(
        self,
        *,
        layer_group_id: int,
        dst_base_page_indices: Sequence[int],
        src_base_ptr: int,
        src_base_page_indices: Sequence[int],
        stream: int,
    ) -> None:
        """Enqueue mapped-Host compact KV -> GPU runtime KV."""

        if not self._configured:
            raise RuntimeError("configure() must run before migration")
        self._validate_call(layer_group_id, src_base_ptr, dst_base_page_indices,
                            src_base_page_indices, stream)
        if not self._native_codec.decode(
                layer_group_id, list(dst_base_page_indices), src_base_ptr,
                list(src_base_page_indices), stream):
            raise RuntimeError("NVFP4 cold-page decode submission failed")
