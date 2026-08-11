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

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ..pyexecutor.resource_manager import DataType, KVCacheCompressionManager

ScalePair = Tuple[float, float]


def _load_native_bindings():
    """Load the compression-owned C++ codec and fused kernels lazily."""

    from tensorrt_llm.bindings.internal import \
        kv_cache_compression  # type: ignore

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
        native_config.fp8_scale_orig_quant = config.fp8_scale_orig_quant or (
            1.0, 1.0)
        native_config.fp8_scale_quant_orig = config.fp8_scale_quant_orig or (
            1.0, 1.0)
        native_configs.append(native_config)
    return native.Nvfp4ColdPageCodec(native_configs)


def _positive_scale(value, name: str) -> float:
    """Parse one finite positive calibration scalar."""

    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"{name} must be finite and positive")
    return scale


_LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _load_nvfp4_scales(
    checkpoint_path: str,
    expected_layer_ids: Sequence[int],
) -> dict[int, ScalePair]:
    """Read standard ModelOpt ``k_scale``/``v_scale`` tensors without a model.

    This is the same on-disk contract used by the native fused-QKV loaders. It
    opens only safetensors metadata and scalar scale tensors; model weights are
    never materialized and no runtime module or Attention state is inspected.
    """

    path = Path(checkpoint_path)
    checkpoint_dir = path.parent if path.is_file() else path
    quant_config_path = checkpoint_dir / "hf_quant_config.json"
    if not quant_config_path.is_file():
        raise RuntimeError(
            "NVFP4 scale checkpoints must include hf_quant_config.json")
    from ...quantization.modelopt_config import read_modelopt_quant_config

    try:
        quantization = read_modelopt_quant_config(
            json.loads(quant_config_path.read_text()))
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("Invalid ModelOpt hf_quant_config.json") from error
    if quantization.get("kv_cache_quant_algo") != "NVFP4":
        raise RuntimeError(
            "The scale checkpoint must declare kv_cache_quant_algo=NVFP4; "
            f"got {quantization.get('kv_cache_quant_algo')!r}")

    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(
            f"No safetensors files found in NVFP4 scale checkpoint {path}")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "Loading NVFP4 checkpoint scales requires safetensors") from error

    values: dict[int, dict[str, list[float]]] = {}
    for tensor_file in files:
        with safe_open(str(tensor_file), framework="pt",
                       device="cpu") as archive:
            for key in archive.keys():
                role = ("k" if key.endswith(".k_scale") else
                        "v" if key.endswith(".v_scale") else None)
                if role is None or ".self_attn." not in key:
                    continue
                match = _LAYER_ID_PATTERN.search(key)
                if match is None:
                    continue
                layer_id = int(match.group(1))
                tensor = archive.get_tensor(key)
                if tensor.numel() != 1:
                    raise RuntimeError(f"{key} must be a scalar")
                scale = _positive_scale(tensor.item(), key)
                values.setdefault(layer_id, {}).setdefault(role,
                                                           []).append(scale)

    result = {}
    for layer_id in expected_layer_ids:
        role_values = values.get(layer_id, {})
        if "k" not in role_values or "v" not in role_values:
            raise RuntimeError(
                f"Attention layer {layer_id} must provide both ModelOpt k_scale and v_scale"
            )
        # This matches the native fused-QKV loaders, which take the maximum
        # when a checkpoint supplies more than one K or V scale shard.
        result[layer_id] = (max(role_values["k"]), max(role_values["v"]))
    return result


def _attention_layer_ids(
        gpu_pool_group_descs: Sequence[object]) -> tuple[int, ...]:
    """Return local layers whose authoritative KVCM buffers are K and V.

    Runtime dtype alone is not an Attention discriminator: hybrid models can
    store BF16 recurrent state in the same KVCM. Buffer roles are part of the
    storage descriptor, so this keeps the codec independent of Attention
    objects and Attention metadata while leaving SSM/conv Pages on KVCM's raw
    migration path.
    """

    roles_by_layer: dict[int, set[str]] = {}
    for pool_group in gpu_pool_group_descs:
        for variant in pool_group.slot_desc.variants:
            for coalesced_buffer in variant.coalesced_buffers:
                for buffer_id in coalesced_buffer.buffer_ids:
                    roles_by_layer.setdefault(int(buffer_id.layer_id),
                                              set()).add(str(buffer_id.role))

    return tuple(
        sorted(layer_id for layer_id, roles in roles_by_layer.items()
               if {"key", "value"}.issubset(roles)))


def _build_nvfp4_layer_configs(
    config,
    kv_cache_manager,
    gpu_pool_group_descs: Sequence[object],
) -> tuple[Nvfp4BoundaryLayerConfig, ...]:
    """Combine Attention-KV geometry with manager-owned calibration."""

    runtime_dtype = {
        DataType.HALF: "float16",
        DataType.BF16: "bfloat16",
        DataType.FP8: "fp8_e4m3",
    }.get(kv_cache_manager.dtype)
    if runtime_dtype is None:
        raise RuntimeError(
            "NVFP4 boundary compression supports FP16, BF16, or FP8 runtime "
            f"KV, not {kv_cache_manager.dtype}")

    attention_layer_ids = _attention_layer_ids(gpu_pool_group_descs)
    if not attention_layer_ids:
        raise RuntimeError(
            "NVFP4 boundary compression found no K/V buffers in KVCM V2")

    global_layer_ids = tuple(
        int(kv_cache_manager.pp_layers[layer_id])
        for layer_id in attention_layer_ids)
    calibration = _load_nvfp4_scales(config.scale_checkpoint_path,
                                     global_layer_ids)

    layer_configs = []
    for layer_id in attention_layer_ids:
        global_layer_id = kv_cache_manager.pp_layers[layer_id]
        if global_layer_id not in calibration:
            raise RuntimeError(
                f"NVFP4 calibration has no entry for local model layer {global_layer_id}"
            )
        quant_orig = calibration[global_layer_id]
        orig_quant = tuple(1.0 / scale for scale in quant_orig)

        layer_configs.append(
            Nvfp4BoundaryLayerConfig(
                layer_group_id=int(
                    kv_cache_manager.impl.get_layer_group_id(layer_id)),
                layer_id=layer_id,
                num_kv_heads=int(
                    kv_cache_manager.num_kv_heads_per_layer[layer_id]),
                tokens_per_page=int(kv_cache_manager.tokens_per_block),
                head_dim=int(kv_cache_manager.head_dim_per_layer[layer_id]),
                runtime_dtype=runtime_dtype,
                nvfp4_scale_orig_quant=orig_quant,
                nvfp4_scale_quant_orig=quant_orig,
                # TRT-LLM's current FP8 runtime KV representation uses a unit
                # global scale. Keep that runtime contract internal instead of
                # exposing a second user-controlled calibration surface.
                fp8_scale_orig_quant=((1.0, 1.0)
                                      if runtime_dtype == "fp8_e4m3" else None),
                fp8_scale_quant_orig=((1.0, 1.0)
                                      if runtime_dtype == "fp8_e4m3" else None),
            ))
    return tuple(layer_configs)


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
        gpu_pool_group_descs: Sequence[object],
    ) -> None:
        super().__init__(config, kv_cache_manager, draft_kv_cache_manager=None)
        self._layer_configs = _build_nvfp4_layer_configs(
            config, kv_cache_manager, gpu_pool_group_descs)
        self._native_codec = _create_nvfp4_codec(self._layer_configs)
        self._registered_backend = None
        self._register_with_kv_cache_manager(gpu_pool_group_descs)

    def _register_with_kv_cache_manager(
            self, gpu_pool_group_descs: Sequence[object]) -> None:
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
        missing_layer_groups = sorted({
            layer.layer_group_id
            for layer in self._layer_configs if
            self._native_codec.query_cold_page_bytes(layer.layer_group_id) == 0
        })
        if missing_layer_groups:
            raise RuntimeError(
                "NVFP4 cold-page codec did not configure expected K/V layer "
                f"groups: {missing_layer_groups}")
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
