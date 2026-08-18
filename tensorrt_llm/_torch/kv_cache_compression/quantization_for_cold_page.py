# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantize KVCM V2 cold pages.

``ColdPageQuantizationCompression`` is the lifecycle and configuration authority.
It is created before KVCM, reuses model-owned K/V global scale multipliers, and
creates the native codec selected by ``config.quant``. Codec ownership then
moves into KVCM V2's C++ ``StorageManager`` before any cold Slots are allocated;
C++ never calls back into Python from ``_batchedMigrate``. The provider is
construction-only, so it is not registered in the per-iteration
resource-manager cycle because its two hooks are native storage-boundary calls
rather than scheduler-step callbacks.

KVCM passes all authoritative GPU ``PoolGroupDesc`` objects to the codec once,
reports the resulting compact cold Slot sizes, and later supplies only a
GPU-accessible cold base pointer, possibly non-contiguous base-Page indices, and
the migration CUDA stream. Page selection, Slot admission, staging, events,
publication, rollback, and eviction remain KVCM responsibilities. No Attention
object is retained after construction.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from ..pyexecutor.resource_manager import DataType

ScalePair = Tuple[float, float]


def _load_native_bindings():
    """Load the compression-owned C++ codec and fused kernels lazily."""

    from tensorrt_llm.bindings.internal import kv_cache_compression  # type: ignore

    return kv_cache_compression


@dataclass(frozen=True)
class Nvfp4ColdPageLayerConfig:
    """Algorithm-owned geometry and scale values for one local KV layer.

    GPU Pool indices, Slot strides, buffer offsets, and base pointers are not
    duplicated here. The native codec discovers them from ``PoolGroupDesc``
    during ``configure``. Scale pairs are ordered ``(K, V)``.
    """

    layer_id: int
    num_kv_heads: int
    tokens_per_page: int
    head_dim: int
    runtime_dtype: str
    nvfp4_scale_orig_quant: ScalePair
    nvfp4_scale_quant_orig: ScalePair
    fp8_scale_orig_quant: Optional[ScalePair] = None
    fp8_scale_quant_orig: Optional[ScalePair] = None


def _create_nvfp4_codec(layer_configs: Sequence[Nvfp4ColdPageLayerConfig]):
    """Lower manager-owned layer metadata into the native NVFP4 codec."""

    native = _load_native_bindings()
    runtime_types = {
        "float16": native.Nvfp4BoundaryRuntimeType.FLOAT16,
        "bfloat16": native.Nvfp4BoundaryRuntimeType.BFLOAT16,
        "fp8_e4m3": native.Nvfp4BoundaryRuntimeType.FP8_E4M3,
    }
    native_configs = []
    for config in sorted(layer_configs, key=lambda item: item.layer_id):
        native_config = native.Nvfp4ColdPageLayerConfig()
        native_config.layer_id = config.layer_id
        native_config.runtime_type = runtime_types[config.runtime_dtype]
        native_config.num_kv_heads = config.num_kv_heads
        native_config.tokens_per_page = config.tokens_per_page
        native_config.head_dim = config.head_dim
        native_config.nvfp4_scale_orig_quant = config.nvfp4_scale_orig_quant
        native_config.nvfp4_scale_quant_orig = config.nvfp4_scale_quant_orig
        if config.runtime_dtype == "fp8_e4m3" and (
            config.fp8_scale_orig_quant is None or config.fp8_scale_quant_orig is None
        ):
            raise ValueError("FP8 runtime Pages require explicit K/V FP8 scales")
        native_config.fp8_scale_orig_quant = config.fp8_scale_orig_quant or (1.0, 1.0)
        native_config.fp8_scale_quant_orig = config.fp8_scale_quant_orig or (1.0, 1.0)
        native_configs.append(native_config)
    # The factory allocates the codec with C++ ``new`` and returns an owning
    # unique_ptr. That ownership can be safely relinquished when the native
    # KVCacheManager constructor consumes it; a Python-allocated ``nb::init``
    # instance cannot make that transfer with std::default_delete.
    return native.create_nvfp4_cold_page_codec(native_configs)


def _model_nvfp4_scales(
    attention_layers: Mapping[str, object], global_layer_id: int
) -> tuple[ScalePair, ScalePair]:
    """Read native ``[Q,K,V]`` scales from one loaded Attention layer.

    Cold-page Attention initialization guarantees this existing native scale
    pair even for ordinary QKV methods. The pair remains identity when no
    calibrated K/V values are present. The conversion kernel still derives one
    dynamic E4M3 block scale from every 16 values; identity here only disables
    additional global normalization.
    """

    layer_ref = attention_layers.get(str(global_layer_id))
    if layer_ref is None:
        raise RuntimeError(f"Loaded model has no registered Attention layer {global_layer_id}")
    layer = layer_ref()
    if layer is None:
        raise RuntimeError(f"Registered Attention layer {global_layer_id} is no longer alive")
    qkv_proj = getattr(layer, "qkv_proj", None)
    if qkv_proj is None:
        raise RuntimeError(f"Attention layer {global_layer_id} has no fused QKV scale owner")

    quant_orig_tensor = getattr(qkv_proj, "kv_scales", None)
    orig_quant_tensor = getattr(qkv_proj, "inv_kv_scales", None)
    if quant_orig_tensor is None or orig_quant_tensor is None:
        raise RuntimeError(
            f"Attention layer {global_layer_id} must own both kv_scales and inv_kv_scales"
        )

    def read_pair(tensor, name: str) -> ScalePair:
        if tensor.numel() != 3:
            raise RuntimeError(f"Attention layer {global_layer_id} {name} must contain [Q, K, V]")
        pair = tuple(float(value) for value in tensor.detach().reshape(-1)[1:3].tolist())
        if any(not math.isfinite(value) or value <= 0.0 for value in pair):
            raise RuntimeError(
                f"Attention layer {global_layer_id} {name} K/V values must be finite and positive"
            )
        return pair

    return (
        read_pair(orig_quant_tensor, "inv_kv_scales"),
        read_pair(quant_orig_tensor, "kv_scales"),
    )


def _attention_layer_ids(cache_config) -> tuple[int, ...]:
    """Return local Attention layers after validating their K/V roles.

    Runtime dtype alone is not an Attention discriminator: hybrid models can
    store BF16 recurrent state in the same KVCM. ``SsmLayerConfig`` is KVCM's
    authoritative lifecycle kind, so this keeps the codec independent of
    Attention objects and Attention metadata while leaving SSM/conv Pages on
    the codec's lossless path.
    """

    from tensorrt_llm.runtime.kv_cache_manager_v2 import SsmLayerConfig

    result = []
    for layer in cache_config.layers:
        if isinstance(layer, SsmLayerConfig):
            continue
        roles = {str(buffer.role) for buffer in layer.buffers}
        if not {"key", "value"}.issubset(roles):
            raise RuntimeError(
                f"KVCM Attention layer {int(layer.layer_id)} must contain both key and value buffers"
            )
        result.append(int(layer.layer_id))
    return tuple(sorted(result))


def _build_nvfp4_layer_configs(
    cache_config,
    *,
    attention_layers: Mapping[str, object],
    runtime_dtype,
    pp_layers: Sequence[int],
    num_kv_heads_per_layer: Sequence[int],
    head_dim_per_layer: Sequence[int],
) -> tuple[Nvfp4ColdPageLayerConfig, ...]:
    """Combine KVCM's pre-construction layout with model-owned scales."""

    attention_layer_ids = _attention_layer_ids(cache_config)
    if not attention_layer_ids:
        # A pipeline rank of a hybrid model may own only recurrent-state
        # lifecycles. The native codec then delegates every lifecycle to
        # KVCM's default lossless concat codec.
        return ()

    runtime_dtype_name = {
        DataType.HALF: "float16",
        DataType.BF16: "bfloat16",
        DataType.FP8: "fp8_e4m3",
    }.get(runtime_dtype)
    if runtime_dtype_name is None:
        raise RuntimeError(
            "NVFP4 cold-page compression supports FP16, BF16, or FP8 runtime "
            f"Attention KV, not {runtime_dtype}"
        )

    layer_configs = []
    for layer_id in attention_layer_ids:
        global_layer_id = int(pp_layers[layer_id])
        orig_quant, quant_orig = _model_nvfp4_scales(attention_layers, global_layer_id)

        layer_configs.append(
            Nvfp4ColdPageLayerConfig(
                layer_id=layer_id,
                num_kv_heads=int(num_kv_heads_per_layer[layer_id]),
                tokens_per_page=int(cache_config.tokens_per_block),
                head_dim=int(head_dim_per_layer[layer_id]),
                runtime_dtype=runtime_dtype_name,
                nvfp4_scale_orig_quant=orig_quant,
                nvfp4_scale_quant_orig=quant_orig,
                # TRT-LLM's current PyTorch FP8 KV representation uses a unit
                # *source* global scale. This is independent of the model-owned
                # NVFP4 destination K/V multipliers above.
                fp8_scale_orig_quant=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
                fp8_scale_quant_orig=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
            )
        )
    return tuple(layer_configs)


class ColdPageQuantizationCompression:
    """Control-plane owner for cold-page quantization.

    This manager owns the quantization choice, loaded-model scale adapter,
    layer configuration, and native-codec construction. KVCM V2 invokes the
    transferred codec's native ``encode`` for hot-to-cold offload and ``decode``
    for cold-to-hot onboard. Keeping the data hooks native avoids a
    C++-to-Python callback in ``_batchedMigrate`` while leaving algorithm policy
    in this manager.

    Page selection, destination admission, ready events, publication, rollback,
    and eviction remain KVCM responsibilities. The manager therefore exposes
    no second Python forwarding API or late codec-registration lifetime.
    """

    def __init__(self, config, attention_layers: Mapping[str, object]) -> None:
        # KVCM does not exist yet: the codec must be available to its native
        # constructor so StorageManager can size cold Slots from the codec.
        self.config = config
        self._attention_layers = attention_layers

    def create_cold_page_codec(
        self,
        cache_config,
        *,
        runtime_dtype,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
    ):
        """Create a fresh native codec for one KVCM construction attempt.

        The returned binding owns a ``unique_ptr<IKvCacheColdPageCodec>``;
        native KVCM consumes it exactly once. Immutable algorithm and scale
        metadata stays on this manager so the control-plane contract remains
        inspectable after the native data-plane handle is transferred.
        """

        if self.config.algorithm != "quantization_for_cold_page":
            raise ValueError("ColdPageQuantizationCompression received the wrong config")
        if self.config.quant != "nvfp4":
            raise NotImplementedError(
                f"Unsupported quantization compression format {self.config.quant!r}"
            )

        from tensorrt_llm._utils import is_sm_100f

        if not is_sm_100f():
            raise RuntimeError(
                "NVFP4 cold-page compression requires an SM100-family device (SM100 or SM103)."
            )

        return _create_nvfp4_codec(
            _build_nvfp4_layer_configs(
                cache_config,
                attention_layers=self._attention_layers,
                runtime_dtype=runtime_dtype,
                pp_layers=pp_layers,
                num_kv_heads_per_layer=num_kv_heads_per_layer,
                head_dim_per_layer=head_dim_per_layer,
            )
        )
