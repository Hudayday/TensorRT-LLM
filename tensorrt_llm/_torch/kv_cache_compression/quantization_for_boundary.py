# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization compression at KVCM V2's cold-page migration boundary.

``QuantizationCompression`` is the lifecycle and configuration authority. It is
created before KVCM, loads its calibration, and creates the native codec selected
by ``config.quant``. Codec ownership then moves into KVCM V2's C++
``StorageManager`` before any cold Slots are allocated; C++ never calls back into
Python from ``_batchedMigrate``. The manager is retained by KVCM, not registered
in the per-iteration resource-manager cycle, because its two hooks are native
storage-boundary calls rather than scheduler-step callbacks.

KVCM passes all authoritative GPU ``PoolGroupDesc`` objects to the codec once,
reports the resulting compact Host Slot sizes, and later supplies only a Host
Pool base pointer, possibly non-contiguous base-Page indices, and the migration
CUDA stream. Page selection, Slot admission, events, publication, rollback, and
eviction remain KVCM responsibilities. No Attention object or Attention
metadata is used.
"""

import json
import math
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
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
        raise RuntimeError("NVFP4 scale checkpoints must include hf_quant_config.json")
    from ...quantization.modelopt_config import read_modelopt_quant_config

    try:
        quantization = read_modelopt_quant_config(json.loads(quant_config_path.read_text()))
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("Invalid ModelOpt hf_quant_config.json") from error
    if quantization.get("kv_cache_quant_algo") != "NVFP4":
        raise RuntimeError(
            "The scale checkpoint must declare kv_cache_quant_algo=NVFP4; "
            f"got {quantization.get('kv_cache_quant_algo')!r}"
        )

    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors files found in NVFP4 scale checkpoint {path}")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Loading NVFP4 checkpoint scales requires safetensors") from error

    values: dict[int, dict[str, list[float]]] = {}
    for tensor_file in files:
        with safe_open(str(tensor_file), framework="pt", device="cpu") as archive:
            for key in archive.keys():
                role = (
                    "k" if key.endswith(".k_scale") else "v" if key.endswith(".v_scale") else None
                )
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
                values.setdefault(layer_id, {}).setdefault(role, []).append(scale)

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
    config,
    cache_config,
    *,
    runtime_dtype,
    pp_layers: Sequence[int],
    num_kv_heads_per_layer: Sequence[int],
    head_dim_per_layer: Sequence[int],
) -> tuple[Nvfp4BoundaryLayerConfig, ...]:
    """Combine KVCM's pre-construction layout with manager-owned scales."""

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
            "NVFP4 boundary compression supports FP16, BF16, or FP8 runtime "
            f"Attention KV, not {runtime_dtype}"
        )

    global_layer_ids = tuple(int(pp_layers[layer_id]) for layer_id in attention_layer_ids)
    calibration = _load_nvfp4_scales(config.scale_checkpoint_path, global_layer_ids)

    layer_configs = []
    for layer_id in attention_layer_ids:
        global_layer_id = int(pp_layers[layer_id])
        if global_layer_id not in calibration:
            raise RuntimeError(
                f"NVFP4 calibration has no entry for local model layer {global_layer_id}"
            )
        quant_orig = calibration[global_layer_id]
        orig_quant = tuple(1.0 / scale for scale in quant_orig)

        layer_configs.append(
            Nvfp4BoundaryLayerConfig(
                layer_id=layer_id,
                num_kv_heads=int(num_kv_heads_per_layer[layer_id]),
                tokens_per_page=int(cache_config.tokens_per_block),
                head_dim=int(head_dim_per_layer[layer_id]),
                runtime_dtype=runtime_dtype_name,
                nvfp4_scale_orig_quant=orig_quant,
                nvfp4_scale_quant_orig=quant_orig,
                # TRT-LLM's current PyTorch FP8 KV representation uses a unit
                # *source* global scale. This is independent of the calibrated
                # NVFP4 destination K/V scales above; never replace those
                # checkpoint-loaded values with this FP8 runtime convention.
                fp8_scale_orig_quant=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
                fp8_scale_quant_orig=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
            )
        )
    return tuple(layer_configs)


class QuantizationCompression(KVCacheCompressionManager):
    """Control-plane owner for quantization at cold storage boundaries.

    This manager owns the quantization choice, checkpoint path, calibrated
    layer configuration, and native-codec construction. KVCM V2 invokes the
    transferred codec's native ``encode`` for GPU-to-Host offload and ``decode``
    for Host-to-GPU onboard. Keeping the data hooks native avoids a
    C++-to-Python callback in ``_batchedMigrate`` while leaving algorithm policy
    in this manager.

    Page selection, destination admission, ready events, publication, rollback,
    and eviction remain KVCM responsibilities. The manager therefore exposes
    no second Python forwarding API or late codec-registration lifetime.
    """

    def __init__(self, config) -> None:
        # KVCM does not exist yet: the codec must be available to its native
        # constructor so StorageManager can size cold Slots from the codec.
        self.config = config
        self.layer_configs: tuple[Nvfp4BoundaryLayerConfig, ...] = ()
        self._kv_cache_manager_ref = None
        self.draft_kv_cache_manager = None

    @property
    def kv_cache_manager(self):
        """Return the bound KVCM without creating an ownership cycle."""

        return None if self._kv_cache_manager_ref is None else self._kv_cache_manager_ref()

    @kv_cache_manager.setter
    def kv_cache_manager(self, value) -> None:
        # Base KVCacheCompressionManager initializes this attribute. Boundary
        # compression is retained by KVCM itself, so a strong reverse edge
        # would make KVCM <-> manager a reference cycle at executor shutdown.
        self._kv_cache_manager_ref = None if value is None else weakref.ref(value)

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
        native KVCM consumes it exactly once. Immutable algorithm/calibration
        metadata stays on this manager so the control-plane contract remains
        inspectable after the native data-plane handle is transferred.
        """

        if self.config.algorithm != "quantization_for_boundary":
            raise ValueError("QuantizationCompression received the wrong config")
        if self.config.quant != "nvfp4":
            raise NotImplementedError(
                f"Unsupported quantization compression format {self.config.quant!r}"
            )

        from tensorrt_llm._utils import is_sm_100f

        if not is_sm_100f():
            raise RuntimeError(
                "NVFP4 boundary compression requires an SM100-family device (SM100 or SM103)."
            )

        layer_configs = _build_nvfp4_layer_configs(
            self.config,
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
        )
        if self.layer_configs and self.layer_configs != layer_configs:
            raise RuntimeError("KVCM construction attempts produced different NVFP4 layouts")
        self.layer_configs = layer_configs
        return _create_nvfp4_codec(layer_configs)

    def bind_kv_cache_manager(self, kv_cache_manager) -> None:
        """Bind the manager after native KVCM has consumed its codec."""

        if self.kv_cache_manager is not None:
            raise RuntimeError("QuantizationCompression is already bound to KVCM")
        super().__init__(
            self.config,
            kv_cache_manager,
            draft_kv_cache_manager=None,
        )
