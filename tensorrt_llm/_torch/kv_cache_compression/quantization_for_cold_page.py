# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantize KVCM V2 cold pages.

``ColdPageQuantizationCompression`` is the lifecycle and configuration authority.
It is created before KVCM, reads optional ModelOpt K/V global scale metadata
directly from the resolved checkpoint, and creates the native codec selected by
``config.quant``. Codec ownership then moves into KVCM V2's C++
``StorageManager`` before any cold Slots are allocated; C++ never calls back
into Python from ``_batchedMigrate``. The provider is construction-only, so it
is not registered in the per-iteration resource-manager cycle because its two
hooks are native storage-boundary calls rather than scheduler-step callbacks.

KVCM passes all authoritative GPU ``PoolGroupDesc`` objects to the codec once,
reports the resulting compact cold Slot sizes, and later supplies only a
GPU-accessible cold base pointer, possibly non-contiguous base-Page indices, and
the migration CUDA stream. Page selection, Slot admission, staging, events,
publication, rollback, and eviction remain KVCM responsibilities. No Attention
or model object is involved in the scale path.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple, Union

from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ..pyexecutor.resource_manager import DataType

ScalePair = Tuple[float, float]
ModelSource = Optional[Union[str, os.PathLike]]

_IDENTITY_NVFP4_SCALES = ((1.0, 1.0), (1.0, 1.0))
_MODEL_OPT_KV_SCALE_KEY = re.compile(
    r"(?:^|\.)layers\.(?P<layer_id>\d+)\.self_attn\."
    r"(?P<kind>[kv])_proj\.(?P=kind)_scale$"
)


def _normalize_checkpoint_kv_cache_quant_algo(algorithm) -> Optional[str]:
    """Normalize a checkpoint KV-cache algorithm string."""

    if algorithm is None:
        return None
    if not isinstance(algorithm, str):
        raise ValueError(
            f"ModelOpt kv_cache_quant_algo must be a string, got {type(algorithm).__name__}"
        )
    value = algorithm.upper()
    if value not in {"NVFP4", "FP8", "INT8"}:
        raise ValueError(f"Unsupported checkpoint KV-cache algorithm {value!r}")
    return value


def _checkpoint_metadata_directory(model_source: ModelSource) -> Optional[Path]:
    """Return the directory that owns checkpoint quantization metadata."""

    if model_source is None:
        return None

    source = Path(model_source)
    if not source.exists():
        raise FileNotFoundError(f"Model checkpoint source does not exist: {source}")
    if source.is_dir():
        return source
    if source.is_file():
        return source.parent
    raise ValueError(f"Model checkpoint source is not a file or directory: {source}")


def _read_json_object(path: Path) -> dict:
    """Read one checkpoint JSON object with source-aware diagnostics."""

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read checkpoint metadata {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint metadata {path} must contain a JSON object")
    return raw


def _resolve_checkpoint_kv_cache_quant_algo(model_source: ModelSource) -> Optional[str]:
    """Resolve ModelOpt KV-cache provenance directly from the checkpoint.

    ``hf_quant_config.json`` is authoritative when present. Otherwise a
    ModelOpt-owned ``config.json.quantization_config`` is used as the inline
    fallback. Ordinary model configs are intentionally ignored.
    """

    checkpoint_dir = _checkpoint_metadata_directory(model_source)
    if checkpoint_dir is None:
        return None

    modelopt_path = checkpoint_dir / "hf_quant_config.json"
    if modelopt_path.exists():
        if not modelopt_path.is_file():
            raise ValueError(f"Checkpoint metadata is not a file: {modelopt_path}")
        normalized = read_modelopt_quant_config(_read_json_object(modelopt_path))
        return _normalize_checkpoint_kv_cache_quant_algo(normalized.get("kv_cache_quant_algo"))

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return None
    if not config_path.is_file():
        raise ValueError(f"Checkpoint metadata is not a file: {config_path}")

    inline = _read_json_object(config_path).get("quantization_config")
    if inline is None or not is_modelopt_quant_config(inline):
        return None
    normalized = read_modelopt_quant_config(inline)
    return _normalize_checkpoint_kv_cache_quant_algo(normalized.get("kv_cache_quant_algo"))


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

    from tensorrt_llm.bindings.internal import kv_cache_compression as native  # type: ignore

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


def _checkpoint_safetensor_inputs(
    model_source: ModelSource,
) -> tuple[tuple[Path, Optional[tuple[str, ...]]], ...]:
    """Resolve safetensors inputs like the default Hugging Face weight loader.

    A direct safetensors path is accepted. For a directory, ordinary shards
    win over files whose basename contains ``consolidated``; if only
    consolidated files exist, they are used.

    ``model.safetensors.index.json`` is authoritative when present. Only shards
    referenced by matching scale entries are opened, and each tuple carries
    the exact indexed tensor names to read. A valid checkpoint with no matching
    metadata simply contributes identity global scales.
    """

    if model_source is None:
        return ()

    source = Path(model_source)
    if not source.exists():
        raise FileNotFoundError(f"Model checkpoint source does not exist: {source}")
    if source.is_file():
        return ((source, None),) if source.suffix == ".safetensors" else ()
    if not source.is_dir():
        raise ValueError(f"Model checkpoint source is not a file or directory: {source}")

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read safetensors index {index_path}: {error}") from error
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise ValueError(
                f"Safetensors index {index_path} must contain an object-valued weight_map"
            )

        shard_tensors: dict[str, list[str]] = {}
        for tensor_name, shard_name in index["weight_map"].items():
            if _MODEL_OPT_KV_SCALE_KEY.search(tensor_name) is None:
                continue
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError(
                    f"Safetensors index {index_path} has an invalid shard for {tensor_name}"
                )
            shard_tensors.setdefault(shard_name, []).append(tensor_name)

        result = []
        for shard_name in sorted(shard_tensors):
            relative_shard = Path(shard_name)
            if relative_shard.is_absolute() or ".." in relative_shard.parts:
                raise ValueError(
                    f"Safetensors index {index_path} has an unsafe scale shard path {shard_name!r}"
                )
            shard_path = source / relative_shard
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Safetensors index {index_path} references missing scale shard {shard_path}"
                )
            if shard_path.suffix != ".safetensors":
                raise ValueError(
                    f"Safetensors index {index_path} maps KV scale metadata to "
                    f"non-safetensors shard {shard_path}"
                )
            result.append((shard_path, tuple(sorted(shard_tensors[shard_name]))))
        return tuple(result)

    weight_files = tuple(sorted(source.glob("*.safetensors")))
    ordinary_files = tuple(path for path in weight_files if "consolidated" not in path.name)
    return tuple((path, None) for path in (ordinary_files or weight_files))


def _load_checkpoint_nvfp4_scales(
    model_source: ModelSource,
    checkpoint_kv_cache_quant_algo: Optional[str],
) -> Mapping[int, tuple[ScalePair, ScalePair]]:
    """Read ModelOpt K/V global scales once into immutable Python values.

    ModelOpt exports each dequantization multiplier as scalar
    ``...layers.N.self_attn.{k,v}_proj.{k,v}_scale`` metadata. The checkpoint
    value is quant-to-original; the codec's original-to-quant multiplier is its
    reciprocal. Missing metadata uses identity. A malformed or ambiguous pair
    fails closed before KVCM construction.

    ``TRTLLM_LOAD_KV_SCALES`` deliberately matches the native fused-QKV loader:
    it defaults to ``"1"`` and any other value disables metadata loading.
    """

    if os.environ.get("TRTLLM_LOAD_KV_SCALES", "1") != "1":
        return MappingProxyType({})

    if checkpoint_kv_cache_quant_algo in {"FP8", "INT8"}:
        # These checkpoints use the same tensor names for algorithm-specific
        # source scales. They are not valid NVFP4 global scales. Hot FP8/INT8
        # can still use identity global normalization when encoded to cold
        # NVFP4 by the boundary codec.
        return MappingProxyType({})

    from safetensors import safe_open

    found: dict[tuple[int, str], tuple[float, str, Path]] = {}
    for file_path, indexed_names in _checkpoint_safetensor_inputs(model_source):
        with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
            checkpoint_names = tuple(checkpoint.keys())
            available_names = set(checkpoint_names)
            tensor_names = indexed_names if indexed_names is not None else checkpoint_names
            for tensor_name in tensor_names:
                if tensor_name not in available_names:
                    raise ValueError(
                        f"Safetensors index references {file_path}:{tensor_name}, "
                        "but that tensor is absent from the shard"
                    )
                match = _MODEL_OPT_KV_SCALE_KEY.search(tensor_name)
                if match is None:
                    continue
                if checkpoint_kv_cache_quant_algo is None:
                    raise ValueError(
                        f"Ambiguous ModelOpt KV scale metadata {file_path}:{tensor_name}: "
                        "checkpoint_kv_cache_quant_algo was not resolved as NVFP4"
                    )

                layer_id = int(match.group("layer_id"))
                kind = match.group("kind")
                identity = (layer_id, kind)
                if identity in found:
                    previous_name = found[identity][1]
                    previous_path = found[identity][2]
                    raise ValueError(
                        f"Duplicate ModelOpt {kind.upper()} scale metadata for layer "
                        f"{layer_id}: {previous_path}:{previous_name} and "
                        f"{file_path}:{tensor_name}"
                    )

                tensor = checkpoint.get_tensor(tensor_name)
                if tensor.numel() != 1:
                    raise ValueError(
                        f"ModelOpt KV scale metadata {file_path}:{tensor_name} must "
                        f"contain exactly one value, got {tensor.numel()}"
                    )
                value = float(tensor.reshape(-1)[0].item())
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"ModelOpt KV scale metadata {file_path}:{tensor_name} must "
                        f"be finite and positive, got {value}"
                    )
                found[identity] = (value, tensor_name, file_path)

    result: dict[int, tuple[ScalePair, ScalePair]] = {}
    for layer_id in sorted({identity[0] for identity in found}):
        k_entry = found.get((layer_id, "k"))
        v_entry = found.get((layer_id, "v"))
        if k_entry is None or v_entry is None:
            missing = "K" if k_entry is None else "V"
            present = v_entry if k_entry is None else k_entry
            assert present is not None
            raise ValueError(
                f"ModelOpt KV scale metadata for layer {layer_id} is partial: "
                f"{present[2]}:{present[1]} has no matching {missing} scale"
            )
        quant_orig = (k_entry[0], v_entry[0])
        orig_quant = (1.0 / quant_orig[0], 1.0 / quant_orig[1])
        result[layer_id] = (orig_quant, quant_orig)

    return MappingProxyType(result)


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
    model_nvfp4_scales: Mapping[int, tuple[ScalePair, ScalePair]],
    runtime_dtype,
    pp_layers: Sequence[int],
    num_kv_heads_per_layer: Sequence[int],
    head_dim_per_layer: Sequence[int],
) -> tuple[Nvfp4ColdPageLayerConfig, ...]:
    """Combine KVCM's local layout with checkpoint-owned global scales."""

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
        scales = model_nvfp4_scales.get(global_layer_id, _IDENTITY_NVFP4_SCALES)
        orig_quant, quant_orig = scales

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
                # *source* global scale. This is independent of the
                # checkpoint-owned NVFP4 destination K/V multipliers above.
                fp8_scale_orig_quant=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
                fp8_scale_quant_orig=((1.0, 1.0) if runtime_dtype_name == "fp8_e4m3" else None),
            )
        )
    return tuple(layer_configs)


class ColdPageQuantizationCompression:
    """Control-plane owner for cold-page quantization.

    This manager owns the quantization choice, optional checkpoint auxiliary
    K/V scale metadata, layer configuration, and native-codec construction.
    Scale metadata is read once during construction and retained as immutable
    Python floats; the loaded model and Attention modules remain unaware that
    cold Pages use NVFP4. KVCM V2 invokes the transferred codec's native
    ``encode`` for hot-to-cold offload and ``decode`` for cold-to-hot onboard.
    Keeping the data hooks native avoids a C++-to-Python callback in
    ``_batchedMigrate`` while leaving algorithm policy in this manager.

    Page selection, destination admission, ready events, publication, rollback,
    and eviction remain KVCM responsibilities. The manager therefore exposes
    no second Python forwarding API or late codec-registration lifetime.
    """

    def __init__(
        self,
        config,
        model_source: ModelSource = None,
    ) -> None:
        """Snapshot optional ModelOpt scale metadata from ``model_source``.

        ``model_source`` is the resolved local checkpoint directory (or a
        direct safetensors path). This manager independently resolves ModelOpt
        provenance from the checkpoint: ``hf_quant_config.json`` is
        authoritative, with ModelOpt ``config.json.quantization_config`` as
        the fallback. Only NVFP4 provenance admits the auxiliary scalars.
        Known FP8/INT8 provenance and missing metadata use identity global
        scales. Metadata without provenance fails as ambiguous.

        ``TRTLLM_LOAD_KV_SCALES=0`` disables this auxiliary metadata ingress
        exactly as it does for the native QKV checkpoint loader.
        """

        # KVCM does not exist yet: the codec must be available to its native
        # constructor so StorageManager can size cold Slots from the codec.
        self.config = config
        self.checkpoint_kv_cache_quant_algo = _resolve_checkpoint_kv_cache_quant_algo(model_source)
        self._model_nvfp4_scales = _load_checkpoint_nvfp4_scales(
            model_source,
            self.checkpoint_kv_cache_quant_algo,
        )

    def validate_runtime_support(self) -> None:
        """Fail before KVCM performs any setup on an unsupported device."""
        from tensorrt_llm._utils import is_sm_100f

        if not is_sm_100f():
            raise RuntimeError(
                "NVFP4 cold-page compression requires an SM100-family device (SM100 or SM103)."
            )

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

        return _create_nvfp4_codec(
            _build_nvfp4_layer_configs(
                cache_config,
                model_nvfp4_scales=self._model_nvfp4_scales,
                runtime_dtype=runtime_dtype,
                pp_layers=pp_layers,
                num_kv_heads_per_layer=num_kv_heads_per_layer,
                head_dim_per_layer=head_dim_per_layer,
            )
        )
