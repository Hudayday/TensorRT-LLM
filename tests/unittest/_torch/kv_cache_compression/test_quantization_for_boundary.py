# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for QuantizationCompression's native codec ownership."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationCompression,
    _load_nvfp4_scales,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.llmapi.llm_args import QuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)


def _v2_manager(backend, *, num_layers=2, dtype=DataType.BF16, local_heads=8):
    """Build a real V2 instance without invoking its heavyweight constructor."""
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.impl = backend
    manager.dtype = dtype
    manager.num_local_layers = num_layers
    manager.pp_layers = (10, 4)[:num_layers]
    manager.num_kv_heads_per_layer = (local_heads,) * num_layers
    manager.head_dim_per_layer = (128,) * num_layers
    manager.tokens_per_block = 64
    manager.kv_compression_manages_history = False
    return manager


def _config(path="/modelopt-checkpoint", **kwargs):
    return QuantizationCompressionConfig(scale_checkpoint_path=path, **kwargs)


def _cache_config(*layers):
    configs = []
    for layer in layers:
        layer_id, roles, *kind = layer
        layer_type = SsmLayerConfig if kind == ["ssm"] else AttentionLayerConfig
        configs.append(
            layer_type(
                layer_id=layer_id,
                buffers=[BufferConfig(role=role, size=128) for role in roles],
            )
        )
    return SimpleNamespace(
        tokens_per_block=64,
        layers=tuple(configs),
    )


def _native():
    codec = MagicMock()
    module = SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageLayerConfig=SimpleNamespace,
        create_nvfp4_cold_page_codec=MagicMock(return_value=codec),
    )
    return module, codec


def _manager(path="/modelopt-checkpoint", **kwargs):
    return QuantizationCompression(_config(path, **kwargs))


def test_factory_builds_one_native_codec():
    native, codec = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
            return_value={10: (0.5, 0.25), 4: (0.125, 0.0625)},
        ),
    ):
        manager = _manager()
        codec_owner = manager.create_cold_page_codec(
            _cache_config((0, ("key", "value")), (1, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    assert codec_owner is codec
    native_configs = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert [config.layer_id for config in native_configs] == [0, 1]
    assert [config.runtime_type for config in native_configs] == [
        "native-bf16",
        "native-bf16",
    ]
    assert [config.nvfp4_scale_quant_orig for config in native_configs] == [
        (0.5, 0.25),
        (0.125, 0.0625),
    ]
    assert [config.nvfp4_scale_orig_quant for config in native_configs] == [
        (2.0, 4.0),
        (8.0, 16.0),
    ]
    # Pool addresses, lifecycle IDs, and Pool geometry are intentionally absent:
    # native KVCM owns those values and calls codec.configure() after ownership
    # transfer in its constructor.
    assert not hasattr(native_configs[0], "layer_group_id")
    assert manager.layer_configs
    assert manager.config.scale_checkpoint_path == "/modelopt-checkpoint"


def test_factory_keeps_tp_local_geometry_in_native_codec():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
            return_value={10: (0.5, 0.25)},
        ) as load_scales,
    ):
        _manager().create_cold_page_codec(
            _cache_config((0, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    load_scales.assert_called_once_with("/modelopt-checkpoint", (10,))
    assert native.create_nvfp4_cold_page_codec.call_args.args[0][0].num_kv_heads == 4


def test_control_plane_manager_does_not_register_a_late_codec():
    backend = MagicMock()
    v2_manager = _v2_manager(backend)
    manager = _manager()
    manager.bind_kv_cache_manager(v2_manager)

    assert manager.config.quant == "nvfp4"
    assert manager.config.target_cache_tier == "host"
    assert manager.kv_cache_manager is v2_manager
    assert not hasattr(manager, "_native_codec")
    backend.set_cold_page_codec.assert_not_called()


@pytest.mark.parametrize(
    "quant_config",
    [
        {"quantization": {"kv_cache_quant_algo": "NVFP4"}},
        {
            "producer": {"name": "modelopt"},
            "quant_method": "modelopt",
            "kv_cache_scheme": "NVFP4",
        },
    ],
)
def test_loads_standard_modelopt_scales_and_takes_duplicate_max(tmp_path, quant_config):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config))
    safetensors_torch.save_file(
        {
            "model.layers.4.self_attn.k_proj.k_scale": torch.tensor(0.5),
            "model.layers.4.self_attn.v_proj.v_scale": torch.tensor(0.25),
            "model.layers.10.self_attn.k_proj.k_scale": torch.tensor(0.125),
            "model.layers.10.self_attn.k_proj.extra.k_scale": torch.tensor(0.75),
            "model.layers.10.self_attn.v_proj.v_scale": torch.tensor(0.0625),
            "model.layers.10.mlp.k_scale": torch.tensor(99.0),
        },
        str(tmp_path / "model.safetensors"),
    )

    assert _load_nvfp4_scales(str(tmp_path), (4, 10)) == {
        4: (0.5, 0.25),
        10: (0.75, 0.0625),
    }


def test_rejects_missing_or_invalid_modelopt_scales(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"kv_cache_quant_algo": "NVFP4"}})
    )
    safetensors_torch.save_file(
        {"model.layers.0.self_attn.k_proj.k_scale": torch.tensor(0.0)},
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(RuntimeError, match="finite and positive"):
        _load_nvfp4_scales(str(tmp_path), (0,))


def test_rejects_non_nvfp4_modelopt_checkpoint(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"kv_cache_quant_algo": "FP8"}})
    )
    with pytest.raises(RuntimeError, match="kv_cache_quant_algo=NVFP4"):
        _load_nvfp4_scales(str(tmp_path), (0,))


def test_hybrid_manager_compresses_attention_and_skips_ssm_buffers():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
            return_value={4: (0.5, 0.25)},
        ),
    ):
        _manager().create_cold_page_codec(
            _cache_config((0, ("ssm_state", "conv_state"), "ssm"), (1, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.layer_id == 1


def test_ssm_only_pipeline_rank_builds_lossless_native_codec():
    native, codec = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales"
        ) as load_scales,
    ):
        result = _manager().create_cold_page_codec(
            _cache_config((0, ("ssm_state", "conv_state"), "ssm")),
            # Layer kind is authoritative. An SSM-only rank never enters the
            # Attention dtype gate, even when its recurrent-state dtype is not
            # one of the NVFP4 Attention kernels' source types.
            runtime_dtype=DataType.INT8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(0,),
            head_dim_per_layer=(128,),
        )

    assert result is codec
    native.create_nvfp4_cold_page_codec.assert_called_once_with([])
    load_scales.assert_not_called()


def test_rejects_missing_attention_buffer_roles_when_heads_are_present():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales"
        ) as load_scales,
        pytest.raises(RuntimeError, match="must contain both key and value buffers"),
    ):
        _manager().create_cold_page_codec(
            _cache_config((0, ("unknown",))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    native.create_nvfp4_cold_page_codec.assert_not_called()
    load_scales.assert_not_called()


def test_rejects_one_malformed_attention_layer_in_a_hybrid_config():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales"
        ) as load_scales,
        pytest.raises(RuntimeError, match="Attention layer 2 must contain both key and value"),
    ):
        _manager().create_cold_page_codec(
            _cache_config(
                (0, ("key", "value")),
                (1, ("ssm_state", "conv_state"), "ssm"),
                (2, ("key",)),
            ),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 11, 12),
            num_kv_heads_per_layer=(8, 8, 8),
            head_dim_per_layer=(128, 128, 128),
        )

    native.create_nvfp4_cold_page_codec.assert_not_called()
    load_scales.assert_not_called()


def test_fp8_runtime_uses_trtllm_unit_source_scale_contract():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
            return_value={10: (0.5, 0.25)},
        ),
    ):
        _manager().create_cold_page_codec(
            _cache_config((0, ("key", "value"))),
            runtime_dtype=DataType.FP8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.fp8_scale_orig_quant == (1.0, 1.0)
    assert native_config.fp8_scale_quant_orig == (1.0, 1.0)
