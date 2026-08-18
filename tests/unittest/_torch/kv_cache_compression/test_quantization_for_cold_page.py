# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for NVFP4 cold-page compression."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page import (
    ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime_v2_mod
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)


def _manager(scale_checkpoint_path=None):
    config = ColdPageQuantizationCompressionConfig(
        scale_checkpoint_path=(
            str(scale_checkpoint_path) if scale_checkpoint_path is not None else None
        )
    )
    return ColdPageQuantizationCompression(config)


def _cache_config(*layers):
    configs = []
    for layer_id, kind in layers:
        layer_type = SsmLayerConfig if kind == "ssm" else AttentionLayerConfig
        roles = ("ssm_state", "conv_state") if kind == "ssm" else ("key", "value")
        configs.append(
            layer_type(
                layer_id=layer_id,
                buffers=[BufferConfig(role=role, size=128) for role in roles],
            )
        )
    return SimpleNamespace(tokens_per_block=64, layers=tuple(configs))


def _native():
    class NativeLayerConfig(SimpleNamespace):
        def __init__(self):
            super().__init__(
                fp8_scale_orig_quant=(1.0, 1.0),
                fp8_scale_quant_orig=(1.0, 1.0),
            )

    codec = MagicMock()
    module = SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageLayerConfig=NativeLayerConfig,
        create_nvfp4_cold_page_codec=MagicMock(return_value=codec),
    )
    return module, codec


def _write_scales(directory, scales_by_layer, *, filename="model.safetensors", prefix="model"):
    (directory / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {"kv_cache_quant_algo": "NVFP4"},
            }
        )
    )
    tensors = {}
    for layer_id, (k_scale, v_scale) in scales_by_layer.items():
        base = f"{prefix}.layers.{layer_id}.self_attn"
        tensors[f"{base}.k_proj.k_scale"] = torch.as_tensor(k_scale, dtype=torch.float32)
        tensors[f"{base}.v_proj.v_scale"] = torch.as_tensor(v_scale, dtype=torch.float32)
    save_file(tensors, str(directory / filename))


def test_optional_modelopt_scales_map_pp_layers_into_native_codec(tmp_path):
    native, codec = _native()
    _write_scales(
        tmp_path,
        {10: (0.5, 0.25)},
        filename="model-00001-of-00002.safetensors",
    )
    _write_scales(
        tmp_path,
        {4: (0.125, 0.0625)},
        filename="model-00002-of-00002.safetensors",
        prefix="model.language_model",
    )

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        result = _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "attention"), (1, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    assert result is codec
    configs = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert [config.layer_id for config in configs] == [0, 1]
    assert [config.runtime_type for config in configs] == ["native-bf16"] * 2
    assert [config.nvfp4_scale_quant_orig for config in configs] == [
        (0.5, 0.25),
        (0.125, 0.0625),
    ]
    assert [config.nvfp4_scale_orig_quant for config in configs] == [
        (2.0, 4.0),
        (8.0, 16.0),
    ]


def test_omitted_scale_checkpoint_uses_identity_and_keeps_kv_geometry():
    native, _ = _native()
    cache_config = _cache_config((0, "attention"))
    cache_config.tokens_per_block = 5

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.HALF,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert config.runtime_type == "native-fp16"
    assert config.num_kv_heads == 4
    assert config.tokens_per_page == 5
    assert config.head_dim == 128
    assert config.nvfp4_scale_orig_quant == (1.0, 1.0)
    assert config.nvfp4_scale_quant_orig == (1.0, 1.0)


def test_scale_loader_matches_hf_shard_and_consolidated_policy(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)}, filename="model.safetensors")
    _write_scales(
        tmp_path,
        {7: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )

    consolidated_only = tmp_path / "consolidated-only"
    consolidated_only.mkdir()
    _write_scales(
        consolidated_only,
        {9: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _manager(consolidated_only)._model_nvfp4_scales[9] == (
        (8.0, 16.0),
        (0.125, 0.0625),
    )


def test_scale_loader_reduces_duplicate_shards_like_native_qkv_loader(tmp_path):
    _write_scales(tmp_path, {7: (0.25, 0.125)}, filename="model-00001.safetensors")
    _write_scales(
        tmp_path,
        {7: (0.5, 0.25)},
        filename="model-00002.safetensors",
        prefix="model.language_model",
    )
    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_trtllm_load_kv_scales_zero_uses_identity(tmp_path, monkeypatch):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "0")
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_non_nvfp4_checkpoint_scales_are_not_reused(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {"kv_cache_quant_algo": "FP8"},
            }
        )
    )
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_unquantized_checkpoint_uses_identity_scales(tmp_path):
    save_file({"model.weight": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_explicit_scale_checkpoint_requires_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No safetensors files"):
        _manager(tmp_path)


@pytest.mark.parametrize("present_kind", ["k", "v"])
def test_scale_checkpoint_requires_kv_pair(tmp_path, present_kind):
    _write_scales(tmp_path, {7: (0.5, 0.5)})
    base = "model.layers.7.self_attn"
    name = f"{base}.{present_kind}_proj.{present_kind}_scale"
    save_file({name: torch.tensor(0.5)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="both K and V"):
        _manager(tmp_path)


def test_hybrid_codec_skips_ssm_layers_and_ssm_only_rank_is_lossless(tmp_path):
    native, codec = _native()
    _write_scales(tmp_path, {4: (0.5, 0.25)})
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "ssm"), (1, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(0, 8),
            head_dim_per_layer=(128, 128),
        )
        config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
        assert config.layer_id == 1
        assert config.nvfp4_scale_orig_quant == (2.0, 4.0)

        result = _manager().create_cold_page_codec(
            _cache_config((0, "ssm")),
            runtime_dtype=DataType.INT8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(0,),
            head_dim_per_layer=(128,),
        )

    assert result is codec
    native.create_nvfp4_cold_page_codec.assert_called_with([])


def test_mla_key_only_layout_is_rejected_before_native_codec_creation():
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=64,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role="key", size=128)],
            ),
        ),
    )
    with (
        patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native),
        pytest.raises(NotImplementedError, match="separate key and value"),
    ):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(128,),
        )

    native.create_nvfp4_cold_page_codec.assert_not_called()


def test_fp8_runtime_uses_native_unit_source_scale_default(tmp_path):
    native, _ = _native()
    _write_scales(tmp_path, {10: (0.5, 0.25)})
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.FP8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert config.fp8_scale_orig_quant == (1.0, 1.0)
    assert config.fp8_scale_quant_orig == (1.0, 1.0)
    assert config.nvfp4_scale_quant_orig == (0.5, 0.25)


def test_runtime_admission_is_checked_in_utils_before_manager_creation(monkeypatch):
    config = ColdPageQuantizationCompressionConfig()
    kv_cache_config = SimpleNamespace(enable_block_reuse=False)

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "python")
    with pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"):
        util_mod.validate_kv_cache_compression_compatibility(config, kv_cache_config, None)

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: False)
    with pytest.raises(RuntimeError, match="requires an SM100-family device"):
        util_mod.validate_kv_cache_compression_compatibility(config, kv_cache_config, None)

    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)
    util_mod.validate_kv_cache_compression_compatibility(config, kv_cache_config, None)
