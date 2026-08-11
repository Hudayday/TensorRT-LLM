# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for QuantizationCompression's native codec ownership."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationCompression, _load_nvfp4_scales)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.llmapi.llm_args import QuantizationCompressionConfig


def _v2_manager(backend,
                *,
                num_layers=2,
                dtype=DataType.BF16,
                local_heads=8):
    """Build a real V2 instance without invoking its heavyweight constructor."""
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.impl = backend
    manager.dtype = dtype
    manager.num_local_layers = num_layers
    manager.pp_layers = (10, 4)[:num_layers]
    manager.num_kv_heads_per_layer = (local_heads, ) * num_layers
    manager.head_dim_per_layer = (128, ) * num_layers
    manager.tokens_per_block = 64
    manager.kv_compression_manages_history = False
    return manager


def _config(path="/modelopt-checkpoint", **kwargs):
    return QuantizationCompressionConfig(scale_checkpoint_path=path, **kwargs)


def _pool_group_desc(*layers):
    buffer_ids = [
        SimpleNamespace(layer_id=layer_id, role=role)
        for layer_id, roles in layers for role in roles
    ]
    return SimpleNamespace(slot_desc=SimpleNamespace(variants=(SimpleNamespace(
        coalesced_buffers=(SimpleNamespace(
            buffer_ids=tuple(buffer_ids)), )), )))


def _native():
    codec = MagicMock()
    codec.configure.return_value = True
    codec.query_cold_page_bytes.return_value = 1
    module = SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageLayerConfig=SimpleNamespace,
        Nvfp4ColdPageCodec=MagicMock(return_value=codec),
    )
    return module, codec


def _backend(*pool_groups):
    return SimpleNamespace(
        pool_group_descs=pool_groups,
        get_layer_group_id=lambda layer_id: layer_id + 3,
        set_cold_page_codec=MagicMock(),
    )


def test_factory_configures_registers_and_detaches_one_native_codec():
    native, codec = _native()
    backend = _backend(
        _pool_group_desc((0, ("key", "value")), (1, ("key", "value"))))
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
                return_value={
                    10: (0.5, 0.25),
                    4: (0.125, 0.0625)
                },
            ),
    ):
        manager = util_mod.create_kv_cache_compression_manager(
            _config(), _v2_manager(backend))

    assert isinstance(manager, QuantizationCompression)
    native_configs = native.Nvfp4ColdPageCodec.call_args.args[0]
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
    codec.configure.assert_called_once_with(backend.pool_group_descs[0])
    backend.set_cold_page_codec.assert_called_once_with(codec)
    manager.shutdown()
    assert backend.set_cold_page_codec.call_args_list[-1].args == (None, )


def test_factory_keeps_tp_local_geometry_in_native_codec():
    native, _ = _native()
    backend = _backend(_pool_group_desc((0, ("key", "value"))))
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
                return_value={10: (0.5, 0.25)},
            ) as load_scales,
    ):
        util_mod.create_kv_cache_compression_manager(
            _config(), _v2_manager(backend, num_layers=1, local_heads=4))

    load_scales.assert_called_once_with("/modelopt-checkpoint", (10, ))
    assert native.Nvfp4ColdPageCodec.call_args.args[0][0].num_kv_heads == 4


def test_factory_rejects_zero_stride_before_registration():
    native, codec = _native()
    codec.query_cold_page_bytes.return_value = 0
    backend = _backend(_pool_group_desc((0, ("key", "value"))))
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
                return_value={10: (0.5, 0.25)},
            ),
            pytest.raises(RuntimeError,
                          match="did not configure expected K/V layer groups"),
    ):
        util_mod.create_kv_cache_compression_manager(
            _config(), _v2_manager(backend, num_layers=1))

    backend.set_cold_page_codec.assert_not_called()


def test_requires_explicit_kv_cache_manager_v2():
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2=True"):
        util_mod.validate_kv_cache_compression_compatibility(
            _config(),
            SimpleNamespace(enable_block_reuse=True,
                            use_kv_cache_manager_v2="auto"),
            None,
        )


@pytest.mark.parametrize(
    "quant_config",
    [
        {
            "quantization": {
                "kv_cache_quant_algo": "NVFP4"
            }
        },
        {
            "producer": {
                "name": "modelopt"
            },
            "quant_method": "modelopt",
            "kv_cache_scheme": "NVFP4",
        },
    ],
)
def test_loads_standard_modelopt_scales_and_takes_duplicate_max(
        tmp_path, quant_config):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config))
    safetensors_torch.save_file(
        {
            "model.layers.4.self_attn.k_proj.k_scale": torch.tensor(0.5),
            "model.layers.4.self_attn.v_proj.v_scale": torch.tensor(0.25),
            "model.layers.10.self_attn.k_proj.k_scale": torch.tensor(0.125),
            "model.layers.10.self_attn.k_proj.extra.k_scale":
            torch.tensor(0.75),
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
        json.dumps({"quantization": {
            "kv_cache_quant_algo": "NVFP4"
        }}))
    safetensors_torch.save_file(
        {"model.layers.0.self_attn.k_proj.k_scale": torch.tensor(0.0)},
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(RuntimeError, match="finite and positive"):
        _load_nvfp4_scales(str(tmp_path), (0, ))


def test_rejects_non_nvfp4_modelopt_checkpoint(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {
            "kv_cache_quant_algo": "FP8"
        }}))
    with pytest.raises(RuntimeError, match="kv_cache_quant_algo=NVFP4"):
        _load_nvfp4_scales(str(tmp_path), (0, ))


def test_hybrid_manager_compresses_attention_and_skips_ssm_buffers():
    native, _ = _native()
    backend = _backend(
        _pool_group_desc((0, ("ssm_state", "conv_state"))),
        _pool_group_desc((1, ("key", "value"))),
    )
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
                return_value={4: (0.5, 0.25)},
            ),
    ):
        util_mod.create_kv_cache_compression_manager(_config(),
                                                     _v2_manager(backend))

    native_config = native.Nvfp4ColdPageCodec.call_args.args[0][0]
    assert native_config.layer_id == 1
    assert native_config.layer_group_id == 4


def test_fp8_runtime_uses_trtllm_unit_source_scale_contract():
    native, _ = _native()
    backend = _backend(_pool_group_desc((0, ("key", "value"))))
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_nvfp4_scales",
                return_value={10: (0.5, 0.25)},
            ),
    ):
        util_mod.create_kv_cache_compression_manager(
            _config(),
            _v2_manager(backend, num_layers=1, dtype=DataType.FP8))

    native_config = native.Nvfp4ColdPageCodec.call_args.args[0][0]
    assert native_config.fp8_scale_orig_quant == (1.0, 1.0)
    assert native_config.fp8_scale_quant_orig == (1.0, 1.0)
