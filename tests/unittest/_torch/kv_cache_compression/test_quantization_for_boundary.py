# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for QuantizationCompression's native codec ownership."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    Nvfp4BoundaryLayerConfig,
    QuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.llmapi.llm_args import QuantizationCompressionConfig


def _v2_manager(backend=None):
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.kv_compression_manages_history = False
    if backend is not None:
        manager.impl = backend
    return manager


def _config():
    return QuantizationCompressionConfig()


def _layer_config(*, layer_id=0, layer_group_id=3, runtime_dtype="float16"):
    return Nvfp4BoundaryLayerConfig(
        layer_group_id=layer_group_id,
        layer_id=layer_id,
        num_kv_heads=8,
        tokens_per_page=64,
        head_dim=128,
        runtime_dtype=runtime_dtype,
        nvfp4_scale_orig_quant=(2.0, 4.0),
        nvfp4_scale_quant_orig=(0.5, 0.25),
        fp8_scale_orig_quant=(8.0, 16.0),
        fp8_scale_quant_orig=(0.125, 0.0625),
    )


def _native():
    codec = MagicMock()
    codec.configure.return_value = True
    codec.encode.return_value = True
    codec.decode.return_value = True
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


def _manager(*layer_configs, gpu_descs=("gpu-pg-0", )):
    native, codec = _native()
    with patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
    ):
        manager = QuantizationCompression(
            _config(),
            _v2_manager(),
            layer_configs=layer_configs,
        )
        manager.configure(gpu_pool_group_descs=gpu_descs)
    return manager, codec


def test_offload_passes_compact_base_indices_and_stream_without_expanding_addresses(
):
    manager, codec = _manager(_layer_config())

    manager.on_offload_compress(
        layer_group_id=3,
        dst_base_ptr=0x300000,
        dst_base_page_indices=(2, 5),
        src_base_page_indices=(1, 3),
        stream=0x7000,
    )

    codec.encode.assert_called_once_with(3, 0x300000, [2, 5], [1, 3], 0x7000)


def test_onboard_uses_the_same_codec_with_reversed_storage_roles():
    manager, codec = _manager(_layer_config(runtime_dtype="fp8_e4m3"))

    manager.on_onboard_decompress(
        layer_group_id=3,
        dst_base_page_indices=(1, 3),
        src_base_ptr=0x300000,
        src_base_page_indices=(2, 5),
        stream=0x7000,
    )

    codec.decode.assert_called_once_with(3, [1, 3], 0x300000, [2, 5], 0x7000)


def test_codec_submission_failure_is_fail_closed():
    manager, codec = _manager(_layer_config())
    codec.encode.return_value = False

    with pytest.raises(RuntimeError, match="encode submission failed"):
        manager.on_offload_compress(
            layer_group_id=3,
            dst_base_ptr=0x300000,
            dst_base_page_indices=(2, ),
            src_base_page_indices=(1, ),
            stream=0x7000,
        )


def test_factory_configures_registers_and_detaches_one_native_codec():
    native, codec = _native()
    backend = SimpleNamespace(
        pool_group_descs=("gpu-pg-0", ),
        set_cold_page_codec=MagicMock(),
    )
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
    ):
        manager = util_mod.create_kv_cache_compression_manager(
            _config(),
            _v2_manager(backend),
            boundary_layer_configs=(
                _layer_config(layer_id=1, runtime_dtype="fp8_e4m3"),
                _layer_config(layer_id=0),
            ),
        )

    assert isinstance(manager, QuantizationCompression)
    native.Nvfp4ColdPageCodec.assert_called_once()
    native_configs = native.Nvfp4ColdPageCodec.call_args.args[0]
    assert [config.layer_id for config in native_configs] == [0, 1]
    assert [config.runtime_type for config in native_configs] == [
        "native-fp16",
        "native-fp8",
    ]
    codec.configure.assert_called_once_with("gpu-pg-0")
    backend.set_cold_page_codec.assert_called_once_with(codec)
    manager.shutdown()
    assert backend.set_cold_page_codec.call_args_list[-1].args == (None, )


def test_factory_unit_scale_fallback_is_explicitly_gated():
    manager = _v2_manager(
        SimpleNamespace(
            pool_group_descs=("gpu-pg-0", ),
            get_layer_group_id=lambda _layer_id: 0,
            set_cold_page_codec=MagicMock(),
        ))
    manager.dtype = DataType.BF16
    manager.num_local_layers = 1
    manager.num_kv_heads_per_layer = (8, )
    manager.head_dim_per_layer = (128, )
    manager.tokens_per_block = 64

    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="mechanism E2E test"),
    ):
        util_mod.create_kv_cache_compression_manager(_config(), manager)
