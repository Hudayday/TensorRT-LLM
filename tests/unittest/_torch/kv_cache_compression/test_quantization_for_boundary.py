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


class _NativeLayerConfig:
    pass


class _PythonKvcmBackend:

    def __init__(self):
        self.pool_group_descs = ("gpu-pg-0", )
        self.set_cold_page_codec = MagicMock()


class _CppKvcmBackend(_PythonKvcmBackend):
    """Test double for the nanobind backend with the identical setter ABI."""


def _native(*, cold_page_bytes=73728):
    codec = MagicMock()
    codec.configure.return_value = True
    codec.query_cold_page_bytes.return_value = cold_page_bytes
    codec.encode.return_value = True
    codec.decode.return_value = True
    module = SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageLayerConfig=_NativeLayerConfig,
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
    return manager, native, codec


def test_layer_config_rejects_invalid_geometry_and_scale():
    fields = vars(_layer_config()).copy()
    fields["head_dim"] = 127
    with pytest.raises(ValueError, match="head_dim"):
        Nvfp4BoundaryLayerConfig(**fields)

    fields = vars(_layer_config()).copy()
    fields["nvfp4_scale_orig_quant"] = (float("inf"), 1.0)
    with pytest.raises(ValueError, match="finite positive"):
        Nvfp4BoundaryLayerConfig(**fields)


def test_manager_creates_one_native_codec_and_keeps_its_lifetime():
    manager, native, codec = _manager(
        _layer_config(layer_id=1, runtime_dtype="fp8_e4m3"),
        _layer_config(layer_id=0),
    )

    native.Nvfp4ColdPageCodec.assert_called_once()
    native_configs = native.Nvfp4ColdPageCodec.call_args.args[0]
    assert [config.layer_id for config in native_configs] == [0, 1]
    assert native_configs[0].runtime_type == "native-fp16"
    assert native_configs[1].runtime_type == "native-fp8"
    assert manager.native_codec is codec


def test_configure_passes_only_authoritative_gpu_descriptors():
    manager, _, codec = _manager(_layer_config(),
                                 gpu_descs=("gpu-pg-0", "gpu-pg-1"))

    assert manager.native_codec is codec
    assert codec.configure.call_args_list[0].args == ("gpu-pg-0", )
    assert codec.configure.call_args_list[1].args == ("gpu-pg-1", )
    assert manager.query_cold_page_bytes(3) == 73728


def test_offload_passes_compact_base_indices_and_stream_without_expanding_addresses(
):
    manager, _, codec = _manager(_layer_config())

    manager.on_offload_compress(
        layer_group_id=3,
        dst_base_ptr=0x300000,
        dst_base_page_indices=(2, 5),
        src_base_page_indices=(1, 3),
        stream=0x7000,
    )

    codec.encode.assert_called_once_with(3, 0x300000, [2, 5], [1, 3], 0x7000)


def test_onboard_uses_the_same_codec_with_reversed_storage_roles():
    manager, _, codec = _manager(_layer_config(runtime_dtype="fp8_e4m3"))

    manager.on_onboard_decompress(
        layer_group_id=3,
        dst_base_page_indices=(1, 3),
        src_base_ptr=0x300000,
        src_base_page_indices=(2, 5),
        stream=0x7000,
    )

    codec.decode.assert_called_once_with(3, [1, 3], 0x300000, [2, 5], 0x7000)


def test_codec_submission_failure_is_fail_closed():
    manager, _, codec = _manager(_layer_config())
    codec.encode.return_value = False

    with pytest.raises(RuntimeError, match="encode submission failed"):
        manager.on_offload_compress(
            layer_group_id=3,
            dst_base_ptr=0x300000,
            dst_base_page_indices=(2, ),
            src_base_page_indices=(1, ),
            stream=0x7000,
        )


def test_factory_builds_and_configures_manager_for_kvcm_handoff():
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
            boundary_layer_configs=(_layer_config(), ),
        )

    assert isinstance(manager, QuantizationCompression)
    assert manager.native_codec is codec
    codec.configure.assert_called_once_with("gpu-pg-0")
    backend.set_cold_page_codec.assert_called_once_with(codec)


@pytest.mark.parametrize("backend_type", (_PythonKvcmBackend, _CppKvcmBackend))
def test_same_native_codec_registers_with_either_kvcm_v2_backend(backend_type):
    native, codec = _native()
    backend = backend_type()
    with patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
            return_value=native,
    ):
        manager = QuantizationCompression(
            _config(),
            _v2_manager(backend),
            layer_configs=(_layer_config(), ),
        )
        manager.configure(gpu_pool_group_descs=backend.pool_group_descs)
        manager.register_with_kv_cache_manager()

    backend.set_cold_page_codec.assert_called_once_with(codec)
    manager.shutdown()
    assert backend.set_cold_page_codec.call_args_list[-1].args == (None, )


def test_factory_fails_closed_until_kvcm_accepts_native_codec():
    native, _ = _native()
    with (
            patch.object(util_mod, "is_sm_100f", return_value=True),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
                return_value=native,
            ),
            pytest.raises(RuntimeError, match="set_cold_page_codec"),
    ):
        util_mod.create_kv_cache_compression_manager(
            _config(),
            _v2_manager(SimpleNamespace(pool_group_descs=("gpu-pg-0", ))),
            boundary_layer_configs=(_layer_config(), ),
        )


def test_unknown_quantization_does_not_create_an_nvfp4_codec():
    config = SimpleNamespace(
        quant="future_quant",
        target_cache_tier="host",
        changes_physical_kv_length=False,
    )
    with pytest.raises(RuntimeError, match="codec for 'future_quant'"):
        QuantizationCompression(
            config,
            _v2_manager(),
            layer_configs=(_layer_config(), ),
        )
