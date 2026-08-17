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
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as v2_mod
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    ResourceManagerType,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, QuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
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
    cache_config = _cache_config((0, ("key", "value")))
    cache_config.tokens_per_block = 5
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
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    load_scales.assert_called_once_with("/modelopt-checkpoint", (10,))
    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.num_kv_heads == 4
    assert native_config.tokens_per_page == 5
    assert native_config.head_dim == 128


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


def test_codec_is_created_before_and_transferred_into_native_constructor():
    calls = []
    codec = object()
    cache_config = object()
    compression_manager = _manager()

    def create_codec(*args, **kwargs):
        calls.append(("codec", args, kwargs))
        return codec

    def create_manager(*args, **kwargs):
        calls.append(("manager", args, kwargs))
        assert kwargs["cold_page_codec"] is codec
        return "native-manager"

    with (
        patch.object(
            compression_manager,
            "create_cold_page_codec",
            side_effect=create_codec,
        ),
        patch.object(v2_mod, "KVCacheManagerPy", side_effect=create_manager),
    ):
        result = v2_mod._create_kv_cache_manager_v2_impl(
            cache_config,
            "event-manager",
            compression_manager,
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    assert result == "native-manager"
    assert [call[0] for call in calls] == ["codec", "manager"]
    assert calls[1][1] == (cache_config,)
    assert calls[1][2]["event_manager"] == "event-manager"


def test_normal_kvcm_constructor_path_is_unchanged_without_compression():
    cache_config = object()
    with patch.object(v2_mod, "KVCacheManagerPy", return_value="native-manager") as constructor:
        result = v2_mod._create_kv_cache_manager_v2_impl(
            cache_config,
            "event-manager",
            None,
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    assert result == "native-manager"
    constructor.assert_called_once_with(cache_config, event_manager="event-manager")


def test_resolved_v1_manager_cannot_drop_boundary_compression():
    with pytest.raises(ValueError, match="resolved KV cache manager.*KVCacheManagerV2"):
        util_mod._create_kv_cache_manager(
            model_engine=None,
            kv_cache_manager_cls=object,
            mapping=None,
            kv_cache_config=None,
            tokens_per_block=0,
            max_seq_len=0,
            max_batch_size=0,
            spec_config=None,
            sparse_attention_config=None,
            max_num_tokens=0,
            max_beam_width=1,
            kv_connector_manager=None,
            kv_cache_compression_manager=_manager(),
        )


def test_boundary_codec_requires_cpp_backend_and_host_tier(monkeypatch):
    manager = _manager()
    gpu = GpuCacheTierConfig(quota=1 << 20)
    host = HostCacheTierConfig(quota=1 << 20)

    # Backend selection occurs when the runtime module is imported. A later
    # environment change must not make admission disagree with the loaded API.
    monkeypatch.setattr(v2_mod, "KV_CACHE_MANAGER_V2_BACKEND", "python")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp")
    with pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"):
        v2_mod._validate_cold_page_codec_storage(manager, [gpu, host])

    monkeypatch.setattr(v2_mod, "KV_CACHE_MANAGER_V2_BACKEND", "cpp")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "python")
    with pytest.raises(ValueError, match="positive KVCM V2 Host cache"):
        v2_mod._validate_cold_page_codec_storage(manager, [gpu])

    v2_mod._validate_cold_page_codec_storage(manager, [gpu, host])


def test_build_managers_scopes_boundary_compression_to_final_target():
    target_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                is_generation=True,
                pretrained_config=SimpleNamespace(num_hidden_layers=2),
            )
        ),
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
    )
    draft_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                is_generation=True,
                pretrained_config=SimpleNamespace(num_hidden_layers=2),
            )
        ),
        kv_cache_manager_key=ResourceManagerType.DRAFT_KV_CACHE_MANAGER,
    )
    boundary_manager = object()

    def run_build(estimating):
        with patch.object(
            util_mod.KvCacheCreator,
            "_get_model_kv_cache_manager_cls",
            return_value=KVCacheManagerV2,
        ):
            creator = util_mod.KvCacheCreator(
                model_engine=target_engine,
                draft_model_engine=draft_engine,
                mapping=object(),
                net_max_seq_len=512,
                kv_connector_manager=None,
                max_num_tokens=1024,
                max_beam_width=1,
                tokens_per_block=64,
                max_seq_len=512,
                max_batch_size=2,
                kv_cache_config=KvCacheConfig(),
                llm_args=SimpleNamespace(
                    kv_cache_compression_config=_config(),
                    cache_transceiver_config=None,
                ),
                speculative_config=None,
                sparse_attention_config=None,
                profiling_stage_data=None,
                is_disagg=False,
            )

        manager_calls = []

        def create_manager(**kwargs):
            manager_calls.append(kwargs)
            return SimpleNamespace(max_seq_len=512)

        with (
            patch.object(creator, "_is_encoder_decoder", return_value=True),
            patch.object(
                creator,
                "_split_kv_cache_budget_for_cross",
                return_value=(creator._kv_cache_config, KvCacheConfig()),
            ),
            patch.object(creator, "_needs_gpu_kv_cache_budget_split", return_value=False),
            patch.object(creator, "_should_create_separate_draft_kv_cache", return_value=False),
            patch.object(
                creator,
                "_get_model_kv_cache_manager_cls",
                return_value=KVCacheManagerV2,
            ),
            patch.object(creator, "_get_cross_kv_cache_layout", return_value=(2, 8, 128, 256)),
            patch.object(creator, "_enable_kv_cache_stats", return_value=False),
            patch.object(util_mod, "_create_kv_cache_manager", side_effect=create_manager),
            patch(
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary."
                "QuantizationCompression",
                return_value=boundary_manager,
            ) as manager_constructor,
        ):
            creator.build_managers({}, estimating_kv_cache=estimating)

        return manager_calls, manager_constructor

    final_calls, final_constructor = run_build(estimating=False)
    assert len(final_calls) == 3
    assert final_calls[0]["model_engine"] is target_engine
    assert final_calls[1]["model_engine"] is draft_engine
    assert final_calls[2]["kv_cache_type"] == CacheTypeCpp.CROSS
    assert final_calls[0]["kv_cache_compression_manager"] is boundary_manager
    assert final_calls[1]["kv_cache_compression_manager"] is None
    assert final_calls[2].get("kv_cache_compression_manager") is None
    final_constructor.assert_called_once()

    estimation_calls, estimation_constructor = run_build(estimating=True)
    assert len(estimation_calls) == 3
    assert all(call.get("kv_cache_compression_manager") is None for call in estimation_calls)
    estimation_constructor.assert_not_called()


@pytest.mark.parametrize(
    ("compression_config", "registers_iteration_manager"),
    [
        (_config(), False),
        (SimpleNamespace(algorithm="triattention"), True),
    ],
)
def test_executor_registers_only_iteration_driven_compression(
    compression_config,
    registers_iteration_manager,
):
    kv_cache_manager = MagicMock()
    resources = {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    iteration_manager = MagicMock()
    llm_args = SimpleNamespace(
        enable_low_latency_host_dispatch=False,
        extra_resource_managers={},
        kv_cache_compression_config=compression_config,
        disable_overlap_scheduler=True,
        reorder_policy_config=None,
        enable_early_first_token_response=False,
        kv_cache_config=SimpleNamespace(enable_kv_pool_rebalance=False),
    )
    mapping = SimpleNamespace(
        pp_size=1,
        enable_attention_dp=False,
        has_pp=lambda: False,
    )
    model_engine = SimpleNamespace(
        spec_config=None,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                pretrained_config=SimpleNamespace(),
                is_encoder_decoder=False,
            )
        ),
    )
    scheduler_config = SimpleNamespace(
        use_python_scheduler=True,
        capacity_scheduler_policy="max-utilization",
        enable_prefix_aware_scheduling=True,
        waiting_queue_policy="fcfs",
    )

    with (
        patch.object(util_mod, "set_low_latency_dispatch"),
        patch.object(util_mod, "SeqSlotManager", return_value=MagicMock()),
        patch.object(util_mod, "SimpleUnifiedScheduler", return_value=MagicMock()),
        patch.object(util_mod, "create_kv_cache_transceiver", return_value=None),
        patch.object(util_mod, "is_mla", return_value=False),
        patch.object(
            util_mod,
            "create_kv_cache_compression_manager",
            return_value=iteration_manager,
        ) as compression_factory,
        patch.object(util_mod, "PyExecutor", return_value=MagicMock()) as executor_constructor,
    ):
        util_mod.create_py_executor_instance(
            dist=MagicMock(),
            resources=resources,
            mapping=mapping,
            llm_args=llm_args,
            ctx_chunk_config=None,
            model_engine=model_engine,
            start_worker=False,
            sampler=MagicMock(),
            drafter=None,
            max_seq_len=512,
            max_batch_size=2,
            max_beam_width=1,
            max_num_tokens=1024,
            max_num_sequences=2,
            scheduler_config=scheduler_config,
        )

    registered = executor_constructor.call_args.args[0].resource_managers
    if registers_iteration_manager:
        compression_factory.assert_called_once_with(
            compression_config,
            kv_cache_manager,
            draft_kv_cache_manager=None,
        )
        assert registered[ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER] is iteration_manager
    else:
        compression_factory.assert_not_called()
        assert ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in registered


def test_codec_enabled_host_construction_failure_is_not_retried_gpu_only(monkeypatch):
    class NativeConstructionError(Exception):
        pass

    cache_config = KvCacheConfig(
        max_gpu_total_bytes=1 << 20,
        host_cache_size=1 << 20,
    )
    mapping = SimpleNamespace(
        cp_config={},
        tp_size=1,
        enable_attention_dp=False,
        world_size=1,
        pp_partition=None,
        pp_layers=lambda num_layers: list(range(num_layers)),
        is_last_pp_rank=lambda: True,
    )

    monkeypatch.setattr(v2_mod, "KV_CACHE_MANAGER_V2_BACKEND", "cpp")
    monkeypatch.setattr(v2_mod, "KVCacheOutOfMemoryError", NativeConstructionError)
    with (
        patch.object(KVCacheManagerV2, "_build_base_config", return_value=object()),
        patch.object(KVCacheManagerV2, "_build_cache_config", return_value=object()),
        patch.object(
            v2_mod,
            "_create_kv_cache_manager_v2_impl",
            side_effect=NativeConstructionError("host codec construction failed"),
        ) as native_constructor,
        pytest.raises(NativeConstructionError, match="host codec construction failed"),
    ):
        KVCacheManagerV2(
            cache_config,
            CacheTypeCpp.SELF,
            num_layers=1,
            num_kv_heads=8,
            head_dim=128,
            tokens_per_block=64,
            max_seq_len=512,
            max_batch_size=2,
            mapping=mapping,
            dtype=DataType.BF16,
            execution_stream=object(),
            kv_cache_compression_manager=_manager(),
        )

    native_constructor.assert_called_once()


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
