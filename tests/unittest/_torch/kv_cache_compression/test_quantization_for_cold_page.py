# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for cold-page quantization's native codec ownership."""

import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page import (
    ColdPageQuantizationCompression,
    _model_nvfp4_scales,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as v2_mod
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    ResourceManagerType,
)
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, KvCacheConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime_v2_mod
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)

_MISSING = object()


class _AttentionLayer:
    def __init__(self, kv_scales=_MISSING, inv_kv_scales=_MISSING):
        self.qkv_proj = SimpleNamespace()
        if kv_scales is not _MISSING:
            self.qkv_proj.kv_scales = kv_scales
        if inv_kv_scales is not _MISSING:
            self.qkv_proj.inv_kv_scales = inv_kv_scales


def _attention_registry(scales_by_layer):
    owners = {}
    registry = {}
    for layer_id, (kv_scales, inv_kv_scales) in scales_by_layer.items():
        layer = _AttentionLayer(kv_scales, inv_kv_scales)
        owners[layer_id] = layer
        registry[str(layer_id)] = weakref.ref(layer)
    return registry, owners


def _config(**kwargs):
    return ColdPageQuantizationCompressionConfig(**kwargs)


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


def _manager(attention_layers=None, **kwargs):
    return ColdPageQuantizationCompression(
        _config(**kwargs),
        {} if attention_layers is None else attention_layers,
    )


def test_factory_maps_pp_local_layers_to_model_owned_scales():
    native, codec = _native()
    attention_layers, owners = _attention_registry(
        {
            10: (
                torch.tensor([1.0, 0.5, 0.25]),
                torch.tensor([1.0, 2.0, 4.0]),
            ),
            4: (
                torch.tensor([1.0, 0.125, 0.0625]),
                torch.tensor([1.0, 8.0, 16.0]),
            ),
        }
    )
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
    ):
        manager = _manager(attention_layers)
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
    assert set(owners) == {4, 10}


def test_factory_keeps_tp_local_geometry_in_native_codec():
    native, _ = _native()
    cache_config = _cache_config((0, ("key", "value")))
    cache_config.tokens_per_block = 5
    attention_layers, owners = _attention_registry(
        {
            10: (
                torch.tensor([1.0, 0.5, 0.25]),
                torch.tensor([1.0, 2.0, 4.0]),
            )
        }
    )
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
    ):
        _manager(attention_layers).create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.num_kv_heads == 4
    assert native_config.tokens_per_page == 5
    assert native_config.head_dim == 128
    assert owners[10].qkv_proj.kv_scales.tolist() == [1.0, 0.5, 0.25]


def test_reads_calibrated_model_owned_kv_scales():
    attention_layers, owners = _attention_registry(
        {
            7: (
                torch.tensor([1.0, 0.5, 0.25]),
                torch.tensor([1.0, 2.0, 4.0]),
            )
        }
    )

    assert _model_nvfp4_scales(attention_layers, 7) == (
        (2.0, 4.0),
        (0.5, 0.25),
    )
    assert owners[7].qkv_proj.kv_scales[0] == 1.0


@pytest.mark.parametrize("skip_create_weights_in_init", [False, True])
def test_attention_requests_native_qkv_scale_loading_for_cold_pages(
    skip_create_weights_in_init,
):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.attention import Attention

    model_config = ModelConfig(
        kv_cache_compression_config=_config(),
        skip_create_weights_in_init=skip_create_weights_in_init,
    )
    attention = Attention(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bias=False,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=model_config,
    )

    assert attention.qkv_proj.quant_config.kv_cache_quant_algo is None
    assert attention.qkv_proj.kv_scales.tolist() == [1.0, 1.0, 1.0]
    assert attention.qkv_proj.inv_kv_scales.tolist() == [1.0, 1.0, 1.0]
    assert _model_nvfp4_scales(model_config.extra_attrs["attn_layers"], 0) == (
        (1.0, 1.0),
        (1.0, 1.0),
    )
    if skip_create_weights_in_init:
        attention.qkv_proj.create_weights()
    weights = [
        {},
        {"k_scale": torch.tensor(0.5)},
        {"v_scale": torch.tensor(0.25)},
    ]
    shards = tuple(
        torch.zeros(size, attention.qkv_proj.in_features)
        for _, size in attention.qkv_proj.fused_weight_shard_indices_mapping.values()
    )

    with patch(
        "tensorrt_llm._torch.modules.linear.load_weights_fused_qkv_helper",
        return_value=shards,
    ):
        attention.qkv_proj.load_weights(weights)

    assert attention.qkv_proj.kv_scales.tolist() == [1.0, 0.5, 0.25]
    assert attention.qkv_proj.inv_kv_scales.tolist() == [1.0, 2.0, 4.0]
    assert _model_nvfp4_scales(model_config.extra_attrs["attn_layers"], 0) == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_attention_does_not_allocate_nvfp4_scales_without_cold_page_config():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.attention import Attention

    attention = Attention(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bias=False,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=ModelConfig(),
    )

    assert not hasattr(attention.qkv_proj, "kv_scales")
    assert not hasattr(attention.qkv_proj, "inv_kv_scales")


def test_rejects_missing_model_kv_scale_tensors():
    attention_layers, owners = _attention_registry({7: (_MISSING, _MISSING)})

    with pytest.raises(RuntimeError, match="must own both kv_scales and inv_kv_scales"):
        _model_nvfp4_scales(attention_layers, 7)
    assert owners[7].qkv_proj is not None


@pytest.mark.parametrize(
    ("kv_scales", "inv_kv_scales"),
    [
        (torch.tensor([1.0, 0.5, 0.25]), _MISSING),
        (_MISSING, torch.tensor([1.0, 2.0, 4.0])),
    ],
)
def test_rejects_partial_model_kv_scale_ownership(kv_scales, inv_kv_scales):
    attention_layers, owners = _attention_registry({7: (kv_scales, inv_kv_scales)})

    with pytest.raises(RuntimeError, match="must own both kv_scales and inv_kv_scales"):
        _model_nvfp4_scales(attention_layers, 7)
    assert owners[7].qkv_proj is not None


@pytest.mark.parametrize(
    ("kv_scales", "inv_kv_scales", "match"),
    [
        (
            torch.tensor([1.0, 0.5]),
            torch.tensor([1.0, 2.0, 4.0]),
            r"kv_scales must contain \[Q, K, V\]",
        ),
        (
            torch.tensor([1.0, 0.5, 0.25]),
            torch.tensor([1.0, 2.0]),
            r"inv_kv_scales must contain \[Q, K, V\]",
        ),
        (
            torch.tensor([1.0, 0.0, 0.25]),
            torch.tensor([1.0, 2.0, 4.0]),
            "kv_scales K/V values must be finite and positive",
        ),
        (
            torch.tensor([1.0, 0.5, 0.25]),
            torch.tensor([1.0, float("inf"), 4.0]),
            "inv_kv_scales K/V values must be finite and positive",
        ),
    ],
)
def test_rejects_malformed_model_kv_scale_tensors(
    kv_scales,
    inv_kv_scales,
    match,
):
    attention_layers, owners = _attention_registry({7: (kv_scales, inv_kv_scales)})
    with pytest.raises(RuntimeError, match=match):
        _model_nvfp4_scales(attention_layers, 7)
    assert owners[7].qkv_proj is not None


def test_control_plane_provider_is_construction_only():
    attention_layers = {}
    manager = _manager(attention_layers)
    assert manager.config.quant == "nvfp4"
    assert manager._attention_layers is attention_layers
    assert not hasattr(manager, "_native_codec")
    assert not hasattr(manager, "kv_cache_manager")


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


def test_resolved_v1_manager_cannot_drop_cold_page_quantization():
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
            cold_page_codec_provider=_manager(),
        )


def test_cold_page_codec_requires_cpp_backend(monkeypatch):
    manager = _manager()

    # Backend selection occurs when the runtime module is imported. A later
    # environment change must not make admission disagree with the loaded API.
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "python")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp")
    with pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"):
        v2_mod._validate_cold_page_codec_backend(manager)

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "python")
    v2_mod._validate_cold_page_codec_backend(manager)


def test_build_managers_scopes_cold_page_quantization_to_final_target():
    attention_layers = {}
    compression_config = _config()
    target_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                is_generation=True,
                pretrained_config=SimpleNamespace(num_hidden_layers=2),
                extra_attrs={"attn_layers": attention_layers},
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
    cold_page_manager = object()

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
                    kv_cache_compression_config=compression_config,
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
                "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
                "ColdPageQuantizationCompression",
                return_value=cold_page_manager,
            ) as manager_constructor,
        ):
            creator.build_managers({}, estimating_kv_cache=estimating)

        return manager_calls, manager_constructor

    final_calls, final_constructor = run_build(estimating=False)
    assert len(final_calls) == 3
    assert final_calls[0]["model_engine"] is target_engine
    assert final_calls[1]["model_engine"] is draft_engine
    assert final_calls[2]["kv_cache_type"] == CacheTypeCpp.CROSS
    assert final_calls[0]["cold_page_codec_provider"] is cold_page_manager
    assert final_calls[1]["cold_page_codec_provider"] is None
    assert final_calls[2].get("cold_page_codec_provider") is None
    final_constructor.assert_called_once_with(compression_config, attention_layers)

    estimation_calls, estimation_constructor = run_build(estimating=True)
    assert len(estimation_calls) == 3
    assert all(call.get("cold_page_codec_provider") is None for call in estimation_calls)
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
            pretrained_config=model_engine.model.model_config.pretrained_config,
        )
        assert registered[ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER] is iteration_manager
    else:
        compression_factory.assert_not_called()
        assert ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in registered


def test_codec_enabled_construction_failure_is_not_retried_gpu_only(monkeypatch):
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

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(v2_mod, "KVCacheOutOfMemoryError", NativeConstructionError)
    with (
        patch.object(KVCacheManagerV2, "_build_base_config", return_value=object()),
        patch.object(KVCacheManagerV2, "_build_cache_config", return_value=object()),
        patch.object(
            v2_mod,
            "_create_kv_cache_manager_v2_impl",
            side_effect=NativeConstructionError("codec construction failed"),
        ) as native_constructor,
        pytest.raises(NativeConstructionError, match="codec construction failed"),
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
            cold_page_codec_provider=_manager(),
        )

    native_constructor.assert_called_once()


def test_hybrid_manager_compresses_attention_and_skips_ssm_buffers():
    native, _ = _native()
    attention_layers, owners = _attention_registry(
        {
            4: (
                torch.tensor([1.0, 0.5, 0.25]),
                torch.tensor([1.0, 2.0, 4.0]),
            )
        }
    )
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
    ):
        _manager(attention_layers).create_cold_page_codec(
            _cache_config((0, ("ssm_state", "conv_state"), "ssm"), (1, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.layer_id == 1
    assert owners[4].qkv_proj.inv_kv_scales.tolist() == [1.0, 2.0, 4.0]


def test_ssm_only_pipeline_rank_builds_lossless_native_codec():
    native, codec = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_model_nvfp4_scales"
        ) as model_scales,
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
    model_scales.assert_not_called()


def test_rejects_missing_attention_buffer_roles_when_heads_are_present():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_model_nvfp4_scales"
        ) as model_scales,
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
    model_scales.assert_not_called()


def test_rejects_one_malformed_attention_layer_in_a_hybrid_config():
    native, _ = _native()
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_model_nvfp4_scales"
        ) as model_scales,
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
    model_scales.assert_not_called()


def test_fp8_runtime_uses_trtllm_unit_source_scale_contract():
    native, _ = _native()
    attention_layers, owners = _attention_registry(
        {
            10: (
                torch.tensor([1.0, 0.5, 0.25]),
                torch.tensor([1.0, 2.0, 4.0]),
            )
        }
    )
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
            "_load_native_bindings",
            return_value=native,
        ),
    ):
        _manager(attention_layers).create_cold_page_codec(
            _cache_config((0, ("key", "value"))),
            runtime_dtype=DataType.FP8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.fp8_scale_orig_quant == (1.0, 1.0)
    assert native_config.fp8_scale_quant_orig == (1.0, 1.0)
    assert owners[10].qkv_proj.kv_scales.tolist() == [1.0, 0.5, 0.25]
