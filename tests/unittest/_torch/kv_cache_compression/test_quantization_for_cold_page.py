# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for cold-page quantization's native codec ownership."""

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


def _manager(model_source=None, **kwargs):
    return ColdPageQuantizationCompression(
        _config(**kwargs),
        model_source,
    )


def _scale_tensor_names(layer_id, prefix="model"):
    base = f"{prefix}.layers.{layer_id}.self_attn"
    return (
        f"{base}.k_proj.k_scale",
        f"{base}.v_proj.v_scale",
    )


def _write_scales(
    directory,
    scales_by_layer,
    *,
    filename="model.safetensors",
    prefix="model",
):
    tensors = {}
    for layer_id, (k_scale, v_scale) in scales_by_layer.items():
        k_name, v_name = _scale_tensor_names(layer_id, prefix)
        tensors[k_name] = torch.as_tensor(k_scale, dtype=torch.float32)
        tensors[v_name] = torch.as_tensor(v_scale, dtype=torch.float32)
    path = directory / filename
    save_file(tensors, str(path))
    return path


def _write_safetensors_index(directory, weight_map):
    index_path = directory / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}))
    return index_path


def _write_modelopt_kv_config(directory, algorithm, *, inline=False):
    """Write a real KV-only ModelOpt config in either supported location."""

    value = getattr(algorithm, "value", algorithm)
    if inline:
        payload = {
            "quantization_config": {
                "producer": {"name": "modelopt"},
                "quant_method": "modelopt",
                "quant_algo": None,
                "kv_cache_scheme": value,
                "config_groups": {},
            }
        }
        path = directory / "config.json"
    else:
        payload = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": None,
                "kv_cache_quant_algo": value,
                "quantized_layers": {},
            },
        }
        path = directory / "hf_quant_config.json"
    path.write_text(json.dumps(payload))
    return path


def test_factory_maps_pp_local_layers_to_checkpoint_scales(tmp_path):
    native, codec = _native()
    _write_modelopt_kv_config(tmp_path, "NVFP4")
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
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
    ):
        manager = _manager(tmp_path)
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


def test_manager_snapshots_checkpoint_scales_and_uses_identity_for_missing_pp_layer(
    tmp_path,
):
    native, _ = _native()
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    checkpoint = _write_scales(tmp_path, {10: (0.5, 0.25)})
    manager = _manager(tmp_path)

    # Construction copies immutable Python scalars. Codec creation does no
    # checkpoint I/O and chooses them by PP-global layer ID.
    checkpoint.unlink()
    assert set(manager._model_nvfp4_scales) == {10}
    assert all(
        isinstance(value, float) for pair in manager._model_nvfp4_scales[10] for value in pair
    )
    with pytest.raises(TypeError):
        manager._model_nvfp4_scales[10] = ((4.0, 8.0), (0.25, 0.125))

    with patch(
        "tensorrt_llm.bindings.internal.kv_cache_compression",
        new=native,
    ):
        manager.create_cold_page_codec(
            _cache_config((0, ("key", "value")), (1, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    native_configs = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert native_configs[0].nvfp4_scale_quant_orig == (0.5, 0.25)
    assert native_configs[0].nvfp4_scale_orig_quant == (2.0, 4.0)
    assert native_configs[1].nvfp4_scale_quant_orig == (1.0, 1.0)
    assert native_configs[1].nvfp4_scale_orig_quant == (1.0, 1.0)


def test_factory_keeps_tp_local_geometry_in_native_codec():
    native, _ = _native()
    cache_config = _cache_config((0, ("key", "value")))
    cache_config.tokens_per_block = 5
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
    ):
        _manager().create_cold_page_codec(
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


@pytest.mark.parametrize(
    ("algorithm", "inline"),
    [("nvfp4", False), ("NVFP4", True)],
)
def test_nvfp4_scale_provenance_uses_modelopt_file_or_inline_config(
    tmp_path,
    algorithm,
    inline,
):
    _write_modelopt_kv_config(tmp_path, algorithm, inline=inline)
    k_name, v_name = _scale_tensor_names(7)
    save_file(
        {k_name: torch.tensor(0.5), v_name: torch.tensor(0.25)},
        str(tmp_path / "model.safetensors"),
    )

    manager = _manager(tmp_path)
    assert manager.checkpoint_kv_cache_quant_algo == "NVFP4"
    assert manager._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


@pytest.mark.parametrize("algorithm", ["FP8", "INT8"])
def test_non_nvfp4_modelopt_scales_are_ignored(tmp_path, algorithm):
    _write_modelopt_kv_config(tmp_path, algorithm)
    k_name, v_name = _scale_tensor_names(7)
    save_file(
        {k_name: torch.tensor(0.5), v_name: torch.tensor(0.25)},
        str(tmp_path / "model.safetensors"),
    )

    manager = _manager(tmp_path)
    assert manager.checkpoint_kv_cache_quant_algo == algorithm
    assert manager._model_nvfp4_scales == {}


def test_hf_quant_config_is_authoritative_over_inline_modelopt_config(tmp_path):
    _write_modelopt_kv_config(tmp_path, "FP8")
    _write_modelopt_kv_config(tmp_path, "NVFP4", inline=True)
    _write_scales(tmp_path, {7: (0.5, 0.25)})

    manager = _manager(tmp_path)

    assert manager.checkpoint_kv_cache_quant_algo == "FP8"
    assert manager._model_nvfp4_scales == {}


def test_non_modelopt_inline_config_is_not_nvfp4_provenance(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "kv_cache_scheme": "NVFP4",
                }
            }
        )
    )
    _write_scales(tmp_path, {7: (0.5, 0.25)})

    with pytest.raises(ValueError, match="Ambiguous ModelOpt KV scale metadata"):
        _manager(tmp_path)


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_rejects_malformed_authoritative_modelopt_config(tmp_path, contents):
    (tmp_path / "hf_quant_config.json").write_text(contents)

    with pytest.raises(ValueError, match="[Cc]heckpoint metadata"):
        _manager(tmp_path)


def test_scale_metadata_without_algorithm_provenance_is_rejected(tmp_path):
    k_name, v_name = _scale_tensor_names(7)
    save_file(
        {k_name: torch.tensor(0.5), v_name: torch.tensor(0.25)},
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(ValueError, match="Ambiguous ModelOpt KV scale metadata"):
        _manager(tmp_path)


def test_checkpoint_without_scale_metadata_or_provenance_uses_identity(tmp_path):
    save_file(
        {"model.embed_tokens.weight": torch.ones(1)},
        str(tmp_path / "model.safetensors"),
    )

    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_nvfp4_checkpoint_without_scale_metadata_uses_identity(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    save_file(
        {"model.embed_tokens.weight": torch.ones(1)},
        str(tmp_path / "model.safetensors"),
    )

    manager = _manager(tmp_path)

    assert manager.checkpoint_kv_cache_quant_algo == "NVFP4"
    assert manager._model_nvfp4_scales == {}


def test_reads_one_scale_pair_across_normal_safetensors_shards(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, v_name = _scale_tensor_names(7)
    save_file(
        {k_name: torch.tensor(0.5)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {v_name: torch.tensor(0.25)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )

    manager = _manager(tmp_path)

    assert manager._model_nvfp4_scales[7] == ((2.0, 4.0), (0.5, 0.25))


def test_safetensors_index_selects_only_scale_metadata_shards(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, v_name = _scale_tensor_names(7)
    save_file(
        {k_name: torch.tensor(0.5)},
        str(tmp_path / "model-00001-of-00003.safetensors"),
    )
    save_file(
        {v_name: torch.tensor(0.25)},
        str(tmp_path / "model-00002-of-00003.safetensors"),
    )
    # This unreferenced ordinary shard contains conflicting matching metadata;
    # an index-backed load must not scan or open it.
    _write_scales(
        tmp_path,
        {7: (0.125, 0.0625)},
        filename="model-00003-of-00003.safetensors",
        prefix="model.language_model",
    )
    _write_safetensors_index(
        tmp_path,
        {
            k_name: "model-00001-of-00003.safetensors",
            v_name: "model-00002-of-00003.safetensors",
            "model.layers.7.self_attn.k_proj.weight": ("model-00003-of-00003.safetensors"),
        },
    )

    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_safetensors_index_rejects_missing_scale_shard(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, v_name = _scale_tensor_names(7)
    _write_safetensors_index(
        tmp_path,
        {
            k_name: "missing-00001-of-00002.safetensors",
            v_name: "missing-00002-of-00002.safetensors",
        },
    )

    with pytest.raises(FileNotFoundError, match="references missing scale shard"):
        _manager(tmp_path)


def test_safetensors_index_rejects_missing_indexed_tensor(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, v_name = _scale_tensor_names(7)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(
        {k_name: torch.tensor(0.5)},
        str(tmp_path / shard_name),
    )
    _write_safetensors_index(
        tmp_path,
        {k_name: shard_name, v_name: shard_name},
    )

    with pytest.raises(ValueError, match="tensor is absent from the shard"):
        _manager(tmp_path)


@pytest.mark.parametrize("contents", ["{", "[]", '{"metadata": {}}'])
def test_safetensors_index_rejects_malformed_index(tmp_path, contents):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    (tmp_path / "model.safetensors.index.json").write_text(contents)

    with pytest.raises(ValueError, match="[Ss]afetensors index"):
        _manager(tmp_path)


def test_hf_loader_consolidated_filtering(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    _write_scales(tmp_path, {7: (0.5, 0.25)}, filename="model.safetensors")
    # A normal shard exists, so the default HF-loader policy ignores this
    # otherwise conflicting consolidated copy.
    _write_scales(
        tmp_path,
        {7: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )

    consolidated_only = tmp_path / "consolidated_only"
    consolidated_only.mkdir()
    _write_modelopt_kv_config(consolidated_only, "NVFP4")
    checkpoint = _write_scales(
        consolidated_only,
        {9: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    # Direct paths and directories containing only consolidated files are both
    # valid checkpoint sources.
    expected = ((8.0, 16.0), (0.125, 0.0625))
    assert _manager(consolidated_only)._model_nvfp4_scales[9] == expected
    assert _manager(checkpoint)._model_nvfp4_scales[9] == expected


def test_trtllm_load_kv_scales_zero_skips_checkpoint_ingress(tmp_path, monkeypatch):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, _ = _scale_tensor_names(7)
    save_file({k_name: torch.tensor(0.5)}, str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "0")

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
        "_checkpoint_safetensor_inputs"
    ) as checkpoint_files:
        manager = _manager(tmp_path)

    checkpoint_files.assert_not_called()
    assert manager._model_nvfp4_scales == {}


@pytest.mark.parametrize(
    ("k_scale", "v_scale", "match"),
    [
        (torch.tensor([0.5, 0.25]), torch.tensor(0.25), "exactly one value"),
        (torch.tensor(float("nan")), torch.tensor(0.25), "finite and positive"),
        (torch.tensor(float("inf")), torch.tensor(0.25), "finite and positive"),
        (torch.tensor(0.0), torch.tensor(0.25), "finite and positive"),
        (torch.tensor(0.5), torch.tensor(-0.25), "finite and positive"),
    ],
)
def test_rejects_malformed_checkpoint_scale_metadata(
    tmp_path,
    k_scale,
    v_scale,
    match,
):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    _write_scales(tmp_path, {7: (k_scale, v_scale)})

    with pytest.raises(ValueError, match=match):
        _manager(tmp_path)


@pytest.mark.parametrize("present_kind", ["k", "v"])
def test_rejects_partial_checkpoint_scale_metadata(tmp_path, present_kind):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    k_name, v_name = _scale_tensor_names(7)
    name = k_name if present_kind == "k" else v_name
    save_file({name: torch.tensor(0.5)}, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match=r"layer 7 is partial.*matching [KV] scale"):
        _manager(tmp_path)


def test_rejects_duplicate_checkpoint_scale_metadata(tmp_path):
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    first_k, first_v = _scale_tensor_names(7, "model")
    second_k, second_v = _scale_tensor_names(7, "model.language_model")
    save_file(
        {first_k: torch.tensor(0.5), first_v: torch.tensor(0.25)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {second_k: torch.tensor(0.5), second_v: torch.tensor(0.25)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )

    with pytest.raises(ValueError, match="Duplicate ModelOpt K scale metadata"):
        _manager(tmp_path)


def test_control_plane_provider_is_construction_only():
    manager = _manager()
    assert manager.config.quant == "nvfp4"
    assert manager._model_nvfp4_scales == {}
    assert not hasattr(manager, "_attention_layers")
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


def test_cold_page_codec_provider_admission_fails_closed(monkeypatch):
    manager = _manager()

    # Backend selection occurs when the runtime module is imported. A later
    # environment change must not make admission disagree with the loaded API.
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "python")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp")
    with (
        patch.object(manager, "validate_runtime_support") as validate_runtime,
        pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"),
    ):
        v2_mod._validate_cold_page_codec_provider(manager)
    validate_runtime.assert_not_called()

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setenv("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "python")
    with (
        patch("tensorrt_llm._utils.is_sm_100f", return_value=False),
        pytest.raises(RuntimeError, match="requires an SM100-family device"),
    ):
        v2_mod._validate_cold_page_codec_provider(manager)
    with patch("tensorrt_llm._utils.is_sm_100f", return_value=True):
        v2_mod._validate_cold_page_codec_provider(manager)


def test_build_managers_scopes_cold_page_quantization_to_final_target(tmp_path):
    compression_config = _config()
    resolved_snapshot = tmp_path / "resolved-snapshot"
    resolved_snapshot.mkdir()
    target_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                is_generation=True,
                pretrained_config=SimpleNamespace(
                    num_hidden_layers=2,
                    _name_or_path=str(resolved_snapshot),
                ),
                quant_config=SimpleNamespace(kv_cache_quant_algo=None),
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
                    model="org/model-from-hub",
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
    final_constructor.assert_called_once_with(
        compression_config,
        str(resolved_snapshot),
    )

    estimation_calls, estimation_constructor = run_build(estimating=True)
    assert len(estimation_calls) == 3
    assert all(call.get("cold_page_codec_provider") is None for call in estimation_calls)
    estimation_constructor.assert_not_called()

    target_engine.model.model_config.quant_config.kv_cache_quant_algo = "NVFP4"
    active_nvfp4_calls, active_nvfp4_constructor = run_build(estimating=False)
    assert active_nvfp4_calls[0]["cold_page_codec_provider"] is None
    active_nvfp4_constructor.assert_not_called()


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
        patch("tensorrt_llm._utils.is_sm_100f", return_value=True),
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


def test_hybrid_manager_compresses_attention_and_skips_ssm_buffers(tmp_path):
    native, _ = _native()
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    _write_scales(tmp_path, {4: (0.5, 0.25)})
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
    ):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, ("ssm_state", "conv_state"), "ssm"), (1, ("key", "value"))),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(8, 8),
            head_dim_per_layer=(128, 128),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.layer_id == 1
    assert native_config.nvfp4_scale_orig_quant == (2.0, 4.0)


def test_ssm_only_pipeline_rank_builds_lossless_native_codec():
    native, codec = _native()
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
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


def test_rejects_missing_attention_buffer_roles_when_heads_are_present():
    native, _ = _native()
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
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


def test_rejects_one_malformed_attention_layer_in_a_hybrid_config():
    native, _ = _native()
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
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


def test_fp8_runtime_uses_trtllm_unit_source_scale_contract(tmp_path):
    native, _ = _native()
    _write_modelopt_kv_config(tmp_path, "NVFP4")
    _write_scales(tmp_path, {10: (0.5, 0.25)})
    with (
        patch(
            "tensorrt_llm.bindings.internal.kv_cache_compression",
            new=native,
        ),
    ):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, ("key", "value"))),
            runtime_dtype=DataType.FP8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    native_config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert native_config.fp8_scale_orig_quant == (1.0, 1.0)
    assert native_config.fp8_scale_quant_orig == (1.0, 1.0)
    assert native_config.nvfp4_scale_quant_orig == (0.5, 0.25)
