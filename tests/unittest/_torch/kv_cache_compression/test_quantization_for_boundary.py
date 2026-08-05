# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Address-lowering tests for the boundary-quantization prototype.

The CUDA byte-level contract is covered by nvfp4BoundaryKernelsTest.  These
tests instead prove that the manager converts KVCM Slot address rows into the
six native addresses correctly and keeps launches batched across disjoint
Pages.  Native calls are recorded so the tests do not require a GPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    BoundaryBufferLayout,
    Nvfp4BoundaryLayerLayout,
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm.llmapi.llm_args import QuantizationForBoundaryCompressionConfig


def _v2_manager():
    # The base manager deliberately verifies the real V2 type.  Constructing
    # without __init__ keeps this focused test independent of GPU allocation.
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.kv_compression_manages_history = False
    return manager


def _config():
    return QuantizationForBoundaryCompressionConfig()


def _layout(
    *,
    layer_id=0,
    life_cycle_id=3,
    runtime_dtype="float16",
    raw_offset=0,
    compact_offset=0,
):
    # Typical physical topology: runtime K/V coalesce into Pool 0; Host packed
    # K/V coalesce into Pool 0 and K/V block scales into Pool 1.  Non-zero
    # offsets prove that logical roles are not confused with physical Pools.
    return Nvfp4BoundaryLayerLayout(
        pool_group_index=2,
        life_cycle_id=life_cycle_id,
        layer_id=layer_id,
        runtime_k=BoundaryBufferLayout(0, raw_offset),
        runtime_v=BoundaryBufferLayout(0, raw_offset + 0x100),
        packed_k=BoundaryBufferLayout(0, compact_offset),
        packed_v=BoundaryBufferLayout(0, compact_offset + 0x40),
        block_scale_k=BoundaryBufferLayout(1, compact_offset),
        block_scale_v=BoundaryBufferLayout(1, compact_offset + 0x10),
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
    return SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        nvfp4_boundary_offload_compress=MagicMock(),
        nvfp4_boundary_onboard_decompress=MagicMock(),
    )


def _manager(*layouts):
    return QuantizationForBoundaryCompression(_config(), _v2_manager(), layer_layouts=layouts)


def test_layout_rejects_values_that_cannot_cross_the_native_abi():
    with pytest.raises(ValueError, match="offset"):
        BoundaryBufferLayout(0, 1.5)

    fields = vars(_layout()).copy()
    fields["nvfp4_scale_orig_quant"] = (float("inf"), 1.0)
    with pytest.raises(ValueError, match="finite positive"):
        Nvfp4BoundaryLayerLayout(**fields)


def test_offload_resolves_coalesced_offsets_and_batches_disjoint_pages():
    manager = _manager(_layout())
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        manager.on_offload_compress(
            pool_group_index=2,
            src_life_cycles=(3, 3),
            src_addresses=((0x1000,), (0x2000,)),
            dst_addresses=((0x3000, 0x4000), (0x5000, 0x6000)),
            stream=0x7000,
        )

    native.nvfp4_boundary_offload_compress.assert_called_once_with(
        [
            (0x1000, 0x1100, 0x3000, 0x3040, 0x4000, 0x4010),
            (0x2000, 0x2100, 0x5000, 0x5040, 0x6000, 0x6010),
        ],
        8,
        64,
        128,
        (2.0, 4.0),
        (0.5, 0.25),
        (8.0, 16.0),
        (0.125, 0.0625),
        "native-fp16",
        0x7000,
    )


def test_factory_builds_the_concrete_manager_when_kvcm_hands_off_layout():
    with patch.object(util_mod, "is_sm_100f", return_value=True):
        manager = util_mod.create_kv_cache_compression_manager(
            _config(),
            _v2_manager(),
            boundary_layer_layouts=(_layout(),),
        )

    assert isinstance(manager, QuantizationForBoundaryCompression)


def test_factory_fails_closed_until_kvcm_hands_off_per_level_layout():
    with (
        patch.object(util_mod, "is_sm_100f", return_value=True),
        pytest.raises(RuntimeError, match="per-level boundary layout handoff"),
    ):
        util_mod.create_kv_cache_compression_manager(_config(), _v2_manager())


def test_onboard_reverses_which_row_supplies_raw_and_compact_addresses():
    manager = _manager(_layout(runtime_dtype="fp8_e4m3"))
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        manager.on_onboard_decompress(
            pool_group_index=2,
            src_life_cycles=(3,),
            src_addresses=((0x3000, 0x4000),),
            dst_addresses=((0x1000,),),
            stream=0x7000,
        )

    native.nvfp4_boundary_onboard_decompress.assert_called_once_with(
        [(0x1000, 0x1100, 0x3000, 0x3040, 0x4000, 0x4010)],
        8,
        64,
        128,
        (2.0, 4.0),
        (0.5, 0.25),
        (8.0, 16.0),
        (0.125, 0.0625),
        "native-fp8",
        0x7000,
    )


@pytest.mark.parametrize(
    ("runtime_dtype", "native_dtype"),
    (("float16", "native-fp16"), ("bfloat16", "native-bf16"), ("fp8_e4m3", "native-fp8")),
)
def test_all_runtime_dtypes_dispatch_through_the_same_two_hooks(runtime_dtype, native_dtype):
    manager = _manager(_layout(runtime_dtype=runtime_dtype))
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        manager.on_offload_compress(
            pool_group_index=2,
            src_life_cycles=(3,),
            src_addresses=((0x1000,),),
            dst_addresses=((0x3000, 0x4000),),
            stream=0,
        )

    assert native.nvfp4_boundary_offload_compress.call_args.args[-2] == native_dtype


def test_one_page_expands_to_each_layer_but_stays_one_homogeneous_launch():
    manager = _manager(
        _layout(layer_id=0),
        _layout(layer_id=1, raw_offset=0x200, compact_offset=0x80),
    )
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        manager.on_offload_compress(
            pool_group_index=2,
            src_life_cycles=(3,),
            src_addresses=((0x1000,),),
            dst_addresses=((0x3000, 0x4000),),
            stream=0,
        )

    tasks = native.nvfp4_boundary_offload_compress.call_args.args[0]
    assert tasks == [
        (0x1000, 0x1100, 0x3000, 0x3040, 0x4000, 0x4010),
        (0x1200, 0x1300, 0x3080, 0x30C0, 0x4080, 0x4090),
    ]
    native.nvfp4_boundary_offload_compress.assert_called_once()


def test_heterogeneous_layers_split_by_native_kernel_contract_not_by_page():
    manager = _manager(
        _layout(layer_id=0, runtime_dtype="float16"),
        _layout(layer_id=1, runtime_dtype="fp8_e4m3", raw_offset=0x200, compact_offset=0x80),
    )
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        manager.on_offload_compress(
            pool_group_index=2,
            src_life_cycles=(3, 3),
            src_addresses=((0x1000,), (0x2000,)),
            dst_addresses=((0x3000, 0x4000), (0x5000, 0x6000)),
            stream=0,
        )

    calls = native.nvfp4_boundary_offload_compress.call_args_list
    assert len(calls) == 2
    assert [len(call.args[0]) for call in calls] == [2, 2]
    assert [call.args[-2] for call in calls] == ["native-fp16", "native-fp8"]


def test_complete_batch_is_validated_before_any_kernel_launch():
    manager = _manager(_layout(life_cycle_id=3))
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        with pytest.raises(ValueError, match="life cycle 9"):
            manager.on_offload_compress(
                pool_group_index=2,
                src_life_cycles=(3, 9),
                src_addresses=((0x1000,), (0x2000,)),
                dst_addresses=((0x3000, 0x4000), (0x5000, 0x6000)),
                stream=0,
            )

    native.nvfp4_boundary_offload_compress.assert_not_called()


def test_later_cohort_alignment_is_validated_before_any_kernel_launch():
    manager = _manager(
        _layout(layer_id=0, runtime_dtype="float16"),
        _layout(layer_id=1, runtime_dtype="fp8_e4m3", raw_offset=1, compact_offset=0x80),
    )
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        with pytest.raises(ValueError, match="raw K address"):
            manager.on_offload_compress(
                pool_group_index=2,
                src_life_cycles=(3,),
                src_addresses=((0x1000,),),
                dst_addresses=((0x3000, 0x4000),),
                stream=0,
            )

    native.nvfp4_boundary_offload_compress.assert_not_called()


def test_short_pool_row_fails_before_native_submission():
    manager = _manager(_layout())
    native = _native()

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings",
        return_value=native,
    ):
        with pytest.raises(ValueError, match="selects pool 1"):
            manager.on_offload_compress(
                pool_group_index=2,
                src_life_cycles=(3,),
                src_addresses=((0x1000,),),
                dst_addresses=((0x3000,),),
                stream=0,
            )

    native.nvfp4_boundary_offload_compress.assert_not_called()


def test_empty_batch_is_noop_without_loading_native_extension():
    manager = _manager(_layout())

    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary._load_native_bindings"
    ) as load:
        manager.on_offload_compress(
            pool_group_index=2,
            src_life_cycles=(),
            src_addresses=(),
            dst_addresses=(),
            stream=0,
        )

    load.assert_not_called()
