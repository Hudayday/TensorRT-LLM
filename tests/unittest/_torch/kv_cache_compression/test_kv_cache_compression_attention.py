# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.kv_cache_compression.attention_metadata import (
    KVCacheCompressionAwareTrtllmAttentionMetadata,
    get_kv_cache_compression_attention_metadata_cls,
    requires_paged_draft_kv_length_domain,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.speculative.interface import (
    SpecWorkerBase,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)


def _kv_cache_manager(*, generation_capacity_only: bool) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.generation_capacity_only = generation_capacity_only
    return manager


def _metadata_with_separate_draft_cache(
    *,
    cuda_device: str | torch.device = "cpu",
) -> tuple[KVCacheCompressionAwareTrtllmAttentionMetadata, object, object]:
    metadata = KVCacheCompressionAwareTrtllmAttentionMetadata.__new__(
        KVCacheCompressionAwareTrtllmAttentionMetadata
    )
    target_manager = SimpleNamespace()
    draft_manager = SimpleNamespace(
        host_kv_cache_block_offsets=torch.tensor([300], dtype=torch.int32)
    )
    metadata.kv_cache_manager = target_manager
    metadata.draft_kv_cache_manager = draft_manager
    metadata.draft_kv_length_delta = [0, 37]
    metadata.draft_kv_length_delta_cuda = torch.tensor(
        [0, 37], dtype=torch.int32, device=cuda_device
    )
    metadata.draft_kv_length_delta_cpu = torch.tensor([0, 37], dtype=torch.int32)
    metadata.target_kv_lens_cuda_runtime = torch.tensor(
        [41, 92], dtype=torch.int32, device=cuda_device
    )
    metadata.kv_lens_cuda = metadata.target_kv_lens_cuda_runtime
    metadata.target_kv_lens_runtime = torch.tensor([41, 92], dtype=torch.int32)
    metadata.draft_kv_lens_cuda_runtime = torch.empty(2, dtype=torch.int32, device=cuda_device)
    metadata.draft_kv_lens_runtime = torch.empty(2, dtype=torch.int32)
    metadata.target_host_total_kv_lens = torch.tensor([41, 92], dtype=torch.int32)
    metadata.draft_host_total_kv_lens = torch.empty(2, dtype=torch.int32)
    metadata.host_total_kv_lens = metadata.target_host_total_kv_lens
    metadata.enable_flash_mla = False
    metadata._saved_tensors = {}
    metadata._seq_lens = torch.ones(2, dtype=torch.int32)
    metadata._num_contexts = 1
    metadata._num_generations = 1
    return metadata, target_manager, draft_manager


def test_target_kv_cache_runtime_policy_gates_adapter() -> None:
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    mtp = MTPDecodingConfig(max_draft_len=3)

    assert (
        get_kv_cache_compression_attention_metadata_cls(None, mtp, TrtllmAttentionMetadata)
        is TrtllmAttentionMetadata
    )
    assert (
        get_kv_cache_compression_attention_metadata_cls(
            _kv_cache_manager(generation_capacity_only=False),
            mtp,
            TrtllmAttentionMetadata,
        )
        is TrtllmAttentionMetadata
    )
    assert (
        get_kv_cache_compression_attention_metadata_cls(
            _kv_cache_manager(generation_capacity_only=True),
            mtp,
            TrtllmAttentionMetadata,
        )
        is KVCacheCompressionAwareTrtllmAttentionMetadata
    )


def test_paged_draft_modes_and_dflash_select_expected_length_domain() -> None:
    from tensorrt_llm.llmapi.llm_args import (
        DFlashDecodingConfig,
        DraftTargetDecodingConfig,
        Eagle3DecodingConfig,
        MTPDecodingConfig,
        PARDDecodingConfig,
    )

    target_kv_cache_manager = _kv_cache_manager(generation_capacity_only=True)
    eagle3 = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model="/tmp/qwen3-eagle3-draft",
    )
    mtp = MTPDecodingConfig(max_draft_len=3)
    dflash = DFlashDecodingConfig(max_draft_len=3)
    draft_target = DraftTargetDecodingConfig(
        max_draft_len=3,
        speculative_model="/tmp/draft-target-model",
    )
    pard = PARDDecodingConfig(max_draft_len=3)

    for config in (eagle3, mtp, draft_target, pard):
        assert requires_paged_draft_kv_length_domain(config)
        assert (
            get_kv_cache_compression_attention_metadata_cls(
                target_kv_cache_manager, config, TrtllmAttentionMetadata
            )
            is KVCacheCompressionAwareTrtllmAttentionMetadata
        )

    for config in (dflash,):
        assert not requires_paged_draft_kv_length_domain(config)
        assert (
            get_kv_cache_compression_attention_metadata_cls(
                target_kv_cache_manager, config, TrtllmAttentionMetadata
            )
            is TrtllmAttentionMetadata
        )


def test_target_and_draft_cpu_length_domains_are_independent() -> None:
    metadata, target_manager, draft_manager = _metadata_with_separate_draft_cache()
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()

    metadata.on_kv_cache_manager_changed()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]
    assert metadata.kv_lens_runtime.tolist() == [41, 92]
    assert metadata.host_total_kv_lens is metadata.target_host_total_kv_lens

    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]
    assert metadata.host_total_kv_lens.tolist() == [41, 129]

    metadata.kv_cache_manager = target_manager
    metadata.on_kv_cache_manager_changed()
    assert metadata.kv_lens_cuda_runtime is metadata.target_kv_lens_cuda_runtime
    assert metadata.kv_lens_runtime is metadata.target_kv_lens_runtime
    assert metadata.host_total_kv_lens is metadata.target_host_total_kv_lens


def test_native_draft_cache_context_selects_and_restores_length_domain() -> None:
    metadata, target_manager, draft_manager = _metadata_with_separate_draft_cache()
    target_offsets = torch.tensor([100], dtype=torch.int32)
    draft_offsets = torch.tensor([200], dtype=torch.int32)
    target_host_offsets = torch.tensor([250], dtype=torch.int32)
    metadata.kv_cache_block_offsets = target_offsets
    metadata.draft_kv_cache_block_offsets = draft_offsets
    metadata.host_kv_cache_block_offsets = target_host_offsets
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()

    saved = prepare_attn_metadata_for_draft_replay(metadata, draft_manager)
    assert metadata.kv_cache_manager is draft_manager
    assert metadata.kv_cache_block_offsets is draft_offsets
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]

    restore_attn_metadata_after_draft_replay(metadata, saved)

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_cache_block_offsets is target_offsets
    assert metadata.host_kv_cache_block_offsets is target_host_offsets
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]


def test_eager_draft_cache_context_selects_and_restores_length_domain() -> None:
    metadata, target_manager, draft_manager = _metadata_with_separate_draft_cache()
    target_offsets = torch.tensor([100], dtype=torch.int32)
    draft_offsets = torch.tensor([200], dtype=torch.int32)
    target_host_offsets = torch.tensor([250], dtype=torch.int32)
    metadata.kv_cache_block_offsets = target_offsets
    metadata.draft_kv_cache_block_offsets = draft_offsets
    metadata.host_kv_cache_block_offsets = target_host_offsets
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()

    worker = object()
    with SpecWorkerBase.draft_kv_cache_context(worker, metadata, draft_manager):
        assert metadata.kv_cache_manager is draft_manager
        assert metadata.kv_cache_block_offsets is draft_offsets
        assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_cache_block_offsets is target_offsets
    assert metadata.host_kv_cache_block_offsets is target_host_offsets
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_draft_length_refresh_reuses_graph_stable_buffers() -> None:
    metadata, _, draft_manager = _metadata_with_separate_draft_cache(cuda_device="cuda")
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()
    draft_output_pointer = metadata.draft_kv_lens_cuda_runtime.data_ptr()
    delta_pointer = metadata.draft_kv_length_delta_cuda.data_ptr()
    draft_cpu_output_pointer = metadata.draft_kv_lens_runtime.data_ptr()
    delta_cpu_pointer = metadata.draft_kv_length_delta_cpu.data_ptr()

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        metadata._materialize_draft_device_kv_lengths()

    metadata.target_kv_lens_cuda_runtime.add_(
        torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    )
    metadata.draft_kv_length_delta_cuda.copy_(
        torch.tensor([1, 40], dtype=torch.int32, device="cuda")
    )
    metadata.target_kv_lens_runtime.add_(torch.tensor([2, 5], dtype=torch.int32))
    graph.replay()
    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()

    assert metadata.draft_kv_lens_cuda_runtime.data_ptr() == draft_output_pointer
    assert metadata.draft_kv_length_delta_cuda.data_ptr() == delta_pointer
    assert metadata.draft_kv_lens_runtime.data_ptr() == draft_cpu_output_pointer
    assert metadata.draft_kv_length_delta_cpu.data_ptr() == delta_cpu_pointer
    assert metadata.kv_lens_cuda_runtime.cpu().tolist() == [44, 137]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]


def test_prepare_captures_runtime_views_from_base_metadata() -> None:
    metadata, _, _ = _metadata_with_separate_draft_cache()
    target_cuda = torch.tensor([10, 20], dtype=torch.int32)
    target_cpu = torch.tensor([10, 20], dtype=torch.int32)
    target_totals = torch.tensor([10, 20], dtype=torch.int32)
    metadata.target_kv_lens_cuda_runtime = None
    metadata.target_kv_lens_runtime = None

    def base_prepare(instance: KVCacheCompressionAwareTrtllmAttentionMetadata) -> None:
        instance.kv_lens_cuda = target_cuda
        instance.kv_lens_cuda_runtime = target_cuda
        instance.kv_lens_runtime = target_cpu
        instance.host_total_kv_lens = target_totals

    with mock.patch.object(TrtllmAttentionMetadata, "prepare", base_prepare):
        metadata.prepare()

    assert metadata.target_kv_lens_cuda_runtime is target_cuda
    assert metadata.target_kv_lens_runtime is target_cpu
    assert metadata.target_host_total_kv_lens is target_totals
    assert metadata.draft_kv_lens_runtime.tolist() == [10, 57]
    assert metadata.draft_host_total_kv_lens.tolist() == [10, 57]

    metadata._refresh_device_kv_length_domain()
    assert metadata.draft_kv_lens_cuda_runtime.tolist() == [10, 57]


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            KVCacheCompressionAwareTrtllmAttentionMetadata.on_update_kv_lens,
            id="on-update-kv-lens",
        ),
        pytest.param(
            KVCacheCompressionAwareTrtllmAttentionMetadata.update_for_spec_dec,
            id="update-for-spec-dec",
        ),
    ],
)
def test_native_speculative_update_refreshes_device_domain(
    update: Callable[[KVCacheCompressionAwareTrtllmAttentionMetadata], None],
) -> None:
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()
    metadata.kv_lens_cuda = metadata.kv_lens_cuda.clone()
    metadata.kv_lens_cuda[1].add_(1)

    update(metadata)
    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()

    assert metadata.target_kv_lens_cuda_runtime.data_ptr() == metadata.kv_lens_cuda.data_ptr()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 130]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]


def test_restore_from_spec_dec_refreshes_device_domain() -> None:
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    original_kv_lens_cuda = metadata.kv_lens_cuda
    metadata._saved_tensors["kv_lens_cuda"] = original_kv_lens_cuda
    metadata.kv_lens_cuda = original_kv_lens_cuda.clone().add_(
        torch.tensor([2, 5], dtype=torch.int32)
    )

    metadata.restore_from_spec_dec()
    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()

    assert metadata.kv_lens_cuda is original_kv_lens_cuda
    assert metadata.target_kv_lens_cuda_runtime.data_ptr() == original_kv_lens_cuda.data_ptr()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]


def test_model_engine_adjusts_compression_metadata_before_prepare() -> None:
    calls = []
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    compression_manager = mock.Mock(spec=BaseKVCacheCompressionManager)
    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression_manager,
        }
    )
    metadata = mock.Mock()
    compression_manager.adjust_attention_metadata.side_effect = lambda value: calls.append(
        ("adjust", value)
    )
    metadata.prepare.side_effect = lambda: calls.append(("prepare", metadata))

    engine._prepare_self_attention_metadata(metadata, resource_manager)

    assert calls == [("adjust", metadata), ("prepare", metadata)]


def test_model_engine_skips_inactive_compression_manager() -> None:
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    metadata = mock.Mock()

    engine._prepare_self_attention_metadata(metadata, ResourceManager({}))

    metadata.prepare.assert_called_once_with()
    engine.is_draft_model = True
    compression_manager = mock.Mock(spec=BaseKVCacheCompressionManager)
    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression_manager,
        }
    )
    draft_metadata = mock.Mock()

    engine._prepare_self_attention_metadata(draft_metadata, resource_manager)

    compression_manager.adjust_attention_metadata.assert_not_called()
    draft_metadata.prepare.assert_called_once_with()
