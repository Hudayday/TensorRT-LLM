# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
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
    device: str | torch.device = "cpu",
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
    metadata.draft_kv_length_delta_cuda = torch.tensor([0, 37], dtype=torch.int32, device=device)
    metadata.draft_kv_length_delta_cpu = torch.tensor([0, 37], dtype=torch.int32)
    metadata.target_kv_lens_cuda_runtime = torch.tensor([41, 92], dtype=torch.int32, device=device)
    metadata.kv_lens_cuda = metadata.target_kv_lens_cuda_runtime
    metadata.target_kv_lens_runtime = torch.tensor([41, 92], dtype=torch.int32)
    metadata.draft_kv_lens_cuda_runtime = torch.empty(2, dtype=torch.int32, device=device)
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


def test_generation_length_capability_gates_adapter() -> None:
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    config = MTPDecodingConfig(max_draft_len=3)

    assert (
        get_kv_cache_compression_attention_metadata_cls(None, config, TrtllmAttentionMetadata)
        is TrtllmAttentionMetadata
    )
    assert (
        get_kv_cache_compression_attention_metadata_cls(
            _kv_cache_manager(generation_capacity_only=False), config, TrtllmAttentionMetadata
        )
        is TrtllmAttentionMetadata
    )
    assert (
        get_kv_cache_compression_attention_metadata_cls(
            _kv_cache_manager(generation_capacity_only=True), config, TrtllmAttentionMetadata
        )
        is KVCacheCompressionAwareTrtllmAttentionMetadata
    )


def test_base_metadata_manager_switch_hook_is_safe() -> None:
    metadata = AttentionMetadata.__new__(AttentionMetadata)
    metadata.kv_cache_manager = SimpleNamespace(name="target")
    metadata.kv_cache_manager = SimpleNamespace(name="draft")

    assert metadata.on_kv_cache_manager_changed() is None


def test_only_paged_draft_modes_select_length_adapter() -> None:
    from tensorrt_llm.llmapi.llm_args import (
        DFlashDecodingConfig,
        Eagle3DecodingConfig,
        MTPDecodingConfig,
    )

    target = _kv_cache_manager(generation_capacity_only=True)
    eagle = Eagle3DecodingConfig(max_draft_len=4, speculative_model="/tmp/eagle-draft")
    mtp = MTPDecodingConfig(max_draft_len=3)
    dflash = DFlashDecodingConfig(max_draft_len=3)

    for config in (eagle, mtp):
        assert requires_paged_draft_kv_length_domain(config)
        assert (
            get_kv_cache_compression_attention_metadata_cls(target, config, TrtllmAttentionMetadata)
            is KVCacheCompressionAwareTrtllmAttentionMetadata
        )

    assert not requires_paged_draft_kv_length_domain(dflash)
    assert (
        get_kv_cache_compression_attention_metadata_cls(target, dflash, TrtllmAttentionMetadata)
        is TrtllmAttentionMetadata
    )


def test_target_and_draft_length_domains_are_independent() -> None:
    metadata, target_manager, draft_manager = _metadata_with_separate_draft_cache()
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()

    metadata.on_kv_cache_manager_changed()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]
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


def test_cuda_graph_replay_switch_selects_and_restores_domain() -> None:
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


def test_eager_draft_context_selects_and_restores_domain() -> None:
    metadata, target_manager, draft_manager = _metadata_with_separate_draft_cache()
    metadata.kv_cache_block_offsets = torch.tensor([100], dtype=torch.int32)
    metadata.draft_kv_cache_block_offsets = torch.tensor([200], dtype=torch.int32)
    metadata.host_kv_cache_block_offsets = torch.tensor([250], dtype=torch.int32)
    metadata._materialize_draft_device_kv_lengths()
    metadata._materialize_draft_host_kv_lengths()

    with SpecWorkerBase.draft_kv_cache_context(object(), metadata, draft_manager):
        assert metadata.kv_cache_manager is draft_manager
        assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]


def test_native_speculative_updates_refresh_device_domain() -> None:
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    metadata._materialize_draft_device_kv_lengths()
    metadata.kv_lens_cuda = metadata.kv_lens_cuda.clone()
    metadata.kv_lens_cuda[1].add_(1)

    metadata.on_update_kv_lens()
    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()

    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 130]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_draft_length_refresh_reuses_graph_stable_buffers() -> None:
    metadata, _, draft_manager = _metadata_with_separate_draft_cache(device="cuda")
    metadata._materialize_draft_device_kv_lengths()
    draft_pointer = metadata.draft_kv_lens_cuda_runtime.data_ptr()
    delta_pointer = metadata.draft_kv_length_delta_cuda.data_ptr()

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
    graph.replay()
    metadata.kv_cache_manager = draft_manager
    metadata.on_kv_cache_manager_changed()

    assert metadata.draft_kv_lens_cuda_runtime.data_ptr() == draft_pointer
    assert metadata.draft_kv_length_delta_cuda.data_ptr() == delta_pointer
    assert metadata.kv_lens_cuda_runtime.cpu().tolist() == [44, 137]


def test_model_engine_adjusts_before_prepare() -> None:
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


def test_model_engine_prepares_without_compression_manager() -> None:
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    metadata = mock.Mock()

    engine._prepare_self_attention_metadata(metadata, ResourceManager({}))

    metadata.prepare.assert_called_once_with()


def test_draft_model_does_not_adjust_compression_metadata() -> None:
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.is_draft_model = True
    compression_manager = mock.Mock(spec=BaseKVCacheCompressionManager)
    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression_manager,
        }
    )
    metadata = mock.Mock()

    engine._prepare_self_attention_metadata(metadata, resource_manager)

    compression_manager.adjust_attention_metadata.assert_not_called()
    metadata.prepare.assert_called_once_with()
