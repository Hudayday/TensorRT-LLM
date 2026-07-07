# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.kv_cache_compression.attention import (
    KVCacheCompressionTrtllmAttention,
    KVCacheCompressionTrtllmAttentionMetadata,
    configure_kv_cache_compression_attention_backend,
    get_kv_cache_compression_attention_backend,
    get_model_kv_cache_compression_attention_backend,
    requires_kv_cache_compression_attention_backend,
    requires_paged_draft_kv_length_domain,
)


def _metadata_with_separate_draft_cache():
    metadata = KVCacheCompressionTrtllmAttentionMetadata.__new__(
        KVCacheCompressionTrtllmAttentionMetadata
    )
    target_manager = object()
    draft_manager = object()
    metadata.kv_cache_manager = target_manager
    metadata.draft_kv_cache_manager = draft_manager
    metadata.draft_kv_length_delta = [0, 37]
    metadata.draft_kv_length_delta_cuda = torch.tensor([0, 37], dtype=torch.int32)
    metadata.draft_kv_length_delta_cpu = torch.tensor([0, 37], dtype=torch.int32)
    metadata.target_kv_lens_cuda_runtime = torch.tensor([41, 92], dtype=torch.int32)
    metadata.kv_lens_cuda = metadata.target_kv_lens_cuda_runtime
    metadata.target_kv_lens_runtime = torch.tensor([41, 92], dtype=torch.int32)
    metadata.draft_kv_lens_cuda_runtime = torch.empty(2, dtype=torch.int32)
    metadata.draft_kv_lens_runtime = torch.empty(2, dtype=torch.int32)
    metadata.target_host_total_kv_lens = torch.tensor([41, 92], dtype=torch.int32)
    metadata.draft_host_total_kv_lens = torch.empty(2, dtype=torch.int32)
    metadata.host_total_kv_lens = metadata.target_host_total_kv_lens
    metadata.enable_flash_mla = False
    metadata._seq_lens = torch.ones(2, dtype=torch.int32)
    metadata._num_contexts = 1
    metadata._num_generations = 1
    return metadata, target_manager, draft_manager


def test_backend_uses_standard_attention_with_compression_metadata():
    backend = get_kv_cache_compression_attention_backend(TrtllmAttention)
    assert backend is KVCacheCompressionTrtllmAttention
    assert issubclass(backend, TrtllmAttention)
    assert backend.Metadata is KVCacheCompressionTrtllmAttentionMetadata


def test_model_config_selects_same_backend_for_modules_and_metadata():
    model_config = mock.Mock()
    model_config.extra_attrs = {}

    assert (
        get_model_kv_cache_compression_attention_backend(model_config, TrtllmAttention)
        is TrtllmAttention
    )

    configure_kv_cache_compression_attention_backend(model_config, enabled=True)

    assert (
        get_model_kv_cache_compression_attention_backend(model_config, TrtllmAttention)
        is KVCacheCompressionTrtllmAttention
    )


def test_only_paged_draft_attention_requires_adapter():
    from tensorrt_llm.llmapi.llm_args import (
        DFlashDecodingConfig,
        DraftTargetDecodingConfig,
        Eagle3DecodingConfig,
        MTPDecodingConfig,
        PARDDecodingConfig,
        SADecodingConfig,
    )

    triattention = mock.Mock(algorithm="triattention")
    unknown = mock.Mock(algorithm="unknown")
    paged_draft_configs = [
        Eagle3DecodingConfig(
            max_draft_len=4,
            speculative_model="/tmp/qwen3-eagle3-draft",
        ),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        MTPDecodingConfig(max_draft_len=3),
        DraftTargetDecodingConfig(
            max_draft_len=3,
            speculative_model="/tmp/draft-target-model",
        ),
        PARDDecodingConfig(max_draft_len=3),
    ]

    for speculative in paged_draft_configs:
        assert requires_paged_draft_kv_length_domain(speculative)
        assert requires_kv_cache_compression_attention_backend(triattention, speculative)
        assert not requires_kv_cache_compression_attention_backend(unknown, speculative)

    dflash = DFlashDecodingConfig(max_draft_len=3)
    assert not requires_paged_draft_kv_length_domain(dflash)
    assert not requires_kv_cache_compression_attention_backend(triattention, dflash)

    eagle3_two_model = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model="/tmp/qwen3-eagle3-draft",
        eagle3_one_model=False,
    )
    draft_target_two_model = DraftTargetDecodingConfig(
        max_draft_len=3,
        speculative_model="/tmp/draft-target-model",
    )
    draft_target_two_model._draft_target_one_model = False
    non_paged_draft_configs = [
        SADecodingConfig(max_draft_len=3),
        eagle3_two_model,
        draft_target_two_model,
    ]
    for speculative in non_paged_draft_configs:
        assert not requires_paged_draft_kv_length_domain(speculative)
        assert not requires_kv_cache_compression_attention_backend(triattention, speculative)

    paged_draft_configs[0]._allow_separate_draft_kv_cache = False
    assert not requires_paged_draft_kv_length_domain(paged_draft_configs[0])


def test_complete_target_and_draft_length_domains_are_independent():
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    metadata._refresh_draft_kv_length_domain(refresh_host=True)

    metadata.activate_kv_length_domain()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 92]
    assert metadata.host_total_kv_lens is metadata.target_host_total_kv_lens

    metadata.kv_cache_manager = draft_manager
    metadata.activate_kv_length_domain()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 129]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]
    assert metadata.host_total_kv_lens.tolist() == [41, 129]

    metadata.restore_target_kv_length_domain()
    assert metadata.kv_lens_cuda_runtime is metadata.target_kv_lens_cuda_runtime
    assert metadata.kv_lens_runtime is metadata.target_kv_lens_runtime
    assert metadata.host_total_kv_lens is metadata.target_host_total_kv_lens


def test_prepare_captures_runtime_views_from_legacy_base_path():
    metadata, _, _ = _metadata_with_separate_draft_cache()
    target_cuda = torch.tensor([10, 20], dtype=torch.int32)
    target_host = torch.tensor([10, 20], dtype=torch.int32)
    target_total = torch.tensor([10, 20], dtype=torch.int32)
    metadata.target_kv_lens_cuda_runtime = None
    metadata.target_kv_lens_runtime = None

    def legacy_prepare(instance):
        instance.kv_lens_cuda_runtime = target_cuda
        instance.kv_lens_runtime = target_host
        instance.host_total_kv_lens = target_total

    with mock.patch.object(TrtllmAttentionMetadata, "prepare", legacy_prepare):
        metadata.prepare()

    assert metadata.target_kv_lens_cuda_runtime is target_cuda
    assert metadata.target_kv_lens_runtime is target_host
    assert metadata.target_host_total_kv_lens is target_total
    assert metadata.draft_kv_lens_cuda_runtime.tolist() == [10, 57]
    assert metadata.draft_kv_lens_runtime.tolist() == [10, 57]
    assert metadata.draft_host_total_kv_lens.tolist() == [10, 57]


def test_native_spec_update_refreshes_device_domain_once():
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    metadata._refresh_draft_kv_length_domain(refresh_host=True)
    metadata.kv_lens_cuda = metadata.kv_lens_cuda.clone()
    metadata.kv_lens_cuda[1].add_(1)

    metadata.update_for_spec_dec()
    metadata.kv_cache_manager = draft_manager
    metadata.activate_kv_length_domain()

    assert metadata.target_kv_lens_cuda_runtime.data_ptr() == metadata.kv_lens_cuda.data_ptr()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 130]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]


def test_vanilla_spec_update_refreshes_device_domain_once():
    metadata, _, draft_manager = _metadata_with_separate_draft_cache()
    metadata._refresh_draft_kv_length_domain(refresh_host=True)
    metadata.kv_lens_cuda = metadata.kv_lens_cuda.clone()
    metadata.kv_lens_cuda[1].add_(1)

    metadata.on_update_kv_lens()
    metadata.kv_cache_manager = draft_manager
    metadata.activate_kv_length_domain()

    assert metadata.target_kv_lens_cuda_runtime.data_ptr() == metadata.kv_lens_cuda.data_ptr()
    assert metadata.kv_lens_cuda_runtime.tolist() == [41, 130]
    assert metadata.kv_lens_runtime.tolist() == [41, 129]


def test_compression_attention_restores_target_domain_after_failure():
    attention = KVCacheCompressionTrtllmAttention.__new__(KVCacheCompressionTrtllmAttention)
    metadata = mock.Mock(spec=KVCacheCompressionTrtllmAttentionMetadata)

    with (
        mock.patch.object(
            TrtllmAttention,
            "forward",
            side_effect=RuntimeError("injected failure"),
        ),
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        attention.forward(torch.empty(0), None, None, metadata)

    metadata.activate_kv_length_domain.assert_called_once_with()
    metadata.restore_target_kv_length_domain.assert_called_once_with()
