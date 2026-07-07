# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.kv_cache_compression.attention import (
    KVCacheCompressionTrtllmAttention,
    KVCacheCompressionTrtllmAttentionMetadata,
    configure_kv_cache_compression_attention_backend,
    get_kv_cache_compression_attention_backend,
    get_model_kv_cache_compression_attention_backend,
    requires_kv_cache_compression_attention_backend,
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


def test_only_registered_speculative_compression_requires_adapter():
    speculative = mock.Mock()
    triattention = mock.Mock(algorithm="triattention")
    unknown = mock.Mock(algorithm="unknown")

    with mock.patch(
        "tensorrt_llm._torch.speculative.should_use_separate_draft_kv_cache",
        return_value=True,
    ):
        assert requires_kv_cache_compression_attention_backend(triattention, speculative)
        assert not requires_kv_cache_compression_attention_backend(unknown, speculative)


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
