# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused public-config tests for boundary quantization."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import (QuantizationCompressionConfig,
                                          TorchLlmArgs)


@pytest.mark.cpu_only
def test_quantization_compression_config_is_a_small_storage_init_contract():
    config = TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_cache_compression_config={
            "algorithm": "quantization_for_boundary",
            "quant": "nvfp4",
            "scale_checkpoint_path": "/tmp/nvfp4-kv-scales",
        },
    ).kv_cache_compression_config

    assert isinstance(config, QuantizationCompressionConfig)
    assert config.quant == "nvfp4"
    assert config.scale_checkpoint_path == "/tmp/nvfp4-kv-scales"
    assert "target_cache_tier" not in config.model_dump()
    assert not config.changes_physical_kv_length
    assert config.supports_block_reuse()
    assert not config.supports_speculative_decoding()


@pytest.mark.cpu_only
def test_quantization_compression_config_rejects_unimplemented_format():
    with pytest.raises(ValidationError):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            kv_cache_compression_config={
                "algorithm": "quantization_for_boundary",
                "quant": "fp8",
                "scale_checkpoint_path": "/tmp/nvfp4-kv-scales",
            },
        )
