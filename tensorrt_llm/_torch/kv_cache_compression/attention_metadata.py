# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM attention metadata for independent target and draft KV lengths."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig, SpeculativeConfig


def requires_kv_cache_compression_attention_metadata(
    compression_config: "KvCacheCompressionConfig | None",
    spec_config: "SpeculativeConfig | None",
) -> bool:
    """Return whether target compression creates a paged draft length domain."""
    return (
        compression_config is not None
        and compression_config.adjusts_generation_kv_length
        and requires_paged_draft_kv_length_domain(spec_config)
    )


def requires_paged_draft_kv_length_domain(
    spec_config: "SpeculativeConfig | None",
) -> bool:
    """Whether a separate draft KVCM feeds standard paged attention.

    These modes execute their draft model through the regular attention stack,
    so target-only compaction requires a dense draft length alongside the
    compressed target length. DFlash instead reads its private context K/V and
    does not consume this metadata domain.
    """
    if spec_config is None:
        return False
    from ..speculative import should_use_separate_draft_kv_cache

    if not should_use_separate_draft_kv_cache(spec_config):
        return False
    mode = spec_config.spec_dec_mode
    return (
        mode.is_mtp_one_model()
        or mode.is_eagle3_one_model()
        or mode.is_draft_target_one_model()
        or mode.is_pard()
    )


def get_kv_cache_compression_attention_metadata(
    compression_config: "KvCacheCompressionConfig | None",
    spec_config: "SpeculativeConfig | None",
    metadata_cls: type[AttentionMetadata],
) -> type[AttentionMetadata]:
    """Select metadata that tracks independent target and draft KV lengths."""
    if not requires_kv_cache_compression_attention_metadata(
        compression_config, spec_config
    ):
        return metadata_cls
    if metadata_cls is not TrtllmAttentionMetadata:
        raise ValueError("KV cache compression currently requires TRTLLM attention")
    return KVCacheCompressionTrtllmAttentionMetadata


@dataclass(kw_only=True)
class KVCacheCompressionTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Select graph-stable KV lengths for the currently active KVCM."""

    draft_kv_length_delta: list[int] | None = None
    draft_kv_length_delta_cuda: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_length_delta_cpu: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_lens_cuda_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_lens_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    target_kv_lens_cuda_runtime: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    target_kv_lens_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    target_host_total_kv_lens: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_host_total_kv_lens: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _post_init_with_buffers(self, buffers) -> None:
        super()._post_init_with_buffers(buffers)
        self.target_host_total_kv_lens = self.host_total_kv_lens
        if self.draft_kv_cache_manager is None:
            return
        capture_graph = self.is_cuda_graph
        self.draft_kv_length_delta_cuda = self.get_empty_like(
            buffers,
            self.kv_lens_cuda,
            cache_name="kv_cache_compression_draft_length_delta_cuda",
            capture_graph=capture_graph,
        )
        self.draft_kv_length_delta_cpu = torch.empty_like(
            self.kv_lens,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.draft_kv_lens_cuda_runtime = self.get_empty_like(
            buffers,
            self.kv_lens_cuda,
            cache_name="kv_cache_compression_draft_kv_lens_cuda_runtime",
            capture_graph=capture_graph,
        )
        self.draft_kv_lens_runtime = torch.empty_like(
            self.kv_lens,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.draft_host_total_kv_lens = torch.empty_like(self.target_host_total_kv_lens)

    def set_draft_kv_length_delta(self, delta: Sequence[int]) -> None:
        self.draft_kv_length_delta = list(delta)

    def _bind_runtime_views(
        self,
        *,
        kv_lens_cuda: torch.Tensor,
        kv_lens: torch.Tensor,
        prompt_lens_cuda: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        host_request_types: torch.Tensor,
    ) -> None:
        super()._bind_runtime_views(
            kv_lens_cuda=kv_lens_cuda,
            kv_lens=kv_lens,
            prompt_lens_cuda=prompt_lens_cuda,
            prompt_lens_cpu=prompt_lens_cpu,
            host_request_types=host_request_types,
        )
        self.target_kv_lens_cuda_runtime = kv_lens_cuda
        self.target_kv_lens_runtime = kv_lens

    def prepare(self) -> None:
        if self.target_host_total_kv_lens is not None:
            self.host_total_kv_lens = self.target_host_total_kv_lens
        super().prepare()
        # Older TRT-LLM bases bind these views directly in prepare() instead of
        # routing through _bind_runtime_views(). Capture the authoritative views
        # after the base call so the compression metadata works with both paths.
        self.target_kv_lens_cuda_runtime = self.kv_lens_cuda_runtime
        self.target_kv_lens_runtime = self.kv_lens_runtime
        self.target_host_total_kv_lens = self.host_total_kv_lens
        delta = self.draft_kv_length_delta
        if delta is None:
            return
        if len(delta) != self.num_seqs:
            raise ValueError("draft KV length delta must contain one value per sequence")
        if any(value < 0 for value in delta):
            raise ValueError("draft KV length delta must be non-negative")

        assert self.draft_kv_length_delta_cpu is not None
        assert self.draft_kv_length_delta_cuda is not None
        batch_size = self.num_seqs
        self.draft_kv_length_delta_cpu[:batch_size].copy_(torch.as_tensor(delta, dtype=torch.int))
        self.draft_kv_length_delta_cuda[:batch_size].copy_(
            self.draft_kv_length_delta_cpu[:batch_size], non_blocking=True
        )
        self._refresh_draft_kv_length_domain(refresh_host=True)

    def on_update_kv_lens(self) -> None:
        """Follow in-place KV-length mutations from vanilla speculative decoding."""
        super().on_update_kv_lens()
        self._refresh_device_kv_length_domain()

    def update_for_spec_dec(self) -> None:
        """Follow native speculative mutations of ``kv_lens_cuda``."""
        super().update_for_spec_dec()
        self._refresh_device_kv_length_domain()

    def restore_from_spec_dec(self) -> None:
        super().restore_from_spec_dec()
        self._refresh_device_kv_length_domain()

    def _refresh_device_kv_length_domain(self) -> None:
        if self.draft_kv_length_delta is None:
            return
        self.target_kv_lens_cuda_runtime = self.kv_lens_cuda[: self.num_seqs]
        self._refresh_draft_kv_length_domain(refresh_host=False)

    def _refresh_draft_kv_length_domain(self, *, refresh_host: bool) -> None:
        batch_size = self.num_seqs
        assert self.target_kv_lens_cuda_runtime is not None
        assert self.draft_kv_length_delta_cuda is not None
        assert self.draft_kv_lens_cuda_runtime is not None
        torch.add(
            self.target_kv_lens_cuda_runtime[:batch_size],
            self.draft_kv_length_delta_cuda[:batch_size],
            out=self.draft_kv_lens_cuda_runtime[:batch_size],
        )
        if not refresh_host:
            return

        assert self.target_kv_lens_runtime is not None
        assert self.draft_kv_length_delta_cpu is not None
        assert self.draft_kv_lens_runtime is not None
        assert self.target_host_total_kv_lens is not None
        assert self.draft_host_total_kv_lens is not None
        delta = self.draft_kv_length_delta_cpu[:batch_size]
        torch.add(
            self.target_kv_lens_runtime[:batch_size],
            delta,
            out=self.draft_kv_lens_runtime[:batch_size],
        )
        self.draft_host_total_kv_lens.copy_(self.target_host_total_kv_lens)
        self.draft_host_total_kv_lens[0].add_(delta[: self.num_contexts].sum())
        self.draft_host_total_kv_lens[1].add_(delta[self.num_contexts : batch_size].sum())

    def on_kv_cache_manager_changed(self) -> None:
        """Select lengths for the active target or draft KV cache manager."""
        super().on_kv_cache_manager_changed()
        self.activate_kv_length_domain()

    def activate_kv_length_domain(self) -> None:
        use_draft = (
            self.draft_kv_length_delta is not None
            and self.draft_kv_cache_manager is not None
            and self.kv_cache_manager is self.draft_kv_cache_manager
        )
        if use_draft:
            assert self.draft_kv_lens_cuda_runtime is not None
            assert self.draft_kv_lens_runtime is not None
            assert self.draft_host_total_kv_lens is not None
            self.kv_lens_cuda_runtime = self.draft_kv_lens_cuda_runtime[: self.num_seqs]
            self.kv_lens_runtime = self.draft_kv_lens_runtime[: self.num_seqs]
            self.host_total_kv_lens = self.draft_host_total_kv_lens
            return
        self.restore_target_kv_length_domain()

    def restore_target_kv_length_domain(self) -> None:
        if self.target_kv_lens_cuda_runtime is not None:
            self.kv_lens_cuda_runtime = self.target_kv_lens_cuda_runtime
        if self.target_kv_lens_runtime is not None:
            self.kv_lens_runtime = self.target_kv_lens_runtime
        if self.target_host_total_kv_lens is not None:
            self.host_total_kv_lens = self.target_host_total_kv_lens


__all__ = [
    "KVCacheCompressionTrtllmAttentionMetadata",
    "get_kv_cache_compression_attention_metadata",
    "requires_kv_cache_compression_attention_metadata",
    "requires_paged_draft_kv_length_domain",
]
