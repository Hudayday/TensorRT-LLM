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
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SpeculativeConfig


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


def get_kv_cache_compression_attention_metadata_cls(
    kv_cache_manager: "KVCacheManagerV2 | None",
    spec_config: "SpeculativeConfig | None",
    metadata_cls: type[AttentionMetadata],
) -> type[AttentionMetadata]:
    """Select metadata that tracks independent target and draft KV lengths."""
    if (
        kv_cache_manager is None
        or not kv_cache_manager.generation_capacity_only
        or not requires_paged_draft_kv_length_domain(spec_config)
    ):
        return metadata_cls
    if metadata_cls is not TrtllmAttentionMetadata:
        raise ValueError("KV cache compression currently requires TRTLLM attention")
    return KVCacheCompressionAwareTrtllmAttentionMetadata


@dataclass(kw_only=True)
class KVCacheCompressionAwareTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Provide independent target and draft KV lengths to TRT-LLM attention.

    Physical compression shortens only the target cache, while one-model
    speculative decoding keeps its draft cache dense. The fused attention op
    consumes both host and device length metadata, and CUDA Graph replay
    requires their storage addresses to remain stable. This adapter
    materializes both length views and selects the matching view when the
    existing speculative path switches the active KV cache manager.

    It does not select or move KV entries, resize either cache, or change token
    proposal and acceptance. Those responsibilities remain with the compression
    manager, KV cache managers, and speculative decoding implementation.
    """

    draft_kv_length_delta: list[int] | None = None
    draft_kv_length_delta_cuda: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_length_delta_cpu: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_lens_cuda_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_kv_lens_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    target_kv_lens_cuda_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    target_kv_lens_runtime: torch.Tensor | None = field(default=None, init=False, repr=False)
    target_host_total_kv_lens: torch.Tensor | None = field(default=None, init=False, repr=False)
    draft_host_total_kv_lens: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _post_init_with_buffers(self, buffers) -> None:
        super()._post_init_with_buffers(buffers)
        self.target_host_total_kv_lens = self.host_total_kv_lens
        if self.draft_kv_cache_manager is None:
            raise ValueError(
                "compression-aware attention metadata requires a draft KV cache manager"
            )
        # TRT-LLM attention consumes device lengths, host lengths, and host
        # totals. Keep a separate graph-stable draft domain for all three.
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
        self._materialize_draft_host_kv_lengths()

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
        self._materialize_draft_device_kv_lengths()

    def _materialize_draft_device_kv_lengths(self) -> None:
        batch_size = self.num_seqs
        assert self.target_kv_lens_cuda_runtime is not None
        assert self.draft_kv_length_delta_cuda is not None
        assert self.draft_kv_lens_cuda_runtime is not None
        torch.add(
            self.target_kv_lens_cuda_runtime[:batch_size],
            self.draft_kv_length_delta_cuda[:batch_size],
            out=self.draft_kv_lens_cuda_runtime[:batch_size],
        )

    def _materialize_draft_host_kv_lengths(self) -> None:
        batch_size = self.num_seqs
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
        if self.target_kv_lens_cuda_runtime is not None:
            self.kv_lens_cuda_runtime = self.target_kv_lens_cuda_runtime
        if self.target_kv_lens_runtime is not None:
            self.kv_lens_runtime = self.target_kv_lens_runtime
        if self.target_host_total_kv_lens is not None:
            self.host_total_kv_lens = self.target_host_total_kv_lens
