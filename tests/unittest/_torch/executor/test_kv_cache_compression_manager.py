# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the KV-cache compression manager framework
(``KVCacheCompressionManager`` in ``resource_manager.py``) — the
``BaseResourceManager``-based single-manager design.

Covers:
- :class:`KVCacheCompressionManager` contract: the five request/step hooks
  and two storage-boundary hooks
  default to no-op, zero resource counts, and it inherits
  :class:`BaseResourceManager` (so PyExecutor auto-drives it once registered).
- The resource-manager API -> lifecycle-hook translation, gated on PyExecutor's
  own signals: ``prepare_resources`` fires ``on_request_init`` on each
  request's first prefill chunk (``is_first_context_chunk``);
  ``update_resources`` fires ``on_context_step_end`` once with the
  ``context_requests_last_chunk`` list + one ``on_generation_step_end`` per
  iteration; ``free_resources`` fires ``on_request_finish``.
- :func:`create_kv_cache_compression_manager` factory.

The base class lives in ``resource_manager.py`` (it is a resource manager, not a
sparse-attention backend); the ``create_kv_cache_compression_manager`` factory
lives in ``_util.py`` next to ``_create_kv_cache_manager``.
"""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_boundary import (
    QuantizationForBoundaryCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    DataType,
    KVCacheCompressionManager,
    ResourceManager,
    ResourceManagerType,
)

# ---------------------------------------------------------------------- #
# Mock infra: in-memory managers / requests (avoid touching V2 / model).  #
# ---------------------------------------------------------------------- #


class _RecordingMixin:
    """Mixin that records every hook invocation, to assert the RM-API -> hook
    translation without real algorithm side-effects."""

    def __init__(self, kv_cache_manager, record_list, name="m"):
        super().__init__(kv_cache_manager)
        self._record_list = record_list
        self._name = name

    def _record(self, hook_name: str):
        self._record_list.append(f"{self._name}:{hook_name}")


class _MockCompressionManager(_RecordingMixin, KVCacheCompressionManager):
    """Mock manager that records the request/step lifecycle hooks."""

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_context_step_end(self, requests):
        self._record(f"on_context_step_end[{len(requests)}]")

    def on_generation_step_end(self, scheduled_batch):
        self._record("on_generation_step_end")

    def on_request_finish(self, request):
        self._record("on_request_finish")


class _LengthAdjustingCompressionManager(KVCacheCompressionManager):
    adjusts_generation_kv_length: ClassVar[bool] = True


class _KVCacheManagerV2Harness(KVCacheManagerV2):
    """Import-light V2 subtype with only the compression-owner contract."""

    def __init__(self, *, is_draft: bool = False) -> None:
        self.enable_block_reuse = False
        self.kv_compression_manages_history = False
        self.is_draft = is_draft
        self.is_disagg = False
        self.dtype = DataType.BF16
        self.tokens_per_block = 4
        self.boundary_compression_quant = "nvfp4"
        self.num_kv_heads_per_layer = [1]
        self.head_dim_per_layer = [32]
        self._boundary_compression_manager = None
        self.impl = MagicMock(name="KVCacheManagerPy")
        self.impl.pool_group_descs = []


def _v2_manager(*, is_draft: bool = False):
    return _KVCacheManagerV2Harness(is_draft=is_draft)


@pytest.fixture
def fake_kv_cache_manager():
    """A stand-in KVCacheManagerV2. The framework reads enable_block_reuse off
    it in __init__; default it to False, like a normal run with reuse off."""
    return _v2_manager(is_draft=False)


def _req(rid, first_chunk=True):
    r = MagicMock(name=f"req{rid}")
    r.py_request_id = rid
    r.is_first_context_chunk = first_chunk
    return r


def _batch(context=(), generation=(), last_chunk=()):
    b = MagicMock(name="ScheduledRequests")
    b.context_requests = list(context)
    b.generation_requests = list(generation)
    b.context_requests_last_chunk = list(last_chunk)
    return b


# ---------------------------------------------------------------------- #
# 1. KVCacheCompressionManager contract                                   #
# ---------------------------------------------------------------------- #


class TestBaseABC:
    def test_inherits_base_resource_manager(self):
        # So PyExecutor's main loop auto-invokes prepare/update/free_resources.
        assert issubclass(KVCacheCompressionManager, BaseResourceManager)

    def test_seven_hooks_default_noop(self, fake_kv_cache_manager):
        m = KVCacheCompressionManager(fake_kv_cache_manager)
        assert m.on_request_init(MagicMock()) is None
        assert m.on_context_step_end([MagicMock()]) is None
        assert m.on_generation_step_begin(MagicMock()) is None
        assert m.on_generation_step_end(MagicMock()) is None
        assert m.on_request_finish(MagicMock()) is None
        assert m.on_offload_compress(future_arg=1) is None
        assert m.on_onboard_decompress(future_arg=1) is None

    def test_hooks_accept_extra_kwargs(self, fake_kv_cache_manager):
        # **kwargs lets the framework pass new args later without breaking
        # existing overrides.
        m = KVCacheCompressionManager(fake_kv_cache_manager)
        assert m.on_request_init(MagicMock(), future_arg=1) is None
        assert m.on_generation_step_end(MagicMock(), future_arg=1) is None

    def test_resource_counts_are_zero(self, fake_kv_cache_manager):
        m = KVCacheCompressionManager(fake_kv_cache_manager)
        # The manager owns no physical resources (the V2 cache manager does),
        # so it must not gate the scheduler.
        assert m.get_max_resource_count() == 0
        assert m.get_needed_resource_to_completion(MagicMock()) == 0

    def test_length_adjustment_marks_target_and_draft_v2(self):
        # The draft cache is compacted together with the target, so both
        # managers diverge from the logical length in the same way.
        target = _v2_manager(is_draft=False)
        draft = _v2_manager(is_draft=True)

        manager = _LengthAdjustingCompressionManager(target, draft)

        assert manager.kv_cache_manager is target
        assert manager.draft_kv_cache_manager is draft
        assert manager.has_independent_draft_kv_cache
        assert target.kv_compression_manages_history is True
        assert draft.kv_compression_manages_history is True

    def test_rejects_non_v2_ownership(self):
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            KVCacheCompressionManager(MagicMock())
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            KVCacheCompressionManager(_v2_manager(is_draft=False), MagicMock())

    def test_request_field_defaults_to_zero(self):
        """LlmRequest carries the compression count (the manager's only
        channel to the runtime); a fresh request must default to 0 so runs
        without a compression manager are unchanged."""
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        from tensorrt_llm.bindings import SamplingConfig

        request = LlmRequest(
            request_id=1,
            max_new_tokens=8,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert request.py_num_compressed_tokens == 0


# ---------------------------------------------------------------------- #
# 2. Resource-manager API -> lifecycle-hook translation                   #
#    (gated on PyExecutor signals, no manager-side bookkeeping)            #
# ---------------------------------------------------------------------- #


class TestResourceManagerAPI:
    def test_target_update_receives_metadata_before_final_compression(self):
        calls = []
        metadata = MagicMock(name="attention_metadata")
        draft = MagicMock(name="draft_kv_cache_manager")
        target = MagicMock(name="target_kv_cache_manager")
        compression = MagicMock(name="compression_manager")
        draft.update_resources.side_effect = lambda *args: calls.append(("draft", args))
        target.update_resources.side_effect = lambda *args: calls.append(("target", args))
        compression.update_resources.side_effect = lambda *args: calls.append(("compression", args))
        manager = ResourceManager(
            {
                ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft,
                ResourceManagerType.KV_CACHE_MANAGER: target,
                ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression,
            }
        )
        batch = _batch(generation=[_req(1)])

        manager.update_resources(batch, metadata, 2.0)

        assert calls == [
            ("draft", (batch,)),
            ("target", (batch, metadata, 2.0)),
            ("compression", (batch,)),
        ]

    def test_real_v2_target_receives_relocation_metadata(self):
        from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kv_cache_v2_module

        target = _v2_manager(is_draft=False)
        target.kv_cache_map = {}
        batch = _batch(generation=[_req(1)])
        metadata = MagicMock(name="attention_metadata")
        manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: target})

        with patch.object(kv_cache_v2_module, "_update_kv_cache_draft_token_location") as relocate:
            manager.update_resources(batch, metadata, 2.0)

        relocate.assert_called_once_with(target, batch, metadata, 2.0)

    def test_prepare_fires_init_on_first_chunk_only(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        # First prefill chunk -> init fires.
        m.prepare_resources(_batch(context=[_req(1, first_chunk=True)]))
        # A later (non-first) chunk of the same request -> no re-init.
        m.prepare_resources(_batch(context=[_req(1, first_chunk=False)]))
        assert rec == ["s:on_request_init"]

    def test_update_fires_context_end_on_last_chunk(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        req = _req(1)
        # Final prefill chunks this iteration -> one batched context_step_end.
        req2 = _req(2)
        m.update_resources(_batch(generation=[req], last_chunk=[req, req2]))
        assert "s:on_context_step_end[2]" in rec
        assert rec[-1] == "s:on_generation_step_end"
        # Subsequent decode iteration (not in last_chunk) -> no context_step_end.
        rec.clear()
        m.update_resources(_batch(generation=[req]))
        assert rec == ["s:on_generation_step_end"]

    def test_step_end_fires_once_per_iteration(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        m.update_resources(_batch(generation=[_req(1), _req(2)]))
        assert rec.count("s:on_generation_step_end") == 1

    def test_free_fires_finish(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        m.free_resources(_req(1))
        assert rec == ["s:on_request_finish"]


# ---------------------------------------------------------------------- #
# 3. Factory                                                              #
# ---------------------------------------------------------------------- #


class TestFactory:
    def test_returns_none_when_no_algorithm_registered(self, fake_kv_cache_manager):
        # Framework-only: no concrete algorithm ships, so any config -> None.
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        assert create_kv_cache_compression_manager(cfg, fake_kv_cache_manager) is None

    def test_warns_for_unregistered_algorithm(self, fake_kv_cache_manager):
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        with patch.object(util_mod, "logger") as mock_logger:
            create_kv_cache_compression_manager(cfg, fake_kv_cache_manager)
            mock_logger.warning.assert_called_once()

    def test_factory_accepts_independent_draft_manager(self):
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        target = _v2_manager(is_draft=False)
        draft = _v2_manager(is_draft=True)

        assert (
            create_kv_cache_compression_manager(
                cfg,
                target,
                draft_kv_cache_manager=draft,
            )
            is None
        )

    def test_builds_the_one_nvfp4_boundary_manager(self):
        from tensorrt_llm._torch.kv_cache_compression.interface import KvCacheCompressionMode
        from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig

        config = KvCacheCompressionConfig(
            algorithm="quantization_for_boundary",
            quant="nvfp4",
        )
        target = _v2_manager()
        target.enable_block_reuse = True

        manager = create_kv_cache_compression_manager(config, target)

        assert type(manager) is QuantizationForBoundaryCompression
        assert manager.quant == "nvfp4"
        assert manager.kv_cache_manager is target
        assert target._boundary_compression_manager is manager
        target.impl.bind_boundary_compression_hooks.assert_called_once_with(
            manager.on_offload_compress,
            manager.on_onboard_decompress,
        )
        assert config.kv_cache_compression_mode == KvCacheCompressionMode.QUANTIZATION_FOR_BOUNDARY

    def test_boundary_config_requires_nvfp4(self):
        from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig

        with pytest.raises(ValueError, match="requires quant='nvfp4'"):
            KvCacheCompressionConfig(algorithm="quantization_for_boundary")
        with pytest.raises(ValueError, match="quant is only valid"):
            KvCacheCompressionConfig(algorithm="offload", quant="nvfp4")

    def test_eviction_method_predicate_defaults_false(self):
        # Non-evicting methods (e.g. offloading) are never restricted by the
        # speculative mode: the call-site gate reads this config predicate.
        from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig

        config = KvCacheCompressionConfig(algorithm="offload")
        assert config.kv_cache_compression_mode.is_eviction_method() is False
        m = KVCacheCompressionManager(_v2_manager(is_draft=False))
        assert not hasattr(m, "spec_config")

    def test_spec_gate_only_restricts_eviction_methods(self):
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
        from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig

        # Non-evicting methods pass with any speculative mode; no exception.
        config = KvCacheCompressionConfig(algorithm="offload")
        spec_config = SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.DFLASH)
        validate_kv_cache_compression_with_spec(config, spec_config, None)
        validate_kv_cache_compression_with_spec(config, None, None)


# ---------------------------------------------------------------------- #
# 4. Canonical names live in resource_manager, not in the sparse module   #
# ---------------------------------------------------------------------- #


class TestCanonicalImports:
    def test_names_importable_from_canonical_modules(self):
        from tensorrt_llm._torch.pyexecutor import _util, resource_manager

        # Base class stays in resource_manager (it IS a resource manager); the
        # factory lives in _util next to _create_kv_cache_manager.
        assert hasattr(resource_manager, "KVCacheCompressionManager")
        assert hasattr(_util, "create_kv_cache_compression_manager")

    def test_names_not_in_sparse_module(self):
        # The framework moved out of attention_backend/sparse/ (it is not a
        # sparse-attention backend); the sparse package no longer exports it.
        from tensorrt_llm._torch.attention_backend import sparse

        assert not hasattr(sparse, "KVCacheCompressionManager")
        assert not hasattr(sparse, "create_kv_cache_compression_manager")


# ---------------------------------------------------------------------- #
# 5. Block-reuse guard                                                    #
# ---------------------------------------------------------------------- #


class TestBlockReuseGuard:
    """__init__ refuses block reuse for a method that changes the stored keys
    and values, the same check RocketKVCacheManager makes."""

    def _mgr(self, enable_block_reuse):
        m = _v2_manager(is_draft=False)
        m.enable_block_reuse = enable_block_reuse
        return m

    def test_raises_when_reuse_on(self):
        with pytest.raises(ValueError, match="block reuse"):
            KVCacheCompressionManager(self._mgr(enable_block_reuse=True))

    def test_ok_when_reuse_off(self):
        KVCacheCompressionManager(self._mgr(enable_block_reuse=False))  # no raise


# ---------------------------------------------------------------------- #
# 6. GPU/Host NVFP4 boundary compression                                #
# ---------------------------------------------------------------------- #


class TestNVFP4BoundaryOffload:
    @staticmethod
    def _manager(**overrides):
        target = _v2_manager(is_draft=overrides.pop("is_draft", False))
        target.enable_block_reuse = overrides.pop("enable_block_reuse", True)
        target.dtype = overrides.pop("dtype", DataType.BF16)
        assert not overrides
        return (
            QuantizationForBoundaryCompression(
                target,
                quant="nvfp4",
            ),
            target,
        )

    def test_accepts_reuse_on_or_off_but_requires_raw_target_cache(self):
        manager, target = self._manager(enable_block_reuse=False)
        assert manager.kv_cache_manager is target
        with pytest.raises(ValueError, match="target KV cache"):
            self._manager(is_draft=True)
        with pytest.raises(ValueError, match="must remain FP16 or BF16"):
            self._manager(dtype=DataType.NVFP4)

    def test_rejects_draft_cache_and_second_manager(self):
        target = _v2_manager()
        draft = _v2_manager(is_draft=True)
        with pytest.raises(ValueError, match="does not support a draft"):
            QuantizationForBoundaryCompression(
                target,
                draft_kv_cache_manager=draft,
                quant="nvfp4",
            )

        QuantizationForBoundaryCompression(target, quant="nvfp4")
        with pytest.raises(ValueError, match="already bound"):
            QuantizationForBoundaryCompression(target, quant="nvfp4")

    def test_partial_page_zeroes_unused_rows_before_scale_and_quantization(self):
        manager, _ = self._manager()
        raw = torch.arange(1, 129, dtype=torch.bfloat16).reshape(4, 32)
        raw[3].fill_(4096)
        packed = torch.zeros((4, 16), dtype=torch.uint8)
        linear_scales = torch.zeros(8, dtype=torch.uint8)

        with patch.object(torch.ops.trtllm, "fp4_quantize", create=True) as quantize:
            quantize.return_value = packed, linear_scales
            compressed = manager.compress_tensor(raw, valid_token_count=3)

        compressed_packed, compressed_scales, inverse_global_scale = compressed
        assert compressed_packed is packed
        assert compressed_scales.shape == (4, 2)
        assert compressed_scales.data_ptr() == linear_scales.data_ptr()
        assert inverse_global_scale.shape == (1,)
        torch.testing.assert_close(
            inverse_global_scale,
            torch.tensor([96.0 / (448.0 * 6.0)]),
        )
        args = quantize.call_args.args
        assert args[0] is not raw
        torch.testing.assert_close(args[0][:3], raw[:3])
        assert torch.count_nonzero(args[0][3]) == 0
        assert torch.all(raw[3] == 4096)
        assert args[2:] == (16, False, False)

    def test_hnd_partial_page_zeroes_tail_for_every_head(self):
        manager, _ = self._manager()
        raw = torch.arange(1, 257, dtype=torch.bfloat16).reshape(2, 4, 32)
        raw[:, 3].fill_(4096)
        packed = torch.zeros((8, 16), dtype=torch.uint8)
        linear_scales = torch.zeros(16, dtype=torch.uint8)

        with patch.object(torch.ops.trtllm, "fp4_quantize", create=True) as quantize:
            quantize.return_value = packed, linear_scales
            manager.compress_tensor(raw, valid_token_count=3)

        quant_input = quantize.call_args.args[0].view(2, 4, 32)
        torch.testing.assert_close(quant_input[:, :3], raw[:, :3])
        assert torch.count_nonzero(quant_input[:, 3]) == 0
        assert torch.all(raw[:, 3] == 4096)

    def test_rejects_unaligned_feature_width_before_quantization(self):
        manager, _ = self._manager()
        raw = torch.ones((4, 15), dtype=torch.bfloat16)
        with patch.object(torch.ops.trtllm, "fp4_quantize", create=True) as quantize:
            with pytest.raises(ValueError, match="divisible by 16"):
                manager.compress_tensor(raw, valid_token_count=4)
        quantize.assert_not_called()

    def test_requires_explicit_valid_token_count_for_normalized_page(self):
        manager, _ = self._manager()
        raw = torch.ones((4, 16), dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="valid_token_count"):
            manager.compress_tensor(raw, valid_token_count=0)
        with pytest.raises(ValueError, match="valid_token_count"):
            manager.compress_tensor(raw, valid_token_count=5)
        with pytest.raises(ValueError, match="expects NHD"):
            manager.compress_tensor(
                raw.reshape(1, 2, 2, 16), valid_token_count=2
            )

    def test_compact_record_size_contains_payload_scales_and_global_scale(self):
        manager, _ = self._manager()
        raw_size = 4 * 32 * 2
        host_size = manager._record_size(raw_size)
        packed_size, scale_size, inverse_scale_offset = (
            manager._record_sections(raw_size, host_size)
        )
        assert packed_size == 64
        assert scale_size == 8
        assert inverse_scale_offset == 72
        assert host_size == 80
        assert host_size < raw_size
