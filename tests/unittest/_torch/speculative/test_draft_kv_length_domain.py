# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import inspect
import textwrap
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.speculative.interface import (
    SpecWorkerBase,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _CpuTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Construct real metadata state without allocating CUDA runtime buffers."""

    def __post_init__(self) -> None:
        AttentionMetadata.__post_init__(self)


def _make_metadata(
    metadata_type=_CpuTrtllmAttentionMetadata,
) -> tuple[TrtllmAttentionMetadata, Mock, Mock]:
    metadata = metadata_type(
        max_num_requests=2,
        max_num_tokens=2,
        seq_lens=None,
        num_contexts=0,
        seq_lens_kv=None,
        max_seq_len=512,
    )
    target_manager = Mock(name="target_manager")
    draft_manager = Mock(name="draft_manager")
    draft_manager.host_kv_cache_block_offsets = torch.tensor([101, 102])

    metadata._seq_lens = torch.tensor([1, 1], dtype=torch.int32)
    metadata._num_contexts = 0
    metadata.enable_flash_mla = False
    metadata.kv_cache_manager = target_manager
    metadata.draft_kv_cache_manager = draft_manager
    metadata.kv_cache_block_offsets = torch.tensor([1, 2])
    metadata.host_kv_cache_block_offsets = torch.tensor([3, 4])
    metadata.draft_kv_cache_block_offsets = torch.tensor([101, 102])
    metadata.use_distinct_draft_kv_lengths = True

    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[8, 19])
    metadata.kv_lens = torch.tensor([11, 21], dtype=torch.int32)
    metadata.kv_lens_cuda = torch.tensor([9, 20], dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([11, 21], dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = metadata.kv_lens_cuda
    metadata.host_total_kv_lens = torch.tensor([0, 32], dtype=torch.int32)

    metadata.draft_kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[100, 200]
    )
    metadata.draft_kv_lens = torch.tensor([101, 201], dtype=torch.int32)
    metadata.draft_kv_lens_cuda = torch.empty(2, dtype=torch.int32)
    metadata.draft_kv_lens_runtime = torch.tensor([101, 201], dtype=torch.int32)
    metadata.draft_kv_lens_cuda_runtime = metadata.draft_kv_lens_cuda
    metadata.draft_host_total_kv_lens = torch.tensor([0, 302], dtype=torch.int32)
    metadata._target_cached_token_lens_baseline = [10, 20]
    metadata._draft_cached_token_lens_baseline = [100, 200]
    metadata._target_kv_lens_cuda_baseline = torch.tensor([11, 21], dtype=torch.int32)
    metadata._draft_kv_lens_cuda_baseline = torch.tensor([101, 201], dtype=torch.int32)
    metadata._draft_kv_lens_cuda_delta = torch.empty(2, dtype=torch.int32)
    metadata._draft_cached_token_lens = None
    return metadata, target_manager, draft_manager


def _assert_draft_domain(metadata: TrtllmAttentionMetadata, draft_manager: Mock) -> None:
    assert metadata.kv_cache_manager is draft_manager
    torch.testing.assert_close(metadata.kv_cache_block_offsets, torch.tensor([101, 102]))
    torch.testing.assert_close(metadata.kv_lens_cuda, torch.tensor([99, 200], dtype=torch.int32))
    torch.testing.assert_close(
        metadata.kv_lens_cuda_runtime, torch.tensor([99, 200], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.kv_lens_runtime, torch.tensor([101, 201], dtype=torch.int32)
    )
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [98, 199]
    torch.testing.assert_close(
        metadata.host_total_kv_lens, torch.tensor([0, 302], dtype=torch.int32)
    )


def _assert_target_domain(metadata: TrtllmAttentionMetadata, target_manager: Mock) -> None:
    assert metadata.kv_cache_manager is target_manager
    torch.testing.assert_close(metadata.kv_cache_block_offsets, torch.tensor([1, 2]))
    torch.testing.assert_close(metadata.kv_lens_cuda, torch.tensor([9, 20], dtype=torch.int32))
    torch.testing.assert_close(
        metadata.kv_lens_cuda_runtime, torch.tensor([9, 20], dtype=torch.int32)
    )
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [8, 19]
    torch.testing.assert_close(
        metadata.host_total_kv_lens, torch.tensor([0, 32], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "target",
    [
        TrtllmAttentionMetadata.set_draft_cached_token_lens,
        TrtllmAttentionMetadata._prepare_draft_kv_length_domain,
        TrtllmAttentionMetadata.activate_draft_kv_length_domain,
        TrtllmAttentionMetadata.restore_target_kv_length_domain,
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
        SpecWorkerBase.draft_kv_cache_context,
    ],
)
def test_distinct_draft_length_contract_uses_explicit_attributes(target) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    reflection_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "hasattr", "setattr"}
    ]

    assert reflection_calls == []


def test_draft_context_switches_page_table_and_length_domain() -> None:
    metadata, target_manager, draft_manager = _make_metadata()

    with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
        _assert_draft_domain(metadata, draft_manager)
        metadata.kv_lens_cuda.add_(1)

    _assert_target_domain(metadata, target_manager)

    # A second entry must rebuild from the baselines rather than accumulating
    # the previous draft loop's in-place length changes.
    with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
        _assert_draft_domain(metadata, draft_manager)


def test_native_draft_context_does_not_change_target_length_domain() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    metadata.use_distinct_draft_kv_lengths = False
    target_kv_lens = metadata.kv_lens
    target_kv_lens_cuda = metadata.kv_lens_cuda
    target_kv_cache_params = metadata.kv_cache_params

    with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
        assert metadata.kv_cache_manager is draft_manager
        assert metadata.kv_lens is target_kv_lens
        assert metadata.kv_lens_cuda is target_kv_lens_cuda
        assert metadata.kv_cache_params is target_kv_cache_params

    assert metadata.kv_cache_manager is target_manager


def test_native_draft_context_without_block_offsets_is_a_noop() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    metadata.use_distinct_draft_kv_lengths = False
    metadata.draft_kv_cache_block_offsets = None

    with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
        _assert_target_domain(metadata, target_manager)

    _assert_target_domain(metadata, target_manager)


def test_draft_context_rejects_missing_manager_for_distinct_lengths() -> None:
    metadata, target_manager, _ = _make_metadata()

    with pytest.raises(RuntimeError, match="require a draft KV cache manager"):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, None):
            pytest.fail("context body must not run without a draft manager")

    _assert_target_domain(metadata, target_manager)


def test_draft_replay_switches_and_restores_length_domain() -> None:
    metadata, target_manager, draft_manager = _make_metadata()

    saved_state = prepare_attn_metadata_for_draft_replay(metadata, draft_manager)
    assert saved_state is not None
    _assert_draft_domain(metadata, draft_manager)

    restore_attn_metadata_after_draft_replay(metadata, saved_state)
    _assert_target_domain(metadata, target_manager)


def test_native_draft_replay_without_block_offsets_is_a_noop() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    metadata.use_distinct_draft_kv_lengths = False
    metadata.draft_kv_cache_block_offsets = None

    assert prepare_attn_metadata_for_draft_replay(metadata, draft_manager) is None
    _assert_target_domain(metadata, target_manager)


def test_draft_replay_rejects_missing_manager_for_distinct_lengths() -> None:
    metadata, target_manager, _ = _make_metadata()

    with pytest.raises(RuntimeError, match="require a draft KV cache manager"):
        prepare_attn_metadata_for_draft_replay(metadata, None)

    _assert_target_domain(metadata, target_manager)


def test_draft_replay_rejects_missing_block_offsets_for_distinct_lengths() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    metadata.draft_kv_cache_block_offsets = None

    with pytest.raises(RuntimeError, match="require draft KV cache block offsets"):
        prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    _assert_target_domain(metadata, target_manager)


def test_draft_replay_prepare_rolls_back_after_page_table_failure() -> None:
    metadata, target_manager, _ = _make_metadata()

    class FailingDraftManager:
        @property
        def host_kv_cache_block_offsets(self):
            raise RuntimeError("intentional page-table failure")

    draft_manager = FailingDraftManager()
    metadata.draft_kv_cache_manager = draft_manager

    with pytest.raises(RuntimeError, match="intentional page-table failure"):
        prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    _assert_target_domain(metadata, target_manager)


def test_draft_replay_prepare_restores_dsa_state_after_recompute_failure() -> None:
    from tensorrt_llm._torch.attention_backend.sparse import dsa

    class _CpuDSAMetadata(dsa.DSAtrtllmAttentionMetadata):
        def __init__(self, **kwargs):
            TrtllmAttentionMetadata.__init__(self, **kwargs)

        def __post_init__(self) -> None:
            AttentionMetadata.__post_init__(self)

    metadata, target_manager, _ = _make_metadata(_CpuDSAMetadata)
    draft_manager = Mock(spec=dsa.DSACacheManager, name="draft_dsa_manager")
    draft_manager.host_kv_cache_block_offsets = torch.tensor([101, 102])
    draft_manager.index_head_dim = 128
    metadata.draft_kv_cache_manager = draft_manager
    metadata.host_indexer_k_cache_block_offsets = torch.tensor([[1, 2], [3, 4]])
    metadata.indexer_k_cache_block_offsets = torch.tensor([[5, 6], [7, 8]])
    metadata.host_slot_mapping_fp8 = torch.tensor([9, 10])
    metadata.host_slot_mapping_scale = torch.tensor([11, 12])
    metadata.slot_mapping_fp8 = torch.tensor([13, 14])
    metadata.slot_mapping_scale = torch.tensor([15, 16])
    metadata._get_pool_block_indices = Mock(return_value=torch.tensor([[20, 21], [22, 23]]))
    original_host_offsets = metadata.host_indexer_k_cache_block_offsets.clone()
    original_device_offsets = metadata.indexer_k_cache_block_offsets.clone()
    original_host_fp8 = metadata.host_slot_mapping_fp8.clone()
    original_host_scale = metadata.host_slot_mapping_scale.clone()
    original_device_fp8 = metadata.slot_mapping_fp8.clone()
    original_device_scale = metadata.slot_mapping_scale.clone()

    with (
        patch.object(
            dsa.Indexer,
            "recompute_slot_mappings",
            side_effect=RuntimeError("intentional DSA recompute failure"),
        ),
        pytest.raises(RuntimeError, match="intentional DSA recompute failure"),
    ):
        prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    _assert_target_domain(metadata, target_manager)
    torch.testing.assert_close(metadata.host_indexer_k_cache_block_offsets, original_host_offsets)
    torch.testing.assert_close(metadata.indexer_k_cache_block_offsets, original_device_offsets)
    torch.testing.assert_close(metadata.host_slot_mapping_fp8, original_host_fp8)
    torch.testing.assert_close(metadata.host_slot_mapping_scale, original_host_scale)
    torch.testing.assert_close(metadata.slot_mapping_fp8, original_device_fp8)
    torch.testing.assert_close(metadata.slot_mapping_scale, original_device_scale)


def test_activate_draft_length_domain_is_exception_atomic() -> None:
    metadata, target_manager, _ = _make_metadata()
    original_update = metadata.on_update_kv_lens
    calls = 0

    def fail_first_update():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("intentional length-domain failure")
        original_update()

    metadata.on_update_kv_lens = fail_first_update

    with pytest.raises(RuntimeError, match="intentional length-domain failure"):
        metadata.activate_draft_kv_length_domain()

    _assert_target_domain(metadata, target_manager)


def test_triattention_uses_distinct_draft_lengths_before_first_eviction() -> None:
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention

    metadata, _, draft_manager = _make_metadata()
    metadata.request_ids = [7, 8]
    metadata.prompt_lens = [0, 0]
    metadata.use_distinct_draft_kv_lengths = False
    metadata._draft_cached_token_lens = None
    target_manager = Mock(name="target_manager")
    target_manager.enable_block_reuse = False
    manager = TriAttention(
        target_manager,
        top_B=8,
        draft_kv_cache_manager=draft_manager,
        skip_swa=False,
    )
    manager._evicted = {}

    manager.adjust_attention_metadata(metadata)

    assert metadata.use_distinct_draft_kv_lengths
    assert metadata._draft_cached_token_lens == [8, 19]
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [8, 19]


@CUDA_REQUIRED
def test_cuda_graph_draft_nodes_keep_the_distinct_length_pointer() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    device = torch.device("cuda")
    metadata.kv_lens_cuda = torch.tensor([11, 21], dtype=torch.int32, device=device)
    metadata.kv_lens_cuda_runtime = metadata.kv_lens_cuda
    metadata.draft_kv_lens_cuda = torch.empty(2, dtype=torch.int32, device=device)
    metadata.draft_kv_lens_cuda_runtime = metadata.draft_kv_lens_cuda
    metadata._target_kv_lens_cuda_baseline = torch.tensor(
        [11, 21], dtype=torch.int32, device=device
    )
    metadata._draft_kv_lens_cuda_baseline = torch.tensor(
        [101, 201], dtype=torch.int32, device=device
    )
    metadata._draft_kv_lens_cuda_delta = torch.empty(2, dtype=torch.int32, device=device)
    metadata.kv_cache_block_offsets = torch.tensor([1, 2], device=device)
    metadata.draft_kv_cache_block_offsets = torch.tensor([101, 102], device=device)
    output = torch.empty(2, dtype=torch.int32, device=device)

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
            output.copy_(metadata.kv_lens_cuda)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
            output.copy_(metadata.kv_lens_cuda)

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_lens_cuda.data_ptr() != metadata.draft_kv_lens_cuda.data_ptr()
    metadata.kv_lens_cuda.copy_(torch.tensor([41, 42], dtype=torch.int32, device=device))
    output.zero_()
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output, torch.tensor([131, 222], dtype=torch.int32, device=device))
    assert metadata.kv_cache_manager is target_manager


def test_set_draft_cached_token_lens_is_explicit_and_validated() -> None:
    metadata, _, _ = _make_metadata()

    metadata.set_draft_cached_token_lens([30, 40])
    assert metadata._draft_cached_token_lens == [30, 40]

    with pytest.raises(ValueError, match="one value per sequence"):
        metadata.set_draft_cached_token_lens([30])
    with pytest.raises(ValueError, match="non-negative"):
        metadata.set_draft_cached_token_lens([30, -1])

    metadata.draft_kv_cache_manager = None
    with pytest.raises(RuntimeError, match="separate draft KV cache manager"):
        metadata.set_draft_cached_token_lens([30, 40])


def test_prepare_builds_draft_lengths_from_preserved_cached_prefix() -> None:
    metadata, _, draft_manager = _make_metadata()
    draft_manager.max_seq_len = 512
    metadata._seq_lens_kv = torch.tensor([1, 2], dtype=torch.int32)
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[5, 6],
        num_extra_kv_tokens=3,
    )
    metadata.kv_lens_cuda = torch.tensor([6, 8], dtype=torch.int32)
    metadata.draft_kv_lens = torch.empty(2, dtype=torch.int32)
    metadata.draft_kv_lens_cuda = torch.empty(2, dtype=torch.int32)
    metadata._draft_kv_lens_cuda_baseline = torch.empty(2, dtype=torch.int32)
    metadata._target_kv_lens_cuda_baseline = torch.empty(2, dtype=torch.int32)
    metadata.draft_host_total_kv_lens = torch.empty(2, dtype=torch.int32)

    metadata.set_draft_cached_token_lens([50, 60])
    metadata._prepare_draft_kv_length_domain()

    torch.testing.assert_close(metadata.draft_kv_lens, torch.tensor([54, 65], dtype=torch.int32))
    torch.testing.assert_close(
        metadata.draft_kv_lens_cuda, torch.tensor([51, 62], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata._target_kv_lens_cuda_baseline, torch.tensor([6, 8], dtype=torch.int32)
    )
    assert metadata.draft_kv_cache_params.num_cached_tokens_per_seq == [50, 60]
    assert metadata._draft_cached_token_lens is None

    with pytest.raises(RuntimeError, match=r"set_draft_cached_token_lens\(\)"):
        metadata._prepare_draft_kv_length_domain()


def test_prepare_distinct_lengths_requires_draft_manager() -> None:
    metadata, _, _ = _make_metadata()
    metadata.draft_kv_cache_manager = None
    metadata._draft_cached_token_lens = [50, 60]

    with pytest.raises(RuntimeError, match="separate draft KV cache manager"):
        metadata._prepare_draft_kv_length_domain()


def test_draft_context_rejects_a_different_manager() -> None:
    metadata, _, _ = _make_metadata()

    with pytest.raises(RuntimeError, match="different draft KV cache manager"):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, Mock()):
            pass


def test_draft_context_rejects_missing_block_offsets_for_distinct_lengths() -> None:
    metadata, target_manager, draft_manager = _make_metadata()
    metadata.draft_kv_cache_block_offsets = None

    with pytest.raises(RuntimeError, match="require draft KV cache block offsets"):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
            pytest.fail("context body must not run without a draft page table")

    _assert_target_domain(metadata, target_manager)


def test_draft_context_restores_target_domain_after_exception() -> None:
    metadata, target_manager, draft_manager = _make_metadata()

    with pytest.raises(RuntimeError, match="intentional"):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
            _assert_draft_domain(metadata, draft_manager)
            raise RuntimeError("intentional")

    _assert_target_domain(metadata, target_manager)


def test_draft_context_rolls_back_when_setup_raises() -> None:
    metadata, target_manager, _ = _make_metadata()

    class FailingDraftManager:
        @property
        def host_kv_cache_block_offsets(self):
            raise RuntimeError("intentional draft context setup failure")

    draft_manager = FailingDraftManager()
    metadata.draft_kv_cache_manager = draft_manager

    with pytest.raises(RuntimeError, match="intentional draft context setup failure"):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, draft_manager):
            pytest.fail("context body must not run after setup failure")

    _assert_target_domain(metadata, target_manager)
