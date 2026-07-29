# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light tests for the KVCM V2 boundary-migration dispatch seam."""

from importlib.util import find_spec

if find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2._common import CacheTier
    from kv_cache_manager_v2._storage_manager import _select_boundary_migration_hook
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import CacheTier
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import (
        _select_boundary_migration_hook,
    )


def _offload(**kwargs) -> None:
    del kwargs


def _onboard(**kwargs) -> None:
    del kwargs


def test_selects_only_direct_gpu_host_boundaries() -> None:
    assert (
        _select_boundary_migration_hook(
            _offload,
            _onboard,
            CacheTier.GPU_MEM,
            CacheTier.HOST_MEM,
            False,
        )
        is _offload
    )
    assert (
        _select_boundary_migration_hook(
            _offload,
            _onboard,
            CacheTier.HOST_MEM,
            CacheTier.GPU_MEM,
            False,
        )
        is _onboard
    )


def test_defrag_and_non_host_migrations_keep_the_native_copy_path() -> None:
    assert (
        _select_boundary_migration_hook(
            _offload,
            _onboard,
            CacheTier.GPU_MEM,
            CacheTier.HOST_MEM,
            True,
        )
        is None
    )
    assert (
        _select_boundary_migration_hook(
            _offload,
            _onboard,
            CacheTier.HOST_MEM,
            CacheTier.DISK,
            False,
        )
        is None
    )
    assert (
        _select_boundary_migration_hook(
            None,
            None,
            CacheTier.GPU_MEM,
            CacheTier.HOST_MEM,
            False,
        )
        is None
    )
