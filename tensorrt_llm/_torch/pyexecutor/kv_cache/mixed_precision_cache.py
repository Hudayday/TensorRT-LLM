# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-history and SWA/sink storage for mixed-precision Attention.

This module does not allocate storage or decide how far to reuse. KVCM owns
both pools and their page tables; Attention owns the format-specific writes
and reads. A high window is a storage lifetime, not an Attention mask.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    AttentionReusePolicy,
    Batch,
    BatchDeviceArray,
    BufferConfig,
    DataRole,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    PageIndexConverter,
    PageIndexMode,
)


@dataclass(frozen=True)
class MixedPrecisionLayerPair:
    history_layer: LayerId
    high_layer: LayerId

    def page_tables(
        self, manager: KVCacheManager, batch: Batch
    ) -> tuple[BatchDeviceArray, BatchDeviceArray]:
        """Borrow native tables, preserving missing-page sentinels and row slots.

        The caller uses Batch.publish/wait_ready/record_read as for ordinary
        SWA. A nonnegative high index means storage exists, not that its values
        are initialized. Reuse gaps must be initialized before reading them.
        Reads select exactly one source per token; writes update history and
        any allocated high page from the same pre-quantization input.
        """
        return (
            batch.page_table(manager.get_layer_group_id(self.history_layer)),
            batch.page_table(manager.get_layer_group_id(self.high_layer)),
        )

    def buffer_addressing(
        self, manager: KVCacheManager, role: DataRole, *, high: bool
    ) -> tuple[int, int, PageIndexConverter]:
        """Existing pool base, page stride and index conversion for one format.

        Use each pool's own addressing; raw slot IDs from different pools are
        not interchangeable. Scale buffers can be queried by their own role.
        """
        layer = self.high_layer if high else self.history_layer
        return (
            int(manager.get_mem_pool_base_address(layer, role, PageIndexMode.SHARED)),
            int(manager.get_page_stride(layer, role)),
            manager.get_page_index_converter(layer, role),
        )


def add_mixed_precision_windows(
    config: KVCacheManagerConfig,
    high_buffers: Mapping[LayerId, Sequence[BufferConfig]],
    *,
    first_n: int,
    last_n: int,
) -> tuple[KVCacheManagerConfig, tuple[MixedPrecisionLayerPair, ...]]:
    """Add OPTIONAL SWA/sink layers alongside unchanged REQUIRED history.

    Keys name existing full-history layers. Buffer sizes/roles come from the
    dtype or cold-page caller, not from a quantizer-specific assumption here.
    Auxiliary model state is unchanged. Cold-page callers give only history
    layers to the quantizer; these additional layers use the native raw copy.
    Pool ratios, if supplied, describe the final native lifecycle groups.
    """
    if first_n < 0 or last_n < 0 or not (first_n or last_n):
        raise ValueError("Mixed precision requires nonnegative, nonzero First/Last counts")
    if not high_buffers:
        raise ValueError("At least one full-history layer needs a high window")
    layers = list(config.layers)
    by_id = {int(layer.layer_id): layer for layer in layers}
    next_layer_id = max(by_id, default=-1) + 1
    pairs = []
    for history_id, buffers in high_buffers.items():
        history = by_id.get(int(history_id))
        if not isinstance(history, AttentionLayerConfig) or (
            history.reuse_policy != AttentionReusePolicy.REQUIRED
            or history.sliding_window_size is not None
        ):
            raise ValueError(f"History layer {history_id} must be full REQUIRED Attention")
        if not buffers or any(buffer.is_sparse for buffer in buffers):
            raise ValueError("High windows require nonempty, nonsparse buffers")
        high_id = LayerId(next_layer_id)
        next_layer_id += 1
        layers.append(
            AttentionLayerConfig(
                layer_id=high_id,
                buffers=list(buffers),
                # Native SWA includes the next write as well as written history.
                sliding_window_size=last_n + 1,
                num_sink_tokens=first_n,
                reuse_policy=AttentionReusePolicy.OPTIONAL,
            )
        )
        pairs.append(MixedPrecisionLayerPair(LayerId(history_id), high_id))
    return replace(config, layers=layers), tuple(pairs)
