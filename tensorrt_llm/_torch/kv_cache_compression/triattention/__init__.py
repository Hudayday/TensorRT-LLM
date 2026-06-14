"""TriAttention KV-cache compression: periodic physical KV eviction driven by
trigonometric importance scoring.

Public surface:
  - ``TriAttention`` -- the ``BaseKVCacheCompressionManager`` (the eviction
    manager; runs in ``on_generation_step_end`` / ``prepare_resources``).
  - ``TriAttentionTrtllmAttention`` / ``TriAttentionTrtllmAttentionMetadata`` --
    the attention backend + metadata shim (reconciles ``num_cached`` after
    compaction).
  - ``TriAttentionKVCacheManagerV2`` -- the optional block-free ``KVCacheManagerV2``
    subclass (reclaims eviction-freed blocks; pure pass-through when disabled).
"""
from .triattention import (
    TriAttention,
    TriAttentionTrtllmAttention,
    TriAttentionTrtllmAttentionMetadata,
)
from .triattention_kv_manager import TriAttentionKVCacheManagerV2

__all__ = [
    "TriAttention",
    "TriAttentionTrtllmAttention",
    "TriAttentionTrtllmAttentionMetadata",
    "TriAttentionKVCacheManagerV2",
]
