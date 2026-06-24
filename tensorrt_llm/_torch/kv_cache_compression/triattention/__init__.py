"""TriAttention KV-cache compression: periodic physical KV eviction driven by
trigonometric importance scoring.

TriAttention is a pure KV-cache compression method -- no sparse-attention config
and no attention backend of its own. Decode runs the model's standard attention
over the compacted cache; the manager reconciles the cached-token count via the
framework's ``adjust_attention_metadata`` hook.

Public surface:
  - ``TriAttention`` -- the ``BaseKVCacheCompressionManager`` (the eviction
    manager; runs in the pre-forward ``on_generation_step_begin`` hook).
  - ``TriAttentionKVCacheManagerV2`` -- the optional block-free ``KVCacheManagerV2``
    subclass (reclaims eviction-freed blocks; pure pass-through when disabled).
"""

from .triattention import TriAttention
from .triattention_kv_manager import TriAttentionKVCacheManagerV2

__all__ = [
    "TriAttention",
    "TriAttentionKVCacheManagerV2",
]
