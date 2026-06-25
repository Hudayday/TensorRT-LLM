"""TriAttention KV-cache compression: periodic physical KV eviction driven by
trigonometric importance scoring.

TriAttention is a pure KV-cache compression method -- no sparse-attention config
and no attention backend of its own. Decode runs the model's standard attention
over the compacted cache; the manager reconciles the cached-token count via the
framework's ``adjust_attention_metadata`` hook.

Public surface:
  - ``TriAttention`` -- the ``BaseKVCacheCompressionManager`` (the eviction
    manager; runs in the pre-forward ``on_generation_step_begin`` hook). Block
    reclaim goes through ``_KVCache.fork()`` from the V2 adapter's
    ``update_resources``, so there is no KV-cache-manager subclass.
"""

from .triattention import TriAttention

__all__ = [
    "TriAttention",
]
