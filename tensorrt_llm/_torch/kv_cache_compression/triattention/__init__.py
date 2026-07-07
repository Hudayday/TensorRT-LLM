"""TriAttention KV-cache compression: periodic physical KV eviction driven by
trigonometric importance scoring.

TriAttention is a pure KV-cache compression method. Decode still runs the model's
standard attention over the compacted cache. The generic KV-cache-compression
metadata contract keeps an uncompressed speculative draft cache on its native
length domain without changing the speculative-decoding implementation.

Public surface:
  - ``TriAttention`` -- the ``BaseKVCacheCompressionManager`` (the eviction
    manager; snapshots allocation metadata before forward and compacts the
    finalized prefix in ``on_generation_step_end``). It uses V2 capacity-only
    decode, so there is no KV-cache-manager subclass.
"""

from .triattention import TriAttention

__all__ = ["TriAttention"]
