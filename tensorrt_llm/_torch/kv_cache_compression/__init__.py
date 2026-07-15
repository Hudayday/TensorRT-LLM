"""KV-cache compression algorithms.

Each algorithm subpackage provides a ``BaseKVCacheCompressionManager``
subclass, dispatched from ``_util.create_kv_cache_compression_manager``.
A separate draft KV cache is compacted together with the target.
"""
