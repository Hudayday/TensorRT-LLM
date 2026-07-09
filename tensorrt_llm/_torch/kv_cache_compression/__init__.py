"""KV-cache compression algorithms.

Each algorithm lives in its own subpackage and provides a
``BaseKVCacheCompressionManager`` subclass. Concrete algorithms are dispatched
from ``_util.create_kv_cache_compression_manager``.
"""
