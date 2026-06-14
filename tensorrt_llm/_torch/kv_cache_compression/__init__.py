"""KV-cache compression algorithms.

Each algorithm lives in its own subpackage and provides a
``BaseKVCacheCompressionManager`` subclass (plus, optionally, an attention
backend and a ``KVCacheManagerV2`` subclass). Concrete algorithms are dispatched
from ``resource_manager.create_kv_cache_compression_manager`` (the compression
manager) and ``attention_backend/sparse/utils.py`` (the attention backend and
KV-cache manager class).
"""
