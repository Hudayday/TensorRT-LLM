"""KV-cache compression algorithms.

Each algorithm lives in its own subpackage and provides a
``BaseKVCacheCompressionManager`` subclass (plus, optionally, an attention
backend and a ``KVCacheManagerV2`` subclass). Concrete algorithms are dispatched
from ``_util.create_kv_cache_compression_manager`` (the compression
manager) and ``attention_backend/sparse/utils.py`` (the attention backend and
KV-cache manager class).
"""

from .attention import (
    KVCacheCompressionTrtllmAttention,
    KVCacheCompressionTrtllmAttentionMetadata,
    configure_kv_cache_compression_attention_backend,
    get_kv_cache_compression_attention_backend,
    get_model_kv_cache_compression_attention_backend,
    is_kv_cache_compression_attention_backend_enabled,
    requires_kv_cache_compression_attention_backend,
)

__all__ = [
    "KVCacheCompressionTrtllmAttention",
    "KVCacheCompressionTrtllmAttentionMetadata",
    "configure_kv_cache_compression_attention_backend",
    "get_kv_cache_compression_attention_backend",
    "get_model_kv_cache_compression_attention_backend",
    "is_kv_cache_compression_attention_backend_enabled",
    "requires_kv_cache_compression_attention_backend",
]
