"""KV-cache compression algorithms.

Each algorithm lives in its own subpackage and provides a
``BaseKVCacheCompressionManager`` subclass. Concrete algorithms are dispatched
from ``_util.create_kv_cache_compression_manager``.
"""

from .attention_metadata import (
    KVCacheCompressionTrtllmAttentionMetadata,
    get_kv_cache_compression_attention_metadata,
    requires_kv_cache_compression_attention_metadata,
    requires_paged_draft_kv_length_domain,
)

__all__ = [
    "KVCacheCompressionTrtllmAttentionMetadata",
    "get_kv_cache_compression_attention_metadata",
    "requires_kv_cache_compression_attention_metadata",
    "requires_paged_draft_kv_length_domain",
]
