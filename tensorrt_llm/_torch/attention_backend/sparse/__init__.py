# Concrete SparseAttentionManager subclasses (TriAttention, future H2O / SnapKV / ...)
# are deliberately NOT re-exported at package level — they are accessed via the
# factory dispatcher ``create_sparse_attention_manager(config, kv_cache_manager)``,
# which selects and instantiates the right subclass from the Pydantic
# discriminator in ``SparseAttentionConfig``. This mirrors the existing
# attention-backend dispatch pattern (RocketTrtllmAttention / DSACacheManager
# are likewise hidden behind get_trtllm_sparse_attn_attention_backend etc.).
# Users configure via Config objects in LLM(...); production code goes through
# the dispatcher, not direct class imports.
from .sparse_attention_manager import SparseAttentionManager
from .utils import (create_sparse_attention_manager,
                    get_flashinfer_sparse_attn_attention_backend,
                    get_sparse_attn_kv_cache_manager,
                    get_trtllm_sparse_attn_attention_backend,
                    get_vanilla_sparse_attn_attention_backend)

__all__ = [
    "SparseAttentionManager",
    "create_sparse_attention_manager",
    "get_sparse_attn_kv_cache_manager",
    "get_vanilla_sparse_attn_attention_backend",
    "get_trtllm_sparse_attn_attention_backend",
    "get_flashinfer_sparse_attn_attention_backend",
]
