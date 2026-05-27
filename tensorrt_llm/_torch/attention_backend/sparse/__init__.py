# Concrete subclasses (TriAttention, future H2O / SnapKV / ...) are
# deliberately NOT re-exported at package level — they are accessed via the
# factory dispatcher ``create_sparse_attention_manager(config, kv_cache_manager)``
# (single-manager legacy path) or ``create_behavior_coordinator(config, ...)``
# (multi-manager Phase 4+ path), which select and instantiate the right
# subclass from the Pydantic discriminator in ``SparseAttentionConfig``. This
# mirrors the existing attention-backend dispatch pattern
# (RocketTrtllmAttention / DSACacheManager are likewise hidden behind
# get_trtllm_sparse_attn_attention_backend etc.). Users configure via Config
# objects in LLM(...); production code goes through the dispatcher, not
# direct class imports.
from .coordinator import KVCacheBehaviorCoordinator
from .kv_cache_compression_executor import (
    # Canonical names (post 2026-05-27 v17 renames)
    BaseKVCacheCompressionExecutor,
    SparseAttentionExecutor,
    # Backward-compat aliases (pre-rename names committed in `7d74c8dae6`
    # / `bfc910c02b` / `23bfff4a16` / `eaa5c71aaf`). Deprecate over v17+;
    # remove v20+. Existing imports of either old name continue to work.
    BaseKVCacheBehaviorManager,
    SparseAttentionManager,
)
from .utils import (create_behavior_coordinator,
                    create_sparse_attention_manager,
                    get_flashinfer_sparse_attn_attention_backend,
                    get_sparse_attn_kv_cache_manager,
                    get_trtllm_sparse_attn_attention_backend,
                    get_vanilla_sparse_attn_attention_backend)

__all__ = [
    # Canonical names
    "BaseKVCacheCompressionExecutor",
    "SparseAttentionExecutor",
    # Backward-compat aliases
    "BaseKVCacheBehaviorManager",
    "SparseAttentionManager",
    "KVCacheBehaviorCoordinator",
    "create_sparse_attention_manager",
    "create_behavior_coordinator",
    "get_sparse_attn_kv_cache_manager",
    "get_vanilla_sparse_attn_attention_backend",
    "get_trtllm_sparse_attn_attention_backend",
    "get_flashinfer_sparse_attn_attention_backend",
]
