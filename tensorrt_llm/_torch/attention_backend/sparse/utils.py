from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    elif sparse_attn_config.algorithm == "triattention":
        # TriAttention runs on the standard V2 manager; this subclass only
        # reclaims eviction-freed blocks (gated by a sentinel file -- pure
        # pass-through to KVCacheManagerV2 when block-free is disabled).
        from tensorrt_llm._torch.kv_cache_compression.triattention import (
            TriAttentionKVCacheManagerV2)
        return TriAttentionKVCacheManagerV2
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    elif sparse_attn_config.algorithm == "triattention":
        # TriAttention ships a backend only to reconcile num_cached after
        # physical eviction; decode otherwise runs the standard dense kernel.
        from tensorrt_llm._torch.kv_cache_compression.triattention import (
            TriAttentionTrtllmAttention)
        return TriAttentionTrtllmAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
