from typing import TYPE_CHECKING, Optional

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)
from .sparse_attention_manager import SparseAttentionManager
from .triattention import TriAttention

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    """Legacy memory-layer dispatch: returns a KV-cache manager *class* whose
    instance owns sparse-aware physical storage (RocketKV / DSA / plain V1).

    Behavior-layer methods (where ``config.is_behavior_layer_method == True``)
    must not be routed through this factory; callers should short-circuit using
    that property and use the standard V2 cache manager instead, then construct
    the behavior-layer instance via :func:`create_sparse_attention_manager`.
    """
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def create_sparse_attention_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[SparseAttentionManager]:
    """Behavior-layer factory: dispatches on ``config.algorithm`` (the same
    Pydantic ``Literal`` discriminator as the rest of the sparse stack) and
    returns a fully constructed :class:`SparseAttentionManager` subclass
    instance, or ``None`` when the configured algorithm is *not* a behavior-
    layer method (RocketKV / DSA / skip_softmax stay on the legacy cache-
    manager path).

    Callers should also check ``sparse_attn_config.is_behavior_layer_method``
    to decide whether to invoke this factory at all; this function additionally
    guards against misuse by returning ``None`` for non-behavior-layer configs.

    Behavior-layer methods require :class:`KVCacheManagerV2` as the underlying
    KV-cache manager (see ``SparseAttentionManager`` docstring). The legacy V1
    manager lacks the page-table / block-read API the behavior layer depends
    on; passing it raises ``TypeError`` here so the misconfiguration is caught
    at LLM init rather than at the first eviction.
    """
    if not sparse_attn_config.is_behavior_layer_method:
        # Legacy memory-layer methods are not handled by this factory; the
        # caller's ``is_behavior_layer_method`` guard normally prevents reaching
        # here. Returning ``None`` keeps the contract symmetric and defensive.
        return None

    # Local import to avoid circular dependency at module load time.
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        KVCacheManagerV2
    if not isinstance(kv_cache_manager, KVCacheManagerV2):
        raise TypeError(
            f"Sparse attention algorithm '{sparse_attn_config.algorithm}' "
            f"requires KVCacheManagerV2 but received "
            f"{type(kv_cache_manager).__name__}. Enable V2 by setting "
            f"`KvCacheConfig.use_kv_cache_manager_v2=True` in your LLM config.")

    if sparse_attn_config.algorithm == "triattention":
        return TriAttention(
            kv_cache_manager=kv_cache_manager,
            top_B=sparse_attn_config.top_B,
            beta=sparse_attn_config.beta,
            calibration_path=sparse_attn_config.calibration_path,
        )
    raise ValueError(
        f"Unsupported behavior-layer sparse attention algorithm: "
        f"{sparse_attn_config.algorithm}")


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
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
