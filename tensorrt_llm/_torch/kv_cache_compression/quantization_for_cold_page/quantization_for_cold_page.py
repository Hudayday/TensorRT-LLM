# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

from typing import TYPE_CHECKING, Any, Sequence

from tensorrt_llm.logger import logger

from ...attention.backends.interface import RopeParams
from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig


# Models whose RoPE layout the codec knows. Other models ignore the switch.
_SKIP_ROPE_QUANTIZATION_MODEL_TYPES = frozenset(
    {"deepseek_v4", "glm_moe_dsa", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}
)


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Common codec registration and callbacks for cold-page quantizers."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def __init__(
        self,
        config: "ColdPageQuantizationCompressionConfig",
        *,
        pretrained_config: "PretrainedConfig",
    ) -> None:
        super().__init__(config, pretrained_config=pretrained_config)
        selection = config.selective_compression
        self._keep_layers = frozenset(selection.keep_layers if selection else ())
        if self._keep_layers:
            num_layers = pretrained_config.get_text_config().num_hidden_layers
            if max(self._keep_layers) >= num_layers:
                raise ValueError(
                    "selective_compression.keep_layers contains an out-of-range model layer"
                )
        self._skip_rope_quantization = bool(config.skip_rope_quantization)
        model_type = getattr(pretrained_config, "model_type", None)
        if self._skip_rope_quantization and model_type not in _SKIP_ROPE_QUANTIZATION_MODEL_TYPES:
            logger.warning(
                "skip_rope_quantization is supported for model types "
                f"{sorted(_SKIP_ROPE_QUANTIZATION_MODEL_TYPES)} only; ignoring it for "
                f"{model_type!r} and compressing whole K and V vectors."
            )
            self._skip_rope_quantization = False

    def _calculate_quantized_range(
        self,
        row_elements: int,
        *,
        skip_rope: bool,
        rope: tuple[int, int] | None = None,
        key_only: bool = False,
        buffer_name: str,
    ) -> tuple[int, int]:
        """Return (start, count) of the numbers of each K or V vector that are compressed."""

        if not skip_rope:
            return 0, row_elements
        if rope is None:
            # The same RoPE width the attention layers use.
            text_config = self.pretrained_config.get_text_config()
            rope_dim = RopeParams.from_config(text_config).dim
            if key_only:  # MLA latent vector: kv_lora_rank NoPE numbers, then the RoPE numbers.
                kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
                if not isinstance(kv_lora_rank, int) or kv_lora_rank + rope_dim != row_elements:
                    raise NotImplementedError(
                        f"{buffer_name}: head_dim {row_elements} is not kv_lora_rank + "
                        f"qk_rope_head_dim ({kv_lora_rank} + {rope_dim}) of an MLA latent vector, "
                        "so its RoPE part cannot be located; skip_rope_quantization is unsupported here"
                    )
                rope = (kv_lora_rank, rope_dim)
            else:  # GQA head: the leading numbers carry RoPE.
                rope = (0, rope_dim)
        rope_start, rope_elements = rope
        if rope_start < 0 or rope_elements < 0 or rope_start + rope_elements > row_elements:
            raise ValueError(
                f"{buffer_name}: RoPE range [{rope_start}, {rope_start + rope_elements}) lies "
                f"outside the {row_elements}-element row; check partial_rotary_factor and "
                "qk_rope_head_dim in the model config"
            )
        if rope_elements == 0:
            return 0, row_elements
        if rope_elements >= row_elements:
            raise ValueError(
                f"{buffer_name}: every element is position-encoded, so skip_rope_quantization would "
                "leave nothing to quantize; the K vectors of this model are entirely RoPE"
            )
        if rope_start == 0:  # RoPE leads the row (partial-rotary GQA heads).
            start, elements = rope_elements, row_elements - rope_elements
        elif rope_start + rope_elements == row_elements:  # RoPE trails the row (MLA, DeepSeek-V4).
            start, elements = 0, rope_start
        else:
            raise ValueError(
                f"{buffer_name}: RoPE elements [{rope_start}, {rope_start + rope_elements}) sit "
                "inside the row; the kernel quantizes one contiguous range per row"
            )
        return start, elements

    def keep_layer(self, model_layer: int) -> bool:
        """Whether the entire model layer is protected, including its side buffers."""
        return model_layer in self._keep_layers

    @property
    def selects_tokens(self) -> bool:
        selection = self.config.selective_compression
        return bool(
            selection
            and (
                selection.keep_first_tokens
                or selection.keep_last_tokens
                or selection.keep_token_ranges
            )
        )

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Create one native codec with state isolated to this KVCM."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state = self.build_codec_state(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        selection = self.config.selective_compression
        if self.selects_tokens:
            return native.create_python_cold_page_codec(
                self,
                codec_state,
                selection.keep_first_tokens,
                selection.keep_last_tokens,
                selection.keep_token_ranges,
            )
        return native.create_python_cold_page_codec(self, codec_state)

    def configure(self, codec_state: Any, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers and publish each lifecycle's cold-page size."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state.lifecycles = tuple(lifecycles)
        codec_state.lifecycle_metadata = tuple(
            self.build_lifecycle_metadata(codec_state, lifecycle) for lifecycle in lifecycles
        )
        properties = []
        for hot_lifecycle, metadata in zip(codec_state.lifecycles, codec_state.lifecycle_metadata):
            lifecycle = native.ColdPageLifecycleProperties()
            lifecycle.cold_page_bytes = metadata.cold_page_bytes
            lifecycle.page_index_location = native.ColdPageIndexLocation.HOST
            if self.selects_tokens:
                lifecycle.tokens_per_page, lifecycle.max_cold_page_bytes = (
                    self.selected_storage_properties(codec_state, hot_lifecycle)
                )
            properties.append(lifecycle)
        return properties

    def selected_storage_properties(
        self, codec_state: object, lifecycle: object
    ) -> tuple[int, int]:
        """Return the logical page length and maximum encoded size for token selection."""
        raise NotImplementedError

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Build the format-specific state owned by one native codec."""
        raise NotImplementedError

    def build_lifecycle_metadata(self, codec_state: object, lifecycle: object) -> object:
        """Resolve one KVCM lifecycle into format-specific launch metadata."""
        raise NotImplementedError
