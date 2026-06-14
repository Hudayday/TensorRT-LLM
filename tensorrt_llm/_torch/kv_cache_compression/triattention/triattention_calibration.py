"""Offline calibration utility for TriAttention.

Computes per-(layer, head, freq) statistics of the model's Q-pre-RoPE vectors
required by TriAttention's trigonometric importance score (paper Sec. 4.1/4.3).

Output schema (``torch.save`` dict; the manager's ``_resolve_calibration`` loads it
and checks ``_REQUIRED_CALIBRATION_KEYS``). Exactly the keys ``_aggregate`` saves:

  E_q           [L, H, D/2] complex   per-(layer, head, freq) mean of pre-RoPE Q
  E_q_norm      [L, H, D/2] float     per-(layer, head, freq) mean |q_f|
  omega         [D/2]       float     RoPE inverse frequencies (model inv_freq)
  freq_scale_sq [D/2]       float     squared per-freq RoPE amplitude (probed at pos 0)

Reuses TRT-LLM's existing calibration framework rather than rolling its own:
``capture_q_pre_rope_stats`` is the direct sibling of
``tensorrt_llm/models/gemma/smoothquant.py::capture_activation_range`` — same
``@torch.no_grad`` + ``model.eval()`` + ``register_forward_hook`` + calib-dataset
forward loop harness, and the same ``load_calib_dataset`` / ``CalibConfig``
building blocks. The ONLY difference is the statistic accumulated: TriAttention
needs the complex per-frequency mean of pre-RoPE Q (``E_q``) on each layer's
``self_attn.q_proj`` output, which the activation-range harness does not
collect. Pre-deployment offline utility; does not depend on modelopt.
"""

import functools
from typing import Any, Dict, List

import torch
import torch.utils.hooks
from tqdm import tqdm

from tensorrt_llm.llmapi.llm_args import CalibConfig
from tensorrt_llm.models.convert_utils import load_calib_dataset


def _to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Front/back-half complex pairing (upstream pruning_utils.to_complex_pairs,
    style="half"). ``[..., head_dim]`` -> ``[..., head_dim // 2]`` complex."""
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    t = tensor.to(torch.float32)
    freq = t.shape[-1] // 2
    return torch.complex(t[..., :freq].contiguous(), t[..., freq:].contiguous())


def _model_dims(model: torch.nn.Module) -> Dict[str, int]:
    """num_hidden_layers / num_attention_heads / head_dim from the HF config."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model.config is required to size calibration buffers")
    num_layers = int(getattr(config, "num_hidden_layers"))
    num_heads = int(getattr(config, "num_attention_heads"))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(config, "hidden_size")) // num_heads
    head_dim = int(head_dim)
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "freq_count": head_dim // 2,
    }


def _decoder_layers(model: torch.nn.Module):
    """Decoder-layer ModuleList across common HF layouts."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None) or getattr(model, "layers", None)
    if layers is None:
        raise ValueError(
            "Could not locate decoder layers (expected model.model.layers or model.layers)"
        )
    return layers


def _rope_omega(model: torch.nn.Module, freq_count: int, device: torch.device) -> torch.Tensor:
    """RoPE inv_freq from the model's rotary module, with the standard
    ``base ** (-2i/d)`` config fallback."""
    inner = getattr(model, "model", model)
    rotary = getattr(inner, "rotary_emb", None) or getattr(model, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if inv_freq is not None:
        return inv_freq.to(device=device, dtype=torch.float32)[:freq_count].clone()
    base = float(getattr(model.config, "rope_theta", 10000.0))
    head_dim = freq_count * 2
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return (1.0 / (base ** (idx / head_dim)))[:freq_count].clone()


@torch.no_grad()
def capture_q_pre_rope_stats(
    model, tokenizer, dataset, num_samples=512, seq_len=512
) -> List[Dict[str, Any]]:
    """Sibling of ``smoothquant.capture_activation_range`` — identical harness
    (``model.eval()`` + per-module ``register_forward_hook`` + calib-dataset
    forward loop), but accumulates the complex per-frequency Q-pre-RoPE mean on
    each decoder layer's ``self_attn.q_proj`` output instead of activation
    abs-max scales.

    Returns one running-accumulator dict per layer: ``sum_q`` (complex [H, F]),
    ``sum_q_norm`` (float [H, F]), ``count`` (int).
    """
    model.eval()
    device = next(model.parameters()).device
    dims = _model_dims(model)
    num_heads, head_dim, freq_count = (dims["num_heads"], dims["head_dim"], dims["freq_count"])

    accumulators: List[Dict[str, Any]] = [
        {
            "sum_q": torch.zeros(num_heads, freq_count, dtype=torch.complex64, device=device),
            "sum_q_norm": torch.zeros(num_heads, freq_count, dtype=torch.float32, device=device),
            "count": 0,
        }
        for _ in range(dims["num_layers"])
    ]

    # Mirror smoothquant's stat_input_hook: hook captures the module output
    # (q_proj's pre-RoPE Q) and folds the per-frequency complex stat in place.
    def stat_q_hook(m, x, y, layer_idx):
        q = y[0] if isinstance(y, tuple) else y
        q = q.reshape(-1, num_heads, head_dim).to(torch.float32)
        q_complex = _to_complex_pairs(q)  # [N, H, F]
        acc = accumulators[layer_idx]
        acc["sum_q"] += q_complex.sum(dim=0)
        acc["sum_q_norm"] += q_complex.abs().sum(dim=0)
        acc["count"] += int(q_complex.shape[0])

    hooks = []
    for layer_idx, layer in enumerate(_decoder_layers(model)):
        hooks.append(
            layer.self_attn.q_proj.register_forward_hook(
                functools.partial(stat_q_hook, layer_idx=layer_idx)
            )
        )

    num_samples = min(num_samples, len(dataset))
    for i in tqdm(range(num_samples), desc="calibrating TriAttention"):
        line = dataset[i]
        if isinstance(line, dict):
            line = line.get("text") or line.get("article") or next(iter(line.values()))
        input_ids = tokenizer(
            line, return_tensors="pt", max_length=seq_len, truncation=True, padding=False
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()
    return accumulators


def _freq_scale_sq(model: torch.nn.Module, freq_count: int, device: torch.device) -> torch.Tensor:
    """Squared per-frequency RoPE amplitude scaling. Port of upstream
    pruning_utils.compute_frequency_scaling: probe the model's actual rotary at
    position 0 (scale = sqrt(cos[0::2]**2 + sin[0::2]**2)), squared because
    score_keys_for_round consumes freq_scale_sq. Default RoPE -> ~ones; scaled
    RoPE (llama3 / YaRN attention_factor) -> model-specific, so it MUST be
    probed from the live rotary, never hard-coded."""
    inner = getattr(model, "model", model)
    rotary = getattr(inner, "rotary_emb", None) or getattr(model, "rotary_emb", None)
    if rotary is None:
        return torch.ones(freq_count, device=device, dtype=torch.float32)
    head_dim = freq_count * 2
    pos = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=torch.float32)
    cos, sin = rotary(probe, pos)
    cos0, sin0 = cos[0, 0], sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32).pow(2)


def _aggregate(
    accumulators: List[Dict[str, Any]], model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """Reduce running sums to the TriAttention statistic schema
    (E_q / E_q_norm / omega / freq_scale_sq)."""
    dims = _model_dims(model)
    device = next(model.parameters()).device
    L, H, F = dims["num_layers"], dims["num_heads"], dims["freq_count"]

    E_q = torch.zeros(L, H, F, dtype=torch.complex64, device=device)
    E_q_norm = torch.zeros(L, H, F, dtype=torch.float32, device=device)
    for layer_idx, acc in enumerate(accumulators):
        count = max(1, int(acc["count"]))
        E_q[layer_idx] = acc["sum_q"] / count
        E_q_norm[layer_idx] = acc["sum_q_norm"] / count

    omega = _rope_omega(model, F, device)
    freq_scale_sq = _freq_scale_sq(model, F, device)
    return {
        "E_q": E_q,
        "E_q_norm": E_q_norm,
        "omega": omega,
        "freq_scale_sq": freq_scale_sq,
    }


def compute_triattention_calibration(
    model,
    tokenizer,
    calib_config: CalibConfig,
    output_path: str,
) -> Dict[str, torch.Tensor]:
    """Compute + save TriAttention calibration stats, reusing the TRT-LLM calib
    framework: ``load_calib_dataset`` + ``CalibConfig`` + the
    ``capture_activation_range``-style forward-hook harness
    (``capture_q_pre_rope_stats``).

    Args:
        model: HF causal-LM on the calibration device; each layer's
            ``self_attn.q_proj`` output (pre-RoPE Q) is hooked.
        tokenizer: HF tokenizer for the calibration corpus.
        calib_config: TRT-LLM ``CalibConfig`` (dataset / batches / max seq len).
        output_path: destination ``.pt`` (also returned).
    """
    dataset = load_calib_dataset(calib_config.calib_dataset)
    num_samples = len(dataset) if calib_config.calib_batches == -1 else calib_config.calib_batches
    accumulators = capture_q_pre_rope_stats(
        model,
        tokenizer,
        dataset,
        num_samples=num_samples,
        seq_len=calib_config.calib_max_seq_length,
    )
    stats = _aggregate(accumulators, model)
    torch.save(stats, output_path)
    return stats
