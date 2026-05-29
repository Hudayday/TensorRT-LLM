"""V1↔V17 RocketKV intermediate dump utility.

Insert ``rocketkv_dump(...)`` calls at matching points in V1 (rocket.py)
and V17 (rocketkv.py). Each call writes tensors to
``${ROCKETKV_DUMP_DIR}/{stage}_layer{L}_call{N}_{field}.pt`` so an
offline diff script can compare equivalent state.

Set ``ROCKETKV_DUMP_DIR`` env var to enable (default: disabled).
"""
import os
from collections import defaultdict
from typing import Any

import torch

_call_counter = defaultdict(int)


def rocketkv_dump(stage: str, layer: int = -1, **kvs: Any) -> None:
    """No-op when ROCKETKV_DUMP_DIR is unset.

    Args:
        stage: human-readable stage tag (e.g., 'hook2_input',
            'hook2_post_topk', 'hook2_kt_after').
        layer: layer index (-1 if N/A).
        kvs: name → tensor / scalar / dict (saved with torch.save).
    """
    out = os.environ.get("ROCKETKV_DUMP_DIR")
    if not out:
        return
    # Skip when CUDA graph capture is active — torch.save inside capture
    # invalidates the graph (cudaErrorStreamCaptureInvalidated). Capture
    # happens only during warmup; inference replays the graph and Python
    # also runs in replay mode without an active capture.
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    os.makedirs(out, exist_ok=True)
    # Only dump the first call per (stage, layer) combo to keep volume
    # manageable; future calls of same (stage, layer) increment counter
    # but skip the actual save unless ROCKETKV_DUMP_ALL=1 is set.
    key = (stage, layer)
    n = _call_counter[key]
    _call_counter[key] += 1
    cap = int(os.environ.get("ROCKETKV_DUMP_MAX_CALLS_PER_KEY", "2"))
    if n >= cap:
        return
    for name, value in kvs.items():
        if value is None:
            continue
        path = os.path.join(out, f"{stage}_layer{layer}_call{n}_{name}.pt")
        try:
            if isinstance(value, torch.Tensor):
                # Save shape, dtype, first 64 elements flattened, sum/max
                # for fast diff. Use torch.save for round-trippable form.
                tensor = value.detach().cpu()
                torch.save(
                    {
                        "shape": tuple(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "head64": tensor.flatten()[:64].clone(),
                        "sum": tensor.float().sum().item()
                        if tensor.numel() < 1_000_000 else None,
                        "mean": tensor.float().mean().item()
                        if tensor.numel() < 1_000_000 else None,
                        "min": tensor.float().min().item()
                        if tensor.numel() else None,
                        "max": tensor.float().max().item()
                        if tensor.numel() else None,
                    },
                    path,
                )
            else:
                torch.save({"value": repr(value)}, path)
        except Exception as e:
            print(f"[rocketkv_dump] failed {path}: {e}", flush=True)
