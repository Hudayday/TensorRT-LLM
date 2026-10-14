# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal reuse decisions for full history plus original-precision windows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MixedPrecisionReusePolicy:
    """Internal window policy; register OPTIONAL tails with ``last_n + 1``.

    Native SWA windows include the next write. The extra position makes their
    block-rounded coverage include the last ``last_n`` already-written tokens.
    """

    high_groups: tuple[int, ...]
    first_n: int
    last_n: int
    missing_high: Literal["recompute", "reuse_low"] = "recompute"

    def __post_init__(self) -> None:
        if self.first_n < 0 or self.last_n < 0:
            raise ValueError("First/Last token counts must be nonnegative")
        if not self.high_groups or any(group < 0 for group in self.high_groups):
            raise ValueError("Original-precision lifecycle groups must be specified")
        if self.missing_high not in ("recompute", "reuse_low"):
            raise ValueError("missing_high must be 'recompute' or 'reuse_low'")


@dataclass(frozen=True)
class MixedPrecisionReuseDecision:
    reuse_depth: int
    selected_groups: tuple[int, ...]
    # (lifecycle group, begin, end); storage exists after resume, values may not.
    missing_ranges: tuple[tuple[int, int, int], ...]
    selection_pending: bool = True
    # Initialized by conversion from low history, not original-precision values.
    low_copy_ranges: tuple[tuple[int, int, int], ...] = ()


def decide_mixed_precision_reuse(
    policy: MixedPrecisionReusePolicy,
    coverage: Mapping[int, Sequence[tuple[int, int]]],
    *,
    reused_length: int,
    protected_length: int,
    tokens_per_block: int,
) -> MixedPrecisionReuseDecision:
    """Choose high reuse, a shorter prefill claim, or explicit low reuse.

    ``protected_length`` is the first match's endpoint, unchanged across retries.
    High coverage must describe values saved before quantization, not values
    reconstructed from the low history. Missing ranges require initialization
    before Attention; allocating their storage does not initialize their values.
    """
    if tokens_per_block <= 0 or not 0 <= reused_length <= protected_length:
        raise ValueError("Invalid block size or reuse endpoints")
    if any(group not in coverage for group in policy.high_groups):
        raise ValueError("High-window coverage is missing a configured lifecycle group")
    required = _protected_ranges(policy, reused_length, protected_length, tokens_per_block)
    missing = _missing_ranges(policy.high_groups, coverage, required)
    depth = reused_length
    if missing and policy.missing_high == "recompute":
        depth = min(begin // tokens_per_block * tokens_per_block for _, begin, _ in missing)
    # A shorter claim has its own native working window. Report its fresh pages
    # too, without moving the original protection endpoint or shortening again.
    working = _protected_ranges(policy, reused_length, reused_length, tokens_per_block)
    return MixedPrecisionReuseDecision(
        depth,
        policy.high_groups if reused_length else (),
        _missing_ranges(policy.high_groups, coverage, working),
    )


def mixed_precision_publication_exclusions(
    decision: MixedPrecisionReuseDecision,
) -> tuple[tuple[int, int, int], ...]:
    """Exclude low-derived pages, not other fresh pages in the same group."""
    if decision.selection_pending or decision.missing_ranges:
        raise ValueError("High-window storage must be resumed and initialized before publication")
    return decision.low_copy_ranges


def _protected_ranges(
    policy: MixedPrecisionReusePolicy, length: int, endpoint: int, block_size: int
) -> list[tuple[int, int]]:
    head_end = min(length, (policy.first_n + block_size - 1) // block_size * block_size)
    tail_start = max(0, endpoint - policy.last_n) // block_size * block_size
    required = [(0, head_end)] if head_end else []
    if policy.last_n and tail_start < length:
        if required and tail_start <= head_end:
            return [(0, length)]
        required.append((tail_start, length))
    return required


def _missing_ranges(
    groups: tuple[int, ...],
    coverage: Mapping[int, Sequence[tuple[int, int]]],
    required: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int, int], ...]:
    missing = []
    for group in groups:
        for begin, end in required:
            cursor = begin
            for saved_begin, saved_end in sorted(coverage[group]):
                if saved_end <= cursor or saved_begin >= end:
                    continue
                if saved_begin > cursor:
                    missing.append((group, cursor, min(saved_begin, end)))
                cursor = max(cursor, min(saved_end, end))
            if cursor < end:
                missing.append((group, cursor, end))
    return tuple(missing)
