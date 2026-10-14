# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU decision tests; these do not claim native storage or inference validation."""

import importlib.util
import sys
import unittest
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "tensorrt_llm/_torch/pyexecutor/kv_cache/mixed_precision_reuse.py"
)
_SPEC = importlib.util.spec_from_file_location("_mixed_precision_reuse_test_module", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MixedPrecisionReusePolicy = _MODULE.MixedPrecisionReusePolicy
MixedPrecisionReuseDecision = _MODULE.MixedPrecisionReuseDecision
decide_mixed_precision_reuse = _MODULE.decide_mixed_precision_reuse
mixed_precision_publication_exclusions = _MODULE.mixed_precision_publication_exclusions


class MixedPrecisionReusePolicyTest(unittest.TestCase):
    def decide(self, coverage, *, mode="recompute", length=32, endpoint=32, first=4, last=8):
        return decide_mixed_precision_reuse(
            MixedPrecisionReusePolicy((1, 2), first, last, mode),
            coverage,
            reused_length=length,
            protected_length=endpoint,
            tokens_per_block=4,
        )

    def test_complete_high_keeps_entire_prefix(self):
        decision = self.decide({1: [(0, 4), (24, 32)], 2: [(0, 4), (24, 32)]})
        self.assertEqual(decision.reuse_depth, 32)
        self.assertEqual(decision.selected_groups, (1, 2))
        self.assertEqual(decision.missing_ranges, ())

    def test_missing_tail_recomputes_from_first_missing_block(self):
        decision = self.decide({1: [(0, 4), (24, 28)], 2: [(0, 4)]})
        self.assertEqual(decision.reuse_depth, 24)
        self.assertEqual(decision.missing_ranges, ((1, 28, 32), (2, 24, 32)))

    def test_missing_tail_can_explicitly_reuse_low(self):
        decision = self.decide({1: [(0, 4)], 2: [(0, 4)]}, mode="reuse_low")
        self.assertEqual(decision.reuse_depth, 32)
        self.assertEqual(decision.missing_ranges, ((1, 24, 32), (2, 24, 32)))

    def test_missing_head_is_not_mistaken_for_only_missing_tail(self):
        decision = self.decide({1: [(24, 32)], 2: [(0, 4), (24, 32)]})
        self.assertEqual(decision.reuse_depth, 0)
        self.assertEqual(decision.missing_ranges, ((1, 0, 4),))

    def test_retry_keeps_original_endpoint_and_reports_new_working_gaps(self):
        decision = self.decide({1: [(0, 4)], 2: [(0, 4)]}, length=24)
        self.assertEqual(decision.reuse_depth, 24)
        self.assertEqual(decision.missing_ranges, ((1, 16, 24), (2, 16, 24)))

    def test_short_sequence_protects_all_written_tokens(self):
        decision = self.decide({1: [(0, 4)], 2: [(0, 4)]}, length=7, endpoint=7)
        self.assertEqual(decision.reuse_depth, 4)
        self.assertEqual(decision.missing_ranges, ((1, 4, 7), (2, 4, 7)))

    def test_partial_block_gap_recomputes_whole_block(self):
        decision = self.decide({1: [(0, 4), (24, 27)], 2: [(0, 4), (24, 32)]})
        self.assertEqual(decision.reuse_depth, 24)

    def test_unaligned_endpoint_protects_intersecting_tail_block(self):
        decision = self.decide(
            {1: [(0, 4), (24, 31)], 2: [(0, 4), (24, 31)]},
            length=31,
            endpoint=31,
        )
        self.assertEqual(decision.reuse_depth, 20)
        self.assertEqual(decision.missing_ranges, ((1, 20, 24), (2, 20, 24)))

    def test_zero_claim_has_no_restore_selection(self):
        decision = self.decide({1: [], 2: []}, length=0)
        self.assertEqual(decision.reuse_depth, 0)
        self.assertEqual(decision.selected_groups, ())
        self.assertEqual(decision.missing_ranges, ())

    def test_no_protection_does_not_shorten_reuse(self):
        decision = self.decide({1: [], 2: []}, first=0, last=0)
        self.assertEqual(decision.reuse_depth, 32)
        self.assertEqual(decision.missing_ranges, ())

    def test_invalid_missing_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MixedPrecisionReusePolicy((1,), 4, 8, "invalid")

    def test_missing_group_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "configured lifecycle"):
            self.decide({1: [(0, 32)]})

    def test_native_rematch_may_return_less_than_requested_cap(self):
        decision = self.decide({1: [(0, 4)], 2: [(0, 4)]}, length=20)
        self.assertEqual(decision.reuse_depth, 20)
        self.assertEqual(decision.missing_ranges, ((1, 12, 20), (2, 12, 20)))

    def test_first_last_256_on_native_128_blocks(self):
        for mode, coverage, expected in (
            ("recompute", [(0, 256), (768, 1024)], 1024),
            ("recompute", [(0, 256)], 768),
            ("reuse_low", [(0, 256)], 1024),
        ):
            with self.subTest(mode=mode, coverage=coverage):
                decision = decide_mixed_precision_reuse(
                    MixedPrecisionReusePolicy((7,), 256, 256, mode),
                    {7: coverage},
                    reused_length=1024,
                    protected_length=1024,
                    tokens_per_block=128,
                )
                self.assertEqual(decision.reuse_depth, expected)

    def test_publication_requires_resumed_storage(self):
        with self.assertRaisesRegex(ValueError, "resumed and initialized"):
            mixed_precision_publication_exclusions(MixedPrecisionReuseDecision(32, (1,), ()))

    def test_publication_requires_missing_values_to_be_initialized(self):
        with self.assertRaisesRegex(ValueError, "resumed and initialized"):
            mixed_precision_publication_exclusions(
                MixedPrecisionReuseDecision(32, (1,), ((1, 24, 32),), selection_pending=False)
            )

    def test_original_initialized_values_have_no_publication_exclusions(self):
        self.assertEqual(
            mixed_precision_publication_exclusions(
                MixedPrecisionReuseDecision(32, (1,), (), selection_pending=False)
            ),
            (),
        )

    def test_low_derived_head_excludes_only_its_range_not_fresh_tail(self):
        copied_head = ((1, 0, 4),)
        decision = MixedPrecisionReuseDecision(
            32, (1,), (), selection_pending=False, low_copy_ranges=copied_head
        )
        self.assertEqual(mixed_precision_publication_exclusions(decision), copied_head)

    def test_invalid_bounds_and_empty_groups_are_rejected(self):
        for groups, first, last in (((), 4, 8), ((-1,), 4, 8), ((1,), -1, 8), ((1,), 4, -1)):
            with self.subTest(groups=groups, first=first, last=last), self.assertRaises(ValueError):
                MixedPrecisionReusePolicy(groups, first, last)


if __name__ == "__main__":
    unittest.main()
