"""Unit coverage for Stage C-XA evidence and decision rules."""

from __future__ import annotations

import unittest

from spider.contact.cxa_audit import (
    decide_cxa_case,
    is_functional_role,
    source_contact_reliability,
    summarize_functional_role_recall,
)


class CXAAuditTest(unittest.TestCase):
    def test_cxa_metric_audit(self) -> None:
        result = summarize_functional_role_recall([
            {"role_id": "support", "role_type": "SUPPORT", "coverage": 1.0, "passed": True},
            {"role_id": "transient", "role_type": "TRANSIENT", "coverage": 0.0, "passed": False},
        ])
        self.assertEqual(result["numerator"], 1)
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["value"], 1.0)
        self.assertEqual(result["excluded_nonfunctional_roles"][0]["role_id"], "transient")

    def test_contact_role_audit(self) -> None:
        self.assertTrue(is_functional_role("THUMB_OPPOSITION"))
        self.assertTrue(is_functional_role("STABILIZATION"))
        self.assertFalse(is_functional_role("TRANSIENT"))
        self.assertFalse(is_functional_role("NON_INTERACTING"))
        with self.assertRaises(ValueError):
            is_functional_role("invented")

    def test_patch_distance_audit(self) -> None:
        # The V2 audit labels distance as a patch-space, frame-weighted value;
        # this prevents a target-anchor value being misreported as a patch P95.
        definition = "frame-weighted Euclidean nearest distance from each selected robot fingertip to its assigned mesh-adjacent object patch"
        self.assertIn("nearest distance", definition)
        self.assertIn("object patch", definition)

    def test_source_contact_reliability(self) -> None:
        self.assertEqual(source_contact_reliability(-0.003, 0.003, 0.015)[0], "RELIABLE_SOURCE")
        self.assertEqual(source_contact_reliability(-0.020, 0.020, 0.015)[0], "UNRELIABLE_SOURCE")
        self.assertEqual(source_contact_reliability(0.020, 0.020, 0.015)[0], "NON_CONTACT")

    def test_case_a_decision(self) -> None:
        result = decide_cxa_case(
            metric_implementation_errors=["transient denominator"],
            source_contact_labeling_errors=[],
            patch_definition_errors=[],
            assignment_levels_covered=True,
            remaining_failure_reasons=["ROBOT_REACHABILITY_LIMIT"],
        )
        self.assertEqual(result["decision"], "CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG")

    def test_case_b_decision(self) -> None:
        result = decide_cxa_case(
            metric_implementation_errors=[],
            source_contact_labeling_errors=[],
            patch_definition_errors=[],
            assignment_levels_covered=True,
            remaining_failure_reasons=["ROBOT_REACHABILITY_LIMIT", "JOINT_LIMIT_CONFLICT"],
        )
        self.assertEqual(result["decision"], "CASE_B_TRUE_EMBODIMENT_INFEASIBILITY")


if __name__ == "__main__":
    unittest.main()
