"""Unit contracts for the M1R3 contact-truth audit."""

from __future__ import annotations

import unittest

import numpy as np

from spider.tools import grab_stage_c_xae_m1r3 as m1r3


def _record(*, exact: bool, region: bool, step: int = 5) -> dict:
    return {"phase": "post", "sim_step": step, "source_frame": 1461, "exact_pair_present": exact, "assigned_region_present": region, "any_hand_object_present": region, "patch_normal_gap_m": .007, "patch_distance_m": .007, "exact_pair_penetration_m": 0., "assigned_region_penetration_m": .0002, "any_hand_object_penetration_m": .0002, "exact_pair_force_n": 0., "assigned_region_force_n": 0., "any_hand_object_force_n": 0., "reported_penetration_m": .0002, "reported_force_n": 0., "reported_normal_velocity_mps": .45, "reported_tangential_slip_mps": .07, "exact_pairs": [], "assigned_region_pairs": [{"pair": "collision_hand_left_index_7|right_object_0"}] if region else [], "all_hand_object_pairs": [], "telemetry": {"joint_limit_valid": True, "post_init_qpos_write": False, "post_init_qvel_write": False, "post_init_object_qpos_write": False}}


class M1R3TruthTest(unittest.TestCase):
    def test_metric_domains_are_explicit(self) -> None:
        domains = m1r3._metric_domains()
        self.assertEqual(domains["status"], "PASS")
        self.assertIn("semantic patch", domains["normal_gap_m"])
        self.assertIn("any-hand-object", domains["penetration_m"])

    def test_exact_loss_with_region_contact_is_classified_not_hidden(self) -> None:
        replay = {"records": [_record(exact=True, region=True, step=4), _record(exact=False, region=True, step=5)]}
        decision = m1r3._truth_decision(replay)
        self.assertEqual(decision["classification"], "MIXED")
        self.assertEqual(decision["primary_cause"], "CONTACT_PAIR_CLASSIFICATION_ERROR")

    def test_region_gate_rejects_wrong_region_loss(self) -> None:
        replay = {"records": [_record(exact=True, region=True, step=1), _record(exact=False, region=False, step=5)], "first_exact_loss_step": 5, "first_region_loss_step": 5}
        self.assertEqual(m1r3._region_gate(replay, label="test")["status"], "FAIL")

    def test_region_gate_allows_legal_same_region_transition(self) -> None:
        replay = {"records": [_record(exact=True, region=True, step=1), _record(exact=False, region=True, step=5)], "first_exact_loss_step": 5, "first_region_loss_step": None}
        self.assertEqual(m1r3._region_gate(replay, label="test")["status"], "PASS")

    def test_two_frame_gate_requires_region_survival_beyond_step5(self) -> None:
        replay = {"records": [_record(exact=True, region=True, step=1), _record(exact=False, region=True, step=5), _record(exact=False, region=False, step=7)], "first_exact_loss_step": 5, "first_region_loss_step": 7}
        self.assertEqual(m1r3._region_gate(replay, label="two_frame_1461_1462")["status"], "FAIL")

    def test_probe_summary_remains_diagnostic(self) -> None:
        result = {"model_config": "D3_friction_zero", "model_hash": "x", "DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "first_exact_loss_step": 5, "first_region_loss_step": None, "records": [_record(exact=False, region=True)]}
        self.assertTrue(m1r3._probe_summary(result)["DIAGNOSTIC_ONLY_NOT_A_WITNESS"])

    def test_allowed_action_dimensions_are_four_only(self) -> None:
        self.assertEqual(len(m1r3.EXACT), 2)
        self.assertEqual(m1r3.HORIZON, 9)


if __name__ == "__main__":
    unittest.main()
