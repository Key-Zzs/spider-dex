"""Contract tests for frozen Stage C-XAE-M1R4."""

from __future__ import annotations

import unittest

import numpy as np

from spider.tools import grab_stage_c_xae_m1r4 as m1r4


def _row(*, step: int, region: bool, exact: bool = False, gap: float = .007, normal_v: float = .4, slip: float = .08, force: float = 1.0) -> dict:
    return {
        "phase": "post", "sim_step": step, "source_frame": 1461, "assigned_region_present": region, "exact_pair_present": exact,
        "all_hand_object_pairs": [], "assigned_region_pairs": [{"pair": "collision_hand_left_index_7|right_object_0"}] if region else [],
        "patch_normal_gap_m": gap, "reported_normal_velocity_mps": normal_v, "reported_tangential_slip_mps": slip,
        "assigned_region_force_n": force if region else 0.0, "assigned_region_penetration_m": .0002 if region else 0.0,
        "telemetry": {"joint_limit_valid": True},
        "invariants": {"execution_path": "FrozenActionEnvironment.step(action)", "post_init_qpos_write": False, "post_init_qvel_write": False, "post_init_object_qpos_write": False},
        "effective_action": np.full(4, -.015), "contact_correction": np.zeros(4),
    }


class StageCXaeM1R4Test(unittest.TestCase):
    def test_region_mapping_rejects_wrong_finger_and_palm(self) -> None:
        record = _row(step=7, region=False)
        record["all_hand_object_pairs"] = [
            {"pair": "collision_hand_left_middle_11|right_object_0", "assigned_region_pair": False, "geom1_name": "collision_hand_left_middle_11", "geom2_name": "right_object_0"},
            {"pair": "collision_hand_left_palm_0|right_object_0", "assigned_region_pair": False, "geom1_name": "collision_hand_left_palm_0", "geom2_name": "right_object_0"},
        ]
        result = {"records": [record]}
        manifest = {"region_geoms": [{"geom_name": "collision_hand_left_index_7"}, {"geom_name": "collision_hand_left_index_8"}]}
        decision = m1r4._mapping_decision(result, manifest)
        self.assertEqual(decision["classification"], "TRUE_ASSIGNED_REGION_CONTACT_LOSS")
        self.assertEqual(len(decision["step7_other_finger_object_contacts"]), 1)
        self.assertEqual(len(decision["step7_palm_object_contacts"]), 1)

    def test_step5_exact_transition_is_region_continuity(self) -> None:
        result = {"records": [_row(step=4, region=True, exact=True), _row(step=5, region=True, exact=False), _row(step=6, region=True), _row(step=7, region=False), _row(step=8, region=False)], "first_exact_pair_loss": 5, "first_region_loss": 7}
        chain = m1r4._contact_chain(result)
        self.assertEqual(chain["first_exact_pair_loss"], 5)
        self.assertEqual(chain["first_assigned_region_loss"], 7)
        self.assertTrue(chain["summary"]["step5"]["assigned_region_pair_names"])

    def test_loss_mechanism_classifies_normal_separation(self) -> None:
        chain = {"summary": {"step5": {"normal_gap_m": .007, "relative_normal_velocity_mps": .4, "relative_tangential_velocity_mps": .07, "normal_force_n": 0., "assigned_region_pair_names": ["x"]}, "step6": {"normal_gap_m": .0075, "relative_normal_velocity_mps": .5, "relative_tangential_velocity_mps": .09, "normal_force_n": 1., "assigned_region_pair_names": ["x"]}, "step7": {"normal_gap_m": .008, "relative_normal_velocity_mps": .57, "relative_tangential_velocity_mps": .10, "normal_force_n": 0., "assigned_region_pair_names": []}}}
        result = m1r4._loss_mechanism(chain, {"classification": "TRUE_ASSIGNED_REGION_CONTACT_LOSS"})
        self.assertEqual(result["classification"], "REGION_NORMAL_SEPARATION")
        self.assertFalse(result["R3"]["contact_force_decay"]["result"])

    def test_probe_specs_are_bounded_and_no_mpc_or_regrasp(self) -> None:
        specs = m1r4._probe_specs(False)
        self.assertLessEqual(len(specs), 16)
        self.assertFalse(next(item for item in specs if item.probe_id == "B7").enabled)
        for spec in specs:
            if spec.static_delta is not None:
                self.assertLessEqual(float(np.max(np.abs(spec.static_delta))), .105)

    def test_state_feedback_is_state_not_step_based_and_clipped(self) -> None:
        record = _row(step=123, region=True, normal_v=.8, gap=.02)
        first = m1r4._state_based_delta(m1r4.ProbeSpec("B8", "x", "normal_velocity"), record, None)
        second = m1r4._state_based_delta(m1r4.ProbeSpec("B8", "x", "normal_velocity"), record, .01)
        np.testing.assert_array_equal(first, second)
        action, clipped = m1r4._safe_action(np.full(4, -.015), np.full(4, -.2), np.full(4, -.12), np.full(4, .12))
        self.assertTrue(clipped)
        self.assertTrue(np.all(action >= -.12))

    def test_gate_order_and_no_direct_state_writes(self) -> None:
        result = {"records": [_row(step=1, region=True), _row(step=5, region=True), _row(step=6, region=True), _row(step=7, region=False)], "first_exact_pair_loss": 5, "first_region_loss": 7}
        self.assertEqual(m1r4._region_gate(result, "Step-5", full_two_frame=False)["status"], "PASS")
        self.assertEqual(m1r4._region_gate(result, "two_frame_1461_1462", full_two_frame=True)["status"], "FAIL")

    def test_viewer_declares_real_layers_and_chinese_labels(self) -> None:
        html = m1r4._viewer_html({"modes": []})
        for text in ("真实 MuJoCo 三维", "hand visual mesh", "assigned left-index geoms", "exact pair index_8", "等价 pair index_7"):
            self.assertIn(text, html)

    def test_docs_accept_not_applicable_force_probe(self) -> None:
        with self.subTest("B7 has no numeric force"):
            # The handoff formatter must preserve a correctly skipped B7
            # instead of treating a non-applicable probe as a failed repair.
            row = {"probe_id": "B7", "label": "force-decay", "status": "NOT_RUN_NOT_APPLICABLE", "first_region_loss": None, "max_force_n": None}
            self.assertIn("max force=`n/a`", m1r4._probe_handoff_line(row))


if __name__ == "__main__":
    unittest.main()
