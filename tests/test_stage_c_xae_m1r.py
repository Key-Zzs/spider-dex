"""Contracts for the bounded Stage C-XAE-M1R step-level audit."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from spider.tools import grab_stage_c_xae_m1r as m1r


def _row(step: int, *, contact: bool = True, blend: float = 0.25) -> dict[str, object]:
    return {
        "phase": "post", "sim_step": step, "source_frame": 1461,
        "raw_controller_correction": np.ones(4) * .01,
        "clipped_correction": np.ones(4) * .01,
        "final_correction": np.ones(4) * .01,
        "ctrl_delta": np.r_[np.zeros(36), np.ones(4) * .001, np.zeros(12)],
        "actuator_ctrl": np.r_[np.zeros(36), np.ones(4) * .1, np.zeros(24)],
        "qvel": np.r_[np.zeros(36), np.ones(4) * .2, np.zeros(24)],
        "actual_fingertip_linear_velocity_mps": np.asarray((.1, .2, .3)),
        "patch_target_linear_velocity_mps": np.asarray((.2, .2, .3)),
        "bumpless_blend_alpha": blend,
        "assigned_pair_present": contact, "physical_contact_present": contact,
        "patch_distance_m": .001, "normal_gap_m": .001, "relative_normal_velocity_mps": 0.0,
        "tangential_slip_mps": .001, "contact_force_n": 1.0,
        "penetration_m": .001, "joint_limit_valid": True,
        "fingertip_tracking_error_m": .001, "object_tracking_position_m": .001,
        "finite": True, "mode": "RETAIN", "qpos": np.zeros(64),
        "normal_impulse_ns": .0001, "friction_impulse_ns": .0001,
        "all_hand_object_contact_pairs": [],
    }


class StageCXaeM1RTelemetryTest(unittest.TestCase):
    def test_name_based_actuator_mapping_rejects_wrong_column(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body><joint name='a'/><geom size='.01'/></body></worldbody>
        <actuator><position name='l_FFJ0' joint='a'/></actuator></mujoco>""")
        self.assertEqual(m1r._actuator_mapping(model)["status"], "WRONG_ACTUATOR_MAPPING")

    def test_control_profiles_keep_left_index_only_and_forbid_regrasp(self) -> None:
        for name in ("R0_frozen_surface_aligned_baseline", "R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo", "C1_immediate_surface_correction"):
            profile = m1r._profile(name, immediate=name.startswith("C1"))
            self.assertEqual(tuple(profile["controlled_columns"]), (36, 37, 38, 39))
            self.assertFalse(profile["allow_regrasp"])

    def test_o1_reports_a_realization_delay_not_zeroed_commands(self) -> None:
        mapping = {"status": "PASS", "rows": []}
        r0 = [_row(i) for i in range(6)]
        r2 = [_row(i) for i in range(6)]
        r3 = [_row(i) for i in range(6)]
        for row in r2:
            row["actuator_ctrl"] = np.asarray(row["actuator_ctrl"]) + .002
        for row in r3:
            row["actuator_ctrl"] = np.asarray(row["actuator_ctrl"]) + .003
        result = m1r._o1_control_authority({
            "R0_frozen_surface_aligned_baseline": {"rows": r0, "mapping": mapping},
            "R2_object_motion_velocity_feedforward": {"rows": r2},
            "R3_normal_relative_velocity_servo": {"rows": r3},
        })
        self.assertEqual(result["status"], "CONTROL_DELAY_ERROR")
        self.assertTrue(result["comparisons"]["R2_object_motion_velocity_feedforward"]["commands_not_zeroed"])

    def test_no_contact_diagnostic_is_never_a_witness(self) -> None:
        result = m1r._o3_contact_ablation(
            {"rows": [_row(0, contact=False)], "contact_ablation": {"pair_count": 1}},
            {"rows": [_row(0, contact=False)]},
        )
        self.assertTrue(result["no_contact"]["DIAGNOSTIC_ONLY"])
        self.assertTrue(result["no_contact"]["NOT_A_WITNESS"])

    def test_o3_disables_only_the_assigned_explicit_pair(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>
        <body><geom name='collision_hand_left_index_8' type='sphere' size='.01'/></body>
        <body pos='.1 0 0'><geom name='right_object_0' type='sphere' size='.01'/></body>
        <body pos='.2 0 0'><geom name='other_object' type='sphere' size='.01'/></body></worldbody>
        <contact><pair geom1='collision_hand_left_index_8' geom2='right_object_0'/></contact></mujoco>""")
        result = m1r._disable_assigned_pair_only(model)
        self.assertEqual(result["explicit_pair_count"], 1)
        self.assertTrue(result["all_other_model_terms_unchanged"])

    def test_step5_gate_fails_closed_before_later_gates(self) -> None:
        rollout = {"rows": [_row(i, contact=i < 5) for i in range(9)], "warnings": [], "terminal_mode": "FAILED", "first_loss": None}
        result = m1r._gate_summary(rollout, required_steps=9, label="step-5")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["assigned_contact_through_step5"])

    def test_viewer_has_real_mesh_and_chinese_control_layers(self) -> None:
        html = m1r._viewer_html({"frames": [], "curves": []})
        self.assertIn("mesh3d", html)
        self.assertIn("真实三维", html)
        self.assertIn("actual fingertip velocity", html)


if __name__ == "__main__":
    unittest.main()
