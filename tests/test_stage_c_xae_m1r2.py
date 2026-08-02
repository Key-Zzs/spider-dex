"""Unit contracts for the frozen M1R2 actuator-identification stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from spider.tools import grab_stage_c_xae_m1r2 as m1r2


def _model() -> mujoco.MjModel:
    joints = "".join(f"<joint name='{name}' type='hinge' range='-1 1'/><geom type='sphere' size='.01'/>" for name in m1r2.m1r.JOINT_NAMES)
    actuators = "".join(f"<position name='{actuator}' joint='{joint}' ctrlrange='-1 1'/>" for actuator, joint in zip(m1r2.m1r.ACTUATOR_NAMES, m1r2.m1r.JOINT_NAMES, strict=True))
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody><body>{joints}</body></worldbody><actuator>{actuators}</actuator></mujoco>")


class StageCXaeM1R2Test(unittest.TestCase):
    def test_name_mapping_resolves_joint_and_state_columns(self) -> None:
        mapping = m1r2._resolve_mapping(_model())
        self.assertEqual(mapping["status"], "PASS")
        self.assertEqual([row["actuator_name"] for row in mapping["rows"]], list(m1r2.m1r.ACTUATOR_NAMES))
        self.assertEqual(len(mapping["controlled_qvel_columns"]), 4)

    def test_safe_excitation_is_deterministic_and_split_disjoint(self) -> None:
        train_names, train = m1r2._excitation_sequences(np.full(4, .03), seed=11, split="train")
        repeat_names, repeat = m1r2._excitation_sequences(np.full(4, .03), seed=11, split="train")
        validation_names, _ = m1r2._excitation_sequences(np.full(4, .03), seed=12, split="validation")
        self.assertEqual(train_names, repeat_names)
        np.testing.assert_array_equal(train, repeat)
        self.assertFalse(set(train_names) & set(validation_names))
        self.assertLessEqual(float(np.max(np.abs(train))), .03)

    def test_fir_and_state_space_rollout_have_expected_shape(self) -> None:
        rng = np.random.default_rng(3)
        actions = rng.normal(0.0, .01, size=(4, m1r2.HORIZON, 4))
        baseline = np.zeros((m1r2.HORIZON, len(m1r2.OUTPUT_NAMES)))
        outputs = np.zeros((4, m1r2.HORIZON, len(m1r2.OUTPUT_NAMES)))
        for sample in range(len(actions)):
            for step in range(m1r2.HORIZON):
                outputs[sample, step, :4] = actions[sample, step]
                outputs[sample, step, 7] = actions[sample, : step + 1, 0].sum()
        fir = m1r2._build_fir(actions, outputs, baseline)
        state = m1r2._build_state_space(actions, outputs, baseline)
        self.assertEqual(m1r2._predict_fir(fir, actions[0], baseline).shape, baseline.shape)
        self.assertEqual(m1r2._predict_state_space(state, actions[0], baseline).shape, baseline.shape)

    def test_accuracy_gate_requires_all_frozen_metrics(self) -> None:
        good = {"fingertip_velocity": {"normalized_rmse": .09}, "normal_velocity": {"normalized_rmse": .09, "sign_accuracy": .95}, "peak_response_timing_error_steps": 1.0, "stable_prediction": True}
        self.assertTrue(m1r2._accurate(good))
        bad = {**good, "normal_velocity": {"normalized_rmse": .09, "sign_accuracy": .94}}
        self.assertFalse(m1r2._accurate(bad))

    def test_model_selection_never_prefers_an_inaccurate_lower_rmse_model(self) -> None:
        fir = {"fingertip_velocity": {"normalized_rmse": .01}, "normal_velocity": {"normalized_rmse": .01, "sign_accuracy": 1.0}, "peak_response_timing_error_steps": 2.0, "stable_prediction": True, "multi_step_rollout_rmse": .01}
        state = {"fingertip_velocity": {"normalized_rmse": .02}, "normal_velocity": {"normalized_rmse": .02, "sign_accuracy": 1.0}, "peak_response_timing_error_steps": 0.0, "stable_prediction": True, "multi_step_rollout_rmse": .02}
        self.assertEqual(m1r2._select_model({"FIR": fir, "STATE_SPACE": state}), "STATE_SPACE")

    def test_inverse_control_stays_inside_dynamic_action_bounds(self) -> None:
        model = {"model_type": "FIR_impulse_response", "order": 1, "B": np.zeros((1, 4, len(m1r2.OUTPUT_NAMES)))}
        model["B"][0, 0, 7] = -1.0
        baseline = np.zeros((m1r2.HORIZON, len(m1r2.OUTPUT_NAMES)))
        baseline[:, 7] = .1
        result = m1r2._inverse_control(model, baseline, np.full(4, -.02), np.full(4, .02))
        self.assertTrue(result["success"])
        self.assertTrue(np.all(result["actions"] <= .02 + 1e-12))
        self.assertTrue(np.all(result["actions"] >= -.02 - 1e-12))

    def test_mpc_is_prohibited_when_model_is_accurate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = m1r2._mpc_status(root, "MODEL_ACCURATE", reason="unit test")
            self.assertEqual(result["status"], "NOT_RUN_MODEL_ACCURATE")
            self.assertTrue((root / "scheme2_mpc/mpc_real_step5.json").is_file())

    def test_no_contact_result_is_never_a_witness(self) -> None:
        row = {"phase": "post", "joint_limit_valid": True, "actuator_saturation": np.zeros(8, dtype=bool), "fingertip_tracking_error_m": 0.0, "finite": True}
        rollout = {"rows": [dict(row) for _ in range(m1r2.HORIZON)], "warnings": [], "contact_enabled": False, "mapping": {"controlled_actuator_indices": [0, 1, 2, 3]}}
        result = m1r2._no_contact_validation(np.zeros((m1r2.HORIZON, 12)), np.zeros((m1r2.HORIZON, 12)), rollout)
        self.assertEqual(result["status"], "SCHEME1_NO_CONTACT_PASS")
        self.assertEqual(result["classification"], "DIAGNOSTIC_ONLY_NOT_A_WITNESS")

    def test_visual_manifest_requires_all_expected_nonempty_event_view_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screenshots = []
            for event in range(8):
                for view in ("world", "object", "wrist"):
                    target = root / f"event_{event}_{view}.png"
                    target.write_bytes(b"real-png-content")
                    screenshots.append({"event": f"event_{event}", "view": view, "path": str(target), "status": "FAIL", "returncode": -5})
            result = m1r2._visual_manifest(screenshots)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["screenshots"]), 24)
            self.assertTrue(all(item["renderer_status"] == "FAIL" and item["status"] == "PASS" for item in result["screenshots"]))


if __name__ == "__main__":
    unittest.main()
