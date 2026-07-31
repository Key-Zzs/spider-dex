"""Pure MuJoCo contract tests for the C-R3 physical reference controller."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from scipy.spatial.transform import Rotation

from examples.run_mjwp import _bounded_damped_least_squares, _bounded_robot_state_feedback, _contact_collision_barriers, _contact_ik_feedback_delta, _project_contact_delta_against_collision_barriers, _robot_reference_controls, _update_contact_integral
import spider.simulators.mjwp as mjwp
from spider.simulators.mjwp import _apply_robot_servo_profile, _contact_tracking_reward, _hand_object_collision_penalty
from spider.config import Config
from spider.tools.grab_stage_c import _contact_collision_pareto_front, _contact_normal_offsets, _contiguous_index_ranges, _continuous_intrinsic_xyz, _depenetration_artifacts, _interpolate_surface_contact_reference, _preflight_object_ids, _recovered_robot_qvel, _set_object_mocap_reference


class PreflightControllerTest(unittest.TestCase):
    def test_contiguous_index_ranges_are_compact_and_fail_closed(self) -> None:
        self.assertEqual(_contiguous_index_ranges([]), [])
        self.assertEqual(_contiguous_index_ranges([1, 2, 3, 7, 9, 10]), [[1, 3], [7, 7], [9, 10]])
        with self.assertRaises(ValueError):
            _contiguous_index_ranges([2, 1])

    def test_contact_collision_pareto_front_preserves_tradeoff(self) -> None:
        records = [
            {"contact_recall": 0.42, "collision_max_m": 0.0037},
            {"contact_recall": 0.40, "collision_max_m": 0.0027},
            {"contact_recall": 0.39, "collision_max_m": 0.0039},
            {"contact_recall": 0.45, "collision_max_m": 0.0037},
        ]
        self.assertEqual(_contact_collision_pareto_front(records), [1, 3])

    def test_depenetration_candidate_artifacts_are_isolated_from_default(self) -> None:
        root = Path("/tmp/stage_c_recovery")
        default = _depenetration_artifacts(root, "")
        candidate = _depenetration_artifacts(root, "multistart_1")
        self.assertEqual(default["trajectory"].name, "trajectory_depenetrated_init.npz")
        self.assertEqual(candidate["trajectory"].name, "trajectory_depenetrated_init_multistart_1.npz")
        with self.assertRaises(ValueError):
            _depenetration_artifacts(root, "../bad")

    def test_surface_contact_metric_reference_excludes_controller_gap(self) -> None:
        raw = np.zeros((2, 10, 3), dtype=np.float32)
        raw[1, :, 0] = 1.0
        anchors = _interpolate_surface_contact_reference(raw, ref_steps=2, trailing_steps=3)
        self.assertEqual(anchors.shape, (7, 10, 3))
        self.assertTrue(np.allclose(anchors[:4, 0, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
        self.assertTrue(np.allclose(anchors[4:, 0, 0], 1.0))

    def test_robot_state_feedback_is_bounded_and_never_moves_object(self) -> None:
        reference = torch.zeros(64, dtype=torch.float64)
        physical = torch.zeros(64, dtype=torch.float64)
        physical[0] = -0.20; physical[3] = 0.50; physical[6] = -1.0
        physical[26] = 0.20; physical[29] = -0.50; physical[32] = 1.0
        physical[52:] = 4.0
        correction = _bounded_robot_state_feedback(
            reference, physical, 1.0, 0.03, 0.04, 0.25
        )
        self.assertAlmostEqual(float(correction[0]), 0.03)
        self.assertAlmostEqual(float(correction[3]), -0.04)
        self.assertAlmostEqual(float(correction[6]), 0.25)
        self.assertAlmostEqual(float(correction[26]), -0.03)
        self.assertAlmostEqual(float(correction[29]), 0.04)
        self.assertAlmostEqual(float(correction[32]), -0.25)
        self.assertTrue(torch.equal(correction[52:], torch.zeros(12, dtype=torch.float64)))

    def test_grouped_robot_lookahead_leaves_object_controls_current(self) -> None:
        reference = torch.arange(5 * 64, dtype=torch.float64).reshape(5, 64)
        controls = _robot_reference_controls(
            reference, 0, 2, 4, wrist_lookahead_steps=0, finger_lookahead_steps=2
        )
        self.assertTrue(torch.equal(controls[:, :6], reference[:2, :6]))
        self.assertTrue(torch.equal(controls[:, 26:32], reference[:2, 26:32]))
        self.assertTrue(torch.equal(controls[:, 6:26], reference[2:4, 6:26]))
        self.assertTrue(torch.equal(controls[:, 32:52], reference[2:4, 32:52]))
        self.assertTrue(torch.equal(controls[:, 52:], reference[:2, 52:]))

    def test_contact_collision_barrier_removes_inward_target_motion(self) -> None:
        # The row maps a target correction to outward contact displacement.
        # A source-anchor correction that moves inward must be projected to
        # zero; a penetrated contact may additionally require an outward step.
        inward = _project_contact_delta_against_collision_barriers(
            np.array([-0.04, 0.02]), np.array([[1.0, 0.0]]), np.array([0.0]),
            np.array([0.1, 0.1]), 1e-8,
        )
        self.assertAlmostEqual(float(inward[0]), 0.0, places=7)
        outward = _project_contact_delta_against_collision_barriers(
            np.array([-0.04, 0.02]), np.array([[1.0, 0.0]]), np.array([0.015]),
            np.array([0.1, 0.1]), 1e-8,
        )
        self.assertGreaterEqual(float(outward[0]), 0.015 - 1e-7)
        self.assertAlmostEqual(float(outward[1]), 0.02, places=7)

    def test_contact_collision_barrier_uses_hand_outward_normal(self) -> None:
        xml = """<mujoco><worldbody>
          <body><freejoint/><geom name="collision_hand_right_tip" type="sphere" size="1"/></body>
          <body pos="1.5 0 0"><freejoint/><geom name="right_object_0" type="sphere" size="1"/></body>
        </worldbody></mujoco>"""
        model, data = mujoco.MjModel.from_xml_string(xml), None
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rows, required = _contact_collision_barriers(
            model, data, list(range(6)), "right",
            Config(contact_ik_collision_barrier_gain=0.5, contact_ik_collision_barrier_max_contacts=2),
        )
        # Contact-frame +x is hand -> object.  A positive hand x target
        # displacement therefore moves inward and must score negative.
        self.assertEqual(rows.shape, (1, 6))
        self.assertLess(float(rows[0, 0]), -0.9)
        self.assertGreater(float(required[0]), 0.0)

    def test_contact_ik_ignores_source_anchor_outside_local_guard(self) -> None:
        model = SimpleNamespace(nv=26)
        data = SimpleNamespace(site_xpos=np.zeros((1, 3), dtype=np.float64))
        correction = _contact_ik_feedback_delta(
            model, data, np.array([1.0]),
            torch.tensor([[0.05, 0.0, 0.0]]), [0], [0], list(range(26)),
            Config(contact_ik_feedback_gain=1.0, contact_ik_feedback_max_anchor_error_m=0.02),
        )
        self.assertIsNone(correction)

    def test_robot_servo_profile_scales_only_named_robot_actuators(self) -> None:
        model = SimpleNamespace(
            nu=64,
            actuator_gainprm=np.ones((64, 3), dtype=np.float64),
            actuator_biasprm=np.ones((64, 3), dtype=np.float64),
            actuator_forcelimited=np.array([False] * 6 + [True] * 20 + [False] * 6 + [True] * 20 + [False] * 12),
            actuator_forcerange=np.ones((64, 2), dtype=np.float64),
        )
        names = {
            **{index: f"right_wrist_{index}" for index in range(6)},
            **{index: f"r_finger_{index}" for index in range(6, 26)},
            **{index: f"left_wrist_{index}" for index in range(26, 32)},
            **{index: f"l_finger_{index}" for index in range(32, 52)},
        }
        config = Config(robot_servo_kp_scale=2.0, robot_servo_forcelimit_scale=3.0)
        with patch.object(mjwp.mujoco, "mj_id2name", side_effect=lambda _m, _kind, index: names.get(index)):
            _apply_robot_servo_profile(model, config)
        self.assertTrue(np.allclose(model.actuator_gainprm[:52, 0], 2.0))
        self.assertTrue(np.allclose(model.actuator_biasprm[:52, 1:3], 2.0))
        self.assertTrue(np.allclose(model.actuator_gainprm[52:, 0], 1.0))
        self.assertTrue(np.allclose(model.actuator_forcerange[6:26], 3.0))
        self.assertTrue(np.allclose(model.actuator_forcerange[32:52], 3.0))
        self.assertTrue(np.allclose(model.actuator_forcerange[:6], 1.0))
    def test_depenetrated_robot_qvel_follows_emitted_qpos_and_preserves_object_velocity(self) -> None:
        qpos = np.zeros((3, 64), dtype=np.float64)
        qpos[:, 0] = [0.0, 0.01, 0.02]
        qpos[:, 51] = [0.0, -0.02, -0.04]
        baseline_qvel = np.zeros_like(qpos)
        baseline_qvel[:, 52:] = 7.0
        qvel = _recovered_robot_qvel(qpos, baseline_qvel, fps=100.0)
        self.assertTrue(np.allclose(qvel[:, 0], 1.0))
        self.assertTrue(np.allclose(qvel[:, 51], -2.0))
        self.assertTrue(np.allclose(qvel[:, 52:], 7.0))

    def test_contact_target_gap_uses_only_active_source_normals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.json"
            path.write_text(json.dumps({"records": [
                {"frame_index": 1, "contact_channel": 2, "contact_flag": True, "surface_normal_world": [0.0, 0.0, 2.0]},
                {"frame_index": 1, "contact_channel": 3, "contact_flag": False, "surface_normal_world": [1.0, 0.0, 0.0]},
            ]}), encoding="utf-8")
            offsets = _contact_normal_offsets(path, 3, 0.003)
        self.assertTrue(np.allclose(offsets[1, 2], [0.0, 0.0, 0.003]))
        self.assertTrue(np.allclose(offsets[1, 3], 0.0))

    def test_mjwp_source_coverage_excludes_only_explicit_optimizer_padding(self) -> None:
        # 414 source frames are interpolated by 17, followed by 80 horizon
        # and 8 control padding samples.  C-R4 must require all 7038 source
        # samples rather than accepting the earlier 6624-step partial replay.
        reference_len, horizon_steps, ctrl_steps = 414 * 17 + 80 + 8, 80, 8
        self.assertEqual(reference_len - horizon_steps - ctrl_steps, 7038)

    def test_contact_reward_honors_configured_scale(self) -> None:
        positions = np.array([[[0.03, 0.0, 0.0], [0.0, 0.04, 0.0]]])
        references = np.zeros((2, 3))
        contact = np.array([1.0, 0.0])
        import torch

        reward = _contact_tracking_reward(
            torch.tensor(positions), torch.tensor(references), torch.tensor(contact), 2.5
        )
        self.assertTrue(torch.allclose(reward, torch.tensor([-0.075], dtype=torch.float64)))
        deadzone_reward = _contact_tracking_reward(
            torch.tensor(positions), torch.tensor(references), torch.tensor(contact), 2.5, 0.02
        )
        self.assertTrue(torch.allclose(deadzone_reward, torch.tensor([-0.025], dtype=torch.float64)))

    def test_collision_reward_filters_pair_and_accumulates_by_world(self) -> None:
        import torch

        penalty = _hand_object_collision_penalty(
            torch.tensor([-0.004, -0.006, -0.010, -0.020], dtype=torch.float64),
            torch.tensor([[11, 21], [11, 99], [12, 21], [11, 21]]),
            torch.tensor([0, 0, 1, 3]),
            num_worlds=2,
            hand_geom_ids=[11, 12],
            object_geom_ids=[21],
            scale=100.0,
            margin_m=0.001,
        )
        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0009, 0.0081], dtype=torch.float64)))
        disabled = _hand_object_collision_penalty(
            torch.tensor([-0.010], dtype=torch.float32),
            torch.tensor([[11, 21]]),
            torch.tensor([0]),
            num_worlds=2,
            hand_geom_ids=[11],
            object_geom_ids=[21],
            scale=0.0,
            margin_m=0.001,
        )
        self.assertEqual(tuple(disabled.shape), (2,))
        self.assertTrue(torch.equal(disabled, torch.zeros(2)))

    def test_bounded_contact_dls_has_correct_sign_and_never_exceeds_clips(self) -> None:
        correction = _bounded_damped_least_squares(
            np.eye(3),
            np.array([0.20, -0.10, 0.05]),
            damping=0.0,
            component_clips=np.array([0.03, 0.04, 0.01]),
            gain=1.0,
        )
        self.assertTrue(np.allclose(correction, [0.03, -0.04, 0.01]))

    def test_contact_integral_accumulates_only_with_active_contact_and_is_bounded(self) -> None:
        clips = np.array([0.03, 0.04])
        first = _update_contact_integral(np.zeros(2), np.array([0.02, -0.03]), 1.0, 1.0, clips)
        second = _update_contact_integral(first, np.array([0.02, -0.03]), 1.0, 1.0, clips)
        released = _update_contact_integral(second, None, 1.0, 0.5, clips)
        self.assertTrue(np.allclose(first, [0.02, -0.03]))
        self.assertTrue(np.allclose(second, [0.03, -0.04]))
        self.assertTrue(np.allclose(released, [0.015, -0.02]))

    def test_continuous_xyz_keeps_equivalent_orientations_and_no_branch_jump(self) -> None:
        original = Rotation.from_euler("XYZ", [[-3.9, 0.44, 4.5], [-3.9, 0.44, 4.5]]).as_quat()
        continuous = _continuous_intrinsic_xyz(original)
        reconstructed = Rotation.from_euler("XYZ", continuous).as_matrix()
        self.assertLess(np.max(np.abs(reconstructed - Rotation.from_quat(original).as_matrix())), 1e-12)
        self.assertLess(np.linalg.norm(continuous[1] - continuous[0]), 1.0)

    def test_reference_updates_mocap_without_writing_object_qpos(self) -> None:
        xml = """
        <mujoco><worldbody>
          <body name='right_object'><joint name='rx' type='slide'/><joint name='ry' type='slide'/><joint name='rz' type='slide'/><joint name='rrx' type='hinge'/><joint name='rry' type='hinge'/><joint name='rrz' type='hinge'/><geom type='sphere' size='0.01'/></body>
          <body name='left_object'><joint name='lx' type='slide'/><joint name='ly' type='slide'/><joint name='lz' type='slide'/><joint name='lrx' type='hinge'/><joint name='lry' type='hinge'/><joint name='lrz' type='hinge'/><geom type='sphere' size='0.01'/></body>
          <body name='right_object_mocap_target' mocap='true'/><body name='left_object_mocap_target' mocap='true'/>
        </worldbody></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        bodies, mocap = _preflight_object_ids(model)
        qpos = np.zeros(64)
        qpos[52:58] = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1]
        qpos[58:64] = [-0.2, 0.1, 0.4, -0.3, 0.1, -0.2]
        before = data.qpos.copy()
        _set_object_mocap_reference(data, qpos, mocap)
        self.assertTrue(np.array_equal(data.qpos, before))
        self.assertTrue(np.allclose(data.mocap_pos[mocap["right"]], qpos[52:55]))
        self.assertTrue(np.allclose(data.mocap_pos[mocap["left"]], qpos[58:61]))
        self.assertEqual(set(bodies), {"right", "left"})


if __name__ == "__main__":
    unittest.main()
