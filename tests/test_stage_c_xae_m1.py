"""Unit contracts for the independent Stage C-XAE-M1 retention runner."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import trimesh

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine, ContactObservation
from spider.tools import grab_stage_c_xae_m1 as m1


def observation(step: int, **changes: object) -> ContactObservation:
    values: dict[str, object] = {
        "source_frame": 1461,
        "source_timestamp_s": 1461 / 120.0,
        "sim_step": step,
        "substep": step,
        "role_active": True,
        "assigned_patch": m1.PATCH_ID,
        "assigned_robot_region": m1.REGION,
        "physical_contact_present": True,
        "correct_geom_pair": True,
        "geom_pair": "collision_hand_left_index_8|right_object_0",
        "patch_distance_m": 0.001,
        "patch_membership": True,
        "normal_cosine": 1.0,
        "tangential_slip_m": 0.0,
        "normal_gap_m": 0.0,
        "penetration_m": 0.0,
        "force_n": 1.0,
        "force_impulse_ns": 0.0,
        "joint_margin_fraction": 0.1,
        "wrist_tracking_error_m": 0.0,
        "fingertip_tracking_error_m": 0.0,
        "object_tracking_position_m": 0.0,
        "object_tracking_rotation_rad": 0.0,
        "finite": True,
        "joint_limit_valid": True,
        "warning_count": 0,
    }
    values.update(changes)
    return ContactObservation(**values)  # type: ignore[arg-type]


class StageCXaeM1TargetTest(unittest.TestCase):
    def test_nearest_surface_target_is_not_a_fixed_anchor(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            faces=np.asarray(((0, 1, 2),)), process=False,
        )
        point, normal, distance, face = m1.nearest_surface_target(
            np.asarray((1.25, 2.25, 3.4)), np.asarray((1.0, 2.0, 3.0)), np.eye(3), mesh
        )
        np.testing.assert_allclose(point, (1.25, 2.25, 3.0))
        np.testing.assert_allclose(normal, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(distance, 0.4)
        self.assertEqual(face, 0)

    def test_object_local_target_updates_with_object_pose(self) -> None:
        mesh = trimesh.Trimesh(vertices=np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))), faces=np.asarray(((0, 1, 2),)), process=False)
        first = m1.nearest_surface_target(np.asarray((.2, .2, .1)), np.zeros(3), np.eye(3), mesh)[0]
        second = m1.nearest_surface_target(np.asarray((1.2, -.8, .1)), np.asarray((1., -1., 0.)), np.eye(3), mesh)[0]
        np.testing.assert_allclose(second - first, (1., -1., 0.))

    def test_continuous_timing_handles_non_integer_substeps(self) -> None:
        self.assertEqual(m1.continuous_source_index(0.0, 6), (0, 0.0))
        index, alpha = m1.continuous_source_index(0.0005, 6)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(alpha, 0.06)
        self.assertEqual(m1.continuous_source_index(1.0 / 120.0, 6)[0], 1)

    def test_interpolation_keeps_endpoints_and_slerps_object_rotation(self) -> None:
        start = np.zeros(64); end = np.zeros(64)
        start[52:58] = (0, 0, 0, 0, 0, np.deg2rad(179))
        end[52:58] = (.01, .02, .03, 0, 0, np.deg2rad(-179))
        np.testing.assert_allclose(m1.interpolate_source_state(start, end, 0.0), start)
        np.testing.assert_allclose(m1.interpolate_source_state(start, end, 1.0), end)
        self.assertGreater(abs(m1.interpolate_source_state(start, end, .5)[57]), np.deg2rad(170))


class StageCXaeM1ModeTest(unittest.TestCase):
    def test_initial_contact_starts_retain_pending_then_retain(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=2, max_regrasp_attempts=0, allow_regrasp=False))
        self.assertEqual(machine.observe(observation(0)), ContactMode.RETAIN_PENDING)
        self.assertEqual(machine.observe(observation(1)), ContactMode.RETAIN_PENDING)
        self.assertEqual(machine.observe(observation(2)), ContactMode.RETAIN)

    def test_m1_contact_loss_fails_without_regrasp(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=2, max_regrasp_attempts=0, allow_regrasp=False))
        machine.observe(observation(0)); machine.observe(observation(1)); machine.observe(observation(2))
        lost = observation(3, physical_contact_present=False, correct_geom_pair=False, geom_pair="NONE", patch_distance_m=.03, patch_membership=False)
        self.assertEqual(machine.observe(lost), ContactMode.FAILED)
        self.assertNotIn("REGRASP", [item.mode.value for item in machine.transitions])


class StageCXaeM1IntegrityTest(unittest.TestCase):
    def test_profile_is_left_index_only_and_never_enables_regrasp(self) -> None:
        for name in ("R0_frozen_surface_aligned_baseline", "R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo", "R4_tangential_slip_compensation"):
            profile = m1._candidate_profile(name)
            self.assertEqual(profile["controlled_joint_set"], ["left_index"])
            self.assertEqual(profile["controlled_columns"], [36, 37, 38, 39])
            self.assertFalse(profile["allow_regrasp"])

    def test_runner_excludes_fixed_anchor_and_post_init_qpos_workarounds(self) -> None:
        source = inspect.getsource(m1.run_rollout)
        self.assertIn("object_qpos_written_after_initialization", source)
        self.assertIn("_set_object_mocap_reference", source)
        self.assertNotIn("anchors", source)

    def test_viewer_contains_real_mesh_and_chinese_layers(self) -> None:
        page = m1._viewer_html({"status": "FAIL", "frames": [], "curves": [], "disclaimer": "x", "layer_names": []})
        self.assertIn("mesh3d", page)
        self.assertIn("semantic patch", page)
        self.assertIn("真实 mesh", page)


if __name__ == "__main__":
    unittest.main()
