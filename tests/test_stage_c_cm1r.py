"""Unit contracts for Stage C-M1R integrity and state semantics."""

from __future__ import annotations

import unittest

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine, ContactObservation, FailureCode
import numpy as np

from spider.tools.grab_stage_c_cm1r import _controlled_columns, _profile, effective_profile_hash, interpolate_source_state, m1_recovery_profiles, normalize_effective_profile, profile_matrix_coverage, starts_new_bumpless_episode, validate_profiles


def _observation(step: int, **override: object) -> ContactObservation:
    values: dict[str, object] = {
        "source_frame": 1461,
        "source_timestamp_s": 1461 / 120.0,
        "sim_step": step,
        "substep": step,
        "role_active": True,
        "assigned_patch": "patch:s5__cylindermedium_lift:0",
        "assigned_robot_region": "left_index_fingertip",
        "physical_contact_present": True,
        "correct_geom_pair": True,
        "geom_pair": "collision_hand_left_index_8|right_object_0",
        "patch_distance_m": 0.010,
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
    values.update(override)
    return ContactObservation(**values)  # type: ignore[arg-type]


class CM1RProfileIntegrityTest(unittest.TestCase):
    def test_effective_config_is_complete_and_object_local(self) -> None:
        profile = normalize_effective_profile(_profile("m0", "M0", seed=1))
        self.assertEqual(profile["contact_target_frame"], "object_local")
        self.assertEqual(profile["controlled_joint_set"], ("left_wrist", "left_index"))

    def test_xae_m0_finger_only_profile_excludes_every_wrist_column(self) -> None:
        profile = normalize_effective_profile(
            _profile("m0_xae", "M0", seed=1, controlled_joint_set=("left_index",))
        )
        self.assertEqual(_controlled_columns(profile), (36, 37, 38, 39))

    def test_duplicate_effective_profiles_are_rejected_even_with_new_names(self) -> None:
        first = _profile("first", "M1", seed=1)
        second = _profile("second", "M1", seed=2)
        self.assertEqual(effective_profile_hash(first), effective_profile_hash(second))
        with self.assertRaisesRegex(ValueError, "duplicate effective profile hash"):
            validate_profiles([first, second])

    def test_hysteresis_name_value_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "hysteresis_030"):
            normalize_effective_profile(_profile("e3_hysteresis_030", "M1", seed=1, retain_hysteresis_distance_m=0.025))

    def test_matrix_builder_covers_every_requested_dimension(self) -> None:
        report = profile_matrix_coverage()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["missing_dimensions"], {key: [] for key in report["requested_dimensions"]})
        self.assertEqual(report["unique_profile_count"], 216)

    def test_m1_repairs_are_bounded_and_unique(self) -> None:
        profiles = m1_recovery_profiles()
        self.assertEqual(len(profiles), 4)
        self.assertEqual(len({row["EFFECTIVE_PROFILE_HASH"] for row in profiles}), 4)
        self.assertTrue(all(row["interpolate_source_targets"] for row in profiles))
        self.assertTrue(profiles[-1]["object_motion_feedforward"])
        self.assertEqual(profiles[1]["initial_velocity"], "trajectory_forward_difference")
        self.assertEqual(profiles[2]["initial_velocity"], "contact_consistent_forward_difference")

    def test_source_interval_interpolation_preserves_endpoints_and_shortest_rotation(self) -> None:
        start = np.zeros(64, dtype=np.float64)
        end = np.zeros(64, dtype=np.float64)
        start[52:58] = [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(179.0)]
        end[52:58] = [0.016, -0.008, 0.004, 0.0, 0.0, np.deg2rad(-179.0)]
        np.testing.assert_allclose(interpolate_source_state(start, end, 0.0), start)
        np.testing.assert_allclose(interpolate_source_state(start, end, 1.0), end)
        middle = interpolate_source_state(start, end, 0.5)
        np.testing.assert_allclose(middle[52:55], [0.008, -0.004, 0.002])
        self.assertGreater(abs(middle[57]), np.deg2rad(170.0))

    def test_retain_confirmation_does_not_restart_bumpless_episode(self) -> None:
        self.assertFalse(starts_new_bumpless_episode(ContactMode.RETAIN_PENDING, ContactMode.RETAIN))
        self.assertTrue(starts_new_bumpless_episode(ContactMode.PRE_CONTACT, ContactMode.RETAIN_PENDING))
        self.assertTrue(starts_new_bumpless_episode(ContactMode.RETAIN, ContactMode.REGRASP))


class CM1RTransitionTest(unittest.TestCase):
    def test_provisional_acquire_loss_enters_regrasp_without_timeout(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=4, max_regrasp_attempts=2))
        self.assertEqual(machine.observe(_observation(0, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.030, patch_membership=False)), ContactMode.ACQUIRE)
        self.assertEqual(machine.observe(_observation(1)), ContactMode.ACQUIRE)
        self.assertEqual(machine.observe(_observation(2, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.030, patch_membership=False)), ContactMode.REGRASP)
        self.assertEqual(machine.regrasp_attempts, 1)
        self.assertIn("provisional", machine.transitions[-1].reason)

    def test_two_regrasp_attempts_are_real_and_second_can_retain(self) -> None:
        config = ContactModeConfig(confirmation_substeps=2, regrasp_timeout_ms=20, max_regrasp_attempts=2)
        machine = ContactModeMachine(config)
        machine.observe(_observation(0))
        machine.observe(_observation(1))
        machine.observe(_observation(2))
        self.assertEqual(machine.mode, ContactMode.RETAIN)
        loss = dict(physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.030, patch_membership=False)
        self.assertEqual(machine.observe(_observation(3, **loss)), ContactMode.REGRASP)
        for step in range(4, 44):
            machine.observe(_observation(step, **loss))
        self.assertEqual(machine.mode, ContactMode.REGRASP)
        self.assertEqual(machine.regrasp_attempts, 2)
        self.assertEqual(machine.observe(_observation(44)), ContactMode.REGRASP)
        self.assertEqual(machine.observe(_observation(45)), ContactMode.RETAIN)

    def test_active_role_never_releases_on_contact_loss(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=2, max_regrasp_attempts=1))
        machine.observe(_observation(0))
        machine.observe(_observation(1))
        machine.observe(_observation(2))
        loss = _observation(3, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.030, patch_membership=False)
        self.assertEqual(machine.observe(loss), ContactMode.REGRASP)
        self.assertNotEqual(machine.mode, ContactMode.RELEASE)

    def test_role_end_is_the_only_release_path(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=2))
        self.assertEqual(machine.observe(_observation(0, role_active=False)), ContactMode.RELEASE)
        self.assertEqual(machine.finish(_observation(1, role_active=False)), ContactMode.COMPLETE)


if __name__ == "__main__":
    unittest.main()
