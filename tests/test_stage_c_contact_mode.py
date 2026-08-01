"""C-M1 state-machine and C-M2 bounded-profile contract tests."""

from __future__ import annotations

import unittest

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine, ContactObservation, FailureCode
from spider.tools.grab_stage_c_contact_mode import DIAGNOSTIC_ROOT, WINDOW_SOURCE_FRAMES, _profile_specs
from spider.tools.grab_stage_c_contact_mode_viewer import _document


def _observation(**overrides: object) -> ContactObservation:
    payload: dict[str, object] = {
        "source_frame": 1461,
        "source_timestamp_s": 1461 / 120.0,
        "sim_step": 0,
        "substep": 0,
        "role_active": True,
        "assigned_patch": "patch:s5__cylindermedium_lift:0",
        "assigned_robot_region": "left_index_fingertip",
        "physical_contact_present": True,
        "correct_geom_pair": True,
        "geom_pair": "collision_hand_left_index_8|right_object_0",
        "patch_distance_m": 0.010,
        "patch_membership": True,
        "normal_cosine": 0.80,
        "tangential_slip_m": 0.0,
        "normal_gap_m": 0.0,
        "penetration_m": 0.001,
        "force_n": 1.0,
        "force_impulse_ns": 0.0,
        "joint_margin_fraction": 0.05,
        "wrist_tracking_error_m": 0.001,
        "fingertip_tracking_error_m": 0.001,
        "object_tracking_position_m": 0.001,
        "object_tracking_rotation_rad": 0.001,
        "finite": True,
        "joint_limit_valid": True,
        "warning_count": 0,
        "regrasp_attempt": 0,
    }
    payload.update(overrides)
    return ContactObservation(**payload)  # type: ignore[arg-type]


class ContactModeStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ContactModeConfig(confirmation_substeps=2, acquire_timeout_ms=20, regrasp_timeout_ms=20)

    def test_state_enum_contains_explicit_modes(self) -> None:
        self.assertEqual({mode.value for mode in ContactMode}, {"PRE_CONTACT", "ACQUIRE", "RETAIN_PENDING", "RETAIN", "RELEASE", "REGRASP", "COMPLETE", "FAILED"})

    def test_initial_contact_enters_retain_pending_and_confirmation_to_retain(self) -> None:
        machine = ContactModeMachine(self.config)
        self.assertEqual(machine.observe(_observation()), ContactMode.RETAIN_PENDING)
        self.assertEqual(machine.observe(_observation(sim_step=1)), ContactMode.RETAIN_PENDING)
        self.assertEqual(machine.observe(_observation(sim_step=2)), ContactMode.RETAIN)

    def test_no_initial_contact_enters_acquire(self) -> None:
        machine = ContactModeMachine(self.config)
        self.assertEqual(machine.observe(_observation(physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.029, patch_membership=False)), ContactMode.ACQUIRE)

    def test_retain_loss_enters_regrasp_and_success_returns_to_retain(self) -> None:
        machine = ContactModeMachine(self.config)
        machine.observe(_observation())
        machine.observe(_observation(sim_step=1))
        machine.observe(_observation(sim_step=2))
        self.assertEqual(machine.mode, ContactMode.RETAIN)
        lost = _observation(sim_step=3, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.024, patch_membership=False)
        self.assertEqual(machine.observe(lost), ContactMode.REGRASP)
        self.assertEqual(machine.observe(_observation(sim_step=4)), ContactMode.REGRASP)
        self.assertEqual(machine.observe(_observation(sim_step=5)), ContactMode.RETAIN)
        self.assertEqual(machine.regrasp_attempts, 1)

    def test_release_is_only_entered_after_role_end(self) -> None:
        machine = ContactModeMachine(self.config)
        machine.observe(_observation())
        machine.observe(_observation(sim_step=1))
        machine.observe(_observation(sim_step=2))
        self.assertNotEqual(machine.mode, ContactMode.RELEASE)
        self.assertEqual(machine.observe(_observation(sim_step=3, role_active=False)), ContactMode.RELEASE)
        self.assertEqual(machine.finish(_observation(sim_step=4, role_active=False)), ContactMode.COMPLETE)

    def test_regrasp_attempt_bound_and_timeout_failure(self) -> None:
        machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=2, regrasp_timeout_ms=20, max_regrasp_attempts=1))
        machine.observe(_observation())
        machine.observe(_observation(sim_step=1))
        machine.observe(_observation(sim_step=2))
        self.assertEqual(machine.mode, ContactMode.RETAIN)
        loss = _observation(sim_step=3, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.029, patch_membership=False)
        self.assertEqual(machine.observe(loss), ContactMode.REGRASP)
        for sim_step in range(4, 44):
            mode = machine.observe(_observation(sim_step=sim_step, physical_contact_present=False, correct_geom_pair=False, patch_distance_m=0.029, patch_membership=False))
        self.assertEqual(mode, ContactMode.FAILED)
        self.assertEqual(machine.failure_code, FailureCode.REGRASP_TIMEOUT)
        self.assertEqual(machine.regrasp_attempts, 1)

    def test_hard_safety_failure_is_fail_closed(self) -> None:
        machine = ContactModeMachine(self.config)
        self.assertEqual(machine.observe(_observation(penetration_m=0.004)), ContactMode.FAILED)
        self.assertEqual(machine.failure_code, FailureCode.PENETRATION_VIOLATION)
        force_machine = ContactModeMachine(self.config)
        self.assertEqual(force_machine.observe(_observation(force_n=151.0)), ContactMode.FAILED)
        self.assertEqual(force_machine.failure_code, FailureCode.FORCE_VIOLATION)

    def test_threshold_and_mapping_are_immutable(self) -> None:
        with self.assertRaises(ValueError):
            ContactModeConfig(patch_distance_m=0.021)
        machine = ContactModeMachine(self.config)
        self.assertEqual(machine.observe(_observation(assigned_patch="")), ContactMode.FAILED)
        self.assertEqual(machine.failure_code, FailureCode.INVALID_MAPPING)

    def test_profile_matrix_and_historical_baseline_are_bounded(self) -> None:
        profiles = _profile_specs()
        self.assertEqual(len(profiles), 12)
        self.assertEqual(len({row["profile_id"] for row in profiles}), 12)
        self.assertEqual(WINDOW_SOURCE_FRAMES.tolist(), list(range(1461, 1481)))
        self.assertTrue((DIAGNOSTIC_ROOT / "first_failure_trace.npz").is_file())

    def test_transition_timeline_serializes_modes(self) -> None:
        machine = ContactModeMachine(self.config)
        machine.observe(_observation())
        rows = machine.transition_payload()
        self.assertEqual(rows[0]["previous_mode"], "PRE_CONTACT")
        self.assertEqual(rows[0]["mode"], "RETAIN_PENDING")

    def test_viewer_contains_required_layers_and_bounded_disclaimer(self) -> None:
        payload = {
            "disclaimer": "C-M2 FAILURE DIAGNOSTIC — NOT AN ACCEPTANCE ARTIFACT",
            "selected_profile": "p",
            "series": {"p": {}},
            "summary": {"experiments": {"E1": [], "E2": [], "E3": []}},
            "baseline": {"first_failure_source_frame": 1465, "patch_distance_m": []},
            "baseline_patch_distance_mm": [],
            "layers": ["semantic patch", "actual MuJoCo contacts", "contact normals", "force vectors", "penetration", "state-mode labels"],
        }
        html = _document(payload)
        self.assertIn("C-M2 FAILURE DIAGNOSTIC", html)
        self.assertIn("actual MuJoCo contacts", html)
        self.assertIn("state-mode labels", html)
        self.assertNotIn("FULL STAGE C PASS", html)


if __name__ == "__main__":
    unittest.main()
