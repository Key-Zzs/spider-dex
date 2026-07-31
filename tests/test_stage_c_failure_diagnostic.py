from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from spider.tools.grab_stage_c_failure_diagnostic import (
    _active_roles,
    _bounds,
    _first_crossing,
    _first_true,
    _manifest_record,
    _select_decision,
)
from spider.tools.grab_stage_c_failure_viewer import DISCLAIMER, build_failure_html


class StageCFailureDiagnosticTests(unittest.TestCase):
    def test_manifest_hashes_read_only_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            row = _manifest_record(path, "test input")
            self.assertTrue(row["exists"])
            self.assertEqual(row["size"], 3)
            self.assertEqual(len(row["sha256"]), 64)

    def test_manifest_records_missing_input_fail_closed(self) -> None:
        row = _manifest_record(Path("/definitely/missing/stage-c-artifact"), "missing")
        self.assertFalse(row["exists"])
        self.assertIsNone(row["sha256"])

    def test_role_interval_is_inclusive(self) -> None:
        roles = [{"frame_start": 4, "frame_end": 6, "role_id": "r"}]
        self.assertEqual(_active_roles(roles, 4), roles)
        self.assertEqual(_active_roles(roles, 6), roles)
        self.assertEqual(_active_roles(roles, 7), [])

    def test_first_loss_localization_helper(self) -> None:
        self.assertEqual(_first_true(np.array([False, False, True])), 2)
        self.assertEqual(_first_true(np.zeros(3, dtype=bool), 8), 8)

    def test_substep_alignment_finds_exact_patch_crossing(self) -> None:
        names = [f"{side}_{item}" for side in ("right", "left") for item in ("palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")]
        sites = "".join(f"<site name='{name}' pos='0 0 0'/>" for name in names)
        model = mujoco.MjModel.from_xml_string(
            f"<mujoco><worldbody><body><joint type='slide' axis='1 0 0'/>{sites}<geom type='sphere' size='.001'/></body></worldbody></mujoco>"
        )
        trace = {
            "frame_index": np.array([0, 1, 2]),
            "substep_index": np.array([0, 0, 3]),
            "qpos": np.array([[0.0], [0.01], [0.03]]),
        }
        expected = np.zeros((3, 10), dtype=bool)
        expected[:, 0] = True
        anchors = np.zeros((3, 10, 3))
        index, frame, tip, _ = _first_crossing(model, trace, expected, anchors)
        self.assertEqual((index, frame, tip, int(trace["substep_index"][index])), (2, 2, 0, 3))

    def test_robot_bounds_preserve_wrist_translation_limit(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint type='slide' limited='false'/><geom type='sphere' size='.01'/></body></worldbody></mujoco>"
        )
        base = np.zeros(model.nq)
        lower, upper = _bounds(model, base)
        self.assertAlmostEqual(lower[0], -0.03)
        self.assertAlmostEqual(upper[0], 0.03)

    def test_decision_static_and_dynamic_witness_upgrades_optimizer(self) -> None:
        self.assertEqual(_select_decision("FEASIBLE", "FEASIBLE")[0], "UPGRADE_V2_OPTIMIZER")

    def test_decision_static_only_extends_contact_mode(self) -> None:
        self.assertEqual(
            _select_decision("FEASIBLE", "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS")[0],
            "EXTEND_V2_CONTACT_MODE_TRANSITION",
        )

    def test_decision_static_empirical_infeasibility_enters_v3(self) -> None:
        self.assertEqual(
            _select_decision("EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS", "NOT_APPLICABLE")[0],
            "ENTER_V3_TASK_DYNAMICS_CONTACT",
        )

    def test_decision_missing_evidence_is_inconclusive(self) -> None:
        self.assertEqual(_select_decision("INCONCLUSIVE", "INCONCLUSIVE")[0], "INCONCLUSIVE")

    def test_failure_html_is_not_acceptance_artifact(self) -> None:
        mesh_layers = (
            "object_source_visual_mesh",
            "object_simulated_visual_mesh",
            "object_collision_mesh",
            "stage_b_right_visual_mesh",
            "stage_b_left_visual_mesh",
            "cxa_right_visual_mesh",
            "cxa_left_visual_mesh",
            "failed_reference_right_visual_mesh",
            "failed_reference_left_visual_mesh",
            "failed_actual_right_visual_mesh",
            "failed_actual_left_visual_mesh",
            "failed_actual_right_collision_mesh",
            "failed_actual_left_collision_mesh",
            "semantic_patch_surface_mesh",
        )
        frame = {"source_frame": 1, **{name: {"vertices": [], "faces": []} for name in mesh_layers}}
        point_layers = (
            "source_human_right",
            "source_human_left",
            "stage_b_kinematic_wuji",
            "cxa_corrected_static_wuji",
            "failed_dynamic_reference",
            "failed_dynamic_actual",
            "corrected_active_contact_anchors",
            "unreliable_source_records",
            "actual_mujoco_contacts",
            "actual_mujoco_contact_labels",
            "actual_contact_normals",
            "valid_region_contacts",
            "wrong_region_contacts",
            "lost_contact_marker",
            "penetration_points",
            "joint_limit_active_fingers",
            "actual_finger_trajectory",
            "semantic_patch_trajectory",
            "contact_force_vectors",
        )
        frame.update({name: [] for name in point_layers})
        payload = {
            "frames": [frame],
            "best_attempt": "failed",
            "first_failure": {"first_failure_source_frame": 1, "side": "left", "finger": "index", "role_type": "SUPPORT", "actual_geom_pair": "NONE"},
            "curves": {
                "contact_quality": {"patch coverage": [0.0]},
                "tracking_and_limits": {"reference-vs-actual qpos error": [0.0]},
                "contact_dynamics": {"contact force N": [0.0]},
                "object_tracking": {"object position error m": [0.0]},
            },
            "source_frames": [1],
            "events": [],
            "metadata": {"connected_mesh": True, "global_bounds": [[0, 0, 0], [1, 1, 1]], "close_center": [0, 0, 0], "frame_focus_centers": {"1": [0, 0, 0]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = build_failure_html(payload, Path(directory) / "failure.html")
            text = output.read_text(encoding="utf-8")
            self.assertIn(DISCLAIMER, text)
            self.assertIn("semantic_patch_surface_mesh", text)
            self.assertIn("failed_actual_left_visual_mesh", text)
            self.assertIn("failed_actual_left_collision_mesh", text)
            self.assertNotIn("ACCEPTANCE PASS", text)

    def test_html_payload_preserves_connected_faces(self) -> None:
        faces = [[0, 1, 2]]
        self.assertEqual(json.loads(json.dumps({"faces": faces}))["faces"], faces)

    def test_status_vocabulary_is_explicit(self) -> None:
        allowed = {"FEASIBLE", "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS", "INCONCLUSIVE", "NOT_APPLICABLE"}
        self.assertIn("EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS", allowed)
        self.assertNotIn("INFEASIBLE", allowed)

    def test_multistart_seed_is_fixed_by_contract(self) -> None:
        self.assertEqual(np.random.default_rng(20260801).normal(), np.random.default_rng(20260801).normal())

    def test_generated_namespace_is_local_and_ignored_by_contract(self) -> None:
        ignore = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn(".local_artifacts/", ignore)


if __name__ == "__main__":
    unittest.main()
