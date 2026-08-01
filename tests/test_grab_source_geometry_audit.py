"""Unit contracts for the frozen GRAB source-geometry audit primitives."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from spider.tools.grab_source_geometry_audit import (
    cxa_gate,
    compose_transform,
    detect_frame_offset,
    detect_global_frame_only_difference,
    detect_left_hand_mirror,
    detect_quaternion_order,
    detect_root_translation_double_apply,
    detect_rotation_transpose,
    detect_unit_scale,
    invert_transform,
    object_relative,
    rotation_residual_rad,
    summarize_signed_surface,
    stage_b_gate,
    transform_from_rt,
)
from spider.tools.grab_source_geometry_audit_viewer import _document


class SourceGeometryTransformTest(unittest.TestCase):
    def test_transform_inversion_and_object_relative_transform(self) -> None:
        object_world = transform_from_rt(Rotation.from_euler("z", 0.5).as_matrix(), np.array([1.0, -2.0, 0.2]))
        hand_world = transform_from_rt(Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix(), np.array([1.3, -1.5, 0.8]))
        identity = compose_transform(object_world, invert_transform(object_world))
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-12)
        relative = object_relative(object_world, hand_world)
        np.testing.assert_allclose(compose_transform(object_world, relative), hand_world, atol=1e-12)

    def test_global_frame_only_difference_preserves_object_relative_hand_pose(self) -> None:
        source_object = np.stack([transform_from_rt(np.eye(3), [0.1 * frame, 0.0, 0.5]) for frame in range(3)])
        source_hands = np.stack([
            np.stack((transform_from_rt(np.eye(3), [0.1 * frame, -0.2, 0.7]), transform_from_rt(np.eye(3), [0.1 * frame, 0.2, 0.7])))
            for frame in range(3)
        ])
        global_transform = transform_from_rt(Rotation.from_euler("x", 0.4).as_matrix(), [0.8, -0.4, 0.2])
        result = detect_global_frame_only_difference(source_object, source_hands, compose_transform(global_transform, source_object), compose_transform(global_transform, source_hands))
        self.assertEqual(result["classification"], "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY")

    def test_rotation_transpose_and_quaternion_order_detection(self) -> None:
        rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
        self.assertTrue(detect_rotation_transpose(rotation[None], rotation.T[None]))
        wxyz = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        self.assertEqual(detect_quaternion_order(wxyz[None], rotation[None]), "WXYZ")
        self.assertEqual(detect_quaternion_order(wxyz[[1, 2, 3, 0]][None], rotation[None]), "QUATERNION_ORDER_ERROR")

    def test_unit_scale_root_double_apply_and_left_mirror(self) -> None:
        reference = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(detect_unit_scale(reference, reference * 1000.0)["classification"], "UNIT_SCALE_ERROR")
        root = np.array([0.1, -0.2, 0.3])
        self.assertTrue(detect_root_translation_double_apply(reference, reference + root, root))
        self.assertTrue(detect_left_hand_mirror(reference, reference * np.array([-1.0, 1.0, 1.0])))

    def test_frame_offset_and_rotation_precision(self) -> None:
        source = np.arange(10.0)[:, None]
        candidate = np.concatenate((np.array([[-1.0]]), source[:-1]))
        self.assertEqual(detect_frame_offset(source, candidate)["best_offset"], -1)
        almost_identity = np.eye(3) + np.array([[0.0, -1e-8, 0.0], [1e-8, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertLess(float(rotation_residual_rad(np.eye(3), almost_identity)), 1e-6)

    def test_signed_surface_summary_keeps_explicit_inside_outside_convention(self) -> None:
        # trimesh's convention is positive inside a watertight mesh.  The
        # audit must never silently call a negative value "penetration".
        summary = summarize_signed_surface(
            np.asarray(((-0.03, 0.01), (-0.04, 0.02))),
            np.asarray((1460, 1461)),
        )
        self.assertAlmostEqual(summary["max_penetration_m"], 0.02)
        self.assertEqual(summary["max_penetration_source_frame"], 1461)
        self.assertAlmostEqual(summary["max_suspension_gap_m"], 0.04)
        self.assertEqual(summary["max_suspension_source_frame"], 1461)


class SourceGeometryGateAndViewerTest(unittest.TestCase):
    def test_stage_b_and_cxa_gates_fail_closed(self) -> None:
        self.assertEqual(stage_b_gate("FAIL"), "NOT_RUN_DUE_TO_RAW_LOADER_FAILURE")
        self.assertEqual(stage_b_gate("PASS"), "RUN")
        self.assertEqual(cxa_gate("PASS", "FAIL"), "NOT_RUN_DUE_TO_STAGE_B_FAILURE")
        self.assertEqual(cxa_gate("PASS", "PASS"), "RUN")

    def test_chinese_viewer_has_real_mesh_and_coordinate_mode_controls(self) -> None:
        html = _document({"source": {"frames": []}, "raw": {}, "object": {}, "stageb": {"frames": []}, "cxa": {}, "robotStageB": {}, "robotCXA": {}, "contact": {}, "patches": [], "metrics": [], "curves": {}, "final": {}})
        self.assertIn("mesh3d", html)
        self.assertIn("物体坐标", html)
        self.assertIn("左腕坐标", html)
        self.assertIn("Stage B vs C-XA", html)
        self.assertIn("semantic patch", html)


if __name__ == "__main__":
    unittest.main()
