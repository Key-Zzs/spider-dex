"""Unit tests for pure Stage C collision-audit accounting."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from spider.geometry.collision_audit import bbox_report, contact_pair_summary, summarize_signed_distances


class CollisionAuditTest(unittest.TestCase):
    def test_bbox_scale_and_center_gates(self) -> None:
        visual = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        collision = visual.copy()
        report = bbox_report(visual, collision)
        self.assertTrue(report["within_extent_gate"])
        self.assertTrue(report["within_center_gate"])
        collision.apply_scale(1.2)
        self.assertFalse(bbox_report(visual, collision)["within_extent_gate"])

    def test_signed_summary_keeps_nonpenetrating_samples_separate(self) -> None:
        summary = summarize_signed_distances(
            np.asarray([0.01, -0.003, -0.001]),
            np.zeros((3, 3)), np.tile([0.0, 0.0, 1.0], (3, 1)),
        )
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["penetrating_count"], 2)
        self.assertAlmostEqual(summary["max_penetration_m"], 0.003)

    def test_contact_pairs_do_not_merge_geometries(self) -> None:
        records = [
            {"geom_pair": "a|o", "frame_index": 1, "penetration_m": 0.001, "position_world": [0, 0, 0], "normal_world": [0, 0, 1]},
            {"geom_pair": "b|o", "frame_index": 1, "penetration_m": 0.004, "position_world": [1, 0, 0], "normal_world": [0, 1, 0]},
        ]
        summary = contact_pair_summary(records)
        self.assertEqual(set(summary), {"a|o", "b|o"})
        self.assertAlmostEqual(summary["b|o"]["max_penetration_m"], 0.004)

    def test_serial_xyz_coordinates_round_trip_a_quaternion_orientation(self) -> None:
        source = Rotation.from_rotvec([0.2, -0.9, 2.7])
        serial_xyz = source.as_euler("XYZ")
        self.assertLess(Rotation.from_matrix(source.as_matrix().T @ Rotation.from_euler("XYZ", serial_xyz).as_matrix()).magnitude(), 1e-12)


if __name__ == "__main__":
    unittest.main()
