"""Pure unit contracts for Stage C-XAE surface-aligned refinement."""

from __future__ import annotations

import unittest

import numpy as np

from spider.tools.grab_stage_c_xae import closest_point_triangle, independent_patch_distance


class StageCXaeGeometryTest(unittest.TestCase):
    def test_independent_triangle_distance_handles_face_edge_and_vertex_regions(self) -> None:
        triangle = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        np.testing.assert_allclose(closest_point_triangle(np.asarray((0.2, 0.3, 1.0)), triangle), (0.2, 0.3, 0.0))
        np.testing.assert_allclose(closest_point_triangle(np.asarray((0.7, 0.7, 0.0)), triangle), (0.5, 0.5, 0.0))
        np.testing.assert_allclose(closest_point_triangle(np.asarray((-1.0, -1.0, 0.0)), triangle), (0.0, 0.0, 0.0))

    def test_independent_patch_distance_selects_nearest_triangle(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0))
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)))
        distance, face, point = independent_patch_distance(np.asarray((0.2, 0.2, 1.8)), vertices, faces)
        self.assertEqual(face, 1)
        self.assertAlmostEqual(distance, 0.2)
        np.testing.assert_allclose(point, (0.2, 0.2, 2.0))


if __name__ == "__main__":
    unittest.main()
