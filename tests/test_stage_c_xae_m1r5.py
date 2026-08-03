"""Unit contracts for Stage C-XAE-M1R5 source-surface authority."""

from __future__ import annotations

import unittest

import numpy as np

from spider.tools import grab_stage_c_xae_m1r5 as m1r5


class StageCXaeM1R5Test(unittest.TestCase):
    def test_source_range_and_named_hand_regions_are_frozen(self) -> None:
        self.assertEqual((m1r5.SOURCE_START, m1r5.SOURCE_END, m1r5.FPS), (1460, 1876, 120.0))
        self.assertEqual(m1r5.LEFT_JOINTS["index"], (25, 26, 27))
        self.assertEqual(m1r5.RIGHT_JOINTS["index"], (40, 41, 42))
        self.assertIn("wrist", m1r5.LEFT_JOINTS)

    def test_contact_interval_extraction_uses_source_frame_identity(self) -> None:
        self.assertEqual(m1r5._intervals(np.array([False, True, True, False, True])), [[1461, 1463], [1464, 1465]])

    def test_object_local_supports_surface_batches_and_centroids(self) -> None:
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[:, :3, 3] = ((1, 2, 3), (4, 5, 6))
        centroid = np.array(((2, 3, 4), (6, 7, 8)), dtype=float)
        np.testing.assert_allclose(m1r5._object_local(centroid, transforms), ((1, 1, 1), (2, 2, 2)))
        points = centroid[:, None, :]
        np.testing.assert_allclose(m1r5._object_local(points, transforms)[:, 0], ((1, 1, 1), (2, 2, 2)))

    def test_responsibility_does_not_force_single_finger(self) -> None:
        self.assertEqual(m1r5._responsibility_label({"left_index": .8}), "LEFT_INDEX_DOMINANT")
        self.assertEqual(m1r5._responsibility_label({"left_index": .6, "left_palm": .5}), "LEFT_INDEX_SHARED")
        self.assertEqual(m1r5._responsibility_label({"left_index": .2, "left_palm": .8}), "PALM_DOMINANT")
        self.assertEqual(m1r5._responsibility_label({}), "NO_RELIABLE_CONTACT")

    def test_subset_mesh_preserves_source_triangle_ids_without_dilation(self) -> None:
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
        faces = np.array(((0, 1, 2), (0, 1, 3)), dtype=int)
        patch = m1r5._subset_mesh(vertices, faces, [1])
        self.assertEqual(len(patch["faces"]), 1)
        self.assertEqual(len(patch["vertices"]), 3)

    def test_root_cause_prioritizes_invalid_source_premise(self) -> None:
        authority = {"decision": "CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID", "left_index_premise": "SHARED"}
        static = {"status": "ALL_FRAMES_STATIC_FEASIBLE"}
        continuous = {"status": "CONTINUOUS_PATH_FEASIBLE"}
        action = {"status": "ACTION_CONTRACT_PASS"}
        result = m1r5._root_cause(authority, static, continuous, action, {"status": "FAIL", "classification": "DYNAMIC_REALIZATION_FAILURE"})
        self.assertEqual(result["primary_root_cause"], "CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID")
        self.assertTrue(result["no_false_embodiment_claim"])

    def test_root_cause_distinguishes_static_continuous_action_and_dynamic(self) -> None:
        authority = {"decision": "SOURCE_ROLE_AND_PATCH_PASS", "left_index_premise": "VALID"}
        self.assertEqual(m1r5._root_cause(authority, {"status": "SOME_FRAMES_STATIC_INFEASIBLE"}, {"status": "x"}, {"status": "x"}, {"status": "NOT_RUN"})["primary_root_cause"], "FINGER_ONLY_STATIC_CONTACT_INFEASIBLE")
        self.assertEqual(m1r5._root_cause(authority, {"status": "ALL_FRAMES_STATIC_FEASIBLE"}, {"status": "STATIC_FEASIBLE_BUT_CONTINUOUS_PATH_INFEASIBLE"}, {"status": "x"}, {"status": "NOT_RUN"})["primary_root_cause"], "FINGER_ONLY_CONTINUOUS_CONTACT_PATH_INFEASIBLE")
        self.assertEqual(m1r5._root_cause(authority, {"status": "ALL_FRAMES_STATIC_FEASIBLE"}, {"status": "CONTINUOUS_PATH_FEASIBLE"}, {"status": "ACTION_INDEX_MAPPING_ERROR"}, {"status": "NOT_RUN"})["primary_root_cause"], "ACTION_CONTRACT_IMPLEMENTATION_ERROR")

    def test_htmls_are_real_3d_chinese_layered_viewers(self) -> None:
        source = m1r5._source_html({"frames": [], "patch_coverage": 0.0, "premise": "INVALID"})
        wuji = m1r5._wuji_html({"modes": []})
        for text in ("完整 source SMPL-X body", "左右手 surface mesh", "旧 role/旧 semantic patch", "物体坐标接触近景"):
            self.assertIn(text, source)
        for text in ("Wuji visual mesh", "assigned left-index collision region", "actual MuJoCo contacts", "反事实诊断"):
            self.assertIn(text, wuji)

    def test_static_and_dynamic_boundaries_are_explicit(self) -> None:
        self.assertEqual(m1r5.PATCH_GATE_M, .020)
        self.assertEqual(m1r5.PENETRATION_GATE_M, .003)
        self.assertEqual(m1r5.M1_WINDOW, (1461, 1462, 1463, 1464, 1465, 1466))


if __name__ == "__main__":
    unittest.main()
