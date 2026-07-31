"""Contract-only tests for Stage C-X task-equivalent contact reassignment."""

import unittest

import numpy as np

from spider.contact.embodiment_assignment import (
    RobotRegion, allowed_region, assignment_cost, classify_role, default_robot_regions,
    select_minimum_successful_level, thin_wall_compatible, viterbi_assignment,
)


class EmbodimentAssignmentTest(unittest.TestCase):
    def test_source_role_classification_is_source_only(self):
        self.assertEqual(classify_role("thumb", 20, ["thumb", "index"])[0], "THUMB_OPPOSITION")
        self.assertEqual(classify_role("index", 20, ["thumb", "index"])[0], "PRIMARY_GRASP")
        self.assertEqual(classify_role("ring", 2, ["ring"])[0], "TRANSIENT")

    def test_same_side_thumb_and_neighbors_are_enforced(self):
        regions = {item.region_id: item for item in default_robot_regions()}
        self.assertTrue(allowed_region(regions["right_thumb_fingertip"], "right", "thumb", "THUMB_OPPOSITION", 4))
        self.assertFalse(allowed_region(regions["right_index_fingertip"], "right", "thumb", "THUMB_OPPOSITION", 4))
        self.assertFalse(allowed_region(regions["left_index_fingertip"], "right", "index", "PRIMARY_GRASP", 4))
        self.assertTrue(allowed_region(regions["right_middle_fingertip"], "right", "index", "PRIMARY_GRASP", 2))
        self.assertFalse(allowed_region(regions["right_pinky_fingertip"], "right", "index", "PRIMARY_GRASP", 2))

    def test_palm_is_support_only_and_never_exact_fingertip(self):
        palm = RobotRegion("right_palm_support", "right", None, "palm", ("SUPPORT", "STABILIZATION"))
        self.assertFalse(allowed_region(palm, "right", "index", "PRIMARY_GRASP", 4))
        self.assertFalse(allowed_region(palm, "right", "index", "SUPPORT", 2))
        self.assertTrue(allowed_region(palm, "right", "index", "SUPPORT", 3))

    def test_thin_wall_rejects_opposite_or_disconnected_surface(self):
        self.assertTrue(thin_wall_compatible(4, 4, np.array([0, 0, 1]), np.array([0, 0, 1])))
        self.assertFalse(thin_wall_compatible(4, 5, np.array([0, 0, 1]), np.array([0, 0, 1])))
        self.assertFalse(thin_wall_compatible(4, 4, np.array([0, 0, 1]), np.array([0, 0, -1])))

    def test_assignment_cost_requires_metric_separation(self):
        keys = ("surface_patch", "normal", "functional_role", "identity_change", "reachability", "collision_risk", "tracking_deviation", "temporal_switch")
        terms = {key: 1.0 for key in keys}; weights = {key: 2.0 for key in keys}
        self.assertEqual(assignment_cost(terms, weights), 16.0)
        with self.assertRaises(ValueError):
            assignment_cost({"surface_patch": 1.0}, weights)

    def test_temporal_assignment_is_switch_bounded(self):
        frame_costs = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        path, _cost, switches = viterbi_assignment(frame_costs, switch_cost=10.0, max_switches=1)
        self.assertLessEqual(switches, 1)
        self.assertEqual(len(path), 4)

    def test_minimum_relaxation_level_selection_fails_closed(self):
        self.assertEqual(select_minimum_successful_level({1: "FAIL", 2: "PASS", 3: "NOT_RUN", 4: "NOT_RUN"}), 2)
        self.assertIsNone(select_minimum_successful_level({1: "NOT_RUN"}))
        with self.assertRaises(ValueError):
            select_minimum_successful_level({1: "NOT_RUN", 2: "PASS"})
