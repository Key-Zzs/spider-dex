"""Pure contract tests for the fail-closed corrected V2 dynamic gates."""

from __future__ import annotations

import unittest

import numpy as np

from spider.tools.grab_stage_c_v2_dynamic import _has_consecutive, _has_exponential_growth, _select_keyframes


class DynamicAcceptanceContractTest(unittest.TestCase):
    def test_growth_detector_requires_consecutive_unbounded_growth(self) -> None:
        self.assertTrue(_has_exponential_growth(np.array([1.0, 11.0, 121.0]), 10.0, 3))
        self.assertFalse(_has_exponential_growth(np.array([1.0, 11.0, 10.0, 200.0]), 10.0, 3))

    def test_persistent_collision_detector_rejects_a_multistep_run(self) -> None:
        self.assertTrue(_has_consecutive(np.array([False, True, True, True, False]), 3))
        self.assertFalse(_has_consecutive(np.array([True, True, False, True, True]), 3))

    def test_keyframe_selection_preserves_real_events_and_distinct_frames(self) -> None:
        qpos = np.zeros((20, 64))
        expected = np.zeros((20, 10), dtype=bool)
        expected[5, 0] = True
        expected[12, :3] = True
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".npz") as stream:
            np.savez(stream.name, collision_before_m=np.arange(20))
            selected = _select_keyframes(qpos, expected, stream.name)  # type: ignore[arg-type]
        self.assertGreaterEqual(len(set(selected.values())), 5)
        self.assertEqual(selected["first_corrected_high_confidence_contact"], 5)
        self.assertEqual(selected["peak_stage_b_kinematic_penetration"], 19)


if __name__ == "__main__":
    unittest.main()
