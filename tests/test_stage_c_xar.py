"""Focused unit gates for the C-XAR correction-leakage repair."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from spider.tools.grab_source_geometry_audit import rotation_residual_rad
from spider.tools.grab_stage_c import _assert_locked_dofs, _cxa_variable_indices, _joint_bounds


class StageCXarTest(unittest.TestCase):
    def test_default_variable_mask_excludes_every_root_wrist_and_object_coordinate(self) -> None:
        mutable, locked = _cxa_variable_indices(False)
        self.assertEqual(mutable.tolist(), list(range(6, 26)) + list(range(32, 52)))
        self.assertEqual(locked.tolist(), [0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31])
        self.assertFalse(set(mutable).intersection(locked))

    def test_explicit_wrist_opt_in_has_no_implicit_object_variable(self) -> None:
        mutable, locked = _cxa_variable_indices(True)
        self.assertEqual(mutable.tolist(), list(range(52)))
        self.assertEqual(locked.tolist(), [])

    def test_finger_only_bounds_lock_both_wrist_translation_and_rotation(self) -> None:
        baseline = np.linspace(-0.2, 0.2, 52)
        model = SimpleNamespace(jnt_range=np.tile(np.array((-2.0, 2.0)), (52, 1)))
        profile = {"wrist_translation_bound_m": 0.015, "wrist_rotation_bound_rad": 0.2}
        bounds = _joint_bounds(model, baseline, profile, finger_only=True)
        for index in (0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31):
            self.assertEqual(bounds[index], (baseline[index], baseline[index]))
        self.assertEqual(bounds[6], (-2.0, 2.0))
        self.assertEqual(bounds[32], (-2.0, 2.0))

    def test_locked_invariant_rejects_solver_leakage(self) -> None:
        base = np.zeros(52)
        _assert_locked_dofs(base.copy(), base, np.array((0, 1, 2)), "unit-test")
        leaked = base.copy(); leaked[1] = 1e-5
        with self.assertRaisesRegex(RuntimeError, "locked-DOF invariant"):
            _assert_locked_dofs(leaked, base, np.array((0, 1, 2)), "unit-test")

    def test_wxyz_and_intrinsic_xyz_roundtrips_are_physical_rotation_identity(self) -> None:
        samples = (np.zeros(3), np.deg2rad((1.0, -1.0, 1.0)), np.array((0.01, -0.02, 0.03)), np.array((0.0, np.pi / 2 - 1e-6, 0.0)))
        for rotvec in samples:
            physical = Rotation.from_rotvec(rotvec).as_matrix()
            quat = Rotation.from_matrix(physical).as_quat()
            wxyz = quat[[3, 0, 1, 2]]
            from_wxyz = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
            euler = Rotation.from_matrix(physical).as_euler("XYZ")
            from_euler = Rotation.from_euler("XYZ", euler).as_matrix()
            self.assertLessEqual(float(rotation_residual_rad(physical, from_wxyz)), 1e-10)
            self.assertLessEqual(float(rotation_residual_rad(physical, from_euler)), 1e-10)

    def test_finger_only_correction_cannot_change_a_locked_wrist_coordinate(self) -> None:
        mutable, locked = _cxa_variable_indices(False)
        baseline = np.zeros((3, 52))
        correction = np.zeros_like(baseline)
        correction[:, mutable] = 0.1
        candidate = baseline + correction
        self.assertTrue(np.array_equal(candidate[:, locked], baseline[:, locked]))
        _assert_locked_dofs(candidate[0], baseline[0], locked, "finger-only unit-test")


if __name__ == "__main__":
    unittest.main()
