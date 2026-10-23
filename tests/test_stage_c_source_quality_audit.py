"""Contract tests for the Stage C source-quality authority audit."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from spider.tools import grab_stage_c_source_quality_audit as sqa


class StageCSourceQualityAuditTest(unittest.TestCase):
    def test_static_exec_contract_is_frozen(self) -> None:
        self.assertEqual(sqa.STATIC_EXEC_SIGN, "xyz_majority")
        self.assertEqual(sqa.QUERY_SAMPLES, 20_000)
        self.assertEqual(sqa.QUERY_VOXEL_M, 0.001)
        self.assertEqual(len(sqa.WRIST_LOOP), 16)
        self.assertTrue(sqa._implementation_map({})["fixed_contract"]["final_no_sampling_after_cache"])

    def test_fixed_wrist_cap_is_closed_and_no_repair_is_called(self) -> None:
        vertices = np.zeros((1, 778, 3), dtype=np.float32)
        vertices[0, :, 2] = np.linspace(0.0, 0.2, 778)
        faces = np.zeros((1538, 3), dtype=np.int32)
        # The real topology is checked by the smoke/full audit; this test only
        # asserts that the fixed cap adds exactly one vertex and 16 triangles.
        capped_vertices, capped_faces = sqa._fixed_wrist_cap(vertices, faces, "left")
        self.assertEqual(capped_vertices.shape, (1, 779, 3))
        self.assertEqual(capped_faces.shape, (1554, 3))
        self.assertEqual(capped_faces[-1].tolist(), [778, sqa.WRIST_LOOP[0], sqa.WRIST_LOOP[-1]])
        source = inspect.getsource(sqa._fixed_wrist_cap)
        self.assertNotIn("repair", source.lower())

    def test_contact_and_penetration_are_separate(self) -> None:
        signed = np.asarray([0.005, 0.0005, -0.002, -0.006], dtype=float)
        metrics = sqa._row_metrics(signed, np.ones(4) / 4.0, formal=True)
        self.assertEqual(metrics["categories"]["near_surface_contact"], 1)
        self.assertEqual(metrics["categories"]["shallow_penetration"], 1)
        self.assertEqual(metrics["categories"]["deep_penetration"], 1)
        self.assertEqual(metrics["penetrating_point_count"], 2)

    def test_non_watertight_stage_cannot_be_formal_authority(self) -> None:
        valid, status = sqa._classify_stage_mesh("C_STAGE_B", {"left": {"watertight": False, "winding_consistent": True}}, True)
        self.assertFalse(valid)
        self.assertEqual(status, "SDF_INVALID_FOR_STAGE")

    def test_frame_mapping_keeps_source_endpoints_and_cropped_stage(self) -> None:
        mapping = np.arange(1461, 1875, dtype=np.int64)
        rows = sqa._frame_map(mapping, mapping)
        self.assertEqual(len(rows), 416)
        self.assertIsNone(rows[0]["stage_b_local_index"])
        self.assertEqual(rows[1]["stage_b_local_index"], 0)
        self.assertIsNone(rows[-1]["cxa_local_index"] if "cxa_local_index" in rows[-1] else rows[-1]["repaired_cxa_local_index"])


if __name__ == "__main__":
    unittest.main()
