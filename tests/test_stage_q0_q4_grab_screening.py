"""Focused contract coverage for the isolated Q0--Q4 GRAB screen."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider.tools import grab_stage_q0_q4_screening as q


class StageQ0Q4GrabScreeningTest(unittest.TestCase):
    def test_q0_registry_rejects_old_primary_and_preserves_history_boundary(self) -> None:
        registry = q.yaml.safe_load(q.REGISTRY.read_text(encoding="utf-8"))
        old = next(row for row in registry["sequences"] if row["sequence_id"] == q.EXCLUDED)
        self.assertEqual(old["quality_status"], "SOURCE_QUALITY_REJECTED")
        self.assertIn("primary_acceptance", old["forbidden_uses"])
        self.assertEqual(registry["recommended_primary_sequence"], "s7/stanfordbunny_pass_1")
        self.assertEqual(registry["recommended_backup_sequence"], "s10/spherelarge_pass_1")

    def test_frozen_selection_is_exactly_ten_deterministic_and_diverse(self) -> None:
        rows = []
        for index in range(16):
            rows.append({"source_id": f"s{index // 2}/obj{index // 2}_{index}", "subject": f"s{index // 2}", "object": f"obj{index // 3}", "status": "METADATA_ELIGIBLE"})
        first = q.freeze_candidates(rows)
        second = q.freeze_candidates(rows)
        self.assertEqual(len(first), 10)
        self.assertEqual([row["source_id"] for row in first], [row["source_id"] for row in second])
        self.assertLessEqual(max(sum(row["subject"] == subject for row in first) for subject in {row["subject"] for row in first}), 2)
        self.assertLessEqual(max(sum(row["object"] == obj for row in first) for obj in {row["object"] for row in first}), 2)

    def test_reverse_sdf_contract_and_category_separation_are_explicit(self) -> None:
        self.assertEqual((q.QUERY_SAMPLES, q.QUERY_VOXEL_M, q.NEAR_M, q.SHALLOW_M), (20_000, .001, .001, .003))
        row = q._sdf_row(np.asarray([.02, .0005, -.002, -.006]), 7)
        self.assertEqual(row["categories"]["NEAR_SURFACE_CONTACT"], 1)
        self.assertEqual(row["categories"]["SHALLOW_PENETRATION"], 1)
        self.assertEqual(row["categories"]["DEEP_PENETRATION"], 1)
        source = inspect.getsource(q._static_query)
        self.assertIn("query_targets_with_meshsdf", source)
        self.assertIn("xyz_majority", source)

    def test_deep_penetration_is_not_authority_or_patch_evidence(self) -> None:
        source = inspect.getsource(q.build_authority)
        self.assertIn("np.abs(signed_all[side]) <= NEAR_M", source)
        self.assertIn("deep_penetration_excluded", source)
        self.assertIn("no dilation", source)

    def test_isolated_output_and_stage_boundary_contract(self) -> None:
        source = inspect.getsource(q.run) + inspect.getsource(q._write_index)
        self.assertIn("FileExistsError", source)
        stage = inspect.getsource(q._run_stage_ab)
        self.assertIn("grab_pipeline", stage)
        self.assertIn("run-wuji-ik", stage)
        self.assertIn("SDF_INVALID_FOR_STAGE", stage)

    def test_six_html_names_are_fixed_and_chinese_labels_are_present(self) -> None:
        source = inspect.getsource(q.run) + inspect.getsource(q._write_index)
        for name in ("01_ten_candidate_source_quality.html", "02_primary_source_contact_authority.html", "03_backup_source_contact_authority.html", "04_primary_source_vs_stage_b.html", "05_backup_source_vs_stage_b.html", "index.html"):
            self.assertIn(name, source)
        page = q._html("测试", {"status": "PASS", "frames": [], "body_faces": [], "left_faces": [], "right_faces": [], "object": {"vertices": [], "faces": []}, "patch": {"vertices": [], "faces": []}, "schedule": {}, "categories": {}}, "authority")
        for label in ("完整 source SMPL-X body", "左手 surface", "右手 surface", "semantic patch connected surface", "世界坐标"):
            self.assertIn(label, page)

    def test_q1_visual_page_replays_only_frozen_candidate_queries(self) -> None:
        source = inspect.getsource(q._candidate_visual_payload) + inspect.getsource(q.refresh_q1_visuals)
        self.assertIn("MeshSDFfromAsset", source)
        self.assertIn("Q1_VISUALIZATION_MANIFEST.json", source)
        self.assertIn("visualization-only repeat", source)
        self.assertIn("exactly the original ten", source)


if __name__ == "__main__":
    unittest.main()
