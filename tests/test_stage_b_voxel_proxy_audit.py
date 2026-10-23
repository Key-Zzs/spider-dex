"""Focused invariants for the analysis-only Stage-B voxel proxy audit."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import trimesh

from spider.tools import grab_stage_b_voxel_proxy_audit as audit


class StageBVoxelProxyAuditTest(unittest.TestCase):
    def test_voxel_proxy_is_closed_and_reports_bidirectional_error(self) -> None:
        raw = trimesh.creation.box(extents=(.04, .03, .02))
        proxy, status = audit._proxy_from_union(raw, .002)
        error = audit._surface_error(raw, proxy, .002, 3)
        self.assertTrue(status["proxy_watertight"])
        self.assertTrue(status["proxy_winding_consistent"])
        self.assertIn("raw_to_proxy_mean_m", error)
        self.assertIn("proxy_to_raw_mean_m", error)
        self.assertTrue(error["p95_within_two_voxels"])

    def test_negative_metrics_keep_the_signed_and_depth_conventions(self) -> None:
        signed = np.asarray([2.0, -1.0, -3.0, .5])
        weights = np.ones(4)
        metrics = audit._frame_metrics(signed, weights, np.zeros((4, 3)), np.ones(4, dtype=bool))
        self.assertEqual(metrics["negative_point_count"], 2)
        self.assertAlmostEqual(metrics["negative_signed_distance_mean_mm"], -2.0)
        self.assertAlmostEqual(metrics["negative_signed_distance_variance_mm2"], 1.0)
        self.assertAlmostEqual(metrics["max_penetration_depth_mm"], 3.0)
        self.assertAlmostEqual(metrics["negative_point_ratio"], .5)

    def test_contract_uses_actual_collision_union_then_one_proxy_sdf(self) -> None:
        source = inspect.getsource(audit._collision_specs) + inspect.getsource(audit._collision_union) + inspect.getsource(audit._proxy_from_union) + inspect.getsource(audit._run_sequence)
        self.assertIn("geom_group[geom]) != 2", source)
        self.assertIn("voxelized", source)
        self.assertIn("marching_cubes", source)
        self.assertIn("_proxy_targets(proxies)", source)
        self.assertEqual(audit.PROXY_LABEL, "ANALYSIS_ONLY_VOXEL_UNION_PROXY")

    def test_conclusion_vocabulary_keeps_contact_separation_distinct(self) -> None:
        source = inspect.getsource(audit._conclusion)
        self.assertIn("PENETRATION_REDUCED_BY_CONTACT_SEPARATION", source)
        self.assertIn("INCONCLUSIVE_PROXY_SENSITIVE", source)
        self.assertIn("loader_A_B_consistent", source)
        self.assertIn("ratio_delta_vs_spider", source)

    def test_merge_recomputes_worst_deterioration_from_per_frame_evidence(self) -> None:
        source = inspect.getsource(audit._paired_stage_b_vs_official) + inspect.getsource(audit.consolidate)
        self.assertIn("max_penetration_depth_deterioration_frame", source)
        self.assertIn("_paired_stage_b_vs_official", source)

    def test_summary_tables_expose_each_required_hand_scope(self) -> None:
        source = inspect.getsource(audit._comparison_table) + inspect.getsource(audit._write_summary_artifacts)
        self.assertIn('("left", "right", "both")', source)
        self.assertIn('"hand_scope"', source)
        self.assertIn("左手、右手与双手 union aggregate", source)

    def test_geometry_refresh_uses_union_boundary_not_internal_link_surfaces(self) -> None:
        source = inspect.getsource(audit.refresh_geometry_validation)
        self.assertIn("FINE_VOXEL_OCCUPANCY_UNION_BOUNDARY", source)
        self.assertIn("link-overlap interior", source)
        self.assertIn("qpos, object pose, query point", source)

    def test_html_object_transform_is_flat_browser_affine(self) -> None:
        source = inspect.getsource(audit._run_sequence) + inspect.getsource(audit.consolidate)
        self.assertIn('"object_T": np.asarray', source)
        self.assertIn('frame["object_T"] = np.asarray', source)
        self.assertIn("reshape(-1).tolist()", source)

    def test_c_viewer_keeps_complete_collision_and_proxy_triangles(self) -> None:
        source = inspect.getsource(audit._run_sequence) + inspect.getsource(audit.refresh_full_c_html_payload) + inspect.getsource(audit._html)
        self.assertIn("_full_mesh(raw[side].vertices, raw[side].faces)", source)
        self.assertIn("FULL_TRIANGLES_LAZY_LOAD", source)
        self.assertIn("full_c_mesh_script", source)
        self.assertIn("完整 mesh", source)


if __name__ == "__main__":
    unittest.main()
