"""Fail-closed and bounded-state tests for Stage C-V2R2E recovery."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from spider.tools.grab_stage_c_v2r2e import (
    DYNAMIC_CANDIDATES,
    STATE_ORDER,
    _failure_taxonomy,
    _hash_payload,
)
from spider.tools.grab_stage_c_v2r2e_downstream import _not_run
from spider.tools.grab_stage_c_v2r2e_viewer import build_html


class V2R2EDecisionTest(unittest.TestCase):
    def test_primary_precedes_smokes_and_downstream(self) -> None:
        self.assertLess(STATE_ORDER.index("D2_TEST"), STATE_ORDER.index("SMOKE_1"))
        self.assertLess(STATE_ORDER.index("SMOKE_2"), STATE_ORDER.index("HTML"))
        self.assertLess(STATE_ORDER.index("HTML"), STATE_ORDER.index("SCREENSHOT_REVIEW"))

    def test_dynamic_search_is_bounded_and_unique(self) -> None:
        self.assertLessEqual(len(DYNAMIC_CANDIDATES), 12)
        ids = [row["candidate_id"] for row in DYNAMIC_CANDIDATES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("contact_ik_gain", DYNAMIC_CANDIDATES[-1])
        self.assertLess(STATE_ORDER.index("CONTACT_DYNAMICS_REPAIR"), STATE_ORDER.index("OBJECT_GUIDANCE_REPAIR"))

    def test_failure_taxonomy_prioritizes_numeric_and_safety_failures(self) -> None:
        self.assertEqual(_failure_taxonomy({"gates": {"finite": False}}), "NUMERICAL")
        self.assertEqual(_failure_taxonomy({"gates": {"finite": True, "joint_limits": False}}), "JOINT_LIMIT")
        self.assertEqual(_failure_taxonomy({"gates": {"finite": True, "joint_limits": True, "smoothness": False}}), "SMOOTHNESS")

    def test_profile_hash_is_order_independent(self) -> None:
        self.assertEqual(_hash_payload({"a": 1, "b": 2}), _hash_payload({"b": 2, "a": 1}))

    def test_guidance_and_timing_bounds_are_frozen(self) -> None:
        with open("configs/project/grab_wuji_stage_c_v2r2e.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        search = config["search"]
        self.assertLessEqual(len(search["object_guidance_candidates"]), search["max_object_guidance_candidates"])
        self.assertEqual(search["timing_variants"], [1.0, 1.25, 1.5, 2.0])
        self.assertEqual(config["timing"]["variant"], "V2_ORIGINAL_TIMING")

    def test_contact_region_mapping_is_explicit_and_complete(self) -> None:
        with open("configs/project/wuji_hand2_contact_regions.yaml", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
        seen: set[str] = set()
        for side, fingers in mapping["regions"].items():
            self.assertEqual(set(fingers), {"thumb", "index", "middle", "ring", "pinky"})
            for region in fingers.values():
                self.assertIn(region["distal_geom"], region["finger_geoms"])
                self.assertNotIn(region["distal_geom"], seen)
                seen.add(region["distal_geom"])

    def test_downstream_is_fail_closed_without_d2_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _not_run("D2 report is absent", root)
            self.assertEqual(payload["status"], "NOT_RUN")
            self.assertFalse((root / "html").exists())
            self.assertTrue((root / "reports/downstream_gate_report.json").is_file())
            self.assertEqual(payload["smokes"]["s1__mug_lift"]["status"], "NOT_RUN")

    def test_viewer_requires_primary_and_both_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_html(root)
            self.assertEqual(payload["status"], "NOT_RUN")
            self.assertFalse((root / "html/index.html").exists())

    def test_downstream_and_html_fail_closed_without_d2(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            downstream = _not_run("D2 report is absent", root)
            self.assertEqual(downstream["status"], "NOT_RUN")
            html = build_html(root)
            self.assertEqual(html["status"], "NOT_RUN")
            self.assertFalse((root / "html/index.html").exists())


if __name__ == "__main__":
    unittest.main()
