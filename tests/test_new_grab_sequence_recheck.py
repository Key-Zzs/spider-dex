"""Focused regression coverage for the isolated new-GRAB recheck tool."""

from __future__ import annotations

import base64
import gzip
import inspect
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider.tools import new_grab_sequence_recheck as recheck


def _source_values(frames: int = 3) -> dict[str, np.ndarray]:
    identity = np.tile(np.eye(4), (frames, 1, 1))
    identity[:, 0, 3] = np.arange(frames, dtype=float) * 0.01
    body = np.zeros((frames, 4, 3), dtype=np.float32)
    hand = np.zeros((frames, 5, 3), dtype=np.float32)
    result: dict[str, np.ndarray] = {
        "source_frame_indices": np.arange(frames),
        "timestamps_s": np.arange(frames, dtype=float) / 120.0,
        "body_vertices_world": body,
        "body_faces": np.array([[0, 1, 2]], dtype=np.int32),
        "body_joints_world": body,
        "body_vertices_world_context": body.copy(),
        "body_faces_context": np.array([[0, 1, 2]], dtype=np.int32),
        "T_world_object": identity.copy(),
        "object_asset_vertices": np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=np.float32),
        "object_faces": np.array([[0, 1, 2]], dtype=np.int32),
    }
    for side in ("left", "right"):
        result[f"{side}_joints_world"] = np.zeros((frames, 21, 3), dtype=np.float32)
        result[f"{side}_vertices_world"] = hand.copy()
        result[f"{side}_hand_faces"] = np.array([[0, 1, 2]], dtype=np.int32)
        result[f"T_world_{side}_wrist"] = identity.copy()
    return result


def _viewer_payload(page: str) -> dict[str, object]:
    """Decode the deterministic gzip payload embedded in an audit HTML."""
    match = re.search(r"const P='([^']+)'", page)
    if match is None:
        raise AssertionError("missing packed viewer payload")
    raw = gzip.decompress(base64.b64decode(match.group(1)))
    (metadata_size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + metadata_size].decode("utf-8"))


class NewGrabSequenceRecheckTest(unittest.TestCase):
    def test_deterministic_score_and_excluded_sequence(self) -> None:
        self.assertEqual(recheck.safe_id("s2/bowl_pass_1"), "s2__bowl_pass_1")
        self.assertGreater(recheck.candidate_score("s2", "bowl", "pass", 600), recheck.candidate_score("s5", "cylindermedium", "inspect", 600))
        self.assertEqual(recheck.longest_true_run(np.array([False, True, True, False, True])), 2)
        self.assertEqual(recheck.EXCLUDED_SEQUENCE, "s5/cylindermedium_lift")

    def test_parameter_contract_rejects_incomplete_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, n_frames=np.asarray(2), gender=np.asarray("female"), body=np.asarray({"params": {"transl": np.zeros((2, 3))}}, dtype=object), object=np.asarray({"params": {"transl": np.zeros((2, 3))}}, dtype=object))
            ok, errors = recheck._candidate_parameter_contract(path)
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_independent_reconstruction_does_not_call_adapter(self) -> None:
        source = inspect.getsource(recheck.reconstruct_official)
        self.assertNotIn("GrabAdapter(", source)
        self.assertIn("smplx.create", source)
        self.assertIn("flat_hand_mean=False", source)

    def test_object_frame_comparison_and_global_only_detection(self) -> None:
        official = _source_values()
        spider = _source_values()
        comparison, temporal, global_view = recheck.compare_source(official, spider, 120.0)
        self.assertEqual(comparison["decision"]["status"], "PASS")
        self.assertEqual(temporal["object"]["best_offset"], 0)
        self.assertEqual(global_view["classification"], "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY")

    def test_source_html_has_real_mesh_layers_and_chinese_labels(self) -> None:
        official = _source_values()
        spider = _source_values()
        comparison, _, _ = recheck.compare_source(official, spider, 120.0)
        contact = {"sides": {side: {"max_suspension_proxy_frame": 0, "tip_distance_m": np.zeros((3, 5)), "nearest_point_world": np.zeros((3, 5, 3))} for side in ("left", "right")}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "html").mkdir()
            html, index = recheck.write_source_html(root, official, spider, comparison, contact)
            text = html.read_text(encoding="utf-8")
            self.assertTrue(index.is_file())
        payload = _viewer_payload(text)
        self.assertIn("官方/独立 SMPL-X 全身关节点", text)
        self.assertIn("Spider 已加载左右手网格", text)
        self.assertIn("官方物体可视网格", text)
        self.assertIn("Spider 物体可视网格", text)
        self.assertIn("最近表面连接线", text)
        self.assertIn("世界坐标", text)
        self.assertIn("const camera", text)
        self.assertIn("qp.get('compare')", text)
        self.assertIn("DecompressionStream('gzip')", text)
        self.assertIn("<script type='module'>const P=", text)
        self.assertIn("__binary_array__", text)
        self.assertIn("const U=new Uint8Array", text)
        self.assertIn("mesh('官方物体',pts(f.object,D.object.vertices)", text)
        self.assertIn("mesh('Spider物体',pts(f.object,D.object.vertices)", text)
        self.assertEqual(len(payload["frames"]), 3)
        self.assertEqual([frame["source_frame"] for frame in payload["frames"]], [0, 1, 2])
        self.assertIn("body_joints", payload["frames"][0]["official"])
        self.assertNotIn("body", payload["frames"][0]["official"])

    def test_failure_html_fallback_and_two_mandatory_html_names(self) -> None:
        official = _source_values()
        spider = _source_values()
        contact = {"sides": {side: {"nearest_point_world": np.zeros((3, 5, 3))} for side in ("left", "right")}}
        stages = {"input_source_gate": "PASS", "stage_a": {"status": "PASS"}, "stage_b": {"status": "FAIL"}, "cxa": {"status": "NOT_RUN", "reason": "非标准步骤"}, "final_retarget": "RETARGET_FAILED_STAGE_B"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "html").mkdir()
            html, index = recheck.write_retarget_html(root, official, spider, contact, stages, None)
            text = html.read_text(encoding="utf-8")
            self.assertTrue(index.is_file())
            self.assertEqual(html.name, "02_wuji_spider_retarget_audit.html")
        self.assertIn("失败诊断", text)
        self.assertIn("Stage B", text)
        self.assertIn("FINAL RETARGET: FAIL", _viewer_payload(text)["status"])
        self.assertIn("Wuji 掌到指尖骨架", text)

    def test_key_frame_labels_and_contact_payload_are_serialized(self) -> None:
        official = _source_values(4)
        spider = _source_values(4)
        comparison, _, _ = recheck.compare_source(official, spider, 120.0)
        contact = {"sides": {side: {"max_suspension_proxy_frame": 3, "tip_distance_m": np.full((4, 5), 0.04), "tip_penetration_depth_m": np.zeros((4, 5)), "nearest_point_world": np.zeros((4, 5, 3))} for side in ("left", "right")}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "html").mkdir()
            html, _ = recheck.write_source_html(root, official, spider, comparison, contact, [("接触帧", 2), ("结束帧", 3)])
            payload = _viewer_payload(html.read_text(encoding="utf-8"))
        self.assertEqual(payload["key_frames"], [{"label": "接触帧", "frame": 2}, {"label": "结束帧", "frame": 3}])
        self.assertEqual(len(payload["frames"]), 4)
        self.assertEqual(payload["frames"][2]["contact"]["left"]["tip_distance"], [0.04] * 5)

    def test_capture_is_before_retarget_and_records_actual_adapter_output(self) -> None:
        source = inspect.getsource(recheck.capture_spider)
        self.assertIn("GrabAdapter", source)
        self.assertIn("not_retargeted", source)
        self.assertIn("load_sequence", source)

    def test_standard_pipeline_map_captures_stage_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "retarget").mkdir()
            (root / "retarget" / "paths.generated.yaml").write_text("workspace: {}\n", encoding="utf-8")
            recheck.write_standard_pipeline_map(root)
            mapping = json.loads((root / "reports" / "standard_pipeline_map.json").read_text(encoding="utf-8"))
            markdown = (root / "reports" / "STANDARD_PIPELINE_MAP.md").read_text(encoding="utf-8")
        self.assertEqual(mapping["stages"][-1]["name"], "C-XA")
        self.assertIn("NOT_RUN", mapping["stages"][-1]["output"])
        self.assertIn("run-wuji-ik", markdown)

    def test_partial_retarget_serialization_and_fresh_workspace_guard(self) -> None:
        official = _source_values()
        spider = _source_values()
        contact = {"sides": {side: {"nearest_point_world": np.zeros((3, 5, 3))} for side in ("left", "right")}}
        stages = {"input_source_gate": "PASS", "stage_a": {"status": "PASS"}, "stage_b": {"status": "PARTIAL"}, "cxa": {"status": "NOT_RUN", "reason": "非标准步骤"}, "final_retarget": "RETARGET_PARTIAL"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "html").mkdir()
            html, _ = recheck.write_retarget_html(root, official, spider, contact, stages, None)
            page = html.read_text(encoding="utf-8")
        self.assertIn("FINAL RETARGET: PARTIAL", _viewer_payload(page)["status"])
        audit = inspect.getsource(recheck.audit_stage_b)
        self.assertIn("scratch_workspace.exists() or workspace.exists()", audit)
        self.assertIn("FileExistsError", audit)
        self.assertIn("shutil.move", audit)

    def test_refresh_is_derived_only_and_marks_all_nonrerun_boundaries(self) -> None:
        source = inspect.getsource(recheck.refresh_existing_evidence)
        self.assertNotIn("capture_spider(", source)
        self.assertNotIn("audit_stage_b(", source)
        self.assertIn("DERIVED_EVIDENCE_REFRESH", source)
        self.assertIn("not_rerun", source)
        self.assertIn("source_v2", source)
        self.assertIn("retarget_v2", source)

    def test_capture_focuses_object_and_wrist_views_without_hiding_interactive_layers(self) -> None:
        source = inspect.getsource(recheck._ChromeCdpPage.set_view_and_capture)
        self.assertIn("officialbody','spiderbody','sourcebody", source)
        self.assertIn("coordinate == 'world'", source)
        self.assertIn("window.__auditDrawPromise", source)
        viewer = inspect.getsource(recheck._html_document_v2)
        self.assertIn("window.__auditDrawPromise=Plotly.react", viewer)
        self.assertIn("for(let v of pts(T,L.vertices))vv.push(v)", viewer)

    def test_fresh_output_namespace_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exists"; root.mkdir()
            with self.assertRaises(FileExistsError):
                recheck.run("configs/local/paths.yaml", temporary, "exists")


if __name__ == "__main__":
    unittest.main()
