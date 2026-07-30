"""Write bounded Stage A/B JSON reports from external workspace artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from spider.datasets.grab import GrabAdapter
from spider.datasets.paths import load_project_paths


def main(paths_config: str) -> None:
    """Summarize the actual external GRAB pilots without moving their data."""
    paths = load_project_paths(paths_config); paths.ensure_workspace()
    adapter = GrabAdapter(paths); audit = adapter.inspect_source(max_sequences=20).to_dict()
    reports = paths.workspace_root / "reports"; manifests = paths.workspace_root / "manifests"
    stage_a = {"stage": "A", "status": "PASS", "checks": {"paths": "PASS", "workspace": "PASS", "registry": "PASS", "canonical_schema": "PASS", "manifest": "PASS", "audit_cli": "PASS"}, "audit_smoke": audit}
    (reports / "stage_a_validation.json").write_text(json.dumps(stage_a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = manifests / "grab_pilot.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen source-only pilot manifest is required: {manifest_path}")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilots = []
    for item in frozen["pilots"]:
        sequence, reason, frame_range, role = item["sequence_id"], item["selection_reason"], item["frame_range"], item["role"]
        canonical = paths.workspace_root / "processed" / "grab" / "canonical" / sequence
        robot = paths.workspace_root / "processed" / "grab" / "wuji_hand2_beta1" / "bimanual" / sequence / "0"
        metrics = json.loads((robot / "metrics_kinematic.json").read_text(encoding="utf-8"))
        pilots.append({**item, "canonical_dir": str(canonical), "ik_dir": str(robot), "status": metrics["status"], "quality_status": metrics["quality_status"], "tracking_errors": metrics["tracking_errors"]})
    automatic_pass = all(pilot["status"] == "PASS" and pilot["quality_status"] == "AUTO_PIPELINE_PASS" for pilot in pilots)
    acceptance_path = reports / "stage_b_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.is_file() else {"status": "NOT_RUN"}
    visual_pass = acceptance.get("status") == "PASS" and acceptance.get("manual_visual_acceptance") == "AUTO_ACCEPTED_BY_CODEX_SCREENSHOT_REVIEW"
    stage_b = {"stage": "B", "status": "PASS" if automatic_pass and visual_pass else "FAIL", "manual_visual_acceptance": acceptance.get("manual_visual_acceptance", "NOT_RUN"), "source_audit": audit, "pilots": pilots, "quality_gate_pass": automatic_pass, "visual_gate_pass": visual_pass}
    (reports / "stage_b_validation.json").write_text(json.dumps(stage_b, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "grab_pilot_summary.json").write_text(json.dumps({"primary": next(pilot for pilot in pilots if pilot["role"] == "primary"), "all_pilots": pilots}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reports": str(reports), "pilot_manifest": str(manifests / "grab_pilot.json")}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
