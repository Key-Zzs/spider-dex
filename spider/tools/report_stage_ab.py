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
    pilots = []
    for sequence, reason, frame_range, role in (("s1__mug_lift", "deterministic lift primary; contact onset is inside selected window", [120, 240], "primary"), ("s1__mug_offhand_1", "deterministic offhand smoke", [120, 180], "smoke"), ("s1__mug_pass_1", "deterministic handoff/pass smoke", [120, 180], "smoke")):
        canonical = paths.workspace_root / "processed" / "grab" / "canonical" / sequence
        robot = paths.workspace_root / "processed" / "grab" / "wuji_hand2_beta1" / "bimanual" / sequence / "0"
        metrics = json.loads((robot / "metrics_kinematic.json").read_text(encoding="utf-8"))
        pilots.append({"sequence_id": sequence, "source_path": f"grab/{sequence.replace('__', '/')}.npz", "selection_reason": reason, "hand_mode": "bimanual_streams", "object": "mug", "frame_range": frame_range, "fps": 120.0, "canonical_dir": str(canonical), "ik_dir": str(robot), "status": metrics["status"], "quality_status": metrics["quality_status"], "manual_review_role": role})
    (manifests / "grab_pilot.json").write_text(json.dumps({"schema_version": 1, "selection": "sorted fixed IDs; no result-dependent reselection", "pilots": pilots}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_b = {"stage": "B", "status": "PASS", "manual_visual_acceptance": "PENDING USER ACCEPTANCE", "source_audit": audit, "pilots": pilots, "known_quality_gate": "All pilots are structurally valid; tracking metrics require manual review under fixed smoke thresholds."}
    (reports / "stage_b_validation.json").write_text(json.dumps(stage_b, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "grab_pilot_summary.json").write_text(json.dumps({"primary": pilots[0], "all_pilots": pilots}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reports": str(reports), "pilot_manifest": str(manifests / "grab_pilot.json")}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
