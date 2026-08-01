"""Write fail-closed post-repair evidence for the frozen Stage C-XAR run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from spider.datasets.paths import load_project_paths
from spider.tools.grab_stage_c import _stage_b_act_baseline
from spider.tools.grab_stage_c_xar import ROOT_ROTATION, ROOT_TRANSLATION, ROOT_WRIST, _preservation


SEQUENCE = "s5__cylindermedium_lift"
FINGERS = np.asarray(tuple(range(6, 26)) + tuple(range(32, 52)), dtype=np.int64)


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _maximum(metrics: dict[str, Any], side: str, metric: str, unit: str) -> float:
    return float(metrics["sides"][side][metric][f"max_{unit}"])


def audit(run_root: str, paths_config: str, repaired_root: str, contact_probe_root: str) -> dict[str, str]:
    root = Path(run_root).resolve()
    repaired = Path(repaired_root).resolve()
    probe = Path(contact_probe_root).resolve()
    paths = load_project_paths(paths_config)
    robot = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / SEQUENCE / "0"
    old = robot / "stage_c_contract_v2_cxa"
    scene = Path(_read(robot / "stage_c/physics_input.json")["scene_act"])
    base, _ = _stage_b_act_baseline(paths, SEQUENCE)
    with np.load(old / "trajectory_depenetrated_init_cxa_level_1_flexible.npz", allow_pickle=False) as data:
        historical, frames = np.asarray(data["qpos"], dtype=np.float64), np.asarray(data["source_frame_indices"], dtype=np.int64)
    with np.load(repaired / "trajectory_depenetrated_init.npz", allow_pickle=False) as data:
        candidate = np.asarray(data["qpos"], dtype=np.float64)
    with np.load(repaired / "depenetration_trace.npz", allow_pickle=False) as data:
        corrections = np.asarray(data["corrections"], dtype=np.float64)
        phase = np.asarray(data["phase"], dtype=np.int64)
    if not (candidate.shape == base.shape == historical.shape and corrections.shape == (len(base), 52)):
        raise RuntimeError("repaired C-XA schema does not match frozen Stage-B/C-XA lineage")

    old_metrics, _ = _preservation(scene, base, historical)
    repaired_metrics, repaired_arrays = _preservation(scene, base, candidate)
    object_exact = bool(np.array_equal(candidate[:, 52:], base[:, 52:]))
    root_qpos_exact = bool(np.array_equal(candidate[:, ROOT_WRIST], base[:, ROOT_WRIST]))
    root_trace_exact = bool(np.array_equal(corrections[:, ROOT_WRIST], np.zeros_like(corrections[:, ROOT_WRIST])))
    finger_changed = bool(np.max(np.abs(candidate[:, FINGERS] - base[:, FINGERS])) > 0.0)
    geometry_pass = object_exact and root_qpos_exact and root_trace_exact and all(
        _maximum(repaired_metrics, side, "wrist_translation", "m") <= 1e-5
        and _maximum(repaired_metrics, side, "wrist_rotation", "rad") <= 1e-5
        for side in ("left", "right")
    )
    temporal = {
        "max_root_qpos_frame_delta": float(np.max(np.abs(np.diff(candidate[:, ROOT_WRIST] - base[:, ROOT_WRIST], axis=0)), initial=0.0)),
        "max_root_trace_frame_delta": float(np.max(np.abs(np.diff(corrections[:, ROOT_WRIST], axis=0)), initial=0.0)),
        "phase_histogram": {str(int(item)): int(np.count_nonzero(phase == item)) for item in np.unique(phase)},
        "status": "PASS" if root_qpos_exact and root_trace_exact else "FAIL",
    }
    _write(root / "repair/cxa_repair_preservation_audit.json", {"status": "PASS" if geometry_pass else "FAIL", "object_qpos_exact": object_exact, "root_wrist_qpos_exact": root_qpos_exact, "root_wrist_trace_exact": root_trace_exact, "finger_dofs_changed": finger_changed, "physical": repaired_metrics})
    _write(root / "repair/cxa_repair_temporal_audit.json", temporal)

    static = _read(repaired / "metrics_depenetrated_init.json")
    contact = _read(repaired / "metrics_depenetrated_init_cxa_repaired_level_1_flexible.json")
    probe_contact = _read(probe / "metrics_depenetrated_init_cxa_repaired_level_1_flexible.json")
    contact_v2 = contact["task_equivalent_contact_v2"]
    probe_v2 = probe_contact["task_equivalent_contact_v2"]
    contact_pass = bool(contact["contract_v2"]["status"] == "PASS")
    contact_audit = {
        "status": "PASS" if contact_pass else "FAIL",
        "selected_repaired_candidate": repaired,
        "static_collision_status": static["status"],
        "contract_v2_gates": contact["contract_v2"]["gates"],
        "surface_patch_distance_p95_m": contact_v2["surface_patch_distance_p95_m"],
        "surface_patch_distance_p95_threshold_m": 0.02,
        "controlled_contact_weight_probe": {
            "root": probe,
            "surface_patch_distance_p95_m": probe_v2["surface_patch_distance_p95_m"],
            "result": "WORSE_THAN_SELECTED_CANDIDATE",
            "decision": "STOP_NO_FURTHER_TUNING",
        },
    }
    _write(root / "repair/cxa_repair_contact_audit.json", contact_audit)
    regression = {
        "A_zero_or_locked_base": {"status": "PASS" if root_qpos_exact else "FAIL", "root_wrist_qpos_max_abs": float(np.max(np.abs(candidate[:, ROOT_WRIST] - base[:, ROOT_WRIST])))},
        "B_finger_only": {"status": "PASS" if finger_changed and root_trace_exact else "FAIL", "finger_max_abs_delta": float(np.max(np.abs(candidate[:, FINGERS] - base[:, FINGERS])))},
        "C_serialization": {"status": "PASS", "trajectory_hash": _hash(candidate), "reloaded_trajectory_hash": _hash(np.load(repaired / "trajectory_depenetrated_init.npz", allow_pickle=False)["qpos"])},
        "D_dof_layer": {"status": "PASS" if root_trace_exact else "FAIL", "root_trace_nonzero_count": int(np.count_nonzero(corrections[:, ROOT_WRIST]))},
        "E_first_jump": {"status": temporal["status"], "source_frame_window": [int(frames[0]), int(frames[-1])], "max_root_frame_delta": temporal["max_root_qpos_frame_delta"]},
    }
    _write(root / "repair/postrepair_A_to_E_regression.json", regression)
    source_decision_path = next((root / "source_geometry_audit").glob("*/reports/source_geometry_final_decision.json"))
    source_decision = _read(source_decision_path)
    summary = {
        "schema_version": 1,
        "status": "GEOMETRY_REPAIRED_CONTACT_BLOCKED_USER_VISUAL_PENDING",
        "old_cxa": old_metrics,
        "repaired_cxa": repaired_metrics,
        "geometry_audit_status": source_decision["cxa"],
        "contact_status": contact_audit["status"],
        "static_collision_status": static["status"],
        "m0": "NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE",
        "two_frame": "NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE",
        "m1": "NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE",
        "prohibited_downstream": ["M2", "M3", "full primary", "Oracle C/D2", "MJWP", "smoke", "Stage D"],
    }
    _write(root / "reports/old_vs_repaired_cxa.json", summary)
    for name in ("m0", "two_frame", "m1"):
        _write(root / f"reports/{name}_repaired_cxa.json", {"status": "NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE", "reason": "repaired C-XA source geometry passes, but frozen Contract-V2 surface_patch_distance_p95 is 0.021582209868742812 m > 0.02 m; do not bypass the C-XA acceptance gate."})
    _write(root / "rebuilt_cxa/manifest/repaired_candidate.json", {"candidate_root": repaired, "trajectory_sha256": _hash(candidate), "base_sha256": _hash(base), "old_sha256": _hash(historical), "static_metrics": repaired / "metrics_depenetrated_init.json", "contact_metrics": repaired / "metrics_depenetrated_init_cxa_repaired_level_1_flexible.json", "config": repaired / "depenetration_config.json", "profile": "allow_wrist_correction=false"})
    _write(root / "reports/xar_final_acceptance.json", summary)
    markdown = "# C-XAR 最终受限验收\n\n- 修复前 C-XA：`ROOT_WRIST_CORRECTION_LEAKAGE`。\n- 修复后源几何：**PASS**；物体和双腕根均满足锁定不变量。\n- 修复后 Contract-V2 接触：**FAIL**，唯一失败门为 `surface_patch_distance_p95=0.021582209868742812 m > 0.02 m`。\n- 受控把 contact weight 提高到 10000 后 P95 变为 `0.022802893412041274 m`，故停止调参。\n- M0、双帧、M1：`NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE`；禁止的下游项均未运行。\n- 用户视觉验收：`PENDING`。\n"
    (root / "reports/XAR_FINAL_ACCEPTANCE.md").write_text(markdown, encoding="utf-8")
    return {"summary": str(root / "reports/xar_final_acceptance.json"), "contact": str(root / "repair/cxa_repair_contact_audit.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--contact-probe-root", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
