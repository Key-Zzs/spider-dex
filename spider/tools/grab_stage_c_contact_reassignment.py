"""Stage C-X artifact builder for Contract V2 task-equivalent contact.

This tool only derives V2 artifacts below ``stage_c_contract_v2``.  It never
rewrites Stage B, V1 recovery artifacts, raw GRAB, or the frozen-pilot list.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import yaml

from spider.contact.embodiment_assignment import (
    FINGERS, allowed_region, assignment_cost, classify_role, default_robot_regions,
    select_minimum_successful_level, viterbi_assignment,
)
from spider.datasets.paths import load_project_paths

PILOTS = {
    "s5__cylindermedium_lift": [1460, 1876],
    "s1__mug_lift": [120, 240],
    "s1__mug_offhand_1": [120, 180],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(paths_config: str):
    return load_project_paths(paths_config)


def _robot_dir(workspace: Path, sequence_id: str) -> Path:
    return workspace / "processed/grab/wuji_hand2_beta1/bimanual" / sequence_id / "0"


def _v2_dir(workspace: Path, sequence_id: str) -> Path:
    return _robot_dir(workspace, sequence_id) / "stage_c_contract_v2"


def freeze_v1_manifest(paths_config: str) -> str:
    """Write a small external V1 inventory without copying large artifacts."""
    paths = _paths(paths_config)
    workspace = paths.workspace_root
    primary = _robot_dir(workspace, "s5__cylindermedium_lift")
    report = primary / "stage_c_recovery/primary_r4_infeasibility_report.json"
    validation = workspace / "reports/stage_c_validation.json"
    conflict = primary / "stage_c_recovery/primary_r4_contact_collision_conflict.json"
    pareto = primary / "stage_c_recovery/depenetration_multistart_pareto.json"
    sanity = _robot_dir(workspace, "s1__mug_pass_1") / "stage_c_recovery/auxiliary_mjwp_sanity.json"
    v1_contract = Path("configs/project/grab_wuji_stage_c_contract.yaml")
    v1_profile = Path("configs/project/grab_wuji_depenetration.yaml")
    inputs = {"v1_contract": v1_contract, "v1_profile": v1_profile, "infeasibility_report": report,
              "stage_c_validation": validation, "contact_collision_conflict": conflict,
              "multistart_pareto": pareto, "mjwp_sanity": sanity}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V1 freeze inputs missing: {missing}")
    infeasible = json.loads(report.read_text(encoding="utf-8"))
    if infeasible.get("status") != "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT":
        raise RuntimeError("Refusing to freeze a V1 result that is not the immutable blocked state")
    conflict_data = json.loads(conflict.read_text(encoding="utf-8"))
    pareto_data = json.loads(pareto.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "stage": "C-X0",
        "v1_status": "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT",
        "immutable_statement": "V1 result is immutable and remains BLOCKED.",
        "frozen_primary": {"sequence_id": "s5__cylindermedium_lift", "frame_range": PILOTS["s5__cylindermedium_lift"]},
        "best_dynamic_candidate": conflict_data.get("best_bounded_candidate"),
        "v1_profile_hash": _hash(v1_profile),
        "pareto_front_candidate_ids": pareto_data.get("pareto_front_candidate_ids"),
        "active_joint_limits": {str(row.get("candidate_id")): row.get("joint_limit_active_set") for row in pareto_data.get("candidates", [])},
        "artifacts": {name: {"path": str(path), "sha256": _hash(path), "size_bytes": path.stat().st_size} for name, path in inputs.items()},
    }
    output = workspace / "reports/stage_c_contract_v1_manifest.json"
    _write_json(output, manifest)
    return str(output)


def build_source_contact_roles(
    paths_config: str, sequence_id: str, contract_path: str = "configs/project/grab_wuji_stage_c_contract_v2.yaml"
) -> str:
    """Classify active source contacts before any robot-region evaluation."""
    if sequence_id not in PILOTS:
        raise ValueError("V2 roles are restricted to frozen pilots")
    paths = _paths(paths_config)
    contract_file = Path(contract_path)
    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    if contract.get("contract_name") != "TASK_EQUIVALENT_CONTACT" or contract.get("contract_version") != 2:
        raise RuntimeError("invalid Contract V2")
    source = _robot_dir(paths.workspace_root, sequence_id) / "stage_c/contact_reference.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    active = [row for row in payload.get("records", []) if row.get("contact_flag") and row.get("confidence") == "high"]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in active:
        key = (str(row["side"]), str(row["finger_region"]), int(row["contact_interval_id"]))
        groups.setdefault(key, []).append(row)
    roles: list[dict[str, Any]] = []
    for role_id, ((side, finger, interval_id), rows) in enumerate(sorted(groups.items())):
        frames = sorted(int(row["frame_index"]) for row in rows)
        overlap = {other_finger for (other_side, other_finger, _other_interval), other_rows in groups.items()
                   if other_side == side and set(frames).intersection(int(row["frame_index"]) for row in other_rows)}
        label, confidence, rationale = classify_role(finger, len(frames), overlap)
        points = np.asarray([row["surface_point_world"] for row in rows], dtype=np.float64)
        normals = np.asarray([row["surface_normal_world"] for row in rows], dtype=np.float64)
        roles.append({
            "role_id": f"{sequence_id}:{role_id}", "side": side, "source_finger": finger,
            "source_contact_interval_id": interval_id, "frame_start": min(frames), "frame_end": max(frames),
            "frame_count": len(frames), "functional_role": label, "confidence": confidence,
            "generation_basis": rationale + ["source contact interval", "source object surface projection", "source normal", "source motion interval"],
            "source_anchor_world_mean": points.mean(axis=0), "source_normal_world_mean": normals.mean(axis=0),
            "source_frame_indices": frames,
        })
    result = {"schema_version": 1, "stage": "C-X1", "sequence_id": sequence_id, "contract_name": contract["contract_name"],
              "contract_hash": _hash(contract_file), "source_only": True, "roles": roles,
              "inactive_contact_policy": "NON_INTERACTING; no assignment candidates are created for inactive source contacts"}
    output = _v2_dir(paths.workspace_root, sequence_id) / "source_contact_roles.json"
    _write_json(output, result)
    np.savez_compressed(output.with_suffix(".npz"), role_frame_starts=np.asarray([row["frame_start"] for row in roles], dtype=np.int32), role_frame_ends=np.asarray([row["frame_end"] for row in roles], dtype=np.int32), role_confidence=np.asarray([row["confidence"] for row in roles], dtype=np.float64))
    return str(output)


def build_assignment_candidates(
    paths_config: str, sequence_id: str, level: int, contract_path: str = "configs/project/grab_wuji_stage_c_contract_v2.yaml"
) -> str:
    """Build bounded same-side candidate sets and select interval-stable assignments."""
    if sequence_id not in PILOTS or level not in range(1, 5):
        raise ValueError("V2 candidate generation requires frozen pilot and level 1..4")
    paths = _paths(paths_config); workspace = paths.workspace_root
    contract_file = Path(contract_path); contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    role_path = _v2_dir(workspace, sequence_id) / "source_contact_roles.json"
    if not role_path.is_file():
        build_source_contact_roles(paths_config, sequence_id, contract_path)
    roles = json.loads(role_path.read_text(encoding="utf-8"))["roles"]
    regions = default_robot_regions(); weights = contract["assignment"]["weights"]
    candidates: list[dict[str, Any]] = []; selected: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []
    max_switches = max(0, int(np.floor((max((role["frame_count"] for role in roles), default=0) / 120.0) * contract["assignment"]["max_switches_per_second"])))
    for role in roles:
        allowed = [region for region in regions if allowed_region(region, role["side"], role["source_finger"], role["functional_role"], level)]
        if not allowed:
            trace.append({"role_id": role["role_id"], "status": "NO_ALLOWED_REGION", "level": level})
            continue
        options: list[dict[str, Any]] = []
        for region in allowed:
            identity = 0.0 if region.finger == role["source_finger"] else 1.0
            terms = {"surface_patch": 0.0, "normal": 0.0, "functional_role": 0.0, "identity_change": identity,
                     "reachability": 0.0, "collision_risk": 0.0, "tracking_deviation": identity, "temporal_switch": 0.0}
            options.append({"robot_region": region.region_id, "cost_terms": terms, "cost": assignment_cost(terms, weights), "capacity": region.capacity})
        candidates.append({"role_id": role["role_id"], "level": level, "options": options})
        frame_costs = np.repeat(np.asarray([[item["cost"] for item in options]], dtype=float), role["frame_count"], axis=0)
        path, cost, switches = viterbi_assignment(frame_costs, float(weights["temporal_switch"]), max_switches)
        selected_option = options[int(path[0])]
        selected.append({"role_id": role["role_id"], "source_side": role["side"], "source_finger": role["source_finger"],
                         "functional_role": role["functional_role"], "frame_start": role["frame_start"], "frame_end": role["frame_end"],
                         "selected_robot_region": selected_option["robot_region"], "relaxation_level": level,
                         "assignment_cost": cost, "cost_terms": selected_option["cost_terms"], "switch_count": switches,
                         "assignment_duration_frames": role["frame_count"], "source_patch_id": f"patch:{role['role_id']}"})
        trace.append({"role_id": role["role_id"], "path": path.tolist(), "switch_count": switches, "max_switches": max_switches, "status": "SELECTED"})
    status = "PASS_CANDIDATE_GENERATION" if len(selected) == len(roles) else "FAIL_CANDIDATE_GENERATION"
    root = _v2_dir(workspace, sequence_id)
    candidate_path = root / f"contact_assignment_candidates_level_{level}.json"
    _write_json(candidate_path, {"schema_version": 1, "stage": "C-X3", "sequence_id": sequence_id, "level": level, "status": status, "candidates": candidates})
    selected_path = root / f"selected_contact_assignment_level_{level}.json"
    _write_json(selected_path, {"schema_version": 1, "stage": "C-X3", "sequence_id": sequence_id, "level": level, "status": status, "selected": selected, "source_roles": str(role_path), "contract_hash": _hash(contract_file)})
    _write_json(root / f"contact_assignment_trace_level_{level}.json", {"schema_version": 1, "sequence_id": sequence_id, "level": level, "trace": trace})
    np.savez_compressed(root / f"selected_contact_assignment_level_{level}.npz", assignment_cost=np.asarray([row["assignment_cost"] for row in selected]), switch_count=np.asarray([row["switch_count"] for row in selected], dtype=np.int32))
    return str(selected_path)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"freeze-v1-manifest": freeze_v1_manifest, "build-source-contact-roles": build_source_contact_roles, "build-assignment-candidates": build_assignment_candidates})
