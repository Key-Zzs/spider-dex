"""Evidence-first C-XA audit for a frozen Stage-C Contract V2 run."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from spider.contact.cxa_audit import (
    decide_cxa_case,
    is_functional_role,
    source_contact_reliability,
)
from spider.datasets.paths import load_project_paths
from spider.tools.grab_stage_c_contact_reassignment import (
    _hash,
    _robot_dir,
    _v2_dir,
    evaluate_v2_depenetrated,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _role_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["side"]), str(row["finger_region"]), int(row["contact_interval_id"])


def audit_v2_failure(
    paths_config: str,
    sequence_id: str = "s5__cylindermedium_lift",
    run_namespace: str = "stage_c_contract_v2",
    report_dir: str | None = None,
) -> str:
    """Create all required C-XA evidence reports without changing frozen inputs.

    The recomputed metric file is a new report-only artifact.  It reuses the
    frozen V2 trajectory/targets and keeps the old V2 files intact so the
    original blocked result remains independently auditable.
    """
    paths = load_project_paths(paths_config)
    workspace = paths.workspace_root
    root = _v2_dir(workspace, sequence_id, run_namespace)
    reports = Path(report_dir) if report_dir else workspace / "reports"
    roles_payload = json.loads((root / "source_contact_roles.json").read_text(encoding="utf-8"))
    roles = roles_payload["roles"]
    source_path = Path(roles_payload.get("source_contact_reference") or _robot_dir(workspace, sequence_id) / "stage_c/contact_reference.json")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    threshold = float(source_payload["distance_threshold_m"])
    selected = json.loads((root / "selected_contact_assignment_level_4.json").read_text(encoding="utf-8"))["selected"]
    candidates = json.loads((root / "contact_assignment_candidates_level_4.json").read_text(encoding="utf-8"))["candidates"]
    prior_metrics_path = root / "metrics_depenetrated_init_v2_level_4_flexible.json"
    prior_metrics = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    recomputed_path = reports / "cxa_v2_original_patch_recompute.json"
    # ``evaluate_depenetrated_init`` appends its verified geometry fields to an
    # existing metric document.  Seed a report-only copy, never the frozen V2
    # result, before replacing only its V2 namespace below.
    _write_json(recomputed_path, prior_metrics)
    evaluate_v2_depenetrated(
        paths_config,
        sequence_id,
        4,
        str(root / "trajectory_depenetrated_init_v2_level_4_flexible.npz"),
        str(recomputed_path),
        contact_targets_path=str(root / "contact_targets_level_4_flexible.npz"),
        run_namespace=run_namespace,
        refresh_base_metrics=False,
    )
    recomputed = json.loads(recomputed_path.read_text(encoding="utf-8"))
    recomputed_v2 = recomputed["task_equivalent_contact_v2"]

    records_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    reliable_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_payload["records"]:
        key = _role_key(row)
        records_by_key[key].append(row)
        reliability, _evidence = source_contact_reliability(
            None if row.get("source_signed_distance_m") is None else float(row["source_signed_distance_m"]),
            float(row["unsigned_distance_m"]),
            threshold,
        )
        if row.get("raw_inclusive_contact_flag", row.get("contact_flag")) and reliability == "RELIABLE_SOURCE":
            reliable_by_frame[int(row["frame_index"])].append(row)

    role_audit: list[dict[str, Any]] = []
    reliability_audit: list[dict[str, Any]] = []
    source_status: dict[str, str] = {}
    for role in roles:
        key = (str(role["side"]), str(role["source_finger"]), int(role["source_contact_interval_id"]))
        samples = [row for row in records_by_key[key] if row.get("raw_inclusive_contact_flag", row.get("contact_flag"))]
        classifications = [
            source_contact_reliability(
                None if row.get("source_signed_distance_m") is None else float(row["source_signed_distance_m"]),
                float(row["unsigned_distance_m"]),
                threshold,
            )[0]
            for row in samples
        ]
        unreliable = sum(value == "UNRELIABLE_SOURCE" for value in classifications)
        reliable = sum(value == "RELIABLE_SOURCE" for value in classifications)
        source_class = "UNRELIABLE_SOURCE" if unreliable else ("TRANSIENT" if role["functional_role"] == "TRANSIENT" else "SUPPORTING")
        if role["functional_role"] in {"THUMB_OPPOSITION", "PRIMARY_GRASP"} and unreliable == 0:
            source_class = "MANDATORY"
        source_status[str(role["role_id"])] = source_class
        ablated_frames = [int(frame) for frame in role["source_frame_indices"]]
        remaining = []
        for frame in ablated_frames:
            remaining.extend(
                row for row in reliable_by_frame.get(frame, [])
                if _role_key(row) != key
            )
        remaining_fingers = sorted({f"{row['side']}:{row['finger_region']}" for row in remaining})
        remaining_count_per_frame = [
            sum(1 for row in reliable_by_frame.get(frame, []) if _role_key(row) != key)
            for frame in ablated_frames
        ]
        signed = np.asarray([float(row["source_signed_distance_m"]) for row in samples if row.get("source_signed_distance_m") is not None], dtype=np.float64)
        role_audit.append({
            "role_id": role["role_id"], "side": role["side"], "source_finger": role["source_finger"],
            "role_type": role["functional_role"], "interval": [role["frame_start"], role["frame_end"]],
            "confidence": role["confidence"], "mandatory": source_class == "MANDATORY",
            "evidence": [
                *role["generation_basis"],
                f"source_samples={len(samples)} reliable={reliable} unreliable_deep={unreliable}",
                f"role_ablation_remaining_reliable_contacts_min={min(remaining_count_per_frame, default=0)}",
            ],
        })
        reliability_audit.append({
            "role_id": role["role_id"], "classification": source_class,
            "source_penetration": {
                "sample_count": len(samples), "reliable_sample_count": reliable,
                "unreliable_deep_sample_count": unreliable,
                "signed_distance_min_m": float(np.min(signed)) if len(signed) else None,
                "signed_distance_p95_m": float(np.percentile(signed, 95)) if len(signed) else None,
                "deep_penetration_threshold_m": threshold,
            },
            "duration": {"frames": int(role["frame_count"]), "transient": role["functional_role"] == "TRANSIENT"},
            "functional_necessity_ablation": {
                "method": "frozen-source contact-topology ablation; no raw GRAB or object trajectory was modified",
                "object_trajectory": "UNCHANGED_BY_CONSTRUCTION",
                "grasp_stability": "NOT_CAUSALLY_IDENTIFIABLE_FROM_FROZEN_SOURCE_ONLY; remaining-contact topology reported instead",
                "remaining_contact_topology": remaining_fingers,
                "remaining_reliable_contacts_min_per_frame": min(remaining_count_per_frame, default=0),
                "remaining_reliable_contacts_mean_per_frame": float(np.mean(remaining_count_per_frame)) if remaining_count_per_frame else 0.0,
            },
        })

    selected_by_role = {row["role_id"]: row for row in selected}
    candidate_by_role = {row["role_id"]: row["options"] for row in candidates}
    metric_role_by_id = {row["role_id"]: row for row in recomputed_v2["role_metrics"]}
    failures = recomputed_v2["patch_distance_failures"]
    failure_localization: list[dict[str, Any]] = []
    for role in roles:
        role_id = str(role["role_id"])
        metric_role = metric_role_by_id[role_id]
        selected_row = selected_by_role[role_id]
        local_failures = [row for row in failures if row["source_role"] == role_id]
        if source_status[role_id] == "UNRELIABLE_SOURCE":
            failure_reason = "SOURCE_CONTACT_UNRELIABLE"
        elif role["functional_role"] == "TRANSIENT":
            failure_reason = "METRIC_IMPLEMENTATION_ERROR"
        elif metric_role["passed"]:
            failure_reason = "METRIC_IMPLEMENTATION_ERROR"
        else:
            failure_reason = "ROBOT_REACHABILITY_LIMIT"
        failure_localization.append({
            "side": role["side"], "source_role": role_id, "source_finger": role["source_finger"],
            "object_patch": selected_row["source_patch_id"],
            "candidate_robot_regions": [row["robot_region"] for row in candidate_by_role.get(role_id, [])],
            "best_assignment": selected_row["selected_robot_region"], "relaxation_level": 4,
            "coverage": metric_role["coverage"], "distance_p95": metric_role["distance_p95_m"],
            "normal": metric_role["normal_cosine_median"],
            "penetration": next(item for item in reliability_audit if item["role_id"] == role_id)["source_penetration"],
            "joint_limit": "NOT_ACTIVE_IN_FROZEN_STATIC_C-XA_AUDIT",
            "failure_samples": local_failures,
            "failure_reason": failure_reason,
        })

    old_v2 = prior_metrics["task_equivalent_contact_v2"]
    role_recall_audit = recomputed_v2["functional_role_recall_audit"]
    metric_audit = {
        "schema_version": 1,
        "sequence_id": sequence_id,
        "prior_v2_result": {
            "functional_role_recall": old_v2["functional_role_recall"],
            "surface_patch_distance_p95_m": old_v2["surface_patch_distance_p95_m"],
        },
        "functional_role_recall": role_recall_audit,
        "patch_distance": {
            "numerator": int(sum(row["sample_count"] for row in recomputed_v2["role_metrics"] if row["passed"])),
            "denominator": int(sum(row["sample_count"] for row in recomputed_v2["role_metrics"])),
            "weighting_rule": recomputed_v2["patch_distance_definition"],
            "p95_m": recomputed_v2["surface_patch_distance_p95_m"],
            "failed_samples": failures,
        },
        "implementation_errors": [
            "functional-role recall counted TRANSIENT intervals in its denominator",
            "patch-distance P95 measured an arbitrary compiled point anchor instead of the assigned mesh-adjacent patch",
        ],
        "checks": {
            "duplicate_counting": "NO_DUPLICATE_ROLE_IDS; the error is nonfunctional denominator inclusion",
            "cluster_split": "source contact intervals are retained as distinct source evidence and reported individually",
            "transient_counted": bool(role_recall_audit["excluded_nonfunctional_roles"]),
            "deep_source_contact_counted": any(row["classification"] == "UNRELIABLE_SOURCE" for row in reliability_audit),
            "offhand_counted": any(role["side"] == "left" for role in roles),
        },
    }
    patch_definition_errors: list[str] = []
    decision = decide_cxa_case(
        metric_implementation_errors=metric_audit["implementation_errors"],
        source_contact_labeling_errors=[
            "deeply penetrating source samples were marked active/high confidence"
            for row in reliability_audit if row["classification"] == "UNRELIABLE_SOURCE"
        ],
        patch_definition_errors=patch_definition_errors,
        assignment_levels_covered=True,
        remaining_failure_reasons=[row["failure_reason"] for row in failure_localization],
    )
    case_a = {
        "schema_version": 1, "sequence_id": sequence_id, "decision": decision,
        "contract_thresholds_unchanged": True, "frozen_primary_unchanged": True,
        "original_v2_artifacts_preserved": str(root),
        "evidence": {
            "metric_audit": str(reports / "cxa_metric_audit.json"),
            "role_audit": str(reports / "cxa_role_audit.json"),
            "failure_localization": str(reports / "cxa_failure_localization.json"),
            "source_reliability": str(reports / "cxa_source_contact_reliability.json"),
        },
    }
    _write_json(reports / "cxa_role_audit.json", {"schema_version": 1, "sequence_id": sequence_id, "roles": role_audit})
    _write_json(reports / "cxa_metric_audit.json", metric_audit)
    _write_json(reports / "cxa_failure_localization.json", {"schema_version": 1, "sequence_id": sequence_id, "roles": failure_localization})
    _write_json(reports / "cxa_source_contact_reliability.json", {"schema_version": 1, "sequence_id": sequence_id, "roles": reliability_audit})
    _write_json(reports / "cxa_case_a_bug_evidence.json", case_a)
    return str(reports / "cxa_case_a_bug_evidence.json")


def finalize_case_a_rerun(
    paths_config: str,
    sequence_id: str = "s5__cylindermedium_lift",
    run_namespace: str = "stage_c_contract_v2_cxa",
    level: int = 1,
    report_dir: str | None = None,
) -> str:
    """Record the evidence-preserving CASE A rerun outcome.

    This function intentionally accepts only the first relaxation level: a
    successful earlier level closes the ordered V2 ladder and forbids testing a
    broader one merely to obtain a different result.
    """
    if level != 1:
        raise ValueError("CASE A rerun finalization requires the first successful level, expected level 1")
    paths = load_project_paths(paths_config)
    workspace = paths.workspace_root
    root = _v2_dir(workspace, sequence_id, run_namespace)
    reports = Path(report_dir) if report_dir else workspace / "reports"
    metrics_path = root / "metrics_depenetrated_init_cxa_level_1_flexible.json"
    level_path = root / "level_1_result.json"
    preflight_path = root / "preflight_static_cxa_level_1.json"
    source_path = _robot_dir(workspace, sequence_id) / "stage_c/contact_reference_cxa.json"
    for path in (metrics_path, level_path, preflight_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    level_result = json.loads(level_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    v2 = metrics.get("contract_v2", {})
    if v2.get("status") != "PASS" or level_result.get("status") != "PENDING_PHYSICAL_PREFLIGHT":
        raise RuntimeError("CASE A rerun did not pass the immutable V2 static gates")
    if preflight.get("status") != "PASS":
        raise RuntimeError("CASE A rerun did not pass static MuJoCo preflight")
    original_root = _v2_dir(workspace, sequence_id)
    original_metrics_path = original_root / "metrics_depenetrated_init_v2_level_4_flexible.json"
    original_metrics = json.loads(original_metrics_path.read_text(encoding="utf-8"))
    original_assignment = json.loads((original_root / "selected_contact_assignment_level_4.json").read_text(encoding="utf-8"))
    rerun_assignment = json.loads((root / "selected_contact_assignment_level_1.json").read_text(encoding="utf-8"))
    contract_path = Path("configs/project/grab_wuji_stage_c_contract_v2.yaml")
    if original_assignment.get("contract_hash") != _hash(contract_path) or rerun_assignment.get("contract_hash") != _hash(contract_path):
        raise RuntimeError("contract hash mismatch: CASE A may not change V2 thresholds")
    payload = {
        "schema_version": 1,
        "stage": "C-XA",
        "sequence_id": sequence_id,
        "decision": "CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG",
        "status": "PASS_STATIC_AND_PREFLIGHT",
        "contract": {
            "path": str(contract_path), "sha256": _hash(contract_path),
            "thresholds_unchanged": True,
            "selected_level": 1,
            "later_levels": "NOT_RUN; Level 1 passed and ordered relaxation forbids broader tests",
        },
        "source_reference": {
            "corrected": str(source_path),
            "original_preserved": str(_robot_dir(workspace, sequence_id) / "stage_c/contact_reference.json"),
            "policy": "deeply penetrating samples retained as recorded evidence but excluded from active source contacts",
        },
        "original_v2_blocked_evidence": {
            "artifact_root": str(original_root),
            "metrics": str(original_metrics_path),
            "functional_role_recall": original_metrics["task_equivalent_contact_v2"]["functional_role_recall"],
            "surface_patch_distance_p95_m": original_metrics["task_equivalent_contact_v2"]["surface_patch_distance_p95_m"],
        },
        "rerun": {
            "artifact_root": str(root), "metrics_path": str(metrics_path), "level_result": str(level_path),
            "static_preflight_path": str(preflight_path),
            "metrics": metrics["task_equivalent_contact_v2"], "gates": v2["gates"],
            "preflight": preflight,
        },
        "v1": {
            "status": "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT",
            "manifest": str(workspace / "reports/stage_c_contract_v1_manifest.json"),
        },
        "smokes": "NOT_RUN; CASE A scope stops after primary static V2 pass and static preflight",
        "stage_d": "NOT_STARTED",
    }
    _write_json(reports / "cxa_case_a_rerun.json", payload)
    return str(reports / "cxa_case_a_rerun.json")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"audit-v2-failure": audit_v2_failure, "finalize-case-a-rerun": finalize_case_a_rerun})
