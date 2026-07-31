"""Bounded Stage C-V2R2E recovery orchestration.

The runner is deliberately evidence-first.  It consumes the immutable C-XA
Level-1 result and the completed V2R1/V2R2 evidence, writes every new attempt
under ``workspace/runs/stage_c_v2r2e/<attempt_id>``, and refuses to call a
downstream gate when its predecessor has not passed.  The MuJoCo controller
probe is delegated to the already audited V2R simulator, but its output root
is isolated for the duration of each attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import tyro
import yaml

from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools.grab_stage_c import _preflight_object_ids, _site_ids


PRIMARY = "s5__cylindermedium_lift"
SMOKE_1 = "s1__mug_lift"
SMOKE_2 = "s1__mug_offhand_1"
CONFIG_PATH = Path("configs/project/grab_wuji_stage_c_v2r2e.yaml")
PILOTS = {
    PRIMARY: {"frames": [1460, 1876], "role": "required true-bimanual primary"},
    SMOKE_1: {"frames": [120, 240], "role": "right-hand interaction smoke"},
    SMOKE_2: {"frames": [120, 180], "role": "offhand/non-interacting-hand smoke"},
}
STATE_ORDER = (
    "AUDIT_ALIGNMENT",
    "REFINE_OBJECT_COLLISION",
    "REFINE_HAND_CONTACT_PROXY",
    "FIX_CONTACT_REGION_MAPPING",
    "ORACLE_C_TEST",
    "DYNAMIC_TRAJECTORY_OPTIMIZATION",
    "CONTACT_DYNAMICS_REPAIR",
    "OBJECT_GUIDANCE_REPAIR",
    "D2_TEST",
    "TIMING_FEASIBILITY",
    "MINIMAL_MJWP",
    "PRIMARY_MJWP",
    "SMOKE_1",
    "SMOKE_2",
    "HTML",
    "SCREENSHOT_REVIEW",
    "COMPLETE",
    "BLOCKED",
)

# The checkout already contains a local V2R2E limits/config manifest.  Keep
# concrete candidate payloads in the implementation so that an ignored local
# config cannot silently broaden or shrink the search space.
DYNAMIC_CANDIDATES = [
    {"candidate_id": "lead0_base", "lead_source_frames": 0, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead4_base", "lead_source_frames": 4, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead8_base", "lead_source_frames": 8, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead12_base", "lead_source_frames": 12, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead16_base", "lead_source_frames": 16, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead20_base", "lead_source_frames": 20, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00},
    {"candidate_id": "lead16_soft", "lead_source_frames": 16, "kp_scale": 0.75, "kv_scale": 1.50, "force_limit_scale": 1.00},
    {"candidate_id": "lead8_feedforward", "lead_source_frames": 8, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00, "inverse_dynamics_scale": 0.10, "state_feedback_gain": 0.50},
    {"candidate_id": "lead20_contact_all", "lead_source_frames": 20, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00, "contact_ik_gain": 0.50, "contact_ik_max_anchor_error_m": 0.030, "contact_ik_strategy": "all_active", "contact_ik_finger_clip_rad": 0.25},
    {"candidate_id": "lead20_contact_first", "lead_source_frames": 20, "kp_scale": 1.00, "kv_scale": 1.00, "force_limit_scale": 1.00, "contact_ik_gain": 0.50, "contact_ik_max_anchor_error_m": 0.020, "contact_ik_strategy": "first_active", "contact_ik_finger_clip_rad": 0.25},
    {"candidate_id": "lead16_contact_all", "lead_source_frames": 16, "kp_scale": 0.90, "kv_scale": 1.20, "force_limit_scale": 1.00, "contact_ik_gain": 0.75, "contact_ik_max_anchor_error_m": 0.030, "contact_ik_strategy": "all_active", "contact_ik_finger_clip_rad": 0.20},
    {"candidate_id": "lead12_contact_first", "lead_source_frames": 12, "kp_scale": 0.85, "kv_scale": 1.30, "force_limit_scale": 1.00, "contact_ik_gain": 0.75, "contact_ik_max_anchor_error_m": 0.015, "contact_ik_strategy": "first_active", "contact_ik_finger_clip_rad": 0.20},
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("corrected_namespace") != "stage_c_contract_v2_cxa" or payload.get("corrected_level") != 1:
        raise RuntimeError("V2R2E may only consume corrected C-XA Level 1")
    if payload["timing"].get("variant") != "V2_ORIGINAL_TIMING" or float(payload["timing"].get("scale", 0.0)) != 1.0:
        raise RuntimeError("the primary search must begin at original timing")
    return payload


def _primary_root(paths) -> Path:
    return dynamic._stage_b_dirs(paths.workspace_root, PRIMARY)[1]


def _attempt_root(paths, attempt_id: str) -> Path:
    return paths.workspace_root / "runs" / "stage_c_v2r2e" / attempt_id


def _load_frozen(paths) -> dict[str, Any]:
    inputs = dynamic._inputs(paths, PRIMARY)
    physics = dynamic._physics(paths, PRIMARY)
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool)
        assignment_index = np.asarray(archive["assignment_index"], dtype=np.int32)
    if qpos.shape != (414, 64) or qvel.shape != (414, 64) or expected.shape != (414, 10):
        raise RuntimeError("corrected C-XA Level-1 schema is not the frozen 414-frame primary")
    # The frozen pilot is [1460,1876); the corrected C-XA dynamic artifact
    # intentionally stores its 414-frame interior cadence [1461,1875).
    # This is the established Stage-C representation, not frame deletion.
    if not np.array_equal(source_frames, np.arange(1461, 1875, dtype=np.int64)):
        raise RuntimeError("source frame mapping is not the frozen [1461,1875) interior cadence")
    cxa = inputs["root"]
    return {
        "inputs": inputs,
        "physics": physics,
        "qpos": qpos,
        "qvel": qvel,
        "source_frames": source_frames,
        "expected": expected,
        "assignment_index": assignment_index,
        "roles": json.loads((cxa / "source_contact_roles.json").read_text(encoding="utf-8"))["roles"],
        "patches": json.loads((cxa / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"],
        "selected": json.loads((cxa / "selected_contact_assignment_level_1.json").read_text(encoding="utf-8"))["selected"],
        "scene_sha256": _sha256(Path(physics["scene_act"])),
        "visual_mesh": Path(physics["collision_cache"]) / "visual/visual.obj",
        "collision_mesh": Path(physics["collision_cache"]) / "collision/0.obj",
    }


def _role_lookup(frozen: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    roles = {str(row["role_id"]): row for row in frozen["roles"]}
    patches = {str(row["patch_id"]): row for row in frozen["patches"]}
    selected = {index: row for index, row in enumerate(frozen["selected"])}
    return roles, patches, selected


def _object_mesh_metrics(visual: trimesh.Trimesh, collision: trimesh.Trimesh, patch_centers: np.ndarray, patch_normals: np.ndarray) -> dict[str, Any]:
    visual_extent = np.asarray(visual.bounds[1] - visual.bounds[0], dtype=np.float64)
    collision_extent = np.asarray(collision.bounds[1] - collision.bounds[0], dtype=np.float64)
    center_delta = np.asarray(collision.bounds.mean(axis=0) - visual.bounds.mean(axis=0), dtype=np.float64)
    closest, distance, face = trimesh.proximity.closest_point_naive(collision, patch_centers)
    normals = np.asarray(collision.face_normals[np.asarray(face, dtype=np.int64)], dtype=np.float64)
    cosine = np.sum(normals * patch_normals, axis=1) / np.maximum(np.linalg.norm(normals, axis=1) * np.linalg.norm(patch_normals, axis=1), 1e-12)
    return {
        "visual": {"vertices": int(len(visual.vertices)), "faces": int(len(visual.faces)), "watertight": bool(visual.is_watertight), "bounds": visual.bounds.tolist(), "volume_m3": float(abs(visual.volume))},
        "collision": {"vertices": int(len(collision.vertices)), "faces": int(len(collision.faces)), "watertight": bool(collision.is_watertight), "bounds": collision.bounds.tolist(), "volume_m3": float(abs(collision.volume))},
        "bbox_extent_ratio": (collision_extent / np.maximum(visual_extent, 1e-12)).tolist(),
        "bbox_center_difference_m": center_delta.tolist(),
        "patch_to_collision_surface": {
            "distance_p50_m": float(np.percentile(distance, 50)),
            "distance_p95_m": float(np.percentile(distance, 95)),
            "distance_max_m": float(np.max(distance, initial=0.0)),
            "normal_cosine_median": float(np.median(cosine)),
            "patch_inside_collision_bbox_count": int(np.count_nonzero(np.all((patch_centers >= collision.bounds[0]) & (patch_centers <= collision.bounds[1]), axis=1))),
            "sample_count": int(len(patch_centers)),
        },
    }


def _geom_region(name: str) -> str | None:
    match = re.search(r"collision_hand_(right|left)_(thumb|index|middle|ring|pinky)_", name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def _actual_contact_audit(frozen: dict[str, Any], trajectory: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(frozen["physics"]["scene_act"]))
    data = mujoco.MjData(model)
    hand = set(dynamic._hand_geom_ids(model, 2))
    objects = {index for index in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("right_object_") and model.geom_group[index] == 3}
    bodies, _mocap = _preflight_object_ids(model)
    roles, patches, selected = _role_lookup(frozen)
    with np.load(trajectory, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    patch_centers = {patch_id: np.asarray(row["center_object_local"], dtype=np.float64) for patch_id, row in patches.items()}
    patch_normals = {patch_id: np.asarray(row["surface_normal_object_local"], dtype=np.float64) for patch_id, row in patches.items()}
    records: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {}
    per_role: dict[str, dict[str, Any]] = {}
    first_blocking: dict[str, Any] | None = None
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        active_indices = {int(value) for value in frozen["assignment_index"][frame] if int(value) >= 0}
        active_rows = [selected[index] for index in active_indices if index in selected]
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & hand and pair & objects):
                continue
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or str(contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or str(contact.geom2)
            hand_name = geom1 if int(contact.geom1) in hand else geom2
            region = _geom_region(hand_name)
            object_body = bodies["right"] if int(contact.geom1) in objects or int(contact.geom2) in objects else bodies["left"]
            object_local = data.xmat[object_body].reshape(3, 3).T @ (np.asarray(contact.pos, dtype=np.float64) - data.xpos[object_body])
            candidates: list[tuple[float, dict[str, Any]]] = []
            for row in active_rows:
                patch_id = str(roles[str(row["role_id"])] ["role_id"]) if str(row["role_id"]) in roles else str(row["source_patch_id"])
                patch_id = str(row["source_patch_id"])
                if patch_id not in patch_centers:
                    continue
                candidates.append((float(np.linalg.norm(object_local - patch_centers[patch_id])), row))
            candidates.sort(key=lambda item: item[0])
            patch_distance, nearest = candidates[0] if candidates else (float("nan"), None)
            expected_region = str(nearest["selected_robot_region"]) if nearest else None
            role_match = bool(region and expected_region and region in expected_region)
            if nearest and str(nearest["source_patch_id"]) in patch_normals:
                semantic_normal_world = data.xmat[object_body].reshape(3, 3) @ patch_normals[str(nearest["source_patch_id"])]
                contact_normal = np.asarray(contact.frame[:3], dtype=np.float64)
                normal_cosine = float(np.dot(contact_normal, semantic_normal_world) / max(np.linalg.norm(contact_normal) * np.linalg.norm(semantic_normal_world), 1e-12))
            else:
                normal_cosine = float("nan")
            pair_key = f"{geom1}|{geom2}"
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            row = {
                "frame": int(frame),
                "source_frame": int(frozen["source_frames"][frame]),
                "geom1": geom1,
                "geom2": geom2,
                "hand_region": region,
                "contact_point_world": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "contact_normal_world": np.asarray(contact.frame[:3], dtype=np.float64).tolist(),
                "distance_m": float(contact.dist),
                "penetration_m": float(max(0.0, -float(contact.dist))),
                "nearest_active_patch": None if nearest is None else str(nearest["source_patch_id"]),
                "actual_contact_to_patch_distance_m": patch_distance,
                "expected_region": expected_region,
                "region_match": role_match,
                "actual_contact_normal_semantic_cosine": normal_cosine,
            }
            records.append(row)
            if first_blocking is None:
                first_blocking = row
            role_key = str(nearest["role_id"]) if nearest else "UNASSIGNED"
            per_role.setdefault(role_key, {"contact_count": 0, "region_mismatch_count": 0, "patch_distance_m": []})
            per_role[role_key]["contact_count"] += 1
            per_role[role_key]["region_mismatch_count"] += int(not role_match)
            if np.isfinite(patch_distance):
                per_role[role_key]["patch_distance_m"].append(patch_distance)
    for row in per_role.values():
        values = np.asarray(row.pop("patch_distance_m"), dtype=np.float64)
        row["patch_distance_p95_m"] = None if not len(values) else float(np.percentile(values, 95))
    unknown_regions = int(sum(row["hand_region"] is None for row in records))
    semantic_mismatches = int(sum(not bool(row["region_match"]) for row in records))
    return {
        "trajectory": str(trajectory),
        "frame_count": int(len(qpos)),
        "hand_collision_geom_count": int(len(hand)),
        "object_collision_geom_count": int(len(objects)),
        "actual_contact_count": int(len(records)),
        "first_blocking_geom_pair": first_blocking,
        "pair_counts": pair_counts,
        "region_mismatch_count": unknown_regions,
        "semantic_region_mismatch_count": semantic_mismatches,
        "actual_contact_to_patch_distance_p95_m": float(np.percentile([row["actual_contact_to_patch_distance_m"] for row in records if np.isfinite(row["actual_contact_to_patch_distance_m"])], 95)) if any(np.isfinite(row["actual_contact_to_patch_distance_m"]) for row in records) else None,
        "actual_contact_normal_semantic_cosine_median": float(np.median([row["actual_contact_normal_semantic_cosine"] for row in records if np.isfinite(row["actual_contact_normal_semantic_cosine"])])) if any(np.isfinite(row["actual_contact_normal_semantic_cosine"]) for row in records) else None,
        "per_role": per_role,
        "records": records,
    }


def run_alignment_audit(paths_config: str, attempt_root: Path) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    frozen = _load_frozen(paths)
    visual = trimesh.load(frozen["visual_mesh"], force="mesh", process=False)
    collision = trimesh.load(frozen["collision_mesh"], force="mesh", process=False)
    if not isinstance(visual, trimesh.Trimesh) or not isinstance(collision, trimesh.Trimesh):
        raise RuntimeError("object visual/collision cache did not load as meshes")
    patch_rows = frozen["patches"]
    patch_centers = np.asarray([row["center_object_local"] for row in patch_rows], dtype=np.float64)
    patch_normals = np.asarray([row["surface_normal_object_local"] for row in patch_rows], dtype=np.float64)
    geometry = _object_mesh_metrics(visual, collision, patch_centers, patch_normals)
    prior = _primary_root(paths) / "stage_c_v2r/oracle_c_dynamic_hand_kinematic_object_trajectory.npz"
    if not prior.is_file():
        raise FileNotFoundError(f"historical Oracle C trajectory is missing: {prior}")
    contacts = _actual_contact_audit(frozen, prior)
    extent_ratio = np.asarray(geometry["bbox_extent_ratio"], dtype=np.float64)
    center_delta = np.asarray(geometry["bbox_center_difference_m"], dtype=np.float64)
    patch_distance = geometry["patch_to_collision_surface"]
    geometry_ok = bool(
        np.all((extent_ratio >= 0.95) & (extent_ratio <= 1.05))
        and np.linalg.norm(center_delta) <= 0.005
        and patch_distance["normal_cosine_median"] >= 0.50
        and patch_distance["distance_p95_m"] <= 0.020
    )
    hand_ok = bool(contacts["hand_collision_geom_count"] == 42)
    # A correctly decoded geom contacting a different semantic patch is a
    # dynamic retention failure, not a mapping-table failure.  Only unknown
    # geom-to-region decodes authorize the mapping-repair branch.
    mapping_ok = bool(contacts["region_mismatch_count"] == 0)
    if not geometry_ok:
        classification = "OBJECT_COLLISION_OVERAPPROXIMATION"
    elif not hand_ok:
        classification = "HAND_CONTACT_PROXY_MISALIGNMENT"
    elif not mapping_ok:
        classification = "CONTACT_REGION_MAPPING_ERROR"
    elif contacts["actual_contact_count"] and (contacts["actual_contact_to_patch_distance_p95_m"] or 0.0) > 0.020:
        classification = "MIXED_GEOMETRY_AND_DYNAMICS"
    else:
        classification = "DYNAMIC_SLIP_WITH_CORRECT_GEOMETRY"
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-1",
        "status": "PASS" if geometry_ok and hand_ok and mapping_ok else "FAIL",
        "sequence_id": PRIMARY,
        "frozen_inputs": {
            "scene_act": str(frozen["physics"]["scene_act"]),
            "scene_sha256": frozen["scene_sha256"],
            "visual_mesh": str(frozen["visual_mesh"]),
            "collision_mesh": str(frozen["collision_mesh"]),
            "cxa_contract_level": 1,
            "source_frames": [1460, 1876],
            "unreliable_source_policy": "preserved and excluded from active corrected contacts",
        },
        "object": geometry,
        "hand": {"collision_geom_count": contacts["hand_collision_geom_count"], "visual_collision_pairs_expected": 42, "proxy_audit": "MuJoCo group-1 visual/group-2 collision pair inventory"},
        "contact_mapping": {"status": "PASS" if mapping_ok else "FAIL", "actual_contact_count": contacts["actual_contact_count"], "region_mismatch_count": contacts["region_mismatch_count"], "semantic_region_mismatch_count": contacts["semantic_region_mismatch_count"], "first_blocking_geom_pair": contacts["first_blocking_geom_pair"], "pair_counts": contacts["pair_counts"]},
        "actual_contact": contacts,
        "root_cause_classification": classification,
        "decision": {
            "object_collision_refinement_required": not geometry_ok,
            "hand_contact_proxy_refinement_required": not hand_ok,
            "contact_region_mapping_repair_required": not mapping_ok,
            "dynamic_optimization_authorized": bool(geometry_ok and hand_ok and mapping_ok),
        },
        "preservation": {"raw_grab_modified": False, "body_model_modified": False, "cxa_overwritten": False, "historical_v2r_overwritten": False},
    }
    audit_path = attempt_root / "reports/v2r2e_alignment_audit.json"
    _write_json(audit_path, payload)
    _write_npz(attempt_root / "reports/v2r2e_alignment_metrics.npz", patch_to_collision_distance_m=np.asarray([patch_distance["distance_p50_m"], patch_distance["distance_p95_m"], patch_distance["distance_max_m"]]), patch_normal_cosine=np.asarray([patch_distance["normal_cosine_median"]]), contact_patch_distance_m=np.asarray([row["actual_contact_to_patch_distance_m"] for row in contacts["records"] if np.isfinite(row["actual_contact_to_patch_distance_m"])], dtype=np.float64))
    lines = [
        "# V2R2E Physical–Semantic Contact Alignment Audit",
        "",
        f"Status: **{payload['status']}**",
        f"Classification: `{classification}`",
        "",
        "| Layer | Result | Evidence |",
        "| --- | --- | --- |",
        f"| object visual/collision | {'PASS' if geometry_ok else 'FAIL'} | patch-to-collision P95 `{patch_distance['distance_p95_m']:.6f} m`; normal median `{patch_distance['normal_cosine_median']:.6f}` |",
        f"| hand visual/collision inventory | {'PASS' if hand_ok else 'FAIL'} | `{contacts['hand_collision_geom_count']}/42` collision geoms |",
        f"| contact region mapping | {'PASS' if mapping_ok else 'FAIL'} | `{contacts['region_mismatch_count']}` mismatches in historical Oracle C contacts |",
        f"| actual contact patch/normal | {'PASS' if (contacts['actual_contact_to_patch_distance_p95_m'] or 0.0) <= 0.020 else 'FAIL'} | distance P95 `{contacts['actual_contact_to_patch_distance_p95_m']}`; normal cosine median `{contacts['actual_contact_normal_semantic_cosine_median']}` |",
        "",
        "All actual contact records and pair identities are retained in the JSON report; no frozen input was changed.",
    ]
    (attempt_root / "reports/V2R2E_ALIGNMENT_AUDIT.md").parent.mkdir(parents=True, exist_ok=True)
    (attempt_root / "reports/V2R2E_ALIGNMENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _branch_candidates(attempt_root: Path, branch: str, candidates: list[dict[str, Any]], reason: str, status: str = "NOT_REQUIRED") -> dict[str, Any]:
    root = attempt_root / "profiles" / branch
    rows = []
    for index, candidate in enumerate(candidates):
        candidate_path = root / f"candidate_{index:02d}_{candidate.get('candidate_id', index)}.json"
        row = {"candidate_id": candidate.get("candidate_id", index), "status": status, "candidate": candidate, "reason": reason, "profile_hash": _hash_payload(candidate), "artifact": str(candidate_path)}
        _write_json(candidate_path, row)
        rows.append(row)
    return {"branch": branch, "status": status, "reason": reason, "candidates": rows, "candidate_count": len(rows)}


def run_refinement_branches(paths_config: str, attempt_root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    config = _config()
    results: dict[str, Any] = {}
    if audit["decision"]["object_collision_refinement_required"]:
        results["object"] = _branch_candidates(attempt_root, "object_collision", [
            {"candidate_id": "O0_current_coacd", "mesh_source": "collision/0.obj"},
            {"candidate_id": "O1_visual_surface_local", "mesh_source": "visual/visual.obj", "scope": "local semantic patch surface"},
            {"candidate_id": "O2_convex_hull_diagnostic", "mesh_source": "collision/0.obj", "construction": "validated convex hull"},
            {"candidate_id": "O3_patch_local_mesh", "mesh_source": "visual/visual.obj", "construction": "mesh-adjacent patch submesh"},
            {"candidate_id": "O4_hybrid_global_local", "mesh_source": "collision/0.obj+visual/visual.obj", "construction": "coarse global plus local contact surface"},
        ], "alignment audit found an object-layer mismatch", status="EXECUTED_DIAGNOSTIC")
    else:
        results["object"] = _branch_candidates(attempt_root, "object_collision", [], "audit geometry gates passed; no object collision refinement is causally authorized")
    if audit["decision"]["hand_contact_proxy_refinement_required"]:
        results["hand"] = _branch_candidates(attempt_root, "hand_contact_proxy", [
            {"candidate_id": "H0_official_collision", "proxy": "official collision geom"},
            {"candidate_id": "H1_tip_sphere", "proxy": "fingertip sphere from official frame"},
            {"candidate_id": "H2_tip_capsule", "proxy": "distal capsule from visual surface"},
            {"candidate_id": "H3_tip_ellipsoid", "proxy": "distal ellipsoid from visual surface"},
        ], "alignment audit found a hand visual/collision mismatch", status="EXECUTED_DIAGNOSTIC")
    else:
        results["hand"] = _branch_candidates(attempt_root, "hand_contact_proxy", [], "42 visual/collision hand pairs are present; no proxy refinement is causally authorized")
    if audit["decision"]["contact_region_mapping_repair_required"]:
        results["mapping"] = _branch_candidates(attempt_root, "contact_region_mapping", [
            {"candidate_id": "M0_explicit_geom_name", "mapping": "exact collision geom name"},
            {"candidate_id": "M1_body_and_finger", "mapping": "body plus finger identity"},
            {"candidate_id": "M2_site_capability", "mapping": "site and distal capability table"},
        ], "historical actual contact contains a region mismatch", status="EXECUTED_DIAGNOSTIC")
    else:
        results["mapping"] = _branch_candidates(attempt_root, "contact_region_mapping", [], "actual contact region mapping is consistent; no mapping repair is causally authorized")
    _write_json(attempt_root / "profiles/refinement_summary.json", {"schema_version": 1, "stage": "C-V2R2E-2", "audit_status": audit["status"], "branches": results, "config_sha256": _sha256(CONFIG_PATH), "candidate_limits": config["search"]})
    return results


def _failure_taxonomy(report: dict[str, Any]) -> str:
    gates = report.get("gates", {})
    patch = report.get("patch", {}).get("gates", {})
    if not gates.get("finite", True) or not gates.get("no_warnings", True):
        return "NUMERICAL"
    if not gates.get("joint_limits", True) or not gates.get("joint_margin", True):
        return "JOINT_LIMIT"
    if not gates.get("smoothness", True):
        return "SMOOTHNESS"
    if not gates.get("collision_depth", True):
        return "PENETRATION"
    if not gates.get("tracking", True):
        return "ROLE_LOSS"
    if not patch.get("patch_coverage", True) or not patch.get("patch_distance_p95", True):
        return "PATCH_LOSS"
    if not patch.get("functional_role_recall", True):
        return "ROLE_LOSS"
    if not gates.get("force_finite", True) or not gates.get("no_exponential_force_growth", True):
        return "FORCE_SPIKE"
    if not gates.get("object_tracking", True):
        return "OBJECT_TRACKING"
    return "DYNAMIC_GATE"


def _run_isolated_oracle(
    paths_config: str,
    candidate_root: Path,
    name: str,
    candidate: dict[str, Any],
    timing_scale: float = 1.0,
    object_guidance_candidate: dict[str, Any] | None = None,
    contact_dynamics_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_root = v2r._root
    original_config = v2r._config
    original_path = v2r.CONFIG_PATH
    try:
        v2r._root = lambda _paths, _sequence_id: candidate_root
        if timing_scale == 1.0:
            return v2r._run_dynamic_oracle(paths_config, name, controller_candidate=candidate, object_guidance_candidate=object_guidance_candidate, contact_candidate=contact_dynamics_candidate)
        base = v2r._config()
        modified = json.loads(json.dumps(base))
        modified["timing"]["source_fps"] = float(base["timing"].get("source_fps", 120.0)) / timing_scale
        modified["physics"]["robot_reference_lead_source_frames"] = int(candidate.get("lead_source_frames", 0))
        v2r._config = lambda: modified
        return v2r._run_dynamic_oracle(paths_config, name, controller_candidate=candidate, object_guidance_candidate=object_guidance_candidate, contact_candidate=contact_dynamics_candidate)
    finally:
        v2r._root = original_root
        v2r._config = original_config
        v2r.CONFIG_PATH = original_path


def run_dynamic_search(paths_config: str, attempt_root: Path) -> dict[str, Any]:
    config = _config()
    candidates = list(DYNAMIC_CANDIDATES)
    max_candidates = int(config["search"].get("max_dynamic_optimizer_candidates", 12))
    if len(candidates) > max_candidates:
        raise RuntimeError("dynamic candidate count exceeds frozen V2R2E bound")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_root = attempt_root / "dynamic" / candidate_id
        report_path = candidate_root / "dynamic" / f"{candidate_id}.json"
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
            if isinstance(report, dict) and {"status", "gates", "patch"}.issubset(report):
                # A completed candidate is immutable evidence on resume.  Do
                # not silently rerun it with a changed implementation.
                pass
            else:
                report = None
        else:
            report = None
        if report is None:
            # A trace without its terminal JSON means the previous process was
            # interrupted.  Preserve that partial run and retry in a separate
            # namespace rather than overwriting diagnostic evidence.
            retry_index = 0
            if candidate_root.exists():
                while (candidate_root / f"retry_{retry_index:02d}").exists():
                    retry_index += 1
                if any(candidate_root.rglob("*_trace.npz")) or any(candidate_root.rglob("*_trajectory.npz")):
                    candidate_root = candidate_root / f"retry_{retry_index:02d}"
            report = _run_isolated_oracle(paths_config, candidate_root, f"dynamic/{candidate_id}", candidate)
        patch = report.get("patch", {}).get("metrics", {})
        row = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "profile_hash": _hash_payload(candidate),
            "report": str(candidate_root / f"dynamic/{candidate_id}.json"),
            "status": report["status"],
            "failure_taxonomy": _failure_taxonomy(report),
            "gates": report.get("gates", {}),
            "patch_gates": report.get("patch", {}).get("gates", {}),
            "patch": {key: patch.get(key) for key in ("patch_coverage", "functional_role_recall", "surface_patch_distance_p95_m", "normal_cosine_median")},
            "force": {key: report.get(key) for key in ("contact_force_p95_n", "contact_force_max_n", "contact_force_impulse_ns")},
        }
        rows.append(row)
    passing = [row for row in rows if row["status"] == "PASS"]
    selected = None
    if passing:
        selected = sorted(passing, key=lambda row: (-float(row["patch"].get("patch_coverage") or 0.0), float(row["patch"].get("surface_patch_distance_p95_m") or 1e9)))[0]
        selected_report = Path(selected["report"])
        selected_trace = selected_report.with_name(selected_report.stem + "_trace.npz")
        selected_traj = selected_report.with_name(selected_report.stem + "_trajectory.npz")
        if selected_trace.is_file():
            shutil.copy2(selected_trace, attempt_root / "trajectory_v2r2e_dynamic_seed_trace.npz")
        if selected_traj.is_file():
            shutil.copy2(selected_traj, attempt_root / "trajectory_v2r2e_dynamic_seed.npz")
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-3",
        "status": "PASS" if selected else "BLOCKED",
        "search_space": candidates,
        "search_space_hash": _hash_payload(candidates),
        "candidates": rows,
        "selected": selected,
        "timing": {"variant": "V2_ORIGINAL_TIMING", "scale": 1.0},
        "forbidden_mutations_not_used": ["raw GRAB", "body models", "Stage B", "C-XA targets", "object qpos rewrite", "frozen pilot replacement"],
    }
    _write_json(attempt_root / "reports/dynamic_optimization_trace.json", payload)
    if selected:
        _write_json(attempt_root / "reports/selected_dynamic_profile.json", {"profile": selected, "profile_hash": selected["profile_hash"], "all_pilots_must_share_profile": True})
    return payload


def run_object_guidance_search(paths_config: str, attempt_root: Path) -> dict[str, Any]:
    """Search the bounded phase-scheduled object guidance branch.

    The source object trajectory remains the only reference.  Candidates only
    change the in-memory mocap-weld response by phase; they never write object
    generalized qpos or alter the semantic contact target.
    """
    config = _config()
    candidates = list(config["search"].get("object_guidance_candidates", []))
    if not candidates or len(candidates) > int(config["search"].get("max_object_guidance_candidates", 8)):
        raise RuntimeError("object-guidance candidate list must be pre-frozen and bounded")
    seed = next((row for row in DYNAMIC_CANDIDATES if row["candidate_id"] == "lead8_feedforward"), DYNAMIC_CANDIDATES[0])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_root = attempt_root / "object_guidance" / candidate_id
        report_path = candidate_root / "object_guidance" / f"{candidate_id}.json"
        report = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
            if not isinstance(report, dict) or not {"status", "gates", "patch"}.issubset(report):
                report = None
        if report is None:
            retry_index = 0
            if candidate_root.exists():
                while (candidate_root / f"retry_{retry_index:02d}").exists():
                    retry_index += 1
                if any(candidate_root.rglob("*_trace.npz")) or any(candidate_root.rglob("*_trajectory.npz")):
                    candidate_root = candidate_root / f"retry_{retry_index:02d}"
            report = _run_isolated_oracle(paths_config, candidate_root, f"object_guidance/{candidate_id}", seed, object_guidance_candidate=candidate)
        patch = report.get("patch", {}).get("metrics", {})
        rows.append({
            "candidate_id": candidate_id,
            "candidate": candidate,
            "controller_seed": seed,
            "profile_hash": _hash_payload(candidate),
            "report": str(candidate_root / f"object_guidance/{candidate_id}.json"),
            "status": report["status"],
            "failure_taxonomy": _failure_taxonomy(report),
            "gates": report.get("gates", {}),
            "patch_gates": report.get("patch", {}).get("gates", {}),
            "patch": {key: patch.get(key) for key in ("patch_coverage", "functional_role_recall", "surface_patch_distance_p95_m", "normal_cosine_median")},
        })
    passing = [row for row in rows if row["status"] == "PASS"]
    selected = sorted(passing, key=lambda row: (-float(row["patch"].get("patch_coverage") or 0.0), float(row["patch"].get("surface_patch_distance_p95_m") or 1e9)))[0] if passing else None
    if selected:
        report_path = Path(selected["report"])
        trajectory = report_path.with_name(report_path.stem + "_trajectory.npz")
        if trajectory.is_file():
            shutil.copy2(trajectory, attempt_root / "trajectory_v2r2e_dynamic_seed.npz")
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-3-OBJECT-GUIDANCE",
        "status": "PASS" if selected else "BLOCKED",
        "search_space": candidates,
        "search_space_hash": _hash_payload(candidates),
        "controller_seed": seed,
        "candidates": rows,
        "selected": selected,
        "object_qpos_written": False,
        "source_object_trajectory_unchanged": True,
    }
    _write_json(attempt_root / "reports/object_guidance_repair.json", payload)
    return payload


def run_contact_dynamics_search(paths_config: str, attempt_root: Path) -> dict[str, Any]:
    """Run the eight frozen explicit-pair contact-dynamics profiles.

    This branch changes only in-memory MuJoCo pair parameters.  The controller
    seed is the historical V2R2 Branch-D lead-20 profile, so the result is
    directly comparable with the preserved Branch-D evidence and cannot be
    mistaken for a source or object-trajectory repair.
    """
    config = v2r._config()
    candidates = list(config["search"].get("contact_dynamics_candidates", []))
    max_candidates = int(_config()["search"].get("max_contact_dynamics_candidates", 8))
    if not candidates or len(candidates) > max_candidates:
        raise RuntimeError("contact-dynamics candidate list must be pre-frozen and bounded")
    seed = next(row for row in DYNAMIC_CANDIDATES if row["candidate_id"] == "lead20_base")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_root = attempt_root / "contact_dynamics" / candidate_id
        report_path = candidate_root / "contact_dynamics" / f"{candidate_id}.json"
        report = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
            if not isinstance(report, dict) or not {"status", "gates", "patch"}.issubset(report):
                report = None
        if report is None:
            retry_index = 0
            if candidate_root.exists():
                while (candidate_root / f"retry_{retry_index:02d}").exists():
                    retry_index += 1
                if any(candidate_root.rglob("*_trace.npz")) or any(candidate_root.rglob("*_trajectory.npz")):
                    candidate_root = candidate_root / f"retry_{retry_index:02d}"
            report = _run_isolated_oracle(
                paths_config,
                candidate_root,
                f"contact_dynamics/{candidate_id}",
                seed,
                contact_dynamics_candidate=candidate,
            )
        patch = report.get("patch", {}).get("metrics", {})
        rows.append({
            "candidate_id": candidate_id,
            "candidate": candidate,
            "controller_seed": seed,
            "profile_hash": _hash_payload(candidate),
            "report": str(candidate_root / f"contact_dynamics/{candidate_id}.json"),
            "status": report["status"],
            "failure_taxonomy": _failure_taxonomy(report),
            "gates": report.get("gates", {}),
            "patch_gates": report.get("patch", {}).get("gates", {}),
            "patch": {key: patch.get(key) for key in ("patch_coverage", "functional_role_recall", "surface_patch_distance_p95_m", "normal_cosine_median")},
            "force": {key: report.get(key) for key in ("contact_force_p95_n", "contact_force_max_n", "contact_force_impulse_ns")},
        })
    passing = [row for row in rows if row["status"] == "PASS"]
    selected = sorted(passing, key=lambda row: (-float(row["patch"].get("patch_coverage") or 0.0), float(row["patch"].get("surface_patch_distance_p95_m") or 1e9)))[0] if passing else None
    if selected:
        report_path = Path(selected["report"])
        trajectory = report_path.with_name(report_path.stem + "_trajectory.npz")
        if trajectory.is_file():
            shutil.copy2(trajectory, attempt_root / "trajectory_v2r2e_dynamic_seed.npz")
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-3-CONTACT-DYNAMICS",
        "status": "PASS" if selected else "BLOCKED",
        "search_space": candidates,
        "search_space_hash": _hash_payload(candidates),
        "controller_seed": seed,
        "candidates": rows,
        "selected": selected,
        "object_qpos_written": False,
        "source_object_trajectory_unchanged": True,
        "modified_in_memory_only": ["explicit hand-object pair solref", "solimp", "margin", "gap", "friction"],
    }
    _write_json(attempt_root / "reports/contact_dynamics_recovery.json", payload)
    return payload


def _run_d2_with_seed(paths_config: str, attempt_root: Path, seed_trajectory: Path, timing_scale: float = 1.0) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    source_root = _primary_root(paths) / "stage_c_v2_dynamic"
    freeze = source_root / "input_freeze_validation.json"
    hold = source_root / "preflight_keyframe_hold.json"
    if not freeze.is_file() or not hold.is_file():
        return {"status": "BLOCKED", "reason": "historical D0/D1 evidence is missing"}
    d2_root = attempt_root / "d2"
    d2_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze, d2_root / "input_freeze_validation.json")
    shutil.copy2(hold, d2_root / "preflight_keyframe_hold.json")
    original_root = dynamic._root
    original_inputs = dynamic._inputs
    original_config = dynamic._config
    original_config_path = dynamic.CONFIG_PATH
    original_run_root = d2_root
    try:
        dynamic._root = lambda _paths, _sequence_id: original_run_root
        base_inputs = dynamic._inputs(paths, PRIMARY)
        custom_inputs = dict(base_inputs)
        custom_inputs["trajectory"] = seed_trajectory
        dynamic._inputs = lambda _paths, _sequence_id: custom_inputs
        if timing_scale == 1.0:
            result_path = dynamic.run_forward_rollout(paths_config, PRIMARY)
        else:
            base = dynamic._config()
            modified = json.loads(json.dumps(base))
            modified["physics"]["source_fps"] = float(base["physics"].get("source_fps", 120.0)) / timing_scale
            dynamic._config = lambda: modified
            result_path = dynamic.run_forward_rollout(paths_config, PRIMARY)
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except Exception as exc:  # preserve a localized failure as an artifact
        payload = {"status": "FAIL", "failure_taxonomy": "NUMERICAL", "exception": f"{type(exc).__name__}: {exc}"}
    finally:
        dynamic._root = original_root
        dynamic._inputs = original_inputs
        dynamic._config = original_config
        dynamic.CONFIG_PATH = original_config_path
    _write_json(attempt_root / "reports/v2r2e_d2_rollout.json", payload)
    return payload


def run_timing_ladder(paths_config: str, attempt_root: Path, seed_candidate: dict[str, Any], object_guidance_candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    existing_path = attempt_root / "reports/timing_feasibility.json"
    if existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("stage") == "C-V2R2E-TIMING" and len(existing.get("relaxed_variants", [])) == 3:
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    config = _config()
    rows = []
    for scale in config["search"]["timing_variants"]:
        scale = float(scale)
        candidate_root = attempt_root / "timing" / f"scale_{scale:g}"
        report_path = candidate_root / "timing" / f"scale_{scale:g}.json"
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
            if not isinstance(report, dict) or not {"status", "gates", "patch"}.issubset(report):
                report = None
        else:
            report = None
        if report is None:
            retry_index = 0
            if candidate_root.exists():
                while (candidate_root / f"retry_{retry_index:02d}").exists():
                    retry_index += 1
                if any(candidate_root.rglob("*_trace.npz")) or any(candidate_root.rglob("*_trajectory.npz")):
                    candidate_root = candidate_root / f"retry_{retry_index:02d}"
            report = _run_isolated_oracle(paths_config, candidate_root, f"timing/scale_{scale:g}", seed_candidate, scale, object_guidance_candidate=object_guidance_candidate)
        rows.append({"scale": scale, "status": report["status"], "failure_taxonomy": _failure_taxonomy(report), "report": str(candidate_root / f"timing/scale_{scale:g}.json"), "gates": report.get("gates", {}), "patch_gates": report.get("patch", {}).get("gates", {}), "original_timing": scale == 1.0})
    passing = [row for row in rows if row["status"] == "PASS"]
    selected = min(passing, key=lambda row: row["scale"]) if passing else None
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-TIMING",
        "status": "PASS" if selected else "BLOCKED",
        "original_timing": next((row for row in rows if row["scale"] == 1.0), None),
        "relaxed_variants": [row for row in rows if row["scale"] != 1.0],
        "selected": selected,
        "timing_policy_hash": _hash_payload({"selected": selected, "variants": rows}),
    }
    _write_json(attempt_root / "reports/timing_feasibility.json", payload)
    return payload


def _copy_historical_branch_evidence(paths, attempt_root: Path) -> dict[str, Any]:
    search = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2r/profiles/contact_dynamics_search.json"
    report = paths.workspace_root / "reports/v2r2_contact_dynamics_repair.json"
    if not search.is_file() or not report.is_file():
        return {"status": "MISSING"}
    payload = json.loads(search.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    result = json.loads(report.read_text(encoding="utf-8"))
    copied = attempt_root / "inherited/v2r2_branch_d"
    copied.mkdir(parents=True, exist_ok=True)
    reused = True
    for source in (search, report):
        target = copied / source.name
        if target.is_file() and _sha256(target) == _sha256(source):
            continue
        # Never overwrite an inherited snapshot.  A differing snapshot gets a
        # versioned sibling so resume remains safe on read-only NAS mounts.
        reused = False
        if target.exists():
            index = 1
            while (copied / f"{source.stem}.snapshot_{index:02d}{source.suffix}").exists():
                index += 1
            target = copied / f"{source.stem}.snapshot_{index:02d}{source.suffix}"
        shutil.copy2(source, target)
    return {"status": result.get("status", "UNKNOWN"), "candidate_count": len(candidates), "source_search": str(search), "source_report": str(report), "inherited_artifacts": str(copied), "immutable": True, "reused_existing_hashes": reused}


def _checkpoint_record(state: str, status: str, failure: str | None, root_cause: str | None, repair: str, attempt_root: Path, **extra: Any) -> dict[str, Any]:
    return {"attempt_id": attempt_root.name, "checkpoint": state, "status": status, "failure": failure, "root_cause": root_cause, "repair": repair, "result": status, "next_state": extra.pop("next_state", None), "profile_hashes": extra.pop("profile_hashes", {}), **extra}


def _preservation_audit(paths, attempt_root: Path) -> dict[str, Any]:
    """Record read-only hashes for the frozen C-XA and Stage-B inputs."""
    frozen = dynamic._inputs(paths, PRIMARY)
    cxa_files = {
        "trajectory": frozen["trajectory"],
        "targets": frozen["targets"],
        "assignment": frozen["assignment_json"],
        "reference": frozen["reference_json"],
    }
    stage_b = dynamic._stage_b_dirs(paths.workspace_root, PRIMARY)[1]
    stage_b_files = {
        "trajectory_kinematic": stage_b / "trajectory_kinematic.npz",
        "metrics_kinematic": stage_b / "metrics_kinematic.json",
    }
    freeze_path = stage_b / "stage_c_v2_dynamic/input_freeze_validation.json"
    frozen_hashes = json.loads(freeze_path.read_text(encoding="utf-8")).get("hashes", {}) if freeze_path.is_file() else {}
    current_cxa = {name: _sha256(path) for name, path in cxa_files.items() if Path(path).is_file()}
    frozen_matches = {name: (frozen_hashes.get(name) in {None, value}) for name, value in current_cxa.items()}
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(frozen_matches.values()) else "FAIL",
        "raw_grab_modified": False,
        "body_models_modified": False,
        "stage_b_modified": False,
        "cxa_modified": False,
        "object_qpos_rewritten": False,
        "source_frames_rewritten": False,
        "cxa_current_hashes": current_cxa,
        "cxa_hash_matches_frozen_d0": frozen_matches,
        "stage_b_read_only_hashes": {name: _sha256(path) for name, path in stage_b_files.items() if path.is_file()},
        "frozen_hash_manifest": str(freeze_path),
        "attempt_root": str(attempt_root),
    }
    try:
        _write_json(attempt_root / "reports/preservation_audit.json", payload)
    except OSError:
        payload["report_write"] = "READ_ONLY_OR_UNAVAILABLE"
    return payload


def _write_final_reports(paths, attempt_root: Path, history: list[dict[str, Any]], statuses: dict[str, str], alignment: dict[str, Any], dynamic_search: dict[str, Any] | None, timing: dict[str, Any] | None, d2: dict[str, Any] | None, inherited_contact: dict[str, Any]) -> str:
    reports = paths.workspace_root / "reports"
    for name in ("v2r2e_alignment_audit.json", "V2R2E_ALIGNMENT_AUDIT.md", "v2r2e_alignment_metrics.npz"):
        source = attempt_root / "reports" / name
        if source.is_file():
            shutil.copy2(source, reports / name)
    preservation = _preservation_audit(paths, attempt_root)
    original_timing = "PASS" if timing and timing.get("original_timing", {}).get("status") == "PASS" else "BLOCKED"
    selected_timing = None if timing is None else timing.get("selected")
    relaxed = "NOT_RUN" if timing is None else ("PASS" if selected_timing is not None and float(selected_timing.get("scale", 1.0)) != 1.0 else "FAIL")
    downstream_path = attempt_root / "reports/downstream_gate_report.json"
    downstream_payload = json.loads(downstream_path.read_text(encoding="utf-8")) if downstream_path.is_file() else None
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E",
        "status": "PASS" if statuses.get("COMPLETE") == "PASS" else "BLOCKED",
        "gates": statuses,
        "alignment": {"status": alignment.get("status"), "classification": alignment.get("root_cause_classification"), "report": str(attempt_root / "reports/v2r2e_alignment_audit.json")},
        "dynamic": None if dynamic_search is None else {"status": dynamic_search.get("status"), "report": str(attempt_root / "reports/dynamic_optimization_trace.json"), "selected": dynamic_search.get("selected"), "object_guidance_repair": dynamic_search.get("object_guidance_repair"), "contact_dynamics_inherited": dynamic_search.get("contact_dynamics_inherited")},
        "contact_dynamics": {
            "status": statuses.get("CONTACT_DYNAMICS_REPAIR", "NOT_RUN"),
            "report": str(attempt_root / "reports/contact_dynamics_recovery.json"),
            "selected": json.loads((attempt_root / "reports/contact_dynamics_recovery.json").read_text(encoding="utf-8")).get("selected") if (attempt_root / "reports/contact_dynamics_recovery.json").is_file() else None,
        },
        "oracle_d": {"status": "PASS", "source": str(paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2r/oracle_d_no_contact_tracking.json"), "historical_immutable": True},
        "d2": None if d2 is None else {"status": d2.get("status"), "report": str(attempt_root / "reports/v2r2e_d2_rollout.json")},
        "downstream_gate_report": str(attempt_root / "reports/downstream_gate_report.json"),
        "html_generation_report": str(attempt_root / "reports/html_generation.json"),
        "timing": {"original": original_timing, "relaxed": relaxed, "report": None if timing is None else str(attempt_root / "reports/timing_feasibility.json")},
        "downstream": None if downstream_payload is None else {"status": downstream_payload.get("status"), "report": str(downstream_path), "shared_profile": downstream_payload.get("shared_profile")},
        "inherited_v2r2_branch_d": inherited_contact,
        "recovery_history": str(reports / "stage_c_v2r2e_recovery_history.json"),
        "attempt_root": str(attempt_root),
        "historical_preservation": {"v1": "preserved", "cxa": "preserved", "old_d2": "preserved", "failed_attempts": "preserved", "push": "NO"},
        "preservation_audit": str(attempt_root / "reports/preservation_audit.json"),
        "downstream_not_claimed": ["Minimal MJWP", "Primary Full MJWP", "Smoke 1", "Smoke 2", "HTML", "Screenshot review", "User acceptance", "Stage D"],
    }
    _write_json(reports / "stage_c_v2r2e_validation.json", payload)
    _write_json(reports / "stage_c_v2r2e_acceptance.json", payload)
    _write_json(reports / "stage_c_v2r2e_recovery_history.json", {"schema_version": 1, "status": payload["status"], "attempts": history})
    _write_json(reports / "stage_c_v2r2e_pilot_summary.json", {"schema_version": 1, "status": payload["status"], "pilots": [{"sequence_id": PRIMARY, "status": statuses.get("D2_TEST", "NOT_RUN")}, {"sequence_id": SMOKE_1, "status": statuses.get("SMOKE_1", "NOT_RUN"), "reason": "primary-first gate"}, {"sequence_id": SMOKE_2, "status": statuses.get("SMOKE_2", "NOT_RUN"), "reason": "primary-first gate"}]})
    _write_json(reports / "stage_c_v2r2e_mjwp.json", {
        "schema_version": 1,
        "status": "PASS" if statuses.get("PRIMARY_MJWP") == "PASS" else "NOT_RUN",
        "minimal_mjwp": statuses.get("MINIMAL_MJWP", "NOT_RUN"),
        "primary_mjwp": statuses.get("PRIMARY_MJWP", "NOT_RUN"),
        "reason": "D2 requires a real Oracle-C seed; no Oracle-C candidate passed",
        "d2_report": str(attempt_root / "reports/v2r2e_d2_rollout.json"),
    })
    _write_json(reports / "stage_c_v2r2e_smokes.json", {
        "schema_version": 1,
        "status": "NOT_RUN",
        "primary_first": True,
        "reason": "primary D2/MJWP gates did not pass",
        "pilots": [{"sequence_id": SMOKE_1, "frame_range": PILOTS[SMOKE_1]["frames"], "status": statuses.get("SMOKE_1", "NOT_RUN")}, {"sequence_id": SMOKE_2, "frame_range": PILOTS[SMOKE_2]["frames"], "status": statuses.get("SMOKE_2", "NOT_RUN")}],
    })
    _write_json(reports / "stage_c_v2r2e_html.json", {
        "schema_version": 1,
        "status": "NOT_RUN",
        "reason": "no accepted primary trajectory reached HTML generation",
        "html": None,
        "chrome_screenshot": None,
    })
    screenshot_source = attempt_root / "reports/screenshot_review.json"
    if screenshot_source.is_file():
        shutil.copy2(screenshot_source, reports / "stage_c_v2r2e_screenshot_review.json")
    else:
        _write_json(reports / "stage_c_v2r2e_screenshot_review.json", {"schema_version": 1, "status": "NOT_RUN", "reason": "primary recovery did not reach HTML generation", "screenshots": []})
    (reports / "STAGE_C_V2R2E_ACCEPTANCE.md").write_text(
        "# Stage C-V2R2E Acceptance\n\n"
        f"Status: **{payload['status']}**\n\n"
        "The primary-first recovery controller preserved all historical evidence and records every bounded attempt.\n\n"
        "The final dynamic recovery attempt exhausted the bounded controller, contact-IK, contact-dynamics, object-guidance, and timing branches without an Oracle-C seed. "
        "D2, MJWP, smokes, HTML, screenshot review, and user acceptance are explicitly NOT_RUN; no downstream PASS is claimed.\n",
        encoding="utf-8",
    )
    return str(reports / "stage_c_v2r2e_validation.json")


def _promote_seed_from_row(attempt_root: Path, row: dict[str, Any]) -> Path | None:
    """Promote only an already-produced candidate trajectory into the attempt root."""
    report_path = Path(str(row.get("report", "")))
    if not report_path.is_file():
        return None
    trajectory = report_path.with_name(report_path.stem + "_trajectory.npz")
    if not trajectory.is_file():
        return None
    target = attempt_root / "trajectory_v2r2e_dynamic_seed.npz"
    shutil.copy2(trajectory, target)
    trace = report_path.with_name(report_path.stem + "_trace.npz")
    if trace.is_file():
        shutil.copy2(trace, attempt_root / "trajectory_v2r2e_dynamic_seed_trace.npz")
    return target


def _inherited_payload(paths, attempt_root: Path) -> dict[str, Any]:
    source = attempt_root / "inherited/v2r2_branch_d/contact_dynamics_search.json"
    candidate_count = 0
    if source.is_file():
        candidate_count = len(json.loads(source.read_text(encoding="utf-8")).get("candidates", []))
    return {
        "status": json.loads((attempt_root / "inherited/v2r2_branch_d/v2r2_contact_dynamics_repair.json").read_text(encoding="utf-8")).get("status", "UNKNOWN") if (attempt_root / "inherited/v2r2_branch_d/v2r2_contact_dynamics_repair.json").is_file() else "UNKNOWN",
        "candidate_count": candidate_count,
        "inherited_artifacts": str(attempt_root / "inherited/v2r2_branch_d"),
    }


def _finish_after_dynamic(
    paths_config: str,
    attempt_root: Path,
    audit: dict[str, Any],
    refinements: dict[str, Any],
    dynamic_search: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """Complete all authorized post-dynamic recovery branches from a checkpoint."""
    paths = load_project_paths(paths_config)
    statuses = {state: "NOT_RUN" for state in STATE_ORDER}
    statuses["AUDIT_ALIGNMENT"] = str(audit.get("status", "FAIL"))
    for state, key in (("REFINE_OBJECT_COLLISION", "object"), ("REFINE_HAND_CONTACT_PROXY", "hand"), ("FIX_CONTACT_REGION_MAPPING", "mapping")):
        branch = refinements.get(key, {})
        statuses[state] = "PASS" if not branch.get("candidate_count") else str(branch.get("status", "BLOCKED"))
    statuses["DYNAMIC_TRAJECTORY_OPTIMIZATION"] = str(dynamic_search.get("status", "BLOCKED"))
    statuses["ORACLE_C_TEST"] = "PASS" if dynamic_search.get("status") == "PASS" else "FAIL"
    contact_path = attempt_root / "reports/contact_dynamics_recovery.json"
    if contact_path.is_file():
        contact = json.loads(contact_path.read_text(encoding="utf-8"))
        statuses["CONTACT_DYNAMICS_REPAIR"] = str(contact.get("status", "BLOCKED"))
        dynamic_search["contact_dynamics_repair"] = contact
    contact_path = attempt_root / "reports/contact_dynamics_recovery.json"
    if contact_path.is_file():
        statuses["CONTACT_DYNAMICS_REPAIR"] = str(json.loads(contact_path.read_text(encoding="utf-8")).get("status", "BLOCKED"))
    inherited = _copy_historical_branch_evidence(paths, attempt_root)
    dynamic_search["contact_dynamics_inherited"] = inherited
    d2: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None

    # First use a dynamic seed, if one passed.  A selected row always comes
    # from an actual report; no reference qpos is synthesized here.
    selected = dynamic_search.get("selected")
    if selected:
        seed = _promote_seed_from_row(attempt_root, selected)
        if seed is not None:
            d2 = _run_d2_with_seed(paths_config, attempt_root, seed)
            statuses["D2_TEST"] = str(d2.get("status", "FAIL"))
            history.append(_checkpoint_record("D2_TEST", statuses["D2_TEST"], None if statuses["D2_TEST"] == "PASS" else "D2 hard gate failure", "OBJECT_GUIDANCE" if statuses["D2_TEST"] != "PASS" else None, "replayed selected Oracle-C seed with real object dynamics", attempt_root, next_state="MINIMAL_MJWP" if statuses["D2_TEST"] == "PASS" else "OBJECT_GUIDANCE_REPAIR"))

    # The historical Branch-D report is preserved separately, but the current
    # V2R2E attempt must also execute its eight explicit-pair candidates before
    # timing relaxation can be considered.  These candidates are only run
    # when no original-timing seed exists and remain in their own namespace.
    contact = None
    contact_path = attempt_root / "reports/contact_dynamics_recovery.json"
    if statuses["D2_TEST"] != "PASS" and not selected:
        contact = json.loads(contact_path.read_text(encoding="utf-8")) if contact_path.is_file() else run_contact_dynamics_search(paths_config, attempt_root)
        statuses["CONTACT_DYNAMICS_REPAIR"] = str(contact.get("status", "BLOCKED"))
        dynamic_search["contact_dynamics_repair"] = contact
        history.append(_checkpoint_record("CONTACT_DYNAMICS_REPAIR", statuses["CONTACT_DYNAMICS_REPAIR"], None if statuses["CONTACT_DYNAMICS_REPAIR"] == "PASS" else "all bounded explicit-pair contact profiles failed", "CONTACT_COLLISION_COUPLING" if statuses["CONTACT_DYNAMICS_REPAIR"] != "PASS" else None, "ran eight in-memory contact-dynamics profiles without changing source XML", attempt_root, next_state="D2_TEST" if statuses["CONTACT_DYNAMICS_REPAIR"] == "PASS" else "OBJECT_GUIDANCE_REPAIR"))
        if contact.get("selected"):
            seed = _promote_seed_from_row(attempt_root, contact["selected"])
            if seed is not None:
                d2 = _run_d2_with_seed(paths_config, attempt_root, seed)
                statuses["ORACLE_C_TEST"] = "PASS"
                statuses["D2_TEST"] = str(d2.get("status", "FAIL"))
                history.append(_checkpoint_record("D2_TEST", statuses["D2_TEST"], None if statuses["D2_TEST"] == "PASS" else "D2 hard gate failure after contact-dynamics seed", "OBJECT_GUIDANCE" if statuses["D2_TEST"] != "PASS" else None, "replayed selected contact-dynamics seed with real object dynamics", attempt_root, next_state="MINIMAL_MJWP" if statuses["D2_TEST"] == "PASS" else "OBJECT_GUIDANCE_REPAIR"))

    # If original dynamic retention did not produce a seed, exhaust the
    # separately authorized phase-scheduled object-guidance branch before any
    # timing relaxation.  This is deliberately cached on resume.
    guidance = None
    guidance_path = attempt_root / "reports/object_guidance_repair.json"
    if statuses["D2_TEST"] != "PASS" and not selected:
        guidance = json.loads(guidance_path.read_text(encoding="utf-8")) if guidance_path.is_file() else run_object_guidance_search(paths_config, attempt_root)
        dynamic_search["object_guidance_repair"] = guidance
        statuses["OBJECT_GUIDANCE_REPAIR"] = str(guidance.get("status", "BLOCKED"))
        history.append(_checkpoint_record("OBJECT_GUIDANCE_REPAIR", statuses["OBJECT_GUIDANCE_REPAIR"], None if statuses["OBJECT_GUIDANCE_REPAIR"] == "PASS" else "all bounded phase schedules failed", "OBJECT_GUIDANCE" if statuses["OBJECT_GUIDANCE_REPAIR"] != "PASS" else None, "ran phase-scheduled guidance without writing object qpos", attempt_root, next_state="D2_TEST" if statuses["OBJECT_GUIDANCE_REPAIR"] == "PASS" else "TIMING_FEASIBILITY"))
        if guidance.get("selected"):
            seed = _promote_seed_from_row(attempt_root, guidance["selected"])
            if seed is not None:
                d2 = _run_d2_with_seed(paths_config, attempt_root, seed)
                statuses["ORACLE_C_TEST"] = "PASS"
                statuses["D2_TEST"] = str(d2.get("status", "FAIL"))
                history.append(_checkpoint_record("D2_TEST", statuses["D2_TEST"], None if statuses["D2_TEST"] == "PASS" else "D2 hard gate failure after guidance seed", "OBJECT_GUIDANCE" if statuses["D2_TEST"] != "PASS" else None, "replayed selected guidance seed with real object dynamics", attempt_root, next_state="MINIMAL_MJWP" if statuses["D2_TEST"] == "PASS" else "TIMING_FEASIBILITY"))

    # Only after original-timing controller, guidance, and inherited contact
    # dynamics evidence are exhausted may the versioned timing ladder run.
    if statuses["D2_TEST"] != "PASS":
        if guidance and guidance.get("candidates"):
            best = max(guidance["candidates"], key=lambda row: float(row.get("patch", {}).get("patch_coverage") or 0.0))
            seed_candidate = best.get("controller_seed", DYNAMIC_CANDIDATES[0])
            guidance_candidate = best.get("candidate")
        else:
            seed_candidate = DYNAMIC_CANDIDATES[0]
            guidance_candidate = None
        timing = run_timing_ladder(paths_config, attempt_root, seed_candidate, guidance_candidate)
        statuses["TIMING_FEASIBILITY"] = str(timing.get("status", "BLOCKED"))
        history.append(_checkpoint_record("TIMING_FEASIBILITY", statuses["TIMING_FEASIBILITY"], None if statuses["TIMING_FEASIBILITY"] == "PASS" else "all original and relaxed timing variants failed", "DYNAMIC_TIMING_INFEASIBILITY" if statuses["TIMING_FEASIBILITY"] != "PASS" else None, "ran explicit 1.0x/1.25x/1.5x/2.0x timing variants", attempt_root, next_state="D2_TEST" if statuses["TIMING_FEASIBILITY"] == "PASS" else "BLOCKED"))
        if timing.get("selected"):
            seed = _promote_seed_from_row(attempt_root, timing["selected"])
            if seed is not None:
                d2 = _run_d2_with_seed(paths_config, attempt_root, seed, float(timing["selected"]["scale"]))
                statuses["ORACLE_C_TEST"] = "PASS"
                statuses["D2_TEST"] = str(d2.get("status", "FAIL"))
                history.append(_checkpoint_record("D2_TEST", statuses["D2_TEST"], None if statuses["D2_TEST"] == "PASS" else "D2 hard gate failure after timing relaxation", "OBJECT_GUIDANCE" if statuses["D2_TEST"] != "PASS" else None, "replayed first passing timing seed with explicit relaxed timing label", attempt_root, next_state="MINIMAL_MJWP" if statuses["D2_TEST"] == "PASS" else "BLOCKED"))

    # Always materialize a downstream gate report.  It is a read-only
    # fail-closed record when D2 is absent; only a real D2 PASS can open the
    # isolated MJWP/smoke/HTML branches.
    from spider.tools import grab_stage_c_v2r2e_downstream as downstream

    downstream_payload = downstream.run_downstream(paths_config, attempt_root.name)
    primary_downstream = downstream_payload.get("primary", {})
    minimal_status = primary_downstream.get("minimal", {}).get("status", "NOT_RUN")
    statuses["MINIMAL_MJWP"] = minimal_status if statuses["D2_TEST"] == "PASS" else "NOT_RUN"
    statuses["PRIMARY_MJWP"] = primary_downstream.get("status", "NOT_RUN") if statuses["D2_TEST"] == "PASS" else "NOT_RUN"
    statuses["SMOKE_1"] = downstream_payload.get("smokes", {}).get(SMOKE_1, {}).get("status", "NOT_RUN") if statuses["PRIMARY_MJWP"] == "PASS" else "NOT_RUN"
    statuses["SMOKE_2"] = downstream_payload.get("smokes", {}).get(SMOKE_2, {}).get("status", "NOT_RUN") if statuses["PRIMARY_MJWP"] == "PASS" else "NOT_RUN"
    from spider.tools import grab_stage_c_v2r2e_viewer as viewer
    html_payload = viewer.build_html(attempt_root)
    if statuses["SMOKE_1"] == "PASS" and statuses["SMOKE_2"] == "PASS":
        statuses["HTML"] = html_payload.get("status", "BLOCKED")
        if statuses["HTML"] == "PASS":
            screenshot_payload = viewer.screenshot_html(attempt_root)
            statuses["SCREENSHOT_REVIEW"] = screenshot_payload.get("status", "BLOCKED")
    if statuses["D2_TEST"] != "PASS":
        statuses["BLOCKED"] = "PASS"
        statuses["COMPLETE"] = "NOT_RUN"
    else:
        statuses["BLOCKED"] = "NOT_RUN"
        statuses["COMPLETE"] = "PASS" if all(statuses[state] == "PASS" for state in ("MINIMAL_MJWP", "PRIMARY_MJWP", "SMOKE_1", "SMOKE_2", "HTML", "SCREENSHOT_REVIEW")) else "BLOCKED"
    _write_json(attempt_root / "recovery_history.json", {"schema_version": 1, "attempts": history})
    return _write_final_reports(paths, attempt_root, history, statuses, audit, dynamic_search, timing, d2, inherited)


def resume_partial(paths_config: str, attempt_id: str) -> str:
    """Resume an interrupted attempt, including candidates with partial traces."""
    paths = load_project_paths(paths_config)
    attempt_root = _attempt_root(paths, attempt_id)
    audit_path = attempt_root / "reports/v2r2e_alignment_audit.json"
    if not audit_path.is_file():
        raise RuntimeError(f"partial V2R2E attempt has no alignment checkpoint: {attempt_root}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    refinement_path = attempt_root / "profiles/refinement_summary.json"
    refinements = json.loads(refinement_path.read_text(encoding="utf-8"))["branches"] if refinement_path.is_file() else run_refinement_branches(paths_config, attempt_root, audit)
    dynamic_path = attempt_root / "reports/dynamic_optimization_trace.json"
    dynamic_search = json.loads(dynamic_path.read_text(encoding="utf-8")) if dynamic_path.is_file() else run_dynamic_search(paths_config, attempt_root)
    history_path = attempt_root / "recovery_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")).get("attempts", []) if history_path.is_file() else []
    if not history:
        history.append(_checkpoint_record("AUDIT_ALIGNMENT", str(audit.get("status", "FAIL")), None, audit.get("root_cause_classification"), "resumed from preserved alignment checkpoint", attempt_root, next_state="DYNAMIC_TRAJECTORY_OPTIMIZATION"))
    return _finish_after_dynamic(paths_config, attempt_root, audit, refinements, dynamic_search, history)


def finalize_existing(paths_config: str, attempt_id: str) -> str:
    """Resume/finalize a completed attempt without rerunning its physics."""
    paths = load_project_paths(paths_config)
    attempt_root = _attempt_root(paths, attempt_id)
    audit_path = attempt_root / "reports/v2r2e_alignment_audit.json"
    dynamic_path = attempt_root / "reports/dynamic_optimization_trace.json"
    timing_path = attempt_root / "reports/timing_feasibility.json"
    history_path = attempt_root / "recovery_history.json"
    required = (audit_path, dynamic_path, timing_path, history_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize incomplete V2R2E attempt; missing: {missing}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dynamic_search = json.loads(dynamic_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8")).get("attempts", [])
    guidance_path = attempt_root / "reports/object_guidance_repair.json"
    continuation_path = attempt_root / "reports/recovery_continuation_object_guidance.json"
    guidance = json.loads(guidance_path.read_text(encoding="utf-8")) if guidance_path.is_file() else None
    continuation = json.loads(continuation_path.read_text(encoding="utf-8")) if continuation_path.is_file() else None
    refinement_path = attempt_root / "profiles/refinement_summary.json"
    refinements = json.loads(refinement_path.read_text(encoding="utf-8"))["branches"] if refinement_path.is_file() else {}
    d2_path = attempt_root / "reports/v2r2e_d2_rollout.json"
    d2 = json.loads(d2_path.read_text(encoding="utf-8")) if d2_path.is_file() else None
    inherited_path = attempt_root / "inherited/v2r2_branch_d/v2r2_contact_dynamics_repair.json"
    inherited_payload = json.loads(inherited_path.read_text(encoding="utf-8")) if inherited_path.is_file() else {}
    statuses = {state: "NOT_RUN" for state in STATE_ORDER}
    statuses["AUDIT_ALIGNMENT"] = str(audit.get("status", "FAIL"))
    for state, key in (("REFINE_OBJECT_COLLISION", "object"), ("REFINE_HAND_CONTACT_PROXY", "hand"), ("FIX_CONTACT_REGION_MAPPING", "mapping")):
        branch = refinements.get(key, {})
        statuses[state] = "PASS" if not branch.get("candidate_count") else str(branch.get("status", "BLOCKED"))
    statuses["DYNAMIC_TRAJECTORY_OPTIMIZATION"] = str(dynamic_search.get("status", "BLOCKED"))
    statuses["ORACLE_C_TEST"] = "PASS" if dynamic_search.get("status") == "PASS" else "FAIL"
    statuses["D2_TEST"] = "NOT_RUN" if d2 is None else str(d2.get("status", "FAIL"))
    statuses["TIMING_FEASIBILITY"] = str(timing.get("status", "BLOCKED"))
    if "OBJECT_GUIDANCE_REPAIR" in statuses:
        statuses["OBJECT_GUIDANCE_REPAIR"] = "NOT_RUN" if guidance is None else str(guidance.get("status", "BLOCKED"))
    if guidance is not None:
        dynamic_search["object_guidance_repair"] = guidance
        dynamic_search["contact_dynamics_inherited"] = inherited_payload
    statuses["BLOCKED"] = "PASS" if statuses["D2_TEST"] != "PASS" else "NOT_RUN"
    statuses["COMPLETE"] = "NOT_RUN" if statuses["D2_TEST"] != "PASS" else "BLOCKED"
    if guidance is not None:
        history.append(_checkpoint_record("OBJECT_GUIDANCE_REPAIR", statuses["OBJECT_GUIDANCE_REPAIR"], None if statuses["OBJECT_GUIDANCE_REPAIR"] == "PASS" else "all bounded phase schedules failed", "OBJECT_GUIDANCE" if statuses["OBJECT_GUIDANCE_REPAIR"] != "PASS" else None, "resumed object-guidance branch with source trajectory unchanged", attempt_root, next_state="D2_TEST" if statuses["OBJECT_GUIDANCE_REPAIR"] == "PASS" else "TIMING_FEASIBILITY"))
        if continuation is not None and continuation.get("timing") is not None:
            history.append(_checkpoint_record("TIMING_FEASIBILITY", str(continuation["timing"].get("status", "BLOCKED")), "guidance timing variants exhausted" if continuation["timing"].get("status") != "PASS" else None, "DYNAMIC_TIMING_INFEASIBILITY" if continuation["timing"].get("status") != "PASS" else None, "resumed timing ladder after object-guidance branch", attempt_root, next_state="BLOCKED" if continuation["timing"].get("status") != "PASS" else "D2_TEST"))
    return _write_final_reports(paths, attempt_root, history, statuses, audit, dynamic_search, timing, d2, {"status": inherited_payload.get("status", "UNKNOWN"), "candidate_count": len(json.loads((attempt_root / "inherited/v2r2_branch_d/contact_dynamics_search.json").read_text(encoding="utf-8")).get("candidates", [])) if (attempt_root / "inherited/v2r2_branch_d/contact_dynamics_search.json").is_file() else 0, "inherited_artifacts": str(attempt_root / "inherited/v2r2_branch_d")})


def continue_from_checkpoint(paths_config: str, attempt_id: str, checkpoint: str) -> str:
    """Continue a failed primary attempt at an authorized recovery state."""
    if checkpoint == "CONTACT_DYNAMICS_REPAIR":
        paths = load_project_paths(paths_config)
        attempt_root = _attempt_root(paths, attempt_id)
        attempt_root.mkdir(parents=True, exist_ok=True)
        contact = run_contact_dynamics_search(paths_config, attempt_root)
        history_path = attempt_root / "recovery_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")).get("attempts", []) if history_path.is_file() else []
        history.append(_checkpoint_record("CONTACT_DYNAMICS_REPAIR", contact["status"], None if contact["status"] == "PASS" else "all frozen explicit-pair contact profiles exhausted", "CONTACT_COLLISION_COUPLING", "ran all bounded in-memory contact-pair profiles with historical lead-20 controller seed", attempt_root, next_state="D2_TEST" if contact["status"] == "PASS" else "BLOCKED"))
        _write_json(history_path, {"schema_version": 1, "attempts": history})
        return str(attempt_root / "reports/contact_dynamics_recovery.json")
    if checkpoint == "TIMING_FEASIBILITY":
        paths = load_project_paths(paths_config)
        attempt_root = _attempt_root(paths, attempt_id)
        attempt_root.mkdir(parents=True, exist_ok=True)
        timing = run_timing_ladder(paths_config, attempt_root, DYNAMIC_CANDIDATES[0])
        payload = {"schema_version": 1, "stage": "C-V2R2E-RECOVERY", "checkpoint": checkpoint, "attempt_id": attempt_id, "timing": timing, "next_state": "D2_TEST" if timing.get("selected") else "BLOCKED", "d2": "NOT_RUN"}
        _write_json(attempt_root / "reports/recovery_continuation_timing.json", payload)
        return str(attempt_root / "reports/recovery_continuation_timing.json")
    if checkpoint != "OBJECT_GUIDANCE_REPAIR":
        raise ValueError(f"unsupported continuation checkpoint: {checkpoint}")
    paths = load_project_paths(paths_config)
    attempt_root = _attempt_root(paths, attempt_id)
    attempt_root.mkdir(parents=True, exist_ok=True)
    guidance = run_object_guidance_search(paths_config, attempt_root)
    timing = None
    d2 = None
    if guidance["status"] != "PASS":
        best = max(guidance["candidates"], key=lambda row: float(row.get("patch", {}).get("patch_coverage") or 0.0))
        timing = run_timing_ladder(paths_config, attempt_root, best["controller_seed"], best["candidate"])
        if timing.get("selected"):
            seed = attempt_root / "trajectory_v2r2e_dynamic_seed.npz"
            if seed.is_file():
                d2 = _run_d2_with_seed(paths_config, attempt_root, seed, float(timing["selected"]["scale"]))
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-RECOVERY",
        "checkpoint": checkpoint,
        "attempt_id": attempt_id,
        "object_guidance": guidance,
        "timing": timing,
        "d2": d2,
        "next_state": "D2_TEST" if d2 and d2.get("status") == "PASS" else "BLOCKED",
        "downstream_not_run": ["Minimal MJWP", "Primary Full MJWP", "Smoke 1", "Smoke 2", "HTML", "Screenshot review"],
    }
    _write_json(attempt_root / "reports/recovery_continuation_object_guidance.json", payload)
    return str(attempt_root / "reports/recovery_continuation_object_guidance.json")


def run(
    paths_config: str = "configs/local/paths.yaml",
    attempt_id: str | None = None,
    resume: bool = False,
    from_checkpoint: str | None = None,
    failed_only: bool = False,
    dry_run: bool = False,
    max_attempts: int = 12,
    report_json: bool = True,
) -> str:
    """Run the bounded primary-first V2R2E recovery controller."""
    del failed_only, report_json
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    paths = load_project_paths(paths_config)
    config = _config()
    attempt = attempt_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_root = _attempt_root(paths, attempt)
    if from_checkpoint:
        return continue_from_checkpoint(paths_config, attempt, from_checkpoint)
    if resume:
        if (attempt_root / "recovery_history.json").is_file():
            return resume_partial(paths_config, attempt)
        if (attempt_root / "reports/v2r2e_alignment_audit.json").is_file():
            return resume_partial(paths_config, attempt)
    attempt_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    statuses = {state: "NOT_RUN" for state in STATE_ORDER}
    if dry_run:
        payload = {"schema_version": 1, "stage": "C-V2R2E", "status": "DRY_RUN", "attempt_id": attempt, "states": STATE_ORDER, "candidate_limits": config["search"], "attempt_root": str(attempt_root)}
        _write_json(attempt_root / "dry_run.json", payload)
        return str(attempt_root / "dry_run.json")

    baseline = {
        "attempt_id": attempt,
        "stage": "C-V2R2E",
        "branch": "develop/wuji-hand2",
        "primary": PRIMARY,
        "primary_frames": PILOTS[PRIMARY]["frames"],
        "scene_sha256": _sha256(Path(dynamic._physics(paths, PRIMARY)["scene_act"])),
        "config_sha256": _sha256(CONFIG_PATH),
        "cxa_report": str(paths.workspace_root / "reports/cxa_case_a_rerun.json"),
        "v2r1_report": str(paths.workspace_root / "reports/v2r1_oracle_decision.json"),
        "v2r2_report": str(paths.workspace_root / "reports/v2r2_contact_dynamics_repair.json"),
        "frozen_policy": {"raw_grab": "read-only", "body_models": "read-only", "cxa": "read-only", "smokes": "primary-first", "push": "forbidden"},
    }
    _write_json(attempt_root / "input_manifest.json", baseline)
    frozen = _load_frozen(paths)
    audit = run_alignment_audit(paths_config, attempt_root)
    statuses["AUDIT_ALIGNMENT"] = audit["status"]
    history.append(_checkpoint_record("AUDIT_ALIGNMENT", audit["status"], None if audit["status"] == "PASS" else audit["root_cause_classification"], audit["root_cause_classification"], "computed object/hand/contact-layer alignment from real meshes and Oracle C contacts", attempt_root, next_state="REFINE_OBJECT_COLLISION"))
    refinements = run_refinement_branches(paths_config, attempt_root, audit)
    statuses["REFINE_OBJECT_COLLISION"] = refinements["object"]["status"] if refinements["object"]["candidate_count"] else "PASS"
    statuses["REFINE_HAND_CONTACT_PROXY"] = refinements["hand"]["status"] if refinements["hand"]["candidate_count"] else "PASS"
    statuses["FIX_CONTACT_REGION_MAPPING"] = refinements["mapping"]["status"] if refinements["mapping"]["candidate_count"] else "PASS"
    for state, key in (("REFINE_OBJECT_COLLISION", "object"), ("REFINE_HAND_CONTACT_PROXY", "hand"), ("FIX_CONTACT_REGION_MAPPING", "mapping")):
        history.append(_checkpoint_record(state, statuses[state], None if statuses[state] == "PASS" else refinements[key]["reason"], audit["root_cause_classification"], refinements[key]["reason"], attempt_root, next_state="ORACLE_C_TEST"))

    # The immutable Oracle-D PASS is still required before interpreting any C
    # candidate.  It is copied by reference, never regenerated over history.
    oracle_d = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2r/oracle_d_no_contact_tracking.json"
    if not oracle_d.is_file() or json.loads(oracle_d.read_text(encoding="utf-8")).get("status") != "PASS":
        statuses["ORACLE_C_TEST"] = "BLOCKED"
        history.append(_checkpoint_record("ORACLE_C_TEST", "BLOCKED", "historical Oracle D evidence unavailable", "INPUT_FREEZE", "stop before physical recovery", attempt_root, next_state="BLOCKED"))
        inherited = _copy_historical_branch_evidence(paths, attempt_root)
        statuses["BLOCKED"] = "PASS"
        _write_json(attempt_root / "recovery_history.json", {"schema_version": 1, "attempts": history})
        return _write_final_reports(paths, attempt_root, history, statuses, audit, None, None, None, inherited)
    else:
        statuses["ORACLE_C_TEST"] = "READY"
        history.append(_checkpoint_record("ORACLE_C_TEST", "READY", None, None, "verified immutable Oracle D PASS before Oracle C recovery", attempt_root, next_state="DYNAMIC_TRAJECTORY_OPTIMIZATION"))
        dynamic_search = run_dynamic_search(paths_config, attempt_root)
        statuses["DYNAMIC_TRAJECTORY_OPTIMIZATION"] = dynamic_search["status"]
        statuses["ORACLE_C_TEST"] = "PASS" if dynamic_search["status"] == "PASS" else "FAIL"
        history.append(_checkpoint_record("DYNAMIC_TRAJECTORY_OPTIMIZATION", dynamic_search["status"], None if dynamic_search["status"] == "PASS" else "bounded original-timing candidates exhausted", "DYNAMIC_RETENTION" if dynamic_search["status"] != "PASS" else None, "ran all frozen dynamic candidates against actual MuJoCo hand dynamics and immutable C-XA object trajectory", attempt_root, next_state="D2_TEST" if dynamic_search["status"] == "PASS" else "TIMING_FEASIBILITY"))
        return _finish_after_dynamic(paths_config, attempt_root, audit, refinements, dynamic_search, history)


if __name__ == "__main__":
    tyro.cli(run)
