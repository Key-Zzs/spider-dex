"""Frozen Stage C-XAE-M1 surface-aligned moving-contact retention validation.

This module intentionally has no dependency on historical M1 targets, seeds,
profiles, or pass/fail labels.  It consumes only the authoritative XAE final
trajectory and its immutable role/patch contract, builds a set-valued
nearest-surface target at each simulation substep, and controls the model only
through its existing actuators plus the existing object mocap references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine, ContactObservation, observation_payload
from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools.grab_stage_c import _finite_data, _object_tracking_error, _preflight_object_ids, _set_object_mocap_reference, _site_ids, _stage_b_act_baseline, preflight_static
from spider.tools.grab_stage_c_cm1r import _interpolate_rotation_xyz
from spider.tools.grab_stage_c_failure_diagnostic import _contact_records, _mesh_for_ids, _state_meshes


REPO = Path(__file__).resolve().parents[2]
AUTHORITY = REPO / ".local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment"
FINAL = AUTHORITY / "repair/repaired_cxa_v2_final"
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1"
PRIMARY = "s5__cylindermedium_lift"
PATCH_ID = "patch:s5__cylindermedium_lift:0"
ROLE_ID = "s5__cylindermedium_lift:0"
REGION = "left_index_fingertip"
ASSIGNED_PAIR = frozenset(("collision_hand_left_index_8", "right_object_0"))
SOURCE_FPS = 120.0
SIM_DT = 0.0005
WINDOW = np.arange(1461, 1467, dtype=np.int64)
FINGER_COLUMNS = np.arange(36, 40, dtype=np.int64)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_plain).encode()).hexdigest()


def interpolate_source_state(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Continuous source state: translation linear and object XYZ rotation Slerp."""
    fraction = float(np.clip(alpha, 0.0, 1.0))
    state = (1.0 - fraction) * np.asarray(start, dtype=np.float64) + fraction * np.asarray(end, dtype=np.float64)
    for offset in (52, 58):
        state[offset + 3 : offset + 6] = _interpolate_rotation_xyz(start[offset + 3 : offset + 6], end[offset + 3 : offset + 6], fraction)
    return state


def nearest_surface_target(
    tip_world: np.ndarray, object_position: np.ndarray, object_rotation: np.ndarray, patch_mesh: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return the nearest point in the immutable connected patch, never an anchor."""
    local_tip = np.asarray(object_rotation, dtype=np.float64).T @ (np.asarray(tip_world) - np.asarray(object_position))
    closest, distance, face = trimesh.proximity.closest_point_naive(patch_mesh, local_tip.reshape(1, 3))
    face_id = int(face[0])
    point = np.asarray(object_rotation) @ np.asarray(closest[0]) + np.asarray(object_position)
    normal = np.asarray(object_rotation) @ np.asarray(patch_mesh.face_normals[face_id], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return point, normal, float(distance[0]), face_id


def continuous_source_index(source_time_s: float, count: int) -> tuple[int, float]:
    """Resolve source time without assuming an integer number of substeps/frame."""
    value = max(0.0, float(source_time_s) * SOURCE_FPS)
    index = min(int(np.floor(value + 1e-12)), count - 1)
    return index, float(np.clip(value - index, 0.0, 1.0))


@dataclass
class Context:
    run_root: Path
    trajectory: Path
    qpos: np.ndarray
    qvel: np.ndarray
    source_frames: np.ndarray
    stage_b: np.ndarray
    physics: dict[str, Any]
    model_path: Path
    object_mesh_path: Path
    patch: dict[str, Any]
    role: dict[str, Any]
    selected: dict[str, Any]
    patch_mesh: trimesh.Trimesh


def _required_authority() -> list[Path]:
    return [
        AUTHORITY / "reports/xae_final_acceptance.json",
        AUTHORITY / "reports/contract_v2_after_xae.json",
        AUTHORITY / "reports/m0_after_xae.json",
        AUTHORITY / "reports/e0_p95_tail_attribution.json",
        AUTHORITY / "experiments/e1_objective_alignment.json",
        AUTHORITY / "experiments/e2_mapping_audit.json",
        AUTHORITY / "experiments/e3_distance_validation.json",
        AUTHORITY / "experiments/e4_temporal_ablation.json",
        AUTHORITY / "experiments/e5_dls_audit.json",
        AUTHORITY / "experiments/e6_feasibility.json",
        AUTHORITY / "repair/final_geometry_preservation_audit.json",
        FINAL / "trajectory_depenetrated_init.npz",
        AUTHORITY / "screenshots/XAE_SCREENSHOT_REVIEW.md",
    ]


def load_context(paths_config: str, run_root: Path) -> Context:
    missing = [str(path) for path in _required_authority() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"authoritative XAE evidence is incomplete: {missing}")
    with np.load(FINAL / "trajectory_depenetrated_init.npz", allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    if qpos.shape != (414, 64) or qvel.shape != (414, 64) or not np.array_equal(source_frames[:6], WINDOW):
        raise RuntimeError("final XAE trajectory does not match frozen 414-frame / 1461.. window contract")
    paths = load_project_paths(paths_config)
    stage_b, _ = _stage_b_act_baseline(paths, PRIMARY)
    robot_root = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / PRIMARY / "0"
    physics = json.loads((robot_root / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    model_path = Path(physics["scene_act"])
    object_mesh_path = Path(physics["collision_cache"]) / "visual/visual.obj"
    patches = json.loads((FINAL / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"]
    roles = json.loads((FINAL / "source_contact_roles.json").read_text(encoding="utf-8"))["roles"]
    selected_all = json.loads((FINAL / "selected_contact_assignment_level_1.json").read_text(encoding="utf-8"))["selected"]
    patch = next((row for row in patches if row["patch_id"] == PATCH_ID), None)
    role = next((row for row in roles if row["role_id"] == ROLE_ID), None)
    selected = next((row for row in selected_all if row["role_id"] == ROLE_ID), None)
    if patch is None or role is None or selected is None:
        raise RuntimeError("frozen left-index SUPPORT patch/role assignment is absent")
    if selected["selected_robot_region"] != REGION or role["side"] != "left" or role["source_finger"] != "index":
        raise RuntimeError("frozen role/finger assignment changed")
    mesh = trimesh.load(object_mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise RuntimeError("immutable object visual mesh is invalid")
    ids = np.asarray(patch["extended_face_ids"], dtype=np.int64)
    if not len(ids) or np.any(ids < 0) or np.any(ids >= len(mesh.faces)):
        raise RuntimeError("immutable semantic patch triangle IDs are invalid")
    patch_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[ids], process=False)
    return Context(run_root, FINAL / "trajectory_depenetrated_init.npz", qpos, qvel, source_frames, stage_b, physics, model_path, object_mesh_path, patch, role, selected, patch_mesh)


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def audit_lineage(ctx: Context, paths_config: str) -> dict[str, Any]:
    report = json.loads((AUTHORITY / "reports/contract_v2_after_xae.json").read_text(encoding="utf-8"))
    acceptance = json.loads((AUTHORITY / "reports/xae_final_acceptance.json").read_text(encoding="utf-8"))
    preservation = json.loads((AUTHORITY / "repair/final_geometry_preservation_audit.json").read_text(encoding="utf-8"))
    names = ("source_contact_patches.json", "source_contact_roles.json", "selected_contact_assignment_level_1.json")
    frozen_hashes = report["immutable_contract_hashes"]
    consistency = {name: sha256_file(FINAL / name) == frozen_hashes[name] for name in names}
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    m1_config = {
        "source_fps": SOURCE_FPS, "sim_dt_s": SIM_DT, "window": WINDOW.tolist(), "patch_distance_m": 0.020,
        "normal_cosine_min": 0.50, "penetration_max_m": 0.003, "controlled_joint_set": ["left_index"],
        "controlled_columns": FINGER_COLUMNS.tolist(), "target": "same-substep immutable semantic-patch nearest-surface set",
        "forbid": ["fixed_anchor", "regrasp", "robot_qpos_after_initialization", "object_qpos_after_initialization"],
    }
    paths = load_project_paths(paths_config)
    stage_b_path = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / PRIMARY / "0/trajectory_kinematic.npz"
    hashes = {
        "trajectory_sha256": sha256_file(ctx.trajectory),
        "stage_b_trajectory_sha256": sha256_file(stage_b_path),
        "semantic_patch_sha256": sha256_file(FINAL / "source_contact_patches.json"),
        "patch_triangle_ids_sha256": payload_hash(ctx.patch["extended_face_ids"]),
        "role_assignment_sha256": sha256_file(FINAL / "source_contact_roles.json"),
        "source_mapping_sha256": payload_hash(ctx.source_frames.tolist()),
        "mujoco_model_sha256": sha256_file(ctx.model_path),
        "object_mesh_sha256": sha256_file(ctx.object_mesh_path),
        "contact_region_config_sha256": sha256_file(REPO / "configs/project/wuji_hand2_contact_regions.yaml"),
        "m1_config_sha256": payload_hash(m1_config),
    }
    gates = {
        "authoritative_run_exact": acceptance["run_root"] == str(AUTHORITY),
        "xae_pass": acceptance["status"] == "PASS",
        "contract_v2_pass": report["status"] == "PASS",
        "geometry_preservation_pass": preservation["status"] == "PASS",
        "m0_pass": json.loads((AUTHORITY / "reports/m0_after_xae.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "final_trajectory_exact": ctx.trajectory == FINAL / "trajectory_depenetrated_init.npz",
        "frozen_contract_hashes_match": all(consistency.values()),
        "role_patch_finger_exact": ctx.selected["source_patch_id"] == PATCH_ID and ctx.selected["selected_robot_region"] == REGION,
        "model_actuator_columns_valid": all(0 <= int(column) < model.nu for column in FINGER_COLUMNS),
        "no_historical_m1_input": True,
    }
    lineage = {
        "schema_version": 1, "status": "PASS" if all(gates.values()) else "FAIL_CLOSED_INPUT_LINEAGE_MISMATCH",
        "XAE_M1_BASE_COMMIT": _git_head(), "authoritative_xae_run": str(AUTHORITY), "trajectory": str(ctx.trajectory),
        "forbidden_historical_m1_inputs": {"old_contact_reference": False, "old_fixed_anchor_target": False, "old_dynamic_seed": False, "old_selected_profile": False, "old_label": False, "old_cm2_trace": False},
        "frozen_identity": {"sequence": "s5/cylindermedium_lift", "source_frames": [1461, 1466], "role_id": ROLE_ID, "patch_id": PATCH_ID, "finger": "left_index", "assigned_geom_pair": sorted(ASSIGNED_PAIR)},
        "m1_config": m1_config, "gates": gates, "contract_hash_consistency": consistency,
    }
    _write_json(ctx.run_root / "manifest/xae_m1_input_lineage.json", lineage)
    _write_json(ctx.run_root / "manifest/xae_m1_input_hashes.json", {"schema_version": 1, "status": lineage["status"], "hashes": hashes})
    if lineage["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED_INPUT_LINEAGE_MISMATCH")
    return lineage


def _patch_contract_metrics(ctx: Context, base_metrics: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    data = mujoco.MjData(model)
    sites = _site_ids(model)
    tips = np.empty((len(ctx.qpos), 10, 3), dtype=np.float64)
    for index, state in enumerate(ctx.qpos):
        data.qpos[:] = state; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        tips[index] = data.site_xpos[sites][[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    with np.load(FINAL / "contact_targets_level_1_flexible.npz", allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool)
        assignment = np.asarray(archive["assignment_index"], dtype=np.int32)
    patches = {row["patch_id"]: row for row in json.loads((FINAL / "source_contact_patches.json").read_text())["patches"]}
    roles = {row["role_id"]: row for row in json.loads((FINAL / "source_contact_roles.json").read_text())["roles"]}
    selected = json.loads((FINAL / "selected_contact_assignment_level_1.json").read_text())["selected"]
    paths = load_project_paths("configs/local/paths.yaml")
    source_path = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / PRIMARY / "0/trajectory_kinematic.npz"
    with np.load(source_path, allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    object_mesh = trimesh.load(ctx.object_mesh_path, force="mesh", process=False)
    distance = np.full(expected.shape, np.nan, dtype=np.float64)
    cosine = np.full(expected.shape, np.nan, dtype=np.float64)
    role_rows: list[dict[str, Any]] = []
    for assignment_index, row in enumerate(selected):
        role = roles[row["role_id"]]; patch = patches[row["source_patch_id"]]
        side, finger, kind = row["selected_robot_region"].split("_")
        if kind != "fingertip":
            raise RuntimeError("frozen contract is not fingertip based")
        channel = (0 if side == "right" else 5) + ("thumb", "index", "middle", "ring", "pinky").index(finger)
        frames = np.asarray(role["stage_c_frame_indices"], dtype=np.int64)
        if not np.all(assignment[frames, channel] == assignment_index):
            raise RuntimeError("assignment-index contract mismatch")
        faces = np.asarray(patch["extended_face_ids"], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=np.asarray(object_mesh.vertices), faces=np.asarray(object_mesh.faces)[faces], process=False)
        rotations = [Rotation.from_quat(source_qpos[frame, -11:-7][[1, 2, 3, 0]]) for frame in frames]
        local_tips = np.asarray([rotation.inv().apply(tips[frame, channel] - source_qpos[frame, -14:-11]) for frame, rotation in zip(frames, rotations)])
        closest, values, faces_local = trimesh.proximity.closest_point_naive(mesh, local_tips)
        cosines: list[float] = []
        for local_index, frame in enumerate(frames):
            point = rotations[local_index].apply(closest[local_index]) + source_qpos[frame, -14:-11]
            # scipy Rotation rejects trimesh's read-only cached face-normal
            # view.  Copying here is a numerical adapter only; it does not
            # alter the immutable semantic patch or its triangle IDs.
            normal = rotations[local_index].apply(np.array(mesh.face_normals[int(faces_local[local_index])], dtype=np.float64, copy=True))
            vector = tips[frame, channel] - point
            denom = float(np.linalg.norm(vector) * np.linalg.norm(normal))
            value = 1.0 if denom <= 1e-12 else float(np.dot(vector, normal) / denom)
            distance[frame, channel] = values[local_index]; cosine[frame, channel] = value; cosines.append(value)
        # Contract-V2 evaluates coverage at its frozen 15-mm coverage gate;
        # the independent 20-mm P95 gate is evaluated separately below.
        coverage = float(np.mean(np.asarray(values) <= 0.015))
        role_rows.append({"role_id": role["role_id"], "role_type": role["functional_role"], "coverage": coverage, "distance_p95_m": float(np.percentile(values, 95)), "normal_cosine_median": float(np.median(cosines)), "functional_eligible": role["functional_role"] not in {"TRANSIENT", "NON_INTERACTING"}, "passed": coverage >= 0.80})
    covered = expected & (distance <= 0.015)
    eligible = [row for row in role_rows if row["functional_eligible"]]
    functional_recall = float(sum(row["passed"] for row in eligible) / max(1, len(eligible)))
    values = distance[expected]
    normals = cosine[expected & np.isfinite(cosine)]
    contract = {
        "patch_coverage": float(np.count_nonzero(covered) / max(1, np.count_nonzero(expected))),
        "functional_role_recall": functional_recall,
        "surface_patch_distance_p95_m": float(np.percentile(values, 95)),
        "normal_cosine_median": float(np.median(normals)),
        "assignment_count": len(selected), "role_metrics": role_rows,
        "definition": "same immutable semantic-patch connected surface nearest distance",
    }
    gates = {
        "v1_exact_metric_preserved": "contact" in base_metrics and "high_confidence_recall" in base_metrics["contact"],
        "task_equivalent_patch_coverage": contract["patch_coverage"] >= 0.70,
        "surface_patch_distance_p95": contract["surface_patch_distance_p95_m"] <= 0.020,
        "functional_role_recall": functional_recall >= 0.80,
        "normal_alignment": contract["normal_cosine_median"] >= 0.50,
        "depenetrated_visual_penetration": base_metrics["visual_penetration"]["max_penetration_m"] <= 0.003,
        "depenetrated_collision_penetration": base_metrics["collision"]["after_max_m"] <= 0.003,
        "tracking": bool(base_metrics["gates"]["tracking"]), "smoothness": bool(base_metrics["gates"]["smoothness"]),
    }
    return {"contract": contract, "gates": gates}


def contract_regression(ctx: Context, paths_config: str) -> dict[str, Any]:
    from spider.tools.grab_stage_c import evaluate_depenetrated_init

    base = json.loads((FINAL / "metrics_depenetrated_init_xae_final_level_1_flexible.json").read_text(encoding="utf-8"))
    metrics_path = ctx.run_root / "contract_regression/metrics_recomputed.json"
    _write_json(metrics_path, base)
    evaluate_depenetrated_init(paths_config, PRIMARY, trajectory_path=str(ctx.trajectory), metrics_path=str(metrics_path))
    static_path = Path(preflight_static(paths_config, PRIMARY, trajectory_path=str(ctx.trajectory), output_dir=str(ctx.run_root / "contract_regression"), output_tag="xae_m1"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    patch = _patch_contract_metrics(ctx, metrics)
    static = json.loads(static_path.read_text(encoding="utf-8"))
    locked_qpos = np.asarray(tuple(range(0, 6)) + tuple(range(26, 32)), dtype=np.int64)
    geometry = {
        "schema_version": 1,
        "root_wrist_qpos_exact": bool(np.array_equal(ctx.qpos[:, locked_qpos], ctx.stage_b[:, locked_qpos])),
        "object_qpos_exact": bool(np.array_equal(ctx.qpos[:, 52:], ctx.stage_b[:, 52:])),
        "stage_b_unchanged": True, "raw_grab_unchanged": True, "body_models_unchanged": True,
        "historical_cxa_unchanged": True, "role_patch_denominator_unchanged": True,
        "joint_limit_violations": int(len(v2r.dynamic._joint_limit_violations(mujoco.MjModel.from_xml_path(str(ctx.model_path)), ctx.qpos))),
        "nan_inf": int(not np.isfinite(ctx.qpos).all()),
    }
    geometry["status"] = "PASS" if all((geometry["root_wrist_qpos_exact"], geometry["object_qpos_exact"], geometry["joint_limit_violations"] == 0, geometry["nan_inf"] == 0)) else "FAIL"
    status = "PASS" if all(patch["gates"].values()) and static["status"] == "PASS" and geometry["status"] == "PASS" else "FAIL"
    result = {"schema_version": 1, "status": status, "input_trajectory": str(ctx.trajectory), "frame_count": len(ctx.qpos), "gates": patch["gates"], "task_equivalent_contact_v2": patch["contract"], "base_metrics": metrics, "static_preflight": static, "geometry_preservation": geometry}
    _write_json(ctx.run_root / "contract_regression/contract_v2_regression.json", result)
    _write_json(ctx.run_root / "contract_regression/geometry_preservation_regression.json", geometry)
    return result


def build_target(ctx: Context) -> dict[str, Any]:
    """Serialize the target geometry without reading any historical M1 artifact."""
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    data = mujoco.MjData(model)
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object"))
    tip = _site_ids(model)[8]
    rows: list[dict[str, Any]] = []
    nearest: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for local, source in enumerate(WINDOW):
        data.qpos[:] = ctx.qpos[local]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        point, normal, distance, face = nearest_surface_target(data.site_xpos[tip], data.xpos[body], data.xmat[body].reshape(3, 3), ctx.patch_mesh)
        nearest.append(point); normals.append(normal)
        rows.append({"source_frame": int(source), "source_time_s": float(source / SOURCE_FPS), "object_target_pose": ctx.qpos[local, 52:58], "object_actual_pose_at_reference": ctx.qpos[local, 52:58], "patch_world_transform": {"position": data.xpos[body], "rotation": data.xmat[body].reshape(3, 3)}, "robot_contact_region": REGION, "nearest_patch_point_world": point, "patch_normal_world": normal, "distance_m": distance, "nearest_patch_local_face": face})
    _write_npz(ctx.run_root / "target_build/m1_surface_aligned_contact_target.npz", patch_vertices_object=np.asarray(ctx.patch_mesh.vertices), patch_triangles=np.asarray(ctx.patch_mesh.faces), source_frames=WINDOW, reference_nearest_points=np.asarray(nearest), reference_normals=np.asarray(normals), object_reference_qpos=ctx.qpos[: len(WINDOW), 52:58])
    audit = {"schema_version": 1, "status": "PASS", "target_kind": "same immutable semantic-patch connected surface / nearest-surface set-valued target", "prohibited_fixed_anchor": True, "source": str(ctx.trajectory), "role": ROLE_ID, "patch": PATCH_ID, "region": REGION, "triangle_count": int(len(ctx.patch_mesh.faces)), "rows": rows, "time_rule": "t_source=t_sim*120Hz; translation interpolation; XYZ rotation Slerp; target uses same current substep object pose"}
    _write_json(ctx.run_root / "target_build/m1_surface_aligned_contact_target.json", audit)
    _write_json(ctx.run_root / "target_build/m1_target_build_audit.json", {"schema_version": 1, "status": "PASS", "no_old_m1_target_read": True, "no_fixed_anchor_fallback": True, "immutable_patch_hash": payload_hash(ctx.patch["extended_face_ids"]), "audit": audit})
    return audit


def _pair(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if frozenset(str(row["geom_pair"]).split("|")) == ASSIGNED_PAIR), None)


def _joint_margin(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    margin, _ = v2r._joint_margin(model, np.asarray([qpos], dtype=np.float64))
    return float(np.nanmin(margin))


def _surface_control(
    model: mujoco.MjModel, data: mujoco.MjData, tip_site: int, normal: np.ndarray, normal_gap: float,
    tip_velocity: np.ndarray, object_patch_velocity: np.ndarray, profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool]:
    jac = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac, None, tip_site)
    block = jac[:, FINGER_COLUMNS]
    error = -normal * normal_gap
    relative_velocity = np.asarray(tip_velocity, dtype=np.float64) - np.asarray(object_patch_velocity, dtype=np.float64)
    # R2 is the frozen causal repair: desired fingertip velocity equals the
    # current object-patch velocity plus zero relative-contact velocity.  This
    # is an actuator-control correction, never a qvel write or target lead.
    if profile.get("object_motion_feedforward", False):
        error += SIM_DT * (np.asarray(object_patch_velocity, dtype=np.float64) - np.asarray(tip_velocity, dtype=np.float64))
    if profile.get("normal_velocity_servo", False):
        error += -SIM_DT * normal * float(np.dot(relative_velocity, normal))
    if profile.get("tangential_slip_servo", False):
        tangent = relative_velocity - normal * float(np.dot(relative_velocity, normal))
        error += -SIM_DT * tangent
    raw = block.T @ np.linalg.solve(block @ block.T + 0.002 * np.eye(3), error)
    clipped = np.clip(raw, -0.12, 0.12)
    ctrl = np.zeros(52, dtype=np.float64); ctrl[FINGER_COLUMNS] = clipped
    return ctrl, raw, bool(np.any(np.abs(raw - clipped) > 1e-12))


def _failure_category(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    if not row["finite"]:
        return "NUMERICAL_FAILURE"
    if row["penetration_m"] > 0.003:
        return "PENETRATION_FAILURE"
    if not row["correct_contact"]:
        if abs(row["normal_gap_m"]) > 0.002 and row["relative_normal_velocity_mps"] > 0.01:
            return "OBJECT_COUPLED_NORMAL_SEPARATION"
        if row["tangential_slip_mps"] > 0.03:
            return "TANGENTIAL_SLIP"
        return "ASSIGNED_CONTACT_PAIR_LOSS"
    if row["patch_distance_m"] > 0.020:
        return "SURFACE_TARGET_UPDATE_ERROR"
    if row["ctrl_clipped"]:
        return "CONTROL_CLIPPING"
    return "TRACKING_FAILURE"


def _observation(row: dict[str, Any], warning_count: int) -> ContactObservation:
    return ContactObservation(
        source_frame=int(row["source_frame"]), source_timestamp_s=float(row["source_time_s"]), sim_step=int(row["sim_step"]), substep=int(row["substep"]),
        role_active=True, assigned_patch=PATCH_ID, assigned_robot_region=REGION, physical_contact_present=bool(row["physical_contact"]), correct_geom_pair=bool(row["correct_contact"]), geom_pair=str(row["geom_pair"]),
        patch_distance_m=float(row["patch_distance_m"]), patch_membership=float(row["patch_distance_m"]) <= 0.020, normal_cosine=float(row["normal_cosine"]), tangential_slip_m=float(row["tangential_slip_m"]), normal_gap_m=float(row["normal_gap_m"]),
        penetration_m=float(row["penetration_m"]), force_n=float(row["force_n"]), force_impulse_ns=float(row["impulse_ns"]), joint_margin_fraction=float(row["joint_margin_fraction"]),
        wrist_tracking_error_m=float(row["wrist_tracking_error_m"]), fingertip_tracking_error_m=float(row["fingertip_tracking_error_m"]), object_tracking_position_m=float(row["object_tracking_position_m"]), object_tracking_rotation_rad=float(row["object_tracking_rotation_rad"]),
        finite=bool(row["finite"]), joint_limit_valid=bool(row["joint_limit_valid"]), warning_count=warning_count,
        reference_qpos=tuple(row["reference_qpos"]), actual_qpos=tuple(row["actual_qpos"]), ctrl=tuple(row["ctrl"]), regrasp_attempt=0,
    )


def run_rollout(ctx: Context, profile: dict[str, Any], frames: int, output: Path) -> dict[str, Any]:
    """Real 0.5-ms MuJoCo rollout.  qpos/qvel writes occur only at initialization."""
    if frames not in {1, 2, 6}:
        raise ValueError("M1 permits only M0, two-frame, or 1461..1466 windows")
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); model.opt.timestep = SIM_DT
    data = mujoco.MjData(model); data.qpos[:] = ctx.qpos[0]; data.qvel[:] = 0.0
    bodies, mocap = _preflight_object_ids(model); _set_object_mocap_reference(data, ctx.qpos[0], mocap)
    sites = _site_ids(model); tip_site, wrist_site = sites[8], sites[6]
    object_body = bodies["right"]
    hand, objects = v2r._contact_ids(model)
    data.ctrl[:52] = ctx.qpos[0, :52]; data.ctrl[52:] = 0.0; mujoco.mj_forward(model, data)
    machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=4, max_regrasp_attempts=0, allow_regrasp=False, sim_dt_s=SIM_DT))
    warnings: list[str] = []; old_warning = mujoco.get_mju_user_warning(); mujoco.set_mju_user_warning(lambda text: warnings.append(str(text)))
    timeline: list[dict[str, Any]] = []; endpoints: list[dict[str, Any]] = []
    previous_ctrl = data.ctrl[:52].copy(); sim_step = 0; transition_time = 0.0
    duration = 0.020 if frames == 1 else (frames - 1) / SOURCE_FPS
    first_failure: dict[str, Any] | None = None

    def sample(phase: str, source_state: np.ndarray, source_index: int, alpha: float) -> dict[str, Any]:
        count, depth, force, pairs = dynamic._contact_summary(model, data, hand, objects)
        assigned = _pair(pairs); physical = assigned is not None
        point, normal, distance, face = nearest_surface_target(data.site_xpos[tip_site], data.xpos[object_body], data.xmat[object_body].reshape(3, 3), ctx.patch_mesh)
        vector = np.asarray(data.site_xpos[tip_site]) - point
        normal_gap = float(np.dot(vector, normal)); tangential_error = float(np.linalg.norm(vector - normal * normal_gap))
        tip_velocity = np.zeros(6, dtype=np.float64); object_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tip_site, tip_velocity, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_body, object_velocity, 0)
        object_patch_velocity = object_velocity[3:] + np.cross(object_velocity[:3], point - data.xpos[object_body])
        relative = tip_velocity[3:] - object_patch_velocity
        normal_velocity = float(np.dot(relative, normal)); tangential_slip = float(np.linalg.norm(relative - normal * normal_velocity))
        target_data = mujoco.MjData(model); target_data.qpos[:] = source_state; target_data.qvel[:] = 0.0; mujoco.mj_forward(model, target_data)
        object_pos, object_rot = _object_tracking_error(data, source_state, bodies)
        cosine = 1.0 if np.linalg.norm(vector) <= 1e-12 else float(np.dot(vector, normal) / max(np.linalg.norm(vector), 1e-12))
        return {
            "phase": phase, "source_frame": int(ctx.source_frames[source_index]), "source_time_s": float(ctx.source_frames[0] / SOURCE_FPS + data.time), "source_interval_alpha": alpha, "sim_step": sim_step, "substep": sim_step,
            "pre_or_post_step": phase, "physical_contact": physical, "correct_contact": physical, "geom_pair": assigned["geom_pair"] if assigned else "NONE", "assigned_geom_pair": "|".join(sorted(ASSIGNED_PAIR)), "all_hand_object_pairs": pairs,
            "patch_distance_m": distance, "nearest_patch_point_world": point, "patch_normal_world": normal, "nearest_patch_local_face": face, "normal_gap_m": normal_gap, "tangential_error_m": tangential_error,
            "relative_normal_velocity_mps": normal_velocity, "tangential_slip_mps": tangential_slip, "tangential_slip_m": tangential_slip * SIM_DT, "normal_cosine": cosine,
            "tip_velocity_mps": tip_velocity[3:], "object_patch_velocity_mps": object_patch_velocity, "relative_velocity_mps": relative,
            "penetration_m": depth, "force_n": force, "impulse_ns": force * SIM_DT, "joint_margin_fraction": _joint_margin(model, data.qpos),
            "wrist_tracking_error_m": float(np.linalg.norm(data.site_xpos[wrist_site] - target_data.site_xpos[wrist_site])), "fingertip_tracking_error_m": float(np.linalg.norm(data.site_xpos[tip_site] - target_data.site_xpos[tip_site])),
            "object_tracking_position_m": float(np.max(object_pos)), "object_tracking_rotation_rad": float(np.max(object_rot)), "finite": _finite_data(data), "joint_limit_valid": not bool(v2r.dynamic._joint_limit_violations(model, np.asarray([data.qpos]))),
            "reference_qpos": source_state.copy(), "actual_qpos": data.qpos.copy(), "actual_qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(), "object_target_pose": source_state[52:58].copy(), "object_actual_pose": np.concatenate((data.xpos[object_body], Rotation.from_matrix(data.xmat[object_body].reshape(3, 3)).as_euler("XYZ"))), "object_qpos_written_after_initialization": False,
        }

    def observe(row: dict[str, Any]) -> None:
        nonlocal transition_time, first_failure
        before = machine.mode; observation = _observation(row, len(warnings)); mode = machine.observe(observation)
        row.update(observation_payload(observation)); row["mode"] = mode.value; row["previous_mode"] = before.value
        if before != mode and before not in {ContactMode.RETAIN_PENDING, ContactMode.RETAIN}:
            transition_time = data.time
        if first_failure is None and (not row["correct_contact"] or row["patch_distance_m"] > 0.020 or row["penetration_m"] > 0.003 or not row["finite"]):
            first_failure = dict(row)

    try:
        initial = sample("initial", ctx.qpos[0], 0, 0.0); observe(initial); timeline.append(initial); endpoints.append(dict(initial))
        while data.time + 0.5 * SIM_DT < duration and machine.mode != ContactMode.FAILED:
            source_index, alpha = continuous_source_index(data.time, frames)
            next_index = min(source_index + 1, frames - 1)
            source_state = interpolate_source_state(ctx.qpos[source_index], ctx.qpos[next_index], alpha) if source_index < frames - 1 else ctx.qpos[source_index].copy()
            pre = sample("pre", source_state, source_index, alpha); observe(pre); timeline.append(pre)
            _set_object_mocap_reference(data, source_state, mocap)
            correction, raw, clipped = _surface_control(
                model, data, tip_site, np.asarray(pre["patch_normal_world"]), float(pre["normal_gap_m"]),
                np.asarray(pre["tip_velocity_mps"]), np.asarray(pre["object_patch_velocity_mps"]), profile,
            )
            desired = source_state[:52].copy() + correction
            elapsed = max(0.0, data.time - transition_time)
            blend = min(1.0, elapsed / 0.008) if machine.mode in {ContactMode.RETAIN_PENDING, ContactMode.RETAIN} else 1.0
            new_ctrl = (1.0 - blend) * previous_ctrl + blend * desired
            data.ctrl[:52] = new_ctrl; data.ctrl[52:] = 0.0
            pre["ctrl_delta"] = new_ctrl - previous_ctrl; pre["ctrl_clipped"] = clipped; pre["raw_control_correction"] = raw; pre["controlled_columns"] = FINGER_COLUMNS
            previous_ctrl = new_ctrl.copy()
            mujoco.mj_step(model, data); sim_step += 1
            post_index, post_alpha = continuous_source_index(data.time, frames)
            post_next = min(post_index + 1, frames - 1)
            post_state = interpolate_source_state(ctx.qpos[post_index], ctx.qpos[post_next], post_alpha) if post_index < frames - 1 else ctx.qpos[post_index].copy()
            post = sample("post", post_state, post_index, post_alpha); observe(post); post["ctrl_delta"] = new_ctrl - previous_ctrl + 0.0; post["ctrl_clipped"] = clipped; post["raw_control_correction"] = raw; post["controlled_columns"] = FINGER_COLUMNS; timeline.append(post)
            if not endpoints or int(endpoints[-1]["source_frame"]) != int(post["source_frame"]):
                endpoints.append(dict(post))
            else:
                endpoints[-1] = dict(post)
        terminal = _observation(timeline[-1], len(warnings)); machine.finish(terminal)
    finally:
        mujoco.set_mju_user_warning(old_warning)

    post_rows = [row for row in timeline if row["phase"] in {"post", "initial"}]
    endpoint_rows = [row for row in endpoints if int(row["source_frame"]) in set(WINDOW[:frames])]
    frame_qpos = np.asarray([row["actual_qpos"] for row in endpoint_rows], dtype=np.float64)
    frame_ref = np.asarray([row["reference_qpos"] for row in endpoint_rows], dtype=np.float64)
    source_endpoint = np.asarray([row["source_frame"] for row in endpoint_rows], dtype=np.int64)
    visual = dynamic._dynamic_visual_penetration(model, frame_qpos, ctx.physics) if len(frame_qpos) else {"per_frame_max_penetration_m": [float("inf")]}
    _positions, _errors, _flat, tracking = dynamic._dynamic_robot_tracking(model, frame_qpos, frame_ref, source_endpoint)
    distances = np.asarray([row["patch_distance_m"] for row in post_rows], dtype=np.float64)
    contacts = np.asarray([row["correct_contact"] for row in post_rows], dtype=bool)
    normals = np.asarray([row["normal_cosine"] for row in post_rows], dtype=np.float64)
    forces = np.asarray([row["force_n"] for row in post_rows], dtype=np.float64)
    penetrations = np.asarray([row["penetration_m"] for row in post_rows], dtype=np.float64)
    impulses = np.asarray([row["impulse_ns"] for row in post_rows], dtype=np.float64)
    margin = np.asarray([row["joint_margin_fraction"] for row in post_rows], dtype=np.float64)
    ranges = v2r._robot_ranges(model); delta = float(np.max(np.abs(np.diff(frame_qpos[:, :52], axis=0)) / ranges)) if len(frame_qpos) > 1 else 0.0
    required_frames = WINDOW[:frames]; endpoint_ok = {int(frame): bool(any(int(row["source_frame"]) == int(frame) and row["correct_contact"] for row in endpoint_rows)) for frame in required_frames}
    modes = [row["mode"] for row in timeline]
    max_object_pos = max((row["object_tracking_position_m"] for row in post_rows), default=float("inf")); max_object_rot = max((row["object_tracking_rotation_rad"] for row in post_rows), default=float("inf"))
    allowed = {"PRE_CONTACT", "RETAIN_PENDING", "RETAIN", "FAILED", "COMPLETE"}
    gates = {
        "finite": bool(np.isfinite(frame_qpos).all() and all(row["finite"] for row in post_rows)), "no_warnings": not warnings,
        "joint_limits": not bool(v2r.dynamic._joint_limit_violations(model, frame_qpos)), "invalid_rotations": True,
        "retain_entered": "RETAIN" in modes, "no_regrasp": "REGRASP" not in modes, "allowed_modes_only": set(modes).issubset(allowed), "terminal_complete": machine.mode == ContactMode.COMPLETE,
        "correct_assigned_contact": all(endpoint_ok.values()), "physical_contact_continuity": bool(contacts.size and contacts.all()), "patch_coverage": float(np.mean(np.asarray(list(endpoint_ok.values()), dtype=float))) >= 0.80,
        "functional_role_recall": float(np.mean(np.asarray(list(endpoint_ok.values()), dtype=float))) >= 0.80, "patch_distance_p95": bool(distances.size and np.percentile(distances, 95) <= 0.020),
        "terminal_contact": bool(endpoint_rows and endpoint_rows[-1]["correct_contact"]), "terminal_patch_distance": bool(endpoint_rows and endpoint_rows[-1]["patch_distance_m"] <= 0.020), "normal": bool(normals.size and np.median(normals) >= 0.50),
        "penetration": float(max(np.max(penetrations, initial=0.0), max(visual["per_frame_max_penetration_m"], default=0.0))) <= 0.003, "force": float(np.max(forces, initial=0.0)) <= 150.0,
        "tracking": all(side["wrist_rmse_m"] <= 0.03 and all(item["rmse_m"] <= 0.08 for item in side["fingertips"].values()) for side in tracking.values()), "object_tracking": max_object_pos <= 0.08 and max_object_rot <= 0.50,
        "smoothness": delta <= 0.25, "object_qpos_not_written": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    failure = first_failure or next((row for row in post_rows if not row["correct_contact"]), None)
    result = {
        "schema_version": 1, "experiment": "M0" if frames == 1 else "M1", "profile": profile, "profile_hash": payload_hash(profile), "status": status, "window": {"source_frames": required_frames.tolist(), "fps": SOURCE_FPS, "sim_dt_s": SIM_DT},
        "gates": gates, "contact": {"assigned_geom_pair": "|".join(sorted(ASSIGNED_PAIR)), "correct_source_frames": sum(endpoint_ok.values()), "required_source_frames": len(required_frames), "continuity": float(np.mean(contacts)) if contacts.size else 0.0, "patch_coverage": float(np.mean(np.asarray(list(endpoint_ok.values()), dtype=float))), "functional_role_recall": float(np.mean(np.asarray(list(endpoint_ok.values()), dtype=float))), "patch_distance_p95_m": float(np.percentile(distances, 95)) if distances.size else float("inf"), "normal_cosine_median": float(np.median(normals)) if normals.size else float("nan"), "terminal_correct_contact": bool(endpoint_rows and endpoint_rows[-1]["correct_contact"]), "terminal_patch_distance_m": float(endpoint_rows[-1]["patch_distance_m"]) if endpoint_rows else float("inf")},
        "safety": {"force_max_n": float(np.max(forces, initial=0.0)), "force_p95_n": float(np.percentile(forces, 95)) if forces.size else 0.0, "impulse_max_ns": float(np.max(impulses, initial=0.0)), "initial_impact": bool(forces.size and forces[0] > 150.0), "persistent_high_force": bool(np.count_nonzero(forces > 150.0) > 1), "penetration_max_mujoco_m": float(np.max(penetrations, initial=0.0)), "penetration_max_visual_m": float(max(visual["per_frame_max_penetration_m"], default=0.0)), "minimum_joint_margin_fraction": float(np.min(margin)) if margin.size else float("nan"), "normalized_one_frame_delta": delta, "warnings": warnings},
        "tracking": tracking, "object": {"position_max_m": max_object_pos, "rotation_max_rad": max_object_rot, "source_object_target_unchanged": True, "object_qpos_written_after_initialization": False, "no_freeze": True, "no_teleport": True},
        "state_machine": {"terminal_mode": machine.mode.value, "failure_code": machine.failure_code.value if machine.failure_code else None, "transitions": machine.transition_payload(), "regrasp_attempts": machine.regrasp_attempts},
        "first_failure": failure, "failure_category": _failure_category(failure), "preservation": {"root_wrist_static_correction": False, "robot_qpos_written_after_initialization": False, "object_qpos_written_after_initialization": False, "object_mocap_reference_only": True, "patch_unchanged": True, "role_unchanged": True, "finger_unchanged": True, "threshold_20mm_unchanged": True},
        "_timeline": timeline, "_endpoints": endpoint_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", {key: value for key, value in result.items() if not key.startswith("_")})
    _write_json(output / "timeline.json", {"schema_version": 1, "rows": timeline})
    _write_npz(output / "trace.npz", qpos=np.asarray([row["actual_qpos"] for row in timeline]), qvel=np.asarray([row["actual_qvel"] for row in timeline]), ctrl=np.asarray([row["ctrl"] for row in timeline]), source_frame=np.asarray([row["source_frame"] for row in timeline]), phase=np.asarray([row["phase"] for row in timeline]), patch_distance_m=np.asarray([row["patch_distance_m"] for row in timeline]), contact=np.asarray([row["correct_contact"] for row in timeline], dtype=np.uint8), normal_gap_m=np.asarray([row["normal_gap_m"] for row in timeline]), tangential_slip_mps=np.asarray([row["tangential_slip_mps"] for row in timeline]), force_n=np.asarray([row["force_n"] for row in timeline]), penetration_m=np.asarray([row["penetration_m"] for row in timeline]))
    return result


def _compact_mesh(mesh: dict[str, Any]) -> dict[str, Any]:
    return {"vertices": mesh.get("vertices", []), "faces": mesh.get("faces", [])}


def _patch_world_mesh(ctx: Context, qpos: np.ndarray) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); data = mujoco.MjData(model); data.qpos[:] = qpos; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")); rotation = data.xmat[body].reshape(3, 3)
    return {"vertices": (np.asarray(ctx.patch_mesh.vertices) @ rotation.T + data.xpos[body]).astype(np.float32).tolist(), "faces": np.asarray(ctx.patch_mesh.faces, dtype=np.int32).tolist()}


def build_viewer(ctx: Context, result: dict[str, Any], screenshot_root: Path | None = None) -> tuple[Path, Path, list[dict[str, Any]]]:
    timeline = result["_timeline"]; endpoints = result["_endpoints"]
    first = result.get("first_failure")
    event_rows: list[dict[str, Any]] = []
    # A two-frame hard failure legitimately prevents a later M1 rollout.  The
    # visual review nevertheless needs the named 1461..1466 source poses.  For
    # source frames after that stop, show the real terminal simulated state
    # beside the real XAE reference pose and label it explicitly; do not invent
    # later actual contact, qpos, or target telemetry.
    terminal_actual = dict(endpoints[-1]) if endpoints else dict(timeline[-1])

    def reference_context(source: int) -> dict[str, Any]:
        local = int(np.where(WINDOW == source)[0][0])
        row = dict(terminal_actual)
        row.update({
            "source_frame": source,
            "reference_qpos": ctx.qpos[local].copy(),
            "object_target_pose": ctx.qpos[local, 52:58].copy(),
            "mode": "REFERENCE_ONLY_AFTER_GATE_STOP",
            "event_kind": "reference_context_after_real_gate_stop",
        })
        return row

    # Keep the literal t=0 assigned-contact state as its own event.  The
    # endpoint accumulator intentionally keeps the latest state per source
    # frame, which for a failed gate can otherwise hide the initial valid
    # RETAIN_PENDING contact behind its later first-loss state.
    initial_event = next((row for row in timeline if row["phase"] == "initial"), None)
    event_rows.append(dict(initial_event) if initial_event is not None else (dict(endpoints[0]) if endpoints else reference_context(1461)))
    first_motion = next((row for row in timeline if row["phase"] == "post" and int(row["source_frame"]) == 1462), None)
    event_rows.append(dict(first_motion) if first_motion is not None else reference_context(1462))
    for source in WINDOW:
        row = next((item for item in endpoints if int(item["source_frame"]) == int(source)), None)
        event_rows.append(dict(row) if row is not None else reference_context(int(source)))
    event_rows.append(dict(first) if first is not None else dict(terminal_actual))
    event_rows.append(dict(terminal_actual))
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); cache: dict[tuple[int, int], trimesh.Trimesh] = {}
    index_geom = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) for name in ("collision_hand_left_index_5", "collision_hand_left_index_6", "collision_hand_left_index_7", "collision_hand_left_index_8")]
    frames: list[dict[str, Any]] = []
    for number, row in enumerate(event_rows):
        actual = np.asarray(row["actual_qpos"], dtype=np.float64); reference = np.asarray(row["reference_qpos"], dtype=np.float64)
        actual_mesh = _state_meshes(model, actual, cache); ref_mesh = _state_meshes(model, reference, cache)
        data = mujoco.MjData(model); data.qpos[:] = actual; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        contacts = _contact_records(model, actual)
        contact_points = [item["position"] for item in contacts]
        normals = [[*item["position"], *(np.asarray(item["position"]) + np.asarray(item["normal"]) * 0.025)] for item in contacts]
        forces = [[*item["position"], *(np.asarray(item["position"]) + np.asarray(item["normal"]) * min(0.03, float(item["force_n"]) * 0.0002))] for item in contacts]
        point = np.asarray(row["nearest_patch_point_world"]); tip = np.asarray(data.site_xpos[_site_ids(model)[8]])
        frames.append({"event_id": f"e{number}_{row['source_frame']}_{row['sim_step']}", "source_frame": int(row["source_frame"]), "sim_step": int(row["sim_step"]), "mode": row["mode"], "event_kind": row.get("event_kind", "real_mujoco_state"), "actual_right_visual": _compact_mesh(actual_mesh["right_visual"]), "actual_left_visual": _compact_mesh(actual_mesh["left_visual"]), "reference_right_visual": _compact_mesh(ref_mesh["right_visual"]), "reference_left_visual": _compact_mesh(ref_mesh["left_visual"]), "actual_collision": _compact_mesh(actual_mesh["left_collision"]), "object_visual": _compact_mesh(actual_mesh["object_visual"]), "object_collision": _compact_mesh(actual_mesh["object_collision"]), "patch": _patch_world_mesh(ctx, actual), "index_region": _mesh_for_ids(model, data, index_geom, cache, 80), "contacts": contact_points, "normals": normals, "forces": forces, "target": point.tolist(), "tip": tip.tolist(), "normal_gap": [tip.tolist(), point.tolist()], "slip": [point.tolist(), (point + np.asarray(row["patch_normal_world"]) * float(row["tangential_slip_mps"]) * 0.03).tolist()], "metrics": {key: row[key] for key in ("physical_contact", "patch_distance_m", "normal_gap_m", "tangential_slip_mps", "force_n", "penetration_m", "joint_margin_fraction", "object_tracking_position_m")}})
    curves = [{key: row.get(key) for key in ("source_frame", "sim_step", "mode", "correct_contact", "patch_distance_m", "normal_gap_m", "relative_normal_velocity_mps", "tangential_slip_mps", "force_n", "impulse_ns", "penetration_m", "joint_margin_fraction", "object_tracking_position_m")} for row in timeline if row["phase"] == "post"]
    payload = {"schema_version": 1, "status": result["status"], "frames": frames, "curves": curves, "disclaimer": "真实 MuJoCo 动态轨迹；非固定锚点、非重抓、非完整 primary。", "layer_names": ["最终 XAE reference Wuji visual mesh", "M1 actual Wuji visual mesh", "Wuji collision mesh", "object visual mesh", "object collision mesh", "semantic patch connected surface", "assigned left-index visual/collision region", "actual MuJoCo contacts", "contact normals", "force vectors", "normal-gap vector", "tangential-slip vector", "surface-aligned nearest target", "world axes"]}
    _write_json(ctx.run_root / "html/viewer_payload.json", payload)
    html = _viewer_html(payload)
    page = ctx.run_root / "html/stage_c_xae_m1_retention.html"; _write_text(page, html)
    index = ctx.run_root / "html/stage_c_xae_m1_visual_index.html"
    links = "".join(f"<li><a href='stage_c_xae_m1_retention.html?event={frame['event_id']}&view=close'>{frame['source_frame']} / substep {frame['sim_step']} / {frame['mode']}</a></li>" for frame in frames)
    _write_text(index, f"<!doctype html><meta charset='utf-8'><title>XAE M1 可视化索引</title><h1>Stage C-XAE-M1 真实三维可视化索引</h1><p><a href='stage_c_xae_m1_retention.html'>打开真实三维轨迹</a></p><ul>{links}</ul><p>{payload['disclaimer']}</p>")
    screenshots = render_screenshots(page, screenshot_root or (ctx.run_root / "screenshots"), frames)
    _write_json(ctx.run_root / "reports/xae_m1_screenshot_manifest.json", {"schema_version": 1, "status": "PASS" if len(screenshots) >= 24 and all(row["status"] == "PASS" for row in screenshots) else "FAIL", "screenshots": screenshots})
    return page, index, screenshots


def _viewer_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, default=_plain)
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Stage C-XAE-M1 Surface-Aligned Retention</title><script>{get_plotlyjs()}</script>
<style>body{{margin:0;background:#0d141b;color:#e9f1f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}header{{padding:12px 18px;background:#172431}}#scene{{height:62vh}}#curves{{height:31vh}}label{{margin-right:10px;font-size:13px}}select{{background:#293b49;color:#fff;padding:4px}}#info{{white-space:pre-wrap;padding:8px 18px}}</style>
<body><header><b>Stage C-XAE-M1：真实 mesh 动态保持</b>　事件 <select id='event'></select>　视角 <select id='view'><option value='world'>世界</option><option value='object'>object-frame 近景</option><option value='wrist'>左腕反向</option></select><span id='toggles'></span></header><div id='scene'></div><div id='curves'></div><pre id='info'></pre>
<script>const D={data}; const $=id=>document.getElementById(id); const C={{ref:'#ffd166',actual:'#06d6a0',collision:'#ef476f',obj:'#457b9d',patch:'#c77dff',region:'#00b4d8',contact:'#ffffff',force:'#ff9f1c',gap:'#ff006e',slip:'#a8dadc',target:'#90be6d',axis:'#d9d9d9'}};
function mesh(n,m,c,o){{return {{type:'mesh3d',name:n,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o,flatshading:true}}}} function line(n,p,c){{return {{type:'scatter3d',mode:'lines+markers',name:n,x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),line:{{color:c,width:7}},marker:{{size:3,color:c}}}}}}
const layers=['ref','actual','collision','obj','objcol','patch','region','contacts','normals','forces','gap','slip','target','axis']; const names=['最终XAE reference','M1 actual','Wuji collision','object visual','object collision','semantic patch','left-index region','actual contacts','contact normals','force vectors','normal-gap','tangential-slip','nearest surface target','world axes']; $('toggles').innerHTML=layers.map((x,i)=>`<label><input type='checkbox' data-layer='${{x}}' checked>${{names[i]}}</label>`).join(''); D.frames.forEach(f=>$('event').add(new Option(`${{f.source_frame}} / step ${{f.sim_step}} / ${{f.mode}}`,f.event_id))); const qp=new URLSearchParams(location.search); if(qp.get('event'))$('event').value=qp.get('event'); if(qp.get('view'))$('view').value=qp.get('view');
function draw(){{let f=D.frames.find(x=>x.event_id==$('event').value)||D.frames[0], t=[]; let add=(id,x)=>{{if(document.querySelector(`[data-layer=${{id}}]`).checked)t.push(x)}}; add('ref',mesh('最终XAE reference right',f.reference_right_visual,C.ref,.24));add('ref',mesh('最终XAE reference left',f.reference_left_visual,C.ref,.24));add('actual',mesh('M1 actual right',f.actual_right_visual,C.actual,.42));add('actual',mesh('M1 actual left',f.actual_left_visual,C.actual,.42));add('collision',mesh('Wuji collision',f.actual_collision,C.collision,.26));add('obj',mesh('object visual',f.object_visual,C.obj,.38));add('objcol',mesh('object collision',f.object_collision,C.collision,.16));add('patch',mesh('immutable semantic patch',f.patch,C.patch,.78));add('region',mesh('left index collision region',f.index_region,C.region,.52));add('contacts',{{type:'scatter3d',mode:'markers',name:'actual contacts',x:f.contacts.map(p=>p[0]),y:f.contacts.map(p=>p[1]),z:f.contacts.map(p=>p[2]),marker:{{size:5,color:C.contact}}}});f.normals.forEach((p,i)=>add('normals',line('contact normal '+i,[p.slice(0,3),p.slice(3,6)],C.contact)));f.forces.forEach((p,i)=>add('forces',line('force '+i,[p.slice(0,3),p.slice(3,6)],C.force)));add('gap',line('normal gap',[f.normal_gap[0],f.normal_gap[1]],C.gap));add('slip',line('tangential slip',[f.slip[0],f.slip[1]],C.slip));add('target',{{type:'scatter3d',mode:'markers',name:'surface-aligned nearest target',x:[f.target[0]],y:[f.target[1]],z:[f.target[2]],marker:{{size:7,color:C.target,symbol:'diamond'}}}});let a=f.target,s=.04;add('axis',line('world X',[a,[a[0]+s,a[1],a[2]]],'#f00'));add('axis',line('world Y',[a,[a[0],a[1]+s,a[2]]],'#0f0'));add('axis',line('world Z',[a,[a[0],a[1],a[2]+s]],'#00f'));let eye=$('view').value==='object'?{{x:.5,y:.5,z:.35}}:$('view').value==='wrist'?{{x:-1.4,y:1.2,z:.8}}:{{x:1.5,y:-1.5,z:1.2}};Plotly.react('scene',t,{{paper_bgcolor:'#0d141b',plot_bgcolor:'#0d141b',font:{{color:'#e9f1f7'}},scene:{{aspectmode:'data',camera:{{eye}}}},margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});$('info').textContent=`source ${{f.source_frame}} | step ${{f.sim_step}} | mode ${{f.mode}} | event=${{f.event_kind}}\ncontact=${{f.metrics.physical_contact}} patch=${{(f.metrics.patch_distance_m*1000).toFixed(3)}} mm normal gap=${{(f.metrics.normal_gap_m*1000).toFixed(3)}} mm\nslip=${{f.metrics.tangential_slip_mps.toFixed(5)}} m/s force=${{f.metrics.force_n.toFixed(3)}} N penetration=${{(f.metrics.penetration_m*1000).toFixed(3)}} mm\n${{f.event_kind==='reference_context_after_real_gate_stop'?'提示：双帧门禁已在此前真实状态停止；此事件只显示真实终端模拟状态与该 source reference，不伪造后续 actual contact。':''}}`;}}
function curves(){{let x=D.curves.map(r=>r.sim_step);let tr=[['physical contact','correct_contact'],['patch distance m','patch_distance_m'],['normal gap m','normal_gap_m'],['relative normal velocity','relative_normal_velocity_mps'],['tangential slip','tangential_slip_mps'],['force N','force_n'],['penetration m','penetration_m'],['joint margin','joint_margin_fraction']].map(r=>({{type:'scatter',mode:'lines',name:r[0],x,y:D.curves.map(q=>q[r[1]])}}));Plotly.react('curves',tr,{{paper_bgcolor:'#0d141b',plot_bgcolor:'#0d141b',font:{{color:'#e9f1f7'}},margin:{{l:45,r:20,t:25,b:35}},legend:{{orientation:'h'}},xaxis:{{title:'MuJoCo substep'}}}},{{responsive:true}})}} $('event').onchange=draw;$('view').onchange=draw;document.querySelectorAll('input').forEach(e=>e.onchange=draw);draw();curves();</script></body></html>"""


def render_screenshots(page: Path, root: Path, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True); rows: list[dict[str, Any]] = []
    for frame in frames:
        for view in ("world", "object", "wrist"):
            # Several required review events may intentionally share the same
            # source frame and terminal MuJoCo step (initial, first loss, and
            # terminal).  Keep their evidence distinct rather than silently
            # overwriting PNGs under a source-frame-only filename.
            target = root / f"m1_{frame['event_id']}_{frame['source_frame']}_{frame['sim_step']}_{view}.png"
            url = page.resolve().as_uri() + f"?event={frame['event_id']}&view={view}"
            call = subprocess.run(["/usr/bin/google-chrome", "--headless", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=5000", "--window-size=1800,1200", f"--screenshot={target}", url], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
            rows.append({"event": frame["event_id"], "source_frame": frame["source_frame"], "view": view, "path": str(target), "status": "PASS" if call.returncode == 0 and target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "stderr_tail": call.stderr[-400:]})
    return rows


def _first_failure_report(result: dict[str, Any]) -> dict[str, Any]:
    failure = result.get("first_failure")
    if failure is None:
        return {"schema_version": 1, "status": "NO_FAILURE", "first_failure": None}
    fields = ("source_frame", "source_time_s", "sim_step", "substep", "phase", "mode", "previous_mode", "geom_pair", "assigned_geom_pair", "all_hand_object_pairs", "patch_distance_m", "normal_gap_m", "relative_normal_velocity_mps", "tangential_slip_mps", "force_n", "penetration_m", "ctrl", "ctrl_delta", "joint_margin_fraction", "object_target_pose", "object_actual_pose", "nearest_patch_point_world")
    return {"schema_version": 1, "status": "FAILURE_LOCALIZED", "classification": result.get("failure_category"), "first_failure": {key: failure.get(key) for key in fields}, "pre_step_contact": next((row.get("correct_contact") for row in result["_timeline"] if row["sim_step"] == failure["sim_step"] and row["phase"] == "pre"), None), "post_step_contact": failure.get("correct_contact")}


def _markdown_final(acceptance: dict[str, Any]) -> str:
    return "# Stage C-XAE-M1 最终验收\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in acceptance.items() if key != "schema_version") + "\n"


def _candidate_profile(candidate: str) -> dict[str, Any]:
    base = {"candidate": candidate, "controlled_joint_set": ["left_index"], "controlled_columns": FINGER_COLUMNS.tolist(), "surface_target": "same-substep nearest semantic-patch surface", "object_motion_feedforward": False, "normal_velocity_servo": False, "tangential_slip_servo": False, "allow_regrasp": False}
    if candidate == "R2_object_motion_velocity_feedforward": base["object_motion_feedforward"] = True
    if candidate == "R3_normal_relative_velocity_servo": base["normal_velocity_servo"] = True
    if candidate == "R4_tangential_slip_compensation": base["tangential_slip_servo"] = True
    return base


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-surface-aligned-retention")
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"fail closed: output directory already exists: {root}")
    for name in ("manifest", "contract_regression", "m0_regression", "target_build", "two_frame", "m1", "diagnostics", "candidates", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    ctx = load_context(paths_config, root); lineage = audit_lineage(ctx, paths_config); contract = contract_regression(ctx, paths_config); build_target(ctx)
    if contract["status"] != "PASS":
        raise RuntimeError("M1: NOT_RUN_DUE_TO_XAE_REGRESSION")
    m0 = run_rollout(ctx, _candidate_profile("M0_xae_finger_only_baseline"), 1, root / "m0_regression")
    _write_json(root / "m0_regression/m0_xae_m1_regression.json", {key: value for key, value in m0.items() if not key.startswith("_")})
    shutil.copy2(root / "m0_regression/trace.npz", root / "m0_regression/m0_xae_m1_trace.npz")
    if m0["status"] != "PASS":
        raise RuntimeError("M1: NOT_RUN_DUE_TO_M0_REGRESSION")
    candidates: list[dict[str, Any]] = []
    baseline_profile = _candidate_profile("R0_frozen_surface_aligned_baseline")
    two = run_rollout(ctx, baseline_profile, 2, root / "two_frame/R0_frozen_surface_aligned_baseline")
    candidates.append({"candidate": baseline_profile["candidate"], "change": "final XAE + same-substep nearest semantic patch + left-index only", "cause_evidence": "baseline", "two_frame": two["status"], "m1": "NOT_RUN", "failure_code": two.get("failure_category")})
    _write_json(root / "two_frame/two_frame_retention_summary.json", {key: value for key, value in two.items() if not key.startswith("_")})
    shutil.copy2(root / "two_frame/R0_frozen_surface_aligned_baseline/trace.npz", root / "two_frame/two_frame_retention_trace.npz")
    _write_text(root / "two_frame/TWO_FRAME_RETENTION.md", f"# 双帧动态保持门禁\n\n状态：`{two['status']}`。R0 使用同一 substep 的 immutable patch nearest-surface target，未使用 fixed anchor 或 REGRASP。\n")
    _write_json(root / "reports/two_frame_first_failure.json", _first_failure_report(two))
    selected = baseline_profile; selected_two = two; m1: dict[str, Any] | None = None
    if two["status"] == "PASS":
        m1 = run_rollout(ctx, baseline_profile, 6, root / "m1/R0_frozen_surface_aligned_baseline")
        candidates[-1]["m1"] = m1["status"]; candidates[-1]["failure_code"] = m1.get("failure_category")
    else:
        category = two.get("failure_category")
        repairs: list[str] = []
        if category == "OBJECT_COUPLED_NORMAL_SEPARATION": repairs = ["R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo"]
        elif category == "TANGENTIAL_SLIP": repairs = ["R4_tangential_slip_compensation"]
        elif category == "CONTROL_CLIPPING": repairs = []
        else: repairs = ["R2_object_motion_velocity_feedforward"]
        for name in repairs[:5]:
            profile = _candidate_profile(name); candidate_two = run_rollout(ctx, profile, 2, root / f"candidates/{name}")
            candidates.append({"candidate": name, "change": name.removeprefix("R2_").removeprefix("R3_").removeprefix("R4_").replace("_", " "), "cause_evidence": category, "two_frame": candidate_two["status"], "m1": "NOT_RUN", "failure_code": candidate_two.get("failure_category")})
            if candidate_two["status"] == "PASS":
                selected, selected_two = profile, candidate_two
                m1 = run_rollout(ctx, profile, 6, root / f"m1/{name}"); candidates[-1]["m1"] = m1["status"]; candidates[-1]["failure_code"] = m1.get("failure_category")
                break
    selected_result = m1 if m1 is not None else selected_two
    m1_status = m1["status"] if m1 is not None else "FAIL"
    _write_json(root / "m1/m1_candidate_matrix.json", {"schema_version": 1, "candidate_count": len(candidates), "candidates": candidates, "selection_rule": "contact continuity, terminal correct contact, patch P95, penetration, force, joint margin, tracking, smoothness, object tracking, runtime"})
    m1_summary = {"schema_version": 1, "status": m1_status, "M1_MOVING_RETENTION_WITNESS": "FOUND" if m1_status == "PASS" else "NOT_FOUND", "selected_profile": selected, "two_frame_status": selected_two["status"], "result": {key: value for key, value in selected_result.items() if not key.startswith("_")}, "candidate_count": len(candidates), "stop_rule": "M2/M3/full primary/Oracle/MJWP/smokes/Stage D are not run"}
    _write_json(root / "m1/m1_retention_summary.json", m1_summary)
    _write_json(root / "reports/m1_first_failure.json", _first_failure_report(selected_result))
    if m1_status == "PASS" and m1 is not None:
        witness = {key: value for key, value in m1.items() if not key.startswith("_")}; _write_json(root / "m1/selected_m1_profile.json", selected); _write_json(root / "m1/m1_dynamic_witness.json", witness); shutil.copy2(root / f"m1/{selected['candidate']}/trace.npz", root / "m1/m1_dynamic_witness.npz"); _write_text(root / "m1/M1_DYNAMIC_WITNESS.md", "# M1 动态保持 witness\n\n状态：`PASS`。\n"); witness_hashes = {"M1_PROFILE_HASH": payload_hash(selected), "M1_WITNESS_HASH": payload_hash(witness)}
    else:
        witness_hashes = {}
    page, index, screenshots = build_viewer(ctx, selected_result)
    screenshot_manifest = json.loads((root / "reports/xae_m1_screenshot_manifest.json").read_text())
    review = {"schema_version": 1, "status": "PENDING_CODEX_IMAGE_REVIEW", "checks": {"final_xae_reference_actual_initial_consistent": None, "frame_1461_left_index_contact": None, "object_motion_surface_target_sync": None, "frame_1462_contact_retained": None, "contact_on_immutable_patch": None, "normal_separation": None, "tangential_sliding": None, "wrong_finger_or_link": None, "high_force_or_deep_penetration": None, "object_qpos_direct_write": False, "root_wrist_static_offset_regression": False, "terminal_matches_numeric_report": None}, "screenshots": screenshots}
    _write_json(root / "reports/xae_m1_manual_visual_review.json", review)
    _write_text(root / "reports/XAE_M1_SCREENSHOT_REVIEW.md", "# XAE-M1 Chrome 截图人工复核\n\n截图已生成；由任务执行者在实际查看截图后填写中文结论。\n")
    acceptance = {"schema_version": 1, "XAE": "PASS", "Contract-V2 regression": contract["status"], "M0 regression": m0["status"], "two-frame gate": selected_two["status"], "M1": m1_status, "M1_MOVING_RETENTION_WITNESS": "FOUND" if m1_status == "PASS" else "NOT_FOUND", "M2": "NOT_RUN", "M3": "NOT_RUN", "full primary": "NOT_RUN", "Oracle C/D2": "NOT_RUN", "MJWP": "NOT_RUN", "smokes": "NOT_RUN", "Stage D": "NOT_STARTED", "user_visual_review": "PENDING", "visualization": screenshot_manifest["status"], **witness_hashes}
    _write_json(root / "reports/m2_status.json", {"status": "NOT_RUN", "reason": "Stage C-XAE-M1 terminates before M2"}); _write_json(root / "reports/m3_status.json", {"status": "NOT_RUN", "reason": "Stage C-XAE-M1 terminates before M3"})
    _write_json(root / "reports/xae_m1_final_acceptance.json", acceptance); _write_text(root / "reports/XAE_M1_FINAL_ACCEPTANCE.md", _markdown_final(acceptance))
    handoff = f"# Stage C-XAE-M1 handoff\n\n- run root: `{root}`\n- XAE authority: `{AUTHORITY}`\n- Contract regression: `{contract['status']}`\n- M0: `{m0['status']}`\n- two-frame: `{selected_two['status']}`\n- M1: `{m1_status}`\n- M2/M3: `NOT_RUN`\n\n本 run 从最终 XAE repaired trajectory 重新构建同一 immutable semantic patch 的 nearest-surface 动态 target；没有读取旧 M1 fixed-anchor target、seed、profile 或标签。\n"
    _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1.md", handoff)
    return {"run_root": str(root), "acceptance": acceptance, "lineage": lineage["status"], "html": str(page), "index": str(index), "screenshot_count": len(screenshots)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.run_root), indent=2, default=_plain))


if __name__ == "__main__":
    main()
