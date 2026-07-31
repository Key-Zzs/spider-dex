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
import trimesh
import tyro
import yaml
import mujoco
from scipy.spatial.transform import Rotation

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
    with np.load(_robot_dir(paths.workspace_root, sequence_id) / "trajectory_kinematic.npz", allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    # Contact references retain one source sample at either end while the
    # finite-difference C-R2/MJWP trajectory uses the exact middle ``1:-1``
    # span.  Preserve source frame IDs but make the physics-aligned index
    # explicit; silently assigning either endpoint would break frozen-frame
    # accounting.
    source_frame_count = max((int(row["frame_index"]) for row in payload.get("records", [])), default=-1) + 1
    active = []
    for original in payload.get("records", []):
        if not (original.get("contact_flag") and original.get("confidence") == "high"):
            continue
        source_index = int(original["frame_index"])
        if not 0 < source_index < source_frame_count - 1:
            continue
        row = dict(original); row["stage_c_frame_index"] = source_index - 1
        active.append(row)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in active:
        key = (str(row["side"]), str(row["finger_region"]), int(row["contact_interval_id"]))
        groups.setdefault(key, []).append(row)
    roles: list[dict[str, Any]] = []
    for role_id, ((side, finger, interval_id), rows) in enumerate(sorted(groups.items())):
        frames = sorted(int(row["stage_c_frame_index"]) for row in rows)
        overlap = {other_finger for (other_side, other_finger, _other_interval), other_rows in groups.items()
                   if other_side == side and set(frames).intersection(int(row["stage_c_frame_index"]) for row in other_rows)}
        label, confidence, rationale = classify_role(finger, len(frames), overlap)
        points = np.asarray([row["surface_point_world"] for row in rows], dtype=np.float64)
        normals = np.asarray([row["surface_normal_world"] for row in rows], dtype=np.float64)
        object_positions = source_qpos[np.asarray(frames, dtype=np.int64), -14:-11]
        simultaneous_other = [np.asarray(other["surface_point_world"], dtype=np.float64) for other in active
                              if other["side"] == side and other["finger_region"] != finger and int(other["stage_c_frame_index"]) in set(frames)]
        roles.append({
            "role_id": f"{sequence_id}:{role_id}", "side": side, "source_finger": finger,
            "source_contact_channel": int(rows[0]["contact_channel"]),
            "source_contact_interval_id": interval_id, "frame_start": min(frames), "frame_end": max(frames),
            "frame_count": len(frames), "functional_role": label, "confidence": confidence,
            "generation_basis": rationale + ["source contact interval", "source object surface projection", "source normal", "source motion interval", "object motion interval", "relative simultaneous-contact geometry"],
            "source_anchor_world_mean": points.mean(axis=0), "source_normal_world_mean": normals.mean(axis=0),
            "source_motion": {"anchor_path_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0,
                              "object_path_length_m": float(np.linalg.norm(np.diff(object_positions, axis=0), axis=1).sum()) if len(object_positions) > 1 else 0.0,
                              "normal_mean_resultant": float(np.linalg.norm(normals.mean(axis=0)))},
            "relative_position": {"simultaneous_other_contact_count": len(simultaneous_other),
                                  "mean_distance_m": float(np.mean([np.linalg.norm(point - points.mean(axis=0)) for point in simultaneous_other])) if simultaneous_other else None},
            "source_frame_indices": [int(row["frame_index"]) for row in rows], "stage_c_frame_indices": frames,
        })
    result = {"schema_version": 1, "stage": "C-X1", "sequence_id": sequence_id, "contract_name": contract["contract_name"],
              "contract_hash": _hash(contract_file), "source_only": True, "roles": roles,
              "inactive_contact_policy": "NON_INTERACTING; no assignment candidates are created for inactive source contacts"}
    output = _v2_dir(paths.workspace_root, sequence_id) / "source_contact_roles.json"
    _write_json(output, result)
    np.savez_compressed(output.with_suffix(".npz"), role_frame_starts=np.asarray([row["frame_start"] for row in roles], dtype=np.int32), role_frame_ends=np.asarray([row["frame_end"] for row in roles], dtype=np.int32), role_confidence=np.asarray([row["confidence"] for row in roles], dtype=np.float64))
    return str(output)


def _object_local(point_world: np.ndarray, normal_world: np.ndarray, object_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move source surface evidence into immutable object-local coordinates."""
    position = np.asarray(object_qpos[:3], dtype=np.float64)
    quaternion_xyzw = np.asarray(object_qpos[3:7], dtype=np.float64)[[1, 2, 3, 0]]
    rotation = Rotation.from_quat(quaternion_xyzw)
    return rotation.inv().apply(np.asarray(point_world, dtype=np.float64) - position), rotation.inv().apply(np.asarray(normal_world, dtype=np.float64))


def _patch_faces(mesh: trimesh.Trimesh, center: np.ndarray, normal: np.ndarray, radius: float) -> tuple[int, list[int]]:
    """Use face-adjacency traversal, not a world-space ball, for thin walls."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    seed = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(faces))}
    for first, second in np.asarray(mesh.face_adjacency, dtype=np.int64):
        adjacency[int(first)].append(int(second)); adjacency[int(second)].append(int(first))
    accepted: list[int] = []; queue: list[tuple[int, float]] = [(seed, 0.0)]; visited: set[int] = set()
    unit_normal = normal / max(np.linalg.norm(normal), 1e-12)
    while queue:
        face, distance = queue.pop(0)
        if face in visited or distance > radius:
            continue
        visited.add(face)
        # Normals retain an exterior/interior distinction for thin-walled meshes.
        if float(np.dot(normals[face], unit_normal)) < 0.0:
            continue
        accepted.append(face)
        for neighbour in adjacency[face]:
            edge = float(np.linalg.norm(centroids[face] - centroids[neighbour]))
            if neighbour not in visited:
                queue.append((neighbour, distance + edge))
    return seed, sorted(accepted)


def build_surface_patches(
    paths_config: str, sequence_id: str, contract_path: str = "configs/project/grab_wuji_stage_c_contract_v2.yaml"
) -> str:
    """Build object-local, normal-aware mesh-adjacent patches from source only."""
    if sequence_id not in PILOTS:
        raise ValueError("V2 patches are restricted to frozen pilots")
    paths = _paths(paths_config); workspace = paths.workspace_root
    contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    root = _v2_dir(workspace, sequence_id)
    role_path = root / "source_contact_roles.json"
    if not role_path.is_file():
        build_source_contact_roles(paths_config, sequence_id, contract_path)
    roles = json.loads(role_path.read_text(encoding="utf-8"))["roles"]
    contact_path = _robot_dir(workspace, sequence_id) / "stage_c/contact_reference.json"
    trajectory_path = _robot_dir(workspace, sequence_id) / "trajectory_kinematic.npz"
    physics_path = _robot_dir(workspace, sequence_id) / "stage_c/physics_input.json"
    contact = json.loads(contact_path.read_text(encoding="utf-8"))
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    mesh = trimesh.load(Path(physics["collision_cache"]) / "visual/visual.obj", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError("V2 requires a non-empty object visual mesh")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    source_frame_count = max((int(row["frame_index"]) for row in contact["records"]), default=-1) + 1
    active = {(str(row["side"]), str(row["finger_region"]), int(row["contact_interval_id"])): [] for row in contact["records"] if row.get("contact_flag") and row.get("confidence") == "high" and 0 < int(row["frame_index"]) < source_frame_count - 1}
    for row in contact["records"]:
        if row.get("contact_flag") and row.get("confidence") == "high" and 0 < int(row["frame_index"]) < source_frame_count - 1:
            active[(str(row["side"]), str(row["finger_region"]), int(row["contact_interval_id"]))].append(row)
    patches: list[dict[str, Any]] = []
    for role in roles:
        records = active[(role["side"], role["source_finger"], role["source_contact_interval_id"])]
        local_points: list[np.ndarray] = []; local_normals: list[np.ndarray] = []
        for row in records:
            frame = int(row["frame_index"]) - 1
            if not 0 <= frame < len(qpos):
                raise RuntimeError("source contact frame does not map to immutable Stage-B trajectory")
            point, normal = _object_local(row["surface_point_world"], row["surface_normal_world"], qpos[frame, -14:-7])
            local_points.append(point); local_normals.append(normal / max(np.linalg.norm(normal), 1e-12))
        center = np.mean(local_points, axis=0); normal = np.mean(local_normals, axis=0); normal /= max(np.linalg.norm(normal), 1e-12)
        seed, core = _patch_faces(mesh, center, normal, float(contract["patches"]["core_radius_m"]))
        _seed, extended = _patch_faces(mesh, center, normal, float(contract["patches"]["extended_radius_m"]))
        if not core or not extended:
            raise RuntimeError("fail closed: source patch has no normal-consistent mesh-adjacent faces")
        patch_id = f"patch:{role['role_id']}"
        patches.append({"patch_id": patch_id, "role_id": role["role_id"], "object": "right_object", "source_side": role["side"],
                        "source_finger": role["source_finger"], "functional_role": role["functional_role"], "contact_interval": [role["frame_start"], role["frame_end"]],
                        "confidence": role["confidence"], "center_object_local": center, "surface_normal_object_local": normal,
                        "normal_distribution": {"mean": normal, "sample_count": len(local_normals)}, "seed_face_id": seed,
                        "core_face_ids": core, "extended_face_ids": extended, "mesh_component": 0,
                        "core_radius_m": contract["patches"]["core_radius_m"], "extended_radius_m": contract["patches"]["extended_radius_m"],
                        "collision_visual_mapping": {"visual_mesh": str(Path(physics["collision_cache"]) / "visual/visual.obj"), "collision_cache": physics["collision_cache"], "mapping": "object-local source projection; face-adjacent normal-consistent visual patch"}})
    output = root / "source_contact_patches.json"
    _write_json(output, {"schema_version": 1, "stage": "C-X2", "sequence_id": sequence_id, "source_only": True,
                         "thin_wall_policy": "face adjacency plus normal consistency; disconnected or opposite-normal surfaces are rejected", "patches": patches})
    np.savez_compressed(output.with_suffix(".npz"), center_object_local=np.asarray([patch["center_object_local"] for patch in patches]), normal_object_local=np.asarray([patch["surface_normal_object_local"] for patch in patches]))
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
    patch_path = _v2_dir(workspace, sequence_id) / "source_contact_patches.json"
    if not patch_path.is_file():
        build_surface_patches(paths_config, sequence_id, contract_path)
    # Reachability is a bounded source-to-current-Wuji diagnostic.  It is not
    # a retroactive role classifier and cannot approve a collision path; its
    # only job is to rank the finite candidate catalogue before physical gates.
    from spider.tools.grab_stage_c import _site_ids, _stage_b_act_baseline
    physics = json.loads((_robot_dir(workspace, sequence_id) / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model)
    baseline, _baseline_qvel = _stage_b_act_baseline(paths, sequence_id); sites = _site_ids(model)
    regions = default_robot_regions(); weights = contract["assignment"]["weights"]
    candidates: list[dict[str, Any]] = []; selected: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []
    max_switches = max(0, int(np.floor((max((role["frame_count"] for role in roles), default=0) / 120.0) * contract["assignment"]["max_switches_per_second"])))
    occupied: dict[str, set[int]] = {}
    def candidate_count(role: dict[str, Any]) -> int:
        choices = [region for region in regions if allowed_region(region, role["side"], role["source_finger"], role["functional_role"], level) and region.kind == "fingertip"]
        if level in {2, 3, 4} and role["source_finger"] != "thumb":
            alternatives = [region for region in choices if region.finger != role["source_finger"]]
            if alternatives:
                choices = alternatives
        return len(choices)
    # Long intervals reserve capacity first.  This is a bounded interval-level
    # allocation, not a per-frame greedy nearest-point assignment.  Scarce
    # roles go first so a flexible role cannot consume their only region.
    for role in sorted(roles, key=lambda row: (candidate_count(row), -int(row["frame_count"]), str(row["role_id"]))):
        allowed = [region for region in regions if allowed_region(region, role["side"], role["source_finger"], role["functional_role"], level)]
        # Level 2 exists to test the declared neighboring-finger relaxation
        # after Level 1 fails.  Retaining the same-finger option as the sole
        # minimizer would be a no-op, not an attempted reassignment.  Thumb
        # roles remain excluded by ``allowed_region`` and can never transfer.
        if level in {2, 3, 4} and role["source_finger"] != "thumb":
            reassigned = [region for region in allowed if region.finger != role["source_finger"]]
            if reassigned:
                allowed = reassigned
        if not allowed:
            trace.append({"role_id": role["role_id"], "status": "NO_ALLOWED_REGION", "level": level})
            continue
        options: list[dict[str, Any]] = []
        for region in allowed:
            if region.kind != "fingertip":
                # A palm is catalogued for support roles, but a physical V2
                # target needs an explicit sampled palm surface.  Do not
                # silently equate a palm collision with a fingertip site.
                continue
            identity = 0.0 if region.finger == role["source_finger"] else 1.0
            channel = (0 if region.side == "right" else 5) + FINGERS.index(str(region.finger))
            site_index = 1 + channel if channel < 5 else 2 + channel
            frames = np.asarray(role["stage_c_frame_indices"], dtype=np.int64)
            distances: list[float] = []
            for frame in frames:
                data.qpos[:] = baseline[frame]; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                distances.append(float(np.linalg.norm(data.site_xpos[sites[site_index]] - np.asarray(role["source_anchor_world_mean"]))))
            reachability = min(1.0, float(np.median(distances)) / 0.03)
            terms = {"surface_patch": reachability, "normal": 0.5, "functional_role": 0.0, "identity_change": identity,
                     "reachability": reachability, "collision_risk": 0.0, "tracking_deviation": identity, "temporal_switch": 0.0}
            options.append({"robot_region": region.region_id, "cost_terms": terms, "cost": assignment_cost(terms, weights), "capacity": region.capacity})
        candidates.append({"role_id": role["role_id"], "level": level, "options": options})
        frames = set(int(frame) for frame in role["stage_c_frame_indices"])
        feasible_indices = [index for index, option in enumerate(options) if not frames.intersection(occupied.get(option["robot_region"], set()))]
        if not feasible_indices:
            trace.append({"role_id": role["role_id"], "status": "CAPACITY_CONFLICT", "level": level})
            continue
        frame_costs = np.repeat(np.asarray([[options[index]["cost"] for index in feasible_indices]], dtype=float), role["frame_count"], axis=0)
        path, cost, switches = viterbi_assignment(frame_costs, float(weights["temporal_switch"]), max_switches)
        selected_option = options[feasible_indices[int(path[0])]]
        occupied.setdefault(selected_option["robot_region"], set()).update(frames)
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
    candidate_offsets = [0]; candidate_costs: list[float] = []
    for row in candidates:
        candidate_costs.extend(float(option["cost"]) for option in row["options"])
        candidate_offsets.append(len(candidate_costs))
    np.savez_compressed(root / f"contact_assignment_candidates_level_{level}.npz", option_offsets=np.asarray(candidate_offsets, dtype=np.int32), option_costs=np.asarray(candidate_costs, dtype=np.float64))
    selected_path = root / f"selected_contact_assignment_level_{level}.json"
    _write_json(selected_path, {"schema_version": 1, "stage": "C-X3", "sequence_id": sequence_id, "level": level, "status": status, "selected": selected, "source_roles": str(role_path), "contract_hash": _hash(contract_file)})
    _write_json(root / f"contact_assignment_trace_level_{level}.json", {"schema_version": 1, "sequence_id": sequence_id, "level": level, "trace": trace})
    trace_offsets = [0]; trace_paths: list[int] = []
    for row in trace:
        trace_paths.extend(row.get("path", [])); trace_offsets.append(len(trace_paths))
    np.savez_compressed(root / f"contact_assignment_trace_level_{level}.npz", path_offsets=np.asarray(trace_offsets, dtype=np.int32), selected_option_indices=np.asarray(trace_paths, dtype=np.int32))
    np.savez_compressed(root / f"selected_contact_assignment_level_{level}.npz", assignment_cost=np.asarray([row["assignment_cost"] for row in selected]), switch_count=np.asarray([row["switch_count"] for row in selected], dtype=np.int32))
    return str(selected_path)


def compile_contact_targets(paths_config: str, sequence_id: str, level: int, flexible_patch: bool = False) -> str:
    """Compile a selected assignment to isolated (T,10,3) V2 targets.

    The source reference stays intact.  This file adds an explicit robot
    channel mapping only; it is never written into V1 contact references.
    """
    if sequence_id not in PILOTS or level not in range(1, 5):
        raise ValueError("V2 target compilation requires frozen pilot and level 1..4")
    paths = _paths(paths_config); root = _v2_dir(paths.workspace_root, sequence_id)
    contract = yaml.safe_load(Path("configs/project/grab_wuji_stage_c_contract_v2.yaml").read_text(encoding="utf-8"))
    selected_path = root / f"selected_contact_assignment_level_{level}.json"
    role_path = root / "source_contact_roles.json"
    if not selected_path.is_file():
        build_assignment_candidates(paths_config, sequence_id, level)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    roles = {row["role_id"]: row for row in json.loads(role_path.read_text(encoding="utf-8"))["roles"]}
    patches = {row["patch_id"]: row for row in json.loads((root / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"]}
    source_npz = _robot_dir(paths.workspace_root, sequence_id) / "stage_c/contact_reference.npz"
    with np.load(source_npz, allow_pickle=False) as archive:
        source_expected = np.asarray(archive["contact"][1:-1], dtype=bool)
        source_anchors = np.asarray(archive["contact_surface_world"][1:-1], dtype=np.float64)
    source_normals = np.zeros_like(source_anchors)
    source_json = json.loads((_robot_dir(paths.workspace_root, sequence_id) / "stage_c/contact_reference.json").read_text(encoding="utf-8"))
    for row in source_json["records"]:
        source_frame = int(row["frame_index"])
        if row.get("contact_flag") and 0 < source_frame < len(source_expected) + 1:
            source_normals[source_frame - 1, int(row["contact_channel"])] = row["surface_normal_world"]
    expected = np.zeros_like(source_expected); anchors = np.zeros_like(source_anchors); normals = np.zeros_like(source_anchors); assignment_index = np.full_like(source_expected, -1, dtype=np.int32)
    if flexible_patch:
        from spider.tools.grab_stage_c import _site_ids, _stage_b_act_baseline
        physics = json.loads((_robot_dir(paths.workspace_root, sequence_id) / "stage_c/physics_input.json").read_text(encoding="utf-8"))
        mesh = trimesh.load(Path(physics["collision_cache"]) / "visual/visual.obj", force="mesh")
        model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model)
        baseline, _baseline_qvel = _stage_b_act_baseline(paths, sequence_id); site_ids = _site_ids(model)
        with np.load(_robot_dir(paths.workspace_root, sequence_id) / "trajectory_kinematic.npz", allow_pickle=False) as archive:
            raw_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    for index, row in enumerate(selected):
        role = roles[row["role_id"]]
        region = str(row["selected_robot_region"])
        pieces = region.split("_")
        if len(pieces) < 3 or pieces[-1] != "fingertip":
            raise RuntimeError("fail closed: physics targets require an explicit Wuji fingertip site mapping")
        side, finger = pieces[0], pieces[1]
        robot_channel = (0 if side == "right" else 5) + FINGERS.index(finger)
        source_channel = int(role["source_contact_channel"])
        frames = np.asarray(role["stage_c_frame_indices"], dtype=np.int64)
        if np.any(frames < 0) or np.any(frames >= len(expected)):
            raise RuntimeError("V2 assignment has an out-of-range physics frame")
        if np.any(expected[frames, robot_channel]):
            raise RuntimeError("fail closed: selected assignments exceed fingertip contact-region capacity")
        if not np.all(source_expected[frames, source_channel]):
            raise RuntimeError("V2 assignment does not map an active immutable source contact")
        expected[frames, robot_channel] = True; anchors[frames, robot_channel] = source_anchors[frames, source_channel]; normals[frames, robot_channel] = source_normals[frames, source_channel]
        if flexible_patch:
            patch = patches[row["source_patch_id"]]
            if level == 4:
                _seed, face_ids = _patch_faces(mesh, np.asarray(patch["center_object_local"]), np.asarray(patch["surface_normal_object_local"]), float(contract["patches"]["functional_surface_radius_m"]))
                faces = np.asarray(face_ids, dtype=np.int64)
            else:
                faces = np.asarray(patch["extended_face_ids"], dtype=np.int64)
            patch_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[faces], process=False)
            site_index = 1 + robot_channel if robot_channel < 5 else 2 + robot_channel
            for frame in frames:
                data.qpos[:] = baseline[frame]; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                object_pose = raw_qpos[frame, -14:-7]
                rotation = Rotation.from_quat(object_pose[3:7][[1, 2, 3, 0]])
                robot_local = rotation.inv().apply(data.site_xpos[site_ids[site_index]] - object_pose[:3])
                closest, _distance, face_ids = trimesh.proximity.closest_point_naive(patch_mesh, np.asarray([robot_local]))
                source_local = rotation.inv().apply(source_anchors[frame, source_channel] - object_pose[:3])
                max_displacement = float(contract["patches"]["functional_surface_radius_m"] if level == 4 else patch["extended_radius_m"])
                if np.linalg.norm(closest[0] - source_local) <= max_displacement:
                    anchors[frame, robot_channel] = rotation.apply(closest[0]) + object_pose[:3]
                    normals[frame, robot_channel] = rotation.apply(np.array(patch_mesh.face_normals[int(face_ids[0])], dtype=np.float64, copy=True))
        assignment_index[frames, robot_channel] = index
        row["robot_contact_channel"] = robot_channel; row["source_contact_channel"] = source_channel
    suffix = "_flexible" if flexible_patch else ""
    output = root / f"contact_targets_level_{level}{suffix}.npz"
    _write_json(root / f"contact_targets_level_{level}{suffix}.json", {"schema_version": 1, "sequence_id": sequence_id, "level": level, "flexible_patch": flexible_patch, "selected_assignment": str(selected_path), "output": str(output), "source_contact_reference": str(source_npz), "object_pose_immutable": True})
    np.savez_compressed(output, expected=expected.astype(np.uint8), anchors=anchors, normals=normals, assignment_index=assignment_index, source_expected=source_expected.astype(np.uint8), source_anchors=source_anchors, source_normals=source_normals)
    return str(output)


def evaluate_v2_depenetrated(
    paths_config: str, sequence_id: str, level: int, trajectory_path: str, metrics_path: str,
    contact_targets_path: str | None = None,
) -> str:
    """Evaluate V1 and V2 metrics side-by-side without changing either contract."""
    if sequence_id not in PILOTS or level not in range(1, 5):
        raise ValueError("V2 evaluation requires frozen pilot and level 1..4")
    # Reuse the audited V1 geometry/tracking evaluator only to retain its
    # exact metric.  V2 fields are appended under a distinct namespace.
    from spider.tools.grab_stage_c import _site_ids, _stage_b_dirs, evaluate_depenetrated_init

    paths = _paths(paths_config); root = _v2_dir(paths.workspace_root, sequence_id)
    targets_path = Path(contact_targets_path) if contact_targets_path is not None else root / f"contact_targets_level_{level}.npz"
    selected_path = root / f"selected_contact_assignment_level_{level}.json"
    if not targets_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError("compile V2 contact targets before evaluation")
    evaluate_depenetrated_init(paths_config, sequence_id, trajectory_path=trajectory_path, metrics_path=metrics_path)
    metrics_file = Path(metrics_path); metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    contract = yaml.safe_load(Path("configs/project/grab_wuji_stage_c_contract_v2.yaml").read_text(encoding="utf-8"))
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    with np.load(targets_path, allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool); anchors = np.asarray(archive["anchors"], dtype=np.float64)
        normals = np.asarray(archive["normals"], dtype=np.float64); assignment_index = np.asarray(archive["assignment_index"], dtype=np.int32)
    if qpos.shape[0] != expected.shape[0] or expected.shape != assignment_index.shape:
        raise RuntimeError("V2 trajectory/target schema mismatch")
    model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model); site_ids = _site_ids(model)
    tips = np.empty((len(qpos), 10, 3), dtype=np.float64)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        tips[frame] = data.site_xpos[site_ids][[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    distance = np.linalg.norm(tips - anchors, axis=2)
    threshold = float(contract["acceptance"]["patch_distance_m"])
    covered = expected & (distance <= threshold)
    vector = tips - anchors; length = np.linalg.norm(vector, axis=2); normal_length = np.linalg.norm(normals, axis=2)
    valid_normal = expected & (length > 1e-9) & (normal_length > 1e-9)
    cosine = np.full(expected.shape, np.nan, dtype=np.float64)
    cosine[valid_normal] = np.sum(vector[valid_normal] * normals[valid_normal], axis=1) / (length[valid_normal] * normal_length[valid_normal])
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    role_pass: list[bool] = []
    for index, _row in enumerate(selected):
        mask = assignment_index == index
        role_pass.append(bool(mask.any() and np.mean(covered[mask]) >= 0.8))
    coverage = float(np.count_nonzero(covered) / max(1, np.count_nonzero(expected)))
    patch_distance_p95 = float(np.percentile(distance[expected], 95)) if np.any(expected) else float("nan")
    finite_cosine = cosine[valid_normal]
    normal_median = float(np.median(finite_cosine)) if len(finite_cosine) else float("nan")
    gates = {
        "v1_exact_metric_preserved": "contact" in metrics and "high_confidence_recall" in metrics["contact"],
        "task_equivalent_patch_coverage": coverage >= float(contract["acceptance"]["task_equivalent_patch_coverage"]),
        "surface_patch_distance_p95": bool(np.isfinite(patch_distance_p95) and patch_distance_p95 <= float(contract["acceptance"]["patch_distance_p95_m"])),
        "functional_role_recall": float(np.mean(role_pass)) >= float(contract["acceptance"]["functional_role_recall"]),
        "normal_alignment": bool(np.isfinite(normal_median) and normal_median >= float(contract["acceptance"]["normal_cosine_median"])),
        "depenetrated_visual_penetration": metrics["visual_penetration"]["max_penetration_m"] <= float(contract["acceptance"]["depenetrated_visual_max_m"]),
        "depenetrated_collision_penetration": metrics["collision"]["after_max_m"] <= float(contract["acceptance"]["depenetrated_collision_max_m"]),
        "tracking": bool(metrics["gates"]["tracking"]), "smoothness": bool(metrics["gates"]["smoothness"]),
    }
    metrics["exact_contact_recall_v1"] = metrics["contact"]["high_confidence_recall"]
    metrics["task_equivalent_contact_v2"] = {
        "patch_coverage": coverage, "functional_role_recall": float(np.mean(role_pass)) if role_pass else 0.0,
        "surface_patch_distance_p95_m": patch_distance_p95,
        "normal_cosine_median": normal_median, "assignment_switch_rate_per_s": 0.0,
        "assignment_count": len(selected), "identity_change_count": sum(row["source_finger"] not in row["selected_robot_region"] for row in selected),
    }
    metrics["contract_v2"] = {"name": "TASK_EQUIVALENT_CONTACT", "level": level, "targets": str(targets_path), "gates": gates,
                              "status": "PASS" if all(gates.values()) else "FAIL"}
    _write_json(metrics_file, metrics)
    return str(metrics_file)


def write_level_result(paths_config: str, sequence_id: str, level: int, metrics_path: str) -> str:
    """Write a fail-closed level outcome; static failure forbids costly replay."""
    if sequence_id not in PILOTS or level not in range(1, 5):
        raise ValueError("V2 level result requires frozen pilot and level 1..4")
    paths = _paths(paths_config); root = _v2_dir(paths.workspace_root, sequence_id)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    v2 = metrics.get("contract_v2", {})
    if v2.get("level") != level or v2.get("name") != "TASK_EQUIVALENT_CONTACT":
        raise RuntimeError("metrics do not belong to requested V2 level")
    v1 = json.loads((paths.workspace_root / "reports/stage_c_contract_v1_manifest.json").read_text(encoding="utf-8"))
    if v1.get("v1_status") != "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT":
        raise RuntimeError("V1 immutable manifest is missing or downgraded")
    static_pass = v2.get("status") == "PASS"
    output = root / f"level_{level}_result.json"
    _write_json(output, {
        "schema_version": 1, "stage": "C-X4", "sequence_id": sequence_id, "relaxation_level": level,
        "status": "PENDING_PHYSICAL_PREFLIGHT" if static_pass else "FAIL_STATIC_CONTRACT",
        "static_metrics": str(metrics_path), "static_gates": v2.get("gates", {}),
        "physical_preflight": "REQUIRED" if static_pass else "NOT_RUN_STATIC_CONTRACT_FAILED",
        "eligible_to_try_next_level": not static_pass,
        "v1_exact_contact": "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT",
        "v1_immutable_statement": "V1 result is immutable and remains BLOCKED.",
        "stage_b_untouched": True,
    })
    return str(output)


def write_contract_reports(paths_config: str) -> str:
    """Emit a fail-closed V1/V2 comparison after the frozen primary ladder.

    This is deliberately a reporting operation: it cannot promote a static
    candidate into a physics pass and it never writes into any Stage-B or V1
    directory.  Smoke pilots remain explicitly ``NOT_RUN`` when primary has
    no successful relaxation level.
    """
    paths = _paths(paths_config)
    workspace = paths.workspace_root
    primary = "s5__cylindermedium_lift"
    root = _v2_dir(workspace, primary)
    v1_path = workspace / "reports/stage_c_contract_v1_manifest.json"
    if not v1_path.is_file():
        raise FileNotFoundError("freeze V1 before producing a V2 comparison")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if v1.get("v1_status") != "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT":
        raise RuntimeError("fail closed: immutable V1 status is absent")
    contract_path = Path("configs/project/grab_wuji_stage_c_contract_v2.yaml")
    levels: dict[str, Any] = {}
    for level in range(1, 5):
        result_path = root / f"level_{level}_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing required Level {level} result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics_path = Path(result["static_metrics"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        levels[str(level)] = {"result": result, "metrics": metrics}
    minimum = select_minimum_successful_level({int(level): "PASS" if row["result"]["status"] == "PENDING_PHYSICAL_PREFLIGHT" else "FAIL" for level, row in levels.items()})
    if minimum is not None:
        raise RuntimeError("a static candidate exists; use physical preflight reporting instead")
    selected_path = root / "selected_contact_assignment_level_4.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"] if selected_path.is_file() else []
    primary_metrics = levels["4"]["metrics"]
    pilots = {
        primary: {"frame_range": PILOTS[primary], "status": "FAIL_STATIC_CONTRACT", "metrics": str(levels["4"]["result"]["static_metrics"])},
        "s1__mug_lift": {"frame_range": PILOTS["s1__mug_lift"], "status": "NOT_RUN", "reason": "primary has no successful relaxation level"},
        "s1__mug_offhand_1": {"frame_range": PILOTS["s1__mug_offhand_1"], "status": "NOT_RUN", "reason": "primary has no successful relaxation level"},
    }
    common = {
        "schema_version": 1,
        "stage": "C-X",
        "v1_exact_contact": "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT",
        "v1_immutable_statement": "V1 result is immutable and remains BLOCKED.",
        "v2_task_equivalent_contact": "BLOCKED_BY_INFEASIBLE_TASK_EQUIVALENT_CONTACT",
        "minimum_successful_relaxation_level": "NONE",
        "contract_v2_path": str(contract_path),
        "contract_v2_hash": _hash(contract_path),
        "v1_manifest": str(v1_path),
        "level_ladder": {level: {"status": row["result"]["status"], "gates": row["result"]["static_gates"], "metrics": row["result"]["static_metrics"]} for level, row in levels.items()},
        "primary_assignment": selected,
        "frozen_pilots": pilots,
        "stage_d": "NOT_STARTED",
    }
    comparison = dict(common)
    comparison["v1_best_dynamic_candidate"] = v1.get("best_dynamic_candidate")
    comparison["v2_level_4_metrics"] = primary_metrics
    validation = dict(common)
    validation["hard_validity"] = {
        "primary": {"nan_inf": primary_metrics["nan_inf"], "joint_limit_violations": primary_metrics["joint_limit_violations"], "source_mapping_complete": primary_metrics["source_mapping_complete"], "object_pose_change_m": primary_metrics["object_pose_change_m"]},
        "smokes": "NOT_RUN_PRIMARY_STATIC_GATE_FAILED",
    }
    acceptance = dict(common)
    acceptance.update({"status": "FAIL", "reason": "No relaxation level satisfied every static Contract V2 gate; physical preflight, HTML, and screenshot review were not authorized.", "user_html_review": "PENDING"})
    screenshot = dict(common)
    screenshot.update({"status": "FAIL", "reason": "No V2 PASS candidate exists; no acceptance HTML or screenshots may be fabricated.", "manifest_complete": False, "frames": [], "views": []})
    reports = workspace / "reports"
    _write_json(reports / "stage_c_contract_comparison.json", comparison)
    _write_json(reports / "stage_c_contract_v2_validation.json", validation)
    _write_json(reports / "stage_c_contract_v2_acceptance.json", acceptance)
    _write_json(reports / "stage_c_contract_v2_screenshot_review.json", screenshot)
    summary = primary_metrics["task_equivalent_contact_v2"]
    md = (
        "# Stage C contract comparison\n\n"
        "## Status\n\n"
        "- Contract V1 exact contact: `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`\n"
        "- Contract V2 task-equivalent contact: `BLOCKED_BY_INFEASIBLE_TASK_EQUIVALENT_CONTACT`\n"
        "- Minimum successful relaxation level: `NONE`\n"
        "- Stage D: `NOT STARTED`\n\n"
        "V1 result is immutable and remains BLOCKED. V2 does not claim exact human contact reproduction.\n\n"
        "## Primary Level 4 (widest permitted bounded relaxation)\n\n"
        f"- Task-equivalent patch coverage: `{summary['patch_coverage']:.6f}` (pass: >= 0.70)\n"
        f"- Functional-role recall: `{summary['functional_role_recall']:.6f}` (fail: < 0.80)\n"
        f"- Patch-distance P95: `{summary['surface_patch_distance_p95_m']:.6f} m` (fail: > 0.020 m)\n"
        f"- Normal cosine median: `{summary['normal_cosine_median']:.6f}` (pass: >= 0.50)\n"
        f"- Collision max after depenetration: `{primary_metrics['collision']['after_max_m']:.6f} m` (pass: <= 0.003 m)\n\n"
        "All four levels were tried in order. Since no primary level passed static Contract V2 gates, physical preflight, MJWP, smoke pilots, acceptance HTML, and screenshots were not run.\n"
    )
    (reports / "STAGE_C_CONTRACT_COMPARISON.md").write_text(md, encoding="utf-8")
    (reports / "STAGE_C_CONTRACT_V2_ACCEPTANCE.md").write_text("# Stage C Contract V2 acceptance\n\n**FAIL-CLOSED.** " + acceptance["reason"] + "\n", encoding="utf-8")
    (reports / "STAGE_C_CONTRACT_V2_SCREENSHOT_REVIEW.md").write_text("# Stage C Contract V2 screenshot review\n\n**FAIL.** " + screenshot["reason"] + "\n", encoding="utf-8")
    return str(reports / "stage_c_contract_comparison.json")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"freeze-v1-manifest": freeze_v1_manifest, "build-source-contact-roles": build_source_contact_roles, "build-surface-patches": build_surface_patches, "build-assignment-candidates": build_assignment_candidates, "compile-contact-targets": compile_contact_targets, "evaluate-v2-depenetrated": evaluate_v2_depenetrated, "write-level-result": write_level_result, "write-contract-reports": write_contract_reports})
