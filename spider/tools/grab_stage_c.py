"""Fail-closed Stage C support for frozen GRAB -> Wuji Hand2 pilots.

This module deliberately keeps all generated data under the configured external
workspace.  It never rewrites raw GRAB, canonical Stage B artefacts, or the
repository asset tree.  It derives source geometry/contact references, builds
hash-addressed collision caches, and prepares isolated MJWP input sandboxes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import trimesh
import tyro
import yaml
from scipy.spatial.transform import Rotation

from spider.datasets.grab import GrabAdapter
from spider.datasets.paths import load_project_paths
from spider.datasets.schema import CanonicalHOISequence
from spider.io import get_processed_data_dir
from spider.preprocess.generate_xml import main as generate_xml


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_IDS = (4, 8, 12, 16, 20)
PILOTS = (
    {"sequence_id": "s5__cylindermedium_lift", "source_sequence_id": "s5/cylindermedium_lift", "role": "true-bimanual physics primary", "hand_mode": "bimanual", "frame_range": [1460, 1876], "selection_reason": "frozen Stage B primary", "required_checks": ["bimanual", "tracking", "penetration", "contact", "object", "smoothness"]},
    {"sequence_id": "s1__mug_lift", "source_sequence_id": "s1/mug_lift", "role": "right-hand physics smoke", "hand_mode": "bimanual", "frame_range": [120, 240], "selection_reason": "frozen Stage B right-hand smoke", "required_checks": ["right_hand", "tracking", "penetration", "contact", "object", "smoothness"]},
    {"sequence_id": "s1__mug_offhand_1", "source_sequence_id": "s1/mug_offhand_1", "role": "offhand/non-interacting-hand smoke", "hand_mode": "bimanual", "frame_range": [120, 180], "selection_reason": "frozen Stage B offhand smoke", "required_checks": ["offhand", "tracking", "false_contact", "object", "smoothness"]},
)
_RAYCAST_SCENES: dict[tuple[int, int], o3d.t.geometry.RaycastingScene] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_hash(profile_path: Path) -> str:
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def _pilot(sequence_id: str) -> dict[str, Any]:
    for pilot in PILOTS:
        if pilot["sequence_id"] == sequence_id:
            return dict(pilot)
    raise ValueError(f"Not a frozen Stage C pilot: {sequence_id}")


def _paths(paths_config: str):
    return load_project_paths(paths_config)


def _stage_b_dirs(workspace: Path, sequence_id: str) -> tuple[Path, Path]:
    canonical = workspace / "processed/grab/canonical" / sequence_id
    robot = workspace / "processed/grab/wuji_hand2_beta1/bimanual" / sequence_id / "0"
    if not canonical.is_dir() or not robot.is_dir():
        raise FileNotFoundError(f"Missing frozen Stage B input for {sequence_id}: {canonical}, {robot}")
    return canonical, robot


def _object_mesh(sequence: CanonicalHOISequence) -> Path:
    return Path(sequence.objects[0].source_metadata["resolved_local_mesh_path"])


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh at {path}")
    return loaded


def _local_points(points_world: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    return rotation.inv().apply(np.asarray(points_world) - pos)


def _world_points(points_local: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).apply(points_local) + pos


def _closest_with_sign(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    # Open3D's BVH query is materially faster than trimesh's R-tree fallback
    # on the 40k-face GRAB meshes.  It is still a closest-*surface* query, not
    # a nearest-vertex approximation.  Reuse the immutable mesh's BVH across
    # all frames so diagnostics remain practical without weakening the metric.
    key = (id(mesh.vertices), id(mesh.faces))
    scene = _RAYCAST_SCENES.get(key)
    if scene is None:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
        _RAYCAST_SCENES[key] = scene
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    closest = scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
    unsigned = np.linalg.norm(np.asarray(points) - closest, axis=1)
    method, confidence = "unsigned_closest_surface", "low"
    signed = np.full(len(points), np.nan, dtype=np.float64)
    if mesh.is_watertight and mesh.is_winding_consistent:
        signed = scene.compute_signed_distance(query).numpy().astype(np.float64)
        method, confidence = "open3d_watertight_signed_distance", "high"
    return closest, unsigned, signed, method, confidence


def _intervals(mask: np.ndarray) -> list[list[int]]:
    out: list[list[int]] = []
    start: int | None = None
    for index, enabled in enumerate(mask.astype(bool)):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            out.append([start, index])
            start = None
    if start is not None:
        out.append([start, len(mask)])
    return out


def _diagnose(sequence: CanonicalHOISequence, mesh: trimesh.Trimesh) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    object_item = sequence.objects[0]
    frame_count = sequence.num_frames
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"sides": {}, "object_mesh": {"is_watertight": bool(mesh.is_watertight), "is_winding_consistent": bool(mesh.is_winding_consistent), "bbox": mesh.bounds, "vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces)), "scale": [1.0, 1.0, 1.0]}}
    for side, hand in (("right", sequence.right_hand), ("left", sequence.left_hand)):
        if hand is None or hand.vertices_world is None:
            raise ValueError(f"Stage C source diagnostic requires reconstructed {side} vertices")
        n_vertices = hand.vertices_world.shape[1]
        signed = np.full((frame_count, n_vertices), np.nan, np.float32)
        unsigned = np.zeros((frame_count, n_vertices), np.float32)
        closest = np.zeros((frame_count, n_vertices, 3), np.float32)
        sign_method = "unsigned_only"
        confidence = "low"
        for frame in range(frame_count):
            local = _local_points(hand.vertices_world[frame], object_item.translation[frame], object_item.orientation[frame])
            point, distance, sign, sign_method, confidence = _closest_with_sign(mesh, local)
            unsigned[frame] = distance
            signed[frame] = sign
            closest[frame] = _world_points(point, object_item.translation[frame], object_item.orientation[frame])
        negative = np.where(np.isfinite(signed) & (signed < 0), signed, 0.0)
        penetrating = negative < 0
        per_count = penetrating.sum(axis=1)
        arrays.update({f"{side}_signed_distance_m": signed, f"{side}_unsigned_distance_m": unsigned, f"{side}_closest_surface_world": closest, f"{side}_penetrating_vertex_count": per_count.astype(np.int32)})
        summary["sides"][side] = {"sign_method": sign_method, "sign_confidence": confidence, "per_frame": {"penetrating_vertex_count": per_count.astype(int), "penetrating_vertex_ratio": (per_count / n_vertices), "mean_negative_depth_m": np.where(per_count > 0, -negative.sum(axis=1) / np.maximum(per_count, 1), 0.0), "p95_negative_depth_m": np.array([np.percentile(-row[row < 0], 95) if np.any(row < 0) else 0.0 for row in negative]), "max_penetration_depth_m": -negative.min(axis=1), "integrated_penetration_score_m": -negative.sum(axis=1)}, "aggregate": {"max_penetration_depth_m": float((-negative).max()), "integrated_penetration_score_m": float((-negative).sum())}}
    return arrays, summary


def make_manifest(paths_config: str) -> str:
    paths = _paths(paths_config)
    records: list[dict[str, Any]] = []
    for frozen in PILOTS:
        canonical, robot = _stage_b_dirs(paths.workspace_root, frozen["sequence_id"])
        metadata = json.loads((canonical / "canonical_metadata.json").read_text(encoding="utf-8"))
        task = json.loads((robot.parent / "task_info.json").read_text(encoding="utf-8"))
        record = dict(frozen)
        record.update({"fps": metadata["fps"], "canonical_path": str(canonical), "kinematic_path": str(robot / "trajectory_kinematic.npz"), "object_mesh": metadata["objects"][0]["source_metadata"]["resolved_local_mesh_path"], "stage_b_config_hash": task["config_hash"], "stage_b_metrics": str(robot / "metrics_kinematic.json"), "stage_b_html": str(robot / "visualization_ik_interactive.html")})
        records.append(record)
    target = paths.workspace_root / "manifests/grab_stage_c_pilots.json"
    _atomic_json(target, {"schema_version": 1, "frozen": True, "pilots": records})
    return str(target)


def diagnose_source(paths_config: str, sequence_id: str) -> str:
    paths = _paths(paths_config)
    frozen = _pilot(sequence_id)
    adapter = GrabAdapter(paths)
    sequence = adapter.load_sequence(frozen["source_sequence_id"], frame_start=frozen["frame_range"][0], frame_end=frozen["frame_range"][1], include_vertices=True)
    mesh_path = _object_mesh(sequence)
    arrays, summary = _diagnose(sequence, _mesh(mesh_path))
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id)
    output = robot / "stage_c"
    _atomic_npz(output / "source_geometry_diagnostics.npz", source_frame_indices=np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64), **arrays)
    summary.update({"schema_version": 1, "sequence_id": sequence_id, "source_sequence_id": frozen["source_sequence_id"], "raw_grab_modified": False, "canonical_overwritten": False, "object_mesh": {**summary["object_mesh"], "path": str(mesh_path), "sha256": _hash_file(mesh_path)}})
    _atomic_json(output / "source_geometry_diagnostics.json", summary)
    return str(output / "source_geometry_diagnostics.json")


def build_contact_reference(paths_config: str, sequence_id: str, distance_threshold_m: float = 0.015) -> str:
    paths = _paths(paths_config)
    frozen = _pilot(sequence_id)
    adapter = GrabAdapter(paths)
    sequence = adapter.load_sequence(frozen["source_sequence_id"], frame_start=frozen["frame_range"][0], frame_end=frozen["frame_range"][1], include_vertices=False)
    mesh = _mesh(_object_mesh(sequence)); obj = sequence.objects[0]
    records: list[dict[str, Any]] = []
    contact = np.zeros((sequence.num_frames, 10), dtype=np.uint8)
    positions = np.zeros((sequence.num_frames, 10, 3), dtype=np.float32)
    for side_index, (side, hand) in enumerate((("right", sequence.right_hand), ("left", sequence.left_hand))):
        if hand is None: continue
        source_points = hand.joints_world[:, TIP_IDS, :]
        local_points = np.empty_like(source_points, dtype=np.float64)
        for frame in range(sequence.num_frames):
            local_points[frame] = _local_points(source_points[frame], obj.translation[frame], obj.orientation[frame])
        closest, unsigned, signed, method, confidence = _closest_with_sign(mesh, local_points.reshape(-1, 3))
        closest = closest.reshape(sequence.num_frames, 5, 3); unsigned = unsigned.reshape(sequence.num_frames, 5); signed = signed.reshape(sequence.num_frames, 5)
        for frame in range(sequence.num_frames):
            surface_points = _world_points(closest[frame], obj.translation[frame], obj.orientation[frame])
            rotation = Rotation.from_quat(obj.orientation[frame][[1, 2, 3, 0]])
            for finger_index, finger in enumerate(FINGERS):
                channel = side_index * 5 + finger_index; source = source_points[frame, finger_index]; surface = surface_points[finger_index]
                nearest_vertex = int(np.argmin(np.einsum("ij,ij->i", mesh.vertices - closest[frame, finger_index], mesh.vertices - closest[frame, finger_index])))
                normal = rotation.apply(np.array(mesh.vertex_normals[nearest_vertex], dtype=np.float64, copy=True))
                flag = bool(unsigned[frame, finger_index] <= distance_threshold_m or (np.isfinite(signed[frame, finger_index]) and signed[frame, finger_index] < 0))
                contact[frame, channel] = flag; positions[frame, channel] = surface
                records.append({"frame_index": frame, "source_frame": int(sequence.source_metadata["source_frame_indices"][frame]), "side": side, "finger_region": finger, "source_point_world": source, "surface_point_world": surface, "surface_normal_world": normal, "source_signed_distance_m": signed[frame, finger_index], "unsigned_distance_m": unsigned[frame, finger_index], "projected_distance_m": float(np.linalg.norm(source - surface)), "confidence": confidence, "sign_method": method, "contact_flag": flag, "contact_channel": channel})
    for channel in range(10):
        interval_id = np.full(sequence.num_frames, -1, dtype=np.int32)
        for value, (start, end) in enumerate(_intervals(contact[:, channel])): interval_id[start:end] = value
        for row in records:
            if row["contact_channel"] == channel: row["contact_interval_id"] = int(interval_id[row["frame_index"]])
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id); output = robot / "stage_c"
    _atomic_npz(output / "contact_reference.npz", source_frame_indices=np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64), contact=contact, contact_surface_world=positions)
    _atomic_json(output / "contact_reference.json", {"schema_version": 1, "sequence_id": sequence_id, "source_only": True, "distance_threshold_m": distance_threshold_m, "records": records, "intervals": {str(index): _intervals(contact[:, index]) for index in range(10)}})
    return str(output / "contact_reference.json")


def build_collision_cache(paths_config: str, sequence_id: str, max_convex_hulls: int = 8) -> str:
    paths = _paths(paths_config)
    frozen = _pilot(sequence_id); adapter = GrabAdapter(paths)
    sequence = adapter.load_sequence(frozen["source_sequence_id"], frame_start=frozen["frame_range"][0], frame_end=frozen["frame_range"][1])
    source = _object_mesh(sequence); digest = _hash_file(source); root = paths.workspace_root / "cache/objects" / digest[:16]
    visual_dir, collision_dir = root / "visual", root / "collision"; manifest = root / "manifest.json"
    if manifest.is_file(): return str(manifest)
    temporary = root.with_name(root.name + ".tmp")
    if temporary.exists(): shutil.rmtree(temporary)
    (temporary / "visual").mkdir(parents=True); (temporary / "collision").mkdir()
    # CoACD must be imported after SPIDER/Torch: importing its native module
    # first can crash Python 3.12 during Torch operator registration.
    import coacd

    mesh = _mesh(source); mesh.export(temporary / "visual/visual.obj")
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
    parts = coacd.run_coacd(coacd_mesh, max_convex_hull=int(max_convex_hulls))
    if not parts: raise RuntimeError("CoACD returned no collision parts")
    part_meta = []
    for index, part in enumerate(parts):
        vertices, faces = part
        component = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        if len(component.faces) == 0 or not np.isfinite(component.vertices).all() or component.area <= 0: raise RuntimeError(f"Invalid CoACD part {index}")
        component.export(temporary / f"collision/{index}.obj"); part_meta.append({"file": f"{index}.obj", "vertices": int(len(component.vertices)), "faces": int(len(component.faces)), "area": float(component.area), "volume": float(abs(component.volume))})
    validation = {"visual_bbox": mesh.bounds, "collision_bbox": trimesh.util.concatenate([_mesh(temporary / f"collision/{index}.obj") for index in range(len(parts))]).bounds, "same_bbox": bool(np.allclose(mesh.bounds, trimesh.util.concatenate([_mesh(temporary / f"collision/{index}.obj") for index in range(len(parts))]).bounds, atol=2e-3)), "finite": True}
    if not validation["same_bbox"]: raise RuntimeError("Collision bbox does not match visual mesh")
    _atomic_json(temporary / "validation.json", validation)
    _atomic_json(temporary / "manifest.json", {"schema_version": 1, "source_visual_mesh": str(source), "source_sha256": digest, "decomposition": {"tool": "coacd", "max_convex_hulls": max_convex_hulls}, "visual": {"bbox": mesh.bounds, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "volume": float(abs(mesh.volume))}, "collision_parts": part_meta})
    os.replace(temporary, root)
    return str(manifest)


def prepare_physics_input(paths_config: str, sequence_id: str) -> str:
    """Build an isolated, contact-guided MJWP input tree; Stage B remains read-only."""
    paths = _paths(paths_config); frozen = _pilot(sequence_id); workspace = paths.workspace_root
    _, robot = _stage_b_dirs(workspace, sequence_id); stage = robot / "stage_c"
    contact_path = stage / "contact_reference.npz"; cache_manifest = Path(build_collision_cache(paths_config, sequence_id))
    if not contact_path.is_file(): raise FileNotFoundError(f"Run build_contact_reference first: {contact_path}")
    sandbox = workspace / "stage_c_inputs" / sequence_id
    task = sequence_id
    mano_dir = sandbox / "processed/grab/mano/bimanual" / task / "0"; robot_dir = sandbox / "processed/grab/wuji_hand2_beta1/bimanual" / task / "0"; task_dir = robot_dir.parent
    mano_dir.mkdir(parents=True, exist_ok=True); robot_dir.mkdir(parents=True, exist_ok=True)
    source_keypoints = workspace / "processed/grab/mano/bimanual" / task / "0/trajectory_keypoints.npz"
    with np.load(source_keypoints, allow_pickle=False) as keypoints, np.load(contact_path, allow_pickle=False) as refs:
        values = {key: keypoints[key] for key in keypoints.files}
        values.update({"contact_right": refs["contact"][:, :5].astype(bool), "contact_left": refs["contact"][:, 5:].astype(bool), "contact_pos_right": refs["contact_surface_world"][:, :5].mean(axis=0), "contact_pos_left": refs["contact_surface_world"][:, 5:].mean(axis=0)})
        _atomic_npz(mano_dir / "trajectory_keypoints.npz", **values)
    cache_root = cache_manifest.parent
    task_info = json.loads((robot.parent / "task_info.json").read_text(encoding="utf-8"))
    task_info.update({"canonical_dir": str(workspace / "processed/grab/canonical" / task), "right_object_mesh_dir": os.path.relpath(cache_root / "visual", sandbox), "right_object_convex_dir": os.path.relpath(cache_root / "collision", sandbox), "left_object_mesh_dir": None, "left_object_convex_dir": None, "stage_c_collision_cache": str(cache_root)})
    _atomic_json(task_dir / "task_info.json", task_info)
    # generate_xml reads this metadata through the MANO/keypoint branch before
    # it writes the robot-side copy, so keep both isolated sandbox locations
    # synchronized.  Neither points at the Stage B tree.
    _atomic_json(mano_dir.parent / "task_info.json", task_info)
    # Generate both normal and object-actuator scenes through SPIDER's own path.
    generate_xml(dataset_dir=str(sandbox), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type="bimanual", task=task, data_id=0, use_visual_mesh_as_collision=False, show_viewer=False, act_scene=False)
    generate_xml(dataset_dir=str(sandbox), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type="bimanual", task=task, data_id=0, use_visual_mesh_as_collision=False, show_viewer=False, act_scene=True)
    act_model = mujoco.MjModel.from_xml_path(str(task_dir / "scene_act.xml"))
    contact_site_ids = [int(mujoco.mj_name2id(act_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip")) for side in ("right", "left") for finger in FINGERS]
    if any(site < 0 for site in contact_site_ids):
        raise RuntimeError(f"Missing Wuji contact sites: {contact_site_ids}")
    task_info["contact_site_ids"] = contact_site_ids
    _atomic_json(task_dir / "task_info.json", task_info)
    _atomic_json(mano_dir.parent / "task_info.json", task_info)
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as trajectory, np.load(contact_path, allow_pickle=False) as refs:
        qpos, qvel = trajectory["qpos"], trajectory["qvel"]
        # Object actuators expect translation and rotvec controls, after the 52
        # Wuji controls.  This explicitly avoids treating free-joint quaternions
        # as actuator commands.
        objects = qpos[:, -14:].reshape(len(qpos), 2, 7)
        object_ctrl = np.concatenate([objects[:, :, :3], Rotation.from_quat(objects[:, :, 3:][:, :, [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec().reshape(len(qpos), 2, 3)], axis=2).reshape(len(qpos), 12)
        # scene_act represents each object as xyz + rotvec (6 DoF), whereas
        # the Stage B scene stores free-joint xyz + quaternion (7 DoF).
        # Convert the *state* as well as controls; otherwise nq=64 would be
        # fed an invalid 66-column free-joint state.
        qpos_act = np.concatenate([qpos[:, :52], object_ctrl], axis=1)
        ctrl = qpos_act.copy()
        _atomic_npz(robot_dir / "trajectory_kinematic_act.npz", qpos=qpos_act, qvel=qvel, ctrl=ctrl, contact=refs["contact"][1:-1], contact_pos=refs["contact_surface_world"][1:-1], frequency=np.asarray(120.0))
    _atomic_json(stage / "physics_input.json", {"sandbox": str(sandbox), "scene_act": str(task_dir / "scene_act.xml"), "trajectory": str(robot_dir / "trajectory_kinematic_act.npz"), "collision_cache": str(cache_root), "baseline_untouched": True})
    return str(stage / "physics_input.json")


def write_failure_report(paths_config: str) -> str:
    """Materialize a fail-closed report for the observed MJWP start-state failure."""
    paths = _paths(paths_config); reports = paths.workspace_root / "reports"; profile = Path("configs/project/grab_wuji_stage_c.yaml")
    pilots: list[dict[str, Any]] = []
    for frozen in PILOTS:
        _, robot = _stage_b_dirs(paths.workspace_root, frozen["sequence_id"]); stage = robot / "stage_c"; diagnostic = stage / "source_geometry_diagnostics.json"; contact = stage / "contact_reference.json"
        pilots.append({"sequence_id": frozen["sequence_id"], "role": frozen["role"], "frames": frozen["frame_range"], "diagnostic": str(diagnostic) if diagnostic.is_file() else None, "contact_reference": str(contact) if contact.is_file() else None, "collision_cache": str(paths.workspace_root / "cache/objects"), "status": "NOT_RUN_AFTER_PRIMARY_START_STATE_FAILURE" if frozen["sequence_id"] != "s5__cylindermedium_lift" else "FAIL_MJWP_NAN_REWARD"})
    payload = {"schema_version": 1, "stage": "C", "status": "FAIL", "substatus": ["SOURCE_GEOMETRY_AND_CONTACT_READY", "COLLISION_CACHE_READY", "MJWP_CONTACT_GUIDANCE_START_STATE_UNSTABLE", "NO_OPTIMIZED_TRAJECTORY_ACCEPTED", "SCREENSHOT_REVIEW_NOT_RUN"], "profile": {"path": str(profile), "sha256": _profile_hash(profile), "shared_profile_verified": False}, "primary_failure": {"pilot": "s5__cylindermedium_lift", "command": "examples/run_mjwp.py contact_guidance=true max_sim_steps=1", "evidence": ["MJWP logged NaNs or infs in rews: 16/16", "MJWP final object tracking error: pos=nan, quat=nan", "CPU replay: qacc max 4.218e4 at 0.01s, 1.846e6 at 0.02s, 5.538e7 at 0.03s, 1.239e26 at 0.04s", "MuJoCo warned of huge QPOS/QACC at 0.04s"], "root_cause": "The frozen Stage B kinematic state enters the physics scene with deep hand-object overlap. SPIDER MJWP therefore cannot obtain a finite first optimization rollout from that state.", "not_accepted_workarounds": ["static hand translation", "frame deletion", "hiding mesh", "changing frozen pilots", "relaxing thresholds"]}, "gates": {"hard_validity": "FAIL", "tracking": "NOT_EVALUABLE", "visual_penetration": "BASELINE_DIAGNOSTIC_ONLY", "contact_preservation": "NOT_EVALUABLE", "object_tracking": "FAIL", "smoothness": "FAIL", "all_pilots_shared_profile": "NOT_EVALUABLE", "regression_tests": "NOT_RUN", "html": "NOT_GENERATED_AFTER_FAIL", "codex_screenshot_review": "FAIL"}, "raw_grab_modified": False, "stage_b_outputs_overwritten": False, "user_html_review": "PENDING", "pilots": pilots}
    _atomic_json(reports / "stage_c_validation.json", payload)
    _atomic_json(reports / "grab_stage_c_pilot_summary.json", {"status": "FAIL", "pilots": pilots})
    _atomic_json(reports / "stage_c_acceptance.json", {"status": "FAIL", "codex_visual_acceptance": "FAIL", "reason": "No finite optimized trajectory exists to review."})
    _atomic_json(reports / "stage_c_screenshot_review.json", {"status": "FAIL", "screenshots": [], "reason": "Fail-closed: screenshot generation/review is not valid without a finite optimized trajectory."})
    (reports / "STAGE_C_ACCEPTANCE.md").write_text("# Stage C acceptance\n\nStatus: **FAIL**. The primary contact-guided MJWP start-state rollout returned NaN rewards; no optimized trajectory, HTML, or screenshot acceptance is claimed. See `stage_c_validation.json`.\n", encoding="utf-8")
    (reports / "STAGE_C_SCREENSHOT_REVIEW.md").write_text("# Stage C screenshot review\n\nStatus: **FAIL**. No screenshots were generated because the primary optimized trajectory is non-finite.\n", encoding="utf-8")
    return str(reports / "stage_c_validation.json")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"make-manifest": make_manifest, "diagnose-source": diagnose_source, "build-contact-reference": build_contact_reference, "build-collision-cache": build_collision_cache, "prepare-physics-input": prepare_physics_input, "write-failure-report": write_failure_report})
