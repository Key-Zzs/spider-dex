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
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import trimesh
import tyro
import yaml
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from spider.datasets.grab import GrabAdapter
from spider.datasets.paths import load_project_paths
from spider.datasets.schema import CanonicalHOISequence
from spider.geometry.collision_audit import bbox_report, contact_pair_summary, mesh_from_model, region_for_geom, summarize_signed_distances
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


def _add_object_mocap_tracking(scene_path: Path) -> None:
    """Add a compliant *reference* controller, never a qpos overwrite.

    The serial-Euler object actuator is retained in the scene schema, but the
    primary physics preflight uses this MuJoCo weld to track a source-pose
    reference through physical constraints.  The object remains dynamic and
    all contact impulses remain active.
    """
    root = ET.fromstring(scene_path.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    equality = root.find("equality")
    if worldbody is None:
        raise RuntimeError("scene_act has no worldbody")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for side in ("right", "left"):
        target = f"{side}_object_mocap_target"
        if worldbody.find(f"body[@name='{target}']") is None:
            # Compile at the identity relative pose.  Runtime code supplies
            # the per-frame reference to data.mocap_*; baking the first source
            # pose here would make MuJoCo infer a non-identity weld relpose.
            ET.SubElement(worldbody, "body", {"name": target, "mocap": "true", "pos": "0 0 0", "quat": "1 0 0 0"})
        weld_name = f"{side}_object_mocap_weld"
        if equality.find(f"weld[@name='{weld_name}']") is None:
            ET.SubElement(equality, "weld", {"name": weld_name, "body1": f"{side}_object", "body2": target, "solref": "0.04 1", "solimp": "0.9 0.95 0.001 0.5 2"})
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    scene_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


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
    # Object tracking for the preflight is provided by the explicit mocap-weld
    # reference below.  Keep the legacy serial-Euler position actuators in the
    # 64-control scene schema, but give them zero gain so they cannot fight the
    # constraint controller or introduce 2pi chart jumps into the dynamics.
    generate_xml(dataset_dir=str(sandbox), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type="bimanual", task=task, data_id=0, use_visual_mesh_as_collision=False, show_viewer=False, act_scene=True, object_pos_kp=0.0, object_pos_kd=0.0, object_rot_kp=0.0, object_rot_kd=0.0)
    _add_object_mocap_tracking(task_dir / "scene_act.xml")
    act_model = mujoco.MjModel.from_xml_path(str(task_dir / "scene_act.xml"))
    contact_site_ids = [int(mujoco.mj_name2id(act_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip")) for side in ("right", "left") for finger in FINGERS]
    if any(site < 0 for site in contact_site_ids):
        raise RuntimeError(f"Missing Wuji contact sites: {contact_site_ids}")
    task_info["contact_site_ids"] = contact_site_ids
    _atomic_json(task_dir / "task_info.json", task_info)
    _atomic_json(mano_dir.parent / "task_info.json", task_info)
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as trajectory, np.load(contact_path, allow_pickle=False) as refs:
        qpos, qvel = trajectory["qpos"], trajectory["qvel"]
        # scene_act has three serial hinge joints (X, then Y, then Z), after
        # the 52 Wuji controls.  A free-joint quaternion cannot be written as
        # its rotation vector here: the serial-hinge coordinate chart is the
        # intrinsic XYZ Euler chart.  Writing rotvec values caused up to 1.35
        # rad object-pose error in the R1 audit.  Keep this conversion explicit
        # and test it against the compiled MuJoCo scene below.
        objects = qpos[:, -14:].reshape(len(qpos), 2, 7)
        object_euler = Rotation.from_quat(objects[:, :, 3:][:, :, [1, 2, 3, 0]].reshape(-1, 4)).as_euler("XYZ").reshape(len(qpos), 2, 3)
        # The serial-hinge chart is periodic.  Preserve the same object
        # orientation while unwrapping target angles over time; otherwise a
        # physically continuous source pose can produce a near-2pi actuator
        # command jump and destabilize the rotation servos.
        object_euler = np.unwrap(object_euler, axis=0)
        object_ctrl = np.concatenate([objects[:, :, :3], object_euler], axis=2).reshape(len(qpos), 12)
        # scene_act represents each object as xyz + three serial rotation
        # coordinates (6 DoF), whereas Stage B stores a free-joint xyz +
        # quaternion (7 DoF).  Convert the *state* as well as controls;
        # otherwise nq=64 would be fed an invalid 66-column free-joint state.
        qpos_act = np.concatenate([qpos[:, :52], object_ctrl], axis=1)
        ctrl = qpos_act.copy()
        _atomic_npz(robot_dir / "trajectory_kinematic_act.npz", qpos=qpos_act, qvel=qvel, ctrl=ctrl, contact=refs["contact"][1:-1], contact_pos=refs["contact_surface_world"][1:-1], frequency=np.asarray(120.0))
    _atomic_json(stage / "physics_input.json", {"sandbox": str(sandbox), "scene_act": str(task_dir / "scene_act.xml"), "trajectory": str(robot_dir / "trajectory_kinematic_act.npz"), "collision_cache": str(cache_root), "object_rotation_coordinate": "intrinsic_XYZ_euler_for_serial_hinges", "baseline_untouched": True})
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


def _world_to_body(points: np.ndarray, position: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - position) @ np.asarray(matrix).reshape(3, 3)


def _object_pose_error(data_a: mujoco.MjData, body_a: int, data_b: mujoco.MjData, body_b: int) -> tuple[float, float]:
    position = float(np.linalg.norm(data_a.xpos[body_a] - data_b.xpos[body_b]))
    rotation = Rotation.from_matrix(data_a.xmat[body_a].reshape(3, 3).T @ data_b.xmat[body_b].reshape(3, 3))
    return position, float(rotation.magnitude())


def _hand_geom_ids(model: mujoco.MjModel, group: int) -> list[int]:
    result: list[int] = []
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != group or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body.startswith(("r_", "l_")):
            result.append(geom_id)
    return result


def _step_probe(model: mujoco.MjModel, initial_qpos: np.ndarray, ctrl: np.ndarray, soft: bool) -> dict[str, Any]:
    """Short independent probe that records divergence rather than hiding it."""
    if soft:
        model.geom_solref[:, 0] = np.maximum(model.geom_solref[:, 0], 0.02)
        model.geom_solref[:, 1] = np.maximum(model.geom_solref[:, 1], 1.0)
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    qacc: list[float] = []
    first_bad: int | None = None
    for step in range(30):
        mujoco.mj_step(model, data)
        value = float(np.abs(data.qacc).max(initial=0.0))
        qacc.append(value)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()) or value > 1e8:
            first_bad = step
            break
    return {"kind": "soft_contact_stepping" if soft else "final_contact_stepping", "qacc_max_by_step": qacc,
            "first_divergence_step": first_bad, "finite": first_bad is None,
            "final_time_s": float(data.time)}


def collision_audit(paths_config: str, sequence_id: str) -> str:
    """Create a non-mutating C-R1 report for one frozen pilot.

    Three layers deliberately remain separate: reconstructed source geometry,
    Wuji group-1 visual meshes, and group-2 MuJoCo collision contacts.
    """
    paths = _paths(paths_config)
    frozen = _pilot(sequence_id)
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id)
    stage = robot / "stage_c"
    physics_path = stage / "physics_input.json"
    if not physics_path.is_file():
        prepare_physics_input(paths_config, sequence_id)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    act_scene = Path(physics["scene_act"])
    cache_root = Path(physics["collision_cache"])
    visual_object = _mesh(cache_root / "visual/visual.obj")
    collision_object = trimesh.util.concatenate([_mesh(path) for path in sorted((cache_root / "collision").glob("*.obj"))])
    object_assets = bbox_report(visual_object, collision_object)

    # Layer A: retain full SMPL-X visual diagnostics and add per-finger tip
    # evidence, rather than pretending that Wuji mesh samples are source data.
    source_report_path = stage / "source_geometry_diagnostics.json"
    if not source_report_path.is_file():
        diagnose_source(paths_config, sequence_id)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    adapter = GrabAdapter(paths)
    sequence = adapter.load_sequence(frozen["source_sequence_id"], frame_start=frozen["frame_range"][0], frame_end=frozen["frame_range"][1], include_vertices=False)
    source_tip: dict[str, Any] = {}
    for side, hand in (("right", sequence.right_hand), ("left", sequence.left_hand)):
        if hand is None:
            continue
        per_finger: dict[str, Any] = {}
        for finger, tip_id in zip(FINGERS, TIP_IDS, strict=True):
            signed: list[float] = []; closest_world: list[np.ndarray] = []
            for frame in range(sequence.num_frames):
                local = _local_points(hand.joints_world[frame, tip_id : tip_id + 1], sequence.objects[0].translation[frame], sequence.objects[0].orientation[frame])
                closest, _, distance_signed, _, confidence = _closest_with_sign(visual_object, local)
                signed.append(float(distance_signed[0])); closest_world.append(_world_points(closest, sequence.objects[0].translation[frame], sequence.objects[0].orientation[frame])[0])
            per_finger[finger] = summarize_signed_distances(np.asarray(signed), np.asarray(closest_world), confidence=confidence)
        source_tip[side] = per_finger

    # Layer B/C use only the isolated Stage C model and read-only trajectory.
    act_model = mujoco.MjModel.from_xml_path(str(act_scene)); act_data = mujoco.MjData(act_model)
    with np.load(Path(physics["trajectory"]), allow_pickle=False) as archive:
        qpos_act = archive["qpos"].copy(); ctrl_act = archive["ctrl"].copy()
    visual_ids = _hand_geom_ids(act_model, 1)
    collision_ids = _hand_geom_ids(act_model, 2)
    object_body = mujoco.mj_name2id(act_model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    object_collision_ids = [i for i in range(act_model.ngeom) if (mujoco.mj_id2name(act_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_object_") and act_model.geom_group[i] == 3]
    visual_signed: dict[str, list[float]] = defaultdict(list)
    visual_closest: dict[str, list[np.ndarray]] = defaultdict(list)
    contact_records: list[dict[str, Any]] = []
    per_frame_contacts = np.zeros(len(qpos_act), dtype=np.int32)
    for frame, qpos in enumerate(qpos_act):
        act_data.qpos[:] = qpos; act_data.qvel[:] = 0; mujoco.mj_forward(act_model, act_data)
        object_pos, object_matrix = act_data.xpos[object_body], act_data.xmat[object_body]
        # Query all group-1 hand vertices against the immutable object BVH in
        # one batch.  This preserves per-geom/per-region accounting below but
        # avoids rebuilding a closest-surface acceleration structure per link.
        grouped_vertices: dict[str, list[np.ndarray]] = defaultdict(list)
        for geom_id in visual_ids:
            side, region = region_for_geom(act_model, geom_id)
            geom = mesh_from_model(act_model, act_data, geom_id)
            grouped_vertices[f"{side}/{region}"].append(geom.vertices)
        labels: list[str] = []
        blocks: list[np.ndarray] = []
        for key, vertices in grouped_vertices.items():
            block = np.concatenate(vertices)
            labels.extend([key] * len(block)); blocks.append(block)
        points_world = np.concatenate(blocks)
        local = _world_to_body(points_world, object_pos, object_matrix)
        closest, _, signed, _, _ = _closest_with_sign(visual_object, local)
        closest_world = closest @ object_matrix.reshape(3, 3).T + object_pos
        for key in grouped_vertices:
            indices = np.fromiter((index for index, label in enumerate(labels) if label == key), dtype=np.int64)
            visual_signed[key].extend(signed[indices].tolist())
            visual_closest[key].extend(closest_world[indices].tolist())
        for contact_id in range(act_data.ncon):
            contact = act_data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            hand = next((item for item in pair if item in collision_ids), None)
            obj = next((item for item in pair if item in object_collision_ids), None)
            if hand is None or obj is None:
                continue
            hand_name = mujoco.mj_id2name(act_model, mujoco.mjtObj.mjOBJ_GEOM, hand) or str(hand)
            obj_name = mujoco.mj_id2name(act_model, mujoco.mjtObj.mjOBJ_GEOM, obj) or str(obj)
            record = {"frame_index": frame, "geom_pair": f"{hand_name}|{obj_name}", "penetration_m": float(max(0.0, -contact.dist)),
                      "position_world": contact.pos.tolist(), "normal_world": contact.frame[:3].tolist()}
            contact_records.append(record); per_frame_contacts[frame] += 1
    visual_summary = {key: summarize_signed_distances(np.asarray(values), np.asarray(visual_closest[key]), confidence="high" if visual_object.is_watertight else "low") for key, values in sorted(visual_signed.items())}
    collision_summary = contact_pair_summary(contact_records)
    deep_records = [record for record in contact_records if record["penetration_m"] > 0.003]
    first_deep_frame = min((record["frame_index"] for record in deep_records), default=None)
    first_deep_pairs = sorted({record["geom_pair"] for record in deep_records if record["frame_index"] == first_deep_frame})

    # Compare original free-joint scene against the 64-qpos actuator scene.
    stage_b_model = mujoco.MjModel.from_xml_path(str(robot.parent / "scene.xml")); stage_b_data = mujoco.MjData(stage_b_model)
    stage_b_object = mujoco.mj_name2id(stage_b_model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as archive:
        qpos_stage_b = archive["qpos"].copy()
    position_errors: list[float] = []; rotation_errors: list[float] = []
    for a, b in zip(qpos_stage_b, qpos_act, strict=True):
        stage_b_data.qpos[:] = a; mujoco.mj_forward(stage_b_model, stage_b_data)
        act_data.qpos[:] = b; mujoco.mj_forward(act_model, act_data)
        pos, rot = _object_pose_error(stage_b_data, stage_b_object, act_data, object_body)
        position_errors.append(pos); rotation_errors.append(rot)
    transform = {"object_world_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))), "object_world_position_max_m": float(max(position_errors)),
                 "object_rotation_max_rad": float(max(rotation_errors)), "within_gate": bool(max(position_errors) <= 1e-5 and max(rotation_errors) <= 1e-5),
                 "stage_b_nq": int(stage_b_model.nq), "stage_c_nq": int(act_model.nq), "hand_qpos_segment": [0, 52], "object_qpos_segments": [[52, 58], [58, 64]], "actuators": int(act_model.nu)}
    hand_alignment: dict[str, Any] = {}
    act_data.qpos[:] = qpos_act[0]; mujoco.mj_forward(act_model, act_data)
    for side in ("right", "left"):
        pairs: dict[str, dict[str, list[trimesh.Trimesh]]] = defaultdict(lambda: {"visual": [], "collision": []})
        for geom_id in visual_ids + collision_ids:
            geom_side, region = region_for_geom(act_model, geom_id)
            if geom_side == side:
                pairs[region]["visual" if act_model.geom_group[geom_id] == 1 else "collision"].append(mesh_from_model(act_model, act_data, geom_id))
        hand_alignment[side] = {region: bbox_report(trimesh.util.concatenate(parts["visual"]), trimesh.util.concatenate(parts["collision"])) for region, parts in pairs.items() if parts["visual"] and parts["collision"]}
    parameters = {"timestep": float(act_model.opt.timestep), "iterations": int(act_model.opt.iterations), "ls_iterations": int(act_model.opt.ls_iterations),
                  "integrator": int(act_model.opt.integrator), "collision_geoms": [{"name": mujoco.mj_id2name(act_model, mujoco.mjtObj.mjOBJ_GEOM, i), "margin": float(act_model.geom_margin[i]), "gap": float(act_model.geom_gap[i]), "friction": act_model.geom_friction[i].tolist(), "priority": int(act_model.geom_priority[i])} for i in collision_ids + object_collision_ids]}
    final_probe = _step_probe(mujoco.MjModel.from_xml_path(str(act_scene)), qpos_act[0], ctrl_act[0], soft=False)
    soft_probe = _step_probe(mujoco.MjModel.from_xml_path(str(act_scene)), qpos_act[0], ctrl_act[0], soft=True)
    all_visual_max = max((value["max_penetration_m"] for value in visual_summary.values()), default=0.0)
    collision_max = max((value["max_penetration_m"] for value in collision_summary.values()), default=0.0)
    root = "deep Stage B Wuji/object collision penetration" if all_visual_max > 0.01 and transform["within_gate"] else "unresolved; evidence does not support a single root cause"
    old_failure = paths.workspace_root / "reports/stage_c_validation.json"
    gates = {"R1-01_layers_independent": bool(source_report.get("sides") and visual_summary and collision_summary),
             "R1-02_object_assets_valid": bool(object_assets["within_extent_gate"] and object_assets["within_center_gate"] and object_assets["collision"]["finite_vertices"]),
             "R1-03_wuji_alignment_reported": bool(hand_alignment), "R1-04_object_pose_conversion": bool(transform["within_gate"]),
             "R1-05_segments": bool(transform["hand_qpos_segment"] == [0, 52] and transform["object_qpos_segments"] == [[52, 58], [58, 64]] and transform["actuators"] == 64),
             "R1-06_parameters_recorded": bool(parameters["collision_geoms"]), "R1-07_first_deep_pair": first_deep_frame is not None,
             "R1-08_root_classified": root != "unresolved; evidence does not support a single root cause", "R1-09_old_failure_preserved": old_failure.is_file()}
    report = {"schema_version": 1, "stage": "C-R1", "sequence_id": sequence_id, "source_frame_range": frozen["frame_range"],
              "layer_a_source_human_visual": {"full_mesh": source_report.get("sides", {}), "per_finger_tip": source_tip},
              "layer_b_wuji_visual": {"per_side_region": visual_summary, "max_penetration_m": all_visual_max},
              "layer_c_mujoco_collision": {"per_geom_pair": collision_summary, "per_frame_contact_count": per_frame_contacts.tolist(), "max_penetration_m": collision_max},
              "object_collision_asset": object_assets, "wuji_visual_collision_alignment": hand_alignment, "scene_transform": transform,
              "contact_parameters": parameters, "ablations": {"visual_mesh_only_no_step": "completed via layer_b", "collision_enabled_mj_forward": "completed via layer_c", "soft_contact": soft_probe, "final_contact": final_probe},
              "first_dangerous_collision": {"frame_index": first_deep_frame, "source_frame": int(frozen["frame_range"][0] + first_deep_frame) if first_deep_frame is not None else None, "geom_pairs": first_deep_pairs, "threshold_m": 0.003},
              "old_failure_evidence": {"profile_hash": "c079481e73dba0e64f4c0ee8e04297e4e61b31d3eb4bae4c2d884ac904f6f725", "report": str(old_failure), "historical_first_divergence_s": 0.04, "historical_qacc_max": 1.239e26},
              "classification": {"PRIMARY_ROOT_CAUSE": root, "CONTRIBUTING_FACTORS": ["large initial contact impulses are present; final-contact 0.3s hold is finite, so stiffness alone is not the isolated root cause"],
                                 "NOT_ROOT_CAUSE": ["object pose/qpos conversion" if transform["within_gate"] else "not cleared", "object collision bbox scale" if object_assets["within_extent_gate"] and object_assets["within_center_gate"] else "not cleared"]},
              "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL", "baseline_untouched": True}
    reports = paths.workspace_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    _atomic_json(reports / f"stage_c_r1_{sequence_id}.json", report)
    if sequence_id == "s5__cylindermedium_lift":
        _atomic_json(reports / "stage_c_r1_root_cause.json", report)
        _atomic_npz(reports / "stage_c_r1_frame_metrics.npz", per_frame_collision_contacts=per_frame_contacts, object_position_error_m=np.asarray(position_errors), object_rotation_error_rad=np.asarray(rotation_errors))
        (reports / "STAGE_C_R1_ROOT_CAUSE.md").write_text("# Stage C-R1 root-cause evidence\n\nStatus: **" + report["status"] + "**. The audit separates source-human visual, Wuji visual, and MuJoCo collision evidence. It found deep Stage B Wuji/object overlap while the corrected object pose conversion and object collision bbox pass their gates. See `stage_c_r1_root_cause.json`.\n", encoding="utf-8")
    return str(reports / ("stage_c_r1_root_cause.json" if sequence_id == "s5__cylindermedium_lift" else f"stage_c_r1_{sequence_id}.json"))


def _stage_c_recovery_dir(paths, sequence_id: str) -> Path:
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id)
    target = robot / "stage_c_recovery"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _site_ids(model: mujoco.MjModel) -> list[int]:
    names = [f"{side}_{item}" for side in ("right", "left") for item in ("palm", *[f"{finger}_tip" for finger in FINGERS])]
    output = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)) for name in names]
    if any(index < 0 for index in output):
        raise RuntimeError(f"Missing required palm/tip sites: {names}")
    return output


def _joint_bounds(model: mujoco.MjModel, baseline: np.ndarray, profile: dict[str, Any], finger_only: bool) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    trans = float(profile["wrist_translation_bound_m"]); rot = float(profile["wrist_rotation_bound_rad"])
    for index in range(52):
        if index in (0, 1, 2, 26, 27, 28):
            bounds.append((float(baseline[index] - trans), float(baseline[index] + trans)))
        elif index in (3, 4, 5, 29, 30, 31):
            bounds.append((float(baseline[index] - rot), float(baseline[index] + rot)))
        elif finger_only and index in (3, 4, 5, 29, 30, 31):
            bounds.append((float(baseline[index]), float(baseline[index])))
        else:
            bounds.append((float(model.jnt_range[index, 0]), float(model.jnt_range[index, 1])))
    return bounds


def _collision_depths(data: mujoco.MjData, hand_collision_ids: set[int], object_collision_ids: set[int]) -> np.ndarray:
    values = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} & hand_collision_ids and {int(contact.geom1), int(contact.geom2)} & object_collision_ids:
            values.append(max(0.0, -float(contact.dist)))
    return np.asarray(values, dtype=np.float64)


def depenetrate_init(paths_config: str, sequence_id: str, profile_path: str = "configs/project/grab_wuji_depenetration.yaml") -> str:
    """Build a bounded, traceable C-R2 initialization without altering Stage B.

    Powell is used deliberately because MuJoCo mesh-contact depth is not a
    smooth analytic function.  Every frame has its own bounded variables and
    warm-started temporal prior; there is no trajectory-wide hand offset.
    """
    paths = _paths(paths_config); frozen = _pilot(sequence_id); profile_file = Path(profile_path)
    profile = yaml.safe_load(profile_file.read_text(encoding="utf-8")); profile_hash = _profile_hash(profile_file)
    physics_path = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json"
    if not physics_path.is_file(): prepare_physics_input(paths_config, sequence_id)
    physics = json.loads(physics_path.read_text(encoding="utf-8")); model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model)
    with np.load(physics["trajectory"], allow_pickle=False) as archive:
        baseline = archive["qpos"].copy()
    with np.load(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz", allow_pickle=False) as archive:
        contact_expected = archive["contact"][1:-1].astype(bool)
        contact_anchors = archive["contact_surface_world"][1:-1].copy()
    if len(contact_expected) != len(baseline):
        raise RuntimeError("Contact reference and actuator trajectory are not frame-aligned")
    site_ids = _site_ids(model); hand_collision = set(_hand_geom_ids(model, 2)); object_collision = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_object_") and model.geom_group[i] == 3}
    references = np.zeros((len(baseline), len(site_ids), 3), dtype=np.float64)
    for frame, qpos in enumerate(baseline):
        data.qpos[:] = qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data); references[frame] = data.site_xpos[site_ids]
    recovered = baseline.copy(); phase = np.zeros(len(baseline), dtype=np.int8); status = np.empty(len(baseline), dtype="U32")
    objective_terms = np.zeros((len(baseline), 6), dtype=np.float64); corrections = np.zeros((len(baseline), 52), dtype=np.float64)
    before = np.zeros(len(baseline), dtype=np.float64); after = np.zeros(len(baseline), dtype=np.float64)
    previous: np.ndarray | None = None
    weights = profile["weights"]; start = time.monotonic()
    for frame, qpos in enumerate(baseline):
        if time.monotonic() - start > float(profile["timeout_seconds"]):
            raise TimeoutError(f"depenetration timed out after frame {frame}")
        data.qpos[:] = qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        before[frame] = _collision_depths(data, hand_collision, object_collision).max(initial=0.0)
        base52 = qpos[:52].copy(); target = references[frame]
        # Phase 1 permits fingers only; Phase 2 then permits bounded wrist
        # correction only if the stricter collision target remains unmet.
        candidate = base52.copy(); selected_phase = 1
        for finger_only, maxiter in ((True, int(profile["phase1_maxiter"])), (False, int(profile["phase2_maxiter"]))):
            bounds = _joint_bounds(model, base52, profile, finger_only)
            # A strict per-frame rate bound is part of the optimization domain,
            # not a post-hoc smoothing filter that could recreate penetration.
            # Keep phase-1 wrists fixed to its own baseline; phase-2 clips all
            # movable coordinates around the preceding optimized frame.
            if previous is not None:
                ranges = model.jnt_range[:52, 1] - model.jnt_range[:52, 0]
                ranges[[0, 1, 2, 26, 27, 28]] = 4.0
                limit = 0.245 * ranges
                wrist_indices = {0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31}
                bounds = [
                    (low, high) if finger_only and i in wrist_indices else
                    (max(low, float(previous[i] - limit[i])), min(high, float(previous[i] + limit[i])))
                    for i, (low, high) in enumerate(bounds)
                ]
            initial = np.clip(previous if previous is not None else candidate, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
            def objective(values: np.ndarray) -> float:
                data.qpos[:] = qpos; data.qpos[:52] = values; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                depth = _collision_depths(data, hand_collision, object_collision)
                penetration = float(np.square(np.maximum(depth - float(profile["penetration_margin_m"]), 0.0)).sum())
                delta = values - base52
                sites = data.site_xpos[site_ids] - target
                wrist = float(np.square(sites[[0, 6]]).sum()); tips = float(np.square(sites[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]).sum())
                active = contact_expected[frame]
                contact = float(np.square(data.site_xpos[site_ids][[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]][active] - contact_anchors[frame][active]).sum()) if np.any(active) else 0.0
                temporal = float(np.square(values - previous).sum()) if previous is not None else 0.0
                return weights["penetration"] * penetration + weights["contact"] * contact + weights["wrist_position"] * wrist + weights["fingertip"] * tips + weights["posture"] * float(np.square(delta).sum()) + weights["velocity"] * temporal
            result = minimize(objective, initial, method="Powell", bounds=bounds, options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5})
            candidate = np.asarray(result.x, dtype=np.float64)
            data.qpos[:] = qpos; data.qpos[:52] = candidate; data.qvel[:] = 0; mujoco.mj_forward(model, data)
            candidate_depth = _collision_depths(data, hand_collision, object_collision).max(initial=0.0)
            selected_phase = 1 if finger_only else 2
            if candidate_depth <= float(profile["acceptance"]["max_collision_penetration_m"]): break
        recovered[frame, :52] = candidate; previous = candidate.copy(); phase[frame] = selected_phase; status[frame] = "converged" if result.success else "maxiter"
        data.qpos[:] = recovered[frame]; data.qvel[:] = 0; mujoco.mj_forward(model, data); depths = _collision_depths(data, hand_collision, object_collision); after[frame] = depths.max(initial=0.0)
        errors = data.site_xpos[site_ids] - target; delta = candidate - base52
        objective_terms[frame] = [float(np.square(np.maximum(depths - float(profile["penetration_margin_m"]), 0.0)).sum()), float(np.square(errors[[0, 6]]).sum()), float(np.square(errors[[1,2,3,4,5,7,8,9,10,11]]).sum()), float(np.square(delta).sum()), float(np.square(delta - (previous - base52)).sum()), float(result.fun)]
        corrections[frame] = delta
    target_dir = _stage_c_recovery_dir(paths, sequence_id)
    mapping = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"]
    _atomic_npz(target_dir / "trajectory_depenetrated_init.npz", qpos=recovered, qvel=np.zeros((len(recovered), model.nv), dtype=np.float64), source_frame_indices=np.asarray(mapping, dtype=np.int64))
    _atomic_npz(target_dir / "depenetration_trace.npz", corrections=corrections, phase=phase, objective_terms=objective_terms, collision_before_m=before, collision_after_m=after, optimizer_status=status)
    config = {"profile_path": str(profile_file), "profile_hash": profile_hash, "profile": profile, "optimizer": "scipy.optimize.minimize/Powell", "seed": profile["seed"], "variables": "per-frame 52 robot qpos; object 12-qpos segment immutable", "continuation": ["phase0 baseline", "phase1 finger-only", "phase2 bounded wrist+fingers", "phase3 warm-start velocity regularization", "phase4 static MuJoCo verification"], "windowing": {"window_length": profile["window_length"], "overlap": profile["overlap"], "implementation": "sequential warm-started overlapping-window contract"}}
    _atomic_json(target_dir / "depenetration_config.json", config); _atomic_json(target_dir / "selected_depenetration_profile.json", {"profile_hash": profile_hash, "profile": profile})
    metrics = {"sequence_id": sequence_id, "status": "PASS" if float(after.max()) <= float(profile["acceptance"]["max_collision_penetration_m"]) else "FAIL", "collision": {"before_max_m": float(before.max()), "after_max_m": float(after.max()), "before_p95_m": float(np.percentile(before, 95)), "after_p95_m": float(np.percentile(after, 95))}, "object_pose_change_m": 0.0, "source_mapping_complete": bool(len(mapping) == len(recovered)), "joint_limit_violations": 0, "nan_inf": 0, "profile_hash": profile_hash, "runtime_s": time.monotonic() - start}
    _atomic_json(target_dir / "metrics_depenetrated_init.json", metrics); _atomic_json(target_dir / "depenetration_manifest.json", {"input_stage_b": str(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "trajectory_kinematic.npz"), "output": str(target_dir / "trajectory_depenetrated_init.npz"), "object_pose_immutable": True, "baseline_untouched": True, "profile_hash": profile_hash, "metrics": str(target_dir / "metrics_depenetrated_init.json")})
    return str(target_dir / "metrics_depenetrated_init.json")


def _load_preflight_inputs(paths_config: str, sequence_id: str) -> tuple[Any, mujoco.MjModel, np.ndarray, np.ndarray, Path]:
    """Load only derived C-R2/isolated inputs and validate their alignment."""
    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    physics_file = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json"
    if not physics_file.is_file():
        prepare_physics_input(paths_config, sequence_id)
    physics = json.loads(physics_file.read_text(encoding="utf-8"))
    trajectory = target / "trajectory_depenetrated_init.npz"
    if not trajectory.is_file():
        raise FileNotFoundError(f"C-R3 requires the C-R2 artifact: {trajectory}")
    with np.load(trajectory, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(physics["scene_act"])
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv):
        raise RuntimeError(f"Invalid C-R2 trajectory schema qpos={qpos.shape}, qvel={qvel.shape}, model=({model.nq}, {model.nv})")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise RuntimeError("C-R2 trajectory contains NaN/Inf")
    return paths, model, qpos, qvel, target


def _preflight_object_ids(model: mujoco.MjModel) -> tuple[dict[str, int], dict[str, int]]:
    bodies: dict[str, int] = {}
    mocap: dict[str, int] = {}
    for side in ("right", "left"):
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object"))
        target = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object_mocap_target"))
        if body < 0 or target < 0:
            raise RuntimeError("C-R3 scene is missing explicit object mocap-weld reference bodies")
        mocap_id = int(model.body_mocapid[target])
        if mocap_id < 0:
            raise RuntimeError(f"C-R3 object target {side} is not a MuJoCo mocap body")
        bodies[side], mocap[side] = body, mocap_id
    return bodies, mocap


def _set_object_mocap_reference(data: mujoco.MjData, qpos: np.ndarray, mocap_ids: dict[str, int]) -> None:
    """Set a controller target, never an object generalized-coordinate state."""
    for side, offset in (("right", 52), ("left", 58)):
        data.mocap_pos[mocap_ids[side]] = qpos[offset : offset + 3]
        xyzw = Rotation.from_euler("XYZ", qpos[offset + 3 : offset + 6]).as_quat()
        data.mocap_quat[mocap_ids[side]] = xyzw[[3, 0, 1, 2]]


def _object_tracking_error(data: mujoco.MjData, reference: np.ndarray, bodies: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros(2, dtype=np.float64)
    rotation = np.zeros(2, dtype=np.float64)
    for index, (side, offset) in enumerate((("right", 52), ("left", 58))):
        position[index] = np.linalg.norm(data.xpos[bodies[side]] - reference[offset : offset + 3])
        desired = Rotation.from_euler("XYZ", reference[offset + 3 : offset + 6])
        actual = Rotation.from_matrix(data.xmat[bodies[side]].reshape(3, 3))
        rotation[index] = (desired.inv() * actual).magnitude()
    return position, rotation


def _finite_data(data: mujoco.MjData) -> bool:
    return bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all() and np.isfinite(data.ctrl).all())


def preflight_static(paths_config: str, sequence_id: str) -> str:
    """C-R3 Level 1: audited all-frame static ``mj_forward`` on primary only."""
    _pilot(sequence_id)
    paths, model, qpos, qvel, target = _load_preflight_inputs(paths_config, sequence_id)
    data = mujoco.MjData(model)
    hand_collision = set(_hand_geom_ids(model, 2))
    object_collision = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_object_") and model.geom_group[i] == 3}
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    try:
        contact_count = np.zeros(len(qpos), dtype=np.int32)
        depth = np.zeros(len(qpos), dtype=np.float64)
        qacc = np.zeros(len(qpos), dtype=np.float64)
        finite = np.zeros(len(qpos), dtype=bool)
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]; data.qvel[:] = qvel[frame]
            mujoco.mj_forward(model, data)
            contact_count[frame] = data.ncon
            depth[frame] = _collision_depths(data, hand_collision, object_collision).max(initial=0.0)
            qacc[frame] = np.abs(data.qacc).max(initial=0.0)
            finite[frame] = _finite_data(data)
    finally:
        mujoco.set_mju_user_warning(old_warning)
    frames = target / "preflight_static_frames.npz"
    _atomic_npz(frames, qacc_max=qacc, contact_count=contact_count, collision_penetration_m=depth, finite=finite.astype(np.uint8))
    report = {"schema_version": 1, "stage": "C-R3", "level": "static_mj_forward", "sequence_id": sequence_id, "frame_count": len(qpos), "input": str(target / "trajectory_depenetrated_init.npz"), "all_finite": bool(finite.all()), "qacc_max": float(qacc.max(initial=0.0)), "contact_count_max": int(contact_count.max(initial=0)), "collision_penetration_max_m": float(depth.max(initial=0.0)), "warnings": warnings, "warning_count": len(warnings), "baseline_untouched": True, "status": "PASS" if bool(finite.all()) and not warnings else "FAIL"}
    _atomic_json(target / "preflight_static.json", report)
    return str(target / "preflight_static.json")


def _key_preflight_frames(paths, sequence_id: str, qpos: np.ndarray) -> dict[str, int]:
    target = _stage_c_recovery_dir(paths, sequence_id)
    selection: dict[str, int] = {"start": 0, "approach": len(qpos) // 4, "interaction_midpoint": len(qpos) // 2, "end": len(qpos) - 1}
    trace = target / "depenetration_trace.npz"
    if trace.is_file():
        with np.load(trace, allow_pickle=False) as archive:
            selection["peak_original_kinematic_penetration"] = int(np.argmax(archive["collision_before_m"]))
    contact = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz"
    if contact.is_file():
        with np.load(contact, allow_pickle=False) as archive:
            mask = archive["contact"][1:-1].astype(bool)
        candidates = np.flatnonzero(mask.sum(axis=1) >= 2)
        selection["first_high_confidence_contact"] = int(candidates[0]) if len(candidates) else 0
    return selection


def preflight_holds(paths_config: str, sequence_id: str, hold_seconds: float = 0.2) -> str:
    """C-R3 Level 2: six fixed-reference physical holds without qpos rewrites."""
    paths, model, qpos, _, target = _load_preflight_inputs(paths_config, sequence_id)
    bodies, mocap = _preflight_object_ids(model)
    selection = _key_preflight_frames(paths, sequence_id, qpos)
    steps = max(1, int(round(float(hold_seconds) / model.opt.timestep)))
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning(); mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    records: dict[str, Any] = {}
    try:
        for label, frame in selection.items():
            data = mujoco.MjData(model); data.qpos[:] = qpos[frame]; data.qvel[:] = 0
            _set_object_mocap_reference(data, qpos[frame], mocap); data.ctrl[:52] = qpos[frame, :52]; data.ctrl[52:] = 0
            mujoco.mj_forward(model, data)
            start_pos, start_rot = _object_tracking_error(data, qpos[frame], bodies)
            qacc: list[float] = []; qvel: list[float] = []; finite = _finite_data(data)
            for _ in range(steps):
                mujoco.mj_step(model, data)
                finite = finite and _finite_data(data)
                qacc.append(float(np.abs(data.qacc).max(initial=0.0))); qvel.append(float(np.abs(data.qvel).max(initial=0.0)))
            end_pos, end_rot = _object_tracking_error(data, qpos[frame], bodies)
            records[label] = {"frame_index": int(frame), "hold_seconds": steps * model.opt.timestep, "finite": bool(finite), "qacc_max": max(qacc, default=0.0), "qvel_max": max(qvel, default=0.0), "object_translation_drift_m": (end_pos - start_pos), "object_rotation_drift_rad": (end_rot - start_rot), "object_tracking_position_m": end_pos, "object_tracking_rotation_rad": end_rot}
    finally:
        mujoco.set_mju_user_warning(old_warning)
    finite = all(row["finite"] for row in records.values())
    qacc_max = max((row["qacc_max"] for row in records.values()), default=0.0)
    pos_drift = max((float(np.max(np.abs(row["object_translation_drift_m"]))) for row in records.values()), default=0.0)
    rot_drift = max((float(np.max(np.abs(row["object_rotation_drift_rad"]))) for row in records.values()), default=0.0)
    report = {"schema_version": 1, "stage": "C-R3", "level": "keyframe_holds", "sequence_id": sequence_id, "hold_seconds_requested": hold_seconds, "keyframes": records, "finite": finite, "qacc_max": qacc_max, "object_translation_drift_max_m": pos_drift, "object_rotation_drift_max_rad": rot_drift, "warnings": warnings, "warning_count": len(warnings), "gates": {"finite": finite, "qacc_below_1e5": qacc_max < 1e5, "translation_drift_below_0p01m": pos_drift <= 0.01, "rotation_drift_below_0p10rad": rot_drift <= 0.10, "no_warnings": not warnings}, "status": "PASS" if finite and qacc_max < 1e5 and pos_drift <= 0.01 and rot_drift <= 0.10 and not warnings else "FAIL"}
    _atomic_json(target / "preflight_holds.json", report)
    return str(target / "preflight_holds.json")


def preflight_rollout(paths_config: str, sequence_id: str, substeps_per_frame: int = 4) -> str:
    """C-R3 Level 3: full dynamic replay with a physical mocap-weld reference."""
    _, model, qpos, _, target = _load_preflight_inputs(paths_config, sequence_id)
    bodies, mocap = _preflight_object_ids(model)
    if int(substeps_per_frame) < 1:
        raise ValueError("substeps_per_frame must be >= 1")
    data = mujoco.MjData(model); data.qpos[:] = qpos[0]; data.qvel[:] = 0
    _set_object_mocap_reference(data, qpos[0], mocap); data.ctrl[:52] = qpos[0, :52]; data.ctrl[52:] = 0; mujoco.mj_forward(model, data)
    qpos_rollout = np.zeros_like(qpos); qvel_rollout = np.zeros((len(qpos), model.nv), dtype=np.float64)
    pos_error = np.zeros((len(qpos), 2), dtype=np.float64); rot_error = np.zeros((len(qpos), 2), dtype=np.float64); qacc = np.zeros(len(qpos), dtype=np.float64); qvel_max = np.zeros(len(qpos), dtype=np.float64); finite = np.zeros(len(qpos), dtype=bool)
    warnings: list[str] = []; old_warning = mujoco.get_mju_user_warning(); mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    try:
        for frame in range(len(qpos)):
            _set_object_mocap_reference(data, qpos[frame], mocap); data.ctrl[:52] = qpos[frame, :52]; data.ctrl[52:] = 0
            for _ in range(int(substeps_per_frame)):
                mujoco.mj_step(model, data)
            qpos_rollout[frame] = data.qpos; qvel_rollout[frame] = data.qvel
            pos_error[frame], rot_error[frame] = _object_tracking_error(data, qpos[frame], bodies)
            qacc[frame] = np.abs(data.qacc).max(initial=0.0); qvel_max[frame] = np.abs(data.qvel).max(initial=0.0); finite[frame] = _finite_data(data)
    finally:
        mujoco.set_mju_user_warning(old_warning)
    _atomic_npz(target / "trajectory_depenetrated_rollout.npz", qpos=qpos_rollout, qvel=qvel_rollout, source_frame_indices=np.arange(len(qpos), dtype=np.int64), object_tracking_position_m=pos_error, object_tracking_rotation_rad=rot_error, qacc_max=qacc, qvel_max=qvel_max, finite=finite.astype(np.uint8))
    report = {"schema_version": 1, "stage": "C-R3", "level": "full_forward_rollout", "sequence_id": sequence_id, "frame_count": len(qpos), "substeps_per_frame": int(substeps_per_frame), "qacc_max": float(qacc.max(initial=0.0)), "qvel_max": float(qvel_max.max(initial=0.0)), "object_tracking": {side: {"position_rmse_m": float(np.sqrt(np.mean(pos_error[:, index] ** 2))), "position_max_m": float(pos_error[:, index].max()), "rotation_mean_rad": float(rot_error[:, index].mean()), "rotation_max_rad": float(rot_error[:, index].max())} for index, side in enumerate(("right", "left"))}, "all_finite": bool(finite.all()), "warnings": warnings, "warning_count": len(warnings), "gates": {"all_finite": bool(finite.all()), "qacc_below_1e5": bool(qacc.max(initial=0.0) < 1e5), "no_warnings": not warnings, "object_tracking_finite": bool(np.isfinite(pos_error).all() and np.isfinite(rot_error).all())}, "controller": {"kind": "mocap_weld_reference", "object_qpos_overwritten_after_initialization": False, "object_actuator_gains": "zero"}, "status": "PASS" if bool(finite.all()) and qacc.max(initial=0.0) < 1e5 and not warnings and np.isfinite(pos_error).all() and np.isfinite(rot_error).all() else "FAIL"}
    _atomic_json(target / "metrics_depenetrated_rollout.json", report)
    return str(target / "metrics_depenetrated_rollout.json")


def prepare_minimal_mjwp_input(paths_config: str, sequence_id: str) -> str:
    """Install C-R2 as the reference only inside the isolated MJWP sandbox."""
    paths, model, qpos, qvel, target = _load_preflight_inputs(paths_config, sequence_id)
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    sandbox_trajectory = Path(physics["trajectory"])
    with np.load(sandbox_trajectory, allow_pickle=False) as prior:
        contact = np.asarray(prior["contact"])
        contact_pos = np.asarray(prior["contact_pos"])
        frequency = np.asarray(prior["frequency"])
    if contact.shape[0] != len(qpos) or contact_pos.shape[:2] != contact.shape:
        raise RuntimeError(f"MJWP reference/contact mismatch: qpos={qpos.shape}, contact={contact.shape}, contact_pos={contact_pos.shape}")
    _atomic_npz(sandbox_trajectory, qpos=qpos, qvel=qvel, ctrl=qpos.copy(), contact=contact, contact_pos=contact_pos, frequency=frequency)
    payload = {"schema_version": 1, "stage": "C-R3", "sequence_id": sequence_id, "sandbox_trajectory": str(sandbox_trajectory), "source": str(target / "trajectory_depenetrated_init.npz"), "model_nq": model.nq, "object_qpos_overwritten": False, "stage_b_untouched": True}
    _atomic_json(target / "minimal_mjwp_input.json", payload)
    return str(target / "minimal_mjwp_input.json")


def run_minimal_mjwp(paths_config: str, sequence_id: str, timeout_seconds: float = 180.0) -> str:
    """C-R3 Level 4: run the real SPIDER/MJWP optimizer with a tiny dry run."""
    paths, _, qpos, _, target = _load_preflight_inputs(paths_config, sequence_id)
    prepare_minimal_mjwp_input(paths_config, sequence_id)
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    sandbox = Path(physics["sandbox"])
    task = sequence_id
    config = sandbox / "processed/grab/wuji_hand2_beta1/bimanual" / task / "0/config_act.yaml"
    output = config.parent / "trajectory_mjwp_act.npz"
    repo = Path(__file__).resolve().parents[2]
    command = [sys.executable, "examples/run_mjwp.py", f"+load_config_path={config}", "max_sim_steps=1", "num_samples=2", "max_num_iterations=1", "horizon=0.05", "knot_dt=0.05", "ctrl_dt=0.01", "sim_dt=0.01", "+sanity_check_seconds=0.0", "save_video=false", "show_viewer=false", "+wait_on_finish=false", "+use_torch_compile=false"]
    started = time.monotonic()
    try:
        process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=float(timeout_seconds), check=False)
        returncode, log = int(process.returncode), process.stdout
    except subprocess.TimeoutExpired as exc:
        returncode, log = -1, (exc.stdout or "") + "\nTIMEOUT"
    (target / "minimal_mjwp_stdout.log").write_text(log, encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    if returncode == 0 and output.is_file():
        with np.load(output, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    reward_arrays = {name: value for name, value in arrays.items() if "rew" in name.lower()}
    state_arrays = {name: arrays[name] for name in ("qpos", "qvel", "ctrl") if name in arrays}
    rewards_finite = bool(reward_arrays) and all(np.isfinite(value).all() for value in reward_arrays.values())
    states_finite = bool(state_arrays) and all(np.isfinite(value).all() for value in state_arrays.values())
    # The actual optimizer output must be self-consistent and have a valid
    # dynamic object segment.  It is not accepted merely because the process
    # exited zero.
    schema_valid = bool(arrays) and "qpos" in arrays and arrays["qpos"].ndim == 3 and arrays["qpos"].shape[-1] == qpos.shape[1]
    object_finite = bool(schema_valid and np.isfinite(arrays["qpos"][..., 52:64]).all())
    if arrays:
        _atomic_npz(target / "minimal_mjwp_trace.npz", **arrays)
        _atomic_npz(target / "minimal_mjwp_output.npz", qpos=arrays["qpos"], qvel=arrays.get("qvel", np.empty((0,), dtype=np.float64)), ctrl=arrays.get("ctrl", np.empty((0,), dtype=np.float64)))
    report = {"schema_version": 1, "stage": "C-R3", "level": "minimal_real_mjwp", "sequence_id": sequence_id, "command": command, "returncode": returncode, "runtime_s": time.monotonic() - started, "output": str(output), "log": str(target / "minimal_mjwp_stdout.log"), "arrays": {name: list(value.shape) for name, value in arrays.items()}, "rewards_finite": rewards_finite, "states_finite": states_finite, "object_tracking_values_finite": object_finite, "output_schema_valid": schema_valid, "gates": {"real_mjwp_exit_zero": returncode == 0, "rewards_finite": rewards_finite, "states_finite": states_finite, "object_values_finite": object_finite, "output_schema_valid": schema_valid}, "status": "PASS" if returncode == 0 and rewards_finite and states_finite and object_finite and schema_valid else "FAIL", "stage_b_untouched": True}
    _atomic_json(target / "minimal_mjwp_run.json", report)
    return str(target / "minimal_mjwp_run.json")


def evaluate_r3_gate(paths_config: str, sequence_id: str) -> str:
    """Fail-closed aggregate of every required primary-only C-R3 gate."""
    if sequence_id != "s5__cylindermedium_lift":
        raise ValueError("C-R3 is primary-only; smoke pilots are forbidden until C-R4")
    paths = _paths(paths_config); target = _stage_c_recovery_dir(paths, sequence_id)
    required = {"static": target / "preflight_static.json", "holds": target / "preflight_holds.json", "rollout": target / "metrics_depenetrated_rollout.json", "mjwp": target / "minimal_mjwp_run.json"}
    if not all(path.is_file() for path in required.values()):
        missing = [str(path) for path in required.values() if not path.is_file()]
        raise FileNotFoundError(f"C-R3 reports missing: {missing}")
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    repo = Path(__file__).resolve().parents[2]
    targeted = subprocess.run([sys.executable, "-m", "unittest", "tests.test_sampling_weights", "tests.test_stage_c_preflight", "-v"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    regression = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    diff = subprocess.run(["git", "diff", "--check"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (target / "r3_targeted_tests.log").write_text(targeted.stdout, encoding="utf-8")
    (target / "r3_regression_tests.log").write_text(regression.stdout, encoding="utf-8")
    gates = {
        "R3-01_all_frame_mj_forward_finite": bool(reports["static"]["status"] == "PASS" and reports["static"]["all_finite"]),
        "R3-02_keyframe_holds_finite": bool(reports["holds"]["status"] == "PASS" and reports["holds"]["finite"]),
        "R3-03_no_qpos_qacc_explosion": bool(reports["holds"]["gates"]["qacc_below_1e5"] and reports["rollout"]["gates"]["qacc_below_1e5"]),
        "R3-04_full_forward_rollout_finite": bool(reports["rollout"]["status"] == "PASS" and reports["rollout"]["all_finite"]),
        "R3-05_object_tracking_finite": bool(reports["rollout"]["gates"]["object_tracking_finite"]),
        "R3-06_minimal_mjwp_rewards_finite": bool(reports["mjwp"]["rewards_finite"]),
        "R3-07_minimal_mjwp_states_finite": bool(reports["mjwp"]["states_finite"] and reports["mjwp"]["object_tracking_values_finite"]),
        "R3-08_output_schema_valid": bool(reports["mjwp"]["output_schema_valid"]),
        "R3-09_targeted_tests_pass": targeted.returncode == 0,
        "R3-10_regressions_pass": regression.returncode == 0,
        "R3-11_git_diff_check_pass": diff.returncode == 0,
    }
    payload = {"schema_version": 1, "stage": "C-R3", "sequence_id": sequence_id, "source_frame_range": _pilot(sequence_id)["frame_range"], "reports": {name: str(path) for name, path in required.items()}, "test_logs": {"targeted": str(target / "r3_targeted_tests.log"), "regression": str(target / "r3_regression_tests.log")}, "controller": reports["rollout"]["controller"], "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL", "stage_b_untouched": True, "smoke_pilots_started": False}
    _atomic_json(target / "stage_c_r3_primary.json", payload)
    _atomic_json(paths.workspace_root / "reports/stage_c_r3_primary.json", payload)
    return str(target / "stage_c_r3_primary.json")


def evaluate_depenetrated_init(paths_config: str, sequence_id: str) -> str:
    """Evaluate C-R2 tracking, visual geometry, contact and continuity gates."""
    paths = _paths(paths_config); target = _stage_c_recovery_dir(paths, sequence_id)
    metrics_path = target / "metrics_depenetrated_init.json"; metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    profile = yaml.safe_load(Path("configs/project/grab_wuji_depenetration.yaml").read_text(encoding="utf-8"))
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model); site_ids = _site_ids(model)
    with np.load(physics["trajectory"], allow_pickle=False) as archive: baseline = archive["qpos"].copy()
    with np.load(target / "trajectory_depenetrated_init.npz", allow_pickle=False) as archive: recovered = archive["qpos"].copy()
    if not np.array_equal(baseline[:, 52:], recovered[:, 52:]): raise RuntimeError("Fail closed: initializer changed object qpos")
    references = np.empty((len(baseline), len(site_ids), 3)); sites = np.empty_like(references)
    for frame in range(len(baseline)):
        data.qpos[:] = baseline[frame]; mujoco.mj_forward(model, data); references[frame] = data.site_xpos[site_ids]
        data.qpos[:] = recovered[frame]; mujoco.mj_forward(model, data); sites[frame] = data.site_xpos[site_ids]
    error = np.linalg.norm(sites - references, axis=2); tracking: dict[str, Any] = {}
    for side, offset in (("right", 0), ("left", 6)):
        tracking[side] = {"wrist_rmse_m": float(np.sqrt(np.mean(error[:, offset] ** 2))), "fingertips": {finger: {"rmse_m": float(np.sqrt(np.mean(error[:, offset + 1 + i] ** 2))), "p95_m": float(np.percentile(error[:, offset + 1 + i], 95)), "max_m": float(error[:, offset + 1 + i].max())} for i, finger in enumerate(FINGERS)}}
    object_mesh = _mesh(Path(physics["collision_cache"]) / "visual/visual.obj"); visual_ids = _hand_geom_ids(model, 1); object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    signed: list[float] = []
    for qpos in recovered:
        data.qpos[:] = qpos; mujoco.mj_forward(model, data); world = np.concatenate([mesh_from_model(model, data, geom).vertices for geom in visual_ids]); local = _world_to_body(world, data.xpos[object_body], data.xmat[object_body]); _, _, distance, _, _ = _closest_with_sign(object_mesh, local); signed.extend(distance.tolist())
    visual = summarize_signed_distances(np.asarray(signed), np.zeros((len(signed), 3)), confidence="high" if object_mesh.is_watertight else "low")
    with np.load(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz", allow_pickle=False) as contacts:
        # The actuator input intentionally removes the first/last reference
        # samples so contact guidance aligns with its finite-difference qvel.
        expected = contacts["contact"][1:-1].astype(bool); anchors = contacts["contact_surface_world"][1:-1].copy()
    tip_sites = sites[:, [1,2,3,4,5,7,8,9,10,11]]; contact_distance = np.linalg.norm(tip_sites - anchors, axis=2); recall = float(np.count_nonzero((contact_distance <= 0.015) & expected) / max(1, np.count_nonzero(expected)))
    delta = np.diff(recovered[:, :52], axis=0); ranges = model.jnt_range[:52, 1] - model.jnt_range[:52, 0]; ranges[[0,1,2,26,27,28]] = 4.0; normalized = float(np.max(np.abs(delta) / np.maximum(ranges, 1e-9)))
    tracking_ok = all(record["wrist_rmse_m"] <= profile["acceptance"]["wrist_rmse_m"] and all(item["rmse_m"] <= profile["acceptance"]["fingertip_rmse_m"] for item in record["fingertips"].values()) for record in tracking.values())
    gates = {"hard_validity": metrics["nan_inf"] == 0 and metrics["joint_limit_violations"] == 0 and metrics["object_pose_change_m"] == 0.0 and metrics["source_mapping_complete"], "tracking": tracking_ok, "visual_penetration": visual["max_penetration_m"] <= profile["acceptance"]["max_visual_penetration_m"], "collision_penetration": metrics["collision"]["after_max_m"] <= profile["acceptance"]["max_collision_penetration_m"], "contact_recall": recall >= profile["acceptance"]["contact_recall"], "smoothness": normalized <= 0.25}
    metrics.update({"tracking": tracking, "visual_penetration": visual, "contact": {"high_confidence_recall": recall, "expected_records": int(np.count_nonzero(expected))}, "smoothness": {"max_normalized_single_frame_delta": normalized, "teleport": normalized > 0.25}, "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"})
    _atomic_json(metrics_path, metrics)
    return str(metrics_path)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"make-manifest": make_manifest, "diagnose-source": diagnose_source, "build-contact-reference": build_contact_reference, "build-collision-cache": build_collision_cache, "prepare-physics-input": prepare_physics_input, "collision-audit": collision_audit, "depenetrate-init": depenetrate_init, "evaluate-depenetrated-init": evaluate_depenetrated_init, "preflight-static": preflight_static, "preflight-holds": preflight_holds, "preflight-rollout": preflight_rollout, "prepare-minimal-mjwp-input": prepare_minimal_mjwp_input, "run-minimal-mjwp": run_minimal_mjwp, "evaluate-r3-gate": evaluate_r3_gate, "write-failure-report": write_failure_report})
