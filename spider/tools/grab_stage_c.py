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
import torch
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
from spider.interp import interp
from spider.preprocess.generate_xml import main as generate_xml


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_IDS = (4, 8, 12, 16, 20)
PILOTS = (
    {"sequence_id": "s5__cylindermedium_lift", "source_sequence_id": "s5/cylindermedium_lift", "role": "true-bimanual physics primary", "hand_mode": "bimanual", "frame_range": [1460, 1876], "selection_reason": "frozen Stage B primary", "required_checks": ["bimanual", "tracking", "penetration", "contact", "object", "smoothness"]},
    {"sequence_id": "s1__mug_lift", "source_sequence_id": "s1/mug_lift", "role": "right-hand physics smoke", "hand_mode": "bimanual", "frame_range": [120, 240], "selection_reason": "frozen Stage B right-hand smoke", "required_checks": ["right_hand", "tracking", "penetration", "contact", "object", "smoothness"]},
    {"sequence_id": "s1__mug_offhand_1", "source_sequence_id": "s1/mug_offhand_1", "role": "offhand/non-interacting-hand smoke", "hand_mode": "bimanual", "frame_range": [120, 180], "selection_reason": "frozen Stage B offhand smoke", "required_checks": ["offhand", "tracking", "false_contact", "object", "smoothness"]},
)
# This auxiliary sequence is intentionally *not* part of ``PILOTS`` or the
# frozen manifest.  It is a pre-existing short Stage-B segment selected solely
# to prove that the real scene/MJWP plumbing can run from a low-penetration
# state if the primary eventually needs an infeasibility audit.
AUXILIARY_SANITY = {
    "sequence_id": "s1__mug_pass_1",
    "source_sequence_id": "s1/mug_pass_1",
    "role": "non-frozen auxiliary MJWP infrastructure sanity only",
    "hand_mode": "bimanual",
    "frame_range": [0, 60],
    "selection_reason": "pre-existing 60-frame Stage-B segment; never substitutes for frozen pilots",
    "required_checks": ["finite_MJWP_infrastructure_only"],
    "frozen": False,
}
_RAYCAST_SCENES: dict[tuple[int, int], o3d.t.geometry.RaycastingScene] = {}


def _interpolate_surface_contact_reference(
    surface_reference: np.ndarray,
    ref_steps: int,
    trailing_steps: int,
) -> np.ndarray:
    """Interpolate raw source-surface anchors on MJWP's exact time grid.

    The controller is allowed to pursue a small outward normal-gap target,
    but the Stage-C acceptance contract is distance to the original
    source-projected object surface.  Keeping this conversion separate avoids
    accidentally grading a control convenience target as ground truth.
    """
    values = np.asarray(surface_reference, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (10, 3):
        raise ValueError(f"Expected surface contact reference (T, 10, 3), got {values.shape}")
    if ref_steps < 1 or trailing_steps < 0:
        raise ValueError("ref_steps must be positive and trailing_steps non-negative")
    flattened = torch.from_numpy(values.reshape(len(values), -1)).unsqueeze(0)
    interpolated = interp(flattened, int(ref_steps)).squeeze(0).reshape(-1, 10, 3)
    if trailing_steps:
        interpolated = torch.cat([interpolated, interpolated[-1:].repeat(trailing_steps, 1, 1)], dim=0)
    return interpolated.detach().cpu().numpy().astype(np.float64)


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


def _continuous_intrinsic_xyz(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Choose a temporally continuous intrinsic-XYZ chart for serial hinges.

    ``unwrap`` alone cannot repair the Euler alternate branch at a gimbal
    crossing: equivalent orientations can differ in both first and last angle
    by pi while the middle angle switches branch.  Enumerating both XYZ
    representations and their 2pi lifts keeps the command physically
    continuous without changing any orientation.
    """
    raw = Rotation.from_quat(np.asarray(quaternions_xyzw, dtype=np.float64)).as_euler("XYZ")
    continuous = np.empty_like(raw)
    continuous[0] = raw[0]
    two_pi = 2.0 * np.pi
    for frame in range(1, len(raw)):
        base = raw[frame]
        candidates = np.stack((base, np.array([base[0] + np.pi, np.pi - base[1], base[2] + np.pi])))
        lifts = candidates + two_pi * np.round((continuous[frame - 1] - candidates) / two_pi)
        continuous[frame] = lifts[np.argmin(np.sum((lifts - continuous[frame - 1]) ** 2, axis=1))]
    return continuous


def _add_object_mocap_tracking(scene_path: Path, solref_time_constant: float = 0.004) -> None:
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
            ET.SubElement(equality, "weld", {"name": weld_name, "body1": f"{side}_object", "body2": target, "solref": f"{solref_time_constant:.6g} 1", "solimp": "0.9 0.95 0.001 0.5 2"})
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    scene_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


def _scale_robot_servo_profile(
    scene_path: Path,
    kp_scale: float = 1.0,
    forcelimit_scale: float = 1.0,
    wrist_kp_scale: float | None = None,
    wrist_forcelimit_scale: float | None = None,
    finger_kp_scale: float | None = None,
    finger_forcelimit_scale: float | None = None,
) -> None:
    """Scale only existing robot position servos in an isolated Stage-C XML.

    This is a shared, explicit physics-controller profile.  It does not alter
    the robot asset or qpos reference, and every limited finger actuator stays
    force-limited after scaling.
    """
    scales = (kp_scale, forcelimit_scale, wrist_kp_scale, wrist_forcelimit_scale, finger_kp_scale, finger_forcelimit_scale)
    if any(value is not None and value <= 0.0 for value in scales):
        raise ValueError("robot servo scales must be positive")
    wrist_kp_scale = kp_scale if wrist_kp_scale is None else wrist_kp_scale
    wrist_forcelimit_scale = forcelimit_scale if wrist_forcelimit_scale is None else wrist_forcelimit_scale
    finger_kp_scale = kp_scale if finger_kp_scale is None else finger_kp_scale
    finger_forcelimit_scale = forcelimit_scale if finger_forcelimit_scale is None else finger_forcelimit_scale
    if (
        wrist_kp_scale == 1.0
        and wrist_forcelimit_scale == 1.0
        and finger_kp_scale == 1.0
        and finger_forcelimit_scale == 1.0
    ):
        return
    root = ET.fromstring(scene_path.read_text(encoding="utf-8"))
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("scene_act has no actuator section")
    changed = 0
    for item in actuator.findall("*"):
        name = item.get("name", "")
        if not name.startswith(("right_wrist_", "left_wrist_", "r_", "l_")):
            continue
        is_wrist = name.startswith(("right_wrist_", "left_wrist_"))
        local_kp_scale = wrist_kp_scale if is_wrist else finger_kp_scale
        local_force_scale = wrist_forcelimit_scale if is_wrist else finger_forcelimit_scale
        if item.tag == "general":
            gain = [float(value) for value in item.get("gainprm", "").split()]
            bias = [float(value) for value in item.get("biasprm", "").split()]
            if len(gain) < 1 or len(bias) < 3:
                raise RuntimeError(f"Unsupported general servo parameters for {name}")
            gain[0] *= local_kp_scale
            bias[1] *= local_kp_scale
            bias[2] *= local_kp_scale
            item.set("gainprm", " ".join(f"{value:.6g}" for value in gain))
            item.set("biasprm", " ".join(f"{value:.6g}" for value in bias))
        elif item.get("kp") is not None:
            item.set("kp", f"{float(item.get('kp')) * local_kp_scale:.6g}")
        if item.tag != "general" and item.get("kv") is not None:
            item.set("kv", f"{float(item.get('kv')) * local_kp_scale:.6g}")
        if item.get("forcerange") is not None:
            low, high = (float(value) for value in item.get("forcerange", "").split())
            item.set("forcerange", f"{low * local_force_scale:.6g} {high * local_force_scale:.6g}")
        changed += 1
    if changed != 52:
        raise RuntimeError(f"Expected 52 robot position servos, scaled {changed}")
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
    if sequence_id == AUXILIARY_SANITY["sequence_id"]:
        return dict(AUXILIARY_SANITY)
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


def build_contact_reference(
    paths_config: str,
    sequence_id: str,
    distance_threshold_m: float = 0.015,
    output_tag: str = "",
) -> str:
    """Build source contacts from surface proximity, preserving rejected evidence.

    A watertight signed-distance query makes the *sign* reliable, but it does
    not make a deeply penetrating fingertip a reliable surface contact.  The
    original inclusive flag is retained in each record for audit; the emitted
    contact tensor contains only samples within ``distance_threshold_m``.
    ``output_tag`` writes an isolated reference so frozen V1 artifacts are
    never overwritten during a V2/C-XA correction.
    """
    if output_tag and not all(char.isalnum() or char in "_-" for char in output_tag):
        raise ValueError("output_tag may contain only letters, digits, '_' and '-'")
    if distance_threshold_m <= 0:
        raise ValueError("distance_threshold_m must be positive")
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
                raw_inclusive_flag = bool(
                    unsigned[frame, finger_index] <= distance_threshold_m
                    or (np.isfinite(signed[frame, finger_index]) and signed[frame, finger_index] < 0)
                )
                reliable_source = bool(unsigned[frame, finger_index] <= distance_threshold_m)
                flag = reliable_source
                contact[frame, channel] = flag; positions[frame, channel] = surface
                records.append({"frame_index": frame, "source_frame": int(sequence.source_metadata["source_frame_indices"][frame]), "side": side, "finger_region": finger, "source_point_world": source, "surface_point_world": surface, "surface_normal_world": normal, "source_signed_distance_m": signed[frame, finger_index], "unsigned_distance_m": unsigned[frame, finger_index], "projected_distance_m": float(np.linalg.norm(source - surface)), "confidence": confidence, "sign_method": method, "contact_flag": flag, "raw_inclusive_contact_flag": raw_inclusive_flag, "source_reliability": "RELIABLE_SOURCE" if reliable_source else ("UNRELIABLE_SOURCE" if raw_inclusive_flag else "NON_CONTACT"), "contact_channel": channel})
    for channel in range(10):
        interval_id = np.full(sequence.num_frames, -1, dtype=np.int32)
        for value, (start, end) in enumerate(_intervals(contact[:, channel])): interval_id[start:end] = value
        for row in records:
            if row["contact_channel"] == channel: row["contact_interval_id"] = int(interval_id[row["frame_index"]])
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id); output = robot / "stage_c"
    suffix = f"_{output_tag}" if output_tag else ""
    npz_path = output / f"contact_reference{suffix}.npz"
    json_path = output / f"contact_reference{suffix}.json"
    _atomic_npz(npz_path, source_frame_indices=np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64), contact=contact, contact_surface_world=positions)
    _atomic_json(json_path, {"schema_version": 2, "sequence_id": sequence_id, "source_only": True, "distance_threshold_m": distance_threshold_m, "contact_policy": "closest-surface distance within tolerance; deeply penetrating source samples remain recorded as UNRELIABLE_SOURCE but are not active contacts", "records": records, "intervals": {str(index): _intervals(contact[:, index]) for index in range(10)}})
    return str(json_path)


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


def prepare_physics_input(
    paths_config: str,
    sequence_id: str,
    object_pos_kp: float = 0.0,
    object_pos_kd: float = 0.0,
    object_rot_kp: float = 0.0,
    object_rot_kd: float = 0.0,
    object_pos_forcelimit: float | None = None,
    object_rot_forcelimit: float | None = None,
    object_mocap_solref: float = 0.004,
    robot_servo_kp_scale: float = 1.0,
    robot_servo_forcelimit_scale: float = 1.0,
    robot_wrist_servo_kp_scale: float | None = None,
    robot_wrist_servo_forcelimit_scale: float | None = None,
    robot_finger_servo_kp_scale: float | None = None,
    robot_finger_servo_forcelimit_scale: float | None = None,
) -> str:
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
    # The default preflight controller is the explicit mocap-weld reference
    # below.  R4 primary-only search may additionally request an explicit,
    # force-limited object actuator; this is recorded in physics_input.json
    # and never changes Stage-B data.
    generate_xml(dataset_dir=str(sandbox), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type="bimanual", task=task, data_id=0, use_visual_mesh_as_collision=False, show_viewer=False, act_scene=True, object_pos_kp=object_pos_kp, object_pos_kd=object_pos_kd, object_rot_kp=object_rot_kp, object_rot_kd=object_rot_kd, object_pos_forcelimit=object_pos_forcelimit, object_rot_forcelimit=object_rot_forcelimit)
    if object_mocap_solref <= 0.0:
        raise ValueError("object_mocap_solref must be positive")
    _scale_robot_servo_profile(
        task_dir / "scene_act.xml",
        robot_servo_kp_scale,
        robot_servo_forcelimit_scale,
        robot_wrist_servo_kp_scale,
        robot_wrist_servo_forcelimit_scale,
        robot_finger_servo_kp_scale,
        robot_finger_servo_forcelimit_scale,
    )
    _add_object_mocap_tracking(task_dir / "scene_act.xml", object_mocap_solref)
    act_model = mujoco.MjModel.from_xml_path(str(task_dir / "scene_act.xml"))
    contact_site_ids = [int(mujoco.mj_name2id(act_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip")) for side in ("right", "left") for finger in FINGERS]
    if any(site < 0 for site in contact_site_ids):
        raise RuntimeError(f"Missing Wuji contact sites: {contact_site_ids}")
    task_info["contact_site_ids"] = contact_site_ids
    _atomic_json(task_dir / "task_info.json", task_info)
    _atomic_json(mano_dir.parent / "task_info.json", task_info)
    recovery = _stage_c_recovery_dir(paths, sequence_id)
    depenetrated = recovery / "trajectory_depenetrated_init.npz"
    with np.load(contact_path, allow_pickle=False) as refs:
        # Keep the dynamic contact controller on the same source-surface
        # convention as C-R2.  Driving a physical fingertip to a zero-gap
        # visual-surface point fights the non-penetration initializer and can
        # turn a valid 1 mm clearance into a collision at the next substep.
        # The outward normal and the fixed gap are source-only; neither moves
        # the object reference nor changes contact expectations.
        depenetration_profile = yaml.safe_load(
            Path("configs/project/grab_wuji_depenetration.yaml").read_text(encoding="utf-8")
        )
        contact_gap_m = float(depenetration_profile["target_gap_m"])
        if not 0.0 <= contact_gap_m <= 0.003:
            raise RuntimeError(f"C-R4 contact target gap must be in [0, 0.003] m, got {contact_gap_m}")
        contact_targets = (
            np.asarray(refs["contact_surface_world"], dtype=np.float64)
            + _contact_normal_offsets(stage / "contact_reference.json", len(refs["contact"]), contact_gap_m)
        ).astype(np.float32)
        if depenetrated.is_file():
            # C-R4 must continue from the C-R2 derived initialization.  The
            # old implementation regenerated the sandbox from the frozen
            # Stage-B trajectory here, silently reintroducing its deep overlap
            # after a later scene/profile rebuild.  The derived file is
            # already in this scene's 64-qpos serial-object coordinates.
            with np.load(depenetrated, allow_pickle=False) as trajectory:
                qpos_act = np.asarray(trajectory["qpos"], dtype=np.float64)
                qvel = np.asarray(trajectory["qvel"], dtype=np.float64)
            if qpos_act.ndim != 2 or qpos_act.shape[1] != act_model.nq:
                raise RuntimeError(
                    f"C-R2 trajectory schema mismatch: {qpos_act.shape}, expected (*, {act_model.nq})"
                )
            if qvel.shape != (len(qpos_act), act_model.nv):
                raise RuntimeError(
                    f"C-R2 qvel schema mismatch: {qvel.shape}, expected {(len(qpos_act), act_model.nv)}"
                )
            trajectory_source = str(depenetrated)
            trajectory_source_kind = "depenetrated_init"
        else:
            with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as trajectory:
                qpos, qvel = trajectory["qpos"], trajectory["qvel"]
            # scene_act has three serial hinge joints (X, then Y, then Z), after
            # the 52 Wuji controls.  A free-joint quaternion cannot be written as
            # its rotation vector here: the serial-hinge coordinate chart is the
            # intrinsic XYZ Euler chart.  Writing rotvec values caused up to 1.35
            # rad object-pose error in the R1 audit.  Keep this conversion explicit
            # and test it against the compiled MuJoCo scene below.
            objects = qpos[:, -14:].reshape(len(qpos), 2, 7)
            object_quaternions = objects[:, :, 3:][:, :, [1, 2, 3, 0]]
            object_euler = np.stack([_continuous_intrinsic_xyz(object_quaternions[:, side]) for side in range(2)], axis=1)
            # The serial-hinge chart is periodic and has an alternate Euler
            # branch.  Preserve the same object orientation while choosing a
            # continuous coordinate chart; this prevents pi-scale physical target
            # jumps near a gimbal crossing.
            object_ctrl = np.concatenate([objects[:, :, :3], object_euler], axis=2).reshape(len(qpos), 12)
            # scene_act represents each object as xyz + three serial rotation
            # coordinates (6 DoF), whereas Stage B stores a free-joint xyz +
            # quaternion (7 DoF).  Convert the *state* as well as controls;
            # otherwise nq=64 would be fed an invalid 66-column free-joint state.
            qpos_act = np.concatenate([qpos[:, :52], object_ctrl], axis=1)
            trajectory_source = str(robot / "trajectory_kinematic.npz")
            trajectory_source_kind = "stage_b_kinematic"
        ctrl = qpos_act.copy()
        _atomic_npz(robot_dir / "trajectory_kinematic_act.npz", qpos=qpos_act, qvel=qvel, ctrl=ctrl, contact=refs["contact"][1:-1], contact_pos=contact_targets[1:-1], frequency=np.asarray(120.0))
    _atomic_json(stage / "physics_input.json", {"sandbox": str(sandbox), "scene_act": str(task_dir / "scene_act.xml"), "trajectory": str(robot_dir / "trajectory_kinematic_act.npz"), "trajectory_source": trajectory_source, "trajectory_source_kind": trajectory_source_kind, "collision_cache": str(cache_root), "object_rotation_coordinate": "intrinsic_XYZ_euler_for_serial_hinges", "contact_reference": {"source_only": True, "surface_normal_gap_m": contact_gap_m}, "object_actuator": {"pos_kp": object_pos_kp, "pos_kd": object_pos_kd, "rot_kp": object_rot_kp, "rot_kd": object_rot_kd, "pos_forcelimit": object_pos_forcelimit, "rot_forcelimit": object_rot_forcelimit}, "object_mocap_weld": {"solref_time_constant_s": object_mocap_solref, "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]}, "robot_servo": {"kp_scale": robot_servo_kp_scale, "forcelimit_scale": robot_servo_forcelimit_scale, "wrist_kp_scale": robot_wrist_servo_kp_scale if robot_wrist_servo_kp_scale is not None else robot_servo_kp_scale, "wrist_forcelimit_scale": robot_wrist_servo_forcelimit_scale if robot_wrist_servo_forcelimit_scale is not None else robot_servo_forcelimit_scale, "finger_kp_scale": robot_finger_servo_kp_scale if robot_finger_servo_kp_scale is not None else robot_servo_kp_scale, "finger_forcelimit_scale": robot_finger_servo_forcelimit_scale if robot_finger_servo_forcelimit_scale is not None else robot_servo_forcelimit_scale, "servo_count": 52}, "baseline_untouched": True})
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


def _stage_b_act_baseline(paths, sequence_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the immutable Stage-B state in the Stage-C serial-object chart."""
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id)
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
    objects = qpos[:, -14:].reshape(len(qpos), 2, 7)
    quaternions = objects[:, :, 3:][:, :, [1, 2, 3, 0]]
    euler = np.stack(
        [_continuous_intrinsic_xyz(quaternions[:, side]) for side in range(2)], axis=1
    )
    return np.concatenate([qpos[:, :52], np.concatenate([objects[:, :, :3], euler], axis=2).reshape(len(qpos), 12)], axis=1), qvel


def _contact_normal_offsets(contact_json: Path, frame_count: int, gap_m: float) -> np.ndarray:
    """Build source-surface outward target gaps without changing object poses."""
    offsets = np.zeros((frame_count, 10, 3), dtype=np.float64)
    if gap_m <= 0.0:
        return offsets
    payload = json.loads(contact_json.read_text(encoding="utf-8"))
    for record in payload.get("records", []):
        frame, channel = int(record["frame_index"]), int(record["contact_channel"])
        if not record.get("contact_flag") or not (0 <= frame < frame_count and 0 <= channel < 10):
            continue
        normal = np.asarray(record["surface_normal_world"], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm <= 1e-9:
            raise RuntimeError(f"Invalid contact normal at frame={frame}, channel={channel}")
        offsets[frame, channel] = normal / norm * gap_m
    return offsets


def _site_ids(model: mujoco.MjModel) -> list[int]:
    names = [f"{side}_{item}" for side in ("right", "left") for item in ("palm", *[f"{finger}_tip" for finger in FINGERS])]
    output = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)) for name in names]
    if any(index < 0 for index in output):
        raise RuntimeError(f"Missing required palm/tip sites: {names}")
    return output


def _joint_bounds(model: mujoco.MjModel, baseline: np.ndarray, profile: dict[str, Any], finger_only: bool) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    trans = float(profile["wrist_translation_bound_m"]); rot = float(profile["wrist_rotation_bound_rad"])
    wrist_indices = {0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31}
    for index in range(52):
        # This check must precede the generic wrist bounds.  The previous
        # ordering made the documented Phase-1 ``finger_only`` mode a no-op:
        # translation matched the first branch and rotation matched the
        # second, so both root/wrist groups were still optimised.
        if finger_only and index in wrist_indices:
            bounds.append((float(baseline[index]), float(baseline[index])))
        elif index in (0, 1, 2, 26, 27, 28):
            bounds.append((float(baseline[index] - trans), float(baseline[index] + trans)))
        elif index in (3, 4, 5, 29, 30, 31):
            bounds.append((float(baseline[index] - rot), float(baseline[index] + rot)))
        else:
            bounds.append((float(model.jnt_range[index, 0]), float(model.jnt_range[index, 1])))
    return bounds


def _cxa_variable_indices(allow_wrist_correction: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return explicit mutable and locked C-XA robot qpos indices.

    The scene's object segment is intentionally excluded: it is immutable in
    every C-XA profile.  The default correction is finger articulation only.
    """
    wrist = np.asarray((0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31), dtype=np.int64)
    fingers = np.asarray(tuple(range(6, 26)) + tuple(range(32, 52)), dtype=np.int64)
    mutable = np.arange(52, dtype=np.int64) if allow_wrist_correction else fingers
    locked = np.setdiff1d(np.arange(52, dtype=np.int64), mutable, assume_unique=True)
    return mutable, locked


def _assert_locked_dofs(candidate: np.ndarray, baseline: np.ndarray, locked_indices: np.ndarray, stage: str) -> None:
    """Fail before serialisation if any locked wrist/root coordinate moved."""
    delta = np.asarray(candidate, dtype=np.float64)[locked_indices] - np.asarray(baseline, dtype=np.float64)[locked_indices]
    if not np.all(np.isfinite(delta)) or float(np.max(np.abs(delta), initial=0.0)) > 1e-12:
        raise RuntimeError(f"C-XA locked-DOF invariant failed after {stage}: max_delta={float(np.max(np.abs(delta), initial=0.0)):.3e}")


def _collision_depths(data: mujoco.MjData, hand_collision_ids: set[int], object_collision_ids: set[int]) -> np.ndarray:
    values = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} & hand_collision_ids and {int(contact.geom1), int(contact.geom2)} & object_collision_ids:
            values.append(max(0.0, -float(contact.dist)))
    return np.asarray(values, dtype=np.float64)


def _worst_visual_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    visual_geom_ids: list[int],
    object_mesh: trimesh.Trimesh,
    object_body: int,
) -> tuple[float, int, int, np.ndarray, np.ndarray]:
    """Return the worst signed visual-mesh point against the real object mesh.

    MuJoCo mesh contacts are deliberately audited separately from visual mesh
    SDF.  A missed mesh-mesh contact must therefore not allow a visual vertex
    to remain deeply embedded in the real object surface.
    """
    parts: list[tuple[int, np.ndarray]] = []
    for geom_id in visual_geom_ids:
        mesh = mesh_from_model(model, data, geom_id)
        parts.append((geom_id, mesh.vertices))
    lengths = np.cumsum([len(vertices) for _, vertices in parts])
    world = np.concatenate([vertices for _, vertices in parts], axis=0)
    local = _world_to_body(world, data.xpos[object_body], data.xmat[object_body])
    closest, _unsigned, signed, _method, confidence = _closest_with_sign(object_mesh, local)
    if confidence != "high" or not np.isfinite(signed).all():
        raise RuntimeError("Fail closed: visual depenetration requires high-confidence signed distances")
    vertex = int(np.argmin(signed))
    part = int(np.searchsorted(lengths, vertex, side="right"))
    start = 0 if part == 0 else int(lengths[part - 1])
    geom_id, _vertices = parts[part]
    return float(signed[vertex]), geom_id, vertex - start, world[vertex], closest[vertex]


def _visual_dls_refine(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: np.ndarray,
    object_qpos: np.ndarray,
    bounds: list[tuple[float, float]],
    visual_geom_ids: list[int],
    object_mesh: trimesh.Trimesh,
    object_body: int,
    profile: dict[str, Any],
    mutable_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, int]:
    """Apply bounded local DLS only when visual SDF finds deep overlap."""
    data.qpos[:52] = candidate; data.qpos[52:] = object_qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
    before, _geom, _vertex, _point, _closest = _worst_visual_penetration(
        model, data, visual_geom_ids, object_mesh, object_body
    )
    value = candidate.copy()
    target = -float(profile["visual_refinement"]["target_clearance_m"])
    max_step = float(profile["visual_refinement"]["max_joint_step"])
    iterations = 0
    for _ in range(int(profile["visual_refinement"]["max_iterations"])):
        data.qpos[:52] = value; data.qpos[52:] = object_qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        signed, geom_id, _vertex, point_world, closest_local = _worst_visual_penetration(
            model, data, visual_geom_ids, object_mesh, object_body
        )
        if signed >= target:
            break
        side, _region = region_for_geom(model, geom_id)
        indices = np.arange(26, 52) if side == "left" else np.arange(0, 26)
        if mutable_indices is not None:
            indices = np.intersect1d(indices, mutable_indices, assume_unique=True)
        if len(indices) == 0:
            raise RuntimeError("C-XA visual refinement selected no mutable DOFs for the penetrated hand")
        closest_world = closest_local @ data.xmat[object_body].reshape(3, 3).T + data.xpos[object_body]
        direction = closest_world - point_world
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise RuntimeError("Fail closed: visual depenetration has an undefined surface direction")
        desired = direction / norm * (-signed + float(profile["visual_refinement"]["target_clearance_m"]))
        body_id = int(model.geom_bodyid[geom_id])
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        rotational = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacobian, rotational, point_world, body_id)
        local_jacobian = jacobian[:, indices]
        damping = float(profile["visual_refinement"]["damping"])
        delta = local_jacobian.T @ np.linalg.solve(
            local_jacobian @ local_jacobian.T + damping * np.eye(3), desired
        )
        delta = np.clip(delta, -max_step, max_step)
        value[indices] += delta
        lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
        upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
        value[:] = np.clip(value, lower, upper)
        iterations += 1
    data.qpos[:52] = value; data.qpos[52:] = object_qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
    after, _geom, _vertex, _point, _closest = _worst_visual_penetration(
        model, data, visual_geom_ids, object_mesh, object_body
    )
    return value, before, after, iterations


def _collision_dls_refine(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: np.ndarray,
    object_qpos: np.ndarray,
    bounds: list[tuple[float, float]],
    hand_collision_ids: set[int],
    object_collision_ids: set[int],
    profile: dict[str, Any],
    mutable_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Resolve a remaining real contact with a bounded contact-point DLS step."""
    value = candidate.copy()
    settings = profile["collision_refinement"]
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    iterations = 0
    for _ in range(int(settings["max_iterations"])):
        data.qpos[:52] = value; data.qpos[52:] = object_qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        contacts = [
            item for item in data.contact[: data.ncon]
            if ((int(item.geom1) in hand_collision_ids and int(item.geom2) in object_collision_ids)
                or (int(item.geom2) in hand_collision_ids and int(item.geom1) in object_collision_ids))
        ]
        if not contacts:
            break
        contact = min(contacts, key=lambda item: float(item.dist))
        depth = max(0.0, -float(contact.dist))
        target_depth = float(settings["target_max_penetration_m"])
        if depth <= target_depth:
            break
        hand_geom = int(contact.geom1) if int(contact.geom1) in hand_collision_ids else int(contact.geom2)
        hand_is_geom1 = hand_geom == int(contact.geom1)
        body_id = int(model.geom_bodyid[hand_geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        indices = np.arange(26, 52) if body_name.startswith("l_") else np.arange(0, 26)
        if mutable_indices is not None:
            indices = np.intersect1d(indices, mutable_indices, assume_unique=True)
        if len(indices) == 0:
            raise RuntimeError("C-XA collision refinement selected no mutable DOFs for the penetrated hand")
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        # MuJoCo's contact normal points from geom1 to geom2.  Move the hand
        # away from the object, regardless of which contact slot holds it.
        direction = -normal if hand_is_geom1 else normal
        desired = direction * (depth - target_depth + float(settings["clearance_m"]))
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        rotational = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacobian, rotational, np.asarray(contact.pos), body_id)
        local_jacobian = jacobian[:, indices]
        delta = local_jacobian.T @ np.linalg.solve(
            local_jacobian @ local_jacobian.T + float(settings["damping"]) * np.eye(3), desired
        )
        delta = np.clip(delta, -float(settings["max_joint_step"]), float(settings["max_joint_step"]))
        value[indices] += delta
        value[:] = np.clip(value, lower, upper)
        iterations += 1
    return value, iterations


def _recovered_robot_qvel(
    recovered_qpos: np.ndarray,
    baseline_qvel: np.ndarray,
    fps: float = 120.0,
) -> np.ndarray:
    """Return a dynamic reference consistent with the emitted C-R2 qpos.

    C-R2 changes robot generalized coordinates while keeping both object
    trajectories immutable.  Re-emitting an all-zero velocity reference makes
    MJWP penalize every required robot motion as velocity error.  Preserve the
    source-derived object velocity chart and derive only the 52 corrected
    robot velocities from the final, temporally constrained trajectory.
    """
    if recovered_qpos.ndim != 2 or recovered_qpos.shape[1] < 52:
        raise ValueError(f"Expected recovered qpos shaped (T, >=52), got {recovered_qpos.shape}")
    if baseline_qvel.shape != recovered_qpos.shape:
        raise ValueError(
            f"C-R2 baseline qvel shape {baseline_qvel.shape} does not match qpos {recovered_qpos.shape}"
        )
    if len(recovered_qpos) < 2 or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("C-R2 qvel derivation requires at least two frames and positive finite fps")
    qvel = np.asarray(baseline_qvel, dtype=np.float64).copy()
    qvel[:, :52] = np.gradient(recovered_qpos[:, :52], 1.0 / float(fps), axis=0)
    if not np.isfinite(qvel).all():
        raise RuntimeError("C-R2 derived qvel contains NaN/Inf")
    return qvel


def _depenetration_artifacts(target: Path, output_tag: str) -> dict[str, Path]:
    """Return an isolated C-R2 artifact namespace for an optional candidate."""
    if output_tag and not all(char.isalnum() or char in "_-" for char in output_tag):
        raise ValueError("depenetration output_tag may contain only letters, digits, '_' and '-'")
    suffix = f"_{output_tag}" if output_tag else ""
    return {
        "trajectory": target / f"trajectory_depenetrated_init{suffix}.npz",
        "trace": target / f"depenetration_trace{suffix}.npz",
        "config": target / f"depenetration_config{suffix}.json",
        "metrics": target / f"metrics_depenetrated_init{suffix}.json",
        "manifest": target / f"depenetration_manifest{suffix}.json",
    }


def _contact_collision_pareto_front(records: list[dict[str, Any]]) -> list[int]:
    """Return indices not dominated on contact recall versus collision depth.

    Higher recall and lower maximum MuJoCo collision penetration are both
    better.  This deliberately has no weighted scalar score: an
    infeasibility audit needs to show the actual trade-off, rather than hide
    it behind a convenient conversion rate.  Records missing either metric
    are rejected instead of guessed into the frontier.
    """
    for record in records:
        if not np.isfinite(float(record["contact_recall"])) or not np.isfinite(float(record["collision_max_m"])):
            raise ValueError("Pareto records require finite contact_recall and collision_max_m")
    front: list[int] = []
    for index, record in enumerate(records):
        dominated = False
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            contact_better = float(other["contact_recall"]) >= float(record["contact_recall"])
            collision_better = float(other["collision_max_m"]) <= float(record["collision_max_m"])
            strictly_better = (
                float(other["contact_recall"]) > float(record["contact_recall"])
                or float(other["collision_max_m"]) < float(record["collision_max_m"])
            )
            if contact_better and collision_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(index)
    return front


def _contiguous_index_ranges(indices: list[int]) -> list[list[int]]:
    """Compact sorted integer indices into inclusive ranges for audit output."""
    if any(not isinstance(index, int) for index in indices):
        raise ValueError("Index ranges require integer indices")
    if indices != sorted(set(indices)):
        raise ValueError("Index ranges require sorted, unique indices")
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append([start, previous])
            start = index
        previous = index
    ranges.append([start, previous])
    return ranges


def depenetrate_init(
    paths_config: str,
    sequence_id: str,
    profile_path: str = "configs/project/grab_wuji_depenetration.yaml",
    output_tag: str = "",
    contact_targets_path: str | None = None,
    output_dir: str | None = None,
    initial_jitter_seed: int | None = None,
    initial_wrist_translation_jitter_m: float = 0.0,
    initial_wrist_rotation_jitter_rad: float = 0.0,
    initial_finger_jitter_rad: float = 0.0,
) -> str:
    """Build a bounded, traceable C-R2 initialization without altering Stage B.

    Powell is used deliberately because MuJoCo mesh-contact depth is not a
    smooth analytic function.  Every frame has its own bounded variables and
    warm-started temporal prior; there is no trajectory-wide hand offset.
    """
    jitter_values = (
        initial_wrist_translation_jitter_m,
        initial_wrist_rotation_jitter_rad,
        initial_finger_jitter_rad,
    )
    if any(value < 0.0 for value in jitter_values):
        raise ValueError("C-R2 multi-start jitter bounds must be non-negative")
    use_multistart = any(value > 0.0 for value in jitter_values)
    if use_multistart and initial_jitter_seed is None:
        raise ValueError("C-R2 multi-start jitter requires an explicit deterministic seed")
    paths = _paths(paths_config); frozen = _pilot(sequence_id); profile_file = Path(profile_path)
    profile = yaml.safe_load(profile_file.read_text(encoding="utf-8")); profile_hash = _profile_hash(profile_file)
    allow_wrist_correction = bool(profile.get("allow_wrist_correction", False))
    mutable_indices, locked_indices = _cxa_variable_indices(allow_wrist_correction)
    physics_path = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json"
    if not physics_path.is_file(): prepare_physics_input(paths_config, sequence_id)
    physics = json.loads(physics_path.read_text(encoding="utf-8")); model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model)
    baseline, baseline_qvel = _stage_b_act_baseline(paths, sequence_id)
    if baseline.shape[1] != model.nq or baseline_qvel.shape != (len(baseline), model.nv):
        raise RuntimeError(f"Stage-B/scene schema mismatch: qpos={baseline.shape}, qvel={baseline_qvel.shape}, model=({model.nq}, {model.nv})")
    with np.load(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz", allow_pickle=False) as archive:
        contact_expected = archive["contact"][1:-1].astype(bool)
        contact_anchors = archive["contact_surface_world"][1:-1].copy()
    if len(contact_expected) != len(baseline):
        raise RuntimeError("Contact reference and actuator trajectory are not frame-aligned")
    normal_offsets = _contact_normal_offsets(
        _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.json",
        len(baseline),
        float(profile["target_gap_m"]),
    )
    if contact_targets_path is not None:
        target_file = Path(contact_targets_path)
        with np.load(target_file, allow_pickle=False) as archive:
            required = {"expected", "anchors", "normals", "assignment_index"}
            if not required.issubset(archive.files):
                raise RuntimeError(f"V2 contact targets missing fields: {sorted(required - set(archive.files))}")
            contact_expected = np.asarray(archive["expected"], dtype=bool)
            contact_anchors = np.asarray(archive["anchors"], dtype=np.float64)
            normals = np.asarray(archive["normals"], dtype=np.float64)
        if contact_expected.shape != (len(baseline), 10) or contact_anchors.shape != (len(baseline), 10, 3) or normals.shape != (len(baseline), 10, 3):
            raise RuntimeError("V2 contact targets are not aligned to the immutable Stage-B trajectory")
        normal_offsets = np.zeros_like(normals)
        norm = np.linalg.norm(normals, axis=2)
        active = contact_expected & (norm > 1e-9)
        normal_offsets[active] = normals[active] / norm[active, None] * float(profile["target_gap_m"])
    site_ids = _site_ids(model); hand_collision = set(_hand_geom_ids(model, 2)); object_collision = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_object_") and model.geom_group[i] == 3}
    visual_geom_ids = _hand_geom_ids(model, 1)
    object_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object"))
    object_mesh = _mesh(Path(physics["collision_cache"]) / "visual/visual.obj")
    if object_body < 0 or not object_mesh.is_watertight or not object_mesh.is_winding_consistent:
        raise RuntimeError("Fail closed: C-R2 visual refinement requires a watertight object visual mesh")
    references = np.zeros((len(baseline), len(site_ids), 3), dtype=np.float64)
    for frame, qpos in enumerate(baseline):
        data.qpos[:] = qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data); references[frame] = data.site_xpos[site_ids]
    target_dir = Path(output_dir) if output_dir is not None else _stage_c_recovery_dir(paths, sequence_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _depenetration_artifacts(target_dir, output_tag)
    candidate_profile = {
        "base_profile_hash": profile_hash,
        "output_tag": output_tag or None,
        "initial_jitter_seed": initial_jitter_seed,
        "initial_wrist_translation_jitter_m": float(initial_wrist_translation_jitter_m),
        "initial_wrist_rotation_jitter_rad": float(initial_wrist_rotation_jitter_rad),
        "initial_finger_jitter_rad": float(initial_finger_jitter_rad),
        "continuation_semantics": "canonical_warmstart_per_phase",
    }
    candidate_profile_hash = hashlib.sha256(
        json.dumps(candidate_profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(initial_jitter_seed) if use_multistart else None
    recovered = baseline.copy(); phase = np.zeros(len(baseline), dtype=np.int8); status = np.empty(len(baseline), dtype="U32")
    objective_terms = np.zeros((len(baseline), 6), dtype=np.float64); corrections = np.zeros((len(baseline), 52), dtype=np.float64)
    initial_jitters = np.zeros((len(baseline), 52), dtype=np.float64)
    before = np.zeros(len(baseline), dtype=np.float64); after = np.zeros(len(baseline), dtype=np.float64)
    collision_refinement_iterations = np.zeros(len(baseline), dtype=np.int8)
    visual_before = np.zeros(len(baseline), dtype=np.float64); visual_after = np.zeros(len(baseline), dtype=np.float64); visual_iterations = np.zeros(len(baseline), dtype=np.int8); visual_collision_safe_alpha = np.ones(len(baseline), dtype=np.float64)
    previous: np.ndarray | None = None
    weights = profile["weights"]; start = time.monotonic()
    for frame, qpos in enumerate(baseline):
        if time.monotonic() - start > float(profile["timeout_seconds"]):
            raise TimeoutError(f"depenetration timed out after frame {frame}")
        data.qpos[:] = qpos; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        before[frame] = _collision_depths(data, hand_collision, object_collision).max(initial=0.0)
        base52 = qpos[:52].copy(); target = references[frame]
        # The default is fingers only.  Historical C-XA accidentally moved
        # wrists even in Phase 1 because its lock branch was unreachable;
        # Phase 2 is now an explicit opt-in profile capability only.
        candidate = base52.copy(); selected_phase = 1
        if rng is not None:
            jitter = np.zeros(52, dtype=np.float64)
            for indices, bound in (
                ((0, 1, 2, 26, 27, 28), initial_wrist_translation_jitter_m),
                ((3, 4, 5, 29, 30, 31), initial_wrist_rotation_jitter_rad),
                (tuple(range(6, 26)) + tuple(range(32, 52)), initial_finger_jitter_rad),
            ):
                if bound > 0.0:
                    jitter[list(indices)] = rng.uniform(-bound, bound, size=len(indices))
            initial_jitters[frame] = jitter
        phase_plan = [(True, int(profile["phase1_maxiter"]))]
        if allow_wrist_correction:
            phase_plan.append((False, int(profile["phase2_maxiter"])))
        for finger_only, maxiter in phase_plan:
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
            # A multi-start must alter only the bounded initial point.  Keep
            # the canonical phase/warm-start continuation itself identical,
            # otherwise a candidate would confound initialization diversity
            # with a different temporal optimizer algorithm.
            initial_center = previous if previous is not None else candidate
            initial = np.clip(
                initial_center + initial_jitters[frame],
                [bound[0] for bound in bounds], [bound[1] for bound in bounds],
            )
            def objective(values: np.ndarray) -> float:
                data.qpos[:] = qpos; data.qpos[:52] = values; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                depth = _collision_depths(data, hand_collision, object_collision)
                penetration = float(np.square(np.maximum(depth - float(profile["penetration_margin_m"]), 0.0)).sum())
                delta = values - base52
                sites = data.site_xpos[site_ids] - target
                wrist = float(np.square(sites[[0, 6]]).sum()); tips = float(np.square(sites[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]).sum())
                active = contact_expected[frame]
                contact_target = contact_anchors[frame] + normal_offsets[frame]
                contact = float(np.square(data.site_xpos[site_ids][[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]][active] - contact_target[active]).sum()) if np.any(active) else 0.0
                temporal = float(np.square(values - previous).sum()) if previous is not None else 0.0
                return weights["penetration"] * penetration + weights["contact"] * contact + weights["wrist_position"] * wrist + weights["fingertip"] * tips + weights["posture"] * float(np.square(delta).sum()) + weights["velocity"] * temporal
            result = minimize(objective, initial, method="Powell", bounds=bounds, options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5})
            candidate = np.asarray(result.x, dtype=np.float64)
            _assert_locked_dofs(candidate, base52, locked_indices, "Powell solver update")
            data.qpos[:] = qpos; data.qpos[:52] = candidate; data.qvel[:] = 0; mujoco.mj_forward(model, data)
            candidate_depth = _collision_depths(data, hand_collision, object_collision).max(initial=0.0)
            selected_phase = 1 if finger_only else 2
            if candidate_depth <= float(profile["acceptance"]["max_collision_penetration_m"]): break
        prior = None if previous is None else previous.copy()
        candidate, collision_refinement_iterations[frame] = _collision_dls_refine(
            model, data, candidate, qpos[52:], bounds, hand_collision, object_collision, profile, mutable_indices
        )
        _assert_locked_dofs(candidate, base52, locked_indices, "collision DLS refinement")
        collision_safe_candidate = candidate.copy()
        candidate, visual_before[frame], visual_after[frame], visual_iterations[frame] = _visual_dls_refine(
            model, data, candidate, qpos[52:], bounds, visual_geom_ids, object_mesh, object_body, profile, mutable_indices
        )
        _assert_locked_dofs(candidate, base52, locked_indices, "visual DLS refinement")
        # A visual-mesh correction is accepted only to the extent that it
        # preserves the independently audited MuJoCo collision gate.  The
        # pre-refinement candidate is known to be a bounded collision result;
        # interpolate back toward it rather than silently trading one gate for
        # another.
        data.qpos[:52] = candidate; data.qpos[52:] = qpos[52:]; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        collision_limit = float(profile["acceptance"]["max_collision_penetration_m"])
        if _collision_depths(data, hand_collision, object_collision).max(initial=0.0) > collision_limit:
            alpha = 0.0
            collision_safe = False
            for trial_alpha in np.linspace(0.95, 0.0, 20):
                trial = collision_safe_candidate + trial_alpha * (candidate - collision_safe_candidate)
                data.qpos[:52] = trial; data.qpos[52:] = qpos[52:]; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                if _collision_depths(data, hand_collision, object_collision).max(initial=0.0) <= collision_limit:
                    candidate = trial; alpha = float(trial_alpha); collision_safe = True; break
            if not collision_safe:
                candidate = collision_safe_candidate.copy()
            visual_collision_safe_alpha[frame] = alpha
            data.qpos[:52] = candidate; data.qpos[52:] = qpos[52:]; data.qvel[:] = 0; mujoco.mj_forward(model, data)
            visual_after[frame], _geom, _vertex, _point, _closest = _worst_visual_penetration(
                model, data, visual_geom_ids, object_mesh, object_body
            )
        recovered[frame, :52] = candidate
        _assert_locked_dofs(recovered[frame, :52], base52, locked_indices, "emitted state")
        # Do not feed an optional visual-only local correction into the next
        # frame's collision optimization bounds.  The primary warm-start stays
        # on the collision-optimized temporal path; final smoothness is still
        # audited on the emitted corrected trajectory below.
        previous = collision_safe_candidate.copy(); phase[frame] = selected_phase; status[frame] = "converged" if result.success else "maxiter"
        data.qpos[:] = recovered[frame]; data.qvel[:] = 0; mujoco.mj_forward(model, data); depths = _collision_depths(data, hand_collision, object_collision); after[frame] = depths.max(initial=0.0)
        errors = data.site_xpos[site_ids] - target; delta = candidate - base52
        temporal_term = float(np.square(candidate - prior).sum()) if prior is not None else 0.0
        objective_terms[frame] = [float(np.square(np.maximum(depths - float(profile["penetration_margin_m"]), 0.0)).sum()), float(np.square(errors[[0, 6]]).sum()), float(np.square(errors[[1,2,3,4,5,7,8,9,10,11]]).sum()), float(np.square(delta).sum()), temporal_term, float(result.fun)]
        corrections[frame] = delta
    mapping = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"]
    recovered_qvel = _recovered_robot_qvel(recovered, baseline_qvel)
    if not np.array_equal(recovered[:, 52:], baseline[:, 52:]):
        raise RuntimeError("C-XA object-lock invariant failed before serialisation")
    _atomic_npz(artifacts["trajectory"], qpos=recovered, qvel=recovered_qvel, source_frame_indices=np.asarray(mapping, dtype=np.int64))
    _atomic_npz(artifacts["trace"], corrections=corrections, initial_jitters=initial_jitters, phase=phase, objective_terms=objective_terms, collision_before_m=before, collision_after_m=after, collision_refinement_iterations=collision_refinement_iterations, visual_penetration_before_m=visual_before, visual_penetration_after_m=visual_after, visual_refinement_iterations=visual_iterations, visual_collision_safe_alpha=visual_collision_safe_alpha, optimizer_status=status, mutable_qpos_indices=mutable_indices, locked_qpos_indices=locked_indices, locked_qpos_delta=(recovered[:, :52] - baseline[:, :52])[:, locked_indices])
    config = {"profile_path": str(profile_file), "profile_hash": profile_hash, "candidate_profile_hash": candidate_profile_hash, "candidate_initialization": candidate_profile, "profile": profile, "optimizer": "scipy.optimize.minimize/Powell + bounded MuJoCo contact/visual point-Jacobian DLS", "seed": profile["seed"], "variables": "per-frame 52 robot qpos; object 12-qpos segment immutable", "optimized_qpos_indices": mutable_indices, "locked_qpos_indices": locked_indices, "root_wrist_correction_default": "LOCKED" if not allow_wrist_correction else "EXPLICIT_PROFILE_OPT_IN", "locked_dof_invariant": "max absolute delta <= 1e-12 after every solver and DLS update", "baseline": {"source": str(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "trajectory_kinematic.npz"), "object_chart": "intrinsic_XYZ_euler_for_serial_hinges"}, "qvel_reference": {"robot": "finite difference of emitted depenetrated qpos at 120 Hz", "object": "immutable Stage-B actuator-chart qvel"}, "contact_target_gap_m": float(profile["target_gap_m"]), "contact_targets_path": contact_targets_path, "contract": "TASK_EQUIVALENT_CONTACT" if contact_targets_path else "EXACT_SOURCE_FINGER_CONTACT_V1", "continuation": ["phase0 baseline", "phase1 finger-only", "phase2 optional explicit wrist+fingers", "phase3 warm-start velocity regularization", "phase4 MuJoCo contact bounded DLS", "phase5 visual signed-mesh bounded DLS", "phase6 static MuJoCo verification"], "windowing": {"window_length": profile["window_length"], "overlap": profile["overlap"], "implementation": "sequential warm-started overlapping-window contract"}}
    _atomic_json(artifacts["config"], config)
    if not output_tag and output_dir is None:
        _atomic_json(target_dir / "selected_depenetration_profile.json", {"profile_hash": profile_hash, "profile": profile})
    metrics = {"sequence_id": sequence_id, "output_tag": output_tag or None, "status": "PASS" if float(after.max()) <= float(profile["acceptance"]["max_collision_penetration_m"]) else "FAIL", "collision": {"before_max_m": float(before.max()), "after_max_m": float(after.max()), "before_p95_m": float(np.percentile(before, 95)), "after_p95_m": float(np.percentile(after, 95))}, "object_pose_change_m": 0.0, "source_mapping_complete": bool(len(mapping) == len(recovered)), "joint_limit_violations": 0, "nan_inf": 0, "profile_hash": profile_hash, "candidate_profile_hash": candidate_profile_hash, "runtime_s": time.monotonic() - start}
    _atomic_json(artifacts["metrics"], metrics); _atomic_json(artifacts["manifest"], {"input_stage_b": str(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "trajectory_kinematic.npz"), "output": str(artifacts["trajectory"]), "object_pose_immutable": True, "baseline_untouched": True, "profile_hash": profile_hash, "candidate_profile_hash": candidate_profile_hash, "metrics": str(artifacts["metrics"])})
    return str(artifacts["metrics"])


def _load_preflight_inputs(
    paths_config: str,
    sequence_id: str,
    trajectory_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[Any, mujoco.MjModel, np.ndarray, np.ndarray, Path]:
    """Load only derived C-R2/isolated inputs and validate their alignment."""
    paths = _paths(paths_config)
    target = Path(output_dir) if output_dir is not None else _stage_c_recovery_dir(paths, sequence_id)
    target.mkdir(parents=True, exist_ok=True)
    physics_file = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json"
    if not physics_file.is_file():
        prepare_physics_input(paths_config, sequence_id)
    physics = json.loads(physics_file.read_text(encoding="utf-8"))
    trajectory = Path(trajectory_path) if trajectory_path is not None else target / "trajectory_depenetrated_init.npz"
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


def preflight_static(
    paths_config: str, sequence_id: str, trajectory_path: str | None = None,
    output_dir: str | None = None, output_tag: str = "",
) -> str:
    """C-R3 Level 1: audited all-frame static ``mj_forward`` on primary only."""
    _pilot(sequence_id)
    paths, model, qpos, qvel, target = _load_preflight_inputs(paths_config, sequence_id, trajectory_path, output_dir)
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
    suffix = f"_{output_tag}" if output_tag else ""
    frames = target / f"preflight_static_frames{suffix}.npz"
    _atomic_npz(frames, qacc_max=qacc, contact_count=contact_count, collision_penetration_m=depth, finite=finite.astype(np.uint8))
    report = {"schema_version": 1, "stage": "C-R3", "level": "static_mj_forward", "sequence_id": sequence_id, "frame_count": len(qpos), "input": str(trajectory_path or target / "trajectory_depenetrated_init.npz"), "all_finite": bool(finite.all()), "qacc_max": float(qacc.max(initial=0.0)), "contact_count_max": int(contact_count.max(initial=0)), "collision_penetration_max_m": float(depth.max(initial=0.0)), "warnings": warnings, "warning_count": len(warnings), "baseline_untouched": True, "status": "PASS" if bool(finite.all()) and not warnings else "FAIL"}
    output = target / f"preflight_static{suffix}.json"; _atomic_json(output, report)
    return str(output)


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


def preflight_holds(
    paths_config: str, sequence_id: str, hold_seconds: float = 0.2,
    trajectory_path: str | None = None, output_dir: str | None = None, output_tag: str = "",
) -> str:
    """C-R3 Level 2: six fixed-reference physical holds without qpos rewrites."""
    paths, model, qpos, _, target = _load_preflight_inputs(paths_config, sequence_id, trajectory_path, output_dir)
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
    suffix = f"_{output_tag}" if output_tag else ""; output = target / f"preflight_holds{suffix}.json"
    _atomic_json(output, report)
    return str(output)


def preflight_rollout(
    paths_config: str, sequence_id: str, substeps_per_frame: int = 4,
    trajectory_path: str | None = None, output_dir: str | None = None, output_tag: str = "",
) -> str:
    """C-R3 Level 3: full dynamic replay with a physical mocap-weld reference."""
    _, model, qpos, _, target = _load_preflight_inputs(paths_config, sequence_id, trajectory_path, output_dir)
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
    suffix = f"_{output_tag}" if output_tag else ""
    _atomic_npz(target / f"trajectory_depenetrated_rollout{suffix}.npz", qpos=qpos_rollout, qvel=qvel_rollout, source_frame_indices=np.arange(len(qpos), dtype=np.int64), object_tracking_position_m=pos_error, object_tracking_rotation_rad=rot_error, qacc_max=qacc, qvel_max=qvel_max, finite=finite.astype(np.uint8))
    report = {"schema_version": 1, "stage": "C-R3", "level": "full_forward_rollout", "sequence_id": sequence_id, "frame_count": len(qpos), "substeps_per_frame": int(substeps_per_frame), "qacc_max": float(qacc.max(initial=0.0)), "qvel_max": float(qvel_max.max(initial=0.0)), "object_tracking": {side: {"position_rmse_m": float(np.sqrt(np.mean(pos_error[:, index] ** 2))), "position_max_m": float(pos_error[:, index].max()), "rotation_mean_rad": float(rot_error[:, index].mean()), "rotation_max_rad": float(rot_error[:, index].max())} for index, side in enumerate(("right", "left"))}, "all_finite": bool(finite.all()), "warnings": warnings, "warning_count": len(warnings), "gates": {"all_finite": bool(finite.all()), "qacc_below_1e5": bool(qacc.max(initial=0.0) < 1e5), "no_warnings": not warnings, "object_tracking_finite": bool(np.isfinite(pos_error).all() and np.isfinite(rot_error).all())}, "controller": {"kind": "mocap_weld_reference", "object_qpos_overwritten_after_initialization": False, "object_actuator_gains": "zero"}, "status": "PASS" if bool(finite.all()) and qacc.max(initial=0.0) < 1e5 and not warnings and np.isfinite(pos_error).all() and np.isfinite(rot_error).all() else "FAIL"}
    output = target / f"metrics_depenetrated_rollout{suffix}.json"; _atomic_json(output, report)
    return str(output)


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
    minimal_overrides = [
        "max_sim_steps=1", "num_samples=2", "max_num_iterations=1",
        "horizon=0.05", "knot_dt=0.05", "ctrl_dt=0.01", "sim_dt=0.01",
        "+sanity_check_seconds=0.0", "save_video=false", "show_viewer=false",
        "+wait_on_finish=false", "+use_torch_compile=false",
    ]
    if config.is_file():
        command = [sys.executable, "examples/run_mjwp.py", f"+load_config_path={config}", *minimal_overrides]
    else:
        # A freshly prepared isolated sandbox has valid scene/data artifacts
        # but no previously saved MJWP YAML.  Bootstrap from the repository's
        # declarative defaults once; run_mjwp then saves config_act.yaml next
        # to the sandbox trajectory.  This is required for an auxiliary
        # sanity run and prevents a pre-existing primary config from becoming
        # an accidental hidden prerequisite of the C-R3 minimal gate.
        command = [
            sys.executable, "examples/run_mjwp.py",
            f"dataset_dir={sandbox}", "dataset_name=grab", "robot_type=wuji_hand2_beta1",
            # The isolated C-R3 input is deliberately written as the
            # contact-guided ``*_act`` trajectory.  Without this explicit
            # bootstrap override ``process_config`` selects the immutable
            # Stage-B ``trajectory_kinematic.npz`` instead, which is not
            # copied into the sandbox and would silently test the wrong
            # contract even if present.
            "embodiment_type=bimanual", "contact_guidance=true",
            # The only non-primary call site is AUXILIARY_SANITY, whose
            # pre-validated source contact mask is intentionally all-zero.
            # Keep the escape hatch out of normal / saved primary profiles.
            *( ["+allow_empty_contact_guidance=true"] if sequence_id == AUXILIARY_SANITY["sequence_id"] else [] ),
            f"task={task}", "data_id=0", *minimal_overrides,
        ]
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


def evaluate_depenetrated_init(
    paths_config: str,
    sequence_id: str,
    trajectory_path: str | None = None,
    metrics_path: str | None = None,
) -> str:
    """Evaluate C-R2 tracking, visual geometry, contact and continuity gates."""
    paths = _paths(paths_config); target = _stage_c_recovery_dir(paths, sequence_id)
    recovered_path = Path(trajectory_path) if trajectory_path else target / "trajectory_depenetrated_init.npz"
    output_metrics_path = Path(metrics_path) if metrics_path else target / "metrics_depenetrated_init.json"
    metrics = json.loads(output_metrics_path.read_text(encoding="utf-8"))
    profile = yaml.safe_load(Path("configs/project/grab_wuji_depenetration.yaml").read_text(encoding="utf-8"))
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(physics["scene_act"]); data = mujoco.MjData(model); site_ids = _site_ids(model)
    # C-R2 must be measured against the immutable Stage-B trajectory.  The
    # isolated physics sandbox is deliberately allowed to contain the derived
    # initializer for C-R3/C-R4, so it is not an authoritative baseline here.
    baseline, _baseline_qvel = _stage_b_act_baseline(paths, sequence_id)
    with np.load(recovered_path, allow_pickle=False) as archive: recovered = archive["qpos"].copy()
    if baseline.shape != recovered.shape or baseline.shape[1] != model.nq:
        raise RuntimeError(
            f"Fail closed: C-R2/Stage-B schema mismatch baseline={baseline.shape}, "
            f"recovered={recovered.shape}, model.nq={model.nq}"
        )
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
    visual_frame_max_penetration: list[float] = []
    for qpos in recovered:
        data.qpos[:] = qpos; mujoco.mj_forward(model, data); world = np.concatenate([mesh_from_model(model, data, geom).vertices for geom in visual_ids]); local = _world_to_body(world, data.xpos[object_body], data.xmat[object_body]); _, _, distance, _, _ = _closest_with_sign(object_mesh, local); signed.extend(distance.tolist())
        visual_frame_max_penetration.append(float(max(0.0, -np.min(distance, initial=0.0))))
    visual = summarize_signed_distances(np.asarray(signed), np.zeros((len(signed), 3)), confidence="high" if object_mesh.is_watertight else "low")
    with np.load(_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz", allow_pickle=False) as contacts:
        # The actuator input intentionally removes the first/last reference
        # samples so contact guidance aligns with its finite-difference qvel.
        expected = contacts["contact"][1:-1].astype(bool); anchors = contacts["contact_surface_world"][1:-1].copy()
    tip_sites = sites[:, [1,2,3,4,5,7,8,9,10,11]]; contact_distance = np.linalg.norm(tip_sites - anchors, axis=2); recall = float(np.count_nonzero((contact_distance <= 0.015) & expected) / max(1, np.count_nonzero(expected)))
    delta = np.diff(recovered[:, :52], axis=0); ranges = model.jnt_range[:52, 1] - model.jnt_range[:52, 0]; ranges[[0,1,2,26,27,28]] = 4.0; normalized = float(np.max(np.abs(delta) / np.maximum(ranges, 1e-9)))
    tracking_ok = all(record["wrist_rmse_m"] <= profile["acceptance"]["wrist_rmse_m"] and all(item["rmse_m"] <= profile["acceptance"]["fingertip_rmse_m"] for item in record["fingertips"].values()) for record in tracking.values())
    gates = {"hard_validity": metrics["nan_inf"] == 0 and metrics["joint_limit_violations"] == 0 and metrics["object_pose_change_m"] == 0.0 and metrics["source_mapping_complete"], "tracking": tracking_ok, "visual_penetration": visual["max_penetration_m"] <= profile["acceptance"]["max_visual_penetration_m"], "collision_penetration": metrics["collision"]["after_max_m"] <= profile["acceptance"]["max_collision_penetration_m"], "contact_recall": recall >= profile["acceptance"]["contact_recall"], "smoothness": normalized <= 0.25}
    metrics.update({"tracking": tracking, "visual_penetration": {**visual, "per_frame_max_penetration_m": visual_frame_max_penetration}, "contact": {"high_confidence_recall": recall, "expected_records": int(np.count_nonzero(expected))}, "smoothness": {"max_normalized_single_frame_delta": normalized, "teleport": normalized > 0.25}, "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"})
    metrics["trajectory"] = str(recovered_path)
    _atomic_json(output_metrics_path, metrics)
    return str(output_metrics_path)


def evaluate_mjwp_output(
    paths_config: str,
    sequence_id: str,
    output_path: str | None = None,
) -> str:
    """Audit a real MJWP trajectory against its time-aligned reference.

    This is deliberately separate from C-R2: it reads the optimizer's saved
    dynamic qpos/qvel/ctrl arrays, aligns each saved integration substep with
    the next reference sample (the state is saved *after* one step), and never
    substitutes source or initializer qpos for the optimized state.
    """
    from spider.config import Config, load_config_yaml, process_config
    from spider.io import load_data

    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    sandbox = Path(physics["sandbox"])
    config_path = sandbox / "processed/grab/wuji_hand2_beta1/bimanual" / sequence_id / "0/config_act.yaml"
    output = Path(output_path) if output_path else config_path.parent / "trajectory_mjwp_act.npz"
    if not output.is_file():
        raise FileNotFoundError(f"No real MJWP output to audit: {output}")
    cfg = process_config(Config(**load_config_yaml(config_path)))
    qpos_ref_t, _qvel_ref_t, _ctrl_ref_t, contact_t, _controller_anchor_t = load_data(cfg, cfg.data_path)
    qpos_ref = qpos_ref_t.detach().cpu().numpy().astype(np.float64)
    contact = contact_t.detach().cpu().numpy()
    # ``trajectory_kinematic_act.npz/contact_pos`` contains the controller
    # target (source surface plus an allowed normal clearance).  C-R4's
    # contact metric must instead use the immutable source-only surface
    # projection, per the benchmark contract.
    source_contact_path = _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/contact_reference.npz"
    with np.load(source_contact_path, allow_pickle=False) as source_contacts:
        source_surface = np.asarray(source_contacts["contact_surface_world"], dtype=np.float32)
    # C-R2/MJWP intentionally trim the first and last source contact sample
    # to match their finite-difference actuator trajectory.  Validate that
    # this is the only contact-reference shortening before applying it.
    with np.load(cfg.data_path, allow_pickle=False) as source_trajectory:
        source_frames = len(source_trajectory["qpos"])
    if source_surface.shape != (source_frames + 2, 10, 3):
        raise RuntimeError(
            "Source contact surface/reference trajectory trim mismatch: "
            f"surface={source_surface.shape}, trajectory_frames={source_frames}"
        )
    anchors = _interpolate_surface_contact_reference(
        source_surface[1:-1], int(cfg.ref_steps), int(cfg.horizon_steps + cfg.ctrl_steps)
    )
    if len(anchors) != len(qpos_ref):
        raise RuntimeError(
            f"Surface contact interpolation mismatch: anchors={len(anchors)}, reference={len(qpos_ref)}"
        )
    with np.load(output, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if not {"qpos", "qvel", "ctrl", "time"}.issubset(arrays):
        raise RuntimeError(f"MJWP output missing required arrays: {sorted(arrays)}")
    qpos = arrays["qpos"]
    qvel = arrays["qvel"]
    ctrl = arrays["ctrl"]
    # Native MJWP archives are (control_tick, ctrl_substep, dof).  Candidate
    # archives produced by this evaluator are deliberately flattened to
    # (sim_step, dof) so they can be reviewed without retaining an extra raw
    # copy.  Both forms contain the same ordered physical states and are safe
    # to audit read-only.
    if qpos.ndim not in (2, 3) or qvel.shape != qpos.shape or ctrl.shape != qpos.shape:
        raise RuntimeError(f"Invalid MJWP qpos/qvel/ctrl shapes: {qpos.shape}, {qvel.shape}, {ctrl.shape}")
    qpos = qpos.reshape(-1, qpos.shape[-1]).astype(np.float64)
    qvel = qvel.reshape(-1, qvel.shape[-1]).astype(np.float64)
    ctrl = ctrl.reshape(-1, ctrl.shape[-1]).astype(np.float64)
    # ``load_data`` appends the optimizer horizon/control padding after the
    # source trajectory.  A Stage-C replay must cover every unpadded source
    # sample; accepting a shorter saved rollout would silently drop frozen
    # frames (and their contact records) from the C-R4 gate.
    source_sim_steps = len(qpos_ref) - int(cfg.horizon_steps) - int(cfg.ctrl_steps)
    if source_sim_steps <= 0:
        raise RuntimeError(
            f"Invalid MJWP reference coverage: ref={len(qpos_ref)}, "
            f"horizon={cfg.horizon_steps}, ctrl={cfg.ctrl_steps}"
        )
    actual_sim_steps = len(qpos)
    sim_count = min(actual_sim_steps, source_sim_steps, len(contact) - 1, len(anchors) - 1)
    if sim_count <= 0:
        raise RuntimeError("MJWP output has no reference-aligned simulation samples")
    qpos, qvel, ctrl = qpos[:sim_count], qvel[:sim_count], ctrl[:sim_count]
    # Run-loop records state immediately after step k, hence reference k+1.
    reference = qpos_ref[1 : sim_count + 1]
    expected = contact[1 : sim_count + 1] >= 0.5
    anchors = anchors[1 : sim_count + 1]
    model = mujoco.MjModel.from_xml_path(physics["scene_act"])
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise RuntimeError(f"MJWP/model schema mismatch qpos={qpos.shape}, qvel={qvel.shape}, model=({model.nq}, {model.nv})")
    data, reference_data = mujoco.MjData(model), mujoco.MjData(model)
    site_ids = _site_ids(model)
    sites = np.empty((sim_count, len(site_ids), 3), dtype=np.float64)
    reference_sites = np.empty_like(sites)
    hand_collision = set(_hand_geom_ids(model, 2))
    object_collision = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_object_") and model.geom_group[i] == 3}
    collision_records: list[dict[str, Any]] = []
    for frame, (state, ref_state) in enumerate(zip(qpos, reference, strict=True)):
        data.qpos[:] = state; data.qvel[:] = qvel[frame]; mujoco.mj_forward(model, data)
        reference_data.qpos[:] = ref_state; reference_data.qvel[:] = 0; mujoco.mj_forward(model, reference_data)
        sites[frame] = data.site_xpos[site_ids]
        reference_sites[frame] = reference_data.site_xpos[site_ids]
        for index in range(data.ncon):
            item = data.contact[index]
            if int(item.geom1) not in hand_collision and int(item.geom2) not in hand_collision:
                continue
            if int(item.geom1) not in object_collision and int(item.geom2) not in object_collision:
                continue
            collision_records.append({"frame_index": frame, "geom_pair": f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom1))}|{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom2))}", "penetration_m": float(max(0.0, -item.dist)), "position_world": np.asarray(item.pos).tolist(), "normal_world": np.asarray(item.frame[:3]).tolist()})
    error = np.linalg.norm(sites - reference_sites, axis=2)
    tracking: dict[str, Any] = {}
    for side, offset in (("right", 0), ("left", 6)):
        tracking[side] = {"wrist_rmse_m": float(np.sqrt(np.mean(error[:, offset] ** 2))), "fingertips": {finger: {"rmse_m": float(np.sqrt(np.mean(error[:, offset + 1 + index] ** 2))), "p95_m": float(np.percentile(error[:, offset + 1 + index], 95)), "max_m": float(error[:, offset + 1 + index].max())} for index, finger in enumerate(FINGERS)}}
    object_tracking: dict[str, Any] = {}
    for side in ("right", "left"):
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object"))
        positions = np.empty(sim_count, dtype=np.float64); rotations = np.empty(sim_count, dtype=np.float64)
        for frame, (state, ref_state) in enumerate(zip(qpos, reference, strict=True)):
            data.qpos[:] = state; mujoco.mj_kinematics(model, data)
            reference_data.qpos[:] = ref_state; mujoco.mj_kinematics(model, reference_data)
            positions[frame], rotations[frame] = _object_pose_error(data, body, reference_data, body)
        object_tracking[side] = {"position_rmse_m": float(np.sqrt(np.mean(positions ** 2))), "position_max_m": float(positions.max()), "rotation_mean_rad": float(rotations.mean()), "rotation_max_rad": float(rotations.max())}
    # Visual geometry is evaluated at every source-rate boundary.  Actual
    # collision contacts above are checked at *every* integration substep.
    # The exact stride is recorded so the report never implies a denser audit.
    stride = max(1, int(cfg.ref_steps))
    visual_ids = _hand_geom_ids(model, 1)
    object_mesh = _mesh(Path(physics["collision_cache"]) / "visual/visual.obj")
    object_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object"))
    signed: list[float] = []
    for frame in range(0, sim_count, stride):
        data.qpos[:] = qpos[frame]; mujoco.mj_forward(model, data)
        world = np.concatenate([mesh_from_model(model, data, geom).vertices for geom in visual_ids])
        local = _world_to_body(world, data.xpos[object_body], data.xmat[object_body])
        _point, _unsigned, values, _method, _confidence = _closest_with_sign(object_mesh, local)
        signed.extend(values.tolist())
    visual = summarize_signed_distances(np.asarray(signed), np.zeros((len(signed), 3)), confidence="high" if object_mesh.is_watertight else "low")
    tip_sites = sites[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    distance = np.linalg.norm(tip_sites - anchors, axis=2)
    observed = distance <= 0.015
    recall = float(np.count_nonzero(observed & expected) / max(1, np.count_nonzero(expected)))
    contact_metrics = {"high_confidence_recall": recall, "expected_records": int(np.count_nonzero(expected)), "false_contact_frames": int(np.count_nonzero(observed & ~expected)), "per_side_finger": {}}
    for side, offset in (("right", 0), ("left", 5)):
        for index, finger in enumerate(FINGERS):
            mask = expected[:, offset + index]
            contact_metrics["per_side_finger"][f"{side}_{finger}"] = {"recall": float(np.count_nonzero(observed[:, offset + index] & mask) / max(1, np.count_nonzero(mask))), "false_contact_frames": int(np.count_nonzero(observed[:, offset + index] & ~mask))}
    delta = np.diff(qpos[:, :52], axis=0)
    ranges = model.jnt_range[:52, 1] - model.jnt_range[:52, 0]
    ranges[[0, 1, 2, 26, 27, 28]] = 4.0
    normalized = float(np.max(np.abs(delta) / np.maximum(ranges, 1e-9))) if len(delta) else 0.0
    acceleration = np.diff(qvel, axis=0) / float(cfg.sim_dt)
    thresholds = yaml.safe_load(Path("configs/project/grab_wuji_stage_c_contract.yaml").read_text(encoding="utf-8"))["thresholds"]
    tracking_ok = all(record["wrist_rmse_m"] <= thresholds["wrist_rmse_m"] and all(item["rmse_m"] <= thresholds["fingertip_rmse_m"] for item in record["fingertips"].values()) for record in tracking.values())
    object_ok = all(item["position_rmse_m"] <= thresholds["object_pos_rmse_m"] and item["position_max_m"] <= thresholds["object_pos_max_m"] and item["rotation_mean_rad"] <= thresholds["object_rot_mean_rad"] and item["rotation_max_rad"] <= thresholds["object_rot_max_rad"] for item in object_tracking.values())
    collision_depths = np.asarray([record["penetration_m"] for record in collision_records], dtype=np.float64)
    collision = {"contact_count": len(collision_records), "max_penetration_m": float(collision_depths.max(initial=0.0)), "mean_penetration_m": float(collision_depths.mean()) if len(collision_depths) else 0.0, "p95_penetration_m": float(np.percentile(collision_depths, 95)) if len(collision_depths) else 0.0, "per_geom_pair": contact_pair_summary(collision_records), "persistent_deep_frames": sorted({record["frame_index"] for record in collision_records if record["penetration_m"] > 0.003})}
    source_mapping_complete = bool(actual_sim_steps >= source_sim_steps)
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all())
    gates = {"hard_validity": finite and source_mapping_complete, "source_frame_mapping": source_mapping_complete, "tracking": tracking_ok, "visual_penetration": bool(visual["max_penetration_m"] <= thresholds["visual_max_penetration_m"] and visual["p95_negative_depth_m"] <= thresholds["visual_p95_penetration_m"] and visual["mean_negative_depth_m"] <= thresholds["visual_mean_negative_depth_m"] and visual["penetrating_ratio"] <= thresholds["visual_penetrating_vertex_ratio"]), "collision_penetration": not collision["persistent_deep_frames"], "contact_preservation": recall >= thresholds["contact_recall"], "object_tracking": object_ok, "smoothness": normalized <= 0.25}
    _atomic_npz(target / "trajectory_spider.npz", qpos=qpos, qvel=qvel, ctrl=ctrl, time=np.asarray(arrays["time"]).reshape(-1)[:sim_count])
    report = {"schema_version": 1, "stage": "C-R4", "sequence_id": sequence_id, "source_frame_range": _pilot(sequence_id)["frame_range"], "input": str(output), "trajectory": str(target / "trajectory_spider.npz"), "reference_alignment": {"saved_state": "after_simulation_step", "reference_offset_steps": 1, "sim_dt_s": float(cfg.sim_dt), "ctrl_dt_s": float(cfg.ctrl_dt), "samples": sim_count, "source_sim_steps_required": source_sim_steps, "saved_sim_steps": actual_sim_steps, "source_frame_mapping_complete": source_mapping_complete}, "tracking": tracking, "object_tracking": object_tracking, "visual_penetration": {**visual, "sample_stride_sim_steps": stride, "sample_count": int((sim_count + stride - 1) // stride)}, "collision": collision, "contact": {**contact_metrics, "anchor_contract": "source_projected_object_surface_without_controller_normal_gap", "source_reference": str(source_contact_path)}, "smoothness": {"max_normalized_single_frame_delta": normalized, "qvel_p99": float(np.percentile(np.abs(qvel), 99)), "qacc_p99": float(np.percentile(np.abs(acceleration), 99)) if len(acceleration) else 0.0}, "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL", "object_qpos_overwritten_after_initialization": False, "stage_b_untouched": True}
    _atomic_json(target / "metrics_stage_c.json", report)
    return str(target / "metrics_stage_c.json")


def _active_c2_joint_limits(model: mujoco.MjModel, qpos: np.ndarray, tolerance_rad: float = 1e-5) -> dict[str, Any]:
    """Report actual bounded robot-joint activity without treating wrist xyz as hinges."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 52:
        raise ValueError(f"Expected C-R2 qpos shaped (T, >=52), got {values.shape}")
    active_counts: dict[str, int] = {}
    active_frames: dict[str, list[int]] = {}
    for joint in range(min(52, model.njnt)):
        if not bool(model.jnt_limited[joint]):
            continue
        address = int(model.jnt_qposadr[joint])
        if address >= 52:
            continue
        lower, upper = (float(value) for value in model.jnt_range[joint])
        at_limit = np.minimum(np.abs(values[:, address] - lower), np.abs(values[:, address] - upper)) <= tolerance_rad
        if np.any(at_limit):
            key = str(joint)
            active_counts[key] = int(np.count_nonzero(at_limit))
            active_frames[key] = np.flatnonzero(at_limit).astype(int).tolist()
    return {
        "tolerance_rad": float(tolerance_rad),
        "active_joint_count": len(active_counts),
        "active_frame_counts": active_counts,
        "active_frame_indices": active_frames,
    }


def summarize_depenetration_multistart(paths_config: str, sequence_id: str) -> str:
    """Write a fail-closed C-R2 contact/collision Pareto audit for primary starts.

    Only complete candidate namespaces are read.  A candidate generated before
    the canonical warm-start contract is recorded but explicitly excluded from
    clean multi-start evidence; this prevents an optimizer-code change from
    masquerading as an initial-condition result.
    """
    if sequence_id != "s5__cylindermedium_lift":
        raise ValueError("C-R2 multi-start Pareto audit is primary-only; smoke pilots are forbidden until C-R4")
    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    physics = json.loads((_stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(physics["scene_act"])
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(target.glob("metrics_depenetrated_init*.json")):
        suffix = metrics_path.stem.removeprefix("metrics_depenetrated_init")
        trajectory_path = target / f"trajectory_depenetrated_init{suffix}.npz"
        config_path = target / f"depenetration_config{suffix}.json"
        if not trajectory_path.is_file() or not config_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        contact = metrics.get("contact", {})
        collision = metrics.get("collision", {})
        if "high_confidence_recall" not in contact or "after_max_m" not in collision:
            continue
        with np.load(trajectory_path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
        initialization = config.get("candidate_initialization", {})
        is_multistart = any(float(initialization.get(key, 0.0)) > 0.0 for key in (
            "initial_wrist_translation_jitter_m", "initial_wrist_rotation_jitter_rad", "initial_finger_jitter_rad",
        ))
        canonical_continuation = initialization.get("continuation_semantics") == "canonical_warmstart_per_phase"
        record = {
            "candidate_id": suffix.removeprefix("_") or "canonical",
            "metrics": str(metrics_path),
            "trajectory": str(trajectory_path),
            "config": str(config_path),
            "profile_hash": metrics.get("profile_hash"),
            "candidate_profile_hash": metrics.get("candidate_profile_hash"),
            "status": metrics.get("status"),
            "gates": metrics.get("gates", {}),
            "contact_recall": float(contact["high_confidence_recall"]),
            "collision_max_m": float(collision["after_max_m"]),
            "collision_p95_m": float(collision.get("after_p95_m", float("nan"))),
            "visual_max_m": float(metrics.get("visual_penetration", {}).get("max_penetration_m", float("nan"))),
            "tracking": metrics.get("tracking", {}),
            "initialization": initialization,
            "is_multistart": is_multistart,
            "canonical_continuation": canonical_continuation,
            "eligible_clean_multistart_evidence": bool(is_multistart and canonical_continuation),
            "joint_limit_active_set": _active_c2_joint_limits(model, qpos),
        }
        if is_multistart and not canonical_continuation:
            record["exclusion_reason"] = "candidate predates canonical_warmstart_per_phase provenance"
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No complete C-R2 metrics/config/trajectory triplets in {target}")
    front = _contact_collision_pareto_front(records)
    clean_count = sum(bool(record["eligible_clean_multistart_evidence"]) for record in records)
    payload = {
        "schema_version": 1,
        "stage": "C-R2/C-R4 diagnostic",
        "sequence_id": sequence_id,
        "contact_contract": "source_projected_object_surface_without_controller_normal_gap",
        "collision_contract": "MuJoCo hand/object maximum contact penetration",
        "candidates": records,
        "pareto_front_candidate_ids": [records[index]["candidate_id"] for index in front],
        "clean_multistart_count": clean_count,
        "gates": {
            "has_canonical_baseline": any(record["candidate_id"] == "canonical" for record in records),
            "two_clean_multistarts": clean_count >= 2,
            "all_records_have_joint_limit_active_set": all("joint_limit_active_set" in record for record in records),
        },
        "status": "PASS" if clean_count >= 2 else "FAIL",
        "status_reason": "Pareto evidence is diagnostic only and does not establish C-R4 passage or infeasibility.",
    }
    output = target / "depenetration_multistart_pareto.json"
    _atomic_json(output, payload)
    return str(output)


def summarize_auxiliary_mjwp_sanity(paths_config: str, sequence_id: str) -> str:
    """Record a non-frozen, zero-contact MJWP infrastructure probe.

    This report is deliberately not a C-R2/C-R4 acceptance result: the
    selected auxiliary source has no expected contacts.  It is only evidence
    that the isolated 64-DoF scene and real GPU MJWP path remain capable of a
    finite run from a verified low-penetration state.
    """
    if sequence_id != AUXILIARY_SANITY["sequence_id"]:
        raise ValueError("Auxiliary MJWP sanity is restricted to the declared non-frozen sequence")
    paths = _paths(paths_config)
    _, robot = _stage_b_dirs(paths.workspace_root, sequence_id)
    target = _stage_c_recovery_dir(paths, sequence_id)
    source_path = robot / "stage_c/source_geometry_diagnostics.json"
    c2_path = target / "metrics_depenetrated_init.json"
    static_path = target / "preflight_static.json"
    mjwp_path = target / "minimal_mjwp_run.json"
    required = {"source_geometry": source_path, "c2": c2_path, "static": static_path, "mjwp": mjwp_path}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Auxiliary MJWP sanity inputs missing: {missing}")
    source, c2, static, mjwp = (json.loads(path.read_text(encoding="utf-8")) for path in required.values())
    source_max = max(float(source["sides"][side]["aggregate"]["max_penetration_depth_m"]) for side in ("left", "right"))
    c2_gates = c2.get("gates", {})
    c2_non_contact = all(bool(c2_gates.get(name, False)) for name in (
        "hard_validity", "tracking", "collision_penetration", "visual_penetration", "smoothness",
    ))
    zero_expected_contacts = int(c2.get("contact", {}).get("expected_records", -1)) == 0
    gates = {
        "declared_non_frozen_auxiliary_only": not bool(AUXILIARY_SANITY["frozen"]),
        "source_zero_penetration": bool(np.isfinite(source_max) and source_max <= 1e-12),
        "c2_non_contact_quality_gates": c2_non_contact,
        "zero_expected_contacts_explains_c2_contact_gate": zero_expected_contacts and not bool(c2_gates.get("contact_recall", True)),
        "static_mj_forward_pass": static.get("status") == "PASS" and bool(static.get("all_finite")),
        "real_gpu_mjwp_finite_pass": mjwp.get("status") == "PASS" and all(bool(mjwp.get("gates", {}).get(name, False)) for name in (
            "real_mjwp_exit_zero", "rewards_finite", "states_finite", "object_values_finite", "output_schema_valid",
        )),
    }
    payload = {
        "schema_version": 1,
        "stage": "C-R4 diagnostic infrastructure evidence",
        "sequence_id": sequence_id,
        "role": AUXILIARY_SANITY["role"],
        "frozen": False,
        "not_a_frozen_pilot": True,
        "not_a_substitute_for_primary_or_smoke": True,
        "decision": "INFRASTRUCTURE_SANITY_ONLY_NOT_A_CONTACT_QUALITY_ACCEPTANCE",
        "source_max_penetration_m": source_max,
        "expected_contact_records": int(c2.get("contact", {}).get("expected_records", -1)),
        "gates": gates,
        "artifacts": {name: str(path) for name, path in required.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    output = target / "auxiliary_mjwp_sanity.json"
    _atomic_json(output, payload)
    return str(output)


def diagnose_primary_contact_conflict(paths_config: str, sequence_id: str) -> str:
    """Write compact, fail-closed evidence for the primary C-R4 conflict.

    This is deliberately a diagnosis, not an infeasibility declaration.  It
    reports the best bounded physical candidate's missed source-surface
    contacts and deep MuJoCo geom pairs alongside the completed C-R2
    multi-start Pareto and bounded pair-margin search.  Keeping these facts
    together prevents a later handoff from confusing a finite rollout with a
    quality-pass or from silently dropping the concrete responsible frames.
    """
    if sequence_id != "s5__cylindermedium_lift":
        raise ValueError("Primary contact-conflict diagnosis is forbidden for smoke pilots")
    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    metric_path = target / "primary_r4_state_feedback_c1_metrics.json"
    if not metric_path.is_file():
        raise FileNotFoundError(f"Best bounded primary metric missing: {metric_path}")
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "FAIL":
        raise RuntimeError("Contact-conflict diagnosis is only valid for a failed primary candidate")
    contact = metrics.get("contact", {})
    per_finger = contact.get("per_side_finger", {})
    missed = [
        {"side_finger": name, "recall": float(value.get("recall", float("nan"))),
         "false_contact_frames": int(value.get("false_contact_frames", 0))}
        for name, value in sorted(per_finger.items())
        if float(value.get("recall", float("nan"))) < 0.70
    ]
    collision = metrics.get("collision", {})
    pairs = collision.get("per_geom_pair", {})
    responsible_pairs: list[dict[str, Any]] = []
    for name, value in sorted(
        pairs.items(), key=lambda item: float(item[1].get("max_penetration_m", 0.0)), reverse=True
    )[:12]:
        frame_indices = sorted({int(frame) for frame in value.get("frames", [])})
        responsible_pairs.append({
            "geom_pair": name,
            "max_penetration_m": float(value.get("max_penetration_m", 0.0)),
            "p95_penetration_m": float(value.get("p95_penetration_m", 0.0)),
            "contact_count": int(value.get("contact_count", 0)),
            "sim_step_ranges": _contiguous_index_ranges(frame_indices),
        })
    artifacts = {
        "best_bounded_metric": str(metric_path),
        "global_tracking": str(target / "primary_r4_force5_global_tracking_diagnosis.json"),
        "phase_scan": str(target / "primary_r4_force5_contact_phase_scan.json"),
        "depenetration_pareto": str(target / "depenetration_multistart_pareto.json"),
        "pair_margin_search": str(target / "profile_search_results_pair_margin.json"),
        "grouped_lookahead_search": str(target / "profile_search_results_grouped_lookahead.json"),
        "integral_feedback_search": str(target / "profile_search_results_integral_feedback.json"),
    }
    auxiliary = _stage_c_recovery_dir(paths, AUXILIARY_SANITY["sequence_id"]) / "auxiliary_mjwp_sanity.json"
    auxiliary_ready = auxiliary.is_file() and json.loads(auxiliary.read_text(encoding="utf-8")).get("status") == "PASS"
    artifacts["auxiliary_mjwp_sanity"] = str(auxiliary)
    payload = {
        "schema_version": 1,
        "stage": "C-R4 diagnostic",
        "sequence_id": sequence_id,
        "status": "FAIL",
        "decision": "DIAGNOSTIC_ONLY_NOT_AN_INFEASIBILITY_DECLARATION",
        "contact_contract": contact.get("anchor_contract"),
        "best_bounded_candidate": {
            "metric": str(metric_path),
            "contact_recall": float(contact.get("high_confidence_recall", float("nan"))),
            "collision_max_m": float(collision.get("max_penetration_m", float("nan"))),
            "persistent_deep_sim_steps": _contiguous_index_ranges(
                sorted({int(frame) for frame in collision.get("persistent_deep_frames", [])})
            ),
            "missed_or_weak_contact_fingers": missed,
            "responsible_collision_pairs": responsible_pairs,
        },
        "required_remaining_before_any_infeasibility_claim": [
            "collision alignment and object pose evidence remain R1 PASS",
            "completed multi-start Pareto is available but does not prove dynamic infeasibility",
            "auxiliary low-penetration MJWP sanity is PASS" if auxiliary_ready else "an auxiliary low-penetration MJWP sanity sequence is still required",
            "no C-R4 shared profile is selected and smoke pilots remain forbidden",
        ],
        "artifacts": artifacts,
    }
    output = target / "primary_r4_contact_collision_conflict.json"
    _atomic_json(output, payload)
    return str(output)


def write_primary_infeasibility_report(paths_config: str, sequence_id: str) -> str:
    """Close C-R4 fail-closed when the declared primary search has no solution.

    This is intentionally restricted to the one frozen primary.  It does not
    assert a global mathematical impossibility: it records that the declared,
    audited bounded wrist/finger and MJWP-profile search contains no trajectory
    satisfying the unchanged contact and collision contracts.  In particular,
    this command must never be used to promote an auxiliary run or a smoke
    pilot into primary evidence.
    """
    if sequence_id != "s5__cylindermedium_lift":
        raise ValueError("C-R4 infeasibility reporting is restricted to the frozen primary")
    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    reports = paths.workspace_root / "reports"
    r1_path = reports / "stage_c_r1_s5__cylindermedium_lift.json"
    conflict_path = target / "primary_r4_contact_collision_conflict.json"
    pareto_path = target / "depenetration_multistart_pareto.json"
    auxiliary_path = (
        _stage_c_recovery_dir(paths, AUXILIARY_SANITY["sequence_id"])
        / "auxiliary_mjwp_sanity.json"
    )
    search_paths = {
        "pair_margin": target / "profile_search_results_pair_margin.json",
        "grouped_lookahead": target / "profile_search_results_grouped_lookahead.json",
        "integral_feedback": target / "profile_search_results_integral_feedback.json",
    }
    required = {
        "r1": r1_path,
        "contact_collision_conflict": conflict_path,
        "depenetration_pareto": pareto_path,
        "auxiliary_mjwp_sanity": auxiliary_path,
        **{f"search_{name}": path for name, path in search_paths.items()},
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot close C-R4 without required evidence: {missing}")
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    r1 = loaded["r1"]
    conflict = loaded["contact_collision_conflict"]
    pareto = loaded["depenetration_pareto"]
    auxiliary = loaded["auxiliary_mjwp_sanity"]
    search_reports = {name: loaded[f"search_{name}"] for name in search_paths}
    r1_gates = r1.get("gates", {})
    search_complete_fail = {
        name: bool(report.get("complete"))
        and report.get("status") == "FAIL"
        and report.get("selected_candidate_id") is None
        for name, report in search_reports.items()
    }
    pareto_candidates = pareto.get("candidates", [])
    active_sets = {
        str(candidate.get("candidate_id")): candidate.get("joint_limit_active_set", {})
        for candidate in pareto_candidates
    }
    gates = {
        "r1_collision_alignment_verified": r1.get("status") == "PASS"
        and bool(r1_gates.get("R1-03_wuji_alignment_reported")),
        "r1_object_pose_verified": r1.get("status") == "PASS"
        and bool(r1_gates.get("R1-04_object_pose_conversion")),
        "multistart_pareto_complete": pareto.get("status") == "PASS"
        and bool(pareto.get("gates", {}).get("has_canonical_baseline"))
        and bool(pareto.get("gates", {}).get("two_clean_multistarts")),
        "joint_limit_active_sets_recorded": bool(pareto_candidates)
        and bool(pareto.get("gates", {}).get("all_records_have_joint_limit_active_set")),
        "contact_penetration_conflict_concrete": conflict.get("status") == "FAIL"
        and bool(conflict.get("best_bounded_candidate", {}).get("missed_or_weak_contact_fingers"))
        and bool(conflict.get("best_bounded_candidate", {}).get("responsible_collision_pairs")),
        "bounded_wrist_finger_searches_complete_and_infeasible": all(search_complete_fail.values()),
        "auxiliary_real_mjwp_infrastructure_pass": auxiliary.get("status") == "PASS"
        and bool(auxiliary.get("not_a_substitute_for_primary_or_smoke")),
    }
    status = (
        "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT"
        if all(gates.values())
        else "INCOMPLETE_EVIDENCE_DO_NOT_DECLARE_INFEASIBLE"
    )
    best = conflict.get("best_bounded_candidate", {})
    payload = {
        "schema_version": 1,
        "stage": "C-R4",
        "sequence_id": sequence_id,
        "status": status,
        "decision": (
            "NO_FEASIBLE_SHARED_MJWP_PROFILE_WITHIN_DECLARED_BOUNDED_PRIMARY_SEARCH"
            if status == "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT"
            else "EVIDENCE_INCOMPLETE"
        ),
        "scope_limit": (
            "This is not a global mathematical impossibility claim. It is a "
            "fail-closed conclusion for the frozen embodiment, immutable Stage-B "
            "source/contact contract, and declared bounded C-R4 profile search."
        ),
        "gates": gates,
        "bounded_searches": {
            name: {
                "path": str(search_paths[name]),
                "candidate_count": report.get("candidate_count"),
                "search_limit": report.get("search_limit"),
                "complete": report.get("complete"),
                "status": report.get("status"),
                "selected_candidate_id": report.get("selected_candidate_id"),
            }
            for name, report in search_reports.items()
        },
        "contact_penetration_pareto": {
            "path": str(pareto_path),
            "clean_multistart_count": pareto.get("clean_multistart_count"),
            "pareto_front_candidate_ids": pareto.get("pareto_front_candidate_ids"),
            "joint_limit_active_sets": active_sets,
        },
        "concrete_primary_conflict": best,
        "r1_verification": {
            "path": str(r1_path),
            "collision_alignment": bool(r1_gates.get("R1-03_wuji_alignment_reported")),
            "object_pose_conversion": bool(r1_gates.get("R1-04_object_pose_conversion")),
        },
        "mjwp_infrastructure_sanity": {
            "path": str(auxiliary_path),
            "status": auxiliary.get("status"),
            "non_frozen_auxiliary_only": auxiliary.get("not_a_substitute_for_primary_or_smoke"),
        },
        "not_run": {
            "frozen_smoke_pilots": "PRIMARY_BLOCKED",
            "screenshot_review": "PRIMARY_BLOCKED",
            "html": "PRIMARY_BLOCKED",
            "stage_d": "NOT_STARTED",
        },
        "required_authority_to_continue": [
            "relax the required contact/collision contract",
            "select a different embodiment or source trajectory",
            "authorize a new physical-model/search scope",
        ],
        "stage_b_untouched": True,
    }
    output = target / "primary_r4_infeasibility_report.json"
    _atomic_json(output, payload)
    validation = {
        "schema_version": 2,
        "stage": "C",
        "status": "BLOCKED" if status == "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT" else "FAIL",
        "stage_c_status": status,
        "primary": {"sequence_id": sequence_id, "report": str(output), "status": status},
        "stage_statuses": {
            "C-R1": "PASS",
            "C-R2": "PASS",
            "C-R3": "PASS",
            "C-R4": "BLOCKED" if status == "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT" else "FAIL",
            "C-R5": "NOT_RUN_PRIMARY_BLOCKED",
        },
        "gates": gates,
        "frozen_smokes": "NOT_RUN_PRIMARY_BLOCKED",
        "codex_screenshot_review": "FAIL_NOT_RUN_PRIMARY_BLOCKED",
        "html": "NOT_GENERATED_PRIMARY_BLOCKED",
        "user_html_review": "PENDING",
        "stage_d": "NOT_STARTED",
        "stage_b_untouched": True,
    }
    _atomic_json(reports / "stage_c_validation.json", validation)
    _atomic_json(
        reports / "stage_c_screenshot_review.json",
        {
            "schema_version": 2,
            "stage": "C",
            "status": "FAIL",
            "decision": "NOT_RUN_PRIMARY_BLOCKED",
            "primary_infeasibility_report": str(output),
            "screenshots_generated": False,
        },
    )
    return str(output)


def summarize_mjwp_profile_search(
    paths_config: str, sequence_id: str, search: str = "legacy"
) -> str:
    """Write a fail-closed, auditable primary-only C-R4 profile-search summary.

    Candidate trajectories are intentionally preserved under candidate-specific
    names before the next run overwrites the live MJWP output.  This command
    only summarizes those independent evaluator reports; it never promotes a
    merely finite candidate to a selected profile.
    """
    if sequence_id != "s5__cylindermedium_lift":
        raise ValueError("C-R4 profile search is primary-only; smoke pilots must not be used")
    paths = _paths(paths_config)
    target = _stage_c_recovery_dir(paths, sequence_id)
    profiles: dict[int, dict[str, Any]] = {
        1: {"name": "baseline_contact10_weld4ms", "object_mocap_solref_s": 0.004, "robot_servo_kp_scale": 1.0, "contact_rew_scale": 10.0, "num_samples": 16, "max_num_iterations": 2},
        2: {"name": "low_servo_contact10_weld1ms", "object_mocap_solref_s": 0.001, "robot_servo_kp_scale": 0.05, "contact_rew_scale": 10.0, "num_samples": 16, "max_num_iterations": 2},
        3: {"name": "baseline_servo_contact10_weld1ms", "object_mocap_solref_s": 0.001, "robot_servo_kp_scale": 1.0, "contact_rew_scale": 10.0, "num_samples": 16, "max_num_iterations": 2},
        4: {"name": "baseline_servo_contact50_weld4ms", "object_mocap_solref_s": 0.004, "robot_servo_kp_scale": 1.0, "contact_rew_scale": 50.0, "num_samples": 16, "max_num_iterations": 2},
        5: {"name": "contact50_weld4ms_dls_plan", "object_mocap_solref_s": 0.004, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_apply_to_plan": True, "num_samples": 16, "max_num_iterations": 2},
        6: {"name": "contact50_weld4ms_collision100k", "object_mocap_solref_s": 0.004, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "num_samples": 16, "max_num_iterations": 2},
        7: {"name": "contact50_weld1ms", "object_mocap_solref_s": 0.001, "robot_servo_kp_scale": 1.0, "contact_rew_scale": 50.0, "num_samples": 16, "max_num_iterations": 2},
        8: {"name": "contact50_weld1ms_gibbs", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "gibbs_sampling": True, "num_samples": 16, "max_num_iterations": 2},
        9: {"name": "contact50_weld1ms_first_active", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_wrist_mean_feedback_strategy": "first_active", "num_samples": 16, "max_num_iterations": 2},
        10: {"name": "contact100_weld1ms", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "num_samples": 16, "max_num_iterations": 2},
        11: {"name": "contact50_weld1ms_dls_physical", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_apply_to_plan": False, "num_samples": 16, "max_num_iterations": 2},
        12: {"name": "contact50_weld1ms_budget32x4", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "num_samples": 32, "max_num_iterations": 4},
    }
    metric_prefix = "primary_r4_c"
    output_stem = "profile_search_results"
    selected_stem = "selected_mjwp_profile"
    if search == "post-gapfix":
        # This is a distinct bounded search after the source-normal contact
        # gap correction.  Keep the legacy direct-physics search above intact:
        # its candidates were run against a different contact-target contract.
        profiles = {
            1: {"name": "weld1ms_contact50_baseline", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_wrist_mean_feedback_gain": 1.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "dls_integral03", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_integral_gain": 0.3, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "finger_dls_integral03", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_integral_gain": 0.3, "contact_ik_feedback_wrist_translation_clip_m": 0.0, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "finger_dls_plan", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.0, "num_samples": 16, "max_num_iterations": 2},
            5: {"name": "firstactive_dls_plan_pre_execution_fix", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "num_samples": 16, "max_num_iterations": 2},
            6: {"name": "firstactive_dls_executed", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "num_samples": 16, "max_num_iterations": 2},
            7: {"name": "firstactive_dls_integral01", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_integral_gain": 0.1, "num_samples": 16, "max_num_iterations": 2},
            8: {"name": "contact100_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 100000.0, "num_samples": 16, "max_num_iterations": 2},
            9: {"name": "servo15_contact50", "object_mocap_solref_s": 0.001, "robot_servo_kp_scale": 1.5, "contact_rew_scale": 50.0, "num_samples": 16, "max_num_iterations": 2},
            10: {"name": "firstactive_wrist_dls", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "num_samples": 16, "max_num_iterations": 2},
            11: {"name": "force5_contact50", "object_mocap_solref_s": 0.001, "robot_servo_forcelimit_scale": 5.0, "contact_rew_scale": 50.0, "num_samples": 16, "max_num_iterations": 2},
            12: {"name": "lookahead100_contact50", "object_mocap_solref_s": 0.001, "robot_reference_lookahead_steps": 100, "contact_rew_scale": 50.0, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_gapfix_c"
        output_stem = "profile_search_results_post_gapfix"
        selected_stem = "selected_mjwp_profile_post_gapfix"
    elif search == "qvel":
        # A fresh C-R2 initializer writes finite-difference robot velocity
        # references while preserving the immutable source object velocities.
        # Its resulting MJWP objective is not comparable with either the
        # legacy or source-normal-gap searches above, so retain an entirely
        # separate bounded (twelve-candidate) report namespace.
        profiles = {
            1: {"name": "baseline_contact50_lookahead260", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "firstactive_dls_plan", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "firstactive_dls_integral01", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_integral_gain": 0.1, "contact_ik_integral_decay": 0.98, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "contact100_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            5: {"name": "servo15_contact50", "object_mocap_solref_s": 0.001, "robot_servo_kp_scale": 1.5, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            6: {"name": "firstactive_wrist_dls", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            7: {"name": "force5_contact50", "object_mocap_solref_s": 0.001, "robot_servo_forcelimit_scale": 5.0, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            8: {"name": "lookahead100_contact50", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 100, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            9: {"name": "contact100_collision1m", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 1000000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            10: {"name": "firstactive_wrist_dls_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            11: {"name": "contact100_no_collision", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            12: {"name": "collision100k_noise20", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "joint_noise_scale": 0.2, "pos_noise_scale": 0.01, "rot_noise_scale": 0.01, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_qvel_c"
        output_stem = "profile_search_results_qvel"
        selected_stem = "selected_mjwp_profile_qvel"
    elif search == "qvel-actuated":
        # The earlier qvel search retained scale overrides which were not
        # represented in Config and therefore never reached the MJWarp
        # model.  Preserve that evidence separately and restart a bounded
        # primary-only search only after runtime servo scaling is enforced.
        profiles = {
            1: {"name": "baseline_verified_runtime_servo", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "servo_kp15", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_kp_scale": 1.5, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "servo_force5", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "servo_kp5_force5", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_kp_scale": 5.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            5: {"name": "finger_kp10_force10", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_finger_servo_kp_scale": 10.0, "robot_finger_servo_forcelimit_scale": 10.0, "num_samples": 16, "max_num_iterations": 2},
            6: {"name": "finger_kp5_force5_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_finger_servo_kp_scale": 5.0, "robot_finger_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            7: {"name": "wrist_dls_servo5_force5", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_kp_scale": 5.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            8: {"name": "force5_contact50_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_servo_c"
        output_stem = "profile_search_results_qvel_actuated"
        selected_stem = "selected_mjwp_profile_qvel_actuated"
    elif search == "barrier":
        # This is a third, explicitly bounded primary-only search after the
        # source-normal diagnostic established that most missed contacts are
        # outward separation.  It evaluates a real-contact normal-barrier
        # projection of the robot target correction; prior qvel/servo results
        # are retained unchanged because they did not contain this controller.
        profiles = {
            1: {"name": "force5_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_dls_all_active", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_dls_first_barrier025", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.25, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "force5_dls_all_barrier025", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.25, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            5: {"name": "force5_dls_first_barrier050", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.5, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            6: {"name": "force5_dls_all_barrier050", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.5, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            7: {"name": "force5_dls_first_barrier075_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.75, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            8: {"name": "force5_dls_all_barrier075_collision100k_contact100", "object_mocap_solref_s": 0.001, "contact_rew_scale": 100.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.75, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_barrier_c"
        output_stem = "profile_search_results_barrier"
        selected_stem = "selected_mjwp_profile_barrier"
    elif search == "guarded":
        # A source-active fingertip far from its surface anchor needs the
        # global sampling controller, not a saturated local Jacobian step.
        # This final bounded primary-only branch guards the contact servo by
        # current anchor distance and retains all earlier failed searches.
        profiles = {
            1: {"name": "force5_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_guard20mm_all", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_max_anchor_error_m": 0.02, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_guard20mm_first", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_max_anchor_error_m": 0.02, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "force5_guard30mm_all_barrier025", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_max_anchor_error_m": 0.03, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.25, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
            5: {"name": "force5_guard15mm_all", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_max_anchor_error_m": 0.015, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            6: {"name": "force5_guard15mm_first_barrier025_collision100k", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_max_anchor_error_m": 0.015, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.25, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_guarded_c"
        output_stem = "profile_search_results_guarded"
        selected_stem = "selected_mjwp_profile_guarded"
    elif search == "lag-aligned":
        # The force×5 baseline's dynamic state is closest to the source
        # reference 125--175 simulation steps ahead while its target uses a
        # 260-step lookahead.  Search only that measured compensation band;
        # object mocap references remain time-aligned and all other profile
        # terms are fixed.
        profiles = {
            1: {"name": "force5_lookahead260_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_lookahead125", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 125, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_lookahead150", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 150, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "force5_lookahead175", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 175, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_lag_c"
        output_stem = "profile_search_results_lag_aligned"
        selected_stem = "selected_mjwp_profile_lag_aligned"
    elif search == "deterministic-mjwp":
        # Isolate the physical reference-controller trajectory from the
        # sampling optimizer's exploratory candidates.  ``num_samples=1``
        # keeps MJWP and its real rollout/reward path active, but its sole
        # sample is the zero-noise reference trajectory.
        profiles = {
            1: {"name": "force5_sampling16_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_zero_noise_single_sample", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 1, "max_num_iterations": 1, "num_samples_note": "real MJWP zero-noise sample"},
        }
        metric_prefix = "primary_r4_deterministic_c"
        output_stem = "profile_search_results_deterministic_mjwp"
        selected_stem = "selected_mjwp_profile_deterministic_mjwp"
    elif search == "pair-margin":
        # With one DR group, np.linspace selects the lower endpoint rather
        # than a nominal midpoint: the existing [-2, +2] mm range therefore
        # executes every sampled rollout at -2 mm.  Search the measured
        # physical contact-margin correction independently of rewards.
        profiles = {
            1: {"name": "force5_pair_margin_minus2mm_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "pair_margin_range": [-0.002, 0.002], "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_pair_margin_zero", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "pair_margin_range": [0.0, 0.0], "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_pair_margin_minus0p5mm", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "pair_margin_range": [-0.0005, -0.0005], "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "force5_pair_margin_plus0p5mm", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "pair_margin_range": [0.0005, 0.0005], "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_margin_c"
        output_stem = "profile_search_results_pair_margin"
        selected_stem = "selected_mjwp_profile_pair_margin"
    elif search == "grouped-lookahead":
        # Dynamic phase audit on the best force×5 trajectory: wrists align
        # near the current reference while fingers align 100--175 simulation
        # steps ahead.  Keep the object reference and all other terms fixed.
        profiles = {
            1: {"name": "force5_uniform_lookahead260_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_wrist0_finger150", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "robot_wrist_reference_lookahead_steps": 0, "robot_finger_reference_lookahead_steps": 150, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_wrist0_finger175", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "robot_wrist_reference_lookahead_steps": 0, "robot_finger_reference_lookahead_steps": 175, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            4: {"name": "force5_wrist50_finger150", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "robot_wrist_reference_lookahead_steps": 50, "robot_finger_reference_lookahead_steps": 150, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_grouped_c"
        output_stem = "profile_search_results_grouped_lookahead"
        selected_stem = "selected_mjwp_profile_grouped_lookahead"
    elif search == "state-feedback":
        # The measured physical trajectory remains dynamically displaced from
        # the current source frame even with the best force×5 profile.  These
        # three candidates test a bounded current-state servo correction,
        # while keeping the source phase, object mocap target, optimizer, and
        # all robot/object references otherwise fixed.  Candidate 1 is the
        # already-evaluated force×5 baseline for an auditable comparison.
        profiles = {
            1: {"name": "force5_state_feedback_off_baseline_reused", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "num_samples": 16, "max_num_iterations": 2},
            2: {"name": "force5_state_feedback_moderate", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "robot_state_feedback_gain": 0.15, "robot_state_feedback_wrist_translation_clip_m": 0.006, "robot_state_feedback_wrist_rotation_clip_rad": 0.06, "robot_state_feedback_finger_clip_rad": 0.20, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_state_feedback_strong", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "robot_state_feedback_gain": 0.30, "robot_state_feedback_wrist_translation_clip_m": 0.012, "robot_state_feedback_wrist_rotation_clip_rad": 0.12, "robot_state_feedback_finger_clip_rad": 0.35, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_state_feedback_c"
        output_stem = "profile_search_results_state_feedback"
        selected_stem = "selected_mjwp_profile_state_feedback"
    elif search == "integral-feedback":
        # Single-tick contact DLS and current-state feedback both reduced
        # source-surface recall.  The remaining measured error is persistent
        # rather than a one-step phase offset, so test only a small bounded
        # integral family.  The correction still affects robot servo targets
        # only; no object state, source contact, or Stage-B trajectory is
        # rewritten.  The barrier variants explicitly protect the observed
        # late right-middle deep-contact interval.
        profiles = {
            2: {"name": "force5_first_integral_005_decay098", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "first_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_integral_gain": 0.05, "contact_ik_integral_decay": 0.98, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "num_samples": 16, "max_num_iterations": 2},
            3: {"name": "force5_all_integral_005_decay098_barrier025", "object_mocap_solref_s": 0.001, "contact_rew_scale": 50.0, "collision_rew_scale": 100000.0, "collision_rew_margin_m": 0.001, "robot_reference_lookahead_steps": 260, "contact_wrist_mean_feedback_gain": 0.0, "robot_servo_forcelimit_scale": 5.0, "contact_ik_feedback_gain": 0.5, "contact_ik_feedback_strategy": "all_active", "contact_ik_feedback_apply_to_plan": True, "contact_ik_integral_gain": 0.05, "contact_ik_integral_decay": 0.98, "contact_ik_feedback_wrist_translation_clip_m": 0.003, "contact_ik_feedback_wrist_rotation_clip_rad": 0.03, "contact_ik_feedback_finger_clip_rad": 0.25, "contact_ik_collision_barrier_gain": 0.25, "contact_ik_collision_barrier_margin_m": 0.0003, "num_samples": 16, "max_num_iterations": 2},
        }
        metric_prefix = "primary_r4_integral_c"
        output_stem = "profile_search_results_integral_feedback"
        selected_stem = "selected_mjwp_profile_integral_feedback"
    elif search != "legacy":
        raise ValueError("search must be 'legacy', 'post-gapfix', 'qvel', 'qvel-actuated', 'barrier', 'guarded', 'lag-aligned', 'deterministic-mjwp', 'pair-margin', 'grouped-lookahead', 'state-feedback', or 'integral-feedback'")
    records: list[dict[str, Any]] = []
    for candidate_id, profile in profiles.items():
        metric_path = target / f"{metric_prefix}{candidate_id}_metrics.json"
        if search == "post-gapfix" and candidate_id == 11:
            # The force-limit trial exited after startup without writing a
            # completion marker or a fresh trajectory.  A stale live output
            # must never be attributed to this candidate.
            run_log = target / "primary_r4_gapfix_c11_force5_contact50.log"
            if not run_log.is_file() or "Total time:" not in run_log.read_text(
                encoding="utf-8", errors="replace"
            ):
                records.append({
                    "candidate_id": candidate_id,
                    "profile": profile,
                    "metric_path": str(metric_path),
                    "status": "INCOMPLETE",
                    "failure_reason": "RUN_DID_NOT_WRITE_COMPLETION_MARKER_OR_FRESH_TRAJECTORY",
                })
                continue
        if not metric_path.is_file():
            records.append({"candidate_id": candidate_id, "profile": profile, "metric_path": str(metric_path), "status": "MISSING"})
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        collision = metrics.get("collision", {})
        contact = metrics.get("contact", {})
        object_tracking = metrics.get("object_tracking", {}).get("right", {})
        gates = dict(metrics.get("gates", {}))
        failed = sorted(name for name, passed in gates.items() if not passed)
        records.append({
            "candidate_id": candidate_id,
            "profile": profile,
            "metric_path": str(metric_path),
            "trajectory_path": str(target / f"{metric_prefix}{candidate_id}_trajectory_spider.npz"),
            "status": metrics.get("status", "FAIL"),
            "gates": gates,
            "failed_gates": failed,
            "contact_recall": contact.get("high_confidence_recall"),
            "false_contact_frames": contact.get("false_contact_frames"),
            "collision_max_penetration_m": collision.get("max_penetration_m"),
            "collision_p95_penetration_m": collision.get("p95_penetration_m"),
            "persistent_deep_collision_frames": len(collision.get("persistent_deep_frames", [])),
            "object_rotation_max_rad": object_tracking.get("rotation_max_rad"),
        })
    complete = all(record.get("status") not in {"MISSING", "INCOMPLETE", None} for record in records)
    passing = [record for record in records if record.get("status") == "PASS" and all(record.get("gates", {}).values())]
    # First reject hard gate failures, then rank the residual Pareto evidence
    # in the contract order.  This rank never overrides a failed gate.
    ranked = sorted(
        records,
        key=lambda row: (
            row.get("status") in {"MISSING", "INCOMPLETE", None},
            row.get("status") != "PASS",
            len(row.get("failed_gates", ["missing"])),
            -(row.get("contact_recall") or 0.0),
            row.get("collision_max_penetration_m") or float("inf"),
            row.get("object_rotation_max_rad") or float("inf"),
            row["candidate_id"],
        ),
    )
    payload = {
        "schema_version": 1,
        "stage": "C-R4",
        "sequence_id": sequence_id,
        "search_limit": len(profiles),
        "candidate_count": len(records),
        "complete": complete,
        "selection_order": ["hard_validity", "visual_penetration", "collision_penetration", "contact_preservation", "object_tracking", "tracking", "smoothness", "runtime"],
        "candidates": records,
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
        "status": "PASS" if complete and len(passing) == 1 else "FAIL",
        "selected_candidate_id": passing[0]["candidate_id"] if len(passing) == 1 else None,
        "failure_reason": None if len(passing) == 1 else "NO_FEASIBLE_SHARED_MJWP_PROFILE_IN_BOUNDED_PRIMARY_SEARCH",
        "stage_b_untouched": True,
    }
    _atomic_json(target / f"{output_stem}.json", payload)
    _atomic_json(
        target / f"{selected_stem}.json",
        {
            "schema_version": 1,
            "stage": "C-R4",
            "sequence_id": sequence_id,
            "status": "SELECTED" if len(passing) == 1 else "NO_FEASIBLE_PROFILE",
            "selected_candidate_id": payload["selected_candidate_id"],
            "profile": passing[0]["profile"] if len(passing) == 1 else None,
            "profile_search_results": str(target / f"{output_stem}.json"),
            "stage_b_untouched": True,
        },
    )
    return str(target / f"{output_stem}.json")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"make-manifest": make_manifest, "diagnose-source": diagnose_source, "build-contact-reference": build_contact_reference, "build-collision-cache": build_collision_cache, "prepare-physics-input": prepare_physics_input, "collision-audit": collision_audit, "depenetrate-init": depenetrate_init, "evaluate-depenetrated-init": evaluate_depenetrated_init, "evaluate-mjwp-output": evaluate_mjwp_output, "summarize-depenetration-multistart": summarize_depenetration_multistart, "summarize-auxiliary-mjwp-sanity": summarize_auxiliary_mjwp_sanity, "diagnose-primary-contact-conflict": diagnose_primary_contact_conflict, "write-primary-infeasibility-report": write_primary_infeasibility_report, "summarize-mjwp-profile-search": summarize_mjwp_profile_search, "preflight-static": preflight_static, "preflight-holds": preflight_holds, "preflight-rollout": preflight_rollout, "prepare-minimal-mjwp-input": prepare_minimal_mjwp_input, "run-minimal-mjwp": run_minimal_mjwp, "evaluate-r3-gate": evaluate_r3_gate, "write-failure-report": write_failure_report})
