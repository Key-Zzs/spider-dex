"""Separated visual/collision and scene-transform audits for Stage C.

The functions here intentionally do not change a MuJoCo model or an input
trajectory.  They inspect immutable Stage B states and the isolated Stage C
scene, returning JSON/NPZ-friendly evidence that can be used to decide whether
an initializer is justified.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def mesh_from_model(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> trimesh.Trimesh:
    """Return a world-space mesh for one MuJoCo mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(f"geom {geom_id} is not a mesh")
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64).copy()
    face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32).copy()
    if len(faces) and faces.max() >= count:
        faces -= start
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def region_for_geom(model: mujoco.MjModel, geom_id: int) -> tuple[str, str]:
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
    side = "right" if body_name.startswith("r_") else "left" if body_name.startswith("l_") else "unknown"
    lower = body_name.lower()
    region = next((finger for finger in FINGERS if finger in lower), "palm")
    return side, region


def bbox_report(visual: trimesh.Trimesh, collision: trimesh.Trimesh) -> dict[str, Any]:
    visual_extent = visual.bounds[1] - visual.bounds[0]
    collision_extent = collision.bounds[1] - collision.bounds[0]
    ratio = collision_extent / np.maximum(visual_extent, 1e-12)
    center = lambda mesh: (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    return {
        "visual_bbox": visual.bounds.tolist(),
        "collision_bbox": collision.bounds.tolist(),
        "extent_ratio": ratio.tolist(),
        "center_difference_m": float(np.linalg.norm(center(visual) - center(collision))),
        "within_extent_gate": bool(np.all((ratio >= 0.95) & (ratio <= 1.05))),
        "within_center_gate": bool(np.linalg.norm(center(visual) - center(collision)) <= 0.005),
        "visual": mesh_quality(visual),
        "collision": mesh_quality(collision),
    }


def mesh_quality(mesh: trimesh.Trimesh) -> dict[str, Any]:
    finite = bool(np.isfinite(mesh.vertices).all())
    degenerate = int(np.count_nonzero(mesh.area_faces <= 1e-14))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "finite_vertices": finite,
        "degenerate_faces": degenerate,
        "volume_m3": float(abs(mesh.volume)) if mesh.is_watertight else None,
    }


def empty_penetration_summary() -> dict[str, Any]:
    return {"samples": 0, "penetrating_count": 0, "penetrating_ratio": 0.0,
            "mean_negative_depth_m": 0.0, "p95_negative_depth_m": 0.0,
            "max_penetration_m": 0.0, "closest_surface_point_world": None,
            "contact_normal_world": None, "confidence": "not_observed"}


def summarize_signed_distances(
    signed: np.ndarray, closest: np.ndarray, normals: np.ndarray | None = None, confidence: str = "high"
) -> dict[str, Any]:
    signed = np.asarray(signed, dtype=np.float64)
    negative = signed[np.isfinite(signed) & (signed < 0)]
    output = empty_penetration_summary()
    output["samples"] = int(signed.size)
    output["penetrating_count"] = int(len(negative))
    output["penetrating_ratio"] = float(len(negative) / max(1, signed.size))
    output["confidence"] = confidence
    if not len(negative):
        return output
    output.update({
        "mean_negative_depth_m": float(-negative.mean()),
        "p95_negative_depth_m": float(np.percentile(-negative, 95)),
        "max_penetration_m": float(-negative.min()),
    })
    peak = np.nanargmin(signed)
    output["closest_surface_point_world"] = np.asarray(closest).reshape(-1, 3)[peak].tolist()
    if normals is not None:
        output["contact_normal_world"] = np.asarray(normals).reshape(-1, 3)[peak].tolist()
    return output


def contact_pair_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual MuJoCo contact records without inventing signed depth."""
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[record["geom_pair"]].append(record)
    result: dict[str, Any] = {}
    for pair, rows in sorted(by_pair.items()):
        depths = np.asarray([row["penetration_m"] for row in rows], dtype=np.float64)
        peak = rows[int(np.argmax(depths))]
        result[pair] = {
            "contact_count": len(rows),
            "frames": sorted(set(int(row["frame_index"]) for row in rows)),
            "mean_penetration_m": float(depths.mean()),
            "p95_penetration_m": float(np.percentile(depths, 95)),
            "max_penetration_m": float(depths.max()),
            "peak_position_world": peak["position_world"],
            "peak_normal_world": peak["normal_world"],
        }
    return result
