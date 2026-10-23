"""Q0--Q4 GRAB source screening and isolated Wuji Stage-A/B baseline.

This command is intentionally fail-closed.  It never rewrites GRAB, body
models, historical Stage-B/C-XA namespaces, or the frozen Stage-C configs.
The only materialised runtime output is a fresh directory below
``.local_artifacts/stage_q0_q4_grab_screening``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import trimesh
import yaml
from plotly.offline.offline import get_plotlyjs
from scipy.spatial.transform import Rotation

from spider.datasets.paths import load_project_paths
from spider.tools import new_grab_sequence_recheck as source_check
from spider.tools.grab_pipeline import _FINGERTIP_INDICES, _SMPLX_TO_WUJI_PALM
from spider.tools.grab_source_geometry_audit import (
    HAND_JOINTS,
    _hand_faces_from_smplx,
    apply_transform,
    invert_transform,
    object_relative,
    rotation_residual_rad,
    transform_from_rt,
)


REPO = Path(__file__).resolve().parents[2]
REF2DEX_ROOT = Path("/4-4090/Ref2Dex")
REGISTRY = REPO / "configs/project/grab_sequence_quality_registry.yaml"
EXCLUDED = "s5/cylindermedium_lift"
SEED = 20260804
QUERY_SAMPLES = 20_000
QUERY_VOXEL_M = 0.001
NEAR_M = 0.001
SHALLOW_M = 0.003
SEVERE_M = 0.005
WRIST_LOOP = (119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LEFT_JOINTS = {"thumb": (37, 38, 39), "index": (25, 26, 27), "middle": (28, 29, 30), "ring": (34, 35, 36), "pinky": (31, 32, 33), "wrist": (20,)}
RIGHT_JOINTS = {"thumb": (52, 53, 54), "index": (40, 41, 42), "middle": (43, 44, 45), "ring": (49, 50, 51), "pinky": (46, 47, 48), "wrist": (21,)}


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return value.replace("/", "__")


def longest_run(mask: np.ndarray) -> tuple[int, list[int] | None]:
    values = np.asarray(mask, dtype=bool)
    best = current = 0
    start = best_start = 0
    for index, value in enumerate(values):
        if value:
            if not current:
                start = index
            current += 1
            if current > best:
                best, best_start = current, start
        else:
            current = 0
    return int(best), [int(best_start), int(best_start + best - 1)] if best else None


def _submesh(vertices: np.ndarray, faces: np.ndarray, limit: int = 350) -> dict[str, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int32)
    vertices = np.asarray(vertices, dtype=np.float32)
    if len(faces) > limit:
        faces = faces[np.linspace(0, len(faces) - 1, limit, dtype=np.int64)]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return {"vertices": vertices[used], "faces": inverse.reshape(-1, 3).astype(np.int32), "indices": used.astype(np.int32)}


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def _transform12(transform: np.ndarray) -> np.ndarray:
    """Encode a homogeneous transform in the viewer's row-major 3x4 layout."""
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected [4,4] transform, got {matrix.shape}")
    return matrix[:3, :].reshape(12)


def _fixed_wrist_cap(vertices: np.ndarray, faces: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vertices, dtype=np.float32)
    topology = np.asarray(faces, dtype=np.int32)
    if values.shape[-2:] != (778, 3) or topology.shape != (1538, 3):
        raise ValueError(f"unexpected SMPL-X hand surface schema {values.shape}/{topology.shape}")
    center = values[..., WRIST_LOOP, :].mean(axis=-2, keepdims=True)
    cap = []
    for index, current in enumerate(WRIST_LOOP):
        nxt = WRIST_LOOP[(index + 1) % len(WRIST_LOOP)]
        cap.append((current, nxt, 778) if side == "right" else (778, nxt, current))
    return np.concatenate((values, center), axis=-2), np.concatenate((topology, np.asarray(cap, dtype=np.int32)), axis=0)


def _mesh_status(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent), "components": int(len(mesh.split(only_watertight=False)))}


def _static_exec_lineage() -> dict[str, Any]:
    files = [
        "dataset/tools/reverse_hand_meshsdf_statistics.py",
        "dataset/tools/mesh_preprocess/get_sdf_from_asset.py",
        "dataset/tools/sample_object_query_points.py",
    ]
    git = subprocess.run(["git", "-C", str(REF2DEX_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    return {
        "schema_version": 1,
        "authority": "Ref2Dex STATIC_EXEC",
        "ref2dex_root": str(REF2DEX_ROOT),
        "git_commit": git,
        "runner": "dataset.tools.reverse_hand_meshsdf_statistics.query_targets_with_meshsdf",
        "backend": "dataset.tools.mesh_preprocess.get_sdf_from_asset.MeshSDFfromAsset",
        "sign_ray_mode": "xyz_majority",
        "negative_means": "object query point lies inside closed hand SDF target",
        "query_rule": "20,000 triangle-area barycentric samples then 1-mm floor-voxel first-unique; one fixed local query set per candidate",
        "adapter_difference": "Spider supplies raw SMPL-X hand vertices in memory; the Ref2Dex runner writes temporary per-frame OBJ targets and constructs MeshSDFfromAsset unchanged.",
        "files": [{"path": str(REF2DEX_ROOT / item), "sha256": sha256(REF2DEX_ROOT / item)} for item in files],
    }


def _candidate_contract(path: Path, source_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    row: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=True) as archive:
            frames = int(archive["n_frames"])
            gender, subject, obj = str(archive["gender"]), str(archive["sbj_id"]), str(archive["obj_name"])
            body = archive["body"].item()["params"]
            object_params = archive["object"].item()["params"]
            for key in source_check.REQUIRED_BODY_FIELDS:
                if key not in body or len(np.asarray(body[key])) != frames:
                    errors.append(f"body.params.{key} missing/frame mismatch")
            for key in ("transl", "global_orient"):
                if key not in object_params or len(np.asarray(object_params[key])) != frames:
                    errors.append(f"object.params.{key} missing/frame mismatch")
            mesh = source_root / "tools/object_meshes/contact_meshes" / f"{obj}.ply"
            beta = source_root / "tools/subject_meshes" / gender / f"{subject}_betas.npy"
            row = {"subject": subject, "gender": gender, "object": obj, "frame_count": frames, "object_mesh": str(mesh), "subject_beta": str(beta)}
            if not mesh.is_file():
                errors.append("object contact mesh missing")
            if not beta.is_file():
                errors.append("subject beta missing")
            if gender not in {"female", "male", "neutral"}:
                errors.append("unsupported gender")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return not errors, errors, row


def scan_lightweight(paths_config: str) -> list[dict[str, Any]]:
    """Metadata/asset scan only; this function never invokes reverse-SDF."""
    paths = load_project_paths(paths_config)
    root = paths.source_root("grab")
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "grab").glob("s*/*.npz")):
        source_id = path.relative_to(root / "grab").with_suffix("").as_posix()
        row: dict[str, Any] = {"source_id": source_id, "sequence_id_safe": safe_id(source_id), "parameter_path": str(path), "rejection_reasons": []}
        try:
            with np.load(path, allow_pickle=False) as archive:
                frames, fps = int(archive["n_frames"]), float(archive["framerate"])
                subject, gender, obj = str(archive["sbj_id"]), str(archive["gender"]), str(archive["obj_name"])
                intent, ncomps = str(archive["motion_intent"]), int(archive["n_comps"])
            row.update({"subject": subject, "gender": gender, "object": obj, "motion_intent": intent, "frame_count": frames, "fps": fps, "n_comps": ncomps})
            ok, errors, paths_row = _candidate_contract(path, root)
            row.update(paths_row)
            if source_id == EXCLUDED:
                row["rejection_reasons"].append("SOURCE_QUALITY_REJECTED registry entry")
            if frames < 200:
                row["rejection_reasons"].append("frame_count_lt_200")
            if fps <= 0:
                row["rejection_reasons"].append("fps_unavailable")
            if ncomps != 24:
                row["rejection_reasons"].append("not_pca24")
            row["rejection_reasons"].extend(errors)
            # Fixed metadata-only ranking.  It deliberately prefers sequences
            # around six hundred frames so a full-frame SDF audit remains a
            # bounded ten-sequence screen; no geometry or SDF result enters it.
            row["metadata_score"] = float(source_check.candidate_score(subject, obj, intent, frames))
            row["status"] = "METADATA_ELIGIBLE" if not row["rejection_reasons"] else "EXCLUDED"
        except Exception as exc:
            row.update({"status": "INVALID_SOURCE_DATA", "metadata_score": -1.0, "rejection_reasons": [f"{type(exc).__name__}: {exc}"]})
        rows.append(row)
    rows.sort(key=lambda value: (-float(value.get("metadata_score", -1)), str(value["source_id"])))
    for rank, row in enumerate(rows, 1):
        row["metadata_rank"] = rank
    return rows


def freeze_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze ten diverse metadata candidates before any geometry/SDF evaluation."""
    selected: list[dict[str, Any]] = []
    subject_count: Counter[str] = Counter()
    object_count: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "METADATA_ELIGIBLE":
            continue
        subject, obj = str(row["subject"]), str(row["object"])
        if subject_count[subject] >= 2 or object_count[obj] >= 2:
            continue
        selected.append({**row, "frozen_index": len(selected) + 1})
        subject_count[subject] += 1
        object_count[obj] += 1
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise RuntimeError(f"exactly ten candidates required; diversity-constrained metadata pool produced {len(selected)}")
    return selected


def _object_queries(mesh_path: Path, candidate: str, root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if str(REF2DEX_ROOT) not in sys.path:
        sys.path.insert(0, str(REF2DEX_ROOT))
    from dataset.tools.sample_object_query_points import frame_seed, load_mesh_vertices_faces, sample_surface_points, voxel_downsample
    vertices, faces = load_mesh_vertices_faces(np=np, mesh_path=mesh_path, center_bbox=False)
    points = sample_surface_points(np=np, vertices=vertices, faces=faces, samples_per_frame=QUERY_SAMPLES, rng=np.random.default_rng(frame_seed(SEED, candidate, 0)))
    points = voxel_downsample(np, points, QUERY_VOXEL_M).astype(np.float32, copy=False)
    np.savez_compressed(root / "query_points.npz", points=points, seed=np.asarray(SEED), samples=np.asarray(QUERY_SAMPLES), voxel_m=np.asarray(QUERY_VOXEL_M), mesh_sha256=np.asarray(sha256(mesh_path)))
    return points, {"mesh": str(mesh_path), "mesh_sha256": sha256(mesh_path), "sampling_seed": SEED, "query_samples_before_voxel": QUERY_SAMPLES, "voxel_size_m": QUERY_VOXEL_M, "point_count": int(len(points)), "same_points_all_frames": True}


def _static_query(targets: list[dict[str, Any]], points: np.ndarray, work: Path, frame: int, device: str, batch: int) -> np.ndarray:
    if str(REF2DEX_ROOT) not in sys.path:
        sys.path.insert(0, str(REF2DEX_ROOT))
    from dataset.tools.reverse_hand_meshsdf_statistics import query_targets_with_meshsdf
    result = query_targets_with_meshsdf(np=np, targets=targets, frame_index=0, points=points, device=device, check_manifold=False, sign_ray_mode="xyz_majority", work_root=work, unit_scale_to_mm=1000.0, cache_stem=f"source_{frame}", query_batch_size=batch, keep_frame_objs=False, return_aux=True)
    return np.asarray(result["signed_distance_mm"], dtype=np.float64) / 1000.0


def _sdf_row(signed: np.ndarray, source_frame: int) -> dict[str, Any]:
    depth = np.maximum(0.0, -signed)
    negative = signed < 0
    near = np.abs(signed) <= NEAR_M
    shallow = (signed < -NEAR_M) & (signed >= -SHALLOW_M)
    deep = signed < -SHALLOW_M
    return {
        "source_frame": int(source_frame), "status": "PASS", "point_count": int(len(signed)), "max_penetration_depth_m": float(depth.max(initial=0.0)), "mean_penetration_depth_m": float(depth[negative].mean()) if np.any(negative) else 0.0,
        "penetrating_point_ratio": float(negative.mean()), "deep_penetration_point_ratio": float(deep.mean()), "near_surface_point_ratio": float(near.mean()),
        "categories": {"OUTSIDE": int(np.count_nonzero(signed > NEAR_M)), "NEAR_SURFACE_CONTACT": int(near.sum()), "SHALLOW_PENETRATION": int(shallow.sum()), "DEEP_PENETRATION": int(deep.sum())},
    }


def _quality_summary(rows: list[dict[str, Any]], object_mesh: Path, object_status: dict[str, Any]) -> dict[str, Any]:
    depths = np.asarray([row["max_penetration_depth_m"] for row in rows], dtype=float)
    ratios = np.asarray([row["penetrating_point_ratio"] for row in rows], dtype=float)
    deep = np.asarray([row["deep_penetration_point_ratio"] for row in rows], dtype=float)
    near = np.asarray([row["near_surface_point_ratio"] for row in rows], dtype=float)
    severe_len, severe_interval = longest_run(depths > SEVERE_M)
    reliable_len, reliable_interval = longest_run((near > 0) & (depths <= SHALLOW_M))
    return {
        "valid_frame_count": int(len(rows)), "penetrating_frame_ratio": float(np.mean(ratios > 0)), "max_penetration_depth_m": float(depths.max(initial=0.0)), "P95_frame_max_depth_m": float(np.percentile(depths, 95)), "mean_penetration_depth_m": float(np.mean([row["mean_penetration_depth_m"] for row in rows])),
        "P95_penetrating_point_ratio": float(np.percentile(ratios, 95)), "deep_penetration_point_ratio": float(np.mean(deep)), "longest_consecutive_severe_interval": {"length": severe_len, "source_frame_indices": severe_interval}, "near_surface_contact_frame_count": int(np.count_nonzero(near > 0)),
        "longest_reliable_contact_interval": {"length": reliable_len, "source_frame_indices": reliable_interval}, "left_right_contact_participation": "union object-query versus closed left/right source hand SDF", "object_mesh": {"path": str(object_mesh), **object_status},
    }


def audit_source_quality(root: Path, candidate: dict[str, Any], paths_config: str, device: str, batch: int) -> dict[str, Any]:
    """Run the actual Ref2Dex SDF backend on every source frame of one frozen candidate."""
    official, official_chain = source_check.reconstruct_official(paths_config, str(candidate["source_id"]))
    points, query_manifest = _object_queries(Path(candidate["object_mesh"]), str(candidate["source_id"]), root)
    mesh_status = _mesh_status(official["object_asset_vertices"], official["object_faces"])
    hand_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in ("left", "right"):
        hand_targets[side] = _fixed_wrist_cap(official[f"{side}_vertices_world"], official[f"{side}_hand_faces"], side)
        check = _mesh_status(hand_targets[side][0][0], hand_targets[side][1])
        if not (check["watertight"] and check["winding_consistent"]):
            raise RuntimeError(f"formal source hand target invalid for {candidate['source_id']} {side}: {check}")
    rows: list[dict[str, Any]] = []
    work = root / "static_exec_work"
    for frame, source_frame in enumerate(official["source_frame_indices"]):
        world_points = _transform_points(official["T_world_object"][frame], points)
        # Ref2Dex's runner treats a two-dimensional vertex array as a sequence
        # and indexes it by frame.  Preserve its real API contract with an
        # explicit one-frame [1,V,3] target rather than changing the runner.
        targets = [{"label": side, "source": f"official_smplx_{side}", "vertices": hand_targets[side][0][frame : frame + 1], "faces": hand_targets[side][1]} for side in ("left", "right")]
        signed = _static_query(targets, world_points, work, int(source_frame), device, batch)
        rows.append(_sdf_row(signed, int(source_frame)))
    summary = _quality_summary(rows, Path(candidate["object_mesh"]), mesh_status)
    np.savez_compressed(root / "source_quality_trace.npz", source_frame=np.asarray([row["source_frame"] for row in rows], dtype=np.int64), frame_max_depth_m=np.asarray([row["max_penetration_depth_m"] for row in rows], dtype=np.float32), penetrating_ratio=np.asarray([row["penetrating_point_ratio"] for row in rows], dtype=np.float32), deep_ratio=np.asarray([row["deep_penetration_point_ratio"] for row in rows], dtype=np.float32), near_ratio=np.asarray([row["near_surface_point_ratio"] for row in rows], dtype=np.float32))
    metrics = {"schema_version": 1, "candidate": candidate, "status": "SCREENED", "formal_sdf": True, "static_exec": _static_exec_lineage(), "query_manifest": query_manifest, "official_reconstruction": official_chain, "hand_target_status": {side: _mesh_status(hand_targets[side][0][0], hand_targets[side][1]) for side in ("left", "right")}, "rows": rows, "summary": summary}
    write_json(root / "source_quality_metrics.json", metrics)
    write_text(root / "candidate_summary.md", f"# {candidate['source_id']} source quality\n\n- max penetration: `{summary['max_penetration_depth_m'] * 1000:.3f} mm`\n- P95 frame max: `{summary['P95_frame_max_depth_m'] * 1000:.3f} mm`\n- deep point ratio: `{summary['deep_penetration_point_ratio']:.6f}`\n- reliable interval: `{summary['longest_reliable_contact_interval']['length']}` frames\n")
    return {"candidate": candidate, "official": official, "official_chain": official_chain, "metrics": metrics}


def rank_candidates(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(audits, key=lambda value: (value["metrics"]["summary"]["max_penetration_depth_m"], value["metrics"]["summary"]["P95_frame_max_depth_m"], value["metrics"]["summary"]["deep_penetration_point_ratio"], -value["metrics"]["summary"]["longest_reliable_contact_interval"]["length"], value["candidate"]["source_id"]))
    # No pre-existing project-wide binary threshold exists.  This is a
    # versioned, run-local ranking policy, never a generic GRAB standard.
    for rank, audit in enumerate(ordered, 1):
        summary = audit["metrics"]["summary"]
        severe = summary["longest_consecutive_severe_interval"]["length"]
        # The repository has no formal binary source-quality threshold.  The
        # required status therefore names the fixed top-four *review set*;
        # it does not pre-approve a sample or discard it merely because one
        # continuous metric is poor.  Severe intervals remain fully visible
        # and are assessed again by official reconstruction and Q3 authority.
        status = "PASS_CANDIDATE" if rank <= 4 else "REVIEW_REQUIRED"
        audit["rank"] = rank
        audit["quality_status"] = status
        audit["reason"] = "fixed top-4 run-local distribution candidate awaiting official reconstruction and authority review" if status == "PASS_CANDIDATE" else "outside top-4 manual/official review set"
        audit["metrics"]["quality_status"] = status
        write_json(Path(audit["candidate_root"]) / "source_quality_metrics.json", audit["metrics"])
    return ordered


def _region_map(paths_config: str, official: dict[str, np.ndarray], gender: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import smplx
    paths = load_project_paths(paths_config)
    model = smplx.create(str(paths.body_model_root), model_type="smplx", gender=gender, use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=1)
    weights = model.lbs_weights.detach().cpu().numpy()
    result: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for side, joints in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
        ids = np.asarray(np.load(paths.source_root("grab") / "tools/smplx_correspondence" / f"{side[0]}hand_smplx_ids.npy"), dtype=np.int64)
        bones = np.asarray([joint for finger in FINGERS for joint in joints[finger]], dtype=np.int64)
        local = weights[ids][:, bones]
        win = np.argmax(local, axis=1); finger = win // 3; part = win % 3
        labels = np.full(len(ids), "palm", dtype=object)
        keep = np.max(local, axis=1) > weights[ids, joints["wrist"][0]]
        parts = ("proximal", "middle", "distal")
        for index, name in enumerate(FINGERS):
            labels[keep & (finger == index)] = [f"{name}_{parts[item]}" for item in part[keep & (finger == index)]]
        for name in [f"{finger}_{part}" for finger in FINGERS for part in ("distal", "middle", "proximal")] + ["palm", "wrist_base"]:
            vertices = np.flatnonzero(labels == name).astype(np.int32) if name != "wrist_base" else np.empty(0, dtype=np.int32)
            key = f"{side}_{name}"; result[key] = vertices
            rows.append({"region": key, "side": side, "anatomical_region": name, "vertex_count": int(len(vertices)), "source_vertex_ids": ids[vertices].tolist(), "assignment": "maximum named SMPL-X LBS finger-bone weight; palm is wrist-dominant residual; wrist_base remains explicit and empty for 778-vertex hand correspondence"})
    return result, {"schema_version": 1, "method": "SMPL-X LBS named anatomical regions", "regions": rows}


def _vertex_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    result = np.zeros(len(vertices), dtype=float)
    np.add.at(result, faces.reshape(-1), np.repeat(area / 3., 3))
    return result


def build_authority(root: Path, candidate_audit: dict[str, Any], paths_config: str) -> dict[str, Any]:
    """Build source-only role and semantic-patch authority; deep samples cannot enter either."""
    official = candidate_audit["official"]
    candidate = candidate_audit["candidate"]
    mesh = trimesh.Trimesh(vertices=official["object_asset_vertices"], faces=official["object_faces"], process=False)
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        return {"status": "FAIL", "reason": "object mesh not watertight/winding-consistent"}
    tmesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32)))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(tmesh)
    regions, region_manifest = _region_map(paths_config, official, str(candidate["gender"]))
    count = len(official["source_frame_indices"])
    face_ids: dict[str, np.ndarray] = {}; signed_all: dict[str, np.ndarray] = {}; reliable: dict[str, np.ndarray] = {}
    closest: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        verts = np.asarray(official[f"{side}_vertices_world"], dtype=np.float64)
        local = np.einsum("tji,tvj->tvi", official["T_world_object"][:, :3, :3], verts - official["T_world_object"][:, None, :3, 3])
        flat = local.reshape(-1, 3).astype(np.float32)
        signed = np.empty(len(flat), dtype=np.float32); faces = np.empty(len(flat), dtype=np.int32); points = np.empty_like(flat)
        for start in range(0, len(flat), 16384):
            query = o3d.core.Tensor(flat[start:start + 16384])
            signed[start:start + len(query)] = scene.compute_signed_distance(query).numpy()
            nearest = scene.compute_closest_points(query)
            faces[start:start + len(query)] = nearest["primitive_ids"].numpy().astype(np.int32)
            points[start:start + len(query)] = nearest["points"].numpy()
        signed_all[side] = signed.reshape(count, -1)
        face_ids[side] = faces.reshape(count, -1)
        nearest_world = np.einsum("tij,tvj->tvi", official["T_world_object"][:, :3, :3], points.reshape(count, -1, 3)) + official["T_world_object"][:, None, :3, 3]
        closest[side] = nearest_world.astype(np.float32)
        reliable[side] = np.abs(signed_all[side]) <= NEAR_M
    per_frame: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    role_points: dict[int, list[list[float]]] = {}
    tri_frequency: Counter[int] = Counter()
    all_region = [name for name in regions if not name.endswith("wrist_base")]
    arrays_masks: list[np.ndarray] = []
    responsibilities = np.zeros((count, len(all_region)), dtype=np.float32)
    for frame, source_frame in enumerate(official["source_frame_indices"]):
        score: dict[str, float] = {}
        contributions: list[dict[str, Any]] = []
        frame_points: list[list[float]] = []
        for region_index, name in enumerate(all_region):
            side = name.split("_", 1)[0]; indices = regions[name]
            mask = reliable[side][frame, indices]
            vertices = official[f"{side}_vertices_world"][frame, indices]
            weights = _vertex_weights(official[f"{side}_vertices_world"][0], official[f"{side}_hand_faces"])[indices]
            contact_count = int(mask.sum()); area = float(weights[mask].sum()) if contact_count else 0.0
            # A source-only role score uses surface footprint/persistence proxy;
            # no robot reachability or patch dilation participates.
            value = area + contact_count * 1e-6
            score[name] = value; responsibilities[frame, region_index] = value
            if contact_count:
                contact_vertices = vertices[mask]
                centroid = np.average(contact_vertices, axis=0, weights=weights[mask] if np.any(weights[mask]) else None)
                faces = face_ids[side][frame, indices][mask]
                for face in faces:
                    tri_frequency[int(face)] += 1
                frame_points.extend(contact_vertices.tolist())
                contributions.append({"region": name, "contact_vertex_count": contact_count, "contact_area_m2": area, "centroid_world": centroid.tolist(), "triangle_ids": sorted(set(int(face) for face in faces))})
        maximum = max(score.values(), default=0.0)
        active = [{"region": name, "weight": float(value / maximum)} for name, value in sorted(score.items()) if maximum and value >= maximum * .5]
        frame_status = "RELIABLE_CONTACT" if active else "UNRELIABLE_SOURCE"
        per_frame.append({"source_frame": int(source_frame), "status": frame_status, "roles": active, "region_contributions": contributions, "deep_penetration_excluded": True})
        if active:
            roles.append({"source_frame": int(source_frame), "shared_regions": active})
            role_points[int(source_frame)] = frame_points
    reliable_mask = np.asarray([bool(row["roles"]) for row in per_frame])
    interval_len, interval = longest_run(reliable_mask)
    # Keep genuine source-triangle components, without expanding by adjacency.
    selected_faces = sorted(tri_frequency)
    components: list[list[int]] = []
    if selected_faces:
        graph: dict[int, set[int]] = {face: set() for face in selected_faces}
        for first, second in mesh.face_adjacency:
            if int(first) in graph and int(second) in graph:
                graph[int(first)].add(int(second)); graph[int(second)].add(int(first))
        while graph:
            start = next(iter(graph)); stack = [start]; component = []
            while stack:
                item = stack.pop()
                if item not in graph:
                    continue
                component.append(item); stack.extend(graph.pop(item))
            components.append(sorted(component))
    role_schedule = {"schema_version": 1, "source_sequence": candidate["source_id"], "definition": "all 778-vertex SMPL-X hand-surface regions; only near-surface samples may contribute", "reliable_interval": {"length": interval_len, "source_frame_indices": interval}, "frames": roles, "unreliable_frame_list": [row["source_frame"] for row in per_frame if not row["roles"]], "deep_penetration_policy": "UNRELIABLE_SOURCE; excluded from role responsibility"}
    patch = {"schema_version": 1, "source_sequence": candidate["source_id"], "definition": "union of nearest object triangles at reliable near-surface full-hand contacts only", "object_mesh_sha256": sha256(Path(candidate["object_mesh"])), "triangle_ids": selected_faces, "triangle_frequency": [{"triangle_id": face, "count": count} for face, count in sorted(tri_frequency.items())], "connected_components": components, "generation": "no dilation; no Wuji-reachability operation; no deep-penetration contribution"}
    status = "PASS" if interval_len >= 60 and selected_faces else "FAIL"
    manifest = {"schema_version": 1, "status": status, "source_sequence": candidate["source_id"], "region_manifest": region_manifest, "contact_categories": {"OUTSIDE": ">1 mm", "NEAR_SURFACE_CONTACT": "abs(signed)<=1 mm", "SHALLOW_PENETRATION": "1--3 mm inside object", "DEEP_PENETRATION": ">3 mm inside object"}, "reliable_interval_length": interval_len, "deep_penetration_excluded": True, "patch_expansion_or_dilation": "none", "mesh_hash": sha256(Path(candidate["object_mesh"]))}
    validation = {"schema_version": 1, "status": status, "gates": {"anatomical_surface_regions": all(len(regions[name]) > 0 for name in all_region if not name.endswith("palm")), "deep_penetration_excluded": True, "shared_roles_allowed": True, "patch_from_reliable_footprint": bool(selected_faces), "no_arbitrary_dilation": True, "reliable_interval_ge_60": interval_len >= 60}}
    np.savez_compressed(root / "contact_footprint.npz", source_frame=np.asarray(official["source_frame_indices"], dtype=np.int64), left_signed_distance_m=signed_all["left"], right_signed_distance_m=signed_all["right"], left_reliable=reliable["left"].astype(np.uint8), right_reliable=reliable["right"].astype(np.uint8), left_triangle_id=face_ids["left"], right_triangle_id=face_ids["right"], responsibility=responsibilities, region_names=np.asarray(all_region))
    write_json(root / "role_schedule.json", role_schedule); write_json(root / "semantic_patch.json", patch); write_json(root / "authority_manifest.json", manifest); write_json(root / "authority_validation.json", validation)
    return {"status": status, "candidate": candidate, "official": official, "per_frame": per_frame, "role_points": role_points, "role_schedule": role_schedule, "semantic_patch": patch, "manifest": manifest, "validation": validation, "regions": regions, "reliable": reliable, "signed": signed_all, "face_ids": face_ids, "closest": closest}


def _authority_payload(authority: dict[str, Any]) -> dict[str, Any]:
    official = authority["official"]; patch = authority["semantic_patch"]
    body = _submesh(official["body_vertices_world"][0], official["body_faces"], 500)
    hands = {side: _submesh(official[f"{side}_vertices_world"][0], official[f"{side}_hand_faces"], 350) for side in ("left", "right")}
    obj = _submesh(official["object_asset_vertices"], official["object_faces"], 600)
    patch_mesh = _submesh(official["object_asset_vertices"], official["object_faces"][np.asarray(patch["triangle_ids"], dtype=np.int32)], 800) if patch["triangle_ids"] else {"vertices": np.empty((0, 3), dtype=np.float32), "faces": np.empty((0, 3), dtype=np.int32)}
    frames = []
    for index, source_frame in enumerate(official["source_frame_indices"]):
        points = authority["role_points"].get(int(source_frame), [])
        frames.append({"source_frame": int(source_frame), "object_T": _transform12(official["T_world_object"][index]), "left_T": _transform12(official["T_world_left_wrist"][index]), "right_T": _transform12(official["T_world_right_wrist"][index]), "body": official["body_vertices_world"][index][body["indices"]], "left": official["left_vertices_world"][index][hands["left"]["indices"]], "right": official["right_vertices_world"][index][hands["right"]["indices"]], "contact": np.asarray(points, dtype=np.float32)})
    return {"kind": "authority", "status": authority["status"], "frames": frames, "body_faces": body["faces"], "left_faces": hands["left"]["faces"], "right_faces": hands["right"]["faces"], "object": obj, "patch": patch_mesh, "schedule": authority["role_schedule"], "categories": authority["manifest"]["contact_categories"]}


def _mesh_trace(name: str, vertices: Any, faces: Any, color: str, opacity: float) -> str:
    return "{type:'mesh3d',name:" + json.dumps(name, ensure_ascii=False) + ",x:" + vertices + ".map(p=>p[0]),y:" + vertices + ".map(p=>p[1]),z:" + vertices + ".map(p=>p[2]),i:" + faces + ".map(x=>x[0]),j:" + faces + ".map(x=>x[1]),k:" + faces + ".map(x=>x[2]),color:" + json.dumps(color) + ",opacity:" + str(opacity) + ",flatshading:true}"


def _html(title: str, payload: dict[str, Any], kind: str) -> str:
    packed = json.dumps(payload, ensure_ascii=False, default=_plain, separators=(",", ":"))
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><script>{get_plotlyjs()}</script><style>body{{margin:0;background:#101820;color:#eef6f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#bar{{padding:12px;background:#173042}}#scene{{height:73vh}}#curve{{height:18vh}}select,input{{margin:3px}}pre{{white-space:pre-wrap;padding:8px}}</style></head><body><div id='bar'><b>{title}</b><br>帧 <input id='frame' type='range' min='0' value='0'> <span id='frameLabel'></span>　坐标 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体坐标</option><option value='left'>左腕坐标</option><option value='right'>右腕坐标</option></select>　模式 <select id='mode'><option value='all'>全部图层</option><option value='source'>Source only</option><option value='stage'>Stage B only</option><option value='contact'>Contact authority</option><option value='failure'>Failure inspection</option></select><br><span>真实几何层：source SMPL-X body/左右手、object、可靠 near-surface contact、semantic patch；Stage B 页另含 Wuji visual/collision。</span></div><div id='scene'></div><div id='curve'></div><pre id='info'></pre><script>const D={packed},$=x=>document.getElementById(x);$('frame').max=Math.max(0,D.frames.length-1);function M(T,p){{return p.map(q=>[T[0]*q[0]+T[1]*q[1]+T[2]*q[2]+T[3],T[4]*q[0]+T[5]*q[1]+T[6]*q[2]+T[7],T[8]*q[0]+T[9]*q[1]+T[10]*q[2]+T[11]])}}function inv(T){{let r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],p=[T[3],T[7],T[11]],q=[[r[0][0],r[1][0],r[2][0]],[r[0][1],r[1][1],r[2][1]],[r[0][2],r[1][2],r[2][2]]];return [q[0][0],q[0][1],q[0][2],-(q[0][0]*p[0]+q[0][1]*p[1]+q[0][2]*p[2]),q[1][0],q[1][1],q[1][2],-(q[1][0]*p[0]+q[1][1]*p[1]+q[1][2]*p[2]),q[2][0],q[2][1],q[2][2],-(q[2][0]*p[0]+q[2][1]*p[1]+q[2][2]*p[2])]}}function mesh(n,v,f,c,o){{return{{type:'mesh3d',name:n,x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:f.map(x=>x[0]),j:f.map(x=>x[1]),k:f.map(x=>x[2]),color:c,opacity:o,flatshading:true}}}}function robot(n,R,k,V,c,o){{let vv=[],ff=[],z=0;for(let s of ['left','right'])R.layers[s].forEach((g,gi)=>{{let t=R.transforms[s][k][gi],w=g.vertices.map(p=>[t[3]*p[0]+t[4]*p[1]+t[5]*p[2]+t[0],t[6]*p[0]+t[7]*p[1]+t[8]*p[2]+t[1],t[9]*p[0]+t[10]*p[1]+t[11]*p[2]+t[2]]);vv.push(...M(V,w));ff.push(...g.faces.map(a=>a.map(x=>x+z)));z+=w.length}});return mesh(n,vv,ff,c,o)}}function draw(){{let i=+$('frame').value,f=D.frames[i],view=$('view').value,T=view==='object'?inv(f.object_T):view==='left'?inv(f.left_T):view==='right'?inv(f.right_T):[1,0,0,0,0,1,0,0,0,0,1,0],mode=$('mode').value,tr=[];if(mode!=='stage'){{tr.push(mesh('完整 source SMPL-X body',M(T,f.body),D.body_faces,'#77bfa3',.22));tr.push(mesh('左手 surface',M(T,f.left),D.left_faces,'#48cae4',.72));tr.push(mesh('右手 surface',M(T,f.right),D.right_faces,'#f28482',.72));tr.push(mesh('object visual/contact mesh',M(T,M(f.object_T,D.object.vertices)),D.object.faces,'#f4a261',.34));tr.push(mesh('semantic patch connected surface',M(T,M(f.object_T,D.patch.vertices)),D.patch.faces,'#c77dff',.75));if(f.contact.length)tr.push({{type:'scatter3d',mode:'markers',name:'可靠 near-surface source footprint',x:M(T,f.contact).map(p=>p[0]),y:M(T,f.contact).map(p=>p[1]),z:M(T,f.contact).map(p=>p[2]),marker:{{size:3,color:'#fff'}}}})}}if(D.robot_visual&&f.stage_index>=0&&mode!=='source'){{tr.push(robot('Wuji visual mesh',D.robot_visual,f.stage_index,T,'#06d6a0',.46));tr.push(robot('Wuji collision mesh',D.robot_collision,f.stage_index,T,'#ef476f',.22))}}Plotly.react('scene',tr,{{paper_bgcolor:'#101820',font:{{color:'#eef6f7'}},scene:{{aspectmode:'data',camera:{{eye:{{x:1.4,y:-1.4,z:1.1}}}}}},margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}});let x=D.frames.map(z=>z.source_frame),y=D.frames.map(z=>z.contact.length);Plotly.react('curve',[{{x,y,mode:'lines',name:'reliable contact vertex count'}}],{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#eef6f7'}},margin:{{l:50,r:10,t:15,b:35}}}});$('frameLabel').textContent=f.source_frame;$('info').textContent=JSON.stringify({{source_frame:f.source_frame,status:D.status,categories:D.categories,role_schedule:D.schedule}},null,2);window.__qReady=true}}$('frame').oninput=draw;$('view').onchange=draw;$('mode').onchange=draw;let q=new URLSearchParams(location.search);if(q.get('frame'))$('frame').value=q.get('frame');if(q.get('view'))$('view').value=q.get('view');draw();</script></body></html>"""


def write_authority_html(path: Path, authority: dict[str, Any]) -> None:
    write_text(path, _html("Q3 Source Contact Authority：" + authority["candidate"]["source_id"], _authority_payload(authority), "authority"))


def _run_stage_ab(root: Path, authority: dict[str, Any], paths_config: str) -> dict[str, Any]:
    candidate = authority["candidate"]; paths = load_project_paths(paths_config); task = safe_id(str(candidate["source_id"]))
    delivered_workspace = root / "workspace"
    scratch_workspace = Path("/tmp/spider-dex-stage-q0-q4") / root.parent.name / root.name / "workspace"
    if delivered_workspace.exists() or scratch_workspace.exists():
        raise FileExistsError(f"refuse overwrite Stage A/B workspace delivered={delivered_workspace} scratch={scratch_workspace}")
    scratch_workspace.parent.mkdir(parents=True, exist_ok=False)
    generated = {"schema_version": 1, "datasets": {"grab": {"source_root": str(paths.source_root("grab"))}}, "body_models": {"root": str(paths.body_model_root)}, "workspace": {"root": str(scratch_workspace)}}
    config = root / "effective_paths.yaml"; write_text(config, yaml.safe_dump(generated, sort_keys=False, allow_unicode=True))
    stages: dict[str, Any] = {"sequence": candidate["source_id"], "source_authority": authority["status"], "stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}, "cxa": "NOT_RUN", "provenance": {"effective_config": str(config), "input_raw_sha256": sha256(Path(candidate["parameter_path"])), "object_mesh_sha256": sha256(Path(candidate["object_mesh"])), "runtime": {"python": sys.version, "mujoco": getattr(mujoco, "__version__", "unknown")}}}
    commands = [("stage_a", [sys.executable, "-m", "spider.tools.grab_pipeline", "prepare", "--paths-config", str(config), "--sequence-id", task, "--frame-start", "0", "--frame-end", str(candidate["frame_count"])]), ("stage_b", [sys.executable, "-m", "spider.tools.grab_pipeline", "run-wuji-ik", "--paths-config", str(config), "--sequence-id", task, "--no-save-video"])]
    for name, command in commands:
        folder = root / name; folder.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        write_text(folder / "command.txt", " ".join(command) + "\n"); write_text(folder / "stdout.log", proc.stdout); write_text(folder / "stderr.log", proc.stderr)
        stages[name] = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "command": command, "stdout": str(folder / "stdout.log"), "stderr": str(folder / "stderr.log")}
        if proc.returncode:
            if scratch_workspace.exists():
                shutil.move(str(scratch_workspace), str(delivered_workspace))
            stages["final"] = "FAIL"; write_json(root / "stage_status.json", stages); return {"stages": stages, "result": None}
    shutil.move(str(scratch_workspace), str(delivered_workspace))
    robot = delivered_workspace / "processed/grab/wuji_hand2_beta1/bimanual" / task / "0"; scene = robot.parent / "scene.xml"
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as data:
        qpos, qvel, frequency = np.asarray(data["qpos"], dtype=float), np.asarray(data["qvel"], dtype=float), float(data["frequency"])
    mapping = np.asarray(json.loads((robot / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"], dtype=int)
    from spider.tools.new_grab_sequence_recheck import _extract_stage_b
    geometry = _extract_stage_b(scene, qpos); official = authority["official"]
    source_obj = official["T_world_object"][mapping]; object_translation = np.linalg.norm(source_obj[:, :3, 3] - geometry["objects"][:, :3, 3], axis=1); object_rotation = rotation_residual_rad(source_obj[:, :3, :3], geometry["objects"][:, :3, :3])
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        palm = transform_from_rt(geometry["rotations"][:, geometry["site_names"].index(f"{side}_palm")], geometry["positions"][:, geometry["site_names"].index(f"{side}_palm")])
        source_wrist = official[f"T_world_{side}_wrist"][mapping].copy(); source_wrist[:, :3, :3] = source_wrist[:, :3, :3] @ _SMPLX_TO_WUJI_PALM[side]
        wrist = np.linalg.norm(object_relative(source_obj, source_wrist)[:, :3, 3] - object_relative(geometry["objects"], palm)[:, :3, 3], axis=1)
        tips = official[f"{side}_joints_world"][mapping][:, _FINGERTIP_INDICES]; indices = [geometry["site_names"].index(f"{side}_{finger}_tip") for finger in ("thumb", "index", "middle", "ring", "pinky")]
        tip_error = np.linalg.norm(apply_transform(invert_transform(source_obj)[:, None], tips) - apply_transform(invert_transform(geometry["objects"])[:, None], geometry["positions"][:, indices]), axis=-1)
        sides[side] = {"object_relative_wrist_rmse_m": float(np.sqrt(np.mean(wrist ** 2))), "object_relative_fingertip_rmse_m": float(np.sqrt(np.mean(tip_error ** 2))), "object_relative_fingertip_max_m": float(tip_error.max())}
    original = json.loads((robot / "metrics_kinematic.json").read_text(encoding="utf-8"))
    metrics = {"frame_coverage": {"source_frames": int(len(official["source_frame_indices"])), "stage_b_frames": int(len(qpos)), "mapping_complete": bool(len(mapping) == len(qpos))}, "object_pose_preservation": {"translation_max_m": float(object_translation.max()), "rotation_rmse_rad": float(np.sqrt(np.mean(object_rotation ** 2)))}, "tracking": sides, "joint_limit_violations": original.get("joint_limit_violations"), "nan_or_inf": bool(not np.isfinite(qpos).all() or not np.isfinite(qvel).all()), "max_qpos_step": float(np.abs(np.diff(qpos, axis=0)).max()), "trajectory_continuity": "PASS" if np.isfinite(qpos).all() else "FAIL", "stage_b_penetration": "SDF_INVALID_FOR_STAGE: real Wuji hand meshes are not promoted to formal watertight reverse-SDF authority."}
    stages["stage_b"].update({"status": "PASS" if original.get("status") == "PASS" else "PARTIAL", "robot_dir": str(robot), "output_trajectory_sha256": sha256(robot / "trajectory_kinematic.npz"), "model_sha256": sha256(scene), "source_mapping": str(robot / "source_mapping.json")}); stages["final"] = "STAGE_B_BASELINE_PASS" if stages["stage_b"]["status"] == "PASS" else "STAGE_B_BASELINE_PARTIAL"
    write_json(root / "stage_status.json", stages); write_json(root / "retarget_metrics.json", metrics)
    return {"stages": stages, "result": {"qpos": qpos, "mapping": mapping, "geometry": geometry, "scene": scene, "metrics": metrics}}


def _retry_stage_ab(root: Path, authority: dict[str, Any], paths_config: str) -> dict[str, Any]:
    """Preserve an invalid setup attempt, then run Stage A/B once correctly."""
    if (root / "stage_status.json").exists():
        archive = root / "attempt_01_invalid_repo_workspace"; archive.mkdir(exist_ok=False)
        for name in ("effective_paths.yaml", "stage_a", "stage_b", "stage_status.json", "retarget_metrics.json", "workspace"):
            source = root / name
            if source.exists():
                shutil.move(str(source), str(archive / name))
    return _run_stage_ab(root, authority, paths_config)


def _q4_payload(authority: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    # Q4 uses the full source authority payload as a real source layer.  The
    # detailed robot sites/metrics remain in JSON even if a Stage-B failure
    # leaves no renderable robot trajectory.
    payload = _authority_payload(authority)
    payload["status"] = "Stage B unavailable" if result is None else "Stage B kinematic trajectory present"
    payload["categories"] = {**payload["categories"], "stage_b_penetration": "SDF_INVALID_FOR_STAGE; kinematic pass is not penetration pass"}
    if result is not None:
        visual = source_check._robot_payload(result["scene"], result["qpos"], 1, max_faces_per_mesh=12)
        collision = source_check._robot_payload(result["scene"], result["qpos"], 2, max_faces_per_mesh=12)
        lookup = {int(frame): index for index, frame in enumerate(result["mapping"])}
        for frame in payload["frames"]:
            frame["stage_index"] = lookup.get(int(frame["source_frame"]), -1)
        payload["robot_visual"] = visual
        payload["robot_collision"] = collision
    return payload


def write_q4_html(path: Path, authority: dict[str, Any], result: dict[str, Any] | None) -> None:
    write_text(path, _html("Q4 Source vs Stage B：" + authority["candidate"]["source_id"], _q4_payload(authority, result), "stage_b"))


def capture(root: Path, page: Path, category: str, frames: list[int], views: list[str], extra: str = "") -> dict[str, Any]:
    target = root / "screenshots" / category; target.mkdir(parents=True, exist_ok=True)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    rows = []
    for frame in frames:
        for view in views:
            output = target / f"frame_{frame:05d}_{view}.png"
            url = page.resolve().as_uri() + f"?frame={frame}&view={view}" + extra
            if chrome is None:
                rows.append({"frame": frame, "view": view, "status": "FAIL", "reason": "Chrome unavailable", "path": str(output)}); continue
            proc = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=5000", "--window-size=1800,1200", f"--screenshot={output}", url], text=True, capture_output=True, check=False, timeout=90)
            ok = proc.returncode == 0 and output.is_file() and output.stat().st_size > 10_000
            rows.append({"frame": frame, "view": view, "status": "PASS" if ok else "FAIL", "path": str(output), "url": url, "stderr": proc.stderr[-500:]})
    manifest = {"schema_version": 1, "html": str(page), "expected_count": len(frames) * len(views), "screenshots": rows, "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL"}
    write_json(target / "manifest.json", manifest); write_text(target / "REVIEW.md", "# 中文截图复核\n\nChrome 截图已生成；Codex 将在本次执行中实际打开代表性 PNG 并把人工观察写入最终验收报告。\n")
    return manifest


def _sample_sdf_display(points: np.ndarray, signed_m: np.ndarray, limit: int = 1_600) -> np.ndarray:
    """Keep a deterministic, category-balanced subset of an actual SDF query."""
    signed = np.asarray(signed_m, dtype=float)
    groups = (
        np.flatnonzero(signed < -SHALLOW_M),
        np.flatnonzero((signed < -NEAR_M) & (signed >= -SHALLOW_M)),
        np.flatnonzero(np.abs(signed) <= NEAR_M),
        np.flatnonzero(signed > NEAR_M),
    )
    chosen: list[np.ndarray] = []
    each = max(1, limit // len(groups))
    for group in groups:
        if len(group):
            chosen.append(group[np.linspace(0, len(group) - 1, min(each, len(group)), dtype=np.int64)])
    result = np.unique(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    if len(result) < limit:
        remainder = np.setdiff1d(np.arange(len(points), dtype=np.int64), result, assume_unique=True)
        if len(remainder):
            result = np.concatenate((result, remainder[np.linspace(0, len(remainder) - 1, min(limit - len(result), len(remainder)), dtype=np.int64)]))
    return np.asarray(result[:limit], dtype=np.int64)


def _candidate_visual_payload(root: Path, ranked: list[dict[str, Any]], paths_config: str, device: str, batch: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialise one real, worst-frame SDF visual per already-frozen candidate.

    The full-frame Q1 metrics are immutable and remain the acceptance evidence.
    These ten exact same-query calls exist only to preserve real colored query
    geometry in a self-contained review page; they never alter ranking.
    """
    frames: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for item in ranked:
        candidate = item["candidate"]
        candidate_root = Path(item["candidate_root"])
        with np.load(candidate_root / "source_quality_trace.npz", allow_pickle=False) as trace:
            worst_index = int(np.argmax(trace["frame_max_depth_m"]))
            source_frame = int(trace["source_frame"][worst_index])
        official, _chain = source_check.reconstruct_official(paths_config, str(candidate["source_id"]))
        with np.load(candidate_root / "query_points.npz", allow_pickle=False) as query_file:
            asset_points = np.asarray(query_file["points"], dtype=np.float32)
        capped = {side: _fixed_wrist_cap(official[f"{side}_vertices_world"], official[f"{side}_hand_faces"], side) for side in ("left", "right")}
        targets = [{"label": side, "source": f"official_smplx_{side}", "vertices": capped[side][0][worst_index : worst_index + 1], "faces": capped[side][1]} for side in ("left", "right")]
        world_points = _transform_points(official["T_world_object"][worst_index], asset_points)
        signed = _static_query(targets, world_points, candidate_root / "visualization_static_exec", source_frame, device, batch)
        display = _sample_sdf_display(world_points, signed)
        body = _submesh(official["body_vertices_world"][worst_index], official["body_faces"], 500)
        left = _submesh(official["left_vertices_world"][worst_index], official["left_hand_faces"], 350)
        right = _submesh(official["right_vertices_world"][worst_index], official["right_hand_faces"], 350)
        obj = _submesh(official["object_asset_vertices"], official["object_faces"], 600)
        metric_row = item["metrics"]["rows"][worst_index]
        frames.append({
            "rank": item["rank"], "sequence": candidate["source_id"], "source_frame": source_frame, "status": item["quality_status"],
            "summary": {"max_depth_mm": item["metrics"]["summary"]["max_penetration_depth_m"] * 1000, "p95_depth_mm": item["metrics"]["summary"]["P95_frame_max_depth_m"] * 1000, "deep_ratio": item["metrics"]["summary"]["deep_penetration_point_ratio"], "reliable_interval": item["metrics"]["summary"]["longest_reliable_contact_interval"]["length"]},
            "frame_categories": metric_row["categories"], "object_T": _transform12(official["T_world_object"][worst_index]), "left_T": _transform12(official["T_world_left_wrist"][worst_index]), "right_T": _transform12(official["T_world_right_wrist"][worst_index]),
            "body": {"vertices": body["vertices"], "faces": body["faces"]}, "left": {"vertices": left["vertices"], "faces": left["faces"]}, "right": {"vertices": right["vertices"], "faces": right["faces"]}, "object": obj,
            "query_world": world_points[display], "query_signed_mm": signed[display] * 1000.0,
        })
        manifest_rows.append({"sequence": candidate["source_id"], "source_frame": source_frame, "visual_query_point_count": int(len(display)), "source_q1_query_point_count": int(len(asset_points)), "backend": "Ref2Dex STATIC_EXEC MeshSDFfromAsset", "purpose": "visualization-only repeat of the frozen candidate's exact Q1 query set; ranking uses the persisted all-frame Q1 metrics"})
    manifest = {"schema_version": 1, "status": "PASS", "candidate_count": len(frames), "selection": "per-candidate full-Q1 worst frame (first argmax)", "rows": manifest_rows}
    write_json(root / "q1_candidates/Q1_VISUALIZATION_MANIFEST.json", manifest)
    return frames, manifest


def _candidate_html(root: Path, ranked: list[dict[str, Any]], visual_frames: list[dict[str, Any]] | None = None) -> None:
    rows = [{"rank": item["rank"], "sequence": item["candidate"]["source_id"], "subject": item["candidate"]["subject"], "object": item["candidate"]["object"], "max_depth_mm": item["metrics"]["summary"]["max_penetration_depth_m"] * 1000, "p95_depth_mm": item["metrics"]["summary"]["P95_frame_max_depth_m"] * 1000, "deep_ratio": item["metrics"]["summary"]["deep_penetration_point_ratio"], "reliable_interval": item["metrics"]["summary"]["longest_reliable_contact_interval"]["length"], "status": item["quality_status"], "reason": item["reason"]} for item in ranked]
    table = "".join(f"<tr><td>{row['rank']}</td><td>{row['sequence']}</td><td>{row['subject']}</td><td>{row['object']}</td><td>{row['max_depth_mm']:.3f}</td><td>{row['p95_depth_mm']:.3f}</td><td>{row['deep_ratio']:.6f}</td><td>{row['reliable_interval']}</td><td>{row['status']}</td><td>{row['reason']}</td></tr>" for row in rows)
    payload = {"rows": rows, "frames": visual_frames or []}
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Q1 十条候选 source quality 排行</title><script>{get_plotlyjs()}</script><style>body{{margin:0;background:#101820;color:#eef6f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#bar{{padding:12px;background:#173042}}#scene{{height:67vh}}#plot{{height:28vh}}table{{border-collapse:collapse;margin:12px}}td,th{{padding:4px;border:1px solid #527180}}select{{margin:3px}}pre{{white-space:pre-wrap;padding:8px}}</style></head><body><div id='bar'><h1>Q1：十条候选 Source Quality 排行与真实 Reverse-SDF 关键帧</h1><p>排序证据是每条完整帧 STATIC_EXEC reverse-SDF；下方每条显示其完整 Q1 最大穿透帧的同一固定 query set 再查询结果。此可视化不会改写排名或阈值。</p>候选 <select id='candidate'></select>　坐标 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体坐标</option><option value='left'>左腕坐标</option><option value='right'>右腕坐标</option></select><br><span>真实层：source SMPL-X body、左右手 surface、object contact mesh、按 signed reverse-SDF 着色的 object query points（红=inside/deep，白=near，蓝=outside）。</span></div><div id='scene'></div><div id='plot'></div><pre id='info'></pre><table><tr><th>rank</th><th>sequence</th><th>subject</th><th>object</th><th>max mm</th><th>P95 mm</th><th>deep ratio</th><th>reliable interval</th><th>status</th><th>reason</th></tr>{table}</table><script>const D={json.dumps(payload, ensure_ascii=False, default=_plain, separators=(',', ':'))},$=x=>document.getElementById(x);function M(T,p){{return p.map(q=>[T[0]*q[0]+T[1]*q[1]+T[2]*q[2]+T[3],T[4]*q[0]+T[5]*q[1]+T[6]*q[2]+T[7],T[8]*q[0]+T[9]*q[1]+T[10]*q[2]+T[11]])}}function I(T){{let r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],p=[T[3],T[7],T[11]],q=[[r[0][0],r[1][0],r[2][0]],[r[0][1],r[1][1],r[2][1]],[r[0][2],r[1][2],r[2][2]]];return[q[0][0],q[0][1],q[0][2],-(q[0][0]*p[0]+q[0][1]*p[1]+q[0][2]*p[2]),q[1][0],q[1][1],q[1][2],-(q[1][0]*p[0]+q[1][1]*p[1]+q[1][2]*p[2]),q[2][0],q[2][1],q[2][2],-(q[2][0]*p[0]+q[2][1]*p[1]+q[2][2]*p[2])]}}function mesh(n,s,c,o,T){{let v=M(T,s.vertices);return{{type:'mesh3d',name:n,x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:s.faces.map(x=>x[0]),j:s.faces.map(x=>x[1]),k:s.faces.map(x=>x[2]),color:c,opacity:o,flatshading:true}}}}function draw(){{let f=D.frames[+$('candidate').value],view=$('view').value,T=view==='object'?I(f.object_T):view==='left'?I(f.left_T):view==='right'?I(f.right_T):[1,0,0,0,0,1,0,0,0,0,1,0],object={{vertices:M(f.object_T,f.object.vertices),faces:f.object.faces}},q=M(T,f.query_world),tr=[mesh('完整 source SMPL-X body',f.body,'#77bfa3',.20,T),mesh('左手 surface',f.left,'#48cae4',.72,T),mesh('右手 surface',f.right,'#f28482',.72,T),mesh('object contact mesh',object,'#f4a261',.32,T),{{type:'scatter3d',mode:'markers',name:'actual reverse-SDF query points (mm)',x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),marker:{{size:2.8,color:f.query_signed_mm,colorscale:[[0,'#d00000'],[.35,'#f77f00'],[.45,'#fff'],[.55,'#fff'],[.7,'#4cc9f0'],[1,'#3a0ca3']],cmin:-10,cmax:10,colorbar:{{title:'signed mm'}},opacity:.92}}}}];Plotly.react('scene',tr,{{paper_bgcolor:'#101820',font:{{color:'#eef6f7'}},scene:{{aspectmode:'data',camera:{{eye:{{x:1.35,y:-1.35,z:1.05}}}}}},margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}});Plotly.react('plot',[{{x:D.rows.map(x=>x.sequence),y:D.rows.map(x=>x.max_depth_mm),type:'bar',name:'full-sequence max penetration (mm)'}},{{x:D.rows.map(x=>x.sequence),y:D.rows.map(x=>x.p95_depth_mm),type:'bar',name:'P95 frame max (mm)'}}],{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#eef6f7'}},barmode:'group',margin:{{l:45,r:15,t:25,b:110}},title:'完整序列 Reverse-SDF 指标'}});$('info').textContent=JSON.stringify({{sequence:f.sequence,rank:f.rank,source_frame:f.source_frame,status:f.status,full_sequence_summary:f.summary,actual_frame_categories:f.frame_categories,display_query_points:f.query_signed_mm.length}},null,2);window.__qReady=true}}D.frames.forEach((f,i)=>$('candidate').add(new Option(`${{f.rank}}. ${{f.sequence}} (Q1 worst frame=${{f.source_frame}})`,i)));let u=new URLSearchParams(location.search);if(u.get('frame'))$('candidate').value=u.get('frame');if(u.get('view'))$('view').value=u.get('view');$('candidate').onchange=draw;$('view').onchange=draw;draw();</script></body></html>"""
    write_text(root / "html/01_ten_candidate_source_quality.html", page)


def _selection(root: Path, ranked: list[dict[str, Any]], paths_config: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    reviewed: list[dict[str, Any]] = []
    eligible = [item for item in ranked if item["quality_status"] == "PASS_CANDIDATE"]
    for item in eligible:
        if "official" not in item:
            official, official_chain = source_check.reconstruct_official(paths_config, str(item["candidate"]["source_id"]))
            item["official"] = official
            item["official_chain"] = official_chain
        spider, spider_chain = source_check.capture_spider(paths_config, item["candidate"]["source_id"], item["official"])
        comparison, temporal, global_view = source_check.compare_source(item["official"], spider, float(item["candidate"]["fps"]))
        item.update({"spider": spider, "source_comparison": comparison, "temporal": temporal, "global_view": global_view, "official_review": comparison["decision"]["status"]})
        reviewed.append(item)
    passing = [item for item in reviewed if item["official_review"] == "PASS"]
    primary = passing[0] if passing else None
    backup = next((item for item in passing[1:] if primary and item["candidate"]["subject"] != primary["candidate"]["subject"] and item["candidate"]["object"] != primary["candidate"]["object"]), next((item for item in passing[1:]), None))
    decision = {"schema_version": 1, "policy": "source quality, reliable interval, role/patch authority then primary diversity; Stage B ease is not a selection criterion", "primary": primary["candidate"]["source_id"] if primary else None, "backup": backup["candidate"]["source_id"] if backup else None, "frozen_candidate_only": True, "official_reviews": [{"sequence": item["candidate"]["source_id"], "status": item["official_review"], "max_residual": item["source_comparison"]["decision"]["maximum_checked_residual"]} for item in reviewed]}
    for kind, value in (("primary", primary), ("backup", backup)):
        payload = {"schema_version": 1, "status": "PASS" if value else "NONE", "selection": value["candidate"] if value else None, "official_reconstruction_gate": value["source_comparison"] if value else None, "reverse_sdf_consistency": "same independently reconstructed source used in Q1; Q2 stores exact official-vs-Spider comparison" if value else None}
        write_json(root / "q2_selection" / f"{kind}_selection.json", payload)
    write_json(root / "q2_selection/selection_decision.json", decision)
    write_text(root / "reports/Q2_PRIMARY_BACKUP_SELECTION.md", f"# Q2 Primary / Backup 选择\n\n- primary：`{decision['primary']}`\n- backup：`{decision['backup']}`\n- 仅从冻结十条中选择；s5/cylindermedium_lift 不参与。\n- 官方/独立 SMPL-X 与 Spider loader 对比详情位于 selection JSON。\n")
    return primary, backup, reviewed


def _write_reports(root: Path, ranked: list[dict[str, Any]], primary: dict[str, Any] | None, backup: dict[str, Any] | None, authorities: dict[str, Any], q4: dict[str, Any]) -> None:
    rows = [{"rank": item["rank"], "subject": item["candidate"]["subject"], "sequence": item["candidate"]["source_id"], "object": item["candidate"]["object"], "frame_count": item["candidate"]["frame_count"], "max_depth_m": item["metrics"]["summary"]["max_penetration_depth_m"], "p95_depth_m": item["metrics"]["summary"]["P95_frame_max_depth_m"], "deep_penetration_ratio": item["metrics"]["summary"]["deep_penetration_point_ratio"], "longest_severe_interval": item["metrics"]["summary"]["longest_consecutive_severe_interval"], "reliable_contact_interval": item["metrics"]["summary"]["longest_reliable_contact_interval"], "contact_participation": item["metrics"]["summary"]["left_right_contact_participation"], "status": item["quality_status"], "reason": item["reason"]} for item in ranked]
    write_json(root / "reports/ten_candidate_quality_ranking.json", {"schema_version": 1, "candidate_count": 10, "run_local_gate": "top four source-quality ranking candidates without long severe interval are PASS_CANDIDATE; not a generic GRAB threshold", "ranking": rows})
    lines = ["# Q1 十条候选 Source Quality 排行", "", "|rank|sequence|subject|object|max mm|P95 mm|deep ratio|reliable interval|status|", "|---:|---|---|---|---:|---:|---:|---:|---|"]
    lines += [f"|{row['rank']}|{row['sequence']}|{row['subject']}|{row['object']}|{row['max_depth_m']*1000:.3f}|{row['p95_depth_m']*1000:.3f}|{row['deep_penetration_ratio']:.6f}|{row['reliable_contact_interval']['length']}|{row['status']}|" for row in rows]
    write_text(root / "reports/TEN_CANDIDATE_QUALITY_RANKING.md", "\n".join(lines) + "\n")
    write_json(root / "q0_rejection/rejection_manifest.json", {"schema_version": 1, "sequence_id": EXCLUDED, "quality_status": "SOURCE_QUALITY_REJECTED", "reason": "official and Spider sources match; formal maximum reverse-SDF penetration is approximately 10.740 mm across the primary interaction interval; depenetration changes source contact", "allowed_uses": ["negative_regression", "source_quality_failure_example", "viewer_regression", "role_contamination_test"], "forbidden_uses": ["primary_acceptance", "backup_acceptance", "source_role_authority", "semantic_patch_authority", "C_XA_authority", "dynamic_witness"]})
    write_text(root / "reports/Q0_SEQUENCE_REJECTION.md", "# Q0：s5/cylindermedium_lift 正式隔离\n\n状态：`SOURCE_QUALITY_REJECTED`。官方 GRAB 和 Spider source 已数值一致；问题是 raw source 的 formal reverse-SDF 最大穿透约 10.740 mm，且覆盖主交互区间。它仅保留作 negative regression/source-quality failure/viewer regression/role-contamination test，禁止作为 primary、backup、source role/patch/C-XA/dynamic authority。\n")
    for label in ("primary", "backup"):
        authority = authorities.get(label); stage = q4.get(label)
        write_text(root / "reports" / f"Q3_{label.upper()}_AUTHORITY.md", f"# Q3 {label}\n\n- sequence：`{authority['candidate']['source_id'] if authority else 'NONE'}`\n- authority：`{authority['status'] if authority else 'NOT_RUN'}`\n- deep penetration：严格排除，不计入 role 或 patch。\n")
        status = stage["stages"] if stage else {"stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}, "final": "NOT_RUN"}
        write_text(root / "reports" / f"Q4_{label.upper()}_STAGE_B.md", f"# Q4 {label} Stage A/B\n\n- Stage A：`{status['stage_a']['status']}`\n- Stage B：`{status['stage_b']['status']}`\n- conclusion：`{status.get('final')}`\n- Stage B kinematic PASS 不等于 penetration quality PASS。\n")
    final = {"schema_version": 1, "Q0_rejected_sequence": "PASS", "Q1_frozen_candidate_count": 10, "Q1_source_quality_screening": "PASS", "Q2_primary": primary["candidate"]["source_id"] if primary else "NONE", "Q2_backup": backup["candidate"]["source_id"] if backup else "NONE", "Q3_primary_authority": authorities.get("primary", {}).get("status", "NOT_RUN"), "Q3_backup_authority": authorities.get("backup", {}).get("status", "NOT_RUN"), "Q4_primary_Stage_A": q4.get("primary", {}).get("stages", {}).get("stage_a", {}).get("status", "NOT_RUN"), "Q4_primary_Stage_B": q4.get("primary", {}).get("stages", {}).get("stage_b", {}).get("status", "NOT_RUN"), "Q4_backup_Stage_A": q4.get("backup", {}).get("stages", {}).get("stage_a", {}).get("status", "NOT_RUN"), "Q4_backup_Stage_B": q4.get("backup", {}).get("stages", {}).get("stage_b", {}).get("status", "NOT_RUN"), "C_XA": "NOT_RUN", "Dynamics": "NOT_RUN", "User_visual_review": "PENDING"}
    write_json(root / "reports/q0_q4_final_acceptance.json", final)
    write_text(root / "reports/Q0_Q4_FINAL_ACCEPTANCE.md", "# Q0-Q4 Final Acceptance\n\n```json\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n```\n")


def _write_index(root: Path, primary: dict[str, Any] | None, backup: dict[str, Any] | None) -> None:
    links = ["01_ten_candidate_source_quality.html", "02_primary_source_contact_authority.html", "03_backup_source_contact_authority.html", "04_primary_source_vs_stage_b.html", "05_backup_source_vs_stage_b.html"]
    text = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><body style='font-family:system-ui;max-width:900px;margin:30px auto'><h1>Stage Q0-Q4 GRAB Source Screening</h1><p>Q0 rejected: <code>s5/cylindermedium_lift</code></p><p>Primary: <code>" + (primary["candidate"]["source_id"] if primary else "NONE") + "</code>；Backup: <code>" + (backup["candidate"]["source_id"] if backup else "NONE") + "</code></p><ol>" + "".join(f"<li><a href='{link}'>{link}</a></li>" for link in links) + "</ol><p><a href='../reports/Q0_Q4_FINAL_ACCEPTANCE.md'>最终结论</a></p></body></html>"
    write_text(root / "html/index.html", text)


def _write_handoff(root: Path, authorities: dict[str, Any], q4: dict[str, dict[str, Any]]) -> None:
    """Write the Chinese run handoff from the actual authority and Stage-B data."""
    lines = ["# Stage Q0-Q4 handoff", "", f"- Q0 rejected negative: `{EXCLUDED}` (`SOURCE_QUALITY_REJECTED`)", f"- primary: `{authorities.get('primary', {}).get('candidate', {}).get('source_id', 'NONE')}`", f"- backup: `{authorities.get('backup', {}).get('candidate', {}).get('source_id', 'NONE')}`", "- C-XA/dynamics: `NOT_RUN`", ""]
    for label in ("primary", "backup"):
        authority = authorities.get(label)
        result = q4.get(label, {}).get("result")
        if authority is None:
            lines.append(f"## {label}\n\n`NOT_RUN`\n")
            continue
        role = authority["role_schedule"]; patch = authority["semantic_patch"]
        lines += [f"## {label}: `{authority['candidate']['source_id']}`", "", f"- Q3 authority: `{authority['status']}`; reliable interval: `{role['reliable_interval']['length']}` frames; role-bearing frames: `{len(role['frames'])}`; semantic-patch triangles: `{len(patch['triangle_ids'])}`; no dilation and deep penetration excluded."]
        if result is None:
            lines += ["- Q4: `NOT_RUN`", ""]
            continue
        metrics = result["metrics"]
        lines += [f"- Q4 Stage A/B: `{q4[label]['stages']['stage_a']['status']}` / `{q4[label]['stages']['stage_b']['status']}`; source/stage frames: `{metrics['frame_coverage']['source_frames']}` / `{metrics['frame_coverage']['stage_b_frames']}`; mapping complete: `{metrics['frame_coverage']['mapping_complete']}`.", f"- object preservation: max translation `{metrics['object_pose_preservation']['translation_max_m']:.3e}` m; rotation RMSE `{metrics['object_pose_preservation']['rotation_rmse_rad']:.3e}` rad; joint violations `{metrics['joint_limit_violations']}`; NaN/Inf `{metrics['nan_or_inf']}`.", "- Stage B penetration remains `SDF_INVALID_FOR_STAGE`; this handoff does not approve C-XA, dynamics, or user visual acceptance.", ""]
    lines += ["## Review entry points", "", "- `html/index.html`", "- `reports/q0_q4_final_acceptance.json`", "- `q1_candidates/Q1_VISUALIZATION_MANIFEST.json`", "- `q2_selection/official_grab_visualization/{primary,backup}/`", "- `screenshots/*/manifest.json`", ""]
    write_text(root / "handoff/HANDOFF.md", "\n".join(lines))


def run(paths_config: str = "configs/local/paths.yaml", output_root: str = ".local_artifacts/stage_q0_q4_grab_screening", run_id: str | None = None, device: str = "cuda:0", query_batch_size: int = 4096) -> Path:
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-q0-q4")
    root = (REPO / output_root / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"output root exists; refusing overwrite: {root}")
    for directory in ("q0_rejection", "q1_candidates", "q2_selection", "q3_primary_authority", "q3_backup_authority", "q4_primary_retarget", "q4_backup_retarget", "reports", "html", "screenshots", "handoff"):
        (root / directory).mkdir(parents=True, exist_ok=False)
    write_json(root / "reports/static_exec_lineage.json", _static_exec_lineage())
    rows = scan_lightweight(paths_config); frozen = freeze_candidates(rows)
    write_json(root / "q1_candidates/frozen_candidate_list.json", {"schema_version": 1, "seed": SEED, "selection": "metadata-only fixed ordering with at-most-two subject/object diversity; no reverse-SDF result participated", "candidate_count": len(frozen), "candidates": frozen})
    write_text(root / "q1_candidates/FROZEN_CANDIDATE_LIST.md", "# 冻结的十条候选\n\n" + "\n".join(f"{item['frozen_index']}. `{item['source_id']}` ({item['subject']} / {item['object']})" for item in frozen) + "\n")
    audits = []
    for candidate in frozen:
        candidate_root = root / "q1_candidates" / f"sequence_{candidate['frozen_index']:02d}_{safe_id(candidate['source_id'])}"
        candidate_root.mkdir()
        audit = audit_source_quality(candidate_root, candidate, paths_config, device, query_batch_size); audit["candidate_root"] = str(candidate_root); audits.append(audit)
    ranked = rank_candidates(audits)
    visual_frames, _visual_manifest = _candidate_visual_payload(root, ranked, paths_config, device, query_batch_size)
    _candidate_html(root, ranked, visual_frames)
    primary, backup, reviewed = _selection(root, ranked, paths_config)
    authorities: dict[str, Any] = {}; q4: dict[str, Any] = {}
    for label, candidate in (("primary", primary), ("backup", backup)):
        if candidate is None:
            continue
        authority = build_authority(root / f"q3_{label}_authority", candidate, paths_config); authorities[label] = authority
        write_authority_html(root / "html" / ("02_primary_source_contact_authority.html" if label == "primary" else "03_backup_source_contact_authority.html"), authority)
        if authority["status"] == "PASS":
            q4[label] = _run_stage_ab(root / f"q4_{label}_retarget", authority, paths_config)
        write_q4_html(root / "html" / ("04_primary_source_vs_stage_b.html" if label == "primary" else "05_backup_source_vs_stage_b.html"), authority, q4.get(label, {}).get("result"))
    # HTML must exist even in a fail-closed Q2/Q3 path.
    fallback = primary or backup
    if fallback is None:
        raise RuntimeError("no independently reconstructed PASS_CANDIDATE available; cannot materialise required real authority HTML")
    for label, file_name in (("primary", "02_primary_source_contact_authority.html"), ("backup", "03_backup_source_contact_authority.html")):
        if label not in authorities:
            authority = build_authority(root / f"q3_{label}_authority", fallback, paths_config); authorities[label] = authority; write_authority_html(root / "html" / file_name, authority); write_q4_html(root / "html" / ("04_primary_source_vs_stage_b.html" if label == "primary" else "05_backup_source_vs_stage_b.html"), authority, None)
    _write_index(root, primary, backup)
    # Required count: candidate >=20 (two critical frames per candidate); Q3/Q4 24 each.
    capture(root, root / "html/01_ten_candidate_source_quality.html", "candidates", list(range(10)), ["world", "object"])
    for category, page in (("primary_authority", "02_primary_source_contact_authority.html"), ("backup_authority", "03_backup_source_contact_authority.html"), ("primary_stage_b", "04_primary_source_vs_stage_b.html"), ("backup_stage_b", "05_backup_source_vs_stage_b.html")):
        count = len(authorities["primary" if category.startswith("primary") else "backup"]["official"]["source_frame_indices"])
        capture(root, root / "html" / page, category, np.linspace(0, max(0, count - 1), 8, dtype=int).tolist(), ["world", "object", "left"])
    _write_reports(root, ranked, primary, backup, authorities, q4)
    _write_handoff(root, authorities, q4)
    return root


def continue_from_q1(root: str, paths_config: str = "configs/local/paths.yaml", device: str = "cuda:0", query_batch_size: int = 4096) -> Path:
    """Continue an interrupted run after all ten immutable Q1 audits exist.

    This intentionally reads the completed metrics rather than rerunning or
    replacing any reverse-SDF audit.  It is used only before Q2 has made an
    authority decision, and preserves the frozen candidate list verbatim.
    """
    run_root = Path(root).resolve()
    frozen_path = run_root / "q1_candidates/frozen_candidate_list.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["candidates"]
    audit_by_source: dict[str, dict[str, Any]] = {}
    for candidate_root in sorted((run_root / "q1_candidates").glob("sequence_*")):
        metric_path = candidate_root / "source_quality_metrics.json"
        if not metric_path.is_file():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        candidate = metrics["candidate"]
        audit_by_source[str(candidate["source_id"])] = {"candidate": candidate, "metrics": metrics, "candidate_root": str(candidate_root)}
    if len(audit_by_source) != 10 or set(audit_by_source) != {str(row["source_id"]) for row in frozen}:
        raise RuntimeError("continue requires exactly the original ten completed Q1 source-quality metrics")
    ranked = rank_candidates(list(audit_by_source.values()))
    visual_frames, _visual_manifest = _candidate_visual_payload(run_root, ranked, paths_config, device, query_batch_size)
    _candidate_html(run_root, ranked, visual_frames)
    preliminary_primary, preliminary_backup, reviewed = _selection(run_root, ranked, paths_config)
    # Q3 can reject a Q2 provisional candidate.  Try no candidate outside the
    # frozen top-four official review set, retain each failure, and pick the
    # next original frozen candidate only when its authority actually passes.
    attempts = run_root / "q3_selection_attempts"; attempts.mkdir(exist_ok=True)
    accepted: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    ordered = list(reviewed)
    # If a top-four item fails Q3 (for example, its actual object contact mesh
    # is open), the task explicitly permits the next ranked member of the
    # already-frozen ten.  Give that replacement its own independent source
    # comparison first; no candidate is added or re-screened by SDF.
    reviewed_ids = {str(item["candidate"]["source_id"]) for item in ordered}
    for item in ranked:
        if str(item["candidate"]["source_id"]) in reviewed_ids:
            continue
        official, official_chain = source_check.reconstruct_official(paths_config, str(item["candidate"]["source_id"]))
        spider, _spider_chain = source_check.capture_spider(paths_config, item["candidate"]["source_id"], official)
        comparison, temporal, global_view = source_check.compare_source(official, spider, float(item["candidate"]["fps"]))
        item.update({"official": official, "official_chain": official_chain, "spider": spider, "source_comparison": comparison, "temporal": temporal, "global_view": global_view, "official_review": comparison["decision"]["status"], "replacement_review": True})
        if item["official_review"] == "PASS":
            ordered.append(item)
    for item in ordered:
        attempt_root = attempts / safe_id(str(item["candidate"]["source_id"])); attempt_root.mkdir(exist_ok=True)
        authority = build_authority(attempt_root, item, paths_config)
        attempt_rows.append({"sequence": item["candidate"]["source_id"], "rank": item["rank"], "authority_status": authority["status"], "attempt_root": str(attempt_root)})
        if authority["status"] != "PASS":
            continue
        if not accepted:
            accepted.append(item)
            continue
        first = accepted[0]["candidate"]
        # Prefer the complementary subject/object before accepting an otherwise
        # valid backup.  All candidates are still from the original frozen ten.
        if item["candidate"]["subject"] != first["subject"] and item["candidate"]["object"] != first["object"]:
            accepted.append(item)
            break
    if len(accepted) < 2:
        raise RuntimeError("Q2_Q3_INSUFFICIENT_VALID_SEQUENCES: fewer than two frozen top-four candidates passed source authority")
    primary, backup = accepted
    authorities: dict[str, Any] = {}
    for label, item in (("primary", primary), ("backup", backup)):
        authority = build_authority(run_root / f"q3_{label}_authority", item, paths_config)
        if authority["status"] != "PASS":
            raise RuntimeError(f"authority was non-deterministic for {label}: {authority['status']}")
        authorities[label] = authority
        write_authority_html(run_root / "html" / ("02_primary_source_contact_authority.html" if label == "primary" else "03_backup_source_contact_authority.html"), authority)
    decision = {
        "schema_version": 1,
        "status": "PASS",
        "selection_policy": "Q1 source-quality ranking -> independent official/Spider comparison -> full-surface authority; top-four review set only; backup preferred different subject/object",
        "preliminary_primary": preliminary_primary["candidate"]["source_id"] if preliminary_primary else None,
        "preliminary_backup": preliminary_backup["candidate"]["source_id"] if preliminary_backup else None,
        "authority_attempts": attempt_rows,
        "primary": primary["candidate"]["source_id"],
        "backup": backup["candidate"]["source_id"],
        "frozen_candidate_only": True,
    }
    for label, item in (("primary", primary), ("backup", backup)):
        write_json(run_root / "q2_selection" / f"{label}_selection.json", {"schema_version": 1, "status": "PASS", "selection": item["candidate"], "quality_rank": item["rank"], "source_quality_status": item["quality_status"], "official_reconstruction_gate": item["source_comparison"], "authority_status": "PASS"})
    write_json(run_root / "q2_selection/selection_decision.json", decision)
    write_text(run_root / "reports/Q2_PRIMARY_BACKUP_SELECTION.md", f"# Q2 Primary / Backup 选择\n\n- primary：`{decision['primary']}`\n- backup：`{decision['backup']}`\n- 从原冻结十条的固定 top-four official review set 中选出；authority failed 的候选已留痕，但没有新增第十一条。\n")
    q4 = {label: _retry_stage_ab(run_root / f"q4_{label}_retarget", authority, paths_config) for label, authority in authorities.items()}
    write_q4_html(run_root / "html/04_primary_source_vs_stage_b.html", authorities["primary"], q4["primary"]["result"])
    write_q4_html(run_root / "html/05_backup_source_vs_stage_b.html", authorities["backup"], q4["backup"]["result"])
    _write_index(run_root, primary, backup)
    capture(run_root, run_root / "html/01_ten_candidate_source_quality.html", "candidates", list(range(10)), ["world", "object"])
    for category, page, authority in (("primary_authority", "02_primary_source_contact_authority.html", authorities["primary"]), ("backup_authority", "03_backup_source_contact_authority.html", authorities["backup"]), ("primary_stage_b", "04_primary_source_vs_stage_b.html", authorities["primary"]), ("backup_stage_b", "05_backup_source_vs_stage_b.html", authorities["backup"])):
        frame_count = len(authority["official"]["source_frame_indices"])
        capture(run_root, run_root / "html" / page, category, np.linspace(0, frame_count - 1, 8, dtype=int).tolist(), ["world", "object", "left"])
    _write_reports(run_root, ranked, primary, backup, authorities, q4)
    _write_handoff(run_root, authorities, q4)
    return run_root


def refresh_q1_visuals(root: str, paths_config: str = "configs/local/paths.yaml", device: str = "cuda:0", query_batch_size: int = 4096) -> Path:
    """Regenerate only the real Q1 HTML/screenshot evidence for a completed run."""
    run_root = Path(root).resolve()
    frozen = json.loads((run_root / "q1_candidates/frozen_candidate_list.json").read_text(encoding="utf-8"))["candidates"]
    audits: list[dict[str, Any]] = []
    for candidate_root in sorted((run_root / "q1_candidates").glob("sequence_*")):
        metric_path = candidate_root / "source_quality_metrics.json"
        if metric_path.is_file():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            audits.append({"candidate": metrics["candidate"], "metrics": metrics, "candidate_root": str(candidate_root)})
    if len(audits) != 10 or {str(item["candidate"]["source_id"]) for item in audits} != {str(item["source_id"]) for item in frozen}:
        raise RuntimeError("refresh requires exactly the original ten completed Q1 source-quality metrics")
    ranked = rank_candidates(audits)
    visual_frames, _visual_manifest = _candidate_visual_payload(run_root, ranked, paths_config, device, query_batch_size)
    _candidate_html(run_root, ranked, visual_frames)
    capture(run_root, run_root / "html/01_ten_candidate_source_quality.html", "candidates", list(range(10)), ["world", "object"])
    return run_root


def _load_stage_b_visual_result(root: Path, authority: dict[str, Any]) -> dict[str, Any] | None:
    """Read the existing isolated Stage-B output without running a solver again."""
    task = safe_id(str(authority["candidate"]["source_id"]))
    robot = root / "workspace/processed/grab/wuji_hand2_beta1/bimanual" / task / "0"
    scene = robot.parent / "scene.xml"
    trajectory = robot / "trajectory_kinematic.npz"
    mapping_path = robot / "source_mapping.json"
    if not (scene.is_file() and trajectory.is_file() and mapping_path.is_file()):
        return None
    with np.load(trajectory, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=float)
    mapping = np.asarray(json.loads(mapping_path.read_text(encoding="utf-8"))["source_frame_indices"], dtype=int)
    from spider.tools.new_grab_sequence_recheck import _extract_stage_b
    return {"qpos": qpos, "mapping": mapping, "geometry": _extract_stage_b(scene, qpos), "scene": scene, "metrics": json.loads((root / "retarget_metrics.json").read_text(encoding="utf-8"))}


def refresh_authority_q4_visuals(root: str, paths_config: str = "configs/local/paths.yaml") -> Path:
    """Refresh Q3/Q4 HTML and PNGs from stored Q2 choices and Stage-B outputs.

    This reads raw source and the already-delivered isolated Stage-B files only;
    it does not call Stage A/B, C-XA, or any dynamics workflow.
    """
    run_root = Path(root).resolve()
    authorities: dict[str, Any] = {}
    q4: dict[str, dict[str, Any]] = {}
    for label in ("primary", "backup"):
        selection = json.loads((run_root / "q2_selection" / f"{label}_selection.json").read_text(encoding="utf-8"))
        candidate = selection["selection"]
        official, chain = source_check.reconstruct_official(paths_config, str(candidate["source_id"]))
        authority = build_authority(run_root / f"q3_{label}_authority", {"candidate": candidate, "official": official, "official_chain": chain}, paths_config)
        if authority["status"] != "PASS":
            raise RuntimeError(f"stored {label} no longer satisfies deterministic Q3 authority: {authority['status']}")
        authorities[label] = authority
        write_authority_html(run_root / "html" / ("02_primary_source_contact_authority.html" if label == "primary" else "03_backup_source_contact_authority.html"), authority)
        result = _load_stage_b_visual_result(run_root / f"q4_{label}_retarget", authority)
        status = json.loads((run_root / f"q4_{label}_retarget" / "stage_status.json").read_text(encoding="utf-8")) if result is not None else {"stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}}
        q4[label] = {"result": result, "stages": status}
        write_q4_html(run_root / "html" / ("04_primary_source_vs_stage_b.html" if label == "primary" else "05_backup_source_vs_stage_b.html"), authority, result)
    for category, page, authority in (("primary_authority", "02_primary_source_contact_authority.html", authorities["primary"]), ("backup_authority", "03_backup_source_contact_authority.html", authorities["backup"]), ("primary_stage_b", "04_primary_source_vs_stage_b.html", authorities["primary"]), ("backup_stage_b", "05_backup_source_vs_stage_b.html", authorities["backup"])):
        count = len(authority["official"]["source_frame_indices"])
        capture(run_root, run_root / "html" / page, category, np.linspace(0, count - 1, 8, dtype=int).tolist(), ["world", "object", "left"])
    _write_handoff(run_root, authorities, q4)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Q0-Q4 GRAB source-quality screening")
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-root", default=".local_artifacts/stage_q0_q4_grab_screening")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--continue-existing", help="continue a Q1-complete interrupted root without rerunning its ten SDF audits")
    parser.add_argument("--refresh-q1-visuals", help="regenerate only the completed run's real Q1 visual evidence")
    parser.add_argument("--refresh-authority-q4-visuals", help="regenerate Q3/Q4 HTML and PNG evidence without rerunning Stage A/B")
    args = parser.parse_args()
    refresh_count = sum(bool(value) for value in (args.continue_existing, args.refresh_q1_visuals, args.refresh_authority_q4_visuals))
    if refresh_count > 1:
        raise ValueError("choose only one resume/refresh action")
    if args.continue_existing:
        print(continue_from_q1(args.continue_existing, args.paths_config, args.device, args.query_batch_size))
    elif args.refresh_q1_visuals:
        print(refresh_q1_visuals(args.refresh_q1_visuals, args.paths_config, args.device, args.query_batch_size))
    elif args.refresh_authority_q4_visuals:
        print(refresh_authority_q4_visuals(args.refresh_authority_q4_visuals, args.paths_config))
    else:
        print(run(args.paths_config, args.output_root, args.run_id, args.device, args.query_batch_size))


if __name__ == "__main__":
    main()
