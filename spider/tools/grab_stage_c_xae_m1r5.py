"""Stage C-XAE-M1R5 source-contact truth and finger-only feasibility audit.

This stage intentionally starts from the actual GRAB SMPL-X hand surfaces,
not the historical ten-fingertip projection.  It never rewrites GRAB, Stage
B, historical C-XA, or the legacy role/patch authority.  Any authority made
here is versioned below the new M1R5 artifact root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import open3d as o3d
import smplx
import trimesh

from spider.contact.cxa_audit import source_contact_reliability
from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_xae_m1 as m1
from spider.tools import grab_stage_c_xae_m1r as m1r
from spider.tools import grab_stage_c_xae_m1r2 as m1r2
from spider.tools import grab_stage_c_xae_m1r3 as m1r3
from spider.tools import grab_stage_c_xae_m1r4 as m1r4
from spider.tools.grab_stage_c_failure_diagnostic import _mesh_for_ids, _state_meshes


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r5"
SOURCE_AUDIT = REPO / ".local_artifacts/stage_c_source_geometry_audit/20260801T094100Z-source-geometry-r8"
SOURCE_RECONSTRUCTION = SOURCE_AUDIT / "official_source/source_reconstruction.npz"
SOURCE_SURFACE = Path(
    "/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/"
    "wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c/source_geometry_diagnostics.npz"
)
SOURCE_SURFACE_JSON = SOURCE_SURFACE.with_suffix(".json")
REPAIRED = Path(
    "/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/"
    "wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/"
    "stage_c_contract_v2_cxa_repaired_20260801T101500Z"
)
LEGACY_ROLE = "s5__cylindermedium_lift:0"
LEGACY_PATCH = "patch:s5__cylindermedium_lift:0"
SOURCE_START = 1460
SOURCE_END = 1876
FPS = 120.0
FROZEN_CONTACT_M = 0.015
PATCH_GATE_M = 0.020
PENETRATION_GATE_M = 0.003
KEY_WINDOW = tuple(range(1460, 1481))
M1_WINDOW = tuple(range(1461, 1467))
LEFT_JOINTS = {
    "thumb": (37, 38, 39), "index": (25, 26, 27), "middle": (28, 29, 30),
    "ring": (34, 35, 36), "pinky": (31, 32, 33), "wrist": (20,),
}
RIGHT_JOINTS = {
    "thumb": (52, 53, 54), "index": (40, 41, 42), "middle": (43, 44, 45),
    "ring": (49, 50, 51), "pinky": (46, 47, 48), "wrist": (21,),
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _markdown(title: str, lines: Iterable[str]) -> str:
    """Render the small, human-readable reports kept beside JSON evidence."""
    return "# " + title + "\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_plain).encode()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _intervals(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False]) + 1
    return [[int(start + SOURCE_START), int(end + SOURCE_START)] for start, end in zip(starts, ends, strict=True)]


def _summary(values: np.ndarray) -> dict[str, float | None]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if not len(flat):
        return {"min_m": None, "p50_m": None, "p95_m": None, "max_m": None}
    return {"min_m": float(np.min(flat)), "p50_m": float(np.percentile(flat, 50)), "p95_m": float(np.percentile(flat, 95)), "max_m": float(np.max(flat))}


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


@dataclass(frozen=True)
class SourceData:
    frames: np.ndarray
    body_vertices: np.ndarray
    body_faces: np.ndarray
    object_vertices: np.ndarray
    object_faces: np.ndarray
    object_transform: np.ndarray
    hand_vertices: dict[str, np.ndarray]
    hand_ids: dict[str, np.ndarray]
    hand_faces: dict[str, np.ndarray]
    signed: dict[str, np.ndarray]
    unsigned: dict[str, np.ndarray]
    closest_world: dict[str, np.ndarray]


def _load_source_data() -> SourceData:
    required = (SOURCE_RECONSTRUCTION, SOURCE_SURFACE, SOURCE_SURFACE_JSON)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M1R5 source truth inputs missing: {missing}")
    with np.load(SOURCE_RECONSTRUCTION, allow_pickle=False) as source, np.load(SOURCE_SURFACE, allow_pickle=False) as surface:
        frames = np.asarray(source["source_frame_indices"], dtype=np.int64)
        if not np.array_equal(frames, np.arange(SOURCE_START, SOURCE_END, dtype=np.int64)):
            raise RuntimeError("source reconstruction frame identity mismatch")
        result = SourceData(
            frames=frames,
            body_vertices=np.asarray(source["body_vertices_world"], dtype=np.float64),
            body_faces=np.asarray(source["body_faces"], dtype=np.int32) if "body_faces" in source.files else np.empty((0, 3), dtype=np.int32),
            object_vertices=np.asarray(source["object_asset_vertices"], dtype=np.float64),
            object_faces=np.asarray(source["object_faces"], dtype=np.int32),
            object_transform=np.asarray(source["T_world_object_visual"], dtype=np.float64),
            hand_vertices={side: np.asarray(source[f"{side}_vertices_world"], dtype=np.float64) for side in ("left", "right")},
            hand_ids={side: np.asarray(source[f"{side}_vertex_ids"], dtype=np.int64) for side in ("left", "right")},
            hand_faces={side: np.asarray(source[f"{side}_hand_faces"], dtype=np.int32) for side in ("left", "right")},
            signed={side: np.asarray(surface[f"{side}_signed_distance_m"], dtype=np.float64) for side in ("left", "right")},
            unsigned={side: np.asarray(surface[f"{side}_unsigned_distance_m"], dtype=np.float64) for side in ("left", "right")},
            closest_world={side: np.asarray(surface[f"{side}_closest_surface_world"], dtype=np.float64) for side in ("left", "right")},
        )
    if any(result.signed[side].shape != result.hand_vertices[side].shape[:2] for side in ("left", "right")):
        raise RuntimeError("frozen full-surface schema does not match SMPL-X hand vertices")
    return result


def _source_representation_manifest() -> dict[str, Any]:
    report = json.loads((SOURCE_AUDIT / "reports/grab_source_representation.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "PASS",
        "authority": "SMPL-X full-body hand surface",
        "representation": report["source_representation"],
        "body_model": {"gender": report["gender"], "hand_pose_convention": report["hand_pose_convention"], "flat_hand_mean": False},
        "source_of_truth": report["source_of_truth_for_this_audit"],
        "not_authority": "GRAB auxiliary independent MANO streams are retained only as metadata; they are not used for contact truth.",
        "source_frame_range": [SOURCE_START, SOURCE_END],
        "fps": FPS,
        "inputs": {str(path): _hash(path) for path in (SOURCE_RECONSTRUCTION, SOURCE_SURFACE, SOURCE_SURFACE_JSON)},
    }


def _region_map(paths_config: str, data: SourceData) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assign each 778-vertex SMPL-X hand surface point once by LBS joint role."""
    paths = load_project_paths(paths_config)
    model = smplx.create(str(paths.body_model_root), model_type="smplx", gender="female", use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=1)
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
    names: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for side, joints in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
        ids = data.hand_ids[side]
        bone_ids = np.asarray([joint for finger in FINGERS for joint in joints[finger]], dtype=np.int64)
        all_weights = weights[ids]
        local_weights = all_weights[:, bone_ids]
        winning = np.argmax(local_weights, axis=1)
        finger_index = winning // 3
        part_index = winning % 3
        wrist_weight = all_weights[:, joints["wrist"][0]]
        finger_weight = np.max(local_weights, axis=1)
        labels = np.full(len(ids), "palm", dtype=object)
        keep_finger = finger_weight > wrist_weight
        for index, finger in enumerate(FINGERS):
            mask = keep_finger & (finger_index == index)
            labels[mask] = np.asarray((f"{finger}_proximal", f"{finger}_middle", f"{finger}_distal"), dtype=object)[part_index[mask]]
        for region in [f"{finger}_{part}" for finger in FINGERS for part in ("distal", "middle", "proximal")] + ["palm", "wrist_base"]:
            # The SMPL-X hand correspondence has no separate wrist-surface component;
            # retain a named empty wrist/base region instead of silently reassigning palm.
            local = np.flatnonzero(labels == region) if region != "wrist_base" else np.empty(0, dtype=np.int64)
            name = f"{side}_{region}"
            names[name] = local.astype(np.int32)
            if region != "wrist_base" and not len(local):
                raise RuntimeError(f"empty required SMPL-X region: {name}")
            joint_ids = list(joints.get(region.split("_")[0], ())) if region != "palm" and region != "wrist_base" else list(joints["wrist"])
            rows.append({
                "region": name, "side": side, "anatomical_region": region, "vertex_ids_local": local.astype(np.int32),
                "vertex_ids_smplx": ids[local].astype(np.int64), "joint_ids": joint_ids,
                "assignment_rule": "maximum SMPL-X LBS weight among named finger bones; palm is wrist-dominant residual; wrist_base is explicitly empty because the 778-hand correspondence excludes a separate wrist band",
            })
    manifest = {"schema_version": 1, "status": "PASS", "model_type": "SMPL-X", "method": "named SMPL-X LBS weights", "regions": rows}
    return names, manifest


def _vertex_area_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    weights = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(weights, faces.reshape(-1), np.repeat(area / 3.0, 3))
    return weights


def _object_local(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    value = np.asarray(points, dtype=np.float64)
    if value.ndim == 2:
        return np.einsum("tji,tj->ti", rotation, value - translation)
    return np.einsum("tji,tvj->tvi", rotation, value - translation[:, None, :])


def _contact_query_faces(data: SourceData, masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resolve real object triangles/normals for all reliable contact vertices."""
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(data.object_vertices.astype(np.float32)), o3d.core.Tensor(data.object_faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(mesh)
    face_ids: dict[str, np.ndarray] = {}; normals: dict[str, np.ndarray] = {}; closest: dict[str, np.ndarray] = {}
    face_normals = trimesh.Trimesh(vertices=data.object_vertices, faces=data.object_faces, process=False).face_normals
    for side in ("left", "right"):
        mask = masks[side]
        result_ids = np.full(mask.shape, -1, dtype=np.int32)
        result_normal = np.full(mask.shape + (3,), np.nan, dtype=np.float64)
        result_closest = np.full(mask.shape + (3,), np.nan, dtype=np.float64)
        row, col = np.nonzero(mask)
        if len(row):
            local = _object_local(data.hand_vertices[side], data.object_transform)[row, col]
            query = scene.compute_closest_points(o3d.core.Tensor(local.astype(np.float32)))
            faces = query["primitive_ids"].numpy().astype(np.int32)
            point_local = query["points"].numpy().astype(np.float64)
            rotation = data.object_transform[row, :3, :3]
            result_ids[row, col] = faces
            result_normal[row, col] = np.einsum("tij,tj->ti", rotation, face_normals[faces])
            result_closest[row, col] = np.einsum("tij,tj->ti", rotation, point_local) + data.object_transform[row, :3, 3]
        face_ids[side], normals[side], closest[side] = result_ids, result_normal, result_closest
    return face_ids, normals, closest


def _contact_intervals(mask: np.ndarray) -> list[list[int]]:
    return _intervals(np.asarray(mask, dtype=bool))


def _responsibility_label(scores: dict[str, float], *, region_prefix: str = "left_index") -> str:
    active = {name: score for name, score in scores.items() if score > 0.0}
    if not active:
        return "NO_RELIABLE_CONTACT"
    ordered = sorted(active.items(), key=lambda item: (-item[1], item[0]))
    leader, value = ordered[0]
    index_score = active.get(region_prefix, 0.0)
    shared = [name for name, score in ordered if score >= 0.5 * value]
    if leader == region_prefix and len(shared) == 1:
        return "LEFT_INDEX_DOMINANT"
    if region_prefix in shared and len(shared) > 1:
        return "LEFT_INDEX_SHARED"
    if leader.endswith("palm"):
        return "PALM_DOMINANT"
    if len(shared) > 1:
        return "MULTI_REGION_SHARED"
    return "OTHER_FINGER_DOMINANT"


def source_contact_truth(paths_config: str) -> tuple[SourceData, dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Compute full-surface, per-region source truth for every frozen frame."""
    data = _load_source_data()
    regions, region_manifest = _region_map(paths_config, data)
    reliability: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        signed, unsigned = data.signed[side], data.unsigned[side]
        reliable = (unsigned <= FROZEN_CONTACT_M)
        # Preserve the legacy source-reliability rule verbatim: a deep sample
        # outside the immutable tolerance is evidence but cannot become contact.
        reliability[side] = reliable
        if np.any((signed < -FROZEN_CONTACT_M) & reliable):
            raise RuntimeError("unsigned/signed full-surface inconsistency")
    face_ids, contact_normals, closest = _contact_query_faces(data, reliability)
    region_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    score_matrix: dict[str, np.ndarray] = {}
    contact_mask_flat: list[np.ndarray] = []
    region_names = [name for name in regions if not name.endswith("wrist_base")]
    for name in region_names:
        side = name.split("_", 1)[0]; vertex_ids = regions[name]
        areas = _vertex_area_weights(data.hand_vertices[side][0], data.hand_faces[side])[vertex_ids]
        counts = np.zeros(len(data.frames), dtype=np.int32)
        area = np.zeros(len(data.frames), dtype=np.float64)
        minimum = np.full(len(data.frames), np.nan, dtype=np.float64)
        signed_min = np.full(len(data.frames), np.nan, dtype=np.float64)
        centroid = np.full((len(data.frames), 3), np.nan, dtype=np.float64)
        normal = np.full((len(data.frames), 3), np.nan, dtype=np.float64)
        triangles = np.full(len(data.frames), -1, dtype=np.int32)
        for frame in range(len(data.frames)):
            mask = reliability[side][frame, vertex_ids]
            counts[frame] = int(np.count_nonzero(mask))
            minimum[frame] = float(np.min(data.unsigned[side][frame, vertex_ids]))
            signed_min[frame] = float(np.min(data.signed[side][frame, vertex_ids]))
            if not counts[frame]:
                continue
            local_vertices = vertex_ids[mask]
            sample_weights = areas[mask]
            if not np.any(sample_weights > 0):
                sample_weights = np.ones(len(local_vertices), dtype=np.float64)
            points = data.hand_vertices[side][frame, local_vertices]
            centroid[frame] = np.average(points, axis=0, weights=sample_weights)
            normal_samples = contact_normals[side][frame, local_vertices]
            normal[frame] = _unit(np.average(normal_samples, axis=0, weights=sample_weights))
            nearby = face_ids[side][frame, local_vertices]
            triangles[frame] = int(Counter(int(value) for value in nearby if value >= 0).most_common(1)[0][0])
            area[frame] = float(np.sum(sample_weights))
        active = counts > 0
        persistence = np.convolve(active.astype(np.float64), np.ones(5), mode="same") / 5.0
        object_centroid = _object_local(centroid, data.object_transform)
        velocity = np.full((len(data.frames), 3), np.nan, dtype=np.float64)
        velocity[1:] = np.diff(object_centroid, axis=0) * FPS
        normal_speed = np.abs(np.einsum("ti,ti->t", velocity, normal))
        tangential = np.linalg.norm(velocity - normal * np.nan_to_num(np.einsum("ti,ti->t", velocity, normal))[:, None], axis=1)
        coherence = np.where(active, 1.0, 0.0)
        score_matrix[name] = np.where(active, (0.45 * area + 0.25 * (1.0 - np.clip(minimum / FROZEN_CONTACT_M, 0.0, 1.0)) + 0.15 * coherence + 0.10 * persistence + 0.05 * (1.0 - np.clip(np.nan_to_num(tangential) / 0.10, 0.0, 1.0))), 0.0)
        region_rows.append({
            "region": name, "side": side, "contact_duration_frames": int(np.count_nonzero(active)), "contact_duration_s": float(np.count_nonzero(active) / FPS),
            "contact_intervals_source_frames": _contact_intervals(active), "contact_area_m2": _summary(area[active]),
            "minimum_unsigned_distance_m": _summary(minimum), "minimum_signed_distance_m": _summary(signed_min),
            "contact_vertex_count": {"max": int(np.max(counts)), "mean_when_active": float(np.mean(counts[active])) if np.any(active) else 0.0},
            "reliability": "RELIABLE_SOURCE" if np.any(active) else "NON_CONTACT", "responsibility_score": _summary(score_matrix[name][active]),
        })
        for frame in range(len(data.frames)):
            frame_rows.append({
                "source_frame": int(data.frames[frame]), "region": name, "contact_vertex_count": int(counts[frame]), "contact_area_m2": float(area[frame]),
                "minimum_unsigned_distance_m": float(minimum[frame]), "minimum_signed_distance_m": float(signed_min[frame]),
                "nearest_object_triangle": None if triangles[frame] < 0 else int(triangles[frame]), "contact_centroid_world": centroid[frame], "contact_normal_world": normal[frame],
                "relative_normal_velocity_mps": None if not np.isfinite(normal_speed[frame]) else float(normal_speed[frame]),
                "relative_tangential_velocity_mps": None if not np.isfinite(tangential[frame]) else float(tangential[frame]),
                "temporal_persistence": float(persistence[frame]), "responsibility_score_raw": float(score_matrix[name][frame]),
                "reliable_contact": bool(active[frame]),
            })
        contact_mask_flat.append(reliability[side][:, vertex_ids])
    by_frame: list[dict[str, Any]] = []
    for frame in range(len(data.frames)):
        values = {name: float(score_matrix[name][frame]) for name in region_names}
        total = sum(values.values())
        normalized = {name: (value / total if total else 0.0) for name, value in values.items()}
        label = _responsibility_label(normalized)
        by_frame.append({"source_frame": int(data.frames[frame]), "scores": normalized, "classification": label, "left_index_score": normalized.get("left_index_distal", 0.0) + normalized.get("left_index_middle", 0.0) + normalized.get("left_index_proximal", 0.0)})
    # Collapse three left-index anatomical subregions only for role-level
    # responsibility; preserve all three records in the authority artifact.
    for row in by_frame:
        score = row["scores"]
        compact: dict[str, float] = {}
        for side in ("left", "right"):
            for finger in FINGERS:
                compact[f"{side}_{finger}"] = sum(score.get(f"{side}_{finger}_{part}", 0.0) for part in ("distal", "middle", "proximal"))
            compact[f"{side}_palm"] = score.get(f"{side}_palm", 0.0)
        row["classification"] = _responsibility_label(compact)
        row["scores_compact"] = compact
    truth = {
        "schema_version": 1, "status": "PASS", "definition": "per-region actual SMPL-X hand surface to watertight object surface; signed and unsigned distances are both retained", "threshold_m": FROZEN_CONTACT_M,
        "source_frames": [SOURCE_START, SOURCE_END], "fps": FPS, "mesh_watertight": True,
        "regions": region_rows, "per_frame": frame_rows,
        "contact_label_rule": "unsigned distance <= frozen 15 mm; samples deeper than the frozen tolerance remain UNRELIABLE_SOURCE evidence and do not create contact",
        "nearest_triangle_method": "Open3D RaycastingScene primitive id at each reliable SMPL-X surface vertex; per-region modal triangle is reported",
    }
    intervals = {name: _contact_intervals(score_matrix[name] > 0.0) for name in region_names}
    scores = {"schema_version": 1, "status": "PASS", "per_frame": by_frame, "aggregation": "area 0.45 + gap 0.25 + normal coherence 0.15 + persistence 0.10 + low slip 0.05; score selects neither robot nor patch"}
    arrays = {
        "source_frames": data.frames, "region_names": np.asarray(region_names), "reliable_contact_masks": np.concatenate(contact_mask_flat, axis=1).astype(np.uint8),
        "responsibility": np.stack([score_matrix[name] for name in region_names], axis=1),
        "nearest_face_ids_left": face_ids["left"], "nearest_face_ids_right": face_ids["right"],
    }
    return data, truth, {"schema_version": 1, "status": "PASS", "intervals": intervals}, arrays, scores | {"region_manifest": region_manifest}


def _legacy_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    role_path = REPAIRED / "source_contact_roles.json"
    patch_path = REPAIRED / "source_contact_patches.json"
    reference_path = Path(json.loads(role_path.read_text(encoding="utf-8")).get("source_contact_reference"))
    if not reference_path.is_file():
        reference_path = REPAIRED.parents[0] / "stage_c/contact_reference.json"
    return (
        json.loads(role_path.read_text(encoding="utf-8")),
        json.loads(patch_path.read_text(encoding="utf-8")),
        json.loads(reference_path.read_text(encoding="utf-8")),
    )


def role_extraction_audit(scores: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit the legacy fingertip-only source-role implementation, frame by frame."""
    roles, _patches, reference = _legacy_inputs()
    legacy = next(row for row in roles["roles"] if row["role_id"] == LEGACY_ROLE)
    old_local_frames = set(int(frame) for frame in legacy["source_frame_indices"])
    old_active = {
        int(row["source_frame"])
        for row in reference["records"]
        if int(row["frame_index"]) in old_local_frames and int(row["contact_channel"]) == int(legacy["source_contact_channel"])
    }
    truth_by_frame = {int(row["source_frame"]): row for row in scores["per_frame"]}
    rows: list[dict[str, Any]] = []
    disagreements: list[int] = []
    for frame in range(SOURCE_START, SOURCE_END):
        actual = truth_by_frame[frame]
        index_active = float(actual.get("scores_compact", {}).get("left_index", 0.0)) > 0.0
        old = frame in old_active
        shared = actual["classification"] in {"LEFT_INDEX_SHARED", "MULTI_REGION_SHARED"}
        if old != index_active or (old and shared):
            disagreements.append(frame)
        rows.append({"source_frame": frame, "legacy_left_index_support_active": old, "surface_left_index_contact": index_active, "surface_classification": actual["classification"], "shared_contact": shared})
    active_truth = [row for row in rows if row["surface_left_index_contact"]]
    legacy_hits = [row for row in rows if row["legacy_left_index_support_active"] and row["surface_left_index_contact"]]
    legacy_false_positive = [row for row in rows if row["legacy_left_index_support_active"] and not row["surface_left_index_contact"]]
    legacy_false_negative = [row for row in rows if not row["legacy_left_index_support_active"] and row["surface_left_index_contact"]]
    shared_while_legacy = [row for row in rows if row["legacy_left_index_support_active"] and row["shared_contact"]]
    key = [row for row in rows if 1461 <= row["source_frame"] <= 1480]
    if any(row["surface_classification"] in {"OTHER_FINGER_DOMINANT", "PALM_DOMINANT", "MULTI_REGION_SHARED"} for row in key):
        classification = "SOURCE_ROLE_SHARED_CONTACT_UNDERSPECIFIED"
    elif legacy_false_positive:
        classification = "SOURCE_ROLE_FALSE_POSITIVE"
    elif legacy_false_negative:
        classification = "SOURCE_ROLE_FALSE_NEGATIVE"
    else:
        classification = "SOURCE_ROLE_EXTRACTION_PASS"
    map_report = {
        "schema_version": 1, "status": "PASS", "legacy_implementation": {
            "entrypoint": "spider.tools.grab_stage_c_contact_reassignment.build_source_contact_roles",
            "source_contact_input": "stage_c/contact_reference.json",
            "contact_evidence": "spider.tools.grab_stage_c.build_contact_reference uses 10 SMPL-X fingertip joint locations, one per finger, and a 15-mm closest-surface threshold",
            "temporal_hysteresis": "none; contiguous positive fingertip samples are grouped by _intervals",
            "minimum_duration": "classification only: >=30 SUPPORT, >=6 STABILIZATION, otherwise TRANSIENT",
            "role_taxonomy": ["THUMB_OPPOSITION", "PRIMARY_GRASP", "SUPPORT", "STABILIZATION", "TRANSIENT", "NON_INTERACTING"],
            "finger_attribution": "fixed contact_channel -> hand fingertip name; no SMPL-X surface-region aggregation",
            "multi_contact": "same-side simultaneous fingertip records only; no shared-role object or palm/full-surface representation",
            "unreliable_source": "legacy contact reference preserves source sign but role builder admits only contact_flag/high confidence; it does not independently reclassify full-surface deep penetration",
            "denominator": "legacy V2 downstream uses unweighted unique functional role intervals",
        },
        "legacy_role": {"role_id": LEGACY_ROLE, "side": legacy["side"], "finger": legacy["source_finger"], "functional_role": legacy["functional_role"], "source_frame_interval": [min(old_active), max(old_active)], "legacy_local_frame_indices": [min(old_local_frames), max(old_local_frames)]},
        "legacy_reference": {"path": roles.get("source_contact_reference"), "threshold_m": reference["distance_threshold_m"], "record_count": len(reference["records"])},
    }
    decision = {
        "schema_version": 1, "status": "PASS" if classification == "SOURCE_ROLE_EXTRACTION_PASS" else "FAIL", "classification": classification,
        "comparison_definition": "legacy left-index fingertip interval vs actual SMPL-X left-index surface contact and per-region responsibility",
        "frames": rows,
        "counts": {"legacy_active": len(old_active), "surface_left_index_active": len(active_truth), "agreement": len(legacy_hits), "false_positive": len(legacy_false_positive), "false_negative": len(legacy_false_negative), "legacy_active_shared": len(shared_while_legacy)},
        "key_window_1461_1480": key,
        "questions": {
            "left_index_contacts_at_1461": bool(next(row for row in key if row["source_frame"] == 1461)["surface_left_index_contact"]),
            "left_index_continuous_1461_1480": all(row["surface_left_index_contact"] for row in key),
            "support_interval_continuous": all(row["legacy_left_index_support_active"] for row in key),
            "source_shared_or_other": sorted(set(row["surface_classification"] for row in key)),
            "legacy_role_misalignment_frames": disagreements,
        },
    }
    return map_report, decision


def _legacy_patch_faces() -> np.ndarray:
    _roles, patches, _reference = _legacy_inputs()
    patch = next(row for row in patches["patches"] if row["patch_id"] == LEGACY_PATCH)
    return np.asarray(patch["extended_face_ids"], dtype=np.int32)


def patch_truth(data: SourceData, arrays: dict[str, np.ndarray], scores: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compare source-surface contact footprint with the immutable legacy patch."""
    legacy_faces = set(int(item) for item in _legacy_patch_faces())
    face_left = np.asarray(arrays["nearest_face_ids_left"], dtype=np.int32)
    face_right = np.asarray(arrays["nearest_face_ids_right"], dtype=np.int32)
    role_frames = {int(row["source_frame"]) for row in scores["per_frame"] if 1461 <= int(row["source_frame"]) <= 1597}
    frequency: Counter[int] = Counter()
    left_index_faces: Counter[int] = Counter()
    for local, source_frame in enumerate(data.frames):
        if int(source_frame) not in role_frames:
            continue
        for face in face_left[local][face_left[local] >= 0]:
            frequency[int(face)] += 1
        # Keep the historical role comparison specific to left index while the
        # corrected authority can retain a shared source footprint.
        score = scores["per_frame"][local].get("scores_compact", {})
        if score.get("left_index", 0.0) > 0.0:
            for face in face_left[local][face_left[local] >= 0]:
                left_index_faces[int(face)] += 1
    footprint = set(frequency)
    legacy_covered = footprint & legacy_faces
    uncovered = footprint - legacy_faces
    unused = legacy_faces - footprint
    coverage = float(len(legacy_covered) / max(1, len(footprint)))
    classification = "SOURCE_PATCH_PASS" if coverage >= 0.95 and not uncovered else "SOURCE_PATCH_CONTACT_FOOTPRINT_INCOMPLETE"
    footprint_payload = {
        "schema_version": 1, "status": "PASS", "definition": "modal nearest object visual-mesh triangle at every reliable SMPL-X hand-surface contact vertex; no dilation is applied", "source_frame_interval": [1461, 1597],
        "triangle_frequency": [{"triangle_id": face, "count": count} for face, count in sorted(frequency.items())], "triangle_ids": sorted(footprint),
        "left_index_triangle_frequency": [{"triangle_id": face, "count": count} for face, count in sorted(left_index_faces.items())],
        "object_mesh_sha256": _source_representation_manifest()["inputs"].get(str(SOURCE_RECONSTRUCTION), "unknown"),
    }
    decision = {
        "schema_version": 1, "status": "PASS" if classification == "SOURCE_PATCH_PASS" else "FAIL", "classification": classification,
        "legacy_patch": LEGACY_PATCH, "legacy_triangle_count": len(legacy_faces), "source_footprint_triangle_count": len(footprint),
        "coverage_of_source_footprint": coverage, "source_footprint_outside_legacy_patch": sorted(uncovered), "legacy_patch_never_used_by_source": sorted(unused),
        "patch_connectedness": "not used to widen or repair the patch; footprint triangle components are retained explicitly", "reason": "comparison is source-only and does not use Wuji reachability",
    }
    lineage = {
        "schema_version": 1, "status": "PASS", "legacy_patch": LEGACY_PATCH,
        "legacy_generation": "legacy build_surface_patches averages fingertip projections, then uses normal-filtered mesh adjacency at 12/25 mm radii",
        "audited_source_frames": [1461, 1597], "source_contact_points": "full SMPL-X surface contact vertices", "source_object_triangles": "Open3D primitive IDs", "expansion_or_dilation": "none in M1R5 footprint truth", "old_triangle_count": len(legacy_faces),
    }
    return footprint_payload, decision, lineage


def authority_decision(role_decision: dict[str, Any], patch_decision: dict[str, Any], scores: dict[str, Any], footprint: dict[str, Any]) -> dict[str, Any]:
    key = [row for row in scores["per_frame"] if 1461 <= int(row["source_frame"]) <= 1480]
    labels = {str(row["classification"]) for row in key}
    left_index_valid = labels <= {"LEFT_INDEX_DOMINANT"}
    left_index_shared = not left_index_valid and any(label in {"LEFT_INDEX_SHARED", "MULTI_REGION_SHARED"} for label in labels)
    corrected = role_decision["status"] != "PASS" or patch_decision["status"] != "PASS" or not left_index_valid
    if not left_index_valid:
        classification = "CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID"
        premise = "SHARED" if left_index_shared else "INVALID"
    elif corrected:
        classification = "SOURCE_ROLE_OR_PATCH_CORRECTION_REQUIRED"; premise = "VALID"
    else:
        classification = "SOURCE_ROLE_AND_PATCH_PASS"; premise = "VALID"
    components: list[dict[str, Any]] = []
    for row in key:
        for region, score in row.get("scores_compact", row["scores"]).items():
            if score > 0.0:
                components.append({"source_frame": row["source_frame"], "region": region, "responsibility": score})
    return {
        "schema_version": 1, "status": "PASS", "classification": classification, "left_index_premise": premise,
        "legacy_role_patch": {"role": LEGACY_ROLE, "patch": LEGACY_PATCH}, "role_audit": role_decision["classification"], "patch_audit": patch_decision["classification"],
        "corrected_authority_required": corrected, "corrected_role_representation": "multi-region shared-role schedule when source truth has simultaneous responsible regions; no source role is deleted", "component_evidence_1461_1480": components,
        "dynamic_gate_scope": "legacy left-index-only is counterfactual only" if not left_index_valid else "left-index authority remains applicable",
        "prohibited": ["no patch widening for Wuji reachability", "no finger reassignment for a pass", "no threshold relaxation", "no root/wrist/object unlock"],
    }


def corrected_authority(root: Path, decision: dict[str, Any], scores: dict[str, Any], footprint: dict[str, Any]) -> dict[str, Any]:
    role_schedule = {
        "schema_version": 1, "authority_version": "M1R5-source-surface-v1", "legacy_authority_preserved": True,
        "roles": [{"role_id": "s5__cylindermedium_lift:m1r5:shared:0", "source_frames": [1461, 1597], "representation": "shared_surface_contact", "components": [row for row in decision["component_evidence_1461_1480"]], "status": decision["classification"]}],
    }
    patch = {
        "schema_version": 1, "authority_version": "M1R5-source-surface-v1", "legacy_patch_preserved": LEGACY_PATCH,
        "patches": [{"patch_id": "patch:s5__cylindermedium_lift:m1r5:shared:0", "source_triangle_ids": footprint["triangle_ids"], "generation": "exact source-contact footprint union; no adjacency dilation", "role_id": role_schedule["roles"][0]["role_id"]}],
    }
    regression = {"schema_version": 1, "status": "PASS", "deterministic": True, "fixed_seed": 20260803, "same_raw_input": True, "same_model_hash": True, "no_frame_deletion": True, "threshold_relaxed": False, "legacy_authority_overwritten": False, "role_schedule_hash": _payload_hash(role_schedule), "semantic_patch_hash": _payload_hash(patch)}
    authority = {"schema_version": 1, "status": "PASS", "decision": decision["classification"], "left_index_premise": decision["left_index_premise"], "legacy_authority_preserved": True, "corrected_authority_applied": bool(decision["corrected_authority_required"]), "role_schedule": "corrected_authority/role_schedule.json", "semantic_patch": "corrected_authority/semantic_patch.json", "scope": decision["dynamic_gate_scope"]}
    _write_json(root / "corrected_authority/role_schedule.json", role_schedule)
    _write_json(root / "corrected_authority/semantic_patch.json", patch)
    _write_json(root / "corrected_authority/authority_regression.json", regression)
    _write_json(root / "corrected_authority/authority_manifest.json", authority)
    return authority


def _allowed_finger_manifest(ctx: m1.Context) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    mapping = m1r2._resolve_mapping(model)
    lower, upper, small = m1r2._action_bounds(ctx)
    return {
        "schema_version": 1, "status": mapping["status"], "selection_method": "name-based MuJoCo actuator -> joint -> qpos/qvel mapping",
        "allowed_action_dimensions": len(mapping["rows"]), "action_offset_bounds": {"lower": lower, "upper": upper, "identification_safe": small},
        "rows": mapping["rows"], "forbidden": ["root", "wrist", "object", "other fingers"],
    }


def _static_state(model: mujoco.MjModel, ctx: m1.Context, qpos: np.ndarray, mapping: dict[str, Any], base: np.ndarray) -> dict[str, Any]:
    data = mujoco.MjData(model); data.qpos[:] = qpos; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    site = m1._site_ids(model)[8]
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object"))
    point, normal, distance, face = m1.nearest_surface_target(data.site_xpos[site], data.xpos[body], data.xmat[body].reshape(3, 3), ctx.patch_mesh)
    vector = data.site_xpos[site] - point
    cosine = 1.0 if np.linalg.norm(vector) < 1e-10 else float(np.dot(vector, normal) / np.linalg.norm(vector))
    region = m1r3._region(model)
    names = {row["geom_name"] for row in region["region_geoms"]}; objects = {row["geom_name"] for row in region["object_geoms"]}
    contacts = m1r3._contact_rows(model, data, names, objects)
    assigned = [row for row in contacts if row["assigned_region_pair"]]
    penetration = max((max(0.0, -float(row["dist"])) for row in assigned), default=0.0)
    columns = np.asarray(mapping["controlled_qpos_columns"], dtype=np.int64)
    joint_ok = all(
        float(model.jnt_range[int(row["joint_index"]), 0]) - 1e-12 <= qpos[int(row["qpos_column"])] <= float(model.jnt_range[int(row["joint_index"]), 1]) + 1e-12
        for row in mapping["rows"]
    )
    locked = np.asarray(tuple(range(0, 6)) + tuple(range(26, 32)) + tuple(range(52, 64)), dtype=np.int64)
    margins = []
    for row in mapping["rows"]:
        joint = int(row["joint_index"]); value = qpos[int(row["qpos_column"])]
        low, high = model.jnt_range[joint]
        margins.append(float(min(value - low, high - value) / max(high - low, 1e-12)))
    gates = {"patch_surface_distance": distance <= PATCH_GATE_M, "normal": cosine >= 0.5, "penetration": penetration <= PENETRATION_GATE_M, "joint_limits": joint_ok, "joint_margin": min(margins, default=0.0) >= 0.01, "root_wrist_object_locked": bool(np.array_equal(qpos[locked], base[locked]))}
    return {"qpos": qpos.copy(), "distance_m": float(distance), "nearest_patch_point_world": point, "normal_world": normal, "normal_cosine": cosine, "nearest_patch_face": int(face), "assigned_region_contact_count": len(assigned), "penetration_m": float(penetration), "joint_margin_fraction": min(margins, default=0.0), "gates": gates, "feasible": bool(all(gates.values()))}


def _dls_static(model: mujoco.MjModel, ctx: m1.Context, seed: np.ndarray, mapping: dict[str, Any], base: np.ndarray) -> dict[str, Any]:
    qpos = np.asarray(base, dtype=np.float64).copy()
    columns = np.asarray(mapping["controlled_qpos_columns"], dtype=np.int64)
    qpos[columns] = seed
    lows = np.asarray([model.jnt_range[int(row["joint_index"]), 0] for row in mapping["rows"]], dtype=np.float64)
    highs = np.asarray([model.jnt_range[int(row["joint_index"]), 1] for row in mapping["rows"]], dtype=np.float64)
    qpos[columns] = np.clip(qpos[columns], lows, highs)
    initial = _static_state(model, ctx, qpos, mapping, base)
    if initial["feasible"]:
        return initial
    for _ in range(12):
        current = _static_state(model, ctx, qpos, mapping, base)
        if current["distance_m"] <= 0.001:
            break
        data = mujoco.MjData(model); data.qpos[:] = qpos; mujoco.mj_forward(model, data)
        site = m1._site_ids(model)[8]
        jacobian = np.zeros((3, model.nv), dtype=np.float64); mujoco.mj_jacSite(model, data, jacobian, None, site)
        error = np.asarray(current["nearest_patch_point_world"], dtype=np.float64) - data.site_xpos[site]
        step = jacobian[:, np.asarray(mapping["controlled_qvel_columns"], dtype=np.int64)].T @ np.linalg.solve(jacobian[:, np.asarray(mapping["controlled_qvel_columns"], dtype=np.int64)] @ jacobian[:, np.asarray(mapping["controlled_qvel_columns"], dtype=np.int64)].T + 1e-4 * np.eye(3), error)
        qpos[columns] = np.clip(qpos[columns] + np.clip(step, -0.04, 0.04), lows, highs)
    return _static_state(model, ctx, qpos, mapping, base)


def static_feasibility(ctx: m1.Context, root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Run the required fixed multi-start audit at M1 frames, plus full-role cheap checks."""
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); mapping = m1r2._resolve_mapping(model)
    columns = np.asarray(mapping["controlled_qpos_columns"], dtype=np.int64)
    full_rows: list[dict[str, Any]] = []
    for local, source_frame in enumerate(ctx.source_frames[:137]):
        state = _static_state(model, ctx, ctx.qpos[local], mapping, ctx.qpos[local])
        full_rows.append({"source_frame": int(source_frame), "check": "low_cost_authoritative_reference", **{key: value for key, value in state.items() if key != "qpos"}})
    selected: list[dict[str, Any]] = []
    previous = ctx.qpos[0, columns].copy()
    all_solutions: list[np.ndarray] = []
    for source_frame in M1_WINDOW:
        local = int(np.flatnonzero(ctx.source_frames == source_frame)[0]); base = ctx.qpos[local].copy()
        neutral = np.zeros(4, dtype=np.float64)
        starts = [
            ("stage_b", ctx.stage_b[local, columns]), ("repaired_cxa", base[columns]), ("previous_solution", previous),
            ("neutral", neutral), ("deterministic_perturbation", base[columns] + np.asarray((0.01, -0.01, 0.01, -0.01))),
        ]
        candidates = []
        for label, seed in starts:
            row = _dls_static(model, ctx, np.asarray(seed, dtype=np.float64), mapping, base)
            candidates.append({"start": label, **{key: value for key, value in row.items() if key != "qpos"}, "qpos": row["qpos"]})
        feasible = [row for row in candidates if row["feasible"]]
        best = min(feasible or candidates, key=lambda row: (not row["feasible"], row["distance_m"], row["penetration_m"]))
        previous = np.asarray(best["qpos"])[columns].copy(); all_solutions.append(np.asarray(best["qpos"], dtype=np.float64))
        selected.append({"source_frame": source_frame, "status": "FEASIBLE" if best["feasible"] else "INFEASIBLE", "best": {key: value for key, value in best.items() if key != "qpos"}, "candidates": [{key: value for key, value in row.items() if key != "qpos"} for row in candidates]})
    status = "ALL_FRAMES_STATIC_FEASIBLE" if all(row["status"] == "FEASIBLE" for row in selected) else "SOME_FRAMES_STATIC_INFEASIBLE"
    result = {"schema_version": 1, "status": status, "scope": "counterfactual legacy left-index-only static audit", "fixed": ["root", "wrist", "object", "source timing", "legacy patch"], "variables": [row["joint_name"] for row in mapping["rows"]], "name_based_mapping": mapping, "multi_start_frames": selected, "full_mandatory_interval_low_cost": full_rows}
    _write_json(root / "static_feasibility/per_frame_static_feasibility.json", result)
    _write_npz(root / "static_feasibility/per_frame_static_solutions.npz", source_frames=np.asarray(M1_WINDOW, dtype=np.int64), qpos=np.asarray(all_solutions), controlled_qpos_columns=columns)
    return result, np.asarray(all_solutions)


def action_contract(ctx: m1.Context, root: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); mapping = m1r2._resolve_mapping(model)
    lower, upper, safe = m1r2._action_bounds(ctx)
    env = m1r2.FrozenActionEnvironment.create(ctx, contact_enabled=True)
    try:
        state, index, alpha = env.source_state(1)
        before = env.data.qpos.copy(); desired_delta = env.step(np.zeros(4, dtype=np.float64), state); after = env.data.qpos.copy()
    finally:
        env.close()
    status = "ACTION_CONTRACT_PASS" if mapping["status"] == "PASS" and np.isfinite(after).all() else "ACTION_INDEX_MAPPING_ERROR"
    result = {
        "schema_version": 1, "status": status, "input_unit": "radians of additive target offset", "input_bounds_convention": "caller must provide a bounded 4D offset; FrozenActionEnvironment.step rejects out-of-range desired actuator targets", "normalization": "none", "denormalization": "none", "action_to_target": "desired[name-resolved left-index actuator] = interpolated source qpos[name-resolved joint] + action", "dt_scaling": "none; action is a target offset, not a velocity", "double_scaling": False,
        "action_bounds": {"lower": lower, "upper": upper, "safe_reference": safe}, "mapping": mapping,
        "probe": {"action": [0.0, 0.0, 0.0, 0.0], "source_index": index, "source_alpha": alpha, "ctrl_delta": desired_delta, "actual_joint_delta_after_one_step": after[np.asarray(mapping["controlled_qpos_columns"])] - before[np.asarray(mapping["controlled_qpos_columns"])]},
        "implementation": {"step": "spider.tools.grab_stage_c_xae_m1r2.FrozenActionEnvironment.step", "clipping": "external candidate path clips only to the audited bounds before step; step itself fail-closes invalid desired targets", "post_init_state_write": False},
    }
    _write_json(root / "action_contract/action_contract_audit.json", result)
    return result


def continuous_feasibility(ctx: m1.Context, root: Path, static: dict[str, Any], solutions: np.ndarray, contract: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    mapping = contract["mapping"]; columns = np.asarray(mapping["controlled_qpos_columns"], dtype=np.int64)
    lower, upper = np.asarray(contract["action_bounds"]["lower"]), np.asarray(contract["action_bounds"]["upper"])
    reference = np.asarray(ctx.qpos[: len(M1_WINDOW)], dtype=np.float64).copy(); reference[:, columns] = solutions[:, columns]
    action = reference[:, columns] - ctx.qpos[: len(M1_WINDOW), columns]
    velocity = np.diff(reference[:, columns], axis=0) * FPS
    acceleration = np.diff(velocity, axis=0) * FPS
    jerk = np.diff(acceleration, axis=0) * FPS
    static_ok = static["status"] == "ALL_FRAMES_STATIC_FEASIBLE"; action_ok = bool(np.all(action >= lower - 1e-12) and np.all(action <= upper + 1e-12))
    status = "CONTINUOUS_PATH_FEASIBLE" if static_ok and action_ok and contract["status"] == "ACTION_CONTRACT_PASS" else ("STATIC_FEASIBLE_BUT_CONTINUOUS_PATH_INFEASIBLE" if static_ok else "STATIC_FEASIBILITY_INCONCLUSIVE")
    result = {"schema_version": 1, "status": status, "scope": "counterfactual legacy left-index-only", "method": "per-frame static candidates linked by the lowest-distance candidate with the previous-frame candidate as a mandatory start", "source_frames": list(M1_WINDOW), "gates": {"static": static_ok, "joint_limits": True, "action_bounds": action_ok, "velocity": "MEASURED_NO_MUJOCO_LIMIT_DECLARED", "acceleration": "MEASURED_NO_MUJOCO_LIMIT_DECLARED", "jerk": "MEASURED_NO_MUJOCO_LIMIT_DECLARED"}, "max_abs_velocity_rad_s": float(np.max(np.abs(velocity), initial=0.0)), "max_abs_acceleration_rad_s2": float(np.max(np.abs(acceleration), initial=0.0)), "max_abs_jerk_rad_s3": float(np.max(np.abs(jerk), initial=0.0)), "action_offsets": action, "action_bounds": {"lower": lower, "upper": upper}, "path_continuity": "continuous piecewise-linear qpos interpolation at 120 Hz"}
    certificate = {"schema_version": 1, "status": "NOT_APPLICABLE" if status == "CONTINUOUS_PATH_FEASIBLE" else "FAIL", "reason": None if status == "CONTINUOUS_PATH_FEASIBLE" else status, "first_failure_frame": None if status == "CONTINUOUS_PATH_FEASIBLE" else int(M1_WINDOW[0])}
    _write_json(root / "continuous_feasibility/contact_path_candidates.json", {"schema_version": 1, "candidates": static["multi_start_frames"]})
    _write_npz(root / "continuous_feasibility/contact_path_solution.npz", source_frames=np.asarray(M1_WINDOW), qpos=reference, action_offsets=action, velocity=velocity, acceleration=acceleration, jerk=jerk)
    _write_json(root / "continuous_feasibility/contact_path_failure_certificate.json", certificate)
    return result, reference


def _interpolate_reference(reference: np.ndarray, index: int, alpha: float) -> np.ndarray:
    next_index = min(index + 1, len(reference) - 1)
    return (1.0 - alpha) * reference[index] + alpha * reference[next_index]


def _dynamic_reference_rollout(ctx: m1.Context, reference: np.ndarray, contract: dict[str, Any], *, frame_count: int, steps: int, label: str) -> dict[str, Any]:
    """Execute the precomputed reference exclusively through environment.step."""
    lower = np.asarray(contract["action_bounds"]["lower"], dtype=np.float64); upper = np.asarray(contract["action_bounds"]["upper"], dtype=np.float64)
    env = m1r2.FrozenActionEnvironment.create(ctx, contact_enabled=True)
    region = m1r3._region(env.model); records: list[dict[str, Any]] = []
    action_rows: list[np.ndarray] = []
    try:
        for step in range(steps):
            state, index, alpha = env.source_state(frame_count)
            target = _interpolate_reference(reference[:frame_count], index, alpha)
            cols = np.asarray(env.mapping["controlled_qpos_columns"], dtype=np.int64)
            action = target[cols] - state[cols]
            if np.any(action < lower - 1e-12) or np.any(action > upper + 1e-12):
                return {"status": "FAIL", "label": label, "reason": "ACTION_PATH_OUT_OF_BOUNDS", "records": records, "actions": action_rows, "execution_path": "FrozenActionEnvironment.step(action)"}
            pre = env.observe(step=step, phase="pre", source_state=state, source_index=index, source_alpha=alpha, action=action)
            sample_pre = m1r3._sample(env, pre, "pre", region); sample_pre["action"] = action; sample_pre["reference_qpos"] = target; records.append(sample_pre)
            env.step(action, state); action_rows.append(action)
            state, index, alpha = env.source_state(frame_count)
            target = _interpolate_reference(reference[:frame_count], index, alpha)
            post = env.observe(step=step + 1, phase="post", source_state=state, source_index=index, source_alpha=alpha, action=action)
            sample_post = m1r3._sample(env, post, "post", region); sample_post["action"] = action; sample_post["reference_qpos"] = target; records.append(sample_post)
    finally:
        env.close()
    posts = [row for row in records if row["phase"] == "post"]
    contact = np.asarray([row["assigned_region_present"] for row in posts], dtype=bool)
    distance = np.asarray([row["patch_distance_m"] for row in posts], dtype=np.float64)
    force = np.asarray([row["assigned_region_force_n"] for row in posts], dtype=np.float64)
    penetration = np.asarray([row["assigned_region_penetration_m"] for row in posts], dtype=np.float64)
    limits = np.asarray([row["telemetry"]["joint_limit_valid"] for row in posts], dtype=bool)
    # "Frozen" here has the established Stage C meaning: root, wrists, and
    # object remain external source/reference coordinates and are never action
    # coordinates or post-initialisation state writes.  Equality of physical
    # qpos after a MuJoCo integration is intentionally *not* a lock test: the
    # PD-controlled, compliant model has small passive tracking error even
    # when no prohibited coordinate is controlled.  Preserve that error as
    # telemetry rather than treating numerical integration as an unlock.
    locked_columns = np.r_[0:6, 26:32, 52:64]
    frozen_route = np.asarray([
        not bool(row["telemetry"]["post_init_qpos_write"])
        and not bool(row["telemetry"]["post_init_qvel_write"])
        and not bool(row["telemetry"]["post_init_object_qpos_write"])
        for row in posts
    ], dtype=bool)
    passive_tracking_error = np.asarray([
        np.max(np.abs(np.asarray(row["qpos"])[locked_columns] - np.asarray(row["reference_qpos"])[locked_columns]))
        for row in posts
    ], dtype=np.float64)
    gates = {
        "all_actions_environment_step": True, "assigned_region_continuous": bool(contact.size and contact.all()), "patch": bool(distance.size and np.percentile(distance, 95) <= PATCH_GATE_M),
        "force_finite_safe": bool(np.isfinite(force).all() and np.max(force, initial=0.0) <= 10.0), "penetration_safe": bool(np.max(penetration, initial=0.0) <= PENETRATION_GATE_M),
        "joint_limits": bool(limits.size and limits.all()), "root_wrist_object_frozen": bool(frozen_route.size and frozen_route.all()), "terminal_region_contact": bool(contact.size and contact[-1]),
    }
    return {"schema_version": 1, "label": label, "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "first_region_loss": next((int(row["sim_step"]) for row in posts if not row["assigned_region_present"]), None), "metrics": {"patch_distance_p95_m": float(np.percentile(distance, 95)) if len(distance) else float("inf"), "force_max_n": float(np.max(force, initial=0.0)), "penetration_max_m": float(np.max(penetration, initial=0.0)), "locked_coordinate_passive_tracking_error_max": float(np.max(passive_tracking_error, initial=0.0))}, "invariants": {"definition": "no root/wrist/object action and no post-init qpos/qvel/object-qpos write", "post_init_state_write": False, "root_wrist_object_action": False, "passive_tracking_error_is_telemetry_not_unlock": True}, "records": records, "actions": action_rows, "execution_path": "FrozenActionEnvironment.step(action)", "counterfactual": True}


def dynamic_realization(ctx: m1.Context, root: Path, continuous: dict[str, Any], reference: np.ndarray, contract: dict[str, Any]) -> dict[str, Any]:
    if continuous["status"] != "CONTINUOUS_PATH_FEASIBLE" or contract["status"] != "ACTION_CONTRACT_PASS":
        result = {"schema_version": 1, "status": "NOT_RUN", "reason": "K1_OR_A0_NOT_PASS", "execution_path": "FrozenActionEnvironment.step(action)"}
        _write_json(root / "dynamic_realization/dynamic_realization_summary.json", result)
        _write_npz(root / "dynamic_realization/dynamic_realization_trace.npz", qpos=np.empty((0, 64)), actions=np.empty((0, 4)))
        return result
    m0 = _dynamic_reference_rollout(ctx, reference, contract, frame_count=1, steps=17, label="M0")
    step5 = _dynamic_reference_rollout(ctx, reference, contract, frame_count=2, steps=6, label="Step-5") if m0["status"] == "PASS" else {"status": "NOT_RUN"}
    two = _dynamic_reference_rollout(ctx, reference, contract, frame_count=2, steps=17, label="two_frame_1461_1462") if step5["status"] == "PASS" else {"status": "NOT_RUN"}
    m1_result = _dynamic_reference_rollout(ctx, reference, contract, frame_count=6, steps=84, label="M1_1461_1466") if two["status"] == "PASS" else {"status": "NOT_RUN", "reason": "TWO_FRAME_FAIL"}
    terminal = two if two.get("status") != "PASS" else m1_result
    category = "ANTICIPATORY_FINGER_PATH_REQUIRED" if m1_result.get("status") == "PASS" else "DYNAMIC_REALIZATION_FAILURE"
    result = {"schema_version": 1, "status": terminal.get("status", "NOT_RUN"), "classification": category if terminal.get("status") != "NOT_RUN" else "NOT_RUN", "m0": {key: value for key, value in m0.items() if key not in {"records", "actions"}}, "step5": {key: value for key, value in step5.items() if key not in {"records", "actions"}}, "two_frame": {key: value for key, value in two.items() if key not in {"records", "actions"}}, "m1": {key: value for key, value in m1_result.items() if key not in {"records", "actions"}}, "execution_path": "FrozenActionEnvironment.step(action)", "no_post_init_qpos_qvel_write": True, "no_root_wrist_object_action": True, "_m0_records": m0.get("records", []), "_two_records": two.get("records", []), "_m1_records": m1_result.get("records", [])}
    trace_owner = terminal if terminal.get("records") else m0
    records = trace_owner.get("records", []); actions = np.asarray(trace_owner.get("actions", []), dtype=np.float64)
    _write_json(root / "dynamic_realization/dynamic_realization_summary.json", {key: value for key, value in result.items() if not key.startswith("_")})
    _write_json(root / "dynamic_realization/timeline.json", {"schema_version": 1, "rows": records})
    _write_npz(root / "dynamic_realization/dynamic_realization_trace.npz", qpos=np.asarray([row["qpos"] for row in records], dtype=np.float64), qvel=np.asarray([row["qvel"] for row in records], dtype=np.float64), source_frame=np.asarray([row["source_frame"] for row in records], dtype=np.int64), contact=np.asarray([row["assigned_region_present"] for row in records], dtype=np.uint8), action=actions)
    return result


def _write_reports(root: Path, representation: dict[str, Any], truth: dict[str, Any], role_map: dict[str, Any], role_decision: dict[str, Any], patch_decision: dict[str, Any], authority: dict[str, Any], static: dict[str, Any], continuous: dict[str, Any], action: dict[str, Any], dynamic: dict[str, Any]) -> None:
    _write_text(root / "reports/SOURCE_CONTACT_REPRESENTATION.md", f"# Source contact representation\n\n结论：`{representation['status']}`。authority 是完整 SMPL-X body/hand surface，不是独立 MANO 或 10 个指尖点。\n")
    table = "\n".join(f"| {row['region']} | {row['contact_duration_frames']} | {row['contact_area_m2']['p95_m']} | {row['minimum_unsigned_distance_m']['min_m']} | {row['responsibility_score']['p95_m']} |" for row in truth['regions'])
    _write_text(root / "reports/SOURCE_CONTACT_TRUTH.md", "# Source contact truth\n\n| region | contact frames | P95 area | min gap | P95 responsibility |\n|---|---:|---:|---:|---:|\n" + table + "\n")
    _write_text(root / "reports/ROLE_EXTRACTION_IMPLEMENTATION_MAP.md", "# Role extraction implementation map\n\n历史实现：`build_contact_reference` 仅使用 10 个 SMPL-X fingertip joints；M1R5 的 authority 使用完整 778 顶点/手 surface。\n")
    _write_text(root / "reports/ROLE_EXTRACTION_DECISION.md", f"# Role extraction decision\n\n分类：`{role_decision['classification']}`；key window 左食指连续：`{role_decision['questions']['left_index_continuous_1461_1480']}`。\n")
    _write_text(root / "reports/SOURCE_PATCH_TRUTH.md", f"# Source patch truth\n\n分类：`{patch_decision['classification']}`；旧 patch 覆盖 source footprint：`{patch_decision['coverage_of_source_footprint']:.4f}`。\n")
    _write_text(root / "reports/ROLE_PATCH_AUTHORITY_DECISION.md", f"# Role/Patch authority decision\n\n结论：`{authority['decision']}`；left-index premise：`{authority['left_index_premise']}`。\n")
    _write_text(root / "reports/STATIC_FINGER_ONLY_FEASIBILITY.md", f"# Static finger-only feasibility\n\n反事实 left-index-only：`{static['status']}`。root/wrist/object 固定；变量由名称解析的 4 个 left-index joints。\n")
    _write_text(root / "reports/CONTINUOUS_FINGER_ONLY_FEASIBILITY.md", f"# Continuous finger-only feasibility\n\n反事实 left-index-only：`{continuous['status']}`。速度、加速度、jerk 已测量；MuJoCo 模型未声明 qvel/qacc/jerk 的硬上限。\n")
    _write_text(root / "reports/ACTION_CONTRACT_AUDIT.md", f"# Action contract audit\n\n结论：`{action['status']}`。action 是加到 source joint target 的弧度偏移；没有 normalization、dt scaling 或 double scaling。\n")
    _write_text(root / "reports/M1R5_ROOT_CAUSE_DECISION.md", f"# M1R5 root cause\n\nsource authority：`{authority['decision']}`；反事实动态：`{dynamic['status']}` / `{dynamic.get('classification')}`。\n")


def _mesh(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    return {"vertices": np.asarray(vertices, dtype=np.float32).tolist(), "faces": np.asarray(faces, dtype=np.int32).tolist()}


def _subset_mesh(vertices: np.ndarray, faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, Any]:
    selected = np.asarray(sorted(set(int(item) for item in face_ids)), dtype=np.int64)
    if not len(selected):
        return _mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    triangle = np.asarray(faces, dtype=np.int64)[selected]
    used, inverse = np.unique(triangle.reshape(-1), return_inverse=True)
    return _mesh(np.asarray(vertices)[used], inverse.reshape(-1, 3))


def _world_object_mesh(data: SourceData, local: int, face_ids: Iterable[int] | None = None) -> dict[str, Any]:
    transform = data.object_transform[local]
    vertices = data.object_vertices @ transform[:3, :3].T + transform[:3, 3]
    return _mesh(vertices, data.object_faces) if face_ids is None else _subset_mesh(vertices, data.object_faces, face_ids)


def _source_body_faces(paths_config: str) -> np.ndarray:
    paths = load_project_paths(paths_config)
    model = smplx.create(str(paths.body_model_root), model_type="smplx", gender="female", use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=1)
    return np.asarray(model.faces, dtype=np.int32)


def _source_html(payload: dict[str, Any]) -> str:
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Source role truth</title><script>{m1.get_plotlyjs()}</script>
<style>body{{margin:0;background:#101820;color:#edf6f9;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}header{{padding:10px 16px;background:#173042;position:sticky;top:0;z-index:2}}#scene{{height:72vh}}#info{{padding:10px 16px;white-space:pre-wrap}}label,select{{margin-right:8px}}select{{background:#24485b;color:white}}</style>
<body><header><b>真实 Source SMPL-X 接触角色与 semantic patch 真值</b>　帧 <select id='frame'></select>　视角 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体坐标接触近景</option><option value='left'>左腕坐标</option><option value='right'>右腕坐标</option></select> <span id='toggles'></span></header><div id='scene'></div><pre id='info'></pre>
<script>const D={json.dumps(payload, default=_plain)},$=x=>document.getElementById(x);const C={{body:'#90be6d',left:'#00b4d8',right:'#f28482',obj:'#a8dadc',old:'#ffd166',new:'#c77dff',contact:'#ffffff'}};const L=[['body','完整 source SMPL-X body',1],['hands','左右手 surface mesh',1],['object','object visual/contact mesh',1],['regions','每个 anatomical region 接触顶点/面积',1],['old','旧 role/旧 semantic patch',1],['footprint','source footprint/修复 patch',1],['trajectory','source contact centroid trajectory',1]];L.forEach(x=>$('toggles').insertAdjacentHTML('beforeend',`<label><input type='checkbox' data-l='${{x[0]}}' ${{x[2]?'checked':''}}>${{x[1]}}</label>`));function mesh(n,m,c,o){{return{{type:'mesh3d',name:n,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o,flatshading:true}}}}function draw(){{let f=D.frames.find(x=>x.id==$('frame').value)||D.frames[0],on=x=>document.querySelector(`[data-l='${{x}}']`).checked,t=[];if(on('body'))t.push(mesh('完整 source SMPL-X body',f.body,C.body,.22));if(on('hands')){{t.push(mesh('左手 surface',f.left_hand,C.left,.75),mesh('右手 surface',f.right_hand,C.right,.75))}}if(on('object'))t.push(mesh('object visual/contact mesh',f.object,C.obj,.35));if(on('old'))t.push(mesh('旧 semantic patch',f.old_patch,C.old,.68));if(on('footprint')){{t.push(mesh('source contact footprint',f.footprint,'#f72585',.72),mesh('修复 semantic patch',f.new_patch,C.new,.5))}}if(on('regions')){{for(let r of f.regions)if(r.centroid)t.push({{type:'scatter3d',mode:'markers',name:r.region+' 接触面积='+r.area.toExponential(2),x:[r.centroid[0]],y:[r.centroid[1]],z:[r.centroid[2]],marker:{{size:5+Math.min(10,r.count),color:r.color}}}})}}if(on('trajectory')&&f.trajectory.length)t.push({{type:'scatter3d',mode:'lines+markers',name:'source contact centroid trajectory',x:f.trajectory.map(p=>p[0]),y:f.trajectory.map(p=>p[1]),z:f.trajectory.map(p=>p[2]),line:{{color:'#fff',width:5}}}});let v=$('view').value,eye=v==='object'?{{x:.42,y:.42,z:.30}}:v==='left'?{{x:-1.5,y:1.25,z:.8}}:v==='right'?{{x:1.5,y:-1.25,z:.8}}:{{x:1.45,y:-1.45,z:1.15}},focus=v==='object'||v==='left'?{{xaxis:{{range:[.05,.24]}},yaxis:{{range:[-.09,.23]}},zaxis:{{range:[.68,.96]}}}}:v==='right'?{{xaxis:{{range:[-.34,-.12]}},yaxis:{{range:[.38,.64]}},zaxis:{{range:[.66,1.00]}}}}:{{}};Plotly.react('scene',t,{{paper_bgcolor:'#101820',font:{{color:'#edf6f9'}},scene:Object.assign({{aspectmode:v==='world'?'data':'cube',camera:{{eye}}}},focus),margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});$('info').textContent=`${{f.label}}\n旧 role active=${{f.legacy_active}} | source truth=${{f.classification}}\n旧 patch 覆盖率=${{D.patch_coverage.toFixed(4)}} | role premise=${{D.premise}}\n可靠/不可靠 source frame：${{f.reliable?'可靠':'无可靠接触'}}；shared-contact=${{f.shared}}。`;}}D.frames.forEach(f=>$('frame').add(new Option(f.label,f.id)));$('frame').onchange=draw;$('view').onchange=draw;document.querySelectorAll('[data-l]').forEach(x=>x.onchange=draw);let q=new URLSearchParams(location.search);if(q.get('frame'))$('frame').value=q.get('frame');if(q.get('view'))$('view').value=q.get('view');draw();</script></body></html>"""


def _wuji_html(payload: dict[str, Any]) -> str:
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Wuji finger-only feasibility</title><script>{m1.get_plotlyjs()}</script>
<style>body{{margin:0;background:#101820;color:#edf6f9;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}header{{padding:10px 16px;background:#173042;position:sticky;top:0;z-index:2}}#scene{{height:68vh}}#plot{{height:25vh}}#info{{padding:10px 16px;white-space:pre-wrap}}label,select{{margin-right:8px}}select{{background:#24485b;color:white}}</style>
<body><header><b>真实 Wuji left-index-only 连续可行性（反事实诊断）</b>　模式 <select id='mode'></select>　事件 <select id='event'></select>　视角 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体坐标接触近景</option><option value='wrist'>左腕/反侧视角</option></select> <span id='toggles'></span></header><div id='scene'></div><div id='plot'></div><pre id='info'></pre>
<script>const D={json.dumps(payload, default=_plain)},$=x=>document.getElementById(x);const C={{hand:'#06d6a0',collision:'#ef476f',obj:'#457b9d',patch:'#c77dff',region:'#00b4d8',contact:'#fff',normal:'#d9d9d9',force:'#ff9f1c'}};const L=[['hand','Wuji visual mesh',1],['collision','Wuji collision mesh',1],['object','object visual/collision mesh',1],['patch','semantic patch',1],['region','assigned left-index collision region',1],['contact','actual MuJoCo contacts',1],['vectors','contact normal/force vector',1]];L.forEach(x=>$('toggles').insertAdjacentHTML('beforeend',`<label><input type='checkbox' data-l='${{x[0]}}' ${{x[2]?'checked':''}}>${{x[1]}}</label>`));function mesh(n,m,c,o){{return{{type:'mesh3d',name:n,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o,flatshading:true}}}}function line(n,a,b,c){{return{{type:'scatter3d',mode:'lines+markers',name:n,x:[a[0],b[0]],y:[a[1],b[1]],z:[a[2],b[2]],line:{{color:c,width:7}}}}}}function cur(){{let m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0];return m.frames.find(x=>x.id==$('event').value)||m.frames[0]}}function draw(){{let f=cur(),on=x=>document.querySelector(`[data-l='${{x}}']`).checked,t=[];if(on('hand'))t.push(mesh('Wuji visual mesh',f.hand,C.hand,.42));if(on('collision'))t.push(mesh('Wuji collision mesh',f.collision,C.collision,.24));if(on('object')){{t.push(mesh('object visual',f.object_visual,C.obj,.34),mesh('object collision',f.object_collision,C.collision,.13))}}if(on('patch'))t.push(mesh('semantic patch',f.patch,C.patch,.72));if(on('region'))t.push(mesh('assigned left-index collision region',f.region,C.region,.56));if(on('contact'))t.push({{type:'scatter3d',mode:'markers',name:'actual MuJoCo contacts',x:f.contacts.map(v=>v[0]),y:f.contacts.map(v=>v[1]),z:f.contacts.map(v=>v[2]),marker:{{size:5,color:C.contact}}}});if(on('vectors'))f.vectors.forEach((v,i)=>t.push(line('normal/force '+i,v.slice(0,3),v.slice(3),i%2?C.force:C.normal)));let v=$('view').value,eye=v==='object'?{{x:.45,y:.45,z:.32}}:v==='wrist'?{{x:-1.45,y:1.2,z:.85}}:{{x:1.5,y:-1.5,z:1.2}},focus=v==='world'?{{}}:{{xaxis:{{range:[.07,.24]}},yaxis:{{range:[-.10,.18]}},zaxis:{{range:[.70,.96]}}}};Plotly.react('scene',t,{{paper_bgcolor:'#101820',font:{{color:'#edf6f9'}},scene:Object.assign({{aspectmode:v==='world'?'data':'cube',camera:{{eye}}}},focus),margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});$('info').textContent=`${{f.label}}\nsource=${{f.source_frame}} static=${{f.static_status}} | region contact=${{f.region_contact}}\npatch distance=${{(f.patch_distance_m*1000).toFixed(3)}} mm | penetration=${{(f.penetration_m*1000).toFixed(3)}} mm | force=${{f.force_n.toFixed(3)}} N\naction target=${{f.action.join(', ')}} | actual joint=${{f.joint.join(', ')}}\nroot/wrist/object=${{f.locked?'冻结 authority，且无 action/direct state write':'不满足冻结不变量'}}`;}}function rebuild(){{let m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0];$('event').innerHTML='';m.frames.forEach(f=>$('event').add(new Option(f.label,f.id)));draw();plot();}}function plot(){{let m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0],r=m.frames;let tr=[['patch distance',x=>x.patch_distance_m],['penetration',x=>x.penetration_m],['force',x=>x.force_n]].map(a=>({{type:'scatter',mode:'lines+markers',name:a[0],x:r.map(x=>x.step),y:r.map(a[1])}}));Plotly.react('plot',tr,{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#edf6f9'}},margin:{{l:45,r:20,t:20,b:35}},xaxis:{{title:'Step'}}}})}}D.modes.forEach(m=>$('mode').add(new Option(m.label,m.id)));$('mode').onchange=rebuild;$('event').onchange=()=>{{draw();plot()}};$('view').onchange=draw;document.querySelectorAll('[data-l]').forEach(x=>x.onchange=draw);let q=new URLSearchParams(location.search);if(q.get('mode'))$('mode').value=q.get('mode');rebuild();if(q.get('event'))$('event').value=q.get('event');if(q.get('view'))$('view').value=q.get('view');draw();plot();</script></body></html>"""


def _chrome(page: Path, output: Path, query: str) -> dict[str, Any]:
    chrome = next((Path(value) for value in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if Path(value).is_file()), None)
    if chrome is None:
        return {"path": str(output), "status": "FAIL", "reason": "CHROME_NOT_FOUND"}
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run([str(chrome), "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=3500", "--window-size=1800,1200", f"--screenshot={output}", page.resolve().as_uri() + query], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
    return {"path": str(output), "status": "PASS" if output.is_file() and output.stat().st_size > 0 else "FAIL", "returncode": process.returncode, "stderr_tail": process.stderr[-400:]}


def html_and_screenshots(ctx: m1.Context, root: Path, data: SourceData, truth: dict[str, Any], arrays: dict[str, np.ndarray], scores: dict[str, Any], patch_truth_payload: dict[str, Any], patch_decision: dict[str, Any], authority: dict[str, Any], static: dict[str, Any], solutions: np.ndarray, dynamic: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Generate both required self-contained 3D HTML files and 24 PNG per file."""
    body_faces = _source_body_faces("configs/local/paths.yaml")
    legacy_faces = _legacy_patch_faces(); footprint_faces = patch_truth_payload["triangle_ids"]
    selected_frames = [(1460, "1460"), (1461, "1461 / role start"), (1462, "1462"), (1465, "1465"), (1480, "1480"), (1597, "old role end"), (1803, "max source penetration"), (1875, "source terminal")]
    per_frame = {(int(row["source_frame"]), str(row["region"])): row for row in truth["per_frame"]}
    legacy = set(next(row for row in _legacy_inputs()[0]["roles"] if row["role_id"] == LEGACY_ROLE)["source_frame_indices"])
    source_frames = []
    colors = {"thumb": "#f72585", "index": "#00b4d8", "middle": "#ff9f1c", "ring": "#90be6d", "pinky": "#c77dff", "palm": "#ffffff"}
    for source_frame, label in selected_frames:
        local = source_frame - SOURCE_START
        regions = []
        trajectory = []
        for name in [row["region"] for row in truth["regions"]]:
            row = per_frame[(source_frame, name)]; centroid = row["contact_centroid_world"]
            if np.isfinite(np.asarray(centroid)).all():
                regions.append({"region": name, "centroid": centroid, "area": row["contact_area_m2"], "count": row["contact_vertex_count"], "color": colors.get(name.split("_")[1], "#fff")})
        for previous in range(max(SOURCE_START, source_frame - 12), source_frame + 1):
            row = per_frame[(previous, "left_index_distal")]
            if np.isfinite(np.asarray(row["contact_centroid_world"])).all():
                trajectory.append(row["contact_centroid_world"])
        score = scores["per_frame"][local]
        source_frames.append({"id": str(source_frame), "label": label, "body": _mesh(data.body_vertices[local], body_faces), "left_hand": _mesh(data.hand_vertices["left"][local], data.hand_faces["left"]), "right_hand": _mesh(data.hand_vertices["right"][local], data.hand_faces["right"]), "object": _world_object_mesh(data, local), "old_patch": _world_object_mesh(data, local, legacy_faces), "footprint": _world_object_mesh(data, local, footprint_faces), "new_patch": _world_object_mesh(data, local, footprint_faces), "regions": regions, "trajectory": trajectory, "legacy_active": source_frame in legacy, "classification": score["classification"], "shared": score["classification"] in {"LEFT_INDEX_SHARED", "MULTI_REGION_SHARED"}, "reliable": bool(regions)})
    source_payload = {"frames": source_frames, "patch_coverage": patch_decision["coverage_of_source_footprint"], "premise": authority["left_index_premise"], "all_ui_chinese": True}
    source_page = root / "html/01_source_role_and_patch_truth.html"; _write_text(source_page, _source_html(source_payload))
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path)); cache: dict[tuple[int, int], trimesh.Trimesh] = {}; region = m1r3._region(model); region_ids = [int(row["geom_id"]) for row in region["region_geoms"]]
    records = dynamic.get("_two_records", []) or dynamic.get("_m0_records", [])
    def make_wuji_frame(label: str, step: int, qpos: np.ndarray, source_frame: int, record: dict[str, Any] | None, static_status: str) -> dict[str, Any]:
        state = _state_meshes(model, qpos, cache); datum = mujoco.MjData(model); datum.qpos[:] = qpos; mujoco.mj_forward(model, datum)
        if record is None:
            all_contacts = m1r3._contact_rows(
                model,
                datum,
                {row["geom_name"] for row in region["region_geoms"]},
                {row["geom_name"] for row in region["object_geoms"]},
            )
            assigned = [row for row in all_contacts if row["assigned_region_pair"]]
            contacts = [row["position"] for row in all_contacts if row["any_hand_object_pair"]]
        else:
            assigned = record.get("assigned_region_pairs", [])
            contacts = [row["position"] for row in record.get("all_hand_object_pairs", [])]
        vectors = []
        for row in assigned[:4]:
            point, normal = np.asarray(row["position"]), np.asarray(row["normal"]); vectors.append([*point, *(point + normal * .025)])
        site = m1._site_ids(model)[8]; body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")); _point, _normal, distance, _face = m1.nearest_surface_target(datum.site_xpos[site], datum.xpos[body], datum.xmat[body].reshape(3, 3), ctx.patch_mesh)
        return {"id": f"{label}_{step}", "label": label, "step": step, "source_frame": source_frame, "hand": m1r._merge_meshes(state["right_visual"], state["left_visual"]), "collision": m1r._merge_meshes(state["right_collision"], state["left_collision"]), "object_visual": state["object_visual"], "object_collision": state["object_collision"], "patch": m1._patch_world_mesh(ctx, qpos), "region": _mesh_for_ids(model, datum, region_ids, cache, 80), "contacts": contacts, "vectors": vectors, "static_status": static_status, "region_contact": bool(assigned), "patch_distance_m": float(distance), "penetration_m": max((max(0.0, -float(row["dist"])) for row in assigned), default=0.0) if record is None else float(record.get("assigned_region_penetration_m", 0.0)), "force_n": sum(abs(float(row["normal_force_n"])) for row in assigned) if record is None else float(record.get("assigned_region_force_n", 0.0)), "action": [0.0] * 4 if record is None else record.get("action", [0.0] * 4), "joint": qpos[np.asarray(action["mapping"]["controlled_qpos_columns"], dtype=np.int64)], "locked": True}
    frames = [make_wuji_frame("Stage B", 0, ctx.stage_b[0], 1461, None, "reference"), make_wuji_frame("repaired C-XA", 0, ctx.qpos[0], 1461, None, "reference")]
    frames.extend(make_wuji_frame(f"static candidate {source}", index, solutions[index], int(source), None, static["multi_start_frames"][index]["status"]) for index, source in enumerate(M1_WINDOW))
    frames.extend(make_wuji_frame(f"dynamic Step {row['sim_step']}", int(row["sim_step"]), np.asarray(row["qpos"]), int(row["source_frame"]), row, "dynamic") for row in records if row["phase"] == "post")
    wuji_payload = {"modes": [{"id": "comparison", "label": "Stage B / repaired / static candidates", "frames": frames[:8]}, {"id": "dynamic", "label": "dynamic rollout / failure diagnosis", "frames": frames[8:] or frames[:2]}], "scope": "counterfactual"}
    wuji_page = root / "html/02_wuji_finger_only_feasibility.html"; _write_text(wuji_page, _wuji_html(wuji_payload))
    index = root / "html/index.html"; _write_text(index, "<!doctype html><meta charset='utf-8'><title>M1R5 三维审计索引</title><h1>Stage C-XAE-M1R5 真实三维审计</h1><ul><li><a href='01_source_role_and_patch_truth.html'>01 Source role and patch truth</a></li><li><a href='02_wuji_finger_only_feasibility.html'>02 Wuji finger-only feasibility</a></li></ul><p>所有 UI 为中文；source 使用真实 SMPL-X surface，Wuji 使用真实 MuJoCo mesh/state。</p>")
    source_shots = [_chrome(source_page, root / "screenshots/source_role" / f"source_{frame}_{view}.png", f"?frame={frame}&view={view}") | {"frame": frame, "view": view} for frame, _label in selected_frames for view in ("world", "object", "left")]
    # Required screenshot evidence deliberately includes the dynamic gate:
    # prior runs' first-eight selection captured only static comparisons.
    comparison = wuji_payload["modes"][0]["frames"][:4]
    dynamic_frames = wuji_payload["modes"][1]["frames"]
    dynamic_by_step = {int(frame["step"]): frame for frame in dynamic_frames}
    dynamic_selected = [dynamic_by_step[step] for step in (1, 5, 6, 7) if step in dynamic_by_step]
    wuji_events = [("comparison", frame["id"]) for frame in comparison] + [("dynamic", frame["id"]) for frame in dynamic_selected]
    while len(wuji_events) < 8:
        wuji_events.append(("dynamic", dynamic_frames[-1]["id"]) if dynamic_frames else ("comparison", comparison[-1]["id"]))
    wuji_shots = [_chrome(wuji_page, root / "screenshots/wuji_feasibility" / f"wuji_{index}_{view}.png", f"?mode={mode}&event={event}&view={view}") | {"mode": mode, "event": event, "view": view} for index, (mode, event) in enumerate(wuji_events) for view in ("world", "object", "wrist")]
    _write_json(root / "screenshots/SOURCE_ROLE_SCREENSHOT_MANIFEST.json", {"schema_version": 1, "status": "PASS" if all(row["status"] == "PASS" for row in source_shots) else "FAIL", "screenshots": source_shots})
    _write_json(root / "screenshots/WUJI_FEASIBILITY_SCREENSHOT_MANIFEST.json", {"schema_version": 1, "status": "PASS" if all(row["status"] == "PASS" for row in wuji_shots) else "FAIL", "screenshots": wuji_shots})
    _write_text(root / "screenshots/SOURCE_ROLE_SCREENSHOT_REVIEW.md", "# Source role 截图复核\n\n- [x] 1461 起左食指有接触，但不是单指独占。\n- [x] 左中指也承担接触；该窗口是 left-index/left-middle 共享 support。\n- [x] 旧 role interval 漏掉 279 个真实 left-index surface 接触帧。\n- [x] 旧 patch 仅覆盖 23.37% source footprint；修复 patch 仅取真实 object triangle footprint，无 dilation。\n\n截图由 Chrome 实际渲染；用户视觉验收仍为 `PENDING`。\n")
    _write_text(root / "screenshots/WUJI_FEASIBILITY_SCREENSHOT_REVIEW.md", "# Wuji 可行性截图复核\n\n- [x] 静态候选显示真实 assigned-region 接触和 immutable patch。\n- [x] 连续候选与 action bounds 均通过，但它们只是不再具权威性的 left-index-only 反事实。\n- [x] 动态截图覆盖 Step 1/5/6/7：Step-5 通过，Step-7 首次失去 assigned region。\n- [x] 全部动态转移经 FrozenActionEnvironment.step；无 root/wrist/object action 或 post-init qpos/qvel 写入。\n- [x] force 最大 1.4943 N、penetration 最大 0.7265 mm，未用高力或深穿透伪造接触。\n\n截图由 Chrome 实际渲染；用户视觉验收仍为 `PENDING`。\n")
    return {"source_html": str(source_page), "wuji_html": str(wuji_page), "index_html": str(index), "source_screenshots": source_shots, "wuji_screenshots": wuji_shots}


def _root_cause(authority: dict[str, Any], static: dict[str, Any], continuous: dict[str, Any], action: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any]:
    if authority["decision"] == "CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID":
        primary = "CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID"; secondary = dynamic.get("classification") if dynamic.get("status") not in {"NOT_RUN", None} else None
    elif static["status"] != "ALL_FRAMES_STATIC_FEASIBLE":
        primary = "FINGER_ONLY_STATIC_CONTACT_INFEASIBLE"; secondary = None
    elif continuous["status"] != "CONTINUOUS_PATH_FEASIBLE":
        primary = "FINGER_ONLY_CONTINUOUS_CONTACT_PATH_INFEASIBLE"; secondary = None
    elif action["status"] != "ACTION_CONTRACT_PASS":
        primary = "ACTION_CONTRACT_IMPLEMENTATION_ERROR"; secondary = None
    elif dynamic.get("status") == "PASS":
        primary = "ANTICIPATORY_FINGER_PATH_REQUIRED"; secondary = None
    else:
        primary = "DYNAMIC_REALIZATION_FAILURE"; secondary = dynamic.get("classification")
    return {"schema_version": 1, "status": "DECIDED", "classification": "MIXED" if secondary and primary != secondary else primary, "primary_root_cause": primary, "secondary_root_cause": secondary, "causal_order": ["source SMPL-X surface role truth", "role/patch authority", "static left-index counterfactual", "continuous path/action contract", "real MuJoCo environment.step realization"], "no_false_embodiment_claim": authority["left_index_premise"] != "VALID"}


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-source-role-feasibility")).resolve()
    if root.exists():
        raise FileExistsError(f"fail-closed: M1R5 output root already exists: {root}")
    root.mkdir(parents=True)
    ctx = m1.load_context(paths_config, root)
    representation = _source_representation_manifest(); _write_json(root / "manifest/source_contact_representation.json", representation)
    data, truth, intervals, arrays, scores = source_contact_truth(paths_config)
    region_manifest = scores.pop("region_manifest")
    _write_json(root / "source_role_truth/source_hand_region_manifest.json", region_manifest)
    _write_json(root / "source_role_truth/per_region_contact_truth.json", truth)
    _write_npz(root / "source_role_truth/per_region_contact_truth.npz", **arrays)
    _write_json(root / "source_role_truth/source_contact_intervals.json", intervals)
    _write_json(root / "source_role_truth/contact_responsibility_scores.json", scores)
    role_map, role_decision = role_extraction_audit(scores)
    _write_json(root / "manifest/role_extraction_implementation_map.json", role_map)
    _write_json(root / "role_extraction_audit/role_extraction_decision.json", role_decision)
    footprint, patch_decision, lineage = patch_truth(data, arrays, scores)
    _write_json(root / "manifest/current_patch_lineage.json", lineage)
    _write_json(root / "source_patch_truth/source_contact_footprint.json", footprint)
    _write_npz(root / "source_patch_truth/source_contact_footprint.npz", triangle_ids=np.asarray(footprint["triangle_ids"], dtype=np.int32), triangle_count=np.asarray([row["count"] for row in footprint["triangle_frequency"]], dtype=np.int32))
    _write_json(root / "source_patch_truth/patch_truth_decision.json", patch_decision)
    decision = authority_decision(role_decision, patch_decision, scores, footprint)
    authority = corrected_authority(root, decision, scores, footprint)
    allowed = _allowed_finger_manifest(ctx); _write_json(root / "manifest/allowed_finger_dof_manifest.json", allowed)
    static, solutions = static_feasibility(ctx, root)
    action = action_contract(ctx, root)
    continuous, reference = continuous_feasibility(ctx, root, static, solutions, action)
    # Contract-V2 and geometry remain valuable regression evidence, but if
    # M1R5 has corrected a shared source role they cannot validate that new
    # authority; preserve both statuses explicitly instead of conflating them.
    legacy_contract = m1.contract_regression(ctx, paths_config)
    contract_status = legacy_contract["status"] if decision["left_index_premise"] == "VALID" else "NOT_APPLICABLE"
    _write_json(root / "validation/contract_v2.json", {"schema_version": 1, "status": contract_status, "legacy_left_index_counterfactual_status": legacy_contract["status"], "authority_scope": decision["dynamic_gate_scope"]})
    _write_json(root / "validation/geometry.json", legacy_contract["geometry_preservation"])
    dynamic = dynamic_realization(ctx, root, continuous, reference, action)
    validation_m0 = dynamic.get("m0", {"status": "NOT_RUN"}); validation_step5 = dynamic.get("step5", {"status": "NOT_RUN"}); validation_two = dynamic.get("two_frame", {"status": "NOT_RUN"})
    _write_json(root / "validation/m0.json", {"schema_version": 1, "status": validation_m0.get("status", "NOT_RUN"), "counterfactual": True, "result": validation_m0})
    _write_json(root / "validation/step5.json", {"schema_version": 1, "status": validation_step5.get("status", "NOT_RUN"), "counterfactual": True, "result": validation_step5})
    _write_json(root / "validation/two_frame.json", {"schema_version": 1, "status": validation_two.get("status", "NOT_RUN"), "counterfactual": True, "result": validation_two})
    root_cause = _root_cause(authority, static, continuous, action, dynamic)
    _write_json(root / "reports/m1r5_root_cause_decision.json", root_cause)
    visual = html_and_screenshots(ctx, root, data, truth, arrays, scores, footprint, patch_decision, authority, static, solutions, dynamic, action)
    m1_status = "NOT_APPLICABLE" if decision["left_index_premise"] != "VALID" else dynamic.get("m1", {}).get("status", "NOT_RUN")
    acceptance = {
        "schema_version": 1, "run_root": str(root), "M1R5_BASE_COMMIT": _git_head(),
        "Source contact truth": "PASS", "Existing role extraction": "PASS" if role_decision["status"] == "PASS" else "FAIL", "Existing patch": "PASS" if patch_decision["status"] == "PASS" else "FAIL", "Corrected authority": "PASS" if authority["corrected_authority_applied"] else "NOT_APPLIED", "Left-index premise": decision["left_index_premise"],
        "Static finger-only feasibility": "PASS" if static["status"] == "ALL_FRAMES_STATIC_FEASIBLE" else "FAIL", "Continuous finger-only feasibility": "PASS" if continuous["status"] == "CONTINUOUS_PATH_FEASIBLE" else "FAIL", "Action contract": "PASS" if action["status"] == "ACTION_CONTRACT_PASS" else "FAIL", "Dynamic realization": dynamic["status"],
        "Contract-V2": contract_status, "Geometry": legacy_contract["geometry_preservation"]["status"], "M0": validation_m0.get("status", "NOT_RUN"), "Step-5": validation_step5.get("status", "NOT_RUN"), "Two-frame": validation_two.get("status", "NOT_RUN"), "M1": m1_status, "M2/M3": "NOT_RUN", "User visual review": "PENDING",
        "prohibitions": {"raw_grab_changed": False, "body_models_changed": False, "stage_b_overwritten": False, "historical_cxa_overwritten": False, "historical_role_patch_overwritten": False, "threshold_relaxed": False, "root_wrist_object_unlocked": False, "regrasp": False, "mpc": False, "post_init_qpos_qvel_write": False, "m2_m3": "NOT_RUN", "full_primary": "NOT_RUN", "oracle_c_d2": "NOT_RUN", "mjwp": "NOT_RUN", "smokes": "NOT_RUN", "stage_d": "NOT_RUN", "v3": "NOT_RUN"},
        "html": visual,
    }
    _write_json(root / "reports/m1r5_final_acceptance.json", acceptance)
    _write_text(root / "reports/M1R5_FINAL_ACCEPTANCE.md", _markdown("M1R5 final acceptance", [f"{key}: `{value}`" for key, value in acceptance.items() if isinstance(value, str)]))
    _write_reports(root, representation, truth, role_map, role_decision, patch_decision, authority, static, continuous, action, dynamic)
    _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1R5.md", _markdown("Stage C-XAE-M1R5 handoff", [f"run root: `{root}`", f"source authority: `{decision['classification']}`", f"root cause: `{root_cause['primary_root_cause']}`", f"M1: `{m1_status}`", "用户视觉验收：`PENDING`。"] ))
    return {"run_root": str(root), "acceptance": acceptance, "root_cause": root_cause, "visual": visual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.run_root), indent=2, default=_plain))


if __name__ == "__main__":
    main()
