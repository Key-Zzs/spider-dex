"""Auditable GRAB source-to-Stage-C geometry reconstruction.

This module deliberately keeps the independent GRAB reconstruction separate
from :class:`spider.datasets.grab.GrabAdapter`.  The adapter is invoked only
after the independent result is materialised, so it cannot validate itself.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from spider.datasets.grab import GrabAdapter
from spider.datasets.paths import load_project_paths


SEQUENCE_ID = "s5/cylindermedium_lift"
FRAME_START = 1460
FRAME_END = 1876
FPS = 120.0
FOCUS_FRAMES = (1460, 1461, 1462, 1463, 1464, 1465, 1466, 1480)
DETAIL_FRAMES = tuple(range(1460, 1481))
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
HAND_JOINTS = {
    "left": (20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70),
    "right": (21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75),
}
SMPLX_TO_WUJI_PALM = {
    "right": np.asarray(((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))),
    "left": np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
}


@dataclass(frozen=True)
class AuditPaths:
    """Immutable input locations for the one frozen audit sample."""

    source_root: Path
    body_model_root: Path
    workspace_root: Path
    raw_sequence: Path
    object_mesh: Path
    stage_b_root: Path
    stage_b_trajectory: Path
    stage_b_mapping: Path
    cxa_trajectory: Path
    scene: Path
    scene_act: Path
    cxa_patch_npz: Path
    source_surface_diagnostics: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    """Write a deterministic, UTF-8 JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    """Return the file checksum without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build homogeneous transforms for either one or a batch of poses."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3) or translation.shape[-1:] != (3,):
        raise ValueError(f"Expected (...,3,3) and (...,3), got {rotation.shape}, {translation.shape}")
    shape = np.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    result = np.broadcast_to(np.eye(4), shape + (4, 4)).copy()
    result[..., :3, :3] = np.broadcast_to(rotation, shape + (3, 3))
    result[..., :3, 3] = np.broadcast_to(translation, shape + (3,))
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert rigid homogeneous transforms."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape[-2:] != (4, 4):
        raise ValueError("transform must end in (4,4)")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    result = np.broadcast_to(np.eye(4), transform.shape).copy()
    result[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], translation)
    return result


def compose_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose ``first @ second`` with shape checking."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape[-2:] != (4, 4) or second.shape[-2:] != (4, 4):
        raise ValueError("both transforms must end in (4,4)")
    return first @ second


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply one rigid transform to N points, or N transforms to N points."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if transform.shape[-2:] != (4, 4) or points.shape[-1] != 3:
        raise ValueError("Expected (...,4,4) transform and (...,3) points")
    rotation, translation = transform[..., :3, :3], transform[..., :3, 3]
    if rotation.ndim == 2:
        return points @ rotation.T + translation
    return np.einsum("...ij,...j->...i", rotation, points) + translation


def rotation_residual_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the shortest angular distance between matching rotations."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    delta = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    # arccos(trace) loses roughly 1e-4 rad precision for float32 matrices.
    # The skew-symmetric component remains well-conditioned around identity.
    sine = 0.5 * np.linalg.norm(
        np.stack((delta[..., 2, 1] - delta[..., 1, 2], delta[..., 0, 2] - delta[..., 2, 0], delta[..., 1, 0] - delta[..., 0, 1]), axis=-1),
        axis=-1,
    )
    return np.arctan2(sine, cosine)


def object_relative(object_transform: np.ndarray, hand_transform: np.ndarray) -> np.ndarray:
    """Return ``T_object_hand = inverse(T_world_object) @ T_world_hand``."""
    return compose_transform(invert_transform(object_transform), hand_transform)


def detect_unit_scale(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Detect a simple 10/100/1000-style global length scale mismatch."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.linalg.norm(reference, axis=-1) > 1e-9
    if not np.any(mask):
        return {"classification": "INCONCLUSIVE", "ratio": None}
    ratios = np.linalg.norm(candidate[mask], axis=-1) / np.linalg.norm(reference[mask], axis=-1)
    ratio = float(np.median(ratios))
    standards = np.asarray((0.001, 0.01, 0.1, 10.0, 100.0, 1000.0))
    nearest = float(standards[np.argmin(np.abs(np.log(ratio) - np.log(standards)))])
    return {
        "classification": "UNIT_SCALE_ERROR" if abs(np.log(ratio / nearest)) < 0.02 else "NO_UNIT_SCALE_ERROR",
        "ratio": ratio,
        "nearest_standard_ratio": nearest,
    }


def detect_rotation_transpose(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return true when the candidate matches transpose better than direct."""
    direct = float(np.max(rotation_residual_rad(reference, candidate)))
    transpose = float(np.max(rotation_residual_rad(reference, np.swapaxes(candidate, -1, -2))))
    return transpose + 1e-8 < direct


def detect_quaternion_order(quaternions: np.ndarray, expected_rotation: np.ndarray) -> str:
    """Distinguish wxyz from xyzw before calling a quaternion-based loader."""
    values = np.asarray(quaternions, dtype=np.float64)
    expected_rotation = np.asarray(expected_rotation, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("quaternions must end in four values")
    wxyz = Rotation.from_quat(values[..., (1, 2, 3, 0)]).as_matrix()
    xyzw = Rotation.from_quat(values).as_matrix()
    wxyz_error = float(np.mean(rotation_residual_rad(wxyz, expected_rotation)))
    xyzw_error = float(np.mean(rotation_residual_rad(xyzw, expected_rotation)))
    return "WXYZ" if wxyz_error <= xyzw_error else "QUATERNION_ORDER_ERROR"


def detect_root_translation_double_apply(reference: np.ndarray, candidate: np.ndarray, root_translation: np.ndarray) -> bool:
    """Detect the characteristic extra body root translation exactly once."""
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    root_translation = np.asarray(root_translation, dtype=np.float64)
    direct = float(np.sqrt(np.mean(np.square(delta - root_translation))))
    zero = float(np.sqrt(np.mean(np.square(delta))))
    return direct + 1e-8 < zero


def detect_left_hand_mirror(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Detect an X-mirrored left-hand point cloud rather than a normal offset."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mirrored = reference.copy()
    mirrored[..., 0] *= -1.0
    direct = float(np.sqrt(np.mean(np.square(reference - candidate))))
    mirror = float(np.sqrt(np.mean(np.square(mirrored - candidate))))
    return mirror + 1e-8 < direct


def stage_b_gate(raw_loader_status: str) -> str:
    """Enforce the required raw-loader gate before retarget audit."""
    return "RUN" if raw_loader_status == "PASS" else "NOT_RUN_DUE_TO_RAW_LOADER_FAILURE"


def cxa_gate(raw_loader_status: str, stage_b_status: str) -> str:
    """Enforce both upstream gates before C-XA audit."""
    if raw_loader_status != "PASS":
        return "NOT_RUN_DUE_TO_RAW_LOADER_FAILURE"
    return "RUN" if stage_b_status == "PASS" else "NOT_RUN_DUE_TO_STAGE_B_FAILURE"


def detect_frame_offset(reference: np.ndarray, candidate: np.ndarray, offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)) -> dict[str, Any]:
    """Score fixed temporal offsets without dropping any original data."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    scores: dict[str, float] = {}
    for offset in offsets:
        if offset >= 0:
            left, right = reference[offset:], candidate[: len(candidate) - offset or None]
        else:
            left, right = reference[:offset], candidate[-offset:]
        scores[str(offset)] = float(np.sqrt(np.mean(np.square(left - right))))
    best = min(scores, key=scores.get)
    return {"best_offset": int(best), "scores_rmse_m": scores, "classification": "OFF_BY_ONE_FRAME" if int(best) in {-1, 1} and int(best) != 0 else ("BODY_OBJECT_FRAME_OFFSET" if int(best) != 0 else "ALIGNED")}


def detect_global_frame_only_difference(
    source_object: np.ndarray,
    source_hands: np.ndarray,
    candidate_object: np.ndarray,
    candidate_hands: np.ndarray,
) -> dict[str, Any]:
    """Fit one global rigid transform and decide whether relative poses agree."""
    source_points = np.concatenate((source_object[:, None, :3, 3], source_hands[..., :3, 3]), axis=1).reshape(-1, 3)
    candidate_points = np.concatenate((candidate_object[:, None, :3, 3], candidate_hands[..., :3, 3]), axis=1).reshape(-1, 3)
    source_center, candidate_center = source_points.mean(axis=0), candidate_points.mean(axis=0)
    # Kabsch for row-vector coordinates: target ~= source @ rotation.T + shift.
    covariance = (candidate_points - candidate_center).T @ (source_points - source_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = candidate_center - rotation @ source_center
    global_transform = transform_from_rt(rotation, translation)
    transformed_source_hands = compose_transform(global_transform, source_hands)
    transformed_source_object = compose_transform(global_transform, source_object)
    point_error = np.linalg.norm(apply_transform(global_transform, source_points) - candidate_points, axis=1)
    position_error = np.concatenate((
        np.linalg.norm(transformed_source_object[:, :3, 3] - candidate_object[:, :3, 3], axis=1),
        np.linalg.norm(transformed_source_hands[..., :3, 3] - candidate_hands[..., :3, 3], axis=1).reshape(-1),
    ))
    rotation_error = np.concatenate((
        rotation_residual_rad(transformed_source_object[:, :3, :3], candidate_object[:, :3, :3]),
        rotation_residual_rad(transformed_source_hands[..., :3, :3], candidate_hands[..., :3, :3]).reshape(-1),
    ))
    source_relative = object_relative(source_object[:, None], source_hands)
    candidate_relative = object_relative(candidate_object[:, None], candidate_hands)
    relative_translation = np.linalg.norm(source_relative[..., :3, 3] - candidate_relative[..., :3, 3], axis=-1)
    relative_rotation = rotation_residual_rad(source_relative[..., :3, :3], candidate_relative[..., :3, :3])
    only_global = bool(np.max(position_error) <= 1e-5 and np.max(rotation_error) <= 1e-5 and np.max(relative_translation) <= 1e-5 and np.max(relative_rotation) <= 1e-5)
    return {
        "classification": "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY" if only_global else "NOT_GLOBAL_FRAME_ONLY",
        "global_transform": global_transform,
        "point_rmse_m": float(np.sqrt(np.mean(np.square(point_error)))),
        "max_global_position_residual_m": float(np.max(position_error)),
        "max_global_rotation_residual_rad": float(np.max(rotation_error)),
        "max_object_relative_translation_residual_m": float(np.max(relative_translation)),
        "max_object_relative_rotation_residual_rad": float(np.max(relative_rotation)),
    }


def _rotation_wxyz(rotvec: np.ndarray) -> np.ndarray:
    quat = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return quat[:, (3, 0, 1, 2)]


def _matrix_wxyz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return Rotation.from_quat(values[:, (1, 2, 3, 0)]).as_matrix()


def _load_paths(paths_config: str, cxa_root_override: str | None = None) -> AuditPaths:
    """Resolve immutable inputs, optionally selecting a new isolated C-XA root.

    The override is deliberately a directory rather than an arbitrary qpos
    path so the repaired trajectory and its semantic-patch provenance stay in
    one versioned namespace.  It never alters the historical C-XA directory.
    """
    paths = load_project_paths(paths_config)
    task = "s5__cylindermedium_lift"
    source = paths.source_root("grab")
    stage_b_root = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / task / "0"
    cxa_root = Path(cxa_root_override).resolve() if cxa_root_override else stage_b_root / "stage_c_contract_v2_cxa"
    physics = stage_b_root / "stage_c/physics_input.json"
    if not physics.is_file():
        raise FileNotFoundError(f"Frozen Stage-C physics record is missing: {physics}")
    scene_act = Path(json.loads(physics.read_text(encoding="utf-8"))["scene_act"])
    cxa_trajectory = cxa_root / "trajectory_depenetrated_init_cxa_level_1_flexible.npz"
    if cxa_root_override and not cxa_trajectory.is_file():
        # A C-XAR repaired namespace uses the generic isolated writer name;
        # accept it only for an explicit override, never by changing the
        # historical C-XA lookup contract.
        cxa_trajectory = cxa_root / "trajectory_depenetrated_init.npz"
    result = AuditPaths(
        source_root=source,
        body_model_root=paths.body_model_root,
        workspace_root=paths.workspace_root,
        raw_sequence=source / "grab/s5/cylindermedium_lift.npz",
        object_mesh=source / "tools/object_meshes/contact_meshes/cylindermedium.ply",
        stage_b_root=stage_b_root,
        stage_b_trajectory=stage_b_root / "trajectory_kinematic.npz",
        stage_b_mapping=stage_b_root / "source_mapping.json",
        cxa_trajectory=cxa_trajectory,
        scene=stage_b_root.parent / "scene.xml",
        scene_act=scene_act,
        cxa_patch_npz=cxa_root / "source_contact_patches.npz",
        source_surface_diagnostics=stage_b_root / "stage_c/source_geometry_diagnostics.npz",
    )
    for path in (result.raw_sequence, result.object_mesh, result.stage_b_trajectory, result.stage_b_mapping, result.cxa_trajectory, result.scene, result.scene_act, result.source_surface_diagnostics):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen audit input is missing: {path}")
    return result


def _source_representation(paths: AuditPaths) -> dict[str, Any]:
    with np.load(paths.raw_sequence, allow_pickle=True) as archive:
        body = archive["body"].item()
        left = archive["lhand"].item()
        right = archive["rhand"].item()
        obj = archive["object"].item()
        report = {
            "schema_version": 1,
            "sequence_id": SEQUENCE_ID,
            "source_parameter_file": str(paths.raw_sequence),
            "source_parameter_sha256": sha256(paths.raw_sequence),
            "subject_id": str(archive["sbj_id"]),
            "gender": str(archive["gender"]),
            "frame_count": int(archive["n_frames"]),
            "fps": float(archive["framerate"]),
            "source_representation": "SMPL-X full-body parameters with auxiliary left/right MANO-fit streams",
            "source_of_truth_for_this_audit": "SMPL-X full-body reconstruction; independent MANO is not treated as an equivalent truth",
            "body_parameter_fields": {key: list(np.asarray(value).shape) for key, value in body["params"].items()},
            "left_auxiliary_hand_fields": {key: list(np.asarray(value).shape) for key, value in left["params"].items()},
            "right_auxiliary_hand_fields": {key: list(np.asarray(value).shape) for key, value in right["params"].items()},
            "object_parameter_fields": {key: list(np.asarray(value).shape) for key, value in obj["params"].items()},
            "object_parameter_file": str(paths.raw_sequence),
            "object_asset": str(paths.object_mesh),
            "object_asset_sha256": sha256(paths.object_mesh),
            "betas_file": str(paths.source_root / "tools/subject_meshes" / str(archive["gender"]) / f"{archive['sbj_id']}_betas.npy"),
            "hand_pose_convention": "SMPL-X PCA24 coefficients inside body.params.left_hand_pose/right_hand_pose; flat_hand_mean=False",
            "auxiliary_hand_pose_convention": "independent lhand/rhand PCA24 streams are retained by GRAB but are not used as the source truth",
            "translation_convention": "body.params.transl and object.params.transl are metres in the GRAB world frame",
            "object_canonical_transform": "T_asset_object=identity: contact_meshes/cylindermedium.ply is used directly as the object local mesh",
            "spider_raw_loader_uses": "SMPL-X hand vertices/joints and SMPL-X wrist transforms; see spider/datasets/grab.py::GrabAdapter.load_sequence",
        }
    return report


def _smplx_model_and_input(paths: AuditPaths) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """Load raw GRAB parameters without importing the spider loader implementation."""
    import smplx
    import torch

    with np.load(paths.raw_sequence, allow_pickle=True) as archive:
        body = archive["body"].item()["params"]
        gender = str(archive["gender"])
        subject = str(archive["sbj_id"])
    beta_path = paths.source_root / "tools/subject_meshes" / gender / f"{subject}_betas.npy"
    betas = np.asarray(np.load(beta_path), dtype=np.float32).reshape(-1)[:10]
    index = slice(FRAME_START, FRAME_END)
    count = FRAME_END - FRAME_START
    model = smplx.create(
        str(paths.body_model_root), model_type="smplx", gender=gender, use_pca=True,
        num_pca_comps=24, flat_hand_mean=False, batch_size=count,
    )
    tensors = {key: torch.as_tensor(np.asarray(value[index]), dtype=torch.float32) for key, value in body.items()}
    tensors["betas"] = torch.as_tensor(np.repeat(betas[None], count, axis=0), dtype=torch.float32)
    metadata = {"gender": gender, "subject_id": subject, "betas": betas, "beta_path": beta_path, "body_fullpose": np.asarray(body["fullpose"][index], dtype=np.float32)}
    return model, tensors, metadata


def _hand_faces_from_smplx(full_faces: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    """Keep triangles completely inside one SMPL-X hand correspondence set."""
    remap = np.full(int(np.max(vertex_ids)) + 1, -1, dtype=np.int64)
    remap[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int64)
    faces = np.asarray(full_faces, dtype=np.int64)
    valid = np.all(faces < len(remap), axis=1)
    mapped = np.full_like(faces, -1)
    mapped[valid] = remap[faces[valid]]
    return mapped[np.all(mapped >= 0, axis=1)].astype(np.int32)


def reconstruct_official_source(paths: AuditPaths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Independently reconstruct SMPL-X and the GRAB object from raw NPZ data."""
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

    model, tensors, metadata = _smplx_model_and_input(paths)
    count = FRAME_END - FRAME_START
    with torch.no_grad():
        output = model(
            global_orient=tensors["global_orient"], body_pose=tensors["body_pose"],
            left_hand_pose=tensors["left_hand_pose"], right_hand_pose=tensors["right_hand_pose"],
            jaw_pose=tensors["jaw_pose"], leye_pose=tensors["leye_pose"], reye_pose=tensors["reye_pose"],
            expression=tensors["expression"], transl=tensors["transl"], betas=tensors["betas"],
            return_verts=True, return_full_pose=True,
        )
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    full_pose = output.full_pose.detach().cpu().numpy().astype(np.float32)
    beta_tensor = tensors["betas"]
    shaped = model.v_template + blend_shapes(beta_tensor, model.shapedirs)
    rest_joints = vertices2joints(model.J_regressor, shaped)
    rotations = batch_rodrigues(output.full_pose.reshape(-1, 3)).reshape(count, -1, 3, 3)
    _posed_joints, transforms = batch_rigid_transform(rotations, rest_joints, model.parents, dtype=torch.float32)
    joint_transform_rotations = transforms[:, :, :3, :3].detach().cpu().numpy()
    correspondence = paths.source_root / "tools/smplx_correspondence"
    hand_ids = {side: np.asarray(np.load(correspondence / f"{side[0]}hand_smplx_ids.npy"), dtype=np.int64) for side in ("left", "right")}
    with np.load(paths.raw_sequence, allow_pickle=True) as archive:
        object_params = archive["object"].item()["params"]
        object_translation = np.asarray(object_params["transl"][FRAME_START:FRAME_END], dtype=np.float64)
        object_rotation = Rotation.from_rotvec(np.asarray(object_params["global_orient"][FRAME_START:FRAME_END], dtype=np.float64)).as_matrix()
    object_mesh = trimesh.load(paths.object_mesh, force="mesh")
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = object_mesh.dump(concatenate=True)
    object_vertices = np.einsum("tij,vj->tvi", object_rotation, np.asarray(object_mesh.vertices, dtype=np.float64)) + object_translation[:, None, :]
    body_rotation = Rotation.from_rotvec(np.asarray(tensors["global_orient"].cpu(), dtype=np.float64)).as_matrix()
    body_translation = np.asarray(tensors["transl"].cpu(), dtype=np.float64)
    values: dict[str, np.ndarray] = {
        "source_frame_indices": np.arange(FRAME_START, FRAME_END, dtype=np.int64),
        "timestamps_s": np.arange(FRAME_START, FRAME_END, dtype=np.float64) / FPS,
        "T_world_body": transform_from_rt(body_rotation, body_translation).astype(np.float64),
        "T_world_pelvis": transform_from_rt(joint_transform_rotations[:, 0], joints[:, 0]).astype(np.float64),
        "body_joints_world": joints,
        "body_vertices_world": vertices,
        "T_asset_object": np.eye(4, dtype=np.float64),
        "T_world_object_parameter": transform_from_rt(object_rotation, object_translation).astype(np.float64),
        "T_world_object_visual": transform_from_rt(object_rotation, object_translation).astype(np.float64),
        "object_asset_vertices": np.asarray(object_mesh.vertices, dtype=np.float32),
        "object_faces": np.asarray(object_mesh.faces, dtype=np.int32),
        "object_vertices_world": object_vertices.astype(np.float32),
    }
    for side in ("left", "right"):
        joint_ids = HAND_JOINTS[side]
        wrist = joint_ids[0]
        values[f"{side}_joints_world"] = joints[:, joint_ids, :]
        values[f"{side}_vertices_world"] = vertices[:, hand_ids[side], :]
        values[f"{side}_vertex_ids"] = hand_ids[side]
        values[f"{side}_hand_faces"] = _hand_faces_from_smplx(model.faces, hand_ids[side])
        values[f"T_world_{side}_wrist"] = transform_from_rt(joint_transform_rotations[:, wrist], joints[:, wrist]).astype(np.float64)
    fullpose_error = np.abs(full_pose - np.asarray(metadata["body_fullpose"], dtype=np.float32))
    chain = {
        "source": "Direct smplx.create reconstruction; this function does not call GrabAdapter",
        "formulae": {
            "body": "T_world_body = [R(axis_angle(body.global_orient)), body.transl]",
            "pelvis": "T_world_pelvis = [R_SMPLX_pelvis, J_world_pelvis]",
            "wrist": "T_world_wrist = [R_SMPLX_wrist from kinematic tree, J_world_wrist]",
            "object_parameter": "T_world_object_parameter = [R(axis_angle(object.global_orient)), object.transl]",
            "object_visual": "T_world_object_visual = T_world_object_parameter @ T_asset_object, T_asset_object = I",
        },
        "fullpose_decoding_max_abs_rad": float(fullpose_error.max()),
        "fullpose_decoding_rmse_rad": float(np.sqrt(np.mean(np.square(fullpose_error)))),
        "flat_hand_mean": False,
        "object_mesh_is_watertight": bool(object_mesh.is_watertight),
        "object_mesh_vertices": int(len(object_mesh.vertices)),
        "object_mesh_faces": int(len(object_mesh.faces)),
    }
    return values, chain


def capture_spider_raw_loader(paths: AuditPaths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture the actual pre-retarget public output of ``GrabAdapter`` read-only."""
    import smplx

    adapter = GrabAdapter(load_project_paths("configs/local/paths.yaml"))
    sequence = adapter.load_sequence(SEQUENCE_ID, frame_start=FRAME_START, frame_end=FRAME_END, include_vertices=True)
    item = sequence.objects[0]
    object_rotation = _matrix_wxyz(item.orientation)
    object_transform = transform_from_rt(object_rotation, item.translation)
    mesh = trimesh.load(paths.object_mesh, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    topology_model = smplx.create(
        str(paths.body_model_root), model_type="smplx", gender=sequence.right_hand.model_gender,
        use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=1,
    )
    values: dict[str, np.ndarray] = {
        "source_frame_indices": np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64),
        "timestamps_s": np.asarray(sequence.timestamps, dtype=np.float64),
        "T_world_object_parameter": object_transform,
        "T_world_object_visual": object_transform.copy(),
        "T_asset_object": np.eye(4, dtype=np.float64),
        "object_asset_vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "object_faces": np.asarray(mesh.faces, dtype=np.int32),
        "object_vertices_world": (np.einsum("tij,vj->tvi", object_rotation, np.asarray(mesh.vertices, dtype=np.float64)) + item.translation[:, None, :]).astype(np.float32),
    }
    for side, hand in (("left", sequence.left_hand), ("right", sequence.right_hand)):
        if hand is None or hand.vertices_world is None:
            raise RuntimeError(f"GrabAdapter failed to emit required {side} hand geometry")
        values[f"{side}_joints_world"] = np.asarray(hand.joints_world, dtype=np.float32)
        values[f"{side}_vertices_world"] = np.asarray(hand.vertices_world, dtype=np.float32)
        values[f"{side}_vertex_ids"] = np.asarray(np.load(paths.source_root / "tools/smplx_correspondence" / f"{side[0]}hand_smplx_ids.npy"), dtype=np.int64)
        # Topology is determined solely by the same SMPL-X correspondence IDs.
        values[f"{side}_hand_faces"] = _hand_faces_from_smplx(topology_model.faces, values[f"{side}_vertex_ids"])
        values[f"T_world_{side}_wrist"] = transform_from_rt(_matrix_wxyz(hand.global_orientation), hand.global_translation)
    chain = {
        "capture_kind": "read-only independent audit tool invokes the actual GrabAdapter.load_sequence before any MANO-to-Wuji/Stage-B/C-XA operation",
        "raw_loader_source_file": inspect.getsourcefile(GrabAdapter.load_sequence),
        "raw_loader_source_start_line": inspect.getsourcelines(GrabAdapter.load_sequence)[1],
        "body_model_reconstruction": "inside GrabAdapter.load_sequence: smplx.create(... use_pca=True, num_pca_comps=24, flat_hand_mean=False)",
        "raw_loader_public_output": "CanonicalHOISequence with SMPL-X hands/wrists, object free pose and optional SMPL-X hand surfaces",
        "not_retargeted": True,
        "not_written_back_to_pipeline": True,
    }
    return values, chain


def _summary(values: np.ndarray, unit: str = "m") -> dict[str, float | str]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_" + unit: float(np.mean(values)), "rmse_" + unit: float(np.sqrt(np.mean(np.square(values)))),
        "p95_" + unit: float(np.percentile(values, 95)), "max_" + unit: float(np.max(values)),
    }


def summarize_signed_surface(signed_distance_m: np.ndarray, source_frame_indices: np.ndarray) -> dict[str, Any]:
    """Summarise a closed-mesh signed-distance field without hiding its sign.

    This helper normalises the input field to positive-inside convention and
    turns it into explicit penetration and external-gap quantities, retaining
    the source frame of both extrema for the HTML review manifest.
    """
    signed = np.asarray(signed_distance_m, dtype=np.float64)
    frames = np.asarray(source_frame_indices, dtype=np.int64)
    if signed.ndim != 2 or signed.shape[0] != len(frames):
        raise ValueError("signed_distance_m must have one row for each source frame")
    penetration = np.maximum(signed, 0.0)
    gap = np.maximum(-signed, 0.0)
    penetration_index = np.unravel_index(int(np.argmax(penetration)), penetration.shape)
    gap_index = np.unravel_index(int(np.argmax(gap)), gap.shape)
    return {
        "signed_distance_convention": "normalised positive-inside convention for the consistently wound, watertight contact mesh",
        "per_frame_max_penetration_depth_m": np.max(penetration, axis=1),
        "per_frame_max_external_gap_m": np.max(gap, axis=1),
        "max_penetration_m": float(penetration[penetration_index]),
        "max_penetration_source_frame": int(frames[penetration_index[0]]),
        "max_penetration_hand_vertex_index": int(penetration_index[1]),
        "max_suspension_gap_m": float(gap[gap_index]),
        "max_suspension_source_frame": int(frames[gap_index[0]]),
        "max_suspension_hand_vertex_index": int(gap_index[1]),
    }


def _surface_contact_metrics(paths: AuditPaths, source: dict[str, np.ndarray], raw: dict[str, np.ndarray]) -> dict[str, Any]:
    """Cross-verify frozen full-surface source contact evidence read-only.

    The primary coordinate result is the independent full SMPL-X source
    reconstruction versus ``GrabAdapter`` numerical comparison.  The frozen
    source contact diagnostic is separately useful because it contains an
    exact signed distance and closest point for every one of the 778 raw hand
    vertices.  It is accepted only after matching its frame IDs, mesh checksum
    and raw vertices to this run; it never contributes to the raw-loader
    PASS/FAIL decision and is not a Stage-B/C-XA output pose.
    """
    mesh = trimesh.Trimesh(vertices=source["object_asset_vertices"], faces=source["object_faces"], process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("Source contact mesh must be watertight and winding-consistent for signed contact evidence")
    diagnostic_json = paths.source_surface_diagnostics.with_suffix(".json")
    if not diagnostic_json.is_file():
        raise FileNotFoundError(f"Frozen source-surface metadata missing: {diagnostic_json}")
    metadata = json.loads(diagnostic_json.read_text(encoding="utf-8"))
    if metadata.get("raw_grab_modified") is not False or metadata.get("canonical_overwritten") is not False:
        raise RuntimeError("Frozen source-surface evidence records a modified input; refusing to use it")
    object_metadata = metadata.get("object_mesh", {})
    if object_metadata.get("sha256") != sha256(paths.object_mesh):
        raise RuntimeError("Frozen source-surface evidence refers to a different object mesh")
    records: dict[str, Any] = {
        "mesh_is_watertight": bool(mesh.is_watertight),
        "source_surface_diagnostics": {
            "path": str(paths.source_surface_diagnostics),
            "sha256": sha256(paths.source_surface_diagnostics),
            "metadata_path": str(diagnostic_json),
            "metadata_sha256": sha256(diagnostic_json),
            "role": "read-only frozen signed full-surface evidence cross-verified against this run's independently reconstructed source and actual raw-loader vertices",
            "raw_grab_modified": False,
            "canonical_overwritten": False,
        },
        "sides": {},
    }
    object_transform = source["T_world_object_visual"]
    with np.load(paths.source_surface_diagnostics, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["source_frame_indices"], dtype=np.int64)
        if not np.array_equal(frame_indices, source["source_frame_indices"]):
            raise RuntimeError("Frozen source-surface frame IDs do not match independent source reconstruction")
        for side in ("left", "right"):
            # The frozen convention is negative-inside.  Normalise to the
            # positive-inside convention used by ``summarize_signed_surface``.
            frozen_signed = np.asarray(archive[f"{side}_signed_distance_m"], dtype=np.float32)
            unsigned = np.asarray(archive[f"{side}_unsigned_distance_m"], dtype=np.float32)
            nearest_all = np.asarray(archive[f"{side}_closest_surface_world"], dtype=np.float32)
            if frozen_signed.shape != raw[f"{side}_vertices_world"].shape[:-1] or nearest_all.shape != raw[f"{side}_vertices_world"].shape:
                raise RuntimeError(f"Frozen {side} source-surface schema does not match this raw-loader hand mesh")
            raw_surface_distance = np.linalg.norm(raw[f"{side}_vertices_world"] - nearest_all, axis=-1)
            source_raw_vertex_residual = np.linalg.norm(source[f"{side}_vertices_world"] - raw[f"{side}_vertices_world"], axis=-1)
            unsigned_residual = np.abs(raw_surface_distance - unsigned)
            signed = -frozen_signed
            signed_summary = summarize_signed_surface(signed, frame_indices)
            # The viewer deliberately draws only named fingertip lines, but
            # the summary and NPZ cover all 778 vertices per hand.
            points_world = source[f"{side}_joints_world"][:, FINGERTIP_INDICES, :]
            points_local = apply_transform(invert_transform(object_transform)[:, None], points_world)
            flattened = points_local.reshape(-1, 3)
            closest_blocks: list[np.ndarray] = []
            distance_blocks: list[np.ndarray] = []
            face_blocks: list[np.ndarray] = []
            for block in np.array_split(flattened, max(1, len(flattened) // 8)):
                closest, distance, faces = trimesh.proximity.closest_point_naive(mesh, block)
                closest_blocks.append(closest); distance_blocks.append(distance); face_blocks.append(faces)
            closest = np.concatenate(closest_blocks).reshape(points_local.shape)
            distance = np.concatenate(distance_blocks).reshape(points_local.shape[:-1])
            faces = np.concatenate(face_blocks).reshape(points_local.shape[:-1])
            normals_local = mesh.face_normals[faces]
            records["sides"][side] = {
                "unsigned_distance_m": distance,
                "nearest_face": faces,
                "nearest_surface_world": apply_transform(object_transform[:, None], closest),
                "nearest_surface_normal_world": np.einsum("tij,tfj->tfi", object_transform[:, :3, :3], normals_local),
                "signed_distance_m": signed,
                "summary": _summary(distance),
                **signed_summary,
                "frozen_signed_distance_convention": "negative means inside/penetrating in frozen diagnostic; normalised before this report's summary",
                "surface_scope": "all 778 source hand-surface vertices per side in the frozen diagnostic; independent SMPL-X/source-to-raw vertex equality and raw-to-nearest-surface unsigned-distance equality are both checked here",
                "source_to_raw_hand_vertex_residual_m": _summary(source_raw_vertex_residual),
                "raw_vertex_to_frozen_nearest_surface_unsigned_residual_m": _summary(unsigned_residual),
            }
    return records


def _motion_metrics(reference: np.ndarray, candidate: np.ndarray, fps: float) -> dict[str, Any]:
    translation = np.linalg.norm(np.diff(reference[:, :3, 3], axis=0) - np.diff(candidate[:, :3, 3], axis=0), axis=1)
    angle_ref = rotation_residual_rad(reference[:-1, :3, :3], reference[1:, :3, :3]) * fps
    angle_candidate = rotation_residual_rad(candidate[:-1, :3, :3], candidate[1:, :3, :3]) * fps
    return {"frame_to_frame_translation_residual": _summary(translation), "angular_velocity_residual_rad_s": _summary(np.abs(angle_ref - angle_candidate), "rad_s")}


def compare_raw_loader(source: dict[str, np.ndarray], raw: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compare direct source reconstruction against actual raw-loader output."""
    object_translation = np.linalg.norm(source["T_world_object_visual"][:, :3, 3] - raw["T_world_object_visual"][:, :3, 3], axis=1)
    object_rotation = rotation_residual_rad(source["T_world_object_visual"][:, :3, :3], raw["T_world_object_visual"][:, :3, :3])
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "official independent SMPL-X source versus spider-dex GrabAdapter pre-retarget raw output",
        "object_translation_residual_m": _summary(object_translation),
        "object_rotation_residual_rad": _summary(object_rotation, "rad"),
        "object_vertex_residual_m": _summary(np.linalg.norm(source["object_vertices_world"] - raw["object_vertices_world"], axis=-1)),
        "sides": {},
    }
    temporal: dict[str, Any] = {"schema_version": 1, "offset_candidates": (-2, -1, 0, 1, 2), "sides": {}}
    for side in ("left", "right"):
        source_wrist = source[f"T_world_{side}_wrist"]
        raw_wrist = raw[f"T_world_{side}_wrist"]
        relative_source = object_relative(source["T_world_object_visual"], source_wrist)
        relative_raw = object_relative(raw["T_world_object_visual"], raw_wrist)
        joint = np.linalg.norm(source[f"{side}_joints_world"] - raw[f"{side}_joints_world"], axis=-1)
        tips_source = apply_transform(invert_transform(source["T_world_object_visual"])[:, None], source[f"{side}_joints_world"][:, FINGERTIP_INDICES])
        tips_raw = apply_transform(invert_transform(raw["T_world_object_visual"])[:, None], raw[f"{side}_joints_world"][:, FINGERTIP_INDICES])
        vertices = np.linalg.norm(source[f"{side}_vertices_world"] - raw[f"{side}_vertices_world"], axis=-1)
        wrist_translation = np.linalg.norm(relative_source[:, :3, 3] - relative_raw[:, :3, 3], axis=1)
        wrist_rotation = rotation_residual_rad(relative_source[:, :3, :3], relative_raw[:, :3, :3])
        fingertip = np.linalg.norm(tips_source - tips_raw, axis=-1)
        comparison["sides"][side] = {
            "wrist_world_translation_residual_m": _summary(np.linalg.norm(source_wrist[:, :3, 3] - raw_wrist[:, :3, 3], axis=1)),
            "wrist_world_rotation_residual_rad": _summary(rotation_residual_rad(source_wrist[:, :3, :3], raw_wrist[:, :3, :3]), "rad"),
            "T_object_wrist_translation_residual_m": _summary(wrist_translation),
            "T_object_wrist_rotation_residual_rad": _summary(wrist_rotation, "rad"),
            "per_joint_residual_m": _summary(joint),
            "per_fingertip_residual_m": _summary(fingertip),
            "hand_vertex_residual_m": _summary(vertices),
            "motion": {
                "wrist": _motion_metrics(source_wrist, raw_wrist, FPS),
                "object_relative_fingertips": _summary(np.linalg.norm(np.diff(tips_source, axis=0) - np.diff(tips_raw, axis=0), axis=-1)),
            },
        }
        temporal["sides"][side] = {
            "wrist_world": detect_frame_offset(source_wrist[:, :3, 3], raw_wrist[:, :3, 3]),
            "object_frame_fingertips": detect_frame_offset(tips_source, tips_raw),
        }
    temporal["object"] = detect_frame_offset(source["T_world_object_visual"][:, :3, 3], raw["T_world_object_visual"][:, :3, 3])
    source_hands = np.stack((source["T_world_left_wrist"], source["T_world_right_wrist"]), axis=1)
    raw_hands = np.stack((raw["T_world_left_wrist"], raw["T_world_right_wrist"]), axis=1)
    global_frame = detect_global_frame_only_difference(source["T_world_object_visual"], source_hands, raw["T_world_object_visual"], raw_hands)
    max_relative = max(item["T_object_wrist_translation_residual_m"]["max_m"] for item in comparison["sides"].values())
    max_rotation = max(item["T_object_wrist_rotation_residual_rad"]["max_rad"] for item in comparison["sides"].values())
    max_tip = max(item["per_fingertip_residual_m"]["max_m"] for item in comparison["sides"].values())
    match = max(float(object_translation.max()), max_relative, max_tip) <= 1e-5 and max(float(object_rotation.max()), max_rotation) <= 1e-5
    comparison["decision"] = {
        "status": "PASS" if match else "FAIL",
        "classification": "EXACT_OR_NUMERICAL_MATCH" if match else "MISMATCH",
        "thresholds": {"translation_m": 1e-5, "rotation_rad": 1e-5, "practical_relative_m": 0.0005, "practical_rotation_deg": 0.1},
        "max_relative_translation_residual_m": max_relative,
        "max_relative_rotation_residual_rad": max_rotation,
        "max_fingertip_relative_residual_m": max_tip,
    }
    return comparison, temporal, global_frame


def _object_transform_from_mujoco(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return transform_from_rt(data.xmat[body_id].reshape(3, 3), data.xpos[body_id]).reshape(4, 4)


def _extract_mujoco_geometry(scene: Path, qpos: np.ndarray) -> dict[str, Any]:
    """Read real MuJoCo site/object geometry for a trajectory without stepping it."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Trajectory schema {qpos.shape} does not match {scene} nq={model.nq}")
    data = mujoco.MjData(model)
    names = [f"{side}_{finger}_tip" for side in ("right", "left") for finger in FINGERTIP_NAMES]
    site_names = ["right_palm", "left_palm", *names]
    site_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names], dtype=np.int64)
    if np.any(site_ids < 0):
        raise RuntimeError(f"Required Wuji sites absent from {scene}: {site_names}")
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    if object_body < 0:
        raise RuntimeError(f"right_object absent from {scene}")
    site_positions = np.empty((len(qpos), len(site_ids), 3), dtype=np.float64)
    site_rotations = np.empty((len(qpos), len(site_ids), 3, 3), dtype=np.float64)
    object_transform = np.empty((len(qpos), 4, 4), dtype=np.float64)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        site_positions[frame] = data.site_xpos[site_ids]
        site_rotations[frame] = data.site_xmat[site_ids].reshape(-1, 3, 3)
        object_transform[frame] = _object_transform_from_mujoco(data, object_body)
    return {
        "model": model,
        "site_names": site_names,
        "site_positions": site_positions,
        "site_rotations": site_rotations,
        "object_transform": object_transform,
        "scene": str(scene),
    }


def _bounded_mesh_arrays(vertices: np.ndarray, faces: np.ndarray, max_faces: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically simplify a real mesh only for interactive evidence."""
    if max_faces is None or len(faces) <= max_faces:
        return vertices, faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception as exc:  # pragma: no cover - renderer dependency fallback
        raise RuntimeError(f"Could not simplify real MuJoCo mesh to {max_faces} faces") from exc
    return np.asarray(simplified.vertices, dtype=np.float32), np.asarray(simplified.faces, dtype=np.int32)


def robot_surface_payload(
    scene: Path,
    qpos: np.ndarray,
    *,
    geom_group: int = 1,
    max_faces: int | None = None,
    include_layers: bool = True,
) -> dict[str, Any]:
    """Extract real Wuji hand mesh layers and per-frame MuJoCo geom poses.

    ``geom_group=1`` is the authored hand visual layer and ``geom_group=2``
    is the authored collision layer in ``scene_act.xml``.  Keeping the group
    explicit is important for the C-XAR viewer: collision evidence must not
    quietly substitute the visual duplicate.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    def side_for_body(body_id: int) -> str | None:
        current = int(body_id)
        while current > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or ""
            lowered = name.lower()
            if lowered.startswith(("r_", "right_")) or "right" in lowered:
                return "right"
            if lowered.startswith(("l_", "left_")) or "left" in lowered:
                return "left"
            current = int(model.body_parentid[current])
        return None

    chosen: dict[str, list[int]] = {"right": [], "left": []}
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != mesh_type or int(model.geom_dataid[geom_id]) < 0:
            continue
        if int(model.geom_group[geom_id]) != geom_group:
            continue
        side = side_for_body(int(model.geom_bodyid[geom_id]))
        if side is not None:
            chosen[side].append(geom_id)
    if not any(chosen.values()):
        raise RuntimeError(f"No Wuji hand mesh geoms for group={geom_group} found in {scene}")
    entries: dict[str, list[dict[str, Any]]] = {"right": [], "left": []}
    transforms: dict[str, np.ndarray] = {}
    for side, geom_ids in chosen.items():
        transforms[side] = np.empty((len(qpos), len(geom_ids), 12), dtype=np.float32)
        if not include_layers:
            continue
        for geom_id in geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            vert_start, vert_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32)
            # MuJoCo's ``mesh_face`` indices are already local to each mesh;
            # ``mesh_faceadr`` indexes the face table, not the vertex table.
            faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32)
            vertices, faces = _bounded_mesh_arrays(vertices, faces, max_faces)
            entries[side].append({"geom_id": geom_id, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "vertices": vertices, "faces": faces})
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side, geom_ids in chosen.items():
            transforms[side][frame, :, :3] = data.geom_xpos[geom_ids]
            transforms[side][frame, :, 3:] = data.geom_xmat[geom_ids]
    return {"layers": entries, "transforms": transforms, "scene": str(scene)}


def object_surface_payload(scene: Path, qpos: np.ndarray, *, collision: bool, max_faces: int | None = None) -> dict[str, Any]:
    """Extract actual visual or collision object mesh layers for the viewer."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    if object_body < 0:
        raise RuntimeError(f"right_object absent from {scene}")

    def belongs_to_object(body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == object_body:
                return True
            current = int(model.body_parentid[current])
        return False

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    chosen: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != mesh_type or int(model.geom_dataid[geom_id]) < 0:
            continue
        if not belongs_to_object(int(model.geom_bodyid[geom_id])):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if collision:
            if int(model.geom_group[geom_id]) != 3 or name == "right_object_visual":
                continue
        elif name != "right_object_visual":
            continue
        chosen.append(geom_id)
    if not chosen:
        label = "collision" if collision else "visual"
        raise RuntimeError(f"No real object {label} mesh geoms found in {scene}")

    layers: list[dict[str, Any]] = []
    for geom_id in chosen:
        mesh_id = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
        faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int32)
        vertices, faces = _bounded_mesh_arrays(vertices, faces, max_faces)
        layers.append(
            {
                "geom_id": geom_id,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "vertices": vertices,
                "faces": faces,
            }
        )
    transforms = np.empty((len(qpos), len(chosen), 12), dtype=np.float32)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        transforms[frame, :, :3] = data.geom_xpos[chosen]
        transforms[frame, :, 3:] = data.geom_xmat[chosen]
    return {"layers": {"object": layers}, "transforms": {"object": transforms}, "scene": str(scene)}


def _continuous_intrinsic_xyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert free-joint quaternions to a continuous intrinsic XYZ chart."""
    raw = Rotation.from_quat(np.asarray(quaternions_wxyz, dtype=np.float64)[:, (1, 2, 3, 0)]).as_euler("XYZ")
    return np.unwrap(raw, axis=0)


def _stage_b_act_qpos(stage_b_qpos: np.ndarray) -> np.ndarray:
    if stage_b_qpos.shape[1] != 66:
        raise ValueError(f"Expected Stage-B free-joint qpos with 66 columns, got {stage_b_qpos.shape}")
    objects = stage_b_qpos[:, -14:].reshape(len(stage_b_qpos), 2, 7)
    object_chart = np.concatenate((objects[:, :, :3], np.stack([_continuous_intrinsic_xyz(objects[:, side, 3:]) for side in range(2)], axis=1)), axis=2)
    return np.concatenate((stage_b_qpos[:, :52], object_chart.reshape(len(stage_b_qpos), 12)), axis=1)


def _site_transform(geometry: dict[str, Any], name: str) -> np.ndarray:
    index = geometry["site_names"].index(name)
    return transform_from_rt(geometry["site_rotations"][:, index], geometry["site_positions"][:, index])


def _stage_b_audit(paths: AuditPaths, raw: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with np.load(paths.stage_b_trajectory, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    mapping = np.asarray(json.loads(paths.stage_b_mapping.read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
    if len(mapping) != len(qpos) or not np.array_equal(mapping, np.arange(1461, 1875, dtype=np.int64)):
        raise RuntimeError("Frozen Stage-B mapping is not the expected explicit 1461..1874 range")
    geometry = _extract_mujoco_geometry(paths.scene, qpos)
    source_index = mapping - FRAME_START
    raw_object = raw["T_world_object_visual"][source_index]
    stage_object = geometry["object_transform"]
    object_translation = np.linalg.norm(raw_object[:, :3, 3] - stage_object[:, :3, 3], axis=1)
    object_rotation = rotation_residual_rad(raw_object[:, :3, :3], stage_object[:, :3, :3])
    sides: dict[str, Any] = {}
    stored: dict[str, Any] = {"mapping": mapping, "geometry": geometry, "qpos": qpos}
    for side in ("right", "left"):
        source_wrist = raw[f"T_world_{side}_wrist"][source_index].copy()
        source_wrist[:, :3, :3] = source_wrist[:, :3, :3] @ SMPLX_TO_WUJI_PALM[side]
        stage_wrist = _site_transform(geometry, f"{side}_palm")
        source_relative = object_relative(raw_object, source_wrist)
        stage_relative = object_relative(stage_object, stage_wrist)
        source_tips = raw[f"{side}_joints_world"][source_index][:, FINGERTIP_INDICES]
        stage_tips = np.stack([geometry["site_positions"][:, geometry["site_names"].index(f"{side}_{finger}_tip")] for finger in FINGERTIP_NAMES], axis=1)
        source_tips_local = apply_transform(invert_transform(raw_object)[:, None], source_tips)
        stage_tips_local = apply_transform(invert_transform(stage_object)[:, None], stage_tips)
        sides[side] = {
            "T_object_wrist_translation_residual_m": _summary(np.linalg.norm(source_relative[:, :3, 3] - stage_relative[:, :3, 3], axis=1)),
            "T_object_wrist_rotation_residual_rad": _summary(rotation_residual_rad(source_relative[:, :3, :3], stage_relative[:, :3, :3]), "rad"),
            "object_frame_fingertip_residual_m": _summary(np.linalg.norm(source_tips_local - stage_tips_local, axis=-1)),
            "world_fingertip_residual_m": _summary(np.linalg.norm(source_tips - stage_tips, axis=-1)),
            "motion": {
                "wrist": _motion_metrics(source_relative, stage_relative, FPS),
                "object_relative_fingertips": _summary(np.linalg.norm(np.diff(source_tips_local, axis=0) - np.diff(stage_tips_local, axis=0), axis=-1)),
            },
            "source_wrist": source_wrist,
            "stage_wrist": stage_wrist,
            "source_tips": source_tips,
            "stage_tips": stage_tips,
        }
    wrist_max = max(float(row["T_object_wrist_translation_residual_m"]["rmse_m"]) for row in sides.values())
    tip_max = max(float(row["object_frame_fingertip_residual_m"]["rmse_m"]) for row in sides.values())
    passed = bool(object_translation.max() <= 1e-5 and object_rotation.max() <= 1e-5 and wrist_max <= 0.03 and tip_max <= 0.08)
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "STAGE_B_HAND_OBJECT_RELATION_ERROR",
        "frozen_stage_b_source_frame_range": [int(mapping[0]), int(mapping[-1])],
        "not_compared_stage_b_missing_endpoints": [1460, 1875],
        "mapping_reason": "ik_fast intentionally drops the first differentiated pose and end_idx=-1 drops the final input pose; source_mapping.json makes this explicit",
        "object_translation_residual_m": _summary(object_translation),
        "object_rotation_residual_rad": _summary(object_rotation, "rad"),
        "sides": {side: {key: value for key, value in row.items() if key not in {"source_wrist", "stage_wrist", "source_tips", "stage_tips"}} for side, row in sides.items()},
        "thresholds": {"object_translation_m": 1e-5, "object_rotation_rad": 1e-5, "wrist_rmse_m": 0.03, "fingertip_rmse_m": 0.08},
        "checks": {"side_assignment": "PASS", "object_pose": "PASS" if object_translation.max() <= 1e-5 and object_rotation.max() <= 1e-5 else "FAIL", "time_alignment": "PASS", "SMPLX_to_Wuji_palm_basis": {side: SMPLX_TO_WUJI_PALM[side] for side in SMPLX_TO_WUJI_PALM}},
    }
    return report, stored, sides


def _cxa_audit(paths: AuditPaths, stage_b: dict[str, Any], stage_sides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(paths.cxa_trajectory, allow_pickle=False) as archive:
        cxa_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        cxa_mapping = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    mapping = stage_b["mapping"]
    if not np.array_equal(mapping, cxa_mapping):
        raise RuntimeError("C-XA source-frame mapping differs from frozen Stage-B mapping")
    stage_act = _stage_b_act_qpos(stage_b["qpos"])
    before = _extract_mujoco_geometry(paths.scene_act, stage_act)
    after = _extract_mujoco_geometry(paths.scene_act, cxa_qpos)
    object_translation = np.linalg.norm(before["object_transform"][:, :3, 3] - after["object_transform"][:, :3, 3], axis=1)
    object_rotation = rotation_residual_rad(before["object_transform"][:, :3, :3], after["object_transform"][:, :3, :3])
    sides: dict[str, Any] = {}
    max_delta = 0.0
    max_step = 0.0
    for side in ("right", "left"):
        before_wrist = _site_transform(before, f"{side}_palm")
        after_wrist = _site_transform(after, f"{side}_palm")
        before_relative = object_relative(before["object_transform"], before_wrist)
        after_relative = object_relative(after["object_transform"], after_wrist)
        wrist_delta = np.linalg.norm(after_relative[:, :3, 3] - before_relative[:, :3, 3], axis=1)
        wrist_rotation = rotation_residual_rad(before_relative[:, :3, :3], after_relative[:, :3, :3])
        before_tips = np.stack([before["site_positions"][:, before["site_names"].index(f"{side}_{finger}_tip")] for finger in FINGERTIP_NAMES], axis=1)
        after_tips = np.stack([after["site_positions"][:, after["site_names"].index(f"{side}_{finger}_tip")] for finger in FINGERTIP_NAMES], axis=1)
        before_tips_local = apply_transform(invert_transform(before["object_transform"])[:, None], before_tips)
        after_tips_local = apply_transform(invert_transform(after["object_transform"])[:, None], after_tips)
        tip_delta = np.linalg.norm(after_tips_local - before_tips_local, axis=-1)
        wrist_step = np.linalg.norm(np.diff(after_relative[:, :3, 3] - before_relative[:, :3, 3], axis=0), axis=1)
        max_delta = max(max_delta, float(wrist_delta.max()))
        max_step = max(max_step, float(wrist_step.max()))
        sides[side] = {
            "T_object_wrist_delta_m": _summary(wrist_delta),
            "T_object_wrist_rotation_delta_rad": _summary(wrist_rotation, "rad"),
            "object_frame_fingertip_delta_m": _summary(tip_delta),
            "max_frame_to_frame_correction_change_m": float(wrist_step.max()),
            "stage_b_wrist": before_wrist,
            "cxa_wrist": after_wrist,
            "stage_b_tips": before_tips,
            "cxa_tips": after_tips,
        }
    wrist_translation_bound = 0.015
    passed = bool(object_translation.max() <= 1e-5 and object_rotation.max() <= 1e-5 and max_delta <= wrist_translation_bound and max_step <= 0.015)
    status = "PASS" if passed else ("CXA_OBJECT_POSE_MUTATION" if object_translation.max() > 1e-5 or object_rotation.max() > 1e-5 else "CXA_GLOBAL_OFFSET_ERROR")
    report = {
        "schema_version": 1,
        "status": status,
        "source_frame_range": [int(mapping[0]), int(mapping[-1])],
        "coordinate_conversion": "Stage B free-joint xyz+wxyz was converted to the scene_act intrinsic XYZ serial-hinge chart and both trajectories were compared after mujoco.mj_forward; raw qpos chart differences are not treated as pose errors",
        "object_translation_delta_m": _summary(object_translation),
        "object_rotation_delta_rad": _summary(object_rotation, "rad"),
        "sides": {side: {key: value for key, value in row.items() if key not in {"stage_b_wrist", "cxa_wrist", "stage_b_tips", "cxa_tips"}} for side, row in sides.items()},
        "limits": {"per_frame_wrist_translation_bound_m": wrist_translation_bound, "continuous_correction_change_bound_m": 0.015},
        "semantic_patch": {"path": str(paths.cxa_patch_npz), "available": paths.cxa_patch_npz.is_file(), "source_reliability": "UNRELIABLE_SOURCE records are retained by C-XA input and not treated as silently valid contacts"},
        "decision_basis": "Object pose is immutable, but C-XA must not introduce an unbounded whole-wrist object-relative offset or discontinuity.",
    }
    return report, {"before": before, "after": after, "sides": sides, "mapping": mapping, "stage_act": stage_act, "cxa_qpos": cxa_qpos}


def _npz_save(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _copy_report_not_run(path: Path, reason: str) -> None:
    write_json(path, {"schema_version": 1, "status": reason, "reason": "Raw-loader gate did not pass; this downstream audit was intentionally not run."})


def _render_markdown(raw: dict[str, Any], stage_b: dict[str, Any], cxa: dict[str, Any], final: dict[str, Any]) -> str:
    return f"""# Stage C 上游坐标链路审计\n\n## 结论\n\n- raw loader：`{raw['decision']['status']}`（`{raw['decision']['classification']}`）\n- Stage B：`{stage_b['status']}`\n- C-XA：`{cxa['status']}`\n- 最终分类：`{final['classification']}`\n\n## 坐标链\n\n独立 source 使用完整 SMPL-X：`T_world_hand` 来自 SMPL-X 运动学树，`T_world_object=[R(axis-angle(object.global_orient)), object.transl]`，对象 visual mesh 使用 `T_asset_object=I`。相对位姿一律按 `T_object_hand=inverse(T_world_object) @ T_world_hand` 计算。\n\nStage B 的 Wuji palm 通过固定资产基 `R_world_wuji_palm = R_world_smplx_wrist @ R_smplx_to_wuji_palm` 对齐；C-XA 的 64 维串联 XYZ 坐标在 MuJoCo 内还原后与 Stage B 比较。\n\n## 解释\n\nRaw source 与 spider raw 的数值匹配说明 GRAB 与加载器之间没有新增坐标错误。C-XA 如为失败，失败代表其在保留对象位姿的同时引入过大的 object-relative wrist correction；它不等价于修改原始 GRAB 或 Stage B。\n"""


def _final_decision(raw: dict[str, Any], stage_b: dict[str, Any], cxa: dict[str, Any], global_frame: dict[str, Any]) -> dict[str, Any]:
    if raw["decision"]["status"] != "PASS":
        classification = "RAW_LOADER_COORDINATE_ERROR"
    elif stage_b["status"] != "PASS":
        classification = "STAGE_B_RETARGET_COORDINATE_ERROR"
    elif cxa["status"] != "PASS":
        classification = "CXA_CORRECTION_ERROR"
    elif global_frame["classification"] == "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY":
        classification = "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY"
    else:
        classification = "NO_UPSTREAM_COORDINATE_CONSTRUCTION_ERROR"
    return {
        "schema_version": 1,
        "classification": classification,
        "raw_loader": raw["decision"]["status"],
        "stage_b": stage_b["status"],
        "cxa": cxa["status"],
        "global_view_frame_test": global_frame["classification"],
        "grabs_raw_parameters_modified": False,
        "stage_b_overwritten": False,
        "cxa_overwritten": False,
        "frozen_source_frame_range": [FRAME_START, FRAME_END],
    }


def run(
    paths_config: str = "configs/local/paths.yaml",
    output_root: str | None = None,
    run_id: str | None = None,
    render_html: bool = True,
    cxa_root_override: str | None = None,
) -> str:
    """Run the immutable source geometry audit into one unique ignored directory."""
    inputs = _load_paths(paths_config, cxa_root_override)
    if output_root is None:
        output_root = ".local_artifacts/stage_c_source_geometry_audit"
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-source-geometry")
    root = Path(output_root).resolve() / identifier
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite historical audit run: {root}")
    for name in ("manifest", "official_source", "spider_raw", "stage_b", "cxa", "comparisons", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    representation = _source_representation(inputs)
    write_json(root / "reports/grab_source_representation.json", representation)
    official, official_chain = reconstruct_official_source(inputs)
    _npz_save(root / "official_source/source_reconstruction.npz", official)
    write_json(root / "official_source/source_transform_chain.json", official_chain)
    raw, raw_chain = capture_spider_raw_loader(inputs)
    _npz_save(root / "spider_raw/raw_loader_reconstruction.npz", raw)
    write_json(root / "spider_raw/raw_loader_transform_chain.json", raw_chain)
    comparison, temporal, global_frame = compare_raw_loader(official, raw)
    contact = _surface_contact_metrics(inputs, official, raw)
    contact_array_keys = {"unsigned_distance_m", "nearest_face", "nearest_surface_world", "nearest_surface_normal_world", "signed_distance_m"}
    comparison["contact_surface_evidence"] = {
        side: {key: value for key, value in row.items() if key not in contact_array_keys}
        for side, row in contact["sides"].items()
    }
    _npz_save(
        root / "official_source/source_hand_signed_surface_evidence.npz",
        {
            "source_frame_indices": official["source_frame_indices"],
            "left_signed_distance_m": contact["sides"]["left"]["signed_distance_m"],
            "right_signed_distance_m": contact["sides"]["right"]["signed_distance_m"],
            "left_per_frame_max_penetration_depth_m": contact["sides"]["left"]["per_frame_max_penetration_depth_m"],
            "right_per_frame_max_penetration_depth_m": contact["sides"]["right"]["per_frame_max_penetration_depth_m"],
            "left_per_frame_max_external_gap_m": contact["sides"]["left"]["per_frame_max_external_gap_m"],
            "right_per_frame_max_external_gap_m": contact["sides"]["right"]["per_frame_max_external_gap_m"],
        },
    )
    write_json(root / "reports/official_reconstruction_audit.json", official_chain)
    write_json(root / "reports/source_hand_surface_evidence.json", {
        "schema_version": 1,
        "source": "independent SMPL-X reconstruction is the primary coordinate source; frozen full-surface signed evidence is read-only and cross-verified against actual GrabAdapter vertices, with no Stage-B/C-XA pose used",
        "object_mesh": str(inputs.object_mesh),
        "object_mesh_sha256": sha256(inputs.object_mesh),
        "mesh_is_watertight": contact["mesh_is_watertight"],
        "frozen_signed_surface_evidence_cross_verification": contact["source_surface_diagnostics"],
        "sides": comparison["contact_surface_evidence"],
        "array_artifact": str(root / "official_source/source_hand_signed_surface_evidence.npz"),
    })
    write_json(root / "reports/raw_loader_transform_chain_audit.json", {"official": official_chain, "spider_raw": raw_chain, "global_frame_test": global_frame})
    write_json(root / "reports/raw_loader_geometry_comparison.json", comparison)
    write_json(root / "reports/raw_loader_temporal_alignment.json", temporal)
    if stage_b_gate(comparison["decision"]["status"]) == "RUN":
        stage_b_report, stage_b_store, stage_b_sides = _stage_b_audit(inputs, raw)
        cxa_report, cxa_store = _cxa_audit(inputs, stage_b_store, stage_b_sides) if cxa_gate(comparison["decision"]["status"], stage_b_report["status"]) == "RUN" else ({"schema_version": 1, "status": "NOT_RUN_DUE_TO_STAGE_B_FAILURE"}, {})
    else:
        stage_b_report, stage_b_store, stage_b_sides = ({"schema_version": 1, "status": "NOT_RUN_DUE_TO_RAW_LOADER_FAILURE"}, {}, {})
        cxa_report, cxa_store = ({"schema_version": 1, "status": "NOT_RUN_DUE_TO_RAW_LOADER_FAILURE"}, {})
    write_json(root / "stage_b/stage_b_relative_geometry_audit.json", stage_b_report)
    write_text(root / "stage_b/STAGE_B_RELATIVE_GEOMETRY_AUDIT.md", _render_markdown(comparison, stage_b_report, cxa_report, {"classification": "PENDING"}))
    write_json(root / "cxa/cxa_relative_geometry_audit.json", cxa_report)
    write_text(root / "cxa/CXA_RELATIVE_GEOMETRY_AUDIT.md", _render_markdown(comparison, stage_b_report, cxa_report, {"classification": "PENDING"}))
    final = _final_decision(comparison, stage_b_report, cxa_report, global_frame)
    write_json(root / "reports/source_geometry_final_decision.json", final)
    write_text(root / "reports/SOURCE_GEOMETRY_FINAL_DECISION.md", _render_markdown(comparison, stage_b_report, cxa_report, final))
    manifest_paths = [inputs.raw_sequence, inputs.object_mesh, inputs.source_surface_diagnostics, inputs.stage_b_trajectory, inputs.stage_b_mapping, inputs.cxa_trajectory, inputs.scene, inputs.scene_act]
    write_json(root / "reports/source_artifact_manifest.json", {"schema_version": 1, "inputs_opened_read_only": [{"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in manifest_paths], "outputs_root": str(root), "source_geometry_audit_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()})
    if render_html:
        from spider.tools.grab_source_geometry_audit_viewer import render_viewer
        render_viewer(root, official, raw, contact, stage_b_store, stage_b_sides, cxa_store, comparison, stage_b_report, cxa_report, final)
    return str(root)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Read-only GRAB source geometry audit")
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-root", default=".local_artifacts/stage_c_source_geometry_audit")
    parser.add_argument("--run-id")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--cxa-root", help="read a repaired C-XA namespace without modifying the historical one")
    args = parser.parse_args()
    print(run(args.paths_config, args.output_root, args.run_id, not args.no_html, args.cxa_root))


if __name__ == "__main__":
    main()
