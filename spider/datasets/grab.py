"""GRAB source adapter using the locally inspected PCA-24 SMPL-X contract."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .base import DatasetAdapter, DatasetAudit, SequenceRecord
from .paths import ProjectPaths
from .schema import CanonicalHOISequence, HandSequence, ObjectSequence, sha256_file


_HAND_JOINTS = {
    "left": (20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70),
    "right": (21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75),
}
_HAND_NAMES = ("wrist", "thumb1", "thumb2", "thumb3", "thumb_tip", "index1", "index2", "index3", "index_tip", "middle1", "middle2", "middle3", "middle_tip", "ring1", "ring2", "ring3", "ring_tip", "pinky1", "pinky2", "pinky3", "pinky_tip")
_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
_FINGERTIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def validate_canonical_hand_order(joint_names: tuple[str, ...]) -> None:
    """Reject a hand skeleton unless its 21 joints have the public contract order.

    The numerical values alone cannot reveal a swapped finger stream.  Keep the
    source-to-IK ordering explicit so canonical, SPIDER and MuJoCo mappings can
    be checked independently.
    """
    if joint_names != _HAND_NAMES:
        raise ValueError(
            "GRAB canonical hand order must be wrist; thumb/index/middle/ring/"
            f"pinky chains, got {joint_names!r}"
        )
    if tuple(joint_names[index] for index in _FINGERTIP_INDICES) != _FINGERTIP_NAMES:
        raise ValueError("GRAB canonical fingertip indices must be [4, 8, 12, 16, 20]")


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _wxyz(rotvec: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return xyzw[:, [3, 0, 1, 2]].astype(np.float32)


class GrabAdapter(DatasetAdapter):
    """Read official GRAB NPZ files without a write or robot dependency."""

    dataset_name = "grab"

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.source_root = paths.source_root(self.dataset_name)
        self.sequence_root = self.source_root / "grab"

    def _path_for(self, sequence_id: str) -> Path:
        source_id = sequence_id.replace("__", "/") if "__" in sequence_id else sequence_id
        candidate = self.sequence_root / f"{source_id}.npz"
        if not candidate.is_file():
            raise FileNotFoundError(f"GRAB sequence not found for {sequence_id!r}: {candidate}")
        return candidate

    def _record(self, path: Path) -> SequenceRecord:
        relative = path.relative_to(self.sequence_root).with_suffix("")
        source_id = relative.as_posix()
        try:
            with np.load(path, allow_pickle=True) as data:
                frames = int(data["n_frames"])
                fps = float(data["framerate"])
                object_name = str(data["obj_name"])
                # GRAB files contain explicit PCA streams for both SMPL-X hands.
                hand_presence = "bimanual_streams" if "lhand" in data and "rhand" in data else "incomplete"
                metadata = {"gender": str(data["gender"]), "subject_id": str(data["sbj_id"]), "motion_intent": str(data["motion_intent"])}
            status = "DISCOVERED"
        except Exception as exc:  # discovery must not stop at an isolated bad input
            frames, fps, object_name, hand_presence, metadata = None, None, "", "unknown", {"error": f"{type(exc).__name__}: {exc}"}
            status = "INVALID_SOURCE_DATA"
        return SequenceRecord(self.dataset_name, _safe_id(source_id), source_id, path.relative_to(self.source_root).as_posix(), frames, fps, hand_presence, (object_name,) if object_name else (), status, metadata)

    def discover_sequences(self, max_sequences: int | None = None) -> list[SequenceRecord]:
        if not self.sequence_root.is_dir():
            raise FileNotFoundError(f"GRAB sequence directory not found: {self.sequence_root}")
        paths = sorted(self.sequence_root.glob("s*/*.npz"))
        if max_sequences is not None:
            paths = paths[:max_sequences]
        return [self._record(path) for path in paths]

    def describe_sequence(self, sequence_id: str) -> SequenceRecord:
        return self._record(self._path_for(sequence_id))

    def resolve_object_mesh(self, object_id: str) -> Path:
        candidates = [self.source_root / "tools" / "object_meshes" / "contact_meshes" / f"{object_id}.ply", self.source_root / f"{object_id}.stl"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"GRAB object mesh not found for {object_id!r}; checked: {candidates}")

    def inspect_source(self, max_sequences: int | None = None) -> DatasetAudit:
        records = self.discover_sequences(max_sequences=max_sequences)
        total_sequence_count = len(list(self.sequence_root.glob("s*/*.npz")))
        failures = [record for record in records if record.status != "DISCOVERED"]
        fps = Counter(record.fps for record in records if record.fps is not None)
        objects = Counter(item for record in records for item in record.object_ids)
        return DatasetAudit(self.dataset_name, "PASS" if not failures else "PARTIAL", str(self.source_root), checks={"sequence_root_exists": self.sequence_root.is_dir(), "object_mesh_root_exists": (self.source_root / "tools" / "object_meshes" / "contact_meshes").is_dir(), "body_model_root_exists": self.paths.body_model_root.is_dir(), "smplx_male_exists": (self.paths.body_model_root / "smplx" / "SMPLX_MALE.npz").is_file(), "smplx_female_exists": (self.paths.body_model_root / "smplx" / "SMPLX_FEMALE.npz").is_file()}, summary={"sequence_count": total_sequence_count, "scanned_sequence_count": len(records), "fps_distribution": {str(key): value for key, value in sorted(fps.items())}, "hand_stream_coverage": dict(Counter(record.hand_presence for record in records)), "objects_seen": len(objects), "sample_pose_dimensions": {"body_pose": 63, "left_hand_pca": 24, "right_hand_pca": 24, "betas": 10}, "invalid_source_records": len(failures)}, errors=[record.metadata["error"] for record in failures])

    def _betas(self, gender: str, subject_id: str) -> tuple[np.ndarray, Path]:
        path = self.source_root / "tools" / "subject_meshes" / gender / f"{subject_id}_betas.npy"
        if not path.is_file():
            raise FileNotFoundError(f"GRAB subject beta file not found: {path}")
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)[:10], path

    def load_sequence(self, sequence_id: str, *, frame_start: int = 0, frame_end: int | None = None, include_vertices: bool = False) -> CanonicalHOISequence:
        """Reconstruct explicit world-space hand joints through source SMPL-X."""
        path = self._path_for(sequence_id)
        with np.load(path, allow_pickle=True) as data:
            frames_total = int(data["n_frames"])
            start, end = max(0, frame_start), frames_total if frame_end is None else min(frame_end, frames_total)
            if start >= end:
                raise ValueError(f"Invalid GRAB frame range [{frame_start}, {frame_end}) for {frames_total} frames")
            index = slice(start, end)
            gender, subject_id, obj_name, fps, ncomps = str(data["gender"]), str(data["sbj_id"]), str(data["obj_name"]), float(data["framerate"]), int(data["n_comps"])
            body = data["body"].item()["params"]
            left = data["lhand"].item()["params"]
            right = data["rhand"].item()["params"]
            object_params = data["object"].item()["params"]
            object_mesh_relative = str(data["object"].item()["object_mesh"])
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported GRAB gender {gender!r} in {path}")
        if ncomps != 24 or left["hand_pose"].shape[1] != 24 or right["hand_pose"].shape[1] != 24:
            raise ValueError(f"Expected GRAB PCA-24 hand poses, got n_comps={ncomps}, left={left['hand_pose'].shape}, right={right['hand_pose'].shape}")
        betas, beta_path = self._betas(gender, subject_id)
        try:
            import smplx
            import torch
        except ImportError as exc:
            raise RuntimeError("GRAB SMPL-X reconstruction requires the 'smplx' package") from exc
        count = end - start
        # GRAB's stored body.fullpose is decoded with the SMPL-X hand mean.
        # ``flat_hand_mean=True`` changes every PCA hand pose; on s1/mug_lift
        # it disagreed with the source fullpose by 0.142 rad on average.
        model = smplx.create(str(self.paths.body_model_root), model_type="smplx", gender=gender, use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=count)
        tensor = lambda value: torch.as_tensor(np.asarray(value[index]), dtype=torch.float32)
        with torch.no_grad():
            output = model(global_orient=tensor(body["global_orient"]), body_pose=tensor(body["body_pose"]), left_hand_pose=tensor(body["left_hand_pose"]), right_hand_pose=tensor(body["right_hand_pose"]), jaw_pose=tensor(body["jaw_pose"]), leye_pose=tensor(body["leye_pose"]), reye_pose=tensor(body["reye_pose"]), expression=tensor(body["expression"]), transl=tensor(body["transl"]), betas=torch.as_tensor(np.repeat(betas[None], count, axis=0), dtype=torch.float32), return_verts=include_vertices, return_full_pose=True)
        joints = output.joints.detach().cpu().numpy().astype(np.float32)
        vertices = output.vertices.detach().cpu().numpy().astype(np.float32) if include_vertices else None
        # Derive world wrist frames from the same SMPL-X forward pass. GRAB's
        # standalone lhand/rhand global orientations belong to auxiliary MANO
        # fits and are not interchangeable with the body SMPL-X wrist frame.
        from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

        beta_tensor = torch.as_tensor(np.repeat(betas[None], count, axis=0), dtype=torch.float32)
        shaped = model.v_template + blend_shapes(beta_tensor, model.shapedirs)
        rest_joints = vertices2joints(model.J_regressor, shaped)
        matrices = batch_rodrigues(output.full_pose.reshape(-1, 3)).reshape(count, -1, 3, 3)
        _, transforms = batch_rigid_transform(matrices, rest_joints, model.parents, dtype=torch.float32)
        wrist_orientations = {side: Rotation.from_matrix(transforms[:, index, :3, :3].detach().cpu().numpy()).as_quat()[:, [3, 0, 1, 2]].astype(np.float32) for side, index in (("left", 20), ("right", 21))}
        hand_vertices: dict[str, np.ndarray | None] = {"left": None, "right": None}
        if vertices is not None:
            correspondence = self.source_root / "tools" / "smplx_correspondence"
            for side in ("left", "right"):
                ids = np.asarray(np.load(correspondence / f"{side[0]}hand_smplx_ids.npy"), dtype=np.int64)
                hand_vertices[side] = vertices[:, ids, :]

        def make_hand(side: str, params: dict[str, np.ndarray]) -> HandSequence:
            indices = _HAND_JOINTS[side]
            # The standalone GRAB hand translation is the MANO model origin,
            # not necessarily the anatomical wrist center. Use the reconstructed
            # SMPL-X wrist to keep the canonical wrist and joints co-located.
            validate_canonical_hand_order(_HAND_NAMES)
            return HandSequence(side=side, valid_mask=np.ones(count, dtype=bool), global_translation=joints[:, indices[0], :], global_orientation=wrist_orientations[side], mano_pose=np.asarray(params["hand_pose"][index], dtype=np.float32), mano_shape=betas, joints_world=joints[:, indices, :], joint_names=_HAND_NAMES, pose_representation="PCA24_axis_angle_root", model_type="SMPL-X", model_gender=gender, vertices_world=hand_vertices[side])

        mesh = self.source_root / object_mesh_relative
        if not mesh.is_file():
            mesh = self.resolve_object_mesh(obj_name)
        source_rel = path.relative_to(self.source_root).as_posix()
        canonical = CanonicalHOISequence(dataset_name=self.dataset_name, sequence_id=_safe_id(sequence_id), source_sequence_id=sequence_id.replace("__", "/"), fps=fps, timestamps=np.arange(start, end, dtype=np.float64) / fps, coordinate_system={"world_frame": "GRAB SMPL-X global mocap world", "axis_convention": "right-handed source XYZ", "handedness": "right", "rotation_representation": "wxyz", "quaternion_order": "wxyz", "source_to_canonical_transform": "identity; source SMPL-X outputs are metres"}, length_unit="m", right_hand=make_hand("right", right), left_hand=make_hand("left", left), objects=[ObjectSequence(object_id=_safe_id(obj_name), object_name=obj_name, mesh_path=object_mesh_relative, valid_mask=np.ones(count, dtype=bool), translation=np.asarray(object_params["transl"][index], dtype=np.float32), orientation=_wxyz(object_params["global_orient"][index]), scale=np.ones(3, dtype=np.float32), source_metadata={"source_relative_mesh_path": object_mesh_relative, "resolved_local_mesh_path": str(mesh.resolve())})], primary_object_id=_safe_id(obj_name), source_metadata={"source_relative_path": source_rel, "resolved_local_path": str(path.resolve()), "subject_id": subject_id, "gender": gender, "source_frame_indices": list(range(start, end)), "motion_intent": "recorded"}, provenance={"source_dataset": "GRAB", "source_file_relative_path": source_rel, "source_file_sha256": sha256_file(path), "source_file_size": path.stat().st_size, "source_file_mtime_ns": path.stat().st_mtime_ns, "body_model_type": "SMPL-X", "body_model_source": "local_config.body_models.root", "body_model_beta_relative_path": beta_path.relative_to(self.source_root).as_posix(), "mano_pca_components": ncomps, "flat_hand_mean": False, "creation_environment": "spider-dex"})
        canonical.validate()
        return canonical
