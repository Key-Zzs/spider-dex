"""Versioned, pickle-free canonical HOI sequence serialization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
_SAFE_SEQUENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _array(value: np.ndarray | Any, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype == object:
        raise ValueError(f"{name} must not have object dtype")
    if not np.issubdtype(result.dtype, np.number) and result.dtype != np.bool_:
        raise ValueError(f"{name} must be numeric or boolean")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    if np.issubdtype(result.dtype, np.number) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _validate_orientation(value: np.ndarray, name: str, frames: int) -> None:
    if value.shape[0] != frames:
        raise ValueError(f"{name} time dimension must equal timestamps ({frames})")
    if value.shape[1:] == (4,):
        norms = np.linalg.norm(value, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"{name} quaternion must be normalized (wxyz)")
    elif value.shape[1:] == (3, 3):
        identity = np.eye(3)
        if not np.allclose(value @ np.swapaxes(value, 1, 2), identity, atol=1e-4):
            raise ValueError(f"{name} rotation matrices must be orthogonal")
        if not np.allclose(np.linalg.det(value), 1.0, atol=1e-4):
            raise ValueError(f"{name} rotation matrices must have determinant +1")
    else:
        raise ValueError(f"{name} must be (T,4) wxyz quaternion or (T,3,3) rotation matrix")


@dataclass
class HandSequence:
    """One real hand; absent hands are represented by ``None`` at sequence level."""

    side: str
    valid_mask: np.ndarray
    global_translation: np.ndarray
    global_orientation: np.ndarray
    mano_pose: np.ndarray
    mano_shape: np.ndarray
    joints_world: np.ndarray
    joint_names: tuple[str, ...]
    pose_representation: str = "axis_angle"
    model_type: str = "smplx"
    model_gender: str = "unknown"
    vertices_world: np.ndarray | None = None

    def validate(self, frames: int) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError(f"hand.side must be left or right, got {self.side!r}")
        mask = _array(self.valid_mask, f"{self.side}.valid_mask", 1)
        trans = _array(self.global_translation, f"{self.side}.global_translation", 2)
        orient = _array(self.global_orientation, f"{self.side}.global_orientation")
        pose = _array(self.mano_pose, f"{self.side}.mano_pose", 2)
        shape = _array(self.mano_shape, f"{self.side}.mano_shape")
        joints = _array(self.joints_world, f"{self.side}.joints_world", 3)
        if mask.shape != (frames,) or trans.shape != (frames, 3) or pose.shape[0] != frames:
            raise ValueError(f"{self.side} hand arrays must have a shared time dimension of {frames}")
        if joints.shape[0] != frames or joints.shape[2] != 3:
            raise ValueError(f"{self.side}.joints_world must be (T,J,3)")
        if len(self.joint_names) != joints.shape[1]:
            raise ValueError(f"{self.side}.joint_names must match joints_world joint count")
        if shape.ndim not in (1, 2) or (shape.ndim == 2 and shape.shape[0] not in (1, frames)):
            raise ValueError(f"{self.side}.mano_shape must be (B,) or (T,B)")
        _validate_orientation(orient, f"{self.side}.global_orientation", frames)
        if self.vertices_world is not None:
            vertices = _array(self.vertices_world, f"{self.side}.vertices_world", 3)
            if vertices.shape[0] != frames or vertices.shape[2] != 3:
                raise ValueError(f"{self.side}.vertices_world must be (T,V,3)")


@dataclass
class ObjectSequence:
    """One source object pose stream in the canonical world frame."""

    object_id: str
    object_name: str
    mesh_path: str
    valid_mask: np.ndarray
    translation: np.ndarray
    orientation: np.ndarray
    scale: np.ndarray
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, frames: int) -> None:
        if not self.object_id or not _SAFE_SEQUENCE_ID.match(self.object_id):
            raise ValueError(f"object_id is not path-safe: {self.object_id!r}")
        mask = _array(self.valid_mask, f"object.{self.object_id}.valid_mask", 1)
        translation = _array(self.translation, f"object.{self.object_id}.translation", 2)
        scale = _array(self.scale, f"object.{self.object_id}.scale")
        orientation = _array(self.orientation, f"object.{self.object_id}.orientation")
        if mask.shape != (frames,) or translation.shape != (frames, 3):
            raise ValueError(f"object.{self.object_id} arrays must have a shared time dimension of {frames}")
        if scale.ndim not in (1, 2) or (scale.ndim == 2 and scale.shape[0] not in (1, frames)):
            raise ValueError(f"object.{self.object_id}.scale must be (S,) or (T,S)")
        _validate_orientation(orientation, f"object.{self.object_id}.orientation", frames)


@dataclass
class CanonicalHOISequence:
    """Canonical, validated HOI sequence with JSON + numerical NPZ storage."""

    dataset_name: str
    sequence_id: str
    source_sequence_id: str
    fps: float
    timestamps: np.ndarray
    coordinate_system: dict[str, Any]
    length_unit: str
    right_hand: HandSequence | None
    left_hand: HandSequence | None
    objects: list[ObjectSequence]
    primary_object_id: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def num_frames(self) -> int:
        return int(np.asarray(self.timestamps).shape[0])

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported canonical schema_version {self.schema_version}; expected {SCHEMA_VERSION}")
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        if not self.sequence_id or not _SAFE_SEQUENCE_ID.match(self.sequence_id):
            raise ValueError(f"sequence_id is not path-safe: {self.sequence_id!r}")
        if not self.source_sequence_id:
            raise ValueError("source_sequence_id must be non-empty")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        timestamps = _array(self.timestamps, "timestamps", 1)
        if len(timestamps) == 0 or not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be non-empty and strictly increasing")
        if self.length_unit != "m":
            raise ValueError(f"canonical length_unit must be 'm', got {self.length_unit!r}")
        required_coordinates = {"world_frame", "axis_convention", "handedness", "rotation_representation", "source_to_canonical_transform"}
        missing = required_coordinates.difference(self.coordinate_system)
        if missing:
            raise ValueError(f"coordinate_system missing fields: {sorted(missing)}")
        if self.right_hand is None and self.left_hand is None:
            raise ValueError("canonical sequence must contain at least one real hand")
        if self.right_hand is not None:
            if self.right_hand.side != "right":
                raise ValueError("right_hand.side must be 'right'")
            self.right_hand.validate(self.num_frames)
        if self.left_hand is not None:
            if self.left_hand.side != "left":
                raise ValueError("left_hand.side must be 'left'")
            self.left_hand.validate(self.num_frames)
        if not self.objects:
            raise ValueError("canonical sequence must contain at least one object")
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)) or self.primary_object_id not in ids:
            raise ValueError("object IDs must be unique and include primary_object_id")
        for obj in self.objects:
            obj.validate(self.num_frames)
        json.dumps(self.source_metadata, sort_keys=True)
        json.dumps(self.provenance, sort_keys=True)

    def summary(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "sequence_id": self.sequence_id,
            "source_sequence_id": self.source_sequence_id,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "hand_presence": {"right": self.right_hand is not None, "left": self.left_hand is not None},
            "object_ids": [item.object_id for item in self.objects],
            "primary_object_id": self.primary_object_id,
        }

    def _metadata(self) -> dict[str, Any]:
        def hand_meta(hand: HandSequence | None) -> dict[str, Any] | None:
            if hand is None:
                return None
            return {"side": hand.side, "joint_names": list(hand.joint_names), "pose_representation": hand.pose_representation, "model_type": hand.model_type, "model_gender": hand.model_gender, "has_vertices": hand.vertices_world is not None}
        return {**self.summary(), "coordinate_system": self.coordinate_system, "length_unit": self.length_unit, "right_hand": hand_meta(self.right_hand), "left_hand": hand_meta(self.left_hand), "objects": [{"object_id": obj.object_id, "object_name": obj.object_name, "mesh_path": obj.mesh_path, "source_metadata": obj.source_metadata} for obj in self.objects], "source_metadata": self.source_metadata, "provenance": self.provenance}

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {"timestamps": np.asarray(self.timestamps)}
        for prefix, hand in (("right", self.right_hand), ("left", self.left_hand)):
            if hand is None:
                continue
            arrays.update({f"{prefix}_valid_mask": np.asarray(hand.valid_mask), f"{prefix}_global_translation": np.asarray(hand.global_translation), f"{prefix}_global_orientation": np.asarray(hand.global_orientation), f"{prefix}_mano_pose": np.asarray(hand.mano_pose), f"{prefix}_mano_shape": np.asarray(hand.mano_shape), f"{prefix}_joints_world": np.asarray(hand.joints_world)})
            if hand.vertices_world is not None:
                arrays[f"{prefix}_vertices_world"] = np.asarray(hand.vertices_world)
        for index, obj in enumerate(self.objects):
            prefix = f"object_{index}"
            arrays.update({f"{prefix}_valid_mask": np.asarray(obj.valid_mask), f"{prefix}_translation": np.asarray(obj.translation), f"{prefix}_orientation": np.asarray(obj.orientation), f"{prefix}_scale": np.asarray(obj.scale)})
        return arrays

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        """Atomically save numerical data and metadata without pickle support."""
        self.validate()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays = self._arrays()
        for name, value in arrays.items():
            if value.dtype == object:
                raise ValueError(f"Refusing object dtype in canonical NPZ field {name}")
        metadata = self._metadata()
        npz_path, metadata_path = target / "canonical_sequence.npz", target / "canonical_metadata.json"
        with tempfile.NamedTemporaryFile(dir=target, suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp, **arrays)
            temp_npz = Path(tmp.name)
        with tempfile.NamedTemporaryFile(mode="w", dir=target, suffix=".json", encoding="utf-8", delete=False) as tmp:
            json.dump(metadata, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            temp_json = Path(tmp.name)
        os.replace(temp_npz, npz_path)
        os.replace(temp_json, metadata_path)
        return npz_path, metadata_path

    @classmethod
    def load(cls, directory: str | Path) -> "CanonicalHOISequence":
        target = Path(directory)
        with (target / "canonical_metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported canonical schema_version {metadata.get('schema_version')}; expected {SCHEMA_VERSION}")
        with np.load(target / "canonical_sequence.npz", allow_pickle=False) as data:
            def hand(prefix: str) -> HandSequence | None:
                info = metadata.get(f"{prefix}_hand")
                if info is None:
                    return None
                return HandSequence(side=info["side"], valid_mask=data[f"{prefix}_valid_mask"].copy(), global_translation=data[f"{prefix}_global_translation"].copy(), global_orientation=data[f"{prefix}_global_orientation"].copy(), mano_pose=data[f"{prefix}_mano_pose"].copy(), mano_shape=data[f"{prefix}_mano_shape"].copy(), joints_world=data[f"{prefix}_joints_world"].copy(), joint_names=tuple(info["joint_names"]), pose_representation=info["pose_representation"], model_type=info["model_type"], model_gender=info["model_gender"], vertices_world=data[f"{prefix}_vertices_world"].copy() if info.get("has_vertices") else None)
            objects = [ObjectSequence(object_id=info["object_id"], object_name=info["object_name"], mesh_path=info["mesh_path"], valid_mask=data[f"object_{index}_valid_mask"].copy(), translation=data[f"object_{index}_translation"].copy(), orientation=data[f"object_{index}_orientation"].copy(), scale=data[f"object_{index}_scale"].copy(), source_metadata=info.get("source_metadata", {})) for index, info in enumerate(metadata["objects"])]
            result = cls(dataset_name=metadata["dataset_name"], sequence_id=metadata["sequence_id"], source_sequence_id=metadata["source_sequence_id"], fps=float(metadata["fps"]), timestamps=data["timestamps"].copy(), coordinate_system=metadata["coordinate_system"], length_unit=metadata["length_unit"], right_hand=hand("right"), left_hand=hand("left"), objects=objects, primary_object_id=metadata["primary_object_id"], source_metadata=metadata.get("source_metadata", {}), provenance=metadata.get("provenance", {}), schema_version=metadata["schema_version"])
        result.validate()
        return result


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for a concrete processed source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
