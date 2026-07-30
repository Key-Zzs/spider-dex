"""Focused tests for external-data infrastructure, without NAS access."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider.datasets.base import DatasetAdapter, DatasetAudit, SequenceRecord
from spider.datasets.manifest import config_hash, write_manifest
from spider.datasets.paths import ProjectPaths, load_project_paths
from spider.datasets.registry import DatasetRegistry
from spider.datasets.schema import CanonicalHOISequence, HandSequence, ObjectSequence
from spider.datasets.grab import validate_canonical_hand_order
from spider.tools.grab_pipeline import canonical_to_wuji_wrist_orientation


def _coordinates() -> dict[str, object]:
    return {"world_frame": "source world", "axis_convention": "right-handed xyz", "handedness": "right", "rotation_representation": "wxyz", "source_to_canonical_transform": "identity"}


def _hand(side: str, frames: int = 3) -> HandSequence:
    return HandSequence(side=side, valid_mask=np.ones(frames, dtype=bool), global_translation=np.zeros((frames, 3)), global_orientation=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)), mano_pose=np.zeros((frames, 45)), mano_shape=np.zeros(10), joints_world=np.zeros((frames, 21, 3)), joint_names=tuple(f"joint_{i}" for i in range(21)))


def _sequence(*, right: bool = True, left: bool = False, objects: int = 1) -> CanonicalHOISequence:
    frames = 3
    return CanonicalHOISequence(dataset_name="fake", sequence_id="s1_demo", source_sequence_id="source/s1/demo", fps=30.0, timestamps=np.array([0.0, 1 / 30, 2 / 30]), coordinate_system=_coordinates(), length_unit="m", right_hand=_hand("right", frames) if right else None, left_hand=_hand("left", frames) if left else None, objects=[ObjectSequence(object_id=f"object_{idx}", object_name="cube", mesh_path="meshes/cube.ply", valid_mask=np.ones(frames, dtype=bool), translation=np.zeros((frames, 3)), orientation=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)), scale=np.ones(3)) for idx in range(objects)], primary_object_id="object_0")


class FakeAdapter(DatasetAdapter):
    dataset_name = "fake"

    def discover_sequences(self, max_sequences=None):
        return []

    def inspect_source(self, max_sequences=None):
        return DatasetAudit("fake", "PASS", "/tmp")

    def load_sequence(self, sequence_id, **kwargs):
        return _sequence()

    def resolve_object_mesh(self, object_id):
        return Path("/tmp/mesh.ply")

    def describe_sequence(self, sequence_id):
        return SequenceRecord("fake", sequence_id, sequence_id, f"{sequence_id}.npz")


class PathConfigTest(unittest.TestCase):
    def _config(self, root: Path, workspace: str) -> Path:
        source, body = root / "source", root / "models"
        source.mkdir()
        body.mkdir()
        config = root / "paths.yaml"
        config.write_text(f"schema_version: 1\ndatasets:\n  fake:\n    source_root: {source}\nbody_models:\n  root: {body}\nworkspace:\n  root: {workspace}\n", encoding="utf-8")
        return config

    def test_valid_config_workspace_and_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, "${SPIDER_TEST_WORKSPACE}")
            with mock.patch.dict(os.environ, {"SPIDER_TEST_WORKSPACE": str(root / "outside")}, clear=False):
                paths = load_project_paths(config, source_overrides={"fake": root / "source"})
            paths.ensure_workspace(repository_root=root / "unrelated_repo")
            self.assertTrue(all((paths.workspace_root / name).is_dir() for name in ("manifests", "processed", "runs", "reports", "cache")))

    def test_invalid_roots_and_repo_workspace_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, str(root / "source" / "output"))
            paths = load_project_paths(config)
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                paths.validate(repository_root=root / "repository")
            missing = ProjectPaths({"fake": root / "missing"}, root / "models", root / "external")
            with self.assertRaisesRegex(FileNotFoundError, "datasets.fake.source_root"):
                missing.validate()
            repo = root / "repository"
            repo.mkdir(); (repo / ".git").mkdir()
            bad = ProjectPaths({"fake": root / "source"}, root / "models", repo / "workspace")
            with self.assertRaisesRegex(ValueError, "outside the Git repository"):
                bad.validate(repository_root=repo)

    def test_unwritable_workspace_is_clear(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; models = root / "models"; source.mkdir(); models.mkdir()
            paths = ProjectPaths({"fake": source}, models, root / "external")
            with mock.patch("spider.datasets.paths.os.access", return_value=False):
                with self.assertRaisesRegex(PermissionError, "not readable"):
                    paths.ensure_workspace(repository_root=root / "repo")


class RegistryManifestTest(unittest.TestCase):
    def test_registry_and_deterministic_manifest(self):
        registry = DatasetRegistry(); adapter = FakeAdapter(); registry.register(adapter)
        self.assertIs(registry.get("fake"), adapter)
        with self.assertRaisesRegex(KeyError, "Unregistered dataset"):
            registry.get("missing")
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(adapter)
        with tempfile.TemporaryDirectory() as temp:
            target = write_manifest([SequenceRecord("fake", "z", "z", "z.npz"), SequenceRecord("fake", "a", "a", "a.npz")], Path(temp) / "manifest.json", metadata={"config_hash": config_hash({"b": 2, "a": 1})})
            value = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual([item["sequence_id"] for item in value["records"]], ["a", "z"])
            self.assertEqual(config_hash({"a": 1, "b": 2}), value["metadata"]["config_hash"])


class CanonicalSchemaTest(unittest.TestCase):
    def test_right_left_bimanual_multi_object_and_round_trip(self):
        for sequence in (_sequence(right=True), _sequence(right=False, left=True), _sequence(right=True, left=True, objects=2)):
            sequence.validate()
            with tempfile.TemporaryDirectory() as temp:
                npz, metadata = sequence.save(temp)
                self.assertTrue(npz.is_file() and metadata.is_file())
                with np.load(npz, allow_pickle=False) as arrays:
                    self.assertFalse(any(value.dtype == object for value in arrays.values()))
                self.assertEqual(sequence.summary(), CanonicalHOISequence.load(temp).summary())

    def test_invalid_sequence_rejected(self):
        sequence = _sequence()
        sequence.timestamps = np.array([0.0, 0.1, 0.1])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sequence.validate()
        sequence = _sequence(); sequence.right_hand.global_orientation[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "normalized"):
            sequence.validate()
        sequence = _sequence(); sequence.right_hand.joints_world[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            sequence.validate()

    def test_grab_joint_order_and_wuji_wrist_basis_are_explicit(self):
        names = (
            "wrist", "thumb1", "thumb2", "thumb3", "thumb_tip", "index1", "index2", "index3", "index_tip", "middle1", "middle2", "middle3", "middle_tip", "ring1", "ring2", "ring3", "ring_tip", "pinky1", "pinky2", "pinky3", "pinky_tip",
        )
        validate_canonical_hand_order(names)
        with self.assertRaisesRegex(ValueError, "canonical hand order"):
            validate_canonical_hand_order(tuple(reversed(names)))
        source = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        for side in ("right", "left"):
            target = canonical_to_wuji_wrist_orientation(side, source)
            self.assertEqual(target.shape, (2, 4))
            self.assertTrue(np.allclose(np.linalg.norm(target, axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
