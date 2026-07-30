"""Portable project-path configuration and safe external workspace setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_DIRS = ("manifests", "processed", "runs", "reports", "cache")


def _expanded_path(value: str | Path, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _repo_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved read-only source/model roots and writable external workspace."""

    dataset_roots: dict[str, Path]
    body_model_root: Path
    workspace_root: Path

    def source_root(self, dataset_name: str) -> Path:
        try:
            return self.dataset_roots[dataset_name]
        except KeyError as exc:
            raise KeyError(f"No source_root configured for dataset {dataset_name!r}") from exc

    def validate(self, repository_root: Path | None = None) -> None:
        for name, root in self.dataset_roots.items():
            if not root.exists():
                raise FileNotFoundError(f"datasets.{name}.source_root does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"datasets.{name}.source_root is not a directory: {root}")
            if not os.access(root, os.R_OK | os.X_OK):
                raise PermissionError(f"datasets.{name}.source_root is not readable: {root}")
        if not self.body_model_root.exists():
            raise FileNotFoundError(f"body_models.root does not exist: {self.body_model_root}")
        if not self.body_model_root.is_dir() or not os.access(self.body_model_root, os.R_OK | os.X_OK):
            raise PermissionError(f"body_models.root is not a readable directory: {self.body_model_root}")
        repo = repository_root.resolve() if repository_root else _repo_ancestor(Path.cwd())
        if repo and _is_inside(self.workspace_root, repo):
            raise ValueError(f"workspace.root must be outside the Git repository {repo}: {self.workspace_root}")
        for name, source in self.dataset_roots.items():
            if self.workspace_root == source:
                raise ValueError(f"workspace.root must differ from datasets.{name}.source_root: {source}")
            if _is_inside(self.workspace_root, source):
                raise ValueError(f"workspace.root must not be inside datasets.{name}.source_root: {self.workspace_root}")
            if _is_inside(source, self.workspace_root):
                raise ValueError(f"workspace.root must not contain datasets.{name}.source_root: {self.workspace_root}")

    def ensure_workspace(self, repository_root: Path | None = None) -> Path:
        self.validate(repository_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.workspace_root, os.W_OK | os.X_OK):
            raise PermissionError(f"workspace.root is not writable: {self.workspace_root}")
        for name in WORKSPACE_DIRS:
            (self.workspace_root / name).mkdir(exist_ok=True)
        return self.workspace_root

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "datasets": {key: {"source_root": str(value)} for key, value in self.dataset_roots.items()},
            "body_models": {"root": str(self.body_model_root)},
            "workspace": {"root": str(self.workspace_root)},
        }


def load_project_paths(
    config_path: str | Path,
    *,
    source_overrides: dict[str, str | Path] | None = None,
    body_model_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> ProjectPaths:
    """Load schema-v1 YAML paths; explicit arguments override file values."""
    path = _expanded_path(config_path, "paths_config")
    if not path.is_file():
        raise FileNotFoundError(f"paths_config does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if data.get("schema_version") != 1:
        raise ValueError(f"paths_config schema_version must be 1: {path}")
    datasets = data.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"paths_config datasets must be a non-empty mapping: {path}")
    roots: dict[str, Path] = {}
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            raise ValueError(f"datasets.{name} must be a mapping in {path}")
        roots[str(name)] = _expanded_path(entry.get("source_root"), f"datasets.{name}.source_root")
    for name, value in (source_overrides or {}).items():
        roots[name] = _expanded_path(value, f"datasets.{name}.source_root override")
    configured_models = (data.get("body_models") or {}).get("root")
    configured_workspace = (data.get("workspace") or {}).get("root")
    result = ProjectPaths(
        dataset_roots=roots,
        body_model_root=_expanded_path(body_model_root or configured_models, "body_models.root"),
        workspace_root=_expanded_path(workspace_root or configured_workspace, "workspace.root"),
    )
    return result
