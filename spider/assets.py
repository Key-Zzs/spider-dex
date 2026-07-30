"""Packaged robot-asset discovery and safe runtime staging.

Robot models are versioned with the Python package, while demonstrations and
generated scenes live outside the repository.  This module is the single
place that bridges those two locations without copying any dataset content.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def get_packaged_robot_asset_dir(robot_type: str) -> Path:
    """Return the packaged asset directory for ``robot_type``.

    Raises:
        FileNotFoundError: If the named robot is not shipped by this checkout.
    """
    if not robot_type or Path(robot_type).name != robot_type:
        raise ValueError(f"robot_type must be a simple directory name: {robot_type!r}")
    asset_dir = Path(__file__).resolve().parent / "assets" / "robots" / robot_type
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"Packaged assets for robot_type={robot_type!r} were not found at "
            f"{asset_dir}."
        )
    return asset_dir


def _manifest_identity(asset_dir: Path) -> dict | None:
    manifest_path = asset_dir / "ASSET_MANIFEST.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ensure_robot_assets(
    dataset_dir: str | Path,
    dataset_name: str,
    robot_type: str,
) -> Path:
    """Stage one packaged robot model below a processed dataset root.

    Existing manifest-backed directories are reused only when their manifest
    is byte-for-byte equivalent to the packaged one.  A conflicting directory
    is never overwritten silently, which prevents a generated scene from
    accidentally loading a model from a different source revision.
    """
    source_dir = get_packaged_robot_asset_dir(robot_type)
    destination = (
        Path(dataset_dir).expanduser().resolve()
        / "processed"
        / dataset_name
        / "assets"
        / "robots"
        / robot_type
    )
    source_manifest = _manifest_identity(source_dir)

    if destination.exists():
        destination_manifest = _manifest_identity(destination)
        if source_manifest is not None and destination_manifest == source_manifest:
            return destination
        # Legacy upstream robot assets have no manifest. Preserve an existing
        # staged legacy directory rather than changing their historical
        # behavior; only manifest-backed assets opt into strict provenance.
        if source_manifest is None:
            return destination
        raise RuntimeError(
            "Refusing to overwrite an existing robot asset directory with a "
            f"different or missing manifest: {destination}. Remove or rename "
            "that directory only after verifying its provenance."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{robot_type}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source_dir, staging)
    staging.replace(destination)
    return destination
