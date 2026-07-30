#!/usr/bin/env python3
"""Static, MuJoCo, hold, and staging validation for Wuji Hand2 Beta1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from spider.assets import ensure_robot_assets, get_packaged_robot_asset_dir


ROBOT_TYPE = "wuji_hand2_beta1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MODELS = ("right", "left", "bimanual")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        mujoco.mj_id2name(model, object_type, index) or f"<unnamed:{index}>"
        for index in range(count)
    ]


def _validate_manifest(asset_dir: Path) -> None:
    manifest = json.loads((asset_dir / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["robot_type"] == ROBOT_TYPE
    assert manifest["source_release"] == "v2026.7.23"
    assert (asset_dir / "LICENSE_WUJI").is_file()
    for entry in manifest["files"]:
        path = asset_dir / entry["destination_relative_path"]
        assert path.is_file(), f"Manifest file missing: {path}"
        assert path.stat().st_size == entry["size_bytes"], f"Size mismatch: {path}"
        assert _sha256(path) == entry["sha256"], f"SHA-256 mismatch: {path}"


def _validate_relative_paths(asset_dir: Path) -> None:
    for path in [*asset_dir.glob("*.xml"), *asset_dir.glob("urdf/*.urdf")]:
        root = ET.parse(path).getroot()
        for element in root.iter():
            for value in element.attrib.values():
                assert not Path(value).is_absolute(), f"Absolute path in {path}: {value}"
                assert "wuji-description" not in value, f"External source path in {path}: {value}"


def _required_sites(side: str) -> set[str]:
    result = {f"{side}_palm"}
    for finger in FINGERS:
        result.update(
            {
                f"{side}_{finger}_tip",
                f"track_hand_{side}_{finger}_tip",
                f"trace_hand_{side}_{finger}_tip",
            }
        )
    return result


def _validate_model(asset_dir: Path, variant: str) -> dict:
    model = mujoco.MjModel.from_xml_path(str(asset_dir / f"{variant}.xml"))
    expected_dof = 52 if variant == "bimanual" else 26
    assert (model.nq, model.nv, model.nu) == (expected_dof,) * 3
    expected_sides = ("right", "left") if variant == "bimanual" else (variant,)
    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_names = set(_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
    assert len(actuator_names) == len(set(actuator_names))
    assert len(joint_names) == len(set(joint_names))
    assert all(_required_sites(side) <= site_names for side in expected_sides)

    for side_index, side in enumerate(expected_sides):
        start = side_index * 26
        wrist_names = [f"{side}_wrist_{suffix}_actuator" for suffix in ("tx", "ty", "tz", "roll", "pitch", "yaw")]
        assert actuator_names[start : start + 6] == wrist_names
        qpos_addresses = []
        for actuator_id in range(start, start + 26):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            assert joint_id >= 0
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            assert model.actuator_ctrlrange[actuator_id, 0] <= model.actuator_ctrlrange[actuator_id, 1]
        assert qpos_addresses == list(range(start, start + 26))

    data = mujoco.MjData(model)
    data.ctrl[:] = np.clip(0.0, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[int(model.jnt_qposadr[joint_id])] = data.ctrl[actuator_id]
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    steps = max(1, math.ceil(2.5 / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert np.isfinite(data.actuator_force).all()
    max_qpos_delta = float(np.max(np.abs(data.qpos - initial_qpos)))
    assert max_qpos_delta < 1e-4, f"Neutral hold drift too large: {max_qpos_delta}"
    collision_geoms = sum(int(model.geom_group[index]) == 2 for index in range(model.ngeom))
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "bodies": int(model.nbody),
        "sites": int(model.nsite),
        "geoms": int(model.ngeom),
        "collision_geoms": collision_geoms,
        "neutral_hold_seconds": steps * model.opt.timestep,
        "max_neutral_qpos_delta": max_qpos_delta,
    }


def validate(asset_dir: Path | None = None) -> dict:
    """Run all validation gates and return model statistics."""
    asset_dir = asset_dir or get_packaged_robot_asset_dir(ROBOT_TYPE)
    required = [
        "right.xml",
        "left.xml",
        "bimanual.xml",
        "retarget_config.yaml",
        "LICENSE_WUJI",
        "ASSET_MANIFEST.json",
        "urdf/right_6dof.urdf",
        "urdf/left_6dof.urdf",
    ]
    for relative in required:
        assert (asset_dir / relative).is_file(), f"Required asset missing: {relative}"
    _validate_manifest(asset_dir)
    _validate_relative_paths(asset_dir)
    for path in asset_dir.glob("urdf/*.urdf"):
        ET.parse(path)
    stats = {variant: _validate_model(asset_dir, variant) for variant in MODELS}
    with tempfile.TemporaryDirectory(prefix="spider-wuji-stage-") as tmp:
        staged = ensure_robot_assets(tmp, "unit_test", ROBOT_TYPE)
        assert staged.is_dir()
        staged_model = mujoco.MjModel.from_xml_path(str(staged / "bimanual.xml"))
        assert staged_model.nu == 52
        # A manifest-backed directory refuses a provenance conflict.
        (staged / "ASSET_MANIFEST.json").write_text("{}\n", encoding="utf-8")
        try:
            ensure_robot_assets(tmp, "unit_test", ROBOT_TYPE)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Manifest conflict did not fail closed")
        # Legacy packaged robots have no manifest and retain their existing
        # stage/reuse behavior.
        legacy = ensure_robot_assets(tmp, "legacy_test", "xhand")
        assert ensure_robot_assets(tmp, "legacy_test", "xhand") == legacy
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    stats = validate(args.asset_dir)
    output = json.dumps(stats, indent=2, sort_keys=True)
    print(output)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
