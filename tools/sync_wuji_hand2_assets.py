#!/usr/bin/env python3
"""Reproducibly import and adapt Wuji Hand2 Beta1 assets for SPIDER.

The source checkout is read-only.  This script copies only the files used by
the adapter, then derives SPIDER MJCF/URDF wrappers and a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROBOT_TYPE = "wuji_hand2_beta1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SOURCE_RELATIVE = Path("hand2/hand2_beta1/body")
WRIST_JOINTS = (
    ("tx", "slide", "1 0 0", "-2 2"),
    ("ty", "slide", "0 1 0", "-2 2"),
    ("tz", "slide", "0 0 1", "-2 2"),
    ("roll", "hinge", "0 0 1", "-6.2 6.2"),
    ("pitch", "hinge", "1 0 0", "-6.2 6.2"),
    ("yaw", "hinge", "0 1 0", "-6.2 6.2"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copytree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _side_prefix(side: str) -> str:
    return "r" if side == "right" else "l"


def _wrapper_body(side: str, original_wrist: ET.Element) -> ET.Element:
    """Add six scalar wrist controls before the vendor wrist body."""
    # The default bimanual assembly must not spawn two colliding hands at the
    # same origin.  IK controls remain free to move each wrist around this
    # neutral, symmetric separation.
    neutral_y = "-0.16" if side == "right" else "0.16"
    root = ET.Element(
        "body",
        {
            "name": f"{side}_wrist_tx",
            "pos": f"0 {neutral_y} 0",
            # The hand is controlled kinematically at the wrist.  Compensate
            # only its own gravity so standard scene gravity remains active
            # for later object bodies.
            "gravcomp": "1",
        },
    )
    ET.SubElement(
        root,
        "inertial",
        {"pos": "0 0 0", "mass": "0.001", "diaginertia": "1e-6 1e-6 1e-6"},
    )
    current = root
    for suffix, joint_type, axis, joint_range in WRIST_JOINTS:
        joint = ET.SubElement(
            current,
            "joint",
            {
                "name": f"{side}_wrist_{suffix}",
                "type": joint_type,
                "axis": axis,
                "range": joint_range,
                "damping": "1",
                "armature": "0.01",
            },
        )
        del joint
        if suffix == "yaw":
            current.append(original_wrist)
            break
        next_body = ET.SubElement(current, "body", {"name": f"{side}_wrist_{suffix}_link"})
        next_body.set("gravcomp", "1")
        ET.SubElement(
            next_body,
            "inertial",
            {"pos": "0 0 0", "mass": "0.001", "diaginertia": "1e-6 1e-6 1e-6"},
        )
        current = next_body
    return root


def _rename_collision_geoms(body: ET.Element, side: str) -> None:
    """Name vendor collision geoms without duplicating their geometry."""
    prefix = _side_prefix(side)
    for index, child in enumerate(body.iter("body")):
        body_name = child.get("name", "")
        if not body_name.startswith(f"{prefix}_"):
            continue
        if "wrist" in body_name:
            finger, part = "palm", "0"
        else:
            tokens = body_name.split("_")
            finger = next((name for name in FINGERS if name in tokens), "link")
            part = str(index)
        for geom in child.findall("geom"):
            if geom.get("group") == "2" and geom.get("name") is None:
                geom.set("name", f"collision_hand_{side}_{finger}_{part}")


def _add_spider_sites(wrist: ET.Element, side: str) -> None:
    prefix = _side_prefix(side)
    ET.SubElement(
        wrist,
        "site",
        {
            "name": f"{side}_palm",
            "pos": "0 0 0",
            "size": "0.008",
            "type": "sphere",
            "rgba": "0 0.7 1 1",
            "group": "3",
        },
    )
    for finger in FINGERS:
        vendor_name = (
            f"{prefix}_{finger}_tip"
            if finger in {"thumb", "pinky"}
            else f"{prefix}_{finger}_finger_tip"
        )
        parent = next(
            (body for body in wrist.iter("body") if body.find(f"site[@name='{vendor_name}']") is not None),
            None,
        )
        if parent is None:
            raise ValueError(f"Could not find vendor fingertip site {vendor_name}")
        vendor_site = parent.find(f"site[@name='{vendor_name}']")
        assert vendor_site is not None
        attrs = {key: value for key, value in vendor_site.attrib.items() if key != "name"}
        for name in (
            f"{side}_{finger}_tip",
            f"track_hand_{side}_{finger}_tip",
            f"trace_hand_{side}_{finger}_tip",
        ):
            ET.SubElement(parent, "site", {"name": name, **attrs})


def _adapter_xml(vendor_xml: Path, output_xml: Path, side: str) -> None:
    tree = ET.parse(vendor_xml)
    root = tree.getroot()
    root.set("model", f"{ROBOT_TYPE}_{side}")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    # Keep mesh files self-contained so both sides can be included by the
    # bimanual adapter without a global compiler.meshdir collision.
    compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("./asset/mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is not None:
            mesh.set("file", f"vendor/meshes/{side}/{mesh_file}")
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    if worldbody is None or actuator is None:
        raise ValueError(f"Unexpected vendor MJCF layout: {vendor_xml}")
    prefix = _side_prefix(side)
    wrist = worldbody.find(f"body[@name='{prefix}_wrist']")
    if wrist is None:
        raise ValueError(f"Could not find {prefix}_wrist in {vendor_xml}")
    for body in wrist.iter("body"):
        body.set("gravcomp", "1")
    _rename_collision_geoms(wrist, side)
    _add_spider_sites(wrist, side)
    worldbody.remove(wrist)
    worldbody.append(_wrapper_body(side, wrist))

    wrist_actuators = []
    for suffix, _, _, joint_range in WRIST_JOINTS:
        wrist_actuators.append(
            ET.Element(
                "position",
                {
                    "name": f"{side}_wrist_{suffix}_actuator",
                    "joint": f"{side}_wrist_{suffix}",
                    "kp": "100" if suffix in {"tx", "ty", "tz"} else "20",
                    "kv": "20" if suffix in {"tx", "ty", "tz"} else "5",
                    "ctrlrange": joint_range,
                    "ctrllimited": "true",
                },
            )
        )
    for actuator_element in reversed(wrist_actuators):
        actuator.insert(0, actuator_element)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def _adapter_urdf(vendor_urdf: Path, output_urdf: Path, side: str) -> None:
    tree = ET.parse(vendor_urdf)
    root = tree.getroot()
    root.set("name", f"{ROBOT_TYPE}_{side}_6dof")
    prefix = _side_prefix(side)
    root_link = f"{prefix}_wrist"
    base_link = f"{side}_wrist_base"
    links = [base_link]
    links.extend(f"{side}_wrist_{suffix}_link" for suffix, *_ in WRIST_JOINTS[:-1])
    for link_name in reversed(links):
        root.insert(0, ET.Element("link", {"name": link_name}))
    parent = base_link
    for index, (suffix, joint_type, axis, joint_range) in enumerate(WRIST_JOINTS):
        child = root_link if index == len(WRIST_JOINTS) - 1 else links[index + 1]
        lower, upper = joint_range.split()
        joint = ET.Element(
            "joint",
            {"name": f"{side}_wrist_{suffix}", "type": "prismatic" if joint_type == "slide" else "revolute"},
        )
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", {"lower": lower, "upper": upper, "effort": "100", "velocity": "10"})
        root.insert(0, joint)
        parent = child
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)


def _manifest(destination: Path, source_root: Path) -> dict:
    records = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file() and item.name != "ASSET_MANIFEST.json"):
        relative = path.relative_to(destination)
        vendor_original = str(relative).startswith("vendor/") or relative.name == "LICENSE_WUJI"
        records.append(
            {
                "destination_relative_path": str(relative),
                "source_relative_path": str(SOURCE_RELATIVE / relative.relative_to("vendor")) if str(relative).startswith("vendor/") else None,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "vendor_original": vendor_original,
            }
        )
    commit_timestamp = subprocess.check_output(
        ["git", "-C", str(source_root), "show", "-s", "--format=%ct", "HEAD"],
        text=True,
    ).strip()
    return {
        "schema_version": 1,
        "robot_type": ROBOT_TYPE,
        "source_repository": "https://github.com/wuji-technology/wuji-description",
        "source_release": "v2026.7.23",
        "source_root": str(SOURCE_RELATIVE),
        # A deterministic source-revision timestamp keeps repeated syncs
        # byte-identical while still recording the provenance time basis.
        "copy_timestamp_utc": datetime.fromtimestamp(
            int(commit_timestamp), tz=timezone.utc
        ).replace(microsecond=0).isoformat(),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wuji-description-root",
        default=os.environ.get("WUJI_DESCRIPTION_ROOT"),
        required=os.environ.get("WUJI_DESCRIPTION_ROOT") is None,
    )
    parser.add_argument("--destination", default=Path(__file__).parents[1] / "spider/assets/robots" / ROBOT_TYPE, type=Path)
    args = parser.parse_args()

    source_root = Path(args.wuji_description_root).expanduser().resolve()
    source_body = source_root / SOURCE_RELATIVE
    if not (source_body / "mjcf/right.xml").is_file():
        raise FileNotFoundError(f"Expected Wuji Hand2 Beta1 assets below {source_body}")
    destination = args.destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "vendor/mjcf").mkdir(parents=True)
    (destination / "vendor/urdf").mkdir(parents=True)
    (destination / "urdf").mkdir(parents=True)
    _copytree(source_body / "meshes", destination / "vendor/meshes")
    for side in ("right", "left"):
        shutil.copy2(source_body / "mjcf" / f"{side}.xml", destination / "vendor/mjcf" / f"{side}.xml")
        shutil.copy2(source_body / "urdf" / f"{side}.urdf", destination / "vendor/urdf" / f"{side}.urdf")
        _adapter_xml(source_body / "mjcf" / f"{side}.xml", destination / f"{side}.xml", side)
        _adapter_urdf(source_body / "urdf" / f"{side}.urdf", destination / "urdf" / f"{side}_6dof.urdf", side)
    shutil.copy2(source_root / "LICENSE", destination / "LICENSE_WUJI")
    (destination / "bimanual.xml").write_text(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        f"<mujoco model=\"{ROBOT_TYPE}_bimanual\">\n"
        "  <include file=\"right.xml\" />\n"
        "  <include file=\"left.xml\" />\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        "# Wuji Hand2 Beta1 SPIDER adapter\n\n"
        "Generated from wuji-technology/wuji-description v2026.7.23. The `vendor/` "
        "subtree preserves the required upstream MJCF, URDF, and mesh assets. "
        "Top-level MJCF files add SPIDER's six scalar wrist controls, tracking "
        "sites, and collision names without duplicating collision geometry.\n",
        encoding="utf-8",
    )
    (destination / "retarget_config.yaml").write_text(
        "robot_type: wuji_hand2_beta1\n"
        "schema_version: 1\n"
        "current_spider_consumer: null\n"
        "purpose: future URDF-based retarget configuration; current SPIDER IK uses MJCF sites\n"
        "right:\n  urdf: urdf/right_6dof.urdf\n  root_link: right_wrist_base\n  palm_link: r_wrist\n"
        "left:\n  urdf: urdf/left_6dof.urdf\n  root_link: left_wrist_base\n  palm_link: l_wrist\n"
        "fingertips:\n  mano_21_indices: {thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20}\n",
        encoding="utf-8",
    )
    manifest = _manifest(destination, source_root)
    manifest["source_commit"] = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    (destination / "ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
