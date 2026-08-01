"""Render bounded, real-MuJoCo Stage C-XAR screenshots without WebGL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from spider.datasets.paths import load_project_paths
from spider.tools.grab_stage_c import _stage_b_act_baseline
from spider.tools.grab_stage_c_xar_viewer import _baked_robot_surface, _hand_points
from spider.tools.grab_source_geometry_audit import object_surface_payload


SNAPSHOT_FRAMES = np.asarray((1461, 1462, 1463, 1464, 1465, 1466, 1480, 1619, 1798, 1874), dtype=np.int64)
STAGES = (("base", "Stage B", "#ffd166"), ("old", "旧 C-XA", "#ef476f"), ("repaired", "修复 C-XA", "#06d6a0"))
VIEWS = (("world_visual", "世界坐标 / visual", -55.0, False, None), ("object_collision", "物体坐标 / collision", 26.0, True, "object"), ("left_visual_collision", "左腕坐标 / visual+collision", 112.0, False, "left"))


def _inverse(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def _transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def _draw_mesh(ax: Any, vertices: np.ndarray, faces: np.ndarray, color: str, alpha: float) -> None:
    if not len(vertices) or not len(faces):
        return
    polygons = np.asarray(vertices)[np.asarray(faces, dtype=np.int64)]
    ax.add_collection3d(Poly3DCollection(polygons, facecolor=color, edgecolor="none", alpha=alpha, linewidth=0.0))


def _object_mesh(payload: dict[str, Any], frame: int, coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for index, layer in enumerate(payload["layers"]["object"]):
        packed = np.asarray(payload["transforms"]["object"][frame, index], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = packed[3:].reshape(3, 3)
        transform[:3, 3] = packed[:3]
        local = _transform(np.asarray(layer["vertices"]), transform)
        vertices.append(_transform(local, coordinate))
        faces.append(np.asarray(layer["faces"], dtype=np.int64) + offset)
        offset += len(local)
    return np.concatenate(vertices), np.concatenate(faces)


def render(run_root: str, paths_config: str, repaired_root: str, old_root: str | None = None) -> dict[str, Any]:
    root = Path(run_root).resolve()
    output_dir = root / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = load_project_paths(paths_config)
    robot = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0"
    old = Path(old_root).resolve() if old_root else robot / "stage_c_contract_v2_cxa"
    repaired = Path(repaired_root).resolve()
    physics = json.loads((robot / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    scene = Path(physics["scene_act"])
    base, _ = _stage_b_act_baseline(paths, "s5__cylindermedium_lift")
    with np.load(old / "trajectory_depenetrated_init_cxa_level_1_flexible.npz", allow_pickle=False) as data:
        old_qpos, frames = np.asarray(data["qpos"]), np.asarray(data["source_frame_indices"])
    with np.load(repaired / "trajectory_depenetrated_init.npz", allow_pickle=False) as data:
        repaired_qpos = np.asarray(data["qpos"])
    frame_lookup = {int(frame): index for index, frame in enumerate(frames)}
    indices = np.asarray([frame_lookup[int(frame)] for frame in SNAPSHOT_FRAMES], dtype=np.int64)
    stage_qpos = {"base": base[indices], "old": old_qpos[indices], "repaired": repaired_qpos[indices]}
    visual = {stage: _baked_robot_surface(scene, value, geom_group=1, max_faces=400) for stage, value in stage_qpos.items()}
    collision = {stage: _baked_robot_surface(scene, value, geom_group=2, max_faces=280) for stage, value in stage_qpos.items()}
    points = {stage: _hand_points(scene, value) for stage, value in stage_qpos.items()}
    object_visual = object_surface_payload(scene, stage_qpos["base"], collision=False, max_faces=350)
    object_collision = object_surface_payload(scene, stage_qpos["base"], collision=True, max_faces=280)
    with np.load(repaired / "source_contact_patches.npz", allow_pickle=False) as data:
        patch_local = np.asarray(data["center_object_local"], dtype=np.float64)

    outputs: list[dict[str, Any]] = []
    for frame_index, source_frame in enumerate(SNAPSHOT_FRAMES):
        for view_name, view_label, azimuth, collision_only, anchor in VIEWS:
            reference = points["base"]
            if anchor == "object":
                coordinate = _inverse(reference["object"][frame_index])
            elif anchor == "left":
                coordinate = _inverse(reference["left"]["palm"][frame_index])
            else:
                coordinate = np.eye(4, dtype=np.float64)
            figure = plt.figure(figsize=(12.8, 9.0), dpi=150)
            axis = figure.add_subplot(111, projection="3d")
            cloud: list[np.ndarray] = []
            mesh_source = collision if collision_only else visual
            for stage, label, color in STAGES:
                for side in ("right", "left"):
                    layer = mesh_source[stage][side][frame_index]
                    vertices = _transform(layer["vertices"], coordinate)
                    _draw_mesh(axis, vertices, layer["faces"], color, 0.30 if collision_only else 0.46)
                    cloud.append(vertices)
            object_payload = object_collision if collision_only else object_visual
            object_vertices, object_faces = _object_mesh(object_payload, frame_index, coordinate)
            _draw_mesh(axis, object_vertices, object_faces, "#457b9d", 0.38)
            cloud.append(object_vertices)
            object_world = _transform(patch_local, reference["object"][frame_index])
            patch = _transform(object_world, coordinate)
            axis.scatter(patch[:, 0], patch[:, 1], patch[:, 2], c="#c77dff", s=14, marker="D", depthshade=False, label="semantic patch")
            all_vertices = np.concatenate(cloud)
            center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) * 0.5
            radius = max(float(np.ptp(all_vertices, axis=0).max()) * 0.56, 0.08)
            axis.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius))
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=22.0, azim=azimuth)
            axis.set_title(f"Stage C-XAR 真实 scene_act 网格 | source frame {int(source_frame)} | {view_label}\n黄=Stage B，红=旧 C-XA，绿=修复 C-XA，蓝=物体，紫=semantic patch", fontsize=11)
            axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)"); axis.set_zlabel("z (m)")
            axis.grid(False)
            filename = f"xar_{int(source_frame)}_{view_name}.png"
            destination = output_dir / filename
            figure.tight_layout()
            figure.savefig(destination, facecolor="white")
            plt.close(figure)
            outputs.append({"source_frame": int(source_frame), "view": view_name, "path": str(destination), "mesh_role": "collision" if collision_only else "visual"})
    manifest = {"schema_version": 1, "status": "COMPLETE", "renderer": "matplotlib_3d_real_mujoco_mesh", "scene": str(scene), "note": "Every displayed triangle is sampled from a real scene_act MuJoCo mesh after the recorded qpos transform; no alignment or qpos rewrite is applied.", "screenshots": outputs}
    manifest_path = output_dir / "XAR_SCREENSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "count": len(outputs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--old-root")
    print(json.dumps(render(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
