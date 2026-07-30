"""Bounded GRAB canonical-to-SPIDER preparation and source replay tools."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
import trimesh
import tyro
import yaml
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from spider.datasets.grab import GrabAdapter
from spider.datasets.manifest import config_hash
from spider.datasets.paths import load_project_paths
from spider.datasets.schema import CanonicalHOISequence
from spider.io import get_processed_data_dir

matplotlib.use("Agg")


def _pose7(translation: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    return np.concatenate([translation, orientation], axis=1).astype(np.float32)


def prepare(
    paths_config: str,
    sequence_id: str,
    frame_start: int = 0,
    frame_end: int = 120,
    data_id: int = 0,
    include_vertices: bool = False,
) -> str:
    """Create canonical data and legacy SPIDER keypoints in external workspace."""
    paths = load_project_paths(paths_config)
    paths.ensure_workspace()
    adapter = GrabAdapter(paths)
    sequence = adapter.load_sequence(sequence_id, frame_start=frame_start, frame_end=frame_end, include_vertices=include_vertices)
    canonical_dir = paths.workspace_root / "processed" / "grab" / "canonical" / sequence.sequence_id
    sequence.save(canonical_dir)
    task = sequence.sequence_id
    embodiment = "bimanual" if sequence.right_hand and sequence.left_hand else ("right" if sequence.right_hand else "left")
    mano_dir = Path(get_processed_data_dir(str(paths.workspace_root), "grab", "mano", embodiment, task, data_id))
    mano_dir.mkdir(parents=True, exist_ok=True)
    object_item = sequence.objects[0]
    mesh_source = Path(object_item.source_metadata["resolved_local_mesh_path"])
    mesh_dir = paths.workspace_root / "processed" / "grab" / "assets" / "objects" / sequence.sequence_id
    mesh_dir.mkdir(parents=True, exist_ok=True)
    visual_mesh = mesh_dir / "visual.obj"
    if not visual_mesh.exists():
        mesh = trimesh.load(mesh_source, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.export(visual_mesh)
    frames = sequence.num_frames
    identity = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1))
    zeros = np.zeros((frames, 3), dtype=np.float32)
    def hand_values(hand):
        if hand is None:
            return _pose7(zeros, identity), np.concatenate([np.zeros((frames, 5, 3), dtype=np.float32), np.tile(identity[:, None], (1, 5, 1))], axis=2)
        tips = hand.joints_world[:, [4, 8, 12, 16, 20], :]
        return _pose7(hand.global_translation, hand.global_orientation), np.concatenate([tips, np.tile(hand.global_orientation[:, None], (1, 5, 1))], axis=2)
    wrist_right, finger_right = hand_values(sequence.right_hand)
    wrist_left, finger_left = hand_values(sequence.left_hand)
    object_right = _pose7(object_item.translation, object_item.orientation)
    object_left = _pose7(zeros, identity)
    np.savez_compressed(mano_dir / "trajectory_keypoints.npz", qpos_wrist_right=wrist_right, qpos_finger_right=finger_right, qpos_wrist_left=wrist_left, qpos_finger_left=finger_left, qpos_obj_right=object_right, qpos_obj_left=object_left, source_frame_indices=np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64))
    mesh_rel = str(mesh_dir.relative_to(paths.workspace_root))
    run_config = {"sequence_id": sequence.sequence_id, "source_sequence_id": sequence.source_sequence_id, "frame_range": [frame_start, frame_end], "data_id": data_id, "embodiment": embodiment, "robot_type": "wuji_hand2_beta1"}
    task_info = {"task": task, "dataset_name": "grab", "robot_type": "mano", "embodiment_type": embodiment, "data_id": data_id, "right_object_mesh_dir": mesh_rel, "left_object_mesh_dir": None, "right_object_convex_dir": None, "left_object_convex_dir": None, "ref_dt": 1.0 / sequence.fps, "n_frames": frames, "source_sequence_id": sequence.source_sequence_id, "source_frame_indices": sequence.source_metadata["source_frame_indices"], "canonical_dir": str(canonical_dir), "config_hash": config_hash(run_config), "compatibility_missing_hand_zero_fill": embodiment != "bimanual"}
    task_info_path = mano_dir.parent / "task_info.json"
    task_info_path.write_text(json.dumps(task_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (canonical_dir / "source_manifest.json").write_text(json.dumps({"record": adapter.describe_sequence(sequence_id).to_dict(), "canonical_summary": sequence.summary(), "spider_task": task, "spider_data_id": data_id}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"canonical_dir": str(canonical_dir), "mano_dir": str(mano_dir), "task": task, "embodiment": embodiment}, indent=2))
    return str(canonical_dir)


def render_source(canonical_dir: str, output: str | None = None, max_frames: int = 120) -> str:
    """Headlessly render canonical source mesh plus both hand skeletons."""
    sequence = CanonicalHOISequence.load(canonical_dir)
    target = Path(output) if output else Path(canonical_dir) / "visualization" / "source_replay.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    obj = sequence.objects[0]
    mesh = trimesh.load(Path(obj.source_metadata["resolved_local_mesh_path"]), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    faces = mesh.faces[: min(len(mesh.faces), 800)]
    vertices = mesh.vertices.astype(np.float32)
    steps = np.linspace(0, sequence.num_frames - 1, min(max_frames, sequence.num_frames), dtype=int)
    frames: list[np.ndarray] = []
    for frame in steps:
        fig = plt.figure(figsize=(7, 7), dpi=100); axis = fig.add_subplot(111, projection="3d")
        quat = obj.orientation[frame]; rotated = Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(vertices) + obj.translation[frame]
        axis.plot_trisurf(rotated[:, 0], rotated[:, 1], rotated[:, 2], triangles=faces, color="#8ab6d6", alpha=0.55, linewidth=0.05)
        for hand, color in ((sequence.right_hand, "#e45756"), (sequence.left_hand, "#4c78a8")):
            if hand is None: continue
            points = hand.joints_world[frame]
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=9)
            for start in (0, 1, 5, 9, 13, 17):
                if start == 0: continue
                chain = [0, start, start + 1, start + 2, start + 3]
                axis.plot(points[chain, 0], points[chain, 1], points[chain, 2], color=color, linewidth=1.5)
        all_points = np.vstack([rotated, *[hand.joints_world[frame] for hand in (sequence.right_hand, sequence.left_hand) if hand is not None]])
        center = all_points.mean(axis=0); extent = max(np.ptp(all_points, axis=0).max() * 0.65, 0.12)
        axis.set(xlim=(center[0]-extent, center[0]+extent), ylim=(center[1]-extent, center[1]+extent), zlim=(center[2]-extent, center[2]+extent), title=f"{sequence.source_sequence_id} frame {frame} / {sequence.num_frames-1}")
        axis.set_box_aspect((1, 1, 1)); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(target, frames, fps=max(1, round(sequence.fps * len(steps) / sequence.num_frames)))
    print(target)
    return str(target)


def run_wuji_ik(paths_config: str, sequence_id: str, data_id: int = 0, save_video: bool = True) -> None:
    """Generate Wuji scene and run the existing kinematic IK entry point."""
    paths = load_project_paths(paths_config); task = sequence_id.replace("/", "__")
    canonical = CanonicalHOISequence.load(paths.workspace_root / "processed" / "grab" / "canonical" / task)
    embodiment = "bimanual" if canonical.right_hand and canonical.left_hand else ("right" if canonical.right_hand else "left")
    from spider.preprocess.generate_xml import main as generate_scene
    from spider.preprocess.ik_fast import main as run_ik
    generate_scene(dataset_dir=str(paths.workspace_root), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type=embodiment, task=task, data_id=data_id, use_visual_mesh_as_collision=True, show_viewer=False)
    run_ik(dataset_dir=str(paths.workspace_root), dataset_name="grab", robot_type="wuji_hand2_beta1", embodiment_type=embodiment, task=task, data_id=data_id, show_viewer=False, show_viser_viewer=False, save_video=save_video, start_idx=0, end_idx=-1, wrist_pos_cost=10.0, wrist_ori_cost=3.0, finger_pos_cost=1.0, wrist_init_steps=200, finger_init_steps=300, average_frame_size=1)
    robot_dir = Path(get_processed_data_dir(str(paths.workspace_root), "grab", "wuji_hand2_beta1", embodiment, task, data_id))
    model = mujoco.MjModel.from_xml_path(str(robot_dir.parent / "scene.xml"))
    with np.load(robot_dir / "trajectory_kinematic.npz", allow_pickle=False) as trajectory:
        qpos, qvel, frequency = trajectory["qpos"], trajectory["qvel"], float(trajectory["frequency"])
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("OUTPUT_VALIDATION_FAILED: IK trajectory contains NaN/Inf")
    robot_qpos = qpos[:, :52 if embodiment == "bimanual" else 26]
    margins = []
    violations = 0
    for joint_id in range(model.njnt):
        qadr = model.jnt_qposadr[joint_id]
        if qadr >= robot_qpos.shape[1] or model.jnt_limited[joint_id] == 0:
            continue
        lower, upper = model.jnt_range[joint_id]
        values = robot_qpos[:, qadr]
        violations += int(np.count_nonzero((values < lower - 1e-6) | (values > upper + 1e-6)))
        margins.append(float(np.minimum(values - lower, upper - values).min()))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    # ik_fast applies Python ``[:-1]`` for end_idx=-1 and drops the first
    # differentiated pose, so persist the exact retained source frame mapping.
    retained_indices = canonical.source_metadata["source_frame_indices"][1:-1]
    if len(retained_indices) != qpos.shape[0]:
        raise ValueError("OUTPUT_VALIDATION_FAILED: source-frame mapping length does not match trajectory")
    source_mapping = {"source_sequence_id": canonical.source_sequence_id, "source_frame_indices": retained_indices, "canonical_dir": str(paths.workspace_root / "processed" / "grab" / "canonical" / task), "source_replay": str(paths.workspace_root / "processed" / "grab" / "canonical" / task / "visualization" / "source_replay.mp4")}
    (robot_dir / "source_mapping.json").write_text(json.dumps(source_mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (robot_dir / "run_config.json").write_text(json.dumps({"robot_asset": "wuji_hand2_beta1", "embodiment": embodiment, "frequency_hz": frequency, "spider_ik": "ik_fast", "wrist_pos_cost": 10.0, "finger_pos_cost": 1.0, "wrist_init_steps": 200, "finger_init_steps": 300, "config_hash": config_hash({"sequence": task, "embodiment": embodiment, "frequency": frequency})}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ref_slice = slice(1, -1)
    tracking: dict[str, list[float]] = {"wrist": [], "thumb": [], "index": [], "middle": [], "ring": [], "pinky": []}
    data = mujoco.MjData(model)
    hand_refs = (("right", canonical.right_hand), ("left", canonical.left_hand))
    finger_names = ("thumb", "index", "middle", "ring", "pinky")
    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]; mujoco.mj_forward(model, data)
        for side, hand in hand_refs:
            if hand is None: continue
            wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
            tracking["wrist"].append(float(np.linalg.norm(data.site_xpos[wrist_id] - hand.global_translation[ref_slice][frame])))
            for name, joint_index in zip(finger_names, (4, 8, 12, 16, 20), strict=True):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{name}_tip")
                tracking[name].append(float(np.linalg.norm(data.site_xpos[site_id] - hand.joints_world[ref_slice][frame, joint_index])))
    tracking_summary = {name: {"mean_m": float(np.mean(values)), "rmse_m": float(np.sqrt(np.mean(np.square(values)))), "p95_m": float(np.percentile(values, 95)), "max_m": float(np.max(values))} for name, values in tracking.items() if values}
    smoke_config = Path(__file__).parents[2] / "configs" / "project" / "grab_ik_smoke.yaml"
    quality = yaml.safe_load(smoke_config.read_text(encoding="utf-8"))["quality"]
    worst_fingertip_rmse = max(item["rmse_m"] for name, item in tracking_summary.items() if name != "wrist")
    tracking_within_smoke = tracking_summary["wrist"]["rmse_m"] <= quality["wrist_rmse_m"] and worst_fingertip_rmse <= quality["fingertip_rmse_m"]
    status = "PASS" if violations == 0 else "OUTPUT_VALIDATION_FAILED"
    quality_status = "AUTO_PIPELINE_PASS" if tracking_within_smoke else "AUTO_PIPELINE_PASS_MANUAL_REVIEW_REQUIRED"
    metrics = {"status": status, "quality_status": quality_status, "frames": int(qpos.shape[0]), "qpos_dimension": int(qpos.shape[1]), "qvel_dimension": int(qvel.shape[1]), "model_nq": int(model.nq), "model_nv": int(model.nv), "model_nu": int(model.nu), "frequency_hz": frequency, "joint_limit_violations": violations, "minimum_joint_limit_margin_rad": min(margins) if margins else None, "maximum_velocity": float(np.abs(qvel).max()), "maximum_acceleration": float(np.abs(np.diff(qvel, axis=0) * frequency).max()) if len(qvel) > 1 else 0.0, "actuator_names": names, "source_frame_mapping_complete": len(source_mapping["source_frame_indices"]) == qpos.shape[0], "tracking_smoke_thresholds": quality, "tracking_errors": tracking_summary, "ik_replay": str(robot_dir / "visualization_ik.mp4")}
    (robot_dir / "metrics_kinematic.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise ValueError(f"OUTPUT_VALIDATION_FAILED: {violations} joint-limit violations")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"prepare": prepare, "render-source": render_source, "run-wuji-ik": run_wuji_ik})
