"""Self-contained Plotly acceptance viewers for canonical GRAB and Wuji IK."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro
from scipy.spatial.transform import Rotation

from spider.datasets.schema import CanonicalHOISequence


_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    return np.linspace(0, count - 1, min(count, maximum), dtype=int)


def _mesh(mesh_path: Path, max_faces: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) > max_faces:
        # A uniformly sampled face list makes an object look like a sparse
        # point cloud. Quadric simplification keeps a connected, inspectable
        # cup/hand surface at a bounded HTML size.
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _mesh_trace(vertices: np.ndarray, faces: np.ndarray, name: str, color: str, opacity: float) -> go.Mesh3d:
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name, color=color, opacity=opacity, flatshading=True, hoverinfo="skip")


def _skeleton(points: np.ndarray, name: str, color: str) -> go.Scatter3d:
    xs: list[float | None] = []; ys: list[float | None] = []; zs: list[float | None] = []
    for chain in _CHAINS:
        for index in chain: xs.append(float(points[index, 0])); ys.append(float(points[index, 1])); zs.append(float(points[index, 2]))
        xs.append(None); ys.append(None); zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", name=name, line={"color": color, "width": 5}, marker={"size": 3, "color": color})


def _layout(title: str, steps: list[dict]) -> dict:
    return {"title": title, "scene": {"aspectmode": "data", "xaxis": {"title": "X (m)"}, "yaxis": {"title": "Y (m)"}, "zaxis": {"title": "Z (m)"}, "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.15}}}, "updatemenus": [{"type": "buttons", "showactive": False, "x": 0.05, "y": 0, "buttons": [{"label": "播放", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]}, {"label": "暂停", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]}]}], "sliders": [{"active": 0, "currentvalue": {"prefix": "帧: "}, "steps": steps, "x": 0.12, "len": 0.82, "y": 0}]}


def source_html(canonical_dir: str, output: str | None = None, max_frames: int = 90) -> str:
    """Write an orbit/zoom/timeline canonical source acceptance HTML."""
    sequence = CanonicalHOISequence.load(canonical_dir)
    target = Path(output) if output else Path(canonical_dir) / "visualization" / "source_replay_interactive.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    obj = sequence.objects[0]
    vertices, faces = _mesh(Path(obj.source_metadata["resolved_local_mesh_path"]))
    indices = _sample_indices(sequence.num_frames, max_frames)
    def object_vertices(frame: int) -> np.ndarray:
        quat = obj.orientation[frame]
        return Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(vertices) + obj.translation[frame]
    first = object_vertices(int(indices[0]))
    data: list[go.BaseTraceType] = [_mesh_trace(first, faces, f"物体：{obj.object_name}", "#7db7e8", 0.72)]
    for side, hand, color in (("右手", sequence.right_hand, "#e45756"), ("左手", sequence.left_hand, "#3e7cb1")):
        data.append(_skeleton(hand.joints_world[indices[0]] if hand is not None else np.zeros((21, 3)), side, color))
    origin = obj.translation[indices[0]]
    axis = 0.12
    for vector, color, label in (([axis, 0, 0], "#d62728", "X"), ([0, axis, 0], "#2ca02c", "Y"), ([0, 0, axis], "#1f77b4", "Z")):
        end = origin + np.asarray(vector)
        data.append(go.Scatter3d(x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]], mode="lines+text", text=["", label], name=f"世界轴 {label}", line={"color": color, "width": 7}))
    frames_out = []
    steps = []
    for out_index, frame in enumerate(indices):
        updates: list[go.BaseTraceType] = [_mesh_trace(object_vertices(int(frame)), faces, f"物体：{obj.object_name}", "#7db7e8", 0.72)]
        for side, hand, color in (("右手", sequence.right_hand, "#e45756"), ("左手", sequence.left_hand, "#3e7cb1")):
            updates.append(_skeleton(hand.joints_world[frame] if hand is not None else np.zeros((21, 3)), side, color))
        frames_out.append(go.Frame(data=updates, name=str(out_index), traces=[0, 1, 2]))
        steps.append({"label": str(int(frame)), "method": "animate", "args": [[str(out_index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
    figure = go.Figure(data=data, frames=frames_out)
    figure.update_layout(**_layout(f"Canonical GRAB source | {sequence.source_sequence_id} | 拖动、旋转、缩放并检查世界轴", steps))
    figure.write_html(target, include_plotlyjs=True, full_html=True)
    print(target)
    return str(target)


def _robot_meshes(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return separate right/left visual meshes, preserving their real geometry."""
    parts: dict[str, tuple[list[np.ndarray], list[np.ndarray], int]] = {
        "right": ([], [], 0), "left": ([], [], 0),
    }
    for geom_id in range(model.ngeom):
        # Wuji visual meshes are explicitly group 1; group 2 contains the
        # collision duplicate and group 0 is reserved for scene geometry.
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom_id] != 1:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "object" in name:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        side = "right" if body.startswith("r_") else "left" if body.startswith("l_") else None
        if side is None:
            continue
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0: continue
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:start + count].copy()
        face_start, face_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        faces = model.mesh_face[face_start:face_start + face_count].copy()
        if faces.max(initial=0) >= count: faces -= start
        if len(faces) > 180:
            simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).simplify_quadric_decimation(face_count=180)
            vertices = np.asarray(simplified.vertices)
            faces = np.asarray(simplified.faces)
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        all_vertices, all_faces, offset = parts[side]
        all_vertices.append(vertices @ rotation.T + data.geom_xpos[geom_id]); all_faces.append(faces + offset)
        parts[side] = (all_vertices, all_faces, offset + len(vertices))
    return {side: (np.concatenate(vertices), np.concatenate(faces)) for side, (vertices, faces, _) in parts.items()}


def wuji_html(robot_dir: str, canonical_dir: str, output: str | None = None, max_frames: int = 40) -> str:
    """Write an interactive robot-vs-source acceptance HTML with real robot mesh."""
    robot = Path(robot_dir); sequence = CanonicalHOISequence.load(canonical_dir)
    target = Path(output) if output else robot / "visualization_ik_interactive.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(robot / "trajectory_kinematic.npz", allow_pickle=False) as archive: qpos = archive["qpos"]
    model = mujoco.MjModel.from_xml_path(str(robot.parent / "scene.xml")); data = mujoco.MjData(model)
    mapping = json.loads((robot / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"]
    canonical_lookup = {value: index for index, value in enumerate(sequence.source_metadata["source_frame_indices"])}
    obj = sequence.objects[0]
    object_vertices, object_faces = _mesh(Path(obj.source_metadata["resolved_local_mesh_path"]))
    frames_index = _sample_indices(len(qpos), max_frames)
    def robot_state(index: int):
        data.qpos[:] = qpos[index]; mujoco.mj_forward(model, data)
        meshes = _robot_meshes(model, data)
        hands = []
        for side in ("right", "left"):
            site_names = [f"{side}_palm", *[f"{side}_{finger}_tip" for finger in _FINGERS]]
            hands.append(np.vstack([data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)] for name in site_names]))
        source_index = canonical_lookup[mapping[index]]
        return meshes, hands, source_index
    def source_object(frame: int) -> np.ndarray:
        quat = obj.orientation[frame]
        return Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(object_vertices) + obj.translation[frame]
    meshes0, hands0, source0 = robot_state(int(frames_index[0]))
    data_traces: list[go.BaseTraceType] = [
        _mesh_trace(*meshes0["right"], "Wuji 右手（紫）", "#8b6bb3", 0.82),
        _mesh_trace(*meshes0["left"], "Wuji 左手（青绿）", "#2a9d8f", 0.82),
        _mesh_trace(source_object(source0), object_faces, f"源物体：{obj.object_name}", "#e9c46a", 0.48),
    ]
    for label, hand, color in (("源右手", sequence.right_hand, "#e45756"), ("源左手", sequence.left_hand, "#3e7cb1")):
        data_traces.append(_skeleton(hand.joints_world[source0] if hand else np.zeros((21, 3)), label, color))
    frames_out = []; steps = []
    for out_index, index in enumerate(frames_index):
        meshes, _, source_index = robot_state(int(index))
        updates = [
            _mesh_trace(*meshes["right"], "Wuji 右手（紫）", "#8b6bb3", 0.82),
            _mesh_trace(*meshes["left"], "Wuji 左手（青绿）", "#2a9d8f", 0.82),
            _mesh_trace(source_object(source_index), object_faces, f"源物体：{obj.object_name}", "#e9c46a", 0.48),
            _skeleton(sequence.right_hand.joints_world[source_index] if sequence.right_hand else np.zeros((21, 3)), "源右手（红）", "#e45756"),
            _skeleton(sequence.left_hand.joints_world[source_index] if sequence.left_hand else np.zeros((21, 3)), "源左手（蓝）", "#3e7cb1"),
        ]
        frames_out.append(go.Frame(data=updates, name=str(out_index), traces=[0, 1, 2, 3, 4]))
        steps.append({"label": str(mapping[index]), "method": "animate", "args": [[str(out_index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
    figure = go.Figure(data=data_traces, frames=frames_out)
    figure.update_layout(**_layout("Wuji IK：右紫、左青绿；源：右红、左蓝、物体黄 | 拖动、旋转、缩放，滑块为原始 GRAB frame", steps))
    figure.write_html(target, include_plotlyjs=True, full_html=True)
    print(target)
    return str(target)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"source": source_html, "wuji": wuji_html})
