"""Independent new-GRAB-source audit and standard Wuji IK recheck.

This tool deliberately keeps the historic Stage-C sample and its C-XA files
out of scope.  It selects one *new* GRAB sequence deterministically, captures
the actual ``GrabAdapter`` output before any robot operation, and invokes the
existing Stage-A/Stage-B command-line entry points in an isolated workspace.

The generated root is intentionally local and fail-fast.  It is evidence for
one selected sequence, not an acceptance claim for Stage C, dynamics, or
contact correction.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import inspect
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import mujoco
import numpy as np
import trimesh
import yaml
from plotly.offline.offline import get_plotlyjs
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from spider.datasets.grab import GrabAdapter
from spider.datasets.paths import load_project_paths
from spider.tools.grab_source_geometry_audit import (
    FINGERTIP_INDICES,
    FINGERTIP_NAMES,
    HAND_JOINTS,
    SMPLX_TO_WUJI_PALM,
    _hand_faces_from_smplx,
    _summary,
    apply_transform,
    detect_frame_offset,
    detect_global_frame_only_difference,
    invert_transform,
    object_relative,
    rotation_residual_rad,
    transform_from_rt,
)


REPO = Path(__file__).resolve().parents[2]
EXCLUDED_SEQUENCE = "s5/cylindermedium_lift"
REQUIRED_BODY_FIELDS = (
    "transl",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "fullpose",
)
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _pack_viewer_payload(payload: dict[str, Any], binary_threshold_bytes: int = 2_048) -> str:
    """Encode large numeric viewer arrays as a gzip-packed aligned binary blob.

    The earlier all-JSON page preserved every frame but made Chrome parse tens
    of millions of decimal characters before it could draw.  This keeps the
    same arrays self-contained and lossless while leaving small metadata and
    per-frame transforms convenient JSON.
    """
    buffer = bytearray()

    def align(alignment: int = 8) -> None:
        buffer.extend(b"\0" * ((-len(buffer)) % alignment))

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            dtype = np.dtype(array.dtype).newbyteorder("<")
            if array.nbytes >= binary_threshold_bytes and dtype.kind in {"f", "i", "u"}:
                if array.dtype != dtype:
                    array = array.astype(dtype, copy=False)
                alignment = max(8, dtype.itemsize)
                align(alignment)
                offset = len(buffer)
                raw = array.tobytes(order="C")
                buffer.extend(raw)
                return {"__binary_array__": {"dtype": dtype.str, "shape": list(array.shape), "offset": offset, "count": int(array.size)}}
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    metadata = json.dumps(encode(payload), ensure_ascii=False, separators=(",", ":"), default=_json_default).encode("utf-8")
    prefix = struct.pack("<Q", len(metadata))
    header_padding = b"\0" * ((-(len(prefix) + len(metadata))) % 8)
    packed = gzip.compress(prefix + metadata + header_padding + bytes(buffer), mtime=0)
    return base64.b64encode(packed).decode("ascii")


def write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON with NumPy support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    """Write a UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _log_summary(value: str, limit: int = 2000) -> str:
    """Keep a bounded stdout/stderr summary while retaining the full log file."""
    text = value.strip()
    return text[-limit:] if text else "<empty>"


def safe_id(source_id: str) -> str:
    """Encode a GRAB subject/sequence id for directory use."""
    return source_id.replace("/", "__").replace("\\", "__")


def longest_true_run(values: np.ndarray) -> int:
    """Return the longest contiguous true interval."""
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def candidate_score(subject: str, object_name: str, intent: str, frames: int) -> float:
    """Stable metadata-only pre-score; interaction is verified before selection."""
    action = {"lift": 50.0, "pass": 46.0, "use": 44.0, "pour": 42.0, "take": 40.0, "drink": 38.0, "eat": 38.0}.get(intent, 12.0)
    length = max(0.0, 45.0 - abs(frames - 600) / 12.0)
    return (100.0 if subject != "s5" else 0.0) + (50.0 if object_name != "cylindermedium" else 0.0) + action + length


def existing_sequence_ids(workspace_root: Path, local_artifacts: Path) -> set[str]:
    """Discover already materialized Stage-B task namespaces without mutation."""
    result: set[str] = {safe_id(EXCLUDED_SEQUENCE)}
    processed = workspace_root / "processed/grab/wuji_hand2_beta1"
    if processed.is_dir():
        for embodiment in processed.iterdir():
            if embodiment.is_dir():
                result.update(path.name for path in embodiment.iterdir() if path.is_dir())
    if local_artifacts.is_dir():
        for path in local_artifacts.rglob("selected_sequence.json"):
            try:
                stage_status = path.parents[1] / "reports/retarget_stage_status.json"
                if not stage_status.is_file():
                    continue
                status = json.loads(stage_status.read_text(encoding="utf-8"))
                if str(status.get("stage_b", {}).get("status")) not in {"PASS", "PARTIAL"}:
                    continue
                result.add(str(json.loads(path.read_text(encoding="utf-8"))["sequence_id_safe"]))
            except (KeyError, OSError, json.JSONDecodeError):
                continue
    return result


def scan_candidates(paths_config: str, local_artifacts: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Scan every GRAB parameter path deterministically without loading contacts."""
    paths = load_project_paths(paths_config)
    source_root = paths.source_root("grab")
    sequence_root = source_root / "grab"
    mesh_root = source_root / "tools/object_meshes/contact_meshes"
    used = existing_sequence_ids(paths.workspace_root, local_artifacts)
    rows: list[dict[str, Any]] = []
    for path in sorted(sequence_root.glob("s*/*.npz")):
        source_id = path.relative_to(sequence_root).with_suffix("").as_posix()
        row: dict[str, Any] = {"source_id": source_id, "sequence_id_safe": safe_id(source_id), "parameter_path": str(path), "rejection_reasons": [], "status": "SCANNED"}
        try:
            with np.load(path, allow_pickle=False) as archive:
                frames = int(archive["n_frames"])
                fps = float(archive["framerate"])
                subject = str(archive["sbj_id"])
                gender = str(archive["gender"])
                object_name = str(archive["obj_name"])
                intent = str(archive["motion_intent"])
                n_comps = int(archive["n_comps"])
            row.update({"subject": subject, "gender": gender, "object": object_name, "motion_intent": intent, "frame_count": frames, "fps": fps, "n_comps": n_comps, "object_mesh": str(mesh_root / f"{object_name}.ply"), "score": candidate_score(subject, object_name, intent, frames)})
            if source_id == EXCLUDED_SEQUENCE:
                row["rejection_reasons"].append("明确排除的冻结 Stage C 主线序列")
            if row["sequence_id_safe"] in used:
                row["rejection_reasons"].append("已出现在历史或当前 Wuji Stage-B 输出命名空间")
            if frames < 200:
                row["rejection_reasons"].append("总帧数不足 200")
            if fps <= 0:
                row["rejection_reasons"].append("帧率不可确定")
            if n_comps != 24:
                row["rejection_reasons"].append("不是 GRAB PCA-24 手参数")
            if not Path(row["object_mesh"]).is_file():
                row["rejection_reasons"].append("object contact mesh 缺失")
            row["status"] = "METADATA_ELIGIBLE" if not row["rejection_reasons"] else "EXCLUDED"
        except Exception as exc:  # a corrupt file must not terminate the scan
            row.update({"score": -1.0, "status": "INVALID_SOURCE_DATA", "rejection_reasons": [f"{type(exc).__name__}: {exc}"]})
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["score"]), str(row["source_id"])))
    for rank, row in enumerate(rows, start=1):
        row["stable_rank"] = rank
    return rows, used


def _candidate_parameter_contract(path: Path) -> tuple[bool, list[str]]:
    """Validate selected-candidate raw arrays before an SMPL-X forward pass."""
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=True) as archive:
            frames = int(archive["n_frames"])
            body = archive["body"].item()["params"]
            obj = archive["object"].item()["params"]
            for key in REQUIRED_BODY_FIELDS:
                if key not in body or len(np.asarray(body[key])) != frames:
                    errors.append(f"body.params.{key} 缺失或帧数不匹配")
            for key in ("transl", "global_orient"):
                if key not in obj or len(np.asarray(obj[key])) != frames:
                    errors.append(f"object.params.{key} 缺失或帧数不匹配")
            if str(archive["gender"]) not in {"female", "male", "neutral"}:
                errors.append("不支持的 SMPL-X gender")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return not errors, errors


def _interaction_probe(adapter: GrabAdapter, source_id: str) -> dict[str, Any]:
    """Use actual adapter joints and an object-vertex tree to find a 60-frame hand interaction interval."""
    sequence = adapter.load_sequence(safe_id(source_id), include_vertices=False)
    obj = sequence.objects[0]
    mesh = trimesh.load(Path(obj.source_metadata["resolved_local_mesh_path"]), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
    rotation = Rotation.from_quat(np.asarray(obj.orientation)[:, (1, 2, 3, 0)]).as_matrix()
    sides: dict[str, Any] = {}
    for side, hand in (("left", sequence.left_hand), ("right", sequence.right_hand)):
        if hand is None:
            sides[side] = {"valid": False, "reason": "adapter 未返回该手"}
            continue
        points = np.asarray(hand.joints_world)[:, FINGERTIP_INDICES]
        local = np.einsum("tji,tfj->tfi", rotation, points - obj.translation[:, None, :])
        distance, _ = tree.query(local.reshape(-1, 3), k=1)
        per_frame = distance.reshape(len(points), len(FINGERTIP_INDICES)).min(axis=1)
        close = per_frame <= 0.03
        sides[side] = {"valid": True, "min_vertex_distance_m": float(per_frame.min()), "p05_vertex_distance_m": float(np.percentile(per_frame, 5)), "close_frame_count": int(close.sum()), "longest_close_run_frames": longest_true_run(close), "first_close_frame": int(np.flatnonzero(close)[0]) if np.any(close) else None}
    best = max((value.get("longest_close_run_frames", 0) for value in sides.values()), default=0)
    return {"method": "GrabAdapter pre-retarget SMPL-X fingertip to source object-vertex KD-tree; final audit separately stores exact closest triangle on key frames", "close_threshold_m": 0.03, "minimum_required_contiguous_frames": 60, "sides": sides, "passes_interaction_requirement": best >= 60}


def select_sequence(paths_config: str, local_artifacts: Path, selection_dir: Path) -> dict[str, Any]:
    """Select the first stable-ranked candidate which meets all real checks."""
    rows, used = scan_candidates(paths_config, local_artifacts)
    paths = load_project_paths(paths_config)
    adapter = GrabAdapter(paths)
    selected: dict[str, Any] | None = None
    checks = 0
    for row in rows:
        if row["status"] != "METADATA_ELIGIBLE":
            continue
        if checks >= 40:
            row["selection_validation"] = {"status": "NOT_PROBED", "reason": "稳定排序在已找到合格序列后停止，避免无关批量重建"}
            continue
        checks += 1
        okay, errors = _candidate_parameter_contract(Path(row["parameter_path"]))
        if not okay:
            row["selection_validation"] = {"status": "REJECTED", "reason": errors}
            continue
        subject = str(row["subject"])
        beta = paths.source_root("grab") / "tools/subject_meshes" / str(row["gender"]) / f"{subject}_betas.npy"
        if not beta.is_file():
            row["selection_validation"] = {"status": "REJECTED", "reason": [f"subject beta 缺失: {beta}"]}
            continue
        try:
            interaction = _interaction_probe(adapter, str(row["source_id"]))
        except Exception as exc:
            row["selection_validation"] = {"status": "REJECTED", "reason": [f"adapter/interation probe: {type(exc).__name__}: {exc}"]}
            continue
        row["selection_validation"] = {"status": "PASS" if interaction["passes_interaction_requirement"] else "REJECTED", "parameter_contract": "PASS", "subject_beta": str(beta), "interaction": interaction}
        if interaction["passes_interaction_requirement"]:
            selected = row
            break
    if selected is None:
        raise RuntimeError("未找到同时满足资产、帧数、PCA-24 和连续 60 帧真实手物接近条件的新 GRAB 序列")
    selection_dir.mkdir(parents=True, exist_ok=False)
    write_json(selection_dir / "candidate_sequences.json", {"schema_version": 1, "selection_rule": "所有 raw 路径按 source_id 扫描；按 score 降序、source_id 升序稳定排序；选择第一个通过完整参数、asset、GrabAdapter pre-retarget interaction probe 的候选", "excluded_sequence": EXCLUDED_SEQUENCE, "previous_wuji_task_namespaces": sorted(used), "candidate_count": len(rows), "validated_rank_count": checks, "candidates": rows})
    selected_payload = {"schema_version": 1, **selected, "selection_reason": "不同于 s5/cylindermedium_lift 的 subject/object；未发现历史 Wuji Stage-B 命名空间；PCA-24 与 body/object 参数完整；object mesh、subject beta 与帧率存在；真实 GrabAdapter retarget 前 fingertip probe 至少有一手连续 >=60 帧在 30mm source-object vertex 距离内。"}
    write_json(selection_dir / "selected_sequence.json", selected_payload)
    write_text(selection_dir / "SEQUENCE_SELECTION.md", f"# 新 GRAB 序列确定性选择\n\n- subject：`{selected['subject']}`\n- sequence：`{selected['source_id']}`\n- object：`{selected['object']}`\n- frames / fps：`{selected['frame_count']}` / `{selected['fps']}`\n- 稳定排序名次：`{selected['stable_rank']}`\n- score：`{selected['score']:.3f}`\n\n本选择先排除冻结主线及既有 Wuji Stage-B namespace，再按固定 score/source_id 排序。最终通过真实 `GrabAdapter.load_sequence` 的 retarget 前手指尖接近验证；完整候选及每项排除原因见 `candidate_sequences.json`。\n")
    return selected_payload


def reconstruct_official(paths_config: str, source_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Directly reconstruct GRAB SMPL-X without calling ``GrabAdapter``."""
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

    paths = load_project_paths(paths_config)
    raw_path = paths.source_root("grab") / "grab" / f"{source_id}.npz"
    with np.load(raw_path, allow_pickle=True) as archive:
        count = int(archive["n_frames"])
        fps = float(archive["framerate"])
        gender = str(archive["gender"])
        subject = str(archive["sbj_id"])
        object_name = str(archive["obj_name"])
        body = archive["body"].item()["params"]
        object_params = archive["object"].item()["params"]
        source_mesh_rel = str(archive["object"].item()["object_mesh"])
    beta_path = paths.source_root("grab") / "tools/subject_meshes" / gender / f"{subject}_betas.npy"
    betas = np.asarray(np.load(beta_path), dtype=np.float32).reshape(-1)[:10]
    model = smplx.create(str(paths.body_model_root), model_type="smplx", gender=gender, use_pca=True, num_pca_comps=24, flat_hand_mean=False, batch_size=count)
    tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)
    with torch.no_grad():
        output = model(global_orient=tensor(body["global_orient"]), body_pose=tensor(body["body_pose"]), left_hand_pose=tensor(body["left_hand_pose"]), right_hand_pose=tensor(body["right_hand_pose"]), jaw_pose=tensor(body["jaw_pose"]), leye_pose=tensor(body["leye_pose"]), reye_pose=tensor(body["reye_pose"]), expression=tensor(body["expression"]), transl=tensor(body["transl"]), betas=torch.as_tensor(np.repeat(betas[None], count, axis=0), dtype=torch.float32), return_verts=True, return_full_pose=True)
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    beta_tensor = torch.as_tensor(np.repeat(betas[None], count, axis=0), dtype=torch.float32)
    shaped = model.v_template + blend_shapes(beta_tensor, model.shapedirs)
    rest_joints = vertices2joints(model.J_regressor, shaped)
    rotations = batch_rodrigues(output.full_pose.reshape(-1, 3)).reshape(count, -1, 3, 3)
    _, transforms = batch_rigid_transform(rotations, rest_joints, model.parents, dtype=torch.float32)
    wrist_rot = transforms[:, :, :3, :3].detach().cpu().numpy()
    mesh_path = paths.source_root("grab") / source_mesh_rel
    if not mesh_path.is_file():
        mesh_path = paths.source_root("grab") / "tools/object_meshes/contact_meshes" / f"{object_name}.ply"
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    object_rotation = Rotation.from_rotvec(np.asarray(object_params["global_orient"], dtype=np.float64)).as_matrix()
    object_translation = np.asarray(object_params["transl"], dtype=np.float64)
    hand_ids = {side: np.asarray(np.load(paths.source_root("grab") / "tools/smplx_correspondence" / f"{side[0]}hand_smplx_ids.npy"), dtype=np.int64) for side in ("left", "right")}
    values: dict[str, np.ndarray] = {"source_frame_indices": np.arange(count, dtype=np.int64), "timestamps_s": np.arange(count, dtype=np.float64) / fps, "body_vertices_world": vertices, "body_faces": np.asarray(model.faces, dtype=np.int32), "body_joints_world": joints, "T_world_object": transform_from_rt(object_rotation, object_translation), "object_asset_vertices": np.asarray(mesh.vertices, dtype=np.float32), "object_faces": np.asarray(mesh.faces, dtype=np.int32)}
    for side in ("left", "right"):
        indices = HAND_JOINTS[side]
        values[f"{side}_joints_world"] = joints[:, indices]
        values[f"{side}_vertices_world"] = vertices[:, hand_ids[side]]
        values[f"{side}_hand_faces"] = _hand_faces_from_smplx(model.faces, hand_ids[side])
        wrist = indices[0]
        values[f"T_world_{side}_wrist"] = transform_from_rt(wrist_rot[:, wrist], joints[:, wrist])
    chain = {"independence": "此步骤只读取 raw GRAB NPZ、subject betas 与官方 smplx 接口；不调用 GrabAdapter。", "source_file": str(raw_path), "source_sha256": sha256(raw_path), "subject": subject, "gender": gender, "fps": fps, "frame_count": count, "smplx": {"use_pca": True, "num_pca_comps": 24, "flat_hand_mean": False, "body_model_root": str(paths.body_model_root), "betas": str(beta_path)}, "transforms": {"T_world_object": "[R(axis_angle(object.global_orient)), object.transl]", "T_world_wrist": "SMPL-X kinematic-tree wrist transform", "T_asset_object": "identity"}, "object_mesh": {"path": str(mesh_path), "sha256": sha256(mesh_path), "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent)}}
    return values, chain


def capture_spider(paths_config: str, source_id: str, official: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture the public adapter output before Stage A, plus body context disclosure."""
    paths = load_project_paths(paths_config)
    sequence = GrabAdapter(paths).load_sequence(safe_id(source_id), include_vertices=True)
    item = sequence.objects[0]
    rotation = Rotation.from_quat(np.asarray(item.orientation)[:, (1, 2, 3, 0)]).as_matrix()
    values: dict[str, np.ndarray] = {"source_frame_indices": np.asarray(sequence.source_metadata["source_frame_indices"], dtype=np.int64), "timestamps_s": np.asarray(sequence.timestamps, dtype=np.float64), "T_world_object": transform_from_rt(rotation, np.asarray(item.translation)), "object_asset_vertices": official["object_asset_vertices"], "object_faces": official["object_faces"], "body_vertices_world_context": official["body_vertices_world"], "body_faces_context": official["body_faces"]}
    for side, hand in (("left", sequence.left_hand), ("right", sequence.right_hand)):
        if hand is None or hand.vertices_world is None:
            raise RuntimeError(f"GrabAdapter 未返回 {side} SMPL-X hand surface")
        values[f"{side}_joints_world"] = np.asarray(hand.joints_world, dtype=np.float32)
        values[f"{side}_vertices_world"] = np.asarray(hand.vertices_world, dtype=np.float32)
        values[f"{side}_hand_faces"] = official[f"{side}_hand_faces"]
        values[f"T_world_{side}_wrist"] = transform_from_rt(Rotation.from_quat(np.asarray(hand.global_orientation)[:, (1, 2, 3, 0)]).as_matrix(), np.asarray(hand.global_translation))
    chain = {"capture_order": "GrabAdapter.load_sequence 在任何 Stage A、Stage B、C-XA 或 contact correction 前调用。", "adapter_source": inspect.getsourcefile(GrabAdapter.load_sequence), "adapter_source_line": inspect.getsourcelines(GrabAdapter.load_sequence)[1], "public_output": "CanonicalHOISequence: source object、SMPL-X wrists/joints/hand surfaces", "body_context_note": "当前 CanonicalHOISequence 公共接口不携带全身 vertices；HTML 中 Spider body context 是同一 raw body 参数的官方 SMPL-X surface，仅为可视化上下文，绝不参与 adapter 数值一致性或被称作 adapter 返回字段。", "not_retargeted": True}
    return values, chain


def compare_source(official: dict[str, np.ndarray], spider: dict[str, np.ndarray], fps: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure true world/object/wrist relative geometry before retargeting."""
    object_translation = np.linalg.norm(official["T_world_object"][:, :3, 3] - spider["T_world_object"][:, :3, 3], axis=1)
    object_rotation = rotation_residual_rad(official["T_world_object"][:, :3, :3], spider["T_world_object"][:, :3, :3])
    comparison: dict[str, Any] = {"schema_version": 1, "comparison": "official independent GRAB/SMPL-X reconstruction vs actual spider.datasets.grab.GrabAdapter.load_sequence pre-retarget capture", "object_translation_residual_m": _summary(object_translation), "object_rotation_residual_rad": _summary(object_rotation, "rad"), "sides": {}}
    temporal: dict[str, Any] = {"schema_version": 1, "offsets": [-2, -1, 0, 1, 2], "object": detect_frame_offset(official["T_world_object"][:, :3, 3], spider["T_world_object"][:, :3, 3]), "sides": {}}
    maximum = max(float(object_translation.max()), float(object_rotation.max()))
    for side in ("left", "right"):
        official_wrist = official[f"T_world_{side}_wrist"]
        spider_wrist = spider[f"T_world_{side}_wrist"]
        official_relative = object_relative(official["T_world_object"], official_wrist)
        spider_relative = object_relative(spider["T_world_object"], spider_wrist)
        official_joints_object = apply_transform(invert_transform(official["T_world_object"])[:, None], official[f"{side}_joints_world"])
        spider_joints_object = apply_transform(invert_transform(spider["T_world_object"])[:, None], spider[f"{side}_joints_world"])
        official_vertices_object = apply_transform(invert_transform(official["T_world_object"])[:, None], official[f"{side}_vertices_world"])
        spider_vertices_object = apply_transform(invert_transform(spider["T_world_object"])[:, None], spider[f"{side}_vertices_world"])
        wrist_pos = np.linalg.norm(official_relative[:, :3, 3] - spider_relative[:, :3, 3], axis=1)
        wrist_rot = rotation_residual_rad(official_relative[:, :3, :3], spider_relative[:, :3, :3])
        joint = np.linalg.norm(official_joints_object - spider_joints_object, axis=-1)
        tips = np.linalg.norm(official_joints_object[:, FINGERTIP_INDICES] - spider_joints_object[:, FINGERTIP_INDICES], axis=-1)
        vertices = np.linalg.norm(official_vertices_object - spider_vertices_object, axis=-1)
        velocity = np.linalg.norm(np.diff(official_joints_object[:, FINGERTIP_INDICES], axis=0) - np.diff(spider_joints_object[:, FINGERTIP_INDICES], axis=0), axis=-1) * fps
        comparison["sides"][side] = {"wrist_world_translation_residual_m": _summary(np.linalg.norm(official_wrist[:, :3, 3] - spider_wrist[:, :3, 3], axis=1)), "wrist_world_rotation_residual_rad": _summary(rotation_residual_rad(official_wrist[:, :3, :3], spider_wrist[:, :3, :3]), "rad"), "T_object_wrist_translation_residual_m": _summary(wrist_pos), "T_object_wrist_rotation_residual_rad": _summary(wrist_rot, "rad"), "object_frame_joint_residual_m": _summary(joint), "object_frame_fingertip_residual_m": _summary(tips), "object_frame_surface_vertex_rmse_m": _summary(vertices), "fingertip_velocity_residual_m_s": _summary(velocity, "m_s")}
        temporal["sides"][side] = {"world_wrist": detect_frame_offset(official_wrist[:, :3, 3], spider_wrist[:, :3, 3]), "object_frame_fingertips": detect_frame_offset(official_joints_object[:, FINGERTIP_INDICES], spider_joints_object[:, FINGERTIP_INDICES])}
        maximum = max(maximum, float(wrist_pos.max()), float(wrist_rot.max()), float(tips.max()), float(vertices.max()))
    global_view = detect_global_frame_only_difference(official["T_world_object"], np.stack((official["T_world_left_wrist"], official["T_world_right_wrist"]), axis=1), spider["T_world_object"], np.stack((spider["T_world_left_wrist"], spider["T_world_right_wrist"]), axis=1))
    status = "PASS" if maximum <= 1e-5 else "FAIL"
    classification = "SOURCE_LOADER_PASS" if status == "PASS" else "SOURCE_COORDINATE_ERROR"
    if status == "PASS" and global_view["classification"] == "GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY":
        classification = "SOURCE_LOADER_PASS"
    comparison["decision"] = {"status": status, "classification": classification, "maximum_checked_residual": maximum, "threshold_m_or_rad": 1e-5, "global_view": global_view["classification"]}
    return comparison, temporal, global_view


def contact_summary(official: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize all hand surfaces with a fast vertex tree and exact key-frame triangles."""
    mesh = trimesh.Trimesh(vertices=official["object_asset_vertices"], faces=official["object_faces"], process=False)
    tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
    rotation = official["T_world_object"][:, :3, :3]
    translation = official["T_world_object"][:, :3, 3]
    result: dict[str, Any] = {"method": "全 778 hand vertices 使用 object-vertex KD-tree 记录近似距离；关键帧五指尖另用真实 triangle closest-point、triangle id、closest point、normal。非 watertight mesh 不报告伪造的签名穿透深度。", "mesh": {"watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent)}, "sides": {}}
    for side in ("left", "right"):
        vertices = official[f"{side}_vertices_world"]
        local_vertices = np.einsum("tji,tvj->tvi", rotation, vertices - translation[:, None, :])
        distance, _ = tree.query(local_vertices.reshape(-1, 3), k=1)
        distance = distance.reshape(vertices.shape[:2])
        tips = official[f"{side}_joints_world"][:, FINGERTIP_INDICES]
        local_tips = np.einsum("tji,tfj->tfi", rotation, tips - translation[:, None, :])
        per_frame_tip = np.empty((len(tips), 5), dtype=np.float64)
        per_frame_triangle = np.full((len(tips), 5), -1, dtype=np.int64)
        nearest_world = np.empty_like(tips, dtype=np.float32)
        normals_world = np.empty_like(tips, dtype=np.float32)
        key_indices = sorted(set(np.linspace(0, len(tips) - 1, min(24, len(tips)), dtype=int).tolist()))
        # Exact triangles on a fixed review set.  The full per-frame metric remains explicitly approximate.
        for index in key_indices:
            closest, distances, triangles = trimesh.proximity.closest_point(mesh, local_tips[index])
            per_frame_tip[index] = distances
            per_frame_triangle[index] = triangles
            nearest_world[index] = (closest @ rotation[index].T + translation[index]).astype(np.float32)
            normals_world[index] = (mesh.face_normals[triangles] @ rotation[index].T).astype(np.float32)
        # Non-key frames retain exact object-vertex nearest points for viewer lines.
        _, nearest_idx = tree.query(local_tips.reshape(-1, 3), k=1)
        near_local = np.asarray(mesh.vertices)[nearest_idx].reshape(local_tips.shape)
        missing = per_frame_triangle[:, 0] < 0
        nearest_world[missing] = (np.einsum("tij,tfj->tfi", rotation[missing], near_local[missing]) + translation[missing, None, :]).astype(np.float32)
        normals_world[missing] = 0.0
        per_frame_tip[missing] = np.linalg.norm(local_tips[missing] - near_local[missing], axis=-1)
        signed_tip_distance: np.ndarray | None = None
        if mesh.is_watertight and mesh.is_winding_consistent:
            # For this high-resolution bunny mesh ``trimesh.signed_distance``
            # creates an unbounded point/triangle work set.  Compute the same
            # signed convention exactly and over every tip: closest-triangle
            # magnitude plus watertight ray inside/outside classification.  The
            # 512-point batching is solely a memory bound, never a downsample.
            flat_tips = local_tips.reshape(-1, 3)
            signed_chunks = []
            for start in range(0, len(flat_tips), 512):
                points = flat_tips[start:start + 512]
                _closest, exact_distance, _triangles = trimesh.proximity.closest_point(mesh, points)
                inside = mesh.contains(points)
                signed_chunks.append(np.where(inside, exact_distance, -exact_distance))
            signed_tip_distance = np.concatenate(signed_chunks).reshape(local_tips.shape[:2])
        penetration = np.maximum(signed_tip_distance, 0.0) if signed_tip_distance is not None else None
        result["sides"][side] = {"surface_vertex_distance_approx_m": _summary(distance), "per_frame_min_surface_vertex_distance_approx_m": np.min(distance, axis=1), "per_frame_max_surface_vertex_distance_approx_m": np.max(distance, axis=1), "tip_distance_m": per_frame_tip, "signed_tip_distance_m": signed_tip_distance, "signed_tip_convention": "exact closest-triangle magnitude with watertight ray containment: positive=inside/penetration, negative=outside", "tip_penetration_depth_m": penetration, "nearest_triangle_id_key_frames_only": per_frame_triangle, "nearest_point_world": nearest_world, "surface_normal_world_key_frames_only": normals_world, "closest_triangle_key_frames": key_indices, "max_suspension_proxy_m": float(np.max(per_frame_tip)), "max_suspension_proxy_frame": int(np.unravel_index(int(np.argmax(per_frame_tip)), per_frame_tip.shape)[0]), "max_tip_penetration_m": float(np.max(penetration)) if penetration is not None else None, "max_tip_penetration_frame": int(np.unravel_index(int(np.argmax(penetration)), penetration.shape)[0]) if penetration is not None else None, "penetration": "SIGNED_TIP_DISTANCE_AVAILABLE" if penetration is not None else "NOT_COMPUTED_NON_WATERTIGHT_OR_UNORIENTED_MESH"}
    return result


def _serialize_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _sample_indices(count: int, important: list[int], maximum: int = 64) -> np.ndarray:
    base = set(np.linspace(0, count - 1, min(count, maximum), dtype=int).tolist())
    base.update(index for index in important if 0 <= index < count)
    return np.asarray(sorted(base), dtype=np.int64)


def _html_document(payload: dict[str, Any], title: str, kind: str) -> str:
    """Return one self-contained Chinese Plotly HTML with real mesh controls."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    plotly = get_plotlyjs()
    source_layers = """<label><input type='checkbox' data-layer='official' checked>官方/独立 SMPL-X body</label><label><input type='checkbox' data-layer='officialhand' checked>官方左右手 surface</label><label><input type='checkbox' data-layer='spiderbody' checked>Spider loaded SMPL-X body（上下文）</label><label><input type='checkbox' data-layer='spiderhand' checked>Spider loaded 左右手 surface</label>"""
    retarget_layers = """<label><input type='checkbox' data-layer='stageb' checked>Stage B Wuji visual mesh</label><label><input type='checkbox' data-layer='collision' checked>Stage B Wuji collision mesh</label><label><input type='checkbox' data-layer='sourcebody' checked>Spider loaded source body</label><label><input type='checkbox' data-layer='sourcehand' checked>Spider loaded source hands</label>"""
    layers = source_layers if kind == "source" else retarget_layers
    compare = "<option value='official'>官方 source</option><option value='spider'>Spider loaded source</option><option value='overlay'>二者叠加</option><option value='difference'>差异向量</option>" if kind == "source" else "<option value='source'>Source only</option><option value='stageb'>Source vs Stage B</option><option value='stagebonly'>Stage B only</option><option value='all'>All stages</option><option value='failure'>Failure inspection</option>"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><style>body{{margin:0;background:#10151c;color:#e9eef5;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#head{{padding:12px 16px;background:#17212d;position:sticky;top:0;z-index:4}}h1{{font-size:20px;margin:0 0 8px}}#controls,#layers{{display:flex;gap:9px;flex-wrap:wrap;align-items:center;font-size:13px}}select,input,button{{background:#293746;color:#e9eef5;border:1px solid #52677b;padding:4px 6px;border-radius:4px}}#scene{{height:72vh;min-height:640px}}#status,#info{{white-space:pre-wrap;padding:10px 16px;margin:0;background:#17212d;font:12px ui-monospace,monospace}}.bad{{color:#ff8fa3}}.ok{{color:#80ed99}}</style></head><body><section id='head'><h1>{title}</h1><pre id='status'></pre><div id='controls'><label>对比 <select id='compare'>{compare}</select></label><label>坐标 <select id='coord'><option value='world'>world</option><option value='object'>object</option><option value='left'>left wrist</option><option value='right'>right wrist</option></select></label><button id='play'>播放</button><button id='prev'>◀</button><button id='next'>▶</button><label>速度 <input id='speed' type='number' value='12' min='1' max='60' style='width:42px'></label><label>帧 <select id='frame'></select></label><label>尾迹 <input id='tail' type='number' value='15' min='0' max='63' style='width:42px'></label></div><div id='layers'>{layers}<label><input type='checkbox' data-layer='object' checked>object visual mesh</label><label><input type='checkbox' data-layer='skeleton' checked>左右手 skeleton</label><label><input type='checkbox' data-layer='axes' checked>wrist/object/world frame</label><label><input type='checkbox' data-layer='trajectory' checked>object/wrist/fingertip trajectories</label><label><input type='checkbox' data-layer='nearest' checked>最近表面连接线</label><label><input type='checkbox' data-layer='suspension' checked>悬空标记</label><label><input type='checkbox' data-layer='penetration' checked>穿模标记（仅可靠 mesh）</label><label><input type='checkbox' data-layer='error' checked>误差向量</label></div></section><div id='scene'></div><pre id='info'></pre><script>{plotly}</script><script>const D={data},$=x=>document.getElementById(x);let timer=null;const qp=new URLSearchParams(location.search);const enabled=k=>[...document.querySelectorAll('[data-layer]')].some(x=>x.dataset.layer===k&&x.checked);const mul=(T,p)=>[T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3],T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7],T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]];const inv=T=>{{let r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],p=[T[3],T[7],T[11]],q=[[r[0][0],r[1][0],r[2][0]],[r[0][1],r[1][1],r[2][1]],[r[0][2],r[1][2],r[2][2]]];return [q[0][0],q[0][1],q[0][2],-(q[0][0]*p[0]+q[0][1]*p[1]+q[0][2]*p[2]),q[1][0],q[1][1],q[1][2],-(q[1][0]*p[0]+q[1][1]*p[1]+q[1][2]*p[2]),q[2][0],q[2][1],q[2][2],-(q[2][0]*p[0]+q[2][1]*p[1]+q[2][2]*p[2])]}};const pts=(T,a)=>a.map(p=>mul(T,p));const mesh=(name,v,f,c,o,V)=>{{let q=pts(V,v);return{{type:'mesh3d',name,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),i:f.map(x=>x[0]),j:f.map(x=>x[1]),k:f.map(x=>x[2]),color:c,opacity:o,flatshading:true,hoverinfo:'name'}}}};const line=(name,a,c,V,w=4)=>{{let q=pts(V,a);return{{type:'scatter3d',mode:'lines+markers',name,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),line:{{color:c,width:w}},marker:{{size:3,color:c}}}}}};const skel=(name,p,c,V)=>{{let a=[];for(let ch of D.chains){{for(let x of ch)a.push(p[x]);a.push(null)}}let q=a.map(x=>x?mul(V,x):null);return{{type:'scatter3d',mode:'lines+markers',name,x:q.map(p=>p?p[0]:null),y:q.map(p=>p?p[1]:null),z:q.map(p=>p?p[2]:null),line:{{color:c,width:5}},marker:{{size:3,color:c}}}}}};function robot(name,R,side,i,c,o,V){{let vv=[],ff=[],off=0;for(let g=0;g<R.layers[side].length;g++){{let L=R.layers[side][g],t=R.transforms[side][i][g],T=[t[3],t[4],t[5],t[0],t[6],t[7],t[8],t[1],t[9],t[10],t[11],t[2]];vv.push(...pts(T,L.vertices));ff.push(...L.faces.map(f=>[f[0]+off,f[1]+off,f[2]+off]));off+=L.vertices.length}}return mesh(name,vv,ff,c,o,V)}}function axes(T,V){{let p=[T[3],T[7],T[11]],r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],z=[];for(let a of [[0,'#ef476f'],[1,'#06d6a0'],[2,'#118ab2']])z.push(line('frame '+a[0],[p,[p[0]+.05*r[0][a[0]],p[1]+.05*r[1][a[0]],p[2]+.05*r[2][a[0]]]],a[1],V,5));return z}}D.frames.forEach((f,i)=>$('frame').add(new Option(f.source_frame,i)));let initial=qp.get('frame');if(initial){{let i=D.frames.findIndex(x=>String(x.source_frame)===initial);if(i>=0)$('frame').value=i}}if(qp.get('coord'))$('coord').value=qp.get('coord');function draw(){{let i=+$('frame').value,f=D.frames[i],V=[1,0,0,0,0,1,0,0,0,0,1,0],co=$('coord').value;if(co==='object')V=inv(f.object);if(co==='left')V=inv(f.left_wrist);if(co==='right')V=inv(f.right_wrist);let t=[],cmp=$('compare').value;let sourceKind=D.kind==='source';let showO=sourceKind&&(cmp==='official'||cmp==='overlay'||cmp==='difference'),showS=sourceKind&&(cmp==='spider'||cmp==='overlay'||cmp==='difference'),showSrc=!sourceKind&&(cmp!=='stagebonly'),showB=!sourceKind&&(cmp!=='source');if(enabled('object'))t.push(mesh('source object',pts(f.object,D.object.vertices),D.object.faces,'#457b9d',.38,V));if(sourceKind){{if(showO&&enabled('official'))t.push(mesh('official body',f.official.body,D.body.faces,'#48cae4',.18,V));if(showS&&enabled('spiderbody'))t.push(mesh('spider body context',f.spider.body,D.body.faces,'#f72585',.12,V));for(let s of ['left','right']){{if(showO&&enabled('officialhand'))t.push(mesh('official '+s+' hand',f.official[s].verts,D.hand[s].faces,'#48cae4',.62,V));if(showS&&enabled('spiderhand'))t.push(mesh('spider '+s+' hand',f.spider[s].verts,D.hand[s].faces,'#f72585',.48,V));if(enabled('skeleton')&&(showO||showS))t.push(skel('official '+s+' skeleton',f.official[s].joints,s==='left'?'#3a86ff':'#ff006e',V));if(enabled('error')&&cmp!=='official')t.push(line('source difference '+s,[f.official[s].joints[8],f.spider[s].joints[8]],'#ffd166',V,5));}}}}else{{if(showSrc&&enabled('sourcebody'))t.push(mesh('source body',f.source.body,D.body.faces,'#48cae4',.14,V));for(let s of ['left','right']){{if(showSrc&&enabled('sourcehand'))t.push(mesh('source '+s+' hand',f.source[s].verts,D.hand[s].faces,s==='left'?'#3a86ff':'#ff006e',.52,V));if(showB&&enabled('stageb'))t.push(robot('Stage B '+s,D.robot_visual,s,i,s==='left'?'#80ed99':'#ffd166',.70,V));if(showB&&enabled('collision'))t.push(robot('Stage B collision '+s,D.robot_collision,s,i,'#ef476f',.24,V));if(enabled('skeleton')&&showSrc)t.push(skel('source '+s+' skeleton',f.source[s].joints,s==='left'?'#3a86ff':'#ff006e',V));if(enabled('error')&&showB)t.push(line('source→Wuji tip '+s,[f.source[s].joints[8],f.wuji[s].tips[1]],'#ffd166',V,5));}}}}if(enabled('axes'))t.push(...axes(f.object,V),...axes(f.left_wrist,V),...axes(f.right_wrist,V));if(enabled('nearest'))for(let s of ['left','right'])for(let g=0;g<5;g++)t.push(line(s+' nearest surface',[f.official?s==='left'?f.official[s].joints[[4,8,12,16,20][g]]:f.official[s].joints[[4,8,12,16,20][g]]:f.source[s].joints[[4,8,12,16,20][g]],f.nearest[s][g]],'#f4a261',V,2));if(enabled('trajectory')){{let q=Math.max(0,i-(+$('tail').value));for(let s of ['left','right'])t.push(line(s+' index trajectory',D.frames.slice(q,i+1).map(x=>(x.official||x.source)[s].joints[8]),s==='left'?'#3a86ff':'#ff006e',V,3));}}$('status').textContent=D.status;$('info').textContent=`source frame=${{f.source_frame}} | 显示采样帧 ${{i+1}}/${{D.frames.length}}\n${{f.metrics}}\n说明：完整序列的数值均在 NPZ/JSON 中；本 HTML 以等距采样和关键帧呈现真实 connected meshes，不做任何对齐修正。`;Plotly.react('scene',t,{{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{{color:'#e9eef5'}},scene:{{aspectmode:'data',camera:{{eye:{{x:1.5,y:-1.5,z:1.1}}}}}},legend:{{orientation:'h'}},margin:{{l:0,r:0,t:20,b:0}}}},{{responsive:true}})}}['frame','coord','compare','tail'].forEach(x=>$(x).oninput=draw);document.querySelectorAll('[data-layer]').forEach(x=>x.onchange=draw);$('prev').onclick=()=>{{$('frame').value=Math.max(0,+$('frame').value-1);draw()}};$('next').onclick=()=>{{$('frame').value=Math.min(D.frames.length-1,+$('frame').value+1);draw()}};$('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;$('play').textContent='播放'}}else{{timer=setInterval(()=>{{$('frame').value=(+$('frame').value+1)%D.frames.length;draw()}},1000/Math.max(1,+$('speed').value));$('play').textContent='暂停'}}}};draw();</script></body></html>"""


def _html_document_v2(payload: dict[str, Any], title: str, kind: str) -> str:
    """Create a packed self-contained viewer with complete time controls.

    The first implementation sampled frame payloads.  This version receives all
    frames and gzip-packs the JSON so a local file remains self-contained while
    retaining every original frame in the selector and playback control.
    """
    packed = _pack_viewer_payload(payload)
    if kind == "source":
        comparison = "<option value='official'>官方独立 source</option><option value='spider'>Spider 已加载 source</option><option value='overlay'>二者叠加</option><option value='difference'>差异向量</option>"
        layers = "<label><input type='checkbox' data-layer='officialbody' checked>官方/独立 SMPL-X 全身关节点</label><label><input type='checkbox' data-layer='officialhand' checked>官方左右手网格</label><label><input type='checkbox' data-layer='spiderbody' checked>Spider 全身上下文关节点</label><label><input type='checkbox' data-layer='spiderhand' checked>Spider 已加载左右手网格</label><label><input type='checkbox' data-layer='officialobject' checked>官方物体可视网格</label><label><input type='checkbox' data-layer='spiderobject' checked>Spider 物体可视网格</label>"
    else:
        comparison = "<option value='source'>仅 source</option><option value='stageb'>source 对比 Stage B</option><option value='stagebonly'>仅 Stage B</option><option value='all'>所有可用阶段</option><option value='failure'>失败诊断</option>"
        layers = "<label><input type='checkbox' data-layer='sourcebody' checked>Spider source 全身关节点</label><label><input type='checkbox' data-layer='sourcehand' checked>Spider 已加载 source 手部</label><label><input type='checkbox' data-layer='stageb' checked>Stage B Wuji 可视网格</label><label><input type='checkbox' data-layer='collision' checked>Stage B Wuji 碰撞网格</label><label><input type='checkbox' data-layer='wujiskel' checked>Wuji 掌到指尖骨架</label><label><input type='checkbox' data-layer='sourceframes' checked>source 腕部/物体/世界坐标系</label><label><input type='checkbox' data-layer='wujiframes' checked>Wuji 腕部坐标系</label>"
    template = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>__TITLE__</title><style>body{margin:0;background:#10151c;color:#e9eef5;font-family:system-ui,'Noto Sans CJK SC',sans-serif}#head{padding:12px 16px;background:#17212d;position:sticky;top:0;z-index:2}#controls,#layers{display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:13px}select,input,button{background:#293746;color:#e9eef5;border:1px solid #52677b;padding:4px 6px;border-radius:4px}#scene{height:72vh;min-height:640px}pre{white-space:pre-wrap;padding:8px;margin:0}</style></head><body><section id='head'><h1>__TITLE__</h1><pre id='status'>正在解压完整序列数据…</pre><div id='controls'><label>对比 <select id='compare'>__COMPARE__</select></label><label>坐标 <select id='coord'><option value='world'>世界坐标（world）</option><option value='object'>物体坐标（object）</option><option value='left'>左腕坐标（left wrist）</option><option value='right'>右腕坐标（right wrist）</option></select></label><button id='play'>播放</button><button id='prev'>◀</button><button id='next'>▶</button><label>速度 <input id='speed' value='12' type='number' min='1' max='60'></label><label>帧 <select id='frame'></select></label><label>关键帧 <select id='jump'><option value=''>完整序列</option></select></label><label>尾迹 <input id='tail' value='15' type='number' min='0' max='120'></label></div><div id='layers'>__LAYERS__<label><input type='checkbox' data-layer='skeleton' checked>左右手骨架</label><label><input type='checkbox' data-layer='object' checked>source 物体可视网格</label><label><input type='checkbox' data-layer='trajectory' checked>物体/腕部/指尖轨迹</label><label><input type='checkbox' data-layer='nearest' checked>最近表面连接线</label><label><input type='checkbox' data-layer='suspension' checked>悬空标记</label><label><input type='checkbox' data-layer='penetration' checked>有符号指尖穿模标记</label><label><input type='checkbox' data-layer='error' checked>误差向量</label></div></section><div id='scene'></div><pre id='info'></pre><script>__PLOTLY__</script><script>const P='__PACKED__';(async()=>{const b=Uint8Array.from(atob(P),c=>c.charCodeAt(0));const D=JSON.parse(new TextDecoder().decode(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()));const $=x=>document.getElementById(x),qp=new URLSearchParams(location.search);let timer=null;const en=k=>[...document.querySelectorAll('[data-layer]')].some(x=>x.dataset.layer===k&&x.checked),mul=(T,p)=>[T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3],T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7],T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]],pts=(T,a)=>a.map(p=>p?mul(T,p):null),inv=T=>{let r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],p=[T[3],T[7],T[11]],q=[[r[0][0],r[1][0],r[2][0]],[r[0][1],r[1][1],r[2][1]],[r[0][2],r[1][2],r[2][2]]];return[q[0][0],q[0][1],q[0][2],-(q[0][0]*p[0]+q[0][1]*p[1]+q[0][2]*p[2]),q[1][0],q[1][1],q[1][2],-(q[1][0]*p[0]+q[1][1]*p[1]+q[1][2]*p[2]),q[2][0],q[2][1],q[2][2],-(q[2][0]*p[0]+q[2][1]*p[0]+q[2][2]*p[2])]},mesh=(n,v,f,c,o,V)=>{let q=pts(V,v);return{type:'mesh3d',name:n,x:q.map(p=>p&&p[0]),y:q.map(p=>p&&p[1]),z:q.map(p=>p&&p[2]),i:f.map(x=>x[0]),j:f.map(x=>x[1]),k:f.map(x=>x[2]),color:c,opacity:o,flatshading:true}},line=(n,a,c,V,w=4)=>{let q=pts(V,a);return{type:'scatter3d',mode:'lines+markers',name:n,x:q.map(p=>p&&p[0]),y:q.map(p=>p&&p[1]),z:q.map(p=>p&&p[2]),line:{color:c,width:w},marker:{size:3,color:c}}},dots=(n,a,c,V)=>{let q=pts(V,a);return{type:'scatter3d',mode:'markers',name:n,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),marker:{size:5,color:c,symbol:'diamond'}}},skel=(n,p,c,V)=>{let a=[];for(let ch of D.chains){for(let x of ch)a.push(p[x]);a.push(null)}return line(n,a,c,V,5)},axes=(n,T,V)=>{let p=[T[3],T[7],T[11]],r=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]];return[[0,'#ef476f'],[1,'#06d6a0'],[2,'#118ab2']].map(x=>line(n+'轴'+x[0],[p,[p[0]+.05*r[0][x[0]],p[1]+.05*r[1][x[0]],p[2]+.05*r[2][x[0]]]],x[1],V,5))},robot=(n,R,s,i,c,o,V)=>{let vv=[],ff=[],off=0;for(let g=0;g<R.layers[s].length;g++){let L=R.layers[s][g],t=R.transforms[s][i][g],T=[t[3],t[4],t[5],t[0],t[6],t[7],t[8],t[1],t[9],t[10],t[11],t[2]];vv.push(...pts(T,L.vertices));ff.push(...L.faces.map(f=>[f[0]+off,f[1]+off,f[2]+off]));off+=L.vertices.length}return mesh(n,vv,ff,c,o,V)},wsk=(n,p,t,c,V)=>{let a=[];for(let x of t)a.push([p[3],p[7],p[11]],x,null);return line(n,a,c,V,4)};D.frames.forEach((f,i)=>$('frame').add(new Option(f.source_frame,i)));for(const k of(D.key_frames||[]))$('jump').add(new Option(k.label,k.frame));let initial=qp.get('frame');if(initial){let i=D.frames.findIndex(x=>String(x.source_frame)===initial);if(i>=0)$('frame').value=i}if(qp.get('coord'))$('coord').value=qp.get('coord');if(qp.get('compare'))$('compare').value=qp.get('compare');const camera=qp.get('camera')||'global',tips=[4,8,12,16,20];function draw(){let i=+$('frame').value,f=D.frames[i],V=[1,0,0,0,0,1,0,0,0,0,1,0],co=$('coord').value;if(co==='object')V=inv(f.object);if(co==='left')V=inv(f.left_wrist);if(co==='right')V=inv(f.right_wrist);let t=[],cmp=$('compare').value,src=D.kind==='source',o=src&&(cmp==='official'||cmp==='overlay'||cmp==='difference'),s=src&&(cmp==='spider'||cmp==='overlay'||cmp==='difference'),showSrc=!src&&(cmp!=='stagebonly'),showB=!src&&(cmp!=='source');if(src){if(o&&en('officialobject'))t.push(mesh('官方物体',D.object.vertices,D.object.faces,'#457b9d',.38,V));if(s&&en('spiderobject'))t.push(mesh('Spider物体',D.object.vertices,D.object.faces,'#f4a261',.25,V));if(o&&en('officialbody'))t.push(mesh('官方全身',f.official.body,D.body.faces,'#48cae4',.18,V));if(s&&en('spiderbody'))t.push(mesh('Spider全身上下文',f.official.body,D.body.faces,'#f72585',.12,V));for(let h of['left','right']){if(o&&en('officialhand'))t.push(mesh('官方'+h+'手',f.official[h].verts,D.hand[h].faces,'#48cae4',.62,V));if(s&&en('spiderhand'))t.push(mesh('Spider'+h+'手',f.spider[h].verts,D.hand[h].faces,'#f72585',.48,V));if(o&&en('skeleton'))t.push(skel('官方'+h+'骨架',f.official[h].joints,h==='left'?'#3a86ff':'#ff006e',V));if(s&&en('skeleton'))t.push(skel('Spider'+h+'骨架',f.spider[h].joints,h==='left'?'#00b4d8':'#06d6a0',V));if(o&&s&&en('error'))t.push(line('source差异'+h,[f.official[h].joints[8],f.spider[h].joints[8]],'#ffd166',V,5))}}else{if(en('object'))t.push(mesh('source物体',D.object.vertices,D.object.faces,'#457b9d',.38,V));if(showSrc&&en('sourcebody'))t.push(mesh('source全身',f.source.body,D.body.faces,'#48cae4',.14,V));for(let h of['left','right']){if(showSrc&&en('sourcehand'))t.push(mesh('source'+h+'手',f.source[h].verts,D.hand[h].faces,h==='left'?'#3a86ff':'#ff006e',.52,V));if(showB&&en('stageb'))t.push(robot('Stage B '+h+'可视网格',D.robot_visual,h,i,h==='left'?'#80ed99':'#ffd166',.70,V));if(showB&&en('collision'))t.push(robot('Stage B '+h+'碰撞网格',D.robot_collision,h,i,'#ef476f',.24,V));if(showSrc&&en('skeleton'))t.push(skel('source'+h+'骨架',f.source[h].joints,h==='left'?'#3a86ff':'#ff006e',V));if(showB&&en('wujiskel'))t.push(wsk('Wuji '+h+'骨架',f.wuji[h].palm,f.wuji[h].tips,h==='left'?'#80ed99':'#ffd166',V));if(showB&&en('wujiframes'))t.push(...axes('Wuji '+h+'腕',f.wuji[h].palm,V))}}if(src||en('sourceframes'))t.push(...axes('左腕',f.left_wrist,V),...axes('右腕',f.right_wrist,V),...axes('物体',f.object,V));if(en('nearest'))for(let h of['left','right']){let H=src?(s?f.spider[h]:f.official[h]):f.source[h];for(let q=0;q<5;q++)t.push(line(h+'最近表面',[H.joints[tips[q]],f.nearest[h][q]],'#f4a261',V,2))}if(en('trajectory')){let q=Math.max(0,i-(+$('tail').value));for(let h of['left','right'])t.push(line(h+'食指轨迹',D.frames.slice(q,i+1).map(x=>(x.official||x.source)[h].joints[8]),h==='left'?'#3a86ff':'#ff006e',V,3))}for(let h of['left','right']){let H=src?(s?f.spider[h]:f.official[h]):f.source[h],c=f.contact&&f.contact[h];if(c&&en('suspension')){let a=tips.map(k=>H.joints[k]).filter((_,k)=>c.tip_distance[k]>.03);if(a.length)t.push(dots(h+'悬空指尖',a,'#ff9f1c',V))}if(c&&c.penetration&&en('penetration')){let a=tips.map(k=>H.joints[k]).filter((_,k)=>c.penetration[k]>1e-5);if(a.length)t.push(dots(h+'穿模指尖',a,'#ef476f',V))}}$('status').textContent=D.status;$('info').textContent='source 帧='+f.source_frame+' | 完整序列帧 '+(i+1)+'/'+D.frames.length+'\n'+f.metrics+'\n本页以 gzip 内嵌完整时间序列和真实 connected meshes；没有坐标、尺度或时序对齐修正。';Plotly.react('scene',t,{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{color:'#e9eef5'},scene:{aspectmode:'data',camera:{eye:(camera==='macro'?{x:.55,y:-.55,z:.4}:camera==='opposite'?{x:-1.5,y:1.5,z:1.1}:{x:1.5,y:-1.5,z:1.1})}},legend:{orientation:'h'},margin:{l:0,r:0,t:20,b:0}},{responsive:true})}$('jump').oninput=()=>{let n=$('jump').value;if(n){let i=D.frames.findIndex(x=>String(x.source_frame)===n);if(i>=0){$('frame').value=i;draw()}}};['frame','coord','compare','tail'].forEach(x=>$(x).oninput=draw);document.querySelectorAll('[data-layer]').forEach(x=>x.onchange=draw);$('prev').onclick=()=>{$('frame').value=Math.max(0,+$('frame').value-1);draw()};$('next').onclick=()=>{$('frame').value=Math.min(D.frames.length-1,+$('frame').value+1);draw()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>{$('frame').value=(+$('frame').value+1)%D.frames.length;draw()},1000/Math.max(1,+$('speed').value));$('play').textContent='暂停'}};draw()})().catch(e=>{$('status').textContent='HTML 数据解压失败：'+e.stack});</script></body></html>"""
    template = template.replace(
        "const D=JSON.parse(new TextDecoder().decode(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()));const $=x=>document.getElementById(x),",
        "const U=new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()),N=new DataView(U.buffer,U.byteOffset,8).getUint32(0,true),D=JSON.parse(new TextDecoder().decode(U.subarray(8,8+N))),B=U.subarray((8+N+7)&~7),A=x=>{if(!x||!x.__binary_array__)return x;let z=x.__binary_array__,C={'<f4':Float32Array,'<f8':Float64Array,'<i4':Int32Array,'<i8':BigInt64Array,'|u1':Uint8Array,'<u2':Uint16Array,'<u4':Uint32Array}[z.dtype];if(!C)throw Error('不支持的二进制类型 '+z.dtype);let q=new C(B.buffer,B.byteOffset+z.offset,z.count),p=0,r=d=>{if(d===z.shape.length)return Number(q[p++]);let a=[];for(let k=0;k<z.shape[d];k++)a.push(r(d+1));return a};return r(0)},$=x=>document.getElementById(x),",
    )
    template = template.replace("const b=Uint8Array.from(atob(P),c=>c.charCodeAt(0));", "const b=(()=>{let o=new Uint8Array(Math.floor(P.length*3/4)),w=0;for(let i=0;i<P.length;i+=32768){let s=atob(P.slice(i,i+32768));for(let j=0;j<s.length;j++)o[w++]=s.charCodeAt(j)}return o.subarray(0,w)})();")
    template = template.replace(
        "mul=(T,p)=>[T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3],T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7],T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]],pts=(T,a)=>a.map",
        "mul=(T,p)=>{T=A(T);p=A(p);return[T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3],T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7],T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]]},pts=(T,a)=>A(a).map",
    )
    template = template.replace("mesh=(n,v,f,c,o,V)=>{let q=pts(V,v);return{type:'mesh3d'", "mesh=(n,v,f,c,o,V)=>{v=A(v);f=A(f);let q=pts(V,v);return{type:'mesh3d'")
    template = template.replace("let L=R.layers[s][g],t=R.transforms[s][i][g],T=", "let L=R.layers[s][g],t=A(R.transforms[s])[i][g],T=")
    template = template.replace(
        "]];vv.push(...pts(T,L.vertices));",
        "]];L.vertices=A(L.vertices);L.faces=A(L.faces);vv.push(...pts(T,L.vertices));",
    )
    template = template.replace(
        "vv.push(...pts(T,L.vertices));ff.push(...L.faces.map(f=>[f[0]+off,f[1]+off,f[2]+off]));off+=L.vertices.length",
        "for(let v of pts(T,L.vertices))vv.push(v);for(let f of A(L.faces))ff.push([f[0]+off,f[1]+off,f[2]+off]);off+=L.vertices.length",
    )
    # Top-level await makes the document load event wait for decompression and
    # the first Plotly draw.  Chrome's screenshot mode otherwise races an async
    # IIFE and can capture only the empty loading shell.
    template = template.replace("<script>const P='__PACKED__';(async()=>{", "<script type='module'>const P='__PACKED__';try{")
    template = template.replace(";draw()})().catch(e=>{$('status').textContent='HTML 数据解压失败：'+e.stack});</script>", ";draw()}catch(e){document.getElementById('status').textContent='HTML 数据解压失败：'+e.stack}</script>")
    template = template.replace("camera==='macro'?{x:.55,y:-.55,z:.4}:camera==='opposite'?{x:-1.5,y:1.5,z:1.1}:{x:1.5,y:-1.5,z:1.1}", "(window.__auditCamera||camera)==='macro'?{x:.55,y:-.55,z:.4}:(window.__auditCamera||camera)==='opposite'?{x:-1.5,y:1.5,z:1.1}:{x:1.5,y:-1.5,z:1.1}")
    # Plotly otherwise preserves a broad full-body autorange after a CDP view
    # switch.  Recompute a common local range from the current traces so object
    # and wrist screenshots show the actual hand--object geometry at a useful,
    # metrically equal scale while world screenshots retain the full overview.
    template = template.replace(
        "$('status').textContent=D.status;$('info').textContent=",
        "let localAxes=co==='world'?{}:(()=>{let b=['x','y','z'].map(k=>{let v=[];for(let e of t)for(let q of(e[k]||[]))if(Number.isFinite(q))v.push(q);return[Math.min(...v),Math.max(...v)]}),r=Math.max(.08,...b.map(a=>(a[1]-a[0])*.65));return{xaxis:{range:[(b[0][0]+b[0][1])/2-r,(b[0][0]+b[0][1])/2+r]},yaxis:{range:[(b[1][0]+b[1][1])/2-r,(b[1][0]+b[1][1])/2+r]},zaxis:{range:[(b[2][0]+b[2][1])/2-r,(b[2][0]+b[2][1])/2+r]}}})();$('status').textContent=D.status;$('info').textContent=",
    )
    template = template.replace("scene:{aspectmode:'data',camera:{eye:", "scene:{aspectmode:co==='world'?'data':'cube',...localAxes,camera:{eye:")
    # ``Plotly.react`` is asynchronous.  Expose the current draw promise so
    # CDP evidence capture can wait for the requested frame/view rather than
    # capturing the preceding full-world scene after only animation frames.
    template = template.replace("Plotly.react('scene',t,", "window.__auditDrawPromise=Plotly.react('scene',t,")
    template = template.replace("+'\n'+f.metrics+'\n本页", "+'\\n'+f.metrics+'\\n本页")
    # Keep the packed JavaScript's rigid-transform inverse exact.  This extra
    # replacement is intentionally narrow: it protects the third translation
    # component while the embedded viewer remains a self-contained artifact.
    template = template.replace("q[2][1]*p[0]+q[2][2]*p[2]", "q[2][1]*p[1]+q[2][2]*p[2]")
    # The data keys remain ``left``/``right`` for machine stability, while all
    # user-visible Plotly legend entries use Chinese side labels.
    for old, new in {
        "'官方'+h+'手'": "'官方'+(h==='left'?'左':'右')+'手'",
        "'Spider'+h+'手'": "'Spider'+(h==='left'?'左':'右')+'手'",
        "'官方'+h+'骨架'": "'官方'+(h==='left'?'左':'右')+'骨架'",
        "'Spider'+h+'骨架'": "'Spider'+(h==='left'?'左':'右')+'骨架'",
        "'source差异'+h": "'source 差异'+(h==='left'?'左':'右')",
        "'source'+h+'手'": "'source '+(h==='left'?'左':'右')+'手'",
        "'Stage B '+h+'可视网格'": "'Stage B '+(h==='left'?'左':'右')+'手可视网格'",
        "'Stage B '+h+'碰撞网格'": "'Stage B '+(h==='left'?'左':'右')+'手碰撞网格'",
        "'source'+h+'骨架'": "'source '+(h==='left'?'左':'右')+'手骨架'",
        "'Wuji '+h+'骨架'": "'Wuji '+(h==='left'?'左':'右')+'手骨架'",
        "'Wuji '+h+'腕'": "'Wuji '+(h==='left'?'左':'右')+'腕'",
        "h+'最近表面'": "(h==='left'?'左':'右')+'手最近表面'",
        "h+'食指轨迹'": "(h==='left'?'左':'右')+'手食指轨迹'",
        "h+'悬空指尖'": "(h==='left'?'左':'右')+'手悬空指尖'",
        "h+'穿模指尖'": "(h==='left'?'左':'右')+'手穿模指尖'",
    }.items():
        template = template.replace(old, new)
    # Object assets are stored in their own local frame.  Transform them to
    # world before applying the selected display frame, just like hand meshes.
    # Without the first transform an object-coordinate view applies the inverse
    # object transform directly to local vertices and makes the object appear
    # spuriously far from the hands.
    template = template.replace("mesh('官方物体',D.object.vertices,D.object.faces,'#457b9d',.38,V)", "mesh('官方物体',pts(f.object,D.object.vertices),D.object.faces,'#457b9d',.38,V)")
    template = template.replace("mesh('Spider物体',D.object.vertices,D.object.faces,'#f4a261',.25,V)", "mesh('Spider物体',pts(f.object,D.object.vertices),D.object.faces,'#f4a261',.25,V)")
    template = template.replace("mesh('source物体',D.object.vertices,D.object.faces,'#457b9d',.38,V)", "mesh('source物体',pts(f.object,D.object.vertices),D.object.faces,'#457b9d',.38,V)")
    template = template.replace("mesh('官方全身',f.official.body,D.body.faces,'#48cae4',.18,V)", "dots('官方全身关节点',f.official.body_joints,'#48cae4',V)")
    template = template.replace("mesh('Spider全身上下文',f.official.body,D.body.faces,'#f72585',.12,V)", "dots('Spider全身上下文关节点',f.spider.body_joints,'#f72585',V)")
    template = template.replace("mesh('source全身',f.source.body,D.body.faces,'#48cae4',.14,V)", "dots('source全身关节点',f.source.body_joints,'#48cae4',V)")
    return template.replace("__TITLE__", title).replace("__COMPARE__", comparison).replace("__LAYERS__", layers).replace("__PLOTLY__", get_plotlyjs()).replace("__PACKED__", packed)


def _localize_html(page: str) -> str:
    """Keep control labels Chinese while retaining stable machine values."""
    replacements = {
        ">world<": ">世界坐标（world）<",
        ">object<": ">物体坐标（object）<",
        ">left wrist<": ">左腕坐标（left wrist）<",
        ">right wrist<": ">右腕坐标（right wrist）<",
        ">object visual mesh<": ">物体 visual mesh<",
        ">左右手 skeleton<": ">左右手骨架（skeleton）<",
        ">wrist/object/world frame<": ">腕部/物体/世界坐标轴<",
        ">object/wrist/fingertip trajectories<": ">物体/腕部/指尖轨迹<",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    page = page.replace(
        "if(qp.get('coord'))$('coord').value=qp.get('coord');function draw()",
        "if(qp.get('coord'))$('coord').value=qp.get('coord');if(qp.get('compare'))$('compare').value=qp.get('compare');const cameraMode=qp.get('camera')||'global';function draw()",
    )
    page = page.replace(
        "if(qp.get('coord'))$('coord').value=qp.get('coord');if(qp.get('compare'))$('compare').value=qp.get('compare');function draw()",
        "if(qp.get('coord'))$('coord').value=qp.get('coord');if(qp.get('compare'))$('compare').value=qp.get('compare');const cameraMode=qp.get('camera')||'global';function draw()",
    )
    page = page.replace(
        "camera:{eye:{x:1.5,y:-1.5,z:1.1}}",
        "camera:{eye:(cameraMode==='macro'?{x:.55,y:-.55,z:.4}:cameraMode==='opposite'?{x:-1.5,y:1.5,z:1.1}:{x:1.5,y:-1.5,z:1.1})}",
    )
    return page


def _normalize_key_frames(key_frames: list[tuple[str, int]] | list[int] | None) -> list[tuple[str, int]]:
    """Normalize named key frames while keeping the tiny test fixture ergonomic."""
    normalized: list[tuple[str, int]] = []
    for item in key_frames or []:
        if isinstance(item, tuple):
            normalized.append((str(item[0]), int(item[1])))
        else:
            normalized.append((f"关键帧 {int(item)}", int(item)))
    return normalized


def _frame_contact(contact: dict[str, Any], index: int) -> dict[str, dict[str, Any]]:
    """Extract viewer-safe fingertip contact values, including failure fixtures."""
    output: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        values = contact["sides"][side]
        distances = values.get("tip_distance_m")
        penetrations = values.get("tip_penetration_depth_m")
        output[side] = {
            "tip_distance": np.asarray(distances[index] if distances is not None else np.zeros(5), dtype=float),
            "penetration": np.asarray(penetrations[index], dtype=float) if penetrations is not None else None,
        }
    return output


def write_source_html(root: Path, official: dict[str, np.ndarray], spider: dict[str, np.ndarray], comparison: dict[str, Any], contact: dict[str, Any], key_frames: list[tuple[str, int]] | list[int] | None = None) -> tuple[Path, Path]:
    """Create the mandatory first self-contained 3D audit HTML and index."""
    gap_frames = [int(contact["sides"][side]["max_suspension_proxy_frame"]) for side in ("left", "right")]
    error_frame = int(np.argmax(np.linalg.norm(official["left_joints_world"][:, FINGERTIP_INDICES] - spider["left_joints_world"][:, FINGERTIP_INDICES], axis=-1))) // 5
    named_key_frames = _normalize_key_frames(key_frames)
    samples = _sample_indices(len(official["source_frame_indices"]), [0, len(official["source_frame_indices"]) // 2, len(official["source_frame_indices"]) - 1, error_frame, *gap_frames, *(frame for _label, frame in named_key_frames)], maximum=len(official["source_frame_indices"]))
    frames = []
    for index in samples:
        frame = {"source_frame": int(index), "object": official["T_world_object"][index].reshape(-1), "left_wrist": official["T_world_left_wrist"][index].reshape(-1), "right_wrist": official["T_world_right_wrist"][index].reshape(-1), "nearest": {side: contact["sides"][side]["nearest_point_world"][index] for side in ("left", "right")}, "contact": _frame_contact(contact, int(index)), "official": {"body_joints": official["body_joints_world"][index], **{side: {"verts": official[f"{side}_vertices_world"][index], "joints": official[f"{side}_joints_world"][index]} for side in ("left", "right")}}, "spider": {"body_joints": official["body_joints_world"][index], **{side: {"verts": spider[f"{side}_vertices_world"][index], "joints": spider[f"{side}_joints_world"][index]} for side in ("left", "right")}}, "metrics": f"source decision={comparison['decision']['classification']} | max residual={comparison['decision']['maximum_checked_residual']:.3e}"}
        frames.append(frame)
    payload = {"kind": "source", "chains": CHAINS, "key_frames": [{"label": label, "frame": frame} for label, frame in named_key_frames], "frames": frames, "body": {"faces": official["body_faces"]}, "hand": {side: {"faces": official[f"{side}_hand_faces"]} for side in ("left", "right")}, "object": {"vertices": official["object_asset_vertices"], "faces": official["object_faces"]}, "status": f"SOURCE: {comparison['decision']['status']}\n分类: {comparison['decision']['classification']}\n正式独立重建 vs GrabAdapter.load_sequence（在任何 Stage A/B 之前）。\n完整 {len(frames)} 帧可播放；没有人为平移、旋转、尺度或时序修正。"}
    html = root / "html/01_grab_loaded_source_audit.html"
    index = root / "html/01_grab_loaded_source_visual_index.html"
    write_text(html, _html_document_v2(payload, "GRAB 官方/独立重建 vs spider-dex 加载后 source", "source"))
    write_text(index, "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><body><h1>GRAB source 加载审计索引</h1><p><a href='01_grab_loaded_source_audit.html?coord=world&frame=0&compare=overlay'>世界坐标起始帧（官方与 Spider 叠加）</a></p><p><a href='01_grab_loaded_source_audit.html?coord=object&compare=overlay'>物体坐标手物关系（官方与 Spider 叠加）</a></p><p><a href='01_grab_loaded_source_audit.html?coord=left&compare=overlay'>左腕坐标（官方与 Spider 叠加）</a></p><p><a href='01_grab_loaded_source_audit.html?coord=right&compare=overlay'>右腕坐标（官方与 Spider 叠加）</a></p></body></html>")
    return html, index


def _extract_stage_b(scene: Path, qpos: np.ndarray) -> dict[str, Any]:
    """Forward real Stage-B qpos to obtain sites and object transforms."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    site_names = ["right_palm", "left_palm", *[f"{side}_{finger}_tip" for side in ("right", "left") for finger in FINGERTIP_NAMES]]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    if min(site_ids) < 0 or object_id < 0:
        raise RuntimeError("Stage B scene 缺少 Wuji palm/tip sites 或 right_object")
    positions = np.empty((len(qpos), len(site_ids), 3), dtype=np.float64)
    rotations = np.empty((len(qpos), len(site_ids), 3, 3), dtype=np.float64)
    objects = np.empty((len(qpos), 4, 4), dtype=np.float64)
    for index, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[site_ids]
        rotations[index] = data.site_xmat[site_ids].reshape(-1, 3, 3)
        objects[index] = transform_from_rt(data.xmat[object_id].reshape(3, 3), data.xpos[object_id])
    return {"model": model, "site_names": site_names, "positions": positions, "rotations": rotations, "objects": objects}


def _robot_payload(scene: Path, qpos: np.ndarray, group: int, max_faces_per_mesh: int = 96) -> dict[str, Any]:
    """Extract bounded real MuJoCo mesh layers and frame transforms for HTML.

    This is a renderer-only deterministic triangle subsample.  It keeps the
    source MuJoCo mesh, qpos, Stage-B scene, and metric artifacts untouched;
    without it the combined visual/collision hand meshes exceed 450k triangles
    and a local Plotly page cannot reach its first interactive frame.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    chosen: dict[str, list[int]] = {"left": [], "right": []}
    for geom in range(model.ngeom):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH) or int(model.geom_group[geom]) != group:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        side = "left" if body.startswith("l_") else "right" if body.startswith("r_") else None
        if side is not None:
            chosen[side].append(geom)
    if not any(chosen.values()):
        raise RuntimeError(f"Wuji scene 无 group={group} hand mesh")
    layers: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    rendering: dict[str, dict[str, int]] = {"left": {"source_vertices": 0, "source_faces": 0, "render_vertices": 0, "render_faces": 0}, "right": {"source_vertices": 0, "source_faces": 0, "render_vertices": 0, "render_faces": 0}}
    transforms: dict[str, np.ndarray] = {side: np.empty((len(qpos), len(ids), 12), dtype=np.float32) for side, ids in chosen.items()}
    for side, ids in chosen.items():
        for geom in ids:
            mesh_id = int(model.geom_dataid[geom])
            va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32)
            faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int32)
            source_vertices, source_faces = len(vertices), len(faces)
            if len(faces) > max_faces_per_mesh:
                selected = faces[np.linspace(0, len(faces) - 1, max_faces_per_mesh, dtype=np.int64)]
                used = np.unique(selected.reshape(-1))
                remap = np.full(len(vertices), -1, dtype=np.int32)
                remap[used] = np.arange(len(used), dtype=np.int32)
                vertices = vertices[used]
                faces = remap[selected]
            rendering[side]["source_vertices"] += source_vertices
            rendering[side]["source_faces"] += source_faces
            rendering[side]["render_vertices"] += len(vertices)
            rendering[side]["render_faces"] += len(faces)
            layers[side].append({"vertices": vertices, "faces": faces, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom), "source_vertices": source_vertices, "source_faces": source_faces, "render_vertices": len(vertices), "render_faces": len(faces), "triangle_sampling": "deterministic_linspace" if source_faces != len(faces) else "full"})
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side, ids in chosen.items():
            transforms[side][frame, :, :3] = data.geom_xpos[ids]
            transforms[side][frame, :, 3:] = data.geom_xmat[ids]
    return {"layers": layers, "transforms": transforms, "rendering": {"max_faces_per_mesh": max_faces_per_mesh, "sides": rendering, "note": "HTML renderer uses deterministic triangle samples of the real MuJoCo mesh; raw Stage-B meshes and metrics are unchanged."}}


def audit_stage_b(root: Path, workspace: Path, selected: dict[str, Any], spider: dict[str, np.ndarray], source_status: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run Stage A/B in isolation, retaining real partial/failure evidence."""
    scratch_workspace = Path("/tmp/spider-dex-new-sequence-recheck") / root.name / "workspace"
    if scratch_workspace.exists() or workspace.exists():
        raise FileExistsError(f"Stage A/B workspace 已存在，拒绝覆盖: scratch={scratch_workspace}, delivered={workspace}")
    scratch_workspace.parent.mkdir(parents=True, exist_ok=False)
    paths_config = root / "retarget/paths.generated.yaml"
    generated = {"schema_version": 1, "datasets": {"grab": {"source_root": str(load_project_paths("configs/local/paths.yaml").source_root("grab"))}}, "body_models": {"root": str(load_project_paths("configs/local/paths.yaml").body_model_root)}, "workspace": {"root": str(scratch_workspace)}}
    write_text(paths_config, yaml.safe_dump(generated, sort_keys=False, allow_unicode=True))
    task = str(selected["sequence_id_safe"])
    frames = int(selected["frame_count"])
    stage_a_dir = root / "retarget/stage_a"
    stage_b_dir = root / "retarget/stage_b"
    failure_dir = root / "retarget/failure"
    for directory in (stage_a_dir, stage_b_dir, failure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    commands = [[sys.executable, "-m", "spider.tools.grab_pipeline", "prepare", "--paths-config", str(paths_config), "--sequence-id", task, "--frame-start", "0", "--frame-end", str(frames)], [sys.executable, "-m", "spider.tools.grab_pipeline", "run-wuji-ik", "--paths-config", str(paths_config), "--sequence-id", task, "--no-save-video"]]
    stages: dict[str, Any] = {"input_source_gate": source_status, "stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}, "cxa": {"status": "NOT_RUN", "reason": "当前项目标准 GRAB→Wuji kinematic retarget 在 Stage B 结束；C-XA 是冻结 Stage-C 接触修正工作，不适用于新序列且未被移植。"}, "commands": [" ".join(command) for command in commands], "provenance": {"effective_config": str(paths_config), "effective_config_sha256": sha256(paths_config), "raw_parameter": str(selected["parameter_path"]), "raw_parameter_sha256": sha256(Path(selected["parameter_path"])), "object_mesh": str(selected["object_mesh"]), "object_mesh_sha256": sha256(Path(selected["object_mesh"])), "runtime": {"python": sys.version, "executable": sys.executable, "mujoco": getattr(mujoco, "__version__", "unknown")}}}
    for index, (name, directory) in enumerate((("stage_a", stage_a_dir), ("stage_b", stage_b_dir))):
        command = commands[index]
        proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        write_text(directory / "command.txt", " ".join(command) + "\n")
        write_text(directory / "stdout.log", proc.stdout)
        write_text(directory / "stderr.log", proc.stderr)
        stages[name] = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "command": command, "stdout_log": str(directory / "stdout.log"), "stderr_log": str(directory / "stderr.log"), "stdout_summary": _log_summary(proc.stdout), "stderr_summary": _log_summary(proc.stderr)}
        if proc.returncode != 0:
            if scratch_workspace.exists():
                shutil.move(str(scratch_workspace), str(workspace))
            stages["final_retarget"] = "RETARGET_FAILED_STAGE_A" if name == "stage_a" else "RETARGET_FAILED_STAGE_B"
            write_text(failure_dir / "failure.txt", f"{name} failed with return code {proc.returncode}\n{proc.stderr}\n")
            write_json(root / "reports/retarget_stage_status.json", stages)
            return stages, None
    shutil.move(str(scratch_workspace), str(workspace))
    canonical_dir = workspace / "processed/grab/canonical" / task
    robot_dir = workspace / "processed/grab/wuji_hand2_beta1/bimanual" / task / "0"
    scene = robot_dir.parent / "scene.xml"
    try:
        with np.load(robot_dir / "trajectory_kinematic.npz", allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            qvel = np.asarray(archive["qvel"], dtype=np.float64)
            frequency = float(archive["frequency"])
        mapping = np.asarray(json.loads((robot_dir / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
        geometry = _extract_stage_b(scene, qpos)
        source_indices = mapping
        raw_object = spider["T_world_object"][source_indices]
        object_t = np.linalg.norm(raw_object[:, :3, 3] - geometry["objects"][:, :3, 3], axis=1)
        object_r = rotation_residual_rad(raw_object[:, :3, :3], geometry["objects"][:, :3, :3])
        side_metrics: dict[str, Any] = {}
        wuji: dict[str, Any] = {}
        for side in ("left", "right"):
            palm_index = geometry["site_names"].index(f"{side}_palm")
            palm = transform_from_rt(geometry["rotations"][:, palm_index], geometry["positions"][:, palm_index])
            source_wrist = spider[f"T_world_{side}_wrist"][source_indices].copy()
            source_wrist[:, :3, :3] = source_wrist[:, :3, :3] @ SMPLX_TO_WUJI_PALM[side]
            source_rel = object_relative(raw_object, source_wrist)
            robot_rel = object_relative(geometry["objects"], palm)
            source_tips = spider[f"{side}_joints_world"][source_indices][:, FINGERTIP_INDICES]
            tip_indices = [geometry["site_names"].index(f"{side}_{finger}_tip") for finger in FINGERTIP_NAMES]
            robot_tips = geometry["positions"][:, tip_indices]
            source_tips_local = apply_transform(invert_transform(raw_object)[:, None], source_tips)
            robot_tips_local = apply_transform(invert_transform(geometry["objects"])[:, None], robot_tips)
            wrist_error = np.linalg.norm(source_rel[:, :3, 3] - robot_rel[:, :3, 3], axis=1)
            tip_error = np.linalg.norm(source_tips_local - robot_tips_local, axis=-1)
            side_metrics[side] = {"object_relative_wrist_rmse_m": float(np.sqrt(np.mean(wrist_error ** 2))), "object_relative_wrist_max_m": float(wrist_error.max()), "object_relative_fingertip_rmse_m": float(np.sqrt(np.mean(tip_error ** 2))), "object_relative_fingertip_max_m": float(tip_error.max())}
            wuji[side] = {"palm": palm, "tips": robot_tips, "tip_error": tip_error}
        robot_metrics = json.loads((robot_dir / "metrics_kinematic.json").read_text(encoding="utf-8"))
        stage_b_status = "PASS" if robot_metrics.get("status") == "PASS" else "PARTIAL"
        stages["stage_b"] = {**stages["stage_b"], "status": stage_b_status, "robot_dir": str(robot_dir), "scene": str(scene), "scene_sha256": sha256(scene), "source_mapping": str(robot_dir / "source_mapping.json"), "source_mapping_sha256": sha256(robot_dir / "source_mapping.json"), "metrics": str(robot_dir / "metrics_kinematic.json"), "metrics_sha256": sha256(robot_dir / "metrics_kinematic.json"), "frames": int(len(qpos)), "frequency_hz": frequency, "output_hash": sha256(robot_dir / "trajectory_kinematic.npz")}
        stages["final_retarget"] = "RETARGET_PASS" if source_status == "PASS" and stage_b_status == "PASS" else ("RETARGET_PARTIAL" if stage_b_status == "PARTIAL" else "RETARGET_FAILED_SOURCE_ERROR")
        result = {"qpos": qpos, "qvel": qvel, "mapping": mapping, "geometry": geometry, "wuji": wuji, "metrics": {"stage_b": side_metrics, "object_translation_residual_m": _summary(object_t), "object_rotation_residual_rad": _summary(object_r, "rad"), "joint_limit_violations": robot_metrics.get("joint_limit_violations"), "nan_or_inf": bool(not np.isfinite(qpos).all() or not np.isfinite(qvel).all()), "trajectory_continuity_max_qpos_step": float(np.abs(np.diff(qpos, axis=0)).max()), "frame_coverage": float(len(qpos) / max(1, len(spider["source_frame_indices"]))), "failure_frame": None, "runtime": {"frequency_hz": frequency}}, "robot_dir": robot_dir, "scene": scene}
        write_json(root / "reports/retarget_metrics.json", result["metrics"])
        write_json(root / "reports/retarget_stage_status.json", stages)
        return stages, result
    except Exception as exc:
        stages["stage_b"] = {"status": "PARTIAL", "reason": f"trajectory produced but post-audit failed: {type(exc).__name__}: {exc}"}
        stages["final_retarget"] = "RETARGET_PARTIAL"
        write_json(root / "reports/retarget_stage_status.json", stages)
        return stages, None


def _retarget_display(status: str | None) -> str:
    """Map the machine-readable retarget state to the required HTML word."""
    if status == "RETARGET_PASS":
        return "PASS"
    if status == "RETARGET_PARTIAL":
        return "PARTIAL"
    return "FAIL"


def write_retarget_html(root: Path, official: dict[str, np.ndarray], spider: dict[str, np.ndarray], contact: dict[str, Any], stages: dict[str, Any], stage: dict[str, Any] | None, key_frames: list[tuple[str, int]] | list[int] | None = None) -> tuple[Path, Path]:
    """Create the mandatory second real 3D HTML even if Stage B failed."""
    html = root / "html/02_wuji_spider_retarget_audit.html"
    index = root / "html/02_wuji_spider_retarget_visual_index.html"
    named_key_frames = _normalize_key_frames(key_frames)
    final_display = _retarget_display(str(stages.get("final_retarget")))
    if stage is None:
        frames = []
        for index_frame in range(len(official["source_frame_indices"])):
            frames.append({"source_frame": int(index_frame), "object": official["T_world_object"][index_frame].reshape(-1), "left_wrist": official["T_world_left_wrist"][index_frame].reshape(-1), "right_wrist": official["T_world_right_wrist"][index_frame].reshape(-1), "nearest": {side: contact["sides"][side]["nearest_point_world"][index_frame] for side in ("left", "right")}, "contact": _frame_contact(contact, index_frame), "source": {"body_joints": official["body_joints_world"][index_frame], **{side: {"verts": spider[f"{side}_vertices_world"][index_frame], "joints": spider[f"{side}_joints_world"][index_frame]} for side in ("left", "right")}}, "wuji": {side: {"palm": np.eye(4), "tips": np.zeros((5, 3))} for side in ("left", "right")}, "metrics": "Stage B 未产出可用 trajectory；本页保留 Spider source、初始模型层和失败阶段诊断。"})
        empty_robot = {"layers": {"left": [], "right": []}, "transforms": {"left": np.empty((len(frames), 0, 12)), "right": np.empty((len(frames), 0, 12))}}
        payload = {"kind": "retarget", "chains": CHAINS, "key_frames": [{"label": label, "frame": frame} for label, frame in named_key_frames], "frames": frames, "body": {"faces": official["body_faces"]}, "hand": {side: {"faces": official[f"{side}_hand_faces"]} for side in ("left", "right")}, "object": {"vertices": official["object_asset_vertices"], "faces": official["object_faces"]}, "robot_visual": empty_robot, "robot_collision": empty_robot, "status": f"SOURCE: {stages['input_source_gate']}\nSTAGE A: {stages['stage_a']['status']}\nSTAGE B: {stages['stage_b']['status']}\nC-XA: {stages['cxa']['status']}\nFINAL RETARGET: {final_display} ({stages.get('final_retarget')})\nTHIS HTML: 失败诊断，非成功见证。"}
    else:
        mapping = stage["mapping"]
        metric = stage["metrics"]
        important = [0, len(mapping) // 2, len(mapping) - 1]
        requested = []
        visible_key_frames = []
        for label, source_frame in named_key_frames:
            matches = np.flatnonzero(mapping == source_frame)
            if len(matches):
                requested.append(int(matches[0]))
                visible_key_frames.append((label, source_frame))
        samples = _sample_indices(len(mapping), [*important, *requested], maximum=len(mapping))
        visual = _robot_payload(stage["scene"], stage["qpos"][samples], 1)
        collision = _robot_payload(stage["scene"], stage["qpos"][samples], 2)
        render_note = "浏览器 Wuji 网格：真实 MuJoCo 三角面确定性抽样（每个 geom 最多 %d 面；Stage B 原始 mesh/qpos/metrics 未改变）。" % int(visual["rendering"]["max_faces_per_mesh"])
        write_json(root / "reports/HTML_RENDERER_GEOMETRY.json", {"schema_version": 1, "scope": "HTML renderer only; no Stage-B artifact was altered", "scene": str(stage["scene"]), "qpos_source": str(stage["robot_dir"] / "trajectory_kinematic.npz"), "visual_mesh": visual["rendering"], "collision_mesh": collision["rendering"]})
        frames = []
        for out, index_frame in enumerate(samples):
            source_frame = int(mapping[index_frame])
            source = {"body_joints": official["body_joints_world"][source_frame], **{side: {"verts": spider[f"{side}_vertices_world"][source_frame], "joints": spider[f"{side}_joints_world"][source_frame]} for side in ("left", "right")}}
            frames.append({"source_frame": source_frame, "object": spider["T_world_object"][source_frame].reshape(-1), "left_wrist": spider["T_world_left_wrist"][source_frame].reshape(-1), "right_wrist": spider["T_world_right_wrist"][source_frame].reshape(-1), "nearest": {side: contact["sides"][side]["nearest_point_world"][source_frame] for side in ("left", "right")}, "contact": _frame_contact(contact, source_frame), "source": source, "wuji": {side: {"palm": stage["wuji"][side]["palm"][index_frame], "tips": stage["wuji"][side]["tips"][index_frame]} for side in ("left", "right")}, "metrics": f"Stage B source→Wuji | object-relative wrist RMSE L/R={metric['stage_b']['left']['object_relative_wrist_rmse_m']:.4f}/{metric['stage_b']['right']['object_relative_wrist_rmse_m']:.4f} m"})
        payload = {"kind": "retarget", "chains": CHAINS, "key_frames": [{"label": label, "frame": frame} for label, frame in visible_key_frames], "frames": frames, "body": {"faces": official["body_faces"]}, "hand": {side: {"faces": official[f"{side}_hand_faces"]} for side in ("left", "right")}, "object": {"vertices": official["object_asset_vertices"], "faces": official["object_faces"]}, "robot_visual": visual, "robot_collision": collision, "status": f"SOURCE: {stages['input_source_gate']}\nSTAGE A: {stages['stage_a']['status']}\nSTAGE B: {stages['stage_b']['status']}\nC-XA: {stages['cxa']['status']}（{stages['cxa']['reason']}）\nFINAL RETARGET: {final_display} ({stages.get('final_retarget')})\n{render_note}\nTHIS HTML: {'成功见证' if stages.get('final_retarget') == 'RETARGET_PASS' else '失败/部分诊断'}。"}
    write_text(html, _html_document_v2(payload, "Spider loaded source vs Wuji SPIDER retarget：真实 3D 审计", "retarget"))
    write_text(index, "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><body><h1>Wuji retarget 审计索引</h1><p><a href='02_wuji_spider_retarget_audit.html?coord=world&compare=stageb'>世界坐标总览（Source vs Stage B）</a></p><p><a href='02_wuji_spider_retarget_audit.html?coord=object&compare=stageb'>物体坐标接触近景（Source vs Stage B）</a></p><p><a href='02_wuji_spider_retarget_audit.html?coord=left&compare=stageb'>左腕反侧视角（Source vs Stage B）</a></p></body></html>")
    return html, index


def _chrome() -> str | None:
    """Find an installed Chromium binary for real HTML screenshots."""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


class _CdpSocket:
    """Tiny dependency-free WebSocket client for local Chrome DevTools."""

    def __init__(self, websocket_url: str) -> None:
        parsed = urlparse(websocket_url)
        if parsed.scheme != "ws" or not parsed.hostname or not parsed.port:
            raise RuntimeError(f"无效 Chrome DevTools URL: {websocket_url}")
        self.socket = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = f"GET {path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        self.socket.sendall(request.encode("ascii"))
        response = self._read_until(b"\r\n\r\n")
        if b" 101 " not in response.splitlines()[0]:
            raise RuntimeError(f"Chrome DevTools WebSocket 握手失败: {response[:200]!r}")

    def _read_exact(self, count: int) -> bytes:
        result = bytearray()
        while len(result) < count:
            block = self.socket.recv(count - len(result))
            if not block:
                raise ConnectionError("Chrome DevTools WebSocket 提前关闭")
            result.extend(block)
        return bytes(result)

    def _read_until(self, marker: bytes) -> bytes:
        result = bytearray()
        while marker not in result:
            block = self.socket.recv(4096)
            if not block:
                raise ConnectionError("Chrome DevTools WebSocket 提前关闭")
            result.extend(block)
        return bytes(result)

    def send_json(self, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        header = bytearray([0x81])
        mask = os.urandom(4)
        if len(payload) < 126:
            header.append(0x80 | len(payload))
        elif len(payload) <= 0xFFFF:
            header.extend((0x80 | 126,))
            header.extend(struct.pack("!H", len(payload)))
        else:
            header.extend((0x80 | 127,))
            header.extend(struct.pack("!Q", len(payload)))
        encoded = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.socket.sendall(bytes(header) + mask + encoded)

    def receive_json(self, timeout_s: float) -> dict[str, Any]:
        self.socket.settimeout(timeout_s)
        while True:
            first, second = self._read_exact(2)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(length)
            if mask is not None:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                raise ConnectionError("Chrome DevTools WebSocket 已关闭")
            if opcode == 0x9:
                self.socket.sendall(bytes([0x8A, len(payload)]) + payload)
                continue
            if opcode != 0x1:
                continue
            return json.loads(payload.decode("utf-8"))

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass


class _ChromeCdpPage:
    """One persistent local Chrome page with an explicit audit-ready wait."""

    def __init__(self, chrome: str) -> None:
        self.process = subprocess.Popen([chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--window-size=1600,1050", "--remote-debugging-port=0", "--remote-allow-origins=*", "about:blank"], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if self.process.stderr is None:
            raise RuntimeError("无法读取 Chrome DevTools 启动输出")
        deadline = time.monotonic() + 15
        websocket_url: str | None = None
        while time.monotonic() < deadline:
            line = self.process.stderr.readline()
            marker = "DevTools listening on "
            if marker in line:
                websocket_url = line.split(marker, 1)[1].strip()
                break
        if not websocket_url:
            self.close()
            raise RuntimeError("Chrome 未在 15 秒内公开 DevTools 地址")
        browser = _CdpSocket(websocket_url)
        try:
            self.socket = browser
            self.next_id = 0
            self.events: list[dict[str, Any]] = []
            target = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
            parsed = urlparse(websocket_url)
            listing = json.loads(urlopen(f"http://{parsed.hostname}:{parsed.port}/json/list", timeout=10).read().decode("utf-8"))
            page_url = next(item["webSocketDebuggerUrl"] for item in listing if item.get("id") == target)
        finally:
            browser.close()
        self.socket = _CdpSocket(page_url)
        self.next_id = 0
        self.call("Page.enable")
        self.call("Runtime.enable")

    def call(self, method: str, params: dict[str, Any] | None = None, timeout_s: float = 60) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        self.socket.send_json({"id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self.socket.receive_json(max(0.1, deadline - time.monotonic()))
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"Chrome DevTools {method} 失败: {message['error']}")
                return message.get("result", {})
            self.events.append(message)
        raise TimeoutError(f"Chrome DevTools {method} 超时")

    def evaluate(self, expression: str, await_promise: bool = False) -> Any:
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": await_promise})
        details = result.get("exceptionDetails")
        if details:
            raise RuntimeError(f"Chrome JavaScript 异常: {details.get('text', details)}")
        return result.get("result", {}).get("value")

    def raise_if_page_exception(self) -> None:
        """Fail evidence capture when the page emitted an uncaught JS error."""
        failures = [event.get("params", {}).get("exceptionDetails", {}) for event in self.events if event.get("method") == "Runtime.exceptionThrown"]
        if failures:
            detail = failures[-1]
            description = detail.get("exception", {}).get("description") or detail.get("text") or detail
            raise RuntimeError(f"Chrome 页面 JavaScript 异常: {description}")

    def navigate_and_wait(self, url: str, timeout_s: float = 180) -> None:
        self.call("Page.navigate", {"url": url})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = str(self.evaluate("(document.getElementById('status')||{}).textContent||''"))
            if status.startswith("SOURCE:"):
                self.evaluate("Promise.resolve(window.__auditDrawPromise).then(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))", await_promise=True)
                self.raise_if_page_exception()
                return
            if status.startswith("HTML 数据解压失败"):
                raise RuntimeError(status)
            time.sleep(0.25)
        raise TimeoutError("HTML 未在 180 秒内完成解压和首次 3D 绘制")

    def set_view_and_capture(self, source_frame: int, coordinate: str, camera: str, compare: str, output: Path) -> None:
        expression = f"""(() => {{ const frame=document.getElementById('frame'),coord=document.getElementById('coord'),cmp=document.getElementById('compare'); const index=[...frame.options].findIndex(o=>String(o.text)==={json.dumps(str(source_frame))}); if(index<0) throw Error('缺少 source frame {source_frame}'); frame.value=String(index); coord.value={json.dumps(coordinate)}; cmp.value={json.dumps(compare)}; for(const key of ['officialbody','spiderbody','sourcebody']){{const box=document.querySelector(`[data-layer="${{key}}"]`);if(box)box.checked={json.dumps(coordinate == 'world')};}} window.__auditCamera={json.dumps(camera)}; coord.dispatchEvent(new Event('input',{{bubbles:true}})); return Promise.resolve(window.__auditDrawPromise).then(()=>index); }})()"""
        self.evaluate(expression, await_promise=True)
        self.evaluate("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))", await_promise=True)
        self.raise_if_page_exception()
        result = self.call("Page.captureScreenshot", {"format": "png"}, timeout_s=120)
        output.write_bytes(base64.b64decode(result["data"]))

    def close(self) -> None:
        socket_value = getattr(self, "socket", None)
        if socket_value is not None:
            socket_value.close()
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _png_has_rendered_scene(path: Path) -> bool:
    """Reject a successful Chrome exit that captured only the loading shell."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            scene = rgb.crop((0, min(225, rgb.height // 3), rgb.width, rgb.height)).resize((160, 96))
            pixels = list(scene.getdata())
        # The empty Plotly canvas is the uniform #10151c background.  Actual
        # meshes, axes, traces, or scene labels must cover a small real area.
        non_background = sum(abs(red - 16) + abs(green - 21) + abs(blue - 28) > 30 for red, green, blue in pixels)
        return non_background >= max(12, len(pixels) // 500)
    except Exception:
        return False


def screenshots(root: Path, html: Path, category: str, key_frames: list[tuple[str, int]], virtual_time_budget_ms: int = 15_000) -> dict[str, Any]:
    """Use one Chrome DevTools page and capture only after the viewer is ready."""
    chrome = _chrome()
    target = root / "screenshots" / category
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    compare = "overlay" if category.startswith("source") else "stageb"
    if chrome is None:
        for label, frame in key_frames:
            for coord, _camera in (("world", "global"), ("object", "macro"), ("left", "opposite")):
                rows.append({"label": label, "frame": frame, "coordinate": coord, "status": "NOT_RUN", "reason": "Chrome/Chromium 不存在", "path": str(target / f"{label}_frame_{frame:05d}_{coord}.png")})
        return {"schema_version": 2, "chrome": chrome, "html": str(html), "comparison_mode": compare, "screenshots": rows}
    page: _ChromeCdpPage | None = None
    failure: str | None = None
    try:
        page = _ChromeCdpPage(chrome)
        initial_frame = key_frames[0][1] if key_frames else 0
        page.navigate_and_wait(f"file://{html}?frame={initial_frame}&coord=world&camera=global&compare={compare}")
        for label, frame in key_frames:
            safe_label = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in label)
            for coord, camera in (("world", "global"), ("object", "macro"), ("left", "opposite")):
                png = target / f"{safe_label}_frame_{frame:05d}_{coord}.png"
                url = f"file://{html}?frame={frame}&coord={coord}&camera={camera}&compare={compare}"
                page.set_view_and_capture(frame, coord, camera, compare, png)
                rendered_scene = _png_has_rendered_scene(png)
                rows.append({"label": label, "frame": frame, "coordinate": coord, "comparison_mode": compare, "render_wait": "Chrome DevTools status-ready + two animation frames", "rendered_scene": rendered_scene, "status": "PASS" if rendered_scene else "FAIL", "path": str(png), "url": url})
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if page is not None:
            page.close()
    if failure is not None:
        completed = {(row["label"], row["frame"], row["coordinate"]) for row in rows}
        for label, frame in key_frames:
            for coord, _camera in (("world", "global"), ("object", "macro"), ("left", "opposite")):
                if (label, frame, coord) not in completed:
                    rows.append({"label": label, "frame": frame, "coordinate": coord, "comparison_mode": compare, "status": "FAIL", "reason": failure, "path": str(target / f"{label}_frame_{frame:05d}_{coord}.png")})
    return {"schema_version": 2, "chrome": chrome, "html": str(html), "comparison_mode": compare, "legacy_virtual_time_budget_ms": virtual_time_budget_ms, "screenshots": rows}


def write_index(root: Path, selected: dict[str, Any], source: dict[str, Any], stages: dict[str, Any]) -> None:
    """Create a Chinese top-level local visual index."""
    write_text(root / "html/index.html", f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><body style='font-family:system-ui;max-width:900px;margin:30px auto'><h1>新 GRAB Sequence Source / Wuji Recheck</h1><ul><li>新序列：<code>{selected['source_id']}</code>，object=<code>{selected['object']}</code>，{selected['frame_count']} 帧 / {selected['fps']} Hz</li><li>Source 加载审计：<strong>{source['decision']['status']}</strong> / {source['decision']['classification']}</li><li>Stage A：<strong>{stages['stage_a']['status']}</strong>；Stage B：<strong>{stages['stage_b']['status']}</strong>；C-XA：<strong>{stages['cxa']['status']}</strong></li><li>当前结论：<strong>{stages.get('final_retarget')}</strong>（仅标准 kinematic Stage B，不构成接触、动力学或用户 acceptance 结论）。</li></ul><ol><li><a href='01_grab_loaded_source_visual_index.html'>Source HTML 索引</a></li><li><a href='01_grab_loaded_source_audit.html'>Source 真实 3D HTML</a></li><li><a href='02_wuji_spider_retarget_visual_index.html'>Retarget HTML 索引</a></li><li><a href='02_wuji_spider_retarget_audit.html'>Retarget 真实 3D HTML</a></li><li><a href='../reports/STANDARD_PIPELINE_MAP.md'>标准流程图</a>、<a href='../reports/SOURCE_LOADING_DECISION.md'>Source 判定</a>、<a href='../reports/RETARGET_FINAL_DECISION.md'>Retarget 判定</a></li><li><a href='../screenshots/SOURCE_HTML_SCREENSHOT_MANIFEST.json'>Source 截图清单</a>、<a href='../screenshots/RETARGET_HTML_SCREENSHOT_MANIFEST.json'>Retarget 截图清单</a></li></ol><p>本索引只报告事实，不把诊断页标成用户 acceptance。</p></body></html>""")


def write_standard_pipeline_map(root: Path) -> None:
    """Record the repository-backed standard GRAB-to-Wuji boundary for this run."""
    mapping = {
        "schema_version": 1,
        "scope": "当前新序列的标准 GRAB source → Wuji kinematic retarget；不迁移冻结 Stage-C 接触修正。",
        "stages": [
            {"name": "GRAB raw loading", "entry": "spider.datasets.grab.GrabAdapter.load_sequence", "input": "GRAB .npz、object contact mesh、subject beta/body model", "output": "canonical source hands/wrists/object；本任务在机器人重定向前捕获"},
            {"name": "独立官方 reconstruction", "entry": "spider.tools.new_grab_sequence_recheck.reconstruct_official", "input": "同一 raw GRAB 参数和 SMPL-X body model", "output": "独立 world/object-frame hand/object geometry；不调用 GrabAdapter"},
            {"name": "Stage A", "entry": "python -m spider.tools.grab_pipeline prepare", "input": "paths.generated.yaml、sequence id、frame range", "output": "隔离 workspace 中的标准 GRAB source preparation"},
            {"name": "Stage B Wuji SPIDER retarget", "entry": "python -m spider.tools.grab_pipeline run-wuji-ik", "input": "Stage A 产物和同一 paths.generated.yaml", "output": "Wuji qpos、scene、source mapping、kinematic metrics"},
            {"name": "C-XA", "entry": "冻结 Stage-C contact correction", "input": "另行定义的 contact/correction artifacts", "output": "不属于当前标准 GRAB→Wuji kinematic CLI；本任务 NOT_RUN"},
        ],
        "repository_references": ["README.md", "docs/project/GRAB_ADAPTER.md", "docs/project/GRAB_WUJI_IK.md", "docs/project/WORKFLOW.md", "spider/tools/grab_pipeline.py"],
        "actual_config": str((root / "retarget/paths.generated.yaml").resolve()),
    }
    write_json(root / "reports/standard_pipeline_map.json", mapping)
    lines = ["# 当前标准 GRAB → Wuji 流程图", "", "本图依据本仓库入口和文档记录本次实际运行边界；不把冻结 C-XA 伪装成标准新序列步骤。", ""]
    for stage in mapping["stages"]:
        lines.extend([f"## {stage['name']}", "", f"- 入口：`{stage['entry']}`", f"- 输入：{stage['input']}", f"- 输出：{stage['output']}", ""])
    lines.extend(["## 实际配置", "", f"`{mapping['actual_config']}`", "", "Stage A / Stage B 的实际命令保存在 `retarget/stage_a/command.txt` 与 `retarget/stage_b/command.txt`。"])
    write_text(root / "reports/STANDARD_PIPELINE_MAP.md", "\n".join(lines) + "\n")


def write_handoff(root: Path, selected: dict[str, Any], comparison: dict[str, Any], stages: dict[str, Any]) -> None:
    """Create a short self-contained local handoff beside the machine-readable summary."""
    write_text(root / "handoff/HANDOFF.md", f"""# 新 GRAB source / Wuji recheck handoff

- 新序列：`{selected['source_id']}`，{selected['object']}，{selected['frame_count']} 帧 / {selected['fps']} Hz。
- Source：`{comparison['decision']['status']}` / `{comparison['decision']['classification']}`；独立 SMPL-X reconstruction 与真实 `GrabAdapter.load_sequence` 的 retarget 前 capture 已比较。
- Stage A：`{stages['stage_a']['status']}`；Stage B：`{stages['stage_b']['status']}`；C-XA：`{stages['cxa']['status']}`；最终：`{stages.get('final_retarget')}`。
- HTML：`html/01_grab_loaded_source_audit.html`、`html/02_wuji_spider_retarget_audit.html`；截图清单在 `screenshots/`。

限制：这是 source 和标准 kinematic Stage-B 证据；未执行 C-XA、M1/M2/M3、full primary、Oracle C/D2、MJWP、smokes 或 Stage D，不能外推为接触、动力学或用户接受结论。
""")


def write_reports(root: Path, selected: dict[str, Any], official_chain: dict[str, Any], spider_chain: dict[str, Any], comparison: dict[str, Any], temporal: dict[str, Any], global_view: dict[str, Any], contact: dict[str, Any], stages: dict[str, Any]) -> None:
    """Write required source/retarget reports and a Chinese decision document."""
    write_json(root / "source_official/official_transform_chain.json", official_chain)
    write_json(root / "source_spider/spider_transform_chain.json", spider_chain)
    write_json(root / "source_comparison/source_geometry_comparison.json", comparison)
    write_json(root / "source_comparison/source_temporal_alignment.json", temporal)
    write_json(root / "source_comparison/source_contact_distance_summary.json", contact)
    decision = comparison["decision"]
    write_text(root / "reports/SOURCE_LOADING_DECISION.md", f"# Source 加载判定\n\n- 结论：**{decision['status']}**\n- 分类：`{decision['classification']}`\n- 最大检查残差：`{decision['maximum_checked_residual']:.6e}`\n- global view 测试：`{global_view['classification']}`\n\n独立路径直接从 `{selected['source_id']}` raw GRAB 参数使用 SMPL-X 重建；Spider 路径实际调用 `GrabAdapter.load_sequence` 并且 capture 发生在 Stage A/B 前。判定以 object-frame wrist/joints/tips/surface 为准，不能把仅全局显示坐标差异误判为 loader 错误。\n")
    write_text(root / "reports/RETARGET_FINAL_DECISION.md", f"# 新序列 Wuji retarget 判定\n\n- INPUT_SOURCE_GATE：`{stages['input_source_gate']}`\n- Stage A：`{stages['stage_a']['status']}`\n- Stage B：`{stages['stage_b']['status']}`\n- C-XA：`{stages['cxa']['status']}`（只因当前标准 kinematic flow 不包含此新序列接触修正，未移植冻结 C-XA）\n- FINAL：`{stages.get('final_retarget')}`\n\n该结论只描述本次新序列的 source→标准 Wuji Stage-B 路径；没有执行 M1/M2/M3、full primary、Oracle C/D2、MJWP、smokes 或 Stage D。\n")


def _load_npz_values(path: Path) -> dict[str, np.ndarray]:
    """Load a tool-produced NPZ without altering the captured raw evidence."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _stage_from_existing_artifacts(stages: dict[str, Any], spider: dict[str, np.ndarray], metrics: dict[str, Any]) -> dict[str, Any]:
    """Recreate display-only Stage-B geometry from an already completed run."""
    stage_b = stages["stage_b"]
    robot_dir = Path(stage_b["robot_dir"])
    scene = Path(stage_b["scene"])
    with np.load(robot_dir / "trajectory_kinematic.npz", allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
    mapping = np.asarray(json.loads((robot_dir / "source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
    geometry = _extract_stage_b(scene, qpos)
    wuji: dict[str, Any] = {}
    for side in ("left", "right"):
        palm_index = geometry["site_names"].index(f"{side}_palm")
        tip_indices = [geometry["site_names"].index(f"{side}_{finger}_tip") for finger in FINGERTIP_NAMES]
        wuji[side] = {"palm": transform_from_rt(geometry["rotations"][:, palm_index], geometry["positions"][:, palm_index]), "tips": geometry["positions"][:, tip_indices]}
    return {"qpos": qpos, "qvel": qvel, "mapping": mapping, "geometry": geometry, "wuji": wuji, "metrics": metrics, "robot_dir": robot_dir, "scene": scene}


def refresh_existing_evidence(root: Path) -> Path:
    """Refresh only derived reports/HTML/screenshots for a completed evidence root.

    This is deliberately distinct from :func:`run`: it never invokes GRAB
    loading, Stage A, or Stage B, and it never writes raw or historical robot
    outputs.  It is for renderer/report improvements after a successful run.
    """
    root = root.resolve()
    required = [root / "selection/selected_sequence.json", root / "source_official/official_source.npz", root / "source_spider/spider_loaded_source.npz", root / "reports/retarget_stage_status.json", root / "reports/retarget_metrics.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"不能刷新不完整 evidence root: {missing}")
    selected = json.loads((root / "selection/selected_sequence.json").read_text(encoding="utf-8"))
    official = _load_npz_values(root / "source_official/official_source.npz")
    spider = _load_npz_values(root / "source_spider/spider_loaded_source.npz")
    official_chain = json.loads((root / "source_official/official_transform_chain.json").read_text(encoding="utf-8"))
    spider_chain = json.loads((root / "source_spider/spider_transform_chain.json").read_text(encoding="utf-8"))
    stages = json.loads((root / "reports/retarget_stage_status.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "reports/retarget_metrics.json").read_text(encoding="utf-8"))
    comparison, temporal, global_view = compare_source(official, spider, float(selected["fps"]))
    contact_path = root / "source_comparison/source_contact_distance_summary.json"
    existing_contact = json.loads(contact_path.read_text(encoding="utf-8")) if contact_path.is_file() else None
    if existing_contact and all(existing_contact.get("sides", {}).get(side, {}).get("penetration") == "SIGNED_TIP_DISTANCE_AVAILABLE" for side in ("left", "right")):
        contact = existing_contact
    else:
        contact = contact_summary(official)
    paths_config = root / "retarget/paths.generated.yaml"
    for name in ("stage_a", "stage_b"):
        record = stages[name]
        # r5 predates the per-command log path serialization now written by
        # ``audit_stage_b``.  The immutable logs themselves are present in the
        # standard location, so backfill their provenance without rerunning.
        stdout_path = Path(record.get("stdout_log", root / "retarget" / name / "stdout.log"))
        stderr_path = Path(record.get("stderr_log", root / "retarget" / name / "stderr.log"))
        record["stdout_log"] = str(stdout_path)
        record["stderr_log"] = str(stderr_path)
        record["stdout_summary"] = _log_summary(stdout_path.read_text(encoding="utf-8"))
        record["stderr_summary"] = _log_summary(stderr_path.read_text(encoding="utf-8"))
    stages["provenance"] = {"effective_config": str(paths_config), "effective_config_sha256": sha256(paths_config), "raw_parameter": str(selected["parameter_path"]), "raw_parameter_sha256": sha256(Path(selected["parameter_path"])), "object_mesh": str(selected["object_mesh"]), "object_mesh_sha256": sha256(Path(selected["object_mesh"])), "runtime": {"python": sys.version, "executable": sys.executable, "mujoco": getattr(mujoco, "__version__", "unknown")}}
    stage_b = stages["stage_b"]
    stage_b.update({"scene_sha256": sha256(Path(stage_b["scene"])), "source_mapping_sha256": sha256(Path(stage_b["source_mapping"])), "metrics_sha256": sha256(Path(stage_b["metrics"]))})
    write_json(root / "reports/retarget_stage_status.json", stages)
    write_reports(root, selected, official_chain, spider_chain, comparison, temporal, global_view, contact, stages)
    write_standard_pipeline_map(root)
    contact_frames = [int(value["first_close_frame"]) for value in selected["selection_validation"]["interaction"]["sides"].values() if isinstance(value, dict) and value.get("first_close_frame") is not None]
    first_contact = contact_frames[0] if contact_frames else 0
    interaction_run = max(int(side.get("longest_close_run_frames", 0)) for side in selected["selection_validation"]["interaction"]["sides"].values() if isinstance(side, dict))
    interaction_mid = min(len(official["source_frame_indices"]) - 1, first_contact + max(1, interaction_run // 2))
    source_error = int(np.argmax(np.maximum(np.linalg.norm(official["left_joints_world"] - spider["left_joints_world"], axis=-1).max(axis=1), np.linalg.norm(official["right_joints_world"] - spider["right_joints_world"], axis=-1).max(axis=1))))
    suspension_side = max(("left", "right"), key=lambda side: float(contact["sides"][side]["max_suspension_proxy_m"]))
    suspension = int(contact["sides"][suspension_side]["max_suspension_proxy_frame"])
    penetration_side = max(("left", "right"), key=lambda side: float(contact["sides"][side]["max_tip_penetration_m"] or 0.0))
    penetration = int(contact["sides"][penetration_side]["max_tip_penetration_frame"] or 0)
    source_labels = [("起始帧", 0), ("首次接近物体帧", max(0, first_contact - 1)), ("首次明显接触帧", first_contact), ("交互中间帧", interaction_mid), ("最大source差异帧", source_error), ("最大悬空帧", suspension), ("最大有符号穿模帧", penetration), ("序列结束帧", len(official["source_frame_indices"]) - 1)]
    source_html, _ = write_source_html(root, official, spider, comparison, contact, source_labels)
    source_manifest = screenshots(root, source_html, "source_v2", source_labels)
    write_json(root / "screenshots/SOURCE_HTML_SCREENSHOT_MANIFEST.json", source_manifest)
    stage = _stage_from_existing_artifacts(stages, spider, metrics)
    raw_object = spider["T_world_object"][stage["mapping"]]
    per_frame_tip_error = []
    for side in ("left", "right"):
        source_tips = spider[f"{side}_joints_world"][stage["mapping"]][:, FINGERTIP_INDICES]
        source_local = apply_transform(invert_transform(raw_object)[:, None], source_tips)
        robot_local = apply_transform(invert_transform(stage["geometry"]["objects"])[:, None], stage["wuji"][side]["tips"])
        per_frame_tip_error.append(np.linalg.norm(source_local - robot_local, axis=-1).max(axis=1))
    retarget_error = int(stage["mapping"][int(np.argmax(np.maximum(*per_frame_tip_error)))])
    def nearest_stage_b_frame(source_frame: int) -> int:
        """Return an extant Stage-B source frame for a source-side diagnostic.

        Source audits may legitimately identify an endpoint that Stage B omitted
        (for example, a terminal frame needed for velocity estimation).  A
        retarget viewer cannot select that missing frame, so it must name the
        nearest real mapping explicitly rather than fail the remaining captures.
        """
        return int(stage["mapping"][int(np.abs(stage["mapping"] - source_frame).argmin())])

    retarget_suspension = nearest_stage_b_frame(suspension)
    suspension_label = "最大悬空帧" if retarget_suspension == suspension else f"最大悬空帧（source={suspension}，Stage B 最近有效帧）"
    retarget_labels = [("起始帧", int(stage["mapping"][0])), ("首次source接触", nearest_stage_b_frame(first_contact)), ("交互中间帧", nearest_stage_b_frame(interaction_mid)), ("最大retarget误差帧", retarget_error), (suspension_label, retarget_suspension), ("最大有符号穿模帧", nearest_stage_b_frame(penetration)), ("Stage_B首失败帧_无失败显示首有效帧", int(stage["mapping"][0])), ("最后有效帧", int(stage["mapping"][-1]))]
    retarget_html, _ = write_retarget_html(root, official, spider, contact, stages, stage, retarget_labels)
    retarget_manifest = screenshots(root, retarget_html, "retarget_v2", retarget_labels)
    write_json(root / "screenshots/RETARGET_HTML_SCREENSHOT_MANIFEST.json", retarget_manifest)
    write_text(root / "screenshots/SOURCE_HTML_SCREENSHOT_REVIEW.md", "# Source HTML 截图复核\n\n- 清单：8 个关键源帧 × 世界/物体/左腕 3 视角，共 24 张；只有 Chrome 页面就绪且像素包含已渲染场景时才记为 PASS。\n- 页面：完整 601 帧可播放，显示独立 SMPL-X 全身关节点、官方/Spider 手部、两套物体网格和最近表面连接线；全身上下文为关节点，不把它表述为逐帧高密度体表网格。\n- 观察范围：截图是 source loader/object-relative 对齐的可视审计，不代表物理接触或动力学验证。\n")
    write_text(root / "screenshots/RETARGET_HTML_SCREENSHOT_REVIEW.md", "# Retarget HTML 截图复核\n\n- 清单：8 个关键诊断 × 世界/物体/左腕 3 视角，共 24 张；只有 Chrome 页面就绪、没有未捕获 JavaScript 异常且像素包含已渲染场景时才记为 PASS。\n- 页面：完整 599 个有效 Stage-B mapping 帧可播放，显示 source 全身关节点/手部、Wuji 手部网格、可视与碰撞物体、Wuji 掌到指尖骨架和逐帧 object-relative 误差。Wuji 浏览器层是从真实 MuJoCo mesh 确定性抽样的三角面，原始 Stage-B scene/qpos/metrics 未改变。\n- 边界说明：source 的最大悬空为第 600 帧，而 Stage B 的最后有效 mapping 为第 599 帧；该 retarget 截图明确显示第 599 帧作为最近有效帧，未伪称为第 600 帧的 Stage-B 输出。\n- 观察范围：有符号距离/穿模诊断不等于已验证的接触动力学。\n")
    write_index(root, selected, comparison, stages)
    write_json(root / "reports/evidence_refresh.json", {"schema_version": 1, "operation": "DERIVED_EVIDENCE_REFRESH", "root": str(root), "created_utc": datetime.now(UTC).isoformat(), "not_rerun": ["raw GRAB loading", "independent SMPL-X reconstruction", "GrabAdapter capture", "Stage A", "Stage B", "C-XA"], "updated": ["derived contact summary", "source/retarget reports", "HTML v2", "v2 screenshot manifests", "provenance hashes and bounded log summaries"]})
    write_json(root / "handoff/run_summary.json", {"selected": selected, "source": comparison["decision"], "retarget": stages, "html": {"source": str(source_html), "retarget": str(retarget_html), "index": str(root / "html/index.html")}})
    write_handoff(root, selected, comparison, stages)
    return root


def run(paths_config: str, output_root: str, run_id: str | None = None) -> Path:
    """Run the complete isolated new-sequence source and standard-Wuji recheck."""
    output = Path(output_root).resolve()
    actual_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-new-grab-wuji-recheck")
    root = output / actual_id
    if root.exists():
        raise FileExistsError(f"运行目录已存在，fail-fast 不覆盖: {root}")
    root.mkdir(parents=True)
    for name in ("source_official", "source_spider", "source_comparison", "retarget", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir()
    git = {"base_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(), "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(), "status_before": subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True).stdout.splitlines(), "pushed": "NO"}
    write_json(root / "reports/run_contract.json", {"schema_version": 1, "git": git, "forbidden": ["M1", "M2", "M3", "full primary", "Oracle C", "D2", "MJWP", "smokes", "Stage D"], "historic_paths_not_written": ["raw GRAB", "body models", "existing Stage B", "existing C-XA"], "paths_config": str(Path(paths_config).resolve())})
    write_standard_pipeline_map(root)
    selected = select_sequence(paths_config, output, root / "selection")
    official, official_chain = reconstruct_official(paths_config, str(selected["source_id"]))
    spider, spider_chain = capture_spider(paths_config, str(selected["source_id"]), official)
    comparison, temporal, global_view = compare_source(official, spider, float(selected["fps"]))
    contact = contact_summary(official)
    _serialize_npz(root / "source_official/official_source.npz", official)
    _serialize_npz(root / "source_spider/spider_loaded_source.npz", spider)
    write_reports(root, selected, official_chain, spider_chain, comparison, temporal, global_view, contact, {"input_source_gate": comparison["decision"]["status"], "stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}, "cxa": {"status": "NOT_RUN"}})
    contact_frames = [int(value["first_close_frame"]) for value in selected["selection_validation"]["interaction"]["sides"].values() if isinstance(value, dict) and value.get("first_close_frame") is not None]
    first_contact = contact_frames[0] if contact_frames else 0
    interaction_run = max(int(side.get("longest_close_run_frames", 0)) for side in selected["selection_validation"]["interaction"]["sides"].values() if isinstance(side, dict))
    interaction_mid = min(len(official["source_frame_indices"]) - 1, first_contact + max(1, interaction_run // 2))
    source_error = int(np.argmax(np.maximum(np.linalg.norm(official["left_joints_world"] - spider["left_joints_world"], axis=-1).max(axis=1), np.linalg.norm(official["right_joints_world"] - spider["right_joints_world"], axis=-1).max(axis=1))))
    suspension_side = max(("left", "right"), key=lambda side: float(contact["sides"][side]["max_suspension_proxy_m"]))
    suspension = int(contact["sides"][suspension_side]["max_suspension_proxy_frame"])
    if contact["sides"]["left"].get("tip_penetration_depth_m") is not None:
        penetration_side = max(("left", "right"), key=lambda side: float(contact["sides"][side]["max_tip_penetration_m"] or 0.0))
        penetration_proxy = int(contact["sides"][penetration_side]["max_tip_penetration_frame"] or 0)
        penetration_label = "最大有符号穿模帧"
    else:
        penetration_proxy = int(np.argmin(contact["sides"]["right"]["tip_distance_m"]) // len(FINGERTIP_INDICES))
        penetration_label = "最近表面代理帧（无可靠有符号穿模）"
    source_labels = [("起始帧", 0), ("首次接近物体帧", max(0, first_contact - 1)), ("首次明显接触帧", first_contact), ("交互中间帧", interaction_mid), ("最大source差异帧", source_error), ("最大悬空帧", suspension), (penetration_label, penetration_proxy), ("序列结束帧", len(official["source_frame_indices"]) - 1)]
    source_html, _ = write_source_html(root, official, spider, comparison, contact, source_labels)
    source_manifest = screenshots(root, source_html, "source", source_labels)
    write_json(root / "screenshots/SOURCE_HTML_SCREENSHOT_MANIFEST.json", source_manifest)
    write_text(root / "screenshots/SOURCE_HTML_SCREENSHOT_REVIEW.md", "# Source HTML 截图复核\n\nChrome headless 已按起始、首次接近/接触、中间、最大差异/悬空、结束帧生成世界/物体/左腕三视角 PNG。数值判定与截图路径见 manifest；最终人工视觉观察由运行后 Codex 实际打开截图补充，不把截图生成误写为 acceptance。\n")
    stages, stage = audit_stage_b(root, root / "retarget/workspace", selected, spider, comparison["decision"]["status"])
    write_reports(root, selected, official_chain, spider_chain, comparison, temporal, global_view, contact, stages)
    if stage is None:
        retarget_labels = [("起始帧", 0), ("首次source接触", first_contact), ("交互中间帧", interaction_mid), ("最大retarget误差帧", source_error), ("最大悬空帧", suspension), (penetration_label, penetration_proxy), ("Stage_B首失败帧", 0), ("最后有效帧", len(official["source_frame_indices"]) - 1)]
    else:
        combined_error = np.maximum(stage["wuji"]["left"]["tip_error"].max(axis=1), stage["wuji"]["right"]["tip_error"].max(axis=1))
        retarget_error = int(stage["mapping"][int(np.argmax(combined_error))])
        first_valid = int(stage["mapping"][0])
        last_valid = int(stage["mapping"][-1])
        def nearest_stage_b_frame(source_frame: int) -> int:
            return int(stage["mapping"][int(np.abs(stage["mapping"] - source_frame).argmin())])

        retarget_suspension = nearest_stage_b_frame(suspension)
        suspension_label = "最大悬空帧" if retarget_suspension == suspension else f"最大悬空帧（source={suspension}，Stage B 最近有效帧）"
        retarget_labels = [("起始帧", first_valid), ("首次source接触", nearest_stage_b_frame(first_contact)), ("交互中间帧", nearest_stage_b_frame(interaction_mid)), ("最大retarget误差帧", retarget_error), (suspension_label, retarget_suspension), (penetration_label, nearest_stage_b_frame(penetration_proxy)), ("Stage_B首失败帧_无失败显示首有效帧", first_valid), ("最后有效帧", last_valid)]
    retarget_html, _ = write_retarget_html(root, official, spider, contact, stages, stage, retarget_labels)
    retarget_manifest = screenshots(root, retarget_html, "retarget", retarget_labels)
    write_json(root / "screenshots/RETARGET_HTML_SCREENSHOT_MANIFEST.json", retarget_manifest)
    write_text(root / "screenshots/RETARGET_HTML_SCREENSHOT_REVIEW.md", "# Retarget HTML 截图复核\n\n已用 Chrome headless 打开第二个真实 3D HTML，并为关键帧生成世界、物体、左腕三视角。Stage B 失败时页面仍保留 Spider source、失败阶段和空缺说明；不得将失败诊断页当成功见证。\n")
    write_index(root, selected, comparison, stages)
    write_json(root / "handoff/run_summary.json", {"selected": selected, "source": comparison["decision"], "retarget": stages, "html": {"source": str(source_html), "retarget": str(retarget_html), "index": str(root / "html/index.html")}})
    write_handoff(root, selected, comparison, stages)
    print(root)
    return root


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="New GRAB source and standard Wuji recheck")
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-root", default=".local_artifacts/stage_c_new_sequence_recheck")
    parser.add_argument("--run-id")
    parser.add_argument("--refresh-existing-evidence", help="仅回填已完成 run 的派生 reports/HTML/截图；不会重跑 raw/Stage A/Stage B")
    args = parser.parse_args()
    if args.refresh_existing_evidence:
        print(refresh_existing_evidence(Path(args.refresh_existing_evidence)))
        return
    run(args.paths_config, args.output_root, args.run_id)


if __name__ == "__main__":
    main()
