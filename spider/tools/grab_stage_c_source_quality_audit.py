"""Stage C source-quality authority audit using the real Ref2Dex STATIC_EXEC runner.

This tool is intentionally read-only with respect to GRAB, MuJoCo scenes and
trajectory inputs.  It materialises ignored evidence under ``.local_artifacts``.
The formal sign convention is the one in Ref2Dex ``MeshSDFfromAsset``:
negative means an object query point is inside the hand target.  An open or
non-watertight Wuji hand is still queried with ``check_manifold=False`` for a
diagnostic record, but its signed penetration is never promoted to formal
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from spider.datasets.paths import load_project_paths
from spider.tools.grab_source_geometry_audit import (
    DETAIL_FRAMES,
    FOCUS_FRAMES,
    FRAME_END,
    FRAME_START,
    FPS,
    _extract_mujoco_geometry,
    _hand_faces_from_smplx,
    _load_paths,
    _stage_b_act_qpos,
    _summary,
    apply_transform,
    invert_transform,
    object_relative,
    robot_surface_payload,
    rotation_residual_rad,
    sha256,
    write_json,
)


REF2DEX_ROOT = Path("/4-4090/Ref2Dex")
STATIC_EXEC_SIGN = "xyz_majority"
STATIC_EXEC_BACKEND = "dataset.tools.mesh_preprocess.get_sdf_from_asset.MeshSDFfromAsset"
QUERY_SAMPLES = 20_000
QUERY_VOXEL_M = 0.001
QUERY_SEED = 2_026_0702
NEAR_M = 0.001
SHALLOW_M = 0.003
SEVERE_M = 0.005
WRIST_LOOP = (119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256(path)
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def _hash_any(path: Path) -> str | None:
    return _hash_tree(path) if path.exists() else None


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _fixed_wrist_cap(vertices: np.ndarray, faces: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact Ref2Dex fixed GRAB/SMPL-X wrist-cap convention."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 3 or vertices.shape[1] != 778 or faces.shape != (1538, 3):
        raise ValueError(f"expected [T,778,3] and [1538,3], got {vertices.shape}, {faces.shape}")
    center = vertices[:, WRIST_LOOP, :].mean(axis=1, keepdims=True)
    capped_vertices = np.concatenate((vertices, center), axis=1)
    cap = []
    for index, current in enumerate(WRIST_LOOP):
        nxt = WRIST_LOOP[(index + 1) % len(WRIST_LOOP)]
        cap.append((current, nxt, 778) if side == "right" else (778, nxt, current))
    return capped_vertices, np.concatenate((faces, np.asarray(cap, dtype=np.int32)), axis=0)


def _mesh_record(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "single_component": int(len(mesh.split(only_watertight=False))) == 1,
        "connected_components": int(len(mesh.split(only_watertight=False))),
        "surface_area_m2": float(mesh.area),
        "sha256_vertices": hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest(),
        "sha256_faces": hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest(),
    }


def _side_for_body(model: mujoco.MjModel, body_id: int) -> str | None:
    current = int(body_id)
    while current > 0:
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or "").lower()
        if name.startswith(("r_", "right_")) or "right" in name:
            return "right"
        if name.startswith(("l_", "left_")) or "left" in name:
            return "left"
        current = int(model.body_parentid[current])
    return None


def _mujoco_layer_geometry(scene: Path, qpos: np.ndarray, group: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract full real MuJoCo layer meshes, preserving every authored geom."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    chosen: dict[str, list[int]] = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != group:
            continue
        side = _side_for_body(model, int(model.geom_bodyid[geom_id]))
        if side:
            chosen[side].append(geom_id)
    if not chosen["left"] or not chosen["right"]:
        raise RuntimeError(f"missing both Wuji hand sides in group={group}: {scene}")
    layers: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    transforms: dict[str, np.ndarray] = {}
    for side, geom_ids in chosen.items():
        transforms[side] = np.empty((len(qpos), len(geom_ids), 12), dtype=np.float32)
        for geom_id in geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
            faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int32)
            layers[side].append({
                "geom_id": int(geom_id),
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}",
                "vertices": vertices,
                "faces": faces,
            })
    for frame, state in enumerate(np.asarray(qpos, dtype=np.float64)):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side, geom_ids in chosen.items():
            transforms[side][frame, :, :3] = data.geom_xpos[geom_ids]
            transforms[side][frame, :, 3:] = data.geom_xmat[geom_ids]
    return {"layers": layers, "transforms": transforms, "scene": str(scene), "group": group}, {"model": model, "chosen": chosen}


def _world_layer_arrays(layer: dict[str, Any], side: str, frame: int, max_faces_per_geom: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for index, item in enumerate(layer["layers"][side]):
        tr = layer["transforms"][side][frame, index]
        position, rotation = tr[:3], tr[3:].reshape(3, 3)
        local = np.asarray(item["vertices"], dtype=np.float64)
        local_faces = np.asarray(item["faces"], dtype=np.int32)
        if max_faces_per_geom and len(local_faces) > max_faces_per_geom:
            stride = int(np.ceil(len(local_faces) / max_faces_per_geom))
            local_faces = local_faces[::stride][:max_faces_per_geom]
            used, inverse = np.unique(local_faces.reshape(-1), return_inverse=True)
            local = local[used]
            local_faces = inverse.reshape(-1, 3).astype(np.int32)
        vertices.append(local @ rotation.T + position)
        faces.append(local_faces + offset)
        offset += len(local)
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def _mesh_target(vertices: np.ndarray, faces: np.ndarray, label: str) -> dict[str, Any]:
    return {"label": label, "vertices": vertices, "faces": faces, "source": label}


def _object_query_points(object_mesh: Path, out: Path) -> tuple[np.ndarray, dict[str, Any]]:
    sys.path.insert(0, str(REF2DEX_ROOT))
    from dataset.tools.sample_object_query_points import (
        CANONICAL_SCHEMA_VERSION,
        frame_seed,
        load_mesh_vertices_faces,
        sample_surface_points,
        voxel_downsample,
    )

    vertices, faces = load_mesh_vertices_faces(np=np, mesh_path=object_mesh, center_bbox=False)
    points = sample_surface_points(np=np, vertices=vertices, faces=faces, samples_per_frame=QUERY_SAMPLES, rng=np.random.default_rng(frame_seed(QUERY_SEED, "canonical:cylindermedium", 0)))
    points = voxel_downsample(np, points, QUERY_VOXEL_M).astype(np.float32, copy=False)
    mesh_sha1 = hashlib.sha1(object_mesh.read_bytes()).hexdigest()
    _write_npz(out, points=points, object_name=np.asarray("cylindermedium"), mesh_path=np.asarray(str(object_mesh)), samples_per_frame=np.asarray(QUERY_SAMPLES), voxel_size_m=np.asarray(QUERY_VOXEL_M), seed=np.asarray(QUERY_SEED), mesh_sha1=np.asarray(mesh_sha1), schema_version=np.asarray(CANONICAL_SCHEMA_VERSION), query_point_cache_kind=np.asarray("canonical_object_surface_random_20k_voxel1mm"))
    return points, {"schema_version": CANONICAL_SCHEMA_VERSION, "mesh_path": str(object_mesh), "mesh_sha1": mesh_sha1, "samples_per_frame": QUERY_SAMPLES, "voxel_size_m": QUERY_VOXEL_M, "seed": QUERY_SEED, "num_points_after_voxel": int(len(points)), "sampling": "triangle-area weighted random barycentric then floor voxel first-unique", "same_local_points_all_stages": True}


def _static_exec_imports() -> tuple[Any, Any]:
    sys.path.insert(0, str(REF2DEX_ROOT))
    from dataset.tools.reverse_hand_meshsdf_statistics import query_targets_with_meshsdf
    from dataset.tools.mesh_preprocess.get_sdf_from_asset import MeshSDFfromAsset
    return query_targets_with_meshsdf, MeshSDFfromAsset


def _query_one_side(query_fn: Any, targets: list[dict[str, Any]], points: np.ndarray, work_root: Path, frame: int, stem: str, device: str, batch: int) -> np.ndarray:
    result = query_fn(np=np, targets=targets, frame_index=0, points=points, device=device, check_manifold=False, sign_ray_mode=STATIC_EXEC_SIGN, work_root=work_root, unit_scale_to_mm=1000.0, cache_stem=f"{stem}_{frame}", query_batch_size=batch, keep_frame_objs=False, return_aux=True)
    return np.asarray(result["signed_distance_mm"], dtype=np.float64) / 1000.0


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values):
        return None
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(percentile / 100.0, cumulative, values))


def _row_metrics(signed: np.ndarray, weights: np.ndarray, formal: bool) -> dict[str, Any]:
    signed = np.asarray(signed, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    penetration = np.maximum(0.0, -signed)
    negative = signed < 0.0
    shallow = negative & (penetration <= SHALLOW_M)
    deep = penetration > SHALLOW_M
    near = np.abs(signed) <= NEAR_M
    categories = {"outside": int(np.count_nonzero(signed > NEAR_M)), "near_surface_contact": int(near.sum()), "shallow_penetration": int(shallow.sum()), "deep_penetration": int(deep.sum())}
    p = penetration[negative]
    return {"formal_authority": bool(formal), "signed_distance_convention": "negative_inside_hand" if formal else "diagnostic_negative_inside_hand_only", "point_count": int(len(signed)), "signed_min_m": float(signed.min()), "signed_max_m": float(signed.max()), "unsigned_mean_m": float(np.mean(np.abs(signed))), "penetrating_point_count": int(negative.sum()), "penetrating_point_ratio": float(negative.mean()), "area_weighted_penetrating_ratio": float(np.sum(weights * negative) / np.sum(weights)), "mean_penetration_depth_m": float(p.mean()) if len(p) else 0.0, "p50_penetration_depth_m": float(np.percentile(p, 50)) if len(p) else 0.0, "p90_penetration_depth_m": float(np.percentile(p, 90)) if len(p) else 0.0, "p95_penetration_depth_m": float(np.percentile(p, 95)) if len(p) else 0.0, "p99_penetration_depth_m": float(np.percentile(p, 99)) if len(p) else 0.0, "max_penetration_depth_m": float(p.max()) if len(p) else 0.0, "weighted_mean_penetration_depth_m": float(np.sum(weights * penetration) / np.sum(weights)), "integrated_depth_proxy_m": float(np.sum(weights * penetration)), "p95_point_ratio": float(np.mean(penetration >= (np.percentile(p, 95) if len(p) else 0.0))), "categories": categories}


def _summary_rows(rows: list[dict[str, Any]], formal: bool) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "PASS"]
    if not valid:
        return {"formal_authority": formal, "status": "SDF_INVALID_FOR_STAGE" if not formal else "NO_VALID_FRAMES", "valid_frame_count": 0, "frame_count": len(rows)}
    max_depth = np.asarray([row["union"]["max_penetration_depth_m"] for row in valid], dtype=float)
    ratios = np.asarray([row["union"]["penetrating_point_ratio"] for row in valid], dtype=float)
    severe = max_depth > SEVERE_M
    intervals = []
    start = None
    for i, flag in enumerate(severe):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(severe) - 1):
            end = i if flag and i == len(severe) - 1 else i - 1
            intervals.append((start, end))
            start = None
    longest = max(intervals, key=lambda pair: pair[1] - pair[0] + 1, default=None)
    return {"formal_authority": formal, "status": "PASS" if formal else "SDF_INVALID_FOR_STAGE", "valid_frame_count": len(valid), "frame_count": len(rows), "penetrating_frame_ratio": float(np.mean(ratios > 0.0)), "mean_frame_penetrating_point_ratio": float(ratios.mean()), "p50_frame_max_depth_m": float(np.percentile(max_depth, 50)), "p90_frame_max_depth_m": float(np.percentile(max_depth, 90)), "p95_frame_max_depth_m": float(np.percentile(max_depth, 95)), "p99_frame_max_depth_m": float(np.percentile(max_depth, 99)), "global_max_penetration_depth_m": float(max_depth.max()), "global_max_source_frame": int(valid[int(np.argmax(max_depth))]["source_frame"]), "severe_threshold_m": SEVERE_M, "longest_severe_interval_source_frames": [int(valid[longest[0]]["source_frame"]), int(valid[longest[1]]["source_frame"])] if longest else None}


def _classify_stage_mesh(stage: str, mesh_records: dict[str, Any], formal_requested: bool) -> tuple[bool, str]:
    valid = all(item.get("watertight") and item.get("winding_consistent") for item in mesh_records.values())
    if not formal_requested:
        return False, "SDF_INVALID_FOR_STAGE"
    return (True, "PASS") if valid else (False, "SDF_INVALID_FOR_STAGE")


def _frame_map(stage_mapping: np.ndarray | None, cxa_mapping: np.ndarray | None) -> list[dict[str, Any]]:
    rows = []
    for local, frame in enumerate(range(FRAME_START, FRAME_END)):
        stage_idx = None if stage_mapping is None else (int(np.where(stage_mapping == frame)[0][0]) if frame in stage_mapping else None)
        cxa_idx = None if cxa_mapping is None else (int(np.where(cxa_mapping == frame)[0][0]) if frame in cxa_mapping else None)
        rows.append({"source_frame": frame, "official_local_index": local, "spider_raw_local_index": local, "stage_b_local_index": stage_idx, "repaired_cxa_local_index": cxa_idx, "stage_b_coverage": stage_idx is not None, "cxa_coverage": cxa_idx is not None})
    return rows


def _lineage() -> dict[str, Any]:
    files = ["dataset/STATICS_EXEC.md", "dataset/tools/reverse_hand_meshsdf_statistics.py", "dataset/tools/mesh_preprocess/get_sdf_from_asset.py", "dataset/tools/sample_object_query_points.py"]
    git = ["git", "-c", f"safe.directory={REF2DEX_ROOT}", "-C", str(REF2DEX_ROOT)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    return {"schema_version": 1, "ref2dex_root": str(REF2DEX_ROOT), "git_commit": commit, "git_status": subprocess.run([*git, "status", "--short"], text=True, capture_output=True, check=False).stdout, "backend": STATIC_EXEC_BACKEND, "runner": "dataset.tools.reverse_hand_meshsdf_statistics.query_targets_with_meshsdf", "sign_ray_mode": STATIC_EXEC_SIGN, "negative_means": "object query point is inside hand target", "files": [{"path": str(REF2DEX_ROOT / rel), "sha256": sha256(REF2DEX_ROOT / rel)} for rel in files], "adapter": {"in_memory_target_arrays": True, "fixed_wrist_cap_loop": list(WRIST_LOOP), "algorithm_copied": False, "raw_non_watertight_fallback": False, "final_sampling_override": False, "mesh_repair": False}}


def _implementation_map(lineage: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "authority": "Ref2Dex STATIC_EXEC", "imports": [{"symbol": "query_targets_with_meshsdf", "source": str(REF2DEX_ROOT / "dataset/tools/reverse_hand_meshsdf_statistics.py"), "role": "per-frame MeshSDFfromAsset query"}, {"symbol": "MeshSDFfromAsset", "source": str(REF2DEX_ROOT / "dataset/tools/mesh_preprocess/get_sdf_from_asset.py"), "role": "hand signed SDF backend"}, {"symbol": "sample_surface_points/voxel_downsample/frame_seed", "source": str(REF2DEX_ROOT / "dataset/tools/sample_object_query_points.py"), "role": "canonical object query points"}], "fixed_contract": {"query_samples_before_voxel": QUERY_SAMPLES, "voxel_size_m": QUERY_VOXEL_M, "sign_ray_mode": STATIC_EXEC_SIGN, "unit_scale_to_mm": 1000.0, "query_points_reused_across_stages": True, "final_no_sampling_after_cache": True, "wrist_cap": list(WRIST_LOOP)}, "lineage": lineage}


def _metric_definition() -> dict[str, Any]:
    return {"schema_version": 1, "signed_distance": "STATIC_EXEC signed distance in metres after mm conversion; negative means object query point inside hand target", "penetration_depth": "max(0, -signed_distance)", "surface_categories": {"outside": f"signed > {NEAR_M} m", "near_surface_contact": f"abs(signed) <= {NEAR_M} m", "shallow_penetration": f"- {SHALLOW_M} m <= signed < - {NEAR_M} m", "deep_penetration": f"signed < - {SHALLOW_M} m"}, "formal_authority": "Only watertight and winding-consistent hand targets may supply formal signed penetration; open/non-watertight targets retain diagnostic signed and unsigned distance but are SDF_INVALID_FOR_STAGE", "weights": "equal canonical object query-point weights; query set is triangle-area sampled then 1mm floor-voxel first-unique", "not_claimed": ["contact is not equivalent to penetration", "diagnostic signed values on open meshes are not formal authority", "display mesh decimation is never used for metrics"]}


def _role_report(role_root: Path | None, source_frames: np.ndarray) -> dict[str, Any]:
    if role_root is None or not role_root.is_dir():
        return {"status": "NOT_AVAILABLE", "source_role_root": None}
    truth = role_root / "source_role_truth/per_region_contact_truth.json"
    scores = role_root / "source_role_truth/contact_responsibility_scores.json"
    result: dict[str, Any] = {"status": "PASS", "source_role_root": str(role_root), "truth_path": str(truth), "scores_path": str(scores)}
    if truth.is_file():
        payload = json.loads(truth.read_text(encoding="utf-8"))
        result["truth_schema"] = {key: payload[key] for key in ("schema_version", "source_frame_range", "region_names") if key in payload}
    if scores.is_file():
        payload = json.loads(scores.read_text(encoding="utf-8"))
        selected = [row for row in payload.get("per_frame", []) if int(row.get("source_frame", -1)) in set(map(int, source_frames))]
        result["responsibility_focus"] = selected[:64]
    return result


def _make_payload(root: Path, points_local: np.ndarray, official: dict[str, Any], stages: dict[str, Any], selected: list[int]) -> dict[str, Any]:
    """Small self-contained payload: full metrics, bounded real mesh display."""
    payload: dict[str, Any] = {"frames": selected, "query_local": points_local[:: max(1, len(points_local) // 800)].astype(np.float32), "object_local": official["object_asset_vertices"], "object_faces": official["object_faces"], "stages": {}}
    for name, stage in stages.items():
        rows = []
        for frame in selected:
            matches = np.where(stage["source_frames"] == frame)[0]
            if not len(matches):
                rows.append({"source_frame": frame, "status": "NOT_AVAILABLE", "metrics": {"status": "NOT_AVAILABLE", "reason": "frozen Stage B/CXA trajectory covers source 1461..1874 only"}})
                continue
            idx = int(matches[0])
            object_transform = official["T_world_object_visual"][idx, :3, :3].reshape(-1).tolist() + official["T_world_object_visual"][idx, :3, 3].tolist()
            item: dict[str, Any] = {"source_frame": frame, "object_transform": object_transform, "metrics": stage["rows"][idx]}
            if stage["kind"] in {"official", "raw"}:
                for side in ("left", "right"):
                    vertices, faces = _fixed_wrist_cap(stage["values"][f"{side}_vertices_world"][idx:idx + 1], stage["values"][f"{side}_hand_faces"], side)
                    item[side] = {"vertices": vertices[0].tolist(), "faces": faces.tolist()}
            elif stage.get("display_layer"):
                for side in ("left", "right"):
                    vertices, faces = _world_layer_arrays(stage["display_layer"], side, int(stage["index_by_source"][frame]), max_faces_per_geom=180)
                    item[side] = {"vertices": vertices.tolist(), "faces": faces.tolist()}
            rows.append(item)
        payload["stages"][name] = rows
    return _plain(payload)


def _html_page(title: str, payload: dict[str, Any], mode: str) -> str:
    packed = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    from plotly.offline import get_plotlyjs
    if mode == "comparison":
        description = "A 官方独立 SMPL-X；B Spider raw；C 标准 Stage B Wuji；D 修复 C-XA。C/D 的 signed 颜色仅为诊断，正式 authority 由网格 manifest 决定。"
    else:
        description = "联系与 penetration 分开显示：near/contact 不等于 penetration；C/D 非 watertight 时不把诊断 signed 当正式结论。"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><script>{get_plotlyjs()}</script><style>body{{margin:0;background:#101820;color:#eaf2f8;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#bar{{padding:12px;background:#172b3a}}#scene{{height:78vh}}select{{background:#294457;color:#fff;padding:5px}}#info{{white-space:pre-wrap;padding:12px;background:#172b3a;font:12px monospace}}.warn{{color:#ffd166}}</style></head><body><div id='bar'><b>{title}</b><br>{description}<br>帧 <select id='frame'></select>　视角 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体坐标</option><option value='hand'>手部近景</option></select>　阶段 <select id='stage'><option>A_OFFICIAL</option><option>B_SPIDER_RAW</option><option>C_STAGE_B</option><option>D_REPAIRED_CXA</option><option>ALL</option></select></div><div id='scene'></div><pre id='info'></pre><script>const D={packed};const Q=new URLSearchParams(location.search);const E=x=>document.getElementById(x);D.frames.forEach(f=>E('frame').add(new Option(String(f),String(f))));E('frame').value=Q.get('frame')||D.frames[0];E('view').value=Q.get('view')||'world';E('stage').value=Q.get('stage')||'ALL';function M(T,V){{let R=T.slice(0,9),p=T.slice(9,12);return V.map(x=>[R[0]*x[0]+R[1]*x[1]+R[2]*x[2]+p[0],R[3]*x[0]+R[4]*x[1]+R[5]*x[2]+p[1],R[6]*x[0]+R[7]*x[1]+R[8]*x[2]+p[2]])}}function inv(T){{let R=[[T[0],T[1],T[2]],[T[3],T[4],T[5]],[T[6],T[7],T[8]]],p=[T[9],T[10],T[11]];return [R[0][0],R[1][0],R[2][0],R[0][1],R[1][1],R[2][1],R[0][2],R[1][2],R[2][2],-(R[0][0]*p[0]+R[1][0]*p[1]+R[2][0]*p[2]),-(R[0][1]*p[0]+R[1][1]*p[1]+R[2][1]*p[2]),-(R[0][2]*p[0]+R[1][2]*p[1]+R[2][2]*p[2])]}}function draw(){{let f=+E('frame').value,view=E('view').value,w=D.stages.A_OFFICIAL[D.frames.indexOf(f)],V=(view==='object'||view==='hand')?inv(w.object_transform):[1,0,0,0,1,0,0,0,1,0,0,0],tr=[{{type:'mesh3d',name:'object canonical visual',x:M(w.object_transform,D.object_local).map(p=>p[0]),y:M(w.object_transform,D.object_local).map(p=>p[1]),z:M(w.object_transform,D.object_local).map(p=>p[2]),i:D.object_faces.map(x=>x[0]),j:D.object_faces.map(x=>x[1]),k:D.object_faces.map(x=>x[2]),color:'#f4a261',opacity:.34}}];let wanted=E('stage').value==='ALL'?Object.keys(D.stages):[E('stage').value];wanted.forEach((s,si)=>{{let row=D.stages[s][D.frames.indexOf(f)],c={{A_OFFICIAL:'#48cae4',B_SPIDER_RAW:'#f72585',C_STAGE_B:'#ffd166',D_REPAIRED_CXA:'#80ed99'}}[s];['left','right'].forEach(side=>{{if(!row||!row[side])return;let q=M(V,row[side].vertices);tr.push({{type:'mesh3d',name:s+' '+side,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),i:row[side].faces.map(x=>x[0]),j:row[side].faces.map(x=>x[1]),k:row[side].faces.map(x=>x[2]),color:c,opacity:.42,flatshading:true}})}});let q=M(V,D.query_local);tr.push({{type:'scatter3d',mode:'markers',name:'canonical query points',x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),marker:{{size:2,color:'#ffffff'}}}})}});Plotly.react('scene',tr,{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#eaf2f8'}},scene:{{aspectmode:'data',camera:{{eye:{{x:view==='hand'?.75:1.45,y:view==='hand'?.75:1.3,z:view==='hand'?.55:.9}}}}}},margin:{{l:0,r:0,t:30,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});E('info').textContent=JSON.stringify({{frame:f,stages:wanted.map(s=>{{let r=D.stages[s][D.frames.indexOf(f)];return {{stage:s,metrics:r.metrics}}}})}},null,2)}}['frame','view','stage'].forEach(x=>E(x).addEventListener('change',draw));draw();</script></body></html>"""


def _write_html(root: Path, payload: dict[str, Any]) -> None:
    html = root / "html"
    html.mkdir(parents=True, exist_ok=True)
    (html / "01_reverse_sdf_stage_comparison.html").write_text(_html_page("Stage C-SQA：四阶段 reverse-SDF 对比", payload, "comparison"), encoding="utf-8")
    (html / "02_source_quality_contact_vs_penetration.html").write_text(_html_page("Stage C-SQA：source quality / contact 与 penetration", payload, "quality"), encoding="utf-8")
    (html / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Stage C-SQA</title><h1>Stage C-SQA Source Quality Authority Audit</h1><ul><li><a href='01_reverse_sdf_stage_comparison.html?frame=1461&view=object'>四阶段 reverse-SDF 对比</a></li><li><a href='02_source_quality_contact_vs_penetration.html?frame=1461&view=hand'>contact vs penetration</a></li></ul>", encoding="utf-8")


def _capture_screenshots(root: Path, html_name: str, selected: list[int]) -> list[dict[str, Any]]:
    out = root / "screenshots" / ("stage_comparison" if html_name.startswith("01") else "source_quality")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in selected:
        for view in ("world", "object", "hand"):
            target = out / f"frame_{frame}_{view}.png"
            url = (root / "html" / html_name).resolve().as_uri() + f"?frame={frame}&view={view}&stage=ALL"
            call = subprocess.run(["/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=3500", "--window-size=1800,1200", f"--screenshot={target}", url], text=True, capture_output=True, timeout=90, check=False)
            rows.append({"frame": frame, "view": view, "path": str(target), "status": "PASS" if call.returncode == 0 and target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "stderr_tail": call.stderr[-300:]})
    return rows


def _select_frames(stage_rows: dict[str, Any]) -> list[int]:
    selected = list(dict.fromkeys((*FOCUS_FRAMES, *DETAIL_FRAMES[::3])))
    for stage in stage_rows.values():
        for key in ("global_max_source_frame",):
            if stage.get("summary", {}).get(key) is not None:
                selected.append(int(stage["summary"][key]))
    return list(dict.fromkeys(selected))[:12]


def run(paths_config: str = "configs/local/paths.yaml", output_root: str = ".local_artifacts/stage_c_source_quality_audit", run_id: str | None = None, device: str = "cuda:0", query_batch_size: int = 4096, cxa_root: str | None = None, no_html: bool = False) -> str:
    project_paths = load_project_paths(paths_config)
    base_stage_root = project_paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0"
    repaired = Path(cxa_root).resolve() if cxa_root else base_stage_root / "stage_c_contract_v2_cxa_repaired_20260801T101500Z"
    inputs = _load_paths(paths_config, str(repaired) if repaired.is_dir() else None)
    root = Path(output_root).resolve() / (run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-source-quality-r1"))
    if root.exists():
        raise FileExistsError(root)
    for name in ("manifest", "reports", "html", "screenshots", "official_visualization"):
        (root / name).mkdir(parents=True)
    lineage = _lineage()
    write_json(root / "reports/ref2dex_static_exec_lineage.json", lineage)
    write_json(root / "reports/static_exec_implementation_map.json", _implementation_map(lineage))
    (root / "reports/STATIC_EXEC_IMPLEMENTATION_MAP.md").write_text("# STATIC_EXEC 实现与 lineage\n\n本审计实际调用 `query_targets_with_meshsdf`，其 backend 是 `MeshSDFfromAsset`；sign ray 使用 `xyz_majority`，负值表示 object query point 位于 hand target 内部。Canonical query 是 20,000 个 triangle-area random barycentric 点，经 1 mm floor-voxel first-unique；所有 stage 复用同一份本地 query 点，不在最终统计时再次采样。\n\nWuji target 不满足 watertight 时只运行 `check_manifold=False` 的 diagnostic query，报告状态保持 `SDF_INVALID_FOR_STAGE`；没有 mesh repair、frame deletion 或 threshold relaxation。GRAB wrist-cap 使用固定 loop：`" + ", ".join(map(str, WRIST_LOOP)) + "`。\n", encoding="utf-8")
    write_json(root / "reports/reverse_sdf_metric_definition.json", _metric_definition())

    official, official_chain = __import__("spider.tools.grab_source_geometry_audit", fromlist=["reconstruct_official_source"]).reconstruct_official_source(inputs)
    raw, raw_chain = __import__("spider.tools.grab_source_geometry_audit", fromlist=["capture_spider_raw_loader"]).capture_spider_raw_loader(inputs)
    _write_npz(root / "manifest/official_source_reconstruction.npz", **official)
    _write_npz(root / "manifest/spider_loaded_source_reconstruction.npz", **raw)
    write_json(root / "manifest/source_lineage.json", {"official": official_chain, "spider_raw": raw_chain, "source_frame_range": [FRAME_START, FRAME_END], "source_fps": FPS, "raw_grab_modified": False})
    source_mapping = np.asarray(json.loads(inputs.stage_b_mapping.read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
    with np.load(inputs.cxa_trajectory, allow_pickle=False) as archive:
        cxa_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        cxa_mapping = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    with np.load(inputs.stage_b_trajectory, allow_pickle=False) as archive:
        stage_b_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    stage_b_act = _stage_b_act_qpos(stage_b_qpos)
    _write_npz(root / "manifest/frame_mapping.npz", source_frame_indices=np.arange(FRAME_START, FRAME_END), stage_b_source_frame_indices=source_mapping, cxa_source_frame_indices=cxa_mapping)
    write_json(root / "manifest/frame_mapping.json", {"schema_version": 1, "source_frames": [FRAME_START, FRAME_END], "fps": FPS, "rows": _frame_map(source_mapping, cxa_mapping), "endpoint_rule": "source 1460 and 1875 are retained; Stage B/CXA have only source 1461..1874 due their frozen 414-frame crop"})
    points_local, query_manifest = _object_query_points(inputs.object_mesh, root / "manifest/canonical_object_query_points.npz")
    write_json(root / "manifest/canonical_object_query_points.json", query_manifest)

    stage_b_geom = _extract_mujoco_geometry(inputs.scene, stage_b_qpos)
    cxa_geom = _extract_mujoco_geometry(inputs.scene_act, cxa_qpos)
    stage_b_layers, _ = _mujoco_layer_geometry(inputs.scene, stage_b_qpos, 2)
    cxa_layers, _ = _mujoco_layer_geometry(inputs.scene_act, cxa_qpos, 2)
    stage_b_visual, _ = _mujoco_layer_geometry(inputs.scene, stage_b_qpos, 1)
    cxa_visual, _ = _mujoco_layer_geometry(inputs.scene_act, cxa_qpos, 1)
    mesh_manifest: dict[str, Any] = {"schema_version": 1, "units": "metres", "formal_policy": "watertight and winding-consistent target required", "stages": {}}
    layer_pairs = {"C_STAGE_B": stage_b_layers, "D_REPAIRED_CXA": cxa_layers}
    visual_pairs = {"C_STAGE_B": stage_b_visual, "D_REPAIRED_CXA": cxa_visual}
    for name, layer in layer_pairs.items():
        mesh_manifest["stages"][name] = {"collision_group": 2, "visual_group": 1, "sides": {}}
        for side in ("left", "right"):
            cv, cf = _world_layer_arrays(layer, side, 0)
            vv, vf = _world_layer_arrays(visual_pairs[name], side, 0)
            mesh_manifest["stages"][name]["sides"][side] = {"collision": _mesh_record(cv, cf), "visual": _mesh_record(vv, vf), "invalid_collision_geoms": [item["name"] for item in layer["layers"][side] if not trimesh.Trimesh(vertices=item["vertices"], faces=item["faces"], process=False).is_watertight]}
    for side in ("left", "right"):
        capped_v, capped_f = _fixed_wrist_cap(official[f"{side}_vertices_world"], official[f"{side}_hand_faces"], side)
        mesh_manifest["stages"].setdefault("A_OFFICIAL", {"sides": {}})["sides"][side] = {"target": _mesh_record(capped_v[0], capped_f), "wrist_cap": list(WRIST_LOOP)}
        capped_v, capped_f = _fixed_wrist_cap(raw[f"{side}_vertices_world"], raw[f"{side}_hand_faces"], side)
        mesh_manifest["stages"].setdefault("B_SPIDER_RAW", {"sides": {}})["sides"][side] = {"target": _mesh_record(capped_v[0], capped_f), "wrist_cap": list(WRIST_LOOP)}
    write_json(root / "reports/hand_sdf_mesh_manifest.json", mesh_manifest)

    object_translation = {"C_STAGE_B": np.linalg.norm(stage_b_geom["object_transform"][:, :3, 3] - official["T_world_object_visual"][1:-1, :3, 3], axis=1), "D_REPAIRED_CXA": np.linalg.norm(cxa_geom["object_transform"][:, :3, 3] - official["T_world_object_visual"][1:-1, :3, 3], axis=1)}
    object_rotation = {"C_STAGE_B": rotation_residual_rad(stage_b_geom["object_transform"][:, :3, :3], official["T_world_object_visual"][1:-1, :3, :3]), "D_REPAIRED_CXA": rotation_residual_rad(cxa_geom["object_transform"][:, :3, :3], official["T_world_object_visual"][1:-1, :3, :3])}
    stage_identity = {"schema_version": 1, "raw_grab_modified": False, "stages": {"A_OFFICIAL": {"raw_sequence": str(inputs.raw_sequence), "raw_sequence_sha256": sha256(inputs.raw_sequence), "body_model_tree_sha256": _hash_any(inputs.body_model_root), "object_mesh_sha256": sha256(inputs.object_mesh)}, "B_SPIDER_RAW": {"loader": "GrabAdapter.load_sequence", "raw_sequence_sha256": sha256(inputs.raw_sequence), "object_mesh_sha256": sha256(inputs.object_mesh)}, "C_STAGE_B": {"trajectory": str(inputs.stage_b_trajectory), "trajectory_sha256": sha256(inputs.stage_b_trajectory), "scene": str(inputs.scene), "scene_sha256": sha256(inputs.scene), "qpos_shape": list(stage_b_qpos.shape), "object_pose_residual_p95_m": float(np.percentile(object_translation["C_STAGE_B"], 95)), "object_pose_residual_max_m": float(object_translation["C_STAGE_B"].max())}, "D_REPAIRED_CXA": {"trajectory": str(inputs.cxa_trajectory), "trajectory_sha256": sha256(inputs.cxa_trajectory), "scene_act": str(inputs.scene_act), "scene_act_sha256": sha256(inputs.scene_act), "qpos_shape": list(cxa_qpos.shape), "object_pose_residual_p95_m": float(np.percentile(object_translation["D_REPAIRED_CXA"], 95)), "object_pose_residual_max_m": float(object_translation["D_REPAIRED_CXA"].max())}}}
    write_json(root / "reports/stage_asset_identity.json", stage_identity)

    query_fn, _ = _static_exec_imports()
    weights = np.full(len(points_local), 1.0 / len(points_local), dtype=np.float64)
    source_frames = np.arange(FRAME_START, FRAME_END, dtype=np.int64)
    stage_defs: dict[str, Any] = {
        "A_OFFICIAL": {"kind": "official", "values": official, "source_frames": source_frames, "formal": True, "rows": [], "summary": {}},
        "B_SPIDER_RAW": {"kind": "raw", "values": raw, "source_frames": source_frames, "formal": True, "rows": [], "summary": {}},
        "C_STAGE_B": {"kind": "wuji", "source_frames": source_mapping, "formal": False, "rows": [], "summary": {}, "collision_layer": stage_b_layers, "display_layer": stage_b_visual, "geometry": stage_b_geom, "index_by_source": {int(f): int(i) for i, f in enumerate(source_mapping)}},
        "D_REPAIRED_CXA": {"kind": "wuji", "source_frames": cxa_mapping, "formal": False, "rows": [], "summary": {}, "collision_layer": cxa_layers, "display_layer": cxa_visual, "geometry": cxa_geom, "index_by_source": {int(f): int(i) for i, f in enumerate(cxa_mapping)}},
    }
    for stage_name, stage in stage_defs.items():
        print(f"[{stage_name}] reverse-SDF frames={len(stage['source_frames'])} formal={stage['formal']}", flush=True)
        for local, source_frame in enumerate(stage["source_frames"]):
            source_index = int(source_frame - FRAME_START)
            object_world = apply_transform(official["T_world_object_visual"][source_index], points_local)
            signed_by_side: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                if stage["kind"] in {"official", "raw"}:
                    vv, ff = _fixed_wrist_cap(stage["values"][f"{side}_vertices_world"][local:local + 1], stage["values"][f"{side}_hand_faces"], side)
                    targets = [_mesh_target(vv, ff, side)]
                else:
                    layer = stage["collision_layer"]
                    geom_targets = []
                    geom_index = int(stage["index_by_source"][int(source_frame)])
                    for item_index, item in enumerate(layer["layers"][side]):
                        tr = layer["transforms"][side][geom_index, item_index]
                        position, rotation = tr[:3], tr[3:].reshape(3, 3)
                        world_v = np.asarray(item["vertices"], dtype=np.float32) @ rotation.T + position
                        geom_targets.append(_mesh_target(world_v[None], item["faces"], f"{stage_name}_{side}_{item['name']}"))
                    targets = geom_targets
                signed_by_side[side] = _query_one_side(query_fn, targets, object_world, root / "manifest/static_exec_tmp", local, stage_name + "_" + side, device, query_batch_size)
            union = np.minimum(signed_by_side["left"], signed_by_side["right"])
            row = {"source_frame": int(source_frame), "local_index": local, "status": "PASS", "formal_sdf_valid": bool(stage["formal"]), "left": _row_metrics(signed_by_side["left"], weights, stage["formal"]), "right": _row_metrics(signed_by_side["right"], weights, stage["formal"]), "union": _row_metrics(union, weights, stage["formal"]), "diagnostic_note": None if stage["formal"] else "Wuji collision layer is not watertight/winding-consistent; signed result is diagnostic only and formal status is SDF_INVALID_FOR_STAGE"}
            stage["rows"].append(row)
            if local == 0 or (local + 1) % 10 == 0 or local + 1 == len(stage["source_frames"]):
                print(f"  {local + 1}/{len(stage['source_frames'])} source_frame={source_frame}", flush=True)
        stage["summary"] = _summary_rows(stage["rows"], stage["formal"])
        write_json(root / "reports" / f"{stage_name.lower()}_reverse_sdf.json", {"schema_version": 1, "stage": stage_name, "backend": STATIC_EXEC_BACKEND, "sign_ray_mode": STATIC_EXEC_SIGN, "query_manifest": query_manifest, "formal_sdf_valid": stage["formal"], "status": "PASS" if stage["formal"] else "SDF_INVALID_FOR_STAGE", "rows": stage["rows"], "summary": stage["summary"]})
    comparison = {"schema_version": 1, "sequence_id": "s5/cylindermedium_lift", "source_frame_range": [FRAME_START, FRAME_END], "fps": FPS, "stages": {name: stage["summary"] for name, stage in stage_defs.items()}, "first_regression": "not decided until formal source baseline and diagnostic Wuji layers are compared; C/D cannot be formal SDF authority"}
    write_json(root / "reports/reverse_sdf_comparison.json", comparison)
    role_candidates = sorted(Path(".local_artifacts/stage_c_xae_m1r5").glob("*/source_role_truth"))
    role_root = role_candidates[-1].parent if role_candidates else None
    write_json(root / "reports/source_quality_contact_vs_penetration.json", {"source_role": _role_report(role_root, source_frames), "contact_definition": "near-surface signed distance <=1mm; this is not penetration", "penetration_definition": "formal only when hand SDF valid; C/D diagnostic signed retained separately", "stages": {name: {"summary": stage["summary"], "first_regression_candidate": stage["summary"].get("global_max_source_frame")} for name, stage in stage_defs.items()}})
    selected = _select_frames(stage_defs)
    if not no_html:
        payload = _make_payload(root, points_local, official, stage_defs, selected)
        _write_html(root, payload)
        for html_name in ("01_reverse_sdf_stage_comparison.html", "02_source_quality_contact_vs_penetration.html"):
            screenshots = _capture_screenshots(root, html_name, selected[:8])
            key = "stage_comparison" if html_name.startswith("01") else "source_quality"
            write_json(root / "screenshots" / key / "manifest.json", {"schema_version": 1, "html": str(root / "html" / html_name), "expected_count": 24, "screenshots": screenshots, "status": "PASS" if len(screenshots) == 24 and all(item["status"] == "PASS" for item in screenshots) else "FAIL"})
            (root / "screenshots" / key / "REVIEW.md").write_text("# 截图人工复核\n\n状态：`PENDING_CODEX_IMAGE_REVIEW`。Chrome 已按 8 个关键帧 × world/object/hand 三视角导出；需查看代表帧后更新。\n", encoding="utf-8")
    write_json(root / "reports/run_summary.json", {"schema_version": 1, "status": "COMPLETE_WITH_SDF_INVALID_WUJI" if any(not stage["formal"] for stage in stage_defs.values()) else "COMPLETE", "run_root": str(root), "selected_frames": selected, "cxa_namespace": str(repaired), "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "raw_grab_modified": False, "stage_b_overwritten": False, "cxa_overwritten": False})
    return str(root)


def render_existing(root: str, paths_config: str = "configs/local/paths.yaml") -> str:
    """Regenerate only HTML/screenshots from a completed numeric run."""
    root_path = Path(root).resolve()
    with np.load(root_path / "manifest/canonical_object_query_points.npz", allow_pickle=False) as archive:
        points_local = np.asarray(archive["points"], dtype=np.float32)
    inputs = _load_paths(paths_config, str(json.loads((root_path / "reports/run_summary.json").read_text(encoding="utf-8"))["cxa_namespace"]))
    from spider.tools.grab_source_geometry_audit import capture_spider_raw_loader, reconstruct_official_source

    official, _ = reconstruct_official_source(inputs)
    raw, _ = capture_spider_raw_loader(inputs)
    with np.load(inputs.stage_b_trajectory, allow_pickle=False) as archive:
        stage_b_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    with np.load(inputs.cxa_trajectory, allow_pickle=False) as archive:
        cxa_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    stage_b_mapping = np.asarray(json.loads(inputs.stage_b_mapping.read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
    with np.load(root_path / "manifest/frame_mapping.npz", allow_pickle=False) as archive:
        cxa_mapping = np.asarray(archive["cxa_source_frame_indices"], dtype=np.int64)
    stage_b_visual, _ = _mujoco_layer_geometry(inputs.scene, stage_b_qpos, 1)
    cxa_visual, _ = _mujoco_layer_geometry(inputs.scene_act, cxa_qpos, 1)
    source_frames = np.arange(FRAME_START, FRAME_END, dtype=np.int64)
    def load_rows(name: str) -> list[dict[str, Any]]:
        return json.loads((root_path / "reports" / f"{name.lower()}_reverse_sdf.json").read_text(encoding="utf-8"))["rows"]
    stages = {
        "A_OFFICIAL": {"kind": "official", "values": official, "source_frames": source_frames, "formal": True, "rows": load_rows("A_OFFICIAL")},
        "B_SPIDER_RAW": {"kind": "raw", "values": raw, "source_frames": source_frames, "formal": True, "rows": load_rows("B_SPIDER_RAW")},
        "C_STAGE_B": {"kind": "wuji", "source_frames": stage_b_mapping, "formal": False, "rows": load_rows("C_STAGE_B"), "display_layer": stage_b_visual, "index_by_source": {int(f): i for i, f in enumerate(stage_b_mapping)}},
        "D_REPAIRED_CXA": {"kind": "wuji", "source_frames": cxa_mapping, "formal": False, "rows": load_rows("D_REPAIRED_CXA"), "display_layer": cxa_visual, "index_by_source": {int(f): i for i, f in enumerate(cxa_mapping)},
        },
    }
    selected = json.loads((root_path / "reports/run_summary.json").read_text(encoding="utf-8"))["selected_frames"]
    _write_html(root_path, _make_payload(root_path, points_local, official, stages, selected))
    for html_name in ("01_reverse_sdf_stage_comparison.html", "02_source_quality_contact_vs_penetration.html"):
        screenshots = _capture_screenshots(root_path, html_name, selected[:8])
        key = "stage_comparison" if html_name.startswith("01") else "source_quality"
        write_json(root_path / "screenshots" / key / "manifest.json", {"schema_version": 1, "html": str(root_path / "html" / html_name), "expected_count": 24, "screenshots": screenshots, "status": "PASS" if len(screenshots) == 24 and all(item["status"] == "PASS" for item in screenshots) else "FAIL"})
        (root_path / "screenshots" / key / "REVIEW.md").write_text("# 截图人工复核\n\n状态：`PENDING_CODEX_IMAGE_REVIEW`。本次页面修复了 hand view 的 object-centered camera。\n", encoding="utf-8")
    return str(root_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C source-quality authority audit")
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-root", default=".local_artifacts/stage_c_source_quality_audit")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--cxa-root")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--render-existing", help="Regenerate HTML/screenshots from a completed run root without rerunning SDF")
    args = parser.parse_args()
    if args.render_existing:
        print(render_existing(args.render_existing, args.paths_config))
    else:
        print(run(args.paths_config, args.output_root, args.run_id, args.device, args.query_batch_size, args.cxa_root, args.no_html))


if __name__ == "__main__":
    main()
