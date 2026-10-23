"""Analysis-only Stage-B Wuji collision-union voxel-SDF comparison.

This command never changes Stage-B qpos, MuJoCo assets, object poses, or
trajectories.  It builds a per-frame collision-union occupancy surface only
for a signed-distance analysis proxy, and keeps that proxy separate from every
simulation, controller, and visual asset contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import trimesh
from plotly.offline.offline import get_plotlyjs

from spider.tools import new_grab_sequence_recheck as source_check
from spider.tools.grab_stage_q0_q4_screening import (
    REF2DEX_ROOT,
    _fixed_wrist_cap,
    _plain,
    _submesh,
    safe_id,
    sha256,
    write_json,
    write_text,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_Q0Q4_ROOT = REPO / ".local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1"
PROXY_LABEL = "ANALYSIS_ONLY_VOXEL_UNION_PROXY"
QUERY_BATCH = 4096
NEAR_M = 0.001
SUSPENSION_M = 0.03
GEOMETRY_SAMPLE_COUNT = 4096


def _import_ref2dex() -> Any:
    if str(REF2DEX_ROOT) not in sys.path:
        sys.path.insert(0, str(REF2DEX_ROOT))
    from dataset.tools.reverse_hand_meshsdf_statistics import query_targets_with_meshsdf_layers
    return query_targets_with_meshsdf_layers


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def _face_area_weights(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign deterministic surface-area weights to the frozen object queries."""
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(mesh)
    primitive = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, dtype=np.float32)))["primitive_ids"].numpy().astype(np.int64)
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    counts = np.bincount(primitive, minlength=len(areas))
    weights = areas[primitive] / np.maximum(counts[primitive], 1)
    weights *= areas.sum() / weights.sum()
    return weights.astype(np.float64), primitive


def _collision_specs(model: mujoco.MjModel) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    result: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {"left": [], "right": []}
    for geom in range(model.ngeom):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH) or int(model.geom_group[geom]) != 2:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        side = "left" if body.startswith("l_") else "right" if body.startswith("r_") else None
        if side is None:
            continue
        mesh_id = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        result[side].append((geom, np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64), np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)))
    if not all(result.values()):
        raise RuntimeError("Stage-B scene has no complete collision mesh set for both Wuji hands")
    return result


def _collision_union(specs: list[tuple[int, np.ndarray, np.ndarray]], data: mujoco.MjData) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []; faces: list[np.ndarray] = []; offset = 0
    for geom, local_vertices, local_faces in specs:
        world = local_vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        vertices.append(world); faces.append(local_faces + offset); offset += len(world)
    return trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)


def _proxy_from_union(raw: trimesh.Trimesh, voxel_m: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Rasterise the whole hand collision union, fill occupancy, then MC it."""
    voxel = raw.voxelized(pitch=voxel_m, method="subdivide").fill()
    # Do not use VoxelGrid.marching_cubes here: it can leave an open sheet
    # when the filled union reaches the grid AABB.  The explicit operation
    # pads the occupancy by one empty cell before extracting the surface.
    # ``voxel.transform`` below supplies the voxel pitch and origin.
    proxy = trimesh.voxel.ops.matrix_to_marching_cubes(voxel.matrix)
    proxy.apply_transform(voxel.transform)
    # trimesh 4 exposes these as face masks (the older mutating methods were
    # removed); keeping this explicit also makes proxy sanitation auditable.
    proxy.update_faces(proxy.unique_faces())
    proxy.update_faces(proxy.nondegenerate_faces())
    proxy.remove_unreferenced_vertices()
    proxy.fix_normals()
    # A rare marching-cubes saddle can emit an isolated two-triangle,
    # zero-volume sheet.  It is not an occupied component or a repaired gap;
    # discard only that mathematically degenerate fragment, and fail closed
    # for any larger non-watertight component.
    parts = list(proxy.split(only_watertight=False))
    degenerate = [part for part in parts if not part.is_watertight and len(part.faces) <= 4]
    invalid = [part for part in parts if not part.is_watertight and len(part.faces) > 4]
    if invalid:
        raise RuntimeError(f"voxel marching-cubes produced non-degenerate open component(s): {[len(part.faces) for part in invalid]}")
    if degenerate:
        proxy = trimesh.util.concatenate([part for part in parts if part.is_watertight])
        proxy.remove_unreferenced_vertices(); proxy.fix_normals()
    raw_edges = raw.edges_sorted; unique, counts = np.unique(raw_edges, axis=0, return_counts=True)
    status = {
        "label": PROXY_LABEL, "voxel_size_m": voxel_m,
        "raw_vertices": int(len(raw.vertices)), "raw_faces": int(len(raw.faces)),
        "raw_boundary_edges": int(np.count_nonzero(counts == 1)), "raw_nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "raw_components": int(len(raw.split(only_watertight=False))),
        "occupancy_voxel_count": int(len(voxel.sparse_indices)), "occupancy_volume_m3": float(len(voxel.sparse_indices) * voxel_m ** 3),
        "removed_degenerate_mc_components": int(len(degenerate)),
        "proxy_vertices": int(len(proxy.vertices)), "proxy_faces": int(len(proxy.faces)), "proxy_components": int(len(proxy.split(only_watertight=False))),
        "proxy_volume_m3": float(abs(proxy.volume)), "proxy_watertight": bool(proxy.is_watertight), "proxy_winding_consistent": bool(proxy.is_winding_consistent),
    }
    status["proxy_minus_occupancy_volume_m3"] = status["proxy_volume_m3"] - status["occupancy_volume_m3"]
    status["proxy_vs_occupancy_volume_ratio"] = status["proxy_volume_m3"] / status["occupancy_volume_m3"] if status["occupancy_volume_m3"] else None
    return proxy, status


def _distance_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    triangle = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32)))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(triangle)
    return scene


def _surface_error(raw: trimesh.Trimesh, proxy: trimesh.Trimesh, voxel_m: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw_points, _ = trimesh.sample.sample_surface(raw, GEOMETRY_SAMPLE_COUNT, seed=rng)
    proxy_points, _ = trimesh.sample.sample_surface(proxy, GEOMETRY_SAMPLE_COUNT, seed=rng)
    to_proxy = _distance_scene(proxy).compute_distance(o3d.core.Tensor(raw_points.astype(np.float32))).numpy().astype(float)
    to_raw = _distance_scene(raw).compute_distance(o3d.core.Tensor(proxy_points.astype(np.float32))).numpy().astype(float)
    both = np.concatenate((to_proxy, to_raw))
    return {
        "sample_count_per_direction": GEOMETRY_SAMPLE_COUNT,
        "raw_to_proxy_mean_m": float(to_proxy.mean()), "proxy_to_raw_mean_m": float(to_raw.mean()),
        "surface_distance_p95_m": float(np.percentile(both, 95)), "hausdorff_sampled_m": float(both.max()),
        "chamfer_mean_squared_m2": float(np.mean(to_proxy ** 2) + np.mean(to_raw ** 2)),
        "chamfer_rmse_m": float(np.sqrt(.5 * (np.mean(to_proxy ** 2) + np.mean(to_raw ** 2)))),
        "p95_gate_m": 2 * voxel_m, "p95_within_two_voxels": bool(np.percentile(both, 95) <= 2 * voxel_m),
    }


def _source_targets(values: dict[str, np.ndarray], frame: int, label: str) -> list[dict[str, Any]]:
    targets = []
    for side in ("left", "right"):
        vertices, faces = _fixed_wrist_cap(np.asarray(values[f"{side}_vertices_world"])[frame:frame + 1], np.asarray(values[f"{side}_hand_faces"]), side)
        targets.append({"label": side, "source": label, "vertices": vertices, "faces": faces})
    return targets


def _proxy_targets(proxy: dict[str, trimesh.Trimesh]) -> list[dict[str, Any]]:
    return [{"label": side, "source": PROXY_LABEL, "vertices": np.asarray(mesh.vertices, dtype=np.float32)[None], "faces": np.asarray(mesh.faces, dtype=np.int32)} for side, mesh in proxy.items()]


def _full_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    """Preserve every C-stage mesh triangle for the inspection viewer."""
    return {"vertices": np.asarray(vertices, dtype=np.float32), "faces": np.asarray(faces, dtype=np.int32)}


def _query_layers(targets: list[dict[str, Any]], world_points: np.ndarray, work: Path, frame: int, cache: str, device: str) -> dict[str, np.ndarray]:
    query = _import_ref2dex()
    values = query(np=np, targets=targets, frame_index=0, points=world_points, device=device, check_manifold=False, sign_ray_mode="xyz_majority", work_root=work, unit_scale_to_mm=1000.0, cache_stem=cache, query_batch_size=QUERY_BATCH, keep_frame_objs=False)
    return {"left": np.asarray(values["by_label"]["left"]["signed_distance_mm"], dtype=np.float64), "right": np.asarray(values["by_label"]["right"]["signed_distance_mm"], dtype=np.float64), "both": np.asarray(values["union_signed_distance_mm"], dtype=np.float64)}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values): return 0.0
    order = np.argsort(values); v, w = values[order], weights[order]; cdf = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cdf, q, side="left")])


def _frame_metrics(signed_mm: np.ndarray, weights: np.ndarray, points_local: np.ndarray, patch: np.ndarray) -> dict[str, Any]:
    signed = np.asarray(signed_mm, dtype=float); negative = signed < 0.0; depth = np.maximum(0.0, -signed); near = np.abs(signed) <= NEAR_M * 1000.0
    neg = signed[negative]; neg_weights = weights[negative]
    centroid = np.average(points_local[near], axis=0, weights=weights[near]).tolist() if np.any(near) else None
    patch_weight = weights[patch].sum()
    return {
        "negative_signed_distance_mean_mm": float(neg.mean()) if len(neg) else 0.0,
        "negative_signed_distance_variance_mm2": float(neg.var()) if len(neg) else 0.0,
        "min_signed_distance_mm": float(signed.min()), "max_penetration_depth_mm": float(depth.max()),
        "negative_point_ratio": float(negative.mean()), "negative_point_count": int(negative.sum()),
        "p95_penetration_depth_mm": float(np.percentile(depth[negative], 95)) if len(neg) else 0.0,
        "area_weighted_negative_point_ratio": float(weights[negative].sum() / weights.sum()),
        "area_weighted_mean_penetration_depth_mm": float(np.sum(neg_weights * depth[negative]) / np.sum(neg_weights)) if len(neg) else 0.0,
        "near_surface_contact_point_ratio": float(near.mean()), "hand_object_min_surface_distance_mm": float(np.abs(signed).min()),
        "source_contact_footprint_coverage": float(weights[patch & near].sum() / patch_weight) if patch_weight else None,
        "contact_centroid_object_local_m": centroid, "obvious_suspension": bool(np.abs(signed).min() > SUSPENSION_M * 1000.0),
    }


@dataclass
class Aggregate:
    rows: list[dict[str, Any]] = field(default_factory=list)
    negative_depths: list[np.ndarray] = field(default_factory=list)
    negative_weights: list[np.ndarray] = field(default_factory=list)
    point_count: int = 0
    negative_count: int = 0
    total_area: float = 0.0
    negative_area: float = 0.0

    def add(self, row: dict[str, Any], signed: np.ndarray, weights: np.ndarray) -> None:
        self.rows.append(row); negative = signed < 0.0; self.point_count += len(signed); self.negative_count += int(negative.sum()); self.total_area += float(weights.sum()); self.negative_area += float(weights[negative].sum())
        if np.any(negative): self.negative_depths.append(-signed[negative]); self.negative_weights.append(weights[negative])

    def result(self) -> dict[str, Any]:
        depths = np.concatenate(self.negative_depths) if self.negative_depths else np.empty(0)
        weights = np.concatenate(self.negative_weights) if self.negative_weights else np.empty(0)
        frame_keys = ("negative_signed_distance_mean_mm", "negative_signed_distance_variance_mm2", "max_penetration_depth_mm", "negative_point_ratio", "area_weighted_negative_point_ratio", "area_weighted_mean_penetration_depth_mm", "near_surface_contact_point_ratio", "hand_object_min_surface_distance_mm", "source_contact_footprint_coverage")
        per_frame = {key: {"mean": float(np.nanmean([row[key] if row[key] is not None else np.nan for row in self.rows])), "p50": float(np.nanpercentile([row[key] if row[key] is not None else np.nan for row in self.rows], 50)), "p90": float(np.nanpercentile([row[key] if row[key] is not None else np.nan for row in self.rows], 90)), "p95": float(np.nanpercentile([row[key] if row[key] is not None else np.nan for row in self.rows], 95)), "p99": float(np.nanpercentile([row[key] if row[key] is not None else np.nan for row in self.rows], 99)), "max": float(np.nanmax([row[key] if row[key] is not None else np.nan for row in self.rows]))} for key in frame_keys}
        worst = max(self.rows, key=lambda row: row["max_penetration_depth_mm"])
        return {"frame_count": len(self.rows), "query_point_count": self.point_count, "negative_point_count": self.negative_count, "negative_point_ratio": self.negative_count / self.point_count if self.point_count else 0.0, "area_weighted_negative_point_ratio": self.negative_area / self.total_area if self.total_area else 0.0, "negative_signed_distance_mean_mm": -float(depths.mean()) if len(depths) else 0.0, "negative_signed_distance_variance_mm2": float(depths.var()) if len(depths) else 0.0, "min_signed_distance_mm": -float(depths.max()) if len(depths) else 0.0, "max_penetration_depth_mm": float(depths.max()) if len(depths) else 0.0, "p50_penetration_depth_mm": float(np.percentile(depths, 50)) if len(depths) else 0.0, "p90_penetration_depth_mm": float(np.percentile(depths, 90)) if len(depths) else 0.0, "p95_penetration_depth_mm": float(np.percentile(depths, 95)) if len(depths) else 0.0, "p99_penetration_depth_mm": float(np.percentile(depths, 99)) if len(depths) else 0.0, "area_weighted_mean_penetration_depth_mm": float(np.sum(weights * depths) / np.sum(weights)) if len(depths) else 0.0, "obvious_suspension_frame_ratio": float(np.mean([row["obvious_suspension"] for row in self.rows])), "max_penetration_frame": int(worst["source_frame"]), "per_frame": per_frame}


def _stage_files(run_root: Path, label: str) -> tuple[Path, np.ndarray, np.ndarray, Path]:
    sequence = "s7__stanfordbunny_pass_1" if label == "primary" else "s10__spherelarge_pass_1"
    root = run_root / f"q4_{label}_retarget/workspace/processed/grab/wuji_hand2_beta1/bimanual" / sequence
    qpos = np.load(root / "0/trajectory_kinematic.npz", allow_pickle=False)["qpos"]
    mapping = np.asarray(json.loads((root / "0/source_mapping.json").read_text(encoding="utf-8"))["source_frame_indices"], dtype=np.int64)
    return root / "scene.xml", qpos, mapping, root


def _representative_indices(count: int) -> set[int]:
    return set(np.linspace(0, count - 1, min(count, 16), dtype=int).tolist())


def _paired_stage_b_vs_official(official_rows: list[dict[str, Any]], proxy_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Frame-aligned deltas, including the contact-separation guard."""
    baseline = {row["source_frame"]: row for row in official_rows}
    result: dict[str, dict[str, Any]] = {}
    for side, rows in proxy_rows.items():
        paired = [(baseline[row["source_frame"]]["sides"][side], row) for row in rows]
        ratio_delta = np.asarray([candidate["negative_point_ratio"] - source["negative_point_ratio"] for source, candidate in paired])
        depth_delta = np.asarray([candidate["area_weighted_mean_penetration_depth_mm"] - source["area_weighted_mean_penetration_depth_mm"] for source, candidate in paired])
        max_depth_delta = np.asarray([candidate["max_penetration_depth_mm"] - source["max_penetration_depth_mm"] for source, candidate in paired])
        centroid = []
        for source, candidate in paired:
            a, b = source["contact_centroid_object_local_m"], candidate["contact_centroid_object_local_m"]
            if a is not None and b is not None:
                centroid.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 1000.0))
        worst_ratio = int(np.argmax(ratio_delta)); worst_depth = int(np.argmax(depth_delta)); worst_max_depth = int(np.argmax(max_depth_delta))
        result[side] = {
            "paired_frame_count": len(paired),
            "max_negative_ratio_deterioration": float(ratio_delta[worst_ratio]),
            "max_negative_ratio_deterioration_frame": int(rows[worst_ratio]["source_frame"]),
            "max_weighted_depth_deterioration_mm": float(depth_delta[worst_depth]),
            "max_weighted_depth_deterioration_frame": int(rows[worst_depth]["source_frame"]),
            "max_penetration_depth_deterioration_mm": float(max_depth_delta[worst_max_depth]),
            "max_penetration_depth_deterioration_frame": int(rows[worst_max_depth]["source_frame"]),
            "contact_centroid_drift_mean_mm": float(np.mean(centroid)) if centroid else None,
            "contact_centroid_drift_p95_mm": float(np.percentile(centroid, 95)) if centroid else None,
            "contact_centroid_compared_frame_count": len(centroid),
        }
    return result


def _run_sequence(run_root: Path, output: Path, label: str, voxel_sizes: list[float], device: str, max_frames: int | None = None) -> dict[str, Any]:
    selection = json.loads((run_root / "q2_selection" / f"{label}_selection.json").read_text(encoding="utf-8"))["selection"]
    sequence_id = str(selection["source_id"]); candidate_root = next((run_root / "q1_candidates").glob(f"sequence_*_{safe_id(sequence_id)}"))
    with np.load(candidate_root / "query_points.npz", allow_pickle=False) as archive: points_local = np.asarray(archive["points"], dtype=np.float32)
    official, official_chain = source_check.reconstruct_official("configs/local/paths.yaml", sequence_id)
    spider, spider_chain = source_check.capture_spider("configs/local/paths.yaml", sequence_id, official)
    scene_path, qpos, mapping, stage_root = _stage_files(run_root, label)
    if max_frames is not None:
        qpos, mapping = qpos[:max_frames], mapping[:max_frames]
    weights, primitive = _face_area_weights(official["object_asset_vertices"], official["object_faces"], points_local)
    patch_faces = set(json.loads((run_root / f"q3_{label}_authority/semantic_patch.json").read_text(encoding="utf-8"))["triangle_ids"])
    patch = np.isin(primitive, np.asarray(sorted(patch_faces), dtype=np.int64))
    source_lookup = {int(value): index for index, value in enumerate(official["source_frame_indices"])}
    spider_lookup = {int(value): index for index, value in enumerate(spider["source_frame_indices"])}
    model = mujoco.MjModel.from_xml_path(str(scene_path)); data = mujoco.MjData(model); specs = _collision_specs(model)
    target_root = output / label; target_root.mkdir(parents=True, exist_ok=False)
    work = target_root / "static_exec_work"; reps = _representative_indices(len(qpos))
    source_records: dict[str, list[dict[str, Any]]] = {"official": [], "spider": []}; source_aggregate = {stage: {side: Aggregate() for side in ("left", "right", "both")} for stage in source_records}
    proxy_records: dict[str, dict[str, list[dict[str, Any]]]] = {str(size): {side: [] for side in ("left", "right", "both")} for size in voxel_sizes}
    proxy_aggregate = {str(size): {side: Aggregate() for side in ("left", "right", "both")} for size in voxel_sizes}
    geometry: dict[str, list[dict[str, Any]]] = {str(size): [] for size in voxel_sizes}; html_frames: list[dict[str, Any]] = []
    for stage_index, source_frame in enumerate(mapping):
        if int(source_frame) not in source_lookup or int(source_frame) not in spider_lookup: raise RuntimeError(f"source mapping frame missing in source captures: {source_frame}")
        oi, si = source_lookup[int(source_frame)], spider_lookup[int(source_frame)]
        world_points = _transform_points(official["T_world_object"][oi], points_local)
        display_signed: dict[str, np.ndarray] = {}
        for name, values, index in (("official", official, oi), ("spider", spider, si)):
            signed = _query_layers(_source_targets(values, index, name), world_points, work, int(source_frame), f"{name}_{stage_index}", device)
            display_signed[name] = signed["both"]
            row = {"source_frame": int(source_frame), "stage_index": stage_index, "sides": {}}
            for side in ("left", "right", "both"):
                metrics = _frame_metrics(signed[side], weights, points_local, patch); row["sides"][side] = metrics; source_aggregate[name][side].add({"source_frame": int(source_frame), **metrics}, signed[side], weights)
            source_records[name].append(row)
        data.qpos[:] = qpos[stage_index]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        raw = {side: _collision_union(specs[side], data) for side in ("left", "right")}
        for voxel_m in voxel_sizes:
            key = str(voxel_m); proxies: dict[str, trimesh.Trimesh] = {}; status: dict[str, Any] = {}
            for side in ("left", "right"):
                proxies[side], status[side] = _proxy_from_union(raw[side], voxel_m)
                if not (status[side]["proxy_watertight"] and status[side]["proxy_winding_consistent"]): raise RuntimeError(f"{PROXY_LABEL} invalid at {label}/{stage_index}/{side}/{voxel_m}: {status[side]}")
                if stage_index in reps: geometry[key].append({"source_frame": int(source_frame), "stage_index": stage_index, "side": side, "proxy": status[side], "surface_error": _surface_error(raw[side], proxies[side], voxel_m, 20260804 + stage_index)})
            signed = _query_layers(_proxy_targets(proxies), world_points, work, int(source_frame), f"proxy_{voxel_m}_{stage_index}", device)
            for side in ("left", "right", "both"):
                metrics = _frame_metrics(signed[side], weights, points_local, patch); row = {"source_frame": int(source_frame), "stage_index": stage_index, **metrics}; proxy_records[key][side].append(row); proxy_aggregate[key][side].add(row, signed[side], weights)
            if voxel_m == min(voxel_sizes) and stage_index in reps:
                display = np.linspace(0, len(points_local) - 1, min(1600, len(points_local)), dtype=int)
                html_frames.append({
                    "source_frame": int(source_frame), "stage_index": stage_index,
                    # The browser multiplies this as a flat row-major 3x4
                    # affine prefix; serialising a nested 4x4 would coerce
                    # JavaScript arrays to NaN and hide object/query layers.
                    "object_T": np.asarray(official["T_world_object"][oi], dtype=float).reshape(-1).tolist(),
                    "official": {side: _submesh(official[f"{side}_vertices_world"][oi], official[f"{side}_hand_faces"], 350) for side in ("left", "right")},
                    "spider": {side: _submesh(spider[f"{side}_vertices_world"][si], spider[f"{side}_hand_faces"], 350) for side in ("left", "right")},
                    "collision": {side: _submesh(raw[side].vertices, raw[side].faces, 800) for side in ("left", "right")},
                    "proxy": {side: _submesh(proxies[side].vertices, proxies[side].faces, 800) for side in ("left", "right")},
                    "query_local": points_local[display],
                    "signed_mm": {"official": display_signed["official"][display], "spider": display_signed["spider"][display], "proxy": signed["both"][display], "collision": signed["both"][display]},
                    "metrics": {"official": source_records["official"][-1]["sides"]["both"], "spider": source_records["spider"][-1]["sides"]["both"], "proxy": proxy_records[key]["both"][-1], "collision": {"distance_query": "proxy signed distance only; raw union is not used for SDF"}},
                })
        if stage_index % 5 == 0 or stage_index + 1 == len(qpos): print(f"{label}: frame {stage_index + 1}/{len(qpos)}", flush=True)
    result = {"schema_version": 1, "sequence": sequence_id, "stage_b_proxy": PROXY_LABEL, "selection": {"run_root": str(run_root), "stage_scene_sha256": sha256(scene_path), "qpos_sha256": sha256(stage_root / "0/trajectory_kinematic.npz"), "object_query_sha256": sha256(candidate_root / "query_points.npz"), "source_mapping_sha256": sha256(stage_root / "0/source_mapping.json")}, "source_lineage": {"official": official_chain, "spider": spider_chain}, "query_contract": {"same_object_asset": str(selection["object_mesh"]), "same_object_local_query_points": int(len(points_local)), "sampling": "reuses frozen Q1 triangle-area sample followed by 1mm voxel first-unique", "area_weights": "object triangle area divided among frozen query points whose nearest object triangle is that face; normalized to total mesh area", "frame_mapping": mapping.tolist(), "object_pose": "frozen official GRAB object pose for all three stages"}, "official": {side: source_aggregate["official"][side].result() for side in ("left", "right", "both")}, "spider": {side: source_aggregate["spider"][side].result() for side in ("left", "right", "both")}, "proxy": {key: {side: proxy_aggregate[key][side].result() for side in ("left", "right", "both")} for key in proxy_aggregate}, "geometry_validation": geometry, "per_frame": {"official": source_records["official"], "spider": source_records["spider"], "proxy": proxy_records}}
    for key, records in proxy_records.items():
        paired = _paired_stage_b_vs_official(source_records["official"], records)
        for side in ("left", "right", "both"):
            result["proxy"][key][side]["vs_official"] = paired[side]
    write_json(target_root / "sequence_result.json", result); write_json(target_root / "html_payload.json", {"frames": html_frames, "object": _submesh(official["object_asset_vertices"], official["object_faces"], 800), "sequence": sequence_id, "voxel_main_m": min(voxel_sizes)})
    return result


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=_plain, separators=(",", ":"))
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Stage-B voxel proxy 三阶段对比</title><script>__PLOTLY__</script><style>body{margin:0;background:#101820;color:#eef6f7;font-family:system-ui}#bar{padding:12px;background:#173042}#scene{height:76vh}select{margin:3px}pre{white-space:pre-wrap;padding:8px}</style><body><div id='bar'><b>三阶段 object-local 穿模对比：official / Spider / Stage-B collision union / voxel-MC proxy</b><br>帧 <select id='frame'></select>　阶段 <select id='stage'><option value='official'>A official source</option><option value='spider'>B Spider loaded source</option><option value='collision'>C 原始 collision union（完整 mesh，只显示）</option><option value='proxy'>C voxel-MC watertight proxy（完整 mesh）</option></select><p>红/橙 query points 为 selected stage 的 negative signed depth。选择 C 时会按帧加载全部三角面；proxy 仅供 analysis-only SDF，不替换真实 Stage-B collision/visual mesh。</p></div><div id='scene'></div><pre id='info'></pre><script>
const D=__PAYLOAD__,$=x=>document.getElementById(x),I=[1,0,0,0,0,1,0,0,0,0,1,0];
function mt(T,p){return p.map(q=>[T[0]*q[0]+T[1]*q[1]+T[2]*q[2]+T[3],T[4]*q[0]+T[5]*q[1]+T[6]*q[2]+T[7],T[8]*q[0]+T[9]*q[1]+T[10]*q[2]+T[11]])}
function m(n,s,c,o,T){let v=mt(T,s.vertices);return{type:'mesh3d',name:n,x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:s.faces.map(x=>x[0]),j:s.faces.map(x=>x[1]),k:s.faces.map(x=>x[2]),color:c,opacity:o,flatshading:true}}
function selectedFullC(f){
  const bank=window.__C_FULL_MESH||{};
  if(bank[f.stage_index]) return bank[f.stage_index];
  if(!f.full_c_mesh_script) return null;
  window.__C_PENDING=window.__C_PENDING||{};
  if(!window.__C_PENDING[f.stage_index]){
    window.__C_PENDING[f.stage_index]=true;
    const e=document.createElement('script'); e.src=f.full_c_mesh_script;
    e.onload=()=>{delete window.__C_PENDING[f.stage_index];draw()};
    e.onerror=()=>{$('info').textContent='完整 C mesh 加载失败：'+f.full_c_mesh_script;delete window.__C_PENDING[f.stage_index]};
    document.head.appendChild(e);
  }
  return 'loading';
}
D.frames.forEach((f,i)=>$('frame').add(new Option(f.source_frame,i)));
function draw(){
  let f=D.frames[+$('frame').value],s=$('stage').value,T=f.object_T.slice(0,12),full=null;
  if(s==='collision'||s==='proxy'){full=selectedFullC(f);if(full==='loading'){$('info').textContent='正在加载此帧完整 C mesh…';return}}
  let tr=[m('object',D.object,'#f4a261',.35,T)],layer=full?full[s]:f[s];
  if(s==='official'||s==='spider')for(let h of ['left','right'])tr.push(m(s+' '+h+' source hand',layer[h],h==='left'?'#48cae4':'#f28482',.7,I));
  else for(let h of ['left','right'])tr.push(m(s+' '+h+' 完整 C mesh',layer[h],h==='left'?'#06d6a0':'#ef476f',.55,I));
  let p=mt(T,f.query_local),z=f.signed_mm[s];
  tr.push({type:'scatter3d',mode:'markers',name:'frozen Q1 signed distance (mm)',x:p.map(x=>x[0]),y:p.map(x=>x[1]),z:p.map(x=>x[2]),marker:{size:2.4,color:z,colorscale:[[0,'#d00000'],[.5,'#fff'],[1,'#3a86ff']],cmin:-10,cmax:10,colorbar:{title:'mm'}}});
  Plotly.react('scene',tr,{paper_bgcolor:'#101820',font:{color:'#eef6f7'},scene:{aspectmode:'data'},margin:{l:0,r:0,t:15,b:0}});
  $('info').textContent=JSON.stringify({source_frame:f.source_frame,stage_index:f.stage_index,stage:s,metrics:f.metrics[s],proxy:'__PROXY__',c_mesh_rendering:full?'FULL_TRIANGLES_NO_DECIMATION':'preview'},null,2)
}
$('frame').onchange=draw;$('stage').onchange=draw;draw();</script></body></html>"""
    return page.replace("__PLOTLY__", get_plotlyjs()).replace("__PAYLOAD__", data).replace("__PROXY__", PROXY_LABEL)


def _comparison_table(results: dict[str, Any], voxel_sizes: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows: list[dict[str, Any]] = []; sensitivity: list[dict[str, Any]] = []
    for label, result in results.items():
        for side in ("left", "right", "both"):
            for stage, values in (("A official", result["official"][side]), ("B Spider", result["spider"][side]), (f"C proxy {min(voxel_sizes) * 1000:g} mm", result["proxy"][str(min(voxel_sizes))][side])):
                row = {"selection": label, "sequence": result["sequence"], "hand_scope": side, "stage": stage, **{key: values[key] for key in ("negative_signed_distance_mean_mm", "negative_signed_distance_variance_mm2", "min_signed_distance_mm", "max_penetration_depth_mm", "negative_point_ratio", "negative_point_count", "p95_penetration_depth_mm", "area_weighted_negative_point_ratio", "area_weighted_mean_penetration_depth_mm", "obvious_suspension_frame_ratio", "max_penetration_frame")}}
                # Contact summaries are deliberately frame aggregates rather
                # than a pooled signed-distance statistic; retain their mean
                # explicitly instead of pretending they are top-level values.
                row.update({f"{key}_frame_mean": values["per_frame"][key]["mean"] for key in ("near_surface_contact_point_ratio", "hand_object_min_surface_distance_mm", "source_contact_footprint_coverage")})
                if "vs_official" in values:
                    row.update(values["vs_official"])
                primary_rows.append(row)
            for size in voxel_sizes:
                values = result["proxy"][str(size)][side]; source = result["official"][side]
                sensitivity.append({"selection": label, "sequence": result["sequence"], "hand_scope": side, "voxel_size_mm": size * 1000, "negative_point_ratio": values["negative_point_ratio"], "max_penetration_depth_mm": values["max_penetration_depth_mm"], "area_weighted_mean_penetration_depth_mm": values["area_weighted_mean_penetration_depth_mm"], "delta_negative_ratio_vs_official": values["negative_point_ratio"] - source["negative_point_ratio"], "delta_weighted_depth_vs_official_mm": values["area_weighted_mean_penetration_depth_mm"] - source["area_weighted_mean_penetration_depth_mm"]})
    return primary_rows, sensitivity


def _conclusion(result: dict[str, Any], voxel_sizes: list[float]) -> dict[str, Any]:
    """Classify only direction-consistent proxy results, with a contact guard."""
    both = [result["proxy"][str(size)]["both"] for size in voxel_sizes]
    official = result["official"]["both"]
    spider = result["spider"]["both"]
    geometry = [entry for rows in result["geometry_validation"].values() for entry in rows]
    geometry_ok = bool(geometry) and all(entry["proxy"]["proxy_watertight"] and entry["proxy"]["proxy_winding_consistent"] and entry["surface_error"]["p95_within_two_voxels"] for entry in geometry)
    ratio_delta = [item["negative_point_ratio"] - official["negative_point_ratio"] for item in both]
    depth_delta = [item["area_weighted_mean_penetration_depth_mm"] - official["area_weighted_mean_penetration_depth_mm"] for item in both]
    ratio_delta_vs_spider = [item["negative_point_ratio"] - spider["negative_point_ratio"] for item in both]
    depth_delta_vs_spider = [item["area_weighted_mean_penetration_depth_mm"] - spider["area_weighted_mean_penetration_depth_mm"] for item in both]
    coverage = [item["per_frame"]["source_contact_footprint_coverage"]["mean"] for item in both]
    baseline_coverage = official["per_frame"]["source_contact_footprint_coverage"]["mean"]
    loader_delta = {key: float(spider[key] - official[key]) for key in ("negative_point_ratio", "max_penetration_depth_mm", "area_weighted_mean_penetration_depth_mm")}
    loader_consistent = abs(loader_delta["negative_point_ratio"]) <= 1e-6 and abs(loader_delta["max_penetration_depth_mm"]) <= .01 and abs(loader_delta["area_weighted_mean_penetration_depth_mm"]) <= .01
    separated = any(item["obvious_suspension_frame_ratio"] > official["obvious_suspension_frame_ratio"] or (np.isfinite(value) and np.isfinite(baseline_coverage) and value < .5 * baseline_coverage) for item, value in zip(both, coverage))
    all_deltas = ratio_delta + depth_delta + ratio_delta_vs_spider + depth_delta_vs_spider
    improves = all(value <= 0.0 for value in all_deltas) and any(value < 0.0 for value in all_deltas)
    worsens = all(value >= 0.0 for value in all_deltas) and any(value > 0.0 for value in all_deltas)
    if not geometry_ok:
        status = "INCONCLUSIVE_PROXY_SENSITIVE"
    elif separated and improves:
        status = "PENETRATION_REDUCED_BY_CONTACT_SEPARATION"
    elif improves:
        status = "DIRECTION_CONSISTENT_PENETRATION_IMPROVEMENT"
    elif worsens:
        status = "DIRECTION_CONSISTENT_PENETRATION_WORSENING"
    else:
        status = "INCONCLUSIVE_PROXY_SENSITIVE"
    return {"status": status, "geometry_gate_passed": geometry_ok, "loader_A_B_consistent": loader_consistent, "loader_B_minus_A": loader_delta, "resolution_mm": [size * 1000 for size in voxel_sizes], "delta_negative_ratio_vs_official": ratio_delta, "delta_weighted_mean_penetration_depth_mm_vs_official": depth_delta, "delta_negative_ratio_vs_spider": ratio_delta_vs_spider, "delta_weighted_mean_penetration_depth_mm_vs_spider": depth_delta_vs_spider, "proxy_patch_coverage_mean": coverage, "official_patch_coverage_mean": baseline_coverage, "rule": "Only the frozen-Q1 direction-consistent 1/2mm result against both official A and Spider-loaded B is classified. A reduced negative distance is rejected as quality improvement when the contact coverage/suspension guard indicates separation."}


def _write_summary_artifacts(root: Path, results: dict[str, Any], voxel_sizes: list[float]) -> None:
    """Write the comparison deliverables from complete per-trajectory results."""
    primary, sensitivity = _comparison_table(results, voxel_sizes)
    geometry = [{"selection": label, "voxel_size_mm": float(key) * 1000, **row} for label, result in results.items() for key, rows in result["geometry_validation"].items() for row in rows]
    conclusions = {label: _conclusion(result, voxel_sizes) for label, result in results.items()}
    write_json(root / "three_stage_main_table.json", primary); write_json(root / "voxel_sensitivity_table.json", sensitivity); write_json(root / "proxy_geometry_error_table.json", geometry); write_json(root / "conclusion.json", conclusions)
    lines = ["# Stage-B voxel-union proxy 三阶段对比", "", "C 是 `ANALYSIS_ONLY_VOXEL_UNION_PROXY`，绝不替换真实 collision mesh。每个 selection 同时列出左手、右手与双手 union aggregate。", "", "|selection|hand scope|stage|negative mean mm|negative var mm2|min signed mm|max penetration mm|negative ratio|negative count|P95 depth mm|weighted negative ratio|weighted mean depth mm|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['selection']}|{r['hand_scope']}|{r['stage']}|{r['negative_signed_distance_mean_mm']:.6f}|{r['negative_signed_distance_variance_mm2']:.6f}|{r['min_signed_distance_mm']:.6f}|{r['max_penetration_depth_mm']:.6f}|{r['negative_point_ratio']:.8f}|{r['negative_point_count']}|{r['p95_penetration_depth_mm']:.6f}|{r['area_weighted_negative_point_ratio']:.8f}|{r['area_weighted_mean_penetration_depth_mm']:.6f}|" for r in primary]
    write_text(root / "THREE_STAGE_MAIN_TABLE.md", "\n".join(lines) + "\n")
    for label in results:
        payload = json.loads((root / label / "html_payload.json").read_text(encoding="utf-8")); write_text(root / f"{label}_three_stage_proxy.html", _html(payload))
    links = "".join(f"<li><a href='{label}_three_stage_proxy.html'>{label} 三阶段可视化</a></li>" for label in results)
    links += "<li><a href='THREE_STAGE_MAIN_TABLE.md'>三阶段主表（Markdown）</a></li><li><a href='three_stage_main_table.json'>三阶段主表（JSON）</a></li><li><a href='voxel_sensitivity_table.json'>体素敏感性表</a></li><li><a href='proxy_geometry_error_table.json'>proxy 几何误差表</a></li><li><a href='conclusion.json'>结论与门槛</a></li>"
    write_text(root / "index.html", f"<!doctype html><meta charset='utf-8'><h1>Stage-B voxel proxy analysis-only</h1><p>C 为 <code>{PROXY_LABEL}</code>，不替换真实 Stage-B collision/visual mesh。</p><ul>{links}</ul>")


def refresh_geometry_validation(run_root: Path, shard_root: Path, label: str, voxel_sizes: list[float], reference_voxel_m: float = .0005) -> Path:
    """Refresh only representative-frame proxy geometry evidence.

    The direct concatenated collision meshes include overlapping-link interior
    triangles, which are not part of an occupancy *union* boundary.  A finer
    independently reconstructed occupancy-union surface is therefore used as
    the validation reference.  This function writes only derived geometry
    evidence in an already-complete shard result; it never changes any scene,
    qpos, object pose, query point, or SDF penetration statistic.
    """
    result_path = shard_root / label / "sequence_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"complete shard result required: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_sizes = {str(size) for size in voxel_sizes}
    if set(result["proxy"]) != expected_sizes:
        raise ValueError(f"shard voxel sizes do not match: {set(result['proxy'])}")
    scene_path, qpos, mapping, _ = _stage_files(run_root, label)
    if len(mapping) != result["official"]["both"]["frame_count"]:
        raise ValueError("geometry refresh refuses a partial/full-result frame mismatch")
    model = mujoco.MjModel.from_xml_path(str(scene_path)); data = mujoco.MjData(model); specs = _collision_specs(model)
    refreshed: dict[str, list[dict[str, Any]]] = {str(size): [] for size in voxel_sizes}
    for stage_index in sorted(_representative_indices(len(qpos))):
        data.qpos[:] = qpos[stage_index]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        raw = {side: _collision_union(specs[side], data) for side in ("left", "right")}
        reference = {side: _proxy_from_union(raw[side], reference_voxel_m) for side in ("left", "right")}
        for voxel_m in voxel_sizes:
            key = str(voxel_m)
            for side in ("left", "right"):
                proxy, status = _proxy_from_union(raw[side], voxel_m)
                if not (status["proxy_watertight"] and status["proxy_winding_consistent"]):
                    raise RuntimeError(f"{PROXY_LABEL} invalid at {label}/{stage_index}/{side}/{voxel_m}: {status}")
                ref_mesh, ref_status = reference[side]
                refreshed[key].append({
                    "source_frame": int(mapping[stage_index]), "stage_index": int(stage_index), "side": side,
                    "proxy": status,
                    "surface_error": _surface_error(ref_mesh, proxy, voxel_m, 20260901 + stage_index),
                    "reference": {"kind": "FINE_VOXEL_OCCUPANCY_UNION_BOUNDARY", "voxel_size_m": reference_voxel_m, **ref_status},
                    "validation_note": "Reference excludes overlapping-link interior triangles and represents the same collision occupancy union at a finer voxel pitch.",
                })
        print(f"geometry-refresh {label}: representative frame {stage_index + 1}/{len(qpos)}", flush=True)
    result["geometry_validation"] = refreshed
    result["geometry_validation_contract"] = {"reference": "FINE_VOXEL_OCCUPANCY_UNION_BOUNDARY", "reference_voxel_size_m": reference_voxel_m, "reason": "the raw concatenated collision mesh contains link-overlap interior surfaces which are not part of the union boundary"}
    write_json(result_path, result)
    return result_path


def refresh_full_c_html_payload(run_root: Path, shard_root: Path, label: str, voxel_sizes: list[float]) -> Path:
    """Write per-frame complete C meshes for lazy browser loading.

    Embedding all complete 100k+ triangle meshes for all representative frames
    makes a multi-gigabyte HTML file.  The base viewer therefore stays small
    and loads the selected C frame's complete raw-union/proxy triangles from a
    local JavaScript payload only when the C selector is chosen.
    """
    payload_path = shard_root / label / "html_payload.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"complete shard HTML payload required: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    scene_path, qpos, mapping, _ = _stage_files(run_root, label)
    frame_by_index = {int(frame["stage_index"]): frame for frame in payload["frames"]}
    expected = _representative_indices(len(qpos))
    if set(frame_by_index) != expected:
        raise ValueError("full-C HTML refresh requires exactly the frozen representative frames")
    model = mujoco.MjModel.from_xml_path(str(scene_path)); data = mujoco.MjData(model); specs = _collision_specs(model)
    display_voxel_m = min(voxel_sizes)
    mesh_root = payload_path.parent / "full_c_meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    for stage_index in sorted(expected):
        data.qpos[:] = qpos[stage_index]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        raw = {side: _collision_union(specs[side], data) for side in ("left", "right")}
        proxy = {side: _proxy_from_union(raw[side], display_voxel_m)[0] for side in ("left", "right")}
        frame = frame_by_index[stage_index]
        complete = {
            "collision": {side: _full_mesh(raw[side].vertices, raw[side].faces) for side in ("left", "right")},
            "proxy": {side: _full_mesh(proxy[side].vertices, proxy[side].faces) for side in ("left", "right")},
        }
        chunk = f"frame_{stage_index:04d}.js"
        chunk_body = f"window.__C_FULL_MESH=window.__C_FULL_MESH||{{}};window.__C_FULL_MESH[{stage_index}]={json.dumps(complete, ensure_ascii=False, default=_plain, separators=(',', ':'))};"
        write_text(mesh_root / chunk, chunk_body)
        # Retain only a small preview in the base HTML.  It is replaced by the
        # complete mesh immediately after the per-frame script has loaded.
        frame["collision"] = {side: _submesh(raw[side].vertices, raw[side].faces, 800) for side in ("left", "right")}
        frame["proxy"] = {side: _submesh(proxy[side].vertices, proxy[side].faces, 800) for side in ("left", "right")}
        frame["full_c_mesh_script"] = f"full_c_meshes/{chunk}"
        frame["c_mesh_rendering"] = "FULL_TRIANGLES_LAZY_LOAD"
        print(f"full-C-html {label}: representative frame {stage_index + 1}/{len(qpos)}", flush=True)
    write_json(payload_path, payload)
    return payload_path


def consolidate(primary_root: Path, backup_root: Path, output_root: Path, run_id: str, voxel_sizes: list[float]) -> Path:
    """Merge independently completed trajectories without recomputing frames."""
    root = (output_root / run_id).resolve()
    if root.exists(): raise FileExistsError(f"refuse to overwrite analysis root: {root}")
    sources = {"primary": primary_root.resolve(), "backup": backup_root.resolve()}
    results: dict[str, Any] = {}
    root.mkdir(parents=True)
    for label, source in sources.items():
        source_result, source_payload = source / label / "sequence_result.json", source / label / "html_payload.json"
        if not source_result.is_file() or not source_payload.is_file(): raise FileNotFoundError(f"incomplete {label} source root: {source}")
        result = json.loads(source_result.read_text(encoding="utf-8"))
        if set(result["proxy"]) != {str(size) for size in voxel_sizes}: raise ValueError(f"{label} voxel sizes do not match requested merge: {set(result['proxy'])}")
        # Recompute this from the serialized per-frame evidence so merging
        # also carries any later-added worst-deterioration fields exactly.
        for key, records in result["per_frame"]["proxy"].items():
            paired = _paired_stage_b_vs_official(result["per_frame"]["official"], records)
            for side in ("left", "right", "both"):
                result["proxy"][key][side]["vs_official"] = paired[side]
        results[label] = result
        destination = root / label; destination.mkdir()
        write_json(destination / "sequence_result.json", result)
        payload = json.loads(source_payload.read_text(encoding="utf-8"))
        # Accept completed shards generated before object_T was made explicit
        # as a flat browser affine, without touching their computed statistics.
        for frame in payload["frames"]:
            frame["object_T"] = np.asarray(frame["object_T"], dtype=float).reshape(-1).tolist()
            if "full_c_mesh_script" in frame:
                frame["full_c_mesh_script"] = f"{label}/{frame['full_c_mesh_script']}"
        write_json(destination / "html_payload.json", payload)
        source_mesh_root = source / label / "full_c_meshes"
        if source_mesh_root.is_dir():
            shutil.copytree(source_mesh_root, destination / "full_c_meshes")
    write_json(root / "merge_provenance.json", {"operation": "merge_complete_independent_trajectory_results", "sources": {label: {"root": str(source), "sequence_result_sha256": sha256(source / label / "sequence_result.json"), "html_payload_sha256": sha256(source / label / "html_payload.json")} for label, source in sources.items()}})
    _write_summary_artifacts(root, results, voxel_sizes)
    return root


def run(run_root: Path, output_root: Path, run_id: str, voxel_sizes: list[float], device: str, max_frames: int | None = None, selections: tuple[str, ...] = ("primary", "backup")) -> Path:
    root = (output_root / run_id).resolve()
    if root.exists(): raise FileExistsError(f"refuse to overwrite analysis root: {root}")
    root.mkdir(parents=True); results = {label: _run_sequence(run_root, root, label, voxel_sizes, device, max_frames=max_frames) for label in selections}
    _write_summary_artifacts(root, results, voxel_sizes)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-q4-root", type=Path, default=DEFAULT_Q0Q4_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO / ".local_artifacts/stage_b_voxel_proxy_audit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--voxel-sizes-mm", default="1,2", help="main/sensitivity proxy resolutions; use 0.5,1,2 only when budget permits")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, help="bounded diagnostic only; never a full-trajectory result")
    parser.add_argument("--selection", choices=("both", "primary", "backup"), default="both", help="independent trajectory execution; use both for an all-in-one result")
    parser.add_argument("--merge-primary-root", type=Path, help="completed primary-only root to merge")
    parser.add_argument("--merge-backup-root", type=Path, help="completed backup-only root to merge")
    parser.add_argument("--refresh-geometry-shard-root", type=Path, help="refresh only geometry validation in a complete one-selection shard")
    parser.add_argument("--refresh-full-c-html-shard-root", type=Path, help="refresh complete C collision/proxy mesh payload in a complete one-selection shard")
    args = parser.parse_args(); sizes = [float(value) / 1000.0 for value in args.voxel_sizes_mm.split(",")]
    if not sizes or any(value <= 0 for value in sizes): raise ValueError("voxel sizes must be positive")
    if args.max_frames is not None and args.max_frames <= 0: raise ValueError("max frames must be positive")
    selections = ("primary", "backup") if args.selection == "both" else (args.selection,)
    merge_roots = (args.merge_primary_root, args.merge_backup_root)
    refresh_roots = (args.refresh_geometry_shard_root, args.refresh_full_c_html_shard_root)
    if sum(value is not None for value in refresh_roots) > 1:
        raise ValueError("choose at most one derived refresh operation")
    if args.refresh_geometry_shard_root is not None:
        if any(merge_roots) or args.max_frames is not None or args.selection == "both":
            raise ValueError("geometry refresh requires exactly one selection, no merge, and no max-frame override")
        print(refresh_geometry_validation(args.q0_q4_root.resolve(), args.refresh_geometry_shard_root.resolve(), selections[0], sorted(set(sizes))))
    elif args.refresh_full_c_html_shard_root is not None:
        if any(merge_roots) or args.max_frames is not None or args.selection == "both":
            raise ValueError("full-C HTML refresh requires exactly one selection, no merge, and no max-frame override")
        print(refresh_full_c_html_payload(args.q0_q4_root.resolve(), args.refresh_full_c_html_shard_root.resolve(), selections[0], sorted(set(sizes))))
    elif any(merge_roots):
        if not all(merge_roots) or args.max_frames is not None or args.selection != "both": raise ValueError("merge requires both completed roots and no selection/max-frame override")
        print(consolidate(args.merge_primary_root, args.merge_backup_root, args.output_root.resolve(), args.run_id, sorted(set(sizes))))
    else:
        print(run(args.q0_q4_root.resolve(), args.output_root.resolve(), args.run_id, sorted(set(sizes)), args.device, max_frames=args.max_frames, selections=selections))


if __name__ == "__main__":
    main()
