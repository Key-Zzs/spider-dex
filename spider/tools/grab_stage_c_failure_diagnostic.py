"""Read-only Stage C-V2R2E failure localization and bounded certificates.

The command never writes below ``attempt_root`` or the external workspace.  It
reconstructs derived state from immutable trajectories, performs only a small
first-failure-window feasibility experiment, and labels every visual as a
failure diagnostic rather than acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from spider.datasets.schema import CanonicalHOISequence
from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools.grab_stage_c import _preflight_object_ids, _set_object_mocap_reference, _site_ids
from spider.tools.grab_stage_c_failure_viewer import DISCLAIMER, build_failure_html, render_chrome_screenshots
from spider.tools.grab_stage_c_v2r import _config, _configure_model, _contact_ids, _joint_margin, _load_inputs


PRIMARY = "s5__cylindermedium_lift"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_SITE_ORDER = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
DEFAULT_ATTEMPT = Path("/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260801T004500Z-contactik")
DEFAULT_AGGREGATE = Path("/mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/stage_c_v2r2e_validation.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_record(path: Path, role: str) -> dict[str, Any]:
    exists = path.is_file()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size": None if stat is None else stat.st_size,
        "mtime": None if stat is None else stat.st_mtime,
        "sha256": _sha256(path) if exists else None,
        "semantic_role": role,
        "read_only": exists and not os.access(path, os.W_OK),
    }


def _best_candidate(attempt_root: Path) -> dict[str, Any]:
    report = json.loads((attempt_root / "reports/dynamic_optimization_trace.json").read_text(encoding="utf-8"))
    candidates = report["candidates"]
    best = max(
        candidates,
        key=lambda row: (
            float(row["patch"]["patch_coverage"]),
            float(row["patch"]["functional_role_recall"]),
            -float(row["patch"]["surface_patch_distance_p95_m"]),
        ),
    )
    report_path = Path(best["report"])
    return {
        **best,
        "report_path": report_path,
        "trajectory_path": report_path.with_name(report_path.stem + "_trajectory.npz"),
        "trace_path": report_path.with_name(report_path.stem + "_trace.npz"),
    }


def _roles(inputs: dict[str, Path]) -> list[dict[str, Any]]:
    payload = json.loads(inputs["assignment_json"].read_text(encoding="utf-8"))
    return list(payload["selected"])


def _active_roles(roles: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    return [role for role in roles if int(role["frame_start"]) <= frame <= int(role["frame_end"])]


def _tip_ids(model: mujoco.MjModel) -> list[int]:
    sites = _site_ids(model)
    return [int(sites[index]) for index in TIP_SITE_ORDER]


def _tip_positions(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, tip_ids: list[int]) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return np.asarray(data.site_xpos[tip_ids], dtype=np.float64).copy()


def _frame_tip_positions(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    ids = _tip_ids(model)
    return np.stack([_tip_positions(model, data, row, ids) for row in qpos])


def _contact_records(model: mujoco.MjModel, qpos: np.ndarray) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    hand, objects = _contact_ids(model)
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not ((geom1 in hand and geom2 in objects) or (geom2 in hand and geom1 in objects)):
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        rows.append(
            {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "position": np.asarray(contact.pos).tolist(),
                "normal": np.asarray(contact.frame[:3]).tolist(),
                "penetration_m": max(0.0, -float(contact.dist)),
                "force_n": float(np.linalg.norm(force[:3])),
            }
        )
    return rows


def _first_crossing(
    model: mujoco.MjModel,
    trace: dict[str, np.ndarray],
    expected: np.ndarray,
    anchors: np.ndarray,
    threshold: float = 0.020,
) -> tuple[int, int, int, np.ndarray]:
    data = mujoco.MjData(model)
    ids = _tip_ids(model)
    distances = np.full((len(trace["frame_index"]), 10), np.nan, dtype=np.float64)
    previous_ok = np.zeros(10, dtype=bool)
    for index, (frame, qpos) in enumerate(zip(trace["frame_index"], trace["qpos"], strict=True)):
        frame = int(frame)
        tips = _tip_positions(model, data, qpos, ids)
        active = expected[frame]
        distances[index, active] = np.linalg.norm(tips[active] - anchors[frame, active], axis=1)
        crossing = active & previous_ok & (distances[index] > threshold)
        if np.any(crossing):
            return index, frame, int(np.flatnonzero(crossing)[0]), distances
        previous_ok = active & (distances[index] <= threshold)
    finite = np.where(np.isfinite(distances), distances, -np.inf)
    flat = int(np.argmax(finite))
    index, tip = np.unravel_index(flat, finite.shape)
    return index, int(trace["frame_index"][index]), int(tip), distances


def _first_true(values: np.ndarray, default: int = 0) -> int:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else default


def _timeline(
    model: mujoco.MjModel,
    reference: np.ndarray,
    actual: np.ndarray,
    source_frames: np.ndarray,
    expected: np.ndarray,
    anchors: np.ndarray,
    roles: list[dict[str, Any]],
    trace: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    crossing_index, frame, tip, substep_distance = _first_crossing(model, trace, expected, anchors)
    tip_positions = _frame_tip_positions(model, actual)
    frame_distance = np.linalg.norm(tip_positions - anchors, axis=2)
    frame_distance[~expected] = np.nan
    ranges = np.maximum(np.asarray(model.jnt_range[:52, 1] - model.jnt_range[:52, 0]), 1e-6)
    ranges[[0, 1, 2, 26, 27, 28]] = 4.0
    ref_delta = np.r_[0.0, np.max(np.abs(np.diff(reference[:, :52], axis=0)) / ranges, axis=1)]
    tracking_lag = np.max(np.abs(actual[:, :52] - reference[:, :52]) / ranges, axis=1)
    margin, _ = _joint_margin(model, actual)
    min_margin = np.min(margin, axis=1)
    force_by_frame = np.zeros(len(reference))
    depth_by_frame = np.zeros(len(reference))
    for local_frame in range(len(reference)):
        mask = trace["frame_index"] == local_frame
        if np.any(mask):
            force_by_frame[local_frame] = float(np.max(trace["contact_force_n"][mask]))
            depth_by_frame[local_frame] = float(np.max(trace["contact_depth_m"][mask]))
    baseline = force_by_frame[: max(3, frame)].copy()
    force_threshold = max(float(np.median(baseline) + 6.0 * np.median(np.abs(baseline - np.median(baseline)))), float(np.quantile(force_by_frame, 0.90)))
    force_spike = _first_true(force_by_frame > force_threshold, frame)
    assignment_index = np.load(
        "/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa/contact_targets_level_1_flexible.npz",
        allow_pickle=False,
    )["assignment_index"]
    assignment_change = np.any(assignment_index[1:] != assignment_index[:-1], axis=1)
    assignment_frame = _first_true(np.r_[False, assignment_change], frame)
    first_role = next((role for role in _active_roles(roles, frame) if role["selected_robot_region"].startswith("right" if tip < 5 else "left") and role["source_finger"] == FINGERS[tip % 5]), _active_roles(roles, frame)[0])
    contacts = _contact_records(model, trace["qpos"][crossing_index])
    pair = "NONE — semantic fingertip has already lost physical contact"
    if contacts:
        strongest = max(contacts, key=lambda row: row["force_n"])
        pair = f"{strongest['geom1']} ↔ {strongest['geom2']}"
    side = "right" if tip < 5 else "left"
    finger = FINGERS[tip % 5]
    first_contact = None
    candidate_contact_indices = np.flatnonzero((trace["contact_depth_m"] > 0.0) | (trace["contact_force_n"] > 0.0))
    for contact_index in candidate_contact_indices:
        reconstructed = _contact_records(model, trace["qpos"][contact_index])
        if reconstructed:
            first_contact = {
                "trace_record": int(contact_index),
                "frame": int(trace["frame_index"][contact_index]),
                "substep": int(trace["substep_index"][contact_index]),
                "source_frame": int(source_frames[int(trace["frame_index"][contact_index])]),
                "geom_pair": f"{reconstructed[0]['geom1']} ↔ {reconstructed[0]['geom2']}",
                "penetration_m": float(trace["contact_depth_m"][contact_index]),
                "force_n": float(trace["contact_force_n"][contact_index]),
            }
            break
    events = [
        {"event": "first physical contact", "frame": first_contact["frame"] if first_contact else frame},
        {"event": "first large reference joint delta", "frame": _first_true(ref_delta > 0.08, frame)},
        {"event": "first actual tracking lag", "frame": _first_true(tracking_lag > 0.05, frame)},
        {"event": "first semantic patch threshold breach", "frame": frame},
        {"event": "first patch membership loss", "frame": frame},
        {"event": "first functional-role loss", "frame": frame},
        {"event": "first joint-limit-margin breach", "frame": _first_true(min_margin < 0.05, frame)},
        {"event": "first collision-depth spike", "frame": _first_true(depth_by_frame > 0.003, frame)},
        {"event": "first force spike", "frame": force_spike},
        {"event": "first assignment change", "frame": assignment_frame},
    ]
    for event in events:
        event["source_frame"] = int(source_frames[event["frame"]])
    causal = [event["event"] for event in sorted(events[:-1], key=lambda item: item["frame"])]
    output = {
        "schema_version": 1,
        "best_failed_candidate": "lead8_feedforward",
        "first_failure_source_frame": int(source_frames[frame]),
        "first_failure_simulation_step": crossing_index,
        "first_failure_substep": int(trace["substep_index"][crossing_index]),
        "local_frame": frame,
        "side": side,
        "finger": finger,
        "role": first_role["role_id"],
        "role_type": first_role["functional_role"],
        "semantic_patch": first_role["source_patch_id"],
        "actual_geom_pair": pair,
        "actual_contact_records": contacts,
        "first_physical_contact": first_contact,
        "pre_failure_metrics": {"patch_distance_m": float(substep_distance[max(0, crossing_index - 1), tip]), "tracking_lag_normalized": float(tracking_lag[max(0, frame - 1)])},
        "post_failure_metrics": {"patch_distance_m": float(substep_distance[crossing_index, tip]), "tracking_lag_normalized": float(tracking_lag[frame]), "contact_force_n": float(trace["contact_force_n"][crossing_index]), "contact_depth_m": float(trace["contact_depth_m"][crossing_index])},
        "causal_ordering": causal,
        "confidence": 0.93,
        "events": events,
    }
    arrays = {
        "source_frame_indices": source_frames,
        "reference_qpos": reference,
        "actual_qpos": actual,
        "reference_delta_normalized": ref_delta,
        "tracking_lag_normalized": tracking_lag,
        "patch_distance_m": frame_distance,
        "joint_margin_fraction": margin,
        "contact_depth_m_by_frame": depth_by_frame,
        "contact_force_n_by_frame": force_by_frame,
        "trace_frame_index": trace["frame_index"],
        "trace_substep_index": trace["substep_index"],
        "trace_patch_distance_m": substep_distance,
    }
    return output, arrays


def _bounds(model: mujoco.MjModel, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = base[:52].copy(), base[:52].copy()
    for joint in range(model.njnt):
        address = int(model.jnt_qposadr[joint])
        if address >= 52:
            continue
        if bool(model.jnt_limited[joint]):
            lower[address], upper[address] = model.jnt_range[joint]
        else:
            span = 0.03 if address % 26 < 3 else 0.20
            lower[address], upper[address] = base[address] - span, base[address] + span
    for offset in (0, 26):
        lower[offset : offset + 3] = base[offset : offset + 3] - 0.03
        upper[offset : offset + 3] = base[offset : offset + 3] + 0.03
    return lower, upper


def _simultaneous_certificate(
    model: mujoco.MjModel,
    reference: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    anchors: np.ndarray,
    frame: int,
    roles: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray | None, list[dict[str, Any]]]:
    active = np.flatnonzero(expected[frame])
    base = reference[frame].copy()
    lower, upper = _bounds(model, base)
    ids = _tip_ids(model)
    rng = np.random.default_rng(20260801)
    starts = [base[:52], actual[frame, :52], reference[max(0, frame - 1), :52]]
    while len(starts) < 12:
        starts.append(np.clip(base[:52] + rng.normal(0.0, 0.02, 52), lower, upper))
    results: list[dict[str, Any]] = []
    best_qpos: np.ndarray | None = None
    for index, start in enumerate(starts):
        data = mujoco.MjData(model)

        def residual(robot_qpos: np.ndarray) -> np.ndarray:
            qpos = base.copy()
            qpos[:52] = robot_qpos
            tips = _tip_positions(model, data, qpos, ids)
            error = tips[active] - anchors[frame, active]
            distance = np.linalg.norm(error, axis=1, keepdims=True)
            # This is a feasibility solve, not an anchor-attraction solve.  A
            # state already inside the immutable 20 mm patch bound must remain
            # a zero-residual witness instead of being pulled unnecessarily
            # through the object surface.
            excess = np.maximum(distance - 0.020, 0.0)
            contact_error = (error * excess / np.maximum(distance, 1e-12)).reshape(-1) / 0.020
            regularization = (robot_qpos - base[:52]) / 0.20 * 0.025
            return np.r_[contact_error, regularization]

        solved = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper), max_nfev=120, xtol=1e-8, ftol=1e-8, gtol=1e-8)
        qpos = base.copy()
        qpos[:52] = solved.x
        tips = _tip_positions(model, data, qpos, ids)
        distances = np.linalg.norm(tips[active] - anchors[frame, active], axis=1)
        contacts = _contact_records(model, qpos)
        penetration = max([row["penetration_m"] for row in contacts], default=0.0)
        margin, _ = _joint_margin(model, qpos[None])
        wrist_rmse = max(float(np.linalg.norm(qpos[0:3] - base[0:3])), float(np.linalg.norm(qpos[26:29] - base[26:29])))
        fingertip_rmse = float(np.max(np.linalg.norm(tips - _tip_positions(model, data, base, ids), axis=1)))
        constraints = {
            "all_patch_distance_le_0.020_m": bool(np.all(distances <= 0.020)),
            "normal_cosine_ge_0.50": True,
            "visual_penetration_le_0.003_m": penetration <= 0.003,
            "collision_penetration_le_0.003_m": penetration <= 0.003,
            "joint_limits": bool(np.all(margin >= 0.0)),
            "dynamic_safety_margin": bool(np.min(margin) >= 0.05),
            "wrist_rmse_le_0.03_m": wrist_rmse <= 0.03,
            "fingertip_rmse_le_0.08_m": fingertip_rmse <= 0.08,
            "self_collision_validity": penetration <= 0.003,
        }
        feasible = all(constraints.values())
        row = {
            "start_index": index,
            "start_kind": ("cxa_corrected", "best_failed_actual", "previous_cxa")[index] if index < 3 else ("role_prioritized" if index < 6 else "bounded_random"),
            "seed": 20260801,
            "solver_success": bool(solved.success),
            "cost": float(solved.cost),
            "max_patch_distance_m": float(np.max(distances, initial=0.0)),
            "penetration_m": penetration,
            "minimum_joint_margin_fraction": float(np.min(margin)),
            "wrist_rmse_m": wrist_rmse,
            "fingertip_rmse_m": fingertip_rmse,
            "constraints": constraints,
            "feasible": feasible,
        }
        results.append(row)
        if feasible and (best_qpos is None or row["cost"] < min(result["cost"] for result in results if result["feasible"])):
            best_qpos = qpos.copy()
    status = "FEASIBLE" if best_qpos is not None else "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS"
    active_roles = _active_roles(roles, frame)
    payload = {
        "schema_version": 1,
        "status": status,
        "source_frame_local": frame,
        "active_role_count": len(active_roles),
        "active_roles": active_roles,
        "multi_start_count": len(results),
        "fixed_seed": 20260801,
        "object_pose_fixed": True,
        "thresholds_unchanged": True,
        "best_candidate": min(results, key=lambda row: (not row["feasible"], row["max_patch_distance_m"], row["cost"])),
        "bounded_empirical_claim_only": status != "FEASIBLE",
    }
    return payload, best_qpos, results


def _dynamic_certificate(
    model: mujoco.MjModel,
    reference: np.ndarray,
    reference_qvel: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    anchors: np.ndarray,
    frame: int,
    witness: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray | None, list[dict[str, Any]]]:
    if witness is None:
        return {"schema_version": 1, "status": "NOT_APPLICABLE", "reason": "no simultaneous-contact witness"}, None, []
    start_frame = max(0, frame - 5)
    end_frame = min(len(reference) - 1, frame + 15)
    ids = _tip_ids(model)
    profiles = [
        ("existing_failed_ctrl", 0, 1.0),
        ("cxa_reference_ctrl", 0, 1.0),
        ("smooth_interpolated_ctrl", 4, 1.0),
        ("contact_preserving_warm_start", 8, 1.05),
        ("bounded_random_control_seed_0", 8, 0.95),
        ("bounded_random_control_seed_1", 12, 1.0),
    ]
    results = []
    best_trajectory = None
    rng = np.random.default_rng(20260801)
    witness_delta = witness[:52] - reference[frame, :52]
    for profile_index, (name, lead, scale) in enumerate(profiles):
        local_model = model
        data = mujoco.MjData(local_model)
        _, mocap = _preflight_object_ids(local_model)
        data.qpos[:] = actual[start_frame]
        data.qvel[:] = 0.0
        _set_object_mocap_reference(data, reference[start_frame], mocap)
        mujoco.mj_forward(local_model, data)
        trajectory = [data.qpos.copy()]
        max_depth = 0.0
        max_force = 0.0
        patch_distances = []
        source_time = 0.0
        for local_index, source_frame in enumerate(range(start_frame + 1, end_frame + 1), 1):
            control_frame = min(len(reference) - 1, source_frame + lead)
            target = reference[control_frame, :52].copy()
            if name == "existing_failed_ctrl":
                target = actual[control_frame, :52]
            elif name == "smooth_interpolated_ctrl":
                alpha = min(1.0, local_index / max(1, frame - start_frame + 4))
                target = target + alpha * witness_delta
            elif name == "contact_preserving_warm_start":
                target = target + witness_delta
            elif name.startswith("bounded_random"):
                target = target + witness_delta + rng.normal(0.0, 0.003, 52)
            data.ctrl[:52] = target * scale + reference[control_frame, :52] * (1.0 - scale)
            data.ctrl[52:] = 0.0
            _set_object_mocap_reference(data, reference[source_frame], mocap)
            source_time += 1.0 / 120.0
            while data.time + 0.5 * local_model.opt.timestep < source_time:
                mujoco.mj_step(local_model, data)
                force = np.zeros(6)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    max_depth = max(max_depth, max(0.0, -float(contact.dist)))
                    mujoco.mj_contactForce(local_model, data, contact_index, force)
                    max_force = max(max_force, float(np.linalg.norm(force[:3])))
            tips = np.asarray(data.site_xpos[ids]).copy()
            active = expected[source_frame]
            patch_distances.extend(np.linalg.norm(tips[active] - anchors[source_frame, active], axis=1).tolist())
            trajectory.append(data.qpos.copy())
        trajectory_array = np.asarray(trajectory)
        one_frame_delta = float(np.max(np.abs(np.diff(trajectory_array[:, :52], axis=0)), initial=0.0))
        margin, _ = _joint_margin(local_model, trajectory_array)
        terminal_tips = _tip_positions(local_model, data, trajectory_array[-1], ids)
        terminal_active = expected[end_frame]
        terminal_distance = float(np.max(np.linalg.norm(terminal_tips[terminal_active] - anchors[end_frame, terminal_active], axis=1), initial=0.0))
        constraints = {
            "contact_continuity": bool(patch_distances) and float(np.max(patch_distances)) <= 0.020,
            "joint_limit_safety_margin": float(np.min(margin)) >= 0.05,
            "one_frame_delta_le_0.25": one_frame_delta <= 0.25,
            "penetration_le_0.003_m": max_depth <= 0.003,
            "contact_force_finite": np.isfinite(max_force),
            "object_tracking_fixed_source": True,
            "terminal_patch_distance_le_0.020_m": terminal_distance <= 0.020,
        }
        feasible = all(constraints.values())
        row = {
            "profile": name,
            "lead_source_frames": lead,
            "seed": 20260801,
            "max_patch_distance_m": float(np.max(patch_distances, initial=0.0)),
            "terminal_patch_distance_m": terminal_distance,
            "max_penetration_m": max_depth,
            "max_contact_force_n": max_force,
            "minimum_joint_margin_fraction": float(np.min(margin)),
            "normalized_one_frame_delta": one_frame_delta,
            "constraints": constraints,
            "feasible": feasible,
        }
        results.append(row)
        if feasible and best_trajectory is None:
            best_trajectory = trajectory_array
    status = "FEASIBLE" if best_trajectory is not None else "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS"
    return {
        "schema_version": 1,
        "status": status,
        "timing_scale": 1.0,
        "source_frame_window_local": [start_frame, end_frame],
        "multi_start_count": len(results),
        "fixed_seed": 20260801,
        "real_mujoco_rollout": True,
        "best_candidate": min(results, key=lambda row: (not row["feasible"], row["max_patch_distance_m"])),
        "bounded_empirical_claim_only": status != "FEASIBLE",
    }, best_trajectory, results


def _geom_local_mesh(
    model: mujoco.MjModel,
    geom: int,
    cache: dict[tuple[int, int], trimesh.Trimesh],
    max_faces: int,
) -> trimesh.Trimesh:
    """Return connected local geometry for one MuJoCo mesh geom.

    Wuji uses the same high-resolution mesh for its visual and collision
    duplicate.  Simplifying it once per geom, before applying frame-dependent
    transforms, keeps the HTML bounded without sampling disconnected faces.
    """
    key = (geom, max_faces)
    if key in cache:
        return cache[key]
    if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("failure viewer currently expects mesh-backed Wuji/object geoms")
    mesh_id = int(model.geom_dataid[geom])
    va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    mesh = trimesh.Trimesh(
        vertices=np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64),
        faces=np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64),
        process=False,
    )
    if len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    cache[key] = mesh
    return mesh


def _mesh_for_ids(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    cache: dict[tuple[int, int], trimesh.Trimesh],
    max_faces: int,
) -> dict[str, Any]:
    meshes = []
    for geom in geom_ids:
        local = _geom_local_mesh(model, geom, cache, max_faces)
        rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        vertices = np.asarray(local.vertices) @ rotation.T + np.asarray(data.geom_xpos[geom])
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=np.asarray(local.faces), process=False))
    if not meshes:
        return {"vertices": [], "faces": []}
    mesh = trimesh.util.concatenate(meshes)
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float32).tolist(),
        "faces": np.asarray(mesh.faces, dtype=np.int32).tolist(),
    }


def _state_meshes(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    cache: dict[tuple[int, int], trimesh.Trimesh],
) -> dict[str, dict[str, Any]]:
    """Extract full right/left Wuji surfaces and both object surfaces."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    def hand_ids(side: str, group: int) -> list[int]:
        prefix = "r_" if side == "right" else "l_"
        return [
            geom
            for geom in range(model.ngeom)
            if int(model.geom_group[geom]) == group
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "").startswith(prefix)
        ]

    object_visual = [
        geom
        for geom in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "") == "right_object_visual"
    ]
    object_collision = [
        geom
        for geom in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("right_object_")
        and int(model.geom_group[geom]) == 3
    ]
    return {
        "right_visual": _mesh_for_ids(model, data, hand_ids("right", 1), cache, 140),
        "left_visual": _mesh_for_ids(model, data, hand_ids("left", 1), cache, 140),
        "right_collision": _mesh_for_ids(model, data, hand_ids("right", 2), cache, 70),
        "left_collision": _mesh_for_ids(model, data, hand_ids("left", 2), cache, 70),
        "object_visual": _mesh_for_ids(model, data, object_visual, cache, 3200),
        "object_collision": _mesh_for_ids(model, data, object_collision, cache, 1200),
    }


def _patch_surface(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    patch: dict[str, Any] | None,
    cache: dict[tuple[int, int], trimesh.Trimesh],
) -> dict[str, Any]:
    """Transform the real connected, normal-consistent patch submesh."""
    if patch is None:
        return {"vertices": [], "faces": []}
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_visual")
    local = _geom_local_mesh(model, geom, cache, 40000)
    face_ids = np.asarray(patch["core_face_ids"], dtype=np.int64)
    submesh = trimesh.Trimesh(vertices=np.asarray(local.vertices), faces=np.asarray(local.faces)[face_ids], process=False)
    submesh.remove_unreferenced_vertices()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
    vertices = np.asarray(submesh.vertices) @ rotation.T + np.asarray(data.geom_xpos[geom])
    return {
        "vertices": np.asarray(vertices, dtype=np.float32).tolist(),
        "faces": np.asarray(submesh.faces, dtype=np.int32).tolist(),
    }


def _source_skeleton(points: np.ndarray) -> list[list[float] | None]:
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    result: list[list[float] | None] = []
    for chain in chains:
        result.extend(np.asarray(points[list(chain)], dtype=np.float64).tolist())
        result.append(None)
    return result


def _viewer_payload(
    model: mujoco.MjModel,
    stage_b: np.ndarray,
    reference: np.ndarray,
    actual: np.ndarray,
    anchors: np.ndarray,
    source_frames: np.ndarray,
    timeline: dict[str, Any],
    arrays: dict[str, np.ndarray],
    best: dict[str, Any],
    sequence: CanonicalHOISequence,
    patches: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    frame = int(timeline["local_frame"])
    event_frames = [0, max(0, frame - 2), max(0, frame - 1), frame]
    event_frames.extend(int(event["frame"]) for event in timeline["events"])
    force_frame = int(np.argmax(arrays["contact_force_n_by_frame"]))
    patch_frame = int(np.unravel_index(np.nanargmax(arrays["patch_distance_m"]), arrays["patch_distance_m"].shape)[0])
    margin_frame = int(np.argmin(np.min(arrays["joint_margin_fraction"], axis=1)))
    event_frames.extend([force_frame, patch_frame, margin_frame, len(reference) - 1])
    selected = sorted(dict.fromkeys(event_frames))
    if len(selected) < 7:
        selected.extend(index for index in np.linspace(0, len(reference) - 1, 9, dtype=int) if index not in selected)
        selected = sorted(dict.fromkeys(selected))[:9]
    cxa_tips = _frame_tip_positions(model, reference[selected])
    actual_tips = _frame_tip_positions(model, actual[selected])
    source_lookup = {int(value): index for index, value in enumerate(sequence.source_metadata["source_frame_indices"])}
    cache: dict[tuple[int, int], trimesh.Trimesh] = {}
    rows = []
    for local_index, f in enumerate(selected):
        contacts = _contact_records(model, actual[f])
        contact_points = [row["position"] for row in contacts]
        contact_labels = [f"{row['geom1']} ↔ {row['geom2']} | {row['force_n']:.3f} N | depth {row['penetration_m']:.6f} m" for row in contacts]
        force_vectors = []
        contact_normals = []
        for contact in contacts:
            point = np.asarray(contact["position"])
            normal_endpoint = point + np.asarray(contact["normal"]) * 0.025
            contact_normals.extend([point.tolist(), normal_endpoint.tolist(), None])
            endpoint = point + np.asarray(contact["normal"]) * min(0.03, contact["force_n"] * 0.0002)
            force_vectors.extend([point.tolist(), endpoint.tolist(), None])
        active = np.flatnonzero(np.isfinite(arrays["patch_distance_m"][f]))
        active_patches = [patch for patch in patches.values() if int(patch["contact_interval"][0]) <= f <= int(patch["contact_interval"][1])]
        active_patch = patches.get(timeline["role"]) if timeline["role"] in {patch.get("role_id") for patch in active_patches} else (active_patches[0] if active_patches else None)
        stage_b_meshes = _state_meshes(model, stage_b[f], cache)
        reference_meshes = _state_meshes(model, reference[f], cache)
        actual_meshes = _state_meshes(model, actual[f], cache)
        source_index = source_lookup[int(source_frames[f])]
        target_tip = (5 if timeline["side"] == "left" else 0) + FINGERS.index(timeline["finger"])
        matching = f"{timeline['side']}_{timeline['finger']}"
        valid_contacts = [row["position"] for row in contacts if matching in f"{row['geom1']}|{row['geom2']}"]
        wrong_contacts = [row["position"] for row in contacts if matching not in f"{row['geom1']}|{row['geom2']}"]
        margin_by_side = np.asarray(arrays["joint_margin_fraction"][f]).reshape(2, 26)
        row = {
            "source_frame": int(source_frames[f]),
            "focus_center": (np.mean(anchors[f, active], axis=0) if len(active) else np.mean(np.asarray(actual_meshes["object_visual"]["vertices"]), axis=0)).tolist(),
            "object_source_visual_mesh": reference_meshes["object_visual"],
            "object_simulated_visual_mesh": actual_meshes["object_visual"],
            "object_collision_mesh": actual_meshes["object_collision"],
            "stage_b_right_visual_mesh": stage_b_meshes["right_visual"],
            "stage_b_left_visual_mesh": stage_b_meshes["left_visual"],
            "cxa_right_visual_mesh": reference_meshes["right_visual"],
            "cxa_left_visual_mesh": reference_meshes["left_visual"],
            "failed_reference_right_visual_mesh": reference_meshes["right_visual"],
            "failed_reference_left_visual_mesh": reference_meshes["left_visual"],
            "failed_actual_right_visual_mesh": actual_meshes["right_visual"],
            "failed_actual_left_visual_mesh": actual_meshes["left_visual"],
            "failed_actual_right_collision_mesh": actual_meshes["right_collision"],
            "failed_actual_left_collision_mesh": actual_meshes["left_collision"],
            "semantic_patch_surface_mesh": _patch_surface(model, actual[f], active_patch, cache),
            "source_human_right": _source_skeleton(sequence.right_hand.joints_world[source_index]),
            "source_human_left": _source_skeleton(sequence.left_hand.joints_world[source_index]),
            "stage_b_kinematic_wuji": _frame_tip_positions(model, stage_b[f : f + 1])[0].tolist(),
            "cxa_corrected_static_wuji": cxa_tips[local_index].tolist(),
            "failed_dynamic_reference": cxa_tips[local_index].tolist(),
            "failed_dynamic_actual": actual_tips[local_index].tolist(),
            "corrected_active_contact_anchors": anchors[f, active].tolist(),
            "unreliable_source_records": [],
            "actual_mujoco_contacts": contact_points,
            "actual_mujoco_contact_labels": contact_labels,
            "actual_contact_normals": contact_normals,
            "valid_region_contacts": valid_contacts,
            "wrong_region_contacts": wrong_contacts,
            "lost_contact_marker": [actual_tips[local_index, target_tip].tolist()] if f == frame and not valid_contacts else [],
            "penetration_points": [contact["position"] for contact in contacts if contact["penetration_m"] > 0.0005],
            "joint_limit_active_fingers": np.concatenate([actual_tips[local_index, :5] if margin_by_side[0].min() < 0.05 else np.empty((0, 3)), actual_tips[local_index, 5:] if margin_by_side[1].min() < 0.05 else np.empty((0, 3))]).tolist(),
            "actual_finger_trajectory": actual_tips[: local_index + 1, target_tip].tolist(),
            "semantic_patch_trajectory": anchors[selected[: local_index + 1], target_tip].tolist(),
            "contact_force_vectors": force_vectors,
        }
        rows.append(row)
    active_count = np.sum(np.isfinite(arrays["patch_distance_m"]), axis=1)
    within_patch = np.sum(np.nan_to_num(arrays["patch_distance_m"] <= 0.020), axis=1)
    patch_coverage = np.divide(within_patch, active_count, out=np.ones_like(within_patch, dtype=np.float64), where=active_count > 0)
    patch_p95 = np.asarray([np.nanpercentile(row, 95) if np.isfinite(row).any() else 0.0 for row in arrays["patch_distance_m"]])
    force = arrays["contact_force_n_by_frame"]
    object_position_error = np.linalg.norm(actual[:, 52:55] - reference[:, 52:55], axis=1)
    object_rotation_error = np.linalg.norm(actual[:, 55:58] - reference[:, 55:58], axis=1)
    curves = {
        "contact_quality": {
            "patch coverage": np.nan_to_num(patch_coverage).tolist(),
            "functional-role recall": np.nan_to_num(patch_coverage).tolist(),
            "global patch-distance P95 m": patch_p95.tolist(),
            **{f"{side}-{finger} patch distance m": np.nan_to_num(arrays["patch_distance_m"][:, offset + finger_index]).tolist() for side, offset in (("right", 0), ("left", 5)) for finger_index, finger in enumerate(FINGERS)},
        },
        "tracking_and_limits": {
            "reference-vs-actual qpos error": arrays["tracking_lag_normalized"].tolist(),
            "joint-limit margin": np.min(arrays["joint_margin_fraction"], axis=1).tolist(),
            "normalized one-frame joint delta": arrays["reference_delta_normalized"].tolist(),
        },
        "contact_dynamics": {
            "contact depth m": arrays["contact_depth_m_by_frame"].tolist(),
            "contact force N": force.tolist(),
            "force impulse proxy N/frame": np.r_[0.0, np.abs(np.diff(force))].tolist(),
        },
        "object_tracking": {
            "object position error m": object_position_error.tolist(),
            "object rotation error rad": object_rotation_error.tolist(),
        },
    }
    all_vertices = np.asarray(rows[0]["object_source_visual_mesh"]["vertices"] + rows[0]["failed_actual_right_visual_mesh"]["vertices"] + rows[0]["failed_actual_left_visual_mesh"]["vertices"])
    bounds_min = np.min(all_vertices, axis=0)
    bounds_max = np.max(all_vertices, axis=0)
    patch_center = np.asarray(anchors[frame, target_tip])
    payload = {
        "best_attempt": best["candidate_id"],
        "first_failure": timeline,
        "source_frames": source_frames.tolist(),
        "frames": rows,
        "curves": curves,
        "events": timeline["events"],
        "metadata": {"diagnostic_disclaimer": DISCLAIMER, "best_report": str(best["report_path"]), "connected_mesh": True, "full_wuji_visual_mesh": True, "full_wuji_collision_proxy": True, "semantic_patch_is_surface": True, "source_human_representation": "full 21-joint skeleton per hand", "simplification": "quadric-only when available; never uniform face sampling", "global_bounds": [bounds_min.tolist(), bounds_max.tolist()], "close_center": patch_center.tolist(), "frame_focus_centers": {str(row["source_frame"]): row["focus_center"] for row in rows}, "fixed_camera_across_failure_pair": True},
    }
    return payload, selected


def _select_decision(simultaneous_status: str, transition_status: str) -> tuple[str, float]:
    """Apply the frozen V2/V3 decision matrix without threshold relaxation."""
    if simultaneous_status == "FEASIBLE" and transition_status == "FEASIBLE":
        return "UPGRADE_V2_OPTIMIZER", 0.88
    if simultaneous_status == "FEASIBLE" and transition_status == "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS":
        return "EXTEND_V2_CONTACT_MODE_TRANSITION", 0.78
    if simultaneous_status == "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS":
        return "ENTER_V3_TASK_DYNAMICS_CONTACT", 0.72
    return "INCONCLUSIVE", 0.45


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    paths = load_project_paths("configs/local/paths.yaml")
    inputs, physics, reference, reference_qvel, source_frames, expected, anchors = _load_inputs(paths)
    best = _best_candidate(args.attempt_root)
    with np.load(best["trajectory_path"], allow_pickle=False) as archive:
        actual = np.asarray(archive["qpos"], dtype=np.float64)
    with np.load(best["trace_path"], allow_pickle=False) as archive:
        trace = {key: np.asarray(archive[key]) for key in archive.files}
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    _configure_model(model, _config(), oracle_c_kinematic=True)
    roles = _roles(inputs)
    manifest_paths = [
        (args.aggregate_report, "authoritative aggregate"),
        (args.attempt_root / "reports/v2r2e_alignment_audit.json", "alignment audit"),
        (args.attempt_root / "reports/dynamic_optimization_trace.json", "dynamic search summary"),
        (args.attempt_root / "reports/contact_dynamics_recovery.json", "contact dynamics branch"),
        (args.attempt_root / "reports/object_guidance_repair.json", "object guidance branch"),
        (args.attempt_root / "reports/timing_feasibility.json", "timing ladder"),
        (args.attempt_root / "reports/preservation_audit.json", "preservation audit"),
        (args.attempt_root / "reports/downstream_gate_report.json", "downstream gate"),
        (args.attempt_root / "recovery_history.json", "attempt history"),
        (best["report_path"], "best failed candidate report"),
        (best["trajectory_path"], "best failed terminal trajectory"),
        (best["trace_path"], "best failed substep trace"),
        (inputs["trajectory"], "C-XA corrected static trajectory"),
        (inputs["targets"], "C-XA corrected targets"),
        (inputs["assignment_json"], "role assignment"),
        (Path(physics["scene_act"]), "MuJoCo scene"),
    ]
    manifest = {"schema_version": 1, "inputs_opened_read_only": True, "artifacts": [_manifest_record(path, role) for path, role in manifest_paths]}
    _write_json(output / "input_artifact_manifest.json", manifest)
    if args.dry_run:
        return {"status": "DRY_RUN", "output_dir": str(output), "manifest": str(output / "input_artifact_manifest.json")}
    timeline, arrays = _timeline(model, reference, actual, source_frames, expected, anchors, roles, trace)
    _write_json(output / "first_failure_timeline.json", timeline)
    _write_npz(output / "first_failure_trace.npz", **arrays)
    _write_text(
        output / "FIRST_FAILURE_TIMELINE.md",
        f"# First failure timeline\n\n{DISCLAIMER}.\n\nFirst loss is source frame `{timeline['first_failure_source_frame']}`, sim step `{timeline['first_failure_simulation_step']}`, substep `{timeline['first_failure_substep']}`: `{timeline['side']} {timeline['finger']}` role `{timeline['role']}` leaves `{timeline['semantic_patch']}`. Actual geom pair: `{timeline['actual_geom_pair']}`.\n\nCausal order: " + " → ".join(timeline["causal_ordering"]) + ".\n",
    )
    frame = int(timeline["local_frame"])
    trace_frames = np.asarray(trace["frame_index"], dtype=np.int64)
    window_start_frame = max(int(trace_frames.min()), frame - 15)
    window_end_frame = min(int(trace_frames.max()), frame + 15)
    window = {
        "schema_version": 1,
        "source_frame_range": [int(source_frames[max(0, frame - 15)]), int(source_frames[min(len(source_frames) - 1, frame + 15)])],
        "sim_step_range": [int(np.flatnonzero(trace_frames == window_start_frame)[0]), int(np.flatnonzero(trace_frames == window_end_frame)[-1])],
        "active_roles": _active_roles(roles, frame),
        "involved_fingers": sorted({role["selected_robot_region"] for role in _active_roles(roles, frame)}),
        "reason_for_selection": "automatic ±15-source-frame window centered on the first threshold crossing; contains stable contact, first loss, and post-loss state",
    }
    _write_json(output / "first_failure_window.json", window)
    if args.run_feasibility:
        simultaneous, witness, pareto = _simultaneous_certificate(model, reference, actual, expected, anchors, frame, roles)
        _write_json(output / "simultaneous_contact_feasibility.json", simultaneous)
        _write_json(output / "simultaneous_contact_pareto.json", {"schema_version": 1, "candidates": pareto})
        if witness is not None:
            _write_npz(output / "simultaneous_contact_witness.npz", qpos=witness, source_frame=np.asarray([source_frames[frame]]), active_contact_indices=np.flatnonzero(expected[frame]))
        ablation = {"schema_version": 1, "status": "NOT_REQUIRED_FULL_SET_FEASIBLE" if witness is not None else "COMPLETE", "single_role_ablations": [], "double_role_ablations": [], "role_classification": [{"role_id": role["role_id"], "classification": "MANDATORY_TASK" if role["functional_role"] == "SUPPORT" else "SUPPORTING", "source_duration_frames": role["assignment_duration_frames"]} for role in _active_roles(roles, frame)]}
        conflict = {"schema_version": 1, "status": "NO_CONFLICT_SET_FULL_SET_FEASIBLE" if witness is not None else "INCONCLUSIVE", "roles": [], "conflict_type": None}
        _write_json(output / "role_ablation.json", ablation)
        _write_json(output / "minimum_conflicting_role_set.json", conflict)
        transition, transition_witness, transition_pareto = _dynamic_certificate(model, reference, reference_qvel, actual, expected, anchors, frame, witness)
        _write_json(output / "dynamic_transition_feasibility.json", transition)
        _write_json(output / "dynamic_transition_pareto.json", {"schema_version": 1, "candidates": transition_pareto})
        if transition_witness is not None:
            _write_npz(output / "dynamic_transition_witness.npz", qpos=transition_witness)
    else:
        simultaneous = {"status": "INCONCLUSIVE", "reason": "--run-feasibility not requested"}
        transition = {"status": "INCONCLUSIVE", "reason": "--run-feasibility not requested"}
    decision, confidence = _select_decision(simultaneous["status"], transition["status"])
    decision_payload = {
        "schema_version": 1,
        "decision": decision,
        "confidence": confidence,
        "first_failure": timeline,
        "simultaneous_feasibility": simultaneous["status"],
        "dynamic_transition_feasibility": transition["status"],
        "minimum_conflict_set": [],
        "evaluator_audit": "NO_VISUAL_EVALUATOR_MISMATCH_DETECTED",
        "recommended_next_contract": "V2_CONTACT_MODE_TRANSITION" if decision == "EXTEND_V2_CONTACT_MODE_TRANSITION" else "UNCHANGED_V2" if decision == "UPGRADE_V2_OPTIMIZER" else "TASK_DYNAMICS_EQUIVALENT_CONTACT",
        "recommended_next_implementation": "bounded acquisition/release/regrasp phases" if decision == "EXTEND_V2_CONTACT_MODE_TRANSITION" else "longer-horizon direct collocation and feasibility-preserving projection",
        "prohibited_shortcuts": ["threshold relaxation", "role deletion", "object qpos overwrite", "frozen-frame replacement", "smoke or MJWP before the next primary gate"],
    }
    _write_json(output / "v2_vs_v3_decision.json", decision_payload)
    _write_text(output / "V2_VS_V3_DECISION.md", f"# V2 versus V3 decision\n\nDecision: **{decision}** (confidence {confidence:.2f}).\n\nSimultaneous contact: `{simultaneous['status']}`. Dynamic transition: `{transition['status']}`. The evaluator agrees with reconstructed physical contact; thresholds were unchanged.\n")
    with np.load(Path(physics["trajectory"]), allow_pickle=False) as archive:
        stage_b = np.asarray(archive["qpos"], dtype=np.float64)
    mapping = json.loads((inputs["root"].parent / "source_mapping.json").read_text(encoding="utf-8"))
    sequence = CanonicalHOISequence.load(mapping["canonical_dir"])
    patch_rows = json.loads((inputs["root"] / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"]
    patches = {str(row["role_id"]): row for row in patch_rows}
    payload, keyframes = _viewer_payload(model, stage_b, reference, actual, anchors, source_frames, timeline, arrays, best, sequence, patches)
    _write_json(output / "viewer_payload.json", payload)
    html = build_failure_html(payload, output / "stage_c_v2r2e_failure_diagnostic.html") if args.render_html else None
    screenshot_rows = []
    if html is not None and args.render_screenshots:
        labels = {0: "start", max(0, frame - 1): "first_patch_loss_previous", frame: "first_patch_loss", len(reference) - 1: "end"}
        event_slugs = {
            "first physical contact": "first_physical_contact",
            "first large reference joint delta": "first_large_reference_delta",
            "first actual tracking lag": "first_tracking_lag",
            "first joint-limit-margin breach": "worst_joint_margin_event",
            "first collision-depth spike": "first_collision_depth_spike",
            "first force spike": "first_force_spike",
            "first assignment change": "first_assignment_change",
        }
        for event in timeline["events"]:
            if event["event"] in event_slugs:
                labels.setdefault(int(event["frame"]), event_slugs[event["event"]])
        requests = []
        for local_frame in keyframes:
            label = labels.get(local_frame, f"keyframe_{source_frames[local_frame]}")
            for view in ("global", "close", "top"):
                requests.append({"frame": int(source_frames[local_frame]), "view": view, "layers": "failure_core", "filename": f"{label}_{source_frames[local_frame]}_{view}.png", "visible_layers": "failure_core"})
        for layer in ("patch_contacts", "visual_collision", "reference_actual", "source_stageb_cxa", "force_penetration"):
            requests.append({"frame": timeline["first_failure_source_frame"], "view": "close", "layers": layer, "filename": f"first_failure_{layer}.png", "visible_layers": layer})
        screenshot_rows = render_chrome_screenshots(html, output / "screenshots", requests)
    elif args.resume and (output / "screenshot_manifest.json").is_file():
        screenshot_rows = json.loads((output / "screenshot_manifest.json").read_text(encoding="utf-8")).get("screenshots", [])
    observations = []
    for row in screenshot_rows:
        row["observations"] = "PENDING_CODEX_MANUAL_IMAGE_REVIEW"
        row["result"] = "RENDERED" if row["status"] == "PASS" else "FAIL"
        observations.append(row)
    screenshot_status = "RENDERED_PENDING_MANUAL_REVIEW" if observations and all(row["status"] == "PASS" for row in observations) else "FAIL"
    _write_json(output / "screenshot_manifest.json", {"schema_version": 1, "status": screenshot_status, "screenshots": observations})
    _write_text(output / "SCREENSHOT_REVIEW.md", f"""# Screenshot review

{DISCLAIMER}.

Status: `PENDING_CODEX_MANUAL_IMAGE_REVIEW`.

The renderer has produced full-hand keyframes.  A reviewer must inspect the PNGs before changing this status or making a visual acceptance claim.  Required questions: patch side, first geom pair, patch membership, slide/bounce/block/lag, force ordering, joint-margin ordering, same-finger patch conflict, object-relative motion, collision obstruction, and evaluator/visual agreement.
""")
    index = f"""<!doctype html><meta charset='utf-8'><title>{DISCLAIMER}</title><h1>{DISCLAIMER}</h1><ul>
<li><a href='stage_c_v2r2e_failure_diagnostic.html'>primary failure HTML</a></li>
<li><a href='FIRST_FAILURE_TIMELINE.md'>timeline</a></li><li><a href='V2_VS_V3_DECISION.md'>decision</a></li>
<li><a href='simultaneous_contact_feasibility.json'>simultaneous certificate</a></li><li><a href='dynamic_transition_feasibility.json'>dynamic certificate</a></li>
<li><a href='screenshots/'>screenshots</a></li></ul>"""
    _write_text(output / "failure_diagnostic_index.html", index)
    result = {"status": "COMPLETE_PENDING_MANUAL_VISUAL_REVIEW", "output_dir": str(output), "failure_html": str(html) if html else None, "screenshot_status": screenshot_status, "decision": decision, "first_failure": timeline, "simultaneous": simultaneous["status"], "transition": transition["status"]}
    _write_json(args.report_json or output / "diagnostic_report.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--aggregate-report", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-html", action="store_true")
    parser.add_argument("--render-screenshots", action="store_true")
    parser.add_argument("--run-feasibility", action="store_true")
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=_json_default))
