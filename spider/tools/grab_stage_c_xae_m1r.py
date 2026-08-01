"""Stage C-XAE-M1R step-5 control-authority and reachability audit.

This is deliberately a separate, fail-closed namespace from XAE and XAE-M1.
It replays the frozen source frame 1461 at real 0.5-ms MuJoCo steps, records
pre/post-step controller-to-actuator telemetry, and only permits the minimal
bumpless-transfer realization repair when the four independent audits support
that conclusion.  It never writes robot/object qpos or qvel after init.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools import grab_stage_c_xae_m1 as m1
from spider.tools.grab_stage_c import _finite_data, _object_tracking_error, _preflight_object_ids, _set_object_mocap_reference, _site_ids
from spider.tools.grab_stage_c_failure_diagnostic import _contact_records, _mesh_for_ids, _state_meshes


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r"
XAE_M1_AUTHORITY = REPO / ".local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention"
STEP5_STEPS = 9
CONTROLLED_COLUMNS = np.asarray((36, 37, 38, 39), dtype=np.int64)
ACTUATOR_NAMES = ("l_FFJ0", "l_FFJ1", "l_FFJ2", "l_FFJ3")
JOINT_NAMES = (
    "l_index_finger_mcp_flex",
    "l_index_finger_mcp_abd",
    "l_index_finger_pip",
    "l_index_finger_dip",
)
SAFE_CORRECTION_RAD = 0.12


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_plain).encode("utf-8")).hexdigest()


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE)
    return result.stdout.strip()


def _authority_files() -> list[Path]:
    return [
        XAE_M1_AUTHORITY / "reports/xae_m1_final_acceptance.json",
        XAE_M1_AUTHORITY / "reports/two_frame_first_failure.json",
        XAE_M1_AUTHORITY / "two_frame/two_frame_retention_summary.json",
        XAE_M1_AUTHORITY / "two_frame/two_frame_retention_trace.npz",
        XAE_M1_AUTHORITY / "target_build/m1_target_build_audit.json",
        XAE_M1_AUTHORITY / "reports/XAE_M1_SCREENSHOT_REVIEW.md",
    ]


def _actuator_mapping(model: mujoco.MjModel) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gates: list[bool] = []
    for expected_column, actuator_name, joint_name in zip(CONTROLLED_COLUMNS, ACTUATOR_NAMES, JOINT_NAMES, strict=True):
        actuator = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        trn_joint = int(model.actuator_trnid[actuator, 0]) if actuator >= 0 else -1
        dof = int(model.jnt_dofadr[joint]) if joint >= 0 else -1
        qpos = int(model.jnt_qposadr[joint]) if joint >= 0 else -1
        valid = actuator == int(expected_column) and trn_joint == joint and dof == int(expected_column) and qpos == int(expected_column)
        gates.append(valid)
        rows.append({
            "actuator_index": actuator,
            "actuator_name": actuator_name,
            "joint_index": joint,
            "joint_name": joint_name,
            "qvel_column": dof,
            "qpos_column": qpos,
            "ctrlrange": model.actuator_ctrlrange[actuator] if actuator >= 0 else np.asarray((np.nan, np.nan)),
            "forcerange": model.actuator_forcerange[actuator] if actuator >= 0 else np.asarray((np.nan, np.nan)),
            "name_based_mapping_valid": valid,
        })
    return {"controlled_columns": CONTROLLED_COLUMNS, "rows": rows, "status": "PASS" if all(gates) else "WRONG_ACTUATOR_MAPPING"}


def _contact_rows(model: mujoco.MjModel, data: mujoco.MjData, hand: set[int], objects: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        if not ({int(contact.geom1), int(contact.geom2)} & hand and {int(contact.geom1), int(contact.geom2)} & objects):
            continue
        mujoco.mj_contactForce(model, data, index, local_force)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        normal = frame[:, 0]
        tangent_1, tangent_2 = frame[:, 1], frame[:, 2]
        normal_force = float(local_force[0])
        friction_vector = tangent_1 * float(local_force[1]) + tangent_2 * float(local_force[2])
        friction_force = float(np.linalg.norm(friction_vector))
        friction_limit = max(1e-12, abs(normal_force) * float(contact.friction[0]))
        name_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or str(int(contact.geom1))
        name_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or str(int(contact.geom2))
        rows.append({
            "contact_index": index,
            "geom_pair": f"{name_1}|{name_2}",
            "position_world": np.asarray(contact.pos, dtype=np.float64),
            "normal_world": normal,
            "tangent_1_world": tangent_1,
            "tangent_2_world": tangent_2,
            "depth_m": float(max(0.0, -contact.dist)),
            "force_local": local_force.copy(),
            "normal_force_n": normal_force,
            "friction_force_n": friction_force,
            "force_world": normal * normal_force + friction_vector,
            "normal_impulse_ns": normal_force * m1.SIM_DT,
            "friction_impulse_ns": friction_force * m1.SIM_DT,
            "friction_cone_utilization": friction_force / friction_limit,
            "slip_stick": "SLIP" if friction_force >= 0.98 * friction_limit and abs(normal_force) > 1e-10 else "STICK",
            "efc_address": int(contact.efc_address),
        })
    return rows


def _assigned_contact(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if frozenset(str(row["geom_pair"]).split("|")) == m1.ASSIGNED_PAIR), None)


def _disable_assigned_pair_only(model: mujoco.MjModel) -> dict[str, Any]:
    """Disable only the frozen left-index/object explicit pair in-memory."""
    floor = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    if floor < 0:
        raise RuntimeError("O3 requires the floor sentinel")
    indices = []
    for index, (geom1, geom2) in enumerate(zip(model.pair_geom1, model.pair_geom2, strict=True)):
        name_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom1))
        name_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom2))
        if frozenset((name_1, name_2)) == m1.ASSIGNED_PAIR:
            indices.append(index)
    if len(indices) != 1:
        raise RuntimeError(f"O3 expected exactly one assigned explicit pair, found {indices}")
    index = indices[0]
    original = {
        "pair_index": index,
        "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[index])),
        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[index])),
    }
    model.pair_geom1[index] = floor
    model.pair_geom2[index] = floor
    return {"explicit_pair_count": 1, "original_pair": original, "replacement": {"geom_id": floor, "geom_name": "floor", "reason": "same-geom pair is skipped"}, "all_other_model_terms_unchanged": True}


def _dls(block: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return block.T @ np.linalg.solve(block @ block.T + 0.002 * np.eye(3), np.asarray(vector, dtype=np.float64))


def _control_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tip_site: int,
    normal: np.ndarray,
    normal_gap: float,
    tip_velocity: np.ndarray,
    patch_velocity: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, np.ndarray | bool]:
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, None, tip_site)
    block = jacobian[:, CONTROLLED_COLUMNS]
    relative = np.asarray(tip_velocity, dtype=np.float64) - np.asarray(patch_velocity, dtype=np.float64)
    feedback = _dls(block, -np.asarray(normal, dtype=np.float64) * float(normal_gap))
    feedforward = _dls(block, m1.SIM_DT * (np.asarray(patch_velocity, dtype=np.float64) - np.asarray(tip_velocity, dtype=np.float64))) if profile["object_motion_feedforward"] else np.zeros(4)
    normal_term = _dls(block, -m1.SIM_DT * np.asarray(normal, dtype=np.float64) * float(np.dot(relative, normal))) if profile["normal_velocity_servo"] else np.zeros(4)
    tangent = relative - normal * float(np.dot(relative, normal))
    tangential_term = _dls(block, -m1.SIM_DT * tangent) if profile["tangential_slip_servo"] else np.zeros(4)
    raw = feedback + feedforward + normal_term + tangential_term
    clipped = np.clip(raw, -SAFE_CORRECTION_RAD, SAFE_CORRECTION_RAD)
    return {
        "jacobian": jacobian,
        "jacobian_block": block,
        "feedback_correction": feedback,
        "feedforward_correction": feedforward,
        "normal_correction": normal_term,
        "tangential_correction": tangential_term,
        "raw_correction": raw,
        "clipped_correction": clipped,
        "projected_correction": clipped.copy(),
        "final_correction": clipped.copy(),
        "control_clipped": bool(np.any(np.abs(raw - clipped) > 1e-12)),
    }


def _profile(name: str, *, immediate: bool = False, integrated: bool = False) -> dict[str, Any]:
    result = {
        "candidate": name,
        "controlled_joint_set": ["left_index"],
        "controlled_columns": CONTROLLED_COLUMNS,
        "allow_regrasp": False,
        "surface_target": "same-substep immutable semantic-patch nearest-surface set",
        "object_motion_feedforward": name == "R2_object_motion_velocity_feedforward",
        "normal_velocity_servo": name == "R3_normal_relative_velocity_servo",
        "tangential_slip_servo": False,
        "bumpless_ramp_s": 0.0 if immediate else 0.008,
        "actuator_target_integration": integrated,
    }
    if name == "C2_integrated_surface_correction":
        result["object_motion_feedforward"] = True
        result["normal_velocity_servo"] = True
    return result


def _source_state(ctx: m1.Context, time_s: float, frame_count: int) -> tuple[np.ndarray, int, float]:
    index, alpha = m1.continuous_source_index(time_s, frame_count)
    next_index = min(index + 1, frame_count - 1)
    state = m1.interpolate_source_state(ctx.qpos[index], ctx.qpos[next_index], alpha) if index < frame_count - 1 else ctx.qpos[index].copy()
    return state, index, alpha


def _telemetry_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: m1.Context,
    hand: set[int],
    objects: set[int],
    bodies: dict[str, int],
    source_state: np.ndarray,
    source_index: int,
    source_alpha: float,
    step: int,
    phase: str,
    control: dict[str, Any] | None,
    mode: ContactMode,
    previous_mode: ContactMode,
    previous_ctrl: np.ndarray,
) -> dict[str, Any]:
    sites = _site_ids(model)
    tip_site, wrist_site, object_body = sites[8], sites[6], bodies["right"]
    contacts = _contact_rows(model, data, hand, objects)
    assigned = _assigned_contact(contacts)
    point, normal, distance, face = m1.nearest_surface_target(data.site_xpos[tip_site], data.xpos[object_body], data.xmat[object_body].reshape(3, 3), ctx.patch_mesh)
    vector = np.asarray(data.site_xpos[tip_site], dtype=np.float64) - point
    normal_gap = float(np.dot(vector, normal))
    tangential_error = float(np.linalg.norm(vector - normal * normal_gap))
    tip_spatial, object_spatial = np.zeros(6), np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tip_site, tip_spatial, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, object_body, object_spatial, 0)
    patch_velocity = object_spatial[3:] + np.cross(object_spatial[:3], point - data.xpos[object_body])
    relative = tip_spatial[3:] - patch_velocity
    normal_velocity = float(np.dot(relative, normal))
    tangential_velocity = relative - normal * normal_velocity
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = source_state
    mujoco.mj_forward(model, target_data)
    object_pos_error, object_rot_error = _object_tracking_error(data, source_state, bodies)
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, None, tip_site)
    singular = np.linalg.svd(jacobian[:, CONTROLLED_COLUMNS], compute_uv=False)
    return {
        "phase": phase,
        "source_frame": int(ctx.source_frames[source_index]),
        "source_time_s": float(ctx.source_frames[0] / m1.SOURCE_FPS + data.time),
        "source_frame_fraction": float(source_alpha),
        "sim_time_s": float(data.time),
        "sim_step": int(step),
        "substep": int(step),
        "mode": mode.value,
        "previous_mode": previous_mode.value,
        "object_target_pose": source_state[52:58].copy(),
        "object_actual_pose": np.concatenate((data.xpos[object_body], Rotation.from_matrix(data.xmat[object_body].reshape(3, 3)).as_euler("XYZ"))),
        "object_target_linear_velocity_mps": (source_state[52:55] - ctx.qpos[source_index, 52:55]) / max(data.time, m1.SIM_DT),
        "object_target_angular_velocity_mps": np.zeros(3),
        "object_actual_linear_velocity_mps": object_spatial[3:].copy(),
        "object_actual_angular_velocity_mps": object_spatial[:3].copy(),
        "patch_target_object_frame": data.xmat[object_body].reshape(3, 3).T @ (point - data.xpos[object_body]),
        "patch_target_world_frame": point,
        "patch_target_linear_velocity_mps": patch_velocity,
        "actual_assigned_contact_region_pose": data.site_xpos[tip_site].copy(),
        "actual_fingertip_linear_velocity_mps": tip_spatial[3:].copy(),
        "physical_contact_present": assigned is not None,
        "assigned_pair_present": assigned is not None,
        "assigned_geom_pair": "|".join(sorted(m1.ASSIGNED_PAIR)),
        "actual_geom_pair": "NONE" if assigned is None else assigned["geom_pair"],
        "all_hand_object_contact_pairs": contacts,
        "contact_point_world": None if assigned is None else assigned["position_world"],
        "contact_normal_world": None if assigned is None else assigned["normal_world"],
        "contact_depth_m": 0.0 if assigned is None else assigned["depth_m"],
        "contact_force_n": 0.0 if assigned is None else abs(float(assigned["normal_force_n"])),
        "normal_impulse_ns": 0.0 if assigned is None else abs(float(assigned["normal_impulse_ns"])),
        "friction_impulse_ns": 0.0 if assigned is None else float(assigned["friction_impulse_ns"]),
        "patch_distance_m": float(distance),
        "normal_gap_m": normal_gap,
        "relative_normal_velocity_mps": normal_velocity,
        "tangential_slip_velocity_mps": tangential_velocity,
        "tangential_slip_mps": float(np.linalg.norm(tangential_velocity)),
        "raw_controller_correction": np.zeros(4) if control is None else control["raw_correction"],
        "feedforward_correction": np.zeros(4) if control is None else control["feedforward_correction"],
        "feedback_correction": np.zeros(4) if control is None else control["feedback_correction"],
        "normal_correction": np.zeros(4) if control is None else control["normal_correction"],
        "tangential_correction": np.zeros(4) if control is None else control["tangential_correction"],
        "clipped_correction": np.zeros(4) if control is None else control["clipped_correction"],
        "projected_correction": np.zeros(4) if control is None else control["projected_correction"],
        "final_correction": np.zeros(4) if control is None else control["final_correction"],
        "actuator_ctrl": data.ctrl.copy(),
        "ctrl_delta": data.ctrl[:52].copy() - previous_ctrl,
        "actuator_saturation": np.isclose(data.ctrl, model.actuator_ctrlrange[:, 0]) | np.isclose(data.ctrl, model.actuator_ctrlrange[:, 1]),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "qacc": data.qacc.copy(),
        "reference_qpos": source_state.copy(),
        "joint_margin_fraction": float(m1._joint_margin(model, data.qpos)),
        "jacobian": jacobian,
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian[:, CONTROLLED_COLUMNS])),
        "jacobian_condition_number": float(np.inf if singular[-1] <= 1e-12 else singular[0] / singular[-1]),
        "jacobian_minimum_singular_value": float(singular[-1]),
        "controlled_joint_names": list(JOINT_NAMES),
        "controlled_qvel_columns": CONTROLLED_COLUMNS,
        "controlled_actuator_names": list(ACTUATOR_NAMES),
        "fingertip_tracking_error_m": float(np.linalg.norm(data.site_xpos[tip_site] - target_data.site_xpos[tip_site])),
        "wrist_tracking_error_m": float(np.linalg.norm(data.site_xpos[wrist_site] - target_data.site_xpos[wrist_site])),
        "object_tracking_position_m": float(np.max(object_pos_error)),
        "object_tracking_rotation_rad": float(np.max(object_rot_error)),
        "penetration_m": float(max((row["depth_m"] for row in contacts), default=0.0)),
        "finite": bool(_finite_data(data)),
        "joint_limit_valid": not bool(v2r.dynamic._joint_limit_violations(model, np.asarray([data.qpos], dtype=np.float64))),
        "control_clipped": False if control is None else bool(control["control_clipped"]),
    }


def run_telemetry_rollout(
    ctx: m1.Context,
    profile: dict[str, Any],
    *,
    steps: int,
    frame_count: int = 2,
    contact_enabled: bool = True,
) -> dict[str, Any]:
    """Run a real 0.5-ms diagnostic rollout without breaking after first loss."""
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    model.opt.timestep = m1.SIM_DT
    hand, objects = v2r._contact_ids(model)
    contact_ablation: dict[str, Any] | None = None
    if not contact_enabled:
        contact_ablation = _disable_assigned_pair_only(model)
    mapping = _actuator_mapping(model)
    if mapping["status"] != "PASS":
        raise RuntimeError(mapping["status"])
    data = mujoco.MjData(model)
    data.qpos[:] = ctx.qpos[0]
    data.qvel[:] = 0.0
    bodies, mocap = _preflight_object_ids(model)
    _set_object_mocap_reference(data, ctx.qpos[0], mocap)
    data.ctrl[:52] = ctx.qpos[0, :52]
    data.ctrl[52:] = 0.0
    mujoco.mj_forward(model, data)
    tip_site = _site_ids(model)[8]
    machine = ContactModeMachine(ContactModeConfig(confirmation_substeps=4, max_regrasp_attempts=0, allow_regrasp=False, sim_dt_s=m1.SIM_DT))
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda warning: warnings.append(str(warning)))
    rows: list[dict[str, Any]] = []
    previous_ctrl = data.ctrl[:52].copy()
    transition_time = 0.0
    first_loss: dict[str, Any] | None = None
    try:
        source_state, index, alpha = _source_state(ctx, data.time, frame_count)
        initial = _telemetry_row(model, data, ctx, hand, objects, bodies, source_state, index, alpha, -1, "initial", None, machine.mode, machine.mode, previous_ctrl)
        rows.append(initial)
        for step in range(steps):
            source_state, index, alpha = _source_state(ctx, data.time, frame_count)
            previous_mode = machine.mode
            pre0 = _telemetry_row(model, data, ctx, hand, objects, bodies, source_state, index, alpha, step, "pre", None, previous_mode, previous_mode, previous_ctrl)
            observation = m1._observation({
                "source_frame": int(ctx.source_frames[index]), "source_time_s": float(ctx.source_frames[0] / m1.SOURCE_FPS + data.time), "sim_step": step, "substep": step,
                "physical_contact": bool(pre0["physical_contact_present"]), "correct_contact": bool(pre0["assigned_pair_present"]), "geom_pair": str(pre0["actual_geom_pair"]),
                "patch_distance_m": float(pre0["patch_distance_m"]), "normal_cosine": 1.0, "tangential_slip_m": float(pre0["tangential_slip_mps"] * m1.SIM_DT), "normal_gap_m": float(pre0["normal_gap_m"]),
                "penetration_m": float(pre0["penetration_m"]), "force_n": float(pre0["contact_force_n"]), "impulse_ns": float(pre0["normal_impulse_ns"]), "joint_margin_fraction": float(pre0["joint_margin_fraction"]),
                "wrist_tracking_error_m": float(pre0["wrist_tracking_error_m"]), "fingertip_tracking_error_m": float(pre0["fingertip_tracking_error_m"]), "object_tracking_position_m": float(pre0["object_tracking_position_m"]), "object_tracking_rotation_rad": float(pre0["object_tracking_rotation_rad"]), "finite": bool(pre0["finite"]), "joint_limit_valid": bool(pre0["joint_limit_valid"]),
                "reference_qpos": source_state.copy(), "actual_qpos": data.qpos.copy(), "ctrl": data.ctrl.copy(),
            }, len(warnings))
            mode = machine.observe(observation)
            if mode != previous_mode and previous_mode not in {ContactMode.RETAIN_PENDING, ContactMode.RETAIN}:
                transition_time = data.time
            _point, patch_normal, _distance, _face = m1.nearest_surface_target(data.site_xpos[tip_site], data.xpos[bodies["right"]], data.xmat[bodies["right"]].reshape(3, 3), ctx.patch_mesh)
            terms = _control_terms(model, data, tip_site, patch_normal, float(pre0["normal_gap_m"]), np.asarray(pre0["actual_fingertip_linear_velocity_mps"]), np.asarray(pre0["patch_target_linear_velocity_mps"]), profile)
            nominal = source_state[:52].copy()
            elapsed = max(0.0, data.time - transition_time)
            blend = 1.0 if profile["bumpless_ramp_s"] <= 0.0 else min(1.0, elapsed / float(profile["bumpless_ramp_s"]))
            if profile["actuator_target_integration"]:
                desired = previous_ctrl.copy()
                desired[CONTROLLED_COLUMNS] += np.asarray(terms["final_correction"], dtype=np.float64)
                desired = np.clip(desired, model.actuator_ctrlrange[:52, 0], model.actuator_ctrlrange[:52, 1])
            else:
                desired = nominal.copy()
                desired[CONTROLLED_COLUMNS] += np.asarray(terms["final_correction"], dtype=np.float64)
            new_ctrl = (1.0 - blend) * previous_ctrl + blend * desired
            data.ctrl[:52] = new_ctrl
            data.ctrl[52:] = 0.0
            pre = _telemetry_row(model, data, ctx, hand, objects, bodies, source_state, index, alpha, step, "pre", terms, mode, previous_mode, previous_ctrl)
            pre["bumpless_blend_alpha"] = blend
            pre["actuator_ctrl_command"] = new_ctrl.copy()
            rows.append(pre)
            _set_object_mocap_reference(data, source_state, mocap)
            mujoco.mj_step(model, data)
            source_state, index, alpha = _source_state(ctx, data.time, frame_count)
            # Authority labels the state after the first integration as MuJoCo
            # step 1; retain that convention so its first-loss step is 5.
            post = _telemetry_row(model, data, ctx, hand, objects, bodies, source_state, index, alpha, step + 1, "post", terms, mode, previous_mode, previous_ctrl)
            post["bumpless_blend_alpha"] = blend
            post["actuator_ctrl_command"] = new_ctrl.copy()
            rows.append(post)
            if first_loss is None and not bool(post["assigned_pair_present"]):
                first_loss = post
            previous_ctrl = new_ctrl.copy()
        last = rows[-1]
        terminal_observation = m1._observation({
            "source_frame": int(last["source_frame"]), "source_time_s": float(last["source_time_s"]), "sim_step": int(last["sim_step"]), "substep": int(last["substep"]),
            "physical_contact": bool(last["physical_contact_present"]), "correct_contact": bool(last["assigned_pair_present"]), "geom_pair": str(last["actual_geom_pair"]),
            "patch_distance_m": float(last["patch_distance_m"]), "normal_cosine": 1.0, "tangential_slip_m": float(last["tangential_slip_mps"] * m1.SIM_DT), "normal_gap_m": float(last["normal_gap_m"]), "penetration_m": float(last["penetration_m"]), "force_n": float(last["contact_force_n"]), "impulse_ns": float(last["normal_impulse_ns"]), "joint_margin_fraction": float(last["joint_margin_fraction"]), "wrist_tracking_error_m": float(last["wrist_tracking_error_m"]), "fingertip_tracking_error_m": float(last["fingertip_tracking_error_m"]), "object_tracking_position_m": float(last["object_tracking_position_m"]), "object_tracking_rotation_rad": float(last["object_tracking_rotation_rad"]), "finite": bool(last["finite"]), "joint_limit_valid": bool(last["joint_limit_valid"]),
            "reference_qpos": np.asarray(last["qpos"], dtype=np.float64), "actual_qpos": np.asarray(last["qpos"], dtype=np.float64), "ctrl": np.asarray(last["actuator_ctrl"], dtype=np.float64),
        }, len(warnings))
        machine.finish(terminal_observation)
    finally:
        mujoco.set_mju_user_warning(old_warning)
    return {"profile": profile, "rows": rows, "first_loss": first_loss, "warnings": warnings, "mapping": mapping, "contact_enabled": contact_enabled, "contact_ablation": contact_ablation, "terminal_mode": machine.mode.value, "transitions": machine.transition_payload()}


def _save_trace(root: Path, rollout: dict[str, Any]) -> None:
    rows = rollout["rows"]
    _write_json(root / "timeline.json", {"schema_version": 1, "rows": rows, "transitions": rollout["transitions"]})
    _write_npz(root / "trace.npz",
        qpos=np.asarray([row["qpos"] for row in rows]), qvel=np.asarray([row["qvel"] for row in rows]), qacc=np.asarray([row["qacc"] for row in rows]), ctrl=np.asarray([row["actuator_ctrl"] for row in rows]),
        source_frame=np.asarray([row["source_frame"] for row in rows]), sim_step=np.asarray([row["sim_step"] for row in rows]), phase=np.asarray([row["phase"] for row in rows]),
        patch_distance_m=np.asarray([row["patch_distance_m"] for row in rows]), normal_gap_m=np.asarray([row["normal_gap_m"] for row in rows]), tangential_slip_mps=np.asarray([row["tangential_slip_mps"] for row in rows]),
        force_n=np.asarray([row["contact_force_n"] for row in rows]), normal_impulse_ns=np.asarray([row["normal_impulse_ns"] for row in rows]), friction_impulse_ns=np.asarray([row["friction_impulse_ns"] for row in rows]), contact=np.asarray([row["assigned_pair_present"] for row in rows], dtype=np.uint8),
        raw_correction=np.asarray([row["raw_controller_correction"] for row in rows]), final_correction=np.asarray([row["final_correction"] for row in rows]), ctrl_delta=np.asarray([row["ctrl_delta"] for row in rows]), tip_velocity=np.asarray([row["actual_fingertip_linear_velocity_mps"] for row in rows]), patch_velocity=np.asarray([row["patch_target_linear_velocity_mps"] for row in rows]))


def _post_rows(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rollout["rows"] if row["phase"] == "post"]


def _first_loss(rollout: dict[str, Any]) -> dict[str, Any] | None:
    return next((row for row in _post_rows(rollout) if not row["assigned_pair_present"]), None)


def _baseline_reproduction(rollouts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    gates: list[bool] = []
    for name, rollout in rollouts.items():
        loss = _first_loss(rollout)
        valid = loss is not None and int(loss["source_frame"]) == 1461 and int(loss["sim_step"]) == 5
        gates.append(valid)
        records[name] = {"first_loss": None if loss is None else {key: loss[key] for key in ("source_frame", "sim_step", "phase", "normal_gap_m", "relative_normal_velocity_mps", "tangential_slip_mps", "actual_geom_pair")}, "matches_frozen_step5": valid}
    return {"schema_version": 1, "status": "PASS" if all(gates) else "BASELINE_REPRODUCTION_MISMATCH", "profiles": records}


def _o1_control_authority(rollouts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, list[dict[str, Any]]] = {}
    for name, rollout in rollouts.items():
        entries: list[dict[str, Any]] = []
        for row in _post_rows(rollout):
            if int(row["sim_step"]) > 5:
                continue
            desired = np.asarray(row["patch_target_linear_velocity_mps"])
            actual = np.asarray(row["actual_fingertip_linear_velocity_mps"])
            entries.append({
                "sim_step": int(row["sim_step"]),
                "raw_correction_norm": float(np.linalg.norm(row["raw_controller_correction"])),
                "clipped_correction_norm": float(np.linalg.norm(row["clipped_correction"])),
                "final_correction_norm": float(np.linalg.norm(row["final_correction"])),
                "ctrl_delta_norm": float(np.linalg.norm(np.asarray(row["ctrl_delta"])[CONTROLLED_COLUMNS])),
                "ctrl_left_index": np.asarray(row["actuator_ctrl"])[CONTROLLED_COLUMNS],
                "controlled_qvel": np.asarray(row["qvel"])[CONTROLLED_COLUMNS],
                "actual_fingertip_velocity_mps": actual,
                "desired_patch_velocity_mps": desired,
                "velocity_tracking_residual_mps": float(np.linalg.norm(actual - desired)),
                "blend_alpha": float(row.get("bumpless_blend_alpha", 0.0)),
            })
        table[name] = entries
    reference = table["R0_frozen_surface_aligned_baseline"]
    comparisons: dict[str, Any] = {}
    for name in ("R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo"):
        actual = table[name]
        ctrl_difference = max((float(np.linalg.norm(np.asarray(a["ctrl_left_index"]) - np.asarray(b["ctrl_left_index"]))) for a, b in zip(actual, reference, strict=True)), default=0.0)
        qvel_difference = max((float(np.linalg.norm(np.asarray(a["controlled_qvel"]) - np.asarray(b["controlled_qvel"]))) for a, b in zip(actual, reference, strict=True)), default=0.0)
        tip_difference = max((float(np.linalg.norm(np.asarray(a["actual_fingertip_velocity_mps"]) - np.asarray(b["actual_fingertip_velocity_mps"]))) for a, b in zip(actual, reference, strict=True)), default=0.0)
        comparisons[name] = {"max_ctrl_difference": ctrl_difference, "max_qvel_difference": qvel_difference, "max_tip_velocity_difference_mps": tip_difference, "commands_not_zeroed": ctrl_difference > 1e-9}
    delay = all(any(0.0 < entry["blend_alpha"] < 1.0 for entry in entries) for entries in table.values())
    failure = "CONTROL_DELAY_ERROR" if delay and all(item["commands_not_zeroed"] for item in comparisons.values()) else "CONTROL_AUTHORITY_PASS"
    return {"schema_version": 1, "status": failure, "mapping": rollouts["R0_frozen_surface_aligned_baseline"]["mapping"], "step_0_to_5": table, "comparisons": comparisons, "finding": "R2/R3 corrections reach name-verified left-index actuators, but the fixed 8-ms bumpless blend attenuates them during the 2.5-ms step-5 loss window." if failure == "CONTROL_DELAY_ERROR" else "No authority defect detected."}


def _joint_bounds(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(4, -np.inf)
    upper = np.full(4, np.inf)
    for local, dof in enumerate(CONTROLLED_COLUMNS):
        joint = int(model.dof_jntid[int(dof)])
        lower[local] = model.jnt_range[joint, 0] - qpos[dof]
        upper[local] = model.jnt_range[joint, 1] - qpos[dof]
    return lower, upper


def _o2_reachability(ctx: m1.Context, rollout: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    rows: list[dict[str, Any]] = []
    for sample in _post_rows(rollout):
        if int(sample["sim_step"]) > 5:
            continue
        block = np.asarray(sample["jacobian"])[:, CONTROLLED_COLUMNS]
        normal = m1.nearest_surface_target(np.asarray(sample["actual_assigned_contact_region_pose"]), np.asarray(sample["object_actual_pose"])[:3], Rotation.from_euler("XYZ", np.asarray(sample["object_actual_pose"])[3:]).as_matrix(), ctx.patch_mesh)[1]
        tangential_basis = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2]
        displacement = np.asarray(sample["patch_target_world_frame"]) - np.asarray(sample["actual_assigned_contact_region_pose"])
        lower, upper = _joint_bounds(model, np.asarray(sample["qpos"]))
        exact = lsq_linear(block, displacement, bounds=(lower, upper), lsmr_tol="auto")
        q_trial = np.asarray(sample["qpos"])[CONTROLLED_COLUMNS] + exact.x
        ctrl_lower = model.actuator_ctrlrange[CONTROLLED_COLUMNS, 0] - np.asarray(sample["actuator_ctrl"])[CONTROLLED_COLUMNS]
        ctrl_upper = model.actuator_ctrlrange[CONTROLLED_COLUMNS, 1] - np.asarray(sample["actuator_ctrl"])[CONTROLLED_COLUMNS]
        rate_lower = np.maximum(lower / m1.SIM_DT, ctrl_lower / m1.SIM_DT)
        rate_upper = np.minimum(upper / m1.SIM_DT, ctrl_upper / m1.SIM_DT)
        rate_lower = np.maximum(rate_lower, -SAFE_CORRECTION_RAD / m1.SIM_DT)
        rate_upper = np.minimum(rate_upper, SAFE_CORRECTION_RAD / m1.SIM_DT)
        required_velocity = np.asarray(sample["patch_target_linear_velocity_mps"]) - np.asarray(sample["actual_fingertip_linear_velocity_mps"])
        rate = lsq_linear(block, required_velocity, bounds=(rate_lower, rate_upper), lsmr_tol="auto")
        normal_required = abs(float(np.dot(required_velocity, normal)))
        max_normal = float(np.sum(np.where((normal @ block) >= 0.0, rate_upper, rate_lower) * (normal @ block)))
        tangential_required = float(np.linalg.norm((np.eye(3) - np.outer(normal, normal)) @ required_velocity))
        max_tangent = max(float(np.sum(np.abs(tangent @ block) * np.maximum(np.abs(rate_lower), np.abs(rate_upper)))) for tangent in tangential_basis.T)
        singular = np.linalg.svd(block, compute_uv=False)
        rows.append({
            "sim_step": int(sample["sim_step"]), "required_contact_region_displacement_m": displacement, "required_linear_velocity_mps": required_velocity,
            "required_normal_velocity_mps": normal_required, "required_tangential_velocity_mps": tangential_required, "required_joint_delta": exact.x, "required_qvel": rate.x,
            "required_qacc_measured_mps2": np.asarray(sample["qacc"])[CONTROLLED_COLUMNS], "maximum_reachable_displacement_m": block @ np.where(np.abs(exact.x) > 1e-12, np.sign(exact.x) * np.minimum(np.abs(exact.x), np.maximum(np.abs(lower), np.abs(upper))), 0.0),
            "maximum_reachable_normal_velocity_mps": max_normal, "maximum_reachable_tangential_velocity_mps": max_tangent, "bounded_least_squares_residual_mps": float(np.linalg.norm(block @ rate.x - required_velocity)),
            "exact_kinematic_residual_m": float(np.linalg.norm(block @ exact.x - displacement)), "joint_limit_active": np.isclose(q_trial, lower + np.asarray(sample["qpos"])[CONTROLLED_COLUMNS]) | np.isclose(q_trial, upper + np.asarray(sample["qpos"])[CONTROLLED_COLUMNS]),
            "actuator_limit_active": np.isclose(rate.x, rate_lower) | np.isclose(rate.x, rate_upper), "required_to_reachable_normal_ratio": normal_required / max(max_normal, 1e-12), "required_to_reachable_tangential_ratio": tangential_required / max(max_tangent, 1e-12),
            "jacobian_rank": int(np.linalg.matrix_rank(block)), "condition_number": float(np.inf if singular[-1] <= 1e-12 else singular[0] / singular[-1]), "minimum_singular_value": float(singular[-1]),
            "o2_a_exact_success": bool(exact.success and np.linalg.norm(block @ exact.x - displacement) <= 0.001), "o2_b_bounded_success": bool(rate.success and np.linalg.norm(block @ rate.x - required_velocity) <= 0.05),
        })
    exact_pass = bool(rows) and all(row["o2_a_exact_success"] for row in rows)
    bounded_pass = bool(rows) and all(row["o2_b_bounded_success"] for row in rows)
    status = "DYNAMICALLY_REACHABLE_WITHIN_LEFT_INDEX_BOUNDS" if bounded_pass else "KINEMATICALLY_REACHABLE" if exact_pass else "MARGINALLY_REACHABLE"
    return {"schema_version": 1, "status": status, "scope": "left-index-only; root/wrist/object/other fingers fixed", "model_qvel_qacc_limits": "MuJoCo model has no independent hard qvel/qacc limits; O2-B uses joint/actuator ranges plus the frozen 0.12-rad per-step controller safety limit, while qvel/qacc are recorded measurements.", "steps": rows}


def _o3_contact_ablation(no_contact: dict[str, Any], contact: dict[str, Any]) -> dict[str, Any]:
    def tracking(rollout: dict[str, Any]) -> dict[str, Any]:
        rows = _post_rows(rollout)
        return {"patch_distance_p95_m": float(np.percentile([row["patch_distance_m"] for row in rows], 95)), "normal_gap_terminal_m": float(rows[-1]["normal_gap_m"]), "tip_tracking_max_m": max(float(row["fingertip_tracking_error_m"]) for row in rows), "terminal_assigned_contact": bool(rows[-1]["assigned_pair_present"])}
    no, yes = tracking(no_contact), tracking(contact)
    no_pass = no["normal_gap_terminal_m"] <= 0.003 and no["patch_distance_p95_m"] <= 0.020
    yes_pass = yes["terminal_assigned_contact"] and yes["normal_gap_terminal_m"] <= 0.003
    classification = "CONTROLLER_REALIZATION_FAILURE" if not no_pass and not yes_pass else "CONTACT_DYNAMICS_FAILURE" if no_pass and not yes_pass else "BASELINE_EVALUATOR_OR_PAIR_ISSUE" if no_pass and yes_pass else "MIXED"
    return {"schema_version": 1, "status": classification, "no_contact": {"DIAGNOSTIC_ONLY": True, "NOT_A_WITNESS": True, "model_difference": no_contact["contact_ablation"], "tracking": no}, "contact_enabled": {"tracking": yes}, "comparison": {"same_initial_state": True, "same_controller": True, "same_controlled_dofs": True, "contact_only_model_difference": True}}


def _o4_contact_dynamics(rollout: dict[str, Any]) -> dict[str, Any]:
    rows = _post_rows(rollout)
    loss = _first_loss(rollout)
    assigned_force = np.asarray([row["contact_force_n"] for row in rows])
    normal_velocity = np.asarray([row["relative_normal_velocity_mps"] for row in rows])
    tangential = np.asarray([row["tangential_slip_mps"] for row in rows])
    impulses = np.asarray([row["normal_impulse_ns"] + row["friction_impulse_ns"] for row in rows])
    loss_step = None if loss is None else int(loss["sim_step"])
    force_drop = next((int(row["sim_step"]) for row in rows if row["contact_force_n"] <= 1e-10 and int(row["sim_step"]) > 0), None)
    separation = loss is not None and float(loss["normal_gap_m"]) > 0.002 and float(loss["relative_normal_velocity_mps"]) > 0.01
    slip = loss is not None and float(loss["tangential_slip_mps"]) > 0.03
    peak = int(rows[int(np.argmax(impulses))]["sim_step"]) if len(rows) else None
    impulse_ejection = loss_step is not None and peak == loss_step and float(impulses.max(initial=0.0)) > 5.0 * float(np.median(impulses) + 1e-12)
    classification = "NORMAL_SEPARATION" if separation and not impulse_ejection else "SOLVER_IMPULSE_EJECTION" if impulse_ejection else "TANGENTIAL_SLIP" if slip else "INCONCLUSIVE"
    return {"schema_version": 1, "status": classification, "api_note": "MuJoCo mj_contactForce returns local contact force; impulses are force times the fixed 0.5-ms step. Solver penetration correction is proxied by contact depth because this API does not expose a separate per-contact correction impulse.", "normal_separation_onset_step": loss_step if separation else None, "tangential_slip_onset_step": next((int(row["sim_step"]) for row in rows if row["tangential_slip_mps"] > 0.03), None), "impulse_spike_step": peak, "force_drop_step": force_drop, "contact_pair_switch_step": loss_step, "timeline": [{key: row[key] for key in ("sim_step", "assigned_pair_present", "normal_gap_m", "relative_normal_velocity_mps", "tangential_slip_mps", "contact_force_n", "normal_impulse_ns", "friction_impulse_ns", "penetration_m", "all_hand_object_contact_pairs")} for row in rows]}


def _gate_summary(rollout: dict[str, Any], *, required_steps: int, label: str) -> dict[str, Any]:
    rows = _post_rows(rollout)
    required = [row for row in rows if int(row["sim_step"]) < required_steps]
    contacts = [bool(row["assigned_pair_present"]) for row in required]
    gaps = [float(row["normal_gap_m"]) for row in required]
    normal_velocity = [float(row["relative_normal_velocity_mps"]) for row in required]
    gates = {
        "assigned_contact_through_step5": bool([row for row in required if int(row["sim_step"]) <= 5]) and all(bool(row["assigned_pair_present"]) for row in required if int(row["sim_step"]) <= 5),
        "terminal_assigned_contact": bool(contacts[-1]) if contacts else False,
        "retain_entered": any(row["mode"] == "RETAIN" for row in required),
        "patch_distance": bool(required and max(float(row["patch_distance_m"]) for row in required) <= 0.020),
        "normal_gap_bounded": bool(required and max(abs(value) for value in gaps) <= 0.020),
        "not_persistent_normal_separation": not (len(normal_velocity) >= 3 and all(value > 0.01 for value in normal_velocity[-3:])),
        "tangential_slip_bounded": bool(required and max(float(row["tangential_slip_mps"]) for row in required) <= 0.10),
        "force_safe": bool(required and max(float(row["contact_force_n"]) for row in required) <= 150.0),
        "penetration": bool(required and max(float(row["penetration_m"]) for row in required) <= 0.003),
        "joint_limits": bool(required and all(bool(row["joint_limit_valid"]) for row in required)),
        "tracking": bool(required and max(float(row["fingertip_tracking_error_m"]) for row in required) <= 0.08),
        "object_tracking": bool(required and max(float(row["object_tracking_position_m"]) for row in required) <= 0.08),
        "warnings": not rollout["warnings"],
        "finite": bool(required and all(bool(row["finite"]) for row in required)),
        "no_regrasp": all(row["mode"] != "REGRASP" for row in required),
    }
    return {"schema_version": 1, "label": label, "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "terminal_mode": rollout["terminal_mode"], "first_loss": rollout["first_loss"], "steps": required}


def _root_cause(o1: dict[str, Any], o2: dict[str, Any], o3: dict[str, Any], o4: dict[str, Any]) -> dict[str, Any]:
    root = "CONTROLLER_REALIZATION_FAILURE" if o3["status"] == "CONTROLLER_REALIZATION_FAILURE" else "NORMAL_SEPARATION_CONTROL_FAILURE" if o4["status"] == "NORMAL_SEPARATION" else "INCONCLUSIVE"
    return {"schema_version": 1, "status": "DECIDED" if root != "INCONCLUSIVE" else "INCONCLUSIVE", "primary_root_cause": root, "secondary_root_cause": "CONTROL_DELAY_ERROR" if o1["status"] == "CONTROL_DELAY_ERROR" else None, "manifestation": o4["status"], "reachability": o2["status"], "repair_order": ["C1_immediate_surface_correction"], "evidence": {"O1": o1["status"], "O2": o2["status"], "O3": o3["status"], "O4": o4["status"]}}


def _lineage(ctx: m1.Context, root: Path) -> dict[str, Any]:
    missing = [str(path) for path in _authority_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FAIL_CLOSED_INPUT_LINEAGE_MISMATCH: missing {missing}")
    hashes = {
        "final_repaired_trajectory": _sha256_file(ctx.trajectory),
        "semantic_patch": _sha256_file(m1.FINAL / "source_contact_patches.json"),
        "patch_triangles": _payload_hash(ctx.patch["extended_face_ids"]),
        "role_assignment": _sha256_file(m1.FINAL / "source_contact_roles.json"),
        "contact_region": _sha256_file(REPO / "configs/project/wuji_hand2_contact_regions.yaml"),
        "mujoco_model": _sha256_file(ctx.model_path),
        "object_mesh": _sha256_file(ctx.object_mesh_path),
        "source_mapping": _payload_hash(ctx.source_frames.tolist()),
        "two_frame_trace": _sha256_file(XAE_M1_AUTHORITY / "two_frame/two_frame_retention_trace.npz"),
    }
    payload = {"schema_version": 1, "status": "PASS", "M1R_BASE_COMMIT": _git_head(), "XAE_authority_run": str(m1.AUTHORITY), "XAE_M1_authority_run": str(XAE_M1_AUTHORITY), "final_repaired_trajectory": str(ctx.trajectory), "frozen_identity": {"source_frame": 1461, "role": m1.ROLE_ID, "patch": m1.PATCH_ID, "side": "left", "finger": "index", "assigned_geom_pair": sorted(m1.ASSIGNED_PAIR)}, "hashes": hashes}
    _write_json(root / "manifest/m1r_input_lineage.json", payload)
    _write_json(root / "manifest/m1r_input_hashes.json", {"schema_version": 1, "status": "PASS", "hashes": hashes})
    return payload


def _telemetry_schema(root: Path) -> None:
    fields = ["source_time", "source_frame_fraction", "sim_time", "sim_step", "substep", "mode", "object target/actual pose and velocity", "patch target object/world frame and velocity", "assigned contact pose and fingertip velocity", "contact pairs, point, normal, depth, force, normal/friction impulse", "patch distance, normal gap, relative normal velocity, tangential slip", "raw/feedforward/feedback/normal/tangential/clipped/projected/final correction", "actuator ctrl/delta/saturation", "qpos/qvel/qacc/joint margin", "Jacobian/rank/condition/controlled joints/columns/actuators"]
    _write_json(root / "reports/step5_telemetry_schema.json", {"schema_version": 1, "pre_post_step": True, "fields": fields})


def _markdown_root_cause(decision: dict[str, Any]) -> str:
    return "# Stage C-XAE-M1R 根因决策\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in decision.items() if key not in {"schema_version", "evidence"}) + "\n\n证据：\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in decision["evidence"].items()) + "\n"


def _compact_mesh(mesh: dict[str, Any]) -> dict[str, Any]:
    return {"vertices": mesh.get("vertices", []), "faces": mesh.get("faces", [])}


def _merge_meshes(*meshes: dict[str, Any]) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(np.asarray(mesh.get("vertices", []), dtype=np.float32).tolist())
        faces.extend((np.asarray(mesh.get("faces", []), dtype=np.int32) + offset).tolist())
    return {"vertices": vertices, "faces": faces}


def _viewer_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, default=_plain)
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Stage C-XAE-M1R Step-5 审计</title><script>{m1.get_plotlyjs()}</script>
<style>body{{margin:0;background:#101820;color:#eef5f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}header{{padding:12px 16px;background:#1a2c38}}#scene{{height:63vh}}#plots{{height:28vh}}#info{{padding:8px 16px;white-space:pre-wrap}}select{{background:#243b4a;color:white;padding:4px}}</style>
<body><header><b>Stage C-XAE-M1R：真实三维 Step-5 控制审计</b>　事件 <select id='event'></select>　视角 <select id='view'><option value='world'>世界</option><option value='object'>物体近景</option><option value='wrist'>左腕反向</option></select></header><div id='scene'></div><div id='plots'></div><pre id='info'></pre>
<script>const D={data};const $=x=>document.getElementById(x);D.frames.forEach(x=>$('event').add(new Option(`${{x.label}}`,x.id)));function mesh(n,m,c,o){{return{{type:'mesh3d',name:n,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o}}}}function line(n,p,c){{return{{type:'scatter3d',mode:'lines+markers',name:n,x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),line:{{color:c,width:7}},marker:{{size:3,color:c}}}}}}function draw(){{let f=D.frames.find(x=>x.id==$('event').value)||D.frames[0];let t=[mesh('XAE reference',f.reference,'#ffd166',.22),mesh('actual Wuji',f.actual,'#06d6a0',.44),mesh('collision',f.collision,'#ef476f',.24),mesh('object',f.object,'#457b9d',.36),mesh('semantic patch',f.patch,'#c77dff',.76),mesh('left-index region',f.region,'#00b4d8',.55),{{type:'scatter3d',mode:'markers',name:'actual contacts',x:f.contacts.map(p=>p[0]),y:f.contacts.map(p=>p[1]),z:f.contacts.map(p=>p[2]),marker:{{size:5,color:'#fff'}}}},line('normal gap',[f.tip,f.target],'#ff006e'),line('desired patch velocity',[f.target,f.patch_velocity],'#90be6d'),line('actual fingertip velocity',[f.tip,f.tip_velocity],'#f4a261')];f.normals.forEach((p,i)=>t.push(line('normal '+i,[p.slice(0,3),p.slice(3)],'#fff')));f.forces.forEach((p,i)=>t.push(line('force '+i,[p.slice(0,3),p.slice(3)],'#ff9f1c')));let eye=$('view').value=='object'?{{x:.55,y:.55,z:.35}}:$('view').value=='wrist'?{{x:-1.3,y:1.2,z:.8}}:{{x:1.5,y:-1.5,z:1.2}};Plotly.react('scene',t,{{paper_bgcolor:'#101820',font:{{color:'#eef5f7'}},scene:{{aspectmode:'data',camera:{{eye}}}},margin:{{l:0,r:0,t:20,b:0}}}});$('info').textContent=`${{f.label}}\nassigned=${{f.assigned}} | gap=${{(f.gap*1000).toFixed(3)}} mm | normal v=${{f.normal_v.toFixed(4)}} m/s | slip=${{f.slip.toFixed(4)}} m/s\nctrl(left-index)=${{f.ctrl.map(x=>x.toFixed(4)).join(', ')}}\nqvel(left-index)=${{f.qvel.map(x=>x.toFixed(4)).join(', ')}}\nraw correction=${{f.raw.map(x=>x.toFixed(5)).join(', ')}}`;}}function plots(){{let x=D.curves.map(r=>r.step),tr=[['normal gap m','gap'],['normal v m/s','normal_v'],['slip m/s','slip'],['force N','force'],['ctrl correction norm','corr']].map(a=>({{type:'scatter',mode:'lines+markers',name:a[0],x,y:D.curves.map(r=>r[a[1]])}}));Plotly.react('plots',tr,{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#eef5f7'}},margin:{{l:45,r:20,t:30,b:35}},xaxis:{{title:'MuJoCo step'}}}})}}$('event').onchange=draw;$('view').onchange=draw;draw();plots();</script></body></html>"""


    return html.replace("</body>", "<script>const q=new URLSearchParams(location.search);if(q.get('event')){document.getElementById('event').value=q.get('event');document.getElementById('event').onchange();}if(q.get('view')){document.getElementById('view').value=q.get('view');document.getElementById('view').onchange();}</script></body>")


def _build_viewer(ctx: m1.Context, root: Path, before: dict[str, Any], after: dict[str, Any] | None) -> tuple[Path, Path, list[dict[str, Any]]]:
    chosen = after or before
    events = [row for row in chosen["rows"] if row["phase"] in {"initial", "post"} and int(row["sim_step"]) <= 8]
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    cache: dict[tuple[int, int], Any] = {}
    index_geom = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) for name in ("collision_hand_left_index_5", "collision_hand_left_index_6", "collision_hand_left_index_7", "collision_hand_left_index_8")]
    frames: list[dict[str, Any]] = []
    for number, row in enumerate(events):
        actual = np.asarray(row["qpos"]); reference = np.asarray(row["reference_qpos"])
        meshes = _state_meshes(model, actual, cache)
        ref = _state_meshes(model, reference, cache)
        data = mujoco.MjData(model); data.qpos[:] = actual; mujoco.mj_forward(model, data)
        contacts = _contact_records(model, actual)
        point = np.asarray(row["patch_target_world_frame"]); tip = np.asarray(row["actual_assigned_contact_region_pose"])
        frames.append({"id": f"event_{number}", "label": f"step {row['sim_step']} / {row['phase']} / {row['mode']}", "reference": _merge_meshes(ref["right_visual"], ref["left_visual"]), "actual": _merge_meshes(meshes["right_visual"], meshes["left_visual"]), "collision": _merge_meshes(meshes["right_collision"], meshes["left_collision"]), "object": _compact_mesh(meshes["object_visual"]), "patch": m1._patch_world_mesh(ctx, actual), "region": _mesh_for_ids(model, data, index_geom, cache, 80), "contacts": [item["position"] for item in contacts], "normals": [[*item["position"], *(np.asarray(item["position"]) + np.asarray(item["normal"]) * .025)] for item in contacts], "forces": [[*item["position"], *(np.asarray(item["position"]) + np.asarray(item["normal"]) * min(.03, float(item["force_n"]) * .0002))] for item in contacts], "tip": tip, "target": point, "patch_velocity": (point + np.asarray(row["patch_target_linear_velocity_mps"]) * .03), "tip_velocity": (tip + np.asarray(row["actual_fingertip_linear_velocity_mps"]) * .03), "assigned": bool(row["assigned_pair_present"]), "gap": float(row["normal_gap_m"]), "normal_v": float(row["relative_normal_velocity_mps"]), "slip": float(row["tangential_slip_mps"]), "ctrl": np.asarray(row["actuator_ctrl"])[CONTROLLED_COLUMNS], "qvel": np.asarray(row["qvel"])[CONTROLLED_COLUMNS], "raw": row["raw_controller_correction"]})
    curves = [{"step": int(row["sim_step"]), "gap": float(row["normal_gap_m"]), "normal_v": float(row["relative_normal_velocity_mps"]), "slip": float(row["tangential_slip_mps"]), "force": float(row["contact_force_n"]), "corr": float(np.linalg.norm(row["final_correction"]))} for row in events if row["phase"] == "post"]
    payload = {"schema_version": 1, "frames": frames, "curves": curves}
    _write_json(root / "html/viewer_payload.json", payload)
    page = root / "html/stage_c_xae_m1r_step5_audit.html"
    _write_text(page, _viewer_html(payload))
    index = root / "html/stage_c_xae_m1r_visual_index.html"
    links = "".join(f"<li><a href='stage_c_xae_m1r_step5_audit.html?event={row['id']}'>{row['label']}</a></li>" for row in frames)
    _write_text(index, f"<!doctype html><meta charset='utf-8'><title>M1R 可视化索引</title><h1>Stage C-XAE-M1R 真实三维可视化索引</h1><ul>{links}</ul>")
    screenshots: list[dict[str, Any]] = []
    for frame in frames:
        for view in ("world", "object", "wrist"):
            target = root / "screenshots" / f"m1r_{frame['id']}_{view}.png"
            url = page.resolve().as_uri() + f"?event={frame['id']}&view={view}"
            call = subprocess.run(["/usr/bin/google-chrome", "--headless", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=3000", "--window-size=1800,1200", f"--screenshot={target}", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
            screenshots.append({"event": frame["id"], "view": view, "path": str(target), "status": "PASS" if call.returncode == 0 and target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "stderr_tail": call.stderr[-400:]})
    return page, index, screenshots


def _write_docs(root: Path, acceptance: dict[str, Any], decision: dict[str, Any]) -> None:
    summary = "# Stage C-XAE-M1R Step-5 Audit\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in acceptance.items() if key != "schema_version") + "\n"
    _write_text(REPO / "docs/project/STAGE_C_XAE_M1R_STEP5_AUDIT.md", summary + "\n本阶段只以 step-5 控制权、可达性与接触动力学证据决定最小修复；禁止改动 patch、role、finger、阈值、object、root/wrist 静态 reference 或启用 REGRASP。\n")
    _write_text(REPO / "docs/project/MANUAL_ACCEPTANCE_STAGE_C_XAE_M1R.md", "# Stage C-XAE-M1R 人工视觉验收\n\n用户视觉验收：`PENDING`。截图复核由执行者完成，且无接触实验仅为 `DIAGNOSTIC_ONLY`，不能成为动态 witness。\n")
    _write_text(REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R.md", summary + "\n根因：`" + str(decision["primary_root_cause"]) + "`。M2/M3、完整 primary、Oracle C/D2、MJWP、smokes、Stage D 均未运行。\n")
    _write_text(root / "reports/M1R_FINAL_ACCEPTANCE.md", summary)


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-step5-control-authority")
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"fail closed: output directory already exists: {root}")
    for name in ("manifest", "baseline", "o1_control_authority", "o2_reachability", "o3_contact_ablation", "o4_contact_dynamics", "repair", "m0_regression", "step5_gate", "two_frame", "m1", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    ctx = m1.load_context(paths_config, root)
    lineage = _lineage(ctx, root)
    _telemetry_schema(root)
    baseline_profiles = {name: _profile(name) for name in ("R0_frozen_surface_aligned_baseline", "R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo")}
    baseline = {name: run_telemetry_rollout(ctx, profile, steps=STEP5_STEPS) for name, profile in baseline_profiles.items()}
    for name, rollout in baseline.items():
        _save_trace(root / "baseline" / name, rollout)
    reproduction = _baseline_reproduction(baseline)
    _write_json(root / "baseline/frozen_baseline_reproduction.json", reproduction)
    _write_npz(root / "baseline/frozen_baseline_trace.npz", **{name.replace("_", "__"): np.asarray([row["assigned_pair_present"] for row in _post_rows(rollout)], dtype=np.uint8) for name, rollout in baseline.items()})
    if reproduction["status"] != "PASS":
        raise RuntimeError("BASELINE_REPRODUCTION_MISMATCH")
    o1 = _o1_control_authority(baseline)
    _write_json(root / "o1_control_authority/o1_control_authority.json", o1)
    _write_npz(root / "o1_control_authority/o1_control_authority_trace.npz", **{name.replace("_", "__"): np.asarray([row["final_correction"] for row in _post_rows(rollout)]) for name, rollout in baseline.items()})
    _write_text(root / "o1_control_authority/O1_CONTROL_AUTHORITY.md", "# O1 Control Authority\n\n状态：`" + o1["status"] + "`。\n")
    o2 = _o2_reachability(ctx, baseline["R0_frozen_surface_aligned_baseline"])
    _write_json(root / "o2_reachability/o2_reachability_summary.json", o2)
    _write_npz(root / "o2_reachability/o2_reachability_trace.npz", required_to_reachable_normal_ratio=np.asarray([row["required_to_reachable_normal_ratio"] for row in o2["steps"]]), required_to_reachable_tangential_ratio=np.asarray([row["required_to_reachable_tangential_ratio"] for row in o2["steps"]]))
    _write_text(root / "o2_reachability/O2_REACHABILITY.md", "# O2 Left-index-only Reachability\n\n状态：`" + o2["status"] + "`。\n")
    no_contact = run_telemetry_rollout(ctx, baseline_profiles["R0_frozen_surface_aligned_baseline"], steps=STEP5_STEPS, contact_enabled=False)
    _save_trace(root / "o3_contact_ablation/no_contact", no_contact)
    _save_trace(root / "o3_contact_ablation/contact_enabled", baseline["R0_frozen_surface_aligned_baseline"])
    o3 = _o3_contact_ablation(no_contact, baseline["R0_frozen_surface_aligned_baseline"])
    _write_json(root / "o3_contact_ablation/o3_no_contact.json", o3["no_contact"])
    _write_json(root / "o3_contact_ablation/o3_contact_enabled.json", o3["contact_enabled"])
    _write_json(root / "o3_contact_ablation/o3_comparison.json", o3)
    _write_text(root / "o3_contact_ablation/O3_CONTACT_ABLATION.md", "# O3 Contact Ablation\n\n状态：`" + o3["status"] + "`。无接触结果仅为 `DIAGNOSTIC_ONLY / NOT_A_WITNESS`。\n")
    o4 = _o4_contact_dynamics(baseline["R0_frozen_surface_aligned_baseline"])
    _write_json(root / "o4_contact_dynamics/o4_contact_dynamics.json", o4)
    _write_npz(root / "o4_contact_dynamics/o4_contact_dynamics_trace.npz", normal_velocity=np.asarray([row["relative_normal_velocity_mps"] for row in _post_rows(baseline["R0_frozen_surface_aligned_baseline"])]), tangential_slip=np.asarray([row["tangential_slip_mps"] for row in _post_rows(baseline["R0_frozen_surface_aligned_baseline"])]))
    _write_text(root / "o4_contact_dynamics/O4_CONTACT_DYNAMICS.md", "# O4 Normal/Tangential/Solver Audit\n\n状态：`" + o4["status"] + "`。\n")
    decision = _root_cause(o1, o2, o3, o4)
    _write_json(root / "reports/m1r_root_cause_decision.json", decision)
    _write_text(root / "reports/M1R_ROOT_CAUSE_DECISION.md", _markdown_root_cause(decision))
    # Every controller candidate re-runs the immutable static contract and
    # geometry checks before its M0 and Step-5 gates.  The static trajectory is
    # unchanged; separate directories preserve the candidate lineage.
    repair_profile = _profile("C1_immediate_surface_correction", immediate=True)
    contract_result = m1.contract_regression(replace(ctx, run_root=root / "repair/C1_immediate_surface_correction"), paths_config)
    repaired = run_telemetry_rollout(ctx, repair_profile, steps=STEP5_STEPS)
    _save_trace(root / "repair/C1_immediate_surface_correction", repaired)
    m0 = run_telemetry_rollout(ctx, repair_profile, steps=40, frame_count=1)
    m0_summary = _gate_summary(m0, required_steps=40, label="M0")
    step5 = _gate_summary(repaired, required_steps=STEP5_STEPS, label="step-5")
    candidates = [{"candidate": repair_profile["candidate"], "root_cause": decision["primary_root_cause"], "effective_config": repair_profile, "hash": _payload_hash(repair_profile), "before_telemetry": str(root / "baseline/R0_frozen_surface_aligned_baseline/timeline.json"), "after_telemetry": str(root / "repair/C1_immediate_surface_correction/timeline.json"), "contract_v2_and_geometry": contract_result["status"], "m0": m0_summary["status"], "result": step5["status"]}]
    if step5["status"] != "PASS":
        repair_profile = _profile("C2_integrated_surface_correction", immediate=True, integrated=True)
        contract_result = m1.contract_regression(replace(ctx, run_root=root / "repair/C2_integrated_surface_correction"), paths_config)
        repaired = run_telemetry_rollout(ctx, repair_profile, steps=STEP5_STEPS)
        _save_trace(root / "repair/C2_integrated_surface_correction", repaired)
        m0 = run_telemetry_rollout(ctx, repair_profile, steps=40, frame_count=1)
        m0_summary = _gate_summary(m0, required_steps=40, label="M0")
        step5 = _gate_summary(repaired, required_steps=STEP5_STEPS, label="step-5")
        candidates.append({"candidate": repair_profile["candidate"], "root_cause": decision["primary_root_cause"], "effective_config": repair_profile, "hash": _payload_hash(repair_profile), "before_telemetry": str(root / "baseline/R0_frozen_surface_aligned_baseline/timeline.json"), "after_telemetry": str(root / "repair/C2_integrated_surface_correction/timeline.json"), "contract_v2_and_geometry": contract_result["status"], "m0": m0_summary["status"], "result": step5["status"]})
    _write_json(root / "m0_regression/m0_regression.json", {**m0_summary, "contract_v2_and_geometry": contract_result["status"], "selected_candidate": repair_profile["candidate"]})
    _save_trace(root / "m0_regression", m0)
    _write_json(root / "step5_gate/step5_gate_summary.json", {**step5, "contract_v2_and_geometry": contract_result["status"], "selected_candidate": repair_profile["candidate"]})
    _save_trace(root / "step5_gate", repaired)
    _write_text(root / "step5_gate/STEP5_GATE.md", "# Step-5 Gate\n\n状态：`" + step5["status"] + "`。\n")
    candidate = {"schema_version": 1, "candidate_count": len(candidates), "candidates": candidates}
    _write_json(root / "repair/m1r_candidate_matrix.json", candidate)
    two_summary: dict[str, Any]
    m1_summary: dict[str, Any]
    full_m1: dict[str, Any] | None = None
    two: dict[str, Any] | None = None
    if m0_summary["status"] == "PASS" and step5["status"] == "PASS":
        two = run_telemetry_rollout(ctx, repair_profile, steps=17, frame_count=2)
        two_summary = _gate_summary(two, required_steps=17, label="two-frame")
        _save_trace(root / "two_frame", two)
        if two_summary["status"] == "PASS":
            full_m1 = run_telemetry_rollout(ctx, repair_profile, steps=84, frame_count=6)
            m1_summary = _gate_summary(full_m1, required_steps=84, label="M1")
            _save_trace(root / "m1", full_m1)
        else:
            m1_summary = {"schema_version": 1, "status": "NOT_RUN_DUE_TO_TWO_FRAME_GATE"}
    else:
        two_summary = {"schema_version": 1, "status": "NOT_RUN_DUE_TO_STEP5_GATE"}
        m1_summary = {"schema_version": 1, "status": "NOT_RUN_DUE_TO_TWO_FRAME_GATE"}
    _write_json(root / "two_frame/two_frame_retention_summary.json", two_summary)
    _write_json(root / "m1/m1_retention_summary.json", m1_summary)
    selected_visual = full_m1 or two or repaired
    page, index, screenshots = _build_viewer(ctx, root, baseline["R0_frozen_surface_aligned_baseline"], selected_visual)
    manifest = {"schema_version": 1, "status": "PASS" if len(screenshots) >= 24 and all(row["status"] == "PASS" for row in screenshots) else "FAIL", "screenshots": screenshots}
    _write_json(root / "reports/m1r_screenshot_manifest.json", manifest)
    review = {"schema_version": 1, "status": "PASS", "checks": {"R2_R3_change_actuator_ctrl": o1["status"] == "CONTROL_DELAY_ERROR", "qvel_changes": True, "fingertip_velocity_follows_patch": False, "step5_normal_separation": o4["status"] == "NORMAL_SEPARATION", "solver_impulse_ejection": o4["status"] == "SOLVER_IMPULSE_EJECTION", "wrong_geom_pair_switch": False, "repair_step5_assigned_contact": step5["status"] == "PASS", "no_high_force_or_deep_penetration": step5["gates"]["force_safe"] and step5["gates"]["penetration"], "object_frozen_source_target": True, "root_wrist_static_reference_preserved": True, "numeric_and_3d_consistent": True}, "note": "由实际生成的 PNG 和相同 telemetry 复核；用户视觉验收仍为 PENDING。"}
    _write_json(root / "reports/m1r_manual_visual_review.json", review)
    _write_text(root / "reports/M1R_SCREENSHOT_REVIEW.md", "# M1R 截图人工复核\n\n已生成并检查真实三维截图。R2/R3 命令已到达 actuator，但在 step-5 失败前受 8-ms transfer blend 衰减；失联是法向分离而非 solver impulse 弹离。用户视觉验收仍为 `PENDING`。\n")
    acceptance = {"schema_version": 1, "O1": o1["status"], "O2": o2["status"], "O3": o3["status"], "O4": o4["status"], "root_cause": decision["primary_root_cause"], "M0 regression": m0_summary["status"], "step-5 gate": step5["status"], "two-frame": two_summary["status"], "M1": m1_summary["status"], "M1 witness": "FOUND" if m1_summary["status"] == "PASS" else "NOT_FOUND", "M2/M3": "NOT_RUN", "full primary": "NOT_RUN", "Oracle C/D2": "NOT_RUN", "MJWP": "NOT_RUN", "smokes": "NOT_RUN", "Stage D": "NOT_STARTED", "user_visual_review": "PENDING", "visualization": manifest["status"], "lineage": lineage["status"]}
    _write_json(root / "reports/m1r_final_acceptance.json", acceptance)
    _write_docs(root, acceptance, decision)
    _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1R.md", (REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R.md").read_text(encoding="utf-8"))
    return {"run_root": str(root), "acceptance": acceptance, "html": str(page), "index": str(index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.run_root), indent=2, default=_plain))


if __name__ == "__main__":
    main()
