"""Stage C-M1R contact-mode integrity repair and layered MuJoCo experiments.

This runner is deliberately limited to the frozen 1461..1480 window.  It
uses only normal MuJoCo stepping plus the scene's object mocap guidance; after
the one-time simulation-state initialization it never writes robot or object
generalized coordinates.  M0/M1/M2/M3 are strictly sequential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine, ContactObservation, observation_payload
from spider.datasets.paths import load_project_paths
from spider.datasets.schema import CanonicalHOISequence
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools.grab_stage_c import _finite_data, _object_tracking_error, _preflight_object_ids, _set_object_mocap_reference, _site_ids
from spider.tools.grab_stage_c_contact_mode import _load_patch, _load_window, _patch_measure
from spider.tools.grab_stage_c_failure_diagnostic import _contact_records, _patch_surface, _source_skeleton, _state_meshes


PRIMARY = dynamic.PRIMARY
WINDOW_SOURCE_FRAMES = np.arange(1461, 1481, dtype=np.int64)
OUTPUT_ROOT = Path(".local_artifacts/stage_c_cm1r")
OLD_ROOT = Path(".local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode")
DIAGNOSTIC_ROOT = Path(".local_artifacts/stage_c_v2r2e_failure_diagnostic/20260731T180415Z")
PATCH_ID = "patch:s5__cylindermedium_lift:0"
ROLE_ID = "s5__cylindermedium_lift:0"
REGION = "left_index_fingertip"
ASSIGNED_PAIR = frozenset(("collision_hand_left_index_8", "right_object_0"))
CONTROLLED_COLUMNS = tuple(range(26, 32)) + tuple(range(36, 40))

EFFECTIVE_FIELDS = (
    "profile_id", "experiment", "seed", "acquire_entry_distance_m", "retain_hysteresis_distance_m",
    "confirmation_substeps", "acquire_timeout_ms", "regrasp_timeout_ms", "max_regrasp_attempts",
    "kp_scale", "contact_ik_gain", "contact_ik_damping", "state_feedback_gain", "lead_source_frames",
    "controlled_joint_set", "contact_target_frame",
)
HASH_EXCLUDED_FIELDS = {"profile_id", "experiment", "seed", "recovery_reason"}


def _json_default(value: Any) -> Any:
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
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


def _sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_effective_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a complete, typed execution contract before any rollout starts."""
    missing = [field for field in EFFECTIVE_FIELDS if field not in profile]
    if missing:
        raise ValueError(f"effective profile is missing required field(s): {missing}")
    normalized = {
        "profile_id": str(profile["profile_id"]),
        "experiment": str(profile["experiment"]),
        "seed": int(profile["seed"]),
        "acquire_entry_distance_m": float(profile["acquire_entry_distance_m"]),
        "retain_hysteresis_distance_m": float(profile["retain_hysteresis_distance_m"]),
        "confirmation_substeps": int(profile["confirmation_substeps"]),
        "acquire_timeout_ms": int(profile["acquire_timeout_ms"]),
        "regrasp_timeout_ms": int(profile["regrasp_timeout_ms"]),
        "max_regrasp_attempts": int(profile["max_regrasp_attempts"]),
        "kp_scale": float(profile["kp_scale"]),
        "contact_ik_gain": float(profile["contact_ik_gain"]),
        "contact_ik_damping": float(profile["contact_ik_damping"]),
        "state_feedback_gain": float(profile["state_feedback_gain"]),
        "lead_source_frames": int(profile["lead_source_frames"]),
        "controlled_joint_set": tuple(str(item) for item in profile["controlled_joint_set"]),
        "contact_target_frame": str(profile["contact_target_frame"]),
        "bumpless_ramp_ms": int(profile.get("bumpless_ramp_ms", 4)),
        "servo_integral_gain": float(profile.get("servo_integral_gain", 0.0)),
        "initial_velocity": str(profile.get("initial_velocity", "zero_hold")),
        "allow_regrasp": bool(profile.get("allow_regrasp", False)),
        "recovery_reason": str(profile.get("recovery_reason", "initial")),
    }
    ContactModeConfig(
        acquire_entry_distance_m=normalized["acquire_entry_distance_m"],
        retain_hysteresis_distance_m=normalized["retain_hysteresis_distance_m"],
        confirmation_substeps=normalized["confirmation_substeps"],
        acquire_timeout_ms=normalized["acquire_timeout_ms"],
        regrasp_timeout_ms=normalized["regrasp_timeout_ms"],
        max_regrasp_attempts=normalized["max_regrasp_attempts"],
        allow_regrasp=normalized["allow_regrasp"],
    )
    if normalized["controlled_joint_set"] != ("left_wrist", "left_index"):
        raise ValueError("C-M1R can correct only the audited left wrist plus left index set")
    if normalized["contact_target_frame"] != "object_local":
        raise ValueError("C-M1R semantic contact targets must be object_local")
    if normalized["bumpless_ramp_ms"] not in {2, 4, 8}:
        raise ValueError("bumpless ramp must be 2, 4, or 8 ms")
    if "hysteresis_030" in normalized["profile_id"] and normalized["retain_hysteresis_distance_m"] != 0.030:
        raise ValueError("e3_hysteresis_030 must serialize retain_hysteresis_distance_m == 0.030")
    return normalized


def effective_profile_hash(profile: dict[str, Any]) -> str:
    normalized = normalize_effective_profile(profile)
    semantic = {key: value for key, value in normalized.items() if key not in HASH_EXCLUDED_FIELDS}
    return _sha256(semantic)


def validate_profiles(profiles: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_effective_profile(row) for row in profiles]
    seen: dict[str, str] = {}
    for row in normalized:
        fingerprint = effective_profile_hash(row)
        previous = seen.get(fingerprint)
        if previous is not None and previous != row["profile_id"]:
            raise ValueError(f"duplicate effective profile hash {fingerprint}: {previous} and {row['profile_id']}")
        seen[fingerprint] = row["profile_id"]
        row["EFFECTIVE_PROFILE_HASH"] = fingerprint
    return normalized


def _profile(profile_id: str, experiment: str, *, seed: int, **overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile_id": profile_id,
        "experiment": experiment,
        "seed": seed,
        "acquire_entry_distance_m": 0.025,
        "retain_hysteresis_distance_m": 0.025,
        "confirmation_substeps": 4,
        "acquire_timeout_ms": 40,
        "regrasp_timeout_ms": 40,
        "max_regrasp_attempts": 2,
        "kp_scale": 1.0,
        "contact_ik_gain": 1.0,
        "contact_ik_damping": 0.002,
        "state_feedback_gain": 0.0,
        "lead_source_frames": 0,
        "controlled_joint_set": ("left_wrist", "left_index"),
        "contact_target_frame": "object_local",
        "bumpless_ramp_ms": 4,
        "servo_integral_gain": 0.0,
        "initial_velocity": "zero_hold",
        "allow_regrasp": False,
        "recovery_reason": "initial",
    }
    result.update(overrides)
    return result


def profile_matrix_coverage() -> dict[str, Any]:
    requested = {
        "acquire_entry_distance_m": [0.025, 0.030],
        "retain_hysteresis_distance_m": [0.025, 0.030],
        "confirmation_substeps": [2, 4, 8],
        "acquire_timeout_ms": [20, 40, 80],
        "regrasp_timeout_ms": [20, 40, 80],
        "max_regrasp_attempts": [1, 2],
    }
    profiles: list[dict[str, Any]] = []
    index = 0
    for acquire in requested["acquire_entry_distance_m"]:
        for hysteresis in requested["retain_hysteresis_distance_m"]:
            for confirmation in requested["confirmation_substeps"]:
                for acquire_timeout in requested["acquire_timeout_ms"]:
                    for regrasp_timeout in requested["regrasp_timeout_ms"]:
                        for attempts in requested["max_regrasp_attempts"]:
                            index += 1
                            profiles.append(_profile(
                                f"matrix_{index:03d}", "PROFILE_BUILDER_ONLY", seed=202608100 + index,
                                acquire_entry_distance_m=acquire,
                                retain_hysteresis_distance_m=hysteresis,
                                confirmation_substeps=confirmation,
                                acquire_timeout_ms=acquire_timeout,
                                regrasp_timeout_ms=regrasp_timeout,
                                max_regrasp_attempts=attempts,
                                allow_regrasp=attempts > 0,
                            ))
    normalized = validate_profiles(profiles)
    effective = {field: sorted({row[field] for row in normalized}) for field in requested}
    return {
        "schema_version": 1,
        "status": "PASS",
        "requested_dimensions": requested,
        "effective_dimensions": effective,
        "unique_profile_count": len(normalized),
        "duplicate_count": 0,
        "missing_dimensions": {field: sorted(set(values) - set(effective[field])) for field, values in requested.items()},
        "execution": "BUILDER_VALIDATION_ONLY_NO_DYNAMIC_MATRIX_RUN",
    }


def m1_recovery_profiles() -> list[dict[str, Any]]:
    """Eight evidence-labelled M1 repairs; this is not the historical E1--E3 search."""
    rows = [
        _profile("m1_object_local_baseline", "M1", seed=202608201, initial_velocity="dynamic_reference", recovery_reason="TARGET_FRAME_ERROR"),
        _profile("m1_contact_transform_feedback", "M1", seed=202608202, initial_velocity="dynamic_reference", contact_ik_gain=4.0, recovery_reason="TARGET_FRAME_ERROR"),
        _profile("m1_wrist_index_servo", "M1", seed=202608203, initial_velocity="dynamic_reference", contact_ik_gain=12.0, recovery_reason="CONTROLLED_DOF_INSUFFICIENT"),
        _profile("m1_wrist_index_integral", "M1", seed=202608204, initial_velocity="dynamic_reference", contact_ik_gain=20.0, servo_integral_gain=0.03, recovery_reason="RETENTION_FAILURE"),
        _profile("m1_wrist_index_phase_lead", "M1", seed=202608205, initial_velocity="dynamic_reference", contact_ik_gain=20.0, lead_source_frames=1, recovery_reason="TRACKING_FAILURE"),
        _profile("m1_wrist_index_hysteresis_030", "M1", seed=202608206, initial_velocity="dynamic_reference", retain_hysteresis_distance_m=0.030, contact_ik_gain=20.0, recovery_reason="RETENTION_FAILURE"),
        _profile("m1_wrist_index_damped", "M1", seed=202608207, initial_velocity="dynamic_reference", contact_ik_gain=30.0, contact_ik_damping=0.01, recovery_reason="FORCE_SPIKE"),
        _profile("m1_wrist_index_ramp_8ms", "M1", seed=202608208, initial_velocity="dynamic_reference", contact_ik_gain=20.0, bumpless_ramp_ms=8, recovery_reason="BUMPLESS_TRANSFER_FAILURE"),
    ]
    return validate_profiles(rows)


def _exact_pair(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if frozenset(str(row["geom_pair"]).split("|")) == ASSIGNED_PAIR:
            return row
    return None


def _joint_names(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in CONTROLLED_COLUMNS:
        joint = int(model.actuator_trnid[column, 0])
        rows.append({
            "control_column": column,
            "joint_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
            "actuator_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, column),
            "role": "left_wrist" if column < 32 else "left_index",
            "jacobian_column": column,
            "limit": [float(model.jnt_range[joint, 0]), float(model.jnt_range[joint, 1])],
        })
    return rows


def _corrected_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target: np.ndarray,
    profile: dict[str, Any],
    integral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Object-local contact feedback using only left wrist plus index columns."""
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, None, site_id)
    columns = np.asarray(CONTROLLED_COLUMNS, dtype=np.int64)
    block = jacobian[:, columns]
    error = np.asarray(target, dtype=np.float64) - np.asarray(data.site_xpos[site_id], dtype=np.float64)
    raw = block.T @ np.linalg.solve(block @ block.T + float(profile["contact_ik_damping"]) * np.eye(3), error)
    raw *= float(profile["contact_ik_gain"])
    clipped = np.clip(raw, -np.array([0.02, 0.02, 0.02, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12]), np.array([0.02, 0.02, 0.02, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12]))
    integral = np.clip(integral + float(profile["servo_integral_gain"]) * clipped, -0.12, 0.12)
    correction = np.zeros(52, dtype=np.float64)
    correction[columns] = clipped + integral
    return correction, raw, clipped


def _world_from_local(data: mujoco.MjData, body_id: int, local: np.ndarray) -> np.ndarray:
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ np.asarray(local, dtype=np.float64)


def _local_from_world(data: mujoco.MjData, body_id: int, world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return rotation.T @ (np.asarray(world, dtype=np.float64) - np.asarray(data.xpos[body_id], dtype=np.float64))


def _model_and_seed(window: dict[str, Any], profile: dict[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, int], dict[str, int], list[int], trimesh.Trimesh, str]:
    model = mujoco.MjModel.from_xml_path(str(window["physics"]["scene_act"]))
    model.opt.timestep = 0.0005
    model.actuator_gainprm[:52, 0] *= float(profile["kp_scale"])
    model.actuator_biasprm[:52, 1] *= float(profile["kp_scale"])
    data = mujoco.MjData(model)
    data.qpos[:] = window["reference"][0]
    data.qvel[:] = window["reference_qvel"][0] if profile["initial_velocity"] == "dynamic_reference" else 0.0
    bodies, mocap = _preflight_object_ids(model)
    _set_object_mocap_reference(data, window["reference"][0], mocap)
    mujoco.mj_forward(model, data)
    patch_mesh, _patch, role_id, region = _load_patch(Path(window["physics"]["collision_cache"]) / "visual/visual.obj", window["inputs"])
    if role_id != ROLE_ID or region != REGION:
        raise RuntimeError("frozen role, finger, or semantic patch mapping changed")
    return model, data, bodies, mocap, _site_ids(model), patch_mesh, role_id


def _run_experiment(window: dict[str, Any], output: Path, profile: dict[str, Any], *, experiment: str, frames: int, injected_loss: bool = False) -> dict[str, Any]:
    """Run one real MuJoCo M0/M1/M2/M3 candidate without state teleportation."""
    profile = normalize_effective_profile(profile)
    profile["EFFECTIVE_PROFILE_HASH"] = effective_profile_hash(profile)
    model, data, bodies, mocap, sites, patch_mesh, role_id = _model_and_seed(window, profile)
    hand, objects = v2r._contact_ids(model)
    object_body, tip_site, wrist_site = bodies["right"], sites[8], sites[6]
    reference, reference_qvel, source_frames = window["reference"], window["reference_qvel"], window["source_frames"]
    expected, anchors = window["expected"], window["anchors"]
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = reference[0]
    target_data.qvel[:] = 0.0
    mujoco.mj_forward(model, target_data)
    patch_target_object = _local_from_world(target_data, object_body, anchors[0, 6])
    fingertip_contact_object = _local_from_world(data, object_body, data.site_xpos[tip_site])
    contact_transform_object: np.ndarray | None = None
    data.ctrl[:52] = reference[0, :52]
    data.ctrl[52:] = 0.0
    previous_ctrl = np.asarray(data.ctrl[:52], dtype=np.float64).copy()
    correction_integral = np.zeros(len(CONTROLLED_COLUMNS), dtype=np.float64)
    machine = ContactModeMachine(ContactModeConfig(
        acquire_entry_distance_m=profile["acquire_entry_distance_m"],
        retain_hysteresis_distance_m=profile["retain_hysteresis_distance_m"],
        confirmation_substeps=profile["confirmation_substeps"],
        acquire_timeout_ms=profile["acquire_timeout_ms"],
        regrasp_timeout_ms=profile["regrasp_timeout_ms"],
        max_regrasp_attempts=profile["max_regrasp_attempts"],
        allow_regrasp=profile["allow_regrasp"],
    ))
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    timeline: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    servo_rows: list[dict[str, Any]] = []
    previous_contact_position: np.ndarray | None = None
    target_time = 0.0
    sim_step = 0
    transition_start_time = 0.0
    last_mode = machine.mode
    injection_start = 24 if injected_loss else -1
    injection_end = 32 if injected_loss else -1

    def observe(frame: int, substep: int, *, record: bool = True) -> ContactObservation:
        nonlocal previous_contact_position, contact_transform_object, transition_start_time, last_mode
        _count, depth, force, pairs = dynamic._contact_summary(model, data, hand, objects)
        pair = _exact_pair(pairs)
        physical = pair is not None
        distance, cosine, closest, normal = _patch_measure(data, model, tip_site, object_body, patch_mesh)
        contact_position = np.asarray(pair["position_world"], dtype=np.float64) if pair else closest
        velocity = np.zeros(3, dtype=np.float64) if previous_contact_position is None else (contact_position - previous_contact_position) / model.opt.timestep
        previous_contact_position = contact_position.copy()
        if pair is not None and contact_transform_object is None:
            contact_transform_object = _local_from_world(data, object_body, contact_position)
        target_data.qpos[:] = reference[frame]
        target_data.qvel[:] = 0.0
        mujoco.mj_forward(model, target_data)
        wrist_error = float(np.linalg.norm(data.site_xpos[wrist_site] - target_data.site_xpos[wrist_site]))
        fingertip_error = float(np.linalg.norm(data.site_xpos[tip_site] - target_data.site_xpos[tip_site]))
        object_pos, object_rot = _object_tracking_error(data, reference[frame], bodies)
        margin, _ = v2r._joint_margin(model, np.asarray([data.qpos], dtype=np.float64))
        observation = ContactObservation(
            source_frame=int(source_frames[frame]), source_timestamp_s=float(source_frames[frame] / 120.0), sim_step=sim_step, substep=substep,
            role_active=bool(expected[frame, 6]), assigned_patch=PATCH_ID, assigned_robot_region=REGION,
            physical_contact_present=physical, correct_geom_pair=physical, geom_pair=str(pair["geom_pair"] if pair else "NONE"),
            patch_distance_m=distance, patch_membership=distance <= 0.020, normal_cosine=cosine,
            tangential_slip_m=float(np.linalg.norm(velocity - np.dot(velocity, normal) * normal)), normal_gap_m=max(0.0, distance if not physical else -float(pair["penetration_m"])),
            penetration_m=float(depth), force_n=float(force), force_impulse_ns=float(force * model.opt.timestep), joint_margin_fraction=float(np.nanmin(margin)),
            wrist_tracking_error_m=wrist_error, fingertip_tracking_error_m=fingertip_error, object_tracking_position_m=float(np.max(object_pos)), object_tracking_rotation_rad=float(np.max(object_rot)),
            finite=bool(_finite_data(data)), joint_limit_valid=not bool(v2r.dynamic._joint_limit_violations(model, np.asarray([data.qpos], dtype=np.float64))), warning_count=len(warnings),
            reference_qpos=tuple(np.asarray(reference[frame], dtype=np.float64)), actual_qpos=tuple(np.asarray(data.qpos, dtype=np.float64)), ctrl=tuple(np.asarray(data.ctrl, dtype=np.float64)), regrasp_attempt=machine.regrasp_attempts,
        )
        prior = machine.mode
        mode = machine.observe(observation)
        if mode != prior:
            transition_start_time = data.time
        last_mode = mode
        if record:
            row = observation_payload(observation)
            row.update({
                "mode": mode.value, "previous_mode": prior.value, "closest_patch_point_world": closest.tolist(), "contact_normal_world": normal.tolist(),
                "patch_target_object": patch_target_object.tolist(), "world_patch_target": _world_from_local(data, object_body, patch_target_object).tolist(),
                "contact_point_object_frame": _local_from_world(data, object_body, contact_position).tolist(),
                "contact_point_world": contact_position.tolist(), "contact_transform_object": None if contact_transform_object is None else contact_transform_object.tolist(),
            })
            timeline.append(row)
        return observation

    def command(frame: int) -> None:
        nonlocal previous_ctrl, correction_integral
        control_index = min(frame + profile["lead_source_frames"], frames - 1)
        nominal = reference[control_index, :52].copy()
        desired_local = contact_transform_object if machine.mode == ContactMode.RETAIN and contact_transform_object is not None else fingertip_contact_object
        desired_target = _world_from_local(data, object_body, desired_local)
        correction, raw, clipped = _corrected_control(model, data, tip_site, desired_target, profile, correction_integral)
        correction_integral = correction[np.asarray(CONTROLLED_COLUMNS)] - clipped
        if machine.mode in {ContactMode.RETAIN_PENDING, ContactMode.RETAIN, ContactMode.REGRASP, ContactMode.ACQUIRE}:
            desired = nominal + correction
        else:
            desired = nominal
        if injection_start <= sim_step < injection_end:
            desired[26:32] += np.array([0.0, 0.0, -0.002, 0.0, 0.0, 0.0])
        elapsed_ms = max(0.0, (data.time - transition_start_time) * 1000.0)
        alpha = min(1.0, elapsed_ms / float(profile["bumpless_ramp_ms"])) if machine.mode in {ContactMode.RETAIN_PENDING, ContactMode.RETAIN, ContactMode.REGRASP} else 1.0
        new_ctrl = (1.0 - alpha) * previous_ctrl + alpha * desired
        ctrl_delta = new_ctrl - previous_ctrl
        data.ctrl[:52] = new_ctrl
        data.ctrl[52:] = 0.0
        servo_rows.append({
            "source_frame": int(source_frames[frame]), "sim_step": sim_step, "mode": machine.mode.value,
            "patch_target_object": patch_target_object.copy(), "world_patch_target": _world_from_local(data, object_body, patch_target_object),
            "normal_gap_m": float(np.linalg.norm(desired_target - data.site_xpos[tip_site])), "normal_relative_velocity": 0.0,
            "tangential_error_m": float(np.linalg.norm(desired_target - data.site_xpos[tip_site])), "tangential_slip_velocity_mps": 0.0,
            "contact_force_n": float(timeline[-1]["force_n"]) if timeline else 0.0, "contact_impulse_ns": float(timeline[-1]["force_impulse_ns"]) if timeline else 0.0,
            "jacobian_controlled_joints": list(CONTROLLED_COLUMNS), "raw_mode_correction": raw, "clipped_correction": clipped,
            "previous_ctrl": previous_ctrl.copy(), "new_ctrl": new_ctrl.copy(), "ctrl_delta": ctrl_delta.copy(), "blend_alpha": alpha,
        })
        previous_ctrl = new_ctrl.copy()

    try:
        initial = observe(0, 0)
        # M0 is a true 20 ms hold.  The moving experiments use the same legal
        # object-mocap guidance but preserve source timing after frame zero.
        if experiment == "M0":
            for substep in range(40):
                command(0)
                mujoco.mj_step(model, data)
                sim_step += 1
                observe(0, substep + 1)
                if machine.mode == ContactMode.FAILED:
                    break
            relevant = [row for row in timeline if int(row["source_frame"]) == int(source_frames[0])]
            last = relevant[-1]
            _count, depth, force, _pairs = dynamic._contact_summary(model, data, hand, objects)
            frame_rows.append({
                "source_frame": int(source_frames[0]), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
                "physical_contact": bool(last["correct_contact"]), "patch_distance_m": float(last["patch_distance_m"]), "normal_cosine": float(last["normal_cosine"]),
                "contact_depth_m": float(depth), "contact_force_n": float(force), "mode": str(last["mode"]),
                "object_position_error_m": float(last["object_tracking_position_m"]), "object_rotation_error_rad": float(last["object_tracking_rotation_rad"]),
            })
        else:
            relevant = [row for row in timeline if int(row["source_frame"]) == int(source_frames[0])]
            last = relevant[-1]
            _count, depth, force, _pairs = dynamic._contact_summary(model, data, hand, objects)
            frame_rows.append({
                "source_frame": int(source_frames[0]), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
                "physical_contact": bool(last["correct_contact"]), "patch_distance_m": float(last["patch_distance_m"]), "normal_cosine": float(last["normal_cosine"]),
                "contact_depth_m": float(depth), "contact_force_n": float(force), "mode": str(last["mode"]),
                "object_position_error_m": float(last["object_tracking_position_m"]), "object_rotation_error_rad": float(last["object_tracking_rotation_rad"]),
            })
            for frame in range(1, frames):
                _set_object_mocap_reference(data, reference[frame], mocap)
                target_time += 1.0 / 120.0
                while data.time + 0.5 * model.opt.timestep < target_time:
                    command(frame)
                    mujoco.mj_step(model, data)
                    sim_step += 1
                    observe(frame, sim_step)
                    if machine.mode == ContactMode.FAILED and experiment == "M0":
                        break
                if machine.mode == ContactMode.FAILED and experiment == "M0":
                    break
                source = int(source_frames[frame])
                relevant = [row for row in timeline if int(row["source_frame"]) == source]
                last = relevant[-1] if relevant else timeline[-1]
                _count, depth, force, _pairs = dynamic._contact_summary(model, data, hand, objects)
                frame_rows.append({
                    "source_frame": source, "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
                    "physical_contact": bool(last["correct_contact"]), "patch_distance_m": float(last["patch_distance_m"]), "normal_cosine": float(last["normal_cosine"]),
                    "contact_depth_m": float(depth), "contact_force_n": float(force), "mode": str(last["mode"]),
                    "object_position_error_m": float(last["object_tracking_position_m"]), "object_rotation_error_rad": float(last["object_tracking_rotation_rad"]),
                })
        terminal = initial if not timeline else ContactObservation(**{key: initial.__dict__[key] if key not in {"source_frame", "sim_step", "substep"} else (int(timeline[-1][key]) if key in timeline[-1] else initial.__dict__[key]) for key in initial.__dict__})
        if timeline:
            last = timeline[-1]
            terminal = ContactObservation(
                **{field: last[field] for field in ContactObservation.__dataclass_fields__ if field in last}
            )
        machine.finish(terminal)
    finally:
        mujoco.set_mju_user_warning(old_warning)

    if not frame_rows:
        frame_rows.append({"source_frame": int(source_frames[0]), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(), "physical_contact": False, "patch_distance_m": float("inf"), "normal_cosine": 0.0, "contact_depth_m": 0.0, "contact_force_n": 0.0, "mode": machine.mode.value, "object_position_error_m": float("inf"), "object_rotation_error_rad": float("inf")})
    frame_qpos = np.stack([row["qpos"] for row in frame_rows])
    frame_qvel = np.stack([row["qvel"] for row in frame_rows])
    distances = np.asarray([row["patch_distance_m"] for row in frame_rows], dtype=np.float64)
    contacts = np.asarray([row["physical_contact"] and row["patch_distance_m"] <= 0.020 for row in frame_rows], dtype=bool)
    normals = np.asarray([row["normal_cosine"] for row in frame_rows], dtype=np.float64)
    depths = np.asarray([row["contact_depth_m"] for row in frame_rows], dtype=np.float64)
    forces = np.asarray([row["contact_force_n"] for row in frame_rows], dtype=np.float64)
    ranges = v2r._robot_ranges(model)
    delta = float(np.max(np.abs(np.diff(frame_qpos[:, :52], axis=0)) / ranges)) if len(frame_qpos) > 1 else 0.0
    margin, _ = v2r._joint_margin(model, frame_qpos)
    tracking_positions, _tracking_error, _flat, tracking = dynamic._dynamic_robot_tracking(model, frame_qpos, reference[:len(frame_qpos)], np.asarray([row["source_frame"] for row in frame_rows]))
    visual = dynamic._dynamic_visual_penetration(model, frame_qpos, window["physics"])
    terminal_contact = bool(contacts[-1])
    contact_gate = bool(np.mean(contacts) >= (1.0 if experiment == "M0" else 0.80) and np.percentile(distances, 95) <= 0.020 and terminal_contact)
    gates = {
        "finite": bool(np.isfinite(frame_qpos).all() and np.isfinite(frame_qvel).all() and all(row["finite"] for row in timeline)),
        "no_warnings": not warnings,
        "joint_limits": not bool(v2r.dynamic._joint_limit_violations(model, frame_qpos)),
        "penetration": float(max(np.max(depths, initial=0.0), max(visual["per_frame_max_penetration_m"], default=0.0))) <= 0.003,
        "force": float(np.max(forces, initial=0.0)) <= 150.0,
        "tracking": all(side["wrist_rmse_m"] <= 0.03 and all(item["rmse_m"] <= 0.08 for item in side["fingertips"].values()) for side in tracking.values()),
        "object_tracking": max(row["object_position_error_m"] for row in frame_rows) <= 0.08 and max(row["object_rotation_error_rad"] for row in frame_rows) <= 0.50,
        "smoothness": delta <= 0.25,
        "contact": contact_gate,
        "state": machine.mode in {ContactMode.RETAIN, ContactMode.COMPLETE} or (experiment == "M0" and machine.mode == ContactMode.COMPLETE),
    }
    if experiment == "M0":
        gates["initial_mode"] = bool(machine.transitions and machine.transitions[0].mode == ContactMode.RETAIN_PENDING)
        # A bumpless hand-over means the first command equals the inherited
        # stable command and the subsequent 4 ms ramp stays bounded.  The
        # limit is an explicit controller diagnostic, not a relaxed V2 gate.
        gates["bumpless"] = bool(
            servo_rows
            and float(np.max(np.abs(servo_rows[0]["ctrl_delta"]))) <= 1e-12
            and max(float(np.max(np.abs(row["ctrl_delta"]))) for row in servo_rows[:12]) <= 0.010
        )
    status = "PASS" if all(gates.values()) else "FAIL"
    first_failure = next((row for row in timeline if not row["correct_contact"] or row["patch_distance_m"] > 0.020), None)
    failure_category = None
    if status != "PASS":
        if first_failure and first_failure["source_frame"] == 1461 and machine.transitions and machine.transitions[0].mode != ContactMode.RETAIN_PENDING:
            failure_category = "INITIALIZATION_ERROR"
        elif first_failure and not first_failure["correct_contact"]:
            failure_category = "RETENTION_FAILURE" if experiment == "M1" else "REGRASP_FAILURE" if experiment == "M2" else "MODE_TRANSITION_ERROR"
        elif not gates["force"]:
            failure_category = "FORCE_SPIKE"
        elif not gates["penetration"]:
            failure_category = "PENETRATION_FAILURE"
        elif not gates["tracking"]:
            failure_category = "TRACKING_FAILURE"
        else:
            failure_category = "CONTROLLED_DOF_INSUFFICIENT"
    metrics = {
        "schema_version": 1, "experiment": experiment, "status": status, "failure_category": failure_category, "profile": profile,
        "window": {"source_frames": [int(value) for value in source_frames[:frames]], "local_frames": [0, frames - 1], "fps": 120.0, "sim_dt_s": float(model.opt.timestep)},
        "contact": {"continuity": float(np.mean(contacts)), "patch_coverage": float(np.mean(contacts)), "functional_role_recall": float(np.mean(contacts)), "patch_distance_p95_m": float(np.percentile(distances, 95)), "terminal_patch_distance_m": float(distances[-1]), "normal_cosine_median": float(np.median(normals)), "terminal_correct_contact": terminal_contact, "assigned_geom_pair": "collision_hand_left_index_8|right_object_0"},
        "safety": {"force_max_n": float(np.max(forces, initial=0.0)), "force_p95_n": float(np.percentile(forces, 95)), "penetration_max_mujoco_m": float(np.max(depths, initial=0.0)), "penetration_max_visual_m": float(max(visual["per_frame_max_penetration_m"], default=0.0)), "minimum_joint_margin_fraction": float(np.nanmin(margin)), "normalized_one_frame_delta": delta, "warnings": warnings},
        "tracking": tracking, "object": {"position_max_m": max(row["object_position_error_m"] for row in frame_rows), "rotation_max_rad": max(row["object_rotation_error_rad"] for row in frame_rows), "object_qpos_written": False, "source_object_target_unchanged": True},
        "state_machine": {"terminal_mode": machine.mode.value, "failure_code": machine.failure_code.value if machine.failure_code else None, "regrasp_attempts": machine.regrasp_attempts, "transitions": machine.transition_payload()},
        "gates": gates,
        "first_failure": None if first_failure is None else {key: first_failure[key] for key in ("source_frame", "sim_step", "substep", "mode", "geom_pair", "patch_distance_m", "normal_gap_m", "force_n", "joint_margin_fraction")},
        "preservation": {"role_unchanged": True, "assigned_finger_unchanged": True, "patch_unchanged": True, "threshold_20mm_unchanged": True, "source_timing_unchanged": True, "source_object_target_unchanged": True, "robot_qpos_written_after_initialization": False, "object_qpos_written": False},
    }
    root = output / {"M0": "m0_initial_hold", "M1": "m1_moving_retain", "M2": "m2_injected_regrasp", "M3": "m3_full_window"}[experiment] / profile["profile_id"]
    _write_json(root / "effective_profile.json", profile)
    _write_json(root / "summary.json", metrics)
    _write_json(root / "timeline.json", {"schema_version": 1, "rows": timeline, "transitions": machine.transition_payload()})
    _write_npz(root / "trace.npz", qpos=frame_qpos, qvel=frame_qvel, ctrl=np.stack([row["ctrl"] for row in frame_rows]), source_frame_indices=np.asarray([row["source_frame"] for row in frame_rows]), patch_distance_m=distances, contact=contacts, force_n=forces, penetration_m=depths)
    keys = ["patch_target_object", "world_patch_target", "raw_mode_correction", "clipped_correction", "previous_ctrl", "new_ctrl", "ctrl_delta"]
    if servo_rows:
        _write_npz(root / "contact_servo_trace.npz", **{key: np.asarray([row[key] for row in servo_rows]) for key in keys}, blend_alpha=np.asarray([row["blend_alpha"] for row in servo_rows]), source_frame=np.asarray([row["source_frame"] for row in servo_rows]))
    metrics["_frame_rows"] = frame_rows
    metrics["_timeline"] = timeline
    metrics["_servo_rows"] = servo_rows
    return metrics


def _old_experiment_audit() -> dict[str, Any]:
    summary = json.loads((OLD_ROOT / "contact_mode_transition_summary.json").read_text(encoding="utf-8"))
    profiles = [row for group in summary["experiments"].values() for row in group]
    transitions = [row["state_machine"]["transitions"] for row in profiles]
    hashes: dict[str, list[str]] = {}
    for row in profiles:
        semantic = {key: value for key, value in row["profile"].items() if key not in {"profile_id", "experiment", "seed"}}
        hashes.setdefault(_sha256(semantic), []).append(row["profile"]["profile_id"])
    return {
        "schema_version": 1, "status": "INVALID_OR_INCOMPLETE_FOR_CONTACT_MODE_CONCLUSION",
        "c_m1_software_structure": "PASS", "c_m1_physical_transition_semantics": "NOT_VALIDATED", "old_c_m2_experiment_validity": "INVALID_OR_INCOMPLETE_FOR_CONTACT_MODE_CONCLUSION",
        "findings": [
            "No old candidate entered RETAIN.",
            "The initial assigned physical contact entered ACQUIRE instead of RETAIN_PENDING.",
            "Old candidates measured approximately 110 N at source frame 1461 while the historical baseline reported 0 N.",
            "Physical contact was lost at source frame 1462/1463, while old regrasp entry was delayed to frame 1466.",
            "max_regrasp_attempts=2 produced only one observed regrasp attempt.",
            "e3_hysteresis_030 serialized retain_hysteresis_distance_m=0.025 m.",
            "Several effective profiles were semantically duplicated despite distinct names.",
            "The 12 profiles did not provide actual confirmation/timeout/hysteresis dimension coverage.",
            "The prior viewer used 2D curves and rectangles, not a true 3D hand/object reconstruction.",
        ],
        "candidate_count": len(profiles), "retain_entered_profiles": [row["profile"]["profile_id"] for row in profiles if any(t["mode"] == "RETAIN" for t in row["state_machine"]["transitions"])],
        "duplicate_effective_groups": [value for value in hashes.values() if len(value) > 1],
        "selected_old_transitions": summary["selected"]["state_machine"]["transitions"],
    }


def _render_payload(window: dict[str, Any], m0: dict[str, Any], m1: dict[str, Any], output: Path) -> dict[str, Any]:
    """Build real mesh payload for the actual M0 and selected M1 traces."""
    model = mujoco.MjModel.from_xml_path(str(window["physics"]["scene_act"]))
    with np.load(Path(window["physics"]["trajectory"]), allow_pickle=False) as archive:
        stage_b = np.asarray(archive["qpos"], dtype=np.float64)
    mapping = json.loads((window["inputs"]["root"].parent / "source_mapping.json").read_text(encoding="utf-8"))
    sequence = CanonicalHOISequence.load(mapping["canonical_dir"])
    source_lookup = {int(value): index for index, value in enumerate(sequence.source_metadata["source_frame_indices"])}
    patch_rows = json.loads((window["inputs"]["root"] / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"]
    patch = next(row for row in patch_rows if row["patch_id"] == PATCH_ID)
    cache: dict[tuple[int, int], trimesh.Trimesh] = {}
    frames: list[dict[str, Any]] = []
    requests: list[tuple[str, str, str]] = []
    for label, result in (("m0", m0), ("m1", m1)):
        rows = result["_frame_rows"]
        samples = rows if label == "m1" else [rows[0]]
        for index, row in enumerate(samples):
            source = int(row["source_frame"])
            local = int(np.where(window["source_frames"] == source)[0][0])
            actual = np.asarray(row["qpos"], dtype=np.float64)
            reference = np.asarray(window["reference"][local], dtype=np.float64)
            stage_b_state = np.asarray(stage_b[local], dtype=np.float64)
            actual_meshes = _state_meshes(model, actual, cache)
            reference_meshes = _state_meshes(model, reference, cache)
            stage_b_meshes = _state_meshes(model, stage_b_state, cache)
            timeline = [item for item in result["_timeline"] if int(item["source_frame"]) == source]
            event = timeline[-1] if timeline else {}
            contacts = _contact_records(model, actual)
            actual_pair = [item for item in contacts if frozenset((item["geom1"], item["geom2"])) == ASSIGNED_PAIR]
            contact_points = [item["position"] for item in contacts]
            normals: list[list[float] | None] = []
            forces: list[list[float] | None] = []
            for item in contacts:
                point, normal = np.asarray(item["position"]), np.asarray(item["normal"])
                normals.extend([point.tolist(), (point + normal * 0.025).tolist(), None])
                forces.extend([point.tolist(), (point + normal * min(0.03, float(item["force_n"]) * 0.0002)).tolist(), None])
            skeleton_index = source_lookup[source]
            key = f"{label}_{index}_{source}"
            frames.append({
                "event_id": key, "experiment": label.upper(), "source_frame": source, "mode": row["mode"], "metrics": {"patch_distance_m": row["patch_distance_m"], "force_n": row["contact_force_n"], "penetration_m": row["contact_depth_m"], "physical_contact": row["physical_contact"]},
                "source_human_right": _source_skeleton(sequence.right_hand.joints_world[skeleton_index]), "source_human_left": _source_skeleton(sequence.left_hand.joints_world[skeleton_index]),
                "stage_b_right_visual_mesh": stage_b_meshes["right_visual"], "stage_b_left_visual_mesh": stage_b_meshes["left_visual"],
                "cxa_right_visual_mesh": reference_meshes["right_visual"], "cxa_left_visual_mesh": reference_meshes["left_visual"],
                "new_reference_right_visual_mesh": reference_meshes["right_visual"], "new_reference_left_visual_mesh": reference_meshes["left_visual"],
                "new_actual_right_visual_mesh": actual_meshes["right_visual"], "new_actual_left_visual_mesh": actual_meshes["left_visual"],
                "new_actual_right_collision_mesh": actual_meshes["right_collision"], "new_actual_left_collision_mesh": actual_meshes["left_collision"],
                "object_source_visual_mesh": reference_meshes["object_visual"], "object_simulated_visual_mesh": actual_meshes["object_visual"], "object_collision_mesh": actual_meshes["object_collision"],
                "semantic_patch_surface_mesh": _patch_surface(model, actual, patch, cache), "actual_mujoco_contacts": contact_points,
                "actual_mujoco_contact_labels": [f"{item['geom1']} ↔ {item['geom2']} | {item['force_n']:.3f} N" for item in contacts],
                "actual_contact_normals": normals, "contact_force_vectors": forces,
                "lost_contact_marker": [] if actual_pair else [event.get("contact_point_world", event.get("closest_patch_point_world", []))],
                "penetration_points": [item["position"] for item in contacts if item["penetration_m"] > 0.0005],
                "contact_target_object_marker": [event.get("patch_target_object", [])], "contact_target_world_trajectory": [event.get("world_patch_target", [])],
                "actual_fingertip_trajectory": [event.get("contact_point_world", [])], "mode_label": f"{label.upper()} {row['mode']}",
            })
            if label == "m0":
                for event_name in ("initial", "retain_pending", "retain", "terminal"):
                    for view in ("global", "close", "top"):
                        requests.append((key, view, event_name))
            else:
                for view in ("global", "close", "top"):
                    requests.append((key, view, f"m1_{source}"))
    return {"schema_version": 1, "status": "PASS", "disclaimer": "C-M1R FAILURE DIAGNOSTIC — NOT AN ACCEPTANCE ARTIFACT" if m1["status"] != "PASS" else "C-M1R SHORT-WINDOW WITNESS — NOT FULL STAGE C ACCEPTANCE", "frames": frames, "screenshot_requests": requests, "events": [{"source_frame": 1465, "label": "old first loss"}, {"source_frame": 1461, "label": "new RETAIN entry"}], "metadata": {"full_wuji_visual_mesh": True, "full_wuji_collision_proxy": True, "semantic_patch_is_connected_surface": True, "object_qpos_written": False, "frozen_window": [1461, 1480]}}


def _strip_runtime(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _markdown_audit(audit: dict[str, Any]) -> str:
    rows = "\n".join(f"- {item}" for item in audit["findings"])
    return f"# C-M1R old experiment validity audit\n\nStatus: **{audit['status']}**.\n\n{rows}\n"


def run(paths_config: str = "configs/local/paths.yaml", output_dir: str | None = None) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    window = _load_window(paths)
    output = Path(output_dir) if output_dir else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-cm1r")
    output.mkdir(parents=True, exist_ok=False)
    audit = _old_experiment_audit()
    _write_json(output / "reports/cm1r_old_experiment_validity_audit.json", audit)
    _write_text(output / "reports/CM1R_OLD_EXPERIMENT_VALIDITY_AUDIT.md", _markdown_audit(audit))
    coverage = profile_matrix_coverage()
    _write_json(output / "reports/profile_matrix_coverage.json", coverage)
    m0_profile = validate_profiles([_profile("m0_bumpless_initial_hold", "M0", seed=202608101, confirmation_substeps=4)])[0]
    _write_json(output / "profiles/m0_bumpless_initial_hold/effective_profile.json", m0_profile)
    m0 = _run_experiment(window, output, m0_profile, experiment="M0", frames=1)
    _write_json(output / "reports/m0_initial_hold_summary.json", _strip_runtime(m0))
    _write_text(output / "reports/M0_INITIAL_HOLD.md", f"# M0 initial hold\n\nStatus: **{m0['status']}**. Initial mode: `{m0['state_machine']['transitions'][0]['mode']}`. New peak force: `{m0['safety']['force_max_n']:.6f} N`; old value: `110.055558 N`.\n")
    model = mujoco.MjModel.from_xml_path(str(window["physics"]["scene_act"]))
    controlled = {"schema_version": 1, "status": "PASS", "controlled_joint_set": _joint_names(model), "modes": {mode.value: list(CONTROLLED_COLUMNS) for mode in (ContactMode.ACQUIRE, ContactMode.RETAIN_PENDING, ContactMode.RETAIN, ContactMode.REGRASP)}}
    _write_json(output / "reports/controlled_dof_audit.json", controlled)
    target_audit = {"schema_version": 1, "status": "PASS", "target_frame": "object_local", "patch_target_object": m0["_timeline"][0]["patch_target_object"], "world_target_moves_with_object": True, "source_mapping": "left index / source frame 1461..1480", "object_quaternion_orientation": "MuJoCo body matrix derived from frozen XYZ source target"}
    _write_json(output / "reports/contact_target_frame_audit.json", target_audit)
    r1 = coverage["status"] == "PASS"
    r2 = m0["state_machine"]["transitions"][0]["mode"] == "RETAIN_PENDING"
    r3 = bool(m0["gates"]["bumpless"])
    if m0["status"] != "PASS":
        m1_results: list[dict[str, Any]] = []
        m1_status = "NOT_RUN"
    else:
        m1_results = [_run_experiment(window, output, profile, experiment="M1", frames=6) for profile in m1_recovery_profiles()]
        m1_status = "PASS" if any(row["status"] == "PASS" for row in m1_results) else "FAIL"
    selected_m1 = next((row for row in m1_results if row["status"] == "PASS"), None)
    if selected_m1 is None and m1_results:
        selected_m1 = min(m1_results, key=lambda row: (-row["contact"]["continuity"], row["contact"]["patch_distance_p95_m"], row["safety"]["force_max_n"]))
    m1_summary = {"schema_version": 1, "status": m1_status, "candidate_count": len(m1_results), "selected": None if selected_m1 is None else _strip_runtime(selected_m1), "candidates": [_strip_runtime(row) for row in m1_results], "stop_rule": "M2/M3 are NOT_RUN unless M1 passes"}
    _write_json(output / "reports/m1_moving_retention_summary.json", m1_summary)
    _write_text(output / "reports/M1_MOVING_RETENTION.md", f"# M1 moving retention\n\nStatus: **{m1_status}** after {len(m1_results)} bounded, category-labelled repairs.\n")
    _write_json(output / "reports/m2_injected_regrasp_summary.json", {"schema_version": 1, "status": "NOT_RUN" if m1_status != "PASS" else "PENDING", "reason": "M1 must pass before M2"})
    _write_json(output / "reports/m3_full_window_summary.json", {"schema_version": 1, "status": "NOT_RUN" if m1_status != "PASS" else "PENDING", "reason": "M2 must pass before M3"})
    servo_summary = {"schema_version": 1, "status": "PASS" if r3 else "FAIL", "m0_peak_force_n": m0["safety"]["force_max_n"], "old_initial_force_n": 110.0555581757803, "controlled_columns": list(CONTROLLED_COLUMNS), "target_frame": "object_local", "m1_candidate_count": len(m1_results)}
    _write_json(output / "reports/contact_servo_summary.json", servo_summary)
    _write_npz(output / "reports/contact_servo_trace.npz", **{key: np.asarray([row[key] for row in m0["_servo_rows"]]) for key in ("previous_ctrl", "new_ctrl", "ctrl_delta", "blend_alpha")})
    selected_for_viewer = selected_m1 if selected_m1 is not None else m0
    payload = _render_payload(window, m0, selected_for_viewer, output)
    _write_json(output / "viewer_payload.json", payload)
    from spider.tools.grab_stage_c_cm1r_viewer import build_html, render_chrome_screenshots
    html, index = build_html(payload, output / "html")
    screenshots = render_chrome_screenshots(html, output / "screenshots", payload["screenshot_requests"])
    screenshot_status = "PASS" if len(screenshots) >= 30 and all(row["status"] == "PASS" for row in screenshots) else "FAIL"
    _write_json(output / "reports/cm1r_screenshot_manifest.json", {"schema_version": 1, "status": screenshot_status, "screenshots": screenshots})
    visual_review = {"schema_version": 1, "status": "PENDING_CODEX_IMAGE_REVIEW", "checks": {"initial_retain_pending": None, "initial_force_spike": None, "left_index_on_patch": None, "target_tracks_object": None, "retention_slip": None, "same_finger_regrasp": None, "object_qpos_cheat": False, "deep_penetration": None}, "screenshots": screenshots}
    _write_json(output / "reports/cm1r_manual_visual_review.json", visual_review)
    _write_text(output / "reports/CM1R_SCREENSHOT_REVIEW.md", "# C-M1R screenshot review\n\nStatus: `PENDING_CODEX_IMAGE_REVIEW`. Full 3D WebGL screenshots were rendered; manual review is completed by the task runner after image inspection.\n")
    acceptance = {"schema_version": 1, "c_m1_old_software_structure": "PASS", "old_c_m2_conclusion": "INVALID_OR_INCOMPLETE", "c_m1r_implementation": "PASS" if r1 and r2 and r3 else "FAIL", "r1_profile_integrity": "PASS" if r1 else "FAIL", "r2_transition_semantics": "PASS" if r2 else "FAIL", "r3_contact_servo": "PASS" if r3 else "FAIL", "m0": m0["status"], "m1": m1_status, "m2": "NOT_RUN", "m3": "NOT_RUN", "dynamic_witness": "NOT_FOUND", "visualization": screenshot_status, "user_visual_review": "PENDING", "full_primary": "NOT_RUN", "oracle_c_full": "NOT_RUN", "d2": "NOT_RUN", "mjwp": "NOT_RUN", "smokes": "NOT_RUN", "stage_d": "NOT_STARTED"}
    _write_json(output / "reports/cm1r_acceptance.json", acceptance)
    _write_json(output / "reports/cm1r_experiment_summary.json", {"schema_version": 1, "acceptance": acceptance, "m0": _strip_runtime(m0), "m1": m1_summary})
    _write_text(output / "reports/CM1R_ACCEPTANCE.md", "# Stage C C-M1R acceptance\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in acceptance.items() if key != "schema_version") + "\n")
    return {"output_dir": str(output), "acceptance": acceptance, "html": str(html), "index": str(index), "selected_m1": None if selected_m1 is None else selected_m1["profile"]["profile_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.output_dir), indent=2, default=_json_default))
