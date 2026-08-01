"""C-M1/C-M2 bounded V2 contact-mode transition experiment.

Only the frozen first-loss window is simulated here.  The runner consumes the
corrected C-XA Level-1 input read-only, drives both object mocap targets from
the unchanged source object trajectory, and records real MuJoCo substeps.  It
never writes a simulated object qpos or rewrites a robot qpos after the
initial state assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from spider.contact.contact_mode import (
    ContactMode,
    ContactModeConfig,
    ContactModeMachine,
    ContactObservation,
    observation_payload,
)
from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools.grab_stage_c import (
    _finite_data,
    _object_tracking_error,
    _preflight_object_ids,
    _set_object_mocap_reference,
    _site_ids,
)


PRIMARY = dynamic.PRIMARY
WINDOW_SOURCE_FRAMES = np.arange(1461, 1481, dtype=np.int64)
DIAGNOSTIC_ROOT = Path(".local_artifacts/stage_c_v2r2e_failure_diagnostic/20260731T180415Z")
OUTPUT_ROOT = Path(".local_artifacts/stage_c_cm1_cm2")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
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


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()


def _load_window(paths) -> dict[str, Any]:
    inputs = dynamic._inputs(paths, PRIMARY)
    physics = dynamic._physics(paths, PRIMARY)
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        reference = np.asarray(archive["qpos"], dtype=np.float64)
        reference_qvel = np.asarray(archive["qvel"], dtype=np.float64)
    with np.load(inputs["reference_npz"], allow_pickle=False) as archive:
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)[1:-1]
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool)
        anchors = np.asarray(archive["anchors"], dtype=np.float64)
        normals = np.asarray(archive["normals"], dtype=np.float64)
    indices = np.flatnonzero(np.isin(source_frames, WINDOW_SOURCE_FRAMES))
    if not np.array_equal(source_frames[indices], WINDOW_SOURCE_FRAMES) or not np.array_equal(indices, np.arange(20)):
        raise RuntimeError("C-M2 requires the frozen local window 0..19 / source 1461..1480")
    if reference.shape != (414, 64) or reference_qvel.shape != (414, 64) or expected.shape != (414, 10):
        raise RuntimeError("C-M2 frozen C-XA input schema changed")
    return {
        "inputs": inputs,
        "physics": physics,
        "reference": reference[:20],
        "reference_qvel": reference_qvel[:20],
        "source_frames": source_frames[:20],
        "expected": expected[:20],
        "anchors": anchors[:20],
        "normals": normals[:20],
        "input_hashes": {name: _sha256(path) for name, path in inputs.items() if name != "root"},
    }


def _load_patch(mesh_path: Path, inputs: dict[str, Path]) -> tuple[trimesh.Trimesh, dict[str, Any], str, str]:
    assignment = json.loads(inputs["assignment_json"].read_text(encoding="utf-8"))
    selected = [row for row in assignment["selected"] if row["role_id"] == "s5__cylindermedium_lift:0"]
    if len(selected) != 1 or selected[0]["selected_robot_region"] != "left_index_fingertip":
        raise RuntimeError("C-M2 immutable left-index SUPPORT mapping is missing or changed")
    role_id = selected[0]["role_id"]
    patches = json.loads((inputs["root"] / "source_contact_patches.json").read_text(encoding="utf-8"))["patches"]
    patch = next(row for row in patches if row["patch_id"] == selected[0]["source_patch_id"])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError("C-M2 semantic patch mesh is empty")
    face_ids = np.asarray(patch["extended_face_ids"], dtype=np.int64)
    if len(face_ids) == 0 or np.any(face_ids < 0) or np.any(face_ids >= len(mesh.faces)):
        raise RuntimeError("C-M2 immutable semantic patch face selection is invalid")
    patch_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[face_ids], process=False)
    return patch_mesh, patch, role_id, selected[0]["selected_robot_region"]


def _patch_measure(data: mujoco.MjData, model: mujoco.MjModel, site_id: int, body_id: int, patch_mesh: trimesh.Trimesh) -> tuple[float, float, np.ndarray, np.ndarray]:
    tip = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    position = np.asarray(data.xpos[body_id], dtype=np.float64)
    matrix = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    local = (tip - position) @ matrix
    closest, distance, face_index = trimesh.proximity.closest_point_naive(patch_mesh, local.reshape(1, 3))
    closest_local = np.asarray(closest[0], dtype=np.float64)
    closest_world = closest_local @ matrix.T + position
    normal_world = np.asarray(patch_mesh.face_normals[int(face_index[0])], dtype=np.float64) @ matrix.T
    vector = tip - closest_world
    denom = float(np.linalg.norm(vector) * np.linalg.norm(normal_world))
    cosine = float(np.dot(vector, normal_world) / denom) if denom > 1e-12 else 1.0
    return float(distance[0]), cosine, closest_world, normal_world


def _contact_correction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target: np.ndarray,
    gain: float,
    damping: float,
    finger_clip: float,
    wrist_translation_clip: float,
    wrist_rotation_clip: float,
) -> np.ndarray:
    if gain <= 0.0:
        return np.zeros(52, dtype=np.float64)
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, None, site_id)
    block = jacobian[:, 26:52]
    error = np.asarray(target, dtype=np.float64) - np.asarray(data.site_xpos[site_id], dtype=np.float64)
    delta = block.T @ np.linalg.solve(block @ block.T + damping * np.eye(3), error) * gain
    clips = np.full(26, finger_clip, dtype=np.float64)
    clips[:3] = wrist_translation_clip
    clips[3:6] = wrist_rotation_clip
    result = np.zeros(52, dtype=np.float64)
    result[26:52] = np.clip(delta, -clips, clips)
    return result


def _profile_specs() -> list[dict[str, Any]]:
    """Fixed, representative matrix: 3 E1 + 3 E2 + 6 E3 = 12 profiles."""
    common = {"acquire_entry_distance_m": 0.025, "retain_hysteresis_distance_m": 0.025, "confirmation_substeps": 4, "acquire_timeout_ms": 40, "regrasp_timeout_ms": 40}
    return [
        {"experiment": "E1", "profile_id": "e1_baseline", "seed": 20260801, "allow_regrasp": False, "max_regrasp_attempts": 0, "lead_source_frames": 20, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 0.0, **common},
        {"experiment": "E1", "profile_id": "e1_short_lead", "seed": 20260802, "allow_regrasp": False, "max_regrasp_attempts": 0, "lead_source_frames": 8, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 0.0, **common},
        {"experiment": "E1", "profile_id": "e1_feedback", "seed": 20260803, "allow_regrasp": False, "max_regrasp_attempts": 0, "lead_source_frames": 8, "kp_scale": 1.25, "state_feedback_gain": 0.50, "contact_ik_gain": 0.0, **common},
        {"experiment": "E2", "profile_id": "e2_anchor_regrasp", "seed": 20260804, "allow_regrasp": True, "max_regrasp_attempts": 1, "lead_source_frames": 20, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 5.0, "contact_ik_damping": 0.002, **common},
        {"experiment": "E2", "profile_id": "e2_strong_regrasp", "seed": 20260805, "allow_regrasp": True, "max_regrasp_attempts": 1, "lead_source_frames": 20, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 10.0, "contact_ik_damping": 0.0001, **common},
        {"experiment": "E2", "profile_id": "e2_feedback_regrasp", "seed": 20260806, "allow_regrasp": True, "max_regrasp_attempts": 1, "lead_source_frames": 8, "kp_scale": 1.25, "state_feedback_gain": 0.50, "contact_ik_gain": 5.0, "contact_ik_damping": 0.002, **common},
        {"experiment": "E3", "profile_id": "e3_hysteresis_030", "seed": 20260807, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 20, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 5.0, "contact_ik_damping": 0.002, "retain_hysteresis_distance_m": 0.030, **common},
        {"experiment": "E3", "profile_id": "e3_strong_anchor", "seed": 20260808, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 20, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 10.0, "contact_ik_damping": 0.0001, **common},
        {"experiment": "E3", "profile_id": "e3_short_lead_anchor", "seed": 20260809, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 8, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 10.0, "contact_ik_damping": 0.0001, **common},
        {"experiment": "E3", "profile_id": "e3_no_lead_anchor", "seed": 20260810, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 0, "kp_scale": 1.0, "state_feedback_gain": 0.0, "contact_ik_gain": 10.0, "contact_ik_damping": 0.0001, **common},
        {"experiment": "E3", "profile_id": "e3_high_kp_anchor", "seed": 20260811, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 12, "kp_scale": 1.50, "state_feedback_gain": 0.0, "contact_ik_gain": 5.0, "contact_ik_damping": 0.002, **common},
        {"experiment": "E3", "profile_id": "e3_feedback_anchor", "seed": 20260812, "allow_regrasp": True, "max_regrasp_attempts": 2, "lead_source_frames": 8, "kp_scale": 1.25, "state_feedback_gain": 0.50, "contact_ik_gain": 10.0, "contact_ik_damping": 0.0001, **common},
    ]


def _historical_baseline(window: dict[str, Any]) -> dict[str, Any]:
    timeline = json.loads((DIAGNOSTIC_ROOT / "first_failure_timeline.json").read_text(encoding="utf-8"))
    with np.load(DIAGNOSTIC_ROOT / "first_failure_trace.npz", allow_pickle=False) as archive:
        distance = np.asarray(archive["patch_distance_m"], dtype=np.float64)[:20, 6]
        depth = np.asarray(archive["contact_depth_m_by_frame"], dtype=np.float64)[:20]
        force = np.asarray(archive["contact_force_n_by_frame"], dtype=np.float64)[:20]
    return {
        "status": "REPLAYED_HISTORICAL_ARTIFACT",
        "rerun": False,
        "source": str(DIAGNOSTIC_ROOT),
        "first_failure_source_frame": timeline["first_failure_source_frame"],
        "first_failure_local_frame": timeline["local_frame"],
        "patch_distance_m": distance.tolist(),
        "contact_depth_m": depth.tolist(),
        "force_n": force.tolist(),
        "window_source_frames": WINDOW_SOURCE_FRAMES.tolist(),
    }


def _failure_category(metrics: dict[str, Any]) -> str:
    if not metrics["gates"]["finite"] or not metrics["gates"]["no_warnings"]:
        return "NUMERICAL_FAILURE"
    if not metrics["gates"]["joint_limits"]:
        return "JOINT_MARGIN_FAILURE"
    if not metrics["gates"]["penetration"]:
        return "PENETRATION_FAILURE"
    if not metrics["contact"]["correct_physical_pair"]:
        return "WRONG_REGION_CONTACT"
    if not metrics["contact"]["patch_distance_p95"] <= 0.020:
        return "RETENTION_FAILURE"
    if not metrics["gates"]["tracking"]:
        return "TRACKING_FAILURE"
    if not metrics["gates"]["object_tracking"]:
        return "OBJECT_FAILURE"
    return "SAFETY_OR_TERMINAL_FAILURE"


def _run_profile(window: dict[str, Any], output: Path, profile: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(profile["seed"]))
    del rng  # The profile matrix is deterministic; the seed is part of its audit identity.
    reference = window["reference"]
    reference_qvel = window["reference_qvel"]
    source_frames = window["source_frames"]
    expected = window["expected"]
    anchors = window["anchors"]
    physics = window["physics"]
    inputs = window["inputs"]
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    model.opt.timestep = 0.0005
    model.eq_solref[:, 0] = 0.001
    model.actuator_gainprm[:52, 0] *= float(profile["kp_scale"])
    model.actuator_biasprm[:52, 1] *= float(profile["kp_scale"])
    hand, objects = v2r._contact_ids(model)
    bodies, mocap = _preflight_object_ids(model)
    sites = _site_ids(model)
    patch_mesh, _patch, role_id, robot_region = _load_patch(Path(physics["collision_cache"]) / "visual/visual.obj", inputs)
    object_body = bodies["right"]
    tip_site = sites[8]  # left_index in the immutable _site_ids ordering
    wrist_site = sites[6]
    config = v2r._config()
    feedforward = np.zeros((20, 52), dtype=np.float64)
    if float(profile.get("inverse_dynamics_scale", 0.0)):
        full_ff = v2r._inverse_dynamics_feedforward(model, reference, reference_qvel, config, profile)
        feedforward = full_ff[:20, :52]
    machine = ContactModeMachine(
        ContactModeConfig(
            acquire_entry_distance_m=float(profile["acquire_entry_distance_m"]),
            retain_hysteresis_distance_m=float(profile["retain_hysteresis_distance_m"]),
            confirmation_substeps=int(profile["confirmation_substeps"]),
            acquire_timeout_ms=int(profile["acquire_timeout_ms"]),
            regrasp_timeout_ms=int(profile["regrasp_timeout_ms"]),
            max_regrasp_attempts=int(profile["max_regrasp_attempts"]),
            allow_regrasp=bool(profile["allow_regrasp"]),
        )
    )
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = reference_qvel[0]
    _set_object_mocap_reference(data, reference[0], mocap)
    mujoco.mj_forward(model, data)
    target = mujoco.MjData(model)
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    timeline: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    previous_contact_position: np.ndarray | None = None
    sim_step = 0
    target_time = 0.0

    def observe(frame: int, substep: int, force_impulse: float) -> ContactObservation:
        nonlocal previous_contact_position
        count, depth, force, pairs = dynamic._contact_summary(model, data, hand, objects)
        exact = [row for row in pairs if {"collision_hand_left_index_8", "right_object_0"} == set(row["geom_pair"].split("|"))]
        pair = exact[0] if exact else (pairs[0] if pairs else None)
        physical = bool(exact)
        distance, cosine, closest, normal = _patch_measure(data, model, tip_site, object_body, patch_mesh)
        contact_position = np.asarray(pair["position_world"], dtype=np.float64) if pair else closest
        slip = 0.0 if previous_contact_position is None else float(np.linalg.norm(contact_position - previous_contact_position))
        previous_contact_position = contact_position.copy()
        target.qpos[:] = reference[frame]
        target.qvel[:] = 0.0
        mujoco.mj_forward(model, target)
        wrist_error = float(np.linalg.norm(data.site_xpos[wrist_site] - target.site_xpos[wrist_site]))
        fingertip_error = float(np.linalg.norm(data.site_xpos[tip_site] - target.site_xpos[tip_site]))
        object_pos, object_rot = _object_tracking_error(data, reference[frame], bodies)
        margin, _ = v2r._joint_margin(model, np.asarray([data.qpos], dtype=np.float64))
        warning_count = len(warnings)
        observation = ContactObservation(
            source_frame=int(source_frames[frame]),
            source_timestamp_s=float(source_frames[frame] / 120.0),
            sim_step=sim_step,
            substep=substep,
            role_active=bool(expected[frame, 6]),
            assigned_patch="patch:s5__cylindermedium_lift:0",
            assigned_robot_region=robot_region,
            physical_contact_present=physical,
            correct_geom_pair=physical,
            geom_pair=str(pair["geom_pair"] if pair else "NONE"),
            patch_distance_m=distance,
            patch_membership=distance <= 0.020,
            normal_cosine=cosine,
            tangential_slip_m=slip,
            normal_gap_m=max(0.0, distance if not physical else float(pair["penetration_m"])),
            penetration_m=float(depth),
            force_n=float(force),
            force_impulse_ns=float(force_impulse),
            joint_margin_fraction=float(np.nanmin(margin)),
            wrist_tracking_error_m=wrist_error,
            fingertip_tracking_error_m=fingertip_error,
            object_tracking_position_m=float(np.max(object_pos)),
            object_tracking_rotation_rad=float(np.max(object_rot)),
            finite=bool(_finite_data(data)),
            joint_limit_valid=not bool(v2r.dynamic._joint_limit_violations(model, np.asarray([data.qpos], dtype=np.float64))),
            warning_count=warning_count,
            reference_qpos=tuple(np.asarray(reference[frame], dtype=np.float64)),
            actual_qpos=tuple(np.asarray(data.qpos, dtype=np.float64)),
            ctrl=tuple(np.asarray(data.ctrl, dtype=np.float64)),
            regrasp_attempt=machine.regrasp_attempts,
        )
        previous_mode = machine.mode.value
        mode = machine.observe(observation)
        row = observation_payload(observation)
        row["mode"] = mode.value
        row["previous_mode"] = previous_mode
        row["closest_patch_point_world"] = closest.tolist()
        row["contact_normal_world"] = normal.tolist()
        timeline.append(row)
        return observation

    try:
        # Record the initial physical state, then integrate every frozen source interval.
        observe(0, 0, 0.0)
        count, depth, force, _pairs = dynamic._contact_summary(model, data, hand, objects)
        position, rotation = _object_tracking_error(data, reference[0], bodies)
        frame_rows.append({
            "source_frame": int(source_frames[0]),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "patch_distance_m": float(timeline[-1]["patch_distance_m"]),
            "patch_membership": bool(timeline[-1]["patch_membership"]),
            "normal_cosine": float(timeline[-1]["normal_cosine"]),
            "physical_contact": bool(timeline[-1]["correct_contact"]),
            "physical_contact_present": bool(timeline[-1]["physical_contact_present"]),
            "correct_geom_pair": bool(timeline[-1]["correct_geom_pair"]),
            "contact_depth_m": float(depth),
            "contact_force_n": float(force),
            "object_position_error_m": float(np.max(position)),
            "object_rotation_error_rad": float(np.max(rotation)),
            "mode": machine.mode.value,
        })
        for frame in range(1, 20):
            _set_object_mocap_reference(data, reference[frame], mocap)
            target_time += 1.0 / 120.0
            lead = int(profile["lead_source_frames"])
            control_index = min(frame + lead, 19)
            base = reference[control_index, :52] + feedforward[control_index]
            while data.time + 0.5 * model.opt.timestep < target_time:
                control = base.copy()
                if float(profile.get("state_feedback_gain", 0.0)):
                    limits = np.full(52, 0.08, dtype=np.float64)
                    limits[[0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31]] = 0.02
                    control += np.clip(float(profile["state_feedback_gain"]) * (reference[control_index, :52] - data.qpos[:52]), -limits, limits)
                if machine.mode in {ContactMode.ACQUIRE, ContactMode.RETAIN, ContactMode.REGRASP}:
                    gain = float(profile.get("contact_ik_gain", 0.0))
                    if machine.mode == ContactMode.REGRASP:
                        gain *= 1.25
                    control += _contact_correction(
                        model,
                        data,
                        tip_site,
                        anchors[frame, 6],
                        gain,
                        float(profile.get("contact_ik_damping", 0.002)),
                        0.12,
                        0.003,
                        0.03,
                    )
                data.ctrl[:52] = control
                data.ctrl[52:] = 0.0
                mujoco.mj_step(model, data)
                sim_step += 1
                observe(frame, int(sim_step), 0.0)
            count, depth, force, pairs = dynamic._contact_summary(model, data, hand, objects)
            position, rotation = _object_tracking_error(data, reference[frame], bodies)
            frame_rows.append({
                "source_frame": int(source_frames[frame]),
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "patch_distance_m": float(timeline[-1]["patch_distance_m"]),
                "patch_membership": bool(timeline[-1]["patch_membership"]),
                "normal_cosine": float(timeline[-1]["normal_cosine"]),
                "physical_contact": bool(timeline[-1]["correct_contact"]),
                "physical_contact_present": bool(timeline[-1]["physical_contact_present"]),
                "correct_geom_pair": bool(timeline[-1]["correct_geom_pair"]),
                "contact_depth_m": float(depth),
                "contact_force_n": float(force),
                "object_position_error_m": float(np.max(position)),
                "object_rotation_error_rad": float(np.max(rotation)),
                "mode": machine.mode.value,
            })
        # Reuse the last real observation for terminal state evaluation.
        last = timeline[-1]
        terminal = ContactObservation(
            source_frame=int(last["source_frame"]), source_timestamp_s=float(last["source_timestamp_s"]), sim_step=int(last["sim_step"]), substep=int(last["substep"]),
            role_active=bool(last["role_active"]), assigned_patch=str(last["assigned_patch"]), assigned_robot_region=str(last["assigned_robot_region"]),
            physical_contact_present=bool(last["physical_contact_present"]), correct_geom_pair=bool(last["correct_geom_pair"]), geom_pair=str(last["geom_pair"]),
            patch_distance_m=float(last["patch_distance_m"]), patch_membership=bool(last["patch_membership"]), normal_cosine=float(last["normal_cosine"]),
            tangential_slip_m=float(last["tangential_slip_m"]), normal_gap_m=float(last["normal_gap_m"]), penetration_m=float(last["penetration_m"]),
            force_n=float(last["force_n"]), force_impulse_ns=float(last["force_impulse_ns"]), joint_margin_fraction=float(last["joint_margin_fraction"]),
            wrist_tracking_error_m=float(last["wrist_tracking_error_m"]), fingertip_tracking_error_m=float(last["fingertip_tracking_error_m"]),
            object_tracking_position_m=float(last["object_tracking_position_m"]), object_tracking_rotation_rad=float(last["object_tracking_rotation_rad"]),
            finite=bool(last["finite"]), joint_limit_valid=bool(last["joint_limit_valid"]), warning_count=int(last["warning_count"]),
            regrasp_attempt=int(last["regrasp_attempt"]),
        )
        machine.finish(terminal)
    finally:
        mujoco.set_mju_user_warning(old_warning)

    frame_qpos = np.stack([row["qpos"] for row in frame_rows], axis=0) if frame_rows else reference[:1]
    frame_qvel = np.stack([row["qvel"] for row in frame_rows], axis=0) if frame_rows else reference_qvel[:1]
    frame_ctrl = np.stack([row["ctrl"] for row in frame_rows], axis=0) if frame_rows else np.zeros((1, model.nu))
    source_frame_values = np.asarray([row["source_frame"] for row in frame_rows], dtype=np.int64)
    if len(frame_qpos) != 20:
        raise RuntimeError(f"C-M2 frame summary has {len(frame_qpos)} rows; expected 20")
    distances = np.asarray([row["patch_distance_m"] for row in frame_rows], dtype=np.float64)
    contacts = np.asarray([row["correct_geom_pair"] and row["patch_membership"] for row in frame_rows], dtype=bool)
    normals = np.asarray([row["normal_cosine"] for row in frame_rows], dtype=np.float64)
    depths = np.asarray([row["contact_depth_m"] for row in frame_rows], dtype=np.float64)
    forces = np.asarray([row["contact_force_n"] for row in frame_rows], dtype=np.float64)
    finite_trace = bool(all(row["finite"] for row in timeline))
    warnings_payload = list(dict.fromkeys(warnings))
    ranges = v2r._robot_ranges(model)
    normalized_delta = float(np.max(np.abs(np.diff(frame_qpos[:, :52], axis=0)) / ranges)) if len(frame_qpos) > 1 else 0.0
    margin, _ = v2r._joint_margin(model, frame_qpos)
    tracking_positions, tracking_error, tracking_flat, tracking_by_side = dynamic._dynamic_robot_tracking(model, frame_qpos, reference, source_frames)
    object_errors = np.asarray([[row["object_position_error_m"], row["object_rotation_error_rad"]] for row in frame_rows], dtype=np.float64)
    visual = dynamic._dynamic_visual_penetration(model, frame_qpos, physics)
    metrics = {
        "schema_version": 1,
        "experiment": profile["experiment"],
        "profile": profile,
        "window": {"source_frames": WINDOW_SOURCE_FRAMES.tolist(), "local_frames": [0, 19], "fps": 120.0, "sim_dt_s": 0.0005},
        "contact": {
            "assigned_role": role_id,
            "assigned_robot_region": robot_region,
            "correct_physical_pair": bool(np.any([row["correct_geom_pair"] for row in frame_rows])),
            "patch_coverage": float(np.mean(contacts)) if len(contacts) else 0.0,
            "functional_role_recall": float(np.mean(contacts)) if len(contacts) else 0.0,
            "patch_distance_p95": float(np.percentile(distances, 95)) if len(distances) else float("inf"),
            "patch_distance_max": float(np.max(distances, initial=0.0)),
            "normal_cosine_median": float(np.median(normals[np.isfinite(normals)])) if np.isfinite(normals).any() else float("nan"),
            "physical_contact_frames": int(np.count_nonzero([row["physical_contact_present"] for row in frame_rows])),
            "correct_contact_frames": int(np.count_nonzero(contacts)),
            "first_loss_source_frame": int(source_frames[np.flatnonzero(~contacts)[0]]) if np.any(~contacts) else None,
            "terminal_correct_contact": bool(contacts[-1]) if len(contacts) else False,
        },
        "safety": {
            "warnings": warnings_payload,
            "penetration_max_mujoco": float(np.max(depths, initial=0.0)),
            "penetration_max_visual": float(max(visual["per_frame_max_penetration_m"], default=0.0)),
            "force_max_n": float(np.max(forces, initial=0.0)),
            "force_p95_n": float(np.percentile(forces, 95)) if len(forces) else 0.0,
            "minimum_joint_margin_fraction": float(np.nanmin(margin)),
            "normalized_one_frame_delta": normalized_delta,
        },
        "tracking": tracking_by_side,
        "object": {
            "position_max_m": float(np.max(object_errors[:, 0], initial=0.0)),
            "rotation_max_rad": float(np.max(object_errors[:, 1], initial=0.0)),
            "object_qpos_written": False,
            "source_object_target_unchanged": True,
        },
        "state_machine": {"terminal_mode": machine.mode.value, "failure_code": machine.failure_code.value if machine.failure_code else None, "regrasp_attempts": machine.regrasp_attempts, "transitions": machine.transition_payload()},
        "gates": {
            "finite": finite_trace and bool(np.isfinite(frame_qpos).all()) and bool(np.isfinite(frame_qvel).all()),
            "no_warnings": not warnings_payload,
            "joint_limits": not bool(v2r.dynamic._joint_limit_violations(model, frame_qpos)),
            "penetration": float(np.max(depths, initial=0.0)) <= 0.003 and float(max(visual["per_frame_max_penetration_m"], default=0.0)) <= 0.003,
            "smoothness": normalized_delta <= 0.25,
            "tracking": all(side["wrist_rmse_m"] <= 0.03 and all(item["rmse_m"] <= 0.08 for item in side["fingertips"].values()) for side in tracking_by_side.values()),
            "object_tracking": float(np.max(object_errors[:, 0], initial=0.0)) <= 0.08 and float(np.max(object_errors[:, 1], initial=0.0)) <= 0.50,
            "force": float(np.max(forces, initial=0.0)) <= 150.0,
            "contact_contract": bool(np.mean(contacts) >= 0.70 and np.mean(contacts) >= 0.80 and np.percentile(distances, 95) <= 0.020 and np.median(normals[np.isfinite(normals)]) >= 0.50) if len(distances) and np.isfinite(normals).any() else False,
            "terminal": bool(contacts[-1]) if len(contacts) else False,
        },
        "preservation": {"object_qpos_written": False, "source_frames_unchanged": True, "source_object_target_unchanged": True, "v2_thresholds_unchanged": True, "patch_unchanged": True, "role_unchanged": True},
    }
    metrics["status"] = "PASS" if all(metrics["gates"].values()) else "FAIL"
    metrics["failure_category"] = None if metrics["status"] == "PASS" else _failure_category(metrics)
    profile_root = output / "profiles" / str(profile["profile_id"])
    _write_json(profile_root / "candidate.json", metrics)
    _write_json(profile_root / "contact_mode_timeline.json", {"schema_version": 1, "rows": timeline, "transitions": machine.transition_payload()})
    _write_npz(profile_root / "contact_mode_trace.npz", qpos=frame_qpos, qvel=frame_qvel, ctrl=frame_ctrl, source_frame_indices=source_frame_values, patch_distance_m=distances, patch_membership=contacts, normal_cosine=normals, contact_depth_m=depths, contact_force_n=forces, expected=expected, reference_qpos=reference)
    return metrics


def run(paths_config: str = "configs/local/paths.yaml", output_dir: str | None = None) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    window = _load_window(paths)
    output = Path(output_dir) if output_dir else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-contact-mode")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "input_preservation.json", {"schema_version": 1, "input_hashes": window["input_hashes"], "window": WINDOW_SOURCE_FRAMES.tolist(), "historical_diagnostic_root": str(DIAGNOSTIC_ROOT), "object_qpos_written": False})
    baseline = _historical_baseline(window)
    _write_json(output / "historical_baseline_summary.json", baseline)
    specs = _profile_specs()
    if len(specs) != 12:
        raise RuntimeError("C-M2 profile bound is exactly twelve dynamic profiles")
    results = [_run_profile(window, output, spec) for spec in specs]
    passing = [row for row in results if row["status"] == "PASS"]
    ordered = sorted(results, key=lambda row: (-row["contact"]["correct_contact_frames"], row["contact"]["patch_distance_p95"], row["safety"]["penetration_max_mujoco"], row["safety"]["force_p95_n"], row["state_machine"]["regrasp_attempts"], row["profile"]["profile_id"]))
    best = passing[0] if passing else ordered[0]
    best_root = output / "profiles" / str(best["profile"]["profile_id"])
    shutil.copy2(best_root / "candidate.json", output / "selected_contact_mode_profile.json")
    shutil.copy2(best_root / "contact_mode_timeline.json", output / "contact_mode_timeline.json")
    shutil.copy2(best_root / "contact_mode_trace.npz", output / "contact_mode_trace.npz")
    status = "PASS" if passing else "EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS"
    profile_hash = _payload_hash({"profiles": [row["profile"] for row in results], "selected": best["profile"]})
    summary = {
        "schema_version": 1,
        "stage": "C-M1/C-M2",
        "status": status,
        "c_m1": "PASS",
        "c_m2": status,
        "dynamic_witness": "FOUND" if passing else "NOT_FOUND",
        "bounded_empirical_claim_only": not bool(passing),
        "window": {"source_frames": WINDOW_SOURCE_FRAMES.tolist(), "local_frames": [0, 19]},
        "e0": baseline,
        "experiments": {"E1": [row for row in results if row["experiment"] == "E1"], "E2": [row for row in results if row["experiment"] == "E2"], "E3": [row for row in results if row["experiment"] == "E3"]},
        "selected": best,
        "candidate_count": len(results),
        "candidate_pass_count": len(passing),
        "CONTACT_MODE_PROFILE_HASH": profile_hash,
        "limitations": ["Only source frames 1461..1480 were simulated", "Full primary, Oracle C full sequence, D2, MJWP, smokes, and Stage D were not run", "No user visual review has been performed"],
    }
    _write_json(output / "contact_mode_transition_summary.json", summary)
    if passing:
        _write_npz(output / "contact_mode_dynamic_witness.npz", **{key: value for key, value in np.load(best_root / "contact_mode_trace.npz", allow_pickle=False).items()})
        witness = {"status": "PASS", "profile_id": best["profile"]["profile_id"], "profile_hash": profile_hash, "metrics": best, "witness_hash": _sha256(output / "contact_mode_dynamic_witness.npz")}
        _write_json(output / "contact_mode_dynamic_witness.json", witness)
    return {"output_dir": str(output), "status": status, "selected": best["profile"]["profile_id"], "profile_hash": profile_hash, "candidate_count": len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.output_dir), indent=2))
