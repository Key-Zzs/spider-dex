# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A standalone script to run DIAL MPC with Mujoco + Warp

Up to now, domain randomization is not supported. Will add it later.

Author: Chaoyi Pan
Date: 2025-08-11
"""

from __future__ import annotations

import sys
import time
from dataclasses import fields
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    _initial_state_sanity_check,
    compute_contact_point_delta,
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    set_object_mocap_reference,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
)
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}


def _parse_override_tokens(tokens: list[str]) -> dict:
    allowed = {field.name for field in fields(Config)}
    override_dict: dict = {}
    for item in tokens:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        if key not in allowed:
            continue
        parsed = OmegaConf.to_container(
            OmegaConf.from_dotlist([f"{key}={value}"]), resolve=True
        )
        if isinstance(parsed, dict) and key in parsed:
            override_dict[key] = parsed[key]
    return override_dict


def _extract_cli_overrides(cfg: DictConfig) -> dict:
    """Extract CLI overrides so they can be applied on top of a loaded config."""
    overrides = OmegaConf.select(cfg, "hydra.overrides.task") or []
    override_dict = _parse_override_tokens(overrides)
    if override_dict:
        return override_dict
    return _parse_override_tokens(sys.argv[1:])


def _assert_object_actuator_gains_zero(
    env, config: Config, stage: str, atol: float = 1e-4
) -> None:
    if config.allow_object_actuator_guidance or not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol), (
        f"Object actuator Kp not near zero at {stage}: max={np.max(np.abs(kp))}"
    )
    assert np.allclose(kd, 0.0, atol=atol), (
        f"Object actuator Kd not near zero at {stage}: max={np.max(np.abs(kd))}"
    )


def _robot_reference_controls(
    ctrl_ref: torch.Tensor,
    start: int,
    length: int,
    lookahead_steps: int,
    wrist_lookahead_steps: int | None = None,
    finger_lookahead_steps: int | None = None,
) -> torch.Tensor:
    """Return a shared optional phase lead for robot actuators only."""
    if lookahead_steps < 0:
        raise ValueError("robot reference lookahead must be non-negative")
    wrist_lead = lookahead_steps if wrist_lookahead_steps is None else wrist_lookahead_steps
    finger_lead = lookahead_steps if finger_lookahead_steps is None else finger_lookahead_steps
    if wrist_lead < 0 or finger_lead < 0:
        raise ValueError("robot group lookaheads must be non-negative")
    indices = torch.arange(start, start + length, device=ctrl_ref.device).clamp(
        max=ctrl_ref.shape[0] - 1
    )
    controls = ctrl_ref[indices].clone()
    for lead, actuator_ids in (
        (wrist_lead, (slice(0, 6), slice(26, 32))),
        (finger_lead, (slice(6, 26), slice(32, 52))),
    ):
        if lead <= 0:
            continue
        future = ctrl_ref[(indices + lead).clamp(max=ctrl_ref.shape[0] - 1)]
        for actuator_id in actuator_ids:
            controls[:, actuator_id] = future[:, actuator_id]
    return controls


def _bounded_robot_state_feedback(
    reference_qpos: torch.Tensor,
    physical_qpos: torch.Tensor,
    gain: float,
    wrist_translation_clip_m: float,
    wrist_rotation_clip_rad: float,
    finger_clip_rad: float,
) -> torch.Tensor:
    """Return a bounded robot-only position-target correction.

    The correction is deliberately computed from the *current* source frame,
    rather than a future controller reference.  This gives the position
    servos a bounded way to reject accumulated physical-state error without
    advancing the object or silently shifting the source phase.  The trailing
    object coordinates are always exactly zero in the returned tensor.
    """
    if reference_qpos.ndim != 1 or physical_qpos.ndim != 1:
        raise ValueError("robot state feedback expects one-dimensional qpos tensors")
    if reference_qpos.shape != physical_qpos.shape or reference_qpos.numel() < 52:
        raise ValueError("robot state feedback qpos shape mismatch")
    if gain < 0.0:
        raise ValueError("robot state feedback gain must be non-negative")
    clips = (wrist_translation_clip_m, wrist_rotation_clip_rad, finger_clip_rad)
    if any(value < 0.0 for value in clips):
        raise ValueError("robot state feedback clips must be non-negative")
    correction = torch.zeros_like(reference_qpos)
    if gain == 0.0 or not any(value > 0.0 for value in clips):
        return correction
    error = (reference_qpos[:52] - physical_qpos[:52]) * float(gain)
    for target, slices, clip in (
        (correction, (slice(0, 3), slice(26, 29)), wrist_translation_clip_m),
        (correction, (slice(3, 6), slice(29, 32)), wrist_rotation_clip_rad),
        (correction, (slice(6, 26), slice(32, 52)), finger_clip_rad),
    ):
        if clip <= 0.0:
            continue
        for coordinates in slices:
            target[coordinates] = torch.clip(error[coordinates], -float(clip), float(clip))
    return correction


def _bounded_damped_least_squares(
    jacobian: np.ndarray,
    position_error: np.ndarray,
    damping: float,
    component_clips: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Return a bounded DLS correction in joint/actuator coordinates.

    The helper is deliberately NumPy-only so its sign convention and bounds
    are unit-testable without a live Warp scene.  ``position_error`` is the
    desired displacement (anchor minus current site position).
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    position_error = np.asarray(position_error, dtype=np.float64).reshape(-1)
    component_clips = np.asarray(component_clips, dtype=np.float64).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[0] != position_error.size:
        raise ValueError("DLS Jacobian/error shape mismatch")
    if jacobian.shape[1] != component_clips.size:
        raise ValueError("DLS Jacobian/clip shape mismatch")
    if damping < 0.0 or gain < 0.0:
        raise ValueError("DLS damping and gain must be non-negative")
    lhs = jacobian.T @ jacobian + float(damping) * np.eye(jacobian.shape[1])
    correction = np.linalg.solve(lhs, jacobian.T @ position_error) * float(gain)
    return np.clip(correction, -component_clips, component_clips)


def _contact_ik_feedback_delta(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_mask_step: torch.Tensor,
    contact_pos_ref_step: torch.Tensor,
    hand_contact_site_ids: list[int | None],
    contact_indices: list[int],
    actuator_ids: list[int],
    config: Config,
) -> np.ndarray | None:
    """Compute one hand's bounded physical contact-servo target correction.

    This reads the current dynamic robot state through MuJoCo Jacobians.  The
    returned delta is applied only to robot position actuators by the caller;
    it never changes an object body, mocap target, or qpos array.
    """
    if config.contact_ik_feedback_gain <= 0.0:
        return None
    if len(actuator_ids) != 26:
        raise ValueError(f"Expected 26 hand actuator ids, got {actuator_ids}")
    max_anchor_error = float(config.contact_ik_feedback_max_anchor_error_m)
    if max_anchor_error <= 0.0:
        raise ValueError("contact IK max anchor error must be positive")
    active: list[tuple[int, int]] = []
    for index in contact_indices:
        if index >= len(hand_contact_site_ids) or index >= contact_pos_ref_step.shape[0]:
            continue
        site_id = hand_contact_site_ids[index]
        if site_id is not None and float(contact_mask_step[index]) > 0.5:
            error = (
                contact_pos_ref_step[index].detach().cpu().numpy()
                - np.asarray(data.site_xpos[site_id], dtype=np.float64)
            )
            if np.linalg.norm(error) <= max_anchor_error:
                active.append((index, int(site_id)))
    if not active:
        return None
    if config.contact_ik_feedback_strategy == "first_active":
        active = active[:1]
    elif config.contact_ik_feedback_strategy != "all_active":
        raise ValueError(
            "contact_ik_feedback_strategy must be 'all_active' or 'first_active', "
            f"got {config.contact_ik_feedback_strategy!r}"
        )
    jacobians: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for index, site_id in active:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        jacobians.append(jacp[:, actuator_ids])
        errors.append(
            contact_pos_ref_step[index].detach().cpu().numpy()
            - np.asarray(data.site_xpos[site_id], dtype=np.float64)
        )
    clips = np.concatenate(
        [
            np.full(3, config.contact_ik_feedback_wrist_translation_clip_m),
            np.full(3, config.contact_ik_feedback_wrist_rotation_clip_rad),
            np.full(20, config.contact_ik_feedback_finger_clip_rad),
        ]
    )
    if np.any(clips < 0.0):
        raise ValueError("contact IK feedback clips must be non-negative")
    return _bounded_damped_least_squares(
        np.concatenate(jacobians, axis=0),
        np.concatenate(errors, axis=0),
        config.contact_ik_feedback_damping,
        clips,
        config.contact_ik_feedback_gain,
    )


def _project_contact_delta_against_collision_barriers(
    correction: np.ndarray,
    barrier_rows: np.ndarray,
    required_outward_displacement: np.ndarray,
    component_clips: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Project a robot-target delta into contact-safe local half-spaces.

    Each row is an outward hand-contact point Jacobian.  The returned delta
    therefore cannot have a smaller outward displacement than the requested
    bound for an already-observed hand/object collision.  The operation is a
    bounded controller projection, not a state rewrite; callers still apply
    the result only through the 26 robot position actuators.
    """
    delta = np.asarray(correction, dtype=np.float64).copy()
    rows = np.asarray(barrier_rows, dtype=np.float64)
    required = np.asarray(required_outward_displacement, dtype=np.float64).reshape(-1)
    clips = np.asarray(component_clips, dtype=np.float64).reshape(-1)
    if rows.size == 0:
        return np.clip(delta, -clips, clips)
    if rows.ndim != 2 or rows.shape[1] != delta.size or rows.shape[0] != required.size:
        raise ValueError("collision barrier shape mismatch")
    if clips.size != delta.size or damping < 0.0 or np.any(required < 0.0):
        raise ValueError("invalid collision barrier bounds")
    # A small deterministic sequential projection is adequate here: only the
    # current hand/object contacts are active and each correction is local to
    # one 10-ms control tick.  Repeating once after component clipping avoids
    # silently reintroducing an inward component on coupled finger rows.
    for _ in range(2):
        for row, lower_bound in zip(rows, required, strict=True):
            residual = float(lower_bound - row @ delta)
            norm_sq = float(row @ row)
            if residual > 0.0 and norm_sq > 1e-12:
                delta += residual * row / (norm_sq + damping)
                delta = np.clip(delta, -clips, clips)
    return delta


def _contact_collision_barriers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    side: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward Jacobian rows for real current hand/object contacts."""
    if config.contact_ik_collision_barrier_gain <= 0.0:
        return np.empty((0, len(actuator_ids))), np.empty(0)
    if config.contact_ik_collision_barrier_margin_m < 0.0:
        raise ValueError("contact collision barrier margin must be non-negative")
    if config.contact_ik_collision_barrier_max_contacts < 1:
        raise ValueError("contact collision barrier max contacts must be positive")
    hand_prefix = f"collision_hand_{side}_"
    candidates: list[tuple[float, np.ndarray, float]] = []
    for contact_index in range(data.ncon):
        item = data.contact[contact_index]
        geom1, geom2 = int(item.geom1), int(item.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        hand_geom = geom1 if name1.startswith(hand_prefix) else geom2 if name2.startswith(hand_prefix) else None
        object_name = name2 if hand_geom == geom1 else name1 if hand_geom == geom2 else ""
        if hand_geom is None or "_object_" not in object_name:
            continue
        depth = max(0.0, -float(item.dist))
        # MuJoCo's contact-frame x-axis points geom1 -> geom2.  Convert it
        # to the hand's outward direction before taking the robot Jacobian.
        outward = -np.asarray(item.frame[:3], dtype=np.float64) if hand_geom == geom1 else np.asarray(item.frame[:3], dtype=np.float64)
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, None, item.pos, int(model.geom_bodyid[hand_geom]))
        row = outward @ jacp[:, actuator_ids]
        required = float(config.contact_ik_collision_barrier_gain) * max(
            0.0, depth - float(config.contact_ik_collision_barrier_margin_m)
        )
        candidates.append((depth, row, required))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    candidates = candidates[: int(config.contact_ik_collision_barrier_max_contacts)]
    if not candidates:
        return np.empty((0, len(actuator_ids))), np.empty(0)
    return np.stack([entry[1] for entry in candidates]), np.asarray([entry[2] for entry in candidates])


def _contact_feedback_component_clips(config: Config) -> np.ndarray:
    """Return the fixed 6-wrist + 20-finger controller bounds for one hand."""
    clips = np.concatenate(
        [
            np.full(3, config.contact_ik_feedback_wrist_translation_clip_m),
            np.full(3, config.contact_ik_feedback_wrist_rotation_clip_rad),
            np.full(20, config.contact_ik_feedback_finger_clip_rad),
        ]
    )
    if np.any(clips < 0.0):
        raise ValueError("contact IK feedback clips must be non-negative")
    return clips


def _update_contact_integral(
    previous: np.ndarray,
    correction: np.ndarray | None,
    gain: float,
    decay: float,
    component_clips: np.ndarray,
) -> np.ndarray:
    """Update a bounded contact-only integral target correction."""
    if gain < 0.0 or not 0.0 <= decay <= 1.0:
        raise ValueError("contact integral gain must be non-negative and decay in [0, 1]")
    updated = np.asarray(previous, dtype=np.float64) * decay
    if correction is not None:
        updated += gain * np.asarray(correction, dtype=np.float64)
    clips = np.asarray(component_clips, dtype=np.float64)
    if updated.shape != clips.shape:
        raise ValueError("contact integral/clip shape mismatch")
    return np.clip(updated, -clips, clips)


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = (
        Path(config.output_dir)
        / f"config{'_act' if config.contact_guidance else ''}.yaml"
    )
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = (
            int(config.object_action_dims) if config.object_action_dims > 0 else 12
        )
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    right_ids = list(range(0, half))
    left_ids = list(range(half, robot_nu))
    return right_ids, left_ids


def _apply_noise_mask(
    base_noise_scale: torch.Tensor, zero_indices: list[int]
) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(
            zero_indices, device=base_noise_scale.device, dtype=torch.long
        )
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def main(config: Config):
    """Run the SPIDER using MuJoCo Warp backend"""
    # process config, set defaults and derived fields
    config = process_config(config)
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning(
            "contact_guidance requires improvement_threshold <= 0; overriding to 0.0."
        )
        config.improvement_threshold = 0.0

    # load reference data (already interpolated and extended)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    if (
        config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        loguru.logger.info(
            "Using qpos as ctrl reference for contact guidance (ctrl dims: {} -> {}).",
            ctrl_ref.shape[1],
            config.nu,
        )
        ctrl_ref = qpos_ref[:, : config.nu]
    if (
        config.contact_guidance
        and torch.all(contact <= 0)
        and not config.allow_empty_contact_guidance
    ):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    # hack: start from step 500
    # qpos_ref = qpos_ref[500:]
    # qvel_ref = qvel_ref[500:]
    # ctrl_ref = ctrl_ref[500:]
    # contact = contact[500:]
    # contact_pos = contact_pos[500:]
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    # setup env with initial state from first sim qpos
    env = setup_env(config, ref_data)

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    _assert_object_actuator_gains_zero(env, config, "start")

    # Initial-state stability sanity check (CPU MuJoCo only). Disabled when
    # config.sanity_check_seconds <= 0.0.
    if config.sanity_check_seconds > 0.0:
        _initial_state_sanity_check(
            mj_model,
            mj_data,
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            config,
            save_video_dir=config.output_dir,
        )
        # Restore initial scene state after the check.
        mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
        mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
        mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
        mujoco.mj_step(mj_model, mj_data)
        mj_data.time = 0.0

    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    contact_guidance_enabled = (
        config.contact_guidance and len(config.object_actuator_ids) > 0
    )
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning(
            "contact_guidance is enabled but no object actuators were resolved."
        )
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        if (
            config.contact_len != len(config.contact_order)
            or config.contact_len != contact.shape[1]
        ):
            loguru.logger.warning(
                "Contact length mismatch (mask={}, pos={}, expected={}); truncating to {}.",
                contact.shape[1],
                contact_pos.shape[1],
                len(config.contact_order),
                config.contact_len,
            )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[
            : config.contact_len
        ]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    # setup env params
    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0 and not config.allow_object_actuator_guidance:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array(
            [
                (
                    config.init_rot_actuator_gain
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_gain
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        base_kd = np.array(
            [
                (
                    config.init_rot_actuator_bias
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_bias
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                kp_i = np.zeros_like(base_kp, dtype=np.float32)
                kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {
                "xy_offset": xy_offset_list[j],
                "pair_margin": pair_margin_list[j],
            }
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    # setup optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)
    base_noise_scale = config.noise_scale.clone()
    gibbs_enabled = config.gibbs_sampling and config.embodiment_type == "bimanual"
    if config.gibbs_sampling and not gibbs_enabled:
        loguru.logger.warning(
            "gibbs_sampling is enabled but embodiment_type is {}, disabling.",
            config.embodiment_type,
        )
    if gibbs_enabled:
        right_ids, left_ids = _get_bimanual_hand_indices(config)
        right_only_zero = left_ids
        left_only_zero = right_ids

    # initial controls
    ctrls = _robot_reference_controls(
        ctrl_ref, 0, config.horizon_steps, config.robot_reference_lookahead_steps,
        config.robot_wrist_reference_lookahead_steps,
        config.robot_finger_reference_lookahead_steps,
    )
    # buffers for saving info and trajectory
    info_list = []
    contact_integral = {
        "right": np.zeros(26, dtype=np.float64),
        "left": np.zeros(26, dtype=np.float64),
    }
    contact_integral_clips = _contact_feedback_component_clips(config)

    # run viewer + control loop
    t_start = time.perf_counter()
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # optimize using future reference window at control-rate (+1 lookahead)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            # Stage-C scenes can supply object references through explicit
            # mocap-weld constraints.  Seed the target at the current physics
            # time for optimization.  The actual stepping loop below then
            # advances it once per simulation substep.  Holding a target for
            # a complete control tick makes the last ``ctrl_steps - 1``
            # substeps chase a stale source pose when ``ctrl_dt > sim_dt``.
            # This updates only the kinematic mocap target; object qpos is
            # never overwritten here.
            env = set_object_mocap_reference(config, env, qpos_ref[sim_step])
            # This low-level state feedback is shared and bounded.  It reads
            # the current Warp state but corrects only robot servo targets;
            # the object reference continues through the mocap-weld path.
            robot_state_feedback = _bounded_robot_state_feedback(
                qpos_ref[sim_step],
                get_qpos(config, env)[0],
                config.robot_state_feedback_gain,
                config.robot_state_feedback_wrist_translation_clip_m,
                config.robot_state_feedback_wrist_rotation_clip_rad,
                config.robot_state_feedback_finger_clip_rad,
            )
            # Keep the CPU MuJoCo data synchronized for the optional bounded
            # contact feedback below.  It is used for hand-site Jacobians only
            # and is never copied back into the Warp physical state.
            contact_feedback: list[tuple[list[int], np.ndarray]] = []
            if (
                contact_guidance_enabled
                and config.contact_ik_feedback_gain > 0.0
                and config.contact_len > 0
            ):
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mujoco.mj_forward(mj_model, mj_data)
                contact_mask_step = contact[sim_step][
                    contact_offset : contact_offset + config.contact_len
                ]
                contact_pos_ref_step = contact_pos[sim_step]
                for side, contact_indices, actuator_ids in (
                    ("right", config.right_contact_indices, list(range(26))),
                    ("left", config.left_contact_indices, list(range(26, 52))),
                ):
                    correction = _contact_ik_feedback_delta(
                        mj_model,
                        mj_data,
                        contact_mask_step,
                        contact_pos_ref_step,
                        config.hand_contact_site_ids,
                        contact_indices,
                        actuator_ids,
                        config,
                    )
                    if correction is not None:
                        rows, required = _contact_collision_barriers(
                            mj_model, mj_data, actuator_ids, side, config
                        )
                        correction = _project_contact_delta_against_collision_barriers(
                            correction,
                            rows,
                            required,
                            contact_integral_clips,
                            config.contact_ik_feedback_damping,
                        )
                    if config.contact_ik_integral_gain > 0.0:
                        contact_integral[side] = _update_contact_integral(
                            contact_integral[side],
                            correction,
                            config.contact_ik_integral_gain,
                            config.contact_ik_integral_decay,
                            contact_integral_clips,
                        )
                        if correction is not None:
                            contact_feedback.append((actuator_ids, contact_integral[side]))
                    elif correction is not None:
                        contact_feedback.append((actuator_ids, correction))
            ctrls_for_opt = ctrls
            if contact_feedback and config.contact_ik_feedback_apply_to_plan:
                ctrls_for_opt = ctrls.clone()
                for actuator_ids, correction in contact_feedback:
                    ctrls_for_opt[: config.ctrl_steps, actuator_ids] += torch.as_tensor(
                        correction,
                        device=ctrls_for_opt.device,
                        dtype=ctrls_for_opt.dtype,
                    )
            if contact_guidance_enabled and config.contact_len > 0:
                contact_mask_step = contact[sim_step][
                    contact_offset : contact_offset + config.contact_len
                ]
                contact_pos_ref_step = contact_pos[sim_step]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]
                ref_ctrl_slice = _robot_reference_controls(
                    ctrl_ref,
                    sim_step,
                    ctrls.shape[0],
                    config.robot_reference_lookahead_steps,
                    config.robot_wrist_reference_lookahead_steps,
                    config.robot_finger_reference_lookahead_steps,
                )

                right_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.right_contact_indices,
                    config.contact_wrist_mean_feedback_strategy,
                )
                left_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.left_contact_indices,
                    config.contact_wrist_mean_feedback_strategy,
                )
                if (
                    right_delta is not None
                    and config.right_wrist_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    ctrls_for_opt = ctrls_for_opt.clone()
                    ctrls_for_opt[:, config.right_wrist_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.right_wrist_pos_ctrl_ids
                    ] - config.contact_wrist_mean_feedback_gain * torch.clip(
                        right_delta, -0.01, 0.01
                    )
                if (
                    left_delta is not None
                    and config.left_wrist_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                    ctrls_for_opt[:, config.left_wrist_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.left_wrist_pos_ctrl_ids
                    ] - config.contact_wrist_mean_feedback_gain * torch.clip(
                        left_delta, -0.01, 0.01
                    )
            if gibbs_enabled:
                config.noise_scale = _apply_noise_mask(
                    base_noise_scale, right_only_zero
                )
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                config.noise_scale = _apply_noise_mask(base_noise_scale, left_only_zero)
                ctrls, infos = optimize(config, env, ctrls, ref_slice)
                config.noise_scale = base_noise_scale
            else:
                config.noise_scale = base_noise_scale
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)

            # Compute trace_ref from reference qpos over the horizon
            if len(config.trace_site_ids) > 0:
                trace_ref = []
                qpos_ref_horizon = ref_slice[0]
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    site_xpos = np.array(
                        [mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]
                    )
                    trace_ref.append(site_xpos)
                # (H, K, 3) -> (1, 1, H, K, 3) to match trace_sample shape
                trace_ref_np = np.stack(trace_ref, axis=0)[None, None, :, :, :]
                infos["trace_ref"] = trace_ref_np

            # step environment for ctrl_steps
            # Preserve compact controller telemetry alongside the physical
            # rollout.  It is diagnostic-only: recording these scalars does
            # not alter the optimizer, the commanded controls, or any object
            # state.  In particular it lets Stage-C distinguish an ineffective
            # local contact controller from a controller that is active but
            # faces a genuine contact/collision trade-off.
            contact_feedback_l2 = float(
                sum(np.linalg.norm(correction) for _actuator_ids, correction in contact_feedback)
            )
            step_info = {
                "qpos": [], "qvel": [], "time": [], "ctrl": [],
                "contact_feedback_l2": [], "contact_feedback_active_hands": [],
            }
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]
                if contact_feedback:
                    # The feedback must also reach the executed physical
                    # target.  ``contact_ik_feedback_apply_to_plan`` controls
                    # whether it additionally seeds/scopes the optimizer
                    # above; the sampling optimizer returns a fresh control
                    # sequence and therefore does not retain that additive
                    # correction by itself.  This remains a bounded
                    # low-level robot-only servo: it never alters qpos or an
                    # object actuator.
                    ctrl_step = ctrl_step.clone()
                    for actuator_ids, correction in contact_feedback:
                        ctrl_step[actuator_ids] += torch.as_tensor(
                            correction, device=ctrl_step.device, dtype=ctrl_step.dtype
                        )
                if torch.count_nonzero(robot_state_feedback).item() > 0:
                    ctrl_step = ctrl_step.clone()
                    ctrl_step[:52] += robot_state_feedback[:52].to(
                        device=ctrl_step.device, dtype=ctrl_step.dtype
                    )

                # Keep the physical weld target time-aligned with this exact
                # simulation substep.  ``step_env`` integrates one ``sim_dt``
                # after this point, hence the +1 reference sample.
                target_index = min(sim_step + i + 1, qpos_ref.shape[0] - 1)
                env = set_object_mocap_reference(
                    config, env, qpos_ref[target_index]
                )

                # option 1: use mujoco step
                # mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                # mujoco.mj_step(mj_model, mj_data)
                # option 2: use warp step
                step_env(config, env, ctrl_step)
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        image = render_image(
                            config, renderer, mj_model, mj_data, mj_data_ref
                        )
                        images.append(image)
                if "rerun" in config.viewer or "viser" in config.viewer:
                    mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                    )
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
                step_info["contact_feedback_l2"].append(contact_feedback_l2)
                step_info["contact_feedback_active_hands"].append(len(contact_feedback))
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            # sync env state
            sync_env(config, env, mj_data)

            # receding horizon update
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = _robot_reference_controls(
                ctrl_ref,
                sim_step + prev_ctrl.shape[0],
                config.ctrl_steps,
                config.robot_reference_lookahead_steps,
                config.robot_wrist_reference_lookahead_steps,
                config.robot_finger_reference_lookahead_steps,
            )
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # sync viewer state and render
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            # progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                end="\r",
            )

            # record info/trajectory at control tick
            # rule out "trace"
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"Total time: {t_end - t_start:.4f}s")

    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
        )

    # save video
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, data_type
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )

    _assert_object_actuator_gains_zero(env, config, "end")

    if "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info(
            "Optimization complete! Keeping Viser server alive. Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_main(cfg: DictConfig) -> None:
    """Entry point for Hydra configuration runner."""
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Optionally load a saved config YAML and merge; CLI overrides take priority.
    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")  # Let the default factory handle it

    # Convert lists to tuples where needed
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
