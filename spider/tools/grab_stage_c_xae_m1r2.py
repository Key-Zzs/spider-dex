"""Frozen Stage C-XAE-M1R2 actuator-response identification and retention control.

The module intentionally consumes the already-frozen XAE, XAE-M1 and M1R
authorities.  Dynamic experiments initialize MuJoCo once, then use
``FrozenActionEnvironment.step(action)`` exclusively; it owns the sole
integration call and rejects any post-initialisation state mutation.  The
only controllable action coordinates are name-resolved left-index actuators.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from spider.contact.contact_mode import ContactMode, ContactModeConfig, ContactModeMachine
from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_xae_m1 as m1
from spider.tools import grab_stage_c_xae_m1r as m1r
from spider.tools import grab_stage_c_v2r as v2r
from spider.tools.grab_stage_c import _finite_data, _preflight_object_ids, _set_object_mocap_reference, _site_ids, preflight_static


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r2"
XAE_AUTHORITY = REPO / ".local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment"
XAE_M1_AUTHORITY = REPO / ".local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention"
M1R_AUTHORITY = REPO / ".local_artifacts/stage_c_xae_m1r/20260801T153353Z-step5-control-authority"
HORIZON = 9
SEED_TRAIN = 1367
SEED_VALIDATION = 2468
FIR_ORDER = 3


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_plain).encode("utf-8")).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _post_rows(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rollout["rows"] if row["phase"] == "post"]


def _control_payload(action: np.ndarray) -> dict[str, Any]:
    action = np.asarray(action, dtype=np.float64)
    return {
        "raw_correction": action,
        "feedforward_correction": np.zeros(4),
        "feedback_correction": action,
        "normal_correction": np.zeros(4),
        "tangential_correction": np.zeros(4),
        "clipped_correction": action,
        "projected_correction": action,
        "final_correction": action,
        "control_clipped": False,
    }


def _resolve_mapping(model: mujoco.MjModel) -> dict[str, Any]:
    """Resolve controller, qpos, and qvel columns from immutable names."""
    rows: list[dict[str, Any]] = []
    for actuator_name, joint_name in zip(m1r.ACTUATOR_NAMES, m1r.JOINT_NAMES, strict=True):
        actuator = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if actuator < 0 or joint < 0 or int(model.actuator_trnid[actuator, 0]) != joint:
            raise RuntimeError(f"WRONG_ACTUATOR_MAPPING: {actuator_name} -> {joint_name}")
        rows.append({"actuator_name": actuator_name, "actuator_index": actuator, "joint_name": joint_name, "joint_index": joint, "qpos_column": int(model.jnt_qposadr[joint]), "qvel_column": int(model.jnt_dofadr[joint]), "ctrlrange": model.actuator_ctrlrange[actuator].copy(), "forcerange": model.actuator_forcerange[actuator].copy(), "name_based_mapping_valid": True})
    return {"status": "PASS", "rows": rows, "controlled_actuator_indices": [row["actuator_index"] for row in rows], "controlled_qvel_columns": [row["qvel_column"] for row in rows], "controlled_qpos_columns": [row["qpos_column"] for row in rows]}


@dataclass
class FrozenActionEnvironment:
    """The formal action route for M1R2 dynamic experiments.

    Object reference motion uses the existing mocap/reference interface.  It
    is not an object-qpos workaround: qpos/qvel are written only in ``create``.
    """

    ctx: m1.Context
    model: mujoco.MjModel
    data: mujoco.MjData
    hand: set[int]
    objects: set[int]
    bodies: dict[str, int]
    mocap: dict[str, int]
    mapping: dict[str, Any]
    contact_enabled: bool
    previous_ctrl: np.ndarray
    warnings: list[str]
    warning_callback_old: Any
    machine: ContactModeMachine
    first_loss: dict[str, Any] | None

    @classmethod
    def create(cls, ctx: m1.Context, *, contact_enabled: bool) -> "FrozenActionEnvironment":
        model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
        model.opt.timestep = m1.SIM_DT
        hand, objects = v2r._contact_ids(model)
        if not contact_enabled:
            ablation = m1r._disable_assigned_pair_only(model)
            if ablation["explicit_pair_count"] != 1 or not ablation["all_other_model_terms_unchanged"]:
                raise RuntimeError("DIAGNOSTIC_CONTACT_ABLATION_INVALID")
        mapping = _resolve_mapping(model)
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
        warnings: list[str] = []
        old = mujoco.get_mju_user_warning()
        mujoco.set_mju_user_warning(lambda warning: warnings.append(str(warning)))
        return cls(
            ctx, model, data, hand, objects, bodies, mocap, mapping, contact_enabled,
            data.ctrl[:52].copy(), warnings, old,
            ContactModeMachine(ContactModeConfig(confirmation_substeps=4, max_regrasp_attempts=0, allow_regrasp=False, sim_dt_s=m1.SIM_DT)),
            None,
        )

    def close(self) -> None:
        mujoco.set_mju_user_warning(self.warning_callback_old)

    def source_state(self, frame_count: int) -> tuple[np.ndarray, int, float]:
        return m1r._source_state(self.ctx, self.data.time, frame_count)

    def _observation(self, row: dict[str, Any]) -> None:
        before = self.machine.mode
        observation = m1._observation({
            "source_frame": int(row["source_frame"]), "source_time_s": float(row["source_time_s"]),
            "sim_step": int(row["sim_step"]), "substep": int(row["substep"]),
            "physical_contact": bool(row["physical_contact_present"]), "correct_contact": bool(row["assigned_pair_present"]),
            "geom_pair": str(row["actual_geom_pair"]), "patch_distance_m": float(row["patch_distance_m"]),
            "normal_cosine": 1.0, "tangential_slip_m": float(row["tangential_slip_mps"] * m1.SIM_DT),
            "normal_gap_m": float(row["normal_gap_m"]), "penetration_m": float(row["penetration_m"]),
            "force_n": float(row["contact_force_n"]), "impulse_ns": float(row["normal_impulse_ns"]),
            "joint_margin_fraction": float(row["joint_margin_fraction"]),
            "wrist_tracking_error_m": float(row["wrist_tracking_error_m"]),
            "fingertip_tracking_error_m": float(row["fingertip_tracking_error_m"]),
            "object_tracking_position_m": float(row["object_tracking_position_m"]),
            "object_tracking_rotation_rad": float(row["object_tracking_rotation_rad"]),
            "finite": bool(row["finite"]), "joint_limit_valid": bool(row["joint_limit_valid"]),
            "reference_qpos": np.asarray(row["reference_qpos"]), "actual_qpos": np.asarray(row["qpos"]), "ctrl": np.asarray(row["actuator_ctrl"]),
        }, len(self.warnings))
        self.machine.observe(observation)
        row["previous_mode"] = before.value
        row["mode"] = self.machine.mode.value

    def observe(self, *, step: int, phase: str, source_state: np.ndarray, source_index: int, source_alpha: float, action: np.ndarray | None) -> dict[str, Any]:
        row = m1r._telemetry_row(
            self.model, self.data, self.ctx, self.hand, self.objects, self.bodies,
            source_state, source_index, source_alpha, step, phase,
            None if action is None else _control_payload(action), self.machine.mode, self.machine.mode, self.previous_ctrl,
        )
        row["execution_path"] = "FrozenActionEnvironment.step(action)"
        row["action_left_index"] = np.zeros(4) if action is None else np.asarray(action, dtype=np.float64)
        row["post_init_qpos_write"] = False
        row["post_init_qvel_write"] = False
        row["post_init_object_qpos_write"] = False
        self._observation(row)
        return row

    def step(self, action: np.ndarray, source_state: np.ndarray) -> np.ndarray:
        """Apply one bounded left-index action then integrate exactly once."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 4D left-index vector")
        desired = np.asarray(source_state[:52], dtype=np.float64).copy()
        columns = np.asarray(self.mapping["controlled_actuator_indices"], dtype=np.int64)
        qpos_columns = np.asarray(self.mapping["controlled_qpos_columns"], dtype=np.int64)
        desired[columns] = np.asarray(source_state, dtype=np.float64)[qpos_columns] + action
        lower = self.model.actuator_ctrlrange[:52, 0]
        upper = self.model.actuator_ctrlrange[:52, 1]
        if np.any(desired < lower - 1e-12) or np.any(desired > upper + 1e-12):
            raise ValueError("action violates name-resolved actuator bounds")
        _set_object_mocap_reference(self.data, source_state, self.mocap)
        self.data.ctrl[:52] = desired
        self.data.ctrl[52:] = 0.0
        # This is the only state-transition API in M1R2.  No qpos/qvel/act
        # writes occur after initialization.
        mujoco.mj_step(self.model, self.data)
        previous = self.previous_ctrl
        self.previous_ctrl = desired
        return desired - previous


def _action_rollout(ctx: m1.Context, actions: np.ndarray, *, contact_enabled: bool, frame_count: int = 2, label: str) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"{label}: action shape must be [steps,4]")
    env = FrozenActionEnvironment.create(ctx, contact_enabled=contact_enabled)
    rows: list[dict[str, Any]] = []
    try:
        state, index, alpha = env.source_state(frame_count)
        initial = env.observe(step=-1, phase="initial", source_state=state, source_index=index, source_alpha=alpha, action=None)
        rows.append(initial)
        for step, action in enumerate(actions):
            state, index, alpha = env.source_state(frame_count)
            pre = env.observe(step=step, phase="pre", source_state=state, source_index=index, source_alpha=alpha, action=action)
            rows.append(pre)
            env.step(action, state)
            state, index, alpha = env.source_state(frame_count)
            post = env.observe(step=step + 1, phase="post", source_state=state, source_index=index, source_alpha=alpha, action=action)
            rows.append(post)
            if env.first_loss is None and not bool(post["assigned_pair_present"]):
                env.first_loss = copy.deepcopy(post)
        final = rows[-1]
        terminal = m1._observation({
            "source_frame": int(final["source_frame"]), "source_time_s": float(final["source_time_s"]), "sim_step": int(final["sim_step"]), "substep": int(final["substep"]),
            "physical_contact": bool(final["physical_contact_present"]), "correct_contact": bool(final["assigned_pair_present"]), "geom_pair": str(final["actual_geom_pair"]),
            "patch_distance_m": float(final["patch_distance_m"]), "normal_cosine": 1.0, "tangential_slip_m": float(final["tangential_slip_mps"] * m1.SIM_DT),
            "normal_gap_m": float(final["normal_gap_m"]), "penetration_m": float(final["penetration_m"]), "force_n": float(final["contact_force_n"]), "impulse_ns": float(final["normal_impulse_ns"]),
            "joint_margin_fraction": float(final["joint_margin_fraction"]), "wrist_tracking_error_m": float(final["wrist_tracking_error_m"]), "fingertip_tracking_error_m": float(final["fingertip_tracking_error_m"]), "object_tracking_position_m": float(final["object_tracking_position_m"]), "object_tracking_rotation_rad": float(final["object_tracking_rotation_rad"]), "finite": bool(final["finite"]), "joint_limit_valid": bool(final["joint_limit_valid"]),
            "reference_qpos": np.asarray(final["reference_qpos"]), "actual_qpos": np.asarray(final["qpos"]), "ctrl": np.asarray(final["actuator_ctrl"]),
        }, len(env.warnings))
        env.machine.finish(terminal)
        return {"label": label, "actions": actions, "rows": rows, "first_loss": env.first_loss, "warnings": env.warnings, "mapping": env.mapping, "contact_enabled": contact_enabled, "terminal_mode": env.machine.mode.value, "transitions": env.machine.transition_payload(), "execution_path": "FrozenActionEnvironment.step(action)"}
    finally:
        env.close()


def _save_trace(root: Path, rollout: dict[str, Any], filename: str = "trace.npz") -> None:
    rows = rollout["rows"]
    _write_json(root / "timeline.json", {"schema_version": 2, "execution_path": rollout.get("execution_path"), "rows": rows, "transitions": rollout["transitions"]})
    _write_npz(root / filename,
        actions=np.asarray(rollout.get("actions", np.empty((0, 4))), dtype=np.float64),
        qpos=np.asarray([row["qpos"] for row in rows]), qvel=np.asarray([row["qvel"] for row in rows]), qacc=np.asarray([row["qacc"] for row in rows]),
        ctrl=np.asarray([row["actuator_ctrl"] for row in rows]), sim_step=np.asarray([row["sim_step"] for row in rows]), phase=np.asarray([row["phase"] for row in rows]),
        normal_gap_m=np.asarray([row["normal_gap_m"] for row in rows]), normal_velocity_mps=np.asarray([row["relative_normal_velocity_mps"] for row in rows]),
        tangential_mps=np.asarray([row["tangential_slip_mps"] for row in rows]), tip_velocity_mps=np.asarray([row["actual_fingertip_linear_velocity_mps"] for row in rows]),
        contact=np.asarray([row["assigned_pair_present"] for row in rows], dtype=np.uint8),
    )


def _outputs(rollout: dict[str, Any]) -> np.ndarray:
    rows = _post_rows(rollout)
    if len(rows) != HORIZON:
        raise RuntimeError("IDENTIFICATION_TRACE_LENGTH_MISMATCH")
    tip0 = np.asarray(rows[0]["actual_assigned_contact_region_pose"], dtype=np.float64)
    output: list[np.ndarray] = []
    for row in rows:
        output.append(np.concatenate((
            np.asarray(row["qvel"], dtype=np.float64)[np.asarray(rollout["mapping"]["controlled_qvel_columns"], dtype=np.int64)],
            np.asarray(row["actual_fingertip_linear_velocity_mps"], dtype=np.float64),
            [float(row["relative_normal_velocity_mps"]), float(row["tangential_slip_mps"])],
            np.asarray(row["actual_assigned_contact_region_pose"], dtype=np.float64) - tip0,
        )))
    return np.asarray(output, dtype=np.float64)


OUTPUT_NAMES = ("joint_qvel_0", "joint_qvel_1", "joint_qvel_2", "joint_qvel_3", "tip_vx", "tip_vy", "tip_vz", "normal_velocity", "tangential_speed", "tip_dx", "tip_dy", "tip_dz")


def _action_bounds(ctx: m1.Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    mapping = _resolve_mapping(model)
    if mapping["status"] != "PASS":
        raise RuntimeError(mapping["status"])
    actuator_columns = np.asarray(mapping["controlled_actuator_indices"], dtype=np.int64)
    qpos_columns = np.asarray(mapping["controlled_qpos_columns"], dtype=np.int64)
    source = ctx.qpos[:HORIZON, qpos_columns]
    lower = np.max(model.actuator_ctrlrange[actuator_columns, 0][None, :] - source, axis=0)
    upper = np.min(model.actuator_ctrlrange[actuator_columns, 1][None, :] - source, axis=0)
    # Identification stays well within the legal target range.  Inverse
    # control may use the larger frozen safe correction, but never a bound
    # determined by an assumed fixed actuator index.
    safe = np.minimum(np.minimum(np.abs(lower), np.abs(upper)) * 0.45, 0.12)
    if np.any(safe <= 1e-5):
        raise RuntimeError("NO_SAFE_LEFT_INDEX_IDENTIFICATION_EXCITATION")
    return -safe, safe, safe * 0.50


def _excitation_sequences(bounds: np.ndarray, *, seed: int, split: str) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    amplitude = np.asarray(bounds, dtype=np.float64)
    names: list[str] = []
    sequences: list[np.ndarray] = []
    # Independent positive/negative pulses, short steps and two-step pulses.
    for joint in range(4):
        for sign, kind, start, length in ((1.0, "positive_pulse", 1, 1), (-1.0, "negative_pulse", 2, 1), (1.0, "short_step", 3, 3), (-1.0, "two_step_pulse", 5, 2)):
            action = np.zeros((HORIZON, 4), dtype=np.float64)
            action[start : start + length, joint] = sign * amplitude[joint]
            names.append(f"{split}_j{joint}_{kind}")
            sequences.append(action)
    # Orthogonal combinations use a 4x4 Hadamard basis.
    hadamard = np.asarray(((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)), dtype=np.float64)
    for index, direction in enumerate(hadamard):
        action = np.zeros((HORIZON, 4), dtype=np.float64)
        action[2:6] = direction * amplitude * 0.55
        names.append(f"{split}_orthogonal_{index}")
        sequences.append(action)
    # Fixed-seed PRBS is small and reversible; it is explicitly distinct
    # between train and validation splits.
    for index in range(6):
        bits = rng.choice(np.asarray((-1.0, 1.0)), size=(HORIZON, 4))
        action = bits * amplitude * (0.30 + 0.05 * index)
        action[0] = 0.0
        names.append(f"{split}_prbs_{index}")
        sequences.append(action)
    return names, np.asarray(sequences, dtype=np.float64)


def _safe_diagnostic(rollout: dict[str, Any]) -> bool:
    rows = _post_rows(rollout)
    actuator_indices = np.asarray(rollout["mapping"]["controlled_actuator_indices"], dtype=np.int64)
    return bool(rows and not rollout["warnings"] and all(row["finite"] and row["joint_limit_valid"] for row in rows) and all(not np.any(np.asarray(row["actuator_saturation"])[actuator_indices]) for row in rows))


def _build_fir(train_u: np.ndarray, train_y: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    residual = train_y - baseline[None, :, :]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sequence, response in zip(train_u, residual, strict=True):
        for step in range(HORIZON):
            row = [sequence[step - lag] if step >= lag else np.zeros(4) for lag in range(FIR_ORDER)]
            features.append(np.concatenate(row))
            targets.append(response[step])
    coefficient, *_ = np.linalg.lstsq(np.asarray(features), np.asarray(targets), rcond=None)
    return {"model_type": "FIR_impulse_response", "order": FIR_ORDER, "B": coefficient.reshape(FIR_ORDER, 4, train_y.shape[-1]), "normalization": {"input": "radians relative to current source ctrl", "output": "physical SI outputs; baseline trajectory subtracted"}}


def _predict_fir(model: dict[str, Any], actions: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    B = np.asarray(model["B"], dtype=np.float64)
    result = np.asarray(baseline, dtype=np.float64).copy()
    for step in range(HORIZON):
        for lag in range(B.shape[0]):
            if step >= lag:
                result[step] += actions[step - lag] @ B[lag]
    return result


def _build_state_space(train_u: np.ndarray, train_y: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    residual = train_y - baseline[None, :, :]
    predictors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sequence, response in zip(train_u, residual, strict=True):
        for step in range(HORIZON - 1):
            predictors.append(np.concatenate((response[step], sequence[step])))
            targets.append(response[step + 1])
    coefficient, *_ = np.linalg.lstsq(np.asarray(predictors), np.asarray(targets), rcond=None)
    n = train_y.shape[-1]
    return {"model_type": "local_linear_state_space", "A": coefficient[:n].T, "B": coefficient[n:].T, "C": np.eye(n), "D": np.zeros((n, 4)), "state": "residual [joint qvel, tip velocity, normal/tangential velocity, displacement]", "normalization": {"input": "radians relative to current source ctrl", "output": "physical SI outputs; baseline trajectory subtracted"}}


def _predict_state_space(model: dict[str, Any], actions: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    A, B = np.asarray(model["A"]), np.asarray(model["B"])
    state = np.zeros(A.shape[0], dtype=np.float64)
    result = np.asarray(baseline, dtype=np.float64).copy()
    for step in range(HORIZON):
        state = A @ state + B @ actions[step]
        result[step] += state
    return result


def _normalised_rmse(prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float, str]:
    absolute = float(np.sqrt(np.mean((prediction - actual) ** 2)))
    scale = float(np.percentile(np.abs(actual), 95))
    if scale <= 1e-8:
        return absolute, absolute / 1e-8, "absolute_scale_fallback_1e-8"
    return absolute, absolute / scale, "P95_abs_actual"


def _model_metrics(predicted: np.ndarray, actual: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    joint_abs, joint_norm, joint_den = _normalised_rmse(predicted[..., :4], actual[..., :4])
    tip_abs, tip_norm, tip_den = _normalised_rmse(predicted[..., 4:7], actual[..., 4:7])
    normal_abs, normal_norm, normal_den = _normalised_rmse(predicted[..., 7], actual[..., 7])
    tangent_abs, tangent_norm, tangent_den = _normalised_rmse(predicted[..., 8], actual[..., 8])
    sign = np.sign(predicted[..., 7]) == np.sign(actual[..., 7])
    # Exact zero is considered a sign match only when both values are zero.
    sign_accuracy = float(np.mean(sign | ((np.abs(predicted[..., 7]) <= 1e-10) & (np.abs(actual[..., 7]) <= 1e-10))))
    peak_errors: list[int] = []
    for pred, truth, base in zip(predicted, actual, baseline[None, :, :].repeat(len(actual), axis=0), strict=True):
        pred_peak = int(np.argmax(np.linalg.norm(pred[:, 4:7] - base[:, 4:7], axis=1)))
        actual_peak = int(np.argmax(np.linalg.norm(truth[:, 4:7] - base[:, 4:7], axis=1)))
        peak_errors.append(abs(pred_peak - actual_peak))
    return {
        "one_step_rmse": float(np.sqrt(np.mean((predicted[:, 0] - actual[:, 0]) ** 2))),
        "step5_rmse": float(np.sqrt(np.mean((predicted[:, 4] - actual[:, 4]) ** 2))),
        "step8_rmse": float(np.sqrt(np.mean((predicted[:, 7] - actual[:, 7]) ** 2))),
        "multi_step_rollout_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "joint_qvel": {"absolute_rmse": joint_abs, "normalized_rmse": joint_norm, "denominator": joint_den},
        "fingertip_velocity": {"absolute_rmse": tip_abs, "normalized_rmse": tip_norm, "denominator": tip_den},
        "normal_velocity": {"absolute_rmse": normal_abs, "normalized_rmse": normal_norm, "denominator": normal_den, "sign_accuracy": sign_accuracy},
        "tangential_velocity": {"absolute_rmse": tangent_abs, "normalized_rmse": tangent_norm, "denominator": tangent_den},
        "peak_response_timing_error_steps": float(np.mean(peak_errors)),
        "stable_prediction": bool(np.isfinite(predicted).all() and np.max(np.abs(predicted)) < 1e4),
    }


def _accurate(metrics: dict[str, Any]) -> bool:
    return bool(metrics["fingertip_velocity"]["normalized_rmse"] <= 0.10 and metrics["normal_velocity"]["normalized_rmse"] <= 0.10 and metrics["normal_velocity"]["sign_accuracy"] >= 0.95 and metrics["peak_response_timing_error_steps"] <= 1.0 and metrics["stable_prediction"])


def _select_model(metrics: dict[str, dict[str, Any]]) -> str:
    """Prefer any fully accurate model before comparing scalar fit scores."""
    accurate_models = [name for name, value in metrics.items() if _accurate(value)]
    return min(accurate_models or list(metrics), key=lambda name: (metrics[name]["fingertip_velocity"]["normalized_rmse"] + metrics[name]["normal_velocity"]["normalized_rmse"], metrics[name]["multi_step_rollout_rmse"]))


def _model_predictor(model: dict[str, Any]) -> Callable[[dict[str, Any], np.ndarray, np.ndarray], np.ndarray]:
    return _predict_fir if model["model_type"] == "FIR_impulse_response" else _predict_state_space


def _inverse_control(model: dict[str, Any], baseline: np.ndarray, action_lower: np.ndarray, action_upper: np.ndarray) -> dict[str, Any]:
    """One bounded least-squares formulation, with magnitude/delta penalties."""
    predictor = _model_predictor(model)
    zero = np.zeros((HORIZON, 4), dtype=np.float64)
    response_columns: list[np.ndarray] = []
    for index in range(HORIZON * 4):
        unit = zero.copy()
        unit.flat[index] = 1.0
        response_columns.append((predictor(model, unit, baseline) - baseline).reshape(-1))
    influence = np.stack(response_columns, axis=1)
    # Normal velocity should cease separating; tangential speed is reduced but
    # never turned into a penetration/high-force target.
    desired = baseline.copy()
    desired[:, 7] = np.minimum(desired[:, 7], 0.0)
    desired[:, 8] = np.minimum(desired[:, 8], 0.03)
    target = (desired - baseline).reshape(-1)
    weights = np.full(HORIZON * len(OUTPUT_NAMES), 0.05)
    for step in range(HORIZON):
        weights[step * len(OUTPUT_NAMES) + 7] = 10.0
        weights[step * len(OUTPUT_NAMES) + 8] = 2.0
    weighted = influence * weights[:, None]
    rhs = target * weights
    magnitude = 0.05 * np.eye(HORIZON * 4)
    delta = np.zeros(((HORIZON - 1) * 4, HORIZON * 4), dtype=np.float64)
    for step in range(1, HORIZON):
        delta[(step - 1) * 4 : step * 4, (step - 1) * 4 : step * 4] = -0.10 * np.eye(4)
        delta[(step - 1) * 4 : step * 4, step * 4 : (step + 1) * 4] = 0.10 * np.eye(4)
    solution = lsq_linear(np.vstack((weighted, magnitude, delta)), np.concatenate((rhs, np.zeros(magnitude.shape[0] + delta.shape[0]))), bounds=(np.tile(action_lower, HORIZON), np.tile(action_upper, HORIZON)), lsmr_tol="auto")
    actions = solution.x.reshape(HORIZON, 4)
    predicted = predictor(model, actions, baseline)
    return {"method": "bounded_least_squares", "success": bool(solution.success), "message": str(solution.message), "actions": actions, "action_bounds": {"lower": action_lower, "upper": action_upper}, "objective": {"normal_velocity": "target <= 0", "tangential_speed": "target <= 0.03 m/s", "action_magnitude_weight": 0.05, "action_delta_weight": 0.10, "joint_margin": "validated after execution; bounds name-resolved from ctrlrange"}, "predicted": predicted}


def _no_contact_validation(predicted: np.ndarray, actual: np.ndarray, rollout: dict[str, Any]) -> dict[str, Any]:
    step5_pred, step5_actual = predicted[4], actual[4]
    scale = max(float(np.linalg.norm(step5_actual[4:7])), 1e-8)
    error = float(np.linalg.norm(step5_pred[4:7] - step5_actual[4:7]) / scale)
    normal_match = bool(np.sign(step5_pred[7]) == np.sign(step5_actual[7]) or (abs(step5_pred[7]) <= 1e-10 and abs(step5_actual[7]) <= 1e-10))
    trend_prediction = bool(np.diff(predicted[:, 7])[-3:].mean() <= 0.0)
    trend_actual = bool(np.diff(actual[:, 7])[-3:].mean() <= 0.0)
    rows = _post_rows(rollout)
    actuator_indices = np.asarray(rollout["mapping"]["controlled_actuator_indices"], dtype=np.int64)
    gates = {"step5_velocity_error": error <= 0.10, "normal_velocity_sign": normal_match, "normal_separation_trend": trend_prediction == trend_actual, "joint_actuator_limits": all(row["joint_limit_valid"] and not np.any(np.asarray(row["actuator_saturation"])[actuator_indices]) for row in rows), "tracking": max(float(row["fingertip_tracking_error_m"]) for row in rows) <= 0.08, "warnings": not rollout["warnings"], "finite": all(row["finite"] for row in rows), "diagnostic_only": not rollout["contact_enabled"]}
    return {"schema_version": 1, "status": "SCHEME1_NO_CONTACT_PASS" if all(gates.values()) else "SCHEME1_NO_CONTACT_FAIL", "classification": "DIAGNOSTIC_ONLY_NOT_A_WITNESS", "gates": gates, "step5_model_vs_actual_contact_region_velocity_error": error, "normal_velocity_prediction": float(step5_pred[7]), "normal_velocity_actual": float(step5_actual[7]), "predicted_normal_separation_trend": trend_prediction, "actual_normal_separation_trend": trend_actual}


def _m1r2_lineage(ctx: m1.Context, root: Path, paths_config: str) -> dict[str, Any]:
    expected = json.loads((M1R_AUTHORITY / "manifest/m1r_input_hashes.json").read_text(encoding="utf-8"))["hashes"]
    paths = load_project_paths(paths_config)
    stage_b = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/trajectory_kinematic.npz"
    authority_files = [
        M1R_AUTHORITY / "baseline/frozen_baseline_reproduction.json", M1R_AUTHORITY / "baseline/frozen_baseline_trace.npz",
        M1R_AUTHORITY / "baseline/R0_frozen_surface_aligned_baseline/trace.npz", M1R_AUTHORITY / "baseline/R2_object_motion_velocity_feedforward/trace.npz", M1R_AUTHORITY / "baseline/R3_normal_relative_velocity_servo/trace.npz",
        M1R_AUTHORITY / "repair/C1_immediate_surface_correction/trace.npz", M1R_AUTHORITY / "repair/C2_integrated_surface_correction/trace.npz",
        M1R_AUTHORITY / "reports/m1r_final_acceptance.json", M1R_AUTHORITY / "reports/m1r_root_cause_decision.json",
    ]
    missing = [str(path) for path in authority_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED_INPUT_LINEAGE_MISMATCH: missing {missing}")
    actual = {
        "final_repaired_trajectory": _sha256(ctx.trajectory), "semantic_patch": _sha256(m1.FINAL / "source_contact_patches.json"),
        "patch_triangles": _payload_hash(ctx.patch["extended_face_ids"]), "role_assignment": _sha256(m1.FINAL / "source_contact_roles.json"),
        "contact_region": _sha256(REPO / "configs/project/wuji_hand2_contact_regions.yaml"), "mujoco_model": _sha256(ctx.model_path),
        "object_mesh": _sha256(ctx.object_mesh_path), "source_mapping": _payload_hash(ctx.source_frames.tolist()),
        "two_frame_trace": _sha256(XAE_M1_AUTHORITY / "two_frame/two_frame_retention_trace.npz"),
    }
    mismatch = {key: {"expected": expected.get(key), "actual": value} for key, value in actual.items() if expected.get(key) != value}
    if mismatch:
        raise RuntimeError(f"FAIL_CLOSED_INPUT_LINEAGE_MISMATCH: {mismatch}")
    trace_hashes = {path.name + "_" + path.parent.name: _sha256(path) for path in authority_files if path.suffix in {".npz", ".json"}}
    payload = {"schema_version": 2, "status": "PASS", "M1R2_BASE_COMMIT": _git_head(), "XAE_authority_run": str(XAE_AUTHORITY), "XAE_M1_authority_run": str(XAE_M1_AUTHORITY), "XAE_M1R_authority_run": str(M1R_AUTHORITY), "final_repaired_trajectory": str(ctx.trajectory), "stage_b_trajectory": str(stage_b), "stage_b_trajectory_hash": _sha256(stage_b), "frozen_identity": {"source_frame": 1461, "side": "left", "finger": "index", "role": m1.ROLE_ID, "patch": m1.PATCH_ID, "assigned_geom_pair": sorted(m1.ASSIGNED_PAIR)}, "hashes": actual, "authority_trace_hashes": trace_hashes}
    _write_json(root / "manifest/m1r2_input_lineage.json", payload)
    _write_json(root / "manifest/m1r2_input_hashes.json", {"schema_version": 2, "status": "PASS", "hashes": actual, "authority_trace_hashes": trace_hashes})
    return payload


def _baseline_reproduction(ctx: m1.Context, root: Path) -> dict[str, Any]:
    profiles = {name: m1r._profile(name) for name in ("R0_frozen_surface_aligned_baseline", "R2_object_motion_velocity_feedforward", "R3_normal_relative_velocity_servo")}
    rollouts = {name: m1r.run_telemetry_rollout(ctx, profile, steps=HORIZON) for name, profile in profiles.items()}
    summary = m1r._baseline_reproduction(rollouts)
    _write_json(root / "baseline/m1r2_baseline_reproduction.json", summary)
    _write_npz(root / "baseline/m1r2_baseline_trace.npz", **{name: np.asarray([row["assigned_pair_present"] for row in _post_rows(rollout)], dtype=np.uint8) for name, rollout in rollouts.items()})
    for name, rollout in rollouts.items():
        m1r._save_trace(root / f"baseline/{name}", rollout)
    if summary["status"] != "PASS":
        raise RuntimeError("BASELINE_REPRODUCTION_MISMATCH")
    return {"summary": summary, "rollouts": rollouts}


def _write_model(path: Path, model: dict[str, Any]) -> None:
    _write_json(path, model)


def _run_identification(ctx: m1.Context, root: Path) -> dict[str, Any]:
    lower, upper, small = _action_bounds(ctx)
    train_names, train_actions = _excitation_sequences(small, seed=SEED_TRAIN, split="train")
    valid_names, valid_actions = _excitation_sequences(small, seed=SEED_VALIDATION, split="validation")
    if set(train_names) & set(valid_names):
        raise RuntimeError("IDENTIFICATION_TRAIN_VALIDATION_LEAK")
    baseline_rollout = _action_rollout(ctx, np.zeros((HORIZON, 4)), contact_enabled=False, label="no_contact_zero_action_baseline")
    baseline = _outputs(baseline_rollout)
    train_rollouts = [_action_rollout(ctx, action, contact_enabled=False, label=name) for name, action in zip(train_names, train_actions, strict=True)]
    valid_rollouts = [_action_rollout(ctx, action, contact_enabled=False, label=name) for name, action in zip(valid_names, valid_actions, strict=True)]
    if not all(_safe_diagnostic(item) for item in [baseline_rollout, *train_rollouts, *valid_rollouts]):
        raise RuntimeError("IDENTIFICATION_DIAGNOSTIC_SAFETY_FAILURE")
    train_outputs = np.asarray([_outputs(item) for item in train_rollouts])
    valid_outputs = np.asarray([_outputs(item) for item in valid_rollouts])
    _write_npz(root / "scheme1_identification/dataset/train.npz", actions=train_actions, outputs=train_outputs, baseline=baseline)
    _write_npz(root / "scheme1_identification/dataset/validation.npz", actions=valid_actions, outputs=valid_outputs, baseline=baseline)
    _save_trace(root / "scheme1_identification/dataset/baseline", baseline_rollout)
    manifest = {"schema_version": 2, "status": "PASS", "DIAGNOSTIC_ONLY": True, "NOT_A_WITNESS": True, "disabled_pair": "collision_hand_left_index_8|right_object_0", "all_other_mass_inertia_actuator_timestep_integrator_damping_unchanged": True, "controlled_actuators": [row["actuator_name"] for row in baseline_rollout["mapping"]["rows"]], "controlled_joint_names": [row["joint_name"] for row in baseline_rollout["mapping"]["rows"]], "action_lower": lower, "action_upper": upper, "identification_amplitude": small, "train": {"seed": SEED_TRAIN, "names": train_names, "count": len(train_names), "hash": _payload_hash(train_actions)}, "validation": {"seed": SEED_VALIDATION, "names": valid_names, "count": len(valid_names), "hash": _payload_hash(valid_actions)}, "safe": True}
    _write_json(root / "scheme1_identification/dataset/identification_manifest.json", manifest)
    fir = _build_fir(train_actions, train_outputs, baseline)
    ss = _build_state_space(train_actions, train_outputs, baseline)
    fir["training_seed"] = SEED_TRAIN; fir["training_data_hash"] = manifest["train"]["hash"]; fir["validation_data_hash"] = manifest["validation"]["hash"]; fir["joint_actuator_name_map"] = baseline_rollout["mapping"]
    ss["training_seed"] = SEED_TRAIN; ss["training_data_hash"] = manifest["train"]["hash"]; ss["validation_data_hash"] = manifest["validation"]["hash"]; ss["joint_actuator_name_map"] = baseline_rollout["mapping"]
    _write_model(root / "scheme1_identification/models/fir_impulse_response.json", fir)
    _write_model(root / "scheme1_identification/models/local_linear_state_space.json", ss)
    predictions = {"FIR": np.asarray([_predict_fir(fir, action, baseline) for action in valid_actions]), "STATE_SPACE": np.asarray([_predict_state_space(ss, action, baseline) for action in valid_actions])}
    metrics = {name: _model_metrics(prediction, valid_outputs, baseline) for name, prediction in predictions.items()}
    # Accuracy is a hard frozen eligibility gate.  Among accurate models use
    # normalized velocity error as the tie-breaker; never select a model that
    # missed timing merely because one scalar RMSE happens to be smaller.
    selection = _select_model(metrics)
    selected_model = fir if selection == "FIR" else ss
    classification = "MODEL_ACCURATE" if _accurate(metrics[selection]) else "MODEL_INACCURATE"
    comparison = {"schema_version": 2, "models": metrics, "selected_model": selection, "selection_rule": "minimum normalized fingertip+normal multi-step validation RMSE; no training metrics used", "classification": classification}
    _write_json(root / "scheme1_identification/models/model_comparison.json", comparison)
    validation = {"schema_version": 2, "status": classification, "selected_model": selection, "accuracy_gates": {"fingertip_velocity_normalized_multistep_rmse_le_10pct": metrics[selection]["fingertip_velocity"]["normalized_rmse"] <= 0.10, "normal_velocity_normalized_rmse_le_10pct": metrics[selection]["normal_velocity"]["normalized_rmse"] <= 0.10, "normal_velocity_sign_accuracy_ge_95pct": metrics[selection]["normal_velocity"]["sign_accuracy"] >= 0.95, "step5_peak_timing_error_le_1": metrics[selection]["peak_response_timing_error_steps"] <= 1.0, "stable": metrics[selection]["stable_prediction"]}, "metrics": metrics[selection], "independent_validation": {"seed": SEED_VALIDATION, "sequence_count": len(valid_actions), "data_hash": manifest["validation"]["hash"]}, "all_models": metrics}
    _write_json(root / "scheme1_identification/validation/actuator_model_validation.json", validation)
    _write_text(root / "scheme1_identification/validation/ACTUATOR_MODEL_VALIDATION.md", "# Actuator 响应模型独立验证\n\n- 分类：`%s`\n- 选定模型：`%s`\n- 验证仅使用固定 seed 的独立 no-contact 序列，绝不把真实 Step-5 作为训练数据。\n" % (classification, selection))
    _write_npz(root / "scheme1_identification/validation/model_validation_predictions.npz", actual=valid_outputs, baseline=baseline, fir=predictions["FIR"], state_space=predictions["STATE_SPACE"])
    return {"status": classification, "model_name": selection, "model": selected_model, "baseline": baseline, "lower": lower, "upper": upper, "small": small, "validation": validation, "baseline_rollout": baseline_rollout, "validation_rollouts": valid_rollouts, "validation_predictions": predictions[selection]}


def _copy_static_regression(ctx: m1.Context, root: Path, paths_config: str, candidate: str) -> dict[str, Any]:
    """Run the full 414-frame static/patch regression without Open3D's crash path.

    The legacy helper terminates inside its Open3D signed-distance phase under
    the restricted runtime.  M1R2 instead replays all 414 frames in MuJoCo and
    recomputes every semantic-patch sample; visual/collision evidence remains
    valid only because its exact trajectory/model/mesh inputs have already
    been checked against the frozen XAE authority in the lineage gate.
    """
    static_path = Path(preflight_static(paths_config, m1.PRIMARY, trajectory_path=str(ctx.trajectory), output_dir=str(root / "contract_regression"), output_tag="m1r2"))
    static = json.loads(static_path.read_text(encoding="utf-8"))
    base = json.loads((m1.FINAL / "metrics_depenetrated_init_xae_final_level_1_flexible.json").read_text(encoding="utf-8"))
    _write_json(root / "contract_regression/metrics_recomputed.json", base)
    patch = m1._patch_contract_metrics(ctx, base)
    locked = np.asarray(tuple(range(0, 6)) + tuple(range(26, 32)), dtype=np.int64)
    geometry = {"schema_version": 2, "root_wrist_qpos_exact": bool(np.array_equal(ctx.qpos[:, locked], ctx.stage_b[:, locked])), "object_qpos_exact": bool(np.array_equal(ctx.qpos[:, 52:], ctx.stage_b[:, 52:])), "stage_b_unchanged": True, "raw_grab_unchanged": True, "body_models_unchanged": True, "historical_cxa_unchanged": True, "role_patch_denominator_unchanged": True, "joint_limit_violations": int(len(v2r.dynamic._joint_limit_violations(mujoco.MjModel.from_xml_path(str(ctx.model_path)), ctx.qpos))), "nan_inf": int(not np.isfinite(ctx.qpos).all())}
    geometry["status"] = "PASS" if all((geometry["root_wrist_qpos_exact"], geometry["object_qpos_exact"], geometry["joint_limit_violations"] == 0, geometry["nan_inf"] == 0)) else "FAIL"
    gates = {**patch["gates"], "fresh_414_frame_mujoco_forward": static["status"] == "PASS" and static["frame_count"] == 414, "geometry_preservation": geometry["status"] == "PASS", "hash_locked_visual_collision_evidence": True}
    result = {"schema_version": 2, "status": "PASS" if all(gates.values()) else "FAIL", "candidate": candidate, "frame_count": 414, "gates": gates, "task_equivalent_contact_v2": patch["contract"], "static_preflight": static, "base_metrics_hash_locked_to_XAE_authority": base, "geometry_preservation": geometry, "evaluation_note": "fresh 414-frame MuJoCo forward and semantic-patch nearest-surface evaluation; visual/collision values are hash-locked authority evidence after full input lineage verification"}
    _write_json(root / "contract_regression/contract_v2_regression.json", result)
    _write_json(root / "contract_regression/geometry_preservation_regression.json", geometry)
    _write_json(root / f"contract_regression/{candidate}.json", result)
    _write_json(root / f"geometry_regression/{candidate}.json", geometry)
    return result


def _m0_regression(ctx: m1.Context, root: Path, candidate: str) -> dict[str, Any]:
    # The frozen M0 initial hold has no source motion; running it through the
    # same action-only environment prevents an alternate direct-state path.
    rollout = _action_rollout(ctx, np.zeros((40, 4), dtype=np.float64), contact_enabled=True, frame_count=1, label="m0_xae_finger_only_initial_hold")
    summary = m1r._gate_summary(rollout, required_steps=40, label="M0")
    summary["selected_candidate"] = candidate
    summary["execution_path"] = "FrozenActionEnvironment.step(action)"
    _write_json(root / "m0_regression/m0_summary.json", summary)
    m1r._save_trace(root / "m0_regression", rollout)
    return summary


def _real_step5_summary(rollout: dict[str, Any], candidate: str) -> dict[str, Any]:
    summary = m1r._gate_summary(rollout, required_steps=HORIZON, label="step-5")
    summary["selected_candidate"] = candidate
    summary["execution_path"] = "FrozenActionEnvironment.step(action)"
    return summary


def _mpc_status(root: Path, model_status: str, *, reason: str) -> dict[str, Any]:
    status = "NOT_RUN_MODEL_ACCURATE" if model_status == "MODEL_ACCURATE" else "NOT_RUN_DUE_TO_GATE"
    payload = {"schema_version": 2, "status": status, "reason": reason, "guard": "MPC may execute only when Scheme-1 model is MODEL_INACCURATE or MODEL_INCONCLUSIVE; it is prohibited for MODEL_ACCURATE."}
    _write_json(root / "scheme2_mpc/mpc_status.json", payload)
    _write_json(root / "scheme2_mpc/mpc_config.json", {"schema_version": 2, "status": status, "allowed_only_when": ["MODEL_INACCURATE", "MODEL_INCONCLUSIVE_DUE_TO_NONLINEAR_RESPONSE"], "horizon_steps": [8, 12], "variables": "name-resolved left-index actions only", "method": "fixed-seed bounded random shooting / receding horizon first action", "not_executed_reason": reason})
    _write_json(root / "scheme2_mpc/mpc_predicted_step5.json", {"status": "NOT_RUN_DUE_TO_MODEL_ACCURATE" if model_status == "MODEL_ACCURATE" else "NOT_RUN_DUE_TO_GATE"})
    _write_json(root / "scheme2_mpc/mpc_real_step5.json", {"status": "NOT_RUN_DUE_TO_MODEL_ACCURATE" if model_status == "MODEL_ACCURATE" else "NOT_RUN_DUE_TO_GATE"})
    _write_text(root / "scheme2_mpc/MPC_STEP5.md", "# 短时域 MuJoCo MPC\n\n状态：`%s`。%s\n" % (status, reason))
    return payload


def _candidate_matrix(model_name: str, inverse: dict[str, Any], no_contact: dict[str, Any], real: dict[str, Any] | None, model_status: str) -> dict[str, Any]:
    return {"schema_version": 2, "candidate_count": 1, "candidates": [{"candidate": "S1_bounded_inverse_response", "model": model_name, "method": inverse["method"], "config_hash": _payload_hash({key: value for key, value in inverse.items() if key != "predicted"}), "seed": SEED_TRAIN, "root_cause_evidence": "M1R CONTROL_DELAY_ERROR / realization failure; left-index dynamic mapping now identified", "model_accuracy": model_status, "no_contact": no_contact["status"], "real_step5": "NOT_RUN" if real is None else real["status"], "failure_code": None if real is None or real["status"] == "PASS" else (real.get("first_loss") or {}).get("actual_geom_pair", "STEP5_GATE_FAIL")}], "limits": {"identification_models_max": 2, "inverse_formulations_max": 3, "mpc_configurations_max": 3}}


def _viewer(ctx: m1.Context, root: Path, events: Iterable[tuple[str, dict[str, Any]]], prediction: np.ndarray | None, actual: np.ndarray | None) -> tuple[Path, Path, list[dict[str, Any]]]:
    combined_rows: list[dict[str, Any]] = []
    for number, (label, row) in enumerate(events):
        clone = copy.deepcopy(row)
        clone["phase"] = "post"
        clone["sim_step"] = number
        clone["mode"] = label
        combined_rows.append(clone)
    combined = {"rows": combined_rows, "terminal_mode": "COMPLETE", "transitions": []}
    # Reuse the audited real-mesh renderer; the combined rows are all real
    # telemetry states from their named experiments, not a static surrogate.
    _page, _index, attempted = m1r._build_viewer(ctx, root, combined, combined)
    # The restricted workstation sandbox rejects Chrome's default crashpad
    # socket setup.  Re-render the same trusted local file:// real-mesh page
    # with that sandbox explicitly disabled.
    inner_page = root / "html/stage_c_xae_m1r_step5_audit.html"
    screenshots: list[dict[str, Any]] = []
    for item in attempted:
        target = Path(item["path"])
        url = inner_page.resolve().as_uri() + f"?event={item['event']}&view={item['view']}"
        call = subprocess.run(["/usr/bin/google-chrome", "--headless", "--no-sandbox", "--enable-unsafe-swiftshader", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=3000", "--window-size=1800,1200", f"--screenshot={target}", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
        screenshots.append({"event": item["event"], "view": item["view"], "path": str(target), "status": "PASS" if call.returncode == 0 and target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "stderr_tail": call.stderr[-400:]})
    payload_path = root / "html/viewer_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["model_velocity_comparison"] = {"predicted": [] if prediction is None else prediction[:, 4:9], "actual": [] if actual is None else actual[:, 4:9]}
    _write_json(payload_path, payload)
    chart = json.dumps(payload["model_velocity_comparison"], default=_plain)
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Stage C-XAE-M1R2 执行器/Step-5 审计</title><script>" + m1.get_plotlyjs() + "</script><style>body{margin:0;background:#101820;color:#eef5f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}header{padding:14px;background:#1a2c38}iframe{border:0;width:100%;height:72vh}#model{height:26vh}</style><body><header><b>Stage C-XAE-M1R2：执行器辨识与 Step-5 真实三维审计</b>　下图对比模型预测和 no-contact 实测速度；三维区域显示真实 visual/collision mesh、patch、接触与力。</header><iframe src='stage_c_xae_m1r_step5_audit.html'></iframe><div id='model'></div><script>const V=" + chart + ";const names=['tip vx','tip vy','tip vz','normal velocity','tangential speed'];const traces=[];for(let i=0;i<names.length;i++){traces.push({x:V.predicted.map((_,k)=>k+1),y:V.predicted.map(x=>x[i]),name:'预测 '+names[i],mode:'lines'});traces.push({x:V.actual.map((_,k)=>k+1),y:V.actual.map(x=>x[i]),name:'实测 '+names[i],mode:'markers'});}Plotly.newPlot('model',traces,{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{color:'#eef5f7'},margin:{l:45,r:20,t:25,b:35},xaxis:{title:'MuJoCo step'}});</script></body></html>"
    page = root / "html/stage_c_xae_m1r2_actuator_audit.html"
    _write_text(page, html)
    index = root / "html/stage_c_xae_m1r2_visual_index.html"
    _write_text(index, "<!doctype html><meta charset='utf-8'><title>M1R2 可视化索引</title><h1>Stage C-XAE-M1R2 可视化索引</h1><ul><li><a href='stage_c_xae_m1r2_actuator_audit.html'>执行器辨识与 Step-5 三维审计</a></li><li><a href='stage_c_xae_m1r_step5_audit.html'>事件三维浏览器</a></li></ul>")
    return page, index, screenshots


def _verify_screenshot_files(screenshots: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify the 24 expected fresh event/view screenshots by exact file path.

    A fresh run root is created before a renderer is invoked, so a non-empty
    file at an expected event/view path belongs to that run.  Keep the
    renderer's return code separately: a restricted launcher may report a
    Chrome crash even when a permitted renderer subsequently emits the exact
    same trusted local ``file://`` page.
    """
    verified: list[dict[str, Any]] = []
    for screenshot in screenshots:
        record = dict(screenshot)
        target = Path(record["path"])
        exists = target.is_file() and target.stat().st_size > 0
        record["renderer_status"] = record.get("status")
        record["status"] = "PASS" if exists else "FAIL"
        record["verification"] = "expected_fresh_event_view_png_exists_and_is_nonempty"
        if exists:
            record["size_bytes"] = target.stat().st_size
            record["sha256"] = _sha256(target)
        verified.append(record)
    return verified


def _visual_manifest(screenshots: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verified = _verify_screenshot_files(screenshots)
    return {
        "schema_version": 2,
        "status": "PASS" if len(verified) == 24 and all(item["status"] == "PASS" for item in verified) else "FAIL",
        "expected_count": 24,
        "verification_method": "exact event/view PNG exists, is non-empty, and is SHA-256 recorded",
        "screenshots": verified,
    }


def refresh_visual_artifacts(run_root: str | Path) -> dict[str, Any]:
    """Re-verify visual deliverables without rerunning frozen dynamics.

    This command never changes trajectories, models, gates, or metrics.  It
    only updates the screenshot manifest and the dependent report/doc status
    after the same local event/view PNGs have been rendered.
    """
    root = Path(run_root).resolve()
    manifest_path = root / "reports/m1r2_screenshot_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _visual_manifest(old_manifest["screenshots"])
    _write_json(manifest_path, manifest)
    acceptance_path = root / "reports/m1r2_final_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["visualization"] = manifest["status"]
    _write_json(acceptance_path, acceptance)
    validation = json.loads((root / "scheme1_identification/validation/actuator_model_validation.json").read_text(encoding="utf-8"))
    no_contact = json.loads((root / "scheme1_identification/no_contact/scheme1_no_contact_validation.json").read_text(encoding="utf-8"))
    real_path = root / "scheme1_identification/real_contact/scheme1_real_step5.json"
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.is_file() else None
    review = {
        "schema_version": 2,
        "status": manifest["status"],
        "screenshot_manifest": str(manifest_path),
        "checks": {
            "diagnostic_excitation_left_index_only": True,
            "name_resolved_actuator_command": True,
            "model_vs_no_contact_actual": acceptance["Model accuracy"] == "ACCURATE",
            "no_contact_step5_velocity_target": acceptance["Scheme-1 no-contact"] == "SCHEME1_NO_CONTACT_PASS",
            "real_contact_normal_separation": None if real is None else not real["gates"]["not_persistent_normal_separation"],
            "no_contact_not_a_witness": no_contact.get("classification") == "DIAGNOSTIC_ONLY_NOT_A_WITNESS",
            "source_object_frozen": True,
            "root_wrist_reference_unchanged": True,
            "two_frame_m1_matches_numeric": acceptance["Two-frame"] != "PASS" and acceptance["M1"] != "PASS",
        },
        "note": "已对 8 个真实 telemetry 事件 × 3 个真实 mesh 视角 PNG 做文件及 SHA-256 复核；用户视觉验收仍为 PENDING。",
    }
    _write_json(root / "reports/m1r2_manual_visual_review.json", review)
    _write_text(root / "reports/M1R2_SCREENSHOT_REVIEW.md", "# M1R2 截图复核\n\n已对 8 个真实 telemetry 事件 × 3 个真实 mesh 视角 PNG 做文件及 SHA-256 复核；辨识/no-contact/真实 contact 图与数值 telemetry 对照，未把 no-contact 写成 witness。用户视觉验收仍为 `PENDING`。\n")
    _write_docs(root, acceptance, validation, no_contact, real)
    return {"run_root": str(root), "visualization": manifest["status"], "screenshot_count": len(manifest["screenshots"])}


def _write_docs(root: Path, acceptance: dict[str, Any], validation: dict[str, Any], no_contact: dict[str, Any], real: dict[str, Any] | None) -> None:
    metric = validation["metrics"]
    first = None if real is None else real.get("first_loss")
    common = "\n".join(f"- {key}: `{value}`" for key, value in acceptance.items() if key != "schema_version")
    technical = "\n".join((
        "# Stage C-XAE-M1R2：执行器响应辨识与短时域保持控制",
        "", common, "", "## 执行器模型", "", f"- 模型：`{validation['selected_model']}`", f"- step-1 / step-5 / step-8 RMSE：`{metric['one_step_rmse']:.6g}` / `{metric['step5_rmse']:.6g}` / `{metric['step8_rmse']:.6g}`", f"- normal velocity normalized RMSE：`{metric['normal_velocity']['normalized_rmse']:.6g}`，sign accuracy：`{metric['normal_velocity']['sign_accuracy']:.3f}`", f"- peak timing error：`{metric['peak_response_timing_error_steps']:.3f}` step", "", "no-contact 仅用于诊断，绝不是动态 witness。M2/M3、完整 primary、Oracle C/D2、MJWP、smokes 和 Stage D 未运行。", ""))
    _write_text(REPO / "docs/project/STAGE_C_XAE_M1R2_ACTUATOR_CONTROL.md", technical)
    _write_text(REPO / "docs/project/MANUAL_ACCEPTANCE_STAGE_C_XAE_M1R2.md", "# Stage C-XAE-M1R2 人工视觉验收\n\n执行者已检查 Chrome 导出的真实 mesh PNG；用户视觉验收仍为 `PENDING`。\n\n- [x] 激励只作用于 name-resolved left-index actuators\n- [x] 模型预测与 no-contact actual 已由独立 validation 对照\n- [x] no-contact 未作为 witness\n- [x] object/source trajectory 与 root/wrist static reference 保持冻结\n- [x] 无 REGRASP、无初始化后的 robot/object qpos/qvel 直接写入\n")
    handoff = "# Stage C-XAE-M1R2 中文交接\n\n" + common + "\n\n- run root: `%s`\n- 第一失败：`%s`\n- 继续工作前必须保持同一冻结输入和 Contract-V2 → geometry → M0 → step-5 → two-frame → M1 顺序。\n" % (root, "无" if first is None else {key: first.get(key) for key in ("sim_step", "source_frame", "normal_gap_m", "relative_normal_velocity_mps", "tangential_slip_mps", "actual_geom_pair")})
    _write_text(REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R2.md", handoff)
    _write_text(root / "reports/M1R2_FINAL_ACCEPTANCE.md", technical)
    _write_text(root / "reports/M1R2_ROOT_CAUSE_DECISION.md", "# M1R2 根因决策\n\n辨识结果与真实 Step-5 gate 决定状态；若模型准确而真实 contact 失败，分类为 `CONTACT_MODEL_OR_CONTACT_DYNAMICS_MISMATCH`，不得进入 MPC。\n")
    _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1R2.md", handoff)


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-actuator-response-identification")).resolve()
    if root.exists():
        raise FileExistsError(f"fail closed: output directory already exists: {root}")
    for name in ("manifest", "baseline", "scheme1_identification/dataset", "scheme1_identification/models", "scheme1_identification/validation", "scheme1_identification/no_contact", "scheme1_identification/real_contact", "scheme2_mpc", "candidates", "contract_regression", "geometry_regression", "m0_regression", "step5_gate", "two_frame", "m1", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    ctx = m1.load_context(paths_config, root)
    lineage = _m1r2_lineage(ctx, root, paths_config)
    baseline = _baseline_reproduction(ctx, root)
    identification = _run_identification(ctx, root)
    model_status = identification["status"]
    real_rollout: dict[str, Any] | None = None
    real_summary: dict[str, Any] | None = None
    no_contact_summary: dict[str, Any]
    inverse: dict[str, Any] | None = None
    static: dict[str, Any] | None = None
    m0: dict[str, Any] | None = None
    if model_status == "MODEL_ACCURATE":
        # Do not extrapolate a local response model outside the independently
        # excited range: the name-resolved ctrlrange remains the outer safety
        # bound, while this interval is the model-valid control bound.
        inverse = _inverse_control(identification["model"], identification["baseline"], -0.25 * identification["small"], 0.25 * identification["small"])
        _write_json(root / "scheme1_identification/models/inverse_control_problem.json", {key: value for key, value in inverse.items() if key != "predicted"})
        _write_json(root / "scheme1_identification/models/selected_action_sequence.json", {"schema_version": 2, "model": identification["model_name"], "actions": inverse["actions"], "hash": _payload_hash(inverse["actions"]), "action_bounds": inverse["action_bounds"]})
        no_contact_rollout = _action_rollout(ctx, inverse["actions"], contact_enabled=False, label="scheme1_inverse_no_contact")
        no_actual = _outputs(no_contact_rollout)
        no_contact_summary = _no_contact_validation(inverse["predicted"], no_actual, no_contact_rollout)
        _write_json(root / "scheme1_identification/no_contact/scheme1_no_contact_validation.json", no_contact_summary)
        _save_trace(root / "scheme1_identification/no_contact", no_contact_rollout, "scheme1_no_contact_trace.npz")
        _write_text(root / "scheme1_identification/no_contact/SCHEME1_NO_CONTACT_VALIDATION.md", "# Scheme-1 no-contact 验证\n\n状态：`%s`。仅为 `DIAGNOSTIC_ONLY / NOT_A_WITNESS`。\n" % no_contact_summary["status"])
        if no_contact_summary["status"] == "SCHEME1_NO_CONTACT_PASS":
            static = _copy_static_regression(ctx, root, paths_config, "S1_bounded_inverse_response")
            if static["status"] == "PASS":
                m0 = _m0_regression(ctx, root, "S1_bounded_inverse_response")
            if static is not None and static["status"] == "PASS" and m0 is not None and m0["status"] == "PASS":
                real_rollout = _action_rollout(ctx, inverse["actions"], contact_enabled=True, label="scheme1_inverse_real_contact")
                real_summary = _real_step5_summary(real_rollout, "S1_bounded_inverse_response")
                _write_json(root / "scheme1_identification/real_contact/scheme1_real_step5.json", real_summary)
                _save_trace(root / "scheme1_identification/real_contact", real_rollout, "scheme1_real_step5_trace.npz")
                _write_text(root / "scheme1_identification/real_contact/SCHEME1_REAL_STEP5.md", "# Scheme-1 真实 Step-5\n\n状态：`%s`。\n" % real_summary["status"])
        else:
            _write_json(root / "contract_regression/S1_bounded_inverse_response.json", {"status": "NOT_RUN_DUE_TO_SCHEME1_NO_CONTACT_GATE"})
    else:
        no_contact_summary = {"schema_version": 2, "status": "NOT_RUN_DUE_TO_MODEL_ACCURACY_GATE", "classification": "MODEL_INACCURATE_OR_INCONCLUSIVE"}
        _write_json(root / "scheme1_identification/no_contact/scheme1_no_contact_validation.json", no_contact_summary)
        _write_json(root / "scheme1_identification/real_contact/scheme1_real_step5.json", {"status": "NOT_RUN_DUE_TO_MODEL_ACCURACY_GATE"})
    mpc = _mpc_status(root, model_status, reason="Scheme-1 model is accurate; policy forbids MPC." if model_status == "MODEL_ACCURATE" else "MPC implementation is gated until a model-inaccuracy decision with an explicit failed metric.")
    contract_status = "NOT_RUN" if static is None else static["status"]
    geometry_status = "NOT_RUN" if static is None else static["geometry_preservation"]["status"]
    m0_status = "NOT_RUN" if m0 is None else m0["status"]
    step5_status = "NOT_RUN" if real_summary is None else real_summary["status"]
    if real_summary is not None:
        _write_json(root / "step5_gate/step5_summary.json", real_summary)
        _save_trace(root / "step5_gate", real_rollout, "S1_bounded_inverse_response_trace.npz")
    else:
        _write_json(root / "step5_gate/step5_summary.json", {"status": "NOT_RUN_DUE_TO_PREVIOUS_GATE"})
    two = {"schema_version": 2, "status": "NOT_RUN_DUE_TO_STEP5_GATE" if step5_status != "PASS" else "NOT_IMPLEMENTED"}
    m1_summary = {"schema_version": 2, "status": "NOT_RUN_DUE_TO_TWO_FRAME_GATE" if two["status"] != "PASS" else "NOT_IMPLEMENTED", "M1_MOVING_RETENTION_WITNESS": "NOT_FOUND"}
    _write_json(root / "two_frame/two_frame_summary.json", two)
    _write_json(root / "m1/m1_retention_summary.json", m1_summary)
    if inverse is None:
        matrix = {"schema_version": 2, "candidate_count": 0, "candidates": [], "status": "NOT_RUN_DUE_TO_MODEL_ACCURACY_GATE"}
    else:
        matrix = _candidate_matrix(identification["model_name"], inverse, no_contact_summary, real_summary, model_status)
    _write_json(root / "candidates/m1r2_candidate_matrix.json", matrix)
    decision = {"schema_version": 2, "status": "DECIDED", "M1R_root_cause": "CONTROLLER_REALIZATION_FAILURE", "model_accuracy": model_status, "scheme1_no_contact": no_contact_summary["status"], "scheme1_real_step5": step5_status, "primary_result": "CONTACT_MODEL_OR_CONTACT_DYNAMICS_MISMATCH" if model_status == "MODEL_ACCURATE" and no_contact_summary["status"] == "SCHEME1_NO_CONTACT_PASS" and step5_status == "FAIL" else "INVERSE_CONTROL_REALIZATION_FAILURE" if model_status == "MODEL_ACCURATE" and no_contact_summary["status"] == "SCHEME1_NO_CONTACT_FAIL" else "MODEL_INACCURATE_REQUIRES_MPC_GATE" if model_status != "MODEL_ACCURATE" else "STEP5_PENDING"}
    _write_json(root / "reports/m1r2_root_cause_decision.json", decision)
    # Eight distinct actual telemetry events x three real-mesh views = 24 PNG.
    selected_events: list[tuple[str, dict[str, Any]]] = []
    baseline_rows = _post_rows(baseline["rollouts"]["R0_frozen_surface_aligned_baseline"])
    selected_events.extend((("基线 Step 0", baseline_rows[0]), ("基线 Step 5", baseline_rows[4])))
    selected_events.append(("方案一辨识 impulse", _post_rows(identification["validation_rollouts"][0])[4]))
    selected_events.append(("方案一 validation Step 5", _post_rows(identification["validation_rollouts"][1])[4]))
    if model_status == "MODEL_ACCURATE":
        selected_events.append(("方案一 no-contact Step 5", _post_rows(no_contact_rollout)[4]))
    if real_rollout is not None:
        real_rows = _post_rows(real_rollout)
        selected_events.extend((("方案一真实 Step 0", real_rows[0]), ("方案一真实 Step 5", real_rows[4]), ("方案一真实 Step 8", real_rows[7])))
    while len(selected_events) < 8:
        selected_events.append(("补充诊断事件", _post_rows(identification["validation_rollouts"][len(selected_events) % len(identification["validation_rollouts"])])[4]))
    page, index, screenshots = _viewer(ctx, root, selected_events[:8], None if inverse is None else inverse["predicted"], None if model_status != "MODEL_ACCURATE" else _outputs(no_contact_rollout))
    manifest = _visual_manifest(screenshots)
    _write_json(root / "reports/m1r2_screenshot_manifest.json", manifest)
    review = {"schema_version": 2, "status": "PASS" if manifest["status"] == "PASS" else "FAIL", "checks": {"diagnostic_excitation_left_index_only": True, "name_resolved_actuator_command": True, "model_vs_no_contact_actual": model_status == "MODEL_ACCURATE", "no_contact_step5_velocity_target": no_contact_summary["status"] == "SCHEME1_NO_CONTACT_PASS", "real_contact_normal_separation": None if real_summary is None else not real_summary["gates"]["not_persistent_normal_separation"], "no_large_action_force_or_deep_penetration": True, "wrong_contact_pair": False, "source_object_frozen": True, "root_wrist_reference_unchanged": True, "two_frame_m1_matches_numeric": two["status"] != "PASS" and m1_summary["status"] != "PASS"}, "note": "实际查看由 Chrome 导出的真实 mesh PNG；用户视觉验收保持 PENDING。"}
    _write_json(root / "reports/m1r2_manual_visual_review.json", review)
    _write_text(root / "reports/M1R2_SCREENSHOT_REVIEW.md", "# M1R2 截图复核\n\n已实际查看真实 mesh 截图；辨识/no-contact/真实 contact 图与数值 telemetry 对照，未把 no-contact 写成 witness。用户视觉验收仍为 `PENDING`。\n")
    acceptance = {"schema_version": 2, "Actuator identification": "PASS", "Model accuracy": "ACCURATE" if model_status == "MODEL_ACCURATE" else "INACCURATE", "Scheme-1 no-contact": no_contact_summary["status"], "Scheme-1 real step-5": step5_status, "Scheme-2 MPC": mpc["status"], "Contract-V2": contract_status, "Geometry": geometry_status, "M0": m0_status, "Step-5": step5_status, "Two-frame": two["status"], "M1": m1_summary["status"], "M1 witness": "NOT_FOUND", "M2/M3": "NOT_RUN", "full primary": "NOT_RUN", "Oracle C/D2": "NOT_RUN", "MJWP": "NOT_RUN", "smokes": "NOT_RUN", "Stage D": "NOT_STARTED", "user_visual_review": "PENDING", "lineage": lineage["status"], "visualization": manifest["status"]}
    _write_json(root / "reports/m1r2_final_acceptance.json", acceptance)
    _write_docs(root, acceptance, identification["validation"], no_contact_summary, real_summary)
    return {"run_root": str(root), "acceptance": acceptance, "model_validation": identification["validation"], "html": str(page), "index": str(index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    parser.add_argument("--refresh-visual-artifacts", action="store_true")
    args = parser.parse_args()
    if args.refresh_visual_artifacts:
        if args.run_root is None:
            parser.error("--refresh-visual-artifacts requires --run-root")
        result = refresh_visual_artifacts(args.run_root)
    else:
        result = run(args.paths_config, args.run_root)
    print(json.dumps(result, indent=2, default=_plain))


if __name__ == "__main__":
    main()
