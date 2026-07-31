"""Stage C-V2R1 real-MuJoCo causal diagnostics for corrected V2 contact.

This module is intentionally diagnosis-only.  It neither changes C-XA data nor
reuses the historical ``stage_c_v2_dynamic`` directory.  The four oracles
share the exact frozen source cadence and differ in only the mechanism being
isolated, so their decision is evidence rather than a controller-tuning hunch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro
import yaml

from spider.datasets.paths import load_project_paths
from spider.tools import grab_stage_c_v2_dynamic as dynamic
from spider.tools.grab_stage_c import (
    _finite_data,
    _hand_geom_ids,
    _object_tracking_error,
    _preflight_object_ids,
    _set_object_mocap_reference,
    _site_ids,
)
from spider.tools.grab_stage_c_contact_reassignment import evaluate_v2_depenetrated


PRIMARY = dynamic.PRIMARY
CONFIG_PATH = Path("configs/project/grab_wuji_stage_c_v2r.yaml")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("corrected_namespace") != "stage_c_contract_v2_cxa" or config.get("corrected_level") != 1:
        raise RuntimeError("V2R may only consume immutable C-XA Level 1")
    if config["timing"].get("variant") != "V2_ORIGINAL_TIMING" or float(config["timing"].get("scale", 0.0)) != 1.0:
        raise RuntimeError("V2R1 diagnoses original 120-Hz timing only")
    return config


def _root(paths, sequence_id: str) -> Path:
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R1 is primary-only; smoke pilots are forbidden before primary gates pass")
    return dynamic._stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c_v2r"


def _configure_model(model: mujoco.MjModel, config: dict[str, Any], *, oracle_c_kinematic: bool = False) -> dict[str, Any]:
    physics = config["physics"]
    timestep = float(physics["sim_timestep_s"])
    solref = float(physics["oracle_c_kinematic_mocap_solref_s"] if oracle_c_kinematic else physics["object_mocap_solref_s"])
    if timestep <= 0.0 or solref <= 0.0:
        raise ValueError("V2R physics timestep and mocap weld solref must be positive")
    model.opt.timestep = timestep
    model.eq_solref[:, 0] = solref
    return {
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "sim_timestep_s": timestep,
        "object_mocap_solref_s": solref,
        "robot_reference_lead_source_frames": int(physics["robot_reference_lead_source_frames"]),
        "source_interval_scheduler": physics["source_interval_scheduler"],
        "oracle_c_kinematic_mocap": oracle_c_kinematic,
    }


def _load_inputs(paths) -> tuple[dict[str, Path], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = dynamic._inputs(paths, PRIMARY)
    physics = dynamic._physics(paths, PRIMARY)
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        reference = np.asarray(archive["qpos"], dtype=np.float64)
        reference_qvel = np.asarray(archive["qvel"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool)
        source_anchors = np.asarray(archive["source_anchors"], dtype=np.float64)
    if reference.shape != (414, 64) or expected.shape != (414, 10):
        raise RuntimeError("C-XA primary input schema unexpectedly changed")
    return inputs, physics, reference, reference_qvel, source_frames, expected, source_anchors


def _robot_ranges(model: mujoco.MjModel) -> np.ndarray:
    ranges = np.asarray(model.jnt_range[:52, 1] - model.jnt_range[:52, 0], dtype=np.float64)
    ranges[[0, 1, 2, 26, 27, 28]] = 4.0
    return np.maximum(ranges, 1e-9)


def _joint_margin(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    margin = np.full((len(qpos), 52), np.inf, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for joint in range(model.njnt):
        if not bool(model.jnt_limited[joint]):
            continue
        address = int(model.jnt_qposadr[joint])
        if address >= 52:
            continue
        lower, upper = model.jnt_range[joint]
        current = np.minimum(qpos[:, address] - lower, upper - qpos[:, address]) / max(upper - lower, 1e-9)
        margin[:, address] = current
        records.append({"joint": int(joint), "qpos_address": address, "minimum_fraction": float(current.min())})
    return margin, records


def _derivatives(qpos: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    velocity = np.diff(qpos[:, :52], axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    return {"qvel_fd": velocity, "qacc_fd": acceleration, "jerk_fd": jerk}


def _contact_ids(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    hand = set(_hand_geom_ids(model, 2))
    objects = {
        index
        for index in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("right_object_")
        and model.geom_group[index] == 3
    }
    if not hand or not objects:
        raise RuntimeError("V2R requires hand and right-object collision geoms")
    return hand, objects


def _is_hand_object_pair(geom1: int, geom2: int, hand: set[int], objects: set[int]) -> bool:
    return (geom1 in hand and geom2 in objects) or (geom2 in hand and geom1 in objects)


def _disable_hand_object_explicit_pairs(
    model: mujoco.MjModel, hand: set[int], objects: set[int]
) -> dict[str, Any]:
    """Disable only explicit hand-object pairs in this in-memory oracle model.

    The scene deliberately uses explicit ``<pair>`` declarations because both
    the hand and object collision geoms otherwise have disabled masks.  Those
    pairs bypass MuJoCo's contact-filter callback, so Oracle D must neutralize
    just their runtime pair entries.  Mapping them to the same floor geom is a
    no-contact sentinel: MuJoCo skips same-geom pairs.  The source XML and all
    self-collision / floor-object pairs remain unchanged.
    """
    sentinel = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    if sentinel < 0:
        raise RuntimeError("Oracle D requires the floor geom as its no-contact pair sentinel")
    indices = [
        index
        for index, (geom1, geom2) in enumerate(zip(model.pair_geom1, model.pair_geom2, strict=True))
        if _is_hand_object_pair(int(geom1), int(geom2), hand, objects)
    ]
    if not indices:
        raise RuntimeError("Oracle D found no explicit hand-object pairs to disable")
    original_pairs = [
        {
            "pair_index": index,
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[index])),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[index])),
        }
        for index in indices
    ]
    model.pair_geom1[indices] = sentinel
    model.pair_geom2[indices] = sentinel
    return {
        "explicit_pair_count": len(indices),
        "pair_indices": indices,
        "original_pairs": original_pairs,
        "replacement": {"geom_id": sentinel, "geom_name": "floor", "reason": "same-geom pairs are skipped"},
    }


def _hand_object_pair_indices(model: mujoco.MjModel, hand: set[int], objects: set[int]) -> list[int]:
    """Return the explicit hand-object contact pairs under the scene policy."""
    indices = [
        index
        for index, (geom1, geom2) in enumerate(zip(model.pair_geom1, model.pair_geom2, strict=True))
        if _is_hand_object_pair(int(geom1), int(geom2), hand, objects)
    ]
    if not indices:
        raise RuntimeError("V2R contact-dynamics profile requires explicit hand-object pairs")
    return indices


def _contact_pair_profile(model: mujoco.MjModel, indices: list[int]) -> dict[str, Any]:
    """Serialize exactly the mutable pairwise contact settings for audit."""
    return {
        "pair_indices": indices,
        "pair_count": len(indices),
        "solref": np.asarray(model.pair_solref[indices], dtype=np.float64)[0].tolist(),
        "solimp": np.asarray(model.pair_solimp[indices], dtype=np.float64)[0].tolist(),
        "margin_m": float(np.asarray(model.pair_margin[indices], dtype=np.float64)[0]),
        "gap_m": float(np.asarray(model.pair_gap[indices], dtype=np.float64)[0]),
        "friction": np.asarray(model.pair_friction[indices], dtype=np.float64)[0].tolist(),
    }


def _apply_contact_dynamics_profile(
    model: mujoco.MjModel, candidate: dict[str, Any] | None, hand: set[int], objects: set[int]
) -> dict[str, Any]:
    """Apply a bounded R2 Branch-D contact profile to explicit hand-object pairs.

    This changes only the in-memory contact pair parameters used by the
    diagnostic rollout.  Robot controls, source object targets, the C-XA
    reference, and all self/floor contacts are deliberately untouched.
    """
    indices = _hand_object_pair_indices(model, hand, objects)
    baseline = _contact_pair_profile(model, indices)
    if candidate is None:
        return {"candidate_id": "baseline", "baseline": baseline, "applied": baseline}
    required = {"candidate_id", "solref", "solimp", "margin_m", "gap_m", "friction"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"contact-dynamics candidate is missing keys: {missing}")
    solref = np.asarray(candidate["solref"], dtype=np.float64)
    solimp = np.asarray(candidate["solimp"], dtype=np.float64)
    friction = np.asarray(candidate["friction"], dtype=np.float64)
    if solref.shape != (2,) or solimp.shape != (5,) or friction.shape != (5,):
        raise ValueError("contact-dynamics candidate has an invalid MuJoCo pair parameter shape")
    if solref[0] <= 0.0 or solref[1] <= 0.0 or not (0.0 < solimp[0] <= solimp[1] < 1.0):
        raise ValueError("contact-dynamics candidate has invalid solref/solimp impedance")
    if float(candidate["margin_m"]) < 0.0 or float(candidate["gap_m"]) < 0.0:
        raise ValueError("contact-dynamics candidates may not hide penetration with negative margin or gap")
    if np.any(friction < 0.0):
        raise ValueError("contact-dynamics friction must be non-negative")
    model.pair_solref[indices] = solref
    model.pair_solimp[indices] = solimp
    model.pair_margin[indices] = float(candidate["margin_m"])
    model.pair_gap[indices] = float(candidate["gap_m"])
    model.pair_friction[indices] = friction
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "baseline": baseline,
        "applied": _contact_pair_profile(model, indices),
        "scope": "explicit hand-object pairs only; self-collision and floor-object pairs unchanged",
    }


def _base_contact_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    reference: np.ndarray,
    source_frames: np.ndarray,
    source_anchors: np.ndarray,
    physics: dict[str, Any],
    collision_max: float,
    tracking_ok: bool,
    smoothness_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    positions, tracking_error, _flat, tracking_by_side = dynamic._dynamic_robot_tracking(model, qpos, reference, source_frames)
    tips = positions[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    anchor_distance = np.linalg.norm(tips - source_anchors, axis=2)
    observed = anchor_distance <= 0.015
    contact = {
        "high_confidence_recall": float(np.count_nonzero(observed) / max(1, observed.size)),
        "expected_records": int(observed.size),
        "false_contact_frames": 0,
        "anchor_contract": "immutable C-XA source anchors; diagnostic only",
    }
    visual = dynamic._dynamic_visual_penetration(model, qpos, physics)
    payload = {
        "schema_version": 1,
        "contact": contact,
        "visual_penetration": visual,
        "collision": {"after_max_m": float(collision_max)},
        "gates": {"tracking": bool(tracking_ok), "smoothness": bool(smoothness_ok)},
    }
    return payload, tracking_by_side, tracking_error, positions


def _evaluate_patch(
    paths_config: str,
    root: Path,
    name: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    source_frames: np.ndarray,
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    trajectory = root / f"{name}_trajectory.npz"
    metrics = root / f"{name}.json"
    _write_npz(trajectory, qpos=qpos, qvel=qvel, source_frame_indices=source_frames)
    _write_json(metrics, base_metrics)
    inputs = dynamic._inputs(load_project_paths(paths_config), PRIMARY)
    evaluate_v2_depenetrated(
        paths_config,
        PRIMARY,
        1,
        str(trajectory),
        str(metrics),
        str(inputs["targets"]),
        "stage_c_contract_v2_cxa",
        refresh_base_metrics=False,
    )
    return json.loads(metrics.read_text(encoding="utf-8"))


def _patch_gates(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    values = metrics["task_equivalent_contact_v2"]
    oracle = config["oracle"]
    return {
        "patch_coverage": values["patch_coverage"] >= float(oracle["patch_coverage"]),
        "functional_role_recall": values["functional_role_recall"] >= float(oracle["functional_role_recall"]),
        "patch_distance_p95": values["surface_patch_distance_p95_m"] <= float(oracle["patch_distance_m"]),
        "normal_alignment": values["normal_cosine_median"] >= float(oracle["normal_cosine"]),
    }


def _object_summary(position: np.ndarray, rotation: np.ndarray, source_frames: np.ndarray) -> tuple[dict[str, Any], bool]:
    rows: dict[str, Any] = {}
    for index, side in enumerate(("right", "left")):
        rows[side] = {
            "position_rmse_m": float(np.sqrt(np.mean(position[:, index] ** 2))),
            "position_max_m": float(position[:, index].max(initial=0.0)),
            "rotation_mean_rad": float(rotation[:, index].mean()),
            "rotation_max_rad": float(rotation[:, index].max(initial=0.0)),
            "worst_source_frame": int(source_frames[int(np.argmax(rotation[:, index]))]),
        }
    config = _config()["oracle"]
    passed = all(
        row["position_max_m"] <= float(config["object_position_max_m"])
        and row["rotation_max_rad"] <= float(config["object_rotation_max_rad"])
        for row in rows.values()
    )
    return rows, passed


def _tracking_gate(tracking: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(
        side["wrist_rmse_m"] <= float(config["oracle"]["wrist_rmse_m"])
        and all(value["rmse_m"] <= float(config["oracle"]["fingertip_rmse_m"]) for value in side["fingertips"].values())
        for side in tracking.values()
    )


def _apply_controller_profile(model: mujoco.MjModel, candidate: dict[str, Any] | None) -> dict[str, Any]:
    """Change only robot position-actuator gains/limits for a bounded R2 probe."""
    selected = {
        "lead_source_frames": int(candidate["lead_source_frames"]) if candidate else None,
        "kp_scale": float(candidate["kp_scale"]) if candidate else 1.0,
        "kv_scale": float(candidate["kv_scale"]) if candidate else 1.0,
        "force_limit_scale": float(candidate["force_limit_scale"]) if candidate else 1.0,
        "joint_kp_overrides": {int(index): float(scale) for index, scale in (candidate or {}).get("joint_kp_overrides", {}).items()},
    }
    if selected["lead_source_frames"] is not None and selected["lead_source_frames"] < 0:
        raise ValueError("reference lead must be non-negative")
    if not 0.5 <= selected["kp_scale"] <= 1.5 or not 0.5 <= selected["kv_scale"] <= 1.5 or not 0.5 <= selected["force_limit_scale"] <= 1.25:
        raise ValueError("controller candidate exceeds the frozen bounded R2 search range")
    model.actuator_gainprm[:52, 0] *= selected["kp_scale"]
    model.actuator_biasprm[:52, 1] *= selected["kp_scale"]
    model.actuator_biasprm[:52, 2] *= selected["kv_scale"]
    for index, scale in selected["joint_kp_overrides"].items():
        if not 0 <= index < 52 or not 0.5 <= scale <= 1.5:
            raise ValueError("joint-specific controller override is outside the frozen R2 search range")
        model.actuator_gainprm[index, 0] *= scale
        model.actuator_biasprm[index, 1] *= scale
    limited = np.flatnonzero(model.actuator_forcelimited[:52])
    model.actuator_forcerange[limited] *= selected["force_limit_scale"]
    return selected


def _contact_target_correction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    expected: np.ndarray,
    anchors: np.ndarray,
    frame: int,
    candidate: dict[str, Any],
) -> np.ndarray:
    """Compute a bounded robot-only Jacobian correction for active contacts.

    This is a controller target correction, not a state or reference edit.  It
    follows the audited MJWP contact-feedback convention: each hand is solved
    independently from current site Jacobians, the immutable source anchor is
    the target, and the result is clipped before being added to ``data.ctrl``.
    Object mocap and generalized object qpos are never touched here.
    """
    gain = float(candidate.get("contact_ik_gain", 0.0))
    if gain <= 0.0:
        return np.zeros(52, dtype=np.float64)
    damping = float(candidate.get("contact_ik_damping", 0.002))
    max_error = float(candidate.get("contact_ik_max_anchor_error_m", 0.03))
    if damping < 0.0 or max_error <= 0.0:
        raise ValueError("contact IK damping and anchor guard must be non-negative/positive")
    strategy = str(candidate.get("contact_ik_strategy", "all_active"))
    if strategy not in {"all_active", "first_active"}:
        raise ValueError("contact IK strategy must be all_active or first_active")
    finger_clip = float(candidate.get("contact_ik_finger_clip_rad", 0.12))
    wrist_translation_clip = float(candidate.get("contact_ik_wrist_translation_clip_m", 0.003))
    wrist_rotation_clip = float(candidate.get("contact_ik_wrist_rotation_clip_rad", 0.03))
    clips = np.full(26, finger_clip, dtype=np.float64)
    clips[:3] = wrist_translation_clip
    clips[3:6] = wrist_rotation_clip
    if np.any(clips < 0.0):
        raise ValueError("contact IK clips must be non-negative")
    site_names = _site_ids(model)
    contact_site_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    correction = np.zeros(52, dtype=np.float64)
    active = np.flatnonzero(np.asarray(expected[frame], dtype=bool))
    for side, contact_indices, actuator_slice, site_offset in (
        ("right", [index for index in active if index < 5], slice(0, 26), 0),
        ("left", [index - 5 for index in active if index >= 5], slice(26, 52), 6),
    ):
        if strategy == "first_active" and contact_indices:
            contact_indices = contact_indices[:1]
        rows: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        for local_index in contact_indices:
            anchor_index = local_index if side == "right" else local_index + 5
            site_id = site_names[contact_site_indices[site_offset + local_index]]
            error = np.asarray(anchors[frame, anchor_index], dtype=np.float64) - np.asarray(data.site_xpos[site_id], dtype=np.float64)
            if not np.isfinite(error).all() or np.linalg.norm(error) > max_error:
                continue
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            rows.append(jacp[:, actuator_slice])
            errors.append(error)
        if not rows:
            continue
        jacobian = np.concatenate(rows, axis=0)
        error = np.concatenate(errors, axis=0)
        lhs = jacobian.T @ jacobian + damping * np.eye(26)
        delta = np.linalg.solve(lhs, jacobian.T @ error) * gain
        correction[actuator_slice] = np.clip(delta, -clips, clips)
    return correction


def _inverse_dynamics_feedforward(
    model: mujoco.MjModel, reference: np.ndarray, reference_qvel: np.ndarray, config: dict[str, Any], candidate: dict[str, Any]
) -> np.ndarray:
    """Convert a bounded inverse-dynamics torque seed into position controls."""
    repair = config["controller_feedforward_repair"]
    fps = float(config["timing"]["source_fps"])
    acceleration = np.zeros((len(reference), model.nv), dtype=np.float64)
    acceleration[1:-1, :52] = (reference[2:, :52] - 2.0 * reference[1:-1, :52] + reference[:-2, :52]) * fps * fps
    acceleration[0, :52] = acceleration[1, :52]
    acceleration[-1, :52] = acceleration[-2, :52]
    data = mujoco.MjData(model)
    offset = np.empty((len(reference), 52), dtype=np.float64)
    gain = np.maximum(np.abs(model.actuator_gainprm[:52, 0]), 1e-6)
    limits = np.full(52, float(repair["finger_ctrl_offset_limit_rad"]), dtype=np.float64)
    limits[[0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31]] = float(repair["wrist_ctrl_offset_limit_rad"])
    for frame in range(len(reference)):
        data.qpos[:] = reference[frame]
        data.qvel[:] = reference_qvel[frame]
        data.qacc[:] = acceleration[frame]
        mujoco.mj_inverse(model, data)
        offset[frame] = np.clip(float(candidate["inverse_dynamics_scale"]) * data.qfrc_inverse[:52] / gain, -limits, limits)
    # The offset is a control seed, so it must be temporally continuous too.
    offset[1:] = np.clip(offset[1:], offset[:-1] - limits, offset[:-1] + limits)
    return offset


def _run_dynamic_oracle(
    paths_config: str,
    name: str,
    *,
    perfect_hand: bool = False,
    kinematic_object: bool = False,
    no_hand_object_contact: bool = False,
    controller_candidate: dict[str, Any] | None = None,
    contact_candidate: dict[str, Any] | None = None,
    object_guidance_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one A-D isolation experiment without altering frozen inputs."""
    paths = load_project_paths(paths_config)
    config = _config()
    root = _root(paths, PRIMARY)
    _inputs, physics, reference, reference_qvel, source_frames, expected, source_anchors = _load_inputs(paths)
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    profile = _configure_model(model, config, oracle_c_kinematic=kinematic_object)
    object_guidance_profile = object_guidance_candidate or {
        "candidate_id": "G0_current_baseline",
        "pre_contact_solref_s": float(profile["object_mocap_solref_s"]),
        "acquisition_solref_s": float(profile["object_mocap_solref_s"]),
        "retention_solref_s": float(profile["object_mocap_solref_s"]),
        "release_solref_s": float(profile["object_mocap_solref_s"]),
    }
    for key in ("pre_contact_solref_s", "acquisition_solref_s", "retention_solref_s", "release_solref_s"):
        if float(object_guidance_profile[key]) <= 0.0:
            raise ValueError(f"object guidance solref must be positive: {key}")
    controller_profile = _apply_controller_profile(model, controller_candidate)
    hand, objects = _contact_ids(model)
    contact_profile = _apply_contact_dynamics_profile(model, contact_candidate, hand, objects)
    collision_policy: dict[str, Any] = {"hand_object_contact": "enabled", "self_collision": "unchanged"}
    previous_contact_filter = None
    if no_hand_object_contact:
        # Explicit <pair> declarations bypass MuJoCo's callback.  Neutralize
        # those in-memory pair entries first, then retain the callback for any
        # non-explicit pair.  Neither operation mutates the source XML.
        explicit_pair_policy = _disable_hand_object_explicit_pairs(model, hand, objects)
        previous_contact_filter = mujoco.get_mjcb_contactfilter()
        mujoco.set_mjcb_contactfilter(
            lambda _model, _data, geom1, geom2: int(_is_hand_object_pair(int(geom1), int(geom2), hand, objects))
        )
        collision_policy["hand_object_contact"] = "disabled_for_oracle_d_only"
        collision_policy["implementation"] = "in-memory explicit-pair floor-sentinel remap plus mjcb_contactfilter; self collision unchanged"
        collision_policy["explicit_pair_policy"] = explicit_pair_policy
    bodies, mocap = _preflight_object_ids(model)
    feedforward_enabled = bool((controller_candidate or {}).get("inverse_dynamics_scale", 0.0))
    feedforward = _inverse_dynamics_feedforward(model, reference, reference_qvel, config, controller_candidate or {}) if feedforward_enabled else np.zeros((len(reference), 52), dtype=np.float64)
    feedback_gain = float((controller_candidate or {}).get("state_feedback_gain", 0.0))
    contact_ik_enabled = float((controller_candidate or {}).get("contact_ik_gain", 0.0)) > 0.0
    repair = config.get("controller_feedforward_repair", {})
    feedback_limits = np.full(52, float(repair.get("finger_state_feedback_limit_rad", 0.0)), dtype=np.float64)
    feedback_limits[[0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31]] = float(repair.get("wrist_state_feedback_limit_rad", 0.0))
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = reference_qvel[0]
    _set_object_mocap_reference(data, reference[0], mocap)
    lead = int(controller_profile["lead_source_frames"] if controller_profile["lead_source_frames"] is not None else profile["robot_reference_lead_source_frames"])
    data.ctrl[:52] = reference[min(lead, len(reference) - 1), :52] + feedforward[min(lead, len(reference) - 1)]
    data.ctrl[52:] = 0.0
    mujoco.mj_forward(model, data)
    qpos = np.empty_like(reference)
    qvel = np.empty_like(reference_qvel)
    ctrl = np.empty((len(reference), model.nu), dtype=np.float64)
    position = np.empty((len(reference), 2), dtype=np.float64)
    rotation = np.empty((len(reference), 2), dtype=np.float64)
    finite = np.empty(len(reference), dtype=np.uint8)
    depth = np.empty(len(reference), dtype=np.float64)
    force = np.empty(len(reference), dtype=np.float64)
    contact_count = np.empty(len(reference), dtype=np.int32)
    hand_object_count = np.empty(len(reference), dtype=np.int32)
    clamps_impulse: list[float] = []
    clamps_energy: list[float] = []
    steps: dict[str, list[np.ndarray]] = {key: [] for key in (
        "frame_index", "substep_index", "sim_time", "qpos", "qvel", "qacc", "ctrl", "actuator_force",
        "contact_count", "hand_object_contact_count", "contact_depth_m", "contact_force_n", "object_position", "object_orientation",
        "object_linear_velocity", "object_angular_velocity", "clamp_impulse_norm", "clamp_energy_delta",
    )}
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    target_time = 0.0
    interval_counts: list[int] = []
    try:
        for frame in range(len(reference)):
            substep = 0
            active_contact = bool(np.any(expected[frame]))
            previous_active = bool(np.any(expected[max(0, frame - 1)]))
            if not active_contact and not previous_active:
                guidance_solref = float(object_guidance_profile["pre_contact_solref_s"])
            elif active_contact and not previous_active:
                guidance_solref = float(object_guidance_profile["acquisition_solref_s"])
            elif active_contact:
                guidance_solref = float(object_guidance_profile["retention_solref_s"])
            else:
                guidance_solref = float(object_guidance_profile["release_solref_s"])
            model.eq_solref[:, 0] = guidance_solref
            if frame:
                _set_object_mocap_reference(data, reference[frame], mocap)
                control_index = min(frame + lead, len(reference) - 1)
                base_ctrl = reference[control_index, :52] + feedforward[control_index]
                data.ctrl[:52] = base_ctrl
                data.ctrl[52:] = 0.0
                target_time += 1.0 / float(config["timing"]["source_fps"])
                previous = reference[frame - 1, :52]
                while data.time + 0.5 * model.opt.timestep < target_time:
                    if contact_ik_enabled:
                        data.ctrl[:52] = base_ctrl + _contact_target_correction(
                            model, data, expected, source_anchors, frame, controller_candidate or {}
                        )
                    if feedback_gain:
                        correction = np.clip(feedback_gain * (reference[control_index, :52] - data.qpos[:52]), -feedback_limits, feedback_limits)
                        data.ctrl[:52] = base_ctrl + correction
                    before_qvel = data.qvel[:52].copy()
                    mujoco.mj_step(model, data)
                    clamp_impulse = 0.0
                    clamp_energy = 0.0
                    if perfect_hand:
                        alpha = min(1.0, (substep + 1) * model.opt.timestep * float(config["timing"]["source_fps"]))
                        desired = previous * (1.0 - alpha) + reference[frame, :52] * alpha
                        desired_velocity = (reference[frame, :52] - previous) * float(config["timing"]["source_fps"])
                        delta_velocity = desired_velocity - data.qvel[:52]
                        clamp_impulse = float(np.linalg.norm(delta_velocity))
                        clamp_energy = float(0.5 * (np.dot(desired_velocity, desired_velocity) - np.dot(data.qvel[:52], data.qvel[:52])))
                        data.qpos[:52] = desired
                        data.qvel[:52] = desired_velocity
                        mujoco.mj_forward(model, data)
                    count, collision_depth, contact_force, pairs = dynamic._contact_summary(model, data, hand, objects)
                    if no_hand_object_contact and pairs:
                        raise RuntimeError("Oracle D integrity failure: hand-object contact survived the pair exclusion")
                    clamps_impulse.append(clamp_impulse)
                    clamps_energy.append(clamp_energy)
                    steps["frame_index"].append(np.asarray(frame, dtype=np.int32))
                    steps["substep_index"].append(np.asarray(substep, dtype=np.int32))
                    steps["sim_time"].append(np.asarray(data.time))
                    steps["qpos"].append(data.qpos.copy())
                    steps["qvel"].append(data.qvel.copy())
                    steps["qacc"].append(data.qacc.copy())
                    steps["ctrl"].append(data.ctrl.copy())
                    steps["actuator_force"].append(data.actuator_force.copy())
                    steps["contact_count"].append(np.asarray(count, dtype=np.int32))
                    steps["hand_object_contact_count"].append(np.asarray(len(pairs), dtype=np.int32))
                    steps["contact_depth_m"].append(np.asarray(collision_depth))
                    steps["contact_force_n"].append(np.asarray(contact_force))
                    steps["object_position"].append(np.stack([data.xpos[bodies["right"]], data.xpos[bodies["left"]]]))
                    steps["object_orientation"].append(np.stack([data.xmat[bodies["right"]], data.xmat[bodies["left"]]]))
                    steps["object_linear_velocity"].append(np.stack([data.qvel[52:55], data.qvel[58:61]]))
                    steps["object_angular_velocity"].append(np.stack([data.qvel[55:58], data.qvel[61:64]]))
                    steps["clamp_impulse_norm"].append(np.asarray(clamp_impulse))
                    steps["clamp_energy_delta"].append(np.asarray(clamp_energy))
                    substep += 1
                interval_counts.append(substep)
            count, collision_depth, contact_force, pairs = dynamic._contact_summary(model, data, hand, objects)
            if no_hand_object_contact and pairs:
                raise RuntimeError("Oracle D integrity failure: hand-object contact survived the pair exclusion")
            qpos[frame] = data.qpos
            qvel[frame] = data.qvel
            ctrl[frame] = data.ctrl
            position[frame], rotation[frame] = _object_tracking_error(data, reference[frame], bodies)
            finite[frame] = _finite_data(data)
            depth[frame] = collision_depth
            force[frame] = contact_force
            contact_count[frame] = count
            hand_object_count[frame] = len(pairs)
    finally:
        mujoco.set_mju_user_warning(old_warning)
        if no_hand_object_contact:
            mujoco.set_mjcb_contactfilter(previous_contact_filter)
    arrays = {key: np.stack(value) for key, value in steps.items()}
    trace_name = {"oracle_b": "oracle_b_trace.npz", "oracle_c": "oracle_c_trace.npz", "oracle_d": "oracle_d_trace.npz"}.get(name, f"{name}_trace.npz")
    _write_npz(root / trace_name, **arrays)
    ranges = _robot_ranges(model)
    normalized_delta = float(np.max(np.abs(np.diff(qpos[:, :52], axis=0)) / ranges))
    joint_margin, margin_records = _joint_margin(model, qpos)
    tracking_payload, tracking_by_side, tracking_error, _positions = _base_contact_metrics(
        model, qpos, reference, source_frames, source_anchors, physics, float(depth.max(initial=0.0)), False, normalized_delta <= float(config["oracle"]["max_normalized_joint_delta"])
    )
    tracking_ok = _tracking_gate(tracking_by_side, config)
    tracking_payload["gates"]["tracking"] = tracking_ok
    if no_hand_object_contact:
        patch = {"status": "NOT_RUN", "reason": "Oracle D disables hand-object contact by design"}
    else:
        patch_name = {"oracle_b": "oracle_b_perfect_hand_dynamic_object", "oracle_c": "oracle_c_dynamic_hand_kinematic_object"}.get(name, name)
        patch_report = _evaluate_patch(paths_config, root, patch_name, qpos, qvel, source_frames, tracking_payload)
        patch = {"status": patch_report["contract_v2"]["status"], "metrics": patch_report["task_equivalent_contact_v2"], "gates": _patch_gates(patch_report, config)}
    object_tracking, object_ok = _object_summary(position, rotation, source_frames)
    derivatives = _derivatives(qpos, float(config["timing"]["source_fps"]))
    step_qacc = np.abs(arrays["qacc"]).max(axis=1)
    step_force = arrays["contact_force_n"]
    gates = {
        "finite": bool(finite.all() and np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all()),
        "no_warnings": not warnings,
        "joint_limits": not dynamic._joint_limit_violations(model, qpos),
        "joint_margin": bool(np.nanmin(joint_margin) >= float(config["oracle"]["min_joint_margin_fraction"])),
        "tracking": tracking_ok,
        "smoothness": normalized_delta <= float(config["oracle"]["max_normalized_joint_delta"]),
        "object_tracking": object_ok,
        "collision_depth": float(depth.max(initial=0.0)) <= float(config["oracle"]["collision_depth_m"]),
        "force_finite": bool(np.isfinite(step_force).all()),
        "no_exponential_force_growth": not dynamic._has_exponential_growth(step_force, 10.0, 3),
    }
    if no_hand_object_contact:
        passed = all(gates[key] for key in ("finite", "no_warnings", "joint_limits", "joint_margin", "tracking", "smoothness"))
    else:
        passed = all(gates.values()) and all(patch["gates"].values())
    title = {"oracle_b": "B: perfect hand / dynamic object", "oracle_c": "C: dynamic hand / kinematic object", "oracle_d": "D: no-contact controller tracking"}.get(name, "R2 controller candidate")
    report = {
        "schema_version": 1,
        "stage": "C-V2R1",
        "oracle": name.upper().replace("ORACLE_", ""),
        "setup": title,
        "status": "PASS" if passed else "FAIL",
        "profile": profile,
        "controller_profile": controller_profile,
        "contact_dynamics_profile": contact_profile,
        "object_guidance_profile": object_guidance_profile,
        "feedforward": {
            "enabled": feedforward_enabled,
            "inverse_dynamics_offset_abs_max_rad": float(np.abs(feedforward).max(initial=0.0)),
            "state_feedback_gain": feedback_gain,
            "state_feedback_limits": feedback_limits,
        },
        "contact_ik_feedback": {
            "enabled": contact_ik_enabled,
            "controller_target_only": True,
            "source_anchor_trajectory_unchanged": True,
            "object_qpos_written": False,
        },
        "collision_policy": collision_policy,
        "hand_object_contact_integrity": {
            "dynamic_frame_event_count": int(hand_object_count.sum()),
            "dynamic_substep_event_count": int(arrays["hand_object_contact_count"].sum()),
            "max_dynamic_depth_m": float(depth.max(initial=0.0)),
            "max_dynamic_force_n": float(force.max(initial=0.0)),
            "verified": bool(not no_hand_object_contact or (not hand_object_count.any() and not arrays["hand_object_contact_count"].any())),
        },
        "source_frame_mapping_complete": bool(np.array_equal(source_frames, np.arange(1460, 1874))),
        "frame_count": len(reference),
        "source_interval_step_counts": {"min": min(interval_counts, default=0), "max": max(interval_counts, default=0), "total": sum(interval_counts)},
        "warnings": warnings,
        "gates": gates,
        "patch": patch,
        "object_tracking": object_tracking,
        "robot_tracking": tracking_by_side,
        "joint_margin": {"minimum_fraction": float(np.nanmin(joint_margin)), "per_joint": margin_records},
        "normalized_one_frame_delta": normalized_delta,
        "derivatives": {key + "_max": float(np.abs(value).max(initial=0.0)) for key, value in derivatives.items()},
        "collision_depth_max_m": float(depth.max(initial=0.0)),
        "contact_force_max_n": float(force.max(initial=0.0)),
        "contact_force_p95_n": float(np.percentile(force, 95)),
        "contact_force_impulse_ns": float(step_force.sum() * model.opt.timestep),
        "qacc_max": float(step_qacc.max(initial=0.0)),
        "trace": str(root / trace_name),
        "perfect_hand_clamp": {
            "enabled": perfect_hand,
            "constraint_impulse_norm_sum": float(sum(clamps_impulse)),
            "constraint_impulse_norm_max": float(max(clamps_impulse, default=0.0)),
            "constraint_energy_delta_sum": float(sum(clamps_energy)),
            "constraint_energy_delta_abs_max": float(max((abs(value) for value in clamps_energy), default=0.0)),
            "interpretation": "diagnostic hard-clamp accounting only; never a valid final rollout" if perfect_hand else "not used",
        },
    }
    filename = {"oracle_b": "oracle_b_perfect_hand_dynamic_object.json", "oracle_c": "oracle_c_dynamic_hand_kinematic_object.json", "oracle_d": "oracle_d_no_contact_tracking.json"}.get(name, f"{name}.json")
    _write_json(root / filename, report)
    return report


def run_oracle_a(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Oracle A: exact C-XA qpos/source-object mj_forward, never stepping."""
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R1 must run frozen primary first")
    paths = load_project_paths(paths_config)
    config = _config()
    root = _root(paths, sequence_id)
    _inputs, physics, reference, reference_qvel, source_frames, expected, source_anchors = _load_inputs(paths)
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    profile = _configure_model(model, config)
    hand, objects = _contact_ids(model)
    data = mujoco.MjData(model)
    finite = np.empty(len(reference), dtype=np.uint8)
    collision = np.empty(len(reference), dtype=np.float64)
    contact_count = np.empty(len(reference), dtype=np.int32)
    qacc = np.empty(len(reference), dtype=np.float64)
    for frame, state in enumerate(reference):
        data.qpos[:] = state
        data.qvel[:] = reference_qvel[frame]
        mujoco.mj_forward(model, data)
        count, depth, _force, _pairs = dynamic._contact_summary(model, data, hand, objects)
        finite[frame] = _finite_data(data)
        collision[frame] = depth
        contact_count[frame] = count
        qacc[frame] = np.abs(data.qacc).max(initial=0.0)
    ranges = _robot_ranges(model)
    normalized_delta = float(np.max(np.abs(np.diff(reference[:, :52], axis=0)) / ranges))
    margin, margin_records = _joint_margin(model, reference)
    derivative = _derivatives(reference, float(config["timing"]["source_fps"]))
    base, tracking, tracking_error, positions = _base_contact_metrics(
        model, reference, reference, source_frames, source_anchors, physics, float(collision.max(initial=0.0)), True, normalized_delta <= float(config["oracle"]["max_normalized_joint_delta"])
    )
    evaluated = _evaluate_patch(paths_config, root, "oracle_a_kinematic_reference", reference, reference_qvel, source_frames, base)
    patch_gates = _patch_gates(evaluated, config)
    gates = {
        "finite": bool(finite.all()),
        "patch_contract": all(patch_gates.values()),
        "joint_limits": not dynamic._joint_limit_violations(model, reference),
        "joint_margin": bool(np.nanmin(margin) >= float(config["oracle"]["min_joint_margin_fraction"])),
        "one_frame_delta": normalized_delta <= float(config["oracle"]["max_normalized_joint_delta"]),
        "velocity": float(np.abs(derivative["qvel_fd"]).max(initial=0.0)) <= float(config["oracle"]["max_robot_velocity_rad_s"]),
        "acceleration": float(np.abs(derivative["qacc_fd"]).max(initial=0.0)) <= float(config["oracle"]["max_robot_acceleration_rad_s2"]),
        "jerk": float(np.abs(derivative["jerk_fd"]).max(initial=0.0)) <= float(config["oracle"]["max_robot_jerk_rad_s3"]),
        "assignment_continuity": evaluated["task_equivalent_contact_v2"]["assignment_switch_rate_per_s"] == 0.0,
    }
    report = {
        "schema_version": 1,
        "stage": "C-V2R1",
        "oracle": "A",
        "setup": "exact C-XA robot qpos plus source object pose; mj_forward only; no stepping",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "profile": profile,
        "frame_count": len(reference),
        "source_frame_mapping_complete": bool(np.array_equal(source_frames, np.arange(1460, 1874))),
        "gates": gates,
        "patch": {"metrics": evaluated["task_equivalent_contact_v2"], "gates": patch_gates},
        "joint_margin": {"minimum_fraction": float(np.nanmin(margin)), "per_joint": margin_records},
        "normalized_one_frame_delta": normalized_delta,
        "derivatives": {key + "_max": float(np.abs(value).max(initial=0.0)) for key, value in derivative.items()},
        "object_source_step": {"position_max_m": float(np.abs(np.diff(reference[:, 52:55], axis=0)).max(initial=0.0)), "rotation_chart_max": float(np.abs(np.diff(reference[:, 55:58], axis=0)).max(initial=0.0))},
        "static_collision_depth_max_m": float(collision.max(initial=0.0)),
        "static_qacc_max": float(qacc.max(initial=0.0)),
    }
    _write_npz(root / "oracle_a_kinematic_reference.npz", qpos=reference, qvel=reference_qvel, source_frame_indices=source_frames, finite=finite, collision_depth_m=collision, contact_count=contact_count, qacc=qacc, joint_margin_fraction=margin, robot_site_tracking_error_m=tracking_error, **derivative)
    _write_json(root / "oracle_a_kinematic_reference.json", report)
    return str(root / "oracle_a_kinematic_reference.json")


def run_oracle_d(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Run the pairwise-filtered no-contact controller-tracking oracle."""
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R1 must run frozen primary first")
    _ = load_project_paths(paths_config)
    _run_dynamic_oracle(paths_config, "oracle_d", no_hand_object_contact=True)
    return str(_root(load_project_paths(paths_config), sequence_id) / "oracle_d_no_contact_tracking.json")


def _first_index(values: np.ndarray, predicate) -> int | None:
    candidates = np.flatnonzero(predicate(values))
    return None if not len(candidates) else int(candidates[0])


def _timeline(paths_config: str, root: Path) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    old_root = dynamic._root(paths, PRIMARY)
    old_metrics = json.loads((old_root / "metrics_corrected_v2_forward_rollout.json").read_text(encoding="utf-8"))
    _inputs, _physics, reference, _reference_qvel, source_frames, expected, source_anchors = _load_inputs(paths)
    with np.load(old_root / "trajectory_corrected_v2_forward_rollout.npz", allow_pickle=False) as archive:
        actual = np.asarray(archive["qpos"], dtype=np.float64)
    with np.load(old_root / "forward_rollout_trace.npz", allow_pickle=False) as archive:
        trace = {name: np.asarray(archive[name]) for name in archive.files}
    config = _config()["oracle"]
    model = mujoco.MjModel.from_xml_path(str(dynamic._physics(paths, PRIMARY)["scene_act"]))
    ranges = _robot_ranges(model)
    reference_delta = np.max(np.abs(np.diff(reference[:, :52], axis=0)) / ranges, axis=1)
    lag = np.max(np.abs(actual[:, :52] - reference[:, :52]) / ranges, axis=1)
    site_error = trace["robot_site_tracking_error_m"]
    tip_positions = dynamic._dynamic_robot_tracking(model, actual, reference, source_frames)[0][:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    patch_proxy = np.linalg.norm(tip_positions - source_anchors, axis=2)
    first_patch = _first_index(np.max(np.where(expected, patch_proxy, 0.0), axis=1), lambda value: value > float(config["patch_distance_m"]))
    step_depth = trace["contact_depth_m"]
    step_force = trace["contact_force_n"]
    collision_step = _first_index(step_depth, lambda value: value > float(config["collision_depth_m"]))
    force_step = _first_index(step_force, lambda value: value > float(config["force_spike_n"]))
    events: list[dict[str, Any]] = []
    for kind, frame in (("first_large_reference_joint_delta", _first_index(reference_delta, lambda value: value > 0.20)), ("first_controller_tracking_lag", _first_index(site_error.max(axis=1), lambda value: value > 0.015)), ("first_patch_distance_threshold_breach", first_patch), ("first_contact_role_loss_proxy", first_patch)):
        if frame is not None:
            events.append({"event": kind, "source_frame": int(source_frames[frame]), "trajectory_index": frame, "substep": 0})
    violations = old_metrics.get("joint_limit_violation_records", [])
    if violations:
        frame = min(min(row["frames"]) for row in violations)
        events.append({"event": "first_joint_limit_margin_breach", "source_frame": int(source_frames[frame]), "trajectory_index": int(frame), "substep": 0, "joints": [row["joint"] for row in violations if frame in row["frames"]]})
    for kind, step in (("first_collision_depth_spike", collision_step), ("first_force_spike", force_step)):
        if step is not None:
            frame = int(trace["frame_index"][step])
            events.append({"event": kind, "source_frame": int(source_frames[frame]), "trajectory_index": frame, "substep": int(trace["substep_index"][step]), "value": float(step_depth[step] if kind.endswith("depth_spike") else step_force[step])})
    events.append({"event": "first_assignment_discontinuity", "source_frame": None, "trajectory_index": None, "substep": None, "value": "NONE; immutable assignment switch rate is 0"})
    ordered = sorted((item for item in events if item["source_frame"] is not None), key=lambda item: (item["source_frame"], item["substep"]))
    payload = {
        "schema_version": 1,
        "stage": "C-V2R1",
        "source": "preserved fine_dt0005_lead20_simstep_gates D2 trace",
        "status": "PASS",
        "events": events,
        "causal_event_order": ordered,
        "first_failure": ordered[0] if ordered else None,
        "per_side": old_metrics.get("robot_tracking_by_side", {}),
        "per_joint": {"joint_limit_violation_records": violations, "first_normalized_reference_delta": float(reference_delta.max(initial=0.0)), "first_normalized_actual_lag": float(lag.max(initial=0.0))},
        "per_role": old_metrics.get("task_equivalent_contact_v2", {}).get("role_metrics", []),
        "per_geom_pair": old_metrics.get("collision", {}).get("per_geom_pair", {}),
        "assignment": old_metrics.get("task_equivalent_contact_v2", {}).get("assignment_switch_rate_per_s"),
    }
    _write_npz(root / "v2r1_causality_trace.npz", source_frame_indices=source_frames, reference_qpos=reference, actual_qpos=actual, reference_delta_normalized=reference_delta, controller_lag_normalized=lag, controller_site_lag_m=site_error, patch_distance_proxy_m=patch_proxy, trace_frame_index=trace["frame_index"], trace_substep_index=trace["substep_index"], trace_qpos=trace["qpos"], trace_qvel=trace["qvel"], trace_qacc=trace["qacc"], trace_ctrl=trace["ctrl"], trace_actuator_force=trace["actuator_force"], trace_contact_depth_m=step_depth, trace_contact_force_n=step_force)
    _write_json(paths.workspace_root / "reports/v2r1_causality_timeline.json", payload)
    markdown = "# V2R1 causality timeline\n\n| Event | Source frame | Substep |\n| --- | ---: | ---: |\n" + "\n".join(f"| {row['event']} | {row['source_frame'] if row['source_frame'] is not None else 'NONE'} | {row['substep'] if row['substep'] is not None else '-'} |" for row in events) + "\n\nThe ordering above is measured from the preserved D2 trace; no C-XA input or controller parameter was modified.\n"
    (paths.workspace_root / "reports/V2R1_CAUSALITY_TIMELINE.md").write_text(markdown, encoding="utf-8")
    return payload


def classify_oracles(oracle_a: dict[str, Any], oracle_b: dict[str, Any], oracle_c: dict[str, Any], oracle_d: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen decision matrix without result-dependent thresholds."""
    status = {"A": oracle_a["status"] == "PASS", "B": oracle_b["status"] == "PASS", "C": oracle_c["status"] == "PASS", "D": oracle_d["status"] == "PASS"}
    if not status["A"]:
        primary, secondary = "REFERENCE_TRAJECTORY_TIMING_INFEASIBLE", []
    elif not status["D"]:
        primary, secondary = "CONTROLLER_TRACKING_INSUFFICIENT", []
    elif not status["C"]:
        primary, secondary = "CONTACT_COLLISION_COUPLING", []
    elif not status["B"]:
        primary, secondary = "OBJECT_GUIDANCE_CONTACT_INCOMPATIBLE", []
    else:
        primary, secondary = "OBJECT_GUIDANCE_CONTACT_INCOMPATIBLE", ["CONTACT_COLLISION_COUPLING"]
    classification = "MULTI_FACTOR" if len(secondary) > 1 else primary
    return {"status": "PASS", "oracle_status": status, "root_cause_branch": classification, "primary_cause": primary, "secondary_causes": secondary, "branch_frozen": True}


def refresh_v2r1_decision(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Regenerate timeline/decision from completed immutable oracle artifacts."""
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R1 is primary-only")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    reports = {
        letter: json.loads((root / filename).read_text(encoding="utf-8"))
        for letter, filename in {
            "A": "oracle_a_kinematic_reference.json",
            "B": "oracle_b_perfect_hand_dynamic_object.json",
            "C": "oracle_c_dynamic_hand_kinematic_object.json",
            "D": "oracle_d_no_contact_tracking.json",
        }.items()
    }
    timeline = _timeline(paths_config, root)
    decision = classify_oracles(reports["A"], reports["B"], reports["C"], reports["D"])
    decision.update({
        "stage": "C-V2R1",
        "sequence_id": sequence_id,
        "timing": _config()["timing"],
        "first_failure": timeline["first_failure"],
        "causal_event_order": timeline["causal_event_order"],
        "oracles": {letter: str(root / filename) for letter, filename in {
            "A": "oracle_a_kinematic_reference.json", "B": "oracle_b_perfect_hand_dynamic_object.json",
            "C": "oracle_c_dynamic_hand_kinematic_object.json", "D": "oracle_d_no_contact_tracking.json",
        }.items()},
        "profile_hashes": {"V2R_REFERENCE_PROFILE_HASH": _sha256(CONFIG_PATH), "V2R_CONTROLLER_PROFILE_HASH": "NOT_APPLICABLE", "V2R_OBJECT_GUIDANCE_PROFILE_HASH": "NOT_APPLICABLE", "V2R_CONTACT_DYNAMICS_PROFILE_HASH": "PENDING"},
    })
    _write_json(paths.workspace_root / "reports/v2r1_oracle_decision.json", decision)
    body = "# V2R1 Oracle Decision\n\n" + "\n".join((f"- Oracle {key}: {'PASS' if value else 'FAIL'}" for key, value in decision["oracle_status"].items())) + f"\n\nPrimary cause: **{decision['primary_cause']}**\n\nRoot-cause branch: **{decision['root_cause_branch']}**\n\nThe branch is frozen from the recorded A-D evidence; V2R2 must use only its corresponding minimal repair.\n"
    (paths.workspace_root / "reports/V2R1_ORACLE_DECISION.md").write_text(body, encoding="utf-8")
    return str(paths.workspace_root / "reports/v2r1_oracle_decision.json")


def _force_thresholds(paths) -> dict[str, float]:
    """Freeze force limits from D1 stable holds, never from a D2 failure."""
    config = _config()["oracle"]
    old_root = dynamic._root(paths, PRIMARY)
    with np.load(old_root / "preflight_keyframe_hold_steps.npz", allow_pickle=False) as archive:
        force = np.asarray(archive["contact_force"], dtype=np.float64)
    dt = float(dynamic._config()["physics"]["sim_timestep_s"])
    return {
        "reference": "preserved D1 keyframe holds",
        "p95_n": float(np.percentile(force, 95) * float(config["stable_hold_force_p95_multiplier"])),
        "max_n": float(force.max(initial=0.0) * float(config["stable_hold_force_max_multiplier"])),
        "impulse_ns": float(force.sum() * dt * float(config["stable_hold_force_impulse_multiplier"])),
        "source_p95_n": float(np.percentile(force, 95)),
        "source_max_n": float(force.max(initial=0.0)),
        "source_impulse_ns": float(force.sum() * dt),
    }


def _candidate_force_metrics(trace_path: Path, timestep: float) -> dict[str, float]:
    with np.load(trace_path, allow_pickle=False) as archive:
        force = np.asarray(archive["contact_force_n"], dtype=np.float64)
    return {
        "p95_n": float(np.percentile(force, 95)),
        "max_n": float(force.max(initial=0.0)),
        "impulse_ns": float(force.sum() * timestep),
    }


def _r2_candidate_gates(report: dict[str, Any], force: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
    patch = report.get("patch", {}).get("gates", {})
    gates = report["gates"]
    return {
        "finite": bool(gates["finite"] and gates["no_warnings"]),
        "joint_limits_and_margin": bool(gates["joint_limits"] and gates["joint_margin"]),
        "smoothness": bool(gates["smoothness"]),
        "robot_tracking": bool(gates["tracking"]),
        "object_tracking": bool(gates["object_tracking"]),
        "collision_depth": bool(gates["collision_depth"]),
        "patch_coverage": bool(patch.get("patch_coverage", False)),
        "functional_role_recall": bool(patch.get("functional_role_recall", False)),
        "patch_distance_p95": bool(patch.get("patch_distance_p95", False)),
        "normal_alignment": bool(patch.get("normal_alignment", False)),
        "force_p95": force["p95_n"] <= thresholds["p95_n"],
        "force_max": force["max_n"] <= thresholds["max_n"],
        "force_impulse": force["impulse_ns"] <= thresholds["impulse_ns"],
    }


def run_v2r2_controller_repair(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """R2 bounded controller-only search after an Oracle-D controller verdict."""
    if sequence_id != PRIMARY:
        raise RuntimeError("primary R2 must complete before any smoke")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    decision_path = paths.workspace_root / "reports/v2r1_oracle_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("V2R1 decision is required before any repair")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("root_cause_branch") != "CONTROLLER_TRACKING_INSUFFICIENT":
        raise RuntimeError("R2 controller repair is not authorized by the frozen V2R1 decision")
    config = _config()
    candidates = list(config["search"]["controller_candidates"])
    if not candidates or len(candidates) > int(config["search"]["max_candidates_per_branch"]):
        raise RuntimeError("controller candidate list must be pre-frozen and bounded")
    thresholds = _force_thresholds(paths)
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        name = f"profiles/controller_jointwise_candidates/candidate_{index:02d}"
        report = _run_dynamic_oracle(paths_config, name, controller_candidate=candidate)
        force = _candidate_force_metrics(Path(report["trace"]), float(report["profile"]["sim_timestep_s"]))
        gates = _r2_candidate_gates(report, force, thresholds)
        patch_metrics = report.get("patch", {}).get("metrics", {})
        candidate_rows.append({
            "candidate_id": index,
            "profile": report["controller_profile"],
            "oracle_rollout_report": str(root / f"{name}.json"),
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "force": force,
            "patch": {key: patch_metrics.get(key) for key in ("patch_coverage", "functional_role_recall", "surface_patch_distance_p95_m", "normal_cosine_median")},
            "failure_gates": [key for key, value in gates.items() if not value],
        })
    passing = [row for row in candidate_rows if row["status"] == "PASS"]
    search = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "branch": "CONTROLLER_TRACKING_INSUFFICIENT",
        "search_space_frozen": candidates,
        "preliminary_scalar_audit": {
            "status": "NOT_USED_FOR_SELECTION",
            "directory": str(root / "profiles/controller_candidates"),
            "reason": "Seven completed scalar lead/global-gain probes and two partial traces are preserved, but they do not satisfy the required joint-specific controller audit scope.",
        },
        "force_thresholds": thresholds,
        "candidates": candidate_rows,
        "status": "PASS" if passing else "BLOCKED",
        "forbidden_mutations_not_used": ["raw GRAB", "body models", "Stage B", "C-XA contact targets", "object qpos rewrites", "contact dynamics", "object guidance"],
    }
    _write_json(root / "profiles/controller_profile_search.json", search)
    if not passing:
        blocked = {
            "schema_version": 1,
            "stage": "C-V2R2",
            "status": "BLOCKED",
            "reason": "Every pre-frozen bounded controller-only candidate failed one or more immutable dynamic gates; object guidance/contact changes are not authorized by the controller-only V2R1 branch.",
            "controller_profile_search": str(root / "profiles/controller_profile_search.json"),
            "v2r3": "NOT_RUN",
            "v2r4": "NOT_RUN",
            "v2r5": "NOT_RUN",
        }
        _write_json(paths.workspace_root / "reports/v2r2_controller_repair.json", blocked)
        return str(paths.workspace_root / "reports/v2r2_controller_repair.json")
    selected = sorted(passing, key=lambda row: (-float(row["patch"]["patch_coverage"]), float(row["patch"]["surface_patch_distance_p95_m"]), float(row["force"]["p95_n"])))[0]
    selected_path = root / "profiles/selected_controller_profile.json"
    selected_payload = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "branch": "CONTROLLER_TRACKING_INSUFFICIENT",
        "selected": selected,
        "all_pilots_must_share_this_profile": True,
        "profile_hash": _payload_hash(selected["profile"]),
    }
    _write_json(selected_path, selected_payload)
    selected_report_path = Path(selected["oracle_rollout_report"])
    selected_report = json.loads(selected_report_path.read_text(encoding="utf-8"))
    trajectory_path = selected_report_path.with_name(selected_report_path.stem + "_trajectory.npz")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        _write_npz(root / "trajectory_v2r_dynamic_reference.npz", **{name: np.asarray(archive[name]) for name in archive.files})
    with np.load(Path(selected_report["trace"]), allow_pickle=False) as archive:
        _write_npz(root / "controller_refinement_trace.npz", **{name: np.asarray(archive[name]) for name in archive.files})
        _write_npz(root / "ctrl_v2r.npz", ctrl=np.asarray(archive["ctrl"]), frame_index=np.asarray(archive["frame_index"]), substep_index=np.asarray(archive["substep_index"]))
    _write_json(root / "metrics_v2r_dynamic_reference.json", selected_report)
    result = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "status": "PASS",
        "branch": "CONTROLLER_TRACKING_INSUFFICIENT",
        "selected_controller_profile": str(selected_path),
        "V2R_REFERENCE_PROFILE_HASH": _sha256(CONFIG_PATH),
        "V2R_CONTROLLER_PROFILE_HASH": selected_payload["profile_hash"],
        "V2R_OBJECT_GUIDANCE_PROFILE_HASH": "NOT_APPLICABLE",
        "V2R_CONTACT_DYNAMICS_PROFILE_HASH": "NOT_APPLICABLE",
        "trajectory": str(root / "trajectory_v2r_dynamic_reference.npz"),
        "metrics": str(root / "metrics_v2r_dynamic_reference.json"),
        "controller_trace": str(root / "controller_refinement_trace.npz"),
        "force_thresholds": thresholds,
    }
    _write_json(paths.workspace_root / "reports/v2r2_controller_repair.json", result)
    return str(paths.workspace_root / "reports/v2r2_controller_repair.json")


def run_v2r2_contact_dynamics_repair(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Run the frozen bounded Branch-D contact-pair search on the primary."""
    if sequence_id != PRIMARY:
        raise RuntimeError("primary R2 must complete before any smoke")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    decision_path = paths.workspace_root / "reports/v2r1_oracle_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("V2R1 decision is required before any repair")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("root_cause_branch") != "CONTACT_COLLISION_COUPLING":
        raise RuntimeError("R2 contact-dynamics repair is not authorized by the frozen V2R1 decision")
    config = _config()
    candidates = list(config["search"]["contact_dynamics_candidates"])
    if not candidates or len(candidates) > int(config["search"]["max_candidates_per_branch"]):
        raise RuntimeError("contact-dynamics candidate list must be pre-frozen and bounded")
    ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if len(set(ids)) != len(ids) or any(not candidate_id for candidate_id in ids):
        raise RuntimeError("contact-dynamics candidate ids must be unique and non-empty")
    thresholds = _force_thresholds(paths)
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        name = f"profiles/contact_dynamics_candidates/candidate_{index:02d}_{candidate['candidate_id']}"
        report = _run_dynamic_oracle(paths_config, name, contact_candidate=candidate)
        force = _candidate_force_metrics(Path(report["trace"]), float(report["profile"]["sim_timestep_s"]))
        gates = _r2_candidate_gates(report, force, thresholds)
        patch_metrics = report.get("patch", {}).get("metrics", {})
        candidate_rows.append({
            "candidate_id": str(candidate["candidate_id"]),
            "profile": report["contact_dynamics_profile"],
            "oracle_rollout_report": str(root / f"{name}.json"),
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "force": force,
            "patch": {key: patch_metrics.get(key) for key in ("patch_coverage", "functional_role_recall", "surface_patch_distance_p95_m", "normal_cosine_median")},
            "failure_gates": [key for key, value in gates.items() if not value],
        })
    passing = [row for row in candidate_rows if row["status"] == "PASS"]
    search = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "branch": "CONTACT_COLLISION_COUPLING",
        "search_space_frozen": candidates,
        "search_space_hash": _payload_hash({"candidates": candidates, "config": config["r2"]}),
        "r2_profile": config["r2"],
        "force_thresholds": thresholds,
        "candidates": candidate_rows,
        "status": "PASS" if passing else "BLOCKED",
        "scope": {
            "modified_in_memory_only": ["explicit hand-object pair solref", "pair solimp", "pair margin", "pair gap", "pair friction"],
            "unchanged": ["raw GRAB", "body models", "Stage B", "C-XA contact targets", "reference qpos", "controller profile", "object target trajectory", "self-collision", "floor-object contact"],
            "no_negative_margin_or_gap": True,
        },
    }
    search_path = root / "profiles/contact_dynamics_search.json"
    _write_json(search_path, search)
    report_path = paths.workspace_root / "reports/v2r2_contact_dynamics_repair.json"
    if not passing:
        blocked = {
            "schema_version": 1,
            "stage": "C-V2R2",
            "status": "BLOCKED",
            "branch": "CONTACT_COLLISION_COUPLING",
            "reason": "Every pre-frozen bounded contact-dynamics candidate failed one or more immutable dynamic gates; reference, controller, and object guidance changes are not authorized by the Branch-D diagnosis.",
            "contact_dynamics_search": str(search_path),
            "force_thresholds": thresholds,
            "v2r3": "NOT_RUN",
            "v2r4": "NOT_RUN",
            "v2r5": "NOT_RUN",
        }
        _write_json(report_path, blocked)
        return str(report_path)
    selected = sorted(
        passing,
        key=lambda row: (
            -float(row["patch"]["patch_coverage"]),
            float(row["patch"]["surface_patch_distance_p95_m"]),
            float(row["force"]["p95_n"]),
        ),
    )[0]
    selected_payload = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "branch": "CONTACT_COLLISION_COUPLING",
        "selected": selected,
        "all_pilots_must_share_this_profile": True,
        "profile_hash": _payload_hash(selected["profile"]),
    }
    selected_path = root / "profiles/selected_contact_dynamics_profile.json"
    _write_json(selected_path, selected_payload)
    selected_report_path = Path(selected["oracle_rollout_report"])
    selected_report = json.loads(selected_report_path.read_text(encoding="utf-8"))
    _inputs, _physics, reference, reference_qvel, source_frames, _expected, _anchors = _load_inputs(paths)
    _write_npz(
        root / "trajectory_v2r_dynamic_reference.npz",
        qpos=reference,
        qvel=reference_qvel,
        source_frame_indices=source_frames,
    )
    with np.load(Path(selected_report["trace"]), allow_pickle=False) as archive:
        _write_npz(root / "contact_dynamics_refinement_trace.npz", **{name: np.asarray(archive[name]) for name in archive.files})
        _write_npz(root / "ctrl_v2r.npz", ctrl=np.asarray(archive["ctrl"]), frame_index=np.asarray(archive["frame_index"]), substep_index=np.asarray(archive["substep_index"]))
    _write_json(root / "metrics_v2r_dynamic_reference.json", selected_report)
    result = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "status": "PASS",
        "branch": "CONTACT_COLLISION_COUPLING",
        "selected_contact_dynamics_profile": str(selected_path),
        "V2R_REFERENCE_PROFILE_HASH": _sha256(CONFIG_PATH),
        "V2R_CONTROLLER_PROFILE_HASH": "NOT_APPLICABLE",
        "V2R_OBJECT_GUIDANCE_PROFILE_HASH": "NOT_APPLICABLE",
        "V2R_CONTACT_DYNAMICS_PROFILE_HASH": selected_payload["profile_hash"],
        "trajectory": str(root / "trajectory_v2r_dynamic_reference.npz"),
        "metrics": str(root / "metrics_v2r_dynamic_reference.json"),
        "contact_dynamics_trace": str(root / "contact_dynamics_refinement_trace.npz"),
        "force_thresholds": thresholds,
        "object_trajectory_target_unchanged": True,
    }
    _write_json(report_path, result)
    return str(report_path)


def invalidate_pre_oracle_d_controller_branch(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Retain controller probes made before the valid Oracle-D pair fix.

    The probes are useful diagnostic history but cannot be selected after the
    corrected oracle proves a Branch-D contact/collision cause.  The original
    JSON evidence is copied under the external attempt directory before the
    top-level controller report is marked ``NOT_RUN`` for this valid branch.
    """
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R evidence invalidation is primary-only")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    decision_path = paths.workspace_root / "reports/v2r1_oracle_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("V2R1 decision is required before invalidating a provisional branch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("root_cause_branch") != "CONTACT_COLLISION_COUPLING":
        raise RuntimeError("controller evidence may be invalidated only after a valid Branch-D decision")
    archive = root / "attempts/pre_oracle_d_pair_fix_controller_evidence"
    archive.mkdir(parents=True, exist_ok=True)
    sources = {
        "v2r2_controller_repair.json": paths.workspace_root / "reports/v2r2_controller_repair.json",
        "controller_profile_search.json": root / "profiles/controller_profile_search.json",
        "feedforward_verification.json": root / "profiles/feedforward_controller/feedforward_verification.json",
    }
    copied: dict[str, str] = {}
    for name, source in sources.items():
        if source.is_file():
            destination = archive / name
            shutil.copy2(source, destination)
            copied[name] = str(destination)
    audit = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "status": "NOT_RUN",
        "reason": "Controller probes were completed before Oracle D correctly neutralized explicit hand-object pairs. The valid A/PASS, D/PASS, C/FAIL decision authorizes only CONTACT_COLLISION_COUPLING repair, so these probes are retained but not eligible for selection.",
        "valid_root_cause_branch": decision["root_cause_branch"],
        "valid_oracle_d": str(root / "oracle_d_no_contact_tracking.json"),
        "preserved_json_evidence": copied,
        "preserved_trace_directories": [str(root / "profiles/controller_candidates"), str(root / "profiles/controller_jointwise_candidates"), str(root / "profiles/feedforward_controller")],
    }
    _write_json(archive / "branch_selection_audit.json", audit)
    _write_json(paths.workspace_root / "reports/v2r2_controller_repair.json", audit)
    return str(archive / "branch_selection_audit.json")


def run_v2r2_feedforward_repair(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Verify one bounded inverse-dynamics/feedforward controller construction."""
    if sequence_id != PRIMARY:
        raise RuntimeError("primary-only V2R2 controller verification")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    search_path = root / "profiles/controller_profile_search.json"
    if not search_path.is_file() or json.loads(search_path.read_text(encoding="utf-8")).get("status") != "BLOCKED":
        raise RuntimeError("feedforward verification follows the completed bounded jointwise search only")
    config = _config()
    repair = config["controller_feedforward_repair"]
    candidate = {
        "lead_source_frames": int(repair["lead_source_frames"]), "kp_scale": 1.0, "kv_scale": 1.0, "force_limit_scale": 1.0,
        "inverse_dynamics_scale": float(repair["inverse_dynamics_scale"]), "state_feedback_gain": float(repair["state_feedback_gain"]),
    }
    report = _run_dynamic_oracle(paths_config, "profiles/feedforward_controller/fixed_profile", controller_candidate=candidate)
    force = _candidate_force_metrics(Path(report["trace"]), float(report["profile"]["sim_timestep_s"]))
    thresholds = _force_thresholds(paths)
    gates = _r2_candidate_gates(report, force, thresholds)
    verification = {
        "schema_version": 1,
        "stage": "C-V2R2",
        "kind": "deterministic_inverse_dynamics_feedforward_plus_bounded_state_feedback",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "candidate": candidate,
        "report": str(root / "profiles/feedforward_controller/fixed_profile.json"),
        "gates": gates,
        "force": force,
        "force_thresholds": thresholds,
        "patch": report.get("patch", {}).get("metrics", {}),
        "forbidden_mutations_not_used": ["robot qpos state rewrite", "object qpos rewrite", "C-XA contact target edit", "source-frame modification"],
    }
    output = root / "profiles/feedforward_controller/feedforward_verification.json"
    _write_json(output, verification)
    summary_path = paths.workspace_root / "reports/v2r2_controller_repair.json"
    if verification["status"] == "PASS":
        _write_json(summary_path, {"schema_version": 1, "stage": "C-V2R2", "status": "PASS", "branch": "CONTROLLER_TRACKING_INSUFFICIENT", "controller_profile_search": str(search_path), "feedforward_verification": str(output), "selected_profile": str(root / "profiles/feedforward_controller/fixed_profile.json")})
    else:
        _write_json(summary_path, {"schema_version": 1, "stage": "C-V2R2", "status": "BLOCKED", "reason": "The pre-frozen jointwise controller search and the deterministic inverse-dynamics/feedforward plus bounded-feedback verification both fail immutable dynamic gates.  No object-guidance or contact-dynamics repair is authorized by the controller-only V2R1 decision.", "controller_profile_search": str(search_path), "feedforward_verification": str(output), "v2r3": "NOT_RUN", "v2r4": "NOT_RUN", "v2r5": "NOT_RUN"})
    return str(summary_path)


def write_v2r_blocked_acceptance(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Write explicit NOT_RUN downstream statuses after a genuine R2 block."""
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R blocked acceptance is primary-only")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    decision = paths.workspace_root / "reports/v2r1_oracle_decision.json"
    if not decision.is_file():
        raise RuntimeError("V2R1 decision is required")
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    branch = decision_payload.get("root_cause_branch")
    repair_names = {
        "CONTROLLER_TRACKING_INSUFFICIENT": "v2r2_controller_repair.json",
        "CONTACT_COLLISION_COUPLING": "v2r2_contact_dynamics_repair.json",
    }
    repair_name = repair_names.get(branch)
    if repair_name is None:
        raise RuntimeError(f"blocked aggregate has no implemented report for branch {branch!r}")
    repair = paths.workspace_root / "reports" / repair_name
    if not decision.is_file() or not repair.is_file():
        raise RuntimeError("V2R1 decision and V2R2 repair report are required")
    repair_payload = json.loads(repair.read_text(encoding="utf-8"))
    if decision_payload.get("status") != "PASS" or repair_payload.get("status") != "BLOCKED":
        raise RuntimeError("blocked aggregate is valid only after V2R1 PASS and V2R2 BLOCKED")
    status = {
        "schema_version": 1,
        "stage": "C-V2R",
        "status": "BLOCKED",
        "historical": {
            "v1_exact_contact": {"status": "BLOCKED", "note": "infeasible embodiment contact"},
            "original_v2": {"status": "BLOCKED", "note": "historical CASE-A-invalid evidence"},
            "cxa_corrected_static": {"status": "PASS"},
            "old_d2": {"status": "FAIL"},
        },
        "gates": {"V2R1_oracle_diagnosis": "PASS", "V2R2_minimal_repair": "BLOCKED", "V2R3_primary_D1": "NOT_RUN", "V2R3_primary_D2": "NOT_RUN", "V2R3_minimal_MJWP": "NOT_RUN", "V2R4_primary_full_MJWP": "NOT_RUN", "mug_lift": "NOT_RUN", "mug_offhand_1": "NOT_RUN", "html_delivery": "NOT_RUN", "codex_screenshot_review": "NOT_RUN", "user_html_review": "NOT_RUN", "stage_d": "NOT_RUN"},
        "root_cause": decision_payload["primary_cause"],
        "root_cause_branch": branch,
        "reason": repair_payload["reason"],
        "reports": {"causality_timeline": str(paths.workspace_root / "reports/v2r1_causality_timeline.json"), "oracle_decision": str(decision), "minimal_repair": str(repair), "contact_dynamics_search": str(root / "profiles/contact_dynamics_search.json"), "oracle_d": str(root / "oracle_d_no_contact_tracking.json")},
        "no_later_artifacts_claimed": True,
    }
    _write_json(paths.workspace_root / "reports/stage_c_v2r_validation.json", status)
    _write_json(paths.workspace_root / "reports/stage_c_v2r_pilot_summary.json", {"schema_version": 1, "status": "BLOCKED", "pilots": [{"sequence_id": PRIMARY, "status": "BLOCKED", "stage": "V2R2", "reason": status["reason"]}, {"sequence_id": "s1__mug_lift", "status": "NOT_RUN", "reason": "primary V2R2 blocked"}, {"sequence_id": "s1__mug_offhand_1", "status": "NOT_RUN", "reason": "primary V2R2 blocked"}]})
    _write_json(paths.workspace_root / "reports/stage_c_v2r_acceptance.json", status)
    _write_json(paths.workspace_root / "reports/stage_c_v2r_screenshot_review.json", {"schema_version": 1, "status": "NOT_RUN", "reason": "V2R2 minimal repair blocked; HTML/screenshots would be false acceptance evidence.", "screenshots": []})
    return str(paths.workspace_root / "reports/stage_c_v2r_validation.json")


def run_v2r1(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Complete Oracle A-D, timeline localization, and frozen decision report."""
    if sequence_id != PRIMARY:
        raise RuntimeError("V2R1 must complete frozen primary before any smoke")
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    root.mkdir(parents=True, exist_ok=True)
    oracle_a_path = Path(run_oracle_a(paths_config, sequence_id))
    oracle_a = json.loads(oracle_a_path.read_text(encoding="utf-8"))
    oracle_b = _run_dynamic_oracle(paths_config, "oracle_b", perfect_hand=True)
    oracle_c = _run_dynamic_oracle(paths_config, "oracle_c", kinematic_object=True)
    oracle_d = _run_dynamic_oracle(paths_config, "oracle_d", no_hand_object_contact=True)
    _ = oracle_a, oracle_b, oracle_c, oracle_d, oracle_a_path
    return refresh_v2r1_decision(paths_config, sequence_id)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"oracle-a": run_oracle_a, "oracle-d": run_oracle_d, "run-v2r1": run_v2r1, "refresh-v2r1-decision": refresh_v2r1_decision, "run-v2r2-controller-repair": run_v2r2_controller_repair, "run-v2r2-contact-dynamics-repair": run_v2r2_contact_dynamics_repair, "invalidate-pre-oracle-d-controller-branch": invalidate_pre_oracle_d_controller_branch, "run-v2r2-feedforward-repair": run_v2r2_feedforward_repair, "write-v2r-blocked-acceptance": write_v2r_blocked_acceptance})
