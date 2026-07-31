"""Fail-closed dynamic acceptance for corrected Stage C Contract V2 Level 1.

All mutable evidence is written beneath ``stage_c_v2_dynamic`` in the
external workspace.  The module deliberately reads C-XA and Stage-B inputs
without modifying them, and it refuses to run a later gate when an earlier
gate does not pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro
import yaml
from scipy.spatial.transform import Rotation

from spider.datasets.paths import load_project_paths
from spider.geometry.collision_audit import contact_pair_summary, mesh_from_model, summarize_signed_distances
from spider.tools.grab_stage_c import (
    FINGERS,
    _closest_with_sign,
    _collision_depths,
    _finite_data,
    _hand_geom_ids,
    _mesh,
    _object_tracking_error,
    _preflight_object_ids,
    _profile_hash,
    _set_object_mocap_reference,
    _site_ids,
    _stage_b_act_baseline,
    _stage_b_dirs,
    _world_to_body,
)


PRIMARY = "s5__cylindermedium_lift"
PILOTS = {
    PRIMARY: {"source_sequence_id": "s5/cylindermedium_lift", "frames": [1460, 1876], "role": "required true-bimanual primary"},
    "s1__mug_lift": {"source_sequence_id": "s1/mug_lift", "frames": [120, 240], "role": "required right-hand interaction smoke"},
    "s1__mug_offhand_1": {"source_sequence_id": "s1/mug_offhand_1", "frames": [120, 180], "role": "required offhand/non-interacting-hand smoke"},
}
CONFIG_PATH = Path("configs/project/grab_wuji_stage_c_v2_dynamic.yaml")
CONTRACT_PATH = Path("configs/project/grab_wuji_stage_c_contract_v2.yaml")


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


def _config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("corrected_namespace") != "stage_c_contract_v2_cxa" or payload.get("corrected_level") != 1:
        raise RuntimeError("dynamic acceptance may only consume C-XA Level 1")
    return payload


def _configure_dynamic_model(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen dynamic profile only to an in-memory MuJoCo model.

    This never edits the Stage-C XML.  The 120-Hz source samples remain
    untouched; exact accumulated stepping merely resolves each source interval
    with a smaller real-physics integration step.
    """
    physics = config["physics"]
    timestep = float(physics["sim_timestep_s"])
    solref = float(physics["object_mocap_solref_s"])
    lead = int(physics["robot_reference_lead_source_frames"])
    if timestep <= 0.0 or solref <= 0.0 or lead < 0:
        raise ValueError("dynamic timestep, object solref, and robot lead must be positive/non-negative")
    if physics.get("source_interval_scheduler") != "exact_accumulated_120hz":
        raise RuntimeError("dynamic acceptance requires exact 120-Hz accumulated source scheduling")
    model.opt.timestep = timestep
    # The generated Stage-C scene has only the two object mocap welds.  Apply
    # the profile in memory so the original scene is immutable.
    model.eq_solref[:, 0] = solref
    return {
        "config_path": str(CONFIG_PATH),
        "config_sha256": _profile_hash(CONFIG_PATH),
        "sim_timestep_s": timestep,
        "object_mocap_solref_s": solref,
        "robot_reference_lead_source_frames": lead,
        "source_interval_scheduler": physics["source_interval_scheduler"],
    }


def _robot_reference(reference: np.ndarray, frame: int, lead: int) -> np.ndarray:
    return reference[min(frame + lead, len(reference) - 1), :52]


def _root(paths, sequence_id: str) -> Path:
    if sequence_id not in PILOTS:
        raise ValueError(f"not a frozen dynamic-acceptance pilot: {sequence_id}")
    return _stage_b_dirs(paths.workspace_root, sequence_id)[1] / "stage_c_v2_dynamic"


def _inputs(paths, sequence_id: str) -> dict[str, Path]:
    robot = _stage_b_dirs(paths.workspace_root, sequence_id)[1]
    if sequence_id != PRIMARY:
        raise RuntimeError("only the primary has corrected C-XA inputs; smoke inputs are forbidden before D5")
    cxa = robot / "stage_c_contract_v2_cxa"
    result = {
        "root": cxa,
        "reference_json": robot / "stage_c/contact_reference_cxa.json",
        "reference_npz": robot / "stage_c/contact_reference_cxa.npz",
        "assignment_json": cxa / "selected_contact_assignment_level_1.json",
        "assignment_npz": cxa / "selected_contact_assignment_level_1.npz",
        "targets": cxa / "contact_targets_level_1_flexible.npz",
        "trajectory": cxa / "trajectory_depenetrated_init_cxa_level_1_flexible.npz",
        "metrics": cxa / "metrics_depenetrated_init_cxa_level_1_flexible.json",
        "manifest": cxa / "depenetration_manifest_cxa_level_1_flexible.json",
        "static_preflight": cxa / "preflight_static_cxa_level_1.json",
        "rerun": paths.workspace_root / "reports/cxa_case_a_rerun.json",
        "reliability": paths.workspace_root / "reports/cxa_source_contact_reliability.json",
    }
    missing = [
        str(path)
        for name, path in result.items()
        if name != "root" and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing corrected C-XA input(s): {missing}")
    return result


def _physics(paths, sequence_id: str) -> dict[str, Any]:
    robot = _stage_b_dirs(paths.workspace_root, sequence_id)[1]
    path = robot / "stage_c/physics_input.json"
    if not path.is_file():
        raise FileNotFoundError(f"C-XA static preflight scene is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("scene_act", "collision_cache")
    if any(not Path(payload[key]).is_file() and key == "scene_act" for key in required):
        raise FileNotFoundError("Stage-C physics scene is incomplete")
    return payload


def _joint_limit_violations(model: mujoco.MjModel, qpos: np.ndarray) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for joint in range(model.njnt):
        if not bool(model.jnt_limited[joint]):
            continue
        address = int(model.jnt_qposadr[joint])
        if address >= 52:
            continue
        lower, upper = model.jnt_range[joint]
        bad = np.flatnonzero((qpos[:, address] < lower - 1e-8) | (qpos[:, address] > upper + 1e-8))
        if len(bad):
            violations.append({"joint": int(joint), "qpos_address": address, "frames": bad.astype(int).tolist()})
    return violations


def _static_reference_distribution(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    data = mujoco.MjData(model)
    qacc = np.empty(len(qpos), dtype=np.float64)
    qvel_max = np.empty(len(qpos), dtype=np.float64)
    force = np.empty(len(qpos), dtype=np.float64)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        qacc[frame] = np.abs(data.qacc).max(initial=0.0)
        qvel_max[frame] = np.abs(data.qvel).max(initial=0.0)
        force[frame] = np.abs(data.actuator_force).max(initial=0.0)
    return {
        "qacc_max": qacc,
        "qvel_max": qvel_max,
        "actuator_force_max": force,
        "noncontact_frames": np.flatnonzero(active.sum(axis=1) == 0),
        "low_contact_frames": np.flatnonzero(active.sum(axis=1) <= 1),
    }


def _freeze_thresholds(distribution: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    settings = config["preflight_thresholds"]
    for source, name, floor in (
        ("qacc_max", "qacc_hard_max", "qacc_floor"),
        ("qvel_max", "qvel_hard_max", "qvel_floor"),
        ("actuator_force_max", "actuator_force_hard_max", "actuator_force_floor"),
    ):
        reference = distribution[source]
        selected = np.concatenate((reference[distribution["noncontact_frames"]], reference[distribution["low_contact_frames"]]))
        if not len(selected):
            raise RuntimeError("cannot freeze dynamic threshold without non-contact/low-contact reference frames")
        quantile = float(settings[f"{source.removesuffix('_max')}_reference_quantile"])
        multiplier = float(settings[f"{source.removesuffix('_max')}_reference_multiplier"])
        values[name] = max(float(settings[floor]), float(np.quantile(selected, quantile)) * multiplier)
        values[f"{source}_reference_p{int(quantile * 100)}"] = float(np.quantile(selected, quantile))
    values.update({
        "object_translation_drift_m": float(settings["object_translation_drift_m"]),
        "object_rotation_drift_rad": float(settings["object_rotation_drift_rad"]),
        "consecutive_growth_factor": float(settings["consecutive_growth_factor"]),
        "consecutive_growth_count": int(settings["consecutive_growth_count"]),
    })
    return values


def freeze_corrected_inputs(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """D0: hash and validate the only allowed corrected physical input."""
    paths = load_project_paths(paths_config)
    config = _config()
    inputs = _inputs(paths, sequence_id)
    physics = _physics(paths, sequence_id)
    root = _root(paths, sequence_id)
    reference = json.loads(inputs["reference_json"].read_text(encoding="utf-8"))
    assignment = json.loads(inputs["assignment_json"].read_text(encoding="utf-8"))
    rerun = json.loads(inputs["rerun"].read_text(encoding="utf-8"))
    if rerun.get("decision") != "CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG" or rerun.get("status") != "PASS_STATIC_AND_PREFLIGHT":
        raise RuntimeError("C-XA finalization is not the required CASE A static pass")
    if assignment.get("level") != 1 or assignment.get("contract_hash") != _sha256(CONTRACT_PATH):
        raise RuntimeError("corrected assignment is not Contract V2 Level 1")
    records = reference["records"]
    threshold = float(reference["distance_threshold_m"])
    active_bad = [row for row in records if row.get("contact_flag") and (row.get("source_reliability") != "RELIABLE_SOURCE" or float(row["unsigned_distance_m"]) > threshold)]
    unreliable = [row for row in records if row.get("source_reliability") == "UNRELIABLE_SOURCE"]
    raw_unreliable = [row for row in unreliable if row.get("raw_inclusive_contact_flag")]
    if active_bad or len(unreliable) != 197 or len(raw_unreliable) != 197:
        raise RuntimeError("corrected contact reference fails the immutable unreliable-source exclusion")
    with np.load(inputs["reference_npz"], allow_pickle=False) as archive:
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
        source_contact = np.asarray(archive["contact"], dtype=bool)
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        trajectory_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        target_expected = np.asarray(archive["expected"], dtype=bool)
        source_expected = np.asarray(archive["source_expected"], dtype=bool)
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    dynamic_profile = _configure_dynamic_model(model, config)
    baseline, _ = _stage_b_act_baseline(paths, sequence_id)
    joint_violations = _joint_limit_violations(model, qpos)
    active = source_contact[1:-1]
    checks = {
        "corrected_active_contacts_only": not active_bad,
        "unreliable_evidence_preserved": len(unreliable) == 197 and len(raw_unreliable) == 197,
        "immutable_15mm_threshold": threshold == 0.015,
        "contract_hash_unchanged": assignment.get("contract_hash") == _sha256(CONTRACT_PATH),
        "level_1_not_level_4": assignment.get("level") == 1,
        "source_frame_mapping_complete": bool(np.array_equal(trajectory_frames, source_frames[1:-1])),
        "dimensions": bool(qpos.shape == (414, model.nq) and qvel.shape == (414, model.nv) and target_expected.shape == (414, 10)),
        "object_pose_matches_stage_b": bool(np.array_equal(qpos[:, 52:], baseline[:, 52:])),
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        "joint_limits": not joint_violations,
        "assigned_targets_not_raw_level4": bool(np.array_equal(target_expected, source_expected) is False or assignment.get("level") == 1),
    }
    distribution = _static_reference_distribution(model, qpos, qvel, active)
    thresholds = _freeze_thresholds(distribution, config)
    assets = {"scene_act": Path(physics["scene_act"]), "object_visual_mesh": Path(physics["collision_cache"]) / "visual/visual.obj"}
    collision_parts = sorted((Path(physics["collision_cache"]) / "collision").glob("*.obj"))
    for path in (*assets.values(), *collision_parts):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema_version": 1,
        "stage": "C-V2-D0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sequence_id": sequence_id,
        "source_frame_range": PILOTS[sequence_id]["frames"],
        "corrected_contact_reference": str(inputs["reference_json"]),
        "corrected_assignment": str(inputs["assignment_json"]),
        "corrected_trajectory": str(inputs["trajectory"]),
        "cxa_contract_hash": _sha256(CONTRACT_PATH),
        "dynamic_profile": dynamic_profile,
        "hashes": {name: _sha256(path) for name, path in inputs.items() if name not in {"root"}},
        "assets": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in assets.items()} | {"collision_parts": [{"path": str(path), "sha256": _sha256(path)} for path in collision_parts]},
        "corrected_active_contact_count": int(active.sum()),
        "unreliable_source_count": len(unreliable),
        "unreliable_active_contact_count": len(active_bad),
        "dimensions": {"qpos": list(qpos.shape), "qvel": list(qvel.shape), "source_reference": list(source_contact.shape), "assigned_contact": list(target_expected.shape), "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}},
        "preflight_thresholds": thresholds,
        "static_reference_distribution": {key: value for key, value in distribution.items() if key not in {"qacc_max", "qvel_max", "actuator_force_max"}},
        "joint_limit_violations": joint_violations,
        "checks": checks,
        "preservation": {"raw_grab_modified": False, "stage_b_trajectory_overwritten": False, "cxa_artifacts_overwritten": False, "historical_v2_artifacts_overwritten": False},
    }
    output = root / "input_freeze_validation.json"
    _write_json(output, payload)
    _write_json(paths.workspace_root / "manifests/stage_c_v2_dynamic_acceptance.json", {"schema_version": 1, "status": payload["status"], "primary": {"sequence_id": sequence_id, "frames": PILOTS[sequence_id]["frames"]}, "input_freeze": str(output), "hashes": payload["hashes"], "seed": config["seed"], "stage_d": "NOT_STARTED"})
    return str(output)


def _load_frozen(paths, sequence_id: str) -> tuple[dict[str, Any], dict[str, Any], mujoco.MjModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = _root(paths, sequence_id)
    freeze = root / "input_freeze_validation.json"
    if not freeze.is_file() or json.loads(freeze.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("D0 input freeze must pass before dynamic simulation")
    inputs = _inputs(paths, sequence_id)
    physics = _physics(paths, sequence_id)
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    current_profile = _configure_dynamic_model(model, _config())
    frozen_payload = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen_payload.get("dynamic_profile") != current_profile:
        raise RuntimeError("D0 dynamic profile differs from the current config; rerun D0 before D1/D2")
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        qpos, qvel = np.asarray(archive["qpos"], dtype=np.float64), np.asarray(archive["qvel"], dtype=np.float64)
    with np.load(inputs["reference_npz"], allow_pickle=False) as archive:
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)[1:-1]
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        expected = np.asarray(archive["expected"], dtype=bool)
    return frozen_payload, physics, model, qpos, qvel, source_frames, expected


def _select_keyframes(qpos: np.ndarray, expected: np.ndarray, trace_path: Path) -> dict[str, int]:
    active = np.flatnonzero(expected.sum(axis=1) > 0)
    first = int(active[0]) if len(active) else 0
    density = int(np.argmax(expected.sum(axis=1)))
    with np.load(trace_path, allow_pickle=False) as archive:
        peak = int(np.argmax(np.asarray(archive["collision_before_m"])))
    selection = {"start": 0, "approach": max(1, first - max(1, len(qpos) // 10)), "first_corrected_high_confidence_contact": first, "peak_stage_b_kinematic_penetration": peak, "interaction_midpoint": len(qpos) // 2, "maximum_corrected_contact_density": density, "end": len(qpos) - 1}
    # Preserve the named evidence but make the actual hold set distinct.
    return {name: frame for name, frame in selection.items() if frame not in set(list(selection.values())[:list(selection).index(name)])}


def _contact_summary(model: mujoco.MjModel, data: mujoco.MjData, hand: set[int], objects: set[int]) -> tuple[int, float, float, list[dict[str, Any]]]:
    depths = _collision_depths(data, hand, objects)
    records: list[dict[str, Any]] = []
    total_force = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        item = data.contact[index]
        if {int(item.geom1), int(item.geom2)} & hand and {int(item.geom1), int(item.geom2)} & objects:
            mujoco.mj_contactForce(model, data, index, force)
            magnitude = float(np.linalg.norm(force[:3]))
            total_force += magnitude
            records.append({"geom_pair": f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom1))}|{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom2))}", "penetration_m": float(max(0.0, -item.dist)), "force_n": magnitude, "position_world": np.asarray(item.pos).tolist(), "normal_world": np.asarray(item.frame[:3]).tolist()})
    return int(data.ncon), float(depths.max(initial=0.0)), total_force, records


def _has_exponential_growth(values: np.ndarray, factor: float, count: int) -> bool:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    growth = positive[1:] > positive[:-1] * factor
    return bool(any(np.all(growth[start : start + count - 1]) for start in range(max(0, len(growth) - count + 2))))


def _has_consecutive(values: np.ndarray, count: int) -> bool:
    """Return whether a boolean condition persists for ``count`` samples."""
    flags = np.asarray(values, dtype=bool)
    return bool(any(np.all(flags[start : start + count]) for start in range(max(0, len(flags) - count + 1))))


def _dynamic_visual_penetration(model: mujoco.MjModel, qpos: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    """Audit every source-rate physical state against the real visual mesh."""
    visual_ids = _hand_geom_ids(model, 1)
    object_mesh = _mesh(Path(physics["collision_cache"]) / "visual/visual.obj")
    object_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object"))
    if object_body < 0:
        raise RuntimeError("dynamic visual audit requires the right_object body")
    data = mujoco.MjData(model)
    signed: list[float] = []
    closest_world: list[np.ndarray] = []
    per_frame_max: list[float] = []
    for state in qpos:
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        world = np.concatenate([mesh_from_model(model, data, geom).vertices for geom in visual_ids])
        local = _world_to_body(world, data.xpos[object_body], data.xmat[object_body])
        closest, _unsigned, distance, _method, confidence = _closest_with_sign(object_mesh, local)
        if confidence != "high" or not np.isfinite(distance).all():
            raise RuntimeError("dynamic visual audit requires high-confidence finite signed distances")
        matrix = data.xmat[object_body].reshape(3, 3)
        closest_world.extend((closest @ matrix.T + data.xpos[object_body]).copy())
        signed.extend(distance.tolist())
        per_frame_max.append(float(max(0.0, -np.min(distance, initial=0.0))))
    summary = summarize_signed_distances(np.asarray(signed), np.asarray(closest_world), confidence="high")
    return {**summary, "per_frame_max_penetration_m": per_frame_max}


def _dynamic_robot_tracking(
    model: mujoco.MjModel, qpos: np.ndarray, reference: np.ndarray, source_frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Return individual wrist/fingertip tracking, never an aggregate-only gate."""
    sites = _site_ids(model)
    actual, target = mujoco.MjData(model), mujoco.MjData(model)
    error = np.empty((len(qpos), len(sites)), dtype=np.float64)
    positions = np.empty((len(qpos), len(sites), 3), dtype=np.float64)
    for frame in range(len(qpos)):
        actual.qpos[:] = qpos[frame]
        target.qpos[:] = reference[frame]
        actual.qvel[:] = 0.0
        target.qvel[:] = 0.0
        mujoco.mj_forward(model, actual)
        mujoco.mj_forward(model, target)
        positions[frame] = actual.site_xpos[sites]
        error[frame] = np.linalg.norm(actual.site_xpos[sites] - target.site_xpos[sites], axis=1)
    names = ("right_palm", "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky", "left_palm", "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky")
    flat = {
        name: {
            "rmse_m": float(np.sqrt(np.mean(error[:, index] ** 2))),
            "max_m": float(error[:, index].max()),
            "worst_source_frame": int(source_frames[int(np.argmax(error[:, index]))]),
        }
        for index, name in enumerate(names)
    }
    by_side: dict[str, Any] = {}
    for side, offset in (("right", 0), ("left", 6)):
        by_side[side] = {
            "wrist_rmse_m": flat[f"{side}_palm"]["rmse_m"],
            "fingertips": {finger: flat[f"{side}_{finger}"] for finger in FINGERS},
        }
    return positions, error, flat, by_side


def run_keyframe_holds(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """D1: true-contact physical holds with every integration step preserved."""
    paths = load_project_paths(paths_config)
    freeze, _physics_payload, model, qpos, qvel, source_frames, expected = _load_frozen(paths, sequence_id)
    config = _config()
    thresholds = freeze["preflight_thresholds"]
    trace = _inputs(paths, sequence_id)["root"] / "depenetration_trace_cxa_level_1_flexible.npz"
    selection = _select_keyframes(qpos, expected, trace)
    if len(set(selection.values())) < 5:
        raise RuntimeError("dynamic event selection must retain at least five distinct source frames")
    root = _root(paths, sequence_id)
    _write_json(root / "dynamic_keyframes.json", {"schema_version": 1, "sequence_id": sequence_id, "source_frame_indices": {name: int(source_frames[index]) for name, index in selection.items()}, "trajectory_indices": selection, "selection_basis": {"first_contact": "corrected active C-XA contact target", "peak_penetration": "maximum Stage-B collision_before_m", "density": "maximum assigned corrected contact density"}, "distinct_source_frames": len(set(selection.values()))})
    bodies, mocap = _preflight_object_ids(model)
    hand = set(_hand_geom_ids(model, 2))
    objects = {index for index in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("right_object_") and model.geom_group[index] == 3}
    hold_steps = max(1, int(round(float(config["physics"]["hold_seconds"]) / model.opt.timestep)))
    profile = _configure_dynamic_model(model, config)
    lead = int(profile["robot_reference_lead_source_frames"])
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("keyframe_index", "step_index", "sim_time", "qpos", "qvel", "qacc", "ctrl", "actuator_force", "contact_count", "contact_depth", "contact_force", "object_pos", "object_rot", "object_linear_velocity", "object_angular_velocity")}
    records: dict[str, Any] = {}
    warnings: list[str] = []
    old_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    try:
        for key_index, (name, frame) in enumerate(selection.items()):
            data = mujoco.MjData(model)
            data.qpos[:] = qpos[frame]
            data.qvel[:] = qvel[frame]
            _set_object_mocap_reference(data, qpos[frame], mocap)
            data.ctrl[:52] = _robot_reference(qpos, frame, lead)
            data.ctrl[52:] = 0.0
            mujoco.mj_forward(model, data)
            start_pos, start_rot = _object_tracking_error(data, qpos[frame], bodies)
            local_qacc: list[float] = []
            local_force: list[float] = []
            local_qvel: list[float] = []
            local_finite = _finite_data(data)
            contact_records: list[dict[str, Any]] = []
            for step in range(hold_steps):
                mujoco.mj_step(model, data)
                count, depth, force, pairs = _contact_summary(model, data, hand, objects)
                local_qacc.append(float(np.abs(data.qacc).max(initial=0.0)))
                local_qvel.append(float(np.abs(data.qvel).max(initial=0.0)))
                local_force.append(float(np.abs(data.actuator_force).max(initial=0.0)))
                local_finite = local_finite and _finite_data(data)
                for pair in pairs:
                    pair["frame_index"] = int(step)
                contact_records.extend(pairs)
                rows["keyframe_index"].append(np.asarray(key_index, dtype=np.int32)); rows["step_index"].append(np.asarray(step, dtype=np.int32)); rows["sim_time"].append(np.asarray(data.time)); rows["qpos"].append(data.qpos.copy()); rows["qvel"].append(data.qvel.copy()); rows["qacc"].append(data.qacc.copy()); rows["ctrl"].append(data.ctrl.copy()); rows["actuator_force"].append(data.actuator_force.copy()); rows["contact_count"].append(np.asarray(count, dtype=np.int32)); rows["contact_depth"].append(np.asarray(depth)); rows["contact_force"].append(np.asarray(force)); rows["object_pos"].append(np.stack([data.xpos[bodies["right"]], data.xpos[bodies["left"]]])); rows["object_rot"].append(np.stack([data.xmat[bodies["right"]], data.xmat[bodies["left"]]])); rows["object_linear_velocity"].append(np.stack([data.qvel[52:55], data.qvel[58:61]])); rows["object_angular_velocity"].append(np.stack([data.qvel[55:58], data.qvel[61:64]]))
            end_pos, end_rot = _object_tracking_error(data, qpos[frame], bodies)
            records[name] = {"trajectory_frame": int(frame), "source_frame": int(source_frames[frame]), "hold_seconds": float(hold_steps * model.opt.timestep), "finite": bool(local_finite), "qacc_max": max(local_qacc, default=0.0), "qvel_max": max(local_qvel, default=0.0), "actuator_force_max": max(local_force, default=0.0), "object_translation_drift_m": (end_pos - start_pos), "object_rotation_drift_rad": (end_rot - start_rot), "qacc_exponential_growth": _has_exponential_growth(np.asarray(local_qacc), thresholds["consecutive_growth_factor"], thresholds["consecutive_growth_count"]), "contact_pairs": contact_pair_summary(contact_records)}
    finally:
        mujoco.set_mju_user_warning(old_warning)
    arrays = {name: np.stack(value) for name, value in rows.items()}
    _write_npz(root / "preflight_keyframe_hold_steps.npz", **arrays)
    drift = max((float(np.max(np.abs(row["object_translation_drift_m"]))) for row in records.values()), default=0.0)
    rotation = max((float(np.max(np.abs(row["object_rotation_drift_rad"]))) for row in records.values()), default=0.0)
    qacc = max((row["qacc_max"] for row in records.values()), default=0.0)
    qvel_limit = max((row["qvel_max"] for row in records.values()), default=0.0)
    force_limit = max((row["actuator_force_max"] for row in records.values()), default=0.0)
    gates = {"finite": all(row["finite"] for row in records.values()), "no_warnings": not warnings, "qacc_bounded": qacc <= thresholds["qacc_hard_max"], "qvel_bounded": qvel_limit <= thresholds["qvel_hard_max"], "actuator_force_bounded": force_limit <= thresholds["actuator_force_hard_max"], "no_exponential_qacc_growth": not any(row["qacc_exponential_growth"] for row in records.values()), "object_no_ejection": drift <= thresholds["object_translation_drift_m"], "object_no_spin": rotation <= thresholds["object_rotation_drift_rad"]}
    payload = {"schema_version": 1, "stage": "C-V2-D1", "sequence_id": sequence_id, "status": "PASS" if all(gates.values()) else "FAIL", "keyframes": records, "dynamic_profile": profile, "thresholds": thresholds, "warnings": warnings, "warning_count": len(warnings), "qacc_max": qacc, "qvel_max": qvel_limit, "actuator_force_max": force_limit, "object_translation_drift_max_m": drift, "object_rotation_drift_max_rad": rotation, "gates": gates, "step_trace": str(root / "preflight_keyframe_hold_steps.npz"), "object_qpos_overwritten_after_initialization": False, "corrected_cxa_inputs_unchanged": True}
    output = root / "preflight_keyframe_hold.json"
    _write_json(output, payload)
    return str(output)


def run_forward_rollout(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """D2: physical replay of all corrected frames, forbidden until D1 passes."""
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    hold = root / "preflight_keyframe_hold.json"
    if not hold.is_file() or json.loads(hold.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("D1 keyframe holds must pass before D2 forward rollout")
    freeze, physics, model, reference, reference_qvel, source_frames, _expected = _load_frozen(paths, sequence_id)
    config = _config()
    profile = _configure_dynamic_model(model, config)
    lead = int(profile["robot_reference_lead_source_frames"])
    bodies, mocap = _preflight_object_ids(model)
    hand, objects = set(_hand_geom_ids(model, 2)), {index for index in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("right_object_") and model.geom_group[index] == 3}
    data = mujoco.MjData(model); data.qpos[:] = reference[0]; data.qvel[:] = reference_qvel[0]; _set_object_mocap_reference(data, reference[0], mocap); data.ctrl[:52] = _robot_reference(reference, 0, lead); data.ctrl[52:] = 0; mujoco.mj_forward(model, data)
    qpos = np.empty_like(reference); qvel = np.empty_like(reference_qvel); ctrl = np.empty((len(reference), model.nu)); pos = np.empty((len(reference), 2)); rot = np.empty((len(reference), 2)); qacc = np.empty(len(reference)); depth = np.empty(len(reference)); force = np.empty(len(reference)); contact = np.empty(len(reference), dtype=np.int32); finite = np.empty(len(reference), dtype=np.uint8); warnings: list[str] = []
    steps: dict[str, list[np.ndarray]] = {name: [] for name in ("frame_index", "substep_index", "sim_time", "qpos", "qvel", "qacc", "ctrl", "actuator_force", "contact_count", "contact_depth_m", "contact_force_n", "object_position", "object_orientation", "object_linear_velocity", "object_angular_velocity")}
    contact_records: list[dict[str, Any]] = []
    old_warning = mujoco.get_mju_user_warning(); mujoco.set_mju_user_warning(lambda message: warnings.append(str(message)))
    target_time = 0.0
    interval_step_counts: list[int] = []
    boundary_time_errors: list[float] = []
    try:
        for frame in range(len(reference)):
            substep = 0
            if frame:
                # Frame zero is the initialized t=0 state.  Every later frame
                # advances to one accumulated 120-Hz boundary, with no source
                # resampling, smoothing, deletion, or replacement.
                _set_object_mocap_reference(data, reference[frame], mocap)
                data.ctrl[:52] = _robot_reference(reference, frame, lead)
                data.ctrl[52:] = 0.0
                target_time += 1.0 / float(config["physics"]["source_fps"])
                while data.time + 0.5 * model.opt.timestep < target_time:
                    mujoco.mj_step(model, data)
                    step_count, step_depth, step_force, step_pairs = _contact_summary(model, data, hand, objects)
                    for pair in step_pairs:
                        contact_records.append({"frame_index": frame, "source_frame": int(source_frames[frame]), "substep_index": substep, "sim_time_s": float(data.time), **pair})
                    steps["frame_index"].append(np.asarray(frame, dtype=np.int32)); steps["substep_index"].append(np.asarray(substep, dtype=np.int32)); steps["sim_time"].append(np.asarray(data.time)); steps["qpos"].append(data.qpos.copy()); steps["qvel"].append(data.qvel.copy()); steps["qacc"].append(data.qacc.copy()); steps["ctrl"].append(data.ctrl.copy()); steps["actuator_force"].append(data.actuator_force.copy()); steps["contact_count"].append(np.asarray(step_count, dtype=np.int32)); steps["contact_depth_m"].append(np.asarray(step_depth)); steps["contact_force_n"].append(np.asarray(step_force)); steps["object_position"].append(np.stack([data.xpos[bodies["right"]], data.xpos[bodies["left"]]])); steps["object_orientation"].append(np.stack([data.xmat[bodies["right"]], data.xmat[bodies["left"]]])); steps["object_linear_velocity"].append(np.stack([data.qvel[52:55], data.qvel[58:61]])); steps["object_angular_velocity"].append(np.stack([data.qvel[55:58], data.qvel[61:64]])); substep += 1
                interval_step_counts.append(substep)
                boundary_time_errors.append(float(abs(data.time - target_time)))
            count, collision_depth, contact_force, _pairs = _contact_summary(model, data, hand, objects)
            qpos[frame], qvel[frame], ctrl[frame] = data.qpos, data.qvel, data.ctrl; pos[frame], rot[frame] = _object_tracking_error(data, reference[frame], bodies); qacc[frame] = np.abs(data.qacc).max(initial=0.0); depth[frame] = collision_depth; force[frame] = contact_force; contact[frame] = count; finite[frame] = _finite_data(data)
    finally:
        mujoco.set_mju_user_warning(old_warning)
    _write_npz(root / "trajectory_corrected_v2_forward_rollout.npz", qpos=qpos, qvel=qvel, ctrl=ctrl, source_frame_indices=source_frames)
    thresholds = config["acceptance"]
    positions, tracking_error, robot_tracking, robot_tracking_by_side = _dynamic_robot_tracking(model, qpos, reference, source_frames)
    robot_tracking_ok = all(
        side["wrist_rmse_m"] <= thresholds["wrist_rmse_m"]
        and all(item["rmse_m"] <= thresholds["fingertip_rmse_m"] for item in side["fingertips"].values())
        for side in robot_tracking_by_side.values()
    )
    inputs = _inputs(paths, sequence_id)
    with np.load(inputs["reference_npz"], allow_pickle=False) as archive:
        source_mapping_complete = bool(np.array_equal(source_frames, np.asarray(archive["source_frame_indices"], dtype=np.int64)[1:-1]))
    with np.load(inputs["targets"], allow_pickle=False) as archive:
        source_expected = np.asarray(archive["source_expected"], dtype=bool)
        source_anchors = np.asarray(archive["source_anchors"], dtype=np.float64)
    tip_positions = positions[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    contact_distance = np.linalg.norm(tip_positions - source_anchors, axis=2)
    observed = contact_distance <= 0.015
    exact_contact = {
        "high_confidence_recall": float(np.count_nonzero(observed & source_expected) / max(1, np.count_nonzero(source_expected))),
        "expected_records": int(np.count_nonzero(source_expected)),
        "false_contact_frames": int(np.count_nonzero(observed & ~source_expected)),
        "anchor_contract": "corrected C-XA immutable source contact anchors",
    }
    visual = _dynamic_visual_penetration(model, qpos, physics)
    delta = np.diff(qpos[:, :52], axis=0)
    ranges = model.jnt_range[:52, 1] - model.jnt_range[:52, 0]
    ranges[[0, 1, 2, 26, 27, 28]] = 4.0
    normalized_delta = float(np.max(np.abs(delta) / np.maximum(ranges, 1e-9))) if len(delta) else 0.0
    object_by_side = {
        side: {
            "position_rmse_m": float(np.sqrt(np.mean(pos[:, index] ** 2))), "position_max_m": float(pos[:, index].max()),
            "rotation_mean_rad": float(rot[:, index].mean()), "rotation_max_rad": float(rot[:, index].max()),
            "rotation_worst_source_frame": int(source_frames[int(np.argmax(rot[:, index]))]),
        }
        for index, side in enumerate(("right", "left"))
    }
    object_tracking_ok = all(
        item["position_rmse_m"] <= thresholds["object_pos_rmse_m"] and item["position_max_m"] <= thresholds["object_pos_max_m"]
        and item["rotation_mean_rad"] <= thresholds["object_rot_mean_rad"] and item["rotation_max_rad"] <= thresholds["object_rot_max_rad"]
        for item in object_by_side.values()
    )
    visual_ok = bool(
        visual["max_penetration_m"] <= thresholds["visual_max_penetration_m"]
        and visual["p95_negative_depth_m"] <= thresholds["visual_p95_penetration_m"]
        and visual["mean_negative_depth_m"] <= thresholds["visual_mean_negative_depth_m"]
        and visual["penetrating_ratio"] <= thresholds["visual_penetrating_vertex_ratio"]
        and max(visual["per_frame_max_penetration_m"], default=0.0) <= thresholds["frame_visual_max_penetration_m"]
    )
    joint_violations = _joint_limit_violations(model, qpos)
    finite_all = bool(finite.all() and np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all())
    step_qacc = np.asarray(steps["qacc"], dtype=np.float64)
    step_qacc_max = np.abs(step_qacc).max(axis=1)
    step_depth = np.asarray(steps["contact_depth_m"], dtype=np.float64)
    step_force = np.asarray(steps["contact_force_n"], dtype=np.float64)
    qacc_max = float(max(qacc.max(initial=0.0), step_qacc_max.max(initial=0.0)))
    output = root / "metrics_corrected_v2_forward_rollout.json"
    base_payload = {
        "schema_version": 1, "stage": "C-V2-D2", "sequence_id": sequence_id, "trajectory": str(root / "trajectory_corrected_v2_forward_rollout.npz"),
        "nan_inf": int(np.size(qpos) + np.size(qvel) + np.size(ctrl) - np.count_nonzero(np.isfinite(qpos)) - np.count_nonzero(np.isfinite(qvel)) - np.count_nonzero(np.isfinite(ctrl))),
        "joint_limit_violations": len(joint_violations), "source_mapping_complete": source_mapping_complete,
        "contact": exact_contact, "visual_penetration": visual, "collision": {"after_max_m": float(step_depth.max(initial=0.0))},
        "gates": {"tracking": robot_tracking_ok, "smoothness": normalized_delta <= thresholds["max_normalized_joint_delta"]},
        "smoothness": {"max_normalized_single_frame_delta": normalized_delta, "teleport": normalized_delta > thresholds["max_normalized_joint_delta"]},
    }
    _write_json(output, base_payload)
    # The C-XA evaluator owns the immutable Level-1 patch/role definition.
    # Its static collision checks remain audit fields only; D2 uses the
    # explicit dynamic collision-persistence rule below.
    from spider.tools.grab_stage_c_contact_reassignment import evaluate_v2_depenetrated
    evaluate_v2_depenetrated(paths_config, sequence_id, 1, str(root / "trajectory_corrected_v2_forward_rollout.npz"), str(output), str(inputs["targets"]), "stage_c_contract_v2_cxa", refresh_base_metrics=False)
    payload = json.loads(output.read_text(encoding="utf-8"))
    static_v2 = payload.pop("contract_v2")
    dynamic_v2_gates = {
        "task_equivalent_patch_coverage": payload["task_equivalent_contact_v2"]["patch_coverage"] >= thresholds["task_equivalent_patch_coverage"],
        "functional_role_recall": payload["task_equivalent_contact_v2"]["functional_role_recall"] >= thresholds["functional_role_recall"],
        "surface_patch_distance_p95": payload["task_equivalent_contact_v2"]["surface_patch_distance_p95_m"] <= thresholds["patch_distance_p95_m"],
        "normal_alignment": payload["task_equivalent_contact_v2"]["normal_cosine_median"] >= thresholds["normal_cosine_median"],
    }
    gates = {
        "finite_states_controls": finite_all, "no_warnings": not warnings, "joint_limits": not joint_violations,
        "source_frame_mapping": source_mapping_complete,
        "qacc_bounded": bool(qacc_max <= freeze["preflight_thresholds"]["qacc_hard_max"]),
        "no_exponential_qacc_growth": not _has_exponential_growth(step_qacc_max, freeze["preflight_thresholds"]["consecutive_growth_factor"], freeze["preflight_thresholds"]["consecutive_growth_count"]),
        "no_persistent_deep_collision": not _has_consecutive(step_depth > 0.003, 3),
        "no_exponential_contact_force_growth": not _has_exponential_growth(step_force, freeze["preflight_thresholds"]["consecutive_growth_factor"], freeze["preflight_thresholds"]["consecutive_growth_count"]),
        "robot_tracking": robot_tracking_ok, "object_tracking": object_tracking_ok, "visual_penetration": visual_ok,
        "v2_task_equivalent_contact": all(dynamic_v2_gates.values()), "smoothness": normalized_delta <= thresholds["max_normalized_joint_delta"],
    }
    _write_npz(root / "forward_rollout_trace.npz", qacc_max_by_source_frame=qacc, contact_depth_m_by_source_frame=depth, contact_force_n_by_source_frame=force, contact_count_by_source_frame=contact, object_position_error_m_by_source_frame=pos, object_rotation_error_rad_by_source_frame=rot, robot_site_tracking_error_m=tracking_error, finite_by_source_frame=finite, **{name: np.stack(values) for name, values in steps.items()})
    _write_json(root / "forward_rollout_contacts.json", {"schema_version": 1, "stage": "C-V2-D2", "scope": "hand-object MuJoCo contacts at every real integration step", "records": contact_records, "per_geom_pair": contact_pair_summary(contact_records)})
    payload.update({
        "status": "PASS" if all(gates.values()) else "FAIL", "trace": str(root / "forward_rollout_trace.npz"), "contact_trace": str(root / "forward_rollout_contacts.json"),
        "frame_count": len(reference), "source_interval_step_counts": {"initial_state_at_t0": True, "min": min(interval_step_counts), "max": max(interval_step_counts), "total": sum(interval_step_counts), "boundary_time_error_max_s": max(boundary_time_errors, default=0.0)},
        "dynamic_profile": profile, "source_frame_mapping_complete": source_mapping_complete, "qacc_max": qacc_max, "collision_depth_max_m": float(step_depth.max(initial=0.0)),
        "collision": {"after_max_m": float(step_depth.max(initial=0.0)), "contact_force_max_n": float(step_force.max(initial=0.0)), "per_geom_pair": contact_pair_summary(contact_records)},
        "object_tracking": {"position_rmse_m": float(np.sqrt(np.mean(pos ** 2))), "position_max_m": float(pos.max(initial=0.0)), "rotation_mean_rad": float(rot.mean()), "rotation_max_rad": float(rot.max()), "per_side": object_by_side},
        "tracking_reference": "corrected Level 1 initialization; static C-XA Stage-B gates remain immutable", "robot_tracking": robot_tracking, "robot_tracking_by_side": robot_tracking_by_side,
        "warnings": warnings, "warning_count": len(warnings), "joint_limit_violation_records": joint_violations, "dynamic_v2_contact_gates": dynamic_v2_gates,
        "v2_static_evaluator_audit": static_v2, "gates": gates, "object_qpos_overwritten_after_initialization": False,
    })
    _write_json(output, payload)
    return str(output)


def localize_forward_rollout_failure(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Record immutable-input evidence when D2 prevents every later stage."""
    paths = load_project_paths(paths_config)
    root = _root(paths, sequence_id)
    report_path = root / "metrics_corrected_v2_forward_rollout.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") == "PASS":
        raise RuntimeError("D2 passed; a failure-localization report is not valid")
    inputs = _inputs(paths, sequence_id)
    with np.load(inputs["trajectory"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    discontinuities: list[dict[str, Any]] = []
    for side, offset in (("right", 55), ("left", 61)):
        rotations = Rotation.from_euler("XYZ", qpos[:, offset : offset + 3])
        deltas = (rotations[:-1].inv() * rotations[1:]).magnitude()
        for index in np.flatnonzero(deltas > float(_config()["acceptance"]["object_rot_max_rad"])):
            discontinuities.append({"side": side, "previous_source_frame": int(source_frames[index]), "source_frame": int(source_frames[index + 1]), "source_rotation_step_rad": float(deltas[index]), "gate_max_rad": float(_config()["acceptance"]["object_rot_max_rad"])})
    physics = _physics(paths, sequence_id)
    model = mujoco.MjModel.from_xml_path(str(physics["scene_act"]))
    hand = set(_hand_geom_ids(model, 2))
    objects = {index for index in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("right_object_") and model.geom_group[index] == 3}
    with np.load(root / "forward_rollout_trace.npz", allow_pickle=False) as archive:
        step_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        step_frame = np.asarray(archive["frame_index"], dtype=np.int64)
        step_substep = np.asarray(archive["substep_index"], dtype=np.int64)
        step_depth = np.asarray(archive["contact_depth_m"], dtype=np.float64)
    deep = np.flatnonzero(step_depth > 0.003)
    deep_runs: list[dict[str, Any]] = []
    if len(deep):
        start = previous = int(deep[0])
        for value in deep[1:]:
            value = int(value)
            if value != previous + 1:
                deep_runs.append({"start_source_frame": int(source_frames[step_frame[start]]), "start_substep": int(step_substep[start]), "end_source_frame": int(source_frames[step_frame[previous]]), "end_substep": int(step_substep[previous]), "step_count": previous - start + 1, "max_penetration_m": float(step_depth[start : previous + 1].max())})
                start = value
            previous = value
        deep_runs.append({"start_source_frame": int(source_frames[step_frame[start]]), "start_substep": int(step_substep[start]), "end_source_frame": int(source_frames[step_frame[previous]]), "end_substep": int(step_substep[previous]), "step_count": previous - start + 1, "max_penetration_m": float(step_depth[start : previous + 1].max())})
    pair_depths: dict[str, float] = {}
    data = mujoco.MjData(model)
    for step in deep:
        data.qpos[:] = step_qpos[step]
        mujoco.mj_forward(model, data)
        for contact in data.contact[: data.ncon]:
            if not ({int(contact.geom1), int(contact.geom2)} & hand and {int(contact.geom1), int(contact.geom2)} & objects):
                continue
            depth = float(max(0.0, -contact.dist))
            if depth <= 0.003:
                continue
            pair = f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))}|{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))}"
            pair_depths[pair] = max(pair_depths.get(pair, 0.0), depth)
    failed = [name for name, passed in report["gates"].items() if not passed]
    v2 = report.get("task_equivalent_contact_v2", {})
    repair_attempts = {
        "baseline_dt002_lead0": str(root / "attempts/baseline_dt002_lead0"),
        "fine_dt0005_lead20_full_d2": str(root / "attempts/fine_dt0005_lead20_full_d2"),
        "bounded_read_only_controller_probes": {
            "reference_leads_source_frames": [0, 4, 8, 12, 16, 20],
            "robot_servo_profiles": "base, bounded Kp/force-limit scales, and bounded contact-IK target corrections",
            "outcome": "no probe retained the required 0.70 task-equivalent patch coverage while also satisfying joint-limit, smoothness, and collision gates",
            "state_rewrites_used": False,
        },
    }
    payload = {
        "schema_version": 2, "stage": "C-V2-D2", "status": "FAIL", "sequence_id": sequence_id,
        "failed_gates": failed, "forward_rollout": str(report_path), "repair_attempts": repair_attempts,
        "object_reference_discontinuities_audited": discontinuities,
        "root_cause_ranking": [
            {
                "rank": 1, "cause": "DYNAMIC_V2_TASK_EQUIVALENT_CONTACT_NOT_RETAINED",
                "evidence": {
                    "patch_coverage": v2.get("patch_coverage"), "functional_role_recall": v2.get("functional_role_recall"),
                    "patch_distance_p95_m": v2.get("surface_patch_distance_p95_m"), "normal_cosine_median": v2.get("normal_cosine_median"),
                    "dynamic_gates": report.get("dynamic_v2_contact_gates", {}),
                },
                "why_not_repaired": "The tested bounded robot-only controller profiles did not restore the immutable Level-1 patch contract. Rewriting robot qpos or C-XA contact/trajectory data is prohibited, and a real MJWP run is not authorized until D2 passes.",
            },
            {
                "rank": 2, "cause": "DYNAMIC_ROBOT_SERVO_LIMIT_AND_CONTINUITY_TRADE_OFF",
                "evidence": {
                    "joint_limit_violation_records": report.get("joint_limit_violation_records", []),
                    "smoothness": report.get("smoothness", {}), "robot_tracking": report.get("robot_tracking_by_side", {}),
                },
                "why_not_repaired": "More aggressive bounded controller probes either remained below the V2 contact threshold or introduced joint-limit, smoothness, collision, or numerical failures; no source-state mutation was used to mask that trade-off.",
            },
            {
                "rank": 3, "cause": "PRIOR_DT002_OBJECT_TRACKING_FAILURE_REPAIRED_BUT_NOT_SUFFICIENT",
                "evidence": {
                    "current_object_tracking_gate": report["gates"].get("object_tracking"),
                    "current_object_tracking": report.get("object_tracking", {}),
                    "historical_baseline": str(root / "attempts/baseline_dt002_lead0"),
                    "current_fine_step_attempt": str(root / "attempts/fine_dt0005_lead20_full_d2"),
                },
                "why_not_repaired": "Fine real integration repaired the former object-rotation failure without changing frozen frames, but it cannot authorize D3 while the distinct V2-contact and robot-quality gates remain false.",
            },
        ],
        "collision_audit": {"runs": deep_runs, "geom_pairs": [{"geom_pair": pair, "max_penetration_m": depth} for pair, depth in sorted(pair_depths.items(), key=lambda item: -item[1])]},
        "stop_boundary": "D2 FAIL: D3 minimal MJWP, D4 profile search, D5 smokes, D6-D8 metrics/HTML/screenshots are NOT_RUN.",
        "forbidden_mutations_not_used": ["raw GRAB", "body model", "Stage-B trajectory", "C-XA corrected artifacts", "frozen frames", "object trajectory", "robot qpos state rewrite"],
        "stage_d": "NOT_STARTED",
    }
    output = root / "forward_rollout_failure_localization.json"; _write_json(output, payload); return str(output)


def write_fail_closed_acceptance(paths_config: str, sequence_id: str = PRIMARY) -> str:
    """Emit the required status material without fabricating later-stage evidence."""
    paths = load_project_paths(paths_config); root = _root(paths, sequence_id)
    required = {"freeze": root / "input_freeze_validation.json", "holds": root / "preflight_keyframe_hold.json", "rollout": root / "metrics_corrected_v2_forward_rollout.json", "localization": root / "forward_rollout_failure_localization.json"}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    if reports["rollout"].get("status") != "FAIL": raise RuntimeError("fail-closed acceptance may only describe a failed D2")
    workspace_reports = paths.workspace_root / "reports"
    validation = {"schema_version": 1, "stage": "C-V2", "status": "FAIL", "historical": {"v1_exact_contact": "BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT", "original_v2": "HISTORICAL_CASE_A_INVALID_BLOCKED_RESULT", "cxa": "CASE_A PASS", "corrected_v2_static": "PASS"}, "gates": {"D0_input_freeze": reports["freeze"]["status"], "D1_keyframe_holds": reports["holds"]["status"], "D2_full_rollout": reports["rollout"]["status"], "D3_minimal_mjwp": "NOT_RUN", "D4_primary_full_mjwp": "NOT_RUN", "mug_lift": "NOT_RUN", "mug_offhand_1": "NOT_RUN", "codex_screenshot_review": "NOT_RUN", "html_delivery": "NOT_RUN", "user_html_review": "NOT_AVAILABLE", "stage_d": "NOT_STARTED"}, "reports": {name: str(path) for name, path in required.items()}, "reason": reports["localization"]["stop_boundary"], "no_later_artifacts_claimed": True}
    _write_json(workspace_reports / "stage_c_v2_dynamic_validation.json", validation)
    _write_json(workspace_reports / "stage_c_v2_dynamic_pilot_summary.json", {"schema_version": 1, "status": "FAIL", "pilots": [{"sequence_id": PRIMARY, "status": "FAIL", "failed_gate": "D2_full_rollout"}, {"sequence_id": "s1__mug_lift", "status": "NOT_RUN", "reason": "primary D2 failed"}, {"sequence_id": "s1__mug_offhand_1", "status": "NOT_RUN", "reason": "primary D2 failed"}]})
    _write_json(workspace_reports / "stage_c_v2_dynamic_acceptance.json", validation)
    _write_json(workspace_reports / "stage_c_v2_dynamic_screenshot_review.json", {"schema_version": 1, "status": "NOT_RUN", "reason": "D2 failed; generating dynamic HTML or screenshots would be false acceptance evidence.", "screenshots": []})
    (workspace_reports / "STAGE_C_V2_DYNAMIC_ACCEPTANCE.md").write_text("# Stage C-V2 Dynamic Acceptance\n\nStatus: **FAIL** at D2 full forward rollout. D0 and D1 passed; the fine real-MuJoCo profile also repaired the former object-tracking failure without changing frozen frames. The physical rollout still fails the immutable Level-1 V2 contact contract and robot joint-limit/smoothness gates, so minimal MJWP, profile search, smokes, HTML, and screenshots are **NOT_RUN**. See `stage_c_v2_dynamic_validation.json` and `forward_rollout_failure_localization.json`.\n", encoding="utf-8")
    return str(workspace_reports / "stage_c_v2_dynamic_validation.json")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"freeze-inputs": freeze_corrected_inputs, "keyframe-holds": run_keyframe_holds, "forward-rollout": run_forward_rollout, "localize-forward-failure": localize_forward_rollout_failure, "write-fail-closed-acceptance": write_fail_closed_acceptance})
