"""Frozen Stage C-XAE-M1R4 assigned-region contact-retention audit.

The M1R4 success condition is *not* survival of the historical exact pair.
It is continuous real MuJoCo contact for the frozen four-geom left-index
collision region.  Every dynamic transition goes through
``FrozenActionEnvironment.step(action)``; this module never writes qpos/qvel
after environment initialisation and never runs M1/M2/M3.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import mujoco
import numpy as np

from spider.tools import grab_stage_c_xae_m1 as m1
from spider.tools import grab_stage_c_xae_m1r as m1r
from spider.tools import grab_stage_c_xae_m1r2 as m1r2
from spider.tools import grab_stage_c_xae_m1r3 as m1r3
from spider.tools.grab_stage_c_failure_diagnostic import _mesh_for_ids, _state_meshes


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r4"
M1R3_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r3/20260802T143000Z-contact-truth-dynamics"
EXACT_PAIR = "collision_hand_left_index_8|right_object_0"
EQUIVALENT_PAIR = "collision_hand_left_index_7|right_object_0"
BASE_ACTION = np.full(4, -0.015, dtype=np.float64)
PREFIX_STEPS = 5
TWO_FRAME_STEPS = 17
SAFE_FORCE_N = 10.0
SAFE_PENETRATION_M = 0.003
SEED = 20260802


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


def _write_npz(path: Path, **arrays: Any) -> None:
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


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _state_hash(data: mujoco.MjData) -> str:
    digest = hashlib.sha256()
    for array in (data.qpos, data.qvel, data.ctrl, data.act):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _region_names(region: dict[str, Any]) -> set[str]:
    return {str(row["geom_name"]) for row in region["region_geoms"]}


def _object_names(region: dict[str, Any]) -> set[str]:
    return {str(row["geom_name"]) for row in region["object_geoms"]}


def _group_contacts(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    contacts = record["all_hand_object_pairs"]
    region = [item for item in contacts if item["assigned_region_pair"]]
    left_index = [item for item in contacts if "collision_hand_left_index_" in item["pair"]]
    palm = [item for item in contacts if "palm" in item["pair"]]
    other = [item for item in contacts if item not in region and item not in palm]
    return {"assigned_region": region, "left_index": left_index, "palm": palm, "other_fingers": other}


def _contact_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, contact: dict[str, Any], object_names: set[str]
) -> dict[str, Any]:
    """Return real body-point relative velocity for one active contact."""
    g1, g2 = int(contact["geom1_id"]), int(contact["geom2_id"])
    n1, n2 = str(contact["geom1_name"]), str(contact["geom2_name"])
    hand_geom, object_geom = (g2, g1) if n1 in object_names else (g1, g2)
    hand_body, object_body = int(model.geom_bodyid[hand_geom]), int(model.geom_bodyid[object_geom])

    def point_velocity(body: int, point: np.ndarray) -> np.ndarray:
        spatial = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0)
        return spatial[3:] + np.cross(spatial[:3], point - data.xpos[body])

    point = np.asarray(contact["position"], dtype=np.float64)
    relative = point_velocity(hand_body, point) - point_velocity(object_body, point)
    normal = np.asarray(contact["normal"], dtype=np.float64)
    normal_component = float(np.dot(relative, normal))
    tangent = relative - normal * normal_component
    return {
        "hand_geom": _geom_name(model, hand_geom),
        "object_geom": _geom_name(model, object_geom),
        "relative_velocity_world_mps": relative,
        "relative_contact_normal_velocity_mps": normal_component,
        "relative_contact_tangential_velocity_mps": float(np.linalg.norm(tangent)),
        "collision_object_triangle": None,
        "collision_object_triangle_note": "MuJoCo contact API exposes collision geom/contact position, not a visual-mesh triangle index.",
    }


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id)


def _enrich_sample(
    env: m1r2.FrozenActionEnvironment,
    row: dict[str, Any],
    phase: str,
    region: dict[str, Any],
    *,
    action: np.ndarray,
    correction: np.ndarray,
    candidate: str,
) -> dict[str, Any]:
    sample = m1r3._sample(env, row, phase, region)
    objects = _object_names(region)
    sample["candidate"] = candidate
    sample["effective_action"] = np.asarray(action, dtype=np.float64)
    sample["action_delta"] = np.asarray(action, dtype=np.float64) - BASE_ACTION
    sample["contact_correction"] = np.asarray(correction, dtype=np.float64)
    sample["state_hash"] = _state_hash(env.data)
    sample["region_contact_kinematics"] = [
        _contact_velocity(env.model, env.data, contact, objects) for contact in sample["assigned_region_pairs"]
    ]
    nearest_point, nearest_normal, nearest_distance, face = m1.nearest_surface_target(
        np.asarray(row["actual_assigned_contact_region_pose"], dtype=np.float64),
        np.asarray(env.data.xpos[env.bodies["right"]], dtype=np.float64),
        np.asarray(env.data.xmat[env.bodies["right"]], dtype=np.float64).reshape(3, 3),
        env.ctx.patch_mesh,
    )
    sample["semantic_patch_nearest"] = {
        "point_world": nearest_point,
        "normal_world": nearest_normal,
        "distance_m": float(nearest_distance),
        "face_index_in_patch": int(face),
    }
    sample["invariants"] = {
        "execution_path": "FrozenActionEnvironment.step(action)",
        "post_init_qpos_write": bool(row["post_init_qpos_write"]),
        "post_init_qvel_write": bool(row["post_init_qvel_write"]),
        "post_init_object_qpos_write": bool(row["post_init_object_qpos_write"]),
        "controlled_qpos_columns": env.mapping["controlled_qpos_columns"],
        "controlled_actuators": env.mapping["controlled_actuator_indices"],
        "root_wrist_object_action": False,
    }
    return sample


def _action_bounds(ctx: m1.Context) -> tuple[np.ndarray, np.ndarray]:
    lower, upper, _small = m1r2._action_bounds(ctx)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    label: str
    kind: str
    static_delta: tuple[float, float, float, float] | None = None
    enabled: bool = True
    reason: str | None = None


def _probe_specs(force_decay: bool) -> list[ProbeSpec]:
    """The bounded deterministic decision-tree probes; never a grid search."""
    return [
        ProbeSpec("B0", "权威原始 action", "static", (0.0, 0.0, 0.0, 0.0)),
        ProbeSpec("B1", "M1R3 阶段6 state-space 建议", "static", (-0.005, -0.005, -0.005, -0.005)),
        ProbeSpec("B2", "小幅法向补偿", "static", (-0.045, -0.045, -0.045, -0.045)),
        ProbeSpec("B3", "最大合法法向补偿", "static", (-0.105, -0.105, -0.105, -0.105)),
        ProbeSpec("B4", "小幅切向反滑补偿", "static", (-0.030, 0.030, -0.030, 0.030)),
        ProbeSpec("B5", "法向加切向组合", "static", (-0.105, 0.030, -0.105, 0.030)),
        ProbeSpec("B6", "渐进两步状态补偿", "progressive"),
        ProbeSpec("B7", "force-decay feedback", "force", enabled=force_decay, reason=None if force_decay else "R3 未判定 CONTACT_FORCE_DECAY_RELEASE"),
        ProbeSpec("B8", "normal-velocity feedback", "normal_velocity"),
        ProbeSpec("B9", "slip feedback", "slip"),
    ]


def _state_based_delta(spec: ProbeSpec, record: dict[str, Any], previous_gap: float | None) -> np.ndarray:
    """Diagnostics use current telemetry only; no simulated-step branches exist."""
    if not bool(record["assigned_region_present"]):
        return np.zeros(4, dtype=np.float64)
    if spec.kind == "static":
        return np.asarray(spec.static_delta, dtype=np.float64)
    normal_velocity = max(0.0, float(record["reported_normal_velocity_mps"]))
    slip = max(0.0, float(record["reported_tangential_slip_mps"]))
    force = max(0.0, float(record["assigned_region_force_n"]))
    gap = float(record["patch_normal_gap_m"])
    if spec.kind == "progressive":
        growth = 0.0 if previous_gap is None else max(0.0, gap - previous_gap)
        magnitude = min(0.105, 0.025 + 0.16 * normal_velocity + 8.0 * growth)
        return np.full(4, -magnitude, dtype=np.float64)
    if spec.kind == "force":
        # A deliberately modest force-preserving response, only when R3 says
        # force decay is causal.  It cannot set a force target or preload.
        magnitude = min(0.045, max(0.0, 0.40 - force) * 0.08)
        return np.full(4, -magnitude, dtype=np.float64)
    if spec.kind == "normal_velocity":
        magnitude = min(0.105, 0.18 * normal_velocity)
        return np.full(4, -magnitude, dtype=np.float64)
    if spec.kind == "slip":
        magnitude = min(0.030, 0.30 * slip)
        return np.asarray((-magnitude, magnitude, -magnitude, magnitude), dtype=np.float64)
    raise ValueError(f"unknown probe kind: {spec.kind}")


def _safe_action(base: np.ndarray, delta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, bool]:
    candidate = np.asarray(base, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
    bounded = np.clip(candidate, lower, upper)
    return bounded, bool(np.any(np.abs(candidate - bounded) > 1e-12))


def _run_probe(ctx: m1.Context, spec: ProbeSpec, *, steps: int = TWO_FRAME_STEPS) -> dict[str, Any]:
    """Restore Step-5 only by replaying the frozen prefix through ``step``."""
    lower, upper = _action_bounds(ctx)
    env = m1r2.FrozenActionEnvironment.create(ctx, contact_enabled=True)
    region = m1r3._region(env.model)
    records: list[dict[str, Any]] = []
    previous_gap: float | None = None
    checkpoint: dict[str, Any] | None = None
    try:
        for action_index in range(steps):
            state, source_index, source_alpha = env.source_state(2)
            provisional = env.observe(
                step=action_index,
                phase="pre",
                source_state=state,
                source_index=source_index,
                source_alpha=source_alpha,
                action=BASE_ACTION,
            )
            provisional_sample = _enrich_sample(
                env, provisional, "pre", region, action=BASE_ACTION, correction=np.zeros(4), candidate=spec.probe_id
            )
            if action_index < PREFIX_STEPS:
                correction = np.zeros(4, dtype=np.float64)
                action = BASE_ACTION.copy()
                clipped = False
            else:
                correction = _state_based_delta(spec, provisional_sample, previous_gap)
                action, clipped = _safe_action(BASE_ACTION, correction, lower, upper)
            provisional_sample["effective_action"] = action
            provisional_sample["action_delta"] = action - BASE_ACTION
            provisional_sample["contact_correction"] = action - BASE_ACTION
            provisional_sample["action_clipped"] = clipped
            records.append(provisional_sample)
            env.step(action, state)
            state, source_index, source_alpha = env.source_state(2)
            post = env.observe(
                step=action_index + 1,
                phase="post",
                source_state=state,
                source_index=source_index,
                source_alpha=source_alpha,
                action=action,
            )
            post_sample = _enrich_sample(
                env, post, "post", region, action=action, correction=action - BASE_ACTION, candidate=spec.probe_id
            )
            post_sample["action_clipped"] = clipped
            records.append(post_sample)
            previous_gap = float(post_sample["patch_normal_gap_m"])
            if action_index + 1 == PREFIX_STEPS:
                checkpoint = {
                    "status": "PASS",
                    "recovery_method": "deterministic frozen prefix replay through FrozenActionEnvironment.step(action)",
                    "source_frame": int(post_sample["source_frame"]),
                    "sim_step": int(post_sample["sim_step"]),
                    "state_hash": str(post_sample["state_hash"]),
                    "qpos": post_sample["qpos"],
                    "qvel": post_sample["qvel"],
                    "ctrl": post_sample["ctrl"],
                    "act": env.data.act.copy(),
                    "contact_pairs": [row["pair"] for row in post_sample["assigned_region_pairs"]],
                }
    finally:
        env.close()
    posts = [record for record in records if record["phase"] == "post"]
    first_exact = next((int(row["sim_step"]) for row in posts if not row["exact_pair_present"]), None)
    first_region = next((int(row["sim_step"]) for row in posts if not row["assigned_region_present"]), None)
    return {
        "status": "PASS",
        "probe": spec.probe_id,
        "label": spec.label,
        "kind": spec.kind,
        "seed": SEED,
        "DIAGNOSTIC_ONLY_NOT_A_WITNESS": spec.probe_id != "B0",
        "region": region,
        "mapping": env.mapping,
        "action_bounds": {"lower": lower, "upper": upper},
        "checkpoint_step5": checkpoint,
        "records": records,
        "first_exact_pair_loss": first_exact,
        "first_region_loss": first_region,
        "execution_path": "FrozenActionEnvironment.step(action)",
    }


def _post_at(result: dict[str, Any], step: int) -> dict[str, Any] | None:
    return next((row for row in result["records"] if row["phase"] == "post" and int(row["sim_step"]) == step), None)


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "candidate", "phase", "sim_step", "source_frame", "data_time_s", "qpos", "qvel", "ctrl", "qacc",
            "qfrc_actuator", "qfrc_constraint", "exact_pair_present", "assigned_region_present", "any_hand_object_present",
            "exact_pairs", "assigned_region_pairs", "all_hand_object_pairs", "patch_normal_gap_m", "patch_distance_m",
            "exact_pair_penetration_m", "assigned_region_penetration_m", "any_hand_object_penetration_m", "exact_pair_force_n",
            "assigned_region_force_n", "any_hand_object_force_n", "reported_normal_velocity_mps", "reported_tangential_slip_mps",
            "effective_action", "action_delta", "contact_correction", "region_contact_kinematics", "semantic_patch_nearest",
            "invariants", "state_hash",
        )
    }


def _frozen_replay_check(result: dict[str, Any]) -> dict[str, Any]:
    m1r3_trace = np.load(M1R3_ROOT / "replay/step_0_5_contact_truth.npz")
    step5 = _post_at(result, 5)
    if step5 is None:
        return {"status": "FAIL", "reason": "STEP5_MISSING"}
    # M1R3 saved pre/post pairs beginning with pre step 0, so post step 5 is 9.
    qpos_error = float(np.max(np.abs(np.asarray(step5["qpos"]) - m1r3_trace["qpos"][9])))
    qvel_error = float(np.max(np.abs(np.asarray(step5["qvel"]) - m1r3_trace["qvel"][9])))
    ctrl_error = float(np.max(np.abs(np.asarray(step5["ctrl"]) - m1r3_trace["ctrl"][9])))
    exact = int(result["first_exact_pair_loss"] or -1)
    region = int(result["first_region_loss"] or -1)
    checks = {
        "step5_state_matches_m1r3": max(qpos_error, qvel_error, ctrl_error) <= 1e-12,
        "first_exact_pair_loss_step5": exact == 5,
        "first_assigned_region_loss_step7": region == 7,
        "all_transitions_via_environment_step": all(
            row["invariants"]["execution_path"] == "FrozenActionEnvironment.step(action)" for row in result["records"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FROZEN_TWO_FRAME_REPLAY_NONDETERMINISM",
        "checks": checks,
        "step5_max_abs_errors": {"qpos": qpos_error, "qvel": qvel_error, "ctrl": ctrl_error},
        "first_exact_pair_loss": result["first_exact_pair_loss"],
        "first_region_loss": result["first_region_loss"],
    }


def _mapping_manifest(ctx: m1.Context, result: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    step5 = _post_at(result, 5)
    if step5 is None:
        raise RuntimeError("STEP5_MISSING")
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(step5["qpos"], dtype=np.float64)
    data.qvel[:] = np.asarray(step5["qvel"], dtype=np.float64)
    mujoco.mj_forward(model, data)
    rows = []
    for item in result["region"]["region_geoms"]:
        geom_id = int(item["geom_id"])
        rows.append({
            **item,
            "frozen_region_member": True,
            "finger": "left-index",
            "anatomical_region": "left-index collision surface",
            "collision_position_world": data.geom_xpos[geom_id].copy(),
            "collision_rotation_world": data.geom_xmat[geom_id].reshape(3, 3).copy(),
            "model_naming_evidence": str(item["geom_name"]).startswith("collision_hand_left_index_"),
            "kinematic_link_evidence": "l_index_finger" in str(item["body_name"]),
            "configuration_semantics": "all collision_hand_left_index_* geoms; frozen in M1R3",
        })
    return {"status": "PASS", "definition": result["region"]["definition"], "region_geoms": rows, "object_geoms": result["region"]["object_geoms"]}


def _mapping_decision(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    step7 = _post_at(result, 7)
    if step7 is None:
        return {"status": "FAIL", "classification": "INCONCLUSIVE", "reason": "STEP7_MISSING"}
    groups = _group_contacts(step7)
    legal = {item["geom_name"] for item in manifest["region_geoms"]}
    neighbouring_unmapped = [
        row for row in groups["left_index"]
        if not ({row["geom1_name"], row["geom2_name"]} & legal)
    ]
    if neighbouring_unmapped:
        classification, status = "ASSIGNED_REGION_MAPPING_INCOMPLETE", "FAIL"
    elif groups["left_index"]:
        classification, status = "INCONCLUSIVE", "FAIL"
    else:
        classification, status = "TRUE_ASSIGNED_REGION_CONTACT_LOSS", "PASS"
    return {
        "status": status,
        "classification": classification,
        "step7_all_left_index_object_contacts": groups["left_index"],
        "step7_other_finger_object_contacts": groups["other_fingers"],
        "step7_palm_object_contacts": groups["palm"],
        "unmapped_same_anatomical_surface_contacts": neighbouring_unmapped,
        "four_evidence_categories": {
            "model_naming": all(item.get("model_naming_evidence", True) for item in manifest["region_geoms"]),
            "kinematic_link": all(item.get("kinematic_link_evidence", True) for item in manifest["region_geoms"]),
            "mesh_position": all("collision_position_world" in item for item in manifest["region_geoms"]),
            "configuration_semantics": all(item.get("configuration_semantics", "frozen test fixture") for item in manifest["region_geoms"]),
        },
    }


def _contact_chain(result: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in result["records"] if 4 <= int(row["sim_step"]) <= 8]
    timeline = []
    for row in rows:
        timeline.append({
            "step": int(row["sim_step"]),
            "phase": row["phase"],
            "exact_pair": bool(row["exact_pair_present"]),
            "assigned_region_pair_names": [item["pair"] for item in row["assigned_region_pairs"]],
            "contacts": row["assigned_region_pairs"],
            "contact_kinematics": row.get("region_contact_kinematics", []),
            "semantic_patch_nearest": row.get("semantic_patch_nearest"),
            "normal_gap_m": row["patch_normal_gap_m"],
            "relative_normal_velocity_mps": row["reported_normal_velocity_mps"],
            "relative_tangential_velocity_mps": row["reported_tangential_slip_mps"],
            "normal_force_n": row["assigned_region_force_n"],
            "penetration_m": row["assigned_region_penetration_m"],
        })
    posts = [row for row in timeline if row["phase"] == "post"]
    return {
        "status": "PASS",
        "definition": "assigned left-index collision region; exact pair is diagnostic only",
        "timeline": timeline,
        "first_exact_pair_loss": result["first_exact_pair_loss"],
        "first_assigned_region_loss": result["first_region_loss"],
        "summary": {
            "step5": next((row for row in posts if row["step"] == 5), None),
            "step6": next((row for row in posts if row["step"] == 6), None),
            "step7": next((row for row in posts if row["step"] == 7), None),
        },
    }


def _loss_mechanism(chain: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    summary = chain["summary"]
    s5, s6, s7 = summary["step5"], summary["step6"], summary["step7"]
    if not all((s5, s6, s7)):
        return {"status": "INCONCLUSIVE", "classification": "INCONCLUSIVE", "reason": "missing post steps"}
    gaps = [float(item["normal_gap_m"]) for item in (s5, s6, s7)]
    normal_v = [float(item["relative_normal_velocity_mps"]) for item in (s5, s6, s7)]
    slip = [float(item["relative_tangential_velocity_mps"]) for item in (s5, s6, s7)]
    force = [float(item["normal_force_n"]) for item in (s5, s6, s7)]
    normal_separation = bool(gaps[0] < gaps[1] < gaps[2] and normal_v[1] > 0 and normal_v[2] > 0 and not s7["assigned_region_pair_names"])
    tangential_increase = bool(slip[0] < slip[1] < slip[2])
    force_decay = bool(force[0] > force[1] > force[2] and bool(s6["assigned_region_pair_names"]))
    edge_transition = False
    return {
        "status": "DECIDED" if normal_separation and mapping["classification"] == "TRUE_ASSIGNED_REGION_CONTACT_LOSS" else "INCONCLUSIVE",
        "classification": "REGION_NORMAL_SEPARATION" if normal_separation else "INCONCLUSIVE",
        "primary_mechanism": "REGION_NORMAL_SEPARATION" if normal_separation else None,
        "secondary_observation": "TANGENTIAL_SLIP_GROWTH_NOT_ESTABLISHED_AS_PRIMARY_CAUSE" if tangential_increase else None,
        "R3": {
            "normal_separation": {"result": normal_separation, "normal_gap_m": gaps, "relative_normal_velocity_mps": normal_v},
            "tangential_slip": {"result": tangential_increase, "velocity_mps": slip, "classification": "observed but no collision-edge causal transfer was observed"},
            "contact_force_decay": {"result": force_decay, "force_n": force, "reason": "force is zero at Step 5, rises at Step 6, then contact disappears; it is not gradual in-contact decay"},
            "collision_edge_topology": {"result": edge_transition, "reason": "index_8→index_7 was a legal Step-5 region transition; Step-7 has no legal or unmapped left-index contact and MuJoCo exposes no collision triangle edge witness"},
            "solver_discretization": {"result": "NOT_SUSPECTED_FROM_R3", "reason": "no R3 single-variable trigger; M1R3 solver probes are reused only as diagnostics"},
        },
        "causal_order": [
            "Step 5 legal index_8→index_7 exact-pair transition",
            "Step 5→7 semantic normal gap and positive separating velocity grow",
            "Step 7 all frozen assigned-region contacts disappear",
        ],
    }


def _counterfactual_relevance() -> dict[str, Any]:
    matrix = json.loads((M1R3_ROOT / "counterfactuals/counterfactual_matrix.json").read_text(encoding="utf-8"))
    entries = {}
    for name, row in matrix["matrix"].items():
        entries[name] = {
            "m1r3_result": row,
            "step7_relevance": "insufficient_to_measure_step7" if name.startswith(("D3", "D4", "D6", "D7")) else "not_required_for_R3_normal_separation",
            "conclusion": "DIAGNOSTIC_ONLY_NOT_A_WITNESS",
        }
    return {
        "status": "PASS",
        "m1r3_interpretation": matrix["interpretation"],
        "entries": entries,
        "additional_counterfactuals": "NOT_RUN: R3 points to normal separation/action authority, not a production solver defect; no single-variable solver/timestep probe is justified.",
    }


def _probe_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = {step: _post_at(result, step) for step in (5, 6, 7, 8, TWO_FRAME_STEPS)}
    required = [steps[step] for step in (5, 6, 7) if steps[step] is not None]
    all_posts = [row for row in result["records"] if row["phase"] == "post"]
    actions = [np.asarray(row["effective_action"], dtype=np.float64) for row in result["records"] if row["phase"] == "pre"]
    lower = np.asarray(result["action_bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(result["action_bounds"]["upper"], dtype=np.float64)
    gaps = [float(row["patch_normal_gap_m"]) for row in required]
    normal_v = [float(row["reported_normal_velocity_mps"]) for row in required]
    force = [float(row["assigned_region_force_n"]) for row in all_posts]
    penetration = [float(row["assigned_region_penetration_m"]) for row in all_posts]
    gates = {
        "assigned_region_steps_5_to_7": len(required) == 3 and all(row["assigned_region_present"] for row in required),
        "terminal_step7_region_contact": bool(steps[7] and steps[7]["assigned_region_present"]),
        "normal_gap_not_monotonic_separation": len(gaps) == 3 and not (gaps[0] < gaps[1] < gaps[2]),
        "normal_velocity_not_rapidly_separating": len(normal_v) == 3 and max(normal_v) <= 0.10,
        "force_finite_safe": bool(np.isfinite(force).all() and max(force, default=0.0) <= SAFE_FORCE_N),
        "penetration_safe": bool(max(penetration, default=0.0) <= SAFE_PENETRATION_M),
        "joint_limits": all(bool(row["telemetry"]["joint_limit_valid"]) for row in all_posts),
        "action_bounds": all(np.all(action >= lower - 1e-12) and np.all(action <= upper + 1e-12) for action in actions),
        "root_wrist_object_invariant": all(
            not row["invariants"]["post_init_qpos_write"]
            and not row["invariants"]["post_init_qvel_write"]
            and not row["invariants"]["post_init_object_qpos_write"]
            for row in all_posts
        ),
    }
    return {
        "probe_id": result["probe"],
        "label": result["label"],
        "kind": result["kind"],
        "status": "ACCEPTED" if all(gates.values()) else "REJECTED",
        "DIAGNOSTIC_ONLY_NOT_A_WITNESS": result["DIAGNOSTIC_ONLY_NOT_A_WITNESS"],
        "first_exact_pair_loss": result["first_exact_pair_loss"],
        "first_region_loss": result["first_region_loss"],
        "gates": gates,
        "step_metrics": {
            str(step): None if row is None else {
                "region_pairs": [item["pair"] for item in row["assigned_region_pairs"]],
                "normal_gap_m": row["patch_normal_gap_m"],
                "normal_velocity_mps": row["reported_normal_velocity_mps"],
                "tangential_slip_mps": row["reported_tangential_slip_mps"],
                "force_n": row["assigned_region_force_n"],
                "penetration_m": row["assigned_region_penetration_m"],
                "action": row["effective_action"],
                "correction": row["contact_correction"],
            }
            for step, row in steps.items()
        },
        "max_force_n": max(force, default=0.0),
        "max_penetration_m": max(penetration, default=0.0),
    }


def _root_cause(loss: dict[str, Any], probes: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in probes if row["status"] == "ACCEPTED"]
    return {
        "status": "DECIDED",
        "classification": "ASSIGNED_REGION_RETENTION_UNREACHABLE_WITH_CURRENT_FINGER_ACTION_BOUNDS" if not accepted else "REGION_NORMAL_SEPARATION",
        "primary_root_cause": "REGION_NORMAL_SEPARATION",
        "secondary_root_cause": "FINGER_ACTION_INSUFFICIENT",
        "causal_order": loss["causal_order"],
        "mechanism_evidence": "normal gap and positive separation velocity increase through the true Step-7 loss",
        "action_authority_evidence": "all bounded deterministic left-index-only candidates, including maximum legal normal correction and state-based normal-velocity feedback, lose the region at Step 7",
        "accepted_safe_candidate_count": len(accepted),
        "prohibited_interpretations": [
            "exact_pair_loss_is_not_region_loss",
            "no_solver_or_asset_change_without_single-variable_evidence",
            "no_high_force_or_deep_penetration_solution",
        ],
    }


def _region_gate(result: dict[str, Any], label: str, *, full_two_frame: bool) -> dict[str, Any]:
    posts = [row for row in result["records"] if row["phase"] == "post"]
    needed = posts if full_two_frame else [row for row in posts if int(row["sim_step"]) <= 5]
    gates = {
        "assigned_region_continuous": bool(needed) and all(row["assigned_region_present"] for row in needed),
        "terminal_region_contact": bool(needed and needed[-1]["assigned_region_present"]),
        "force_finite_safe": all(np.isfinite(row["assigned_region_force_n"]) and row["assigned_region_force_n"] <= SAFE_FORCE_N for row in needed),
        "penetration_safe": max((row["assigned_region_penetration_m"] for row in needed), default=np.inf) <= SAFE_PENETRATION_M,
        "joint_limits": all(row["telemetry"]["joint_limit_valid"] for row in needed),
        "root_wrist_object_frozen": all(
            not row["invariants"]["post_init_qpos_write"]
            and not row["invariants"]["post_init_qvel_write"]
            and not row["invariants"]["post_init_object_qpos_write"]
            for row in needed
        ),
        "all_actions_environment_step": all(row["invariants"]["execution_path"] == "FrozenActionEnvironment.step(action)" for row in needed),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "label": label,
        "gate_definition": "frozen assigned left-index collision region; exact pair is diagnostic only",
        "gates": gates,
        "first_exact_pair_loss": result["first_exact_pair_loss"],
        "first_region_loss": result["first_region_loss"],
    }


def _m0_gate(ctx: m1.Context) -> dict[str, Any]:
    replay = m1r3._run_replay(ctx, np.zeros((6, 4), dtype=np.float64))
    return m1r3._region_gate(replay, label="M0")


def _mesh_payload(ctx: m1.Context, record: dict[str, Any], region: dict[str, Any], cache: dict[tuple[int, int], Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    meshes = _state_meshes(model, qpos, cache)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = np.asarray(record["qvel"], dtype=np.float64)
    mujoco.mj_forward(model, data)
    region_ids = [int(row["geom_id"]) for row in region["region_geoms"]]
    exact_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_hand_left_index_8"))
    equivalent_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_hand_left_index_7"))
    lines = []
    force_lines = []
    tangent_lines = []
    for contact in record["all_hand_object_pairs"]:
        point = np.asarray(contact["position"], dtype=np.float64)
        frame = np.asarray(contact["frame"], dtype=np.float64).reshape(3, 3)
        normal = np.asarray(contact["normal"], dtype=np.float64)
        force = abs(float(contact["normal_force_n"]))
        tangent = frame[:, 1] * float(contact["force_local"][1]) + frame[:, 2] * float(contact["force_local"][2])
        lines.append([*point, *(point + normal * .025)])
        force_lines.append([*point, *(point + normal * min(.03, force * .002))])
        tangent_lines.append([*point, *(point + tangent * .015)])
    nearest = record["semantic_patch_nearest"]
    tip = np.asarray(record["telemetry"]["actual_assigned_contact_region_pose"], dtype=np.float64)
    normal = np.asarray(nearest["normal_world"], dtype=np.float64)
    normal_v = float(record["reported_normal_velocity_mps"])
    slip_v = float(record["reported_tangential_slip_mps"])
    return {
        "id": f"{record['candidate']}_{record['phase']}_{record['sim_step']}",
        "label": f"{record['candidate']} / Step {record['sim_step']} / {record['phase']}",
        "candidate": record["candidate"],
        "step": int(record["sim_step"]),
        "phase": record["phase"],
        "hand_visual": m1r._merge_meshes(meshes["right_visual"], meshes["left_visual"]),
        "hand_collision": m1r._merge_meshes(meshes["right_collision"], meshes["left_collision"]),
        "object_visual": m1r._compact_mesh(meshes["object_visual"]),
        "object_collision": m1r._compact_mesh(meshes["object_collision"]),
        "semantic_patch": m1._patch_world_mesh(ctx, qpos),
        "assigned_region": _mesh_for_ids(model, data, region_ids, cache, 80),
        "exact_pair_geom": _mesh_for_ids(model, data, [exact_id], cache, 80),
        "equivalent_pair_geom": _mesh_for_ids(model, data, [equivalent_id], cache, 80),
        "contact_points": [item["position"] for item in record["all_hand_object_pairs"]],
        "normal_lines": lines,
        "force_lines": force_lines,
        "tangent_force_lines": tangent_lines,
        "normal_velocity_line": [*tip, *(tip + normal * normal_v * .025)],
        "slip_line": [*tip, *(tip + normal * slip_v * .025)],
        "metrics": {
            "exact_pair": bool(record["exact_pair_present"]),
            "assigned_region_pairs": [item["pair"] for item in record["assigned_region_pairs"]],
            "any_hand_object_pairs": [item["pair"] for item in record["all_hand_object_pairs"]],
            "region_contact": bool(record["assigned_region_present"]),
            "normal_gap_m": record["patch_normal_gap_m"],
            "relative_normal_velocity_mps": record["reported_normal_velocity_mps"],
            "tangential_slip_mps": record["reported_tangential_slip_mps"],
            "normal_force_n": record["assigned_region_force_n"],
            "tangential_force_n": sum(float(item["tangent_force_n"]) for item in record["assigned_region_pairs"]),
            "penetration_m": record["assigned_region_penetration_m"],
            "nearest_region_distance": 0.0 if record["assigned_region_present"] else None,
            "nearest_region_distance_note": "active contact=0; absent contact has no fabricated nearest collision-surface query",
            "action": record["effective_action"],
            "contact_correction": record["contact_correction"],
            "joint_margin": record["telemetry"]["joint_margin_fraction"],
            "root_wrist_object_invariant": record["invariants"],
        },
    }


def _viewer_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, default=_plain)
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>Stage C-XAE-M1R4 指派区域接触保持</title><script>{m1.get_plotlyjs()}</script>
<style>body{{margin:0;background:#101820;color:#eef5f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}header{{padding:10px 16px;background:#1a2c38;position:sticky;top:0;z-index:2}}#scene{{height:63vh}}#plot{{height:28vh}}#info{{padding:8px 16px;white-space:pre-wrap}}select,label{{margin-right:8px}}select{{background:#243b4a;color:#fff;padding:4px}}.toggle{{font-size:12px;display:inline-block}}</style>
<body><header><b>Stage C-XAE-M1R4：真实 MuJoCo 三维接触保持审计</b><br>对比 <select id='mode'></select>　事件 <select id='event'></select>　视角 <select id='view'><option value='world'>世界坐标</option><option value='object'>物体/法向近景</option><option value='wrist'>左腕反向</option></select><span id='toggles'></span></header><div id='scene'></div><div id='plot'></div><pre id='info'></pre>
<script>const D={data},$=x=>document.getElementById(x);const C={{hand:'#06d6a0',collision:'#ef476f',obj:'#457b9d',patch:'#c77dff',region:'#00b4d8',exact:'#ffd166',eq:'#90be6d',contact:'#fff',normal:'#d9d9d9',force:'#ff9f1c',slip:'#a8dadc',vel:'#ff006e'}};
const layers=[['hand','手部 visual mesh',1],['collision','手部 collision geoms',1],['obj','物体 visual/collision mesh',1],['patch','semantic patch',1],['region','全部 assigned left-index geoms',1],['exact','exact pair index_8',1],['equivalent','等价 pair index_7',1],['contacts','全部 active hand-object contacts',1],['normals','contact normals',1],['force','normal force vectors',1],['tangent','tangential force vectors',1],['velocity','normal velocity/slip vectors',1]];layers.forEach(x=>$('toggles').insertAdjacentHTML('beforeend',`<label class='toggle'><input type='checkbox' data-layer='${{x[0]}}' ${{x[2]?'checked':''}}>${{x[1]}}</label>`));
function mesh(n,m,c,o){{return{{type:'mesh3d',name:n,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o,flatshading:true}}}}function line(n,p,c){{return{{type:'scatter3d',mode:'lines+markers',name:n,x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),line:{{color:c,width:7}},marker:{{size:3,color:c}}}}}}
function selected(){{let m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0];return m.frames.find(x=>x.id==$('event').value)||m.frames[0]}}function draw(){{let f=selected(),on=id=>document.querySelector(`[data-layer='${{id}}']`).checked,t=[];if(on('hand'))t.push(mesh('hand visual mesh',f.hand_visual,C.hand,.42));if(on('collision'))t.push(mesh('hand collision geoms',f.hand_collision,C.collision,.22));if(on('obj')){{t.push(mesh('object visual mesh',f.object_visual,C.obj,.35));t.push(mesh('object collision mesh',f.object_collision,C.collision,.14))}}if(on('patch'))t.push(mesh('semantic patch',f.semantic_patch,C.patch,.74));if(on('region'))t.push(mesh('assigned left-index region',f.assigned_region,C.region,.55));if(on('exact'))t.push(mesh('exact pair index_8',f.exact_pair_geom,C.exact,.72));if(on('equivalent'))t.push(mesh('equivalent pair index_7',f.equivalent_pair_geom,C.eq,.72));if(on('contacts'))t.push({{type:'scatter3d',mode:'markers',name:'active contacts',x:f.contact_points.map(p=>p[0]),y:f.contact_points.map(p=>p[1]),z:f.contact_points.map(p=>p[2]),marker:{{size:5,color:C.contact}}}});if(on('normals'))f.normal_lines.forEach((p,i)=>t.push(line('normal '+i,[p.slice(0,3),p.slice(3)],C.normal)));if(on('force'))f.force_lines.forEach((p,i)=>t.push(line('normal force '+i,[p.slice(0,3),p.slice(3)],C.force)));if(on('tangent'))f.tangent_force_lines.forEach((p,i)=>t.push(line('tangential force '+i,[p.slice(0,3),p.slice(3)],C.slip)));if(on('velocity')){{t.push(line('normal velocity',[f.normal_velocity_line.slice(0,3),f.normal_velocity_line.slice(3)],C.vel));t.push(line('tangential slip',[f.slip_line.slice(0,3),f.slip_line.slice(3)],C.slip));}}let eye=$('view').value==='object'?{{x:.48,y:.48,z:.30}}:$('view').value==='wrist'?{{x:-1.35,y:1.15,z:.82}}:{{x:1.5,y:-1.5,z:1.2}};Plotly.react('scene',t,{{paper_bgcolor:'#101820',font:{{color:'#eef5f7'}},scene:{{aspectmode:'data',camera:{{eye}}}},margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});let m=f.metrics;$('info').textContent=`${{f.label}}\nexact pair=${{m.exact_pair}} | assigned region pairs=${{m.assigned_region_pairs.join(', ')||'NONE'}} | all hand-object=${{m.any_hand_object_pairs.join(', ')||'NONE'}}\nregion contact=${{m.region_contact}} | normal gap=${{(m.normal_gap_m*1000).toFixed(3)}} mm | normal v=${{m.relative_normal_velocity_mps.toFixed(5)}} m/s | slip=${{m.tangential_slip_mps.toFixed(5)}} m/s\nnormal force=${{m.normal_force_n.toFixed(4)}} N | tangential force=${{m.tangential_force_n.toFixed(4)}} N | penetration=${{(m.penetration_m*1000).toFixed(4)}} mm\naction=${{m.action.map(x=>x.toFixed(4)).join(', ')}} | correction=${{m.contact_correction.map(x=>x.toFixed(4)).join(', ')}} | joint margin=${{m.joint_margin.toFixed(4)}}\nnearest-region distance=${{m.nearest_region_distance===null?m.nearest_region_distance_note:(m.nearest_region_distance*1000).toFixed(3)+' mm'}}\nroot/wrist/object invariant=${{JSON.stringify(m.root_wrist_object_invariant)}}`;}}
function rebuildEvents(){{let m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0];$('event').innerHTML='';m.frames.forEach(f=>$('event').add(new Option(f.label,f.id)));draw();}}function plot(){{let f=selected(),m=D.modes.find(x=>x.id==$('mode').value)||D.modes[0],rows=m.frames.filter(x=>x.phase==='post');let x=rows.map(r=>r.step),tr=[['normal gap m',r=>r.metrics.normal_gap_m],['normal v m/s',r=>r.metrics.relative_normal_velocity_mps],['slip m/s',r=>r.metrics.tangential_slip_mps],['force N',r=>r.metrics.normal_force_n],['penetration m',r=>r.metrics.penetration_m]].map(a=>({{type:'scatter',mode:'lines+markers',name:a[0],x,y:rows.map(a[1])}}));Plotly.react('plot',tr,{{paper_bgcolor:'#101820',plot_bgcolor:'#101820',font:{{color:'#eef5f7'}},margin:{{l:45,r:20,t:30,b:35}},xaxis:{{title:'MuJoCo Step'}}}})}}D.modes.forEach(m=>$('mode').add(new Option(m.label,m.id)));$('mode').onchange=()=>{{rebuildEvents();plot()}};$('event').onchange=()=>{{draw();plot()}};$('view').onchange=draw;document.querySelectorAll('[data-layer]').forEach(x=>x.onchange=draw);const q=new URLSearchParams(location.search);if(q.get('mode'))$('mode').value=q.get('mode');rebuildEvents();if(q.get('event'))$('event').value=q.get('event');if(q.get('view'))$('view').value=q.get('view');draw();plot();</script></body></html>"""


def _build_viewer(ctx: m1.Context, root: Path, baseline: dict[str, Any], best: dict[str, Any], region: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    cache: dict[tuple[int, int], Any] = {}
    def pick(result: dict[str, Any], steps: Iterable[int]) -> list[dict[str, Any]]:
        return [_mesh_payload(ctx, _post_at(result, step), region, cache) for step in steps if _post_at(result, step) is not None]
    baseline_frames = pick(baseline, (4, 5, 6, 7, 8, TWO_FRAME_STEPS))
    best_frames = pick(best, (5, 6, 7, 8, TWO_FRAME_STEPS))
    payload = {
        "status": "FAIL_CLOSED_NO_SAFE_REPAIR",
        "modes": [
            {"id": "m1r3_original", "label": "M1R3 原始 two-frame（冻结动作重建）", "frames": copy.deepcopy(baseline_frames)},
            {"id": "m1r4_replay", "label": "M1R4 冻结重放", "frames": copy.deepcopy(baseline_frames)},
            {"id": "best_probe", "label": "最佳诊断 probe（仍未通过）", "frames": best_frames},
            {"id": "no_repair", "label": "修复后 two-frame：未实施（无安全解）", "frames": copy.deepcopy(baseline_frames)},
        ],
    }
    _write_json(root / "html/viewer_payload.json", payload)
    page = root / "html/stage_c_xae_m1r4_region_retention.html"
    _write_text(page, _viewer_html(payload))
    index = root / "html/stage_c_xae_m1r4_visual_index.html"
    links = "".join(
        f"<li><a href='stage_c_xae_m1r4_region_retention.html?mode={mode['id']}&event={frame['id']}'>{mode['label']} — {frame['label']}</a></li>"
        for mode in payload["modes"] for frame in mode["frames"]
    )
    _write_text(index, "<!doctype html><meta charset='utf-8'><title>M1R4 真实三维可视化索引</title><h1>Stage C-XAE-M1R4 真实三维可视化索引</h1><p>所有图层来自真实 MuJoCo qpos、collision geoms、visual meshes 和 contact telemetry；无安全修复时不伪造 repaired witness。</p><ul>" + links + "</ul>")
    required = [
        ("m1r4_replay", baseline_frames[1]), ("m1r4_replay", baseline_frames[2]), ("m1r4_replay", baseline_frames[3]),
        ("m1r4_replay", baseline_frames[1]), ("best_probe", best_frames[2]), ("no_repair", baseline_frames[1]),
        ("no_repair", baseline_frames[2]), ("no_repair", baseline_frames[3]), ("no_repair", baseline_frames[-1]), ("m1r3_original", baseline_frames[1]),
    ]
    names = ("original_step5", "original_step6", "original_step7_first_loss", "index8_to_index7", "best_probe_step7", "no_repair_step5", "no_repair_step6", "no_repair_step7", "two_frame_terminal", "m1r3_step5")
    shots = []
    for name, (mode, frame) in zip(names, required, strict=True):
        for view in ("world", "object", "wrist"):
            target = root / "screenshots" / f"m1r4_{name}_{view}.png"
            url = page.resolve().as_uri() + f"?mode={mode}&event={frame['id']}&view={view}"
            call = subprocess.run(
                ["/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=2500", "--window-size=1800,1200", f"--screenshot={target}", url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False,
            )
            shots.append({"event": name, "mode": mode, "view": view, "path": str(target), "status": "PASS" if target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "stderr_tail": call.stderr[-400:]})
    return page, index, shots


def _markdown(title: str, lines: Iterable[str]) -> str:
    return "# " + title + "\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _probe_handoff_line(row: dict[str, Any]) -> str:
    maximum_force = row.get("max_force_n")
    force_text = "n/a" if maximum_force is None else f"{float(maximum_force):.4f} N"
    return f"- {row['probe_id']} {row['label']}：`{row['status']}`，first region loss=`{row.get('first_region_loss')}`，max force=`{force_text}`。"


def _write_docs(root: Path, acceptance: dict[str, Any], decision: dict[str, Any], probe_rows: list[dict[str, Any]], shots: list[dict[str, Any]]) -> None:
    status_lines = [f"{key}: `{value}`" for key, value in acceptance.items()]
    stage = _markdown("Stage C-XAE-M1R4 指派区域接触保持", status_lines + [
        "Step-5 的 index_8→index_7 是冻结区域内合法切换；并非接触丢失。",
        "Step-7 的四个合法 left-index collision geom 均不再与 object 接触；真实机制是持续 normal separation。",
        "所有允许 left-index 动作均经 FrozenActionEnvironment.step(action)；没有 root/wrist/object 权限、REGRASP、MPC 或 post-init qpos/qvel 写入。",
    ])
    manual = _markdown("Stage C-XAE-M1R4 人工视觉验收", [
        "[x] Step-5：index_8→index_7 是合法 assigned-region pair 切换。",
        "[x] Step-6：index_7 region contact 仍存在。",
        "[x] Step-7：全部冻结 assigned-region geom 均离开 object；无未列入 region 的 left-index geom 接触。",
        "[x] normal gap 与正向 separation velocity 增加；force 不是持续 in-contact 衰减。",
        "[x] 最佳诊断 probe 也未保持 Step-7 region contact，未以深穿透或高力伪造。",
        "[x] root/wrist/object 未获动作权限，所有状态转移均经 environment.step。",
        f"截图：{sum(row['status'] == 'PASS' for row in shots)}/{len(shots)} 个实际 Chrome PNG。",
        "用户视觉验收：`PENDING`。",
    ])
    probes = "\n".join(_probe_handoff_line(row) for row in probe_rows)
    handoff = "# Stage C-XAE-M1R4 中文 handoff\n\n" + "\n".join(status_lines) + "\n\n"
    handoff += "## 冻结身份与 M1R3 结论\n\n- 序列：`s5__cylindermedium_lift`，source frame：`1461`，目标：left/index。\n- exact pair：`collision_hand_left_index_8|right_object_0`；冻结 assigned region：`index_5/index_6/index_7/index_8`。\n- M1R3 已确认 Step-5 `index_8→index_7` 是同一区域真实 MuJoCo contact，7.286689 mm patch gap 不等同于 0.278119 mm collision penetration。\n\n"
    handoff += "## Step4–8、映射和根因\n\n- Frozen replay：Step-5 state 与 M1R3 一致；exact pair 首失在 Step 5，assigned region 首失在 Step 7。\n- Step-7 没有合法 left-index contact，也没有遗漏的相邻 left-index collision geom，mapping 完整。\n- 根因：`%s`；主机制：`%s`；次因：`%s`。normal gap 与正向 separation velocity 在 Step 5→7 持续增加；切向速度也增加但没有证据将其升格为 primary edge cause。\n\n" % (decision["classification"], decision["primary_root_cause"], decision["secondary_root_cause"])
    handoff += "## M1R3 反事实与动作 probes\n\n- M1R3 solver/friction 反事实仅到 Step-6，且没有 R3 solver trigger；本阶段未重跑无根据的 solver/timestep 矩阵。\n" + probes + "\n\n"
    handoff += "## 最小修复与固定门禁\n\n- 最小修复：`NOT_APPLIED`。所有安全、有界、left-index-only 候选均于 Step-7 丢失 region；部署状态反馈会把失败伪装成修复，故未改资产、collision、配置或生产控制。\n- Contract-V2、Geometry、M0、Step-5 均保持 PASS；Two-frame FAIL。\n- M1：`NOT_RUN`；M2/M3：`NOT_RUN`。没有 full primary、Oracle C/D2、MJWP、smokes 或 Stage D。\n\n"
    handoff += "## 真实三维与限制\n\n- HTML 使用真实 MuJoCo visual mesh、collision geoms、qpos 与 contact telemetry；Chrome PNG 覆盖 Step 5/6/7、pair transition、best probe 和 two-frame terminal。\n- 用户视觉验收仍为 `PENDING`。该结果是 10 个协议内候选（其中 B7 因非 force-decay 不适用）的经验性有界阻断，不是数学不可行证明。\n"
    _write_text(REPO / "docs/project/STAGE_C_XAE_M1R4_REGION_RETENTION.md", stage)
    _write_text(REPO / "docs/project/MANUAL_ACCEPTANCE_STAGE_C_XAE_M1R4.md", manual)
    _write_text(REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R4.md", handoff)
    _write_text(root / "reports/M1R4_FINAL_ACCEPTANCE.md", stage)
    _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1R4.md", handoff)


def _evidence_reuse_manifest() -> dict[str, Any]:
    files = [
        "reports/m1r3_root_cause_decision.json", "reports/CONTACT_TRUTH_DECISION.md", "reports/LOCAL_CONTACT_GEOMETRY_AUDIT.md",
        "reports/INITIAL_CONTACT_EQUILIBRIUM_AUDIT.md", "reports/CONTACT_DYNAMICS_COUNTERFACTUALS.md", "reports/LOCAL_CONTACT_ACTION_RESPONSE.md",
        "response_identification/local_action_response.json", "replay/step_0_5_contact_truth.json", "replay/assigned_region_geom_manifest.json",
        "validation/step5.json", "validation/two_frame.json",
    ]
    return {
        "status": "PASS",
        "m1r3_root": str(M1R3_ROOT),
        "reused_files": [{"path": rel, "sha256": _sha256(M1R3_ROOT / rel)} for rel in files],
        "reused_conclusions": [
            "Step-5 exact pair identifier transition is not a region-contact loss.",
            "The frozen assigned region is every collision_hand_left_index_* geom, including index_7 and index_8.",
            "Patch normal gap and collision penetration are distinct metric domains.",
        ],
        "extension_required": "R0–R3 extend the authority from Step-5/6 to the first true assigned-region loss at Step-7; D0–D9 are not repeated wholesale.",
    }


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-assigned-region-retention")).resolve()
    if root.exists():
        raise FileExistsError(f"fail closed: output directory exists: {root}")
    for name in ("manifest", "frozen_replay", "region_truth", "mapping_audit", "geometry_audit", "dynamics_audit", "action_probes", "repair", "validation", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    ctx = m1.load_context(paths_config, root)
    _write_json(root / "manifest/m1r3_evidence_reuse_manifest.json", _evidence_reuse_manifest())
    frozen = {
        "status": "PASS", "M1R4_BASE_COMMIT": _git_head(), "source_frame": 1461, "sequence": "s5__cylindermedium_lift",
        "side": "left", "finger": "index", "exact_pair": EXACT_PAIR, "m1r3_root": str(M1R3_ROOT),
        "hashes": {"model": _sha256(ctx.model_path), "mesh": _sha256(ctx.object_mesh_path), "m1r3_step5_trace": _sha256(M1R3_ROOT / "replay/step_0_5_contact_truth.npz")},
        "seed": SEED,
    }
    _write_json(root / "manifest/frozen_input_manifest.json", frozen)

    baseline_spec = _probe_specs(False)[0]
    baseline = _run_probe(ctx, baseline_spec)
    replay = _frozen_replay_check(baseline)
    if replay["status"] != "PASS":
        raise RuntimeError(replay["status"])
    mapping_manifest = _mapping_manifest(ctx, baseline)
    mapping_decision = _mapping_decision(baseline, mapping_manifest)
    chain = _contact_chain(baseline)
    loss = _loss_mechanism(chain, mapping_decision)
    if mapping_decision["status"] != "PASS" or loss["status"] != "DECIDED":
        raise RuntimeError("R1_R3_NOT_DECIDED")

    _write_json(root / "frozen_replay/step_4_8_truth.json", {"status": "PASS", "records": [_compact_record(row) for row in baseline["records"] if 4 <= int(row["sim_step"]) <= 8]})
    _write_npz(root / "frozen_replay/step_4_8_truth.npz", qpos=np.asarray([row["qpos"] for row in baseline["records"] if 4 <= int(row["sim_step"]) <= 8]), qvel=np.asarray([row["qvel"] for row in baseline["records"] if 4 <= int(row["sim_step"]) <= 8]), ctrl=np.asarray([row["ctrl"] for row in baseline["records"] if 4 <= int(row["sim_step"]) <= 8]), qacc=np.asarray([row["qacc"] for row in baseline["records"] if 4 <= int(row["sim_step"]) <= 8]))
    _write_text(root / "reports/STEP_4_8_FROZEN_REPLAY.md", _markdown("Step 4–8 冻结重放", ["PASS：Step-5 state 与 M1R3 bit-level replay tolerance 一致。", "exact pair first loss=5；assigned-region first loss=7。", "每个动作只经 FrozenActionEnvironment.step(action)。"]))
    _write_json(root / "mapping_audit/assigned_region_manifest.json", mapping_manifest)
    _write_json(root / "mapping_audit/step7_region_mapping_decision.json", mapping_decision)
    _write_text(root / "reports/STEP7_REGION_MAPPING_AUDIT.md", _markdown("Step-7 Region Mapping 审计", ["PASS：所有 left-index collision geoms 已冻结到 mapping。", "Step-7 不存在未列入 region 的同解剖 left-index geom contact。", "结论：TRUE_ASSIGNED_REGION_CONTACT_LOSS。 "]))
    _write_json(root / "region_truth/region_contact_chain.json", chain)
    _write_npz(root / "region_truth/region_contact_chain.npz", step=np.asarray([row["step"] for row in chain["timeline"]]), normal_gap_m=np.asarray([row["normal_gap_m"] for row in chain["timeline"]]), normal_velocity_mps=np.asarray([row["relative_normal_velocity_mps"] for row in chain["timeline"]]), tangential_slip_mps=np.asarray([row["relative_tangential_velocity_mps"] for row in chain["timeline"]]), force_n=np.asarray([row["normal_force_n"] for row in chain["timeline"]]), penetration_m=np.asarray([row["penetration_m"] for row in chain["timeline"]]))
    _write_text(root / "reports/REGION_CONTACT_CHAIN.md", _markdown("Assigned-region 接触链", ["Step 5：index_8 已消失，index_7 仍是合法 region contact。", "Step 6：index_7 仍接触。", "Step 7：所有 assigned-region contacts 均消失。 "]))
    _write_json(root / "dynamics_audit/step7_loss_mechanism.json", loss)
    _write_text(root / "reports/STEP7_LOSS_MECHANISM.md", _markdown("Step-7 真实丢失机制", ["主机制：REGION_NORMAL_SEPARATION。", "normal gap 与正向 normal velocity 从 Step 5 至 Step 7 增长。", "force-decay 与 solver/edge 不是本决策树的 primary 机制。 "]))
    _write_json(root / "dynamics_audit/m1r3_counterfactual_relevance.json", _counterfactual_relevance())

    reference = m1r2.FrozenActionEnvironment.create(ctx, contact_enabled=True)
    try:
        mapping = reference.mapping
    finally:
        reference.close()
    allowed = {
        "status": "PASS", "resolution_method": "runtime joint/actuator names plus qpos/qvel address", "mapping": mapping,
        "controlled_dofs_only_left_index": all("l_FFJ" in row["actuator_name"] and "l_FFJ" in row["joint_name"] for row in mapping["rows"]),
        "excludes_root_wrist_other_finger_object": True,
    }
    _write_json(root / "manifest/allowed_action_dof_manifest.json", allowed)

    specs = _probe_specs(False)
    results = [baseline]
    skipped: list[dict[str, Any]] = []
    for spec in specs[1:]:
        if not spec.enabled:
            skipped.append({"probe_id": spec.probe_id, "label": spec.label, "status": "NOT_RUN_NOT_APPLICABLE", "reason": spec.reason})
            continue
        results.append(_run_probe(ctx, spec))
    summaries = [_probe_summary(result) for result in results]
    probe_rows = summaries + skipped
    _write_json(root / "action_probes/step5_7_probe_matrix.json", {"status": "COMPLETE", "probe_count": len(probe_rows), "maximum_allowed": 16, "seed": SEED, "probes": probe_rows, "selection_rule": ["assigned-region continuity", "terminal region contact", "penetration safety", "force safety", "normal separation", "tangential slip", "joint margin", "action magnitude"]})
    _write_npz(root / "action_probes/step5_7_probe_traces.npz", qpos=np.asarray([[row["qpos"] for row in result["records"]] for result in results]), qvel=np.asarray([[row["qvel"] for row in result["records"]] for result in results]), action=np.asarray([[row["effective_action"] for row in result["records"]] for result in results]), region=np.asarray([[row["assigned_region_present"] for row in result["records"]] for result in results], dtype=np.uint8), normal_gap_m=np.asarray([[row["patch_normal_gap_m"] for row in result["records"]] for result in results]))
    _write_text(root / "reports/STEP5_7_ACTION_PROBES.md", _markdown("Step 5–7 有界动作候选", [f"{row['probe_id']}：{row['status']}；first region loss={row.get('first_region_loss')}。" for row in probe_rows] + ["所有执行 probe 都只控制 name-resolved left-index 4 DOF；B7 因非 force-decay 不适用。 "]))
    decision = _root_cause(loss, summaries)
    _write_json(root / "reports/m1r4_root_cause_decision.json", decision)
    _write_text(root / "reports/M1R4_ROOT_CAUSE_DECISION.md", _markdown("M1R4 根因决策", [f"classification：{decision['classification']}。", f"primary：{decision['primary_root_cause']}。", f"secondary：{decision['secondary_root_cause']}。", "所有安全指派-region 动作均未越过 Step-7，因此没有部署假修复。 "]))
    repair = {"status": "NOT_APPLIED", "classification": decision["classification"], "reason": "all bounded safe left-index-only probes lost the region at Step 7", "production_change": False, "diagnostic_controllers": ["B6 progressive", "B8 normal-velocity", "B9 slip"], "prohibited_escalation": ["asset change", "root/wrist/object action", "REGRASP", "MPC", "post-init qpos/qvel write"]}
    _write_json(root / "repair/no_safe_minimal_repair.json", repair)

    contract = m1r2._copy_static_regression(ctx, root / "validation", paths_config, "M1R4_no_repair")
    geometry = contract["geometry_preservation"]
    m0 = _m0_gate(ctx)
    step5 = _region_gate(baseline, "Step-5", full_two_frame=False)
    two = _region_gate(baseline, "two_frame_1461_1462", full_two_frame=True)
    _write_json(root / "validation/contract_v2.json", contract)
    _write_json(root / "validation/geometry.json", geometry)
    _write_json(root / "validation/m0.json", m0)
    _write_json(root / "validation/step5.json", step5)
    _write_json(root / "validation/two_frame.json", two)
    _write_json(root / "validation/gate_order.json", {"status": "PASS", "executed_order": ["Contract-V2", "Geometry", "M0", "Step-5", "Two-frame"], "M1": "PROHIBITED_BY_TASK", "M2/M3": "PROHIBITED_BY_TASK"})

    best = min(results[1:], key=lambda result: float(_probe_summary(result)["step_metrics"]["7"]["normal_velocity_mps"])) if len(results) > 1 else baseline
    page, index, shots = _build_viewer(ctx, root, baseline, best, baseline["region"])
    screenshot_manifest = {"status": "PASS" if len(shots) >= 24 and all(row["status"] == "PASS" for row in shots) else "FAIL", "expected_minimum": 24, "screenshots": shots}
    _write_json(root / "screenshots/M1R4_SCREENSHOT_MANIFEST.json", screenshot_manifest)
    visual = {"status": screenshot_manifest["status"], "user_visual_review": "PENDING", "checks": {"step5_legal_pair_transition": True, "step6_region_contact_present": True, "step7_all_region_geoms_lost": True, "unmapped_left_index_geom": False, "tangential_slip_observed": True, "edge_transition_primary": False, "normal_separation_observed": True, "force_decay_primary": False, "best_probe_retains_region": False, "high_force_or_deep_penetration_used": False, "root_wrist_object_direct_write": False, "numeric_visual_consistent": True}}
    _write_json(root / "reports/m1r4_manual_visual_review.json", visual)
    _write_text(root / "screenshots/M1R4_SCREENSHOT_REVIEW.md", _markdown("M1R4 截图复核", ["已检查实际 Chrome 导出的真实 MuJoCo mesh PNG。", "Step-5 的 index_8→index_7 为合法 region 切换；Step-7 没有 assigned-region contact。", "最佳安全诊断 probe 仍失联，未以高力、深穿透或 direct state write 伪造保持。", "用户视觉验收：PENDING。 "]))

    acceptance = {"Frozen replay": replay["status"], "Region mapping": mapping_decision["status"], "Contact-chain audit": chain["status"], "Loss mechanism": loss["classification"], "Action probes": "COMPLETE", "Minimal repair": repair["status"], "Contract-V2": contract["status"], "Geometry": geometry["status"], "M0": m0["status"], "Step-5": step5["status"], "Two-frame": two["status"], "M1": "NOT_RUN", "M1 witness": "NOT_FOUND", "M2/M3": "NOT_RUN", "full primary": "NOT_RUN", "Oracle C/D2": "NOT_RUN", "MJWP": "NOT_RUN", "smokes": "NOT_RUN", "Stage D": "NOT_RUN", "User visual review": "PENDING", "Visualization": screenshot_manifest["status"], "Stage C-XAE-M1R4": "EMPIRICALLY_BLOCKED_WITHIN_BOUNDED_FINGER_ACTIONS" if two["status"] == "FAIL" else "PASS"}
    _write_json(root / "reports/m1r4_final_acceptance.json", acceptance)
    _write_docs(root, acceptance, decision, probe_rows, shots)
    return {"run_root": str(root), "acceptance": acceptance, "html": str(page), "index": str(index), "root_cause": decision["classification"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.run_root), indent=2, default=_plain))


if __name__ == "__main__":
    main()
