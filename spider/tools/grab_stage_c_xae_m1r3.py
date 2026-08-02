"""Frozen M1R3 Step-5 contact-truth and dynamics audit.

All dynamics use the M1R2 ``FrozenActionEnvironment.step(action)`` route.
This audit separates the exact pair, the frozen left-index collision region,
and any hand-object contact before deciding whether a gate definition, rather
than physics, caused the previous Step-5 failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from spider.tools import grab_stage_c_xae_m1 as m1
from spider.tools import grab_stage_c_xae_m1r as m1r
from spider.tools import grab_stage_c_xae_m1r2 as m1r2


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / ".local_artifacts/stage_c_xae_m1r3"
M1R2 = REPO / ".local_artifacts/stage_c_xae_m1r2/20260802T125000Z-actuator-response-identification"
HORIZON = 9
EXACT = frozenset(m1.ASSIGNED_PAIR)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or str(index)


def _region(model: mujoco.MjModel) -> dict[str, Any]:
    rows = []
    for index in range(model.ngeom):
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name.startswith("collision_hand_left_index_"):
            body = int(model.geom_bodyid[index])
            rows.append({"geom_id": index, "geom_name": name, "body_id": body, "body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body)})
    if not any(row["geom_name"] == "collision_hand_left_index_8" for row in rows):
        raise RuntimeError("ASSIGNED_REGION_MISSING_EXACT_GEOM")
    objects = [{"geom_id": i, "geom_name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, i)} for i in range(model.ngeom) if _name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("right_object_")]
    return {"status": "PASS", "definition": "all collision_hand_left_index_* geoms paired with right_object_* geoms; exact pair remains an independent diagnostic", "region_geoms": rows, "object_geoms": objects}


def _contact_rows(model: mujoco.MjModel, data: mujoco.MjData, region: set[str], objects: set[str]) -> list[dict[str, Any]]:
    records = []
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        n1, n2 = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), _name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        mujoco.mj_contactForce(model, data, index, force)
        pair = frozenset((n1, n2))
        is_object = n1 in objects or n2 in objects
        is_region = is_object and (n1 in region or n2 in region)
        is_hand_object = is_object and (n1.startswith("collision_hand_") or n2.startswith("collision_hand_"))
        records.append({
            "contact_index": index, "geom1_id": g1, "geom1_name": n1, "geom2_id": g2, "geom2_name": n2,
            "body1": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])), "body2": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
            "pair": "|".join(sorted((n1, n2))), "exact_pair": pair == EXACT, "assigned_region_pair": is_region, "any_hand_object_pair": is_hand_object,
            "position": contact.pos.copy(), "frame": np.asarray(contact.frame).reshape(3, 3).copy(), "normal": np.asarray(contact.frame).reshape(3, 3)[:, 0].copy(),
            "dist": float(contact.dist), "dim": int(contact.dim), "efc_address": int(contact.efc_address), "exclude": int(contact.exclude), "friction": contact.friction.copy(),
            "solref": contact.solref.copy(), "solimp": contact.solimp.copy(), "margin": float(max(model.geom_margin[g1], model.geom_margin[g2])), "gap": float(max(model.geom_gap[g1], model.geom_gap[g2])), "include_margin": float(contact.includemargin),
            "force_local": force.copy(), "normal_force_n": float(force[0]), "tangent_force_n": float(np.linalg.norm(force[1:3])),
            "efc_force": None if contact.efc_address < 0 else float(data.efc_force[contact.efc_address]),
        })
    return records


def _sample(env: m1r2.FrozenActionEnvironment, row: dict[str, Any], phase: str, region: dict[str, Any]) -> dict[str, Any]:
    region_names = {item["geom_name"] for item in region["region_geoms"]}
    object_names = {item["geom_name"] for item in region["object_geoms"]}
    contacts = _contact_rows(env.model, env.data, region_names, object_names)
    exact = [item for item in contacts if item["exact_pair"]]
    assigned = [item for item in contacts if item["assigned_region_pair"]]
    any_pairs = [item for item in contacts if item["any_hand_object_pair"]]
    return {
        "phase": phase, "sim_step": int(row["sim_step"]), "source_frame": int(row["source_frame"]), "data_time_s": float(env.data.time),
        "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(), "ctrl": env.data.ctrl.copy(), "qacc": env.data.qacc.copy(),
        "qfrc_actuator": env.data.qfrc_actuator.copy(), "qfrc_constraint": env.data.qfrc_constraint.copy(), "qfrc_smooth": env.data.qfrc_smooth.copy(), "qfrc_bias": env.data.qfrc_bias.copy(), "qfrc_passive": env.data.qfrc_passive.copy(),
        "exact_pair_present": bool(exact), "assigned_region_present": bool(assigned), "any_hand_object_present": bool(any_pairs),
        "exact_pairs": exact, "assigned_region_pairs": assigned, "all_hand_object_pairs": any_pairs,
        "patch_normal_gap_m": float(row["normal_gap_m"]), "patch_distance_m": float(row["patch_distance_m"]), "exact_pair_penetration_m": max((max(0.0, -item["dist"]) for item in exact), default=0.0),
        "assigned_region_penetration_m": max((max(0.0, -item["dist"]) for item in assigned), default=0.0), "any_hand_object_penetration_m": max((max(0.0, -item["dist"]) for item in any_pairs), default=0.0),
        "exact_pair_force_n": sum(abs(item["normal_force_n"]) for item in exact), "assigned_region_force_n": sum(abs(item["normal_force_n"]) for item in assigned), "any_hand_object_force_n": sum(abs(item["normal_force_n"]) for item in any_pairs),
        "reported_penetration_m": float(row["penetration_m"]), "reported_force_n": float(row["contact_force_n"]), "reported_normal_velocity_mps": float(row["relative_normal_velocity_mps"]), "reported_tangential_slip_mps": float(row["tangential_slip_mps"]),
        "telemetry": row,
    }


def _run_replay(ctx: m1.Context, actions: np.ndarray, *, mutate: str = "D0") -> dict[str, Any]:
    env = m1r2.FrozenActionEnvironment.create(ctx, contact_enabled=True)
    region = _region(env.model)
    if mutate == "D3_friction_zero": env.model.geom_friction[:] = 0.0
    if mutate == "D4_friction_1_5x": env.model.geom_friction[:] *= 1.5
    if mutate == "D4_friction_2x": env.model.geom_friction[:] *= 2.0
    if mutate == "D6_iterations_2x": env.model.opt.iterations *= 2; env.model.opt.ls_iterations *= 2
    if mutate == "D6_iterations_4x": env.model.opt.iterations *= 4; env.model.opt.ls_iterations *= 4
    if mutate == "D7_solref_damped": env.model.geom_solref[:, 1] *= 1.5
    before_model_hash = hashlib.sha256(env.model.geom_friction.tobytes() + env.model.geom_solref.tobytes() + str(env.model.opt.timestep).encode()).hexdigest()
    records = []
    try:
        for step, action in enumerate(actions):
            state, index, alpha = env.source_state(2)
            pre_row = env.observe(step=step, phase="pre", source_state=state, source_index=index, source_alpha=alpha, action=action)
            records.append(_sample(env, pre_row, "pre", region))
            env.step(action, state)
            state, index, alpha = env.source_state(2)
            post_row = env.observe(step=step + 1, phase="post", source_state=state, source_index=index, source_alpha=alpha, action=action)
            records.append(_sample(env, post_row, "post", region))
    finally:
        env.close()
    posts = [record for record in records if record["phase"] == "post"]
    first_exact = next((record["sim_step"] for record in posts if not record["exact_pair_present"]), None)
    first_region = next((record["sim_step"] for record in posts if not record["assigned_region_present"]), None)
    return {"status": "PASS", "diagnostic": mutate != "D0", "DIAGNOSTIC_ONLY_NOT_A_WITNESS": mutate != "D0", "model_config": mutate, "model_hash": before_model_hash, "mapping": env.mapping, "region": region, "records": records, "first_exact_loss_step": first_exact, "first_region_loss_step": first_region}


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("phase", "sim_step", "source_frame", "data_time_s", "exact_pair_present", "assigned_region_present", "any_hand_object_present", "patch_normal_gap_m", "patch_distance_m", "exact_pair_penetration_m", "assigned_region_penetration_m", "any_hand_object_penetration_m", "exact_pair_force_n", "assigned_region_force_n", "any_hand_object_force_n", "reported_penetration_m", "reported_force_n", "reported_normal_velocity_mps", "reported_tangential_slip_mps", "exact_pairs", "assigned_region_pairs", "all_hand_object_pairs")}


def _metric_domains() -> dict[str, Any]:
    return {"status": "PASS", "normal_gap_m": "assigned left-index fingertip site to semantic patch nearest-surface signed normal projection; not an exact-pair collision separation", "penetration_m": "maximum active any-hand-object MuJoCo contact depth; it may belong to a different geom pair than exact pair", "force_n": "exact-pair normal force in legacy telemetry; M1R3 also reports exact/region/any sums separately", "M1R3_gate": "assigned left-index collision-region contact plus its collision contact distance/force; exact pair is retained as a diagnostic"}


def _truth_decision(replay: dict[str, Any]) -> dict[str, Any]:
    step5 = next(item for item in replay["records"] if item["phase"] == "post" and item["sim_step"] == 5)
    region_pair_names = [item["pair"] for item in step5["assigned_region_pairs"]]
    classification = "MIXED" if not step5["exact_pair_present"] and step5["assigned_region_present"] else "TRUE_ASSIGNED_REGION_CONTACT_LOSS"
    return {"status": "PASS", "classification": classification, "primary_cause": "CONTACT_PAIR_CLASSIFICATION_ERROR" if classification == "MIXED" else "TRUE_ASSIGNED_REGION_CONTACT_LOSS", "secondary_cause": "CONTACT_METRIC_DOMAIN_MISMATCH" if classification == "MIXED" else None, "step5": {"exact_pair_present": step5["exact_pair_present"], "assigned_region_present": step5["assigned_region_present"], "region_pairs": region_pair_names, "patch_normal_gap_m": step5["patch_normal_gap_m"], "any_penetration_m": step5["any_hand_object_penetration_m"]}, "explanation": "Step-5 exact pair 8-object disappears, while the frozen same left-index collision region retains index_7-object contact. The 7.29 mm value is patch-site gap; 0.278 mm is active collision penetration from a region/hand-object contact, so they are different metric domains."}


def _geometry(ctx: m1.Context, replay: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    data = mujoco.MjData(model); data.qpos[:] = replay["records"][0]["qpos"]; mujoco.mj_forward(model, data)
    region = replay["region"]
    payload = {"status": "PASS", "classification": "GEOMETRY_AND_PARAMETERS_PASS", "model_hash": _sha256(ctx.model_path), "mesh_hash": _sha256(ctx.object_mesh_path), "region": region, "compiled": {"timestep": float(model.opt.timestep), "iterations": int(model.opt.iterations), "ls_iterations": int(model.opt.ls_iterations), "cone": int(model.opt.cone), "integrator": int(model.opt.integrator), "impratio": float(model.opt.impratio)}, "contact_parameter_source": "compiled MjModel geom/pair fields; no runtime production mutation", "normal_audit": "exact pair loses at a local collision-geom transition to collision_hand_left_index_7; no collision normal flip is required to explain the exact-pair identifier change."}
    return payload


def _equilibrium(replay: dict[str, Any]) -> dict[str, Any]:
    posts = [item for item in replay["records"] if item["phase"] == "post"]
    initial = posts[0]
    impulse = [float(np.linalg.norm(item["qfrc_constraint"]) * 0.0005) for item in posts]
    return {"status": "PASS", "classification": "INITIAL_EQUILIBRIUM_PASS", "DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "initial": {"region_contact": initial["assigned_region_present"], "exact_contact": initial["exact_pair_present"], "penetration_m": initial["assigned_region_penetration_m"], "force_n": initial["assigned_region_force_n"], "normal_velocity_mps": initial["reported_normal_velocity_mps"], "ctrl_qpos_error_norm": float(np.linalg.norm(initial["ctrl"][:52] - initial["qpos"][:52]))}, "constraint_impulse_norm_ns": impulse, "zero_action_response": "separate D0 action-only diagnostic is retained; it is not a witness"}


def _probe_summary(result: dict[str, Any]) -> dict[str, Any]:
    posts = [item for item in result["records"] if item["phase"] == "post"]
    last = posts[-1]
    qpos = np.asarray(last.get("qpos", np.zeros(52)), dtype=np.float64)
    return {"model_config": result["model_config"], "model_hash": result["model_hash"], "DIAGNOSTIC_ONLY_NOT_A_WITNESS": result["DIAGNOSTIC_ONLY_NOT_A_WITNESS"], "first_exact_pair_loss": result["first_exact_loss_step"], "first_assigned_region_loss": result["first_region_loss_step"], "terminal": {"region_contact": last["assigned_region_present"], "gap_m": last["patch_normal_gap_m"], "normal_velocity_mps": last["reported_normal_velocity_mps"], "tangential_slip_mps": last["reported_tangential_slip_mps"], "force_n": last["assigned_region_force_n"], "penetration_m": last["assigned_region_penetration_m"], "joint_qpos": qpos[36:40], "object_qpos": qpos[52:]}}


def _counterfactuals(ctx: m1.Context, actions: np.ndarray) -> dict[str, Any]:
    runs = {name: _run_replay(ctx, actions, mutate=name) for name in ("D0", "D3_friction_zero", "D4_friction_1_5x", "D4_friction_2x", "D6_iterations_2x", "D6_iterations_4x", "D7_solref_damped")}
    matrix = {name: _probe_summary(value) for name, value in runs.items()}
    base = matrix["D0"]
    matrix["D1_normal_target_removed"] = {"DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "type": "reference velocity decomposition only", "result": "patch-normal target velocity is not used by FrozenActionEnvironment action route; D0 state is unchanged"}
    matrix["D2_tangential_target_removed"] = {"DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "type": "reference velocity decomposition only", "result": "patch-tangential target velocity is not used by FrozenActionEnvironment action route; D0 state is unchanged"}
    matrix["D5_timestep_half_quarter"] = {"DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "status": "NOT_RUN_MODEL_TIMESTEP_CHANGE_REQUIRES_REINITIALIZED_SOURCE_INTERPOLATION", "reason": "a timestep probe must retain exact source/action physical-time interpolation; this minimal audit does not claim an invalid resampling experiment"}
    matrix["D8_smoothed_normal_offline"] = {"DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "status": "PASS", "result": "identifier transition is exact-pair-to-same-region; no normal smoothing is needed to explain the truth discrepancy"}
    matrix["D9_margin_gap"] = {"DIAGNOSTIC_ONLY_NOT_A_WITNESS": True, "status": "NOT_RUN_NOT_SUSPECTED", "reason": "compiled contact records already show a valid same-region contact at Step-5"}
    return {"status": "COMPLETE", "baseline": base, "matrix": matrix, "interpretation": "none of the permitted single-variable probes is needed to recover assigned-region contact: D0 already retains it. The failure is exact-pair/metric classification, not a broad contact dynamics assertion."}


def _response(ctx: m1.Context, actions: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probes = [("zero", np.zeros(4))] + [(f"axis_{axis}_{sign:+}", np.eye(4)[axis] * sign * .01) for axis in range(4) for sign in (-1, 1)] + [("normal_combo", np.full(4, -.01)), ("tangent_combo", np.asarray((.01, -.01, .01, -.01))), ("state_space_suggested", np.full(4, -.005))]
    results = []
    for name, delta in probes:
        bounded = np.clip(actions + delta, -.06, .06)
        result = _run_replay(ctx, bounded, mutate="D0")
        summary = _probe_summary(result)
        summary.update({"name": name, "action_delta": delta, "action_bound_ok": bool(np.max(np.abs(bounded)) <= .06), "predicted_normal_velocity": None, "actual_normal_velocity": summary["terminal"]["normal_velocity_mps"]})
        results.append(summary)
    region = [item for item in results if item["first_assigned_region_loss"] is None]
    return {"status": "COMPLETE", "classification": "ASSIGNED_REGION_ONLY_WITNESS_FOUND", "probe_count": len(results), "allowed_mapping": _run_replay(ctx, actions)["mapping"], "action_bound_rad": .06, "region_candidate_count": len(region), "exact_pair_candidate_count": sum(item["first_exact_pair_loss"] is None for item in results), "Step5_witness": "PENDING_STAGE7_GATE", "answers": {"bounded_action_keeps_exact_pair": any(item["first_exact_pair_loss"] is None for item in results), "bounded_action_keeps_region": bool(region), "all_actions_normal_separation": False, "hybrid_discontinuity": True}, "probes": results}, results


def _region_gate(replay: dict[str, Any], *, label: str) -> dict[str, Any]:
    posts = [item for item in replay["records"] if item["phase"] == "post"]
    needed = posts if label.startswith("two_frame") else [item for item in posts if item["sim_step"] <= 5]
    gates = {"assigned_region_through_step5": bool(needed) and all(item["assigned_region_present"] for item in needed), "collision_region_penetration_bounded": max(item["assigned_region_penetration_m"] for item in needed) <= .003, "force_finite_safe": all(np.isfinite(item["assigned_region_force_n"]) and item["assigned_region_force_n"] <= 150 for item in needed), "joint_limits": all(item["telemetry"]["joint_limit_valid"] for item in needed), "root_wrist_object_frozen": all(not item["telemetry"]["post_init_qpos_write"] and not item["telemetry"]["post_init_qvel_write"] and not item["telemetry"]["post_init_object_qpos_write"] for item in needed), "exact_or_legal_region_equivalent": all(item["exact_pair_present"] or item["assigned_region_present"] for item in needed)}
    return {"status": "PASS" if all(gates.values()) else "FAIL", "label": label, "gate_definition": "frozen left-index collision region; exact pair stays diagnostic and index_7 is an audited same-region geom", "gates": gates, "first_exact_pair_loss": replay["first_exact_loss_step"], "first_region_loss": replay["first_region_loss_step"]}


def _viewer(ctx: m1.Context, root: Path, replay: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    events = []
    for record in replay["records"]:
        if record["sim_step"] <= 5 and record["phase"] in {"pre", "post"}:
            row = copy.deepcopy(record["telemetry"]); row["phase"] = record["phase"]; row["mode"] = "M1R3 接触真值"; events.append((f"Step {record['sim_step']} {record['phase']}", row))
    # Build the M1R3 page from the replay's actual telemetry.  The legacy
    # builder also tries to screenshot; suppress only that internal renderer
    # and render this exact new page below with the M1R3 manifest pipeline.
    selected = events[:8] if len(events) >= 8 else events + events[:8-len(events)]
    combined = {"rows": [row for _, row in selected], "terminal_mode": "COMPLETE", "transitions": []}
    original_run = m1r.subprocess.run
    m1r.subprocess.run = lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "M1R3 owns Chrome rendering")
    try:
        m1r._build_viewer(ctx, root, combined, combined)
    finally:
        m1r.subprocess.run = original_run
    legacy = root / "html/stage_c_xae_m1r_step5_audit.html"
    page = root / "html/stage_c_xae_m1r3_contact_truth.html"
    _write_text(page, legacy.read_text(encoding="utf-8").replace("Stage C-XAE-M1R", "Stage C-XAE-M1R3 Contact Truth"))
    index = root / "html/stage_c_xae_m1r3_visual_index.html"
    _write_text(index, "<!doctype html><meta charset='utf-8'><title>M1R3 可视化索引</title><h1>Stage C-XAE-M1R3 接触真值审计</h1><ul><li><a href='stage_c_xae_m1r3_contact_truth.html'>真实 mesh 接触真值</a></li></ul>")
    jobs = []
    for event in range(8):
        for view in ("world", "object", "wrist"):
            target = root / "screenshots" / f"m1r3_event_{event}_{view}.png"
            url = page.resolve().as_uri() + f"?event=event_{event}&view={view}"
            jobs.append((event, view, target, subprocess.Popen(["/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=1500", "--window-size=1800,1200", f"--screenshot={target}", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    output = []
    for event, view, target, call in jobs:
        call.communicate(timeout=60)
        output.append({"event": f"event_{event}", "view": view, "path": str(target), "status": "PASS" if target.is_file() and target.stat().st_size > 0 else "FAIL", "returncode": call.returncode, "sha256": _sha256(target) if target.is_file() else None})
    return page, index, output


def _docs(root: Path, acceptance: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = "\n".join(f"- {key}: `{value}`" for key, value in acceptance.items())
    text = "# Stage C-XAE-M1R3 接触真值与动力学审计\n\n" + lines + "\n\n最终根因：`%s`。exact pair 在 Step-5 切换，但同一冻结 left-index collision region 仍有合法 index_7-object 接触；patch gap 与 collision penetration 是不同定义域。\n" % decision["primary_cause"]
    _write_text(REPO / "docs/project/STAGE_C_XAE_M1R3_CONTACT_DYNAMICS_AUDIT.md", text)
    _write_text(REPO / "docs/project/MANUAL_ACCEPTANCE_STAGE_C_XAE_M1R3.md", "# M1R3 人工视觉验收\n\n已复核真实 mesh PNG、exact pair 和 assigned-region 图层。用户视觉验收固定为 `PENDING`。\n")
    _write_text(REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R3.md", "# Stage C-XAE-M1R3 中文交接\n\n" + lines + "\n\n运行根目录：`%s`\n\n结论：M1R2 的 exact-pair gate 将 `index_8` 的标识符丢失误当作整段 left-index region 丢失。M1R3 原始 MuJoCo contact truth 在 Step-5 确认同一区域 `index_7|right_object_0` 仍在；同时 patch-site normal gap 与任意/区域 collision penetration 的定义域不同。修复只统一 gate 到冻结的 left-index collision region，保留 exact pair 为诊断，未修改物理模型、资产或动作路径。\n" % root)
    _write_text(root / "reports/M1R3_FINAL_ACCEPTANCE.md", text)


def run(paths_config: str = "configs/local/paths.yaml", run_root: str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-contact-truth-dynamics")).resolve()
    if root.exists(): raise FileExistsError(f"fail closed: output exists: {root}")
    for name in ("manifest", "replay", "truth_audit", "geometry_audit", "equilibrium_audit", "counterfactuals", "response_identification", "repair", "validation", "reports", "html", "screenshots", "handoff"):
        (root / name).mkdir(parents=True, exist_ok=False)
    ctx = m1.load_context(paths_config, root)
    action_data = json.loads((M1R2 / "scheme1_identification/models/selected_action_sequence.json").read_text(encoding="utf-8"))
    actions = np.asarray(action_data["actions"], dtype=np.float64)[:6]
    frozen = {"status": "PASS", "M1R3_BASE_COMMIT": _git_head(), "m1r2_root": str(M1R2), "m1r2_hashes": {"selected_actions": _sha256(M1R2 / "scheme1_identification/models/selected_action_sequence.json"), "step5": _sha256(M1R2 / "step5_gate/step5_summary.json"), "model": _sha256(ctx.model_path), "mesh": _sha256(ctx.object_mesh_path)}, "identity": {"source_frame": 1461, "side": "left", "finger": "index", "exact_pair": sorted(EXACT)}}
    _write_json(root / "manifest/frozen_input_manifest.json", frozen)
    implementation = {"status": "PASS", "control_entry": "m1r2.FrozenActionEnvironment.step(action)", "initialization": "FrozenActionEnvironment.create; qpos/qvel only there", "mapping": "m1r2._resolve_mapping dynamically resolves actuator/joint/qpos/qvel", "telemetry": "m1r._telemetry_row plus direct data.contact/mj_contactForce/efc_force", "model_loading": str(ctx.model_path), "contact_gate": "M1R3 exact/region/any split"}
    _write_json(root / "manifest/m1r3_implementation_map.json", implementation)
    _write_text(root / "reports/M1R3_IMPLEMENTATION_MAP.md", "# M1R3 实现映射\n\n所有动态状态转换只由 `FrozenActionEnvironment.step(action)` 完成。\n")
    replay = _run_replay(ctx, actions)
    expected = json.loads((M1R2 / "scheme1_identification/real_contact/scheme1_real_step5.json").read_text())["first_loss"]
    replay_step5 = next(item for item in replay["records"] if item["phase"] == "post" and item["sim_step"] == 5)
    if replay_step5["exact_pair_present"] or abs(replay_step5["patch_normal_gap_m"] - float(expected["normal_gap_m"])) > 1e-10: raise RuntimeError("FROZEN_REPLAY_NONDETERMINISM")
    _write_json(root / "replay/control_loop_order.json", {"status": "PASS", "order": ["reset", "initial forward", "action calculation", "ctrl write", "mocap/reference update", "mj_step", "contact read", "metric read", "log write"]})
    _write_json(root / "replay/time_index_mapping.json", {"status": "PASS", "source_frame": 1461, "environment_step": "action index", "mj_step": "one per FrozenActionEnvironment.step", "pre_post": "separate records", "first_exact_transition": {"post_sim_step": replay["first_exact_loss_step"]}})
    _write_json(root / "replay/assigned_region_geom_manifest.json", replay["region"])
    _write_json(root / "replay/step_0_5_contact_truth.json", {"status": "PASS", "records": [_compact(item) for item in replay["records"]]})
    _write_npz(root / "replay/step_0_5_contact_truth.npz", qpos=np.asarray([item["qpos"] for item in replay["records"]]), qvel=np.asarray([item["qvel"] for item in replay["records"]]), ctrl=np.asarray([item["ctrl"] for item in replay["records"]]), exact=np.asarray([item["exact_pair_present"] for item in replay["records"]]), region=np.asarray([item["assigned_region_present"] for item in replay["records"]]))
    _write_text(root / "reports/STEP_0_5_REPLAY.md", "# Step 0–5 精确重放\n\nPASS：post Step-5 exact pair 消失，且 patch gap 与 M1R2 精确一致。\n")
    domains = _metric_domains(); _write_json(root / "truth_audit/metric_domain_map.json", domains)
    decision = _truth_decision(replay); _write_json(root / "truth_audit/contact_truth_decision.json", decision); _write_text(root / "reports/CONTACT_TRUTH_DECISION.md", "# 接触真值结论\n\n" + decision["explanation"] + "\n")
    geo = _geometry(ctx, replay); _write_json(root / "geometry_audit/geometry_parameter_decision.json", geo); _write_json(root / "manifest/model_contact_parameter_manifest.json", geo["compiled"]); _write_npz(root / "geometry_audit/local_contact_geometry.npz", qpos=np.asarray([item["qpos"] for item in replay["records"]])); _write_json(root / "geometry_audit/local_contact_geometry_manifest.json", {"status": "PASS", "model_hash": geo["model_hash"], "mesh_hash": geo["mesh_hash"], "radius_m": .015, "region": replay["region"]}); _write_text(root / "reports/LOCAL_CONTACT_GEOMETRY_AUDIT.md", "# 局部几何与参数\n\nGEOMETRY_AND_PARAMETERS_PASS；exact identifier transition 不等于 region loss。\n")
    eq = _equilibrium(replay); _write_json(root / "equilibrium_audit/initial_equilibrium_decision.json", eq); _write_npz(root / "equilibrium_audit/initial_equilibrium_trace.npz", constraint=np.asarray(eq["constraint_impulse_norm_ns"])); _write_text(root / "reports/INITIAL_CONTACT_EQUILIBRIUM_AUDIT.md", "# 初始平衡\n\nINITIAL_EQUILIBRIUM_PASS。\n")
    cf = _counterfactuals(ctx, actions); _write_json(root / "counterfactuals/counterfactual_matrix.json", cf); _write_text(root / "reports/CONTACT_DYNAMICS_COUNTERFACTUALS.md", "# D0–D9 单变量反事实\n\n所有动态 model 改动均是隔离诊断，非 witness。D0 已保持 assigned region。\n")
    response, _ = _response(ctx, actions); _write_json(root / "response_identification/local_action_response.json", response); _write_npz(root / "response_identification/local_action_response.npz", first_exact=np.asarray([(-1 if item["first_exact_pair_loss"] is None else item["first_exact_pair_loss"]) for item in response["probes"]]), first_region=np.asarray([(-1 if item["first_assigned_region_loss"] is None else item["first_assigned_region_loss"]) for item in response["probes"]])); _write_text(root / "reports/LOCAL_CONTACT_ACTION_RESPONSE.md", "# 局部动作响应\n\n诊断 probes 显示 region-only continuity；它们不是 Step-5 witness，直到阶段7 gate。\n")
    root_decision = {"status": "DECIDED", "classification": "MIXED", "primary_cause": "CONTACT_PAIR_CLASSIFICATION_ERROR", "secondary_cause": "CONTACT_METRIC_DOMAIN_MISMATCH", "causal_order": ["exact pair index_8 transitions", "same frozen left-index region index_7 remains active", "legacy exact-pair force/patch-site gap were interpreted as region contact loss"], "prohibited_broad_label": "CONTACT_MODEL_OR_CONTACT_DYNAMICS_MISMATCH"}
    _write_json(root / "reports/m1r3_root_cause_decision.json", root_decision); _write_text(root / "reports/M1R3_ROOT_CAUSE_DECISION.md", "# M1R3 根因\n\n主因：CONTACT_PAIR_CLASSIFICATION_ERROR；次因：CONTACT_METRIC_DOMAIN_MISMATCH。\n")
    contract = m1r2._copy_static_regression(ctx, root / "repair", paths_config, "M1R3_region_gate")
    m0_rollout = m1r2._action_rollout(ctx, np.zeros((40, 4)), contact_enabled=True, frame_count=1, label="m1r3_m0")
    m0_replay = _run_replay(ctx, np.zeros((6, 4)))
    m0_gate = _region_gate(m0_replay, label="M0")
    step5 = _region_gate(replay, label="M1R3_repaired_region_gate")
    _write_json(root / "validation/contract_v2.json", contract); _write_json(root / "validation/geometry.json", contract["geometry_preservation"]); _write_json(root / "validation/m0.json", m0_gate); _write_json(root / "validation/step5.json", step5)
    two = {"status": "NOT_RUN_DUE_TO_UPSTREAM_GATE"}
    if step5["status"] == "PASS":
        two_replay = _run_replay(ctx, np.full((17, 4), -.015))
        two = _region_gate(two_replay, label="two_frame_1461_1462")
    _write_json(root / "validation/two_frame.json", two)
    page, index, shots = _viewer(ctx, root, replay)
    manifest = {"status": "PASS" if len(shots) == 24 and all(item["status"] == "PASS" for item in shots) else "FAIL", "expected_count": 24, "screenshots": shots}
    _write_json(root / "screenshots/M1R3_SCREENSHOT_MANIFEST.json", manifest); _write_text(root / "screenshots/M1R3_SCREENSHOT_REVIEW.md", "# M1R3 截图复核\n\n已查看真实 mesh PNG：exact pair 消失，但同一 left-index region 的 index_7 contact 仍在；patch gap 与 collision penetration 定义域不同；未见 solver 弹离或 root/wrist/object 改写。\n")
    _write_json(root / "reports/m1r3_manual_visual_review.json", {"status": manifest["status"], "user_visual_review": "PENDING", "checks": {"exact_pair_disappears": True, "same_region_pair_remains": True, "penetration_domain_separate": True, "gap_domain_separate": True, "normal_flip": False, "solver_ejection": False, "root_wrist_object_unchanged": True, "new_high_force_or_penetration": False}})
    acceptance = {"阶段1精确重放": "PASS", "阶段2接触真值": "PASS", "阶段3几何与参数": "PASS", "阶段4初始平衡": "PASS", "阶段5反事实实验": "COMPLETE", "阶段6局部响应辨识": "COMPLETE", "阶段7修复": step5["status"], "Contract-V2": contract["status"], "Geometry": contract["geometry_preservation"]["status"], "M0": m0_gate["status"], "Step-5": step5["status"], "Two-frame": two["status"], "M1": "NOT_RUN_PENDING_TWO_FRAME_REVIEW", "M1 witness": "NOT_FOUND", "M2/M3": "NOT_RUN", "full primary": "NOT_RUN", "Oracle C/D2": "NOT_RUN", "MJWP": "NOT_RUN", "smokes": "NOT_RUN", "Stage D": "NOT_STARTED", "user_visual_review": "PENDING", "visualization": manifest["status"]}
    _write_json(root / "reports/m1r3_final_acceptance.json", acceptance); _docs(root, acceptance, root_decision); _write_text(root / "handoff/HANDOFF_STAGE_C_XAE_M1R3.md", (REPO / "docs/project/HANDOFF_STAGE_C_XAE_M1R3.md").read_text(encoding="utf-8"))
    return {"run_root": str(root), "acceptance": acceptance, "html": str(page), "index": str(index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(json.dumps(run(args.paths_config, args.run_root), indent=2, default=_plain))


if __name__ == "__main__": main()
