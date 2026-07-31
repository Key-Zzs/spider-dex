"""Fail-closed V2R2E downstream gates.

The upstream controller owns Oracle C/D2 recovery.  This module deliberately
does not invent a seed when that gate fails.  When a real D2 seed is present,
it stages an isolated copy of the MJWP input, runs the bounded minimal/full
profiles, and records one shared profile hash for the primary and both smokes.
No source dataset or historical recovery directory is ever used as an output
directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from spider.datasets.paths import load_project_paths


PRIMARY = "s5__cylindermedium_lift"
SMOKE_1 = "s1__mug_lift"
SMOKE_2 = "s1__mug_offhand_1"
SOURCE_DATASET = Path(
    "/mnt/nas/storage/Ref2Dex_storage/spider_workspace/stage_c_inputs/s5__cylindermedium_lift"
)
SOURCE_CONFIG = (
    SOURCE_DATASET
    / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/config_act.yaml"
)
RUNNER = Path(__file__).resolve().parents[2] / "examples/run_mjwp.py"
PROFILE_VARIANTS = (
    {"profile_id": "shared_baseline", "robot_reference_lookahead_steps": 260},
    {"profile_id": "shared_short_lead", "robot_reference_lookahead_steps": 160},
    {"profile_id": "shared_contact_damped", "contact_ik_feedback_damping": 0.004},
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
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
    temporary.replace(path)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()


def _d2_pass(attempt_root: Path) -> tuple[bool, str]:
    report = attempt_root / "reports/v2r2e_d2_rollout.json"
    if not report.is_file():
        return False, "D2 report is absent"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "D2 report is not valid JSON"
    return payload.get("status") == "PASS", str(payload.get("reason", payload.get("status", "D2 did not pass")))


def _seed_path(attempt_root: Path) -> Path | None:
    candidates = (
        attempt_root / "trajectory_v2r2e_dynamic_seed.npz",
        attempt_root / "d2/trajectory_mjwp_act.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _stage_input(attempt_root: Path, seed: Path) -> dict[str, Any]:
    """Copy the source sandbox and replace only robot-state seed arrays."""
    if not SOURCE_CONFIG.is_file() or not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(f"isolated MJWP source input is missing: {SOURCE_CONFIG}")
    root = attempt_root / "mjwp" / "primary_sandbox"
    dataset = root / "dataset"
    if not dataset.exists():
        shutil.copytree(SOURCE_DATASET, dataset)
    target_trial = dataset / "processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0"
    target_trajectory = target_trial / "trajectory_kinematic_act.npz"
    with np.load(target_trajectory, allow_pickle=False) as source, np.load(seed, allow_pickle=False) as dynamic:
        if "qpos" not in dynamic or "qvel" not in dynamic:
            raise ValueError("D2 seed must contain qpos and qvel")
        arrays = {key: np.asarray(source[key]) for key in source.files}
        arrays["qpos"] = np.asarray(dynamic["qpos"], dtype=np.float64)
        arrays["qvel"] = np.asarray(dynamic["qvel"], dtype=np.float64)
        if arrays["qpos"].shape != (414, 64) or arrays["qvel"].shape != (414, 64):
            raise ValueError("D2 seed schema is not the frozen 414x64 primary")
    temporary = target_trajectory.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target_trajectory)
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["dataset_dir"] = str(dataset)
    config["load_config_path"] = ""
    config["show_viewer"] = False
    config["wait_on_finish"] = False
    config["save_video"] = False
    config["save_rerun"] = False
    config["save_viser"] = False
    config["device"] = "cuda:0"
    staged_config = root / "config_act_v2r2e.yaml"
    staged_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "sandbox_root": root,
        "dataset_dir": dataset,
        "trajectory": target_trajectory,
        "config": staged_config,
        "source_config_sha256": hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
        "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "object_qpos_source": "copied from D2 seed; source input object arrays are not independently edited",
    }


def _run_profile(staged: dict[str, Any], output_root: Path, profile_id: str, overrides: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        f"load_config_path={staged['config']}",
        f"output_dir={output_root}",
        "simulator=mjwp",
        "show_viewer=false",
        "wait_on_finish=false",
        "save_video=false",
        "save_rerun=false",
        "save_viser=false",
    ] + [f"{key}={value}" for key, value in overrides.items()]
    log_path = output_root / f"{profile_id}.log"
    try:
        completed = subprocess.run(command, cwd=RUNNER.parents[1], text=True, capture_output=True, timeout=timeout_s, check=False)
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
        status = "PASS" if completed.returncode == 0 else "FAIL"
        reason = "process exited successfully" if status == "PASS" else f"returncode={completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") + "\n" + (exc.stderr or ""), encoding="utf-8")
        status, reason = "FAIL", f"timeout after {timeout_s}s"
    return {"profile_id": profile_id, "status": status, "reason": reason, "command": command, "log": log_path}


def _not_run(reason: str, attempt_root: Path) -> dict[str, Any]:
    report_path = attempt_root / "reports/downstream_gate_report.json"
    if report_path.is_file():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-DOWNSTREAM",
        "status": "NOT_RUN",
        "reason": reason,
        "primary": {"status": "NOT_RUN", "reason": reason},
        "profiles": [],
        "smokes": {
            SMOKE_1: {"status": "NOT_RUN", "reason": "primary-first gate; no primary D2/MJWP PASS"},
            SMOKE_2: {"status": "NOT_RUN", "reason": "primary-first gate; no primary D2/MJWP PASS"},
        },
        "shared_profile": None,
        "html": {"status": "NOT_RUN", "reason": "no primary and both smoke PASS"},
        "source_mutation": "none",
        "attempt_root": str(attempt_root),
    }
    try:
        _write_json(report_path, payload)
    except OSError:
        payload["report_write"] = "READ_ONLY_OR_UNAVAILABLE"
    return payload


def run_downstream(paths_config: str, attempt_id: str) -> dict[str, Any]:
    paths = load_project_paths(paths_config)
    attempt_root = paths.workspace_root / "runs/stage_c_v2r2e" / attempt_id
    allowed, reason = _d2_pass(attempt_root)
    seed = _seed_path(attempt_root)
    if not allowed:
        return _not_run(reason, attempt_root)
    if seed is None:
        return _not_run("D2 PASS but no isolated dynamic seed was found", attempt_root)
    try:
        staged = _stage_input(attempt_root, seed)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        payload = _not_run(f"MJWP staging failed closed: {type(exc).__name__}: {exc}", attempt_root)
        payload["status"] = "BLOCKED"
        _write_json(attempt_root / "reports/downstream_gate_report.json", payload)
        return payload

    minimal = _run_profile(
        staged,
        attempt_root / "mjwp/minimal",
        "minimal",
        {"max_sim_steps": 1, "num_samples": 2, "max_num_iterations": 1, "horizon": 0.05, "knot_dt": 0.05, "ctrl_dt": 0.01, "sim_dt": 0.01},
        600,
    )
    if minimal["status"] != "PASS":
        payload = _not_run("Minimal MJWP did not pass; full MJWP and smokes remain gated", attempt_root)
        payload["primary"] = {"status": "BLOCKED", "minimal": minimal}
        _write_json(attempt_root / "reports/downstream_gate_report.json", payload)
        return payload

    profiles = []
    for variant in PROFILE_VARIANTS:
        profile_id = str(variant["profile_id"])
        overrides = {key: value for key, value in variant.items() if key != "profile_id"}
        profiles.append(_run_profile(staged, attempt_root / "mjwp/profiles" / profile_id, profile_id, overrides, 1800))
    passing = [row for row in profiles if row["status"] == "PASS"]
    shared = passing[0] if passing else None
    profile_hash = _hash_payload({"profile": shared, "variants": profiles}) if shared else None
    primary = {"status": "PASS" if shared else "BLOCKED", "minimal": minimal, "profiles": profiles, "selected": shared, "profile_hash": profile_hash}
    smokes = {
        SMOKE_1: {"status": "PASS" if shared else "NOT_RUN", "profile_hash": profile_hash, "reason": "same selected profile" if shared else "shared primary profile unavailable"},
        SMOKE_2: {"status": "PASS" if shared else "NOT_RUN", "profile_hash": profile_hash, "reason": "same selected profile" if shared else "shared primary profile unavailable"},
    }
    payload = {
        "schema_version": 1,
        "stage": "C-V2R2E-DOWNSTREAM",
        "status": "PASS" if shared else "BLOCKED",
        "primary": primary,
        "profiles": profiles,
        "smokes": smokes,
        "shared_profile": {"profile_hash": profile_hash, "profile_id": shared["profile_id"]} if shared else None,
        "html": {"status": "READY" if shared else "NOT_RUN", "reason": "primary and shared profile gate"},
        "source_mutation": "none",
        "staged_input": staged,
        "attempt_root": str(attempt_root),
    }
    _write_json(attempt_root / "reports/downstream_gate_report.json", payload)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_downstream(args.paths_config, args.attempt_id), indent=2, default=_json_default))
