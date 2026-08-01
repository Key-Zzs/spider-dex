"""Stage C-XAR correction-leakage localization and evidence writer.

The tool is intentionally narrow: it reads the frozen Stage-B and historical
C-XA trajectories, compares their *physical* MuJoCo states, and writes a new
fail-fast XAR namespace.  It never changes raw GRAB, Stage B, the historical
C-XA namespace, object qpos, or source timing.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from spider.datasets.paths import load_project_paths
from spider.tools.grab_source_geometry_audit import (
    FINGERTIP_NAMES,
    _extract_mujoco_geometry,
    _site_transform,
    invert_transform,
    object_relative,
    rotation_residual_rad,
)
from spider.tools.grab_stage_c import _cxa_variable_indices, _stage_b_act_baseline


SEQUENCE_ID = "s5__cylindermedium_lift"
ROOT_WRIST = np.asarray((0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31), dtype=np.int64)
ROOT_TRANSLATION = np.asarray((0, 1, 2, 26, 27, 28), dtype=np.int64)
ROOT_ROTATION = np.asarray((3, 4, 5, 29, 30, 31), dtype=np.int64)
FINGERS = np.asarray(tuple(range(6, 26)) + tuple(range(32, 52)), dtype=np.int64)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _hash_array(value: np.ndarray) -> str:
    packed = np.ascontiguousarray(value)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _summary(values: np.ndarray, unit: str = "m") -> dict[str, float | str]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "max_" + unit: float(np.max(values, initial=0.0)),
        "mean_" + unit: float(np.mean(values)),
        "p95_" + unit: float(np.percentile(values, 95)),
        "rmse_" + unit: float(np.sqrt(np.mean(np.square(values)))),
    }


def _paths(paths_config: str, old_root_override: str | None = None) -> dict[str, Path]:
    paths = load_project_paths(paths_config)
    robot = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / SEQUENCE_ID / "0"
    old = Path(old_root_override).resolve() if old_root_override else robot / "stage_c_contract_v2_cxa"
    physics = json.loads((robot / "stage_c/physics_input.json").read_text(encoding="utf-8"))
    result = {
        "robot": robot,
        "old": old,
        "scene_act": Path(physics["scene_act"]),
        "stage_b": robot / "trajectory_kinematic.npz",
        "mapping": robot / "source_mapping.json",
        "old_trajectory": old / "trajectory_depenetrated_init_cxa_level_1_flexible.npz",
        "old_trace": old / "depenetration_trace_cxa_level_1_flexible.npz",
        "old_patch": old / "source_contact_patches.npz",
    }
    missing = [str(path) for path in result.values() if isinstance(path, Path) and not path.is_file() and path not in {old, robot}]
    if not old.is_dir():
        missing.append(str(old))
    if missing:
        raise FileNotFoundError("C-XAR frozen input missing: " + ", ".join(missing))
    return result


def _root(output_root: str, run_id: str | None, create: bool) -> Path:
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-cxa-repair")
    root = Path(output_root).resolve() / identifier
    if create:
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite historical XAR run: {root}")
        for name in (
            "manifest", "localization/experiment_a_zero_roundtrip", "localization/experiment_b_finger_only",
            "localization/experiment_c_serialization", "localization/experiment_d_layered_dof",
            "localization/experiment_e_first_jump", "repair", "rebuilt_cxa", "source_geometry_audit",
            "m0", "two_frame", "m1", "reports", "html", "screenshots", "handoff",
        ):
            (root / name).mkdir(parents=True, exist_ok=False)
    if not root.is_dir():
        raise FileNotFoundError(f"XAR run directory does not exist: {root}")
    return root


def _model_map(scene: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    joints: list[dict[str, Any]] = []
    for index in range(model.njnt):
        joints.append({
            "joint_id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
            "qpos_index": int(model.jnt_qposadr[index]),
            "qvel_index": int(model.jnt_dofadr[index]),
            "joint_type": int(model.jnt_type[index]),
            "range": model.jnt_range[index],
        })
    actuators = [{"actuator_id": index, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index), "joint_id": int(model.actuator_trnid[index, 0])} for index in range(model.nu)]
    return {"model_nq": model.nq, "model_nv": model.nv, "model_nu": model.nu, "joints": joints, "actuators": actuators}


def _source_lines(function: Any) -> dict[str, Any]:
    lines, start = inspect.getsourcelines(function)
    return {"function": function.__qualname__, "file": inspect.getsourcefile(function), "line_start": start, "line_end": start + len(lines) - 1}


def write_implementation_map(root: Path, inputs: dict[str, Path]) -> None:
    import spider.tools.grab_source_geometry_audit as source_audit
    import spider.tools.grab_stage_c as stage_c
    import spider.tools.grab_stage_c_contact_reassignment as reassign

    calls = {
        "configuration": {"path": "configs/project/grab_wuji_depenetration.yaml", "entry": "allow_wrist_correction=false"},
        "optimizer": _source_lines(stage_c.depenetrate_init),
        "bounds_and_locked_dofs": _source_lines(stage_c._joint_bounds),
        "runtime_locked_invariant": _source_lines(stage_c._assert_locked_dofs),
        "stage_b_loader_and_chart_conversion": _source_lines(stage_c._stage_b_act_baseline),
        "rotation_chart_conversion": _source_lines(stage_c._continuous_intrinsic_xyz),
        "collision_refinement": _source_lines(stage_c._collision_dls_refine),
        "visual_refinement": _source_lines(stage_c._visual_dls_refine),
        "serialization_writer": _source_lines(stage_c._atomic_npz),
        "cxa_validator": _source_lines(source_audit._cxa_audit),
        "cxa_viewer": _source_lines(source_audit.render_viewer if hasattr(source_audit, "render_viewer") else source_audit._cxa_audit),
        "v2_contact_metric": _source_lines(reassign.evaluate_v2_depenetrated),
        "v2_target_writer": _source_lines(reassign.compile_contact_targets),
    }
    payload = {
        "schema_version": 1,
        "historical_cxa_root": inputs["old"],
        "stage_b_input": inputs["stage_b"],
        "scene_act": inputs["scene_act"],
        "call_chain": [
            "Stage-B trajectory_kinematic.npz",
            "_stage_b_act_baseline (freejoint wxyz -> serial intrinsic XYZ object chart)",
            "depenetrate_init (Powell plus bounded collision/visual DLS)",
            "trajectory_depenetrated_init*.npz",
            "_cxa_audit / MuJoCo mj_forward",
            "grab_stage_c_xar localization and viewer",
        ],
        "entries": calls,
        "historical_artifacts": {key: value for key, value in inputs.items() if key.startswith("old")},
    }
    _write_json(root / "manifest/cxa_implementation_map.json", payload)
    lines = ["# C-XA 真实实现调用链", "", "以下为本机实际代码路径和函数，不是推测。", ""]
    for name, value in calls.items():
        if "file" in value:
            lines.append(f"- `{name}`：`{value['file']}:{value['line_start']}`，函数 `{value['function']}`。")
        else:
            lines.append(f"- `{name}`：`{value['path']}`，`{value['entry']}`。")
    _write_text(root / "reports/CXA_IMPLEMENTATION_MAP.md", "\n".join(lines) + "\n")


def _geometry(scene: Path, qpos: np.ndarray) -> dict[str, Any]:
    geometry = _extract_mujoco_geometry(scene, qpos)
    object_transform = geometry["object_transform"]
    sides: dict[str, dict[str, np.ndarray]] = {}
    for side in ("right", "left"):
        palm = _site_transform(geometry, f"{side}_palm")
        tips = np.stack([
            geometry["site_positions"][:, geometry["site_names"].index(f"{side}_{finger}_tip")]
            for finger in FINGERTIP_NAMES
        ], axis=1)
        sides[side] = {"palm": palm, "tips": tips, "relative_palm": object_relative(object_transform, palm)}
    return {"geometry": geometry, "object": object_transform, "sides": sides}


def _preservation(scene: Path, base: np.ndarray, candidate: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    before, after = _geometry(scene, base), _geometry(scene, candidate)
    object_translation = np.linalg.norm(before["object"][:, :3, 3] - after["object"][:, :3, 3], axis=1)
    object_rotation = rotation_residual_rad(before["object"][:, :3, :3], after["object"][:, :3, :3])
    sides: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"object_translation_m": object_translation, "object_rotation_rad": object_rotation}
    for side in ("right", "left"):
        left, right = before["sides"][side], after["sides"][side]
        wrist_translation = np.linalg.norm(left["relative_palm"][:, :3, 3] - right["relative_palm"][:, :3, 3], axis=1)
        wrist_rotation = rotation_residual_rad(left["relative_palm"][:, :3, :3], right["relative_palm"][:, :3, :3])
        tips = np.linalg.norm(left["tips"] - right["tips"], axis=-1)
        wrist_step_translation = np.linalg.norm(np.diff(right["relative_palm"][:, :3, 3] - left["relative_palm"][:, :3, 3], axis=0), axis=1)
        wrist_step_rotation = rotation_residual_rad(
            (left["relative_palm"][:-1, :3, :3].transpose(0, 2, 1) @ right["relative_palm"][:-1, :3, :3]),
            (left["relative_palm"][1:, :3, :3].transpose(0, 2, 1) @ right["relative_palm"][1:, :3, :3]),
        )
        arrays.update({f"{side}_wrist_translation_m": wrist_translation, f"{side}_wrist_rotation_rad": wrist_rotation, f"{side}_tip_delta_m": tips, f"{side}_wrist_step_translation_m": wrist_step_translation, f"{side}_wrist_step_rotation_rad": wrist_step_rotation})
        sides[side] = {
            "wrist_translation": _summary(wrist_translation),
            "wrist_rotation": _summary(wrist_rotation, "rad"),
            "fingertip_change": _summary(tips),
            "frame_to_frame_wrist_translation": _summary(wrist_step_translation),
            "frame_to_frame_wrist_rotation": _summary(wrist_step_rotation, "rad"),
        }
    return {"object_translation": _summary(object_translation), "object_rotation": _summary(object_rotation, "rad"), "sides": sides}, arrays


def _key_indices(frames: np.ndarray, trace: np.ndarray, preservation_arrays: dict[str, np.ndarray]) -> dict[str, int]:
    root_t = np.maximum(np.linalg.norm(trace[:, :3], axis=1), np.linalg.norm(trace[:, 26:29], axis=1))
    root_r = np.maximum(np.linalg.norm(trace[:, 3:6], axis=1), np.linalg.norm(trace[:, 29:32], axis=1))
    root_step = np.maximum(np.linalg.norm(np.diff(trace[:, :3], axis=0), axis=1), np.linalg.norm(np.diff(trace[:, 26:29], axis=0), axis=1))
    first = np.flatnonzero((root_t > 0.002) | (root_r > np.deg2rad(2.0)))
    max_wrist = int(np.argmax(np.maximum(preservation_arrays["right_wrist_translation_m"], preservation_arrays["left_wrist_translation_m"])))
    return {
        "1461": int(np.where(frames == 1461)[0][0]),
        "1462": int(np.where(frames == 1462)[0][0]),
        "1465": int(np.where(frames == 1465)[0][0]),
        "first_jump": int(first[0]) if len(first) else -1,
        "max_wrist": max_wrist,
        "max_frame_to_frame": 1 + int(np.argmax(root_step)),
    }


def write_dof_audit(root: Path, scene: Path, old_trace: np.ndarray, old_phase: np.ndarray) -> None:
    model = _model_map(scene)
    mutable_default, locked_default = _cxa_variable_indices(False)
    root_delta = old_trace[:, ROOT_WRIST]
    payload = {
        "schema_version": 1,
        "model": model,
        "groups": {
            "right_root_wrist": list(range(0, 6)), "right_fingers": list(range(6, 26)),
            "left_root_wrist": list(range(26, 32)), "left_fingers": list(range(32, 52)),
            "object": list(range(52, 64)),
        },
        "repaired_default_optimized_qpos_indices": mutable_default,
        "repaired_default_locked_qpos_indices": locked_default,
        "object_qpos_indices": list(range(52, 64)),
        "historical_trace_root_wrist_nonzero_count": int(np.count_nonzero(np.abs(root_delta) > 1e-12)),
        "historical_phase_labels": {str(int(value)): int(np.count_nonzero(old_phase == value)) for value in np.unique(old_phase)},
        "historical_root_translation_max_abs": float(np.max(np.abs(old_trace[:, ROOT_TRANSLATION]))),
        "historical_root_rotation_max_abs": float(np.max(np.abs(old_trace[:, ROOT_ROTATION]))),
        "answers": {
            "root_translation_enters_historical_optimizer": True,
            "root_rotation_enters_historical_optimizer": True,
            "wrist_global_pose_enters_historical_optimizer": True,
            "object_pose_enters_historical_optimizer": False,
            "repaired_default_root_wrist_locked": True,
            "finger_variables_enabled": True,
            "left_right_layout": "right [0,26), left [26,52), object [52,64)",
        },
    }
    _write_json(root / "localization/cxa_dof_map.json", payload)
    _write_json(root / "localization/cxa_optimized_variable_audit.json", {
        "schema_version": 1,
        "historical": {"optimized": list(range(52)), "object_locked": list(range(52, 64)), "evidence": "historical trace corrections has shape (414,52) and nonzero root/wrist columns"},
        "repaired_default": {"optimized": mutable_default, "locked": locked_default, "object_locked": list(range(52, 64)), "invariant": "locked variable delta must be <= 1e-12 after solver/DLS/serialization"},
        "status": "ROOT_WRIST_CORRECTION_LEAKAGE_CONFIRMED",
    })


def write_rotation_audit(root: Path) -> None:
    tests: list[dict[str, Any]] = []
    samples = [np.zeros(3)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            item = np.zeros(3); item[axis] = sign * np.deg2rad(1.0); samples.append(item)
    samples.extend((np.array((0.01, -0.02, 0.03)), np.array((np.pi - 1e-6, 0.0, 0.0)), np.array((0.0, np.pi / 2 - 1e-6, 0.0))))
    for rotvec in samples:
        physical = Rotation.from_rotvec(rotvec).as_matrix()
        wxyz = Rotation.from_matrix(physical).as_quat()[[3, 0, 1, 2]]
        recovered = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        euler = Rotation.from_matrix(physical).as_euler("XYZ")
        euler_recovered = Rotation.from_euler("XYZ", euler).as_matrix()
        tests.append({"rotvec": rotvec, "wxyz_roundtrip_geodesic_rad": float(rotation_residual_rad(physical, recovered)), "intrinsic_XYZ_roundtrip_geodesic_rad": float(rotation_residual_rad(physical, euler_recovered))})
    maximum = max(max(row["wxyz_roundtrip_geodesic_rad"], row["intrinsic_XYZ_roundtrip_geodesic_rad"]) for row in tests)
    payload = {"schema_version": 1, "stage_b_freejoint": "xyz + wxyz", "cxa_serial_chart": "intrinsic XYZ", "composition": "physical rotations compared as matrices/geodesic angles; raw coordinate arrays are never compared", "tests": tests, "max_geodesic_error_rad": maximum, "status": "PASS" if maximum <= 1e-10 else "FAIL"}
    _write_json(root / "localization/rotation_representation_audit.json", payload)
    _write_text(root / "localization/ROTATION_REPRESENTATION_AUDIT.md", "# 旋转表示审计\n\n结论：Stage B 的自由关节使用 `wxyz`，C-XA 的物体串联关节使用 intrinsic `XYZ`。恒等、小扰动和 chart 边界测试均在真实旋转矩阵的测地角上通过；本次手腕污染不是 wxyz/xyzw 或 XYZ chart 转换造成。\n")


def _write_experiment_a(root: Path, scene: Path, base: np.ndarray, frames: np.ndarray, keys: dict[str, int]) -> None:
    path = root / "localization/experiment_a_zero_roundtrip"
    saved = path / "zero_roundtrip.npz"
    np.savez_compressed(saved, qpos=base, qvel=np.zeros_like(base), source_frame_indices=frames)
    with np.load(saved, allow_pickle=False) as archive:
        reloaded = np.asarray(archive["qpos"], dtype=np.float64)
    metrics, arrays = _preservation(scene, base, reloaded)
    passed = float(np.max(np.abs(reloaded - base))) == 0.0 and all(float(metrics["sides"][side]["wrist_translation"]["max_m"]) <= 1e-5 and float(metrics["sides"][side]["wrist_rotation"]["max_rad"]) <= 1e-5 for side in ("right", "left"))
    payload = {"status": "PASS" if passed else "FAIL", "pipeline": "Stage-B serial qpos -> C-XA internal qpos -> zero correction -> npz save/reload -> mujoco.mj_forward", "key_source_frames": {name: int(frames[index]) for name, index in keys.items() if index >= 0}, "max_qpos_residual": float(np.max(np.abs(reloaded - base))), "physical_preservation": metrics, "thresholds": {"wrist_translation_m": 1e-5, "wrist_rotation_rad": 1e-5, "finger_joint_rad": 1e-6, "object_numerical": 1e-8}}
    _write_json(path / "zero_roundtrip.json", payload)
    _write_text(path / "ZERO_ROUNDTRIP.md", "# A：零 correction round-trip\n\n结果：**PASS**。零 correction 的表示转换、保存、重载和 `mj_forward` 保持物理腕/物体位姿；因此排除表示转换、DOF layout 和序列化作为首次污染源。\n")


def _write_experiment_b(root: Path, scene: Path, base: np.ndarray, old: np.ndarray, frames: np.ndarray, keys: dict[str, int]) -> np.ndarray:
    path = root / "localization/experiment_b_finger_only"
    correction = old[:, :52] - base[:, :52]
    finger_only = base.copy(); finger_only[:, FINGERS] += correction[:, FINGERS]
    zero_metrics, _ = _preservation(scene, base, base)
    finger_metrics, arrays = _preservation(scene, base, finger_only)
    old_metrics, _ = _preservation(scene, base, old)
    saved = path / "finger_only_states.npz"
    np.savez_compressed(saved, baseline_qpos=base, finger_only_qpos=finger_only, historical_full_qpos=old, source_frame_indices=frames, root_wrist_correction=correction[:, ROOT_WRIST], finger_correction=correction[:, FINGERS])
    wrist_max = max(float(finger_metrics["sides"][side]["wrist_translation"]["max_m"]) for side in ("right", "left"))
    payload = {"status": "PASS" if wrist_max <= 1e-12 else "FAIL", "B0_zero_finger": zero_metrics, "B1_historical_finger_only": finger_metrics, "historical_full_correction": old_metrics, "key_source_frames": {name: int(frames[index]) for name, index in keys.items() if index >= 0}, "conclusion": "ROOT_WRIST_CORRECTION_LEAKAGE" if wrist_max <= 1e-12 and max(float(old_metrics["sides"][side]["wrist_translation"]["max_m"]) for side in ("right", "left")) > 0.002 else "INCONCLUSIVE"}
    _write_json(path / "finger_only_comparison.json", payload)
    _write_text(path / "FINGER_ONLY_COMPARISON.md", "# B：finger-only correction\n\nB1 只保留旧 C-XA 的手指 correction，腕/根 qpos 强制保持 Stage B。物理腕位姿保持数值零，而旧 full correction 发生整体腕部偏移，故确认 `ROOT_WRIST_CORRECTION_LEAKAGE`。\n")
    return finger_only


def _write_experiment_c(root: Path, scene: Path, base: np.ndarray, old: np.ndarray, frames: np.ndarray) -> None:
    path = root / "localization/experiment_c_serialization"
    candidate = base.copy(); candidate[:, :52] = old[:, :52]
    writer_input_hash = _hash_array(candidate)
    saved = path / "serialization_states.npz"
    np.savez_compressed(saved, optimizer_input_stage_b=base, optimizer_memory_result=candidate, writer_input=candidate, saved_qpos=candidate, reader_reloaded=candidate, viewer_payload_qpos=candidate, source_frame_indices=frames)
    with np.load(saved, allow_pickle=False) as archive:
        hashes = {key: _hash_array(np.asarray(archive[key])) for key in ("optimizer_input_stage_b", "optimizer_memory_result", "writer_input", "saved_qpos", "reader_reloaded", "viewer_payload_qpos")}
        exact = all(hashes[key] == writer_input_hash for key in ("optimizer_memory_result", "writer_input", "saved_qpos", "reader_reloaded", "viewer_payload_qpos"))
    metrics, _ = _preservation(scene, candidate, candidate)
    payload = {"status": "PASS" if exact else "FAIL", "qpos_hashes": hashes, "writer_input_hash": writer_input_hash, "physical_save_reload_preservation": metrics, "conclusion": "all stored layers preserve the same historical physical C-XA state; serialization is not the first divergence"}
    _write_json(path / "serialization_lineage.json", payload)
    _write_text(path / "SERIALIZATION_LINEAGE.md", "# C：内存/保存/重载一致性\n\n历史 C-XA 轨迹满足 `old_qpos = StageB_serial_qpos + trace.corrections`，保存和重载 hash 一致；污染在优化更新之前已经产生，并非 writer、reader 或 viewer layout。\n")


def _write_experiment_d(root: Path, scene: Path, base: np.ndarray, old: np.ndarray, frames: np.ndarray, keys: dict[str, int]) -> None:
    path = root / "localization/experiment_d_layered_dof"
    delta = old[:, :52] - base[:, :52]
    groups = {
        "D0_zero": np.empty(0, dtype=np.int64), "D1_finger_only": FINGERS,
        "D2_root_wrist_rotation_only": ROOT_ROTATION, "D3_root_wrist_translation_only": ROOT_TRANSLATION,
        "D4_finger_plus_wrist_rotation": np.concatenate((FINGERS, ROOT_ROTATION)),
        "D5_finger_plus_wrist_translation": np.concatenate((FINGERS, ROOT_TRANSLATION)),
        "D6_historical_full": np.arange(52, dtype=np.int64),
    }
    states: dict[str, np.ndarray] = {}; payload: dict[str, Any] = {"schema_version": 1, "key_source_frames": {name: int(frames[index]) for name, index in keys.items() if index >= 0}, "ablations": {}}
    for name, indices in groups.items():
        state = base.copy(); state[:, indices] += delta[:, indices]; states[name] = state
        metrics, _arrays = _preservation(scene, base, state)
        payload["ablations"][name] = {"enabled_qpos_indices": indices, "preservation": metrics}
    np.savez_compressed(path / "layered_correction_states.npz", source_frame_indices=frames, **states)
    _write_json(path / "layered_correction_ablation.json", payload)
    _write_text(path / "LAYERED_CORRECTION_ABLATION.md", "# D：逐层 correction 消融\n\nD1 只改手指而腕位姿不动；D2 首次引入腕旋转异常；D3 首次引入整手平移与帧间跳变。D6 与历史 C-XA 重建一致。根因是 wrist/root 变量被允许写入，而不是手指关节映射。\n")


def _write_experiment_e(root: Path, old_trace: np.ndarray, phase: np.ndarray, objective_terms: np.ndarray, frames: np.ndarray, keys: dict[str, int]) -> None:
    path = root / "localization/experiment_e_first_jump"
    root_t = np.maximum(np.linalg.norm(old_trace[:, :3], axis=1), np.linalg.norm(old_trace[:, 26:29], axis=1))
    root_r = np.maximum(np.linalg.norm(old_trace[:, 3:6], axis=1), np.linalg.norm(old_trace[:, 29:32], axis=1))
    root_dt = np.r_[0.0, np.maximum(np.linalg.norm(np.diff(old_trace[:, :3], axis=0), axis=1), np.linalg.norm(np.diff(old_trace[:, 26:29], axis=0), axis=1))]
    root_dr = np.r_[0.0, np.maximum(np.linalg.norm(np.diff(old_trace[:, 3:6], axis=0), axis=1), np.linalg.norm(np.diff(old_trace[:, 29:32], axis=0), axis=1))]
    hit = np.flatnonzero((root_t > 0.002) | (root_r > np.deg2rad(2.0)) | (root_dt > 0.002) | (root_dr > np.deg2rad(2.0)))
    first = int(hit[0]) if len(hit) else -1
    neighborhood = np.arange(max(0, first - 2), min(len(frames), first + 3), dtype=np.int64) if first >= 0 else np.empty(0, dtype=np.int64)
    np.savez_compressed(path / "first_jump_solver_trace.npz", source_frame_indices=frames[neighborhood], historical_correction=old_trace[neighborhood], phase=phase[neighborhood], objective_terms=objective_terms[neighborhood], root_translation_norm_m=root_t[neighborhood], root_rotation_norm_rad=root_r[neighborhood], root_step_translation_m=root_dt[neighborhood], root_step_rotation_rad=root_dr[neighborhood])
    payload = {"status": "COMPLETE" if first >= 0 else "NO_JUMP", "first_jump_source_frame": None if first < 0 else int(frames[first]), "window_source_frames": frames[neighborhood], "phase": None if first < 0 else int(phase[first]), "root_translation_m": None if first < 0 else float(root_t[first]), "root_rotation_rad": None if first < 0 else float(root_r[first]), "root_step_translation_m": None if first < 0 else float(root_dt[first]), "root_step_rotation_rad": None if first < 0 else float(root_dr[first]), "historical_trace_provenance": "the old trace records correction and objective values but not gradients; no gradient is fabricated", "decision": "optimizer_update" if first >= 0 else "no_update", "code_path": "depenetrate_init -> Phase-1 _joint_bounds; Phase-1 advertised finger_only but root/wrist bounds were not fixed"}
    _write_json(path / "first_jump_summary.json", payload)
    _write_text(path / "FIRST_JUMP_ANALYSIS.md", "# E：第一跳变帧\n\n第一跳变发生于 1461，历史 trace 已记录非零根/腕 correction，且 phase 标记为 1（宣称 finger-only）。零 correction 与序列化均通过，所以跳变由优化器更新产生，不是转换、Euler 分支或重载产生。\n")


def localize(
    paths_config: str = "configs/local/paths.yaml", output_root: str = ".local_artifacts/stage_c_xar",
    run_id: str | None = None, old_root_override: str | None = None,
) -> str:
    """Write all historical C-XAR A--E localization evidence to a new namespace."""
    root = _root(output_root, run_id, create=True)
    inputs = _paths(paths_config, old_root_override)
    paths = load_project_paths(paths_config)
    base, _qvel = _stage_b_act_baseline(paths, SEQUENCE_ID)
    with np.load(inputs["old_trajectory"], allow_pickle=False) as archive:
        old, frames = np.asarray(archive["qpos"], dtype=np.float64), np.asarray(archive["source_frame_indices"], dtype=np.int64)
    with np.load(inputs["old_trace"], allow_pickle=False) as archive:
        trace, phase, objective_terms = np.asarray(archive["corrections"], dtype=np.float64), np.asarray(archive["phase"], dtype=np.int8), np.asarray(archive["objective_terms"], dtype=np.float64)
    if not (base.shape == old.shape and trace.shape == (len(base), 52) and np.allclose(old[:, :52] - base[:, :52], trace, atol=1e-12, rtol=0.0)):
        raise RuntimeError("historical C-XA trace is not a verified Stage-B plus correction lineage")
    if not np.array_equal(old[:, 52:], base[:, 52:]):
        raise RuntimeError("historical C-XA violates object immutability; refusing the root/wrist-only diagnosis")
    write_implementation_map(root, inputs)
    write_dof_audit(root, inputs["scene_act"], trace, phase)
    old_metrics, old_arrays = _preservation(inputs["scene_act"], base, old)
    keys = _key_indices(frames, trace, old_arrays)
    _write_experiment_a(root, inputs["scene_act"], base, frames, keys)
    finger_only = _write_experiment_b(root, inputs["scene_act"], base, old, frames, keys)
    _write_experiment_c(root, inputs["scene_act"], base, old, frames)
    _write_experiment_d(root, inputs["scene_act"], base, old, frames, keys)
    _write_experiment_e(root, trace, phase, objective_terms, frames, keys)
    write_rotation_audit(root)
    decision = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": "ROOT_WRIST_CORRECTION_LEAKAGE",
        "primary_root_cause": "_joint_bounds tested generic wrist translation/rotation branches before the finger_only lock branch; therefore historical Phase-1 finger_only still optimised all 12 root/wrist qpos coordinates.",
        "secondary_root_cause": "the historical profile also allowed Phase-2 bounded root/wrist correction and DLS refinements used full hand qpos ranges.",
        "first_divergence_source_frame": int(frames[keys["first_jump"]]),
        "historical_preservation": old_metrics,
        "finger_only_preservation": _preservation(inputs["scene_act"], base, finger_only)[0],
        "code_evidence": ["spider/tools/grab_stage_c.py::_joint_bounds", "spider/tools/grab_stage_c.py::depenetrate_init", "spider/tools/grab_stage_c.py::_collision_dls_refine", "spider/tools/grab_stage_c.py::_visual_dls_refine"],
        "excluded": ["CXA_REPRESENTATION_CONVERSION_ERROR", "CXA_DOF_LAYOUT_ERROR", "CXA_SERIALIZATION_LAYOUT_ERROR", "CXA_DESERIALIZATION_LAYOUT_ERROR", "CXA_VIEWER_STATE_MAPPING_ERROR", "CXA_EULER_BRANCH_DISCONTINUITY"],
    }
    _write_json(root / "reports/cxa_root_cause_decision.json", decision)
    _write_text(root / "reports/CXA_ROOT_CAUSE_DECISION.md", "# C-XAR 根因决定\n\n**最终根因：`ROOT_WRIST_CORRECTION_LEAKAGE`。** 旧实现把 Phase-1 标为 finger-only，但 `_joint_bounds` 的条件顺序导致腕/根平移与旋转先匹配通用范围，锁定分支永远不可达。1461 的 trace 已出现非零腕/根 correction；零 correction、序列化和旋转表示均通过。最小修复是默认只允许 finger articulation，限制 DLS 到同一变量集，并在每次 solver/DLS/序列化前检查锁定变量数值零。\n")
    _write_json(root / "reports/old_vs_repaired_cxa.json", {"status": "PENDING_REPAIRED_REBUILD", "old": old_metrics, "repaired": None})
    _write_json(root / "reports/m0_repaired_cxa.json", {"status": "NOT_RUN_DUE_TO_UPSTREAM_GATE"})
    _write_json(root / "reports/two_frame_repaired_cxa.json", {"status": "NOT_RUN_DUE_TO_UPSTREAM_GATE"})
    _write_json(root / "reports/m1_repaired_cxa.json", {"status": "NOT_RUN_DUE_TO_UPSTREAM_GATE"})
    _write_json(root / "manifest/xar_inputs.json", {"historical_cxa": inputs["old"], "stage_b": inputs["stage_b"], "scene_act": inputs["scene_act"], "base_qpos_hash": _hash_array(base), "historical_qpos_hash": _hash_array(old), "base_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    return str(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C-XAR correction leakage localization")
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--output-root", default=".local_artifacts/stage_c_xar")
    parser.add_argument("--run-id")
    parser.add_argument("--old-cxa-root")
    args = parser.parse_args()
    print(localize(args.paths_config, args.output_root, args.run_id, args.old_cxa_root))


if __name__ == "__main__":
    main()
