"""Inspect the data-free Wuji Hand2 Beta1 adapter in MuJoCo.

Examples:
    python examples/inspect_wuji_hand2.py --side right --mode neutral
    python examples/inspect_wuji_hand2.py --side bimanual --mode sweep --show-sites
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from spider.assets import get_packaged_robot_asset_dir


def _model_name(side: str) -> str:
    return "bimanual.xml" if side == "bimanual" else f"{side}.xml"


def _apply_sweep(model: mujoco.MjModel, ctrl: np.ndarray, elapsed: float) -> tuple[int, int]:
    """Pulse one whole finger and return its ``(hand, finger)`` indices.

    Controls are ordered as six wrist DOFs followed by five groups of four
    finger joints.  The bimanual adapter concatenates complete right and left
    26-DOF hand segments.  Alternating hands makes both segments visibly
    testable while keeping the inactive hand at its neutral pose.
    """
    controls_per_hand = 26
    wrist_controls = 6
    controls_per_finger = 4
    fingers_per_hand = 5
    hand_count = model.nu // controls_per_hand
    if model.nu != hand_count * controls_per_hand or hand_count not in (1, 2):
        raise ValueError(f"Unexpected Wuji actuator layout: nu={model.nu}")

    phase_duration = 1.0
    phase_index = int(elapsed // phase_duration)
    hand_index = phase_index % hand_count
    finger_index = (phase_index // hand_count) % fingers_per_hand
    phase = (elapsed % phase_duration) / phase_duration
    amplitude = 0.30 * np.sin(np.pi * phase)

    first = hand_index * controls_per_hand + wrist_controls + finger_index * controls_per_finger
    actuator_ids = np.arange(first, first + controls_per_finger)
    low = model.actuator_ctrlrange[actuator_ids, 0]
    high = model.actuator_ctrlrange[actuator_ids, 1]
    ctrl[actuator_ids] = (low + high) / 2 + amplitude * (high - low)
    return hand_index, finger_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("right", "left", "bimanual"), default="right")
    parser.add_argument("--mode", choices=("neutral", "sweep"), default="neutral")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--show-sites", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless/sweep duration in seconds.")
    parser.add_argument("--no-viewer", action="store_true", help="Run the same smoke motion without opening a viewer.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(get_packaged_robot_asset_dir("wuji_hand2_beta1") / _model_name(args.side)))
    data = mujoco.MjData(model)
    neutral = np.clip(0.0, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[int(model.jnt_qposadr[joint_id])] = neutral[actuator_id]
    mujoco.mj_forward(model, data)
    start = time.monotonic()

    def step() -> None:
        elapsed = time.monotonic() - start
        data.ctrl[:] = neutral
        if args.mode == "sweep":
            # Wrist controls remain neutral.  Pulse all four joints of one
            # finger through the central 60 percent of its control range.
            # For bimanual models, phases alternate right/left so both control
            # segments are exercised independently.
            _apply_sweep(model, data.ctrl, elapsed)
        mujoco.mj_step(model, data)

    if args.no_viewer:
        while time.monotonic() - start < args.duration:
            step()
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Non-finite state during Wuji smoke motion")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[2] = int(args.show_collision)
        viewer.opt.sitegroup[3] = int(args.show_sites)
        while viewer.is_running() and time.monotonic() - start < args.duration:
            step()
            viewer.sync()
            time.sleep(model.opt.timestep)

    # `launch_passive` owns a GLFW render thread.  On this host, returning to
    # Python immediately after the context closes can race GLFW/X11 teardown
    # and segfault during interpreter shutdown.  Give the render thread a
    # short, bounded interval to finish before Python finalizes its modules.
    time.sleep(0.25)


if __name__ == "__main__":
    main()
