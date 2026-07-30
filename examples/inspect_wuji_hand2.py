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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("right", "left", "bimanual"), default="right")
    parser.add_argument("--mode", choices=("neutral", "sweep"), default="neutral")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--show-sites", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless/sweep duration in seconds.")
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
            # Wrist controls remain neutral; sweep one finger actuator at a
            # time through its safe central 60 percent range.
            finger_id = 6 + int(elapsed * 1.2) % 20
            low, high = model.actuator_ctrlrange[finger_id]
            data.ctrl[finger_id] = (low + high) / 2 + 0.3 * (high - low) * np.sin(elapsed * 2.0)
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


if __name__ == "__main__":
    main()
