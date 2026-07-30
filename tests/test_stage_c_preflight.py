"""Pure MuJoCo contract tests for the C-R3 physical reference controller."""

import unittest

import mujoco
import numpy as np

from spider.tools.grab_stage_c import _preflight_object_ids, _set_object_mocap_reference


class PreflightControllerTest(unittest.TestCase):
    def test_reference_updates_mocap_without_writing_object_qpos(self) -> None:
        xml = """
        <mujoco><worldbody>
          <body name='right_object'><joint name='rx' type='slide'/><joint name='ry' type='slide'/><joint name='rz' type='slide'/><joint name='rrx' type='hinge'/><joint name='rry' type='hinge'/><joint name='rrz' type='hinge'/><geom type='sphere' size='0.01'/></body>
          <body name='left_object'><joint name='lx' type='slide'/><joint name='ly' type='slide'/><joint name='lz' type='slide'/><joint name='lrx' type='hinge'/><joint name='lry' type='hinge'/><joint name='lrz' type='hinge'/><geom type='sphere' size='0.01'/></body>
          <body name='right_object_mocap_target' mocap='true'/><body name='left_object_mocap_target' mocap='true'/>
        </worldbody></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        bodies, mocap = _preflight_object_ids(model)
        qpos = np.zeros(64)
        qpos[52:58] = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1]
        qpos[58:64] = [-0.2, 0.1, 0.4, -0.3, 0.1, -0.2]
        before = data.qpos.copy()
        _set_object_mocap_reference(data, qpos, mocap)
        self.assertTrue(np.array_equal(data.qpos, before))
        self.assertTrue(np.allclose(data.mocap_pos[mocap["right"]], qpos[52:55]))
        self.assertTrue(np.allclose(data.mocap_pos[mocap["left"]], qpos[58:61]))
        self.assertEqual(set(bodies), {"right", "left"})


if __name__ == "__main__":
    unittest.main()
