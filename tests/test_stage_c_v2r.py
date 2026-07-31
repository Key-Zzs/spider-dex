"""Pure decision and schema tests for the Stage C-V2R diagnostic gate."""

from __future__ import annotations

import unittest

import mujoco

from spider.tools.grab_stage_c_v2r import (
    _apply_contact_dynamics_profile,
    _disable_hand_object_explicit_pairs,
    _is_hand_object_pair,
    _payload_hash,
    _r2_candidate_gates,
    classify_oracles,
)


def _oracle(status: str) -> dict[str, str]:
    return {"status": status}


class V2RDecisionTest(unittest.TestCase):
    def test_reference_failure_has_priority(self) -> None:
        result = classify_oracles(_oracle("FAIL"), _oracle("PASS"), _oracle("PASS"), _oracle("PASS"))
        self.assertEqual(result["primary_cause"], "REFERENCE_TRAJECTORY_TIMING_INFEASIBLE")

    def test_controller_failure_is_not_overstated_as_independent_contact_failure(self) -> None:
        result = classify_oracles(_oracle("PASS"), _oracle("PASS"), _oracle("FAIL"), _oracle("FAIL"))
        self.assertEqual(result["primary_cause"], "CONTROLLER_TRACKING_INSUFFICIENT")
        self.assertEqual(result["secondary_causes"], [])

    def test_contact_coupling_requires_a_and_d_to_pass(self) -> None:
        result = classify_oracles(_oracle("PASS"), _oracle("PASS"), _oracle("FAIL"), _oracle("PASS"))
        self.assertEqual(result["primary_cause"], "CONTACT_COLLISION_COUPLING")

    def test_object_guidance_requires_perfect_hand_failure_after_c_passes(self) -> None:
        result = classify_oracles(_oracle("PASS"), _oracle("FAIL"), _oracle("PASS"), _oracle("PASS"))
        self.assertEqual(result["primary_cause"], "OBJECT_GUIDANCE_CONTACT_INCOMPATIBLE")

    def test_r2_gates_fail_closed_on_one_missing_patch_metric(self) -> None:
        report = {"gates": {key: True for key in ("finite", "no_warnings", "joint_limits", "joint_margin", "smoothness", "tracking", "object_tracking", "collision_depth")}, "patch": {"gates": {"patch_coverage": True, "functional_role_recall": True, "patch_distance_p95": False, "normal_alignment": True}}}
        force = {"p95_n": 1.0, "max_n": 1.0, "impulse_ns": 1.0}
        thresholds = {"p95_n": 2.0, "max_n": 2.0, "impulse_ns": 2.0}
        self.assertFalse(all(_r2_candidate_gates(report, force, thresholds).values()))

    def test_profile_payload_hash_is_order_independent(self) -> None:
        self.assertEqual(_payload_hash({"a": 1, "b": 2}), _payload_hash({"b": 2, "a": 1}))

    def test_oracle_d_filter_excludes_only_hand_object_pairs(self) -> None:
        hand, objects = {2, 4}, {85}
        self.assertTrue(_is_hand_object_pair(2, 85, hand, objects))
        self.assertFalse(_is_hand_object_pair(2, 4, hand, objects))
        self.assertFalse(_is_hand_object_pair(85, 86, hand, objects))

    def test_oracle_d_disables_only_explicit_hand_object_pairs(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <geom name="floor" type="plane" size="1 1 0.1"/>
                <body pos="0 0 1"><geom name="hand" type="sphere" size="0.1"/></body>
                <body pos="0.3 0 1"><geom name="object" type="sphere" size="0.1"/></body>
                <body pos="0.6 0 1"><geom name="self" type="sphere" size="0.1"/></body>
              </worldbody>
              <contact>
                <pair geom1="hand" geom2="object"/>
                <pair geom1="hand" geom2="self"/>
              </contact>
            </mujoco>
            """
        )
        hand = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand")}
        objects = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object")}
        self_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "self")
        policy = _disable_hand_object_explicit_pairs(model, hand, objects)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.assertEqual(policy["explicit_pair_count"], 1)
        self.assertEqual(tuple(model.pair_geom1), (floor, next(iter(hand))))
        self.assertEqual(tuple(model.pair_geom2), (floor, self_geom))

    def test_contact_profile_changes_only_hand_object_pair_parameters(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <geom name="floor" type="plane" size="1 1 0.1"/>
                <body pos="0 0 1"><geom name="hand" type="sphere" size="0.1"/></body>
                <body pos="0.3 0 1"><geom name="object" type="sphere" size="0.1"/></body>
                <body pos="0.6 0 1"><geom name="self" type="sphere" size="0.1"/></body>
              </worldbody>
              <contact>
                <pair geom1="hand" geom2="object"/>
                <pair geom1="hand" geom2="self"/>
              </contact>
            </mujoco>
            """
        )
        hand = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand")}
        objects = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object")}
        self_solref = model.pair_solref[1].copy()
        result = _apply_contact_dynamics_profile(
            model,
            {
                "candidate_id": "test",
                "solref": [0.03, 1.0],
                "solimp": [0.8, 0.9, 0.002, 0.5, 2.0],
                "margin_m": 0.001,
                "gap_m": 0.0,
                "friction": [0.7, 0.7, 0.07, 0.0, 0.0],
            },
            hand,
            objects,
        )
        self.assertEqual(result["candidate_id"], "test")
        self.assertEqual(model.pair_solref[0].tolist(), [0.03, 1.0])
        self.assertEqual(model.pair_margin[0], 0.001)
        self.assertEqual(model.pair_solref[1].tolist(), self_solref.tolist())


if __name__ == "__main__":
    unittest.main()
