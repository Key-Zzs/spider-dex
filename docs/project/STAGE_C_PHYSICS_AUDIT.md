# Stage C physics audit

Stage C uses the local `examples/run_mjwp.py` contact-guidance path, not the
fast runner: `run_mjwp_fast.py` explicitly rejects `contact_guidance`.
`process_config()` selects `scene_act.xml` and `trajectory_kinematic_act.npz`
when guidance is enabled.  The normal Stage B input is
`trajectory_kinematic.npz`; it contains free-joint object xyz+quaternion state
(66 qpos), while `scene_act.xml` represents the two objects as xyz+rotvec
(64 qpos).  `spider.tools.grab_stage_c.prepare-physics-input` makes that
conversion in an isolated external sandbox and creates `ctrl` explicitly.

The runner initializes MJ-Warp from the kinematic state, optimizes controls
with `spider.optimizers.sampling`, and writes `trajectory_mjwp_act.npz` beside
its sandbox input. Contact reward uses actual Wuji tip site IDs, not contact
channel numbers. Object actuator controls and reward/contact guidance are the
current relevant robot-specific indices; the Stage C adapter resolves them by
MuJoCo names.

`decompose_fast.py`, `decompose.py`, and `generate_xml.py` are the existing
collision workflow. Stage C reuses `generate_xml.py`; it uses CoACD only to
make a hash-addressed collision cache external to the checkout. The visual mesh
and convex collision pieces remain separate.

## Current result

The primary contact-guided GPU probe is a hard failure, not an optimization
pass: all 16 sampled rewards are NaN on its first rollout. CPU MuJoCo replay
shows escalating collision acceleration and a huge-QPOS/QACC warning at 0.04 s.
The frozen Stage B kinematic trajectory starts deeply interpenetrating the
object, so MJWP has no finite physical start state. This blocks profile search,
accepted optimized output, metrics, HTML and screenshot review. Do not use a
static hand offset, delete frames, or relax gates as a workaround.
