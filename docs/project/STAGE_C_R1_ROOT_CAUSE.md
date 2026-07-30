# Stage C-R1: collision assets and initial-penetration root cause

Run the audit on a frozen pilot without changing Stage B:

```bash
conda run --no-capture-output -n spider-dex \
  python spider/tools/grab_stage_c.py collision-audit \
  --paths-config configs/local/paths.yaml \
  --sequence-id s5__cylindermedium_lift
```

The primary report is written outside Git at
`<workspace>/reports/stage_c_r1_root_cause.json` and deliberately keeps these
layers separate:

1. reconstructed source-human visual mesh, plus source fingertip regions;
2. Stage B Wuji group-1 visual mesh; and
3. Stage C group-2 MuJoCo collision contacts against CoACD pieces.

The audit also compares the 66-qpos Stage B free-joint scene to the isolated
64-qpos actuator scene.  The latter uses three serial X/Y/Z hinge coordinates:
the valid mapping is a quaternion to intrinsic `XYZ` Euler conversion, not a
quaternion rotation vector.  This is an implementation detail of the existing
scene layout and is verified by a body-pose equivalence gate.

The old direct-physics failure profile and report remain immutable historical
evidence.  C-R1 does not run an optimizer, alter object poses, or modify raw
GRAB or Stage B trajectories.
