# Stage C C-M1R contact-mode repair

## Scope and status

This repair is limited to frozen primary `s5/cylindermedium_lift`, source
frames `1461..1480`, 120 Hz, and the immutable left-index `SUPPORT` role
`s5__cylindermedium_lift:0` on
`patch:s5__cylindermedium_lift:0`.  It never changes the 20 mm threshold,
source object target, timing, patch, role, or assigned finger.

The final isolated evidence is
`.local_artifacts/stage_c_cm1r/20260801T140000Z-cm1r/`.

| gate | result |
| --- | --- |
| R1 profile integrity | PASS |
| R2 initialization and transitions | PASS |
| R3 object-local contact servo | PASS |
| M0 initial hold | PASS |
| M1 moving retention | FAIL after 8 bounded repairs |
| M2 injected regrasp | NOT_RUN (M1 gate) |
| M3 full window | NOT_RUN (M2 gate) |
| real 3D HTML and Chrome capture | PASS |

This is not a dynamic-contact infeasibility conclusion.  M1 failure is a
remaining retention/controller failure; the next work remains V2 contact-mode
implementation repair, not V3.

## Implemented repairs

- Every profile is normalized and hashed before execution. Equivalent profiles
  are rejected, and the matrix builder proves all confirmation, timeout,
  hysteresis, acquire-distance, and regrasp-attempt dimensions are expressible.
- Initial valid physical contact enters `RETAIN_PENDING`, confirms for bounded
  substeps, then enters `RETAIN`; only absent physical contact enters
  `ACQUIRE`.
- A provisional physical contact lost during `ACQUIRE` enters `REGRASP` on the
  next observation. Regrasp timeout resets into a real second attempt when the
  configured maximum is two.
- The contact target is object-local and is recomputed in world coordinates at
  each MuJoCo substep. The Jacobian correction is limited to left wrist and
  left-index actuator columns. Controller transitions start with the prior
  command and blend through a bounded ramp.
- The 3D viewer contains full connected Wuji and object meshes, collision
  layers, a connected semantic patch, actual contacts/normals/forces, target
  and fingertip markers, source skeletons, Stage B/C-XA/reference layers, and
  synchronized metrics.

## Measured result

M0 entered `RETAIN_PENDING` at substep 0 and `RETAIN` at substep 4. It held
the assigned physical pair continuously for 20 ms, with patch P95
`0.007174894 m`, peak force `0.336134 N`, maximum penetration
`0.000254550 m`, minimum margin `0.083622`, no warnings, and no post-init
robot/object qpos writes. The old run measured `110.055558 N` at frame 1461.

M1 used source frames `1461..1466` with normal MuJoCo stepping and object
mocap targets. All eight targeted repairs first lost the assigned pair at
source frame 1462 / simulation step 2. The selected
`m1_wrist_index_phase_lead` candidate retained one of six frame samples,
reported patch P95 `0.020345737 m`, and never restored the pair. This is
`RETENTION_FAILURE`; do not run M2 or M3 from it.
