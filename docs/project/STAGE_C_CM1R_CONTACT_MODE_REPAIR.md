# Stage C C-M1R contact-mode repair

## Frozen scope

This work remains limited to `s5/cylindermedium_lift`, frames `1461..1480`,
120 Hz, with the immutable left-index SUPPORT assignment on
`patch:s5__cylindermedium_lift:0`. The 20 mm patch gate, source timing,
object trajectory, role, finger, raw GRAB, and body models are unchanged.

## M1 recovery result

Latest evidence is in
`.local_artifacts/stage_c_cm1r/20260801T072352Z-m1-recovery/`.

| gate | status |
| --- | --- |
| M0 static regression | PASS |
| 1461→1462 causal audit | PASS |
| S0 static hold | PASS |
| S1 object motion only | FAIL |
| S2 hand target motion only | PASS |
| S3 both, position feedback | FAIL |
| S4 both, rigid-motion feedforward | FAIL |
| S5 source-derived initial velocity | FAIL_SAFE_FORCE |
| two-frame retention | FAIL |
| M1 | FAIL |
| M2/M3 | NOT_RUN |

## Implemented common repairs

1. Source object and robot targets are interpolated per simulation substep;
   the old whole-frame object target update was a timing defect.
2. The object-local target uses current actual object pose and records
   target/source/control/step times separately.
3. The retained-controller ramp is continuous across
   `RETAIN_PENDING → RETAIN`.
4. Source rigid-body contact velocity is available as a deterministic
   feedforward term and source-derived t0 velocity initialization is audited.

The remaining failure is object-coupled normal separation, not a relaxed
evaluator, fake regrasp, or insufficient legal wrist/index DOFs. M2 and M3
must remain gated until a safe two-frame and M1 witness exists.
