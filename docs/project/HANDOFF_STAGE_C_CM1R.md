# Stage C C-M1R handoff

## Current status

- Base: `e9f248ec6aa581ae139fae8d8f7f5a32a858b65b` on `develop/wuji-hand2`.
- Frozen role: left-index `SUPPORT`, `s5__cylindermedium_lift:0`,
  `patch:s5__cylindermedium_lift:0`; source frames remain `1461..1480` at
  120 Hz.
- Historical M1 is preserved as `HISTORICAL_M1_FAILURE` under
  `.local_artifacts/stage_c_cm1r/20260801T140000Z-cm1r/`.
- Latest recovery run:
  `.local_artifacts/stage_c_cm1r/20260801T072352Z-m1-recovery/`.
- M0 regression: `PASS` (100% continuity, P95 7.22 mm, peak 0.339 N).
- 1461→1462 causal audit: `PASS`.
- Two-frame retention: `FAIL`; M1: `FAIL`; M2/M3: `NOT_RUN`.
- Dynamic witness: `NOT_FOUND`; user visual review: `PENDING`.

## Confirmed repairs

The shared loop now interpolates frozen 120-Hz robot/object endpoints at each
0.5-ms MuJoCo substep, uses the actual object pose for object-local contact
targets, and records separate pre/post-step times. The bumpless controller no
longer resets at `RETAIN_PENDING → RETAIN`, avoiding a second zero-alpha
command during motion. The M1 profiles are a four-item causal sequence, not a
Kp/lead grid.

## Remaining blocker

The specific remaining classification is
`TARGET_TIME_ALIGNMENT_ERROR_PLUS_OBJECT_COUPLED_NORMAL_SEPARATION`.
Interpolation delays the historical loss from substep 2 to substep 8 and
allows RETAIN entry, but it does not retain the assigned pair through frame
1462. Object-only motion fails while hand-only motion passes. A source-derived
contact-consistent velocity initialization gives a 0.00146 m/s fingertip
velocity residual but creates a 107.35 N peak, so it is rejected rather than
presented as a witness. This is not an empirical-infeasibility conclusion.

## Boundaries

No raw GRAB/body model/role/patch/timing/object target changed. There are no
post-initialization robot or object qpos writes. Full primary, Oracle C, D2,
MJWP, smokes, and Stage D remain unrun.
