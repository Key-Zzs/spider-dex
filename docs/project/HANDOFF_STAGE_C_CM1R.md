# Stage C C-M1R handoff

## Status

- C-M1 old software structure: `PASS`.
- Old C-M2 conclusion: `INVALID_OR_INCOMPLETE_FOR_CONTACT_MODE_CONCLUSION`.
- C-M1R implementation (R1/R2/R3): `PASS`.
- M0 initial hold: `PASS`.
- M1 moving retention: `FAIL` after eight bounded implementation repairs.
- M2 injected regrasp: `NOT_RUN`; M1 did not pass.
- M3 frozen 1461..1480 window: `NOT_RUN`; M2 did not run.
- Dynamic witness: `NOT_FOUND`.
- Real 3D visualization / Chrome screenshots: `PASS` (30 required captures plus comparison views).
- Full primary, Oracle C full, D2, MJWP, smokes: `NOT_RUN`; Stage D: `NOT_STARTED`.
- User visual review: `PENDING`.

## Contract and preservation

Repository `/home/deepcybo/workspace/dex/retarget/spider-dex`, branch
`develop/wuji-hand2`, base commit `f14cbd7efbbd29068cfaa900f977b46c42759aa0`,
conda `spider-dex`.  Frozen primary is `s5/cylindermedium_lift`, source frames
`[1460,1876)`, 120 Hz; this run uses `1461..1480`.  The historical first loss
is left-index `SUPPORT`, role `s5__cylindermedium_lift:0`, patch
`patch:s5__cylindermedium_lift:0`, source frame 1465.  Role, assigned finger,
patch, 20 mm threshold, timing, source object targets, raw GRAB, and body
models were unchanged. Object guidance uses mocap targets; robot/object qpos
are initialized once and never written during stepping.

## What changed

`spider/contact/contact_mode.py` adds `RETAIN_PENDING`, initial-contact
classification, immediate provisional-loss regrasp, and real two-attempt
regrasp resets. `spider/tools/grab_stage_c_cm1r.py` implements normalized
effective profiles and hashes, object-local targets, left wrist/index Jacobian
control, bumpless ramps, M0/M1 gates, complete failure localization, and
ignored artifacts. `spider/tools/grab_stage_c_cm1r_viewer.py` creates a
full-mesh Plotly/WebGL diagnostic. Tests are in `tests/test_stage_c_cm1r.py`.

## Results and next action

M0 physically held the exact pair: 100% contact continuity, P95
`0.007174894 m`, peak `0.336134 N` (versus old `110.055558 N`), penetration
`0.000254550 m`, margin `0.083622`, and valid `PRE_CONTACT →
RETAIN_PENDING → RETAIN → COMPLETE` transitions.

M1 first lost the exact pair at source frame 1462 / simulation step 2 for all
eight targeted profiles. The selected phase-lead repair had continuity `1/6`
and patch P95 `0.020345737 m`; it is `RETENTION_FAILURE`, not a valid
contact-mode infeasibility result. Continue V2 retention/longer-horizon
controller repair from M1 only. Do not enter V3 or run M2/M3/MJWP/Stage D.

## Evidence

Run root: `.local_artifacts/stage_c_cm1r/20260801T140000Z-cm1r/`.

- `reports/cm1r_old_experiment_validity_audit.json`
- `reports/profile_matrix_coverage.json`
- `reports/m0_initial_hold_summary.json`
- `reports/m1_moving_retention_summary.json`
- `reports/cm1r_manual_visual_review.json`
- `html/stage_c_cm1r_contact_mode.html`
- `html/stage_c_cm1r_visual_index.html`
- `screenshots/repaired_close/`
