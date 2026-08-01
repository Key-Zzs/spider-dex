# Stage C C-M1/C-M2 Contact-Mode Transition

Status: `C-M1 PASS`; `C-M2 EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS`.

This implementation is deliberately bounded to the frozen first-loss window
for `s5/cylindermedium_lift`: source frames `1461..1480`, local frames `0..19`,
120 Hz, real MuJoCo integration timestep `0.0005 s`. It does not claim full
Stage C acceptance.

## Contract preserved

- V2 patch threshold remains `0.020 m`; coverage and role-recall gates remain
  `0.70` and `0.80`; normal cosine remains `0.50`.
- The immutable role is left index distal / `SUPPORT`, role
  `s5__cylindermedium_lift:0`, patch
  `patch:s5__cylindermedium_lift:0`, robot region
  `left_index_fingertip`.
- Source timing, source/object targets, frozen primary, raw GRAB, body models,
  and historical diagnostic artifacts were not edited.
- Object pose is supplied only through the existing mocap target path; the
  runner never writes object qpos and never rewrites robot qpos after initial
  state assignment.

## C-M1 state machine

`PRE_CONTACT -> ACQUIRE -> RETAIN -> RELEASE -> COMPLETE` is the nominal path.
Contact loss or acquire timeout enters bounded `REGRASP`; a successful
confirmation returns to `RETAIN`, while timeout or hard safety failure enters
`FAILED`. The implementation is in
`spider/contact/contact_mode.py` and records every transition and observation.

The bounded defaults used in the run were acquire entry `0.025 m`, retain
hysteresis `0.025 m`, four confirmation substeps, 40 ms acquire/regrasp
timeouts, and one regrasp attempt for E1/E2 (two for E3). The evaluator still
uses `0.020 m`; hysteresis is only a mode-switch guard.

## C-M2 result

E0 reused the immutable historical diagnostic and reproduced the recorded
first-loss evidence at source frame `1465` (local frame `4`); it was not
rerun. E1/E2/E3 ran 12 fixed profiles. No profile produced a dynamic witness.
The best failed profile was `e3_high_kp_anchor`:

| metric | result |
|---|---:|
| correct assigned contact frames | 2/20 |
| functional-role recall | 0.10 |
| patch coverage | 0.10 |
| patch-distance P95 | 0.052461 m |
| normal-cosine median | 1.0 |
| terminal correct contact | false |
| max MuJoCo penetration | below 0.003 m |
| max force | below 150 N |
| normalized one-frame delta | below 0.25 |

The selected profile entered `ACQUIRE`, then bounded `REGRASP`, then
`FAILED` with `REGRASP_TIMEOUT`. All finite/warning, joint-limit,
penetration, force, robot-tracking, and object-tracking gates passed; the
contact contract and terminal gates failed. This is empirical bounded
evidence, not a mathematical infeasibility proof.

## Evidence

The authoritative new run is:

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode/`

It contains the 12 per-profile JSON/NPZ/timeline records, the historical E0
summary, selected profile, transition summary, HTML, 24 Chrome PNGs, and
manual diagnostic review. `.local_artifacts/` is ignored by Git.

Full primary, Oracle C full sequence, D2, Minimal/Full MJWP, smokes, and Stage
D remain `NOT_RUN`. Since no bounded witness exists, the next decision is to
upgrade the V2 contact planner/optimizer; do not enter V3 based on this run.
