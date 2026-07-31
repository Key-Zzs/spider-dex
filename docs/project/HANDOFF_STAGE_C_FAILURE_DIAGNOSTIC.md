# Stage C-V2R2E failure-diagnostic handoff

## Status

- Stage A: PASS
- Stage B: PASS
- V1 exact contact: BLOCKED
- C-XA corrected V2 static: PASS
- Stage C-V2R2E dynamic: BLOCKED
- Failure visualization: PASS
- Feasibility certificate: COMPLETE
- Decision: `EXTEND_V2_CONTACT_MODE_TRANSITION`
- MJWP: NOT_RUN
- smokes: NOT_RUN
- Stage D: NOT_STARTED

## Environment and Git state

Repository `/home/deepcybo/workspace/dex/retarget/spider-dex`, branch
`develop/wuji-hand2`, diagnostic base HEAD
`f78cfeda29b01ca0cd896ed1efe51642da28177a`, conda environment `spider-dex`.
The worktree already contained user-owned staged and unstaged Stage C changes;
they were preserved. `.local_artifacts/` is Git ignored. No push was performed.

## Read-only lineage

The authoritative aggregate is
`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/stage_c_v2r2e_validation.json`.
The immutable attempt is
`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260801T004500Z-contactik/`.
The best historical candidate is `lead8_feedforward` (coverage `0.317039`, role
recall `0.117647`, patch P95 `0.081810 m`). Every input path, size, mtime, hash,
semantic role, and read-only state is recorded in `input_artifact_manifest.json`.

## First failure and visual evidence

The exact first crossing is source frame 1465, trace record/simulation step 50,
substep 0. The left index `SUPPORT` role
`s5__cylindermedium_lift:0` leaves semantic patch
`patch:s5__cylindermedium_lift:0`: distance rises from `0.016573 m` to
`0.020829 m`. There is no actual hand–object geom pair, contact force, or
penetration at the crossing. Force/tracking/joint-margin failures occur later,
so this is early physical contact absence, not a force spike, joint-limit
block, or evaluator-only loss.

The viewer now renders complete left/right Wuji surfaces and collision proxies
for Stage B, C-XA, failed reference, and failed actual states. It also renders
the connected 382-face semantic patch surface, separate object
visual/collision meshes, contact normals/forces, and source hand skeletons.

Forty-seven Chrome screenshots over 14 distinct source frames were generated
and inspected. Frames 1464 and 1465 share the same close camera. Frame 1464
remains on the allowed side of the patch bound; frame 1465 separates the
actual left index from the green lower end-face patch and contains no physical
contact marker. The first earlier physical pair is frame 1462, substep 0,
`collision_hand_left_index_8 ↔ right_object_0`. The visual/collision layer
does not show a thin-wall mapping inversion, and the evaluator agrees with the
reconstructed physical absence.

The motion is retention loss/separation, not an initial bounce or collision
block. Force/tracking lag, joint margin, and collision depth become failures
only at source frames 1502, 1622, and 1708 respectively. The complete manual
review is `manual_visual_review.json` and `SCREENSHOT_REVIEW.md`.

## Certificates and decision

Simultaneous contact is `FEASIBLE`: 12 deterministic starts produce a complete
witness for the single active role with all frozen patch, penetration, margin,
tracking, and self-collision constraints passing. There is no role conflict,
so ablation is `NOT_REQUIRED_FULL_SET_FEASIBLE` and the minimum conflict set is
empty.

Original-timing transition is
`EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS`: six real-MuJoCo starts over local
frames `[0,19]` preserve penetration, joint margin, smooth one-frame motion,
finite force, and the source object targets, but all fail contact continuity
and terminal patch distance. This is bounded empirical evidence only.

Decision confidence is `0.78`. Implement `V2_CONTACT_MODE_TRANSITION` with
explicit bounded acquisition, retention, release, and regrasp states. Do not
lower recall, widen patches, delete roles, write object qpos, replace the
primary, or start MJWP/smokes/Stage D. V3 is not justified because the complete
active role set has a valid same-frame Wuji witness.

## Outputs and next entry

Local output root:
`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_v2r2e_failure_diagnostic/20260731T180415Z/`.

- failure HTML: `stage_c_v2r2e_failure_diagnostic.html`
- index: `failure_diagnostic_index.html`
- screenshots: `screenshots/`
- manual visual review: `manual_visual_review.json`
- timeline: `first_failure_timeline.json`
- simultaneous certificate: `simultaneous_contact_feasibility.json`
- dynamic certificate: `dynamic_transition_feasibility.json`
- decision: `v2_vs_v3_decision.json`

The next implementation entry is a new, isolated V2 contact-mode-transition
profile evaluated first on this same primary and first-loss window. Oracle C
and D2 may be rerun only after the implementation/evaluator contract is frozen
and a primary transition witness exists.

Validation completed with 15 new focused tests, 18 focused V2R/V2R2E tests,
and 81 complete discovery tests passing. `python -m compileall spider tests`
and `git diff --check` also pass. The artifact audit found all required
timeline, HTML, 47 screenshot, manual-review, certificate, witness, Pareto,
decision, and handoff outputs; all generated media remain under the ignored
local namespace.
