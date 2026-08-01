# Stage C C-M1/C-M2 handoff

## Final status

- C-M1 explicit contact-mode state machine: `PASS`.
- C-M2 bounded first-loss-window witness: `EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS`.
- Dynamic witness: `NOT_FOUND` after 12 fixed real-MuJoCo profiles.
- Full primary: `NOT_RUN`.
- Oracle C full sequence: `NOT_RUN`.
- D2: `NOT_RUN`.
- MJWP: `NOT_RUN`.
- smokes: `NOT_RUN`.
- Stage D: `NOT_STARTED`.
- User visual review: `PENDING`; Codex diagnostic visual review: `REVIEWED_DIAGNOSTIC`.

## Environment and frozen lineage

- Repository: `/home/deepcybo/workspace/dex/retarget/spider-dex`.
- Branch: `develop/wuji-hand2`.
- Base commit at task start: `f14cbd7efbbd29068cfaa900f977b46c42759aa0`.
- Conda: `spider-dex`; GPU: NVIDIA RTX 5080.
- External workspace: `/mnt/nas/storage/Ref2Dex_storage/spider_workspace`.
- GRAB: `/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB`.
- Body models: `/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models`.
- Frozen primary: `s5/cylindermedium_lift`, source range `[1460,1876)`, 120 Hz.
- First historical failure: source frame `1465`, local frame `4`, simulation
  step `50`, substep `0`, left-index `SUPPORT`, patch
  `patch:s5__cylindermedium_lift:0`.
- Historical static witness: `FEASIBLE`; historical transition certificate:
  `EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS`.

## C-M1 implementation

`spider/contact/contact_mode.py` defines `PRE_CONTACT`, `ACQUIRE`, `RETAIN`,
`RELEASE`, `REGRASP`, `COMPLETE`, and `FAILED`, with dataclass observations,
bounded timeouts, confirmation hysteresis, immutable mapping checks, bounded
regrasp attempts, explicit failure codes, and serialized transition traces.

`spider/tools/grab_stage_c_contact_mode.py` is the real-MuJoCo short-window
runner. `configs/project/grab_wuji_stage_c_contact_mode.yaml` contains only
portable contract/profile values; machine paths remain in
`configs/local/paths.yaml`.

## C-M2 matrix and result

The only legal window was source frames `1461..1480` / local `0..19`. E0
reused the immutable historical artifact. E1 had three no-regrasp profiles;
E2 had three bounded regrasp profiles; E3 had six representative profiles.
All 12 dynamic profiles failed contact continuity. The best failed profile
was `e3_high_kp_anchor`:

- correct assigned physical contact: `2/20` frames;
- role recall and patch coverage: `0.10`;
- patch P95: `0.0524610791 m`;
- normal median: `1.0`;
- terminal assigned contact: `false`;
- mode trace: `ACQUIRE -> REGRASP -> FAILED`;
- failure code: `REGRASP_TIMEOUT`;
- finite/warnings, joint limits, penetration, force, smoothness, robot
  tracking, and object tracking: pass.

The result is an empirical bounded failure, not a mathematical proof. V3 is
not justified because the same-frame static role witness remains feasible.

## Evidence paths

Run root:
`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode/`

- HTML: `stage_c_contact_mode_transition.html`.
- Visual index: `contact_mode_visual_index.html`.
- Selected profile: `selected_contact_mode_profile.json`.
- Selected timeline: `contact_mode_timeline.json`.
- Selected trace: `contact_mode_trace.npz`.
- Matrix summary: `contact_mode_transition_summary.json`.
- Historical E0: `historical_baseline_summary.json`.
- Screenshots: `screenshots/` (24 PNGs, 12 distinct source frames).
- Screenshot manifest: `contact_mode_screenshot_manifest.json`.
- Codex review: `contact_mode_manual_visual_review.json` and
  `CONTACT_MODE_SCREENSHOT_REVIEW.md`.

## Validation

```bash
conda run --no-capture-output -n spider-dex \
  python -m unittest discover -s tests -v
```

Result: `91 tests passed`.

```bash
conda run --no-capture-output -n spider-dex \
  python -m unittest tests.test_stage_c_contact_mode tests.test_stage_c_v2r \
  tests.test_stage_c_v2r2e tests.test_stage_c_failure_diagnostic \
  tests.test_stage_c_collision_audit tests.test_cxa_audit -v
```

Result: `53 tests passed`.

```bash
conda run --no-capture-output -n spider-dex python -m compileall spider tests
git diff --check
```

Both passed. Chrome generated 24/24 screenshots; Codex inspected key frames
1461, 1465, and 1480. User visual review remains pending.

## Git and preservation

No push was performed. No local commit was created; implementation changes
remain uncommitted for review. `.local_artifacts/` remains ignored. Raw GRAB,
body models, historical attempts, frozen timing, source/object targets, role,
patch, and V2 thresholds were not changed. `configs/local/paths.yaml` remains
local/ignored. No MJWP, smoke, full primary, Oracle C full sequence, D2, or
Stage D command was run.

## Next decision

Upgrade the V2 contact planner/optimizer or use longer-horizon feasibility
preserving planning while retaining the same contract. Do not enter V3. If a
future bounded run finds a witness, first extend only to `1461..1520`; do not
automatically run MJWP or full primary from this handoff.
