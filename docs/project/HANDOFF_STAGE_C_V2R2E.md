# Stage C-V2R2E Handoff

## Status

`BLOCKED` at primary dynamic retention after all authorized bounded geometry,
controller, contact-IK, object-guidance, contact-dynamics, and timing branches.
Oracle C did not pass. D2, Minimal MJWP, Full MJWP, both smokes, HTML,
screenshots, user acceptance, and Stage D are `NOT_RUN` by the primary-first
gate.

## Environment and frozen scope

- Repository: `/home/deepcybo/workspace/dex/retarget/spider-dex`
- Branch: `develop/wuji-hand2`
- Base/current HEAD at CP0: `f78cfeda29b01ca0cd896ed1efe51642da28177a`
- Conda environment: `spider-dex`, Python `3.12.13`
- GPU check: NVIDIA GeForce RTX 5080; PyTorch CUDA is available. The bounded
  Oracle-C physics evidence was run with CPU MuJoCo and isolated external
  artifact roots.
- GRAB: `/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB`
- Body models: `/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models`
- External workspace: `/mnt/nas/storage/Ref2Dex_storage/spider_workspace`

Frozen primary: `s5/cylindermedium_lift`, frames `[1460,1876)`, 120 Hz.
Smokes remain `s1/mug_lift [120,240)` and `s1/mug_offhand_1 [120,180)` and
were not run because primary-first is mandatory.

## Historical lineage preserved

- V1 exact source finger contact: `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`.
- C-XA: `CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG`; corrected V2 Level 1
  static PASS.
- Corrected V2 dynamic D1 holds: PASS; historical D2: FAIL.
- V2R1: A PASS, B FAIL, C FAIL, D PASS; frozen cause
  `CONTACT_COLLISION_COUPLING`.
- V2R2 Branch-D: eight contact-dynamics candidates, all failed.
- The 197 `UNRELIABLE_SOURCE` records remain excluded from active corrected
  contacts. Raw GRAB, body models, Stage B, C-XA, and frozen frames were not
  changed.

## Recovery evidence

Primary attempt used for the final aggregate:

`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260801T004500Z-contactik/`

Key reports:

- `reports/v2r2e_alignment_audit.json`: alignment PASS; object bbox ratio
  `[1,1,1]`, patch/collision P95 `0.002015 m`, normal median `0.99999999`.
- `reports/dynamic_optimization_trace.json`: 12 actual MuJoCo profiles,
  including four bounded Jacobian contact-target profiles; no accepted
  profile. Best coverage `0.317039`, role recall `0.117647`, patch P95
  `0.081810 m`.
- `reports/contact_dynamics_recovery.json`: eight new in-memory explicit-pair
  profiles, all FAIL; no object qpos or source XML mutation.
- `reports/object_guidance_repair.json`: five phase schedules G0–G4, all FAIL.
- `reports/timing_feasibility.json`: original and 1.25x/1.5x/2.0x variants,
  all FAIL.
- `reports/preservation_audit.json`: frozen C-XA hashes match D0 and Stage-B
  read-only hashes were recorded; raw/body-model/source-frame mutation flags
  are false.
- `inherited/v2r2_branch_d/`: preserved historical eight-candidate Branch-D
  evidence, separate from the new contact-dynamics branch.

Timing replay:

`reports/timing_feasibility.json` in the final attempt namespace above.

All four variants (`1.0x`, `1.25x`, `1.5x`, `2.0x`) failed hard dynamic
gates. Original and relaxed timing are therefore distinct and no relaxed PASS
is claimed.

## Gate summary

| gate | status |
|---|---|
| V2R2E-1 alignment | PASS |
| object collision refinement | NOT REQUIRED by audit |
| hand proxy refinement | NOT REQUIRED by audit |
| region mapping repair | NOT REQUIRED by audit |
| dynamic optimizer, 12 candidates including contact-IK | BLOCKED |
| contact dynamics, 8 new candidates | BLOCKED |
| object guidance, 5 candidates | BLOCKED |
| timing ladder, 4 variants | BLOCKED |
| Oracle D | historical PASS |
| Oracle C | FAIL / no candidate |
| D2 | NOT_RUN / no Oracle C seed |
| Minimal/Full MJWP | NOT_RUN |
| Smoke 1/2 | NOT_RUN |
| HTML/screenshots | NOT_RUN |
| user acceptance | PENDING |
| Stage D | NOT_STARTED |

## Tests and implementation

Added/updated local implementation:

- `configs/project/grab_wuji_stage_c_v2r2e.yaml`
- `configs/project/wuji_hand2_contact_regions.yaml`
- `spider/tools/grab_stage_c_v2r2e.py` (alignment, dynamic, contact-IK,
  contact-dynamics, object-guidance, timing, and fail-closed continuation)
- `spider/tools/grab_stage_c_v2r2e_downstream.py` (isolated MJWP/shared-profile
  gate; refuses to run without a real D2 seed)
- `spider/tools/grab_stage_c_v2r2e_viewer.py` (primary-first HTML/Chrome gate;
  refuses to emit success pages without primary and both smoke passes)
- `spider/tools/grab_stage_c_v2r.py`
- `tests/test_stage_c_v2r2e.py`

Focused tests and the complete local suite passed after the final report
refresh:

```bash
conda run --no-capture-output -n spider-dex python -m unittest tests.test_stage_c_v2r2e tests.test_stage_c_v2r -v
conda run --no-capture-output -n spider-dex python -m unittest discover -s tests -v
conda run --no-capture-output -n spider-dex python -m compileall spider tests
git diff --check
```

The focused run passed 18 tests and the complete discovery run passed 66
tests. Compile and whitespace checks also passed. The suite emitted only the
known no-NVML/CUDA and temporary Matplotlib-cache warnings; no test failed.

## Limitations and next entry

No dynamic seed passed the CPU MuJoCo Oracle C gates. Do not fabricate MJWP or screenshot
acceptance from static/C-XA material. A future continuation must begin from
the preserved attempt namespaces, keep all hashes and failed candidates, and
may enter D2 only after a real Oracle C PASS. Stage D requires primary D2,
same-profile smokes, screenshot review, user HTML review, and frozen shared
profiles.

At final audit the NAS NFS mount reported `ro`; resume logic was made
idempotent and refuses to overwrite existing immutable artifacts. Existing
NAS reports remain authoritative; no remount or destructive workaround was
performed.

Final aggregate reports are under
`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/`, with the complete
attempt history under the attempt root above. No push was performed. No raw
data, body model, generated trajectory, HTML, or screenshot was added to Git.
The fail-closed downstream report is under the attempt root at
`reports/downstream_gate_report.json`; the aggregate HTML gate report is
`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/stage_c_v2r2e_html.json`.
