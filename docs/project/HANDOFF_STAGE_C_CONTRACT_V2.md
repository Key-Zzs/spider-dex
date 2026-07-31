# Stage C Contract V2 handoff

## Final C-X state

- V1 `EXACT_SOURCE_FINGER_CONTACT` is immutable:
  `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`.
- V2 `TASK_EQUIVALENT_CONTACT` is
  `BLOCKED_BY_INFEASIBLE_TASK_EQUIVALENT_CONTACT`.
- Minimum successful relaxation level: `NONE`.
- Stage D is **not started**.

The frozen primary was `s5/cylindermedium_lift`, frames `[1460, 1876)`. The
smokes `s1/mug_lift [120, 240)` and `s1/mug_offhand_1 [120, 180)` were not run:
the primary did not pass the static V2 contract, so running them would violate
the primary-first gate.

## Immutable V1 evidence

The external manifest is `<workspace>/reports/stage_c_contract_v1_manifest.json`.
It inventories the C-R4 infeasibility report, collision conflict, multi-start
Pareto, MJWP sanity, and Stage C validation by path, SHA-256, and size. The
immutable V1 profile hash is
`c1122082604e5e56aaa29f12ff1c74ac935a7714b94bc826b846352d48589e47`. Its
bounded best candidate retained exact recall `0.425792` and maximum collision
penetration `0.003720 m`; it must never be promoted or overwritten.

## V2 contract and result

V2 is defined in `configs/project/grab_wuji_stage_c_contract_v2.yaml`. Source
roles and mesh-adjacent, normal-consistent object-local patches are produced
before robot reachability. Assignments preserve side, role, interval, patch,
capacity, and temporal stability. Thumb opposition is thumb-only; palm and
cross-hand shortcuts are excluded. The ordered ladder is Level 1 same-finger,
Level 2 neighbour-finger, Level 3 same-hand set, and Level 4 bounded
functional-surface patch.

All four levels were executed on the primary. Level 4, the widest permitted
bounded relaxation, passed depenetration, tracking, normal alignment, and
coverage, but failed role recall and patch-distance P95:

| Metric | Level 4 value | Gate |
| --- | ---: | ---: |
| task-equivalent patch coverage | 0.906900 | >= 0.70 |
| functional-role recall | 0.750000 | >= 0.80 |
| patch-distance P95 | 0.032007 m | <= 0.020 m |
| normal cosine median | 0.993448 | >= 0.50 |
| collision max after depenetration | 0.002823 m | <= 0.003 m |

No physical rollout, MJWP, smoke, HTML, or screenshots were run after this
static failure. This is intentional fail-closed behavior, not missing output.

## Profiles, artifacts, and checks

- V2 contract hash is recorded in each `selected_contact_assignment_level_*.json`
  and in `<workspace>/reports/stage_c_contract_comparison.json`.
- V2 artifacts remain isolated under
  `<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2/`.
- Comparison, validation, acceptance, and screenshot-review reports are in
  `<workspace>/reports/`.
- The task-specific contract suite and Stage C preflight suite pass; run the
  complete test suite and `git diff --check` before any future change.

## Limits and next entry criterion

Raw GRAB, body models, frozen pilots, Stage B, V1 artifacts, and existing
Stage C recovery reports are unchanged. V2 is task-equivalent contact, never
exact human-finger reproduction; no real hardware was tested. Do not start
Stage D unless a new, pre-frozen contract and evidence basis is approved. Do
not silently broaden the V2 patch radius, wrist/joint bounds, candidate count,
or acceptance thresholds to turn this blocked result into a pass.
