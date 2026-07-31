# Stage C-V2R Dynamic Retention Recovery handoff

## Status

**BLOCKED at V2R2 Branch-D contact/collision repair.** V2R3–V2R5, MJWP,
frozen smokes, HTML, screenshots, user review, and Stage D are **NOT_RUN**.
No V2R acceptance artifact is claimed.

## Environment and preservation

- Repository: `spider-dex`; branch: `develop/wuji-hand2`.
- Runtime: conda `spider-dex`, NVIDIA RTX 5080.
- All generated evidence is external under
  `<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2r/`
  and `<workspace>/reports/`; no push was performed.
- Raw GRAB, body models, Stage B, C-XA Level-1 targets, frozen source timing,
  object targets, and the 197 `UNRELIABLE_SOURCE` exclusions remain unchanged.

## V2R1 valid oracle conclusion

The corrected A–D matrix is `A=PASS`, `B=FAIL`, `C=FAIL`, `D=PASS`, so the
frozen primary cause is `CONTACT_COLLISION_COUPLING`, with no independently
established secondary cause.

- Oracle A verifies the exact C-XA reference is statically feasible.
- Oracle D now neutralizes all 42 explicit hand–object MuJoCo pairs only in
  memory, with a callback for any non-explicit pair. Its trace records zero
  hand–object contact events, depth, and force, while self-collision remains
  active; controller tracking, limits, margin, and smoothness all pass.
- Oracle C fails with real hand dynamics and contact against a continuously
  prescribed object trajectory. Oracle B also fails as a diagnostic hard-clamp
  run, and is not a valid physical rollout.

The initial mask-only and callback-only Oracle-D traces are retained under
`stage_c_v2r/attempts/`; they are not eligible causal evidence because the
explicit `<pair>` declarations still permitted hand–object contacts.

## V2R2 Branch-D bounded result

The fixed eight-candidate search changed only the 42 explicit hand–object pair
`solref`, `solimp`, non-negative `margin`/`gap`, and friction in memory. It
did not alter robot control, source timing, C-XA qpos, object targets,
self-collision, or floor contact. Force limits were frozen from the preserved
D1 holds: P95 `26.424 N`, maximum `857.639 N`, impulse `7.486 Ns`.

Every candidate failed at least one immutable dynamic gate. The least-bad
contact profile (`soft_30ms_early_1mm`) cleared robot/collision/force gates but
still had only coverage `0.241620` (required `>= 0.70`), role recall `0.0`
(required `>= 0.80`), and patch-distance P95 `0.075773 m` (required
`<= 0.020 m`). No candidate is selected.

The earlier controller probes are preserved at
`stage_c_v2r/attempts/pre_oracle_d_pair_fix_controller_evidence/` but marked
`NOT_RUN` for selection: they preceded the valid Oracle-D pair fix and the
frozen result authorizes only the Branch-D contact repair.

## Evidence and resume boundary

- `reports/v2r1_causality_timeline.json` and `reports/v2r1_oracle_decision.json`
  hold the valid causal result.
- `profiles/contact_dynamics_search.json` and
  `reports/v2r2_contact_dynamics_repair.json` hold the eight-candidate block.
- `reports/stage_c_v2r_validation.json` records all downstream work as
  `NOT_RUN`.

Any future continuation requires an explicit re-scope before trying a larger
contact formulation or a different causal branch. It must preserve this
oracle evidence, all failed candidates, original timing, C-XA targets,
unreliable-record exclusions, and D1-derived thresholds.
