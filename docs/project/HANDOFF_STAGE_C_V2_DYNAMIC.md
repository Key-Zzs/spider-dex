# Stage C-V2 Dynamic Acceptance handoff

## Stop state

Stage C-V2 is **FAIL at D2 full forward rollout**. Do not run D3 minimal
MJWP, D4 profile search, D5 smokes, D6 final metrics, D7 HTML/screenshots, or
Stage D from this handoff.

## Environment and input lineage

- Repository: `spider-dex`, branch `develop/wuji-hand2`; no push was made.
- Runtime: Conda environment `spider-dex`, RTX 5080.
- Paths are configured only through ignored `configs/local/paths.yaml`.
- Frozen primary: `s5/cylindermedium_lift`, source frames `[1460, 1876)`.
- C-XA input: corrected Contract V2 Level 1 from
  `<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa/`.
- Corrected source reference:
  `stage_c/contact_reference_cxa.json`; 197 deep records remain
  `UNRELIABLE_SOURCE` and zero are active contacts.
- V1 remains `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`; original V2 Level 4
  remains historical CASE-A-invalid blocked evidence.

## Gates and evidence

The separate dynamic output namespace is
`<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2_dynamic/`.

| Gate | Status | Main evidence |
| --- | --- | --- |
| D0 input freeze | PASS | `input_freeze_validation.json` |
| D1 holds | PASS | `dynamic_keyframes.json`, `preflight_keyframe_hold.json`, `preflight_keyframe_hold_steps.npz` |
| D2 forward rollout | FAIL | `trajectory_corrected_v2_forward_rollout.npz`, `forward_rollout_trace.npz`, `metrics_corrected_v2_forward_rollout.json` |
| D2 localization | FAIL | `forward_rollout_failure_localization.json` |
| D3/D4/D5 | NOT_RUN | primary-first fail-closed boundary |
| HTML/screenshots/user review | NOT_RUN / NOT_AVAILABLE | no fabricated material |

D1 ran five deterministic distinct frames, was finite and warning-free, with
max qacc `29272.873`, max object translation drift `1.96e-06 m`, and max
rotation drift `3.45e-04 rad`.

The original 2-ms/zero-lead failure is preserved below
`attempts/baseline_dt002_lead0/`. A real 0.5-ms, exact-accumulated-120-Hz
retry repaired its object failure without touching any C-XA frame: right-object
position maximum is `6.45e-05 m`, rotation maximum `0.03991 rad`, and visual
mesh penetration maximum `0.002190 m`.

That repaired D2 still fails, so it does not authorize MJWP. Its required
dynamic V2 contact result is patch coverage `0.255587` (minimum `0.70`),
functional-role recall `0.0` (minimum `0.80`), and patch-distance P95
`0.066588 m` (maximum `0.020 m`). The controller also has three robot
joint-limit violations, a normalized one-frame delta of `0.296308` (maximum
`0.25`), and persistent substep hand/object contacts: maximum depth
`0.006424 m` and maximum force `337.631 N`. State, controls, warnings,
object tracking, per-side robot RMS tracking, and visual-penetration gates
remain finite/passing.

The current root cause is therefore **dynamic retention of the immutable
Level-1 patch contract**, not the earlier object-orientation discontinuity.
Read-only bounded controller probes (reference lead, servo Kp/force-limit,
and Jacobian contact-target feedback) did not attain the V2 thresholds without
introducing joint-limit, smoothness, collision, or numerical failures. Robot
qpos rewrites, C-XA/contact-target edits, raw GRAB edits, object-qpos writes,
and source-frame changes were not used. See
`forward_rollout_failure_localization.json`; the current complete retry is
also preserved under `attempts/fine_dt0005_lead20_simstep_gates/`.

External aggregate reports are under `<workspace>/reports/`:
`stage_c_v2_dynamic_validation.json`, `stage_c_v2_dynamic_pilot_summary.json`,
`stage_c_v2_dynamic_acceptance.json`, and
`stage_c_v2_dynamic_screenshot_review.json`.

## Validation and entry criteria

Focused dynamic and Stage-C preflight tests passed (`22` tests), as did
compileall and `git diff --check`.

Stage D entry still requires: primary dynamic PASS, both same-profile smokes
PASS, Codex screenshot PASS, user HTML PASS, and frozen shared profiles. None
are satisfied here. No raw data, body model, generated trajectory, HTML, or
screenshot was added to Git.
