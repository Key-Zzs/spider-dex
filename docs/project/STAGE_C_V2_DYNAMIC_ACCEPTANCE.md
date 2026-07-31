# Stage C-V2 dynamic acceptance

## Result

**FAIL at D2 — full corrected forward physics rollout.** D0 input freeze and
D1 keyframe holds passed. D3 minimal MJWP, D4 profile search, both smokes,
HTML, screenshots, and Stage D are not run.

This is intentionally fail closed. A finite MuJoCo rollout is not sufficient
for acceptance: all tracking, collision, V2-contact, penetration, and
smoothness gates must pass before MJWP may start.

## Immutable history

- V1 `EXACT_SOURCE_FINGER_CONTACT` remains
  `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`.
- The original V2 Level-4 result remains a historical CASE-A-invalid blocked
  artifact; it was neither deleted nor used as the dynamic input.
- C-XA is `CASE_A PASS`: the corrected Contract-V2 Level-1 static result is
  the sole input to this run. It retains 197 `UNRELIABLE_SOURCE` records,
  none of which are active targets.

## Dynamic evidence

All generated evidence is external under
`<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_v2_dynamic/`.

| Gate | Result | Evidence |
| --- | --- | --- |
| D0 corrected input freeze | PASS | `input_freeze_validation.json` |
| D1 real-contact keyframe holds | PASS | `preflight_keyframe_hold.json`, `preflight_keyframe_hold_steps.npz` |
| D2 full forward rollout | FAIL | `metrics_corrected_v2_forward_rollout.json`, `forward_rollout_trace.npz` |
| D3 minimal MJWP | NOT_RUN | D2 primary-first stop boundary |
| D4 primary MJWP | NOT_RUN | D3 was not allowed |
| D5 shared-profile smokes | NOT_RUN | primary was not accepted |
| D7 HTML/screenshots | NOT_RUN | no false visual acceptance evidence |

The D1 five distinct deterministic source frames had finite state, zero
warnings, maximum qacc `29272.873`, maximum object translation drift
`1.96e-06 m`, and maximum object rotation drift `3.45e-04 rad`.

The preserved original 2-ms/zero-lead D2 failure is retained at
`attempts/baseline_dt002_lead0/`. A second real-MuJoCo attempt uses a 0.5-ms
integration step with exact accumulated 120-Hz scheduling. It preserves every
frozen C-XA source frame, drives objects only through the mocap weld, and
repairs the former object-tracking failure: right-object rotation maximum is
`0.03991 rad`, visual maximum penetration is `0.002190 m`, and all states are
finite with zero warnings.

The retry is nevertheless **FAIL** at D2. Its actual physical V2 result is
patch coverage `0.255587` (required `>= 0.70`), functional-role recall `0.0`
(required `>= 0.80`), and patch-distance P95 `0.066588 m` (required
`<= 0.020 m`). At real integration-step resolution it also has three
robot-joint-limit violations, a `0.296308` normalized single-frame joint
delta, persistent hand/object collision depth up to `0.006424 m`, and
exponential contact-force-growth evidence. Bounded read-only probes of phase
lead, servo Kp/force limits, and Jacobian contact feedback did not obtain a
profile satisfying every D2 gate. The complete attempt is preserved at
`attempts/fine_dt0005_lead20_simstep_gates/`; its diagnosis is
`forward_rollout_failure_localization.json`.

## Stop boundary

No MJWP candidate, profile selection, smoke, HTML, screenshot, user review,
or Stage D claim is valid from this result. A future scoped task would need a
new, explicitly versioned dynamic contact/controller contract; it cannot
alter or reinterpret this C-XA/Stage-C-V2 evidence.
