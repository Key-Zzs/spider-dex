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
warnings, maximum qacc `7252.795`, maximum object translation drift
`1.68e-05 m`, and maximum object rotation drift `8.71e-04 rad`.

D2 remained finite and warning-free, but failed object tracking (right-object
rotation maximum `0.768767 rad` against `0.50 rad`), repeated real MuJoCo
hand-object penetration (maximum `0.005793 m`), and robot tracking. The D2
localization records two immutable corrected object-reference single-frame
orientation discontinuities: `1.350221 rad` at source frame 1854 and
`1.413447 rad` at source frame 1858. Smoothing, replacing, deleting, or
moving this frozen trajectory would violate the Stage C-V2 contract.

## Stop boundary

No MJWP candidate, profile selection, smoke, HTML, screenshot, user review,
or Stage D claim is valid from this result. A future scoped task must decide
whether a new versioned dynamic object-continuity contract is permitted;
it cannot alter or reinterpret this C-XA/Stage-C-V2 evidence.
