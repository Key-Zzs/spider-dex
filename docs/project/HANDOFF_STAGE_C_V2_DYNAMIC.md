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
max qacc `7252.795`, max object translation drift `1.68e-05 m`, and max
rotation drift `8.71e-04 rad`.

D2 was finite and warning-free but failed three gates:

1. right-object rotation maximum `0.768767 rad` exceeds `0.50 rad`;
2. real MuJoCo penetration persisted in four runs, maximum `0.005793 m`;
3. robot tracking exceeded the wrist/fingertip thresholds (right palm RMSE
   `0.063978 m`, right middle-tip RMSE `0.090402 m`).

The primary cause is in the immutable corrected object reference: consecutive
right-object orientation changes of `1.350221 rad` at source frame 1854 and
`1.413447 rad` at source frame 1858. Do not smooth/reselect/delete/replace
those frames, move object qpos, modify raw GRAB, modify body models, overwrite
Stage B, or overwrite C-XA. Such a change needs a new versioned contract.

External aggregate reports are under `<workspace>/reports/`:
`stage_c_v2_dynamic_validation.json`, `stage_c_v2_dynamic_pilot_summary.json`,
`stage_c_v2_dynamic_acceptance.json`, and
`stage_c_v2_dynamic_screenshot_review.json`.

## Validation and entry criteria

Focused dynamic, existing Stage-C preflight, and collision-audit tests passed
(`25` tests), as did compileall and `git diff --check`.

Stage D entry still requires: primary dynamic PASS, both same-profile smokes
PASS, Codex screenshot PASS, user HTML PASS, and frozen shared profiles. None
are satisfied here. No raw data, body model, generated trajectory, HTML, or
screenshot was added to Git.
