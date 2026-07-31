# Stage C-XA V2 failure-localization handoff

## Result and stop boundary

C-XA selected `CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG`.  The corrected,
isolated rerun of frozen primary `s5/cylindermedium_lift`, frames `[1460, 1876)`,
passes Contract V2 static evaluation at the first permitted relaxation level
and passes its static MuJoCo preflight.  This closes the required CASE A path.

Do **not** run C-XB, C-XC, smoke pilots, MJWP, hardware, acceptance HTML, or
Stage D from this handoff.  The CASE A order stops after the primary static
pass and preflight; later levels are deliberately `NOT_RUN`.

## Preserved baseline

- V1 `EXACT_SOURCE_FINGER_CONTACT` remains immutable and
  `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`.
- The original V2 Level 4 artifacts remain unchanged at
  `<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2/`.
- The original blocked values remain evidence, not a superseded pass:
  functional-role recall `0.750000`, patch-distance P95 `0.032007 m`,
  patch coverage `0.906900`, and collision maximum `0.002823 m`.
- Raw GRAB, frozen primary, Stage B, contract thresholds, and source records
  were not overwritten.  The source correction is a new tagged reference;
  it preserves the original inclusive flag and records rejected deep samples.

## C-XA evidence and correction

The C-XA reports are in `<workspace>/reports/`:

- `cxa_role_audit.json`
- `cxa_metric_audit.json`
- `cxa_failure_localization.json`
- `cxa_source_contact_reliability.json`
- `cxa_case_a_bug_evidence.json`
- `cxa_v2_original_patch_recompute.json`
- `cxa_case_a_rerun.json`

The audit localized three CASE A defects:

1. Deeply penetrating source samples were active contacts because a
   watertight inside sign was treated as sufficient contact evidence.  The
   correction requires closest-surface distance within the immutable `15 mm`
   tolerance and retains `UNRELIABLE_SOURCE` records for audit.
2. Functional-role recall counted `TRANSIENT` roles in its denominator.  It
   now uses unweighted, unique functional role intervals and reports
   nonfunctional roles separately.
3. V2 patch-distance P95 measured distance to a compiled point anchor, rather
   than nearest distance to the selected mesh-adjacent object patch.  It now
   evaluates the assigned fingertip against that patch in the immutable
   Stage-B object-pose frame.

The source reliability report records `197` unreliable deep samples; they are
preserved as source evidence but are not promoted into corrected active
contacts.

## Corrected isolated primary result

Corrected artifacts are isolated under
`<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa/`.
The tagged source reference is
`<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c/contact_reference_cxa.json`.

| Gate | Corrected Level 1 value | Result |
| --- | ---: | --- |
| task-equivalent patch coverage | `0.938547` | pass (`>= 0.70`) |
| functional-role recall | `0.882353` (15/17) | pass (`>= 0.80`) |
| patch-distance P95 | `0.017376 m` | pass (`<= 0.020 m`) |
| normal cosine median | `1.000000` | pass (`>= 0.50`) |
| collision maximum after depenetration | `0.002300 m` | pass (`<= 0.003 m`) |
| static MuJoCo preflight | finite 414 frames; 0 warnings | pass |

`cxa_case_a_rerun.json` records the unchanged contract hash, V1 preservation,
static result, preflight result, and the explicit no-smokes/no-Stage-D stop
state.

## Next entry

This goal is complete at the static-and-preflight boundary.  Any request to
continue needs a new explicitly scoped acceptance plan; it must retain the
frozen V1 evidence and contract thresholds, and may not reinterpret this
static result as an authorized physical or hardware pass.
