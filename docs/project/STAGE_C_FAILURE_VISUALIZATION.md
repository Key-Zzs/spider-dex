# Stage C failure visualization

The failure viewer is independent of the success/acceptance viewer. Every page
is headed `FAILURE DIAGNOSTIC — NOT AN ACCEPTANCE ARTIFACT` and may be produced
from a failed candidate without relaxing the primary-first acceptance gate.

Run it from the repository root with the frozen attempt and a new ignored local
output directory:

```bash
conda run --no-capture-output -n spider-dex python -m spider.tools.grab_stage_c_failure_diagnostic \
  --attempt-root /mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260801T004500Z-contactik \
  --aggregate-report /mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/stage_c_v2r2e_validation.json \
  --output-dir .local_artifacts/stage_c_v2r2e_failure_diagnostic/<run_id> \
  --render-html --render-screenshots --run-feasibility
```

The viewer renders every Wuji link as a connected surface for both hands. It
includes Stage B, C-XA, failed-reference, failed-actual, and actual collision
states, plus source 21-joint hand skeletons. It uses quadric simplification on
each connected link only; it never uniformly samples faces or substitutes
fingertip dots for the hand. The semantic patch is the real connected,
normal-consistent object submesh. Object visual and collision meshes are
separate layers.

Chrome screenshots support global, close, and top views plus patch/contact,
visual/collision, reference/actual, source/Stage-B/C-XA, and
force/penetration layer groups. Frames 1464 and 1465 use the same close camera;
later close views follow the active patch so moving events remain in frame.
The validated run contains 47 reviewed PNGs over 14 distinct source frames.

The validated local run is
`.local_artifacts/stage_c_v2r2e_failure_diagnostic/20260731T180415Z/`.
Its first exact semantic crossing is source frame 1465, trace record 50,
substep 0: left index role `s5__cylindermedium_lift:0` leaves patch
`patch:s5__cylindermedium_lift:0`. No hand–object MuJoCo geom pair exists at
that state; the contact is physically absent rather than evaluator-only lost.
The first physical pair occurs earlier at source frame 1462, substep 0:
`collision_hand_left_index_8 ↔ right_object_0`.
