# Validation and manual acceptance

## Stage A external-data validation

Run the targeted infrastructure suite and configuration audit before a dataset
adapter is enabled:

```bash
conda run -n spider-dex python -m unittest tests.test_dataset_infrastructure -v
conda run -n spider-dex python -m spider.tools.audit_dataset \
  --dataset grab --paths-config configs/local/paths.yaml --dry-run
conda run -n spider-dex python -m compileall spider tests
git diff --check
```

The Stage A audit verifies paths and workspace even before GRAB registration;
`ADAPTER_NOT_REGISTERED` is an explicit pre-adapter state, never a fabricated
dataset scan result. Once an adapter is registered, the same command performs
its source audit and reports a structured JSON result.

## Stage B bounded GRAB pilots

The primary `s1/mug_lift` and two deterministic smoke pilots have structural
output checks for source/canonical data, object mesh, scene, Wuji model, named
trajectory, limits, mapping, and videos. Their fixed fingertip smoke threshold
is exceeded, so the accurate quality state is
`AUTO_PIPELINE_PASS_MANUAL_REVIEW_REQUIRED`; see
[manual acceptance](MANUAL_ACCEPTANCE_GRAB_WUJI.md). This is not a physics or
contact-optimization result.

## Automated validation

Run from the repository root:

```bash
conda run -n spider-dex python tools/validate_wuji_hand2.py
conda run -n spider-dex python -m unittest discover -s tests -p 'test_wuji_hand2.py' -v
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side bimanual --mode sweep --no-viewer --duration 0.1
```

The validator checks required files, XML/URDF parsing, relative paths, manifest
SHA-256s, license/provenance, name uniqueness, required palm/fingertip/track/
trace sites, actuator-to-joint mapping, control/qpos order, MuJoCo model loads,
2.5-second neutral holds, and temporary runtime staging.

Latest Wuji target-model automated result: **PASS**. Overall S2 automatic
acceptance is **NOT YET PASS** because the required default upstream-baseline
sanity gate currently fails on its unchanged local sample, as recorded below.

| Model | nq | nv | nu | joints | actuators | bodies | sites | geoms | collision geoms | Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| right | 26 | 26 | 26 | 26 | 26 | 28 | 21 | 42 | 21 | 2.5 s, no NaN/Inf, max qpos delta 0 |
| left | 26 | 26 | 26 | 26 | 26 | 28 | 21 | 42 | 21 | 2.5 s, no NaN/Inf, max qpos delta 0 |
| bimanual | 52 | 52 | 52 | 52 | 52 | 55 | 42 | 84 | 42 | 2.5 s, no NaN/Inf, max qpos delta 0 |

The baseline command is intentionally bounded and uses only already-local inputs.
The existing OakInk/XHand sample was run with one sample, one optimizer iteration,
one requested planning step, no viewer/video, and the baseline sanity check
explicitly disabled. It exited successfully and wrote its ignored local output;
the final object tracking error was `pos=0.0242`, `quat=0.6530`.

The same sample with the default three-second sanity gate did **not** pass: its
right object drift was 1.311 cm against a 1.000 cm threshold. This is a current
local baseline-data gate failure, not a Wuji regression claim—the existing sample
and robot are unchanged by S2. The bounded completion establishes that the
unmodified XHand path remains executable once that pre-existing gate is disabled.

The former dataset-dependent Wuji scene/IK smoke blocker is resolved for the
bounded external GRAB pilots described above. It does not change the historical
upstream baseline gate statement.

## Manual visual acceptance

**PENDING USER ACCEPTANCE**

Run these commands in a graphical MuJoCo session:

```bash
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side right --mode neutral --show-collision --show-sites
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side right --mode sweep --show-collision --show-sites
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side left --mode sweep --show-collision --show-sites
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side bimanual --mode sweep --show-collision --show-sites
```

`sweep` keeps wrist controls neutral and pulses all four joints of one finger
at a time.  In bimanual mode it alternates the active hand once per second, so
each hand is visibly exercised while the other remains neutral.  The default
ten-second duration completes one five-finger cycle for each hand; pass, for
example, `--duration 20` for a longer inspection.

Right hand checklist:

- [ ] It is a right hand, with correct palm orientation and finger ordering.
- [ ] Each joint's positive sweep is consistent with flexion/abduction semantics.
- [ ] Thumb opposition, fingertip sites, palm frame, neutral pose, and visual vs
      collision geometry look correct.

Left hand checklist:

- [ ] It is a correct mirror rather than a duplicated right hand.
- [ ] Mirrored joint axes and thumb side are correct.
- [ ] Palm frame, sites, and collision geometry look correct.

Bimanual checklist:

- [ ] Both hands render without duplicate-name warnings.
- [ ] Both control segments are exercised: the active hand moves while the
      other hand remains still, then the roles alternate.
- [ ] Camera/scene behavior is sensible; no jump, explosion, or unexpected coupling occurs.

Do not mark the complete S2 acceptance as finished until this checklist has been
reviewed by a human. The accurate current statement is: **Wuji target-model
automated validation: PASS; overall S2 automatic acceptance: BLOCKED BY
UPSTREAM-BASELINE SANITY GATE; S2 manual visual acceptance: PENDING.**
