# Repository audit — S0

## Audit snapshot

| Field | Value |
| --- | --- |
| Branch | `develop/wuji-hand2` |
| Base commit | `71238456bf97a7eeb3d0471aa31974e2d404d4ae` |
| Initial dirty state | Pre-existing modification to `setup.py` (`spider` package name changed to `spider-dex`) |
| Scope | Minimal Wuji Hand2 Beta1 embodiment integration; no dataset adapter or full optimization work |

## Top-level architecture

`spider/config.py` builds runtime configuration and derives model dimensions.
`spider/process_datasets/` contains dataset-specific processors;
`spider/preprocess/` handles contact detection, scene generation, and IK;
`spider/simulators/` and `spider/optimizers/` perform the physics-informed
optimization; `spider/postprocess/` and `spider/viewers/` export and inspect
results. Assets are packaged below `spider/assets/`. VitePress documents live
below `docs/`; package metadata lives in `pyproject.toml` and `setup.py`.

## Data flow

```text
raw dataset
 -> dataset-specific processing
 -> MANO keypoints/contact trajectory under processed/<dataset>/mano/...
 -> packaged robot asset staging under processed/<dataset>/assets/robots/<robot>
 -> generate_xml.py scene generation
 -> ik.py or ik_fast.py
 -> MJWP/other simulator and optimizer
 -> postprocess/export and viewers
```

`spider.io.get_processed_data_dir()` defines the per-task output path. Scene
generation selects `<robot>/right.xml`, `left.xml`, or `bimanual.xml` from the
staged asset location. The repository does not have a central `ROBOT_TYPES`
registry: the string is an implicit asset-directory selector.

## Robot assets and runtime staging

Before this change, the shared robot staging pattern was only present in the
GMR processor, using an unrestricted `copytree`; `generate_xml.py` expected
assets to have already been copied. S2 adds `spider.assets`:

- `get_packaged_robot_asset_dir(robot_type)` resolves the package-relative asset.
- `ensure_robot_assets(dataset_dir, dataset_name, robot_type)` stages exactly
  one robot directory under the processed dataset tree.
- Existing staged assets are reused only when `ASSET_MANIFEST.json` is equal;
  a missing/different manifest fails instead of silently overwriting provenance.

`generate_xml.py` now uses this helper. Existing legacy robot directories without
manifests retain their previous staging behavior; no dataset processor is
rewritten and no dataset data is copied by the helper.

## Integration interfaces reviewed

- `generate_xml.py` reads staged robot XML and rewrites mesh paths relative to
  the generated scene's `assets` root.
- `ik.py` and `ik_fast.py` require `right/left_palm` and five named fingertip
  sites; their MANO reference order is fixed at palm plus thumb/index/middle/
  ring/pinky.
- `Config.get_noise_scale()` derives robot partitions from `nu` and object
  action dimensions. The bimanual partition uses equal halves, which is valid
  for this 26+26 embodiment.
- `mjwp_eq.py` likewise derives the bimanual half from `config.nu // 2`.
- Object qpos is appended after robot qpos by scene generation; the stand-alone
  robot models intentionally have no object degrees of freedom.
- `retarget_config.yaml` had no current consumer in this checkout. The supplied
  file is future-facing and explicitly marked as such; current IK remains MJCF-site based.

## Hard-coded assumptions

| Finding | S2 decision |
| --- | --- |
| `allegro` and `metahand` remove pinky sites in scene generation/IK | Not applicable: Wuji is a full five-finger hand and takes the default path. |
| MANO reference fingertip order is fixed | Compatible: adapter aliases use the exact existing five names. |
| Bimanual optimizer assumes equal hand DOF halves | Compatible: Wuji is exactly 26 right + 26 left. |
| Object action dimensionality is 6 single / 12 bimanual for contact guidance, or free-joint 7 / 14 otherwise | Kept; it applies to later generated scenes, not robot-only models. |
| GMR asset copy lacks provenance verification | Deferred: unrelated legacy processor; Wuji staging is centralized at scene generation. |
| Dataset processors have robot names/defaults of their own | Deferred: adding GRAB/OakInk adapters is S3+. |
| ManipTrans and DexMachina have simulator-specific hand factories/checkpoints | Deferred and out of S2 scope. |

## Incremental architecture and compatibility risk

The adapter preserves the vendor kinematic tree, limits, inertia, convex-hull
collision geometry, collision exclusions, and finger actuators. It adds only a
six-joint scalar wrist chain, six corresponding position actuators, hand-only
gravity compensation, site aliases, stable collision names, and a bimanual
include assembly. Existing robots and their defaults are untouched.

Risks left open: the palm-frame alignment has automated name/position checks but
requires human visual confirmation; dataset-specific wrist coordinate offsets and
full IK/physics behavior cannot be claimed until S3/S4 data work. The source's
known absent fingertip soft-pad collision is documented rather than replaced.
