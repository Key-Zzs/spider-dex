# Stage A/B handoff

## State

- Repository: `/home/deepcybo/workspace/dex/retarget/spider-dex`
- Branch/base commit: `develop/wuji-hand2` / `1985442cea46921c4bd7746a4a5f07164d10ff1f`
- No commit or push was made by this stage.
- Local-only, untracked paths: GRAB source
  `/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB`; body models
  `/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models`; workspace
  `/mnt/nas/storage/Ref2Dex_storage/spider_workspace`; config
  `configs/local/paths.yaml`.

## Completed architecture

`spider.datasets.paths` validates separate source/model/workspace roots;
`registry` registers adapters explicitly; `schema` serializes canonical v1
pickle-free HOI data; `manifest` provides deterministic records; `grab` is the
SMPL-X adapter; `grab_pipeline` bridges canonical GRAB to the legacy SPIDER
layout and existing Wuji IK. `report_stage_ab` writes external validation
reports. `generate_xml.py` now uses the material it creates, fixing portable
Wuji scene compilation.

## GRAB and pilots

GRAB has sorted `grab/s1..s10/*.npz` PCA-24 body/hand data, 120 Hz on the
selected subject, subject gender/betas, ContactDB object meshes, and source
world-space body/object transforms. The concrete pilot manifest is
`<workspace>/manifests/grab_pilot.json`.

Primary: `s1/mug_lift`, frames `[120,240)`, source replay at
`<workspace>/processed/grab/canonical/s1__mug_lift/visualization/source_replay.mp4`,
IK replay/metrics at
`<workspace>/processed/grab/wuji_hand2_beta1/bimanual/s1__mug_lift/0/`.

## Commands

```bash
conda run -n spider-dex python -m spider.tools.audit_dataset --dataset grab --paths-config configs/local/paths.yaml
conda run -n spider-dex python -m spider.tools.grab_pipeline prepare --paths-config configs/local/paths.yaml --sequence-id s1__mug_lift --frame-start 120 --frame-end 240
conda run -n spider-dex python -m spider.tools.grab_pipeline render-source --canonical-dir /mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/canonical/s1__mug_lift
conda run -n spider-dex python -m spider.tools.grab_pipeline run-wuji-ik --paths-config configs/local/paths.yaml --sequence-id s1__mug_lift
conda run -n spider-dex python -m spider.tools.report_stage_ab --paths-config configs/local/paths.yaml
```

## Limitations and next gate

No physics optimization, contact/penetration tuning, batch conversion, OakInk,
or real robot work occurred. The generated pilots have valid trajectories and
videos but exceed fixed fingertip tracking smoke thresholds, so their quality
status is `AUTO_PIPELINE_PASS_MANUAL_REVIEW_REQUIRED`. Stage B manual
acceptance remains pending. Do not enter Stage C until the user reviews
`MANUAL_ACCEPTANCE_GRAB_WUJI.md` and explicitly passes it.
