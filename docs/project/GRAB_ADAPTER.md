# GRAB adapter

`spider.datasets.grab.GrabAdapter` discovers local `grab/s*/` files in sorted
order, reads metadata only during discovery, resolves source-local ContactDB
meshes, and reconstructs a selected range with SMPL-X. It does not perform IK
or write to the raw source root.

The inspected release stores PCA-24 left/right hand parameters in the SMPL-X
body block. The adapter uses `use_pca=True`, `num_pca_comps=24`, and
`flat_hand_mean=True`, source subject beta files, source gender, body global
orientation, body translation, and body pose. The canonical 21-joint order is
wrist then thumb/index/middle/ring/pinky chains with explicit fingertip
vertices. Wrist translations and orientations come from the reconstructed
SMPL-X wrist transforms; the independent GRAB hand-fit translation is retained
as source information only because it is a MANO model-origin, not a stable
anatomical wrist center.

```bash
conda run -n spider-dex python -m spider.tools.audit_dataset \
  --dataset grab --paths-config configs/local/paths.yaml --max-sequences 20
conda run -n spider-dex python -m spider.tools.grab_pipeline prepare \
  --paths-config configs/local/paths.yaml --sequence-id s1__mug_lift \
  --frame-start 120 --frame-end 240
```
