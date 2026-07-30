# GRAB local data audit

This audit is from bounded inspection of the configured local source, not an
online format assumption.

- Source root: configured locally; the inspected machine uses the GRAB release
  root containing `grab/s1` through `grab/s10` and `tools/`.
- Sequence format: 1,335 `grab/<subject>/*.npz` files with object-array blocks
  `body`, `lhand`, `rhand`, `object`, `table`, and `contact`.
- Top-level metadata includes `gender`, `sbj_id`, `framerate`, `obj_name`,
  `n_frames`, `n_comps`, and `motion_intent`.
- Object assets are source-local ContactDB meshes; contact meshes are under
  `tools/object_meshes/contact_meshes`.
- The configured body-model root contains MANO left/right models and SMPL-X
  female/male/neutral models. Stage B must use the source gender and beta data.

The Stage B adapter records the exact params-key mapping and validates units,
world transforms, handedness, and frame counts per selected sequence before any
Wuji IK. It must not infer missing hands or silently mirror one side.
