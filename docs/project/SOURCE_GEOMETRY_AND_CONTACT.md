# Source geometry and contact references

`spider.tools.grab_stage_c diagnose-source` reconstructs GRAB SMPL-X hand
vertices from immutable raw parameters and writes separate Stage C diagnostics.
It does not overwrite canonical data. The object distance query is exact
closest-surface BVH querying through Open3D; watertight, winding-consistent
objects use signed distance (negative means inside). Non-watertight objects
remain unsigned/low-confidence rather than receiving a fabricated inside label.

`build-contact-reference` projects source fingertips to the real object surface
and records source point, projected point, normal, signed/unsigned distance,
confidence, contact flag, and contact interval. The reference is source-only:
projected anchors are not a claim that raw GRAB is collision-free and cannot be
used to alter the raw data.
