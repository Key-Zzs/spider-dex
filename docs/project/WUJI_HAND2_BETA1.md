# Wuji Hand2 Beta1 adapter

## Source model

The audited source is the `body` model at the provenance path in
[ASSET_PROVENANCE.md](ASSET_PROVENANCE.md). It has `{r,l}_wrist` roots, five
fingertip query sites, twenty actuated revolute joints per side, relative STL
mesh paths, vendor position actuators, convex-hull collision meshes, and ten
assembly-overlap excludes per hand. Mesh units and vendor scale are retained.

The source documentation reports group 1 visual geometry, group 2 collision
geometry, and group 3 fingertip sites. Its current limitations are preserved:
fingertip soft-pad STL files exist but are not collision geometry, and drive gains
await hardware system identification.

## SPIDER adaptation

Each adapter preserves vendor joints, axes, limits, inertial parameters, visual
meshes, collision meshes, actuator force limits, and contact exclusions. It adds:

- six scalar controls before the vendor wrist: `tx`, `ty`, `tz`, `roll`, `pitch`,
  `yaw`, with translation axes X/Y/Z and rotation axes Z/X/Y;
- six position actuators before the 20 vendor finger actuators;
- gravity compensation on hand-only adapter bodies, while retaining normal global
  gravity for later scene object bodies;
- `right_palm`/`left_palm` plus 15 aliases per side for standard, `track_hand`,
  and `trace_hand` fingertip sites;
- deterministic `collision_hand_<side>_<finger>_<part>` names on existing group-2
  collision geoms, without adding duplicate collision bodies;
- a symmetric bimanual neutral placement (`right y=-0.16 m`, `left y=+0.16 m`)
  so robot-only stability testing does not begin with two interpenetrating hands.

The palm site is at the vendor wrist root. It uses the vendor root frame and is
visually checkable; no dataset-specific MANO wrist alignment is claimed yet.

## Adapter layout

```text
spider/assets/robots/wuji_hand2_beta1/
├── {right,left,bimanual}.xml       SPIDER MJCF adapters
├── urdf/{right,left}_6dof.urdf     future-facing URDF wrappers
├── retarget_config.yaml            no current consumer; documented future schema
├── vendor/mjcf, vendor/urdf, vendor/meshes
├── LICENSE_WUJI
└── ASSET_MANIFEST.json
```

`bimanual.xml` includes the two adapters, with all asset/body/joint/geom/site and
actuator names unique by side. The bimanual control/qpos partition is
`[right 26][left 26]`; later scene object qpos follows those robot coordinates.

## Joint and actuator mapping

The vendor name/body and joint order below were parsed from the source MJCF and
rechecked through `mujoco.MjModel`. Right uses `+X` flexion and `+Y` abduction;
left mirrors flexion axes as `-X` while retaining `+Y` abduction. Left names are
the exact `l_` counterpart and have the same per-side actuator/qpos index.

| Side | Finger | Wuji joint / body | Axis (R / L) | Limit rad | SPIDER actuator / index | qpos index |
| --- | --- | --- | --- | --- | --- |
| R / L | thumb | `{r,l}_thumb_cmc_flex` / `{r,l}_thumb_proximal` | +X / -X | -1.187, 1.291 | `{r,l}_THJ0` / 6 | 6 |
| R / L | thumb | `{r,l}_thumb_cmc_abd` / `{r,l}_thumb_proximal_abd` | +Y / +Y | -1.484, 0.698 | `{r,l}_THJ1` / 7 | 7 |
| R / L | thumb | `{r,l}_thumb_mcp` / `{r,l}_thumb_middle` | +X / -X | -1.047, 1.570 | `{r,l}_THJ2` / 8 | 8 |
| R / L | thumb | `{r,l}_thumb_ip` / `{r,l}_thumb_distal` | +X / -X | -1.047, 1.570 | `{r,l}_THJ3` / 9 | 9 |
| R / L | index | `{r,l}_index_finger_mcp_flex` / `{r,l}_index_finger_proximal` | +X / -X | -1.047, 1.570 | `{r,l}_FFJ0` / 10 | 10 |
| R / L | index | `{r,l}_index_finger_mcp_abd` / `{r,l}_index_finger_proximal_abd` | +Y / +Y | -0.698, 0.698 | `{r,l}_FFJ1` / 11 | 11 |
| R / L | index | `{r,l}_index_finger_pip` / `{r,l}_index_finger_middle` | +X / -X | -1.047, 2.094 | `{r,l}_FFJ2` / 12 | 12 |
| R / L | index | `{r,l}_index_finger_dip` / `{r,l}_index_finger_distal` | +X / -X | -1.047, 1.570 | `{r,l}_FFJ3` / 13 | 13 |
| R / L | middle | `{r,l}_middle_finger_mcp_flex` / `{r,l}_middle_finger_proximal` | +X / -X | -1.047, 1.570 | `{r,l}_MFJ0` / 14 | 14 |
| R / L | middle | `{r,l}_middle_finger_mcp_abd` / `{r,l}_middle_finger_proximal_abd` | +Y / +Y | -0.698, 0.698 | `{r,l}_MFJ1` / 15 | 15 |
| R / L | middle | `{r,l}_middle_finger_pip` / `{r,l}_middle_finger_middle` | +X / -X | -1.047, 2.094 | `{r,l}_MFJ2` / 16 | 16 |
| R / L | middle | `{r,l}_middle_finger_dip` / `{r,l}_middle_finger_distal` | +X / -X | -1.047, 1.570 | `{r,l}_MFJ3` / 17 | 17 |
| R / L | ring | `{r,l}_ring_finger_mcp_flex` / `{r,l}_ring_finger_proximal` | +X / -X | -1.047, 1.570 | `{r,l}_RFJ0` / 18 | 18 |
| R / L | ring | `{r,l}_ring_finger_mcp_abd` / `{r,l}_ring_finger_proximal_abd` | +Y / +Y | -0.698, 0.698 | `{r,l}_RFJ1` / 19 | 19 |
| R / L | ring | `{r,l}_ring_finger_pip` / `{r,l}_ring_finger_middle` | +X / -X | -1.047, 2.094 | `{r,l}_RFJ2` / 20 | 20 |
| R / L | ring | `{r,l}_ring_finger_dip` / `{r,l}_ring_finger_distal` | +X / -X | -1.047, 1.570 | `{r,l}_RFJ3` / 21 | 21 |
| R / L | pinky | `{r,l}_pinky_mcp_flex` / `{r,l}_pinky_proximal` | +X / -X | -1.047, 1.570 | `{r,l}_LFJ0` / 22 | 22 |
| R / L | pinky | `{r,l}_pinky_mcp_abd` / `{r,l}_pinky_proximal_abd` | +Y / +Y | -0.698, 0.698 | `{r,l}_LFJ1` / 23 | 23 |
| R / L | pinky | `{r,l}_pinky_pip` / `{r,l}_pinky_middle` | +X / -X | -1.047, 2.094 | `{r,l}_LFJ2` / 24 | 24 |
| R / L | pinky | `{r,l}_pinky_dip` / `{r,l}_pinky_distal` | +X / -X | -1.047, 1.570 | `{r,l}_LFJ3` / 25 | 25 |

The preceding six per-side rows are wrist controls: indices 0–2 are X/Y/Z
translation in [-2, 2] m; 3–5 are Z/X/Y rotation in [-6.2, 6.2] rad. This gives
each single model 26 joints/actuators and bimanual 52.
