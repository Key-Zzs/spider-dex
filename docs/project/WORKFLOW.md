# Workflow contract

```text
External HOI datasets
        | configurable external path
        v
Dataset-specific adapter
        v
Unified MANO/object trajectory
        v
SPIDER keypoint/contact preprocessing
        v
Wuji Hand2 Beta1 kinematic IK
        v
SPIDER physics-informed optimization
        +--> visualization
        +--> metrics
        +--> downstream export
```

S2 ends at the target-embodiment boundary. `wuji_hand2_beta1` is an implicit
asset-directory robot type, not a new global registry. `generate_xml.py` calls
`ensure_robot_assets()` before loading an adapter, staging exactly one
manifest-checked robot directory under:

```text
<dataset-root>/processed/<dataset-name>/assets/robots/wuji_hand2_beta1
```

The helper copies no data and refuses to overwrite a conflicting staged model.
Scene generation selects `right.xml`, `left.xml`, or `bimanual.xml` by the
existing `embodiment_type` convention. The existing IK code consumes palm and
five-fingertip sites; the adapter supplies the required names.

Current SPIDER output paths remain:

```text
<dataset-root>/processed/<dataset-name>/<robot-type>/<embodiment>/<task>/<data-id>/
```

The scene places robot qpos before later object qpos. For bimanual Wuji, controls
and robot qpos are `[right 26][left 26]`; a future generated object remains
after those 52 robot coordinates. Full dataset adapters and physics optimization
configuration are future stages, not an S2 claim.

## Stage C contact contracts

Stage C keeps two independent contracts. V1 requires the source hand, finger,
and object patch exactly and is permanently blocked by the Wuji embodiment
contact conflict. V2 permits only bounded, same-hand reassignment to a
functional robot contact region while preserving source role, interval, and
mesh-adjacent object surface patch. It never modifies raw GRAB, object
trajectory, Stage B, or V1. See [contact contracts](CONTACT_CONTRACTS.md) and
the [V2 handoff](HANDOFF_STAGE_C_CONTRACT_V2.md).

Corrected C-XA Level-1 may enter dynamic acceptance only in a separate
`stage_c_v2_dynamic` external namespace. A fine-step D2 retry repaired object
tracking but still failed dynamic V2-contact and robot-quality gates, so MJWP,
shared-profile smokes, HTML, screenshots, and Stage D remain prohibited; see
[the dynamic handoff](HANDOFF_STAGE_C_V2_DYNAMIC.md).

## External source contract

Stage A adds a separate, external workspace contract without changing the
legacy `dataset_dir` layout consumed by existing SPIDER processors. Configure
source data, body models, and workspace through an ignored local copy of
`configs/project/paths.example.yaml`. Canonical HOI outputs are the adapter
boundary; the later GRAB bridge materializes the established SPIDER layout only
inside the external workspace.
