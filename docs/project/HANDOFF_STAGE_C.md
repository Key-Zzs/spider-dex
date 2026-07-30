# Stage C handoff

Repository: `/home/deepcybo/workspace/dex/retarget/spider-dex` on
`develop/wuji-hand2`. Stage C began at `ef44c3d46b4fc76cb208721f53c9954c7c2f0e2a`.
No commit, push, raw GRAB modification, or Stage B overwrite was performed.

Completed and externally materialized: frozen pilot manifest; source geometry
diagnostics; surface-projected contact references; validated hash-addressed
collision cache; isolated contact-guided MJWP input for the primary; and
fail-closed reports. The shared profile is
`configs/project/grab_wuji_stage_c.yaml`; its orchestration contract is
`configs/project/grab_wuji_stage_c_contract.yaml`.

The primary `s5/cylindermedium_lift` (frames `[1460,1876)`) fails before a
finite physics trajectory can be accepted. MJWP reports NaN rewards for all
16 samples; CPU replay reaches qacc 5.538e7 at 0.03s and 1.239e26 at 0.04s.
This is caused by the frozen kinematic hand-object penetration at physics
initialization. The right-hand and offhand smoke pilots retain their frozen
identities and were not substituted after this primary hard failure.

Important external paths are in `reports/stage_c_validation.json`, each
pilot's `<Stage B robot dir>/stage_c/`, `cache/objects/`, and
`stage_c_inputs/s5__cylindermedium_lift/`. Raw GRAB remains unchanged.

Resume by reading this file and the JSON report. A future Stage C repair must
implement a physically valid, optimizer-integrated depenetrating initialization
that is evaluated against the same frozen references; it must not translate
hands statically, delete frames, hide geometry, or relax gates. Only after the
primary gets a finite rollout may profile search, smoke runs, metrics, HTML,
and screenshot review resume. Stage D is not started.
