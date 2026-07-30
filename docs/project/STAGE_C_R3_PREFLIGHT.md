# Stage C-R3 primary physics preflight

Status: **PASS** for the frozen primary `s5/cylindermedium_lift`, source
frames `[1460, 1876)`.  Smoke pilots were not started.

The derived C-R2 trajectory is replayed only from the external Stage-C recovery
directory.  A MuJoCo mocap-weld supplies an explicit physical object reference;
the implementation never rewrites object `qpos` after initialization.  The
legacy serial-Euler object actuators remain schema-compatible but have zero
gain during this preflight, avoiding periodic-coordinate servo jumps.

The reports are external artifacts under:

`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_recovery/`

They include all-frame static forward (`preflight_static.json`), six 0.2-second
holds (`preflight_holds.json`), full dynamic replay
(`metrics_depenetrated_rollout.json`), the actual bounded
`examples/run_mjwp.py` trace (`minimal_mjwp_run.json`), and the fail-closed
aggregate (`stage_c_r3_primary.json`).  The aggregate records all R3-01 through
R3-11 as passing, including finite real-MJWP rewards/states and the complete
regression suite.

The minimal-MJWP repair uses a population standard deviation for top-sample
normalization.  A one-element retained candidate set has an undefined unbiased
standard deviation in PyTorch, which previously created NaN controls despite
finite physics.  The replacement is mathematically defined for singleton
candidate sets and is covered by `tests/test_sampling_weights.py`.
