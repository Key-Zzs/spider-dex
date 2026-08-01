# Stage C C-M1R M1 recovery handoff

Environment: `/home/deepcybo/workspace/dex/retarget/spider-dex`, conda
`spider-dex`, RTX 5080, branch `develop/wuji-hand2`, base
`e9f248ec6aa581ae139fae8d8f7f5a32a858b65b`.

Frozen primary: `s5/cylindermedium_lift`; source `1461..1480`, 120 Hz;
left-index SUPPORT, role `s5__cylindermedium_lift:0`, patch
`patch:s5__cylindermedium_lift:0`, 20 mm gate. Latest run:
`.local_artifacts/stage_c_cm1r/20260801T072352Z-m1-recovery/`.

M0: PASS, continuity 1.0, patch P95 7.22 mm, peak 0.339 N. Historical M1
had 8/8 profiles first lose the assigned pair at 1462/substep 2. The causal
audit found the old loop applied the whole 120-Hz object target before each
0.5-ms integration step and reset the bumpless controller on
`RETAIN_PENDING → RETAIN`.

Implemented: substep source interpolation, actual-object object-local target
transform, deterministic rigid-body target velocity feedforward, separate
pre/post logs, and continuous confirmation handover. S0 PASS, S1 FAIL, S2
PASS, S3 FAIL, S4 FAIL, S5 FAIL_SAFE_FORCE; S6 is not required because left
wrist/index is already the legal controller set.

Two-frame and M1 both FAIL. The remaining classification is
`TARGET_TIME_ALIGNMENT_ERROR_PLUS_OBJECT_COUPLED_NORMAL_SEPARATION`.
Contact-consistent source velocity initialization reaches 0.00146 m/s
residual but peaks at 107.35 N, so it is rejected. M2/M3 remain NOT_RUN.
Real 3D HTML and Chrome screenshots exist; Codex image review is recorded in
the artifact reports and user review remains PENDING. No raw GRAB/body model
changes, direct qpos writes, V3, Oracle C, D2, MJWP, smokes, or Stage D.
