# Stage C feasibility certificate

The certificate is bounded evidence, not a mathematical proof of
infeasibility. It freezes the source frame, object pose, C-XA roles, semantic
patches, thresholds, timing, joint limits, and tracking bounds.

At source frame 1465 the active set contains one mandatory support role:
left index on `patch:s5__cylindermedium_lift:0`. Twelve deterministic starts
(C-XA, failed actual, previous C-XA, role-prioritized, and bounded random)
find a simultaneous-contact witness. The best witness has patch distance
`0.006705 m`, penetration `0.000593 m`, minimum joint margin `0.078528`, wrist
RMSE `0.000016 m`, and fingertip RMSE `0.000016 m`; every frozen constraint
passes. The simultaneous result is `FEASIBLE`.

Six original-timing MuJoCo transition starts over local frames `[0,19]` fail
contact continuity and the terminal patch gate. The best bounded candidate has
maximum/terminal patch distance recorded in
`dynamic_transition_feasibility.json`; penetration, joint margin, one-frame
delta, finite force, and fixed source-object tracking remain valid. The result
is `EMPIRICALLY_INFEASIBLE_WITHIN_BOUNDS`, never a claim of mathematical
infeasibility.

Because the complete active set is feasible, role ablation is not needed and
the minimum conflicting-role set is empty. V3 morphology/contact-topology
replacement is therefore not supported by this first-loss evidence.

