# Embodiment-aware task-equivalent contact retargeting

Contract V2 is a constrained retargeting layer for the Wuji Hand2 embodiment.
It derives source functional roles and object-local surface patches before
considering robot reachability, then considers the relaxation ladder in order:

1. Same finger with a flexible local contact point.
2. Same functional non-thumb finger group.
3. Same-hand task-equivalent contact set.
4. Same-hand functional-surface grasp.

Each level has a bounded temporal assignment: sliding-window evidence,
warm-startable Viterbi state, explicit switch cost, region capacity, and a
maximum of two switches per second per contact interval.  Mesh adjacency and
normal consistency reject thin-wall opposite-surface shortcuts.  Cross-hand
transfer, offhand compensation, arbitrary palm substitution, object motion,
and proximal-link fallback are not permitted.

Only a full physical pass of the first feasible level may be selected.  A
candidate-generation result is not a physics pass: it must still pass static
forward, holds, rollout, finite MJWP, penetration, tracking, patch coverage,
role recall, and visual screenshot review.
