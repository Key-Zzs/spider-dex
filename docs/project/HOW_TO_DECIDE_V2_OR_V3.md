# How to decide V2 or V3 from the failure viewer

Open `failure_diagnostic_index.html`, then start with source frames 1464 and
1465. Enable `reference_actual` to compare the complete purple/cyan reference
hands with the magenta/teal actual hands, then enable `patch_contacts` to
inspect the green patch surface, anchor, and physical MuJoCo contact point.
Use `visual_collision` for the complete actual hand and collision proxy and
`source_stageb_cxa` for the source skeleton, Stage B, and corrected C-XA state.
Finally check `force_penetration` and the four synchronized curve panels.

Fix V2 implementation when the physical contact point is visibly inside the
green patch but the evaluator reports loss, when the correct fingertip is
mapped to another link, or when a thin-wall outer contact is matched to the
inside surface. Keep all thresholds unchanged while fixing the mapping or time
alignment.

Extend V2 contact modes when static contact before/after the event is feasible,
but original-timing acquisition, release, retention, or regrasp has no bounded
dynamic witness. Use explicit, short acquisition/retention/release/regrasp
windows; do not lower global recall.

Upgrade the V2 optimizer only when both the simultaneous state and a real
dynamic-transition witness exist but the current search still fails. That is
evidence for longer-horizon shooting, direct collocation, mode scheduling, or
feasibility-preserving projection without changing the V2 contract.

Enter V3 only when the complete same-frame role set is empirically infeasible,
a feasible ablated subset exists, and the minimum conflict is attributable to
Wuji span, palm width, joint limits, self-collision, or contact topology while
the removed roles are functionally redundant. One optimizer failure, an ugly
HTML view, a near-threshold metric, or a pass after threshold relaxation is not
V3 evidence.

For the current run, the left-index distal collision geom first contacts the
object at frame 1462. By frame 1465 the actual left index has separated from
the lower end-face patch and no physical hand–object contact remains. The
single-role static witness is feasible, but every bounded original-timing
transition fails. The decision is `EXTEND_V2_CONTACT_MODE_TRANSITION`.
