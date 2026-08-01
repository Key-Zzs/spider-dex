# Stage C C-M1R M1 recovery

The recovery run repairs two common implementation defects: whole-frame
object-target updates before a 0.5-ms step, and a bumpless-ramp reset on the
confirmation transition. The real differential experiment remains fail-closed:
object motion causes assigned-pair loss even after the timing repair, whereas
hand-only motion passes.

The source-derived t0 contact-velocity projection is reproducible and reduces
the initial velocity residual to 0.00146 m/s, but causes a 107.35 N force
peak. It is archived as failed safety evidence, not accepted as a normal
preload or witness. Therefore the authoritative result is M1 `FAIL`, M2/M3
`NOT_RUN`, and no dynamic-contact witness.
