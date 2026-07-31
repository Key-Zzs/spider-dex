# Stage C-V2R2E Contact-Aware Dynamic Recovery

Status: **BLOCKED after bounded recovery branches**.

The primary-first controller ran 12 original-timing actual MuJoCo dynamic
profiles. The best observed profile reached patch coverage `0.317039`,
functional-role recall `0.117647`, and patch-distance P95 `0.081810 m`; it
still failed joint limits, collision depth, tracking, and force-growth gates.

The dynamic search also ran four bounded Jacobian contact-target profiles; they
did not recover the immutable patch/role contract. The next object-guidance
branch ran five phase-scheduled profiles (`G0`–`G4`), all failed. A new
V2R2E contact-dynamics branch ran eight in-memory explicit-pair profiles, all
failed; the preserved historical V2R2 Branch-D evidence remains separate. An
isolated timing ladder independently failed at `1.0x`, `1.25x`, `1.5x`, and
`2.0x`.

The final attempt is recorded under:

`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260801T004500Z-contactik/`

The runner never writes object qpos, replaces a frozen frame, edits raw data,
or converts a failed candidate into a seed. Since Oracle C never passed,
Oracle D2, Minimal MJWP, Full MJWP, smokes, HTML, screenshots, and Stage D
remain not run.
