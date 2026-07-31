# Stage C-V2R2E Physical–Semantic Contact Alignment

Status: **PASS for alignment / BLOCKED for downstream dynamic retention**.

The frozen primary is `s5/cylindermedium_lift`, source frames `[1460, 1876)`.
The audit consumes the immutable corrected C-XA Level-1 input and the actual
MuJoCo scene under the configured external workspace. Raw GRAB, body models,
Stage B, C-XA outputs, and the 197 `UNRELIABLE_SOURCE` records are unchanged.

Evidence:

- `/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260731_v2r2e_primary_v1/reports/v2r2e_alignment_audit.json`
- `/mnt/nas/storage/Ref2Dex_storage/spider_workspace/runs/stage_c_v2r2e/20260731_v2r2e_primary_v1/reports/V2R2E_ALIGNMENT_AUDIT.md`
- `/mnt/nas/storage/Ref2Dex_storage/spider_workspace/reports/stage_c_v2r2e_validation.json`

The object visual/collision bbox ratio is `1.0` on every axis; patch-to-
collision P95 is `0.002015 m`; normal cosine median is `0.99999999`. The
explicit Wuji mapping covers both sides and all five fingers. The first
preserved dynamic contact is `collision_hand_left_index_8 ↔ right_object_0`,
which maps to the expected left-index region, but contact-to-semantic-patch
P95 is `0.050128 m`; this is a dynamic retention failure, not a justified
geometry or mapping rewrite.
