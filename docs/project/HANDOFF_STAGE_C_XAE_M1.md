# Handoff：Stage C-XAE-M1

本阶段在 two-frame gate 后停止。唯一上游是 XAE authority
`.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment` 的 final repaired trajectory；
run root 为 `.local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention`。

```text
XAE / Contract-V2 regression / M0 regression: PASS
two-frame: FAIL
M1: FAIL (完整 M1 未获双帧 gate 授权)
M1 witness: NOT_FOUND
M2/M3/full primary/Oracle C-D2/MJWP/smokes: NOT_RUN
Stage D: NOT_STARTED
用户视觉验收: PENDING
```

lineage manifest 证明没有使用历史 M1 target/seed/profile/label；target 是在每个 substep
基于 final XAE trajectory、immutable patch triangles 和 current actual object pose 重建的
nearest-surface set。M0 重回归通过。R0、R2 和 R3 都在 1461 / step 5 首次丢失 assigned pair，
分类为 `OBJECT_COUPLED_NORMAL_SEPARATION`；R2/R3 没有改善，未做无依据的 Kp、lead 或 contact
weight 网格搜索。

继续工作必须从这个 fail evidence 开始；不得把本次 FAIL 转写为 M2/M3 或完整 M1 结论，也不得
解锁 root/wrist、移动 object、放宽 patch/20-mm/penetration gate 或启用 REGRASP。
