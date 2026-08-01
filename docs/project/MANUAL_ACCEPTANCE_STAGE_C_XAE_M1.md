# Stage C-XAE-M1 人工可视化验收

本阶段的 Codex Chrome 复核已完成；用户验收仍为 `PENDING`。

```bash
google-chrome \
  "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention/html/stage_c_xae_m1_retention.html"
```

优先检查 `1461 / step 0 / RETAIN_PENDING`、`1461 / step 5 / FAILED`，以及明确标为
`REFERENCE_ONLY_AFTER_GATE_STOP` 的 1462..1466 reference context。后者不是 later-M1
actual witness：双帧 gate 已经在 source 1461 失败，M1 因此未获运行授权。

人工复核报告：
`.local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention/reports/XAE_M1_SCREENSHOT_REVIEW.md`。

复核应确认：initial left-index assigned contact、同 substep surface target、first normal
separation、没有深穿透/高力、没有 object qpos 写入、没有 root/wrist static offset，以及终端
`FAILED` 与数值报告一致。
