# Stage C-XAE-M1R 人工视觉验收

有效证据目录：

```text
.local_artifacts/stage_c_xae_m1r/20260801T153353Z-step5-control-authority
```

本记录是执行者对真实 3D viewer 导出的 27 张 PNG 所作复核；最终用户视觉验收固定为 `PENDING`，不能自动转为 PASS。

| 检查项 | 执行者复核 | 证据 |
| --- | --- | --- |
| source frame 可见完整手、物体、语义 patch、left-index 区域与接触向量 | [x] | `screenshots/m1r_event_0_world.png` |
| 候选 C2 的命令/ctrl 确有变化，未被 action mask 清零 | [x] | `o1_control_authority/o1_control_authority.json` |
| step-5 指尖没有足够即时跟随该变化 | [x] | `o1_control_authority/o1_control_authority.json`；tip 速度差仅 2.08e-05 m/s |
| step-5 是 assigned pair 法向分离，非切向阈值单独触发 | [x] | `screenshots/m1r_event_5_object.png`，`o4_contact_dynamics/o4_contact_dynamics.json` |
| 没有 loss 同步的 solver 冲量峰或高力/深穿透 | [x] | O4 中峰值在 step 1，力于 step 2 已降为零；C2 penetration 0.286 mm |
| 没有把 assigned pair 静默替换为错误 contact pair | [x] | step-5 的 assigned geom 为 `NONE` |
| root/wrist/object/reference 保持冻结，未启用 regrasp | [x] | `reports/m1r_final_acceptance.json` 与 C2 Step-5 gate |
| 无接触实验未被用作成功 witness | [x] | `o3_contact_ablation/o3_comparison.json` 标为 `DIAGNOSTIC_ONLY` |

建议人工复核顺序：先看 `m1r_event_0_world.png`，再看 `m1r_event_5_object.png` 和 `m1r_event_8_wrist.png`；随后用 HTML viewer 的 event/view/phase 选择器核对同一时刻。若人工观察与报告冲突，保持门禁 FAIL 并记录差异，不得放宽判据或以静态图替代动态 witness。

用户视觉验收：`PENDING`。
