# Stage C-XAE-M1R 交接

## 当前冻结状态

有效运行：`.local_artifacts/stage_c_xae_m1r/20260801T153353Z-step5-control-authority`。

- O1 `CONTROL_DELAY_ERROR`：R2/R3 实际有 ctrl/qvel 差异，但 8 ms bumpless ramp 在 step-5 短窗中没有转换为足够的指尖运动。
- O2 `DYNAMICALLY_REACHABLE_WITHIN_LEFT_INDEX_BOUNDS`：局部 rank=3，condition=4.5063；有界求解成功、无关节或 actuator 触界。
- O3 `CONTROLLER_REALIZATION_FAILURE`：只关闭 assigned pair 104 的克隆模型诊断与 contact-enabled 结果近似，不能当 witness。
- O4 `NORMAL_SEPARATION`：法向分离从 step 5 开始；没有 loss 对齐的 solver 冲量峰。
- C1/C2 都完成 ContractV2 + geometry 与 M0（PASS），但 Step-5 gate 均 FAIL；C2 为最终选定候选。

因此本阶段状态是 `FAIL`，根因 `CONTROLLER_REALIZATION_FAILURE`。two-frame、M1、M2/M3、完整 primary、Oracle C/D2、MJWP、smokes 与 Stage D 都未运行；M1 witness `NOT_FOUND`。用户视觉验收 `PENDING`。

## 给后续执行者 / 外部 GPT 的中文提示词

```text
请只做 Spider-Dex Stage C-XAE-M1R 的后续“诊断/方案设计”审阅，不得声称门禁已通过。

冻结事实：有效运行在 .local_artifacts/stage_c_xae_m1r/20260801T153353Z-step5-control-authority；R0/R2/R3 都在 step 5 失去 assigned contact。O1 显示 left-index 的 R2/R3 命令与 ctrl 并未被清零，但 8 ms bumpless ramp 在约 2.5 ms 窗口内把物理影响衰减到 tip velocity diff <=2.08e-05 m/s。O2 已证明左食指在 joint/actuator bounds 内局部可达（rank=3、condition=4.5063、有界解成功）。O3 仅关闭 immutable assigned pair collision_hand_left_index_8|right_object_0 的克隆模型，且结果证明是控制实现失败；该无接触实验只能作 DIAGNOSTIC_ONLY，不能当 witness。O4 显示法向分离起点 step 5，solver 冲量峰在 step 1、力在 step 2 已归零，不能说 loss 是 solver 弹飞。C1 immediate composition 和 C2 immediate actuator-target integration 均先通过 ContractV2+geometry、M0，仍 Step-5 FAIL；C2 step5 gap=7.270mm、normal velocity=0.4339m/s、tangential=0.07677m/s、penetration=0.286mm、force=0、assigned=false。

请输出：1) 只基于现有证据的根因判断；2) 至多两个不扩大权限的下一步控制方案（仍只允许 left-index 控制层，仍须 env.step(action)）；3) 每个方案的可证伪实验与 fail-closed gate；4) 明确列出禁止项。禁止改动 raw GRAB、body model、XAE/M1 authority、patch/role/finger/threshold、object、root/wrist static reference；禁止 regrasp、M2/M3、完整 primary、Oracle C/D2、MJWP、smokes、Stage D；禁止放宽判据或把静态/无接触诊断称为动态成功。若需要任何新增权限，请先停止并说明所需用户授权。
```

审阅入口：`reports/m1r_final_acceptance.json`、`o1_control_authority/o1_control_authority.json`、`o2_reachability/o2_reachability_summary.json`、`o3_contact_ablation/o3_comparison.json`、`o4_contact_dynamics/o4_contact_dynamics.json`、`html/stage_c_xae_m1r_step5_audit.html` 与 `screenshots/`。
