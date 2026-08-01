# Stage C-XAE-M1R：Step-5 控制权与可达性审计

## 结论与边界

本审计在冻结的 Stage C-XAE / XAE-M1 输入上完成；实际执行仅使用原有 `env.step(action)` 路径，未修改 GRAB、body model、patch、role、finger、阈值、物体位姿、root/wrist 静态 reference，也未开启 `REGRASP`。有效运行目录为：

```text
.local_artifacts/stage_c_xae_m1r/20260801T153353Z-step5-control-authority
```

最终根因是 `CONTROLLER_REALIZATION_FAILURE`：短时隙中确有左食指控制变化，但其对指尖运动的实际实现不足以阻止接触法向分离。它不是“左食指在约束内不可达”，也没有证据表明是 loss 同步的 solver 冲量弹飞。

| 项目 | 结果 | 关键证据 |
| --- | --- | --- |
| O1 控制权 | `CONTROL_DELAY_ERROR` | R2/R3 的命令没有被清零；step 5 前 ctrl 最大差为 0.000612，qvel 最大差为 0.000333，指尖速度最大差仅为 2.08e-05 m/s。8 ms bumpless ramp 在约 2.5 ms 的 step-5 窗口中过度衰减。 |
| O2 左食指可达性 | `DYNAMICALLY_REACHABLE_WITHIN_LEFT_INDEX_BOUNDS` | 局部雅可比 rank=3、condition=4.5063、最小奇异值=0.02406；法向需求/可达比=0.01158，切向比=0.00195；精确和有界求解均成功，未触发 joint/actuator limit。 |
| O3 隔离无接触诊断 | `CONTROLLER_REALIZATION_FAILURE` | 仅在克隆模型中关闭不可变 assigned pair `collision_hand_left_index_8|right_object_0`（pair index 104），其余模型项不变。contact/no-contact 均呈近似相同的间隙与跟踪劣化，因此无接触实验只是 `DIAGNOSTIC_ONLY`，绝非 witness。 |
| O4 接触动力学 | `NORMAL_SEPARATION` | 法向分离起点 step 5，切向滑移起点 step 3；冲量峰值在 step 1、力在 step 2 已降为零，不能归因于 loss 同步冲量峰。 |

## 冻结基线复现

R0/R2/R3 均在 step 5 首次丢失 assigned contact。R0 于该步的 normal gap 为 7.289 mm、relative normal velocity 为 0.4577 m/s、tangential speed 为 0.07813 m/s；R2/R3 与其一致到微小数值差异。实际丢失的 assigned geom 为 `NONE`，而非被其它 contact pair 静默替代。

## 有界修复与门禁

仅尝试两个允许的最小控制修复，均在每个候选上先重跑 ContractV2 + geometry 和 M0：

| 候选 | 允许改动 | ContractV2/geometry | M0 | Step-5 |
| --- | --- | --- | --- | --- |
| C1 | 立即组合 surface correction | PASS | PASS | FAIL |
| C2（选定） | 立即积分 actuator target 的 surface correction | PASS | PASS | FAIL |

C2 在 step 5 仍为 assigned=false：gap=7.270 mm、normal velocity=0.4339 m/s、tangential speed=0.07677 m/s、penetration=0.286 mm、force=0。它通过 finite、force 安全、joint、无 regrasp、normal/patch gap、penetration、tracking/object/warning 等项，但未满足 assigned contact 持续、terminal assigned contact 与“无持续法向分离”，故 Step-5 gate 必须 `FAIL`。

由此，two-frame 为 `NOT_RUN_DUE_TO_STEP5_GATE`，M1 为 `NOT_RUN_DUE_TO_TWO_FRAME_GATE`，M1 witness 为 `NOT_FOUND`；M2/M3、完整 primary、Oracle C/D2、MJWP、smokes 和 Stage D 均未运行（分别为 `NOT_RUN`/`NOT_STARTED`）。不得把这些未运行项目描述为成功。

## 可视化与人工验收

运行产出 27 张 PNG，manifest 全部 PASS。`m1r_event_0_world.png` 显示 source 的语义 patch、index 区域和接触向量；`m1r_event_5_object.png` 显示 `assigned=false`、7.270 mm gap 与法向分离；`m1r_event_8_wrist.png` 显示 FAILED 后仍在分离。可视化不支持“高力/深穿透”、求解器冲量弹飞或错误 pair 接管的结论。

执行者已完成截图一致性复核；用户视觉验收仍为 `PENDING`，不得以执行者复核取代用户人工验收。
