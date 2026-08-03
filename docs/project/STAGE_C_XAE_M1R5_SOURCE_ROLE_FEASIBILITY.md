# Stage C-XAE-M1R5：Source 接触角色真值与单指连续可行性

## 结论

冻结样本 `s5/cylindermedium_lift`、source `[1460,1876)`、120 Hz 的真实 full-body female SMPL-X surface 审计表明：1461..1480 的 SUPPORT 不是左食指单指独占，而是左食指和左中指共享。1461 的责任度分别为 0.5844 与 0.4156；整个关键窗口均为 `LEFT_INDEX_SHARED`。因此 `CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID`，历史 left-index-only 动态链只能保留为反事实诊断，不能据此宣称 Wuji embodiment 失败。

M1R5 创建了隔离的 `M1R5-source-surface-v1` authority；历史 role 与 patch 都保持原样。修复 patch 是真实 source object-contact triangle footprint 的并集，无 dilation、无阈值变化、无为 Wuji 可达性而扩张。历史 patch 只覆盖该 footprint 的 23.3681%。

## Source 接触摘要

阈值仍是既有的 15 mm source-contact 口径；表中为 full-surface anatomical region 的全序列统计，面积为 P95（m²），最小间隙为 unsigned minimum（m），责任度为 P95。

| 区域 | 接触帧数 | P95 面积 | 最小间隙 | P95 责任度 | 结论 |
|---|---:|---:|---:|---:|---|
| left_index_distal | 396 | 9.7489e-4 | 0.000001 | 0.5424 | 持续接触、共享 |
| left_index_middle | 393 | 1.0766e-3 | 0.000002 | 0.5408 | 持续接触、共享 |
| left_index_proximal | 394 | 1.6765e-3 | 0.000000 | 0.5399 | 持续接触、共享 |
| left_middle_distal | 382 | 9.3194e-4 | 0.000001 | 0.5401 | 持续接触、共享 |
| left_middle_middle | 416 | 1.1605e-3 | 0.000002 | 0.5383 | 持续接触、共享 |
| left_middle_proximal | 338 | 1.6482e-3 | 0.000003 | 0.5360 | 持续接触、共享 |
| left_palm | 114 | 1.8976e-3 | 0.000046 | 0.4994 | 辅助接触，不是关键窗口主因 |

历史 10-tip role 的绝对帧映射审计得到：137 帧 agreement、279 帧 false negative、0 帧 false positive；其 137 个 active 帧全部是 shared-contact 帧。故现有 role extraction 为 `SOURCE_ROLE_FALSE_NEGATIVE`，并且对 shared support 欠表达。

## 从旧 authority 到修复 authority

| 项目 | 历史 | M1R5 真值/修复 |
|---|---|---|
| contact evidence | 10 个 SMPL-X fingertip joints | 左右各 778 个真实 SMPL-X hand-surface vertices |
| role | `left/index/SUPPORT` 单一标签 | 帧级 `left_index + left_middle` shared support |
| interval | 1461..1597，137 帧 | full source surface truth `[1460,1876)`；关键窗连续共享 |
| patch | `patch:s5__cylindermedium_lift:0` | source object-contact triangle footprint，保留 triangle IDs 与频率 |
| patch 覆盖 | source footprint 的 23.3681% | 100% 自身的 source-footprint evidence；不做 dilation |
| authority | 历史 frozen，保留 | `M1R5-source-surface-v1`，隔离版本化 |

实现位于 `spider/tools/grab_stage_c_xae_m1r5.py`：用命名 SMPL-X LBS 权重把每个 778-vertex hand surface 划入 distal/middle/proximal/palm 区域；role attribution 先在同一 finger 的三个子区域聚合，再做 responsibility/shared 分类，避免把中指三个子区域拆开后错误地让食指“胜出”。

## left-index-only 反事实可行性链

该链不再是修复 authority 的门禁；它只回答历史左食指假设的局部反事实问题。

| 节点 | 结果 | 真实证据 |
|---|---|---|
| K0 静态 | PASS | 1461..1466 的固定五起点均找到可行 left-index-only 状态；root/wrist/object 不作为变量 |
| K1 连续 | PASS | 逐帧候选、120 Hz 速度/加速度/jerk 测量、合法 4D action offset |
| A0 action | PASS | 名称解析 `l_FFJ0..3` 到 qpos/qvel 36..39；单位是 additive radians，无 normalize/dt/double scaling |
| D0 M0 | PASS | 全部经 `FrozenActionEnvironment.step(action)`；最大 force 1.2446 N，penetration 0.7265 mm |
| Step-5 | PASS | 最大 force 1.4943 N，patch P95 7.4706 mm |
| Two-frame | FAIL | Step 7 首次丢失 assigned left-index region；终点无 region contact |
| M1 1461..1466 | NOT_APPLICABLE | Two-frame 未过，且 corrected authority 不是单食指；未伪造 M1 PASS |

root/wrist/object 的冻结语义与历史 M1R3/M1R4 一致：它们仅由既有 source/reference trajectory 驱动，不是 action 坐标，也没有 post-init qpos/qvel/object-qpos 写入。MuJoCo PD 积分中的被动 tracking error 被保存为 telemetry，而不被混同为“解锁”。

## 范围与禁止项

本阶段未修改 raw GRAB、body models、Stage B、历史 C-XA、历史 role/patch；没有阈值放宽、任意 patch widening、换手指补偿、REGRASP、MPC、attach/weld、post-init qpos/qvel 写入。M2/M3、full primary、Oracle C/D2、MJWP、smokes、Stage D 与 V3 均未运行。

最终 artifact 根目录：

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final`
