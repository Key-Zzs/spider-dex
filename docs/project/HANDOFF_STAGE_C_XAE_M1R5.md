# Stage C-XAE-M1R5 Handoff

## 一句话结论

真实 SMPL-X surface 证据否定了“左食指单指持续 SUPPORT”的前提：关键窗口是 left-index 与 left-middle 的共享接触。历史 role/patch 需要隔离修复，但不能覆盖旧 authority。Wuji 左食指的静态和连续反事实路径存在、action contract 正确；真实动力学仍在 two-frame Step 7 丢失 assigned region。这个动态失败不能被解释为 Wuji 对真实 source role 的 embodiment failure。

## 已完成

- 对 `[1460,1876)` 的 full female SMPL-X hand surfaces（每手 778 vertices）完成 anatomical region、signed/unsigned gap、area、object triangle、normal/velocity、temporal persistence 和 responsibility 审计。
- `1461..1480` 全帧为 `LEFT_INDEX_SHARED`；1461 聚合责任度为 index 0.5844、middle 0.4156。
- 审计历史 10-tip role：137 agreement / 279 false negative / 0 false positive；active 帧均 shared。
- 审计历史 patch：只覆盖真实 source footprint 的 23.3681%。
- 创建隔离 `M1R5-source-surface-v1` role schedule 和 source-footprint semantic patch；无 dilation，保留 old/new triangle lineage。
- 完成 K0/K1/A0 与真实 `FrozenActionEnvironment.step(action)` 动态复现。
- 生成两个自包含真实 3D HTML、48 个 Chrome PNG 与中文 review。

## 决策链

| Gate | 结果 | 说明 |
|---|---|---|
| Source contact truth | PASS | full SMPL-X surface，不是独立 MANO/10-tip authority |
| Existing role | FAIL | `SOURCE_ROLE_FALSE_NEGATIVE`，并漏表达 shared support |
| Existing patch | FAIL | `SOURCE_PATCH_CONTACT_FOOTPRINT_INCOMPLETE` |
| Corrected authority | PASS | `CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID`，left-index premise = `SHARED` |
| Static finger-only | PASS（反事实） | 1461..1466 多起点可行 |
| Continuous finger-only | PASS（反事实） | action offsets 合法，速度/加速度/jerk 已测量 |
| Action contract | PASS | 4D additive-radian target offsets，名称映射正确 |
| M0 / Step-5 | PASS / PASS | 无 direct state write |
| Two-frame | FAIL | Step 7 首次 region loss |
| M1 | NOT_APPLICABLE | Two-frame 未通过，且 shared source role 使旧单指 M1 失去权威资格 |

根因是 `MIXED`：primary `CURRENT_LEFT_INDEX_ONLY_PREMISE_INVALID`，secondary `DYNAMIC_REALIZATION_FAILURE`。因果顺序是 source role truth → role/patch authority → 单食指反事实静态/连续/action → 真实动态。

## 交接约束

- 不得把现有 Two-frame fail 写成真实 source 的单食指 embodiment fail。
- 不得将 M1R5 corrected shared role 用当前 left-index-only Wuji 动态链硬验证；新的 shared-role dynamic stage 需要单独授权。
- 不得改 raw GRAB/body models/Stage B/历史 C-XA/历史 role-patch，不得放宽 20 mm 或 3 mm gate，不得扩大 action bound。
- 未运行：M2/M3、full primary、Oracle C/D2、MJWP、smokes、Stage D、V3。

## 关键路径

- 运行产物：`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final`
- 实现：`spider/tools/grab_stage_c_xae_m1r5.py`
- 单测：`tests/test_stage_c_xae_m1r5.py`
- 人工验收：`docs/project/MANUAL_ACCEPTANCE_STAGE_C_XAE_M1R5.md`

Base commit 是 `499ee3d7cc319f7af489f49aa160253f3522dc78`；本阶段无 push。
