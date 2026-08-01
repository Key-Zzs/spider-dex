# Stage C-XAR：C-XA correction leakage 修复与受限验收

## 范围与冻结边界

本轮只处理 `s5__cylindermedium_lift/0` 的 C-XA wrist/root correction 泄漏；原始 GRAB、body model、Stage B、历史 C-XA、物体 pose、冻结帧、阈值均未修改。新证据只写入：

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair/`

修复候选位于外部派生产物目录：

`/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa_repaired_20260801T101500Z/`

## 根因：`ROOT_WRIST_CORRECTION_LEAKAGE`

历史 C-XA 的 `depenetration_trace_cxa_level_1_flexible.npz` 为 `(414, 52)` correction；所有帧均标为 Phase-1 `finger_only`，但 12 个 root/wrist qpos 仍有非零 correction。第一跳变是 source frame 1461。

实因是 `spider/tools/grab_stage_c.py::_joint_bounds` 的条件顺序：通用 wrist translation/rotation 范围早于 `finger_only` 的锁定分支，令锁定分支不可达。历史 profile 的 Phase-2 与 DLS 又使用完整手 DOF 范围，构成次要放大因素。A 零 correction、C 序列化、旋转表示审计均通过，故不是 wxyz/XYZ、DOF layout、writer/reader 或 viewer 映射问题。

历史错误幅度（相对 Stage B）：左/右腕最大平移分别为 25.946 / 25.780 mm，最大旋转 0.374279 / 0.294064 rad；最大相邻帧腕平移跳变为 29.016 / 24.339 mm。

## 最小修复

- `configs/project/grab_wuji_depenetration.yaml` 新增并默认固定 `allow_wrist_correction: false`。
- `_joint_bounds` 先锁定 finger-only 模式下的 12 个 root/wrist DOF。
- Powell、collision DLS、visual DLS 共享同一 mutable finger 集；每个阶段和序列化前执行 locked-DOF 不变量检查。
- object qpos（52–63）显式保持来自 Stage B，禁止成为优化变量。
- `tests/test_stage_c_xar.py` 覆盖 root/wrist 锁定、显式 opt-in、泄漏拒绝与物理旋转恒等。

## 真实后审计

后审计：`repair/cxa_repair_preservation_audit.json`、`repair/postrepair_A_to_E_regression.json`、`repair/cxa_repair_temporal_audit.json`。

修复候选满足：

- object qpos 与 Stage B 完全相等；物理 object translation/rotation 最大值均为 0。
- 左/右 wrist relative translation/rotation 最大值均为 0；root/wrist correction trace 精确为 0。
- 手指仍实际变化：左/右最大 fingertip 改动 85.464 / 53.894 mm。
- Raw loader：PASS；Stage B：PASS；修复 C-XA source-relative geometry：PASS。

源几何正式结论在：

`source_geometry_audit/20260801T102900Z-repaired-cxa-geometry/reports/source_geometry_final_decision.json`

## 接触门禁与停止条件

修复候选的静态 depenetration 为 PASS：最大 collision penetration 为 1.562 mm，object pose change 为 0，joint-limit violation 为 0。

但是冻结 Contract-V2 仍为 FAIL，且只失败：

| 指标 | 修复候选 | 冻结阈值 | 结论 |
| --- | ---: | ---: | --- |
| surface patch distance P95 | 21.582 mm | <= 20.000 mm | FAIL |
| patch coverage | 0.889665 | >= 0.70 | PASS |
| functional role recall | 0.882353 | 通过 | PASS |
| normal cosine median | 0.999996 | 通过 | PASS |

唯一受控 probe 仅将 contact weight 提升到 10000；root/wrist 锁定仍完全保持，但 P95 恶化到 22.803 mm。因此记录为 `STOP_NO_FURTHER_TUNING`，未通过牺牲腕锁定或放宽冻结阈值来换取结果。

最终状态是 `GEOMETRY_REPAIRED_CONTACT_BLOCKED_USER_VISUAL_PENDING`，不是 C-XA 全部验收通过。

## 后续实验状态

- M0：`NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE`
- 两帧：`NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE`
- M1：`NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE`
- M2、M3、full primary、Oracle C/D2、MJWP、smoke、Stage D：未运行且不在本轮授权范围。

禁止在接触 P95 未通过时把任何下游完成状态称为成功。

