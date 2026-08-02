# Stage C-XAE-M1R4 中文 handoff

Frozen replay: `PASS`
Region mapping: `PASS`
Contact-chain audit: `PASS`
Loss mechanism: `REGION_NORMAL_SEPARATION`
Action probes: `COMPLETE`
Minimal repair: `NOT_APPLIED`
Contract-V2: `PASS`
Geometry: `PASS`
M0: `PASS`
Step-5: `PASS`
Two-frame: `FAIL`
M1: `NOT_RUN`
M1 witness: `NOT_FOUND`
M2/M3: `NOT_RUN`
full primary: `NOT_RUN`
Oracle C/D2: `NOT_RUN`
MJWP: `NOT_RUN`
smokes: `NOT_RUN`
Stage D: `NOT_RUN`
User visual review: `PENDING`
Visualization: `PASS`
Stage C-XAE-M1R4: `EMPIRICALLY_BLOCKED_WITHIN_BOUNDED_FINGER_ACTIONS`

## 冻结身份与 M1R3 结论

- 序列：`s5__cylindermedium_lift`，source frame：`1461`，目标：left/index。
- exact pair：`collision_hand_left_index_8|right_object_0`；冻结 assigned region：`index_5/index_6/index_7/index_8`。
- M1R3 已确认 Step-5 `index_8→index_7` 是同一区域真实 MuJoCo contact，7.286689 mm patch gap 不等同于 0.278119 mm collision penetration。

## Step4–8、映射和根因

- Frozen replay：Step-5 state 与 M1R3 一致；exact pair 首失在 Step 5，assigned region 首失在 Step 7。
- Step-7 没有合法 left-index contact，也没有遗漏的相邻 left-index collision geom，mapping 完整。
- 根因：`ASSIGNED_REGION_RETENTION_UNREACHABLE_WITH_CURRENT_FINGER_ACTION_BOUNDS`；主机制：`REGION_NORMAL_SEPARATION`；次因：`FINGER_ACTION_INSUFFICIENT`。normal gap 与正向 separation velocity 在 Step 5→7 持续增加；切向速度也增加但没有证据将其升格为 primary edge cause。

## M1R3 反事实与动作 probes

- M1R3 solver/friction 反事实仅到 Step-6，且没有 R3 solver trigger；本阶段未重跑无根据的 solver/timestep 矩阵。
- B0 权威原始 action：`REJECTED`，first region loss=`7`，max force=`1.5877 N`。
- B1 M1R3 阶段6 state-space 建议：`REJECTED`，first region loss=`7`，max force=`1.6170 N`。
- B2 小幅法向补偿：`REJECTED`，first region loss=`7`，max force=`1.8516 N`。
- B3 最大合法法向补偿：`REJECTED`，first region loss=`7`，max force=`2.2052 N`。
- B4 小幅切向反滑补偿：`REJECTED`，first region loss=`7`，max force=`1.5573 N`。
- B5 法向加切向组合：`REJECTED`，first region loss=`7`，max force=`1.7442 N`。
- B6 渐进两步状态补偿：`REJECTED`，first region loss=`7`，max force=`2.1638 N`。
- B8 normal-velocity feedback：`REJECTED`，first region loss=`7`，max force=`2.0702 N`。
- B9 slip feedback：`REJECTED`，first region loss=`7`，max force=`1.5640 N`。
- B7 force-decay feedback：`NOT_RUN_NOT_APPLICABLE`，first region loss=`None`，max force=`n/a`。

## 最小修复与固定门禁

- 最小修复：`NOT_APPLIED`。所有安全、有界、left-index-only 候选均于 Step-7 丢失 region；部署状态反馈会把失败伪装成修复，故未改资产、collision、配置或生产控制。
- Contract-V2、Geometry、M0、Step-5 均保持 PASS；Two-frame FAIL。
- M1：`NOT_RUN`；M2/M3：`NOT_RUN`。没有 full primary、Oracle C/D2、MJWP、smokes 或 Stage D。

## 真实三维与限制

- HTML 使用真实 MuJoCo visual mesh、collision geoms、qpos 与 contact telemetry；Chrome PNG 覆盖 Step 5/6/7、pair transition、best probe 和 two-frame terminal。
- 用户视觉验收仍为 `PENDING`。该结果是 10 个协议内候选（其中 B7 因非 force-decay 不适用）的经验性有界阻断，不是数学不可行证明。

## 环境、Git 与产物

- 仓库：`/home/deepcybo/workspace/dex/retarget/spider-dex`；分支：`develop/wuji-hand2`；M1R4 开始时的冻结 base commit：`2a6f8b0621c53797d57f0f5513fca8bf8429acbf`；本阶段本地 commit：`feat(stage-c): audit M1R4 region contact retention`；`pushed: NO`。
- Conda：`spider-dex`。M1R3 权威 run：`.local_artifacts/stage_c_xae_m1r3/20260802T143000Z-contact-truth-dynamics/`。
- 本阶段权威 run：`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r4/20260802T160108Z-assigned-region-retention-final/`；产物目录为 `.gitignore` 覆盖的本地产物，历史 authority 未覆盖。
- 新增源码：`spider/tools/grab_stage_c_xae_m1r4.py`；新增测试：`tests/test_stage_c_xae_m1r4.py`；新增中文阶段、人工验收与本 handoff 文档。未修改 raw GRAB、body model、Stage B、C-XA authority、semantic patch、role/finger/threshold、root/wrist/object 或 collision asset；未 push。

## 真实三维验收命令

```bash
google-chrome \
  "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r4/20260802T160108Z-assigned-region-retention-final/html/stage_c_xae_m1r4_region_retention.html"

google-chrome \
  "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r4/20260802T160108Z-assigned-region-retention-final/html/stage_c_xae_m1r4_visual_index.html"

cd "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r4/20260802T160108Z-assigned-region-retention-final"
python -m http.server 8000
```

打开 `http://127.0.0.1:8000/html/stage_c_xae_m1r4_visual_index.html`。Chrome headless 已实际生成并人工检查 `30/30` PNG。

## 测试

- `conda run --no-capture-output -n spider-dex python -m unittest tests.test_stage_c_xae_m1r4 tests.test_stage_c_xae_m1r3 tests.test_stage_c_xar tests.test_stage_c_cm1r tests.test_stage_c_contact_mode tests.test_stage_c_failure_diagnostic tests.test_stage_c_v2r tests.test_stage_c_v2r2e -v`：`77 tests, OK`。
- `conda run --no-capture-output -n spider-dex python -m unittest discover -s tests -v`：`160 tests, OK`。
- `conda run --no-capture-output -n spider-dex python -m compileall spider tests`：PASS；`git diff --check`：PASS。

## 下一步

Two-frame `FAIL`，禁止运行 M1。若要继续，必须先由用户重新授权改变冻结边界或采用新的、明确审计的接触/机构方案；不得把此处的 diagnostic state-feedback probe 当作已部署修复。
