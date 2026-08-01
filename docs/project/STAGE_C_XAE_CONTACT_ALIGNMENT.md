# Stage C-XAE：Contact Evaluation Alignment Experiments

日期：2026-08-01
分支：`develop/wuji-hand2`
实验基线 commit：`5b5332dc87f257c4f1c6b02ee600e4f148f6c941`
权威 run：`.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment`

## 最终状态

```text
XAE: PASS
Contract-V2: PASS
M0: PASS
M1: NOT_RUN
M2: NOT_RUN
M3: NOT_RUN
```

本轮没有运行 M1/M2/M3/MJWP，也没有 push。Raw GRAB、body models、Stage B、historical C-XA 与输入 repaired C-XA 均未覆盖。

## 固定输入与边界

输入 repaired C-XA：

```text
/mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/
wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/
stage_c_contract_v2_cxa_repaired_20260801T101500Z
```

冻结约束：root translation/rotation、双腕、object、semantic patch、role assignment、functional denominator、20 mm gate、penetration threshold 全部不变；不得换 finger、删失败帧、改 evaluator、加 static offset 或继续增大 contact weight。

输入哈希：

- repaired C-XA trajectory：`ffbdf3f357a04a717f1c45a6cb0d35713200a8a65db90344823fe8b484ba016d`
- Stage B trajectory：`816e3759450393656de0581543c75fb658d4afecb84d313ecf3d7d947182e2dd`

## 实验总览

| 实验 | 状态 | 分类 / 结论 |
| --- | --- | --- |
| E0 | PASS | 716 个样本中 52 个大于 20 mm；不是少数离群帧 |
| E1 | PASS | `CONTACT_OBJECTIVE_EVALUATOR_MISMATCH` |
| E2 | PASS | 52/52 失败样本的 role / finger / site / visual / collision / patch mapping 全部一致 |
| E3 | PASS | `EVALUATOR_VALID`；独立实现的距离、最近三角形与法向量全部一致 |
| E4 | PASS | `STATIC_FEASIBLE_TEMPORAL_BLOCKED` |
| E5 | PASS | `DLS_VALID` |
| E6 | NOT_RUN | `NOT_RUN_IMPLEMENTATION_ERROR_FOUND`；E1 已发现实现错误，按协议不得运行 |

### E0：P95 尾部归因

全 414 帧、716 个冻结 assignment sample 的分布：

| 指标 | 距离 |
| --- | ---: |
| P50 | 7.511901 mm |
| P75 | 10.534290 mm |
| P90 | 16.324656 mm |
| P95 | 21.582210 mm |
| P99 | 25.304448 mm |
| max | 28.908359 mm |

- `distance > 20 mm`：52 个；完整逐样本表含 frame、side、finger、role、patch id、robot contact region、distance、normal、penetration。
- 主导 finger：`left_middle`；52 个失败中 50 个来自左手。
- 主导 role：`s5__cylindermedium_lift:8`（SUPPORT），38/52，占 73.08%。
- 主导时间段：source frame `1778..1831`；其余为分散的 right-index / left-index 样本。
- 只有 6/52 位于 role 边界两帧内，因此不是 role-boundary concentration。
- 结论：这是持续区间尾部，不是少数帧或可删除的离群值。

证据：`reports/e0_p95_tail_attribution.json` 与 `reports/E0_P95_TAIL_ATTRIBUTION.md`。

### E1：优化目标与验收定义

同一批 716 个 sample 的结果：

| 距离定义 | P95 |
| --- | ---: |
| optimizer fixed anchor | 39.813675 mm |
| patch centroid | 42.160236 mm |
| nearest semantic-patch surface / Contract-V2 | 21.582210 mm |
| point-to-plane | 10.488509 mm |

fixed-anchor 与 nearest-surface 的 Pearson correlation 为 `0.775658`；anchor minus surface 的中位数为 `5.573343 mm`，P95 为 `20.441779 mm`。

optimizer 使用“每个 role/frame 一个预编译 fixed anchor 的平方欧氏残差”，而 evaluator 使用“同一冻结 semantic patch 上所有三角形的最近表面欧氏距离”。semantic patch 与 robot contact region 相同，但 distance definition 不同，因此分类为 `CONTACT_OBJECTIVE_EVALUATOR_MISMATCH`。这也解释了 contact weight 从 5000 增至 10000 反而使 P95 恶化到 22.802893 mm：放大错误目标不能修复定义错位。

### E2：mapping 审计

对 E0 的 52 个失败样本逐条检查 source role、source finger、robot finger、fingertip site/body、visual geoms、collision geoms、contact region 与 semantic patch triangle IDs；`audited_sample_count=52`，结果全部 `PASS`。没有换 finger、扩 patch或改 role。

### E3：距离实现交叉验证

实现 A 为 `trimesh.proximity.closest_point_naive`，实现 B 为独立 Ericson point-to-triangle region 算法。对最差 10 个样本，mesh SHA256、patch triangle IDs hash、object-local coordinate frame、nearest triangle 与 normal 均被序列化；距离最大绝对误差为 `0 m`，法向量最大绝对误差为 `0`，最近三角形 10/10 一致。因此 evaluator 有效，不修改 evaluator。

### E4：单帧 / 时间窗口消融

四个 candidate 都只改 finger DOF，root/wrist/object 精确锁定：

| variant | P95 | max | >20 mm | coverage | collision max | visual max | smoothness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E4-A single frame | 19.904335 mm | 28.605703 mm | 34 | 0.892458 | 2.796385 mm | 2.796123 mm | 0.244980 |
| E4-B window 3 | 19.859192 mm | 28.571583 mm | 33 | 0.893855 | 2.795719 mm | 4.916016 mm | 0.244980 |
| E4-C window 5 | 19.877947 mm | 28.660584 mm | 34 | 0.895251 | 2.795719 mm | 2.795788 mm | 0.244980 |
| E4-C window 9 | 19.932865 mm | 28.753934 mm | 36 | 0.896648 | 2.791813 mm | 2.791930 mm | 0.244980 |

单帧 candidate 满足冻结 P95 与 penetration gates，说明 finger-only 静态可行；3-frame 平滑产生 `4.916016 mm` visual penetration，因此时间平滑可能稀释正确的 surface correction，最终不采用平滑 variant。`max` 仍可超过 20 mm 不构成门禁失败：冻结 Contract-V2 gate 是全样本 P95，不是逐样本 max。

### E5：DLS 方向与尺度

审计最差 10 帧，记录 exact patch gradient、Jacobian rank/condition number、`damping=1e-4`、raw/clipped/projected/final step。19 个被 line-search 接受的 step 全部满足 `gradient dot step < 0`，故为 `DLS_VALID`；没有 DLS direction error 或 clipping error。

### E6：按协议未运行

E6 只有在 E0-E5 没有发现实现错误时才允许运行。E1 已明确发现 objective/evaluator mismatch，因此输出 `NOT_RUN_IMPLEMENTATION_ERROR_FOUND`，没有把本问题误写成 finger-only 数学不可行或 embodiment limitation。

## 根因排序与排除项

1. 第一根因：contact optimizer 的 fixed-anchor 目标与 Contract-V2 的 semantic-patch nearest-surface 定义错位。E1 的同样本距离差异直接证明这一点。
2. 次要因素：3-frame temporal averaging 会稀释有效 correction，并在 E4-B 触发 visual penetration gate；最终没有使用该 variant。
3. 已排除：role/patch/contact-region mapping error（E2 PASS）、evaluator/坐标系/triangle 实现错误（E3 PASS）、DLS 方向或 clipping 错误（E5 PASS）、少数离群帧或 role boundary 集中（E0 否定）。
4. 未声称：finger-only embodiment limitation。静态候选和最终 Contract 都已给出反证；E6 也因协议门禁未运行。

## 最小修复

修复只新增一个最终 refinement：

- 对冻结 assignment 中原先大于 20 mm 的样本，使用同一 semantic patch surface、同一 robot fingertip contact region、同一欧氏 nearest-surface 定义；
- exact surface gradient + finger-only DLS；`damping=1e-4`，单步 clip `0.04 rad`；
- 每个 trial 在真实 MuJoCo state 上做 collision-safe line search，限制 `2.8 mm`；
- 不改 contact weight、patch、role、finger、functional denominator、阈值或 evaluator；
- root、wrist、object qpos 与输入 repaired C-XA bit-exact 保持。

before / after：

| 项目 | before | after |
| --- | ---: | ---: |
| optimizer refinement objective | fixed anchor squared residual | frozen semantic-patch nearest-surface Euclidean distance |
| surface patch distance P95 | 21.582210 mm | 19.904335 mm |
| locked root/wrist max abs | 0 | 0 |
| object max abs | 0 | 0 |

输出轨迹：`repair/repaired_cxa_v2_final/trajectory_depenetrated_init.npz`。

## Contract-V2 复审

Contract-V2：`PASS`，9/9 gates 为 true。

| 指标 | 结果 |
| --- | ---: |
| surface patch distance P95 | 19.904335 mm (`<=20 mm`) |
| patch coverage | 0.892458101 |
| functional role recall | 0.882352941（15/17，denominator 未改） |
| normal cosine median | 0.999995859 |
| assignment switch rate | 0 / s |
| identity change count | 0 |
| collision penetration max | 2.796385 mm |
| visual penetration max | 2.796123 mm |
| static frames / warnings / finite | 414 / 0 / true |

geometry preservation：`PASS`。Raw GRAB、body models、Stage B、historical C-XA 均 unchanged；root/wrist qpos exact、object qpos exact、locked DOF max abs `0`、joint-limit violations `0`、NaN/Inf `0`、role/patch denominator unchanged。

因此满足 `Contract-V2 PASS + geometry preservation PASS`，结论是：**允许运行 M0**。

## M0

按门禁自动运行，profile 为 `m0_xae_finger_only_initial_hold`，`controlled_joint_set=[left_index]`，实际受控列只能是 `36..39`；root/wrist/object correction 仍锁定。最终 `PASS`：

| 指标 | 结果 |
| --- | ---: |
| contact continuity | 1.0 |
| patch distance P95 | 6.866389 mm |
| force max / P95 | 0.756245 N / 0.756245 N |
| MuJoCo penetration max | 0.555370 mm |
| visual penetration max | 0.247743 mm |
| minimum joint margin fraction | 0.101809 |
| warnings | 0 |
| terminal mode | COMPLETE |
| bumpless | PASS（8 ms ramp） |

12/12 M0 gates 为 true；object qpos 未被写入，source object target、role、patch、assigned finger、source timing 与 20 mm threshold 均保持。M1/M2/M3 仍为 `NOT_RUN`。

## 真实 mesh 可视化

HTML：

- `.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment/html/stage_c_xae_contact_audit.html`
- `.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment/html/stage_c_xae_visual_index.html`

渲染命令：

```bash
conda run --no-capture-output -n spider-dex python -m spider.tools.grab_stage_c_xae_viewer \
  --run-root .local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment \
  --paths-config configs/local/paths.yaml \
  --repaired-root /mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa_repaired_20260801T101500Z
```

Chrome headless manifest 为 15/15 `PASS`：P95 contributor、max distance、typical pass、E4 static best、role-boundary tail 各有 world/object/contact close-up。人工审查同样为 `PASS`。E6 未运行，所以没有 E6 failure frame，也没有伪造截图。

## 测试

- 用户指定六模块回归：62 tests，`OK`。
- `python -m unittest discover -s tests -v`：120 tests，`OK`；包含 2 个新 XAE independent geometry tests。
- `python -m compileall spider tests`：exit 0。
- `git diff --check`：PASS（最终 git 审计执行）。

## 代码边界

- `spider/tools/grab_stage_c_xae.py`：E0-E6、surface-aligned repair、Contract-V2、preservation audit 与自动 M0 orchestration。
- `spider/tools/grab_stage_c_xae_viewer.py`：自包含 Plotly real-mesh HTML 与 Chrome 三视角截图。
- `spider/tools/grab_stage_c_cm1r.py`：在不改变 historical wrist+index profiles 的前提下，增加显式 `left_index`-only M0 controlled-set。
- `tests/test_stage_c_xae.py` 与 `tests/test_stage_c_cm1r.py`：独立三角面距离和 wrist-column exclusion 回归。
