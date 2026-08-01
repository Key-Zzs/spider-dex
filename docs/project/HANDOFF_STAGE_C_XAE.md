# Handoff：Stage C-XAE

日期：2026-08-01
权威 run：`.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment`

## 交接状态

```text
XAE: PASS
Contract-V2: PASS
M0: PASS
M1: NOT_RUN
M2: NOT_RUN
M3: NOT_RUN
```

当前任务已在 Stage C-XAE/M0 边界停止。没有运行或暗示 M1、M2、M3、MJWP，没有 push。

以下命名空间被保留但不用于最终结论：`20260801T123713Z-contact-alignment`（只到 E3）、`20260801T124758Z-contact-alignment`（4 ms legacy bumpless 诊断失败）、`20260801T125249Z-contact-alignment`（数值通过但 E0/E2/E3 证据粒度尚未完全展开）和 `20260801T141100Z-contact-alignment`（新增 E3 normal audit 遇到只读 buffer 后 fail-closed）。没有覆盖或删除这些历史产物；最终结论只引用 `20260801T141129Z-contact-alignment`。

completion audit 另以当前代码完整重放到 `20260801T141349Z-contact-alignment`：E0-E6、Contract-V2、geometry preservation、M0 状态及全部数值与权威 run 一致，15 张截图逐文件 SHA256 也一致。该 replay 仅作重复性确认；唯一 handoff 主路径仍为 `20260801T141129Z-contact-alignment`。

## 可直接接管的结论

repaired C-XA 的唯一失败门不是 finger-only embodiment limitation，而是优化/验收定义错位：旧 refinement 优化 fixed anchor，Contract-V2 验收同一 semantic patch 的 nearest surface。E2 已排除 mapping，E3 已用独立 triangle-distance 实现排除 evaluator，E5 已排除 DLS 方向错误。

最小 surface-aligned finger-only refinement 将 P95 从 `21.582210 mm` 降至 `19.904335 mm`，不改变 root、wrist、object、patch、role、finger、functional denominator、20 mm threshold 或 evaluator。Contract-V2 与 geometry preservation 均 PASS，因此自动运行的 finger-only M0 也 PASS。

## 权威证据位置

根目录：

```text
/home/deepcybo/workspace/dex/retarget/spider-dex/
.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment/
```

关键文件：

- `reports/xae_final_acceptance.json`：总状态与 E0-E6/M0/M1-M3 状态。
- `reports/e0_p95_tail_attribution.json`：52 个失败 sample 的完整字段和尾部归因。
- `experiments/e1_objective_alignment.json`：objective/evaluator mismatch 证据。
- `experiments/e2_mapping_audit.json`：52/52 失败样本逐条 role/patch/contact-region mapping。
- `experiments/e3_distance_validation.json`：独立 nearest-surface 距离、最近三角形与法向量交叉验证。
- `experiments/e4_temporal_ablation.json`：single/3/5/9-frame 消融。
- `experiments/e5_dls_audit.json`：gradient、Jacobian、DLS 与 line-search 记录。
- `experiments/e6_feasibility.json`：按协议未运行的原因。
- `repair/repaired_cxa_v2_final/`：最终独立 repaired C-XA v2；没有覆盖上游目录。
- `repair/final_geometry_preservation_audit.json`：冻结几何与输入哈希审计。
- `reports/contract_v2_after_xae.json`：Contract-V2 9/9 gates PASS。
- `reports/m0_after_xae.json`：M0 12/12 gates PASS。
- `screenshots/XAE_SCREENSHOT_MANIFEST.json`：Chrome 15/15 PASS。
- `screenshots/XAE_SCREENSHOT_REVIEW.md`：15 张图的人工视觉审查。

## 关键数字

### E0 before

- 716 samples；52 个 `>20 mm`。
- P50/P75/P90/P95/P99/max：`7.511901 / 10.534290 / 16.324656 / 21.582210 / 25.304448 / 28.908359 mm`。
- `left_middle`、role `s5__cylindermedium_lift:8`、source `1778..1831` 主导；不是少数帧或 role-boundary 集中。

### Contract-V2 after

- surface patch P95：`19.904335 mm`。
- patch coverage：`0.892458101`。
- functional role recall：`0.882352941`，denominator `17` 未改。
- normal cosine median：`0.999995859`。
- collision/visual penetration max：`2.796385 / 2.796123 mm`。
- 414 frames finite、0 warnings；所有 Contract gates true。
- locked root/wrist max abs 与 object max abs 均为 `0`。

### M0

- `controlled_joint_set=[left_index]`，没有 wrist/root/object correction。
- continuity `1.0`，patch P95 `6.866389 mm`。
- force max `0.756245 N`。
- MuJoCo/visual penetration max `0.555370 / 0.247743 mm`。
- minimum joint margin fraction `0.101809`，0 warnings，terminal `COMPLETE`。

## 复现命令

XAE 全流程会创建新的 run root；不得指向现有权威 run 以免覆盖：

```bash
conda run --no-capture-output -n spider-dex python -m spider.tools.grab_stage_c_xae \
  --run-root .local_artifacts/stage_c_xae/<new-run-id> \
  --paths-config configs/local/paths.yaml \
  --repaired-root /mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa_repaired_20260801T101500Z
```

可视化重建命令见 `docs/project/STAGE_C_XAE_CONTACT_ALIGNMENT.md`。权威 HTML 可直接打开：

```text
.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment/html/stage_c_xae_visual_index.html
```

## 下游边界

本 handoff 只证明 XAE 和指定 M0。若未来用户明确授权继续 M1，必须从这个最终 repaired trajectory 和本轮冻结 contract 出发，重新建立 M1 的独立动态门禁；不得把历史 M1/M2/M3 结果重标为本轮结果，也不得因个别 final sample 仍大于 20 mm 而改成 max gate、删帧或放宽阈值。

不得：解锁 wrist/root、移动 object、修改 patch/role/finger/denominator/evaluator、提高 20 mm threshold、增加 contact weight、覆盖历史 C-XA 或上游 repaired C-XA。

## 验证状态

- 指定回归：62 tests，`OK`。
- 全量 discovery：120 tests，`OK`。
- compileall：PASS。
- git diff check：PASS。
