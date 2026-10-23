# Stage C-SQA 交接

## Ready / blocked

**数值审计 ready，正式 Wuji penetration authority blocked。** A/B 已完成 formal signed source baseline；C/D 已完成同一 query set 的 diagnostic reverse-SDF，但 Wuji hand collision/visual layer 不是 watertight closed target，因此不能把 C/D signed 数字当作正式 penetration acceptance。

## 最重要的回归

- A/B source：最大正式 penetration depth 10.740 mm，最大帧 source 1831；P95 frame max 9.811 mm；mean frame penetrating point ratio 3.480%。
- A/B 完全一致：object transform 最大绝对 residual 8.388e-8，left/right hand vertex residual 0；说明 source loader 没有新增几何/坐标差异。
- C 标准 Stage B：diagnostic max 7.415 mm，frame 1827；正式状态 `SDF_INVALID_FOR_STAGE`。
- D repaired C-XA：diagnostic max 0.961 mm，frame 1611；正式状态仍为 `SDF_INVALID_FOR_STAGE`，不能从这个数字推导 acceptance。

## 复现

主工具：

```bash
conda run --no-capture-output -n spider-dex \
  python -m spider.tools.grab_stage_c_source_quality_audit \
  --paths-config configs/local/paths.yaml \
  --output-root .local_artifacts/stage_c_source_quality_audit \
  --run-id 20260804T-source-quality-r2 \
  --device cuda:0 --query-batch-size 4096
```

只重新生成 HTML/截图（不重跑 SDF）：

```bash
conda run --no-capture-output -n spider-dex \
  python -m spider.tools.grab_stage_c_source_quality_audit \
  --render-existing .local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2
```

浏览器：

```bash
google-chrome --new-window \
  file:///home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/html/index.html
```

## 测试与 Git

新增合同测试：`tests/test_stage_c_source_quality_audit.py`。验证结果：指定 unittest `41 PASS`，全量 discovery `187 PASS`，compileall `PASS`，`git diff --check` `PASS`；明细见运行根目录 `reports/verification_summary.json`。不要 push，不要提交 ignored `.local_artifacts` 或 `configs/local/paths.yaml`。

当前基线 commit：`23959dbfb5d8e4a73f2814b736d80e1b8235f389`，branch `develop/wuji-hand2`。本任务未修改 raw GRAB/body models/Stage B/repaired C-XA 输入，未创建 commit、未 push。

## 下一步

若要把 C/D 纳入正式 authority，需要提供不改 mesh geometry 的、闭合且 winding-consistent 的 Wuji hand target contract，并重新生成与本报告同样的 canonical query、lineage、full-sequence metrics 和 screenshots。不能用 open-mesh heuristic、unsigned fallback、删帧或阈值放宽替代这一 gate。
