# 新 GRAB sequence source / Wuji recheck handoff

## 环境与不可变范围

- 仓库：`/home/deepcybo/workspace/dex/retarget/spider-dex`
- 分支 / 基线：`develop/wuji-hand2` / `5494a4d091e87253874eb70fc59cbc865d07aa4a`
- 环境：`conda run -n spider-dex ...`
- 本地配置：`configs/local/paths.yaml`
- 运行根：`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/`
- Git：没有 stage、commit 或 push；生成物均位于忽略的 `.local_artifacts/`。

未写入 raw GRAB、body models、既有 Stage B、既有 C-XA、历史 M1R3/M1R4 产物或权威配置。禁止范围 M1/M2/M3、full primary、Oracle C/D2、MJWP、smokes、Stage D 均未执行。

## 新序列与选择理由

选择 `s7/stanfordbunny_pass_1`：female subject `s7`、object `stanfordbunny`、`601` 帧、`120 Hz`、PCA-24。它是固定排序第 2 名，先排除了冻结 `s5/cylindermedium_lift` 和已有 Stage-B namespaces，并通过 asset/body 参数检查。真实 `GrabAdapter` retarget 前检查显示左手在 `30 mm` 阈值内连续接近 `319` 帧（首次 frame 130，门槛 60）。证据：`selection/selected_sequence.json`、`selection/candidate_sequences.json`。

## Source 表示、重建与加载结论

独立链路直接以 raw GRAB、`s7_betas.npy` 和 SMPL-X PCA-24 重建，`flat_hand_mean=False`；它不调用 adapter。Spider 链路实际调用 `GrabAdapter.load_sequence`，在任何 Stage A/B 之前捕获 source object、wrist、joints 和 hand surfaces。object/world/object-relative 比较数据在 `source_comparison/`，transform provenance 在 `source_official/official_transform_chain.json` 与 `source_spider/spider_transform_chain.json`。

结论是 **PASS** / `SOURCE_LOADER_PASS`：最大已检查残差 `1.195796e-07`，object、双腕、双手 object-frame fingertips 的 offset 都为 0。没有 loader 坐标错误、手物相对偏差、左右手交换、左手镜像或帧偏移证据。HTML 中的 Spider body 是明确披露的可视上下文，不是 public adapter return。

接触距离仅作为 source 诊断：watertight/winding-consistent bunny mesh、778 顶点的 KD-tree 近似距离、24 个关键帧的 exact triangle closest-point，以及完整 `601 × 双手 × 5` 指尖的小批量 signed-distance。不得把它误说成 contact success。

## 标准 Wuji 结果

标准图：`reports/STANDARD_PIPELINE_MAP.md`、`reports/standard_pipeline_map.json`。

```text
GrabAdapter source → grab_pipeline prepare (Stage A) → run-wuji-ik (Stage B)
```

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 输入 source 门 | `PASS` | `reports/SOURCE_LOADING_DECISION.md` |
| Stage A | `PASS` | `retarget/stage_a/{command,stdout,stderr}.txt` |
| Stage B | `PASS`，599 帧 / 120 Hz | `retarget/workspace/.../metrics_kinematic.json` |
| C-XA | `NOT_RUN` | 它不是当前标准 kinematic flow，未移植冻结 contact correction |
| 最终 | `RETARGET_PASS` | `reports/retarget_stage_status.json` |

Stage B 的 object-relative wrist RMSE L/R=`0.000094/0.000091 m`，tip RMSE L/R=`0.001516/0.001953 m`；object translation max=`0`，rotation RMSE=`4.245889e-17 rad`；failure frame=`null`。因此“首失败帧”视觉槽位展示首有效 frame 1，不能误称失败。

## HTML、截图与 Codex 实际观察

- `html/01_grab_loaded_source_audit.html`：独立 official vs Spider loaded source。
- `html/02_wuji_spider_retarget_audit.html`：Spider source vs Stage B Wuji visual/collision meshes。
- `html/index.html`：中文总入口；`handoff/HANDOFF.md` 为运行内短交接。
- 两个 HTML gzip 内嵌完整时间序列：source `601` 帧；retarget `599` 个实际 Stage-B mapping 帧。renderer v2 截图在 `screenshots/source_v2/` 和 `screenshots/retarget_v2/`，各 24 张；对应 manifest 和 Markdown review 位于 `screenshots/`。

Codex 实际查看了 source frame 130 的 world/object 视图、retarget frame 130 的 world/object 视图，以及 retarget frame 208 的 left-wrist 视图。观察与数值一致：source 叠加没有新增 loader 偏差；Stage B 显示在同一 object-relative 关系中，未见可见整体 wrist 漂移或左右映射错误。retarget 浏览器层对每个真实 MuJoCo geom 最多显示 96 个确定性抽样三角面，原始 Stage-B scene/qpos/metrics 未修改，详见 `reports/HTML_RENDERER_GEOMETRY.json`。未看到明显新增几何异常，但这不替代 signed collision/contact/dynamics 结论。

## 测试与后续

已通过：

```bash
conda run --no-capture-output -n spider-dex python -m unittest -v \
  tests.test_new_grab_sequence_recheck
# 13 tests: OK
```

已通过 `conda run --no-capture-output -n spider-dex python -m py_compile spider/tools/new_grab_sequence_recheck.py`；`git diff --check` 也通过。下一步若需要接触/动力学，应获得单独授权并以新的 fail-fast run 做最小范围工作；不要复用或改写本证据。
