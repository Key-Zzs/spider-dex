# 新 GRAB source 与标准 Wuji recheck

## 范围与结论

本次只审计并重定向一条确定性选择的全新 GRAB 序列：`s7/stanfordbunny_pass_1`。执行根目录为：

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/`

Source 加载结论为 **PASS**；标准 Wuji kinematic 重定向结论为 **RETARGET_PASS**。这只证明该条序列的 raw source → Stage A → Stage B 路径，没有执行或证明 C-XA、接触修正、动力学、M1/M2/M3、full primary、Oracle C/D2、MJWP、smokes 或 Stage D。

## 确定性选择

| 项目 | 值 |
| --- | --- |
| subject | `s7`（female） |
| sequence | `s7/stanfordbunny_pass_1` |
| object | `stanfordbunny` |
| 帧数 / 频率 | `601` / `120 Hz` |
| 稳定排序 | 第 `2` 名，score `240.917` |
| 参数 | PCA-24，body/object 参数、subject beta 和 contact mesh 均存在 |
| 真实交互检查 | 真实 `GrabAdapter.load_sequence` 的 retarget 前左手指尖在 `<=30 mm` 内连续 `319` 帧，首次为 frame `130`；门槛为 `60` 帧 |

选择过程扫描所有 raw `grab/s*/*.npz`，先排除冻结主线 `s5/cylindermedium_lift`、既有 Wuji Stage-B namespace 和不完整资产，再按固定 score 降序、source id 升序排序。完整候选和逐条排除原因见 `selection/candidate_sequences.json`；最终理由见 `selection/selected_sequence.json`。

## source 链路与数值审计

独立路径直接读取同一 raw GRAB `.npz`，以 SMPL-X `use_pca=True`、`num_pca_comps=24`、`flat_hand_mean=False` 重建全身、双手和物体；它不调用 `GrabAdapter`。Spider 路径实际调用 `GrabAdapter.load_sequence`，并在 Stage A/B 前捕获 object、双腕、joints 和 hand surfaces。由于公共 `CanonicalHOISequence` 不提供全身 vertices，HTML 中 Spider body 仅是同一 raw body 参数的可视上下文，绝不作为 adapter 输出或比较依据。

| 检查 | 结果 |
| --- | --- |
| 判定 | `SOURCE_LOADER_PASS`，阈值 `1e-5` |
| 最大已检查残差 | `1.195796e-07` |
| object 平移 / 旋转最大残差 | `6.511608e-08 m` / `6.669121e-08 rad` |
| 左/右 `T_object_wrist` 平移最大残差 | `6.698726e-08 m` / `7.370679e-08 m` |
| 左/右 `T_object_wrist` 旋转最大残差 | `8.625275e-08 rad` / `1.195796e-07 rad` |
| offset 搜索 | object、双腕、双手 object-frame fingertips 均为 `0` |

因此，没有真实的 object-relative 手物偏差，也没有帧偏移、左右交换或左手镜像证据。`GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY` 是显示坐标判别字段，不是用来掩盖 relative check 的对齐操作：object-frame joints、tips、surface 均独立比较过。

`stanfordbunny.ply` 是 watertight 且 winding-consistent。全 `778` 个手 surface vertices 使用 object-vertex KD-tree 记录近似距离，24 个固定关键帧的五指尖另做 exact closest-triangle 记录；完整 `601 × 双手 × 5` 指尖另用小批量的真实 signed-distance 计算，正值按 trimesh 约定表示物体内部/穿透深度。右手远离物体及早/晚帧的大间隔是原始 motion 状态，而非 Spider loader 新增。该距离审计只描述几何关系，不能替代接触修正、动力学或用户 acceptance。

## 标准流程与实际运行

当前标准入口图在 `reports/STANDARD_PIPELINE_MAP.md` 和 `reports/standard_pipeline_map.json`：

```text
GRAB raw / GrabAdapter.load_sequence
→ Stage A: python -m spider.tools.grab_pipeline prepare
→ Stage B: python -m spider.tools.grab_pipeline run-wuji-ik
```

C-XA 是冻结的 Stage-C contact correction，不是当前标准新序列 kinematic CLI，故本次明确为 `NOT_RUN`，没有移植历史 C-XA。

实际命令与 stdout/stderr 分别保存在 `retarget/stage_a/`、`retarget/stage_b/`。它们使用隔离的 `retarget/workspace/`，不会复用或覆盖历史 Stage-B/C-XA：

```bash
python -m spider.tools.grab_pipeline prepare \
  --paths-config retarget/paths.generated.yaml \
  --sequence-id s7__stanfordbunny_pass_1 --frame-start 0 --frame-end 601
python -m spider.tools.grab_pipeline run-wuji-ik \
  --paths-config retarget/paths.generated.yaml \
  --sequence-id s7__stanfordbunny_pass_1 --no-save-video
```

| 阶段 | 状态 | 事实 |
| --- | --- | --- |
| 输入 source 门 | `PASS` | 使用上表 source 审计结果 |
| Stage A | `PASS` | return code `0` |
| Stage B | `PASS` | `599` 帧，`120 Hz`，trajectory SHA-256 `4643b3ea4f552d91e3bdbb495c85dcc3f9e8d1b07a5885b6f903414aca0f42f3` |
| C-XA | `NOT_RUN` | 非当前标准流程，未迁移冻结修正 |
| 最终 | `RETARGET_PASS` | 本次 source→标准 Stage-B kinematic 路径 |

Stage B 保持物体：平移最大残差 `0 m`，旋转 RMSE `4.245889e-17 rad`。object-relative wrist RMSE 左/右为 `0.0938/0.0907 mm`（最大 `0.1148/0.1013 mm`）；指尖 RMSE 左/右为 `1.516/1.953 mm`（最大 `3.284/2.724 mm`）。`nan_or_inf=false`、joint-limit violations=`0`、最大 qpos step=`0.323735`。没有 Stage-B 失败帧；原始 601 帧中首/尾各一帧受标准 mapping 裁剪，覆盖率为 `599/601 = 0.996672`，不是时间错位。

## HTML、截图与实际视觉复核

- [source 审计 HTML](/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/html/01_grab_loaded_source_audit.html)
- [Wuji retarget HTML](/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/html/02_wuji_spider_retarget_audit.html)
- [总索引](/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/html/index.html)

两个 HTML 都 gzip 内嵌完整序列：source `601` 帧，retarget 为 `599` 个实际 Stage-B mapping 帧；帧选择器、播放和关键帧跳转均作用于全帧，而非展示抽样。Chrome headless 为每个 HTML 输出 8 个关键帧 × 世界/物体/左腕三视角，共 `24 + 24` 张 PNG；两个最终 manifest 均为 `24/24 PASS`，并要求页面 ready、无未捕获 JavaScript 异常、且截图含已渲染场景。renderer v2 的截图目录为 `screenshots/source_v2/` 与 `screenshots/retarget_v2/`，source 截图强制 `overlay`，retarget 截图强制 `stageb`，真实 URL、对比模式、结果和路径在各自 manifest 中。为使 Plotly 在本机可交互，retarget 的浏览器 Wuji visual/collision 层对每个真实 MuJoCo geom 确定性抽样至最多 `96` 个三角面；原始 Stage-B scene、mesh、qpos 与 metrics 未改变，数量见 `reports/HTML_RENDERER_GEOMETRY.json`。实际图像复核记录在 `screenshots/SOURCE_HTML_SCREENSHOT_REVIEW.md` 和 `screenshots/RETARGET_HTML_SCREENSHOT_REVIEW.md`：source 的官方与 Spider 叠加无可见新增误差；Stage B 的 Wuji visual/collision 显示层、掌到指尖骨架与 source 同时显示，未见左右手映射反转或可见整体漂移。视觉审查不取代 signed collision/contact 或用户 acceptance。

## 验证与限制

在 `spider-dex` conda 环境中通过：`tests.test_new_grab_sequence_recheck` 的 `13` 项针对性回归，以及 `python -m py_compile spider/tools/new_grab_sequence_recheck.py`。另执行 `git diff --check`。生成物都在忽略的 `.local_artifacts/`；未修改 raw GRAB、body models、历史 Stage B/C-XA 或权威配置，且没有 push。

下一步若需要物理或接触结论，应从独立、明确授权的 C-XA/contact 或动态阶段开始；不得把本报告的 `RETARGET_PASS` 外推为该结论。
