# Stage C 上游 source 几何与坐标链路审计

## 冻结范围与结论

- 样本：`GRAB s5/cylindermedium_lift`，女性 subject `s5`，120 Hz。
- source 时间窗：`[1460, 1876)`；重点帧为 1460–1466、1480；详细可视轨迹为 1460–1480。
- 审计基线提交：`8beeb6da3f442b2f531c50590c3202b0cbcc6353`。
- 完整运行：`.local_artifacts/stage_c_source_geometry_audit/20260801T094100Z-source-geometry-r8/`。
- 最终分类：`CXA_CORRECTION_ERROR`。

明确门禁结论如下：

| 门禁 | 结果 | 含义 |
| --- | --- | --- |
| 官方 source → spider `GrabAdapter` 原始输出 | **PASS** | 独立 SMPL-X reconstruction 与实际 raw loader 数值级一致；不是上游坐标构造错误。 |
| raw → Stage B | **PASS** | Stage B 保持物体位姿，手腕/指尖误差属于有限的 retarget 跟踪误差。 |
| Stage B → C-XA | **FAIL: `CXA_GLOBAL_OFFSET_ERROR`** | C-XA 保持物体，但对手在物体坐标中的相对位姿施加了超过门限的整体改动。 |

因此：GRAB 原始参数在本范围内的解码、SMPL-X 复原和 raw 坐标链路为 **正确**；不应把 C-XA 的问题归因于 GRAB source 或 `GrabAdapter.load_sequence`。这不等价于“GRAB 的每个记录接触都物理无穿透”：source 自身有可观测的全顶点穿透/分离，见后文。

## 方法与链路边界

1. 直接读取冻结 GRAB `.npz`，使用 full-body SMPL-X 参数独立构造世界坐标手、身体、物体；此步骤不调用 `GrabAdapter`。
2. 再单独调用真实 `spider.datasets.grab.GrabAdapter.load_sequence` 捕获 retarget 前 raw 输出。
3. 在世界、物体、左右腕坐标中比较 `T_world_object`、`T_world_wrist`、21 joints、5 fingertips 和每手 778 个 surface vertices；同时检查单位、旋转转置、四元数顺序、root double apply、镜像、左右交换与 ±2 帧偏移。
4. 仅在 raw PASS 后读取 Stage B；仅在 Stage B PASS 后读取 C-XA。Stage B free-joint `xyz+wxyz` 和 C-XA serial intrinsic `XYZ` chart 不直接数组相减，而是一律经过 `mujoco.mj_forward` 后比较真实物理位姿。

不做的事：不调用 V3、不优化/修复轨迹、不覆盖 frozen Stage B/C-XA、不给 raw source 加 canonical 对齐，也不把 semantic patch 当成坐标正确性的证据。

## 原始 source / raw 数值结果

`reports/raw_loader_geometry_comparison.json` 的 raw 判定为 `EXACT_OR_NUMERICAL_MATCH`：

- 物体平移/旋转最大残差：`3.054e-08 m` / `8.400e-08 rad`。
- 左/右 `T_object_wrist` 平移最大残差：`3.608e-08 m` / `3.368e-08 m`。
- 左/右 `T_object_wrist` 旋转最大残差：`1.038e-07 rad` / `1.017e-07 rad`。
- 指尖相对物体最大残差：`3.426e-08 m`；每手 778 surface vertex 的 source→raw 最大残差均为 `0`。
- 时序搜索的最佳 offset 对物体、双腕、双手指尖均为 `0`；例如右手指尖偏移 ±1 帧的 RMSE 为约 `3.41 mm`，零偏移约 `1.02e-08 m`。

全局视图测试标为 `GLOBAL_VIEW_FRAME_DIFFERENCE_ONLY`，但这是恒等全局变换这个特例；并非只在“看起来的世界方向”一致，物体坐标相对姿态也数值级一致。

## source 全表面接触证据

`cylindermedium.ply` 是 watertight、winding-consistent 的真实 contact mesh（20,002 顶点、40,000 三角形，SHA-256 `ddee...bee6`）。签名距离来自冻结的 source 全手表面诊断：每帧、每手 778 顶点；它被本运行严格交叉验证，而不参与 raw PASS 判定。

- 诊断记录 `raw_grab_modified=false`、`canonical_overwritten=false`。
- 诊断帧号与 1460–1875 完全一致，网格 checksum 一致。
- 本次独立 source→raw surface vertex 最大残差：左右均为 `0 m`。
- raw vertex 到冻结最近表面无符号距离残差最大值：左 `3.35e-08 m`、右 `1.19e-07 m`。
- source 自身最大穿透：左 `37.854 mm`（frame 1803）、右 `37.802 mm`（frame 1857）。
- source 自身最大外部分离：左 `156.590 mm`（frame 1623）、右 `647.465 mm`（frame 1477）。

这些 source 接触现象不由 raw loader 新增；它们是 source 记录的属性，故不能用来指控上游坐标链路。

## Stage B / C-XA 实测结果

Stage B 比较范围为 source frame 1461–1874。1460 和 1875 缺失是 `source_mapping.json` 明示的 `ik_fast` 首帧导数和末帧裁剪，不是 temporal error。

- Stage B 物体：平移最大 `0 m`，旋转最大 `2.01e-16 rad`。
- Stage B 左/右腕 object-relative RMSE：`0.244 mm` / `0.180 mm`；最大 `0.338 mm` / `0.275 mm`。
- Stage B 左/右指尖 object-relative RMSE：`6.334 mm` / `5.625 mm`；最大 `16.117 mm` / `15.654 mm`，门限 `80 mm` 内。

C-XA 物体仍严格不动（平移 `0 m`、旋转最大 `2.49e-15 rad`），但手发生了不可接受的 object-relative 改动：

- 左/右腕最大位移：`25.946 mm` / `25.780 mm`，超过 `15 mm` 门限。
- 左/右腕最大旋转：`0.3743 rad` / `0.2941 rad`。
- 左/右最大帧间 correction change：`29.016 mm` / `24.339 mm`，超过 `15 mm` 连续性门限。
- 左/右指尖最大改动：`54.514 mm` / `53.556 mm`。

这支持“C-XA correction 是当前污染点”。它将 M1 的“上游坐标链路错误”解释降为 **不成立**；对于更广义的动态接触成因，结论仍是 **PARTIAL**，不能借此宣称已证明完整动力学根因。

## 产物与复现

- 最终判定：`reports/source_geometry_final_decision.json`。
- source/raw 误差：`reports/raw_loader_geometry_comparison.json`、`reports/raw_loader_temporal_alignment.json`。
- 全手接触：`reports/source_hand_surface_evidence.json`、`official_source/source_hand_signed_surface_evidence.npz`。
- Stage B/C-XA：`stage_b/stage_b_relative_geometry_audit.json`、`cxa/cxa_relative_geometry_audit.json`。
- 自包含交互 HTML：`html/grab_source_geometry_trajectory_audit.html`。
- 26 张有效 Chrome 截图和复核：`screenshots/source_geometry_screenshot_manifest.json`、`screenshots/SOURCE_GEOMETRY_SCREENSHOT_REVIEW.md`。

从仓库根目录复现（不可复用已有 run id）：

```bash
conda run -n spider-dex python -m spider.tools.grab_source_geometry_audit \
  --paths-config configs/local/paths.yaml \
  --output-root .local_artifacts/stage_c_source_geometry_audit \
  --run-id <新的唯一run-id>
```

输出根目录存在时工具 fail-fast，绝不覆盖历史审计产物。
