# 外部 GPT 交接：GRAB 上游坐标审计

请在 `/home/deepcybo/workspace/dex/retarget/spider-dex` 工作，先阅读 `docs/project/STAGE_C_SOURCE_GEOMETRY_AUDIT.md`。这是一个冻结、只读定位任务：样本固定为 `s5/cylindermedium_lift`，source frames `[1460,1876)`，120 Hz；基线 commit 为 `8beeb6da3f442b2f531c50590c3202b0cbcc6353`。不要调用 V3，不要覆盖 source、Stage B、C-XA 或 `.local_artifacts` 的任意历史 run，也不要把 Stage B 的 66 维 free-joint qpos 与 C-XA 的 64 维 serial-XYZ qpos 直接逐元素相减。

已验证事实：

- 独立 full SMPL-X source reconstruction 与实际 `GrabAdapter.load_sequence` raw 输出 PASS。物体/腕/指尖最大误差约 `1e-8`–`1e-7`，每手 778 source→raw vertex 残差为 `0`，最佳时间 offset 是 0。上游 raw 坐标构造不是问题。
- Stage B PASS：物体不动；双腕 object-relative RMSE 小于 `0.25 mm`，双手指尖 RMSE 小于 `6.4 mm`。
- C-XA FAIL `CXA_GLOBAL_OFFSET_ERROR`：物体仍不动，但左右腕相对物体最大变化 `25.946/25.780 mm`，超过 15 mm 门限；最大指尖改动 `54.514/53.556 mm`。
- source 自身确实有穿透和分离；这是 source 记录属性，不能归责给 raw loader。冻结的全顶点有符号场与本次 raw 顶点—最近表面距离交叉验证到最大 `1.19e-7 m`。

唯一可写的新证据必须放到 `.local_artifacts/stage_c_source_geometry_audit/<新的唯一run-id>/`。若重跑审计，请使用：

```bash
conda run -n spider-dex python -m spider.tools.grab_source_geometry_audit \
  --paths-config configs/local/paths.yaml \
  --output-root .local_artifacts/stage_c_source_geometry_audit \
  --run-id <new-unique-id>
```

先看真实 HTML 和截图：

- `.local_artifacts/stage_c_source_geometry_audit/20260801T094100Z-source-geometry-r8/html/grab_source_geometry_trajectory_audit.html`
- `.local_artifacts/stage_c_source_geometry_audit/20260801T094100Z-source-geometry-r8/screenshots/SOURCE_GEOMETRY_SCREENSHOT_REVIEW.md`

下一步应限制在 C-XA correction 的约束、目标函数或接触语义诊断；先维持 object pose immutable，并在 MuJoCo forward 后比较 `T_object_wrist`/指尖，不得用静态对齐、hidden phase/attach shortcut 或改 source 来“通过”门禁。M1 的上游坐标解释现为不成立；完整动态接触根因仍是 PARTIAL，人工验收是 PENDING。
