# Stage C-SQA Source Quality Authority Audit

## 结论

主序列为 `s5/cylindermedium_lift`，源帧 `[1460,1876)`，120 Hz。四阶段都使用同一份 Ref2Dex STATIC_EXEC canonical object query：20,000 个 triangle-area random barycentric surface samples，1 mm floor-voxel first-unique 后得到 14,756 点；所有 stage 复用这些本地 query points。

| 阶段 | 帧数 | SDF authority | 结果 | 关键数字 |
|---|---:|---|---|---|
| A official / independent SMPL-X | 416 | formal | PASS | max depth 10.740 mm；P95 frame max 9.811 mm；mean frame point ratio 3.480% |
| B Spider loaded source | 416 | formal | PASS | 与 A 完全相同；object max abs residual 8.388e-8，hand vertex residual 0 |
| C standard Stage B Wuji | 414 | diagnostic only | `SDF_INVALID_FOR_STAGE` | heuristic max 7.415 mm，最深诊断帧 1827 |
| D repaired C-XA | 414 | diagnostic only | `SDF_INVALID_FOR_STAGE` | heuristic max 0.961 mm，最深诊断帧 1611 |

A/B 的正式结论是：原始 source 本身存在 signed object-query penetration；这不是 Spider raw loader 新增的坐标误差。C/D 的数值只作开放网格上的 diagnostic signed/unsigned 观测，不能作为正式 penetration authority，也不能据此宣称 repaired C-XA 已通过 source-quality authority。

## SDF 合同

实际调用 `/4-4090/Ref2Dex/dataset/tools/reverse_hand_meshsdf_statistics.py::query_targets_with_meshsdf`，backend 为 `MeshSDFfromAsset`，sign 为 `xyz_majority`；负值表示 object query point 位于 hand target 内部，penetration depth 为 `max(0,-signed_distance)`。正式 hand target 要求 watertight 且 winding-consistent。

官方/Spider SMPL-X 通过固定 16 点 wrist cap 后为 779 vertices / 1554 faces、单连通、watertight。Wuji group 2 collision layer 的完整手为 left 55,600 vertices / 111,444 faces / 121 components、right 56,815 / 113,886 / 129 components，watertight 均为 false；没有在本审计中修复、删帧、替换 mesh 或放宽阈值。

## 接触与 penetration

`near_surface_contact` 使用 `abs(signed)<=1 mm`，`shallow_penetration` 为 1–3 mm，`deep_penetration` 为大于 3 mm。接触近表面与 penetration 是两个字段；C/D 的签名结果保留在逐帧 JSON 中，但正式 authority 字段为 false。

Stage B/C-XA 轨迹只覆盖 source 1461..1874；source 1460 与 1875 在 frame mapping 中保留并标为 NOT_AVAILABLE，不是被删除的 source frame。

## 证据位置

本次运行根目录：
`.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2`

- `reports/ref2dex_static_exec_lineage.json`
- `reports/static_exec_implementation_map.json` 与 `reports/STATIC_EXEC_IMPLEMENTATION_MAP.md`
- `reports/hand_sdf_mesh_manifest.json`
- `reports/stage_asset_identity.json`
- `reports/reverse_sdf_metric_definition.json`
- `reports/stage_comparison_summary.json`
- `reports/a_official_reverse_sdf.json`、`b_spider_raw_reverse_sdf.json`、`c_stage_b_reverse_sdf.json`、`d_repaired_cxa_reverse_sdf.json`
- `manifest/frame_mapping.json`、`manifest/canonical_object_query_points.npz`

HTML：

- `.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/html/01_reverse_sdf_stage_comparison.html`
- `.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/html/02_source_quality_contact_vs_penetration.html`
- `.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/html/index.html`

Chrome 截图：`screenshots/stage_comparison` 与 `screenshots/source_quality` 各 24 张，两个 manifest 均为 PASS；代表性 world/object/hand 截图已人工检查。

## 控制与官方可视化

`s7/stanfordbunny_pass_1` 仅作为 control：prior isolated evidence 显示 source loader PASS、Stage A PASS、Stage B PASS（599 frames at 120 Hz），不替代本次 s5 结论；本次没有把它重新跑成主审计。

Ref2Dex 官方可视化命令来自真实 README/代码：

```bash
PYOPENGL_PLATFORM=egl conda run --no-capture-output -n ref2dex-grab \
  python /4-4090/Ref2Dex/dataset/GRAB/examples/render_grab.py \
  --grab-path /mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB/grab \
  --model-path /mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models \
  --render-path /home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/official_visualization/renderings
```

官方脚本真实生成 3,184 张 PNG；最后在 `OffscreenRenderer.is_active` cleanup 兼容性 bug 处 exit 1，详见 `reports/official_grab_visualization_command.json`。这些 PNG 是 ancillary evidence，不替代 s5 HTML。
