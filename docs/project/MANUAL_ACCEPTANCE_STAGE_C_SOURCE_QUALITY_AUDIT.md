# Stage C-SQA 人工验收

运行根目录：`.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2`

## Screenshot acceptance

- [x] `screenshots/stage_comparison/manifest.json`：24/24 Chrome PNG，returncode 0
- [x] `screenshots/source_quality/manifest.json`：24/24 Chrome PNG，returncode 0
- [x] world view：能看到 object、左右 hand layers、canonical query cloud
- [x] object view：能检查 object-centered hand/object 相对关系
- [x] hand view：object-centered close-up 能看到 source/Wuji hand surface，不是空白 canvas
- [x] 中文标题、frame/view/stage controls 和 JSON metrics panel 可见
- [x] C/D 页面明确标注 diagnostic signed，不把开放网格结果写成 formal authority
- [x] 代表帧 1460、1461、1465、1480 已检查；截图 review 文件已更新为 PASS

## 数值验收边界

- [x] A/B formal signed source baseline 完成 416/416
- [x] C/D diagnostic full crop 完成 414/414，未 silent skip
- [x] 统一 query cache、签名 convention、wrist cap 和 frame mapping 已写入 manifest
- [x] contact / near-surface 与 penetration depth 分字段报告
- [x] non-watertight Wuji target 被 fail-closed 为 `SDF_INVALID_FOR_STAGE`
- [x] raw GRAB、body models、Stage B、repaired C-XA 输入保持只读

人工复核文件：

- `.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/screenshots/stage_comparison/REVIEW.md`
- `.local_artifacts/stage_c_source_quality_audit/20260804T-source-quality-r2/screenshots/source_quality/REVIEW.md`
