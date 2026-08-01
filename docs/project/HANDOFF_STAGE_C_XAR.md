# Stage C-XAR handoff（受限）

## 结论

已定位并最小修复 C-XA wrist/root correction leakage。根因是 finger-only bounds 的条件顺序错误，使历史 Phase-1 实际允许 root/wrist correction。修复后 raw loader、Stage B、修复 C-XA source-relative geometry 均 PASS，且 object / wrist-root 锁定不变量为零差异。

本轮未取得完全 C-XA 接受：冻结 Contract-V2 的 `surface_patch_distance_p95=0.021582209868742812 m`，大于 `0.02 m`。一次 `contact=10000` 的受控尝试变为 `0.022802893412041274 m`，所以已停止调参。M0、两帧、M1 均为 `NOT_RUN_DUE_TO_UPSTREAM_CONTACT_GATE`。

## 关键产物

- XAR 根目录：`.local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair/`
- 根因：`reports/cxa_root_cause_decision.json`
- 修复后比较/最终门禁：`reports/old_vs_repaired_cxa.json`、`reports/xar_final_acceptance.json`
- 锁定与 A–E 回归：`repair/cxa_repair_preservation_audit.json`、`repair/postrepair_A_to_E_regression.json`
- source 几何 PASS：`source_geometry_audit/20260801T102900Z-repaired-cxa-geometry/reports/source_geometry_final_decision.json`
- 30 张真实网格图：`screenshots/XAR_SCREENSHOT_MANIFEST.json`

## 已修改的源码和配置

- `configs/project/grab_wuji_depenetration.yaml`
- `configs/project/grab_wuji_depenetration_cxa_repaired.yaml`（只作为受控失败 probe 的配置）
- `spider/tools/grab_stage_c.py`
- `spider/tools/grab_source_geometry_audit.py`
- `spider/tools/grab_stage_c_xar.py`
- `spider/tools/grab_stage_c_xar_viewer.py`
- `spider/tools/grab_stage_c_xar_static_screenshots.py`
- `spider/tools/grab_stage_c_xar_postaudit.py`
- `tests/test_stage_c_xar.py`

## 可复核命令

```bash
conda run --no-capture-output -n spider-dex python -m unittest tests.test_stage_c_xar tests.test_grab_source_geometry_audit -v
conda run --no-capture-output -n spider-dex python -m spider.tools.grab_stage_c_xar_postaudit \
  --run-root .local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair \
  --paths-config configs/local/paths.yaml \
  --repaired-root /mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa_repaired_20260801T101500Z \
  --contact-probe-root /mnt/nas/storage/Ref2Dex_storage/spider_workspace/processed/grab/wuji_hand2_beta1/bimanual/s5__cylindermedium_lift/0/stage_c_contract_v2_cxa_repaired_contact_20260801T101500Z
```

## 不得继续的事项

不得通过解除 wrist/object 锁、修改冻结阈值、改 raw/Stage B/历史 C-XA 或把 P95 FAIL 标成 PASS 来运行 M0、两帧或 M1；同样不得运行 M2/M3/full primary/Oracle C-D2/MJWP/smoke/Stage D。下一步必须先获得明确授权，且应从 contact P95 失败的可解释根因开始，而不是继续盲调。

