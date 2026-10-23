# Stage Q0-Q4：GRAB Source Quality Screening 与新 Wuji 基线

本阶段只执行 Q0、Q1、Q2、Q3 和标准 Q4 Stage A/B。运行入口为：

```bash
conda run --no-capture-output -n spider-dex \
  python -m spider.tools.grab_stage_q0_q4_screening \
  --paths-config configs/local/paths.yaml \
  --output-root .local_artifacts/stage_q0_q4_grab_screening \
  --run-id <fresh-run-id> --device cuda:0
```

该入口先冻结且只冻结十条候选，再对这十条运行 Ref2Dex `STATIC_EXEC`
`MeshSDFfromAsset` reverse-SDF。它不扫描全 GRAB 的 SDF，不改 raw GRAB、body
models、Stage B 算法、历史 Stage-B/C-XA 产物，也不运行 C-XA、M0、Step-5、
Two-frame、M1/M2/M3、full primary、Oracle C/D2、MJWP、smokes、Stage D 或 V3。

Q0 的推荐质量 registry 是 `configs/project/grab_sequence_quality_registry.yaml`。
其中 `s5/cylindermedium_lift` 已标记为 `SOURCE_QUALITY_REJECTED`：它只能作负例
回归，不能作 primary、backup、source-role、semantic-patch、C-XA 或 dynamic
authority。

Q3 只允许 full SMPL-X hand-surface 的可靠近表面点进入 role 和 patch。深穿模是
`UNRELIABLE_SOURCE`，不能增加 responsibility，也不能扩大 patch。Q4 的
`STAGE_B_BASELINE_PASS` 只是 kinematic 结论；Wuji mesh 没有新的 watertight SDF
proxy 时，Stage-B penetration 一律保持 `SDF_INVALID_FOR_STAGE`。
