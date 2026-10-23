# Stage Q0-Q4 中文 handoff

已完成的基线运行位于
`.local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1`。
这次运行冻结十条候选并且只对这十条执行全帧 Ref2Dex `STATIC_EXEC`
reverse-SDF。Q0 仍将 `s5/cylindermedium_lift` 固定为
`SOURCE_QUALITY_REJECTED`；它只能是负例回归，不是任何 source-role、patch、
C-XA 或动力学 authority。

Q2/Q3 后的新基线是：

- primary：`s7/stanfordbunny_pass_1`；Q3 reliable interval 70 帧，146 个
  role-bearing frame，1,574 个未膨胀 patch triangle。
- backup：`s10/spherelarge_pass_1`；Q3 reliable interval 159 帧，271 个
  role-bearing frame，2,178 个未膨胀 patch triangle。

二者均为 Q3 `PASS`，deep penetration 从 role/patch 中严格排除；两者 Q4
Stage A/B 均 `PASS`。primary 的 object translation max 为 `6.512e-08 m`、
rotation RMSE 为 `1.396e-08 rad`；backup 分别为 `6.176e-08 m` 和
`2.229e-08 rad`。二者均无 NaN/Inf 和 joint-limit violation。这个结论仅是
`STAGE_B_BASELINE_PASS`：Wuji 没有额外 watertight SDF authority，因此
Stage-B penetration 明确为 `SDF_INVALID_FOR_STAGE`。

复核入口依次是：

1. `html/index.html`（六个自包含页面）；
2. `reports/ten_candidate_quality_ranking.json` 与
   `reports/q0_q4_final_acceptance.json`；
3. `q1_candidates/Q1_VISUALIZATION_MANIFEST.json`（十条真实 query-set 的
   worst-frame reverse-SDF 可视化留证）；
4. `q2_selection/official_grab_visualization/primary/` 和 `backup/`
   （官方 GRAB renderer 的精确 pass 序列 PNG）；
5. `screenshots/*/manifest.json` 以及 `handoff/HANDOFF.md`。

如需只重建展示证据而不重跑 Q1 全帧统计或 Stage A/B：

```bash
conda run --no-capture-output -n spider-dex \
  python -m spider.tools.grab_stage_q0_q4_screening \
  --paths-config configs/local/paths.yaml \
  --refresh-q1-visuals .local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1 \
  --device cuda:0

conda run --no-capture-output -n spider-dex \
  python -m spider.tools.grab_stage_q0_q4_screening \
  --paths-config configs/local/paths.yaml \
  --refresh-authority-q4-visuals .local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1
```

官方示例的 CLI 只随机匹配 `*eat*.npz`，因此不能精确渲染这两条 `pass` 序列。
下列命令直接调用该示例的公开 `vis_sequence`，仍使用其原生 SMPL-X、object、table
及 contact-coloring 实现；它们只写入隔离 run root。离屏 renderer 用 `delete()`
释放，而不用该示例与交互 viewer 不兼容的 `close_viewer()`：

```bash
PYOPENGL_PLATFORM=egl conda run --no-capture-output -n ref2dex-grab python -c "from examples import render_grab as r; r.grab_path='/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB/grab'; cfg=r.Config(**{'grab_path':r.grab_path,'model_path':'/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models','render_path':'$PWD/.local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1/q2_selection/official_grab_visualization/primary'}); mv=r.MeshViewer(width=1600,height=1200,offscreen=True); r.vis_sequence(cfg,r.grab_path+'/s7/stanfordbunny_pass_1.npz',mv); mv.viewer.delete()"

PYOPENGL_PLATFORM=egl conda run --no-capture-output -n ref2dex-grab python -c "from examples import render_grab as r; r.grab_path='/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB/grab'; cfg=r.Config(**{'grab_path':r.grab_path,'model_path':'/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models','render_path':'$PWD/.local_artifacts/stage_q0_q4_grab_screening/20260804T-q0-q4-r1/q2_selection/official_grab_visualization/backup'}); mv=r.MeshViewer(width=1600,height=1200,offscreen=True); r.vis_sequence(cfg,r.grab_path+'/s10/spherelarge_pass_1.npz',mv); mv.viewer.delete()"
```

不得把本 handoff 扩展成 C-XA、接触修正、动力学、M0/Step-5 或用户最终视觉验收；这些
保持 `NOT_RUN`/`PENDING`，直到有单独授权。
