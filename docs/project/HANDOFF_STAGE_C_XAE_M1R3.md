# Stage C-XAE-M1R3 中文交接

- Contract-V2: `PASS`
- Geometry: `PASS`
- M0: `PASS`
- M1: `NOT_RUN_PENDING_TWO_FRAME_REVIEW`
- M1 witness: `NOT_FOUND`
- M2/M3: `NOT_RUN`
- MJWP: `NOT_RUN`
- Oracle C/D2: `NOT_RUN`
- Stage D: `NOT_STARTED`
- Step-5: `PASS`
- Two-frame: `FAIL`
- full primary: `NOT_RUN`
- smokes: `NOT_RUN`
- user_visual_review: `PENDING`
- visualization: `PASS`
- 阶段1精确重放: `PASS`
- 阶段2接触真值: `PASS`
- 阶段3几何与参数: `PASS`
- 阶段4初始平衡: `PASS`
- 阶段5反事实实验: `COMPLETE`
- 阶段6局部响应辨识: `COMPLETE`
- 阶段7修复: `PASS`

运行根目录：`.local_artifacts/stage_c_xae_m1r3/20260802T143000Z-contact-truth-dynamics`

结论：M1R2 的 exact-pair gate 将 `index_8` 的标识符丢失误当作整段 left-index region 丢失。M1R3 原始 MuJoCo contact truth 在 Step-5 确认同一区域 `index_7|right_object_0` 仍在；同时 patch-site normal gap 与任意/区域 collision penetration 的定义域不同。修复只统一 gate 到冻结的 left-index collision region，保留 exact pair 为诊断，未修改物理模型、资产或动作路径。
