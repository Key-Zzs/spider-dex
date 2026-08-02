# Stage C-XAE-M1R3 接触真值与动力学审计

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

最终根因：`CONTACT_PAIR_CLASSIFICATION_ERROR`。exact pair 在 Step-5 切换，但同一冻结 left-index collision region 仍有合法 index_7-object 接触；patch gap 与 collision penetration 是不同定义域。
