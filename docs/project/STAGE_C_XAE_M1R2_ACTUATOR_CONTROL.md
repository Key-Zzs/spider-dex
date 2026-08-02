# Stage C-XAE-M1R2：执行器响应辨识与短时域保持控制

- Actuator identification: `PASS`
- Contract-V2: `PASS`
- Geometry: `PASS`
- M0: `PASS`
- M1: `NOT_RUN_DUE_TO_TWO_FRAME_GATE`
- M1 witness: `NOT_FOUND`
- M2/M3: `NOT_RUN`
- MJWP: `NOT_RUN`
- Model accuracy: `ACCURATE`
- Oracle C/D2: `NOT_RUN`
- Scheme-1 no-contact: `SCHEME1_NO_CONTACT_PASS`
- Scheme-1 real step-5: `FAIL`
- Scheme-2 MPC: `NOT_RUN_MODEL_ACCURATE`
- Stage D: `NOT_STARTED`
- Step-5: `FAIL`
- Two-frame: `NOT_RUN_DUE_TO_STEP5_GATE`
- full primary: `NOT_RUN`
- lineage: `PASS`
- smokes: `NOT_RUN`
- user_visual_review: `PENDING`
- visualization: `PASS`

## 执行器模型

- 模型：`STATE_SPACE`
- step-1 / step-5 / step-8 RMSE：`0` / `0.00935831` / `0.0125575`
- normal velocity normalized RMSE：`0.001333`，sign accuracy：`1.000`
- peak timing error：`0.538` step

no-contact 仅用于诊断，绝不是动态 witness。M2/M3、完整 primary、Oracle C/D2、MJWP、smokes 和 Stage D 未运行。
