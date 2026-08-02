# Stage C-XAE-M1R2 中文交接

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

- run root: `/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r2/20260802T125000Z-actuator-response-identification`
- 第一失败：`{'sim_step': 5, 'source_frame': 1461, 'normal_gap_m': 0.007286688889570437, 'relative_normal_velocity_mps': 0.4564462742623059, 'tangential_slip_mps': 0.07802777575644051, 'actual_geom_pair': 'NONE'}`
- 继续工作前必须保持同一冻结输入和 Contract-V2 → geometry → M0 → step-5 → two-frame → M1 顺序。
