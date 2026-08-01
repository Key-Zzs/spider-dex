# Stage C-XAE-M1：Surface-Aligned Moving Contact Retention

权威 XAE 输入为 `.local_artifacts/stage_c_xae/20260801T141129Z-contact-alignment` 的最终
`repair/repaired_cxa_v2_final/trajectory_depenetrated_init.npz`。本阶段不会读取历史 M1
fixed-anchor target、seed、profile、标签或 C-M2 trace。

本次 run：`.local_artifacts/stage_c_xae_m1/20260801T162500Z-surface-aligned-retention`。

## 状态

```text
XAE: PASS
Contract-V2 regression: PASS
M0 regression: PASS
two-frame gate: FAIL
M1: FAIL
M1 moving retention witness: NOT_FOUND
M2: NOT_RUN
M3: NOT_RUN
full primary: NOT_RUN
Oracle C/D2: NOT_RUN
MJWP: NOT_RUN
smokes: NOT_RUN
Stage D: NOT_STARTED
用户视觉验收: PENDING
```

Contract-V2 重新计算 414 帧的九个门均通过：patch P95 `19.904335 mm`、coverage
`0.892458101`、functional recall `0.882352941`、normal median `0.999995859`、MuJoCo
collision penetration `2.796385 mm`。M0 以 `[left_index]` 和 actuator columns `36..39`
通过，continuity `1.0`、patch P95 `6.866426 mm`、force max `1.466972 N`。

## 动态 target 与控制边界

每个 0.5-ms substep 在当前 object actual pose 下，对 immutable connected semantic patch
做 nearest-surface 查询。它不是 fixed anchor、patch centroid 或单一 source point。source
translation 使用连续插值，object XYZ rotation 使用 Slerp，object 仅由现有 mocap/reference
接口驱动；初始化后没有 robot/object qpos 或 qvel 写入。

M1 只允许 `PRE_CONTACT → RETAIN_PENDING → RETAIN → COMPLETE`，失联直接到 `FAILED`；
`REGRASP` 被禁止。

## 结果与停止依据

R0 两帧门禁在 source `1461`、sim step/substep `5` 首次失联：指定
`left_index_8|right_object_0` 变为 `NONE`，normal gap `7.288531 mm`、relative normal
velocity `0.457675 m/s`、tangential slip `0.078134 m/s`。patch distance 仍为
`7.288531 mm`，因此主因是 `OBJECT_COUPLED_NORMAL_SEPARATION`，不是放宽 patch gate。

只运行了有因果依据的 R2 object-motion velocity feedforward 和 R3 normal-relative-velocity
servo；两者均在同一 first-loss substep 失败。force max `1.364645 N`、penetration max
`0.726697 mm`、joint margin `0.103097`、tracking/object tracking 均安全，不能以高力、深
穿透或 REGRASP 掩盖失败。双帧未过，故完整 1461..1466 M1 未运行。
