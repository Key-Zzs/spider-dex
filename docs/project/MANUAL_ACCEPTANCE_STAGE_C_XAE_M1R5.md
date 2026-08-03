# Stage C-XAE-M1R5 人工验收

最终 run：

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final`

## 打开真实 3D 证据

```bash
google-chrome "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final/html/01_source_role_and_patch_truth.html"
google-chrome "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final/html/02_wuji_finger_only_feasibility.html"
google-chrome "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final/html/index.html"
```

或：

```bash
cd "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_xae_m1r5/20260803T054100Z-source-role-feasibility-visual-final"
python -m http.server 8000
```

打开 `http://127.0.0.1:8000/html/index.html`。

## 已由 Codex 实际查看 Chrome PNG 的结论

- [x] source 1461 的左食指真实接触，但左中指也同时承担接触；不是单指 SUPPORT。
- [x] 1461..1480 是 `LEFT_INDEX_SHARED`，旧 role interval 不能代表该共享真值。
- [x] 旧 patch 与真实 source footprint 有显著遗漏；修复 patch 的 triangles 来自 source contact，而非 Wuji 可达性。
- [x] Wuji 静态候选确有 assigned left-index collision-region 接触，且距 immutable patch 小于 20 mm。
- [x] 连续候选的 action 都在已审计 bounds 内。
- [x] 动态截图包含 Step 1、5、6、7；Step-5 保持，Step-7 首次丢失 assigned region。
- [x] 无高力/深穿透伪造：Two-frame 最大 force 1.4943 N、最大 penetration 0.7265 mm。
- [x] 动态状态转移均经 `FrozenActionEnvironment.step(action)`；无 root/wrist/object action 或 post-init qpos/qvel 写入。

## 用户待确认项

用户视觉验收仍为 `PENDING`。确认时请分别切换 source 的 world/object/left/right 视角，以及 Wuji 的 comparison/dynamic 模式和 world/object/wrist 视角；特别核对 Step 7 的 assigned-region contact 消失与页面数值一致。

## 截图覆盖

两个 HTML 均有 8 个事件 × 3 个视角，即 24 张 PNG：

- `screenshots/source_role/`：1460、1461、1462、1465、1480、旧 role 结束、最大 source penetration、末帧。
- `screenshots/wuji_feasibility/`：Stage B、repaired C-XA、两个静态候选、dynamic Step 1/5/6/7。

对应 manifest：`screenshots/SOURCE_ROLE_SCREENSHOT_MANIFEST.json` 与 `screenshots/WUJI_FEASIBILITY_SCREENSHOT_MANIFEST.json`。
