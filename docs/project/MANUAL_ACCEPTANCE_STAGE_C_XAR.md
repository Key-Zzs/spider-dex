# Stage C-XAR 人工几何验收

## 当前状态

人工验收为 **PENDING**。本文件只提供真实 mesh 证据与检查准则；不得将截图或数值 PASS 自动替代人工接受。

## 证据入口

- 交互页：`.local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair/html/stage_c_xar_correction_audit.html`
- 索引页：`.local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair/html/stage_c_xar_visual_index.html`
- 30 张离线真实网格截图：`.local_artifacts/stage_c_xar/20260801T101500Z-cxa-repair/screenshots/`
- 截图清单：`screenshots/XAR_SCREENSHOT_MANIFEST.json`

截图覆盖 source frames `1461, 1462, 1463, 1464, 1465, 1466, 1480, 1619, 1798, 1874`；每帧均有：

- `world_visual`：世界坐标、真实 visual mesh；
- `object_collision`：物体坐标、真实 collision mesh；
- `left_visual_collision`：左腕坐标、真实 visual + collision 对照。

黄为 Stage B，红为旧 C-XA，绿为修复 C-XA，蓝为物体，紫为 semantic patch。所有手/物体顶点均由 `scene_act.xml` 的 MuJoCo mesh 在记录 qpos 上 `mj_forward` 后得到；为使软件渲染可用，显示层只对真实三角面做有界抽样，未进行几何对齐或 qpos 重写。

## 应检查什么

1. 在 1461、1619、1798 对比红色旧 C-XA 与黄/绿色：旧轨迹应可见 wrist/root 漂移，修复轨迹的腕根应与 Stage B 重合。
2. 在 object 坐标下确认物体没有位姿变化；在 left-wrist 坐标下确认修复不是通过整体手腕移动取得。
3. visual 与 collision 图都应从同一真实 `scene_act` mesh 生成，不能把 visual duplicate 当作 collision 证据。
4. 指尖允许变化；不能把允许的 finger articulation 误判为 wrist leakage。
5. 不要据此宣告 contact 成功：Contract-V2 P95 仍为 21.582 mm，超过 20 mm 门槛。

## 人工结论记录

- [ ] 接受修复候选的 wrist/root 几何保持。
- [ ] 拒绝：注明 source frame、坐标系、颜色层与可复现原因。
- [ ] 接触质量仍为 `FAIL`，不授予 M0/双帧/M1 运行资格。

