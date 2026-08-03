# 新 GRAB sequence recheck：人工验收说明

## 当前机器证据

本说明对应 `s7/stanfordbunny_pass_1`，运行根目录为：

`/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_new_sequence_recheck/20260802T174856Z-new-grab-wuji-recheck-r5/`

机器判定为 source `PASS`、Stage A `PASS`、Stage B `PASS`、C-XA `NOT_RUN`、最终 `RETARGET_PASS`。它仍不是用户验收，尤其不证明接触、穿透、安全、动力学或 C-XA。

## 打开方式

从该运行根目录用浏览器打开以下本地 HTML，而非截图替身：

1. `html/01_grab_loaded_source_audit.html`
2. `html/02_wuji_spider_retarget_audit.html`
3. `html/index.html` 可作为总入口。

两个页面均 gzip 内嵌完整时间序列，包含全帧选择、播放、关键帧跳转、world/object/left wrist/right wrist 坐标、显示层、轨迹和最近表面连线开关。source 为 `601` 帧；retarget 为 `599` 个实际 Stage-B mapping 帧。截图索引为 `screenshots/SOURCE_HTML_SCREENSHOT_MANIFEST.json` 和 `screenshots/RETARGET_HTML_SCREENSHOT_MANIFEST.json`；renderer v2 的 24+24 张 PNG 落在 `screenshots/source_v2/` 和 `screenshots/retarget_v2/`。

## Source 人工检查

在第一个页面选择“二者叠加”，至少检查 frame 0、129、130、131、187、600 及页面提供的最大差异/悬空关键帧：

- world 与 object 坐标均检查 object、左右 wrist、左右手 skeleton/surface 是否同位。
- 确认 frame 130 左手接近 `stanfordbunny` 时，官方 source 与 Spider loaded source 没有相对滑移、镜像或交换。
- 检查 object mesh 原点与最近表面连线是否一致；不应依赖人为平移/旋转来取得叠加。
- 将结果与数值核对：最大检查残差 `1.195796e-07`，所有 temporal best offset 为 `0`。

已由 Codex 实际查看 frame 130 的 world/object 视图，观察到官方/Spider 叠加一致、无可见左右互换或新增 loader 误差。原始右手远离物体和非接触帧的悬空不是 loader 错误；完整有符号指尖距离只是一项几何诊断，不能提升为接触或用户 acceptance。

## Wuji 人工检查

在第二个页面选择“Source vs Stage B”，至少检查 frame 1、130、131、187、599：

- 在 object 坐标近景确认 Stage B visual mesh 与 collision mesh 都出现，并在同一物体关系下包围/跟随对应 source 手。
- 确认没有整体 wrist 漂移或左右手交叉映射；对照 object-relative wrist RMSE 左/右 `0.0938/0.0907 mm`。
- 确认 object 位姿未漂移（平移最大 `0 m`，旋转 RMSE `4.245889e-17 rad`）。
- 对照 `reports/retarget_metrics.json`：指尖 RMSE 左/右 `1.516/1.953 mm`，最大 `3.284/2.724 mm`，没有 `failure_frame`。

已由 Codex 实际查看 frame 130 的 object 近景、frame 130 的 world 总览和 frame 208 的左腕视图；可见 Stage B Wuji mesh 与 source 同时出现，未看到明显左右映射错误或整体漂移。浏览器层的 Wuji mesh 是每个真实 MuJoCo geom 最多 96 个三角面的确定性显示抽样，原始 Stage-B 数据未改动。视觉上未见明显新增悬空/穿模，但没有运行完整 signed collision、接触修正或动力学测试，不能将“未见”升级为安全或接触 PASS。

## 应记录的人工决定

人工验收应单独记录以下结果，不要复写或修改本次原始证据：

| 项目 | 建议记录 |
| --- | --- |
| source 坐标/时间/左右手 | ACCEPT / REJECT，附帧号与截图 |
| Stage B 几何跟随 | ACCEPT / REJECT，附帧号与截图 |
| object 位姿保持 | ACCEPT / REJECT，附帧号与截图 |
| 可见接触/穿模疑点 | INCONCLUSIVE，除非另有 signed collision/contact 证据 |
| C-XA、动力学、真实机器人 | NOT_RUN，不得以本次结果替代 |

若发现异常，保留 screenshot、frame 号、坐标模式和 comparison 模式，并创建新的隔离 run；不要覆盖本次 run、raw GRAB 或历史 Stage-B/C-XA。
