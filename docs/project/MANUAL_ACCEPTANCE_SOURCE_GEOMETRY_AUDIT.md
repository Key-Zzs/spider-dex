# Stage C Source Geometry Audit 人工验收

工程侧审计状态：**完成**。用户接触语义验收状态：**PENDING**。

打开 `.local_artifacts/stage_c_source_geometry_audit/20260801T094100Z-source-geometry-r8/html/grab_source_geometry_visual_index.html`，按以下顺序检查：

1. 选择“官方 source vs spider raw”、物体坐标，依次检查 1460、1461、1462、1463、1465、1480。
2. 切到世界坐标复查 1461、1465、1480；确认 raw/source 的重合不依赖单一观看视角。
3. 选择“spider raw vs Stage B”，查看 1631 和 1803 的最大已测误差帧；确认是局部跟踪误差，不是整手漂移。
4. 选择“Stage B vs C-XA”，查看 1461、1465、1798、1816；保持 error 与 semantic patch 图层开启，确认 C-XA 改动在物体坐标仍存在。
5. 复核 `screenshots/SOURCE_GEOMETRY_SCREENSHOT_REVIEW.md`，其中的 26 张 `valid_*.png` 为 Chrome 真正加载 HTML 的证据；没有 `valid_` 前缀的首批文件为相对 URL 失败记录，不能使用。

验收问题：

- [ ] 是否接受：GRAB source 到 spider raw 的坐标链路正确，不能再作为 C-XA 故障根因？
- [ ] 是否接受：source 的原生穿透/悬空记录应与 raw loader 责任分离？
- [ ] 是否接受：Stage B 保持原有 tracking 质量，C-XA 的 >15 mm object-relative correction 是当前失败点？
- [ ] 是否允许下一阶段只针对 C-XA correction/接触约束排查，而不改动 frozen source、Stage B 或该审计目录？

未得到明确人工接受前，不得把本报告升级为“接触语义已验收”或“动态接触问题已修复”。
