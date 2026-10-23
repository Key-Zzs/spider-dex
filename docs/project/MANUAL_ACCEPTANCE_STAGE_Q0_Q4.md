# Stage Q0-Q4 人工验收

运行完成后，先打开 `<run-root>/html/index.html`，并实际查看截图 manifests 所列 PNG。

Q3 primary/backup 都检查：可靠接触是否在物体表面；deep penetration 是否排除；
role 是否和 source 手指/掌部一致；shared role 是否合理；patch 是否覆盖 reliable
footprint 且不含无依据区域；role 边界是否合理；source interaction 是否足以进入
后续独立授权阶段。

Q4 primary/backup 都检查：object pose 是否保持；腕部是否整体漂移；左右映射是否正确；
Wuji 是否跟随 source；role 相关区域是否大致对应；是否有明显新穿模/悬空；patch 与
Wuji 区域关系是否合理；数值和视觉是否一致。

视觉检查不能把 Stage B kinematic 结论扩大成 C-XA、contact correction、dynamics 或
用户 acceptance；最终用户视觉验收状态仍应是 `PENDING`。
