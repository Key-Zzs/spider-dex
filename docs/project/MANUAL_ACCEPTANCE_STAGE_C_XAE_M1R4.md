# Stage C-XAE-M1R4 人工视觉验收

- [x] Step-5：index_8→index_7 是合法 assigned-region pair 切换。
- [x] Step-6：index_7 region contact 仍存在。
- [x] Step-7：全部冻结 assigned-region geom 均离开 object；无未列入 region 的 left-index geom 接触。
- [x] normal gap 与正向 separation velocity 增加；force 不是持续 in-contact 衰减。
- [x] 最佳诊断 probe 也未保持 Step-7 region contact，未以深穿透或高力伪造。
- [x] root/wrist/object 未获动作权限，所有状态转移均经 environment.step。
- 截图：30/30 个实际 Chrome PNG。
- 用户视觉验收：`PENDING`。
