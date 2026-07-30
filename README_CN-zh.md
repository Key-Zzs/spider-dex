# SPIDER-Dex

SPIDER-Dex 是 [SPIDER](README_SPIDER.md) 的定向 fork，用于将 HOI/MANO
示范重定向到灵巧手；当前首先接入 Wuji Hand2 Beta1
(`wuji_hand2_beta1`)。

本仓库保留 SPIDER 的物理信息驱动重定向流程，并增加了自包含、可追溯的
Wuji 手模型适配层。仓库只保存代码、机器人描述、配置与测试，不保存原始
数据集或大规模生成轨迹。

[English](README.md) · [原始 SPIDER README](README_SPIDER.md)

## 当前状态

- S0：仓库审查与项目脚手架，已完成。
- S1：上游 SPIDER baseline，用户已报告完成；本次保留了有界回归检查。
- S2：Wuji Hand2 Beta1，自动化 embodiment 验证已完成；**人工可视化验收仍待用户完成**。
- Stage A：外部数据路径、workspace、canonical HOI 契约、registry、manifest
  与审计工具，自动验证已通过。
- Stage B：有界 GRAB 到 Wuji 运动学 pilot，自动流程已通过；追踪质量仍待人工复核。

当前没有声称已经完成 GRAB、OakInk、OakInk2 重定向、完整物理优化调参或
真机验证。

## 范围与非目标

当前阶段提供五指、20 个主动关节的 Wuji Hand2 Beta1，以及符合 SPIDER
控制语义的六自由度标量腕部。它支持无数据集的 MuJoCo 加载、运行时资产
staging、scene generation 和后续 IK 接入。

本阶段不复制或移动数据集，不运行全量处理或全量 physics optimization，也不
接入真机或导出真机指令。

## 快速开始

在已有的 `spider-dex` Conda 环境中运行：

```bash
conda run -n spider-dex python tools/validate_wuji_hand2.py
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side right --mode neutral
```

第一条命令会验证打包的右手、左手和双手模型；第二条命令打开 MuJoCo 用于
人工检查。验收前请阅读[验证与检查表](docs/project/VALIDATION.md)。

## 本地路径与数据政策

可把 `.env.example` 复制为被忽略的 `.env.local`，仅作为本地路径提示。公共
代码和 MJCF 只使用可移植的相对路径：

```bash
SPIDER_DATA_ROOT=/path/to/Ref2Dex_storage
SPIDER_MANO_ROOT=/path/to/shared_assets/body_models
WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
```

数据集和 processed 输出始终放在仓库外。不要提交 GRAB、OakInk、OakInk2、
MANO 模型、视频或大规模处理轨迹。

数据转换请把 `configs/project/paths.example.yaml` 复制为被忽略的
`configs/local/paths.yaml`。其中明确分离只读 dataset/body model root 与
可写的外部 workspace；详见[数据基础设施](docs/project/DATA_INFRASTRUCTURE.md)
和[canonical schema](docs/project/CANONICAL_HOI_SCHEMA.md)。

## 工作流

```text
外部 HOI 数据集
       | 可配置的外部路径
       v
数据集专用 adapter
       v
统一的 MANO/object trajectory
       v
SPIDER keypoint/contact preprocessing
       v
Wuji Hand2 Beta1 运动学 IK
       v
SPIDER physics-informed optimization
       +--> 可视化 / 指标 / 下游导出
```

S2 只完成图中的目标机器人 embodiment；数据集 adapter 从 S3 开始。详细契约
见 [WORKFLOW.md](docs/project/WORKFLOW.md)。

## 文档

- [项目索引](docs/project/index.md)
- [仓库审查](docs/project/REPOSITORY_AUDIT.md)
- [路线图](docs/project/ROADMAP.md)
- [Wuji Hand2 Beta1 适配](docs/project/WUJI_HAND2_BETA1.md)
- [资产溯源](docs/project/ASSET_PROVENANCE.md)
- [自动验证与人工验收](docs/project/VALIDATION.md)
- [Stage A/B 交接](docs/project/HANDOFF_STAGE_AB.md)

## 许可证、致谢与引用

根目录的 [LICENSE](LICENSE) 是 SPIDER 的 CC-BY-NC 许可证，整个 fork 不能被
错误描述为 MIT。复制的 Wuji Hand2 Beta1 资产子集使用其上游 MIT 许可证，见
`spider/assets/robots/wuji_hand2_beta1/LICENSE_WUJI`。边界与第三方说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；本仓库不声明任何数据集再分发权。

本 fork 致谢 Meta SPIDER 与 Wuji Technology description package。Wuji source
未提供论文 citation，因此只记录仓库、release 与资产溯源，而不虚构论文引用。
请按英文 README 中原样给出的 BibTeX 引用原始 SPIDER 工作。
