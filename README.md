# SPIDER-Dex

SPIDER-Dex is a focused fork of [SPIDER](README_SPIDER.md) for retargeting
human-object interaction (HOI) and MANO demonstrations to robot hands, starting
with the Wuji Hand2 Beta1 embodiment (`wuji_hand2_beta1`).

It preserves SPIDER's physics-informed retargeting pipeline while adding a
versioned, self-contained Wuji hand adapter. This repository contains code,
robot descriptions, configuration, and tests—not raw datasets or generated
trajectory collections.

[中文说明](README_CN-zh.md) · [Upstream SPIDER README](README_SPIDER.md)

## Status

- S0 — repository audit and project scaffolding: complete in this fork.
- S1 — upstream SPIDER baseline: reported previously completed by the user;
  this change retains a bounded regression check.
- S2 — Wuji Hand2 Beta1 integration: automated embodiment validation complete;
  **manual visual acceptance remains pending**.
- S3–S8 — dataset adapters, retargeting experiments, evaluation, and export:
  not started.

No claim is made here that GRAB, OakInk, OakInk2, full physics optimization, or
real-hardware validation has been completed.

## Scope and non-goals

The current target is a five-finger, 20-actuator Wuji Hand2 Beta1 model with
SPIDER-compatible six-DoF scalar wrist controls. It supports data-free MuJoCo
loading, runtime asset staging, scene generation, and later IK integration.

This stage deliberately does not add a GRAB/OakInk/OakInk2 adapter, copy or move
datasets, process full datasets, tune full physics optimization, communicate with
hardware, or export hardware joint commands.

## Repository overview

```text
spider/                         SPIDER package and packaged robot assets
spider/assets/robots/wuji_hand2_beta1/
                                Wuji vendor subset and SPIDER adapters
tools/sync_wuji_hand2_assets.py reproducible asset import/adapter generator
tools/validate_wuji_hand2.py    static, MuJoCo, hold, and staging validator
examples/inspect_wuji_hand2.py  visual/manual acceptance helper
docs/project/                   fork-specific audit, workflow, and records
```

## Quick start

Create the upstream environment as described by SPIDER, then use the existing
`spider-dex` Conda environment:

```bash
conda run -n spider-dex python tools/validate_wuji_hand2.py
conda run -n spider-dex python examples/inspect_wuji_hand2.py --side right --mode neutral
```

The first command is non-interactive and verifies the packaged right, left, and
bimanual models. The second command opens MuJoCo for human inspection. See the
[validation checklist](docs/project/VALIDATION.md) before accepting a model.

## Local paths and data policy

Copy `.env.example` to an ignored `.env.local` if a local workflow needs path
reminders. Public code and MJCF use relative paths; no command depends on a
developer-specific location.

```bash
SPIDER_DATA_ROOT=/path/to/Ref2Dex_storage
SPIDER_MANO_ROOT=/path/to/shared_assets/body_models
WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
```

Datasets stay outside this repository. Processed scenes and trajectories should
also be placed under an external dataset/output root. Do not commit raw GRAB,
OakInk, OakInk2, MANO model files, videos, or large processed trajectories.

## Workflow

```text
External HOI datasets
        | configurable external path
        v
Dataset-specific adapter
        v
Unified MANO/object trajectory
        v
SPIDER keypoint/contact preprocessing
        v
Wuji Hand2 Beta1 kinematic IK
        v
SPIDER physics-informed optimization
        +--> visualization / metrics / downstream export
```

S2 supplies only the target embodiment in this diagram. Dataset-specific adapter
work begins at S3. The detailed contract is in [WORKFLOW.md](docs/project/WORKFLOW.md).

## Roadmap and documentation

- [Project index](docs/project/index.md)
- [Repository audit](docs/project/REPOSITORY_AUDIT.md)
- [Roadmap](docs/project/ROADMAP.md)
- [Wuji Hand2 Beta1 adapter](docs/project/WUJI_HAND2_BETA1.md)
- [Asset provenance](docs/project/ASSET_PROVENANCE.md)
- [Validation and manual acceptance](docs/project/VALIDATION.md)

## License and third-party assets

SPIDER-Dex retains SPIDER's root [CC-BY-NC license](LICENSE); this does not make
the repository MIT-licensed. The copied Wuji Hand2 Beta1 asset subset is under
the upstream MIT license in `spider/assets/robots/wuji_hand2_beta1/LICENSE_WUJI`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the boundary between
those licenses. No dataset redistribution rights are asserted.

## Acknowledgments and citation

This fork builds on Meta's SPIDER and the Wuji Technology description package.
Wuji is acknowledged by repository, release, and asset provenance; no paper
citation is invented because the source package provides none.

Please cite the original SPIDER work:

```bibtex
@article{pan2025spiderscalablephysicsinformeddexterous,
      title={SPIDER: Scalable Physics-Informed Dexterous Retargeting},
      author={Chaoyi Pan and Changhao Wang and Haozhi Qi and Zixi Liu and Homanga Bharadhwaj and Akash Sharma and Tingfan Wu and Guanya Shi and Jitendra Malik and Francois Hogan},
      year={2025},
      eprint={2511.09484},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.09484},
}
```
