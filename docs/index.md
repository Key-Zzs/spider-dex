---
layout: home

hero:
  name: "SPIDER"
  text: "Scalable Physics-Informed DExterous Retargeting"
  tagline: A general framework for physics-based retargeting from human to diverse robot embodiments
  image:
    src: /figs/teaser.png
    alt: SPIDER
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/facebookresearch/spider

features:
  - icon: 🔬
    title: Physics-Based
    details: First general physics-based retargeting pipeline for both dexterous hand and humanoid robot manipulation

  - icon: ⚡
    title: Fast Simulation
    details: GPU-accelerated batched simulation with MuJoCo Warp achieving 10-20x speedup over sequential execution

  - icon: 📊
    title: Rich Datasets, Robots and Simulators
    details: Works with 6+ datasets out of the box including GigaHand, Hot3D, OakInk, and more. Supports 9+ robot embodiments including dexterous hands (Allegro, Inspire, Xhand) and humanoid robots (G1, H1, T1). Supports multiple simulators including MuJoCo Warp, Genesis, and HDMI.

  - icon: 🔄
    title: Sim2Real Ready
    details: Optimized trajectories can be directly deployed to real-world robots with minimal adjustments
---

## SPIDER-Dex project

The fork-specific documentation is collected in the
[project index](project/index.md), including the Wuji Hand2 Beta1 adapter,
audit, provenance, workflow, and validation status.

## Quick Example

```bash
# Clone example datasets
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets

# Install with uv
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .

# Run retargeting
uv run examples/run_mjwp.py
```
