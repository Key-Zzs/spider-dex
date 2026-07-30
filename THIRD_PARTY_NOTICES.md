# Third-party notices

## SPIDER

This repository is a fork of Meta's SPIDER. The root [LICENSE](LICENSE) is the
upstream CC-BY-NC license and continues to govern the SPIDER code and this
fork's distribution boundary. This repository is not globally MIT-licensed.

## Wuji Hand2 Beta1 assets

`spider/assets/robots/wuji_hand2_beta1/vendor/` contains the required subset
of the Wuji Technology `wuji-description` package, release `v2026.7.23`:
MJCF, URDF, and the referenced left/right STL meshes. The upstream MIT notice
is preserved verbatim in `LICENSE_WUJI` in the same asset directory.

The top-level `right.xml`, `left.xml`, `bimanual.xml`, and `urdf/*_6dof.urdf`
are SPIDER adapter files. They add scalar wrist controls, SPIDER site aliases,
safe collision names, and a bimanual assembly; they do not duplicate or replace
the vendor collision geometry. `ASSET_MANIFEST.json` records every copied or
derived file and its SHA-256.

## Datasets and MANO models

External datasets and body models are neither copied into nor redistributed by
this repository. Their licenses and access restrictions remain with their
respective providers.
