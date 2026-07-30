# Asset provenance

| Field | Recorded value |
| --- | --- |
| Robot type | `wuji_hand2_beta1` |
| Source repository | `https://github.com/wuji-technology/wuji-description` |
| Requested release | `v2026.7.23` |
| Actual source branch at audit | `main` |
| Actual source commit | `1407beed7f478f6ba472d1c42bbdee6f4eec8f7f` |
| Source `describe` | `v2026.7.23` |
| Source dirty state | clean |
| Source path contract | `$WUJI_DESCRIPTION_ROOT/hand2/hand2_beta1/body` |
| License | Upstream MIT, retained as `LICENSE_WUJI` |

The source was not changed, checked out, reset, cleaned, or otherwise written.
Although the requested release branch name was not checked out, the clean commit
is exactly described by `v2026.7.23`; this record intentionally does not claim a
different branch.

## Imported subset

The reproducible command is:

```bash
conda run -n spider-dex python tools/sync_wuji_hand2_assets.py \
  --wuji-description-root "$WUJI_DESCRIPTION_ROOT"
```

It imports only `mjcf/{left,right}.xml`, `urdf/{left,right}.urdf`, and their
referenced `meshes/{left,right}/` directories. It excludes STEP, USD, ROS launch
assets, CAD, and unrelated packages. The imported asset directory is about 20 MB.
`ASSET_MANIFEST.json` enumerates all copied and generated files with SHA-256,
size, source-relative location, and `vendor_original` status. Its timestamp is
deterministic from the source revision so repeated syncs are byte-identical.

Top-level adapter XML/URDF files are generated, not vendor originals. They retain
relative paths only. The package does not use Git LFS, and S2 does not introduce
it.
