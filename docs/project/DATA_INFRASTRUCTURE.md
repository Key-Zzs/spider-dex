# External data infrastructure

SPIDER-Dex keeps three roots deliberately separate:

- `source_root`: immutable raw dataset files, readable only;
- `body_model_root`: immutable MANO/SMPL-X model assets, readable only;
- `workspace_root`: writable external SPIDER manifests, processed data, runs,
  reports, and cache.

Use `configs/project/paths.example.yaml` as the tracked template and create an
ignored `configs/local/paths.yaml` for a machine-specific configuration. Values
support `~` and environment variables; explicit CLI values override YAML.
`workspace_root` cannot be the source root, lie below it, contain it, or be
inside a Git repository. No fallback writes to this checkout are permitted.

Create/check a workspace and inspect a dataset with:

```bash
conda run -n spider-dex python -m spider.tools.audit_dataset \
  --dataset grab --paths-config configs/local/paths.yaml \
  --json-output /path/to/spider_workspace/reports/grab_audit.json
```

The workspace is created idempotently with `manifests/`, `processed/`, `runs/`,
`reports/`, and `cache/`. The audit CLI reports a missing adapter explicitly;
it never scans a NAS at import time.

## Registry and manifests

`spider.datasets.registry` owns explicit adapter registration. Each adapter is
robot-independent and exposes deterministic discovery, inspection, loading,
mesh resolution, and sequence description. `SequenceRecord` manifests contain
only source-relative paths and lightweight metadata by default. A full source
hash is reserved for an explicitly processed file, not a dataset-wide default.
