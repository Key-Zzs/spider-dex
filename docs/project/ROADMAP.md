# Roadmap

| Stage | Goal | Inputs | Outputs and automated acceptance | Visual/manual acceptance | Status |
| --- | --- | --- | --- | --- | --- |
| S0 | Repository audit and project scaffolding | Current fork | Audit, bilingual README, workflow, provenance and notices | No | Complete |
| S1 | Upstream SPIDER baseline reproduction | Existing local example | Bounded baseline regression | Existing example may be viewed | User reported complete; regression recorded separately |
| S2 | Wuji Hand2 Beta1 model integration | Versioned Wuji asset source | Packaged adapters, staging, model/load/hold tests | Required; pending user acceptance | Target-model checks pass; overall S2 automatic acceptance blocked by baseline gate |
| S3 | Dataset path/config layer and dataset adapters | External dataset roots | Explicit path/config layer and adapter contracts | Per-adapter review | Not started |
| S3A | External data infrastructure | External workspace and local-only paths | Registry, canonical HOI schema, manifests, audit CLI | No | Complete |
| S4 | Single-sequence GRAB retargeting | One external GRAB sequence | Bounded trajectory and diagnostics | Required | Automated pipeline complete; manual review pending |
| S5 | Physics/contact/collision tuning and evaluation | S4 trajectory | Reproducible evaluation record | Required | Not started |
| Stage C-X | Embodiment-aware task-equivalent contact | Frozen Stage B pilots and immutable V1 evidence | C-XA corrected V2 Level-1 static pass; dynamic D0/D1 pass, D2 fail-closed | No dynamic review material after D2 failure | Dynamic primary blocked; MJWP/smokes/Stage D not started |
| S6 | Batch GRAB conversion | Frozen batch selection | External processed outputs and report | Sampling review | Not started |
| S7 | OakInk and OakInk2 integration | External datasets | Separate adapters and tests | Required | Not started |
| S8 | Export, benchmark, documentation and extensibility | Accepted prior stages | Export/benchmark docs and extension guide | Release review | Not started |

S3–S8 are intentionally not implemented by S2. No raw dataset or large processed
artifact belongs in the repository.
