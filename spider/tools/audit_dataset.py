"""Validate external paths and run a registered adapter's source audit."""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from spider.datasets.base import DatasetAudit
from spider.datasets.paths import load_project_paths
from spider.datasets.registry import registry


def main(
    dataset: str,
    paths_config: str,
    json_output: str | None = None,
    max_sequences: int = 20,
    dry_run: bool = False,
    source_root: str | None = None,
    body_model_root: str | None = None,
    workspace_root: str | None = None,
) -> None:
    """Run configuration/workspace checks before any dataset processing."""
    paths = load_project_paths(paths_config, source_overrides={dataset: source_root} if source_root else None, body_model_root=body_model_root, workspace_root=workspace_root)
    paths.ensure_workspace()
    if dataset == "grab" and dataset not in registry.names():
        from spider.datasets.grab import GrabAdapter

        registry.register(GrabAdapter(paths))
    checks = {"paths_valid": True, "workspace": str(paths.workspace_root), "registered_adapters": list(registry.names()), "dry_run": dry_run}
    if dataset not in registry.names():
        audit = DatasetAudit(dataset_name=dataset, status="ADAPTER_NOT_REGISTERED", source_root=str(paths.source_root(dataset)), checks=checks, summary={"max_sequences": max_sequences}, errors=[])
    else:
        audit = registry.get(dataset).inspect_source(max_sequences=max_sequences)
        audit.checks = {**checks, **audit.checks}
    result = audit.to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    if json_output:
        output = Path(json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tyro.cli(main)
