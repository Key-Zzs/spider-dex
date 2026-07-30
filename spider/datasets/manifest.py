"""Deterministic manifests and reproducible lightweight provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import SequenceRecord


def config_hash(config: dict[str, Any]) -> str:
    """Hash semantic configuration independent of dictionary insertion order."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(records: list[SequenceRecord], path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    """Write records sorted by safe sequence ID as portable JSON."""
    ordered = sorted(records, key=lambda item: item.sequence_id)
    value = {"schema_version": 1, "records": [item.to_dict() for item in ordered], "metadata": metadata or {}}
    json.dumps(value, sort_keys=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
