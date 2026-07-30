"""Dataset-adapter protocol and lightweight discovery records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SequenceRecord:
    """A deterministic source-sequence index entry."""

    dataset_name: str
    sequence_id: str
    source_sequence_id: str
    source_relative_path: str
    num_frames: int | None = None
    fps: float | None = None
    hand_presence: str = "unknown"
    object_ids: tuple[str, ...] = ()
    status: str = "DISCOVERED"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["object_ids"] = list(self.object_ids)
        return value


@dataclass
class DatasetAudit:
    """Structured result returned by adapters and the audit CLI."""

    dataset_name: str
    status: str
    source_root: str
    checks: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter(ABC):
    """Robot-independent dataset boundary.

    Adapters may inspect source metadata during discovery but must not scan at
    import time, reconstruct a body model during discovery, or perform IK.
    """

    dataset_name: str

    @abstractmethod
    def discover_sequences(self, max_sequences: int | None = None) -> list[SequenceRecord]:
        """Return source records in deterministic order."""

    @abstractmethod
    def inspect_source(self, max_sequences: int | None = None) -> DatasetAudit:
        """Inspect source data without robot conversion."""

    @abstractmethod
    def load_sequence(self, sequence_id: str, **kwargs: Any):
        """Load one sequence into the canonical representation."""

    @abstractmethod
    def resolve_object_mesh(self, object_id: str) -> Path:
        """Resolve one source mesh without copying it."""

    @abstractmethod
    def describe_sequence(self, sequence_id: str) -> SequenceRecord:
        """Return metadata for one source sequence."""
