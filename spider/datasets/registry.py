"""Explicit registry for dataset adapters."""

from __future__ import annotations

from .base import DatasetAdapter


class DatasetRegistry:
    """A small registry with no import-time source-data side effects."""

    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}

    def register(self, adapter: DatasetAdapter) -> None:
        name = getattr(adapter, "dataset_name", "")
        if not isinstance(name, str) or not name:
            raise ValueError("Dataset adapter must define a non-empty dataset_name")
        if name in self._adapters:
            raise ValueError(f"Dataset adapter already registered: {name!r}")
        self._adapters[name] = adapter

    def get(self, name: str) -> DatasetAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters)) or "(none)"
            raise KeyError(f"Unregistered dataset {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


registry = DatasetRegistry()
