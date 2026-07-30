"""External-dataset infrastructure for SPIDER-Dex.

The package intentionally separates immutable source data and body models from
the writable, external SPIDER workspace.  It does not change SPIDER's existing
``dataset_dir`` compatibility contract.
"""

from .base import DatasetAdapter, DatasetAudit, SequenceRecord
from .paths import ProjectPaths, load_project_paths
from .registry import DatasetRegistry, registry
from .schema import CanonicalHOISequence, HandSequence, ObjectSequence

__all__ = [
    "CanonicalHOISequence",
    "DatasetAdapter",
    "DatasetAudit",
    "DatasetRegistry",
    "HandSequence",
    "ObjectSequence",
    "ProjectPaths",
    "SequenceRecord",
    "load_project_paths",
    "registry",
]
