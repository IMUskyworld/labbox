"""``artifacts.json`` — where outputs live, not copies of them (§7.1)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from labbox.domain.common import SchemaModel


class ArtifactKind(str, Enum):
    CHECKPOINT = "checkpoint"
    LOG = "log"
    CONFIG = "config"
    METRICS = "metrics"
    IMAGE = "image"
    OTHER = "other"


class HashState(str, Enum):
    NOT_COMPUTED = "not_computed"
    COMPUTED = "computed"
    DEFERRED_TOO_LARGE = "deferred_too_large"
    FAILED = "failed"
    SKIPPED_DISABLED = "skipped_disabled"


class ArtifactEntry(SchemaModel):
    kind: ArtifactKind = ArtifactKind.OTHER
    path: str
    abs_path: str | None = None
    size_bytes: int | None = None
    mtime: datetime | None = None
    sha256: str | None = None
    hash_state: HashState = HashState.NOT_COMPUTED
    hash_bytes_read: int | None = None
    archived: bool = False
    storage_uri: str | None = None
    discovered_by: str = "config_glob"  # config_glob | cli | checkpoint_suffix | metrics_file
    error: str | None = None


class ArtifactManifest(SchemaModel):
    artifacts: list[ArtifactEntry] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    scanned_globs: list[str] = Field(default_factory=list)
    hash_threshold_bytes: int = 2 * 1024 * 1024 * 1024
    copied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(a.size_bytes or 0 for a in self.artifacts)

    def by_kind(self, kind: ArtifactKind) -> list[ArtifactEntry]:
        return [a for a in self.artifacts if a.kind is kind]
