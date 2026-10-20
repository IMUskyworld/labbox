"""``dataset-manifest.json`` — how we claim to know which data was used.

The digest definition in DEVELOPMENT.md §8.4 is a hard rule: **mtime is not an input**.
A copy, unzip or fresh checkout changes every mtime in the tree, and a digest that
includes mtime would report "the dataset changed" for every one of those runs — which
would destroy the single most important diagnostic in this product.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from labbox.domain.common import SchemaModel


class DatasetStrategy(str, Enum):
    FAST_MANIFEST = "fast_manifest"
    FULL_CONTENT = "full_content"
    EXTERNAL_VERSION = "external_version"
    DECLARED = "declared"


class DatasetKind(str, Enum):
    FILESYSTEM = "filesystem"
    EXTERNAL = "external"
    DECLARED = "declared"


class FileEntry(SchemaModel):
    path: str  # POSIX-style path relative to the dataset root
    size_bytes: int
    head_digest: str | None = None
    tail_digest: str | None = None
    mtime: datetime | None = None  # metadata only, never part of the digest
    content_full: str | None = None  # only for strategy=full_content


class DatasetEntry(SchemaModel):
    id: str
    source: str
    kind: DatasetKind = DatasetKind.FILESYSTEM
    strategy: DatasetStrategy = DatasetStrategy.FAST_MANIFEST
    digest: str | None = None
    digest_inputs: str = "rel_path+size+head64+tail64"
    file_count: int | None = None
    total_bytes: int | None = None
    generated_at: datetime | None = None
    elapsed_seconds: float | None = None
    exclusions: list[str] = Field(default_factory=list)
    external_version: str | None = None
    declared_by_user: bool = False
    verified: bool = False
    zero_byte_files: int = 0
    sampled_only: bool = True
    truncated: bool = False
    cancellation_reason: str | None = None
    touched_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    entries: list[FileEntry] = Field(default_factory=list, exclude=True)

    @property
    def is_content_proof(self) -> bool:
        """Only a full-content or externally pinned version can be called a proof."""
        return self.strategy in (DatasetStrategy.FULL_CONTENT, DatasetStrategy.EXTERNAL_VERSION)


class DatasetManifest(SchemaModel):
    datasets: list[DatasetEntry] = Field(default_factory=list)
    discovery: str = "declared"  # declared | cli | config
    warnings: list[str] = Field(default_factory=list)

    @property
    def not_declared(self) -> bool:
        return not self.datasets
