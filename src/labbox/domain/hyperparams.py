"""Hyperparameter values and the provenance of each value (§9.6).

``compare`` needs to say ``train.batch_size 64 -> 128``. That is only honest if we know
*where* each value came from, so every value carries a source list and a confidence.
When the same key is set both on the command line and in a config file, both sources are
kept; silently picking one would make the diff wrong in the worst possible way.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from labbox.domain.common import SchemaModel


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ValueSource(str, Enum):
    COMMAND_ARG = "command.argv"
    CONFIG_FILE = "config.file"
    ENV = "env"
    CODE_DEFAULT = "code.default"
    USER_DECLARED = "user.declared"
    LOG = "log"


class HyperValue(SchemaModel):
    path: str  # dotted, e.g. "train.batch_size"
    value: str | int | float | bool | None = None
    value_repr: str = ""  # what we show, always a string so types cannot be confused
    sources: list[str] = Field(default_factory=list)  # e.g. ["configs/base.yaml:12", "command.argv"]
    source_kinds: list[ValueSource] = Field(default_factory=list)
    confidence: Confidence = Confidence.HIGH
    raw_type: str = ""
    ambiguous: bool = False


class HyperparamSet(SchemaModel):
    values: list[HyperValue] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    files_skipped: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    flatten_depth_limit: int = 6

    def as_map(self) -> dict[str, HyperValue]:
        return {v.path: v for v in self.values}
