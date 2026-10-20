"""``metrics.json`` and the hardware sampling series."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from labbox.domain.common import SchemaModel


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    UNKNOWN = "unknown"


class BestMetric(SchemaModel):
    value: float
    step: int | float | None = None
    epoch: int | None = None
    timestamp: datetime | None = None


class MetricSummary(SchemaModel):
    """Aggregated metrics for one run.

    A missing metric file must produce ``metrics_missing=True`` with ``final=None``, never
    an empty dict and never a guessed number (DEVELOPMENT.md §9.4).
    """

    source: str | None = None
    parser: str | None = None  # jsonl | json | regex:<name> | builtin:<name> | none
    primary_metric: str | None = None
    direction: MetricDirection = MetricDirection.UNKNOWN
    final: dict[str, float] | None = None
    best: dict[str, BestMetric] = Field(default_factory=dict)
    first: dict[str, float] = Field(default_factory=dict)
    point_count: int = 0
    series_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics_missing: bool = False
    parse_errors: list[dict] = Field(default_factory=list)

    def get_final(self, key: str) -> float | None:
        if not self.final:
            return None
        value = self.final.get(key)
        return None if value is None else float(value)


class GpuSample(SchemaModel):
    index: int
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    fan_percent: float | None = None


class SamplePoint(SchemaModel):
    """One row of ``samples/system.jsonl``."""

    t: datetime
    elapsed_seconds: float
    cpu_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_percent: float | None = None
    gpus: list[GpuSample] = Field(default_factory=list)
    error: str | None = None
