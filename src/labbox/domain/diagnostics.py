"""Diagnostics, diff facts and comparison results (``diagnostics.json`` / compare output).

The vocabulary here is the whole point of LabBox: ``unknown`` and ``incomparable`` are
outcomes as valuable as ``changed``, because a diff that silently treats "not captured"
as "equal" is worse than no diff at all.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from labbox.domain.common import SchemaModel


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    INFO = "info"


class DiffKind(str, Enum):
    CHANGED = "changed"
    ADDED = "added"
    REMOVED = "removed"
    NUMERIC_DELTA = "numeric_delta"
    METRIC_DELTA = "metric_delta"
    UNKNOWN = "unknown"
    INCOMPARABLE = "incomparable"
    EQUAL = "equal"


class ReproState(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ReproBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Diagnostic(SchemaModel):
    id: str = ""
    severity: Severity = Severity.INFO
    code: str = ""
    message: str = ""
    evidence: list[str] = Field(default_factory=list)
    suggestion: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    rule: str | None = None


class ReproCheck(SchemaModel):
    id: str
    state: ReproState = ReproState.UNKNOWN
    reason: str = ""
    weight: int = 0


class Reproducibility(SchemaModel):
    """Checklist first, number second (§12.1)."""

    band: ReproBand = ReproBand.LOW
    score: int = 0
    checklist: list[ReproCheck] = Field(default_factory=list)
    rules_version: str = "1"

    def state_of(self, check_id: str) -> ReproState:
        for check in self.checklist:
            if check.id == check_id:
                return check.state
        return ReproState.UNKNOWN


class DiffFact(SchemaModel):
    id: str = ""
    path: str = ""
    kind: DiffKind = DiffKind.CHANGED
    left: str | int | float | bool | None = None
    right: str | int | float | bool | None = None
    left_repr: str = ""
    right_repr: str = ""
    delta: float | None = None
    delta_repr: str | None = None
    category: str = (
        "other"  # code | environment | hardware | command | config | data | randomness | metrics | artifacts
    )
    sources: list[str] = Field(default_factory=list)
    confidence: str = "high"
    severity: Severity = Severity.INFO
    comparable: bool = True
    unknown_side: str | None = None  # "left" | "right" | "both"
    reason: str | None = None


class DiagnosticsFile(SchemaModel):
    generated_at: datetime
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    rules_version: str = "1"


class CompareResult(SchemaModel):
    """What ``labbox compare`` renders and ``explain`` consumes."""

    left_run_id: str
    right_run_id: str
    left_name: str = ""
    right_name: str = ""
    generated_at: datetime
    facts: list[DiffFact] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    reproducibility: tuple[Reproducibility, Reproducibility] | None = None
    metric_deltas: list[DiffFact] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    incomparable: bool = False
    metrics_missing: bool = False
    unknown_count: int = 0

    def fact(self, fact_id: str) -> DiffFact | None:
        for f in self.facts:
            if f.id == fact_id:
                return f
        return None

    @property
    def changed_fact_count(self) -> int:
        return len([f for f in self.facts if f.kind is not DiffKind.UNKNOWN])


class FactIdPool:
    """Allocates stable ``F-001`` ids in the order facts are produced."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        self._n += 1
        return f"F-{self._n:03d}"


class DiagnosticIdPool:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        self._n += 1
        return f"D-{self._n:03d}"
