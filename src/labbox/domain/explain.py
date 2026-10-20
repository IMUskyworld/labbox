"""The only data structure the LLM layer is allowed to see (§11.2).

Everything here is built from already-structured facts. Raw logs, full environment
dumps, git diffs and any credential-shaped string must be filtered out *before* an
``ExplainContext`` is constructed — the model never gets a chance to read them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator

from labbox.domain.common import SchemaModel
from labbox.domain.diagnostics import Diagnostic, DiffFact, Reproducibility


class TaskType(str, Enum):
    SINGLE_RUN = "single_run"
    COMPARE = "compare"


class PrivacyLevel(str, Enum):
    LOCAL_ONLY = "local_only"
    STRUCTURED_ONLY = "structured_only"
    LOGS_ALLOWED = "logs_allowed"


class ExplainContext(SchemaModel):
    task: TaskType = TaskType.SINGLE_RUN
    run_ids: list[str] = Field(default_factory=list)
    facts: list[DiffFact] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_names: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.STRUCTURED_ONLY
    log_excerpt: str | None = None
    max_log_excerpt_chars: int = 4000
    allowed_fact_ids: list[str] = Field(default_factory=list)
    language: str = "zh-CN"

    @field_validator("log_excerpt")
    @classmethod
    def _cap_excerpt(cls, v: str | None) -> str | None:
        # Truncation here is a hard ceiling, not a courtesy: an unbounded excerpt from a
        # 2 GB training log would be sent verbatim to a remote endpoint.
        return None if v is None else v[:4000]


class Hypothesis(SchemaModel):
    text: str
    confidence: str = "low"
    evidence_fact_ids: list[str] = Field(default_factory=list)
    validation: str | None = None

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"confidence must be one of {sorted(allowed)}")
        return v


class FactInterpretation(SchemaModel):
    fact_id: str
    interpretation: str


class ExplainResult(SchemaModel):
    """Validated LLM output (§11.3). A failed validation means the output is dropped."""

    summary: str = ""
    facts: list[FactInterpretation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    rejected_fact_ids: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)

    def filter_to_known_facts(self, context: ExplainContext) -> ExplainResult:
        """Drop any citation the model invented; keep the rest but record the violation."""
        allowed = set(context.allowed_fact_ids) | {f.id for f in context.facts}
        kept = [f for f in self.facts if f.fact_id in allowed]
        kept_h = [h for h in self.hypotheses if all(i in allowed for i in h.evidence_fact_ids)]
        dropped = [f.fact_id for f in self.facts if f.fact_id not in allowed]
        dropped += [i for h in self.hypotheses for i in h.evidence_fact_ids if i not in allowed]
        errors = list(self.validation_errors)
        if dropped:
            errors.append("model cited unknown fact ids: " + ", ".join(sorted(set(dropped))[:10]))
        return self.model_copy(
            update={
                "facts": kept,
                "hypotheses": kept_h,
                "rejected_fact_ids": sorted(set(dropped)),
                "validation_errors": errors,
            }
        )


class ExplainEnvelope(SchemaModel):
    reproducibility: Reproducibility | None = None
    deterministic: dict = Field(default_factory=dict)
    explanation: ExplainResult | None = None
    llm_used: bool = False
