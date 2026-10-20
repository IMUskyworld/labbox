"""Explain service: facts in, model out, citations verified (§4.5, §11.3).

The service never lets a model result stand on its own: the deterministic analysis is
always produced, the model output is validated, any citation to a fact id that was never
issued is stripped, and if what is left is unusable the caller shows only the
deterministic part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labbox.analysis.record import RunBundle
from labbox.analysis.rules import score_reproducibility
from labbox.config.model import LabboxConfig
from labbox.domain.explain import ExplainEnvelope, PrivacyLevel
from labbox.errors import LabBoxError
from labbox.infrastructure.store import FilesystemStore
from labbox.llm.context import build_context, data_categories, refusal_reason
from labbox.llm.provider import LLMProvider, NullProvider, get_provider
from labbox.paths import ProjectLayout
from labbox.services.compare_service import Comparison, compare_runs


@dataclass(slots=True)
class ExplainOutcome:
    envelope: ExplainEnvelope
    categories: list[str] = field(default_factory=list)
    refused: str | None = None
    sent: bool = False
    payload_preview: dict[str, Any] | None = None
    saved_to: str | None = None

    @property
    def usable(self) -> bool:
        explanation = self.envelope.explanation
        return bool(explanation and (explanation.summary or explanation.hypotheses))


def explain_run(
    layout: ProjectLayout,
    run_query: str,
    *,
    against: str | None = None,
    config: LabboxConfig | None = None,
    provider: LLMProvider | None = None,
    dry_run: bool = False,
    include_logs: bool = False,
) -> ExplainOutcome:
    store = FilesystemStore(layout)
    loaded_config = config or LabboxConfig.load(layout.config_path)
    bundles = [RunBundle.load(store, layout.resolve_run_id(run_query))]
    comparison: Comparison | None = None
    if against:
        comparison = compare_runs(layout, run_query, against)
        bundles = [comparison.left, comparison.right]

    context = build_context(
        bundles=bundles,
        config=loaded_config,
        compare=comparison.result if comparison else None,
        include_log_excerpt=include_logs,
    )
    reproducibility = None if comparison is not None else score_reproducibility(bundles[0])
    envelope = ExplainEnvelope(
        reproducibility=reproducibility,
        deterministic={
            "run_ids": context.run_ids,
            "fact_count": len(context.facts),
            "diagnostics": [item.code for item in context.diagnostics],
            "missing_fields": context.missing_fields,
            "metrics_missing": bool(comparison and comparison.result.metrics_missing),
            "incomparable": bool(comparison and comparison.result.incomparable),
        },
    )
    categories = data_categories(context)
    reason = refusal_reason(context, loaded_config)
    if dry_run:
        return ExplainOutcome(
            envelope=envelope,
            categories=categories,
            refused=reason,
            payload_preview=_preview(context),
        )
    if reason:
        return ExplainOutcome(envelope=envelope, categories=categories, refused=reason)

    chosen = provider or get_provider(loaded_config)
    if isinstance(chosen, NullProvider):
        return ExplainOutcome(envelope=envelope, categories=categories, refused="llm provider is null")
    try:
        result = chosen.explain(context)
    except LabBoxError as exc:
        return ExplainOutcome(
            envelope=envelope.model_copy(update={"validation_note": exc.message}),
            categories=categories,
            refused=f"{exc.code}: {exc.message}",
        )
    filtered = result.filter_to_known_facts(context).model_copy(
        update={"provider": chosen.name, "model": getattr(chosen, "model", "") or ""}
    )
    usable = bool(filtered.summary or filtered.hypotheses)
    if filtered.validation_errors and not usable:
        return ExplainOutcome(
            envelope=envelope,
            categories=categories,
            refused="model output failed schema validation: " + "; ".join(filtered.validation_errors[:3]),
        )
    envelope = envelope.model_copy(update={"explanation": filtered, "llm_used": usable})
    saved = _save(store, bundles[-1], envelope)
    return ExplainOutcome(envelope=envelope, categories=categories, sent=usable, saved_to=saved)


def _save(store: FilesystemStore, bundle: RunBundle, envelope: ExplainEnvelope) -> str | None:
    target = bundle.paths.resolve("explain.json")
    try:
        store.write_json_atomic(target, envelope)
        return "explain.json"
    except LabBoxError:
        return None


def _preview(context: Any) -> dict[str, Any]:
    """What would be sent, with values reduced to shapes (§16.2 requires this view)."""
    return {
        "task": context.task.value,
        "privacy": context.privacy.value,
        "categories": data_categories(context),
        "run_ids": context.run_ids,
        "fact_ids": context.allowed_fact_ids[:20],
        "fact_paths": [fact.path for fact in context.facts][:20],
        "diagnostic_codes": [item.code for item in context.diagnostics][:20],
        "metric_names": context.metric_names,
        "missing_fields": context.missing_fields[:20],
        "log_excerpt_chars": len(context.log_excerpt or ""),
        "note": "environment values, file contents and raw logs are never included unless send_logs is on",
    }


def format_outcome(outcome: ExplainOutcome) -> str:
    lines: list[str] = []
    explanation = outcome.envelope.explanation
    if outcome.refused and not outcome.usable:
        lines.append(f"[llm not used] {outcome.refused}")
    deterministic = outcome.envelope.deterministic
    lines.append("Deterministic analysis")
    lines.append(f"  facts compared   {deterministic.get('fact_count')}")
    lines.append(f"  incomparable     {deterministic.get('incomparable')}")
    lines.append(f"  metrics missing  {deterministic.get('metrics_missing')}")
    if deterministic.get("diagnostics"):
        lines.append("  diagnostics      " + ", ".join(deterministic["diagnostics"][:10]))
    if deterministic.get("missing_fields"):
        lines.append("  missing evidence " + ", ".join(deterministic["missing_fields"][:6]))
    if explanation:
        lines.append("")
        lines.append("Model interpretation (LLM-generated, not evidence)")
        if explanation.summary:
            lines.append(f"  summary  {explanation.summary}")
        for item in explanation.facts[:10]:
            lines.append(f"  {item.fact_id}: {item.interpretation}")
        for hypothesis in explanation.hypotheses[:8]:
            refs = ", ".join(hypothesis.evidence_fact_ids) or "no citation"
            lines.append(f"  [{hypothesis.confidence}] {hypothesis.text} (refs {refs})")
            if hypothesis.validation:
                lines.append(f"      validate: {hypothesis.validation}")
        for unknown in explanation.unknowns[:8]:
            lines.append(f"  unknown: {unknown}")
        for step in explanation.next_steps[:8]:
            lines.append(f"  next: {step}")
        if explanation.rejected_fact_ids:
            lines.append(
                "  [dropped] the model cited unknown fact ids: " + ", ".join(explanation.rejected_fact_ids[:6])
            )
        if explanation.validation_errors:
            lines.append("  [schema] " + "; ".join(explanation.validation_errors[:3]))
    if outcome.envelope.reproducibility:
        score = outcome.envelope.reproducibility
        lines.append(f"\nReproducibility: {score.band.value} (score {score.score})")
    return "\n".join(lines)


__all__ = ["ExplainOutcome", "PrivacyLevel", "explain_run", "format_outcome"]
