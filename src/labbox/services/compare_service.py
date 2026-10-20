"""Compare two runs: DiffEngine + rules, and the §4.4 terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labbox.analysis.diff import DiffEngine
from labbox.analysis.record import RunBundle
from labbox.analysis.rules import compare_diagnostics, confounders, score_reproducibility
from labbox.domain.diagnostics import CompareResult, DiffFact, DiffKind, Severity
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout


@dataclass(slots=True)
class Comparison:
    result: CompareResult
    left: RunBundle
    right: RunBundle
    reproducibility: tuple[Any, Any]
    saved_to: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = self.result.model_dump(mode="json")
        payload["reproducibility"] = {
            "left": self.reproducibility[0].model_dump(mode="json"),
            "right": self.reproducibility[1].model_dump(mode="json"),
        }
        return payload


def compare_runs(
    layout: ProjectLayout,
    left_query: str,
    right_query: str,
    *,
    save: bool = False,
) -> Comparison:
    store = FilesystemStore(layout)
    left_dir = layout.resolve_run_id(left_query)
    right_dir = layout.resolve_run_id(right_query)
    left = RunBundle.load(store, left_dir)
    right = RunBundle.load(store, right_dir)
    result = DiffEngine(left, right).compare()
    result.diagnostics = compare_diagnostics(result, [left, right])
    result.confounders = confounders(result)
    reproducibility = (score_reproducibility(left), score_reproducibility(right))
    saved_to: Path | None = None
    if save:
        saved_to = right_dir / "compare.json"
        payload = {
            "schema_version": "1.0",
            **Comparison(result=result, left=left, right=right, reproducibility=reproducibility).as_dict(),
        }
        store.write_json_atomic(saved_to, payload)
    return Comparison(result=result, left=left, right=right, reproducibility=reproducibility, saved_to=saved_to)


# --- terminal rendering ------------------------------------------------------
def format_changed_facts(facts: list[DiffFact], limit: int = 60) -> list[str]:
    lines: list[str] = []
    for fact in facts[:limit]:
        marker = {
            DiffKind.CHANGED: "",
            DiffKind.NUMERIC_DELTA: "",
            DiffKind.METRIC_DELTA: "",
            DiffKind.ADDED: "(added)",
            DiffKind.REMOVED: "(removed)",
            DiffKind.UNKNOWN: "(unknown)",
            DiffKind.INCOMPARABLE: "(incomparable)",
        }.get(fact.kind, "")
        left = fact.left_repr if fact.left_repr not in ("", "-") else "-"
        right = fact.right_repr if fact.right_repr not in ("", "-") else "-"
        tail = f"  {marker}" if marker else ""
        lines.append(f"  [{fact.id}] {fact.path:<28} {left} -> {right}{tail}")
    if len(facts) > limit:
        lines.append(f"  ... {len(facts) - limit} more")
    return lines


def format_diagnostics(diagnostics: list[Any]) -> list[str]:
    order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.WARNING: 2,
        Severity.INFO: 3,
    }
    lines: list[str] = []
    for diagnostic in sorted(diagnostics, key=lambda item: order.get(item.severity, 9)):
        ids = " ".join(diagnostic.fact_ids[:4])
        suffix = f"  refs: {ids}" if ids else ""
        lines.append(f"  [{diagnostic.severity.value.upper():>8}] {diagnostic.message}{suffix}")
    return lines


def format_metrics(facts: list[DiffFact]) -> list[str]:
    lines: list[str] = []
    width = max((len(fact.path.split(".", 1)[-1]) for fact in facts), default=7)
    for fact in facts:
        name = fact.path.split(".", 1)[-1]
        lines.append(f"  {name:<{width}}   {fact.left_repr} -> {fact.right_repr}   {fact.delta_repr or ''}")
    return lines


def render(comparison: Comparison, *, color: bool = False) -> str:  # pragma: no cover - formatting only
    result = comparison.result
    blocks: list[str] = []
    blocks.append(f"Comparing {result.left_name} ({result.left_run_id}) vs {result.right_name} ({result.right_run_id})")
    if result.incomparable:
        blocks.append("[HIGH] The two runs are not directly comparable: see data/code diagnostics below.")
    blocks.append("")
    blocks.append("Changed facts")
    changed = format_changed_facts(result.facts)
    blocks.extend(changed or ["  (none)"])
    unchanged = sorted(set(result.unchanged))
    if unchanged:
        blocks.append("")
        blocks.append("Unchanged")
        blocks.append("  " + ", ".join(unchanged[:14]) + (", ..." if len(unchanged) > 14 else ""))
    blocks.append("")
    blocks.append("Diagnostics")
    blocks.extend(format_diagnostics(result.diagnostics) or ["  (none)"])
    blocks.append("")
    blocks.append("Metrics")
    metric_lines = format_metrics(result.metric_deltas)
    if not metric_lines:
        if result.metrics_missing:
            metric_lines = ["  no metrics recorded in at least one run"]
        else:
            metric_lines = ["  metrics are identical in both runs"]
    blocks.extend(metric_lines)
    blocks.append("")
    blocks.append("Reproducibility")
    for label, score in zip(("left ", "right "), comparison.reproducibility, strict=True):
        summary = ", ".join(f"{check.id}={check.state.value}" for check in score.checklist)
        blocks.append(f"  {label} {score.band.value} (score {score.score}): {summary}")
    return "\n".join(blocks)


__all__ = [
    "Comparison",
    "compare_runs",
    "format_changed_facts",
    "format_diagnostics",
    "format_metrics",
    "render",
]
