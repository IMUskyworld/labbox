"""DiffEngine: turn two run bundles into citable facts (§10).

Three properties this module is built around:

* **A fact needs a path and a source.** ``git.commit`` with sources ``["git"]`` can be
  quoted in a report; a bare "the environment changed" cannot.
* **"Not captured" is never "equal".** If one run failed to record the dataset digest and
  the other recorded one, the answer is ``unknown``, not "no change".
* **Some differences destroy comparability.** If the data changed, comparing accuracy
  between the two runs is not a measurement of the code change; that is
  ``incomparable``, and it is reported instead of hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from labbox.analysis.record import RunBundle
from labbox.domain.common import now_local
from labbox.domain.diagnostics import (
    CompareResult,
    DiffFact,
    DiffKind,
    FactIdPool,
    Severity,
)

HIGH_IMPACT_CATEGORIES = {"data", "code", "config", "randomness"}
_NUMERIC_TOLERANCE = 1e-12

#: Metric-name heuristics, used only to *avoid* an unfounded claim in the opposite
#: direction: a name nobody recognises gets no better/worse verdict rather than a guess.
#: Matched per underscore-separated segment, prefix-wise, so ``eval_accuracy`` is a maximize
#: name, ``train_loss`` a minimize one, and ``mapping_loss`` stays a loss.
MINIMIZE_TOKENS = ("loss", "err", "perplex", "mse", "mae", "cer", "wer", "rmse", "nll")
MAXIMIZE_TOKENS = ("acc", "f1", "bleu", "rouge", "iou", "precision", "recall", "auc", "reward", "map", "score")


def _direction_from_name(name: str) -> str | None:
    """Guess the direction from a metric name, or return ``None`` to say nothing."""
    for segment in re.split(r"[^0-9a-z]+", name.lower()):
        if not segment:
            continue
        if any(segment.startswith(token) for token in MINIMIZE_TOKENS):
            return "minimize"
        if any(segment.startswith(token) for token in MAXIMIZE_TOKENS):
            return "maximize"
    return None


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class DiffEngine:
    def __init__(self, left: RunBundle, right: RunBundle) -> None:
        self.left = left
        self.right = right
        self.facts: list[DiffFact] = []
        self.unchanged: list[str] = []
        self.pool = FactIdPool()

    # --- entry point ---------------------------------------------------------
    def compare(self) -> CompareResult:
        result = CompareResult(
            left_run_id=self.left.run_id,
            right_run_id=self.right.run_id,
            left_name=self.left.display_name,
            right_name=self.right.display_name,
            generated_at=now_local(),
        )
        self._status_facts()
        self._code_facts()
        self._environment_facts()
        self._hardware_facts()
        self._command_facts()
        self._hyperparameter_facts()
        self._data_facts()
        self._randomness_facts()
        self._artifact_facts()
        metric_facts = self._metric_facts()
        result.facts = [fact for fact in self.facts if fact.kind is not DiffKind.EQUAL]
        result.unchanged = self.unchanged
        result.metric_deltas = metric_facts
        result.unknown_count = len([fact for fact in result.facts if fact.kind is DiffKind.UNKNOWN])
        result.incomparable = any(
            fact.kind in {DiffKind.INCOMPARABLE} or (fact.category == "data" and fact.kind is not DiffKind.UNKNOWN)
            for fact in result.facts
        )
        result.metrics_missing = not self.left.final_metrics() or not self.right.final_metrics()
        return result

    # --- one fact ------------------------------------------------------------
    def _add(
        self,
        path: str,
        category: str,
        left: Any,
        right: Any,
        *,
        left_known: bool = True,
        right_known: bool = True,
        sources: Sequence[str] = (),
        confidence: str = "high",
        severity: Severity = Severity.INFO,
    ) -> DiffFact | None:
        if not left_known and not right_known:
            # Both sides blind: reporting a "difference" would be noise, and reporting
            # "equal" would be a lie. The gap belongs in the per-run diagnostics instead.
            return None
        if not left_known or not right_known:
            kind = DiffKind.UNKNOWN
            reason = "only one run recorded this value"
        elif _same(left, right):
            self.unchanged.append(path)
            return None
        elif _is_number(left) and _is_number(right):
            kind = DiffKind.NUMERIC_DELTA
            reason = None
        elif left is None and right is not None:
            kind, reason = DiffKind.ADDED, "value appeared in the right run"
        elif right is None and left is not None:
            kind, reason = DiffKind.REMOVED, "value disappeared in the right run"
        else:
            kind, reason = DiffKind.CHANGED, None
        fact = DiffFact(
            id=self.pool.next(),
            path=path,
            category=category,
            kind=kind,
            left=left if _is_scalar(left) else _text(left),
            right=right if _is_scalar(right) else _text(right),
            left_repr=_text(left),
            right_repr=_text(right),
            delta=_delta(left, right),
            delta_repr=None if _delta(left, right) is None else _text(_delta(left, right)),
            sources=list(sources) or [category],
            confidence=confidence,
            severity=severity,
            reason=reason,
            unknown_side=None if (left_known and right_known) else ("left" if not left_known else "right"),
            comparable=kind is not DiffKind.UNKNOWN,
        )
        self.facts.append(fact)
        return fact

    # --- sections ------------------------------------------------------------
    def _status_facts(self) -> None:
        left = self.left.record
        right = self.right.record
        if left is None or right is None:
            self._add(
                "run.record",
                "other",
                None if left is None else "present",
                None if right is None else "present",
                left_known=left is not None,
                right_known=right is not None,
                severity=Severity.HIGH,
                sources=["run.json"],
            )
            return
        self._add(
            "run.exit_code",
            "other",
            left.exit_code,
            right.exit_code,
            left_known=left.exit_code is not None,
            right_known=right.exit_code is not None,
            sources=["run.json#exit_code"],
            severity=Severity.WARNING,
        )
        self._add("run.status", "other", left.status.value, right.status.value, sources=["run.json#status"])

    def _code_facts(self) -> None:
        left_git, right_git = self.left.git_state(), self.right.git_state()
        for key, severity in (
            ("commit", Severity.HIGH),
            ("branch", Severity.INFO),
            ("dirty", Severity.HIGH),
            ("diff_hash", Severity.HIGH),
        ):
            self._add(
                f"git.{key}",
                "code",
                left_git.get(key),
                right_git.get(key),
                left_known=bool(left_git) and self._git_known(left_git, key),
                right_known=bool(right_git) and self._git_known(right_git, key),
                sources=["environment.json#git"],
                severity=severity,
            )

    @staticmethod
    def _git_known(state: dict[str, Any], key: str) -> bool:
        if not state.get("has_repo"):
            return False
        if key == "diff_hash":
            return state.get("dirty") is True
        return state.get(key) is not None

    def _environment_facts(self) -> None:
        self._add(
            "python.version",
            "environment",
            self.left.env_value("python.version"),
            self.right.env_value("python.version"),
            left_known=not self.left.is_absent("python.version"),
            right_known=not self.right.is_absent("python.version"),
            sources=["environment.json#python"],
            severity=Severity.HIGH,
        )
        frameworks = merge_keys(self.left.env_value("frameworks") or {}, self.right.env_value("frameworks") or {})
        for name in frameworks:
            self._add(
                f"frameworks.{name}",
                "environment",
                (self.left.env_value("frameworks") or {}).get(name),
                (self.right.env_value("frameworks") or {}).get(name),
                left_known=not self.left.is_absent("frameworks"),
                right_known=not self.right.is_absent("frameworks"),
                sources=["environment.json#frameworks"],
                severity=Severity.HIGH,
            )
        for path, severity in (
            ("cuda.driver", Severity.HIGH),
            ("cuda.runtime", Severity.HIGH),
            ("cuda.cudnn", Severity.WARNING),
        ):
            self._add(
                path,
                "environment",
                self.left.env_value(path),
                self.right.env_value(path),
                left_known=not self.left.is_absent(path),
                right_known=not self.right.is_absent(path),
                sources=["environment.json#cuda"],
                severity=severity,
            )
        left_packages = self.left.env_value("packages") or {}
        right_packages = self.right.env_value("packages") or {}
        for name in merge_keys(left_packages, right_packages):
            left_value, right_value = left_packages.get(name), right_packages.get(name)
            if _same(left_value, right_value):
                continue
            self.facts.append(
                DiffFact(
                    id=self.pool.next(),
                    path=f"packages.{name}",
                    category="environment",
                    kind=DiffKind.CHANGED
                    if left_value and right_value
                    else (DiffKind.ADDED if right_value else DiffKind.REMOVED),
                    left=left_value,
                    right=right_value,
                    left_repr=_text(left_value),
                    right_repr=_text(right_value),
                    sources=["environment.json#packages"],
                    confidence="high",
                    severity=Severity.WARNING,
                )
            )

    def _hardware_facts(self) -> None:
        def gpu_names(bundle: RunBundle) -> list[str]:
            if bundle.environment is None:
                return []
            return [gpu.name for gpu in bundle.environment.gpus]

        left_gpus, right_gpus = gpu_names(self.left), gpu_names(self.right)
        self._add(
            "hardware.gpu_names",
            "hardware",
            ", ".join(left_gpus) or None,
            ", ".join(right_gpus) or None,
            left_known=bool(left_gpus),
            right_known=bool(right_gpus),
            sources=["environment.json#gpus"],
            severity=Severity.HIGH,
        )
        for path in ("host.cpu_model", "host.logical_cores", "host.ram_bytes"):
            self._add(
                path,
                "hardware",
                self.left.env_value(path),
                self.right.env_value(path),
                left_known=self.left.env_value(path) is not None,
                right_known=self.right.env_value(path) is not None,
                sources=["environment.json#host"],
                severity=Severity.WARNING,
            )

    def _command_facts(self) -> None:
        left = self.left.record
        right = self.right.record
        if left is None or right is None:
            return
        self._add(
            "command.argv",
            "command",
            " ".join(left.command.argv),
            " ".join(right.command.argv),
            sources=["run.json#command"],
            severity=Severity.HIGH,
        )
        self._add(
            "command.cwd",
            "command",
            left.command.cwd,
            right.command.cwd,
            sources=["run.json#command"],
            severity=Severity.INFO,
        )

    def _hyperparameter_facts(self) -> None:
        left_values, right_values = self.left.hyperparams(), self.right.hyperparams()
        for key in merge_keys(left_values, right_values):
            self._add(
                key,
                "config",
                left_values.get(key),
                right_values.get(key),
                left_known=key in left_values,
                right_known=key in right_values,
                sources=_sources(self.left, self.right, key),
                severity=Severity.HIGH,
            )

    def _data_facts(self) -> None:
        left_digests, right_digests = self.left.dataset_digests(), self.right.dataset_digests()
        ids = merge_keys(left_digests, right_digests)
        if not ids:
            self._add(
                "dataset.digest",
                "data",
                None,
                None,
                left_known=False,
                right_known=False,
                sources=["dataset-manifest.json"],
                severity=Severity.WARNING,
            )
            return
        for dataset_id in ids:
            left_value, right_value = left_digests.get(dataset_id), right_digests.get(dataset_id)
            fact = self._add(
                f"dataset.{dataset_id}.digest",
                "data",
                _short(left_value),
                _short(right_value),
                left_known=left_value is not None,
                right_known=right_value is not None,
                sources=["dataset-manifest.json"],
                severity=Severity.HIGH,
            )
            if fact is not None and fact.kind not in {DiffKind.UNKNOWN}:
                fact.kind = DiffKind.INCOMPARABLE
                fact.comparable = False
                fact.reason = "data version differs; metric differences cannot be attributed to code"
        for dataset_id in ids:
            for field in ("file_count", "total_bytes"):
                left_value = _dataset_field(self.left, dataset_id, field)
                right_value = _dataset_field(self.right, dataset_id, field)
                if left_value is None and right_value is None:
                    continue
                self._add(
                    f"dataset.{dataset_id}.{field}",
                    "data",
                    left_value,
                    right_value,
                    left_known=left_value is not None,
                    right_known=right_value is not None,
                    sources=["dataset-manifest.json"],
                    severity=Severity.WARNING,
                )

    def _randomness_facts(self) -> None:
        for path, source in (
            ("seeds.declared", "config-values.json"),
            ("env.PYTHONHASHSEED", "environment.json#env"),
        ):
            left_value = self._seed_value(self.left, path)
            right_value = self._seed_value(self.right, path)
            self._add(
                path,
                "randomness",
                left_value,
                right_value,
                left_known=left_value is not None,
                right_known=right_value is not None,
                sources=[source],
                severity=Severity.HIGH,
            )

    def _seed_value(self, bundle: RunBundle, path: str) -> Any:
        if path.startswith("seeds."):
            params = bundle.hyperparams()
            for key in ("train.seed", "seed", "train.random_seed", "train.manual_seed"):
                if key in params:
                    return params[key]
            return None
        value = bundle.env_value("env.entries.PYTHONHASHSEED")
        return value

    def _artifact_facts(self) -> None:
        left_hashes, right_hashes = self.left.checkpoint_hashes(), self.right.checkpoint_hashes()
        for name in merge_keys(left_hashes, right_hashes):
            if name.rsplit(".", 1)[-1] not in {"pt", "ckpt", "safetensors", "pth", "bin"}:
                continue
            self._add(
                f"artifact.{name}.sha256",
                "artifacts",
                _short(left_hashes.get(name)),
                _short(right_hashes.get(name)),
                left_known=left_hashes.get(name) is not None,
                right_known=right_hashes.get(name) is not None,
                sources=["artifacts.json"],
                severity=Severity.INFO,
            )

    def _metric_facts(self) -> list[DiffFact]:
        left_metrics, right_metrics = self.left.final_metrics(), self.right.final_metrics()
        produced: list[DiffFact] = []
        for name in merge_keys(left_metrics, right_metrics):
            left_value, right_value = left_metrics.get(name), right_metrics.get(name)
            if left_value is not None and right_value is not None and _same(left_value, right_value):
                self.unchanged.append(f"metrics.{name}")
                continue
            if left_value is None or right_value is None:
                fact = DiffFact(
                    id=self.pool.next(),
                    path=f"metrics.{name}",
                    category="metrics",
                    kind=DiffKind.UNKNOWN,
                    left_repr=_text(left_value),
                    right_repr=_text(right_value),
                    sources=["metrics.json"],
                    confidence="unknown",
                    severity=Severity.WARNING,
                    reason="metric present in only one run",
                    unknown_side="left" if left_value is None else "right",
                    comparable=False,
                )
            else:
                direction = self._direction(name)
                delta = float(right_value) - float(left_value)
                improved = None if direction is None else (delta > 0 if direction == "maximize" else delta < 0)
                fact = DiffFact(
                    id=self.pool.next(),
                    path=f"metrics.{name}",
                    category="metrics",
                    kind=DiffKind.METRIC_DELTA,
                    left=left_value,
                    right=right_value,
                    left_repr=_text(left_value),
                    right_repr=_text(right_value),
                    delta=delta,
                    delta_repr=_delta_text(name, delta, improved),
                    sources=["metrics.json"],
                    confidence="high" if improved is not None else "unknown",
                    severity=Severity.INFO,
                    reason=_delta_reason(direction, improved),
                )
            self.facts.append(fact)
            produced.append(fact)
        return produced

    def _direction(self, name: str) -> str | None:
        """Whether a higher value is better, or ``None`` when nothing recorded it.

        An explicit recorded direction always wins. When the recorded direction is
        ``unknown``, falling through to "unknown means minimize" silently inverted every
        metric that rises when things improve: a real ingest sets ``direction=unknown``
        unless the user configured one, so accuracy 0.7 -> 0.9 came out labelled *worse*.
        The name heuristic below is itself a guess, so it is only used to *not claim* the
        opposite — when neither the record nor the name says anything, the answer is None
        and the fact reports a bare delta with no better/worse claim at all.
        """
        summary = self.right.metrics or self.left.metrics
        recorded = summary.direction.value if summary is not None else ""
        if summary is not None and summary.primary_metric == name and recorded in {"maximize", "minimize"}:
            return recorded
        return _direction_from_name(name)


def _sources(left: RunBundle, right: RunBundle, key: str) -> list[str]:
    out: list[str] = []
    for bundle, label in ((left, "left"), (right, "right")):
        if bundle.config_values is None:
            continue
        value = bundle.config_values.as_map().get(key)
        if value is not None:
            for source in value.sources:
                entry = f"{label}:{source}"
                if entry not in out:
                    out.append(entry)
    return out or ["config-values.json"]


def _dataset_field(bundle: RunBundle, dataset_id: str, field: str) -> Any:
    if bundle.datasets is None:
        return None
    for entry in bundle.datasets.datasets:
        if entry.id == dataset_id:
            return getattr(entry, field)
    return None


def merge_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted({*left, *right})


def _same(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return abs(float(left) - float(right)) <= _NUMERIC_TOLERANCE
    return left == right


def _delta(left: Any, right: Any) -> float | None:
    if not (_is_number(left) and _is_number(right)):
        return None
    return float(right) - float(left)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _short(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= 24:
        return value
    return value[:12] + "..." + value[-6:]


def _delta_text(name: str, delta: float, improved: bool | None) -> str:
    """A signed delta with a verdict only when one is actually justified.

    ``improved=None`` means "nobody recorded which way is better". Printing "worse" then
    would be a fabrication, and printing nothing but the number would still be honest.
    """
    unit = "pp" if name.lower().startswith(("acc", "f1", "map", "iou")) and abs(delta) <= 1.0 else ""
    scaled = delta * 100 if unit == "pp" else delta
    sign = "+" if scaled >= 0 else ""
    verdict = "" if improved is None else " (better)" if improved else " (worse)"
    return f"{sign}{scaled:.4g}{unit}{verdict}"


def _delta_reason(direction: str | None, improved: bool | None) -> str:
    if direction is None or improved is None:
        return (
            "direction was not recorded and the metric name implies nothing, so better/worse "
            "is unknown; only the signed difference is stated"
        )
    return f"{'better' if improved else 'worse'} by direction {direction}"


__all__ = ["HIGH_IMPACT_CATEGORIES", "DiffEngine", "merge_keys"]
