"""Metrics ingestion (§9.4) — structured input only, never a guessed number.

Priority order is fixed: ``--metric-file`` > config ``[metrics] files`` > user regex >
builtin line parser. If nothing yields a metric, the result says so
(``metrics_missing=True``) and ``compare`` reports ``METRICS_MISSING``; inventing a value
from "the last float in the log" is the failure mode this module exists to avoid.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from labbox.config.model import LabboxConfig
from labbox.domain.metrics import BestMetric, MetricDirection, MetricSummary

# `epoch 3/20 - loss: 0.21 - acc: 0.81` style lines from hand-written training loops.
_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_./-]{0,40})[\s]*[:=][\s]*(?P<value>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
_FLOAT_ONLY = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")
NOISY_KEYS = frozenset(
    {
        "epoch",
        "step",
        "iter",
        "iteration",
        "batch",
        "batch_idx",
        "idx",
        "index",
        "lr",
        "lrate",
        "num",
        "count",
        "size",
        "bs",
        "batch_size",
        "seed",
        "rank",
        "world_size",
        "total",
        "n",
        "nexamples",
        "elapsed",
        "time",
        "eta",
        "percent",
        "pct",
    }
)


class MetricsIngestor:
    def __init__(
        self,
        *,
        project_root: Path,
        config: LabboxConfig,
        metric_file: str | None = None,
        max_series_points: int | None = None,
    ) -> None:
        self.root = Path(project_root)
        self.config = config
        self.explicit = metric_file
        self.max_points = max_series_points or config.metrics.max_series_points

    # --- public API ----------------------------------------------------------
    def ingest(self, run_dir: Path, history_path: Path | None = None) -> MetricSummary:
        candidates = self._candidate_files()
        parse_errors: list[dict[str, Any]] = []
        for path in candidates:
            if not path.is_file():
                continue
            points, errors = self._read_structured(path)
            parse_errors.extend(errors)
            if points:
                summary = self._summarize(points, source=self._display(path), parser=_format_of(path))
                if history_path is not None and summary.point_count:
                    # Only advertise the series reference when the file is really there: a
                    # dangling link in metrics.json would read as "recorded, somewhere".
                    if _write_history(points[-self.max_points :], history_path):
                        summary.series_ref = _relative(run_dir, history_path)
                    else:
                        summary.warnings.append(
                            f"metrics history could not be written to {history_path}; metrics.json itself is complete"
                        )
                summary.parse_errors = parse_errors
                return summary
        lines_summary = self._from_stdout_logs(run_dir)
        if lines_summary is not None:
            lines_summary.parse_errors = parse_errors
            return lines_summary
        return MetricSummary(
            source=None,
            parser="none",
            metrics_missing=True,
            final=None,
            warnings=[
                "metrics_missing: no structured metrics file was declared or found, "
                "and LabBox does not guess values out of free-form logs"
            ]
            + [f"{error['code']}: {error['message']}" for error in parse_errors[:5]],
            parse_errors=parse_errors,
        )

    # --- file discovery ------------------------------------------------------
    def _candidate_files(self) -> list[Path]:
        ordered: list[str] = []
        if self.explicit:
            ordered.append(self.explicit)
        ordered.extend(self.config.metrics.files)
        out: list[Path] = []
        seen: set[str] = set()
        for spec in ordered:
            for resolved in self._expand(spec):
                key = str(resolved)
                if key in seen or resolved.is_dir():
                    continue
                seen.add(key)
                out.append(resolved)
        for pattern in ("outputs/metrics.json", "metrics.json", "results.json"):
            candidate = self.root / pattern
            if candidate.is_file() and str(candidate) not in seen:
                seen.add(str(candidate))
                out.append(candidate)
        return out

    def _expand(self, spec: str) -> list[Path]:
        """Resolve one declared spec to concrete files, project-root relative.

        A glob is expanded with the project root as the anchor. Rooting ``Path.glob`` at
        the magic component itself (``.../outputs/**``) matches nothing, because glob
        treats its receiver as a literal directory and no directory is called ``**``.
        """
        if not any(char in spec for char in "*?["):
            path = Path(spec)
            return [path if path.is_absolute() else self.root / path]
        path = Path(spec)
        if path.is_absolute():
            return _glob_absolute(path)
        try:
            return sorted(self.root.glob(spec))
        except (OSError, ValueError):  # pragma: no cover - malformed pattern
            return []

    def _display(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except (ValueError, OSError):
            return str(path)

    # --- parsers -------------------------------------------------------------
    def _read_structured(self, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        text = _read_text(path)
        if text is None:
            return [], [{"code": "METRICS_FILE_INVALID", "message": f"{path} is not readable"}]
        if path.suffix.lower() == ".jsonl":
            return _parse_jsonl(text, path)
        if path.suffix.lower() == ".json":
            return _parse_json(text, path)
        # An extension-less file may still be JSONL; try it, then give up.
        points, errors = _parse_jsonl(text, path)
        if points:
            return points, errors
        return [], errors

    def _from_stdout_logs(self, run_dir: Path) -> MetricSummary | None:
        """Only the configured regex rules and the documented builtin line parser."""
        stdout = run_dir / "logs" / "stdout.plain.log"
        if not stdout.is_file():
            stdout = run_dir / "logs" / "stdout.log"
        if not stdout.is_file():
            return None
        text = _read_text(stdout) or ""
        lines = text.splitlines()
        named = self._apply_user_regex(lines)
        if named:
            return self._summarize(named, source=_relative(run_dir, stdout), parser="regex:configured")
        points = _parse_key_value_lines(lines)
        if not points:
            return None
        summary = self._summarize(points, source=_relative(run_dir, stdout), parser="builtin:key_value")
        summary.warnings.append(
            "values were parsed from stdout with the builtin key=value parser; declare --metric-file for exact metrics"
        )
        return summary

    def _apply_user_regex(self, lines: Sequence[str]) -> list[dict[str, Any]]:
        rules = {name: _compile(pattern) for name, pattern in self.config.metrics.regex.items()}
        rules = {name: rule for name, rule in rules.items() if rule is not None}
        if not rules:
            return []
        points: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            for name, rule in rules.items():
                match = rule.search(line)
                if not match:
                    continue
                value = _first_number(match)
                if value is None:
                    continue
                points.append({"step": index, name: value})
        return points

    # --- aggregation ---------------------------------------------------------
    def _summarize(self, points: list[dict[str, Any]], *, source: str, parser: str) -> MetricSummary:
        keys = _metric_keys(points)
        final: dict[str, float] = {}
        first: dict[str, float] = {}
        best: dict[str, BestMetric] = {}
        for key in keys:
            series = [
                (point.get("step", index), float(point[key]))
                for index, point in enumerate(points)
                if _is_number(point.get(key))
            ]
            if not series:
                continue
            first[key] = series[0][1]
            final[key] = series[-1][1]
            direction = _direction_for(key, self.config)
            best_value = (
                max(series, key=lambda item: item[1])
                if direction is not MetricDirection.MINIMIZE
                else min(series, key=lambda item: item[1])
            )
            best[key] = BestMetric(value=float(best_value[1]), step=_as_number_or_none(best_value[0]))
        primary = self.config.metrics.primary or _pick_primary(keys)
        return MetricSummary(
            source=source,
            parser=parser,
            primary_metric=primary,
            direction=MetricDirection(self.config.metrics.direction)
            if self.config.metrics.direction
            else MetricDirection.UNKNOWN,
            final=final or None,
            first=first,
            best=best,
            point_count=len(points),
            warnings=[] if points else ["no usable metric rows"],
        )


def _glob_absolute(pattern: Path) -> list[Path]:
    """Expand an absolute glob by anchoring on its literal prefix."""
    anchor = pattern.anchor
    parts = pattern.parts[1:] if anchor else pattern.parts
    literal: list[str] = []
    for part in parts:
        if any(char in part for char in "*?["):
            break
        literal.append(part)
    base = Path(anchor).joinpath(*literal) if literal else Path(anchor or ".")
    try:
        relative = pattern.relative_to(base)
    except ValueError:  # pragma: no cover - defensive
        return []
    try:
        return sorted(base.glob(str(relative)))
    except (OSError, ValueError):  # pragma: no cover - malformed pattern
        return []


def _format_of(path: Path) -> str:
    suffix = path.suffix.lower()
    return "jsonl" if suffix == ".jsonl" else "json" if suffix == ".json" else "auto"


def _parse_jsonl(text: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped == "[":
            continue
        stripped = stripped.rstrip(",")
        if stripped in {"]", "{"}:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append({"code": "METRICS_FILE_INVALID", "message": f"{path.name}:{number}: {exc}"})
            continue
        if isinstance(row, dict):
            points.append(_flatten_row(row))
    return points, errors


def _parse_json(text: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], [{"code": "METRICS_FILE_INVALID", "message": f"{path.name}: {exc}"}]
    if isinstance(data, dict):
        # HuggingFace Trainer writes {"epoch": [...], "loss": [...]} column-wise, but a
        # single-run summary also looks like that: {"loss": 0.3, "step": 100} is one row,
        # not two. Only treat the document as column-wise when *every* value is a list and
        # the lists are long enough to be a series.
        if _is_columnar(data):
            length = max(len(values) for values in data.values())
            rows: list[dict[str, Any]] = []
            for index in range(length):
                row = {key: values[index] for key, values in data.items() if index < len(values)}
                if row:
                    rows.append(_flatten_row({**row, "step": index}))
            if rows:
                return rows, []
        return [_flatten_row(data)], []
    if isinstance(data, list):
        return [_flatten_row(row) for row in data if isinstance(row, dict)], []
    return [], [{"code": "METRICS_FILE_INVALID", "message": f"{path.name}: unsupported JSON shape"}]


#: A document shorter than this is a single result, not a series, however list-shaped it is.
_MIN_COLUMNAR_ROWS = 2


def _is_columnar(data: dict[str, Any]) -> bool:
    """True only for a ``{name: [v, v, v]}`` series document.

    Everything else — a dict of scalars, or a dict mixing scalars and lists — is one row.
    Getting this wrong previously turned ``{"loss": 0.3, "notes": ["x"]}`` into a
    fabricated row whose ``loss`` was ``None``.
    """
    if not data or not all(isinstance(value, list) for value in data.values()):
        return False
    return max((len(value) for value in data.values()), default=0) >= _MIN_COLUMNAR_ROWS


def _flatten_row(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_row(value, prefix=f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
        elif isinstance(value, str) and _FLOAT_ONLY.match(value.strip()):
            flat[name] = float(value)
    if "step" not in flat and "epoch" in flat:
        flat["step"] = flat["epoch"]
    return flat


def _parse_key_value_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        values: dict[str, Any] = {}
        for match in _KEY_VALUE_RE.finditer(line):
            key = match.group("key").strip().lower()
            if key in NOISY_KEYS or len(key) > 40:
                continue
            number = _coerce_number(match.group("value"))
            if number is not None:
                values[key] = number
        if values:
            values["step"] = index
            points.append(values)
    return points


def _metric_keys(points: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for point in points:
        for key in point:
            if key in {"step", "timestamp", "time", "epoch"} or not _is_number(point[key]):
                continue
            if key not in seen:
                seen.append(key)
    return seen


def _pick_primary(keys: Sequence[str]) -> str | None:
    preferred = ("accuracy", "acc", "f1", "bleu", "rouge", "map", "mAP", "reward", "similarity")
    for name in preferred:
        for key in keys:
            if key.lower() == name.lower():
                return key
    for key in keys:
        if "loss" in key.lower():
            continue
        return key
    return keys[-1] if keys else None


def _direction_for(key: str, config: LabboxConfig) -> MetricDirection:
    """Which way is better for this metric, for the purpose of tracking its best value.

    ``best`` is only a convenience, so the configured direction wins when the user named
    this metric explicitly. Otherwise the name decides. The token list is shared with the
    comparison engine so "best" in ``metrics.json`` and "better" in ``compare`` can never
    disagree.
    """
    from labbox.analysis.diff import _direction_from_name

    if config.metrics.primary == key and config.metrics.direction:
        return MetricDirection(config.metrics.direction)
    guessed = _direction_from_name(key)
    if guessed is None:
        return MetricDirection.MAXIMIZE
    return MetricDirection(guessed)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number_or_none(value: Any) -> int | float | None:
    return value if _is_number(value) else None


def _coerce_number(text: str) -> float | None:
    cleaned = text.strip().rstrip("%").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first_number(match: re.Match[str]) -> float | None:
    """The captured number from a user rule.

    A named ``value`` group is honoured first (that is what the docs ask for), then any
    other capturing group. A pattern with *no* capturing group is common in the wild and
    used to raise ``IndexError`` inside the ingest loop, which the run service turns into
    a stage failure — one bad regex silently cost the run all of its metrics.
    """
    groups = match.groups()
    if not groups:
        return None
    named = match.groupdict().get("value")
    for candidate in (named, *groups):
        if candidate is None:
            continue
        number = _coerce_number(candidate)
        if number is not None:
            return number
    return None


def _compile(pattern: str) -> re.Pattern[str] | None:
    """Compile a user rule. A bad regex is dropped, not fatal (§16: config is untrusted)."""
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _relative(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return str(path)


def _write_history(points: Sequence[dict[str, Any]], history_path: Path) -> bool:
    """Write the convenience copy; report whether it landed.

    The history file is a convenience copy and ``metrics.json`` already carries the
    result, so a failure here must not fail the run — but the caller has to know, because
    an advertised ``series_ref`` pointing at a file that was never written is a broken link
    in exactly the place the product promises completeness.
    """
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", encoding="utf-8", newline="\n") as fh:
            for point in points:
                fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        return False
    return True
