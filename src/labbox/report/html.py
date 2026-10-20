"""The offline, single-file HTML report (§13).

Constraints from DEVELOPMENT.md §13 and §16 that shaped every choice here:

* One self-contained document. The stylesheet is inlined, charts are inline SVG built in
  Python below, and the output contains no remote font, script, image, frame, stylesheet
  or tracking reference of any kind -- not even an SVG ``xmlns`` URL -- so a report opened
  with no network still shows everything it knows.
* Every byte of user-controlled data reaches the template as plain text and is escaped by
  Jinja2 autoescape; nothing is marked safe except markup this module generates itself.
* A section whose source file is absent says so explicitly. An interrupted run legally
  has fewer files, and a silent gap would read as "nothing was wrong".
* :func:`render_report` never raises because a section's data is odd: it degrades to a
  warning block plus an ``<!-- labbox report: section X unavailable -->`` comment. Whether
  a failure is worth reporting to the user is the caller's decision, because a page that
  fails to build must never touch run data (§13).
"""

from __future__ import annotations

import json
import re
import shlex
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, Template
from markupsafe import Markup, escape

from labbox import __version__ as LABBOX_VERSION
from labbox.domain.artifacts import ArtifactManifest
from labbox.domain.common import now_local
from labbox.domain.dataset import DatasetManifest
from labbox.domain.diagnostics import CompareResult, DiagnosticsFile, DiffKind, Severity
from labbox.domain.environment import EnvironmentSnapshot
from labbox.domain.explain import ExplainResult
from labbox.domain.metrics import MetricSummary, SamplePoint
from labbox.domain.run import RunRecord, RunStatus
from labbox.errors import REPORT_GENERATION_FAILED, LabBoxError
from labbox.infrastructure.redact import Redactor
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import RunPaths

#: Why each file can be missing. Missing is a fact about the run, not an error (§24).
NOT_CAPTION = {
    "run.json": "The run index itself is missing; this report can only show what remains.",
    "environment.json": "The environment collector did not run, or the run ended before it did.",
    "metrics.json": (
        "No metrics file was written. Missing metrics are recorded as missing: LabBox never "
        "infers a number from log text."
    ),
    "dataset-manifest.json": "No dataset was declared for this run.",
    "artifacts.json": "The artifact collector did not run, or found nothing to record.",
    "diagnostics.json": "The rule engine produced no diagnostics file for this run.",
    "system.jsonl": "The hardware sampler recorded no rows.",
}

#: Written where a section existed but its data could not be used.
UNAVAILABLE_COMMENT = "<!-- labbox report: section {name} unavailable -->"

_MAX_ROWS = 200
_MAX_CELL_CHARS = 400
_MAX_CHART_POINTS = 600
_MAX_LIST_ITEMS = 60

_REDATOR = Redactor()
_HTTP_SCHEME_RE = re.compile(r"(?i)\bhtt(p?s?)://")
_CLASS_UNSAFE_RE = re.compile(r"[^a-z0-9_.-]")

# Chart palette: distinct on white, distinct in greyscale print (dashed strokes carry the
# second and third series), and never relying on colour alone -- the legend names each one.
_GPU_COLORS = ("#1b6ec2", "#8250df", "#0e7c6b", "#9a6700", "#6639ba")
_CPU_COLOR = "#216e39"
_RAM_COLOR = "#57606a"
_TEMP_COLOR = "#b3261e"

_SVG_WIDTH = 880.0
_PANEL_HEIGHT = 170.0
_PAD_LEFT = 56.0
_PAD_RIGHT = 14.0
_PAD_TOP = 14.0
_AXIS_BAND = 26.0

_SEVERITY_KIND = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.WARNING: "warn",
    Severity.INFO: "neutral",
}
_DIFF_KIND = {
    DiffKind.CHANGED: "error",
    DiffKind.ADDED: "warn",
    DiffKind.REMOVED: "warn",
    DiffKind.NUMERIC_DELTA: "warn",
    DiffKind.METRIC_DELTA: "neutral",
    DiffKind.UNKNOWN: "missing",
    DiffKind.INCOMPARABLE: "missing",
    DiffKind.EQUAL: "ok",
}
_STATE_LABEL = {
    "ok": "captured",
    "partial": "captured, with gaps",
    "not_captured": "not captured",
    "unavailable": "unreadable",
}
_STATE_KIND = {"ok": "ok", "partial": "warn", "not_captured": "missing", "unavailable": "error"}


# ---------------------------------------------------------------------------
# text and value formatting
# ---------------------------------------------------------------------------
def _class_slug(value: Any) -> str:
    return _CLASS_UNSAFE_RE.sub("-", str(value).strip().lower()) or "none"


def _defang(text: str) -> str:
    """Break URL schemes so this document can never carry a live link (§16)."""
    return _HTTP_SCHEME_RE.sub(lambda m: f"hxx{m.group(1)}://", text)


def _text(value: Any, *, fallback: str = "not captured") -> str:
    """Turn user data into display text: scrub secret shapes, defang, clip, never empty."""
    if value is None:
        return fallback
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, datetime):
        return _dt(value)
    if isinstance(value, (list, tuple, set)):
        joined = ", ".join(_text(item, fallback="") for item in value)
        return joined or fallback
    if isinstance(value, dict):
        joined = "; ".join(f"{_text(k, fallback='')}={_text(v, fallback='')}" for k, v in value.items())
        return joined or fallback
    raw = _defang(_REDATOR.scrub_text(str(value)))
    if len(raw) > _MAX_CELL_CHARS:
        raw = raw[:_MAX_CELL_CHARS] + f"... [{len(raw) - _MAX_CELL_CHARS} more chars]"
    return raw if raw.strip() else fallback


def _dt(value: Any, *, fallback: str = "not captured") -> str:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return _text(value)


def _num(value: Any, digits: int = 6, *, fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if number != number or number in (float("inf"), float("-inf")):
        return str(number)
    return f"{number:.{digits}g}"


def _bytes(value: Any, *, fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    try:
        size = int(value)
    except (TypeError, ValueError):
        return _text(value)
    unit_index = min(size.bit_length() // 10, 5)
    unit = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")[unit_index]
    if unit_index == 0:
        return f"{size:,} bytes"
    scaled = size / float(1 << (10 * unit_index))
    return f"{scaled:.4g} {unit} ({size:,} bytes)"


def _duration(seconds: Any) -> str:
    if seconds is None:
        return "not captured"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return _text(seconds)
    if total < 0:
        return f"{_num(total)} s (negative: the clock moved backwards)"
    if total < 60:
        return f"{total:.4g} s"
    hours, rest = divmod(int(total), 3600)
    minutes, secs = divmod(rest, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
        parts.append(f"{minutes:02d}m")
    elif minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts) + f" ({_num(total)} s)"


def _code(value: Any) -> Markup:
    """An inline ``<code>`` fragment; the payload inside is escaped right here."""
    return Markup("<code>{0}</code>").format(escape(_text(value, fallback="")))


def _badge(text: Any, kind: str = "neutral") -> Markup:
    return Markup('<span class="badge b-{0}">{1}</span>').format(_class_slug(kind), escape(_text(text)))


def _state_badge(text: Any, kind: str = "neutral") -> Markup:
    return Markup('<span class="state s-{0}">{1}</span>').format(_class_slug(kind), escape(_text(text)))


_EXIT_FIDELITY_NOTE = (
    "run.json stores both the status the operating system reported and the value a shell "
    "would have shown. On Windows Ctrl+C is 0xC000013A where POSIX reports 130, so both "
    "numbers are printed here: claiming the exit code survives is only honest if the raw "
    "value stays visible."
)


# ---------------------------------------------------------------------------
# content blocks: Python decides content, the template decides markup
# ---------------------------------------------------------------------------
def _banner(text: Any, kind: str = "warn") -> dict[str, Any]:
    return {"kind": "banner", "text": _text(text), "cls": _class_slug(kind)}


def _note(text: Any) -> dict[str, Any]:
    return {"kind": "note", "text": _text(text)}


def _para(text: Any) -> dict[str, Any]:
    return {"kind": "para", "text": _text(text)}


def _sub(text: Any) -> dict[str, Any]:
    return {"kind": "sub", "text": _text(text)}


def _score(text: Any) -> dict[str, Any]:
    return {"kind": "score", "text": _text(text)}


def _cell(value: Any) -> Markup | str:
    if isinstance(value, Markup):
        return value
    return _text(value)


def _pairs(items: list[tuple[Any, Any]]) -> dict[str, Any]:
    return {
        "kind": "kv",
        "rows": [[_text(key, fallback=""), _cell(value)] for key, value in items],
    }


def _more_text(hidden: int, noun: str) -> str:
    return f"{hidden} further {noun} omitted from this view" if hidden > 0 else ""


def _table(head: list[str], rows: list[list[Any]], *, caption: str = "") -> dict[str, Any]:
    shown = rows[:_MAX_ROWS]
    return {
        "kind": "table",
        "head": [_text(h, fallback="") for h in head],
        "rows": [[_cell(value) for value in row] for row in shown],
        "caption": _text(caption) if caption else "",
        "more": _more_text(len(rows) - len(shown), "rows are"),
    }


def _items(values: list[Any], *, limit: int = _MAX_LIST_ITEMS) -> dict[str, Any]:
    shown = [_text(v) for v in values[:limit]]
    return {
        "kind": "list",
        "items": shown,
        "empty": not shown,
        "more": _more_text(len(values) - len(shown), "entries are"),
    }


def _chips(values: list[Any], kind: str = "neutral") -> dict[str, Any]:
    shown = [_badge(v, kind) for v in values[:_MAX_ROWS]]
    return {
        "kind": "chips",
        "items": shown,
        "empty": not shown,
        "more": _more_text(len(values) - len(shown), "chips are"),
    }


def _tokens(values: list[Any]) -> dict[str, Any]:
    """argv rendered one quoted token per box: token boundaries stay visible."""
    shown = [_text(shlex.quote(str(value))) for value in values[:_MAX_ROWS]]
    return {
        "kind": "tokens",
        "items": shown,
        "empty": not shown,
        "more": _more_text(len(values) - len(shown), "arguments are"),
    }


# ---------------------------------------------------------------------------
# source loading: a missing or corrupt file is data, never an exception
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Source:
    name: str
    present: bool = False
    data: dict[str, Any] | None = None
    note: str = ""

    @property
    def missing(self) -> bool:
        return self.data is None

    @property
    def corrupt(self) -> bool:
        """A file that exists but could not be read: different from never captured."""
        return self.present and self.data is None

    @property
    def caption(self) -> str:
        return self.note or NOT_CAPTION.get(self.name, "not present in the run directory")


def _read_json(store: FilesystemStore, path: Path) -> _Source:
    source = _Source(path.name)
    try:
        source.present = path.is_file()
    except OSError as exc:  # pragma: no cover - unreadable run directory
        source.note = f"the run directory could not be inspected ({type(exc).__name__})"
        return source
    if not source.present:
        source.note = NOT_CAPTION.get(path.name, "not present in the run directory")
        return source
    try:
        data = store.read_json(path, strict=False)
    except Exception as exc:  # any read failure degrades this section only
        source.note = f"present but unreadable ({type(exc).__name__}: {_text(exc)})"
        return source
    if data is None:
        source.note = "present but empty or not valid JSON: the write may have been interrupted"
        return source
    if list(data) == ["value"]:
        # The store wraps a JSON document whose root is not an object. No schema has a single
        # "value" field, so treating it as an empty-but-valid file would be a blank illusion.
        source.note = "present, but its JSON root is not an object, so no fields can be read"
        return source
    source.data = data
    return source


def _read_samples(path: Path) -> tuple[list[SamplePoint], int, int, str]:
    """Parse ``samples/system.jsonl`` line by line; one bad row never loses the chart."""
    points: list[SamplePoint] = []
    lines = 0
    skipped = 0
    try:
        if not path.is_file():
            return points, 0, 0, ""
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # a locked or truncated sample file degrades the chart
        return points, 0, 0, f"{path.name} is present but unreadable ({type(exc).__name__})"
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        lines += 1
        try:
            row = json.loads(stripped)
        except ValueError:  # a torn write leaves a partial line; count it, keep going
            skipped += 1
            continue
        if not isinstance(row, dict):
            skipped += 1
            continue
        try:
            points.append(SamplePoint.model_validate(row))
        except Exception:  # a malformed row is skipped, its neighbours survive
            skipped += 1
    points.sort(key=lambda point: point.elapsed_seconds)
    return points, lines, skipped, ""


@dataclass(slots=True)
class _Inputs:
    paths: RunPaths
    run: _Source
    environment: _Source
    metrics: _Source
    datasets: _Source
    artifacts: _Source
    diagnostics: _Source
    samples: list[SamplePoint] = field(default_factory=list)
    sample_lines: int = 0
    sample_skipped: int = 0
    sample_note: str = ""
    compare: CompareResult | None = None
    explanation: ExplainResult | None = None
    generated_at: datetime = field(default_factory=now_local)

    @property
    def record(self) -> RunRecord | None:
        if self.run.data is None:
            return None
        try:
            return RunRecord.model_validate(self.run.data)
        except Exception:  # an unreadable index degrades the header, not the whole report
            return None

    @property
    def has_samples(self) -> bool:
        return bool(self.samples)


def _collect(store: FilesystemStore, paths: RunPaths) -> _Inputs:
    run_dir = paths.dir
    points, lines, skipped, note = _read_samples(paths.system_samples)
    return _Inputs(
        paths=paths,
        run=_read_json(store, run_dir / RunPaths.run),
        environment=_read_json(store, run_dir / RunPaths.environment),
        metrics=_read_json(store, run_dir / RunPaths.metrics),
        datasets=_read_json(store, run_dir / RunPaths.dataset_manifest),
        artifacts=_read_json(store, run_dir / RunPaths.artifacts),
        diagnostics=_read_json(store, run_dir / RunPaths.diagnostics),
        samples=points,
        sample_lines=lines,
        sample_skipped=skipped,
        sample_note=note,
    )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Section:
    key: str
    title: str
    state: str  # ok | partial | not_captured | unavailable
    blocks: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    source: str = ""


def _not_captured(key: str, title: str, source: _Source) -> Section:
    return Section(
        key=key,
        title=title,
        state="not_captured",
        source=source.name,
        note=source.caption,
        blocks=[
            _banner(f"{source.name}: not captured", "missing"),
            _para(source.caption),
            _note(
                "Reported as absent rather than left blank: for this run the fact is that the "
                "data was never captured, and no value below invents one."
            ),
        ],
    )


def _unavailable(key: str, title: str, source: str, detail: str) -> Section:
    return Section(
        key=key,
        title=title,
        state="unavailable",
        source=source,
        note=detail,
        blocks=[
            _banner(f"{source}: present but unusable", "error"),
            _para(detail),
            _note(
                "The section stays in place on purpose. An unreadable file is reported, never "
                "silently dropped, because dropping it would look like a run with no data gap."
            ),
        ],
    )


def _absent_or_unusable(key: str, title: str, source: _Source, detail: str = "") -> Section:
    """Say *which* of the two happened: never written, or written and not usable."""
    if source.corrupt:
        return _unavailable(key, title, source.name, source.caption)
    if source.data is not None:
        return _unavailable(
            key,
            title,
            source.name,
            detail
            or f"{source.name} is present but does not validate as this "
            "section's data, so nothing here is claimed about it",
        )
    return _not_captured(key, title, source)


def _section_summary(inputs: _Inputs) -> Section:
    run = inputs.record
    if run is None:
        unusable = inputs.run.corrupt or inputs.run.data is not None
        state = "unavailable" if unusable else "not_captured"
        caption = (
            inputs.run.caption
            if state == "not_captured"
            else ("run.json is present but cannot be read as a run record, so the header below is all that survives")
        )
        blocks = [
            _banner(
                f"run.json: {_STATE_LABEL[state]}",
                "error" if state == "unavailable" else "missing",
            ),
            _para(caption),
            _pairs([("run directory", _code(inputs.paths.dir.name))]),
            _note(
                "The run index is the only source for status and exit code; nothing is "
                "reconstructed from the remaining files."
            ),
        ]
        return Section("summary", "Run summary", state, blocks, caption, RunPaths.run)

    status_kind = {
        RunStatus.SUCCEEDED: "ok",
        RunStatus.RUNNING: "neutral",
        RunStatus.FAILED: "error",
        RunStatus.INTERRUPTED: "warn",
        RunStatus.START_FAILED: "error",
    }[run.status]
    raw_cell: Any
    if run.exit_code_raw is None:
        raw_cell = _badge("not reported (the process may never have started)", "missing")
    else:
        unsigned = run.exit_code_raw & 0xFFFFFFFF if run.exit_code_raw < 0 else run.exit_code_raw
        raw_cell = (
            f"{run.exit_code_raw} (0x{unsigned:08X})"
            if unsigned >= 0x80000000 or abs(run.exit_code_raw) >= 256
            else str(run.exit_code_raw)
        )
    blocks = [
        _pairs(
            [
                ("run id", _code(run.run_id)),
                ("name", _text(run.name)),
                ("project", _text(run.project)),
                ("status", _badge(run.status.value, status_kind)),
                ("started", _dt(run.started_at)),
                ("ended", _dt(run.ended_at)),
                ("duration", _duration(run.duration_seconds)),
                (
                    "exit code (as a shell would report)",
                    "not captured" if run.exit_code is None else str(run.exit_code),
                ),
                ("exit code (raw, as the OS reported)", raw_cell),
                ("exit meaning", _text(run.exit_meaning.value)),
                ("interrupted", _badge("yes", "warn") if run.interrupted else _text("no")),
                ("forced kill", _text(run.forced_kill)),
                ("killed by LabBox", _text(run.killed_by_labbox)),
                ("LabBox version", _text(run.labbox_version or LABBOX_VERSION)),
                ("schema version", _text(run.schema_version)),
            ]
        ),
        _note(_EXIT_FIDELITY_NOTE),
        _sub("Tags"),
        _chips(list(run.tags)) if run.tags else _note("No tags were recorded for this run."),
        _sub("Files recorded in run.json links"),
        _chips(sorted(run.links)) if run.links else _note("run.json records no links."),
    ]
    if run.schema_note:
        blocks.append(_note(f"schema note: {run.schema_note}"))
    if run.duration_seconds is None and run.ended_at is None:
        blocks.insert(
            1,
            _banner(
                "This run has no end timestamp: it was interrupted, crashed, or is still recorded as running.",
                "warn",
            ),
        )
    return Section("summary", "Run summary", "ok", blocks, "", RunPaths.run)


def _section_command(inputs: _Inputs) -> Section:
    run = inputs.record
    if run is None:
        return _absent_or_unusable("command", "Command", inputs.run)
    command = run.command
    tokens = list(command.argv or [])
    blocks: list[dict[str, Any]] = [
        _pairs(
            [
                ("command line as typed", _code(command.raw)),
                ("working directory", _code(command.cwd)),
                ("cwd relative to the project", _text(command.cwd_is_relative)),
                ("how it was started", _text(command.shell_note)),
            ]
        ),
        _sub("argv, one quoted token per box"),
        _tokens(tokens),
        _note(
            f"{len(tokens)} argv entries. The quoting shown here is POSIX rendering, chosen so "
            "token boundaries and spaces inside arguments stay visible on paper; the recorded argv "
            "list is authoritative and LabBox never re-joins it into a shell string (§24)."
        ),
    ]
    if not tokens:
        blocks.insert(1, _banner("The argv list is empty, so the command cannot be replayed.", "error"))
    if command.cwd_is_relative:
        blocks.insert(1, _note("cwd was recorded relative to the project root."))
    state = "ok" if tokens else "partial"
    return Section("command", "Command", state, blocks, "", RunPaths.run)


def _collected_cell(collected: Any) -> Markup | str:
    """``unknown`` and ``not set`` are different facts; keep them different on the page."""
    status = getattr(collected, "status", None)
    value = getattr(collected, "value", None)
    if status is not None and not getattr(collected, "ok", True):
        return _badge(f"not captured ({_text(getattr(status, 'value', status))})", "missing")
    if value is None:
        return _badge("not set", "neutral")
    detail = getattr(collected, "detail", None)
    return _text(f"{_text(value)} ({_text(detail)})" if detail else _text(value))


def _collector_rows(collectors: dict[str, str]) -> list[list[Any]]:
    return [
        [
            _code(name),
            _state_badge(status, _status_kind(status)),
            _text(_collector_meaning(name)),
        ]
        for name, status in sorted(collectors.items())
    ]


def _collector_meaning(name: str) -> str:
    """The collector key is machine vocabulary; say what it covered in plain words."""
    return {
        "git": "repository state at start",
        "gpu": "GPU inventory and NVML",
        "host": "CPU, cores and memory",
        "python_packages": "interpreter, frameworks and packages",
        "cuda": "CUDA and driver versions",
        "env": "sanitized environment variables",
        "seeds": "randomness settings",
    }.get(name, "")


def _status_kind(status: Any) -> str:
    return {
        "ok": "ok",
        "partial": "warn",
        "unavailable": "missing",
        "timeout": "error",
        "failed": "error",
        "skipped": "neutral",
        "disabled": "neutral",
        "not_captured": "missing",
        "not_applicable": "neutral",
    }.get(_class_slug(status), "warn")


def _section_environment(inputs: _Inputs) -> Section:
    source = inputs.environment
    if source.missing:
        return (
            _unavailable("environment", "Environment", source.name, source.caption)
            if source.corrupt
            else _not_captured("environment", "Environment", source)
        )
    try:
        env = EnvironmentSnapshot.model_validate(source.data)
    except Exception as exc:  # schema drift or a hand-edited file: report it, never guess
        return _unavailable("environment", "Environment", source.name, f"schema validation failed: {_text(exc)}")

    blocks: list[dict[str, Any]] = [
        _pairs(
            [
                ("captured at", _dt(env.captured_at)),
                ("capture phase", _text(env.capture_phase)),
                ("os", _text(f"{env.os.name} {env.os.version} ({env.os.arch})")),
                ("platform", _text(env.os.platform)),
                (
                    "text encodings",
                    _text(f"console {env.os.console_encoding or 'n/a'}, filesystem {env.os.fs_encoding or 'n/a'}"),
                ),
                ("python", _text(env.python.version or "unknown")),
                ("interpreter", _code(env.python.executable)),
                ("interpreter kind", _text(env.python.kind)),
                ("venv prefix", _text(env.python.venv_prefix)),
                ("probe return code", _text(env.python.probe_returncode)),
                ("package source", _text(env.package_source)),
                ("timezone / user", _text(f"{env.os.timezone or 'n/a'} / {env.os.user or 'n/a'}")),
            ]
        ),
        _sub("Collector status"),
    ]
    blocks.append(
        _table(["collector", "status", "what it covered"], _collector_rows(env.collectors))
        if env.collectors
        else _banner(
            "No per-collector status was recorded. The absence of a row is not a claim that every collector succeeded.",
            "warn",
        )
    )
    blocks.append(_sub("Recorded as absent"))
    blocks.append(_chips(list(env.absent), "missing") if env.absent else _note("Nothing was recorded as absent."))
    blocks.append(
        _note(
            '"absent" means a collector did not produce that field. It is not a claim '
            "that the value is empty, and comparisons treat it as unknown."
        )
    )

    framework_rows = [[name, _text(version or "unknown")] for name, version in sorted(env.frameworks.items())]
    blocks.append(_sub("Frameworks"))
    blocks.append(
        _table(["package", "version"], framework_rows)
        if framework_rows
        else _note("No framework versions were probed.")
    )
    blocks.append(_sub("Packages"))
    blocks.append(
        _table(["package", "version"], [[name, _text(version)] for name, version in sorted(env.packages.items())])
        if env.packages
        else _note("No package list was captured.")
    )

    blocks.append(_sub("CUDA and drivers"))
    blocks.append(
        _pairs(
            [
                ("CUDA runtime", _text(env.cuda.runtime)),
                ("driver", _text(env.cuda.driver)),
                ("cuDNN", _text(env.cuda.cudnn)),
                ("NVML", _text(env.cuda.nvml)),
                ("toolkit from env", _text(env.cuda.toolkit_env)),
            ]
        )
    )
    blocks.append(_sub("Git"))
    blocks.append(
        _pairs(
            [
                ("repository found", _text(env.git.has_repo)),
                ("commit", _code(env.git.commit)),
                ("short commit", _text(env.git.short_commit)),
                ("branch", _text(env.git.branch)),
                ("dirty", _text(env.git.dirty)),
                ("diff hash", _text(env.git.diff_hash)),
                ("modified / untracked", _text(f"{env.git.modified_count} / {env.git.untracked_count}")),
                ("remote (credentials stripped)", _text(env.git.remote_origin)),
                ("git version", _text(env.git.git_version)),
            ]
        )
    )
    blocks.append(_sub("Host"))
    blocks.append(
        _pairs(
            [
                ("cpu", _text(env.host.cpu_model)),
                ("cores physical / logical", _text(f"{env.host.physical_cores} / {env.host.logical_cores}")),
                ("memory", _bytes(env.host.ram_bytes)),
                ("memory available", _bytes(env.host.ram_available_bytes)),
                ("load average", _text(env.host.load_avg)),
            ]
        )
    )
    blocks.append(_sub("GPUs"))
    blocks.append(
        _table(
            ["index", "name", "vram", "vram in use", "driver", "compute capability", "persistence"],
            [
                [
                    gpu.index,
                    _text(gpu.name),
                    _bytes(gpu.vram_bytes),
                    _bytes(gpu.vram_used_bytes),
                    _text(gpu.driver),
                    _text(gpu.compute_capability),
                    _text(gpu.persistence_mode),
                ]
                for gpu in env.gpus
            ],
        )
        if env.gpus
        else _note("No GPU was recorded for this run.")
    )
    blocks.append(_sub("Interpreter probe"))
    blocks.append(_pairs([("probe result", _collected_cell(env.interpreter_probe))]))

    blocks.append(_sub("Environment variables"))
    env_rows: list[list[Any]] = [
        [name, _code(value), _text("value was captured (and scrubbed)")]
        for name, value in sorted(env.env.entries.items())
    ]
    env_rows += [
        [
            name,
            _badge("redacted", "missing"),
            _text("the name matched a secret pattern; the value was never stored, so it cannot be shown"),
        ]
        for name in sorted(env.env.redacted_names)
    ]
    blocks.append(
        _table(["name", "value", "capture note"], env_rows)
        if env_rows
        else _note("No environment variables were captured.")
    )
    blocks.append(
        _note(
            f"{env.env.total_count or (len(env.env.entries) + len(env.env.redacted_names))} variables "
            f"existed at capture time: {len(env.env.entries)} are shown, "
            f"{len(env.env.redacted_names)} were redacted by name pattern. Redaction is by name, not "
            "by value, and everything printed here additionally passed through the value scrubber."
        )
    )
    if env.env.redact_patterns:
        blocks.append(_sub("Redaction patterns in force"))
        blocks.append(_items(list(env.env.redact_patterns)))

    blocks.append(_sub("Randomness"))
    blocks.append(
        _pairs(
            [
                ("PYTHONHASHSEED", _text(env.seeds.python_hash_seed)),
                ("torch seed", _collected_cell(env.seeds.torch_seed)),
                ("numpy seed", _collected_cell(env.seeds.numpy_seed)),
                ("deterministic algorithms", _collected_cell(env.seeds.cuda_deterministic)),
                ("benchmark disabled", _collected_cell(env.seeds.benchmark_disabled)),
                ("declared on the command", _text(env.seeds.declared_in_command)),
                ("seed sources", _text(env.seeds.sources)),
            ]
        )
    )
    if env.seeds.warnings:
        blocks.append(_items(list(env.seeds.warnings)))
    if env.errors:
        blocks.append(_sub("Collector errors"))
        blocks.append(
            _items([_text(error) if not isinstance(error, dict) else _json_line(error) for error in env.errors])
        )

    degraded = list(env.absent) + [
        name for name, status in env.collectors.items() if _status_kind(status) in ("error", "missing")
    ]
    state = "partial" if degraded else "ok"
    return Section(
        "environment",
        "Environment",
        state,
        blocks,
        _more_text(len(degraded), "fields or collectors are"),
        RunPaths.environment,
    )


def _json_line(value: dict[str, Any]) -> str:
    try:
        return _text(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - pathological objects
        return _text(value)


def _section_metrics(inputs: _Inputs) -> Section:
    source = inputs.metrics
    if source.missing:
        section = (
            _unavailable("metrics", "Metrics", source.name, source.caption)
            if source.corrupt
            else _not_captured("metrics", "Metrics", source)
        )
        section.blocks.insert(0, _banner("METRICS_MISSING: no metric value was captured for this run", "missing"))
        return section
    try:
        metrics = MetricSummary.model_validate(source.data)
    except Exception as exc:  # schema drift or a hand-edited file: report it, never guess
        return _unavailable("metrics", "Metrics", source.name, f"schema validation failed: {_text(exc)}")

    blocks: list[dict[str, Any]] = []
    if metrics.metrics_missing or not metrics.final:
        blocks.append(_banner("METRICS_MISSING: the metrics source produced no value for this run", "missing"))
        blocks.append(
            _note(
                "Nothing below is inferred from log text. A missing metric stays "
                "missing, and comparisons report it as unknown."
            )
        )
    blocks.append(
        _pairs(
            [
                ("source", _text(metrics.source)),
                ("parser", _text(metrics.parser)),
                ("primary metric", _text(metrics.primary_metric)),
                ("direction", _text(metrics.direction.value)),
                ("sample points", _text(metrics.point_count)),
                ("series reference", _text(metrics.series_ref)),
                ("metrics_missing", _text(metrics.metrics_missing)),
            ]
        )
    )
    blocks.append(_sub("Final values"))
    blocks.append(
        _table(
            ["metric", "final", ""],
            [
                [name, _num(value), _badge("primary", "ok") if name == metrics.primary_metric else _text("")]
                for name, value in sorted((metrics.final or {}).items())
            ],
        )
        if metrics.final
        else _note("No final value was recorded.")
    )
    blocks.append(_sub("Best values"))
    blocks.append(
        _table(
            ["metric", "best", "step", "epoch", "at"],
            [
                [name, _num(best.value), _text(best.step), _text(best.epoch), _dt(best.timestamp)]
                for name, best in sorted(metrics.best.items())
            ],
        )
        if metrics.best
        else _note("No best-value tracking was recorded.")
    )
    if metrics.first:
        blocks.append(_sub("First values"))
        blocks.append(
            _table(["metric", "first"], [[name, _num(value)] for name, value in sorted(metrics.first.items())])
        )
    if metrics.warnings:
        blocks.append(_sub("Warnings"))
        blocks.append(_items(list(metrics.warnings)))
    if metrics.parse_errors:
        blocks.append(_sub("Parse errors"))
        blocks.append(
            _items([_json_line(error) if isinstance(error, dict) else _text(error) for error in metrics.parse_errors])
        )
    state = (
        "not_captured"
        if (metrics.metrics_missing or not metrics.final)
        else ("partial" if (metrics.warnings or metrics.parse_errors) else "ok")
    )
    return Section(
        "metrics",
        "Metrics",
        state,
        blocks,
        "metrics reported as missing" if state == "not_captured" else "",
        RunPaths.metrics,
    )


_PROOF_NOTE = (
    "The digest covers relative path + size + the sha256 of the first and last 64 bytes of "
    "each file. An identical digest therefore does not prove byte-identical content: a file "
    "edited in its unsampled middle keeps this digest unchanged. Modification times are "
    "deliberately excluded, because copying or re-checking out a tree changes every mtime and "
    "would make the data-changed diagnostic cry wolf."
)


def _section_datasets(inputs: _Inputs) -> Section:
    source = inputs.datasets
    if source.missing:
        return (
            _unavailable("datasets", "Dataset manifest", source.name, source.caption)
            if source.corrupt
            else _not_captured("datasets", "Dataset manifest", source)
        )
    try:
        manifest = DatasetManifest.model_validate(source.data)
    except Exception as exc:  # schema drift or a hand-edited file: report it, never guess
        return _unavailable("datasets", "Dataset manifest", source.name, f"schema validation failed: {_text(exc)}")

    blocks: list[dict[str, Any]] = [
        _pairs(
            [
                ("declared via", _text(manifest.discovery)),
                ("datasets", _text(len(manifest.datasets))),
            ]
        )
    ]
    if manifest.not_declared:
        blocks.append(
            _banner(
                "No dataset was declared, so nothing here can say the data was the same or different between runs.",
                "missing",
            )
        )
    for entry in manifest.datasets:
        proof = entry.is_content_proof
        blocks.append(_sub(f"Dataset: {_text(entry.id)}"))
        blocks.append(
            _pairs(
                [
                    ("id", _code(entry.id)),
                    ("source", _code(entry.source)),
                    ("kind", _text(entry.kind.value)),
                    ("strategy", _badge(entry.strategy.value, "ok" if proof else "warn")),
                    ("digest", _code(entry.digest)),
                    ("digest inputs", _text(entry.digest_inputs)),
                    ("files", _text(entry.file_count)),
                    ("total size", _bytes(entry.total_bytes)),
                    ("generated at", _dt(entry.generated_at)),
                    ("scan time", _duration(entry.elapsed_seconds)),
                    ("verified", _text(entry.verified)),
                    ("declared by user", _text(entry.declared_by_user)),
                    ("external version", _text(entry.external_version)),
                    ("zero-byte files", _text(entry.zero_byte_files)),
                    ("sampled only", _text(entry.sampled_only)),
                    ("exclusions", _text(entry.exclusions)),
                ]
            )
        )
        if not proof:
            blocks.append(_note(_PROOF_NOTE))
        if entry.strategy.value == "fast_manifest" and entry.digest is None:
            blocks.append(_banner("fast_manifest recorded no digest: the data version is unknown.", "error"))
        if entry.touched_files:
            blocks.append(
                _banner(
                    "Files may have been modified in place (same path and size, newer mtime): "
                    + ", ".join(_text(name) for name in entry.touched_files[:10]),
                    "warn",
                )
            )
        if entry.warnings:
            blocks.append(_items(list(entry.warnings)))
        if entry.errors:
            blocks.append(
                _items([_json_line(error) if isinstance(error, dict) else _text(error) for error in entry.errors])
            )
    if manifest.warnings:
        blocks.append(_sub("Manifest warnings"))
        blocks.append(_items(list(manifest.warnings)))
    return Section(
        "datasets",
        "Dataset manifest",
        _dataset_state(manifest),
        blocks,
        _dataset_note(manifest),
        RunPaths.dataset_manifest,
    )


def _dataset_state(manifest: DatasetManifest) -> str:
    """Whether the data side of the run was actually captured.

    The file being present says nothing about the data being known: a manifest with no
    entries means nothing was recorded about the dataset, which is a capture gap rather
    than a clean result.
    """
    if not manifest.datasets:
        return "not_captured"
    if any(entry.truncated or entry.errors or entry.strategy.value == "fast_manifest" for entry in manifest.datasets):
        return "partial"
    return "ok"


def _dataset_note(manifest: DatasetManifest) -> str:
    if not manifest.datasets:
        return "no dataset was recorded, so data identity is unknown for this run"
    damaged = [entry.id for entry in manifest.datasets if entry.truncated or entry.errors]
    if damaged:
        return "truncated or unreadable: " + ", ".join(_text(name) for name in damaged[:6])
    sampled = [entry.id for entry in manifest.datasets if not entry.is_content_proof]
    if sampled:
        return "sampled rather than hashed in full: " + ", ".join(_text(name) for name in sampled[:6])
    return ""


def _section_artifacts(inputs: _Inputs) -> Section:
    source = inputs.artifacts
    if source.missing:
        return (
            _unavailable("artifacts", "Artifacts", source.name, source.caption)
            if source.corrupt
            else _not_captured("artifacts", "Artifacts", source)
        )
    try:
        manifest = ArtifactManifest.model_validate(source.data)
    except Exception as exc:  # schema drift or a hand-edited file: report it, never guess
        return _unavailable("artifacts", "Artifacts", source.name, f"schema validation failed: {_text(exc)}")

    blocks: list[dict[str, Any]] = [
        _pairs(
            [
                ("artifacts", _text(len(manifest.artifacts))),
                ("total size", _bytes(manifest.total_bytes)),
                ("hash threshold", _bytes(manifest.hash_threshold_bytes)),
                ("scanned globs", _text(manifest.scanned_globs)),
                ("copied into the run directory", _text(manifest.copied)),
            ]
        ),
        _table(
            ["kind", "path", "size", "sha256", "hash state", "archived", "storage uri", "error"],
            [
                [
                    _text(entry.kind.value),
                    _code(entry.path),
                    _bytes(entry.size_bytes),
                    _text(entry.sha256),
                    _badge(entry.hash_state.value, "ok" if entry.hash_state.value == "computed" else "neutral"),
                    _text(entry.archived),
                    _text(entry.storage_uri),
                    _text(entry.error),
                ]
                for entry in manifest.artifacts
            ],
        )
        if manifest.artifacts
        else _banner(
            "The artifact list is empty: either nothing was declared or nothing matched the configured globs.",
            "missing",
        ),
        _note(
            "LabBox records where outputs live and does not copy large checkpoints (§7.1). A "
            "hash state of deferred_too_large means the digest was deliberately not computed, "
            "which is not the same as a failed hash."
        ),
    ]
    if manifest.warnings:
        blocks.append(_sub("Warnings"))
        blocks.append(_items(list(manifest.warnings)))
    if manifest.errors:
        blocks.append(_sub("Collection errors"))
        blocks.append(
            _items([_json_line(error) if isinstance(error, dict) else _text(error) for error in manifest.errors])
        )
    state = "partial" if (manifest.errors or manifest.warnings) else "ok"
    return Section("artifacts", "Artifacts", state, blocks, "", RunPaths.artifacts)


def _diagnostics_file(inputs: _Inputs) -> DiagnosticsFile | None:
    if inputs.diagnostics.missing:
        return None
    try:
        return DiagnosticsFile.model_validate(inputs.diagnostics.data)
    except Exception:  # each section that consumes this file reports its own failure
        return None


_REPRO_MEANING = {
    "code_locatable": "Can the exact source be found again?",
    "deps_reproducible": "Can the dependency set be rebuilt?",
    "data_verifiable": "Can the data version be verified?",
    "randomness_controlled": "Was randomness controlled?",
    "hardware_identified": "Is the hardware identified?",
    "command_replayable": "Can the command be replayed?",
    "artifacts_locatable": "Can outputs be located?",
}


def _section_reproducibility(inputs: _Inputs) -> Section:
    source = inputs.diagnostics
    if source.missing:
        return (
            _unavailable("reproducibility", "Reproducibility checklist", source.name, source.caption)
            if source.corrupt
            else _not_captured("reproducibility", "Reproducibility checklist", source)
        )
    model = _diagnostics_file(inputs)
    if model is None:
        return _unavailable(
            "reproducibility",
            "Reproducibility checklist",
            source.name,
            "diagnostics.json is present but does not validate as a diagnostics document, so no checklist can be shown",
        )
    body = model.reproducibility
    band_kind = {"high": "ok", "medium": "warn", "low": "error"}.get(body.band.value, "neutral")
    blocks = [
        _pairs(
            [
                ("band", _badge(body.band.value, band_kind)),
                ("rules version", _text(body.rules_version)),
            ]
        ),
        _score(f"score {body.score} / 100"),
        _note(
            "The band and the per-check reasons are the claim. The score above is a secondary "
            "risk hint computed from a versioned weight table, not a probability that the run "
            "reproduces, and it is shown small on purpose: 72 and 68 are not distinguishable "
            "claims."
        ),
        _sub("Checklist (primary output)"),
    ]
    blocks.append(
        _table(
            ["check", "state", "reason", "weight", "question it answers"],
            [
                [
                    _code(check.id),
                    _state_badge(
                        check.state.value,
                        {"ok": "ok", "partial": "warn", "missing": "missing", "unknown": "neutral"}.get(
                            check.state.value, "neutral"
                        ),
                    ),
                    _text(check.reason),
                    _text(check.weight),
                    _text(_REPRO_MEANING.get(check.id)),
                ]
                for check in body.checklist
            ],
        )
    )
    if not body.checklist:
        blocks.append(
            _banner(
                "The checklist is empty: no reproducibility judgement was computed. That is not a low score.",
                "warn",
            )
        )
    return Section(
        "reproducibility",
        "Reproducibility checklist",
        "ok" if body.checklist else "partial",
        blocks,
        "",
        RunPaths.diagnostics,
    )


def _section_diagnostics(inputs: _Inputs) -> Section:
    source = inputs.diagnostics
    if source.missing:
        if source.corrupt:
            return _unavailable("diagnostics", "Diagnostics", source.name, source.caption)
        return _not_captured("diagnostics", "Diagnostics", source)
    model = _diagnostics_file(inputs)
    if model is None:
        return _unavailable(
            "diagnostics", "Diagnostics", source.name, "diagnostics.json is present but does not validate"
        )
    blocks = [
        _pairs(
            [
                ("generated at", _dt(model.generated_at)),
                ("rules version", _text(model.rules_version)),
            ]
        )
    ]
    if not model.diagnostics:
        blocks.append(_banner("No diagnostic fired for this run", "ok"))
        blocks.append(
            _note(
                "An empty list means the rule engine ran at the version above and "
                "matched nothing. It is a statement about those rules, not a guarantee "
                "that the run is sound."
            )
        )
        return Section("diagnostics", "Diagnostics", "ok", blocks, "", RunPaths.diagnostics)
    blocks.append(
        _table(
            ["id", "severity", "code", "message", "evidence", "suggestion", "fact ids", "rule"],
            [
                [
                    _code(diag.id),
                    _badge(diag.severity.value, _SEVERITY_KIND.get(diag.severity, "neutral")),
                    _text(diag.code),
                    _text(diag.message),
                    _text(diag.evidence),
                    _text(diag.suggestion),
                    _text(diag.fact_ids),
                    _text(diag.rule),
                ]
                for diag in model.diagnostics
            ],
        )
    )
    worst = max((diag.severity.value for diag in model.diagnostics), key=_severity_rank, default="")
    # A valid diagnostics.json means the rule engine ran to the end of this version. How
    # severe its findings turned out to be says nothing about whether the file was captured,
    # so it belongs in the note and not in the capture state.
    return Section(
        "diagnostics",
        "Diagnostics",
        "ok",
        blocks,
        f"{len(model.diagnostics)} finding(s); worst severity: {worst}" if worst else "",
        RunPaths.diagnostics,
    )


def _severity_rank(value: str) -> int:
    return {"critical": 3, "high": 2, "warning": 1}.get(_class_slug(value), 0)


def _facts_table(title: str, facts: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [_sub(title)]
    if not facts:
        blocks.append(_note(f"{title}: none recorded."))
        return blocks
    blocks.append(
        _table(
            [
                "id",
                "path",
                "kind",
                "left",
                "right",
                "delta",
                "category",
                "sources",
                "confidence",
                "severity",
                "comparable",
                "unknown side",
                "reason",
            ],
            [
                [
                    _code(fact.id),
                    _text(fact.path),
                    _badge(fact.kind.value, _DIFF_KIND.get(fact.kind, "neutral")),
                    _text(fact.left_repr or fact.left),
                    _text(fact.right_repr or fact.right),
                    _text(fact.delta_repr) if fact.delta_repr else _num(fact.delta, fallback=""),
                    _text(fact.category),
                    _text(fact.sources),
                    _text(fact.confidence),
                    _text(fact.severity.value),
                    _text("" if fact.comparable else "not comparable"),
                    _text(fact.unknown_side),
                    _text(fact.reason),
                ]
                for fact in facts
            ],
        )
    )
    blocks.append(
        _note(
            "kind unknown means at least one side was never captured, so the two "
            "values cannot be compared at all; it is reported as unknown instead of "
            "being guessed equal. incomparable means the two sides are not even the "
            "same kind of fact. Both are printed with their own styling because they "
            "are the conclusions a reader most often misreads."
        )
    )
    return blocks


def _run_label(name: str, run_id: str) -> str:
    return f"{name or run_id} ({run_id})"


def _section_compare(inputs: _Inputs) -> Section | None:
    compare = inputs.compare
    if compare is None:
        return None
    blocks: list[dict[str, Any]] = []
    if compare.incomparable:
        blocks.append(
            _banner(
                "These runs are incomparable as recorded: at least one difference "
                "invalidates reading one result against the other.",
                "error",
            )
        )
    if compare.metrics_missing:
        blocks.append(
            _banner(
                "METRICS_MISSING on at least one side: metric deltas cannot be computed, only reported as unknown.",
                "missing",
            )
        )
    blocks.append(
        _pairs(
            [
                ("left run", _text(_run_label(compare.left_name, compare.left_run_id))),
                ("right run", _text(_run_label(compare.right_name, compare.right_run_id))),
                ("generated at", _dt(compare.generated_at)),
                ("facts other than unknown", _text(compare.changed_fact_count)),
                ("unknown count", _text(compare.unknown_count)),
            ]
        )
    )
    blocks += _facts_table("Changed facts", compare.facts)
    if compare.metric_deltas:
        blocks += _facts_table("Metric deltas", compare.metric_deltas)
    if compare.confounders:
        blocks.append(_sub("Confounders"))
        blocks.append(_items(list(compare.confounders)))
        blocks.append(
            _note(
                "A confounder is a change that co-varies with another: attribution "
                "between the listed facts is impossible from these two runs alone. "
                "LabBox reports that instead of picking a winner."
            )
        )
    if compare.unchanged:
        blocks.append(_sub("Checked and equal"))
        blocks.append(_chips(list(compare.unchanged), "ok"))
    if compare.diagnostics:
        blocks.append(_sub("Comparison diagnostics"))
        blocks.append(
            _table(
                ["severity", "code", "message", "evidence", "suggestion", "fact ids"],
                [
                    [
                        _badge(diag.severity.value, _SEVERITY_KIND.get(diag.severity, "neutral")),
                        _text(diag.code),
                        _text(diag.message),
                        _text(diag.evidence),
                        _text(diag.suggestion),
                        _text(diag.fact_ids),
                    ]
                    for diag in compare.diagnostics
                ],
            )
        )
    if compare.reproducibility is not None:
        blocks.append(_sub("Reproducibility, left then right"))
        blocks.append(
            _chips([f"band {item.band.value}, score {item.score} (secondary)" for item in compare.reproducibility])
        )
    return Section(
        "compare",
        "Changed facts (comparison)",
        "partial" if compare.incomparable else "ok",
        blocks,
        "requested via labbox compare",
        "compare request",
    )


def _known_fact_ids(inputs: _Inputs) -> set[str]:
    ids: set[str] = set()
    if inputs.compare is not None:
        ids |= {fact.id for fact in inputs.compare.facts}
        ids |= {fact.id for fact in inputs.compare.metric_deltas}
        ids |= {fid for diag in inputs.compare.diagnostics for fid in diag.fact_ids}
    model = _diagnostics_file(inputs)
    if model is not None:
        ids |= {fid for diag in model.diagnostics for fid in diag.fact_ids}
    return ids


def _section_explanation(inputs: _Inputs) -> Section | None:
    explanation = inputs.explanation
    if explanation is None:
        return None
    known = _known_fact_ids(inputs)
    blocks: list[dict[str, Any]] = [
        _banner(
            "Model-generated content. Everything below this line was written by an LLM "
            f"({_text(explanation.provider or 'unknown provider')} / "
            f"{_text(explanation.model or 'unknown model')}), not measured by LabBox. It is the "
            "least reliable part of this report, the report is complete and readable without it, "
            "and no remote request was made to produce this file.",
            "ai",
        ),
        _para(explanation.summary or "The model returned no summary."),
    ]
    if explanation.facts:
        rows = []
        for item in explanation.facts:
            if not known:
                check = _badge("no fact list to check against", "neutral")
            elif item.fact_id in known:
                check = _badge("id exists", "ok")
            else:
                check = _badge("unknown id", "error")
            rows.append([_code(item.fact_id), _text(item.interpretation), check])
        blocks.append(_sub("Interpretation of individual facts"))
        blocks.append(_table(["fact id", "model interpretation", "citation check"], rows))
        blocks.append(
            _note(
                "The citation check compares the ids the model quoted against the ids "
                "this report actually contains. An id the model invented is labelled, "
                "not silently dropped."
            )
        )
    if explanation.hypotheses:
        rows = [
            [
                _text(hypo.text),
                _badge(hypo.confidence, "warn" if hypo.confidence == "high" else "neutral"),
                _text(hypo.evidence_fact_ids),
                _text(hypo.validation),
            ]
            for hypo in explanation.hypotheses
        ]
        blocks.append(_sub("Hypotheses (inference, not fact)"))
        blocks.append(_table(["hypothesis", "model confidence", "evidence fact ids", "how to check"], rows))
        blocks.append(
            _note(
                "Model confidence is the model's self-assessment of its own guess. It "
                "is not a statistical measure and it is not LabBox severity."
            )
        )
    if explanation.unknowns:
        blocks.append(_sub("Evidence insufficient"))
        blocks.append(_items(list(explanation.unknowns)))
    if explanation.next_steps:
        blocks.append(_sub("Suggested next steps"))
        blocks.append(_items(list(explanation.next_steps)))
    if explanation.rejected_fact_ids:
        blocks.append(
            _banner(
                "Citations the model invented, already dropped by schema validation: "
                + ", ".join(_text(fid) for fid in explanation.rejected_fact_ids),
                "error",
            )
        )
    if explanation.validation_errors:
        blocks.append(_sub("Output validation errors"))
        blocks.append(_items(list(explanation.validation_errors)))
    return Section(
        "explanation", "AI explanation", "partial", blocks, "model-generated, not measured", "explain request"
    )


# ---------------------------------------------------------------------------
# hardware timeline: inline SVG generated here, so no script is ever involved
# ---------------------------------------------------------------------------
def _median_interval(points: list[SamplePoint]) -> float | None:
    deltas = [
        later.elapsed_seconds - earlier.elapsed_seconds
        for earlier, later in pairwise(points)
        if later.elapsed_seconds > earlier.elapsed_seconds
    ]
    return float(statistics.median(deltas)) if deltas else None


def _point_series(points: list[SamplePoint], getter: Any) -> list[tuple[float, float | None]]:
    out: list[tuple[float, float | None]] = []
    for point in points:
        value = getter(point)
        out.append((point.elapsed_seconds, None if value is None else float(value)))
    return out


def _gpu_series(points: list[SamplePoint], index: int, getter: Any) -> list[tuple[float, float | None]]:
    out: list[tuple[float, float | None]] = []
    for point in points:
        picked: float | None = None
        for gpu in point.gpus:
            if gpu.index == index and getter(gpu) is not None:
                picked = float(getter(gpu))
                break
        out.append((point.elapsed_seconds, picked))
    return out


def _series_values(series: list[tuple[float, float | None]]) -> list[float]:
    return [value for _x, value in series if value is not None]


def _stats_label(values: list[float]) -> str:
    if not values:
        return "no usable samples"
    return (
        f"n={len(values)}   min={_num(min(values), 4)}   "
        f"mean={_num(statistics.fmean(values), 4)}   max={_num(max(values), 4)}"
    )


def _stride(values: list[Any], limit: int = _MAX_CHART_POINTS) -> list[Any]:
    if len(values) <= limit:
        return values
    step = len(values) / limit
    picked = [values[int(index * step)] for index in range(limit)]
    if picked[-1] is not values[-1]:
        picked.append(values[-1])
    return picked


def _x_position(elapsed: float, x0: float, span: float) -> float:
    plot_width = _SVG_WIDTH - _PAD_LEFT - _PAD_RIGHT
    return _PAD_LEFT + (elapsed - x0) / span * plot_width


def _polylines(
    series: list[tuple[float, float | None]],
    *,
    x0: float,
    span: float,
    low: float,
    high: float,
    top: float,
    height: float,
) -> list[str]:
    """Point sets for one series; absent samples break the line instead of drawing zero."""
    segments: list[list[str]] = []
    current: list[str] = []
    for elapsed, value in _stride(series):
        if value is None:
            if current:
                segments.append(current)
                current = []
            continue
        clamped = min(max(value, low), high)
        y = top + height - (clamped - low) / ((high - low) or 1.0) * height
        current.append(f"{_x_position(elapsed, x0, span):.1f},{y:.1f}")
    if current:
        segments.append(current)
    return [" ".join(segment) for segment in segments]


def _axis_panel(top: float, height: float, low: float, high: float, unit: str) -> list[str]:
    elements = [
        f'<rect x="{_PAD_LEFT:.0f}" y="{top:.1f}" '
        f'width="{_SVG_WIDTH - _PAD_LEFT - _PAD_RIGHT:.0f}" height="{height:.1f}" '
        'fill="#ffffff" stroke="#d0d7de" stroke-width="1"/>'
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + (high - low) * fraction
        y = top + height - fraction * height
        elements.append(
            f'<line x1="{_PAD_LEFT:.0f}" y1="{y:.1f}" x2="{_SVG_WIDTH - _PAD_RIGHT:.0f}" '
            f'y2="{y:.1f}" stroke="#eaeef2" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{_PAD_LEFT - 6:.0f}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'class="tick">{_text(f"{value:.4g}")}</text>'
        )
    elements.append(
        f'<text x="{_PAD_LEFT - 6:.0f}" y="{top - 3:.1f}" text-anchor="end" class="axis">{_text(unit)}</text>'
    )
    return elements


def _x_axis(bottom: float, x0: float, span: float) -> list[str]:
    elements = [
        f'<line x1="{_PAD_LEFT:.0f}" y1="{bottom:.1f}" x2="{_SVG_WIDTH - _PAD_RIGHT:.0f}" '
        f'y2="{bottom:.1f}" stroke="#d0d7de" stroke-width="1"/>'
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = _PAD_LEFT + fraction * (_SVG_WIDTH - _PAD_LEFT - _PAD_RIGHT)
        label = f"{x0 + span * fraction:.0f} s"
        elements.append(
            f'<text x="{x:.1f}" y="{bottom + 15:.1f}" text-anchor="middle" class="tick">{_text(label)}</text>'
        )
    elements.append(
        f'<text x="{_SVG_WIDTH - _PAD_RIGHT:.0f}" y="{bottom + 15:.1f}" text-anchor="end" '
        'class="axis">seconds since the first sample</text>'
    )
    return elements


def _polyline(points: str, color: str, *, dashed: bool) -> str:
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"{dash}/>'


def _build_chart(points: list[SamplePoint]) -> dict[str, Any]:
    """One inline SVG: utilization in percent on top, GPU temperature on its own axis below."""
    percent_series: list[tuple[str, list[tuple[float, float | None]], str, bool]] = []
    gpu_indexes = sorted({gpu.index for point in points for gpu in point.gpus})
    for offset, index in enumerate(gpu_indexes[: len(_GPU_COLORS)]):
        series = _gpu_series(points, index, lambda gpu: gpu.utilization_percent)
        if _series_values(series):
            percent_series.append((f"GPU {index} utilization", series, _GPU_COLORS[offset], offset % 2 == 1))
    cpu = _point_series(points, lambda point: point.cpu_percent)
    if _series_values(cpu):
        percent_series.append(("CPU utilization", cpu, _CPU_COLOR, False))
    ram = _point_series(points, lambda point: point.ram_percent)
    if _series_values(ram):
        percent_series.append(("RAM used", ram, _RAM_COLOR, False))
    temp_series: list[tuple[str, list[tuple[float, float | None]], str, bool]] = []
    for offset, index in enumerate(gpu_indexes[:2]):
        series = _gpu_series(points, index, lambda gpu: gpu.temperature_c)
        if _series_values(series):
            temp_series.append((f"GPU {index} temperature", series, _TEMP_COLOR, offset % 2 == 1))
    if not percent_series and not temp_series:
        return {}

    elapsed = [point.elapsed_seconds for point in points]
    x0, x_max = min(elapsed), max(elapsed)
    span = (x_max - x0) or 1.0
    gap = _AXIS_BAND + 8.0 if (percent_series and temp_series) else 0.0
    height_one = _PANEL_HEIGHT if percent_series else 0.0
    height_two = _PANEL_HEIGHT if temp_series else 0.0
    total = _PAD_TOP + height_one + gap + height_two + _AXIS_BAND

    # No xmlns: in an HTML document the parser already puts <svg> in the SVG namespace, and
    # omitting it keeps even a namespace URL out of a file that must contain zero links.
    elements: list[str] = [
        f'<svg class="chart" role="img" width="100%" viewBox="0 0 {_SVG_WIDTH:.0f} {total:.0f}" '
        'aria-labelledby="labbox-chart-title labbox-chart-desc">'
    ]
    elements.append(
        '<title id="labbox-chart-title">Hardware timeline</title>'
        f'<desc id="labbox-chart-desc">Static chart generated by LabBox from '
        f"{_text(len(points))} rows of samples/system.jsonl. No script runs to display it.</desc>"
    )
    top = float(_PAD_TOP)
    if percent_series:
        percent_values = [v for _label, series, _c, _d in percent_series for v in _series_values(series)]
        high = max(100.0, float(max(percent_values)))
        elements += _axis_panel(top, height_one, 0.0, high, "percent")
        for _label, series, color, dashed in percent_series:
            for segment in _polylines(series, x0=x0, span=span, low=0.0, high=high, top=top, height=height_one):
                elements.append(_polyline(segment, color, dashed=dashed))
        top += height_one + gap
    if temp_series:
        temp_values = [v for _label, series, _c, _d in temp_series for v in _series_values(series)]
        temp_high = max(temp_values) + 5.0
        elements += _axis_panel(top, height_two, 0.0, temp_high, "degrees C")
        for _label, series, color, dashed in temp_series:
            for segment in _polylines(series, x0=x0, span=span, low=0.0, high=temp_high, top=top, height=height_two):
                elements.append(_polyline(segment, color, dashed=dashed))
    elements += _x_axis(total - _AXIS_BAND + 8.0, x0, span)
    elements.append("</svg>")

    legend = [
        {"label": label, "color": color, "style": "dashed" if dashed else "solid"}
        for label, _series, color, dashed in [*percent_series, *temp_series]
    ]
    stats = [
        {"label": label, "value": _stats_label(_series_values(series))}
        for label, series, _color, _dash in [*percent_series, *temp_series]
    ]
    return {"svg": Markup("".join(elements)), "legend": legend, "stats": stats}


def _section_timeline(inputs: _Inputs) -> Section:
    name = "system.jsonl"
    if not inputs.has_samples:
        blocks = [
            _banner(f"{RunPaths.samples_dir}/{name}: not captured", "missing"),
            _para(inputs.sample_note or NOT_CAPTION[name]),
            _note(
                "Charts here are inline SVG drawn from the recorded rows, so an absent series "
                "stays absent instead of being drawn as a flat line at zero."
            ),
        ]
        return Section(
            "timeline",
            "Hardware timeline",
            "not_captured",
            blocks,
            "no sampling rows were recorded",
            f"{RunPaths.samples_dir}/{name}",
        )

    chart = _build_chart(inputs.samples)
    interval = _median_interval(inputs.samples)
    span = inputs.samples[-1].elapsed_seconds - inputs.samples[0].elapsed_seconds
    interval_text = (
        f"about {_num(interval, 3)} s (median of the observed gaps between samples)"
        if interval is not None
        else "not determinable from a single sample"
    )
    blocks: list[dict[str, Any]] = [
        _pairs(
            [
                ("samples", _text(len(inputs.samples))),
                ("rows skipped as unreadable", _text(inputs.sample_skipped)),
                ("first sample", _dt(inputs.samples[0].t)),
                ("last sample", _dt(inputs.samples[-1].t)),
                ("span covered", _duration(span)),
                ("sampling interval", interval_text),
            ]
        )
    ]
    if chart:
        blocks.append(
            {
                "kind": "chart",
                "svg": chart["svg"],
                "legend": chart["legend"],
                "caption": (
                    f"Hardware timeline: {len(inputs.samples)} samples spanning "
                    f"{_duration(span)}, recorded in {RunPaths.samples_dir}/{name} with a median "
                    f"sampling interval of {interval_text}. Values are instantaneous per-tick "
                    "readings, so a spike between two ticks is not visible here. The drawing is "
                    "inline SVG generated in Python: no script, no external font, no image "
                    "reference."
                ),
            }
        )
        blocks.append(_sub("Series summary"))
        blocks.append(_table(["series", "over the run"], [[row["label"], row["value"]] for row in chart["stats"]]))
    else:
        blocks.append(
            _banner(
                "Sampling rows exist, but none carries a plottable utilization, CPU "
                "or temperature value; nothing is drawn rather than drawing a flat "
                "line.",
                "warn",
            )
        )
    errors = [point.error for point in inputs.samples if point.error]
    if errors:
        blocks.append(_sub("Errors recorded inside samples"))
        blocks.append(_items(list(errors)))
    state = "partial" if (inputs.sample_skipped or errors) else "ok"
    return Section("timeline", "Hardware timeline", state, blocks, "", f"{RunPaths.samples_dir}/{name}")


def _coverage_section(sections: list[Section], inputs: _Inputs) -> Section:
    rows = [
        [
            _text(section.title),
            _state_badge(_STATE_LABEL[section.state], _STATE_KIND[section.state]),
            _text(section.source),
            _text(section.note),
        ]
        for section in sections
    ]
    blocks = [
        _table(["section", "capture", "source", "note"], rows),
        _note(
            "Self-containment: the stylesheet is inlined; charts are inline SVG; there is no "
            "script, remote font, image, frame or tracking reference in this file, and URL "
            "schemes found in recorded data are printed defanged and never linked. Text taken "
            "from run data passed through the same secret scrubber the collectors use, and every "
            "value is HTML-escaped, so a run name cannot become markup."
        ),
        _pairs(
            [
                ("run directory", _code(inputs.paths.dir)),
                ("report generated at", _dt(inputs.generated_at)),
                ("LabBox version", _text(LABBOX_VERSION)),
                ("comparison requested", _text(inputs.compare is not None)),
                ("AI explanation requested", _text(inputs.explanation is not None)),
            ]
        ),
    ]
    if inputs.explanation is None:
        blocks.append(
            _note(
                "No AI explanation was requested. The report is complete without one: "
                "the model layer is optional by design (§11, §23)."
            )
        )
    return Section("coverage", "Capture coverage", "ok", blocks, "", "run directory")


# ---------------------------------------------------------------------------
# template location and rendering
# ---------------------------------------------------------------------------
_TEMPLATE_RELPATH = ("templates", "report.html.j2")
_TEMPLATE_CACHE: str | None = None


def _template_source() -> str:
    """Read the bundled template via ``importlib.resources``.

    ``labbox/templates`` holds data rather than code, so it is deliberately not a package
    with an ``__init__``: the anchor is the ``labbox`` package itself and the template is a
    path below it, which works from a source tree, an installed wheel and a zip import.
    """
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        try:
            anchor = resources.files("labbox")
            _TEMPLATE_CACHE = anchor.joinpath(*_TEMPLATE_RELPATH).read_text(encoding="utf-8")
        except (
            FileNotFoundError,
            NotADirectoryError,
            IsADirectoryError,
            OSError,
            ModuleNotFoundError,
        ):  # pragma: no cover - odd install layouts
            _TEMPLATE_CACHE = (Path(__file__).resolve().parents[1] / "templates" / "report.html.j2").read_text(
                encoding="utf-8"
            )
    return _TEMPLATE_CACHE


def _render_body(sections: list[Section], context: dict[str, Any]) -> str:
    env = Environment(autoescape=True, undefined=StrictUndefined)
    template: Template = env.from_string(_template_source())
    return template.render(sections=sections, **context)


def _plain_fallback(title: str, sections: list[Section], why: str) -> str:
    """If the bundled template itself fails, still ship a document with the same data."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        "</head><body>",
        f"<h1>{escape(title)}</h1>",
        f"<p><strong>The LabBox report template could not be rendered: {escape(why)}</strong></p>",
        "<p>Sections below carry the recorded text, HTML-escaped, with the same coverage claims.</p>",
    ]
    for section in sections:
        parts.append(f"<h2>{escape(section.title)} ({escape(_STATE_LABEL[section.state])})</h2>")
        if section.state == "unavailable":
            parts.append(UNAVAILABLE_COMMENT.format(name=section.key))
        if section.note:
            parts.append(f"<p>{escape(section.note)}</p>")
        for block in section.blocks:
            text = block.get("text") or block.get("caption")
            if text:
                parts.append(f"<p>{escape(text)}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_document(inputs: _Inputs, sections: list[Section]) -> str:
    run = inputs.record
    name = run.name if run is not None else inputs.paths.dir.name
    project = run.project if run is not None else ""
    status = run.status.value if run is not None else "unknown"
    title = f"LabBox run report - {_text(name)}"
    context = {
        "title": title,
        "run_name": _text(name),
        "run_project": _text(project or "unknown project"),
        "run_id": _text(run.run_id if run is not None else inputs.paths.dir.name),
        "status": _text(status),
        "status_kind": _class_slug(status),
        "generated_at": _dt(inputs.generated_at),
        "labbox_version": LABBOX_VERSION,
        "run_dir": _code(inputs.paths.dir),
    }
    try:
        return _render_body(sections, context)
    except Exception as exc:  # a broken template must still produce a readable page
        return _plain_fallback(title, sections, f"{type(exc).__name__}: {_text(exc)}")


def render_report(
    store: FilesystemStore,
    paths: RunPaths,
    *,
    compare: CompareResult | None = None,
    explanation: ExplainResult | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render the offline report for one run directory.

    Any of the run's files may be missing, empty, half-written or fail schema validation:
    the matching section says so instead of rendering blank. Nothing is written here, and
    odd data never raises -- only a read the operating system refuses shows up as a
    degraded section.

    ``generated_at`` is an input rather than a hidden ``now()`` so a report can be
    re-rendered byte-identically (diffable, cacheable, testable).
    """
    inputs = _collect(store, paths)
    inputs.compare = compare
    inputs.explanation = explanation
    if generated_at is not None:
        inputs.generated_at = generated_at
    builders = (
        _section_summary,
        _section_command,
        _section_environment,
        _section_metrics,
        _section_timeline,
        _section_datasets,
        _section_artifacts,
        _section_reproducibility,
        _section_diagnostics,
        _section_compare,
        _section_explanation,
    )
    sections: list[Section] = []
    for builder in builders:
        key = builder.__name__.removeprefix("_section_")
        try:
            built = builder(inputs)
        except Exception as exc:  # failure isolation: one bad section cannot remove the others
            built = _unavailable(
                key,
                key.replace("_", " ").capitalize(),
                "collected run data",
                f"the renderer could not build this section from the recorded data "
                f"({type(exc).__name__}: {_text(exc)})",
            )
        if built is not None:
            sections.append(built)
    sections.append(_coverage_section(sections, inputs))
    return _render_document(inputs, sections)


def write_report(
    store: FilesystemStore,
    paths: RunPaths,
    **kwargs: Any,
) -> Path:
    """Render and atomically write ``report.html``, returning the path written.

    Write failures are re-raised as a structured :class:`LabBoxError` with code
    ``REPORT_GENERATION_FAILED`` and swallowed nowhere: deciding that a missing page is
    acceptable is the caller's job, and run data is never at risk either way (§13).
    """
    document = render_report(store, paths, **kwargs)
    target = paths.resolve(RunPaths.report)
    try:
        store.write_text_atomic(target, document)
    except LabBoxError:
        raise
    except OSError as exc:  # pragma: no cover - the store normally wraps OSError already
        raise LabBoxError(
            f"Failed to write {RunPaths.report}: {exc}",
            code=REPORT_GENERATION_FAILED,
            component="report",
            details={"path": target.name},
        ) from exc
    return target
