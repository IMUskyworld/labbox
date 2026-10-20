"""Filesystem layout of a LabBox project and of a single run directory.

Two guarantees matter here:

* a run name or run id coming from the CLI must never resolve outside ``.labbox/runs``;
* every path we hand out is absolute, but files store project-relative paths so that a
  project directory can be moved or copied.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from labbox.errors import PathEscapeError

LABBOX_DIR = ".labbox"
CONFIG_FILE = "config.toml"
RUNS_DIR = "runs"

_SLUG_STRIP = re.compile(r"[^0-9A-Za-z\xac-\uffff._-]+")
_TIMESTAMP = re.compile(r"^\d{8}-\d{6}-")
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9A-Za-z\xac-\uffff._-]{1,64}$")


def find_project_root(start: Path | None = None) -> Path | None:
    """Nearest ancestor containing ``.labbox/`` (inclusive of *start*)."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / LABBOX_DIR).is_dir():
            return candidate
    return None


def slugify(name: str, *, max_length: int = 48) -> str:
    """Make *name* safe as a path component while keeping it readable.

    Non-ASCII letters (including CJK) are preserved: Windows paths are not ASCII-only,
    and forcing ASCII would turn ``--name 中文实验`` into something unrecognizable.
    """
    normalized = unicodedata.normalize("NFKC", name).strip()
    slug = _SLUG_STRIP.sub("-", normalized).strip("-.")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        slug = "run"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-.")
    # A name that Windows would refuse: CON, PRN, AUX, NUL, COM1..9, LPT1..9
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", slug):
        slug = f"_{slug}"
    return slug or "run"


def make_run_id(name: str, when: datetime) -> str:
    return f"{when.strftime('%Y%m%d-%H%M%S')}-{slugify(name)}"


def is_run_id(value: str) -> bool:
    return bool(RUN_ID_RE.match(value))


def strip_timestamp(run_id: str) -> str:
    """The readable part of a run id, used for name-based lookups."""
    return _TIMESTAMP.sub("", run_id, count=1)


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Locations inside one run directory.

    ``dir`` is absolute; the other members are relative names, so a run can be zipped
    and unpacked elsewhere.
    """

    dir: Path

    run = "run.json"
    environment = "environment.json"
    command = "command.txt"
    metrics = "metrics.json"
    dataset_manifest = "dataset-manifest.json"
    artifacts = "artifacts.json"
    diagnostics = "diagnostics.json"
    compare = "compare.json"
    config_values = "config-values.json"
    report = "report.html"
    logs_dir = "logs"
    samples_dir = "samples"
    checkpoints_dir = "checkpoints"

    def __getitem__(self, name: str) -> Path:  # pragma: no cover - sugar
        return self.resolve(name)

    def resolve(self, relative: str) -> Path:
        return self.dir / Path(*relative.split("/"))

    # --- summary documents ---------------------------------------------------
    @property
    def run_json(self) -> Path:
        return self.resolve(self.run)

    @property
    def environment_json(self) -> Path:
        return self.resolve(self.environment)

    @property
    def command_txt(self) -> Path:
        return self.resolve(self.command)

    @property
    def metrics_json(self) -> Path:
        return self.resolve(self.metrics)

    @property
    def dataset_manifest_json(self) -> Path:
        return self.resolve(self.dataset_manifest)

    @property
    def artifacts_json(self) -> Path:
        return self.resolve(self.artifacts)

    @property
    def diagnostics_json(self) -> Path:
        return self.resolve(self.diagnostics)

    @property
    def compare_json(self) -> Path:
        return self.resolve(self.compare)

    @property
    def config_values_json(self) -> Path:
        return self.resolve(self.config_values)

    @property
    def report_html(self) -> Path:
        return self.resolve(self.report)

    # --- concrete files ------------------------------------------------------
    @property
    def stdout_log(self) -> Path:
        return self.resolve(f"{self.logs_dir}/stdout.log")

    @property
    def stdout_plain_log(self) -> Path:
        return self.resolve(f"{self.logs_dir}/stdout.plain.log")

    @property
    def stderr_log(self) -> Path:
        return self.resolve(f"{self.logs_dir}/stderr.log")

    @property
    def events_log(self) -> Path:
        return self.resolve(f"{self.logs_dir}/events.jsonl")

    @property
    def system_samples(self) -> Path:
        return self.resolve(f"{self.samples_dir}/system.jsonl")

    @property
    def system_samples_csv(self) -> Path:
        return self.resolve(f"{self.samples_dir}/system.csv")

    @property
    def metrics_history(self) -> Path:
        return self.resolve(f"{self.logs_dir}/metrics-history.jsonl")

    @property
    def reproduce_ps1(self) -> Path:
        return self.resolve("reproduce.ps1")

    @property
    def reproduce_sh(self) -> Path:
        return self.resolve("reproduce.sh")


class ProjectLayout:
    """Path helpers for an initialized project."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.labbox_dir = self.root / LABBOX_DIR
        self.config_path = self.labbox_dir / CONFIG_FILE
        self.runs_dir = self.labbox_dir / RUNS_DIR

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectLayout:
        root = find_project_root(start)
        if root is None:
            from labbox.errors import NotInitializedError

            raise NotInitializedError(
                "No .labbox directory found here. Run `labbox init` first.",
                details={"cwd": str(Path.cwd())},
            )
        return cls(root)

    def ensure(self) -> None:
        self.labbox_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> RunPaths:
        """Location for a run id, existing or not (used when starting a run).

        Lookup of an *existing* run by id or name goes through :meth:`resolve_run_id`,
        which is the one that has to fail when nothing matches.
        """
        if not is_run_id(run_id):
            from labbox.errors import LabBoxError

            raise LabBoxError(
                f"{run_id!r} is not a valid run id (expected YYYYmmdd-HHMMSS-name).",
                code="RUN_NOT_FOUND",
                component="store",
                details={"run_id": run_id},
            )
        return RunPaths(dir=self._guard(self.runs_dir / run_id))

    def resolve_run_id(self, run_id: str) -> Path:
        """Resolve a run id *or* a unique run name to its directory, refusing escapes."""
        candidate = self._candidate(run_id)
        if candidate is None:
            from labbox.errors import RunNotFoundError

            raise RunNotFoundError(f"No run matches {run_id!r}.", details={"query": run_id})
        return candidate

    def _candidate(self, query: str) -> Path | None:
        exact = self.runs_dir / query
        if exact.is_dir() and (exact / "run.json").exists():
            return self._guard(exact)
        matches = self.match_runs(query)
        if len(matches) == 1:
            return self._guard(matches[0])
        if len(matches) > 1:
            from labbox.errors import RunNotFoundError

            raise RunNotFoundError(
                f"{query!r} matches {len(matches)} runs; use the full run id.",
                details={"query": query, "matches": [m.name for m in matches]},
            )
        return None

    def match_runs(self, query: str) -> list[Path]:
        """Runs whose id equals, ends with ``-query``, or whose name matches *query*."""
        needle = strip_timestamp(query).lower()
        out: list[Path] = []
        if not self.runs_dir.is_dir():
            return out
        for entry in sorted(self.runs_dir.iterdir()):
            if not entry.is_dir() or not (entry / "run.json").exists():
                continue
            name = entry.name.lower()
            if name == needle or name.endswith(f"-{needle}") or strip_timestamp(name) == needle:
                out.append(entry)
        return out

    def _guard(self, path: Path) -> Path:
        resolved = path.resolve()
        runs_root = self.runs_dir.resolve()
        if os.path.commonpath([os.path.abspath(resolved), os.path.abspath(runs_root)]) != str(runs_root):
            raise PathEscapeError(
                f"Refusing to touch {path} outside {self.runs_dir}.",
                details={"path": str(path), "runs_dir": str(self.runs_dir)},
            )
        return resolved
