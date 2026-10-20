"""The run lifecycle (DEVELOPMENT.md §6) as a stage pipeline.

The order is fixed, but *what runs in each stage* is injectable: milestones M2-M4 add
stages (metrics, artifacts, dataset, diagnostics, report, explain) without touching the
process handling that must stay correct. Two invariants hold no matter what a stage does:

* the training subprocess is started exactly once and its exit code is what we return;
* a stage that raises becomes a warning plus a diagnostic — it never aborts the run and
  never loses the records already on disk.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labbox import __version__
from labbox.config.model import LabboxConfig
from labbox.domain.common import ErrorInfo, now_local
from labbox.domain.diagnostics import Diagnostic, Reproducibility, Severity
from labbox.domain.environment import EnvironmentSnapshot
from labbox.domain.run import CommandSpec, ExitMeaning, RunRecord, RunStatus
from labbox.errors import LabBoxError
from labbox.infrastructure.nvidia import GpuBackend
from labbox.infrastructure.probe import resolve_probe_target
from labbox.infrastructure.redact import Redactor
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout, RunPaths, make_run_id
from labbox.recorder.environment import EnvironmentCollector
from labbox.recorder.process import ProcessResult, ProcessRunner
from labbox.recorder.sampler import HardwareSampler, SamplerSummary


@dataclass(slots=True)
class RunRequest:
    argv: list[str]
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    project: str | None = None
    cwd: Path | None = None
    datasets: list[str] = field(default_factory=list)
    metric_file: str | None = None
    config_files: list[str] = field(default_factory=list)
    sample_interval: float | None = None
    no_gpu: bool = False
    no_network: bool = False
    extra_redact: list[str] = field(default_factory=list)
    copy_artifacts: bool = False
    interpreter: str | None = None
    echo: bool = True


@dataclass(slots=True)
class RunContext:
    layout: ProjectLayout
    store: FilesystemStore
    paths: RunPaths
    config: LabboxConfig
    request: RunRequest
    record: RunRecord
    started_wall: float
    gpu_backend: GpuBackend | None = None
    environment: EnvironmentSnapshot | None = None
    ended_environment: dict[str, Any] = field(default_factory=dict)
    process: ProcessResult | None = None
    sampler_summary: SamplerSummary | None = None
    metrics: Any | None = None
    datasets: Any | None = None
    artifacts: Any | None = None
    hyperparameters: Any | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    reproducibility: Reproducibility | None = None
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def event(self, kind: str, **payload: Any) -> None:
        write_event(self.store, self.paths, kind, **payload)


@dataclass(slots=True)
class RunOutcome:
    run_id: str
    run_dir: Path
    exit_code: int
    exit_code_raw: int | None
    status: RunStatus
    record: RunRecord
    warnings: list[str]
    duration_seconds: float | None
    sampler_summary: SamplerSummary | None = None


Stage = Callable[[RunContext], None]

#: Upper bound on the ``-2``, ``-3`` ... suffix used to keep run ids unique.
_MAX_ID_ATTEMPTS = 1000


def _claim_run_dir(layout: ProjectLayout, base_id: str) -> tuple[str, RunPaths]:
    """Return an id whose directory does not exist yet, creating nothing.

    ``make_run_id`` resolves to the second, so two runs started inside the same second
    with the same name produce the same id. ``ensure_run_dirs`` then reuses the directory
    and the second run overwrites the first one's evidence. Rather than trust the clock to
    tick, disambiguate here: if the directory already exists, append ``-2``, ``-3`` ... The
    base id is tried first, so the common case produces exactly the id it always did.
    """
    if not (layout.runs_dir / base_id).exists():
        return base_id, layout.run_dir(base_id)
    for attempt in range(2, _MAX_ID_ATTEMPTS + 1):
        candidate = f"{base_id}-{attempt}"
        if not (layout.runs_dir / candidate).exists():
            return candidate, layout.run_dir(candidate)
    raise LabBoxError(  # pragma: no cover - a thousand same-second same-name runs
        f"Could not find a free run id for {base_id!r} after {_MAX_ID_ATTEMPTS} attempts.",
        code="RUN_ID_EXHAUSTED",
        component="run_service",
        details={"run_id": base_id},
    )


def write_event(store: FilesystemStore, paths: RunPaths, kind: str, **payload: Any) -> None:
    record = {"t": now_local().isoformat(timespec="milliseconds"), "kind": kind, **payload}
    try:
        store.append_jsonl(paths.events_log, record)
    except (LabBoxError, OSError) as exc:
        # Events are diagnostics for LabBox itself; losing them cannot affect the run. The
        # fallback note goes to stderr.log and is allowed to fail silently too: this code
        # runs on paths where the disk has already rejected a write, and raising here would
        # turn a lost event into a lost run.
        message = getattr(exc, "message", None) or str(exc)
        store.append_text(paths.stderr_log, f"\nlabbox: failed to record event {kind!r}: {message}\n")


class RunService:
    def __init__(
        self,
        *,
        stages: Sequence[Stage] | None = None,
        gpu_backend: GpuBackend | None = None,
        process_factory: Callable[..., ProcessRunner] = ProcessRunner,
    ) -> None:
        self.gpu_backend = gpu_backend
        self.process_factory = process_factory
        self.stages: list[Stage] = list(stages) if stages is not None else default_stages()

    # --- public API ----------------------------------------------------------
    def execute(self, request: RunRequest, layout: ProjectLayout | None = None) -> RunOutcome:
        layout = layout or ProjectLayout.discover(request.cwd or Path.cwd())
        config = LabboxConfig.load(layout.config_path)
        if request.sample_interval:
            config.run.sample_interval_seconds = float(request.sample_interval)
        if request.no_gpu:
            config.gpu.enabled = False
        if request.no_network:
            config.llm.enabled = False
        config.privacy.redact_env_patterns = list(config.privacy.redact_env_patterns) + list(request.extra_redact)

        name = request.name or default_name_from_argv(request.argv)
        started = now_local()
        store = FilesystemStore(layout)
        run_id, paths = _claim_run_dir(layout, make_run_id(name, started))
        store.ensure_run_dirs(paths)
        record = RunRecord(
            run_id=run_id,
            name=slug_name(name),
            project=request.project or detect_project_name(layout.root),
            status=RunStatus.RUNNING,
            tags=list(request.tags),
            started_at=started,
            command=CommandSpec(
                raw=format_command(request.argv),
                argv=[str(a) for a in request.argv],
                cwd=str(Path(request.cwd or Path.cwd())),
            ),
            links={},
            labbox_version=__version__,
        )
        ctx = RunContext(
            layout=layout,
            store=store,
            paths=paths,
            config=config,
            request=request,
            record=record,
            started_wall=time_monotonic(),
            gpu_backend=self.gpu_backend,
            extras={"process_factory": self.process_factory},
        )
        self._prepare_files(ctx)
        for stage in self.stages:
            try:
                stage(ctx)
            except Exception as exc:  # a broken recorder must never kill a training run
                ctx.warnings.append(f"{getattr(stage, '__name__', 'stage')} failed: {exc}")
                ctx.diagnostics.append(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code=f"stage_failed:{getattr(stage, '__name__', 'stage')}",
                        message=str(exc)[:400],
                        evidence=[f"exception={type(exc).__name__}"],
                        suggestion="run with LABBOX_DEBUG=1 for the traceback",
                    )
                )
                write_event(
                    store,
                    paths,
                    "stage.failed",
                    stage=getattr(stage, "__name__", "stage"),
                    error=ErrorInfo.from_exception(exc, component="run_service").model_dump(),
                )
        return self._finalize(ctx)

    # --- files that must exist no matter what -------------------------------
    def _prepare_files(self, ctx: RunContext) -> None:
        ctx.store.write_text_atomic(ctx.paths.command_txt, format_command(ctx.request.argv) + "\n")
        # Diagnostics first, so the run.json we are about to write already advertises it:
        # a recorder killed moments later still leaves a run directory that points at its
        # own explanation.
        _write_initial_diagnostics(ctx)
        ctx.store.write_json_atomic(ctx.paths.run_json, ctx.record)
        write_event(
            ctx.store,
            ctx.paths,
            "run.started",
            run_id=ctx.record.run_id,
            argv=ctx.record.command.argv,
            cwd=ctx.record.command.cwd,
            labbox_version=__version__,
        )

    def _finalize(self, ctx: RunContext) -> RunOutcome:
        record = ctx.record
        if ctx.process is not None:
            _apply_process_result(record, ctx.process)
        else:
            record.status = RunStatus.START_FAILED
            record.exit_meaning = ExitMeaning.NOT_STARTED
            record.exit_code = record.exit_code if record.exit_code is not None else 1
        for warning in ctx.warnings:
            write_event(ctx.store, ctx.paths, "run.warning", message=warning)
        ctx.store.write_json_atomic(ctx.paths.run_json, record)
        return RunOutcome(
            run_id=record.run_id,
            run_dir=ctx.paths.dir,
            exit_code=record.exit_code if record.exit_code is not None else 1,
            exit_code_raw=record.exit_code_raw,
            status=record.status,
            record=record,
            warnings=list(ctx.warnings),
            duration_seconds=record.duration_seconds,
            sampler_summary=ctx.sampler_summary,
        )


# --- stages ------------------------------------------------------------------
def stage_environment(ctx: RunContext) -> None:
    """Pre-run snapshot, taken before the child exists so nothing it loads can influence it."""
    request = ctx.request
    target = resolve_probe_target(
        request.argv,
        override=request.interpreter,
        cwd=str(request.cwd) if request.cwd is not None else None,
    )
    collector = EnvironmentCollector(
        project_root=ctx.layout.root,
        config=ctx.config,
        redactor=Redactor(ctx.config.privacy.redact_env_patterns),
        gpu_backend=ctx.gpu_backend if ctx.gpu_backend is not None else _backend_for(ctx),
    )
    if ctx.gpu_backend is None:
        ctx.gpu_backend = collector.backend_used
    ctx.environment = collector.collect(
        phase="pre_run",
        argv=list(request.argv),
        env=_child_environment(request),
        probe_target=target,
        declared_datasets=list(request.datasets),
    )
    ctx.extras["probe_target"] = target
    ctx.store.write_json_atomic(ctx.paths.environment_json, ctx.environment)
    ctx.event("environment.captured", collectors=ctx.environment.collectors)


def _backend_for(ctx: RunContext) -> GpuBackend:
    from labbox.infrastructure.nvidia import get_backend

    return get_backend(ctx.config.gpu.backend)


def stage_process(ctx: RunContext) -> None:
    """Start training, sample while it runs, keep its exit code."""
    request = ctx.request
    env = _child_environment(request)
    paths = ctx.paths
    sampler: HardwareSampler | None = None
    if ctx.config.gpu.enabled or ctx.config.run.sample_interval_seconds:
        sampler = HardwareSampler(
            store=ctx.store,
            paths=paths,
            backend=ctx.gpu_backend if ctx.gpu_backend is not None else _backend_for(ctx),
            interval_seconds=ctx.config.run.sample_interval_seconds,
            auto_degrade_after_seconds=ctx.config.run.auto_degrade_interval_after_seconds,
            auto_degrade_interval_seconds=ctx.config.run.auto_degrade_interval_seconds,
            max_consecutive_failures=ctx.config.gpu.max_consecutive_failures,
            gpu_metrics=list(ctx.config.gpu.metrics),
        )
        sampler.start()
        ctx.event("sampling.started", interval_seconds=ctx.config.run.sample_interval_seconds)
    runner = make_process_runner(ctx, request, env)
    ctx.event("process.started", argv=runner.argv, cwd=runner.cwd)
    try:
        ctx.process = runner.run()
    finally:
        if sampler is not None:
            ctx.sampler_summary = sampler.stop()
            ctx.warnings.extend(ctx.sampler_summary.to_warnings())
    if ctx.process is not None:
        if ctx.process.launch_error:
            ctx.event(
                "process.launch_failed",
                error=ctx.process.launch_error,
                code=ctx.process.launch_error_code,
            )
        else:
            ctx.event(
                "process.exited",
                exit_code=ctx.process.exit_code,
                exit_code_raw=ctx.process.exit_code_raw,
                meaning=ctx.process.exit_meaning.value,
                interrupted=ctx.process.interrupted,
            )
        ctx.warnings.extend(ctx.process.reader_timeouts)
        ctx.warnings.extend(f"log write problem: {e}" for e in ctx.process.log_errors)
        _apply_process_result(ctx.record, ctx.process)


def _apply_process_result(record: RunRecord, process: ProcessResult) -> None:
    """Copy the child's outcome into the record and decide the status from it.

    One place, called from ``stage_process`` and again from ``_finalize``. It used to be
    duplicated, and the copy that ran before the report wrote only ``status`` and
    ``exit_code`` — so the generated page showed ``exit_code_raw=None`` and
    ``exit_meaning=unknown`` for a run whose raw status was sitting in memory.
    """
    record.exit_code = process.exit_code if process.exit_code is not None else 1
    record.exit_code_raw = process.exit_code_raw
    record.exit_meaning = process.exit_meaning
    record.interrupted = process.interrupted
    record.forced_kill = process.forced_kill
    record.killed_by_labbox = process.killed_by_labbox
    record.duration_seconds = process.duration_seconds
    record.ended_at = process.ended_at or record.ended_at or now_local()
    if process.launch_error is not None:
        record.status = RunStatus.START_FAILED
    elif process.interrupted:
        record.status = RunStatus.INTERRUPTED
    elif record.exit_code == 0:
        record.status = RunStatus.SUCCEEDED
    else:
        record.status = RunStatus.FAILED


def make_process_runner(ctx: RunContext, request: RunRequest, env: Mapping[str, str]) -> ProcessRunner:
    paths = ctx.paths
    factory: Callable[..., ProcessRunner] = ctx.extras.get("process_factory") or ProcessRunner
    return factory(
        list(request.argv),
        cwd=str(Path(request.cwd or Path.cwd())),
        env=dict(env),
        stdout_path=paths.stdout_log,
        stderr_path=paths.stderr_log,
        plain_stdout_path=paths.stdout_plain_log if ctx.config.run.store_plain_logs else None,
        echo=request.echo,
        echo_stderr=request.echo,
    )


def stage_ended_environment(ctx: RunContext) -> None:
    """What the world looked like right after training, for the run index only."""
    from labbox.infrastructure.git import GitClient

    git = GitClient(ctx.layout.root, timeout=ctx.config.git.timeout_seconds).collect(include_diff_hash=False)
    ended: dict[str, Any] = {
        "captured_at": now_local().isoformat(timespec="seconds"),
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "gpu_count": len(ctx.environment.gpus) if ctx.environment else 0,
        "ram_available_bytes": ctx.environment.host.ram_available_bytes if ctx.environment else None,
        "duration_seconds": ctx.process.duration_seconds if ctx.process else None,
    }
    if ctx.sampler_summary is not None:
        ended["sample_count"] = ctx.sampler_summary.sample_count
        ended["sampling_backend"] = ctx.sampler_summary.backend
    ctx.ended_environment = ended
    ctx.record.ended_environment = ended
    ctx.record.links["ended_environment"] = "run.json#ended_environment"


def stage_links(ctx: RunContext) -> None:
    ctx.record.links.update(
        {
            "environment": "environment.json",
            "logs_stdout": "logs/stdout.log",
            "logs_stderr": "logs/stderr.log",
            "events": "logs/events.jsonl",
        }
    )
    if ctx.config.run.store_plain_logs:
        ctx.record.links["logs_stdout_plain"] = "logs/stdout.plain.log"
    # The sampler never runs when both GPU and CPU sampling are off, so the file is not
    # there. Advertising a link to a file that does not exist turns "we did not sample"
    # into "the samples are somewhere else".
    if ctx.paths.system_samples.is_file():
        ctx.record.links["samples"] = "samples/system.jsonl"
    if ctx.process is not None and ctx.process.interrupted:
        ctx.record.links["note"] = "run was interrupted; see logs/events.jsonl"


def stage_datasets(ctx: RunContext) -> None:
    """Data version, captured *before* training so a mid-run edit cannot hide itself."""
    from labbox.recorder.dataset import DatasetManifestBuilder

    declared = list(ctx.request.datasets) or list(ctx.config.datasets.paths)
    manifest = DatasetManifestBuilder(project_root=ctx.layout.root, config=ctx.config).build(declared)
    ctx.datasets = manifest
    ctx.store.write_json_atomic(ctx.paths.dataset_manifest_json, manifest)
    ctx.record.links["dataset_manifest"] = "dataset-manifest.json"
    if manifest.not_declared:
        ctx.event("dataset.not_declared")


def stage_config_values(ctx: RunContext) -> None:
    from labbox.analysis.config_values import ConfigValueExtractor

    values = ConfigValueExtractor(
        project_root=ctx.layout.root,
        config=ctx.config,
        explicit_files=list(ctx.request.config_files),
        argv=list(ctx.request.argv),
    ).extract()
    ctx.hyperparameters = values
    ctx.store.write_json_atomic(ctx.paths.config_values_json, values)
    ctx.record.links["config_values"] = "config-values.json"
    ctx.event("config_values.extracted", files=values.files_read, count=len(values.values))


def stage_metrics(ctx: RunContext) -> None:
    from labbox.recorder.metrics import MetricsIngestor

    summary = MetricsIngestor(
        project_root=ctx.layout.root,
        config=ctx.config,
        metric_file=ctx.request.metric_file,
    ).ingest(ctx.paths.dir, history_path=ctx.paths.metrics_history)
    ctx.metrics = summary
    ctx.store.write_json_atomic(ctx.paths.metrics_json, summary)
    ctx.record.links["metrics"] = "metrics.json"
    if summary.metrics_missing:
        ctx.event("metrics.missing")


def stage_artifacts(ctx: RunContext) -> None:
    from labbox.recorder.artifacts import ArtifactCollector, copy_artifacts

    started = ctx.record.started_at
    metrics_files = []
    if ctx.request.metric_file:
        candidate = Path(ctx.request.metric_file)
        metrics_files.append(candidate if candidate.is_absolute() else ctx.layout.root / candidate)
    manifest = ArtifactCollector(project_root=ctx.layout.root, config=ctx.config).collect(
        metrics_files=metrics_files, since=started
    )
    if ctx.request.copy_artifacts:
        manifest.copied = copy_artifacts(manifest, ctx.paths.dir, limit_bytes=ctx.config.artifacts.copy_max_bytes)
    ctx.artifacts = manifest
    ctx.store.write_json_atomic(ctx.paths.artifacts_json, manifest)
    ctx.record.links["artifacts"] = "artifacts.json"


def _write_initial_diagnostics(ctx: RunContext) -> None:
    """A run directory must explain itself even if the recorder never comes back.

    Without this, a killed LabBox leaves ``status=running`` plus logs and nothing that says
    why the rest is missing - the reader has to guess, which is exactly the kind of missing
    evidence LabBox is supposed to surface. The post-run stage overwrites this file.
    """
    from labbox.domain.diagnostics import DiagnosticsFile, Severity

    document = DiagnosticsFile(
        generated_at=now_local(),
        diagnostics=[
            Diagnostic(
                severity=Severity.INFO,
                code="run_in_progress",
                message=(
                    "记录进行中：本文件在训练开始前写入初版，正常结束时会被 finalize 阶段覆盖。"
                    "若 run.json 的 status 长期停留在 running，说明 LabBox 当时被终止，"
                    "metrics/artifacts/report/reproduce 等文件不会存在。"
                ),
                evidence=["status=running", "written before the training process started"],
                suggestion="把该 run 当作证据不完整处理；重跑时避免强杀 LabBox 进程",
            )
        ],
    )
    ctx.store.write_json_atomic(ctx.paths.diagnostics_json, document)
    ctx.record.links["diagnostics"] = "diagnostics.json"


def stage_reproduce_scripts(ctx: RunContext) -> None:
    from labbox.infrastructure.shellgen import write_reproduce_scripts

    if ctx.environment is None:
        ctx.warnings.append("reproduce scripts skipped: no environment snapshot was captured")
        return
    names = write_reproduce_scripts(
        ctx.store,
        ctx.paths,
        ctx.record,
        ctx.environment,
        ctx.datasets,
        # A run directory is `<root>/.labbox/runs/<run-id>`, so the project root is three
        # levels up. Writing "." here made the script search the run directory itself and
        # then guess, which worked only by accident of the fallback in the prelude.
        project_root_rel=_project_root_relative(ctx.paths.dir, ctx.layout.root),
    )
    if names:
        ctx.record.links["reproduce"] = names[0]
        ctx.event("reproduce.generated", files=names)


def _project_root_relative(run_dir: Path, project_root: Path) -> str:
    """How to walk from the run directory back to the project root, or "." if unrelated.

    Computed rather than hardcoded so a relocated or symlinked project still produces a
    path that resolves; a wrong hardcoded value would silently degrade every check in the
    generated script to UNKNOWN.
    """
    try:
        relative = Path(os.path.relpath(project_root, run_dir))
    except (ValueError, OSError):  # pragma: no cover - different drives on Windows
        return "."
    text = relative.as_posix()
    return text or "."


def stage_diagnostics(ctx: RunContext) -> None:
    """Per-run diagnostics and the reproducibility checklist (§8.6, §12.1)."""
    from labbox.analysis.rules import run_diagnostics, score_reproducibility
    from labbox.domain.diagnostics import DiagnosticIdPool, DiagnosticsFile

    diagnostics = list(ctx.diagnostics) + run_diagnostics(_bundle_of(ctx))
    pool = DiagnosticIdPool()
    for diagnostic in diagnostics:
        diagnostic.id = pool.next()
    reproducibility = score_reproducibility(_bundle_of(ctx))
    ctx.diagnostics = diagnostics
    ctx.reproducibility = reproducibility
    document = DiagnosticsFile(
        generated_at=now_local(),
        reproducibility=reproducibility,
        diagnostics=diagnostics,
    )
    ctx.store.write_json_atomic(ctx.paths.diagnostics_json, document)
    ctx.record.links["diagnostics"] = "diagnostics.json"


def _bundle_of(ctx: RunContext) -> Any:
    """Wrap the in-memory run as a RunBundle without re-reading the files we just wrote."""
    from labbox.analysis.record import RunBundle

    bundle = RunBundle(ctx.paths, ctx.store)
    bundle.record = ctx.record
    bundle.environment = ctx.environment
    bundle.metrics = ctx.metrics
    bundle.datasets = ctx.datasets
    bundle.artifacts = ctx.artifacts
    bundle.config_values = ctx.hyperparameters
    return bundle


def stage_report(ctx: RunContext) -> None:
    """Offline HTML report (§13). A report failure never affects the records."""
    from labbox.report.html import write_report

    # The report is rendered from the files on disk, so the index it reads has to be the
    # finished one. Writing it here (rather than leaving it to ``_finalize``) is what makes
    # the page show this run's real status, exit code and duration instead of the
    # ``running`` placeholder written before the child started.
    ctx.record.links["report"] = "report.html"
    _persist_record(ctx)
    path = write_report(ctx.store, ctx.paths)
    ctx.event("report.generated", file=str(path.name))


def _persist_record(ctx: RunContext) -> None:
    try:
        ctx.store.write_json_atomic(ctx.paths.run_json, ctx.record)
    except LabBoxError as exc:  # pragma: no cover - disk failure; the run still finalizes
        ctx.warnings.append(f"run.json could not be updated before the report: {exc.message}")


def default_stages() -> list[Stage]:
    """The fixed pipeline.

    Order constraints that matter, both learned the hard way:

    * ``stage_links`` runs before ``stage_report`` so the embedded page lists every file it
      was rendered from, including itself;
    * ``stage_report`` runs after ``stage_diagnostics`` and after the record is persisted,
      so the report is a description of the completed run rather than of a half-written one.
    """
    return [
        stage_environment,
        stage_datasets,
        stage_config_values,
        stage_process,
        stage_ended_environment,
        stage_metrics,
        stage_artifacts,
        stage_reproduce_scripts,
        stage_diagnostics,
        stage_links,
        stage_report,
    ]


# --- helpers -----------------------------------------------------------------
def _child_environment(request: RunRequest) -> dict[str, str]:
    return dict(os.environ)


def default_name_from_argv(argv: Sequence[str]) -> str:
    """Derive a readable default name: the script being run, minus its extension."""
    for token in argv:
        lowered = str(token).lower()
        if lowered.endswith((".py", ".sh", ".ps1", ".ipynb")) and not str(token).startswith("-"):
            return Path(str(token)).stem
        if lowered in {"torchrun", "python", "python3", "py"}:
            continue
    return "run"


def slug_name(name: str) -> str:
    from labbox.paths import slugify

    return slugify(name)


def format_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return " ".join(_quote_windows(str(a)) for a in argv)
    return " ".join(shlex.quote(str(a)) for a in argv)


def _quote_windows(value: str) -> str:
    if value == "":
        return '""'
    if not any(char in value for char in ' \t\n"'):
        return value
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def detect_project_name(root: Path) -> str:
    return root.name


def time_monotonic() -> float:
    import time

    return time.monotonic()


__all__ = [
    "RunContext",
    "RunOutcome",
    "RunRequest",
    "RunService",
    "Stage",
    "default_name_from_argv",
    "default_stages",
    "format_command",
    "write_event",
]
