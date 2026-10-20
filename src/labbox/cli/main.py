"""Typer application: argument parsing, service calls, presentation. Nothing else.

Every LabBox-written message goes to **stderr**. That is not a style choice: writing to
stdout would change what the user sees from their own command, which §2.4 forbids.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from labbox import __version__
from labbox.config.model import LabboxConfig
from labbox.domain.run import RunRecord, RunStatus
from labbox.errors import LabBoxError
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout
from labbox.services.run_service import RunRequest, RunService

# `labbox run` must never write to stdout: that stream belongs to the training command, so
# its summary and warnings all go through `err`. The read-only commands may use `out`.
err = Console(stderr=True)
out = Console()

app = typer.Typer(
    name="labbox",
    help="Local-first recorder and differ for AI training experiments.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)


@app.callback(invoke_without_command=True)
def _version_option(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the LabBox version and exit.",
        is_eager=True,
    ),
) -> None:
    _configure_output_streams()
    if version:
        out.print(f"labbox {__version__}")
        raise typer.Exit()


def _configure_output_streams() -> None:
    """Make LabBox's own stdout/stderr UTF-8, even when the console is cp1252.

    GitHub's ``windows-latest`` runner gives Python a cp1252 pipe. Chinese diagnostic text
    then raises ``UnicodeEncodeError`` and the line is truncated mid-run-id (observed as a
    failing ``doctor`` stale-run test). Training output is untouched: ``ProcessRunner``
    forwards the child's bytes to ``sys.stdout.buffer``, not through this text layer.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _handle_console_break_like_interrupt() -> None:
    """Make Ctrl+Break behave like Ctrl+C on Windows.

    Measured with scripts/spike_process.py: CPython's default handling of
    ``CTRL_BREAK_EVENT`` terminates the process with STATUS_CONTROL_C_EXIT (0xC000013A)
    without raising ``KeyboardInterrupt``, so LabBox died mid-run and never finalized the
    record. Rebinding SIGBREAK to the Ctrl+C handler makes both keys mean "interrupt",
    which is what a user pressing either key expects and what §6.3 promises.
    """
    if os.name != "nt" or not hasattr(signal, "SIGBREAK"):
        return
    try:
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    except (OSError, ValueError):  # pragma: no cover - not a console process
        return


_handle_console_break_like_interrupt()


def _emit(text: str) -> None:
    """Write machine-readable output as UTF-8 bytes.

    Two hazards are handled here: Rich would wrap long lines and corrupt JSON, and on a
    GBK console Python's text-mode stdout would encode CJK paths as cp936, which no JSON
    consumer can decode. Writing the buffer directly keeps the bytes UTF-8.
    """
    newline = chr(10)
    payload = text if text.endswith(newline) else text + newline
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:  # pragma: no cover - captured stdout in tests
        sys.stdout.write(payload)
        sys.stdout.flush()
        return
    buffer.write(payload.encode("utf-8", "replace"))
    buffer.flush()


def _debug_enabled() -> bool:
    return os.environ.get("LABBOX_DEBUG") == "1"


def _fail(message: str, *, code: str = "ERROR", exc: BaseException | None = None) -> NoReturn:
    err.print(f"[red]labbox: {message}[/red]")
    if exc is not None and _debug_enabled():
        import traceback

        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    raise typer.Exit(code=2)


def _load_record(store: FilesystemStore, run_dir: Path) -> RunRecord | None:
    data = store.read_json(run_dir / "run.json", strict=False)
    if data is None:
        return None
    try:
        return RunRecord.model_validate(data)
    except Exception:  # pragma: no cover - corrupt or newer schema
        return None


@app.command("init")
def init_command(
    path: Path = typer.Argument(Path("."), help="Project directory to initialize."),
    project_name: str = typer.Option(None, "--project", help="Project name (default: directory name)."),
    write_gitignore: bool = typer.Option(True, "--gitignore/--no-gitignore", help="Add .labbox/runs to .gitignore."),
    force: bool = typer.Option(False, "--force", help="Rewrite an existing config.toml."),
) -> None:
    """Create .labbox/config.toml and report what this machine can record."""
    root = path.resolve()
    layout = ProjectLayout(root)
    existing = layout.config_path.is_file()
    if existing and not force:
        err.print(f"[yellow]labbox: {layout.config_path} already exists; use --force to overwrite[/yellow]")
    else:
        config = LabboxConfig.default()
        config.project = project_name or root.name
        layout.ensure()
        layout.config_path.write_text(config.to_toml(), encoding="utf-8", newline="\n")
        err.print(f"[green]labbox: wrote {layout.config_path}[/green]")
    if write_gitignore:
        _ensure_gitignored(root, layout.runs_dir)
    _capability_report(root)


def _ensure_gitignored(root: Path, runs_dir: Path) -> None:
    gitignore = root / ".gitignore"
    entry = ".labbox/runs/"
    if gitignore.is_file():
        text = gitignore.read_text(encoding="utf-8", errors="replace")
        if any(line.strip() in {entry, ".labbox/runs", entry.strip("/")} for line in text.splitlines()):
            return
        with gitignore.open("a", encoding="utf-8", newline="\n") as fh:
            if not text.endswith("\n") and text:
                fh.write("\n")
            fh.write(f"\n# LabBox experiment records (local only)\n{entry}\n")
        err.print(f"[green]labbox: added {entry} to {gitignore.name}[/green]")
        return
    if (root / ".git").exists():
        gitignore.write_text(f"# LabBox experiment records (local only)\n{entry}\n", encoding="utf-8", newline="\n")
        err.print(f"[green]labbox: created {gitignore}[/green]")
    else:
        gitignore.write_text(f"# LabBox experiment records (local only)\n{entry}\n", encoding="utf-8", newline="\n")
        err.print(
            f"[yellow]labbox: no git repository here; wrote {gitignore.name} anyway "
            "(add .labbox/runs/ to whatever you do use)[/yellow]"
        )


def _capability_report(root: Path) -> None:
    from labbox.infrastructure.git import GitClient
    from labbox.infrastructure.nvidia import get_backend

    config = LabboxConfig.load(root / ".labbox" / "config.toml", include_user=False)
    table = Table(title="LabBox capabilities", show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result")
    git = GitClient(root, timeout=config.git.timeout_seconds)
    backend = get_backend(config.gpu.backend)
    static = []
    try:
        static = backend.static_info()
    except Exception:
        pass
    finally:
        backend.close()
    try:
        import psutil  # noqa: F401

        psutil_ok = "yes"
    except ImportError:
        psutil_ok = "no (CPU/RAM sampling disabled)"
    table.add_row("python (labbox)", sys.version.split()[0])
    table.add_row("git", "yes" if git.available() else "no (commit state unknown)")
    table.add_row("repository", "yes" if git.in_repo() else "no")
    table.add_row("gpu backend", f"{getattr(backend, 'name', 'none')} ({len(static)} device(s))")
    table.add_row("psutil", psutil_ok)
    table.add_row("llm", "enabled" if config.llm.enabled else "disabled (all core features work without it)")
    err.print(table)


@app.command("run")
def run_command(
    command: list[str] = typer.Argument(
        ...,
        help="The training command after `--`, e.g. `-- python train.py --epochs 20`.",
        metavar="COMMAND...",
    ),
    name: str = typer.Option(None, "--name", help="Readable run name."),
    tag: list[str] = typer.Option([], "--tag", help="Repeatable tag."),
    project: str = typer.Option(None, "--project", help="Project name override."),
    dataset: list[str] = typer.Option([], "--dataset", help="Dataset path or version id (repeatable)."),
    metric_file: str = typer.Option(None, "--metric-file", help="Structured metrics file written by training."),
    config_file: list[str] = typer.Option([], "--config-file", help="Config file whose values feed compare."),
    sample_interval: float = typer.Option(None, "--sample-interval", help="Seconds between hardware samples."),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable GPU sampling."),
    no_network: bool = typer.Option(False, "--no-network", help="Forbid any remote request."),
    redact_env: list[str] = typer.Option([], "--redact-env", help="Extra env-var name patterns to hide."),
    copy_artifacts: bool = typer.Option(False, "--copy-artifacts", help="Copy artifacts into the run dir."),
    python_exe: str = typer.Option(None, "--python", help="Interpreter to probe for package versions."),
    quiet: bool = typer.Option(False, "--quiet", help="Do not print the run summary line."),
) -> None:
    """Run a training command unchanged and record the evidence around it."""
    argv = list(command)
    if not argv:
        _fail("nothing to run; use `labbox run -- <command>`")
    request = RunRequest(
        argv=argv,
        name=name,
        tags=list(tag),
        project=project,
        cwd=Path.cwd(),
        datasets=list(dataset),
        metric_file=metric_file,
        config_files=list(config_file),
        sample_interval=sample_interval,
        no_gpu=no_gpu,
        no_network=no_network,
        extra_redact=list(redact_env),
        copy_artifacts=copy_artifacts,
        interpreter=python_exe,
        echo=not quiet,
    )
    try:
        layout = ProjectLayout.discover(Path.cwd())
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    try:
        outcome = RunService().execute(request, layout)
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    if not quiet:
        colour = {"succeeded": "green", "failed": "red", "interrupted": "yellow", "start_failed": "red"}.get(
            outcome.status.value, "white"
        )
        err.print(
            f"[{colour}]labbox: {outcome.status.value}[/] "
            f"run [bold]{outcome.run_id}[/] exit={outcome.exit_code} "
            f"({outcome.duration_seconds or 0:.1f}s) -> {_relative(layout.root, outcome.run_dir)}"
        )
        for warning in outcome.warnings:
            err.print(f"[yellow]labbox: warning: {warning}[/yellow]")
    raise typer.Exit(code=outcome.exit_code if outcome.exit_code is not None else 1)


@app.command("list")
def list_command(
    limit: int = typer.Option(20, "--limit", help="Show at most N runs."),
    status: str = typer.Option(None, "--status", help="Filter by status."),
    tag: str = typer.Option(None, "--tag", help="Filter by tag."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List recorded runs, newest first."""
    if limit < 0:
        _fail(f"--limit must be 0 or more, got {limit}")
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    store = FilesystemStore(layout)
    rows: list[dict[str, Any]] = []
    if limit:
        for run_dir in reversed(store.list_run_dirs()):
            record = _load_record(store, run_dir)
            if record is None:
                continue
            if status and record.status.value != status:
                continue
            if tag and tag not in record.tags:
                continue
            rows.append(
                {
                    "run_id": record.run_id,
                    "name": record.name,
                    "status": record.status.value,
                    "exit_code": record.exit_code,
                    "duration_seconds": record.duration_seconds,
                    "tags": record.tags,
                    "started_at": record.started_at.isoformat(timespec="seconds"),
                    "project": record.project,
                }
            )
            if len(rows) >= limit:
                break
    if json_out:
        import json

        _emit(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    table = Table(show_header=True, header_style="bold")
    for column in ("run id", "name", "status", "exit", "duration", "started", "tags"):
        table.add_column(column)
    for row in rows:
        colour = {"succeeded": "green", "failed": "red", "interrupted": "yellow"}.get(row["status"], "white")
        duration = row["duration_seconds"]
        table.add_row(
            row["run_id"],
            row["name"],
            f"[{colour}]{row['status']}[/]",
            "" if row["exit_code"] is None else str(row["exit_code"]),
            "-" if duration is None else f"{duration:.1f}s",
            row["started_at"],
            ",".join(row["tags"]),
        )
    if not rows:
        err.print("labbox: no runs recorded yet")
    else:
        out.print(table)


@app.command("show")
def show_command(
    run: str = typer.Argument(..., help="Run id, run name or unique suffix."),
    json_out: bool = typer.Option(False, "--json", help="Dump run.json."),
    logs: int = typer.Option(0, "--logs", help="Also print the last N lines of stdout/stderr."),
) -> None:
    """Show one run: status, command, environment summary, diagnostics."""
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    run_dir = layout.resolve_run_id(run)
    store = FilesystemStore(layout)
    if json_out:
        _emit((run_dir / "run.json").read_text(encoding="utf-8"))
        return
    record = _load_record(store, run_dir)
    if record is None:
        _fail(f"{run_dir / 'run.json'} could not be parsed")
    assert record is not None
    out.print(f"[bold]{record.run_id}[/]  {record.status.value.upper()}")
    out.print(f"  project    {record.project}")
    out.print(f"  name       {record.name}   tags: {', '.join(record.tags) or '-'}")
    out.print(f"  started    {record.started_at.isoformat(timespec='seconds')}")
    out.print(f"  duration   {record.duration_seconds if record.duration_seconds is not None else '-'}s")
    out.print(
        f"  exit       normalized={record.exit_code} raw={record.exit_code_raw} ({record.exit_meaning.value})"
        + ("  interrupted" if record.interrupted else "")
    )
    out.print(f"  command    {record.command.raw}")
    out.print(f"  cwd        {record.command.cwd}")
    environment = store.read_json(run_dir / "environment.json", strict=False) or {}
    if environment:
        python = environment.get("python") or {}
        git = environment.get("git") or {}
        frameworks = environment.get("frameworks") or {}
        gpus = environment.get("gpus") or []
        out.print(f"  python     {python.get('version', '?')}  ({python.get('executable', '?')})")
        out.print(
            f"  git        {git.get('short_commit') or git.get('commit') or 'n/a'}"
            f"{' (dirty)' if git.get('dirty') else ''}"
        )
        out.print(f"  frameworks {', '.join(f'{k}={v}' for k, v in frameworks.items()) or '-'}")
        out.print(f"  gpus       {', '.join(g.get('name', '?') for g in gpus) or '-'}")
        if environment.get("absent"):
            out.print(f"  [yellow]not captured: {', '.join(environment['absent'])}[/yellow]")
    metrics = store.read_json(run_dir / "metrics.json", strict=False) or {}
    if metrics.get("final"):
        out.print("  metrics    " + ", ".join(f"{k}={v}" for k, v in sorted(metrics["final"].items())))
    elif metrics:
        out.print("  [yellow]metrics    not captured (no metrics file was declared or found)[/]")
    diagnostics = store.read_json(run_dir / "diagnostics.json", strict=False) or {}
    for item in diagnostics.get("diagnostics", [])[:12]:
        severity = item.get("severity", "info")
        colour = {"critical": "red", "high": "red", "warning": "yellow"}.get(severity, "cyan")
        out.print(f"  [{colour}]{severity:>8}[/]  {item.get('code')}: {item.get('message')}")
    repro = diagnostics.get("reproducibility") or {}
    if repro:
        out.print(f"  reproducibility: {repro.get('band')} (score {repro.get('score')})")
        for check in repro.get("checklist", []):
            out.print(f"    {check['state']:>8}  {check['id']}: {check['reason']}")
    if logs:
        for name, path in (
            ("stdout", run_dir / "logs" / "stdout.log"),
            ("stderr", run_dir / "logs" / "stderr.log"),
        ):
            if path.is_file():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                out.print(f"  --- last {min(logs, len(lines))} {name} lines ---")
                for line in lines[-logs:]:
                    out.print(f"  {line}")


@app.command("doctor")
def doctor_command(
    check_llm: bool = typer.Option(False, "--check-llm", help="Actually call the configured endpoint."),
) -> None:
    """Report what LabBox can and cannot record in this environment."""
    layout: ProjectLayout | None = None
    problems: list[str] = []
    notes: list[str] = []
    out.print(f"labbox {__version__}  python {sys.version.split()[0]}  pid {os.getpid()}")
    try:
        layout = ProjectLayout.discover()
        out.print(f"[green]project[/] initialized at {layout.root}")
        config = LabboxConfig.load(layout.config_path)
        for problem in config.problems:
            problems.append(f"config: {problem}")
        for bad in config.validate_patterns():
            problems.append(f"config pattern: {bad}")
        out.print(
            "[green]config[/] "
            f"{layout.config_path} (sources: {', '.join(Path(p).name for p in config.source_paths) or 'defaults'})"
        )
    except LabBoxError as exc:
        config = LabboxConfig.default()
        notes.append(str(exc.message))
        out.print("[yellow]project[/] not initialized (run `labbox init`)")
    from labbox.infrastructure.git import GitClient
    from labbox.infrastructure.nvidia import get_backend

    git = GitClient(Path.cwd(), timeout=config.git.timeout_seconds)
    if git.available():
        version = git.version()
        out.print(f"[green]git[/] {version} at {Path.cwd()}")
        state = git.collect(include_diff_hash=False)
        if state.has_repo:
            out.print(
                f"  commit {state.short_commit} branch {state.branch} dirty={state.dirty}"
                + (f" (+{state.untracked_count} untracked)" if state.untracked_count else "")
            )
            if state.dirty:
                notes.append("working tree dirty: commits alone will not reproduce this run")
        else:
            notes.append("not a git repository: code state will be unknown for every run")
    else:
        problems.append("git executable not found: code state cannot be recorded")
    backend = get_backend(config.gpu.backend)
    try:
        gpus = backend.static_info()
    except Exception as exc:  # pragma: no cover
        gpus = []
        problems.append(f"gpu backend {getattr(backend, 'name', '?')} failed: {exc}")
    finally:
        backend.close()
    if gpus:
        out.print(
            f"[green]gpu[/] {getattr(backend, 'name', '?')}: "
            + "; ".join(f"{g.name} {(g.vram_bytes or 0) // 1024**3}GiB" for g in gpus)
        )
    else:
        notes.append("no NVIDIA GPU visible: GPU metrics and driver version will be recorded as unknown")
    try:
        import psutil

        memory = psutil.virtual_memory()
        out.print(f"[green]host[/] {psutil.cpu_count(logical=True)} logical CPUs, {memory.total / 1024**3:.0f} GiB RAM")
    except ImportError:
        problems.append("psutil missing in the LabBox environment: CPU/RAM sampling disabled")
    if config.llm.enabled:
        out.print(f"[cyan]llm[/] {config.llm.provider} {config.llm.base_url} model={config.llm.model}")
        if config.llm.api_key_env:
            if os.environ.get(config.llm.api_key_env):
                out.print(f"  api key found in ${config.llm.api_key_env}")
            else:
                # Previously this branch printed nothing at all, so a misconfigured
                # endpoint looked exactly like a healthy one until the first request failed.
                problems.append(
                    f"llm is enabled and api_key_env={config.llm.api_key_env!r} is not set; "
                    "every explain request will be rejected by the endpoint"
                )
        else:
            out.print("  no api_key_env configured: the endpoint is expected to be local or unauthenticated")
        if check_llm:
            from labbox.llm.provider import OpenAICompatibleProvider

            try:
                reply = OpenAICompatibleProvider.from_config(config).describe()
                out.print(f"  [green]reachable:[/] {reply}")
            except LabBoxError as exc:
                problems.append(f"llm: {exc.message}")
        else:
            notes.append("llm is enabled but no request was sent (use --check-llm to probe the endpoint)")
    else:
        out.print("[cyan]llm[/] disabled - every core feature works without it")
    runs = FilesystemStore(layout).list_run_dirs() if layout is not None else []
    if runs and layout is not None:
        run_store = FilesystemStore(layout)
        out.print(f"[green]runs[/] {len(runs)} recorded")
        stale = []
        for path in runs[-20:]:
            record = _load_record(run_store, path)
            if record is not None and record.status is RunStatus.RUNNING:
                stale.append(path.name)
        for name in stale:
            out.print(f"[yellow]stale[/] {name} 状态仍是 running：LabBox 当时被终止，finalize 之后的文件不会补齐")
    for note in notes:
        out.print(f"[yellow]note:[/] {note}")
    for problem in problems:
        out.print(f"[red]problem:[/] {problem}")
    raise typer.Exit(code=1 if problems else 0)


@app.command("compare")
def compare_command(
    left: str = typer.Argument(..., help="First run id or name."),
    right: str = typer.Argument(..., help="Second run id or name."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable comparison."),
    save: bool = typer.Option(False, "--save", help="Also write compare.json into the second run directory."),
) -> None:
    """List the differences between two runs as citable facts."""
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    from labbox.services.compare_service import compare_runs, render

    try:
        comparison = compare_runs(layout, left, right, save=save)
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    if json_out:
        import json

        _emit(json.dumps(comparison.as_dict(), ensure_ascii=False, indent=2))
        return
    text = render(comparison)
    for line in text.splitlines():
        if line.strip().startswith("[HIGH]") or "[CRITICAL]" in line:
            err.print(f"[red]{line}[/red]")
        elif "[WARNING]" in line:
            err.print(f"[yellow]{line}[/yellow]")
        elif line and not line.startswith(" "):
            err.print(f"[bold]{line}[/bold]")
        else:
            err.print(line)
    if comparison.saved_to:
        err.print(f"\nlabbox: wrote {comparison.saved_to.name} to {comparison.saved_to.parent.name}/")
    raise typer.Exit(code=0)


@app.command("report")
def report_command(
    run: str = typer.Argument(..., help="Run id or name."),
    open_browser: bool = typer.Option(False, "--open", help="Ask the OS to open the report."),
) -> None:
    """Write an offline, self-contained HTML report for one run."""
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    from labbox.report.html import write_report

    store = FilesystemStore(layout)
    run_dir = layout.resolve_run_id(run)
    try:
        path = write_report(store, layout.run_dir(run_dir.name), **_optional_explanation(store, run_dir))
    except LabBoxError as exc:
        _fail(f"report generation failed: {exc.message}", code=exc.code, exc=exc)
    err.print(f"[green]labbox: wrote {path}[/green]")
    if open_browser:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    raise typer.Exit(code=0)


def _optional_explanation(store: FilesystemStore, run_dir: Path) -> dict[str, object]:
    """Fold a stored explain.json into the report; a bad one is simply left out."""
    from labbox.domain.explain import ExplainResult

    data = store.read_json(run_dir / "explain.json", strict=False) or {}
    try:
        return {"explanation": ExplainResult.model_validate(data.get("explanation") or {})}
    except Exception:
        return {}


@app.command("explain")
def explain_command(
    run: str = typer.Argument(..., help="Run id or name to explain."),
    against: str = typer.Option(None, "--against", help="Explain the difference to this second run."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be sent to the model; send nothing."),
    include_logs: bool = typer.Option(False, "--send-logs", help="Allow a sanitized log tail (needs [llm] send_logs)."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable envelope."),
) -> None:
    """Ask the (optional) model to interpret the recorded facts, then verify its citations."""
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    from labbox.services.explain_service import explain_run, format_outcome

    try:
        outcome = explain_run(
            layout,
            run,
            against=against,
            dry_run=dry_run,
            include_logs=include_logs,
        )
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    if json_out:
        _emit(json.dumps(outcome.envelope.model_dump(mode="json"), ensure_ascii=False, indent=2))
        raise typer.Exit(code=0)
    if dry_run:
        err.print("[bold]Dry run: nothing was sent.[/bold]")
        err.print("data the endpoint would receive:")
        for category in outcome.categories:
            err.print(f"  - {category}")
        err.print(json.dumps(outcome.payload_preview, ensure_ascii=False, indent=2))
        if outcome.refused:
            err.print(f"[yellow]would refuse: {outcome.refused}[/yellow]")
        raise typer.Exit(code=0)
    for line in format_outcome(outcome).splitlines():
        err.print(line)
    if outcome.refused and not outcome.sent:
        raise typer.Exit(code=0)
    raise typer.Exit(code=0)


@app.command("clean")
def clean_command(
    runs: list[str] = typer.Argument(None, help="Run ids or names to delete."),
    list_only: bool = typer.Option(False, "--list", help="Show what is stored and how much it costs."),
    keep_last: int = typer.Option(None, "--keep-last", help="Keep only the N newest runs."),
    older_than: str = typer.Option(None, "--older-than", help="Delete runs older than Nd / Ns (days/seconds)."),
    all_runs: bool = typer.Option(False, "--all", help="Delete every recorded run."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be deleted."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Reclaim disk space and remove recorded data that may contain secrets (§4.7)."""
    try:
        layout = ProjectLayout.discover()
    except LabBoxError as exc:
        _fail(str(exc.message), code=exc.code, exc=exc)
    store = FilesystemStore(layout)
    existing = store.list_run_dirs()
    skipped: list[str] = []
    if list_only:
        rows = sorted(existing, key=lambda p: -store.dir_size_bytes(p))
        table = Table(show_header=True, header_style="bold")
        for column in ("run", "size", "since"):
            table.add_column(column)
        for run_dir in rows:
            record = _load_record(store, run_dir)
            table.add_row(
                run_dir.name,
                f"{store.dir_size_bytes(run_dir) / 1024**2:.1f} MiB",
                record.started_at.isoformat(timespec="seconds") if record else "-",
            )
        out.print(table)
        out.print(f"total {store.total_store_bytes() / 1024**2:.1f} MiB in {len(existing)} runs")
        return

    victims: list[Path] = []
    if all_runs:
        victims = list(existing)
    elif keep_last is not None:
        if keep_last < 0:
            _fail(f"--keep-last must be 0 or more, got {keep_last}")
        victims = existing[:-keep_last] if keep_last > 0 else list(existing)
    elif older_than is not None:
        try:
            cutoff = _parse_age(older_than)
        except ValueError as exc:
            _fail(f"invalid --older-than {older_than!r}: {exc}")
        for run_dir in existing:
            record = _load_record(store, run_dir)
            # A run whose record cannot be read has no start time to compare, and deleting
            # it on a guess is the one thing `clean` must never do silently. Keep it and say
            # which one was skipped.
            if record is None:
                skipped.append(run_dir.name)
                continue
            if record.started_at < cutoff:
                victims.append(run_dir)
    elif runs:
        for query in runs:
            try:
                victims.append(layout.resolve_run_id(query))
            except LabBoxError as exc:
                _fail(str(exc.message), code=exc.code, exc=exc)
    else:
        _fail("nothing to do; pass a run, or use --list / --keep-last / --all")

    victims = _dedupe(victims)
    for name in skipped:
        err.print(f"[yellow]labbox: keeping {name}: its run.json could not be read[/yellow]")
    if not victims:
        out.print("labbox: nothing matched")
        return
    total = sum(store.dir_size_bytes(path) for path in victims)
    for path in victims:
        out.print(
            f"{'would delete' if dry_run else 'deleting'} {path.name} ({store.dir_size_bytes(path) / 1024**2:.1f} MiB)"
        )
    if dry_run:
        out.print(f"total {total / 1024**2:.1f} MiB")
        return
    if not yes:
        answer = typer.confirm(f"Delete {len(victims)} run(s), {total / 1024**2:.1f} MiB? This cannot be undone.")
        if not answer:
            out.print("labbox: cancelled")
            raise typer.Exit(code=0)
    deleted = 0
    for path in victims:
        try:
            # A directory that has already gone is not a failure; it just was not there.
            if store.delete_run(path):
                deleted += 1
        except LabBoxError as exc:
            _fail(f"could not delete {path.name}: {exc.message}", code=exc.code, exc=exc)
    out.print(f"[green]labbox: deleted {deleted} run(s)[/]")
    if deleted != len(victims):
        out.print(f"labbox: {len(victims) - deleted} were already gone")


def _dedupe(paths: list[Path]) -> list[Path]:
    """Delete each run at most once, however many ways it was matched."""
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _parse_age(spec: str) -> Any:
    """Parse ``Nd`` / ``Nh`` / ``Nm`` / ``Ns`` into a cutoff timestamp.

    Raises ``ValueError`` on anything else rather than letting ``float()`` raise: an
    unparsable ``--older-than`` used to abort the whole command with a traceback, which
    read as a crash rather than as "that is not a duration".
    """
    from datetime import datetime, timedelta

    text = spec.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([dhms])", text)
    if not match:
        raise ValueError("use Nd / Nh / Nm / Ns, for example 30d or 12h")
    amount = float(match.group(1))
    unit = match.group(2)
    if amount <= 0:
        raise ValueError("the duration must be greater than zero")
    delta = {
        "d": timedelta(days=amount),
        "h": timedelta(hours=amount),
        "m": timedelta(minutes=amount),
        "s": timedelta(seconds=amount),
    }[unit]
    return datetime.now().astimezone() - delta


def _relative(root: Path, target: Path) -> str:
    try:
        return str(target.relative_to(root))
    except ValueError:
        return str(target)


def main() -> None:  # pragma: no cover - console_script entry point
    _configure_output_streams()
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
