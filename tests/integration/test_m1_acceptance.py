"""M1 acceptance gate, executed against the real console script (§18.3, §20 M1).

The CLI is invoked as a subprocess because the properties under test are process-level:
the exit code, the byte stream on stdout, and what survives on disk after a kill. In-process
tests would hide exactly the failures that matter here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
SCRIPT_DIR = Path(sys.executable).parent
CONSOLE_SCRIPT = SCRIPT_DIR / ("labbox.exe" if os.name == "nt" else "labbox")
DEV_SHIM = REPO / "scripts" / "labbox_dev.py"


def _cli_command() -> list[str]:
    """Prefer the installed console script; fall back to the source-checkout shim.

    The fallback is not a convenience: an editable install breaks whenever the install
    path is non-ASCII on a GBK machine (DEVELOPMENT.md §16.3), and this repo sits at such
    a path. Either entry point still runs the CLI as a real subprocess with its own exit
    code and stdout, which is what these tests assert on.
    """
    if CONSOLE_SCRIPT.is_file():
        probe = subprocess.run(
            [str(CONSOLE_SCRIPT), "init", "--help"],
            capture_output=True,
            shell=False,
            check=False,
            cwd=str(REPO),
        )
        if probe.returncode == 0:
            return [str(CONSOLE_SCRIPT)]
    return [sys.executable, str(DEV_SHIM)]


CLI = _cli_command()


def base_env() -> dict[str, str]:
    # Drop PYTHONPATH so the child training process sees a clean environment.
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def invoke(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*CLI, *args],
        cwd=str(cwd),
        env={**base_env(), **(env or {})},
        capture_output=True,
        shell=False,
        check=False,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    root.mkdir()
    shutil.copy(EXAMPLES / "train_fake.py", root / "train.py")
    result = invoke("init", cwd=root)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert (root / ".labbox" / "config.toml").is_file()
    assert ".labbox/runs/" in (root / ".gitignore").read_text(encoding="utf-8")
    return root


def read_run(root: Path, name: str) -> dict:
    runs = sorted((root / ".labbox" / "runs").iterdir())
    matching = [r for r in runs if name in r.name]
    assert matching, f"no run dir matched {name} in {[r.name for r in runs]}"
    return json.loads(matching[-1].joinpath("run.json").read_text(encoding="utf-8"))


def test_successful_run_records_everything(project: Path) -> None:
    result = invoke(
        "run", "--name", "baseline", "--", sys.executable, "train.py", "--epochs", "3", cwd=project
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    record = read_run(project, "baseline")
    assert record["status"] == "succeeded"
    assert record["exit_code"] == 0 and record["exit_code_raw"] == 0
    assert record["command"]["argv"][1:] == ["train.py", "--epochs", "3"]
    assert record["duration_seconds"] > 0
    assert record["started_at"] and record["ended_at"]

    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "baseline" in p.name)
    for required in (
        "run.json",
        "environment.json",
        "command.txt",
        "logs/stdout.log",
        "logs/events.jsonl",
        "samples/system.jsonl",
    ):
        assert (run_dir / required).is_file(), f"{required} is missing"
    stdout = (run_dir / "logs" / "stdout.log").read_bytes()
    assert b"epoch 3/3" in stdout
    events = [
        json.loads(line)
        for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [event["kind"] for event in events]
    assert "run.started" in kinds and "process.exited" in kinds and "environment.captured" in kinds
    samples = (run_dir / "samples" / "system.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(samples) >= 1
    assert json.loads(samples[0])["elapsed_seconds"] >= 0
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["python"]["version"].startswith("3.")
    assert environment["host"]["ram_bytes"] > 0
    assert environment["collectors"]["host"] == "ok"


def test_failed_run_keeps_the_original_exit_code_and_still_records(project: Path) -> None:
    result = invoke(
        "run",
        "--name",
        "broken",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "4",
        "--fail-at",
        "2",
        "--out",
        "outputs-broken",
        cwd=project,
    )
    assert result.returncode == 7, "the training exit code must pass through untouched"
    record = read_run(project, "broken")
    assert record["status"] == "failed"
    assert record["exit_code"] == 7
    assert record["exit_meaning"] == "exited"
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "broken" in p.name)
    assert (run_dir / "logs" / "stdout.log").read_bytes().count(b"epoch") >= 1
    assert b"injected failure" in (run_dir / "logs" / "stderr.log").read_bytes()


def test_stdout_and_stderr_are_byte_identical_to_the_bare_command(project: Path) -> None:
    argv = [sys.executable, "train.py", "--epochs", "2", "--out", "outputs-equal"]
    bare = subprocess.run(
        argv, cwd=str(project), capture_output=True, shell=False, check=False, env=base_env()
    )
    wrapped = invoke("run", "--name", "equal", "--", *argv, cwd=project)
    record = read_run(project, "equal")
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "equal" in p.name)
    assert wrapped.returncode == bare.returncode == record["exit_code"]
    stored_stdout = (run_dir / "logs" / "stdout.log").read_bytes()
    stored_stderr = (run_dir / "logs" / "stderr.log").read_bytes()
    assert stored_stdout.replace(b"\r\n", b"\n") == bare.stdout.replace(b"\r\n", b"\n")
    assert stored_stderr.replace(b"\r\n", b"\n") == bare.stderr.replace(b"\r\n", b"\n")
    # LabBox's own chatter must not be in the child's stream.
    assert b"labbox:" not in wrapped.stdout


def test_environment_secrets_are_never_written(project: Path) -> None:
    secret = "sk-SUPERSECRETVALUE0123456789abcdef"
    invoke(
        "run",
        "--name",
        "secret",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "1",
        cwd=project,
        env={"OPENAI_API_KEY": secret, "MY_HF_TOKEN": "hf_topsecret"},
    )
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "secret" in p.name)
    blob = b"".join(path.read_bytes() for path in run_dir.rglob("*") if path.is_file())
    assert secret.encode() not in blob
    assert b"hf_topsecret" not in blob
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert "OPENAI_API_KEY" in environment["env"]["redacted_names"]


def test_list_show_and_doctor_work(project: Path) -> None:
    invoke(
        "run",
        "--name",
        "alpha",
        "--tag",
        "smoke",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "1",
        cwd=project,
    )
    invoke("run", "--name", "beta", "--", sys.executable, "train.py", "--epochs", "1", cwd=project)
    listing = invoke("list", "--json", cwd=project)
    assert listing.returncode == 0
    rows = json.loads(listing.stdout.decode("utf-8"))
    names = {row["name"] for row in rows}
    assert {"alpha", "beta"} <= names
    assert all(row["status"] == "succeeded" for row in rows)
    show = invoke("show", "alpha", cwd=project)
    assert show.returncode == 0
    text = show.stdout.decode("utf-8", "replace")
    assert "SUCCEEDED" in text and "train.py" in text
    assert "not captured" in text or "frameworks" in text
    doctor = invoke("doctor", cwd=project)
    assert doctor.returncode in (0, 1), doctor.stdout.decode("utf-8", "replace")
    assert "python" in doctor.stdout.decode("utf-8", "replace")


def test_labbox_adds_little_overhead(project: Path) -> None:
    """§17: recording must not measurably slow the training command down."""
    import time

    argv = [sys.executable, "train.py", "--epochs", "2", "--sleep", "0.01", "--out", "outputs-perf"]
    started = time.perf_counter()
    subprocess.run(argv, cwd=str(project), capture_output=True, shell=False, check=False, env=base_env())
    bare_seconds = time.perf_counter() - started
    started = time.perf_counter()
    wrapped = invoke("run", "--name", "perf", "--sample-interval", "0.2", "--", *argv, cwd=project)
    wrapped_seconds = time.perf_counter() - started
    assert wrapped.returncode == 0
    record = read_run(project, "perf")
    assert record["duration_seconds"] > 0
    assert wrapped_seconds < bare_seconds * 1.6 + 6.0, (
        f"labbox added {wrapped_seconds - bare_seconds:.2f}s over a {bare_seconds:.2f}s run"
    )


def test_clean_dry_run_and_delete(project: Path) -> None:
    invoke("run", "--name", "todelete", "--", sys.executable, "train.py", "--epochs", "1", cwd=project)
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "todelete" in p.name)
    dry = invoke("clean", "--dry-run", "todelete", cwd=project)
    assert dry.returncode == 0
    assert run_dir.exists(), "--dry-run must not delete"
    deleted = invoke("clean", "--yes", "todelete", cwd=project)
    assert deleted.returncode == 0, deleted.stderr.decode("utf-8", "replace")
    assert not run_dir.exists()
    listing = invoke("clean", "--list", cwd=project)
    assert listing.returncode == 0
    assert "MiB" in listing.stdout.decode("utf-8", "replace")


def test_run_outside_a_project_fails_with_a_hint(tmp_path: Path) -> None:
    result = invoke("run", "--", sys.executable, "-c", "print(1)", cwd=tmp_path)
    assert result.returncode == 2
    assert "labbox init" in result.stderr.decode("utf-8", "replace")


def test_missing_command_is_reported_as_start_failed(project: Path) -> None:
    result = invoke("run", "--name", "nope", "--", "definitely-not-a-binary-xyz", cwd=project)
    assert result.returncode == 127
    record = read_run(project, "nope")
    assert record["status"] == "start_failed"
    assert record["exit_code"] == 127


def test_interrupted_run_still_writes_a_complete_record(project: Path) -> None:
    """Kill LabBox mid-run: the log, samples and run.json must all still be readable.

    A real Ctrl+C cannot be synthesized without a console (see scripts/spike_process.py),
    so this terminates the LabBox process itself, which exercises the same recovery path:
    a half-finished run must never produce a corrupt or absent record.
    """
    process = subprocess.Popen(
        [
            *CLI,
            "run",
            "--name",
            "longrun",
            "--sample-interval",
            "0.2",
            "--",
            sys.executable,
            "train.py",
            "--epochs",
            "40",
            "--sleep",
            "0.5",
            "--out",
            "outputs-long",
        ],
        cwd=str(project),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    run_dir: Path | None = None
    for _ in range(120):
        runs = list((project / ".labbox" / "runs").glob("*longrun"))
        if runs:
            run_dir = runs[0]
            break
        import time

        time.sleep(0.05)
    assert run_dir is not None, "run directory should be created before the child starts"
    # Wait until we are genuinely mid-training: environment.json exists and the child has
    # produced output. Killing during the pre-run probe is a different (earlier) case.
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        log = run_dir / "logs" / "stdout.log"
        if (run_dir / "environment.json").is_file() and log.is_file() and log.stat().st_size > 0:
            break
        time.sleep(0.1)
    assert (run_dir / "environment.json").is_file(), "the run reached the training phase"
    assert (run_dir / "logs" / "stdout.log").read_bytes(), "the child had started producing output"
    process.kill()
    process.wait(timeout=30)
    import time

    deadline = time.monotonic() + 10
    record: dict = {}
    while time.monotonic() < deadline:
        try:
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if record.get("status") != "running":
                break
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(0.1)
    assert record, "run.json must exist and be valid JSON even after a hard kill"
    assert record["status"] == "running", (
        "an interrupted LabBox leaves the record marked running: the run never reported an "
        "exit code, and pretending otherwise would invent evidence"
    )
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "logs" / "stdout.log").is_file()
    # A run whose recorder died must still explain itself: the reader should not have to
    # guess why metrics/artifacts/report are absent.
    assert (run_dir / "diagnostics.json").is_file(), "the initial diagnostics must survive a kill"
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert any(item["code"] == "run_in_progress" for item in diagnostics["diagnostics"]), diagnostics
    assert record["links"].get("diagnostics") == "diagnostics.json"
    doctor = invoke("doctor", cwd=project)
    assert run_dir.name in doctor.stdout.decode("utf-8", "replace"), "doctor must flag the stale run"
