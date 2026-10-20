"""Command-line behaviour that has to fail loudly instead of guessing.

These run the real console script (or the source-checkout shim) as a subprocess, because
the properties under test are the exit code and the bytes on the streams.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labbox import __version__

REPO = Path(__file__).resolve().parents[2]
CONSOLE_SCRIPT = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
DEV_SHIM = REPO / "scripts" / "labbox_dev.py"


def _cli() -> list[str]:
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


CLI = _cli()


def invoke(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [*CLI, *args], cwd=str(cwd), env={**clean, **(env or {})}, capture_output=True, shell=False, check=False
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    assert invoke("init", cwd=root).returncode == 0
    return root


def test_version_works_outside_a_project(tmp_path: Path) -> None:
    result = invoke("--version", cwd=tmp_path)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8").strip() == f"labbox {__version__}"


def make_run(project: Path, name: str, *, config: str | None = None) -> Path:
    result = invoke("run", "--name", name, "--quiet", "--", sys.executable, "-c", "print('ok')", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if name in p.name)
    if config is not None:
        text = (project / ".labbox" / "config.toml").read_text(encoding="utf-8")
        assert config in text
    return run_dir


def test_doctor_prints_chinese_on_a_non_utf8_console(project: Path) -> None:
    """A cp1252 console must not truncate the stale-run line (§CI windows-latest)."""
    run_dir = make_run(project, "stale")
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["status"] = "running"
    (run_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    result = invoke("doctor", cwd=project, env={"PYTHONIOENCODING": "cp1252"})
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, stderr
    assert run_dir.name in stdout, stdout
    assert "UnicodeEncodeError" not in stdout + stderr


# ---------------------------------------------------------------------------
# clean --older-than
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", ["soon", "30", "d", "-5d", "0d", "5 weeks"])
def test_an_unparsable_older_than_is_a_clean_error_not_a_crash(
    project: Path, spec: str
) -> None:
    run_dir = make_run(project, "keepme")
    result = invoke("clean", "--yes", "--older-than", spec, cwd=project)
    assert result.returncode == 2, result.stderr.decode("utf-8", "replace")
    assert "older-than" in result.stderr.decode("utf-8", "replace")
    assert "Traceback" not in result.stderr.decode("utf-8", "replace")
    assert run_dir.exists(), "an invalid selector must never delete anything"


def test_a_valid_older_than_still_works(project: Path) -> None:
    run_dir = make_run(project, "aged")
    result = invoke("clean", "--yes", "--older-than", "0.001s", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert not run_dir.exists()


def test_a_run_with_an_unreadable_record_is_kept_not_deleted(project: Path) -> None:
    run_dir = make_run(project, "corrupt")
    (run_dir / "run.json").write_text("{ not json", encoding="utf-8")
    result = invoke("clean", "--yes", "--older-than", "1s", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert run_dir.exists(), "an unreadable record has no timestamp to compare"
    assert "could not be read" in text


def test_named_duplicates_are_deleted_once(project: Path) -> None:
    run_dir = make_run(project, "dup")
    result = invoke("clean", "--yes", "dup", "dup", run_dir.name, cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "deleted 1 run(s)" in result.stdout.decode("utf-8", "replace")
    assert not run_dir.exists()


# ---------------------------------------------------------------------------
# list --limit
# ---------------------------------------------------------------------------
def test_limit_zero_lists_nothing(project: Path) -> None:
    make_run(project, "one")
    result = invoke("list", "--limit", "0", "--json", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert json.loads(result.stdout.decode("utf-8")) == []


def test_a_negative_limit_is_rejected(project: Path) -> None:
    make_run(project, "two")
    result = invoke("list", "--limit", "-1", cwd=project)
    assert result.returncode == 2
    assert "limit" in result.stderr.decode("utf-8", "replace")


def test_limit_is_honoured(project: Path) -> None:
    for name in ("l1", "l2", "l3"):
        make_run(project, name)
    result = invoke("list", "--limit", "2", "--json", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert len(json.loads(result.stdout.decode("utf-8"))) == 2


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def test_doctor_says_when_the_api_key_variable_is_not_set(project: Path) -> None:
    config = project / ".labbox" / "config.toml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("[llm]\nenabled = false", "[llm]\nenabled = true")
    text = text.replace(
        'base_url = "http://127.0.0.1:8000/v1"',
        'base_url = "http://127.0.0.1:8000/v1"\napi_key_env = "LABBOX_TEST_KEY"',
    )
    config.write_text(text, encoding="utf-8", newline="\n")
    result = invoke("doctor", cwd=project, env={"LABBOX_TEST_KEY": ""})
    stdout = result.stdout.decode("utf-8", "replace")
    assert "api_key_env" in stdout, "a missing key must be named, not left silent"
    assert result.returncode == 1


def test_doctor_reports_a_present_key_without_printing_it(project: Path) -> None:
    config = project / ".labbox" / "config.toml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("[llm]\nenabled = false", "[llm]\nenabled = true")
    text = text.replace(
        'base_url = "http://127.0.0.1:8000/v1"',
        'base_url = "http://127.0.0.1:8000/v1"\napi_key_env = "LABBOX_TEST_KEY"',
    )
    config.write_text(text, encoding="utf-8", newline="\n")
    result = invoke("doctor", cwd=project, env={"LABBOX_TEST_KEY": "super-secret-value"})
    stdout = result.stdout.decode("utf-8", "replace")
    assert "super-secret-value" not in stdout
    assert "LABBOX_TEST_KEY" in stdout
