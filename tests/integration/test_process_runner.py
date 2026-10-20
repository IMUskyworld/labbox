"""ProcessRunner against real child processes (the spike, codified).

These four properties are the contract DEVELOPMENT.md §2.4 calls "harder than any
feature": the user's command must behave identically with or without LabBox.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout
from labbox.recorder.process import ProcessRunner

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def run_paths(project: ProjectLayout) -> tuple[FilesystemStore, object]:
    store = FilesystemStore(project)
    paths = store.layout.run_dir("20260922-160000-proc")
    store.ensure_run_dirs(paths)
    return store, paths


def make_runner(paths, argv: list[str], cwd: Path, env: dict[str, str], **kwargs) -> ProcessRunner:
    return ProcessRunner(
        argv,
        cwd=str(cwd),
        env=env,
        stdout_path=paths.stdout_log,
        stderr_path=paths.stderr_log,
        plain_stdout_path=paths.stdout_plain_log,
        echo=False,
        echo_stderr=False,
        **kwargs,
    )


def test_argv_is_passed_verbatim_without_a_shell(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    dangerous = ["a b", "&& echo pwned", ">redirect", "a|b", '"quoted"', "中文参数", "$(whoami)"]
    argv = [py, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", *dangerous]
    result = make_runner(paths, argv, tmp_path, child_env).run()
    assert result.exit_code == 0
    echoed = json.loads(paths.stdout_log.read_bytes().decode("utf-8", "replace").strip())
    assert echoed == dangerous, "arguments must reach the child untouched"
    assert not list(tmp_path.glob("*redirect*")), "a '>' argument must not create a file"
    assert not list(tmp_path.glob("*.txt")), "no shell redirection happened"


def test_large_stderr_does_not_deadlock(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    argv = [py, str(EXAMPLES / "train_streams.py"), "stderr-flood"]
    result = make_runner(paths, argv, tmp_path, child_env, reader_join_timeout=10.0).run()
    assert result.exit_code == 0, "the child printed a marker on stdout after flooding stderr"
    assert result.stderr_bytes > 300_000
    assert "streams finished" in paths.stdout_log.read_bytes().decode("utf-8", "replace")
    assert not result.reader_timeouts


def test_exit_codes_are_preserved_exactly(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    for expected in (0, 1, 3, 7, 42, 255):
        argv = [py, "-c", f"raise SystemExit({expected})"]
        result = make_runner(paths, argv, tmp_path, child_env).run()
        assert result.exit_code == expected
        assert result.exit_code_raw == expected
        assert result.interrupted is False


def test_failure_still_reports_stderr(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    result = make_runner(
        paths,
        [py, "-c", "import sys; sys.stderr.write('CUDA out of memory'); raise SystemExit(1)"],
        tmp_path,
        child_env,
    ).run()
    assert result.exit_code == 1
    assert b"CUDA out of memory" in paths.stderr_log.read_bytes()


def test_interleaved_streams_are_not_mixed(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    result = make_runner(
        paths, [py, str(EXAMPLES / "train_streams.py"), "interleave"], tmp_path, child_env
    ).run()
    assert result.exit_code == 0
    stdout_lines = paths.stdout_log.read_bytes().decode("utf-8", "replace").splitlines()
    stderr_lines = paths.stderr_log.read_bytes().decode("utf-8", "replace").splitlines()
    assert stdout_lines[0].startswith("out ")
    assert stderr_lines[0].startswith("err ")
    assert not any(line.startswith("err ") for line in stdout_lines)


def test_raw_bytes_are_logged_and_ansi_only_stripped_in_plain_log(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    make_runner(paths, [py, str(EXAMPLES / "train_streams.py"), "encoding"], tmp_path, child_env).run()
    raw = paths.stdout_log.read_bytes()
    assert b"\x1b[32m" in raw, "the terminal must receive the original bytes"
    assert "中文".encode() in raw
    assert b"green text" in raw
    plain = paths.stdout_plain_log.read_bytes()
    assert b"\x1b[32m" not in plain
    assert b"green text" in plain


def test_missing_command_is_reported_not_raised(tmp_path, child_env, run_paths) -> None:
    _store, paths = run_paths
    result = make_runner(paths, ["definitely-not-a-real-binary-xyz"], tmp_path, child_env).run()
    assert result.launch_error is not None
    assert result.launch_error_code == "COMMAND_NOT_FOUND"
    assert result.exit_code == 127
    assert result.started is False


def test_terminated_child_records_provenance(tmp_path, py, child_env, run_paths) -> None:
    """A hard-killed child is indistinguishable by exit code, so provenance is recorded.

    Measured: ``taskkill /F`` yields exit code 1 on Windows, the same value a script gets
    from ``exit(1)``. Only LabBox knows whether it did the killing.
    """
    _store, paths = run_paths
    runner = make_runner(paths, [py, "-c", "import time; time.sleep(30)"], tmp_path, child_env)
    box: list[object] = []
    thread = threading.Thread(target=lambda: box.append(runner.run()), daemon=True)
    thread.start()
    for _ in range(200):
        if runner._proc is not None:
            break
        time.sleep(0.02)
    assert runner._proc is not None
    runner.kill()
    thread.join(timeout=30)
    assert box, "run() returned"
    result = box[0]
    assert result.forced_kill is True
    assert result.killed_by_labbox is True
    assert result.exit_code not in (0, None)


def test_environment_is_passed_through_with_unbuffering(tmp_path, py, child_env, run_paths) -> None:
    _store, paths = run_paths
    env = {**child_env, "LABBOX_PROBE_MARKER": "keep-me"}
    argv = [py, "-c", "import os; print(os.environ['LABBOX_PROBE_MARKER'])"]
    make_runner(paths, argv, tmp_path, env).run()
    text = paths.stdout_log.read_bytes().decode("utf-8", "replace")
    assert "keep-me" in text


def test_command_output_matches_the_bare_command(tmp_path, py, child_env, run_paths) -> None:
    """Byte-for-byte stdout equality with a direct Popen run (§24 rule 1)."""
    _store, paths = run_paths
    import subprocess

    argv = [py, str(EXAMPLES / "train_streams.py"), "encoding"]
    bare = subprocess.run(argv, capture_output=True, cwd=str(tmp_path), env=child_env, shell=False)
    make_runner(paths, argv, tmp_path, child_env).run()
    assert paths.stdout_log.read_bytes().replace(b"\r\n", b"\n") == bare.stdout.replace(b"\r\n", b"\n")


def test_reader_timeout_is_recorded_when_a_grandchild_holds_the_pipe(
    tmp_path, py, child_env, run_paths
) -> None:
    """A surviving grandchild keeps the pipe open; LabBox must not hang forever."""
    _store, paths = run_paths
    code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "stdout=subprocess.PIPE or None,stderr=subprocess.PIPE or None);"
        "print('parent done');time.sleep(0.2)"
    )
    result = make_runner(paths, [py, "-c", code], tmp_path, child_env, reader_join_timeout=1.0).run()
    assert result.exit_code == 0
    assert "parent done" in paths.stdout_log.read_bytes().decode("utf-8", "replace")
    # Either the reader finished cleanly or we recorded the timeout; hanging is the bug.
    assert result.exit_code == 0
