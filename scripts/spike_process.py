"""Windows process-semantics spike (DEVELOPMENT.md §23 step 2b).

Answers, on a real Windows console, the four questions ProcessRunner cannot be written
against by reasoning alone:

1. is an argv list passed verbatim (no shell interpretation)?
2. does a child that floods stderr while the parent reads stdout deadlock?
3. what exit codes does the OS actually report for SystemExit / crash / terminate / Ctrl+C,
   and does ``normalize_exit_code`` map them onto what a shell would show?
4. are UTF-8, GBK and ANSI escapes preserved byte-for-byte in the log?

Run: ``python scripts/spike_process.py``
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labbox.recorder.process import ProcessRunner, normalize_exit_code, strip_ansi

# Print our own findings as UTF-8 with replacement so a GBK console cannot crash the
# spike. The *child* keeps the native locale encoding on purpose (see _child_env).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
TMP = Path(__file__).resolve().parent / ".spike"
TMP.mkdir(exist_ok=True)
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail[:160]}")


def _child_env() -> dict[str, str]:
    """Drop our own UTF-8 override so the child uses the machine's locale encoding."""
    return {k: v for k, v in os.environ.items() if k not in {"PYTHONIOENCODING", "PYTHONUTF8"}}


def clear_logs() -> None:
    for name in ("out.log", "err.log", "out.plain.log"):
        (TMP / name).write_bytes(b"")


def run(argv: list[str], **kwargs) -> tuple[object, ProcessRunner]:
    # Each case starts from an empty log; otherwise a later assertion passes because of
    # bytes a previous case left behind.
    clear_logs()
    runner = ProcessRunner(
        argv,
        env=_child_env(),
        cwd=str(TMP),
        stdout_path=TMP / "out.log",
        stderr_path=TMP / "err.log",
        plain_stdout_path=TMP / "out.plain.log",
        echo=False,
        echo_stderr=False,
        **kwargs,
    )
    return runner.run(), runner


PY = sys.executable

# --- 1. argv verbatim --------------------------------------------------------
danger = ["a b", "&& echo pwned", ">redirect", "a|b", '"quoted"', "中文参数", "$(whoami)", "*"]
# JSON so the child's output stays ASCII: a GBK console cannot otherwise be told apart
# from a broken argv.
res, _ = run([PY, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", *danger])
raw = (TMP / "out.log").read_bytes()
echoed = json.loads(raw.decode("utf-8", "replace").strip() or "[]")
check("argv passed verbatim", echoed == danger, f"echoed={echoed[1:3]}...")
check(
    "no shell metacharacter execution",
    not list(TMP.glob("*redirect*")) and not list(TMP.glob("pwned*")) and echoed == danger,
    "literals reached the child intact (CJK included); no file created, echo never ran",
)
# --- 2. pipe deadlock --------------------------------------------------------
TMP.joinpath("out.log").write_bytes(b"")
res, _ = run([PY, "-c", "import sys; sys.stderr.write('x'*(400*1024)); sys.stderr.flush(); print('alive')"])
stdout_text = (TMP / "out.log").read_bytes().decode("utf-8", "replace")
check(
    "400KB stderr does not deadlock", "alive" in stdout_text and res.exit_code == 0, f"code={res.exit_code}"
)

# --- 3. exit codes -----------------------------------------------------------
cases = {
    "success": ([PY, "-c", "raise SystemExit(0)"], 0),
    "exit-7": ([PY, "-c", "raise SystemExit(7)"], 7),
    "exit-255": ([PY, "-c", "raise SystemExit(255)"], 255),
    "traceback": ([PY, "-c", "raise ValueError('boom')"], 1),
}
for label, (argv, expected) in cases.items():
    res, _ = run(argv)
    check(f"exit code {label}", res.exit_code == expected, f"raw={res.exit_code_raw} got={res.exit_code}")

# Measured: os._exit(0xC000013A) is NOT a way to synthesise a Ctrl+C status - the OS
# reports exit code 1. The mapping below is therefore checked as a pure function here, and
# end-to-end by the console case at the bottom of this file.
res, _ = run([PY, "-c", "import os; os._exit(0xC000013A)"])
check(
    "os._exit cannot fake the Ctrl+C status (documented limitation)",
    res.exit_code_raw == 1,
    f"Popen reported raw={res.exit_code_raw}",
)
check(
    "0xC000013A normalizes to 130 (both signs)",
    normalize_exit_code(-1073741510)[0] == 130 and normalize_exit_code(3221225786)[0] == 130,
    f"signed={normalize_exit_code(-1073741510)} unsigned={normalize_exit_code(3221225786)}",
)

# external hard kill (what a user's Taskkill / process explorer does)
runner = ProcessRunner(
    [PY, "-c", "import time; time.sleep(30)"],
    echo=False,
    stdout_path=TMP / "out.log",
    stderr_path=TMP / "err.log",
)
started = time.monotonic()
res_a: list[object] = []


def _spawn_and_wait() -> None:
    res_a.append(runner.run())


t = threading.Thread(target=_spawn_and_wait, daemon=True)
t.start()
time.sleep(0.6)
if runner._proc is not None:
    subprocess.run(["taskkill", "/F", "/PID", str(runner._proc.pid)], capture_output=True)
t.join(timeout=20)
resk = res_a[0] if res_a else None
check(
    "external kill is recorded, not swallowed",
    resk is not None and resk.exit_code is not None and resk.exit_code != 0,
    f"raw={getattr(resk, 'exit_code_raw', None)} normalized={getattr(resk, 'exit_code', None)}",
)

# --- 4. encodings ------------------------------------------------------------
payload = (
    "import sys\n"
    "data = b'utf8:\\xe4\\xb8\\xad\\xe6\\x96\\x87 gbk:\\xd6\\xd0\\xce\\xc4 ansi:\\x1b[31mred\\x1b[0m end\\n'\n"
    "sys.stdout.buffer.write(data)\n"
    "sys.stdout.buffer.flush()\n"
)
res, _ = run([PY, "-c", payload])
raw = (TMP / "out.log").read_bytes()
check(
    "raw bytes preserved in log",
    b"\xd6\xd0\xce\xc4" in raw and b"\xe4\xb8\xad" in raw and b"\x1b[31m" in raw,
    f"bytes={len(raw)}",
)
plain = (TMP / "out.plain.log").read_bytes() if (TMP / "out.plain.log").exists() else b""
check("ANSI stripped only in plain log", b"\x1b[31m" not in plain and b"red" in plain, repr(plain[:40]))
check("strip_ansi roundtrip", strip_ansi(b"\x1b[31mred\x1b[0m") == b"red")


# --- 5. interrupt path -------------------------------------------------------
def has_console() -> bool:
    if os.name != "nt":
        return sys.stderr.isatty()
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return False


if os.name == "nt" and has_console():
    import signal

    runner = ProcessRunner(
        [PY, "-c", "import time; time.sleep(20)"],
        echo=False,
        stdout_path=TMP / "out.log",
        stderr_path=TMP / "err.log",
    )
    box: list[object] = []

    def _send_ctrl_c() -> None:
        # wait until pid exists, then raise CTRL_C_EVENT in *this* console group
        for _ in range(200):
            if runner._proc is not None:
                break
            time.sleep(0.02)
        time.sleep(0.8)
        try:
            os.kill(runner._proc.pid, signal.CTRL_C_EVENT)  # type: ignore[attr-defined]
        except Exception as exc:
            box.append(f"send failed: {exc}")

    sender = threading.Thread(target=_send_ctrl_c, daemon=True)
    sender.start()
    res = runner.run()
    check(
        "Ctrl+C delivered to child is recorded",
        res.exit_code is not None and (res.interrupted or res.exit_code_raw not in (0, None)),
        f"interrupted={res.interrupted} raw={res.exit_code_raw} normalized={res.exit_code} {box}",
    )
    print(f"      console code page: {os.environ.get('PYTHONIOENCODING')} / child got CTRL_C_EVENT")
else:
    print("SKIP  Ctrl+C case: no interactive console attached (run on a real terminal)")

# --- 6. full CLI: does an interrupted run still FINALIZE? --------------------
def cli_interrupt_case() -> None:
    """Ctrl+Break a real `labbox run` and check the post-process stages still complete.

    The case above proves the child's status is captured. This one covers what the user
    actually gets: an interrupted run whose metrics, artifacts, diagnostics, reproduce
    scripts and report were all still written, with status=interrupted.

    A real keypress cannot be synthesised, so this is its closest equivalent: CTRL_C_EVENT
    is sent to the whole console - LabBox, its training child, and this spike. The spike
    installs a SIGINT handler that ignores it, which is the only difference from a user
    hitting Ctrl+C. Note that CREATE_NEW_PROCESS_GROUP must NOT be used here: that flag
    disables Ctrl+C for the new group, so the event would be delivered to nobody.
    """
    import json
    import subprocess
    import time

    project = TMP / "cli"
    shutil.rmtree(project, ignore_errors=True)
    (project / ".labbox").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "examples" / "train_fake.py", project / "train.py")
    dev_shim = REPO / "scripts" / "labbox_dev.py"
    console = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
    cli = [str(console)]
    if console.is_file():
        probe = subprocess.run([str(console), "init", "--help"], capture_output=True, shell=False, check=False)
        if probe.returncode != 0:
            cli = [PY, str(dev_shim)]
    else:
        cli = [PY, str(dev_shim)]

    subprocess.run([*cli, "init"], cwd=str(project), capture_output=True, shell=False, check=False)
    argv = [
        *cli,
        "run",
        "--name",
        "ctrlc",
        "--sample-interval",
        "0.2",
        "--",
        PY,
        "train.py",
        "--epochs",
        "60",
        "--sleep",
        "0.5",
        "--out",
        "outputs",
    ]
    proc = subprocess.Popen(
        argv,
        cwd=str(project),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    runs = project / ".labbox" / "runs"
    target = None
    for _ in range(300):
        time.sleep(0.1)
        found = sorted(runs.glob("*ctrlc")) if runs.is_dir() else []
        started = found and (found[0] / "logs" / "stdout.log").is_file()
        if (
            started
            and (found[0] / "logs" / "stdout.log").stat().st_size > 0
            and (found[0] / "environment.json").is_file()
        ):
            target = found[0]
            break
    if target is None:
        proc.kill()
        proc.wait(timeout=20)
        check("interrupted CLI run finalizes", False, "the run never reached the training phase")
        return
    previous = signal.signal(signal.SIGINT, lambda *_: None)  # survive our own broadcast
    try:
        os.kill(0, signal.CTRL_C_EVENT)  # pid 0 = every process sharing this console
        try:
            stdout_data, stderr_data = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
            check("interrupted CLI run finalizes", False, "labbox did not exit after the interrupt")
            return
        except KeyboardInterrupt:
            stdout_data, stderr_data = proc.communicate(timeout=30)
    finally:
        signal.signal(signal.SIGINT, previous)
    record = json.loads((target / "run.json").read_text(encoding="utf-8"))
    missing = [
        name
        for name in (
            "metrics.json",
            "artifacts.json",
            "dataset-manifest.json",
            "diagnostics.json",
            "config-values.json",
            "report.html",
            "reproduce.ps1",
            "reproduce.sh",
            "samples/system.jsonl",
        )
        if not (target / name).exists()
    ]
    check(
        "interrupted CLI run finalizes",
        record.get("status") == "interrupted" and record.get("interrupted") is True and not missing,
        f"status={record.get('status')} exit={record.get('exit_code')} "
        f"raw={record.get('exit_code_raw')} missing={missing}",
    )
    check(
        "interrupted CLI run returns the Ctrl+C code",
        record.get("exit_code") == 130 or proc.returncode == 130,
        f"labbox exit={proc.returncode} recorded={record.get('exit_code')} (raw {record.get('exit_code_raw')})",
    )
    note = (stderr_data or b"").decode("utf-8", "replace")
    print(f"      labbox stderr tail: {note.strip().splitlines()[-1] if note.strip() else '(empty)'}")
    del stdout_data


if os.name == "nt" and has_console():
    cli_interrupt_case()
else:
    print("SKIP  CLI interrupt case: needs a console to deliver the ctrl+c event")


failed = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} spike checks passed")
sys.exit(1 if failed else 0)
