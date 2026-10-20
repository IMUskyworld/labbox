"""Start the training command, forward its output verbatim, keep its exit code (§9.1).

Everything else in LabBox is optional; this module is not, and it is the reason the two
hardest rules exist: *never break training* and *preserve the exit code*. So:

* no shell. ``argv`` is a list, passed straight to ``CreateProcess``/``execvp``; a
  parameter containing ``&&`` or quotes stays a parameter;
* a reader thread per stream. Reading only stdout while the child floods stderr is a
  classic 64 KB-pipe deadlock;
* logs are written as **raw bytes**. Decoding happens for display only, with
  ``errors="replace"``, so a GBK or mixed-encoding line can never abort the run;
* a failure inside our own logging is recorded and swallowed. Losing a log line is
  acceptable; killing a three-day training run is not.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from labbox.domain.common import now_local
from labbox.domain.run import ExitMeaning

# NTSTATUS values Windows reports where POSIX would report a signal. The mapping keeps
# `labbox run -- python -c "..."` comparable with a shell's `$?`.
_NTSTATUS_TO_POSIX: dict[int, int] = {
    0xC000013A: 130,  # STATUS_CONTROL_C_EXIT
    0x40010004: 130,  # DBG_CONTROL_C
    0xC0000005: 139,  # access violation ~ SIGSEGV
    0xC00000FD: 134,  # stack overflow ~ SIGABRT-ish
    0xC0000409: 134,  # STATUS_STACK_BUFFER_OVERRUN / fast fail
    0xC000001D: 132,  # illegal instruction
    0x80000003: 132,  # breakpoint
}
_WINDOWS_SIGNAL_LIKE_MIN = 0xC0000000

_ANSI_RE = re.compile(
    rb"""(?:
        \x1b \[ [0-9;?]* [ -/]* [@-~]     # CSI: intermediates are optional (\x1b[31m has none)
      | \x1b \] [^\x07\x1b]* (?: \x07 | \x1b\\ )  # OSC
      | \x1b [PX^_] [^\x1b]* (?: \x1b\\ )?  # DCS/PM/APC/SOS
      | \x1b [=>()#][0-9A-Za-z]?           # simple escapes
      | \x1b [78hcEM]                       # single char
    )""",
    re.VERBOSE,
)


def strip_ansi(data: bytes) -> bytes:
    return _ANSI_RE.sub(b"", data)


class _AnsiStreamFilter:
    """Strip ANSI escapes from a byte stream, one chunk at a time.

    ``strip_ansi`` is a whole-buffer function, which is only correct when the buffer holds
    complete sequences: a pipe hands over 8 KB at arbitrary boundaries, so an escape can be
    split down the middle and a naive per-chunk strip leaves a fragment of a CSI code in
    the plain log. Anything still incomplete is carried into the next chunk, and a
    trailing incomplete sequence at end of stream is dropped rather than written.
    """

    #: A CSI is bounded (parameters are 0x30-0x3F, intermediates 0x20-0x2F, one final).
    _MAX_CSI = 64
    #: OSC/DCS-family sequences carry arbitrary text and are only ended by BEL or ST.
    _MAX_STRING = 64 * 1024

    def __init__(self, target: object) -> None:
        self._target = target
        self._pending = bytearray()

    def feed(self, chunk: bytes) -> None:
        if self._pending:
            self._pending.extend(chunk)
            data = bytes(self._pending)
            self._pending.clear()
        else:
            data = chunk
        start = data.find(b"\x1b")
        if start < 0:
            self._write(data)
            return
        if start:
            self._write(data[:start])
        rest = data[start:]
        # Re-scan repeatedly: escapes come in runs ("\x1b[2K\r\x1b[31m"), and the text
        # between two escapes is complete and belongs in the output.
        while rest:
            match = _ANSI_RE.match(rest)
            if match is not None:
                rest = rest[match.end() :]
                continue
            following = rest.find(b"\x1b", 1)
            if following > 0:
                head, rest = rest[:following], rest[following:]
                self._write(head)
                continue
            if not _is_escape_prefix(rest):
                self._write(rest)
                return
            self._pending.extend(rest)
            return

    def close(self) -> None:
        # A truncated escape at end of stream is not real content; drop it.
        self._pending.clear()

    def _write(self, data: bytes) -> None:
        if data:
            write = getattr(self._target, "write", None)
            if write is not None:
                write(data)


#: Bytes that can introduce an escape sequence, matching the branches of ``_ANSI_RE``.
_CSI_INTRODUCERS = frozenset(b"[")
_STRING_INTRODUCERS = frozenset(b"]PX^_")
_SHORT_INTRODUCERS = frozenset(b"=><()#78hcEM")


def _is_escape_prefix(prefix: bytes) -> bool:
    """Whether *prefix* may be an escape sequence that is simply not finished yet.

    ``prefix`` starts with ESC. The second byte decides which family it is; each family
    has its own "could this still become valid?" rule. Anything that cannot become valid
    is real data and is written out rather than buffered forever.
    """
    if not prefix or prefix[0] != 0x1B:
        return False
    if len(prefix) == 1:
        return True
    second = prefix[1]
    if second == 0x1B:
        return False
    if second in _CSI_INTRODUCERS:
        return len(prefix) < _AnsiStreamFilter._MAX_CSI and _is_csi_prefix(prefix)
    if second in _STRING_INTRODUCERS:
        return len(prefix) < _AnsiStreamFilter._MAX_STRING
    if second in _SHORT_INTRODUCERS:
        return len(prefix) < 8
    return False


def _is_csi_prefix(prefix: bytes) -> bool:
    """CSI bytes are parameters (0x30-0x3F), intermediates (0x20-0x2F) and one final."""
    return all(0x20 <= byte <= 0x3F for byte in prefix[2:])


def normalize_exit_code(raw: int | None, *, windows: bool | None = None) -> tuple[int | None, ExitMeaning]:
    """Map an OS exit status onto (posix-style code, meaning)."""
    if raw is None:
        return None, ExitMeaning.UNKNOWN
    on_windows = (os.name == "nt") if windows is None else windows
    if on_windows:
        unsigned = raw & 0xFFFFFFFF if raw < 0 else raw
        if unsigned in _NTSTATUS_TO_POSIX:
            return _NTSTATUS_TO_POSIX[unsigned], ExitMeaning.SIGNALED
        if unsigned >= _WINDOWS_SIGNAL_LIKE_MIN and raw < 0:
            return raw, ExitMeaning.KILLED
        return raw, ExitMeaning.EXITED
    if raw < 0:
        return 128 + (-raw), ExitMeaning.SIGNALED
    return raw, ExitMeaning.EXITED


@dataclass(slots=True)
class ProcessResult:
    exit_code: int | None = None
    exit_code_raw: int | None = None
    exit_meaning: ExitMeaning = ExitMeaning.UNKNOWN
    interrupted: bool = False
    forced_kill: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    launch_error: str | None = None
    launch_error_code: str | None = None
    log_errors: list[str] = field(default_factory=list)
    reader_timeouts: list[str] = field(default_factory=list)
    ansi_stripped: bool = False
    child_pid: int | None = None
    killed_by_labbox: bool = False

    @property
    def started(self) -> bool:
        return self.launch_error is None

    @property
    def succeeded(self) -> bool:
        return self.launch_error is None and self.exit_code == 0


class ByteStreamWriter:
    """One output stream: append to disk, optionally echo to the terminal."""

    def __init__(
        self,
        pipe,
        log_path: Path | None,
        *,
        echo_to: object | None = None,
        plain_path: Path | None = None,
    ) -> None:
        self.pipe = pipe
        self.log_path = log_path
        self.plain_path = plain_path
        self.echo_to = echo_to
        self.bytes_written = 0
        self.ansi_stripped = False
        self.errors: list[str] = []
        self.cancel = threading.Event()

    def run(self) -> None:
        plain_fh = None
        if self.plain_path is not None:
            try:
                self.plain_path.parent.mkdir(parents=True, exist_ok=True)
                plain_fh = self.plain_path.open("wb")
            except OSError as exc:
                self.errors.append(f"plain log open failed: {exc}")
                plain_fh = None
        # The plain log is written incrementally rather than buffered to the end. A
        # three-day training run produces tens of gigabytes of stdout, and holding all of
        # it in memory to strip ANSI at the end is a memory bomb that only ever fires on
        # exactly the runs the user most needs recorded.
        filterer = _AnsiStreamFilter(plain_fh) if plain_fh is not None else None
        try:
            log_fh = self.log_path.open("ab") if self.log_path else None
        except OSError as exc:
            self.errors.append(f"open failed: {exc}")
            log_fh = None
        try:
            while True:
                try:
                    chunk = self.pipe.read(8192)
                except (OSError, ValueError) as exc:
                    self.errors.append(f"read failed: {exc}")
                    break
                if not chunk:
                    break
                self.bytes_written += len(chunk)
                if log_fh is not None:
                    try:
                        log_fh.write(chunk)
                        log_fh.flush()
                    except OSError as exc:
                        # A full disk or a locked file must not stop the training process.
                        self.errors.append(f"write failed: {exc}")
                        log_fh = None
                if self.echo_to is not None:
                    _echo(self.echo_to, chunk)
                if filterer is not None:
                    try:
                        filterer.feed(chunk)
                    except OSError as exc:
                        self.errors.append(f"plain log failed: {exc}")
                        filterer = None
                if self.cancel.is_set():
                    break
        finally:
            if filterer is not None:
                try:
                    filterer.close()
                    self.ansi_stripped = True
                except OSError as exc:
                    self.errors.append(f"plain log failed: {exc}")
            if plain_fh is not None:
                try:
                    plain_fh.close()
                except OSError as exc:
                    self.errors.append(f"plain log close failed: {exc}")
            if log_fh is not None:
                try:
                    log_fh.close()
                except OSError:
                    pass


class ProcessRunner:
    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        plain_stdout_path: Path | None = None,
        echo: bool = True,
        echo_stderr: bool = True,
        reader_join_timeout: float = 5.0,
        on_output: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self.argv = [str(a) for a in argv]
        if not self.argv:
            raise ValueError("ProcessRunner requires a non-empty argv")
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.plain_stdout_path = plain_stdout_path
        self.echo = echo
        self.echo_stderr = echo_stderr
        self.reader_join_timeout = reader_join_timeout
        self.on_output = on_output
        self.result = ProcessResult()
        self._proc: subprocess.Popen[bytes] | None = None
        self._writers: list[ByteStreamWriter] = []
        self._threads: list[threading.Thread] = []
        self._interrupts = 0

    # --- public API ----------------------------------------------------------
    def run(self) -> ProcessResult:
        result = self.result
        result.started_at = now_local()
        try:
            self._proc = self._popen()
        except FileNotFoundError as exc:
            result.launch_error = f"command not found: {self.argv[0]}"
            result.launch_error_code = "COMMAND_NOT_FOUND"
            result.exit_code, result.exit_code_raw = 127, None
            result.exit_meaning = ExitMeaning.NOT_STARTED
            result.ended_at = now_local()
            del exc
            return result
        except (OSError, ValueError) as exc:
            result.launch_error = f"failed to start {self.argv[0]}: {exc}"
            result.launch_error_code = "COMMAND_FAILED_TO_START"
            result.exit_code, result.exit_code_raw = 126, None
            result.exit_meaning = ExitMeaning.NOT_STARTED
            result.ended_at = now_local()
            return result

        result.child_pid = self._proc.pid
        self._start_readers()
        self._wait_forever()
        self._join_readers(result)
        raw = self._proc.returncode
        result.exit_code_raw = raw
        result.exit_code, result.exit_meaning = normalize_exit_code(raw)
        if result.interrupted and result.exit_code in (None, 0):
            # A child that ignores Ctrl+C and exits 0 is still an interrupted run.
            result.exit_meaning = ExitMeaning.SIGNALED
        result.ended_at = now_local()
        if result.started_at is not None:
            result.duration_seconds = round((result.ended_at - result.started_at).total_seconds(), 3)
        for writer in self._writers:
            result.log_errors.extend(writer.errors)
            if getattr(writer, "ansi_stripped", False):
                result.ansi_stripped = True
        return result

    def kill(self) -> None:
        """Second-interrupt escalation path: terminate, then kill."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            self.result.forced_kill = True
            self.result.killed_by_labbox = True
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.result.log_errors.append("escalated termination did not confirm")

    # --- internals -----------------------------------------------------------
    def _popen(self) -> subprocess.Popen[bytes]:
        env = self.env
        if env is not None:
            # Without unbuffering, a piped child buffers 8 KB and the log lags the run,
            # which is exactly the "behaves like the bare command" property we promise.
            env.setdefault("PYTHONUNBUFFERED", "1")
        popen_kwargs: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": None,  # inherited: interactive training scripts keep working
            "cwd": self.cwd,
            "env": env,
            "shell": False,
            "bufsize": 0,
        }
        if os.name == "nt":
            # Deliberately no creation flags: the child must stay attached to this console
            # so it receives the Ctrl+C event itself, and we never forward a second one
            # (DEVELOPMENT.md §6.3).
            pass
        else:
            popen_kwargs["start_new_session"] = False
        return subprocess.Popen(self.argv, **popen_kwargs)  # type: ignore[arg-type]

    def _start_readers(self) -> None:
        proc = self._proc
        assert proc is not None
        specs: list[tuple[object | None, Path | None, object | None, Path | None]] = [
            (proc.stdout, self.stdout_path, sys.stdout.buffer if self.echo else None, self.plain_stdout_path),
            (proc.stderr, self.stderr_path, sys.stderr.buffer if self.echo_stderr else None, None),
        ]
        for pipe, log_path, echo_to, plain in specs:
            if pipe is None:
                continue
            writer = ByteStreamWriter(pipe, log_path, echo_to=echo_to, plain_path=plain)
            self._writers.append(writer)
            thread = threading.Thread(
                target=writer.run, name=f"labbox-reader-{log_path.name if log_path else 'mem'}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _wait_forever(self) -> None:
        proc = self._proc
        assert proc is not None
        while True:
            try:
                proc.wait(timeout=0.2)
                return
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                self._handle_interrupt()

    def _handle_interrupt(self) -> None:
        self._interrupts += 1
        self.result.interrupted = True
        if self._interrupts == 1:
            # The child already got the console event; wait for it to finish cleanup.
            self._echo_note("^C received, waiting for the training process to finish...")
            return
        self._echo_note("^C received again, terminating the child process.")
        self.kill()

    def _echo_note(self, text: str) -> None:
        try:
            sys.stderr.write("\n" + text + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _join_readers(self, result: ProcessResult) -> None:
        deadline = time.monotonic() + self.reader_join_timeout
        for writer, thread in zip(self._writers, self._threads, strict=True):
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                # A grandchild can inherit the pipe and keep it open; we must not hang.
                writer.cancel.set()
                name = "stdout" if writer.log_path == self.stdout_path else "stderr"
                result.reader_timeouts.append(
                    f"{name} reader still blocked after {self.reader_join_timeout:.1f}s; "
                    "a child process may still be holding the pipe"
                )
            if writer.log_path == self.stdout_path:
                result.stdout_bytes = writer.bytes_written
            elif writer.log_path == self.stderr_path:
                result.stderr_bytes = writer.bytes_written


def _echo(target: object, data: bytes) -> None:
    """Mirror the child's bytes onto the real terminal without re-encoding them."""
    write = getattr(target, "write", None)
    if write is None:
        return
    try:
        write(data)
        flush = getattr(target, "flush", None)
        if flush is not None:
            flush()
    except Exception:
        # The child's own write succeeded; only our mirror failed.
        pass
