"""Exit-code normalization and ANSI handling — the pure parts of ProcessRunner.

The values here come from measurements on Windows 11 (see scripts/spike_process.py), not
from folklore: ``taskkill /F`` reports exit code 1, and a Ctrl+C-terminated console app
reports 0xC000013A, which is the same observable event a POSIX shell shows as 130.
"""

from __future__ import annotations

import pytest

from labbox.domain.run import ExitMeaning
from labbox.recorder.process import ByteStreamWriter, normalize_exit_code, strip_ansi


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, (0, ExitMeaning.EXITED)),
        (1, (1, ExitMeaning.EXITED)),
        (7, (7, ExitMeaning.EXITED)),
        (255, (255, ExitMeaning.EXITED)),
        (-11, (139, ExitMeaning.SIGNALED)),  # POSIX SIGSEGV -> 128+11
        (-9, (137, ExitMeaning.SIGNALED)),  # SIGKILL -> 128+9
        (None, (None, ExitMeaning.UNKNOWN)),
    ],
)
def test_posix_semantics(raw: int | None, expected: tuple[int | None, ExitMeaning]) -> None:
    assert normalize_exit_code(raw, windows=False) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, (0, ExitMeaning.EXITED)),
        (7, (7, ExitMeaning.EXITED)),
        (-1073741510, (130, ExitMeaning.SIGNALED)),  # 0xC000013A STATUS_CONTROL_C_EXIT
        (3221225786, (130, ExitMeaning.SIGNALED)),  # same value unsigned
        (-1073741819, (139, ExitMeaning.SIGNALED)),  # 0xC0000005 access violation
        (-1073740791, (134, ExitMeaning.SIGNALED)),  # 0xC0000409 fast fail
        (-1073741502, (-1073741502, ExitMeaning.KILLED)),  # unmapped NTSTATUS
        (1, (1, ExitMeaning.EXITED)),  # taskkill /F shows up as a plain exit(1)
    ],
)
def test_windows_semantics(raw: int | None, expected: tuple[int | None, ExitMeaning]) -> None:
    assert normalize_exit_code(raw, windows=True) == expected


def test_control_c_looks_like_posix_130_after_normalization() -> None:
    code, _ = normalize_exit_code(-1073741510, windows=True)
    assert code == 130


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x1b[31mred\x1b[0m", b"red"),
        (b"plain", b"plain"),
        (b"\x1b[1;32mok\x1b[0m tail", b"ok tail"),
        (b"\x1b]0;window title\x07body", b"body"),
        (b"\x1b[2K\rprogress", b"\rprogress"),
        (b"\x1b[38;2;255;0;0mrgb\x1b[m", b"rgb"),
        (b"no escapes \xc3\xa9", b"no escapes \xc3\xa9"),
    ],
)
def test_strip_ansi(data: bytes, expected: bytes) -> None:
    assert strip_ansi(data) == expected


def test_strip_ansi_keeps_bytes_it_cannot_parse() -> None:
    assert strip_ansi(b"\x1b[") == b"\x1b["


# ---------------------------------------------------------------------------
# the plain log is written incrementally, so escapes arrive split
# ---------------------------------------------------------------------------
class _FakePipe:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def read(self, size: int = 8192) -> bytes:  # size is part of the pipe contract
        return self.chunks.pop(0) if self.chunks else b""


def run_writer(tmp_path, chunks: list[bytes]) -> tuple[bytes, ByteStreamWriter]:
    plain = tmp_path / "stdout.plain.log"
    writer = ByteStreamWriter(_FakePipe(chunks), None, plain_path=plain)
    writer.run()
    return plain.read_bytes(), writer


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # an escape split exactly between the introducer and its parameters
        ([b"a\x1b", b"[31mred\x1b[0mb"], b"aredb"),
        # split inside the parameter bytes
        ([b"a\x1b[3", b"1mred\x1b[0mb"], b"aredb"),
        # consecutive escapes split across chunks
        ([b"\x1b[2", b"K\r\x1b", b"[31mred\x1b[0m"], b"\rred"),
        # an OSC title split around its BEL
        ([b"\x1b]0;tit", b"le\x07body"], b"body"),
        # a bare lone ESC at the very end is dropped, not written as a fragment
        ([b"done\x1b"], b"done"),
        # ordinary text with no escapes passes through untouched
        ([b"plain ", b"text\n"], b"plain text\n"),
    ],
    ids=["after-intro", "in-params", "consecutive", "osc", "trailing-esc", "no-escape"],
)
def test_plain_log_is_ansi_stripped_across_chunk_boundaries(
    tmp_path, chunks: list[bytes], expected: bytes
) -> None:
    data, writer = run_writer(tmp_path, chunks)
    assert data == expected
    assert writer.ansi_stripped is True


def test_a_truncated_escape_does_not_swallow_the_following_text(tmp_path) -> None:
    """An unterminated sequence must not eat real output that came after it."""
    data, _writer = run_writer(tmp_path, [b"keep\x1b\x1b[31mred\x1b[0m"])
    assert b"red" in data and data.startswith(b"keep")


def test_plain_log_is_not_buffered_in_memory(tmp_path) -> None:
    """A 1 MiB stream must never be held whole: the writer flushes as it goes."""
    from labbox.recorder.process import ByteStreamWriter

    plain = tmp_path / "plain.log"
    payload = b"x" * (1024 * 1024)
    writer = ByteStreamWriter(_FakePipe([payload[:512 * 1024], payload[512 * 1024 :]]), None, plain_path=plain)
    writer.run()
    assert plain.read_bytes() == payload
    assert writer.bytes_written == len(payload)
