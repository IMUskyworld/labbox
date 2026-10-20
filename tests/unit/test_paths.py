"""Run-id/path handling: non-ASCII names, Windows reserved names, traversal."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from labbox.errors import PathEscapeError, RunNotFoundError
from labbox.paths import (
    ProjectLayout,
    is_run_id,
    make_run_id,
    slugify,
    strip_timestamp,
)

WHEN = datetime(2026, 9, 22, 15, 30, 12)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("baseline", "baseline"),
        ("yolo bs128", "yolo-bs128"),
        ("../../etc/passwd", "etc-passwd"),
        ("a/b/c", "a-b-c"),
        ("中文实验-1", "中文实验-1"),
        ("  spaced  ", "spaced"),
        ("!!!", "run"),
        ("CON", "_CON"),
        ("nul.txt", "nul.txt"),
        ("x" * 200, "x" * 48),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_run_id_shape_and_roundtrip() -> None:
    run_id = make_run_id("中文 实验", WHEN)
    assert run_id == "20260922-153012-中文-实验"
    assert is_run_id(run_id)
    assert strip_timestamp(run_id) == "中文-实验"


def test_layout_resolves_by_name_id_and_unique_suffix(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    for name in ("20260922-153012-baseline", "20260923-090000-run-b"):
        run = layout.runs_dir / name
        run.mkdir(parents=True)
        (run / "run.json").write_text("{}", encoding="utf-8")
    assert layout.resolve_run_id("baseline").name == "20260922-153012-baseline"
    assert layout.resolve_run_id("run-b").name == "20260923-090000-run-b"
    assert layout.resolve_run_id("20260922-153012-baseline").is_dir()
    with pytest.raises(RunNotFoundError):
        layout.resolve_run_id("nope")


def test_ambiguous_name_is_refused(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    for name in ("20260922-153012-exp", "20260923-153012-exp"):
        run = layout.runs_dir / name
        run.mkdir(parents=True)
        (run / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RunNotFoundError) as excinfo:
        layout.resolve_run_id("exp")
    assert "matches 2 runs" in str(excinfo.value)


@pytest.mark.parametrize("evil", ["../outside", "../../x", "..", "a/../../b"])
def test_path_traversal_is_rejected(tmp_path: Path, evil: str) -> None:
    layout = ProjectLayout(tmp_path)
    (tmp_path / "outside").mkdir()
    with pytest.raises((RunNotFoundError, PathEscapeError)):
        layout.resolve_run_id(evil)


def test_run_paths_are_relative_inside_dir(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    paths = layout.run_dir("20260922-153012-a")
    assert paths.dir == layout.runs_dir / "20260922-153012-a"
    assert paths.events_log.name == "events.jsonl"
    assert paths.system_samples.parent.name == "samples"
    assert paths.resolve("logs/stdout.log") == paths.dir / "logs" / "stdout.log"
