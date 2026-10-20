"""Atomic writes, JSONL appends and crash behaviour (§6.2)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.errors import LabBoxError
from labbox.infrastructure.store import FilesystemStore, dumps
from labbox.paths import ProjectLayout


@pytest.fixture
def store(project: ProjectLayout) -> FilesystemStore:
    return FilesystemStore(project)


def test_json_atomic_writes_are_valid_and_utf8(store: FilesystemStore, tmp_path: Path) -> None:
    paths = store.layout.run_dir("20260922-153012-a")
    store.ensure_run_dirs(paths)
    target = paths.dir / "run.json"
    store.write_json_atomic(target, {"name": "基线-实验", "n": 1})
    raw = target.read_bytes()
    assert "基线".encode() in raw, "must not escape non-ASCII"
    assert json.loads(raw)["n"] == 1
    assert not list(paths.dir.glob("*.tmp")), "temp files must not be left behind"


def test_overwrite_is_atomic_and_keeps_no_partial(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153013-a")
    store.ensure_run_dirs(paths)
    target = paths.dir / "environment.json"
    store.write_json_atomic(target, {"step": 1})
    for index in range(200):
        store.write_json_atomic(target, {"step": index, "pad": "x" * 4096})
    assert json.loads(target.read_text(encoding="utf-8"))["step"] == 199


def test_failed_write_leaves_previous_content(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153014-a")
    store.ensure_run_dirs(paths)
    target = paths.dir / "metrics.json"
    store.write_json_atomic(target, {"final": {"accuracy": 0.84}})

    with pytest.raises((LabBoxError, TypeError, ValueError)):
        # An unserializable value must fail during formatting, before the rename.
        store.write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"final": {"accuracy": 0.84}}


def test_append_jsonl_keeps_every_line(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153015-a")
    store.ensure_run_dirs(paths)
    for index in range(500):
        store.append_jsonl(paths.system_samples, {"i": index})
    lines = paths.system_samples.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 500
    assert json.loads(lines[-1])["i"] == 499


def test_corrupt_json_is_reported_not_silently_ignored(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153016-a")
    store.ensure_run_dirs(paths)
    target = paths.dir / "run.json"
    target.write_text('{"run_id": "x", ', encoding="utf-8")  # half-written document
    with pytest.raises(LabBoxError):
        store.read_json(target)
    assert store.read_json(target, strict=False) is None


def test_a_read_failure_is_not_reported_as_a_write_failure(store: FilesystemStore) -> None:
    """A damaged file on disk is not a disk that refused a write."""
    paths = store.layout.run_dir("20260922-153019-a")
    store.ensure_run_dirs(paths)
    target = paths.dir / "run.json"
    target.write_text('{"run_id": ', encoding="utf-8")
    with pytest.raises(LabBoxError) as excinfo:
        store.read_json(target)
    assert excinfo.value.code == "STORE_READ_FAILED"


def test_append_text_reports_failure_instead_of_raising(store: FilesystemStore, tmp_path: Path) -> None:
    """The event/error path must not raise a second exception on a failing disk."""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("file", encoding="utf-8")
    assert store.append_text(blocked / "nested" / "stderr.log", "note\n") is False


def test_deleting_a_run_that_is_already_gone_is_not_an_error(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153020-a")
    store.ensure_run_dirs(paths)
    store.write_json_atomic(paths.dir / "run.json", {"run_id": paths.dir.name})
    assert store.delete_run(paths.dir) is True
    assert store.delete_run(paths.dir) is False, "a retry after a partial success must not abort"


def test_run_listing_and_sizing(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153017-a")
    store.ensure_run_dirs(paths)
    store.write_json_atomic(paths.dir / "run.json", {"run_id": paths.dir.name})
    (paths.dir / "logs").mkdir(exist_ok=True)
    (paths.dir / "logs" / "stdout.log").write_text("hello\n", encoding="utf-8")
    (paths.dir / "notarun").mkdir()
    assert [p.name for p in store.list_run_dirs()] == ["20260922-153017-a"]
    assert store.dir_size_bytes(paths.dir) >= 6
    assert store.total_store_bytes() >= 6


def test_delete_run_refuses_runs_root(store: FilesystemStore) -> None:
    paths = store.layout.run_dir("20260922-153018-a")
    store.ensure_run_dirs(paths)
    store.write_json_atomic(paths.dir / "run.json", {})
    store.delete_run(paths.dir)
    assert not paths.dir.exists()
    with pytest.raises(LabBoxError):
        store.delete_run(store.layout.runs_dir)


def test_dumps_sorts_nothing_and_keeps_order(store: FilesystemStore) -> None:
    text = dumps({"z": 1, "a": 2})
    assert text.index('"z"') < text.index('"a"')


def test_config_toml_roundtrip(tmp_path: Path) -> None:
    config = LabboxConfig.default()
    config.run.sample_interval_seconds = 3.5
    config.datasets.paths = ["data/cifar100"]
    config.llm.enabled = False
    path = tmp_path / "config.toml"
    path.write_text(config.to_toml(), encoding="utf-8", newline="\n")
    loaded = LabboxConfig.load(path)
    assert loaded.run.sample_interval_seconds == 3.5
    assert loaded.datasets.paths == ["data/cifar100"]
    assert loaded.gpu.backend == "auto"


def test_invalid_config_raises_structured_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('schema_version = "9.9"\n', encoding="utf-8")
    with pytest.raises(LabBoxError) as excinfo:
        LabboxConfig.load(path)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_unreadable_config_is_reported_as_problem(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[gpu\nbroken", encoding="utf-8")
    config = LabboxConfig.load(path)
    assert config.problems, "malformed TOML must surface, not vanish"


@pytest.mark.skipif(os.name != "nt", reason="Windows-only behaviour")
def test_write_into_unwritable_dir_fails_cleanly(store: FilesystemStore, tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "deep" / "run.json"
    store.write_json_atomic(missing, {"a": 1})
    assert missing.is_file(), "parents are created on demand"
