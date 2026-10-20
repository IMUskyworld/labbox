"""Artifact discovery and the large-file hash circuit breaker (§9.5)."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.artifacts import ArtifactKind, ArtifactManifest, HashState
from labbox.recorder.artifacts import ArtifactCollector, copy_artifacts, outside


@pytest.fixture
def config() -> LabboxConfig:
    return LabboxConfig.default()


def collect(root: Path, config: LabboxConfig, **overrides) -> ArtifactManifest:
    for key, value in overrides.items():
        setattr(config.artifacts, key, value)
    return ArtifactCollector(project_root=root, config=config).collect()


def test_checkpoints_are_found_by_suffix_and_hashed(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    payload = b"weights" * 100
    (tmp_path / "checkpoints" / "best.pt").write_bytes(payload)
    manifest = collect(tmp_path, config)
    entry = manifest.artifacts[0]
    assert entry.kind is ArtifactKind.CHECKPOINT
    assert entry.size_bytes == len(payload)
    assert entry.sha256 == hashlib.sha256(payload).hexdigest()
    assert entry.hash_state is HashState.COMPUTED
    assert entry.mtime is not None
    assert not entry.path.startswith(str(tmp_path)[0:2]), "paths are stored project-relative"


def test_huge_artifacts_defer_their_hash(tmp_path: Path, config: LabboxConfig) -> None:
    big = tmp_path / "checkpoints"
    big.mkdir()
    (big / "model.safetensors").write_bytes(b"\0" * 4096)
    manifest = collect(tmp_path, config, hash_max_bytes=1024)
    entry = manifest.artifacts[0]
    assert entry.hash_state is HashState.DEFERRED_TOO_LARGE
    assert entry.sha256 is None
    assert entry.size_bytes == 4096, "metadata is still recorded"
    assert "hash_max_bytes" in (entry.error or "")


def test_globs_from_config_are_used(tmp_path: Path, config: LabboxConfig) -> None:
    out = tmp_path / "outputs" / "run1"
    out.mkdir(parents=True)
    (out / "metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    config.artifacts.patterns = ["outputs/**/*.csv"]
    manifest = collect(tmp_path, config)
    assert [entry.path for entry in manifest.artifacts] == ["outputs/run1/metrics.csv".replace("\\", "/")]
    assert manifest.artifacts[0].discovered_by == "config_glob"


def test_nothing_is_copied_by_default(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "last.ckpt").write_bytes(b"x" * 10)
    manifest = collect(tmp_path, config)
    assert manifest.artifacts[0].archived is False
    assert manifest.copied == []
    assert not (tmp_path / ".labbox").exists()


def test_copy_artifacts_respects_the_size_limit(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "a.pt").write_bytes(b"1" * 100)
    (tmp_path / "checkpoints" / "b.pt").write_bytes(b"2" * 5000)
    manifest = collect(tmp_path, config)
    store_dir = tmp_path / "run"
    copied = copy_artifacts(manifest, store_dir, limit_bytes=1000)
    assert copied == ["checkpoints/a.pt".replace("\\", "/")]
    assert manifest.artifacts[0].archived is True
    assert (store_dir / "checkpoints" / "a.pt").is_file()
    assert any("size limit" in warning for warning in manifest.warnings)


def test_same_basename_artifacts_do_not_overwrite_each_other(tmp_path: Path, config: LabboxConfig) -> None:
    """Flattening to the basename silently dropped every ``best.pt`` but the last."""
    for name in ("a", "b"):
        directory = tmp_path / "outputs" / name
        directory.mkdir(parents=True)
        (directory / "best.pt").write_bytes(name.encode() * 32)
    config.artifacts.patterns = ["outputs/**/best.pt"]
    manifest = collect(tmp_path, config)
    store_dir = tmp_path / "run"
    copied = copy_artifacts(manifest, store_dir, limit_bytes=10_000)
    assert len(copied) == 2
    for name in ("a", "b"):
        target = store_dir / "checkpoints" / "outputs" / name / "best.pt"
        assert target.is_file(), f"{name} must not be overwritten by the other one"
        assert target.read_bytes() == name.encode() * 32


def test_copy_refuses_to_escape_the_run_directory(tmp_path: Path, config: LabboxConfig) -> None:
    from labbox.domain.artifacts import ArtifactEntry, ArtifactManifest

    payload = b"x" * 16
    source = tmp_path / "outside.pt"
    source.write_bytes(payload)
    manifest = ArtifactManifest(
        artifacts=[
            ArtifactEntry(
                kind=ArtifactKind.CHECKPOINT,
                path="../../escape.pt",
                abs_path=str(source),
                size_bytes=len(payload),
            )
        ]
    )
    store_dir = tmp_path / "run"
    copy_artifacts(manifest, store_dir, limit_bytes=10_000)
    assert (store_dir / "checkpoints" / "escape.pt").is_file()
    assert not (tmp_path.parent / "escape.pt").exists()


def test_copy_streams_instead_of_reading_the_whole_file(
    tmp_path: Path, config: LabboxConfig, monkeypatch
) -> None:
    """A checkpoint must not be pulled into memory just to be copied."""
    from labbox.domain.artifacts import ArtifactEntry, ArtifactManifest

    source = tmp_path / "checkpoints"
    source.mkdir()
    (source / "big.pt").write_bytes(b"z" * (3 * 1024 * 1024))
    manifest = ArtifactManifest(
        artifacts=[
            ArtifactEntry(
                kind=ArtifactKind.CHECKPOINT,
                path="checkpoints/big.pt",
                abs_path=str(source / "big.pt"),
                size_bytes=3 * 1024 * 1024,
            )
        ]
    )

    def explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("read_bytes() loads the whole file into memory")

    monkeypatch.setattr(Path, "read_bytes", explode)
    copy_artifacts(manifest, tmp_path / "run", limit_bytes=10 * 1024 * 1024)
    assert (tmp_path / "run" / "checkpoints" / "big.pt").stat().st_size == 3 * 1024 * 1024


def test_files_outside_the_project_are_skipped(tmp_path: Path, config: LabboxConfig) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    outside_file = tmp_path / "elsewhere.pt"
    outside_file.write_bytes(b"y" * 8)
    assert outside(outside_file, project) is True
    assert outside(project / "inside.pt", project) is False
    config.artifacts.patterns = ["../*.pt"]
    manifest = collect(project, config)
    assert manifest.artifacts == []


def test_since_filters_pre_existing_checkpoints(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    old = tmp_path / "checkpoints" / "old.pt"
    old.write_bytes(b"1")
    import os
    import time

    stamp = time.time() - 3600
    os.utime(old, (stamp, stamp))
    (tmp_path / "checkpoints" / "new.pt").write_bytes(b"2")
    now = __import__("datetime").datetime.now().astimezone()
    manifest = ArtifactCollector(project_root=tmp_path, config=config).collect(
        since=now - timedelta(minutes=5)
    )
    names = {Path(entry.path).name for entry in manifest.artifacts}
    assert "new.pt" in names
    assert "old.pt" not in names, "a stale checkpoint from a previous run is not this run's artifact"


def test_duplicate_paths_are_deduplicated(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "best.pt").write_bytes(b"z" * 4)
    config.artifacts.patterns = ["checkpoints/*.pt", "checkpoints/**/*.pt"]
    manifest = collect(tmp_path, config)
    assert len(manifest.artifacts) == 1


def test_max_count_caps_the_manifest(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "checkpoints").mkdir()
    for index in range(20):
        (tmp_path / "checkpoints" / f"c{index}.pt").write_bytes(b"1")
    manifest = collect(tmp_path, config, max_count=5)
    assert len(manifest.artifacts) == 5
    assert any("stopped after 5" in warning for warning in manifest.warnings)
