"""Dataset manifest digest semantics — the anti-false-positive tests (§8.4).

The headline property: copying or touching a dataset must NOT change the digest, while a
single byte of content or one extra file must.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.dataset import DatasetStrategy
from labbox.recorder.dataset import DatasetManifestBuilder, digest_entries, is_excluded


@pytest.fixture
def config() -> LabboxConfig:
    return LabboxConfig.default()


def builder(root: Path, config: LabboxConfig, **overrides) -> DatasetManifestBuilder:
    for key, value in overrides.items():
        setattr(config.datasets, key, value)
    return DatasetManifestBuilder(project_root=root, config=config)


def make_dataset(root: Path) -> Path:
    data = root / "data" / "train"
    data.mkdir(parents=True)
    for index in range(10):
        (data / f"sample_{index:03d}.bin").write_bytes(bytes([index]) * 200 + b"tail" * 20)
    return data


def test_digest_survives_a_copy_to_another_location(tmp_path: Path, config: LabboxConfig) -> None:
    """A dataset that moved (different absolute path, same relative layout) keeps its digest.

    This is the property that makes "same data" answerable after a colleague re-downloads
    it into a different directory.
    """
    import shutil

    source = make_dataset(tmp_path)
    first = builder(source.parent, config).build_one(str(source))
    moved = tmp_path / "somewhere else"
    shutil.copytree(source, moved / source.name)
    time.sleep(0.02)
    second = builder(moved, config).build_one(str(moved / source.name))
    assert first.digest and second.digest
    assert first.file_count == second.file_count == 10
    assert first.digest == second.digest


def test_digest_over_the_same_tree_ignores_mtime(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    build = builder(tmp_path, config)
    first = build.build_one(str(data))
    for path in sorted(data.iterdir()):
        os.utime(path, (1_000_000.0, 1_000_000.0))  # mtime jumps back to 1970
    second = builder(tmp_path, config).build_one(str(data))
    assert first.digest == second.digest
    assert first.total_bytes == second.total_bytes


def test_digest_changes_when_content_changes(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    before = builder(tmp_path, config).build_one(str(data)).digest
    victim = sorted(data.iterdir())[4]
    payload = bytearray(victim.read_bytes())
    payload[-1] ^= 0xFF  # flip one bit in the tail region
    victim.write_bytes(bytes(payload))
    after = builder(tmp_path, config).build_one(str(data)).digest
    assert before != after


def test_digest_detects_a_renamed_file(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    before = builder(tmp_path, config).build_one(str(data)).digest
    (data / "sample_000.bin").rename(data / "sample_999.bin")
    after = builder(tmp_path, config).build_one(str(data)).digest
    assert before != after, "the path is part of the digest"


def test_ordering_is_deterministic(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    entries = builder(tmp_path, config).build_one(str(data)).entries
    paths = [entry.path for entry in entries]
    assert paths == sorted(paths)
    assert digest_entries(entries, full=False) == digest_entries(list(reversed(entries)), full=False)
    assert entries and all(entry.path.endswith(".bin") for entry in entries)
    assert entries[0].path != entries[-1].path


def test_exclusions_are_honoured(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    (data / "__pycache__").mkdir()
    (data / "__pycache__" / "junk.pyc").write_bytes(b"x" * 50)
    (data / "scratch.tmp").write_bytes(b"y" * 10)
    entry = builder(tmp_path, config).build_one(str(data))
    assert entry.file_count == 10, [e.path for e in entry.entries]
    assert is_excluded("train/__pycache__/x.pyc", ["**/__pycache__/**"])
    assert is_excluded("a/b/tmp.tmp", ["**/*.tmp"])
    assert not is_excluded("a/b/keep.bin", ["**/*.tmp"])


def test_zero_byte_files_are_flagged_not_silently_equal(tmp_path: Path, config: LabboxConfig) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "empty_a.bin").write_bytes(b"")
    (data / "empty_b.bin").write_bytes(b"")
    (data / "real.bin").write_bytes(b"content")
    entry = builder(tmp_path, config).build_one(str(data))
    assert entry.zero_byte_files == 2
    assert any("zero-byte" in warning for warning in entry.warnings)
    assert entry.sampled_only is True
    assert entry.is_content_proof is False, "fast_manifest is never a content proof"


def test_truncation_is_recorded_and_digest_marked_uncomparable(tmp_path: Path, config: LabboxConfig) -> None:
    data = make_dataset(tmp_path)
    entry = builder(tmp_path, config, max_files=3).build_one(str(data))
    assert entry.file_count == 3
    assert entry.truncated is True
    assert entry.cancellation_reason and "max_files" in entry.cancellation_reason
    assert any("NOT comparable" in warning for warning in entry.warnings)


def test_time_budget_stops_a_large_scan(tmp_path: Path, config: LabboxConfig) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for index in range(200):
        (data / f"f_{index}.bin").write_bytes(bytes(index % 250) * 4000)
    entry = builder(tmp_path, config, time_budget_seconds=0.0).build_one(str(data))
    assert entry.truncated is True
    assert entry.file_count is not None and entry.file_count < 200


def test_full_content_strategy_hashes_whole_file(tmp_path: Path, config: LabboxConfig) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.bin").write_bytes(b"hello world, this payload is longer than sixty four bytes" * 4)
    fast = builder(tmp_path, config).build_one(str(data))
    full = builder(tmp_path, config, strategy="full_content").build_one(str(data))
    assert full.strategy is DatasetStrategy.FULL_CONTENT
    assert full.is_content_proof is True
    assert full.digest != fast.digest


def test_missing_path_and_external_version_are_both_honest(tmp_path: Path, config: LabboxConfig) -> None:
    entry_missing = builder(tmp_path, config).build_one("data/does-not-exist")
    assert entry_missing.digest is None
    assert any("does not exist" in warning for warning in entry_missing.warnings)
    external = builder(tmp_path, config).build_one("huggingface:mnli/revisions/abc123")
    assert external.digest
    assert external.verified is True
    assert external.file_count is None


# ---------------------------------------------------------------------------
# path vs version id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec",
    [
        "s3://bucket/key",
        "https://example.org/dataset.tar",
        "gs://bucket/data",
        "hf://datasets/cifar100",
    ],
)
def test_a_uri_is_never_mistaken_for_a_local_path(tmp_path: Path, config: LabboxConfig, spec: str) -> None:
    """These contain ``/`` and used to be resolved against the filesystem."""
    from labbox.recorder.dataset import looks_like_path

    assert looks_like_path(spec) is False
    entry = builder(tmp_path, config).build_one(spec)
    assert entry.kind.value == "external"
    assert entry.file_count is None
    assert entry.external_version == spec


def test_a_windows_separator_makes_it_a_path(tmp_path: Path, config: LabboxConfig) -> None:
    """``data\\train`` used to fail the test, because the check looked for *two* slashes."""
    from labbox.recorder.dataset import looks_like_path

    assert looks_like_path("data\\train") is True
    directory = tmp_path / "data"
    (directory / "train").mkdir(parents=True)
    (directory / "train" / "a.bin").write_bytes(b"x" * 32)
    entry = builder(tmp_path, config).build_one("data\\train")
    assert entry.kind.value == "filesystem"
    assert entry.file_count == 1
    assert entry.digest


def test_a_bare_name_is_resolved_against_the_project_root_not_the_cwd(
    tmp_path: Path, config: LabboxConfig, monkeypatch
) -> None:
    """A directory next to whoever ran the command is not this project's dataset."""
    project = tmp_path / "proj"
    (project / "data").mkdir(parents=True)
    (project / "data" / "a.bin").write_bytes(b"y" * 32)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "data").mkdir()
    (elsewhere / "data" / "a.bin").write_bytes(b"z" * 32)
    monkeypatch.chdir(elsewhere)

    entry = builder(project, config).build_one("data")
    assert entry.kind.value == "filesystem"
    assert entry.total_bytes == 32, "the project-root copy is 32 bytes, not the cwd one"


def test_a_version_label_is_not_walked(tmp_path: Path, config: LabboxConfig) -> None:
    from labbox.recorder.dataset import looks_like_path

    assert looks_like_path("cifar100-train-v3") is False
    entry = builder(tmp_path, config).build_one("cifar100-train-v3")
    assert entry.kind.value == "external"
    assert entry.strategy is DatasetStrategy.EXTERNAL_VERSION
    assert entry.is_content_proof is True, "an externally pinned version is a proof of itself"
    assert entry.sampled_only is False
    assert entry.digest_inputs == "external_version_id"
    assert entry.external_version == "cifar100-train-v3"


def test_external_entry_no_longer_claims_a_scan_strategy(tmp_path: Path, config: LabboxConfig) -> None:
    """It used to record ``fast_manifest``: a head+tail sampling method that never ran."""
    entry = builder(tmp_path, config).build_one("some-dataset:v2")
    assert entry.strategy is DatasetStrategy.EXTERNAL_VERSION
    assert entry.strategy.value != "fast_manifest"
    assert entry.digest, "the version string is what identifies it"
    assert entry.digest == builder(tmp_path, config).build_one("some-dataset:v2").digest


def test_digest_inputs_name_the_method_actually_used(tmp_path: Path, config: LabboxConfig) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.bin").write_bytes(b"hello world" * 20)
    fast = builder(tmp_path, config).build_one(str(data))
    assert fast.digest_inputs == "rel_path+size+head64+tail64"
    full = builder(tmp_path, config, strategy="full_content").build_one(str(data))
    assert full.digest_inputs == "rel_path+size+full_sha256"


def test_undeclared_datasets_produce_no_guess(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "data").mkdir()
    manifest = builder(tmp_path, config).build([])
    assert manifest.not_declared is True
    assert manifest.datasets == []
    assert any("no dataset declared" in warning for warning in manifest.warnings)


def test_manifest_roundtrips_without_the_file_list(tmp_path: Path, config: LabboxConfig) -> None:
    import json

    data = make_dataset(tmp_path)
    manifest = builder(tmp_path, config).build([str(data)])
    payload = json.loads(json.dumps(manifest.to_json_dict(), ensure_ascii=False))
    assert "entries" not in payload["datasets"][0]
    assert payload["datasets"][0]["file_count"] == 10
    restored = type(manifest).model_validate(payload)
    assert restored.datasets[0].digest == manifest.datasets[0].digest
