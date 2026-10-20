"""Dataset manifests (§8.4).

The digest definition is the whole point of this module, so it is spelled out once and
guarded by tests:

    per_file = f"{rel_path}|{size}|{sha256(head64)}|{sha256(tail64)}"
    digest   = sha256("\\n".join(per_file over entries sorted by rel_path))

Modification time is **never** an input. Copying, unzipping or re-checkout a dataset
rewrites every mtime; including it would make the most important diagnostic in LabBox
("the data changed") fire on nearly every comparison and teach users to ignore it.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path

from labbox.config.model import LabboxConfig
from labbox.domain.common import now_local
from labbox.domain.dataset import (
    DatasetEntry,
    DatasetKind,
    DatasetManifest,
    DatasetStrategy,
    FileEntry,
)

HEAD_BYTES = 64
TAIL_BYTES = 64
_CHUNK = 65536


class DatasetManifestBuilder:
    def __init__(
        self,
        *,
        project_root: Path,
        config: LabboxConfig,
        strategy: DatasetStrategy | None = None,
        now: float | None = None,
    ) -> None:
        self.root = Path(project_root)
        self.config = config
        self.strategy = strategy or DatasetStrategy(config.datasets.strategy)
        self.exclusions: list[str] = list(config.datasets.exclude)
        self.max_files = int(config.datasets.max_files)
        self.budget = float(config.datasets.time_budget_seconds)
        self.follow_symlinks = bool(config.datasets.follow_symlinks)
        self._deadline = (now if now is not None else time.monotonic()) + self.budget

    # --- public API ----------------------------------------------------------
    def build(self, declared: Sequence[str]) -> DatasetManifest:
        manifest = DatasetManifest(discovery="declared" if declared else "none")
        if not declared:
            manifest.warnings.append(
                "no dataset declared: pass --dataset <path|version> or set [datasets] paths in .labbox/config.toml"
            )
            return manifest
        for spec in declared:
            manifest.datasets.append(self.build_one(spec))
        return manifest

    def build_one(self, spec: str) -> DatasetEntry:
        path = self._resolve(spec)
        if path is None:
            return self._external_entry(spec)
        if not path.exists():
            return DatasetEntry(
                id=_short_id(str(path)),
                source=_display_path(self.root, path),
                strategy=self.strategy,
                warnings=[f"path does not exist: {path}"],
                errors=[{"code": "DATASET_PATH_MISSING", "message": f"{path} not found"}],
                generated_at=now_local(),
            )
        if path.is_file():
            return self._single_file_entry(spec, path)
        return self._directory_entry(spec, path)

    # --- internals -----------------------------------------------------------
    def _external_entry(self, spec: str) -> DatasetEntry:
        """A dataset we cannot walk: a version id or a URI, recorded as exactly that.

        The strategy is ``external_version``, not the configured scan strategy. Recording
        ``fast_manifest`` here claimed a head+tail sampling method that never ran, and it
        left ``is_content_proof`` false for a version that *is* pinned by the user.
        """
        text = spec.strip()
        return DatasetEntry(
            id=_short_id(text),
            source=text,
            kind=DatasetKind.EXTERNAL,
            strategy=DatasetStrategy.EXTERNAL_VERSION,
            digest="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            digest_inputs="external_version_id",
            external_version=text,
            verified=True,
            sampled_only=False,
            warnings=[
                "not a local path; recorded as an externally identified version, so the digest "
                "identifies the version string and not the bytes it resolved to"
            ],
            generated_at=now_local(),
        )

    def _resolve(self, spec: str) -> Path | None:
        if not _looks_like_path(spec):
            # A bare name such as ``data`` carries no separator, but it may still be a real
            # directory in this project. Existence is checked against the project root only:
            # testing it against the process working directory recorded whatever happened to
            # sit next to whoever ran the command, which is not the same dataset at all.
            candidate = Path(spec.strip())
            if candidate.is_absolute() or _URI_SCHEME.match(spec.strip()):
                return None
            local = self.root / candidate
            try:
                return local if local.exists() else None
            except OSError:  # pragma: no cover - unreadable project root
                return None
        candidate = Path(spec)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate

    def _directory_entry(self, spec: str, path: Path) -> DatasetEntry:
        started = time.monotonic()
        entries: list[FileEntry] = []
        total_bytes = 0
        zero_byte = 0
        truncated_reason: str | None = None
        errors: list[dict] = []
        for rel, file_path in _walk(path, exclusions=self.exclusions, follow_symlinks=self.follow_symlinks):
            try:
                size = file_path.stat().st_size
            except OSError as exc:
                errors.append({"code": "DATASET_STAT_FAILED", "message": f"{rel}: {exc}", "path": rel})
                continue
            if len(entries) >= self.max_files:
                truncated_reason = f"stopped at max_files={self.max_files}"
                break
            if time.monotonic() > self._deadline:
                truncated_reason = f"stopped after time budget {self.budget:.0f}s"
                break
            head = tail = None
            if size:
                try:
                    head, tail = _head_tail(file_path)
                except OSError as exc:
                    errors.append({"code": "DATASET_READ_FAILED", "message": f"{rel}: {exc}", "path": rel})
                    continue
            else:
                zero_byte += 1
            entries.append(
                FileEntry(
                    path=rel,
                    size_bytes=size,
                    head_digest=head,
                    tail_digest=tail,
                    mtime=datetime.fromtimestamp(file_path.stat().st_mtime).astimezone(),
                )
            )
            total_bytes += size
        if self.strategy is DatasetStrategy.FULL_CONTENT:
            for entry in entries:
                try:
                    entry.content_full = _full_sha256(path / entry.path)
                except OSError as exc:  # pragma: no cover
                    errors.append({"code": "DATASET_READ_FAILED", "message": f"{entry.path}: {exc}"})
        digest = _digest(entries, full=self.strategy is DatasetStrategy.FULL_CONTENT) if entries else None
        warnings: list[str] = []
        if not entries:
            warnings.append("no files found; an empty manifest is not proof of an empty dataset")
        if zero_byte:
            warnings.append(f"{zero_byte} zero-byte file(s): content is not verified for these")
        if truncated_reason:
            warnings.append(f"manifest truncated: {truncated_reason}; digest is NOT comparable with a complete one")
        if self.strategy is DatasetStrategy.FAST_MANIFEST and entries:
            warnings.append("head+tail sampling only: this digest does not prove content immutability")
        return DatasetEntry(
            id=_short_id(spec),
            source=_display_path(self.root, path),
            kind=DatasetKind.FILESYSTEM,
            strategy=self.strategy,
            digest=digest,
            digest_inputs="rel_path+size+full_sha256"
            if self.strategy is DatasetStrategy.FULL_CONTENT
            else "rel_path+size+head64+tail64",
            file_count=len(entries),
            total_bytes=total_bytes,
            generated_at=now_local(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            exclusions=list(self.exclusions),
            zero_byte_files=zero_byte,
            sampled_only=self.strategy is not DatasetStrategy.FULL_CONTENT,
            truncated=truncated_reason is not None,
            cancellation_reason=truncated_reason,
            warnings=warnings,
            errors=errors,
            entries=entries,
        )

    def _single_file_entry(self, spec: str, path: Path) -> DatasetEntry:
        size = path.stat().st_size
        head = tail = None
        if size:
            head, tail = _head_tail(path)
        entry = FileEntry(
            path=_display_path(self.root, path),
            size_bytes=size,
            head_digest=head,
            tail_digest=tail,
            mtime=datetime.fromtimestamp(path.stat().st_mtime).astimezone(),
        )
        return DatasetEntry(
            id=_short_id(spec),
            source=entry.path,
            strategy=self.strategy,
            digest=_digest([entry], full=self.strategy is DatasetStrategy.FULL_CONTENT),
            digest_inputs="rel_path+size+full_sha256"
            if self.strategy is DatasetStrategy.FULL_CONTENT
            else "rel_path+size+head64+tail64",
            file_count=1,
            total_bytes=size,
            generated_at=now_local(),
            zero_byte_files=1 if size == 0 else 0,
            sampled_only=self.strategy is not DatasetStrategy.FULL_CONTENT,
            warnings=["single-file dataset recorded by name"],
        )


_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]{1,15}://")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_like_path(spec: str) -> bool:
    """Decide whether a --dataset value is a local path or an external version id.

    Deliberately conservative: ``cifar100-train-v3`` is a version label, ``data/cifar`` is
    a path. Anything ambiguous is reported as externally identified rather than guessed at,
    because a wrong guess here produces a wrong data digest.

    Three checks that previously misfired:

    * ``s3://bucket/key`` and ``https://host/dataset`` were read as paths because they
      contain ``/``. A URI scheme is never a local path, so it is tested first.
    * ``Path(text).exists()`` was evaluated against the *process* working directory, so a
      directory that happens to exist next to whoever ran ``labbox`` was recorded as this
      project's dataset. Existence is only consulted relative to the project root now.
    * the Windows separator test looked for two literal backslashes (``"\\\\"`` in source),
      which a normal ``data\\train`` never matches; it looked for a doubled separator rather
      than a single one.
    """
    text = spec.strip()
    if not text:
        return False
    if _URI_SCHEME.match(text):
        return False
    if _WINDOWS_DRIVE.match(text):
        return True
    if text.startswith((".", "~", "/", "\\")):
        return True
    if ":" in text[2:]:
        # A colon before any path separator means a scheme or a tag, not a local path.
        return False
    return "/" in text or "\\" in text


def _short_id(spec: str) -> str:
    cleaned = spec.replace("\\", "/").rstrip("/")
    name = cleaned.rsplit("/", 1)[-1] or cleaned or "dataset"
    return name[:64]


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _walk(root: Path, *, exclusions: Sequence[str], follow_symlinks: bool) -> Iterable[tuple[str, Path]]:
    """Yield (posix-relative-path, absolute path) in deterministic sorted order.

    Sorted before any hashing: a digest must not depend on the order the OS happened to
    report directory entries in.
    """
    collected: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        base = Path(dirpath)
        rel_base = base.relative_to(root).as_posix() if base != root else ""
        dirnames[:] = sorted(d for d in dirnames if not _is_excluded(_join(rel_base, d), exclusions))
        for name in sorted(filenames):
            rel = _join(rel_base, name)
            target = base / name
            if _is_excluded(rel, exclusions):
                continue
            if not follow_symlinks and target.is_symlink():
                continue
            collected.append((rel, target))
    collected.sort(key=lambda item: item[0])
    return collected


def _join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _is_excluded(rel_path: str, patterns: Iterable[str]) -> bool:
    """Match a POSIX relative path against config globs.

    ``**`` is accepted in the shapes users actually write (``**/__pycache__/**``,
    ``*.tmp``): the path is tested as-is, as a basename, and with any leading ``**/``
    stripped, so an exclusion never depends on how deep the file happens to be.
    """
    name = rel_path.rsplit("/", 1)[-1]
    for pattern in patterns:
        candidates = (pattern, pattern[3:] if pattern.startswith("**/") else None, pattern.lstrip("*/"))
        for candidate in candidates:
            if candidate and (fnmatchcase(rel_path, candidate) or fnmatchcase(name, candidate)):
                return True
    return False


def _head_tail(path: Path) -> tuple[str, str | None]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        head = fh.read(HEAD_BYTES)
        digest.update(head)
        head_hash = digest.hexdigest()
        tail_hash: str | None = None
        try:
            size = fh.seek(0, os.SEEK_END)
            if size > HEAD_BYTES:
                start = max(size - TAIL_BYTES, HEAD_BYTES)
                fh.seek(start)
                tail = fh.read(TAIL_BYTES)
                tail_hash = hashlib.sha256(tail).hexdigest()
        except OSError:
            tail_hash = None
    return f"sha256:{head_hash}", (f"sha256:{tail_hash}" if tail_hash else None)


def _full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(entries: Sequence[FileEntry], *, full: bool = False) -> str:
    lines: list[str] = []
    for entry in sorted(entries, key=lambda e: e.path):
        if full and entry.content_full:
            content = entry.content_full
        else:
            content = f"{entry.head_digest or ''}+{entry.tail_digest or ''}"
        lines.append(f"{entry.path}|{entry.size_bytes}|{content}")
    payload = "\n".join(lines).encode("utf-8", "surrogatepass")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "HEAD_BYTES",
    "DatasetManifestBuilder",
    "digest_entries",
    "is_excluded",
    "looks_like_path",
    "walk_sorted",
]


# Test-facing aliases keep the private helpers reachable without making them public API.
digest_entries = _digest
is_excluded = _is_excluded
looks_like_path = _looks_like_path
walk_sorted = _walk
