"""Artifact discovery and hashing (§9.5).

Checkpoints are *referenced*, not copied, by default: a 12 GiB checkpoint copied per run
would turn a recording tool into a disk-space incident. Hashing is likewise bounded —
above ``hash_max_bytes`` we record metadata plus ``deferred_too_large`` and say so,
instead of silently spending ten minutes hashing during a run.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from labbox.config.model import LabboxConfig
from labbox.domain.artifacts import ArtifactEntry, ArtifactKind, ArtifactManifest, HashState

_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx", ".h5", ".joblib"}
_CHUNK = 1024 * 1024


class ArtifactCollector:
    def __init__(
        self,
        *,
        project_root: Path,
        config: LabboxConfig,
        extra_patterns: Sequence[str] = (),
        hash_max_bytes: int | None = None,
        scan_roots: Sequence[Path] | None = None,
    ) -> None:
        self.root = Path(project_root)
        self.config = config
        self.patterns = list(config.artifacts.patterns) + list(extra_patterns)
        self.hash_max_bytes = hash_max_bytes if hash_max_bytes is not None else config.artifacts.hash_max_bytes
        self.max_count = int(config.artifacts.max_count)
        self.scan_roots = [Path(p) for p in scan_roots] if scan_roots else [self.root]

    def collect(self, *, metrics_files: Iterable[Path] = (), since: datetime | None = None) -> ArtifactManifest:
        manifest = ArtifactManifest(
            scanned_globs=list(self.patterns),
            hash_threshold_bytes=self.hash_max_bytes,
        )
        seen: set[str] = set()
        for path, kind, discovered_by in self._candidates(manifest, since=since):
            key = _key(path)
            if key in seen:
                continue
            seen.add(key)
            entry = self._describe(path, kind=kind, discovered_by=discovered_by)
            manifest.artifacts.append(entry)
            if len(manifest.artifacts) >= self.max_count:
                manifest.warnings.append(f"stopped after {self.max_count} artifacts")
                break
        for path in metrics_files:
            if _key(path) in seen:
                continue
            entry = self._describe(path, kind=ArtifactKind.METRICS, discovered_by="metrics_file")
            manifest.artifacts.append(entry)
        return manifest

    # --- discovery -----------------------------------------------------------
    def _candidates(self, manifest: ArtifactManifest, *, since: datetime | None):
        seen_before_yield: set[str] = set()
        for pattern in self.patterns:
            for base in self.scan_roots:
                try:
                    matches = sorted(Path(base).glob(pattern))
                except (OSError, ValueError) as exc:
                    manifest.errors.append(
                        {"code": "ARTIFACT_GLOB_FAILED", "message": f"{pattern}: {exc}", "glob": pattern}
                    )
                    continue
                for path in matches:
                    if not path.is_file():
                        continue
                    if outside(path, self.root):
                        manifest.warnings.append(f"skipped {path}: outside the project root")
                        continue
                    if not _fresh_enough(path, since):
                        continue
                    key = _key(path)
                    if key in seen_before_yield:
                        continue
                    seen_before_yield.add(key)
                    yield path, _kind_for(path), "config_glob"

        checkpoint_dirs = [self.root / "checkpoints", self.root / "outputs", self.root / "runs"]
        for directory in checkpoint_dirs:
            if not directory.is_dir():
                continue
            for path in _walk_suffixes(directory, _CHECKPOINT_SUFFIXES):
                if not _fresh_enough(path, since):
                    continue
                key = _key(path)
                if key in seen_before_yield:
                    continue
                seen_before_yield.add(key)
                yield path, ArtifactKind.CHECKPOINT, "checkpoint_suffix"

    def _describe(self, path: Path, *, kind: ArtifactKind, discovered_by: str) -> ArtifactEntry:
        entry = ArtifactEntry(
            kind=kind,
            path=_relative_to(self.root, path),
            abs_path=str(path.resolve()),
            discovered_by=discovered_by,
            hash_state=HashState.NOT_COMPUTED,
        )
        try:
            stat = path.stat()
        except OSError as exc:
            entry.error = str(exc)
            entry.hash_state = HashState.FAILED
            return entry
        entry.size_bytes = int(stat.st_size)
        entry.mtime = datetime.fromtimestamp(stat.st_mtime).astimezone()
        if entry.size_bytes is not None and entry.size_bytes > self.hash_max_bytes:
            entry.hash_state = HashState.DEFERRED_TOO_LARGE
            entry.error = f"larger than hash_max_bytes={self.hash_max_bytes}; pass --hash-artifacts to override"
            return entry
        try:
            entry.sha256 = _sha256(path)
            entry.hash_state = HashState.COMPUTED
            entry.hash_bytes_read = entry.size_bytes
        except OSError as exc:
            entry.error = str(exc)
            entry.hash_state = HashState.FAILED
        return entry


def copy_artifacts(manifest: ArtifactManifest, store_dir: Path, *, limit_bytes: int) -> list[str]:
    """Opt-in copy of artifacts into the run directory, bounded by ``limit_bytes``.

    Two properties this must keep:

    * the recorded relative path is preserved, so ``runs/a/model/best.pt`` and
      ``runs/b/model/best.pt`` land side by side instead of one overwriting the other;
    * a copy is streamed in chunks. ``read_bytes`` on a multi-gigabyte checkpoint is not a
      copy, it is an out-of-memory kill during the *post-run* stage, after the training
      has already been paid for.
    """
    copied: list[str] = []
    budget = limit_bytes
    for entry in manifest.artifacts:
        if entry.size_bytes is None or budget - entry.size_bytes < 0:
            if entry.size_bytes is not None:
                manifest.warnings.append(f"not copied (size limit): {entry.path}")
            continue
        source = Path(entry.abs_path or entry.path)
        if not source.is_file():
            continue
        target = store_dir / "checkpoints" / _archive_target(entry.path, Path(entry.path).name)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_stream(source, target)
        except OSError as exc:
            manifest.errors.append({"code": "ARTIFACT_COPY_FAILED", "message": f"{entry.path}: {exc}"})
            continue
        entry.archived = True
        entry.storage_uri = str(target.resolve())
        budget -= entry.size_bytes
        copied.append(entry.path)
    return copied


def _archive_target(recorded: str, fallback_name: str) -> Path:
    """Where an artifact lands under ``<run>/checkpoints/``.

    The project-relative path is kept, minus a leading ``checkpoints/`` (the archive root
    already says that). Collapsing every artifact to its basename — what this used to do —
    silently overwrote ``a/best.pt`` with ``c/best.pt``, so an archived run directory
    contained one checkpoint instead of two. A path that would climb out of the run
    directory is reduced to its leaf, because a crafted ``abs_path``/``path`` pair must
    never choose where an archive lands.
    """
    candidate = Path(recorded.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return Path(fallback_name)
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if parts and parts[0].lower() == "checkpoints":
        parts = parts[1:]
    return Path(*parts) if parts else Path(fallback_name)


def _copy_stream(source: Path, target: Path) -> None:
    with source.open("rb") as reader, target.open("wb") as writer:
        for chunk in iter(lambda: reader.read(_CHUNK), b""):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def _walk_suffixes(directory: Path, suffixes: set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", ".labbox"}]
        base = Path(dirpath)
        for name in sorted(filenames):
            path = base / name
            if path.suffix.lower() in suffixes:
                yield path


def _fresh_enough(path: Path, since: datetime | None) -> bool:
    """Artifacts discovered by a glob are subject to the same freshness rule.

    Without this, a checkpoint left over from last week's run would be recorded as an
    output of this one simply because it matches ``checkpoints/**/*.pt``.
    """
    if since is None:
        return True
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return False
    return modified >= since


def outside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return False
    except (ValueError, OSError):
        return True


def _kind_for(path: Path) -> ArtifactKind:
    suffix = path.suffix.lower()
    if suffix in _CHECKPOINT_SUFFIXES:
        return ArtifactKind.CHECKPOINT
    if suffix in {".yaml", ".yml", ".toml", ".json", ".ini"}:
        return ArtifactKind.CONFIG
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return ArtifactKind.IMAGE
    if suffix == ".log":
        return ArtifactKind.LOG
    return ArtifactKind.OTHER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return str(path).replace("\\", "/")


def _key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except OSError:
        return str(path).lower()


__all__ = ["ArtifactCollector", "copy_artifacts", "fnmatch", "outside"]
