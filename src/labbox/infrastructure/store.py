"""Atomic, crash-safe writes into ``.labbox/runs/<run-id>/`` (§6.2).

Every summary file is written as ``tmp -> flush -> fsync -> os.replace``. A crash while
writing must leave the *previous* complete file in place, never half a JSON document:
``labbox compare`` is useless if one interrupted run corrupts another run's record.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from labbox.domain.common import SchemaModel, as_dict
from labbox.errors import STORE_READ_FAILED, LabBoxError
from labbox.paths import ProjectLayout, RunPaths


def _json_default(obj: Any) -> Any:
    return as_dict(obj)


def dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=_json_default,
    )


class FilesystemStore:
    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    # --- directories ---------------------------------------------------------
    def ensure_run_dirs(self, paths: RunPaths) -> None:
        for rel in ("", paths.logs_dir, paths.samples_dir, paths.checkpoints_dir):
            (paths.dir / rel).mkdir(parents=True, exist_ok=True)
        # Touch the log files up front: an empty-but-present stdout.log distinguishes
        # "the child produced nothing" from "we never got to the child" later on.
        for path in (paths.stdout_log, paths.stderr_log):
            path.touch(exist_ok=True)

    def run_paths(self, run_id: str) -> RunPaths:
        return self.layout.run_dir(run_id)

    def list_run_dirs(self) -> list[Path]:
        runs = self.layout.runs_dir
        if not runs.is_dir():
            return []
        return sorted(
            (p for p in runs.iterdir() if p.is_dir() and (p / "run.json").exists()),
            key=lambda p: p.name,
        )

    # --- atomic writes -------------------------------------------------------
    def write_json_atomic(self, path: Path, obj: Any) -> None:
        text = obj if isinstance(obj, str) else dumps(_dumpable(obj))
        self.write_text_atomic(path, text)

    def write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", text=True)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                _fsync(fh)
            _replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise LabBoxError(
                f"Failed to write {path.name}: {exc}",
                code="STORE_WRITE_FAILED",
                component="store",
                details={"path": str(path), "errno": getattr(exc, "errno", None)},
            ) from exc

    def append_jsonl(self, path: Path, record: Any) -> None:
        """Append-only writer used for logs, events and samples.

        Kept deliberately simple: no buffering across calls, so an interrupted run still
        has every sample it managed to record before the kill.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (
            json.dumps(_dumpable(record), ensure_ascii=False, default=_json_default)
            if not isinstance(record, str)
            else record
        )
        try:
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line.rstrip("\n") + "\n")
        except OSError as exc:
            raise LabBoxError(
                f"Failed to append to {path.name}: {exc}",
                code="LOG_WRITE_FAILED",
                component="store",
                details={"path": str(path)},
            ) from exc

    def append_text(self, path: Path, text: str) -> bool:
        """Append plain text, reporting rather than raising on failure.

        Used from the event and error paths, where a disk that has already failed must not
        turn a diagnostic into a second, louder exception. Returns whether it landed.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        except OSError:
            return False
        return True

    # --- reads ---------------------------------------------------------------
    def read_json(self, path: Path, *, strict: bool = True) -> dict[str, Any] | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if strict:
                raise
            return None
        except OSError as exc:
            if strict:
                raise LabBoxError(
                    f"Cannot read {path.name}: {exc}",
                    code=STORE_READ_FAILED,
                    component="store",
                    details={"path": str(path)},
                ) from exc
            return None
        if not raw.strip():
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            if strict:
                # A read failure reported as a write failure sends the reader to check a
                # disk that was never touched; the file on disk is the damaged one.
                raise LabBoxError(
                    f"Corrupt JSON in {path}: {exc}",
                    code=STORE_READ_FAILED,
                    component="store",
                    details={"path": str(path)},
                ) from exc
            return None
        return data if isinstance(data, dict) else {"value": data}

    def read_run_record(self, run_dir: Path) -> dict[str, Any] | None:
        return self.read_json(run_dir / "run.json", strict=False)

    def load_model(self, path: Path, model: type[SchemaModel]) -> Any:
        data = self.read_json(path, strict=True) or {}
        return model.model_validate(data)

    # --- maintenance ---------------------------------------------------------
    def delete_run(self, run_dir: Path) -> bool:
        """Delete one run directory. Returns whether anything was there.

        Refuses the runs root. A directory that has already gone is *not* an error:
        ``labbox clean`` builds its victim list first and a concurrent delete (or a retry
        after a partial success) would otherwise abort the whole command over nothing.
        """
        target = self.layout._guard(run_dir)
        if target == self.layout.runs_dir.resolve():
            raise LabBoxError(
                "Refusing to delete the runs root.",
                code="PATH_OUTSIDE_RUNS",
                component="store",
                recoverable=False,
            )
        if not target.exists():
            return False
        import shutil

        shutil.rmtree(target)
        return True

    def dir_size_bytes(self, path: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
        return total

    def total_store_bytes(self) -> int:
        return sum(self.dir_size_bytes(p) for p in self.list_run_dirs())

    def newest_run(self) -> Path | None:
        runs = self.list_run_dirs()
        return runs[-1] if runs else None


def _dumpable(obj: Any) -> Any:
    if isinstance(obj, SchemaModel):
        return obj.model_dump(mode="json")
    return obj


def _fsync(fh: Any) -> None:
    try:
        fh.flush()
        os.fsync(fh.fileno())
    except (OSError, ValueError):
        # fsync is unsupported on some filesystems (and on Windows pipes). Durability is
        # a nice-to-have here; os.replace still guarantees atomic visibility.
        return


def _replace(tmp: Path, target: Path) -> None:
    try:
        os.replace(tmp, target)
    except PermissionError:
        # Antivirus/indexers briefly hold new files on Windows; one retry covers it.
        import time

        time.sleep(0.05)
        os.replace(tmp, target)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")
