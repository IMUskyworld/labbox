"""Git state (§9.2) — the single most useful reproducibility evidence.

Every call is bounded by a timeout and runs with ``git -C <root>`` so no ``cd`` and no
shell are involved. A repository that is not committed is *not* the same as a repository
whose state we failed to read, so ``has_repo`` and ``detail`` stay separate.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from labbox.domain.environment import GitState
from labbox.infrastructure.redact import Redactor

GIT_TIMEOUT_DEFAULT = 10.0


@dataclass(slots=True)
class GitRun:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    code: str | None = None
    error: str | None = None


class GitClient:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        timeout: float = GIT_TIMEOUT_DEFAULT,
        redactor: Redactor | None = None,
    ) -> None:
        self.root = os.fspath(root)
        self.timeout = timeout
        self.redactor = redactor or Redactor()

    # --- plumbing ------------------------------------------------------------
    def _run(self, args: Sequence[str], *, stdin: str | None = None) -> GitRun:
        argv = ["git", "-C", self.root, *args]
        env = dict(os.environ)
        # Keep machine-independent output so parsing does not depend on locale.
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["LC_ALL"] = "C"
        # Never invoke credential helpers: `git status` must not pop a password dialog.
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=env,
                shell=False,
                check=False,
                input=stdin,
            )
        except subprocess.TimeoutExpired:
            return GitRun(False, code="GIT_COMMAND_TIMEOUT", error=f"git {' '.join(args)} timed out")
        except OSError as exc:
            return GitRun(False, code="GIT_UNAVAILABLE", error=str(exc))
        return GitRun(completed.returncode == 0, completed.stdout or "", completed.stderr or "")

    def available(self) -> bool:
        return self._run(["--version"]).ok

    def version(self) -> str | None:
        run = self._run(["--version"])
        if not run.ok:
            return None
        first = run.stdout.strip().splitlines()
        return first[0].replace("git version ", "") if first else None

    def in_repo(self) -> bool:
        return self._run(["rev-parse", "--git-dir"]).ok

    def commit(self) -> str | None:
        run = self._run(["rev-parse", "HEAD"])
        value = run.stdout.strip()
        return value if run.ok and value else None

    def branch(self) -> str | None:
        run = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        value = run.stdout.strip()
        if not run.ok or not value:
            return None
        return "DETACHED" if value == "HEAD" else value

    def porcelain(self) -> list[str]:
        run = self._run(["status", "--porcelain", "--untracked-files=all", "--no-renames"])
        if not run.ok:
            return []
        return [line for line in run.stdout.splitlines() if line.strip()]

    def diff_text(self, *, staged_only: bool = False) -> str:
        args = ["diff", "--no-color", "--text"]
        if staged_only:
            args += ["--cached"]
        else:
            args += ["HEAD"]
        run = self._run(args)
        return run.stdout if run.ok else ""

    def remote_origin(self) -> str | None:
        run = self._run(["config", "--get", "remote.origin.url"])
        if not run.ok:
            return None
        url = run.stdout.strip()
        return self.redactor.scrub_url(url) if url else None

    def ahead_behind(self) -> int | None:
        run = self._run(["rev-list", "--left-right", "--count", "@{upstream}...HEAD", "--no-renames"])
        if not run.ok:
            return None
        parts = run.stdout.strip().split()
        if len(parts) != 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    # --- the snapshot the domain wants ---------------------------------------
    def collect(self, *, include_diff_hash: bool = True) -> GitState:
        if not self.available():
            return GitState(has_repo=False, detail="git executable not found")
        if not self.in_repo():
            return GitState(has_repo=False, detail="not a git repository")

        commit = self.commit()
        lines = self.porcelain()
        modified = sum(1 for line in lines if line[:2].strip() not in {"??"})
        untracked = sum(1 for line in lines if line[:2].strip() == "??")
        dirty = bool(lines)
        diff_hash: str | None = None
        if include_diff_hash and dirty:
            # Untracked files are not in `git diff`, so include their paths: otherwise two
            # runs differing only by a new untracked train.py would look identical.
            payload = self.diff_text() + "\n" + "\n".join(sorted(lines))
            diff_hash = "sha256:" + hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()
        return GitState(
            has_repo=True,
            commit=commit,
            short_commit=commit[:7] if commit else None,
            branch=self.branch(),
            dirty=dirty,
            diff_hash=diff_hash,
            untracked_count=untracked,
            modified_count=modified,
            remote_origin=self.remote_origin(),
            ahead=self.ahead_behind(),
            git_version=self.version(),
            detail=None if dirty else "clean working tree",
        )
