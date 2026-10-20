"""The run record (``run.json``) — the index of everything else in a run directory."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from labbox.domain.common import SchemaModel


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    START_FAILED = "start_failed"


class ExitMeaning(str, Enum):
    EXITED = "exited"
    SIGNALED = "signaled"
    KILLED = "killed"
    NOT_STARTED = "not_started"
    UNKNOWN = "unknown"


class CommandSpec(SchemaModel):
    raw: str
    argv: list[str]
    cwd: str
    cwd_is_relative: bool = False
    shell_note: str = "started without a shell; argv list passed verbatim"


class RunRecord(SchemaModel):
    """One experiment run.

    ``exit_code_raw`` keeps what the OS reported, ``exit_code`` the normalized value the
    user's shell would have seen. Windows reports ``0xC000013A`` for Ctrl+C where POSIX
    reports 130; both must be visible or the "exit code is preserved" claim is false.
    """

    run_id: str
    name: str
    project: str
    status: RunStatus = RunStatus.RUNNING
    tags: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    exit_code_raw: int | None = None
    exit_meaning: ExitMeaning = ExitMeaning.UNKNOWN
    interrupted: bool = False
    forced_kill: bool = False
    killed_by_labbox: bool = False
    command: CommandSpec
    links: dict[str, str] = Field(default_factory=dict)
    #: What the machine looked like right after the child exited. ``links`` points at
    #: ``run.json#ended_environment``, so the field has to exist for that link to be
    #: truthful rather than decorative.
    ended_environment: dict[str, Any] | None = None
    labbox_version: str = ""
    labbox_version_pin: str = ""
    schema_note: str | None = None

    def compute_duration(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at).total_seconds(), 3)

    @property
    def terminal(self) -> bool:
        return self.status is not RunStatus.RUNNING
