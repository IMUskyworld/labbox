"""Structured error model.

Every failure LabBox reports carries a stable code so that terminals, ``events.jsonl``
and tests can all match on the same identifier instead of on message text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# --- error codes (stable public identifiers, do not rename) -------------------
GPU_COLLECTOR_UNAVAILABLE = "GPU_COLLECTOR_UNAVAILABLE"
GPU_SAMPLE_FAILED = "GPU_SAMPLE_FAILED"
GIT_UNAVAILABLE = "GIT_UNAVAILABLE"
GIT_NOT_A_REPOSITORY = "GIT_NOT_A_REPOSITORY"
GIT_COMMAND_TIMEOUT = "GIT_COMMAND_TIMEOUT"
INTERPRETER_PROBE_TIMEOUT = "INTERPRETER_PROBE_TIMEOUT"
INTERPRETER_PROBE_FAILED = "INTERPRETER_PROBE_FAILED"
INTERPRETER_NOT_FOUND = "INTERPRETER_NOT_FOUND"
COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
COMMAND_FAILED_TO_START = "COMMAND_FAILED_TO_START"
LOG_WRITE_FAILED = "LOG_WRITE_FAILED"
DATASET_PATH_MISSING = "DATASET_PATH_MISSING"
DATASET_HASH_TOO_LARGE = "DATASET_HASH_TOO_LARGE"
METRICS_FILE_INVALID = "METRICS_FILE_INVALID"
METRICS_MISSING = "METRICS_MISSING"
ARTIFACT_UNREADABLE = "ARTIFACT_UNREADABLE"
REPORT_GENERATION_FAILED = "REPORT_GENERATION_FAILED"
REPRODUCE_SCRIPT_WRITE_FAILED = "REPRODUCE_SCRIPT_WRITE_FAILED"
REPRODUCE_SCRIPT_RENDER_FAILED = "REPRODUCE_SCRIPT_RENDER_FAILED"
STORE_WRITE_FAILED = "STORE_WRITE_FAILED"
STORE_READ_FAILED = "STORE_READ_FAILED"
PATH_OUTSIDE_RUNS = "PATH_OUTSIDE_RUNS"
CONFIG_INVALID = "CONFIG_INVALID"
RUN_NOT_FOUND = "RUN_NOT_FOUND"
PROJECT_NOT_INITIALIZED = "PROJECT_NOT_INITIALIZED"
LLM_DISABLED = "LLM_DISABLED"
LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
LLM_OUTPUT_INVALID = "LLM_OUTPUT_INVALID"
COMPARE_INCOMPARABLE = "COMPARE_INCOMPARABLE"
UNKNOWN_ERROR = "UNKNOWN_ERROR"


class LabBoxError(Exception):
    """An error with a machine-readable code.

    ``details`` must never contain raw environment values, secrets or full file
    contents: it is rendered to terminals and embedded in reports.
    """

    code: str = UNKNOWN_ERROR
    component: str = "core"
    recoverable: bool = True

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        component: str | None = None,
        recoverable: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if component is not None:
            self.component = component
        if recoverable is not None:
            self.recoverable = recoverable
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "component": self.component,
            "recoverable": self.recoverable,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience for tracebacks
        return f"[{self.code}] {self.message}"


class NotInitializedError(LabBoxError):
    code = PROJECT_NOT_INITIALIZED
    component = "config"


class RunNotFoundError(LabBoxError):
    code = RUN_NOT_FOUND
    component = "store"


class PathEscapeError(LabBoxError):
    code = PATH_OUTSIDE_RUNS
    component = "store"
    recoverable = False


def describe(exc: BaseException, *, fallback_component: str = "core") -> dict[str, Any]:
    """Normalize any exception into the structured error shape used in JSON files."""
    if isinstance(exc, LabBoxError):
        return exc.to_dict()
    return {
        "code": getattr(exc, "code", UNKNOWN_ERROR) if exc else UNKNOWN_ERROR,
        "message": f"{type(exc).__name__}: {exc}",
        "component": fallback_component,
        "recoverable": True,
        "details": {"exception_type": type(exc).__name__},
    }
