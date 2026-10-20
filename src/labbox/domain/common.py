"""Shared building blocks for every domain model.

Two conventions matter for the comparison engine later on:

* ``Opt`` distinguishes "we did not capture this" from "the value is None/0/empty".
* ``absent`` field lists on snapshots make the same distinction serializable, so a
  missing probe result can never be compared as if both runs agreed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from labbox import SCHEMA_VERSION

T = TypeVar("T")


def now_local() -> datetime:
    """Timestamp with the machine's local UTC offset (reports show local time)."""
    return datetime.now().astimezone()


class SchemaModel(BaseModel):
    """Base for anything written to disk.

    ``extra="ignore"`` is deliberate: a newer LabBox must still be able to read an older
    run directory, and vice versa, without a hard failure. Breaking changes go through a
    schema version bump plus a migration, never through a strict parse.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_assignment=False)

    schema_version: str = Field(default=SCHEMA_VERSION)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dump (enums by value, datetimes ISO-8601 with offset, Path as str)."""
        return self.model_dump(mode="json")


class UnknownReason(str, Enum):
    NOT_CAPTURED = "not_captured"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TIMEOUT = "timeout"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_DECLARED = "not_declared"
    DISABLED = "disabled"
    PARSE_ERROR = "parse_error"


class Opt(SchemaModel, Generic[T]):
    """A value that may legitimately be unknown.

    Never use a bare ``None`` for "unknown": ``seed = None`` is itself meaningful
    information (the run did not set a seed), while ``present=False`` means we simply
    do not know what the seed was.
    """

    present: bool = True
    value: T | None = None
    reason: UnknownReason | None = None
    detail: str | None = None

    @classmethod
    def known(cls, value: T) -> Opt[T]:
        return cls(present=True, value=value)

    @classmethod
    def unknown(cls, reason: UnknownReason = UnknownReason.NOT_CAPTURED, detail: str | None = None) -> Opt[T]:
        return cls(present=False, value=None, reason=reason, detail=detail)

    @property
    def is_known(self) -> bool:
        return bool(self.present)

    def unwrap_or(self, default: T) -> T:
        return default if not self.present else self.value  # type: ignore[return-value]

    def __str__(self) -> str:  # pragma: no cover - display helper
        if not self.present:
            return f"<unknown:{self.reason or 'not_captured'}>"
        return str(self.value)


class CollectorStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


class ErrorInfo(SchemaModel):
    """The structured error shape from DEVELOPMENT.md §19."""

    code: str = "UNKNOWN_ERROR"
    message: str = ""
    component: str = "core"
    recoverable: bool = True
    details: Mapping[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException, *, component: str = "core", redact: tuple[str, ...] = ()) -> ErrorInfo:
        from labbox.errors import describe

        data = describe(exc, fallback_component=component)
        message = str(data.get("message", ""))
        for secret in redact:
            if secret:
                message = message.replace(secret, "***")
        return cls(
            code=str(data.get("code", "UNKNOWN_ERROR")),
            message=message[:500],
            component=str(data.get("component", component)),
            recoverable=bool(data.get("recoverable", True)),
            details=dict(data.get("details", {})),
        )


class Collected(SchemaModel, Generic[T]):
    """Result of one collector: a value, or an explicit failure."""

    status: CollectorStatus = CollectorStatus.OK
    value: T | None = None
    error: ErrorInfo | None = None
    duration_ms: int | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (CollectorStatus.OK, CollectorStatus.PARTIAL)

    @property
    def is_known(self) -> bool:
        return self.ok and self.value is not None

    @classmethod
    def of(cls, value: T, status: CollectorStatus = CollectorStatus.OK) -> Collected[T]:
        return cls(status=status, value=value)

    @classmethod
    def failure(cls, error: ErrorInfo, status: CollectorStatus = CollectorStatus.FAILED) -> Collected[T]:
        return cls(status=status, value=None, error=error)


def unknown_collected(reason: UnknownReason = UnknownReason.NOT_CAPTURED, detail: str | None = None) -> Collected[Any]:
    """Default factory for a ``Collected`` field we have not filled in yet."""
    status = {
        UnknownReason.TIMEOUT: CollectorStatus.TIMEOUT,
        UnknownReason.TOOL_UNAVAILABLE: CollectorStatus.UNAVAILABLE,
        UnknownReason.DISABLED: CollectorStatus.DISABLED,
        UnknownReason.NOT_APPLICABLE: CollectorStatus.SKIPPED,
    }.get(reason, CollectorStatus.UNAVAILABLE)
    return Collected(status=status, value=None, detail=detail or reason.value)


def as_dict(value: Any) -> Any:
    """Best-effort JSON-safe conversion used by the store layer."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_dict(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, Enum):
        return value.value
    return value  # type: ignore[no-any-return]
