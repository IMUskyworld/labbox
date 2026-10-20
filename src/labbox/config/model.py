"""``.labbox/config.toml`` plus user-level config, with CLI taking precedence (§15)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from labbox import SCHEMA_VERSION
from labbox.errors import CONFIG_INVALID, LabBoxError

DEFAULT_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "checkpoints/**/*.pt",
    "checkpoints/**/*.ckpt",
    "checkpoints/**/*.safetensors",
    "outputs/**/*.pt",
)
DEFAULT_EXCLUDES: tuple[str, ...] = ("**/__pycache__/**", "**/.git/**", "**/*.tmp")
GIB = 1024**3


class RunSection(BaseModel):
    sample_interval_seconds: float = Field(default=2.0, ge=0.1, le=600)
    store_logs: bool = True
    store_plain_logs: bool = True
    auto_degrade_interval_after_seconds: float = Field(default=1800.0, ge=0)
    auto_degrade_interval_seconds: float = Field(default=5.0, ge=0.1)
    collect_timeout_seconds: float = Field(default=10.0, ge=0.5, le=120)
    kill_grace_seconds: float = Field(default=10.0, ge=0)


class GitSection(BaseModel):
    enabled: bool = True
    include_diff_hash: bool = True
    timeout_seconds: float = Field(default=10.0, ge=0.5, le=60)


class GpuSection(BaseModel):
    enabled: bool = True
    backend: Literal["auto", "nvml", "smi", "none"] = "auto"
    metrics: list[str] = Field(default_factory=lambda: ["utilization", "memory", "temperature", "power"])
    max_consecutive_failures: int = Field(default=3, ge=1)


class DatasetsSection(BaseModel):
    strategy: Literal["fast_manifest", "full_content", "external_version", "declared"] = "fast_manifest"
    paths: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    max_files: int = Field(default=200_000, ge=1)
    time_budget_seconds: float = Field(default=120.0, ge=1)
    follow_symlinks: bool = False


class ArtifactsSection(BaseModel):
    patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_ARTIFACT_PATTERNS))
    hash_max_bytes: int = Field(default=2 * 1024**3, ge=0)
    copy_by_default: bool = False
    copy_max_bytes: int = Field(default=5 * GIB, ge=0)
    max_count: int = Field(default=500, ge=1)


class MetricsSection(BaseModel):
    files: list[str] = Field(
        default_factory=lambda: ["metrics.jsonl", "outputs/**/metrics.jsonl", "outputs/metrics.json"]
    )
    primary: str | None = None
    direction: Literal["maximize", "minimize", "unknown"] = "unknown"
    regex: dict[str, str] = Field(default_factory=dict)  # metric name -> pattern
    builtin_parsers: list[str] = Field(default_factory=lambda: ["key_value", "huggingface_trainer"])
    max_series_points: int = Field(default=20_000, ge=1)


class PrivacySection(BaseModel):
    redact_env_patterns: list[str] = Field(default_factory=list)
    store_raw_logs: bool = True
    keep_env_names: list[str] = Field(default_factory=list)


class LlmSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-model"
    api_key_env: str | None = None  # only the NAME of the variable, never the value
    send_logs: bool = False
    timeout_seconds: float = Field(default=60.0, ge=1, le=600)
    max_output_tokens: int = Field(default=1200, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)


class ConfigFileSection(BaseModel):
    patterns: list[str] = Field(
        default_factory=lambda: ["configs/**/*.yaml", "configs/**/*.yml", "configs/**/*.json", "*.yaml"]
    )
    max_file_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    extract_keys: list[str] = Field(default_factory=list)  # optional allowlist of dotted paths
    drop_key_patterns: list[str] = Field(
        default_factory=lambda: ["(?i).*password.*", "(?i).*token.*", "(?i).*secret.*", "(?i).*api_key.*"]
    )


class StoreSection(BaseModel):
    max_total_bytes: int = Field(default=10 * GIB, ge=0)
    keep_last_runs: int = Field(default=0, ge=0)
    older_than_days: int = Field(default=0, ge=0)


class LabboxConfig(BaseModel):
    """Everything LabBox reads from disk, in one validated object."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    project: str | None = None
    run: RunSection = Field(default_factory=RunSection)
    git: GitSection = Field(default_factory=GitSection)
    gpu: GpuSection = Field(default_factory=GpuSection)
    datasets: DatasetsSection = Field(default_factory=DatasetsSection)
    artifacts: ArtifactsSection = Field(default_factory=ArtifactsSection)
    metrics: MetricsSection = Field(default_factory=MetricsSection)
    privacy: PrivacySection = Field(default_factory=PrivacySection)
    llm: LlmSection = Field(default_factory=LlmSection)
    configs: ConfigFileSection = Field(default_factory=ConfigFileSection)
    store: StoreSection = Field(default_factory=StoreSection)

    source_paths: list[str] = Field(default_factory=list, exclude=True)
    problems: list[str] = Field(default_factory=list, exclude=True)

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: str) -> str:
        major = v.split(".")[0]
        if major not in {"1"}:
            raise ValueError(f"unsupported config schema_version {v!r}")
        return v

    @classmethod
    def default(cls) -> LabboxConfig:
        return cls()

    # --- loading -------------------------------------------------------------
    @classmethod
    def load(
        cls,
        project_config: Path | None = None,
        *,
        user_config: Path | None = None,
        include_user: bool = True,
    ) -> LabboxConfig:
        """Merge defaults < user config < project config."""
        data: dict[str, Any] = {}
        sources: list[str] = []
        problems: list[str] = []
        user_path = user_config if user_config is not None else user_config_path()
        candidates: list[Path] = []
        if include_user and user_path and user_path.is_file():
            candidates.append(user_path)
            sources.append(str(user_path))
        if project_config and project_config.is_file():
            candidates.append(project_config)
            sources.append(str(project_config))
        for path in candidates:
            try:
                with path.open("rb") as fh:
                    loaded = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                problems.append(f"{path}: invalid TOML ({exc})")
                continue
            except OSError as exc:
                problems.append(f"{path}: cannot read ({exc})")
                continue
            data = _deep_merge(data, loaded)
        try:
            config = cls.model_validate(data)
        except Exception as exc:
            raise LabBoxError(
                f"Invalid LabBox configuration: {exc}",
                code=CONFIG_INVALID,
                component="config",
                recoverable=False,
                details={"paths": sources},
            ) from exc
        config.source_paths = sources
        config.problems = problems
        return config

    def to_toml(self) -> str:
        import tomli_w

        # TOML has no null: a `None` means "unset", so those keys are simply omitted and
        # pick up their default on load.
        payload = _drop_nones(self.model_dump(mode="json"))
        body = tomli_w.dumps(payload)
        return "# LabBox project configuration\n# written by `labbox init`; see DEVELOPMENT.md §15\n\n" + body

    def validate_patterns(self) -> list[str]:
        """Report regexes the user got wrong instead of silently ignoring them (doctor)."""
        import re

        bad: list[str] = []
        for pattern in list(self.privacy.redact_env_patterns) + list(self.configs.drop_key_patterns):
            try:
                re.compile(pattern)
            except re.error as exc:
                bad.append(f"{pattern!r}: {exc}")
        return bad


def user_config_path() -> Path | None:
    if os.environ.get("LABBOX_USER_CONFIG"):
        return Path(os.environ["LABBOX_USER_CONFIG"])
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if not base:
        home = Path.home()
        return home / ".config" / "labbox" / "config.toml"
    return Path(base) / "labbox" / "config.toml"


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nones(v) for v in value if v is not None]
    return value


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
