"""``environment.json`` — everything about the machine, interpreter and repo state."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from labbox.domain.common import Collected, SchemaModel, unknown_collected


class OsInfo(SchemaModel):
    name: str = ""
    version: str = ""
    arch: str = ""
    platform: str = ""
    console_encoding: str = ""
    fs_encoding: str = ""
    user: str = ""
    timezone: str = ""


class PythonInfo(SchemaModel):
    """Which interpreter the *training* command uses.

    ``kind`` is the important flag: LabBox runs in its own environment (§14.3), so the
    training interpreter is probed by a read-only subprocess. ``kind="external"`` means
    the command was not a Python interpreter at all, so package versions are unknown.
    """

    version: str = ""
    executable: str = ""
    implementation: str = "CPython"
    kind: str = "probe_target"  # probe_target | external | labbox_own
    venv_prefix: str | None = None
    base_executable: str | None = None
    probe_returncode: int | None = None
    probe_stderr_digest: str | None = None


class CudaInfo(SchemaModel):
    runtime: str | None = None  # torch.version.cuda
    driver: str | None = None  # NVML
    cudnn: str | None = None
    nvml: str | None = None
    toolkit_env: str | None = None


class GitState(SchemaModel):
    has_repo: bool = False
    commit: str | None = None
    short_commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    diff_hash: str | None = None
    untracked_count: int | None = None
    modified_count: int | None = None
    remote_origin: str | None = None  # credentials stripped
    ahead: int | None = None
    git_version: str | None = None
    detail: str | None = None


class HostInfo(SchemaModel):
    cpu_model: str = ""
    physical_cores: int | None = None
    logical_cores: int | None = None
    ram_bytes: int | None = None
    ram_available_bytes: int | None = None
    load_avg: list[float] | None = None


class GpuInfo(SchemaModel):
    index: int
    name: str
    uuid: str | None = None
    pci_bus_id: str | None = None
    vram_bytes: int | None = None
    vram_used_bytes: int | None = None
    driver: str | None = None
    compute_capability: str | None = None
    persistence_mode: str | None = None


class EnvVars(SchemaModel):
    """Sanitized environment snapshot. Never stores unredacted values."""

    entries: dict[str, str] = Field(default_factory=dict)
    redacted_names: list[str] = Field(default_factory=list)
    redact_patterns: list[str] = Field(default_factory=list)
    total_count: int = 0

    @property
    def redacted_count(self) -> int:
        return len(self.redacted_names)


class Seeds(SchemaModel):
    """Randomness settings, each explicitly known or unknown.

    ``deterministic_algorithms`` unknown is not the same as it being false: the first is
    "we did not look", the second is a real fact the rules can flag.
    """

    python_hash_seed: str | None = None
    torch_seed: Collected[int | None] = Field(default_factory=unknown_collected)
    numpy_seed: Collected[int | None] = Field(default_factory=unknown_collected)
    cuda_deterministic: Collected[bool | None] = Field(default_factory=unknown_collected)
    benchmark_disabled: Collected[bool | None] = Field(default_factory=unknown_collected)
    declared_in_command: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EnvironmentSnapshot(SchemaModel):
    captured_at: datetime
    capture_phase: str = "pre_run"  # pre_run | post_run
    os: OsInfo = Field(default_factory=OsInfo)
    python: PythonInfo = Field(default_factory=PythonInfo)
    frameworks: dict[str, str | None] = Field(default_factory=dict)
    packages: dict[str, str] = Field(default_factory=dict)
    package_source: str | None = None
    cuda: CudaInfo = Field(default_factory=CudaInfo)
    git: GitState = Field(default_factory=GitState)
    host: HostInfo = Field(default_factory=HostInfo)
    gpus: list[GpuInfo] = Field(default_factory=list)
    env: EnvVars = Field(default_factory=EnvVars)
    seeds: Seeds = Field(default_factory=Seeds)
    interpreter_probe: Collected[dict[str, Any]] = Field(default_factory=unknown_collected)
    collectors: dict[str, str] = Field(default_factory=dict)
    absent: list[str] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)

    def mark_absent(self, path: str, reason: str = "not_captured") -> None:
        if path not in self.absent:
            self.absent.append(path)
        self.collectors[path] = reason
