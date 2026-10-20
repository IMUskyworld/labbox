"""Build ``environment.json`` (§8.2) out of independent, individually-failable probes.

Each collector is wrapped so its failure becomes data (``collectors[path] = status`` plus
an ``ErrorInfo``) rather than an exception. The one rule that must never bend: nothing
here may change the training environment or claim a value we did not read.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labbox.config.model import LabboxConfig
from labbox.domain.common import Collected, CollectorStatus, ErrorInfo, now_local
from labbox.domain.environment import (
    EnvironmentSnapshot,
    GitState,
    GpuInfo,
    HostInfo,
    OsInfo,
    PythonInfo,
    Seeds,
)
from labbox.infrastructure.git import GitClient
from labbox.infrastructure.nvidia import GpuBackend, NullBackend, get_backend
from labbox.infrastructure.probe import ProbeResult, ProbeTarget, probe_interpreter
from labbox.infrastructure.redact import Redactor

SEED_ARG_RE = r"--?(seed|random_seed|manual_seed|random-seed)"
DETERMINISM_ENV = (
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "NVIDIA_TF32_OVERRIDE",
    "CUDA_MODULE_LOADING",
)


class EnvironmentCollector:
    def __init__(
        self,
        *,
        project_root: Path,
        config: LabboxConfig,
        redactor: Redactor | None = None,
        gpu_backend: GpuBackend | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.config = config
        self.redactor = redactor or Redactor(config.privacy.redact_env_patterns)
        self.timeout = config.run.collect_timeout_seconds
        self._backend = gpu_backend
        self._psutil: Any = None

    @property
    def backend_used(self) -> GpuBackend:
        """The GPU backend actually picked, so the sampler can reuse the same one."""
        return self._backend if self._backend is not None else get_backend(self.config.gpu.backend)

    # --- orchestration -------------------------------------------------------
    def collect(
        self,
        *,
        phase: str = "pre_run",
        argv: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        probe_target: ProbeTarget | None = None,
        declared_datasets: Sequence[str] = (),
    ) -> EnvironmentSnapshot:
        snapshot = EnvironmentSnapshot(captured_at=now_local(), capture_phase=phase)
        run_env = dict(env if env is not None else os.environ)

        snapshot.os = self._collect_os()
        snapshot.host = self._collect_host(snapshot)
        snapshot.env = self.redactor.filter_env(
            run_env, keep_names=DETERMINISM_ENV + tuple(self.config.privacy.keep_env_names)
        )
        snapshot.seeds = self._collect_seeds(argv, run_env)

        target = probe_target or ProbeTarget(argv_prefix=[], kind="external", display="")
        probe = probe_interpreter(
            target,
            extra_packages=self._watched_packages(declared_datasets),
            timeout=self.timeout,
            env=run_env,
        )
        self._apply_probe(snapshot, target, probe)

        snapshot.git = self._collect_git(snapshot)
        self._collect_gpu(snapshot, argv)
        return snapshot

    # --- sections ------------------------------------------------------------
    def _collect_os(self) -> OsInfo:
        try:
            console_encoding = (sys.stdout.encoding or "").lower()
        except Exception:
            console_encoding = ""
        return OsInfo(
            name=platform.system(),
            version=platform.release(),
            arch=platform.machine(),
            platform=platform.platform(),
            console_encoding=console_encoding,
            fs_encoding=sys.getfilesystemencoding(),
            user=_safe_user(),
            timezone=_timezone_name(),
        )

    def _psutil_module(self) -> Any:
        if self._psutil is None:
            try:
                import psutil

                self._psutil = psutil
            except ImportError:
                self._psutil = False
        return self._psutil or None

    def _collect_host(self, snapshot: EnvironmentSnapshot) -> HostInfo:
        info = HostInfo(cpu_model=platform.processor() or platform.machine())
        psutil = self._psutil_module()
        if psutil is None:
            snapshot.collectors["host"] = CollectorStatus.UNAVAILABLE.value
            snapshot.absent.append("host.ram_bytes")
            snapshot.errors.append(
                ErrorInfo(
                    code="HOST_COLLECTOR_UNAVAILABLE",
                    message="psutil is not installed in the LabBox environment",
                    component="environment",
                ).model_dump()
            )
            info.logical_cores = os.cpu_count()
            return info
        try:
            info.physical_cores = psutil.cpu_count(logical=False)
            info.logical_cores = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            info.ram_bytes = int(memory.total)
            info.ram_available_bytes = int(memory.available)
            info.cpu_model = _cpu_model() or info.cpu_model
            if hasattr(psutil, "getloadavg"):
                try:
                    info.load_avg = [round(float(v), 2) for v in psutil.getloadavg()]
                except OSError:
                    info.load_avg = None
            snapshot.collectors["host"] = CollectorStatus.OK.value
        except Exception as exc:
            snapshot.collectors["host"] = CollectorStatus.FAILED.value
            snapshot.errors.append(ErrorInfo.from_exception(exc, component="host").model_dump())
            snapshot.absent.append("host")
        return info

    def _collect_seeds(self, argv: Sequence[str], env: Mapping[str, str]) -> Seeds:
        declared: dict[str, str] = {}
        args = [str(a) for a in argv]
        index = 0
        while index < len(args) - 1:
            token = args[index]
            if os.sep not in token and token.startswith("-"):
                name = token.lstrip("-")
                if any(pattern in name.lower() for pattern in ("seed", "deterministic")):
                    declared[name] = args[index + 1]
            index += 1
        return Seeds(
            python_hash_seed=env.get("PYTHONHASHSEED"),
            # The seed set *inside* the training script is not observable without
            # instrumenting it, and we do not instrument. Saying so is the feature.
            torch_seed=Collected(
                status=CollectorStatus.UNAVAILABLE,
                detail="requires instrumentation or a declared --seed argument",
            ),
            numpy_seed=Collected(
                status=CollectorStatus.UNAVAILABLE,
                detail="requires instrumentation or a declared --seed argument",
            ),
            cuda_deterministic=Collected(
                status=CollectorStatus.UNAVAILABLE,
                detail="torch.use_deterministic_algorithms() is not observable from outside",
            ),
            benchmark_disabled=Collected(
                status=CollectorStatus.UNAVAILABLE,
                detail="torch.backends.cudnn.benchmark is not observable from outside",
            ),
            declared_in_command=declared,
            sources=["env.PYTHONHASHSEED", "command.argv"] if declared else ["env"],
            warnings=[] if declared or env.get("PYTHONHASHSEED") else ["no seed observable"],
        )

    def _watched_packages(self, declared: Sequence[str]) -> list[str]:
        return [str(p) for p in declared if p]

    def _apply_probe(self, snapshot: EnvironmentSnapshot, target: ProbeTarget, probe: ProbeResult) -> None:
        if not target.argv_prefix and target.kind != "python":
            snapshot.python = PythonInfo(kind="external", executable=target.display)
            snapshot.collectors["python_packages"] = CollectorStatus.SKIPPED.value
            snapshot.absent.extend(["python.version", "frameworks", "packages"])
            snapshot.interpreter_probe = Collected(
                status=CollectorStatus.SKIPPED,
                detail="; ".join(target.notes) or "command is not a Python interpreter",
            )
            return
        if not probe.ok:
            snapshot.python = PythonInfo(kind="probe_target", executable=target.display)
            snapshot.collectors["python_packages"] = CollectorStatus.UNAVAILABLE.value
            snapshot.interpreter_probe = Collected(
                status=CollectorStatus.FAILED if probe.code != "INTERPRETER_PROBE_TIMEOUT" else CollectorStatus.TIMEOUT,
                error=ErrorInfo(
                    code=probe.code or "INTERPRETER_PROBE_FAILED",
                    message=(probe.error or "")[:400],
                    component="interpreter_probe",
                ),
                duration_ms=probe.duration_ms,
            )
            if snapshot.interpreter_probe.error is not None:
                snapshot.errors.append(snapshot.interpreter_probe.error.model_dump())
            snapshot.absent.extend(["python.version", "frameworks", "packages"])
            return

        data = probe.data
        snapshot.python = PythonInfo(
            version=str(data.get("python_version") or ""),
            executable=str(data.get("executable") or target.argv_prefix[0]),
            implementation=str(data.get("implementation") or "CPython"),
            kind="probe_target",
            venv_prefix=str(data.get("prefix")) if data.get("in_virtualenv") else None,
            base_executable=str(data.get("base_prefix")) if data.get("base_prefix") else None,
            probe_returncode=probe.returncode,
        )
        packages = {k: v for k, v in dict(data.get("packages") or {}).items() if v}
        frameworks = dict(data.get("frameworks") or {})
        snapshot.frameworks = {k: (v if isinstance(v, str) else None) for k, v in frameworks.items()}
        snapshot.packages = {str(k): str(v) for k, v in packages.items()}
        snapshot.package_source = "importlib.metadata (probe subprocess)"
        snapshot.interpreter_probe = Collected(status=CollectorStatus.OK, value=data, duration_ms=probe.duration_ms)
        status = CollectorStatus.OK if frameworks else CollectorStatus.PARTIAL
        snapshot.collectors["python_packages"] = status.value
        if not frameworks:
            snapshot.absent.append("frameworks")
        if not packages:
            # An empty package list means "we learned nothing", and compare must be able to
            # tell that apart from "this interpreter has no packages installed".
            snapshot.absent.append("packages")
        for key, value in dict(data.get("probe_env") or {}).items():
            snapshot.env.entries.setdefault(str(key), str(value))
        hint = data.get("torch_cuda_hint")
        if hint:
            snapshot.cuda.runtime = snapshot.cuda.runtime or str(hint)

    def _collect_git(self, snapshot: EnvironmentSnapshot) -> GitState:
        if not self.config.git.enabled:
            snapshot.collectors["git"] = CollectorStatus.DISABLED.value
            snapshot.absent.append("git")
            return GitState(has_repo=False, detail="git collection disabled in config")
        client = GitClient(self.project_root, timeout=self.config.git.timeout_seconds, redactor=self.redactor)
        try:
            state = client.collect(include_diff_hash=self.config.git.include_diff_hash)
        except Exception as exc:
            snapshot.collectors["git"] = CollectorStatus.FAILED.value
            snapshot.errors.append(ErrorInfo.from_exception(exc, component="git").model_dump())
            snapshot.absent.append("git")
            return GitState(has_repo=False, detail="git collection failed")
        if not state.has_repo:
            snapshot.collectors["git"] = CollectorStatus.UNAVAILABLE.value
            snapshot.absent.append("git.commit")
        else:
            snapshot.collectors["git"] = CollectorStatus.OK.value
        return state

    def _collect_gpu(self, snapshot: EnvironmentSnapshot, argv: Sequence[str]) -> None:
        if not self.config.gpu.enabled or "--no-gpu" in argv:
            snapshot.collectors["gpu"] = CollectorStatus.DISABLED.value
            snapshot.absent.append("gpus")
            snapshot.gpus = []
            return
        backend = self._backend if self._backend is not None else get_backend(self.config.gpu.backend)
        self._backend = backend
        if isinstance(backend, NullBackend) and not getattr(backend, "static_info", lambda: [])():
            snapshot.collectors["gpu"] = CollectorStatus.UNAVAILABLE.value
            snapshot.absent.append("gpus")
            reason = getattr(backend, "reason", None) or "no NVIDIA GPU detected"
            snapshot.errors.append(
                ErrorInfo(code="GPU_COLLECTOR_UNAVAILABLE", message=reason, component="gpu").model_dump()
            )
            return
        try:
            static = backend.static_info()
        except Exception as exc:
            snapshot.collectors["gpu"] = CollectorStatus.FAILED.value
            snapshot.errors.append(ErrorInfo.from_exception(exc, component="gpu").model_dump())
            snapshot.absent.append("gpus")
            static = []
        snapshot.gpus = [
            GpuInfo(
                index=g.index,
                name=g.name,
                uuid=g.uuid,
                vram_bytes=g.vram_bytes,
                driver=g.driver,
                compute_capability=g.compute_capability,
            )
            for g in static
        ]
        snapshot.collectors["gpu"] = CollectorStatus.OK.value if static else CollectorStatus.UNAVAILABLE.value
        if not static:
            snapshot.absent.append("gpus")

        driver = getattr(backend, "driver_version", None)
        if callable(driver):
            try:
                reported = driver()
                snapshot.cuda.driver = str(reported) if reported else (static[0].driver if static else None)
            except Exception:
                snapshot.cuda.driver = static[0].driver if static else None
        elif static:
            snapshot.cuda.driver = static[0].driver
        cuda_nvml = getattr(backend, "cuda_runtime", None)
        if callable(cuda_nvml):
            try:
                raw = int(str(cuda_nvml()))
                if raw:
                    snapshot.cuda.nvml = _format_cuda_version(raw)
            except Exception:
                pass
        if not snapshot.cuda.driver:
            snapshot.absent.append("cuda.driver")
        if not snapshot.cuda.runtime:
            snapshot.absent.append("cuda.runtime")
        snapshot.cuda.cudnn = _detect_cudnn(snapshot.packages)


def _format_cuda_version(value: int) -> str:
    return f"{value // 1000}.{(value % 1000) // 10}"


def _detect_cudnn(packages: Mapping[str, str]) -> str | None:
    for name, version in packages.items():
        lowered = name.lower()
        if "cudnn" in lowered:
            return version
    return None


def _cpu_model() -> str:
    """CPU model without spawning a process: registry on Windows, /proc/cpuinfo on Linux."""
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(value).strip()
        except OSError:
            return platform.processor() or "unknown"
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _timezone_name() -> str:
    import time

    names = time.tzname or ("", "")
    return names[1] if time.daylight and len(names) > 1 else names[0]


def _safe_user() -> str:
    return os.environ.get("USERNAME") or os.environ.get("USER") or ""


__all__ = ["EnvironmentCollector", "EnvironmentSnapshot"]
