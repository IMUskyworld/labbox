"""GPU information and sampling, behind one protocol (§9.3).

Three implementations so that "no GPU, no driver, no NVML, no ``nvidia-smi``" is a normal
code path rather than an error: ``NvmlBackend``, ``SmiBackend`` and ``NullBackend``. Tests
inject ``FakeBackend`` so the sampler is covered on machines without an NVIDIA card.

NVML is queried directly rather than through torch: ``torch.cuda.is_available()`` would
initialize a CUDA context and could take a few hundred MB of VRAM out of the run we are
supposed to leave untouched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from labbox.domain.metrics import GpuSample
from labbox.errors import GPU_COLLECTOR_UNAVAILABLE, GPU_SAMPLE_FAILED

_SMI_FIELDS = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw")


@dataclass(slots=True)
class GpuStatic:
    index: int
    name: str
    uuid: str | None = None
    vram_bytes: int | None = None
    driver: str | None = None
    compute_capability: str | None = None


class GpuBackend(Protocol):
    name: str

    def open(self) -> tuple[bool, str | None]: ...
    def static_info(self) -> list[GpuStatic]: ...
    def sample(self) -> list[GpuSample]: ...
    def close(self) -> None: ...


class NullBackend:
    name = "none"

    def __init__(self, reason: str = "gpu disabled") -> None:
        self.reason = reason

    def open(self) -> tuple[bool, str | None]:
        return False, self.reason

    def static_info(self) -> list[GpuStatic]:
        return []

    def sample(self) -> list[GpuSample]:
        return []

    def close(self) -> None:
        return None


class NvmlBackend:
    name = "nvml"

    def __init__(self) -> None:
        self._nvml: Any = None
        self._handles: list[Any] = []
        self._count = 0
        self._reason: str | None = None

    def open(self) -> tuple[bool, str | None]:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:
            self._reason = "nvidia-ml-py is not installed in the LabBox environment"
            return False, self._reason
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # pynvml raises NVMLError, type not import-safe here
            self._reason = f"NVML init failed: {exc}"
            return False, self._reason
        self._nvml = pynvml
        try:
            self._count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._count)]
        except Exception as exc:
            self._reason = f"NVML device enumeration failed: {exc}"
            return False, self._reason
        return True, None

    def driver_version(self) -> str | None:
        if self._nvml is None:
            return None
        try:
            return str(self._nvml.nvmlSystemGetDriverVersion())
        except Exception:
            return None

    def cuda_runtime(self) -> int | None:
        if self._nvml is None:
            return None
        try:
            return int(self._nvml.nvmlSystemGetCudaDriverVersion_v2())
        except Exception:
            try:
                return int(self._nvml.nvmlSystemGetCudaDriverVersion())
            except Exception:
                return None

    def static_info(self) -> list[GpuStatic]:
        if self._nvml is None:
            return []
        out: list[GpuStatic] = []
        driver = self.driver_version()
        for index, handle in enumerate(self._handles):
            try:
                name = _decode(self._nvml.nvmlDeviceGetName(handle))
                total = int(self._nvml.nvmlDeviceGetMemoryInfo(handle).total)
                uuid = _decode(self._nvml.nvmlDeviceGetUUID(handle))
            except Exception:
                continue
            out.append(
                GpuStatic(
                    index=index,
                    name=name,
                    uuid=uuid,
                    vram_bytes=total,
                    driver=driver,
                    compute_capability=_compute_capability(self._nvml, handle),
                )
            )
        return out

    def sample(self) -> list[GpuSample]:
        if self._nvml is None:
            return []
        nvml = self._nvml
        samples: list[GpuSample] = []
        for index, handle in enumerate(self._handles):
            errors: list[str] = []
            util = _nvml_read("utilization", _utilization, handle, errors, nvml=nvml)
            memory = _nvml_read("memory", _memory_info, handle, errors, nvml=nvml)
            temp = _nvml_read("temperature", _temperature, handle, errors, nvml=nvml)
            power = _nvml_read("power", _power_milliwatts, handle, errors, nvml=nvml)
            fan = _nvml_read("fan", _fan_speed, handle, errors, nvml=nvml)
            samples.append(
                GpuSample(
                    index=index,
                    utilization_percent=util,
                    memory_used_bytes=_as_int(getattr(memory, "used", None)),
                    memory_total_bytes=_as_int(getattr(memory, "total", None)),
                    temperature_c=temp,
                    power_w=power,
                    fan_percent=fan,
                )
            )
        return samples

    def close(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml = None


class SmiBackend:
    """Fallback that shells out to ``nvidia-smi``. Slower, but needs no extra package."""

    name = "nvidia-smi"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("nvidia-smi")
        self._reason: str | None = None

    def open(self) -> tuple[bool, str | None]:
        if not self.executable:
            self._reason = "nvidia-smi not found on PATH"
            return False, self._reason
        ok, _out = self._call(["--query-gpu=name,memory.total,uuid", "--format=csv,noheader,nounits"])
        if not ok:
            self._reason = "nvidia-smi did not return data"
            return False, self._reason
        return True, None

    def _call(self, args: Sequence[str], timeout: float = 5.0) -> tuple[bool, str]:
        if not self.executable:
            return False, ""
        try:
            completed = subprocess.run(
                [self.executable, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, ""
        if completed.returncode != 0:
            return False, ""
        return True, completed.stdout or ""

    def static_info(self) -> list[GpuStatic]:
        ok, out = self._call(["--query-gpu=name,memory.total,uuid,driver_version", "--format=csv,noheader,MiB"])
        if not ok:
            return []
        result: list[GpuStatic] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0]:
                continue
            memory = _to_int(parts[1])
            result.append(
                GpuStatic(
                    index=len(result),
                    name=parts[0],
                    uuid=parts[2] if len(parts) > 2 and parts[2] not in {"N/A", ""} else None,
                    vram_bytes=(memory * 1024 * 1024) if memory else None,
                    driver=parts[3] if len(parts) > 3 else None,
                )
            )
        return result

    def sample(self) -> list[GpuSample]:
        ok, out = self._call(
            [
                "--query-gpu=index," + ",".join(_SMI_FIELDS),
                "--format=csv,noheader,nounits",
            ]
        )
        if not ok:
            return []
        samples: list[GpuSample] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            index = _to_int(parts[0])
            if index is None:
                continue
            samples.append(
                GpuSample(
                    index=index,
                    utilization_percent=_to_float(parts[1]) if len(parts) > 1 else None,
                    memory_used_bytes=_to_int(parts[2]) if len(parts) > 2 else None,
                    memory_total_bytes=_mib_to_bytes(_to_int(parts[3])) if len(parts) > 3 else None,
                    temperature_c=_to_float(parts[4]) if len(parts) > 4 else None,
                    power_w=_to_float(parts[5]) if len(parts) > 5 else None,
                )
            )
        return samples

    def close(self) -> None:
        return None


@dataclass(slots=True)
class FakeBackend:
    """Deterministic backend for tests and for machines without an NVIDIA GPU."""

    name: str = "fake"
    gpus: list[GpuStatic] | None = None
    script: list[list[GpuSample]] | None = None
    fail_at: int | None = None
    calls: int = 0
    available: bool = True

    def open(self) -> tuple[bool, str | None]:
        if not self.available:
            return False, "fake backend marked unavailable"
        return True, None

    def static_info(self) -> list[GpuStatic]:
        return list(self.gpus or [])

    def sample(self) -> list[GpuSample]:
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("fake gpu failure")
        if not self.script:
            return []
        return list(self.script[min(self.calls - 1, len(self.script) - 1)])

    def close(self) -> None:
        return None


def get_backend(prefer: str = "auto") -> GpuBackend:
    """Pick a working backend, falling back to ``NullBackend`` without raising."""
    order: list[GpuBackend]
    if prefer == "nvml":
        order = [NvmlBackend()]
    elif prefer == "smi":
        order = [SmiBackend()]
    elif prefer == "none":
        return NullBackend("gpu sampling disabled")
    else:
        order = [NvmlBackend(), SmiBackend()]
    for backend in order:
        ok, _reason = backend.open()
        if ok and backend.static_info():
            return backend
        backend.close()
    return NullBackend("no NVIDIA GPU backend available")


def _utilization(nvml: Any, handle: Any) -> float:
    return float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)


def _memory_info(nvml: Any, handle: Any) -> Any:
    return nvml.nvmlDeviceGetMemoryInfo(handle)


def _temperature(nvml: Any, handle: Any) -> float:
    return float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))


def _power_milliwatts(nvml: Any, handle: Any) -> float:
    return float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0


def _fan_speed(nvml: Any, handle: Any) -> float:
    return float(nvml.nvmlDeviceGetFanSpeed(handle))


def _nvml_read(field: str, reader: Any, handle: Any, errors: list[str], *, nvml: Any) -> Any:
    """One NVML read; an unsupported field is recorded and reported as ``None``."""
    try:
        return reader(nvml, handle)
    except Exception as exc:
        errors.append(f"{field}: {exc}")
        return None


def _try(func: Any, convert: Any, errors: list[str]) -> Any:
    """Run one NVML call, recording — never propagating — its failure."""
    try:
        return convert(func())
    except Exception as exc:  # NVML raises for unsupported fields on many consumer cards
        errors.append(str(exc))
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _compute_capability(nvml: Any, handle: Any) -> str | None:
    try:
        major = nvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
        minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
        return f"{major}.{minor}"
    except Exception:
        return None


def _mib_to_bytes(mib: int | None) -> int | None:
    return mib * 1024 * 1024 if mib else None


def _to_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text or "")
    return int(match.group()) if match else None


def _to_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not match:
        return None
    if "[n/a]" in (text or "").lower():
        return None
    return float(match.group())


__all__ = [
    "GPU_COLLECTOR_UNAVAILABLE",
    "GPU_SAMPLE_FAILED",
    "FakeBackend",
    "GpuBackend",
    "GpuStatic",
    "NullBackend",
    "NvmlBackend",
    "SmiBackend",
    "get_backend",
]
