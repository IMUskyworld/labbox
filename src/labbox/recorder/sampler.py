"""Periodic CPU/RAM/GPU sampling while training runs (§9.3).

The sampler runs in its own thread and only ever appends. If it dies, training must keep
going: every exception here is caught, counted, and reported in ``run.json`` warnings.
Two more properties worth stating:

* the interval adapts downward on long runs so a week-long job does not leave a
  multi-hundred-megabyte JSONL file behind;
* after a configurable number of consecutive GPU failures the GPU half is dropped rather
  than retried every tick, because on a broken driver the retry itself is the cost.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from labbox.domain.common import now_local
from labbox.domain.metrics import GpuSample, SamplePoint
from labbox.infrastructure.nvidia import GpuBackend, NullBackend
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import RunPaths


@dataclass(slots=True)
class SamplerSummary:
    sample_count: int = 0
    gpu_sample_count: int = 0
    interval_seconds: float = 0.0
    degraded_interval: bool = False
    gpu_disabled_after_failures: bool = False
    errors: list[str] = field(default_factory=list)
    backend: str = "none"
    duration_seconds: float = 0.0

    def to_warnings(self) -> list[str]:
        notes: list[str] = []
        if self.degraded_interval:
            notes.append(f"sampling interval degraded to {self.interval_seconds:.1f}s on a long run")
        if self.gpu_disabled_after_failures:
            notes.append("GPU sampling stopped after repeated failures")
        if self.errors:
            notes.append(f"sampler errors: {len(self.errors)} (first: {self.errors[0][:160]})")
        return notes


class HardwareSampler:
    def __init__(
        self,
        *,
        store: FilesystemStore,
        paths: RunPaths,
        backend: GpuBackend | None,
        interval_seconds: float = 2.0,
        auto_degrade_after_seconds: float = 1800.0,
        auto_degrade_interval_seconds: float = 5.0,
        max_consecutive_failures: int = 3,
        gpu_metrics: tuple[str, ...] | list[str] = ("utilization", "memory", "temperature", "power"),
    ) -> None:
        self.store = store
        self.paths = paths
        self.backend = backend or NullBackend("no gpu backend")
        self.interval = float(interval_seconds)
        self.base_interval = self.interval
        self.auto_degrade_after = float(auto_degrade_after_seconds)
        self.auto_degrade_interval = float(auto_degrade_interval_seconds)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.gpu_metrics = set(gpu_metrics)
        self.summary = SamplerSummary(interval_seconds=self.interval, backend=getattr(self.backend, "name", "none"))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures = 0
        self._gpu_enabled = not isinstance(self.backend, NullBackend)
        self._psutil: Any = None
        self._started_mono = 0.0

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_mono = time.monotonic()
        # One synchronous baseline sample: a run shorter than the sampling interval still
        # gets a recorded host state, and "no samples at all" stops being timing-dependent.
        try:
            point = self._sample_once(0.0)
            self.store.append_jsonl(self.paths.system_samples, point.to_json_dict())
            self.summary.sample_count += 1
            self.summary.gpu_sample_count += len(point.gpus)
        except Exception as exc:
            self.summary.errors.append(f"baseline sample failed: {type(exc).__name__}: {exc}"[:300])
        self._thread = threading.Thread(target=self._loop, name="labbox-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> SamplerSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():  # pragma: no cover - defensive
                self.summary.errors.append("sampler thread did not stop within timeout")
            self._thread = None
        try:
            self.backend.close()
        except Exception:  # pragma: no cover
            pass
        if self._started_mono:
            self.summary.duration_seconds = round(time.monotonic() - self._started_mono, 3)
        return self.summary

    # --- loop ----------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            tick_started = time.monotonic()
            elapsed = tick_started - self._started_mono
            if self.auto_degrade_after and elapsed > self.auto_degrade_after and self.interval == self.base_interval:
                self.interval = max(self.base_interval, self.auto_degrade_interval)
                self.summary.degraded_interval = True
                self.summary.interval_seconds = self.interval
            try:
                point = self._sample_once(elapsed)
                self.store.append_jsonl(self.paths.system_samples, point.to_json_dict())
                self.summary.sample_count += 1
                if point.gpus:
                    self.summary.gpu_sample_count += len(point.gpus)
            except Exception as exc:
                self.summary.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            overrun = self.interval - (time.monotonic() - tick_started)
            self._stop.wait(timeout=max(0.05, overrun))
        self._write_csv()

    def _psutil_module(self) -> Any:
        if self._psutil is None:
            try:
                import psutil

                self._psutil = psutil
            except ImportError:
                self._psutil = False
        return self._psutil or None

    def _sample_once(self, elapsed: float) -> SamplePoint:
        cpu: float | None = None
        ram_used: int | None = None
        ram_percent: float | None = None
        psutil = self._psutil_module()
        if psutil is not None:
            try:
                cpu = float(psutil.cpu_percent(interval=None))
            except Exception:  # pragma: no cover
                cpu = None
            try:
                memory = psutil.virtual_memory()
                ram_used = int(memory.used)
                ram_percent = float(memory.percent)
            except Exception:  # pragma: no cover
                ram_used = None
        gpus: list[GpuSample] = []
        if self._gpu_enabled:
            try:
                gpus = self._filter_metrics(self.backend.sample())
            except Exception as exc:
                self._failures += 1
                if self._failures >= self.max_consecutive_failures:
                    self._gpu_enabled = False
                    self.summary.gpu_disabled_after_failures = True
                    self.summary.errors.append(f"gpu sampling stopped after {self._failures} failures: {exc}"[:300])
            else:
                self._failures = 0
        return SamplePoint(
            t=now_local(),
            elapsed_seconds=round(elapsed, 3),
            cpu_percent=cpu,
            ram_used_bytes=ram_used,
            ram_percent=ram_percent,
            gpus=gpus,
        )

    def _filter_metrics(self, samples: list[GpuSample]) -> list[GpuSample]:
        if "power" not in self.gpu_metrics:
            for sample in samples:
                sample.power_w = None
        if "temperature" not in self.gpu_metrics:
            for sample in samples:
                sample.temperature_c = None
        return samples

    def _write_csv(self) -> None:
        """Flat CSV next to the JSONL so plots and spreadsheets can read it (§22.2)."""
        try:
            rows = _read_points(self.store, self.paths)
        except Exception as exc:  # pragma: no cover - reporting only
            self.summary.errors.append(f"csv export failed: {exc}"[:200])
            return
        header = [
            "elapsed_seconds",
            "timestamp",
            "cpu_percent",
            "ram_used_bytes",
            "gpu_index",
            "gpu_util_percent",
            "gpu_mem_used_bytes",
            "gpu_mem_total_bytes",
            "gpu_temp_c",
            "gpu_power_w",
        ]
        lines = [",".join(header)]
        for row in rows:
            gpus = row.get("gpus") or [{}]
            for gpu in gpus:
                lines.append(
                    ",".join(
                        _csv_cell(value)
                        for value in (
                            row.get("elapsed_seconds"),
                            row.get("t"),
                            row.get("cpu_percent"),
                            row.get("ram_used_bytes"),
                            gpu.get("index"),
                            gpu.get("utilization_percent"),
                            gpu.get("memory_used_bytes"),
                            gpu.get("memory_total_bytes"),
                            gpu.get("temperature_c"),
                            gpu.get("power_w"),
                        )
                    )
                )
        try:
            self.paths.system_samples_csv.parent.mkdir(parents=True, exist_ok=True)
            self.paths.system_samples_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            self.summary.errors.append(f"csv write failed: {exc}"[:200])


def _read_points(store: FilesystemStore, paths: RunPaths) -> list[dict[str, Any]]:
    import json

    path = paths.system_samples
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text
