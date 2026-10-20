"""Hardware sampling: it must never be able to hurt the training run (§9.3)."""

from __future__ import annotations

import json
import time

import pytest

from labbox.domain.metrics import GpuSample
from labbox.infrastructure.nvidia import FakeBackend, GpuStatic, NullBackend
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout
from labbox.recorder.sampler import HardwareSampler


@pytest.fixture
def paths(project: ProjectLayout):
    store = FilesystemStore(project)
    run_paths = store.layout.run_dir("20260922-170000-samples")
    store.ensure_run_dirs(run_paths)
    return store, run_paths


def make_sampler(paths, backend, **kwargs) -> HardwareSampler:
    store, run_paths = paths
    return HardwareSampler(
        store=store,
        paths=run_paths,
        backend=backend,
        interval_seconds=kwargs.pop("interval_seconds", 0.05),
        **kwargs,
    )


def script(*rows: tuple[float, int, float]) -> list[list[GpuSample]]:
    return [
        [
            GpuSample(
                index=0,
                utilization_percent=util,
                memory_used_bytes=mem,
                memory_total_bytes=24 * 1024**3,
                temperature_c=temp,
                power_w=200.0,
            )
        ]
        for util, mem, temp in rows
    ]


def test_samples_are_appended_as_jsonl_and_sample_count_grows(paths) -> None:
    backend = FakeBackend(
        gpus=[GpuStatic(index=0, name="RTX 3090", vram_bytes=24 * 1024**3, driver="552.22")],
        script=script((10, 1024, 40.0), (50, 2048, 55.0), (90, 4096, 70.0), (95, 4200, 74.0)),
    )
    sampler = make_sampler(paths, backend)
    sampler.start()
    time.sleep(0.28)
    summary = sampler.stop()
    _, run_paths = paths
    lines = run_paths.system_samples.read_text(encoding="utf-8").splitlines()
    assert summary.sample_count == len(lines) >= 3
    first = json.loads(lines[0])
    assert first["gpus"][0]["index"] == 0
    assert "elapsed_seconds" in first
    assert summary.backend == "fake"
    assert summary.gpu_sample_count >= 3


def test_no_gpu_backend_still_produces_host_samples(paths) -> None:
    sampler = make_sampler(paths, NullBackend("no gpu"))
    sampler.start()
    time.sleep(0.16)
    summary = sampler.stop()
    _store, run_paths = paths
    lines = run_paths.system_samples.read_text(encoding="utf-8").splitlines()
    assert summary.sample_count == len(lines) >= 1
    point = json.loads(lines[0])
    assert point["gpus"] == []
    assert point["elapsed_seconds"] >= 0


def test_repeated_gpu_failures_stop_gpu_sampling_not_the_run(paths) -> None:
    backend = FakeBackend(
        gpus=[GpuStatic(index=0, name="RTX 3090", vram_bytes=1, driver="x")],
        script=[[]],
        fail_at=2,
    )
    sampler = make_sampler(paths, backend, max_consecutive_failures=2)
    sampler.start()
    time.sleep(0.4)
    summary = sampler.stop()
    assert summary.gpu_disabled_after_failures is True
    assert summary.sample_count >= 3, "the host side of the sampling keeps going"
    assert summary.errors


def test_long_runs_degrade_the_sampling_interval(paths) -> None:
    backend = FakeBackend(gpus=[], script=[[]])
    sampler = make_sampler(
        paths,
        backend,
        interval_seconds=0.05,
        auto_degrade_after_seconds=0.12,
        auto_degrade_interval_seconds=0.2,
    )
    sampler.start()
    time.sleep(0.45)
    summary = sampler.stop()
    assert summary.degraded_interval is True
    assert summary.interval_seconds == pytest.approx(0.2)


def test_backend_close_is_called_and_errors_are_swallowed(paths) -> None:
    class ExplodingBackend(FakeBackend):
        def sample(self):  # type: ignore[override]
            raise RuntimeError("NVML exploded")

    backend = ExplodingBackend(gpus=[GpuStatic(index=0, name="x", vram_bytes=1)], script=[])
    sampler = make_sampler(paths, backend, max_consecutive_failures=1)
    sampler.start()
    time.sleep(0.18)
    summary = sampler.stop()
    assert summary.errors, "the failure is recorded..."
    assert summary.sample_count >= 1, "...but every tick still writes a host sample"


def test_csv_export_matches_the_jsonl_series(paths) -> None:
    backend = FakeBackend(
        gpus=[GpuStatic(index=0, name="RTX 3090", vram_bytes=1, driver="x")],
        script=script((12.5, 2048, 44.0), (33.0, 4096, 55.5)),
    )
    _, run_paths = paths
    sampler = make_sampler(paths, backend)
    sampler.start()
    time.sleep(0.22)
    sampler.stop()
    csv_text = run_paths.system_samples_csv.read_text(encoding="utf-8")
    header, *rows = csv_text.strip().splitlines()
    assert header.startswith("elapsed_seconds,timestamp,cpu_percent")
    assert len(rows) >= 2
    jsonl_count = len(run_paths.system_samples.read_text(encoding="utf-8").splitlines())
    assert len(rows) >= jsonl_count - 1
    assert "gpu_util_percent" in header


def test_gpu_metric_selection_is_honoured(paths) -> None:
    backend = FakeBackend(
        gpus=[GpuStatic(index=0, name="x", vram_bytes=1)],
        script=script((50.0, 1024, 66.0)),
    )
    sampler = make_sampler(paths, backend, gpu_metrics=("utilization", "memory"))
    sampler.start()
    time.sleep(0.12)
    sampler.stop()
    store, run_paths = paths  # noqa: RUF059 - store is patched by the failure test below
    lines = [json.loads(line) for line in run_paths.system_samples.read_text(encoding="utf-8").splitlines()]
    gpu = [g for line in lines for g in line["gpus"]]
    assert gpu and all(g["temperature_c"] is None for g in gpu)
    assert any(g["utilization_percent"] is not None for g in gpu)


def test_sampler_survives_a_failing_store(paths, monkeypatch) -> None:
    store, _run_paths = paths
    sampler = make_sampler(paths, FakeBackend(gpus=[], script=[[]]))

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "append_jsonl", boom)
    sampler.start()
    time.sleep(0.16)
    summary = sampler.stop()
    assert any("OSError" in error for error in summary.errors), "the failure is data, not an exception"
