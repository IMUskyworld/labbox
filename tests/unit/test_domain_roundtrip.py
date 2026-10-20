"""Domain schema round-trip and forward/backward compatibility (§18.1, §20 M0 gate)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from labbox.domain.artifacts import ArtifactEntry, ArtifactKind, ArtifactManifest, HashState
from labbox.domain.common import Collected, CollectorStatus, Opt, UnknownReason, now_local
from labbox.domain.dataset import DatasetEntry, DatasetManifest, DatasetStrategy, FileEntry
from labbox.domain.diagnostics import (
    CompareResult,
    Diagnostic,
    DiagnosticsFile,
    DiffFact,
    DiffKind,
    ReproCheck,
    Reproducibility,
    ReproState,
    Severity,
)
from labbox.domain.environment import (
    CudaInfo,
    EnvironmentSnapshot,
    EnvVars,
    GitState,
    GpuInfo,
    HostInfo,
    OsInfo,
    PythonInfo,
)
from labbox.domain.explain import ExplainContext, ExplainResult, Hypothesis
from labbox.domain.hyperparams import HyperValue, ValueSource
from labbox.domain.metrics import BestMetric, GpuSample, MetricDirection, MetricSummary, SamplePoint
from labbox.domain.run import CommandSpec, ExitMeaning, RunRecord, RunStatus

START = now_local()


def make_run() -> RunRecord:
    return RunRecord(
        run_id="20260922-153012-baseline",
        name="baseline",
        project="cifar100-resnet",
        status=RunStatus.SUCCEEDED,
        tags=["baseline", "amp"],
        started_at=START,
        ended_at=START + timedelta(seconds=6201.2),
        duration_seconds=6201.2,
        exit_code=0,
        exit_code_raw=0,
        exit_meaning=ExitMeaning.EXITED,
        command=CommandSpec(
            raw="python train.py --epochs 20",
            argv=["python", "train.py", "--epochs", "20"],
            cwd="C:/code/项目 with spaces",
        ),
        links={"environment": "environment.json", "report": "report.html"},
        labbox_version="0.1.0",
    )


def make_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        captured_at=START,
        os=OsInfo(name="Windows", version="11", arch="AMD64", console_encoding="gbk"),
        python=PythonInfo(version="3.11.9", executable="C:/venv/Scripts/python.exe"),
        frameworks={"torch": "2.6.0", "torchvision": "0.21.0"},
        packages={"numpy": "1.26.4"},
        cuda=CudaInfo(runtime="12.4", driver="552.22"),
        git=GitState(has_repo=True, commit="a" * 40, branch="main", dirty=True, diff_hash="sha256:ab"),
        host=HostInfo(cpu_model="Ryzen 9", physical_cores=12, logical_cores=24, ram_bytes=2**36),
        gpus=[GpuInfo(index=0, name="RTX 3090", vram_bytes=25769803776, driver="552.22")],
        env=EnvVars(entries={"PYTHONHASHSEED": "0"}, redacted_names=["AWS_SECRET_ACCESS_KEY"]),
        collectors={"git": "ok", "gpu": "ok"},
        absent=["cuda.cudnn"],
    )


MODELS: list[Any] = [
    make_run(),
    make_environment(),
    MetricSummary(
        source="metrics.jsonl",
        primary_metric="accuracy",
        direction=MetricDirection.MAXIMIZE,
        final={"accuracy": 0.8421, "loss": 0.4312},
        best={"accuracy": BestMetric(value=0.851, step=18, timestamp=START)},
        metrics_missing=False,
    ),
    MetricSummary(metrics_missing=True, final=None, warnings=["metrics_missing"]),
    DatasetManifest(
        datasets=[
            DatasetEntry(
                id="cifar100-train",
                source="data/cifar100/train",
                strategy=DatasetStrategy.FAST_MANIFEST,
                digest="sha256:91ab",
                file_count=50000,
                total_bytes=123456789,
                generated_at=START,
                exclusions=["*.tmp"],
                entries=[FileEntry(path="train/0.bin", size_bytes=3, head_digest="sha256:x")],
            )
        ]
    ),
    ArtifactManifest(
        artifacts=[
            ArtifactEntry(
                kind=ArtifactKind.CHECKPOINT,
                path="checkpoints/best.pt",
                size_bytes=987654321,
                sha256="abc123",
                hash_state=HashState.COMPUTED,
            )
        ],
        errors=[],
    ),
    DiagnosticsFile(
        generated_at=START,
        reproducibility=Reproducibility(
            band="medium",
            score=72,
            checklist=[ReproCheck(id="code_locatable", state=ReproState.PARTIAL, reason="dirty")],
        ),
        diagnostics=[
            Diagnostic(
                id="D-001",
                severity=Severity.WARNING,
                code="git_dirty",
                message="工作区不干净",
                evidence=["git.dirty=true"],
            )
        ],
    ),
    CompareResult(
        left_run_id="a",
        right_run_id="b",
        generated_at=START,
        facts=[
            DiffFact(
                id="F-001",
                path="train.batch_size",
                kind=DiffKind.NUMERIC_DELTA,
                left=64,
                right=128,
                delta=64.0,
                category="config",
                sources=["configs/base.yaml:12"],
            )
        ],
        confounders=["data + batch_size changed together"],
    ),
    ExplainContext(
        task="compare",
        run_ids=["a", "b"],
        facts=[DiffFact(id="F-001", path="x", left=1, right=2)],
        missing_fields=["cuda.cudnn"],
        allowed_fact_ids=["F-001"],
    ),
    ExplainResult(
        summary="s",
        hypotheses=[Hypothesis(text="h", confidence="low", evidence_fact_ids=["F-001"])],
        unknowns=["seed"],
    ),
    SamplePoint(
        t=START,
        elapsed_seconds=12.0,
        cpu_percent=13.5,
        ram_used_bytes=1024,
        gpus=[GpuSample(index=0, utilization_percent=99.0, memory_used_bytes=1, temperature_c=61.0)],
    ),
    HyperValue(path="train.amp", value=True, value_repr="true", source_kinds=[ValueSource.CONFIG_FILE]),
    Collected.of({"a": 1}),
    Opt.known("value"),
    Opt.unknown(UnknownReason.TIMEOUT, detail="probe exceeded 10s"),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_roundtrip_preserves_values(model: Any) -> None:
    payload = json.loads(json.dumps(model.to_json_dict(), ensure_ascii=False))
    restored = type(model).model_validate(payload)
    # Compare the serialized form: a few fields (dataset file entries) are deliberately
    # kept out of the on-disk document, so they cannot survive a round trip.
    assert restored.to_json_dict() == payload
    assert restored.schema_version == model.schema_version


def test_unknown_is_never_confused_with_a_real_value() -> None:
    known_none = Opt[int].known(None)  # type: ignore[arg-type]
    unknown = Opt[int].unknown(UnknownReason.TOOL_UNAVAILABLE)
    assert known_none.present is True and unknown.present is False
    assert unknown.reason is UnknownReason.TOOL_UNAVAILABLE
    assert str(unknown).startswith("<unknown:")


def test_collected_failure_serializes_with_status() -> None:
    failed: Collected[int] = Collected(status=CollectorStatus.TIMEOUT, detail="timeout")
    data = json.loads(json.dumps(failed.model_dump(mode="json")))
    assert data["status"] == "timeout"
    assert data["value"] is None
    assert Collected[int].model_validate(data).ok is False


def test_forward_compat_ignores_unknown_fields_and_fills_defaults() -> None:
    future = make_run().model_dump(mode="json")
    future["brand_new_field_from_labbox_2"] = {"nested": True}
    del future["tags"]
    restored = RunRecord.model_validate(future)
    assert restored.tags == []
    assert restored.name == "baseline"
    assert not hasattr(restored, "brand_new_field_from_labbox_2")


def test_environment_records_what_was_not_captured() -> None:
    snapshot = make_environment()
    assert "cuda.cudnn" in snapshot.absent
    assert snapshot.frameworks["torch"] == "2.6.0"
    assert snapshot.env.redacted_names == ["AWS_SECRET_ACCESS_KEY"]


def test_dataset_digest_definition_excludes_mtime() -> None:
    entry = DatasetEntry(id="d", source="data", generated_at=START)
    entry.entries.append(FileEntry(path="a.bin", size_bytes=10, mtime=START))
    payload = entry.model_dump(mode="json")
    assert "entries" not in payload, "file entries must not bloat dataset-manifest.json"
    assert entry.digest_inputs == "rel_path+size+head64+tail64"
    assert entry.is_content_proof is False, "fast_manifest is not a content proof"


def test_run_status_vocabulary_is_closed() -> None:
    assert {s.value for s in RunStatus} == {
        "running",
        "succeeded",
        "failed",
        "interrupted",
        "start_failed",
    }
    assert RunRecord.model_validate(make_run().model_dump(mode="json")).terminal is True
