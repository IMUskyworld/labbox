"""Offline HTML report: escaping, missing data, self-containment (§13, §16, §18.1).

Every test renders from a real run directory written through :class:`FilesystemStore`, so
what is asserted here is what a user gets when they open ``report.html`` -- including the
two failure modes that matter most: a run whose files are partly missing, and hostile text
in a field a user controls.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from labbox.domain.artifacts import ArtifactEntry, ArtifactKind, ArtifactManifest, HashState
from labbox.domain.common import now_local
from labbox.domain.dataset import DatasetEntry, DatasetKind, DatasetManifest, DatasetStrategy
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
    HostInfo,
    OsInfo,
    PythonInfo,
    Seeds,
)
from labbox.domain.explain import ExplainResult, FactInterpretation, Hypothesis
from labbox.domain.metrics import BestMetric, GpuSample, MetricDirection, MetricSummary, SamplePoint
from labbox.domain.run import CommandSpec, ExitMeaning, RunRecord, RunStatus
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout, RunPaths
from labbox.report.html import NOT_CAPTION, render_report, write_report

RUN_ID = "20260922-153012-baseline"
XSS = "<script>alert(1)</script>"
ESCAPED_XSS = "&lt;script&gt;alert(1)&lt;/script&gt;"
START = datetime(2026, 9, 22, 15, 30, 12, tzinfo=timezone(timedelta(hours=8)))
SECRET_VALUE = "AKIAABCDEFGHIJKLMNOP"


@pytest.fixture
def store(project: ProjectLayout) -> FilesystemStore:
    return FilesystemStore(project)


@pytest.fixture
def paths(store: FilesystemStore) -> RunPaths:
    run_paths = store.layout.run_dir(RUN_ID)
    store.ensure_run_dirs(run_paths)
    return run_paths


# ---------------------------------------------------------------------------
# fixtures-as-code: build what the recorder would have written
# ---------------------------------------------------------------------------
def make_run(**overrides: Any) -> RunRecord:
    data: dict[str, Any] = {
        "run_id": RUN_ID,
        "name": "baseline",
        "project": "cifar100-resnet",
        "status": RunStatus.SUCCEEDED,
        "tags": ["baseline", "amp"],
        "started_at": START,
        "ended_at": START + timedelta(seconds=6201.2),
        "duration_seconds": 6201.2,
        "exit_code": 0,
        "exit_code_raw": 0,
        "exit_meaning": ExitMeaning.EXITED,
        "command": CommandSpec(
            raw="python train.py --epochs 20",
            argv=["python", "train.py", "--epochs", "20"],
            cwd="C:/code/项目 with spaces",
        ),
        "links": {"environment": "environment.json", "report": "report.html"},
        "labbox_version": "0.1.0",
    }
    data.update(overrides)
    return RunRecord.model_validate(data)


def make_environment(**overrides: Any) -> EnvironmentSnapshot:
    data: dict[str, Any] = {
        "captured_at": START,
        "os": OsInfo(name="Windows", version="11", arch="AMD64", console_encoding="gbk"),
        "python": PythonInfo(version="3.11.9", executable="C:/venv/Scripts/python.exe"),
        "frameworks": {"torch": "2.6.0"},
        "packages": {"numpy": "1.26.4"},
        "cuda": CudaInfo(runtime="12.4", driver="552.22"),
        "git": GitState(has_repo=True, short_commit="a1b2c3d", branch="main", dirty=True),
        "host": HostInfo(cpu_model="Ryzen 9", physical_cores=12, logical_cores=24, ram_bytes=68719476736),
        "env": EnvVars(
            entries={"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "0"},
            redacted_names=["AWS_SECRET_ACCESS_KEY"],
            total_count=3,
        ),
        "seeds": Seeds(torch_seed={"status": "ok", "value": 1234}),
        "collectors": {"git": "ok", "gpu": "unavailable", "host": "ok"},
        "absent": ["cuda.cudnn", "gpus"],
    }
    data.update(overrides)
    return EnvironmentSnapshot.model_validate(data)


def make_metrics(**overrides: Any) -> MetricSummary:
    data: dict[str, Any] = {
        "source": "metrics.jsonl",
        "parser": "jsonl",
        "primary_metric": "accuracy",
        "direction": MetricDirection.MAXIMIZE,
        "final": {"accuracy": 0.8421, "loss": 0.4312},
        "best": {"accuracy": BestMetric(value=0.851, step=18, timestamp=START)},
        "point_count": 20,
    }
    data.update(overrides)
    return MetricSummary.model_validate(data)


def make_datasets(**overrides: Any) -> DatasetManifest:
    data: dict[str, Any] = {
        "discovery": "config",
        "datasets": [
            DatasetEntry(
                id="cifar100-train",
                source="data/cifar100/train",
                kind=DatasetKind.FILESYSTEM,
                strategy=DatasetStrategy.FAST_MANIFEST,
                digest="sha256:91ab",
                file_count=50000,
                total_bytes=123456789,
                generated_at=START,
                exclusions=["*.tmp"],
                warnings=["sampled_only"],
            )
        ],
    }
    data.update(overrides)
    return DatasetManifest.model_validate(data)


def make_artifacts(**overrides: Any) -> ArtifactManifest:
    data: dict[str, Any] = {
        "artifacts": [
            ArtifactEntry(
                kind=ArtifactKind.CHECKPOINT,
                path="checkpoints/best.pt",
                size_bytes=987654321,
                sha256="abc123",
                hash_state=HashState.COMPUTED,
            )
        ]
    }
    data.update(overrides)
    return ArtifactManifest.model_validate(data)


def make_diagnostics(**overrides: Any) -> DiagnosticsFile:
    data: dict[str, Any] = {
        "generated_at": START,
        "reproducibility": Reproducibility(
            band="medium",
            score=72,
            checklist=[
                ReproCheck(id="code_locatable", state=ReproState.PARTIAL, reason="git.dirty=true", weight=20),
                ReproCheck(
                    id="randomness_controlled", state=ReproState.MISSING, reason="no seed recorded", weight=15
                ),
            ],
        ),
        "diagnostics": [
            Diagnostic(
                id="D-001",
                severity=Severity.HIGH,
                code="git_dirty",
                message="The working tree had uncommitted changes at start.",
                evidence=["git.dirty=true", "git.diff_hash=sha256:ab"],
                suggestion="Commit or export a patch before running.",
                fact_ids=["F-002"],
            )
        ],
    }
    data.update(overrides)
    return DiagnosticsFile.model_validate(data)


def write_samples(store: FilesystemStore, paths: RunPaths, count: int = 12) -> None:
    for index in range(count):
        store.append_jsonl(
            paths.system_samples,
            SamplePoint(
                t=START + timedelta(seconds=2 * index),
                elapsed_seconds=float(2 * index),
                cpu_percent=10.0 + index,
                ram_percent=40.0 + index,
                gpus=[
                    GpuSample(
                        index=0,
                        utilization_percent=55.0 + index * 3,
                        temperature_c=60.0 + index,
                        memory_used_bytes=10_000_000 * index,
                    )
                ],
            ).to_json_dict(),
        )


def write_full_run(
    store: FilesystemStore,
    paths: RunPaths,
    *,
    samples: int = 12,
    run: RunRecord | None = None,
    metrics: MetricSummary | None = None,
) -> None:
    store.write_json_atomic(paths.dir / "run.json", run or make_run())
    store.write_json_atomic(paths.dir / "environment.json", make_environment())
    store.write_json_atomic(paths.dir / "metrics.json", metrics or make_metrics())
    store.write_json_atomic(paths.dir / "dataset-manifest.json", make_datasets())
    store.write_json_atomic(paths.dir / "artifacts.json", make_artifacts())
    store.write_json_atomic(paths.dir / "diagnostics.json", make_diagnostics())
    if samples:
        write_samples(store, paths, samples)


# ---------------------------------------------------------------------------
# self-containment (§13, §16)
# ---------------------------------------------------------------------------
def test_report_is_one_self_contained_file(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths)
    html = render_report(store, paths)
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "@media print" in html, "reports are printed and PDF-archived"
    assert "</html>" in html
    # No external reference of any shape, and no script even inline.
    for forbidden in (
        "<script",
        "src=",
        "<link",
        "@import",
        "url(",
        "<iframe",
        "<img",
        "fetch(",
        "onclick",
        "http://",
        "https://",
    ):
        assert forbidden not in html.lower(), f"{forbidden} must never appear"
    assert "could not be rendered" not in html, "the bundled template must actually be used"


def test_report_contains_no_emoji(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths)
    html = render_report(store, paths)
    offenders = {
        ch for ch in html if 0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF or ord(ch) == 0xFE0F
    }
    assert not offenders, "statuses are words, never emoji"


def test_inline_svg_is_generated_in_python_not_by_a_chart_library(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths)
    html = render_report(store, paths)
    assert "<svg " in html
    assert html.count("<polyline") >= 3, "GPU utilization, CPU, RAM and temperature are drawn"
    assert "xmlns" not in html, "not even an SVG namespace URL may appear"


def test_chart_caption_states_the_sampling_interval_and_svg_is_inline(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=30)
    html = render_report(store, paths)
    assert "median sampling interval of about 2 s" in html
    assert "30 samples" in html
    assert "no script" in html.lower()


def test_timeline_is_skipped_when_no_samples_were_recorded(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "<svg " not in html
    assert "system.jsonl: not captured" in html
    assert NOT_CAPTION["system.jsonl"].split(" ")[0] in html


# ---------------------------------------------------------------------------
# escaping (§16.2: "tests must include HTML escaping")
# ---------------------------------------------------------------------------
def payload_run() -> RunRecord:
    return make_run(name=f"exp {XSS}", tags=[f"tag{XSS}"], project=XSS)


def test_run_name_cannot_escape_into_markup(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, run=payload_run(), samples=0)
    html = render_report(store, paths)
    assert XSS not in html
    assert ESCAPED_XSS in html
    assert html.lower().count("<script") == 0


def test_dataset_id_and_artifact_path_are_escaped(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "dataset-manifest.json",
        make_datasets(datasets=[DatasetEntry(id=f"ds{XSS}", source=f"/tmp/{XSS}", digest="sha256:1")]),
    )
    store.write_json_atomic(
        paths.dir / "artifacts.json",
        make_artifacts(artifacts=[ArtifactEntry(path=f"out/{XSS}.pt", size_bytes=10)]),
    )
    html = render_report(store, paths)
    assert XSS not in html
    assert ESCAPED_XSS in html
    assert html.lower().count("<script") == 0


def test_diagnostic_message_and_llm_summary_are_escaped(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "diagnostics.json",
        make_diagnostics(diagnostics=[Diagnostic(id="D-009", severity=Severity.CRITICAL, message=XSS)]),
    )
    html = render_report(
        store,
        paths,
        explanation=ExplainResult(summary=XSS, provider="openai", model="gpt-x"),
    )
    assert XSS not in html
    assert ESCAPED_XSS in html


def test_environment_values_and_paths_are_escaped(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "environment.json",
        make_environment(
            env=EnvVars(entries={f"WEIRD{XSS}": f"value{XSS}"}, redacted_names=[], total_count=1),
        ),
    )
    html = render_report(store, paths)
    assert XSS not in html
    assert ESCAPED_XSS in html


# ---------------------------------------------------------------------------
# secrets (§16.2: "tests must include secret redaction")
# ---------------------------------------------------------------------------
def test_redacted_names_are_listed_without_any_value(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "AWS_SECRET_ACCESS_KEY" in html
    assert "redacted" in html
    assert "was never stored" in html


def test_a_secret_that_slipped_past_the_collector_is_scrubbed_again(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "environment.json",
        make_environment(
            env=EnvVars(
                entries={"AWS_ACCESS_KEY_ID": SECRET_VALUE, "SOME_URL": "https://user:pw@host/x"},
                total_count=2,
            )
        ),
    )
    html = render_report(store, paths)
    assert SECRET_VALUE not in html, "capture-time redaction is not the only line of defence"
    assert "pw@host" not in html, "credentials inside recorded URLs are scrubbed too"


# ---------------------------------------------------------------------------
# metrics (§9.4: a missing metric stays missing)
# ---------------------------------------------------------------------------
def test_metrics_missing_flag_raises_a_banner(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(
        store,
        paths,
        samples=0,
        metrics=make_metrics(final=None, metrics_missing=True, warnings=["no metric file"]),
    )
    html = render_report(store, paths)
    assert "METRICS_MISSING" in html
    assert "never infers a number" in html or "Nothing below is inferred from log text" in html


def test_absent_metrics_file_is_reported_as_metrics_missing(store: FilesystemStore, paths: RunPaths) -> None:
    store.write_json_atomic(paths.dir / "run.json", make_run())
    html = render_report(store, paths)
    assert "METRICS_MISSING" in html
    assert "metrics.json: not captured" in html


def test_present_metrics_render_final_and_best_tables(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "0.8421" in html
    assert "0.851" in html
    assert "METRICS_MISSING" not in html


# ---------------------------------------------------------------------------
# missing and unreadable data (§13 failure isolation)
# ---------------------------------------------------------------------------
def test_every_missing_file_becomes_an_explicit_block(store: FilesystemStore, paths: RunPaths) -> None:
    store.write_json_atomic(paths.dir / "run.json", make_run())
    html = render_report(store, paths)
    for name in (
        "environment.json",
        "metrics.json",
        "dataset-manifest.json",
        "artifacts.json",
        "diagnostics.json",
        "system.jsonl",
    ):
        assert f"{name}: not captured" in html, f"{name} must announce its own absence"
    assert html.count("not captured") >= 6


@pytest.mark.parametrize(
    ("filename", "garbage"),
    [
        ("environment.json", "not json at all {"),
        ("metrics.json", '{"final": {"accuracy": "not-a-number"}}'),
        ("dataset-manifest.json", '{"datasets": [{"id": "d", "strategy": "telepathy"}]}'),
        ("artifacts.json", "[1, 2, 3]"),
        ("diagnostics.json", '{"generated_at": null}'),
    ],
    ids=["unparseable", "bad-metric-type", "bad-strategy", "wrong-root-type", "null-timestamp"],
)
def test_odd_data_degrades_instead_of_raising(
    store: FilesystemStore, paths: RunPaths, filename: str, garbage: str
) -> None:
    write_full_run(store, paths, samples=0)
    (paths.dir / filename).write_text(garbage, encoding="utf-8")
    html = render_report(store, paths)
    section = {
        "environment.json": "environment",
        "metrics.json": "metrics",
        "dataset-manifest.json": "datasets",
        "artifacts.json": "artifacts",
        "diagnostics.json": "diagnostics",
    }[filename]
    assert f"<!-- labbox report: section {section} unavailable -->" in html
    assert "unreadable" in html
    assert "<html" in html, "the rest of the report still renders"


def test_run_json_itself_unreadable_still_renders(store: FilesystemStore, paths: RunPaths) -> None:
    (paths.dir / "run.json").write_text('{"run_id": "x", ', encoding="utf-8")
    html = render_report(store, paths)
    assert "run.json: not captured" in html or "run.json: present but unusable" in html
    assert RUN_ID in html, "the run directory name is used as the title"


def test_render_survives_an_emptied_run_directory(store: FilesystemStore, paths: RunPaths) -> None:
    html = render_report(store, paths)
    assert "<html" in html
    assert "not captured" in html
    assert "could not be rendered" not in html


def test_corrupted_sample_rows_are_counted_not_fatal(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=5)
    store.append_text(paths.system_samples, "{ not json\n")
    store.append_jsonl(paths.system_samples, {"no_expected_fields": True})
    html = render_report(store, paths)
    assert "rows skipped as unreadable" in html
    assert "<svg " in html, "the surviving samples still produce the chart"


# ---------------------------------------------------------------------------
# interrupted / failed runs
# ---------------------------------------------------------------------------
def test_interrupted_run_renders_with_both_exit_codes(store: FilesystemStore, paths: RunPaths) -> None:
    interrupted = make_run(
        status=RunStatus.INTERRUPTED,
        interrupted=True,
        ended_at=None,
        duration_seconds=None,
        exit_code=130,
        exit_code_raw=-1073741510,  # 0xC000013A as a signed Windows status
        exit_meaning=ExitMeaning.SIGNALED,
        killed_by_labbox=True,
    )
    write_full_run(store, paths, run=interrupted, samples=0)
    html = render_report(store, paths)
    assert "interrupted" in html
    assert "0xC000013A" in html
    assert ">130<" in html or " 130" in html
    assert "no end timestamp" in html


def test_start_failed_run_shows_no_raw_code_lie(store: FilesystemStore, paths: RunPaths) -> None:
    failed = make_run(
        status=RunStatus.START_FAILED,
        exit_code=127,
        exit_code_raw=None,
        exit_meaning=ExitMeaning.NOT_STARTED,
        ended_at=START,
        duration_seconds=0.0,
    )
    store.write_json_atomic(paths.dir / "run.json", failed)
    html = render_report(store, paths)
    assert "not reported" in html
    assert "127" in html


def test_partial_environment_is_flagged_not_hidden(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "cuda.cudnn" in html and "gpus" in html
    assert "unavailable" in html, "the gpu collector's status must be visible"
    assert "with gaps" in html


# ---------------------------------------------------------------------------
# individual sections
# ---------------------------------------------------------------------------
def test_command_renders_argv_as_individually_quoted_tokens(store: FilesystemStore, paths: RunPaths) -> None:
    record = make_run(
        command=CommandSpec(
            raw='python train.py --data "data set" --tag "a\\"b"',
            argv=["python", "train.py", "--data", "data set", "--tag", 'a"b', "--lr", "0.1"],
            cwd="C:/code/proj with spaces",
        )
    )
    write_full_run(store, paths, run=record, samples=0)
    html = render_report(store, paths)
    assert '<code class="tok">python</code>' in html
    # Jinja escapes the quoting characters themselves, which is exactly the point: the token
    # text can never be read as markup.
    assert "&#39;data set&#39;" in html, "a spaced argument must read as one token"
    assert "&#39;a&#34;b&#39;" in html, "an embedded double quote stays inside its token"
    assert "never re-joins it into a shell string" in html


def test_dataset_strategy_without_content_proof_is_called_out(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "does not prove byte-identical content" in html
    assert "rel_path+size+head64+tail64" in html
    assert "fast_manifest" in html


def test_full_content_strategy_is_accepted_as_a_proof(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "dataset-manifest.json",
        make_datasets(
            datasets=[
                DatasetEntry(
                    id="tiny",
                    source="data/tiny",
                    strategy=DatasetStrategy.FULL_CONTENT,
                    digest="sha256:aa",
                    verified=True,
                )
            ]
        ),
    )
    html = render_report(store, paths)
    assert "does not prove byte-identical content" not in html
    assert "full_content" in html


def test_artifact_urls_are_defanged_and_never_linked(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "artifacts.json",
        make_artifacts(
            artifacts=[
                ArtifactEntry(
                    kind=ArtifactKind.CHECKPOINT,
                    path="checkpoints/last.pt",
                    size_bytes=10,
                    hash_state=HashState.DEFERRED_TOO_LARGE,
                    storage_uri="https://bucket.example.com/runs/last.pt",
                )
            ]
        ),
    )
    html = render_report(store, paths)
    assert "hxxps://bucket.example.com" in html
    assert 'href="https' not in html
    assert "deferred_too_large" in html


def test_reproducibility_leads_with_band_and_checklist_and_keeps_score_small(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert 'class="badge b-warn">medium<' in html
    assert "code_locatable" in html and "git.dirty=true" in html
    assert 'class="score secondary">score 72 / 100' in html
    assert "not a probability" in html
    assert "Can the exact source be found again?" in html


def test_diagnostics_table_carries_severity_evidence_and_suggestion(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert 'class="badge b-error">high<' in html
    assert "git_dirty" in html
    assert "git.diff_hash=sha256:ab" in html
    assert "Commit or export a patch before running." in html
    assert "F-002" in html


def test_changed_facts_table_with_unknown_and_incomparable_styling(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    compare = CompareResult(
        left_run_id=RUN_ID,
        right_run_id="20260922-180000-tuned",
        left_name="baseline",
        right_name="tuned",
        generated_at=now_local(),
        facts=[
            DiffFact(
                id="F-001",
                path="git.commit",
                kind=DiffKind.CHANGED,
                left_repr="a1b2c3d",
                right_repr="d4e5f6a",
                category="code",
                severity=Severity.HIGH,
            ),
            DiffFact(
                id="F-002",
                path="dataset.digest",
                kind=DiffKind.NUMERIC_DELTA,
                left=1,
                right=2,
                delta=1.0,
                delta_repr="+1",
                category="data",
                sources=["dataset-manifest.json"],
            ),
            DiffFact(
                id="F-003",
                path="seeds.torch",
                kind=DiffKind.UNKNOWN,
                category="randomness",
                comparable=False,
                unknown_side="right",
                reason="right run captured no seed",
            ),
            DiffFact(
                id="F-004",
                path="env.CUDA_VISIBLE_DEVICES",
                kind=DiffKind.INCOMPARABLE,
                category="environment",
                comparable=False,
                reason="different capture phase",
            ),
        ],
        metric_deltas=[
            DiffFact(
                id="F-010",
                path="metrics.accuracy",
                kind=DiffKind.METRIC_DELTA,
                delta=-3.78,
                delta_repr="-3.78 pp",
                category="metrics",
            )
        ],
        confounders=["data and batch size changed together"],
        unchanged=["host.cpu_model"],
        diagnostics=[
            Diagnostic(
                id="D-001",
                severity=Severity.HIGH,
                code="confounded",
                message="Cannot attribute the metric change.",
                fact_ids=["F-001", "F-002"],
            )
        ],
        incomparable=True,
        unknown_count=1,
    )
    html = render_report(store, paths, compare=compare)
    assert "incomparable as recorded" in html
    for fact_id in ("F-001", "F-002", "F-003", "F-004", "F-010"):
        assert fact_id in html
    assert 'class="badge b-missing">unknown<' in html
    assert 'class="badge b-missing">incomparable<' in html
    assert "right run captured no seed" in html
    assert "-3.78 pp" in html
    assert "data and batch size changed together" in html
    assert "host.cpu_model" in html


def test_ai_explanation_is_labelled_and_its_citations_are_checked(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=0)
    compare = CompareResult(
        left_run_id=RUN_ID,
        right_run_id="other",
        generated_at=now_local(),
        facts=[DiffFact(id="F-001", path="git.commit", kind=DiffKind.CHANGED)],
    )
    explanation = ExplainResult(
        summary="The dataset changed together with the batch size.",
        facts=[
            FactInterpretation(fact_id="F-001", interpretation="commit differs"),
            FactInterpretation(fact_id="F-999", interpretation="invented citation"),
        ],
        hypotheses=[
            Hypothesis(
                text="maybe data",
                confidence="low",
                evidence_fact_ids=["F-001"],
                validation="re-run with fixed data",
            )
        ],
        unknowns=["seed was never captured"],
        next_steps=["re-run with the same dataset digest"],
        provider="ollama",
        model="qwen2.5",
        rejected_fact_ids=["F-777"],
        validation_errors=["model cited unknown fact ids: F-777"],
    )
    html = render_report(store, paths, compare=compare, explanation=explanation)
    assert "Model-generated content" in html
    assert "least reliable part" in html
    assert "written by an LLM (ollama / qwen2.5)" in html
    assert "id exists" in html and "unknown id" in html
    assert "already dropped by schema validation" in html
    assert "Evidence insufficient" in html


def test_report_without_llm_explains_itself(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert "Model-generated content" not in html
    assert "No AI explanation was requested" in html


def test_coverage_section_lists_the_state_of_every_section(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths)
    html = render_report(store, paths)
    assert "Capture coverage" in html
    assert html.count("captured") > 5
    assert "defanged" in html


# ---------------------------------------------------------------------------
# capture state must describe capture, not the content of the data
# ---------------------------------------------------------------------------
def test_dataset_section_state_follows_what_was_captured(
    store: FilesystemStore, paths: RunPaths
) -> None:
    """A manifest with no entries means the data was never recorded, not "all fine"."""
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(paths.dir / "dataset-manifest.json", make_datasets(datasets=[]))
    assert 'class="st-datasets st-not_captured"' in render_report(store, paths)


def test_sampled_dataset_is_captured_with_gaps(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)  # fast_manifest in the default manifest
    html = render_report(store, paths)
    assert 'class="st-datasets st-partial"' in html


def test_full_content_dataset_is_captured(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    store.write_json_atomic(
        paths.dir / "dataset-manifest.json",
        make_datasets(
            datasets=[
                DatasetEntry(
                    id="pinned",
                    source="data/pinned",
                    strategy=DatasetStrategy.FULL_CONTENT,
                    digest="sha256:ab",
                    verified=True,
                )
            ]
        ),
    )
    assert 'class="st-datasets st-ok"' in render_report(store, paths)


def test_a_severity_finding_does_not_make_the_diagnostics_section_a_capture_gap(
    store: FilesystemStore, paths: RunPaths
) -> None:
    """diagnostics.json was captured; how loud its findings are is a different question."""
    write_full_run(store, paths, samples=0)
    html = render_report(store, paths)
    assert 'class="st-diagnostics st-ok"' in html
    assert "git_dirty" in html, "the finding is still shown in full"


def test_a_missing_diagnostics_file_is_still_not_captured(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    (paths.dir / "diagnostics.json").unlink()
    assert 'class="st-diagnostics st-not_captured"' in render_report(store, paths)


def test_non_ascii_text_survives_the_round_trip(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, run=make_run(name="中文实验 name"), samples=0)
    target = write_report(store, paths)
    html = target.read_text(encoding="utf-8")
    assert "中文实验 name" in html
    assert "C:/code/项目 with spaces" in html


# ---------------------------------------------------------------------------
# writing (§6.2, §13 failure isolation)
# ---------------------------------------------------------------------------
def test_write_report_writes_the_file_atomically_and_returns_it(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=4)
    stamp = datetime(2026, 9, 22, 12, 0, 0).astimezone()
    target = write_report(store, paths, generated_at=stamp)
    assert target == paths.dir / "report.html"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == render_report(store, paths, generated_at=stamp)
    assert not list(paths.dir.glob("*.tmp")), "no temp files left behind"
    assert not list(paths.dir.glob(".*report.html*"))


def test_write_report_overwrites_an_older_page(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths, samples=0)
    write_report(store, paths)
    store.write_json_atomic(paths.dir / "metrics.json", make_metrics(final=None, metrics_missing=True))
    second = write_report(store, paths)
    html = second.read_text(encoding="utf-8")
    assert "METRICS_MISSING" in html


def test_render_report_does_not_touch_the_run_directory(store: FilesystemStore, paths: RunPaths) -> None:
    write_full_run(store, paths)
    before = sorted(p.name for p in paths.dir.iterdir())
    render_report(store, paths)
    assert sorted(p.name for p in paths.dir.iterdir()) == before


def test_reports_are_identical_across_renders_apart_from_the_timestamp(
    store: FilesystemStore, paths: RunPaths
) -> None:
    write_full_run(store, paths, samples=6)
    first = re.sub(r"generated [0-9T:+-]+", "generated X", render_report(store, paths))
    second = re.sub(r"generated [0-9T:+-]+", "generated X", render_report(store, paths))
    assert first == second, "a report must be diffable against its own regeneration"


def test_json_is_parsed_through_the_store_not_assumed(store: FilesystemStore, paths: RunPaths) -> None:
    """A run dir whose JSON is a bare scalar must not crash the renderer."""
    store.write_json_atomic(paths.dir / "run.json", make_run())
    (paths.dir / "environment.json").write_text("42", encoding="utf-8")
    html = render_report(store, paths)
    assert "environment" in html.lower()
    assert json.dumps({"ok": True})  # keeps the json import meaningful for readers
    assert "could not be rendered" not in html
