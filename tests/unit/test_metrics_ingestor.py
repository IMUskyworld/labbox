"""Metrics ingestion: exact parsing where possible, an explicit gap where not (§9.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.metrics import MetricDirection
from labbox.recorder.metrics import MetricsIngestor


@pytest.fixture
def config() -> LabboxConfig:
    return LabboxConfig.default()


def ingest(root: Path, config: LabboxConfig, *, metric_file: str | None = None) -> object:
    return MetricsIngestor(project_root=root, config=config, metric_file=metric_file).ingest(
        root / "run", history_path=root / "run" / "logs" / "metrics-history.jsonl"
    )


def make_run_dir(root: Path, *, stdout: str = "", stderr: str = "") -> Path:
    run = root / "run" / "logs"
    run.mkdir(parents=True, exist_ok=True)
    (run / "stdout.log").write_text(stdout, encoding="utf-8")
    (run / "stdout.plain.log").write_text(stdout, encoding="utf-8")
    (run / "stderr.log").write_text(stderr, encoding="utf-8")
    return root / "run"


def test_jsonl_metrics_are_read_exactly(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    rows = [{"epoch": 1, "loss": 0.5, "accuracy": 0.6}, {"epoch": 2, "loss": 0.3, "accuracy": 0.8}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = ingest(tmp_path, config)
    assert summary.parser == "jsonl"
    assert summary.final == {"loss": 0.3, "accuracy": 0.8}
    assert summary.best["accuracy"].value == 0.8
    assert summary.best["loss"].value == 0.3, "loss is minimized by direction detection"
    assert summary.best["accuracy"].step == 2
    assert summary.point_count == 2
    assert summary.primary_metric == "accuracy"
    assert summary.metrics_missing is False


def test_history_copy_is_written_and_linked(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.jsonl").write_text('{"acc": 0.1}\n{"acc": 0.2}\n', encoding="utf-8")
    summary = ingest(tmp_path, config)
    history = tmp_path / "run" / "logs" / "metrics-history.jsonl"
    assert history.is_file()
    assert len(history.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert summary.series_ref and summary.series_ref.replace("\\", "/").endswith("metrics-history.jsonl")


def test_column_oriented_huggingface_metrics_are_supported(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    payload = {"epoch": [1.0, 2.0], "loss": [0.9, 0.4], "eval_accuracy": [0.5, 0.75]}
    (tmp_path / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    summary = ingest(tmp_path, config)
    assert summary.parser == "json"
    assert summary.final and summary.final["loss"] == 0.4
    assert summary.best["eval_accuracy"].value == 0.75
    assert summary.point_count == 2


def test_missing_metrics_are_reported_as_missing(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path, stdout="training finished\n")
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is True
    assert summary.final is None
    assert any("metrics_missing" in warning for warning in summary.warnings)
    assert summary.point_count == 0


def test_no_metric_is_invented_from_a_lone_float(tmp_path: Path, config: LabboxConfig) -> None:
    """A bare number in the log must never become "accuracy"."""
    make_run_dir(tmp_path, stdout="0.94\n")
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is True, "the builtin parser needs key=value shape"


def test_builtin_key_value_parser_is_used_as_last_resort(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(
        tmp_path,
        stdout=(
            "Epoch 1/5 - loss: 0.42 - accuracy: 0.61\n"
            "Epoch 2/5 - loss: 0.30 - accuracy: 0.78\n"
            "Training complete\n"
        ),
    )
    summary = ingest(tmp_path, config)
    assert summary.parser == "builtin:key_value"
    assert summary.final and summary.final["accuracy"] == 0.78
    assert summary.warnings and "declare --metric-file" in summary.warnings[0]


def test_user_regex_wins_over_the_builtin_parser(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path, stdout="step 1 | my_metric=0.25\nstep 2 | my_metric=0.75\n")
    config.metrics.regex = {"my_metric": r"my_metric=(?P<value>[0-9.]+)"}
    summary = ingest(tmp_path, config)
    assert summary.parser == "regex:configured"
    assert summary.final == {"my_metric": 0.75}


def test_invalid_user_regex_is_dropped_not_fatal(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path, stdout="acc=0.5\n")
    config.metrics.regex = {"broken": "(?P<value>[0-9."}
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is False or summary.parser == "builtin:key_value"


def test_explicit_metric_file_takes_priority(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.jsonl").write_text('{"acc": 0.1}\n', encoding="utf-8")
    (tmp_path / "custom.jsonl").write_text('{"acc": 0.99}\n{"acc": 1.0}\n', encoding="utf-8")
    summary = ingest(tmp_path, config, metric_file="custom.jsonl")
    assert summary.source and "custom.jsonl" in summary.source
    assert summary.final == {"acc": 1.0}


def test_corrupt_rows_are_counted_not_silently_dropped(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.jsonl").write_text('{"acc": 0.5}\nnot json\n{"acc": 0.9}\n', encoding="utf-8")
    summary = ingest(tmp_path, config)
    assert summary.final == {"acc": 0.9}
    assert summary.parse_errors and "metrics.jsonl:2" in summary.parse_errors[0]["message"]


def test_truncated_json_file_is_reported(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.json").write_text('{"acc": [0.1, ', encoding="utf-8")
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is True
    assert summary.parse_errors


def test_direction_defaults_and_primary_metric_selection(config: LabboxConfig) -> None:
    from labbox.recorder.metrics import _direction_for, _pick_primary

    assert _direction_for("val_loss", config) is MetricDirection.MINIMIZE
    assert _direction_for("accuracy", config) is MetricDirection.MAXIMIZE
    assert _pick_primary(["loss", "accuracy"]) == "accuracy"
    assert _pick_primary(["loss"]) == "loss"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def test_absolute_glob_expands_only_its_own_pattern(tmp_path: Path, config: LabboxConfig) -> None:
    """An absolute pattern must not turn into a walk of the whole directory.

    Rooting ``Path.glob`` at the magic component matched nothing at all; rooting it one
    level up and asking for ``**`` returned every file under the directory, so an
    unrelated ``logs/notes.txt`` could be parsed as if it were metrics.
    """
    make_run_dir(tmp_path)
    elsewhere = tmp_path / "outputs" / "run1"
    elsewhere.mkdir(parents=True)
    (elsewhere / "metrics.jsonl").write_text('{"acc": 0.4}\n', encoding="utf-8")
    config.metrics.files = [str(elsewhere / "**" / "metrics.jsonl")]
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is False
    assert summary.final == {"acc": 0.4}
    assert summary.source and "metrics.jsonl" in summary.source


def test_project_relative_glob_still_works(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    nested = tmp_path / "outputs" / "run1"
    nested.mkdir(parents=True)
    (nested / "metrics.jsonl").write_text('{"acc": 0.5}\n', encoding="utf-8")
    config.metrics.files = ["outputs/**/metrics.jsonl"]
    assert ingest(tmp_path, config).final == {"acc": 0.5}


def test_an_unrelated_file_named_metrics_is_not_parsed_as_metrics(
    tmp_path: Path, config: LabboxConfig
) -> None:
    make_run_dir(tmp_path)
    notes = tmp_path / "logs"
    notes.mkdir()
    (notes / "metrics.json").write_text('{"acc": 0.99, "note": ["x"]}', encoding="utf-8")
    config.metrics.files = []
    summary = ingest(tmp_path, config)
    assert summary.metrics_missing is True, "a file nobody declared is not a metrics source"


# ---------------------------------------------------------------------------
# JSON shapes
# ---------------------------------------------------------------------------
def test_a_single_row_json_is_one_row_not_a_columnar_series(
    tmp_path: Path, config: LabboxConfig
) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"loss": 0.3, "accuracy": 0.8, "steps": [1, 2, 3]}), encoding="utf-8"
    )
    summary = ingest(tmp_path, config)
    assert summary.point_count == 1
    assert summary.final == {"loss": 0.3, "accuracy": 0.8}


def test_columnar_series_is_still_transposed(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"epoch": [1.0, 2.0], "loss": [0.9, 0.4]}), encoding="utf-8"
    )
    summary = ingest(tmp_path, config)
    assert summary.point_count == 2
    assert summary.final and summary.final["loss"] == 0.4


# ---------------------------------------------------------------------------
# user regexes
# ---------------------------------------------------------------------------
def test_regex_without_a_capture_group_is_skipped_not_fatal(tmp_path: Path, config: LabboxConfig) -> None:
    """A group-less pattern used to raise inside the ingest loop and lose every metric."""
    make_run_dir(tmp_path, stdout="step 1 | my_metric=0.25\nstep 2 | my_metric=0.75\n")
    config.metrics.regex = {"my_metric": r"my_metric=[0-9.]+"}
    summary = ingest(tmp_path, config)
    assert summary.parser == "builtin:key_value"
    assert summary.final == {"my_metric": 0.75}


def test_regex_with_a_second_group_still_finds_the_number(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path, stdout="acc 0.25\nacc 0.75\n")
    config.metrics.regex = {"acc": r"(acc) ([0-9.]+)"}
    summary = ingest(tmp_path, config)
    assert summary.final == {"acc": 0.75}


def test_named_value_group_still_wins(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path, stdout="step 1 | my_metric=0.25\nstep 2 | my_metric=0.75\n")
    config.metrics.regex = {"my_metric": r"step [0-9]+ \| my_metric=(?P<value>[0-9.]+)"}
    summary = ingest(tmp_path, config)
    assert summary.parser == "regex:configured"
    assert summary.final == {"my_metric": 0.75}


# ---------------------------------------------------------------------------
# history copy
# ---------------------------------------------------------------------------
def test_series_ref_is_not_advertised_when_the_history_write_fails(
    tmp_path: Path, config: LabboxConfig
) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.jsonl").write_text('{"acc": 0.1}\n', encoding="utf-8")
    blocked = tmp_path / "blocked" / "sub"
    blocked.parent.write_text("not a directory", encoding="utf-8")
    summary = MetricsIngestor(project_root=tmp_path, config=config).ingest(
        tmp_path / "run", history_path=blocked / "metrics-history.jsonl"
    )
    assert summary.metrics_missing is False, "metrics.json itself is unaffected"
    assert summary.series_ref is None, "a link to a file that was never written is a broken link"
    assert any("history" in warning for warning in summary.warnings)


def test_cer_and_wer_are_minimized_not_maximized(tmp_path: Path, config: LabboxConfig) -> None:
    from labbox.recorder.metrics import _direction_for

    assert _direction_for("cer", config) is MetricDirection.MINIMIZE
    assert _direction_for("word_error_rate", config) is MetricDirection.MINIMIZE


def test_nested_rows_are_flattened_with_paths(tmp_path: Path, config: LabboxConfig) -> None:
    make_run_dir(tmp_path)
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"train": {"loss": 0.4, "acc": 0.7}})
        + "\n"
        + json.dumps({"train": {"loss": 0.2, "acc": 0.9}})
        + "\n",
        encoding="utf-8",
    )
    summary = ingest(tmp_path, config)
    assert summary.final == {"train.loss": 0.2, "train.acc": 0.9}
