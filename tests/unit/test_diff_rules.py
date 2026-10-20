"""DiffEngine + RuleEngine + reproducibility (§10, §12). The product's core."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from labbox.analysis.diff import DiffEngine
from labbox.analysis.record import RunBundle
from labbox.analysis.rules import (
    compare_diagnostics,
    confounders,
    run_diagnostics,
    score_reproducibility,
)
from labbox.domain.diagnostics import DiffKind, ReproState, Severity
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout

START = datetime(2026, 9, 22, 15, 0, 0).astimezone()


def base_environment(**overrides: Any) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "schema_version": "1.0",
        "captured_at": START.isoformat(),
        "os": {"name": "Windows", "version": "11", "arch": "AMD64"},
        "python": {"version": "3.11.9", "executable": "C:/venv/python.exe", "kind": "probe_target"},
        "frameworks": {"torch": "2.6.0", "torchvision": "0.21.0"},
        "packages": {"numpy": "1.26.4"},
        "cuda": {"runtime": "12.4", "driver": "552.22"},
        "git": {
            "has_repo": True,
            "commit": "a" * 40,
            "short_commit": "aaaaaaa",
            "branch": "main",
            "dirty": False,
            "diff_hash": None,
        },
        "host": {"cpu_model": "Ryzen 9", "physical_cores": 12, "logical_cores": 24, "ram_bytes": 68719476736},
        "gpus": [{"index": 0, "name": "RTX 3090", "vram_bytes": 25769803776, "driver": "552.22"}],
        "env": {"entries": {"CUDA_VISIBLE_DEVICES": "0"}, "redacted_names": []},
        "seeds": {},
        "collectors": {"git": "ok", "gpu": "ok", "host": "ok", "python_packages": "ok"},
        "absent": [],
        "errors": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(environment.get(key), dict):
            environment[key] = {**environment[key], **value}
        else:
            environment[key] = value
    return environment


def base_record(run_id: str, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "name": run_id.split("-", 2)[-1],
        "project": "demo",
        "status": "succeeded",
        "tags": [],
        "started_at": START.isoformat(),
        "ended_at": (START + timedelta(seconds=100)).isoformat(),
        "duration_seconds": 100.0,
        "exit_code": 0,
        "exit_code_raw": 0,
        "exit_meaning": "exited",
        "command": {
            "raw": "python train.py --lr 0.1",
            "argv": ["python", "train.py", "--lr", "0.1"],
            "cwd": "C:/code/demo",
        },
        "links": {},
        "labbox_version": "0.1.0",
    }
    record.update(overrides)
    return record


def make_bundle(
    store: FilesystemStore,
    name: str,
    *,
    record: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = ...,  # type: ignore[assignment]
    metrics: dict[str, Any] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    config_values: dict[str, Any] | None = None,
) -> RunBundle:
    run_id = f"20260922-150000-{name}"
    paths = store.layout.run_dir(run_id)
    store.ensure_run_dirs(paths)
    store.write_json_atomic(paths.run_json, record or base_record(run_id))
    if environment is ...:
        store.write_json_atomic(paths.environment_json, base_environment())
    elif environment is not None:
        store.write_json_atomic(paths.environment_json, environment)
    if metrics is not None:
        store.write_json_atomic(paths.metrics_json, metrics)
    if datasets is not None:
        store.write_json_atomic(paths.dataset_manifest_json, {"schema_version": "1.0", "datasets": datasets})
    if artifacts is not None:
        store.write_json_atomic(paths.artifacts_json, {"schema_version": "1.0", "artifacts": artifacts})
    if config_values is not None:
        store.write_json_atomic(
            paths.config_values_json,
            {
                "schema_version": "1.0",
                "values": [
                    {"path": path, "value": value, "value_repr": str(value), "sources": ["configs/base.yaml"]}
                    for path, value in config_values.items()
                ],
            },
        )
    return RunBundle.load(store, paths.dir)


def dataset_entry(digest: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "cifar-train",
        "source": "data/train",
        "kind": "filesystem",
        "strategy": "fast_manifest",
        "digest": digest,
        "file_count": 50000,
        "total_bytes": 123456789,
        "sampled_only": True,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def store(project: ProjectLayout) -> FilesystemStore:
    return FilesystemStore(project)


def diff(store: FilesystemStore, left: RunBundle, right: RunBundle):
    result = DiffEngine(left, right).compare()
    result.diagnostics = compare_diagnostics(result, [left, right])
    result.confounders = confounders(result)
    return result


def test_identical_runs_produce_no_changed_facts(store: FilesystemStore) -> None:
    left = make_bundle(
        store, "same1", metrics={"final": {"accuracy": 0.8}}, datasets=[dataset_entry("sha256:aaa")]
    )
    right = make_bundle(
        store, "same2", metrics={"final": {"accuracy": 0.8}}, datasets=[dataset_entry("sha256:aaa")]
    )
    result = diff(store, left, right)
    assert [f.path for f in result.facts] == [], "two identical runs must produce no facts at all"
    assert result.unchanged, "equality is reported as unchanged, not as silence"
    assert "git.commit" in result.unchanged
    assert result.incomparable is False


def test_git_commit_change_is_a_high_severity_code_fact(store: FilesystemStore) -> None:
    left = make_bundle(store, "c1")
    right = make_bundle(
        store, "c2", environment=base_environment(git={"commit": "b" * 40, "short_commit": "bbbbbbb"})
    )
    result = diff(store, left, right)
    fact = next(f for f in result.facts if f.path == "git.commit")
    assert fact.kind is DiffKind.CHANGED
    assert fact.category == "code"
    assert fact.severity is Severity.HIGH
    assert fact.id.startswith("F-")
    assert any(diagnostic.code == "code_changed" for diagnostic in result.diagnostics)


def test_dataset_digest_change_makes_metric_comparison_incomparable(store: FilesystemStore) -> None:
    left = make_bundle(
        store, "d1", datasets=[dataset_entry("sha256:91ab1111aaaa")], metrics={"final": {"accuracy": 0.84}}
    )
    right = make_bundle(
        store, "d2", datasets=[dataset_entry("sha256:7c22bbbbcccc")], metrics={"final": {"accuracy": 0.80}}
    )
    result = diff(store, left, right)
    fact = next(f for f in result.facts if f.path.endswith("cifar-train.digest"))
    assert fact.kind is DiffKind.INCOMPARABLE
    assert fact.comparable is False
    assert result.incomparable is True
    diagnostic = next(d for d in result.diagnostics if d.code == "data_changed")
    assert diagnostic.severity is Severity.HIGH
    assert fact.id in diagnostic.fact_ids


def test_numeric_hyperparameter_delta_reports_the_difference(store: FilesystemStore) -> None:
    left = make_bundle(store, "h1", config_values={"train.batch_size": 64, "train.lr": 0.1})
    right = make_bundle(store, "h2", config_values={"train.batch_size": 128, "train.lr": 0.01})
    result = diff(store, left, right)
    batch = next(f for f in result.facts if f.path == "train.batch_size")
    assert batch.kind is DiffKind.NUMERIC_DELTA
    assert batch.delta == 64
    assert batch.sources
    assert all(":" in source for source in batch.sources), "sources are labelled left/right"


def test_missing_on_one_side_is_unknown_never_equal(store: FilesystemStore) -> None:
    left = make_bundle(store, "m1", datasets=[dataset_entry("sha256:abc123")])
    right = make_bundle(store, "m2", datasets=[])
    result = diff(store, left, right)
    dataset_facts = [f for f in result.facts if f.category == "data"]
    assert dataset_facts and all(f.kind is DiffKind.UNKNOWN for f in dataset_facts)
    assert dataset_facts[0].unknown_side == "right"
    assert any(d.code == "evidence_gaps" for d in result.diagnostics)


def test_uncaptured_environment_is_reported_as_unknown(store: FilesystemStore) -> None:
    left = make_bundle(store, "e1")
    right = make_bundle(store, "e2", environment=None)
    result = diff(store, left, right)
    assert any(f.kind is DiffKind.UNKNOWN and f.unknown_side == "right" for f in result.facts)
    assert result.unknown_count >= 1


def test_two_high_impact_groups_changed_cannot_be_attributed(store: FilesystemStore) -> None:
    left = make_bundle(
        store,
        "x1",
        datasets=[dataset_entry("sha256:1111aaaa")],
        config_values={"train.batch_size": 64},
        metrics={"final": {"accuracy": 0.84}},
    )
    right = make_bundle(
        store,
        "x2",
        datasets=[dataset_entry("sha256:2222bbbb")],
        config_values={"train.batch_size": 128},
        metrics={"final": {"accuracy": 0.80}},
    )
    result = diff(store, left, right)
    diagnostic = next(d for d in result.diagnostics if d.code == "multiple_variables_changed")
    assert diagnostic.severity is Severity.HIGH
    assert "cannot attribute" in diagnostic.message
    assert result.confounders, "the confounder list must be non-empty"
    assert not any("caused by" in d.message.lower() for d in result.diagnostics), "no causal claims"


def test_metric_delta_direction_is_respected(store: FilesystemStore) -> None:
    left = make_bundle(store, "g1", metrics={"final": {"loss": 0.5, "accuracy": 0.7}})
    right = make_bundle(store, "g2", metrics={"final": {"loss": 0.3, "accuracy": 0.9}})
    result = diff(store, left, right)
    loss = next(f for f in result.metric_deltas if f.path == "metrics.loss")
    accuracy = next(f for f in result.metric_deltas if f.path == "metrics.accuracy")
    assert loss.delta == pytest.approx(-0.2)
    assert "better" in (loss.delta_repr or "")
    assert "better" in (accuracy.delta_repr or "")


def test_recorded_unknown_direction_never_calls_an_improvement_worse(store: FilesystemStore) -> None:
    """A real ingest records ``direction=unknown``; accuracy rising must not read as worse.

    The name heuristic is a guess, but reversing it is a fabrication: the old code treated
    "not minimize" as "maximize the rest the other way", so every metric that grows when
    things improve came out labelled *worse*.
    """
    metrics = {"primary_metric": "accuracy", "direction": "unknown"}
    left = make_bundle(store, "u1", metrics={**metrics, "final": {"accuracy": 0.70}})
    right = make_bundle(store, "u2", metrics={**metrics, "final": {"accuracy": 0.90}})
    result = diff(store, left, right)
    accuracy = next(f for f in result.metric_deltas if f.path == "metrics.accuracy")
    assert accuracy.delta == pytest.approx(0.2)
    assert "better" in (accuracy.delta_repr or "")
    assert "worse" not in (accuracy.delta_repr or "")


def test_metric_with_no_name_signal_reports_a_bare_delta(store: FilesystemStore) -> None:
    """When nothing says which way is better, only the number is stated."""
    left = make_bundle(store, "b1", metrics={"final": {"thing": 1.0}})
    right = make_bundle(store, "b2", metrics={"final": {"thing": 4.0}})
    fact = next(f for f in diff(store, left, right).metric_deltas if f.path == "metrics.thing")
    assert fact.delta == pytest.approx(3.0)
    assert "better" not in (fact.delta_repr or "")
    assert "worse" not in (fact.delta_repr or "")
    assert fact.confidence == "unknown"
    assert "unknown" in (fact.reason or "")


def test_explicit_recorded_direction_beats_the_name_heuristic(store: FilesystemStore) -> None:
    """A name that looks like a loss but is configured to be maximized stays maximized."""
    left = make_bundle(store, "d1", metrics={"final": {"reward_loss": 0.5}})
    right = make_bundle(
        store, "d2", metrics={"primary_metric": "reward_loss", "direction": "maximize", "final": {"reward_loss": 0.9}}
    )
    fact = next(f for f in diff(store, left, right).metric_deltas if f.path == "metrics.reward_loss")
    assert "better" in (fact.delta_repr or "")


def test_nested_metric_names_use_their_own_segment(store: FilesystemStore) -> None:
    left = make_bundle(store, "n1", metrics={"final": {"train/val_accuracy": 0.5}})
    right = make_bundle(store, "n2", metrics={"final": {"train/val_accuracy": 0.8}})
    fact = next(
        f for f in diff(store, left, right).metric_deltas if f.path == "metrics.train/val_accuracy"
    )
    assert "better" in (fact.delta_repr or "")


def test_seed_undocumented_is_not_fired_when_both_runs_agree_on_a_seed(store: FilesystemStore) -> None:
    """The same seed on both sides is the case where the warning is wrong."""
    left = make_bundle(store, "sd1", config_values={"train.seed": 7})
    right = make_bundle(store, "sd2", config_values={"train.seed": 7})
    result = diff(store, left, right)
    assert "seeds.declared" in result.unchanged
    assert not any(d.code == "seed_undocumented" for d in result.diagnostics)


def test_seed_undocumented_still_fires_when_neither_run_declares_one(store: FilesystemStore) -> None:
    left = make_bundle(store, "su1", metrics={"final": {"accuracy": 0.5}})
    right = make_bundle(store, "su2", metrics={"final": {"accuracy": 0.6}})
    result = diff(store, left, right)
    assert any(d.code == "seed_undocumented" for d in result.diagnostics)


def test_seed_undocumented_fires_when_only_one_side_declares_one(store: FilesystemStore) -> None:
    left = make_bundle(store, "so1", config_values={"train.seed": 7})
    right = make_bundle(store, "so2")
    result = diff(store, left, right)
    assert any(f.path == "seeds.declared" for f in result.facts)
    assert any(d.code == "seed_undocumented" for d in result.diagnostics)


def test_one_sided_metrics_are_never_treated_as_zero(store: FilesystemStore) -> None:
    left = make_bundle(store, "p1", metrics={"final": {"accuracy": 0.8}})
    right = make_bundle(store, "p2", metrics={"metrics_missing": True, "final": None})
    result = diff(store, left, right)
    assert result.metrics_missing is True
    accuracy = next(f for f in result.metric_deltas if f.path == "metrics.accuracy")
    assert accuracy.kind is DiffKind.UNKNOWN
    assert accuracy.comparable is False
    assert any(d.code == "METRICS_MISSING" for d in result.diagnostics)


def test_reproducibility_band_and_checklist(store: FilesystemStore) -> None:
    good = make_bundle(
        store,
        "r1",
        datasets=[dataset_entry("sha256:aaa", strategy="full_content", sampled_only=False)],
        artifacts=[{"kind": "checkpoint", "path": "checkpoints/best.pt", "size_bytes": 10, "sha256": "abc"}],
        config_values={"train.seed": 1234},
    )
    score = score_reproducibility(good)
    assert score.state_of("code_locatable") is ReproState.OK
    assert score.state_of("data_verifiable") is ReproState.OK
    assert score.state_of("artifacts_locatable") is ReproState.OK
    assert score.band.value in {"medium", "high"}
    assert score.score > 55

    bad = make_bundle(
        store, "r2", environment=base_environment(git={"has_repo": False, "commit": None}), datasets=[]
    )
    low = score_reproducibility(bad)
    assert low.state_of("code_locatable") is ReproState.MISSING
    assert low.state_of("data_verifiable") is ReproState.MISSING
    assert low.band.value == "low"
    assert low.score < score.score
    assert all(check.reason for check in low.checklist), "every check must explain itself"


def test_dirty_repository_downgrades_code_locatable(store: FilesystemStore) -> None:
    bundle = make_bundle(
        store,
        "dirty",
        environment=base_environment(
            git={
                "has_repo": True,
                "commit": "c" * 40,
                "short_commit": "ccccccc",
                "dirty": True,
                "diff_hash": "sha256:xyz",
            }
        ),
    )
    score = score_reproducibility(bundle)
    assert score.state_of("code_locatable") is ReproState.PARTIAL
    assert any(d.code == "git_dirty" for d in run_diagnostics(bundle))


def test_seed_visibility_rules(store: FilesystemStore) -> None:
    without = make_bundle(store, "s1")
    with_seed = make_bundle(store, "s2", config_values={"train.seed": 7})
    assert score_reproducibility(without).state_of("randomness_controlled") is ReproState.MISSING
    assert score_reproducibility(with_seed).state_of("randomness_controlled") is ReproState.PARTIAL
    result = diff(store, without, with_seed)
    assert any(f.path == "seeds.declared" and f.kind is not DiffKind.EQUAL for f in result.facts)


def test_interrupted_and_incomplete_runs_still_diagnose(store: FilesystemStore) -> None:
    bundle = make_bundle(
        store,
        "killed",
        record=base_record(
            "20260922-150000-killed", status="running", exit_code=None, exit_meaning="unknown"
        ),
        environment=None,
    )
    diagnostics = run_diagnostics(bundle)
    assert any(d.code == "run_incomplete" for d in diagnostics)
    assert any(d.code == "environment_missing" for d in diagnostics)
    score = score_reproducibility(bundle)
    assert score.band.value == "low"


def test_failed_training_is_recorded_with_its_exit_code(store: FilesystemStore) -> None:
    bundle = make_bundle(
        store,
        "boom",
        record=base_record("20260922-150000-boom", status="failed", exit_code=1, exit_code_raw=1),
    )
    assert any(d.code == "training_failed" and "1" in d.message for d in run_diagnostics(bundle))


def test_absent_artifacts_file_is_unknown_not_a_half_answer(store: FilesystemStore) -> None:
    """No artifacts.json means the collector produced nothing, which is not "partial"."""
    bundle = make_bundle(store, "noart", artifacts=None)
    check = score_reproducibility(bundle).state_of("artifacts_locatable")
    assert check is ReproState.UNKNOWN
    assert any("artifacts.json" in item.reason for item in score_reproducibility(bundle).checklist)


def test_empty_artifacts_manifest_is_still_a_missing_finding(store: FilesystemStore) -> None:
    bundle = make_bundle(store, "emptyart", artifacts=[])
    assert score_reproducibility(bundle).state_of("artifacts_locatable") is ReproState.MISSING


def test_artifacts_without_a_hash_stay_partial(store: FilesystemStore) -> None:
    bundle = make_bundle(
        store,
        "nohash",
        artifacts=[{"kind": "checkpoint", "path": "checkpoints/best.pt", "size_bytes": 10}],
    )
    assert score_reproducibility(bundle).state_of("artifacts_locatable") is ReproState.PARTIAL


def test_compare_is_reflexive_on_facts(store: FilesystemStore) -> None:
    left = make_bundle(store, "rfx1", config_values={"train.lr": 0.1})
    right = make_bundle(store, "rfx2", config_values={"train.lr": 0.2})
    forward = diff(store, left, right)
    backward = diff(store, right, left)
    forward_lr = next(f for f in forward.facts if f.path == "train.lr")
    backward_lr = next(f for f in backward.facts if f.path == "train.lr")
    assert forward_lr.delta is not None and backward_lr.delta is not None
    assert forward_lr.delta == -backward_lr.delta
    assert (forward_lr.left, forward_lr.right) == (backward_lr.right, backward_lr.left)


def test_result_serializes_for_reports(store: FilesystemStore) -> None:
    left = make_bundle(store, "j1", datasets=[dataset_entry("sha256:aaa")])
    right = make_bundle(store, "j2", datasets=[dataset_entry("sha256:bbb")])
    result = diff(store, left, right)
    payload = json.loads(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    assert payload["facts"]
    assert payload["diagnostics"]
    restored = type(result).model_validate(payload)
    assert restored.facts[0].id in restored.diagnostics[0].fact_ids or restored.diagnostics
