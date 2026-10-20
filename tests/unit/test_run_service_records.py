"""What the run service writes into ``run.json`` before the report is rendered.

The report is built from the files on disk, so the order in which the pipeline writes
them is part of the product's honesty, not an implementation detail: a page rendered from
the pre-training placeholder says ``status=running`` about a run that finished minutes ago.
These tests pin that ordering, plus the two record fields that had links pointing at
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.run import RunStatus
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import ProjectLayout, RunPaths, make_run_id
from labbox.recorder.process import ProcessResult
from labbox.services.run_service import (
    RunContext,
    RunRequest,
    _claim_run_dir,
    _project_root_relative,
    default_stages,
    stage_ended_environment,
    stage_links,
    stage_report,
)

WHEN_START = "2026-09-22T15:30:12+00:00"


def make_record(run_id: str, **overrides: object):
    from labbox.domain.run import CommandSpec, RunRecord

    payload = {
        "run_id": run_id,
        "name": "unit",
        "project": "proj",
        "status": RunStatus.SUCCEEDED,
        "started_at": WHEN_START,
        "ended_at": "2026-09-22T15:30:14+00:00",
        "duration_seconds": 2.0,
        "exit_code": 0,
        "exit_code_raw": 0,
        "command": CommandSpec(raw="python train.py", argv=["python", "train.py"], cwd="/tmp"),
    }
    payload.update(overrides)
    return RunRecord(**payload)  # type: ignore[arg-type]


def prepare(project: ProjectLayout, name: str = "unit") -> tuple[FilesystemStore, RunPaths, RunContext]:
    store = FilesystemStore(project)
    paths = store.layout.run_dir("20260922-153012-" + name)
    store.ensure_run_dirs(paths)
    ctx = RunContext(
        layout=store.layout,
        store=store,
        paths=paths,
        config=LabboxConfig.default(),
        request=RunRequest(argv=["python", "train.py"]),
        record=make_record(paths.dir.name),
        started_wall=0.0,
    )
    return store, paths, ctx


# ---------------------------------------------------------------------------
# the report must describe the finished run
# ---------------------------------------------------------------------------
def test_report_is_rendered_from_the_finalized_record(project: ProjectLayout) -> None:
    store, paths, ctx = prepare(project)
    store.write_json_atomic(paths.run_json, ctx.record)
    stage_report(ctx)
    html = paths.report_html.read_text(encoding="utf-8")
    assert "succeeded" in html
    assert "not captured" not in html.lower().split("duration", 1)[-1][:400]


def test_stage_report_persists_the_record_before_rendering(project: ProjectLayout) -> None:
    """Without this the page reads the ``running`` placeholder written pre-training."""
    store, paths, ctx = prepare(project)
    record = make_record(paths.dir.name, status=RunStatus.FAILED, exit_code=7, exit_code_raw=7)
    ctx.record = record
    store.write_json_atomic(paths.run_json, make_record(paths.dir.name))  # status=running
    stage_report(ctx)
    on_disk = json.loads(paths.run_json.read_text(encoding="utf-8"))
    assert on_disk["status"] == "failed", "run.json on disk must be the finished record"
    html = paths.report_html.read_text(encoding="utf-8")
    assert "failed" in html
    assert ">running<" not in html, "the report must not show the pre-training placeholder"


def test_links_run_before_the_report_so_the_page_lists_itself(project: ProjectLayout) -> None:
    stages = default_stages()
    assert stages.index(stage_links) < stages.index(stage_report)
    assert stages[-1] is stage_report, "the report is the last thing written"


# ---------------------------------------------------------------------------
# ended_environment: a link must point at a field that exists
# ---------------------------------------------------------------------------
def test_ended_environment_is_serialized_where_the_link_points(project: ProjectLayout) -> None:
    store, paths, ctx = prepare(project)
    stage_ended_environment(ctx)
    payload = ctx.record.ended_environment
    assert payload, "the ended snapshot must be attached to the record"
    assert ctx.record.links["ended_environment"] == "run.json#ended_environment"
    store.write_json_atomic(paths.run_json, ctx.record)
    reloaded = json.loads(paths.run_json.read_text(encoding="utf-8"))
    assert reloaded["ended_environment"], "the field the link names has to exist in run.json"


# ---------------------------------------------------------------------------
# run ids must not collide
# ---------------------------------------------------------------------------
def test_same_second_runs_with_one_name_do_not_overwrite(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    layout.ensure()
    first, first_paths = _claim_run_dir(layout, "20260922-153012-x")
    first_paths.dir.mkdir(parents=True)
    second, second_paths = _claim_run_dir(layout, "20260922-153012-x")
    assert first != second
    assert second_paths.dir != first_paths.dir
    assert second.startswith("20260922-153012-x-")
    assert (layout.runs_dir / second).exists() is False, "the new id is free, not created yet"


def test_free_run_id_is_used_as_is(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    layout.ensure()
    assert _claim_run_dir(layout, "20260922-153012-fresh")[0] == "20260922-153012-fresh"


def test_make_run_id_shape_is_unchanged() -> None:
    from datetime import datetime

    assert make_run_id("baseline", datetime(2026, 9, 22, 15, 30, 12)) == "20260922-153012-baseline"


# ---------------------------------------------------------------------------
# reproduce scripts must be able to find the project root
# ---------------------------------------------------------------------------
def test_project_root_relative_points_back_at_the_project(project: ProjectLayout) -> None:
    paths = project.run_dir("20260922-153012-rel")
    assert _project_root_relative(paths.dir, project.root) == "../../.."
    assert (paths.dir / "../../..").resolve() == project.root.resolve()


def test_reproduce_scripts_embed_the_real_relative_root(project: ProjectLayout) -> None:
    from labbox.domain.environment import EnvironmentSnapshot

    _, paths, ctx = prepare(project)
    ctx.environment = EnvironmentSnapshot.model_validate(
        {
            "captured_at": WHEN_START,
            "python": {"version": "3.11.9", "executable": "/usr/bin/python"},
            "git": {"has_repo": False},
        }
    )
    from labbox.services.run_service import stage_reproduce_scripts

    stage_reproduce_scripts(ctx)
    shell = paths.reproduce_sh.read_text(encoding="utf-8")
    assert "project_root_rel=../../.." in shell, "the run dir is three levels below the project root"


# ---------------------------------------------------------------------------
# the exit code must reach the record that everything downstream reads
# ---------------------------------------------------------------------------
def test_status_and_exit_code_are_set_before_the_post_run_stages(
    project: ProjectLayout,
) -> None:
    """The stage that resolves the status also has to publish it to the record."""
    from labbox.domain.run import RunRecord
    from labbox.services.run_service import _apply_process_result

    record = make_record("20260922-153012-status")
    record.status = RunStatus.RUNNING
    record.exit_code = None
    result = ProcessResult(
        exit_code=7,
        exit_code_raw=7,
        duration_seconds=1.5,
        ended_at=record.started_at,
    )
    _apply_process_result(record, result)
    assert record.status is RunStatus.FAILED
    assert record.exit_code == 7 and record.exit_code_raw == 7
    assert record.duration_seconds == 1.5
    assert RunRecord.model_validate(json.loads(json.dumps(record.model_dump(mode="json")))).exit_code == 7


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [(0, RunStatus.SUCCEEDED), (7, RunStatus.FAILED), (130, RunStatus.INTERRUPTED)],
)
def test_status_mapping(project: ProjectLayout, exit_code: int, expected: RunStatus) -> None:
    from labbox.services.run_service import _apply_process_result

    record = make_record("20260922-153012-map")
    _apply_process_result(
        record,
        ProcessResult(
            exit_code=exit_code,
            exit_code_raw=exit_code,
            interrupted=exit_code == 130,
            ended_at=record.started_at,
        ),
    )
    assert record.status is expected
