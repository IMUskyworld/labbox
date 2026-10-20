"""M3 acceptance: two real runs, then `labbox compare` explains the difference (§18.3)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
CONSOLE_SCRIPT = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
DEV_SHIM = REPO / "scripts" / "labbox_dev.py"


def _cli() -> list[str]:
    if CONSOLE_SCRIPT.is_file():
        probe = subprocess.run(
            [str(CONSOLE_SCRIPT), "init", "--help"],
            capture_output=True,
            shell=False,
            check=False,
            cwd=str(REPO),
        )
        if probe.returncode == 0:
            return [str(CONSOLE_SCRIPT)]
    return [sys.executable, str(DEV_SHIM)]


CLI = _cli()


def invoke(*args: str, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [*CLI, *args], cwd=str(cwd), env=clean, capture_output=True, shell=False, check=False
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    shutil.copy(EXAMPLES / "train_fake.py", root / "train.py")
    data = root / "data" / "train"
    data.mkdir(parents=True)
    for index in range(4):
        (data / f"shard_{index}.bin").write_bytes(bytes([index]) * 400)
    assert invoke("init", cwd=root).returncode == 0
    return root


def run(project: Path, name: str, *args: str) -> None:
    result = invoke(
        "run", "--name", name, "--dataset", "data", "--", sys.executable, "train.py", *args, cwd=project
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def compare_json(project: Path, left: str, right: str) -> dict:
    result = invoke("compare", left, right, "--json", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8"))


def test_compare_lists_the_single_changed_hyperparameter(project: Path) -> None:
    run(project, "bs64", "--epochs", "1", "--batch-size", "64", "--out", "o-a")
    run(project, "bs128", "--epochs", "1", "--batch-size", "128", "--out", "o-b")
    payload = compare_json(project, "bs64", "bs128")
    paths = {fact["path"] for fact in payload["facts"]}
    assert "train.batch_size" in paths
    fact = next(f for f in payload["facts"] if f["path"] == "train.batch_size")
    assert fact["kind"] == "numeric_delta"
    assert fact["left"] == 64 and fact["right"] == 128
    assert fact["delta"] == 64
    assert fact["id"].startswith("F-")
    diagnostics = {d["code"] for d in payload["diagnostics"]}
    assert "hyperparams_changed" in diagnostics


def test_compare_text_output_is_human_readable(project: Path) -> None:
    run(project, "lr1", "--epochs", "1", "--lr", "0.1", "--out", "o-c")
    run(project, "lr2", "--epochs", "1", "--lr", "0.01", "--out", "o-d")
    result = invoke("compare", "lr1", "lr2", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, text
    assert "Changed facts" in text
    assert "[F-" in text
    assert "Diagnostics" in text and "Metrics" in text
    assert "train.lr" in text
    assert "Comparing lr1" in text


def test_identical_metrics_are_not_reported_as_missing(project: Path) -> None:
    run(project, "same-a", "--epochs", "1", "--out", "o-i")
    run(project, "same-b", "--epochs", "1", "--out", "o-j")
    result = invoke("compare", "same-a", "same-b", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, text
    assert "metrics are identical in both runs" in text
    assert "no metrics recorded" not in text


def test_changed_dataset_is_flagged_as_incomparable(project: Path) -> None:
    run(project, "before", "--epochs", "1", "--out", "o-e")
    data = project / "data" / "train"
    (data / "shard_9.bin").write_bytes(b"new data" * 100)
    run(project, "after", "--epochs", "1", "--out", "o-f")
    payload = compare_json(project, "before", "after")
    data_facts = [f for f in payload["facts"] if f["category"] == "data"]
    assert data_facts, payload["facts"]
    assert any(fact["kind"] == "incomparable" for fact in data_facts)
    assert payload["incomparable"] is True
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "data_changed" in codes


def test_multiple_changes_are_reported_as_unattributable(project: Path) -> None:
    run(project, "combo-a", "--epochs", "1", "--batch-size", "64", "--out", "o-g")
    (project / "data" / "train" / "extra.bin").write_bytes(b"z" * 500)
    run(project, "combo-b", "--epochs", "1", "--batch-size", "128", "--amp", "false", "--out", "o-h")
    payload = compare_json(project, "combo-a", "combo-b")
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "multiple_variables_changed" in codes
    confounders = payload["confounders"]
    assert confounders, "the confounder list must not be empty"
    text = json.dumps(payload, ensure_ascii=False)
    assert "caused by" not in text.lower(), "compare must never assert causation"


def test_compare_reports_equality_honestly(project: Path) -> None:
    run(project, "same-a", "--epochs", "1", "--seed", "7", "--out", "same-out")
    run(project, "same-b", "--epochs", "1", "--seed", "7", "--out", "same-out")
    payload = compare_json(project, "same-a", "same-b")
    categories = {f["category"] for f in payload["facts"]}
    assert "config" not in categories, "identical hyperparameters produce no config facts"
    assert "data" not in categories, "an unchanged dataset must not be reported as a change"
    assert payload["unchanged"], "equal values are reported as unchanged"
    unchanged = set(payload["unchanged"])
    assert {"train.epochs", "train.seed", "train.out"} <= unchanged, unchanged
    assert not [f for f in payload["facts"] if f["category"] == "metrics"], "equal metrics are not facts"


def test_compare_with_missing_metrics_says_so(project: Path) -> None:
    script = project / "quiet.py"
    script.write_text("print('no metrics here')\n", encoding="utf-8")
    for name in ("q1", "q2"):
        result = invoke("run", "--name", name, "--", sys.executable, str(script.name), cwd=project)
        assert result.returncode == 0
    payload = compare_json(project, "q1", "q2")
    assert payload["metrics_missing"] is True
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "METRICS_MISSING" in codes


def test_compare_rejects_unknown_run(project: Path) -> None:
    run(project, "only", "--epochs", "1", "--out", "o-k")
    result = invoke("compare", "only", "does-not-exist", cwd=project)
    assert result.returncode == 2
    assert "does-not-exist" in result.stderr.decode("utf-8", "replace")


def test_compare_save_writes_compare_json(project: Path) -> None:
    run(project, "s1", "--epochs", "1", "--out", "o-l")
    run(project, "s2", "--epochs", "2", "--out", "o-m")
    result = invoke("compare", "s1", "s2", "--save", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    saved = next(p for p in (project / ".labbox" / "runs").iterdir() if "s2" in p.name) / "compare.json"
    assert saved.is_file()
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["left_run_id"] and payload["facts"] is not None
    assert payload["reproducibility"]["left"]["checklist"]


def test_diagnostics_and_reproducibility_land_in_the_run_dir(project: Path) -> None:
    run(project, "diag", "--epochs", "1", "--seed", "3", "--out", "o-n")
    run_dir = next(p for p in (project / ".labbox" / "runs").iterdir() if "diag" in p.name)
    document = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert document["reproducibility"]["band"] in {"low", "medium", "high"}
    assert document["reproducibility"]["checklist"]
    assert all(check["reason"] for check in document["reproducibility"]["checklist"])
    assert all(diagnostic["id"].startswith("D-") for diagnostic in document["diagnostics"])
    assert all(
        diagnostic["code"] != "run_in_progress" for diagnostic in document["diagnostics"]
    ), "the initial placeholder must be overwritten once the run finalizes"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["links"]["diagnostics"] == "diagnostics.json"
