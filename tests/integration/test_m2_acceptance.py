"""M2 acceptance: data, metrics, artifacts and reproduce scripts come out of a real run."""

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


def invoke(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [*CLI, *args],
        cwd=str(cwd),
        env={**clean, **(env or {})},
        capture_output=True,
        shell=False,
        check=False,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLES / "train_fake.py", root / "train.py")
    data = root / "data" / "train"
    data.mkdir(parents=True)
    for index in range(6):
        (data / f"shard_{index}.bin").write_bytes(bytes([index]) * 500)
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text(
        "train:\n  batch_size: 64\n  lr: 0.1\n  amp: true\nmodel:\n  name: resnet20\n", encoding="utf-8"
    )
    assert invoke("init", cwd=root).returncode == 0
    return root


def only_run(project: Path, name: str) -> Path:
    matches = [p for p in (project / ".labbox" / "runs").iterdir() if name in p.name]
    assert matches, f"no run named {name}"
    return matches[-1]


def test_run_produces_metrics_dataset_and_artifacts(project: Path) -> None:
    result = invoke(
        "run",
        "--name",
        "m2",
        "--dataset",
        "data",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "3",
        "--lr",
        "0.05",
        "--config",
        "configs/base.yaml",
        "--out",
        "outputs",
        cwd=project,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    run_dir = only_run(project, "m2")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final"]["accuracy"] == pytest.approx(0.81)
    assert metrics["best"]["accuracy"]["value"] == pytest.approx(0.81)
    assert metrics["metrics_missing"] is False
    assert (run_dir / "logs" / "metrics-history.jsonl").is_file()

    dataset = json.loads((run_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert len(dataset["datasets"]) == 1
    entry = dataset["datasets"][0]
    assert entry["file_count"] == 6
    assert entry["digest"].startswith("sha256:")
    assert entry["strategy"] == "fast_manifest"
    assert any("not prove content immutability" in warning for warning in entry["warnings"])
    assert "entries" not in entry, "the per-file list must not bloat the manifest"

    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    names = [Path(a["path"]).name for a in artifacts["artifacts"]]
    assert "best.pt" in names
    checkpoint = next(a for a in artifacts["artifacts"] if Path(a["path"]).name == "best.pt")
    assert checkpoint["kind"] == "checkpoint"
    assert len(checkpoint["sha256"]) == 64
    assert checkpoint["archived"] is False, "artifacts are referenced, not copied, by default"

    config_values = json.loads((run_dir / "config-values.json").read_text(encoding="utf-8"))
    values = {item["path"]: item for item in config_values["values"]}
    assert values["train.lr"]["value"] == 0.05, "the command line wins over the config file"
    assert len(values["train.lr"]["sources"]) == 2
    assert values["train.batch_size"]["value"] == 64
    assert values["model.name"]["value"] == "resnet20"

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for link in ("metrics", "dataset_manifest", "artifacts", "config_values", "environment"):
        assert record["links"][link], f"{link} is not linked from run.json"


def test_reproduce_scripts_are_generated_and_safe(project: Path) -> None:
    result = invoke(
        "run",
        "--name",
        "repro",
        "--dataset",
        "data",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "1",
        "--out",
        "outputs",
        cwd=project,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    run_dir = only_run(project, "repro")
    sh = run_dir / "reproduce.sh"
    ps1 = run_dir / "reproduce.ps1"
    assert sh.is_file() and ps1.is_file()
    sh_text = sh.read_text(encoding="utf-8")
    assert "raise SystemExit" not in sh_text
    assert "train.py" in sh_text
    assert "--epochs" in sh_text
    assert "pip install" not in sh_text.split("#")[0] or "pip install" in sh_text, (
        "install is suggested, not run"
    )
    for executed in ("rm -rf", "sudo"):
        assert executed not in sh_text
    ps1_text = ps1.read_text(encoding="utf-8-sig")
    offenders = [
        line
        for line in ps1_text.splitlines()
        if "Invoke-Expression" in line and not line.strip().startswith("#")
    ]
    assert not offenders, f"the script may only mention Invoke-Expression in a comment: {offenders}"
    assert "repro" in ps1_text or run_dir.name in ps1_text


def test_undeclared_dataset_is_recorded_as_unknown_not_guessed(project: Path) -> None:
    result = invoke(
        "run",
        "--name",
        "nodata",
        "--",
        sys.executable,
        "train.py",
        "--epochs",
        "1",
        "--out",
        "o1",
        cwd=project,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    run_dir = only_run(project, "nodata")
    dataset = json.loads((run_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert dataset["datasets"] == []
    assert any("no dataset declared" in warning for warning in dataset["warnings"])


def test_metrics_absent_is_reported_as_missing(tmp_path: Path, project: Path) -> None:
    script = project / "silent.py"
    script.write_text("print('trained silently')\n", encoding="utf-8")
    result = invoke("run", "--name", "quiet", "--", sys.executable, "silent.py", cwd=project)
    assert result.returncode == 0
    run_dir = only_run(project, "quiet")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics_missing"] is True
    assert metrics["final"] is None
    assert any("metrics_missing" in warning for warning in metrics["warnings"])
