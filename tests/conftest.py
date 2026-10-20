from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SRC))

from labbox.config.model import LabboxConfig  # noqa: E402
from labbox.paths import ProjectLayout  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CHILD_ENV = {k: v for k, v in os.environ.items() if k not in {"PYTHONIOENCODING", "PYTHONUTF8"}}


@pytest.fixture
def py() -> str:
    return sys.executable


@pytest.fixture
def child_env() -> dict[str, str]:
    """Environment for child scripts: the machine's locale encoding, like a real run."""
    return dict(CHILD_ENV)


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def project(tmp_path: Path) -> ProjectLayout:
    layout = ProjectLayout(tmp_path / "proj")
    layout.labbox_dir.mkdir(parents=True, exist_ok=True)
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    layout.config_path.write_text(LabboxConfig.default().to_toml(), encoding="utf-8", newline="\n")
    return layout


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    for index in range(12):
        (root / "train" / f"file_{index:03d}.bin").write_bytes(bytes([index]) * 300)
    (root / "train" / "empty.bin").write_bytes(b"")
    yield root
