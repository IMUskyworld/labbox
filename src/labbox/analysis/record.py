"""Loading a recorded run back into one object the analysis layer can reason about."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labbox.domain.artifacts import ArtifactManifest
from labbox.domain.dataset import DatasetManifest
from labbox.domain.diagnostics import DiagnosticsFile
from labbox.domain.environment import EnvironmentSnapshot
from labbox.domain.hyperparams import HyperparamSet
from labbox.domain.metrics import MetricSummary
from labbox.domain.run import RunRecord
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import RunPaths


class RunBundle:
    """Everything LabBox knows about one run, tolerant of partially written runs.

    An interrupted run has a ``run.json`` and maybe logs, nothing else. Analysis must
    still be able to say "unknown" about the rest rather than crash — which is exactly
    what a crashed experiment deserves.
    """

    def __init__(self, paths: RunPaths, store: FilesystemStore) -> None:
        self.paths = paths
        self.store = store
        self.record: RunRecord | None = self._model(paths.run_json, RunRecord)
        self.environment: EnvironmentSnapshot | None = self._model(paths.environment_json, EnvironmentSnapshot)
        self.metrics: MetricSummary | None = self._model(paths.metrics_json, MetricSummary)
        self.datasets: DatasetManifest | None = self._model(paths.dataset_manifest_json, DatasetManifest)
        self.artifacts: ArtifactManifest | None = self._model(paths.artifacts_json, ArtifactManifest)
        self.config_values: HyperparamSet | None = self._model(paths.config_values_json, HyperparamSet)
        self.diagnostics: DiagnosticsFile | None = self._model(paths.diagnostics_json, DiagnosticsFile)
        self.raw_run: dict[str, Any] = store.read_json(paths.run_json, strict=False) or {}

    def _model(self, path: Path, model: type[Any]) -> Any | None:
        data = self.store.read_json(path, strict=False)
        if not data:
            return None
        try:
            return model.model_validate(data)
        except Exception:
            return None

    @classmethod
    def load(cls, store: FilesystemStore, run_dir: Path) -> RunBundle:
        return cls(RunPaths(dir=run_dir), store)

    @property
    def run_id(self) -> str:
        if self.record is not None:
            return self.record.run_id
        return str(self.paths.dir.name)

    @property
    def display_name(self) -> str:
        if self.record is not None:
            return self.record.name
        return self.paths.dir.name

    def env_value(self, dotted: str) -> Any:
        """Read a dotted path out of the environment snapshot, or ``None`` if unknown."""
        if self.environment is None:
            return None
        current: Any = self.environment.model_dump()
        for part in dotted.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def is_absent(self, dotted: str) -> bool:
        if self.environment is None:
            return True
        return any(entry == dotted or dotted.startswith(entry + ".") for entry in self.environment.absent)

    def hyperparam(self, dotted: str) -> Any:
        if self.config_values is None:
            return None
        value = self.config_values.as_map().get(dotted)
        return None if value is None else value.value

    def hyperparams(self) -> dict[str, Any]:
        if self.config_values is None:
            return {}
        return {item.path: item.value for item in self.config_values.values}

    def dataset_digests(self) -> dict[str, str | None]:
        if self.datasets is None:
            return {}
        return {entry.id: entry.digest for entry in self.datasets.datasets}

    def checkpoint_hashes(self) -> dict[str, str | None]:
        if self.artifacts is None:
            return {}
        return {entry.path: entry.sha256 for entry in self.artifacts.artifacts}

    def final_metrics(self) -> dict[str, float]:
        if self.metrics is None or not self.metrics.final:
            return {}
        return dict(self.metrics.final)

    def git_state(self) -> dict[str, Any]:
        if self.environment is None:
            return {}
        return self.environment.git.model_dump()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "record_loaded": self.record is not None,
            "sections": {
                "environment": self.environment is not None,
                "metrics": self.metrics is not None,
                "datasets": self.datasets is not None,
                "artifacts": self.artifacts is not None,
                "config_values": self.config_values is not None,
                "diagnostics": self.diagnostics is not None,
            },
        }
