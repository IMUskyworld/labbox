"""Export the on-disk JSON documents as machine-readable JSON Schema (§18.1, docs/schemas).

Run: ``python scripts/export_schemas.py``

The schemas are generated from the pydantic models, so they cannot drift from the code
that writes the files. Each document carries ``schema_version``; a breaking change has to
bump it and add a migration, and this directory is where reviewers compare shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from labbox.domain.artifacts import ArtifactManifest  # noqa: E402
from labbox.domain.dataset import DatasetManifest  # noqa: E402
from labbox.domain.diagnostics import CompareResult, DiagnosticsFile  # noqa: E402
from labbox.domain.environment import EnvironmentSnapshot  # noqa: E402
from labbox.domain.explain import ExplainContext, ExplainResult  # noqa: E402
from labbox.domain.hyperparams import HyperparamSet  # noqa: E402
from labbox.domain.metrics import MetricSummary  # noqa: E402
from labbox.domain.run import RunRecord  # noqa: E402

TARGETS = {
    "run.json": RunRecord,
    "environment.json": EnvironmentSnapshot,
    "metrics.json": MetricSummary,
    "dataset-manifest.json": DatasetManifest,
    "artifacts.json": ArtifactManifest,
    "config-values.json": HyperparamSet,
    "diagnostics.json": DiagnosticsFile,
    "compare.json": CompareResult,
    "explain-input.json": ExplainContext,
    "explain-output.json": ExplainResult,
}


def main() -> int:
    out_dir = REPO / "docs" / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in TARGETS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["x-labbox-document"] = name
        target = out_dir / f"{name.replace('.json', '')}.schema.json"
        target.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        required = len(schema.get("required", []))
        print(f"{target.relative_to(REPO)}: {len(schema.get('properties', {}))} fields, {required} required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
