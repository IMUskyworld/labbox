# Changelog

All notable changes to LabBox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-20

First public release. **Supported platform: Windows 11.**

### Added

- `labbox init` — project bootstrap, `.labbox/config.toml`, `.gitignore` entry, capability report.
- `labbox run --name NAME -- <command>` — zero-instrumentation recording of a training process:
  raw exit code preservation, stdout/stderr byte-for-byte forwarding, environment snapshot,
  Git state, CPU/RAM/GPU sampling, metrics, artifacts, dataset manifest, config values,
  diagnostics, offline HTML report, and `reproduce.ps1` / `reproduce.sh`.
- `labbox list` / `labbox show` — browse recorded runs and inspect their files.
- `labbox compare A B` — structured diff facts (`F-xxx`), diagnostics, and metric deltas;
  explicitly distinguishes facts, inference, missing evidence, and incomparable cases.
- `labbox report` — single-file, offline HTML report with no remote assets.
- `labbox explain` — optional OpenAI-compatible LLM layer that can only cite fact IDs;
  disabled by default, with `--dry-run` showing what would be sent.
- `labbox doctor` — checks Git, GPU, psutil, config, LLM endpoint, and stale runs.
- `labbox clean` — list and delete run directories with confirmation, `--dry-run`,
  `--keep-last`, and `--older-than`.
- `labbox --version`.
- Windows Ctrl+C handling: runs finalize as `interrupted`, raw exit code `0xC000013A`
  is normalized to `130`, and partial records remain explainable.

### Known limitations

- Only Windows 11 is verified. POSIX code paths exist but are not part of the release contract.
- `NvmlBackend` has not been exercised on a real NVIDIA GPU in this release (core features work
  without a GPU; GPU fields are recorded as `unknown`).
- No metric inference from free-form logs: a run without structured metrics reports
  `METRICS_MISSING`.
- Large checkpoints are not copied by default, and files above the hash threshold are deferred.
- `compare` extracts config values only from YAML, JSON, and TOML.
