# Contributing

Thanks for helping LabBox. This project is Windows-first and currently targets
**Windows 11 + Python 3.11/3.12**. Linux code paths are kept for portability but are not part of
the release contract.

## Development setup

```powershell
uv venv
uv pip install -e ".[dev]"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

If `uv` is unavailable, a normal `python -m venv .venv` followed by
`.venv\Scripts\python -m pip install -e ".[dev]"` works too.

> **Path constraint:** an editable install must live in a pure-ASCII path. On a Chinese Windows
> console, `site.py` decodes the generated `.pth` as cp936, so `labbox.exe` fails with
> `ModuleNotFoundError`. For development inside a non-ASCII path, use
> `python scripts/labbox_dev.py ...` instead of the console script. The recorded project itself
> may live anywhere.

## Before opening a pull request

Run the full local gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

It must finish with `all checks passed`. This covers ruff, ruff format, pyright, schema export,
pytest, and the MVP acceptance audit (`scripts/verify_mvp.py`).

## Non-negotiable design rules

These are product requirements, not style preferences (see `DEVELOPMENT.md`):

1. **Never break training.** A recorder failure may add a warning; it must not change the
   training command's stdout, stderr, or exit code.
2. **Evidence before explanation.** Rules establish facts first. LLM output may cite fact IDs but
   may not invent causation.
3. **No guessing.** A missing metric is `METRICS_MISSING`, not the last float in a log. An
   unknown seed is `unknown`, not a default.
4. **Dataset digests never include mtime.** `path + size + head/tail sample` only; see ADR 0002.
5. **Do not initialize a CUDA context.** No `torch.cuda.is_available()`, `device_count()`, or
   `import torch` in the LabBox process. GPU data comes from NVML / `nvidia-smi`.
6. **Local-first.** No network request unless the user explicitly configured it.
7. **Escape everything user-provided** in HTML, shell scripts, and logs.
8. **Every new diagnostic rule needs a unit test and a real counterexample.** Every behavior
   change needs a test that fails before the change.

## Commit style

Use a short imperative subject, for example:

```text
add --version outside a project
fix Ctrl+C finalize on Windows
docs: scope release to Windows 11
```

Keep unrelated changes in separate commits. Do not commit `.labbox/runs/`, `.venv/`, or logs.
