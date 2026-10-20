# ADR 0001 — LabBox is installed separately from the training environment

Date: 2026-09-22
Status: accepted
Relates to: DEVELOPMENT.md §14.3, §24

## Context

LabBox has to report which PyTorch, CUDA and pip package versions a run used. The obvious
implementation — `import torch; torch.__version__` — requires LabBox to be installed in
the same interpreter as the training script. That has two consequences we cannot accept:

* LabBox's own dependencies (pydantic, typer, rich, httpx, jinja2, psutil) would be pushed
  into the environment the experiment depends on, so "recording did not change the run"
  would be false from the first use;
* importing torch initialises parts of the runtime. `torch.cuda.is_available()` creates a
  CUDA context and can take a few hundred MB of VRAM, which changes exactly the memory
  headroom the training run is being measured against.

## Decision

LabBox lives in its own environment (`uv tool install`, `pipx`, or a dedicated venv). To
learn about the training environment it runs a fixed probe script **with the interpreter
named by the training command**:

```text
subprocess.run([training_python, "-c", PROBE_SRC], timeout=10)
```

The probe may only read metadata (`importlib.metadata.version("torch")`), never import the
package. If the command is not a Python interpreter (`bash train.sh`, `srun ...`), the
result is recorded as `interpreter.kind = "external"` and the package facts are marked
`absent` rather than guessed.

## Consequences

* Every environment capture pays one process spawn (~100-300 ms on Windows), so the capture
  budget is `< 3 s`, not `< 0.5 s`.
* Three behaviours are now forbidden in collector code and enforced by review: `import torch`
  inside LabBox, `torch.cuda.is_available()`, and installing anything on the user's behalf.
* Users must not `pip install labbox` into their training env. README and `labbox doctor`
  both say so.
