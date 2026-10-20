"""Cross-interpreter, read-only probing of the *training* environment (§14.3).

LabBox lives in its own environment, so it must never ``import torch`` in its own
process to learn what the training script uses. Instead we run a fixed probe script with
the interpreter the training command names. Two consequences drive this design:

* the probe may only read metadata (``importlib.metadata``), never import the probed
  packages — importing torch would create a CUDA context and change the memory landscape
  of the run we are supposed to leave untouched;
* a probe costs a process spawn (~100-300 ms on Windows), so everything we can learn in
  that interpreter is learned in one call.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MARKER = "::LABBOX_PROBE::"

# Run with the probe target's own interpreter. Keep it import-free apart from stdlib and
# make sure a failure still prints something parseable.
PROBE_SRC = r"""
import json, os, platform, sys

result = {
    "ok": True,
    "python_version": platform.python_version(),
    "python_version_tuple": list(sys.version_info[:3]),
    "implementation": platform.python_implementation(),
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "in_virtualenv": sys.prefix != sys.base_prefix,
    "platform": sys.platform,
    "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
    "packages": {},
    "frameworks": {},
    "package_errors": [],
    "torch_cuda_hint": None,
    "probe_env": {},
}

try:
    names = json.loads(os.environ.get("LABBOX_PROBE_PACKAGES") or "[]")
except Exception:
    names = []

framework_names = {"torch", "torchvision", "torchaudio", "tensorflow", "jax",
                   "transformers", "accelerate", "datasets", "lightning", "pytorch-lightning",
                   "keras", "scikit-learn", "numpy", "pandas", "deepspeed", "peft", "diffusers"}
want = list(dict.fromkeys([n for n in names if n] + sorted(framework_names)))

try:
    import importlib.metadata as _md
except Exception as exc:  # pragma: no cover - very old/broken installs
    result["package_errors"].append("importlib.metadata unavailable: %s" % exc)
    _md = None

if _md is not None:
    for name in want:
        try:
            result["packages"][name] = _md.version(name)
        except Exception:
            result["packages"][name] = None

    # "torch" metadata may be present while the import name differs; only metadata is
    # read here on purpose. Never `import torch` (CUDA context, side effects, startup cost).
    for fw in ("torch", "torchvision", "tensorflow", "jax", "numpy"):
        value = result["packages"].get(fw)
        if value:
            result["frameworks"][fw] = value

    try:
        dist = _md.distribution("torch")
        meta = dist.metadata or {}
        # `torch-cuda` is an optional metadata field some wheels set; absence is normal.
        result["torch_cuda_hint"] = meta.get("torch-cuda")
    except Exception:
        pass

for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
            "TORCH_CUDNN_V8_API_DISABLED", "OMP_NUM_THREADS", "NVIDIA_TF32_OVERRIDE",
            "TOKENIZERS_PARALLELISM", "CUDA_MODULE_LOADING", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
    value = os.environ.get(key)
    if value is not None:
        result["probe_env"][key] = value

print("::LABBOX_PROBE::" + json.dumps(result, ensure_ascii=False, default=str))
"""

PYTHON_BASENAMES = re.compile(r"^(python|python\d|python\d\.\d+|pypy|pypy3|py|conda-run)$", re.I)
_WINDOWS_SHIM = re.compile(r"^python(\d+\.\d+)?\.exe$", re.I)


@dataclass(slots=True)
class ProbeTarget:
    """How to ask a command's interpreter about itself."""

    argv_prefix: list[str]
    kind: str  # "python" | "external"
    display: str
    explicit: bool = False
    notes: list[str] = field(default_factory=list)


def resolve_probe_target(argv: Sequence[str], *, override: str | None = None, cwd: str | None = None) -> ProbeTarget:
    """Decide which interpreter to probe, or admit that we cannot.

    ``conda run -n env python train.py`` and ``venv/bin/python train.py`` both resolve to
    the leading token; a bare ``bash train.sh`` or ``srun python train.py`` is recorded as
    external rather than probed in the wrong interpreter.
    """
    if override:
        exe = _which_or_path(override, cwd)
        return ProbeTarget(
            argv_prefix=[exe],
            kind="python",
            display=override,
            explicit=True,
        )
    if not argv:
        return ProbeTarget(argv_prefix=[], kind="external", display="", notes=["empty command"])

    head = argv[0]
    base = os.path.basename(head).lower()
    stem = re.sub(r"\.exe$", "", base)
    looks_python = bool(PYTHON_BASENAMES.match(stem)) or bool(_WINDOWS_SHIM.match(base))

    if looks_python:
        resolved = _which_or_path(head, cwd)
        if stem == "py":
            # The Windows launcher needs its version flag repeated for the probe.
            flags = [a for a in argv[1:] if re.fullmatch(r"-\d+(\.\d+)?", a)]
            return ProbeTarget(
                argv_prefix=[resolved, *flags],
                kind="python",
                display=" ".join([head, *flags]).strip(),
                notes=["resolved via Windows py launcher"],
            )
        return ProbeTarget(argv_prefix=[resolved], kind="python", display=resolved)

    return ProbeTarget(
        argv_prefix=[],
        kind="external",
        display=head,
        notes=[f"command head {head!r} is not a Python interpreter; package versions unknown"],
    )


def _which_or_path(head: str, cwd: str | None) -> str:
    if os.path.sep in head or (os.path.altsep and os.path.altsep in head):
        full = head if os.path.isabs(head) else os.path.join(cwd or os.getcwd(), head)
        return os.path.abspath(full)
    found = shutil.which(head)
    if found:
        return found
    if head.lower() in {"python", "python3"} and running_from_venv():
        return sys.executable
    return head


def running_from_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


@dataclass(slots=True)
class ProbeResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    code: str | None = None
    returncode: int | None = None
    duration_ms: int = 0


def probe_interpreter(
    target: ProbeTarget,
    *,
    extra_packages: Sequence[str] = (),
    timeout: float = 10.0,
    env: dict[str, str] | None = None,
    text_limit: int = 200_000,
) -> ProbeResult:
    """Run the probe in the target interpreter with a strict timeout."""
    import time

    if target.kind != "python" or not target.argv_prefix:
        return ProbeResult(
            ok=False,
            error="; ".join(target.notes) or "no python interpreter to probe",
            code="INTERPRETER_NOT_FOUND",
        )
    run_env = dict(env if env is not None else os.environ)
    run_env["LABBOX_PROBE_PACKAGES"] = json.dumps(list(extra_packages), ensure_ascii=False)
    run_env["PYTHONIOENCODING"] = "utf-8"
    run_env["PYTHONUTF8"] = "1"
    argv = [*target.argv_prefix, "-c", PROBE_SRC]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=run_env,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            ok=False,
            error=f"probe timed out after {timeout:.1f}s",
            code="INTERPRETER_PROBE_TIMEOUT",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except OSError as exc:
        return ProbeResult(
            ok=False,
            error=f"cannot start {target.display}: {exc}",
            code="INTERPRETER_NOT_FOUND",
        )
    duration_ms = int((time.monotonic() - started) * 1000)
    payload = _extract_payload(completed.stdout or "", text_limit)
    if payload is None:
        return ProbeResult(
            ok=False,
            error=(completed.stderr or "")[-400:] or "probe produced no output",
            code="INTERPRETER_PROBE_FAILED",
            returncode=completed.returncode,
            duration_ms=duration_ms,
        )
    payload.setdefault("executable", target.argv_prefix[0])
    return ProbeResult(ok=True, data=payload, returncode=completed.returncode, duration_ms=duration_ms)


def _extract_payload(stdout: str, limit: int) -> dict[str, Any] | None:
    trimmed = stdout[-limit:]
    index = trimmed.rfind(MARKER)
    if index < 0:
        return None
    raw = trimmed[index + len(MARKER) :].strip().splitlines()
    if not raw:
        return None
    try:
        data = json.loads(raw[0])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
