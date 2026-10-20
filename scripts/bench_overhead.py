"""Performance baseline: what does recording cost? (DEVELOPMENT.md §17, §20 M5)

Compares the bare training command with `labbox run`, on the same machine, same input:
wall clock, LabBox startup cost, and the size of what was written. Numbers are printed for
the record; the assertion is deliberately loose so this stays a benchmark, not a flaky test.

Run: ``python scripts/bench_overhead.py [--epochs 20]``
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
CONSOLE = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
SHIM = REPO / "scripts" / "labbox_dev.py"


def cli() -> list[str]:
    """Prefer the installed console script, but verify it actually starts.

    An editable install under a non-ASCII path yields a console script that dies with
    ModuleNotFoundError (DEVELOPMENT.md §16.3), so the probe is part of the check.
    """
    if CONSOLE.is_file():
        probe = subprocess.run(
            [str(CONSOLE), "--help"], capture_output=True, shell=False, check=False, cwd=str(REPO)
        )
        if probe.returncode in (0, 2):
            return [str(CONSOLE)]
    return [sys.executable, str(SHIM)]


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def measure(command: list[str], cwd: Path, env: dict[str, str]) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), env=env, capture_output=True, shell=False, check=False)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-epochs", type=int, default=8)
    parser.add_argument("--long-epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="labbox-bench-"))
    try:
        shutil.copy(EXAMPLES / "train_fake.py", workspace / "train.py")
        (workspace / "data").mkdir()
        for index in range(50):
            (workspace / "data" / f"f_{index}.bin").write_bytes(bytes([index % 250]) * 900)
        subprocess.run([*cli(), "init"], cwd=str(workspace), capture_output=True, shell=False, check=False)
        clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        counter = {"n": 0}

        def train(epochs: int, sleep: float) -> list[str]:
            return [
                sys.executable,
                "train.py",
                "--epochs",
                str(epochs),
                "--sleep",
                str(sleep),
                "--out",
                "outputs",
            ]

        def labbox_run(epochs: int, sleep: float, *extra: str) -> list[str]:
            counter["n"] += 1
            return [*cli(), "run", "--name", f"bench{counter['n']}", *extra, "--", *train(epochs, sleep)]

        cold_interval = "--sample-interval=600"  # effectively disables the sampling loop
        startup = min(measure([*cli(), "--help"], workspace, clean) for _ in range(args.repeats))
        short_bare = min(
            measure(train(args.short_epochs, 0.02), workspace, clean) for _ in range(args.repeats)
        )
        short_wrapped = min(
            measure(
                labbox_run(args.short_epochs, 0.02, "--dataset", "data", cold_interval),
                workspace,
                clean,
            )
            for _ in range(args.repeats)
        )
        no_hash = min(
            measure(labbox_run(args.short_epochs, 0.02, cold_interval), workspace, clean)
            for _ in range(args.repeats)
        )
        long_bare = min(measure(train(args.long_epochs, args.sleep), workspace, clean) for _ in range(args.repeats))
        long_wrapped = min(
            measure(
                labbox_run(args.long_epochs, args.sleep, "--dataset", "data"),
                workspace,
                clean,
            )
            for _ in range(args.repeats)
        )
        long_quiet = min(
            measure(
                labbox_run(args.long_epochs, args.sleep, "--dataset", "data", cold_interval),
                workspace,
                clean,
            )
            for _ in range(args.repeats)
        )

        runs_root = workspace / ".labbox" / "runs"
        recorded = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.is_dir() else []
        per_run = (dir_bytes(runs_root) / max(len(recorded), 1)) if recorded else 0

        fixed = short_wrapped - short_bare
        sampling_tax = long_wrapped - long_quiet
        tax_share = sampling_tax / long_bare if long_bare else 0.0
        rows = [
            ("labbox startup (interpreter + imports)", startup, "s"),
            (f"bare training, {args.short_epochs} epochs", short_bare, "s"),
            (f"labbox run,  {args.short_epochs} epochs", short_wrapped, "s"),
            ("  fixed recording cost", fixed, "s"),
            ("  fixed cost without --dataset", no_hash - short_bare, "s"),
            (f"bare training, {args.long_epochs} epochs", long_bare, "s"),
            (f"labbox run,  {args.long_epochs} epochs", long_wrapped, "s"),
            ("  total overhead on the long run", long_wrapped - long_bare, "s"),
            ("  sampling tax (2s interval)", sampling_tax, "s"),
            ("  sampling tax as a share", tax_share * 100, "%"),
            ("bytes written per run", per_run / 1024, "KiB"),
        ]
        for label, value, unit in rows:
            print(f"{label:<38} {value:9.3f} {unit}")
        print()
        print("Reading: recording is a fixed per-run price (interpreter probe, git, hashing,")
        print("report), not a per-step tax. The number that must stay near zero is the")
        print("sampling tax, because that is the part a long training run pays repeatedly.")
        failures: list[str] = []
        if startup > 1.5:
            failures.append(f"startup {startup * 1000:.0f} ms exceeds the 1500 ms budget")
        if fixed > 3.0:
            failures.append(f"fixed recording cost {fixed:.2f} s exceeds 3 s")
        if tax_share > 0.05:
            failures.append(f"sampling tax {tax_share * 100:.1f}% of the long run exceeds 5%")
        if failures:
            print("\nBUDGET EXCEEDED:", file=sys.stderr)
            for line in failures:
                print(f"  - {line}", file=sys.stderr)
            return 1
        print("\nwithin budget (§17)")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
