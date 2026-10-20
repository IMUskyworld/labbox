"""Fake training script: the happy path.

Deliberately dependency-free so it runs in *any* interpreter, including one with no
torch installed: LabBox's M1 acceptance gate is about process and record fidelity, not
about deep-learning correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

EPOCHS = [(0.41, 0.62), (0.28, 0.74), (0.21, 0.81), (0.17, 0.85), (0.14, 0.87)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", default="true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--fail-at", type=int, default=None)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step in range(args.epochs):
            # The curve cycles so a 200-epoch benchmark still takes 200 * --sleep seconds.
            loss, accuracy = EPOCHS[step % len(EPOCHS)]
            epoch = step + 1
            if args.fail_at is not None and epoch == args.fail_at:
                print(f"epoch {epoch}: injected failure", file=sys.stderr)
                return 7 if not args.crash else _raise()
            print(
                f"epoch {epoch}/{args.epochs} - loss: {loss:.4f} - acc: {accuracy:.4f} "
                f"- lr: {args.lr} - bs: {args.batch_size} - amp: {args.amp}"
            )
            metrics.write(json.dumps({"epoch": epoch, "loss": loss, "accuracy": accuracy}) + "\n")
            metrics.flush()
            time.sleep(args.sleep)
    checkpoint = out / "best.pt"
    payload = bytes(random.Random(args.seed or 0).getrandbits(8) for _ in range(4096))
    checkpoint.write_bytes(payload)
    print(
        json.dumps(
            {"final_accuracy": EPOCHS[(args.epochs - 1) % len(EPOCHS)][1], "checkpoint": str(checkpoint)}
        )
    )
    print(f"wrote {checkpoint} ({checkpoint.stat().st_size} bytes) in {os.getcwd()}")
    return 0


def _raise() -> int:
    raise RuntimeError("injected crash")


if __name__ == "__main__":
    sys.exit(main())
