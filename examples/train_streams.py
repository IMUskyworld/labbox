"""Behaves badly on purpose: huge stderr, interleaved streams, CJK and ANSI output.

Used by the ProcessRunner integration tests for the two failure modes that are invisible
until they happen in production: pipe deadlock and encoding corruption.
"""

from __future__ import annotations

import sys

FILLER = "x" * (400 * 1024)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in {"all", "stderr-flood"}:
        sys.stderr.write(FILLER)
        sys.stderr.flush()
    if mode in {"all", "interleave"}:
        for i in range(200):
            print(f"out {i}")
            print(f"err {i}", file=sys.stderr)
    if mode in {"all", "encoding"}:
        # Mixed encodings written as raw bytes: valid UTF-8, then GBK, then an ANSI colour.
        sys.stdout.buffer.write("中文中文\n".encode())
        sys.stdout.buffer.write("中文\n".encode("gbk"))
        sys.stdout.buffer.write(b"\x1b[32mgreen text\x1b[0m done\n")
        sys.stdout.buffer.flush()
    if mode in {"all", "exit-code"}:
        return 3
    print("streams finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
