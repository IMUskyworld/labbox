"""Console interrupt handling must match what an experimenter actually presses (§6.3).

Ctrl+C and Ctrl+Break are both "stop this" keys in a Windows console, but CPython only
turns Ctrl+C into ``KeyboardInterrupt`` by default - measured in scripts/spike_process.py,
where Ctrl+Break terminated LabBox outright and the run record was never finalized.
"""

from __future__ import annotations

import os
import signal

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows console semantics")
def test_ctrl_break_is_bound_to_the_interrupt_handler() -> None:
    import labbox.cli.main  # noqa: F401 - importing installs the handler

    handler = signal.getsignal(signal.SIGBREAK)
    assert handler is signal.default_int_handler, (
        "Ctrl+Break must raise KeyboardInterrupt like Ctrl+C, otherwise LabBox dies mid-run "
        "and the run directory is left without its finalize files"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows console semantics")
def test_installing_the_handler_is_idempotent_and_safe() -> None:
    from labbox.cli.main import _handle_console_break_like_interrupt

    for _ in range(3):
        _handle_console_break_like_interrupt()
    assert signal.getsignal(signal.SIGBREAK) is signal.default_int_handler


def test_ctrl_c_handler_still_the_default() -> None:
    """We must not hijack Ctrl+C: the default handler is what lets ProcessRunner see it."""
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
