"""M4 acceptance: offline report and the optional AI layer, end to end (§20 M4).

The LLM path is exercised against a throwaway localhost server, so no external network is
involved and the test still proves the real HTTP round trip, JSON-Schema validation and
citation filtering.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
CONSOLE_SCRIPT = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
DEV_SHIM = REPO / "scripts" / "labbox_dev.py"


def _cli() -> list[str]:
    if CONSOLE_SCRIPT.is_file():
        probe = subprocess.run(
            [str(CONSOLE_SCRIPT), "init", "--help"], capture_output=True, shell=False, check=False, cwd=str(REPO)
        )
        if probe.returncode == 0:
            return [str(CONSOLE_SCRIPT)]
    return [sys.executable, str(DEV_SHIM)]


CLI = _cli()


def invoke(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [*CLI, *args],
        cwd=str(cwd),
        env={**clean, **(env or {})},
        capture_output=True,
        shell=False,
        check=False,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    root.mkdir()
    shutil.copy(EXAMPLES / "train_fake.py", root / "train.py")
    data = root / "data"
    data.mkdir()
    (data / "train.bin").write_bytes(b"q" * 900)
    assert invoke("init", cwd=root).returncode == 0
    return root


def run(project: Path, name: str, *args: str) -> None:
    result = invoke("run", "--name", name, "--dataset", "data", "--", sys.executable, "train.py", *args, cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def only_run(project: Path, name: str) -> Path:
    return next(p for p in (project / ".labbox" / "runs").iterdir() if name in p.name)


def enable_llm(project: Path, port: int) -> None:
    """Rewrite [llm] in the project config to point at the throwaway local endpoint."""
    config = project / ".labbox" / "config.toml"
    lines = config.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    inside = False
    replacements = {
        "enabled": "enabled = true",
        "base_url": f'base_url = "http://127.0.0.1:{port}/v1"',
        "provider": 'provider = "openai_compatible"',
        "model": 'model = "test-model"',
        "send_logs": "send_logs = false",
    }
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == "[llm]"
        if inside:
            key = stripped.split("=", 1)[0].strip()
            if key in replacements:
                output.append(f"{replacements[key]}")
                continue
        output.append(line)
    config.write_text(chr(10).join(output) + chr(10), encoding="utf-8", newline="\n")
    text = config.read_text(encoding="utf-8")
    assert "base_url = " in text and str(port) in text, "the fixture must repoint the endpoint"


_CAPTURED: dict = {}


class _FakeEndpoint(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        _CAPTURED.update(json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)).decode("utf-8")))
        answer = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "两次实验的主要差异是学习率与数据版本；当前证据不足以单独归因。",
                                "facts": [{"fact_id": "F-001", "interpretation": "learning rate changed"}],
                                "hypotheses": [
                                    {
                                        "text": "lower lr may explain the slower convergence",
                                        "confidence": "low",
                                        "evidence_fact_ids": ["F-001", "F-404"],
                                        "validation": "re-run with lr fixed",
                                    }
                                ],
                                "unknowns": ["seed not recorded"],
                                "next_steps": ["change one variable at a time"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        body = json.dumps(answer).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def endpoint():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeEndpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_run_writes_an_offline_report(project: Path) -> None:
    run(project, "rep", "--epochs", "1", "--lr", "0.1", "--out", "o1")
    run_dir = only_run(project, "rep")
    report = run_dir / "report.html"
    assert report.is_file(), "the report is generated as part of the run"
    html = report.read_text(encoding="utf-8")
    assert "rep" in html
    # The run id is derived from the directory that was actually created rather than
    # hardcoded: a fixed timestamp here passed only on the day it was written and failed
    # on every other day, which is a defect in the test, not in the report.
    assert run_dir.name in html, "the report must name the run it describes"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "succeeded"
    assert f"{record['exit_code']}" in html
    assert "not captured" not in html.split("duration", 1)[-1][:400], (
        "the report is rendered from the finished record, not the running placeholder"
    )
    lowered = html.lower()
    for forbidden in ("http://", "https://", "<script", "@import", "url(", "<link"):
        assert forbidden not in lowered, f"the report must be self-contained: found {forbidden!r}"
    assert "reproducibility" in lowered


def test_report_command_regenerates_the_page(project: Path) -> None:
    run(project, "regen", "--epochs", "1", "--out", "o2")
    result = invoke("report", "regen", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "report.html" in result.stderr.decode("utf-8", "replace")
    assert (only_run(project, "regen") / "report.html").is_file()


def test_report_survives_a_failed_and_interrupted_run(project: Path) -> None:
    failed = invoke(
        "run", "--name", "broken", "--dataset", "data", "--",
        sys.executable, "train.py", "--epochs", "2", "--fail-at", "2", "--out", "o3",
        cwd=project,
    )
    assert failed.returncode == 7, "the record is written even though the training command failed"
    result = invoke("report", "broken", cwd=project)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    html = (only_run(project, "broken") / "report.html").read_text(encoding="utf-8")
    assert "failed" in html.lower()
    assert "exit" in html.lower()


def test_explain_refuses_without_a_configured_model(project: Path) -> None:
    run(project, "quiet-mode", "--epochs", "1", "--out", "o4")
    result = invoke("explain", "quiet-mode", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, text
    assert "llm is disabled" in text
    assert "Deterministic analysis" in text


def test_explain_dry_run_shows_what_would_leave_the_machine(project: Path) -> None:
    run(project, "dry", "--epochs", "1", "--out", "o5")
    result = invoke("explain", "dry", "--dry-run", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, text
    assert "Dry run" in text
    assert "diff facts" in text
    assert "no raw logs" in text


def test_explain_uses_the_endpoint_and_drops_invented_citations(project: Path, endpoint) -> None:
    port = endpoint.server_address[1]
    run(project, "explained", "--epochs", "1", "--out", "o6")
    run(project, "explained-b", "--epochs", "2", "--lr", "0.02", "--out", "o7")
    enable_llm(project, port)
    result = invoke("explain", "explained", "--against", "explained-b", cwd=project)
    text = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, text
    assert "Model interpretation" in text, text
    assert "LLM-generated" in text
    assert "F-001" in text
    assert "learning rate changed" in text
    # the canned answer cited F-404, which was never issued
    assert "unknown fact ids" in text or "F-404" in text
    sent = _CAPTURED["messages"][1]["content"]
    assert "guard" in sent
    assert "environment variable values" not in sent
    run_dir = only_run(project, "explained-b")
    envelope = json.loads((run_dir / "explain.json").read_text(encoding="utf-8"))
    assert envelope["llm_used"] is True
    assert envelope["explanation"]["rejected_fact_ids"] == ["F-404"]
    # A hypothesis that cited an invented fact id is dropped wholesale: keeping it "minus
    # the bad id" would present a claim the model never justified with valid evidence.
    assert envelope["explanation"]["hypotheses"] == []
    assert envelope["explanation"]["summary"], "the rest of the answer is still shown"
    assert envelope["explanation"]["facts"][0]["fact_id"] == "F-001"
    assert envelope["deterministic"]["run_ids"]


def test_report_includes_the_model_section_when_present(project: Path, endpoint) -> None:
    port = endpoint.server_address[1]
    run(project, "withmodel", "--epochs", "1", "--out", "o8")
    enable_llm(project, port)
    assert invoke("explain", "withmodel", cwd=project).returncode == 0
    invoke("report", "withmodel", cwd=project)
    html = (only_run(project, "withmodel") / "report.html").read_text(encoding="utf-8")
    assert "learning rate changed" in html or "LLM" in html
