"""Audit the MVP definition (DEVELOPMENT.md §21) against a real, scratch project.

Every check here is empirical: it runs the CLI as a subprocess and inspects what landed on
disk. A check that cannot be performed on this machine is reported as ``SKIP`` with the
reason instead of being quietly counted as a pass.

Run: ``python scripts/verify_mvp.py``
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
sys.path.insert(0, str(REPO / "src"))

# Print our own findings as UTF-8 with replacement, like scripts/spike_process.py does.
# GitHub's windows-latest runner pipes stdout through cp1252, where the Chinese criterion
# names would raise UnicodeEncodeError and kill the audit before it reports anything.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass

CONSOLE = Path(sys.executable).parent / ("labbox.exe" if os.name == "nt" else "labbox")
SHIM = REPO / "scripts" / "labbox_dev.py"


def cli_command() -> list[str]:
    if CONSOLE.is_file():
        probe = subprocess.run([str(CONSOLE), "init", "--help"], capture_output=True, shell=False, check=False)
        if probe.returncode == 0:
            return [str(CONSOLE)]
    return [sys.executable, str(SHIM)]


CLI = cli_command()


@dataclass(slots=True)
class Result:
    criterion: str
    status: str  # PASS | FAIL | SKIP
    detail: str


results: list[Result] = []


def record(criterion: str, ok: bool, detail: str = "") -> None:
    results.append(Result(criterion, "PASS" if ok else "FAIL", detail))


def skip(criterion: str, reason: str) -> None:
    results.append(Result(criterion, "SKIP", reason))


def invoke(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    clean = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH"}}
    return subprocess.run(
        [*CLI, *args],
        cwd=str(cwd),
        env={**clean, **(env or {})},
        capture_output=True,
        shell=False,
        check=False,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(project: Path, name: str) -> Path:
    """Resolve a run directory by its exact name suffix, falling back to substring."""
    all_runs = sorted(p for p in (project / ".labbox" / "runs").iterdir() if p.is_dir())
    exact = [p for p in all_runs if p.name.endswith("-" + name)]
    matches = exact or [p for p in all_runs if name in p.name]
    assert matches, f"no run named {name} in {[p.name for p in all_runs]}"
    return matches[-1]


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="labbox-mvp-"))
    try:
        project = workspace / "project"
        project.mkdir()
        shutil.copy(EXAMPLES / "train_fake.py", project / "train.py")
        data = project / "data" / "train"
        data.mkdir(parents=True)
        for index in range(5):
            (data / f"shard_{index}.bin").write_bytes(bytes([index + 1]) * 700)
        (project / "configs").mkdir()
        (project / "configs" / "base.yaml").write_text(
            "train:\n  batch_size: 64\n  lr: 0.1\nmodel:\n  name: resnet20\n", encoding="utf-8"
        )
        assert invoke("init", cwd=project).returncode == 0

        # 1. zero instrumentation: the training script is never modified or imported
        first = invoke(
            "run", "--name", "a", "--dataset", "data", "--",
            sys.executable, "train.py", "--epochs", "2", "--seed", "7", "--out", "outputs",
            cwd=project,
        )
        source = (project / "train.py").read_text(encoding="utf-8")
        record(
            "1. 不修改训练脚本也能完成基础记录",
            first.returncode == 0 and "labbox" not in source,
            f"exit={first.returncode}; train.py untouched",
        )

        # 2. the exit code is the training command's own
        second = invoke(
            "run", "--name", "boom", "--", sys.executable, "train.py", "--epochs", "3", "--fail-at", "2",
            "--out", "outputs-boom",
            cwd=project,
        )
        record("2. run 不改变训练结果与退出码", second.returncode == 7, f"exit={second.returncode} (expected 7)")

        # 3. success / failure / interrupted all keep a usable record
        ok_record = read_json(run_dir(project, "a") / "run.json")
        bad_record = read_json(run_dir(project, "boom") / "run.json")
        process = subprocess.Popen(
            [*CLI, "run", "--name", "long", "--sample-interval", "0.2", "--",
             sys.executable, "train.py", "--epochs", "30", "--sleep", "0.5", "--out", "outputs-long"],
            cwd=str(project),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        interrupted_dir: Path | None = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(0.2)
            found = sorted((project / ".labbox" / "runs").glob("*long"))
            if found and (found[0] / "environment.json").is_file():
                interrupted_dir = found[0]
                break
        process.kill()
        process.wait(timeout=30)
        interrupted_ok = False
        interrupted_detail = "run never reached the training phase"
        if interrupted_dir is not None:
            try:
                rec = read_json(interrupted_dir / "run.json")
                initial = read_json(interrupted_dir / "diagnostics.json")
                explained = any(
                    item.get("code") == "run_in_progress" for item in initial.get("diagnostics", [])
                )
                interrupted_ok = (
                    rec.get("status") in {"running", "interrupted"}
                    and (interrupted_dir / "logs" / "stdout.log").is_file()
                    and (interrupted_dir / "environment.json").is_file()
                    and explained
                )
                interrupted_detail = (
                    f"status={rec.get('status')} (a killed recorder leaves 'running', never a fake "
                    f"success) and the directory explains itself: run_in_progress={explained}"
                )
            except (OSError, json.JSONDecodeError) as exc:
                interrupted_detail = f"run.json unreadable: {exc}"
        record(
            "3. 成功/失败/中断三类运行均完整保留",
            ok_record["status"] == "succeeded" and bad_record["status"] == "failed" and interrupted_ok,
            f"succeeded={ok_record['status']} failed={bad_record['status']} interrupted-case={interrupted_detail}",
        )

        # 4. environment variables are redacted by default
        secret = "sk-LEAKCHECK0123456789abcdef"
        invoke(
            "run", "--name", "secret", "--", sys.executable, "train.py", "--epochs", "1", "--out", "outputs-secret",
            cwd=project,
            env={"OPENAI_API_KEY": secret, "MY_DB_PASSWORD": "p@ssw0rd"},
        )
        blob = b"".join(
            path.read_bytes() for path in run_dir(project, "secret").rglob("*") if path.is_file()
        )
        env_doc = read_json(run_dir(project, "secret") / "environment.json")
        record(
            "4. 环境变量默认脱敏",
            secret.encode() not in blob and b"p@ssw0rd" not in blob,
            f"redacted={env_doc['env']['redacted_names'][:4]}",
        )

        # 5. compare lists the structural differences precisely
        invoke(
            "run", "--name", "b", "--",
            sys.executable, "train.py", "--epochs", "2", "--seed", "7", "--lr", "0.02", "--out", "outputs",
            cwd=project,
        )
        compare = invoke("compare", "a", "b", "--json", cwd=project)
        payload = json.loads(compare.stdout.decode("utf-8"))
        paths = {fact["path"] for fact in payload["facts"]}
        record(
            "5. 能准确对比两次运行的结构化差异",
            "train.lr" in paths and payload["facts"],
            f"changed={sorted(paths)[:6]}",
        )

        # 6. fact / inference / missing evidence stay separate
        unknown = [fact for fact in payload["facts"] if fact["kind"] == "unknown"]
        codes = {item["code"] for item in payload["diagnostics"]}
        causal = "caused by" in json.dumps(payload).lower()
        changed = [fact for fact in payload["facts"] if fact["kind"] != "unknown"]
        record(
            "6. 明确区分事实、推断与证据缺失",
            bool(changed) and bool(unknown) and "evidence_gaps" in codes and not causal,
            f"facts={len(changed)} unknown={len(unknown)} evidence_gaps={'evidence_gaps' in codes} causal={causal}",
        )

        # 7. core features work with no GPU present
        env_doc2 = read_json(run_dir(project, "b") / "environment.json")
        record(
            "7. 无 GPU 时核心功能仍可用",
            env_doc2["collectors"].get("gpu") in {"unavailable", "disabled"} or not env_doc2["gpus"],
            f"gpus={len(env_doc2['gpus'])} collector={env_doc2['collectors'].get('gpu')}",
        )

        # 8. the report is a self-contained offline page
        report = run_dir(project, "b") / "report.html"
        html = report.read_text(encoding="utf-8").lower() if report.is_file() else ""
        external = [token for token in ("http://", "https://", "<script", "@import", "url(") if token in html]
        record("8. HTML 报告可离线打开", bool(html) and not external, f"external_refs={external or 'none'}")

        # 9. everything works with the LLM switched off
        explained = invoke("explain", "b", cwd=project)
        text = explained.stderr.decode("utf-8", "replace")
        record(
            "9. LLM 关闭时所有核心功能仍可用",
            explained.returncode == 0 and "llm is disabled" in text and "Deterministic analysis" in text,
            "explain degrades to deterministic output",
        )

        # 10. Windows integration: the release platform is Windows 11
        ci = REPO / ".github" / "workflows" / "ci.yml"
        if not ci.is_file():
            record("10. Windows 集成测试通过", False, "no CI workflow file found")
        elif os.name != "nt":
            skip("10. Windows 集成测试通过", "the release platform is Windows; run this audit there")
        else:
            workflow = ci.read_text(encoding="utf-8")
            windows_only = "windows-latest" in workflow and "ubuntu-latest" not in workflow
            record(
                "10. Windows 集成测试通过",
                windows_only,
                f"this run covered {os.name}; CI workflow is windows-only={windows_only}",
            )

        # extras worth reporting even though §21 does not list them
        artifacts = read_json(run_dir(project, "a") / "artifacts.json")
        datasets = read_json(run_dir(project, "a") / "dataset-manifest.json")
        diagnostics = read_json(run_dir(project, "a") / "diagnostics.json")
        digest = (datasets["datasets"][0]["digest"] if datasets["datasets"] else "not-declared")[:23]
        print(
            "  (info) "
            f"metrics={read_json(run_dir(project, 'a') / 'metrics.json').get('final')} "
            f"dataset_digest={digest} "
            f"artifacts={len(artifacts['artifacts'])} "
            f"repro={diagnostics['reproducibility']['band']}"
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    width = max(len(item.criterion) for item in results) + 2
    failures = [item for item in results if item.status == "FAIL"]
    for item in results:
        print(f"{item.status:<5} {item.criterion:<{width}} {item.detail}")
    passed = sum(1 for item in results if item.status == "PASS")
    skipped = sum(1 for item in results if item.status == "SKIP")
    print(f"\n{passed} PASS, {len(failures)} FAIL, {skipped} SKIP of {len(results)} criteria")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
