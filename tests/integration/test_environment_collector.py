"""Environment collection: cross-interpreter probing, redaction, honest failure modes."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.common import CollectorStatus
from labbox.infrastructure.nvidia import FakeBackend, GpuStatic, NullBackend
from labbox.infrastructure.probe import ProbeTarget, resolve_probe_target
from labbox.recorder.environment import EnvironmentCollector


def target_for(executable: str, *flags: str) -> ProbeTarget:
    return ProbeTarget(argv_prefix=[executable, *flags], kind="python", display=executable)


@pytest.fixture
def collector(project) -> EnvironmentCollector:
    return EnvironmentCollector(project_root=project.root, config=LabboxConfig.default())


def test_probe_reads_the_training_interpreter_not_labboxes_own_state(collector) -> None:
    snapshot = collector.collect(argv=[sys.executable, "train.py"], probe_target=target_for(sys.executable))
    assert snapshot.python.version.startswith("3.")
    assert snapshot.python.executable.lower().endswith("python.exe") or snapshot.python.executable.endswith(
        "python"
    )
    assert snapshot.collectors["python_packages"] in {
        CollectorStatus.OK.value,
        CollectorStatus.PARTIAL.value,
    }
    # pydantic is installed in the venv running these tests, torch is not: a framework
    # list must reflect what the probed interpreter actually has.
    if snapshot.packages.get("pydantic"):
        assert "torch" not in snapshot.frameworks
    assert snapshot.host.ram_bytes and snapshot.host.ram_bytes > 0
    assert snapshot.os.name in {"Windows", "Linux", "Darwin"}


def test_probe_never_imports_torch_in_the_parent_process(collector) -> None:
    collector.collect(argv=[sys.executable, "train.py"], probe_target=target_for(sys.executable))
    assert "torch" not in sys.modules, (
        "LabBox must learn framework versions from a probe subprocess, not by importing them"
    )


def test_secrets_never_reach_the_snapshot(collector) -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OPENAI_API_KEY": "sk-abcdefghij0123456789ABCDEF",
        "HF_TOKEN": "hf_" + "A" * 30,
        "DB_PASSWORD": "hunter2",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "0",
    }
    snapshot = collector.collect(
        argv=[sys.executable, "train.py"], env=env, probe_target=target_for(sys.executable)
    )
    assert "OPENAI_API_KEY" not in snapshot.env.entries
    assert "OPENAI_API_KEY" in snapshot.env.redacted_names
    assert "hunter2" not in repr(snapshot.model_dump(mode="json"))
    assert "sk-abcdefghij" not in repr(snapshot.model_dump(mode="json"))
    # reproducibility-relevant variables survive the deny-list
    assert snapshot.env.entries["CUDA_VISIBLE_DEVICES"] == "0"
    assert snapshot.seeds.python_hash_seed == "0"


def test_external_command_records_unknown_instof_guessed_versions(collector) -> None:
    snapshot = collector.collect(argv=["bash", "train.sh"], probe_target=ProbeTarget([], "external", "bash"))
    assert snapshot.python.kind == "external"
    assert snapshot.frameworks == {}
    assert "frameworks" in snapshot.absent
    assert snapshot.collectors["python_packages"] == CollectorStatus.SKIPPED.value
    assert snapshot.interpreter_probe.status is CollectorStatus.SKIPPED


def test_probe_timeout_is_a_recorded_failure_not_a_crash(project) -> None:
    config = LabboxConfig.default()
    config.run.collect_timeout_seconds = 1.0
    strict = EnvironmentCollector(project_root=project.root, config=config)
    slow = target_for(sys.executable, "-c", "import time; time.sleep(5)")
    snapshot = strict.collect(argv=[sys.executable, "train.py"], probe_target=slow)
    assert snapshot.interpreter_probe.status is CollectorStatus.TIMEOUT
    assert snapshot.collectors["python_packages"] == CollectorStatus.UNAVAILABLE.value
    assert "frameworks" in snapshot.absent
    assert any(error["code"] == "INTERPRETER_PROBE_TIMEOUT" for error in snapshot.errors)


def test_git_state_in_a_real_repository(project, collector) -> None:
    repo = project.root
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "initial"],
    ):
        subprocess.run(argv, cwd=str(repo), env=env, check=True, capture_output=True, shell=False)
    snapshot = collector.collect(argv=[sys.executable, "train.py"], probe_target=target_for(sys.executable))
    assert snapshot.git.has_repo is True
    assert snapshot.git.dirty is False
    assert len(snapshot.git.commit or "") == 40
    assert snapshot.git.diff_hash is None

    (repo / "extra.py").write_text("untracked\n", encoding="utf-8")
    dirty = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    assert dirty.git.dirty is True
    assert dirty.git.diff_hash and dirty.git.diff_hash.startswith("sha256:")
    assert dirty.git.untracked_count == 1


def test_no_repository_is_reported_as_such(collector) -> None:
    snapshot = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    if not snapshot.git.has_repo:
        assert snapshot.collectors["git"] == CollectorStatus.UNAVAILABLE.value
        assert "git.commit" in snapshot.absent


def test_gpu_information_comes_from_nvml_not_torch(collector) -> None:
    backend = FakeBackend(
        gpus=[GpuStatic(index=0, name="RTX 3090", vram_bytes=25769803776, driver="552.22")],
        script=[],
    )
    collector._backend = backend
    snapshot = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    assert snapshot.gpus[0].name == "RTX 3090"
    assert snapshot.cuda.driver == "552.22"
    assert snapshot.collectors["gpu"] == "ok"


def test_no_gpu_environment_still_produces_a_full_snapshot(collector) -> None:
    collector._backend = NullBackend("no gpu here")
    snapshot = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    assert snapshot.gpus == []
    assert snapshot.collectors["gpu"] == CollectorStatus.UNAVAILABLE.value
    assert "gpus" in snapshot.absent
    assert snapshot.os.name  # the rest of the run is still recorded
    assert any(error["code"] == "GPU_COLLECTOR_UNAVAILABLE" for error in snapshot.errors)


def test_gpu_collection_can_be_disabled(collector) -> None:
    collector.config.gpu.enabled = False
    snapshot = collector.collect(argv=[sys.executable, "--no-gpu"], probe_target=target_for(sys.executable))
    assert snapshot.collectors["gpu"] == CollectorStatus.DISABLED.value


def test_seeds_declared_on_the_command_line_are_captured(collector) -> None:
    snapshot = collector.collect(
        argv=[sys.executable, "train.py", "--seed", "1234", "--deterministic", "true"],
        probe_target=target_for(sys.executable),
    )
    assert snapshot.seeds.declared_in_command.get("seed") == "1234"
    assert snapshot.seeds.torch_seed.status is CollectorStatus.UNAVAILABLE, (
        "an in-script manual_seed is not observable from outside; never claim it"
    )


def test_snapshot_is_json_serializable_end_to_end(collector) -> None:
    import json

    snapshot = collector.collect(
        argv=[sys.executable, "train.py"],
        env={**os.environ, "OPENAI_API_KEY": "sk-zzzzzzzzzzzzzzzzzzzzzzzz"},
        probe_target=target_for(sys.executable),
    )
    payload = json.loads(json.dumps(snapshot.to_json_dict(), ensure_ascii=False))
    restored = type(snapshot).model_validate(payload)
    assert restored.to_json_dict() == payload, "a snapshot must survive a disk round trip"
    assert restored.captured_at.utcoffset() == snapshot.captured_at.utcoffset()
    assert restored.os.name == snapshot.os.name
    assert restored.env.redacted_names == snapshot.env.redacted_names


def test_timestamps_carry_the_local_utc_offset(collector) -> None:
    snapshot = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    assert snapshot.captured_at.tzinfo is not None
    assert snapshot.captured_at.utcoffset() is not None
    assert "+08:00" in snapshot.captured_at.isoformat() or snapshot.captured_at.utcoffset() is not None


def test_collector_respects_git_disable_flag(collector) -> None:
    collector.config.git.enabled = False
    snapshot = collector.collect(argv=[sys.executable], probe_target=target_for(sys.executable))
    assert snapshot.git.has_repo is False
    assert snapshot.collectors["git"] == CollectorStatus.DISABLED.value


def test_resolve_probe_target_for_python_launchers() -> None:
    own = resolve_probe_target([sys.executable, "train.py"])
    assert own.kind == "python"
    external = resolve_probe_target(["srun", "-n", "2", "python", "train.py"])
    assert external.kind == "external", "a launcher wrapper must not be probed as the interpreter"
    bare = resolve_probe_target(["python", "train.py"])
    assert bare.kind == "python"
    override = resolve_probe_target(["python", "train.py"], override=sys.executable)
    assert override.explicit and override.argv_prefix == [sys.executable]
    empty = resolve_probe_target([])
    assert empty.kind == "external"


def test_probe_reports_the_interpreter_it_actually_reached() -> None:
    from labbox.infrastructure.probe import probe_interpreter

    result = probe_interpreter(target_for(sys.executable), extra_packages=["pydantic"], timeout=20)
    assert result.ok
    assert (
        result.data["executable"]
        .lower()
        .replace("\\", "/")
        .endswith(sys.executable.lower().replace("\\", "/").rsplit("/", 1)[-1])
    )
    assert result.data["packages"].get("pydantic")
    assert result.duration_ms >= 0
    assert isinstance(datetime.now(), datetime)
