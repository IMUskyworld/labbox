"""The reproduce script generator (§12.2): escaping, degraded output, no mutation.

The tests build fixtures from the real domain models and then treat the generated text as
what it is: a program a human will run. So the assertions parse the emitted shell arrays
back into argv, check that only *printed* text ever mentions an installer, and verify the
PASS / FAIL / UNKNOWN contract.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest

from labbox.domain.dataset import (
    DatasetEntry,
    DatasetKind,
    DatasetManifest,
    DatasetStrategy,
)
from labbox.domain.environment import (
    CudaInfo,
    EnvironmentSnapshot,
    GitState,
    GpuInfo,
    HostInfo,
    OsInfo,
    PythonInfo,
    Seeds,
)
from labbox.domain.run import CommandSpec, ExitMeaning, RunRecord, RunStatus
from labbox.errors import LabBoxError
from labbox.infrastructure import shellgen
from labbox.infrastructure.shellgen import (
    PS1_NAME,
    SH_NAME,
    render_reproduce_scripts,
    write_reproduce_scripts,
)
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import RunPaths

T0 = datetime(2026, 9, 22, 15, 30, 12)

# Args chosen to be the worst cases: shell chaining, a pipe, an embedded double quote, a
# single quote, a space, a newline, and non-ASCII. None of them may become syntax.
DANGEROUS_ARGV = [
    "python",
    "实验/train.py",
    "--lr",
    "0.1",
    "--tag",
    'a&&b|"c d"',
    "--note",
    "x'y",
    "--multi",
    "line\nvalue",
    "--redirect",
    "out>file 2>&1",
    "--sub",
    "$(rm -rf /)",
    "--glob",
    "*;reboot",
    "--empty",
    "",
]


def make_record(**overrides: object) -> RunRecord:
    base: dict = {
        "run_id": "20260922-153012-baseline",
        "name": "基线实验",
        "project": "cifar100-resnet",
        "status": RunStatus.SUCCEEDED,
        "started_at": T0,
        "ended_at": T0 + timedelta(seconds=6201),
        "duration_seconds": 6201.0,
        "exit_code": 0,
        "exit_code_raw": 0,
        "exit_meaning": ExitMeaning.EXITED,
        "command": CommandSpec(
            raw="python 实验/train.py --lr 0.1",
            argv=list(DANGEROUS_ARGV),
            cwd="C:/code/实验 项目",
        ),
        "labbox_version": "0.1.0",
    }
    base.update(overrides)
    return RunRecord.model_validate(base)


def make_environment(**overrides: object) -> EnvironmentSnapshot:
    base: dict = {
        "captured_at": T0,
        "os": OsInfo(name="Windows", version="11", arch="AMD64"),
        "python": PythonInfo(
            version="3.11.9",
            executable="C:/code/实验 项目/.venv/Scripts/python.exe",
            venv_prefix="C:/code/实验 项目/.venv",
        ),
        "frameworks": {"torch": "2.6.0"},
        "packages": {"torch": "2.6.0", "numpy": "1.26.4", "transformers": "4.44.0"},
        "cuda": CudaInfo(runtime="12.4", driver="552.22", cudnn="9.1.0"),
        "git": GitState(
            has_repo=True,
            commit="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
            short_commit="a1b2c3d",
            branch="main",
            dirty=False,
        ),
        "host": HostInfo(cpu_model="AMD Ryzen 9", physical_cores=12, logical_cores=24),
        "gpus": [GpuInfo(index=0, name="NVIDIA GeForce RTX 3090", driver="552.22")],
        "seeds": Seeds(python_hash_seed="0"),
        "collectors": {"git": "ok", "gpu": "ok", "python_packages": "partial"},
        "absent": [],
    }
    base.update(overrides)
    return EnvironmentSnapshot.model_validate(base)


def make_manifest() -> DatasetManifest:
    return DatasetManifest(
        datasets=[
            DatasetEntry(
                id="cifar100-train",
                source="data/cifar100/train",
                strategy=DatasetStrategy.FAST_MANIFEST,
                digest="sha256:91abc123",
                file_count=50_000,
                total_bytes=12_345_678,
                warnings=["zero_byte_files"],
            ),
            DatasetEntry(
                id="imagenet-val",
                source="hf:imagenet-1k",
                kind=DatasetKind.EXTERNAL,
                strategy=DatasetStrategy.FULL_CONTENT,
                digest="sha256:ff00dd",
                file_count=50_000,
            ),
        ]
    )


def render(
    record: RunRecord | None = None,
    environment: EnvironmentSnapshot | Literal["missing"] | None = "missing",
    datasets: Sequence[DatasetEntry] | DatasetManifest | Literal["missing"] | None = "missing",
    **kwargs: object,
) -> tuple[str, str]:
    """Render with real domain objects; ``"missing"`` means "use the rich fixture"."""
    env: Any = make_environment() if environment == "missing" else environment
    data: Any = make_manifest() if datasets == "missing" else datasets
    return render_reproduce_scripts(record or make_record(), env, data, **kwargs)


# --------------------------------------------------------------------------- #
# parsing helpers used by the assertions
# --------------------------------------------------------------------------- #
def strip_to_code(text: str) -> str:
    """Return only what the shell would execute: no comments, no quoted literals.

    A real tokenizer rather than a per-line regex, because a quoted literal may hold a
    newline (that is the whole point of the escaping), because ``#`` inside a path is not
    a comment, and because a ``#`` after a command is.
    """
    body = text.lstrip("\ufeff")
    out: list[str] = []
    index = 0
    length = len(body)
    previous = " "
    while index < length:
        char = body[index]
        if char == "#" and previous.isspace():
            while index < length and body[index] != "\n":
                index += 1
            continue
        if char == "'":
            index += 1
            while index < length and body[index] != "'":
                index += 1
            index += 1
            out.append(" ")
            previous = " "
            continue
        if char == '"':
            index += 1
            while index < length and body[index] != '"':
                if body[index] == "\\":
                    index += 1
                index += 1
            index += 1
            out.append(" ")
            previous = " "
            continue
        out.append(char)
        previous = char
        index += 1
    return "".join(out)


def bash_array(text: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\((.*?)\)\n", text, re.MULTILINE | re.DOTALL)
    assert match, f"{name} array is missing"
    return shlex.split(match.group(1))


def ps_array(text: str, name: str) -> list[str]:
    pattern = rf"^{re.escape(name)} = @\((.*?)\)\r?\n"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    assert match, f"{name} array is missing"
    return [found.replace("''", "'") for found in re.findall(r"'((?:[^']|'')*)'", match.group(1))]


def verdict_lines(text: str) -> list[tuple[str, str]]:
    """(status, name) pairs of every emitted check line in a script.

    shlex is used because a name is only quoted when it needs to be, and because
    ``shlex.quote`` splices ``'"'"'`` for embedded quotes.
    """
    pairs: list[tuple[str, str]] = []
    for line in text.replace("\r\n", "\n").splitlines():
        match = re.match(r"\s*(?:report|Write-LabboxCheck)\s+(PASS|FAIL|UNKNOWN|WARN|NOTE)\s+(.+)", line)
        if not match:
            continue
        rest = match.group(2)
        try:
            words = shlex.split(rest, posix=True)
        except ValueError:  # a literal containing a newline; fall back to the raw head
            words = rest.split()
        name = words[0] if words else rest
        pairs.append((match.group(1), name))
    return pairs


def statuses_for(text: str, name: str) -> set[str]:
    return {status for status, line_name in verdict_lines(text) if line_name == name}


# --------------------------------------------------------------------------- #
# argv escaping
# --------------------------------------------------------------------------- #
class TestArgvEscaping:
    def test_bash_array_round_trips_every_dangerous_argument(self) -> None:
        _ps1, sh = render()
        assert bash_array(sh, "labbox_argv") == DANGEROUS_ARGV

    def test_powershell_array_round_trips_every_dangerous_argument(self) -> None:
        ps1, _sh = render()
        assert ps_array(ps1, "$labboxArgv") == DANGEROUS_ARGV

    def test_every_dangerous_value_is_emitted_as_a_quoted_literal(self) -> None:
        ps1, sh = render()
        for value in DANGEROUS_ARGV:
            assert shlex.quote(value) in sh
            assert shellgen._ps(value) in ps1

    def test_chaining_and_redirection_never_reach_executed_code(self) -> None:
        # && and 2>&1 appear in the script's own prelude; what must never appear outside a
        # literal is a piece of a recorded value.
        ps1, sh = render()
        for text in (ps1, sh):
            code = strip_to_code(text)
            for token in ("&&b", ";reboot", "rm -rf /", "$(rm", "out>file", "x'y"):
                assert token not in code, f"{token!r} escaped the quoting"

    def test_no_double_quote_reaches_a_native_argument(self) -> None:
        # Windows PowerShell 5.1 eats double quotes when it hands an argument to a native
        # command, so the embedded probes must not contain any.
        for probe in (
            shellgen._PROBE_PY_VERSION,
            shellgen._PROBE_PKG_VERSION,
            shellgen._PROBE_MANIFEST_DIGEST,
        ):
            assert '"' not in probe

    def test_command_is_printed_not_executed(self) -> None:
        ps1, sh = render()
        assert "Run this yourself" in sh
        assert "Run this yourself" in ps1
        assert "& $labboxArgv" not in ps1
        for text in (ps1, sh):
            code = strip_to_code(text)
            assert "Invoke-Expression" not in code
            assert " iex" not in code
            assert "eval" not in code
            assert "Invoke-Item" not in code

    def test_empty_argv_degrades_instead_of_printing_an_empty_command(self) -> None:
        record = make_record(command=CommandSpec(raw="not recorded", argv=[], cwd="C:/code/实验 项目"))
        ps1, sh = render(record=record)
        for text in (ps1, sh):
            assert "no argv was recorded" in text
        assert "labbox_argv=(" not in sh
        assert "$labboxArgv = @(" not in ps1


# --------------------------------------------------------------------------- #
# paths, CJK and untrusted values
# --------------------------------------------------------------------------- #
class TestUntrustedValues:
    def test_cjk_path_with_space_is_quoted_in_both_scripts(self) -> None:
        ps1, sh = render()
        assert "recorded_cwd='C:/code/实验 项目'" in sh
        assert "$recordedCwd = 'C:/code/实验 项目'" in ps1

    def test_recorded_values_cannot_open_a_new_script_line(self) -> None:
        hostile_cwd = "/tmp/x'\necho pwned\n'"
        record = make_record(
            command=CommandSpec(raw="python train.py", argv=["python", "train.py"], cwd=hostile_cwd)
        )
        ps1, sh = render(record=record)
        # In bash the value stays one quoted word: reading that word back gives the exact
        # original string, so the embedded newline is data and never a command.
        start = sh.index("recorded_cwd=") + len("recorded_cwd=")
        assert shlex.split(sh[start : start + 200], posix=True)[0] == hostile_cwd
        assert shellgen._ps(hostile_cwd) in ps1  # one literal, quotes doubled inside it
        assert "echo pwned" not in strip_to_code(ps1)
        assert "echo pwned" not in strip_to_code(sh)

    def test_dataset_id_folds_to_an_ascii_variable_name(self) -> None:
        datasets = [
            DatasetEntry(
                id="数据 集-one",
                source="data/one",
                digest="sha256:aa",
                strategy=DatasetStrategy.FAST_MANIFEST,
            )
        ]
        ps1, sh = render(datasets=datasets)
        assert "LABBOX_CHECK_DATASET_1_ONE_DIGEST" in sh
        assert "LABBOX_CHECK_DATASET_1_ONE_DIGEST" in ps1
        assert "$supplied1_ONE" in ps1
        # The id itself still has to reach the manifest lookup unharmed.
        assert "'数据 集-one'" in sh
        assert "$expectedDigest1_ONE = 'sha256:aa'" in ps1

    def test_run_name_appears_escaped_in_header(self) -> None:
        ps1, sh = render()
        assert "基线实验" in sh
        assert "基线实验" in ps1
        # The header is comment lines only.
        for text in (ps1, sh):
            for line in text.splitlines():
                if "基线实验" in line:
                    assert line.lstrip().startswith("#")


# --------------------------------------------------------------------------- #
# line endings, BOM, shell hygiene
# --------------------------------------------------------------------------- #
class TestFileFormats:
    def test_powershell_has_bom_crlf_and_requires_5_1(self) -> None:
        record = make_record(command=CommandSpec(raw="python train.py", argv=["python", "train.py"], cwd="."))
        ps1, _sh = render(record=record)
        assert ps1.startswith(shellgen.UTF8_BOM)
        assert "\n" not in ps1.removeprefix(shellgen.UTF8_BOM).replace("\r\n", "")
        assert ps1.removeprefix(shellgen.UTF8_BOM).splitlines()[0] == "#requires -Version 5.1"
        assert "$ErrorActionPreference = 'Continue'" in ps1

    def test_powershell_keeps_a_recorded_newline_as_data_not_as_a_line_break(self) -> None:
        ps1, _sh = render()
        assert "line\nvalue" in ps1
        assert "line\r\nvalue" not in ps1

    def test_shell_has_shebang_and_lf_only(self) -> None:
        _ps1, sh = render()
        assert sh.startswith("#!/usr/bin/env bash")
        assert "\r" not in sh
        assert "set -euo pipefail" in sh
        assert "set -e\n" not in sh

    def test_header_carries_run_id_generated_at_and_version(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            assert "run_id                    : 20260922-153012-baseline" in text
            assert "generated_at" in text
            assert "generator                 : labbox 0.1.0" in text
            assert "This script only checks and prints." in text

    def test_nothing_installing_is_executed(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            code = strip_to_code(text)
            assert not re.search(r"\bpip3?\b", code), "pip must only be printed"
            assert "Invoke-Expression" not in code
            for forbidden in (
                r"\bapt\b",
                r"\bconda\b",
                r"\bcurl\b",
                r"\bwget\b",
                r"Invoke-WebRequest",
                r"\biwr\b",
                r"\bnuget\b",
            ):
                assert not re.search(forbidden, code), forbidden

    def test_pinned_requirements_are_a_comment_suggestion(self) -> None:
        ps1, sh = render()
        assert "#   torch==2.6.0" in sh
        assert "#   numpy==1.26.4" in sh
        assert "#   transformers==4.44.0" in ps1
        for text in (ps1, sh):
            for line in text.splitlines():
                if "pip install" in line:
                    stripped = line.strip()
                    assert stripped.startswith("#") or re.search(r"(printf|Write-Host)", stripped), (
                        f"pip install must not execute: {line!r}"
                    )

    def test_no_write_redirection_except_dev_null(self) -> None:
        _ps1, sh = render()
        targets = re.findall(r">{1,2}\s*([^\s;)]+)", strip_to_code(sh))
        assert "/dev/null" in targets
        for target in targets:
            assert target in {"/dev/null", "&1", "&2"}, f"the script writes to {target}"

    def test_git_is_only_queried_read_only(self) -> None:
        ps1, sh = render()
        for verb in ("checkout", "reset", "stash", "clean", "apply", "cherry-pick"):
            assert verb not in strip_to_code(ps1)
            assert verb not in strip_to_code(sh)


# --------------------------------------------------------------------------- #
# the checks themselves
# --------------------------------------------------------------------------- #
class TestChecks:
    def test_every_check_prints_a_verdict_and_survives_failure(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            statuses = {status for status, _name in verdict_lines(text)}
            assert {"PASS", "FAIL", "UNKNOWN"} <= statuses
        # A FAIL must not stop the script: failures are printed, the counters are inert.
        assert "set -e" in sh
        assert re.search(r"checks_fail=\$\(\(checks_fail \+ 1\)\)", sh)
        assert "$script:ChecksFail++" in ps1

    def test_git_commit_and_dirty_flag(self) -> None:
        ps1, sh = render()
        # The comparison happens at run time, so all three verdicts are compiled in.
        assert statuses_for(sh, "git_commit") == {"PASS", "FAIL", "UNKNOWN"}
        assert statuses_for(ps1, "git_commit") == {"PASS", "FAIL", "UNKNOWN"}
        assert "expected_commit=a1b2c3d4e5f60718293a4b5c6d7e8f9012345678" in sh
        assert "$expectedCommit = 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'" in ps1
        assert "RECORDED RUN WAS DIRTY" not in strip_to_code(sh)

    def test_dirty_run_warns_loudly_and_keeps_the_diff_hash(self) -> None:
        environment = make_environment(
            git=GitState(
                has_repo=True,
                commit="a1b2c3d4e5f6",
                branch="main",
                dirty=True,
                diff_hash="sha256:0f2b7c",
            )
        )
        ps1, sh = render(environment=environment)
        for text in (ps1, sh):
            assert "WARN" in text
            assert "RECORDED RUN WAS DIRTY" in text
            assert "git_dirty_in_recorded_run" in text
            assert "sha256:0f2b7c" in text
            assert "# WARNING: RECORDED RUN WAS DIRTY" in text
        assert statuses_for(sh, "git_dirty_in_recorded_run") == {"WARN"}

    def test_unknown_dirty_flag_is_not_silently_optimistic(self) -> None:
        environment = make_environment(
            git=GitState(has_repo=True, commit="abc123", branch="main", dirty=None)
        )
        ps1, sh = render(environment=environment)
        for text in (ps1, sh):
            assert "git_dirty_unknown" in text

    def test_package_versions_are_compared_not_assumed(self) -> None:
        ps1, sh = render()
        # torch is in both frameworks and packages, and must be checked exactly once.
        assert "expected_names=" in sh
        names = bash_array(sh, "expected_names")
        assert names.count("torch") == 1
        assert set(names) == {"torch", "numpy", "transformers"}
        assert ps_array(ps1, "$expectedPackages") == names
        # The expected values are escaped the same way the probe escapes what it reads.
        # shlex omits quotes when they are not needed; PowerShell quoting is always on.
        assert "printf '%s' 1.26.4" in sh
        assert "$Name -ceq 'numpy'" in ps1

    def test_empty_package_section_degrades_to_unknown(self) -> None:
        environment = make_environment(frameworks={}, packages={}, absent=["frameworks", "packages"])
        ps1, sh = render(environment=environment)
        for text in (ps1, sh):
            assert "no framework or package versions were captured" in text
        assert "expected_names=(" not in sh
        assert "$expectedPackages" not in ps1

    def test_missing_datasets_are_reported_as_a_gap(self) -> None:
        ps1, sh = render(datasets=None)
        for text in (ps1, sh):
            assert "no dataset manifest was recorded" in text
        assert statuses_for(sh, "datasets") == {"UNKNOWN"}

    def test_fast_manifest_is_declared_not_a_content_proof(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            assert "strategy=fast_manifest is NOT a content proof" in text
            assert "byte-identical dataset" in text
            assert "mtime is deliberately excluded" in text
        proof_lines = [name for _status, name in verdict_lines(sh) if name.endswith("_proof")]
        assert "dataset_1_CIFAR100_TRAIN_proof" in proof_lines
        assert "dataset_2_IMAGENET_VAL_proof" in proof_lines
        assert statuses_for(sh, "dataset_1_CIFAR100_TRAIN_proof") == {"WARN"}

    def test_full_content_dataset_is_not_flagged_as_unproven(self) -> None:
        _ps1, sh = render()
        sh_proof = [line for line in sh.splitlines() if "dataset_2_IMAGENET_VAL_proof" in line]
        assert len(sh_proof) == 1
        assert "report NOTE" in sh_proof[0]
        assert "NOT a content proof" not in sh_proof[0]
        assert "report WARN" not in sh_proof[0]

    def test_dataset_digest_is_never_computed_by_the_script(self) -> None:
        ps1, sh = render()
        code = strip_to_code(sh) + strip_to_code(ps1)
        for forbidden in ("sha256sum", "openssl dgst", "Get-FileHash", "md5sum", "hashlib"):
            assert forbidden not in code

    def test_dataset_digest_is_sought_from_the_user(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            assert "LABBOX_CHECK_DATASET_1_CIFAR100_TRAIN_DIGEST" in text
            assert "dataset-manifest.json" in text
            assert "is the expected value, not evidence" in text

    def test_dataset_without_a_digest_is_unknown(self) -> None:
        datasets = [
            DatasetEntry(
                id="declared-only",
                source="hf:whatever",
                strategy=DatasetStrategy.DECLARED,
                digest=None,
            )
        ]
        ps1, sh = render(datasets=datasets)
        for text in (ps1, sh):
            assert "no digest was recorded for declared-only" in text

    def test_workdir_check_runs_against_the_recorded_path(self) -> None:
        ps1, sh = render()
        assert 'if [ -d "$recorded_cwd" ]' in sh
        assert "Test-Path -LiteralPath $recordedCwd -PathType Container" in ps1
        assert statuses_for(sh, "workdir") == {"PASS", "FAIL", "UNKNOWN"}

    def test_hardware_checks_use_nvidia_smi_read_only(self) -> None:
        ps1, sh = render()
        assert "--query-gpu=name,driver_version" in sh
        assert "--query-gpu=name,driver_version" in ps1
        assert "expected_driver=552.22" in sh
        assert "$expectedDriver = '552.22'" in ps1
        assert "gpu_expected_1='NVIDIA GeForce RTX 3090'" in sh
        assert "$gpuExpected1 = 'NVIDIA GeForce RTX 3090'" in ps1

    def test_missing_gpu_section_never_looks_like_a_match(self) -> None:
        environment = make_environment(absent=["gpus", "cuda.driver"])
        ps1, sh = render(environment=environment)
        for text in (ps1, sh):
            assert "gpu_expected_1" not in text
            assert "gpuExpected1" not in text.replace("gpu_expected_1", "")
        assert statuses_for(sh, "cuda_driver") == {"UNKNOWN"}
        assert statuses_for(sh, "hardware") == set()


# --------------------------------------------------------------------------- #
# capture gaps: the "unknown is not a match" rule
# --------------------------------------------------------------------------- #
class TestCaptureGaps:
    def test_absent_sections_are_listed_as_unknown_at_the_end(self) -> None:
        environment = make_environment(absent=["git.commit", "gpus", "python.version"])
        ps1, sh = render(environment=environment)
        for text in (ps1, sh):
            for item in ("git.commit", "gpus", "python.version"):
                assert f"{item}: listed in environment.absent" in text
            assert "unknown, not matched" in text

    def test_absent_section_with_stale_data_cannot_produce_a_pass(self) -> None:
        # git.commit is absent, yet a commit string is present in the snapshot.
        environment = make_environment(
            git=GitState(has_repo=True, commit="abc123", branch="main", dirty=False),
            absent=["git.commit"],
        )
        ps1, sh = render(environment=environment)
        assert statuses_for(sh, "git_commit") == {"UNKNOWN"}
        assert "expected_commit=" not in sh
        assert "$expectedCommit =" not in ps1

    def test_absent_python_version_blocks_the_comparison(self) -> None:
        environment = make_environment(absent=["python.version"])
        _ps1, sh = render(environment=environment)
        assert statuses_for(sh, "python_version") == {"UNKNOWN"}
        assert "actual_python" in sh  # the probe is still there, the verdict is not

    def test_nothing_absent_says_so_instead_of_staying_silent(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            assert "environment.absent is empty" in text

    def test_collector_status_is_echoed(self) -> None:
        ps1, sh = render()
        for text in (ps1, sh):
            assert "python_packages=partial" in text
            assert "git=ok" in text

    def test_a_totally_missing_snapshot_still_renders(self) -> None:
        record = make_record()
        ps1, sh = render_reproduce_scripts(record, None, None)
        for text in (ps1, sh):
            assert "no environment.json" in text
            assert "UNKNOWN" in text


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
class TestWriteReproduceScripts:
    @pytest.fixture
    def store(self, project: object) -> FilesystemStore:
        return FilesystemStore(project)  # type: ignore[arg-type]

    @pytest.fixture
    def paths(self, store: FilesystemStore) -> RunPaths:
        run_paths = store.layout.run_dir("20260922-153012-baseline")
        store.ensure_run_dirs(run_paths)
        return run_paths

    def test_both_files_are_written_and_reported(self, store: FilesystemStore, paths: RunPaths) -> None:
        written = write_reproduce_scripts(store, paths, make_record(), make_environment(), make_manifest())
        assert written == [PS1_NAME, SH_NAME]
        assert paths.reproduce_ps1.is_file()
        assert paths.reproduce_sh.is_file()

    def test_written_bytes_keep_their_encodings(self, store: FilesystemStore, paths: RunPaths) -> None:
        write_reproduce_scripts(store, paths, make_record(), make_environment(), None)
        ps1_bytes = paths.reproduce_ps1.read_bytes()
        sh_bytes = paths.reproduce_sh.read_bytes()
        assert ps1_bytes.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" in ps1_bytes
        assert "基线实验".encode() in ps1_bytes
        assert b"\r" not in sh_bytes
        assert sh_bytes.startswith(b"#!/usr/bin/env bash")

    def test_optional_inputs_never_raise(self, store: FilesystemStore, paths: RunPaths) -> None:
        assert write_reproduce_scripts(store, paths, make_record(), None, None) == [
            PS1_NAME,
            SH_NAME,
        ]

    def test_write_failure_is_a_structured_error(
        self, store: FilesystemStore, paths: RunPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_path: Path, _text: str) -> None:
            raise LabBoxError("disk on fire", code="STORE_WRITE_FAILED", component="store")

        monkeypatch.setattr(store, "write_text_atomic", boom)
        with pytest.raises(LabBoxError) as excinfo:
            write_reproduce_scripts(store, paths, make_record(), None, None)
        assert excinfo.value.code == "REPRODUCE_SCRIPT_WRITE_FAILED"
        assert excinfo.value.component == "reproduce"

    def test_project_root_rel_is_embedded_and_quoted(self, store: FilesystemStore, paths: RunPaths) -> None:
        _ps1, sh = render(project_root_rel="../..")
        assert "project_root_rel=../.." in sh
        ps1, _s = render(project_root_rel="../../实验 dir")
        assert "$projectRootRel = '../../实验 dir'" in ps1
        _p, sh2 = render(project_root_rel="../../实验 dir")
        assert "project_root_rel='../../实验 dir'" in sh2
