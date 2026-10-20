"""The reproduce scripts (``reproduce.ps1`` / ``reproduce.sh``) — DEVELOPMENT.md §12.2.

What these scripts are allowed to do: verify the current directory and the tools, restate
the recorded facts, and print the original command. What they are *never* allowed to do:
install anything, install a GPU driver, activate a virtualenv, overwrite the user's
environment, or imply that a passing checklist means bit-identical results. Every check
therefore prints exactly one of ``PASS`` / ``FAIL`` / ``UNKNOWN`` and then execution
continues, and a fact that was never captured is reported as ``UNKNOWN`` — never as a
match (§24, §16).

Escaping is the load-bearing part of this module. Everything interpolated below — argv
items, paths, dataset ids, commit hashes, collector names — is *untrusted* input written
into a file a human will execute (§11.4 applies to scripts exactly as it does to prompts).
Two rules, applied without exception:

* POSIX: every interpolated value goes through :func:`_sh` (``shlex.quote``); the script
  uses LF endings.
* PowerShell 5.1: every interpolated value becomes a single-quoted literal with ``'``
  doubled (:func:`_ps`); native tools run through the call operator ``&`` and
  ``Invoke-Expression`` never appears anywhere. The text is emitted with a UTF-8 BOM and
  CRLF endings, because Windows PowerShell 5.1 otherwise decodes a BOM-less file as ANSI
  and turns Chinese paths into mojibake.

The generated code is built from static templates plus quoted literals. Values never
become code: the recorded argv is stored as an array and re-quoted at run time, so an
argument containing ``&&`` or a quote cannot turn into an executed command.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from labbox import __version__
from labbox.domain.common import now_local
from labbox.domain.dataset import DatasetEntry, DatasetManifest
from labbox.domain.environment import EnvironmentSnapshot
from labbox.domain.run import RunRecord
from labbox.errors import LabBoxError
from labbox.infrastructure.store import FilesystemStore
from labbox.paths import RunPaths

PS1_NAME = "reproduce.ps1"
SH_NAME = "reproduce.sh"

#: Windows PowerShell 5.1 needs the BOM to read the CJK text in these files as UTF-8.
UTF8_BOM = "﻿"

MANIFEST_NAME = "dataset-manifest.json"

#: Characters a shell can leave unquoted without changing meaning. Identical in both
#: languages so the two scripts print the same command line.
_SAFE_WORD_CHARS = "A-Za-z0-9_./:@+=-"

# --- probe programs -------------------------------------------------------------
# Read-only Python snippets embedded in both scripts. Constraint that is easy to miss:
# they may contain single quotes but **no double quotes**, because Windows PowerShell 5.1
# strips unescaped double quotes when it hands an argument to a native command, so a
# ``"utf-8"`` here would reach python as bare ``utf-8`` and blow up on the syntax error.
# Bash is fine either way (shlex.quote handles embedded quotes), single quotes satisfy
# both. Every dynamic value arrives as a separate argv item, never as source code.
_PROBE_PY_VERSION = "import sys; sys.stdout.write('.'.join([str(part) for part in sys.version_info[:3]]))"
_PROBE_PKG_VERSION = (
    "import sys, importlib.metadata as m;"
    " v = str(m.version(sys.argv[1]));"
    " esc = {chr(37): chr(37) + chr(112), chr(10): chr(37) + chr(110),"
    " chr(13): chr(37) + chr(114), chr(9): chr(37) + chr(116)};"
    " [sys.stdout.write(esc.get(ch, ch)) for ch in v]"
)
_PROBE_MANIFEST_DIGEST = (
    "import json, sys;"
    " data = json.load(open(sys.argv[1], encoding='utf-8-sig'));"
    " rows = data.get('datasets') or [];"
    " byid = next((row for row in rows if str(row.get('id')) == sys.argv[2]), None);"
    " pos = int(sys.argv[3]) if sys.argv[3].isdigit() else 0;"
    " row = byid if byid is not None else (rows[pos - 1] if 0 < pos <= len(rows) else None);"
    " kind = 'id' if byid is not None else ('position' if row is not None else 'none');"
    " sys.stdout.write(kind + ' ' + str((row or {}).get('digest') or ''))"
)

# --- wording shared by both scripts --------------------------------------------
_NO_MUTATION = (
    "This script installs nothing, downloads nothing, activates nothing and never writes "
    "to your environment or to your GPU drivers."
)
_BITWISE = "LabBox does not claim bitwise reproducibility, and no checklist can restore it."
_CLAIM_LIMIT = (
    "All-PASS below does NOT mean the result will be identical; a FAIL does not mean "
    "reproduction is impossible. This is evidence, not a verdict."
)
_DIRTY_WARN = (
    "RECORDED RUN WAS DIRTY: uncommitted changes existed in the working tree and they are "
    "NOT part of any commit. Commit identity alone cannot reconstruct this run."
)
_FAST_MANIFEST = (
    "strategy=fast_manifest is NOT a content proof: the digest covers relative path + size "
    "+ the first and last 64 bytes of each file, and mtime is deliberately excluded. A "
    "matching digest means 'same paths, sizes and sampled content', not 'byte-identical "
    "dataset'."
)
_DIGEST_HOWTO = (
    "This script computes no digests and walks no files: it can only compare a digest you "
    "supply against the recorded one. Set LABBOX_CHECK_DATASET_<N>_<ID>_DIGEST=<digest> "
    "(the exact name is in the per-dataset lines below), or put the dataset-manifest.json "
    "written by your own run in the current directory. The copy inside the run directory is "
    "the expected value, not evidence, so it is never read back."
)
_NO_DATASETS = (
    "no dataset manifest was recorded for this run, so data identity is unverifiable — that "
    "is a gap in the evidence, not a pass"
)
_TOTALS_NOTE = "PASS/FAIL above counts recorded facts, not reproducibility; UNKNOWN means the evidence was missing"

# --- static prelude, POSIX ------------------------------------------------------
_SH_PRELUDE = r"""set -euo pipefail

# 'set -e' stays on, deliberately, and these are the rules that make it safe: every probe
# ends in '|| true', every test sits in an 'if' condition where a non-zero status is
# expected, and 'report' always returns 0. The consequence is the point: a FAIL prints and
# the script keeps going.

checks_pass=0
checks_fail=0
checks_unknown=0
checks_warn=0

report() {
  # report STATUS NAME DETAIL... — the only place a verdict is printed.
  st="$1"; nm="$2"; shift 2
  case "$st" in
    PASS) printf '[PASS]    %s: %s\n' "$nm" "$*"; checks_pass=$((checks_pass + 1)) ;;
    FAIL) printf '[FAIL]    %s: %s\n' "$nm" "$*"; checks_fail=$((checks_fail + 1)) ;;
    WARN) printf '[WARN ]   %s: %s\n' "$nm" "$*"; checks_warn=$((checks_warn + 1)) ;;
    NOTE) printf '[NOTE ]   %s: %s\n' "$nm" "$*" ;;
    *) printf '[UNKNOWN] %s: %s\n' "$nm" "$*"; checks_unknown=$((checks_unknown + 1)) ;;
  esac
  return 0
}

section() {
  printf '\n== %s ==\n' "$1"
  return 0
}

has_literal() {
  # $1 = haystack, $2 = needle. grep -F keeps a needle containing * or [ literal.
  printf '%s\n' "$1" | grep -F -q -- "$2"
}

count_nonempty_lines() {
  # Line count that does not care what the lines contain.
  labbox_n=0
  while IFS= read -r labbox_line || [ -n "$labbox_line" ]; do
    if [ -n "$labbox_line" ]; then labbox_n=$((labbox_n + 1)); fi
  done <<LABBOX_COUNT_LINES
$1
LABBOX_COUNT_LINES
  printf '%s\n' "$labbox_n"
  return 0
}

sh_quote_word() {
  # Re-quote a recorded word for display. Character by character, so no nested quoting
  # trick has to be trusted, and bash 3.2 behaves like bash 5.
  labbox_out=""
  labbox_index=0
  while [ "$labbox_index" -lt "${#1}" ]; do
    labbox_char="${1:$labbox_index:1}"
    if [ "$labbox_char" = "'" ]; then labbox_char="'\\''"; fi
    labbox_out="$labbox_out$labbox_char"
    labbox_index=$((labbox_index + 1))
  done
  printf "'%s'" "$labbox_out"
  return 0
}

# --- where this script sits, and where the project root is ---------------------
script_src=${BASH_SOURCE[0]:-$0}
run_dir=$(cd -- "$(dirname -- "$script_src")" > /dev/null 2>&1 && pwd || true)
if [ -z "$run_dir" ]; then run_dir=$(pwd); fi
project_root_rel=__PROJECT_ROOT_REL__
project_root=$(cd -- "$run_dir/$project_root_rel" > /dev/null 2>&1 && pwd || true)
if [ -z "$project_root" ] || [ ! -d "$project_root/.labbox" ]; then
  guess=$(cd -- "$run_dir/../../.." > /dev/null 2>&1 && pwd || true)
  if [ -n "$guess" ] && [ -d "$guess/.labbox" ]; then project_root=$guess; fi
fi
if [ -d "$run_dir/.labbox" ]; then project_root=$run_dir; fi

# --- which interpreter to ask. LABBOX_PYTHON wins; nothing is ever installed. --
labbox_find_python() {
  if [ -n "${LABBOX_PYTHON:-}" ]; then
    if command -v -- "$LABBOX_PYTHON" > /dev/null 2>&1; then
      printf '%s\n' "$LABBOX_PYTHON"
      return 0
    fi
  fi
  for labbox_candidate in python3 python; do
    if command -v -- "$labbox_candidate" > /dev/null 2>&1; then
      printf '%s\n' "$labbox_candidate"
      return 0
    fi
  done
  return 1
}
PYBIN=$(labbox_find_python || true)
if [ -n "${LABBOX_PYTHON:-}" ] && ! command -v -- "$LABBOX_PYTHON" > /dev/null 2>&1; then
  report UNKNOWN python_probe "LABBOX_PYTHON is not an executable here; using PATH python"
fi

py_version() {
  if [ -z "$PYBIN" ]; then printf ''; return 0; fi
  "$PYBIN" -c __PROBE_PY_VERSION__ 2>/dev/null || true
}

pkg_version() {
  # $1 = distribution name. Control characters come back escaped (%n %r %t %p), exactly
  # like the expected_* literals below, so '=' stays a byte-for-byte comparison.
  if [ -z "$PYBIN" ]; then printf ''; return 0; fi
  "$PYBIN" -c __PROBE_PKG_VERSION__ "$1" 2>/dev/null || true
}

manifest_digest() {
  # $1 = manifest path, $2 = dataset id, $3 = 1-based position. Prints "<how> <digest>",
  # where <how> is id, position or none. Reads a JSON field; never hashes a file.
  if [ -z "$PYBIN" ]; then printf 'none\n'; return 0; fi
  "$PYBIN" -c __PROBE_MANIFEST_DIGEST__ "$1" "$2" "$3" 2>/dev/null || printf 'none\n'
}

nvidia_report() {
  if ! command -v -- nvidia-smi > /dev/null 2>&1; then printf ''; return 0; fi
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
}
GPUS_HERE=$(nvidia_report)
"""

# --- static prelude, PowerShell -------------------------------------------------
_PS1_PRELUDE = r"""$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
    $OutputEncoding = [System.Text.UTF8Encoding]::new()
} catch {
    # A legacy or redirected console may refuse this; the checks still work.
}

# Read-only probes only. Nothing is installed and no command is assembled from a string:
# native tools run through the call operator with an argument array, and this file never
# uses Invoke-Expression (or its iex alias).
$script:ChecksPass = 0
$script:ChecksFail = 0
$script:ChecksUnknown = 0
$script:ChecksWarn = 0

function Write-LabboxCheck {
    param([string]$Status, [string]$Name, [string]$Detail)
    # Anything that is not an explicit verdict counts as UNKNOWN.
    switch ($Status.ToUpperInvariant()) {
        'PASS' { $script:ChecksPass++; $tag = '[PASS]   ' }
        'FAIL' { $script:ChecksFail++; $tag = '[FAIL]   ' }
        'WARN' { $script:ChecksWarn++; $tag = '[WARN ]  ' }
        'NOTE' { $tag = '[NOTE ]  ' }
        default { $script:ChecksUnknown++; $tag = '[UNKNOWN]' }
    }
    Write-Host ('{0} {1}: {2}' -f $tag, $Name, $Detail)
}

function Write-LabboxSection {
    param([string]$Title)
    Write-Host ''
    Write-Host ('== ' + $Title + ' ==')
}

function Invoke-LabboxTool {
    # Returns standard output as text, or '' when the tool is missing or failed. '' makes
    # the caller report UNKNOWN; a missing tool never looks like a match.
    param([string]$Exe, [string[]]$Arguments = @())
    if ([string]::IsNullOrEmpty($Exe)) { return '' }
    $found = Get-Command -Name $Exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $found) { $found = Get-Command -Name $Exe -ErrorAction SilentlyContinue }
    if (-not $found) { return '' }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Stop'
    $raw = ''
    try {
        $raw = & $found.Source @Arguments 2>$null | Out-String
    } catch {
        $raw = ''
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($null -eq $raw) { return '' }
    return ($raw -replace '[\r\n]+$', '')
}

# --- where this script sits, and where the project root is ---------------------
# .NET path helpers are used instead of Split-Path: Windows PowerShell 5.1 refuses
# -Parent/-Leaf together with -LiteralPath, and .NET never globs a bracket in a name.
$script:LabboxRunDir = ''
try {
    $script:LabboxRunDir = [System.IO.Path]::GetDirectoryName([string]$PSCommandPath)
} catch {
    $script:LabboxRunDir = ''
}
if ([string]::IsNullOrEmpty($script:LabboxRunDir)) {
    $script:LabboxRunDir = (Get-Location).ProviderPath
}
$projectRootRel = __PROJECT_ROOT_REL__
$script:LabboxProjectRoot = ''
try {
    $candidate = Join-Path $script:LabboxRunDir $projectRootRel
    if (Test-Path -LiteralPath (Join-Path $candidate '.labbox') -PathType Container) {
        $script:LabboxProjectRoot = (Resolve-Path -LiteralPath $candidate).Path
    }
    if (-not $script:LabboxProjectRoot) {
        $guess = Join-Path $script:LabboxRunDir '..\..\..'
        if (Test-Path -LiteralPath (Join-Path $guess '.labbox') -PathType Container) {
            $script:LabboxProjectRoot = (Resolve-Path -LiteralPath $guess).Path
        }
    }
} catch {
    $script:LabboxProjectRoot = ''
}
if (Test-Path -LiteralPath (Join-Path $script:LabboxRunDir '.labbox') -PathType Container) {
    $script:LabboxProjectRoot = $script:LabboxRunDir
}

# --- which interpreter to ask. $env:LABBOX_PYTHON wins; nothing is installed. ---
$script:LabboxPython = ''
if ($env:LABBOX_PYTHON) {
    $wanted = [string]$env:LABBOX_PYTHON
    $isPath = $wanted.Contains([System.IO.Path]::DirectorySeparatorChar) -or
        $wanted.Contains([System.IO.Path]::AltDirectorySeparatorChar)
    if ($isPath) {
        if (Test-Path -LiteralPath $wanted -PathType Leaf) { $script:LabboxPython = $wanted }
    } elseif (Get-Command -Name $wanted -CommandType Application -ErrorAction SilentlyContinue) {
        $script:LabboxPython = $wanted
    }
    if (-not $script:LabboxPython) {
        Write-LabboxCheck UNKNOWN 'python_probe' ('LABBOX_PYTHON does not resolve to an ' +
            'executable; falling back to python3/python/py on PATH')
    }
}
if (-not $script:LabboxPython) {
    foreach ($candidate in @('python3', 'python', 'py')) {
        if (Get-Command -Name $candidate -CommandType Application -ErrorAction SilentlyContinue) {
            $script:LabboxPython = $candidate
            break
        }
    }
}
$script:PyPrefix = @()
if ($script:LabboxPython -eq 'py') { $script:PyPrefix = @('-3') }
$script:CodePyVersion = __PROBE_PY_VERSION__
$script:CodePkgVersion = __PROBE_PKG_VERSION__
$script:CodeManifestDigest = __PROBE_MANIFEST_DIGEST__

function Get-LabboxPyVersion {
    if (-not $script:LabboxPython) { return '' }
    $argv = @($script:PyPrefix) + @('-c', $script:CodePyVersion)
    return (Invoke-LabboxTool -Exe $script:LabboxPython -Arguments $argv)
}

function Get-LabboxPkgVersion {
    param([string]$PackageName)
    if (-not $script:LabboxPython) { return '' }
    $argv = @($script:PyPrefix) + @('-c', $script:CodePkgVersion, $PackageName)
    return (Invoke-LabboxTool -Exe $script:LabboxPython -Arguments $argv)
}

function Get-LabboxManifestDigest {
    param([string]$ManifestPath, [string]$DatasetId, [int]$Position)
    if (-not $script:LabboxPython) { return 'none' }
    $argv = @($script:PyPrefix) + @(
        '-c', $script:CodeManifestDigest, $ManifestPath, $DatasetId, [string]$Position)
    $text = Invoke-LabboxTool -Exe $script:LabboxPython -Arguments $argv
    if ([string]::IsNullOrEmpty($text)) { return 'none' }
    return $text
}

function Get-LabboxNonemptyCount {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return 0 }
    return @($Text -split '\r?\n' | Where-Object { $_.Trim().Length -gt 0 }).Count
}

$script:LabboxGpus = ''
if (Get-Command -Name 'nvidia-smi' -CommandType Application -ErrorAction SilentlyContinue) {
    $script:LabboxGpus = Invoke-LabboxTool -Exe 'nvidia-smi' -Arguments @(
        '--query-gpu=name,driver_version', '--format=csv,noheader')
}
"""


# --------------------------------------------------------------------------- #
# quoting primitives
# --------------------------------------------------------------------------- #
def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)


def _sh(value: Any) -> str:
    """Quote *value* as one safe POSIX shell word."""
    return shlex.quote(_text(value))


def _ps(value: Any) -> str:
    """Quote *value* as one safe PowerShell single-quoted literal.

    Single-quoted PowerShell strings are literal: ``$``, backticks, ``&&``, ``|`` and ``"``
    all lose their meaning inside them, so doubling ``'`` is the only escape needed.
    """
    return "'" + _text(value).replace("'", "''") + "'"


_CONTROL_DISPLAY = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _one_line(value: Any, *, limit: int = 200) -> str:
    """Collapse control characters so an untrusted value cannot forge a script line.

    A recorded field should not contain a newline, but nothing prevents a crafted run
    directory from saying so. Inside our quoted literals a raw newline stays data, so this
    is about honest display rather than injection: newlines render as ``\\n``, other
    control characters as ``?``, and over-long values end in an ellipsis.
    """
    chunks: list[str] = []
    for char in _text(value):
        shown = _CONTROL_DISPLAY.get(char)
        if shown is not None:
            chunks.append(shown)
        elif ord(char) < 32 or ord(char) == 127:
            chunks.append("?")
        else:
            chunks.append(char)
    collapsed = "".join(chunks)
    if len(collapsed) > limit:
        return collapsed[:limit] + "…"
    return collapsed


def _escape_key(name: str, *, index: int | None = None) -> str:
    """ASCII-only, symbol-free identifier usable as a shell variable / env var suffix.

    Non-ASCII letters are folded to underscores on purpose: ``数据`` is alphanumeric to
    Python, but a PowerShell environment variable or bash identifier built from it is not
    something worth shipping in a generated script. *index* keeps the name unique when two
    dataset ids fold to the same ASCII skeleton.
    """
    cleaned = "".join(char if char.isascii() and char.isalnum() else "_" for char in name).strip("_").upper() or "ITEM"
    return f"{index}_{cleaned}" if index is not None else cleaned


def _version_code(value: Any) -> str:
    """Escape a version the way the in-script probe escapes what it reads.

    Both sides of a comparison pass through this, so equality still means byte equality of
    the original strings while the values stay single-line.
    """
    text = _text(value)
    for char, code in (("%", "%p"), ("\n", "%n"), ("\r", "%r"), ("\t", "%t")):
        text = text.replace(char, code)
    return text


def _fill(template: str, **values: str) -> list[str]:
    """Replace whole-token placeholders in a static template with quoted literals."""
    out = template
    for token, value in values.items():
        out = out.replace(f"__{token}__", value, 1)
    leftover = [line for line in out.splitlines() if "__" in line and "PROBE" in line]
    if leftover:  # pragma: no cover - guards against a template/placeholder typo
        raise LabBoxError(
            "reproduce template placeholder was not substituted",
            code="REPRODUCE_SCRIPT_RENDER_FAILED",
            component="reproduce",
            details={"lines": leftover[:3]},
        )
    return out.splitlines()


def _prelude(template: str, quote: Callable[[Any], str], facts: _Facts) -> list[str]:
    return _fill(
        template,
        PROJECT_ROOT_REL=quote(facts.project_root_rel),
        PROBE_PY_VERSION=quote(_PROBE_PY_VERSION),
        PROBE_PKG_VERSION=quote(_PROBE_PKG_VERSION),
        PROBE_MANIFEST_DIGEST=quote(_PROBE_MANIFEST_DIGEST),
    )


# --------------------------------------------------------------------------- #
# facts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _GitFacts:
    verifiable: bool
    commit: str
    reason: str
    dirty: bool | None
    diff_hash: str | None
    branch: str | None


@dataclass(frozen=True, slots=True)
class _PythonFacts:
    verifiable: bool
    version: str
    reason: str
    executable: str
    venv_prefix: str


@dataclass(frozen=True, slots=True)
class _PackageFacts:
    groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    reason: str

    @property
    def items(self) -> tuple[tuple[str, str], ...]:
        return tuple(item for _label, rows in self.groups for item in rows)

    @property
    def empty(self) -> bool:
        return not self.items


@dataclass(frozen=True, slots=True)
class _HardwareFacts:
    runtime: str
    driver: str
    cudnn: str
    gpus: tuple[tuple[str, str, str], ...]  # index, name, driver
    reason: str


@dataclass(frozen=True, slots=True)
class _DatasetFacts:
    dataset_id: str
    slug: str
    position: int
    env_key: str
    digest: str
    strategy: str
    file_count: str
    total_bytes: str
    content_proof: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Facts:
    header: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]
    raw_command: str
    cwd: str
    cwd_is_relative: bool
    git: _GitFacts
    python: _PythonFacts
    packages: _PackageFacts
    hardware: _HardwareFacts
    datasets: tuple[_DatasetFacts, ...]
    absent: tuple[str, ...]
    collectors: tuple[tuple[str, str], ...]
    project_root_rel: str
    notes: tuple[str, ...] = ()

    @property
    def check_budget(self) -> int:
        """How many recorded values got compiled into the scripts (a comment in output)."""
        return (
            len(self.packages.items)
            + len(self.datasets)
            + len(self.hardware.gpus)
            + (1 if self.git.verifiable else 0)
            + (1 if self.python.verifiable else 0)
            + (1 if self.cwd else 0)
        )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_reproduce_scripts(
    record: RunRecord,
    environment: EnvironmentSnapshot | None,
    datasets: Sequence[DatasetEntry] | DatasetManifest | None,
    *,
    project_root_rel: str = ".",
) -> tuple[str, str]:
    """Render ``(ps1_text, sh_text)`` for one run.

    *environment* and *datasets* may be ``None`` or empty; the matching checks then decay
    into ``UNKNOWN`` lines instead of disappearing. *project_root_rel* is the project root
    expressed relative to the run directory — ``"."`` means "fall back to resolving
    ``<run dir>/../../..`` when that is where a ``.labbox`` directory lives".

    The PowerShell text starts with :data:`UTF8_BOM` and uses CRLF, so writing it with
    :meth:`FilesystemStore.write_text_atomic` (UTF-8, no encoding translation) produces a
    file Windows PowerShell 5.1 reads correctly, CJK paths included.
    """
    facts = _collect_facts(record, environment, datasets, project_root_rel=project_root_rel)
    return _render_ps1(facts), _render_sh(facts)


def write_reproduce_scripts(
    store: FilesystemStore,
    paths: RunPaths,
    record: RunRecord,
    environment: EnvironmentSnapshot | None,
    datasets: Sequence[DatasetEntry] | DatasetManifest | None,
    *,
    project_root_rel: str = ".",
) -> list[str]:
    """Render both scripts and write them into the run directory.

    Returns the run-relative names written. Rendering is total — a missing optional input
    produces ``UNKNOWN`` output, never an exception. A write that genuinely fails is
    re-raised as :class:`LabBoxError` with ``REPRODUCE_SCRIPT_WRITE_FAILED`` so the caller
    can log it as a warning: §6.1 requires that losing a reproduce script cannot damage
    the run record itself.
    """
    ps1_text, sh_text = render_reproduce_scripts(record, environment, datasets, project_root_rel=project_root_rel)
    written: list[str] = []
    for name, target, text in (
        (PS1_NAME, paths.reproduce_ps1, ps1_text),
        (SH_NAME, paths.reproduce_sh, sh_text),
    ):
        try:
            store.write_text_atomic(target, text)
        except LabBoxError as exc:
            raise LabBoxError(
                f"Could not write {name}: {exc.message}",
                code="REPRODUCE_SCRIPT_WRITE_FAILED",
                component="reproduce",
                details={"path": str(target), "run_id": record.run_id},
            ) from exc
        except OSError as exc:  # pragma: no cover - the store wraps these already
            raise LabBoxError(
                f"Could not write {name}: {exc}",
                code="REPRODUCE_SCRIPT_WRITE_FAILED",
                component="reproduce",
                details={"path": str(target), "run_id": record.run_id},
            ) from exc
        written.append(name)
    return written


# --------------------------------------------------------------------------- #
# fact collection: the only place that decides what is verifiable at all
# --------------------------------------------------------------------------- #
def _collect_facts(
    record: RunRecord,
    environment: EnvironmentSnapshot | None,
    datasets: Sequence[DatasetEntry] | DatasetManifest | None,
    *,
    project_root_rel: str,
) -> _Facts:
    env = environment
    absent = tuple(_one_line(item) for item in (env.absent if env else ()))
    command = record.command

    def not_captured(*paths: str) -> bool:
        """True when a section is listed as absent, so it must never print as a match."""
        return any(item == path or item.startswith(f"{path}.") for path in paths for item in absent)

    notes: list[str] = []
    recorded_version = _one_line(record.labbox_version)
    if recorded_version and recorded_version != _one_line(__version__):
        notes.append(
            f"this run was recorded by labbox {recorded_version}, while these scripts were "
            f"generated by labbox {__version__}"
        )
    if command and command.shell_note:
        notes.append(f"start note: {_one_line(command.shell_note)}")

    return _Facts(
        header=_header(record, env),
        argv=tuple(_text(item) for item in (command.argv if command else ())),
        raw_command=_text(command.raw) if command else "",
        cwd=_text(command.cwd) if command else "",
        cwd_is_relative=bool(command.cwd_is_relative) if command else False,
        git=_collect_git(env, not_captured),
        python=_collect_python(env, not_captured),
        packages=_collect_packages(env, not_captured),
        hardware=_collect_hardware(env, not_captured),
        datasets=_collect_datasets(datasets),
        absent=absent,
        collectors=tuple(
            (_one_line(name), _one_line(status)) for name, status in (env.collectors if env else {}).items()
        ),
        project_root_rel=_text(project_root_rel).strip() or ".",
        notes=tuple(notes),
    )


def _header(record: RunRecord, env: EnvironmentSnapshot | None) -> tuple[tuple[str, str], ...]:
    return (
        ("run_id", _one_line(record.run_id)),
        ("run_name", _one_line(record.name)),
        ("project", _one_line(record.project)),
        ("run_status", _one_line(getattr(record.status, "value", record.status))),
        ("started_at", _iso(record.started_at)),
        ("ended_at", _iso(record.ended_at)),
        ("exit_code", _exit_text(record)),
        ("command_raw", _one_line(record.command.raw) if record.command else "not recorded"),
        ("labbox_version", _one_line(record.labbox_version) or "not recorded"),
        ("generator", f"labbox {__version__} infrastructure.shellgen"),
        ("generated_at", _iso(now_local())),
        ("environment_captured_at", _iso(env.captured_at) if env else "no environment.json"),
    )


def _iso(value: datetime | None) -> str:
    if value is None:
        return "not recorded"
    try:
        return _one_line(value.isoformat(timespec="seconds"))
    except (ValueError, TypeError):  # pragma: no cover - pydantic validates already
        return _one_line(value)


def _exit_text(record: RunRecord) -> str:
    if record.exit_code is None and record.exit_code_raw is None:
        return "not recorded"
    meaning = _text(getattr(record.exit_meaning, "value", record.exit_meaning)) or "unknown"
    return f"{record.exit_code} (raw {record.exit_code_raw}, {meaning})"


def _collect_git(
    env: EnvironmentSnapshot | None,
    not_captured: Callable[..., bool],
) -> _GitFacts:
    git = env.git if env else None
    dirty = getattr(git, "dirty", None)
    diff_hash = _one_line(getattr(git, "diff_hash", "")) or None
    branch = _one_line(getattr(git, "branch", "")) or None
    commit = _one_line(getattr(git, "commit", "") or getattr(git, "short_commit", ""))
    if env is None:
        return _GitFacts(False, "", "no environment.json was supplied", dirty, diff_hash, branch)
    if not_captured("git", "git.commit") or not getattr(git, "has_repo", False):
        return _GitFacts(
            False,
            commit,
            "git state is absent from this snapshot (not a repository, or the collector "
            "failed) — this is unknown, not a match",
            dirty,
            diff_hash,
            branch,
        )
    if not commit:
        return _GitFacts(False, "", "no commit hash was recorded", dirty, diff_hash, branch)
    return _GitFacts(True, commit, "", dirty, diff_hash, branch)


def _collect_python(
    env: EnvironmentSnapshot | None,
    not_captured: Callable[..., bool],
) -> _PythonFacts:
    info = env.python if env else None
    version = _one_line(getattr(info, "version", ""))
    executable = _one_line(getattr(info, "executable", ""))
    venv = _one_line(getattr(info, "venv_prefix", ""))
    kind = _text(getattr(info, "kind", ""))
    if env is None:
        return _PythonFacts(False, version, "no environment.json was supplied", executable, venv)
    if not_captured("python.version"):
        return _PythonFacts(False, version, "python.version is listed in environment.absent", executable, venv)
    if not version:
        reason = "the interpreter version was never recorded"
        if kind == "external":
            reason = (
                "the training command did not run a Python interpreter LabBox could probe "
                "(kind=external), so the interpreter is unknown"
            )
        return _PythonFacts(False, version, reason, executable, venv)
    return _PythonFacts(True, version, "", executable, venv)


def _collect_packages(
    env: EnvironmentSnapshot | None,
    not_captured: Callable[..., bool],
) -> _PackageFacts:
    if env is None:
        return _PackageFacts((), "no environment.json was supplied, so no versions are known")
    groups: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    seen: set[str] = set()
    for label, mapping in (
        ("frameworks", dict(env.frameworks or {})),
        ("packages", dict(env.packages or {})),
    ):
        if not_captured(label):
            continue
        items: list[tuple[str, str]] = []
        for name, value in sorted(mapping.items(), key=lambda kv: _text(kv[0]).lower()):
            key = _one_line(name)
            if not key or key.lower() in seen:
                # frameworks is a subset of packages; one check per distribution.
                continue
            seen.add(key.lower())
            items.append((key, "" if value is None else _one_line(_version_code(value), limit=80)))
        if items:
            groups.append((label, tuple(items)))
    reason = ""
    if not groups:
        reason = (
            "no framework or package versions were captured — they are listed in "
            "environment.absent, so nothing here verifies your dependencies"
        )
    return _PackageFacts(tuple(groups), reason)


def _collect_hardware(
    env: EnvironmentSnapshot | None,
    not_captured: Callable[..., bool],
) -> _HardwareFacts:
    cuda = env.cuda if env else None
    driver = _one_line(getattr(cuda, "driver", ""))
    runtime = _one_line(getattr(cuda, "runtime", ""))
    cudnn = _one_line(getattr(cuda, "cudnn", ""))
    gpus: tuple[tuple[str, str, str], ...] = ()
    if env is not None and not not_captured("gpus"):
        gpus = tuple(
            (
                _one_line(getattr(gpu, "index", "")),
                _one_line(getattr(gpu, "name", "")),
                _one_line(getattr(gpu, "driver", "")),
            )
            for gpu in env.gpus
        )
    if env is not None:
        if not_captured("cuda.driver"):
            driver = ""
        if not_captured("cuda.runtime"):
            runtime = ""
    reason = "no environment.json was supplied"
    if env is not None and not (driver or runtime or gpus):
        reason = (
            "CUDA and GPU state were not captured — a missing GPU record is unknown, not evidence that no GPU was used"
        )
    return _HardwareFacts(runtime=runtime, driver=driver, cudnn=cudnn, gpus=gpus, reason=reason)


def _collect_datasets(
    datasets: Sequence[DatasetEntry] | DatasetManifest | None,
) -> tuple[_DatasetFacts, ...]:
    rows: Iterable[Any]
    if isinstance(datasets, DatasetManifest):
        rows = datasets.datasets
    elif datasets is None:
        rows = ()
    else:
        rows = datasets
    facts: list[_DatasetFacts] = []
    for index, entry in enumerate(rows, start=1):
        dataset_id = _one_line(getattr(entry, "id", "")) or f"dataset-{index}"
        slug = _escape_key(dataset_id, index=index)
        facts.append(
            _DatasetFacts(
                dataset_id=dataset_id,
                slug=slug,
                position=index,
                env_key=f"LABBOX_CHECK_DATASET_{slug}_DIGEST",
                digest=_one_line(getattr(entry, "digest", "")),
                strategy=_one_line(getattr(entry, "strategy", "")) or "unknown",
                file_count=_count_text(getattr(entry, "file_count", None)),
                total_bytes=_count_text(getattr(entry, "total_bytes", None)),
                content_proof=bool(getattr(entry, "is_content_proof", False)),
                warnings=tuple(_one_line(w) for w in (getattr(entry, "warnings", ()) or ())),
            )
        )
    return tuple(facts)


def _count_text(value: Any) -> str:
    return "not recorded" if value is None else _one_line(value)


# --------------------------------------------------------------------------- #
# renderers: bash
# --------------------------------------------------------------------------- #
def _render_sh(facts: _Facts) -> str:
    out: list[str] = ["#!/usr/bin/env bash"]
    out.extend(_comment_block(facts, SH_NAME))
    out.append("")
    out.extend(_prelude(_SH_PRELUDE, _sh, facts))

    # --- code: git ---------------------------------------------------------
    out.append('section "code: git commit"')
    out.extend(_git_warnings(facts, is_ps=False))
    if not facts.git.verifiable:
        out.extend(_emit_sh("UNKNOWN", "git_commit", facts.git.reason))
    else:
        out.append(f"expected_commit={_sh(facts.git.commit)}")
        out.extend(
            [
                "if ! command -v -- git > /dev/null 2>&1",
                "then",
                *_emit_sh(
                    "UNKNOWN",
                    "git_commit",
                    "git is not installed on this machine, so HEAD cannot be compared",
                ),
                'elif [ -z "$project_root" ] || [ ! -d "$project_root/.git" ]',
                "then",
                *_emit_sh(
                    "UNKNOWN",
                    "git_commit",
                    "no .git directory was found under the project root resolved from this "
                    "script, so HEAD cannot be compared",
                ),
                "else",
                '  head_commit=$(git -C "$project_root" rev-parse HEAD 2>/dev/null || true)',
                '  if [ -z "$head_commit" ]',
                "  then",
                *_emit_sh(
                    "UNKNOWN",
                    "git_commit",
                    "git rev-parse HEAD returned nothing in the project root resolved from this script",
                ),
                '  elif [ "$head_commit" = "$expected_commit" ] '
                '|| [ "${head_commit#"$expected_commit"}" != "$head_commit" ]',
                "  then",
                *_emit_sh(
                    "PASS",
                    "git_commit",
                    "HEAD matches the recorded commit (a short recorded hash is accepted as a prefix)",
                ),
                "  else",
                *_emit_sh(
                    "FAIL",
                    "git_commit",
                    "HEAD does not match the recorded commit; compare 'git rev-parse HEAD' "
                    "with the run_id in the header of this script",
                ),
                "  fi",
                "fi",
            ]
        )

    # --- interpreter -------------------------------------------------------
    out.append('section "interpreter: python version"')
    out.append(f"expected_python={_sh(facts.python.version)}")
    out.extend(
        [
            "actual_python=$(py_version || true)",
            'if [ -z "$actual_python" ]',
            "then",
            *_emit_sh(
                "UNKNOWN",
                "python_version",
                "no python3/python on PATH: set LABBOX_PYTHON to the interpreter you want "
                "checked. This script installed nothing.",
            ),
        ]
    )
    if not facts.python.verifiable:
        out.extend([*_emit_sh("UNKNOWN", "python_version", facts.python.reason), "fi"])
    else:
        out.extend(
            [
                'elif [ "$actual_python" = "$expected_python" ]',
                "then",
                *_emit_sh(
                    "PASS",
                    "python_version",
                    "the interpreter this script asked reports the recorded version",
                ),
                "else",
                *_emit_sh(
                    "FAIL",
                    "python_version",
                    "the interpreter this script asked reports a different version than the run recorded",
                ),
                "fi",
            ]
        )
    out.extend(_venv_block_sh(facts))

    # --- dependencies ------------------------------------------------------
    out.append('section "dependencies: frameworks and packages"')
    if facts.packages.empty:
        out.extend(_emit_sh("UNKNOWN", "packages", facts.packages.reason))
    else:
        names = " ".join(_sh(name) for name, _version in facts.packages.items)
        out.append(f"expected_names=( {names} )")
        out.append("expected_version_for() {")
        out.append('  case "$1" in')
        for name, version in facts.packages.items:
            out.append(f"    {_sh(name)}) printf '%s' {_sh(version)} ;;")
        out.append("    *) printf '' ;;")
        out.append("  esac")
        out.append("  return 0")
        out.append("}")
        out.extend(
            [
                'if [ -z "$PYBIN" ]',
                "then",
                *_emit_sh(
                    "UNKNOWN",
                    "packages",
                    f"{len(facts.packages.items)} versions were recorded but no interpreter "
                    "is available to ask; the recorded list is printed as comments below",
                ),
                "else",
                '  for package_name in "${expected_names[@]}"',
                "  do",
                '    want=$(expected_version_for "$package_name" || true)',
                '    have=$(pkg_version "$package_name" || true)',
                '    if [ -z "$want" ]',
                "    then",
                '      report NOTE "package $package_name" ' + _sh("recorded as installed with an unknown version"),
                '    elif [ -z "$have" ]',
                "    then",
                '      report FAIL "package $package_name" '
                + _sh("not found in the interpreter this script asked; it may live in another virtualenv"),
                '    elif [ "$have" = "$want" ]',
                "    then",
                '      report PASS "package $package_name" ' + _sh("installed version matches the record"),
                "    else",
                '      report FAIL "package $package_name" '
                + _sh("installed version differs from the recorded version"),
                "    fi",
                "  done",
                "fi",
            ]
        )
    out.extend(_requirements_block_sh(facts))

    # --- hardware ----------------------------------------------------------
    out.append('section "hardware: CUDA driver and GPU model"')
    out.extend(_hardware_block_sh(facts))

    # --- data --------------------------------------------------------------
    out.append('section "data: dataset digests"')
    out.extend(_emit_sh("NOTE", "dataset_method", _DIGEST_HOWTO))
    if not facts.datasets:
        out.extend(_emit_sh("UNKNOWN", "datasets", _NO_DATASETS))
    for dataset in facts.datasets:
        out.extend(_dataset_block_sh(dataset))

    # --- workspace ---------------------------------------------------------
    out.append('section "workspace: recorded working directory"')
    out.extend(_workdir_block_sh(facts))

    # --- capture gaps ------------------------------------------------------
    out.append('section "capture status and gaps"')
    out.extend(_collector_lines_sh(facts))
    out.extend(_absent_lines_sh(facts))
    for note in facts.notes:
        out.extend(_emit_sh("NOTE", "note", note))

    # --- the command -------------------------------------------------------
    out.append('section "the command to run"')
    out.extend(_command_block_sh(facts))
    out.extend(_footer_sh(facts))
    return "\n".join(out) + "\n"


def _emit_sh(status: str, name: str, detail: str, indent: str = "") -> list[str]:
    return [f"{indent}report {status} {_sh(name)} {_sh(detail)}"]


def _dataset_block_sh(dataset: _DatasetFacts) -> list[str]:
    key = dataset.slug
    var = f"supplied_digest_{key}"
    source = f"digest_source_{key}"
    label = f"dataset_{key}"
    out = [
        f"expected_digest_{key}={_sh(dataset.digest)}",
        f"recorded_strategy_{key}={_sh(dataset.strategy)}",
        *_emit_sh(
            "NOTE",
            f"{label}_identity",
            f"dataset {dataset.dataset_id}: strategy={dataset.strategy}, "
            f"file_count={dataset.file_count}, total_bytes={dataset.total_bytes}",
        ),
    ]
    if dataset.content_proof:
        out.extend(
            _emit_sh(
                "NOTE",
                f"{label}_proof",
                f"strategy={dataset.strategy} was recorded as a full-content or externally pinned version",
            )
        )
    else:
        proof = (
            _FAST_MANIFEST
            if dataset.strategy == "fast_manifest"
            else f"strategy={dataset.strategy} is NOT a content proof of the dataset bytes"
        )
        out.extend(_emit_sh("WARN", f"{label}_proof", proof))
    for warning in dataset.warnings:
        out.extend(_emit_sh("WARN", f"{label}_warning", warning))
    if not dataset.digest:
        out.extend(
            _emit_sh(
                "UNKNOWN",
                label,
                f"no digest was recorded for {dataset.dataset_id}, so data identity cannot be checked at all",
            )
        )
        return out
    out.extend(
        [
            f'{var}="${{{dataset.env_key}:-}}"',
            f"{source}=''",
            f'if [ -n "${var}" ]',
            "then",
            f"  {source}=" + _sh(f"environment variable {dataset.env_key}"),
            f'elif [ -f "$PWD/{MANIFEST_NAME}" ] && [ "$(cd -- "$PWD" && pwd)" != "$run_dir" ]',
            "then",
            "  # The copy in the run directory is the expected value, so it is not read.",
            f'  probe=$(manifest_digest "$PWD/{MANIFEST_NAME}" '
            + _sh(dataset.dataset_id)
            + f" {dataset.position} || true)",
            "  kind=${probe%% *}",
            f"  {var}=${{probe#* }}",
            '  if [ "$kind" = none ] || [ -z "$kind" ]; then',
            f"    {var}=''",
            "  else",
            f'    {source}="dataset manifest match by $kind (not by this script)"',
            "  fi",
            "fi",
            f'if [ -z "${var}" ]',
            "then",
            *_emit_sh(
                "UNKNOWN",
                label,
                f"no digest available to compare for {dataset.dataset_id}; the recorded one is {dataset.digest}",
            ),
            f'elif [ "${var}" = "$expected_digest_{key}" ]',
            "then",
            *_emit_sh(
                "PASS",
                label,
                f"digest of {dataset.dataset_id} matches the recorded one (the digest came "
                "from you, not from this script)",
            ),
            "else",
            *_emit_sh(
                "FAIL",
                label,
                f"digest of {dataset.dataset_id} differs from the recorded one; the data you "
                "are about to use is not the data this run consumed",
            ),
            "fi",
            f'if [ -n "${source}" ]',
            "then",
            f'report NOTE {_sh(f"{label}_source")} "${source}"',
            "fi",
        ]
    )
    return out


def _hardware_block_sh(facts: _Facts) -> list[str]:
    hw = facts.hardware
    if not (hw.driver or hw.runtime or hw.gpus):
        return _emit_sh("UNKNOWN", "hardware", hw.reason)
    out: list[str] = []
    if hw.runtime:
        out.extend(
            _emit_sh(
                "NOTE",
                "cuda_runtime",
                f"the run recorded CUDA runtime {hw.runtime}, which is a property of "
                "the torch build, not of this machine's driver",
            )
        )
    if hw.cudnn:
        out.extend(_emit_sh("NOTE", "cudnn", f"the run recorded cuDNN {hw.cudnn}"))
    out.extend(
        [
            'if [ -z "$GPUS_HERE" ]',
            "then",
            *_emit_sh(
                "UNKNOWN",
                "gpu",
                "nvidia-smi produced no output here: either there is no NVIDIA driver "
                "tooling, or it failed. That is unknown, not proof that no GPU was used, and "
                "this script will not install a driver.",
            ),
            "else",
        ]
    )
    if hw.driver:
        out.extend(
            [
                f"  expected_driver={_sh(hw.driver)}",
                "  first_gpu_line=$(printf '%s\\n' \"$GPUS_HERE\" | head -n 1 || true)",
                "  actual_driver=${first_gpu_line##*,}",
                "  actual_driver=${actual_driver# }",
                '  if [ "$actual_driver" = "$expected_driver" ]',
                "  then",
                *_emit_sh("PASS", "cuda_driver", "the driver reported here matches the record"),
                "  else",
                *_emit_sh("FAIL", "cuda_driver", "the driver here differs from the record"),
                "  fi",
            ]
        )
    else:
        out.extend(_emit_sh("UNKNOWN", "cuda_driver", "no driver version was recorded for this run"))
    if hw.gpus:
        out.append(f"  expected_gpu_count={len(hw.gpus)}")
        out.append('  actual_gpu_count=$(count_nonempty_lines "$GPUS_HERE" || true)')
        out.extend(
            [
                '  if [ "$actual_gpu_count" = "$expected_gpu_count" ]',
                "  then",
                *_emit_sh("PASS", "gpu_count", "this machine reports as many GPUs as the run used"),
                "  else",
                *_emit_sh(
                    "FAIL",
                    "gpu_count",
                    "this machine reports a different number of GPUs than the run used",
                ),
                "  fi",
            ]
        )
        for position, gpu_index, name, driver in _gpu_rows(facts):
            out.append(f"  gpu_expected_{position}={_sh(name)}")
            out.extend(
                [
                    f'  if has_literal "$GPUS_HERE" "$gpu_expected_{position}"',
                    "  then",
                    *_emit_sh(
                        "PASS",
                        f"gpu_{position}",
                        f"the GPU model recorded at index {gpu_index} is present here",
                    ),
                    "  else",
                    *_emit_sh(
                        "FAIL",
                        f"gpu_{position}",
                        f"the GPU model recorded at index {gpu_index} is not in this machine's nvidia-smi output",
                    ),
                    "  fi",
                ]
            )
            if driver:
                out.extend(_emit_sh("NOTE", f"gpu_{position}_driver", f"the run saw driver {driver} here"))
        out.extend(
            _emit_sh(
                "NOTE",
                "gpu_identity",
                "GPU models are matched by name only: UUID, board position, clocks and "
                "NVLink topology are not compared here",
            )
        )
    else:
        out.extend(_emit_sh("UNKNOWN", "gpu_model", "no GPU model was recorded, so there is nothing to compare"))
    out.append("fi")
    return out


def _workdir_block_sh(facts: _Facts) -> list[str]:
    if not facts.cwd:
        return _emit_sh(
            "UNKNOWN",
            "workdir",
            "no cwd was recorded with the command, so it cannot be located",
        )
    out = [
        f"recorded_cwd={_sh(facts.cwd)}",
        'if [ -d "$recorded_cwd" ]',
        "then",
        *_emit_sh("PASS", "workdir", "the recorded working directory exists as written"),
        'elif [ -n "$project_root" ] && [ -d "$project_root/$recorded_cwd" ]',
        "then",
        *_emit_sh(
            "PASS",
            "workdir",
            "the recorded cwd resolves under the project root (it was relative)",
        ),
        "else",
        '  leaf=$(basename -- "$recorded_cwd")',
        '  if [ -n "$project_root" ] && [ -d "$project_root/$leaf" ]',
        "  then",
        *_emit_sh(
            "UNKNOWN",
            "workdir",
            "the recorded cwd does not exist, but a directory with the same name sits under "
            "the project root: the project may have been moved",
        ),
        "  else",
        *_emit_sh(
            "FAIL",
            "workdir",
            "the recorded working directory does not exist, and no obvious relocation of it was found",
        ),
        "  fi",
        "fi",
        'workdir_word=$(sh_quote_word "$recorded_cwd" || true)',
        *_emit_sh(
            "NOTE",
            "workdir_usage",
            "cd into the directory above before running the command at the end of this "
            "script; this script does not change your working directory permanently",
        ),
    ]
    if facts.cwd_is_relative:
        out.extend(_emit_sh("NOTE", "workdir_kind", "the recorded cwd was relative to the project root"))
    return out


def _collector_lines_sh(facts: _Facts) -> list[str]:
    out: list[str] = []
    for name, status in facts.collectors:
        out.extend(_emit_sh("NOTE", f"collector_{_escape_key(name)}", f"{name}={status}"))
    if not facts.collectors:
        out.extend(
            _emit_sh(
                "NOTE",
                "collectors",
                "no collector status was recorded, so it is unknown which sections were attempted",
            )
        )
    return out


def _absent_lines_sh(facts: _Facts) -> list[str]:
    if not facts.absent:
        return _emit_sh(
            "NOTE",
            "not_captured",
            "environment.absent is empty: every section in this snapshot was attempted",
        )
    return [
        line
        for item in facts.absent
        for line in _emit_sh(
            "UNKNOWN",
            "not_captured",
            f"{item}: listed in environment.absent, so it is unknown, not matched",
        )
    ]


def _command_block_sh(facts: _Facts) -> list[str]:
    if not facts.argv:
        return [
            *_emit_sh(
                "UNKNOWN",
                "command",
                "no argv was recorded; the raw command line printed below is text only and "
                "must not be treated as a command",
            ),
            "printf '%s\\n' '' " + _sh(facts.raw_command or "not recorded"),
        ]
    items = " ".join(_sh(item) for item in facts.argv)
    out = [
        f"labbox_argv=( {items} )",
        "labbox_command_line=''",
        'for labbox_item in "${labbox_argv[@]}"',
        "do",
        "  case $labbox_item in",
        '    "" | *[!' + _SAFE_WORD_CHARS + "]*)",
        '      labbox_word=$(sh_quote_word "$labbox_item" || true)',
        "      ;;",
        "    *)",
        "      labbox_word=$labbox_item",
        "      ;;",
        "  esac",
        '  if [ -n "$labbox_command_line" ]; then',
        '    labbox_command_line="$labbox_command_line $labbox_word"',
        "  else",
        "    labbox_command_line=$labbox_word",
        "  fi",
        "done",
        "printf '%s\\n' '' " + _sh("Run this yourself - LabBox recorded it and does not start it for you:"),
    ]
    if facts.cwd:
        out.append('printf \'  cd %s && %s\\n\' "$workdir_word" "$labbox_command_line"')
    out.append("printf '  %s\\n' \"$labbox_command_line\"")
    return out


def _footer_sh(facts: _Facts) -> list[str]:
    return [
        "printf '%s\\n' '' " + _sh("--- summary ---"),
        f'report NOTE totals "$checks_pass PASS, $checks_fail FAIL, '
        f'$checks_unknown UNKNOWN, $checks_warn WARN: {_TOTALS_NOTE}"',
        *_emit_sh("NOTE", "limits", _CLAIM_LIMIT),
        *_emit_sh("NOTE", "limits", _NO_MUTATION),
        *_emit_sh("NOTE", "limits", _BITWISE),
        'if [ "$checks_fail" -gt 0 ]',
        "then",
        "  printf '%s\\n' '' "
        + _sh("reproduce.sh: at least one check failed. The lines above are evidence, not a reproduction."),
        "  exit 1",
        "fi",
        "printf '%s\\n' '' "
        + _sh("reproduce.sh: no check failed, but every UNKNOWN line is missing evidence, not a match."),
        "exit 0",
        "",
        f"# {facts.check_budget} recorded values were compiled into this script",
    ]


# --------------------------------------------------------------------------- #
# renderers: PowerShell
# --------------------------------------------------------------------------- #
def _render_ps1(facts: _Facts) -> str:
    out: list[str] = ["#requires -Version 5.1"]
    out.extend(_comment_block(facts, PS1_NAME))
    out.append("")
    out.extend(_prelude(_PS1_PRELUDE, _ps, facts))

    # --- code: git ---------------------------------------------------------
    out.append("Write-LabboxSection 'code: git commit'")
    out.extend(_git_warnings(facts, is_ps=True))
    if not facts.git.verifiable:
        out.extend(_emit_ps("UNKNOWN", "git_commit", facts.git.reason))
    else:
        out.append(f"$expectedCommit = {_ps(facts.git.commit)}")
        out.extend(
            [
                "if (-not (Get-Command -Name 'git' -ErrorAction SilentlyContinue)) {",
                *_emit_ps(
                    "UNKNOWN",
                    "git_commit",
                    "git is not installed on this machine, so HEAD cannot be compared",
                ),
                "} elseif (-not $script:LabboxProjectRoot) {",
                *_emit_ps(
                    "UNKNOWN",
                    "git_commit",
                    "no project root could be resolved from this script's directory, so HEAD cannot be compared",
                ),
                "} elseif (-not (Test-Path -LiteralPath "
                "(Join-Path $script:LabboxProjectRoot '.git') -PathType Container)) {",
                *_emit_ps(
                    "UNKNOWN",
                    "git_commit",
                    "no .git directory was found under the resolved project root",
                ),
                "} else {",
                "    $head = Invoke-LabboxTool -Exe 'git' -Arguments @('-C', "
                "$script:LabboxProjectRoot, 'rev-parse', 'HEAD')",
                "    if ([string]::IsNullOrEmpty($head)) {",
                *_emit_ps(
                    "UNKNOWN",
                    "git_commit",
                    "git rev-parse HEAD returned nothing in the resolved project root",
                ),
                "    } elseif ($head.Equals($expectedCommit, "
                "[System.StringComparison]::Ordinal) -or $head.StartsWith("
                "$expectedCommit, [System.StringComparison]::OrdinalIgnoreCase)) {",
                *_emit_ps(
                    "PASS",
                    "git_commit",
                    "HEAD matches the recorded commit (a short recorded hash is accepted as a prefix)",
                ),
                "    } else {",
                *_emit_ps(
                    "FAIL",
                    "git_commit",
                    "HEAD does not match the recorded commit; compare 'git rev-parse HEAD' "
                    "with the run_id in the header of this script",
                ),
                "    }",
                "}",
            ]
        )

    # --- interpreter -------------------------------------------------------
    out.append("Write-LabboxSection 'interpreter: python version'")
    out.append(f"$expectedPython = {_ps(facts.python.version)}")
    out.extend(
        [
            "$actualPython = Get-LabboxPyVersion",
            "if ([string]::IsNullOrEmpty($actualPython)) {",
            *_emit_ps(
                "UNKNOWN",
                "python_version",
                "no python3/python/py on PATH: set LABBOX_PYTHON to the interpreter you want "
                "checked. This script installed nothing.",
            ),
        ]
    )
    if not facts.python.verifiable:
        out.extend([*_emit_ps("UNKNOWN", "python_version", facts.python.reason), "}"])
    else:
        out.extend(
            [
                "} elseif ($actualPython.Equals($expectedPython, [System.StringComparison]::Ordinal)) {",
                *_emit_ps(
                    "PASS",
                    "python_version",
                    "the interpreter this script asked reports the recorded version",
                ),
                "} else {",
                *_emit_ps(
                    "FAIL",
                    "python_version",
                    "the interpreter this script asked reports a different version than the run recorded",
                ),
                "}",
            ]
        )
    out.extend(_venv_block_ps(facts))

    # --- dependencies ------------------------------------------------------
    out.append("Write-LabboxSection 'dependencies: frameworks and packages'")
    if facts.packages.empty:
        out.extend(_emit_ps("UNKNOWN", "packages", facts.packages.reason))
    else:
        names = ", ".join(_ps(name) for name, _v in facts.packages.items)
        out.append(f"$expectedPackages = @({names})")
        out.append("function Get-LabboxExpectedVersion {")
        out.append("    param([string]$Name)")
        for name, version in facts.packages.items:
            out.append(f"    if ($Name -ceq {_ps(name)}) {{ return {_ps(version)} }}")
        out.append("    return ''")
        out.append("}")
        out.extend(
            [
                "if (-not $script:LabboxPython) {",
                *_emit_ps(
                    "UNKNOWN",
                    "packages",
                    f"{len(facts.packages.items)} versions were recorded but no interpreter "
                    "is available to ask; the recorded list is printed as comments below",
                ),
                "} else {",
                "    foreach ($packageName in $expectedPackages) {",
                "        $want = Get-LabboxExpectedVersion -Name $packageName",
                "        $have = Get-LabboxPkgVersion -PackageName $packageName",
                "        if ([string]::IsNullOrEmpty($want)) {",
                "            Write-LabboxCheck NOTE ('package ' + $packageName) "
                + _ps("recorded as installed with an unknown version"),
                "        } elseif ([string]::IsNullOrEmpty($have)) {",
                "            Write-LabboxCheck FAIL ('package ' + $packageName) "
                + _ps("not found in the interpreter this script asked; it may live in another virtualenv"),
                "        } elseif ($have.Equals($want, [System.StringComparison]::Ordinal)) {",
                "            Write-LabboxCheck PASS ('package ' + $packageName) "
                + _ps("installed version matches the record"),
                "        } else {",
                "            Write-LabboxCheck FAIL ('package ' + $packageName) "
                + _ps("installed version differs from the recorded version"),
                "        }",
                "    }",
                "}",
            ]
        )
    out.extend(_requirements_block_ps(facts))

    # --- hardware ----------------------------------------------------------
    out.append("Write-LabboxSection 'hardware: CUDA driver and GPU model'")
    out.extend(_hardware_block_ps(facts))

    # --- data --------------------------------------------------------------
    out.append("Write-LabboxSection 'data: dataset digests'")
    out.extend(_emit_ps("NOTE", "dataset_method", _DIGEST_HOWTO))
    if not facts.datasets:
        out.extend(_emit_ps("UNKNOWN", "datasets", _NO_DATASETS))
    for dataset in facts.datasets:
        out.extend(_dataset_block_ps(dataset))

    # --- workspace ---------------------------------------------------------
    out.append("Write-LabboxSection 'workspace: recorded working directory'")
    out.extend(_workdir_block_ps(facts))

    # --- capture gaps ------------------------------------------------------
    out.append("Write-LabboxSection 'capture status and gaps'")
    out.extend(_collector_lines_ps(facts))
    out.extend(_absent_lines_ps(facts))
    for note in facts.notes:
        out.extend(_emit_ps("NOTE", "note", note))

    # --- the command -------------------------------------------------------
    out.append("Write-LabboxSection 'the command to run'")
    out.extend(_command_block_ps(facts))
    out.extend(_footer_ps(facts))
    # Join with CRLF instead of replacing: a recorded argument may itself contain a
    # newline, and that byte is data, not a line break of the script.
    return UTF8_BOM + "\r\n".join(out) + "\r\n"


def _emit_ps(status: str, name: str, detail: str, indent: str = "") -> list[str]:
    return [f"{indent}Write-LabboxCheck {status} {_ps(name)} {_ps(detail)}"]


def _dataset_block_ps(dataset: _DatasetFacts) -> list[str]:
    key = dataset.slug
    label = f"dataset_{key}"
    env_name = dataset.env_key
    out = [
        f"$expectedDigest{key} = {_ps(dataset.digest)}",
        f"$recordedStrategy{key} = {_ps(dataset.strategy)}",
        *_emit_ps(
            "NOTE",
            f"{label}_identity",
            f"dataset {dataset.dataset_id}: strategy={dataset.strategy}, "
            f"file_count={dataset.file_count}, total_bytes={dataset.total_bytes}",
        ),
    ]
    if dataset.content_proof:
        out.extend(
            _emit_ps(
                "NOTE",
                f"{label}_proof",
                f"strategy={dataset.strategy} was recorded as a full-content or externally pinned version",
            )
        )
    else:
        proof = (
            _FAST_MANIFEST
            if dataset.strategy == "fast_manifest"
            else f"strategy={dataset.strategy} is NOT a content proof of the dataset bytes"
        )
        out.extend(_emit_ps("WARN", f"{label}_proof", proof))
    for warning in dataset.warnings:
        out.extend(_emit_ps("WARN", f"{label}_warning", warning))
    if not dataset.digest:
        out.extend(
            _emit_ps(
                "UNKNOWN",
                label,
                f"no digest was recorded for {dataset.dataset_id}, so data identity cannot be checked at all",
            )
        )
        return out
    out.extend(
        [
            f"$supplied{key} = ''",
            f"$source{key} = ''",
            f"if ($env:{env_name}) {{",
            f"    $supplied{key} = [string]$env:{env_name}",
            f"    $source{key} = {_ps(f'environment variable {env_name}')}",
            "} elseif (-not [string]::IsNullOrEmpty($script:LabboxRunDir)) {",
            f"    $manifestHere = Join-Path (Get-Location) {_ps(MANIFEST_NAME)}",
            f"    $bundledCopy = Join-Path $script:LabboxRunDir {_ps(MANIFEST_NAME)}",
            "    # The manifest inside the run directory is the expected value, so it is",
            "    # skipped: only a manifest the user produced counts as evidence.",
            "    if ((Test-Path -LiteralPath $manifestHere -PathType Leaf) -and "
            "(-not $manifestHere.Equals($bundledCopy, "
            "[System.StringComparison]::OrdinalIgnoreCase))) {",
            f"        $probe{key} = (Get-LabboxManifestDigest -ManifestPath $manifestHere "
            f"-DatasetId {_ps(dataset.dataset_id)} -Position {dataset.position}) -split ' ', 2",
            f"        if ($probe{key}[0] -ne 'none') {{",
            f"            $supplied{key} = [string]$probe{key}[1]",
            f"            $source{key} = ('dataset manifest match by ' + $probe{key}[0] + ' (not by this script)')",
            "        }",
            "    }",
            "}",
            f"if ([string]::IsNullOrEmpty($supplied{key})) {{",
            *_emit_ps(
                "UNKNOWN",
                label,
                f"no digest available to compare for {dataset.dataset_id}; the recorded one is {dataset.digest}",
                "    ",
            ),
            f"}} elseif ($supplied{key}.Equals($expectedDigest{key}, [System.StringComparison]::Ordinal)) {{",
            *_emit_ps(
                "PASS",
                label,
                f"digest of {dataset.dataset_id} matches the recorded one (the digest came "
                "from you, not from this script)",
                "    ",
            ),
            "    } else {",
            *_emit_ps(
                "FAIL",
                label,
                f"digest of {dataset.dataset_id} differs from the recorded one; the data you "
                "are about to use is not the data this run consumed",
                "    ",
            ),
            "    }",
            f"if ($source{key}) {{",
            f"    Write-LabboxCheck NOTE {_ps(f'{label}_source')} $source{key}",
            "}",
        ]
    )
    return out


def _hardware_block_ps(facts: _Facts) -> list[str]:
    hw = facts.hardware
    if not (hw.driver or hw.runtime or hw.gpus):
        return _emit_ps("UNKNOWN", "hardware", hw.reason)
    out: list[str] = []
    if hw.runtime:
        out.extend(
            _emit_ps(
                "NOTE",
                "cuda_runtime",
                f"the run recorded CUDA runtime {hw.runtime}, which is a property of "
                "the torch build, not of this machine's driver",
            )
        )
    if hw.cudnn:
        out.extend(_emit_ps("NOTE", "cudnn", f"the run recorded cuDNN {hw.cudnn}"))
    out.extend(
        [
            "if ([string]::IsNullOrEmpty($script:LabboxGpus)) {",
            "    Write-LabboxCheck UNKNOWN 'gpu' "
            + _ps(
                "nvidia-smi produced no output here: either there is no NVIDIA driver "
                "tooling, or it failed. That is unknown, not proof that no GPU was used, and "
                "this script will not install a driver."
            ),
            "} else {",
        ]
    )
    if hw.driver:
        out.extend(
            [
                f"    $expectedDriver = {_ps(hw.driver)}",
                "    $firstGpuLine = ($script:LabboxGpus -split '\\r?\\n')[0]",
                "    $actualDriver = (($firstGpuLine -split ',')[-1]).Trim()",
                "    if ($actualDriver.Equals($expectedDriver, [System.StringComparison]::Ordinal)) {",
                *_emit_ps("PASS", "cuda_driver", "the driver reported here matches the record"),
                "    } else {",
                *_emit_ps("FAIL", "cuda_driver", "the driver here differs from the record"),
                "    }",
            ]
        )
    else:
        out.extend(_emit_ps("UNKNOWN", "cuda_driver", "no driver version was recorded for this run"))
    if hw.gpus:
        out.append(f"    $expectedGpuCount = {len(hw.gpus)}")
        out.append("    $actualGpuCount = Get-LabboxNonemptyCount -Text $script:LabboxGpus")
        out.extend(
            [
                "    if ($actualGpuCount -eq $expectedGpuCount) {",
                *_emit_ps("PASS", "gpu_count", "this machine reports as many GPUs as the run used"),
                "    } else {",
                *_emit_ps(
                    "FAIL",
                    "gpu_count",
                    "this machine reports a different number of GPUs than the run used",
                ),
                "    }",
            ]
        )
        for position, gpu_index, name, driver in _gpu_rows(facts):
            out.append(f"    $gpuExpected{position} = {_ps(name)}")
            out.extend(
                [
                    "    if ($script:LabboxGpus.IndexOf("
                    f"$gpuExpected{position}, "
                    "[System.StringComparison]::Ordinal) -ge 0)",
                    "    {",
                    *_emit_ps(
                        "PASS",
                        f"gpu_{position}",
                        f"the GPU model recorded at index {gpu_index} is present here",
                    ),
                    "    } else {",
                    *_emit_ps(
                        "FAIL",
                        f"gpu_{position}",
                        f"the GPU model recorded at index {gpu_index} is not in this machine's nvidia-smi output",
                    ),
                    "    }",
                ]
            )
            if driver:
                out.extend(_emit_ps("NOTE", f"gpu_{position}_driver", f"the run saw driver {driver} here"))
        out.extend(
            _emit_ps(
                "NOTE",
                "gpu_identity",
                "GPU models are matched by name only: UUID, board position, clocks and "
                "NVLink topology are not compared here",
            )
        )
    else:
        out.extend(_emit_ps("UNKNOWN", "gpu_model", "no GPU model was recorded, so there is nothing to compare"))
    out.append("}")
    return out


def _workdir_block_ps(facts: _Facts) -> list[str]:
    if not facts.cwd:
        return _emit_ps(
            "UNKNOWN",
            "workdir",
            "no cwd was recorded with the command, so it cannot be located",
        )
    out = [
        f"$recordedCwd = {_ps(facts.cwd)}",
        "if (Test-Path -LiteralPath $recordedCwd -PathType Container) {",
        *_emit_ps("PASS", "workdir", "the recorded working directory exists as written"),
        "} elseif ($script:LabboxProjectRoot -and (Test-Path -LiteralPath "
        "(Join-Path $script:LabboxProjectRoot $recordedCwd) -PathType Container)) {",
        *_emit_ps(
            "PASS",
            "workdir",
            "the recorded cwd resolves under the project root (it was relative)",
        ),
        "} else {",
        "    $moved = $false",
        "    if ($script:LabboxProjectRoot) {",
        "        $leaf = [System.IO.Path]::GetFileName($recordedCwd.TrimEnd('\\', '/'))",
        "        $moved = Test-Path -LiteralPath (Join-Path $script:LabboxProjectRoot $leaf) -PathType Container",
        "    }",
        "    if ($moved) {",
        *_emit_ps(
            "UNKNOWN",
            "workdir",
            "the recorded cwd does not exist, but a directory with the same name sits under "
            "the project root: the project may have been moved",
        ),
        "    } else {",
        *_emit_ps(
            "FAIL",
            "workdir",
            "the recorded working directory does not exist, and no obvious relocation of it was found",
        ),
        "    }",
        "}",
        *_emit_ps(
            "NOTE",
            "workdir_usage",
            "cd into the directory above before running the command at the end of this "
            "script; this script does not change your working directory permanently",
        ),
    ]
    if facts.cwd_is_relative:
        out.extend(_emit_ps("NOTE", "workdir_kind", "the recorded cwd was relative to the project root"))
    return out


def _venv_block_ps(facts: _Facts) -> list[str]:
    if not facts.python.venv_prefix:
        return []
    return [
        f"$recordedVenv = {_ps(facts.python.venv_prefix)}",
        "if (Test-Path -LiteralPath $recordedVenv -PathType Container) {",
        *_emit_ps("PASS", "venv_prefix", "the recorded virtualenv directory exists here"),
        "} else {",
        *_emit_ps(
            "FAIL",
            "venv_prefix",
            "no virtualenv at the recorded path. Create it yourself; this script will not, "
            "and activating it is your decision",
        ),
        "}",
        *_emit_ps(
            "NOTE",
            "venv_note",
            f"recorded interpreter: {facts.python.executable or 'not recorded'}; this script "
            "used the interpreter on PATH (or LABBOX_PYTHON), not that one",
        ),
    ]


def _venv_block_sh(facts: _Facts) -> list[str]:
    if not facts.python.venv_prefix:
        return []
    return [
        f"recorded_venv={_sh(facts.python.venv_prefix)}",
        'if [ -d "$recorded_venv" ]',
        "then",
        *_emit_sh("PASS", "venv_prefix", "the recorded virtualenv directory exists here"),
        "else",
        *_emit_sh(
            "FAIL",
            "venv_prefix",
            "no virtualenv at the recorded path. Create it yourself; this script will not, "
            "and activating it is your decision",
        ),
        "fi",
        *_emit_sh(
            "NOTE",
            "venv_note",
            f"recorded interpreter: {facts.python.executable or 'not recorded'}; this script "
            "used the interpreter on PATH (or LABBOX_PYTHON), not that one",
        ),
    ]


def _collector_lines_ps(facts: _Facts) -> list[str]:
    out: list[str] = []
    for name, status in facts.collectors:
        out.extend(_emit_ps("NOTE", f"collector_{_escape_key(name)}", f"{name}={status}"))
    if not facts.collectors:
        out.extend(
            _emit_ps(
                "NOTE",
                "collectors",
                "no collector status was recorded, so it is unknown which sections were attempted",
            )
        )
    return out


def _absent_lines_ps(facts: _Facts) -> list[str]:
    if not facts.absent:
        return _emit_ps(
            "NOTE",
            "not_captured",
            "environment.absent is empty: every section in this snapshot was attempted",
        )
    return [
        line
        for item in facts.absent
        for line in _emit_ps(
            "UNKNOWN",
            "not_captured",
            f"{item}: listed in environment.absent, so it is unknown, not matched",
        )
    ]


def _command_block_ps(facts: _Facts) -> list[str]:
    if not facts.argv:
        return [
            *_emit_ps(
                "UNKNOWN",
                "command",
                "no argv was recorded; the raw command line printed below is text only and "
                "must not be treated as a command",
            ),
            "Write-Host ''",
            "Write-Host " + _ps(facts.raw_command or "not recorded"),
        ]
    items = ", ".join(_ps(item) for item in facts.argv)
    out = [
        f"$labboxArgv = @({items})",
        "function Format-LabboxCommand {",
        "    # Re-quote each recorded word for display, with the same rule reproduce.sh",
        "    # uses: anything outside the safe character set becomes a single-quoted",
        "    # literal whose quotes are doubled. Nothing here is executed.",
        "    param([string[]]$Words)",
        "    $parts = foreach ($word in $Words) {",
        "        $text = [string]$word",
        "        if ($text.Length -eq 0) {",
        "            \"''\"",
        f"        }} elseif ($text -cmatch '^[{_SAFE_WORD_CHARS}]+$') {{",
        "            $text",
        "        } else {",
        '            "\'" + $text.Replace("\'", "\'\'") + "\'"',
        "        }",
        "    }",
        "    ($parts -join ' ')",
        "}",
        "$labboxCommandLine = Format-LabboxCommand -Words $labboxArgv",
        "Write-Host ''",
        "Write-Host " + _ps("Run this yourself - LabBox recorded it and does not start it:"),
    ]
    if facts.cwd:
        out.append("Write-Host ('  cd ' + (Format-LabboxCommand -Words @($recordedCwd)) + ' ; ' + $labboxCommandLine)")
    out.append("Write-Host ('  ' + $labboxCommandLine)")
    return out


def _footer_ps(facts: _Facts) -> list[str]:
    return [
        "Write-Host ''",
        "Write-Host " + _ps("--- summary ---"),
        "$script:ChecksTotal = $script:ChecksPass + $script:ChecksFail + $script:ChecksUnknown + $script:ChecksWarn",
        "Write-LabboxCheck NOTE 'totals' ('{0} checks reported: {1} PASS, {2} FAIL, "
        "{3} UNKNOWN, {4} WARN: " + _TOTALS_NOTE + "' -f $script:ChecksTotal, $script:ChecksPass, $script:ChecksFail, "
        "$script:ChecksUnknown, $script:ChecksWarn)",
        *_emit_ps("NOTE", "limits", _CLAIM_LIMIT),
        *_emit_ps("NOTE", "limits", _NO_MUTATION),
        *_emit_ps("NOTE", "limits", _BITWISE),
        "if ($script:ChecksFail -gt 0) {",
        "    Write-Host ''",
        "    Write-Host "
        + _ps("reproduce.ps1: at least one check failed. The lines above are evidence, not a reproduction."),
        "    exit 1",
        "}",
        "Write-Host ''",
        "Write-Host " + _ps("reproduce.ps1: no check failed, but every UNKNOWN line is missing evidence, not a match."),
        "exit 0",
        "",
        f"# {facts.check_budget} recorded values were compiled into this script",
    ]


# --------------------------------------------------------------------------- #
# pieces shared by both renderers
# --------------------------------------------------------------------------- #
def _git_warnings(facts: _Facts, *, is_ps: bool) -> list[str]:
    emit = _emit_ps if is_ps else _emit_sh
    out: list[str] = []
    if facts.git.dirty is True:
        out.extend(emit("WARN", "git_dirty_in_recorded_run", _DIRTY_WARN))
    elif facts.git.dirty is None and facts.git.verifiable:
        out.extend(
            emit(
                "WARN",
                "git_dirty_unknown",
                "the dirty flag was not captured: even if HEAD matches, the recorded run may "
                "have had uncommitted changes, so HEAD alone is not sufficient evidence",
            )
        )
    if facts.git.dirty is False:
        out.extend(
            emit(
                "NOTE",
                "git_worktree",
                "the recorded working tree was clean, so HEAD is the whole code state",
            )
        )
    if facts.git.diff_hash:
        out.extend(
            emit(
                "NOTE",
                "git_diff_hash",
                f"the recorded diff hash is {facts.git.diff_hash}: it identifies the "
                "uncommitted delta, but this script cannot recompute it from git alone",
            )
        )
    if facts.git.branch:
        out.extend(
            emit(
                "NOTE",
                "git_branch",
                f"the run was on branch {facts.git.branch}; only HEAD is compared here",
            )
        )
    return out


def _requirements_block_sh(facts: _Facts) -> list[str]:
    rows = _requirement_rows(facts)
    if not rows:
        return []
    out = _requirement_comments(rows)
    out.append("printf '%s\\n' '' " + _sh(_REQUIREMENT_HEADLINE))
    out.append(
        "printf '  %s\\n' "
        + _sh("python -m pip install -r your-own-requirements.txt")
        + "   # printf only: this script runs no installer"
    )
    return out


def _requirements_block_ps(facts: _Facts) -> list[str]:
    rows = _requirement_rows(facts)
    if not rows:
        return []
    out = _requirement_comments(rows)
    out.append("Write-Host ''")
    out.append("Write-Host " + _ps(_REQUIREMENT_HEADLINE))
    out.append(
        "Write-Host "
        + _ps("  python -m pip install -r your-own-requirements.txt")
        + "   # Write-Host only: this script runs no installer"
    )
    return out


_REQUIREMENT_HEADLINE = (
    "Suggested dependency set, recorded verbatim. NOT installed by this script: copy the "
    "list from the comments above into your own requirements file and install it yourself."
)


def _requirement_comments(rows: list[tuple[str, str]]) -> list[str]:
    out = [
        "",
        "# Recorded dependency versions. This is a suggestion, not a lock file: there are no",
        "# hashes here and transitive dependencies are not resolved. LabBox installs nothing.",
        "#   python -m pip install -r your-own-requirements.txt   (suggestion, not run here)",
    ]
    for name, version in rows:
        out.append(f"#   {name}=={version}" if version else f"#   {name}  (version unknown)")
    return out


def _comment_block(facts: _Facts, script_name: str) -> list[str]:
    is_ps1 = script_name == PS1_NAME
    target = "Windows PowerShell 5.1 or newer" if is_ps1 else "bash 3.2 or newer (also runs under most POSIX sh)"
    usage = "powershell -NoLogo -ExecutionPolicy Bypass -File reproduce.ps1" if is_ps1 else "bash reproduce.sh"
    rule = "# " + "-" * 74
    lines = [
        rule,
        f"# {script_name} - generated by LabBox. This script only checks and prints.",
        rule,
    ]
    for label, value in facts.header:
        lines.append(f"# {label:<26}: {value}")
    lines += [
        f"# {'target':<26}: {target}",
        f"# {'usage':<26}: {usage}",
        "#",
        f"# {_NO_MUTATION}",
        f"# {_BITWISE}",
        f"# {_CLAIM_LIMIT}",
    ]
    if facts.git.dirty is True:
        lines.append(f"# WARNING: {_DIRTY_WARN}")
    return lines


def _gpu_rows(facts: _Facts) -> list[tuple[int, str, str, str]]:
    return [
        (position, gpu_index or str(position), name, driver)
        for position, (gpu_index, name, driver) in enumerate(facts.hardware.gpus, start=1)
        if name
    ]


def _requirement_rows(facts: _Facts) -> list[tuple[str, str]]:
    rows: dict[str, str] = {}
    for name, version in facts.packages.items:
        rows.setdefault(name, version)
    return sorted(rows.items(), key=lambda kv: kv[0].lower())
