"""RuleEngine and reproducibility scoring (§10.4, §12.1).

Rules are pure functions over already-collected evidence, each with a stable ``code``, and
every one of them has a unit test *and* a counter-example — a rule that cannot be shown to
be wrong cannot be trusted to be right.

The causal constraint in §10.5 is enforced structurally: a rule may say "A and B changed
together, so this comparison cannot isolate either", and may never say "the accuracy drop
was caused by A".
"""

from __future__ import annotations

from collections.abc import Sequence

from labbox.analysis.diff import HIGH_IMPACT_CATEGORIES
from labbox.analysis.record import RunBundle
from labbox.domain.diagnostics import (
    CompareResult,
    Diagnostic,
    DiffFact,
    DiffKind,
    ReproBand,
    ReproCheck,
    Reproducibility,
    ReproState,
    Severity,
)

RULES_VERSION = "1"

IMPORTANT_DIMENSIONS = {
    "code_locatable",
    "deps_reproducible",
    "data_verifiable",
    "randomness_controlled",
}
WEIGHTS = {
    "code_locatable": 20,
    "deps_reproducible": 20,
    "data_verifiable": 20,
    "randomness_controlled": 15,
    "hardware_identified": 10,
    "command_replayable": 10,
    "artifacts_locatable": 5,
}
_STATE_FACTOR = {
    ReproState.OK: 1.0,
    ReproState.PARTIAL: 0.5,
    ReproState.MISSING: 0.0,
    ReproState.UNKNOWN: 0.0,
}


# --- single-run diagnostics --------------------------------------------------
def run_diagnostics(bundle: RunBundle) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    record = bundle.record
    environment = bundle.environment
    if record is None:
        out.append(
            Diagnostic(
                severity=Severity.CRITICAL,
                code="run_record_unreadable",
                message="run.json is missing or unreadable; nothing about this run can be compared.",
                evidence=["run.json"],
                suggestion="the run directory is corrupt; re-run with the same command",
            )
        )
        return out
    if record.status.value == "running":
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="run_incomplete",
                message="LabBox did not finish this run: the record is still marked running.",
                evidence=[f"status={record.status.value}"],
                suggestion="the recorder was killed; treat all missing sections as unknown",
            )
        )
    if record.status.value == "failed":
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="training_failed",
                message=f"The training command exited {record.exit_code}.",
                evidence=[f"exit_code={record.exit_code}", f"exit_code_raw={record.exit_code_raw}"],
                suggestion="read logs/stderr.log first; CUDA OOM and shape mismatches dominate here",
            )
        )
    if record.interrupted:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="run_interrupted",
                message="The run was interrupted before the training process exited normally.",
                evidence=[f"exit_code_raw={record.exit_code_raw}", f"forced_kill={record.forced_kill}"],
            )
        )
    if environment is None:
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="environment_missing",
                message="No environment snapshot was captured: git, packages and hardware are all unknown.",
                evidence=["environment.json absent"],
                suggestion="the run was interrupted before collection finished, or collection failed"
                "; see logs/events.jsonl",
            )
        )
    elif environment is not None and environment.git.has_repo:
        if environment.git.dirty:
            out.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="git_dirty",
                    message="训练开始时 Git 工作区存在未提交修改。",
                    evidence=[
                        "git.dirty=true",
                        f"git.diff_hash={environment.git.diff_hash or 'not recorded'}",
                        f"modified={environment.git.modified_count} untracked={environment.git.untracked_count}",
                    ],
                    suggestion="提交或导出补丁后再运行；否则别人 checkout 该 commit 得不到同样结果",
                )
            )
        elif environment.git.commit is None:
            out.append(
                Diagnostic(
                    severity=Severity.HIGH,
                    code="git_head_unresolvable",
                    message="Inside a repository but HEAD could not be resolved (detached or empty?).",
                    evidence=["git.has_repo=true", "git.commit=None"],
                )
            )
    elif environment is not None:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="not_a_repository",
                message="The project is not a Git repository: code state cannot be recorded.",
                evidence=["git.has_repo=false", environment.git.detail or ""],
                suggestion="git init, or accept that code changes will be invisible in comparisons",
            )
        )
    collectors = environment.collectors if environment is not None else {}
    for name, status in sorted(collectors.items()):
        if status in {"unavailable", "failed", "timeout"}:
            out.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code=f"collector_{name}_{status}",
                    message=f"The {name} collector produced no data ({status}).",
                    evidence=[f"collectors.{name}={status}"],
                    suggestion="run `labbox doctor` to see why",
                )
            )
    if bundle.datasets is None or bundle.datasets.not_declared:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="dataset_not_declared",
                message="No dataset was declared, so the data version is unknown for this run.",
                evidence=["dataset-manifest.json datasets=[]"],
                suggestion="pass --dataset <path|version> or set [datasets] paths in .labbox/config.toml",
            )
        )
    else:
        for entry in bundle.datasets.datasets:
            if entry.truncated:
                out.append(
                    Diagnostic(
                        severity=Severity.HIGH,
                        code="dataset_manifest_truncated",
                        message=f"Dataset {entry.id} manifest was truncated ({entry.cancellation_reason}).",
                        evidence=[f"file_count={entry.file_count}"],
                        suggestion="raise [datasets] max_files / time_budget_seconds to get a comparable digest",
                    )
                )
            if not entry.is_content_proof:
                out.append(
                    Diagnostic(
                        severity=Severity.INFO,
                        code="dataset_sampled_digest",
                        message=f"Dataset {entry.id} uses {entry.strategy.value}: head+tail sampling"
                        ", not a content proof.",
                        evidence=[f"strategy={entry.strategy.value}"],
                        suggestion="use strategy=full_content if data integrity is the thing under test",
                    )
                )
    if bundle.metrics is not None and bundle.metrics.metrics_missing:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="METRICS_MISSING",
                message="No structured metrics were found; comparisons will describe code and data only.",
                evidence=["metrics.json metrics_missing=true"],
                suggestion="have training write metrics.jsonl, or point --metric-file at it",
            )
        )
    for artifact in bundle.artifacts.artifacts if bundle.artifacts else []:
        if artifact.hash_state.value == "deferred_too_large":
            out.append(
                Diagnostic(
                    severity=Severity.INFO,
                    code="artifact_hash_deferred",
                    message=f"{artifact.path} is {artifact.size_bytes} bytes: metadata recorded, hash skipped.",
                    evidence=[f"hash_state={artifact.hash_state.value}"],
                    suggestion="--hash-artifacts forces the hash (slow)",
                )
            )
    ambiguous = [value.path for value in bundle.config_values.values if value.ambiguous] if bundle.config_values else []
    if ambiguous:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="hyperparam_source_ambiguous",
                message="Some hyperparameters are set by more than one source; the effective value is a guess.",
                evidence=ambiguous[:8],
                suggestion="set each hyperparameter in exactly one place, or pass --config-file explicitly",
            )
        )
    return out


# --- comparison diagnostics --------------------------------------------------
def compare_diagnostics(result: CompareResult, bundles: Sequence[RunBundle]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    changed = [fact for fact in result.facts if fact.kind not in {DiffKind.UNKNOWN, DiffKind.EQUAL}]
    by_category: dict[str, list[DiffFact]] = {}
    for fact in changed:
        by_category.setdefault(fact.category, []).append(fact)

    for bundle in bundles:
        for diagnostic in run_diagnostics(bundle):
            if not any(existing.code == diagnostic.code for existing in out):
                out.append(diagnostic)

    for fact in changed:
        if fact.category == "code" and fact.path in {"git.commit", "git.diff_hash"}:
            out.append(
                Diagnostic(
                    severity=Severity.HIGH,
                    code="code_changed",
                    message=f"{fact.path}: {fact.left_repr} -> {fact.right_repr}",
                    evidence=fact.sources,
                    fact_ids=[fact.id],
                )
            )
    data_facts = [fact for fact in by_category.get("data", []) if "digest" in fact.path]
    if data_facts:
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="data_changed",
                message="The dataset digest differs, so metric differences cannot be attributed to code.",
                evidence=[f"{fact.path}: {fact.left_repr} -> {fact.right_repr}" for fact in data_facts],
                fact_ids=[fact.id for fact in data_facts],
                suggestion="re-run one side on the other's data, or pin the dataset version explicitly",
            )
        )
    env_versions = [
        fact
        for fact in by_category.get("environment", [])
        if fact.path.startswith(("python.version", "frameworks.", "cuda."))
    ]
    if env_versions:
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="environment_changed",
                message="Runtime/library versions differ between the two runs.",
                evidence=[f"{fact.path}: {fact.left_repr} -> {fact.right_repr}" for fact in env_versions[:6]],
                fact_ids=[fact.id for fact in env_versions],
            )
        )
    hardware = [fact for fact in by_category.get("hardware", []) if fact.category == "hardware"]
    if hardware:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="hardware_changed",
                message="GPU/host differences may account for part of the metric delta.",
                evidence=[f"{fact.path}: {fact.left_repr} -> {fact.right_repr}" for fact in hardware[:5]],
                fact_ids=[fact.id for fact in hardware],
            )
        )
    randomness = by_category.get("randomness", [])
    if randomness:
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="randomness_changed",
                message="Seed or determinism settings differ: the two runs are not repeatable versions of each other.",
                evidence=[f"{fact.path}: {fact.left_repr} -> {fact.right_repr}" for fact in randomness],
                fact_ids=[fact.id for fact in randomness],
            )
        )
    elif "seeds.declared" not in result.unchanged:
        # This diagnostic means "nobody recorded a seed". The fact for that path is only
        # emitted when the two sides *disagree*, so testing for a missing UNKNOWN fact fired
        # it on every run whose seed genuinely matched — the exact case where run-to-run
        # variance is controlled and the warning is wrong. Equal seeds are listed in
        # `unchanged`, so absence from that list is the honest test.
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="seed_undocumented",
                message="Neither run recorded a seed, so run-to-run variance cannot be separated from the change.",
                evidence=["seeds.declared unknown in both runs"],
                suggestion="pass --seed on the command line so it is observable without instrumentation",
            )
        )
    hyperparams = by_category.get("config", [])
    if hyperparams:
        out.append(
            Diagnostic(
                severity=Severity.INFO,
                code="hyperparams_changed",
                message="Hyperparameters differ: " + ", ".join(fact.path for fact in hyperparams[:8]),
                evidence=[f"{fact.path}: {fact.left_repr} -> {fact.right_repr}" for fact in hyperparams[:8]],
                fact_ids=[fact.id for fact in hyperparams],
            )
        )
    high_categories = {
        category for category, facts in by_category.items() if category in HIGH_IMPACT_CATEGORIES and facts
    }
    if len(high_categories) > 1:
        out.append(
            Diagnostic(
                severity=Severity.HIGH,
                code="multiple_variables_changed",
                message=(
                    "Variables in " + ", ".join(sorted(high_categories)) + " all changed, so this comparison "
                    "cannot attribute the metric delta to any single one."
                ),
                evidence=[
                    f"{category}: {len(facts)} changed fact(s)"
                    for category, facts in sorted(by_category.items())
                    if category
                ],
                fact_ids=[
                    fact.id
                    for facts in by_category.values()
                    for fact in facts
                    if fact.category in HIGH_IMPACT_CATEGORIES
                ],
                suggestion="change one variable at a time, or add a third run that isolates one of them",
            )
        )
    if result.metrics_missing:
        out.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="METRICS_MISSING",
                message="At least one side has no metrics: "
                "the comparison covers code, environment, data and command only.",
                evidence=["metrics.json missing or empty in one or both runs"],
            )
        )
    unknowns = [fact for fact in result.facts if fact.kind is DiffKind.UNKNOWN]
    if unknowns:
        out.append(
            Diagnostic(
                severity=Severity.INFO,
                code="evidence_gaps",
                message=f"{len(unknowns)} compared field(s) were only captured on one side "
                "and are reported as unknown.",
                evidence=[f"{fact.path} ({fact.unknown_side or 'both'} side missing)" for fact in unknowns[:8]],
                fact_ids=[fact.id for fact in unknowns],
            )
        )
    for index, diagnostic in enumerate(out, start=1):
        if not diagnostic.id:
            diagnostic.id = f"D-{index:03d}"
    return out


CONFOUND_NOTE = "当前证据不足以把指标变化归因到单一变量：请把它当作相关性而不是因果结论。"


def confounders(result: CompareResult) -> list[str]:
    categories = {
        fact.category
        for fact in result.facts
        if fact.kind not in {DiffKind.UNKNOWN, DiffKind.EQUAL} and fact.category in HIGH_IMPACT_CATEGORIES
    }
    if len(categories) < 2:
        return []
    return [
        "changed groups: " + ", ".join(sorted(categories)) + " — 无法单独归因",
        CONFOUND_NOTE,
    ]


# --- reproducibility ---------------------------------------------------------
def score_reproducibility(bundle: RunBundle) -> Reproducibility:
    checks: list[ReproCheck] = []
    record = bundle.record
    environment = bundle.environment

    git = environment.git if environment is not None else None
    if git is None:
        checks.append(ReproCheck(id="code_locatable", state=ReproState.UNKNOWN, reason="environment not captured"))
    elif not git.has_repo:
        checks.append(ReproCheck(id="code_locatable", state=ReproState.MISSING, reason="not a git repository"))
    elif git.commit is None:
        checks.append(ReproCheck(id="code_locatable", state=ReproState.MISSING, reason="HEAD unresolvable"))
    elif git.dirty and git.diff_hash:
        checks.append(
            ReproCheck(
                id="code_locatable",
                state=ReproState.PARTIAL,
                reason=f"dirty tree, diff hashed {git.diff_hash[:18]}",
            )
        )
    elif git.dirty:
        checks.append(
            ReproCheck(id="code_locatable", state=ReproState.MISSING, reason="dirty tree without a diff hash")
        )
    else:
        checks.append(ReproCheck(id="code_locatable", state=ReproState.OK, reason=f"clean tree at {git.short_commit}"))

    packages = (environment.packages if environment is not None else None) or {}
    frameworks = (environment.frameworks if environment is not None else None) or {}
    if not packages and not frameworks:
        checks.append(
            ReproCheck(
                id="deps_reproducible",
                state=ReproState.MISSING if environment is not None else ReproState.UNKNOWN,
                reason=(environment.package_source or "no package versions captured")
                if environment is not None
                else "environment not captured",
            )
        )
    else:
        source = environment.package_source if environment is not None else ""
        lockfile = source or ""
        lockfile = lockfile if "lock" in lockfile.lower() else None
        checks.append(
            ReproCheck(
                id="deps_reproducible",
                state=ReproState.OK if lockfile else ReproState.PARTIAL,
                reason=f"{len(packages)} package version(s) recorded"
                + ("" if lockfile else "; no lockfile, so a rebuild may resolve differently"),
            )
        )

    datasets = bundle.datasets
    if datasets is None or datasets.not_declared:
        checks.append(ReproCheck(id="data_verifiable", state=ReproState.MISSING, reason="no dataset declared"))
    else:
        strategies = [entry.strategy.value for entry in datasets.datasets]
        proof = any(entry.is_content_proof for entry in datasets.datasets)
        declared_only = any(entry.strategy.value in {"declared"} for entry in datasets.datasets)
        checks.append(
            ReproCheck(
                id="data_verifiable",
                state=ReproState.OK if proof else (ReproState.MISSING if declared_only else ReproState.PARTIAL),
                reason="strategy "
                + ", ".join(sorted(set(strategies)))
                + ("" if proof else "; sampled/declared digests are not content proofs"),
            )
        )

    seeds_known = bool(
        record is not None and bundle.hyperparams() and any(key.endswith("seed") for key in bundle.hyperparams())
    )
    env_seed = bool(environment is not None and (environment.seeds.python_hash_seed or "").strip())
    deterministic_known = environment is not None and environment.seeds.cuda_deterministic.value is not None
    if seeds_known and deterministic_known:
        checks.append(
            ReproCheck(id="randomness_controlled", state=ReproState.OK, reason="seed and determinism recorded")
        )
    elif seeds_known or env_seed:
        checks.append(
            ReproCheck(
                id="randomness_controlled",
                state=ReproState.PARTIAL,
                reason="a seed was observable, but deterministic-algorithm settings are not",
            )
        )
    else:
        checks.append(
            ReproCheck(
                id="randomness_controlled",
                state=ReproState.MISSING,
                reason="no seed observable (in-script seeds are not visible without instrumentation)",
            )
        )

    gpus = environment.gpus if environment is not None else []
    if gpus and environment is not None and environment.cuda.driver:
        checks.append(
            ReproCheck(
                id="hardware_identified",
                state=ReproState.OK if environment.cuda.runtime else ReproState.PARTIAL,
                reason=f"{len(gpus)} GPU(s), driver {environment.cuda.driver}"
                + (", CUDA runtime unknown" if not environment.cuda.runtime else ""),
            )
        )
    elif environment is not None and environment.host.cpu_model:
        checks.append(
            ReproCheck(
                id="hardware_identified",
                state=ReproState.PARTIAL,
                reason="no GPU recorded; host CPU/RAM only",
            )
        )
    else:
        checks.append(ReproCheck(id="hardware_identified", state=ReproState.UNKNOWN, reason="environment not captured"))

    if record is not None and record.command.argv and record.command.cwd:
        checks.append(
            ReproCheck(
                id="command_replayable",
                state=ReproState.OK if record.command.argv[0] else ReproState.MISSING,
                reason=f"argv recorded ({len(record.command.argv)} tokens) at {record.command.cwd}",
            )
        )
    else:
        checks.append(ReproCheck(id="command_replayable", state=ReproState.MISSING, reason="no command recorded"))

    artifacts = bundle.artifacts
    hashed = [entry for entry in artifacts.artifacts if entry.sha256] if artifacts else []
    if artifacts is None:
        # No artifacts.json at all means the collector never produced a manifest. That is
        # missing evidence, not a finding that the run wrote nothing; reporting PARTIAL here
        # scored a killed recorder as a half-answer and dragged the band down for the wrong
        # reason. Every other check that lost its input says UNKNOWN, and so does this one.
        checks.append(
            ReproCheck(
                id="artifacts_locatable",
                state=ReproState.UNKNOWN,
                reason="no artifacts.json: the artifact collector did not produce a manifest",
            )
        )
    elif not artifacts.artifacts:
        checks.append(ReproCheck(id="artifacts_locatable", state=ReproState.MISSING, reason="no artifact discovered"))
    elif hashed:
        checks.append(
            ReproCheck(
                id="artifacts_locatable",
                state=ReproState.OK,
                reason=f"{len(hashed)} artifact(s) with recorded hash",
            )
        )
    else:
        checks.append(
            ReproCheck(
                id="artifacts_locatable",
                state=ReproState.PARTIAL,
                reason="artifacts recorded by path/size only, no hash",
            )
        )

    for check in checks:
        check.weight = WEIGHTS.get(check.id, 0)
    score = round(sum(_STATE_FACTOR.get(check.state, 0.0) * check.weight for check in checks))
    missing_important = any(
        check.state in {ReproState.MISSING, ReproState.UNKNOWN} and check.id in IMPORTANT_DIMENSIONS for check in checks
    )
    band = (
        ReproBand.LOW
        if missing_important or score < 55
        else (ReproBand.HIGH if score >= 80 and not missing_important else ReproBand.MEDIUM)
    )
    return Reproducibility(band=band, score=score, checklist=checks, rules_version=RULES_VERSION)


__all__ = [
    "RULES_VERSION",
    "WEIGHTS",
    "compare_diagnostics",
    "confounders",
    "run_diagnostics",
    "score_reproducibility",
]
