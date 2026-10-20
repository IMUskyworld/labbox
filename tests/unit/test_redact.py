"""Secret redaction and HTML/log escaping (§16.2 requires these tests explicitly)."""

from __future__ import annotations

import pytest

from labbox.infrastructure.redact import SECRET_VALUE_RE, Redactor


@pytest.fixture
def redactor() -> Redactor:
    return Redactor()


SENSITIVE_NAMES = [
    "OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "DB_PASSWORD",
    "MYSQL_PASS",
    "AWSCREDENTIALS",
    "my_secret_value",
    "X_API_TOKEN",
]

SAFE_NAMES = [
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "PATH",
    "HOME",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TOKENIZERS_PARALLELISM",
]


@pytest.mark.parametrize("name", SENSITIVE_NAMES)
def test_sensitive_names_are_blocked(redactor: Redactor, name: str) -> None:
    assert redactor.is_sensitive_name(name), name
    snapshot = redactor.filter_env({name: "super-secret-value", "HOME": "/x"})
    assert name not in snapshot.entries
    assert name in snapshot.redacted_names
    assert "super-secret-value" not in repr(snapshot.entries)


@pytest.mark.parametrize("name", SAFE_NAMES)
def test_reproducibility_relevant_names_survive(redactor: Redactor, name: str) -> None:
    assert not redactor.is_sensitive_name(name), name


def test_extra_patterns_are_applied() -> None:
    redactor = Redactor(extra_patterns=[r"(?i)^MY_COMPANY_.*"])
    assert redactor.is_sensitive_name("MY_COMPANY_DEPLOY_KEY")


def test_invalid_user_pattern_is_skipped_not_fatal() -> None:
    redactor = Redactor(extra_patterns=["(?i)unclosed(["])
    assert redactor.is_sensitive_name("OPENAI_API_KEY")


def test_secret_shaped_values_are_scrubbed_even_in_safe_names(redactor: Redactor) -> None:
    leaked = redactor.scrub_value("connecting with sk-abcdefghij0123456789ABCDEF to api")
    assert "sk-abcdefghij0123456789ABCDEF" not in leaked
    assert "***" in leaked
    assert SECRET_VALUE_RE.search("ghp_" + "A" * 24)


def test_git_remote_credentials_are_stripped(redactor: Redactor) -> None:
    url = "https://user:hunter2@github.com/acme/privaterepo.git"
    scrubbed = redactor.scrub_url(url)
    assert "hunter2" not in scrubbed
    assert "user" not in scrubbed.split("@")[0]
    assert scrubbed.endswith("privaterepo.git")


def test_error_message_never_carries_a_key(redactor: Redactor) -> None:
    from labbox.domain.common import ErrorInfo

    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    error = ErrorInfo.from_exception(
        RuntimeError(f"auth failed for {secret}"), component="llm", redact=(secret,)
    )
    assert secret not in error.message
    assert "***" in error.message


def test_env_snapshot_is_capped(redactor: Redactor) -> None:
    env = {f"VAR_{i}": "y" * 2000 for i in range(500)}
    snapshot = redactor.filter_env(env, max_entries=50, max_value_chars=100)
    assert len(snapshot.entries) <= 50
    assert snapshot.total_count == 500
    assert all(len(v) < 200 for v in snapshot.entries.values())
