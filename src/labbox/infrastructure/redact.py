"""Secret redaction (§8.2, §16).

Nothing may reach ``environment.json``, an error message, the HTML report or an LLM
request without passing through here first. The default deny-list is a name pattern, not
a value pattern, because a value like ``sk-...`` inside an unrelated variable is exactly
the case that a naive value scan misses.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from labbox.domain.environment import EnvVars

DEFAULT_REDACT_PATTERNS: tuple[str, ...] = (
    r"(?i).*token.*",
    r"(?i).*key.*",
    r"(?i).*secret.*",
    r"(?i).*password.*",
    r"(?i).*pass.*",
    r"(?i).*credential.*",
    r"(?i).*auth.*",
    r"(?i).*apiid.*",
    r"(?i).*accessid.*",
    r"(?i).*private.*",
    r"(?i)^AWS_SESSION.*",
    r"(?i)^GH_TOKEN$",
    r"(?i)^GITLAB.*TOKEN.*",
)

# Value shapes that are secrets regardless of the variable name.
SECRET_VALUE_RE = re.compile(
    r"""(
        sk-[A-Za-z0-9]{16,}          # OpenAI-style keys
      | ghp_[A-Za-z0-9]{20,}         # GitHub PAT
      | gho_[A-Za-z0-9]{20,}
      | AKIA[0-9A-Z]{16}             # AWS access key id
      | xox[baprs]-[A-Za-z0-9-]{10,} # Slack
      | -----BEGIN[ A-Z]*PRIVATE KEY-----
    )""",
    re.VERBOSE,
)

URL_CREDENTIALS_RE = re.compile(r"(?i)(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^/:@\s]+)(?::(?P<pw>[^@/\s]*))?@")

REDACTED = "***"
REDACTED_PREFIX = "***:"

# Variables that are useful for reproducibility and safe to keep even though their *name*
# matches a redaction pattern (they carry no secret value).
SAFE_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)


class Redactor:
    def __init__(self, extra_patterns: Iterable[str] = ()) -> None:
        raw = list(DEFAULT_REDACT_PATTERNS) + list(extra_patterns)
        self.patterns: list[str] = []
        compiled: list[re.Pattern[str]] = []
        for pattern in raw:
            try:
                compiled.append(re.compile(pattern))
                self.patterns.append(pattern)
            except re.error:
                # A bad user pattern must not take the whole run down; it is reported by
                # `labbox doctor` and simply not applied here.
                continue
        self._compiled = compiled

    def is_sensitive_name(self, name: str) -> bool:
        if name.upper() in SAFE_EXCEPTIONS:
            return False
        return any(p.search(name) for p in self._compiled)

    def scrub_value(self, value: str) -> str:
        """Remove secret-shaped substrings from a free-text value."""
        cleaned = SECRET_VALUE_RE.sub(REDACTED, value)
        return URL_CREDENTIALS_RE.sub(lambda m: f"{m.group('scheme')}***@", cleaned)

    def scrub_url(self, url: str) -> str:
        return URL_CREDENTIALS_RE.sub(lambda m: f"{m.group('scheme')}***@", url)

    def scrub_text(self, text: str) -> str:
        """For error messages, stderr snippets and anything embedded in a report."""
        return self.scrub_value(text)

    def filter_env(
        self,
        env: Mapping[str, str],
        *,
        keep_names: Iterable[str] = (),
        max_entries: int = 200,
        max_value_chars: int = 300,
    ) -> EnvVars:
        """Build the sanitized environment snapshot stored in ``environment.json``."""
        keep = set(keep_names)
        entries: dict[str, str] = {}
        redacted: list[str] = []
        for name in sorted(env):
            if self.is_sensitive_name(name) and name not in keep:
                redacted.append(name)
                continue
            value = env[name]
            if value is None:
                continue
            if len(value) > max_value_chars:
                value = value[:max_value_chars] + "...[truncated]"
            value = self.scrub_value(value)
            entries[name] = value
            if len(entries) >= max_entries:
                break
        return EnvVars(
            entries=entries,
            redacted_names=redacted,
            redact_patterns=list(self.patterns),
            total_count=len(env),
        )
