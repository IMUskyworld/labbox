"""Extract hyperparameter values from config files and the command line (§9.6).

The rule that keeps `compare` honest: a value is only a fact if we know where it came
from. When the same key is set in a config file *and* on the command line we keep both
sources and prefer argv, and when we cannot tell which one won we mark the value
``confidence=unknown`` and emit a diagnostic — silently picking a side is how a diff
becomes a lie.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from labbox.config.model import LabboxConfig
from labbox.domain.hyperparams import Confidence, HyperparamSet, HyperValue, ValueSource

_SUPPORTED_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
_DEPTH_LIMIT = 6
_KEY_RE = re.compile(r"^[^.=\s][^.=/]{0,63}$")


class ConfigValueExtractor:
    def __init__(
        self,
        *,
        project_root: Path,
        config: LabboxConfig,
        explicit_files: Sequence[str] = (),
        argv: Sequence[str] = (),
    ) -> None:
        self.root = Path(project_root)
        self.config = config
        self.explicit = list(explicit_files)
        self.argv = [str(token) for token in argv]
        # An invalid regex in config is a user typo, not a reason to lose every
        # hyperparameter: it used to raise inside the constructor, which the run service
        # turned into a stage failure and which therefore dropped config-values.json
        # entirely. The bad patterns are reported and the good ones still apply.
        self._drop: list[re.Pattern[str]] = []
        self._invalid_drop: list[str] = []
        for pattern in config.configs.drop_key_patterns:
            try:
                self._drop.append(re.compile(pattern))
            except re.error as exc:
                self._invalid_drop.append(f"{pattern!r}: {exc}")

    def extract(self) -> HyperparamSet:
        result = HyperparamSet(flatten_depth_limit=_DEPTH_LIMIT)
        for problem in self._invalid_drop:
            result.warnings.append(f"config drop_key_pattern ignored (invalid regex): {problem}")
        values: dict[str, HyperValue] = {}
        for path in self._files(result):
            try:
                document = _load_document(path)
            except _ConfigError as exc:
                result.files_skipped.append({"path": str(_display(self.root, path)), "reason": str(exc)})
                continue
            for dotted, scalar in _flatten(document):
                if self._is_dropped(dotted):
                    continue
                if self.config.configs.extract_keys and not _in_allowlist(dotted, self.config.configs.extract_keys):
                    continue
                location = f"{_display(self.root, path)}"
                _add(values, dotted, scalar, source_kind=ValueSource.CONFIG_FILE, location=location)
        for dotted, scalar, location in self._from_argv():
            _add(values, dotted, scalar, source_kind=ValueSource.COMMAND_ARG, location=location, prefer=True)
        result.values = sorted(values.values(), key=lambda item: item.path)
        ambiguity = {warning for value in result.values for warning in _warnings_of(value)}
        result.warnings = list(dict.fromkeys([*result.warnings, *sorted(ambiguity)]))
        return result

    # --- sources -------------------------------------------------------------
    def _files(self, result: HyperparamSet) -> list[Path]:
        candidates: list[Path] = []
        for spec in self.explicit:
            path = Path(spec)
            candidates.append(path if path.is_absolute() else self.root / path)
        # `--config configs/base.yaml` inside the training command is the common case.
        for index, token in enumerate(self.argv):
            if token.startswith("--config"):
                if "=" in token:
                    candidates.append(self.root / token.split("=", 1)[1])
                elif index + 1 < len(self.argv):
                    candidates.append(self.root / self.argv[index + 1])
            elif token.endswith(tuple(_SUPPORTED_SUFFIXES)) and not token.startswith("-"):
                candidates.append(self.root / token)
        for pattern in self.config.configs.patterns:
            try:
                candidates.extend(sorted(self.root.glob(pattern)))
            except OSError:  # pragma: no cover
                continue
        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            if not path.is_file():
                if path.exists():
                    result.files_skipped.append({"path": _display(self.root, path), "reason": "not a file"})
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                result.files_skipped.append(
                    {
                        "path": _display(self.root, path),
                        "reason": f"unsupported format {path.suffix or '(no extension)'}, reads "
                        + "/".join(sorted(_SUPPORTED_SUFFIXES)),
                    }
                )
                continue
            size = path.stat().st_size
            if size > self.config.configs.max_file_bytes:
                result.files_skipped.append(
                    {"path": _display(self.root, path), "reason": f"larger than max_file_bytes={size}"}
                )
                continue
            unique.append(path)
        result.files_read = [_display(self.root, path) for path in unique]
        if not unique:
            result.warnings.append("no config file found: hyperparameters come from the command line only")
        return unique

    def _from_argv(self) -> list[tuple[str, Any, str]]:
        out: list[tuple[str, Any, str]] = []
        index = 0
        while index < len(self.argv):
            token = self.argv[index]
            if token == "--":
                # POSIX end-of-options marker. Without this it was parsed as a flag whose
                # name strips down to "" and produced a hyperparameter literally called
                # "train." with the value True.
                break
            if not token.startswith("-"):
                index += 1
                continue
            name, sep, inline = token.partition("=")
            key = name.lstrip("-").replace("-", "_")
            if not key:
                index += 1
                continue
            if sep:
                out.append((f"train.{key}", _scalar(inline), f"command.argv:{index}"))
            elif index + 1 < len(self.argv) and _is_value(self.argv[index + 1]):
                value = self.argv[index + 1]
                out.append((f"train.{key}", _scalar(value), f"command.argv:{index}"))
                index += 1
            else:
                out.append((f"train.{key}", True, f"command.argv:{index}"))
            index += 1
        return out

    def _is_dropped(self, dotted: str) -> bool:
        leaf = dotted.rsplit(".", 1)[-1]
        return any(pattern.search(leaf) or pattern.search(dotted) for pattern in self._drop)


class _ConfigError(Exception):
    pass


def _load_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _ConfigError(f"invalid JSON: {exc}") from exc
    elif suffix == ".toml":
        import tomllib

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise _ConfigError(f"invalid TOML: {exc}") from exc
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise _ConfigError("PyYAML is not installed in the LabBox environment") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise _ConfigError(f"top level is {type(data).__name__}, not a mapping")
    return data


def _flatten(document: dict[str, Any], prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for raw_key, value in document.items():
        key = str(raw_key)
        if not _KEY_RE.match(key):
            continue
        dotted = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict) and depth < _DEPTH_LIMIT:
            out.extend(_flatten(value, prefix=dotted, depth=depth + 1))
        elif isinstance(value, dict):
            out.append((dotted, f"<nested {len(value)} keys, depth limit>"))
        elif isinstance(value, (list, tuple)):
            out.append((dotted, _compact_list(value)))
        elif value is None or isinstance(value, (bool, int, float, str)):
            out.append((dotted, value))
    return out


def _compact_list(value: Sequence[Any]) -> str:
    parts = [str(item) for item in list(value)[:8]]
    if len(value) > 8:
        parts.append(f"... {len(value) - 8} more")
    return "[" + ", ".join(parts) + "]"


def _is_value(token: str) -> bool:
    """Whether *token* is this flag's value rather than the next flag.

    A leading dash is the usual "this is another flag" signal, but a negative number is a
    perfectly ordinary value (``--lr -0.1``, ``--lr=-0.1``), and consuming the next flag
    instead turned ``--amp -v`` into a hyperparameter ``train.amp = "-v"``.
    """
    if not token.startswith("-"):
        return True
    return _NEGATIVE_NUMBER.match(token) is not None


_NEGATIVE_NUMBER = re.compile(r"^-\d+(\.\d*)?([eE][-+]?\d+)?$")


def _scalar(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _add(
    values: dict[str, HyperValue],
    dotted: str,
    scalar: Any,
    *,
    source_kind: ValueSource,
    location: str,
    prefer: bool = False,
) -> None:
    existing = values.get(dotted)
    if existing is None:
        values[dotted] = HyperValue(
            path=dotted,
            value=scalar,
            value_repr=_repr_value(scalar),
            sources=[location],
            source_kinds=[source_kind],
            raw_type=type(scalar).__name__,
            confidence=Confidence.HIGH if source_kind is not ValueSource.CONFIG_FILE else Confidence.MEDIUM,
        )
        return
    if source_kind in existing.source_kinds and _repr_value(scalar) == existing.value_repr:
        if location not in existing.sources:
            existing.sources.append(location)
        return
    # Two sources disagree (or agree from a second file): record both, never hide it.
    if prefer:
        existing.value = scalar
        existing.value_repr = _repr_value(scalar)
        existing.raw_type = type(scalar).__name__
        existing.confidence = Confidence.HIGH
    else:
        existing.confidence = Confidence.UNKNOWN
        existing.ambiguous = True
    for kind in (source_kind,):
        if kind not in existing.source_kinds:
            existing.source_kinds.append(kind)
    if location not in existing.sources:
        existing.sources.append(location)


def _repr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _warnings_of(value: HyperValue) -> list[str]:
    if value.ambiguous or value.confidence is Confidence.UNKNOWN:
        return [f"hyperparam_source_ambiguous: {value.path} set by {', '.join(value.sources)}"]
    return []


def _in_allowlist(dotted: str, allow: Sequence[str]) -> bool:
    return any(dotted == entry or dotted.startswith(entry + ".") for entry in allow)


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return str(path).replace("\\", "/")
