"""Config value extraction: provenance and ambiguity (§9.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from labbox.config.model import LabboxConfig
from labbox.domain.hyperparams import Confidence, ValueSource


def extractor(root: Path, config: LabboxConfig, *, files=(), argv=()):
    from labbox.analysis.config_values import ConfigValueExtractor

    return ConfigValueExtractor(project_root=root, config=config, explicit_files=list(files), argv=list(argv))


@pytest.fixture
def config() -> LabboxConfig:
    cfg = LabboxConfig.default()
    cfg.configs.patterns = []  # no implicit scanning unless a test asks for it
    return cfg


def test_yaml_values_are_flattened_with_paths(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "base.yaml").write_text(
        "train:\n  batch_size: 64\n  amp: true\n  optimizer:\n    lr: 0.001\n    name: adamw\n"
        "model:\n  name: resnet50\n  dropout: 0.2\n",
        encoding="utf-8",
    )
    result = extractor(tmp_path, config, files=["base.yaml"]).extract()
    values = result.as_map()
    assert values["train.batch_size"].value == 64
    assert values["train.amp"].value is True
    assert values["train.optimizer.lr"].value == 0.001
    assert values["model.name"].value == "resnet50"
    assert values["train.batch_size"].sources == ["base.yaml"]
    assert result.files_read == ["base.yaml"]


def test_json_and_toml_are_supported(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "a.json").write_text('{"epochs": 20, "nested": {"warmup": 0.1}}', encoding="utf-8")
    (tmp_path / "b.toml").write_text("[sched]\ncosine = true\nperiod = 5\n", encoding="utf-8")
    result = extractor(tmp_path, config, files=["a.json", "b.toml"]).extract()
    values = result.as_map()
    assert values["epochs"].value == 20
    assert values["nested.warmup"].value == 0.1
    assert values["sched.cosine"].value is True
    assert values["sched.period"].value == 5


def test_unsupported_formats_are_skipped_with_a_reason(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "conf.ini").write_text("[a]\nb=1\n", encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("a: [1,\n  b: }\n", encoding="utf-8")
    result = extractor(tmp_path, config, files=["conf.ini", "bad.yaml"]).extract()
    skipped = {item["path"] for item in result.files_skipped}
    assert "conf.ini" in skipped
    assert any("bad.yaml" in path for path in skipped), skipped


def test_command_line_overrides_config_and_both_sources_are_kept(
    tmp_path: Path, config: LabboxConfig
) -> None:
    (tmp_path / "base.yaml").write_text("train:\n  batch_size: 64\n", encoding="utf-8")
    argv = ["python", "train.py", "--batch-size", "128"]
    result = extractor(tmp_path, config, files=["base.yaml"], argv=argv).extract()
    value = result.as_map()["train.batch_size"]
    assert value.value == 128
    assert ValueSource.COMMAND_ARG in value.source_kinds
    assert ValueSource.CONFIG_FILE in value.source_kinds
    assert len(value.sources) == 2, "the config-file origin stays visible"


def test_conflicting_file_sources_are_marked_ambiguous(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "a.yaml").write_text("train:\n  lr: 0.1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("train:\n  lr: 0.001\n", encoding="utf-8")
    result = extractor(tmp_path, config, files=["a.yaml", "b.yaml"]).extract()
    value = result.as_map()["train.lr"]
    assert value.confidence in {Confidence.UNKNOWN, Confidence.MEDIUM}
    assert value.ambiguous is True
    assert any("hyperparam_source_ambiguous" in warning for warning in result.warnings)


def test_secret_shaped_keys_are_never_extracted(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "base.yaml").write_text(
        "wandb:\n  api_key: secret-value-123\n  project: myproj\n  password: pw\n", encoding="utf-8"
    )
    result = extractor(tmp_path, config, files=["base.yaml"]).extract()
    blob = repr(result.model_dump(mode="json"))
    assert "secret-value-123" not in blob
    assert "pw" not in blob.split()
    assert "wandb.project" in result.as_map()


def test_argv_flags_without_values_become_booleans(tmp_path: Path, config: LabboxConfig) -> None:
    result = extractor(tmp_path, config, argv=["python", "train.py", "--amp", "--workers=4"]).extract()
    values = result.as_map()
    assert values["train.amp"].value is True
    assert values["train.workers"].value == 4


def test_the_end_of_options_marker_is_not_a_hyperparameter(tmp_path: Path, config: LabboxConfig) -> None:
    """``--`` used to become a value called ``train.`` holding True."""
    result = extractor(tmp_path, config, argv=["python", "train.py", "--amp", "--", "--raw-arg"]).extract()
    values = result.as_map()
    assert "train." not in values
    assert "train.raw_arg" not in values
    assert values["train.amp"].value is True


def test_a_negative_number_is_a_value_not_the_next_flag(tmp_path: Path, config: LabboxConfig) -> None:
    result = extractor(tmp_path, config, argv=["python", "train.py", "--lr", "-0.1", "--temp", "-1"]).extract()
    values = result.as_map()
    assert values["train.lr"].value == pytest.approx(-0.1)
    assert values["train.temp"].value == -1


def test_the_next_flag_is_not_swallowed_as_a_value(tmp_path: Path, config: LabboxConfig) -> None:
    result = extractor(tmp_path, config, argv=["python", "train.py", "--amp", "--verbose"]).extract()
    values = result.as_map()
    assert values["train.amp"].value is True
    assert values["train.verbose"].value is True


def test_an_invalid_drop_pattern_is_reported_not_fatal(tmp_path: Path, config: LabboxConfig) -> None:
    """A typo in drop_key_patterns used to raise and cost the run every hyperparameter."""
    (tmp_path / "base.yaml").write_text("train:\n  lr: 0.1\n", encoding="utf-8")
    config.configs.drop_key_patterns = ["(?i).*password.*", "(unclosed"]
    result = extractor(tmp_path, config, files=["base.yaml"]).extract()
    assert result.as_map()["train.lr"].value == 0.1
    assert any("drop_key_pattern" in warning for warning in result.warnings)


def test_config_path_mentioned_in_argv_is_picked_up(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("train:\n  epochs: 90\n", encoding="utf-8")
    result = extractor(
        tmp_path, config, argv=["python", "train.py", "--config", "configs/base.yaml"]
    ).extract()
    assert result.as_map()["train.epochs"].value == 90
    assert "configs/base.yaml" in result.files_read


def test_depth_and_list_limits(tmp_path: Path, config: LabboxConfig) -> None:
    nested = [
        "a:",
        "  b:",
        "    c:",
        "      d:",
        "        e:",
        "          f:",
        "            g: 1",
        "  list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
    ]
    yaml_text = chr(10).join(nested) + chr(10)
    (tmp_path / "deep.yaml").write_text(yaml_text, encoding="utf-8")
    result = extractor(tmp_path, config, files=["deep.yaml"]).extract()
    values = result.as_map()
    assert any(key.startswith("a.b.c.d.e") for key in values)
    assert values["a.list"].value_repr == "[1, 2, 3, 4, 5, 6, 7, 8, ... 2 more]"


def test_no_config_file_is_reported_not_invented(tmp_path: Path, config: LabboxConfig) -> None:
    result = extractor(tmp_path, config, argv=["python", "train.py", "--lr", "0.2"]).extract()
    assert result.warnings == ["no config file found: hyperparameters come from the command line only"]
    assert result.as_map()["train.lr"].value == 0.2


def test_extract_keys_allowlist_filters_the_output(tmp_path: Path, config: LabboxConfig) -> None:
    (tmp_path / "base.yaml").write_text("train:\n  lr: 0.1\n  seed: 7\n  unused: 3\n", encoding="utf-8")
    config.configs.extract_keys = ["train.lr", "train.seed"]
    result = extractor(tmp_path, config, files=["base.yaml"]).extract()
    assert set(result.as_map()) == {"train.lr", "train.seed"}


def test_paths_with_spaces_and_cjk_are_handled(tmp_path: Path, config: LabboxConfig) -> None:
    directory = tmp_path / "配置 目录"
    directory.mkdir()
    (directory / "base.yaml").write_text("train:\n  epochs: 5\n", encoding="utf-8")
    result = extractor(tmp_path, config, files=[str(directory / "base.yaml")]).extract()
    assert result.as_map()["train.epochs"].value == 5
    assert result.files_read, "an absolute path outside the root is still recorded"
