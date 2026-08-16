"""Tests for experiment configuration.

The behaviour worth protecting is strictness. A config file is the record of
what a result assumed, so a typo that is silently ignored — leaving a default in
place while the reader believes otherwise — is worse than a crash.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trading_research.config import RunConfig, dump_config, load_config


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_defaults_form_a_valid_run() -> None:
    config = RunConfig()
    assert config.data.source == "synthetic"
    assert config.split.embargo is None


def test_minimal_file_loads(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, {"name": "demo"}))
    assert config.name == "demo"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo must fail loudly rather than leave a default in place."""
    path = _write(tmp_path, {"nmae": "demo"})
    with pytest.raises(ValidationError, match="nmae"):
        load_config(path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"costs": {"fee_bp": 2.0}})
    with pytest.raises(ValidationError, match="fee_bp"):
        load_config(path)


def test_split_fractions_must_leave_room_for_test(tmp_path: Path) -> None:
    path = _write(tmp_path, {"split": {"train_fraction": 0.8, "validation_fraction": 0.3}})
    with pytest.raises(ValidationError, match="test block"):
        load_config(path)


def test_real_data_requires_a_path(tmp_path: Path) -> None:
    path = _write(tmp_path, {"data": {"source": "binance", "symbol": "BTCUSDT"}})
    with pytest.raises(ValidationError, match=re.escape("data.path")):
        load_config(path)


def test_negative_costs_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"costs": {"fee_bp_per_side": -1.0}})
    with pytest.raises(ValidationError):
        load_config(path)


def test_label_horizon_must_be_positive(tmp_path: Path) -> None:
    path = _write(tmp_path, {"label": {"horizon": 0}})
    with pytest.raises(ValidationError):
        load_config(path)


def test_config_is_frozen() -> None:
    """A config mutated mid-run would no longer describe the result."""
    config = RunConfig()
    with pytest.raises(ValidationError):
        config.name = "changed"  # type: ignore[misc]


def test_round_trip_through_disk_is_lossless(tmp_path: Path) -> None:
    """A report must be replayable from the config stored beside it."""
    original = RunConfig(name="rt", label={"horizon": 33, "threshold_bp": 2.5})  # type: ignore[arg-type]
    dumped = dump_config(original, tmp_path / "out.yaml")
    assert load_config(dumped) == original


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
