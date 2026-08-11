"""End-to-end tests for the command line interface.

These are the smoke tests that stand in for a user following the quickstart. A
unit test can pass while the command a reader is told to run does not exist, so
the commands are exercised the way the README describes them.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lobml.cli import app
from lobml.data import store
from lobml.data.validate import validate_book, validate_trades

runner = CliRunner()


def test_version_is_reported() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "lobml" in result.stdout


def test_generate_then_validate(tmp_path: Path) -> None:
    """The quickstart path: generate a dataset, then check it."""
    out = tmp_path / "demo"

    generated = runner.invoke(app, ["generate-demo-data", "--output", str(out), "--steps", "800"])
    assert generated.exit_code == 0, generated.stdout

    validated = runner.invoke(app, ["validate-data", "--input", str(out)])
    assert validated.exit_code == 0, validated.stdout
    assert "Validation passed" in validated.stdout


def test_generated_output_is_labelled_synthetic(tmp_path: Path) -> None:
    """A reader must not be able to mistake demo output for a real result."""
    out = tmp_path / "demo"
    result = runner.invoke(app, ["generate-demo-data", "--output", str(out), "--steps", "500"])
    assert "Synthetic data" in result.stdout

    manifest = store.read_manifest(out)
    assert "SYNTHETIC" in manifest["warning"]


def test_generated_data_passes_validation_directly(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    runner.invoke(app, ["generate-demo-data", "--output", str(out), "--steps", "500"])
    assert validate_book(store.read_book(out)).ok
    assert validate_trades(store.read_trades(out)).ok


def test_seed_makes_the_demo_reproducible(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for path in (first, second):
        runner.invoke(
            app, ["generate-demo-data", "--output", str(path), "--steps", "400", "--seed", "99"]
        )
    assert store.read_book(first).equals(store.read_book(second))


def test_signal_strength_can_be_switched_off(tmp_path: Path) -> None:
    """The no-edge dataset the leakage tests need is reachable from the CLI."""
    out = tmp_path / "flat"
    result = runner.invoke(
        app,
        ["generate-demo-data", "--output", str(out), "--steps", "400", "--signal-strength", "0"],
    )
    assert result.exit_code == 0
    assert store.read_manifest(out)["config"]["signal_strength"] == 0.0


def test_validate_reports_a_missing_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-data", "--input", str(tmp_path / "absent")])
    assert result.exit_code == 2


def test_validate_rejects_a_directory_without_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["validate-data", "--input", str(empty)])
    assert result.exit_code == 2


def test_describe_schema_prints_the_contract() -> None:
    result = runner.invoke(app, ["describe-schema", "book"])
    assert result.exit_code == 0
    assert "sequence_id" in result.stdout


def test_describe_schema_rejects_an_unknown_plane() -> None:
    result = runner.invoke(app, ["describe-schema", "orders"])
    assert result.exit_code == 2
