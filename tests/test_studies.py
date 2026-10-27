"""The study runner: every step through the study's own entry point, refusals included."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.pipeline.short_horizon import state as short_horizon
from trading_research.spot import state as spot
from trading_research.studies import (
    EntryPointStudy,
    RegisteredStudy,
    create,
    get,
    null_sha256,
    studies,
)
from trading_research.studies.template import unfilled

REPO = Path(__file__).resolve().parent.parent


def scratch(tmp_path: Path, name: str) -> Path:
    """A repository holding the study's entry point and nothing else: no state, no ledger."""
    (tmp_path / "experiments").mkdir()
    shutil.copyfile(REPO / "experiments" / "__init__.py", tmp_path / "experiments" / "__init__.py")
    shutil.copyfile(REPO / "experiments" / f"{name}.py", tmp_path / "experiments" / f"{name}.py")
    return tmp_path


def test_both_studies_of_this_branch_are_registered_with_their_own_stages() -> None:
    found = studies(REPO)
    assert list(found)[:2] == ["short_horizon_2025", "spot_2026"]
    assert all(isinstance(s, RegisteredStudy) for s in found.values())
    sh, sp = found["short_horizon_2025"], found["spot_2026"]
    assert sh.stages()[: len(short_horizon.STAGES)] == short_horizon.STAGES
    assert sh.stages()[len(short_horizon.STAGES) :] == (
        "open H", "read H", "open F", "read F", "open P", "read P",
    )  # fmt: skip
    assert sp.stages() == (*spot.STAGES, "open H", "read H", "open P", "read P")
    assert sp.registration == REPO / "docs/preregistration/spot_2026.md"
    assert sp.config.exists() and sp.entry_point.exists()
    with pytest.raises(KeyError, match="registered: short_horizon_2025, spot_2026"):
        get("nonesuch", REPO)


def test_arguments_are_the_entry_points_own() -> None:
    sp = get("spot_2026", REPO)
    assert sp.argv("read H") == ["read", "H"]
    assert sp.argv("open", "P", workers=4, frozen_2025=Path("x")) == [
        "open", "P", "--workers", "4", "--frozen-2025", "x",
    ]  # fmt: skip
    assert sp.argv("1", dry_run=False, run=2) == ["1", "--run", "2"]


@pytest.mark.parametrize("name", ["spot_2026", "short_horizon_2025"])
def test_a_read_is_refused_exactly_as_the_entry_point_refuses_it(tmp_path: Path, name: str) -> None:
    repo = scratch(tmp_path, name)
    study = get(name, repo)
    assert isinstance(study, EntryPointStudy) and study.repo == repo.resolve()
    status = study.status()
    assert status.ok and set(status.result.values()) == {"not run"}
    for step in ("read H", "1"):
        found = study.run(step)
        assert found.refused and found.returncode == 2, found.stderr
        assert found.stderr.startswith("refused:")
    assert not (repo / "artifacts").exists() and not (repo / "experiments" / "results").exists()


def test_the_command_relays_the_entry_points_refusal(tmp_path: Path) -> None:
    repo = scratch(tmp_path, "spot_2026")
    runner = CliRunner()
    result = runner.invoke(app, ["study", "run", "spot_2026", "read", "H", "--repo", str(repo)])
    assert result.exit_code == 2
    assert "python -m experiments.spot_2026 read H" in result.output
    status = runner.invoke(app, ["study", "status", "spot_2026", "--repo", str(repo)])
    assert status.exit_code == 0 and '"D": "not run"' in status.output
    assert runner.invoke(app, ["study", "list"]).exit_code == 0
    assert "open P" in runner.invoke(app, ["study", "stages", "spot_2026"]).output


def fill(path: Path) -> None:
    """Every author's placeholder given a value, as an author would."""
    text = path.read_text()
    for placeholder in unfilled(path):
        value = "2031-01-01" if "YYYY" in placeholder else "x"
        text = text.replace(placeholder, value)
    path.write_text(text)


@pytest.mark.parametrize("template", ["taker", "passive", "maker"])
def test_a_new_study_refuses_until_its_registration_is_filled(
    tmp_path: Path, template: str
) -> None:
    (tmp_path / "experiments").mkdir()
    shutil.copyfile(REPO / "experiments" / "__init__.py", tmp_path / "experiments" / "__init__.py")
    made = create(tmp_path, "demo_carry", template)
    config = yaml.safe_load(made.config.read_text())
    assert config["template"] == template and config["family"]["m"] == 1
    assert {"K1", "K-days", "K-placebo"} <= set(config["kill_conditions"])
    assert config["frozen"]["thresholds"] is None
    document = made.document.read_text()
    for heading in ("## Blocks", "## The slots", "### Kill conditions", "### Placebos",
                    "## Status rules", "### The code to come", "### Access",
                    "## Amendment protocol", "Null-placeholder sha256:"):  # fmt: skip
        assert heading in document, heading
    assert "Holm" in document and "Code-freeze commit" in document and "open entry" in document

    study = get("demo_carry", tmp_path)
    assert study.stages() == ("0", "D", "amend", "open H", "read H")
    assert study.status().ok
    refused = study.run("0")
    assert refused.refused and "placeholder" in refused.stderr

    fill(made.config)
    fill(made.document)
    digest = null_sha256(made.config.read_text())
    assert "null-placeholder sha256" in study.run("0").stderr  # the hash is not in the document yet
    made.document.write_text(re.sub(r"Null-placeholder sha256: `[^`]*`",
                                    f"Null-placeholder sha256: `{digest}`", made.document.read_text()))  # fmt: skip
    last = study.run("read", "H")
    assert last.refused and "code is not written yet" in last.stderr
    with pytest.raises(FileExistsError):
        create(tmp_path, "demo_carry", template)


def test_new_refuses_a_bad_name_or_template(tmp_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["study", "new", "Bad-Name", "--repo", str(tmp_path)]).exit_code == 2
    result = runner.invoke(app, ["study", "new", "ok", "--template", "x", "--repo", str(tmp_path)])
    assert result.exit_code == 2 and "template" in result.output
