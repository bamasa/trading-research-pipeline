"""``trp demo --execution all`` on the smallest synthetic set: the page, the manifest, the discipline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from trading_research import fees
from trading_research.cli import app
from trading_research.walkthrough.demo import DemoConfig, fee_line
from trading_research.walkthrough.record import sha256_file

runner = CliRunner()


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("demo") / "out"
    result = runner.invoke(app, ["demo", "--execution", "all", "--small", "--out", str(out)])
    assert result.exit_code == 0, result.output
    return {"out": out, "output": result.output}


def test_the_page_and_the_manifest_are_written(demo: dict[str, Any]) -> None:
    out = demo["out"]
    page = (out / "page.md").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    for title in ("### Taker", "### Passive entry", "### Market maker", "## What was checked"):
        assert title in page
    assert re.search(r"Wall time: \*\*\d+ s\*\*", page)
    assert set(manifest["statuses"]) == {"taker", "passive", "maker"}
    assert manifest["seeds"]["instruments"] == {
        i.symbol: i.seed for i in DemoConfig.small().synthetic
    }


def test_the_fee_in_the_page_is_the_configurations(demo: dict[str, Any]) -> None:
    config = DemoConfig.small()
    row = fees.tier(config.fee)
    page = (demo["out"] / "page.md").read_text()
    assert fee_line(config) in page
    stated = re.search(r"tier (\w+): maker ([\d.]+) bp, taker ([\d.]+) bp", page)
    assert stated is not None
    assert (stated.group(1), float(stated.group(2)), float(stated.group(3))) == (
        row.tier, row.maker_bp, row.taker_bp,
    )  # fmt: skip
    registration = json.loads((demo["out"] / "search" / "registration.json").read_text())
    assert (registration["fee"]["maker_bp"], registration["fee"]["taker_bp"]) == (
        row.maker_bp,
        row.taker_bp,
    )


def test_every_hash_in_the_manifest_is_the_files(demo: dict[str, Any]) -> None:
    out = demo["out"]
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if name.endswith("page.md"):
            continue  # hashed when written; the manifest is written after it
        assert sha256_file(out / name) == digest, name


def test_the_held_out_block_was_read_once_and_a_second_read_refused(demo: dict[str, Any]) -> None:
    search = demo["out"] / "search"
    ledger = json.loads((search / "ledger.json").read_text())["H"]
    assert [entry["read"] for entry in ledger] == ["first read"]
    results = search / "6-read" / "results.json"
    assert ledger[0]["results_sha256"] == sha256_file(results)
    assert ledger[0]["frozen_sha256"] == sha256_file(search / "5-execution" / "frozen.json")
    assert "A second read of the held-out block: refused" in (demo["out"] / "page.md").read_text()
    held = json.loads(results.read_text())["held_out_days"]
    searched = list((search / "views" / "search").rglob("*.parquet"))
    assert searched and not [p for p in searched if p.stem in held]


def test_the_kills_were_registered_before_the_read_and_applied_in_order(
    demo: dict[str, Any],
) -> None:
    search = demo["out"] / "search"
    registration = json.loads((search / "registration.json").read_text())
    results = json.loads((search / "6-read" / "results.json").read_text())
    names = [k["name"] for k in registration["kill_conditions"]]
    assert results["kill_conditions"] == registration["kill_conditions"]
    for row in results["modes"].values():
        assert [c["condition"] for c in row["checks"]] == names
        assert row["status"] in {"killed", "inconclusive", "candidate"}
        assert row["fee_ladder"][0]["tier"] == "base"


def test_a_directory_that_is_not_a_demos_is_refused(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("mine")
    result = runner.invoke(app, ["demo", "--small", "--out", str(tmp_path)])
    assert result.exit_code == 2 and (tmp_path / "keep.txt").exists()
