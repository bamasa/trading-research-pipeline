"""One day's record: every run of every slot, on a stand-in day of prints."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest
import yaml

from trading_research.spot.costs import FeeTable
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import day as dd
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import signal as sg
from trading_research.spot_maker import standin
from trading_research.spot_maker.touch import Proxies

REPO = Path(__file__).resolve().parents[1]
DAYS = [date(2026, 5, 1), date(2026, 5, 2)]


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("day")
    source, repo = root / "source", root / "repo"
    binding = standin.write_source(source, DAYS)
    sg.copy_signal(source, repo, binding)
    cp.copy_days(source, repo, DAYS, bk.development_access())
    config = yaml.safe_load((REPO / rg.CONFIG_PATH).read_text(encoding="utf-8"))
    table = FeeTable.from_config(config["fees"])
    day = list(cp.walk(repo, DAYS, bk.development_access(), table))[1]
    frozen = sg.load_signal(repo, binding)
    decisions = sg.day_decisions(frozen, day.frame, day.history, day.scored_from)
    tape = day.tape()
    ctx = dd.DayContext(tape, Proxies.from_tape(tape, day.carried), decisions, table)
    return dd.day_record(ctx, warm=day.warm)


def test_every_slot_has_every_run(record: dict) -> None:
    assert record["warm"] and record["signal"]["decisions"] > 0
    for slot, spec in rg.SLOTS.items():
        found = record["slots"][slot]
        assert found["void"] == []
        assert len(found["runs"]["cells"]) == 9
        assert ("random" in found["runs"]) == spec.signal
        if spec.signal:
            assert len(found["runs"]["random"]) == rg.RANDOM_DRAWS
        assert found["true"]["fills"] >= 0 and "markout_5s_sum" in found["true"]


def test_the_own_cell_is_the_true_run_and_values_are_numbers(record: dict) -> None:
    for slot, spec in rg.SLOTS.items():
        found = record["slots"][slot]
        own = spec.cells().index((spec.requote_s, spec.skew_bp))
        value = found["value"]
        assert (math.isnan(value) and math.isnan(found["runs"]["cells"][own])) or (
            found["runs"]["cells"][own] == value
        )
        assert found["true"]["tiers"]["zero_fees"] >= found["true"]["net_bp_fill"] or math.isnan(value)
