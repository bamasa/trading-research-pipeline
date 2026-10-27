"""Stage 0's checks: the readers' proof, every audited order re-filled, and probe orders."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from trading_research.spot.costs import FeeTable
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import checks as ck
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import day as dd
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import resting, standin
from trading_research.spot_maker import signal as sg
from trading_research.spot_maker.executor import settings_for
from trading_research.spot_maker.touch import Proxies

REPO = Path(__file__).resolve().parents[1]
DAYS = [date(2026, 3, 3), date(2026, 3, 4)]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("checks")
    source, repo = root / "source", root / "repo"
    binding = standin.write_source(source, DAYS)
    sg.copy_signal(source, repo, binding)
    access = bk.reader_test_access("stage 0 timing")
    cp.copy_days(source, repo, DAYS, bk.reader_test_access("readers' proof"))
    config = yaml.safe_load((REPO / rg.CONFIG_PATH).read_text(encoding="utf-8"))
    table = FeeTable.from_config(config["fees"])
    day = list(cp.walk(repo, DAYS, access, table))[1]
    decisions = sg.day_decisions(sg.load_signal(repo, binding), day.frame, day.history,
                                 day.scored_from)  # fmt: skip
    tape = day.tape()
    ctx = dd.DayContext(tape, Proxies.from_tape(tape, day.carried), decisions, table)
    return {"day": day, "ctx": ctx}


def test_the_readers_proof_checks_contract_sides_and_tick(loaded: dict[str, Any]) -> None:
    found = ck.reader_proof(loaded["day"])
    assert found["status"] == "usable" and found["contract"] and found["sides_present"]
    assert found["tick_registered"] and found["rows"] > 0


def test_every_audited_order_passes_and_a_tampered_one_is_caught(loaded: dict[str, Any]) -> None:
    ctx = loaded["ctx"]
    audit = ck.audit_day(ctx)
    for slot in rg.FAMILY:
        for rule in resting.RULES:
            assert audit[slot][rule]["orders"] > 0
            assert audit[slot][rule]["violations"] == 0 and audit[slot][rule]["voids"] == []
    run = ctx.runner("MK-Q")(settings_for("MK-Q", audit=True), None, "both")
    assert run.orders is not None
    k = next(i for i, o in enumerate(run.orders) if len(o[5]))
    side, price, size, start, end, stamps, sizes = run.orders[k]
    run.orders[k] = (side, price, size, start, end, stamps, sizes * 2.0)  # more than it filled
    assert ck.order_checks(ctx.tape, run, "trade_through")["violations"] == 1
    run.orders[k] = (side, price, size, start, end, stamps - (end - start), sizes)  # outside
    assert ck.order_checks(ctx.tape, run, "trade_through")["violations"] == 1
    with pytest.raises(ValueError):
        ck.order_checks(ctx.tape, ctx.runner("MK-Q")(settings_for("MK-Q"), None, "both"),
                        "trade_through")  # fmt: skip


def test_probe_orders_nest_under_the_three_rules(loaded: dict[str, Any]) -> None:
    ctx = loaded["ctx"]
    found = ck.rule_checks(ctx.tape, ctx.proxies)
    assert found["passes"] and found["orders"] > 2_000
    assert np.isfinite(found["orders"])
