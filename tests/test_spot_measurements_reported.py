"""Measurements the registration names and a review found unreported: FILL-S's
net_bp_trade both ways, the thinned calibration's capacity median, the calibration on
the page, a measured row on an unreadable venue, and the statuses after P before P."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trading_research.spot import access as sa
from trading_research.spot import assemble as asm
from trading_research.spot import pages
from trading_research.spot import registration as sr
from trading_research.spot.reads import final_statuses


def _records() -> dict[str, dict[str, Any]]:
    out = {}
    for k, (bb, pb, rule, walk, thinned) in enumerate([
        (1.0, 2.0, 3.0, 4.0, {"capacity": 0.01, "trades": 5, "binance_prints": 9}),
        (3.0, 4.0, 5.0, 6.0, {"capacity": 0.0, "trades": 0, "binance_prints": 9}),
        (5.0, 6.0, 7.0, 8.0, {"capacity": None, "trades": 3, "binance_prints": 9}),  # infinite
        (9.0, 9.0, 9.0, 9.0, None),  # Binance's day excluded: not used by the thinning
    ]):  # fmt: skip
        day = date(2026, 8, 8 + k).isoformat()
        out[day] = {"day": day, "kept": {sa.BYBIT: True},
                    "rows": {"S25-BB": {"value": bb}, "S25-PB": {"value": pb}},
                    "calibration": {"bb_print_rule": {"value": rule}, "pb_book": {"value": walk},
                                    "thinned": thinned}}  # fmt: skip
    return out


def test_fill_s_both_ways_and_the_thinned_capacity_median() -> None:
    records = _records()
    both = asm.both_ways(records)
    assert both["S25-BB"] == {"book_walk": 4.5, "print_rule": 6.0}
    assert both["S25-PB"] == {"print_rule": 5.25, "book_walk": 6.75}
    assert asm.thinned_capacity_median(records) == 0.01  # median of [0.01, 0, inf]
    records[next(iter(records))]["calibration"]["thinned"]["capacity"] = None
    assert math.isinf(asm.thinned_capacity_median(records) or 0.0)


def test_the_page_carries_the_calibration_and_a_not_assessed_measured_row() -> None:
    fill_s = {"rule": [{"side": "buy", "mean_advantage_bp": 1.5},
                       {"side": "sell", "mean_advantage_bp": -0.5}]}  # fmt: skip
    results = {
        "block": "P", "stamp": "first read", "statuses": {},
        "measurements": {"calibration": {"a_buy": 0.25, "a_sell": 0.75, "fill_s": {"S25-PB": fill_s},
                                         "net_bp_trade_both_ways": {"S25-BB": {"book_walk": -9.0}}},
                         "measured_rows_not_assessed": {"S25-PFT": "not assessed (P unreadable on Binance)"}},
    }  # fmt: skip
    lines = "\n".join(pages.block_lines(results))
    assert "thinned a_buy 0.25, a_sell 0.75 |" in lines
    assert "rule buy 1.50, sell -0.50" in lines and "book walk -9.00" in lines
    assert "| S25-PFT | not assessed (P unreadable on Binance) |" in lines


def _study(tmp_path: Path, deadline: date | None, today: date) -> Any:
    def p_deadline() -> date:
        if deadline is None:
            from trading_research.market_making.prereg import HeldOutLocked

            raise HeldOutLocked("not committed")
        return deadline

    ledger = SimpleNamespace(p_deadline=p_deadline, today=lambda: today)
    return SimpleNamespace(repo=tmp_path, ledger=lambda: ledger)


def test_the_statuses_after_p_before_p_is_read(tmp_path: Path) -> None:
    path = tmp_path / sr.RESULTS_PATHS["H"]
    path.parent.mkdir(parents=True)
    statuses = {"S25-PF": {"status": "candidate"}, "S25-BB": {"status": "killed"},
                "S25-T": {"status": "inconclusive"}}  # fmt: skip
    path.write_text(json.dumps({"block": "H", "blind": True, "statuses": statuses}))
    early = final_statuses(_study(tmp_path, date(2027, 1, 31), date(2027, 1, 2)))
    assert early == {"S25-PF": "candidate (P not yet read)", "S25-BB": "killed",
                     "S25-T": "inconclusive"}  # fmt: skip
    late = final_statuses(_study(tmp_path, date(2027, 1, 31), date(2027, 2, 1)))
    assert late["S25-PF"] == "candidate, not confirmed (P not read by its deadline)"
    assert final_statuses(_study(tmp_path, None, date(2027, 6, 1)))["S25-PF"] == (
        "candidate (P not yet read)"
    )
