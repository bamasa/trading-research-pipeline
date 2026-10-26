"""The replication's test days: S25-PF's decisions run by the spot print executor at two fees.

For each test day the window's fit scores every row from the day's scored start
(``spot.days.predict``), the hold-first rule decides (``spot.runs.decisions``),
and the spot print executor runs the decisions taker in and taker out by the
side-correct print rule at the registered settings
(``spot.print_taker.execute_print_day``), each fill paying the fee of the
registered fee table (``configs/spot_2026.yaml``), whose BTC/FDUSD interval from
2024-04-25 06:00 to 2026-01-29 covers every replication day. Two tiers:

* ``zero``: VIP 2 to 9, 0 maker and 0 taker; the same numbers as the zero fee
  the author's 2025 backtest charged on this pair;
* ``regular_10bp``: the Regular account, 0 maker and 10 bp taker.

Each is run for the two-sided slot and for the long-only variant
(``spot.account.long_only``: every short dropped before the one-position rule).
A fee changes no fill and no decision, so a day's trades are the same at both
tiers and only the money differs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from trading_research.features.timed import stamps_ns
from trading_research.replication.fdusd_fetch import ROOT
from trading_research.replication.fdusd_windows import HOLD_S, LAYOUT, open_day
from trading_research.spot import account as acc
from trading_research.spot import days as dy
from trading_research.spot.archives import BINANCE_SPOT
from trading_research.spot.costs import FeeTable
from trading_research.spot.print_taker import PrintDay, execute_print_day, print_settings
from trading_research.spot.runs import decisions
from trading_research.strategies.hold_first import HoldFirstRule

FEES: dict[str, str] = {"zero": "vip2_to_vip9", "regular_10bp": "regular"}
VARIANTS: tuple[str, ...] = ("two_sided", "long_only")
HOLD_NS = round(HOLD_S * 1e9)
COUNTERS = ("decisions", "entries_busy", "entries_no_print", "exits_no_print",
            "fill_beyond_sweep", "entry_late_print", "exit_late_print")  # fmt: skip


def _median(values: Any) -> float:
    found = np.asarray(values, dtype=np.float64)
    return float(np.median(found)) if len(found) else float("nan")


def money(day: PrintDay) -> dict[str, Any]:
    """One run's day in quote and bp (``spot.account.day_summary``), NaN without a trade."""
    t = day.trades
    out: dict[str, Any] = {k: int(day.counters[k]) for k in COUNTERS}
    if not len(t):
        return out | {"trades": 0, "net_quote": 0.0, "gross_quote": 0.0, "fees_quote": 0.0,
                      "entry_notional": 0.0, "turnover": 0.0}  # fmt: skip
    s = acc.day_summary(t)
    return out | {
        "trades": s.trades, "trades_long": s.trades_long, "trades_short": s.trades_short,
        "net_quote": s.net, "gross_quote": s.gross, "fees_quote": s.fees,
        "entry_notional": s.entry_notional, "turnover": s.turnover, "net_bp": s.net_bp,
        "gross_bp": s.gross_bp, "gross_bp_turnover": s.gross_bp_turnover,
        "net_bp_long": s.net_bp_long, "net_bp_short": s.net_bp_short,
        "net_quote_per_trade": s.net / s.trades,
        "hold_realised_s_median": _median(t["hold_realised_ns"] / 1e9),
        "entry_wait_s_median": _median(t["entry_wait_ns"] / 1e9),
    }  # fmt: skip


def score_day(
    window: str, model: Any, rule: HoldFirstRule, table: FeeTable, day: date,
    root: Path | str = ROOT,
) -> list[dict[str, Any]]:  # fmt: skip
    """Every (variant, fee) run of one test day; none for a day the loader excludes."""
    found = open_day(day, root)
    if not found.usable:
        return []
    frame = found.features()
    grid = stamps_ns(frame)
    tape = found.tape()
    rows, side = decisions(dy.predict(model, frame, LAYOUT, found.scored_from), rule)
    runs = {"two_sided": (grid[rows], side), "long_only": acc.long_only(grid[rows], side)}
    first, last = acc.first_last_print(tape)
    out = []
    for variant in VARIANTS:
        decided, sides = runs[variant]
        for fees, tier in FEES.items():
            settings = print_settings(table, BINANCE_SPOT, tier)
            run = execute_print_day(tape, grid, decided, sides, HOLD_NS, settings)
            out.append({"window": window, "day": day.isoformat(), "variant": variant,
                        "fees": fees, "tier": tier, "warm": found.warm, "grid_rows": len(grid),
                        "market_bp": (last / first - 1.0) * 1e4, **money(run)})  # fmt: skip
    return out
