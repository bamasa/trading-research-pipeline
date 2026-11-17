"""The regime suite: its episodes, its market built from bars, and its summary."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.bench import builder
from trading_research.bench.manifest import FORWARD_STRATEGIES
from trading_research.bench.regimes import (
    EPISODES,
    HALF_SPREAD_BP,
    PANEL,
    EpisodeScore,
    bar_grid,
    needs_sizes,
    page,
    ruined,
    summary,
)
from trading_research.plugins import get


def test_ten_dated_breaks_outside_every_held_out_block() -> None:
    assert len(EPISODES) == 10 and len({e.name for e in EPISODES}) == 10
    held_out = [(date(2025, 7, 1), date(2025, 12, 31)), (date(2026, 10, 9), date(2026, 11, 19))]
    for ep in EPISODES:
        assert ep.fit_start < ep.scored_start < ep.breaks.date() <= ep.last_day
        for first, last in held_out:
            assert ep.last_day < first or ep.fit_start > last, ep.name
    basket = {"BICOUSDT", "ALICEUSDT", "ALGOUSDT", "JTOUSDT", "ZECUSDT", "GALAUSDT", "CAKEUSDT"}
    assert not basket & set(PANEL)
    assert set(HALF_SPREAD_BP) == set(PANEL)


def test_a_bar_is_seen_only_after_it_has_closed() -> None:
    closes = [pd.Timestamp(f"2024-01-01 00:0{m}:59.999", tz="UTC") for m in range(3)]
    bars = pd.DataFrame({"timestamp": closes, "close": [100.0, 101.0, 102.0]})
    grid = bar_grid(bars, "BTCUSDT", date(2024, 1, 1), date(2024, 1, 1))
    mid = (grid["bid_price_0"] + grid["ask_price_0"]) / 2
    at = dict(zip(grid["timestamp"], mid, strict=True))
    assert pd.Timestamp("2024-01-01 00:00:55", tz="UTC") not in at  # nothing had closed yet
    assert at[pd.Timestamp("2024-01-01 00:01:00", tz="UTC")] == 100.0
    assert at[pd.Timestamp("2024-01-01 00:01:55", tz="UTC")] == 100.0
    assert at[pd.Timestamp("2024-01-01 00:02:00", tz="UTC")] == 101.0
    half = (grid["ask_price_0"] - grid["bid_price_0"]) / (2 * mid) * 1e4
    assert np.allclose(half, HALF_SPREAD_BP["BTCUSDT"]) and grid["bid_size_0"].isna().all()


def test_only_the_price_strategies_run_on_bars() -> None:
    runs = {
        name for name, kind in FORWARD_STRATEGIES if not needs_sizes(builder(get(name, kind), {})())
    }
    assert runs == {"relative", "reversion", "spread"}


def test_the_summary_counts_what_held_what_broke_and_what_the_switch_kept() -> None:
    scores = [
        EpisodeScore("a", "holds", net_bp_before=3.0, net_bp_after=2.0,
                     equity_left_running_pct=5.0, equity_kill_switch_pct=5.0),
        EpisodeScore("b", "breaks", net_bp_before=4.0, net_bp_after=-9.0,
                     equity_left_running_pct=-20.0, equity_kill_switch_pct=2.0,
                     stop_hours_after_break=12.0),
        EpisodeScore("c", "never paid", net_bp_before=-1.0, net_bp_after=-2.0,
                     equity_left_running_pct=-8.0, equity_kill_switch_pct=-3.0,
                     stop_hours_after_break=-30.0),
        EpisodeScore("d", "n/a", "needs the book's sizes"),
    ]  # fmt: skip
    s = summary(scores)
    assert (s["episodes"], s["scored"], s["holds"], s["breaks"], s["never_paid"]) == (4, 3, 1, 1, 1)
    assert s["kill_switch_helped"] == 2 and s["stops_before_break"] == 1
    assert s["kill_switch_saved_pct_mean"] == round((0 + 22 + 5) / 3, 2)
    text = page({"reversion": scores})
    assert "| reversion | 3 of 4 | 1 | 1 | 1 | 2 |" in text and "needs the book's sizes" in text


def test_an_account_that_has_lost_everything_stays_at_minus_one_hundred() -> None:
    assert ruined(pd.Series([-50.0, -101.0, -40.0])) == -100.0
    assert ruined(pd.Series([-50.0, -99.0, -40.0])) == -40.0


def test_the_benchmark_page_quotes_the_recorded_suite() -> None:
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    recorded = json.loads((repo / "experiments/results/bench_regimes/regimes.json").read_text())
    text = (repo / "docs/benchmarks.md").read_text(encoding="utf-8")
    for row in recorded["rows"]:
        s = row["summary"]
        line = (f"| `{row['name']}` | {s['holds']} | {s['breaks']} | {s['never_paid']} | "
                f"{s['kill_switch_helped']} of {s['episodes']} |")  # fmt: skip
        assert line in text, line
