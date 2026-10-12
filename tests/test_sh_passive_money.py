"""Stage C's money for the signal-entry quoter: per attempt, over its own entry's notional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import FeeTier
from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market
from trading_research.pipeline.short_horizon.passive import (
    SimCell,
    _maker_values,
    day_values,
    per_100_clip,
)

MS = 1_000_000
CONFIG = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)
CELL = SimCell("S-P", "key", "entry", 0.5)


def _market(bids: list[int], prints: list[tuple[float, int, float, int]]):
    """Snapshots every 100 ms with the ask two ticks above the bid."""
    snapshots = [
        (100.0 * i, [(b - k, 10.0) for k in range(3)], [(b + 2 + k, 10.0) for k in range(3)])
        for i, b in enumerate(bids)
    ]
    return hand_built_market(snapshots, prints, lot=1.0)


def _run(market, decisions_ms: list[float]):
    stamps = np.array([market.day_start_ns + round(t * MS) for t in decisions_ms])
    quoter = SignalEntryQuoter(0.5)
    tapes = entry_tapes(stamps, np.ones(len(stamps)))
    return quoter, simulate_day(market, quoter, CONFIG, tapes)


def test_an_exit_at_the_next_post_stays_in_its_attempt_and_the_day_reads_per_attempt() -> None:
    """Bought at 1.00 at 150 ms, out at 1.03 at 800 ms where the second attempt
    posts at 1.03 and never fills: the first attempt carries the whole day."""
    market = _market([100] * 8 + [103] * 12, [(150.0, 99, 5.0, -1)])
    quoter, result = _run(market, [0, 750])
    row = day_values(CELL, quoter, result, clip=1.0)
    fees = float(result.fills["fee"].sum())
    assert row["attempts"] == 2 and row["trades"] == 1
    assert row["net_per_100"] == pytest.approx((0.03 - fees) * 100.0 / 1.00)
    assert row["gross_bp_trade"] == pytest.approx(300.0)


def test_each_attempt_is_taken_over_its_own_entry_notional() -> None:
    """Two round trips at different entry prices: each net over its own notional."""
    bids = [100] * 8 + [103] * 4 + [110] * 8
    prints = [(150.0, 99, 5.0, -1), (1_250.0, 109, 5.0, -1)]
    market = _market(bids, prints)
    quoter, result = _run(market, [0, 1_150])
    table = quoter.attempts(result.fills)
    assert table["filled"].tolist() == [True, True]
    assert table["notional"].tolist() == pytest.approx([1.00, 1.10])
    row = day_values(CELL, quoter, result, clip=1.0)
    expected = (table["net"] / table["notional"] * 100.0).sum()
    assert row["net_per_100"] == pytest.approx(expected)
    assert row["net_per_100"] != pytest.approx(row["net_usdt"] * 100.0 / 1.05)


def test_funding_is_taken_over_the_mean_entry_notional_and_no_fill_is_zero() -> None:
    filled = pd.DataFrame({"net": [0.02, -0.01], "notional": [1.0, 3.0]})
    assert per_100_clip(filled, 0.0) == pytest.approx(2.0 - 1.0 / 3.0)
    assert per_100_clip(filled, -0.004) == pytest.approx(2.0 - 1.0 / 3.0 - 0.2)
    assert per_100_clip(filled.iloc[:0], 0.5) == 0.0


def test_the_making_part_is_over_the_passive_fills_only_and_its_mean_is_per_fill() -> None:
    """Bought at 1.00, mid 1.01 then 1.045: the entry's 5 s markout is +450 bp. The
    taker exit is not a passive fill: the decomposition's spread + adverse counts
    it, the making part does not."""
    market = _market([100] * 8 + [103] * 12, [(150.0, 99, 5.0, -1)])
    quoter, result = _run(market, [0])
    row = day_values(CELL, quoter, result, clip=1.0)
    maker = result.fills[result.fills["maker"]]
    expected = float((maker["markout_5s"] * maker["price"] * maker["size"]).sum() * 1e-4)
    assert row["making_usdt"] == pytest.approx(expected)
    everything = result.decomposition["spread"] + result.decomposition["adverse"]
    assert row["making_usdt"] != pytest.approx(everything)
    assert row["making_fills"] == 1 and row["making_bp"] == pytest.approx(
        maker["markout_5s"].iloc[0]
    )
    assert row["taker_notional"] == pytest.approx(1.03) and row[
        "passive_notional"
    ] == pytest.approx(1.0)


def test_gross_and_net_per_trade_are_over_the_filled_entry_notional() -> None:
    """Half the clip fills: the day's gross per trade is over the 0.5 filled, not
    the posted clip."""
    market = hand_built_market(
        [(100.0 * i, [(b - k, 10.0) for k in range(3)], [(b + 2 + k, 10.0) for k in range(3)])
         for i, b in enumerate([100] * 8 + [103] * 12)],
        [(150.0, 99, 0.5, -1)], lot=0.1,
    )  # fmt: skip
    quoter, result = _run(market, [0])
    row = day_values(CELL, quoter, result, clip=1.0)
    assert row["entry_notional"] == pytest.approx(0.5)
    assert row["gross_bp_trade"] == pytest.approx(row["gross_usdt"] / 0.5 * 1e4)
    assert row["net_bp_trade"] == pytest.approx(row["net_usdt"] / 0.5 * 1e4)


def test_net_per_100_moves_to_another_fee_pair_exactly_by_its_weights() -> None:
    """The quoter never reads a fee: a rerun at maker 0 / taker 1.5 is the base run's
    net_per_100 less the fee differences on each fill's notional."""
    bids = [100] * 8 + [103] * 4 + [110] * 8
    market = _market(bids, [(150.0, 99, 5.0, -1), (1_250.0, 109, 5.0, -1)])
    quoter, result = _run(market, [0, 1_150])
    base = day_values(CELL, quoter, result, clip=1.0)
    cheap = SignalEntryQuoter(0.5)
    stamps = np.array([market.day_start_ns + t * MS for t in (0, 1_150)])
    other = simulate_day(market, cheap, CONFIG.with_(fees=FeeTier("cheap", 0.0, 1.5)),
                         entry_tapes(stamps, np.ones(2)))  # fmt: skip
    moved = day_values(CELL, cheap, other, clip=1.0)
    shift = (0.0 - 2.0) * 1e-4 * base["per100_maker_weight"] + (1.5 - 5.5) * 1e-4 * base[
        "per100_taker_weight"
    ]
    assert moved["net_per_100"] == pytest.approx(base["net_per_100"] - shift)
    assert base["trades_long"] == 2 and base["net_long_usdt"] == pytest.approx(base["net_usdt"])


def test_the_market_makers_entries_are_flat_to_flat_over_the_clip_at_their_first_fill() -> None:
    """Two entries: bought 1 at 100 and sold at 101; sold 1 at 200 and bought at
    198. Each over the clip's notional at its first fill, not the day's mean mid."""
    fills = pd.DataFrame({"side": [1, -1, -1, 1], "price": [100.0, 101.0, 200.0, 198.0],
                          "size": [1.0] * 4, "fee": [0.0] * 4, "maker": [True, False, True, False],
                          "position_after": [1.0, 0.0, -1.0, 0.0]})  # fmt: skip
    found = _maker_values(fills, 1.0, 0.0)
    assert found["net_per_100"] == pytest.approx(1.0 / 100.0 * 100 + 2.0 / 200.0 * 100)
    assert found["trades"] == 2 and found["trades_long"] == 1 and found["trades_short"] == 1
    assert found["per100_maker_weight"] == pytest.approx(200.0)


def test_stage_c_leaves_a_day_any_run_flagged_out_of_every_passive_cell() -> None:
    """S-M's three leans on ten days, two a month; one lean's run on 03-05 flagged
    stale. Every cell loses 03-05, where the lean 1 cell lost 2 USDT; the making
    part is the mean over the passive fills of the days kept, bp per fill."""
    from datetime import date

    from trading_research.pipeline.short_horizon.passive import admit, flagged_days

    days = [date(2025, m, d) for m in range(2, 7) for d in (4, 5)]
    cells = [SimCell("S-M", "S25KEY", "maker", 50.0, lean=x) for x in (0.5, 1.0, 2.0)]
    rows = []
    for c in cells:
        for d in days:
            bad = d == date(2025, 3, 5)
            rows.append({"label": c.label, "day": d.isoformat(), "excluded": bad and c.lean == 2.0,
                         "net_per_100": -2.0 if bad else 1.0, "trades": 3,
                         "making_bp_sum": -90.0 if bad else 3.0, "making_fills": 1.0 if bad else 2.0})  # fmt: skip
    frame = pd.DataFrame(rows)
    record = {"signal_admitted": {"C": False, "R": False}, "hand_runs": []}
    found = admit(frame, cells, days, "S-M", record)
    assert flagged_days(frame) == {"2025-03-05"} and found["flagged_days"] == ["2025-03-05"]
    assert found["S"] == pytest.approx(1.0) and found["status_on_D"] == "admitted"
    assert found["making_bp_per_fill"] == pytest.approx(1.5)


def test_s_p_and_s_m_have_their_deflated_sharpe_ratio_from_stage_c() -> None:
    from datetime import date

    from trading_research.evaluation.overfitting import deflated_sharpe_ratio, sharpe_variance
    from trading_research.pipeline.short_horizon.passive import passive_dsr

    days = [date(2025, 2, d) for d in range(3, 13)]
    cells = [SimCell("S-P", "a", "entry", 5.0), SimCell("S-P", "b", "entry", 5.0),
             SimCell("S-M", "a", "maker", 5.0, lean=1.0)]  # fmt: skip
    rng = np.random.default_rng(1)
    rows = pd.DataFrame([{"label": c.label, "day": d.isoformat(), "excluded": False,
                          "net_per_100": float(rng.normal(0.2, 1.0))} for c in cells for d in days])  # fmt: skip
    admission = {"S-P": {"cell": cells[0].label}, "S-M": {"cell": cells[2].label}}
    found = passive_dsr(rows, cells, days, 30, admission)
    panel = rows.pivot_table(index="day", columns="label", values="net_per_100")
    expected = deflated_sharpe_ratio(panel[cells[0].label].to_numpy(), 30, sharpe_variance(panel))
    assert set(found) == {"S-P/net", "S-M/net"} and found["S-P/net"] == pytest.approx(expected.dsr)


def test_a_cell_day_records_every_day_sum_the_measurements_beside_the_verdict_read() -> None:
    """What ``beside_verdict`` reads of a passive slot is in ``BESIDE_COLUMNS``, and a
    cell-day of either quoter carries every one of them."""
    from trading_research.pipeline.short_horizon import beside_verdict as bv
    from trading_research.pipeline.short_horizon.passive import BESIDE_COLUMNS

    read = {"net_usdt", "passive_notional", "taker_notional", "entry_notional", "funding_usdt",
            "per100_maker_weight", "per100_taker_weight", "making_usdt", "making_bp_sum",
            "making_fills", "net_long_usdt", "net_short_usdt", "notional_long", "notional_short",
            "trades_long", "trades_short", *bv.PASSIVE_COUNTERS}  # fmt: skip
    assert read <= set(BESIDE_COLUMNS)
    market = _market([100] * 8 + [103] * 12, [(150.0, 99, 5.0, -1)])
    quoter, result = _run(market, [0])
    entry = day_values(CELL, quoter, result, clip=1.0)
    maker = day_values(SimCell("S-M", "k", "maker", 0.5, lean=1.0), quoter, result, clip=1.0)
    assert set(BESIDE_COLUMNS) <= set(entry) and set(BESIDE_COLUMNS) <= set(maker)
