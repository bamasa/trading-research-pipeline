"""The print executor: the side-correct rule on every fill, the one-position rule, the money."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_spot_prints_fixture import CLIP, DAY, grid_of, make_prints
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.spot.costs import FeeTable, load_fee_table
from trading_research.spot.fills_standin import RULES, DayPrints
from trading_research.spot.print_taker import (
    PRINT_COUNTERS,
    PrintSettings,
    execute_print_day,
    print_settings,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "spot_2026.yaml"
TABLE = load_fee_table(CONFIG)
BINANCE = "binance-spot:BTCFDUSD"
START = day_start_ns(DAY)
FLATTEN = START + (23 * 3600 + 59 * 60) * NS_PER_S
HOLD = 50 * NS_PER_S


def at(seconds: float) -> int:
    return START + round(seconds * NS_PER_S)


def hand_prints(rows: list[tuple[float, int, float, float]]) -> DayPrints:
    """Prints from ``(seconds after midnight, side, price, size)``, all reachable."""
    ts = np.array([at(r[0]) for r in rows], dtype=np.int64)
    return DayPrints(DAY, ts, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                     np.array([r[1] for r in rows], dtype=np.int8), np.ones(len(rows), bool))  # fmt: skip


def random_run(seed: int, count: int = 3000, **kw: object) -> tuple[DayPrints, np.ndarray, object]:
    prints = make_prints(seed)
    rows = grid_of(prints)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(rows), count, replace=False))
    side = rng.choice(np.array([-1, 1]), count)
    settings = print_settings(TABLE, BINANCE, **kw)  # type: ignore[arg-type]
    return prints, rows, execute_print_day(prints, rows, rows[pick], side, HOLD, settings)


def first_print(prints: DayPrints, side: int, ns: int, until: int) -> int:
    ok = (prints.side == side) & prints.reachable & (prints.ts >= ns) & (prints.ts < until)
    return int(np.flatnonzero(ok)[0]) if ok.any() else -1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_fill_is_the_rules_and_every_exit_the_labels(seed: int) -> None:
    prints, rows, day = random_run(seed)
    trades = day.trades
    assert len(trades) > 30
    for trade in trades.itertuples():
        s = int(trade.side)
        assert trade.entry_arrival_ns == trade.decision_ns + 10_000_000
        assert trade.entry_position == first_print(prints, s, trade.entry_arrival_ns, FLATTEN)
        due = trade.entry_print_ns + HOLD
        h = np.searchsorted(rows, due)
        decided = min(int(rows[h]) if h < len(rows) else due, FLATTEN)
        assert trade.exit_decision_ns == decided
        assert trade.exit_reason == ("day_end" if decided == FLATTEN else "hold")
        if not trade.exit_no_print:
            want = first_print(prints, -s, trade.exit_arrival_ns, START + 86_400 * NS_PER_S)
            assert trade.exit_position == want and trade.exit_fill_ns == trade.exit_print_ns
        assert trade.entry_fee_bp == trade.exit_fee_bp == 10.0
    assert np.all(trades["decision_ns"].to_numpy()[1:] >= trades["exit_fill_ns"].to_numpy()[:-1])
    c = day.counters
    assert set(c) == set(PRINT_COUNTERS)
    assert c["decisions"] == sum(c[k] for k in ("entries", "entries_outside_hours",
                                                "entries_busy", "entries_no_print"))  # fmt: skip
    assert c["entries"] == len(trades) and c["entries_busy"] > 0
    assert c["entries_outside_hours"] > 0 and c["entry_late_print"] > 0
    assert c["fill_beyond_sweep"] > 0


def test_the_money_is_the_accounts() -> None:
    _, _, day = random_run(3)
    t = day.trades
    entry, exit_ = t["entry_price"] * CLIP, t["exit_price"] * CLIP
    np.testing.assert_allclose(t["fees"], (entry + exit_) * 10.0 * 1e-4, rtol=1e-12)
    np.testing.assert_allclose(t["gross"], t["side"] * (exit_ - entry), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(t["net"], t["gross"] - t["fees"], rtol=1e-12, atol=1e-12)
    assert day.net == pytest.approx(t["net"].sum(), rel=1e-12)
    assert day.turnover == pytest.approx(float((entry + exit_).sum()), rel=1e-12)
    assert day.day_bp() == pytest.approx(t["net"].sum() / t["notional"].sum() * 1e4)
    assert (t["hold_realised_ns"] == t["exit_fill_ns"] - t["entry_print_ns"]).all()
    assert (t["entry_wait_ns"] == t["entry_print_ns"] - t["entry_arrival_ns"]).all()


def run(prints: DayPrints, decisions: list[float], sides: list[int], **kw: object) -> object:
    rows = np.unique(prints.ts)
    stamps = np.array([at(s) for s in decisions], dtype=np.int64)
    settings = print_settings(TABLE, BINANCE, **kw)  # type: ignore[arg-type]
    return execute_print_day(prints, rows, stamps, np.array(sides), HOLD, settings)


def test_a_busy_period_depends_on_the_side() -> None:
    # sellers only from 12:00 to 12:04, then both sides again
    prints = hand_prints([(43200.0 + k, -1, 60_000.0 - k, 0.02) for k in range(240)]
                         + [(43440.5, 1, 59_800.0, 0.02), (43441.0, -1, 59_799.0, 0.02),
                            (43500.0, 1, 59_801.0, 0.02), (43501.0, -1, 59_800.0, 0.02),
                            (43600.0, 1, 59_802.0, 0.02), (43700.0, -1, 59_803.0, 0.02)])  # fmt: skip
    decisions = [43201.0, 43260.0, 43441.0]
    day = run(prints, decisions, [-1, -1, 1])
    t, c = day.trades, day.counters  # type: ignore[attr-defined]
    assert len(t) == 2 and c["entries_busy"] == 1
    short, long = t.iloc[0], t.iloc[1]
    assert short.entry_print_ns == at(43202.0) and short.exit_decision_ns == at(43252.0)
    assert short.exit_print_ns == at(43440.5) and bool(short.exit_late)
    assert long.decision_ns == at(43441.0) and long.entry_print_ns == at(43500.0)
    assert long.exit_decision_ns == at(43600.0) and long.exit_print_ns == at(43700.0)
    flipped = run(prints, decisions, [1, 1, -1])
    t, c = flipped.trades, flipped.counters  # type: ignore[attr-defined]
    assert len(t) == 1 and c["entries_busy"] == 2
    assert t.iloc[0].entry_print_ns == at(43440.5) and t.iloc[0].exit_print_ns == at(43501.0)
    assert bool(t.iloc[0].entry_late) and c["entry_late_print"] == 1


def test_entries_and_exits_without_a_later_print() -> None:
    prints = hand_prints(
        [(600.0, 1, 100.0, 1.0), (700.0, -1, 99.0, 1.0), (86_345.0, -1, 98.0, 1.0)]
    )
    none = run(prints, [750.0], [-1])  # the only later seller prints after 23:59
    assert none.counters["entries_no_print"] == 1 and len(none.trades) == 0  # type: ignore[attr-defined]
    day = run(prints, [650.0], [-1])
    c, trade = day.counters, day.trades.iloc[0]  # type: ignore[attr-defined]
    assert c["exits_no_print"] == 1 and c["exits_no_print_either_side"] == 0
    assert trade.entry_print_ns == at(700.0) and trade.exit_reason == "day_end"
    assert trade.exit_decision_ns == FLATTEN and bool(trade.exit_no_print)
    assert trade.exit_price == 100.0 and trade.exit_print_ns == at(600.0)
    assert trade.exit_fill_ns == trade.exit_arrival_ns == FLATTEN + 10_000_000
    assert trade.exit_wait_ns == at(600.0) - trade.exit_arrival_ns
    assert not bool(trade.exit_late)


def test_an_exit_whose_side_never_prints_is_priced_at_the_other_side() -> None:
    prints = hand_prints([(600.0, 1, 100.0, 1.0), (700.0, 1, 103.0, 1.0)])
    day = run(prints, [500.0], [1])
    t, c = day.trades, day.counters  # type: ignore[attr-defined]
    assert c["exits_no_print"] == 1 and c["exits_no_print_either_side"] == 1
    trade = t.iloc[0]
    assert trade.exit_price == 103.0 and trade.exit_print_ns == at(700.0)
    assert trade.exit_fill_ns == trade.exit_arrival_ns
    assert trade.exit_wait_ns == at(700.0) - trade.exit_arrival_ns < 0


def test_the_brackets_keep_the_rules_fills_and_change_only_prices() -> None:
    runs = {rule: random_run(4, rule=rule)[2] for rule in RULES}
    base = runs["rule"].trades  # type: ignore[attr-defined]
    timing = ["decision_ns", "side", "entry_print_ns", "exit_decision_ns", "exit_fill_ns"]
    for rule, day in runs.items():
        t = day.trades  # type: ignore[attr-defined]
        assert t[timing].equals(base[timing]), rule
        assert day.counters == runs["rule"].counters  # type: ignore[attr-defined]
    found = ~base["exit_no_print"].to_numpy()
    buy_entry = base["side"].to_numpy() > 0
    opt, worst = runs["optimistic"].trades, runs["worse_of_two"].trades  # type: ignore[attr-defined]
    pess = runs["pessimistic_depth"].trades  # type: ignore[attr-defined]
    beyond = base["entry_beyond"].to_numpy() > 0
    assert not np.array_equal(opt["entry_price"], base["entry_price"])
    # worse of two is never better for the account than the pessimistic depth price
    sign = np.where(buy_entry, 1.0, -1.0)
    assert np.all(sign * (worst["entry_price"] - pess["entry_price"]) >= 0)
    assert np.all((-sign * (worst["exit_price"] - pess["exit_price"]))[found] >= 0)
    assert np.array_equal(pess["entry_price"][~beyond], base["entry_price"][~beyond])


def test_k_latency_moves_every_arrival_by_100_ms() -> None:
    _, _, day = random_run(5, latency_ns=110_000_000)
    t = day.trades  # type: ignore[attr-defined]
    assert (t["entry_arrival_ns"] - t["decision_ns"] == 110_000_000).all()
    assert (t["exit_arrival_ns"] - t["exit_decision_ns"] == 110_000_000).all()


def test_each_fill_pays_the_fee_of_its_instant() -> None:
    rows = [{"start": "2026-06-01T00:00:00Z", "end": "2026-06-03T12:00:00Z", "sources": ["B5"],
             "tiers": {"regular": [0.0, 10.0]}},
            {"start": "2026-06-03T12:00:00Z", "end": "2026-06-05T00:00:00Z", "sources": ["B5"],
             "tiers": {"regular": [0.0, 7.5]}}]  # fmt: skip
    table = FeeTable.from_config({"verdict_tier": {BINANCE: "regular"},
                                  "intervals": {BINANCE: rows}})  # fmt: skip
    prints = hand_prints([(43_190.0, 1, 100.0, 1.0), (43_195.0, -1, 99.0, 1.0),
                          (43_230.0, -1, 99.5, 1.0), (43_260.0, 1, 100.5, 1.0)])  # fmt: skip
    settings = print_settings(table, BINANCE)
    day = execute_print_day(prints, np.unique(prints.ts), np.array([at(43_190.0)]),
                            np.array([-1]), 20 * NS_PER_S, settings)  # fmt: skip
    trade = day.trades.iloc[0]
    assert (trade.entry_fee_bp, trade.exit_fee_bp) == (10.0, 7.5)
    assert trade.fees == pytest.approx((99.0 * 10.0 + 100.5 * 7.5) * CLIP * 1e-4, rel=1e-12)


def test_settings_come_from_the_table() -> None:
    with pytest.raises(ValueError, match="no tier"):
        PrintSettings(table=TABLE, instrument=BINANCE, tier="non_vip")
    with pytest.raises(ValueError, match="unknown fill rule"):
        print_settings(TABLE, BINANCE, rule="mid")
    settings = print_settings(TABLE, BINANCE, "vip9")
    assert settings.instrument == "binance_spot_BTCFDUSD"
    assert settings.fee_bp(np.array([at(43_200.0)]))[0] == 2.3
    with pytest.raises(ValueError):
        execute_print_day(make_prints(), np.array([2, 1]), np.array([1]), np.array([1]), HOLD,
                          settings)  # fmt: skip
