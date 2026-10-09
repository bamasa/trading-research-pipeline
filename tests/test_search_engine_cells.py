"""The per-cell driver: thinned tape entries, looked up in walk tables, equal the executor's."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _engine_books import CLIP, DAY, make_book
from trading_research.backtest.taker import TakerBook
from trading_research.market_making.events import NS_PER_S
from trading_research.search_engine.catalogue import Catalogue, parse_key
from trading_research.search_engine.cells import (
    cell_sides,
    ledger_frame,
    ledger_row,
    score_cell,
    score_configuration,
    tape_events,
    trade_rows,
)
from trading_research.search_engine.ledger import TrialLedger
from trading_research.search_engine.tapes import (
    CandidateTape,
    TapeError,
    ev_gate_tape,
    hold_first_tape,
)
from trading_research.search_engine.walk_tables import (
    TABLE_COUNTERS,
    HoldTable,
    WalkTable,
    entry_parity,
    execute_rows,
    hold_table,
)
from trading_research.strategies.ev_gate import gate_events
from trading_research.strategies.hold_first import HoldFirstRule, fit_hold_first
from trading_research.validation.placebo_search import HourCells

C_CONFIG = "BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0"
R_CONFIG = "BTCUSDT|R|bound_ridge|micro|hold_s=5"
RATES = (25, 50, 100, 240)


def c_key(rate: int = 240, gate: str = "open", regime: str = "none") -> str:
    return (
        f"BTCUSDT|C|gbm|micro|{gate}|hold_s=5|trades_per_day={rate}|label_fee_bp=0|regime={regime}"
    )


def r_key(k: str = "0", beta: str = "0", gate: str = "open", regime: str = "none") -> str:
    return f"BTCUSDT|R|bound_ridge|micro|{gate}|hold_s=5|k={k}|beta_bp={beta}|regime={regime}"


def _setting(seed: int = 0) -> tuple[TakerBook, HoldTable, dict[str, np.ndarray], np.ndarray]:
    book = make_book(seed=seed, thin_share=0.05)
    table = hold_table(WalkTable.from_book(book, CLIP), 5 * NS_PER_S)
    rng = np.random.default_rng(seed)
    gates = {
        "quiet_out": rng.random(len(book.ts)) < 0.6,
        "tight_only": rng.random(len(book.ts)) < 0.8,
    }
    flags = np.array([book.ts[0] + 120 * NS_PER_S])
    return book, table, gates, flags


def _rules(seed: int = 0) -> dict[float, HoldFirstRule]:
    validation = np.random.default_rng(seed + 100).dirichlet([1.0, 3.0, 1.0], 20_000)
    return {float(n): fit_hold_first(validation, days=1.0, trades_per_day=n * 20) for n in RATES}


def _c_tape(
    seed: int = 0,
) -> tuple[TakerBook, HoldTable, CandidateTape, dict[float, HoldFirstRule]]:
    book, table, gates, flags = _setting(seed)
    rules = _rules(seed)
    proba = np.random.default_rng(seed + 1).dirichlet([1.0, 3.0, 1.0], len(book.ts))
    tape = hold_first_tape(DAY, C_CONFIG, book.ts, proba, rules[240.0], gates, flags)
    return book, table, tape, rules


def _r_tape(seed: int = 0) -> tuple[TakerBook, HoldTable, CandidateTape]:
    book, table, gates, flags = _setting(seed)
    rng = np.random.default_rng(seed + 2)
    mu = pd.Series(rng.normal(0, 60, len(book.ts))).rolling(8, min_periods=1).mean().to_numpy()
    sigma, walk = rng.uniform(0.2, 4.0, len(book.ts)), rng.uniform(0.0, 1.0, len(book.ts))
    return book, table, ev_gate_tape(DAY, R_CONFIG, book.ts, mu, sigma, walk, gates, flags)


@pytest.mark.parametrize(
    ("rate", "gate", "regime"),
    [(240, "open", "none"), (50, "quiet_out", "pull_4h"), (100, "tight_only", "none")],
)
def test_a_hold_first_cell_is_the_executor_over_its_candidate_rows(
    rate: int, gate: str, regime: str
) -> None:
    book, table, tape, rules = _c_tape()
    cell = parse_key(c_key(rate, gate, regime))
    trades = score_cell(tape, table, cell, rules[float(rate)])
    side, events = cell_sides(tape, cell, rules[float(rate)])
    assert events is None
    rows = tape.rows[side != 0]
    reference = execute_rows(table, rows, side[side != 0])
    np.testing.assert_array_equal(trades.rows, reference.rows)
    assert {k: trades.counters[k] for k in TABLE_COUNTERS} == reference.counters
    assert trades.counters["entries_same_event"] == 0 and trades.trades > 0
    assert entry_parity(book, trades.rows, trades.side, table).ok


def _reference_events(
    tape: CandidateTape, table: HoldTable, side: np.ndarray, events: np.ndarray
) -> list[int]:
    accepted, entered, free_ns = [], set(), -1
    for j in np.flatnonzero(side):
        row = int(tape.rows[j])
        s = 0 if side[j] > 0 else 1
        if not table.allowed[row] or table.walk.ts[row] < free_ns or events[j] in entered:
            continue
        if table.entry_row[row] >= table.rows or table.skip[s, row]:
            continue
        accepted.append(row)
        entered.add(events[j])
        free_ns = int(table.free_ns[row])
    return accepted


@pytest.mark.parametrize(
    ("k", "beta", "gate", "regime"),
    [
        ("0", "0", "open", "none"),
        ("0.5", "1", "quiet_out", "pull_24h"),
        ("1", "2", "tight_only", "none"),
    ],
)
def test_an_ev_gate_cell_enters_once_per_event(k: str, beta: str, gate: str, regime: str) -> None:
    book, table, tape = _r_tape()
    cell = parse_key(r_key(k, beta, gate, regime))
    trades = score_cell(tape, table, cell)
    side, events = cell_sides(tape, cell)
    assert events is not None
    np.testing.assert_array_equal(trades.rows, _reference_events(tape, table, side, events))
    assert len(set(events[np.searchsorted(tape.rows, trades.rows)])) == trades.trades
    assert entry_parity(book, trades.rows, trades.side, table).ok
    assert sum(v for n, v in trades.counters.items() if n not in ("decisions", "entries")) == (
        trades.counters["decisions"] - trades.trades
    )


def test_tape_events_are_the_ev_gates_events_on_the_full_day() -> None:
    _, _, tape = _r_tape(3)
    full = np.zeros(int(tape.rows[-1]) + 2, dtype=np.int64)
    full[tape.rows] = tape.side
    np.testing.assert_array_equal(tape_events(tape.rows, tape.side), gate_events(full)[tape.rows])


def test_a_configuration_scores_all_its_cells_and_a_silent_cell_writes_zeros() -> None:
    _, table, tape, rules = _c_tape(1)
    keys = [
        c_key(n, g, r)
        for n, g, r in itertools.product(
            RATES, ("open", "quiet_out", "tight_only"), ("none", "pull_4h", "pull_24h", "pull_120h")
        )
    ]
    scored = score_configuration(tape, table, [parse_key(k) for k in keys], rules)
    assert len(scored) == 48
    loose, strict = scored[c_key(240)], scored[c_key(25)]
    assert loose.counters["decisions"] >= strict.counters["decisions"]
    silent = score_cell(tape, table, parse_key(c_key(240)), HoldFirstRule(2.0, 2.0))
    row = ledger_row(1, c_key(240), DAY, silent)
    assert row == {"run": 1, "key": c_key(240), "day": DAY, "net_bp": 0.0, "G_bp": 0.0,
                   "pnl_long_bp": 0.0, "pnl_short_bp": 0.0, "trades": 0, "skips": 0}  # fmt: skip


def test_trade_rows_feed_null_a_and_agree_with_the_ledger() -> None:
    _, table, tape, rules = _c_tape(2)
    trades = score_cell(tape, table, parse_key(c_key()), rules[240.0])
    frame = pd.DataFrame(trade_rows(c_key(), table, trades))
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    kept = HourCells.from_trades(frame, [c_key()], [pd.Timestamp(DAY)]).kept()
    values = ledger_row(1, c_key(), DAY, trades)
    assert kept.trades[0, 0] == values["trades"]
    assert kept.net_bp[0, 0] == pytest.approx(values["net_bp"], rel=1e-9)
    assert kept.g_bp[0, 0] == pytest.approx(values["G_bp"], rel=1e-9)


def test_mismatches_are_refused() -> None:
    _, table, tape, rules = _c_tape()
    with pytest.raises(TapeError, match="looser"):
        score_cell(tape, table, parse_key(c_key()), HoldFirstRule(0.0, 0.0))
    with pytest.raises(TapeError, match="not scored from"):
        score_cell(tape, table, parse_key(c_key().replace("gbm", "logistic")), rules[240.0])
    with pytest.raises(ValueError, match="holds"):
        score_cell(tape, table, parse_key(c_key().replace("hold_s=5", "hold_s=20")), rules[240.0])
    with pytest.raises(ValueError, match="thresholds"):
        score_cell(tape, table, parse_key(c_key()))


def test_a_day_of_scored_cells_enters_the_ledger(tmp_path: Path) -> None:
    catalogue = Catalogue.load(
        Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"
    )
    _, table, tape, rules = _c_tape(4)
    keys = [c_key(n, g) for n in RATES for g in ("open", "quiet_out")]
    scored = score_configuration(tape, table, [parse_key(k) for k in keys], rules)
    frame = ledger_frame(1, DAY, scored)
    record = TrialLedger(tmp_path / "trials.parquet", catalogue).append(1, "abc", frame)
    assert record["rows"] == len(keys) and record["n_trials"] == len(keys)
    assert frame["trades"].tolist() == [scored[k].trades for k in keys]
