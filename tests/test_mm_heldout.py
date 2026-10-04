"""The held-out runner's own pieces, on synthetic markets.

The tapes only the loaded day can give (H1's lagged spread, the ladder's
oracle forecast), the placement of each fill against the touch it arrived on,
and the job that runs specs on one day and keeps their rows.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from trading_research.market_making import analysis
from trading_research.market_making.events import NS_PER_S
from trading_research.market_making.heldout import (
    INSIDE,
    TAKER,
    TOUCH,
    Job,
    Spec,
    _simulate,
    forecast_tape,
    kept,
    last_rows,
    placement,
    run_job,
    stale_spread_tape,
)
from trading_research.market_making.quoters import (
    ForecastTouchQuoter,
    InsideQuoter,
    Quoter,
    TouchQuoter,
)
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market

CONFIG = SimConfig(clip_notional=1e6, warmup_s=0.0, clip_touch_share=0.5)


def same(quoter: Quoter) -> Quoter:
    return quoter


def test_the_lagged_spread_is_read_a_lag_after_its_snapshot() -> None:
    market = random_market(1, n_snapshots=900)
    tape = stale_spread_tape(market, lag_s=30.0)
    ts, rows = last_rows(market)
    assert np.all(tape.ts == ts + 30 * NS_PER_S)
    spread = market.ask_px[rows, 0] - market.bid_px[rows, 0]
    assert np.array_equal(tape.values, spread)
    # At a snapshot, the value read is that of the last snapshot strictly
    # before 30 s earlier.
    at = int(market.book_ts[600])
    expected = rows[np.searchsorted(ts, at - 30 * NS_PER_S, side="left") - 1]
    read = tape.strictly_before(np.array([at]))[0]
    assert read == market.ask_px[expected, 0] - market.bid_px[expected, 0]


def test_the_oracle_forecast_has_the_registered_r_squared() -> None:
    market = random_market(4, n_snapshots=20_000, max_step=2)
    ts, rows = last_rows(market)
    mid = analysis.book_mid(market)
    change = analysis.mid_at(market.book_ts, mid, ts + NS_PER_S) - mid[rows]
    perfect = forecast_tape(market, 1.0, seed=0)
    assert np.allclose(perfect.values, change) and np.all(perfect.ts == ts - 1)
    for r2 in (0.1, 0.3):
        values = forecast_tape(market, r2, seed=[0, 1, 2]).values
        residual = np.mean((change - values) ** 2) / np.mean(change**2)
        assert 1.0 - residual == pytest.approx(r2, abs=0.03)
        assert np.array_equal(values, forecast_tape(market, r2, seed=[0, 1, 2]).values)
    with pytest.raises(ValueError):
        forecast_tape(market, 0.0, seed=0)


def test_fills_are_placed_against_the_touch_their_order_arrived_on() -> None:
    market = random_market(7, n_snapshots=3000, max_step=1)
    touch = simulate_day(market, TouchQuoter(), CONFIG)
    where = placement(touch.fills, touch.orders, market)
    assert set(where) <= {TOUCH, TAKER, "behind"}
    assert (where == TAKER).sum() == (~touch.fills["maker"]).sum()
    inside = InsideQuoter(0.0, 0.0, 0.0, 2, sigma_ref=5.0, maker_bp=0.0)
    result = simulate_day(market, inside, CONFIG)
    placed = placement(result.fills, result.orders, market)
    assert (placed == INSIDE).any()
    twin = simulate_day(market, InsideQuoter(0.0, 0.0, 0.0, 2, 5.0, 0.0, inside=False), CONFIG)
    assert not (placement(twin.fills, twin.orders, market) == INSIDE).any()


def job_for(tmp_path: Path, specs: tuple[Spec, ...]) -> Job:
    return Job(
        key="HANDUSDT_test",
        symbol="HANDUSDT",
        day=SYNTHETIC_DAY,
        specs=specs,
        book_roots=(tmp_path / "none",),
        trades_root=tmp_path / "none",
        funding_root=None,
        out=tmp_path,
    )


def test_a_spec_is_simulated_with_its_day_tapes_and_kept(tmp_path: Path) -> None:
    market = random_market(9, n_snapshots=1500)
    oracle = Spec(
        "rung5",
        partial(same, ForecastTouchQuoter(maker_bp=0.0)),
        CONFIG,
        group="ladder",
        built=(("forecast", 1.0),),
        oracle=True,
        keep=True,
    )
    job = job_for(tmp_path, (oracle,))
    row = _simulate(oracle, job, market)
    assert row["oracle"] and row["group"] == "ladder" and row["label"] == "rung5"
    assert row["fills_inside"] == 0 and row["orders_live"] >= row["orders_with_fill"]
    fills = kept(tmp_path, "rung5", "HANDUSDT", "fills")
    assert len(fills) == row["fills"] and "placement" in fills.columns
    # Perfect foresight at no fee does not lose on its passive fills at 1 s.
    maker = fills[fills["placement"] != TAKER]
    assert (maker["markout_1s"] >= -1e-9).mean() > 0.9


def test_a_finished_job_is_read_back_without_loading_the_day(tmp_path: Path) -> None:
    job = job_for(tmp_path, ())
    target = tmp_path / "rows" / f"{job.key}.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps([{"label": "done", "net": 1.5}]), encoding="utf-8")
    assert run_job(job) == [{"label": "done", "net": 1.5}]


def loaded(market: object, *args: object, **kwargs: object) -> object:
    return market


def test_a_failing_spec_is_recorded_and_excluded_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        name = "broken"
        uses_future = False

        def quotes(self, view: object, position: float) -> object:
            raise RuntimeError("no quote")

    market = random_market(2, n_snapshots=300)
    spec = Spec("broken", partial(same, Broken()), CONFIG)
    job = job_for(tmp_path, (spec,))
    from trading_research.market_making import heldout

    monkeypatch.setattr(heldout, "load_day", partial(loaded, market))
    (row,) = run_job(job)
    assert row["status"].startswith("error") and row["excluded"]
    assert row["label"] == "broken"
