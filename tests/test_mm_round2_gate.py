"""The second round's room gate, on synthetic markets.

The room at ``t`` must be what the registration defines, computed from prints
whose five-second horizon has closed by ``t`` and from nothing later; the hour
mask and the kinds must open exactly when registered; the shifted placebo must
keep the share of time open and move only the timing; and ``G(base)`` must be
its base while open and reduce-only while closed.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making import gate, synthetic
from trading_research.market_making.events import NS_PER_DAY, NS_PER_S, DayEvents, assemble
from trading_research.market_making.quoters import MarketView, SkewQuoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig, simulate_day

HOUR = 3600 * NS_PER_S


def market(seed: int = 3, *, day: date = synthetic.SYNTHETIC_DAY, n: int = 4000) -> DayEvents:
    """About an hour of one-second snapshots and their prints."""
    return synthetic.random_market(seed, n_snapshots=n, snapshot_ms=1000, day=day)


def brute_room(prints: gate.PrintMarkouts, window_min: float, t: int) -> float:
    """The registered definition, print by print."""
    window = round(window_min * 60 * NS_PER_S)
    if t < prints.day_start_ns + window:
        return np.nan
    inside = (prints.ts + 5 * NS_PER_S <= t) & (prints.ts >= t - window)
    if inside.sum() < gate.MIN_PRINTS:
        return np.nan
    return float(np.average(prints.markout_bp[inside], weights=prints.size[inside]))


@pytest.mark.parametrize("window_min", [15, 60])
def test_the_room_is_the_registered_definition(window_min: int) -> None:
    prints = gate.print_markouts(market(n=5000))
    rng = np.random.default_rng(0)
    times = prints.day_start_ns + rng.integers(0, 5100 * NS_PER_S, 400)
    # Times exactly on a print's horizon and on the edge of its window, too.
    times = np.r_[
        times, prints.ts[::50] + 5 * NS_PER_S, prints.ts[::50] + window_min * 60 * NS_PER_S
    ]
    times = np.r_[times, times - 1, times + 1]
    got = gate.room_at(prints, window_min, times)
    want = np.array([brute_room(prints, window_min, int(t)) for t in times])
    assert np.isfinite(want).sum() > 100
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    ok = np.isfinite(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9, atol=1e-9)


def rewritten_after(events: DayEvents, cut: int) -> DayEvents:
    """The same day up to ``cut``; after it, a different book and different prints."""
    later_book = events.book_ts > cut
    later_prints = events.trade_ts > cut
    bid_px = np.array(events.bid_px)
    ask_px = np.array(events.ask_px)
    bid_px[later_book] += 40
    ask_px[later_book] += 40
    trade_px = np.array(events.trade_px)
    trade_px[later_prints] -= 7
    trade_sz = np.array(events.trade_sz)
    trade_sz[later_prints] *= 3
    aggressor = np.array(events.trade_aggressor)
    aggressor[later_prints] *= -1
    return assemble(
        events.spec,
        events.day,
        book_ts=events.book_ts,
        bid_px=bid_px,
        bid_sz=events.bid_sz,
        ask_px=ask_px,
        ask_sz=events.ask_sz,
        trade_ts=events.trade_ts,
        trade_px=trade_px,
        trade_sz=trade_sz,
        trade_aggressor=aggressor,
    )


def test_the_room_at_t_reads_nothing_after_t() -> None:
    events = market(n=5000)
    cut = events.day_start_ns + 4200 * NS_PER_S
    prints = gate.print_markouts(events)
    changed = gate.print_markouts(rewritten_after(events, cut))
    times = events.day_start_ns + np.arange(0, 4201, 3) * NS_PER_S
    for window in (15, 60):
        before = gate.room_at(prints, window, times)
        after = gate.room_at(changed, window, times)
        np.testing.assert_array_equal(before, after)
        assert np.isfinite(before).sum() > 100
        # And the change shows once the rewritten prints' horizons close.
        later = cut + np.arange(10, 400, 10) * NS_PER_S
        assert not np.allclose(
            gate.room_at(prints, window, later), gate.room_at(changed, window, later)
        )


def test_the_room_changes_only_with_prints_whose_horizon_has_closed() -> None:
    prints = gate.print_markouts(market(n=5000))
    stamps, values = gate.room_steps(prints, 15)
    # Every step is a horizon closing, a print leaving, the window filling, or midnight.
    start = prints.day_start_ns
    allowed = np.r_[
        prints.ts + 5 * NS_PER_S - 1,
        prints.ts + 15 * 60 * NS_PER_S,
        start - 1,
        start + 900 * NS_PER_S - 1,
    ]
    assert np.isin(stamps, allowed).all()
    # A tape read strictly before a decision gives the room at the decision.
    tape = SignalTape("room", stamps, values)
    times = start + np.arange(900, 5000, 3) * NS_PER_S + 1
    np.testing.assert_array_equal(tape.strictly_before(times), gate.room_at(prints, 15, times))


# -- the hour mask and the kinds ---------------------------------------------------


def test_the_hour_mask_reads_the_fee_and_the_margin() -> None:
    hourly = pd.DataFrame(
        {"hour": np.arange(24), "volume": 1.0, "markout_bp": np.linspace(-1.0, 4.75, 24)}
    )
    assert gate.hour_mask(hourly, 2.0, 0.0) == tuple(range(12, 24))
    assert gate.hour_mask(hourly, 2.0, 2.0) == tuple(range(20, 24))
    # A cheaper fee leaves more room: the mask at maker 0 is the mask at 2 shifted.
    assert gate.hour_mask(hourly, 0.0, 2.0) == gate.hour_mask(hourly, 2.0, 0.0)
    days = [gate.print_markouts(market(seed)) for seed in (1, 2)]
    table = gate.hourly_markouts(days)
    assert table["volume"].sum() == pytest.approx(sum(d.size.sum() for d in days))


def test_each_kind_opens_when_registered() -> None:
    prints = gate.print_markouts(market(n=5000))
    start = prints.day_start_ns
    times = start + np.arange(0, 86_000, 13) * NS_PER_S
    hour = (times - start) // HOUR
    mask = (0, 1, 5)
    room = gate.room_at(prints, 15, times)
    for margin in (0.0, 1.0):
        trailing = gate.state_tape([prints], gate.GateCell("S1", "trailing", 15, margin), 0.0)
        hours = gate.state_tape([prints], gate.GateCell("S1", "hours", None, margin), 0.0, mask)
        both = gate.state_tape([prints], gate.GateCell("S1", "both", 15, margin), 0.0, mask)
        want_trailing = np.isfinite(room) & (room >= margin)
        want_hours = np.isin(hour, mask)
        np.testing.assert_array_equal(trailing.strictly_before(times) > 0.5, want_trailing)
        np.testing.assert_array_equal(hours.strictly_before(times) > 0.5, want_hours)
        np.testing.assert_array_equal(both.strictly_before(times) > 0.5, want_trailing & want_hours)
    # The fee paid moves the trailing gate: at maker 0.5 the room must clear 0.5.
    cheap = gate.state_tape([prints], gate.GateCell("S1", "trailing", 15, 0.0), 0.5)
    np.testing.assert_array_equal(
        cheap.strictly_before(times) > 0.5, np.isfinite(room) & (room >= 0.5)
    )
    # Closed for the first W minutes of the day, whatever the room.
    assert not (trailing.strictly_before(times[times < start + 900 * NS_PER_S]) > 0.5).any()


def test_no_state_crosses_midnight() -> None:
    first = market(1, n=86_000)
    second = market(2, day=first.day + timedelta(days=1), n=4000)
    days = [gate.print_markouts(first), gate.print_markouts(second)]
    cell = gate.GateCell("S1", "trailing", 15, 0.0)
    tape = gate.state_tape(days, cell, -50.0)  # a rebate so large the gate opens all day
    late = first.day_start_ns + NS_PER_DAY - 1
    assert tape.strictly_before(np.array([late]))[0] == 1.0
    early = second.day_start_ns + np.array([0, 1, 899 * NS_PER_S])
    assert (tape.strictly_before(early) == 0.0).all()


# -- the shifted-gate placebo ----------------------------------------------------------


def block_tape() -> tuple[SignalTape, tuple[int, int]]:
    days = [
        gate.print_markouts(
            market(seed, day=synthetic.SYNTHETIC_DAY + timedelta(days=seed), n=9000)
        )
        for seed in range(3)
    ]
    cell = gate.GateCell("S1", "both", 15, 0.0)
    tape = gate.state_tape(days, cell, 0.0, (0, 1, 2, 7, 8, 20))
    start = days[0].day_start_ns
    return tape, (start, start + 3 * NS_PER_DAY)


def test_the_placebo_keeps_the_share_open_and_moves_only_the_timing() -> None:
    tape, (start, end) = block_tape()
    length = end - start
    opened = gate.time_open(tape, start, end)
    assert 0 < opened < length
    for seed in range(5):
        offset = gate.placebo_offset(seed, length)
        assert 6 * HOUR <= offset <= length - 6 * HOUR
        shifted = gate.shift_tape(tape, offset, (start, end))
        assert gate.time_open(shifted, start, end) == opened
        times = start + np.random.default_rng(seed).integers(0, length, 2000)
        times = np.r_[times, start, end - 1, start + offset, start + offset - 1]
        source = start + (times - start - offset) % length
        np.testing.assert_array_equal(shifted.strictly_before(times), tape.strictly_before(source))
    assert gate.placebo_offset(7, length) == gate.placebo_offset(7, length)


# -- G(base) ----------------------------------------------------------------------------


def view(state: float | None, signal: str = gate.GATE_SIGNAL) -> MarketView:
    levels = np.arange(10, dtype=np.int64)
    sizes = np.ones(10)
    signals = {} if state is None else {signal: state}
    return MarketView(
        0, 0.01, 9_999 - levels, sizes, 10_002 + levels, sizes, 9_999, 10_002, 100.005, 5.0,
        10.0, 1.0, 6.0, signals,
    )  # fmt: skip


@pytest.mark.parametrize("position", [-2.0, 0.0, 3.0])
def test_g_is_its_base_while_open_and_reduce_only_while_closed(position: float) -> None:
    base = SkewQuoter(2.0, 0.0, 0.0, 5.0, 2.0)
    quoter = gate.GatedQuoter(base)
    wanted = base.quotes(view(1.0), position)
    assert quoter.quotes(view(1.0), position) == wanted
    for closed in (0.0, None, float("nan")):
        got = quoter.quotes(view(closed), position)
        if position > 0:
            assert got.bid.price is None and got.ask == wanted.ask
        elif position < 0:
            assert got.ask.price is None and got.bid == wanted.bid
        else:
            assert got.bid.price is None and got.ask.price is None
    # It reads its own tape only.
    other = gate.GatedQuoter(base, signal="gate_open|other")
    assert other.quotes(view(1.0), 0.0).bid.price is None


def test_g_open_all_day_is_its_base_and_closed_all_day_never_trades() -> None:
    events = market(n=3000)
    base = SkewQuoter(2.0, 0.0, 0.0, 5.0, 0.0)
    settings = SimConfig(clip_notional=1000.0, warmup_s=0, clip_touch_share=1.0)
    start = np.array([events.day_start_ns - 1])
    quoter = gate.GatedQuoter(base, signal="g")
    plain = simulate_day(events, base, settings)
    opened = simulate_day(events, quoter, settings, {"g": SignalTape("g", start, np.ones(1))})
    closed = simulate_day(events, quoter, settings, {"g": SignalTape("g", start, np.zeros(1))})
    assert plain.counters["fills"] > 100
    pd.testing.assert_frame_equal(opened.fills, plain.fills)
    assert opened.decomposition == plain.decomposition
    assert closed.fills.empty and closed.decomposition["net"] == 0.0


def test_g_with_a_real_gate_unwinds_but_never_adds_while_closed() -> None:
    events = market(n=4000)
    prints = gate.print_markouts(events)
    cell = gate.GateCell("S1", "trailing", 15, 0.0)
    tape = gate.state_tape([prints], cell, 1.0)
    base = SkewQuoter(2.0, 0.0, 0.0, 5.0, 1.0)
    settings = SimConfig(clip_notional=1000.0, warmup_s=0, clip_touch_share=1.0)
    result = simulate_day(events, gate.GatedQuoter(base, signal=cell.signal), settings,
                          {cell.signal: tape})  # fmt: skip
    fills = result.fills[result.fills["maker"]]
    assert len(fills)
    state = tape.strictly_before(fills["ts"].to_numpy())
    before = fills["position_after"] - fills["side"] * fills["size"]
    closed = state < 0.5
    assert 0 < closed.sum() < len(fills)
    # A fill while closed that starts from flat can only be an order placed while
    # open whose cancel was still in flight: within the latency of the closing.
    closings = tape.ts[tape.values < 0.5] + 1  # noqa: PD011
    from_flat = fills["ts"].to_numpy()[closed & (before == 0).to_numpy()]
    since = from_flat - closings[np.searchsorted(closings, from_flat, side="right") - 1]
    assert (since <= settings.cancel_latency_ns).all()


# -- the cells and the choice -------------------------------------------------------


def test_the_42_cells_and_their_neighbourhoods() -> None:
    cells = gate.gate_cells()
    assert len(cells) == 42 and len(set(cells)) == 42
    middle = gate.GateCell("S1", "trailing", 60, 1.0)
    assert len(gate.cell_neighbourhood(middle, cells)) == 9
    corner = gate.GateCell("S1_touch", "both", 15, 0.0)
    assert len(gate.cell_neighbourhood(corner, cells)) == 4
    hours = gate.GateCell("S1", "hours", None, 0.0)
    assert gate.cell_neighbourhood(hours, cells) == [hours, gate.GateCell("S1", "hours", None, 1.0)]
    assert gate.GateCell.from_record(middle.record()) == middle
    with pytest.raises(ValueError, match="window"):
        gate.GateCell("S1", "hours", 15, 0.0)
    with pytest.raises(ValueError, match="not registered"):
        gate.GateCell("S1", "trailing", 30, 0.0)


# -- the readers, end to end on a synthetic D day ------------------------------------


def put_on_disk(events: DayEvents, symbol: str, books: Path, trades: Path) -> None:
    """A synthetic day in the layout the downloaders write, under ``symbol``."""
    tick = events.spec.tick
    name = f"{events.day.isoformat()}.parquet"
    book: dict[str, object] = {"timestamp": pd.to_datetime(events.book_ts, unit="ns", utc=True)}
    for level in range(events.depth):
        book[f"bid_price_{level}"] = np.round(events.bid_px[:, level] * tick, 8)
        book[f"bid_size_{level}"] = events.bid_sz[:, level]
        book[f"ask_price_{level}"] = np.round(events.ask_px[:, level] * tick, 8)
        book[f"ask_size_{level}"] = events.ask_sz[:, level]
    (books / symbol).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(book).to_parquet(books / symbol / name, index=False)
    prints = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(events.trade_ts, unit="ns", utc=True),
            "price": np.round(events.trade_px * tick, 8),
            "size": events.trade_sz,
            "aggressor": events.trade_aggressor.astype(np.int64),
        }
    )
    (trades / symbol).mkdir(parents=True, exist_ok=True)
    prints.to_parquet(trades / symbol / name, index=False)


def test_the_readers_score_and_simulate_a_permitted_day(tmp_path: Path) -> None:
    from trading_research.market_making import heldout, round2, round2_run

    day = round2.block_days("D")[4]
    events = market(5, day=day, n=3000)
    books, trades = tmp_path / "book", tmp_path / "trades"
    put_on_disk(events, "GALAUSDT", books, trades)
    access = round2.development_access()
    scored, missing = round2_run.markouts(
        "GALAUSDT", [day], access=access, book_roots=[books], trades_root=trades,
        out=tmp_path / "markouts",
    )  # fmt: skip
    assert not missing
    np.testing.assert_array_equal(scored[day].markout_bp, gate.print_markouts(events).markout_bp)
    cell = gate.GateCell("S1", "trailing", 15, 0.0)
    tape = gate.state_tape([scored[day]], cell, 0.0)
    base = SkewQuoter(2.0, 0.0, 0.0, 5.0, 0.0)
    settings = SimConfig(clip_notional=1000.0, clip_touch_share=1.0)

    def factory(quoter: object) -> object:
        return quoter

    specs = [
        heldout.Spec("S1", partial(factory, base), settings),
        heldout.Spec(
            "G", partial(factory, gate.GatedQuoter(base, signal=cell.signal)), settings,
            tapes={cell.signal: tape},
        ),
    ]  # fmt: skip
    kwargs = {"access": access, "book_roots": [books], "trades_root": trades,
              "funding_root": None, "out_root": tmp_path / "rows"}  # fmt: skip
    rows, folders = round2_run.run_plans([("GALAUSDT", [day], specs, 2)], **kwargs)
    assert set(rows["label"]) == {"S1", "G"} and (rows["status"] == "ok").all()
    assert rows.set_index("label").loc["S1", "fills"] >= rows.set_index("label").loc["G", "fills"]
    again, _ = round2_run.run_plans([("GALAUSDT", [day], specs, 2)], **kwargs)
    pd.testing.assert_frame_equal(rows.sort_index(axis=1), again.sort_index(axis=1))
    assert folders["GALAUSDT"].name.startswith("GALAUSDT_")
    table = round2_run.screen(["GALAUSDT"], access=access, book_roots=[books], trades_root=trades)
    assert table.loc[0, "coverage_D"] == pytest.approx(1 / 25) and table.loc[0, "coverage_H"] == 0
    assert table.loc[0, "days_unavailable"] == 24
