"""The print label and the forward move on prints: hand cases, fees, the purge's honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import DAY, make_prints, no_network
from trading_research.spot.fills import BUY, SELL, PrintTape, round_trips
from trading_research.spot.grid import grid_stamps_ns
from trading_research.spot.print_label import (
    PrintLabel,
    classify,
    forward_move_bp,
    label_column,
    print_targets,
)

S = 1_000_000_000
MS = 1_000_000
T0 = pd.Timestamp(DAY, tz="UTC").value + 3600 * S


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


def frame(rows: list[tuple[int, float, float, int]]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["ms", "price", "size", "aggressor"])
    out.insert(0, "timestamp", pd.to_datetime(T0 + out.pop("ms") * MS, unit="ns", utc=True))
    return out.assign(rpi=False)


HAND = frame(
    [
        (0, 99.0, 1.0, SELL),  # a row at 0 ms
        (5, 101.0, 1.0, BUY),  # before the decision at 0 ms arrives (10 ms)
        (20, 101.5, 1.0, BUY),  # the long's entry fill
        (30, 99.5, 1.0, SELL),  # the short's entry fill
        (2_000, 103.0, 1.0, BUY),  # a row
        (2_050, 102.5, 1.0, SELL),
        (2_100, 103.5, 1.0, BUY),  # the row both exits are decided at
        (2_150, 101.6, 1.0, SELL),  # the long's exit fill
        (2_200, 104.0, 1.0, BUY),  # the short's exit fill
    ]
)


def test_a_hand_label_by_the_registered_formula() -> None:
    tape, grid = PrintTape.from_frame(HAND, DAY), grid_stamps_ns(HAND)
    out = PrintLabel(2.0).outcomes(tape, grid)
    first = out.iloc[0]
    # Long: buys at 101.5 (20 ms); its exit is decided at the first row at or
    # after 2,020 ms (2,100 ms) and sells at the next seller print (2,150 ms).
    mid_long = (101.5 + 99.0) / 2
    assert first["long_bp"] == pytest.approx((101.6 - 101.5) / mid_long * 1e4)
    # Short: sells at 99.5 (30 ms); exit decided at 2,100 ms, buys at 2,200 ms.
    mid_short = (101.5 + 99.5) / 2
    assert first["short_bp"] == pytest.approx((99.5 - 104.0) / mid_short * 1e4)
    assert first["reads_until_ns"] == T0 + 2_200 * MS
    assert PrintLabel(2.0)(tape, grid).iloc[0] == BUY
    # A fee of 10 bp on both sides turns it into a hold.
    assert PrintLabel(2.0, label_fee_bp=10.0)(tape, grid).iloc[0] == 0
    # The last rows have no exit before the day ends: no label.
    assert np.isnan(PrintLabel(2.0)(tape, grid).iloc[-1])


def test_classes_take_the_larger_when_both_pay_and_hold_on_a_tie() -> None:
    long = np.array([1.0, -1.0, 2.0, 1.0, 3.0, 0.0, np.nan, 1.0])
    short = np.array([-1.0, 1.0, 1.0, 2.0, 3.0, 0.0, 1.0, np.nan])
    expected = [BUY, SELL, BUY, SELL, 0, 0, np.nan, np.nan]
    np.testing.assert_array_equal(classify(long, short), np.array(expected, dtype=float))


def test_the_label_uses_the_executors_round_trips() -> None:
    prints = make_prints(800, seed=8, quiet_at=300)
    tape, grid = PrintTape.from_frame(prints, DAY), grid_stamps_ns(prints)
    out = PrintLabel(20.0).outcomes(tape, grid)
    trips = round_trips(tape, grid, grid, BUY, 20 * S, flatten=False)
    done = trips.complete
    assert np.array_equal(out["long_exit_ns"].to_numpy()[done], trips.exit.print_ns[done])
    assert np.isnan(out["long_bp"].to_numpy()[~done]).all()
    assert done.mean() > 0.5 and not done.all()


def test_the_label_reads_nothing_after_its_reported_stamp() -> None:
    prints = make_prints(600, seed=9, quiet_at=250)
    times = prints["timestamp"].astype("int64").to_numpy()
    tape, grid = PrintTape.from_frame(prints, DAY), grid_stamps_ns(prints)
    full = print_targets(tape, grid, [20.0], [0.0, 10.0])
    labelled = np.flatnonzero(full["reads_until_ns"].to_numpy() >= 0)
    assert len(labelled) > 100
    for row in labelled[[5, len(labelled) // 3, len(labelled) // 2, -1]]:
        reads = int(full["reads_until_ns"].iloc[row])
        assert reads >= grid[row] + 20 * S
        kept = prints[times <= reads]
        cut = print_targets(PrintTape.from_frame(kept, DAY), grid_stamps_ns(kept), [20.0],
                            [0.0, 10.0])  # fmt: skip
        for fee in (0.0, 10.0):
            assert cut[label_column(20.0, fee)].iloc[row] == full[label_column(20.0, fee)].iloc[row]


def test_targets_at_every_fee_equal_the_label_at_that_fee() -> None:
    prints = make_prints(500, seed=10, quiet_at=None)
    tape, grid = PrintTape.from_frame(prints, DAY), grid_stamps_ns(prints)
    at = np.arange(0, len(grid), 10)
    targets = print_targets(tape, grid, [20.0, 50.0], [0.0, 10.0], at=at)
    for hold in (20.0, 50.0):
        for fee in (0.0, 10.0):
            direct = PrintLabel(hold, label_fee_bp=fee)(tape, grid).to_numpy()[at]
            np.testing.assert_array_equal(targets[label_column(hold, fee)].to_numpy(), direct)
        own = PrintLabel(hold).outcomes(tape, grid)["reads_until_ns"].to_numpy()[at]
        np.testing.assert_array_equal(targets[f"reads_until_ns_{hold:g}"].to_numpy(), own)
    assert targets["timestamp"].tolist() == grid[at].tolist()
    np.testing.assert_array_equal(
        targets["reads_until_ns"],
        np.maximum(targets["reads_until_ns_20"], targets["reads_until_ns_50"]),
    )


def test_the_forward_move_is_mid_proxy_to_mid_proxy() -> None:
    tape, grid = PrintTape.from_frame(HAND, DAY), grid_stamps_ns(HAND)
    move = forward_move_bp(tape, grid, 2.0)
    # Row 0: first print after 10 ms is the buyer print at 20 ms (mid 100.25);
    # the first row at or after 2,020 ms is 2,100 ms (mid (103.5 + 102.5) / 2).
    assert move[0] == pytest.approx((103.0 / 100.25 - 1) * 1e4)
    assert np.isnan(move[-1])
    with pytest.raises(ValueError, match="positive"):
        PrintLabel(0.0)
