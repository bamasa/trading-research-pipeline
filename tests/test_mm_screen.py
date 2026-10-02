"""The admission screen: its statistics, its rule, and what it is allowed to read."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making import prereg, screen
from trading_research.market_making.synthetic import hand_built_market

TICK = 0.0001  # one basis point at a price of 1.0


def market() -> object:
    """Three snapshots: two ticks wide for 10 s, one tick for 20 s, then four
    ticks until midnight; four prints of known sizes."""

    def levels(bid: int, ask: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        return [(bid - i, 20.0) for i in range(3)], [(ask + i, 40.0) for i in range(3)]

    snapshots = [
        (0.0, *levels(10_000, 10_002)),
        (10_000.0, *levels(10_000, 10_001)),
        (30_000.0, *levels(10_000, 10_004)),
    ]
    prints = [
        (1_000.0, 10_000, 2.0, -1),
        (2_000.0, 10_002, 4.0, 1),
        (40_000.0, 10_004, 4.0, 1),
        (50_000.0, 10_000, 8.0, -1),
    ]
    return hand_built_market(snapshots, prints, tick=TICK)


def test_statistics_are_time_weighted_over_the_day() -> None:
    stats = screen.day_statistics(market())
    day = 86_400.0
    assert stats.seconds == pytest.approx(day)
    widths_bp = np.array([2, 1, 4]) / np.array([10_001, 10_000.5, 10_002]) * 1e4
    durations = np.array([10.0, 20.0, day - 30.0])
    assert stats.spread_bp_seconds / stats.seconds == pytest.approx(
        np.dot(widths_bp, durations) / day
    )
    assert stats.two_tick_seconds == pytest.approx(10.0 + day - 30.0)
    assert stats.prints == 4
    # The touch is sampled every whole second: (20 + 40) / 2 throughout.
    assert np.nanmedian(stats.touch) == pytest.approx(30.0)
    combined = screen.combine([stats, stats])
    assert combined["median_print"] == pytest.approx(4.0)
    assert combined["median_touch_over_median_print"] == pytest.approx(7.5)
    assert combined["prints_per_day"] == pytest.approx(4.0)
    assert combined["share_at_two_ticks_or_more"] == pytest.approx((day - 20.0) / day)
    assert combined["clip_notional_usdt"] == pytest.approx(
        0.1 * combined["median_touch_notional_usdt"]
    )


def test_the_admission_rule_is_the_registered_one() -> None:
    table = pd.DataFrame(
        {
            "symbol": ["BICOUSDT", "CRVUSDT", "XRPUSDT", "BTCUSDT", "THINUSDT", "GAPUSDT"],
            "spread_bp_time_weighted": [6.0, 2.0, 1.9, 0.02, 9.0, 9.0],
            "share_at_two_ticks_or_more": [0.7, 0.07, 0.0, 0.0, 0.9, 0.9],
            "prints_per_day": [15_000, 70_000, 260_000, 1e6, 4_999, 20_000],
            "coverage_D": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_H": [1.0, 1.0, 1.0, 1.0, 1.0, 0.85],
        }
    )
    out = screen.admit(
        table,
        universe=["BICOUSDT", "CRVUSDT", "XRPUSDT", "BTCUSDT", "GAPUSDT"],
        exclude=["BTCUSDT", "ETHUSDT"],
    )
    assert out["mm_admitted"].tolist() == [True, False, False, False, False, False]
    assert out["h2_admitted"].tolist() == [True, True, True, False, False, False]
    edge = table.iloc[[0]].assign(spread_bp_time_weighted=4.0, share_at_two_ticks_or_more=0.25)
    assert screen.admit(edge, universe=[], exclude=[])["mm_admitted"].item()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_coverage_reads_the_file_system_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = [date(2024, 2, 26) + timedelta(days=i) for i in range(10)]
    for day in days[:9]:
        _touch(tmp_path / "book" / "AUSDT" / f"{day}.parquet")
    for day in days[1:]:
        _touch(tmp_path / "trades" / "AUSDT" / f"{day}.parquet")
    (tmp_path / "trades" / "AUSDT" / f"{days[5]}.parquet").write_bytes(b"")  # empty

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a file was opened")

    monkeypatch.setattr(pd, "read_parquet", never)
    monkeypatch.setattr(Path, "open", never)
    monkeypatch.setattr(Path, "read_bytes", never)
    value = screen.coverage(
        "AUSDT", days, book_roots=[tmp_path / "book"], trades_root=tmp_path / "trades"
    )
    assert value == pytest.approx(7 / 10)


def test_the_screen_reads_no_day_outside_d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a file was opened")

    monkeypatch.setattr(pd, "read_parquet", never)
    with pytest.raises(prereg.HeldOutLocked):
        screen.screen(
            ["BICOUSDT"],
            d_days=[*prereg.block_days("D"), date(2024, 2, 26)],
            h_days=prereg.block_days("H"),
            book_roots=[tmp_path],
            trades_root=tmp_path,
            access=prereg.development_access(),
        )
