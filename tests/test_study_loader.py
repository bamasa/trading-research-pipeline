"""The study's book-day loader: which days are excluded, by which rule, and the clip.

Each exclusion rule is a statement about which days every verdict rests on, so
each is pinned on a synthetic day that breaks exactly that rule and no other.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data import study_loader as sl

DAY = date(2024, 12, 5)


def book(rows: int = 30, *, step: float = 0.1, gaps: int = 0) -> pd.DataFrame:
    start = datetime(2024, 12, 5, tzinfo=UTC)
    data: dict[str, object] = {
        "timestamp": pd.date_range(start, periods=rows, freq="100ms"),
        "symbol": pd.array(["BTCUSDT"] * rows, dtype="string"),
        "source": pd.array(["bybit-linear-ob500"] * rows, dtype="string"),
        "sequence_id": np.arange(rows, dtype=np.int64) * 3 + 1,
    }
    mid = 97_000.0 + step * np.arange(rows) % 7
    for level in range(10):
        data[f"bid_price_{level}"] = np.round(mid - step * (level + 1), 1)
        data[f"bid_size_{level}"] = np.full(rows, 0.5)
        data[f"ask_price_{level}"] = np.round(mid + step * level, 1)
        data[f"ask_size_{level}"] = np.full(rows, 0.5)
    frame = pd.DataFrame(data)
    frame.attrs["sequence_gaps"] = gaps
    return frame


def stored(tmp_path: Path, frame: pd.DataFrame, day: date = DAY) -> Path:
    path = tmp_path / "BTCUSDT" / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return tmp_path


def test_a_clean_day_is_usable_with_its_own_tick_and_findings(tmp_path: Path) -> None:
    root = stored(tmp_path, book())
    loaded = sl.load_book_day("BTCUSDT", DAY, root, max_sequence_gaps=3, spec_tick=0.1)
    assert loaded.usable and loaded.exclusions == () and loaded.notes == ()
    assert loaded.tick == pytest.approx(0.1)
    assert loaded.sequence_gaps == 0 and loaded.frame is not None
    assert loaded.findings["depth"] == {"info": 1}


def test_a_non_positive_price_excludes_the_day(tmp_path: Path) -> None:
    frame = book()
    frame.loc[4, "bid_price_9"] = 0.0
    loaded = sl.load_book_day("BTCUSDT", DAY, stored(tmp_path, frame), max_sequence_gaps=3)
    assert loaded.exclusions == (sl.NON_POSITIVE_PRICE,)
    assert loaded.findings["positive_prices"]["error"] >= 1


def test_a_price_off_any_tick_grid_excludes_the_day(tmp_path: Path) -> None:
    frame = book()
    frame.loc[2, "bid_price_9"] = 3.14159265358979  # on no grid of ten decimals or fewer
    loaded = sl.load_book_day("BTCUSDT", DAY, stored(tmp_path, frame), max_sequence_gaps=3)
    assert sl.OFF_GRID_PRICE in loaded.exclusions
    assert loaded.tick is None


def test_gaps_above_the_frozen_threshold_exclude_and_none_only_records(tmp_path: Path) -> None:
    root = stored(tmp_path, book(gaps=5))
    assert sl.load_book_day("BTCUSDT", DAY, root, max_sequence_gaps=4).exclusions == (
        sl.SEQUENCE_GAPS,
    )
    assert sl.load_book_day("BTCUSDT", DAY, root, max_sequence_gaps=5).usable
    computing = sl.load_book_day("BTCUSDT", DAY, root, max_sequence_gaps=None)
    assert computing.usable and computing.sequence_gaps == 5


@pytest.mark.parametrize(
    ("spec", "relation"), [(0.01, "a multiple of"), (0.3, "not a multiple of")]
)
def test_a_tick_unlike_the_specifications_is_noted_and_kept(
    tmp_path: Path, spec: float, relation: str
) -> None:
    loaded = sl.load_book_day(
        "BTCUSDT", DAY, stored(tmp_path, book()), max_sequence_gaps=0, spec_tick=spec
    )
    assert loaded.usable
    assert loaded.notes and f"({relation} it)" in loaded.notes[0]


def test_a_day_without_a_book_file_is_excluded(tmp_path: Path) -> None:
    loaded = sl.load_book_day("BTCUSDT", DAY, tmp_path, max_sequence_gaps=0)
    assert loaded.exclusions == (sl.NO_BOOK,) and loaded.frame is None


def test_the_gap_threshold_is_the_95th_percentile_rounded_up() -> None:
    assert sl.max_sequence_gaps([0] * 19 + [10]) == 1  # 0.5 by linear interpolation
    assert sl.max_sequence_gaps([3] * 40) == 3  # no float noise added
    assert sl.max_sequence_gaps(list(range(101))) == 95
    with pytest.raises(ValueError, match="no per-day gap counts"):
        sl.max_sequence_gaps([])


def test_exclusions_are_counted_by_reason_and_a_block_can_be_unreadable() -> None:
    days = [
        sl.BookDay("BTCUSDT", DAY, None, None, None, exclusions=(sl.NO_BOOK,)),
        sl.BookDay("BTCUSDT", DAY, None, 9, None, exclusions=(sl.OFF_GRID_PRICE, sl.SEQUENCE_GAPS)),
        sl.BookDay("BTCUSDT", DAY, None, 0, 0.1),
    ]
    assert sl.exclusion_counts(days) == {
        sl.NO_BOOK: 1,
        sl.OFF_GRID_PRICE: 1,
        sl.SEQUENCE_GAPS: 1,
    }
    assert not sl.block_unreadable(11, 44)  # exactly a quarter
    assert sl.block_unreadable(12, 44)
    with pytest.raises(ValueError, match="at least one day"):
        sl.block_unreadable(0, 0)


def test_the_clip_sits_on_the_lot_grid() -> None:
    btc = sl.StudyInstrument.from_spec("BTCUSDT", tick=0.1, lot=0.001)
    assert btc.clip == pytest.approx(0.010)
    with pytest.raises(ValueError, match="not a multiple of the lot"):
        sl.StudyInstrument("BTCUSDT", tick=0.1, lot=0.003, clip=0.010)
    with pytest.raises(KeyError, match="no clip registered"):
        sl.StudyInstrument.from_spec("ETHUSDT", tick=0.01, lot=0.01)
    with pytest.raises(ValueError, match="positive"):
        sl.StudyInstrument("BTCUSDT", tick=0.0, lot=0.001, clip=0.01)


def _info(symbol: str) -> dict:
    return {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": symbol,
                    "priceFilter": {"tickSize": "0.10"},
                    "lotSizeFilter": {"qtyStep": "0.001"},
                }
            ]
        },
    }


def test_the_tick_and_lot_come_from_the_venues_specification() -> None:
    assert sl.parse_instrument_info(_info("BTCUSDT"), "BTCUSDT") == (0.1, 0.001)
    with pytest.raises(KeyError, match="not in the instrument list"):
        sl.parse_instrument_info(_info("ETHUSDT"), "BTCUSDT")
    with pytest.raises(ValueError, match="retCode"):
        sl.parse_instrument_info({"retCode": 10001}, "BTCUSDT")

    asked: list[str] = []

    def opener(url: str, timeout: float) -> io.BytesIO:
        asked.append(url)
        return io.BytesIO(json.dumps(_info("BTCUSDT")).encode())

    spec = sl.fetch_instrument("BTCUSDT", opener=opener)
    assert (spec.tick, spec.lot, spec.clip) == (0.1, 0.001, 0.010)
    assert asked == [sl.INSTRUMENT_INFO_URL.format(symbol="BTCUSDT")]
