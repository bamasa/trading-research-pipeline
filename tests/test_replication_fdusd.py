"""The 2025 BTC/FDUSD replication: its own data root and guard, its windows, its fees."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.models.base import CLASSES
from trading_research.replication import fdusd_fetch as rf
from trading_research.replication import fdusd_scoring as rs
from trading_research.replication import fdusd_windows as rw
from trading_research.replication.fdusd_summary import pooled
from trading_research.spot import synthetic as syn
from trading_research.spot.archives import BINANCE_SPOT, BYBIT_SPOT, PRINTS
from trading_research.spot.binance import BinanceChecksumError
from trading_research.spot.costs import CONFIG_PATH, load_fee_table
from trading_research.spot.plan import purge_ns
from trading_research.strategies.hold_first import HoldFirstRule

REPO = Path(__file__).resolve().parent.parent
DAYS = (date(2025, 2, 3), date(2025, 2, 4))


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


def _venue() -> syn.SpotVenue:
    return syn.SpotVenue(syn.SpotMarket(frozenset(DAYS), step_ms=30_000))


def test_the_guard_permits_the_replication_days_of_binance_prints_only() -> None:
    guard = rf.ReplicationAccess()
    guard.require(BINANCE_SPOT, [date(2024, 11, 1), date(2025, 6, 30)], PRINTS)
    for instrument, day, plane in ((BINANCE_SPOT, date(2025, 3, 15), PRINTS),
                                   (BINANCE_SPOT, date(2026, 8, 8), PRINTS),
                                   (BYBIT_SPOT, date(2025, 6, 1), PRINTS),
                                   (BINANCE_SPOT, date(2025, 6, 1), "book")):  # fmt: skip
        with pytest.raises(PermissionError):
            guard.require(instrument, [day], plane)


def test_a_day_lands_only_under_a_replication_root(tmp_path: Path) -> None:
    root = tmp_path / "data" / "replication"
    assert rf.target_path(DAYS[0], root) == (
        root / "binance" / "prints" / "BTCFDUSD" / "2025-02-03.parquet"
    )
    for wrong in (tmp_path / "data" / "spot", tmp_path / "data" / "replication" / "spot"):
        with pytest.raises(PermissionError):
            rf.target_path(DAYS[0], wrong)


def test_fetch_verifies_the_checksum_and_deletes_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, venue = tmp_path / "replication", _venue()
    published = venue.body
    found = rf.fetch_day(DAYS[0], root, opener=venue)
    assert found["status"] == "ok" and found["rows"] > 0 and found["unit"] == "us"
    assert rf.target_path(DAYS[0], root).exists()
    assert not list((root / "archives").glob("*"))
    assert rf.fetch_day(DAYS[0], root, opener=venue)["status"] == "on_disk"
    wrong = b"0" * 64 + b"  x.zip"
    monkeypatch.setattr(
        venue, "body", lambda url: wrong if url.endswith("CHECKSUM") else published(url)
    )
    with pytest.raises(BinanceChecksumError):
        rf.fetch_day(DAYS[1], root, opener=venue)
    assert not rf.target_path(DAYS[1], root).exists()
    assert not list((root / "archives").glob("*"))


def test_the_windows_are_the_spot_folds_over_their_dates() -> None:
    a, b = rw.WINDOWS["A"].fold(), rw.WINDOWS["B"].fold()
    assert (a.train[0], a.train[-1], len(a.train)) == (date(2024, 11, 5), date(2025, 1, 14), 71)
    assert (a.inner[0], a.inner[-1], a.test[0], a.test[-1]) == (
        date(2025, 1, 15), date(2025, 1, 21), date(2025, 2, 4), date(2025, 2, 26))  # fmt: skip
    assert (len(b.train), len(b.inner), len(b.test)) == (31, 7, 23)
    edge = a.inner_start_ns - purge_ns(rw.HOLD_S)
    keep = a.train_keep(np.array([edge - 1, edge]), np.array([edge, edge]), rw.HOLD_S)
    assert keep.tolist() == [True, False]


class _AlwaysLong:
    def __init__(self) -> None:
        self.columns_ = list(rw.INPUTS)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        out = np.full((len(x), 3), 0.1)
        out[:, CLASSES.index(1)] = 0.8
        return out


def test_the_two_fees_differ_only_by_ten_bp_of_turnover(tmp_path: Path) -> None:
    root, venue = tmp_path / "replication", _venue()
    for day in DAYS:
        rf.fetch_day(day, root, opener=venue)
    rule = HoldFirstRule(theta_long=0.5, theta_short=2.0)
    table = load_fee_table(REPO / CONFIG_PATH)
    rows = pd.DataFrame(rs.score_day("A", _AlwaysLong(), rule, table, DAYS[1], root))
    assert set(zip(rows["variant"], rows["fees"], strict=True)) == {
        (v, f) for v in rs.VARIANTS for f in rs.FEES}  # fmt: skip
    assert rows["warm"].all() and (rows["trades"] > 0).all() and (rows["trades_short"] == 0).all()
    zero, regular = (rows[rows["fees"] == f].set_index("variant") for f in rs.FEES)
    assert (zero["fees_quote"] == 0).all()
    assert np.allclose(regular["fees_quote"], 10e-4 * regular["turnover"])
    assert np.allclose(regular["net_quote"], zero["net_quote"] - regular["fees_quote"])
    summary = pooled(rows, windows=("A",)).set_index(["window", "variant", "fees"]).sort_index()
    break_even = summary.loc[("A", "two_sided", "zero"), "gross_bp_turnover"]
    assert np.isclose(break_even, zero.loc["two_sided", "gross_quote"]
                      / zero.loc["two_sided", "turnover"] * 1e4)  # fmt: skip
