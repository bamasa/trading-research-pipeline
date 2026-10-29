"""The D procedure's summaries: sigma_D and the MDE, Null A, the usable days."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from trading_research.spot import access as sa
from trading_research.spot import development as dv
from trading_research.spot import statuses as st


def slot(values: list[float]) -> st.SlotDays:
    n = len(values)
    days = [date(2026, 6, 1 + k) for k in range(n)]
    nan = np.full(n, np.nan)
    return st.SlotDays("S25-BB", days, np.array(values), np.ones(n), nan, nan,
                       np.full((200, n), np.nan), nan, nan, nan, nan, nan)  # fmt: skip


def test_sigma_and_the_mde_over_the_days_with_a_trade() -> None:
    s = slot([1.0, 3.0, np.nan, 5.0])
    found = dv.sigma_and_mde(s)
    assert found["sigma_d"] == 2.0 and found["share_days_with_trade"] == 0.75
    assert math.isclose(found["mde"], 3.84 * 2.0 / math.sqrt(75 * 0.75))
    assert math.isnan(dv.sigma_and_mde(slot([np.nan, 1.0]))["sigma_d"])


def test_null_a_compares_each_draw_with_the_true_mean() -> None:
    s = slot([1.0, 1.0])
    records = {d.isoformat(): {"rows": {"S25-BB": {"null_a": [0.0, 2.0, None]}}} for d in s.days}
    found = dv.null_a(records, s, 3)
    assert found["true_mean"] == 1.0 and found["p"] == (1 + 1) / (1 + 3)


def test_usable_days_on_either_venue() -> None:
    days = [date(2026, 5, 1), date(2026, 5, 2)]
    record = {"days": {"2026-05-01": {sa.BINANCE: {"reasons": []}, sa.BYBIT: {"reasons": ["x"]}},
                       "2026-05-02": {sa.BINANCE: {"reasons": ["y"]}, sa.BYBIT: {"reasons": ["x"]}}}}  # fmt: skip
    assert dv.usable_days(record, days) == [date(2026, 5, 1)]
    assert dv.usable_days(record, days, sa.BYBIT) == []


def _day_inputs(day: date):  # type: ignore[no-untyped-def]
    import pandas as pd

    from trading_research.data.study_loader import BookDay
    from trading_research.spot import days as dy
    from trading_research.spot import loader as ld

    stamps = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=24, freq="h")
    prints = pd.DataFrame({"timestamp": stamps, "price": 100.0, "size": 1.0, "aggressor": 1})
    book = pd.DataFrame({"timestamp": stamps, "bid_price_0": 99.0, "ask_price_0": 101.0,
                         "bid_size_0": 1.0, "ask_size_0": 1.0})  # fmt: skip
    return dy.DayInputs(day, ld.PrintsDay("prints", day, prints, 0.01),
                        ld.PrintsDay("prints", day, prints.copy(), 0.01),
                        BookDay("bybit-spot:BTCUSDT", day, book, 0, 0.01),
                        {sa.BINANCE: (), sa.BYBIT: ()})  # fmt: skip


def test_the_mutated_day_changes_only_what_lies_after_the_cutoff() -> None:
    from trading_research.spot.plan import midnight_ns

    day = date(2026, 6, 30)
    cutoff = midnight_ns(day) + 12 * 3_600 * 1_000_000_000  # rows 0..12 at or before it
    found = dv.mutated_inputs(_day_inputs(day), cutoff)
    for frame, column, value in ((found.binance.frame, "price", 100.0),
                                 (found.bybit.frame, "price", 100.0),
                                 (found.book.frame, "ask_price_0", 101.0)):  # fmt: skip
        assert (frame[column].iloc[:13] == value).all()
        assert (frame[column].iloc[13:] != value).all()
    assert (found.book.frame["bid_price_0"] < found.book.frame["ask_price_0"]).all()


def _fold_leak(tmp_path, monkeypatch, reads_after: bool):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from trading_research.spot import heldout as ho
    from trading_research.spot.plan import folds

    usable = [date(2026, 5, 1) + timedelta(days=k) for k in range(92)]
    fold = folds(usable)["june"]

    def fit_fold(study, fold_, load, touch, read=None, folder=None):  # type: ignore[no-untyped-def]
        day = fold_.test[0] if reads_after else fold_.inner[-1]  # a theta that reads a test day
        price = float(load(day).binance.frame["price"].iloc[-1])
        return {"fits": {}, "thresholds": {"S25-PF": {"cell": {"theta_long": price}}}}

    honest = {"fits": {}, "thresholds": {"S25-PF": {"cell": {"theta_long": 100.0}}}}
    monkeypatch.setattr(dv, "fit_fold", fit_fold)
    monkeypatch.setattr(ho, "loader", lambda *_: _day_inputs)
    study = SimpleNamespace(work=tmp_path, scale=SimpleNamespace(null_a_draws=0),
                            dir=lambda *parts: tmp_path.joinpath(*parts))  # fmt: skip
    spec = {i.key: {"tick": 0.01, "lot": 1e-6} for i in sa.INSTRUMENTS.values()}
    record1 = {"max_sequence_gaps": 0, "walk": 10, "instrument_spec": spec, "touch_size": 1.0}
    return dv.fold_leak(study, fold, record1, honest)  # type: ignore[arg-type]


def test_the_fold_level_test_runs_the_fit_path_and_sees_a_theta_that_reads_ahead(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    assert _fold_leak(tmp_path, monkeypatch, reads_after=True)["changed"] == ["theta S25-PF"]
    assert not _fold_leak(tmp_path, monkeypatch, reads_after=False)["leaks"]
