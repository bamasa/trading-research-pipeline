"""Tests for the generated feature set, the trade alignment and the selection.

Three things here are worth more than the rest.

The alignment tests pin the direction of the as-of join. A forward or nearest
join reads trades that had not happened yet, produces entirely plausible
numbers, and inflates everything downstream.

The z-score tests pin a fix found on real data: BTCUSDT futures sit at a
one-tick spread almost permanently, so a rolling standard deviation of zero is
the normal case, not an edge case. Dividing by it left 80% of rows unusable.

The selection tests pin that fitting happens on the training block only, and
that the ranking measures direction rather than volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobml.features.align import (
    attach_to_book,
    build_trade_features,
    prepare_trades,
    rolling_trade_stats,
)
from lobml.features.generated import base_quantities, expand, generate, time_of_day
from lobml.features.registry import Feature
from lobml.features.selection import FeatureSelector
from lobml.validation.leakage import check_feature


def make_book(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 30_000 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="100ms", tz="UTC"),
            "symbol": pd.array(["X"] * n, dtype="string"),
            "sequence_id": np.arange(1, n + 1, dtype="int64"),
            "source": pd.array(["test"] * n, dtype="string"),
            "bid_price_0": mid - 0.05,
            "bid_size_0": rng.gamma(4, 0.5, n),
            "ask_price_0": mid + 0.05,
            "ask_size_0": rng.gamma(4, 0.5, n),
        }
    )


def make_trades(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="150ms", tz="UTC"),
            "symbol": pd.array(["X"] * n, dtype="string"),
            "price": 30_000 + rng.normal(0, 1, n),
            "quantity": rng.gamma(2, 0.1, n),
            "is_buyer_maker": rng.random(n) < 0.5,
            "trade_id": np.arange(1, n + 1, dtype="int64"),
            "source": pd.array(["test"] * n, dtype="string"),
        }
    )


# ---------------------------------------------------------------------------
# Trade alignment
# ---------------------------------------------------------------------------


def test_alignment_never_reads_a_later_trade() -> None:
    """The property the whole module exists to guarantee.

    A book row at t may only see trades at or before t. Verified by giving the
    book a row that precedes every trade: it must come back empty, not filled
    from the first trade that follows.
    """
    trades = make_trades(100)
    stats = rolling_trade_stats(prepare_trades(trades), windows=("1s",))

    early = pd.DataFrame({"timestamp": [trades["timestamp"].iloc[0] - pd.Timedelta(5, unit="s")]})
    aligned = attach_to_book(early, stats)
    assert aligned.isna().all(axis=1).iloc[0]


def test_alignment_uses_the_most_recent_trade() -> None:
    trades = make_trades(200)
    stats = rolling_trade_stats(prepare_trades(trades), windows=("1s",))
    at = trades["timestamp"].iloc[100]
    aligned = attach_to_book(pd.DataFrame({"timestamp": [at]}), stats)
    assert aligned["trade_count_1s"].iloc[0] == pytest.approx(stats["trade_count_1s"].iloc[100])


def test_stale_flow_is_left_missing_rather_than_carried_forward() -> None:
    """An hour-old summary is not a description of current flow."""
    trades = make_trades(50)
    stats = rolling_trade_stats(prepare_trades(trades), windows=("1s",))
    much_later = pd.DataFrame(
        {"timestamp": [trades["timestamp"].iloc[-1] + pd.Timedelta(1, unit="h")]}
    )
    aligned = attach_to_book(much_later, stats, tolerance=pd.Timedelta(60, unit="s"))
    assert aligned.isna().all(axis=1).iloc[0]


def test_trade_imbalance_is_bounded_and_signed() -> None:
    book, trades = make_book(500), make_trades(400)
    out = build_trade_features(book, trades, windows=("5s",))
    values = out["trade_imbalance_5s"].dropna()
    assert len(values) > 100
    assert values.between(-1.0, 1.0).all()


def test_all_buys_give_imbalance_of_one() -> None:
    trades = make_trades(300)
    trades["is_buyer_maker"] = False  # every trade buyer-initiated
    out = build_trade_features(make_book(500), trades, windows=("5s",))
    assert out["trade_imbalance_5s"].dropna().iloc[-1] == pytest.approx(1.0)


def test_all_sells_give_imbalance_of_minus_one() -> None:
    trades = make_trades(300)
    trades["is_buyer_maker"] = True
    out = build_trade_features(make_book(500), trades, windows=("5s",))
    assert out["trade_imbalance_5s"].dropna().iloc[-1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Generated features
# ---------------------------------------------------------------------------


def test_generation_produces_a_wide_frame_on_the_book_index() -> None:
    book = make_book()
    out = generate(book, make_trades())
    assert out.shape[1] > 100
    assert out.index.equals(book.index)


def test_price_level_columns_are_dropped() -> None:
    """A model given the price level learns 'February 2024', not 'markets'."""
    out = generate(make_book())
    assert "mid" not in out.columns
    assert "log_mid" not in out.columns


def test_constant_series_gives_a_zero_z_score_not_a_missing_one() -> None:
    """Found on real data: BTCUSDT sits at one tick, so flat windows are normal.

    Treating a zero standard deviation as 'unknown' rather than 'no deviation'
    left 80% of rows unusable on the shortest window.
    """
    book = make_book(500)
    book["bid_price_0"] = 30_000.0 - 0.05
    book["ask_price_0"] = 30_000.0 + 0.05  # spread now exactly constant

    out = expand(base_quantities(book), windows=(5,), lags=(1,), diff_columns=())
    z = out["spread_bp_z5"]
    warm = z.iloc[10:]
    assert warm.notna().all()
    assert np.allclose(warm.to_numpy(), 0.0)


def test_warmup_rows_stay_missing() -> None:
    """Only genuinely absent history is NaN; the fix must not hide that."""
    out = expand(base_quantities(make_book(500)), windows=(50,), lags=(1,), diff_columns=())
    assert out["spread_bp_mean50"].iloc[:49].isna().all()


def test_time_features_are_cyclical() -> None:
    """23:59 and 00:01 must be adjacent, which raw hour numbers make them not."""
    book = make_book(10)
    book["timestamp"] = pd.to_datetime(
        ["2024-02-01 23:59:30", "2024-02-02 00:00:30"] * 5, utc=True
    ).sort_values()
    out = time_of_day(book)
    assert abs(out["tod_cos"].iloc[0] - out["tod_cos"].iloc[-1]) < 0.01


def test_generated_features_do_not_read_the_future() -> None:
    """Every generated column, through the same check the registry uses."""
    book = make_book(2000)
    frame = generate(book, make_trades(1500))
    for name in frame.columns:
        probe = Feature(
            name=name,
            plane="book",
            lookback=0,
            fn=lambda df, _n=name: generate(df, make_trades(1500))[_n],
            description="",
        )
        result = check_feature(probe, book, cut=0.6)
        assert not result.leaks, f"{name}: {result.detail}"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@pytest.fixture
def selection_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A frame where one column predicts direction and another only volatility."""
    rng = np.random.default_rng(3)
    n = 4000
    signal = rng.normal(size=n)
    volatility = np.abs(rng.normal(size=n)) + 0.5
    forward = signal * 2.0 + rng.normal(0, 1, n) * volatility

    x = pd.DataFrame(
        {
            "directional": signal,
            "volatility_only": volatility,
            "noise": rng.normal(size=n),
            "constant": np.ones(n),
            "duplicate": signal * 1.0001,
        }
    )
    y = pd.Series(np.sign(forward) * (np.abs(forward) > 2.0))
    return x, y, pd.Series(forward)


def test_constant_column_is_dropped(selection_data) -> None:
    x, y, fwd = selection_data
    selector = FeatureSelector(max_features=3).fit(x, y, fwd)
    assert "constant" in selector.report_.dropped_low_variance


def test_near_duplicate_is_dropped(selection_data) -> None:
    x, y, fwd = selection_data
    selector = FeatureSelector(max_features=3).fit(x, y, fwd)
    assert "duplicate" in selector.report_.dropped_correlated


def test_ic_ranking_prefers_the_directional_feature(selection_data) -> None:
    """The reason 'ic' is the default: it ranks direction above magnitude."""
    x, y, fwd = selection_data
    selector = FeatureSelector(max_features=1, score_target="ic").fit(x, y, fwd)
    assert selector.selected_ == ["directional"]


def test_ic_ranking_requires_the_forward_return(selection_data) -> None:
    x, y, _ = selection_data
    with pytest.raises(ValueError, match="forward"):
        FeatureSelector(score_target="ic").fit(x, y)


def test_transform_keeps_only_selected_columns(selection_data) -> None:
    x, y, fwd = selection_data
    selector = FeatureSelector(max_features=2).fit(x, y, fwd)
    assert list(selector.transform(x).columns) == selector.selected_


def test_transform_before_fit_is_refused(selection_data) -> None:
    x, _, _ = selection_data
    with pytest.raises(RuntimeError, match="not been fitted"):
        FeatureSelector().transform(x)


def test_missing_selected_column_is_reported(selection_data) -> None:
    x, y, fwd = selection_data
    selector = FeatureSelector(max_features=2).fit(x, y, fwd)
    with pytest.raises(ValueError, match="missing selected"):
        selector.transform(x.drop(columns=selector.selected_[:1]))


def test_selection_refuses_a_block_it_cannot_judge() -> None:
    tiny = pd.DataFrame({"a": np.arange(20.0)})
    with pytest.raises(ValueError, match="complete rows"):
        FeatureSelector().fit(tiny, pd.Series(np.zeros(20)), pd.Series(np.zeros(20)))


def test_report_records_what_each_stage_removed(selection_data) -> None:
    x, y, fwd = selection_data
    report = FeatureSelector(max_features=2).fit(x, y, fwd).report_
    assert report.n_input == 5
    assert len(report.selected) == 2
    assert "->" in report.summary()
