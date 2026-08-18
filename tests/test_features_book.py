"""Tests for the order-book features.

Two kinds of test here. The first checks formulas against hand-computed values
on tiny frames, because a feature that is merely plausible is a feature nobody
can debug later. The second checks *scale invariance*: a feature meant to
transfer between instruments must not change when the instrument's units do.

That second group exists because of a real finding. On February 2024 data,
``ofi_1`` has a standard deviation around 2x104 for XRPUSDT, whose touch holds
~105 units, against single-digit quantities for BTCUSDT. Anything fitted on one
would read the other as permanently extreme — so the transfer experiment would
have measured a unit mismatch and called it a market finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.data.schema import ask_price_col, ask_size_col, bid_price_col, bid_size_col
from trading_research.features.registry import REGISTRY, build

# Features whose value must not depend on the instrument's price or size units.
# A feature outside this list has to justify itself: raw ofi_1 and ofi_20 are
# kept in size units because that is their published definition, and the
# normalised variant is what a cross-instrument model should use.
SCALE_FREE = [
    "spread_bp",
    "queue_imbalance",
    "microprice_dev_bp",
    "log_depth_ratio",
    "mid_return_1_bp",
    "mid_return_20_bp",
    "realized_vol_50_bp",
    "spread_bp_ratio_50",
    "ofi_20_norm",
    # Short-horizon set. Every one is a ratio, a share, a basis-point return or
    # a row count, so none of them carries the instrument's price or size units
    # — which is the property that lets a model fitted on BTCUSDT say anything
    # about XRPUSDT.
    "quote_intensity_20",
    "imbalance_slope_10",
    "touch_persistence_20",
    "mid_reversal_10",
    "spread_pressure_20",
    "size_shock_10",
    "micro_drift_5",
    "realised_vol_20",
]

UNIT_DEPENDENT = ["ofi_1", "ofi_20", "log_total_depth"]


def make_book(
    bid_price: list[float],
    ask_price: list[float],
    bid_size: list[float],
    ask_size: list[float],
) -> pd.DataFrame:
    n = len(bid_price)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="100ms", tz="UTC"),
            "symbol": pd.array(["X"] * n, dtype="string"),
            "sequence_id": np.arange(1, n + 1, dtype="int64"),
            "source": pd.array(["test"] * n, dtype="string"),
            bid_price_col(0): bid_price,
            bid_size_col(0): bid_size,
            ask_price_col(0): ask_price,
            ask_size_col(0): ask_size,
        }
    )


def value(name: str, df: pd.DataFrame, at: int = -1) -> float:
    return float(REGISTRY.get(name)(df).iloc[at])


# ---------------------------------------------------------------------------
# Formulas, against values worked out by hand
# ---------------------------------------------------------------------------


def test_spread_is_in_basis_points_of_mid() -> None:
    df = make_book([99.0], [101.0], [1.0], [1.0])
    # spread 2 on a mid of 100 is 200 bp.
    assert value("spread_bp", df) == pytest.approx(200.0)


def test_queue_imbalance_is_bid_minus_ask_over_total() -> None:
    df = make_book([99.0], [101.0], [3.0], [1.0])
    assert value("queue_imbalance", df) == pytest.approx(0.5)


def test_queue_imbalance_is_positive_when_bids_dominate() -> None:
    """Sign convention: positive means buying pressure, everywhere."""
    heavy_bid = make_book([99.0], [101.0], [9.0], [1.0])
    heavy_ask = make_book([99.0], [101.0], [1.0], [9.0])
    assert value("queue_imbalance", heavy_bid) > 0
    assert value("queue_imbalance", heavy_ask) < 0


def test_queue_imbalance_is_bounded() -> None:
    only_bid = make_book([99.0], [101.0], [5.0], [0.0])
    only_ask = make_book([99.0], [101.0], [0.0], [5.0])
    assert value("queue_imbalance", only_bid) == pytest.approx(1.0)
    assert value("queue_imbalance", only_ask) == pytest.approx(-1.0)


def test_microprice_leans_towards_the_thinner_side() -> None:
    """The microprice weights each side by the size resting opposite it."""
    df = make_book([99.0], [101.0], [3.0], [1.0])
    # micro = (99*1 + 101*3) / 4 = 100.5, i.e. 0.5 above a mid of 100 = 50 bp.
    assert value("microprice_dev_bp", df) == pytest.approx(50.0)


def test_microprice_deviation_is_zero_when_balanced() -> None:
    df = make_book([99.0], [101.0], [2.0], [2.0])
    assert value("microprice_dev_bp", df) == pytest.approx(0.0)


def test_microprice_and_queue_imbalance_agree_in_sign() -> None:
    df = make_book([99.0, 99.0], [101.0, 101.0], [4.0, 1.0], [1.0, 4.0])
    micro = REGISTRY.get("microprice_dev_bp")(df)
    queue = REGISTRY.get("queue_imbalance")(df)
    assert np.sign(micro.to_numpy()).tolist() == np.sign(queue.to_numpy()).tolist()


def test_log_depth_ratio_is_symmetric() -> None:
    more_bid = make_book([99.0], [101.0], [4.0], [1.0])
    more_ask = make_book([99.0], [101.0], [1.0], [4.0])
    assert value("log_depth_ratio", more_bid) == pytest.approx(-value("log_depth_ratio", more_ask))


def test_mid_return_is_in_basis_points() -> None:
    df = make_book([99.0, 99.99], [101.0, 102.01], [1.0, 1.0], [1.0, 1.0])
    # mid goes 100 -> 101.0, a log return of ~99.5 bp.
    assert value("mid_return_1_bp", df) == pytest.approx(1e4 * np.log(101.0 / 100.0))


def test_degenerate_book_yields_nan_not_zero() -> None:
    """No resting size means unknown, not balanced."""
    df = make_book([99.0], [101.0], [0.0], [0.0])
    assert np.isnan(value("queue_imbalance", df))


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------


def test_ofi_counts_added_bid_size_as_buying_pressure() -> None:
    df = make_book([99.0, 99.0], [101.0, 101.0], [1.0, 4.0], [1.0, 1.0])
    assert value("ofi_1", df) == pytest.approx(3.0)


def test_ofi_treats_an_improved_bid_as_the_whole_new_queue() -> None:
    """A better price is new interest, not an increment on the old queue."""
    df = make_book([99.0, 99.5], [101.0, 101.0], [1.0, 2.0], [1.0, 1.0])
    assert value("ofi_1", df) == pytest.approx(2.0)


def test_ofi_treats_a_receding_bid_as_the_old_queue_leaving() -> None:
    df = make_book([99.0, 98.5], [101.0, 101.0], [5.0, 2.0], [1.0, 1.0])
    assert value("ofi_1", df) == pytest.approx(-5.0)


def test_ofi_flips_sign_on_the_ask_side() -> None:
    """Added ask size is selling pressure, so the sign is negative."""
    df = make_book([99.0, 99.0], [101.0, 101.0], [1.0, 1.0], [1.0, 4.0])
    assert value("ofi_1", df) == pytest.approx(-3.0)


def test_ofi_is_antisymmetric_under_swapping_the_sides() -> None:
    normal = make_book([99.0, 99.0], [101.0, 101.0], [1.0, 5.0], [2.0, 2.0])
    mirrored = make_book([99.0, 99.0], [101.0, 101.0], [2.0, 2.0], [1.0, 5.0])
    assert value("ofi_1", normal) == pytest.approx(-value("ofi_1", mirrored))


# ---------------------------------------------------------------------------
# Scale invariance — the property that makes a transfer experiment meaningful
# ---------------------------------------------------------------------------


def rescaled(df: pd.DataFrame, *, price: float, size: float) -> pd.DataFrame:
    """The same market, quoted in different units."""
    out = df.copy()
    for col in (bid_price_col(0), ask_price_col(0)):
        out[col] = out[col] * price
    for col in (bid_size_col(0), ask_size_col(0)):
        out[col] = out[col] * size
    return out


@pytest.fixture
def varied_book(book: pd.DataFrame) -> pd.DataFrame:
    """A depth-1 view of the synthetic book, long enough for every window."""
    return book.head(2000).copy()


@pytest.mark.parametrize("name", SCALE_FREE)
def test_scale_free_features_survive_a_change_of_units(
    name: str, varied_book: pd.DataFrame
) -> None:
    """BTCUSDT and XRPUSDT differ by orders of magnitude in both price and size.

    A feature that moves when the units do cannot transfer between them, and a
    transfer experiment using it would be measuring arithmetic.
    """
    original = REGISTRY.get(name)(varied_book).to_numpy()
    scaled = REGISTRY.get(name)(rescaled(varied_book, price=1e-5, size=1e5)).to_numpy()

    both_nan = np.isnan(original) & np.isnan(scaled)
    close = np.isclose(original, scaled, rtol=1e-6, atol=1e-9)
    assert np.all(close | both_nan)


@pytest.mark.parametrize("name", UNIT_DEPENDENT)
def test_unit_dependent_features_are_known_to_be_so(name: str, varied_book: pd.DataFrame) -> None:
    """Pins the exception rather than leaving it implicit.

    These carry real information and are kept, but a cross-instrument model
    must not use them raw. If one is ever made scale-free, this test fails and
    the decision gets revisited deliberately.
    """
    original = REGISTRY.get(name)(varied_book).to_numpy()
    scaled = REGISTRY.get(name)(rescaled(varied_book, price=1.0, size=1e5)).to_numpy()

    comparable = ~np.isnan(original) & ~np.isnan(scaled)
    assert comparable.any()
    assert not np.allclose(original[comparable], scaled[comparable], rtol=1e-6)


def test_normalised_ofi_has_a_usable_scale(varied_book: pd.DataFrame) -> None:
    """Raw OFI ranges over five orders of magnitude between instruments."""
    values = REGISTRY.get("ofi_20_norm")(varied_book).dropna()
    assert len(values) > 100
    assert values.abs().median() < 10.0


def test_every_registered_book_feature_is_classified() -> None:
    """Nothing may be added without deciding whether it transfers."""
    assert set(SCALE_FREE) | set(UNIT_DEPENDENT) == set(REGISTRY.names("book"))


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def test_warmup_rows_are_nan_not_guessed(book: pd.DataFrame) -> None:
    """Filling warm-up invents data exactly where a model would believe it."""
    values = REGISTRY.get("realized_vol_50_bp")(book)
    assert values.iloc[:50].isna().all()
    assert values.iloc[60:].notna().any()


def test_build_output_matches_individual_features(book: pd.DataFrame) -> None:
    frame = build(book, ["spread_bp", "ofi_20_norm"], plane="book")
    for name in ("spread_bp", "ofi_20_norm"):
        expected = REGISTRY.get(name)(book)
        pd.testing.assert_series_equal(frame[name], expected, check_names=False)
