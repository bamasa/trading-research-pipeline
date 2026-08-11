"""Tests for the look-ahead detector.

The detector is only worth having if it fails on real leaks, so most of this
file is deliberately broken features: a centred window, a backward fill, a
whole-sample normalisation. Each is a way look-ahead actually gets into
research code, and each produces output that looks entirely reasonable.

The last group runs the detector over every registered feature. That test is
the one that keeps working after this file stops being read — any feature added
later is checked automatically, with no need to remember to check it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobml.data.schema import mid_price
from lobml.features import book as book_features  # noqa: F401  (registers features)
from lobml.features.registry import Feature, Registry, build, feature
from lobml.validation.leakage import (
    LeakageError,
    assert_causal,
    assert_lookback_honest,
    check_declared_lookback,
    check_feature,
    check_registry,
)


def make_feature(fn, *, name: str = "probe", lookback: int = 0) -> Feature:
    """Wrap a bare function as a Feature without touching the real registry."""
    return Feature(name=name, plane="book", lookback=lookback, fn=fn, description="")


# ---------------------------------------------------------------------------
# Honest features pass
# ---------------------------------------------------------------------------


def test_current_row_only_is_causal(book: pd.DataFrame) -> None:
    probe = make_feature(lambda df: mid_price(df))
    assert not check_feature(probe, book).leaks


def test_backward_rolling_window_is_causal(book: pd.DataFrame) -> None:
    probe = make_feature(lambda df: mid_price(df).rolling(20, min_periods=20).mean(), lookback=20)
    assert not check_feature(probe, book).leaks


def test_lag_is_causal(book: pd.DataFrame) -> None:
    probe = make_feature(lambda df: mid_price(df).shift(5), lookback=5)
    assert not check_feature(probe, book).leaks


def test_expanding_window_is_causal(book: pd.DataFrame) -> None:
    """Expanding windows grow backwards only, so they are fine."""
    probe = make_feature(lambda df: mid_price(df).expanding().mean(), lookback=0)
    assert not check_feature(probe, book).leaks


# ---------------------------------------------------------------------------
# The three real ways look-ahead gets in
# ---------------------------------------------------------------------------


def test_centred_window_is_caught(book: pd.DataFrame) -> None:
    """`center=True` reads forward. The output looks like a moving average."""
    probe = make_feature(
        lambda df: mid_price(df).rolling(21, center=True, min_periods=1).mean(), lookback=21
    )
    result = check_feature(probe, book)
    assert result.leaks
    assert result.n_diverged > 0


def test_backward_fill_is_caught(book: pd.DataFrame) -> None:
    """`bfill` copies a later observation into an earlier row."""

    def leaky(df: pd.DataFrame) -> pd.Series:
        mid = mid_price(df).copy()
        mid.iloc[::7] = np.nan
        return mid.bfill()

    assert check_feature(make_feature(leaky), book).leaks


def test_whole_sample_normalisation_is_caught(book: pd.DataFrame) -> None:
    """The subtle one: no row reads a later row, yet the test set leaks in."""

    def leaky(df: pd.DataFrame) -> pd.Series:
        mid = mid_price(df)
        return (mid - mid.mean()) / mid.std()

    assert check_feature(make_feature(leaky), book).leaks


def test_negative_shift_is_caught(book: pd.DataFrame) -> None:
    """The blatant case, included so the detector is known to catch it."""
    probe = make_feature(lambda df: mid_price(df).shift(-5))
    result = check_feature(probe, book)
    assert result.leaks
    # A five-row peek should first show up five rows before the cut.
    assert result.first_divergence is not None


def test_reported_divergence_points_at_the_cut(book: pd.DataFrame) -> None:
    """The report has to locate the problem, not merely announce it."""
    probe = make_feature(lambda df: mid_price(df).shift(-3))
    result = check_feature(probe, book, cut=0.5)
    boundary = int(len(book) * 0.5)
    assert result.first_divergence == boundary - 3
    assert "3 row(s) before the cut" in result.detail


def test_forward_looking_label_is_caught(book: pd.DataFrame) -> None:
    """A label depends on the future by design; as a feature it must fail."""
    horizon = 10

    def target(df: pd.DataFrame) -> pd.Series:
        mid = mid_price(df)
        return mid.shift(-horizon) / mid - 1.0

    assert check_feature(make_feature(target), book).leaks


# ---------------------------------------------------------------------------
# Declared lookback
# ---------------------------------------------------------------------------


def test_understated_lookback_is_caught(book: pd.DataFrame) -> None:
    """Not a leak in itself — it understates the embargo, which leaks later."""
    probe = make_feature(lambda df: mid_price(df).rolling(50, min_periods=50).mean(), lookback=5)
    result = check_declared_lookback(probe, book)
    assert result.leaks
    assert "understate" not in result.detail  # phrasing lives in the raiser
    assert "needs more history" in result.detail


def test_honest_lookback_passes(book: pd.DataFrame) -> None:
    probe = make_feature(lambda df: mid_price(df).rolling(10, min_periods=10).mean(), lookback=9)
    assert not check_declared_lookback(probe, book).leaks


def test_overstated_lookback_is_allowed(book: pd.DataFrame) -> None:
    """Rounding up is safe and therefore permitted; rounding down is not."""
    probe = make_feature(lambda df: mid_price(df).rolling(10, min_periods=10).mean(), lookback=99)
    assert not check_declared_lookback(probe, book).leaks


# ---------------------------------------------------------------------------
# Every registered feature, automatically
# ---------------------------------------------------------------------------


def test_all_registered_book_features_are_causal(book: pd.DataFrame) -> None:
    """Runs over whatever is registered, including features added later."""
    assert_causal(book, plane="book")


def test_all_registered_book_features_declare_enough_lookback(book: pd.DataFrame) -> None:
    assert_lookback_honest(book, plane="book")


def test_registry_check_covers_every_book_feature(book: pd.DataFrame) -> None:
    from lobml.features.registry import REGISTRY

    results = check_registry(book, plane="book")
    assert {r.name for r in results} == set(REGISTRY.names("book"))
    assert results, "no book features are registered"


def test_assert_causal_names_the_offender() -> None:
    """A failure has to say which feature, or it cannot be acted on."""
    registry = Registry()

    @feature(name="peeker", plane="book", lookback=0, registry=registry)
    def peeker(df: pd.DataFrame) -> pd.Series:
        return mid_price(df).shift(-1)

    frame = _synthetic_book(300)
    with pytest.raises(LeakageError, match="peeker"):
        assert_causal(frame, plane="book", registry=registry)


def test_short_frame_is_rejected_rather_than_silently_passing(book: pd.DataFrame) -> None:
    """A check that cannot run must say so, not report success."""
    probe = make_feature(lambda df: mid_price(df))
    with pytest.raises(ValueError, match="at least 100 rows"):
        check_feature(probe, book.head(50))


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


def test_build_produces_one_column_per_feature(book: pd.DataFrame) -> None:
    frame = build(book, ["spread_bp", "queue_imbalance"], plane="book")
    assert list(frame.columns) == ["timestamp", "spread_bp", "queue_imbalance"]
    assert len(frame) == len(book)


def test_build_drops_warmup_when_asked(book: pd.DataFrame) -> None:
    frame = build(book, ["realized_vol_50_bp"], plane="book", dropna=True)
    assert not frame["realized_vol_50_bp"].isna().any()
    assert len(frame) < len(book)


def test_build_rejects_an_unknown_feature(book: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="unknown feature"):
        build(book, ["no_such_feature"], plane="book")


def test_build_rejects_mixing_planes(book: pd.DataFrame) -> None:
    registry = Registry()

    @feature(name="a", plane="book", lookback=0, registry=registry)
    def a(df: pd.DataFrame) -> pd.Series:
        return mid_price(df)

    @feature(name="b", plane="trades", lookback=0, registry=registry)
    def b(df: pd.DataFrame) -> pd.Series:
        return df["price"]

    with pytest.raises(KeyError, match="several planes"):
        build(book, ["a", "b"], registry=registry)


def test_duplicate_registration_is_rejected() -> None:
    registry = Registry()

    @feature(name="dup", plane="book", lookback=0, registry=registry)
    def first(df: pd.DataFrame) -> pd.Series:
        return mid_price(df)

    with pytest.raises(ValueError, match="already registered"):

        @feature(name="dup", plane="book", lookback=0, registry=registry)
        def second(df: pd.DataFrame) -> pd.Series:
            return mid_price(df)


def test_negative_lookback_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        feature(name="bad", plane="book", lookback=-1, registry=Registry())


def test_misaligned_feature_is_rejected(book: pd.DataFrame) -> None:
    """A feature returning the wrong length would corrupt the frame silently."""
    probe = make_feature(lambda df: mid_price(df).head(10))
    with pytest.raises(ValueError, match="aligned"):
        probe(book)


def test_max_lookback_sizes_the_embargo() -> None:
    from lobml.features.registry import REGISTRY

    assert REGISTRY.max_lookback(["spread_bp", "realized_vol_50_bp"]) == 51


def _synthetic_book(n: int) -> pd.DataFrame:
    """A minimal valid book frame, for tests that need their own registry."""
    from lobml.data.synthetic import SyntheticConfig, generate

    return generate(SyntheticConfig(n_steps=n, depth=1, seed=3)).book
