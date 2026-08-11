"""Tests for the synthetic market.

Two of these matter more than the rest.

``test_observable_flow_predicts_the_next_return`` checks that the injected
signal is there and is *causal*: information visible at time *t* predicts the
return from *t* to *t+1*.

``test_no_signal_means_no_predictability`` checks the other direction. With the
signal switched off, the same measurement must come back at zero. Together they
give the test suite a dataset with a known answer, which is what later lets a
leakage test distinguish "the model found the signal" from "the model saw the
future". If the generator itself leaked, every such test would pass for the
wrong reason — so the generator is held to the standard it is used to enforce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobml.data.schema import (
    TRADE_SCHEMA,
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    book_schema,
    mid_price,
)
from lobml.data.synthetic import SyntheticConfig, generate, replace_config
from lobml.data.validate import validate_book, validate_trades

# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_generated_planes_satisfy_their_contracts(dataset) -> None:
    book_schema(dataset.config.depth).validate(dataset.book)
    TRADE_SCHEMA.validate(dataset.trades)


def test_generated_planes_pass_validation(dataset) -> None:
    assert validate_book(dataset.book).ok
    assert validate_trades(dataset.trades).ok


def test_every_row_is_marked_synthetic(dataset) -> None:
    """A synthetic result must never be mistakable for a real one."""
    assert (dataset.book["source"] == "synthetic").all()
    assert (dataset.trades["source"] == "synthetic").all()
    for manifest in dataset.manifests.values():
        assert manifest.source == "synthetic"
        assert "Synthetic" in str(manifest.extra["note"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_an_identical_dataset(config: SyntheticConfig) -> None:
    """A demo that changes between runs cannot verify a refactor."""
    first = generate(config)
    second = generate(config)
    pd.testing.assert_frame_equal(first.book, second.book)
    pd.testing.assert_frame_equal(first.trades, second.trades)


def test_different_seeds_give_different_data(config: SyntheticConfig) -> None:
    other = generate(replace_config(config, seed=config.seed + 1))
    base = generate(config)
    assert not np.allclose(other.latent["mid_true"], base.latent["mid_true"])


# ---------------------------------------------------------------------------
# The known answer
# ---------------------------------------------------------------------------


def _queue_imbalance(book: pd.DataFrame) -> np.ndarray:
    bid = book[bid_size_col(0)].to_numpy()
    ask = book[ask_size_col(0)].to_numpy()
    return (bid - ask) / (bid + ask)


def _forward_return(book: pd.DataFrame) -> np.ndarray:
    mid = mid_price(book).to_numpy()
    forward = np.full_like(mid, np.nan)
    forward[:-1] = np.diff(mid) / mid[:-1]
    return forward


def _flow_return_correlation(book: pd.DataFrame) -> float:
    """Correlation between imbalance observable now and the very next return."""
    imbalance = _queue_imbalance(book)
    forward = _forward_return(book)
    ok = ~np.isnan(forward)
    return float(np.corrcoef(imbalance[ok], forward[ok])[0, 1])


def test_observable_flow_predicts_the_next_return(config: SyntheticConfig) -> None:
    """The injected edge is real and is visible from observable state alone."""
    dataset = generate(replace_config(config, n_steps=60_000, signal_strength=1.5))
    assert _flow_return_correlation(dataset.book) > 0.02


def test_no_signal_means_no_predictability(config: SyntheticConfig) -> None:
    """With the edge switched off, the same measurement must vanish.

    This is the control that makes the leakage tests meaningful: a pipeline
    reporting an edge on this dataset is reporting a leak.
    """
    dataset = generate(replace_config(config, n_steps=60_000, signal_strength=0.0))
    assert abs(_flow_return_correlation(dataset.book)) < 0.02


def test_the_signal_leads_the_price_and_does_not_lag_it(config: SyntheticConfig) -> None:
    """The edge must point forward in time, not backward.

    Measured with a memoryless signal (``signal_ar=0``) on purpose. With the
    default persistence of 0.97, ``signal[t]`` and ``signal[t-1]`` are nearly
    the same number, so a leaking generator and a causal one would look
    identical here. Stripping the persistence separates them: the signal
    observable at *t* must predict the return *after* t, and must say nothing
    about the return that already happened.

    If this test ever fails, the injected edge is partly hindsight and every
    downstream leakage test built on this generator is worthless.
    """
    dataset = generate(replace_config(config, n_steps=60_000, signal_ar=0.0, signal_strength=3.0))
    signal = dataset.latent["signal"].to_numpy()
    mid = dataset.latent["mid_true"].to_numpy()

    returns = np.diff(mid) / mid[:-1]
    forward = float(np.corrcoef(signal[:-1], returns)[0, 1])  # signal[t] vs return t → t+1
    backward = float(np.corrcoef(signal[1:], returns)[0, 1])  # signal[t+1] vs return t → t+1

    assert forward > 0.05
    assert abs(backward) < 0.02


def test_latent_state_is_returned_for_diagnostics(dataset) -> None:
    """Tests may see the truth the model may not."""
    latent = dataset.latent
    assert {"mid_true", "volatility", "regime_excited", "signal", "liquidity"} <= set(
        latent.columns
    )
    assert len(latent) == len(dataset.book)


# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------


def test_both_volatility_regimes_occur(config: SyntheticConfig) -> None:
    """Regime stability is a headline claim, so regimes must exist in the data."""
    dataset = generate(replace_config(config, n_steps=80_000))
    excited = dataset.latent["regime_excited"]
    assert excited.any()
    assert not excited.all()


def test_spread_is_wider_in_the_excited_regime(config: SyntheticConfig) -> None:
    dataset = generate(replace_config(config, n_steps=80_000))
    book = dataset.book
    quoted = book[ask_price_col(0)] - book[bid_price_col(0)]
    excited = dataset.latent["regime_excited"].to_numpy()
    assert quoted[excited].mean() > quoted[~excited].mean()


def test_spread_is_always_at_least_one_tick(dataset) -> None:
    book = dataset.book
    quoted = (book[ask_price_col(0)] - book[bid_price_col(0)]).to_numpy()
    assert quoted.min() >= dataset.config.tick_size - 1e-9


def test_prices_sit_on_the_tick_grid(dataset) -> None:
    tick = dataset.config.tick_size
    for level in range(dataset.config.depth):
        for col in (bid_price_col(level), ask_price_col(level)):
            residual = np.abs(np.remainder(dataset.book[col].to_numpy(), tick))
            assert np.all((residual < 1e-6) | (np.abs(residual - tick) < 1e-6))


def test_depth_decays_away_from_the_touch(dataset) -> None:
    book = dataset.book
    touch = book[bid_size_col(0)].mean()
    far = book[bid_size_col(dataset.config.depth - 1)].mean()
    assert touch > far


def test_trades_execute_at_the_touch(dataset) -> None:
    """Aggressive buys pay the ask; aggressive sells hit the bid.

    This is what keeps the two planes consistent: order flow measured from the
    tape and imbalance read off the book describe the same market.
    """
    book = dataset.book.set_index("timestamp")
    trades = dataset.trades
    snapshot = book.reindex(trades["timestamp"], method="ffill")

    aggressive_buy = ~trades["is_buyer_maker"].to_numpy()
    price = trades["price"].to_numpy()
    best_ask = snapshot[ask_price_col(0)].to_numpy()
    best_bid = snapshot[bid_price_col(0)].to_numpy()

    assert np.allclose(price[aggressive_buy], best_ask[aggressive_buy])
    assert np.allclose(price[~aggressive_buy], best_bid[~aggressive_buy])


def test_trade_intensity_rises_with_volatility(config: SyntheticConfig) -> None:
    quiet = generate(replace_config(config, n_steps=40_000, p_calm_to_excited=0.0))
    busy = generate(
        replace_config(config, n_steps=40_000, p_excited_to_calm=0.0, p_calm_to_excited=1.0)
    )
    assert len(busy.trades) > len(quiet.trades)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_steps", 1),
        ("depth", 0),
        ("tick_size", 0.0),
        ("signal_ar", 1.0),
        ("liquidity_ar", -0.1),
        ("base_spread_ticks", 0.5),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        SyntheticConfig(**{field: value})  # type: ignore[arg-type]


def test_unknown_override_is_rejected(config: SyntheticConfig) -> None:
    """A silently ignored typo would change a result with no trace."""
    with pytest.raises(TypeError, match="volatilty"):
        replace_config(config, volatilty=0.1)


def test_zero_trade_intensity_yields_a_valid_empty_frame(config: SyntheticConfig) -> None:
    dataset = generate(
        replace_config(
            config,
            n_steps=200,
            trade_intensity=0.0,
            trade_intensity_vol_sensitivity=0.0,
        )
    )
    assert dataset.trades.empty
    TRADE_SCHEMA.validate(dataset.trades)
