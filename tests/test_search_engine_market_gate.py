"""The market gate's inputs: bp from float64 book prices on row stamps (K-units)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_research.features.normalise import NORMALISED_ATTR, RollingNormaliser
from trading_research.search_engine.market_gate import (
    QUIET_OUT_QUANTILE,
    SPREAD_COLUMN,
    TIGHT_ONLY_QUANTILE,
    VOLATILITY_COLUMN,
    GateLevels,
    fit_gate_levels,
    gate_inputs,
    gate_states,
    market_gate,
    quoted_spread_bp,
    realised_volatility_bp,
)
from trading_research.strategies.ev_gate import UnitsError

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"


def _book(rows: int = 3000, seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = rng.integers(95, 106, rows).astype(np.int64) * 1_000_000
    step[rows // 2] = 90_000_000_000  # a feed gap longer than the window
    stamps = pd.Timestamp("2024-02-11", tz="UTC").value + np.cumsum(step)
    mid = 50_000.0 * np.exp(np.cumsum(rng.normal(0, 2e-5, rows)))
    half = rng.integers(1, 4, rows) * 0.05
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps, utc=True),
            "bid_price_0": (mid - half) * scale,
            "ask_price_0": (mid + half) * scale,
        }
    )


def _brute_volatility(stamps: np.ndarray, mid: np.ndarray, window_s: float) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    for t in range(1, len(mid)):
        rows = [j for j in range(1, t + 1) if stamps[j] > stamps[t] - window_s * 1e9]
        out[t] = np.sqrt(sum(np.log(mid[j] / mid[j - 1]) ** 2 for j in rows)) * 1e4
    return out


def test_volatility_is_the_trailing_sum_of_squared_log_changes() -> None:
    book = _book(400)
    stamps = book["timestamp"].astype("int64").to_numpy()
    mid = ((book.bid_price_0 + book.ask_price_0) / 2).to_numpy()
    ours = realised_volatility_bp(stamps, mid, 2.0)
    np.testing.assert_allclose(ours, _brute_volatility(stamps, mid, 2.0), rtol=1e-9, atol=1e-12)
    assert np.isnan(ours[0]) and np.isfinite(ours[1:]).all()


def test_the_spread_is_in_bp_of_mid() -> None:
    assert quoted_spread_bp(np.array([100.0]), np.array([100.01]))[0] == pytest.approx(
        0.01 / 100.005 * 1e4
    )


def test_k_units_inputs_do_not_depend_on_the_price_unit() -> None:
    plain, scaled = gate_inputs(_book()), gate_inputs(_book(scale=1000.0))
    np.testing.assert_allclose(scaled.to_numpy(), plain.to_numpy(), rtol=1e-9)
    assert plain[SPREAD_COLUMN].between(0.01, 0.2).all()  # a few ticks of 50,000, in bp


def test_k_units_refuses_anything_but_float64_book_prices() -> None:
    book = _book()
    with pytest.raises(UnitsError, match="float64"):
        gate_inputs(book.astype({"bid_price_0": np.float32}))
    normalised = RollingNormaliser(window=50).transform(book[["bid_price_0", "ask_price_0"]])
    normalised["timestamp"] = book["timestamp"]
    with pytest.raises(UnitsError, match="normalised"):
        gate_inputs(normalised)
    broken = book.copy()
    broken.loc[5, "ask_price_0"] = 0.0
    with pytest.raises(UnitsError, match="non-positive"):
        gate_inputs(broken)
    tagged = gate_inputs(book)
    tagged.attrs[NORMALISED_ATTR] = [VOLATILITY_COLUMN]
    with pytest.raises(UnitsError, match="normalised"):
        fit_gate_levels(tagged)
    with pytest.raises(UnitsError, match="normalised"):
        gate_states(tagged, GateLevels(0.1, 0.1))


def test_inputs_read_no_later_row() -> None:
    book = _book()
    full = gate_inputs(book)
    cut = gate_inputs(book.iloc[:1700])
    pd.testing.assert_frame_equal(cut, full.iloc[:1700])


def test_levels_are_training_quantiles_and_the_rules_hold() -> None:
    train, test = gate_inputs(_book(seed=1)), gate_inputs(_book(seed=2))
    levels = fit_gate_levels(train)
    assert levels.min_volatility_bp == pytest.approx(
        train[VOLATILITY_COLUMN].dropna().quantile(QUIET_OUT_QUANTILE)
    )
    assert levels.max_spread_bp == pytest.approx(train[SPREAD_COLUMN].quantile(0.7))
    states = gate_states(test, levels)
    assert states["open"].all()
    vol, spread = test[VOLATILITY_COLUMN].to_numpy(), test[SPREAD_COLUMN].to_numpy()
    np.testing.assert_array_equal(states["quiet_out"], vol >= levels.min_volatility_bp)
    np.testing.assert_array_equal(states["tight_only"], spread <= levels.max_spread_bp)
    assert not states["quiet_out"][0]  # no volatility yet: the gate stays closed
    assert GateLevels.from_dict(levels.to_dict()) == levels


def test_the_gates_are_the_registered_ones() -> None:
    gate = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["search"]["axes"]["gate"]
    assert gate["values"] == ["open", "quiet_out", "tight_only"]
    assert "30th percentile" in gate["quiet_out"] and "trailing 60 s" in gate["quiet_out"]
    assert "70th percentile" in gate["tight_only"]
    assert (QUIET_OUT_QUANTILE, TIGHT_ONLY_QUANTILE) == (0.3, 0.7)
    assert market_gate("open") is None
    with pytest.raises(ValueError, match="unknown gate"):
        market_gate("wide_only")
