"""The fold-level leakage test: fitted artefacts compared by sha256 across a mutation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.validation.fold_leakage import (
    FoldLeakError,
    assert_folds_causal,
    check_fold,
    digest,
    mutate_after_cutoff,
)

CUTOFF = pd.Timestamp("2025-01-01 04:00", tz="UTC")


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price_0": mid - 0.05,
            "ask_price_0": mid + 0.05,
            "feature": rng.normal(0.0, 1.0, n),
            "label": rng.choice([-1, 0, 1], size=n),
            "move_bp": rng.normal(0.0, 5.0, n),
        }
    )


def _causal_fit(frame: pd.DataFrame) -> Mapping[str, Any]:
    """A fit that reads only rows at or before the cutoff: weights, θ and a gate level."""
    train = frame[frame["timestamp"] <= CUTOFF]
    x = np.column_stack([np.ones(len(train)), train["feature"].to_numpy()])
    weights, *_ = np.linalg.lstsq(x, train["move_bp"].to_numpy(), rcond=None)
    spread = (train["ask_price_0"] - train["bid_price_0"]) / train["bid_price_0"] * 1e4
    return {
        "weights": weights,
        "theta": (float(train["feature"].quantile(0.9)), float(train["feature"].quantile(0.1))),
        "gate": {"spread_bp": float(spread.quantile(0.8))},
        "checkpoint": int((train["label"] == 1).sum()),
    }


def test_a_causal_fit_passes() -> None:
    result = check_fold(
        _causal_fit, _frame(), CUTOFF, price_columns=["bid_price_0", "ask_price_0"],
        label_columns=["label", "move_bp"], fold="2025-02",
    )  # fmt: skip
    assert not result.leaks
    assert set(result.digests) == {"weights", "theta", "gate", "checkpoint"}
    assert "unchanged" in str(result)
    assert_folds_causal([result])


def test_a_threshold_set_on_every_row_is_caught() -> None:
    def leaky(frame: pd.DataFrame) -> Mapping[str, Any]:
        fitted = dict(_causal_fit(frame))
        spread = (frame["ask_price_0"] - frame["bid_price_0"]) / frame["bid_price_0"] * 1e4
        fitted["gate"] = {"spread_bp": float(spread.quantile(0.8))}
        return fitted

    result = check_fold(leaky, _frame(), CUTOFF, price_columns=["bid_price_0", "ask_price_0"])
    assert result.changed == ("gate",)
    with pytest.raises(FoldLeakError, match="gate"):
        assert_folds_causal([result])


def test_a_label_read_past_the_cutoff_is_caught() -> None:
    def leaky(frame: pd.DataFrame) -> Mapping[str, Any]:
        fitted = dict(_causal_fit(frame))
        fitted["checkpoint"] = int((frame["label"] == 1).sum())
        return fitted

    result = check_fold(
        leaky, _frame(), CUTOFF, price_columns=["bid_price_0", "ask_price_0"],
        label_columns=["label"],
    )  # fmt: skip
    assert result.changed == ("checkpoint",)


def test_the_mutation_touches_every_price_after_the_cutoff_and_nothing_before() -> None:
    frame = _frame()
    out = mutate_after_cutoff(
        frame, CUTOFF, price_columns=["bid_price_0", "ask_price_0"],
        label_columns=["label", "move_bp"],
    )  # fmt: skip
    before = frame["timestamp"] <= CUTOFF
    pd.testing.assert_frame_equal(out[before], frame[before])
    after = ~before
    assert (out.loc[after, "bid_price_0"] != frame.loc[after, "bid_price_0"]).all()
    assert (out.loc[after, "ask_price_0"] > out.loc[after, "bid_price_0"]).all()
    spread = (out["ask_price_0"] - out["bid_price_0"]) / out["bid_price_0"]
    original = (frame["ask_price_0"] - frame["bid_price_0"]) / frame["bid_price_0"]
    assert (spread[after] > original[after]).all()
    assert set(out.loc[after, "label"]) <= {-1, 0, 1}
    assert (out.loc[after, "move_bp"] != frame.loc[after, "move_bp"]).all()
    assert (out["feature"] == frame["feature"]).all()


def test_the_time_can_be_the_index() -> None:
    frame = _frame().set_index("timestamp")
    out = mutate_after_cutoff(frame, CUTOFF, price_columns=["bid_price_0"], time_column=None)
    assert (out.index == frame.index).all()
    assert (out["bid_price_0"] != frame["bid_price_0"]).sum() == (frame.index > CUTOFF).sum()


@dataclass
class _Levels:
    long: float
    short: float


def test_digests_are_stable_and_sensitive() -> None:
    array = np.arange(6, dtype="float64").reshape(2, 3)
    assert digest(array) == digest(array.copy())
    assert digest(array) != digest(array.astype("float32"))
    assert digest(array) != digest(array.reshape(3, 2))
    assert digest({"b": 1, "a": 2.0}) == digest({"a": 2.0, "b": 1})
    assert digest({"a": 1}) != digest({"a": 1.0})
    assert digest(_Levels(1.0, -1.0)) != digest(_Levels(1.0, -1.5))
    assert digest(_frame()) == digest(_frame())
    assert digest(_frame()) != digest(_frame(seed=1))
    assert digest(["a", "b"]) != digest(["ab"])
    assert digest(None) != digest(0)
    assert len(digest(b"raw")) == 64
