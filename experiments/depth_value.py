"""Do ten levels of book beat one, and how much does a smoothed target flatter a model?

Two questions the Bybit archives finally make answerable.

The first is whether depth helps. Every result in this project was computed on
the touch alone, because Binance publishes nothing else, and the standing
explanation for a weak forecast was that the data was thin. Bybit publishes ten
levels and more, so the explanation can be tested rather than assumed.

The second is the ruler. Order-book models are often scored against a smoothed
target, FI-2010 style, and §21 measured smoothing nearly doubling a coefficient
on its own. Scoring the same model against both shapes of target is the only
way to know how much of a reported coefficient is the model and how much is the
label.

    uv run python -m experiments.depth_value
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.features.depth import build_depth_features
from trading_research.features.normalise import RollingNormaliser
from trading_research.labels.targets import smoothed_move_bp
from trading_research.models.regression import RidgeBaseline

SYMBOLS = ("BTCUSDT", "XRPUSDT", "CRVUSDT", "BICOUSDT")

#: One row is 5 s after subsampling, so this is a two-minute horizon.
HORIZON = 24

#: Smoothing for the FI-2010 style target. Twenty was the best of the sweep in
#: §21; one reproduces the single-point label everything else used.
SMOOTHING = (1, 20)

BOOK_ROOT = Path("data/book")
SUBSAMPLE = 50


def load(symbol: str) -> pd.DataFrame:
    files = sorted((BOOK_ROOT / symbol).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no book data for {symbol}; run trading-research download-book first")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return frame.iloc[::SUBSAMPLE].reset_index(drop=True)


def feature_sets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The touch-only baseline, the same plus depth, and the mid path."""
    mid = ((frame["bid_price_0"] + frame["ask_price_0"]) / 2).astype("float64")
    bid, ask = frame["bid_size_0"].to_numpy(), frame["ask_size_0"].to_numpy()
    spread = ((frame["ask_price_0"] - frame["bid_price_0"]) / mid * 1e4).to_numpy()

    values = mid.to_numpy()
    ret20 = np.full(len(values), np.nan)
    ret20[20:] = np.log(values[20:] / values[:-20]) * 1e4

    touch = pd.DataFrame(
        {"imbalance": (bid - ask) / (bid + ask), "spread_bp": spread, "ret20": ret20}
    )
    deep = pd.concat([touch, build_depth_features(frame)], axis=1)
    return touch, deep, mid


def score(features: pd.DataFrame, target: pd.Series, normaliser: RollingNormaliser) -> dict:
    """Fit on the first 60% and report the correlation on the rest."""
    scaled = normaliser.transform(features, list(features.columns))
    usable = scaled.notna().all(axis=1) & target.notna()
    cut = int(len(features) * 0.6)
    train = usable & (scaled.index < cut)
    test = usable & (scaled.index >= cut)
    if train.sum() < 5000 or test.sum() < 5000:
        return {"ic": float("nan"), "rows": int(test.sum())}

    model = RidgeBaseline(scale_bp=12.0).fit(scaled[train], target[train])
    prediction = model.predict_edge_bp(scaled[test])
    return {
        "ic": float(np.corrcoef(prediction, target[test])[0, 1]),
        "rows": int(test.sum()),
        "target_sd": float(target[test].std()),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    normaliser = RollingNormaliser(window=4000, exclude=frozenset(["imbalance"]))
    rows = []

    for symbol in SYMBOLS:
        frame = load(symbol)
        touch, deep, mid = feature_sets(frame)
        for smoothing in SMOOTHING:
            target = smoothed_move_bp(mid, HORIZON, smoothing=smoothing)
            for name, features in (("touch", touch), ("touch+depth", deep)):
                result = score(features, target, normaliser)
                rows.append(
                    {
                        "symbol": symbol,
                        "features": name,
                        "n_features": len(features.columns),
                        "smoothing": smoothing,
                        **result,
                    }
                )
        print(f"  {symbol} done", flush=True)

    table = pd.DataFrame(rows)
    # Edge is the coefficient times the move it is measured against. Reported
    # from the signed coefficient, not its magnitude: a negative correlation out
    # of sample is a model that is wrong, and calling it edge would require
    # choosing the flip after seeing the test.
    table["edge_bp"] = table["ic"].clip(lower=0) * table["target_sd"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(table, "depth_and_smoothing")


if __name__ == "__main__":
    main()
