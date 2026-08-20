"""How good would the forecast have to be, and how far is that from here?

Every negative result in this project has the same shape: the model predicts
direction slightly better than chance, and the cost of acting is several times
what the prediction is worth. That invites a question the results tables do not
answer — is this a gap that better modelling could close, or one that no
plausible model closes?

It is answerable, and precisely, because the requirement can be measured
without building the better model.

The method
----------
Take the model's actual predictions on the held-out block. Blend them with the
realised future:

    blended(w) = (1 - w) * model_edge + w * realised_move

At w = 0 this is the model as it is. At w = 1 it is an oracle. In between it is
a forecast of exactly the quality the blend implies, applied through the same
trading rule, paying the same costs. Sweeping w traces net profit against
forecast skill, and the crossing point is the answer: the information
coefficient a strategy would need before this instrument, at this horizon, on
this venue, pays for itself.

The oracle is a measuring instrument, not a strategy. It is not achievable and
is not claimed to be; blending with the future is simply the cleanest way to
manufacture a forecast of known quality so the requirement can be read off.

Two references for what the answer means
----------------------------------------
* **What this project achieves.** Queue imbalance against the two-minute move
  correlates at 0.02 to 0.15 depending on instrument (§20). The fitted models
  do not beat that by much — §14 found the best gross edge belonged to a rule
  with no parameters.
* **What the literature achieves.** Deep order-book models — DeepLOB and its
  successors — report accuracies that sound far higher, but §21 showed why: the
  standard FI-2010 label is smoothed two-sided, so it correlates 0.586 with a
  quantity already known before the decision. Measured against a clean forward
  label, published short-horizon skill is in the same range this project
  reaches, not an order of magnitude above it.

So the number this script produces is directly comparable to the best published
work, and that comparison is the point.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.grand_search import (
    COSTS,
    ROWS_PER_DAY,
    Config,
    Data,
    build_features,
    normalise,
    to_grid,
)
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.pipeline.stages import build_model

#: Skill levels swept. Fine near zero, because that is where the answer lives.
BLENDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4, 0.6, 1.0)

#: Trades a day the rule is asked to take. Selectivity is part of the question:
#: a weak forecast used sparingly can beat a weak forecast used constantly.
RATES = (1.0, 5.0, 20.0, 100.0)

SIGNAL = Config(
    plane="micro",
    model="logistic",
    target="direction",
    horizon=24,
    hold=24,
    cooldown=24,
    exit="clock",
    objective="net_per_trade_bp",
    refit="2d",
    train_days=10,
    gate="open",
)


def predictions(data: Data, config: Config, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward model scores over a block, and the realised move beside them.

    The score is the probability of an up move minus the probability of a down
    move — a continuous quantity, so it can be blended and correlated rather
    than only thresholded.
    """
    features = data.frames[config.norm_window]
    columns = data.plane[config.plane]
    target = data.target(config.target, config.horizon)
    forward = data.forward(config.hold)
    usable = features[columns].notna().all(axis=1).to_numpy()
    labelled = usable & target.notna().to_numpy()
    y = target.to_numpy()

    scores = np.full(stop - start, np.nan)
    stride = int(float(config.refit.removesuffix("d")) * ROWS_PER_DAY)
    for point in range(start, stop, stride):
        train_from = max(0, point - int(config.train_days * ROWS_PER_DAY))
        train_to = point - max(config.horizon, config.hold)
        mask = np.zeros(len(features), dtype=bool)
        mask[train_from:train_to] = True
        mask &= labelled
        index = np.flatnonzero(mask)
        if len(index) < 10_000 or len(np.unique(y[index])) < 2:
            continue
        model = build_model(config.model)
        model.fit(
            features.iloc[index[-120_000:]][columns],
            pd.Series(y[index[-120_000:]], index=index[-120_000:]),
        )
        block = np.arange(point, min(stop, point + stride))
        ok = usable[block]
        if ok.sum() < 100:
            continue
        proba = model.predict_proba(features.iloc[block[ok]][columns])
        scores[block[ok] - start] = proba[:, 2] - proba[:, 0]
    return scores, forward[start:stop]


def trade(
    score: np.ndarray,
    forward: np.ndarray,
    spread_bp: np.ndarray,
    rate_per_day: float,
    config: Config,
) -> dict[str, float]:
    """Take the strongest signals at the requested rate, and count the cost."""
    ok = np.isfinite(score) & np.isfinite(forward)
    if ok.sum() < 1_000:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}
    days = len(score) / ROWS_PER_DAY
    wanted = max(1, int(rate_per_day * days))
    share = min(0.999, wanted / int(ok.sum()))
    threshold = float(np.quantile(np.abs(score[ok]), 1 - share))

    decision = np.zeros(len(score), dtype=int)
    strong = ok & (np.abs(score) >= threshold)
    decision[strong] = np.sign(score[strong]).astype(int)

    trades = thin(
        decision,
        np.nan_to_num(forward),
        spread_bp,
        ThinningRules(hold_periods=config.hold, cooldown_periods=config.cooldown),
    )
    if not trades:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}
    net = np.array(
        [t.direction * t.move_bp - float(COSTS.round_trip_bp(t.entry_spread_bp)) for t in trades]
    )
    gross = np.array([t.direction * t.move_bp for t in trades])
    return {
        "trades": float(len(net)),
        "gross_per_trade_bp": float(gross.mean()),
        "net_per_trade_bp": float(net.mean()),
        "net_bp": float(net.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BICOUSDT")
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    files = sorted((args.book / args.symbol).glob("*.parquet"))
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread_bp = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()
    frame = normalise(build_features(book), window=4000).astype("float32")
    frame["mid"] = mid
    cut = int(len(book) * 0.65)
    data = Data({4000: frame}, spread_bp, mid, search_end=cut)

    print(f"{args.symbol}: {len(book):,} rows; scoring the held-out block")
    score, forward = predictions(data, SIGNAL, cut, len(book))
    block_spread = spread_bp[cut:]
    ok = np.isfinite(score) & np.isfinite(forward)
    print(
        f"  {ok.sum():,} scored rows, model IC = {np.corrcoef(score[ok], forward[ok])[0, 1]:+.4f}"
    )
    print(
        f"  cost of a round trip here: {COSTS.round_trip_bp(float(np.median(block_spread))):.2f} bp"
    )

    # The realised move, standardised so the blend weight means the same thing
    # whatever the instrument's scale.
    truth = np.zeros_like(score)
    truth[ok] = (forward[ok] - forward[ok].mean()) / forward[ok].std()
    standard_score = np.zeros_like(score)
    standard_score[ok] = (score[ok] - score[ok].mean()) / score[ok].std()

    rows = []
    for weight in BLENDS:
        blended = np.full_like(score, np.nan)
        blended[ok] = (1 - weight) * standard_score[ok] + weight * truth[ok]
        ic = float(np.corrcoef(blended[ok], forward[ok])[0, 1])
        for rate in RATES:
            result = trade(blended, forward, block_spread, rate, SIGNAL)
            rows.append(
                {
                    "blend_weight": weight,
                    "information_coefficient": ic,
                    "trades_per_day": rate,
                    **result,
                }
            )

    table = pd.DataFrame(rows)
    table.insert(0, "symbol", args.symbol)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / f"how_much_better_{args.symbol}.csv", index=False)
    emit(
        table[
            [
                "blend_weight",
                "information_coefficient",
                "trades_per_day",
                "trades",
                "gross_per_trade_bp",
                "net_per_trade_bp",
            ]
        ],
        f"how_much_better_{args.symbol}_summary",
    )

    # Where does it cross zero, per trade rate?
    crossings = []
    for rate, group in table.groupby("trades_per_day"):
        ordered = group.sort_values("information_coefficient")
        positive = ordered[ordered["net_per_trade_bp"] > 0]
        current = float(ordered["information_coefficient"].iloc[0])
        needed = (
            float(positive["information_coefficient"].iloc[0]) if len(positive) else float("nan")
        )
        crossings.append(
            {
                "trades_per_day": rate,
                "current_ic": current,
                "ic_needed_to_break_even": needed,
                "multiple_required": needed / current
                if current and np.isfinite(needed)
                else float("nan"),
            }
        )
    emit(pd.DataFrame(crossings), f"how_much_better_{args.symbol}_crossing")


if __name__ == "__main__":
    main()
