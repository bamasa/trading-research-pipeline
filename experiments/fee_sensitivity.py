"""§9 — how low the fee would have to go.

The strategy loses by a roughly fixed amount per trade, so the obvious question
is what fee would make it break even. This sweeps the taker fee and reports the
net result, using the per-fold walk-forward output rather than re-fitting: the
fee enters only through the cost model, and the trades a model takes do not
change when the fee does.

That last point is the reason this is a sweep and not a rerun, and it is also
its main limitation — a strategy facing a lower fee *should* trade differently,
so the figures here are optimistic for the same reason they are cheap.

Reads whatever walk-forward runs are present, so run those first:

    for m in logistic xgboost tcn; do
        uv run python -m experiments.walk_forward --model "$m" --symbols BTCUSDT
    done
    uv run python -m experiments.fee_sensitivity
"""

from __future__ import annotations

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser

#: Published Binance USD-M taker rates, standard tier down to the top VIP tier.
FEES_BP = (5.0, 4.0, 3.0, 2.25, 1.7, 1.0, 0.5, 0.0)


def break_even_fee(gross_per_trade_bp: float, spread_and_slippage_bp: float) -> float | None:
    """Fee per side at which a configuration would stop losing money.

    Costs are linear in the fee, so this needs no rerun: whatever the gross edge
    is, the fee that exactly consumes it follows. ``None`` means no fee works —
    the gross edge does not even cover the spread and slippage, so the trade
    loses money at a fee of zero.
    """
    headroom = gross_per_trade_bp - spread_and_slippage_bp
    return headroom / 2.0 if headroom > 0 else None


def main() -> None:
    parser(__doc__ or "").parse_args()

    folds = []
    for path in sorted(RESULTS.glob("walk_forward_*_folds.csv")):
        folds.append(pd.read_csv(path))
    if not folds:
        raise SystemExit(
            "no walk-forward runs found. Run:\n"
            "  uv run python -m experiments.walk_forward --model logistic"
        )
    folds = pd.concat(folds, ignore_index=True)

    # What the recorded cost contains besides the fee: the spread crossed once
    # and slippage on both sides. That part does not move with the tier, which
    # is why a zero fee still does not make every configuration profitable.
    by_setting = folds.groupby(["symbol", "model", "cooldown"]).agg(
        trades=("trades", "sum"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
    )
    fixed_cost = (
        by_setting["gross_per_trade_bp"]
        - by_setting["net_per_trade_bp"]
        - (2 * COSTS.fee_bp_per_side)
    )

    sweep = by_setting.copy()
    for fee in FEES_BP:
        delta = 2 * (COSTS.fee_bp_per_side - fee)
        sweep[f"net_at_{fee:g}bp"] = by_setting["net_per_trade_bp"] + delta
    sweep["spread_and_slippage_bp"] = fixed_cost
    sweep["break_even_fee_bp_per_side"] = [
        break_even_fee(g, c)
        for g, c in zip(by_setting["gross_per_trade_bp"], fixed_cost, strict=True)
    ]

    emit(sweep.reset_index(), "fee_sensitivity")


if __name__ == "__main__":
    main()
