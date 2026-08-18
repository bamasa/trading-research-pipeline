"""§9 — how low the fee would have to go.

The strategy loses by a roughly fixed amount per trade, so the obvious question
is what fee would make it break even. This sweeps the taker fee and reports the
net result, using the per-fold walk-forward output rather than re-fitting: the
fee enters only through the cost model, and the trades a model takes do not
change when the fee does.

That last point is the reason this is a sweep and not a rerun, and it is also
its main limitation — a strategy facing a lower fee *should* trade differently,
so the figures here are optimistic for the same reason they are cheap.

Needs a walk-forward run first:

    uv run python -m experiments.walk_forward --model logistic --symbols BTCUSDT
    uv run python -m experiments.fee_sensitivity
"""

from __future__ import annotations

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser

#: Published Binance USD-M taker rates, standard tier down to the top VIP tier.
FEES_BP = (5.0, 4.0, 3.0, 2.25, 1.7, 1.0, 0.5, 0.0)


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    args = p.parse_args()

    path = RESULTS / f"walk_forward_{args.model}_folds.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n"
            f"  uv run python -m experiments.walk_forward --model {args.model}"
        )
    folds = pd.read_csv(path)

    rows = []
    for fee in FEES_BP:
        # Gross is untouched by the fee; only the charge per trade moves. Two
        # sides, plus the spread and slippage already inside the recorded cost.
        delta = 2 * (COSTS.fee_bp_per_side - fee)
        adjusted = folds.assign(
            net_per_trade_bp=folds["net_per_trade_bp"] + delta,
            net_bp=folds["net_bp"] + delta * folds["trades"],
        )
        by_setting = adjusted.groupby(["symbol", "model", "cooldown"]).agg(
            net_per_trade_bp=("net_per_trade_bp", "mean"),
            net_bp=("net_bp", "sum"),
        )
        best = by_setting["net_per_trade_bp"].idxmax()
        rows.append(
            {
                "fee_bp_per_side": fee,
                "best_setting": f"{best[1]} cd={best[2]}",
                "net_per_trade_bp": float(by_setting["net_per_trade_bp"].max()),
                "folds_positive": int((adjusted["net_bp"] > 0).sum()),
                "folds": len(adjusted),
            }
        )

    emit(pd.DataFrame(rows), f"fee_sensitivity_{args.model}")


if __name__ == "__main__":
    main()
