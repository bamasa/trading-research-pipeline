"""§2 and §3 — prediction works, decays, and why no horizon fixes it.

Two tables from one pass. The first is the correlation of each feature with the
forward return, per horizon: the evidence that there is real signal here. The
second multiplies that coefficient by the volatility of the move over the same
horizon, which is the expected gross edge per trade.

The second table is the central finding, and it is arithmetic rather than
modelling: signal decays at almost exactly the rate volatility grows, so their
product barely moves while the fee stays fixed. That is why choosing a different
horizon is not the lever, and it bounds what any classifier on this data can do.

    uv run python -m experiments.signal_decay --symbols BTCUSDT
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import GRID_MS, SUBSAMPLE, emit, load_book_features, parser, round_trip

#: The features carrying most of the signal, named in the results document.
FEATURES = ("queue_imbalance", "microprice_dev_bp", "ofi_20_norm")

HORIZONS_S = (5, 10, 30, 60, 300, 1200)


def label(seconds: int) -> str:
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min"


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    decay, edge = [], []

    for symbol in args.symbols:
        # Read from the raw book rather than the prepared files: order-flow
        # imbalance needs bid and ask sizes, which prepare does not carry
        # forward. See load_book_features.
        frame = load_book_features(symbol, FEATURES, days=args.days)
        cost = round_trip(symbol)
        mid = frame["mid"].to_numpy()
        present = list(FEATURES)

        for seconds in HORIZONS_S:
            step = seconds * 1000 // (GRID_MS * SUBSAMPLE)
            if step < 1 or step >= len(mid):
                continue
            forward = np.full(len(mid), np.nan)
            forward[:-step] = np.log(mid[step:] / mid[:-step]) * 1e4
            move = pd.Series(forward, index=frame.index)

            row: dict[str, object] = {"symbol": symbol, "horizon": label(seconds)}
            for name in present:
                row[name] = float(frame[name].corr(move))
            decay.append(row)

            # Headline feature only: the edge identity needs one coefficient,
            # and queue imbalance is the strongest at every horizon.
            ic = abs(float(row[present[0]]))  # type: ignore[arg-type]
            sigma = float(move.std())
            edge.append(
                {
                    "symbol": symbol,
                    "horizon": label(seconds),
                    "ic": ic,
                    "sigma_move_bp": sigma,
                    "edge_bp": ic * sigma,
                    "cost_bp": cost,
                    "edge_over_cost": ic * sigma / cost,
                }
            )

    emit(pd.DataFrame(decay), "signal_decay")
    print()
    emit(pd.DataFrame(edge), "edge_against_cost")


if __name__ == "__main__":
    main()
