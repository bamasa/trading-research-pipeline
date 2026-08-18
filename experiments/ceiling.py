"""§1 — the ceiling, before any model.

Share of moments whose future move exceeds the round-trip cost. This is an upper
bound that already assumes the direction is predicted perfectly, so no model can
beat it and every later number sits underneath it.

Run over the same days for every instrument. An earlier version of this table
used 38 days of BTCUSDT against 59 of XRPUSDT and made XRPUSDT look twice as
tradeable; the difference was the window, not the instrument.

    uv run python experiments/ceiling.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import GRID_MS, SUBSAMPLE, emit, load, parser, round_trip

#: Horizons in rows. One row is 5 s, so these run from 1 s to 10 min. The 1 s
#: figure needs the unsubsampled grid and is computed from the raw column
#: instead — see the note in the results document.
HORIZONS_S = (5, 10, 30, 60, 300, 600)


def label(seconds: int) -> str:
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min"


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    rows = []
    for symbol in args.symbols:
        frame = load(symbol, root=args.prepared_root, days=args.days)
        cost = round_trip(symbol, frame)
        mid = frame["mid"].to_numpy()
        for seconds in HORIZONS_S:
            step = seconds * 1000 // (GRID_MS * SUBSAMPLE)
            if step < 1 or step >= len(mid):
                continue
            move = np.abs(np.log(mid[step:] / mid[:-step]) * 1e4)
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": label(seconds),
                    "seconds": seconds,
                    "cost_bp": cost,
                    "share_clearing_cost": float((move > cost).mean()),
                    "n": len(move),
                }
            )

    table = pd.DataFrame(rows)
    wide = table.pivot_table(index="horizon", columns="symbol", values="share_clearing_cost")
    order = [label(s) for s in HORIZONS_S if label(s) in wide.index]
    emit(wide.loc[order].reset_index(), "ceiling")


if __name__ == "__main__":
    main()
