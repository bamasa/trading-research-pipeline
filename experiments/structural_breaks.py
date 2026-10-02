"""Where did BTCUSDT stop behaving as before, and what changed when it did?

Every model in this project is fitted on one stretch of market and expected to
hold until the market changes, and §10 measured what a fixed retraining cadence
is worth: something, and not enough, because a cadence is a guess about how
long conditions last. §25 is what the guess costs when it is wrong. The second
regime detector, ``validation/structural_breaks.py``, lets the data say when it
changed instead — the return series is whitened by its own history and the
Shiryaev-Roberts odds of a change are read per family, scale and dependence,
on the whitened stream.

The question here is the plain one: on BTCUSDT daily bars from January 2024,
where does the monitor flag a break, with which statistic, in which direction,
and how far into its window — the lag being the price of a procedure that
never looks ahead. The daily year is the recorded setting (history 365, online
90); a second row with a shorter memory (250, 60) says how much the answer
depends on the setting.

Two things the table has to carry to mean anything
--------------------------------------------------
**The null count.** The same walk, at the same thresholds, over forty series
built by shuffling the real returns — which keeps their marginal distribution
and destroys every regime — says how many flags the procedure raises on a
series that by construction has no break. A detector is only as good as the
gap between that number and the one on the real series.

**No claim about trading.** Whether refitting at these breaks would have
improved any result is a separate experiment, on the refit-policy axis of
``grand_search.py``, and it is not run here.

The per-step table carries the log returns and the monitor's columns, not the
closes: the figure is drawn as a cumulative return index from 100, which is the
shape of the price without publishing the price.

    uv run python -m experiments.structural_breaks
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.data.ensure import ensure_bars, load_bars
from trading_research.validation.structural_breaks import (
    MonitorSpec,
    detect_breaks,
    log_returns,
)

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START = date(2024, 1, 1)
END = date(2026, 9, 30)

#: The recorded daily setting first; a shorter memory as the sensitivity row.
SETTINGS = {
    "history 365, online 90": MonitorSpec(history_len=365, online_len=90),
    "history 250, online 60": MonitorSpec(history_len=250, online_len=60),
}
NULL_PATHS = 40


def null_count(returns: pd.Series, spec: MonitorSpec, *, paths: int, seed: int = 0) -> dict:
    """Flags raised on shuffled copies of the real returns: the false-alarm reading."""
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    flags, paths_with_flags = 0, 0
    for _ in range(paths):
        result = detect_breaks(rng.permutation(values), spec, on_progress=None)
        flags += len(result.breaks)
        paths_with_flags += bool(result.breaks)
    return {
        "null_paths": paths,
        "null_paths_with_a_flag": paths_with_flags,
        "null_flags_total": flags,
        "null_flags_per_path": flags / paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--null-paths", type=int, default=NULL_PATHS)
    parser.add_argument("--bars-root", type=Path, default=Path("data/bars"))
    args = parser.parse_args()

    ensure_bars(args.symbol, INTERVAL, args.start, args.end, root=args.bars_root)
    bars = load_bars(args.symbol, INTERVAL, root=args.bars_root, start=args.start, end=args.end)
    returns = log_returns(bars.set_index("timestamp")["close"])
    print(
        f"{args.symbol} {INTERVAL}: {len(returns)} returns, "
        f"{returns.index[0]:%Y-%m-%d} → {returns.index[-1]:%Y-%m-%d}"
    )

    tables, summary = [], []
    for label, spec in SETTINGS.items():
        result = detect_breaks(returns, spec, on_progress=print)
        table = result.table()
        table.insert(0, "setting", label)
        table["timestamp"] = pd.to_datetime(table["timestamp"]).dt.strftime("%Y-%m-%d")
        tables.append(table)
        null = null_count(returns, spec, paths=args.null_paths)
        summary.append(
            {
                "setting": label,
                "returns": len(returns),
                "windows": len(result.windows),
                "breaks": len(result.breaks),
                **{f"threshold_{k}": v for k, v in result.thresholds.items()},
                **null,
            }
        )
        print(
            f"  {label}: {len(result.breaks)} break(s), null {null['null_flags_per_path']:.2f}/path"
        )
        if label == next(iter(SETTINGS)):
            steps = result.annotate(returns)
            steps["timestamp"] = pd.to_datetime(steps["timestamp"]).dt.strftime("%Y-%m-%d")
            steps["log_return"] = steps["log_return"].round(8)
            emit(steps.round(4), f"structural_breaks_{args.symbol}")

    emit(pd.concat(tables, ignore_index=True), f"structural_breaks_{args.symbol}_breaks")
    emit(pd.DataFrame(summary), f"structural_breaks_{args.symbol}_null")
    RESULTS.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
