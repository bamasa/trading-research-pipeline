"""A kill switch on the strategy's own profit and loss, and what to do after it fires.

Reads the hourly portfolio equity in ``data.json`` (no archive needed) and
runs Page's CUSUM, the textbook sequential test for a drop in a mean, on the
portfolio's hourly net return. Every value was fixed from the textbook before
it was run: the reference value ``k`` = 0.5 and the threshold ``h`` = 5, in
units of the hourly return's standard deviation (a test tuned to a one-sigma
shift, about 930 hours between false alarms on Gaussian noise), estimated on
the hours before the one being tested; the first 72 hours only estimate it.
At each hour the monitor reads nothing after that hour.

Two policies after an alarm:

- ``stop``: the strategy stops and stays stopped.
- ``shadow``: the strategy stops but keeps trading on paper, and goes live
  again when the same CUSUM, pointed upwards, sees the paper result recover;
  the downward test then restarts from zero.

Writes ``monitor.json``; ``tests/test_posts.py`` checks the README against it.

    python posts/btc_ath_2024/monitor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
K, H, WARM_UP = 0.5, 5.0, 72


def hourly_returns() -> pd.Series:
    """The portfolio's net return in each hour, in % of capital; an hour without a trade is 0."""
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    equity = pd.Series({pd.Timestamp(t, tz="UTC"): v for t, v in data["equity_pct"]})
    grid = pd.date_range(equity.index[0], equity.index[-1], freq="1h")
    equity = equity.reindex(grid).ffill()
    return equity.diff().fillna(equity.iloc[0])


def run(returns: pd.Series, restart: bool) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Equity under the kill switch, and every stop and restart with its hour."""
    x = returns.to_numpy(dtype=np.float64)
    live, down, up, equity = True, 0.0, 0.0, 0.0
    path, events = np.empty(len(x)), []
    for i, value in enumerate(x):
        if live:
            equity += value
        path[i] = equity
        if i < WARM_UP:
            continue
        z = value / float(np.std(x[:i], ddof=1))
        if live:
            down = max(0.0, down - z - K)
            if down > H:
                live, down, up = False, 0.0, 0.0
                events.append({"event": "stop", "at": returns.index[i], "equity_pct": equity})
        elif restart:
            up = max(0.0, up + z - K)
            if up > H:
                live, down, up = True, 0.0, 0.0
                events.append({"event": "restart", "at": returns.index[i], "equity_pct": equity})
    return path, events


def main() -> None:
    returns = hourly_returns()
    iso = "%Y-%m-%dT%H:%M"
    raw = returns.cumsum().to_numpy()
    peak_at = returns.index[int(np.argmax(raw))]
    out: dict[str, object] = {"k": K, "h": H, "warm_up_hours": WARM_UP,
                              "hours": [t.strftime(iso) for t in returns.index],
                              "none": np.round(raw, 3).tolist()}  # fmt: skip
    for name, restart in (("stop", False), ("shadow", True)):
        path, events = run(returns, restart)
        out[name] = np.round(path, 3).tolist()
        out[f"{name}_events"] = [
            {"event": e["event"], "at": e["at"].strftime(iso),  # type: ignore[attr-defined]
             "equity_pct": round(float(e["equity_pct"]), 1),  # type: ignore[arg-type]
             "hours_after_peak": round((e["at"] - peak_at).total_seconds() / 3600)}  # type: ignore[operator]
            for e in events
        ]  # fmt: skip
        out[f"{name}_end_pct"] = round(float(path[-1]), 1)
    (HERE / "monitor.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print({k: v for k, v in out.items() if k.endswith(("events", "end_pct"))})


if __name__ == "__main__":
    main()
