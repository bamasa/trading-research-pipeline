"""The post's kill switch: the pipeline's own, run on the strategy's hourly results.

Reads the hourly portfolio equity in ``data.json`` (no archive needed) and runs
:func:`trading_research.validation.kill_switch.kill_switch`, Page's CUSUM for a
drop in the mean, on the portfolio's hourly net return, at the textbook values
fixed before it was run: ``k`` = 0.5 and ``h`` = 5 in units of the hourly
return's standard deviation, estimated on the hours before the one tested; the
first 72 hours only estimate it. At each hour it reads nothing after that hour.

Two policies after an alarm: ``stop`` (stop and stay stopped) and ``shadow``
(stop, keep trading on paper, and go live again when the same test pointed
upwards sees the paper result recover).

Writes ``monitor.json``; ``tests/test_posts.py`` checks the README against it.

    PYTHONPATH=src python posts/btc_ath_2024/monitor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.validation.kill_switch import TEXTBOOK_H, TEXTBOOK_K, kill_switch

HERE = Path(__file__).parent
K, H, WARM_UP = TEXTBOOK_K, TEXTBOOK_H, 72


def hourly_returns() -> pd.Series:
    """The portfolio's net return in each hour, in % of capital; an hour without a trade is 0."""
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    equity = pd.Series({pd.Timestamp(t, tz="UTC"): v for t, v in data["equity_pct"]})
    grid = pd.date_range(equity.index[0], equity.index[-1], freq="1h")
    equity = equity.reindex(grid).ffill()
    return equity.diff().fillna(equity.iloc[0])


def run(returns: pd.Series, restart: bool) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Equity under the kill switch, and every stop and restart with its hour."""
    result = kill_switch(returns, k=K, h=H, warm_up=WARM_UP, restart=restart)
    events = [{"event": e.kind, "at": e.at, "equity_pct": e.equity} for e in result.events]
    return result.equity.to_numpy(), events


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
