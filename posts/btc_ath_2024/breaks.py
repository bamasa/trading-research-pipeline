"""Where the market's break was: the pipeline's structural-break detector on the index.

The reversion strategy earns when the equal-weighted index's ten-minute moves
partly reverse, that is when the index's ten-minute returns are negatively
dependent on the previous one. :func:`trading_research.validation.structural_breaks.detect_breaks`
(from the author's solution to the ADIA Lab Structural Break Challenge) has a
family for exactly that, ``dependence``: a change in lag-one dependence on a
stream whitened by its own history. It is run here on the index's
non-overlapping ten-minute returns in ``data.json``, with the detector's
default, null-calibrated setting (a history of 365 returns, 90 streamed before
each refit) and the dependence family only; nothing was searched.

A ``-1`` alarm means the dependence fell (moves reverse more: the strategy's
state strengthening); ``+1`` means it rose (moves stop reversing: the state the
strategy needs fading). Writes ``breaks.json``. Needs the ``breaks`` extra.

    PYTHONPATH=src python posts/btc_ath_2024/breaks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_research.validation.structural_breaks import MonitorSpec, detect_breaks

HERE = Path(__file__).parent
HISTORY, ONLINE, FAMILY = 365, 90, "dependence"


def index_returns() -> pd.Series:
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    return pd.Series({pd.Timestamp(t, tz="UTC"): v for t, v in data["index_10min"]})


def alarms() -> list[dict[str, object]]:
    returns = index_returns()
    spec = MonitorSpec(history_len=HISTORY, online_len=ONLINE, statistics=(FAMILY,))
    table = detect_breaks(returns.reset_index(drop=True), spec, on_progress=None).table()
    return [
        {"at": returns.index[int(row.index)].strftime("%Y-%m-%dT%H:%M"),
         "family": str(row.statistic), "direction": int(row.direction)}
        for row in table.itertuples()
    ]  # fmt: skip


def main() -> None:
    out = {"history": HISTORY, "online": ONLINE, "family": FAMILY, "alarms": alarms()}
    (HERE / "breaks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["alarms"], indent=1))


if __name__ == "__main__":
    main()
