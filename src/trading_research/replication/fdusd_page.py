"""The replication's results page: what was run, the tables at both fees, and what they show."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from trading_research.replication.fdusd_report import (
    FEE_TITLES,
    days_table,
    pooled_table,
    signal_table,
    windows_table,
)
from trading_research.replication.fdusd_windows import WINDOWS

HEAD = """# Replication: the 2025 BTC/FDUSD result at two fees

**Exploratory, not a blind test.** This is a descriptive replication, on public
data, of the author's 2025 research on Binance spot BTC/FDUSD. That research
covered these dates, so no day here is held out, and nothing
here is a registered result: no registration is changed, and the spot study's
blocks, statuses and verdicts are untouched. Its one question is where the 2025
profit on this pair came from, asked by running the same decisions at two fees.

## What was run

- **Data**: Binance spot BTCFDUSD daily aggTrades from data.binance.vision,
  {n_days} days ({first} to {last} without the days between the two spans), each
  verified against its published `.CHECKSUM` (`replication_fdusd_2025_fetch.csv`)
  and parsed by the spot study's reader.
- **Recipe**: the spot study's S25-PF code path, reused by import: S25's nine
  print members on the 100 ms print grid, the print label at a 0 bp label fee,
  `gbm` at the repository's defaults (seed 42) on every tenth row, hold 50 s
  closed by time, the hold-first rule at 240 trades a day with θ set on the
  window's threshold days, the spot folds' purges (the hold plus 61 minutes,
  and a training label that reads past the threshold days dropped), one clip
  of 0.010 BTC, taker in and taker out by the side-correct print rule, no
  entry before 00:01 or after 23:58 UTC, flat at 23:59.
- **Windows**: A is the author's 2025 split; B is the same recipe on May and
  June 2025.
- **Fees**, from the spot study's registered fee table, whose BTC/FDUSD interval
  from 2024-04-25 06:00 to 2026-01-29 covers every day here: (a) zero maker and
  zero taker, as the author's 2025 backtest charged on this pair; (b) the
  Regular account, 0 maker and 10 bp taker; (c) VIP 2 to 9, 0 maker and 0 taker
  in 2025, which is the same as (a), so (a) and (c) are one table. VIP 2 then
  required at least 5M USD of 30-day spot volume and 100 BNB held.
- **Long-only**: every short decision dropped before the one-position rule.

Net in quote is in FDUSD, a dollar stablecoin, per 0.010 BTC clip. "Pooled" is
the window's net over its entry notional (every trade weighted by its
notional); "mean of days" and "day t" are over the days with a trade and are
descriptive only.
"""


def _at(summary: pd.DataFrame, window: str, variant: str, fees: str, column: str) -> float:
    row = summary[(summary["window"] == window) & (summary["variant"] == variant)
                  & (summary["fees"] == fees)]  # fmt: skip
    return float(row[column].iloc[0]) if len(row) else float("nan")


def shows(summary: pd.DataFrame) -> str:
    """The plain paragraph: at which fee the result is positive, and why."""

    def v(window: str, fees: str, column: str, variant: str = "two_sided") -> float:
        return _at(summary, window, variant, fees, column)

    zero = {w: v(w, "zero", "net_bp_trade") for w in ("A", "B", "A+B")}
    reg = {w: v(w, "regular_10bp", "net_bp_trade") for w in ("A", "B", "A+B")}
    edge = {w: v(w, "zero", "gross_bp_turnover") for w in ("A", "B", "A+B")}
    where = [name for name, found in (("zero fees", zero), ("10 bp", reg)) if found["A+B"] > 0]
    verdict = f"positive at {' and '.join(where)}" if where else "positive at neither fee"
    out = [
        f"Over both windows the result is {verdict}. At zero fees it is {zero['A']:.2f} bp a "
        f"trade on window A and {zero['B']:.2f} bp on B ({zero['A+B']:.2f} bp over both, "
        f"{v('A+B', 'zero', 'net_quote_per_trade'):.3f} FDUSD a 0.010 BTC trade, day t "
        f"{v('A', 'zero', 'day_t'):.2f} and {v('B', 'zero', 'day_t'):.2f}); at the Regular "
        f"account's 10 bp taker fee it is {reg['A']:.2f} and {reg['B']:.2f} bp, with "
        f"{v('A+B', 'regular_10bp', 'share_days_positive'):.0%} of days positive. The trades "
        "are the same in both tables; only the fee differs. A taker "
        "round trip pays the fee on both fills, so it is positive only when its gross edge "
        "per side exceeds the fee per side: that edge is the break-even fee, here "
        f"{edge['A']:.2f} bp of turnover on A and {edge['B']:.2f} bp on B "
        f"({edge['A+B']:.2f} bp over both).",
    ]
    if edge["A+B"] <= 0:
        out.append(
            "It is not above zero, so no fee, zero included, makes this recipe positive on "
            "these days, and the 10 bp taker fee adds about 20 bp of entry notional a round "
            "trip, which is the whole of the loss at the Regular account."
        )
    else:
        out.append(
            f"The recipe is positive at fees below {edge['A+B']:.2f} bp a side, and at 10 bp "
            f"a side it loses about 20 bp less its gross of {v('A+B', 'zero', 'gross_bp_trade'):.2f}."
        )
    out.append(
        f"The model has little to act on: its mean rank IC on the test days is "
        f"{v('A', 'zero', 'ic_mean'):.4f} on A and {v('B', 'zero', 'ic_mean'):.4f} on B, and "
        f"its thresholds, set for 240 decisions a day on the threshold days, give "
        f"{v('A', 'zero', 'decisions_per_day'):.0f} and {v('B', 'zero', 'decisions_per_day'):.0f} "
        f"a day on the test days, of which the one-position rule keeps "
        f"{v('A', 'zero', 'trades_per_day'):.1f} and {v('B', 'zero', 'trades_per_day'):.1f} trades. "
        f"The long-only variant nets {v('A', 'zero', 'net_bp_trade', 'long_only'):.2f} and "
        f"{v('B', 'zero', 'net_bp_trade', 'long_only'):.2f} bp a trade at zero fees, while "
        f"BTC moved {v('A', 'zero', 'market_bp_day_mean'):.0f} and "
        f"{v('B', 'zero', 'market_bp_day_mean'):.0f} bp a test day on average."
    )
    out.append(
        "So on these windows the print-only recipe does not reproduce a profit, even at the "
        "zero fee the 2025 backtest charged. What the two tables do show is the size of the "
        "fee: at the Regular account's fee a taker recipe needs a gross edge above 20 bp a "
        "round trip, several times the gross edges measured on this pair here and on the spot "
        "study's 2026 days (2.65 bp a trade for S25-PF at zero fees on H), so a profit of this "
        "kind on BTC/FDUSD needed the zero fee. S25 itself reads 31 book features, which the "
        "public archive of this pair does not hold, so this is S25-PF's recipe, not S25's."
    )
    return " ".join(out)


def excluded(days: pd.DataFrame) -> str:
    """Which test days the loader excluded (a day without a print file, a print or a tick)."""
    out = []
    for name, window in WINDOWS.items():
        scored = set(days.loc[days["window"] == name, "day"])
        missing = [d.isoformat() for d in window.fold().test if d.isoformat() not in scored]
        out.append(f"{name}: {', '.join(missing) if missing else 'none'}")
    return "Test days excluded by the spot loader's rules: " + "; ".join(out) + "."


def page(
    fits: Sequence[Mapping[str, Any]], summary: pd.DataFrame, days: pd.DataFrame,
    fetched: pd.DataFrame,
) -> str:  # fmt: skip
    text = HEAD.format(n_days=len(fetched), first=fetched["day"].min(), last=fetched["day"].max())
    parts = [text, "## The windows and their fits\n", windows_table(fits), "", excluded(days), ""]
    for fees, title in FEE_TITLES.items():
        parts += [f"## Results at {title}\n", pooled_table(summary, fees), ""]
    parts += ["## Rate and signal (two-sided, the same at both fees)\n", signal_table(summary), ""]
    parts += ["## What it shows\n", shows(summary), ""]
    for window in ("A", "B"):
        parts += [f"## Window {window}, day by day (two-sided)\n", days_table(days, window), ""]
    parts += ["Every run's per-day row, both variants at both fees, with its counters, is in "
              "`replication_fdusd_2025_days.csv`.", ""]  # fmt: skip
    return "\n".join(parts)
