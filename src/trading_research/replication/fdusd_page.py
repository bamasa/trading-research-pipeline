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
profit on this pair came from, asked by running one recipe's decisions at two fees.

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
  and a training label that reads past the threshold days' first stamp
  dropped), one clip of 0.010 BTC, taker in and taker out by the side-correct
  print rule, no entry before 00:01 or after 23:58 UTC, flat at 23:59.
- **Windows**: A combines the 2025 research's v0.5 training and threshold days
  (2024-11-05 to 2025-01-14; 2025-01-15 to 01-21) with its v0.4 test days
  (2025-02-04 to 02-26). The 13 days between (2025-01-22 to 02-03) are fetched
  but unused, so A's fit is 3 to 6 weeks old on its test days: v0.4's fit,
  trained to 2025-02-03, was 1 day to 3 weeks old on those days, and v0.5
  began its test on 2025-01-22, the day after its threshold days. B is the
  same recipe on May and June 2025.
- **Fees**, from the spot study's registered fee table, whose BTC/FDUSD interval
  from 2024-04-25 06:00 to 2026-01-29 covers every day here: (a) zero maker and
  zero taker, as the author's 2025 backtest charged on this pair; (b) the
  Regular account, 0 maker and 10 bp taker; (c) VIP 2 to 9, 0 maker and 0 taker
  in 2025, which is the same as (a), so (a) and (c) are one table. VIP 2 then
  required at least 5M USD of 30-day spot volume and 100 BNB held.
- **Long-only**: every short decision dropped before the one-position rule.

Net in quote is in FDUSD, a dollar stablecoin, per 0.010 BTC clip. "Pooled" is
the window's net over its entry notional (every trade weighted by its
notional); "days positive", "mean of days" and "day t" are over the days with
a trade, not over every test day, and are descriptive only ({example}).
"""

DIFFERS = """## How this recipe differs from the 2025 research

S25-PF is this repository's 2026 registered configuration
(`docs/preregistration/spot_2026.md`): S25's coordinates with its nine print
inputs, because the public archive of this pair holds prints and no book. It is
not the 2025 research's recipe. The 2025 column is as that research's code and
printed outputs record it.

| | The 2025 research | S25-PF, here |
|---|---|---|
| Inputs | about 150-160 book and trade features selected from 357-359 | 9 print members |
| Label | entry at the 1 BTC book VWAP, exit after 3-30 ticks (about 0.3-3 s), profit threshold 1 FDUSD (about 1 bp) | 50 s print round trip at 0 bp |
| Model | XGBoost, 300 trees, depth 6, learning rate 0.03, class weights 3 / 0.5 / 3 | `gbm`, 300 trees, depth 4, learning rate 0.05, {weights} |
| Volatility filter | rows with volatility at or below 2e-5 dropped and held | none |
| Thresholds | Optuna for a trade target, best 450 trades a day (v0.5) | the hold-first rule at 240 a day |
| Hold | `close_time` 500, unit unverified | 50 s |
"""


def _at(summary: pd.DataFrame, window: str, variant: str, fees: str, column: str) -> float:
    row = summary[(summary["window"] == window) & (summary["variant"] == variant)
                  & (summary["fees"] == fees)]  # fmt: skip
    return float(row[column].iloc[0]) if len(row) else float("nan")


def weights(fits: Sequence[Mapping[str, Any]]) -> str:
    """The fits' inverse-frequency class weights, as ``gbm`` sets them (rows / (3 x count))."""
    holds, sides = [], []
    for fit in fits:
        rows, classes = int(fit["rows"]), fit["classes"]
        hold = int(classes.get("0", 0))
        if hold:
            share = hold / rows
            holds.append(f"{rows / (3 * hold):.1f} on {fit['window']} ({share:.1%} of its labels)")
        sides += [rows / (3 * int(classes[k])) for k in ("1", "-1") if int(classes.get(k, 0))]
    low, high = (f"{min(sides):.1f}", f"{max(sides):.1f}") if sides else ("n/a", "n/a")
    side = f"about {low}" if low == high else f"{low} to {high}"
    return (f"inverse-frequency weights: the hold class weighted {' and '.join(holds)}, "
            f"long and short {side}")  # fmt: skip


def denominators(summary: pd.DataFrame) -> str:
    """How many days the "days positive" share is taken over, for the pooled two-sided row."""
    row = summary[(summary["window"] == "A+B") & (summary["variant"] == "two_sided")
                  & (summary["fees"] == "zero")]  # fmt: skip
    if not len(row):
        return "the days with a trade are in each window's day table"
    with_trade, days = int(row["days_with_trade"].iloc[0]), int(row["days"].iloc[0])
    positive = round(float(row["share_days_positive"].iloc[0]) * with_trade)
    return (f"A+B two-sided at zero fees: {positive} days positive of the {with_trade} with a "
            f"trade, of {days} test days")  # fmt: skip


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
        "So on these windows S25-PF, which shares the pair, the zero fee and the 0.010 BTC "
        "clip with the 2025 research but not its inputs, label, model, filter or thresholds, "
        f"is {verdict}. What the two tables do show is the size of the "
        "fee: at the Regular account's fee a taker recipe needs a gross edge above 20 bp a "
        "round trip, several times the gross edges measured on this pair here and on the spot "
        "study's 2026 days (2.65 bp a trade for S25-PF at zero fees on H), so a profit of this "
        "kind on BTC/FDUSD needed the zero fee. Whether the 2025 research's own recipe would "
        "have made money on these days is not asked here: S25, the configuration S25-PF "
        "varies, reads 31 book features, and the 2025 research read book features and a book "
        "label, none of which the public archive of this pair holds."
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
    text = HEAD.format(n_days=len(fetched), first=fetched["day"].min(), last=fetched["day"].max(),
                       example=denominators(summary))  # fmt: skip
    parts = [text, DIFFERS.format(weights=weights(fits)), "## The windows and their fits\n",
             windows_table(fits), "", excluded(days), ""]  # fmt: skip
    for fees, title in FEE_TITLES.items():
        parts += [f"## Results at {title}\n", pooled_table(summary, fees), ""]
    parts += ["## Rate and signal (two-sided, the same at both fees)\n", signal_table(summary), ""]
    parts += ["## What it shows\n", shows(summary), ""]
    for window in ("A", "B"):
        parts += [f"## Window {window}, day by day (two-sided)\n", days_table(days, window), ""]
    parts += ["Every run's per-day row, both variants at both fees, with its counters, is in "
              "`replication_fdusd_2025_days.csv`.", ""]  # fmt: skip
    return "\n".join(parts)
