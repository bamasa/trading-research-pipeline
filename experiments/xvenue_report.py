"""README and figure for an xvenue_funding result (``development.json`` or ``result.json``).

PYTHONPATH=src python -m experiments.xvenue_report experiments/results/xvenue_funding/development.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def figure(r: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "monospace", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=160, width_ratios=[1.6, 1])
    m = r["monthly"]
    x = np.arange(len(m) + 1)
    for key, label, color, width in (
        ("return_pct", "account", "#222222", 2.6),
        ("funding_pct", "funding received", "#2e8b57", 1.6),
        ("price_pct", "price gap between venues", "#1f4e9c", 1.6),
        ("cost_pct", "fees and slippage", "#c0392b", 1.6),
    ):
        left.plot(
            x, np.r_[0.0, np.cumsum([row[key] for row in m])], color=color, lw=width, label=label
        )
    lo, hi = min(r["placebo_total_pct"]), max(r["placebo_total_pct"])
    left.axhspan(lo, hi, color="#e6e6e6", zorder=0, label="19 random directions, final")
    left.axhline(0, color="#444444", lw=0.8)
    months = r["months"]
    ticks = list(range(0, len(months), max(2, len(months) // 6)))
    left.set_xticks([t + 1 for t in ticks], [months[t] for t in ticks], fontsize=8)
    left.set_title(
        f"cumulative, % of the account, {months[0]} to {months[-1]}", loc="left", fontsize=10
    )
    left.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    s = sorted(r["spread_bp_a_day"].items(), key=lambda kv: kv[1]["mean"])
    names = [c.replace("USDT", "") for c, _ in s]
    vals = [v["mean"] for _, v in s]
    y = np.arange(len(s))
    right.barh(y, vals, color=["#d9622b" if v < 0 else "#2e8b57" for v in vals], height=0.6)
    cost = r["cost_a_day_bp"]
    for edge in (-cost, cost):
        right.axvline(edge, color="#222222", lw=1, ls="--")
    right.text(-cost, len(s) - 0.3, " round trip / 30 days ", fontsize=8, ha="right")
    right.axvline(0, color="#444444", lw=0.8)
    right.set_yticks(y, names, fontsize=8)
    right.set_title("Binance minus Bybit funding, bp a day", loc="left", fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    source = Path(sys.argv[1])
    r = json.loads(source.read_text())
    r.setdefault("cost_a_day_bp", 29.0 / 30)
    figure(r, source.parent / "figure.png")
    s, split, mean = r["summary"], r["split_pct"], r["monthly_mean"]
    placebo = r["placebo_total_pct"]
    negative = sum(v["mean"] < 0 for v in r["spread_bp_a_day"].values())
    status = (
        f"\n| | status | |\n|---|---|---|\n| V1, the spread pays | **{r['V1']['status']}** "
        f"| mean {r['V1']['mean']:+.3f}% a month, *t* {r['V1']['t']:+.2f}, *p* {r['V1']['p']:.3f} |\n"
        f"| V2, the direction matters | **{r['V2']['status']}** | beats {r['V2']['beats']} of "
        f"{r['V2']['of']} placebos |\n"
        if "V1" in r
        else ""
    )
    tier = ("**Held-out months, read once.**" if "V1" in r
            else "**A measurement on development months**; no status.")  # fmt: skip
    text = f"""# The funding spread between Binance and Bybit: {r["months"][0]} to {r["months"][-1]}

Written by `experiments/xvenue_report.py` from `{source.name}`
([registration](../../../docs/preregistration/xvenue_funding_2026_11.md)). Each month the rule holds
up to four of fifteen coins' perpetuals on both venues, short where the last three months' funding
was higher and long where it was lower, at taker fees on both. {tier}

![cumulative result and spreads](figure.png)
{status}
| total % | a year % | Sharpe | worst drawdown % | losing months | mean a month % | *t* |
|--:|--:|--:|--:|--:|--:|--:|
| {s["total_pct"]:+.2f} | {s["a_year_pct"]:+.2f} | {s["sharpe_monthly"]:.2f} | {s["worst_drawdown_pct"]:.2f} | {s["losing_months"]} of {s["months"]} | {mean["mean"]:+.3f} | {mean["t"]:+.2f} |

- **Where it came from:** funding {split["funding_pct"]:+.2f}%, the price gap between the venues
  {split["price_pct"]:+.2f}%, fees and slippage {split["cost_pct"]:+.2f}%; {r["months_empty"]} month(s)
  with nothing held.
- **Bybit's funding is the higher one** on {negative} of {len(r["spread_bp_a_day"])} coins on
  average, so the rule is mostly long on Binance and short on Bybit.
- **The direction matters:** the same coins with a random direction each month ended between
  {min(placebo):+.2f}% and {max(placebo):+.2f}%.
"""
    (source.parent / "README.md").write_text(text)
    print(f"wrote {source.parent / 'README.md'}")


if __name__ == "__main__":
    main()
