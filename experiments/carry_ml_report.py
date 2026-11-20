"""README and figure for a carry_ml result (``development.json`` or ``result.json``).

PYTHONPATH=src python -m experiments.carry_ml_report experiments/results/carry_ml/development.json
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

NAMES = ("rule_lb3", "rule_lb1", "ridge", "lightgbm", "mlp")
KIND = {"rule_lb3": "rule", "rule_lb1": "rule", "ridge": "ML", "lightgbm": "ML", "mlp": "DL"}
LINE = {
    "rule_lb3": ("#222222", "-", 2.6),
    "rule_lb1": ("#8a8a8a", "-", 1.6),
    "ridge": ("#1f4e9c", "-", 1.8),
    "lightgbm": ("#7fa6d9", "-", 1.8),
    "mlp": ("#d9622b", "-", 1.8),
}


def figure(r: dict[str, Any], path: Path, span: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "monospace", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160, width_ratios=[1.6, 1])
    months = r["months"]
    x = np.arange(len(months) + 1)
    for n in NAMES:
        color, style, width = LINE[n]
        equity = np.r_[1.0, np.cumprod(1 + np.array(r["monthly_return_pct"][n]) / 100)]
        left.plot(x, (equity - 1) * 100, style, color=color, lw=width, label=f"{n} ({KIND[n]})")
    lo, hi = min(r["placebo_total_pct"]), max(r["placebo_total_pct"])
    left.axhspan(lo, hi, color="#e6e6e6", zorder=0, label="19 random coins, final")
    left.axhline(0, color="#444444", lw=0.8)
    ticks = list(range(0, len(months), max(1, len(months) // 7)))
    left.set_xticks([t + 1 for t in ticks], [months[t] for t in ticks], fontsize=8)
    left.set_title(f"one-coin carry, account return %, {span}", loc="left", fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    y = np.arange(len(NAMES))[::-1]
    q = [r["forecast_quality"][n]["mean"] for n in NAMES]
    right.barh(y, q, color=[LINE[n][0] for n in NAMES], height=0.55)
    for yi, v, n in zip(y, q, NAMES, strict=True):
        right.text(v + 0.01, yi, f"{v:.2f}  same coin {r['same_coin_as_rule'][n]:.0%}",
                   va="center", fontsize=8)  # fmt: skip
    right.set_yticks(y, NAMES)
    right.set_xlim(0, 1.15)
    right.set_title("next month's funding: rank corr. of forecast", loc="left", fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(r: dict[str, Any]) -> str:
    head = ("| chooser | kind | total % | a year % | Sharpe | worst drawdown % | losing months "
            "| forecast corr. | same coin as `rule_lb3` | coin changes |\n"
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")  # fmt: skip
    lines = []
    for n in NAMES:
        s, c = r["summary"][n], r["coins"][n]
        changes = sum(a != b for a, b in pairwise(c))
        lines.append(
            f"| `{n}` | {KIND[n]} | {s['total_pct']:+.1f} | {s['a_year_pct']:+.1f} "
            f"| {s['sharpe_monthly']:.2f} | {s['worst_drawdown_pct']:.1f} | {s['losing_months']} "
            f"| {r['forecast_quality'][n]['mean']:.2f} | {r['same_coin_as_rule'][n]:.0%} | {changes} |"
        )
    return head + "\n" + "\n".join(lines)


def differences(r: dict[str, Any]) -> str:
    rows = ["| model | return minus `rule_lb3`, % a month | *t* | forecast corr. minus `rule_lb3`'s | *t* |",
            "|---|--:|--:|--:|--:|"]  # fmt: skip
    for n in ("ridge", "lightgbm", "mlp"):
        a, b = r["return_minus_rule"][n], r["quality_minus_rule"][n]
        rows.append(
            f"| `{n}` | {a['mean']:+.3f} | {a['t']:+.2f} | {b['mean']:+.3f} | {b['t']:+.2f} |"
        )
    return "\n".join(rows)


def statuses(r: dict[str, Any]) -> str:
    if "C1" not in r:
        return ""
    out = ["", "| model | C1, chooses better | Holm *p* | C2, forecasts better | Holm *p* |",
           "|---|---|--:|---|--:|"]  # fmt: skip
    for n in ("ridge", "lightgbm", "mlp"):
        a, b = r["C1"][n], r["C2"][n]
        out.append(
            f"| `{n}` | **{a['status']}** | {a['p_holm']:.3f} | **{b['status']}** | {b['p_holm']:.3f} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    source = Path(sys.argv[1])
    r = json.loads(source.read_text())
    out = source.parent
    span = f"{r['months'][0]} to {r['months'][-1]}"
    figure(r, out / "figure.png", span)
    best = max(NAMES, key=lambda n: r["summary"][n]["total_pct"])
    placebo = r["placebo_total_pct"]
    best_q = max(NAMES, key=lambda n: r["forecast_quality"][n]["mean"])
    models_q = [r["forecast_quality"][n]["mean"] for n in ("ridge", "lightgbm", "mlp")]
    tier = (
        "**Held-out months, read once.** C1 and C2's statuses are in `result.json` and below."
        if "C1" in r
        else "**A measurement on development months**, read before by the carry search; no status."
    )
    text = f"""# Machine learning choosing the carry's coin: {span}

Written by `experiments/carry_ml_report.py` from `{source.name}`
([registration](../../../docs/preregistration/carry_ml_2026_11.md)). Each chooser holds one coin a
month as a delta-neutral carry, the whole account, at Binance Regular fees, through the carry
search's engine. The models are refitted every month on the months already ended.
{tier}

![equity and forecast quality](figure.png)

{table(r)}

Months are the observations ({len(r["months"])}). Forecast corr.: each month's Spearman
correlation across candidates of the chooser's score and the month's realised funding.

{differences(r)}

- **Highest return: `{best}`**, {r["summary"][best]["total_pct"]:+.1f}%. Nineteen choosers holding a
  coin drawn at random each month ended between {min(placebo):+.1f}% and {max(placebo):+.1f}%.
- **Best forecast of the month's funding: `{best_q}`** ({r["forecast_quality"][best_q]["mean"]:.2f});
  the three-month mean's is {r["forecast_quality"]["rule_lb3"]["mean"]:.2f}, the models'
  {min(models_q):.2f} to {max(models_q):.2f}. A chooser that changes coin pays a round trip and
  takes the new coin's basis.
{statuses(r)}"""
    (out / "README.md").write_text(text)
    print(f"wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()
