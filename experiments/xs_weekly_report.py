"""README and figure for an xs_weekly result (``development.json`` or ``result.json``).

PYTHONPATH=src python -m experiments.xs_weekly_report experiments/results/xs_weekly/development.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("ridge", "lightgbm", "mlp", "tcn", "xs_transformer")
KIND = {"momentum": "rule", "reversal": "rule", "low_vol": "rule", "funding": "rule",
        "ridge": "ML", "lightgbm": "ML", "mlp": "DL", "tcn": "DL", "xs_transformer": "DL"}  # fmt: skip
LINE = {
    "momentum": ("#b0b0b0", "--", 1.4), "funding": ("#2e8b57", "-", 2.0), "reversal": ("#8a8a8a", ":", 1.6),
    "low_vol": ("#222222", "-", 2.4), "ridge": ("#1f4e9c", "-", 1.8),
    "lightgbm": ("#7fa6d9", "-", 1.8), "mlp": ("#f0a070", "-", 1.6),
    "tcn": ("#d9622b", "--", 1.8), "xs_transformer": ("#b8321a", "-", 2.6),
}  # fmt: skip


def figure(r: dict[str, Any], path: Path) -> None:
    names = list(r["summary"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "monospace", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=160, width_ratios=[1.6, 1])
    weeks = r["weeks"]
    x = np.arange(len(weeks) + 1)
    for n in names:
        color, style, width = LINE[n]
        eq = np.r_[1.0, np.cumprod(1 + np.array(r["weekly_net"][n]))]
        left.plot(x, (eq - 1) * 100, style, color=color, lw=width, label=f"{n} ({KIND[n]})")
    left.axhline(0, color="#444444", lw=0.8)
    ticks = list(range(0, len(weeks), max(4, len(weeks) // 6)))
    left.set_xticks([t + 1 for t in ticks], [weeks[t][:7] for t in ticks], fontsize=8)
    left.set_title(
        f"long-short, net of fees and funding, % ({weeks[0]} to {weeks[-1]})",
        loc="left",
        fontsize=10,
    )
    left.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    y = np.arange(len(names))[::-1]
    ic = [r["summary"][n]["ic"] for n in names]
    right.barh(y, ic, color=[LINE[n][0] for n in names], height=0.6)
    for yi, v in zip(y, ic, strict=True):
        right.text(v + (0.002 if v >= 0 else -0.002), yi, f"{v:+.3f}", va="center",
                   ha="left" if v >= 0 else "right", fontsize=8)  # fmt: skip
    right.axvline(0, color="#444444", lw=0.8)
    right.margins(x=0.25)
    right.set_yticks(y, names)
    right.set_title("mean weekly rank corr. with next week", loc="left", fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    source = Path(sys.argv[1])
    r = json.loads(source.read_text())
    figure(r, source.parent / "figure.png")
    names, base = list(r["summary"]), r["baseline"]
    keys = ("N1", "N2") if "N1" in r else ("W1", "W2")
    rows = []
    for n in names:
        s, t = r["summary"][n], r["net"][n]
        rows.append(f"| `{n}` | {KIND[n]} | {s['total_pct']:+.1f} | {s['a_year_pct']:+.1f} | {s['sharpe']:.2f} "
                    f"| {s['worst_drawdown_pct']:.1f} | {s['price_pct']:+.1f} | {s['funding_pct']:+.1f} "
                    f"| {s['cost_pct']:+.1f} | {s['ic']:+.3f} | {s['turnover']:.0%} | {t['t']:+.2f} |")  # fmt: skip
    diffs = [f"| `{n}` | {r['minus_baseline'][n]['mean'] * 100:+.3f} | {r['minus_baseline'][n]['t']:+.2f} |"
             for n in MODELS]  # fmt: skip
    status = ""
    if keys[0] in r:
        status = f"\n| model | {keys[0]}, pays | Holm *p* | {keys[1]}, beats `{base}` | Holm *p* |\n|---|---|--:|---|--:|\n" + "\n".join(
            f"| `{n}` | **{r[keys[0]][n]['status']}** | {r[keys[0]][n]['p_holm']:.3f} "
            f"| **{r[keys[1]][n]['status']}** | {r[keys[1]][n]['p_holm']:.3f} |" for n in MODELS) + "\n"  # fmt: skip
    best = max(names, key=lambda n: r["summary"][n]["total_pct"])
    tier = ("**Held-out weeks, read once.**" if keys[0] in r
            else "**A measurement on development weeks**; no status.")  # fmt: skip
    text = f"""# ML and DL ranking a hundred perpetuals a week: {r["weeks"][0]} to {r["weeks"][-1]}

Written by `experiments/xs_weekly_report.py` from `{source.name}`
([registration](../../../docs/preregistration/{"xs_weekly_net_2026_11" if r["baseline"] == "momentum" else "xs_weekly_2026_11"}.md)). Every Monday each ranker
goes long its ten highest-ranked and short its ten lowest of the hundred most traded Binance
USDT perpetuals (about {r["universe_size"]:.0f} on average), half the account a side, at taker fees
with funding. {tier}

![equity and rank correlation](figure.png)
{status}
| ranker | kind | total % | a year % | Sharpe | worst drawdown % | price % | funding % | cost % | rank corr. | turnover | weekly *t* |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{chr(10).join(rows)}

{len(r["weeks"])} weeks. Price, funding and cost are summed weekly contributions to the account.

| model | weekly net minus `{base}`'s, % | *t* |
|---|--:|--:|
{chr(10).join(diffs)}

- **Highest total: `{best}`**, {r["summary"][best]["total_pct"]:+.1f}%.
"""
    (source.parent / "README.md").write_text(text)
    print(f"wrote {source.parent / 'README.md'}")


if __name__ == "__main__":
    main()
