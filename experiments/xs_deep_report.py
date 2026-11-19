"""The xs_deep development replay's tables, figure and README, from ``data/xs_deep/dev``.

Called by ``experiments/xs_deep_dev.py``. Days are the observations: a day's
coefficient is the mean over its decision times, its net the mean over its
legs. A measurement on development data; no status is read here.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.xs_deep.evaluate import LEGS, one_sided, spearman
from trading_research.xs_deep.features import TAKER_BP
from trading_research.xs_deep.rankers import NAMES

KIND = {
    "reversal": "rule",
    "ridge": "ML",
    "lightgbm": "ML",
    "mlp": "DL",
    "tcn": "DL",
    "xs_transformer": "DL",
}
COLOR = {"rule": "#8a8a8a", "ML": "#2f6db5", "DL": "#d9622b"}


def load(data: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(data / "xs_deep" / "dev" / f"{n}.parquet").assign(ranker=n) for n in NAMES
    ]
    return pd.concat(parts, ignore_index=True)


def daily(dec: pd.DataFrame) -> pd.DataFrame:
    return dec.groupby(["ranker", "span", "day"], as_index=False)[
        ["ic", "gross", "cost", "net"]
    ].mean()


def summary(day: pd.DataFrame, dec: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for n in NAMES:
        d = day[day["ranker"] == n].set_index("day")
        ic, net = one_sided(d["ic"]), one_sided(d["net"])
        out[n] = {
            "kind": KIND[n],
            "ic": ic["mean"],
            "ic_t": ic["t"],
            "ic_days_positive": float((d["ic"] > 0).mean()),
            "gross_bp": float(d["gross"].mean()),
            "cost_bp": float(d["cost"].mean()),
            "net_bp": net["mean"],
            "net_t": net["t"],
            "net_no_fee_bp": float(d["net"].mean() + 2 * TAKER_BP),
            "by_span_ic": {s: float(g["ic"].mean()) for s, g in d.groupby("span")},
            "by_span_net_bp": {s: float(g["net"].mean()) for s, g in d.groupby("span")},
            "legs": int(LEGS * 2 * (dec["ranker"] == n).sum()),
            "days": ic["days"],
        }
    return out


def difference(day: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    wide = day.pivot_table(index="day", columns="ranker", values="ic")
    return one_sided(wide[a] - wide[b])


def score_correlation(data: Path) -> pd.DataFrame:
    """Mean over decision times of the rank correlation between two rankers' scores."""
    z = {n: np.load(data / "xs_deep" / "dev" / f"{n}.npz") for n in NAMES}
    keys = [k for k in z[NAMES[0]].files if k.endswith("_scores")]
    corr = pd.DataFrame(np.eye(len(NAMES)), index=NAMES, columns=NAMES)
    for a, b in combinations(NAMES, 2):
        values = []
        for k in keys:
            for sa, sb in zip(z[a][k], z[b][k], strict=True):
                ok = np.isfinite(sa) & np.isfinite(sb)
                if ok.sum() >= 3:
                    values.append(spearman(sa[ok], sb[ok]))
        corr.loc[a, b] = corr.loc[b, a] = float(np.mean(values))
    return corr


def figure(day: pd.DataFrame, stats: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "monospace", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160, width_ratios=[1.5, 1])
    for n in NAMES:
        d = day[day["ranker"] == n].sort_values("day")
        x = np.arange(len(d))
        width = 2.6 if n == "xs_transformer" else 1.4
        left.plot(x, d["ic"].cumsum(), "-", lw=width, color=COLOR[KIND[n]],
                  alpha=1.0 if n in ("xs_transformer", "ridge", "reversal") else 0.55, label=n)  # fmt: skip
    first_span = day[(day["ranker"] == NAMES[0])].sort_values("day")
    split = int((first_span["span"] == first_span["span"].iloc[0]).sum())
    left.axvline(split - 0.5, color="#bbbbbb", lw=1, ls="--")
    left.text(
        split - 0.2,
        left.get_ylim()[1] * 0.95,
        "Feb-Mar | Mar-Apr 2024",
        color="#888888",
        fontsize=8,
    )
    left.axhline(0, color="#444444", lw=0.8)
    left.set_title(
        "summed daily information coefficient, scored days only", loc="left", fontsize=10
    )
    left.set_xlabel("scored day")
    left.legend(frameon=False, fontsize=8, ncol=2)
    names = list(NAMES)
    y = np.arange(len(names))[::-1]
    gross = [stats[n]["gross_bp"] for n in names]
    cost = [stats[n]["cost_bp"] for n in names]
    right.barh(y, gross, color=[COLOR[KIND[n]] for n in names], height=0.55)
    right.scatter(cost, y, marker="|", s=400, color="#222222", zorder=3, label="cost of a leg")
    right.set_yticks(y, names)
    right.axvline(0, color="#444444", lw=0.8)
    right.set_title("gross per leg vs its cost, bp", loc="left", fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt(x: float, digits: int = 2, sign: bool = True) -> str:
    return "-" if not np.isfinite(x) else (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}")


def readme(stats: dict[str, dict[str, Any]], diffs: dict[str, dict[str, float]],
           corr: pd.DataFrame, path: Path) -> None:  # fmt: skip
    rows = [
        f"| `{n}` | {s['kind']} | {_fmt(s['ic'], 4)} | {_fmt(s['ic_t'])} | {s['ic_days_positive']:.0%} "
        f"| {_fmt(s['gross_bp'])} | {_fmt(s['cost_bp'], sign=False)} | {_fmt(s['net_bp'])} "
        f"| {_fmt(s['net_t'])} | {_fmt(s['net_no_fee_bp'])} | {s['legs']:,} |"
        for n, s in stats.items()
    ]
    best = max(stats, key=lambda n: stats[n]["ic"])
    mean_cost = float(np.mean([s["cost_bp"] for s in stats.values()]))
    top_gross = max(stats, key=lambda n: stats[n]["gross_bp"])
    paying = [n for n, s in stats.items() if s["net_bp"] > 0]
    cost_line = (
        f"A leg pays about {mean_cost:.1f} bp in fee and spread. The largest gross per leg is "
        f"`{top_gross}`'s, {_fmt(stats[top_gross]['gross_bp'])} bp; "
        + (
            "no ranker's gross clears its cost."
            if not paying
            else ", ".join(f"`{n}`" for n in paying) + " clear it."
        )
    )
    text = f"""# Machine learning and deep learning on the cross-section, replayed on 2024

Written by `experiments/xs_deep_dev.py` (`report`). The six rankers of
[the registration](../../../docs/preregistration/xs_deep_forward_2026_12.md), at its settings,
fitted on 14 days and scored on the next 7, walking forward through 1 February to 10 March and
12 March to 20 April 2024. Every ten minutes each goes long its top {LEGS} and short its bottom
{LEGS} of the 26 perpetuals for ten minutes, entering one minute after the decision, at Bybit's
base taker fee ({TAKER_BP} bp a side) and the recorded spread. **A measurement on development
data, not the study's result**, which is read once on 30 November to 13 December 2026.

![summed daily coefficient and gross against cost](figure.png)

| ranker | kind | mean daily IC | *t* | days IC > 0 | gross / leg, bp | cost / leg, bp | net / leg, bp | net *t* | net at no fee, bp | legs |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{chr(10).join(rows)}

*t* over the {stats[best]["days"]} scored days. IC: Spearman correlation across instruments of the
ranking and the next ten minutes' relative return.

- **Ranking quality.** The highest mean daily coefficient is `{best}`'s, {_fmt(stats[best]["ic"], 4)}.
  `xs_transformer` minus `ridge`: {_fmt(diffs["xs_transformer-ridge"]["mean"], 4)} a day
  (*t* {_fmt(diffs["xs_transformer-ridge"]["t"])}); `xs_transformer` minus `tcn`:
  {_fmt(diffs["xs_transformer-tcn"]["mean"], 4)} (*t* {_fmt(diffs["xs_transformer-tcn"]["t"])}).
  These are X2 and X3's quantities, measured here, read on the test block only.
- **Cost.** {cost_line}
- **By span.** Mean daily IC on February to March against March to April:
  {"; ".join(f"`{n}` {_fmt(s['by_span_ic'].get('2024-02', float('nan')), 3)} / {_fmt(s['by_span_ic'].get('2024-04', float('nan')), 3)}" for n, s in stats.items())}.

Rank correlation of the rankers' scores, mean over decision times:

| | {" | ".join(f"`{n}`" for n in corr.columns)} |
|---|{"--:|" * len(corr.columns)}
{chr(10).join(f"| `{a}` | " + " | ".join(f"{corr.loc[a, b]:+.2f}" for b in corr.columns) + " |" for a in corr.index)}
"""
    path.write_text(text)


def report(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dec = load(data)
    day = daily(dec)
    stats = summary(day, dec)
    diffs = {
        "xs_transformer-ridge": difference(day, "xs_transformer", "ridge"),
        "xs_transformer-tcn": difference(day, "xs_transformer", "tcn"),
    }
    corr = score_correlation(data)
    day.to_csv(out / "daily.csv", index=False, float_format="%.6g")
    payload = {"rankers": stats, "differences": diffs, "score_correlation": corr.round(4).to_dict()}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    figure(day, stats, out / "figure.png")
    readme(stats, diffs, corr, out / "README.md")
    print(f"wrote {out}")
