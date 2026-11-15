"""How much the carry search's choice is worth, given that it chose among sixteen rules.

``trp search carry`` scored sixteen rules on January 2021 to June 2023 (D),
chose the best monthly Sharpe ratio (one coin, three months of funding, no
floor) and read it once on July 2023 to June 2025 (H). This replays the
sixteen with the same engine, checks that each D summary equals the one the
search wrote, and reports beside the choice:

* N_eff, the independent trials among the sixteen, from their monthly returns;
* the deflated Sharpe ratio of the choice on D (against the best of N_eff
  rules of no skill) and on H;
* the probability of backtest overfitting by CSCV on D: 30 months in ten
  blocks of three, the best Sharpe chosen on every half, ranked on the other;
* the minimum track record length: how many months the choice's Sharpe ratio
  needs before it is above zero, and above the best of N_eff null rules, at
  95%;
* Harvey and Liu's haircut of the choice's Sharpe ratio on D, by Bonferroni,
  Holm and BHY over the sixteen.

It also replays the fifteen rules not chosen on H, which the search never
did: where the choice ranks among them there is after the fact, a
measurement, not a test. Nothing here changes the search's status.

    PYTHONPATH=src python experiments/carry_overfitting.py --data data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.evaluation.overfitting import (
    HAIRCUT_METHODS,
    cscv_pbo,
    deflated_sharpe_ratio,
    effective_trials,
    expected_maximum_sharpe,
    haircut_sharpe,
    minimum_track_record_length,
    sharpe_ratios,
    sharpe_variance,
)
from trading_research.search_engine.carry import (
    DEVELOPMENT,
    GRID,
    HELD_OUT,
    load_universe,
    readable,
)
from trading_research.strategies.carry import replay, summary

HERE = Path(__file__).parent
SEARCH = HERE / "results" / "search_carry"
OUT = HERE / "results" / "carry_overfitting"
BLOCKS = 10
A_YEAR = math.sqrt(12)
NORMAL_Z = 1.6448536269514722  # the 95% one-sided normal quantile


def monthly(markets: dict[str, Any], months: pd.PeriodIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every rule's monthly return (a fraction) and its summary on ``months``."""
    returns, summaries = {}, {}
    for r in GRID:
        frame = replay(markets, months, r, readable)
        returns[r.name] = frame["return_pct"].to_numpy() / 100
        summaries[r.name] = summary(frame)
    return pd.DataFrame(returns, index=[str(m) for m in months]), summaries


def pbo(panel: pd.DataFrame) -> dict[str, Any]:
    values = panel.to_numpy()
    every = np.arange(values.shape[1])

    def choose(rows: np.ndarray) -> tuple[int | None, np.ndarray]:
        s = sharpe_ratios(values[rows])
        return int(np.nanargmax(s)), every

    out = cscv_pbo(len(values), choose, lambda rows: sharpe_ratios(values[rows]), BLOCKS)
    return {"pbo": out.pbo, "splits": out.splits, "median_logit": float(np.median(out.logits))}


def checks(d: pd.DataFrame, h: pd.DataFrame, chosen: str) -> dict[str, Any]:
    n = len(GRID)
    n_eff = effective_trials(n, d)
    variance = sharpe_variance(d)
    benchmark = expected_maximum_sharpe(n_eff, variance)
    col = list(d.columns).index(chosen)
    sharpes_d = sharpe_ratios(d)
    out: dict[str, Any] = {
        "trials": n,
        "n_eff": n_eff,
        "sharpe_variance_per_month": variance,
        "null_best_sharpe_per_month": benchmark,
    }
    for label, panel in (("development", d), ("held_out", h)):
        series = panel[chosen].to_numpy()
        dsr = deflated_sharpe_ratio(series, n_eff, variance)
        dsr_all = deflated_sharpe_ratio(series, n, variance)
        psr = deflated_sharpe_ratio(series, 1, variance)
        above_zero = minimum_track_record_length(series, 0.0)
        above_null = minimum_track_record_length(series, expected_maximum_sharpe(n, variance))
        sr = above_zero.sharpe
        normal = 1 + (1 + sr**2 / 2) * (NORMAL_Z / sr) ** 2 if sr > 0 else math.nan
        out[label] = {
            "months": len(series),
            "sharpe_a_year": dsr.sharpe * A_YEAR,
            "skewness": dsr.skewness,
            "kurtosis": dsr.kurtosis,
            "psr": psr.dsr,
            "dsr": dsr.dsr,
            "dsr_sixteen": dsr_all.dsr,
            "min_months_above_zero": above_zero.minimum,
            "min_months_above_zero_if_normal": normal,
            "min_months_above_null_best_of_sixteen": above_null.minimum,
        }
    out["development"]["haircut"] = {
        m: {"adjusted_p": hc.adjusted_p_value, "sharpe_a_year": hc.adjusted_sharpe * A_YEAR,
            "haircut": hc.haircut}
        for m in HAIRCUT_METHODS
        for hc in [haircut_sharpe(sharpes_d, len(d), col, m)]
    }  # fmt: skip
    out["development"]["pbo"] = pbo(d)
    sharpes_h = sharpe_ratios(h)
    out["held_out"]["rank_among_rules"] = int((sharpes_h > sharpes_h[col]).sum()) + 1
    out["held_out"]["best_rule"] = str(h.columns[int(np.nanargmax(sharpes_h))])
    return out


def page(c: dict[str, Any], chosen: str, board_d: dict[str, Any], board_h: dict[str, Any]) -> str:
    d, h = c["development"], c["held_out"]
    hc = d["haircut"]

    def months(x: float) -> str:
        return "never (Sharpe at or below it)" if not math.isfinite(x) else f"{x:.0f}"

    lines = [
        "# The carry search's choice against the size of its search",
        "",
        "**What it says.** The carry survives the size of the search; the choice among the",
        "sixteen does not. Every check on the chosen rule's own months passes with room: its",
        "Sharpe ratio after Harvey and Liu's haircut is still "
        f"{hc['holm']['sharpe_a_year']:.2f} a year, and even counting the",
        "months as normal it needs about "
        f"{d['min_months_above_zero_if_normal']:.0f} of them to be told from zero, against 30 on D and 24",
        "on H. But choosing the best Sharpe ratio among rules whose Sharpe ratios differ by",
        f"a few hundredths is a coin toss: PBO {d['pbo']['pbo']:.2f}, and on H the choice ranks "
        f"{h['rank_among_rules']} of {c['trials']}",
        "while all sixteen are positive. The edge is the carry, not the rule; holding more coins",
        "would have done as well with a fraction of the drawdown.",
        "",
        "Written by `experiments/carry_overfitting.py`. The search chose "
        f"`{chosen}` among {c['trials']} rules on 30 development months (D) and read it once on",
        "24 held-out months (H). Monthly returns after costs; Sharpe ratios annualised by √12.",
        "",
        f"The {c['trials']} rules are {c['n_eff']:.1f} independent trials (their monthly returns are",
        f"correlated); the best of that many rules of no skill would show a Sharpe ratio of about "
        f"{c['null_best_sharpe_per_month'] * A_YEAR:.2f} a year.",
        "",
        "| | D (chosen here) | H (read once) |",
        "|---|--:|--:|",
        f"| months | {d['months']} | {h['months']} |",
        f"| Sharpe ratio, a year | {d['sharpe_a_year']:.2f} | {h['sharpe_a_year']:.2f} |",
        f"| skewness, kurtosis | {d['skewness']:+.2f}, {d['kurtosis']:.1f} | "
        f"{h['skewness']:+.2f}, {h['kurtosis']:.1f} |",
        f"| probability the Sharpe ratio is above 0 (PSR) | {d['psr']:.3f} | {h['psr']:.3f} |",
        f"| deflated Sharpe ratio, against the best of {c['n_eff']:.1f} null rules | "
        f"{d['dsr']:.3f} | {h['dsr']:.3f} |",
        f"| deflated Sharpe ratio, counting all {c['trials']} as independent | "
        f"{d['dsr_sixteen']:.3f} | {h['dsr_sixteen']:.3f} |",
        f"| months needed to be above 0 at 95% | {months(d['min_months_above_zero'])} | "
        f"{months(h['min_months_above_zero'])} |",
        f"| the same, if the months were normal (no skew, no fat tails) | "
        f"{months(d['min_months_above_zero_if_normal'])} | "
        f"{months(h['min_months_above_zero_if_normal'])} |",
        f"| months needed to be above the best of {c['trials']} null rules at 95% | "
        f"{months(d['min_months_above_null_best_of_sixteen'])} | "
        f"{months(h['min_months_above_null_best_of_sixteen'])} |",
        "",
        "Harvey and Liu's haircut of the D Sharpe ratio, for having looked at sixteen:",
        "",
        "| adjustment | adjusted *p* | Sharpe ratio left, a year | haircut |",
        "|---|--:|--:|--:|",
        *[
            f"| {m} | {hc[m]['adjusted_p']:.4f} | {hc[m]['sharpe_a_year']:.2f} | "
            f"{hc[m]['haircut']:.0%} |"
            for m in HAIRCUT_METHODS
        ],
        "",
        f"Probability of backtest overfitting on D (CSCV, {BLOCKS} blocks of three months, "
        f"{d['pbo']['splits']} splits): **{d['pbo']['pbo']:.2f}**, median logit "
        f"{d['pbo']['median_logit']:+.2f}.",
        "",
        "After the fact, all sixteen rules replayed on H (the search read only the choice):",
        f"the choice ranks {h['rank_among_rules']} of {c['trials']} by Sharpe ratio there; the best "
        f"on H is `{h['best_rule']}` ({board_h[h['best_rule']]['sharpe_monthly']:.2f}).",
        "",
        "| rule | D Sharpe | H Sharpe | H total |",
        "|---|--:|--:|--:|",
        *[
            f"| `{k}`{' (chosen)' if k == chosen else ''} | {board_d[k]['sharpe_monthly']:.2f} | "
            f"{board_h[k]['sharpe_monthly']:.2f} | {board_h[k]['total_pct']:+.1f}% |"
            for k in sorted(board_h, key=lambda k: -board_h[k]["sharpe_monthly"])
        ],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    root = parser.parse_args().data / "carry_walkforward"
    dev = json.loads((SEARCH / "development.json").read_text())
    held = json.loads((SEARCH / "held_out.json").read_text())
    chosen = dev["chosen"]
    markets = load_universe(root)
    d, board_d = monthly(markets, DEVELOPMENT)
    h, board_h = monthly(markets, HELD_OUT)
    for name, s in board_d.items():
        assert math.isclose(s["total_pct"], dev["board"][name]["total_pct"], rel_tol=1e-9), name
    assert math.isclose(board_h[chosen]["total_pct"], held["result"]["total_pct"], rel_tol=1e-9)
    c = checks(d, h, chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(c, indent=2, default=float) + "\n")
    (OUT / "held_out_board.json").write_text(json.dumps(board_h, indent=2, default=float) + "\n")
    text = page(c, chosen, board_d, board_h)
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
