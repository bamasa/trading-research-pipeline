"""The model-chosen reversion on the two recorded tracks: one table from the bench cards.

Reads each variant's card (``trp bench run NAME --tracks bybit-2024-02,bybit-2024-04
--output experiments/results/reversion_meta``) and the plain and guarded rule's
numbers on the same tracks, and writes experiments/results/reversion_meta/README.md.
These tracks are where the variants were built: a measurement, not a test. The
test is M1, on the forward fortnight.

    python experiments/reversion_meta_board.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "reversion_meta"
TRACKS = ("bybit-2024-02", "bybit-2024-04")
VARIANTS = {
    "reversion_logistic": "logistic regression",
    "reversion_meta": "XGBoost",
    "reversion_lightgbm": "LightGBM",
    "reversion_transformer": "transformer",
    "reversion_lstm": "LSTM",
    "reversion_gru": "GRU",
    "reversion_tcn": "TCN",
    "reversion_meta_guarded": "XGBoost + kill switch",
}


def scores(name: str) -> dict[str, tuple[int, float, float]]:
    card = json.loads((OUT / name / "card.json").read_text())
    return {s["track"]: (s["trades"], s["net_bp"], s["net_t"]) for s in card["scores"]}


def baselines() -> dict[str, dict[str, tuple[int, float, float]]]:
    board = json.loads((RESULTS / "bench_board.json").read_text())
    plain = next(r for r in board["rows"] if r["name"] == "reversion")
    out = {
        "reversion": {
            t: (plain[t]["trades"], plain[t]["net_bp"], plain[t]["net_t"]) for t in TRACKS
        }
    }
    # the guarded rule's card is a page; its table rows carry trades, gross, net and t
    page = (RESULTS / "reversion_guarded" / "card.md").read_text()
    guarded = {}
    for t in TRACKS:
        cells = re.search(
            rf"^\| {t} \| [^|]+ \| ([\d,]+) \| [^|]+ \| ([-+\d.]+) \| ([-+\d.]+|-) \|", page, re.M
        )
        assert cells, t
        tt = float("nan") if cells[3] == "-" else float(cells[3])
        guarded[t] = (int(cells[1].replace(",", "")), float(cells[2]), tt)
    out["reversion_guarded"] = guarded
    return out


def row(label: str, s: dict[str, tuple[int, float, float]]) -> str:
    cells = []
    total = 0.0
    for t in TRACKS:
        n, net, tt = s[t]
        total += n * net
        t_text = "-" if tt != tt else f"{tt:+.2f}"
        cells.append(f"{n:,} | {net:+.1f} | {t_text}")
    return f"| {label} | {' | '.join(cells)} | {total:+,.0f} |"


def main() -> None:
    base = baselines()
    lines = [
        "# The reversion rule with a model choosing its trades, on the recorded tracks",
        "",
        "Written by `experiments/reversion_meta_board.py` from the bench cards in this folder.",
        "Net per trade in bp after Bybit's base taker fee, slippage and the spread; *t* clustered",
        "by day; the last column sums net bp over both tracks' trades. These are the tracks the",
        "variants were built on: a measurement, not a test. The test is",
        "[M1](../../../docs/preregistration/reversion_meta_forward_2026_11.md), on the forward fortnight.",
        "",
        "| who chooses the trades | Feb trades | Feb net | Feb *t* | Apr trades | Apr net | Apr *t* | summed net, bp |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
        row("nobody: the plain rule", base["reversion"]),
        row("nobody, kill switch (`reversion_guarded`)", base["reversion_guarded"]),
    ]
    lines += [row(f"{label} (`{name}`)", scores(name)) for name, label in VARIANTS.items()]
    lines += [
        "",
        "- **Simple models beat the networks here.** On six inputs and a month of candidates,",
        "  logistic regression and the boosters raise February's net from +7.0 to +12 to +18 bp",
        "  a trade; the transformer, LSTM, GRU and TCN add little and lose the most in April.",
        "- **No model saves April.** Every variant without the kill switch loses there. The",
        "  boosters make each trade less bad (-30 to -35 bp against the rule's -47) and take a",
        "  quarter fewer, but nothing in the six inputs at the hour of the trade says the state",
        "  has changed.",
        "- **The kill switch does.** It stops trading within days of the break: 52 and 55",
        "  April trades instead of about 900. XGBoost choosing the trades and the switch",
        "  stopping them is the best sum over both tracks (+9,927 bp against the switch alone's",
        "  +8,056), on a third of the February trades.",
        "- No February number is told apart from chance (*t* at most +1.92).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[8:20]))


if __name__ == "__main__":
    main()
