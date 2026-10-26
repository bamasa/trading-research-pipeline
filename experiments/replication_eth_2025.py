"""Replicating the author's 2025 ETH result on public data, then testing it frozen.

Part 1 (exploratory) rebuilds the recipe on Bybit's ETHUSDT linear perpetual
for 2025-01-28 .. 2025-02-06: train on the first seven days, validate on the
eighth, test on 2025-02-05 .. 06, every gate configuration the original ran on
those two days replayed. Part 2 (registered in docs/replication_eth_plan.md
before any 2026 day was fetched) applies the frozen recipe to
2026-08-08 .. 2026-08-21.

Stages, each resumable and run separately (training takes hours)::

    python experiments/replication_eth_2025.py fetch 2025-01-28 2025-02-06
    python experiments/replication_eth_2025.py shares 2025-01-28 2025-02-03
    python experiments/replication_eth_2025.py train part1
    python experiments/replication_eth_2025.py predict part1
    python experiments/replication_eth_2025.py backtest part1
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.replication_eth.days import MIN_VOLATILITY
from trading_research.replication_eth.lob import labels
from trading_research.replication_eth.store import fetch_day, load_span, span
from trading_research.replication_eth.sweep import FROZEN, SWEEP, Run

ROOT = Path("data/replication_eth")
RESULTS = Path("experiments/results")

#: The labels' threshold, chosen in part 1 from the original's printed class
#: mix on its training days (see ``shares``) and frozen for part 2.
ALPHA = 5e-4

#: One seed for every fit, fixed before part 1 ran.
SEED = 0


def _date(text: str) -> date:
    return date.fromisoformat(text)


#: Each part's days: seven to train, one to validate (the original's 0.7 / 0.1
#: of ten days, cut at day boundaries), then the test days.
PARTS: dict[str, dict[str, object]] = {
    "part1": {
        "train": span(date(2025, 1, 28), date(2025, 2, 3)),
        "val": [date(2025, 2, 4)],
        "test": span(date(2025, 2, 5), date(2025, 2, 6)),
        "runs": SWEEP,
    },
    "part2": {
        "train": span(date(2026, 7, 31), date(2026, 8, 6)),
        "val": [date(2026, 8, 7)],
        "test": span(date(2026, 8, 8), date(2026, 8, 21)),
        "runs": FROZEN,
    },
}


def _days(part: str, key: str) -> list[date]:
    days = PARTS[part][key]
    assert isinstance(days, list)
    return days


def _xy(days: list[date]) -> tuple[np.ndarray, np.ndarray]:
    from trading_research.replication_eth.train import targets

    data = load_span(ROOT, days)
    x = np.concatenate(data["x"])
    y = targets(
        labels(np.concatenate(data["rel50"]), ALPHA),
        np.concatenate(data["volatility"]),
        MIN_VOLATILITY,
    )
    return x, y


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from trading_research.replication_eth.train import fit

    x_train, y_train = _xy(_days(args.part, "train"))
    x_val, y_val = _xy(_days(args.part, "val"))
    print(f"train {x_train.shape}, val {x_val.shape}, device {_device()}", flush=True)

    def log(row: dict[str, float]) -> None:
        print(json.dumps(row), flush=True)

    model, history = fit(x_train, y_train, x_val, y_val, seed=SEED, device=_device(), log=log)
    target = ROOT / "models" / f"{args.part}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, target)
    pd.DataFrame(history).to_csv(
        RESULTS / f"replication_eth_2025_{args.part}_training.csv", index=False
    )


def cmd_predict(args: argparse.Namespace) -> None:
    import torch

    from trading_research.replication_eth.net import DeepTCN
    from trading_research.replication_eth.train import predict

    x, _ = _xy(_days(args.part, "test"))
    model = DeepTCN(x.shape[1])
    model.load_state_dict(torch.load(ROOT / "models" / f"{args.part}.pt"))
    probs = predict(model.to(_device()), x, _device())
    target = ROOT / "probs" / f"{args.part}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, probs)
    _, y = _xy(_days(args.part, "test"))
    scored = (y != -100) & np.isfinite(probs[:, 0])
    guess = probs[scored].argmax(axis=1)
    mix = np.bincount(y[scored], minlength=3) / scored.sum()
    print(
        json.dumps(
            {
                "rows": len(y),
                "scored": int(scored.sum()),
                "accuracy": float((guess == y[scored]).mean()),
                "always_flat_accuracy": float(mix[1]),
                "class_mix": mix.round(4).tolist(),
                "predicted_mix": (np.bincount(guess, minlength=3) / len(guess)).round(4).tolist(),
            }
        ),
        flush=True,
    )


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Pooled over a run's days, per fee: counts, money, and the gross edge."""
    keys = ["run", "strategy", "source", "fee_bp_per_side"]
    pooled = table.groupby(keys, sort=False).agg(
        days=("day", "nunique"),
        signals=("signals", "sum"),
        round_trips=("round_trips", "sum"),
        gross_usdt=("gross_usdt", "sum"),
        fees_usdt=("fees_usdt", "sum"),
        net_usdt=("net_usdt", "sum"),
        turnover_usdt=("turnover_usdt", "sum"),
        days_net_positive=("net_usdt", lambda v: int((v > 0).sum())),
    )
    pooled["fills_per_day"] = 2 * pooled["round_trips"] / pooled["days"]
    pooled["gross_bp_of_turnover"] = 1e4 * pooled["gross_usdt"] / pooled["turnover_usdt"]
    pooled["break_even_fee_bp_per_side"] = pooled["gross_bp_of_turnover"]
    return pooled.reset_index()


#: The original's printed backtests (Binance spot, 2025-02-05 .. 06, 0.01 ETH):
#: fills, PnL before and after its fee, in USDT, rounded as its tables show
#: them; keyed by the replication's run and holding.
ORIGINAL: dict[tuple[str, str], tuple[int, float, float]] = {
    ("ref37", "signal_to_signal"): (185, 1, 0),
    ("c121", "signal_to_signal"): (298, -1, -2),
    ("c122", "signal_to_signal"): (300, -0, -2),
    ("c124", "signal_to_signal"): (298, -1, -2),
    ("c125", "signal_to_signal"): (496, 1, -2),
    ("c126", "signal_to_signal"): (798, 1, -4),
    ("c133", "signal_to_signal"): (350, 2, 0),
    ("c138", "signal_to_signal"): (592, 3, -1),
    ("c149", "signal_to_signal"): (350, 2, 0),
    ("c150", "signal_to_signal"): (930, 5, -1),
    ("c155", "signal_to_signal"): (796, 1, -4),
    ("c156", "signal_to_signal"): (2424, 3, -12),
    ("c164", "signal_to_signal"): (1591, 4, -6),
    ("c150", "close_by_time_100s"): (1407, 3, -6),
    ("c176", "signal_to_signal"): (932, 5, -1),
    ("c183", "signal_to_signal"): (316, 2, -0),
    ("c189", "close_by_time_100s"): (1936, 6, -6),
    ("c189", "signal_to_signal"): (1458, 7, -2),
    ("c195", "close_by_time_100s"): (1982, 6, -7),
    ("c195", "signal_to_signal"): (1502, 7, -3),
    ("ref48", "signal_to_signal"): (358, 3.1, 0.9),
    ("c205", "signal_to_signal"): (504, 1, -2),
    ("c206", "signal_to_signal"): (352, 3, 1),
    ("ref48", "close_by_time_500s"): (454, 2, -1),
    ("c218", "signal_to_signal"): (348, 3, 1),
    ("c223", "signal_to_signal"): (960, 5, -1),
}


def wide(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per run: counts, gross, net at each fee, the gross edge, and the
    original's printed numbers where it ran the same thing."""
    base = summary[summary["fee_bp_per_side"] == 0.0].set_index(["run", "strategy"])
    out = base[["source", "days", "signals", "fills_per_day", "gross_usdt"]].copy()
    for fee in (2.2, 5.5):
        part = summary[summary["fee_bp_per_side"] == fee].set_index(["run", "strategy"])
        out[f"net_{fee:g}bp_usdt"] = part["net_usdt"]
        out[f"days_net_positive_{fee:g}bp"] = part["days_net_positive"]
    out["gross_bp_of_turnover"] = base["gross_bp_of_turnover"]
    for i, name in enumerate(("original_fills", "original_pnl", "original_pnl_after_fee")):
        out[name] = [ORIGINAL.get(key, (np.nan, np.nan, np.nan))[i] for key in out.index]
    return out.reset_index()


def cmd_backtest(args: argparse.Namespace) -> None:
    from trading_research.replication_eth.evaluate import day_rows, signals
    from trading_research.replication_eth.store import book_path

    days = _days(args.part, "test")
    data = load_span(ROOT, days)
    keys = ("ts", "volatility", "rv_50", "mid_mean_50", "spread_rolling_10")
    arrays = {key: np.concatenate(data[key]) for key in keys}
    probs = np.load(ROOT / "probs" / f"{args.part}.npy")
    books = {day: pd.read_parquet(book_path(ROOT, day)) for day in days}
    runs = PARTS[args.part]["runs"]
    assert isinstance(runs, tuple)
    rows: list[dict[str, object]] = []
    for run in runs:
        assert isinstance(run, Run)
        decision_ns, side = signals(arrays, probs, run.gate)
        rows += day_rows(run, days, books, decision_ns, side)
        print(run.gate.name, run.strategy, len(decision_ns), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"replication_eth_2025_{args.part}_days.csv", index=False)
    summary = summarise(table)
    summary.to_csv(RESULTS / f"replication_eth_2025_{args.part}_summary.csv", index=False)
    table = wide(summary)
    table.to_csv(RESULTS / f"replication_eth_2025_{args.part}_runs.csv", index=False)
    print(table.round(3).to_string(index=False))


def cmd_fetch(args: argparse.Namespace) -> None:
    days = span(args.start, args.end)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(fetch_day, [ROOT] * len(days), days):
            print(json.dumps(result, default=str), flush=True)


def cmd_shares(args: argparse.Namespace) -> None:
    """Class shares on the training days at the two thresholds the original's
    code holds (2e-4 printed when it ran on one day, 5e-4 in its source), against
    the mix it printed for its ten-day training split (5 s, filtered rows:
    9.05 / 81.88 / 9.08 %; 10 s, every row: 7.62 / 84.88 / 7.50 %). The one
    with the smaller largest gap is the replication's threshold."""
    data = load_span(ROOT, span(args.start, args.end))
    vol = np.concatenate(data["volatility"])
    rel50, rel100 = np.concatenate(data["rel50"]), np.concatenate(data["rel100"])
    rows = []
    printed = np.array([0.0905, 0.8188, 0.0908, 0.0762, 0.8488, 0.0750])
    for alpha in (2e-4, 5e-4):
        y50, y100 = labels(rel50, alpha), labels(rel100, alpha)
        kept = vol >= MIN_VOLATILITY
        s50 = np.bincount(y50[kept], minlength=4)[1:] / kept.sum()
        s100 = np.bincount(y100, minlength=4)[1:] / len(y100)
        rows.append(
            {
                "alpha": alpha,
                "h5s_filtered_down": s50[0],
                "h5s_filtered_flat": s50[1],
                "h5s_filtered_up": s50[2],
                "h10s_all_down": s100[0],
                "h10s_all_flat": s100[1],
                "h10s_all_up": s100[2],
                "filtered_out": 1 - kept.mean(),
                "largest_gap_to_printed": float(np.abs(np.r_[s50, s100] - printed).max()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "replication_eth_2025_label_shares.csv", index=False)
    print(table.round(4).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download, replay and store days")
    fetch.add_argument("start", type=_date)
    fetch.add_argument("end", type=_date)
    fetch.add_argument("--workers", type=int, default=2)
    fetch.set_defaults(run=cmd_fetch)
    shares = sub.add_parser("shares", help="label shares at candidate thresholds")
    shares.add_argument("start", type=_date)
    shares.add_argument("end", type=_date)
    shares.set_defaults(run=cmd_shares)
    for name, run in (("train", cmd_train), ("predict", cmd_predict), ("backtest", cmd_backtest)):
        stage = sub.add_parser(name)
        stage.add_argument("part", choices=sorted(PARTS))
        stage.set_defaults(run=run)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
