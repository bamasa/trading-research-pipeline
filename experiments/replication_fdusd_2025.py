"""A descriptive replication of the author's 2025 research on Binance spot BTC/FDUSD.

Exploratory, not a blind test: the author's 2025 research on this pair covered
these dates, so nothing here is held out, and nothing here changes any
registration. It runs the spot study's S25-PF recipe (print-only inputs, ``gbm``, label fee 0, hold
50 s, the hold-first rule at 240 trades a day, taker in and taker out by the
side-correct print rule) on two 2025 windows, at zero fees and at the Regular
account's 10 bp taker fee, to show at which fee the result is positive::

    python -m experiments.replication_fdusd_2025 fetch --workers 4
    python -m experiments.replication_fdusd_2025 fit A --workers 6
    python -m experiments.replication_fdusd_2025 score A --workers 6
    python -m experiments.replication_fdusd_2025 report

Data and work files stay under ``data/replication/`` (ignored by git); the
results are ``experiments/results/replication_fdusd_2025.md`` and its CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
FETCH_CSV = RESULTS / "replication_fdusd_2025_fetch.csv"
DAYS_CSV = RESULTS / "replication_fdusd_2025_days.csv"
PAGE = RESULTS / "replication_fdusd_2025.md"


def _fetch_one(day: date, attempts: int = 4) -> dict[str, Any]:
    from trading_research.replication.fdusd_fetch import ROOT, fetch_day

    for attempt in range(attempts):
        try:
            return fetch_day(day, REPO / ROOT)
        except Exception as exc:  # retried, then recorded: the day is then missing
            if attempt == attempts - 1:
                return {"day": day.isoformat(), "status": "failed", "detail": repr(exc)[:300]}
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def fetch(workers: int) -> int:
    """Every replication day, verified against its published checksum; the log appended."""
    import pandas as pd

    from trading_research.replication.fdusd_fetch import ROOT, replication_days

    log = REPO / ROOT / "fetch_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for found in pool.map(_fetch_one, replication_days()):
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(found) + "\n")
            print(json.dumps(found), flush=True)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    done = pd.DataFrame([r for r in rows if r["status"] == "ok"]).drop_duplicates(
        "day", keep="last"
    )
    done = done.sort_values("day")[["day", "sha256", "bytes", "rows", "unit", "ids_consecutive"]]
    done.to_csv(FETCH_CSV, index=False)
    missing = sorted({d.isoformat() for d in replication_days()} - set(done["day"]))
    print(f"{len(done)} days verified and converted; missing: {missing or 'none'}")
    return 0 if not missing else 1


def _work(window: str) -> Path:
    from trading_research.replication.fdusd_fetch import ROOT

    return REPO / ROOT / "work" / window


def fit(window: str, workers: int) -> int:
    """The training days' stores (in parallel), then the fit and its thresholds."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from trading_research.replication.fdusd_fetch import ROOT
    from trading_research.replication.fdusd_fit import fit_window, write_store
    from trading_research.replication.fdusd_windows import WINDOWS

    found = WINDOWS[window]
    days = found.fold().train
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(write_store, root=REPO / ROOT), days))
    print(f"window {window}: {sum(1 for r in rows if r)} training days stored, {sum(rows)} rows")
    record = fit_window(found, REPO / ROOT)
    path = _work(window) / "fit.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def score(window: str, workers: int) -> int:
    """Every test day of the window at both fees, both variants; the rows written."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    import pandas as pd

    from trading_research.replication.fdusd_fetch import ROOT
    from trading_research.replication.fdusd_scoring import score_day
    from trading_research.replication.fdusd_windows import WINDOWS
    from trading_research.spot.costs import CONFIG_PATH, load_fee_table
    from trading_research.spot.fits import load_model
    from trading_research.strategies.hold_first import HoldFirstRule

    record = json.loads((_work(window) / "fit.json").read_text(encoding="utf-8"))
    model = load_model(_work(window) / "model.pkl", record["sha256"])
    rule = HoldFirstRule(**record["theta"])
    table = load_fee_table(REPO / CONFIG_PATH)
    run = partial(score_day, window, model, rule, table, root=REPO / ROOT)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for day in pool.map(run, WINDOWS[window].fold().test) for r in day]
    frame = pd.DataFrame(rows)
    frame.to_csv(_work(window) / "days.csv", index=False)
    print(frame.groupby(["variant", "fees"])[["trades", "net_quote", "gross_quote"]].sum())
    return 0


def report() -> int:
    """The per-day CSV and the results page, from both windows' fits and days."""
    import pandas as pd

    from trading_research.replication.fdusd_page import page
    from trading_research.replication.fdusd_summary import pooled

    fits = [json.loads((_work(w) / "fit.json").read_text(encoding="utf-8")) for w in "AB"]
    days = pd.concat([pd.read_csv(_work(w) / "days.csv") for w in "AB"], ignore_index=True)
    days = days.sort_values(["window", "variant", "fees", "day"]).reset_index(drop=True)
    days.to_csv(DAYS_CSV, index=False, float_format="%.6g")
    fetched = pd.read_csv(FETCH_CSV)
    PAGE.write_text(page(fits, pooled(days), days, fetched), encoding="utf-8")
    print(PAGE.read_text(encoding="utf-8"))
    return 0


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    found.add_argument("command", choices=["fetch", "fit", "score", "report"])
    found.add_argument("window", nargs="?", choices=["A", "B"])
    found.add_argument("--workers", type=int, default=4)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fetch":
        return fetch(args.workers)
    if args.command in ("fit", "score") and args.window is None:
        parser().error(f"{args.command} needs a window, A or B")
    if args.command == "fit":
        return fit(args.window, args.workers)
    if args.command == "score":
        return score(args.window, args.workers)
    return report()


if __name__ == "__main__":
    sys.exit(main())
