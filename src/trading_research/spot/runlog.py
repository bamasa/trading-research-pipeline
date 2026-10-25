"""The run log (the trial manifest): every computation on a D test day, on record before H.

Registered (``docs/preregistration/spot_2026.md``, "The run log"): each
numbered complete D run of the entry point and each hand run appends a row with
its date, the run (``D<n>`` for a numbered D run, ``hand`` for a hand run) and
the days it read; this study adds the stage that read them and a note. The rows
live in parquet parts under ``data/spot_2026/`` (never committed), one part per
append, ``spot_2026_trials.part<k>.parquet``; the manifest
``experiments/results/spot_2026_trials.json`` lists each part's sha256 and row
count and the whole log's sha256 and rows, as
:meth:`trading_research.spot.ledger.Ledger.verify_trials` reads it. The D
amendment freezes that sha256 and row count under ``compute.record.trials``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.spot import registration as sr
from trading_research.spot.ledger import file_sha256, parquet_rows, run_log_parts, run_log_sha256

COLUMNS: tuple[str, ...] = ("date", "run", "stage", "days", "note")


def scored_test_days(days: Iterable[date]) -> list[date]:
    """The days of ``days`` that are D's test days (June and July)."""
    return [d for d in days if date(2026, 6, 1) <= d <= date(2026, 7, 31)]


def append(
    repo: Path,
    *,
    run: str,
    stage: str,
    days: Iterable[date],
    note: str = "",
    today: date | None = None,
) -> Path | None:
    """Append one row for a computation that read ``days`` (only their test days are
    logged; nothing is written when there is none); returns the new part."""
    read = sorted(set(scored_test_days(days)))
    if not read:
        return None
    single = repo / sr.TRIALS_PATH
    parts = run_log_parts(repo)
    if single in parts:
        raise RuntimeError(f"{single} is a single-file run log; this study appends parts")
    target = single.with_name(f"{single.stem}.part{len(parts) + 1:04d}{single.suffix}")
    frame = pd.DataFrame(
        [
            {
                "date": (today or datetime.now(UTC).date()).isoformat(),
                "run": run,
                "stage": stage,
                "days": ",".join(d.isoformat() for d in read),
                "note": note,
            }
        ],
        columns=list(COLUMNS),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def manifest(repo: Path) -> dict[str, Any]:
    """The manifest of the parts on disk, in order."""
    parts = run_log_parts(repo)
    if not parts:
        raise RuntimeError("the run log is empty: no computation on a D test day is on record")
    listed = [{"file": p.name, "sha256": file_sha256(p), "rows": parquet_rows(p)} for p in parts]
    return {
        "parts": listed,
        "sha256": run_log_sha256(parts),
        "rows": sum(int(p["rows"]) for p in listed),
    }


def write_manifest(repo: Path) -> dict[str, Any]:
    """Write the manifest where the registration names it; its {sha256, rows}."""
    found = manifest(repo)
    path = repo / sr.TRIALS_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"sha256": found["sha256"], "rows": found["rows"]}


def rows(repo: Path) -> pd.DataFrame:
    """Every row of the log, in part order."""
    parts = run_log_parts(repo)
    if not parts:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
