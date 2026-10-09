"""The trial ledger: zstd parquet parts, the sidecar, N_trials and the hand-run log."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.search_engine.catalogue import Catalogue
from trading_research.search_engine.ledger import (
    LEDGER_COLUMNS,
    LEDGER_PATH,
    LedgerError,
    TrialLedger,
    file_sha256,
)
from trading_research.validation.choice import CellDays

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"
DAYS = [date(2025, 2, 1) + timedelta(days=i) for i in range(10)]


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return Catalogue.load(CONFIG)


def _rows(
    catalogue: Catalogue, run: int, keys: int = 6, days: list[date] = DAYS, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = [(k, d) for k in catalogue.keys[:keys] for d in days]
    trades = rng.integers(0, 5, len(grid))
    value = np.where(trades > 0, rng.normal(0, 3, len(grid)), 0.0)
    return pd.DataFrame(
        {
            "run": run,
            "key": [k for k, _ in grid],
            "day": [d for _, d in grid],
            "net_bp": value,
            "G_bp": value / 2,
            "pnl_long_bp": value - 1,
            "pnl_short_bp": -value - 1,
            "trades": trades,
            "skips": rng.integers(0, 2, len(grid)),
        }
    )[list(LEDGER_COLUMNS)]


def test_rows_round_trip_with_their_sidecar(tmp_path: Path, catalogue: Catalogue) -> None:
    ledger = TrialLedger(tmp_path / LEDGER_PATH.name, catalogue)
    rows = _rows(catalogue, 1)
    record = ledger.append(1, "abc1234", rows)
    back = ledger.read()
    pd.testing.assert_frame_equal(back, rows, check_dtype=False)
    assert isinstance(back["day"].iloc[0], date)
    assert record["catalogue_sha256"] == catalogue.sha256 == catalogue.registered_sha256
    assert record["runs"]["1"] == {"commit": "abc1234", "rows": 60, "keys": 6, "days": 10}
    assert record["rows"] == 60 and record["n_trials"] == 6 == ledger.n_trials
    (part,) = record["parts"]
    assert part["file"] == LEDGER_PATH.name and part["rows"] == 60
    assert part["sha256"] == file_sha256(ledger.path)
    assert json.loads(ledger.sidecar_path.read_text()) == record
    assert ledger.sidecar_path.name == "short_horizon_2025_trials.json"


def test_one_run_pivots_into_cell_days(tmp_path: Path, catalogue: Catalogue) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    ledger.append(1, "abc", _rows(catalogue, 1))
    ledger.append(2, "def", _rows(catalogue, 2, seed=1))
    keys = list(catalogue.keys[:6])
    cells = CellDays.from_ledger(ledger.run_rows(2), keys, DAYS)
    expected = _rows(catalogue, 2, seed=1)
    assert cells.trades.sum() == expected["trades"].sum()
    traded = expected[expected["trades"] > 0].iloc[0]
    i, j = keys.index(traded["key"]), DAYS.index(traded["day"])
    assert cells.net_bp[i, j] == traded["net_bp"] and cells.g_bp[i, j] == traded["G_bp"]


def test_n_trials_counts_every_run_and_stages_append(tmp_path: Path, catalogue: Catalogue) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    ledger.append(1, "abc", _rows(catalogue, 1, days=DAYS[:5]))
    ledger.append(1, "abc", _rows(catalogue, 1, days=DAYS[5:]))
    assert ledger.n_trials == 6 and ledger.sidecar()["runs"]["1"]["days"] == 10
    ledger.append(2, "def", _rows(catalogue, 2, keys=8))  # a superseded run 1 stays
    assert ledger.n_trials == 6 + 8 and len(ledger.read()) == 60 + 80


def test_rows_that_cannot_enter_are_refused(tmp_path: Path, catalogue: Catalogue) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    rows = _rows(catalogue, 1)
    ledger.append(1, "abc", rows)
    with pytest.raises(LedgerError, match="already has a row"):
        ledger.append(1, "abc", rows.iloc[:3])
    with pytest.raises(LedgerError, match="written at"):
        ledger.append(1, "other", _rows(catalogue, 1, days=[date(2025, 3, 3)]))
    with pytest.raises(LedgerError, match="must carry run"):
        ledger.append(3, "abc", _rows(catalogue, 2))
    outside = _rows(catalogue, 4)
    outside.loc[0, "key"] = outside.loc[0, "key"].replace("hold_s=", "hold_s=9")
    with pytest.raises(ValueError, match="outside the catalogue"):
        ledger.append(4, "abc", outside)
    missing = _rows(catalogue, 4)
    missing.loc[0, "net_bp"] = np.nan
    with pytest.raises(LedgerError, match="writes 0"):
        ledger.append(4, "abc", missing)
    with pytest.raises(LedgerError, match="exactly the columns"):
        ledger.append(4, "abc", _rows(catalogue, 4).drop(columns="skips"))
    with pytest.raises(LedgerError, match="commit"):
        ledger.append(4, "", _rows(catalogue, 4))
    assert len(ledger.read()) == 60


def test_large_ledgers_split_into_parts_under_the_limit(
    tmp_path: Path, catalogue: Catalogue
) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue, part_bytes=12_000)
    many = [date(2025, 2, 1) + timedelta(days=i) for i in range(150)]
    rows = _rows(catalogue, 1, keys=12, days=many)
    record = ledger.append(1, "abc", rows)
    assert len(record["parts"]) > 1 and not ledger.path.exists()
    for part in record["parts"]:
        assert (tmp_path / part["file"]).stat().st_size < 12_000
    assert sum(p["rows"] for p in record["parts"]) == len(rows)
    pd.testing.assert_frame_equal(ledger.read(), rows, check_dtype=False)


def test_a_changed_part_or_another_catalogue_is_refused(
    tmp_path: Path, catalogue: Catalogue
) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    ledger.append(1, "abc", _rows(catalogue, 1))
    _rows(catalogue, 1, seed=9).to_parquet(ledger.path)
    with pytest.raises(LedgerError, match="sha256"):
        ledger.read()
    record = json.loads(ledger.sidecar_path.read_text())
    record["catalogue_sha256"] = "0" * 64
    ledger.sidecar_path.write_text(json.dumps(record))
    with pytest.raises(LedgerError, match="catalogue"):
        ledger.sidecar()


def test_hand_runs_are_logged(tmp_path: Path, catalogue: Catalogue) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    ledger.log_hand_run(date(2026, 11, 2), "IC of a feature on February", [DAYS[1], DAYS[0]])
    assert ledger.hand_runs == [
        {
            "date": "2026-11-02",
            "computed": "IC of a feature on February",
            "days": ["2025-02-01", "2025-02-02"],
        }
    ]
    with pytest.raises(LedgerError, match="names what"):
        ledger.log_hand_run(date(2026, 11, 2), " ", [])
    ledger.append(1, "abc", _rows(catalogue, 1))
    assert len(ledger.hand_runs) == 1
