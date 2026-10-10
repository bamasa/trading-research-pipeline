"""Simulator trials in the trial ledger: their keys, their rows and N_trials with them."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.search_engine.catalogue import Catalogue
from trading_research.search_engine.ledger import (
    LEDGER_COLUMNS,
    MEASUREMENT_TRIALS,
    LedgerError,
    TrialLedger,
    is_simulator_key,
    require_keys,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"
DAYS = [date(2025, 2, 3) + timedelta(days=i) for i in range(4)]
S25 = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return Catalogue.load(CONFIG)


def _frame(run: int, keys: list[str], value: float = 1.0) -> pd.DataFrame:
    grid = [(k, d) for k in keys for d in DAYS]
    n = len(grid)
    return pd.DataFrame(
        {"run": run, "key": [k for k, _ in grid], "day": [d for _, d in grid],
         "net_bp": np.full(n, value), "G_bp": 0.0, "pnl_long_bp": 0.0, "pnl_short_bp": 0.0,
         "trades": np.ones(n, dtype=np.int64), "skips": np.zeros(n, dtype=np.int64)}
    )[list(LEDGER_COLUMNS)]  # fmt: skip


def simulated() -> list[str]:
    return [f"S-P|{S25}|entry|fees=base", f"S25-Z|{S25}|entry|fees=zero",
            f"S-M|{S25}|maker|fees=base|lean=1|policy=pull_4h", *MEASUREMENT_TRIALS]  # fmt: skip


def test_simulator_keys_are_told_apart_from_catalogue_keys(catalogue: Catalogue) -> None:
    assert not is_simulator_key(S25) and S25 in catalogue
    assert all(is_simulator_key(k) for k in simulated())
    require_keys([S25, *simulated()], catalogue)


@pytest.mark.parametrize(
    "key",
    [
        "S-P|BTCUSDT|C|gbm|micro|open|hold_s=7|trades_per_day=240|label_fee_bp=0|regime=none|entry|fees=base",
        f"S-X|{S25}|entry|fees=base",
        f"S-P|{S25}|quoter|fees=base",
        f"S-M|{S25}|maker|fees=base",
        "FILL|on_touch|fees=zero",
        "BTCUSDT|C|gbm|micro|open|hold_s=7|trades_per_day=240|label_fee_bp=0|regime=none",
    ],
)
def test_a_key_that_is_neither_is_refused(catalogue: Catalogue, key: str) -> None:
    with pytest.raises(LedgerError if is_simulator_key(key) else Exception):
        require_keys([key], catalogue)


def test_simulator_trials_follow_the_run_and_count_in_n_trials(
    tmp_path: Path, catalogue: Catalogue
) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    taker = list(catalogue.keys[:5])
    with pytest.raises(LedgerError, match="no taker rows"):
        ledger.append_simulator(1, "abc", _frame(1, simulated()))
    ledger.append(1, "abc", _frame(1, taker))
    with pytest.raises(LedgerError, match="append_simulator"):
        ledger.append(1, "abc", _frame(1, simulated()))
    record = ledger.append_simulator(1, "def", _frame(1, simulated()))
    assert record["n_trials"] == 5 + len(simulated()) == ledger.n_trials
    assert record["n_trials_taker"] == 5 and record["n_trials_simulator"] == len(simulated())
    assert record["runs"]["1"]["commit"] == "abc"
    assert record["runs"]["1"]["simulator"] == {"commit": "def", "rows": 4 * len(simulated()),
                                                "keys": len(simulated())}  # fmt: skip
    assert set(ledger.run_rows(1)["key"]) == set(taker)
    assert set(ledger.run_rows(1, simulator=True)["key"]) == set(simulated())
    with pytest.raises(LedgerError, match="already"):
        ledger.append_simulator(1, "def", _frame(1, simulated()[:1]))


def test_a_superseded_runs_simulator_trials_still_count(
    tmp_path: Path, catalogue: Catalogue
) -> None:
    ledger = TrialLedger(tmp_path / "trials.parquet", catalogue)
    for run in (1, 2):
        ledger.append(run, "abc", _frame(run, list(catalogue.keys[:3])))
        ledger.append_simulator(run, "abc", _frame(run, simulated()[:2]))
    assert ledger.n_trials == 2 * (3 + 2)


def test_stage_c_rows_become_ledger_rows(catalogue: Catalogue) -> None:
    from trading_research.pipeline.short_horizon.passive import SimCell, trial_rows
    from trading_research.search_engine.ledger import check_rows

    entry = SimCell("S-P", S25, "entry", 50.0)
    maker = SimCell("S-M", S25, "maker", 50.0, lean=0.5, policy="pull_24h")
    rows = pd.DataFrame(
        {"label": [entry.label, entry.label, maker.label, maker.label],
         "day": ["2025-02-03", "2025-02-04", "2025-02-03", "2025-02-04"],
         "excluded": [False, True, False, False], "net_bp_trade": [3.0, 9.0, np.nan, np.nan],
         "net_usdt": [0.1, 0.2, 0.05, 0.0], "passive_notional": [0.0, 0.0, 50.0, 0.0],
         "trades": [2, 3, 1, 0]}
    )  # fmt: skip
    measured = [{"key": "A0|crossing|fees=zero", "day": "2025-02-04", "net_bp": -1.5, "trades": 7}]
    frame = check_rows(trial_rows(rows, 1, measured), catalogue)
    assert list(frame["key"]) == [entry.label, maker.label, maker.label, "A0|crossing|fees=zero"]
    assert frame["net_bp"].tolist() == [3.0, 10.0, 0.0, -1.5]
    assert frame["trades"].tolist() == [2, 1, 0, 7]
    assert (frame[["G_bp", "pnl_long_bp", "pnl_short_bp", "skips"]] == 0).all().all()
