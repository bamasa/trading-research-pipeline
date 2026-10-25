"""The run log: parts appended for D's test days, and the manifest the ledger checks."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from trading_research.spot import registration as sr
from trading_research.spot import runlog
from trading_research.spot.ledger import Ledger


def test_only_test_days_are_logged_and_parts_are_appended(tmp_path: Path) -> None:
    assert runlog.append(tmp_path, run="D1", stage="1", days=[date(2026, 5, 3)]) is None
    first = runlog.append(tmp_path, run="D1", stage="1", days=[date(2026, 6, 2), date(2026, 5, 3)])
    second = runlog.append(tmp_path, run="hand", stage="", days=[date(2026, 7, 9)], note="look")
    assert first is not None and second is not None and first != second
    rows = runlog.rows(tmp_path)
    assert rows["days"].tolist() == ["2026-06-02", "2026-07-09"]
    assert rows["run"].tolist() == ["D1", "hand"]


def test_the_manifest_is_what_the_ledger_verifies(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="empty"):
        runlog.manifest(tmp_path)
    runlog.append(tmp_path, run="D1", stage="D", days=[date(2026, 6, 5)])
    runlog.append(tmp_path, run="D1", stage="amend", days=[date(2026, 7, 5)])
    found = runlog.write_manifest(tmp_path)
    manifest = json.loads((tmp_path / sr.TRIALS_MANIFEST_PATH).read_text())
    assert found == {"sha256": manifest["sha256"], "rows": 2}
    assert [p["rows"] for p in manifest["parts"]] == [1, 1]

    class Frozen:
        trials = found

    ledger = Ledger(repo=tmp_path)
    ledger._committed_unchanged = lambda _: None  # type: ignore[method-assign]
    ledger.verify_trials(Frozen())  # type: ignore[arg-type]
    runlog.append(tmp_path, run="hand", stage="", days=[date(2026, 6, 9)])
    with pytest.raises(Exception, match="parts on disk"):
        ledger.verify_trials(Frozen())  # type: ignore[arg-type]
