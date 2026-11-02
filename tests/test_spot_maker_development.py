"""Stage D's sigma_D and MDEs, the placeholders the amendment fills, and the ledger's helpers."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest

from test_spot_maker_statuses import _records
from trading_research.evaluation.significance import minimum_detectable_effect
from trading_research.spot_maker import development as dv
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.ledger import code_freeze_commit, run_log_digest


def test_sigma_and_mde_over_d_days_with_a_fill() -> None:
    values = {s: [1.0, 3.0, None] * 4 for s in rg.FAMILY}
    records = _records({s: [0.0] * 12 for s in rg.FAMILY})
    for i, record in enumerate(records):
        for slot in rg.FAMILY:
            record["slots"][slot]["value"] = values[slot][i]
    sigma, share, mde = dv.sigma_and_mde(records, "MK-S")
    assert sigma == pytest.approx(math.sqrt(8 / 7)) and share == pytest.approx(8 / 12)
    assert mde == pytest.approx(minimum_detectable_effect(sigma, 75 * 8 / 12))
    assert mde == pytest.approx((3 + 0.84) * sigma / math.sqrt(50))
    empty = _records({s: [0.0] for s in rg.FAMILY})
    empty[0]["slots"]["MK-Q"]["value"] = None
    assert dv.sigma_and_mde(empty, "MK-Q") == (0.0, 0.0, 0.0)


def test_the_amendment_fills_every_placeholder_once(tmp_path: Path) -> None:
    (tmp_path / rg.COPY_LOG_PATH).parent.mkdir(parents=True)
    (tmp_path / rg.COPY_LOG_PATH).write_text('{"files": {}}', encoding="utf-8")
    stage_0 = {"reader_test": {"2026-03-03": {"passes": True}}, "timings": {"copy_s": 1.0},
               "passes": True, "disk": {"stage": "0"}}  # fmt: skip
    stage_1 = {"excluded": [], "disk": {"stage": "1"}, "copies": {"days": 122, "absent": []}}
    d = {"run": "D1", "code_freeze": "abc", "disk": {"stage": "D"},
         "summaries": {s: {"mean": -1.0} for s in rg.FAMILY},
         "sigma_d": dict.fromkeys(rg.FAMILY, 1.0), "mde": dict.fromkeys(rg.FAMILY, 0.5),
         "rehearsal": {"statuses": dict.fromkeys(rg.FAMILY, "killed")},
         "calibration": {"status": "not run (no book day on disk)", "passes": False}}  # fmt: skip

    values = dv.amendment_values(stage_0, stage_1, d, tmp_path, date(2026, 10, 27))
    assert set(values) == set(rg.PLACEHOLDERS)
    record = values[("compute", "record")]
    assert record["code_freeze"] == {"commit": "abc"} and len(record["disk"]) == 3
    assert record["trials"] == {"sha256": None, "rows": 0}
    assert values[("measurements", "result_D")]["labelled"].startswith("development")
    assert values[("calibration", "result")]["status"].startswith("not run")
    assert dv.calibration_line(d["calibration"]).startswith("not run (no book day on disk)")


def test_the_note_and_the_run_log_digest(tmp_path: Path) -> None:
    note = "Code-freeze commit: `abc1234`\n\nCode-freeze commit: `def5678abc`\n"
    assert code_freeze_commit(note) == "def5678abc" and code_freeze_commit("none") is None
    (tmp_path / rg.RUN_LOG_PATH).parent.mkdir(parents=True)
    (tmp_path / rg.RUN_LOG_PATH).write_text(json.dumps([{"run": "D1"}]), encoding="utf-8")
    found = run_log_digest(tmp_path)
    assert found["rows"] == 1 and len(found["sha256"]) == 64
