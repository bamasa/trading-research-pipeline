"""The page: the verdict net of the fees charged, every kill, the measurements beside it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from test_spot_maker_statuses import _records
from trading_research.spot.state import plain
from trading_research.spot_maker import pages as pg
from trading_research.spot_maker import reads as rd
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import statuses as st


def _results() -> dict:
    values = {s: [-1.0, -2.0, 0.5] * 10 for s in rg.FAMILY}
    summaries = {s: st.summarise(_records(values), s) for s in rg.FAMILY}
    statuses = st.assess(summaries, "H", dict.fromkeys(rg.FAMILY, 0.4))
    found = {"block": "H", "read": "first read", "days": 30, "usable_days": 30, "excluded": [],
             "statuses": statuses, "summaries": summaries, "mde": dict.fromkeys(rg.FAMILY, 0.4),
             "placebo_percentile": {s: st.placebo_percentile(v) for s, v in summaries.items()},
             "headlines": st.headlines(summaries, statuses),
             "activity": {"prints_a_day": 1000.0, "inventory_pnl_bp_mean": 1.5}}  # fmt: skip
    loaded: dict = json.loads(json.dumps(plain(found)))
    return loaded


def test_every_slot_has_its_verdict_and_every_kill() -> None:
    text = pg.page({"H": _results()})
    for slot in rg.FAMILY:
        assert f"| {slot} | killed | K1 at -0.83 against 0.00 |" in text
    assert text.count("| MK-S | K") == len(rg.SLOTS["MK-S"].kills)
    assert text.count("| MK-Q | K") == len(rg.SLOTS["MK-Q"].kills)
    assert "zero-fee reference" in text and "Zero fees (reference)" in text
    assert "no round trips" in text and "## H (first read)" in text


def test_the_headlines_never_lead_with_the_zero_fee_number() -> None:
    results = _results()
    for line in results["headlines"]:
        assert "zero" not in line
    assert all(np.isfinite(results["summaries"][s]["mean"]) for s in rg.FAMILY)


def test_a_preset_status_and_the_page_on_disk(tmp_path: Path) -> None:
    preset = {"block": "H", "read": "first read", "statuses": rd.preset("not assessed (H read failed)"),
              "summaries": {}, "headlines": []}  # fmt: skip
    text = pg.page({"H": preset})
    assert text.count("not assessed (H read failed)") == 3 and "- none written" in text
    (tmp_path / rg.RESULTS_PATHS["H"]).parent.mkdir(parents=True)
    (tmp_path / rg.RESULTS_PATHS["H"]).write_text(json.dumps(_results()), encoding="utf-8")

    class _Study:
        repo = tmp_path

    path = pg.write_page(_Study())  # type: ignore[arg-type]
    assert path.read_text("utf-8").startswith("# S25-PF as a maker on Binance BTC/FDUSD, 2026")


def test_activity_counts_only_usable_days() -> None:
    records = [{"usable": True, "prints": 10, "inventory_pnl_bp": 2.0},
               {"usable": False}, {"usable": True, "prints": 30, "inventory_pnl_bp": -1.0}]  # fmt: skip
    found = rd.activity(records)
    assert found["prints_a_day"] == 20.0 and found["inventory_pnl_bp_mean"] == 0.5


def test_the_calibration_label_stands_beside_every_verdict_unless_it_passed() -> None:
    results = _results()
    assert pg.page({"H": results}).count("(uncalibrated fill rule)") == 3
    shares = {"days": 20, "ordering_share": 0.9, "bracket_share": 0.85, "passes": True}
    results["calibration"] = {"status": "run", "passes": True, "label": None, "thinned_to": 50_000,
                              **dict.fromkeys(rg.CALIBRATION_SLOTS, shares)}  # fmt: skip
    text = pg.page({"H": results})
    assert "uncalibrated" not in text and "it passes; no verdict carries a label" in text
