"""The results page: one row per slot and measured row, the ladder, the sentences."""

from __future__ import annotations

import json
from pathlib import Path

from trading_research.spot import pages
from trading_research.spot import registration as sr
from trading_research.spot.context import SpotStudy

RESULTS = {
    "block": "H", "stamp": "first read",
    "statuses": {"S25-PF": {"status": "killed", "mde": 1.2, "first_kill": "K1",
                            "kills": [{"name": "K1", "value": -1.0, "reference": 0.0}]},
                 "S25-BB": {"status": "inconclusive (too few trades)", "kills": [], "first_kill": None},
                 "S25-T": {"status": "not assessed (H unreadable on Bybit)", "kills": []}},
    "measurements": {
        "rows": {"S25-PF": {"mean": -1.0, "t": -2.0, "trades": 300, "tiers": {"regular": -1.0},
                            "brackets": {"stale": -2.0}, "fee_break_even_bp": 3.0}},
        "measured_rows_kills": {"S25-PB": [{"name": "K-label", "value": -0.1, "reference": 0.0,
                                            "fired": True}]},
        "sentences": {"S25-PF": ["S25's coordinates ... killed."], "P": []},
        "calibration": {"a_buy": 0.5, "a_sell": None},
        "edge_ladder": [{"label": "S25-T on spot books (no refit)", "zero_fees": 1.0, "verdict": -9.0,
                         "cheapest_tier": "vip5", "at_cheapest": -4.0}],
    },
}  # fmt: skip


def test_the_page_rows_and_ladder() -> None:
    lines = "\n".join(pages.block_lines(RESULTS))
    assert "| S25-PF | killed | K1 at -1.00 against 0.00 | -1.00 | -2.00 |" in lines
    assert (
        "| S25-PB | measured beside the family; no status | K-label at -0.10 against 0.00 |"
        in lines
    )
    assert "S25's coordinates ... killed." in lines and "a_sell n/a bp" in lines
    assert "| S25-T on spot books (no refit) | 1.00 | -9.00 | vip5 -4.00 |" in lines


def test_the_page_is_written_from_the_results_files_to_the_same_bytes(tmp_path: Path) -> None:
    study = SpotStudy(tmp_path)
    assert pages.publish(study) == {"written": [], "sentences": {}}
    path = tmp_path / sr.RESULTS_PATHS["H"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(RESULTS))
    first = pages.publish(study)
    text = (tmp_path / pages.PAGE_PATH).read_text()
    assert pages.publish(study) == first and (tmp_path / pages.PAGE_PATH).read_text() == text
    assert first["sentences"]["H"]["S25-PF"]
