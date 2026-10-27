"""The results page: one row per slot and measured row, the ladder, the sentences."""

from __future__ import annotations

import json
from pathlib import Path

from trading_research.spot import access as sa
from trading_research.spot import pages
from trading_research.spot import registration as sr
from trading_research.spot.context import SpotStudy
from trading_research.spot.costs import load_fee_table

ROOT = Path(__file__).resolve().parents[1]

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
    assert (
        "| S25-T on spot books (no refit) | 1.00 | -9.00 | vip5 -4.00 (requires Bybit VIP 5 status) |"
        in lines
    )
    assert "| Step | Zero fees (conditional on a zero-fee venue) |" in lines


def test_every_tier_is_labelled_and_one_not_dated_for_the_block_is_named_not_valued() -> None:
    rows = {"S25-PF": {"tiers": {"regular": -1.0, "vip1": None, "vip3": 3.0, "zero_fees": 9.0},
                       "volume_30_day_quote": 2_787_918.9}}  # fmt: skip
    results = {**RESULTS, "measurements": {**RESULTS["measurements"], "rows": rows}}
    labelled = (
        "regular -1.00 (verdict tier); vip3 3.00 (requires Binance VIP 3 status); "
        "zero fees 9.00 (conditional on a zero-fee venue) | 2.79M FDUSD |"
    )
    lines = "\n".join(pages.block_lines(results, load_fee_table(ROOT / sr.CONFIG_PATH)))
    assert labelled in lines and "vip1" not in lines.split("### Tiers")[0]
    assert "Not dated for H: Binance BTC/FDUSD's vip1; no sourced interval" in lines
    unknown = "\n".join(pages.block_lines(results))  # without the fee table nothing is left out
    assert "vip1 n/a (requires Binance VIP 1 status)" in unknown and "Not dated" not in unknown
    assert (
        pages.tier_label(sa.BINANCE, "vip2_to_vip9", (0.0,))
        == "conditional on Binance VIP 2 to 9 eligibility"
    )
    assert pages.tier_label(sa.BYBIT, "non_vip") == "verdict tier"
    assert pages.volume_cell("S25-T", None) == "n/a"


def test_the_page_is_written_from_the_results_files_to_the_same_bytes(tmp_path: Path) -> None:
    study = SpotStudy(tmp_path)
    assert pages.publish(study) == {"written": [], "sentences": {}}
    path = tmp_path / sr.RESULTS_PATHS["H"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(RESULTS))
    first = pages.publish(study)
    text = (tmp_path / pages.PAGE_PATH).read_text()
    assert pages.publish(study) == first and (tmp_path / pages.PAGE_PATH).read_text() == text
    assert pages.CAVEAT in text and "on any day of W, D or H" in pages.CAVEAT
    assert first["sentences"]["H"]["S25-PF"]
