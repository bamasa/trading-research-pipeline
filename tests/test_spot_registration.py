"""The spot study's registration: its hash, placeholders, blocks and fee coverage."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from trading_research.market_making.prereg import placeholder_lines, sha256_bytes, unfrozen_text

FALLBACK = "candidate (H inside S25's information set)"

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "configs" / "spot_2026.yaml").read_text(encoding="utf-8")
CONFIG: dict[str, Any] = yaml.safe_load(TEXT)
DOCUMENT = (ROOT / "docs" / "preregistration" / "spot_2026.md").read_text(encoding="utf-8")

#: The configuration as registered, every placeholder ``null``.
REGISTERED_SHA256 = "886fe1805b721994b7bf74357271aa41bb9a3f56129ad30a9eacb1736eb35451"
PLACEHOLDERS = (
    ("data", "reader_test_result"), ("data", "max_sequence_gaps"), ("data", "excluded_days_D"),
    ("data", "instrument_spec"), ("common", "book_levels", "walk"),
    ("features", "fitted_values_sha256"),
    ("models", "weights_sha256"), ("models", "thresholds"), ("metrics", "mde", "sigma_d"),
    ("metrics", "mde", "value"), ("measurements", "result_D"), ("compute", "record"),
    ("amendment", "date"), ("artefacts", "frozen_slot_files_sha256"),
)  # fmt: skip
INSTRUMENTS = tuple(CONFIG["instruments"])
BLOCKS = CONFIG["blocks"]
ALL = ("W", "D", "E", "H", "P")


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _venues(block: dict[str, Any]) -> tuple[str, ...]:
    return INSTRUMENTS if block["venues"] == "both" else tuple(block["venues"])


def _covered(instrument: str, day: date) -> bool:
    """A day is covered when every instant of it lies in a sourced interval with the verdict tier."""
    tier = CONFIG["fees"]["verdict_tier"][instrument]
    spans = sorted(
        (datetime.fromisoformat(i["start"]), datetime.fromisoformat(i["end"]))
        for i in CONFIG["fees"]["intervals"][instrument]
        if tier in i["tiers"]
    )
    reached = datetime.combine(day, time(), tzinfo=UTC)
    for lo, hi in spans:
        if lo <= reached < hi:
            reached = hi
    return reached >= datetime.combine(day, time(), tzinfo=UTC) + timedelta(days=1)


def test_registered_hash_is_the_documents() -> None:
    assert sha256_bytes(unfrozen_text(TEXT).encode()) == REGISTERED_SHA256
    assert f"`{REGISTERED_SHA256}`" in DOCUMENT


def test_every_placeholder_is_null_and_listed() -> None:
    assert tuple(placeholder_lines(TEXT)) == PLACEHOLDERS
    assert unfrozen_text(TEXT) == TEXT


@pytest.mark.parametrize("name", ["W", "D", "E", "H", "P"])
def test_block_lengths(name: str) -> None:
    block = BLOCKS[name]
    assert len(_days(block["start"], block["end"])) == block["days"]


def test_blocks_follow_each_other_and_the_information_set() -> None:
    one = timedelta(days=1)
    assert BLOCKS["W"]["end"] + one == BLOCKS["D"]["start"]
    assert BLOCKS["D"]["end"] + one == BLOCKS["E"]["start"]
    assert BLOCKS["E"]["end"] + one == BLOCKS["H"]["start"]
    assert BLOCKS["P_embargo"] == [BLOCKS["H"]["end"] + one, BLOCKS["P"]["start"] - one]
    info = CONFIG["information_set"]
    end = info["binance_spot_BTCFDUSD"]["taken_to_end"]
    assert all(BLOCKS[name]["start"] > end for name in ("W", "D", "H", "P"))
    # The 2025 registration confines a 2026 confirmation of S25 to dates after 2025-12-31.
    assert "Z" not in BLOCKS and all(BLOCKS[n]["start"] > date(2025, 12, 31) for n in ALL)
    assert BLOCKS["D"]["start"] == info["first_fitted_day"]
    first_test_month = BLOCKS["D"]["test_months"][0]
    assert info["first_scored_day"].isoformat() == f"{first_test_month}-01"
    assert BLOCKS["H"]["start"] == info["held_out_start"]


def h_is_blind() -> bool:
    """H is blind only when the owner confirmed all three end dates before H's first day
    ("never" for an instrument the author never looked at)."""
    confirmed = CONFIG["information_set"]["confirmed"]
    start = BLOCKS["H"]["start"]
    return confirmed is not None and all(d == "never" or d < start for d in confirmed.values())


def test_h_is_blind_only_under_a_confirmed_information_set() -> None:
    info = CONFIG["information_set"]
    named, start = info["s25_coordinates"]["named"], BLOCKS["H"]["start"]
    assert (named - start).days == info["s25_coordinates"]["h_days_elapsed_when_named"] == 61
    assert FALLBACK in CONFIG["statuses"]["after_H"] and FALLBACK in info["fallback"]
    assert FALLBACK in DOCUMENT
    if info["confirmed"] is None:
        assert not h_is_blind()  # as registered: the fallback holds
    else:
        assert set(info["confirmed"]) == {"s25_coordinates", *INSTRUMENTS}
        assert h_is_blind()  # the owner's amendment of 2026-10-25: every date before H
        assert "### 2026-10-25: the information set confirmed" in DOCUMENT
    data = CONFIG["data"]
    last = {data[k]["present"][1] for k in ("bybit_book", "bybit_prints", "binance_prints")}
    assert last == {BLOCKS["H"]["end"]}  # H ends on the last day published at registration


def test_reader_test_days_lie_in_no_block() -> None:
    blocks = [BLOCKS[name] for name in ALL]
    used = {day for b in blocks for day in _days(b["start"], b["end"])}
    assert not used & (set(BLOCKS["reader_test"]["days"]) | set(BLOCKS["never_in_a_block"]))
    assert BLOCKS["reader_test"]["days"] == CONFIG["data"]["reader_test_days"]


def test_every_day_read_before_p_has_a_sourced_fee_and_p_none_yet() -> None:
    for name in ("W", "D", "H"):
        block = BLOCKS[name]
        for day in _days(block["start"], block["end"]):
            assert all(_covered(i, day) for i in _venues(block)), (name, day)
    assert all(_covered(i, d) for d in BLOCKS["reader_test"]["days"] for i in INSTRUMENTS)
    days = _days(BLOCKS["P"]["start"], BLOCKS["P"]["end"])
    assert not any(_covered(i, day) for day in days for i in INSTRUMENTS)


def test_verdict_tier_is_ten_bp_taker_on_both_venues() -> None:
    fees = CONFIG["fees"]
    for instrument, tier in fees["verdict_tier"].items():
        for interval in fees["intervals"][instrument]:
            assert interval["tiers"][tier][1] == 10.0  # present in every interval, never zero
        undated = fees["undated_cells"].get(instrument, {"tiers": {}})
        assert tier not in undated["tiers"]  # an undated cell never sets the verdict
    assert CONFIG["kill_conditions"]["K-fee"]["taker_bp"] == 10.0


def test_s25_t_binds_the_2025_frozen_files_by_hash() -> None:
    path = ROOT / "configs" / "short_horizon_2025.yaml"
    slot = CONFIG["slots"]["S25-T"]
    assert sha256_bytes(path.read_bytes()) == slot["config_2025_sha256"]
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    finalists = frozen["artefacts"]["frozen_finalist_files_sha256"]
    assert slot["frozen_files_2025"]["slot_file_S25"] == finalists["S25"]
    assert slot["frozen_files_2025"]["fitted_values"] == frozen["features"]["fitted_values_sha256"]
    weights = frozen["models"]["weights_sha256"]
    assert all(weights[key] == digest for key, digest in slot["neighbourhood_fits_2025"].items())


def test_family_and_neighbourhoods() -> None:
    assert CONFIG["family"] == ["S25-PF", "S25-BB", "S25-T"]
    assert CONFIG["statuses"]["holm"]["m"] == len(CONFIG["family"])
    for slot in CONFIG["family"]:
        axes = CONFIG["slots"][slot]["neighbourhood"].values()
        assert len(axes) == 3 and [len(a) for a in axes] == [3, 2, 2]
    features = CONFIG["features"]
    selected = set(features["s25_selected"])
    prints, touch = set(features["print_members"]), set(features["touch_proxy_members"])
    assert len(selected) == 40 and prints <= selected and touch <= selected
    assert len(prints) + len(touch) + features["dropped_on_prints"] == len(selected)
