"""The spot study's registration: its hash, placeholders, blocks and fee coverage."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from trading_research.market_making.prereg import placeholder_lines, sha256_bytes, unfrozen_text

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "configs" / "spot_2026.yaml").read_text(encoding="utf-8")
CONFIG: dict[str, Any] = yaml.safe_load(TEXT)
DOCUMENT = (ROOT / "docs" / "preregistration" / "spot_2026.md").read_text(encoding="utf-8")

#: The configuration as registered, every placeholder ``null``.
REGISTERED_SHA256 = "056035e10c51c88c8381ee7f27238a420ae3da4207b80ef3197e699a312fea6a"
PLACEHOLDERS = (
    ("data", "reader_test_result"), ("data", "max_sequence_gaps"), ("data", "excluded_days_D"),
    ("data", "instrument_spec"), ("features", "fitted_values_sha256"),
    ("models", "weights_sha256"), ("models", "thresholds"), ("metrics", "mde", "sigma_d"),
    ("metrics", "mde", "value"), ("measurements", "result_D"), ("compute", "record"),
    ("amendment", "date"), ("artefacts", "frozen_slot_files_sha256"),
)  # fmt: skip
INSTRUMENTS = tuple(CONFIG["instruments"])
BLOCKS = CONFIG["blocks"]


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _venues(block: dict[str, Any]) -> tuple[str, ...]:
    return INSTRUMENTS if block["venues"] == "both" else tuple(block["venues"])


def _covered(instrument: str, day: date) -> bool:
    """A day is covered when it lies wholly inside one sourced interval."""
    start = datetime.combine(day, time(), tzinfo=UTC)
    for interval in CONFIG["fees"]["intervals"][instrument]:
        lo = datetime.fromisoformat(interval["start"])
        hi = datetime.fromisoformat(interval["end"])
        if lo <= start and start + timedelta(days=1) <= hi:
            return True
    return False


def test_registered_hash_is_the_documents() -> None:
    assert sha256_bytes(unfrozen_text(TEXT).encode()) == REGISTERED_SHA256
    assert f"`{REGISTERED_SHA256}`" in DOCUMENT


def test_every_placeholder_is_null_and_listed() -> None:
    assert tuple(placeholder_lines(TEXT)) == PLACEHOLDERS
    assert unfrozen_text(TEXT) == TEXT


@pytest.mark.parametrize("name", ["W", "D", "E", "H", "Z", "P"])
def test_block_lengths(name: str) -> None:
    block = BLOCKS[name]
    assert len(_days(block["start"], block["end"])) == block["days"]


def test_blocks_follow_each_other_and_the_information_set() -> None:
    one = timedelta(days=1)
    assert BLOCKS["W"]["end"] + one == BLOCKS["D"]["start"]
    assert BLOCKS["D"]["end"] + one == BLOCKS["E"]["start"]
    assert BLOCKS["E"]["end"] + one == BLOCKS["H"]["start"]
    assert BLOCKS["P_embargo"] == [BLOCKS["H"]["end"] + one, BLOCKS["P"]["start"] - one]
    end = CONFIG["information_set"]["taken_to_end"]
    assert all(BLOCKS[name]["start"] > end for name in ("W", "D", "H", "P"))
    assert BLOCKS["Z"]["end"] <= end and BLOCKS["Z"]["venues"] == ["binance_spot_BTCFDUSD"]
    assert BLOCKS["D"]["start"] == CONFIG["information_set"]["first_scored_day"]
    assert BLOCKS["H"]["start"] == CONFIG["information_set"]["held_out_start"]
    data = CONFIG["data"]
    last = {data[k]["present"][1] for k in ("bybit_book", "bybit_prints", "binance_prints")}
    assert last == {BLOCKS["H"]["end"]}  # H ends on the last day published at registration


def test_reader_test_days_lie_in_no_block() -> None:
    blocks = [BLOCKS[name] for name in ("W", "D", "E", "H", "Z", "P")]
    used = {day for b in blocks for day in _days(b["start"], b["end"])}
    assert not used & (set(BLOCKS["reader_test"]["days"]) | set(BLOCKS["never_in_a_block"]))
    assert BLOCKS["reader_test"]["days"] == CONFIG["data"]["reader_test_days"]


def test_every_day_read_before_p_has_a_sourced_fee_and_p_none_yet() -> None:
    for name in ("W", "D", "H", "Z"):
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
            if tier in interval["tiers"]:
                assert interval["tiers"][tier][1] == 10.0
    assert CONFIG["kill_conditions"]["K-fee"]["taker_bp"] == 10.0


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
