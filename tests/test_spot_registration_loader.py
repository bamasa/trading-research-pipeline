"""The spot study's registration loader: hashes, placeholders, freezing, the
information set and the fee amendment's intervals."""

from __future__ import annotations

import hashlib
import socket
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from trading_research.market_making.prereg import (
    PreRegistrationError,
    placeholder_lines,
    unfrozen_text,
)
from trading_research.spot import access as sa
from trading_research.spot import registration as sr

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / sr.CONFIG_PATH).read_text(encoding="utf-8")
DOCUMENT = (ROOT / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
KEYS = tuple(i.key for i in sa.INSTRUMENTS.values())
CELL = "hold_s=50|trades_per_day=240|label_fee_bp=0"
#: A P interval as the fee amendment writes it, one line under each instrument.
P_LINES = {
    "binance_spot_BTCFDUSD": '      - {start: "2026-10-22T00:00:00Z", end: "2027-01-02T00:00:00Z", '
    'sources: [B5, B7], tiers: {regular: [0.0, 10.0]}, notes: "P"}',
    "bybit_spot_BTCUSDT": '      - {start: "2026-10-22T00:00:00Z", end: "2027-01-02T00:00:00Z", '
    'sources: [Y2], tiers: {non_vip: [10.0, 10.0]}, notes: "P"}',
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("tests do not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_values(**files: bytes) -> dict[tuple[str, ...], Any]:
    """A valid set of frozen values; ``files`` gives the frozen files' bytes."""
    weights = files.get("weights", b"weights")
    day = {"status": "on_disk", "stream": {"messages": 1}, "findings": {}, "passes": True}
    return {
        ("data", "reader_test_result"): {k: {d.isoformat(): day for d in sa.READER_TEST_DAYS} for k in KEYS},
        ("data", "max_sequence_gaps"): 3,
        ("data", "excluded_days_D"): {KEYS[1]: [[date(2026, 5, 3), "sequence gaps"]], KEYS[0]: []},
        ("data", "instrument_spec"): {KEYS[0]: {"tick": 0.01, "lot": 1e-05}, KEYS[1]: {"tick": 0.1, "lot": 1e-06}},
        ("common", "book_levels", "walk"): 10,
        ("features", "fitted_values_sha256"): sha(files.get("fitted", b"fitted")),
        ("models", "weights_sha256"): {"S25-PF__frozen.pkl": sha(weights)},
        ("models", "thresholds"): {"S25-PF": {CELL: {"theta_long": 0.6, "theta_short": 0.55}}},
        ("metrics", "mde", "sigma_d"): {"S25-PF": 4.0, "S25-BB": 3.0},
        ("metrics", "mde", "value"): {"S25-PF": 2.0, "S25-BB": 1.5},
        ("measurements", "result_D"): {"S25-PF": {"days": 61}},
        ("compute", "record"): {"trials": {"sha256": sha(files.get("trials", b"t")), "rows": 10}},
        ("amendment", "date"): date(2026, 11, 20),
        ("artefacts", "frozen_slot_files_sha256"): {"S25-PF": sha(files.get("slot", b"slot"))},
    }  # fmt: skip


def with_p_intervals(text: str) -> str:
    """The fee amendment: one P interval appended to each instrument's intervals."""
    lines = text.split("\n")
    for key, added in P_LINES.items():
        start = lines.index(f"    {key}:")
        end = start + 1
        while lines[end].startswith("      - "):
            end += 1
        lines.insert(end, added)
    return "\n".join(lines)


def test_the_registered_file_loads_at_the_registered_hash() -> None:
    registration = sr.Registration.load(repo=ROOT)
    assert sr.latest_null_sha256(DOCUMENT) == sr.REGISTERED_SHA256
    assert registration.base_sha256 == registration.sha256 and not registration.p_intervals
    assert tuple(placeholder_lines(TEXT)) == sr.PLACEHOLDERS
    assert registration.frozen is False
    registration.require_consistent()
    with pytest.raises(PreRegistrationError, match="still null"):
        registration.require_frozen()
    assert registration.h_blind  # the owner's amendment of 2026-10-25
    assert registration.max_sequence_gaps is None and registration.weights_sha256 == {}


def test_a_change_outside_the_placeholders_is_refused() -> None:
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        sr.Registration.from_text(TEXT.replace("clip_btc: 0.010", "clip_btc: 0.020"))
    with pytest.raises(PreRegistrationError, match="superseded"):
        sr.Registration.from_text(TEXT, null_sha256=sr.SUPERSEDED_SHA256[0])
    amended = DOCUMENT + f"\nNull-placeholder sha256: `{'a' * 64}`\n"
    assert sr.latest_null_sha256(amended) == "a" * 64
    with pytest.raises(PreRegistrationError, match="superseded"):
        sr.latest_null_sha256(
            DOCUMENT + f"\nNull-placeholder sha256: `{sr.SUPERSEDED_SHA256[1]}`\n"
        )


def test_freezing_writes_the_placeholders_and_nothing_else() -> None:
    frozen = sr.freeze_text(TEXT, frozen_values())
    registration = sr.Registration.from_text(frozen).require_frozen()
    assert unfrozen_text(frozen) == unfrozen_text(TEXT)
    assert registration.instrument_spec == {sa.BINANCE: (0.01, 1e-05), sa.BYBIT: (0.1, 1e-06)}
    assert registration.walk_levels == 10 and registration.max_sequence_gaps == 3
    assert registration.trials == {"sha256": sha(b"t"), "rows": 10}
    assert registration.mde == {"S25-PF": 2.0, "S25-BB": 1.5}
    assert sr.frozen_hash_line(frozen) == f"Frozen configuration sha256: `{registration.sha256}`"
    with pytest.raises(PreRegistrationError, match="already frozen"):
        sr.freeze_text(frozen, frozen_values())
    values = frozen_values()
    del values[("amendment", "date")]
    with pytest.raises(PreRegistrationError, match="missing"):
        sr.freeze_text(TEXT, values)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("common", "book_levels", "walk"), 50, "10 or 200"),
        (("models", "weights_sha256"), {"fit": "not a hash"}, "sha256"),
        (
            ("models", "thresholds"),
            {"S25-T": {CELL: {"theta_long": 1, "theta_short": 1}}},
            "outside",
        ),
        (("data", "excluded_days_D"), {KEYS[0]: [[date(2026, 8, 8), "x"]]}, "day of W or D"),
        (("compute", "record"), {"workers": 4}, "trials"),
        (("metrics", "mde", "value"), {"S25-PB": 1.0}, "outside"),
    ],
)
def test_frozen_values_are_checked(path: tuple[str, ...], value: Any, match: str) -> None:
    with pytest.raises(PreRegistrationError, match=match):
        sr.freeze_text(TEXT, {**frozen_values(), path: value})


def test_h_is_blind_only_if_every_confirmed_date_is_before_h() -> None:
    before, first = date(2025, 12, 31), sa.BLOCKS["H"][0]
    confirmed = {"s25_coordinates": before, KEYS[0]: before, KEYS[1]: sr.NEVER}
    assert sr.h_is_blind(confirmed)
    assert not sr.h_is_blind(None)  # no confirmation: the fallback holds
    assert not sr.h_is_blind({**confirmed, "s25_coordinates": first})
    assert sr.h_is_blind({**confirmed, KEYS[0]: date(2026, 8, 7)})
    assert sr.h_is_blind(dict.fromkeys(confirmed, sr.NEVER))
    with pytest.raises(PreRegistrationError, match="end dates"):
        sr.h_is_blind({"s25_coordinates": before})
    with pytest.raises(PreRegistrationError, match="a date or"):
        sr.h_is_blind({**confirmed, KEYS[1]: "later"})
    assert sr.with_fallback("killed", True) == "killed"
    assert sr.with_fallback("candidate", False) == "candidate (H inside S25's information set)"


def test_every_day_before_p_is_fee_covered_and_p_not_yet() -> None:
    registration = sr.Registration.from_text(TEXT)
    for name in sa.INSTRUMENTS:
        for block in ("reader_test", "W", "D", "H"):
            assert all(registration.fee_covered(name, d) for d in sa.block_days(block)), block
        assert not any(registration.fee_covered(name, d) for d in sa.block_days("P"))


def test_the_fee_amendment_adds_p_intervals_only_after_the_freeze() -> None:
    frozen = sr.freeze_text(TEXT, frozen_values())
    amended = sr.Registration.from_text(with_p_intervals(frozen))
    assert (
        amended.base_text == frozen
        and amended.base_sha256 == sr.Registration.from_text(frozen).sha256
    )
    assert [key for key, _ in amended.p_intervals] == list(P_LINES)
    assert all(amended.fee_covered(name, d) for name in sa.INSTRUMENTS for d in sa.block_days("P"))
    with pytest.raises(PreRegistrationError, match="still null"):
        sr.Registration.from_text(with_p_intervals(TEXT))  # before the D amendment
    early = with_p_intervals(frozen).replace('{start: "2026-10-22', '{start: "2026-10-21', 1)
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        sr.Registration.from_text(early)  # not a P interval: it changes the registered table
    no_tier = with_p_intervals(frozen).replace(
        "tiers: {regular: [0.0, 10.0]}", "tiers: {vip9: [0.0, 2.3]}"
    )
    with pytest.raises(PreRegistrationError, match="verdict tier"):
        sr.Registration.from_text(no_tier)
    lines = with_p_intervals(frozen).split("\n")
    lines.insert(lines.index(P_LINES[KEYS[1]]), P_LINES[KEYS[1]])
    with pytest.raises(PreRegistrationError, match="overlap"):
        sr.Registration.from_text("\n".join(lines))


def test_s25_t_binds_the_2025_files_by_hash() -> None:
    path = ROOT / sr.CONFIG_2025_PATH
    assert sha(path.read_bytes()) == sr.CONFIG_2025_SHA256
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert (
        frozen["artefacts"]["frozen_finalist_files_sha256"]["S25"]
        == sr.FROZEN_FILES_2025["slot_file_S25"]
    )
    assert frozen["features"]["fitted_values_sha256"] == sr.FROZEN_FILES_2025["fitted_values"]
    weights = frozen["models"]["weights_sha256"]
    assert all(weights[key] == digest for key, digest in sr.NEIGHBOURHOOD_FITS_2025.items())
    assert sr.S25_T_FIT in sr.NEIGHBOURHOOD_FITS_2025


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("family",), ["S25-BB", "S25-PF", "S25-T"], "family"),
        (("blocks", "H", "end"), date(2026, 10, 22), "block H"),
        (("information_set", "confirmed"), {"s25_coordinates": date(2025, 12, 31)}, "end dates"),
        (("statuses", "holm", "m"), 5, "Holm"),
        (("kill_conditions", "per_slot", "S25-PF", "kills"), ["K1"], "kills"),
        (("slots", "S25-T", "config_2025_sha256"), "0" * 64, "2025 configuration"),
        (("access", "ledger"), "elsewhere.json", "ledger"),
    ],
)
def test_the_registered_values_are_the_codes(path: tuple[str, ...], value: Any, match: str) -> None:
    tree = yaml.safe_load(TEXT)
    node = tree
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(PreRegistrationError, match=match):
        sr._validate_registered(tree)
