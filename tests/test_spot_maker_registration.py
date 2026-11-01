"""The maker study's registration: the YAML as registered, its constants and its freeze."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_research.market_making.prereg import PreRegistrationError, unfrozen_text
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.dry_run import registered_document

REPO = Path(__file__).resolve().parents[1]


def _texts() -> tuple[str, str]:
    """The registration as registered, however far the study's own files have run."""
    return (
        unfrozen_text((REPO / rg.CONFIG_PATH).read_text(encoding="utf-8")),
        registered_document((REPO / rg.DOCUMENT_PATH).read_text(encoding="utf-8")),
    )


def test_registered_configuration_loads_and_matches_the_constants() -> None:
    text, document = _texts()
    found = rg.Registration.from_text(text, document)
    assert rg.null_sha256(text) == rg.accepted_null_hash(document)
    assert tuple(found.config["family"]) == rg.FAMILY
    assert found.config["fees"]["verdict_tier"] == {"binance_spot_BTCFDUSD": "regular"}


def test_a_changed_value_or_a_wrong_hash_is_refused() -> None:
    text, document = _texts()
    with pytest.raises(PreRegistrationError):
        rg.Registration.from_text(text.replace("requote_s: 60\n", "requote_s: 45\n", 1), document)
    with pytest.raises(PreRegistrationError):
        rg.check_values({**__import__("yaml").safe_load(text), "family": ["MK-S"]})
    later = document + "\nNull-placeholder sha256: `" + "0" * 64 + "`\n"
    with pytest.raises(PreRegistrationError):
        rg.Registration.from_text(text, later)


def test_freeze_writes_every_placeholder_and_nothing_else() -> None:
    text, document = _texts()
    values = {path: {"k": 1} for path in rg.PLACEHOLDERS}
    values[("amendment", "date")] = "2026-10-28"
    frozen = rg.freeze_text(text, values)
    found = rg.Registration.from_text(frozen, document)
    assert found.frozen
    assert found.value(("compute", "record")) == {"k": 1}
    with pytest.raises(PreRegistrationError):
        rg.freeze_text(frozen, values)
    with pytest.raises(PreRegistrationError):
        rg.freeze_text(text, {rg.PLACEHOLDERS[0]: 1})


def test_amendment_hash_is_read_under_amendments_only() -> None:
    text, document = _texts()
    line = "Frozen configuration sha256: `" + "a" * 64 + "`"
    assert rg.Registration.from_text(text, document).amendment_hash() is None
    assert (
        rg.Registration.from_text(text, document + "\n" + line + "\n").amendment_hash() == "a" * 64
    )
    unit = rg.SLOTS["MK-SQ"].unit
    assert unit == pytest.approx(0.02) and len(rg.SLOTS["MK-Q"].cells()) == 9
