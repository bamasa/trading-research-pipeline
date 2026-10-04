"""The second round's registration as code: the YAML, the access, the ledger.

The loader must accept the committed file and refuse any change outside the
placeholders; the freeze must write those and nothing else; access must be per
instrument and block, D only for development; and the round-two ledger must
refuse every path to a held-out block the protocol forbids while recording the
reads it allows, in the order it allows them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from trading_research.market_making import prereg, round2, round2_run
from trading_research.market_making.prereg import HeldOutLocked, PreRegistrationError
from trading_research.market_making.round2 import (
    BASKET,
    PLACEHOLDERS,
    REGISTERED_SHA256,
    Access,
    Ledger,
    Registration,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "mm_prereg_round2.yaml"
DOCUMENT = ROOT / "docs" / "preregistration" / "market_making_round2.md"

ADMITTED = ("ALGOUSDT", "ZECUSDT")
NEAR = {"chosen": [2, 0.5, 4, 6], "cells": [[2, 0.5, 4, 6, 0.1]], "median": 0.1}
GATE = {"base": "S1_touch", "kind": "trailing", "window_min": 60, "margin_bp": 1}
MASKS = {0: [1, 2, 3], 1: [2], 2: []}

#: A valid value for every placeholder, two instruments admitted.
VALUES: dict[tuple[str, ...], Any] = {
    ("admission", "table"): [
        {
            "symbol": s,
            "spread_bp": 5.0,
            "criterion": "A1" if s in ADMITTED else "none",
            "admitted": s in ADMITTED,
        }
        for s in BASKET
    ],
    ("simulator", "clip", "notional_usdt"): {"ALGOUSDT": 12.5, "ZECUSDT": 30.25},
    ("simulator", "volatility", "sigma_ref"): {"ALGOUSDT": 6.1, "ZECUSDT": 7.5},
    ("strategies", "S1", "chosen"): {"ALGOUSDT": [2, 0.5, 4, 6], "ZECUSDT": [5, 0, 1, 3]},
    ("d_search", "S1", "neighbourhood"): {"ALGOUSDT": NEAR, "ZECUSDT": NEAR},
    ("d_search", "gate", "chosen"): GATE,
    ("d_search", "gate", "chosen_neighbourhood"): {"cells": [[*GATE.values(), 0.2]], "median": 0.2},
    ("d_search", "gate", "hours_masks"): {"ALGOUSDT": MASKS, "ZECUSDT": MASKS},
    ("d_search", "b3_choice", "chosen"): {
        "pro1_altcoin": {"strategy": "G"},
        "programme": {"strategy": "S1"},
    },
    ("d_search", "b4", "chosen"): {
        "base": "S1",
        "kind": "hours",
        "window_min": None,
        "margin_bp": 0,
    },
    ("d_search", "b4", "chosen_neighbourhood"): {
        "cells": [["S1", "hours", None, 0, 0.1]],
        "median": 0.1,
    },
    ("d_search", "b4", "hours_masks"): MASKS,
    ("metrics", "minimum_detectable_effect", "sigma_d"): {
        "B1": 1.2,
        "B2": {"net": 1.0, "minus_base": 0.5},
        "B3a": 1.1,
        "B3b": 1.1,
        "B4": {"net": 2.0, "minus_base": 1.0},
    },
    ("metrics", "minimum_detectable_effect", "value"): {"B1": 0.66, "B4": {"net": 0.75}},
    ("amendment", "date"): date(2026, 10, 4),
}


def none_admitted() -> dict[tuple[str, ...], Any]:
    """Valid values for a D table that admits no instrument."""
    values = dict(VALUES)
    values[("admission", "table")] = [{"symbol": s, "admitted": False} for s in BASKET]
    for path in (*PLACEHOLDERS[1:5], ("d_search", "gate", "hours_masks")):
        values[path] = {}
    for path in (
        ("d_search", "gate", "chosen"),
        ("d_search", "gate", "chosen_neighbourhood"),
        ("d_search", "b3_choice", "chosen"),
    ):
        values[path] = round2.NOT_TESTED
    return values


def registered_text() -> str:
    """The file as registered, whatever state the working copy is in."""
    return prereg.unfrozen_text(CONFIG.read_text(encoding="utf-8"))


# -- the YAML ------------------------------------------------------------------------


def test_the_committed_registration_validates_and_is_the_registered_one() -> None:
    registration = Registration.load(CONFIG).require_consistent()
    assert prereg.sha256_bytes(registered_text().encode()) == REGISTERED_SHA256
    assert set(prereg.placeholder_lines(registration.text)) == set(PLACEHOLDERS)
    assert len(PLACEHOLDERS) == 15
    recorded = prereg.FROZEN_HASH_PATTERN.findall(DOCUMENT.read_text(encoding="utf-8"))
    if registration.frozen:
        assert recorded == [registration.sha256]
    else:
        assert recorded == []
    for name, (start, end) in round2.BLOCKS.items():
        assert registration.value(("blocks", name, "start")) == start
        assert registration.value(("blocks", name, "end")) == end


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("skew_bp: [2, 5, 10]", "skew_bp: [2, 5, 11]"),
        ("    start: 2024-02-26", "    start: 2024-02-27"),
        ("min_prints_in_window: 20", "min_prints_in_window: 10"),
        ("window_min: [15, 60, 240]", "window_min: [15, 60, 120]"),
        ("pro1_altcoin: {maker_bp: 0.0", "pro1_altcoin: {maker_bp: -0.5"),
        ("# No code reads this file yet.", "# No code reads this file."),
    ],
)
def test_any_change_outside_the_placeholders_is_refused(old: str, new: str) -> None:
    text = registered_text()
    assert old in text
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        Registration.from_text(text.replace(old, new, 1))


def test_freezing_writes_the_placeholders_and_nothing_else() -> None:
    text = registered_text()
    frozen = round2.freeze_text(text, VALUES)
    registration = Registration.from_text(frozen).require_frozen()
    for path, value in VALUES.items():
        assert registration.value(path) == value
    assert prereg.unfrozen_text(frozen) == text
    changed = [a for a, b in zip(text.split("\n"), frozen.split("\n"), strict=True) if a != b]
    assert len(changed) == len(PLACEHOLDERS)
    assert registration.admitted == ADMITTED
    assert registration.clip("ZECUSDT") == 30.25 and registration.clip("BICOUSDT") == 20.0434
    assert registration.s1_cell("ALGOUSDT") == {
        "skew_bp": 2, "k": 0.5, "min_edge_bp": 4, "soft_limit_clips": 6
    }  # fmt: skip
    assert registration.gate_cell().label == "S1_touch/trailing/W=60/mu=1"
    assert registration.gate_cell("BICOUSDT").kind == "hours"
    assert registration.hours_masks("BICOUSDT") == {0.0: (1, 2, 3), 1.0: (2,), 2.0: ()}
    with pytest.raises(PreRegistrationError, match="already frozen"):
        round2.freeze_text(frozen, VALUES)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("strategies", "S1", "chosen"), {"ALGOUSDT": [3, 0.5, 4, 6], "ZECUSDT": [5, 0, 1, 3]},
         "not a cell"),
        (("simulator", "clip", "notional_usdt"), {"ALGOUSDT": 1.0, "GALAUSDT": 2.0}, "admitted"),
        (("simulator", "volatility", "sigma_ref"), {"ALGOUSDT": -1.0, "ZECUSDT": 2.0}, "positive"),
        (("d_search", "gate", "hours_masks"), {"ALGOUSDT": {0: [1]}, "ZECUSDT": MASKS}, "margins"),
        (("d_search", "b4", "hours_masks"), {0: [25], 1: [], 2: []}, "hours 0 to 23"),
        (("d_search", "gate", "chosen"), {**GATE, "window_min": 30}, "not a gate cell"),
        (("d_search", "b4", "chosen"), {**GATE, "kind": "hours"}, "not a gate cell"),
        (("d_search", "b3_choice", "chosen"), {"pro1_altcoin": {"strategy": "S2"},
                                               "programme": {"strategy": "G"}}, "S1 or G"),
        (("admission", "table"), [{"symbol": "ALGOUSDT", "admitted": True}], "one row per"),
        (("metrics", "minimum_detectable_effect", "sigma_d"), {"B5": 1.0}, "per primary"),
        (("metrics", "minimum_detectable_effect", "value"), {"B1": -0.1}, "non-negative"),
    ],
)  # fmt: skip
def test_a_frozen_value_of_the_wrong_kind_is_refused(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    with pytest.raises(PreRegistrationError, match=match):
        round2.freeze_text(registered_text(), {**VALUES, path: value})


def test_with_none_admitted_the_basket_values_are_empty_and_not_tested() -> None:
    values = none_admitted()
    registration = Registration.from_text(round2.freeze_text(registered_text(), values))
    assert registration.admitted == ()
    # Not tested is allowed only when nothing was admitted.
    with pytest.raises(PreRegistrationError, match="gate cell"):
        round2.freeze_text(
            registered_text(), {**VALUES, ("d_search", "gate", "chosen"): round2.NOT_TESTED}
        )


def test_freezing_needs_every_placeholder_and_a_partial_freeze_is_inconsistent() -> None:
    partial = dict(VALUES)
    partial.pop(("amendment", "date"))
    with pytest.raises(PreRegistrationError, match="missing"):
        round2.freeze_text(registered_text(), partial)
    one = prereg._set_placeholders(registered_text(), {("amendment", "date"): "2026-10-04"})
    registration = Registration.from_text(one)
    with pytest.raises(PreRegistrationError, match="partly frozen"):
        registration.require_consistent()


# -- access, per instrument and block -----------------------------------------------


def test_development_access_permits_d_for_the_round_and_nothing_else() -> None:
    access = round2.development_access()
    for symbol in round2.INSTRUMENTS:
        access.require(symbol, round2.block_days("D"))
    for block in ("H", "F", "P", "Q"):
        for symbol in round2.INSTRUMENTS:
            with pytest.raises(HeldOutLocked, match=f"in {block}"):
                access.require(symbol, [round2.block_days(block)[0]])
    with pytest.raises(HeldOutLocked):
        access.require("BTCUSDT", [round2.block_days("D")[0]])
    with pytest.raises(HeldOutLocked, match="no block"):
        access.require("GALAUSDT", [date(2024, 3, 10)])
    assert access.symbols("D") == list(round2.INSTRUMENTS) and access.symbols("H") == []
    guard = access.guard("ALICEUSDT")
    guard(round2.block_days("D"))
    with pytest.raises(HeldOutLocked):
        guard([round2.block_days("H")[0]])


def test_access_beyond_d_cannot_be_made_by_hand() -> None:
    with pytest.raises(HeldOutLocked, match=r"issued only by round2\.Ledger\.open"):
        Access(frozenset({("GALAUSDT", "H")}), "first read")
    with pytest.raises(ValueError, match="unknown"):
        Access(frozenset({("GALAUSDT", "Z")}), "development")


def test_the_readers_refuse_before_opening_anything(tmp_path: Path) -> None:
    access = round2.development_access()
    nothing = tmp_path / "nothing"
    h_day = round2.block_days("H")[0]
    with pytest.raises(HeldOutLocked):
        round2_run.markouts(
            "GALAUSDT", [h_day], access=access, book_roots=[nothing], trades_root=nothing,
            out=tmp_path / "out",
        )  # fmt: skip
    with pytest.raises(HeldOutLocked):
        round2_run.run_plans(
            [("BICOUSDT", [round2.block_days("P")[0]], [], 1)],
            access=access, book_roots=[nothing], trades_root=nothing, funding_root=None,
            out_root=tmp_path / "out",
        )  # fmt: skip
    with pytest.raises(TypeError, match=r"round2\.Access"):
        round2_run.screen(
            ["GALAUSDT"], access=prereg.development_access(), book_roots=[nothing],
            trades_root=nothing,
        )  # fmt: skip
    assert not (tmp_path / "out").exists()


# -- the ledger ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=ledger-test@localhost", "-c", "user.name=test",
         *args],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return done.stdout


def make_repo(
    tmp_path: Path,
    *,
    frozen: bool = True,
    record: str | None = "auto",
    values: dict[tuple[str, ...], Any] | None = None,
) -> Path:
    """A repository holding the round-two registration, frozen or not, and a
    document recording ``record`` as the frozen sha256 (the file's own by default)."""
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "docs" / "preregistration").mkdir(parents=True)
    text = registered_text()
    if frozen:
        text = round2.freeze_text(text, values or VALUES)
    (repo / round2.CONFIG_PATH).write_text(text, encoding="utf-8")
    document = "# Pre-registration, second round\n\n## Amendments\n"
    if record is not None:
        digest = prereg.sha256_bytes(text.encode()) if record == "auto" else record
        document += f"\n### Amendment 1\n\nFrozen configuration sha256: `{digest}`\n"
    (repo / round2.DOCUMENT_PATH).write_text(document, encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "registration")
    return repo


def commit_results(repo: Path) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "results")


@pytest.fixture
def need_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")


@pytest.mark.usefixtures("need_git")
@pytest.mark.parametrize(
    ("record", "frozen", "match"),
    [(None, True, "no amendment"), ("0" * 64, True, "changed after it was frozen"),
     ("auto", False, "not valid")],
)  # fmt: skip
def test_the_ledger_refuses_without_a_frozen_committed_configuration(
    tmp_path: Path, record: str | None, frozen: bool, match: str
) -> None:
    ledger = Ledger(make_repo(tmp_path, frozen=frozen, record=record))
    with pytest.raises(HeldOutLocked, match=match):
        ledger.open("first")
    assert not (ledger.repo / ledger.ledger).exists()


@pytest.mark.usefixtures("need_git")
def test_the_ledger_refuses_a_dirty_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("not committed", encoding="utf-8")
    with pytest.raises(HeldOutLocked, match="not clean"):
        Ledger(repo).open("first")
    (repo / "scratch.txt").unlink()
    path = repo / round2.CONFIG_PATH
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(HeldOutLocked, match="changed after it was frozen"):
        Ledger(repo).open("first")


@pytest.mark.usefixtures("need_git")
def test_the_first_read_opens_h_for_the_admitted_and_p_for_bico_once(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = Ledger(repo)
    commit = git(repo, "rev-parse", "HEAD").strip()
    access = ledger.open("first")
    assert access.stamp == "first read"
    for symbol in ADMITTED:
        access.require(symbol, round2.block_days("H") + round2.block_days("D"))
        with pytest.raises(HeldOutLocked):
            access.require(symbol, [round2.block_days("P")[0]])
    access.require("BICOUSDT", round2.block_days("P"))
    with pytest.raises(HeldOutLocked):
        access.require("BICOUSDT", [round2.block_days("H")[0]])
    with pytest.raises(HeldOutLocked):
        access.require("GALAUSDT", [round2.block_days("H")[0]])  # not admitted
    entries = json.loads((repo / round2.LEDGER_PATH).read_text(encoding="utf-8"))
    (entry,) = entries["first"]
    assert entry["commit"] == commit and entry["read"] == "first read"
    assert entry["config_sha256"] == prereg.sha256_bytes((repo / round2.CONFIG_PATH).read_bytes())
    assert sorted(map(tuple, entry["grants"])) == sorted(
        [("ALGOUSDT", "H"), ("BICOUSDT", "P"), ("ZECUSDT", "H")]
    )
    commit_results(repo)
    with pytest.raises(HeldOutLocked, match="second read"):
        ledger.open("first")
    again = ledger.open("first", force=True)
    assert again.stamp == "second read"
    entries = json.loads((repo / round2.LEDGER_PATH).read_text(encoding="utf-8"))
    assert [e["read"] for e in entries["first"]] == ["first read", "second read"]


@pytest.mark.usefixtures("need_git")
def test_p_for_the_basket_and_q_wait_for_the_reads_before_them(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = Ledger(repo)
    with pytest.raises(HeldOutLocked, match="'first' read, which is not recorded"):
        ledger.open("basket_P")
    with pytest.raises(HeldOutLocked, match="not recorded"):
        ledger.open("Q", symbols=["BICOUSDT"])
    ledger.open("first")
    with pytest.raises(HeldOutLocked, match="not clean"):
        ledger.open("basket_P")  # H's results, the ledger among them, are not committed
    commit_results(repo)
    with pytest.raises(HeldOutLocked, match="'basket_P' read"):
        ledger.open("Q", symbols=["ZECUSDT"])
    bico_q = ledger.open("Q", symbols=["BICOUSDT"])
    bico_q.require("BICOUSDT", round2.block_days("Q"))
    commit_results(repo)
    access = ledger.open("basket_P")
    access.require("ZECUSDT", round2.block_days("P"))
    with pytest.raises(HeldOutLocked):
        access.require("ZECUSDT", [round2.block_days("H")[0]])
    commit_results(repo)
    with pytest.raises(HeldOutLocked, match="were not admitted"):
        ledger.open("Q", symbols=["GALAUSDT"])
    ledger.open("Q", symbols=["ZECUSDT"]).require("ZECUSDT", round2.block_days("Q"))
    with pytest.raises(ValueError, match="guards"):
        ledger.open("H")


@pytest.mark.usefixtures("need_git")
def test_with_none_admitted_the_first_read_is_bico_alone(tmp_path: Path) -> None:
    values = none_admitted()
    repo = make_repo(tmp_path, values=values)
    access = Ledger(repo).open("first")
    assert access.symbols("H") == [] and access.symbols("P") == ["BICOUSDT"]
    commit_results(repo)
    with pytest.raises(HeldOutLocked, match="no instrument was admitted"):
        Ledger(repo).open("basket_P")
