"""The pre-registration as code: the YAML's schema, the freeze, and the ledger.

The loader must accept the committed file and refuse any change outside the
placeholders the amendment fills; the freeze must write those placeholders and
nothing else; and the held-out ledger must refuse every path to block H the
protocol forbids — no amendment, a changed or unfrozen configuration, a dirty
tree, a second read — while recording the one read it allows.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_research.market_making import prereg
from trading_research.market_making.prereg import (
    PLACEHOLDERS,
    REGISTERED_SHA256,
    Access,
    HeldOutLedger,
    HeldOutLocked,
    PreRegistration,
    PreRegistrationError,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "mm_prereg.yaml"
DOCUMENT = ROOT / "docs" / "preregistration" / "market_making.md"

NEAR = {"axes": ["skew_bp", "k"], "chosen": [5, 0.5], "cells": [[5, 0.5, 0.1]], "median": 0.1}

#: A valid value for every placeholder.
VALUES: dict[tuple[str, ...], Any] = {
    ("admission", "table"): [{"symbol": "BICOUSDT", "spread_bp": 6.1, "mm_admitted": True}],
    ("simulator", "clip", "notional_usdt"): {"BICOUSDT": 27.25},
    ("simulator", "volatility", "sigma_ref"): {"BICOUSDT": 6.3},
    ("strategies", "S1", "chosen", "skew_bp"): 5,
    ("strategies", "S1", "chosen", "k"): 0.5,
    ("strategies", "S1", "chosen", "min_edge_bp"): 1,
    ("strategies", "S1", "chosen", "soft_limit_clips"): 6,
    ("strategies", "S2", "chosen", "skew_bp"): 10,
    ("strategies", "S2", "chosen", "k"): 0,
    ("strategies", "S2", "chosen", "min_edge_bp"): 2,
    ("strategies", "S2", "chosen", "soft_limit_clips"): 3,
    ("strategies", "S2", "chosen", "m_ticks"): 4,
    ("strategies", "S3", "chosen", "guards"): "S2",
    ("strategies", "S3", "chosen", "action"): "widen",
    ("strategies", "S3", "chosen", "guard_window_min"): 60,
    ("strategies", "S4", "theta_bp"): {"BICOUSDT": 64.7, "CRVUSDT": 40.1},
    ("strategies", "S4", "beta"): {"BICOUSDT": -0.4, "CRVUSDT": 0.05},
    ("strategies", "S4", "chosen", "lambda"): 1,
    ("strategies", "S4", "chosen", "one_sided"): True,
    ("d_search", "neighbourhood", "S1"): NEAR,
    ("d_search", "neighbourhood", "S2"): NEAR,
    ("metrics", "minimum_detectable_effect", "sigma_d_usd_day"): 1.25,
    ("metrics", "minimum_detectable_effect", "value_usd_day"): 0.693,
    ("hypotheses", "H2.3", "gate_threshold"): -0.08,
    ("hypotheses", "H3", "flag_rate_on_d_per_day"): 5.2,
    ("amendment", "date"): date(2026, 10, 2),
}


def registered_text() -> str:
    """The file as registered, whatever state the working copy is in."""
    return prereg.unfrozen_text(CONFIG.read_text(encoding="utf-8"))


# -- the YAML ------------------------------------------------------------------


def test_the_committed_registration_validates_and_is_the_registered_one() -> None:
    registration = PreRegistration.load(CONFIG).require_consistent()
    assert prereg.sha256_bytes(registered_text().encode()) == REGISTERED_SHA256
    assert set(prereg.placeholder_lines(registration.text)) == set(PLACEHOLDERS)
    assert len(PLACEHOLDERS) == 26
    recorded = prereg.FROZEN_HASH_PATTERN.findall(DOCUMENT.read_text(encoding="utf-8"))
    if registration.frozen:
        # After the development amendment: the hash it records is the file's.
        assert recorded == [registration.sha256]
    else:
        assert recorded == []
    for name, (start, end) in prereg.BLOCKS.items():
        assert registration.value(("blocks", name, "start")) == start
        assert registration.value(("blocks", name, "end")) == end


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("skew_bp: [2, 5, 10]", "skew_bp: [2, 5, 11]"),
        ("    start: 2024-02-26", "    start: 2024-02-27"),
        ("keep_fraction: 0.34", "keep_fraction: 0.5"),
        ("# Units are in the key's suffix", "# Units live in the key's suffix"),
        ("  maker_bp: 2.0\n    taker_bp: 5.5", "  maker_bp: 1.0\n    taker_bp: 5.5"),
    ],
)
def test_any_change_outside_the_placeholders_is_refused(old: str, new: str) -> None:
    text = registered_text()
    assert old in text
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        PreRegistration.from_text(text.replace(old, new, 1))


def test_freezing_writes_the_placeholders_and_nothing_else() -> None:
    text = registered_text()
    frozen = prereg.freeze_text(text, VALUES)
    registration = PreRegistration.from_text(frozen)
    assert registration.frozen
    registration.require_frozen()
    for path, value in VALUES.items():
        assert registration.value(path) == value
    assert prereg.unfrozen_text(frozen) == text
    changed = [a for a, b in zip(text.split("\n"), frozen.split("\n"), strict=True) if a != b]
    assert len(changed) == len(PLACEHOLDERS)
    assert all(prereg.PLACEHOLDER_MARK in line for line in changed)
    assert registration.sha256 != REGISTERED_SHA256
    with pytest.raises(PreRegistrationError, match="already frozen"):
        prereg.freeze_text(frozen, VALUES)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("strategies", "S1", "chosen", "skew_bp"), 3, "not on its grid"),
        (("strategies", "S2", "chosen", "m_ticks"), 5, "not on its grid"),
        (("strategies", "S3", "chosen", "guards"), "S0", "guards S1 or S2"),
        (("strategies", "S3", "chosen", "guard_window_min"), 30, "not on its grid"),
        (("strategies", "S4", "chosen", "one_sided"), "yes", "not on its grid"),
        (("strategies", "S4", "theta_bp"), {"BICOUSDT": -1.0}, "positive"),
        (("simulator", "clip", "notional_usdt"), {"BICO": 1.0}, "not an instrument"),
        (("hypotheses", "H2.3", "gate_threshold"), 1.5, r"\[-1, 1\]"),
        (("hypotheses", "H3", "flag_rate_on_d_per_day"), -1.0, "negative"),
        (("d_search", "neighbourhood", "S1"), {"chosen": [1]}, "neighbourhood"),
    ],
)
def test_a_frozen_value_of_the_wrong_kind_is_refused(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    with pytest.raises(PreRegistrationError, match=match):
        prereg.freeze_text(registered_text(), {**VALUES, path: value})


def test_freezing_needs_every_placeholder_and_a_partial_freeze_is_inconsistent() -> None:
    text = registered_text()
    partial = dict(VALUES)
    partial.pop(("amendment", "date"))
    with pytest.raises(PreRegistrationError, match="missing"):
        prereg.freeze_text(text, partial)
    one = prereg._set_placeholders(text, {("strategies", "S1", "chosen", "k"): "0.5"})
    registration = PreRegistration.from_text(one)
    with pytest.raises(PreRegistrationError, match="partly frozen"):
        registration.require_consistent()
    with pytest.raises(PreRegistrationError, match="still null"):
        registration.require_frozen()


@pytest.mark.parametrize(
    "value",
    [5, 0.5, -0.4, True, "widen", date(2026, 10, 2), {"A": 1.5}, [{"symbol": "X", "a": 1}], NEAR],
)
def test_a_value_is_written_on_one_line_and_reads_back(value: Any) -> None:
    import yaml

    rendered = prereg.render_value(value)
    assert "\n" not in rendered
    assert yaml.safe_load(rendered) == value


# -- the blocks ------------------------------------------------------------------


def test_development_access_permits_d_and_nothing_else() -> None:
    access = prereg.development_access()
    access.require(prereg.block_days("D"))
    refused = [
        date(2024, 1, 31),
        *prereg.block_days("H"),
        *prereg.block_days("buffer"),
        *prereg.block_days("F"),
        *prereg.block_days("P")[:3],
    ]
    for day in refused:
        with pytest.raises(HeldOutLocked):
            access.require([day])
    assert prereg.block_of(date(2024, 2, 25)) == "D"
    assert prereg.block_of(date(2024, 2, 26)) == "H"
    assert len(prereg.block_days("D")) == 25 and len(prereg.block_days("H")) == 13


def test_held_out_access_cannot_be_made_by_hand() -> None:
    with pytest.raises(HeldOutLocked, match="issued only by HeldOutLedger"):
        Access(frozenset({"D", "H"}), "first read")


# -- the ledger ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=ledger-test@localhost",
            "-c",
            "user.name=test",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


def make_repo(tmp_path: Path, *, frozen: bool = True, record: str | None = "auto") -> Path:
    """A repository holding the registration, frozen or not, and a document
    that records ``record`` as the frozen sha256 (the file's own by default)."""
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "docs" / "preregistration").mkdir(parents=True)
    text = registered_text()
    if frozen:
        text = prereg.freeze_text(text, VALUES)
    (repo / "configs" / "mm_prereg.yaml").write_text(text, encoding="utf-8")
    digest = prereg.sha256_bytes(text.encode())
    document = "# Pre-registration\n\n## Amendments\n"
    if record is not None:
        hash_ = digest if record == "auto" else record
        document += f"\n### Amendment 2\n\nFrozen configuration sha256: `{hash_}`\n"
    (repo / "docs" / "preregistration" / "market_making.md").write_text(document, encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "registration")
    return repo


@pytest.fixture
def need_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")


@pytest.mark.usefixtures("need_git")
def test_the_ledger_refuses_without_an_amendment(tmp_path: Path) -> None:
    ledger = HeldOutLedger(make_repo(tmp_path, record=None))
    with pytest.raises(HeldOutLocked, match="no amendment"):
        ledger.open("H")


@pytest.mark.usefixtures("need_git")
def test_the_ledger_refuses_a_configuration_changed_after_the_freeze(tmp_path: Path) -> None:
    ledger = HeldOutLedger(make_repo(tmp_path, record="0" * 64))
    with pytest.raises(HeldOutLocked, match="changed after it was frozen"):
        ledger.open("H")
    assert not (ledger.repo / ledger.ledger).exists()


@pytest.mark.usefixtures("need_git")
def test_the_ledger_refuses_an_unfrozen_configuration(tmp_path: Path) -> None:
    ledger = HeldOutLedger(make_repo(tmp_path, frozen=False))
    with pytest.raises(HeldOutLocked, match="not valid"):
        ledger.open("H")


@pytest.mark.usefixtures("need_git")
def test_the_ledger_refuses_a_dirty_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("not committed", encoding="utf-8")
    with pytest.raises(HeldOutLocked, match="not clean"):
        HeldOutLedger(repo).open("H")
    (repo / "scratch.txt").unlink()
    path = repo / "configs" / "mm_prereg.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(HeldOutLocked, match="changed after it was frozen"):
        HeldOutLedger(repo).open("H")


@pytest.mark.usefixtures("need_git")
def test_the_ledger_records_the_first_read_and_refuses_a_second(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = HeldOutLedger(repo)
    commit = git(repo, "rev-parse", "HEAD").strip()
    access = ledger.open("H")
    assert access.stamp == "first read"
    access.require(prereg.block_days("H") + prereg.block_days("D"))
    with pytest.raises(HeldOutLocked):
        access.require([prereg.block_days("F")[0]])
    entries = json.loads((repo / prereg.LEDGER_PATH).read_text(encoding="utf-8"))
    (entry,) = entries["H"]
    assert entry["commit"] == commit
    assert entry["config_sha256"] == prereg.sha256_bytes(
        (repo / "configs" / "mm_prereg.yaml").read_bytes()
    )
    assert entry["read"] == "first read" and entry["utc"].endswith("Z")

    # The ledger itself is committed with H's results; a second read is refused.
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "H results")
    with pytest.raises(HeldOutLocked, match="second read"):
        ledger.open("H")
    again = ledger.open("H", force=True)
    assert again.stamp == "second read"
    entries = json.loads((repo / prereg.LEDGER_PATH).read_text(encoding="utf-8"))
    assert [e["read"] for e in entries["H"]] == ["first read", "second read"]


@pytest.mark.usefixtures("need_git")
def test_f_is_read_only_after_h_is_committed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = HeldOutLedger(repo)
    with pytest.raises(HeldOutLocked, match="only after H"):
        ledger.open("F")
    ledger.open("H")
    with pytest.raises(HeldOutLocked, match="not clean"):
        ledger.open("F")  # H's results, the ledger among them, are not committed yet
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "H results")
    access = ledger.open("F")
    access.require(prereg.block_days("F") + prereg.block_days("buffer"))
    with pytest.raises(ValueError, match="guards H and F"):
        ledger.open("D")


def test_simulating_a_held_out_day_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    from trading_research.market_making.quoters import TouchQuoter
    from trading_research.market_making.simulator import Cell, SimConfig, run_cells

    access = prereg.development_access()
    days = [date(2024, 2, 25), date(2024, 2, 25) + timedelta(days=1)]
    with pytest.raises(HeldOutLocked):
        run_cells(
            "BICOUSDT",
            days,
            [Cell("S0", TouchQuoter, SimConfig(clip_notional=1.0))],
            book_roots=[tmp_path / "nothing"],
            trades_root=tmp_path / "nothing",
            funding_root=None,
            day_guard=access.require,
        )
