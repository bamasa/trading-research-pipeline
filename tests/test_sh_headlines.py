"""The registered headline sentences: verbatim templates, and every outcome branch."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_research.market_making.verdicts import KillCheck
from trading_research.pipeline.regime import Attribution
from trading_research.reporting.headlines import (
    KILL_ORDER,
    TEMPLATES,
    HeadlineError,
    Outcomes,
    PassiveOutcome,
    RegimeAxisOutcome,
    S25Outcome,
    SlotOutcome,
    alternatives,
    break_sentence,
    check_status,
    fill,
    first_kill,
    has_held_out_status,
    headline_sentences,
    ordinal,
    passive_and_maker,
    pattern,
    payoff_on_d,
    regime_axis,
    regime_map_sentence,
    s25_sentence,
    search_on_d,
    slot_on_block,
)
from trading_research.reporting.regime_map import regime_maps
from trading_research.validation.placebo_search import SearchNullResult

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "configs" / "short_horizon_2025.yaml").read_text())


def registered(name: str, sentence: str) -> str:
    """The sentence, after checking its fixed text is the template's."""
    assert pattern(name).fullmatch(sentence), (name, sentence)
    return sentence


def test_the_templates_are_the_configurations_verbatim() -> None:
    written = CONFIG["artefacts"]["headline_sentences"]
    assert {k: " ".join(v.split()) for k, v in written.items()} == TEMPLATES


def test_the_templates_are_the_documents_verbatim() -> None:
    text = (ROOT / "docs" / "preregistration" / "short_horizon_2025.md").read_text()
    table = text.split("**Headline sentences, fixed now.**")[1].split("\n\n")[1]
    quoted = {m.group(1) for m in re.finditer(r'\| "(.+)" \|$', table, flags=re.MULTILINE)}
    assert quoted == set(TEMPLATES.values())


def test_the_kill_order_is_the_configurations() -> None:
    per_slot = CONFIG["kill_conditions"]["per_slot"]
    for slot, order in KILL_ORDER.items():
        names = [k.removesuffix(" with stale") for k in per_slot[slot]["kills"]]
        assert tuple(names) == order, slot


def test_fill_replaces_only_the_brackets() -> None:
    out = fill("regime_map", "S-C", "2", "5")
    assert out == registered("regime_map", out)
    assert out.startswith("Between detector alarms, S-C was positive in 2 of 5 segments;")
    with pytest.raises(HeadlineError):
        fill("regime_map", "S-C", "2")
    with pytest.raises(HeadlineError):
        fill("regime_map", "S-C", "", "5")
    assert alternatives("search_on_D") == ["was", "was not"]
    assert alternatives("break")[2] == "untestable (reason)"


def _null(stratum: str, a: float, count_a: int, count_b: int, cell: int | None = 1):
    p_a, p_b = (1 + count_a) / 1000, (1 + count_b) / 138
    return SearchNullResult(stratum, cell, a, count_a, count_b, p_a, p_b, None, False)


@pytest.mark.parametrize(
    ("count_a", "count_b", "runs", "word"),
    [(49, 5, 1, "was"), (50, 0, 1, "was not"), (0, 6, 1, "was not"), (24, 2, 2, "was"),
     (25, 2, 2, "was not")],
)  # fmt: skip
def test_the_search_sentence_says_was_only_at_p_search_of_five_percent(
    count_a: int, count_b: int, runs: int, word: str
) -> None:
    nulls = {"C": _null("C", 2.5, count_a, count_b), "R": _null("R", 1.0, 0, 0)}
    out = registered("search_on_D", search_on_d(2455, nulls, runs))
    assert f"edge {word} distinguishable" in out
    assert out.startswith("Searched 2,455 configurations (stratum, model, inputs,")
    assert out.endswith(f"; {runs} D runs).")
    assert f"Null A p = {(1 + count_a) / 1000:.3g}, Null B p = {(1 + count_b) / 138:.3g}" in out


def test_the_search_sentence_reads_the_stratum_with_the_larger_a() -> None:
    nulls = {"C": _null("C", 1.0, 0, 0), "R": _null("R", 3.0, 300, 40)}
    out = search_on_d(100, nulls)
    assert "was not distinguishable" in out and "Null A p = 0.301" in out
    empty = {"C": _null("C", 0.0, 0, 0, None), "R": _null("R", 0.0, 0, 0, None)}
    assert "(p_search = 1; Null A p = 1, Null B p = 1; 1 D runs)" in search_on_d(100, empty)
    with pytest.raises(HeadlineError):
        search_on_d(100, nulls, runs=0)


def test_the_payoff_sentence_for_no_slot_and_for_each_admitted_slot() -> None:
    assert payoff_on_d([]) == ["No configuration paid the 5.5 bp taker fee on development data."]
    out = payoff_on_d(["S-M", "S-C"])
    assert out == [
        "S-C was admitted on development data, selected.",
        "S-M was admitted on development data, selected.",
    ]
    for sentence in [*out, *payoff_on_d([])]:
        registered("payoff_on_D", sentence)
    with pytest.raises(HeadlineError):
        payoff_on_d(["S25"])


@pytest.mark.parametrize(
    "status",
    ["candidate", "candidate (H before 2025-08-01)", "not admitted on D (i, iv)",
     "not admitted on D (hand run)", "void (defect found after the F read)",
     "candidate, not confirmed (P unreadable)", "killed on F, assessed on all of F",
     "exploratory: significant on F, not confirmed (P unreadable)",
     "candidate (H before 2025-08-01), conditional on a zero-fee venue",
     "candidate; the break did not last", "not assessed (P read failed)", "same as S25"],
)  # fmt: skip
def test_registered_statuses_are_accepted(status: str) -> None:
    assert check_status(status) == status


@pytest.mark.parametrize(
    "status",
    ["passed", "not admitted on D (v)", "candidate (H unreadable)", "killed by K1", "Candidate",
     "candidate, F unreadable, extra", "not admitted on D ()"],
)  # fmt: skip
def test_other_statuses_are_refused(status: str) -> None:
    with pytest.raises(HeadlineError):
        check_status(status)


def test_ordinals() -> None:
    got = [ordinal(n) for n in (0, 1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 95, 100, 101, 111)]
    assert got == ["0th", "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
                   "23rd", "95th", "100th", "101st", "111th"]  # fmt: skip


def test_the_first_kill_follows_the_slots_order_not_the_inputs() -> None:
    kills = [
        KillCheck("x", "K-latency", "", -1.0, 0.0, True),
        KillCheck("x", "K-queue", "", -2.0, 0.0, True),
    ]
    assert first_kill("S-P", kills).name == "K-latency"  # type: ignore[union-attr]
    assert first_kill("S25-Z", kills).name == "K-queue"  # type: ignore[union-attr]
    assert first_kill("S25", [KillCheck("x", "K1", "", 1.0, 0.0, False)]) is None
    with pytest.raises(HeadlineError):
        first_kill("S25", [KillCheck("x", "K-dir", "", 1.0, 0.0, True)])


def test_break_sentences_for_every_outcome() -> None:
    assert break_sentence("supported") == "R: supported."
    assert break_sentence("not supported") == "R: not supported."
    assert break_sentence("void") == "R: not supported."
    assert break_sentence("not tested") == "R: not tested."
    out = break_sentence("untestable", "no alarm in F")
    assert registered("break", out) == "R: untestable (no alarm in F)."
    for bad in [("untestable", "it rained"), ("supported", "no alarm in F"), ("maybe", "")]:
        with pytest.raises(HeadlineError):
            break_sentence(*bad)


def _kill(name: str, value: float, reference: float = 0.0, fired: bool = True) -> KillCheck:
    return KillCheck("slot", name, "", value, reference, fired)


@pytest.mark.parametrize(
    ("slot", "kill", "tail"),
    [
        ("S25", _kill("K1", -3.214), "first kill K1 at -3.21 against 0.00 bp."),
        ("S-P", _kill("K1", -0.125), "first kill K1 at -0.12 against 0.00 per 100 USDT of clip a day."),
        ("S-C", _kill("K-days", 0.42, 0.5), "first kill K-days at 0.42 against 0.50 (share of days positive)."),
        ("S-R", _kill("K-label", -0.0123), "first kill K-label at -0.012 against 0.000 (rank IC)."),
        ("S-C", _kill("K-fee", 2.1, 5.5), "first kill K-fee at 2.10 against 5.50 bp per side."),
        ("S-M", _kill("K-dir", -0.8), "first kill K-dir at -0.80 against 0.00 bp per fill."),
        ("S25-Z", _kill("K2", -1.0), "first kill K2 at -1.00 against 0.00 bp."),
    ],
)  # fmt: skip
def test_a_killed_slot_names_its_first_kill_in_its_units(
    slot: str, kill: KillCheck, tail: str
) -> None:
    out = registered("slot_on_block", slot_on_block(slot, "H", "killed", [kill]))
    assert out == f"{slot} on H: killed; {tail}"


def test_a_slot_without_a_kill_keeps_the_whole_sentence() -> None:
    for status in ["candidate", "not admitted on D (i)", "inconclusive (too few trades)", "void"]:
        out = slot_on_block(
            "S-C", "H", status, [_kill("K-trades", 40, 100), _kill("K1", 1.0, fired=False)]
        )
        assert (
            registered("slot_on_block", out)
            == f"S-C on H: {status}; first kill none at n/a against n/a."
        )
    voided = slot_on_block("S-R", "F", "void", [_kill("K-leak", 1, 0), _kill("K-nbhd", -1.0)])
    assert voided.endswith("first kill K-nbhd at -1.00 against 0.00 bp.")
    assert slot_on_block("S-C", "P", "killed on P", [_kill("K1", -1.0)]).startswith(
        "S-C on P: killed on P;"
    )


@pytest.mark.parametrize(
    ("slot", "block", "status", "kills"),
    [("S-C", "H", "killed", []), ("S-C", "H", "candidate", [_kill("K1", -1.0)]),
     ("S-C", "H", "inconclusive", [_kill("K-drift", -1.0)]), ("S-X", "H", "killed", [_kill("K1", -1.0)]),
     ("S-C", "D", "killed", [_kill("K1", -1.0)]), ("S-C", "H", "killed off", [_kill("K1", -1.0)])],
)  # fmt: skip
def test_a_slot_sentence_that_contradicts_its_inputs_is_refused(
    slot: str, block: str, status: str, kills: list[KillCheck]
) -> None:
    with pytest.raises(HeadlineError):
        slot_on_block(slot, block, status, kills)


def test_the_s25_sentence() -> None:
    out = s25_sentence(3.456, 3.456, "candidate (H before 2025-08-01)")
    assert registered("S25", out) == (
        "S25's gross edge was 3.46 bp of turnover against the 5.5 bp taker fee; break-even "
        "3.46 bp per side; on H before 2025-08-01: candidate (H before 2025-08-01)."
    )
    assert s25_sentence(-1.0, -1.0, "killed").endswith("on H before 2025-08-01: killed.")
    for bad in ["candidate", "same as S25", "not admitted on D (i)"]:
        with pytest.raises(HeadlineError):
            s25_sentence(1.0, 1.0, bad)


def test_the_passive_sentence_on_d_and_on_h() -> None:
    same = passive_and_maker("S25", "not admitted on D (i)", "not admitted on D (i)", -0.404)
    assert registered("passive_and_maker", same) == (
        "Passive entry and the market maker on S25's signal: S-P and S-M not admitted on D (i); "
        "the market maker's making part on D was -0.40 bp per fill."
    )
    both = passive_and_maker("C*", "killed", "inconclusive", 0.7)
    assert (
        "on C*: S-P killed, S-M inconclusive; the market maker's making part on H was 0.70" in both
    )
    unread = passive_and_maker(
        "R*", "not assessed (H unreadable)", "not assessed (H unreadable)", 0.1
    )
    assert "making part on D was" in unread
    assert "making part on D was" in passive_and_maker("S25", "killed", "killed", 0.1, block="D")
    for args in [("S-C", "killed", "killed", 0.1, None), ("S25", "killed", "killed", 0.1, "F"),
                 ("S25", "killed", "won", 0.1, None)]:  # fmt: skip
        with pytest.raises(HeadlineError):
            passive_and_maker(*args)


def _attribution(difference: float, draws: np.ndarray) -> Attribution:
    return Attribution("pull_4h", difference, float(np.quantile(draws, 0.95)), draws, 7)


def test_the_regime_axis_sentence_did_and_did_not() -> None:
    draws = np.arange(200, dtype=float)
    above = regime_axis("S-C", 4, _attribution(195.0, draws))
    assert registered("regime_axis", above) == (
        "S-C on H: its 4 h guard after detector alarms did add more than random pauses "
        "(97th percentile of the random flag sets)."
    )
    below = regime_axis("S-R", 120, _attribution(20.5, draws))
    assert "its 120 h guard after detector alarms did not add" in below
    assert "(10th percentile of the random flag sets)" in below
    assert "(100th percentile" in regime_axis("S-P", 24, _attribution(1e6, draws))
    with pytest.raises(HeadlineError):
        regime_axis("S-C", 0, _attribution(1.0, draws))
    with pytest.raises(HeadlineError):
        regime_axis("S-C", 4, Attribution("pull_4h", 1.0, 3.0, draws, 7))


def test_the_regime_map_sentence_counts_positive_segments() -> None:
    days = pd.date_range("2025-02-01", periods=40, freq="D")
    daily = pd.Series(np.r_[np.ones(15), -np.ones(15), np.ones(10)], index=days)
    alarms = [
        pd.Timestamp("2025-02-15 12:00", tz="UTC"),
        pd.Timestamp("2025-03-02 03:00", tz="UTC"),
    ]
    segments = regime_maps({"S25": daily, "S-C": -daily}, alarms)
    assert registered("regime_map", regime_map_sentence("S25", segments)) == (
        "Between detector alarms, S25 was positive in 2 of 3 segments; the detector cut the "
        "segments, not the results."
    )
    assert "S-C was positive in 1 of 3 segments" in regime_map_sentence("S-C", segments)
    with pytest.raises(HeadlineError):
        regime_map_sentence("S-M", segments)


def test_nothing_computed_writes_nothing() -> None:
    out = headline_sentences(Outcomes())
    assert list(out) == list(TEMPLATES) and all(v == [] for v in out.values())


def test_every_sentence_at_once_in_the_tables_order() -> None:
    days = pd.date_range("2025-07-08", periods=20, freq="D")
    segments = regime_maps({"S25": pd.Series(np.linspace(-1, 1, 20), index=days)}, [])
    draws = np.arange(200, dtype=float)
    outcomes = Outcomes(
        n_trials=2455,
        nulls={"C": _null("C", 2.5, 10, 2), "R": _null("R", 0.5, 400, 60)},
        admitted=["S-C"],
        slots=[
            SlotOutcome("S25", "H", "killed", (_kill("K1", -2.0),)),
            SlotOutcome("S-C", "H", "candidate"),
            SlotOutcome("S-R", "H", "not admitted on D (i, iv)"),
        ],
        s25=S25Outcome(2.0, 2.0, "killed"),
        passive=PassiveOutcome("C*", "not admitted on D (i)", "not admitted on D (i)", -0.2),
        regime_axis=[RegimeAxisOutcome("S-C", 24, _attribution(150.0, draws))],
        regime_map=segments,
        r=("not tested", ""),
    )
    out = headline_sentences(outcomes)
    assert [len(v) for v in out.values()] == [1, 1, 3, 1, 1, 1, 1, 1]
    for name, sentences in out.items():
        for sentence in sentences:
            registered(name, sentence)
    assert out["slot_on_block"][1] == "S-C on H: candidate; first kill none at n/a against n/a."
    assert out["break"] == ["R: not tested."]


def test_a_status_on_h_is_one_not_set_on_d_and_not_unassessed() -> None:
    assert has_held_out_status("killed") and has_held_out_status("candidate (H before 2025-08-01)")
    assert not has_held_out_status("same as S25")
    assert not has_held_out_status("not run (no admissible cell on D)")
    assert not has_held_out_status("not assessed (H read failed)")
