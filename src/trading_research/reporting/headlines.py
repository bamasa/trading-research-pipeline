"""The registered headline sentences, filled from computed statuses and numbers.

The sentences are fixed in the registration (``artefacts.headline_sentences``
of ``configs/short_horizon_2025.yaml``); their bracketed parts are filled here
and the rest is never edited. Each template is kept verbatim in
:data:`TEMPLATES` (a test checks it against the configuration), and every
filler replaces the template's top-level brackets in order, so the fixed text
cannot drift. A bracket that offers alternatives (``[was / was not]``) is
filled with one of them, chosen by the registered rule, never by hand.

Choices the registration leaves open, made here:

* a bracket with nothing to report is filled with ``none`` (the kill) and
  ``n/a`` (its value and reference), so a slot with no kill keeps the whole
  sentence;
* ``[R_D] D runs`` is filled with the number, whatever its grammar;
* the payoff sentence is written once for each admitted slot;
* a void R is read as not supported, as the registration reads it;
* a percentile of the random flag sets is the share of sets whose difference
  is strictly below the slot's, in whole percent, rounded down.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from trading_research.market_making.verdicts import KillCheck
from trading_research.pipeline.regime import REGIME_WINDOWS_H, Attribution
from trading_research.validation.placebo_search import SearchNullResult, p_search

#: The registered sentences, verbatim.
TEMPLATES: dict[str, str] = {
    "search_on_D": (
        "Searched [N] configurations ([axes]) on 150 development days of BTCUSDT; on "
        "development data, selected, the best direction edge [was / was not] distinguishable "
        "from search noise (p_search = [x]; Null A p = [y], Null B p = [z]; [R_D] D runs)."
    ),
    "payoff_on_D": (
        "[No configuration paid the 5.5 bp taker fee on development data / S-[x] was admitted "
        "on development data, selected]."
    ),
    "slot_on_block": (
        "[slot] on [block]: [status]; first kill [K-x] at [value] against [reference]."
    ),
    "S25": (
        "S25's gross edge was [x] bp of turnover against the 5.5 bp taker fee; break-even [y] "
        "bp per side; on H before 2025-08-01: [status]."
    ),
    "passive_and_maker": (
        "Passive entry and the market maker on [signal]: [D status / H status]; the market "
        "maker's making part on [block] was [v] bp per fill."
    ),
    "regime_axis": (
        "[slot] on H: its [w] h guard after detector alarms [did / did not] add more than "
        "random pauses ([percentile] of the random flag sets)."
    ),
    "regime_map": (
        "Between detector alarms, [slot] was positive in [a] of [b] segments; the detector "
        "cut the segments, not the results."
    ),
    "break": "R: [supported / not supported / untestable (reason) / not tested].",
}

#: Each slot's kill conditions in the order of its row of the kill table.
_TAKER = ("K1", "K-label", "K-placebo", "K-drift", "K-nbhd", "K-days", "K-fee", "K-latency")
KILL_ORDER: dict[str, tuple[str, ...]] = {
    "S25": _TAKER,
    "S-C": _TAKER,
    "S-R": _TAKER,
    "S-P": (*_TAKER, "K-queue"),
    "S-M": (*_TAKER, "K-queue", "K-dir"),
    "S25-Z": (
        "K1", "K2", "K-label", "K-placebo", "K-drift", "K-nbhd", "K-days", "K-queue", "K-latency",
    ),
}  # fmt: skip
#: Conditions that are not kills: they void a result or set its status.
NOT_KILLS = ("K-leak", "K-flag-leak", "K-units", "K-acct", "K-cap", "K-trades", "K-MDE")
SLOTS = tuple(KILL_ORDER)
BLOCKS = ("H", "F", "P")
ALPHA = Fraction(1, 20)


class HeadlineError(ValueError):
    """An input the registered sentences cannot carry."""


def _groups(template: str) -> list[tuple[int, int]]:
    """The spans of the template's top-level brackets (nested ones inside)."""
    spans, depth, start = [], 0, 0
    for i, char in enumerate(template):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    if depth:
        raise HeadlineError(f"unbalanced brackets in {template!r}")
    return spans


def fill(name: str, *values: str) -> str:
    """Template ``name`` with its top-level brackets replaced by ``values``."""
    template = TEMPLATES[name]
    spans = _groups(template)
    if len(values) != len(spans):
        raise HeadlineError(f"{name} has {len(spans)} brackets, {len(values)} values given")
    out, last = [], 0
    for (start, end), value in zip(spans, values, strict=True):
        if not value:
            raise HeadlineError(f"an empty fill in {name}")
        out += [template[last:start], value]
        last = end
    return "".join([*out, template[last:]])


def alternatives(name: str, group: int = 0) -> list[str]:
    """The alternatives a bracket offers, as the template writes them."""
    template = TEMPLATES[name]
    start, end = [s for s in _groups(template) if "/" in template[s[0] : s[1]]][group]
    return [part.strip() for part in template[start + 1 : end - 1].split(" / ")]


def pattern(name: str) -> re.Pattern[str]:
    """A pattern every filled sentence of ``name`` matches."""
    template = TEMPLATES[name]
    parts, last = [], 0
    for start, end in _groups(template):
        parts += [re.escape(template[last:start]), "(.+)"]
        last = end
    return re.compile("".join([*parts, re.escape(template[last:])]))


# ---------------------------------------------------------------------------
# Registered statuses
# ---------------------------------------------------------------------------

_CONDITIONS = r"(?:i|ii|iii|iv)(?:, (?:i|ii|iii|iv))*"
#: Every status the registration lists (``statuses.outcomes``), as patterns.
STATUS_PATTERNS: tuple[str, ...] = (
    r"candidate",
    r"candidate \(H before 2025-08-01\)",
    r"inconclusive",
    r"inconclusive \(too few trades\)",
    r"inconclusive, below the minimum detectable effect",
    r"killed",
    r"void",
    r"not run \(no admissible cell on D\)",
    rf"not admitted on D \((?:hand run|{_CONDITIONS})\)",
    r"same as S25",
    r"not assessed \(H unreadable\)",
    r"not assessed \([HFP] read failed\)",
    r"void \(defect found after the [HFP] read\)",
    r"confirmed",
    r"candidate, not confirmed",
    r"confirmed until the break",
    r"candidate; the break did not last",
    r"confirmed on P, first significant on F",
    r"exploratory: significant on F",
    r"exploratory: significant on F, not confirmed",
    r"killed on F",
    r"killed on P",
)
#: Marks a status may carry, after a comma or in parentheses.
MARKS: tuple[str, ...] = (
    "assessed on all of F",
    "F unreadable",
    "P unreadable",
    "F read failed",
    "P read failed",
    "conditional on a zero-fee venue",
)
#: Statuses set on D: a slot with one has no status on a held-out block.
D_STATUS = re.compile(r"not admitted on D \(.*\)|not run \(no admissible cell on D\)|same as S25")
_STATUS = re.compile("|".join(f"(?:{p})" for p in STATUS_PATTERNS))


def check_status(status: str) -> str:
    """``status`` if it is a registered status, with registered marks only."""
    base = status
    while not _STATUS.fullmatch(base):
        suffixes = [s for m in MARKS for s in (f", {m}", f" ({m})") if base.endswith(s)]
        if not suffixes:
            raise HeadlineError(f"not a registered status: {status!r}")
        base = base[: -len(suffixes[0])]
    return status


def has_held_out_status(status: str) -> bool:
    """Whether a slot's status after H was assessed on H (not set on D, and
    not ``not assessed``)."""
    check_status(status)
    return not (D_STATUS.fullmatch(status) or status.startswith("not assessed"))


# ---------------------------------------------------------------------------
# The fillers, one per registered sentence
# ---------------------------------------------------------------------------

#: The searched axes, as the search sentence names them.
SEARCHED_AXES = (
    "stratum, model, inputs, label fee, hold, decision threshold, market gate, regime policy"
)
SEARCHED_SLOTS = ("S-C", "S-R", "S-P", "S-M")
#: Units of a kill's value and reference: the slot's own unless named here.
KILL_UNITS = {
    "K-days": " (share of days positive)",
    "K-label": " (rank IC)",
    "K-fee": " bp per side",
    "K-dir": " bp per fill",
}
BP_SLOTS = ("S25", "S-C", "S-R", "S25-Z")
_KILL_FREE = re.compile(
    r"candidate|confirmed|exploratory|inconclusive|not admitted|not run|same as|not assessed"
)


def _p(value: float) -> str:
    return f"{value:.3g}"


def _bp(value: float) -> str:
    return f"{value:.2f}"


def search_on_d(
    n_trials: int,
    results: Mapping[str, SearchNullResult],
    runs: int = 1,
    axes: str = SEARCHED_AXES,
) -> str:
    """The search sentence. The p-values are those of the stratum with the
    larger A, as ``placebo_search.p_search`` takes them; "was" when p_search
    is at most 0.05, decided in exact fractions."""
    if runs < 1 or n_trials < 1:
        raise HeadlineError(f"N_trials {n_trials} and R_D {runs} must be positive")
    chosen = [r for r in results.values() if r.cell is not None]
    best = max(chosen, key=lambda r: r.a) if chosen else None
    p_a, p_b = (best.p_a, best.p_b) if best else (1.0, 1.0)
    was = best is not None and Fraction(best.p).limit_denominator(10**7) * runs <= ALPHA
    word = alternatives("search_on_D")[0 if was else 1]
    found = p_search(results, runs)
    return fill("search_on_D", f"{n_trials:,}", axes, word, _p(found), _p(p_a), _p(p_b), str(runs))


def payoff_on_d(admitted: Sequence[str]) -> list[str]:
    """One sentence: no configuration paid; or one per admitted searched slot."""
    unknown = set(admitted) - set(SEARCHED_SLOTS)
    if unknown:
        raise HeadlineError(f"not a searched slot: {sorted(unknown)}")
    nothing, one = alternatives("payoff_on_D")
    if not admitted:
        return [fill("payoff_on_D", nothing)]
    slots = [s for s in SEARCHED_SLOTS if s in set(admitted)]
    return [fill("payoff_on_D", one.replace("[x]", slot.removeprefix("S-"))) for slot in slots]


def first_kill(slot: str, kills: Sequence[KillCheck]) -> KillCheck | None:
    """The first kill that fired, in the order of the slot's kill list."""
    order = KILL_ORDER[slot]
    unknown = {k.name for k in kills} - set(order) - set(NOT_KILLS)
    if unknown:
        raise HeadlineError(f"{slot} has no kill condition {sorted(unknown)}")
    fired = [k for k in kills if k.fired and k.name in order]
    return min(fired, key=lambda k: order.index(k.name)) if fired else None


def slot_on_block(slot: str, block: str, status: str, kills: Sequence[KillCheck] = ()) -> str:
    """One slot's sentence on a held-out block; ``kills`` are its checks there."""
    if slot not in SLOTS or block not in BLOCKS:
        raise HeadlineError(f"slot {slot!r} on block {block!r} is not registered")
    check_status(status)
    kill = first_kill(slot, kills)
    if status.startswith("killed") and kill is None:
        raise HeadlineError(f"{slot} on {block} is {status!r} with no kill fired")
    if kill is not None and _KILL_FREE.match(status):
        raise HeadlineError(f"{slot} on {block} is {status!r} but {kill.name} fired")
    if kill is None:
        return fill("slot_on_block", slot, block, status, "none", "n/a", "n/a")
    digits = 3 if kill.name == "K-label" else 2
    unit = KILL_UNITS.get(kill.name, " bp" if slot in BP_SLOTS else " per 100 USDT of clip a day")
    value, reference = f"{kill.value:.{digits}f}", f"{kill.reference:.{digits}f}{unit}"
    return fill("slot_on_block", slot, block, status, kill.name, value, reference)


def s25_sentence(gross_bp_turnover: float, break_even_bp: float, status: str) -> str:
    """S25's sentence: its gross edge, its break-even per side, and its status
    on H before 2025-08-01, the only status it can take."""
    check_status(status)
    if status == "candidate" or D_STATUS.fullmatch(status):
        raise HeadlineError(f"S25's status cannot be {status!r}")
    return fill("S25", _bp(gross_bp_turnover), _bp(break_even_bp), status)


#: How the passive sentence names the signal S-P and S-M run on.
SIGNALS = {"S25": "S25's signal", "C*": "C*", "R*": "R*"}


def passive_and_maker(
    signal: str,
    s_p_status: str,
    s_m_status: str,
    making_part_bp: float,
    block: str | None = None,
) -> str:
    """S-P's and S-M's statuses (each the one it has: set on D, or on H), and
    S-M's making part, in bp per fill, on H when S-M has a status on H, else
    on D (``block`` overrides only with one of the two)."""
    if signal not in SIGNALS:
        raise HeadlineError(f"the signal is one of {sorted(SIGNALS)}, got {signal!r}")
    check_status(s_p_status)
    where = block or ("H" if has_held_out_status(s_m_status) else "D")
    if where not in ("D", "H"):
        raise HeadlineError(f"the making part is on D or H, got {where!r}")
    if s_p_status == s_m_status:
        statuses = f"S-P and S-M {s_p_status}"
    else:
        statuses = f"S-P {s_p_status}, S-M {s_m_status}"
    return fill("passive_and_maker", SIGNALS[signal], statuses, where, _bp(making_part_bp))


def percentile_of(value: float, draws: Sequence[float] | np.ndarray) -> int:
    """The share of ``draws`` strictly below ``value``, in whole percent, down."""
    array = np.asarray(draws, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or not math.isfinite(value):
        raise HeadlineError("a percentile needs finite draws and a finite value")
    return math.floor(100 * int((array < value).sum()) / len(array))


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``4th``, ``11th``, ``21st``, ..."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def regime_axis(slot: str, window_h: int, attribution: Attribution) -> str:
    """A guarded slot against random pauses on H: "did" when its difference
    from the window-0 twin is above the 95th percentile of the random flag
    sets (``Attribution.exceeds``)."""
    if slot not in SLOTS or window_h not in REGIME_WINDOWS_H:
        raise HeadlineError(f"{slot!r} with a {window_h} h window has no regime sentence")
    draws = np.asarray(attribution.draws, dtype=np.float64)
    if not len(draws) or not np.isclose(attribution.reference, np.quantile(draws, 0.95)):
        raise HeadlineError("the reference is not the 95th percentile of the random flag sets")
    did = alternatives("regime_axis")[0 if attribution.exceeds else 1]
    rank = ordinal(percentile_of(attribution.difference, attribution.draws))
    return fill("regime_axis", slot, f"{window_h:g}", did, f"{rank} percentile")


def regime_map_sentence(slot: str, segments: pd.DataFrame) -> str:
    """How many of the slot's regime-map segments (``reporting.regime_map``)
    have a positive mean."""
    mine = segments.loc[segments["slot"] == slot]
    if mine.empty:
        raise HeadlineError(f"{slot!r} has no segment on the regime map")
    positive = int((mine["mean"].to_numpy(dtype=np.float64) > 0).sum())
    return fill("regime_map", slot, str(positive), str(len(mine)))


#: Why R may be untestable, as the registration lists it.
UNTESTABLE_REASONS = (
    "no alarm in F",
    "fewer than 20 usable days before b",
    "fewer than 10 usable days after b",
    "alarm rate on D above one per 14 days",
    "fewer than 19 days in the reference set besides b",
)


def break_sentence(outcome: str, reason: str = "") -> str:
    """R's sentence. A void R reads as not supported; an untestable one names
    its registered reason."""
    supported, not_supported, untestable, not_tested = alternatives("break")
    if outcome == "untestable":
        if reason not in UNTESTABLE_REASONS:
            raise HeadlineError(f"not a registered reason R is untestable: {reason!r}")
        return fill("break", untestable.replace("(reason)", f"({reason})"))
    if reason:
        raise HeadlineError(f"a reason is given only when R is untestable, not {outcome!r}")
    chosen = {"supported": supported, "not supported": not_supported, "void": not_supported}
    chosen["not tested"] = not_tested
    if outcome not in chosen:
        raise HeadlineError(f"R is supported, not supported, untestable or not tested: {outcome!r}")
    return fill("break", chosen[outcome])


# ---------------------------------------------------------------------------
# Every sentence at once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotOutcome:
    """A slot's status on a held-out block and its kill checks there."""

    slot: str
    block: str
    status: str
    kills: tuple[KillCheck, ...] = ()


@dataclass(frozen=True)
class S25Outcome:
    """S25's gross edge and break-even per side, and its status on H's window."""

    gross_bp_turnover: float
    break_even_bp: float
    status: str


@dataclass(frozen=True)
class PassiveOutcome:
    """S-P's and S-M's signal and statuses, and S-M's making part in bp per fill."""

    signal: str
    s_p_status: str
    s_m_status: str
    making_part_bp: float
    block: str | None = None


@dataclass(frozen=True)
class RegimeAxisOutcome:
    """A slot with a regime window above 0, against random pauses on H."""

    slot: str
    window_h: int
    attribution: Attribution


@dataclass(frozen=True)
class Outcomes:
    """Whatever has been computed so far; a sentence whose inputs are missing
    is not written (its list stays empty)."""

    n_trials: int | None = None
    nulls: Mapping[str, SearchNullResult] | None = None
    runs: int = 1
    admitted: Sequence[str] | None = None
    slots: Sequence[SlotOutcome] = ()
    s25: S25Outcome | None = None
    passive: PassiveOutcome | None = None
    regime_axis: Sequence[RegimeAxisOutcome] = ()
    regime_map: pd.DataFrame | None = None
    regime_map_slots: Sequence[str] | None = None
    r: tuple[str, str] | None = None
    axes: str = SEARCHED_AXES


def headline_sentences(outcomes: Outcomes) -> dict[str, list[str]]:
    """Every registered sentence the outcomes can fill, in the table's order,
    keyed as ``TEMPLATES``. The slot sentences follow the order given; the
    regime map's, the order of first appearance on the map unless
    ``regime_map_slots`` names them."""
    out: dict[str, list[str]] = {name: [] for name in TEMPLATES}
    if outcomes.n_trials is not None and outcomes.nulls is not None:
        out["search_on_D"] = [
            search_on_d(outcomes.n_trials, outcomes.nulls, outcomes.runs, outcomes.axes)
        ]
    if outcomes.admitted is not None:
        out["payoff_on_D"] = payoff_on_d(outcomes.admitted)
    out["slot_on_block"] = [
        slot_on_block(s.slot, s.block, s.status, s.kills) for s in outcomes.slots
    ]
    if outcomes.s25 is not None:
        s = outcomes.s25
        out["S25"] = [s25_sentence(s.gross_bp_turnover, s.break_even_bp, s.status)]
    if outcomes.passive is not None:
        p = outcomes.passive
        out["passive_and_maker"] = [
            passive_and_maker(p.signal, p.s_p_status, p.s_m_status, p.making_part_bp, p.block)
        ]
    out["regime_axis"] = [
        regime_axis(r.slot, r.window_h, r.attribution) for r in outcomes.regime_axis
    ]
    if outcomes.regime_map is not None:
        names = outcomes.regime_map_slots
        if names is None:
            names = list(pd.unique(outcomes.regime_map["slot"]))
        out["regime_map"] = [regime_map_sentence(n, outcomes.regime_map) for n in names]
    if outcomes.r is not None:
        out["break"] = [break_sentence(*outcomes.r)]
    return out
