"""The spot study's status rules on H and P, as pure functions of per-day values.

A read computes, once per usable day of a venue's block, every series a slot's
status reads (:class:`SlotDays`): its primary daily series (``net_bp_trade`` at
the verdict tier), trades, ``gross_bp_turnover``, IC, the placebo series, the
neighbourhood cells, the 110 ms re-run and, for a print slot, the worse-of-two
bracket's sums, its fills and the fills beyond the sweep. Everything here reads
only those values (``docs/preregistration/spot_2026.md``, "Status rules",
"Kill conditions per slot", "Placebos"):

* :func:`kills` evaluates the slot's kill list in the order of its row of the
  kill table; :func:`measured_kills` the measured rows' list, with no status;
* :func:`family` applies the precedence on a block: void (K-leak, K-units,
  K-acct, K-cap, K-inv), then K-trades (fewer than 100 round trips), then any
  kill, then the bar (Holm at m = 3, one-sided at 5%, through
  ``market_making.verdicts``; day-level t at least 3 and Newey-West t at least 2
  through ``evaluation.significance``; median day above 0; 20 days with a
  trade on H, 10 on P), the two caps after Holm, else inconclusive, below the
  MDE when the mean is smaller in size than it;
* :func:`on_h`, :func:`on_p` and :func:`final_status` give the registered
  statuses on H and after P, with the information-set fallback;
* the fillers at the end write the registered headline sentences.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_research.evaluation import significance
from trading_research.market_making.verdicts import KillCheck, day_t, exceeds_placebos, holm
from trading_research.spot import access as sa
from trading_research.spot.registration import (
    FALLBACK_MARK,
    FAMILY,
    INSTRUMENT_OF,
    PRINT_ROWS,
    VERDICT_TAKER_BP,
    with_fallback,
)

#: Holm's procedure over the fixed family, one-sided, on every block.
ALPHA = 0.05
M = len(FAMILY)
T_BAR = significance.T_BAR
NEWEY_WEST_BAR = 2.0
NEWEY_WEST_LAG = significance.NEWEY_WEST_LAG
#: Days with a trade a series needs on each block.
MIN_DAYS: dict[str, int] = {"H": 20, "P": 10}
#: K-trades: fewer round trips over the block make the result inconclusive.
MIN_TRADES = 100
#: The caps, applied after Holm.
FILL_BEYOND_SWEEP_CAP = 0.10
CAPACITY_CAP = 0.02
K_DAYS_SHARE = 0.5
LATENCY_RERUN_MS = 110
PLACEBO_PERCENTILE = 95.0
RANDOM_DRAWS = 200
NEIGHBOURHOOD_CELLS = 12
#: H's days in the frozen MDE's n_eff (75 times the share of D's test days with a trade).
MDE_DAYS_H = 75

VOID_CONDITIONS: tuple[str, ...] = ("K-leak", "K-units", "K-acct", "K-cap", "K-inv")
_BOOK = ("K1", "K-label", "K-placebo", "K-drift", "K-nbhd", "K-days", "K-fee", "K-latency")
#: Each slot's kills in the order of its row of the kill table.
KILL_ORDER: dict[str, tuple[str, ...]] = {
    "S25-PF": (*_BOOK, "K-fill"),
    "S25-BB": _BOOK,
    "S25-T": _BOOK,
}
#: What the measured rows report: no K-nbhd (each is fitted for its own cell only).
MEASURED_KILLS: tuple[str, ...] = (
    "K1", "K-label", "K-placebo", "K-drift", "K-days", "K-fee", "K-latency", "K-fill",
)  # fmt: skip

#: How the statuses and sentences name each venue and pair.
VENUE_NAME: dict[str, str] = {sa.BINANCE: "Binance", sa.BYBIT: "Bybit"}
PAIR_NAME: dict[str, str] = {sa.BINANCE: "BTC/FDUSD", sa.BYBIT: "BTCUSDT"}

CANDIDATE, PASSES = "candidate", "passes"
TOO_FEW = "inconclusive (too few trades)"
BELOW_MDE = "inconclusive, below the minimum detectable effect"
CAP_SWEEP, CAP_CAPACITY = "inconclusive (fill beyond sweep)", "inconclusive (capacity)"
NOT_TAKEN = "not taken to the block"
H_READ_FAILED = "not assessed (H read failed)"


def unreadable_status(name: str, block: str = "H") -> str:
    """The status of a slot on a block its venue's days leave unreadable."""
    return f"not assessed ({block} unreadable on {VENUE_NAME[sa.instrument(name).id]})"


def defect_status(block: str) -> str:
    """What a defect found after a read makes of every status it could affect."""
    if block not in sa.HELD_OUT:
        raise ValueError(f"reads are of {sa.HELD_OUT}, not {block!r}")
    return f"void (defect found after the {block} read)"


@dataclass(frozen=True)
class FillDays:
    """A print slot's worse-of-two bracket as per-day sums: its net in quote, its
    entry notional, and the notional of its buys and of its sells (entries and
    exits alike), from which K-fill's charged series is made exactly."""

    net_quote: np.ndarray
    entry_notional: np.ndarray
    buy_notional: np.ndarray
    sell_notional: np.ndarray

    def charged(self, a_buy: float, a_sell: float) -> np.ndarray:
        """The daily ``net_bp_trade`` once every buy is charged max(0, a_buy) and
        every sell max(0, a_sell) bp of its notional; NaN on a day without a trade."""
        extra = max(0.0, a_buy) * self.buy_notional + max(0.0, a_sell) * self.sell_notional
        net = np.asarray(self.net_quote, dtype=np.float64) - extra / 1e4
        notional = np.asarray(self.entry_notional, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(notional > 0, net / notional * 1e4, np.nan)


@dataclass
class SlotDays:
    """One slot's (or measured row's) per-day values on a block, one value per
    usable day of its venue (``days``), every draw or cell a row with days last.

    ``value`` is ``net_bp_trade`` at the verdict tier, NaN on a day without a
    trade, as is ``gross_bp_turnover``; ``random`` holds the 200 random-direction
    draws; ``neighbourhood`` the 12 cells, the slot's own among them (none for a
    measured row); ``worse_of_two``, ``fills`` and ``beyond_sweep`` are a print
    slot's; ``capacity`` the day's capacity ratio (0 without a trade); ``void``
    the slot's own void conditions that fired in any run of its read; ``mde`` the
    frozen MDE on H.
    """

    slot: str
    days: list[date]
    value: np.ndarray
    trades: np.ndarray
    gross_bp_turnover: np.ndarray
    ic: np.ndarray
    random: np.ndarray
    flipped: np.ndarray
    shifted: np.ndarray
    all_long: np.ndarray
    all_short: np.ndarray
    latency: np.ndarray
    neighbourhood: np.ndarray | None = None
    worse_of_two: FillDays | None = None
    fills: np.ndarray | None = None
    beyond_sweep: np.ndarray | None = None
    capacity: np.ndarray | None = None
    void: tuple[str, ...] = ()
    mde: float | None = None

    def __post_init__(self) -> None:
        n = len(self.days)
        rows = {"value": self.value, "trades": self.trades, "ic": self.ic, "latency": self.latency,
                "gross_bp_turnover": self.gross_bp_turnover, "flipped": self.flipped,
                "shifted": self.shifted, "all_long": self.all_long, "all_short": self.all_short}  # fmt: skip
        for name, values in rows.items():
            if np.shape(values) != (n,):
                raise ValueError(f"{self.slot}.{name} holds one value per day ({n})")
        if np.ndim(self.random) != 2 or np.shape(self.random)[1] != n:
            raise ValueError(f"{self.slot}.random is draws x days")
        unknown = set(self.void) - set(VOID_CONDITIONS)
        if unknown:
            raise ValueError(f"{self.slot}: not a void condition {sorted(unknown)}")


def series(values: np.ndarray) -> np.ndarray:
    """A daily series as the rules read it: one value per day with a trade."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def series_mean(values: np.ndarray) -> float:
    """The mean of a daily series; a run without a trade scores 0."""
    found = series(values)
    return float(found.mean()) if len(found) else 0.0


@dataclass(frozen=True)
class DayStatistics:
    """A daily series' statistics, each computed on the series, never on trade rows."""

    days: int
    mean: float
    t: float
    p_one_sided: float
    newey_west_t: float
    median: float


def day_statistics(values: np.ndarray, days: Sequence[date] | None = None) -> DayStatistics:
    """The mean and day-level t through ``significance.assess`` with one row per
    day, the one-sided p of ``verdicts.day_t`` (1 when undefined or under two
    days), the Newey-West t (Bartlett, lag 5) and the median day."""
    array = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(array)
    found = array[keep]
    n = len(found)
    median = float(np.median(found)) if n else math.nan
    if n < 2:
        mean = float(found.mean()) if n else math.nan
        return DayStatistics(n, mean, math.nan, 1.0, math.nan, median)
    labels = list(range(len(array))) if days is None else list(days)
    frame = pd.DataFrame(
        {"day": [d for d, k in zip(labels, keep, strict=True) if k], "value": found}
    )
    assessed = significance.assess(frame, value="value", cluster="day", series=None)
    p = day_t(found).p_one_sided
    nw = significance.newey_west_t(found, NEWEY_WEST_LAG).t
    return DayStatistics(n, assessed.cluster_mean_bp, assessed.cluster_t,
                         p if math.isfinite(p) else 1.0, nw, median)  # fmt: skip


def placebo_means(s: SlotDays) -> dict[str, float]:
    """Each placebo's mean on the block (a run without a trade scoring 0), and the
    95th percentile of the random draws' means (``verdicts.exceeds_placebos``)."""
    mean = series_mean(s.value)
    draws = [series_mean(row) for row in np.asarray(s.random)]
    level, _ = exceeds_placebos(mean, draws, PLACEBO_PERCENTILE)
    named = {
        "flipped": s.flipped,
        "shifted": s.shifted,
        "all_long": s.all_long,
        "all_short": s.all_short,
    }
    return {name: series_mean(x) for name, x in named.items()} | {"random_p95": level}


def placebo_percentile(s: SlotDays) -> float:
    """The share of random draws whose mean is strictly below the true mean, in percent."""
    draws = np.array([series_mean(row) for row in np.asarray(s.random)], dtype=np.float64)
    return float((draws < series_mean(s.value)).mean() * 100) if len(draws) else math.nan


def neighbourhood_median(s: SlotDays) -> float:
    """K-nbhd's median over the 12 cells (the slot's own among them) of each cell's
    mean on the block, a cell without a trade scoring 0."""
    if s.neighbourhood is None or np.shape(s.neighbourhood) != (NEIGHBOURHOOD_CELLS, len(s.days)):
        raise ValueError(f"{s.slot}: the neighbourhood is {NEIGHBOURHOOD_CELLS} cells x days")
    return float(np.median([series_mean(row) for row in np.asarray(s.neighbourhood)]))


def fee_break_even(s: SlotDays) -> float:
    """The fee break-even per side: the mean over the series' days of each day's
    ``gross_bp_turnover`` (NaN without a day with a trade)."""
    found = series(s.gross_bp_turnover)
    return float(found.mean()) if len(found) else math.nan


def _check(
    slot: str, name: str, test: str, value: float, reference: float, fired: bool
) -> KillCheck:
    return KillCheck(slot, name, test, float(value), float(reference), bool(fired))


def _checks(s: SlotDays, a_buy: float, a_sell: float) -> dict[str, KillCheck]:
    """Every kill condition the slot's values allow, by name."""
    v = series(s.value)
    mean = float(v.mean()) if len(v) else math.nan
    positive = mean > 0
    out: dict[str, KillCheck] = {}
    out["K1"] = _check(s.slot, "K1", "mean <= 0", mean, 0.0, not positive)
    ic = series(s.ic)
    label = float(ic.mean()) if len(ic) else math.nan
    out["K-label"] = _check(s.slot, "K-label", "IC <= 0", label, 0.0, not label > 0)
    placebos = placebo_means(s)
    level = placebos["random_p95"]
    reference = max(
        placebos["flipped"], placebos["shifted"], level if math.isfinite(level) else -math.inf
    )
    exceeds = (
        math.isfinite(level)
        and mean > level
        and mean > placebos["flipped"]
        and mean > placebos["shifted"]
    )
    out["K-placebo"] = _check(s.slot, "K-placebo", "mean does not exceed every placebo", mean,
                              reference, not exceeds)  # fmt: skip
    drift = max(placebos["all_long"], placebos["all_short"])
    out["K-drift"] = _check(s.slot, "K-drift", "mean <= all-long or all-short", mean, drift,
                            not mean > drift)  # fmt: skip
    if s.neighbourhood is not None:
        median = neighbourhood_median(s)
        out["K-nbhd"] = _check(
            s.slot, "K-nbhd", "neighbourhood median <= 0", median, 0.0, not median > 0
        )
    share = float((v > 0).mean()) if len(v) else math.nan
    out["K-days"] = _check(s.slot, "K-days", "half or fewer days positive", share, K_DAYS_SHARE,
                           not share > K_DAYS_SHARE)  # fmt: skip
    breakeven = fee_break_even(s)
    out["K-fee"] = _check(s.slot, "K-fee", "gross_bp_turnover <= taker fee", breakeven,
                          VERDICT_TAKER_BP, not breakeven > VERDICT_TAKER_BP)  # fmt: skip
    latency = series_mean(s.latency)
    out["K-latency"] = _check(s.slot, "K-latency", "positive, but <= 0 at 110 ms", latency, 0.0,
                              positive and not latency > 0)  # fmt: skip
    if s.worse_of_two is not None:
        defined = math.isfinite(a_buy) and math.isfinite(a_sell)
        charged = series_mean(s.worse_of_two.charged(a_buy, a_sell)) if defined else math.nan
        out["K-fill"] = _check(s.slot, "K-fill", "positive, but <= 0 under worse of two less the "
                               "thinned calibration's advantage", charged, 0.0,
                               positive and not charged > 0)  # fmt: skip
    return out


def kills(s: SlotDays, a_buy: float = math.nan, a_sell: float = math.nan) -> list[KillCheck]:
    """A family slot's kill conditions in the order of its row of the kill table.

    ``a_buy`` and ``a_sell`` are the thinned calibration's mean advantages of the
    worse-of-two price over the book walk on Bybit's days of the same block, which
    K-fill charges; when either is undefined K-fill fires on a positive result.
    """
    if s.slot not in KILL_ORDER:
        raise ValueError(f"{s.slot} is not a slot of the family {FAMILY}")
    if s.slot in PRINT_ROWS and s.worse_of_two is None:
        raise ValueError(f"{s.slot} is a print slot: K-fill needs its worse-of-two bracket")
    checks = _checks(s, a_buy, a_sell)
    missing = [name for name in KILL_ORDER[s.slot] if name not in checks]
    if missing:
        raise ValueError(f"{s.slot} lacks the values for {missing}")
    return [checks[name] for name in KILL_ORDER[s.slot]]


def measured_kills(
    s: SlotDays, a_buy: float = math.nan, a_sell: float = math.nan
) -> list[KillCheck]:
    """A measured row's conditions, evaluated and reported: no K-nbhd, K-fill only
    where the row is priced by the print rule, no precedence and no status."""
    checks = _checks(s, a_buy, a_sell)
    return [checks[name] for name in MEASURED_KILLS if name in checks]


def first_kill(slot: str, checks: Sequence[KillCheck]) -> KillCheck | None:
    """The first kill that fired, in the order of the slot's row of the kill table."""
    order = KILL_ORDER.get(slot, MEASURED_KILLS)
    fired = [k for k in checks if k.fired and k.name in order]
    return min(fired, key=lambda k: order.index(k.name)) if fired else None


@dataclass
class Assessment:
    """A slot on a block: its status and what it rests on."""

    slot: str
    block: str
    status: str
    p: float = 1.0
    kills: list[KillCheck] = field(default_factory=list)
    void: list[str] = field(default_factory=list)
    trades: int = 0
    days: int = 0
    mean: float = math.nan
    bar: dict[str, Any] | None = None
    caps: dict[str, float] = field(default_factory=dict)
    mde: float | None = None

    @property
    def first_kill(self) -> KillCheck | None:
        return first_kill(self.slot, self.kills)

    def record(self) -> dict[str, Any]:
        """The assessment as the results file holds it (JSON-ready)."""
        kill = self.first_kill
        return {"slot": self.slot, "block": self.block, "status": self.status, "p": self.p,
                "void": self.void, "trades": self.trades, "days": self.days, "mean": self.mean,
                "bar": self.bar, "caps": self.caps, "mde": self.mde,
                "kills": [asdict(k) for k in self.kills],
                "first_kill": None if kill is None else kill.name}  # fmt: skip


def block_mde(mde_h: float | None, block: str, usable_days: int) -> float | None:
    """The MDE the "below the minimum detectable effect" label reads: the frozen one
    on H; on P the same formula with P's usable days in place of 75, sigma_D and
    the share as frozen, so the frozen MDE times sqrt(75 / P's usable days)."""
    if mde_h is None or block == "H":
        return mde_h
    return float(mde_h * math.sqrt(MDE_DAYS_H / usable_days)) if usable_days > 0 else math.inf


def _screen(
    s: SlotDays, block: str, study_voids: Sequence[str], a_buy: float, a_sell: float
) -> Assessment:
    """Precedence steps 1 to 3: void, too few trades, killed; else the p for Holm."""
    v = series(s.value)
    found = Assessment(s.slot, block, "", trades=int(np.nansum(s.trades)), days=len(v),
                       mean=float(v.mean()) if len(v) else math.nan,
                       mde=block_mde(s.mde, block, len(s.days)))  # fmt: skip
    unknown = set(study_voids) - set(VOID_CONDITIONS)
    if unknown:
        raise ValueError(f"not a void condition {sorted(unknown)}")
    found.void = sorted({*study_voids, *s.void}, key=VOID_CONDITIONS.index)
    if found.void:
        found.status = "void"
        return found
    if found.trades < MIN_TRADES:
        found.status = TOO_FEW
        return found
    found.kills = kills(s, a_buy, a_sell)
    if found.first_kill is not None:
        found.status = "killed"
        return found
    found.p = day_statistics(s.value, s.days).p_one_sided
    return found


def bar(s: SlotDays, passes_holm: bool, block: str) -> dict[str, Any]:
    """The bar on the slot's daily series, each condition separately."""
    found = day_statistics(s.value, s.days)
    out: dict[str, Any] = {"days": found.days, "mean": found.mean, "t": found.t,
                           "p_one_sided": found.p_one_sided, "newey_west_t": found.newey_west_t,
                           "median": found.median, "holm": bool(passes_holm),
                           "t_ok": bool(found.t >= T_BAR),
                           "newey_west_ok": bool(found.newey_west_t >= NEWEY_WEST_BAR),
                           "median_ok": bool(found.median > 0),
                           "days_ok": bool(found.days >= MIN_DAYS[block])}  # fmt: skip
    out["passes"] = all(out[k] for k in ("holm", "t_ok", "newey_west_ok", "median_ok", "days_ok"))
    return out


def caps(s: SlotDays) -> dict[str, float]:
    """The two caps' statistics: a print slot's share of fills with part of the clip
    beyond the sweep, and the median day's capacity ratio."""
    if s.capacity is None or np.shape(s.capacity) != (len(s.days),):
        raise ValueError(f"{s.slot}: the capacity ratio of every usable day is needed")
    out = {
        "capacity_median": float(np.median(np.nan_to_num(np.asarray(s.capacity, float), nan=0.0)))
    }
    if s.slot in PRINT_ROWS:
        if s.fills is None or s.beyond_sweep is None:
            raise ValueError(f"{s.slot} is a print slot: its fills and fills beyond the sweep")
        total = float(np.nansum(s.fills))
        out["fill_beyond_sweep_share"] = float(np.nansum(s.beyond_sweep)) / total if total else 0.0
    return out


def _step_4_to_6(s: SlotDays, found: Assessment, passes_holm: bool, word: str) -> None:
    found.bar = bar(s, passes_holm, found.block)
    if found.bar["passes"]:
        found.caps = caps(s)
        if found.caps.get("fill_beyond_sweep_share", 0.0) > FILL_BEYOND_SWEEP_CAP:
            found.status = CAP_SWEEP
        elif found.caps["capacity_median"] > CAPACITY_CAP:
            found.status = CAP_CAPACITY
        else:
            found.status = word
    elif found.mde is not None and math.isfinite(found.mean) and abs(found.mean) < found.mde:
        found.status = BELOW_MDE
    else:
        found.status = "inconclusive"


def family(
    rows: Mapping[str, SlotDays],
    block: str,
    *,
    blind: bool,
    preset: Mapping[str, str] | None = None,
    study_voids: Sequence[str] = (),
    a_buy: float = math.nan,
    a_sell: float = math.nan,
) -> dict[str, Assessment]:
    """The three slots on ``block`` (H or P): Holm at m = 3 over the slots no earlier
    step settled, every other slot entering with p = 1 (``preset`` gives a status
    set without assessing: not assessed, not taken to the block). On H under the
    fallback (``blind`` False) a candidate reads "candidate (H inside S25's
    information set)"."""
    if block not in MIN_DAYS:
        raise ValueError(f"statuses are given on {tuple(MIN_DAYS)}, not {block!r}")
    preset = dict(preset or {})
    out: dict[str, Assessment] = {}
    for slot in FAMILY:
        if slot in preset or slot not in rows:
            out[slot] = Assessment(slot, block, preset.get(slot, NOT_TAKEN))
        else:
            if rows[slot].slot != slot:
                raise ValueError(f"the values given as {slot} are {rows[slot].slot}'s")
            out[slot] = _screen(rows[slot], block, study_voids, a_buy, a_sell)
    passed = holm({slot: found.p for slot, found in out.items()}, ALPHA)
    word = (CANDIDATE if blind else with_fallback(CANDIDATE, blind)) if block == "H" else PASSES
    for slot, found in out.items():
        if found.status == "":
            _step_4_to_6(rows[slot], found, passed[slot], word)
    return out


def on_h(
    rows: Mapping[str, SlotDays],
    *,
    blind: bool,
    unreadable: Iterable[str] = (),
    read_failed: bool = False,
    study_voids: Sequence[str] = (),
    a_buy: float = math.nan,
    a_sell: float = math.nan,
) -> dict[str, Assessment]:
    """Every slot's status on H. ``unreadable`` names the instruments whose block
    has more than a quarter of its days excluded: their slots are not assessed and
    the other venue's are; a read that closed as failed leaves every slot not assessed."""
    if read_failed:
        return {slot: Assessment(slot, "H", H_READ_FAILED) for slot in FAMILY}
    gone = {sa.instrument(name).id for name in unreadable}
    preset = {
        slot: unreadable_status(INSTRUMENT_OF[slot])
        for slot in FAMILY
        if INSTRUMENT_OF[slot] in gone
    }
    return family(
        rows, "H", blind=blind, preset=preset, study_voids=study_voids, a_buy=a_buy, a_sell=a_sell
    )


def is_candidate(h_status: str) -> bool:
    """A candidate on H, in either form; the only status taken to P."""
    return h_status in (CANDIDATE, with_fallback(CANDIDATE, False))


def on_p(
    rows: Mapping[str, SlotDays],
    h_statuses: Mapping[str, str],
    *,
    unreadable: Iterable[str] = (),
    study_voids: Sequence[str] = (),
    a_buy: float = math.nan,
    a_sell: float = math.nan,
) -> dict[str, Assessment]:
    """Every slot on P: the candidates on H assessed (a candidate whose venue's P is
    unreadable marked so), every other slot not taken to P, entering Holm with p = 1."""
    gone = {sa.instrument(name).id for name in unreadable}
    preset = {}
    for slot in FAMILY:
        if not is_candidate(h_statuses.get(slot, "")):
            preset[slot] = NOT_TAKEN
        elif INSTRUMENT_OF[slot] in gone:
            preset[slot] = unreadable_status(INSTRUMENT_OF[slot], "P")
    return family(
        rows, "P", blind=True, preset=preset, study_voids=study_voids, a_buy=a_buy, a_sell=a_sell
    )


#: What P's read can be, for a candidate.
P_STATES: tuple[str, ...] = ("read", "not yet read", "not read by its deadline", "read failed")


def p_deadline(h_results_committed: date) -> date:
    """The later of 2027-01-31 and 30 days after H's results are committed."""
    return max(date(2027, 1, 31), h_results_committed + timedelta(days=30))


def final_status(h_status: str, p: Assessment | None = None, *, p_state: str, blind: bool) -> str:
    """The status after P, by the registered table; under the fallback every status
    of its last column carries "(H inside S25's information set)"."""
    if p_state not in P_STATES:
        raise ValueError(f"P's read is one of {P_STATES}, not {p_state!r}")
    if not is_candidate(h_status):
        return h_status if h_status.startswith("void (defect") else with_fallback(h_status, blind)
    if p_state == "not yet read":
        found = "candidate (P not yet read)"
    elif p_state == "not read by its deadline":
        found = "candidate, not confirmed (P not read by its deadline)"
    elif p_state == "read failed":
        found = "candidate, not confirmed (P read failed)"
    elif p is None:
        raise ValueError("a read P needs the candidate's assessment on it")
    elif p.status.startswith("not assessed (P unreadable"):
        found = "candidate, not confirmed (P unreadable)"
    elif p.status == "void":
        found = "void"
    elif p.status == "killed":
        found = "killed on P"
    elif p.status == PASSES:
        found = "confirmed"
    else:
        found = "candidate, not confirmed"
    return with_fallback(found, blind)


def results_record(
    block: str,
    assessments: Mapping[str, Assessment],
    *,
    stamp: str,
    session: Sequence[str],
    blind: bool,
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What ``experiments/results/spot_2026_<block>.json`` holds: the read's stamp
    and session, whether H was blind, every status with its kills, and the
    measurements beside them."""
    return {"block": block, "stamp": stamp, "session": list(session), "blind": bool(blind),
            "statuses": {slot: a.record() for slot, a in assessments.items()},
            "measurements": dict(measurements or {})}  # fmt: skip


def candidates_in(results: Mapping[str, Any]) -> list[str]:
    """The slots H's results file records as candidates (a second read's never)."""
    if results.get("block") != "H":
        raise ValueError("candidates are read from H's results")
    if results.get("stamp") != "first read":
        return []
    statuses = results.get("statuses", {})
    return [slot for slot in FAMILY if is_candidate(str(statuses.get(slot, {}).get("status", "")))]


def trades_per_day(s: SlotDays) -> float:
    """Round trips over the block's usable days."""
    return float(np.nansum(s.trades)) / len(s.days) if s.days else math.nan


# ---------------------------------------------------------------------------
# The registered headline sentences
# ---------------------------------------------------------------------------


class HeadlineError(ValueError):
    """An input the registered sentences cannot carry."""


#: The registered sentences, verbatim (``artefacts.headline_sentences``).
TEMPLATES: dict[str, str] = {
    "each_slot_on_H": (
        "[slot] on H, [venue] spot [pair]: [status]; first kill [K-x] at [value] against [reference]."
    ),
    "S25-PF": (
        "S25's coordinates on nine print inputs, on Binance BTC/FDUSD, August to October 2026: "
        "gross edge [x] bp of turnover against the 10 bp taker fee a Regular account paid; "
        "break-even [y] bp per side; [status]."
    ),
    "S25-T": (
        "S25 as frozen on Bybit's perpetual, unchanged on Bybit's spot book: [n] trades a day "
        "against 240 registered; [status]."
    ),
    "calibration": (
        "On Bybit's spot book, the print rule priced buys [x] bp and sells [y] bp "
        "[better / worse] than the book walk."
    ),
    "P": "[slot] on P: [status].",
}


def _groups(template: str) -> list[tuple[int, int]]:
    spans, depth, start = [], 0, 0
    for i, char in enumerate(template):
        if char == "[":
            start = i if depth == 0 else start
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def fill(name: str, *values: str) -> str:
    """Template ``name`` with its brackets replaced by ``values``, the rest unchanged."""
    template = TEMPLATES[name]
    spans = _groups(template)
    if len(values) != len(spans) or not all(values):
        raise HeadlineError(f"{name} has {len(spans)} brackets to fill, given {values!r}")
    out, last = [], 0
    for (start, end), value in zip(spans, values, strict=True):
        out += [template[last:start], value]
        last = end
    return "".join([*out, template[last:]])


def pattern(name: str) -> re.Pattern[str]:
    """A pattern every filled sentence of ``name`` matches."""
    template, parts, last = TEMPLATES[name], [], 0
    for start, end in _groups(template):
        parts += [re.escape(template[last:start]), "(.+)"]
        last = end
    return re.compile("".join([*parts, re.escape(template[last:])]))


_VENUES = "|".join(VENUE_NAME.values())
#: Every registered status, as patterns; under the fallback each may carry its mark.
STATUS_PATTERNS: tuple[str, ...] = (
    r"candidate", r"passes", r"inconclusive", r"inconclusive \(fill beyond sweep\)",
    r"inconclusive \(capacity\)", r"inconclusive \(too few trades\)",
    r"inconclusive, below the minimum detectable effect", r"killed", r"void",
    rf"not assessed \(H unreadable on (?:{_VENUES})\)", r"not assessed \(H read failed\)",
    r"void \(defect found after the [HP] read\)", r"confirmed", r"candidate, not confirmed",
    r"killed on P", r"candidate \(P not yet read\)",
    r"candidate, not confirmed \(P (?:unreadable|read failed|not read by its deadline)\)",
)  # fmt: skip
_STATUS = re.compile(
    "(?:" + "|".join(STATUS_PATTERNS) + r")(?: \(" + re.escape(FALLBACK_MARK) + r"\))?"
)
_KILL_FREE = re.compile(r"candidate|confirmed|inconclusive|not assessed|passes|void")
KILL_UNITS = {
    "K-days": " (share of days positive)",
    "K-label": " (rank IC)",
    "K-fee": " bp per side",
}


def check_status(status: str) -> str:
    """``status`` if it is a registered status (with the fallback's mark at most)."""
    if not _STATUS.fullmatch(status):
        raise HeadlineError(f"not a registered status: {status!r}")
    return status


def _bp(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"


def slot_on_h(slot: str, status: str, checks: Sequence[KillCheck] = ()) -> str:
    """Each slot's sentence on H: its status there and its first kill against its
    reference; a slot with no kill reads ``none`` at ``n/a`` against ``n/a``."""
    if slot not in FAMILY:
        raise HeadlineError(f"{slot!r} is not a slot of the family")
    check_status(status)
    kill = first_kill(slot, checks)
    if status.startswith("killed") and kill is None:
        raise HeadlineError(f"{slot} on H is {status!r} with no kill fired")
    if kill is not None and _KILL_FREE.match(status):
        raise HeadlineError(f"{slot} on H is {status!r} but {kill.name} fired")
    name = INSTRUMENT_OF[slot]
    venue, pair = VENUE_NAME[name], PAIR_NAME[name]
    if kill is None:
        return fill("each_slot_on_H", slot, venue, pair, status, "none", "n/a", "n/a")
    digits = 3 if kill.name == "K-label" else 2
    unit = KILL_UNITS.get(kill.name, " bp")
    value, reference = _bp(kill.value, digits), _bp(kill.reference, digits) + unit
    return fill("each_slot_on_H", slot, venue, pair, status, kill.name, value, reference)


def s25_pf_sentence(gross_bp_turnover: float, status: str) -> str:
    """S25-PF's sentence. Its gross edge in bp of turnover and its break-even per
    side are one registered statistic (``metrics.gross_bp_turnover`` on H, the mean
    of the daily values), so both brackets carry it."""
    check_status(status)
    return fill("S25-PF", _bp(gross_bp_turnover), _bp(gross_bp_turnover), status)


def s25_t_sentence(per_day: float, status: str) -> str:
    """S25-T's sentence: its round trips a day on H's usable days, against 240."""
    check_status(status)
    return fill("S25-T", _bp(per_day, 1), status)


def calibration_sentence(buy_bp: float, sell_bp: float) -> str:
    """The calibration's sentence from FILL-S on H: the print rule's mean advantage
    over the book walk per side, in bp of mid, positive when better for the account.

    One word serves both sides: the sign of the larger advantage in size picks it,
    and each value is written as the advantage in that word's direction, so a side
    of the other sign reads negative."""
    if not (math.isfinite(buy_bp) and math.isfinite(sell_bp)):
        raise HeadlineError("the calibration needs both sides' advantages")
    better, worse = "better", "worse"
    lead = buy_bp if abs(buy_bp) >= abs(sell_bp) else sell_bp
    sign = 1.0 if lead >= 0 else -1.0
    word = better if sign > 0 else worse
    return fill("calibration", _bp(sign * buy_bp + 0.0), _bp(sign * sell_bp + 0.0), word)


def p_sentence(slot: str, status: str) -> str:
    """A slot taken to P: its status after P, by the registered table."""
    if slot not in FAMILY:
        raise HeadlineError(f"{slot!r} is not a slot of the family")
    return fill("P", slot, check_status(status))


def headline_sentences(
    on_h_: Mapping[str, Assessment],
    *,
    s25_pf_gross_bp_turnover: float,
    s25_t_trades_per_day: float,
    calibration: tuple[float, float] | None = None,
    after_p: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Every registered sentence the outcome calls for: one per slot on H, S25-PF's
    and S25-T's, the calibration's when FILL-S was measured on H, and one per slot
    taken to P once ``after_p`` holds its final status."""
    found = {slot: on_h_[slot] for slot in FAMILY}
    out: dict[str, list[str]] = {
        "each_slot_on_H": [slot_on_h(s, a.status, a.kills) for s, a in found.items()],
        "S25-PF": [s25_pf_sentence(s25_pf_gross_bp_turnover, found["S25-PF"].status)],
        "S25-T": [s25_t_sentence(s25_t_trades_per_day, found["S25-T"].status)],
        "calibration": [] if calibration is None else [calibration_sentence(*calibration)],
        "P": [],
    }
    if after_p is not None:
        taken = [s for s, a in found.items() if is_candidate(a.status)]
        missing = [s for s in taken if s not in after_p]
        if missing:
            raise HeadlineError(f"no status after P for {missing}")
        out["P"] = [p_sentence(s, after_p[s]) for s in taken]
    return out
