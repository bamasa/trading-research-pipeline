"""The status rules on a held-out block, as pure functions of per-day values.

A read computes, once per usable day of the block, every series a slot's
status reads (:class:`SlotDays`): its primary daily series, trades, gross and
turnover, IC, the placebo series, the neighbour cells, the 110 ms re-run and,
for the passive slots, the queue brackets and the making part. Everything
below only slices those per-day values by a mask of days, so H's status window,
F-, all of F and every split day of R's reference set are evaluated by the same
code without re-running anything.

* :func:`kills` evaluates the slot's kill list in the order of its row of the
  registration's kill table (``reporting.headlines.KILL_ORDER``);
* :func:`assess` applies the precedence: void, then K-trades, then any kill,
  then the bar (Holm at m = 5, day t >= 3, Newey-West t >= 2, median day > 0,
  the day count), else inconclusive, below the MDE when the mean is smaller in
  size than it;
* :func:`family` runs Holm over the five slots with p = 1 wherever the
  registration sets it, and gives each slot its status on the block.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from trading_research.evaluation.family import FAMILY, MIN_DAYS, bar, family_holm, slot_p_value
from trading_research.market_making.verdicts import KillCheck, k_dir
from trading_research.reporting.headlines import KILL_ORDER

TAKER_FEE_BP = 5.5
MAKER_FEE_BP = 2.0
MIN_TRADES = 100
PLACEBO_PERCENTILE = 95.0
VOID_CONDITIONS = ("K-leak", "K-flag-leak", "K-units", "K-acct", "K-cap")
S25_WINDOW_END = date(2025, 7, 31)


@dataclass
class SlotDays:
    """One slot's per-day values on a block, every array one value per usable day.

    ``value`` is the primary series: ``net_bp_trade`` (NaN on a day without a
    trade) for the taker slots, ``net_per_100_clip_day`` (0 without a fill) for
    S-P and S-M, ``gross_bp_trade`` for S25-Z (with ``second`` its net at 0/0).
    """

    slot: str
    kind: str  # taker, entry, maker
    days: list[date]
    value: np.ndarray
    trades: np.ndarray
    gross: np.ndarray
    turnover: np.ndarray
    ic: np.ndarray
    random: np.ndarray  # draws x days
    flipped: np.ndarray
    shifted: np.ndarray
    long_only: np.ndarray
    short_only: np.ndarray
    neighbours: np.ndarray  # cells x days
    latency: np.ndarray
    stale: np.ndarray | None = None
    brackets: dict[str, np.ndarray] = field(default_factory=dict)
    making: np.ndarray | None = None
    passive_notional: np.ndarray | None = None
    second: np.ndarray | None = None
    void: tuple[str, ...] = ()
    on_s25_signal: bool = False
    window_h: int = 0
    mde: float | None = None
    #: Per-day measurements beside the status, each with days last (one value per
    #: day, or a row per draw): the window-0 twin (``twin``) and the random flag sets
    #: (``flagsets``) of the regime attribution, the control network's IC, the
    #: regime measurements, the passive exit. No rule reads them.
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def mask(self, days: Sequence[date]) -> np.ndarray:
        wanted = set(days)
        return np.array([d in wanted for d in self.days], dtype=bool)


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _series(s: SlotDays, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A series on the masked days as the bar reads it: the taker's days with a
    trade, every usable day for the passive slots."""
    picked = values[mask]
    return picked[np.isfinite(picked)] if s.kind == "taker" else np.nan_to_num(picked, nan=0.0)


def _check(
    slot: str, name: str, test: str, value: float, reference: float, fired: bool
) -> KillCheck:
    return KillCheck(slot, name, test, float(value), float(reference), bool(fired))


def kills(s: SlotDays, mask: np.ndarray) -> list[KillCheck]:
    """The slot's kill conditions on the masked days, in the kill table's order."""
    from trading_research.market_making.verdicts import exceeds_placebos

    v = _series(s, s.value, mask)
    mean = float(v.mean()) if len(v) else float("nan")
    checks: dict[str, KillCheck] = {}
    checks["K1"] = _check(s.slot, "K1", "mean <= 0", mean, 0.0, not mean > 0)
    if s.second is not None:
        second = _series(s, s.second, mask)
        net = float(second.mean()) if len(second) else float("nan")
        checks["K2"] = _check(s.slot, "K2", "net at 0/0 <= 0", net, 0.0, not net > 0)
    ic = _mean(s.ic[mask])
    checks["K-label"] = _check(s.slot, "K-label", "IC <= 0", ic, 0.0, not ic > 0)
    draws = [float(_series(s, row, mask).mean()) if mask.any() else math.nan for row in s.random]
    level, _ = exceeds_placebos(mean, draws)
    others = [
        _mean(_series(s, x, mask))
        for x in (s.flipped, s.shifted, *(() if s.stale is None else (s.stale,)))
    ]
    reference = max([level, *others], key=lambda x: -math.inf if not math.isfinite(x) else x)
    checks["K-placebo"] = _check(s.slot, "K-placebo", "mean does not exceed every placebo", mean,
                                 reference, not (math.isfinite(reference) and mean > reference))  # fmt: skip
    drift = max(_mean(_series(s, s.long_only, mask)), _mean(_series(s, s.short_only, mask)))
    checks["K-drift"] = _check(s.slot, "K-drift", "mean <= all-long or all-short", mean, drift,
                               not (math.isfinite(drift) and mean > drift))  # fmt: skip
    median = neighbourhood_median(s, mask)
    checks["K-nbhd"] = _check(
        s.slot, "K-nbhd", "neighbourhood median <= 0", median, 0.0, not median > 0
    )
    share = float((v > 0).mean()) if len(v) else float("nan")
    checks["K-days"] = _check(
        s.slot, "K-days", "half or fewer days positive", share, 0.5, not share > 0.5
    )
    checks["K-fee"] = fee_check(s, mask)
    latency = _mean(_series(s, s.latency, mask))
    checks["K-latency"] = _check(s.slot, "K-latency", "positive, but <= 0 at 110 ms", latency, 0.0,
                                 mean > 0 and not latency > 0)  # fmt: skip
    if s.brackets:
        worst = min(_mean(_series(s, x, mask)) for x in s.brackets.values())
        checks["K-queue"] = _check(s.slot, "K-queue", "positive, but not under a bracket", worst, 0.0,
                                   mean > 0 and not worst > 0)  # fmt: skip
    if s.making is not None and s.passive_notional is not None:
        notional = float(np.nansum(s.passive_notional[mask]))
        making = float(np.nansum(s.making[mask])) / notional * 1e4 if notional > 0 else float("nan")
        checks["K-dir"] = _check(s.slot, "K-dir", "making part not positive while net is", making, 0.0,
                                 k_dir(making, mean))  # fmt: skip
    return [checks[name] for name in KILL_ORDER[s.slot] if name in checks]


def neighbourhood_median(s: SlotDays, mask: np.ndarray) -> float:
    """K-nbhd's median over the whole neighbourhood, the chosen cell included: the
    slot's own series and each neighbour cell's (``s.neighbours`` holds the other
    cells only), a cell without a trade on the masked days scoring 0."""
    rows = [s.value, *s.neighbours]
    return float(np.median([np.nan_to_num(_mean(_series(s, row, mask)), nan=0.0) for row in rows]))


def fee_check(s: SlotDays, mask: np.ndarray) -> KillCheck:
    """K-fee: the fee break-even per side at or below the base fee for that side."""
    if s.kind == "taker":
        turnover = float(np.nansum(s.turnover[mask]))
        breakeven = (
            float(np.nansum(s.gross[mask])) / turnover * 1e4 if turnover > 0 else float("nan")
        )
        return _check(s.slot, "K-fee", "gross_bp_turnover <= taker fee", breakeven, TAKER_FEE_BP,
                      not breakeven > TAKER_FEE_BP)  # fmt: skip
    notional = float(np.nansum(s.passive_notional[mask])) if s.passive_notional is not None else 0.0
    net = float(np.nansum(s.gross[mask]))  # the passive slots carry their net in ``gross`` here
    breakeven = MAKER_FEE_BP + net / notional * 1e4 if notional > 0 else float("nan")
    return _check(s.slot, "K-fee", "maker fee break-even <= maker fee", breakeven, MAKER_FEE_BP,
                  not breakeven > MAKER_FEE_BP)  # fmt: skip


def trade_count(s: SlotDays, mask: np.ndarray) -> int:
    return int(np.nansum(s.trades[mask]))


def void_reasons(s: SlotDays, study_voids: Sequence[str]) -> list[str]:
    """Every void condition that applies: the study's (K-leak; K-flag-leak only for a
    slot with a regime window above 0) and the slot's own on the block."""
    found = [v for v in study_voids if v == "K-leak" or (v == "K-flag-leak" and s.window_h > 0)]
    return [*found, *s.void]


@dataclass
class Assessment:
    """A slot on a block (or a slice of one): its status and what it rests on."""

    slot: str
    status: str
    p: float
    kills: list[KillCheck] = field(default_factory=list)
    void: list[str] = field(default_factory=list)
    trades: int = 0
    days: int = 0
    mean: float = float("nan")
    bar: dict[str, Any] | None = None

    @property
    def first_kill(self) -> KillCheck | None:
        return next((k for k in self.kills if k.fired), None)

    def record(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {"slot": self.slot, "status": self.status, "p": self.p, "void": self.void,
                "trades": self.trades, "days": self.days, "mean": self.mean, "bar": self.bar,
                "kills": [asdict(k) for k in self.kills],
                "first_kill": None if self.first_kill is None else self.first_kill.name}  # fmt: skip


def _screen(s: SlotDays, mask: np.ndarray, study_voids: Sequence[str]) -> Assessment:
    """Precedence steps 1 to 3: void, too few trades, killed; else the p for Holm."""
    series = _series(s, s.value, mask)
    found = Assessment(s.slot, "", 1.0, trades=trade_count(s, mask), days=len(series),
                       mean=float(series.mean()) if len(series) else float("nan"))  # fmt: skip
    found.void = void_reasons(s, study_voids)
    if found.void:
        found.status = "void"
        return found
    if found.trades < MIN_TRADES:
        found.status = "inconclusive (too few trades)"
        return found
    found.kills = kills(s, mask)
    if found.first_kill is not None:
        found.status = "killed"
        return found
    found.p = slot_p_value(series)
    return found


def _barred(s: SlotDays, mask: np.ndarray, found: Assessment, passes_holm: bool, block: str,
            passes_word: str) -> Assessment:  # fmt: skip
    """Precedence steps 4 and 5 for a slot no earlier step settled."""
    series = _series(s, s.value, mask)
    result = bar(series, passes_holm, MIN_DAYS[block])
    found.bar = {k: getattr(result, k) for k in result.__dataclass_fields__} | {
        "passes": result.passes
    }
    if result.passes:
        found.status = passes_word
    elif s.mde is not None and math.isfinite(found.mean) and abs(found.mean) < s.mde:
        found.status = "inconclusive, below the minimum detectable effect"
    else:
        found.status = "inconclusive"
    return found


def family(
    series: Mapping[str, SlotDays],
    masks: Mapping[str, np.ndarray],
    block: str,
    preset: Mapping[str, str],
    study_voids: Sequence[str] = (),
) -> dict[str, Assessment]:
    """The five slots on a block: Holm at m = 5 over the slots no earlier rule set,
    every other slot entering with p = 1 (``preset``: a status set on D, not taken
    to the block, or a slot on S25's signal on F and P)."""
    out: dict[str, Assessment] = {}
    for slot in FAMILY:
        if slot in preset or slot not in series:
            out[slot] = Assessment(slot, preset.get(slot, "not assessed"), 1.0)
        else:
            out[slot] = _screen(series[slot], masks[slot], study_voids)
    holm = family_holm({slot: found.p for slot, found in out.items()})
    for slot, found in out.items():
        if found.status == "":
            word = "candidate" if block == "H" else "passes"
            if block == "H" and series[slot].on_s25_signal:
                word = "candidate (H before 2025-08-01)"
            _barred(series[slot], masks[slot], found, holm[slot], block, word)
    return out


def s25_z(s: SlotDays, mask: np.ndarray, study_voids: Sequence[str] = ()) -> Assessment:
    """S25-Z alone at alpha / 6 on the larger of its two claims' p-values, every
    condition of the bar on both claims (gross, and net at 0/0)."""
    from trading_research.evaluation.family import passes_alone

    found = _screen(s, mask, study_voids)
    if found.status:
        return found
    claims = [
        _series(s, s.value, mask),
        _series(s, s.second if s.second is not None else s.value, mask),
    ]
    alone = passes_alone([slot_p_value(c) for c in claims])
    bars = [bar(c, alone, MIN_DAYS["H"]) for c in claims]
    found.bar = {"claims": [{k: getattr(b, k) for k in b.__dataclass_fields__} for b in bars],
                 "passes_alone": alone}  # fmt: skip
    if all(b.passes for b in bars):
        found.status = "candidate (H before 2025-08-01), conditional on a zero-fee venue"
    elif s.mde is not None and math.isfinite(found.mean) and abs(found.mean) < s.mde:
        found.status = "inconclusive, below the minimum detectable effect"
    else:
        found.status = "inconclusive"
    return found


def taken_to_f(after_h: Assessment, on_s25_signal: bool) -> bool:
    """A candidate, or inconclusive at step 5 with a positive mean; never a slot on
    S25's signal, and never a result inconclusive under K-trades."""
    if on_s25_signal:
        return False
    if after_h.status == "candidate":
        return True
    step_5 = after_h.status in ("inconclusive", "inconclusive, below the minimum detectable effect")
    return step_5 and math.isfinite(after_h.mean) and after_h.mean > 0


def after_f(h_status: str, on_f: Assessment) -> str:
    """The status once F is read, for a slot taken to F (the registered table's F column)."""
    if on_f.status == "void":
        return "void"
    if on_f.status == "killed":
        return "killed on F"
    if h_status == "candidate":
        return "candidate" if on_f.status == "passes" else "candidate, not confirmed"
    return "exploratory: significant on F" if on_f.status == "passes" else "inconclusive"


def after_p(f_status: str, on_p: Assessment, r_supported: bool) -> str:
    """The final status once P is read, for a slot that passed on F."""
    if on_p.status == "void":
        return "void"
    if f_status == "candidate":
        if r_supported:
            persisted = not (math.isfinite(on_p.mean) and on_p.mean > 0)
            return "confirmed until the break" if persisted else "candidate; the break did not last"
        if on_p.status == "killed":
            return "killed on P"
        return "confirmed" if on_p.status == "passes" else "candidate, not confirmed"
    if f_status == "exploratory: significant on F":
        if on_p.status == "killed":
            return "killed on P"
        if on_p.status == "passes":
            return "confirmed on P, first significant on F"
        return "exploratory: significant on F, not confirmed"
    return f_status


@dataclass(frozen=True)
class Split:
    """F's split at the frozen detector's first alarm in F, when it applies."""

    b: date | None
    minus: tuple[date, ...]
    plus: tuple[date, ...]
    on_all_of_f: bool
    reason: str = ""


def split_f(
    days: Sequence[date], first_alarm: date | None, untestable: bool, r_void: bool
) -> Split:
    """F- and F+ around ``b`` (the alarm's own day in neither); statuses on F- only
    when it has at least 20 usable days, R is testable and not void."""
    if first_alarm is None:
        return Split(None, tuple(days), (), True, "no alarm in F")
    minus = tuple(d for d in days if d < first_alarm)
    plus = tuple(d for d in days if d > first_alarm)
    if len(minus) < 20 or untestable or r_void:
        reason = (
            "R void"
            if r_void
            else "R untestable"
            if untestable
            else "fewer than 20 usable days before b"
        )
        return Split(first_alarm, tuple(days), (), True, reason)
    return Split(first_alarm, minus, plus, False)


def r_test(s: SlotDays, days: Sequence[date], split: Split, untestable_declared: bool,
           passes_f_minus: bool, passes_before: Any) -> dict[str, Any]:  # fmt: skip
    """R for one candidate: supported when it passes on F-, its mean on F+ is <= 0
    and p_R <= 0.05 against every admissible split day at which it also passes."""

    def mean_on(chosen: Sequence[date]) -> float:
        picked = _series(s, s.value, s.mask(chosen))
        return float(picked.mean()) if len(picked) else 0.0

    if split.b is None:
        return {"outcome": "untestable", "reason": "no alarm in F"}
    before = [d for d in days if d < split.b]
    after = [d for d in days if d > split.b]
    if untestable_declared:
        return {
            "outcome": "untestable",
            "reason": "the detector's alarm rate on D above one per 14 days",
        }
    if len(before) < 20 or len(after) < 10:
        return {"outcome": "untestable", "reason": "fewer than 20 usable days before b or 10 after"}
    admissible = [
        d for i, d in enumerate(days) if d != split.b and i >= 20 and len(days) - i - 1 >= 10
    ]
    reference = [d for d in admissible if passes_before(d)]
    if len(reference) < 19:
        return {
            "outcome": "untestable",
            "reason": f"{len(reference)} days in the reference set besides b",
        }
    delta_b = mean_on(before) - mean_on(after)
    deltas = [
        mean_on([x for x in days if x < d]) - mean_on([x for x in days if x > d]) for d in reference
    ]
    p_r = (1 + sum(x >= delta_b for x in deltas)) / (1 + len(reference))
    plus = mean_on(after)
    supported = passes_f_minus and plus <= 0 and p_r <= 0.05
    return {"outcome": "supported" if supported else "not supported", "p_R": p_r, "delta_b": delta_b,
            "mean_F_plus": plus, "reference_days": len(reference)}  # fmt: skip


def subset(s: SlotDays, keep: np.ndarray) -> SlotDays:
    """The same slot on the kept days only (every per-day array sliced alike)."""
    from dataclasses import replace

    def cut(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        return values[..., keep]

    return replace(
        s, days=[d for d, k in zip(s.days, keep, strict=True) if k], value=s.value[keep],
        trades=s.trades[keep], gross=s.gross[keep], turnover=s.turnover[keep], ic=s.ic[keep],
        random=s.random[:, keep], flipped=s.flipped[keep], shifted=s.shifted[keep],
        long_only=s.long_only[keep], short_only=s.short_only[keep], neighbours=s.neighbours[:, keep],
        latency=s.latency[keep], stale=cut(s.stale), brackets={k: v[keep] for k, v in s.brackets.items()},
        making=cut(s.making), passive_notional=cut(s.passive_notional), second=cut(s.second),
        extra={name: values[..., keep] for name, values in s.extra.items()},
    )  # fmt: skip


def merge(a: SlotDays, b: SlotDays) -> SlotDays:
    """One slot on two sets of days as one, in day order: the days excluded for
    sequence gaps put back beside the usable days. An array one side lacks is NaN
    on that side's days; the void conditions are the union."""
    from dataclasses import replace

    days = [*a.days, *b.days]
    if len(set(days)) != len(days):
        raise ValueError("the two sets of days overlap")
    order = np.argsort([d.toordinal() for d in days], kind="stable")
    n_a, n_b = len(a.days), len(b.days)

    def cat(x: np.ndarray | None, y: np.ndarray | None) -> Any:
        if x is None and y is None:
            return None
        if x is None:
            assert y is not None
            x = np.full((*y.shape[:-1], n_a), np.nan)
        if y is None:
            y = np.full((*x.shape[:-1], n_b), np.nan)
        return np.concatenate([x, y], axis=-1)[..., order]

    names = {*a.brackets, *b.brackets}
    keys = {*a.extra, *b.extra}
    return replace(
        a, days=[days[i] for i in order], value=cat(a.value, b.value), trades=cat(a.trades, b.trades),
        gross=cat(a.gross, b.gross), turnover=cat(a.turnover, b.turnover), ic=cat(a.ic, b.ic),
        random=cat(a.random, b.random), flipped=cat(a.flipped, b.flipped),
        shifted=cat(a.shifted, b.shifted), long_only=cat(a.long_only, b.long_only),
        short_only=cat(a.short_only, b.short_only), neighbours=cat(a.neighbours, b.neighbours),
        latency=cat(a.latency, b.latency), stale=cat(a.stale, b.stale),
        brackets={k: cat(a.brackets.get(k), b.brackets.get(k)) for k in sorted(names)},
        making=cat(a.making, b.making), passive_notional=cat(a.passive_notional, b.passive_notional),
        second=cat(a.second, b.second), void=tuple(sorted({*a.void, *b.void})),
        extra={k: cat(a.extra.get(k), b.extra.get(k)) for k in sorted(keys)},
    )  # fmt: skip
