"""The market-maker branch of execution: a quoter, its settings, its days.

:mod:`.execution` decides *that* a run quotes both sides; this module says
*how*: which of the pre-registered quoters (S0 at the touch, S1 skewed and
gated, S2 one tick inside, S3 the regime guard), with which parameters and
simulator settings, on which days, and under which permission to read them.

Parameters arrive either by hand (a research run on the development block or on
days outside the study) or from ``configs/mm_prereg.yaml``'s frozen values
(:func:`frozen_plan`), in which case nothing here chooses anything.

Days of the registered held-out and boundary blocks are refused unless asked
for, and then they are opened only through
:meth:`~trading_research.market_making.prereg.HeldOutLedger.open`, which checks
the frozen configuration and the clean committed tree and records the read
(:func:`access_for`). Days outside every registered block belong to no study
and are read freely; that is what synthetic days are.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making import prereg
from trading_research.market_making.quoters import (
    FLAG_SIGNAL,
    INDEX_SIGNAL,
    LEAN_TRIGGER_SIGNAL,
    NO_QUOTES,
    InsideQuoter,
    MarketView,
    Quote,
    Quoter,
    Quotes,
    RegimeGuard,
    SkewQuoter,
    TouchQuoter,
)
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig, run_days

#: The quoters this branch can run. S4 and X1 need the reversion tapes of the
#: study and stay with its scripts.
STRATEGIES: tuple[str, ...] = ("S0", "S1", "S2", "S3")

#: Each strategy's parameters, and the registered value a hand-run falls back on.
PARAMETERS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": ("skew_bp", "k", "min_edge_bp"),
    "S2": ("skew_bp", "k", "min_edge_bp", "m_ticks"),
    "S3": ("skew_bp", "k", "min_edge_bp", "m_ticks", "guards", "action", "guard_window_min"),
}


@dataclass(frozen=True)
class MakerSources:
    """Where one instrument-day's book, prints and funding live."""

    book_roots: tuple[Path, ...] = (Path("data/book"), Path("data/book_fresh"))
    trades_root: Path = Path("data/trades")
    funding_root: Path | None = Path("data/funding")
    cache: Path | None = None


@dataclass(frozen=True)
class MakerPlan:
    """One quoter and the simulator settings it runs under.

    ``regime_policy`` wraps S0 to S2 in the guard (``guard_pull`` or
    ``guard_widen``) for ``guard_window_min`` after each flag; S3 is the guard
    already and takes its action from its parameters. ``lean`` > 0 makes the
    quoter lean on an external signal (see :func:`signal_tapes`).
    """

    strategy: str
    config: SimConfig
    params: Mapping[str, Any] = field(default_factory=dict)
    sigma_ref: float = 1.0
    regime_policy: str = "none"
    guard_window_min: float = 15.0
    lean: float = 0.0
    lean_window_s: float = 600.0

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.regime_policy not in ("none", "guard_pull", "guard_widen"):
            raise ValueError(f"unknown regime policy {self.regime_policy!r}")
        if self.strategy == "S3" and self.regime_policy != "none":
            raise ValueError("S3 is the regime guard; set its action in its parameters")
        if self.strategy == "S0" and self.regime_policy == "guard_widen":
            raise ValueError("the touch quoter has no half-spread to widen; use guard_pull")
        if self.lean and self.strategy != "S1":
            raise ValueError("the signal lean is built on S1")
        missing = [p for p in PARAMETERS[self.strategy] if p not in self.params]
        if missing:
            raise ValueError(f"{self.strategy} needs {missing}")
        make_quoter(self)  # refuse a plan whose quoter cannot be built

    @property
    def label(self) -> str:
        policy = "" if self.regime_policy == "none" else f"+{self.regime_policy}"
        return f"{self.strategy}{policy}{'+lean' if self.lean else ''}"


def _skew(plan: MakerPlan, maker_bp: float) -> SkewQuoter:
    p = plan.params
    return SkewQuoter(
        float(p["skew_bp"]), float(p["k"]), float(p["min_edge_bp"]), plan.sigma_ref, maker_bp
    )


def _inside(plan: MakerPlan, maker_bp: float) -> InsideQuoter:
    p = plan.params
    return InsideQuoter(
        float(p["skew_bp"]),
        float(p["k"]),
        float(p["min_edge_bp"]),
        int(p["m_ticks"]),
        plan.sigma_ref,
        maker_bp,
    )


def make_quoter(plan: MakerPlan) -> Any:
    """A fresh :class:`Quoter` for one simulated day. Module level, so it pickles.

    Typed ``Any`` because the quoters are frozen dataclasses, whose read-only
    ``name`` the protocol's plain attribute does not admit.
    """
    maker_bp = plan.config.fees.maker_bp
    if plan.strategy == "S0":
        base: Any = TouchQuoter()
    elif plan.strategy == "S1":
        base = _skew(plan, maker_bp)
    elif plan.strategy == "S2":
        base = _inside(plan, maker_bp)
    else:
        guarded = (
            _inside(plan, maker_bp) if plan.params["guards"] == "S2" else _skew(plan, maker_bp)
        )
        return RegimeGuard(
            guarded, str(plan.params["action"]), float(plan.params["guard_window_min"])
        )
    if plan.lean:
        base = SignalLean(base, plan.lean, plan.lean_window_s)
    if plan.regime_policy != "none":
        action = plan.regime_policy.removeprefix("guard_")
        name = f"{plan.strategy}+{plan.regime_policy}"
        if isinstance(base, TouchQuoter):
            return TouchGuard(plan.guard_window_min, name=name)
        return RegimeGuard(base, action, plan.guard_window_min, name=name)
    return base


@dataclass(frozen=True)
class SignalLean:
    """S1 leaning on an external signal: S4's mechanics, the signal in place of the index.

    For ``window_s`` after each signal (``lean_trigger_s``) the reservation
    price becomes ``r * (1 + lam * s * 1e-4)``, with ``s`` the signal's signed
    strength in bp (``index_return_bp``) read live, so a lean of one moves the
    quotes by the move the signal expects. Like S1 it exposes its prices, so
    the regime guard can pull or widen it.
    """

    base: SkewQuoter
    lam: float
    window_s: float
    name: str = "S1+lean"
    uses_future: bool = False

    def quote_prices(
        self, view: MarketView, position: float, *, delta_scale: float = 1.0
    ) -> tuple[int, int] | None:
        base = self.base
        if not base.usable(view):
            return None
        r = base.reservation(view, position)
        trigger = view.signals.get(LEAN_TRIGGER_SIGNAL, math.nan)
        s = view.signals.get(INDEX_SIGNAL, math.nan)
        if trigger == trigger and s == s and view.ts / 1e9 - trigger <= self.window_s:
            r *= 1.0 + self.lam * s * 1e-4
        return base.prices(view, r, base.half_spread_bp(view) * delta_scale)

    def quotes(self, view: MarketView, position: float) -> Quotes:
        prices = self.quote_prices(view, position)
        if prices is None:
            return NO_QUOTES
        return Quotes(Quote(prices[0], view.clip), Quote(prices[1], view.clip))


@dataclass(frozen=True)
class TouchGuard:
    """S0 pulled for ``window_min`` after each flag: the touch quoter's only guard."""

    window_min: float
    name: str = "S0+guard_pull"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        flag = view.signals.get(FLAG_SIGNAL, math.nan)
        if flag == flag and view.ts / 1e9 - flag <= self.window_min * 60.0:
            return NO_QUOTES
        return TouchQuoter().quotes(view, position)


def frozen_plan(
    strategy: str,
    symbol: str,
    *,
    path: Path | str = prereg.CONFIG_PATH,
    **changes: Any,
) -> MakerPlan:
    """The plan the pre-registration froze for ``strategy`` on ``symbol``.

    The clip and the volatility reference are the instrument's; the quoter's
    parameters and soft limit are the strategy's chosen cell (S3 takes the
    parameters of the quoter it guards). ``changes`` are simulator settings
    (fee tier, latency, cancellation attribution), which the registration
    treats as robustness axes rather than choices.
    """
    registration = prereg.PreRegistration.load(path).require_frozen()
    value = registration.value
    clip = dict(value(("simulator", "clip", "notional_usdt")))
    sigma = dict(value(("simulator", "volatility", "sigma_ref")))
    if symbol not in clip:
        raise ValueError(f"{symbol} has no frozen clip; the registration covers {sorted(clip)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    params: dict[str, Any] = {}
    if strategy == "S0":
        soft = float(value(("strategies", "S0", "soft_limit_clips")))
    else:
        guard = dict(value(("strategies", "S3", "chosen"))) if strategy == "S3" else {}
        chosen = dict(value(("strategies", guard.get("guards", strategy), "chosen")))
        soft = float(chosen.pop("soft_limit_clips"))
        params = {**chosen, **guard}
        if strategy == "S3" and "m_ticks" not in params:
            params["m_ticks"] = 2  # unread: S3 guards S1 here
    config = SimConfig(clip_notional=float(clip[symbol]), soft_limit_clips=soft)
    return MakerPlan(
        strategy, config.with_(**changes) if changes else config, params, float(sigma[symbol])
    )


@dataclass(frozen=True)
class OutsideStudy(prereg.Access):
    """Permission to read days that belong to no registered block, and nothing else."""

    def allows(self, day: date) -> bool:
        return prereg.block_of(day) is None


def access_for(
    days: Sequence[date],
    *,
    allow_heldout: bool = False,
    second_read: bool = False,
    repo: Path = Path(),
) -> prereg.Access:
    """The permission a run on ``days`` needs, refused or recorded as the protocol says.

    Days outside every block: :class:`OutsideStudy`. Development days:
    :func:`~trading_research.market_making.prereg.development_access`. Held-out
    or boundary days: refused unless ``allow_heldout``, and then opened only by
    the ledger, which refuses a second read unless ``second_read``.
    """
    blocks = {prereg.block_of(day) for day in days}
    study = {b for b in blocks if b is not None}
    if not study:
        return OutsideStudy(frozenset(), "outside the study")
    if None in blocks:
        raise prereg.HeldOutLocked("a run cannot mix days of the study with days outside it")
    if study <= {"D"}:
        return prereg.development_access()
    held = sorted(study - {"D"})
    if not allow_heldout:
        raise prereg.HeldOutLocked(
            f"days of block(s) {held} are held out; pass --allow-heldout to open them "
            "through the ledger, which records the read"
        )
    if set(held) - {"H", "F"}:
        raise prereg.HeldOutLocked(
            f"the ledger opens H and F only, not {sorted(set(held) - {'H', 'F'})}"
        )
    block = "F" if "F" in held else "H"
    return prereg.HeldOutLedger(repo=repo).open(block, force=second_read)


def flag_tape(
    symbol: str, days: Sequence[date], sources: MakerSources, access: prereg.Access
) -> SignalTape:
    """Both detectors' flags, run continuously up to the last day.

    For days of the study the detectors start where the study started them, on
    the first day of D; otherwise on the first day of the run.
    """
    from trading_research.market_making import flags

    first = prereg.BLOCKS["D"][0] if not isinstance(access, OutsideStudy) else min(days)
    span = [first + timedelta(days=i) for i in range((max(days) - first).days + 1)]
    timeline = flags.build_flags(symbol, span, book_roots=sources.book_roots, access=access)
    return timeline.tape()


def signal_tapes(ts_ns: np.ndarray, signed_strength: np.ndarray) -> dict[str, SignalTape]:
    """The two tapes the lean reads, from a signal's times and signed strength (bp)."""
    ts = np.asarray(ts_ns, dtype=np.int64)
    values = np.asarray(signed_strength, dtype=np.float64)
    last = np.r_[ts[1:] != ts[:-1], True] if len(ts) else np.zeros(0, dtype=bool)
    ts, values = ts[last], values[last]
    return {
        INDEX_SIGNAL: SignalTape(INDEX_SIGNAL, ts, values),
        LEAN_TRIGGER_SIGNAL: SignalTape(LEAN_TRIGGER_SIGNAL, ts, ts / 1e9),
    }


def run(
    symbol: str,
    days: Sequence[date],
    plan: MakerPlan,
    sources: MakerSources,
    *,
    access: prereg.Access,
    workers: int = 1,
    tapes: Mapping[str, SignalTape] | None = None,
) -> pd.DataFrame:
    """Simulate ``plan`` on each day: one row per day, as :func:`run_days` returns it."""
    access.require(days, what="simulate")
    factory: Callable[[], Quoter] = partial(make_quoter, plan)
    return run_days(
        symbol,
        days,
        factory,
        plan.config,
        book_roots=sources.book_roots,
        trades_root=sources.trades_root,
        funding_root=sources.funding_root,
        workers=workers,
        cache=sources.cache,
        tapes=tapes,
    )


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Days that were simulated and are not flagged out of verdicts."""
    return frame[(frame["status"] == "ok") & ~frame["excluded"].astype(bool)]


def attempts(frame: pd.DataFrame) -> pd.DataFrame:
    """One attempt per usable day, in basis points of the day's turnover."""
    days = usable(frame)
    turnover = (days["maker_turnover"] + days["taker_turnover"]).to_numpy(dtype=np.float64)
    scale = np.divide(1e4, turnover, out=np.zeros_like(turnover), where=turnover > 0)
    gross = days["gross"].to_numpy(dtype=np.float64) * scale
    net = days["net"].to_numpy(dtype=np.float64) * scale
    stamps = pd.to_datetime(days["day"]).astype("int64").to_numpy()
    return pd.DataFrame(
        {
            "symbol": days["symbol"].to_numpy(),
            "day": days["day"].to_numpy(),
            "ts": stamps,
            "filled": (days["fills"] > 0).to_numpy(),
            "gross_bp": gross,
            "cost_bp": gross - net,
            "net_bp": net,
            "path": "market_maker",
            "net": days["net"].to_numpy(dtype=np.float64),
            "turnover": turnover,
        }
    )


#: The decomposition's terms, in the order the README's tables use.
TERMS: tuple[str, ...] = ("spread", "adverse", "inventory", "fees", "funding")


def decomposition(frame: pd.DataFrame) -> dict[str, float]:
    """The daily means a market maker is judged on, over usable days."""
    days = usable(frame)
    if days.empty:
        return {"days": 0.0}
    turnover = float((days["maker_turnover"] + days["taker_turnover"]).sum())
    out = {
        "days": float(len(days)),
        "excluded_days": float(len(frame) - len(days)),
        "net_per_day": float(days["net"].mean()),
        "net_bp_of_turnover": float(days["net"].sum()) / turnover * 1e4 if turnover else math.nan,
        "fills_per_day": float(days["fills"].mean()),
        "max_abs_position": float(days["max_abs_position"].max()),
    }
    for term in TERMS:
        out[term] = float(days[term].mean())
    return out
