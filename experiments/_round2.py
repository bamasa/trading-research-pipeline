"""What the second round's scripts share: where the data is, and how a strategy is built.

The development run (``experiments/market_making_round2.py``) and the held-out
run (``experiments/market_making_round2_heldout.py``) build every quoter, every
simulator setting and every gate tape through these functions, so a value
frozen on D is used on the held-out block exactly as it was chosen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.backtest.costs import (
    BYBIT_BASE,
    BYBIT_MM_PROGRAMME_MAKER_BP,
    FeeTier,
    fee_tier,
)
from trading_research.market_making import gate, heldout
from trading_research.market_making.quoters import Quoter, SkewQuoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig

BOOK_ROOTS = (Path("data/book"), Path("data/book_fresh"))
TRADES_ROOT = Path("data/trades")
FUNDING_ROOT = Path("data/funding")
OUT_ROOT = Path("artifacts/mm/round2")
MARKOUTS_ROOT = OUT_ROOT / "markouts"

#: Markouts kept on every main fill.
MARKOUT_HORIZONS_S = (1.0, 5.0, 30.0)

#: The minimum mean passive fills a day for an S1 cell to be ranked.
MIN_FILLS_S1 = 20.0
#: The minimum mean over instruments of passive fills a day for a gate cell.
MIN_FILLS_GATE = 5.0


def report(message: str) -> None:
    print(message, flush=True)


def sig(value: float, digits: int = 6) -> float:
    """``value`` to ``digits`` significant figures, as it is frozen."""
    if value == 0 or value != value or value in (float("inf"), float("-inf")):
        return float(value)
    return float(f"{value:.{digits}g}")


def plain(value: Any) -> Any:
    """A grid value as the YAML holds it: int when whole, else float."""
    number = float(value)
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# Fee tiers
# ---------------------------------------------------------------------------


def tier(name: str, *, taker_bp: float | None = None) -> FeeTier:
    """A registered tier: ``base``, ``pro1_altcoin`` or ``programme`` (maker
    -1.0 bp, taker assumed equal to ``pro1_altcoin``'s), optionally with
    another taker fee."""
    if name == "base":
        chosen = BYBIT_BASE
    elif name == "pro1_altcoin":
        chosen = fee_tier("pro1_altcoin")
    elif name == "programme":
        taker = fee_tier("pro1_altcoin").taker_bp
        chosen = FeeTier(
            "programme", BYBIT_MM_PROGRAMME_MAKER_BP, taker, "altcoin", "on application"
        )
    else:
        raise ValueError(f"no registered tier {name!r}")
    if taker_bp is None:
        return chosen
    return FeeTier(f"{chosen.name}|taker={taker_bp:g}", chosen.maker_bp, taker_bp, chosen.group)


def grid_tier(maker_bp: float) -> FeeTier:
    """A point of the registered fee grid: the maker fee moved, the taker at base."""
    return FeeTier(f"grid_{maker_bp:+.2f}", maker_bp, BYBIT_BASE.taker_bp)


# ---------------------------------------------------------------------------
# Instruments and quoters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrument:
    symbol: str
    clip: float
    sigma_ref: float


def config(
    instrument: Instrument, soft_limit_clips: float, fees: FeeTier = BYBIT_BASE, **changes: Any
) -> SimConfig:
    base = SimConfig(
        clip_notional=instrument.clip,
        soft_limit_clips=float(soft_limit_clips),
        fees=fees,
        markout_horizons_s=MARKOUT_HORIZONS_S,
    )
    return base.with_(**changes) if changes else base


def s1(instrument: Instrument, cell: Mapping[str, float], maker_bp: float = 2.0) -> SkewQuoter:
    """S1 at a cell of its grid (``skew_bp``, ``k``, ``min_edge_bp``)."""
    return SkewQuoter(
        float(cell["skew_bp"]),
        float(cell["k"]),
        float(cell["min_edge_bp"]),
        instrument.sigma_ref,
        maker_bp,
    )


def s1_touch(
    instrument: Instrument, cell: Mapping[str, float], maker_bp: float = 2.0
) -> SkewQuoter:
    """S1 at ``k`` = 0 and no minimum edge, with the cell's skew."""
    return SkewQuoter(
        float(cell["skew_bp"]), 0.0, 0.0, instrument.sigma_ref, maker_bp, name="S1-touch"
    )


def base_quoter(
    base: str, instrument: Instrument, cell: Mapping[str, float], maker_bp: float = 2.0
) -> SkewQuoter:
    if base == "S1":
        return s1(instrument, cell, maker_bp)
    if base == "S1_touch":
        return s1_touch(instrument, cell, maker_bp)
    raise ValueError(f"no base {base!r}")


def gated(
    cell: gate.GateCell,
    instrument: Instrument,
    s1_cell: Mapping[str, float],
    maker_bp: float = 2.0,
    *,
    signal: str | None = None,
) -> gate.GatedQuoter:
    """``G(base)`` for a gate cell, reading the cell's state tape (or ``signal``)."""
    base = base_quoter(cell.base, instrument, s1_cell, maker_bp)
    return gate.GatedQuoter(base, signal=signal or cell.signal, name=f"G[{cell.label}]")


def masks(hourly: pd.DataFrame, maker_bp: float) -> dict[float, tuple[int, ...]]:
    """The hour masks at every registered margin, for the fee paid."""
    return {m: gate.hour_mask(hourly, maker_bp, m) for m in gate.MARGINS_BP}


def state(
    days: Mapping[date, gate.PrintMarkouts],
    cell: gate.GateCell,
    maker_bp: float,
    hourly: pd.DataFrame,
) -> SignalTape:
    """The cell's state tape over ``days``, at the fee the quoter pays."""
    mask = gate.hour_mask(hourly, maker_bp, float(cell.margin_bp))
    return gate.state_tape(list(days.values()), cell, maker_bp, mask)


def same(quoter: Quoter) -> Quoter:
    return quoter


def constant(quoter: Quoter) -> Callable[[], Quoter]:
    """A picklable factory for a stateless (frozen) quoter."""
    return partial(same, quoter)


def spec(
    label: str,
    quoter: Quoter,
    settings: SimConfig,
    group: str = "main",
    tapes: Mapping[str, SignalTape] | None = None,
    *,
    keep: bool = False,
) -> heldout.Spec:
    return heldout.Spec(label, constant(quoter), settings, group=group, tapes=tapes, keep=keep)


# ---------------------------------------------------------------------------
# S1's grid
# ---------------------------------------------------------------------------

S1_AXES = ("skew_bp", "k", "min_edge_bp", "soft_limit_clips")


def s1_label(cell: Mapping[str, Any]) -> str:
    return "S1|" + "/".join(f"{axis}={plain(cell[axis])}" for axis in S1_AXES)


def s1_cell(values: Sequence[float]) -> dict[str, float]:
    return dict(zip(S1_AXES, (float(v) for v in values), strict=True))


def nested_days(days: Sequence[date], budgets: Sequence[int], seed: int) -> dict[int, list[date]]:
    """The seeded, nested subsets: budget b takes the first b of one permutation."""
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(days))
    return {b: sorted(days[int(i)] for i in order[:b]) for b in budgets}


def passive_fills(rows: pd.DataFrame) -> pd.Series:
    """Passive fills a day, flattens excluded, per usable row."""
    ok = rows[(rows["status"] == "ok") & ~rows["excluded"].astype(bool)]
    return ok["fills"] - ok["flattens"]
