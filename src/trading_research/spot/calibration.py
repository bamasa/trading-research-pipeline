"""The print rule's calibration on Bybit (FILL-S), its thinned form, and the capacity ratio.

**FILL-S** (the registration, "Measurements without a hypothesis"). The
orders of a book-walk run, the 2025 executor's trades
(``backtest/taker.execute_day``) on S25-PB's or S25-BB's decisions, are each
priced at their own arrival twice: by the book walk, the run's own price, and
by the side-correct print rule on Bybit's prints, under the rule and every
bracket (:func:`trading_research.spot.fills.fill_orders`). An entry may fill
only at a print before 23:59, as the print executor's; an exit takes any print
before the day ends and falls back as the executor's does; neither is held to
a print after an entry fill, since the book run's entry has none. The stale
bracket reads each order's own decision (the entry's row, the exit's
decision). Per order, in bp of the walked snapshot's mid: the difference
(print less book) and the **advantage**, positive when the print rule is
better for the account (book less print for a buy, print less book for a
sell). Per side: their means and medians and the share of late prints.

**FILL-S thinned.** Each day, Bybit's prints thinned at random to BTC/FDUSD's
count of aggregated prints that day (all kept when Bybit has fewer), with
seeds 0 to 9 (:func:`thin_prints`); the orders of S25-PB's decisions, made on
the full prints, priced on the thinned prints and by the book walk.
``a_buy`` and ``a_sell`` (:func:`k_fill_charges`) are the worse-of-two
bracket's mean advantage per side over every fill of the ten seeds on the
block: what K-fill charges, as ``max(0, a)`` bp of each fill's notional.

**Capacity.** Each day, for each side, a slot's taker volume over the
venue's taker volume on that side from its prints, in BTC, every print
counting; the larger of the two (:func:`capacity_ratio`); its median over the
block's days (:func:`capacity_median`), a day without a trade counting as 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

import numpy as np
import pandas as pd

from trading_research.spot.fills import (
    BRACKETS,
    BUY,
    Fills,
    PrintFillSettings,
    PrintTape,
    fill_orders,
)

BP: float = 1e4
SEEDS: tuple[int, ...] = tuple(range(10))


def book_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Every entry and exit of a book-walk run, one row per order, in time order.

    ``trades`` is the 2025 executor's per-trade table (``backtest/taker.TRADE_COLUMNS``).
    """
    entries = pd.DataFrame(
        {
            "kind": "entry",
            "side": trades["side"].to_numpy(dtype=np.int64),
            "size": trades["size"].to_numpy(dtype=np.float64),
            "decision_ns": trades["decision_ns"].to_numpy(dtype=np.int64),
            "arrival_ns": trades["entry_arrival_ns"].to_numpy(dtype=np.int64),
            "book_price": trades["entry_price"].to_numpy(dtype=np.float64),
            "book_mid": trades["entry_mid"].to_numpy(dtype=np.float64),
        }
    )
    exits = pd.DataFrame(
        {
            "kind": "exit",
            "side": -trades["side"].to_numpy(dtype=np.int64),
            "size": trades["size"].to_numpy(dtype=np.float64),
            "decision_ns": trades["exit_decision_ns"].to_numpy(dtype=np.int64),
            "arrival_ns": trades["exit_arrival_ns"].to_numpy(dtype=np.int64),
            "book_price": trades["exit_price"].to_numpy(dtype=np.float64),
            "book_mid": trades["exit_mid"].to_numpy(dtype=np.float64),
        }
    )
    orders = pd.concat([entries, exits], ignore_index=True)
    return orders.sort_values("arrival_ns", kind="stable").reset_index(drop=True)


def price_orders(
    tape: PrintTape, orders: pd.DataFrame, settings: PrintFillSettings | None = None
) -> pd.DataFrame:
    """``orders`` with the print rule's prices, differences and advantages (module docstring)."""
    out = orders.reset_index(drop=True).copy()
    if out.empty:
        return out
    sizes = out["size"].unique()
    if len(sizes) != 1:
        raise ValueError("one clip per run")
    settings = settings or PrintFillSettings(clip=float(sizes[0]))
    side = out["side"].to_numpy(dtype=np.int64)
    arrival = out["arrival_ns"].to_numpy(dtype=np.int64)
    decision = out["decision_ns"].to_numpy(dtype=np.int64)
    entry = (out["kind"] == "entry").to_numpy()
    until = np.where(entry, tape.flatten_ns(settings), tape.end_ns)
    fills = [
        fill_orders(
            tape,
            side[m],
            arrival[m],
            decision[m],
            until_ns=until[m],
            fallback=fallback,
            settings=settings,
        )
        for m, fallback in ((entry, False), (~entry, True))
    ]
    book, mid = out["book_price"].to_numpy(), out["book_mid"].to_numpy()
    for name in ("found", "fallback", "late", "wait_ns", "beyond", *BRACKETS):
        column = np.empty(len(out), dtype=np.asarray(getattr(fills[0], name)).dtype)
        for m, f in zip((entry, ~entry), fills, strict=True):
            column[m] = getattr(f, name)
        out[name] = column
    for name in BRACKETS:
        diff = (out[name].to_numpy() - book) / mid * BP
        out[f"diff_{name}_bp"] = diff
        out[f"adv_{name}_bp"] = np.where(side == BUY, -diff, diff)
    return out


def side_summary(priced: pd.DataFrame, bracket: str = "rule") -> pd.DataFrame:
    """Per side: priced orders, mean and median advantage and difference, share late."""
    if bracket not in BRACKETS:
        raise ValueError(f"unknown bracket {bracket!r}")
    rows = []
    for side, name in ((1, "buy"), (-1, "sell")):
        mine = priced[(priced["side"] == side) & priced[bracket].notna()]
        adv, diff = mine[f"adv_{bracket}_bp"], mine[f"diff_{bracket}_bp"]
        rows.append(
            {
                "side": name,
                "orders": len(mine),
                "mean_advantage_bp": float(adv.mean()) if len(mine) else np.nan,
                "median_advantage_bp": float(adv.median()) if len(mine) else np.nan,
                "mean_difference_bp": float(diff.mean()) if len(mine) else np.nan,
                "median_difference_bp": float(diff.median()) if len(mine) else np.nan,
                "share_late": float(mine["late"].mean()) if len(mine) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("side")


def thin_prints(prints: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """``count`` of ``prints`` drawn uniformly without replacement, archive order kept.

    All are kept when there are no more than ``count``. The draw is
    ``numpy.random.default_rng(seed).choice`` over the prints' positions.
    """
    if count < 0:
        raise ValueError(f"the count must be non-negative, got {count}")
    if len(prints) <= count:
        return prints
    keep = np.sort(np.random.default_rng(seed).choice(len(prints), size=count, replace=False))
    return prints.iloc[keep]


def thinned_calibration(
    prints: pd.DataFrame,
    count: int,
    trades: pd.DataFrame,
    day: date,
    seeds: Sequence[int] = SEEDS,
    settings: PrintFillSettings | None = None,
) -> pd.DataFrame:
    """One day of FILL-S thinned: every order of ``trades`` priced on each seed's prints."""
    orders = book_orders(trades)
    frames = []
    for seed in seeds:
        tape = PrintTape.from_frame(thin_prints(prints, count, seed), day)
        frames.append(price_orders(tape, orders, settings).assign(seed=seed, day=day))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def k_fill_charges(priced: Iterable[pd.DataFrame]) -> dict[str, float]:
    """``a_buy`` and ``a_sell``: the worse-of-two mean advantage per side over every fill.

    ``priced`` are a block's thinned days (every seed). An order the print rule
    could not price (an entry no print followed) is no fill and is left out.
    """
    frames = [f for f in priced if len(f)]
    if not frames:
        return {"a_buy": float("nan"), "a_sell": float("nan")}
    every = pd.concat(frames, ignore_index=True)
    filled = every[every["worse_of_two"].notna()]
    return {
        "a_buy": float(filled.loc[filled["side"] == 1, "adv_worse_of_two_bp"].mean()),
        "a_sell": float(filled.loc[filled["side"] == -1, "adv_worse_of_two_bp"].mean()),
    }


def k_fill_fee_bp(charges: dict[str, float], side: int) -> float:
    """The extra fee K-fill charges a fill of ``side``, in bp of its notional."""
    value = charges["a_buy" if side == BUY else "a_sell"]
    if np.isnan(value):
        raise ValueError("no thinned fill on this side: K-fill has no charge to apply")
    return max(0.0, value)


def capacity_ratio(fill_side: np.ndarray, fill_size: np.ndarray, prints: pd.DataFrame) -> float:
    """One day: the larger over sides of the slot's taker volume over the venue's on that side."""
    side = np.asarray(fill_side, dtype=np.int64)
    size = np.asarray(fill_size, dtype=np.float64)
    aggressor = prints["aggressor"].to_numpy(dtype=np.int64)
    venue = prints["size"].to_numpy(dtype=np.float64)
    ratios = []
    for s in (1, -1):
        mine = float(size[side == s].sum())
        theirs = float(venue[aggressor == s].sum())
        if mine == 0.0:
            ratios.append(0.0)
        else:
            ratios.append(mine / theirs if theirs > 0 else float("inf"))
    return max(ratios)


def trade_fills(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sides and sizes of every fill of a per-trade table with ``side`` and ``size``."""
    side = trades["side"].to_numpy(dtype=np.int64)
    size = trades["size"].to_numpy(dtype=np.float64)
    return np.concatenate([side, -side]), np.concatenate([size, size])


def capacity_median(daily_ratios: Sequence[float] | np.ndarray) -> float:
    """The median of a block's daily capacity ratios (NaN for a block without days)."""
    values = np.asarray(daily_ratios, dtype=np.float64)
    return float(np.median(values)) if len(values) else float("nan")


def print_rule_checks(tape: PrintTape, fills: Iterable[Fills], clip: float) -> dict[str, float]:
    """Stage 0's records of the print rule: the unreachable share, the median sweep / clip.

    ``fills`` are the rule's fills on the tape (an entry no print followed, or
    an exit priced by the fallback, has no sweep and is left out).
    """
    chosen = list(fills)
    sweep = np.concatenate([np.asarray(f.sweep)[f.found] for f in chosen] or [np.zeros(0)])
    beyond = np.concatenate([np.asarray(f.beyond_sweep)[f.found] for f in chosen] or [np.zeros(0)])
    return {
        "prints": float(len(tape)),
        "unreachable_share": float(tape.rpi.mean()) if len(tape) else float("nan"),
        "median_sweep_over_clip": float(np.median(sweep) / clip) if len(sweep) else float("nan"),
        "share_beyond_sweep": float(beyond.mean()) if len(beyond) else float("nan"),
    }
