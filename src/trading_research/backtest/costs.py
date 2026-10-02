"""Transaction costs, in basis points.

One cost model, used in three places: when labelling, when deciding whether a
prediction is worth acting on, and when computing profit and loss. Backtests
routinely overstate results by labelling against a mid-price move that the
spread would have eaten, so the three must agree by construction rather than by
someone remembering to keep them in step.

Taker only
----------
This project models taking liquidity. Every entry and every exit crosses the
spread and pays the taker fee.

That is the harder assumption and the honest one for a directional model. A
maker strategy is not simply cheaper: a resting order fills only when someone
trades against it, which happens preferentially when the market is about to
move through it. Modelling that needs queue position and adverse-selection
assumptions this project does not have data for, and quoting maker economics
without them would flatter every result.

Published Binance USD-M futures rates are 0.02% maker and 0.05% taker — 2 and 5
basis points respectively. Taking on both sides therefore costs 10 bp before
the spread is paid at all, which is the single most important number in this
repository: it is what any predicted move has to clear before the prediction is
worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Binance USD-M futures taker fee at the standard tier, in basis points.
BINANCE_UM_TAKER_BP: float = 5.0


@dataclass(frozen=True)
class TakerCosts:
    """Round-trip cost of entering and leaving a position by taking liquidity.

    All figures are basis points of notional.

    ``fee_bp_per_side``
        Exchange fee. Charged on entry and on exit.
    ``slippage_bp``
        Everything the quoted book does not capture: the price moving between
        the decision and the order arriving, and the order walking past the top
        level when it is larger than what rests there. Charged per side.
    ``half_spread_multiplier``
        How much of the quoted spread a crossing order pays. One means the
        order crosses fully — buy at the ask, sell at the bid — which is the
        default because that is what taking liquidity means.
    """

    fee_bp_per_side: float = BINANCE_UM_TAKER_BP
    slippage_bp: float = 0.5
    half_spread_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name in ("fee_bp_per_side", "slippage_bp", "half_spread_multiplier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    def round_trip_bp(self, spread_bp: float | pd.Series) -> float | pd.Series:
        """Total cost of a round trip, given the spread prevailing at the time.

        The spread is charged once, not twice. Entering pays half of it against
        mid, and exiting on the other side pays the other half, so a full round
        trip costs one spread — the common mistake of charging it twice
        overstates costs about as badly as ignoring it understates them.
        """
        return (
            2.0 * self.fee_bp_per_side
            + self.half_spread_multiplier * spread_bp
            + 2.0 * self.slippage_bp
        )

    def entry_bp(self, spread_bp: float | pd.Series) -> float | pd.Series:
        """Cost of the entry leg alone, against mid."""
        return (
            self.fee_bp_per_side + 0.5 * self.half_spread_multiplier * spread_bp + self.slippage_bp
        )

    def describe(self) -> dict[str, object]:
        return {
            "execution": "taker",
            "fee_bp_per_side": self.fee_bp_per_side,
            "slippage_bp": self.slippage_bp,
            "half_spread_multiplier": self.half_spread_multiplier,
            "note": (
                "Taker on both legs. Maker economics are not modelled: a resting "
                "order fills preferentially when the market is about to move "
                "through it, and that adverse selection cannot be estimated from "
                "this data."
            ),
        }


def breakeven_share(forward_move_bp: pd.Series, cost_bp: float | pd.Series) -> float:
    """Share of moments whose future move is larger than the cost of trading.

    An upper bound on how often a strategy could possibly be right to trade,
    and it assumes the direction is predicted perfectly. Useful as a reality
    check before any model is fitted: if only a small fraction of moments could
    pay for themselves under perfect foresight, no classifier is going to
    rescue the horizon.
    """
    move = forward_move_bp.abs()
    return float((move > cost_bp).mean())


# ---------------------------------------------------------------------------
# Fee tiers, for the market maker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeTier:
    """One row of a venue's fee schedule, in basis points of notional.

    A negative ``maker_bp`` is a rebate: the venue pays the passive side. The
    market-making simulator charges ``maker_bp`` on every passive fill and
    ``taker_bp`` on every flatten. ``group`` says which instruments the row
    applies to (``all``, or one of the venue's instrument groups) and
    ``requires`` what qualifies an account for it, as the venue states it.
    """

    name: str
    maker_bp: float
    taker_bp: float
    group: str = "all"
    requires: str = ""

    def __post_init__(self) -> None:
        if self.taker_bp < 0:
            raise ValueError(f"a taker fee cannot be a rebate, got {self.taker_bp}")
        if self.maker_bp > self.taker_bp:
            raise ValueError(
                f"maker fee {self.maker_bp} above taker fee {self.taker_bp} is not a schedule"
            )
        if self.group not in FEE_GROUPS:
            raise ValueError(f"unknown fee group {self.group!r}; known: {FEE_GROUPS}")


#: Instrument groups a row of Bybit's schedule can apply to. ``g1`` is what the
#: venue calls its major coins; ``altcoin`` is every other perpetual, mid- and
#: long-tail included.
FEE_GROUPS: tuple[str, ...] = ("all", "g1", "altcoin")

#: Bybit's base tier for USDT perpetuals (VIP 0): 0.02% maker, 0.055% taker. The
#: figure :class:`trading_research.backtest.maker.MakerCosts` already uses, and
#: the one the market-making pre-registration fixes for every verdict.
BYBIT_BASE = FeeTier("base", 2.0, 5.5, requires="none (VIP 0)")

#: Where the schedule was transcribed from: the help-centre page "Trading Fee
#: Structure", tabs "Non-VIP & VIP users" and "Pro Users & Market Makers", which
#: said "Last updated on 2026-09-02". The page refuses scripted requests, so it
#: was read in a browser.
BYBIT_FEE_SOURCE = "https://www.bybit.com/en/help-center/article/Trading-Fee-Structure"

#: When the rows below were transcribed from the source.
BYBIT_FEE_RETRIEVED: str = "2026-10-02"

#: The caveat every number using a tier other than the base must carry.
BYBIT_FEE_CAVEAT = (
    "Bybit's linear-perpetual schedule as published in 2026 (page last updated "
    "2026-09-02, transcribed 2026-10-02), applied to data from February to April "
    "2024, when the schedule may have differed. Volumes are 30-day derivatives "
    "volume in USDT; Pro levels require more than 20% of volume through the API."
)

#: The published linear-perpetual tiers, in the order the venue lists them:
#: VIP levels first (one row for every instrument), then the Pro levels, whose
#: rates differ between major coins (``g1``) and every other perpetual
#: (``altcoin``). Only complete rows are here; the market-maker programme's
#: rebate, which has no published taker rate, is :data:`BYBIT_MM_PROGRAMME_MAKER_BP`.
BYBIT_LINEAR_TIERS: tuple[FeeTier, ...] = (
    BYBIT_BASE,
    FeeTier("vip1", 1.8, 4.0, requires=">= 10M volume, or >= 100K asset balance"),
    FeeTier("vip2", 1.6, 3.75, requires=">= 25M volume"),
    FeeTier("vip3", 1.4, 3.5, requires=">= 50M volume"),
    FeeTier("vip4", 1.2, 3.2, requires=">= 100M volume, API volume <= 20%"),
    FeeTier("vip5", 1.0, 3.2, requires=">= 250M volume, API volume <= 20%"),
    FeeTier("supreme_vip", 0.0, 3.0, requires=">= 500M volume, API volume <= 20%"),
    FeeTier("pro1_g1", 1.0, 2.8, "g1", ">= 100M volume, API volume > 20%"),
    FeeTier("pro1_altcoin", 0.0, 2.8, "altcoin", ">= 100M volume, API volume > 20%"),
    FeeTier("pro2_g1", 0.5, 2.5, "g1", ">= 250M volume, API volume > 20%"),
    FeeTier("pro2_altcoin", 0.0, 2.8, "altcoin", ">= 250M volume, API volume > 20%"),
    FeeTier("pro3_g1", 0.25, 2.2, "g1", ">= 750M volume, API volume > 20%"),
    FeeTier("pro3_altcoin", 0.0, 2.5, "altcoin", ">= 750M volume, API volume > 20%"),
    FeeTier("pro4_g1", 0.1, 2.0, "g1", ">= 1,500M volume, API volume > 20%"),
    FeeTier("pro4_altcoin", 0.0, 2.3, "altcoin", ">= 1,500M volume, API volume > 20%"),
    FeeTier("pro5_g1", 0.0, 1.8, "g1", ">= 3,000M volume, API volume > 20%"),
    FeeTier("pro5_altcoin", 0.0, 2.1, "altcoin", ">= 3,000M volume, API volume > 20%"),
    FeeTier("pro6_g1", 0.0, 1.5, "g1", ">= 5,000M volume, API volume > 20%"),
    FeeTier("pro6_altcoin", 0.0, 1.8, "altcoin", ">= 5,000M volume, API volume > 20%"),
)

#: The market-maker incentive programme, as published: "up to 0.01% maker fee
#: rebate", subject to the programme's requirements and an application. A
#: bound on the maker fee, not a row of the schedule: no taker rate is published
#: for it, so it is kept apart from :data:`BYBIT_LINEAR_TIERS` rather than
#: completed with an assumed one.
BYBIT_MM_PROGRAMME_MAKER_BP: float = -1.0

#: The instruments the page names as major coins (``g1``), by example. The
#: page's list is illustrative ("e.g."), so only these four are classified from
#: it; the market-making study's other instruments, BICOUSDT and CRVUSDT, are
#: mid-tail perpetuals and fall under ``altcoin``.
BYBIT_G1_EXAMPLES: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

#: Groups of the instruments the market-making study uses.
BYBIT_STUDY_GROUPS: dict[str, str] = {
    "BTCUSDT": "g1",
    "XRPUSDT": "g1",
    "BICOUSDT": "altcoin",
    "CRVUSDT": "altcoin",
}


def fee_tier(name: str, tiers: tuple[FeeTier, ...] = BYBIT_LINEAR_TIERS) -> FeeTier:
    """Look a tier up by name, refusing one that has not been transcribed."""
    for tier in tiers:
        if tier.name == name:
            return tier
    known = ", ".join(t.name for t in tiers)
    raise KeyError(f"no fee tier {name!r}; transcribed tiers: {known}")


def tiers_for(group: str, tiers: tuple[FeeTier, ...] = BYBIT_LINEAR_TIERS) -> list[FeeTier]:
    """The rows that apply to an instrument of ``group``: every-instrument rows
    and the group's own."""
    if group not in FEE_GROUPS or group == "all":
        raise ValueError(f"group must be one of {FEE_GROUPS[1:]}, got {group!r}")
    return [t for t in tiers if t.group in ("all", group)]


def best_maker_tier(group: str, tiers: tuple[FeeTier, ...] = BYBIT_LINEAR_TIERS) -> FeeTier:
    """The published row with the lowest maker fee for ``group``; among equal
    maker fees, the lowest taker fee, then the first listed."""
    rows = tiers_for(group, tiers)
    return min(rows, key=lambda t: (t.maker_bp, t.taker_bp))


def breakeven_maker_bp(net: float, turnover: float, maker_bp: float) -> float:
    """The maker fee at which ``net`` would have been zero.

    ``net`` was earned at ``maker_bp`` over ``turnover`` of passive notional.
    Every basis point less fee adds ``turnover * 1e-4`` to the net, so the
    break-even is ``maker_bp + net / turnover * 1e4``. Exact only for a strategy
    whose decisions do not read the fee — the touch quoter does not; a gated
    strategy quotes differently at a different fee and has to be re-run.
    """
    if turnover <= 0:
        raise ValueError(f"turnover must be positive, got {turnover}")
    return maker_bp + net / turnover * 1e4
