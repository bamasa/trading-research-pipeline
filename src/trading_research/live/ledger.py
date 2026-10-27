"""The paper account: fills at the dated fee, limits, a kill switch, a mark at every boundary.

The bookkeeping is :class:`trading_research.market_making.accounting.Account`,
the one every backtest here uses: average cost, cash kept apart from it, and
the identity ``realised + unrealised - fees == cash + position x mark``
checked after every fill and every mark, so a bookkeeping error stops the run
where it happened.

* **Fees.** Each fill pays the maker or taker fee its schedule
  (:mod:`trading_research.live.fees`) gives for its instant, in basis points
  of its notional, booked in the quote currency at the fill price. The quote's
  status (sourced, carried, undated, given, as of) is kept on the fill.
* **Limits.** An order is refused before it is sent if filling it in full
  could take the position beyond ``max_position`` (``refused`` counts them).
* **Kill switch.** Latched by the first of: net equity at a mark at or below
  ``-max_loss``; a position beyond ``max_position`` after a fill; no book row
  for ``stale_after_s``; a sequence gap (when ``kill_on_gap``). Once latched
  nothing new is sent; the execution flattens what is held.
* **Marks.** At every book row the position is marked at the mid: net equity
  is cash plus position times mid, gross adds back the fees paid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_research.live.fees import FeeSchedule
from trading_research.market_making.accounting import Account


@dataclass(frozen=True)
class Limits:
    """The paper account's limits, in base units and quote currency."""

    max_position: float
    max_loss: float
    stale_after_s: float = 5.0
    kill_on_gap: bool = True

    def __post_init__(self) -> None:
        if not self.max_position > 0 or not self.max_loss > 0 or not self.stale_after_s > 0:
            raise ValueError("max position, max loss and the stale threshold must be positive")


#: Columns of the fills table.
FILL_COLUMNS = (
    "ts_ns",
    "oid",
    "side",
    "price",
    "size",
    "notional",
    "maker",
    "fee_bp",
    "fee",
    "fee_status",
    "reason",
    "path",
    "position_after",
    "cash_after",
)


@dataclass
class PaperLedger:
    fees: FeeSchedule
    limits: Limits
    account: Account = field(default_factory=Account)
    fills: list[tuple[Any, ...]] = field(default_factory=list)
    #: ts, mid, position, net equity, fees paid so far.
    marks: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    killed: str | None = None
    killed_ns: int | None = None
    refused: int = 0
    fee_statuses: dict[str, int] = field(default_factory=dict)
    last_mid: float = float("nan")
    #: Fills booked since the last mark (the run's end marks again only if there were any).
    fills_since_mark: int = 0

    def kill(self, reason: str, ts_ns: int) -> None:
        if self.killed is None:
            self.killed = reason
            self.killed_ns = ts_ns

    def allows(self, side: int, size: float) -> bool:
        """Whether an order of ``size`` on ``side`` may be sent: not killed, within the limit."""
        if self.killed is not None:
            return False
        after = self.account.position + side * size
        if abs(after) > self.limits.max_position * (1 + 1e-12):
            self.refused += 1
            return False
        return True

    def fill(
        self,
        ts_ns: int,
        side: int,
        price: float,
        size: float,
        *,
        maker: bool,
        oid: int,
        reason: str,
        path: str,
    ) -> float:
        """Book one fill at the fee in force; returns the fee in quote currency."""
        quote = self.fees.quote(ts_ns)
        fee_bp = quote.maker_bp if maker else quote.taker_bp
        fee = self.account.fill(side, price, size, fee_bp, maker=maker)
        self.account.check_identity(price)
        self.fee_statuses[quote.status] = self.fee_statuses.get(quote.status, 0) + 1
        self.fills.append(
            (
                ts_ns,
                oid,
                side,
                price,
                size,
                price * size,
                maker,
                fee_bp,
                fee,
                quote.status,
                reason,
                path,
                self.account.position,
                self.account.cash,
            )
        )
        self.fills_since_mark += 1
        if abs(self.account.position) > self.limits.max_position * (1 + 1e-12):
            self.kill("max_position", ts_ns)
        return fee

    def mark(self, ts_ns: int, mid: float) -> float:
        """Mark at ``mid``; returns net equity and latches ``max_loss`` when it is reached."""
        self.last_mid = mid
        self.fills_since_mark = 0
        self.account.check_identity(mid)
        equity = self.account.equity(mid)
        self.marks.append((ts_ns, mid, self.account.position, equity, self.account.fees))
        if equity <= -self.limits.max_loss:
            self.kill("max_loss", ts_ns)
        return equity

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fills, columns=list(FILL_COLUMNS))

    def marks_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.marks, columns=["ts_ns", "mid", "position", "net", "fees"])
        frame["gross"] = frame["net"] + frame["fees"]
        return frame

    def summary(self) -> dict[str, Any]:
        """PnL gross and net at the fee used, turnover, drawdown, position, the kill."""
        account = self.account
        mark = self.last_mid
        net = account.equity(mark) if np.isfinite(mark) else account.cash
        marks = self.marks_frame()
        drawdown = 0.0
        if len(marks):
            equity = marks["net"].to_numpy()
            drawdown = float(np.max(np.maximum.accumulate(np.maximum(equity, 0.0)) - equity))
        turnover = account.maker_turnover + account.taker_turnover
        return {
            "net": float(net),
            "gross": float(net + account.fees),
            "fees": float(account.fees),
            "realised": float(account.realised),
            "position": float(account.position),
            "last_mid": float(mark),
            "turnover": float(turnover),
            "maker_turnover": float(account.maker_turnover),
            "taker_turnover": float(account.taker_turnover),
            "net_bp_turnover": float(net / turnover * 1e4) if turnover > 0 else float("nan"),
            "gross_bp_turnover": float((net + account.fees) / turnover * 1e4)
            if turnover > 0
            else float("nan"),
            "fills": len(self.fills),
            "max_drawdown": drawdown,
            "refused_by_limit": self.refused,
            "killed": self.killed,
            "killed_ns": self.killed_ns,
            "fee_statuses": dict(self.fee_statuses),
        }
