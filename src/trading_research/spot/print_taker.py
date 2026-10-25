"""The taker executor on prints: a clip filled by the side-correct print rule and filled back after the hold.

The print slots' executor (S25-PF on Binance BTC/FDUSD; S25-PB and S25-PFT as
measured rows), as registered under "Fills", "On prints, the side-correct
rule". The rules, in the order they apply to one decision at a row stamp ``t``
of the print grid:

* **Entry hours.** No entry decided before 00:01 or after 23:58 UTC
  (``entries_outside_hours``).
* **One position at a time.** A decision is taken only at a row stamped at or
  after the previous exit's fill (``entries_busy`` otherwise). On prints a buy
  and a sell fill at different prints, so the time a position keeps the
  account busy depends on its side.
* **Entry.** The order arrives at ``t`` + latency and fills by the print rule
  (:mod:`.fills_standin`, to be ``spot/fills.py``): at the first reachable print
  of its side stamped at or after its arrival and before 23:59, sweeping that
  stamp's prints of its side. No entry is skipped for depth or for wait. An
  entry no such print follows is not made (``entries_no_print``).
* **Exit**, decided as the print label decides it: at the first row stamped at
  or after the entry fill's print + ``hold`` (at that instant itself when no
  row follows), or at 23:59 if that comes first. It arrives after the latency
  and fills by the same rule before the day ends. An exit no print of its side
  follows is priced, for the whole clip, at the last reachable print of its
  side before its arrival (``exits_no_print``, its age kept as a negative
  wait); if its side has none that day, at the last reachable print of either
  side (``exits_no_print_either_side``). Its fill time is then its arrival.
* **Counted, excluding nothing**: a fill whose print comes more than
  ``late_print_ns`` (1 s) after its arrival (``entry_late_print``,
  ``exit_late_print``), and a fill with part of the clip beyond its sweep
  (``fill_beyond_sweep``, more than ``1e-9`` of the clip).
* **Money.** Each fill pays the taker fee the fee table assigns its venue, tier
  and instant (the fill's print, or the arrival of an exit priced without
  one), checked against the table on every fill (K-units). The account's
  identity holds after every fill and at the day's end, with turnover equal to
  the sum of the fill notionals (K-acct). There is no funding.

The brackets (``rule`` in the settings) change only the prices of fills the
rule finds, never which print fills or when, so every bracket keeps the true
run's busy periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from trading_research.market_making.accounting import Account, IdentityError
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.spot import fills_standin as fills
from trading_research.spot.costs import FeeTable, check_fill_fees, fill_fee, instrument_key

DayPrints = fills.DayPrints
#: A fill with more than this share of the clip beyond its sweep is counted.
BEYOND_SHARE = 1e-9
NS_PER_DAY = 86_400 * NS_PER_S


def clock_ns(moment: time) -> int:
    """A time of day as nanoseconds after midnight."""
    seconds = moment.hour * 3600 + moment.minute * 60 + moment.second
    return seconds * NS_PER_S + moment.microsecond * 1000


@dataclass(frozen=True, kw_only=True)
class PrintSettings:
    """What the print executor depends on besides the prints; fees only from ``table``."""

    table: FeeTable
    instrument: str
    tier: str
    clip: float = 0.010
    latency_ns: int = 10_000_000
    late_print_ns: int = NS_PER_S
    window_ns: int = fills.WINDOW_NS
    rule: str = "rule"
    no_entry_before: time = time(0, 1)
    no_entry_after: time = time(23, 58)
    flatten_at: time = time(23, 59)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument", instrument_key(self.instrument))
        if self.tier not in self.table.tiers(self.instrument):
            raise ValueError(f"the fee table lists no tier {self.tier!r} for {self.instrument}")
        if self.rule not in fills.RULES:
            raise ValueError(f"unknown fill rule {self.rule!r}; registered: {fills.RULES}")
        if not self.clip > 0:
            raise ValueError(f"the clip must be positive, got {self.clip}")
        if self.latency_ns < 0:
            raise ValueError("latency cannot be negative")
        if not self.no_entry_after <= self.flatten_at:
            raise ValueError("the flatten cannot come before the last entry")

    def fee_bp(self, ns: np.ndarray) -> np.ndarray:
        """The taker fee the table assigns each fill instant."""
        return self.table.taker_bp(self.instrument, self.tier, ns)


def print_settings(
    table: FeeTable,
    instrument: str,
    tier: str | None = None,
    *,
    latency_ns: int = 10_000_000,
    rule: str = "rule",
) -> PrintSettings:
    """The registered settings at ``tier`` (the venue's verdict tier by default)."""
    tier = tier or table.verdict_tier(instrument)
    return PrintSettings(table=table, instrument=instrument, tier=tier, latency_ns=latency_ns,
                         rule=rule)  # fmt: skip


#: The per-trade table's columns, in order. A wait is the fill's print stamp less
#: the order's arrival; an exit priced without a later print has a negative one,
#: minus the age of the print it was priced at.
PRINT_TRADE_COLUMNS: tuple[str, ...] = (
    "decision_ns", "side", "size",
    "entry_arrival_ns", "entry_print_ns", "entry_position", "entry_price", "entry_beyond",
    "entry_fee_bp", "entry_wait_ns", "entry_late",
    "exit_decision_ns", "exit_arrival_ns", "exit_print_ns", "exit_position", "exit_fill_ns",
    "exit_price", "exit_beyond", "exit_fee_bp", "exit_wait_ns", "exit_late", "exit_reason",
    "exit_no_print", "hold_realised_ns",
    "notional", "exit_notional", "gross", "fees", "net", "gross_bp", "net_bp",
)  # fmt: skip

#: Counters every day reports, zero when nothing happened.
PRINT_COUNTERS: tuple[str, ...] = (
    "decisions",
    "entries",
    "entries_outside_hours",
    "entries_busy",
    "entries_no_print",
    "exits_no_print",
    "exits_no_print_either_side",
    "entry_late_print",
    "exit_late_print",
    "fill_beyond_sweep",
)


@dataclass(frozen=True)
class PrintDay:
    """One day of the print executor: its round trips and counters."""

    trades: pd.DataFrame
    counters: dict[str, float]
    net: float
    turnover: float

    def day_bp(self, column: str = "net") -> float:
        """The day's value: the sum of ``column`` (quote) over the entry notionals, x 1e4;
        NaN for a day without a trade."""
        notional = float(self.trades["notional"].sum())
        if notional <= 0:
            return float("nan")
        return float(self.trades[column].sum()) / notional * 1e4


def day_sweeps(prints: DayPrints, settings: PrintSettings) -> dict[int, fills.Sweeps]:
    """Both sides' sweeps for the clip, the pessimistic window ending at the flatten."""
    flatten = day_start_ns(prints.day) + clock_ns(settings.flatten_at)
    return {
        s: fills.sweeps(
            prints, s, settings.clip, window_end_ns=flatten, window_ns=settings.window_ns
        )
        for s in (1, -1)
    }


def no_print_exits(
    own: fills.Sweeps, other: fills.Sweeps, arrival_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exits of ``own``'s side that no print follows: ``(position, print_ns, price,
    either_side)`` of the last reachable print of that side before each arrival,
    or, where the side has none, of the other side's."""
    arrival = np.asarray(arrival_ns, dtype=np.int64)
    position = np.full(arrival.shape, -1, dtype=np.int64)
    stamp = np.full(arrival.shape, -1, dtype=np.int64)
    price = np.full(arrival.shape, np.nan)
    either = np.zeros(arrival.shape, dtype=bool)
    for sw, fallback in ((other, True), (own, False)):
        last = np.searchsorted(sw.ts, arrival, side="left") - 1
        ok = last >= 0
        at = np.maximum(last, 0)
        if len(sw.ts):
            position = np.where(ok, sw.position[at], position)
            stamp = np.where(ok, sw.ts[at], stamp)
            price = np.where(ok, sw.price[at], price)
            either = np.where(ok, fallback, either)
    if (position < 0).any():
        raise IdentityError("an exit with no reachable print before it: the day has no entry")
    return position, stamp, price, either


def _check_inputs(
    row_ns: np.ndarray, decisions: np.ndarray, sides: np.ndarray, hold_ns: int
) -> None:
    if len(row_ns) > 1 and np.any(np.diff(row_ns) <= 0):
        raise ValueError("row stamps must be strictly increasing")
    if len(decisions) != len(sides):
        raise ValueError("one side per decision")
    if len(decisions) > 1 and np.any(np.diff(decisions) < 0):
        raise ValueError("decisions must be in time order")
    if not np.isin(sides, (-1, 1)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    if hold_ns <= 0:
        raise ValueError("the hold must be positive")


def execute_print_day(
    prints: DayPrints,
    row_ns: np.ndarray,
    decision_ns: np.ndarray,
    side: np.ndarray,
    hold_ns: int,
    settings: PrintSettings,
) -> PrintDay:
    """Execute one day's decisions (row stamps of the print grid ``row_ns``, in time
    order) with sides +1 / -1."""
    rows = np.asarray(row_ns, dtype=np.int64)
    decisions = np.asarray(decision_ns, dtype=np.int64)
    sides = np.asarray(side, dtype=np.int64)
    _check_inputs(rows, decisions, sides, hold_ns)
    start = day_start_ns(prints.day)
    first_ok = start + clock_ns(settings.no_entry_before)
    last_ok = start + clock_ns(settings.no_entry_after)
    flatten, day_end = start + clock_ns(settings.flatten_at), start + NS_PER_DAY
    latency, clip, rule = settings.latency_ns, settings.clip, settings.rule
    sw = day_sweeps(prints, settings)
    counters = dict.fromkeys(PRINT_COUNTERS, 0.0)
    account = Account()
    free_at = np.iinfo(np.int64).min
    out: list[tuple[object, ...]] = []
    for t, s in zip(decisions.tolist(), sides.tolist(), strict=True):
        counters["decisions"] += 1
        if t < first_ok or t > last_ok:
            counters["entries_outside_hours"] += 1
            continue
        if t < free_at:
            counters["entries_busy"] += 1
            continue
        arrival = t + latency
        entry = fills.fill(sw[s], np.array([arrival]), until_ns=flatten,
                           decision_ns=np.array([t]), rule=rule)  # fmt: skip
        if not entry.found[0]:
            counters["entries_no_print"] += 1
            continue
        e_ns, e_price = int(entry.print_ns[0]), float(entry.price[0])
        h = int(np.searchsorted(rows, e_ns + hold_ns, side="left"))
        decided, reason = (int(rows[h]) if h < len(rows) else e_ns + hold_ns), "hold"
        if decided > flatten:
            decided, reason = flatten, "day_end"
        x_arrival = decided + latency
        exit_ = fills.fill(sw[-s], np.array([x_arrival]), until_ns=day_end,
                           decision_ns=np.array([decided]), rule=rule)  # fmt: skip
        no_print = not bool(exit_.found[0])
        if no_print:
            pos, stamp, price, either = no_print_exits(sw[-s], sw[s], np.array([x_arrival]))
            x_pos, x_ns, x_price, x_beyond = int(pos[0]), int(stamp[0]), float(price[0]), 0.0
            x_fill = x_arrival
            counters["exits_no_print"] += 1
            counters["exits_no_print_either_side"] += float(either[0])
        else:
            x_pos, x_ns = int(exit_.position[0]), int(exit_.print_ns[0])
            x_price, x_beyond, x_fill = float(exit_.price[0]), float(exit_.beyond[0]), x_ns
        fee_e, fee_x = (float(v) for v in settings.fee_bp(np.array([e_ns, x_fill])))
        paid_e = account.fill(s, e_price, clip, fee_e, maker=False)
        account.check_identity(e_price)
        paid_x = account.fill(-s, x_price, clip, fee_x, maker=False)
        account.check_identity(x_price)
        check_fill_fees(fill_fee(np.array([e_price * clip, x_price * clip]), np.array([fee_e, fee_x])),
                        np.array([paid_e, paid_x]))  # fmt: skip
        e_beyond = float(entry.beyond[0])
        counters["fill_beyond_sweep"] += (e_beyond > BEYOND_SHARE * clip) + (
            x_beyond > BEYOND_SHARE * clip
        )
        e_wait, x_wait = e_ns - arrival, x_ns - x_arrival
        e_late, x_late = e_wait > settings.late_print_ns, x_wait > settings.late_print_ns
        counters["entry_late_print"] += e_late
        counters["exit_late_print"] += x_late
        counters["entries"] += 1
        free_at = x_fill
        fees = paid_e + paid_x
        gross = s * (x_price - e_price) * clip
        net = gross - fees
        notional = e_price * clip
        out.append((t, s, clip, arrival, e_ns, int(entry.position[0]), e_price, e_beyond, fee_e,
                    e_wait, e_late, decided, x_arrival, x_ns, x_pos, x_fill, x_price, x_beyond,
                    fee_x, x_wait, x_late, reason, no_print, x_fill - e_ns, notional,
                    x_price * clip, gross, fees, net, gross / notional * 1e4,
                    net / notional * 1e4))  # fmt: skip
    return _finish(account, out, counters)


def _finish(
    account: Account, out: list[tuple[object, ...]], counters: dict[str, float]
) -> PrintDay:
    trades = pd.DataFrame(out, columns=list(PRINT_TRADE_COLUMNS))
    if account.position != 0.0:
        raise IdentityError(f"the day ended holding {account.position!r}")
    turnover = float((trades["entry_price"] + trades["exit_price"]).mul(trades["size"]).sum())
    scale = max(1.0, account.taker_turnover)
    if abs(turnover - account.taker_turnover) > 1e-9 * scale:
        raise IdentityError(
            f"turnover {account.taker_turnover!r} differs from the fills' notionals {turnover!r}"
        )
    if abs(float(trades["net"].sum()) - account.cash) > 1e-9 * scale:
        raise IdentityError(f"the trades' net differs from cash {account.cash!r}")
    return PrintDay(trades, counters, account.cash, account.taker_turnover)
