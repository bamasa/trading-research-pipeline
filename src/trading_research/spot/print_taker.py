"""The taker executor on prints: a clip filled by the side-correct print rule and filled back after the hold.

The print slots' executor (S25-PF on Binance BTC/FDUSD; S25-PB and S25-PFT as
measured rows), as registered under "Fills", "On prints, the side-correct
rule". Every fill comes from :func:`trading_research.spot.fills.round_trips`,
the function the print label calls, so a label, a walk-table entry and an
executed trade on the same decision cannot disagree. The rules, in the order
they apply to one decision at a row stamp ``t`` of the print grid:

* **Entry hours.** No entry decided before 00:01 or after 23:58 UTC
  (``entries_outside_hours``).
* **One position at a time.** A decision is taken only at a row stamped at or
  after the previous round trip's ``busy_until_ns`` (``entries_busy``
  otherwise): its exit's fill print, or the exit's arrival when that exit had
  no later print. On prints a buy and a sell fill at different prints, so the
  time a position keeps the account busy depends on its side.
* **Entry.** The order arrives at ``t`` + latency and fills at the first
  reachable print of its side stamped at or after its arrival and before
  23:59, sweeping that stamp's prints of its side. No entry is skipped for
  depth or for wait. An entry no such print follows is not made
  (``entries_no_print``).
* **Exit**, decided as the print label decides it: at the first row stamped at
  or after the entry fill's print + ``hold``, or at 23:59 if that comes first
  or no row follows. It arrives after the latency and fills by the same rule
  before the day ends, never on a print before the entry's. An exit no print
  of its side follows is priced, for the whole clip, at the last reachable
  print of its side at or before its arrival (``exits_no_print``, its age kept
  as a negative wait); if its side has none that day, at the last reachable
  print of either side (``exits_no_print_either_side``). Its fill time is then
  its arrival.
* **Counted, excluding nothing**: a fill whose print comes more than 1 s after
  its arrival (``entry_late_print``, ``exit_late_print``), and a fill with
  part of the clip beyond its sweep (``fill_beyond_sweep``, more than ``1e-9``
  of the clip).
* **Money.** Each fill pays the taker fee the fee table assigns its venue, tier
  and instant (the fill's print, or the arrival of an exit priced without
  one), checked against the table on every fill (K-units). The account's
  identity holds after every fill and at the day's end, with turnover equal to
  the sum of the fill notionals (K-acct). There is no funding.

The bracket (``bracket`` in the settings, one of
:data:`~trading_research.spot.fills.BRACKETS`) changes only the prices, never
which print fills or when, so every bracket keeps the true run's busy periods.
The loader's day (:class:`~trading_research.spot.loader.PrintsDay`, in the
print contract: ``aggressor`` +1 / -1 is the side, ``rpi`` marks the prints no
order reaches) becomes the rule's tape by :func:`print_tape`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.market_making.accounting import Account, IdentityError
from trading_research.search_engine.walk_tables import clock_ns
from trading_research.spot.costs import FeeTable, check_fill_fees, fill_fee, instrument_key
from trading_research.spot.fills import (
    BRACKETS,
    Fills,
    PrintFillSettings,
    PrintTape,
    RoundTrips,
    round_trips,
)
from trading_research.spot.loader import PrintsDay


def print_tape(day: PrintsDay) -> PrintTape:
    """The loader's day of prints as the print rule's tape; an excluded day is not executed."""
    if day.frame is None or not day.usable:
        reasons = ", ".join(day.exclusions) or "no frame"
        raise ValueError(f"{day.plane} {day.day} is excluded ({reasons}); it is not executed")
    return PrintTape.from_frame(day.frame, day.day)


@dataclass(frozen=True, kw_only=True)
class PrintSettings:
    """What the print executor depends on besides the prints: fees only from ``table``,
    the fill rule's settings (``rule``, the registered defaults) and the ``bracket``."""

    table: FeeTable
    instrument: str
    tier: str
    bracket: str = "rule"
    rule: PrintFillSettings = field(default_factory=PrintFillSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument", instrument_key(self.instrument))
        if self.tier not in self.table.tiers(self.instrument):
            raise ValueError(f"the fee table lists no tier {self.tier!r} for {self.instrument}")
        if self.bracket not in BRACKETS:
            raise ValueError(f"unknown bracket {self.bracket!r}; registered: {BRACKETS}")

    @property
    def clip(self) -> float:
        return self.rule.clip

    @property
    def latency_ns(self) -> int:
        return self.rule.latency_ns

    def entry_hours_ns(self, tape: PrintTape) -> tuple[int, int]:
        """The first and last stamps at which an entry may be decided on ``tape``'s day."""
        return (
            tape.start_ns + clock_ns(self.rule.no_entry_before),
            tape.start_ns + clock_ns(self.rule.no_entry_after),
        )

    def fee_bp(self, ns: np.ndarray) -> np.ndarray:
        """The taker fee the table assigns each fill instant."""
        return self.table.taker_bp(self.instrument, self.tier, ns)


def print_settings(
    table: FeeTable,
    instrument: str,
    tier: str | None = None,
    *,
    latency_ns: int = 10_000_000,
    bracket: str = "rule",
) -> PrintSettings:
    """The registered settings at ``tier`` (the venue's verdict tier by default);
    ``latency_ns=110_000_000`` is K-latency's re-run."""
    tier = tier or table.verdict_tier(instrument)
    return PrintSettings(table=table, instrument=instrument, tier=tier, bracket=bracket,
                         rule=PrintFillSettings(latency_ns=latency_ns))  # fmt: skip


#: The per-trade table's columns, in order. A wait is the fill's print stamp less
#: the order's arrival; an exit priced without a later print has a negative one,
#: minus the age of the print it was priced at, which ``exit_fallback_age_ns``
#: holds apart (-1 for an exit with a later print), so that the ages are reported
#: separately from the waits.
PRINT_TRADE_COLUMNS: tuple[str, ...] = (
    "decision_ns", "side", "size",
    "entry_arrival_ns", "entry_print_ns", "entry_position", "entry_price", "entry_beyond",
    "entry_fee_bp", "entry_wait_ns", "entry_late",
    "exit_decision_ns", "exit_arrival_ns", "exit_print_ns", "exit_position", "exit_fill_ns",
    "exit_price", "exit_beyond", "exit_fee_bp", "exit_wait_ns", "exit_late", "exit_reason",
    "exit_no_print", "exit_fallback_age_ns", "hold_realised_ns",
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


def day_round_trips(
    tape: PrintTape, row_ns: np.ndarray, decision_ns: np.ndarray, side: np.ndarray | int,
    hold_ns: int, settings: PrintSettings,
) -> RoundTrips:  # fmt: skip
    """Every decision's round trip by the print rule, as if it were alone: the
    executor's day (entries before 23:59, the exit flattened at 23:59 and falling
    back when no print follows)."""
    return round_trips(tape, row_ns, decision_ns, side, hold_ns, flatten=True,
                       settings=settings.rule)  # fmt: skip


def exits_either_side(tape: PrintTape, exit_: Fills) -> np.ndarray:
    """Exits priced at a print of the other side: their own side printed nothing reachable
    before them that day (``exits_no_print_either_side``)."""
    if not len(tape):
        return np.zeros(len(exit_.side), dtype=bool)
    at = np.maximum(exit_.print_index, 0)
    return np.asarray(exit_.fallback & (tape.side[at] != exit_.side))


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
    tape: PrintTape,
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
    first_ok, last_ok = settings.entry_hours_ns(tape)
    trips = day_round_trips(tape, rows, decisions, sides, hold_ns, settings)
    entry, exit_, clip = trips.entry, trips.exit, settings.clip
    entry_px = entry.price(settings.bracket).tolist()
    exit_px = exit_.price(settings.bracket).tolist()
    either = exits_either_side(tape, exit_)
    counters = dict.fromkeys(PRINT_COUNTERS, 0.0)
    account = Account()
    free_at = np.iinfo(np.int64).min
    out: list[tuple[object, ...]] = []
    for k, (t, s) in enumerate(zip(decisions.tolist(), sides.tolist(), strict=True)):
        counters["decisions"] += 1
        if t < first_ok or t > last_ok:
            counters["entries_outside_hours"] += 1
            continue
        if t < free_at:
            counters["entries_busy"] += 1
            continue
        if not entry.found[k]:
            counters["entries_no_print"] += 1
            continue
        no_print = bool(exit_.fallback[k])
        if not (no_print or exit_.found[k]):
            raise IdentityError("an exit with no reachable print before it: the day has no entry")
        e_ns, arrival, e_price = int(entry.print_ns[k]), int(entry.arrival_ns[k]), entry_px[k]
        x_ns, x_arrival, x_price = int(exit_.print_ns[k]), int(exit_.arrival_ns[k]), exit_px[k]
        x_fill = int(trips.busy_until_ns[k])
        fee_e, fee_x = (float(v) for v in settings.fee_bp(np.array([e_ns, x_fill])))
        paid_e = account.fill(s, e_price, clip, fee_e, maker=False)
        account.check_identity(e_price)
        paid_x = account.fill(-s, x_price, clip, fee_x, maker=False)
        account.check_identity(x_price)
        check_fill_fees(fill_fee(np.array([e_price * clip, x_price * clip]), np.array([fee_e, fee_x])),
                        np.array([paid_e, paid_x]))  # fmt: skip
        e_beyond, x_beyond = float(entry.beyond[k]), float(exit_.beyond[k])
        e_late, x_late = bool(entry.late[k]), bool(exit_.late[k])
        counters["exits_no_print"] += no_print
        counters["exits_no_print_either_side"] += bool(either[k])
        counters["fill_beyond_sweep"] += bool(entry.beyond_sweep[k]) + bool(exit_.beyond_sweep[k])
        counters["entry_late_print"] += e_late
        counters["exit_late_print"] += x_late
        counters["entries"] += 1
        free_at = x_fill
        fees = paid_e + paid_x
        gross = s * (x_price - e_price) * clip
        net = gross - fees
        notional = e_price * clip
        out.append((t, s, clip, arrival, e_ns, int(entry.print_index[k]), e_price, e_beyond, fee_e,
                    e_ns - arrival, e_late, int(trips.exit_decision_ns[k]), x_arrival, x_ns,
                    int(exit_.print_index[k]), x_fill, x_price, x_beyond, fee_x, x_ns - x_arrival,
                    x_late, str(trips.exit_reason[k]), no_print, int(exit_.age_ns[k]),
                    x_fill - e_ns, notional,
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
