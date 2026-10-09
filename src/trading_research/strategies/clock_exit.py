"""Close by time: the exit of every registered configuration, stated openly.

A position opened by a taker entry is closed by a taker order ``h`` after the
entry, whatever the price did; there is no take-profit and no stop. In detail,
on the replayed book's row stamps:

1. the entry is decided at a row stamped ``t`` and its order is live at
   ``t`` + 10 ms; it fills on the first snapshot at or after that, stamped
   ``s_e``;
2. the exit is decided at ``s_e + h``, or at the day's flatten time (23:59 UTC)
   if that comes first;
3. its order is live 10 ms later and fills on the first snapshot at or after
   that, stamped ``s_x``. The position is open until then, and the next entry
   may be decided from the first row after it.

No entry is decided before 00:01 or after 23:58 UTC. Every time is a row stamp;
nothing is counted in rows.

The walked-book label exits on the first snapshot at or after ``s_e + h``,
without the exit order's 10 ms, as its registration states. The two differ by
at most one snapshot; the label is the target, and the fills here are what is
scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.features.timed import at_or_after, seconds_ns
from trading_research.labels.taker_vwap import LATENCY_MS

NS_PER_DAY = 86_400 * 1_000_000_000


def _clock_ns(text: str) -> int:
    """``"HH:MM"`` as nanoseconds after midnight."""
    hours, minutes = (int(part) for part in text.split(":"))
    return (hours * 3600 + minutes * 60) * 1_000_000_000


@dataclass(frozen=True)
class DayLimits:
    """The registered day: entries between 00:01 and 23:58, flat at 23:59 UTC."""

    no_entry_before: str = "00:01"
    no_entry_after: str = "23:58"
    flatten_at: str = "23:59"

    def entry_allowed(self, stamps: np.ndarray) -> np.ndarray:
        """True for decision stamps inside the day's entry window."""
        clock = np.asarray(stamps, dtype=np.int64) % NS_PER_DAY
        return (clock >= _clock_ns(self.no_entry_before)) & (
            clock <= _clock_ns(self.no_entry_after)
        )

    def flatten_ns(self, stamps: np.ndarray) -> np.ndarray:
        """The flatten time of each stamp's own UTC day."""
        stamps = np.asarray(stamps, dtype=np.int64)
        return stamps - stamps % NS_PER_DAY + _clock_ns(self.flatten_at)


@dataclass(frozen=True)
class ClockExit:
    """Exit ``hold_s`` after the entry fill, or at the flatten time."""

    hold_s: float
    latency_ms: float = LATENCY_MS
    limits: DayLimits = field(default_factory=DayLimits)

    def __post_init__(self) -> None:
        if not self.hold_s > 0:
            raise ValueError(f"hold_s must be positive, got {self.hold_s}")

    @property
    def latency_ns(self) -> int:
        return round(self.latency_ms * 1_000_000)

    def exit_decisions(self, entry_fill_ns: np.ndarray) -> np.ndarray:
        """When each exit is decided: the fill plus the hold, capped at the flatten.

        A fill already past the flatten time (a feed gap across 23:59) is
        exited at once.
        """
        fill = np.asarray(entry_fill_ns, dtype=np.int64)
        capped = np.minimum(fill + seconds_ns(self.hold_s), self.limits.flatten_ns(fill))
        return np.maximum(capped, fill)

    def schedule(self, stamps: np.ndarray) -> pd.DataFrame:
        """For every row taken as an entry decision: its entry and exit snapshots.

        Indices are ``len(stamps)`` where no snapshot follows. ``exit_index``
        is what :func:`trading_research.strategies.ev_gate.thin_entries` takes
        as ``busy_until``; ``allowed`` marks the rows inside the day's entry
        window, and a caller zeroes the decisions outside it before thinning.
        """
        stamps = np.asarray(stamps, dtype=np.int64)
        n = len(stamps)
        entry = at_or_after(stamps, stamps + self.latency_ns)
        exit_ = np.full(n, n, dtype=np.int64)
        known = entry < n
        decided = self.exit_decisions(stamps[entry[known]])
        exit_[known] = at_or_after(stamps, decided + self.latency_ns)
        allowed = self.limits.entry_allowed(stamps)
        return pd.DataFrame({"entry_index": entry, "exit_index": exit_, "allowed": allowed})
