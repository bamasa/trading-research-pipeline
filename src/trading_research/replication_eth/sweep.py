"""The gate configurations the original ran on its two test days, and how it held them.

Read from the author's 2025 sweep notebook, one entry per distinct run: the
gate's parameters that change its signals (``exp_move_bp``, ``depth_min`` and
the sizing parameters cannot: the strategy inputs are never missing, the depth
input is a constant one, and the backtests read signals only) and the strategy
the backtest used. Runs that differ only in those inert parameters (cells 123
and 127 against 121, 187 against 183, 204 against 203) are one entry here.

``close_by_time`` closed after ``close_time`` 100 or 500; the unit is not in the
original's code and is read as seconds, which matches its trade counts (about
two fills per hold, plus the wait for the next signal). The entries marked
``added`` were not in the original: the best gate held 100 s, the strategy the
brief named, and held 10 s, the other reading of the unit; and the best gate
with the original's event code re-enacted, its unstable sort included (see
:mod:`.gate`).
"""

from __future__ import annotations

from typing import NamedTuple

from trading_research.replication_eth.gate import GateConfig

SIGNAL_TO_SIGNAL = "signal_to_signal"


def close_after(seconds: float) -> str:
    return f"close_by_time_{seconds:g}s"


class Run(NamedTuple):
    gate: GateConfig
    strategy: str
    source: str


def _g(
    name: str, tau: float, margin: float, buffer: float, t_min: float, gap: float, cool: float
) -> GateConfig:
    return GateConfig(name, tau, margin, buffer, t_min, gap, cool)


REF37 = _g("ref37", 0.9, 0.40, 3, 2.5, 15, 60)
REF48 = _g("ref48", 0.9, 0.45, 4, 1.6, 15, 10)
REF48_UNSTABLE = GateConfig("ref48_unstable_sort", 0.9, 0.45, 4, 1.6, 15, 10, "unstable_sort")
C150 = _g("c150", 0.9, 0.40, 3, 2.5, 15, 1)
C189 = _g("c189", 0.9, 0.40, 3, 2.5, 15, 0)
C195 = _g("c195", 0.9, 0.45, 4, 1.6, 15, 0)

S2S = SIGNAL_TO_SIGNAL
SWEEP: tuple[Run, ...] = (
    Run(REF37, S2S, "REF cell 37 / sweep cell 108"),
    Run(_g("c121", 0.8, 0.40, 3, 2.5, 15, 60), S2S, "sweep cells 121, 123, 127"),
    Run(_g("c122", 0.8, 0.40, 3, 1.0, 15, 60), S2S, "sweep cell 122"),
    Run(_g("c124", 0.8, 0.40, 3, 2.5, 20, 60), S2S, "sweep cell 124"),
    Run(_g("c125", 0.8, 0.40, 3, 2.5, 15, 30), S2S, "sweep cell 125"),
    Run(_g("c126", 0.8, 0.40, 3, 2.5, 15, 10), S2S, "sweep cell 126"),
    Run(_g("c133", 0.9, 0.40, 3, 2.5, 15, 10), S2S, "sweep cell 133"),
    Run(_g("c138", 0.9, 0.40, 3, 2.5, 15, 2), S2S, "sweep cell 138"),
    Run(_g("c149", 0.9, 0.30, 3, 2.5, 15, 10), S2S, "sweep cell 149"),
    Run(C150, S2S, "sweep cell 150"),
    Run(_g("c155", 0.8, 0.30, 3, 2.5, 15, 10), S2S, "sweep cell 155"),
    Run(_g("c156", 0.8, 0.40, 3, 2.5, 15, 1), S2S, "sweep cell 156"),
    Run(_g("c164", 0.85, 0.45, 3, 2.5, 15, 0.02), S2S, "sweep cell 164"),
    Run(C150, close_after(100), "sweep cell 170"),
    Run(_g("c176", 0.9, 0.30, 3, 2.5, 15, 1), S2S, "sweep cell 176"),
    Run(_g("c183", 0.95, 0.40, 3, 2.5, 15, 1), S2S, "sweep cells 183, 187"),
    Run(C189, close_after(100), "sweep cell 189"),
    Run(C189, S2S, "sweep cell 190"),
    Run(C195, close_after(100), "sweep cell 195"),
    Run(C195, S2S, "sweep cell 196"),
    Run(REF48, S2S, "REF cell 48 / sweep cells 203, 204 (the +0.9 run)"),
    Run(_g("c205", 0.87, 0.30, 4, 1.6, 15, 10), S2S, "sweep cell 205"),
    Run(_g("c206", 0.9, 0.30, 1, 1.0, 8, 10), S2S, "sweep cell 206"),
    Run(REF48, close_after(500), "sweep cell 208"),
    Run(_g("c218", 0.9, 0.40, 4, 1.3, 15, 10), S2S, "sweep cell 218"),
    Run(_g("c223", 0.9, 0.45, 4, 1.6, 15, 1), S2S, "sweep cell 223"),
    Run(REF48, close_after(100), "added: the best gate held 100 s"),
    Run(REF48, close_after(10), "added: the best gate held 10 s"),
    Run(REF48_UNSTABLE, S2S, "added: the best gate with the original's unstable event sort"),
)

#: Part 2's registered runs: the best gate as the original held it, then held 100 s.
FROZEN: tuple[Run, ...] = (
    Run(REF48, S2S, "primary"),
    Run(REF48, close_after(100), "secondary"),
)
