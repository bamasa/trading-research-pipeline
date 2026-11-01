"""The 40 registered cells: an inventory rule and at most one gate, at the touch proxies.

Every cell quotes its bid at the bid proxy and its ask at the ask proxy, one
clip each, or does not quote that side. It never reads the fee.

* **Inventory rule** ``lambda``: the side that would add to |q| is withdrawn while
  ``lambda x (sigma / sigma_ref) x |q| / Q >= 1/2`` (``sigma / sigma_ref`` is 1 while ``sigma`` is
  undefined). ``lambda = 0`` is S0.
* **Gates**: ``spread`` open while the proxy spread is at least ``m`` ticks;
  ``vle`` open while ``sigma`` is at most W's ``k``-quantile; ``vge`` while it is at
  least W's ``k``-quantile. A closed gate withdraws the side that would add to
  |q| (both sides when flat); the reducing side stays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LAMBDAS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
GATES: dict[str, tuple[float, ...]] = {
    "spread": (2.0, 3.0, 5.0),
    "vle": (0.33, 0.67, 0.9),
    "vge": (0.33, 0.67, 0.9),
}


@dataclass(frozen=True, order=True)
class Cell:
    lam: float
    gate: str = "none"
    param: float = 0.0

    @property
    def key(self) -> str:
        lam = f"{self.lam:g}"
        return f"l{lam}|none" if self.gate == "none" else f"l{lam}|{self.gate}{self.param:g}"

    @property
    def gated(self) -> bool:
        return self.gate != "none"


S0 = Cell(0.0)


def grid() -> list[Cell]:
    """Every registered cell, ungated first, in a fixed order."""
    cells = [Cell(lam) for lam in LAMBDAS]
    for gate, params in GATES.items():
        cells += [Cell(lam, gate, p) for lam in LAMBDAS for p in params]
    return cells


def from_key(key: str) -> Cell:
    for cell in grid():
        if cell.key == key:
            return cell
    raise KeyError(key)


def neighbourhood(cell: Cell) -> list[Cell]:
    """Cells within one step on lambda and, for a gated cell, on its gate's parameter."""
    li = LAMBDAS.index(cell.lam)
    lams = LAMBDAS[max(0, li - 1) : li + 2]
    if not cell.gated:
        return [Cell(lam) for lam in lams]
    params = GATES[cell.gate]
    pi = params.index(cell.param)
    return [Cell(lam, cell.gate, p) for lam in lams for p in params[max(0, pi - 1) : pi + 2]]


@dataclass(frozen=True)
class State:
    """What a gate reads at a decision: the proxy spread (ticks) and sigma (bp)."""

    spread: float
    vol: float


def gate_open(cell: Cell, state: State, quantiles: dict[float, float]) -> bool:
    if cell.gate == "none":
        return True
    if cell.gate == "spread":
        return bool(state.spread >= cell.param)
    if math.isnan(state.vol):
        return False
    level = quantiles[cell.param]
    return bool(state.vol <= level) if cell.gate == "vle" else bool(state.vol >= level)


def wanted(
    cell: Cell,
    position: float,
    soft_limit: float,
    vol: float,
    sigma_ref: float,
    state: State,
    quantiles: dict[float, float],
) -> tuple[bool, bool]:
    """Whether to quote the bid and the ask at this decision."""
    bid = ask = True
    adding_bid = position >= 0.0  # a fill on the bid adds to |q| when flat or long
    adding_ask = position <= 0.0
    if cell.lam > 0.0 and position != 0.0:
        ratio = vol / sigma_ref if (vol == vol and sigma_ref > 0) else 1.0
        if cell.lam * ratio * abs(position) / soft_limit >= 0.5:
            bid, ask = (False, True) if position > 0 else (True, False)
    if not gate_open(cell, state, quantiles):
        bid = bid and not adding_bid
        ask = ask and not adding_ask
    return bid, ask
