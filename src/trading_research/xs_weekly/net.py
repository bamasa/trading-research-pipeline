"""The successor ``xs_weekly_net_2026_11``: the same rankers fitted on the return net of funding.

:func:`net` replaces a :class:`Frame`'s target with the week's log return minus
the funding a long pays, less the universe's mean; the book is unchanged, so a
leg still earns its price and funding and pays its fees. The rule ``funding``
ranks by minus the 28-day mean funding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from trading_research.xs_weekly import rankers
from trading_research.xs_weekly.features import TABULAR, Frame, _row_mean

NAMES = ("momentum", "low_vol", "funding", *rankers.MODELS)
FUNDING_28 = TABULAR.index("funding_28")


def net(frame: Frame) -> Frame:
    after = frame.gross - frame.funding_week / 1e4
    return replace(frame, target=after - _row_mean(after, frame.universe)[:, None])


def walk(
    frame: Frame, name: str, mondays: np.ndarray
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    f = net(frame)
    if name != "funding":
        yield from rankers.walk(f, name, mondays)
        return
    for t in mondays:
        tab, _, _, mask = rankers.gather(f, np.array([t]))
        yield int(t), rankers.coins(f, int(t)), np.where(mask[0], -tab[0, :, FUNDING_28], np.nan)
