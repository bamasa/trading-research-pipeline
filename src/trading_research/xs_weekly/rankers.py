"""The eight rankers, walked forward Monday by Monday.

The rules read one feature. The five models are ``xs_deep``'s, imported
unchanged (``xs_deep.rankers.make``), whose settings this study registers:
ridge, LightGBM, the MLP, the TCN and the cross-sectional transformer. They
are refitted every :data:`REFIT_EVERY` Mondays on every day whose week ended
before the Monday, each day's universe of up to a hundred coins a row of the
batch, and score once there are :data:`MIN_TRAIN_DAYS` such days.

Run LightGBM and torch in separate processes (see ``xs_deep.rankers``).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from trading_research.xs_deep.features import Scaler, zscore_rows
from trading_research.xs_deep.rankers import Batch, make
from trading_research.xs_weekly.features import (
    HOLD,
    REL_7,
    REL_28,
    UNIVERSE,
    VOL_28,
    WINDOW,
    Frame,
    windows,
)

RULES = ("momentum", "reversal", "low_vol")
MODELS = ("ridge", "lightgbm", "mlp", "tcn", "xs_transformer")
NAMES = RULES + MODELS
REFIT_EVERY, MIN_TRAIN_DAYS = 4, 365


def coins(frame: Frame, day: int) -> np.ndarray:
    """The day's universe as up to UNIVERSE coin indices, padded with -1."""
    c = np.flatnonzero(frame.universe[day])[:UNIVERSE]
    return np.r_[c, np.full(UNIVERSE - len(c), -1)]


def gather(frame: Frame, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tab (B, N, 12), seq (B, N, W, 4), target (B, N), mask (B, N) for the days' universes."""
    idx = np.stack([coins(frame, d) for d in days])
    safe = np.where(idx >= 0, idx, 0)
    tab = frame.tab[days[:, None], safe]
    seq = np.stack([windows(frame, d, safe[i]) for i, d in enumerate(days)])
    y = frame.target[days[:, None], safe]
    mask = (idx >= 0) & np.isfinite(tab).all(axis=-1)
    return tab, seq.astype("float32"), y, mask


def training_days(frame: Frame, decision: int) -> np.ndarray:
    d = np.arange(WINDOW, decision - HOLD + 1)
    return d[frame.universe[d].sum(axis=1) > 0]


def rule(name: str, tab: np.ndarray) -> np.ndarray:
    if name == "momentum":
        return np.asarray(tab[..., REL_28])
    if name == "reversal":
        return np.asarray(-tab[..., REL_7])
    return np.asarray(-tab[..., VOL_28])


def walk(
    frame: Frame, name: str, mondays: np.ndarray
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(monday, coin indices, scores) for every Monday the ranker can score; NaN where unusable."""
    model, scalers = None, None
    for i, t in enumerate(mondays):
        tab, seq, _, mask = gather(frame, np.array([t]))
        if name in RULES:
            s = rule(name, tab)[0]
        else:
            if model is None or i % REFIT_EVERY == 0:
                days = training_days(frame, int(t))
                if len(days) < MIN_TRAIN_DAYS:
                    continue
                xt, xs, y, m = gather(frame, days)
                ok = m & np.isfinite(y)
                scalers = (
                    Scaler.fit(xt, ok),
                    Scaler.fit(xs[ok][:, -1, :], np.ones(int(ok.sum()), bool)),
                )
                model = make(name)
                model.fit(
                    Batch(
                        scalers[0].apply(xt, ok), scalers[1].apply(xs, ok), ok, zscore_rows(y, ok)
                    )
                )
            assert scalers is not None
            batch = Batch(scalers[0].apply(tab, mask), scalers[1].apply(seq, mask), mask)
            s = model.score(batch)[0]
        yield int(t), coins(frame, int(t)), np.where(mask[0], s, np.nan)
