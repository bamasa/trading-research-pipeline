"""The market state the reversion rule needs, measured on a track's scored span.

Section 27 found the ten-minute cross-sectional reversion pays when the
equal-weighted index's ten-minute return is followed by a ten-minute return
against it (autocorrelation about -0.10) and not when that autocorrelation is
zero. The forward track's registration states, before its days happen, what
the rule should do for each value; this is the number it is read against,
computed as :mod:`experiments.market_making_heldout` computes it for a block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.bench.tracks import Panel, TrackData
from trading_research.strategies.reversion import ReversionConfig, index_level


def index_autocorrelation(data: TrackData, lag: int | None = None) -> float:
    """Correlation of the index's ``lag``-row return with its next one, scored rows only.

    ``lag`` is the reversion rule's hold (ten minutes of five-second rows) by
    default. Every pair counts, overlapping; a pair whose forward return runs
    past the span's end is dropped. One recorded or forward span is one panel;
    a synthetic track's seeds are pooled.
    """
    lag = ReversionConfig().hold if lag is None else lag
    pasts, forwards = [], []
    for part in data.parts:
        if not isinstance(part, Panel):
            raise TypeError("the index needs a panel of instruments")
        log_mid = pd.DataFrame({s: np.log(m) for s, m in part.mids.items()}).sort_index().ffill()
        level = index_level(log_mid.to_numpy(dtype=np.float64))
        inside = level[np.asarray(log_mid.index >= part.start)]
        past = np.full(len(inside), np.nan)
        past[lag:] = inside[lag:] - inside[:-lag]
        forward = np.full(len(inside), np.nan)
        forward[:-lag] = inside[lag:] - inside[:-lag]
        pasts.append(past)
        forwards.append(forward)
    a, b = np.concatenate(pasts), np.concatenate(forwards)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
