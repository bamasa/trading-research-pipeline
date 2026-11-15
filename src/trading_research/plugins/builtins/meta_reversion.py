"""The reversion rule with a model deciding which of its trades to take.

Meta-labelling: :class:`~trading_research.plugins.builtins.strategies.Reversion`
picks the side, and a model (any registered classifier: a booster, a forest, a
transformer, an LSTM, a GRU, the TCN) is asked whether that trade will clear its
cost. On the history it is fitted on, every row where the rule would trade is a
candidate; its label is whether a trade entered there, held the rule's ten
minutes and charged the guarded rule's paper costs (half the touch's spread on
each leg, Bybit's base taker fee and 0.5 bp of slippage a side), made money.

The model reads only what is known at the row, all of it about the state the
rule depends on (``docs/findings.md``: the rule pays while the index's
ten-minute moves reverse and loses when they continue):

- ``strength``: the index move over the threshold that sets the trade rate;
- ``index_vol``: the spread of the index's ten-minute moves over the last hour;
- ``index_autocorr``: the correlation of each ten-minute move with the one
  before it, over the last six hours: the reversal itself, measured as it goes;
- ``trend``: the index's move over the last hour, signed by the trade;
- ``own_gap``: the traded instrument's own ten-minute move against the index's,
  signed by the trade: how far it has run past its peers;
- ``spread_bp``: the touch's spread, the cost the trade pays.

The trade is taken where the model's probability of a profit is in the top
``keep`` of its probabilities on the history's candidates. ``keep`` is fixed at
one half, not chosen on any result. A sequence model reads its window over the
consecutive candidates, the run of signals before this one.

It lives in its own module so the frozen rule's source, which the benchmark's
forward track pins by its sha256, is not touched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.plugins.builtins.guarded import SLIPPAGE_BP, TAKER_BP
from trading_research.plugins.builtins.strategies import Reversion, _log_panel
from trading_research.plugins.registry import register_strategy

#: Fewer labelled candidates than this and no model is fitted: the rule trades unfiltered.
MIN_CANDIDATES = 100
FEATURES = ("strength", "index_vol", "index_autocorr", "trend", "own_gap", "spread_bp")
#: What each named variant puts in the model's place; small windows for the
#: networks, whose sequence is the run of candidates rather than of rows.
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "reversion_meta": ("xgboost", {}),
    "reversion_lightgbm": ("lightgbm", {}),
    "reversion_logistic": ("logistic", {}),
    "reversion_transformer": ("transformer", {"window": 16, "channels": 32}),
    "reversion_lstm": ("lstm", {"window": 16, "channels": 32}),
    "reversion_gru": ("gru", {"window": 16, "channels": 32}),
    "reversion_tcn": ("tcn", {"window": 16, "channels": 16, "levels": 3}),
}


def _rolling(values: np.ndarray, window: int, fn: str) -> np.ndarray:
    return getattr(pd.Series(values).rolling(window, min_periods=window // 2), fn)().to_numpy()


def state_features(frame: pd.DataFrame, raw_signal: np.ndarray, threshold: float, lookback: int,
                   side: np.ndarray) -> pd.DataFrame:  # fmt: skip
    """The six inputs at every row, each from rows at or before it."""
    panel = _log_panel(frame).to_numpy()
    move = raw_signal  # the index's move over the look-back, bp
    lagged = (
        np.r_[np.full(lookback, np.nan), move[:-lookback]]
        if lookback < len(move)
        else move * np.nan
    )
    hour, six_hours = 6 * lookback, 36 * lookback
    product = pd.Series(move * lagged)
    mean_a = _rolling(move, six_hours, "mean")
    mean_b = _rolling(lagged, six_hours, "mean")
    cov = product.rolling(six_hours, min_periods=six_hours // 2).mean().to_numpy() - mean_a * mean_b
    sd_a, sd_b = _rolling(move, six_hours, "std"), _rolling(lagged, six_hours, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        autocorr = cov / (sd_a * sd_b)
    level = np.nanmean(panel[:, 1:], axis=1) if panel.shape[1] > 1 else panel[:, 0]
    trend = np.full(len(level), np.nan)
    trend[hour:] = (level[hour:] - level[:-hour]) * 1e4
    own = np.full(len(level), np.nan)
    own[lookback:] = (panel[lookback:, 0] - panel[:-lookback, 0]) * 1e4
    spread = (
        frame["spread_bp"].to_numpy(dtype=np.float64)
        if "spread_bp" in frame
        else np.zeros(len(frame))
    )
    return pd.DataFrame({
        "strength": np.abs(move) / threshold if threshold > 0 else np.abs(move),
        "index_vol": _rolling(move, hour, "std"),
        "index_autocorr": autocorr,
        "trend": side * trend,
        "own_gap": side * (own - move),
        "spread_bp": spread,
    })  # fmt: skip


def paper_net(frame: pd.DataFrame, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """A long's move over ``hold`` rows from each row and the round trip's cost, bp; NaN at the end."""
    mid = frame["mid"].to_numpy(dtype=np.float64)
    spread = (
        frame["spread_bp"].to_numpy(dtype=np.float64)
        if "spread_bp" in frame
        else np.zeros(len(mid))
    )
    move = np.full(len(mid), np.nan)
    cost = np.full(len(mid), np.nan)
    if hold < len(mid):
        move[:-hold] = (mid[hold:] / mid[:-hold] - 1.0) * 1e4
        cost[:-hold] = (spread[:-hold] + spread[hold:]) / 2 + 2 * (TAKER_BP + SLIPPAGE_BP)
    return move, cost


@register_strategy("reversion_meta")
class ReversionMeta(Reversion):
    """The reversion rule's side, a model's decision whether to take the trade (XGBoost by default)."""

    inputs = ("mid", "peer_*_mid", "spread_bp")  # type: ignore[assignment]

    def __init__(self, *, model: str = "xgboost", keep: float = 0.5, trades_per_day: float = 60.0,
                 **model_params: Any) -> None:  # fmt: skip
        super().__init__(trades_per_day=trades_per_day)
        self.model_name, self.keep, self.model_params = model, keep, model_params
        self.lookback = 36 * self.config.lookback  # the six-hour autocorrelation
        self.model: Any = None
        self.cut = float("nan")

    def _candidates(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        raw = self._signal(frame)
        side = np.asarray(super().decide(frame), dtype=np.int64)
        x = state_features(frame, raw, self.threshold, self.config.lookback, side)
        rows = np.flatnonzero((side != 0) & np.isfinite(x.to_numpy()).all(axis=1))
        return rows, x.iloc[rows].reset_index(drop=True)

    def _p_profit(self, x: pd.DataFrame) -> np.ndarray:
        from trading_research.models.base import CLASSES

        return np.asarray(self.model.predict_proba(x), dtype=np.float64)[:, CLASSES.index(1)]

    def fit(self, history: pd.DataFrame) -> ReversionMeta:
        from trading_research.pipeline.stages import build_model

        super().fit(history)
        rows, x = self._candidates(history)
        side = np.asarray(super().decide(history), dtype=np.int64)[rows]
        move, cost = paper_net(history, int(self.horizon))
        net = side * move[rows] - cost[rows]
        known = np.isfinite(net)
        x, net = x[known].reset_index(drop=True), net[known]
        y = pd.Series(np.where(net > 0, 1, 0))
        self.fitted_rows_, self.fitted_profitable_ = len(y), float(y.mean()) if len(y) else 0.0
        if len(y) < MIN_CANDIDATES or y.nunique() < 2:
            # too short a history to learn from: the rule's own trades, unfiltered
            self.model = None
            return self
        self.model = build_model(self.model_name, **self.model_params)
        self.model.fit(x, y)
        self.cut = float(np.quantile(self._p_profit(x), 1.0 - self.keep))
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        side = np.asarray(super().decide(frame), dtype=np.int64)
        if self.model is None:
            return side.astype(np.int8)
        out = np.zeros(len(frame), dtype=np.int8)
        rows, x = self._candidates(frame)
        if len(rows):
            take = self._p_profit(x) >= self.cut
            out[rows[take]] = side[rows[take]]
        return out


def _variant(name: str, model: str, params: dict[str, Any]) -> None:
    def __init__(self: ReversionMeta, **kwargs: Any) -> None:
        ReversionMeta.__init__(self, **{"model": model, **params, **kwargs})

    cls = type(name.title().replace("_", ""), (ReversionMeta,), {
        "__init__": __init__,
        "__doc__": f"The reversion rule's side, ``{model}`` deciding whether to take the trade.",
        "__module__": __name__,
    })  # fmt: skip
    register_strategy(name)(cls)


for _name, (_model, _params) in VARIANTS.items():
    if _name != "reversion_meta":
        _variant(_name, _model, _params)


@register_strategy("reversion_meta_guarded")
class ReversionMetaGuarded(ReversionMeta):
    """XGBoost choosing the reversion rule's trades, and the kill switch stopping them.

    The model's decisions are kept as a paper account from the end of the fit,
    as :class:`~trading_research.plugins.builtins.guarded.ReversionGuarded`
    keeps the rule's; when Page's CUSUM on its hourly results fires, nothing is
    opened from the hour after. The model picks the trades worth taking while
    the state holds; the switch stops it when the state goes.
    """

    live_from: pd.Timestamp | None = None
    stopped_at: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> ReversionMetaGuarded:
        super().fit(history)
        self.live_from = pd.Timestamp(pd.to_datetime(history["timestamp"], utc=True).iloc[-1])
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.plugins.builtins.guarded import paper_hourly
        from trading_research.validation.kill_switch import kill_switch

        raw = np.asarray(super().decide(frame), dtype=np.int64).copy()
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        live = stamps > self.live_from if self.live_from is not None else np.ones(len(raw), bool)
        if not live.any():
            return raw.astype(np.int8)
        watched = frame[live]
        closed, opened = paper_hourly(raw[live], watched, int(self.horizon), int(self.cooldown))
        if len(closed) <= 72:
            return raw.astype(np.int8)
        stop = kill_switch(closed, warm_up=72, booked=opened).first_stop
        self.stopped_at = None if stop is None else closed.index[stop.position]
        if self.stopped_at is not None:
            raw[stamps > self.stopped_at] = 0
        return raw.astype(np.int8)
