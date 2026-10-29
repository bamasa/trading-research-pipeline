"""The decision rules of ``strategies/``, registered as strategy plugins.

Each adapter wires the existing functions to the frame contract of
:mod:`trading_research.plugins.protocols` and changes none of them. What the
adapter itself chooses (which instrument is the peer, which model feeds a rule,
a trade rate) is a default of the adapter, stated in its docstring, not a value
any registration froze.

``clock_exit`` and ``sizing`` are not registered: one is an exit schedule (every
execution here closes on the clock it is given, the strategy's ``horizon``),
the other a size multiplier applied to realised trades, and neither decides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.plugins.common import (
    book_features,
    days_spanned,
    forward_bp,
    predict_rows,
    rows_per_day,
)
from trading_research.plugins.registry import register_strategy


def _peers(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith("peer_") and c.endswith("_mid"))


def _log_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """The traded instrument's log mid first, then each peer's."""
    columns = {"self": np.log(frame["mid"].to_numpy(dtype="float64"))}
    for peer in _peers(frame):
        columns[peer] = np.log(frame[peer].to_numpy(dtype="float64"))
    return pd.DataFrame(columns, index=frame.index)


@register_strategy("reversion")
class Reversion:
    """Trade against the peers' index move, the frozen ten-minute reversion rule.

    :func:`~trading_research.strategies.reversion.signal` on the equal-weighted
    index of the peers (the traded instrument excluded, as
    :func:`~trading_research.strategies.reversion.index_level` requires), and
    :func:`~trading_research.strategies.reversion.decide` at the threshold that
    gives the configuration's trade rate on the history
    (:func:`~trading_research.strategies.reversion.threshold_for_rate`). Look-back
    and hold are :class:`~trading_research.strategies.reversion.ReversionConfig`'s,
    in rows of whatever grid the frame is on.
    """

    inputs = ("mid", "peer_*_mid")

    def __init__(self, *, trades_per_day: float = 60.0) -> None:
        from trading_research.strategies.reversion import ReversionConfig

        self.config = ReversionConfig(trades_per_day=trades_per_day)
        self.lookback = self.config.lookback
        self.horizon = self.config.hold
        self.threshold = float("nan")

    def _signal(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.strategies.reversion import index_level, signal

        level = index_level(_log_panel(frame).to_numpy(), exclude=0)
        return signal(level, self.config)

    def fit(self, history: pd.DataFrame) -> Reversion:
        from trading_research.strategies.reversion import threshold_for_rate

        self.threshold = threshold_for_rate(
            self._signal(history), rows_per_day(history), self.config
        )
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.strategies.reversion import decide

        return decide(self._signal(frame), self.threshold)


@register_strategy("relative")
class Relative:
    """Long the most depressed name and short the most extended: the traded leg's sign.

    :func:`~trading_research.strategies.relative.deviations` and
    :func:`~trading_research.strategies.relative.weights` over the traded
    instrument and its peers; the decision is the sign of the traded
    instrument's weight. A basket of one a side, since the frame carries four
    instruments and the rule needs twice the basket.
    """

    inputs = ("mid", "peer_*_mid")

    def __init__(self, *, lookback: int = 24, hold: int = 24, basket: int = 1) -> None:
        from trading_research.strategies.relative import CrossSection

        self.config = CrossSection(lookback=lookback, basket=basket, hold=hold)
        self.lookback = lookback
        self.horizon = hold

    def fit(self, history: pd.DataFrame) -> Relative:  # noqa: ARG002
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.strategies.relative import deviations, weights

        weight = weights(deviations(_log_panel(frame), self.config), self.config)["self"]
        return np.sign(weight.to_numpy()).astype(np.int8)


@register_strategy("spread")
class Spread:
    """The pair against its first peer: a hedge and a mean-reverting residual fitted on history.

    :func:`~trading_research.strategies.spread.fit_hedge` and
    :func:`~trading_research.strategies.spread.fit_ou` on the history only, then
    :func:`~trading_research.strategies.spread.z_score` and the band rule
    :func:`~trading_research.strategies.spread.signal`, which holds a position
    between the exit and entry bands: the decision carries state.
    """

    inputs = ("mid", "peer_1_mid")
    stateful = True

    def __init__(self, *, entry: float = 2.0, exit_at: float = 0.5, hold: int = 60) -> None:
        self.entry = entry
        self.exit_at = exit_at
        self.lookback = 0
        self.horizon = hold
        self.hedge: Any = None
        self.ou: Any = None

    def _legs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.log(frame["mid"].to_numpy(dtype="float64")),
            np.log(frame["peer_1_mid"].to_numpy(dtype="float64")),
        )

    def fit(self, history: pd.DataFrame) -> Spread:
        from trading_research.strategies.spread import fit_hedge, fit_ou

        a, b = self._legs(history)
        self.hedge = fit_hedge(a, b)
        self.ou = fit_ou(self.hedge.residual(a, b), step=12)
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.strategies.spread import signal, z_score

        a, b = self._legs(frame)
        z = z_score(self.hedge.residual(a, b), self.ou)
        return signal(z, entry=self.entry, exit_at=self.exit_at).astype(np.int8)


@register_strategy("hold_first")
class HoldFirst:
    """The hold-first rule on a model's class probabilities, thresholds set from a trade rate.

    The model (a registered model plugin, ``logistic`` by default) is fitted on
    the first ``fit_share`` of the history on every registered book feature,
    against a directional label whose threshold leaves ``label_share`` of the
    rows long or short. :func:`~trading_research.strategies.hold_first.fit_hold_first`
    sets each side's threshold on the rest of the history from ``trades_per_day``,
    reading probabilities only; :class:`~trading_research.strategies.hold_first.HoldFirstRule`
    decides.
    """

    inputs = ("bid_price_*", "bid_size_*", "ask_price_*", "ask_size_*")

    def __init__(
        self,
        *,
        model: str = "logistic",
        hold: int = 30,
        trades_per_day: float = 2000.0,
        fit_share: float = 0.7,
        label_share: float = 0.3,
    ) -> None:
        from trading_research.features import REGISTRY

        self.model_name = model
        self.horizon = hold
        self.trades_per_day = trades_per_day
        self.fit_share = fit_share
        self.label_share = label_share
        self.lookback = REGISTRY.max_lookback(REGISTRY.names("book"))
        self.model: Any = None
        self.rule: Any = None

    def fit(self, history: pd.DataFrame) -> HoldFirst:
        from trading_research.labels.directional import DirectionalLabel, threshold_for_share
        from trading_research.plugins.registry import resolve
        from trading_research.strategies.hold_first import fit_hold_first

        split = int(len(history) * self.fit_share)
        train, validation = history.iloc[: split - self.horizon], history.iloc[split:]
        x, _ = book_features(train)
        threshold = threshold_for_share(train, self.horizon, self.label_share)
        y = DirectionalLabel(self.horizon, threshold)(train)
        ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & y.notna().to_numpy()
        self.model = resolve("model", self.model_name)
        self.model.fit(x.loc[ok], y.loc[ok])
        proba = predict_rows(self.model, book_features(history)[0])[split:]
        usable = np.isfinite(proba).all(axis=1)
        self.rule = fit_hold_first(
            proba[usable], days=days_spanned(validation), trades_per_day=self.trades_per_day
        )
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        proba = predict_rows(self.model, book_features(frame)[0])
        out = np.zeros(len(frame), dtype=np.int8)
        usable = np.isfinite(proba).all(axis=1)
        out[usable] = self.rule.decide_proba(proba[usable])
        return out


@register_strategy("ev_gate")
class EVGateStrategy:
    """Enter where the predicted move's lower bound clears fees, the walk and a buffer.

    A heteroscedastic pair of ridge regressions
    (:class:`~trading_research.models.uncertainty.HeteroscedasticRegressor`) is
    fitted on the history's book features against the forward move in bp; the
    gate is :class:`~trading_research.strategies.ev_gate.EVGate` at the base
    tier's taker fee on both sides, with the walk cost of the clip on each row
    (:func:`~trading_research.backtest.walk.walk_cost_bp`). One entry per event
    (:func:`~trading_research.strategies.ev_gate.gate_events`): the decision is
    the first row of each run of open rows.
    """

    inputs = ("bid_price_*", "bid_size_*", "ask_price_*", "ask_size_*")

    def __init__(
        self, *, k: float = 1.0, beta_bp: float = 0.0, hold: int = 30, clip: float = 0.01
    ) -> None:
        from trading_research.backtest.costs import BYBIT_BASE
        from trading_research.features import REGISTRY
        from trading_research.strategies.ev_gate import EVGate

        self.gate = EVGate.at_tier(BYBIT_BASE, k=k, beta_bp=beta_bp)
        self.horizon = hold
        self.clip = clip
        # One more row than the features: an event starts where the row before was closed.
        self.lookback = REGISTRY.max_lookback(REGISTRY.names("book")) + 1
        self.model: Any = None

    def fit(self, history: pd.DataFrame) -> EVGateStrategy:
        from trading_research.models.regression import RidgeBaseline
        from trading_research.models.uncertainty import HeteroscedasticRegressor

        x, _ = book_features(history)
        y = pd.Series(forward_bp(history, self.horizon), index=history.index)
        ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(y.to_numpy())
        cost = self.gate.cost_bp
        self.model = HeteroscedasticRegressor(
            RidgeBaseline(scale_bp=cost), RidgeBaseline(scale_bp=cost)
        )
        self.model.fit(x.loc[ok], y.loc[ok])
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.backtest.walk import walk_cost_bp
        from trading_research.plugins.market import depth_of
        from trading_research.strategies.ev_gate import gate_events

        x, _ = book_features(frame)
        ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1)
        mu = np.full(len(frame), np.nan)
        sigma = np.full(len(frame), np.nan)
        if ok.any():
            mu[ok] = self.model.predict_mean_bp(x.loc[ok])
            sigma[ok] = self.model.predict_sigma_bp(x.loc[ok])
        walk = walk_cost_bp(frame, self.clip, depth_of(frame))
        sides = self.gate.sides(mu, np.nan_to_num(sigma, nan=0.0), np.nan_to_num(walk, nan=0.0))
        sides[~ok | ~np.isfinite(walk)] = 0
        events = gate_events(sides)
        first = (events >= 0) & (np.r_[-1, events[:-1]] != events)
        return np.where(first, sides, 0).astype(np.int8)
