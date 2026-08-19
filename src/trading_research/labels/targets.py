"""What the model is asked to predict.

Everything in this project until now used one target: the sign of the forward
return, where the move cleared the round-trip cost, and zero otherwise. That is
a *proxy* for profitability rather than profitability itself, and the gap
between the two is not decorative.

The proxy cannot tell a 12 bp move from a 60 bp one, though those are different
trades. It says nothing about the path between entry and exit, though the exit
rules act on that path. And it asks the model a question — "will it move more
than eleven basis points" — that nobody wants the answer to, in place of the
one they do: "is opening a position here worth it".

Two problems, not one
---------------------
The first is *what* is asked, and the four targets below answer it differently.
The second is *how the answer is measured*, and every one of them shared the
same defect until now: the future was a single price, ``m(t+H)``.

At these frequencies one future price is mostly noise. The mid oscillates
between bid and ask on every update, so ``m(t+H)`` is the true level plus half a
spread in a direction nobody can predict, and a model asked to fit that spends
its capacity on the oscillation. The standard fix in the order-book literature —
Ntakaris et al. on FI-2010, and Zhang et al. for DeepLOB — is to compare
*averages* rather than points:

    m_plus(t)  = mean of the next k mids
    m_minus(t) = mean of the last k mids
    label = (m_plus - m_minus) / m_minus

Averaging both ends removes the bounce and leaves the drift. It is the same
horizon and a far less noisy measurement of it.

The second fix is normalisation. A 10 bp move means something different in a
calm hour and a violent one, and a target in raw basis points asks the model to
learn the regime as well as the direction. Dividing by a trailing volatility
makes the target stationary — and it makes the trade decision better, not just
the fit: a prediction in units of local volatility converts back to basis points
by multiplying by the volatility prevailing *now*, so the comparison with cost
adapts to the regime instead of being fixed.

``smoothing`` and ``normalise`` are therefore parameters of every target rather
than separate targets of their own.

Four targets, from the proxy outwards.

``direction``
    The original. Three classes, thresholded at the round-trip cost.

``magnitude``
    The forward return itself, as a regression. Keeps the size of the move,
    which the three-class version throws away, and makes the trade decision a
    comparison of expected value against cost rather than a threshold on a
    probability that has to be swept.

``net_pnl``
    What a long opened here actually nets: the forward return minus the round
    trip. The most direct target available — a model fitted on it optimises the
    number the backtest reports, with no proxy in between. A short's result
    follows from the same prediction: if ``y`` is the long's net, the short's is
    ``-y - 2c``, so shorting pays when ``y < -2c``.

``triple_barrier``
    Which of three barriers the price touches first — take-profit, stop-loss,
    or the clock. This is the only target that knows about the path, and it
    encodes the trading rule into the label itself, so the prediction is about
    the outcome of a trade rather than about the movement of a price.

Every target declares its horizon, and that number does real work: the split
purges each training block's tail by it. Deriving the purge from the target
rather than from a config field is what stops the two drifting apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

#: What a target produces. Classification targets are three-class (-1, 0, +1);
#: regression targets are a single number in basis points.
TargetKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class TargetSpec:
    """One way of asking the question, and what it needs to be answered."""

    name: str
    kind: TargetKind
    #: Rows the label reads forward. Used to purge the training block's tail.
    horizon: int
    description: str
    #: True when the label needs the price path rather than only its endpoint.
    needs_path: bool = False


def smoothed_move_bp(
    mid: pd.Series,
    horizon: int,
    *,
    smoothing: int = 1,
) -> pd.Series:
    """Forward move in basis points, measured between averages rather than points.

    ``smoothing`` is *k* in the FI-2010 construction: the mean of the next *k*
    mids against the mean of the last *k*. One reduces to the single-point
    version this project used everywhere before, so the change is opt-in and the
    old numbers remain reproducible.

    The window is centred on the horizon — the forward average is taken over the
    *k* observations ending at ``t + H`` — so the label still reads exactly *H*
    rows ahead and the purge derived from the horizon is still correct. Taking
    the average of the *k* rows *after* ``t + H`` would read further than
    declared, which is the quiet way a smoothed label becomes a leak.
    """
    values = mid.to_numpy(dtype="float64")
    if smoothing <= 1:
        forward = np.full(len(values), np.nan)
        forward[:-horizon] = values[horizon:]
        backward = values
    else:
        rolled = pd.Series(values).rolling(smoothing, min_periods=1).mean().to_numpy()
        forward = np.full(len(values), np.nan)
        forward[:-horizon] = rolled[horizon:]
        backward = rolled
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.log(forward / backward) * 1e4, index=mid.index)


def _move(frame: pd.DataFrame, horizon: int, smoothing: int, normalise: int) -> pd.Series:
    """The move a target is built on, smoothed and normalised as asked."""
    if smoothing > 1 or normalise > 0:
        if "mid" not in frame.columns:
            raise KeyError("smoothing and normalisation need the mid price path")
        move = smoothed_move_bp(frame["mid"], horizon, smoothing=smoothing)
    else:
        move = frame["forward_bp"]

    if normalise > 0:
        if "mid" not in frame.columns:
            raise KeyError("normalisation needs the mid price path")
        # Trailing volatility only: a window that included the future would
        # scale the label by information from after the decision.
        ratio = frame["mid"] / frame["mid"].shift(1)
        returns = pd.Series(np.log(ratio.to_numpy(dtype="float64")), index=frame.index) * 1e4
        sigma = returns.rolling(normalise, min_periods=normalise // 2).std()
        move = move / sigma.where(sigma > 0)
    return move


def direction(
    frame: pd.DataFrame,
    *,
    cost_bp: float,
    horizon: int = 24,
    smoothing: int = 1,
    normalise: int = 0,
    **_: Any,
) -> pd.Series:
    """Sign of the move where it clears the cost, zero otherwise.

    With ``normalise`` the threshold is in units of local volatility rather than
    basis points, so ``cost_bp`` is interpreted as a number of standard
    deviations. The caller converts.
    """
    move = _move(frame, horizon, smoothing, normalise)
    return pd.Series(
        np.where(move.isna(), np.nan, np.sign(move) * (move.abs() > cost_bp)),
        index=frame.index,
    )


def magnitude(
    frame: pd.DataFrame,
    *,
    horizon: int = 24,
    smoothing: int = 1,
    normalise: int = 0,
    **_: Any,
) -> pd.Series:
    """The forward return itself, in basis points or in local volatilities.

    Signed, so it carries direction and size in one number. A model fitted on
    this can be asked for an expected value, which is what the trade decision
    actually needs.

    With ``normalise`` the units become standard deviations of the recent
    return distribution. Converting a prediction back to basis points means
    multiplying by the volatility prevailing at the moment of the decision,
    which makes the comparison against cost regime-aware rather than fixed.
    """
    return _move(frame, horizon, smoothing, normalise)


def net_pnl(frame: pd.DataFrame, *, cost_bp: float, **_: Any) -> pd.Series:
    """What a long opened here nets after the round trip.

    Shifting by the cost rather than dividing by it keeps the units — basis
    points of profit — so a prediction of 3 means three basis points of profit
    and the decision rule is "positive is worth taking".
    """
    return frame["forward_bp"] - cost_bp


def smoothed_direction(
    frame: pd.DataFrame,
    *,
    cost_bp: float,
    horizon: int = 24,
    smoothing: int = 20,
    normalise: int = 0,
    **_: Any,
) -> pd.Series:
    """``direction`` with the FI-2010 smoothing switched on by default."""
    return direction(
        frame, cost_bp=cost_bp, horizon=horizon, smoothing=smoothing, normalise=normalise
    )


def normalised_magnitude(
    frame: pd.DataFrame,
    *,
    horizon: int = 24,
    smoothing: int = 20,
    normalise: int = 200,
    **_: Any,
) -> pd.Series:
    """``magnitude``, smoothed and expressed in local volatilities.

    The form most of the order-book literature actually fits: a stationary
    target that does not ask the model to learn the volatility regime as well
    as the direction.
    """
    return magnitude(frame, horizon=horizon, smoothing=smoothing, normalise=normalise)


def triple_barrier(
    frame: pd.DataFrame,
    *,
    cost_bp: float,
    horizon: int,
    take_profit_bp: float | None = None,
    stop_loss_bp: float | None = None,
    **_: Any,
) -> pd.Series:
    """Which barrier the price touches first: +1 up, -1 down, 0 the clock.

    The one target here that reads the path rather than only the endpoint, and
    the only one that knows the trading rule. A move that ends flat after
    touching the take-profit is a winning trade under a take-profit rule and a
    neutral one under a fixed clock; the endpoint labels cannot tell those
    apart and this one can.

    Barriers default to the round-trip cost, which makes "up" mean "would have
    paid for itself" rather than merely "rose". Asymmetric barriers are allowed
    and are how a stop tighter than the target gets encoded.

    Implemented as a forward scan over the mid path. The horizon caps how far
    it looks, so the label reads exactly ``horizon`` rows ahead and no further
    — which is what the purge assumes.
    """
    if "mid" not in frame.columns:
        raise KeyError("triple_barrier needs the mid price path; prepare carries it as 'mid'")

    upper = take_profit_bp if take_profit_bp is not None else cost_bp
    lower = stop_loss_bp if stop_loss_bp is not None else cost_bp
    mid = frame["mid"].to_numpy(dtype="float64")
    n = len(mid)
    out = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = mid[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        window = mid[i + 1 : i + 1 + horizon]
        move = (window / entry - 1.0) * 1e4
        up = np.flatnonzero(move >= upper)
        down = np.flatnonzero(move <= -lower)
        first_up = up[0] if up.size else n
        first_down = down[0] if down.size else n
        if first_up == n and first_down == n:
            out[i] = 0.0
        else:
            # Ties resolve upward. Bar data cannot say which came first inside
            # one observation, and this is the optimistic reading — stated
            # rather than hidden, and it flatters the label very slightly.
            out[i] = 1.0 if first_up <= first_down else -1.0
    return pd.Series(out, index=frame.index)


#: Every target, addressable by name so a config can select one as a string.
TARGETS: dict[str, tuple[TargetSpec, Callable[..., pd.Series]]] = {
    "direction": (
        TargetSpec(
            name="direction",
            kind="classification",
            horizon=24,
            description="sign of the move where it clears the cost",
        ),
        direction,
    ),
    "magnitude": (
        TargetSpec(
            name="magnitude",
            kind="regression",
            horizon=24,
            description="the forward return in basis points",
        ),
        magnitude,
    ),
    "net_pnl": (
        TargetSpec(
            name="net_pnl",
            kind="regression",
            horizon=24,
            description="what a long nets after the round trip",
        ),
        net_pnl,
    ),
    "smoothed_direction": (
        TargetSpec(
            name="smoothed_direction",
            kind="classification",
            horizon=24,
            description="direction between smoothed prices, FI-2010 style",
            needs_path=True,
        ),
        smoothed_direction,
    ),
    "normalised_magnitude": (
        TargetSpec(
            name="normalised_magnitude",
            kind="regression",
            horizon=24,
            description="smoothed move in units of trailing volatility",
            needs_path=True,
        ),
        normalised_magnitude,
    ),
    "triple_barrier": (
        TargetSpec(
            name="triple_barrier",
            kind="classification",
            horizon=24,
            description="which barrier is touched first: take-profit, stop, or the clock",
            needs_path=True,
        ),
        triple_barrier,
    ),
}


def build_target(
    name: str,
    frame: pd.DataFrame,
    *,
    cost_bp: float,
    horizon: int = 24,
    **params: Any,
) -> tuple[pd.Series, TargetSpec]:
    """Compute a target by name, and return it with what it declares."""
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {', '.join(sorted(TARGETS))}")
    spec, function = TARGETS[name]
    values = function(frame, cost_bp=cost_bp, horizon=horizon, **params)
    return values, TargetSpec(
        name=spec.name,
        kind=spec.kind,
        horizon=horizon,
        description=spec.description,
        needs_path=spec.needs_path,
    )


def describe_targets() -> pd.DataFrame:
    """Tabulate the menu, for the run manifest and the documentation."""
    return pd.DataFrame(
        [
            {
                "name": spec.name,
                "kind": spec.kind,
                "needs_path": spec.needs_path,
                "description": spec.description,
            }
            for spec, _ in TARGETS.values()
        ]
    )
