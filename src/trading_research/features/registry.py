"""A registry of feature functions, each declaring how far back it looks.

The declaration is the point of this module. A feature that says
``lookback=20`` is making a checkable claim: its value at row *t* depends on
rows *t-20…t* and on nothing after *t*. :mod:`trading_research.validation.leakage` then
verifies that claim mechanically, by truncating the data and confirming that
nothing before the cut moved.

Why bother, when "don't use future data" sounds like something one can simply
do: the three common ways look-ahead gets in are all invisible in the output.

``df.rolling(window, center=True)``
    Centred windows read forward. The result looks like a normal moving
    average and is off by half a window.
``df.fillna(method="bfill")``
    Backward fill copies a later observation into an earlier row. Every filled
    row then knows its own future.
``(x - x.mean()) / x.std()``
    Statistics over the whole sample leak the test period into the training
    period. Nothing is out of order, no row reads a later row, and the model
    still sees information it could not have had.

None of these raises, none looks wrong in a plot, and each inflates a backtest.
So the guarantee is expressed as a property that can fail a build rather than
as a rule people are asked to remember.

Writing a feature
-----------------
A feature is a pure function from a frame to a Series on the same index::

    @feature(name="spread_bp", plane="book", lookback=0)
    def spread_bp(df: pd.DataFrame) -> pd.Series:
        return 1e4 * (best_ask(df) - best_bid(df)) / mid_price(df)

Rules the registry enforces or relies on:

- **Pure.** No state between calls, no fitted parameters. Anything that must be
  fitted — a scaler, a selector — belongs downstream of the split, where it can
  be fitted on training data only.
- **Aligned.** The returned Series shares the input's index, so a feature frame
  can be assembled by concatenation without a join.
- **Warm-up is ``NaN``, not a guess.** The first ``lookback`` rows have
  insufficient history. Leaving them ``NaN`` lets the split drop them
  knowingly; filling them invents data at exactly the point where a model is
  most willing to believe it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Plane = Literal["book", "trades"]

FeatureFn = Callable[[pd.DataFrame], "pd.Series[float]"]


class FeatureError(KeyError):
    """A feature is unknown, or is being used on the wrong plane."""


@dataclass(frozen=True)
class Feature:
    """One registered feature and the claim it makes about its inputs."""

    name: str
    plane: Plane
    lookback: int
    fn: FeatureFn
    description: str
    #: Set when a feature's value legitimately depends on later rows. Only
    #: labels may do this. It exists so that the leakage checker can be run
    #: over targets too, asserting the opposite property.
    forward_looking: bool = False

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        result = self.fn(df)
        if len(result) != len(df):
            raise ValueError(
                f"feature {self.name!r} returned {len(result)} values for {len(df)} rows; "
                f"a feature must be aligned to its input"
            )
        return result.rename(self.name)


@dataclass
class Registry:
    """A named collection of features.

    Kept as an object rather than a module-level dict so that tests can build
    an isolated registry, and so a run can record exactly which features
    existed when it happened.
    """

    features: dict[str, Feature] = field(default_factory=dict)

    def add(self, feature: Feature) -> Feature:
        if feature.name in self.features:
            raise ValueError(f"feature {feature.name!r} is already registered")
        self.features[feature.name] = feature
        return feature

    def get(self, name: str) -> Feature:
        try:
            return self.features[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.features)) or "none"
            raise FeatureError(f"unknown feature {name!r}; registered: {known}") from exc

    def names(self, plane: Plane | None = None) -> list[str]:
        return sorted(f.name for f in self.features.values() if plane is None or f.plane == plane)

    def of_plane(self, plane: Plane) -> list[Feature]:
        return [self.features[n] for n in self.names(plane)]

    def max_lookback(self, names: Iterable[str]) -> int:
        """The longest history any of ``names`` needs.

        This is what a split must discard after a boundary: a feature computed
        across a cut reads rows the model is not supposed to have seen on that
        side of it.
        """
        return max((self.get(n).lookback for n in names), default=0)

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": f.name,
                    "plane": f.plane,
                    "lookback": f.lookback,
                    "description": f.description,
                }
                for f in sorted(self.features.values(), key=lambda f: (f.plane, f.name))
            ]
        )

    def __contains__(self, name: object) -> bool:
        return name in self.features

    def __len__(self) -> int:
        return len(self.features)


#: The registry every feature module populates on import.
REGISTRY = Registry()


def feature(
    *,
    name: str,
    plane: Plane,
    lookback: int,
    description: str = "",
    registry: Registry | None = None,
) -> Callable[[FeatureFn], Feature]:
    """Register a feature function.

    ``lookback`` is the number of *earlier* rows the value at row *t* may
    depend on. Zero means the row itself and nothing else. The number is used
    twice: to size the embargo around a split boundary, and as the claim the
    leakage checker tests.

    Overstating it is merely wasteful — a few more rows are discarded than
    needed. Understating it is a silent leak across every split boundary, so
    when in doubt, round up.
    """
    if lookback < 0:
        raise ValueError(f"lookback must be non-negative, got {lookback}")

    target = registry if registry is not None else REGISTRY

    def register(fn: FeatureFn) -> Feature:
        return target.add(
            Feature(
                name=name,
                plane=plane,
                lookback=lookback,
                fn=fn,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
            )
        )

    return register


def build(
    df: pd.DataFrame,
    names: Sequence[str] | None = None,
    *,
    plane: Plane | None = None,
    registry: Registry | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute features over ``df`` and return them as one frame.

    ``names=None`` means every feature registered for ``plane``. That is
    convenient for exploration and wrong for an experiment: a config should
    name its features so the set cannot drift between runs without the change
    being visible in the diff.

    With ``dropna``, the warm-up rows — where the longest lookback has not yet
    been satisfied — are removed. The timestamp column is carried through so
    the result can be aligned or split without going back to the source.
    """
    reg = registry if registry is not None else REGISTRY

    if names is None:
        if plane is None:
            raise ValueError(
                "give either an explicit list of features or a plane to take them all from"
            )
        names = reg.names(plane)
    if not names:
        raise ValueError("no features requested")

    selected = [reg.get(n) for n in names]
    wrong_plane = [f for f in selected if plane is not None and f.plane != plane]
    if wrong_plane:
        detail = ", ".join(f"{f.name} needs {f.plane}" for f in wrong_plane)
        raise FeatureError(f"features do not match plane {plane!r}: {detail}")

    planes = {f.plane for f in selected}
    if len(planes) > 1:
        raise FeatureError(
            f"features span several planes ({sorted(planes)}); build each plane separately, "
            f"then align them with trading_research.features.align"
        )

    columns = {f.name: f(df) for f in selected}
    out = pd.DataFrame(columns, index=df.index)
    if "timestamp" in df.columns:
        out.insert(0, "timestamp", df["timestamp"])

    if dropna:
        warmup = reg.max_lookback(names)
        out = out.iloc[warmup:].dropna()
    return out
