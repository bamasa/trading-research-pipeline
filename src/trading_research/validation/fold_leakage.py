"""The fold-level leakage test: what a fold fits is a function of its past alone.

:mod:`trading_research.validation.leakage` checks features one at a time. A
walk-forward fold fits more than features: model weights, the per-side
thresholds, the gate's levels and a network's checkpoint, each read from the
training rows, their labels and the inner validation. Any of them can read past
the fold's cutoff without a feature doing so — a label computed over a window
that runs into the test month, a threshold set on rows that were meant to be
purged, a checkpoint chosen on a score that saw a test day.

So the test is on the outputs. Fit the fold once; then change every price after
the cutoff, and every label at a row after it, fit again, and compare each
artefact by sha256. Equal digests for every artefact pass. The cutoff is the
end of the fold's inner validation plus the hold: the inner validation's labels
legitimately read prices up to a hold past its end, and nothing fitted reads
later.

The fit is any callable from the fold's frame to a mapping of named artefacts.
:func:`digest` turns each into a stable sha256: arrays and frames by their
bytes, mappings by sorted keys, an estimator through its own raw serialisation
when it has one (``save_raw``, ``get_booster``, ``state_dict``) and through
pickle otherwise. A fit that is not deterministic for a fixed seed fails this
test without leaking, and that is the right answer too: its artefacts could not
be recorded by sha256.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


class FoldLeakError(AssertionError):
    """A fold's fitted artefacts changed when data after its cutoff was changed."""


def _piece(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "little") + payload


def _canonical(value: Any) -> bytes:
    """Bytes that are equal exactly when the artefacts are, for the types fits return."""
    if value is None:
        return b"none"
    if isinstance(value, bool | np.bool_):
        return _piece(b"bool", b"1" if value else b"0")
    if isinstance(value, int | np.integer):
        return _piece(b"int", str(int(value)).encode())
    if isinstance(value, float | np.floating):
        return _piece(b"float", np.float64(value).tobytes())
    if isinstance(value, str):
        return _piece(b"str", value.encode())
    if isinstance(value, bytes | bytearray | memoryview):
        return _piece(b"bytes", bytes(value))
    if isinstance(value, pd.DataFrame):
        parts = [_canonical(list(map(str, value.columns))), _canonical(value.index)]
        parts += [_canonical(value[c]) for c in value.columns]
        return _piece(b"frame", b"".join(parts))
    if isinstance(value, pd.Series | pd.Index):
        name = str(getattr(value, "name", ""))
        return _piece(b"series", _canonical(name) + _canonical(value.to_numpy()))
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return _piece(b"objects", b"".join(_canonical(v) for v in value.ravel().tolist()))
        head = f"{value.dtype.str}{value.shape}".encode()
        return _piece(b"array", head + np.ascontiguousarray(value).tobytes())
    if isinstance(value, Mapping):
        keys = sorted(value, key=str)
        return _piece(b"map", b"".join(_canonical(str(k)) + _canonical(value[k]) for k in keys))
    if isinstance(value, list | tuple):
        return _piece(b"list", b"".join(_canonical(v) for v in value))
    return _estimator(value)


def _estimator(value: Any) -> bytes:
    if hasattr(value, "detach"):  # a tensor
        return _canonical(value.detach().cpu().numpy())
    if callable(getattr(value, "state_dict", None)):  # a network
        return _piece(b"state", _canonical(dict(value.state_dict())))
    if callable(getattr(value, "save_raw", None)):  # a booster
        return _piece(b"raw", bytes(value.save_raw()))
    if callable(getattr(value, "get_booster", None)):  # a boosted estimator
        return _piece(b"raw", bytes(value.get_booster().save_raw()))
    if is_dataclass(value) and not isinstance(value, type):
        named = {f.name: getattr(value, f.name) for f in fields(value)}
        return _piece(b"dataclass", _canonical(type(value).__name__) + _canonical(named))
    return _piece(b"pickle", pickle.dumps(value, protocol=5))


def digest(value: Any) -> str:
    """The sha256 of an artefact, hex: equal for equal artefacts, across runs."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _times(frame: pd.DataFrame, time_column: str | None) -> pd.DatetimeIndex:
    stamps = frame.index if time_column is None else frame[time_column]
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))


def mutate_after_cutoff(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    *,
    price_columns: Sequence[str],
    label_columns: Sequence[str] = (),
    time_column: str | None = "timestamp",
    seed: int = 0,
) -> pd.DataFrame:
    """``frame`` with every price and label at a row after ``cutoff`` changed.

    Prices are moved together by one random walk that starts ten per cent away,
    and a column named as a bid is pushed down and one named as an ask pushed
    up by the same 5 to 15 bp, so a spread changes as well as a level while the
    book stays uncrossed and its levels in order. A label
    with few distinct values (a class) is redrawn from those values; any other
    label is redrawn from a normal three times as wide as its own spread.
    """
    stamp = pd.Timestamp(cutoff)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    after = np.asarray(_times(frame, time_column) > stamp)
    out = frame.copy()
    count = int(after.sum())
    if count == 0:
        return out
    rng = np.random.default_rng(seed)
    factor = 1.1 * np.exp(np.cumsum(rng.normal(0.0, 0.002, count)))
    widen = rng.uniform(5e-4, 15e-4, count)
    for column in price_columns:
        values = out[column].to_numpy(dtype="float64").copy()
        side = 1.0 - widen if "bid" in column else 1.0 + widen if "ask" in column else 1.0
        values[after] = values[after] * factor * side
        out[column] = values
    for column in label_columns:
        values = out[column].to_numpy().copy()
        distinct = pd.unique(values[np.isfinite(values.astype("float64"))])
        if len(distinct) <= 10:
            values[after] = rng.choice(distinct, size=count)
        else:
            spread = float(np.nanstd(values.astype("float64"))) or 1.0
            centre = float(np.nanmean(values.astype("float64")))
            values[after] = rng.normal(centre, 3.0 * spread, count)
        out[column] = values
    return out


@dataclass(frozen=True)
class FoldLeakResult:
    """One fold: the digest of every artefact before and after the mutation."""

    fold: str
    cutoff: pd.Timestamp
    digests: dict[str, str]
    mutated: dict[str, str]

    @property
    def changed(self) -> tuple[str, ...]:
        names = sorted(set(self.digests) | set(self.mutated))
        return tuple(n for n in names if self.digests.get(n) != self.mutated.get(n))

    @property
    def leaks(self) -> bool:
        return bool(self.changed)

    def __str__(self) -> str:
        if not self.leaks:
            return (
                f"fold {self.fold}: {len(self.digests)} artefact(s) unchanged after {self.cutoff}"
            )
        return (
            f"fold {self.fold}: LEAKS — {', '.join(self.changed)} changed when the data "
            f"after {self.cutoff} was changed"
        )


FoldFit = Callable[[pd.DataFrame], Mapping[str, Any]]


def check_fold(
    fit: FoldFit,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    *,
    price_columns: Sequence[str],
    label_columns: Sequence[str] = (),
    time_column: str | None = "timestamp",
    fold: str = "",
    seed: int = 0,
) -> FoldLeakResult:
    """Fit on ``frame`` and on its mutation after ``cutoff``; compare every artefact."""
    stamp = pd.Timestamp(cutoff)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    before = {str(k): digest(v) for k, v in fit(frame).items()}
    changed = mutate_after_cutoff(
        frame, stamp, price_columns=price_columns, label_columns=label_columns,
        time_column=time_column, seed=seed,
    )  # fmt: skip
    after = {str(k): digest(v) for k, v in fit(changed).items()}
    return FoldLeakResult(fold or f"{stamp:%Y-%m-%d}", stamp, before, after)


def assert_folds_causal(results: Sequence[FoldLeakResult]) -> None:
    """Raise if any fold's artefacts moved (the K-leak condition for the fold-level test)."""
    leaking = [r for r in results if r.leaks]
    if leaking:
        detail = "\n".join(f"  {r}" for r in leaking)
        raise FoldLeakError(f"{len(leaking)} fold(s) fitted on data after their cutoff:\n{detail}")
