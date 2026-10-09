"""The catalogue of taker cells, generated from the registration and checked by hash.

The registration writes every taker cell as a key, for stratum C

    <instrument>|C|<model>|<inputs>|<gate>|hold_s=<h>|trades_per_day=<n>|label_fee_bp=<f>|regime=<r>

and for stratum R

    <instrument>|R|<model>|<inputs>|<gate>|hold_s=<h>|k=<k>|beta_bp=<b>|regime=<r>

with numbers written as in the configuration (``0``, ``0.5``, ``5.5``). The
keys, sorted in byte order and each followed by a newline, UTF-8, hash to the
registered sha256. :func:`catalogue_keys` fills the registered templates from
the configuration's axes, so the hash checks the axes and the templates at
once; :class:`Catalogue` refuses to be used if the hash differs and refuses any
key outside it.

:func:`parse_key` turns a key into numbers for the ordered axes: hold in
seconds, the rate in trades a day, k, beta and the label fee in bp, and the
regime window in hours (``none`` 0, ``pull_4h`` 4, ``pull_24h`` 24,
``pull_120h`` 120), never the strings, which would sort ``pull_120h`` before
``pull_4h``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_research.validation.choice import neighbourhood_index

#: The registration, relative to the repository root.
CONFIG_PATH = Path("configs/short_horizon_2025.yaml")
#: The regime policies as windows in hours, ``none`` being window 0.
REGIME_WINDOW_H: dict[str, int] = {"none": 0, "pull_4h": 4, "pull_24h": 24, "pull_120h": 120}
#: The ordered axes, as the numeric columns of :func:`axes_table`.
ORDERED_AXES: tuple[str, ...] = (
    "hold_s",
    "trades_per_day",
    "label_fee_bp",
    "k",
    "beta_bp",
    "regime_h",
)
#: The values a cell shares with every cell of its neighbourhood.
GROUP_COLUMNS: tuple[str, ...] = ("stratum", "model", "inputs", "gate")
STRATA: tuple[str, ...] = ("C", "R")


class CatalogueError(ValueError):
    """The catalogue differs from the registered one, or a key is outside it."""


def written(value: Any, allowed: Iterable[str] | None = None) -> str:
    """A number as the configuration writes it: ``0``, ``0.5``, ``5.5``, ``240``.

    An integer is written without a decimal point and a float in its shortest
    form; with ``allowed``, a number written any other way is refused.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CatalogueError(f"{value!r} is not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise CatalogueError(f"{value!r} is not finite")
    text = str(value) if isinstance(value, int) else repr(float(value))
    if allowed is not None and text not in set(allowed):
        raise CatalogueError(f"{value!r} is written {text!r}, not as registered")
    return text


def _fill(template: str, values: Mapping[str, str]) -> str:
    key = template
    for name, value in values.items():
        placeholder = f"<{name}>"
        if placeholder not in key:
            raise CatalogueError(f"the template {template!r} has no {placeholder}")
        key = key.replace(placeholder, value)
    if "<" in key or ">" in key:
        raise CatalogueError(f"the template {template!r} kept a placeholder: {key!r}")
    return key


def catalogue_keys(search: Mapping[str, Any]) -> list[str]:
    """Every taker cell key of the configuration's ``search`` block, unsorted."""
    axes, catalogue = search["axes"], search["catalogue"]
    allowed = [written(v) for v in catalogue["numbers_written_as"]]

    def numbers(axis: str) -> list[str]:
        return [written(v, allowed) for v in axes[axis]["values"]]

    keys: list[str] = []
    for instrument in search["instruments"]:
        for model, inputs, gate, h, n, f, r in itertools.product(
            axes["model"]["values"]["C"],
            axes["inputs"]["values"]["C"],
            axes["gate"]["values"],
            numbers("hold_s"),
            numbers("trades_per_day"),
            numbers("label_fee_bp"),
            axes["regime"]["values"],
        ):
            values = {"instrument": instrument, "model": model, "inputs": inputs}
            values |= {"gate": gate, "h": h, "n": n, "f": f, "r": r}
            keys.append(_fill(catalogue["key_C"], values))
        r_inputs = axes["inputs"]["values"]["R"]
        for model, gate, h, k, b, r in itertools.product(
            axes["model"]["values"]["R"],
            axes["gate"]["values"],
            numbers("hold_s"),
            numbers("k"),
            numbers("beta_bp"),
            axes["regime"]["values"],
        ):
            values = {"instrument": instrument, "model": model, "inputs": r_inputs[model]}
            values |= {"gate": gate, "h": h, "k": k, "b": b, "r": r}
            keys.append(_fill(catalogue["key_R"], values))
    return keys


def serialise(keys: Iterable[str]) -> bytes:
    """The registered serialisation: byte order, a newline after each key, UTF-8."""
    encoded = sorted(key.encode("utf-8") for key in keys)
    return b"".join(key + b"\n" for key in encoded)


def catalogue_sha256(keys: Iterable[str]) -> str:
    return hashlib.sha256(serialise(keys)).hexdigest()


@dataclass(frozen=True)
class CellKey:
    """A key's values; the ordered axes as numbers, NaN where the stratum has none."""

    key: str
    instrument: str
    stratum: str
    model: str
    inputs: str
    gate: str
    hold_s: float
    trades_per_day: float
    label_fee_bp: float
    k: float
    beta_bp: float
    regime: str
    regime_h: int

    @property
    def configuration(self) -> str:
        """The signal configuration the cell's fit comes from: C model x inputs x
        hold x label fee, R model x hold (the inputs fixed by the model)."""
        parts = self.key.split("|")
        head = "|".join(parts[:4])
        if self.stratum == "C":
            return f"{head}|{parts[5]}|{parts[7]}"
        return f"{head}|{parts[5]}"

    def row(self) -> dict[str, Any]:
        names = ("key", "instrument", *GROUP_COLUMNS, *ORDERED_AXES, "regime")
        return {name: getattr(self, name) for name in names}


_NUMBERS = {"C": ("hold_s", "trades_per_day", "label_fee_bp"), "R": ("hold_s", "k", "beta_bp")}


def parse_key(key: str) -> CellKey:
    """A catalogue key's values, the ordered axes as numbers."""
    parts = key.split("|")
    if len(parts) != 9 or parts[1] not in STRATA:
        raise CatalogueError(f"{key!r} is not a taker cell key")
    instrument, stratum, model, inputs, gate = parts[:5]
    pairs = dict(part.split("=", 1) for part in parts[5:] if "=" in part)
    names = (*_NUMBERS[stratum], "regime")
    if list(pairs) != list(names):
        raise CatalogueError(f"{key!r} does not name {names} in order")
    if pairs["regime"] not in REGIME_WINDOW_H:
        raise CatalogueError(f"{key!r} has an unknown regime {pairs['regime']!r}")
    values = dict.fromkeys(("hold_s", "trades_per_day", "label_fee_bp", "k", "beta_bp"), math.nan)
    for name in _NUMBERS[stratum]:
        try:
            values[name] = float(pairs[name])
        except ValueError as exc:
            raise CatalogueError(f"{key!r}: {name} is not a number") from exc
    regime = pairs["regime"]
    return CellKey(key, instrument, stratum, model, inputs, gate, **values,
                   regime=regime, regime_h=REGIME_WINDOW_H[regime])  # fmt: skip


def axes_table(keys: Sequence[str]) -> pd.DataFrame:
    """One row per key, in the given order: its categorical and numeric values."""
    return pd.DataFrame([parse_key(key).row() for key in keys])


@dataclass(frozen=True)
class Catalogue:
    """The registered catalogue, in byte order, with its registered sha256."""

    keys: tuple[str, ...]
    registered_sha256: str

    def __post_init__(self) -> None:
        if len(set(self.keys)) != len(self.keys):
            raise CatalogueError("the catalogue repeats a key")
        if list(self.keys) != sorted(self.keys, key=lambda k: k.encode("utf-8")):
            raise CatalogueError("the catalogue's keys are not in byte order")

    @classmethod
    def from_search(cls, search: Mapping[str, Any]) -> Catalogue:
        keys = sorted(catalogue_keys(search), key=lambda k: k.encode("utf-8"))
        return cls(tuple(keys), str(search["catalogue"]["sha256"]))

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> Catalogue:
        """From the registration file, checked against its registered hash."""
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_search(config["search"]).verify()

    @property
    def sha256(self) -> str:
        return catalogue_sha256(self.keys)

    def verify(self) -> Catalogue:
        """Refuse a catalogue whose hash differs from the registered one."""
        if self.sha256 != self.registered_sha256:
            raise CatalogueError(
                f"catalogue sha256 {self.sha256} differs from the registered "
                f"{self.registered_sha256}; nothing is scored"
            )
        return self

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: object) -> bool:
        return key in self._positions

    @cached_property
    def _positions(self) -> dict[str, int]:
        return {key: i for i, key in enumerate(self.keys)}

    def require(self, keys: Iterable[str]) -> None:
        """Refuse any key outside the catalogue."""
        positions = self._positions
        outside = sorted({key for key in keys if key not in positions})
        if outside:
            raise CatalogueError(f"{len(outside)} keys outside the catalogue, e.g. {outside[0]!r}")

    def table(self) -> pd.DataFrame:
        return axes_table(self.keys)

    def neighbourhoods(self) -> np.ndarray:
        """Each cell's neighbourhood (row positions, -1 pads) over the ordered
        axes as numbers, within its stratum, model, inputs and gate."""
        return neighbourhood_index(self.table(), ORDERED_AXES, by=GROUP_COLUMNS)

    def strata(self) -> dict[str, np.ndarray]:
        """Each stratum's row positions, for the choice and the nulls."""
        stratum = self.table()["stratum"].to_numpy()
        return {name: np.flatnonzero(stratum == name) for name in STRATA}

    def configurations(self) -> dict[str, list[str]]:
        """The cells of each signal configuration, in catalogue order."""
        out: dict[str, list[str]] = {}
        for key in self.keys:
            out.setdefault(parse_key(key).configuration, []).append(key)
        return out
