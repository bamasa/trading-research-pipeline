"""What every generator in this package shares: truth, stamps, layout and checks.

Every dataset written here is a directory holding

* the planes in the repository's contracts (:mod:`trading_research.data.schema`),
  one Parquet file per instrument-day under ``<plane>/<SYMBOL>/<day>.parquet``
  with a ``manifest.json`` naming the plane, which is the day-partitioned layout
  ``trading-research validate-data`` discovers and the layout the market-making
  loader reads (``book_roots=[<dir>/book]``);
* ``truth.json``: the generator, its spec, its seed and every planted
  parameter, so a check can compare what it recovered with what was put in.

Every row is stamped ``source="synthetic"`` and every manifest says the data is
generated. A result computed on it is a statement about the generator and the
code that read it, never about a market.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import pandas as pd

from trading_research import __version__
from trading_research.data.schema import SCHEMA_VERSION
from trading_research.data.validate import ValidationReport, validate

#: Marker written into the ``source`` column of every generated row.
SYNTHETIC_SOURCE: Final = "synthetic"

#: The first day every generator stamps by default. Arbitrary, and outside the
#: blocks of every registered study (the earliest is 2024).
DEFAULT_START: Final = "2020-01-06"

NOTE: Final = "Synthetic data with a planted, known answer. Not a market; not evidence of edge."

NS_PER_S: Final = 1_000_000_000

SpecT = TypeVar("SpecT")


def with_changes(spec: SpecT, **changes: Any) -> SpecT:
    """A copy of a frozen spec with the named fields replaced, refusing unknown names."""
    if not is_dataclass(spec) or isinstance(spec, type):
        raise TypeError("a spec is a dataclass instance")
    known = {f.name for f in fields(spec)}
    unknown = set(changes) - known
    if unknown:
        raise TypeError(f"unknown spec field(s): {', '.join(sorted(unknown))}")
    return replace(spec, **changes)


def spec_dict(spec: object) -> dict[str, Any]:
    """A spec as plain JSON-ready values."""
    if not is_dataclass(spec) or isinstance(spec, type):
        raise TypeError("a spec is a dataclass instance")
    return {k: _plain(v) for k, v in asdict(spec).items()}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class Truth:
    """What a generator planted, written beside the data as ``truth.json``.

    ``planted`` holds the parameters a check compares its recovery with (sizes
    in bp, half-lives, rates, break times); ``spec`` the whole spec, so the
    dataset can be regenerated bit for bit from it and the seed.
    """

    generator: str
    seed: int
    null: bool
    planted: dict[str, Any]
    spec: dict[str, Any]
    proves: str
    does_not_prove: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "trading_research_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "source": SYNTHETIC_SOURCE,
            "note": NOTE,
            "seed": self.seed,
            "null": self.null,
            "planted": _plain(self.planted),
            "spec": _plain(self.spec),
            "proves": self.proves,
            "does_not_prove": self.does_not_prove,
            **_plain(self.extra),
        }

    def write(self, directory: Path | str) -> Path:
        path = Path(directory) / "truth.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def read_truth(directory: Path | str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((Path(directory) / "truth.json").read_text("utf-8"))
    return loaded


def day_list(start: str, days: int) -> list[date]:
    """``days`` consecutive calendar days from ``start``."""
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    first = date.fromisoformat(start)
    return [first + timedelta(days=k) for k in range(days)]


def day_start_ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS_PER_S


def stamps(ns: np.ndarray) -> pd.DatetimeIndex:
    """Nanoseconds since the epoch as a UTC index."""
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ns, dtype=np.int64), utc=True))


def strings(value: str, n: int) -> Any:
    return pd.array([value] * n, dtype="string")


def write_plane(
    root: Path | str,
    plane: str,
    symbol: str,
    frames: dict[date, pd.DataFrame],
    *,
    generator: str,
    seed: int,
    depth: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one plane, a file per day, with the manifest ``validate-data`` reads."""
    directory = Path(root) / plane / symbol
    directory.mkdir(parents=True, exist_ok=True)
    rows = 0
    for day, frame in sorted(frames.items()):
        frame.to_parquet(directory / f"{day.isoformat()}.parquet", index=False)
        rows += len(frame)
    days = sorted(frames)
    manifest: dict[str, Any] = {
        "plane": plane,
        "symbol": symbol,
        "source": SYNTHETIC_SOURCE,
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "start": days[0].isoformat() if days else None,
        "end": days[-1].isoformat() if days else None,
        "seed": seed,
        "generator": generator,
        "trading_research_version": __version__,
        "note": NOTE,
    }
    if depth is not None:
        manifest["depth"] = depth
    if extra:
        manifest.update(_plain(extra))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return directory


def check_frame(frame: pd.DataFrame, plane: str, *, depth: int | None = None) -> ValidationReport:
    """Validate a generated frame with the repository's own validator, and refuse errors."""
    report = validate(frame, plane, depth=depth)
    report.raise_if_failed()
    return report


def rng_for(seed: int, *keys: int) -> np.random.Generator:
    """A generator seeded by ``seed`` and a path of integers (a day, an instrument)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(k) for k in keys]]))
