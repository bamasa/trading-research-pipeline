"""Where a walkthrough's data lives, which days a step may see, and a synthetic market in that layout.

A dataset is four directories, each in the layout the repository's
downloaders write and its readers read:

``raw/<SYMBOL>/bookTicker/<day>.parquet`` and ``raw/<SYMBOL>/aggTrades/<day>.parquet``
    the best bid and ask and the trades (Binance's contract): what the screen,
    ``prepare`` and the grid executions read;
``book/<SYMBOL>/<day>.parquet``
    ten levels a side (Bybit's book on a grid): what the market maker quotes against;
``trades/<SYMBOL>/<day>.parquet``
    prints with their aggressor side: what fills a resting order;
``funding/<SYMBOL>/<day>.parquet``
    funding settlements: what a held position pays or earns.

A step never reads the dataset directly. It reads a **view**: a directory of
links to the days it is allowed, made by :func:`view`. The search steps get a
view of the search block only, so a held-out day is not in any directory they
can open; the read gets its own view, once. Days a registration of this
repository protects are refused before anything is read
(:func:`protected_overlap`).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLANES = ("raw", "book", "trades", "funding")
RAW_KINDS = ("bookTicker", "aggTrades")


@dataclass(frozen=True)
class Dataset:
    """The four roots of one dataset; ``book``, ``trades`` and ``funding`` are
    optional, and without them the passive and maker executions cannot run."""

    raw: Path
    book: Path | None = None
    trades: Path | None = None
    funding: Path | None = None

    @classmethod
    def under(cls, root: Path | str) -> Dataset:
        """The standard layout below one directory."""
        base = Path(root)
        return cls(base / "raw", base / "book", base / "trades", base / "funding")

    def symbols(self) -> list[str]:
        """Instruments with best bid and ask files, in name order."""
        if not self.raw.is_dir():
            return []
        return sorted(
            p.name
            for p in self.raw.iterdir()
            if p.is_dir() and any((p / "bookTicker").glob("*.parquet"))
        )

    def days(self, symbol: str) -> list[date]:
        """The days of ``symbol``'s best bid and ask files."""
        return sorted(
            date.fromisoformat(p.stem) for p in (self.raw / symbol / "bookTicker").glob("*.parquet")
        )

    def has_event_time(self, symbol: str, days: Iterable[date]) -> bool:
        """Whether book, prints and funding exist for every day: what the market maker needs."""
        roots = (self.book, self.trades, self.funding)
        if any(r is None for r in roots):
            return False
        return all((r / symbol / f"{d}.parquet").exists() for r in roots for d in days)  # type: ignore[operator]

    def as_dict(self) -> dict[str, str | None]:
        return {k: (str(v) if v is not None else None) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Dataset:
        return cls(*(Path(values[k]) if values.get(k) else None for k in PLANES))  # type: ignore[arg-type]


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def view(dataset: Dataset, symbols: Sequence[str], days: Sequence[date], into: Path) -> Dataset:
    """A dataset of links to ``days`` of ``symbols`` only, under ``into``.

    Made afresh each time, so a day that is not asked for is not there. A
    plane's manifest is linked with its days, so the readers that need one
    (``validate-data``) still find it.
    """
    if into.exists():
        for path in sorted(into.rglob("*"), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    wanted = {d.isoformat() for d in days}
    out = Dataset.under(into)
    for symbol in symbols:
        for kind in RAW_KINDS:
            source = dataset.raw / symbol / kind
            for path in sorted(source.glob("*.parquet")):
                if path.stem in wanted:
                    _link(path, out.raw / symbol / kind / path.name)
            if (source / "manifest.json").exists():
                _link(source / "manifest.json", out.raw / symbol / kind / "manifest.json")
        for plane in ("book", "trades", "funding"):
            root = getattr(dataset, plane)
            if root is None:
                continue
            for day in wanted:
                path = root / symbol / f"{day}.parquet"
                if path.exists():
                    _link(path, getattr(out, plane) / symbol / path.name)
    return replace(
        out,
        book=out.book if dataset.book is not None else None,
        trades=out.trades if dataset.trades is not None else None,
        funding=out.funding if dataset.funding is not None else None,
    )


def split_days(days: Sequence[date], search_share: float) -> tuple[list[date], list[date]]:
    """The search block (the first ``search_share`` of the days, whole days) and the rest."""
    ordered = sorted(days)
    if len(ordered) < 2:
        raise ValueError(
            f"{len(ordered)} day(s) cannot be split into a search and a held-out block"
        )
    cut = min(len(ordered) - 1, max(1, round(len(ordered) * search_share)))
    return ordered[:cut], ordered[cut:]


def protected_overlap(days: Iterable[date], repo: Path | None = None) -> list[str]:
    """The ranges of this repository's registrations that ``days`` touch.

    Read by the agent harness's own calendar (``agents/workflows/policy.py``),
    which reads every ``configs/*.yaml``; a checkout without the harness has no
    registration to protect.
    """
    root = repo or Path(__file__).resolve().parents[3]
    if not (root / "agents" / "workflows" / "policy.py").exists():
        return []
    sys.path.insert(0, str(root))
    try:
        from agents.workflows.policy import protected_calendar
    finally:
        sys.path.remove(str(root))
    wanted = sorted(set(days))
    return sorted(
        {p.key for p in protected_calendar(root) for d in wanted if p.first <= d <= p.last}
    )


# ---------------------------------------------------------------------------
# A synthetic market in the layout
# ---------------------------------------------------------------------------

#: Defaults for a synthetic instrument: a mid-priced perpetual whose moves over
#: a few minutes are of the order of a taker's round trip, so every execution
#: has something to decide. Overrides of
#: :class:`trading_research.data.synthetic.SyntheticConfig`; nothing new.
SYNTHETIC_DEFAULTS: dict[str, Any] = {
    "step_ms": 1000,
    "initial_price": 100.0,
    "tick_size": 0.01,
    "lot_size": 0.1,
    "vol_calm": 1.6e-4,
    "vol_excited": 4.0e-4,
    "base_depth_size": 400.0,
    "trade_intensity": 0.5,
    "trade_size_log_mean": 3.0,
}


@dataclass(frozen=True)
class SyntheticInstrument:
    """One generated instrument: its name, its seed and the generator's strength of signal."""

    symbol: str
    seed: int
    signal_strength: float


def write_synthetic(
    root: Path,
    instruments: Sequence[SyntheticInstrument],
    *,
    first_day: date,
    days: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate each instrument with :mod:`trading_research.data.synthetic` and write it
    in every plane of the layout; return what was generated, with each file's sha256.

    Funding is written as zero at the three daily settlements: the generator
    has no funding, and a made-up rate would be a number nobody chose for a
    reason.
    """
    import hashlib

    from trading_research.data import synthetic

    settings = {**SYNTHETIC_DEFAULTS, **(overrides or {})}
    dataset = Dataset.under(root)
    record: dict[str, Any] = {
        "generator": synthetic.GENERATOR_NAME,
        "settings": settings,
        "instruments": [],
    }
    for item in instruments:
        config = synthetic.SyntheticConfig(
            symbol=item.symbol,
            n_steps=int(days * 86_400_000 / settings["step_ms"]),
            start=f"{first_day.isoformat()}T00:00:00Z",
            seed=item.seed,
            signal_strength=item.signal_strength,
            **settings,
        )
        generated = synthetic.generate(config)
        files: dict[str, str] = {}
        for day, book in generated.book.groupby(generated.book["timestamp"].dt.date):
            trades = generated.trades[generated.trades["timestamp"].dt.date == day]
            for path, frame in _planes(dataset, item.symbol, day, book, trades):
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for kind, plane in (("bookTicker", "book"), ("aggTrades", "trades")):
            manifest = generated.manifests[plane].to_dict()
            manifest.update(depth=1 if plane == "book" else None)
            (dataset.raw / item.symbol / kind / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str) + "\n"
            )
        record["instruments"].append({**asdict(item), "config": asdict(config), "files": files})
    (root / "synthetic.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def _planes(
    dataset: Dataset, symbol: str, day: date, book: pd.DataFrame, trades: pd.DataFrame
) -> list[tuple[Path, pd.DataFrame]]:
    name = f"{day.isoformat()}.parquet"
    top = [
        "timestamp",
        "symbol",
        "sequence_id",
        "source",
        "bid_price_0",
        "ask_price_0",
        "bid_size_0",
        "ask_size_0",
    ]
    levels = [c for c in book.columns if c.startswith(("bid_", "ask_"))]
    prints = pd.DataFrame(
        {
            "timestamp": trades["timestamp"].to_numpy(),
            "price": trades["price"].to_numpy(),
            "size": trades["quantity"].to_numpy(),
            # A buyer who is the maker means the seller crossed: aggressor -1.
            "aggressor": np.where(trades["is_buyer_maker"].to_numpy(dtype=bool), -1, 1).astype(
                np.int64
            ),
        }
    )
    start = pd.Timestamp(day.isoformat(), tz="UTC")
    funding = pd.DataFrame(
        {"timestamp": pd.date_range(start, periods=3, freq="8h"), "rate": np.zeros(3)}
    )
    assert dataset.book is not None and dataset.trades is not None and dataset.funding is not None
    return [
        (dataset.raw / symbol / "bookTicker" / name, book[top].reset_index(drop=True)),
        (dataset.raw / symbol / "aggTrades" / name, trades.reset_index(drop=True)),
        (dataset.book / symbol / name, book[["timestamp", *levels]].reset_index(drop=True)),
        (dataset.trades / symbol / name, prints),
        (dataset.funding / symbol / name, funding),
    ]


def calendar(first: date, count: int) -> list[date]:
    return [first + timedelta(days=i) for i in range(count)]
