"""The benchmark's tracks: fixed markets a strategy is scored on, each with its question.

A track is a list of instruments, each a :class:`~trading_research.plugins.market.MarketData`
in the frame contract every strategy plugin reads, and the time trading starts:
the rows before it are the history the strategy is fitted on, the rows after it
are scored.

Two kinds:

* **synthetic** tracks are generated from a fixed seed, so anyone gets the same
  bytes without a download, and they carry a known answer: an edge was planted,
  or nothing was. A strategy built to find the planted effect should find it on
  the planted track and find nothing on its null; one that "finds" money on a
  null is reading the future or fitting noise;
* **recorded** tracks are Bybit's public top-of-book for twenty-six USDT
  perpetuals, two spans of 2024 whose character this repository has already
  measured. They need the data on disk (``trp download``-style archives, see
  :data:`RECORDED_HELP`); without it they are skipped, never faked.

The recorded spans were chosen *after* the repository had read them: the first
is the fortnight where the ten-minute reversion worked, the second the span
after it where it did not. That makes them a benchmark, labelled as such, not
evidence about any strategy: a result on them says how a rule behaves in a
market state already known to exist, not whether that state will come back.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from trading_research.plugins.market import PRINT_COLUMNS, MarketData, demo_market, derive

#: The grid the recorded and cross-section tracks are on: the reversion rule's
#: five seconds, so its frozen look-back and hold keep their meaning.
GRID_SECONDS = 5

#: Where the recorded tracks' archives are expected, relative to the checkout.
RECORDED_HELP = (
    "the recorded tracks read Bybit's public top-of-book for 26 USDT perpetuals, one "
    "parquet per instrument and day under data/universe (1 February to 10 March 2024) "
    "and data/universe_fresh (12 March to 19 April 2024); `trading-research bench fetch` "
    "downloads them from Bybit's free archives (about 2,000 instrument-days)"
)


#: What a track's builder returns: every instrument's market, its first scored
#: row, and its group (one independent market each: a seed, or the one span).
Markets = Union["TrackData"]


@dataclass(frozen=True)
class Track:
    """One market a strategy is scored on, and what a good strategy should do there."""

    name: str
    kind: str  # "synthetic" or "recorded"
    question: str
    expected: str
    source: str
    build: Callable[[Path], Markets] = field(repr=False, compare=False)
    #: For a synthetic track, the effect planted ("" for a null); recorded tracks
    #: carry "" and are read by their own verdict.
    planted: str = ""

    def markets(self, data_root: Path) -> Markets:
        """Each instrument's market and the first scored row."""
        return self.build(data_root)


# --- synthetic --------------------------------------------------------------------

BOOK_STEPS = 216_000  # 200 ms snapshots: twelve hours
#: Three independent markets a track, so a verdict does not rest on one draw.
BOOK_SEEDS = (20261104, 20261204, 20270104)
BOOK_TRAIN_SHARE = 0.4


def _book(signal_strength: float) -> Callable[[Path], Markets]:
    def build(_root: Path) -> Markets:
        parts: list[Panel | list[tuple[MarketData, int]]] = []
        for seed in BOOK_SEEDS:
            market = demo_market(steps=BOOK_STEPS, seed=seed, signal_strength=signal_strength)
            parts.append([(market, int(len(market.frame) * BOOK_TRAIN_SHARE))])
        return TrackData(parts)

    return build


CROSS_INSTRUMENTS = 12
CROSS_DAYS = 8
CROSS_FIT_DAYS = 3
CROSS_SEEDS = (20261105, 20261205, 20270105)
HALF_SPREAD_BP = 1.0
TOUCH_SIZE = 1_000.0


def _cross(null: bool) -> Callable[[Path], Markets]:
    def build(_root: Path) -> Markets:
        from trading_research.synthetic.cross_section import CrossSectionSpec, generate

        parts: list[Panel | list[tuple[MarketData, int]]] = []
        for seed in CROSS_SEEDS:
            spec = CrossSectionSpec(
                n_instruments=CROSS_INSTRUMENTS, days=CROSS_DAYS, bar_s=GRID_SECONDS,
                null=null, seed=seed,
            )  # fmt: skip
            data = generate(spec)
            grids = {s: _touch_from_log(data.log_prices[s]) for s in spec.symbols}
            start = pd.Timestamp(data.days[CROSS_FIT_DAYS], tz="UTC")
            source = (
                f"synthetic: synthetic.cross_section, {CROSS_INSTRUMENTS} instruments, "
                f"{CROSS_DAYS} days of {GRID_SECONDS} s, "
                f"{'null' if null else 'reversal planted'}, seed {seed}, "
                f"a {2 * HALF_SPREAD_BP:g} bp quoted spread"
            )
            parts.append(Panel(grids, source, start=start))
        return TrackData(parts)

    return build


def _touch_from_log(log_price: pd.Series) -> pd.DataFrame:
    """A top-of-book on the bar closes: the price, a fixed spread, a fixed size.

    Prices sit on a tick grid, as a venue's do: a tick of about a millionth of
    the first price. ``exp`` differs in its last bits between platforms; on the
    grid those bits round away, so the track is the same bytes on any machine
    and its fingerprint holds.
    """
    mid = np.exp(log_price.to_numpy(dtype="float64"))
    exponent = int(np.floor(np.log10(mid[0]))) - 6
    tick = float(10**exponent) if exponent >= 0 else 1.0 / 10**-exponent
    mid_ticks = np.rint(mid / tick)
    half_ticks = np.rint(mid_ticks * HALF_SPREAD_BP * 1e-4)
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(log_price.index),
            "bid_price_0": (mid_ticks - half_ticks) * tick,
            "bid_size_0": TOUCH_SIZE,
            "ask_price_0": (mid_ticks + half_ticks) * tick,
            "ask_size_0": TOUCH_SIZE,
        }
    )


# --- recorded ---------------------------------------------------------------------

#: The recorded spans: (directory under the data root, first fitted day, first
#: scored day, last scored day).
RECORDED: dict[str, tuple[str, date, date, date]] = {
    "bybit-2024-02": ("universe", date(2024, 2, 1), date(2024, 2, 26), date(2024, 3, 10)),
    "bybit-2024-04": ("universe_fresh", date(2024, 3, 12), date(2024, 4, 6), date(2024, 4, 19)),
}

BOOK_COLUMNS = ["timestamp", "bid_price_0", "bid_size_0", "ask_price_0", "ask_size_0"]

#: The recorded tracks' instruments, frozen with version 1: the 26 USDT
#: perpetuals of the reversion study (results.md, section 27).
RECORDED_SYMBOLS: tuple[str, ...] = (
    "ADAUSDT", "ALGOUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT",
    "BICOUSDT", "BTCUSDT", "CRVUSDT", "DOGEUSDT", "DOTUSDT", "DYDXUSDT", "ETHUSDT",
    "FILUSDT", "GALAUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "NEARUSDT",
    "OPUSDT", "SOLUSDT", "UNIUSDT", "VETUSDT", "XRPUSDT",
)  # fmt: skip


def recorded_days(name: str) -> list[date]:
    _, first, _, last = RECORDED[name]
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def recorded_present(name: str, data_root: Path) -> list[str]:
    """The track's instruments with every day of the track on disk."""
    directory = data_root / RECORDED[name][0]
    days = recorded_days(name)
    return [
        s
        for s in RECORDED_SYMBOLS
        if all((directory / s / f"{d.isoformat()}.parquet").exists() for d in days)
    ]


def _recorded(name: str) -> Callable[[Path], Markets]:
    def build(data_root: Path) -> Markets:
        from trading_research.data.grid import to_grid

        folder, _, scored, _ = RECORDED[name]
        symbols = recorded_present(name, data_root)
        if len(symbols) < len(RECORDED_SYMBOLS):
            raise FileNotFoundError(
                f"{name}: found {len(symbols)} of {len(RECORDED_SYMBOLS)} instruments with "
                f"every day under {data_root / folder}; {RECORDED_HELP}"
            )
        grids = {}
        for symbol in symbols:
            files = [
                data_root / folder / symbol / f"{d.isoformat()}.parquet"
                for d in recorded_days(name)
            ]
            book = pd.concat(
                [pd.read_parquet(f, columns=BOOK_COLUMNS) for f in files], ignore_index=True
            )
            book["timestamp"] = pd.to_datetime(book["timestamp"], utc=True)
            gridded = to_grid(book.sort_values("timestamp"), GRID_SECONDS, label="right")
            grids[symbol] = gridded.drop_duplicates("timestamp").reset_index(drop=True)
        source = (
            f"recorded: Bybit linear top-of-book, {len(symbols)} instruments, "
            f"{recorded_days(name)[0]} to {recorded_days(name)[-1]}, {GRID_SECONDS} s grid"
        )
        return TrackData([Panel(grids, source, start=pd.Timestamp(scored, tz="UTC"))])

    return build


# --- the panel --------------------------------------------------------------------


class Panel:
    """Instruments on one grid, each made into a market with every other one as a peer.

    Each market is built when it is reached, so a long span of many instruments
    holds one frame with its peers at a time, not all of them.
    """

    def __init__(self, grids: dict[str, pd.DataFrame], source: str, *, start: pd.Timestamp) -> None:
        self.grids = grids
        self.source = source
        self.start = start
        self.mids = {
            s: pd.Series(
                (g["bid_price_0"].to_numpy() + g["ask_price_0"].to_numpy()) / 2.0,
                index=pd.DatetimeIndex(g["timestamp"]),
            )
            for s, g in grids.items()
        }

    def __len__(self) -> int:
        return len(self.grids)

    def __iter__(self) -> Iterator[tuple[MarketData, int]]:
        for symbol in self.grids:
            market = self.market(symbol)
            labels = pd.DatetimeIndex(market.frame["timestamp"])
            yield market, int(labels.searchsorted(self.start))

    def market(self, symbol: str) -> MarketData:
        grid = self.grids[symbol]
        frame = derive(grid[BOOK_COLUMNS].reset_index(drop=True))
        for name in PRINT_COLUMNS:
            frame[name] = 0.0
        labels = pd.DatetimeIndex(frame["timestamp"])
        peers = [s for s in self.grids if s != symbol]
        for k, peer in enumerate(peers, start=1):
            frame[f"peer_{k}_mid"] = self.mids[peer].reindex(labels, method="ffill").to_numpy()
        trades = pd.DataFrame(
            {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "price": [], "size": [],
             "aggressor": pd.Series([], dtype="int64")}
        )  # fmt: skip
        return MarketData(
            symbol, frame, grid[BOOK_COLUMNS], trades, GRID_SECONDS, self.source, 0.0, 0.0
        )


class TrackData:
    """A track's markets, in independent groups, and the fingerprint of what they hold.

    A group is one market that shares nothing with another: a seed of a
    synthetic track, or the one recorded span. Trades are clustered within a
    group (by hour or day), never across groups.
    """

    def __init__(self, parts: list[Panel | list[tuple[MarketData, int]]]) -> None:
        self.parts = parts

    def __iter__(self) -> Iterator[tuple[MarketData, int, int]]:
        for group, part in enumerate(self.parts):
            for market, start in part:
                yield market, start, group

    @property
    def instruments(self) -> int:
        return sum(len(p) for p in self.parts)

    def fingerprint(self) -> str:
        """sha256 of every instrument's gridded top of book, timestamps included.

        The bytes are the numbers, not a file, so the same archives downloaded on
        another machine give the same fingerprint.
        """
        digest = hashlib.sha256()
        for part in self.parts:
            grids = (
                part.grids
                if isinstance(part, Panel)
                else {m.symbol: m.frame[BOOK_COLUMNS] for m, _ in part}
            )
            for symbol in sorted(grids):
                grid = grids[symbol]
                digest.update(symbol.encode())
                stamps = (
                    pd.DatetimeIndex(grid["timestamp"])
                    .as_unit("ns")
                    .to_numpy(dtype="datetime64[ns]")
                    .view("int64")
                )
                digest.update(np.ascontiguousarray(stamps).tobytes())
                for column in BOOK_COLUMNS[1:]:
                    values = grid[column].to_numpy(dtype="float64")
                    digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()


TRACKS: dict[str, Track] = {
    t.name: t
    for t in (
        Track(
            "planted-book",
            "synthetic",
            "Does the strategy find an order-book edge that is known to be there?",
            "a rule reading the touch imbalance earns before fees; others may miss it",
            f"synthetic: data.synthetic.generate, three markets of {BOOK_STEPS} steps of 200 ms, "
            f"seeds {', '.join(map(str, BOOK_SEEDS))}, queue-imbalance signal planted "
            "(strength 0.15)",
            _book(0.15),
            planted="queue imbalance",
        ),
        Track(
            "null-book",
            "synthetic",
            "Does it stay quiet where the same market has nothing planted?",
            "nothing distinguishable from zero before fees",
            "synthetic: the same generator and seeds, signal strength 0",
            _book(0.0),
        ),
        Track(
            "planted-reversion",
            "synthetic",
            "Does a cross-sectional rule find a reversion that is known to be there?",
            "a rule trading against the index move earns before fees",
            f"synthetic: synthetic.cross_section, three universes of {CROSS_INSTRUMENTS} "
            f"instruments and {CROSS_DAYS} days, reversal planted, seeds "
            f"{', '.join(map(str, CROSS_SEEDS))}",
            _cross(False),
            planted="cross-sectional reversion",
        ),
        Track(
            "null-reversion",
            "synthetic",
            "Does it stay quiet on the same universe with the reversion removed?",
            "nothing distinguishable from zero before fees",
            "synthetic: the same universes and seeds, transients replaced by a random walk",
            _cross(True),
        ),
        Track(
            "bybit-2024-02",
            "recorded",
            "A favourable state, known in hindsight: the fortnight where the "
            "ten-minute reversion worked (index ten-minute autocorrelation about -0.10).",
            "a rule betting on short reversion should do its best here",
            "recorded: Bybit top-of-book, 26 USDT perpetuals, fitted 1-25 February, "
            "scored 26 February to 10 March 2024",
            _recorded("bybit-2024-02"),
        ),
        Track(
            "bybit-2024-04",
            "recorded",
            "Three weeks later, after the state changed: the same rule lost here (autocorrelation near zero).",
            "the same rule should lose or stand aside; a rule that only wins in one "
            "state shows it here",
            "recorded: Bybit top-of-book, the same instruments, fitted 12 March to 5 April, "
            "scored 6 to 19 April 2024",
            _recorded("bybit-2024-04"),
        ),
    )
}


def iter_tracks(names: str) -> Iterator[Track]:
    """``all``, or track names separated by commas, in the registry's order."""
    if names == "all":
        yield from TRACKS.values()
        return
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in TRACKS]
    if unknown:
        raise ValueError(f"unknown track(s) {unknown}; known: {list(TRACKS)}")
    for name in TRACKS:
        if name in chosen:
            yield TRACKS[name]


__all__ = ["GRID_SECONDS", "RECORDED", "RECORDED_HELP", "RECORDED_SYMBOLS", "TRACKS", "Markets",
           "Panel", "Track", "TrackData",
           "iter_tracks", "recorded_present"]  # fmt: skip
