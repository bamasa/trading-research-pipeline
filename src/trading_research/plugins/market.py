"""The market a plugin is run on: one instrument's frame, its book and its prints.

:func:`demo_market` builds it from :func:`trading_research.data.synthetic.generate`,
so nothing is downloaded and the answer is known: the generator plants a weak,
causal order-flow signal (``signal_strength``) and nothing else. Peers for the
cross-sectional strategies are further draws of the same generator with other
seeds, so they are independent of the traded instrument by construction: a
cross-sectional or pairs rule has nothing to find on them, and its numbers
measure the machinery only. :func:`market_from_dataset` reads a dataset in the
store's layout instead (``generate-demo-data`` writes one).

Synthetic data is not a market. A result on it says the code ran as stated and
nothing about any venue.

Three views of the same data
----------------------------
``frame``
    What a strategy sees: the book on a regular grid of ``grid_seconds``,
    labelled by the end of each bin (:func:`trading_research.data.grid.to_grid`
    with ``label="right"``), ``mid``, ``log_mid`` and ``spread_bp`` derived from
    it, the prints each row owns under the backward rule
    (:func:`trading_research.features.prints.bucket_prints`) and ``peer_<k>_mid``.
``grid_market``
    What the grid executions read: the same labels, the touch, and the volume
    that would consume a resting order after each label
    (:func:`trading_research.pipeline.execution.aggressive_flow`).
``day_events`` and ``taker_books``
    What the event-time executions read: the raw snapshots and prints of each
    UTC day, in ticks for the market-making simulator.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Columns computed from the level columns, recomputed after any change to them.
DERIVED: tuple[str, ...] = ("mid", "log_mid", "spread_bp")

#: Columns the prints contribute to each row, under the backward rule.
PRINT_COLUMNS: tuple[str, ...] = (
    "buy_volume",
    "sell_volume",
    "trade_count",
    "signed_notional",
    "traded_notional",
    "largest_trade",
    "vwap_deviation_bp",
)

NS_PER_S = 1_000_000_000


def level_columns(depth: int) -> list[str]:
    out = []
    for level in range(depth):
        out += [
            f"bid_price_{level}",
            f"bid_size_{level}",
            f"ask_price_{level}",
            f"ask_size_{level}",
        ]
    return out


def depth_of(frame: pd.DataFrame) -> int:
    level = 0
    while f"bid_price_{level}" in frame.columns and f"ask_price_{level}" in frame.columns:
        level += 1
    return level


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """``mid``, ``log_mid`` and ``spread_bp`` from the touch, in place of any old ones."""
    bid = frame["bid_price_0"].to_numpy(dtype="float64")
    ask = frame["ask_price_0"].to_numpy(dtype="float64")
    mid = (bid + ask) / 2.0
    out = frame.copy()
    out["mid"] = mid
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_mid"] = np.log(mid)
        out["spread_bp"] = (ask - bid) / mid * 1e4
    return out


def to_prints(trades: pd.DataFrame) -> pd.DataFrame:
    """The trade plane in the prints contract: ``timestamp, price, size, aggressor``."""
    if "aggressor" in trades.columns and "size" in trades.columns:
        return trades[["timestamp", "price", "size", "aggressor"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "timestamp": trades["timestamp"].to_numpy(),
            "price": trades["price"].to_numpy(dtype="float64"),
            "size": trades["quantity"].to_numpy(dtype="float64"),
            # The exchange convention: a buyer that was the maker means the
            # seller crossed.
            "aggressor": np.where(trades["is_buyer_maker"].to_numpy(dtype=bool), -1, 1),
        }
    )


def _grid(book: pd.DataFrame, seconds: int, depth: int) -> pd.DataFrame:
    from trading_research.data.grid import to_grid

    columns = ["timestamp", *level_columns(depth)]
    return to_grid(book[columns], seconds, label="right").reset_index(drop=True)


@dataclass
class MarketData:
    """One instrument, as a strategy, a grid execution and an event-time one see it."""

    symbol: str
    frame: pd.DataFrame
    book: pd.DataFrame
    trades: pd.DataFrame
    grid_seconds: int
    source: str
    tick: float
    lot: float
    peers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return depth_of(self.frame)

    @property
    def labels_ns(self) -> np.ndarray:
        return _ns(self.frame["timestamp"])

    @property
    def rows(self) -> pd.DataFrame:
        """The gridded book alone: ``timestamp`` and the level columns."""
        return self.frame[["timestamp", *level_columns(self.depth)]]

    @property
    def prints(self) -> pd.DataFrame:
        return to_prints(self.trades)

    def grid_market(self) -> Any:
        from trading_research.pipeline.execution import GridMarket, aggressive_flow

        bid = self.frame["bid_price_0"].to_numpy(dtype="float64")
        ask = self.frame["ask_price_0"].to_numpy(dtype="float64")
        sell_at_bid, buy_at_ask = aggressive_flow(self.prints, self.frame["timestamp"], bid, ask)
        return GridMarket(
            self.symbol,
            self.labels_ns,
            bid,
            ask,
            self.frame["bid_size_0"].to_numpy(dtype="float64"),
            self.frame["ask_size_0"].to_numpy(dtype="float64"),
            sell_at_bid=sell_at_bid,
            buy_at_ask=buy_at_ask,
            tick=self.tick,
        )

    def _days(self) -> list[tuple[Any, pd.DataFrame, pd.DataFrame]]:
        book_days = pd.DatetimeIndex(self.book["timestamp"]).date
        trade_days = pd.DatetimeIndex(self.trades["timestamp"]).date
        out = []
        for day in sorted(set(book_days)):
            out.append((day, self.book[book_days == day], self.trades[trade_days == day]))
        return out

    def day_events(self) -> list[Any]:
        """Each UTC day's snapshots and prints, in ticks, for the simulator."""
        from trading_research.market_making.events import InstrumentSpec, assemble, to_ticks

        spec = InstrumentSpec(self.symbol, self.tick, self.lot)
        out = []
        depth = depth_of(self.book)
        for day, book, trades in self._days():
            prints = to_prints(trades)
            if book.empty or prints.empty:
                continue
            px = {
                side: to_ticks(
                    book[[f"{side}_price_{i}" for i in range(depth)]].to_numpy(), self.tick
                )
                for side in ("bid", "ask")
            }
            sz = {
                side: book[[f"{side}_size_{i}" for i in range(depth)]].to_numpy(dtype="float64")
                for side in ("bid", "ask")
            }
            out.append(
                assemble(
                    spec,
                    day,
                    book_ts=_ns(book["timestamp"]),
                    bid_px=px["bid"],
                    bid_sz=sz["bid"],
                    ask_px=px["ask"],
                    ask_sz=sz["ask"],
                    trade_ts=_ns(prints["timestamp"]),
                    trade_px=to_ticks(prints["price"].to_numpy(), self.tick),
                    trade_sz=prints["size"].to_numpy(dtype="float64"),
                    trade_aggressor=prints["aggressor"].to_numpy(dtype=np.int8),
                )
            )
        return out

    def taker_books(self) -> list[Any]:
        """Each UTC day's snapshots in quote units, for the event-time taker."""
        from trading_research.backtest.taker import TakerBook

        depth = depth_of(self.book)
        return [TakerBook.from_frame(book, day, depth=depth) for day, book, _ in self._days()]

    def signals(self, decisions: np.ndarray, hold: int) -> Any:
        """Per-row decisions as the stream every execution mode reads."""
        from trading_research.pipeline.execution import SignalStream

        return SignalStream.from_grid(
            self.symbol,
            self.labels_ns,
            np.asarray(decisions, dtype=np.int64),
            hold_rows=hold,
            grid_seconds=self.grid_seconds,
        )

    def until(self, stamp: pd.Timestamp) -> MarketData:
        """The market as it stood at ``stamp``: every snapshot and print after it removed."""
        book = self.book[pd.DatetimeIndex(self.book["timestamp"]) <= stamp]
        trades = self.trades[pd.DatetimeIndex(self.trades["timestamp"]) <= stamp]
        peers = {
            name: (
                peer_book[pd.DatetimeIndex(peer_book["timestamp"]) <= stamp],
                peer_trades,
            )
            for name, (peer_book, peer_trades) in self.peers.items()
        }
        return build_market(
            self.symbol, book, trades, self.grid_seconds, self.source, self.tick, self.lot, peers
        )


def _ns(stamps: pd.Series) -> np.ndarray:
    from trading_research.market_making.events import to_ns

    return to_ns(stamps)


def build_market(
    symbol: str,
    book: pd.DataFrame,
    trades: pd.DataFrame,
    grid_seconds: int,
    source: str,
    tick: float,
    lot: float,
    peers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> MarketData:
    """Grid the book, attach the prints each row owns and the peers' mids."""
    from trading_research.features.prints import bucket_prints

    depth = depth_of(book)
    rows = _grid(book, grid_seconds, depth)
    frame = derive(rows)
    prints = to_prints(trades)
    if not prints.empty:
        buckets = bucket_prints(prints, rows)
        for column in PRINT_COLUMNS:
            frame[column] = buckets[column].to_numpy()
    else:
        for column in PRINT_COLUMNS:
            frame[column] = 0.0
    labels = pd.DatetimeIndex(frame["timestamp"])
    for name, (peer_book, _) in (peers or {}).items():
        peer = derive(_grid(peer_book, grid_seconds, depth_of(peer_book)))
        series = pd.Series(peer["mid"].to_numpy(), index=pd.DatetimeIndex(peer["timestamp"]))
        # Carried forward only: a peer row's label is after what it holds.
        frame[f"{name}_mid"] = series.reindex(labels, method="ffill").to_numpy()
    return MarketData(
        symbol, frame, book.reset_index(drop=True), trades.reset_index(drop=True),
        grid_seconds, source, tick, lot, dict(peers or {}),
    )  # fmt: skip


def demo_market(
    *,
    steps: int = 72_000,
    seed: int = 20260101,
    peers: int = 3,
    grid_seconds: int = 1,
    signal_strength: float = 0.15,
    depth: int = 10,
) -> MarketData:
    """The synthetic demo market: ``steps`` snapshots of 200 ms, gridded to seconds."""
    from trading_research.data import synthetic

    config = synthetic.SyntheticConfig(
        n_steps=steps, seed=seed, depth=depth, signal_strength=signal_strength
    )
    data = synthetic.generate(config)
    others = {}
    for k in range(1, peers + 1):
        peer = synthetic.generate(config, seed=seed + 1000 * k, symbol=f"PEER{k}USDT")
        others[f"peer_{k}"] = (peer.book, peer.trades)
    source = (
        f"synthetic: data.synthetic.generate, {steps} steps of {config.step_ms} ms, seed {seed}, "
        f"signal_strength {signal_strength}, {peers} independent peers"
    )
    return build_market(
        config.symbol, data.book, data.trades, grid_seconds, source,
        config.tick_size, config.lot_size, others,
    )  # fmt: skip


def market_from_dataset(directory: Path | str, *, grid_seconds: int = 1) -> MarketData:
    """A dataset in the store's layout (``generate-demo-data`` writes one)."""
    from trading_research.data import store
    from trading_research.market_making.events import derive_lot, derive_tick

    book = store.read_book(directory)
    trades = store.read_trades(directory)
    manifest = store.read_manifest(directory)
    symbol = str(book["symbol"].iloc[0]) if "symbol" in book.columns else "INSTRUMENT"
    prices = book[[c for c in book.columns if "_price_" in c]].to_numpy().ravel()
    tick = derive_tick(prices)
    lot = derive_lot(trades["quantity"].to_numpy()) if len(trades) else 1.0
    kind = manifest.get("planes", {}).get("book", {}).get("source", "dataset")
    return build_market(symbol, book, trades, grid_seconds, f"{kind}: {directory}", tick, lot)


def matches(patterns: tuple[str, ...], columns: list[str]) -> dict[str, list[str]]:
    """The columns each declared input names (``fnmatch`` patterns allowed)."""
    return {p: [c for c in columns if fnmatch.fnmatchcase(c, p)] for p in patterns}


def mutate_after(frame: pd.DataFrame, row: int, *, seed: int = 0) -> pd.DataFrame:
    """``frame`` with every input after row ``row`` changed, rows up to it untouched.

    Prices (the level prices and the peers' mids) are moved by the repository's
    mutation, :func:`trading_research.validation.fold_leakage.mutate_after_cutoff`:
    one random walk starting ten per cent away, bids pushed down and asks up.
    Every other numeric input after the row is reversed in time and scaled by
    1.5, which keeps its sign and breaks both its alignment and its
    distribution (a quantile of the whole frame moves). ``mid``, ``log_mid`` and
    ``spread_bp`` are then recomputed from the mutated touch.
    """
    from trading_research.validation.fold_leakage import mutate_after_cutoff

    columns = [c for c in frame.columns if c != "timestamp" and c not in DERIVED]
    prices = [
        c for c in columns if "_price_" in c or (c.startswith("peer_") and c.endswith("_mid"))
    ]
    cutoff = pd.Timestamp(frame["timestamp"].iloc[row])
    out = mutate_after_cutoff(frame, cutoff, price_columns=prices, seed=seed)
    after = np.asarray(pd.DatetimeIndex(out["timestamp"]) > cutoff)
    for column in columns:
        if column in prices or not pd.api.types.is_numeric_dtype(out[column]):
            continue
        values = out[column].to_numpy().copy()
        values[after] = values[after][::-1] * 1.5
        out[column] = values
    return derive(out)
