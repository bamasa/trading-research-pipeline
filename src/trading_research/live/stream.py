"""Public market-data streams turned into the repository's own rows and prints.

Two venues, two markets each, public and unauthenticated endpoints only. There
is no key, no signature and no private or order endpoint anywhere in this
package: the clients open the URLs below, send a subscription, and read.

=========  =======  =============================================  =================================
venue      market   book                                           prints
=========  =======  =============================================  =================================
bybit      spot     ``orderbook.200``: snapshot, then deltas        ``publicTrade``, taker side ``S``
bybit      linear   ``orderbook.200`` (the archive's ``ob200``)     ``publicTrade``
binance    spot     ``depth10@100ms``: ten levels every 100 ms      ``aggTrade``, ``m`` = buyer maker
binance    linear   USD-M ``depth10@100ms`` (``/public`` path)      USD-M ``aggTrade`` (``/market``)
=========  =======  =============================================  =================================

**Bybit's book is the archive's book.** The venue's archive is this stream
written to a file, message for message, so the parser here keeps the replay of
:func:`trading_research.data.bybit.reconstruct` exactly: its
:class:`~trading_research.data.bybit.BookState`, a delta whose update id is not
the previous one plus one counted as a gap, and a row emitted at the first
update of each 100 ms interval, stamped with that update's ``ts``, only when
both sides hold ten levels. A test feeds the same messages to both and
compares the frames. On a gap the stream also resubscribes, which makes the
venue send a fresh snapshot.

**Binance's partial depth is a snapshot per push**, ten levels every 100 ms.
USD-M pushes carry the transaction time ``T`` and the previous final update id
``pu``, checked against the last push's ``u``; spot pushes carry only
``lastUpdateId`` and no time at all, so a spot row is stamped at local receipt
(the page says so) and its check is that the id never goes backwards. The grid
rule is the same as Bybit's: at most one row per 100 ms interval, its first.

**Prints.** Bybit's ``S`` is the taker's side; Binance's ``m`` is True when the
buyer was the maker, so the seller crossed. Binance aggregate ids must run on
by one; a jump is a gap. Bybit's ids are strings with no order, so its prints
are checked only for the venue's cross sequence ``seq`` never going backwards.
"""

from __future__ import annotations

import json
import select
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from trading_research.data.bybit import BookState
from trading_research.live.events import DEPTH, GRID_MS, BookRow, Event, Gap, Print
from trading_research.live.ws import WebSocket, WebSocketError

try:  # pragma: no cover - the fallback is exercised only without the extra
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    from json import loads as _loads

VENUES = ("bybit", "binance")
MARKETS = ("spot", "linear")
MS = 1_000_000


@dataclass(frozen=True)
class Connection:
    """One websocket: its URL, what to send after connecting, and how to keep it alive."""

    name: str
    url: str
    subscribe: tuple[str, ...] = ()
    ping_text: str | None = None
    ping_every_s: float = 20.0


def connections(venue: str, market: str, symbol: str) -> tuple[Connection, ...]:
    """The public connections for one instrument; every URL here is a market-data stream."""
    if venue not in VENUES or market not in MARKETS:
        raise ValueError(
            f"unknown venue/market {venue}/{market}; venues {VENUES}, markets {MARKETS}"
        )
    if venue == "bybit":
        topics = [f"orderbook.200.{symbol}", f"publicTrade.{symbol}"]
        return (
            Connection(
                "bybit",
                f"wss://stream.bybit.com/v5/public/{market}",
                (json.dumps({"op": "subscribe", "args": topics}),),
                json.dumps({"op": "ping"}),
            ),
        )
    lower = symbol.lower()
    if market == "spot":
        url = (
            "wss://stream.binance.com:9443/stream?streams="
            f"{lower}@depth{DEPTH}@{GRID_MS}ms/{lower}@aggTrade"
        )
        return (Connection("binance", url),)
    return (
        Connection(
            "binance-book",
            f"wss://fstream.binance.com/public/stream?streams={lower}@depth{DEPTH}@{GRID_MS}ms",
        ),
        Connection(
            "binance-prints", f"wss://fstream.binance.com/market/stream?streams={lower}@aggTrade"
        ),
    )


def source_tag(venue: str, market: str) -> str:
    """The ``source`` column of rows recorded from the live stream."""
    if venue == "bybit":
        return f"bybit-{market}-ob200-live"
    return f"binance-{market}-depth{DEPTH}-live"


# ---------------------------------------------------------------------------
# Parsers: one message in, events out. Shared by the live stream and replays.
# ---------------------------------------------------------------------------


@dataclass
class BybitParser:
    """Bybit ``orderbook`` and ``publicTrade`` messages, replayed as the archive is."""

    depth: int = DEPTH
    grid_ms: int = GRID_MS
    state: BookState = field(default_factory=BookState)
    bucket: int | None = None
    last_trade_seq: int | None = None
    #: Set on a sequence gap: the stream resubscribes to get a fresh snapshot.
    needs_resync: bool = False

    def reset(self) -> None:
        """Forget the book after a reconnect; rows resume at the next snapshot."""
        self.state = BookState()
        self.last_trade_seq = None

    def parse(self, text: str | bytes, recv_ns: int) -> list[Event]:
        message: dict[str, Any] = _loads(text)
        topic = str(message.get("topic", ""))
        if topic.startswith("publicTrade"):
            return self._prints(message, recv_ns)
        if topic.startswith("orderbook") or (not topic and "type" in message and "data" in message):
            return self._book(message, recv_ns)
        return []

    def book_message(self, message: dict[str, Any], recv_ns: int) -> list[Event]:
        """An already decoded book message (the archive replay's path)."""
        return self._book(message, recv_ns)

    def _book(self, message: dict[str, Any], recv_ns: int) -> list[Event]:
        # The loop body of data.bybit.reconstruct, one message at a time.
        kind = message.get("type")
        data = message.get("data", {})
        stamp = int(message["ts"])
        update = data.get("u")
        out: list[Event] = []
        state = self.state
        if kind == "snapshot":
            gaps = state.gaps
            state = self.state = BookState()
            state.gaps = gaps
            state.apply("b", data.get("b", []))
            state.apply("a", data.get("a", []))
            state.last_update = update
        elif kind == "delta":
            if (
                state.last_update is not None
                and update is not None
                and update != state.last_update + 1
            ):
                state.gaps += 1
                self.needs_resync = True
                out.append(
                    Gap(
                        stamp * MS,
                        recv_ns,
                        "sequence",
                        f"book update {update} after {state.last_update}",
                    )
                )
            state.apply("b", data.get("b", []))
            state.apply("a", data.get("a", []))
            state.last_update = update
        else:
            return out
        if not state.ready:
            return out
        current = stamp // self.grid_ms
        if self.bucket is not None and current == self.bucket:
            return out
        self.bucket = current
        bids, asks = state.top(self.depth)
        if len(bids) < self.depth or len(asks) < self.depth:
            return out
        out.append(
            BookRow(
                stamp * MS,
                recv_ns,
                int(update) if update is not None else 0,
                tuple(p for p, _ in bids),
                tuple(s for _, s in bids),
                tuple(p for p, _ in asks),
                tuple(s for _, s in asks),
            )
        )
        return out

    def _prints(self, message: dict[str, Any], recv_ns: int) -> list[Event]:
        out: list[Event] = []
        for trade in message.get("data", []):
            seq = trade.get("seq")
            if seq is not None:
                seq = int(seq)
                if self.last_trade_seq is not None and seq < self.last_trade_seq:
                    out.append(
                        Gap(
                            int(trade["T"]) * MS,
                            recv_ns,
                            "trade_sequence",
                            f"print sequence {seq} after {self.last_trade_seq}",
                        )
                    )
                self.last_trade_seq = seq
            out.append(
                Print(
                    int(trade["T"]) * MS,
                    recv_ns,
                    float(trade["p"]),
                    float(trade["v"]),
                    1 if trade["S"] == "Buy" else -1,
                    str(trade.get("i", "")),
                    bool(trade.get("RPI", False)),
                )
            )
        return out


@dataclass
class BinanceParser:
    """Binance partial-depth and aggregate-trade messages (combined-stream envelopes)."""

    market: str = "spot"
    depth: int = DEPTH
    grid_ms: int = GRID_MS
    bucket: int | None = None
    last_book_id: int | None = None
    last_trade_id: int | None = None
    needs_resync: bool = False

    def reset(self) -> None:
        self.last_book_id = None
        self.last_trade_id = None

    def parse(self, text: str | bytes, recv_ns: int) -> list[Event]:
        message: dict[str, Any] = _loads(text)
        data = message.get("data", message)
        stream = str(message.get("stream", ""))
        if data.get("e") == "aggTrade" or stream.endswith("@aggTrade"):
            return self._print(data, recv_ns)
        if "bids" in data or "b" in data:
            return self._book(data, recv_ns)
        return []

    def _book(self, data: dict[str, Any], recv_ns: int) -> list[Event]:
        out: list[Event] = []
        if self.market == "spot":
            update = int(data["lastUpdateId"])
            stamp_ns = recv_ns
            if self.last_book_id is not None and update < self.last_book_id:
                out.append(
                    Gap(
                        stamp_ns,
                        recv_ns,
                        "sequence",
                        f"depth id {update} after {self.last_book_id}",
                    )
                )
            bids_raw, asks_raw = data["bids"], data["asks"]
        else:
            update = int(data["u"])
            stamp_ns = int(data["T"]) * MS
            previous = data.get("pu")
            if (
                self.last_book_id is not None
                and previous is not None
                and int(previous) != self.last_book_id
            ):
                out.append(
                    Gap(
                        stamp_ns,
                        recv_ns,
                        "sequence",
                        f"depth pu {previous} after u {self.last_book_id}",
                    )
                )
            bids_raw, asks_raw = data["b"], data["a"]
        self.last_book_id = update
        current = (stamp_ns // MS) // self.grid_ms
        if self.bucket is not None and current == self.bucket:
            return out
        self.bucket = current
        bids = [(float(p), float(q)) for p, q in bids_raw if float(q) > 0]
        asks = [(float(p), float(q)) for p, q in asks_raw if float(q) > 0]
        if len(bids) < self.depth or len(asks) < self.depth:
            return out
        bids, asks = bids[: self.depth], asks[: self.depth]
        out.append(
            BookRow(
                stamp_ns,
                recv_ns,
                update,
                tuple(p for p, _ in bids),
                tuple(q for _, q in bids),
                tuple(p for p, _ in asks),
                tuple(q for _, q in asks),
            )
        )
        return out

    def _print(self, data: dict[str, Any], recv_ns: int) -> list[Event]:
        out: list[Event] = []
        trade_id = int(data["a"])
        stamp_ns = int(data["T"]) * MS
        if self.last_trade_id is not None and trade_id != self.last_trade_id + 1:
            out.append(
                Gap(
                    stamp_ns,
                    recv_ns,
                    "trade_sequence",
                    f"aggregate id {trade_id} after {self.last_trade_id}",
                )
            )
        self.last_trade_id = trade_id
        out.append(
            Print(
                stamp_ns,
                recv_ns,
                float(data["p"]),
                float(data["q"]),
                -1 if data["m"] else 1,
                str(trade_id),
            )
        )
        return out


Parser = BybitParser | BinanceParser


def make_parser(venue: str, market: str) -> Parser:
    if venue == "bybit":
        return BybitParser()
    if venue == "binance":
        return BinanceParser(market=market)
    raise ValueError(f"unknown venue {venue!r}")


# ---------------------------------------------------------------------------
# The live client
# ---------------------------------------------------------------------------

RawHook = Callable[[str, int, str], None]


@dataclass
class StreamStats:
    messages: int = 0
    reconnects: int = 0
    resyncs: int = 0
    bytes_received: int = 0
    first_message_ns: int = 0
    last_message_ns: int = 0


@dataclass
class LiveStream:
    """The public stream of one instrument as events, with reconnects and a gap log.

    :meth:`events` yields events in arrival order and ``None`` at least every
    ``tick_s`` so the caller can keep time when the feed is quiet. ``on_raw``
    sees every data message with its receipt time and connection, which is what
    the recorder writes and a replay feeds back through the same parser.
    """

    venue: str
    market: str
    symbol: str
    on_raw: RawHook | None = None
    tick_s: float = 0.2
    max_reconnects: int = 20
    stats: StreamStats = field(default_factory=StreamStats)
    gaps: list[Gap] = field(default_factory=list)
    _sockets: dict[str, WebSocket] = field(default_factory=dict, repr=False)
    _last_ping: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        self.plan = connections(self.venue, self.market, self.symbol)
        self.parser = make_parser(self.venue, self.market)

    def _open(self, conn: Connection) -> None:
        ws = WebSocket(conn.url)
        ws.connect()
        for text in conn.subscribe:
            ws.send_text(text)
        self._sockets[conn.name] = ws
        self._last_ping[conn.name] = time.monotonic()

    def _reopen_all(self, reason: str) -> Gap:
        now = time.time_ns()
        for ws in self._sockets.values():
            ws.close()
        self._sockets.clear()
        self.parser.reset()
        self.stats.reconnects += 1
        gap = Gap(now, now, "reconnect", reason)
        self.gaps.append(gap)
        delay = 1.0
        while True:
            if self.stats.reconnects > self.max_reconnects:
                raise WebSocketError(f"gave up after {self.max_reconnects} reconnects: {reason}")
            try:
                for conn in self.plan:
                    self._open(conn)
                return gap
            except (OSError, WebSocketError) as exc:
                reason = f"{reason}; reconnect failed: {exc}"
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                self.stats.reconnects += 1

    def _resync(self) -> None:
        """Ask Bybit for a fresh book snapshot after a sequence gap."""
        self.parser.needs_resync = False
        if self.venue != "bybit":
            return
        ws = self._sockets.get("bybit")
        if ws is None:
            return
        topic = f"orderbook.200.{self.symbol}"
        ws.send_text(json.dumps({"op": "unsubscribe", "args": [topic]}))
        ws.send_text(json.dumps({"op": "subscribe", "args": [topic]}))
        self.stats.resyncs += 1

    def close(self) -> None:
        for ws in self._sockets.values():
            ws.close()
        self._sockets.clear()

    def events(self, until_ns: int | None = None) -> Iterator[Event | None]:
        for conn in self.plan:
            self._open(conn)
        try:
            while until_ns is None or time.time_ns() < until_ns:
                ready = [name for name, ws in self._sockets.items() if ws.pending()]
                if not ready:
                    by_fd = {ws.fileno(): name for name, ws in self._sockets.items()}
                    readable, _, _ = select.select(list(by_fd), [], [], self.tick_s)
                    ready = [by_fd[fd] for fd in readable]
                try:
                    for name in ready:
                        self._sockets[name].pump(0.0)
                    for name, ws in list(self._sockets.items()):
                        while (text := ws.next_message()) is not None:
                            yield from self._handle(name, text)
                    self._keepalive()
                except (OSError, WebSocketError) as exc:
                    yield self._reopen_all(f"{type(exc).__name__}: {exc}")
                yield None
        finally:
            self.close()

    def _handle(self, name: str, text: str) -> Iterator[Event]:
        recv_ns = time.time_ns()
        if '"op"' in text[:40] or '"success"' in text[:40]:
            return  # subscription acknowledgements and pongs
        self.stats.messages += 1
        self.stats.first_message_ns = self.stats.first_message_ns or recv_ns
        self.stats.last_message_ns = recv_ns
        self.stats.bytes_received += len(text)
        if self.on_raw is not None:
            self.on_raw(name, recv_ns, text)
        for event in self.parser.parse(text, recv_ns):
            if isinstance(event, Gap):
                self.gaps.append(event)
            yield event
        if self.parser.needs_resync:
            self._resync()

    def _keepalive(self) -> None:
        now = time.monotonic()
        for conn in self.plan:
            if conn.ping_text is None or conn.name not in self._sockets:
                continue
            if now - self._last_ping[conn.name] >= conn.ping_every_s:
                self._sockets[conn.name].send_text(conn.ping_text)
                self._last_ping[conn.name] = now
