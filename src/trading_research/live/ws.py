"""A minimal websocket client on the standard library, for public market data only.

No websocket library is a dependency of this repository, and the paper-trading
step needs one thing from one: read text frames from a public, unauthenticated
market-data stream and answer the server's pings. That is a small part of RFC
6455, so it is written here rather than added as a dependency:

* the opening handshake over TLS, with the ``Sec-WebSocket-Accept`` check;
* text, binary and continuation frames, reassembled into one message;
* a ping answered with a pong carrying the same payload, a close answered and
  the connection ended;
* client frames masked, as the protocol requires.

Reads never lose data on a timeout: bytes are kept in the client's own buffer
and a frame is decoded only once all of it has arrived, so :meth:`recv` can be
called with a short timeout in a loop that also keeps time.

Nothing here can authenticate, sign or send an order: the client only opens a
URL it is given and sends the subscription and keep-alive text the stream
module writes.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import socket
import ssl
import struct
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION, OP_TEXT, OP_BINARY = 0x0, 0x1, 0x2
OP_CLOSE, OP_PING, OP_PONG = 0x8, 0x9, 0xA


class WebSocketError(ConnectionError):
    """The handshake failed, the server closed, or a frame was malformed."""


@dataclass
class WebSocket:
    """One client connection. ``connect`` it, ``send_text``, ``recv`` with a timeout, ``close``."""

    url: str
    connect_timeout_s: float = 10.0
    max_message_bytes: int = 16 * 1024 * 1024
    #: Bytes read from the network and not yet decoded.
    _buffer: bytearray = field(default_factory=bytearray, repr=False)
    _sock: socket.socket | None = field(default=None, repr=False)
    _fragments: list[bytes] = field(default_factory=list, repr=False)
    _fragment_op: int = field(default=OP_TEXT, repr=False)
    closed: bool = False
    #: Wall-clock nanoseconds of the last frame of any kind received.
    last_frame_ns: int = 0
    bytes_received: int = 0

    def connect(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("wss", "ws"):
            raise WebSocketError(f"not a websocket url: {self.url!r}")
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "wss" else 80)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        raw = socket.create_connection((host, port), timeout=self.connect_timeout_s)
        sock: socket.socket = raw
        if parts.scheme == "wss":
            sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: trading-research-paper/1\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        head = self._read_headers(sock)
        status = head.split("\r\n", 1)[0]
        if " 101 " not in f"{status} ":
            sock.close()
            raise WebSocketError(f"handshake refused by {host}: {status!r}")
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode()
        headers = {
            line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
            for line in head.split("\r\n")[1:]
            if ":" in line
        }
        if headers.get("sec-websocket-accept") != expected:
            sock.close()
            raise WebSocketError(f"{host} answered the handshake with the wrong accept key")
        self._sock = sock
        self.closed = False
        self.last_frame_ns = time.time_ns()

    def _read_headers(self, sock: socket.socket) -> str:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise WebSocketError("the server closed during the handshake")
            data += chunk
            if len(data) > 65536:
                raise WebSocketError("handshake response too long")
        head, _, rest = bytes(data).partition(b"\r\n\r\n")
        self._buffer += rest
        return head.decode("latin-1")

    # -- sending -------------------------------------------------------------

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._sock is None or self.closed:
            raise WebSocketError("not connected")
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    # -- receiving -----------------------------------------------------------

    def _frame(self) -> tuple[bool, int, bytes] | None:
        """Decode one whole frame from the buffer, or None if it has not all arrived."""
        buf = self._buffer
        if len(buf) < 2:
            return None
        fin = bool(buf[0] & 0x80)
        opcode = buf[0] & 0x0F
        masked = bool(buf[1] & 0x80)
        length = buf[1] & 0x7F
        offset = 2
        if length == 126:
            if len(buf) < 4:
                return None
            length = struct.unpack("!H", bytes(buf[2:4]))[0]
            offset = 4
        elif length == 127:
            if len(buf) < 10:
                return None
            length = struct.unpack("!Q", bytes(buf[2:10]))[0]
            offset = 10
        if length > self.max_message_bytes:
            raise WebSocketError(f"a frame of {length} bytes exceeds the limit")
        mask = b""
        if masked:
            if len(buf) < offset + 4:
                return None
            mask = bytes(buf[offset : offset + 4])
            offset += 4
        if len(buf) < offset + length:
            return None
        payload = bytes(buf[offset : offset + length])
        del buf[: offset + length]
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return fin, opcode, payload

    def fileno(self) -> int:
        if self._sock is None:
            raise WebSocketError("not connected")
        return self._sock.fileno()

    def pending(self) -> bool:
        """Whether bytes are already buffered (here or inside TLS) and need no wait."""
        if self._sock is None:
            return False
        tls = self._sock.pending() if isinstance(self._sock, ssl.SSLSocket) else 0
        return bool(self._buffer) or tls > 0

    def pump(self, timeout_s: float = 0.0) -> None:
        """Read once from the network into the buffer, waiting at most ``timeout_s``."""
        if self._sock is None or self.closed:
            raise WebSocketError("not connected")
        self._sock.settimeout(max(timeout_s, 0.0) or 0.000001)
        try:
            chunk = self._sock.recv(65536)
        except (TimeoutError, ssl.SSLWantReadError):
            return
        if not chunk:
            self.closed = True
            raise WebSocketError("the server closed the connection")
        self.bytes_received += len(chunk)
        self._buffer += chunk

    def next_message(self) -> str | None:
        """The next whole message already buffered, or None; never reads the network."""
        while True:
            frame = self._frame()
            if frame is None:
                return None
            self.last_frame_ns = time.time_ns()
            message = self._handle(*frame)
            if message is not None:
                return message

    def recv(self, timeout_s: float = 1.0) -> str | None:
        """The next whole text message, or None when none arrived within ``timeout_s``.

        Pings are answered here and never returned. A close from the server
        raises :class:`WebSocketError` after answering it.
        """
        if self._sock is None or self.closed:
            raise WebSocketError("not connected")
        deadline = time.monotonic() + timeout_s
        while True:
            frame = self._frame()
            if frame is not None:
                fin, opcode, payload = frame
                self.last_frame_ns = time.time_ns()
                message = self._handle(fin, opcode, payload)
                if message is not None:
                    return message
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(65536)
            except TimeoutError:
                return None
            except ssl.SSLWantReadError:
                continue
            if not chunk:
                self.closed = True
                raise WebSocketError("the server closed the connection")
            self.bytes_received += len(chunk)
            self._buffer += chunk

    def _handle(self, fin: bool, opcode: int, payload: bytes) -> str | None:
        if opcode == OP_PING:
            self._send_frame(OP_PONG, payload)
            return None
        if opcode == OP_PONG:
            return None
        if opcode == OP_CLOSE:
            code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1005
            reason = payload[2:].decode("utf-8", errors="replace")
            with contextlib.suppress(OSError):
                self._send_frame(OP_CLOSE, payload[:2])
            self.close()
            raise WebSocketError(f"the server closed the stream: {code} {reason}".strip())
        if opcode in (OP_TEXT, OP_BINARY):
            if fin:
                return payload.decode("utf-8")
            self._fragments = [payload]
            self._fragment_op = opcode
            return None
        if opcode == OP_CONTINUATION:
            self._fragments.append(payload)
            if sum(len(part) for part in self._fragments) > self.max_message_bytes:
                raise WebSocketError("a fragmented message exceeds the limit")
            if not fin:
                return None
            whole = b"".join(self._fragments)
            self._fragments = []
            return whole.decode("utf-8")
        raise WebSocketError(f"unknown opcode {opcode}")

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._sock = None
        self.closed = True
