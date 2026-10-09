"""Failed downloads: which ones say the venue has no data, and which say nothing.

A study that excludes every day whose download failed would exclude the busy
days first. The largest archives belong to the busiest days, a large archive is
the one most likely to lose its connection half way, and each such day dropped
as "unavailable" would tilt every result toward calm markets without anything
in the output showing it.

So a failure is classified, never just caught. **Only an HTTP 404 means the
venue has no data for that day** (:func:`is_missing_plane`); a timeout, a reset
connection, another status, a short body or a file that does not decompress
says something about the transfer and nothing about the day, and is retried
(:mod:`trading_research.data.fetch_log` decides when to stop).

Every downloader here raises its own error type ``from`` the underlying one, so
the status is read from the exception chain rather than from a message.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The one status that makes a missing plane.
MISSING_STATUS = 404

#: Bytes read per call while streaming a download to disk.
CHUNK_BYTES = 1 << 20

Opener = Callable[..., Any]


class IncompleteDownload(OSError):
    """The body ended before the length the server announced."""


def http_status(exc: BaseException) -> int | None:
    """The HTTP status anywhere in ``exc``'s chain of causes, or None."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, urllib.error.HTTPError):
            return int(current.code)
        current = current.__cause__ or current.__context__
    return None


def is_missing_plane(exc: BaseException) -> bool:
    """True only when the venue answered 404: it has no such file."""
    return http_status(exc) == MISSING_STATUS


@dataclass(frozen=True)
class Download:
    """One file on disk, with what is needed to judge and record it."""

    path: Path
    sha256: str
    size: int
    #: The length the server announced, when it announced one.
    expected: int | None

    @property
    def complete(self) -> bool:
        return self.expected is None or self.size == self.expected


def _announced_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def download_to(
    url: str,
    target: Path | str,
    *,
    timeout: float = 120.0,
    opener: Opener = urllib.request.urlopen,
) -> Download:
    """Stream ``url`` to ``target``, hashing it on the way.

    The body goes to a partial file named with the process id and is renamed
    only when complete, so an interrupted transfer is never mistaken for a
    file. Any failure deletes the partial file and re-raises. ``timeout``
    bounds each socket read, not the whole transfer: a large file does not fail
    by its size, a stalled connection does. A body shorter than its announced
    length raises :class:`IncompleteDownload`, which is a transfer failure.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(url, timeout=timeout) as response, partial.open("wb") as out:
            expected = _announced_length(response)
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if expected is not None and size != expected:
            raise IncompleteDownload(f"{url}: {size} of {expected} bytes")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return Download(path=path, sha256=digest.hexdigest(), size=size, expected=expected)


class _FileResponse:
    """A local file answering like a response: ``read``, ``headers``, a context."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self.headers = {"Content-Length": str(path.stat().st_size)}

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def __enter__(self) -> _FileResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()


class DirectoryOpener:
    """An opener that serves archives from a local directory by their file name.

    For replaying archives already on disk through the same path a download
    takes, and for tests: a URL whose last segment is not in ``directory``
    answers 404, as the venue would. Picklable, so it crosses into workers.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def __call__(self, url: str, timeout: float | None = None) -> _FileResponse:
        import io
        from email.message import Message

        del timeout  # a local read has no socket to time out
        path = self.directory / url.rsplit("/", 1)[-1]
        if not path.is_file():
            raise urllib.error.HTTPError(url, MISSING_STATUS, "Not Found", Message(), io.BytesIO())
        return _FileResponse(path)
