"""The guard a live run passes before it opens a connection.

The paper step uses public market data only. There is no setting for a key,
and none is ever read; the guard keeps it that way if a future option or a
configuration file starts to carry one. A run is refused when:

* any value of its configuration names something that looks like an exchange
  credential: an environment variable such as ``BYBIT_API_KEY`` or
  ``BINANCE_SECRET``, ``$...``/``${...}`` references to one, or a file such as
  ``.env``, ``credentials.json``, ``*.pem`` or ``*secret*``;
* a file the configuration points at contains a line that looks like a key
  assignment (``api_key = ...``, ``"secret": ...``);
* a URL the stream would open is not one of the public market-data endpoints
  (a private websocket, a ``listenKey`` user stream, a REST order path).

The refusal names what matched. It never prints a value.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Words that make a name look like a credential when they sit beside an exchange's name
#: or beside "api".
_KEY_NAME = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret[_-]?key|private[_-]?key|passphrase|"
    r"(bybit|binance)[_-]?(key|secret|token)|access[_-]?token)",
    re.IGNORECASE,
)
_ENV_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
_KEY_FILE = re.compile(
    r"(^|/)(\.env(\..*)?|credentials(\.json|\.ya?ml)?|.*secret.*|.*api[_-]?key.*|.*\.pem|.*\.key)$",
    re.IGNORECASE,
)
_KEY_LINE = re.compile(
    r"""["']?(api[_-]?key|api[_-]?secret|secret|passphrase|private[_-]?key)["']?\s*[:=]""",
    re.IGNORECASE,
)
#: The only hosts and paths a paper run may open.
PUBLIC_STREAMS = (
    re.compile(r"^wss://stream\.bybit\.com/v5/public/(spot|linear)$"),
    re.compile(r"^wss://stream\.binance\.com:9443/stream\?streams=[A-Za-z0-9@/]+$"),
    re.compile(r"^wss://fstream\.binance\.com/(public|market)/stream\?streams=[A-Za-z0-9@/]+$"),
)


class KeyGuardRefusal(PermissionError):
    """The run's configuration references something that looks like an exchange key."""


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Path):
        yield str(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)


def check_config(config: Mapping[str, Any], urls: Iterable[str] = ()) -> list[str]:
    """What in ``config`` and ``urls`` looks like a key or a non-public endpoint (empty: none)."""
    found: list[str] = []
    for text in _strings(config):
        if _KEY_NAME.search(text):
            found.append(f"a value names a credential ({_KEY_NAME.search(text).group(0)!r})")  # type: ignore[union-attr]
        for name in _ENV_REF.findall(text):
            if _KEY_NAME.search(name) or name.upper().endswith(("_KEY", "_SECRET", "_TOKEN")):
                found.append(f"a value references the environment variable {name}")
        if _KEY_FILE.search(text.strip()):
            found.append(
                f"a value names a file that looks like a credential store ({Path(text).name})"
            )
        candidate = Path(text)
        if len(text) < 4096 and candidate.is_file() and candidate.stat().st_size < 1_000_000:
            try:
                body = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                body = ""
            if _KEY_LINE.search(body):
                found.append(
                    f"the file {candidate.name} holds a line that looks like a key assignment"
                )
    for url in urls:
        if not any(pattern.match(url) for pattern in PUBLIC_STREAMS):
            found.append(
                f"the stream would open a URL that is not a public market-data stream: {url}"
            )
    return found


def refuse_keys(config: Mapping[str, Any], urls: Iterable[str] = ()) -> None:
    """Raise :class:`KeyGuardRefusal` if :func:`check_config` finds anything."""
    found = check_config(config, urls)
    if found:
        raise KeyGuardRefusal(
            "a paper run uses public market data only and refuses to start: " + "; ".join(found)
        )
