"""The run directory's audit: what in it looks like a secret, and where.

Every file of a run directory is scanned for the usual shapes of a credential:
a provider key (``sk-...``), an assignment to an API key, a token or a secret,
a bearer header, and long hex or base64 strings. A sha256 (64 hex digits) and a
git commit (40) are the run's own records and are not findings. A finding names
the file, the line and the pattern, never the value, so the audit itself can be
logged and shown.
"""

from __future__ import annotations

import re
from pathlib import Path

#: A name may stand after an underscore (``ANTHROPIC_API_KEY``), not inside a word.
_NAME = r"(?i)(?<![A-Za-z0-9])"
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "provider key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}"),
    "api key": re.compile(_NAME + r"api[_-]?key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-.]{8,}"),
    "token": re.compile(_NAME + r"(?:[a-z]+[_-])?token[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-.]{16,}"),
    "secret": re.compile(
        _NAME + r"(?:[a-z]+[_-])?secret(?:[_-]?key)?[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-.]{8,}"
    ),
    "password": re.compile(_NAME + r"pass(?:word|wd)[\"']?\s*[:=]\s*[\"']?\S{6,}"),
    "bearer": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
HEX = re.compile(r"\b[0-9a-fA-F]{32,}\b")
BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")
RECORDS = (40, 64)  # a git commit, a sha256


def _long_strings(line: str) -> list[str]:
    found = [f"hex of {len(m)}" for m in HEX.findall(line) if len(m) not in RECORDS]
    for match in BASE64.findall(line):
        body = match.rstrip("=")
        mixed = any(c.isupper() for c in body) and any(c.islower() for c in body)
        if mixed and any(c.isdigit() for c in body) and not HEX.fullmatch(body):
            found.append(f"base64 of {len(match)}")
    return found


def scan_text(text: str) -> list[tuple[int, str]]:
    """Each (line number, pattern name) in ``text`` that looks like a secret."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found += [(number, name) for name, rx in SECRET_PATTERNS.items() if rx.search(line)]
        found += [(number, name) for name in _long_strings(line)]
    return found


def scan(run: Path) -> list[str]:
    """Every finding in the run directory, as ``<file>:<line>: <pattern>``."""
    found: list[str] = []
    for path in sorted(p for p in run.rglob("*") if p.is_file()):
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            found.append(f"{path.relative_to(run)}:0: not text")
            continue
        rel = path.relative_to(run).as_posix()
        found += [f"{rel}:{line}: {name}" for line, name in scan_text(text)]
    return found
