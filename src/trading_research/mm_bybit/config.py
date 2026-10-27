"""The registration as code: the YAML read, its placeholders, hashes and fee table.

``configs/mm_bybit_2021.yaml`` holds the registered values. The values set on W
and D are ``null`` lines commented "frozen by the D amendment"; putting ``null``
back into every one of them must give the null-placeholder sha256 the document
records on its latest line ``Null-placeholder sha256: `<hash>``` (an earlier
line is superseded). :func:`freeze_text` writes frozen values into the
placeholders and nothing else; the D amendment records the result's sha256 on a
line ``Frozen configuration sha256: `<hash>```.

The fee comes from :func:`fee_on` only, which refuses a day outside every
sourced interval: the study never substitutes a fee from another date.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from trading_research.market_making.prereg import (
    placeholder_lines,
    render_value,
    sha256_bytes,
    unfrozen_text,
)

CONFIG_PATH = Path("configs/mm_bybit_2021.yaml")
DOCUMENT_PATH = Path("docs/preregistration/mm_bybit_2021.md")

_NULL_LINE = re.compile(r"^Null-placeholder sha256: `([0-9a-f]{64})`\s*$", re.M)
_FROZEN_LINE = re.compile(r"^Frozen configuration sha256: `([0-9a-f]{64})`\s*$", re.M)


class RegistrationError(ValueError):
    """The configuration does not match its registration."""


class FeeNotSourced(LookupError):
    """A day outside every sourced fee interval."""


@dataclass(frozen=True)
class Fees:
    maker_bp: float
    taker_bp: float


def null_placeholder_sha256(document: str) -> str:
    """The latest ``Null-placeholder sha256`` line of the document."""
    found = _NULL_LINE.findall(document)
    if not found:
        raise RegistrationError("the document records no null-placeholder sha256")
    return str(found[-1])


def frozen_sha256(document: str) -> str | None:
    """The latest ``Frozen configuration sha256`` line, or None before the amendment."""
    found = _FROZEN_LINE.findall(document)
    return str(found[-1]) if found else None


def check_registered(config_text: str, document: str) -> None:
    """Raise unless the configuration with its placeholders reset hashes as registered."""
    expected = null_placeholder_sha256(document)
    actual = sha256_bytes(unfrozen_text(config_text).encode())
    if actual != expected:
        raise RegistrationError(
            f"configuration with null placeholders hashes to {actual}, registered {expected}"
        )


def is_frozen(config_text: str) -> bool:
    """True when no placeholder is ``null``."""
    values = yaml.safe_load(config_text)["frozen"]
    return all(values[key] is not None for (_, key) in placeholder_lines(config_text))


def freeze_text(config_text: str, values: Mapping[str, Any]) -> str:
    """Write every placeholder's value (keyed by name) into its line, nothing else."""
    where = {path[-1]: line for path, line in placeholder_lines(config_text).items()}
    if set(values) != set(where):
        raise RegistrationError(
            f"freezing needs every placeholder once: missing {sorted(set(where) - set(values))},"
            f" unknown {sorted(set(values) - set(where))}"
        )
    lines = config_text.split("\n")
    for key, number in where.items():
        head, rest = lines[number].split(": ", 1)
        value, tail = rest.split("  # ", 1)
        if value.strip() != "null":
            raise RegistrationError(f"{key} is already frozen")
        lines[number] = f"{head}: {render_value(values[key])}  # {tail}"
    frozen = "\n".join(lines)
    if unfrozen_text(frozen) != unfrozen_text(config_text):
        raise RegistrationError("freezing changed something other than the placeholders")
    return frozen


def load(root: Path | str = ".", *, check: bool = True) -> dict[str, Any]:
    """The configuration as a dict, checked against the document's hash."""
    base = Path(root)
    text = (base / CONFIG_PATH).read_text()
    if check:
        check_registered(text, (base / DOCUMENT_PATH).read_text())
    loaded: dict[str, Any] = yaml.safe_load(text)
    return loaded


def fee_on(config: Mapping[str, Any], day: date) -> Fees:
    """The fee every account paid on ``day``, from the sourced table only."""
    for interval in config["fees"]["intervals"]:
        if interval["start"] <= day < interval["end_exclusive"]:
            return Fees(float(interval["maker_bp"]), float(interval["taker_bp"]))
    raise FeeNotSourced(f"{day}: no sourced fee interval")
