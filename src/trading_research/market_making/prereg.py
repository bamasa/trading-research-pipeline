"""The pre-registration as code: its values, its blocks, and the held-out ledger.

The market-making study was registered before any of its code existed, in
``docs/preregistration/market_making.md`` and ``configs/mm_prereg.yaml``. This
module is how the code is held to that registration.

The YAML
--------
:class:`PreRegistration` loads the YAML and validates it against the schema it
was registered with: every section, every grid, every fixed setting. The values
chosen on the development block were left as ``null`` placeholders, each
commented "frozen by the D amendment", and the amendment writes them in place
(:func:`freeze_text`). The protocol says nothing else may change, and the loader
checks exactly that: putting ``null`` back into every placeholder must give a
file whose sha256 is the one registered (:data:`REGISTERED_SHA256`). A file in
which anything else moved — a grid value, a block date, a comment — fails.

The blocks, and who may read them
---------------------------------
:data:`BLOCKS` are the registered date ranges. Every function in the study that
reads market data for a day — the admission screen, the reversion tape, the
regime flags, the simulation driver — first passes the day through
:meth:`Access.require`. :func:`development_access` permits block D and nothing
else. The only way to obtain an :class:`Access` that permits H or F is
:meth:`HeldOutLedger.open`, which refuses unless:

* the amendment that freezes the configuration exists and records a sha256;
* ``configs/mm_prereg.yaml`` on disk has that sha256, is fully frozen, and
  differs from the registered file only in its placeholders;
* the YAML and the document are committed, and the working tree is clean;
* the block has not been read before (a second read needs ``force=True``, and
  everything it produces is stamped ``second read``);
* for F, the ledger already records H's read and is itself committed, so H's
  results are committed before F is touched.

The first read appends an entry — block, commit, config sha256, UTC time — to
``experiments/results/mm_heldout_ledger.json`` before the access is returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

#: Where the two halves of the registration live, relative to the repository.
CONFIG_PATH = Path("configs/mm_prereg.yaml")
DOCUMENT_PATH = Path("docs/preregistration/market_making.md")
LEDGER_PATH = Path("experiments/results/mm_heldout_ledger.json")

#: The sha256 of ``configs/mm_prereg.yaml`` as registered (commit ``cb4ae4c``),
#: with every placeholder ``null``. Amendment 1 left it unchanged.
REGISTERED_SHA256 = "27471cd62ef4bbc67a29467bcff0ea5fda4ddd2c7d00413ca05a9541eac9b272"

#: The registered blocks, inclusive, UTC dates.
BLOCKS: dict[str, tuple[date, date]] = {
    "D": (date(2024, 2, 1), date(2024, 2, 25)),
    "H": (date(2024, 2, 26), date(2024, 3, 9)),
    "buffer": (date(2024, 3, 10), date(2024, 3, 11)),
    "F": (date(2024, 3, 12), date(2024, 4, 7)),
    "P": (date(2024, 4, 8), date(2024, 5, 5)),
}

#: The comment every placeholder line carries.
PLACEHOLDER_MARK = "frozen by the D amendment"

#: Every placeholder the amendment fills, as a path of keys.
PLACEHOLDERS: tuple[tuple[str, ...], ...] = (
    ("admission", "table"),
    ("simulator", "clip", "notional_usdt"),
    ("simulator", "volatility", "sigma_ref"),
    ("strategies", "S1", "chosen", "skew_bp"),
    ("strategies", "S1", "chosen", "k"),
    ("strategies", "S1", "chosen", "min_edge_bp"),
    ("strategies", "S1", "chosen", "soft_limit_clips"),
    ("strategies", "S2", "chosen", "skew_bp"),
    ("strategies", "S2", "chosen", "k"),
    ("strategies", "S2", "chosen", "min_edge_bp"),
    ("strategies", "S2", "chosen", "soft_limit_clips"),
    ("strategies", "S2", "chosen", "m_ticks"),
    ("strategies", "S3", "chosen", "guards"),
    ("strategies", "S3", "chosen", "action"),
    ("strategies", "S3", "chosen", "guard_window_min"),
    ("strategies", "S4", "theta_bp"),
    ("strategies", "S4", "beta"),
    ("strategies", "S4", "chosen", "lambda"),
    ("strategies", "S4", "chosen", "one_sided"),
    ("d_search", "neighbourhood", "S1"),
    ("d_search", "neighbourhood", "S2"),
    ("metrics", "minimum_detectable_effect", "sigma_d_usd_day"),
    ("metrics", "minimum_detectable_effect", "value_usd_day"),
    ("hypotheses", "H2.3", "gate_threshold"),
    ("hypotheses", "H3", "flag_rate_on_d_per_day"),
    ("amendment", "date"),
)

#: The marker line in the document that records the frozen file's sha256.
FROZEN_HASH_PATTERN = re.compile(r"Frozen configuration sha256: `([0-9a-f]{64})`")

#: Top-level sections of the registered YAML, in order.
SECTIONS: tuple[str, ...] = (
    "name",
    "registered",
    "registered_against",
    "document",
    "venue",
    "product",
    "computed_before_registration",
    "blocks",
    "funding",
    "admission",
    "fees",
    "simulator",
    "strategies",
    "d_search",
    "metrics",
    "family",
    "statuses",
    "common_kill_conditions",
    "placebo_percentile",
    "hypotheses",
    "ladder",
    "fee_breakeven",
    "robustness",
    "amendment",
)

#: The registered search grids, which the code iterates.
GRIDS: dict[str, dict[str, list[Any]]] = {
    "S1": {
        "skew_bp": [2, 5, 10],
        "k": [0, 0.5, 1, 2],
        "min_edge_bp": [0, 1, 2, 4],
        "soft_limit_clips": [3, 6, 12],
    },
    "S2": {
        "skew_bp": [2, 5, 10],
        "k": [0, 0.5, 1, 2],
        "min_edge_bp": [0, 1, 2, 4],
        "soft_limit_clips": [3, 6, 12],
        "m_ticks": [2, 3, 4],
    },
    "S3": {"action": ["pull", "widen"], "guard_window_min": [15, 60, 240]},
    "S4": {"lambda": [0.5, 1, 2], "one_sided": [False, True]},
}


class PreRegistrationError(ValueError):
    """The YAML does not match the schema it was registered with."""


class HeldOutLocked(PermissionError):
    """A block was about to be read without the permission the protocol requires."""


# ---------------------------------------------------------------------------
# Blocks and access
# ---------------------------------------------------------------------------


def block_of(day: date) -> str | None:
    """The registered block a day belongs to, or None outside every block."""
    for name, (start, end) in BLOCKS.items():
        if start <= day <= end:
            return name
    return None


def block_days(name: str) -> list[date]:
    """Every day of a registered block, in order."""
    start, end = BLOCKS[name]
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


#: Issued only by this module: an :class:`Access` beyond D must carry it.
_ISSUER = object()


@dataclass(frozen=True)
class Access:
    """Permission to read the days of some blocks.

    ``stamp`` travels with everything produced under it: ``development`` for D,
    ``first read`` or ``second read`` for a held-out block.
    """

    blocks: frozenset[str]
    stamp: str
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.blocks <= set(BLOCKS):
            raise ValueError(f"unknown blocks {sorted(self.blocks - set(BLOCKS))}")
        if self.blocks - {"D"} and self.issuer is not _ISSUER:
            raise HeldOutLocked(
                "access to a held-out block is issued only by HeldOutLedger.open, "
                "which checks the frozen configuration and records the read"
            )

    def allows(self, day: date) -> bool:
        return block_of(day) in self.blocks

    def require(self, days: Iterable[date], *, what: str = "read") -> None:
        """Raise :class:`HeldOutLocked` unless every day is in a permitted block.

        Called before any file of the day is opened, so a refused day costs
        nothing and reveals nothing.
        """
        for day in days:
            if not self.allows(day):
                block = block_of(day) or "no block"
                raise HeldOutLocked(
                    f"cannot {what} {day}: it is in {block}, and this access "
                    f"({self.stamp}) permits only {sorted(self.blocks)}"
                )


def development_access() -> Access:
    """Permission to read block D, and nothing else."""
    return Access(frozenset({"D"}), "development")


# ---------------------------------------------------------------------------
# The YAML
# ---------------------------------------------------------------------------

_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.\-]+):(?=\s|$)")
_PLACEHOLDER_LINE = re.compile(
    r"^(?P<head>\s*[A-Za-z0-9_.\-]+: )(?P<value>.*?)(?P<tail>  # [^\n]*"
    + re.escape(PLACEHOLDER_MARK)
    + r"[^\n]*)$"
)


def placeholder_lines(text: str) -> dict[tuple[str, ...], int]:
    """Where each placeholder sits: its key path and its line number (0-based).

    Paths are tracked by indentation, so the same key under two parents is two
    placeholders. Only lines whose comment carries :data:`PLACEHOLDER_MARK` are
    returned.
    """
    lines = text.split("\n")
    stack: list[tuple[int, str]] = []
    found: dict[tuple[str, ...], int] = {}
    for number, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = _KEY_LINE.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, match.group("key")))
        if PLACEHOLDER_MARK in line and _PLACEHOLDER_LINE.match(line):
            found[tuple(key for _, key in stack)] = number
    return found


def _set_placeholders(text: str, values: Mapping[tuple[str, ...], str]) -> str:
    lines = text.split("\n")
    where = placeholder_lines(text)
    for path, rendered in values.items():
        if path not in where:
            raise PreRegistrationError(f"no placeholder at {'.'.join(path)}")
        number = where[path]
        match = _PLACEHOLDER_LINE.match(lines[number])
        assert match is not None
        lines[number] = match.group("head") + rendered + match.group("tail")
    return "\n".join(lines)


def unfrozen_text(text: str) -> str:
    """The file as registered: every placeholder put back to ``null``."""
    where = placeholder_lines(text)
    return _set_placeholders(text, dict.fromkeys(where, "null"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_value(value: Any) -> str:
    """One placeholder value as a single line of YAML that loads back to it."""
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, int):
        rendered = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise PreRegistrationError(f"cannot freeze a non-finite value {value!r}")
        rendered = repr(value)
    elif isinstance(value, date):
        rendered = value.isoformat()
    else:
        rendered = yaml.safe_dump(
            value, default_flow_style=True, sort_keys=False, width=1_000_000
        ).strip()
        if rendered.endswith("..."):
            rendered = rendered[:-3].strip()
    if "\n" in rendered:
        raise PreRegistrationError(f"{value!r} does not fit on one line of YAML")
    if yaml.safe_load(rendered) != value:
        raise PreRegistrationError(f"{value!r} does not survive a YAML round trip")
    return rendered


def freeze_text(text: str, values: Mapping[tuple[str, ...], Any]) -> str:
    """Write the chosen values into their placeholders, and nothing else.

    Every placeholder must be given exactly once and must currently be
    ``null``; the result is validated as a frozen registration before it is
    returned.
    """
    where = placeholder_lines(text)
    missing = set(PLACEHOLDERS) - set(values)
    extra = set(values) - set(PLACEHOLDERS)
    if missing or extra:
        raise PreRegistrationError(
            f"freezing needs every placeholder once; missing {sorted(missing)}, "
            f"unknown {sorted(extra)}"
        )
    lines = text.split("\n")
    for path in values:
        match = _PLACEHOLDER_LINE.match(lines[where[path]])
        if match is None or match.group("value").strip() != "null":
            raise PreRegistrationError(f"{'.'.join(path)} is already frozen")
    frozen = _set_placeholders(text, {path: render_value(v) for path, v in values.items()})
    PreRegistration.from_text(frozen).require_frozen()
    return frozen


def _get(tree: Any, path: Sequence[str]) -> Any:
    node = tree
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise PreRegistrationError(f"missing key {'.'.join(path)}")
        node = node[key]
    return node


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PreRegistrationError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value) or (positive and value <= 0):
        raise PreRegistrationError(
            f"{where} must be a finite{' positive' if positive else ''} number"
        )
    return float(value)


def _per_symbol(value: Any, where: str, *, positive: bool) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise PreRegistrationError(f"{where} must map instruments to numbers, got {value!r}")
    out = {}
    for symbol, number in value.items():
        if not isinstance(symbol, str) or not symbol.endswith("USDT"):
            raise PreRegistrationError(f"{where}: {symbol!r} is not an instrument")
        out[symbol] = _number(number, f"{where}.{symbol}", positive=positive)
    return out


@dataclass(frozen=True)
class PreRegistration:
    """The registered YAML, validated."""

    text: str
    tree: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode())

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> PreRegistration:
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> PreRegistration:
        """Parse and validate; raise :class:`PreRegistrationError` on any departure."""
        where = placeholder_lines(text)
        if set(where) != set(PLACEHOLDERS):
            raise PreRegistrationError(
                f"placeholders differ from the registered ones: missing "
                f"{sorted(set(PLACEHOLDERS) - set(where))}, unexpected "
                f"{sorted(set(where) - set(PLACEHOLDERS))}"
            )
        original = sha256_bytes(unfrozen_text(text).encode())
        if original != REGISTERED_SHA256:
            raise PreRegistrationError(
                "the file differs from the registered one outside its placeholders "
                f"(sha256 with placeholders reset is {original}, registered {REGISTERED_SHA256})"
            )
        tree = yaml.safe_load(text)
        if not isinstance(tree, dict):
            raise PreRegistrationError("the registration must be a mapping")
        registration = cls(text, tree)
        registration._validate()
        return registration

    # -- schema --------------------------------------------------------------

    def _validate(self) -> None:
        tree = self.tree
        if tuple(tree) != SECTIONS:
            raise PreRegistrationError(f"sections differ from the registered {SECTIONS}")
        if tree["name"] != "market_making_preregistration" or tree["venue"] != "bybit":
            raise PreRegistrationError("not the market-making registration")
        for name, (start, end) in BLOCKS.items():
            block = _get(tree, ("blocks", name))
            if block.get("start") != start or block.get("end") != end:
                raise PreRegistrationError(f"block {name} is not {start} to {end}")
            if block.get("days") != (end - start).days + 1:
                raise PreRegistrationError(f"block {name} has the wrong day count")
        base = _get(tree, ("fees", "base"))
        if base != {"maker_bp": 2.0, "taker_bp": 5.5}:
            raise PreRegistrationError(f"the base fee tier is not 2.0 / 5.5 bp: {base}")
        for strategy, grid in GRIDS.items():
            search = _get(tree, ("strategies", strategy, "search"))
            if search != grid:
                raise PreRegistrationError(f"{strategy}'s search grid is not the registered one")
            cells = math.prod(len(v) for v in grid.values())
            if _get(tree, ("strategies", strategy, "cells")) != cells:
                raise PreRegistrationError(f"{strategy} should have {cells} cells")
        d_search = tree["d_search"]
        if d_search.get("budgets_days") != [5, 12, 25] or d_search.get("day_subset_seed") != 0:
            raise PreRegistrationError("the development search's budgets or seed changed")
        self._validate_frozen_values()

    def _validate_frozen_values(self) -> None:
        """Each placeholder is null or a value of its registered kind."""
        for strategy in ("S1", "S2", "S4"):
            for axis, levels in GRIDS[strategy].items():
                value = self.value(("strategies", strategy, "chosen", axis))
                if value is not None and value not in levels:
                    raise PreRegistrationError(
                        f"{strategy}.{axis} = {value!r} is not on its grid {levels}"
                    )
        for axis, levels in GRIDS["S3"].items():
            value = self.value(("strategies", "S3", "chosen", axis))
            if value is not None and value not in levels:
                raise PreRegistrationError(f"S3.{axis} = {value!r} is not on its grid {levels}")
        guards = self.value(("strategies", "S3", "chosen", "guards"))
        if guards is not None and guards not in ("S1", "S2"):
            raise PreRegistrationError(f"S3 guards S1 or S2, not {guards!r}")
        for path, positive in (
            (("simulator", "clip", "notional_usdt"), True),
            (("simulator", "volatility", "sigma_ref"), True),
            (("strategies", "S4", "theta_bp"), True),
            (("strategies", "S4", "beta"), False),
        ):
            value = self.value(path)
            if value is not None:
                _per_symbol(value, ".".join(path), positive=positive)
        for path in (
            ("metrics", "minimum_detectable_effect", "sigma_d_usd_day"),
            ("metrics", "minimum_detectable_effect", "value_usd_day"),
            ("hypotheses", "H3", "flag_rate_on_d_per_day"),
        ):
            value = self.value(path)
            if value is not None:
                _number(value, ".".join(path), positive=False)
                if value < 0:
                    raise PreRegistrationError(f"{'.'.join(path)} cannot be negative")
        gate = self.value(("hypotheses", "H2.3", "gate_threshold"))
        if gate is not None and not -1.0 <= _number(gate, "H2.3.gate_threshold") <= 1.0:
            raise PreRegistrationError("an autocorrelation threshold lies in [-1, 1]")
        table = self.value(("admission", "table"))
        if table is not None and (
            not isinstance(table, list)
            or not all(isinstance(row, Mapping) and "symbol" in row for row in table)
        ):
            raise PreRegistrationError("the admission table is a list of rows with a symbol")
        for strategy in ("S1", "S2"):
            near = self.value(("d_search", "neighbourhood", strategy))
            if near is not None and not (
                isinstance(near, Mapping) and {"chosen", "cells", "median"} <= set(near)
            ):
                raise PreRegistrationError(
                    f"the {strategy} neighbourhood records its chosen cell, cells and median"
                )
        when = self.value(("amendment", "date"))
        if when is not None and not isinstance(when, date):
            raise PreRegistrationError("the amendment date must be a date")

    # -- reading -------------------------------------------------------------

    def value(self, path: Sequence[str]) -> Any:
        return _get(self.tree, path)

    @property
    def frozen(self) -> bool:
        """True when every placeholder holds a value."""
        return all(self.value(p) is not None for p in PLACEHOLDERS)

    def require_frozen(self) -> PreRegistration:
        """Raise unless every placeholder is filled, or every one is still null."""
        empty = [".".join(p) for p in PLACEHOLDERS if self.value(p) is None]
        if empty:
            raise PreRegistrationError(f"placeholders still null: {empty}")
        return self

    def require_consistent(self) -> PreRegistration:
        """Raise when some placeholders are filled and others are not."""
        filled = [self.value(p) is not None for p in PLACEHOLDERS]
        if any(filled) and not all(filled):
            raise PreRegistrationError("the registration is partly frozen")
        return self


# ---------------------------------------------------------------------------
# The held-out ledger
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldOutLocked(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    return done.stdout


@dataclass
class HeldOutLedger:
    """The one gate to the held-out and boundary blocks.

    Paths are relative to ``repo``. :meth:`open` checks the protocol, writes
    the ledger entry and returns the :class:`Access` that the study's readers
    require; nothing else issues one.
    """

    repo: Path = Path()
    config: Path = CONFIG_PATH
    document: Path = DOCUMENT_PATH
    ledger: Path = LEDGER_PATH

    def expected_sha256(self) -> str:
        """The frozen configuration's sha256, as the amendment records it."""
        path = self.repo / self.document
        if not path.exists():
            raise HeldOutLocked(f"no pre-registration document at {path}")
        found = FROZEN_HASH_PATTERN.findall(path.read_text(encoding="utf-8"))
        if not found:
            raise HeldOutLocked(
                "no amendment records the frozen configuration's sha256; the "
                "development-period amendment must be committed before H is read"
            )
        if len(set(found)) != 1:
            raise HeldOutLocked(f"the document records several frozen hashes: {sorted(set(found))}")
        return str(found[0])

    def verify(self) -> tuple[str, str]:
        """Check everything but the ledger itself; return (commit, config sha256)."""
        expected = self.expected_sha256()
        path = self.repo / self.config
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise HeldOutLocked(
                f"{self.config} has sha256 {actual}, but the amendment froze {expected}: "
                "the configuration changed after it was frozen"
            )
        try:
            PreRegistration.load(path).require_frozen()
        except PreRegistrationError as exc:
            raise HeldOutLocked(f"the frozen configuration is not valid: {exc}") from exc
        for tracked in (self.config, self.document):
            _git(self.repo, "ls-files", "--error-unmatch", str(tracked))
        status = _git(self.repo, "status", "--porcelain", "--untracked-files=normal").strip()
        if status:
            raise HeldOutLocked(f"the working tree is not clean:\n{status}")
        committed = _git(self.repo, "show", f"HEAD:{self.config.as_posix()}")
        if sha256_bytes(committed.encode()) != expected:
            raise HeldOutLocked("the committed configuration is not the frozen one")
        commit = _git(self.repo, "rev-parse", "HEAD").strip()
        return commit, actual

    def entries(self) -> dict[str, list[dict[str, Any]]]:
        path = self.repo / self.ledger
        if not path.exists():
            return {}
        loaded: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def open(self, block: str, *, force: bool = False) -> Access:
        """Permission to read ``block`` (``H`` or ``F``), recorded before it is granted."""
        if block not in ("H", "F"):
            raise ValueError(f"the ledger guards H and F, not {block!r}")
        commit, digest = self.verify()
        entries = self.entries()
        if block == "F":
            if not entries.get("H"):
                raise HeldOutLocked("F is read only after H: the ledger has no read of H")
            _git(self.repo, "ls-files", "--error-unmatch", str(self.ledger))
        previous = entries.get(block, [])
        if previous and not force:
            first = previous[0]
            raise HeldOutLocked(
                f"{block} was read on {first['utc']} at commit {first['commit'][:7]}; "
                "a second read needs force=True and is stamped 'second read'"
            )
        stamp = "second read" if previous else "first read"
        entry = {
            "read": stamp,
            "commit": commit,
            "config_sha256": digest,
            "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        entries[block] = [*previous, entry]
        path = self.repo / self.ledger
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # The flags run continuously from the start of D and the tapes read a
        # lookback before each block, so a block's access includes the blocks
        # before it.
        blocks = {"H": {"D", "H"}, "F": {"D", "H", "buffer", "F"}}[block]
        return Access(frozenset(blocks), stamp, issuer=_ISSUER)
