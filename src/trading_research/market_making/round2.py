"""The second round's registration as code: its values, who may read what, its ledger.

The second round was registered in ``docs/preregistration/market_making_round2.md``
with ``configs/mm_prereg_round2.yaml`` before any basket file was fetched. This
module holds the code to that registration, as :mod:`.prereg` does for round
one, and shares none of round one's state: its own access, its own ledger.

The YAML
--------
:class:`Registration` validates the file against the schema it was registered
with. The values chosen on block D are ``null`` placeholders commented "frozen
by the D amendment"; putting ``null`` back into every one of them must give
the registered sha256 (:data:`REGISTERED_SHA256`), so nothing outside them can
move. :func:`freeze_text` writes the chosen values and nothing else.

Who may read what
-----------------
Access is granted **per instrument and block** (:class:`Access`). The
development access permits D for the eight candidates and for BICOUSDT, and
nothing else. Every other access is issued only by :meth:`Ledger.open`, which
refuses unless the amendment records the frozen file's sha256, the file on disk
has it, is fully frozen and committed, and the working tree is clean; it
appends the read to ``experiments/results/mm_round2_ledger.json`` before the
access is returned. Its reads, in the order the registration allows them:

* ``first``: H for the admitted basket together with P for BICOUSDT;
* ``basket_P``: P for the admitted basket, only once the ledger records the
  first read and is committed (H's results are committed with it);
* ``Q``: Q for the instruments named, after the read that covered their P is
  recorded and committed.

A read already recorded is refused; ``force=True`` reads it again, stamped
``second read``, which cannot change a status.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from trading_research.market_making import gate
from trading_research.market_making.prereg import (
    _PLACEHOLDER_LINE,
    FROZEN_HASH_PATTERN,
    HeldOutLocked,
    PreRegistrationError,
    _git,
    _set_placeholders,
    placeholder_lines,
    render_value,
    sha256_bytes,
    unfrozen_text,
)

CONFIG_PATH = Path("configs/mm_prereg_round2.yaml")
DOCUMENT_PATH = Path("docs/preregistration/market_making_round2.md")
LEDGER_PATH = Path("experiments/results/mm_round2_ledger.json")

#: The sha256 of ``configs/mm_prereg_round2.yaml`` as registered, every
#: placeholder ``null`` (stated at the end of the document).
REGISTERED_SHA256 = "98dbec924e13e48ed92680515b36e2cceaa04c7dc98e47ba8c62cec629c07345"

#: The eight candidates, and the instrument of B4.
BASKET: tuple[str, ...] = (
    "ALICEUSDT",
    "ALGOUSDT",
    "JTOUSDT",
    "ZECUSDT",
    "GALAUSDT",
    "GMTUSDT",
    "CAKEUSDT",
    "IOTAUSDT",
)
B4_SYMBOL = "BICOUSDT"
INSTRUMENTS: tuple[str, ...] = (*BASKET, B4_SYMBOL)

#: The registered blocks, inclusive, UTC dates.
BLOCKS: dict[str, tuple[date, date]] = {
    "D": (date(2024, 2, 1), date(2024, 2, 25)),
    "H": (date(2024, 2, 26), date(2024, 3, 9)),
    "F": (date(2024, 3, 12), date(2024, 4, 7)),
    "P": (date(2024, 4, 8), date(2024, 5, 5)),
    "Q": (date(2024, 5, 6), date(2024, 6, 2)),
}

#: BICOUSDT's values frozen by round one and reused here.
BICO_CLIP_USDT = 20.0434
BICO_SIGMA_REF = 7.92472
BICO_BASES: dict[str, list[float]] = {"S1": [2, 0.5, 4, 6], "S1_touch": [2, 0, 0, 6]}

#: The status recorded for B1 to B3b if the D table admits no instrument.
NOT_TESTED = "not tested: no instrument admitted"


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


# ---------------------------------------------------------------------------
# Access, per instrument and block
# ---------------------------------------------------------------------------

#: Issued only by this module: an :class:`Access` beyond development must carry it.
_ISSUER = object()

#: What the development access permits: D, for every instrument of the round.
DEVELOPMENT: frozenset[tuple[str, str]] = frozenset((s, "D") for s in INSTRUMENTS)


@dataclass(frozen=True)
class Access:
    """Permission to read some blocks of some instruments.

    ``grants`` are (instrument, block) pairs. ``stamp`` travels with everything
    produced under it: ``development``, ``first read`` or ``second read``.
    """

    grants: frozenset[tuple[str, str]]
    stamp: str
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        unknown = {(s, b) for s, b in self.grants if s not in INSTRUMENTS or b not in BLOCKS}
        if unknown:
            raise ValueError(f"unknown instruments or blocks {sorted(unknown)}")
        if self.grants - DEVELOPMENT and self.issuer is not _ISSUER:
            raise HeldOutLocked(
                "access beyond D is issued only by round2.Ledger.open, which checks "
                "the frozen configuration and records the read"
            )

    def allows(self, symbol: str, day: date) -> bool:
        return (symbol, block_of(day)) in self.grants

    def require(self, symbol: str, days: Iterable[date], *, what: str = "read") -> None:
        """Raise :class:`HeldOutLocked` unless every day of ``symbol`` is permitted.

        Called before any file of the day is opened, so a refused day costs
        nothing and reveals nothing.
        """
        for day in days:
            if not self.allows(symbol, day):
                block = block_of(day) or "no block"
                permitted = sorted(b for s, b in self.grants if s == symbol)
                raise HeldOutLocked(
                    f"cannot {what} {symbol} on {day}: it is in {block}, and this "
                    f"access ({self.stamp}) permits {symbol} only in {permitted}"
                )

    def guard(self, symbol: str, what: str = "simulate") -> Callable[[Sequence[date]], None]:
        """A day guard for one instrument, as the simulator's runners take."""
        return _Guard(self, symbol, what)

    def symbols(self, block: str) -> list[str]:
        """The instruments this access permits in ``block``, in the round's order."""
        return [s for s in INSTRUMENTS if (s, block) in self.grants]


@dataclass(frozen=True)
class _Guard:
    access: Access
    symbol: str
    what: str

    def __call__(self, days: Sequence[date]) -> None:
        self.access.require(self.symbol, days, what=self.what)


def development_access() -> Access:
    """Permission to read D for the eight candidates and BICOUSDT, and nothing else."""
    return Access(DEVELOPMENT, "development")


# ---------------------------------------------------------------------------
# The YAML
# ---------------------------------------------------------------------------

#: Every placeholder the amendment fills, as a path of keys.
PLACEHOLDERS: tuple[tuple[str, ...], ...] = (
    ("admission", "table"),
    ("simulator", "clip", "notional_usdt"),
    ("simulator", "volatility", "sigma_ref"),
    ("strategies", "S1", "chosen"),
    ("d_search", "S1", "neighbourhood"),
    ("d_search", "gate", "chosen"),
    ("d_search", "gate", "chosen_neighbourhood"),
    ("d_search", "gate", "hours_masks"),
    ("d_search", "b3_choice", "chosen"),
    ("d_search", "b4", "chosen"),
    ("d_search", "b4", "chosen_neighbourhood"),
    ("d_search", "b4", "hours_masks"),
    ("metrics", "minimum_detectable_effect", "sigma_d"),
    ("metrics", "minimum_detectable_effect", "value"),
    ("amendment", "date"),
)

#: Top-level sections of the registered YAML, in order.
SECTIONS: tuple[str, ...] = (
    "name",
    "registered",
    "registered_against",
    "document",
    "venue",
    "product",
    "computed_before_registration",
    "round_one",
    "prior_information",
    "blocks",
    "instruments",
    "fetch",
    "admission",
    "fees",
    "simulator",
    "strategies",
    "gate",
    "d_search",
    "metrics",
    "family",
    "statuses",
    "common_kill_conditions",
    "placebo_percentile",
    "hypotheses",
    "measurements",
    "demonstrative_interval",
    "access",
    "compute",
    "amendment",
)

#: S1's registered grid, in the order of a chosen cell's four values.
S1_GRID: dict[str, list[float]] = {
    "skew_bp": [2, 5, 10],
    "k": [0, 0.5, 1, 2],
    "min_edge_bp": [0, 1, 2, 4],
    "soft_limit_clips": [3, 6, 12],
}

#: The five primaries and the held-out days each is read on first.
PRIMARIES: dict[str, int] = {"B1": 13, "B2": 13, "B3a": 13, "B3b": 13, "B4": 28}

#: The registered fee tiers, maker and taker, bp.
TIERS: dict[str, tuple[float, float]] = {
    "base": (2.0, 5.5),
    "pro1_altcoin": (0.0, 2.8),
    "programme": (-1.0, 2.8),
}


def _get(tree: Any, path: Sequence[str]) -> Any:
    node = tree
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise PreRegistrationError(f"missing key {'.'.join(path)}")
        node = node[key]
    return node


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreRegistrationError(message)


def _positive(value: Any, where: str) -> float:
    ok = isinstance(value, int | float) and not isinstance(value, bool)
    _require(ok and math.isfinite(value) and value > 0, f"{where} must be a positive number")
    return float(value)


def _hours(value: Any, where: str) -> None:
    _require(isinstance(value, list), f"{where} must be a list of hours")
    _require(
        all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h < 24 for h in value),
        f"{where} holds hours 0 to 23",
    )
    _require(value == sorted(set(value)), f"{where} must be sorted and without repeats")


def _masks(value: Any, where: str) -> None:
    _require(isinstance(value, Mapping), f"{where} maps each margin to its hours")
    margins = {float(m) for m in value}
    _require(margins == set(gate.MARGINS_BP), f"{where} needs the margins {gate.MARGINS_BP}")
    for margin, hours in value.items():
        _hours(hours, f"{where}.{margin}")


@dataclass(frozen=True)
class Registration:
    """The second round's registered YAML, validated."""

    text: str
    tree: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode())

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> Registration:
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> Registration:
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
        _require(isinstance(tree, dict), "the registration must be a mapping")
        registration = cls(text, tree)
        registration._validate()
        return registration

    def value(self, path: Sequence[str]) -> Any:
        return _get(self.tree, path)

    def _validate(self) -> None:
        tree = self.tree
        _require(tuple(tree) == SECTIONS, f"sections differ from the registered {SECTIONS}")
        _require(tree["name"] == "market_making_round2_preregistration", "not round two")
        for name, (start, end) in BLOCKS.items():
            block = _get(tree, ("blocks", name))
            _require(block.get("start") == start and block.get("end") == end, f"block {name}")
        _require(
            tuple(self.value(("instruments", "basket_candidates"))) == BASKET, "the basket changed"
        )
        _require(self.value(("instruments", "b4_instrument")) == B4_SYMBOL, "B4's instrument")
        for tier, (maker, taker) in TIERS.items():
            fees = self.value(("fees", tier))
            _require(fees == {"maker_bp": maker, "taker_bp": taker}, f"fee tier {tier}")
        _require(self.value(("strategies", "S1", "search")) == S1_GRID, "S1's grid changed")
        search = self.value(("gate", "search"))
        _require(
            search["window_min"] == list(gate.WINDOWS_MIN)
            and [float(m) for m in search["margin_bp"]] == list(gate.MARGINS_BP)
            and search["base"] == list(gate.BASES)
            and set(search["kind"]) == set(gate.KINDS)
            and self.value(("gate", "cells")) == len(gate.gate_cells()),
            "the gate's search changed",
        )
        _require(self.value(("gate", "horizon_s")) == gate.HORIZON_S, "the gate's horizon")
        _require(self.value(("gate", "min_prints_in_window")) == gate.MIN_PRINTS, "min prints")
        s1 = self.value(("d_search", "S1"))
        _require(
            s1["budgets_days"] == [5, 12, 25] and s1["day_subset_seed"] == 0,
            "the development search's budgets or seed changed",
        )
        n = self.value(("metrics", "minimum_detectable_effect", "n"))
        _require(n == PRIMARIES, "the held-out days per primary changed")
        self._validate_frozen_values()

    @property
    def frozen(self) -> bool:
        """True when every placeholder holds a value."""
        return all(self.value(p) is not None for p in PLACEHOLDERS)

    def require_frozen(self) -> Registration:
        empty = [".".join(p) for p in PLACEHOLDERS if self.value(p) is None]
        if empty:
            raise PreRegistrationError(f"placeholders still null: {empty}")
        return self

    def require_consistent(self) -> Registration:
        filled = [self.value(p) is not None for p in PLACEHOLDERS]
        if any(filled) and not all(filled):
            raise PreRegistrationError("the registration is partly frozen")
        return self

    # -- the frozen values ----------------------------------------------------

    def _validate_frozen_values(self) -> None:
        """Each placeholder is null or a value of its registered kind, and the
        per-instrument values cover exactly the admitted instruments."""
        table = self.value(("admission", "table"))
        admitted: set[str] | None = None
        if table is not None:
            _require(isinstance(table, list), "the admission table is a list of rows")
            symbols = [str(row.get("symbol")) for row in table if isinstance(row, Mapping)]
            _require(sorted(symbols) == sorted(BASKET), "the table has one row per candidate")
            _require(
                all(isinstance(row.get("admitted"), bool) for row in table),
                "each row says whether its instrument is admitted",
            )
            admitted = {row["symbol"] for row in table if row["admitted"]}
        for path in (
            ("simulator", "clip", "notional_usdt"),
            ("simulator", "volatility", "sigma_ref"),
            ("strategies", "S1", "chosen"),
            ("d_search", "S1", "neighbourhood"),
            ("d_search", "gate", "hours_masks"),
        ):
            value = self.value(path)
            if value is None:
                continue
            where = ".".join(path)
            _require(isinstance(value, Mapping), f"{where} maps instruments to values")
            if admitted is not None:
                _require(set(value) == admitted, f"{where} must cover the admitted instruments")
            for symbol, item in value.items():
                if path[-1] in ("notional_usdt", "sigma_ref"):
                    _positive(item, f"{where}.{symbol}")
                elif path[-1] == "chosen":
                    _require(
                        isinstance(item, list)
                        and len(item) == 4
                        and all(
                            v in levels for v, levels in zip(item, S1_GRID.values(), strict=True)
                        ),
                        f"{where}.{symbol} = {item!r} is not a cell of S1's grid",
                    )
                elif path[-1] == "neighbourhood":
                    _require(
                        isinstance(item, Mapping) and {"chosen", "cells", "median"} <= set(item),
                        f"{where}.{symbol} records its chosen cell, cells and median",
                    )
                else:
                    _masks(item, f"{where}.{symbol}")
        none_admitted = admitted is not None and not admitted
        for path in (("d_search", "gate", "chosen"), ("d_search", "b4", "chosen")):
            value = self.value(path)
            if value is None or (none_admitted and value == NOT_TESTED and path[1] == "gate"):
                continue
            try:
                gate.GateCell.from_record(value)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise PreRegistrationError(f"{'.'.join(path)} is not a gate cell: {exc}") from exc
        for path in (
            ("d_search", "gate", "chosen_neighbourhood"),
            ("d_search", "b4", "chosen_neighbourhood"),
        ):
            value = self.value(path)
            if value is None or (none_admitted and value == NOT_TESTED and path[1] == "gate"):
                continue
            _require(
                isinstance(value, Mapping) and {"cells", "median"} <= set(value),
                f"{'.'.join(path)} records its cells and median",
            )
        masks = self.value(("d_search", "b4", "hours_masks"))
        if masks is not None:
            _masks(masks, "d_search.b4.hours_masks")
        b3 = self.value(("d_search", "b3_choice", "chosen"))
        if b3 is not None and not (none_admitted and b3 == NOT_TESTED):
            _require(
                isinstance(b3, Mapping)
                and set(b3) == {"pro1_altcoin", "programme"}
                and all(
                    isinstance(v, Mapping) and v.get("strategy") in ("S1", "G") for v in b3.values()
                ),
                "B3's choice names S1 or G at each tier",
            )
        for name in ("sigma_d", "value"):
            value = self.value(("metrics", "minimum_detectable_effect", name))
            if value is None:
                continue
            _require(
                isinstance(value, Mapping) and set(value) <= set(PRIMARIES),
                f"the minimum detectable effect's {name} is per primary",
            )
            for primary, item in value.items():
                claims = item if isinstance(item, Mapping) else {"": item}
                for claim, number in claims.items():
                    ok = isinstance(number, int | float) and not isinstance(number, bool)
                    _require(
                        ok and math.isfinite(number) and number >= 0,
                        f"{name}.{primary}.{claim} must be a non-negative number",
                    )
        when = self.value(("amendment", "date"))
        _require(when is None or isinstance(when, date), "the amendment date must be a date")

    # -- reading the frozen values ------------------------------------------------

    @property
    def admitted(self) -> tuple[str, ...]:
        """The admitted basket, in the registered order."""
        table = self.value(("admission", "table")) or []
        chosen = {row["symbol"] for row in table if row["admitted"]}
        return tuple(s for s in BASKET if s in chosen)

    def clip(self, symbol: str) -> float:
        if symbol == B4_SYMBOL:
            return BICO_CLIP_USDT
        return float(self.value(("simulator", "clip", "notional_usdt"))[symbol])

    def sigma_ref(self, symbol: str) -> float:
        if symbol == B4_SYMBOL:
            return BICO_SIGMA_REF
        return float(self.value(("simulator", "volatility", "sigma_ref"))[symbol])

    def s1_cell(self, symbol: str) -> dict[str, float]:
        """The instrument's chosen S1 cell, by axis; round one's for BICOUSDT."""
        values = (
            BICO_BASES["S1"]
            if symbol == B4_SYMBOL
            else self.value(("strategies", "S1", "chosen"))[symbol]
        )
        return dict(zip(S1_GRID, (float(v) for v in values), strict=True))

    def gate_cell(self, symbol: str | None = None) -> gate.GateCell:
        """The chosen gate cell: the basket's, or B4's for BICOUSDT."""
        which = "b4" if symbol == B4_SYMBOL else "gate"
        return gate.GateCell.from_record(self.value(("d_search", which, "chosen")))

    def hours_masks(self, symbol: str) -> dict[float, tuple[int, ...]]:
        """The instrument's hour masks at the base fee, by margin."""
        masks = (
            self.value(("d_search", "b4", "hours_masks"))
            if symbol == B4_SYMBOL
            else self.value(("d_search", "gate", "hours_masks"))[symbol]
        )
        return {float(m): tuple(int(h) for h in hours) for m, hours in masks.items()}


class _NoAliases(yaml.SafeDumper):
    """A dumper that writes a repeated object out again rather than as an
    alias: each placeholder is its own line, and an anchor in one would clash
    with the same anchor in another."""

    def ignore_aliases(self, data: Any) -> bool:  # noqa: ARG002
        return True


def render(value: Any) -> str:
    """One placeholder value as a single line of YAML that loads back to it."""
    if value is None or isinstance(value, bool | int | float | date):
        return render_value(value)
    rendered = yaml.dump(
        value, Dumper=_NoAliases, default_flow_style=True, sort_keys=False, width=1_000_000
    ).strip()
    if rendered.endswith("..."):
        rendered = rendered[:-3].strip()
    if "\n" in rendered or yaml.safe_load(rendered) != value:
        raise PreRegistrationError(f"{value!r} does not fit on one line of YAML and back")
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
    frozen = _set_placeholders(text, {path: render(v) for path, v in values.items()})
    Registration.from_text(frozen).require_frozen()
    return frozen


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

#: The reads the ledger guards, and what each one opens.
READS: dict[str, str] = {
    "first": "H for the admitted basket, with P for BICOUSDT",
    "basket_P": "P for the admitted basket, after H's results are committed",
    "Q": "Q for the instruments named, after their P's results are committed",
}


@dataclass
class Ledger:
    """The one gate to the second round's held-out blocks.

    Paths are relative to ``repo``. :meth:`open` checks the protocol, writes the
    ledger entry and returns the :class:`Access` the round's readers require;
    nothing else issues one beyond D.
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
                "development-period amendment must be committed before a held-out read"
            )
        if len(set(found)) != 1:
            raise HeldOutLocked(f"the document records several frozen hashes: {sorted(set(found))}")
        return str(found[0])

    def verify(self) -> tuple[str, Registration]:
        """Everything but the ledger itself; returns the commit and the registration."""
        expected = self.expected_sha256()
        path = self.repo / self.config
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise HeldOutLocked(
                f"{self.config} has sha256 {actual}, but the amendment froze {expected}: "
                "the configuration changed after it was frozen"
            )
        try:
            registration = Registration.load(path).require_frozen()
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
        return _git(self.repo, "rev-parse", "HEAD").strip(), registration

    def entries(self) -> dict[str, list[dict[str, Any]]]:
        path = self.repo / self.ledger
        if not path.exists():
            return {}
        loaded: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def _committed_after(self, read: str, entries: Mapping[str, Any]) -> None:
        """Refuse unless ``read`` is recorded and the ledger holding it is committed."""
        if not entries.get(read):
            raise HeldOutLocked(f"this read comes after the {read!r} read, which is not recorded")
        _git(self.repo, "ls-files", "--error-unmatch", str(self.ledger))

    def open(self, read: str, *, symbols: Sequence[str] = (), force: bool = False) -> Access:
        """Permission for one of :data:`READS`, recorded before it is granted.

        ``symbols`` names Q's instruments. Every access also permits D.
        """
        if read not in READS:
            raise ValueError(f"the ledger guards {sorted(READS)}, not {read!r}")
        commit, registration = self.verify()
        entries = self.entries()
        admitted = registration.admitted
        if read == "first":
            opened = {(s, "H") for s in admitted} | {(B4_SYMBOL, "P")}
        elif read == "basket_P":
            if not admitted:
                raise HeldOutLocked("no instrument was admitted on D: P is not read for the basket")
            self._committed_after("first", entries)
            opened = {(s, "P") for s in admitted}
        else:
            if not symbols:
                raise ValueError("name the instruments Q is read for")
            outside = sorted(set(symbols) - set(admitted) - {B4_SYMBOL})
            if outside:
                raise HeldOutLocked(f"{outside} were not admitted: Q is not read for them")
            self._committed_after("first", entries)
            if set(symbols) - {B4_SYMBOL}:
                self._committed_after("basket_P", entries)
            opened = {(s, "Q") for s in symbols}
        previous = [
            entry
            for entry in entries.get(read, [])
            if {tuple(grant) for grant in entry.get("grants", [])} & opened
        ]
        if previous and not force:
            first = previous[0]
            raise HeldOutLocked(
                f"the {read!r} read was made on {first['utc']} at commit {first['commit'][:7]}; "
                "a second read needs force=True and is stamped 'second read'"
            )
        stamp = "second read" if previous else "first read"
        entry = {
            "read": stamp,
            "commit": commit,
            "config_sha256": registration.sha256,
            "grants": sorted([list(grant) for grant in opened]),
            "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        entries[read] = [*entries.get(read, []), entry]
        path = self.repo / self.ledger
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return Access(frozenset(DEVELOPMENT | opened), stamp, issuer=_ISSUER)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

SPREAD_BP_MIN = 4.0
SHARE_TWO_TICKS_MIN = 0.25
TICK_BP_MIN = 4.0
PRINTS_PER_DAY_MIN = 5_000.0
COVERAGE_MIN = 0.90


def admit(table: Any) -> Any:
    """The registered rule, A1 or A2, on the screen's D table (a DataFrame).

    Adds ``A1``, ``A2``, ``admitted`` and ``criterion`` (``A1``, ``A2``,
    ``A1+A2`` or ``none``).
    """
    out = table.copy()
    covered = (out["coverage_D"] >= COVERAGE_MIN) & (out["coverage_H"] >= COVERAGE_MIN)
    busy = out["prints_per_day"] >= PRINTS_PER_DAY_MIN
    out["A1"] = (
        (out["spread_bp_time_weighted"] >= SPREAD_BP_MIN)
        & (out["share_at_two_ticks_or_more"] >= SHARE_TWO_TICKS_MIN)
        & busy
        & covered
    )
    out["A2"] = (out["tick_bp"] >= TICK_BP_MIN) & busy & covered
    out["admitted"] = out["A1"] | out["A2"]
    out["criterion"] = [
        "A1+A2" if a and b else "A1" if a else "A2" if b else "none"
        for a, b in zip(out["A1"], out["A2"], strict=True)
    ]
    return out
