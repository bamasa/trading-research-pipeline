"""The spot study's registration as code: its YAML validated, its placeholders and hashes.

The study is registered in ``docs/preregistration/spot_2026.md`` with
``configs/spot_2026.yaml``. This module reads that file on the 2025 study's
pattern, bound to this study's blocks (:mod:`trading_research.spot.access`);
the 2025 study's loader is not used.

**Placeholders.** The values set on D are ``null`` lines commented "frozen by
the D amendment" (:data:`PLACEHOLDERS`). Putting ``null`` back into every one of
them must give the registered sha256 (:data:`REGISTERED_SHA256`, the hash after
the information-set amendment and the code's choices of 2026-10-25 and the entry
point's choices and the corrections of 2026-10-26), or the one a later amendment
records on a line ``Null-placeholder sha256: `<hash>``` under ``## Amendments``
(the latest such line is accepted; a superseded hash never is; at the H and P
gates the ledger accepts only a line the document held at the code-freeze commit).
:func:`freeze_text` writes the frozen values and nothing else; the D amendment
records the result's sha256 as ``Frozen configuration sha256: `<hash>```.

**The information set.** H is blind only if the owner confirmed all three end
dates and every one falls before H's first day, ``never`` counting as before
(:func:`h_is_blind`); otherwise the fallback holds and every status carries
"(H inside S25's information set)".

**The fee amendment** may only add intervals for P's days under
``fees.intervals``, each on one line, every one starting at or after
:data:`P_INTERVALS_FROM`. :func:`strip_p_intervals` removes them; the text that
remains is the configuration H was read with.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import yaml

from trading_research.market_making.prereg import (
    PLACEHOLDER_MARK,
    PreRegistrationError,
    placeholder_lines,
    render_value,
    sha256_bytes,
    unfrozen_text,
)
from trading_research.spot import access as sa

CONFIG_PATH = Path("configs/spot_2026.yaml")
DOCUMENT_PATH = Path("docs/preregistration/spot_2026.md")
LEDGER_PATH = Path("experiments/results/spot_2026_ledger.json")
RESULTS_PATHS: dict[str, Path] = {
    block: Path(f"experiments/results/spot_2026_{block}.json") for block in sa.HELD_OUT
}
#: The run log's parts (under ``data/``, never committed) and its committed manifest.
TRIALS_PATH = Path("data/spot_2026/spot_2026_trials.parquet")
TRIALS_MANIFEST_PATH = Path("experiments/results/spot_2026_trials.json")

#: The sha256 of the YAML as registered, every placeholder ``null``, after the
#: information-set amendment of 2026-10-25, the code's choices of that day and the
#: entry point's choices and the corrections of 2026-10-26.
REGISTERED_SHA256 = "4e75c7addf549d8da0ca19230170fd1491a2f444b81055371d2137e097d05afb"
#: Earlier texts' hashes: superseded, never accepted.
SUPERSEDED_SHA256: tuple[str, ...] = (
    "886fe1805b721994b7bf74357271aa41bb9a3f56129ad30a9eacb1736eb35451",
    "81ad1ff49d7190268749cd7f1eaf19ac1a0b8145317efe3eb43d827f1d18ff76",
    "1206d6b71c2f0ad2e76d65b0e91019899f9ceb6c53607fca46dbfec7235e17da",
    "b19beadc6d7e8a70769e279eb9f9d2935587f6035e7b826586e9b6c98b51f5dc",
    "056035e10c51c88c8381ee7f27238a420ae3da4207b80ef3197e699a312fea6a",
)

#: The 2025 study's files S25-T loads, bound by hash (``slots.S25-T``).
CONFIG_2025_PATH = Path("configs/short_horizon_2025.yaml")
CONFIG_2025_SHA256 = "e9cb5e0355e4506e6e450356135836b6106a6f3c6ecaa76aa6a07d0733acbdef"
FROZEN_DIR_2025 = Path("models/short_horizon_2025")
FROZEN_FILES_2025: dict[str, str] = {
    "slot_file_S25": "8ddf4ff5b2e25ee2696da4cff2bd1ec862bd9caf9a112fe30e237963f87e4a2d",
    "fitted_values": "ca327600da8e231edd5296d4ee9ee7ca93472001ca1536de1e513d6af109a6b9",
}
FROZEN_PATHS_2025: dict[str, str] = {
    "slot_file_S25": "models/short_horizon_2025/finalists/S25.json",
    "weights": "models/short_horizon_2025/weights/<fit key>",
    "fitted_values": "models/short_horizon_2025/fitted_values.json",
}
_FIT = "BTCUSDT__C__gbm__micro__hold_s={}__label_fee_bp={}.pkl"
NEIGHBOURHOOD_FITS_2025: dict[str, str] = {
    _FIT.format(20, 0): "23edb21803a80589b782493ebd39ae78e92e69e7a9a4682a7728ec882fa400bf",
    _FIT.format(20, 5.5): "0b007d35d846533f5d93bbd3df46b3217c7273a0015f6d2383ba56d8da359f3a",
    _FIT.format(50, 0): "4364cf9d22df8aa592adfb6509f3a802374aa0ee8628601c394086e9093cc0d1",
    _FIT.format(50, 5.5): "5b6c57fd50687dfd0ef565fa7fd7440bd88b5213e7832c11e836762fc83d9031",
    _FIT.format(120, 0): "be8e7ead202f79c79e92ba2923866e27d2e308a01ad16e45b843c827b68eda11",
    _FIT.format(120, 5.5): "c6b7354d497ab9ab19928d491f8003cde0dd51866819bf0fd55360367a581bed",
}
#: S25-T's own fit, one of the six.
S25_T_FIT = _FIT.format(50, 0)

#: Lines an amendment writes, under the document's ``## Amendments``.
NULL_HASH_PATTERN = re.compile(r"Null-placeholder sha256: `([0-9a-f]{64})`")
FROZEN_HASH_PATTERN = re.compile(r"Frozen configuration sha256: `([0-9a-f]{64})`")
AMENDMENTS_HEADING = "## Amendments"

#: Every placeholder the D amendment fills, as a path of keys, in file order.
PLACEHOLDERS: tuple[tuple[str, ...], ...] = (
    ("data", "reader_test_result"),
    ("data", "max_sequence_gaps"),
    ("data", "excluded_days_D"),
    ("data", "instrument_spec"),
    ("common", "book_levels", "walk"),
    ("features", "fitted_values_sha256"),
    ("models", "weights_sha256"),
    ("models", "thresholds"),
    ("metrics", "mde", "sigma_d"),
    ("metrics", "mde", "value"),
    ("measurements", "result_D"),
    ("compute", "record"),
    ("amendment", "date"),
    ("artefacts", "frozen_slot_files_sha256"),
)

#: Top-level sections of the registered YAML, in order.
SECTIONS: tuple[str, ...] = (
    "name", "registered", "registered_against", "document", "registers", "searched",
    "instruments", "computed_before_registration", "prior_information", "information_set",
    "expected_outcome", "data", "fees", "blocks", "common", "slots", "measured_rows", "labels",
    "features", "models", "d_procedure", "metrics", "family", "statuses", "kill_conditions",
    "placebo_percentile", "placebos", "hypotheses", "measurements", "fetch", "compute",
    "access", "amendment", "artefacts",
)  # fmt: skip

#: The family under Holm, the rows measured beside it, and where each runs.
FAMILY: tuple[str, ...] = ("S25-PF", "S25-BB", "S25-T")
MEASURED_ROWS: tuple[str, ...] = ("S25-PB", "S25-PFT")
ROWS: tuple[str, ...] = (*FAMILY, *MEASURED_ROWS)
INSTRUMENT_OF: dict[str, str] = {
    "S25-PF": sa.BINANCE, "S25-BB": sa.BYBIT, "S25-T": sa.BYBIT,
    "S25-PB": sa.BYBIT, "S25-PFT": sa.BINANCE,
}  # fmt: skip
#: The rows priced by the side-correct print rule.
PRINT_ROWS: tuple[str, ...] = ("S25-PF", "S25-PB", "S25-PFT")
#: Each instrument's verdict tier and its taker fee.
VERDICT_TIER: dict[str, str] = {sa.BINANCE: "regular", sa.BYBIT: "non_vip"}
VERDICT_TAKER_BP = 10.0

#: The information set: the three end dates the owner may confirm.
INFORMATION_SET_KEYS: tuple[str, ...] = (
    "s25_coordinates", sa.INSTRUMENTS[sa.BINANCE].key, sa.INSTRUMENTS[sa.BYBIT].key,
)  # fmt: skip
NEVER = "never"
FALLBACK_MARK = "H inside S25's information set"
#: The fee amendment's intervals start at or after the day after H.
P_INTERVALS_FROM = datetime.combine(sa.BLOCKS["H"][1] + timedelta(days=1), time(), tzinfo=UTC)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.\-]+):(?=\s|$)")


def amendments_text(document: str) -> str:
    """The document's ``## Amendments`` section (empty when it has none)."""
    _, found, rest = document.partition("\n" + AMENDMENTS_HEADING + "\n")
    return rest if found else ""


def latest_null_sha256(document: str) -> str:
    """The null-placeholder sha256 the loader accepts: the latest amendment's, or
    the registered one; a superseded hash is refused."""
    found = NULL_HASH_PATTERN.findall(amendments_text(document))
    latest = str(found[-1]) if found else REGISTERED_SHA256
    if latest in SUPERSEDED_SHA256:
        raise PreRegistrationError(f"the amendment names a superseded hash {latest}")
    return latest


# ---------------------------------------------------------------------------
# The information set
# ---------------------------------------------------------------------------


def _check_confirmed(confirmed: Any) -> None:
    if confirmed is None:
        return
    if not isinstance(confirmed, Mapping) or set(confirmed) != set(INFORMATION_SET_KEYS):
        raise PreRegistrationError(
            f"information_set.confirmed holds the end dates {list(INFORMATION_SET_KEYS)}"
        )
    for key, value in confirmed.items():
        if not (value == NEVER or (isinstance(value, date) and not isinstance(value, datetime))):
            raise PreRegistrationError(f"information_set.confirmed.{key} is a date or {NEVER!r}")


def h_is_blind(confirmed: Mapping[str, Any] | None, first_day: date = sa.BLOCKS["H"][0]) -> bool:
    """H is blind only if every confirmed end date falls before H's first day,
    ``never`` counting as before; without a confirmation the fallback holds."""
    _check_confirmed(confirmed)
    if confirmed is None:
        return False
    return all(value == NEVER or value < first_day for value in confirmed.values())


def with_fallback(status: str, blind: bool) -> str:
    """A status as the fallback writes it: unchanged when H is blind, else carrying
    "(H inside S25's information set)"."""
    return status if blind else f"{status} ({FALLBACK_MARK})"


# ---------------------------------------------------------------------------
# The fee amendment's intervals
# ---------------------------------------------------------------------------


def _moment(value: Any, where: str) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise PreRegistrationError(f"{where} is not a UTC time: {value!r}") from exc
    if moment.tzinfo is None:
        raise PreRegistrationError(f"{where} must carry its time zone (UTC)")
    return moment


def interval_lines(text: str) -> list[tuple[int, str, dict[str, Any]]]:
    """Every list item under ``fees.intervals.<instrument>``: (line number, the
    instrument's key, the interval). Each interval must sit on one line."""
    stack: list[tuple[int, str]] = []
    found: list[tuple[int, str, dict[str, Any]]] = []
    for number, line in enumerate(text.split("\n")):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stripped.startswith("- "):
            path = tuple(key for _, key in stack)
            if len(path) == 3 and path[:2] == ("fees", "intervals"):
                item = yaml.safe_load(stripped[2:])
                if not isinstance(item, dict) or not {"start", "end", "tiers"} <= set(item):
                    raise PreRegistrationError(f"line {number + 1}: an interval on one line")
                found.append((number, path[2], item))
            continue
        match = _KEY_LINE.match(line)
        if match is not None:
            stack.append((indent, match.group("key")))
    return found


def strip_p_intervals(text: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """The text without the fee amendment's intervals, and those intervals."""
    lines = text.split("\n")
    removed: list[tuple[str, dict[str, Any]]] = []
    drop: set[int] = set()
    for number, key, item in interval_lines(text):
        if _moment(item["start"], f"fees.intervals.{key}.start") >= P_INTERVALS_FROM:
            drop.add(number)
            removed.append((key, item))
    return "\n".join(x for i, x in enumerate(lines) if i not in drop), removed


def _check_p_intervals(added: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    keys = {i.key: i.id for i in sa.INSTRUMENTS.values()}
    spans: dict[str, list[tuple[datetime, datetime]]] = {}
    for key, item in added:
        where = f"fees.intervals.{key} (the fee amendment)"
        if key not in keys:
            raise PreRegistrationError(f"{where}: not an instrument of the study")
        start, end = _moment(item["start"], where), _moment(item["end"], where)
        tiers = item["tiers"]
        tier = VERDICT_TIER[keys[key]]
        if not end > start:
            raise PreRegistrationError(f"{where}: the interval ends before it starts")
        if not isinstance(item.get("sources"), list) or not item["sources"]:
            raise PreRegistrationError(f"{where}: an interval names its sources")
        pair = tiers.get(tier) if isinstance(tiers, Mapping) else None
        if not (isinstance(pair, list) and len(pair) == 2 and all(_is_number(v) for v in pair)):
            raise PreRegistrationError(f"{where}: the verdict tier {tier} as [maker_bp, taker_bp]")
        for lo, hi in spans.get(key, []):
            if start < hi and lo < end:
                raise PreRegistrationError(f"{where}: intervals overlap")
        spans.setdefault(key, []).append((start, end))


def covered(intervals: Sequence[Mapping[str, Any]], tier: str, day: date) -> bool:
    """Every instant of ``day`` lies in an interval that carries ``tier``."""
    spans = sorted(
        (_moment(i["start"], "start"), _moment(i["end"], "end"))
        for i in intervals
        if tier in i["tiers"]
    )
    reached = datetime.combine(day, time(), tzinfo=UTC)
    for lo, hi in spans:
        if lo <= reached < hi:
            reached = hi
    return reached >= datetime.combine(day, time(), tzinfo=UTC) + timedelta(days=1)


# ---------------------------------------------------------------------------
# What each placeholder may hold once frozen
# ---------------------------------------------------------------------------


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


def _is_number(value: Any) -> bool:
    ok = isinstance(value, int | float) and not isinstance(value, bool)
    return ok and math.isfinite(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _mapping(value: Any, where: str, *, keys: Sequence[str] | None = None) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and len(value) > 0, f"{where} must be a non-empty mapping")
    if keys is not None:
        outside = sorted(set(value) - set(keys))
        _require(not outside, f"{where} has keys {outside} outside {list(keys)}")
    return value


def _hashes(value: Any, where: str, *, keys: Sequence[str] | None = None) -> None:
    for key, digest in _mapping(value, where, keys=keys).items():
        _require(_is_sha256(digest), f"{where}.{key} must be a sha256 in hex")


_INSTRUMENT_KEYS = tuple(i.key for i in sa.INSTRUMENTS.values())


def _reader_test(value: Any, where: str) -> None:
    """{instrument: {day: {status, stream, findings, passes}}} on the three days."""
    wanted = {d.isoformat() for d in sa.READER_TEST_DAYS}
    for key, days in _mapping(value, where, keys=_INSTRUMENT_KEYS).items():
        found = _mapping(days, f"{where}.{key}")
        _require(
            {str(d) for d in found} == wanted, f"{where}.{key} holds the days {sorted(wanted)}"
        )
        for day, entry in found.items():
            _require(
                isinstance(entry, Mapping)
                and {"status", "stream", "findings", "passes"} <= set(entry)
                and isinstance(entry["passes"], bool),
                f"{where}.{key}.{day} records its status, stream, findings and passes",
            )


def _excluded_days(value: Any, where: str) -> None:
    """{instrument: [[day, reason]]}, each day one of W or D."""
    development = set(sa.block_days("W")) | set(sa.block_days("D"))
    for key, items in _mapping(value, where, keys=_INSTRUMENT_KEYS).items():
        _require(isinstance(items, list), f"{where}.{key} is a list of [day, reason]")
        for item in items:
            _require(
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], date)
                and item[0] in development
                and isinstance(item[1], str),
                f"{where}.{key} holds [day of W or D, reason], not {item!r}",
            )


def _instrument_spec(value: Any, where: str) -> None:
    specs = _mapping(value, where, keys=_INSTRUMENT_KEYS)
    _require(set(specs) == set(_INSTRUMENT_KEYS), f"{where} records both instruments")
    for key, spec in specs.items():
        _require(
            isinstance(spec, Mapping)
            and set(spec) == {"tick", "lot"}
            and all(_is_number(spec[k]) and spec[k] > 0 for k in ("tick", "lot")),
            f"{where}.{key} records a positive tick and lot",
        )


def _thresholds(value: Any, where: str) -> None:
    """{row: {cell: {theta_long, theta_short}}}; S25-T's are the 2025 study's."""
    fitted = tuple(r for r in ROWS if r != "S25-T")
    for row, cells in _mapping(value, where, keys=fitted).items():
        for cell, theta in _mapping(cells, f"{where}.{row}").items():
            _require(
                isinstance(theta, Mapping)
                and set(theta) == {"theta_long", "theta_short"}
                and all(_is_number(v) for v in theta.values()),
                f"{where}.{row}.{cell} records theta_long and theta_short",
            )


def _per_slot(value: Any, where: str) -> None:
    for slot, number in _mapping(value, where, keys=FAMILY).items():
        _require(_is_number(number) and number >= 0, f"{where}.{slot} must be a number at least 0")


_COMMIT = re.compile(r"[0-9a-f]{7,40}")


def _compute_record(value: Any, where: str) -> None:
    """Any record, with the run log's sha256 and row count under ``trials`` and the
    code-freeze commit stage 1 ran under at ``code_freeze.commit``."""
    freeze = _mapping(_mapping(value, where).get("code_freeze"), f"{where}.code_freeze")
    commit = freeze.get("commit")
    _require(isinstance(commit, str) and _COMMIT.fullmatch(commit) is not None,
             f"{where}.code_freeze.commit must be a commit in hex")  # fmt: skip
    trials = _mapping(_mapping(value, where).get("trials"), f"{where}.trials")
    _require(_is_sha256(trials.get("sha256")), f"{where}.trials.sha256 must be a sha256")
    rows = trials.get("rows")
    _require(
        isinstance(rows, int) and not isinstance(rows, bool) and rows >= 0, f"{where}.trials.rows"
    )


#: How each placeholder's frozen value is checked.
FROZEN_KINDS: dict[tuple[str, ...], Any] = {
    ("data", "reader_test_result"): _reader_test,
    ("data", "max_sequence_gaps"): lambda v, w: _require(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0, f"{w} is a count at least 0"
    ),
    ("data", "excluded_days_D"): _excluded_days,
    ("data", "instrument_spec"): _instrument_spec,
    ("common", "book_levels", "walk"): lambda v, w: _require(v in (10, 200), f"{w} is 10 or 200"),
    ("features", "fitted_values_sha256"): lambda v, w: _require(_is_sha256(v), f"{w} is a sha256"),
    ("models", "weights_sha256"): _hashes,
    ("models", "thresholds"): _thresholds,
    ("metrics", "mde", "sigma_d"): _per_slot,
    ("metrics", "mde", "value"): _per_slot,
    ("measurements", "result_D"): lambda v, w: _mapping(v, w),
    ("compute", "record"): _compute_record,
    ("amendment", "date"): lambda v, w: _require(isinstance(v, date), f"{w} must be a date"),
    ("artefacts", "frozen_slot_files_sha256"): lambda v, w: _hashes(v, w, keys=ROWS),
}


# ---------------------------------------------------------------------------
# The registration
# ---------------------------------------------------------------------------


def _validate_registered(tree: Mapping[str, Any]) -> None:
    """The registered values this study's code is built on, unchanged."""
    _require(tuple(tree) == SECTIONS, f"sections differ from the registered {SECTIONS}")
    _require(tree["name"] == "spot_2026_preregistration", "not the spot study's registration")
    _require(tree["searched"] is False, "the study searches nothing")
    for found in sa.INSTRUMENTS.values():
        spec = _get(tree, ("instruments", found.key))
        _require(
            (spec.get("id"), spec.get("venue"), spec.get("symbol"))
            == (found.id, found.venue, found.symbol),
            f"the instrument {found.id} changed",
        )
    _require(len(tree["instruments"]) == len(sa.INSTRUMENTS), "the instruments changed")
    for name, (start, end) in sa.BLOCKS.items():
        block = _get(tree, ("blocks", name))
        _require(
            (block.get("start"), block.get("end"), block.get("days"))
            == (start, end, sa.BLOCK_DAYS[name]),
            f"block {name} differs from the study's constants",
        )
    blocks = tree["blocks"]
    _require(tuple(blocks["reader_test"]["days"]) == sa.READER_TEST_DAYS, "the reader-test days")
    _require(
        tuple(_get(tree, ("data", "reader_test_days"))) == sa.READER_TEST_DAYS, "reader-test days"
    )
    _require(tuple(blocks["P_embargo"]) == sa.P_EMBARGO, "P's embargo changed")
    _require(tuple(blocks["never_in_a_block"]) == sa.NEVER_READ_DAYS, "the days never read changed")
    info = _get(tree, ("information_set",))
    _require(
        info.get("held_out_start") == sa.BLOCKS["H"][0], "H's first day in the information set"
    )
    _require(
        info.get("first_fitted_day") == sa.BLOCKS["D"][0], "D's first day in the information set"
    )
    _check_confirmed(info.get("confirmed"))
    _require(_get(tree, ("common", "clip_btc")) == 0.010, "the clip changed")
    _require(_get(tree, ("common", "latency_ms")) == 10, "the latency changed")
    _require(_get(tree, ("common", "max_position_clips")) == 1, "one position at a time")
    _require(tuple(tree["family"]) == FAMILY, f"the family is {FAMILY}")
    _require(
        tuple(tree["measured_rows"])[: len(MEASURED_ROWS)] == MEASURED_ROWS, "the measured rows"
    )
    for row, name in INSTRUMENT_OF.items():
        where = ("slots", row) if row in FAMILY else ("measured_rows", row)
        key = _get(tree, (*where, "instrument"))
        _require(key == sa.INSTRUMENTS[name].key, f"{row} runs on {name}")
    tiers = _get(tree, ("fees", "verdict_tier"))
    _require(
        tiers == {sa.INSTRUMENTS[n].key: tier for n, tier in VERDICT_TIER.items()}, "verdict tiers"
    )
    s25_t = _get(tree, ("slots", "S25-T"))
    _require(s25_t.get("config_2025_sha256") == CONFIG_2025_SHA256, "the 2025 configuration's hash")
    _require(s25_t.get("frozen_files_2025") == FROZEN_FILES_2025, "the 2025 frozen files' hashes")
    _require(s25_t.get("frozen_paths_2025") == FROZEN_PATHS_2025, "the 2025 frozen files' paths")
    _require(
        s25_t.get("neighbourhood_fits_2025") == NEIGHBOURHOOD_FITS_2025,
        "the 2025 neighbourhood fits",
    )
    _require(_get(tree, ("access", "ledger")) == LEDGER_PATH.as_posix(), "the ledger's path")
    files = [LEDGER_PATH.as_posix(), *(p.as_posix() for p in RESULTS_PATHS.values())]
    _require(_get(tree, ("artefacts", "files")) == files, "the artefacts' paths changed")
    _validate_rules(tree)


def _validate_rules(tree: Mapping[str, Any]) -> None:
    """The status rules' numbers, as :mod:`trading_research.spot.statuses` applies them."""
    from trading_research.spot import statuses as st

    holm = _get(tree, ("statuses", "holm"))
    _require(
        (holm["alpha"], holm["m"], holm["sided"]) == (st.ALPHA, st.M, "one"), "Holm's settings"
    )
    caps = _get(tree, ("statuses", "caps"))
    _require(caps["fill_beyond_sweep"]["share_of_fills_above"] == st.FILL_BEYOND_SWEEP_CAP, "cap")
    _require(caps["capacity"]["median_day_ratio_above"] == st.CAPACITY_CAP, "the capacity cap")
    kills = _get(tree, ("kill_conditions",))
    _require(kills["status_only"]["K-trades"] == st.MIN_TRADES, "K-trades")
    _require(kills["K-fee"]["taker_bp"] == VERDICT_TAKER_BP, "K-fee's taker fee")
    _require(kills["K-days_share_positive_max"] == st.K_DAYS_SHARE, "K-days")
    _require(kills["K-latency"]["latency_ms"] == st.LATENCY_RERUN_MS, "K-latency")
    _require(tuple(kills["void"]) == st.VOID_CONDITIONS, "the void conditions")
    for row in FAMILY:
        _require(tuple(kills["per_slot"][row]["kills"]) == st.KILL_ORDER[row], f"{row}'s kills")
    _require(tree["placebo_percentile"] == st.PLACEBO_PERCENTILE, "the placebo percentile")
    draws = _get(tree, ("placebos", "random_direction"))
    _require(
        (draws["draws"], draws["seeds"]) == (st.RANDOM_DRAWS, [0, st.RANDOM_DRAWS - 1]), "draws"
    )
    _require(_get(tree, ("metrics", "newey_west", "lag_days")) == st.NEWEY_WEST_LAG, "the lag")


@dataclass(frozen=True)
class Registration:
    """The study's YAML, validated against its registration.

    ``text`` is the file as read; ``base_text`` the same without the fee
    amendment's intervals (equal to ``text`` until that amendment), and
    ``p_intervals`` those intervals.
    """

    text: str
    tree: Mapping[str, Any]
    base_text: str
    p_intervals: tuple[tuple[str, Mapping[str, Any]], ...] = ()

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode())

    @property
    def base_sha256(self) -> str:
        """The sha256 of the configuration without P's intervals: H's configuration."""
        return sha256_bytes(self.base_text.encode())

    @classmethod
    def load(
        cls,
        config: Path | str = CONFIG_PATH,
        document: Path | str = DOCUMENT_PATH,
        *,
        repo: Path | str = Path(),
    ) -> Registration:
        """Read and validate, accepting the null-placeholder hash the latest amendment records."""
        doc = (Path(repo) / document).read_text(encoding="utf-8")
        text = (Path(repo) / config).read_text(encoding="utf-8")
        return cls.from_text(text, null_sha256=latest_null_sha256(doc))

    @classmethod
    def from_text(cls, text: str, *, null_sha256: str = REGISTERED_SHA256) -> Registration:
        """Parse and validate; raise :class:`PreRegistrationError` on any departure."""
        if null_sha256 in SUPERSEDED_SHA256:
            raise PreRegistrationError(f"{null_sha256} is a superseded hash")
        base, added = strip_p_intervals(text)
        where = placeholder_lines(base)
        if set(where) != set(PLACEHOLDERS):
            raise PreRegistrationError(
                f"placeholders differ from the registered ones: missing "
                f"{sorted(set(PLACEHOLDERS) - set(where))}, unexpected "
                f"{sorted(set(where) - set(PLACEHOLDERS))}"
            )
        original = sha256_bytes(unfrozen_text(base).encode())
        if original != null_sha256:
            raise PreRegistrationError(
                "the file differs from the registered one outside its placeholders "
                f"(sha256 with placeholders reset is {original}, accepted {null_sha256})"
            )
        tree = yaml.safe_load(text)
        _require(isinstance(tree, dict), "the registration must be a mapping")
        _validate_registered(tree)
        registration = cls(text, tree, base, tuple(added))
        for path in PLACEHOLDERS:
            value = registration.value(path)
            if value is not None:
                FROZEN_KINDS[path](value, ".".join(path))
        if added:
            registration.require_frozen()  # the fee amendment comes after H, so after D's
            _check_p_intervals(added)
        return registration

    def value(self, path: Sequence[str]) -> Any:
        return _get(self.tree, path)

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
        """Null everywhere (development) or frozen everywhere (after the amendment)."""
        filled = [self.value(p) is not None for p in PLACEHOLDERS]
        if any(filled) and not all(filled):
            raise PreRegistrationError("the registration is partly frozen")
        return self

    # -- the information set ---------------------------------------------------

    @property
    def confirmed(self) -> dict[str, Any] | None:
        found = self.value(("information_set", "confirmed"))
        return None if found is None else dict(found)

    @property
    def h_blind(self) -> bool:
        """Whether H is a blind block; False means the fallback holds."""
        return h_is_blind(self.confirmed)

    # -- fees --------------------------------------------------------------------

    def fee_intervals(self, name: str) -> list[dict[str, Any]]:
        """The sourced intervals of one instrument, P's included once amended."""
        return [dict(i) for i in self.value(("fees", "intervals", sa.instrument(name).key))]

    def fee_covered(self, name: str, day: date) -> bool:
        """Every instant of ``day`` lies in a sourced interval at the verdict tier."""
        return covered(self.fee_intervals(name), VERDICT_TIER[name], day)

    # -- the frozen values (None until the D amendment) ----------------------------

    @property
    def max_sequence_gaps(self) -> int | None:
        value = self.value(("data", "max_sequence_gaps"))
        return None if value is None else int(value)

    @property
    def instrument_spec(self) -> dict[str, tuple[float, float]] | None:
        """{instrument id: (tick, lot)} from each venue's spot specification."""
        spec = self.value(("data", "instrument_spec"))
        if spec is None:
            return None
        ids = {i.key: i.id for i in sa.INSTRUMENTS.values()}
        return {ids[k]: (float(v["tick"]), float(v["lot"])) for k, v in spec.items()}

    @property
    def walk_levels(self) -> int | None:
        value = self.value(("common", "book_levels", "walk"))
        return None if value is None else int(value)

    @property
    def weights_sha256(self) -> dict[str, str]:
        return dict(self.value(("models", "weights_sha256")) or {})

    @property
    def thresholds(self) -> dict[str, dict[str, dict[str, float]]]:
        return {k: dict(v) for k, v in (self.value(("models", "thresholds")) or {}).items()}

    @property
    def frozen_slot_files_sha256(self) -> dict[str, str]:
        return dict(self.value(("artefacts", "frozen_slot_files_sha256")) or {})

    @property
    def fitted_values_sha256(self) -> str | None:
        value = self.value(("features", "fitted_values_sha256"))
        return None if value is None else str(value)

    @property
    def sigma_d(self) -> dict[str, float]:
        return {k: float(v) for k, v in (self.value(("metrics", "mde", "sigma_d")) or {}).items()}

    @property
    def mde(self) -> dict[str, float]:
        """{slot: MDE} on H, as frozen."""
        return {k: float(v) for k, v in (self.value(("metrics", "mde", "value")) or {}).items()}

    @property
    def code_freeze(self) -> str | None:
        """The code-freeze commit stage 1 ran under, as the compute record holds it."""
        record = self.value(("compute", "record"))
        return None if record is None else str(record["code_freeze"]["commit"])

    @property
    def trials(self) -> dict[str, Any] | None:
        """The run log's sha256 and row count, as the compute record holds them."""
        record = self.value(("compute", "record"))
        return None if record is None else dict(record["trials"])


_PLACEHOLDER = re.compile(
    r"^(?P<head>\s*[A-Za-z0-9_.\-]+: )(?P<value>.*?)(?P<tail>  # [^\n]*"
    + re.escape(PLACEHOLDER_MARK)
    + r"[^\n]*)$"
)


def freeze_text(
    text: str, values: Mapping[tuple[str, ...], Any], *, null_sha256: str = REGISTERED_SHA256
) -> str:
    """Write the frozen values into their placeholders, and nothing else.

    Every placeholder must be given exactly once and must currently be ``null``;
    each value is written on its line by ``market_making.prereg.render_value``,
    and the result is validated as a frozen registration before it is returned.
    Its sha256 is what the D amendment records (:func:`frozen_hash_line`).
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
    for path, value in values.items():
        if path not in where:
            raise PreRegistrationError(f"no placeholder at {'.'.join(path)}")
        match = _PLACEHOLDER.match(lines[where[path]])
        if match is None or match.group("value").strip() != "null":
            raise PreRegistrationError(f"{'.'.join(path)} is already frozen")
        lines[where[path]] = match.group("head") + render_value(value) + match.group("tail")
    frozen = "\n".join(lines)
    Registration.from_text(frozen, null_sha256=null_sha256).require_frozen()
    return frozen


def frozen_hash_line(text: str) -> str:
    """The line the D amendment records for the frozen file."""
    return f"Frozen configuration sha256: `{sha256_bytes(text.encode())}`"
