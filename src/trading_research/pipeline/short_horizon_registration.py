"""The 2025 study's registration as code: its YAML validated, its placeholders and hashes.

The study was registered in ``docs/preregistration/short_horizon_2025.md`` with
``configs/short_horizon_2025.yaml``. This module reads that file on the pattern
of :mod:`trading_research.market_making.round2`, and shares none of the earlier
rounds' state: the blocks are :mod:`trading_research.data.study_access`'s.

**Placeholders.** The values chosen on D are ``null`` lines commented "frozen
by the D amendment" (:data:`PLACEHOLDERS`). Putting ``null`` back into every one
of them must give the registered sha256 (:data:`REGISTERED_SHA256`), or the one
the latest amendment records, so nothing outside them can move.
:func:`freeze_text` writes the chosen values and nothing else.

**Amendments.** A stage-0 amendment that cuts the catalogue changes the YAML
outside its placeholders; it records the new hashes in the document's
``## Amendments`` section, each on a line of its own:

* ``Null-placeholder sha256: `<hash>``` (the configuration with every
  placeholder ``null``);
* ``Catalogue sha256: `<hash>``` (the catalogue as cut).

The latest such line is the one the loader accepts. The frozen configuration's
sha256 is recorded by the D amendment as ``Frozen configuration sha256:
`<hash>```, which the ledger reads.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from trading_research.data import study_access as sa
from trading_research.market_making.prereg import (
    _PLACEHOLDER_LINE,
    PreRegistrationError,
    _set_placeholders,
    placeholder_lines,
    sha256_bytes,
    unfrozen_text,
)

CONFIG_PATH = Path("configs/short_horizon_2025.yaml")
DOCUMENT_PATH = Path("docs/preregistration/short_horizon_2025.md")
LEDGER_PATH = Path("experiments/results/short_horizon_2025_ledger.json")
#: The trial ledger's parts (under ``data/``, never committed) and its committed manifest.
TRIALS_PATH = Path("data/short_horizon_2025/short_horizon_2025_trials.parquet")
TRIALS_MANIFEST_PATH = Path("experiments/results/short_horizon_2025_trials.json")
RESULTS_PATHS: dict[str, Path] = {
    block: Path(f"experiments/results/short_horizon_2025_{block}.json") for block in sa.HELD_OUT
}

#: The sha256 of the YAML as registered, every placeholder ``null``.
REGISTERED_SHA256 = "28f63504d7e367615360464100328e01552cec9e0f043904c96118ae87c6920f"
#: The registered catalogue's sha256 (``search.catalogue.sha256``).
REGISTERED_CATALOGUE_SHA256 = "dba3b16a5ffb70ca03974c6a53a9529ba4009cdf796da92c2389b3822288538d"
#: The break-detector library's locked commit (``detector.library.commit``).
LOCKED_LIBRARY = "structural-break"
LOCKED_COMMIT = "0636813fdd4ccc9f25490d97f62d6fccecb01881"

#: Lines an amendment writes, and the one the D amendment writes.
NULL_HASH_PATTERN = re.compile(r"Null-placeholder sha256: `([0-9a-f]{64})`")
CATALOGUE_HASH_PATTERN = re.compile(r"Catalogue sha256: `([0-9a-f]{64})`")
AMENDMENTS_HEADING = "## Amendments"

#: Every placeholder the D amendment fills, as a path of keys, in file order.
PLACEHOLDERS: tuple[tuple[str, ...], ...] = (
    ("data", "max_sequence_gaps"),
    ("data", "excluded_days_D"),
    ("data", "instrument_spec"),
    ("data", "ob200_reader_test_result"),
    ("features", "selected"),
    ("features", "fitted_values_sha256"),
    ("models", "weights_sha256"),
    ("models", "checkpoint_epoch"),
    ("search", "result_trials"),
    ("search", "result_choice"),
    ("search", "result_admission"),
    ("search", "result_nulls"),
    ("slots", "S25", "thresholds"),
    ("slots", "S-C", "thresholds"),
    ("metrics", "mde", "sigma_d"),
    ("metrics", "mde", "value"),
    ("measurements", "result_O"),
    ("detector", "flags_D_foldwise"),
    ("detector", "hour_profile"),
    ("detector", "alarm_rate_D"),
    ("detector", "alarms_D"),
    ("detector", "r_untestable_declared"),
    ("compute", "record"),
    ("amendment", "date"),
    ("artefacts", "frozen_finalist_files_sha256"),
)

#: Top-level sections of the registered YAML, in order.
SECTIONS: tuple[str, ...] = (
    "name", "registered", "registered_against", "document", "registers", "venue", "product",
    "instruments", "computed_before_registration", "prior_information", "expected_outcome",
    "data", "blocks", "common", "labels", "features", "models", "search", "d_procedure",
    "slots", "metrics", "family", "statuses", "kill_conditions", "placebo_percentile",
    "placebos", "hypotheses", "measurements", "detector", "fetch", "compute", "access",
    "amendment", "artefacts",
)  # fmt: skip

#: The slots of the family, and S25-Z beside it.
FAMILY: tuple[str, ...] = ("S25", "S-C", "S-R", "S-P", "S-M")
SLOTS: tuple[str, ...] = (*FAMILY, "S25-Z")
#: D's walk-forward test months, as the fold-wise flags are keyed.
TEST_MONTHS: tuple[str, ...] = ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06")
#: The keys a frozen ``search.result_trials`` carries.
RESULT_TRIALS_KEYS: frozenset[str] = frozenset(
    {"N_trials", "N_eff", "R_D", "ledger_sha256", "ledger_rows", "hand_runs"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def amendments_text(document: str) -> str:
    """The document's ``## Amendments`` section (empty when it has none)."""
    _, found, rest = document.partition("\n" + AMENDMENTS_HEADING + "\n")
    return rest if found else ""


def _latest(pattern: re.Pattern[str], document: str, registered: str) -> str:
    found = pattern.findall(amendments_text(document))
    return str(found[-1]) if found else registered


def latest_null_sha256(document: str) -> str:
    """The null-placeholder sha256 the loader accepts: the latest amendment's, or the registered."""
    return _latest(NULL_HASH_PATTERN, document, REGISTERED_SHA256)


def latest_catalogue_sha256(document: str) -> str:
    """The catalogue sha256 in force: the latest amendment's, or the registered one."""
    return _latest(CATALOGUE_HASH_PATTERN, document, REGISTERED_CATALOGUE_SHA256)


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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _non_negative_int(value: Any, where: str) -> None:
    _require(_is_int(value) and value >= 0, f"{where} must be a whole number at least 0")


def _positive_int(value: Any, where: str) -> None:
    _require(_is_int(value) and value > 0, f"{where} must be a positive whole number")


def _non_negative(value: Any, where: str) -> None:
    _require(_is_number(value) and value >= 0, f"{where} must be a number at least 0")


def _mapping(value: Any, where: str, *, keys: Sequence[str] | None = None) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and len(value) > 0, f"{where} must be a non-empty mapping")
    if keys is not None:
        outside = sorted(set(value) - set(keys))
        _require(not outside, f"{where} has keys {outside} outside {list(keys)}")
    return value


def _hashes(value: Any, where: str, *, keys: Sequence[str] | None = None) -> None:
    for key, digest in _mapping(value, where, keys=keys).items():
        _require(_is_sha256(digest), f"{where}.{key} must be a sha256 in hex")


def _excluded_days(value: Any, where: str) -> None:
    _require(isinstance(value, list), f"{where} is a list of [day, reason]")
    development = set(sa.block_days("W")) | set(sa.block_days("D"))
    for item in value:
        _require(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], date)
            and item[0] in development
            and isinstance(item[1], str),
            f"{where} holds [day of W or D, reason], not {item!r}",
        )


def _instrument_spec(value: Any, where: str) -> None:
    spec = _mapping(value, where, keys=("tick", "lot"))
    _require(set(spec) == {"tick", "lot"}, f"{where} records the tick and the lot")
    for key in ("tick", "lot"):
        _require(_is_number(spec[key]) and spec[key] > 0, f"{where}.{key} must be positive")


def _selected(value: Any, where: str) -> None:
    _require(
        isinstance(value, list)
        and 0 < len(value) <= 40
        and all(isinstance(v, str) for v in value)
        and len(set(value)) == len(value),
        f"{where} is a list of at most 40 distinct feature names",
    )


def _possibly_empty(value: Any, where: str) -> Mapping[str, Any]:
    """A mapping that may be empty: nothing of its kind was frozen (no network fit
    when R* is the bound regressor, no C* when stratum C has no admissible cell)."""
    _require(isinstance(value, Mapping), f"{where} must be a mapping")
    return value


def _epochs(value: Any, where: str) -> None:
    for key, epoch in _possibly_empty(value, where).items():
        _require(_is_int(epoch) and 0 <= epoch <= 20, f"{where}.{key} is an epoch from 0 to 20")


def _result_trials(value: Any, where: str) -> None:
    trials = _mapping(value, where)
    _require(set(trials) == RESULT_TRIALS_KEYS, f"{where} records {sorted(RESULT_TRIALS_KEYS)}")
    _positive_int(trials["N_trials"], f"{where}.N_trials")
    _positive_int(trials["R_D"], f"{where}.R_D")
    _non_negative_int(trials["ledger_rows"], f"{where}.ledger_rows")
    _require(_is_number(trials["N_eff"]) and trials["N_eff"] > 0, f"{where}.N_eff > 0")
    _require(_is_sha256(trials["ledger_sha256"]), f"{where}.ledger_sha256 must be a sha256")
    _require(isinstance(trials["hand_runs"], list), f"{where}.hand_runs is a list")


def _thresholds(value: Any, where: str) -> None:
    for cell, theta in _possibly_empty(value, where).items():
        _require(
            isinstance(theta, Mapping)
            and set(theta) == {"theta_long", "theta_short"}
            and all(_is_number(v) for v in theta.values()),
            f"{where}.{cell} records theta_long and theta_short",
        )


def _per_slot_numbers(value: Any, where: str) -> None:
    for slot, number in _mapping(value, where, keys=SLOTS).items():
        _non_negative(number, f"{where}.{slot}")


def _flags_foldwise(value: Any, where: str) -> None:
    for month, flags in _mapping(value, where).items():
        _require(str(month) in TEST_MONTHS, f"{where}: {month!r} is not a test month of D")
        _require(isinstance(flags, list), f"{where}.{month} is a list of flag times")


def _hour_profile(value: Any, where: str) -> None:
    _require(
        isinstance(value, list)
        and len(value) == 24
        and all(_is_number(v) and v > 0 for v in value),
        f"{where} holds 24 positive values, one per UTC hour",
    )


def _alarms(value: Any, where: str) -> None:
    _require(isinstance(value, list), f"{where} is a list of alarms")
    for alarm in value:
        _require(
            isinstance(alarm, Mapping) and {"timestamp", "channel"} <= set(alarm),
            f"{where} records each alarm's timestamp and channel",
        )


def _reader_test(value: Any, where: str) -> None:
    """The ``ob200`` reader test: every reader-test day with its exclusions and whether
    it passed, or failed on its gap count alone; D's statistics beside them."""
    result = _mapping(value, where)
    days = _mapping(result.get("days"), f"{where}.days")
    wanted = {d.isoformat() for d in sa.READER_TEST_DAYS}
    _require(
        {str(d) for d in days} == wanted, f"{where}.days are the reader-test days {sorted(wanted)}"
    )
    for day, entry in days.items():
        _require(
            isinstance(entry, Mapping)
            and isinstance(entry.get("passes"), bool)
            and isinstance(entry.get("fails_on_gaps_alone"), bool)
            and isinstance(entry.get("exclusions"), list),
            f"{where}.days.{day} records its exclusions, passes and fails_on_gaps_alone",
        )
    _mapping(result.get("D"), f"{where}.D")


def _boolean(value: Any, where: str) -> None:
    _require(isinstance(value, bool), f"{where} must be true or false")


def _a_date(value: Any, where: str) -> None:
    _require(isinstance(value, date), f"{where} must be a date")


def _any_mapping(value: Any, where: str) -> None:
    _mapping(value, where)


#: How each placeholder's frozen value is checked.
FROZEN_KINDS: dict[tuple[str, ...], Any] = {
    ("data", "max_sequence_gaps"): _non_negative_int,
    ("data", "excluded_days_D"): _excluded_days,
    ("data", "instrument_spec"): _instrument_spec,
    ("data", "ob200_reader_test_result"): _reader_test,
    ("features", "selected"): _selected,
    ("features", "fitted_values_sha256"): lambda v, w: _require(_is_sha256(v), f"{w} is a sha256"),
    ("models", "weights_sha256"): _hashes,
    ("models", "checkpoint_epoch"): _epochs,
    ("search", "result_trials"): _result_trials,
    ("search", "result_choice"): lambda v, w: _mapping(v, w, keys=("C*", "R*")),
    ("search", "result_admission"): lambda v, w: _mapping(v, w, keys=SLOTS),
    ("search", "result_nulls"): _any_mapping,
    ("slots", "S25", "thresholds"): _thresholds,
    ("slots", "S-C", "thresholds"): _thresholds,
    ("metrics", "mde", "sigma_d"): _per_slot_numbers,
    ("metrics", "mde", "value"): _per_slot_numbers,
    ("measurements", "result_O"): _any_mapping,
    ("detector", "flags_D_foldwise"): _flags_foldwise,
    ("detector", "hour_profile"): _hour_profile,
    ("detector", "alarm_rate_D"): _non_negative,
    ("detector", "alarms_D"): _alarms,
    ("detector", "r_untestable_declared"): _boolean,
    ("compute", "record"): _any_mapping,
    ("amendment", "date"): _a_date,
    ("artefacts", "frozen_finalist_files_sha256"): lambda v, w: _hashes(v, w, keys=SLOTS),
}


# ---------------------------------------------------------------------------
# The registration
# ---------------------------------------------------------------------------


def _validate_registered(tree: Mapping[str, Any], catalogue_sha256: str) -> None:
    """The registered values the study's code is built on, unchanged."""
    from trading_research.data.study_loader import CLIP
    from trading_research.pipeline.regime import SEARCH_POLICIES

    _require(tuple(tree) == SECTIONS, f"sections differ from the registered {SECTIONS}")
    _require(tree["name"] == "short_horizon_2025_preregistration", "not the 2025 study")
    _require(tuple(tree["instruments"]) == sa.INSTRUMENTS, "the instruments changed")
    for name, (start, end) in sa.BLOCKS.items():
        block = _get(tree, ("blocks", name))
        _require(
            (block.get("start"), block.get("end"), block.get("days"))
            == (start, end, sa.BLOCK_DAYS[name]),
            f"block {name} differs from the study's constants",
        )
    o = _get(tree, ("blocks", "O"))
    _require((o.get("start"), o.get("end")) == sa.O_WINDOW, "the O window changed")
    _require(tuple(_get(tree, ("blocks", "cold_start_days"))) == sa.COLD_START_DAYS, "cold starts")
    book = _get(tree, ("data", "book"))
    _require(tuple(book["ob200_reader_test_days"]) == sa.READER_TEST_DAYS, "reader-test days")
    _require(book["archive_change"] == sa.ARCHIVE_CHANGE, "the archive change")
    _require(_get(tree, ("common", "clip_btc")) == CLIP["BTCUSDT"], "the clip changed")
    _require(_get(tree, ("common", "latency_ms")) == 10, "the latency changed")
    regime = _get(tree, ("search", "axes", "regime", "values"))
    _require(tuple(regime) == SEARCH_POLICIES, "the regime axis changed")
    _require(
        _get(tree, ("search", "catalogue", "sha256")) == catalogue_sha256,
        f"the catalogue sha256 is not the registered or amended one ({catalogue_sha256})",
    )
    _require(_get(tree, ("access", "ledger")) == LEDGER_PATH.as_posix(), "the ledger's path")
    results = _get(tree, ("access", "results_files"))
    _require(
        results == {block: path.as_posix() for block, path in RESULTS_PATHS.items()},
        "the results files' paths changed",
    )
    trials = _get(tree, ("search", "trial_ledger"))
    _require(trials["file"] == TRIALS_PATH.as_posix(), "the trial ledger's path")
    _require(
        str(trials["manifest"]).startswith(TRIALS_MANIFEST_PATH.as_posix() + ":"),
        "the trial ledger's manifest path",
    )
    library = _get(tree, ("detector", "library"))
    _require(
        library.get("name") == LOCKED_LIBRARY and library.get("commit") == LOCKED_COMMIT,
        "the detector library's locked commit changed",
    )


@dataclass(frozen=True)
class Registration:
    """The study's YAML, validated against its registration."""

    text: str
    tree: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode())

    @classmethod
    def load(
        cls,
        config: Path | str = CONFIG_PATH,
        document: Path | str = DOCUMENT_PATH,
        *,
        repo: Path | str = Path(),
    ) -> Registration:
        """Read and validate, accepting the hashes the latest amendment records."""
        doc = (Path(repo) / document).read_text(encoding="utf-8")
        return cls.from_text(
            (Path(repo) / config).read_text(encoding="utf-8"),
            null_sha256=latest_null_sha256(doc),
            catalogue_sha256=latest_catalogue_sha256(doc),
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        null_sha256: str = REGISTERED_SHA256,
        catalogue_sha256: str = REGISTERED_CATALOGUE_SHA256,
    ) -> Registration:
        """Parse and validate; raise :class:`PreRegistrationError` on any departure."""
        where = placeholder_lines(text)
        if set(where) != set(PLACEHOLDERS):
            raise PreRegistrationError(
                f"placeholders differ from the registered ones: missing "
                f"{sorted(set(PLACEHOLDERS) - set(where))}, unexpected "
                f"{sorted(set(where) - set(PLACEHOLDERS))}"
            )
        original = sha256_bytes(unfrozen_text(text).encode())
        if original != null_sha256:
            raise PreRegistrationError(
                "the file differs from the registered one outside its placeholders "
                f"(sha256 with placeholders reset is {original}, accepted {null_sha256})"
            )
        tree = yaml.safe_load(text)
        _require(isinstance(tree, dict), "the registration must be a mapping")
        _validate_registered(tree, catalogue_sha256)
        registration = cls(text, tree)
        for path in PLACEHOLDERS:
            value = registration.value(path)
            if value is not None:
                FROZEN_KINDS[path](value, ".".join(path))
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

    # -- reading values -------------------------------------------------------

    @property
    def catalogue_sha256(self) -> str:
        return str(self.value(("search", "catalogue", "sha256")))

    @property
    def max_sequence_gaps(self) -> int | None:
        """D's frozen threshold; None until the amendment."""
        value = self.value(("data", "max_sequence_gaps"))
        return None if value is None else int(value)

    @property
    def instrument_spec(self) -> tuple[float, float] | None:
        """(tick, lot) from the venue's specification; None until the amendment."""
        spec = self.value(("data", "instrument_spec"))
        return None if spec is None else (float(spec["tick"]), float(spec["lot"]))

    @property
    def weights_sha256(self) -> dict[str, str]:
        return dict(self.value(("models", "weights_sha256")) or {})

    @property
    def finalist_files_sha256(self) -> dict[str, str]:
        return dict(self.value(("artefacts", "frozen_finalist_files_sha256")) or {})

    @property
    def fitted_values_sha256(self) -> str | None:
        value = self.value(("features", "fitted_values_sha256"))
        return None if value is None else str(value)

    @property
    def result_trials(self) -> dict[str, Any] | None:
        value = self.value(("search", "result_trials"))
        return None if value is None else dict(value)


def freeze_text(
    text: str,
    values: Mapping[tuple[str, ...], Any],
    *,
    null_sha256: str = REGISTERED_SHA256,
    catalogue_sha256: str = REGISTERED_CATALOGUE_SHA256,
) -> str:
    """Write the chosen values into their placeholders, and nothing else.

    Every placeholder must be given exactly once and must currently be
    ``null``; the result is validated as a frozen registration before it is
    returned. Its sha256 is what the D amendment records.
    """
    from trading_research.market_making.round2 import render

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
    Registration.from_text(
        frozen, null_sha256=null_sha256, catalogue_sha256=catalogue_sha256
    ).require_frozen()
    return frozen


def frozen_hash_line(text: str) -> str:
    """The line the D amendment records for the frozen file."""
    return f"Frozen configuration sha256: `{sha256_bytes(text.encode())}`"
