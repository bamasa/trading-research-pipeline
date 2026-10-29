"""The trial ledger: one row per D run, cell and test day, with its manifest.

Every score on a test day of February to June comes from the entry point and
is written here: ``data/short_horizon_2025/short_horizon_2025_trials.parquet``
(zstd), columns ``run``, ``key``, ``day``, ``net_bp``, ``G_bp``,
``pnl_long_bp``, ``pnl_short_bp``, ``trades`` and ``skips``, split into parts
under 100 MB when it needs to be. The parts stay under ``data/`` and are never
committed. The manifest ``experiments/results/short_horizon_2025_trials.json``,
small and committed, holds the catalogue's sha256, each run's commit, every
part's sha256 and row count, the sha256 of all the parts' bytes in order with
the total row count (what the study ledger checks the parts on disk against
before H opens), the number of trials and the hand-run log.

The ledger only grows: a run's rows are appended once, a superseded run's rows
stay and count, and **N_trials** is the number of distinct ``(run, key)``
pairs, counted even beyond N_max. Rows whose key is outside the catalogue are
refused, as is a ledger whose manifest names another catalogue. Reading checks
every part against the manifest's sha256 and row count.
``validation.choice.CellDays.from_ledger`` reads one run's rows.

**Simulator trials.** The registered trial bound counts the simulator's runs on
D beside the 2,400 taker cells (N_max = 2,400 + 12 + 30 + 9 + 4): S25-Z's cells,
S-P's cells and regime measurements, S-M's leans and guard measurements, and A0
and FILL's four runs on O. Each is one configuration scored on D, so each is a
key of the ledger and counts in N_trials. Their keys are not catalogue keys,
and the catalogue (its keys and its hash) is unchanged: a simulated cell is
written ``<slot>|<catalogue key>|<quoter>|fees=<fees>[|lean=<l>|policy=<p>]``,
S-P, S-M or S25-Z on a catalogue key, and A0 and FILL as the four
:data:`MEASUREMENT_TRIALS`; any other key is refused. They are appended after
stage C (:meth:`TrialLedger.append_simulator`) to a run that stage B wrote. A
simulated configuration has no walked long and short P&L: its ``net_bp`` is the
day's net in bp of its filled notional (0 without a fill), and ``G_bp``,
``pnl_long_bp`` and ``pnl_short_bp`` are written 0.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trading_research.search_engine.catalogue import Catalogue

#: The registered ledger's parts and its committed manifest, relative to the repository root.
LEDGER_PATH = Path("data/short_horizon_2025/short_horizon_2025_trials.parquet")
MANIFEST_PATH = Path("experiments/results/short_horizon_2025_trials.json")
LEDGER_COLUMNS: tuple[str, ...] = (
    "run",
    "key",
    "day",
    "net_bp",
    "G_bp",
    "pnl_long_bp",
    "pnl_short_bp",
    "trades",
    "skips",
)
VALUE_COLUMNS: tuple[str, ...] = ("net_bp", "G_bp", "pnl_long_bp", "pnl_short_bp")
#: Every part stays under this many bytes.
PART_BYTES = 100_000_000
SCHEMA = pa.schema(
    [
        ("run", pa.int64()),
        ("key", pa.string()),
        ("day", pa.date32()),
        *((name, pa.float64()) for name in VALUE_COLUMNS),
        ("trades", pa.int64()),
        ("skips", pa.int64()),
    ]
)


#: The simulated slots whose cells are trials, each on a catalogue key.
SIMULATED_SLOTS: tuple[str, ...] = ("S-P", "S-M", "S25-Z")
#: A0 crossing on O at 0/0, and its posted stream under FILL's three fill rules.
MEASUREMENT_TRIALS: tuple[str, ...] = (
    "A0|crossing|fees=zero",
    "FILL|touch|fees=zero",
    "FILL|trade_through|fees=zero",
    "FILL|queue|fees=zero",
)
#: Fields of a catalogue key (instrument, stratum, model, inputs, gate, hold,
#: rate or k, label fee or beta, regime).
KEY_FIELDS = 9


class LedgerError(ValueError):
    """Rows that cannot enter the ledger, or a ledger that fails its manifest."""


def is_simulator_key(key: str) -> bool:
    """Whether ``key`` names a simulator trial rather than a taker cell."""
    return key in MEASUREMENT_TRIALS or key.split("|", 1)[0] in SIMULATED_SLOTS


def require_simulator_keys(keys: Iterable[str], catalogue: Catalogue) -> None:
    """Refuse a simulator trial key that is not a registered simulated configuration."""
    for key in set(keys):
        if key in MEASUREMENT_TRIALS:
            continue
        parts = key.split("|")
        cell = "|".join(parts[1 : 1 + KEY_FIELDS])
        rest = parts[1 + KEY_FIELDS :]
        kinds = {"entry": 1, "maker": 3}
        ok = (
            parts[0] in SIMULATED_SLOTS
            and cell in catalogue
            and bool(rest)
            and rest[0] in kinds
            and len(rest) == 1 + kinds[rest[0]]
            and rest[1].startswith("fees=")
            and (
                rest[0] == "entry"
                or (rest[2].startswith("lean=") and rest[3].startswith("policy="))
            )
        )
        if not ok:
            raise LedgerError(f"{key!r} is neither a catalogue key nor a simulator trial")


def require_keys(keys: Iterable[str], catalogue: Catalogue) -> None:
    """Taker keys must be catalogue keys; simulator keys registered simulated configurations."""
    keys = list(keys)
    catalogue.require(k for k in keys if not is_simulator_key(k))
    require_simulator_keys((k for k in keys if is_simulator_key(k)), catalogue)


def file_sha256(path: Path) -> str:
    return ledger_sha256([path])


def ledger_sha256(paths: Iterable[Path]) -> str:
    """The sha256 of the parts' bytes in part order (one file's own sha256): the
    ``sha256`` the manifest records and the D amendment's ``ledger_sha256``."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def n_trials(rows: pd.DataFrame) -> int:
    """The number of distinct ``(run, key)`` pairs, taker cells and simulator trials."""
    return len(rows[["run", "key"]].drop_duplicates())


def simulator_rows(rows: pd.DataFrame) -> pd.Series:
    """Which rows are simulator trials."""
    return rows["key"].map(is_simulator_key).astype(bool)


def _day(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def check_rows(rows: pd.DataFrame, catalogue: Catalogue) -> pd.DataFrame:
    """The rows in ledger form, refused unless every value can enter the ledger."""
    if list(rows.columns) != list(LEDGER_COLUMNS):
        raise LedgerError(f"ledger rows need exactly the columns {LEDGER_COLUMNS}")
    require_keys(rows["key"], catalogue)
    out = rows.copy()
    out["day"] = [_day(v) for v in out["day"]]
    for name in ("run", "trades", "skips"):
        if not pd.api.types.is_integer_dtype(out[name]):
            raise LedgerError(f"{name} must be whole numbers")
    if (out["run"] < 1).any() or (out["trades"] < 0).any() or (out["skips"] < 0).any():
        raise LedgerError("runs count from 1; trades and skips are never negative")
    values = out[list(VALUE_COLUMNS)].to_numpy(dtype="float64")
    if not all(math.isfinite(v) for v in values.ravel()):
        raise LedgerError("a ledger value is missing or infinite; a cell without a trade writes 0")
    if out.duplicated(["run", "key", "day"]).any():
        raise LedgerError("two rows share one (run, key, day)")
    return out


def _write_parts(rows: pd.DataFrame, path: Path, part_bytes: int) -> list[dict[str, Any]]:
    """Write ``rows`` as ``path`` or, when that would be too large, as numbered
    parts beside it, each under ``part_bytes``; returns each part's record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(rows, schema=SCHEMA, preserve_index=False)
    parts = 1
    while True:
        size = math.ceil(len(rows) / parts) if len(rows) else 0
        names = [path] if parts == 1 else [_part_name(path, i) for i in range(parts)]
        records = []
        for i, target in enumerate(names):
            chunk = table.slice(i * size, size) if parts > 1 else table
            temporary = target.with_name(target.name + ".writing")
            pq.write_table(chunk, temporary, compression="zstd")
            records.append({"temporary": temporary, "target": target, "rows": chunk.num_rows})
        if all(r["temporary"].stat().st_size < part_bytes for r in records):
            break
        for record in records:
            record["temporary"].unlink()
        if parts >= max(len(rows), 1):
            raise LedgerError("a single ledger row does not fit in a part")
        parts *= 2
    out = []
    for record in records:
        record["temporary"].replace(record["target"])
        target = record["target"]
        out.append({"file": target.name, "sha256": file_sha256(target), "rows": record["rows"]})
    return out


def _part_name(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}.part{index:03d}{path.suffix}")


@dataclass(frozen=True)
class TrialLedger:
    """The ledger's parts at ``path`` for one catalogue, with its manifest at
    ``manifest`` (beside the parts when not given)."""

    path: Path
    catalogue: Catalogue
    part_bytes: int = PART_BYTES
    manifest: Path | None = None

    @classmethod
    def in_repository(cls, repo: Path, catalogue: Catalogue) -> TrialLedger:
        """The registered ledger of ``repo``: parts under ``data/``, manifest committed."""
        return cls(repo / LEDGER_PATH, catalogue, manifest=repo / MANIFEST_PATH)

    @property
    def manifest_path(self) -> Path:
        return self.manifest or self.path.with_suffix(".json")

    def record(self) -> dict[str, Any]:
        """The manifest as written, or a fresh one before the first write."""
        if not self.manifest_path.exists():
            empty: dict[str, Any] = {"catalogue_sha256": self.catalogue.sha256, "parts": []}
            return empty | {"rows": 0, "n_trials": 0, "runs": {}, "hand_runs": []}
        record: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if record.get("catalogue_sha256") != self.catalogue.sha256:
            raise LedgerError(
                f"the ledger was written for catalogue {record.get('catalogue_sha256')}, "
                f"not {self.catalogue.sha256}"
            )
        return record

    def read(self) -> pd.DataFrame:
        """Every row, each part checked against the manifest's sha256 and row count."""
        frames = []
        for part in self.record()["parts"]:
            target = self.path.with_name(part["file"])
            if not target.exists() or file_sha256(target) != part["sha256"]:
                raise LedgerError(f"{part['file']} is missing or differs from its sha256")
            frame = pq.read_table(target).to_pandas()
            if len(frame) != part["rows"]:
                raise LedgerError(f"{part['file']} holds {len(frame)} rows, not {part['rows']}")
            frames.append(frame)
        if not frames:
            return pd.DataFrame({c: pd.Series(dtype=t.to_pandas_dtype()) for c, t in
                                 zip(SCHEMA.names, SCHEMA.types, strict=True)})  # fmt: skip
        return pd.concat(frames, ignore_index=True)

    @property
    def n_trials(self) -> int:
        return int(self.record()["n_trials"])

    def run_rows(self, run: int, *, simulator: bool = False) -> pd.DataFrame:
        """One run's taker rows, as ``CellDays.from_ledger`` reads them (with
        ``simulator``, its simulator trials instead)."""
        rows = self.read()
        mine = (rows["run"] == run) & (simulator_rows(rows) == simulator)
        return rows[mine].reset_index(drop=True)

    def append(self, run: int, commit: str, rows: pd.DataFrame) -> dict[str, Any]:
        """Add one run's rows (a stage's worth), refused if any (run, key, day)
        is already in the ledger or the run was written at another commit."""
        self.catalogue.verify()
        record = self.record()
        new = check_rows(rows, self.catalogue)
        if simulator_rows(new).any():
            raise LedgerError("simulator trials are appended with append_simulator")
        if set(new["run"]) != {run}:
            raise LedgerError(f"every row appended for run {run} must carry run {run}")
        if not commit:
            raise LedgerError("each run's commit is recorded; none was given")
        known = record["runs"].get(str(run))
        if known is not None and known["commit"] != commit:
            raise LedgerError(f"run {run} was written at {known['commit']}, not {commit}")
        old = self.read()
        both = pd.concat([old, new], ignore_index=True) if len(old) else new
        if both.duplicated(["run", "key", "day"]).any():
            raise LedgerError(f"run {run} already has a row for some (key, day) appended")
        mine = both[both["run"] == run]
        record["runs"][str(run)] = {
            "commit": commit,
            "rows": len(mine),
            "keys": int(mine["key"].nunique()),
            "days": int(mine["day"].nunique()),
        }
        return self._commit(both, record)

    def append_simulator(self, run: int, commit: str, rows: pd.DataFrame) -> dict[str, Any]:
        """Add one run's simulator trials (stage C), once, to a run stage B wrote;
        the commit they were computed at is recorded beside the run's."""
        self.catalogue.verify()
        record = self.record()
        new = check_rows(rows, self.catalogue)
        if not simulator_rows(new).all() or set(new["run"]) - {run}:
            raise LedgerError(f"append_simulator takes run {run}'s simulator trials only")
        known = record["runs"].get(str(run))
        if known is None:
            raise LedgerError(f"run {run} has no taker rows; its simulator trials come after them")
        if "simulator" in known:
            raise LedgerError(f"run {run}'s simulator trials are already in the ledger")
        if not commit:
            raise LedgerError("each run's commit is recorded; none was given")
        both = pd.concat([self.read(), new], ignore_index=True)
        known["simulator"] = {"commit": commit, "rows": len(new), "keys": int(new["key"].nunique())}
        return self._commit(both, record)

    def log_hand_run(self, when: date, computed: str, days: Iterable[date]) -> dict[str, Any]:
        """Record a hand run: its date, what it computed and the test days it read.
        The catalogue and its hash are unchanged; the effect on the slots is the
        entry point's to apply (every searched slot not admitted on D)."""
        if not computed.strip():
            raise LedgerError("a hand run names what it computed")
        record = self.record()
        entry = {"date": when.isoformat(), "computed": computed}
        record["hand_runs"].append(entry | {"days": sorted(d.isoformat() for d in days)})
        self._write_manifest(record)
        return record

    @property
    def hand_runs(self) -> list[dict[str, Any]]:
        return list(self.record()["hand_runs"])

    def _commit(self, rows: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
        before = {part["file"] for part in record["parts"]}
        parts = _write_parts(rows, self.path, self.part_bytes)
        for stale in before - {part["file"] for part in parts}:
            self.path.with_name(stale).unlink(missing_ok=True)
        record["parts"] = parts
        record["file"] = self.path.name
        record["sha256"] = ledger_sha256(self.path.with_name(part["file"]) for part in parts)
        record["rows"] = len(rows)
        record["n_trials"] = n_trials(rows)
        simulated = simulator_rows(rows) if len(rows) else pd.Series(dtype=bool)
        record["n_trials_simulator"] = n_trials(rows[simulated]) if len(rows) else 0
        record["n_trials_taker"] = record["n_trials"] - record["n_trials_simulator"]
        self._write_manifest(record)
        return record

    def _write_manifest(self, record: dict[str, Any]) -> None:
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        target = self.manifest_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".writing")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
