"""The maker study's registration as code: its constants, its YAML validated, placeholders and hashes.

The study is registered in ``docs/preregistration/spot_maker_2026.md`` with
``configs/spot_maker_2026.yaml``. The constants below are the registered
values; :meth:`Registration.load` refuses a configuration that disagrees with
them, whose placeholders reset do not give the registered sha256
(:data:`REGISTERED_SHA256`, or the latest ``Null-placeholder sha256:`` line
under the document's ``## Amendments``), or that holds a superseded hash.
:func:`freeze_text` writes the D amendment's values into the placeholders and
nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time
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
from trading_research.spot_maker import blocks as bk

CONFIG_PATH = Path("configs/spot_maker_2026.yaml")
DOCUMENT_PATH = Path("docs/preregistration/spot_maker_2026.md")
CODE_FREEZE_NOTE = Path("docs/spot_maker_2026_code_freeze.md")
LEDGER_PATH = Path("experiments/results/spot_maker_2026_ledger.json")
RESULTS_PATHS = {b: Path(f"experiments/results/spot_maker_2026_{b}.json") for b in bk.HELD_OUT}
PAGE_PATH = Path("experiments/results/spot_maker_2026_page.md")
TRIALS_MANIFEST_PATH = Path("experiments/results/spot_maker_2026_trials.json")
RUN_LOG_PATH = Path("data/spot_maker_2026/run_log.json")
COPY_LOG_PATH = Path("data/spot_maker_2026/copies.json")
WORK_DIR = Path("artifacts/spot_maker_2026")
SIGNAL_DIR = Path("models/spot_maker_2026/signal")
#: The code the freeze covers (the configuration outside its placeholders apart).
CODE_PATHS: tuple[str, ...] = ("src", "experiments/spot_maker_2026.py", "pyproject.toml", "uv.lock")

#: The configuration as registered, every placeholder null (2026-10-27).
REGISTERED_SHA256 = "8af7842a9b56d8108956548197985ee910993a7e87fc7253bf8de130e862ea6d"
SUPERSEDED_SHA256: tuple[str, ...] = ()
NULL_HASH_PATTERN = re.compile(r"Null-placeholder sha256: `([0-9a-f]{64})`")
FROZEN_HASH_PATTERN = re.compile(r"Frozen configuration sha256: `([0-9a-f]{64})`")
AMENDMENTS_HEADING = "## Amendments"

PLACEHOLDERS: tuple[tuple[str, ...], ...] = (
    ("data", "reader_test_result"),
    ("data", "excluded_days_D"),
    ("metrics", "mde", "sigma_d"),
    ("metrics", "mde", "value"),
    ("measurements", "result_D"),
    ("compute", "record"),
    ("amendment", "date"),
)
SECTIONS: tuple[str, ...] = (
    "name", "registered", "registered_against", "document", "registers", "searched",
    "instrument", "computed_before_registration", "prior_information", "information_set",
    "expected_outcome", "data", "fees", "blocks", "common", "signal", "fills", "slots",
    "metrics", "family", "statuses", "kill_conditions", "placebo_percentile", "placebos",
    "measurements", "d_procedure", "copy_budget", "compute", "access", "amendment", "artefacts",
)  # fmt: skip

# --- the registered values --------------------------------------------------------
CLIP = 0.010
TICK = 0.01
NS_PER_S = 1_000_000_000
LATENCY_NS = 10_000_000
LATENCY_RERUN_NS = 110_000_000
POSTING_FIRST, POSTING_LAST, FLATTEN_AT = time(0, 1), time(23, 58), time(23, 59)
MARKOUTS_S: tuple[float, ...] = (1.0, 5.0, 30.0)
MAKING_S = 5.0
RANDOM_DRAWS = 200
PLACEBO_PERCENTILE = 95.0
CAPACITY_LIMIT = 0.10
MIN_FILLS = 100
MIN_DAYS: dict[str, int] = {"H": 20, "P": 10}
ALPHA, T_BAR, NW_BAR, NW_LAG = 0.05, 3.0, 2.0, 5
REMAINDER_BTC = 1e-12
DISK_BUDGET_GB, DISK_MARGIN_GB = 1.0, 10.0

SIGNAL_ROW = "S25-PF"
SIGNAL_SLOT_FILE = "S25-PF.json"
SIGNAL_SLOT_SHA256 = "d4048fa0902a5490c0b2238b52d7f0f4ed9c7bb558dc60345068598c8697a5d8"
SIGNAL_FIT = "S25-PF__hold_s=50__label_fee_bp=0.pkl"
SIGNAL_FIT_SHA256 = "4cf14dd130d361f9f8159da9ca272b5d99d6d30dfee7eacd9e0cb5c6d13ec9ba"
SPOT_CONFIG_SHA256 = "d1625173c0b7a15c8785a957d77596210f46893e792c834449152e39cf5476b7"
SIGNAL_CELL = "hold_s=50|trades_per_day=240|label_fee_bp=0"
SIGNAL_THETA: dict[str, float] = {
    "theta_long": 0.46583154797554016,
    "theta_short": 0.4700717031955719,
}
SIGNAL_HOLD_S = 50.0


@dataclass(frozen=True)
class SlotSpec:
    """One slot's registered coordinates."""

    name: str
    signal: bool
    requote_s: float
    skew_bp: float
    limit_clips: int
    nbhd_requote_s: tuple[float, ...]
    nbhd_skew_bp: tuple[float, ...]
    kills: tuple[str, ...]

    @property
    def unit(self) -> float:
        """The account's unit and the position limit, in BTC."""
        return self.limit_clips * CLIP

    def cells(self) -> list[tuple[float, float]]:
        return [(r, k) for r in self.nbhd_requote_s for k in self.nbhd_skew_bp]


_KILLS = ("K1", "K-dir", "K-placebo", "K-drift", "K-nbhd", "K-days", "K-latency", "K-fill", "K-cap")
SLOTS: dict[str, SlotSpec] = {
    "MK-S": SlotSpec("MK-S", True, 60.0, 0.0, 1, (30.0, 60.0, 120.0), (-1.0, 0.0, 1.0), _KILLS),
    "MK-Q": SlotSpec("MK-Q", False, 60.0, 1.0, 1, (30.0, 60.0, 120.0), (0.0, 1.0, 2.0),
                     tuple(k for k in _KILLS if k != "K-placebo")),
    "MK-SQ": SlotSpec("MK-SQ", True, 60.0, 1.0, 2, (30.0, 60.0, 120.0), (0.0, 1.0, 2.0), _KILLS),
}  # fmt: skip
FAMILY: tuple[str, ...] = tuple(SLOTS)
VOIDS: tuple[str, ...] = ("K-units", "K-acct", "K-inv", "K-pos")


def amendments_text(document: str) -> str:
    """The document from its ``## Amendments`` heading on ('' when it has none)."""
    at = document.find("\n" + AMENDMENTS_HEADING + "\n")
    return "" if at < 0 else document[at:]


def accepted_null_hash(document: str) -> str:
    """The latest ``Null-placeholder sha256:`` line under Amendments, else the registered one."""
    found = NULL_HASH_PATTERN.findall(amendments_text(document))
    return found[-1] if found else REGISTERED_SHA256


def null_sha256(text: str) -> str:
    """The configuration's sha256 with every placeholder reset to null."""
    return sha256_bytes(unfrozen_text(text).encode())


def _expect(where: str, found: Any, wanted: Any) -> None:
    if found != wanted:
        raise PreRegistrationError(f"{where} is {found!r}; the code registers {wanted!r}")


def check_values(config: Mapping[str, Any]) -> None:
    """The configuration's registered values equal the code's constants."""
    _expect("sections", tuple(config), SECTIONS)
    _expect("family", tuple(config["family"]), FAMILY)
    _expect("instrument.id", config["instrument"]["id"], bk.INSTRUMENT)
    _expect("instrument.tick", float(config["instrument"]["tick"]), TICK)
    common = config["common"]
    _expect("common.clip_btc", float(common["clip_btc"]), CLIP)
    _expect("common.latency_ms", int(common["latency_ms"]) * 1_000_000, LATENCY_NS)
    signal = config["signal"]
    for key, wanted in (("slot_file_sha256", SIGNAL_SLOT_SHA256), ("fit", SIGNAL_FIT),
                        ("fit_sha256", SIGNAL_FIT_SHA256), ("cell", SIGNAL_CELL),
                        ("spot_config_sha256", SPOT_CONFIG_SHA256)):  # fmt: skip
        _expect(f"signal.{key}", signal[key], wanted)
    _expect("signal.theta", {k: float(v) for k, v in signal["theta"].items()}, SIGNAL_THETA)
    _expect("signal.hold_s", float(signal["hold_s"]), SIGNAL_HOLD_S)
    for name, spec in SLOTS.items():
        slot = config["slots"][name]
        _expect(f"slots.{name}.requote_s", float(slot["requote_s"]), spec.requote_s)
        _expect(f"slots.{name}.skew_bp_per_clip", float(slot["skew_bp_per_clip"]), spec.skew_bp)
        _expect(f"slots.{name}.limit_clips", int(slot["limit_clips"]), spec.limit_clips)
        nbhd = slot["neighbourhood"]
        _expect(f"slots.{name}.neighbourhood.requote_s",
                tuple(float(v) for v in nbhd["requote_s"]), spec.nbhd_requote_s)  # fmt: skip
        _expect(f"slots.{name}.neighbourhood.skew_bp_per_clip",
                tuple(float(v) for v in nbhd["skew_bp_per_clip"]), spec.nbhd_skew_bp)  # fmt: skip
        _expect(f"kill_conditions.per_slot.{name}",
                tuple(config["kill_conditions"]["per_slot"][name]), spec.kills)  # fmt: skip
        if "hold_s" in slot:
            _expect(f"slots.{name}.hold_s", float(slot["hold_s"]), SIGNAL_HOLD_S)
    _expect("kill_conditions.void", tuple(config["kill_conditions"]["void"]), VOIDS)
    _expect("placebo_percentile", float(config["placebo_percentile"]), PLACEBO_PERCENTILE)
    _expect("statuses.m", int(config["statuses"]["m"]), len(FAMILY))
    for block, (start, end) in bk.BLOCKS.items():
        found = config["blocks"][block]
        _expect(f"blocks.{block}", (found["start"], found["end"]), (start, end))
    _expect("blocks.reader_test.days", tuple(config["blocks"]["reader_test"]["days"]),
            bk.READER_TEST_DAYS)  # fmt: skip


@dataclass(frozen=True)
class Registration:
    """The configuration's text and values, validated."""

    text: str
    config: Mapping[str, Any]
    document: str

    @classmethod
    def load(cls, repo: Path) -> Registration:
        text = (Path(repo) / CONFIG_PATH).read_text(encoding="utf-8")
        document = (Path(repo) / DOCUMENT_PATH).read_text(encoding="utf-8")
        return cls.from_text(text, document)

    @classmethod
    def from_text(cls, text: str, document: str) -> Registration:
        found = null_sha256(text)
        if found in SUPERSEDED_SHA256:
            raise PreRegistrationError(f"the configuration has a superseded hash {found}")
        accepted = accepted_null_hash(document)
        if found != accepted:
            raise PreRegistrationError(
                f"the configuration with its placeholders reset has sha256 {found}, "
                f"not the registered {accepted}"
            )
        _expect("placeholders", tuple(placeholder_lines(text)), PLACEHOLDERS)
        config = yaml.safe_load(text)
        check_values(config)
        return cls(text, config, document)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.text.encode())

    def value(self, path: tuple[str, ...]) -> Any:
        node: Any = self.config
        for key in path:
            node = node[key]
        return node

    @property
    def frozen(self) -> bool:
        return all(self.value(p) is not None for p in PLACEHOLDERS)

    def require_frozen(self) -> None:
        missing = [".".join(p) for p in PLACEHOLDERS if self.value(p) is None]
        if missing:
            raise PreRegistrationError(f"not frozen: {missing}")

    def amendment_hash(self) -> str | None:
        """The ``Frozen configuration sha256:`` the amendment recorded, if any."""
        found = FROZEN_HASH_PATTERN.findall(amendments_text(self.document))
        return found[-1] if found else None


_LINE = re.compile(r"^(?P<head>\s*[A-Za-z0-9_.\-]+: )(?P<value>.*?)(?P<tail>  # [^\n]*"
                   + re.escape(PLACEHOLDER_MARK) + r"[^\n]*)$")  # fmt: skip


def freeze_text(text: str, values: Mapping[tuple[str, ...], Any]) -> str:
    """Every placeholder, each null now, written once with its value; nothing else changes."""
    where = placeholder_lines(text)
    if set(values) != set(PLACEHOLDERS):
        raise PreRegistrationError(f"freezing needs every placeholder once, got {sorted(values)}")
    lines = text.split("\n")
    for path, value in values.items():
        match = _LINE.match(lines[where[path]])
        if match is None or match.group("value").strip() != "null":
            raise PreRegistrationError(f"{'.'.join(path)} is already frozen")
        lines[where[path]] = match.group("head") + render_value(value) + match.group("tail")
    frozen = "\n".join(lines)
    if unfrozen_text(frozen) != unfrozen_text(text):
        raise PreRegistrationError("freezing changed more than the placeholders")
    return frozen
