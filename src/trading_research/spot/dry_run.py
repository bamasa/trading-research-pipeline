"""The dry run: every stage and the H read on a synthetic venue, in a temporary repository.

The temporary repository holds the registration as registered under git: the
configuration with every placeholder reset and without P's fee intervals, the
document without the D amendment or anything after it. So the study's own gates
(the YAML validated against its registered hash, the code-freeze note, the
ledger's preconditions for H, which read git) run exactly as they will on the
real study. Every request is answered by :class:`trading_research.spot.synthetic.SpotVenue`
inside :func:`trading_research.spot.synthetic.no_network`, and no request
reaches the network; free disk is a stand-in measurement.

**S25-T's 2025 files** are a stand-in made here: six small fits of the 2025
code (``gbm`` on S25's 40 inputs, the walked-book label at holds 20, 50, 120 s
and label fees 0 and 5.5 bp) on one synthetic Bybit day outside every block,
their thresholds, a slot file, a fitted-values file and a 2025 configuration
recording their sha256s, which the dry run's :class:`~trading_research.spot.context.Binding2025`
binds. The study checks them by the same code as the registered files.

P is opened only when a slot is a candidate on H: the dry run then writes a
fee amendment covering P's days, commits it and reads P. This is a test of the
code, never a result; the outputs are discarded unless ``keep`` names a folder.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trading_research.market_making.prereg import unfrozen_text
from trading_research.spot import registration as sr
from trading_research.spot import synthetic as syn
from trading_research.spot.context import Binding2025, SpotStudy
from trading_research.spot.ledger import file_sha256
from trading_research.spot.plan import DRY_RUN, LABEL_FEES_2025_BP, RATES, Scale
from trading_research.spot.stage1 import CODE_FREEZE_NOTE
from trading_research.spot.state import STAGES

REPO = Path(__file__).resolve().parents[3]
GITIGNORE = "/data/\n/artifacts/\n/models/\n"
_IDENTITY = {"GIT_AUTHOR_NAME": "dry run", "GIT_AUTHOR_EMAIL": "dry-run@localhost",
             "GIT_COMMITTER_NAME": "dry run", "GIT_COMMITTER_EMAIL": "dry-run@localhost"}  # fmt: skip
#: A synthetic day outside every block, on which the 2025 stand-in is fitted.
STAND_IN_DAY = date(2026, 2, 20)
#: The H day whose book has sequence gaps: excluded on Bybit and put back.
GAP_DAY = date(2026, 9, 9)
D_AMENDMENT = ": the D amendment"
#: The synthetic book's message step: S25's one-second inputs need rows within a second.
STEP_MS = 1_000


def commit(repo: Path, message: str) -> str:
    """Commit everything in the temporary repository; its new HEAD."""
    env = {**os.environ, **_IDENTITY}
    for args in (["add", "-A"], ["commit", "-q", "--allow-empty", "-m", message]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                          text=True)  # fmt: skip
    return done.stdout.strip()


def registered_text(relative: Path, text: str) -> str:
    """A file of the registration as registered: the configuration without P's
    intervals and with every placeholder reset; the document cut before the D
    amendment's section."""
    if Path(relative) == sr.CONFIG_PATH:
        return unfrozen_text(sr.strip_p_intervals(text)[0])
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if line.startswith("### ") and D_AMENDMENT in line:
            return "\n".join(lines[:k]).rstrip("\n") + "\n"
    return text


def market(scale: Scale) -> syn.SpotMarket:
    """The venue's days: every day the scale plans, and the 2025 stand-in's day."""
    days = {d for block in scale.calendar or {} for d in scale.days(block)}
    return syn.SpotMarket(frozenset(days | {STAND_IN_DAY}), step_ms=STEP_MS,
                          gap_days=frozenset({GAP_DAY}), rpi_every=9)  # fmt: skip


def make_repository(source: Path, target: Path) -> Path:
    """The registration as registered under git."""
    target.mkdir(parents=True, exist_ok=True)
    for relative in (sr.CONFIG_PATH, sr.DOCUMENT_PATH):
        text = (source / relative).read_text(encoding="utf-8")
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_text(registered_text(relative, text), encoding="utf-8")
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (target / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=target, check=True, capture_output=True)
    return target


def code_freeze_note(repo: Path) -> str:
    """The code-freeze note naming HEAD, committed."""
    head = commit(repo, "Dry run: before the code freeze")
    note = repo / CODE_FREEZE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(f"# Code freeze (dry run)\n\nCode-freeze commit: `{head}`\n", encoding="utf-8")
    return commit(repo, "Dry run: the code-freeze note")


def stand_in_2025(repo: Path, found: syn.SpotMarket) -> Binding2025:
    """The 2025 study's files as the dry run's stand-in (module docstring)."""
    from trading_research.data.feature_store import median_touch_size
    from trading_research.spot import bybit_book, bybit_prints
    from trading_research.spot import days as dy
    from trading_research.spot.book_inputs import S25_SELECTED, book_day_frame, book_targets
    from trading_research.spot.fits import FIT_2025, KEY_2025, finite_theta, fit_model, save_model
    from trading_research.spot.print_label import label_column
    from trading_research.spot.state import write_json
    from trading_research.strategies.hold_first import fit_hold_first

    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "stand_in.zip"
        archive.write_bytes(found.book_archive(STAND_IN_DAY))
        rows = bybit_book.replay(archive)
    prints = bybit_prints.parse(gzip.decompress(found.bybit_prints_archive(STAND_IN_DAY)))
    touch = median_touch_size([rows])
    x = dy.model_inputs(book_day_frame(rows, prints), S25_SELECTED, dy.layout("book", touch))
    holds = (20.0, 50.0, 120.0)
    targets = book_targets(rows, holds, LABEL_FEES_2025_BP)
    complete = x.notna().all(axis=1).to_numpy()
    base = repo / sr.FROZEN_DIR_2025
    files: dict[Path, str] = {}
    weights, thresholds = {}, {}
    for hold in holds:
        for fee in LABEL_FEES_2025_BP:
            y = targets[label_column(hold, fee)]
            if y.nunique() < 2:  # the synthetic book rarely moves a fee's width: 0 bp's label
                y = targets[label_column(hold, 0.0)]
            keep = complete & y.notna().to_numpy()
            model = fit_model(x[keep], y[keep], DRY_RUN.model_params)
            if model is None:
                raise RuntimeError("the 2025 stand-in has a fit with fewer than two classes")
            name = FIT_2025.format(h=hold, f=fee)
            weights[name] = save_model(model, base / "weights" / name)
            files[sr.FROZEN_DIR_2025 / "weights" / name] = weights[name]
            proba = np.asarray(model.predict_proba(x[complete]))
            for rate in RATES:
                rule = fit_hold_first(proba, days=1.0, trades_per_day=rate)
                thresholds[KEY_2025.format(h=hold, r=rate, f=fee)] = finite_theta(rule)
    slot = write_json(base / "finalists" / "S25.json", {"slot": "S25", "stand_in": "dry run"})
    fitted = write_json(base / "fitted_values.json", {"touch_size": touch})
    files[sr.FROZEN_DIR_2025 / "finalists" / "S25.json"] = slot
    files[sr.FROZEN_DIR_2025 / "fitted_values.json"] = fitted
    config = {"name": "the dry run's stand-in for the 2025 configuration",
              "artefacts": {"frozen_finalist_files_sha256": {"S25": slot}},
              "features": {"fitted_values_sha256": fitted}, "models": {"weights_sha256": weights},
              "slots": {"S25": {"thresholds": thresholds}}}  # fmt: skip
    path = repo / sr.CONFIG_2025_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return Binding2025(sr.CONFIG_2025_PATH, file_sha256(path), files, None)


def fee_amendment(repo: Path) -> None:
    """The dry run's fee amendment: one interval per venue covering P's days, appended."""
    path = repo / sr.CONFIG_PATH
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    added = {
        "binance_spot_BTCFDUSD": "{regular: [0.0, 10.0]}",
        "bybit_spot_BTCUSDT": "{non_vip: [10.0, 10.0]}",
    }
    sources = {"binance_spot_BTCFDUSD": "[B5, B7]", "bybit_spot_BTCUSDT": "[Y2]"}
    last = {key: number for number, key, _ in sr.interval_lines(text)}
    for key in sorted(last, key=last.__getitem__, reverse=True):
        indent = lines[last[key]][: len(lines[last[key]]) - len(lines[last[key]].lstrip())]
        lines.insert(last[key] + 1, f'{indent}- {{start: "2026-10-22T00:00:00Z", end: '
                     f'"2027-01-01T00:00:00Z", sources: {sources[key]}, tiers: {added[key]}, '
                     f'notes: "dry run"}}')  # fmt: skip
    path.write_text("\n".join(lines), encoding="utf-8")
    document = repo / sr.DOCUMENT_PATH
    document.write_text(document.read_text(encoding="utf-8").rstrip("\n")
                        + "\n\n### The fee amendment (dry run)\n\nP's days covered for the dry run.\n",
                        encoding="utf-8")  # fmt: skip


def prepare(root: Path, *, source: Path = REPO, scale: Scale = DRY_RUN) -> SpotStudy:
    """A temporary repository, the 2025 stand-in, and a study on them with the venue."""
    repo = make_repository(source, root / "repo")
    venue = syn.SpotVenue(market(scale))
    binding = stand_in_2025(repo, venue.market)
    commit(repo, "Registration, and the dry run's stand-in for the 2025 configuration")
    return SpotStudy(repo, scale, venue=venue, binding=binding, disk=lambda _: 1_000.0)


def run(study: SpotStudy) -> dict[str, Any]:
    """Every stage in order, committing what the author commits between them: the
    code-freeze note before stage 1, the amendment, the manifest and the frozen
    configuration after ``amend``."""
    from trading_research.spot.runner import run_stage

    records = {}
    for stage in STAGES:
        if stage == "1":
            code_freeze_note(study.repo)
        records[stage] = run_stage(study, stage)
        commit(study.repo, f"Dry run: stage {stage}")
    return records


def read_blocks(study: SpotStudy) -> dict[str, Any]:
    """Open and read H as the registration orders it, committing what the author would;
    P only for H's candidates, after the dry run's fee amendment."""
    from trading_research.market_making.prereg import HeldOutLocked
    from trading_research.spot import reads
    from trading_research.spot import statuses as st

    out: dict[str, Any] = {}
    reads.open_block(study, "H")
    commit(study.repo, "Dry run: H's open entry")
    first = reads.read_block(study, "H")
    commit(study.repo, "Dry run: H's fetch records")
    out["H_first_process"] = first
    out["H"] = reads.read_block(study, "H", commit=commit)
    commit(study.repo, "Dry run: H's results and page")
    candidates = st.candidates_in(reads.h_results(study))
    if not candidates:
        try:
            reads.open_block(study, "P")
        except HeldOutLocked as exc:
            out["P"] = {"opened": False, "refused": str(exc)}
        return out
    fee_amendment(study.repo)
    commit(study.repo, "Dry run: the fee amendment")
    study.reload()
    reads.open_block(study, "P")
    commit(study.repo, "Dry run: P's open entry")
    out["P"] = reads.read_block(study, "P", commit=commit)
    commit(study.repo, "Dry run: P's results and page")
    return out


def changed(repo: Path) -> list[str]:
    done = subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True,
                          capture_output=True, text=True)  # fmt: skip
    return sorted(line[3:] for line in done.stdout.splitlines())


def dry_run(keep: Path | None = None, scale: Scale = DRY_RUN) -> dict[str, Any]:
    """The whole dry run; its files are deleted unless ``keep`` is given."""
    from trading_research.spot.pages import publish

    root = keep or Path(tempfile.mkdtemp(prefix="spot-dry-run-"))
    try:
        with syn.no_network():
            study = prepare(root, scale=scale)
            stages = run(study)
            blocks = read_blocks(study)
            written = publish(study)
        assert study.venue is not None
        if study.venue.unknown:
            raise RuntimeError(f"the dry run asked for what the venue does not serve: "
                               f"{study.venue.unknown[:3]}")  # fmt: skip
        return {
            "stages": list(stages),
            "amendment": stages["amend"]["amendment"],
            "reads": {b: r.get("statuses", r) for b, r in blocks.items() if b in ("H", "P")},
            "H_first_process": blocks.get("H_first_process"),
            "sentences": written["sentences"],
            "pages": {"written": written["written"], "changed_on_rewrite": changed(study.repo)},
            "venue": {"requests": len(study.venue.asked), "unknown": len(study.venue.unknown)},
            "repo": str(study.repo) if keep else "discarded",
        }
    finally:
        if keep is None:
            shutil.rmtree(root, ignore_errors=True)


def binding_of(repo: Path) -> Binding2025:
    """The 2025 stand-in's binding, read back from a dry run's repository."""
    config = repo / sr.CONFIG_2025_PATH
    found = yaml.safe_load(config.read_text(encoding="utf-8"))
    base = sr.FROZEN_DIR_2025
    files = {base / "weights" / k: v for k, v in found["models"]["weights_sha256"].items()}
    files[base / "finalists" / "S25.json"] = found["artefacts"]["frozen_finalist_files_sha256"][
        "S25"
    ]
    files[base / "fitted_values.json"] = found["features"]["fitted_values_sha256"]
    return Binding2025(sr.CONFIG_2025_PATH, file_sha256(config), files, None)
