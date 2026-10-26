"""The spot study's ledger: opening H and P, and the read sessions it binds.

Each test builds a small repository holding the frozen configuration, the
registration with its D amendment, this study's frozen files, synthetic stand-ins
for the 2025 study's files and the run log, and commits it; nothing touches the
real repository, any remote or the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import yaml

from test_spot_registration_loader import frozen_values, with_p_intervals
from trading_research.market_making.prereg import HeldOutLocked, unfrozen_text
from trading_research.spot import access as sa
from trading_research.spot import ledger as sl
from trading_research.spot import registration as sr
from trading_research.spot import statuses as st

ROOT = Path(__file__).resolve().parents[1]
TEXT = unfrozen_text((ROOT / sr.CONFIG_PATH).read_text(encoding="utf-8"))
DOCUMENT = (ROOT / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
H_DAY = datetime(2026, 11, 21, 9, tzinfo=UTC)
P_DAY = datetime(2027, 1, 10, 9, tzinfo=UTC)


def parquet_bytes(rows: int) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"run": list(range(rows))}), sink)
    return bytes(sink.getvalue().to_pybytes())


TRIALS = parquet_bytes(10)
OURS = {"weights": b"weights", "slot": b"slot", "fitted": b"fitted", "trials": TRIALS}
OUR_FILES = {
    sl.WEIGHTS_DIR / "S25-PF__frozen.pkl": OURS["weights"],
    sl.SLOT_FILES_DIR / "S25-PF.json": OURS["slot"],
    sl.FITTED_VALUES_PATH: OURS["fitted"],
}
#: Stand-ins for the 2025 study's files, with a 2025 configuration recording them.
FILES_2025 = {path: f"2025 {path.name}".encode() for path in sl.files_2025()}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_2025() -> bytes:
    weights = {p.name: sha(d) for p, d in FILES_2025.items() if p.parent.name == "weights"}
    tree = {
        "artefacts": {"frozen_finalist_files_sha256": {"S25": sha(FILES_2025[sl._SLOT_FILE_2025])}},
        "features": {"fitted_values_sha256": sha(FILES_2025[sl._FITTED_2025])},
        "models": {"weights_sha256": weights},
    }
    return yaml.safe_dump(tree).encode()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("tests do not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def git(repo: Path, *args: str, when: str | None = None) -> str:
    env = {**os.environ, **({"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when} if when else {})}
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Study", "-c", "user.email=study", *args],
        check=True, capture_output=True, text=True, env=env,
    )  # fmt: skip
    return done.stdout


def commit(repo: Path, message: str = "step", when: str | None = None) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message, when=when)
    return git(repo, "rev-parse", "HEAD").strip()


def write(repo: Path, relative: Path | str, data: bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


#: The study code's stand-in, frozen by the note before stage 1.
CODE = Path("src/trading_research/spot/statuses.py")
FREEZES: dict[Path, str] = {}


def frozen_text(repo: Path) -> str:
    """The frozen configuration, its compute record naming the repository's freeze."""
    return sr.freeze_text(TEXT, frozen_values(FREEZES[repo], **OURS))


def amended_document(text: str) -> str:
    return DOCUMENT + f"\n### 2026-11-20: the D amendment\n\n{sr.frozen_hash_line(text)}\n"


def freeze_code(repo: Path, message: str = "the code-freeze note") -> str:
    """Commit the note naming HEAD as the code-freeze commit; return that commit."""
    frozen_at = git(repo, "rev-parse", "HEAD").strip()
    note = repo / "docs" / "spot_2026_code_freeze.md"
    lines = note.read_text(encoding="utf-8") if note.exists() else "# Code freeze\n"
    write(repo, note.relative_to(repo), (lines + f"\nCode-freeze commit: `{frozen_at}`\n").encode())
    commit(repo, message)
    return frozen_at


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The registration and the study code; the code-freeze note; then, after D, the
    frozen configuration, the D amendment, the frozen files and the run log."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, ".gitignore", b"/models/\n/data/\n")
    write(repo, sr.CONFIG_PATH, TEXT.encode())
    write(repo, sr.DOCUMENT_PATH, DOCUMENT.encode())
    write(repo, sr.CONFIG_2025_PATH, config_2025())
    write(repo, CODE, b"T_BAR = 3.0\n")
    write(repo, "README.md", b"study\n")
    commit(repo, "registration and study code")
    FREEZES.clear()
    FREEZES[repo] = freeze_code(repo)
    text = frozen_text(repo)
    write(repo, sr.CONFIG_PATH, text.encode())
    write(repo, sr.DOCUMENT_PATH, amended_document(text).encode())
    for relative, data in {**OUR_FILES, **FILES_2025}.items():
        write(repo, relative, data)
    write(repo, sr.TRIALS_PATH, TRIALS)
    manifest = {"sha256": sha(TRIALS), "rows": 10,
                "parts": [{"file": sr.TRIALS_PATH.name, "sha256": sha(TRIALS), "rows": 10}]}  # fmt: skip
    write(repo, sr.TRIALS_MANIFEST_PATH, json.dumps(manifest).encode())
    commit(repo, "the D amendment")
    return repo


def make_ledger(repo: Path, *, env: str = "env-a", now: datetime = H_DAY) -> sl.Ledger:
    return sl.Ledger(
        repo=repo, environment=lambda: env, clock=lambda: now,
        config_2025_sha256=sha(config_2025()), files_2025={p: sha(d) for p, d in FILES_2025.items()},
    )  # fmt: skip


def opened(repo: Path) -> sl.ReadSession:
    """H opened, its entry committed, and the first process of its session started."""
    make_ledger(repo).open("H")
    commit(repo, "open H")
    return make_ledger(repo).session("H", workers=4)


def test_open_h_records_the_entry_and_the_session_grants_h(repo: Path) -> None:
    entry = make_ledger(repo).open("H")
    assert entry["read"] == "first read" and entry["environment"] == "env-a"
    assert (
        entry["config_sha256"] == sl.file_sha256(repo / sr.CONFIG_PATH) == entry["h_config_sha256"]
    )
    assert (
        entry["commit"] == git(repo, "rev-parse", "HEAD").strip()
        and entry["utc"] == "2026-11-21T09:00:00Z"
    )
    with pytest.raises(HeldOutLocked, match="commit it first"):
        make_ledger(repo).session("H")
    commit(repo, "open H")
    session = make_ledger(repo).session("H", workers=4)
    assert session.stamp == "first read" and session.key[:3] == (
        "H",
        entry["commit"],
        entry["config_sha256"],
    )
    session.access.require(sa.BYBIT, sa.block_days("H"), sa.BOOK)
    session.access.require(sa.BINANCE, sa.block_days("D"), sa.PRINTS)
    with pytest.raises(HeldOutLocked):
        session.access.require(sa.BYBIT, sa.block_days("P")[:1], sa.PRINTS)
    assert make_ledger(repo).entries()["H"][-1]["processes"] == [
        {"utc": "2026-11-21T09:00:00Z", "workers": 4}
    ]


def test_h_needs_the_d_amendment_and_the_frozen_file(repo: Path) -> None:
    write(repo, sr.DOCUMENT_PATH, DOCUMENT.encode())
    commit(repo)
    with pytest.raises(HeldOutLocked, match="no amendment records"):
        make_ledger(repo).open("H")
    write(repo, sr.DOCUMENT_PATH, amended_document(frozen_text(repo)).encode())
    write(repo, sr.CONFIG_PATH, TEXT.encode())
    commit(repo)
    with pytest.raises(HeldOutLocked, match="still null"):
        make_ledger(repo).open("H")
    other = sr.freeze_text(TEXT, {**frozen_values(FREEZES[repo], **OURS),
                                  ("data", "max_sequence_gaps"): 4})  # fmt: skip
    write(repo, sr.CONFIG_PATH, other.encode())
    commit(repo)
    with pytest.raises(HeldOutLocked, match="the amendment froze"):
        make_ledger(repo).open("H")
    write(repo, sr.CONFIG_PATH, frozen_text(repo).encode())
    commit(repo)
    write(repo, sr.CONFIG_PATH, (frozen_text(repo) + "\n").encode())
    with pytest.raises(HeldOutLocked):
        make_ledger(repo).open("H")  # not the frozen hash on disk, and not committed


def test_h_needs_a_clean_tree_and_no_file_of_h(repo: Path) -> None:
    write(repo, "notes.txt", b"x")
    with pytest.raises(HeldOutLocked, match="besides the ledger"):
        make_ledger(repo).open("H")
    (repo / "notes.txt").unlink()
    write(repo, sa.data_root(sa.BYBIT, sa.BOOK, "data") / "2026-08-08.parquet", b"")
    with pytest.raises(HeldOutLocked, match="already on disk"):
        make_ledger(repo).open("H")


@pytest.mark.parametrize(
    ("relative", "data", "match"),
    [
        (sl.WEIGHTS_DIR / "S25-PF__frozen.pkl", b"other", "registration records"),
        (sl.FITTED_VALUES_PATH, b"other", "registration records"),
        (sl.SLOT_FILES_DIR / "S25-PF.json", b"other", "registration records"),
        (sr.FROZEN_DIR_2025 / "finalists" / "S25.json", b"other", "registration records"),
        (sr.FROZEN_DIR_2025 / "weights" / sr.S25_T_FIT, b"other", "registration records"),
        (sr.TRIALS_PATH, parquet_bytes(11), "differs from its sha256"),
        (sr.TRIALS_PATH.with_name("spot_2026_trials.part1.parquet"), TRIALS, "does not list"),
    ],
)
def test_h_needs_every_frozen_file_at_its_hash(
    repo: Path, relative: Path, data: bytes, match: str
) -> None:
    write(repo, relative, data)
    with pytest.raises(HeldOutLocked, match=match):
        make_ledger(repo).open("H")


def test_h_needs_the_2025_configuration_at_its_frozen_hash(repo: Path) -> None:
    write(repo, sr.CONFIG_2025_PATH, config_2025() + b"# edited\n")
    commit(repo)
    with pytest.raises(HeldOutLocked, match="frozen sha256"):
        make_ledger(repo).open("H")
    ledger = make_ledger(repo)
    ledger.config_2025_sha256 = sha(config_2025() + b"# edited\n")
    ledger.files_2025 = {**ledger.files_2025, sl._FITTED_2025: sha(b"another")}
    with pytest.raises(HeldOutLocked, match="records other hashes"):
        ledger.open("H")
    assert (
        sl.files_2025()[sr.FROZEN_DIR_2025 / "weights" / sr.S25_T_FIT]
        == sr.NEIGHBOURHOOD_FITS_2025[sr.S25_T_FIT]
    )


def test_the_run_log_manifest_is_committed_and_matches(repo: Path) -> None:
    manifest = json.loads((repo / sr.TRIALS_MANIFEST_PATH).read_text())
    write(repo, sr.TRIALS_MANIFEST_PATH, json.dumps({**manifest, "rows": 11}).encode())
    commit(repo)
    with pytest.raises(HeldOutLocked, match="the amendment"):
        make_ledger(repo).open("H")
    write(repo, sr.TRIALS_MANIFEST_PATH, json.dumps(manifest).encode())
    commit(repo)
    (repo / sr.TRIALS_PATH).unlink()
    with pytest.raises(HeldOutLocked, match="not on disk"):
        make_ledger(repo).open("H")


def test_a_second_open_needs_force_and_is_stamped(repo: Path) -> None:
    make_ledger(repo).open("H")
    with pytest.raises(HeldOutLocked, match="not committed"):
        make_ledger(repo).open("H")
    commit(repo, "open H")
    with pytest.raises(HeldOutLocked, match="second read"):
        make_ledger(repo).open("H")
    entry = make_ledger(repo).open("H", force=True)
    assert entry["read"] == "second read"
    commit(repo, "open H again")
    assert make_ledger(repo).session("H").stamp == "second read"
    with pytest.raises(ValueError, match="opens"):
        make_ledger(repo).open("D")
    with pytest.raises(ValueError, match="P only"):
        make_ledger(repo).open("H", force=True, candidates=["S25-PF"])


def test_every_process_is_bound_to_its_commit_and_environment(repo: Path) -> None:
    opened(repo)
    with pytest.raises(HeldOutLocked, match="environment differs"):
        make_ledger(repo, env="env-b").session("H")
    write(repo, "README.md", b"changed\n")
    commit(repo, "a change after the open")
    with pytest.raises(HeldOutLocked, match="changed since the session's commit"):
        make_ledger(repo).session("H")


def test_the_session_records_fetch_failures_and_closes(repo: Path) -> None:
    session = opened(repo)
    with pytest.raises(ValueError, match="download time"):
        session.record_fetch({"book/2026-08-08": {"sha256": "x"}})
    session.record_fetch(
        {"book/2026-08-08": {"sha256": "x", "downloaded_at": "2026-11-21T09:01:00Z"}}
    )
    with pytest.raises(HeldOutLocked, match="not committed"):
        session.require_fetch_committed()
    commit(repo, "fetch")
    session.require_fetch_committed()
    assert session.record_failure("replay", RuntimeError("boom")) is None
    failure = session.record_failure("replay", RuntimeError("boom"))
    assert failure is not None and failure.reason.startswith("the same stage failed again")
    write(repo, sr.RESULTS_PATHS["H"], b"{}")
    digest = session.close()
    assert make_ledger(repo).entries()["H"][-1]["results_sha256"] == digest == sha(b"{}")
    with pytest.raises(HeldOutLocked, match="closed"):
        session.record_fetch({"x": {"sha256": "y", "downloaded_at": "z"}})


def test_a_read_unfinished_after_five_resumptions_must_close(repo: Path) -> None:
    session = opened(repo)
    for _ in range(sl.MAX_RESUMPTIONS):
        commit(repo, "a process")
        session = make_ledger(repo).session("H")
    assert session.resumptions == sl.MAX_RESUMPTIONS and session.must_close() is None
    commit(repo, "a process")
    session = make_ledger(repo).session("H")
    found = session.must_close()
    assert found is not None and found.reason == "not finished after 5 resumptions"


# ---------------------------------------------------------------------------
# P
# ---------------------------------------------------------------------------


def h_closed(repo: Path, statuses: dict[str, str]) -> None:
    """H read and closed, its results committed on 2026-11-20."""
    session = opened(repo)
    found = {slot: st.Assessment(slot, "H", status) for slot, status in statuses.items()}
    record = st.results_record("H", found, stamp=session.stamp, session=session.key, blind=True)
    write(repo, sr.RESULTS_PATHS["H"], json.dumps(record).encode())
    session.close()
    commit(repo, "H's results", when="2026-11-20T12:00:00Z")


def fee_amendment(repo: Path) -> None:
    write(repo, sr.CONFIG_PATH, with_p_intervals(frozen_text(repo)).encode())
    document = (repo / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    write(repo, sr.DOCUMENT_PATH, (document + "\n### 2027-01-05: the fee amendment\n").encode())
    commit(repo, "the fee amendment")


def published(name: str, plane: str, day: date) -> bool:
    return day <= date(2027, 1, 2)


def binance_only(name: str, plane: str, day: date) -> bool:
    return name == sa.BINANCE and published(name, plane, day)


H_STATUSES = {"S25-PF": "candidate", "S25-BB": "killed", "S25-T": "inconclusive"}


def test_p_opens_for_a_candidate_after_the_fee_amendment(repo: Path) -> None:
    h_closed(repo, H_STATUSES)
    fee_amendment(repo)
    ledger = make_ledger(repo, now=P_DAY)
    assert ledger.p_deadline() == date(2027, 1, 31)
    entry = ledger.open("P", candidates=["S25-PF"], published=published)
    h_entry = ledger.committed_entries()["H"][0]
    assert (
        entry["candidates"] == ["S25-PF"] and entry["h_config_sha256"] == h_entry["config_sha256"]
    )
    assert entry["config_sha256"] != entry["h_config_sha256"]
    commit(repo, "open P")
    session = make_ledger(repo, now=P_DAY).session("P")
    session.access.require(sa.BINANCE, sa.block_days("P"), sa.PRINTS)
    with pytest.raises(HeldOutLocked, match="deadline"):
        make_ledger(repo, now=datetime(2027, 2, 1, tzinfo=UTC)).session("P")


def test_p_refuses_without_its_conditions(repo: Path) -> None:
    h_closed(repo, H_STATUSES)
    ledger = make_ledger(repo, now=P_DAY)
    with pytest.raises(HeldOutLocked, match="does not cover"):
        ledger.open("P", candidates=["S25-PF"], published=published)
    fee_amendment(repo)
    with pytest.raises(HeldOutLocked, match="every candidate on H"):
        ledger.open("P", candidates=["S25-BB"], published=published)
    with pytest.raises(HeldOutLocked, match="every candidate on H"):
        ledger.open("P", published=published)
    with pytest.raises(HeldOutLocked, match="not published"):
        ledger.open("P", candidates=["S25-PF"])
    with pytest.raises(HeldOutLocked, match="not published"):
        ledger.open("P", candidates=["S25-PF"], published=binance_only)
    with pytest.raises(HeldOutLocked, match="deadline"):
        make_ledger(repo, now=datetime(2027, 2, 1, tzinfo=UTC)).open(
            "P", candidates=["S25-PF"], published=published
        )
    write(
        repo, sa.data_root(sa.BINANCE, sa.PRINTS, "data") / "BTCFDUSD-aggTrades-2026-11-03.zip", b""
    )
    with pytest.raises(HeldOutLocked, match="already on disk"):
        ledger.open("P", candidates=["S25-PF"], published=published)


def test_p_refuses_a_change_since_h_beyond_results_and_the_fee_amendment(repo: Path) -> None:
    h_closed(repo, H_STATUSES)
    fee_amendment(repo)
    write(repo, "docs/spot_2026_defects.md", b"none\n")
    write(repo, "experiments/results/spot_2026_page.json", b"{}")
    commit(repo, "allowed")
    write(repo, "tests/test_spot_heldout.py", b"# changed\n")
    commit(repo, "not allowed")
    with pytest.raises(HeldOutLocked, match="changed since H's read"):
        make_ledger(repo, now=P_DAY).open("P", candidates=["S25-PF"], published=published)


def test_p_refuses_an_edited_document_or_changed_h_results(repo: Path) -> None:
    h_closed(repo, H_STATUSES)
    fee_amendment(repo)
    document = (repo / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    write(
        repo,
        sr.DOCUMENT_PATH,
        document.replace("## Why this study", "## Why this study, again").encode(),
    )
    commit(repo, "an edit")
    with pytest.raises(HeldOutLocked, match="other than by appending"):
        make_ledger(repo, now=P_DAY).open("P", candidates=["S25-PF"], published=published)
    write(repo, sr.DOCUMENT_PATH, document.encode())
    write(repo, sr.RESULTS_PATHS["H"], b'{"block": "H"}')
    commit(repo, "H's results changed")
    with pytest.raises(HeldOutLocked, match="changed after H's read closed"):
        make_ledger(repo, now=P_DAY).open("P", candidates=["S25-PF"], published=published)


def test_copying_the_2025_files_checks_each_hash(repo: Path, tmp_path: Path) -> None:
    wanted = {p: sha(d) for p, d in FILES_2025.items()}
    copies = sl.copy_frozen_2025(repo, tmp_path / "elsewhere", wanted)
    assert len(copies) == 8 and all(
        sl.file_sha256(c) == wanted[c.relative_to(tmp_path / "elsewhere")] for c in copies
    )
    write(repo, sl._FITTED_2025, b"other")
    with pytest.raises(HeldOutLocked, match="registered sha256"):
        sl.copy_frozen_2025(repo, tmp_path / "again", wanted)


def test_p_is_read_once_for_every_candidate(repo: Path) -> None:
    h_closed(repo, {"S25-PF": "candidate", "S25-BB": "candidate", "S25-T": "killed"})
    fee_amendment(repo)
    ledger = make_ledger(repo, now=P_DAY)
    with pytest.raises(HeldOutLocked, match="every candidate on H"):
        ledger.open("P", candidates=["S25-PF"], published=published)
    entry = ledger.open("P", candidates=["S25-PF", "S25-BB"], published=published)
    assert entry["candidates"] == ["S25-BB", "S25-PF"]


# ---------------------------------------------------------------------------
# The code freeze, the configuration and the document, pinned at every open
# ---------------------------------------------------------------------------


def test_h_refuses_study_code_changed_after_the_freeze(repo: Path) -> None:
    write(repo, CODE, b"T_BAR = 0.5  # the bar lowered after D\n")
    commit(repo, "a change to the study code after the freeze, with no amendment")
    with pytest.raises(HeldOutLocked, match="study code changed after the code-freeze commit"):
        make_ledger(repo).open("H")
    for path in ("experiments/spot_2026.py", "pyproject.toml", "uv.lock"):
        write(repo, CODE, b"T_BAR = 3.0\n")
        write(repo, path, b"changed\n")
        commit(repo, f"{path} changed")
        with pytest.raises(HeldOutLocked, match="code freeze does not hold"):
            make_ledger(repo).open("H")
        (repo / path).unlink()
        commit(repo, f"{path} restored")


def test_a_later_freeze_counts_only_with_an_amendment_and_stage_ones_commit(repo: Path) -> None:
    write(repo, CODE, b"T_BAR = 3.0  # a correction\n")
    commit(repo, "a correction")
    later = freeze_code(repo, "a second code-freeze line")
    with pytest.raises(HeldOutLocked, match="no dated amendment names it"):
        make_ledger(repo).open("H")
    document = (repo / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    write(
        repo,
        sr.DOCUMENT_PATH,
        (document + f"\n### 2026-11-21: a correction\n\n`{later}`\n").encode(),
    )
    commit(repo, "the correction's amendment")
    with pytest.raises(HeldOutLocked, match="not the one stage 1 ran under"):
        make_ledger(repo).open("H")  # the frozen configuration records the first freeze


def test_h_refuses_a_configuration_changed_outside_its_placeholders(repo: Path) -> None:
    """The fee table edited after the freeze, a new null hash and the frozen line rewritten."""
    from trading_research.market_making.prereg import sha256_bytes

    old = "regular: [0.0, 10.0], vip3: [0.0, 6.0]"
    edited = TEXT.replace(old, "regular: [0.0, 2.3], vip3: [0.0, 6.0]")
    null = sha256_bytes(unfrozen_text(edited).encode())
    frozen = sr.freeze_text(edited, frozen_values(FREEZES[repo], **OURS), null_sha256=null)
    document = (
        DOCUMENT
        + f"\n### 2026-11-20: the D amendment\n\n{sr.frozen_hash_line(frozen)}\n"
        + f"\n### 2026-11-20: a later change\n\nNull-placeholder sha256: `{null}`\n"
    )
    write(repo, sr.CONFIG_PATH, frozen.encode())
    write(repo, sr.DOCUMENT_PATH, document.encode())
    commit(repo, "fees edited after the freeze, a null hash and the frozen line rewritten")
    with pytest.raises(HeldOutLocked, match="changed outside its placeholders"):
        make_ledger(repo).open("H")


def test_the_document_gains_one_frozen_hash_line_and_no_null_hash_line(repo: Path) -> None:
    document = (repo / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    extra = f"\n### 2026-11-21: another\n\nNull-placeholder sha256: `{sr.REGISTERED_SHA256}`\n"
    write(repo, sr.DOCUMENT_PATH, (document + extra).encode())
    commit(repo, "a null-hash line after the freeze")
    with pytest.raises(HeldOutLocked, match="exactly one frozen"):
        make_ledger(repo).open("H")
    line = sr.frozen_hash_line(frozen_text(repo))
    write(repo, sr.DOCUMENT_PATH, (document + f"\n{line}\n").encode())
    commit(repo, "the frozen line twice")
    with pytest.raises(HeldOutLocked, match="exactly one frozen"):
        make_ledger(repo).open("H")


def test_every_process_checks_the_freeze_and_the_pins(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened(repo)
    calls: list[str] = []
    real = sl.Ledger.verify_pinned

    def pinned(self: sl.Ledger, registration: sr.Registration, commit_: str) -> None:
        calls.append(commit_)
        real(self, registration, commit_)

    monkeypatch.setattr(sl.Ledger, "verify_pinned", pinned)
    commit(repo, "a process")
    make_ledger(repo).session("H")
    assert calls == [FREEZES[repo]]
