"""The pages: every registered headline sentence, the finalists page, the S25 page and the figures.

Built from what the study has computed so far and nothing else: the records of
stages D, C and E, the files stage D wrote (leaderboard, marginals, nulls), the
finalist files, and each block's results file (statuses, kills, and the numbers
:mod:`.page_numbers` adds in the read). :func:`publish` runs after stage E,
after each block's read, and on demand (the ``pages`` subcommand), and writes:

* ``experiments/results/short_horizon_2025_pages.md``: the headline sentences
  in the registration's order, the finalists on H, then F and P, the S25 page
  (its rank r of N, the edge ladder, the T1 and T2 rows), every figure, the
  rows for ``docs/findings.md`` and the README summary (the sentences and the
  figures' ``<picture>`` blocks as the README reaches them), to paste;
* ``experiments/results/short_horizon_2025_regime_map.csv``: every segment
  between alarms, per slot and block;
* the figures, under ``docs/images`` in a light and a dark theme
  (:mod:`.page_figures`).

A sentence or figure whose inputs do not exist yet is not written; the page
lists it as pending, with the reason. Every D number on the pages carries its
label; only H, F and P statuses and p_search are quoted as results.

Choices the registration leaves open, made here:

* S25's sentence carries one number twice: for a taker, the gross edge in bp of
  turnover is the fee break-even per side (``gross_bp_turnover``, the
  registration's definition of both), on H's days before 2025-08-01;
* the passive sentence gives each of S-P and S-M its status set on D, else its
  status on H, and waits for H when either was admitted;
* R's sentence is written once per slot taken to F, the slot named beside it
  (the template has no slot), and once as "not tested" when no slot was taken
  to F; a void R reads as not supported;
* the regime-map sentence counts the held-out segments only (H, F and P): D's
  segments are on the map and in its file, but a D number is never quoted;
* the finalists page has a row on H for every slot and S25-Z, and on F and P
  for the slots assessed there; a slot whose status was set on D has no
  statistics in its row, and its measured numbers are in a table of their own;
* the ladder's A0 rows are A0's mean daily net at 0/0 crossing, and its net
  over the filled notional under each fill rule (O's days pooled); S25's D
  gross is the leaderboard's, blank until the trial ledger records a gross;
* the regime map's charts keep units apart: the taker slots in bp of traded
  notional, S-P and S-M per 100 USDT of clip a day; S25-Z, whose net is at
  0/0, is in the file and not on a chart.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property, partial
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon.context import Study
from trading_research.reporting.headlines import HeadlineError

PAGES_FILE = "pages.md"
REGIME_MAP_FILE = "regime_map.csv"
BLOCKS = ("H", "F", "P")
NAN = math.nan


def _number(value: Any) -> float:
    """A JSON number as a float; None (a non-finite number written out) as NaN."""
    return NAN if value is None else float(value)


@dataclass
class Computed:
    """What the study has computed so far: complete stages' records, the files
    stage D wrote, the finalist files and the blocks' results files."""

    study: Study

    def record(self, stage: str) -> dict[str, Any] | None:
        return self.study.record(stage) if self.study.state.complete(stage) else None

    @cached_property
    def d(self) -> dict[str, Any] | None:
        return self.record("D")

    @cached_property
    def c(self) -> dict[str, Any] | None:
        return self.record("C")

    @cached_property
    def e(self) -> dict[str, Any] | None:
        return self.record("E")

    def results(self, block: str) -> dict[str, Any] | None:
        from trading_research.pipeline.short_horizon.reads import results_path

        path = results_path(self.study, block)
        if not path.exists():
            return None
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    @cached_property
    def read(self) -> dict[str, dict[str, Any]]:
        """Each block read so far, by name, in the registered order."""
        found = {block: self.results(block) for block in BLOCKS}
        return {block: r for block, r in found.items() if r is not None}

    def numbers(self, block: str, slot: str) -> dict[str, Any] | None:
        """A slot's numbers in a block's read (None when the read gave none)."""
        results = self.read.get(block) or {}
        found: dict[str, Any] | None = results.get("pages", {}).get("slots", {}).get(slot)
        return found

    def measurement(self, block: str, slot: str, span: str) -> dict[str, Any] | None:
        results = self.read.get(block) or {}
        found: dict[str, Any] | None = results.get("measurements", {}).get(slot, {}).get(span)
        return found

    @cached_property
    def finalists(self) -> dict[str, dict[str, Any]]:
        """The frozen finalist files, by slot."""
        from trading_research.pipeline.short_horizon_ledger import FINALISTS_DIR

        folder = self.study.repo / FINALISTS_DIR
        if not folder.exists():
            return {}
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}  # fmt: skip

    @cached_property
    def board(self) -> pd.DataFrame | None:
        from trading_research.reporting.leaderboard import read_leaderboard

        path = self.study.results("leaderboard.csv")
        return read_leaderboard(path)[1] if self.d is not None and path.exists() else None


@dataclass
class Written:
    """The sentences written so far, by registered name, and what is pending."""

    sentences: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)

    def attempt(self, name: str, about: str, make: Callable[[], str]) -> None:
        """Write one sentence; a refusal by the registered fillers is pending, named so."""
        try:
            sentence = make()
        except HeadlineError as exc:
            self.pending.append(f"{name} ({about}): refused: {exc}")
            return
        self.sentences.setdefault(name, []).append({"about": about, "sentence": sentence})

    def attempt_each(self, name: str, about: str, make: Callable[[], list[str]]) -> None:
        """Write the sentences ``make`` returns (one per admitted slot, say)."""
        try:
            found = make()
        except HeadlineError as exc:
            self.pending.append(f"{name} ({about}): refused: {exc}")
            return
        for sentence in found:
            self.sentences.setdefault(name, []).append({"about": about, "sentence": sentence})

    def wait(self, name: str, why: str) -> None:
        self.pending.append(f"{name}: {why}")

    def plain(self) -> dict[str, list[str]]:
        """The sentences alone, keyed by the registration's names."""
        return {name: [x["sentence"] for x in rows] for name, rows in self.sentences.items()}


# ---------------------------------------------------------------------------
# The headline sentences beyond each slot's own
# ---------------------------------------------------------------------------


def search_results(study: Study, record_d: dict[str, Any]) -> dict[str, Any]:
    """Each stratum's chosen cell against both nulls, as stage D recorded it."""
    from trading_research.validation.placebo_search import SearchNullResult

    out = {}
    for stratum, n in record_d["nulls"].items():
        cell = record_d["choice"][f"{stratum}*"]["cell"]
        at = None if cell is None else study.keys.index(cell) if cell in study.keys else 0
        refit = None if n.get("A_refit") is None else float(n["A_refit"])
        out[stratum] = SearchNullResult(stratum, at, _number(n["A"]), int(n["count_A"]), int(n["count_B"]),
                                        _number(n["p_A"]), _number(n["p_B"]), refit, bool(n["passes_iv"]))  # fmt: skip
    return out


def admitted_on_d(c: Computed) -> list[str]:
    """The searched slots admitted on D (S-P and S-M once stage C has run)."""
    assert c.d is not None
    found = [slot for slot, status in c.d["statuses_on_D"].items() if status == "admitted"]
    if c.c is not None:
        found += [s for s in ("S-P", "S-M") if c.c["admission"][s]["status_on_D"] == "admitted"]
    return found


def development_sentences(c: Computed, out: Written) -> None:
    """The search sentence and the payoff sentence, from stage D (and C)."""
    from trading_research.reporting import headlines as hl

    if c.d is None:
        out.wait("search_on_D", "stage D is not complete")
        out.wait("payoff_on_D", "stage D is not complete")
        return
    record_d = c.d
    nulls = search_results(c.study, record_d)
    n_trials, runs = int(record_d["N_trials"]), int(record_d["R_D"])
    out.attempt("search_on_D", "D", lambda: hl.search_on_d(n_trials, nulls, runs))
    if c.c is None:
        out.wait("payoff_on_D", "S-P's and S-M's admission waits for stage C")
        return
    out.attempt_each("payoff_on_D", "D", lambda: hl.payoff_on_d(admitted_on_d(c)))


def assessed_slots(c: Computed, block: str) -> list[str]:
    """The slots a block's statuses were taken on: every slot on H, those taken
    to F on F, those qualifying for P on P."""
    from trading_research.reporting.finalists import ROW_ORDER

    if block == "H":
        return list(ROW_ORDER)
    earlier = c.read.get({"F": "H", "P": "F"}[block]) or {}
    chosen = set(earlier.get("taken_to_F" if block == "F" else "qualifying_for_P", []))
    return [slot for slot in ROW_ORDER if slot in chosen]


def kill_checks(results: dict[str, Any], slot: str) -> tuple[Any, ...]:
    from trading_research.market_making.verdicts import KillCheck

    found = results.get("assessments", {}).get(slot, {}).get("kills", [])
    return tuple(KillCheck(**k) for k in found)


def slot_sentences(c: Computed, out: Written) -> None:
    """Each slot's sentence on each block read, for the slots assessed there."""
    from trading_research.reporting.headlines import slot_on_block

    for block, results in c.read.items():
        for slot in assessed_slots(c, block):
            status = results["statuses"].get(slot)
            if status is None:
                out.wait("slot_on_block", f"{slot} has no status on {block}")
                continue
            kills = kill_checks(results, slot)
            out.attempt("slot_on_block", f"{slot} on {block}",
                        partial(slot_on_block, slot, block, status, kills))  # fmt: skip


def s25_sentence(c: Computed, out: Written) -> None:
    """S25's gross edge and break-even on H's days to 07-31, and its status there."""
    from trading_research.reporting import headlines as hl

    h = c.read.get("H")
    if h is None:
        out.wait("S25", "H is not read yet")
        return
    numbers = c.numbers("H", "S25") or {}
    edge = _number(numbers.get("break_even_per_side"))
    status = h["statuses"]["S25"]
    out.attempt("S25", "H before 2025-08-01", lambda: hl.s25_sentence(edge, edge, status))


def passive_sentence(c: Computed, out: Written) -> None:
    """S-P's and S-M's statuses on their signal, and S-M's making part."""
    from trading_research.reporting import headlines as hl

    if c.c is None:
        out.wait("passive_and_maker", "stage C is not complete")
        return
    admission = c.c["admission"]
    h = c.read.get("H")
    statuses = {}
    for slot in ("S-P", "S-M"):
        on_d = admission[slot]["status_on_D"]
        if on_d != "admitted":
            statuses[slot] = on_d
        elif h is not None:
            statuses[slot] = h["statuses"][slot]
        else:
            out.wait("passive_and_maker", f"{slot} was admitted on D and H is not read yet")
            return
    signal = admission["S-P"]["signal"]

    def sentence() -> str:  # the making part is H's when S-M has a status on H, else D's
        making = _number(admission["S-M"]["making_bp_per_fill"])
        if hl.has_held_out_status(statuses["S-M"]):
            making = _number((c.numbers("H", "S-M") or {}).get("making_bp_per_fill"))
        return hl.passive_and_maker(signal, statuses["S-P"], statuses["S-M"], making)

    out.attempt("passive_and_maker", signal, sentence)


def attribution(record: dict[str, Any] | None) -> Any:
    """A regime attribution as a read records it (``pipeline.regime.Attribution``'s
    fields; ``random_p95`` and ``flags`` accepted for ``reference`` and ``n_flags``)."""
    from trading_research.pipeline.regime import Attribution

    if not record or record.get("draws") is None:
        return None
    reference = record.get("reference", record.get("random_p95"))
    return Attribution(str(record.get("policy", "")), _number(record["difference"]), _number(reference),
                       np.asarray(record["draws"], dtype=np.float64), int(record.get("n_flags", record.get("flags", 0))))  # fmt: skip


def regime_axis_sentences(c: Computed, out: Written) -> None:
    """One per slot with a window above 0, against random pauses on H; the
    attribution is read where H's read records it, beside the slot's status
    (``measurements[slot]["regime_attribution"]``)."""
    from trading_research.reporting import headlines as hl

    h = c.read.get("H")
    if h is None:
        out.wait("regime_axis", "H is not read yet")
        return
    if not h.get("pages"):
        out.wait("regime_axis", "H's results carry no numbers for the pages")
    for slot, numbers in (h.get("pages") or {}).get("slots", {}).items():
        window = int(numbers.get("window_h", 0))
        if window <= 0:
            continue
        beside = (h.get("measurements") or {}).get(slot) or {}
        found = attribution(beside.get("regime_attribution"))
        if found is None:
            out.wait("regime_axis", f"{slot}'s regime attribution is not in H's results")
            continue
        out.attempt("regime_axis", slot, partial(hl.regime_axis, slot, window, found))


def held_out_segments(c: Computed) -> pd.DataFrame:
    """Every held-out block's regime-map segments, in the order read."""
    from trading_research.reporting.regime_map import SEGMENT_COLUMNS

    rows = [row for results in c.read.values() for row in results.get("regime_map", [])]
    frame = pd.DataFrame(rows, columns=list(SEGMENT_COLUMNS))
    for column in ("start", "end"):  # the reads write ISO stamps; the map's days are dates
        frame[column] = [pd.Timestamp(v).date() for v in frame[column]]
    return frame


def regime_map_sentences(c: Computed, out: Written) -> None:
    """For each slot on the held-out map: its segments with a positive mean."""
    from trading_research.reporting import headlines as hl

    segments = held_out_segments(c)
    if segments.empty:
        out.wait("regime_map", "no held-out block has a segment yet")
        return
    for slot in pd.unique(segments["slot"]):
        out.attempt("regime_map", str(slot), partial(hl.regime_map_sentence, str(slot), segments))


def registered_reason(reason: str, results: dict[str, Any]) -> str:
    """R's untestable reason in the registration's words (the read's may carry counts)."""
    from trading_research.reporting.headlines import UNTESTABLE_REASONS

    if reason in UNTESTABLE_REASONS:
        return reason
    if "alarm rate" in reason:
        return UNTESTABLE_REASONS[3]
    if reason.endswith("in the reference set besides b"):
        return UNTESTABLE_REASONS[4]
    if reason.startswith("fewer than 20 usable days before b"):
        split = results.get("split", {})
        b = split.get("b")
        info = results.get("info", {})
        days = sorted(set(info.get("usable_days", [])) - set(info.get("stale_days", [])))
        before = sum(d < str(b) for d in days) if b else 0
        return UNTESTABLE_REASONS[1] if before < 20 else UNTESTABLE_REASONS[2]
    return reason


def break_sentences(c: Computed, out: Written) -> None:
    """R once per slot taken to F; "not tested" when none was."""
    from trading_research.reporting import headlines as hl

    f = c.read.get("F")
    if f is None:
        out.wait("break", "F is not read yet")
        return
    tested = f.get("R", {})
    if not tested:
        out.attempt("break", "every slot", lambda: hl.break_sentence("not tested"))
        return
    for slot, r in tested.items():
        outcome = "void" if f.get("R_void") and r["outcome"] != "not tested" else r["outcome"]
        reason = registered_reason(r.get("reason", ""), f) if outcome == "untestable" else ""
        out.attempt("break", slot, partial(hl.break_sentence, outcome, reason))


def sentences(c: Computed) -> Written:
    """Every registered sentence the computed results can fill, in the table's order."""
    out = Written()
    development_sentences(c, out)
    slot_sentences(c, out)
    s25_sentence(c, out)
    passive_sentence(c, out)
    regime_axis_sentences(c, out)
    regime_map_sentences(c, out)
    break_sentences(c, out)
    return out


# ---------------------------------------------------------------------------
# The finalists page and the S25 page
# ---------------------------------------------------------------------------


def finalist_inputs(c: Computed, block: str, slot: str) -> Any:
    """One row of the finalists page: the status and kills from the read, the
    numbers on the status days, the frozen MDE and H's mean before 08-01."""
    from trading_research.reporting.finalists import FinalistInputs
    from trading_research.reporting.headlines import D_STATUS

    results = c.read[block]
    status = results["statuses"][slot]
    mde = (c.e or {}).get("mde", {}).get(slot)
    common = {"slot": slot, "block": block, "status": status, "kills": kill_checks(results, slot),
              "mde": NAN if mde is None else _number(mde[1])}  # fmt: skip
    if D_STATUS.fullmatch(status):  # a measured row: its numbers are in a table of their own
        return FinalistInputs(daily=(), trades=0, **common)
    common["h_before_0801_mean"] = _number((c.numbers("H", slot) or {}).get("h_before_0801_mean"))
    numbers = c.numbers(block, slot)
    if numbers is None:  # the block gave no series (unreadable, or its read failed)
        return FinalistInputs(daily=(), trades=0, **common)
    found = attribution((results.get("regime_attribution") or {}).get(slot))
    return FinalistInputs(daily=np.asarray(numbers["values"], dtype=np.float64), trades=int(numbers["trades"]),
                          placebo_percentile=_number(numbers.get("placebo_percentile")),
                          k_nbhd_median=_number(numbers.get("k_nbhd_median")),
                          break_even_per_side=_number(numbers.get("break_even_per_side")),
                          tiers_paying=numbers.get("tiers_paying"), attribution=found, **common)  # fmt: skip


def finalists_page(c: Computed, out: Written) -> pd.DataFrame | None:
    """The finalists on every block read, H first; a row the registered sentence
    refuses is left out and named as pending."""
    from trading_research.reporting.finalists import finalist_row, finalists_table

    rows = []
    for block in c.read:
        for slot in assessed_slots(c, block):
            if slot not in c.read[block]["statuses"]:
                continue
            row = finalist_inputs(c, block, slot)
            try:
                finalist_row(row)
            except HeadlineError as exc:
                out.pending.append(f"finalists ({slot} on {block}): refused: {exc}")
                continue
            rows.append(row)
    return finalists_table(rows) if rows else None


MEASURED_COLUMNS = (
    "block",
    "slot",
    "span",
    "days",
    "trades",
    "mean",
    "t",
    "newey_west_t",
    "median_day",
)
SPANS = {"all_of_H": "all of H", "all_of_F": "all of F", "all_of_P": "all of P",
         "H_before_2025_08_01": "H before 2025-08-01",
         "gap_days_included": "status window, gap days put back"}  # fmt: skip


def measured_rows(c: Computed) -> pd.DataFrame:
    """The measurements beside the statuses: every slot's numbers on all of each
    block read, on H before 2025-08-01, on its status window with the gap days put
    back, and each measured series (the regime measurements, the passive exit, the
    control network's IC) under its own name; no status, no placebo. The regime
    attribution and FILL's tables stand beside them and are read elsewhere."""
    rows = []
    for block, results in c.read.items():
        for slot, spans in results.get("measurements", {}).items():
            series = {span: m for span, m in spans.items() if isinstance(m, dict) and "days" in m}
            series |= {name: m for name, m in (spans.get("measured") or {}).items()
                       if isinstance(m, dict) and "days" in m}  # fmt: skip
            for span, m in series.items():
                rows.append({"block": block, "slot": slot, "span": SPANS.get(span, span),
                             **{k: m.get(k) for k in MEASURED_COLUMNS[3:]}})  # fmt: skip
    frame = pd.DataFrame(rows, columns=list(MEASURED_COLUMNS))
    for column in MEASURED_COLUMNS[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def a0_rows(
    record_c: dict[str, Any] | None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """A0 at 0/0 for the ladder: crossing (the mean daily net) and posted under each
    fill rule (net over the filled notional), and FILL's fill ratios, on O."""
    found = (record_c or {}).get("a0_and_fill") or {}
    if not found.get("run"):
        return None, None
    a0: dict[str, float] = {}
    daily = [_number(day["net_bp"]) for day in found["crossing"] if "net_bp" in day]
    finite = [v for v in daily if math.isfinite(v)]
    if daily:
        a0["crossing"] = float(np.mean(finite)) if finite else NAN
    for row in found["fill_summary"]:
        if "notional_filled" in row:
            notional = _number(row["notional_filled"])
            a0[row["rule"]] = _number(row["net"]) / notional * 1e4 if notional > 0 else NAN
    ratios = {row["rule"]: _number(row["fill_ratio"]) for row in found["fill_summary"]}
    return a0, ratios


def ladder(c: Computed) -> pd.DataFrame:
    """S25's edge ladder with every step its block has been read for."""
    from trading_research.reporting.edge_ladder import edge_ladder
    from trading_research.reporting.search_keys import S25_KEY

    a0, ratios = a0_rows(c.c)
    s25_d = None
    if c.board is not None and (c.board["key"] == S25_KEY).any():
        row = c.board.loc[c.board["key"] == S25_KEY].iloc[0]
        s25_d = (_number(row["gross"]), _number(row["net"]))

    def h(name: str, slot: str = "S25") -> float | None:
        return None if "H" not in c.read else _number((c.numbers("H", slot) or {}).get(name))

    def whole(block: str, span: str) -> float | None:
        if block not in c.read:
            return None
        return _number((c.measurement(block, "S25", span) or {}).get("mean"))

    return edge_ladder(a0_zero_fee=a0, s25_d=s25_d, s25_h_window=h("mean"), s25_h_all=whole("H", "all_of_H"),
                       s25_f=whole("F", "all_of_F"), s25_break_even=h("break_even_per_side"),
                       s25z_net_zero_fee=h("net_zero_fee_mean", "S25-Z"), fill_ratios=ratios)  # fmt: skip


def s25_page(c: Computed, table: pd.DataFrame, out: Written) -> str | None:
    """The S25 page: its rank r of N on the D leaderboard, the ladder, T1 and T2."""
    from trading_research.reporting.finalists import s25_page as page
    from trading_research.reporting.leaderboard import region_rows, s25_rank
    from trading_research.reporting.search_keys import catalogue_table
    from trading_research.validation.choice import ChoiceError

    if c.board is None:
        out.wait("the S25 page", "stage D's leaderboard is not written yet")
        return None
    try:
        regions = region_rows(c.board, catalogue_table(list(c.board["key"])))
        rank = s25_rank(c.board)
    except ChoiceError as exc:
        out.pending.append(f"the S25 page: refused: {exc}")
        return None
    return page(table, regions, rank)


# ---------------------------------------------------------------------------
# The regime map's file, the pages file, and publishing both with the figures
# ---------------------------------------------------------------------------


def regime_map_rows(c: Computed) -> pd.DataFrame:
    """D's segments (once stages D and C are complete) and each held-out block's."""
    from trading_research.pipeline.short_horizon.page_numbers import d_regime_map

    parts = [d_regime_map(c.study)] if c.d is not None and c.c is not None else []
    parts.append(held_out_segments(c))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else held_out_segments(c)


def write_regime_map(c: Computed, rows: pd.DataFrame) -> Any:
    from trading_research.reporting.regime_map import SEGMENT_COLUMNS

    path = c.study.results(REGIME_MAP_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# a measurement, never a status; D's segments are on development data\n")
        rows.loc[:, list(SEGMENT_COLUMNS)].to_csv(handle, index=False, float_format="%.6g")
    return path


def _bullets(rows: list[str]) -> str:
    return "\n".join(f"- {row}" for row in rows) if rows else "- none yet"


REGISTER_HEADING = "### The 2025 short-horizon search on BTCUSDT"


def register_rows(c: Computed, written: Written) -> str:
    """The rows for ``docs/findings.md``: the study's entry, pre-registered until a
    block is read, then each slot's latest registered sentence (its status on
    the last block it was assessed on, with the first kill there)."""
    from trading_research.reporting.finalists import ROW_ORDER

    latest = {row["about"].split(" on ")[0]: row["sentence"]
              for row in written.sentences.get("slot_on_block", [])}  # fmt: skip
    read = ", ".join(c.read) or "none"
    status = f"read through {list(c.read)[-1]}" if c.read else "pre-registered, no block read"
    head = (f"**Status: {status}** (blocks read: {read}). Every status is computed by the entry "
            "point and quoted from `experiments/results/short_horizon_2025_pages.md`; a D number "
            "is never one.")  # fmt: skip
    rows = [latest[slot] for slot in ROW_ORDER if slot in latest]
    return "\n\n".join([REGISTER_HEADING, head, _bullets(rows)])


def readme_summary(written: Written, figures: dict[str, str]) -> str:
    """The README's summary: the headline sentences beyond the per-slot ones, in
    the registration's order, then every figure's ``<picture>`` block with the
    images as the README reaches them."""
    from trading_research.pipeline.short_horizon.page_figures import FROM_PAGES, IMAGES
    from trading_research.reporting.headlines import TEMPLATES

    sentences = [row["sentence"] for name in TEMPLATES if name != "slot_on_block"
                 for row in written.sentences.get(name, [])]  # fmt: skip
    pictures = [
        block.replace(f"{FROM_PAGES}/", f"{IMAGES.as_posix()}/") for block in figures.values()
    ]
    return "\n\n".join([" ".join(sentences) or "No sentence written yet.", *pictures])


def markdown(c: Computed, written: Written, finalists: pd.DataFrame | None,
             page: str | None, figures: dict[str, str]) -> str:  # fmt: skip
    """The pages file: sentences, finalists, the S25 page and every figure."""
    from trading_research.reporting.finalists import markdown_table
    from trading_research.reporting.headlines import TEMPLATES

    read = ", ".join(c.read) or "none"
    parts = ["# The 2025 short-horizon search: its pages",
             f"Written by the entry point from the stages' records and the blocks' results; blocks read: "
             f"{read}. Only H, F and P statuses and p_search are quoted as results; every D number is "
             "labelled on development data, selected.", "## Headline sentences"]  # fmt: skip
    rows = [(name, row) for name in TEMPLATES for row in written.sentences.get(name, [])]
    shown = [f"**{row['about']}**: {row['sentence']}" if name == "break" else row["sentence"]
             for name, row in rows]  # fmt: skip
    parts += [_bullets(shown), "### Pending", _bullets(written.pending)]
    sections = (("The placebo search", ("placebo_search",)),
                ("The D leaderboard and marginals", ("leaderboard", "marginals")))  # fmt: skip
    for title, names in sections:
        parts += [f"## {title}", *[figures[n] for n in names if n in figures]]
    parts.append("## The finalists on H, then F and P")
    if finalists is not None:
        parts.append(markdown_table(finalists.drop(columns=["sentence"])))
    else:
        parts.append("No block read yet.")
    measured = measured_rows(c)
    if len(measured):
        parts += [
            "Measurements beside the statuses (no status, no placebo):",
            markdown_table(measured),
        ]
    parts += [
        "## S25",
        page or "Not written yet.",
        *[figures[n] for n in ("edge_ladder", "placebo_s25") if n in figures],
    ]
    parts += [
        "## The regime map",
        *[figures[n] for n in sorted(figures) if n.startswith("regime_map")],
    ]
    parts += ["## The break detector", figures.get("break_view", "Drawn once H is read.")]
    parts += ["## The fill rule", figures.get("fill_rules", "Drawn once stage C is complete.")]
    parts += ["## Rows for docs/findings.md", register_rows(c, written),
              "## The README summary", readme_summary(written, figures)]  # fmt: skip
    return "\n\n".join(parts) + "\n"


def publish(study: Study) -> dict[str, Any]:
    """Write the pages file, the regime map's file and the figures from what has
    been computed; return what was written, every sentence and what is pending."""
    from trading_research.pipeline.short_horizon.page_figures import render

    c = Computed(study)
    written = sentences(c)
    finalists = finalists_page(c, written)
    table = ladder(c)
    page = s25_page(c, table, written)
    segments = regime_map_rows(c)
    files = [write_regime_map(c, segments)] if len(segments) else []
    figures, paths = render(c, table, written)
    target = study.results(PAGES_FILE)
    target.write_text(markdown(c, written, finalists, page, figures), encoding="utf-8")
    files += [target, *paths]
    return {"written": sorted(str(p.relative_to(study.repo)) for p in files),
            "sentences": written.plain(), "pending": written.pending}  # fmt: skip


def publish_or_note(study: Study) -> dict[str, Any]:
    """:func:`publish`, its failure recorded instead of raised: a stage's record or a
    read's closed session is never lost to its pages, which ``pages`` writes again."""
    try:
        return publish(study)
    except Exception as exc:  # recorded; ``pages`` writes them again on demand
        return {"failed": f"{type(exc).__name__}: {exc}"}


def held_out_days(c: Computed) -> dict[str, tuple[date, date]]:
    """Each block's span on the views: D, then every block read."""
    from trading_research.data import study_access as sa

    return {block: sa.BLOCKS[block] for block in ("D", *c.read)}
