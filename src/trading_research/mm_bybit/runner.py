"""The stages in their registered order, each writing its record under ``experiments/results/``.

``stage0`` (the reader's proof and timing on the two reader-test days), ``stage1``
(W and D fetched, W's inputs), ``develop_d`` (every cell on D, the choice and the
nulls), ``amend`` (the D amendment), ``read_h`` (H fetched through the ledger's
access, every run, the statuses). Each refuses to start before what it reads
exists. The author commits between them; nothing is pushed.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.mm_bybit import access as acc
from trading_research.mm_bybit import archive, config, days, development, engine, freeze, funding
from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit import stages as stg

RESULTS = Path("experiments/results")
RUN_LOG = Path("docs/mm_bybit_2021_run_log.md")
CLIP_UNIT = "USDT per 100 USDT of clip a day"
D_VARIANTS = ("primary", "touch", "latency10")


def record_path(repo: Path | str, name: str) -> Path:
    return Path(repo) / RESULTS / f"mm_bybit_2021_{name}.json"


def write(repo: Path | str, name: str, record: dict[str, Any]) -> Path:
    path = record_path(repo, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return path


def read_record(repo: Path | str, name: str) -> dict[str, Any]:
    path = record_path(repo, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run its stage first")
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def stage0(repo: Path | str, root: Path | str, *, opener: Any = None) -> dict[str, Any]:
    """Fetch and prove the reader on the reader-test days; time two cells. Nothing scored."""
    cfg = config.load(repo)
    access = acc.development_access()
    kwargs = {} if opener is None else {"opener": opener}
    found: dict[str, Any] = {}
    for day in acc.READER_TEST_DAYS:
        found[day.isoformat()] = archive.fetch_day(day, root, access, **kwargs)
    funding.fetch_block("reader", root, access, days=acc.READER_TEST_DAYS, **kwargs)
    for day in acc.READER_TEST_DAYS:
        tape = stg.load_day(day, root, access, "reader", cfg)
        entry = found[day.isoformat()]
        if isinstance(tape, str):
            entry["status"] = tape
            continue
        stand_in = float(np.nanmedian(tape.sec_vol))  # timing only; W's values come at stage 1
        inputs = engine.QuoterInputs(stand_in, dict.fromkeys((0.33, 0.67, 0.9), stand_in))
        settings = days.settings_for(cfg, day, "primary")
        timing = {}
        for cell in (q.S0, q.from_key("l1|vle0.67")):
            started = time.perf_counter()
            run = engine.run_day(tape, cell, settings, inputs)
            timing[cell.key] = round(time.perf_counter() - started, 3)
            if not run.identity_ok or run.counters.get("limit_breach"):
                raise RuntimeError(f"{day} {cell.key}: the accounting or the limits failed")
        entry.update(
            status="usable", tick=tape.tick, lot=tape.lot, rows_grid=len(tape.row_ts),
            prints_on_day=len(tape.ts), max_gap_s=round(tape.max_gap_s(), 3),
            funding_settlements=[int(v) for v in tape.funding_ts],
            seconds_per_cell_day=timing,
        )  # fmt: skip
    record = {"reader_test_result": found, "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                     time.gmtime())}  # fmt: skip
    write(repo, "stage0", record)
    return record


def stage1(repo: Path | str, root: Path | str, *, opener: Any = None) -> dict[str, Any]:
    """Fetch W and D (raw files never kept), then W's sigma_ref and quantiles."""
    read_record(repo, "stage0")
    cfg = config.load(repo)
    access = acc.development_access()
    fetched = {}
    for block in ("W", "D"):
        records = stg.fetch_block(block, root, access, opener=opener)
        fetched[block] = {
            "fetched": len(records),
            "statuses": pd.Series([r["status"] for r in records]).value_counts().to_dict(),
            "gz_bytes": int(sum(r.get("gz_bytes", 0) for r in records)),
            "directions": sorted({r.get("direction", "") for r in records} - {""}),
            "rows_outside_day": int(sum(r.get("rows_outside_day", 0) for r in records)),
        }
    record = {"fetch": fetched, **stg.w_inputs(root, access, cfg)}
    write(repo, "stage1", record)
    return record


def quoter_inputs(stage1_record: dict[str, Any]) -> engine.QuoterInputs:
    quantiles = {float(k): float(v) for k, v in stage1_record["vol_quantiles_bp"].items()}
    return engine.QuoterInputs(float(stage1_record["sigma_ref_bp"]), quantiles)


def develop_d(
    repo: Path | str, root: Path | str, *, workers: int = 6, null_a_draws: int = 999
) -> dict[str, Any]:
    """Every cell on every D day (primary rule), then the choice and the nulls."""
    w = read_record(repo, "stage1")
    cfg = config.load(repo)
    # The primary rule decides; the touch bracket and the 10 ms run are measurements.
    jobs = [(cell.key, v) for cell in q.grid() for v in D_VARIANTS]
    started = time.perf_counter()
    rows, excluded, seconds = stg.run_block(
        "D", root, acc.development_access(), cfg, jobs, quoter_inputs(w), workers=workers
    )
    rows.to_parquet(Path(root) / "D_rows.parquet", index=False)
    record = development.develop(rows, null_a_draws=null_a_draws)
    usable = rows.drop_duplicates("day")
    pairs = [(float(t), float(lot)) for t, lot in zip(usable["tick"], usable["lot"], strict=True)]
    pairs += [(float(t), float(lot)) for t, lot in w["instrument_by_day_W"].values()]
    ticks = pd.Series([round(t, 10) for t, _ in pairs]).value_counts()
    lots = pd.Series([round(lot, 10) for _, lot in pairs]).value_counts()
    record["instrument_spec"] = {"tick": float(ticks.index[0]), "lot": float(lots.index[0]),
                                 "days": len(pairs),
                                 "days_other_tick": int(ticks.iloc[1:].sum()),
                                 "days_other_lot": int(lots.iloc[1:].sum())}  # fmt: skip
    record["instrument_by_day_D"] = {
        d: [float(t), float(lot)]
        for d, t, lot in zip(usable["day"], usable["tick"], usable["lot"], strict=True)
    }
    record["excluded_days_D"] = sorted([day, reason] for day, reason in excluded.items())
    record["counters"] = _counters(rows)
    record["seconds"] = {"wall": round(time.perf_counter() - started, 1),
                         "day_median": float(np.median(list(seconds.values()))),
                         "workers": workers, "cells": len(jobs)}  # fmt: skip
    write(repo, "D", record)
    return record


def _counters(rows: pd.DataFrame) -> dict[str, float]:
    """The engine's counters summed over every run (post-only rejections, pauses, ...)."""
    total: dict[str, float] = {}
    for counters in rows["counters"]:
        for k, v in counters.items():
            total[k] = total.get(k, 0.0) + float(v)
    return total


def amend(repo: Path | str, *, code_freeze_commit: str, day: date, d_runs: int = 1) -> str:
    """Write the D amendment; return the frozen configuration's sha256."""
    stage0_record, w, d = (read_record(repo, n) for n in ("stage0", "stage1", "D"))
    log = Path(repo) / RUN_LOG
    if not log.exists():
        raise FileNotFoundError(f"{log} is missing: the amendment records its sha256")
    per_cell_day = [v for entry in stage0_record["reader_test_result"].values()
                    for v in entry.get("seconds_per_cell_day", {}).values()]  # fmt: skip
    compute = {
        "code_freeze_commit": code_freeze_commit, "d_runs": d_runs,
        "run_log_sha256": config.sha256_bytes(log.read_bytes()),
        "seconds": {"D_wall": d["seconds"]["wall"], "D_day_median": d["seconds"]["day_median"],
                    "stage0_cell_day_max": max(per_cell_day) if per_cell_day else None},
    }  # fmt: skip
    values = freeze.frozen_values(stage0_record, w, d, compute)
    return freeze.amend(repo, values, day)


def _verb(value: float) -> str:
    return "made" if value > 0 else "lost"


def _kill_text(kill: dict[str, Any] | None) -> str:
    if not kill:
        return "none at n/a against n/a"
    if kill["kill"] == "K-days":
        return (f"K-days at {100 * kill['value']:.2f}% of days positive against "
                f"{100 * kill['reference']:.2f}%")  # fmt: skip
    return (f"{kill['kill']} at {kill['value']:.2f} {CLIP_UNIT} against "
            f"{kill['reference']:.2f} {CLIP_UNIT}")  # fmt: skip


def sentences(name: str, s: dict[str, Any], m: dict[str, Any]) -> list[str]:
    """The registered headline sentences, filled from the H run under the rule."""
    net, ex, making = (m["net_bp_per_fill"] or 0.0, m["ex_rebate_bp_per_fill"] or 0.0,
                       m["making_bp_per_fill"] or 0.0)  # fmt: skip
    labels = s["labels"]
    return [
        f"{name} ({s['cell']}) on Bybit BTCUSDT, 8 April to 30 June 2021, at −2.5 / +7.5 bp: "  # noqa: RUF001
        f"{s['status']}; first kill {_kill_text(s['first_kill'])}.",
        f"At the fees every account paid in 2021, a touch quoter filled only on trade-throughs "
        f"{_verb(net)} {abs(net):.2f} bp per fill ({s['mean_y']:.2f} USDT per 100 USDT of clip "
        f"a day); without the rebate it {_verb(ex)} {abs(ex):.2f} bp per fill, of which the "
        f"making part was {making:.2f} bp.",
        f"{name}: {'paid' if labels['paid_with_the_rebate'] else 'not paid'} with the rebate; "
        f"{'paid' if labels['paid_ex_rebate'] else 'not paid'} ex-rebate.",
    ]


TABLE_COLUMNS = (
    "primary",
    "cell",
    "status",
    "status_as_a_primary",
    "first_kill",
    "first_kill_value",
    "first_kill_reference",
    "mean_y",
    "t",
    "newey_west_t",
    "median_day",
    "days",
    "net_bp_per_fill",
    "ex_rebate_bp_per_fill",
    "mean_ex_rebate_y",
    "making_bp_per_fill",
    "mean_making_y",
    "fills_per_day",
    "share_days_positive",
    "placebo_95th",
    "shifted_mean",
    "nbhd_median",
    "touch_mean_y",
    "delay_mean_y",
    "latency150_mean_y",
    "latency10_mean_y",
    "capacity_median",
    "mde",
    "holm_p",
    "paid_with_the_rebate",
    "paid_ex_rebate",
    "paid_at_the_touch_only",
)


def table(record: dict[str, Any]) -> pd.DataFrame:
    """One row per primary: the registered table."""
    out = []
    for name in ("F", "S0"):
        entry = record.get("primaries", {}).get(name)
        if entry is None:
            continue
        m = entry.get("measurements", {})
        kill = entry.get("first_kill") or {}
        labels = entry.get("labels", {})
        row = {"primary": name, "cell": entry.get("cell"), "status": entry.get("status"),
               "status_as_a_primary": entry.get("status_as_a_primary"),
               "first_kill": kill.get("kill", "none"), "first_kill_value": kill.get("value"),
               "first_kill_reference": kill.get("reference"),
               **{k: entry.get(k) for k in ("mean_y", "t", "newey_west_t", "median_day", "days",
                                            "placebo_95th", "shifted_mean", "nbhd_median",
                                            "capacity_median", "mde", "holm_p")},
               **{k: m.get(k) for k in ("net_bp_per_fill", "ex_rebate_bp_per_fill",
                                        "mean_ex_rebate_y", "making_bp_per_fill",
                                        "mean_making_y", "fills_per_day", "share_days_positive",
                                        "touch_mean_y", "delay_mean_y", "latency150_mean_y",
                                        "latency10_mean_y")},
               **{k: labels.get(k) for k in ("paid_with_the_rebate", "paid_ex_rebate",
                                             "paid_at_the_touch_only")}}  # fmt: skip
        out.append(row)
    return pd.DataFrame(out, columns=list(TABLE_COLUMNS))


def read_h(
    repo: Path | str, root: Path | str, access: acc.Access, *, opener: Any = None,
    workers: int = 6,
) -> dict[str, Any]:  # fmt: skip
    """H fetched and computed under the ledger's access; every status and measurement."""
    from trading_research.mm_bybit import heldout, statuses

    cfg = config.load(repo)
    frozen = cfg["frozen"]
    name = "H_second_read" if access.second_read else "H"
    w = {"sigma_ref_bp": frozen["sigma_ref_bp"], "vol_quantiles_bp": frozen["vol_quantiles_bp"]}
    fetched = stg.fetch_block("H", root, access, opener=opener)
    chosen = frozen["chosen_cell"] if frozen["chosen_cell"] != "none" else None
    rows, excluded, seconds = stg.run_block(
        "H", root, access, cfg, heldout.jobs(chosen), quoter_inputs(w), workers=workers
    )
    usable = sorted(rows["day"].unique()) if len(rows) else []
    record: dict[str, Any] = {
        "block": "H", "second_read": access.second_read,
        "stamp": "second read" if access.second_read else "first read",
        "excluded_days_H": sorted([d, r] for d, r in excluded.items()),
        "usable_days": len(usable),
        "fetch": {"days": len(fetched),
                  "statuses": pd.Series([r["status"] for r in fetched]).value_counts().to_dict()
                  if fetched else {},
                  "gz_bytes": int(sum(r.get("gz_bytes", 0) for r in fetched)),
                  "rows_outside_day": int(sum(r.get("rows_outside_day", 0) for r in fetched))},
        "frozen_configuration_sha256": config.sha256_bytes(
            (Path(repo) / config.CONFIG_PATH).read_bytes()),
        "seconds": {"day_median": float(np.median(list(seconds.values()))) if seconds else None},
    }  # fmt: skip
    if len(excluded) > cfg["blocks"]["unreadable_share"] * len(acc.block_days("H")):
        record["primaries"] = {n: {"status": "not assessed (H unreadable)"} for n in ("F", "S0")}
        write(repo, name, record)
        return record
    if chosen and q.from_key(chosen).gated:
        sources = {date.fromisoformat(d): date.fromisoformat(usable[i - 1]) for i, d in
                   enumerate(usable)}  # fmt: skip
        extra, _, _ = stg.run_block(
            "H", root, access, cfg, [(chosen, "shifted")], quoter_inputs(w),
            day_list=sorted(sources), shift_sources=sources, workers=workers,
        )  # fmt: skip
        rows = pd.concat([rows, extra], ignore_index=True)
    rows.to_parquet(Path(root) / f"{name}_rows.parquet", index=False)
    record["counters"] = _counters(rows)
    admission, mde = frozen["admission"], frozen["mde"]
    primaries = [heldout.primary("S0", q.S0.key, rows, usable, mde=mde.get("S0", float("nan")))]
    if chosen:
        holm_one = ("same as S0" if chosen == q.S0.key else "" if admission.get("admitted")
                    else f"not admitted on D ({admission.get('reasons')})")  # fmt: skip
        primaries.insert(0, heldout.primary("F", chosen, rows, usable, holm_one=holm_one,
                                            mde=mde.get("F", float("nan"))))  # fmt: skip
    stats = statuses.assess(primaries)
    record["primaries"] = {}
    for p in primaries:
        m = heldout.measurements(rows, p.cell, usable)
        record["primaries"][p.name] = {**stats[p.name], "measurements": m,
                                       "sentences": sentences(p.name, stats[p.name], m)}  # fmt: skip
    if not chosen:
        record["primaries"]["F"] = {"status": "not run (no admissible cell on D)"}
    record["neighbours"] = {
        key: float(rows[(rows["cell"] == key) & (rows["variant"] == "primary")]
                   .set_index("day").loc[usable, "y"].mean())
        for key in sorted({k for k, v in heldout.jobs(chosen) if v == "primary"})
    }  # fmt: skip
    write(repo, name, record)
    path = record_path(repo, f"{name}_table").with_suffix(".csv")
    table(record).to_csv(path, index=False, float_format="%.6g")
    return record
