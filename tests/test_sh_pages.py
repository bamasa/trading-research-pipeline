"""The pages from computed results: the numbers a read adds, every sentence, the finalists page, the ladder."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline.regime import Attribution
from trading_research.pipeline.short_horizon import page_numbers as pn
from trading_research.pipeline.short_horizon import pages
from trading_research.pipeline.short_horizon import statuses as st
from trading_research.pipeline.short_horizon.reads import regime_segments
from trading_research.reporting import headlines as hl
from trading_research.reporting.search_keys import S25_KEY

C_KEY = "BTCUSDT|C|logistic|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=pull_4h"
H_DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(44)]
F_DAYS = [date(2025, 8, 21) + timedelta(days=i) for i in range(60)]


def taker(slot: str = "S25", *, mean: float = -2.0, on_s25: bool = True, window: int = 0,
          days: list[date] | None = None, seed: int = 0) -> st.SlotDays:  # fmt: skip
    days = days or H_DAYS
    n, rng = len(days), np.random.default_rng(seed)
    value = mean + rng.standard_normal(n)
    return st.SlotDays(
        slot, "taker", list(days), value, np.full(n, 10), gross=np.full(n, 3.0),
        turnover=np.full(n, 1_000.0), ic=np.full(n, 0.01), random=rng.normal(-3, 1, (50, n)),
        flipped=np.full(n, -4.0), shifted=np.full(n, -4.0), long_only=np.full(n, -4.0),
        short_only=np.full(n, -4.0), neighbours=np.vstack([value - 1, value + 0.5, value]),
        latency=value - 0.1, on_s25_signal=on_s25, window_h=window,
    )  # fmt: skip


def maker(seed: int = 1) -> st.SlotDays:
    s = taker("S-M", mean=0.5, on_s25=True, seed=seed)
    s.kind = "maker"
    s.making = np.full(len(s.days), -0.2)
    s.passive_notional = np.full(len(s.days), 4_000.0)
    s.brackets = {"pessimistic": s.value - 1.0}
    return s


def test_the_status_days_are_the_registered_windows() -> None:
    on_s25, other = taker(), taker("S-C", on_s25=False)
    assert pn.status_days("H", on_s25) == [d for d in H_DAYS if d <= date(2025, 7, 31)]
    assert len(pn.status_days("H", on_s25)) == 24 and pn.status_days("H", other) == H_DAYS
    f = taker("S-C", on_s25=False, days=F_DAYS)
    split = {"b": "2025-09-20", "on_all_of_F": False}
    assert pn.status_days("F", f, split) == [d for d in F_DAYS if d < date(2025, 9, 20)]
    assert pn.status_days("F", f, split | {"on_all_of_F": True}) == F_DAYS
    assert pn.status_days("P", f) == F_DAYS


def test_a_slots_numbers_are_the_ones_its_kills_read() -> None:
    s = taker("S-C", mean=1.0, on_s25=False)
    days = pn.status_days("H", s)
    numbers = pn.slot_numbers(s, days)
    checks = {k.name: k for k in st.kills(s, s.mask(days))}
    assert numbers["break_even_per_side"] == pytest.approx(checks["K-fee"].value) == 30.0
    assert numbers["k_nbhd_median"] == pytest.approx(checks["K-nbhd"].value)
    assert numbers["mean"] == pytest.approx(checks["K1"].value)
    draws = [float(row.mean()) for row in s.random]
    assert numbers["placebo_percentile"] == hl.percentile_of(numbers["mean"], draws)
    assert numbers["trades"] == 440 and len(numbers["values"]) == numbers["status_days"] == 44
    assert numbers["window_days"] == 44


def test_the_making_part_and_the_net_at_zero_fees_are_kept() -> None:
    m = maker()
    numbers = pn.slot_numbers(m, m.days)
    k_dir = next(k for k in st.kills(m, m.mask(m.days)) if k.name == "K-dir")
    assert numbers["making_bp_per_fill"] == pytest.approx(k_dir.value) == pytest.approx(-0.5)
    z = taker("S25-Z")
    z.second = z.value - 1.0
    assert pn.slot_numbers(z, H_DAYS)["net_zero_fee_mean"] == pytest.approx(z.value.mean() - 1.0)


# ---------------------------------------------------------------------------
# What a stage and a read leave, as the pages read it
# ---------------------------------------------------------------------------


def record_d(**statuses: str) -> dict[str, Any]:
    on_d = {"S-C": "not admitted on D (i)", "S-R": "not run (no admissible cell on D)"}
    nulls = {
        "C": {"A": 1.2, "count_A": 29, "count_B": 3, "p_A": 0.3, "p_B": 0.25, "A_refit": 0.4,
              "passes_iv": False},
        "R": {"A": 0.0, "count_A": 99, "count_B": 7, "p_A": 1.0, "p_B": 1.0, "A_refit": None,
              "passes_iv": False},
    }  # fmt: skip
    return {"N_trials": 2400, "R_D": 1, "statuses_on_D": on_d | statuses, "nulls": nulls,
            "choice": {"C*": {"cell": C_KEY}, "R*": {"cell": None}}}  # fmt: skip


def record_c(s_p: str = "not admitted on D (i)", s_m: str = "not admitted on D (i)") -> dict[str, Any]:  # fmt: skip
    summary = [{"rule": r, "fill_ratio": f, "net": -0.2, "notional_filled": 4_000.0}
               for r, f in (("touch", 0.8), ("trade_through", 0.3), ("queue", 0.45))]  # fmt: skip
    crossing = [{"day": "2025-02-04", "net_bp": -1.0}, {"day": "2025-02-05", "net_bp": -3.0}]
    return {"admission": {"S-P": {"status_on_D": s_p, "signal": "S25", "making_bp_per_fill": NAN},
                          "S-M": {"status_on_D": s_m, "signal": "S25", "making_bp_per_fill": -0.4}},
            "a0_and_fill": {"run": True, "crossing": crossing, "fill_summary": summary}}  # fmt: skip


NAN = math.nan


def read_h(series: dict[str, st.SlotDays], preset: dict[str, str]) -> dict[str, Any]:
    """H's results as a read writes them: statuses, kills, the numbers and the map."""
    masks = {slot: s.mask(pn.status_days("H", s)) for slot, s in series.items()}
    found = dict(st.family(series, masks, "H", preset))
    if "S25-Z" in series:
        found["S25-Z"] = st.s25_z(series["S25-Z"], masks["S25-Z"])
    numbers = {}
    for slot, s in series.items():
        numbers[slot] = pn.slot_numbers(s, pn.status_days("H", s)) | {"window_h": s.window_h,
            "h_before_0801_mean": pn.slot_numbers(s, pn.status_days("H", taker()))["mean"],
            "daily": {"days": [d.isoformat() for d in s.days], "value": list(s.value)}}  # fmt: skip
    taken = [
        k for k, a in found.items() if k in series and st.taken_to_f(a, series[k].on_s25_signal)
    ]
    return {"statuses": {k: a.status for k, a in found.items()},
            "assessments": {k: a.record() for k, a in found.items()}, "taken_to_F": taken,
            "pages": {"slots": numbers, "alarms": []}, "measurements": {},
            "regime_map": regime_segments(series, [], "H")}  # fmt: skip


def computed(**parts: Any) -> pages.Computed:
    """A ``Computed`` holding the given records and results, nothing read from disk."""
    found = pages.Computed(SimpleNamespace(keys=(C_KEY, S25_KEY)))  # type: ignore[arg-type]
    for name in ("d", "c", "e", "board"):
        found.__dict__[name] = parts.get(name)
    found.__dict__["read"] = parts.get("read", {})
    return found


def h_series() -> dict[str, st.SlotDays]:
    """S25 killed, S-C a measured row (its status set on D), S-M on S25's signal."""
    z = taker("S25-Z", mean=-1.0, seed=3)
    z.second = z.value - 2.0
    return {"S25": taker(), "S-C": taker("S-C", on_s25=False, window=4, seed=2), "S-M": maker(),
            "S25-Z": z}  # fmt: skip


PRESET = {"S-C": "not admitted on D (i)", "S-R": "not run (no admissible cell on D)",
          "S-P": "not admitted on D (i)", "S-M": "not admitted on D (i)"}  # fmt: skip


def written(c: pages.Computed) -> pages.Written:
    out = pages.sentences(c)
    assert not [p for p in out.pending if "refused" in p], out.pending
    for name, rows in out.plain().items():
        for sentence in rows:
            assert hl.pattern(name).fullmatch(sentence), sentence
    return out


def test_with_nothing_computed_every_sentence_is_pending() -> None:
    out = written(computed())
    assert out.plain() == {}
    waiting = {p.split(":")[0] for p in out.pending}
    assert waiting >= {"search_on_D", "payoff_on_D", "S25", "passive_and_maker", "regime_axis",
                       "regime_map", "break"}  # fmt: skip


def test_the_development_sentences_follow_stages_d_and_c() -> None:
    out = written(computed(d=record_d()))
    (search,) = out.plain()["search_on_D"]
    assert search.startswith("Searched 2,400 configurations (stratum, model")
    assert "was not distinguishable" in search and "(p_search = 0.3; Null A p = 0.3" in search
    assert "payoff_on_D" not in out.plain() and any("stage C" in p for p in out.pending)
    out = written(computed(d=record_d(**{"S-C": "admitted"}), c=record_c(s_p="admitted")))
    assert out.plain()["payoff_on_D"] == [
        "S-C was admitted on development data, selected.",
        "S-P was admitted on development data, selected.",
    ]
    nothing = written(computed(d=record_d(), c=record_c()))
    assert nothing.plain()["payoff_on_D"] == [hl.alternatives("payoff_on_D")[0] + "."]


def test_the_search_sentence_quotes_n_trials_from_the_committed_manifest(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """N on the pages is the trial ledger's N_trials, the simulator trials counted
    (2,435 on D), not stage D's taker count (2,400)."""
    from trading_research.search_engine import ledger as tl

    manifest = tmp_path / tl.MANIFEST_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"n_trials": 2435}', encoding="utf-8")
    monkeypatch.setattr(tl.TrialLedger, "record", lambda _self: {"n_trials": 2435})
    c = pages.Computed(
        SimpleNamespace(keys=(C_KEY, S25_KEY), repo=tmp_path, catalogue=lambda: None)
    )  # type: ignore[arg-type]
    for name in ("c", "e", "board"):
        c.__dict__[name] = None
    c.__dict__["d"], c.__dict__["read"] = record_d(), {}
    (search,) = written(c).plain()["search_on_D"]
    assert search.startswith("Searched 2,435 configurations")


def test_the_sentences_after_h() -> None:
    h = read_h(h_series(), PRESET)
    out = written(computed(d=record_d(), c=record_c(), read={"H": h}))
    edge = h["pages"]["slots"]["S25"]["break_even_per_side"]
    assert out.plain()["S25"] == [hl.s25_sentence(edge, edge, h["statuses"]["S25"])]
    assert f"was {edge:.2f} bp of turnover" in out.plain()["S25"][0]
    (passive,) = out.plain()["passive_and_maker"]
    assert passive.endswith("S-P and S-M not admitted on D (i); the market maker's making part on D was -0.40 bp per fill.")  # fmt: skip
    slots = out.plain()["slot_on_block"]
    assert [s.split(" on H")[0] for s in slots] == ["S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"]
    assert len(out.plain()["regime_map"]) == len(h_series())
    assert any("S-C's regime attribution" in p for p in out.pending)
    assert "break" not in out.plain() and "break: F is not read yet" in out.pending


def test_an_admitted_passive_slot_waits_for_h_then_takes_its_status_and_making_part() -> None:
    admitted = computed(d=record_d(), c=record_c(s_m="admitted"))
    assert any("S-M was admitted on D" in p for p in written(admitted).pending)
    h = read_h(h_series(), {k: v for k, v in PRESET.items() if k != "S-M"})
    out = written(computed(d=record_d(), c=record_c(s_m="admitted"), read={"H": h}))
    making = h["pages"]["slots"]["S-M"]["making_bp_per_fill"]
    (passive,) = out.plain()["passive_and_maker"]
    assert f"S-M {h['statuses']['S-M']}" in passive and f"on H was {making:.2f} bp" in passive


def test_the_regime_axis_sentence_reads_the_attribution_h_recorded() -> None:
    h = read_h(h_series(), PRESET)
    draws = np.linspace(-1.0, 1.0, 101)
    found = Attribution("pull_4h", 2.0, float(np.quantile(draws, 0.95)), draws, 12)
    beside = h.setdefault("measurements", {}).setdefault("S-C", {})
    beside["regime_attribution"] = asdict(found) | {"draws": list(draws)}
    out = written(computed(d=record_d(), read={"H": h}))
    assert out.plain()["regime_axis"] == [hl.regime_axis("S-C", 4, found)]
    assert "did add more than random pauses (100th percentile" in out.plain()["regime_axis"][0]
    row = found.row() | {"draws": list(draws)}  # the attribution's report row, draws beside it
    rebuilt = pages.attribution(row)
    assert rebuilt.reference == found.reference and rebuilt.n_flags == 12


def read_f(r: dict[str, Any], *, taken: list[str], void: bool = False) -> dict[str, Any]:
    """F's results with R as the read records it, and F's usable days."""
    statuses = dict(PRESET) | {"S25": "killed", "S25-Z": "killed"}
    statuses |= dict.fromkeys(taken, "candidate, not confirmed")
    info = {"usable_days": [d.isoformat() for d in F_DAYS], "stale_days": []}
    return {"statuses": statuses, "assessments": {}, "R": r, "R_void": void, "info": info,
            "split": {"b": "2025-08-30", "on_all_of_F": True}, "regime_map": []}  # fmt: skip


def test_the_break_sentence_per_slot_taken_to_f() -> None:
    h = read_h(h_series(), PRESET)
    nothing = written(computed(read={"H": h, "F": read_f({}, taken=[])}))
    assert nothing.plain()["break"] == ["R: not tested."]
    r = {"S-C": {"outcome": "untestable", "reason": "fewer than 20 usable days before b or 10 after"},
         "S-R": {"outcome": "not tested", "reason": "not a candidate on H"}}  # fmt: skip
    out = written(
        computed(
            read={"H": h | {"taken_to_F": ["S-C", "S-R"]}, "F": read_f(r, taken=["S-C", "S-R"])}
        )
    )
    assert out.sentences["break"] == [
        {"about": "S-C", "sentence": "R: untestable (fewer than 20 usable days before b)."},
        {"about": "S-R", "sentence": "R: not tested."},
    ]
    on_f = [s for s in out.plain()["slot_on_block"] if " on F:" in s]
    assert [s.split(" on F")[0] for s in on_f] == ["S-C", "S-R"]
    void = {"S-C": {"outcome": "supported", "p_R": 0.01}}
    out = written(
        computed(
            read={"H": h | {"taken_to_F": ["S-C"]}, "F": read_f(void, taken=["S-C"], void=True)}
        )
    )
    assert out.plain()["break"] == ["R: not supported."]


def test_untestable_reasons_are_written_in_the_registrations_words() -> None:
    f = read_f({}, taken=[]) | {"split": {"b": "2025-09-25"}}
    reasons = hl.UNTESTABLE_REASONS
    assert pages.registered_reason("no alarm in F", f) == reasons[0]
    assert (
        pages.registered_reason("fewer than 20 usable days before b or 10 after", f) == reasons[2]
    )
    early = f | {"split": {"b": "2025-08-25"}}
    assert (
        pages.registered_reason("fewer than 20 usable days before b or 10 after", early)
        == reasons[1]
    )
    rate = "the detector's alarm rate on D above one per 14 days"
    assert pages.registered_reason(rate, f) == reasons[3]
    assert pages.registered_reason("12 days in the reference set besides b", f) == reasons[4]


def test_the_finalists_page_has_h_for_every_slot_and_f_for_those_taken() -> None:
    h = read_h(h_series(), PRESET)
    e = {"mde": {"S25": [1.0, 0.6], "S-C": [2.0, 1.1]}}
    c = computed(
        d=record_d(), e=e, read={"H": h | {"taken_to_F": ["S-C"]}, "F": read_f({}, taken=["S-C"])}
    )
    out = pages.Written()
    table = pages.finalists_page(c, out)
    assert table is not None and out.pending == []
    rows = table.set_index(["block", "slot"])
    assert list(table["block"]) == ["H"] * 6 + ["F"] and list(rows.loc["H"].index) == [
        "S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"]  # fmt: skip
    s25 = h["pages"]["slots"]["S25"]
    assert rows.loc[("H", "S25"), "break_even_per_side"] == pytest.approx(
        s25["break_even_per_side"]
    )
    assert rows.loc[("H", "S25"), "days"] == 24 and rows.loc[("H", "S25"), "mde"] == 0.6
    assert rows.loc[("H", "S25"), "first_kill"] == h["assessments"]["S25"]["first_kill"]
    assert np.isnan(rows.loc[("H", "S-C"), "mean"]) and rows.loc[("H", "S-C"), "trades"] == 0
    assert rows.loc[("F", "S-C"), "status"] == "candidate, not confirmed"
    assert rows.loc[("F", "S-C"), "h_before_0801_mean"] == pytest.approx(
        h["pages"]["slots"]["S-C"]["h_before_0801_mean"])  # fmt: skip


def test_the_ladder_takes_each_step_from_its_block() -> None:
    board = pd.DataFrame({"key": [S25_KEY], "gross": [NAN], "net": [-7.5]})
    h = read_h(h_series(), PRESET)
    h["measurements"] = {"S25": {"all_of_H": {"mean": -6.0}}}
    c = computed(c=record_c(), board=board, read={"H": h})
    by = pages.ladder(c).set_index(["row", "block"])
    assert by.loc[("A0, crossing", "O"), "value"] == -2.0
    assert by.loc[("A0, posted, queue", "O"), "value"] == pytest.approx(-0.2 / 4_000.0 * 1e4)
    assert by.loc[("fill ratio, queue", "O"), "value"] == 0.45
    assert (
        by.loc[("S25, net", "D"), "value"] == -7.5
        and "no value" in by.loc[("S25, gross", "D"), "label"]
    )
    numbers = h["pages"]["slots"]
    window = by.loc[("S25, net, status window", "H before 2025-08-01"), "value"]
    assert window == pytest.approx(numbers["S25"]["mean"])
    assert by.loc[("S25, net", "H"), "value"] == -6.0
    assert "not read yet" in by.loc[("S25, net", "F"), "label"]
    net_zero = by.loc[("S25-Z, net at 0/0", "H before 2025-08-01"), "value"]
    assert net_zero == pytest.approx(numbers["S25-Z"]["net_zero_fee_mean"])


def test_the_pages_file_carries_every_sentence_and_what_is_pending() -> None:
    h = read_h(h_series(), PRESET)
    c = computed(d=record_d(), c=record_c(), read={"H": h})
    out = pages.sentences(c)
    figures = {"edge_ladder": "<picture>ladder</picture>"}
    text = pages.markdown(c, out, pages.finalists_page(c, out), "the S25 page", figures)
    for rows in out.plain().values():
        assert all(sentence in text for sentence in rows)
    assert all(p in text for p in out.pending) and "<picture>ladder</picture>" in text
    assert "## The finalists on H, then F and P" in text and "the S25 page" in text


def test_the_entry_point_has_a_pages_subcommand() -> None:
    from experiments.short_horizon_2025 import parser

    assert parser().parse_args(["pages"]).command == "pages"


def test_fill_on_the_block_pools_the_passive_day_files(tmp_path: Any) -> None:
    rows = []
    for day, excluded in (("2025-07-08", False), ("2025-07-09", False), ("2025-07-10", True)):
        for variant, posted, filled in (("true", 10.0, 6.0), ("fill/touch", 10.0, 9.0),
                                        ("fill/trade_through", 10.0, 2.0), ("flipped", 10.0, 1.0)):  # fmt: skip
            rows.append({"day": day, "run_slot": "S-P", "variant": variant, "excluded": excluded,
                         "orders_posted": posted, "orders_filled": filled, "volume_posted": 2 * posted,
                         "volume_filled": 2 * filled, "notional_filled": 100.0 * filled,
                         "gross_usdt": 1.0, "fees_usdt": -0.5, "net_usdt": 0.5})  # fmt: skip
    (tmp_path / "passive").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "passive" / "days.parquet", index=False)
    kept = [date(2025, 7, 8), date(2025, 7, 9)]
    found = pn.fill_rule_summary(SimpleNamespace(work=tmp_path), kept)
    assert list(found) == ["S-P"]  # S25-Z ran no day
    by = {row["rule"]: row for row in found["S-P"]}
    assert list(by) == ["touch", "trade_through", "queue"]
    assert by["queue"]["days"] == 2 and by["queue"]["orders_posted"] == 20.0
    assert by["queue"]["fill_ratio"] == pytest.approx(0.6) and by["touch"]["fill_ratio"] == 0.9
    assert by["trade_through"]["net_usdt"] == 1.0 and by["touch"]["bias"] == "optimistic bound"
    assert pn.fill_rule_summary(SimpleNamespace(work=tmp_path / "none"), kept) == {}


def test_the_pages_fill_table_is_the_results_files_on_the_kept_days(tmp_path: Any) -> None:
    """A day another run of the read flagged stale is out of the pages' FILL table,
    as it is out of the results file's (``heldout_measures.fill_rules``)."""
    from trading_research.pipeline.short_horizon import heldout_measures as hm

    columns = ("orders_posted", "orders_filled", "volume_posted", "volume_filled",
               "notional_filled", "gross_usdt", "fees_usdt", "net_usdt")  # fmt: skip

    def row(day: str, variant: str, filled: float, excluded: bool = False) -> dict[str, Any]:
        return {"run_slot": "S25-Z", "variant": variant, "day": day, "excluded": excluded,
                "fill_paths": "{}", **dict.fromkeys(columns, 1.0), "orders_posted": 10.0,
                "orders_filled": filled}  # fmt: skip

    folder = tmp_path / "passive"
    folder.mkdir()
    variants = ("true", "fill/touch", "fill/trade_through")
    good = [row("2025-07-10", v, 2.0) for v in variants]
    # 07-11: the flipped placebo walked a stale snapshot, so the day is out of every series
    stale = [row("2025-07-11", v, 9.0) for v in variants] + [
        row("2025-07-11", "flipped", 0.0, excluded=True)
    ]
    pd.DataFrame(good).to_parquet(folder / "2025-07-10.parquet", index=False)
    pd.DataFrame(stale).to_parquet(folder / "2025-07-11.parquet", index=False)
    kept = [date(2025, 7, 10)]  # compute_block's kept days
    results = hm.fill_rules(pd.DataFrame(good + stale), "S25-Z", kept)["summary"]
    pages = pn.fill_rule_summary(SimpleNamespace(work=tmp_path), kept)["S25-Z"]
    queue = next(r for r in pages if r["rule"] == "queue")
    assert queue["days"] == 1 and queue["fill_ratio"] == 0.2
    assert pages == results


def test_the_rows_for_the_register_and_the_readme_summary() -> None:
    from trading_research.pipeline.short_horizon.page_figures import FROM_PAGES
    from trading_research.reporting.search_figures import picture

    h = read_h(h_series(), PRESET)
    c = computed(d=record_d(), c=record_c(), read={"H": h})
    out = pages.sentences(c)
    rows = pages.register_rows(c, out)
    assert rows.startswith(pages.REGISTER_HEADING) and "**Status: read through H**" in rows
    slots = [line.split(" on H")[0][2:] for line in rows.splitlines() if line.startswith("- ")]
    assert slots == ["S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"]
    assert "pre-registered, no block read" in pages.register_rows(computed(), pages.Written())
    figures = {
        "edge_ladder": picture("short_horizon_2025_edge_ladder", "ladder", images=FROM_PAGES)
    }
    summary = pages.readme_summary(out, figures)
    assert summary.startswith(out.plain()["search_on_D"][0]) and out.plain()["S25"][0] in summary
    assert not any(s in summary for s in out.plain()["slot_on_block"])
    assert 'src="docs/images/short_horizon_2025_edge_ladder_light.png"' in summary
    assert FROM_PAGES not in summary
    text = pages.markdown(c, out, None, None, figures)
    assert "## Rows for docs/findings.md" in text and "## The README summary" in text


def test_the_notes_beside_the_payoff_and_passive_sentences() -> None:
    """Beside "No configuration paid ...": no admissible cell had a positive net, and
    the cells that had one; beside the passive sentence, S-M's (i) when it failed on
    the making part alone while S was positive. The registered text is unchanged."""
    board = pd.DataFrame({
        "key": ["BTCUSDT|R|tcn|a", "BTCUSDT|R|tcn|b", "BTCUSDT|C|gbm|c"],
        "net": [83.2, 73.6, -5.1], "admissible": [False, False, True],
        "trades_per_day": [0.02, 0.02, 7.4],
    })  # fmt: skip
    note = pages.payoff_note(board)
    assert note is not None and note.startswith("no admissible cell had a positive net")
    assert "2 at +73.6 to +83.2 bp on 3 trades" in note and "stratum R" in note
    board.loc[2, "net"] = 1.0
    assert pages.payoff_note(board) is None
    making = pages.making_note(
        {"making_bp_per_fill": -0.4299, "S": 0.000525, "status_on_D": "not admitted on D (i, ii)"}
    )
    assert making is not None
    assert "(i: making part \u22120.43 bp per fill; S = +0.0005)" in making
    assert making.startswith("S-M, not admitted on D (i, ii) on D's own conditions")
    assert pages.making_note({"making_bp_per_fill": 0.1, "S": 0.0005}) is None
    out = pages.Written()
    out.sentences["payoff_on_D"] = [{"about": "D", "sentence": "x."}]
    out.note("payoff_on_D", "no admissible cell had a positive net.")
    assert pages.readme_summary(out, {}) == "x. (no admissible cell had a positive net.)"
