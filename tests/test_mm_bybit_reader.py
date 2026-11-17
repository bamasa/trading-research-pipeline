"""The Bybit 2021 reader, the archive's fetch, funding, the blocks and the fee table."""

from __future__ import annotations

import gzip
from datetime import date

import numpy as np
import pandas as pd
import pytest

from _mm_bybit_days import DAY
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.mm_bybit import access as acc
from trading_research.mm_bybit import archive, config, days, funding, synthetic

HEADER = "timestamp,symbol,side,size,price,tickDirection,trdMatchID\n"
GAP = float(config.load()["settings"]["print_gap_exclusion_s"])


def csv(rows: list[str]) -> bytes:
    return (HEADER + "".join(r + "\n" for r in rows)).encode()


def test_stamps_are_converted_from_their_text_exactly() -> None:
    text = pd.Series(["1609459200.1234", "1609459200", "1609459200.123456789", "1609459201.5"])
    assert archive.stamps_ns(text).tolist() == [
        1609459200_123400000,
        1609459200_000000000,
        1609459200_123456789,
        1609459201_500000000,
    ]


def test_a_newest_first_file_is_reversed_and_ties_keep_the_venues_order() -> None:
    start = day_start_ns(DAY) // NS_PER_S
    oldest_first = [
        f"{start + 10}.5,BTCUSDT,Buy,0.001,100.5,PlusTick,a",
        f"{start + 10}.5,BTCUSDT,Buy,0.002,101.0,PlusTick,b",
        f"{start + 20},BTCUSDT,Sell,0.003,100.0,MinusTick,c",
    ]
    ascending, found_a = archive.parse(csv(oldest_first), DAY)
    descending, found_d = archive.parse(csv(oldest_first[::-1]), DAY)
    assert found_a["direction"] == "ascending" and found_d["direction"] == "descending"
    assert found_a["stamps_ordered"] and found_d["stamps_ordered"]
    for frame in (ascending, descending):
        assert frame["price"].tolist() == [100.5, 101.0, 100.0]
        assert frame["aggressor"].tolist() == [1, 1, -1]
        assert frame["trade_id"].tolist() == [0, 1, 2]
        assert frame["source"].iloc[0] == archive.SOURCE
    assert found_a["stamp_decimals"] == 1
    assert found_a["sides"] == {"buy": 2, "sell": 1}


def test_rows_outside_the_day_are_dropped_and_counted_and_duplicates_recorded() -> None:
    start = day_start_ns(DAY) // NS_PER_S
    rows = [
        f"{start - 1}.9,BTCUSDT,Buy,0.001,100.5,PlusTick,a",
        f"{start + 5},BTCUSDT,Sell,0.001,100.0,MinusTick,b",
        f"{start + 6},BTCUSDT,Sell,0.001,100.0,MinusTick,b",
        f"{start + 86_400},BTCUSDT,Buy,0.001,100.5,PlusTick,c",
    ]
    frame, found = archive.parse(csv(rows), DAY)
    assert len(frame) == 2
    assert found["rows"] == 4 and found["rows_outside_day"] == 2
    assert found["duplicate_match_ids"] == 1 and found["ids_unique"] is False


def test_a_file_that_does_not_parse_is_unreadable() -> None:
    start = day_start_ns(DAY) // NS_PER_S
    with pytest.raises(archive.UnreadableFile):
        archive.parse(csv([f"{start + 5},BTCUSDT,Hold,0.001,100.0,MinusTick,a"]), DAY)
    with pytest.raises(archive.UnreadableFile):
        archive.parse(csv([f"{start + 5},BTCUSDT,Buy,-0.001,100.0,MinusTick,a"]), DAY)


def test_a_fetch_writes_the_contract_and_keeps_no_raw_file(tmp_path) -> None:
    prints = synthetic.random_day(DAY, seed=3, n_prints=2_000)
    fake = synthetic.FakeArchive({DAY: synthetic.csv_gz(prints, descending=True)})
    found = archive.fetch_day(DAY, tmp_path, acc.development_access(), opener=fake)
    stored = pd.read_parquet(archive.prints_path(tmp_path, DAY))
    assert found["direction"] == "descending" and found["rows"] == 2_000
    assert np.allclose(stored["price"], prints["price"]) and np.allclose(
        stored["size"], prints["size"]
    )
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [f"{DAY}.parquet"]
    assert fake.requests == [archive.url(DAY)]


def test_a_404_is_the_only_missing_file_and_an_unfetched_day_is_an_error(tmp_path) -> None:
    fake = synthetic.FakeArchive({})
    with pytest.raises(archive.MissingFile):
        archive.fetch_day(DAY, tmp_path, acc.development_access(), opener=fake)
    archive.write_status(tmp_path, DAY, "no_prints_file", "HTTP 404")
    cfg = config.load()
    settlements = synthetic.funding(DAY)
    fees = config.fee_on(cfg, DAY)
    assert (
        days.load(DAY, tmp_path, acc.development_access(), settlements, fees, print_gap_s=GAP)
        == "no_prints_file"
    )
    other = date(2021, 1, 5)
    with pytest.raises(FileNotFoundError):
        days.load(other, tmp_path, acc.development_access(), settlements, fees, print_gap_s=GAP)


def test_a_day_is_excluded_for_a_print_gap_or_missing_funding(tmp_path) -> None:
    cfg, access = config.load(), acc.development_access()
    prints = synthetic.random_day(DAY, seed=4, n_prints=20_000)
    keep = (prints["timestamp"] < pd.Timestamp(f"{DAY} 12:00", tz="UTC")) | (
        prints["timestamp"] > pd.Timestamp(f"{DAY} 12:11", tz="UTC")
    )
    target = archive.prints_path(tmp_path, DAY)
    target.parent.mkdir(parents=True)
    prints[keep].reset_index(drop=True).to_parquet(target)
    settlements = synthetic.funding(DAY)
    fees = config.fee_on(cfg, DAY)
    assert days.load(DAY, tmp_path, access, settlements, fees, print_gap_s=GAP) == "print_gap"
    prints.to_parquet(target)
    assert not isinstance(days.load(DAY, tmp_path, access, settlements, fees, print_gap_s=GAP), str)
    assert (
        days.load(DAY, tmp_path, access, settlements.iloc[:2], fees, print_gap_s=GAP)
        == "no_funding"
    )


def test_the_development_access_refuses_e_h_and_every_day_outside_w_and_d(tmp_path) -> None:
    access = acc.development_access()
    for day in (date(2020, 12, 15), date(2020, 12, 31), date(2021, 1, 1), date(2021, 3, 31)):
        access.require(day)
    for day in (date(2020, 12, 14), date(2021, 4, 1), date(2021, 4, 8), date(2021, 6, 30),
                date(2021, 7, 1), date(2022, 1, 1)):  # fmt: skip
        with pytest.raises(acc.AccessRefused):
            access.require(day)
    fake = synthetic.FakeArchive({})
    with pytest.raises(acc.AccessRefused):
        archive.fetch_day(date(2021, 4, 8), tmp_path, access, opener=fake)
    assert fake.requests == []  # refused before any request


def test_the_registered_blocks_match_the_configuration() -> None:
    cfg = config.load()
    for name, (first, last) in acc.BLOCKS.items():
        assert cfg["blocks"][name] == [first, last]
    assert list(acc.READER_TEST_DAYS) == cfg["blocks"]["reader_test"]
    assert list(acc.STRESS_DAYS) == cfg["blocks"]["stress_days"]
    assert [len(acc.block_days(b)) for b in ("W", "D", "E", "H")] == [17, 90, 7, 84]


def test_funding_settlements_must_sit_on_the_eight_hour_grid() -> None:
    good = synthetic.funding(DAY)
    funding.check_grid(good)
    stamps, rates = funding.day_settlements(good, DAY)
    assert len(stamps) == 3 and rates.tolist() == [1e-4, 1e-4, 1e-4]
    off = good.assign(timestamp=good["timestamp"] + pd.to_timedelta([0, 1, 0], unit="h"))
    with pytest.raises(funding.FundingOffGrid):
        funding.check_grid(off)
    assert funding.day_settlements(good.iloc[[0, 2]], DAY) is None


def test_the_block_funding_is_fetched_from_the_history_and_written(tmp_path) -> None:
    w = acc.block_days("W")
    settlements = pd.concat([synthetic.funding(d) for d in w], ignore_index=True)
    fake = synthetic.FakeArchive({}, settlements)
    frame = funding.fetch_block("W", tmp_path, acc.development_access(), opener=fake)
    assert len(frame) == 3 * len(w)
    assert all("category=linear" in u and "symbol=BTCUSDT" in u for u in fake.requests)
    with pytest.raises(acc.AccessRefused):
        funding.fetch_block("H", tmp_path, acc.development_access(), opener=fake)


def test_the_configuration_hashes_to_the_registered_null_placeholder_sha256() -> None:
    document = config.DOCUMENT_PATH.read_text()
    # The registration's hash, superseded by the stage-0 amendment's.
    assert "Null-placeholder sha256: `0ece0d7d" in document
    assert config.null_placeholder_sha256(document).startswith("17fca5f8")
    config.check_registered(config.CONFIG_PATH.read_text(), document)
    changed = config.CONFIG_PATH.read_text().replace("clip_btc: 0.010", "clip_btc: 0.020")
    with pytest.raises(config.RegistrationError):
        config.check_registered(changed, document)


def test_freezing_writes_the_placeholders_and_nothing_else() -> None:
    text = config.unfrozen_text(config.CONFIG_PATH.read_text())
    names = [path[-1] for path in config.placeholder_lines(text)]
    values = {name: {"value": k} for k, name in enumerate(names)}
    frozen = config.freeze_text(text, values)
    assert config.is_frozen(frozen) and not config.is_frozen(text)
    config.check_registered(frozen, config.DOCUMENT_PATH.read_text())
    with pytest.raises(config.RegistrationError):
        config.freeze_text(text, dict(list(values.items())[:-1]))
    with pytest.raises(config.RegistrationError):
        config.freeze_text(frozen, values)


def test_the_fee_comes_from_the_sourced_table_only() -> None:
    cfg = config.load()
    for day in (date(2020, 3, 22), date(2021, 1, 1), date(2021, 5, 19), date(2022, 3, 10)):
        assert config.fee_on(cfg, day) == config.Fees(-2.5, 7.5)
    for day in (date(2020, 3, 21), date(2022, 3, 11)):
        with pytest.raises(config.FeeNotSourced):
            config.fee_on(cfg, day)
    built = days.settings_for(cfg, DAY, "latency150")
    assert (built.maker_bp, built.taker_bp, built.latency_ns) == (-2.5, 7.5, 150_000_000)
    assert (built.stop_s, built.flatten_s, built.warmup_s) == (86_280, 86_340, 600)
    with pytest.raises(ValueError, match="unknown variant"):
        days.settings_for(cfg, DAY, "cheaper")
    with pytest.raises(config.FeeNotSourced):
        days.settings_for(cfg, date(2022, 3, 11), "primary")


def test_the_synthetic_files_are_gzip_csv_in_the_venues_columns() -> None:
    raw = gzip.decompress(synthetic.csv_gz(synthetic.random_day(DAY, n_prints=10)))
    assert raw.decode().splitlines()[0].startswith("timestamp,symbol,side,size,price")


def test_the_codes_constants_are_the_registered_values() -> None:
    from trading_research.mm_bybit import development, engine, metrics, selection, statuses

    cfg = config.load()
    s, h = cfg["settings"], cfg["heldout"]
    assert GAP == 600.0
    assert tuple(s["markout_horizons_s"]) == metrics.HORIZONS_S
    assert s["making_horizon_s"] == metrics.MAKING_HORIZON_S
    assert s["warmup_s"] == metrics.QUOTE_FROM_S
    defaults = engine.RunSettings(-2.5, 7.5)
    assert defaults.clip == s["clip_btc"] and defaults.soft_limit_clips == s["soft_limit_clips"]
    assert defaults.latency_ns == s["latency_ms"] * 1_000_000 and defaults.pause_s == s["pause_s"]
    assert days.VARIANTS["latency150"].latency_ms == s["latency_kill_ms"]
    assert days.VARIANTS["latency10"].latency_ms == s["latency_measured_ms"]
    assert cfg["selection"]["min_fills_per_day"] == selection.MIN_FILLS_PER_DAY
    assert h["placebo_draws"] == statuses.PLACEBO_DRAWS
    assert h["mde_z"] == development.MDE_Z and len(acc.block_days("H")) == development.H_DAYS
    assert cfg["fees"]["repricing"] == {"ex_rebate_maker_bp": metrics.REPRICED_BP["ex_rebate_y"],
        "negative_control_maker_bp": metrics.REPRICED_BP["negative_control_y"],
        "positive_control_maker_bp": metrics.REPRICED_BP["positive_control_y"]}  # fmt: skip
